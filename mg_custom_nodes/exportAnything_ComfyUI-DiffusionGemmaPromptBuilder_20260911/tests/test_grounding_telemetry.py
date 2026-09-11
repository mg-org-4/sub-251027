from __future__ import annotations

import gc
import importlib.util
import inspect
import json
import math
from pathlib import Path
import unittest
import weakref


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diffusiongemma_grounding_telemetry_tests",
    ROOT / "grounding_telemetry.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load grounding_telemetry.py")
telemetry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(telemetry)


class FakeInputIds:
    def __init__(self, length: int, batch_size: int = 1) -> None:
        self.shape = (batch_size, length)


class FakeScores:
    """Small fake matching the collector's dependency-free testing seam."""

    def __init__(
        self,
        *,
        batch_size: int = 1,
        canvas_length: int = 4,
        rows: list[list[float]] | None = None,
    ) -> None:
        base_rows = rows or [
            [3.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-2.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ]
        required = batch_size * canvas_length
        self.rows = [list(base_rows[index % len(base_rows)]) for index in range(required)]
        self.shape = (batch_size, canvas_length, len(self.rows[0]))

    def telemetry_rows(self, indices):
        return [self.rows[int(index)] for index in indices]


class BrokenFakeScores(FakeScores):
    def telemetry_rows(self, _indices):
        raise RuntimeError("injected telemetry failure")


class FakeTokens:
    def __init__(self, batches: list[list[int]]) -> None:
        self._batches = [list(batch) for batch in batches]
        self.shape = (
            len(self._batches),
            len(self._batches[0]) if self._batches else 0,
        )

    def tolist(self):
        return [list(batch) for batch in self._batches]


WORDS = {
    1: "I",
    2: "cannot",
    3: "help",
    4: "with",
    5: "that",
    6: "sorry",
    7: "guidelines",
    8: "The",
    9: "image",
    10: "contains",
    11: "a",
    12: "cat",
}


def decode_words(token_ids, **_kwargs) -> str:
    return " ".join(WORDS.get(int(token), f"token-{int(token)}") for token in token_ids)


class GroundingTelemetryPureTests(unittest.TestCase):
    def test_selected_steps_scale_and_deduplicate_short_schedules(self) -> None:
        self.assertEqual(
            telemetry.scaled_selected_steps(48),
            (48, 32, 16, 8, 4, 1),
        )
        self.assertEqual(telemetry.scaled_selected_steps(4), (4, 3, 1))
        self.assertEqual(telemetry.scaled_selected_steps(1), (1,))
        with self.assertRaises(ValueError):
            telemetry.scaled_selected_steps(0)

    def test_deterministic_positions_are_bounded_and_even(self) -> None:
        positions = telemetry.deterministic_positions(256, limit=999)
        self.assertEqual(len(positions), 16)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 255)
        self.assertEqual(positions, telemetry.deterministic_positions(256, limit=16))
        self.assertEqual(telemetry.deterministic_positions(3, limit=16), (0, 1, 2))

    def test_refusal_detection_requires_complete_clause(self) -> None:
        self.assertEqual(
            telemetry.find_complete_refusal_clauses(
                "Sorry, this scene contains harmful material and guidelines."
            ),
            [],
        )
        matches = telemetry.find_complete_refusal_clauses(
            "I cannot help with that, but I can offer something else."
        )
        self.assertEqual([match["clause_id"] for match in matches], ["cannot_help_request"])
        curly = telemetry.find_complete_refusal_clauses(
            "I’m sorry, but I can’t describe this image."
        )
        self.assertEqual([match["clause_id"] for match in curly], ["sorry_but_cannot"])


class GroundingTelemetryCollectorTests(unittest.TestCase):
    def _collector(self, **kwargs):
        return telemetry.GroundingTelemetryCollector(
            initial_input_length=100,
            max_denoising_steps=48,
            decoder=decode_words,
            **kwargs,
        )

    def test_processor_is_identity_and_records_compact_statistics(self) -> None:
        collector = self._collector(position_limit=16)
        processor = telemetry.DiffusionGemmaTelemetryLogitsProcessor(collector)
        scores = FakeScores()

        returned = processor(FakeInputIds(100), scores, cur_step=48)

        self.assertIs(returned, scores)
        summary = collector.summary()
        self.assertEqual(summary["forward_count"], 1)
        self.assertEqual(summary["canvas_forward_counts"], {"1": 1})

        streamer = telemetry.DiffusionGemmaDraftStreamer(collector)
        self.assertFalse(hasattr(streamer, "_takes_logits"))
        streamer.put(FakeTokens([[99] * 100]))  # Initial prompt is ignored.
        streamer.put_draft(FakeTokens([[8, 9, 10, 11, 12]]))
        streamer.put(FakeTokens([[8, 9, 10, 11, 12]]))
        streamer.end()

        snapshot = collector.summary()["snapshots"][0]
        self.assertTrue(snapshot["committed"])
        stats = snapshot["logits_summary"]
        self.assertLessEqual(stats["sampled_position_count"], 16)
        self.assertEqual(stats["positions"][0]["top1_token_id"], 0)
        self.assertEqual(stats["positions"][0]["top2_token_id"], 1)
        self.assertAlmostEqual(stats["positions"][0]["top1_top2_margin"], 1.0)
        self.assertAlmostEqual(
            stats["positions"][1]["entropy"],
            math.log(3.0),
            places=7,
        )
        self.assertEqual(snapshot["draft_token_ids"], [[8, 9, 10, 11, 12]])
        self.assertEqual(snapshot["draft_text"], ["The image contains a cat"])

    def test_processor_signature_matches_diffusiongemma_kwargs_contract(self) -> None:
        collector = self._collector()
        processor = telemetry.DiffusionGemmaTelemetryLogitsProcessor(collector)
        self.assertEqual(
            list(inspect.signature(processor.__call__).parameters),
            ["input_ids", "scores", "cur_step"],
        )

    def test_canvas_number_comes_from_initial_length_and_canvas_size(self) -> None:
        collector = self._collector()
        processor = telemetry.DiffusionGemmaTelemetryLogitsProcessor(collector)
        streamer = telemetry.DiffusionGemmaDraftStreamer(collector)
        streamer.put(FakeTokens([[99] * 100]))

        processor(FakeInputIds(100 + 256), FakeScores(), cur_step=32)
        streamer.put_draft(FakeTokens([[8, 9, 10, 11, 12]]))

        snapshot = collector.summary()["snapshots"][0]
        self.assertEqual(snapshot["canvas"], 2)
        self.assertEqual(snapshot["step"], 32)

    def test_late_refusal_requires_two_of_last_three_snapshots(self) -> None:
        collector = self._collector()
        processor = telemetry.DiffusionGemmaTelemetryLogitsProcessor(collector)
        streamer = telemetry.DiffusionGemmaDraftStreamer(collector)
        streamer.put(FakeTokens([[99] * 100]))

        phrase = FakeTokens([[1, 2, 3, 4, 5]])
        clean = FakeTokens([[8, 9, 10, 11, 12]])
        for step, draft in ((8, phrase), (4, clean), (1, phrase)):
            processor(FakeInputIds(100), FakeScores(), cur_step=step)
            streamer.put_draft(draft)

        refusal = collector.summary()["refusal"]
        self.assertTrue(refusal["persistent_late_match"])
        self.assertTrue(refusal["detected"])
        self.assertEqual(refusal["late_snapshot_match_count"], 2)

    def test_draft_stability_compares_consecutive_selected_drafts(self) -> None:
        collector = self._collector()
        processor = telemetry.DiffusionGemmaTelemetryLogitsProcessor(collector)
        streamer = telemetry.DiffusionGemmaDraftStreamer(collector)
        streamer.put(FakeTokens([[99] * 100]))

        for step, tokens in (
            (48, [1, 2, 3, 4]),
            (32, [1, 2, 9, 4]),
            (16, [1, 2, 9, 4]),
        ):
            processor(FakeInputIds(100), FakeScores(), cur_step=step)
            streamer.put_draft(FakeTokens([tokens]))

        stability = collector.summary()["draft_stability"]
        self.assertEqual(stability["comparison_count"], 2)
        self.assertEqual(stability["canvas_count"], 1)
        canvas = stability["per_canvas"][0]
        self.assertEqual(canvas["canvas"], 1)
        self.assertEqual(canvas["matching_tokens"], 7)
        self.assertEqual(canvas["compared_tokens"], 8)
        self.assertEqual(canvas["mean_agreement"], 0.875)
        self.assertEqual(canvas["minimum_agreement"], 0.75)
        self.assertEqual(canvas["final_agreement"], 1.0)
        self.assertEqual(
            [(item["from_step"], item["to_step"]) for item in canvas["comparisons"]],
            [(48, 32), (32, 16)],
        )

    def test_draft_stability_isolated_by_canvas_and_penalizes_length_changes(self) -> None:
        collector = self._collector()
        processor = telemetry.DiffusionGemmaTelemetryLogitsProcessor(collector)
        streamer = telemetry.DiffusionGemmaDraftStreamer(collector)
        streamer.put(FakeTokens([[99] * 100]))

        for input_length, step, tokens in (
            (100, 48, [1, 2, 3]),
            (100, 32, [1, 2]),
            (356, 48, [8, 9]),
            (356, 32, [8, 10]),
        ):
            processor(FakeInputIds(input_length), FakeScores(), cur_step=step)
            streamer.put_draft(FakeTokens([tokens]))

        stability = collector.summary()["draft_stability"]
        self.assertEqual(stability["comparison_count"], 2)
        self.assertEqual([item["canvas"] for item in stability["per_canvas"]], [1, 2])
        first, second = stability["per_canvas"]
        self.assertEqual(first["matching_tokens"], 2)
        self.assertEqual(first["compared_tokens"], 3)
        self.assertAlmostEqual(first["mean_agreement"], 2 / 3, places=8)
        self.assertEqual(second["mean_agreement"], 0.5)

    def test_isolated_words_and_one_late_clause_do_not_trigger(self) -> None:
        collector = self._collector()
        processor = telemetry.DiffusionGemmaTelemetryLogitsProcessor(collector)
        streamer = telemetry.DiffusionGemmaDraftStreamer(collector)
        streamer.put(FakeTokens([[99] * 100]))

        isolated = FakeTokens([[2, 6, 7]])
        phrase = FakeTokens([[1, 2, 3, 4, 5]])
        for step, draft in ((8, isolated), (4, phrase), (1, isolated)):
            processor(FakeInputIds(100), FakeScores(), cur_step=step)
            streamer.put_draft(draft)

        refusal = collector.summary()["refusal"]
        self.assertFalse(refusal["persistent_late_match"])
        self.assertFalse(refusal["detected"])

    def test_complete_clause_in_committed_output_triggers(self) -> None:
        collector = self._collector()
        processor = telemetry.DiffusionGemmaTelemetryLogitsProcessor(collector)
        streamer = telemetry.DiffusionGemmaDraftStreamer(collector)
        streamer.put(FakeTokens([[99] * 100]))
        processor(FakeInputIds(100), FakeScores(), cur_step=1)
        streamer.put_draft(FakeTokens([[8, 9, 10, 11, 12]]))
        streamer.put(FakeTokens([[1, 2, 3, 4, 5]]))

        summary = collector.summary()
        self.assertTrue(summary["refusal"]["final_output_match"])
        self.assertTrue(summary["refusal"]["detected"])
        self.assertEqual(summary["final_output"]["text"], ["I cannot help with that"])

    def test_telemetry_failure_is_fail_open_and_does_not_retain_tensor(self) -> None:
        collector = self._collector()
        processor = telemetry.DiffusionGemmaTelemetryLogitsProcessor(collector)
        scores = BrokenFakeScores()
        reference = weakref.ref(scores)

        returned = processor(FakeInputIds(100), scores, cur_step=48)
        self.assertIs(returned, scores)
        self.assertTrue(collector.has_errors)
        del returned
        del scores
        gc.collect()
        self.assertIsNone(reference())

        serialized = collector.summary_json()
        self.assertIn("injected telemetry failure", serialized)

    def test_cuda_step_is_derived_without_item_synchronization(self) -> None:
        class CudaStep:
            is_cuda = True

            def item(self):
                raise AssertionError("CUDA cur_step.item() must not be called")

        collector = self._collector()
        processor = telemetry.DiffusionGemmaTelemetryLogitsProcessor(collector)
        streamer = telemetry.DiffusionGemmaDraftStreamer(collector)
        streamer.put(FakeTokens([[99] * 100]))
        for _ in range(17):
            processor(FakeInputIds(100), FakeScores(), cur_step=CudaStep())
            streamer.put_draft(FakeTokens([[1, 2, 3]]))

        summary = collector.summary()
        self.assertEqual(summary["forward_count"], 17)
        self.assertEqual(
            [snapshot["step"] for snapshot in summary["snapshots"]],
            [48, 32],
        )
        self.assertEqual(summary["errors"], [])

    def test_summary_is_capped_and_preserves_first_and_final_snapshots(self) -> None:
        collector = self._collector(max_summary_bytes=4096)
        processor = telemetry.DiffusionGemmaTelemetryLogitsProcessor(collector)
        streamer = telemetry.DiffusionGemmaDraftStreamer(collector)
        streamer.put(FakeTokens([[99] * 100]))

        for canvas in range(5):
            input_length = 100 + (canvas * 256)
            for step in (48, 32, 16, 8, 4, 1):
                processor(FakeInputIds(input_length), FakeScores(), cur_step=step)
                streamer.put_draft(FakeTokens([[8, 9, 10, 11, 12] * 50]))
            streamer.put(FakeTokens([[8, 9, 10, 11, 12] * 50]))

        summary = collector.summary()
        encoded = json.dumps(
            summary,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertLessEqual(len(encoded), 4096)
        self.assertTrue(summary["truncated"])
        self.assertEqual(summary["snapshots"][0]["ordinal"], 1)
        self.assertEqual(summary["snapshots"][-1]["ordinal"], 30)
        self.assertIn("draft_stability", summary)
        self.assertGreater(summary["draft_stability"]["comparison_count"], 0)
        self.assertIn("truncation", summary)

    def test_factory_returns_one_shared_per_call_collector(self) -> None:
        collector, processor, streamer = telemetry.build_diffusiongemma_telemetry(
            initial_input_length=7,
            decoder=decode_words,
        )
        self.assertIs(processor.collector, collector)
        self.assertIs(streamer.collector, collector)
        self.assertIs(
            telemetry.PassiveTelemetryLogitsProcessor,
            telemetry.DiffusionGemmaTelemetryLogitsProcessor,
        )


if __name__ == "__main__":
    unittest.main()
