from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]


def load_timed_module():
    module_name = f"diffusiongemma_timed_lyrics_fixture_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "timed_lyrics_nodes.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load timed lyrics nodes")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_package():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    package_name = f"diffusiongemma_timed_package_fixture_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        package_name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load DiffusionGemma Prompt Builder")
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    return package


def audio_fixture(
    *,
    duration: float = 12.0,
    sample_rate: int = 1_000,
    vocal_sections: tuple[tuple[float, float], ...] = ((2.0, 5.0),),
):
    samples = int(round(duration * sample_rate))
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    accompaniment = (
        0.06 * torch.sin(2.0 * torch.pi * 73.0 * time)
        + 0.025 * torch.sin(2.0 * torch.pi * 117.0 * time)
    )
    vocals = torch.zeros_like(time)
    for start, end in vocal_sections:
        mask = (time >= start) & (time < end)
        # A deterministic voice-like signal which is independent of the mix bed.
        phrase = (
            0.15 * torch.sin(2.0 * torch.pi * 181.0 * time)
            + 0.05 * torch.sin(2.0 * torch.pi * 263.0 * time)
        )
        vocals = torch.where(mask, phrase, vocals)
    final = {
        "waveform": (accompaniment + vocals).reshape(1, 1, -1).contiguous(),
        "sample_rate": sample_rate,
    }
    stem = {
        "waveform": vocals.reshape(1, 1, -1).contiguous(),
        "sample_rate": sample_rate,
    }
    return final, stem


class FakeTranscriber:
    def __init__(self, responses=None, exception: Exception | None = None):
        self.responses = list(responses or [])
        self.exception = exception
        self.calls: list[dict[str, float | int | str]] = []

    def __call__(
        self,
        audio,
        language,
        return_timestamps,
        chunk_offset_seconds,
        absolute_song_start_seconds,
    ):
        self.calls.append(
            {
                "duration": audio["waveform"].shape[-1] / audio["sample_rate"],
                "language": language,
                "return_timestamps": int(bool(return_timestamps)),
                "chunk_offset_seconds": chunk_offset_seconds,
                "absolute_song_start_seconds": absolute_song_start_seconds,
            }
        )
        if self.exception is not None:
            raise self.exception
        if self.responses:
            return self.responses.pop(0)
        return {
            "text": "we light the night",
            "chunks": [
                {"text": "we light the night", "timestamp": [0.25, 1.75]},
            ],
        }


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(_name_or_path="fake/whisper-tiny")
        self.moves: list[str] = []

    def to(self, device):
        self.moves.append(str(device))
        return self


def fake_pipeline(transcriber: FakeTranscriber, *, with_model: bool = False):
    result = {"model_id": "fake/whisper-tiny", "transcribe": transcriber}
    if with_model:
        result["model"] = FakeModel()
    return result


class TimedLyricsAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.timed = load_timed_module()

    def setUp(self) -> None:
        self.timed._ANALYSIS_CACHE.clear()

    def run_lyrics(
        self,
        final,
        stem,
        pipeline,
        *,
        lyrics="[Verse 1]\nWe light the night\nAnd hold the sky",
        song_duration=None,
        excerpt_start=0.0,
        excerpt_duration=None,
        confidence=0.55,
    ):
        duration = final["waveform"].shape[-1] / final["sample_rate"]
        return self.timed.DiffusionGemmaTimedLyricsAnalyzer().analyze(
            final,
            "Lyrics + lip sync",
            lyrics,
            self.timed._waveform_sha256(final),
            duration if song_duration is None else song_duration,
            excerpt_start,
            duration if excerpt_duration is None else excerpt_duration,
            "en",
            confidence,
            stem,
            pipeline,
        )

    def test_input_schema_and_lazy_surface(self) -> None:
        node = self.timed.DiffusionGemmaTimedLyricsAnalyzer
        inputs = node.INPUT_TYPES()
        required = inputs["required"]
        optional = inputs["optional"]
        self.assertEqual(required["performance_mode"][0], "STRING")
        self.assertTrue(required["performance_mode"][1]["forceInput"])
        self.assertTrue(required["lyrics"][1]["forceInput"])
        self.assertTrue(required["lyrics"][1]["multiline"])
        self.assertEqual(required["language"][1]["default"], "en")
        self.assertAlmostEqual(required["minimum_alignment_confidence"][1]["default"], 0.55)
        self.assertTrue(optional["vocal_stem"][1]["lazy"])
        self.assertTrue(optional["whisper_pipeline"][1]["lazy"])
        self.assertEqual(
            node.check_lazy_status(
                performance_mode="Lyrics + lip sync",
                vocal_stem=None,
                whisper_pipeline=None,
            ),
            ["vocal_stem", "whisper_pipeline"],
        )
        self.assertEqual(
            node.check_lazy_status(
                performance_mode="Lyrics + lip sync",
                vocal_stem=object(),
                whisper_pipeline=None,
            ),
            ["whisper_pipeline"],
        )
        self.assertEqual(
            node.check_lazy_status(
                performance_mode="Natural / audio-led sync",
                vocal_stem=None,
                whisper_pipeline=None,
            ),
            [],
        )

    def test_natural_and_dance_never_evaluate_expensive_lazy_inputs(self) -> None:
        final, _stem = audio_fixture()

        class ExplodesOnAccess:
            def __getattribute__(self, _name):
                raise AssertionError("lazy input was evaluated")

        node = self.timed.DiffusionGemmaTimedLyricsAnalyzer()
        for mode in ("Natural / audio-led sync", "Dance / music sync"):
            outputs = node.analyze(
                final,
                mode,
                "We light the night",
                "not-needed",
                12.0,
                0.0,
                12.0,
                "en",
                0.55,
                ExplodesOnAccess(),
                ExplodesOnAccess(),
            )
            report = json.loads(outputs[0])
            self.assertEqual(report["analysis_status"], "not_required")
            self.assertFalse(report["timing_ready"])
            self.assertTrue(outputs[2])
            self.assertEqual(report["events"], [])

    def test_exact_full_mix_separator_fallback_is_rejected_safely(self) -> None:
        final, _stem = audio_fixture()
        transcriber = FakeTranscriber()
        outputs = self.run_lyrics(
            final,
            {"waveform": final["waveform"].clone(), "sample_rate": final["sample_rate"]},
            fake_pipeline(transcriber),
        )
        report = json.loads(outputs[0])
        self.assertEqual(report["analysis_status"], "natural_fallback")
        self.assertFalse(report["timing_ready"])
        self.assertTrue(outputs[2])
        self.assertIn("vocal_stem_matches_full_mix_separator_fallback", report["warnings"])
        self.assertEqual(transcriber.calls, [])

    def test_near_identical_gain_shifted_full_mix_fallback_is_rejected(self) -> None:
        final, _stem = audio_fixture()
        generator = torch.Generator().manual_seed(91)
        nearly_same = 1.013 * final["waveform"] + 1.0e-5 * torch.randn(
            final["waveform"].shape, generator=generator
        )
        transcriber = FakeTranscriber()
        outputs = self.run_lyrics(
            final,
            {"waveform": nearly_same, "sample_rate": final["sample_rate"]},
            fake_pipeline(transcriber),
        )
        report = json.loads(outputs[0])
        comparison = report["diagnostics"]["stem_full_mix_comparison"]
        self.assertTrue(comparison["near_identical_full_mix"])
        self.assertIn("vocal_stem_matches_full_mix_separator_fallback", report["warnings"])
        self.assertEqual(transcriber.calls, [])

    def test_valid_vocals_emit_relative_timing_events_and_intervals(self) -> None:
        final, stem = audio_fixture()
        transcriber = FakeTranscriber()
        lyrics = "[Verse 1]\nWe light the night\nAnd hold the sky"
        outputs = self.run_lyrics(final, stem, fake_pipeline(transcriber), lyrics=lyrics)
        report = json.loads(outputs[0])
        self.assertEqual(report["schema"], "diffusiongemma.timed_lyrics_report")
        self.assertEqual(report["version"], 1)
        self.assertEqual(report["analysis_status"], "timing_ready")
        self.assertTrue(report["timing_ready"])
        self.assertTrue(outputs[2])
        self.assertEqual(report["lyrics_sha256"], hashlib.sha256(lyrics.encode("utf-8")).hexdigest())
        self.assertEqual(report["master_audio_sha256"], self.timed._waveform_sha256(final))
        self.assertEqual(report["model_id"], "fake/whisper-tiny")
        self.assertEqual(report["excerpt"], {"start_seconds": 0.0, "duration_seconds": 12.0, "end_seconds": 12.0})
        self.assertEqual(report["vocal_intervals"], [{"start_seconds": 2.0, "end_seconds": 5.0}])
        self.assertEqual(
            report["instrumental_intervals"],
            [
                {"start_seconds": 0.0, "end_seconds": 2.0},
                {"start_seconds": 5.0, "end_seconds": 12.0},
            ],
        )
        event = report["events"][0]
        self.assertAlmostEqual(event["start_seconds"], 2.25)
        self.assertAlmostEqual(event["end_seconds"], 3.75)
        self.assertEqual(event["text"], "we light the night")
        self.assertEqual(event["transcript"], "we light the night")
        self.assertEqual(event["authored_lines"], ["We light the night"])
        self.assertEqual(event["confidence"], 1.0)
        self.assertIn("Verified authored-lyric timing", outputs[3])

    def test_direct_mtb_features_follow_the_loaded_model_dtype(self) -> None:
        class Tokenizer:
            @staticmethod
            def convert_ids_to_tokens(_row):
                return ["<|0.00|>", "hello", "<|1.00|>"]

        class Processor:
            tokenizer = Tokenizer()

            @staticmethod
            def __call__(_audio, sampling_rate, return_tensors):
                self.assertEqual(sampling_rate, 16_000)
                self.assertEqual(return_tensors, "pt")
                return {"input_features": torch.ones((1, 80, 300), dtype=torch.float32)}

            @staticmethod
            def batch_decode(_predicted_ids, skip_special_tokens):
                self.assertTrue(skip_special_tokens)
                return ["hello"]

        class HalfModel:
            device = torch.device("cpu")
            dtype = torch.float16
            config = SimpleNamespace(max_length=32)

            @staticmethod
            def generate(input_features, **_kwargs):
                self.assertEqual(input_features.dtype, torch.float16)
                return torch.tensor([[1, 2, 3]], dtype=torch.long)

        result = self.timed._direct_mtb_transcription(
            {"processor": Processor(), "model": HalfModel()},
            torch.zeros(16_000, dtype=torch.float32),
            16_000,
            "en",
        )
        self.assertEqual(result["text"], "hello")
        self.assertEqual(result["tokens"], ["<|0.00|>", "hello", "<|1.00|>"])

    def test_instrumental_excerpt_returns_natural_fallback_without_whisper(self) -> None:
        final, stem = audio_fixture(vocal_sections=())
        transcriber = FakeTranscriber()
        outputs = self.run_lyrics(final, stem, fake_pipeline(transcriber))
        report = json.loads(outputs[0])
        self.assertEqual(report["analysis_status"], "natural_fallback")
        self.assertFalse(report["timing_ready"])
        self.assertIn("no_vocal_activity_detected", report["warnings"])
        self.assertEqual(report["vocal_intervals"], [])
        self.assertEqual(
            report["instrumental_intervals"],
            [{"start_seconds": 0.0, "end_seconds": 12.0}],
        )
        self.assertEqual(transcriber.calls, [])

    def test_weak_authored_alignment_falls_back_instead_of_forcing_words(self) -> None:
        final, stem = audio_fixture()
        transcriber = FakeTranscriber(
            [{"chunks": [{"text": "completely unrelated phrase", "timestamp": [0.0, 1.0]}]}]
        )
        outputs = self.run_lyrics(final, stem, fake_pipeline(transcriber), confidence=0.65)
        report = json.loads(outputs[0])
        self.assertEqual(report["analysis_status"], "natural_fallback")
        self.assertFalse(report["timing_ready"])
        self.assertEqual(report["events"], [])
        self.assertIn("alignment_confidence_below_threshold", report["warnings"])

    def test_adjacent_whisper_fragments_can_align_to_the_same_authored_line(self) -> None:
        final, stem = audio_fixture()
        transcriber = FakeTranscriber(
            [
                {
                    "chunks": [
                        {"text": "we light", "timestamp": [0.1, 0.8]},
                        {"text": "the night", "timestamp": [0.8, 1.5]},
                    ]
                }
            ]
        )
        outputs = self.run_lyrics(final, stem, fake_pipeline(transcriber))
        report = json.loads(outputs[0])
        self.assertTrue(report["timing_ready"])
        self.assertEqual(len(report["events"]), 2)
        self.assertEqual(
            [event["authored_lines"] for event in report["events"]],
            [["We light the night"], ["We light the night"]],
        )

    def test_chunk_timestamp_offset_after_fifteen_seconds_is_preserved(self) -> None:
        final, stem = audio_fixture(duration=35.0, vocal_sections=((20.0, 23.0),))
        transcriber = FakeTranscriber(
            [{"chunks": [{"text": "late in the song", "timestamp": [0.4, 1.4]}]}]
        )
        outputs = self.run_lyrics(
            final,
            stem,
            fake_pipeline(transcriber),
            lyrics="[Verse]\nLate in the song",
        )
        report = json.loads(outputs[0])
        self.assertTrue(report["timing_ready"])
        self.assertAlmostEqual(report["events"][0]["start_seconds"], 20.4)
        self.assertAlmostEqual(report["events"][0]["end_seconds"], 21.4)
        self.assertAlmostEqual(transcriber.calls[0]["chunk_offset_seconds"], 20.0)
        self.assertAlmostEqual(transcriber.calls[0]["absolute_song_start_seconds"], 20.0)
        transcription = report["diagnostics"]["transcription"]
        self.assertFalse(transcription["mtb_thirty_second_offset_logic_used"])
        self.assertTrue(transcription["chunk_offsets_added_explicitly"])
        self.assertLessEqual(transcriber.calls[0]["duration"], 15.0)

    def test_excerpt_start_is_added_for_audio_crop_but_events_remain_excerpt_relative(self) -> None:
        final, stem = audio_fixture(duration=40.0, vocal_sections=((21.0, 24.0),))
        transcriber = FakeTranscriber(
            [{"chunks": [{"text": "inside the window", "timestamp": [0.5, 1.5]}]}]
        )
        outputs = self.run_lyrics(
            final,
            stem,
            fake_pipeline(transcriber),
            lyrics="[Verse]\nInside the window",
            excerpt_start=20.0,
            excerpt_duration=10.0,
        )
        report = json.loads(outputs[0])
        self.assertTrue(report["timing_ready"])
        self.assertEqual(report["vocal_intervals"], [{"start_seconds": 1.0, "end_seconds": 4.0}])
        self.assertAlmostEqual(report["events"][0]["start_seconds"], 1.5)
        self.assertAlmostEqual(transcriber.calls[0]["absolute_song_start_seconds"], 21.0)

    def test_transcription_exception_falls_back_offloads_model_and_clears_cuda(self) -> None:
        final, stem = audio_fixture()
        transcriber = FakeTranscriber(exception=RuntimeError("synthetic decoder failure"))
        pipeline = fake_pipeline(transcriber, with_model=True)
        with mock.patch.object(self.timed.torch.cuda, "empty_cache") as empty_cache:
            outputs = self.run_lyrics(final, stem, pipeline)
        report = json.loads(outputs[0])
        self.assertEqual(report["analysis_status"], "natural_fallback")
        self.assertFalse(report["timing_ready"])
        self.assertTrue(outputs[2])
        self.assertEqual(report["diagnostics"]["exception_type"], "RuntimeError")
        self.assertEqual(pipeline["model"].moves[-1], "cpu")
        empty_cache.assert_called_once()

    def test_missing_stem_still_offloads_an_already_loaded_whisper_model(self) -> None:
        final, _stem = audio_fixture()
        pipeline = fake_pipeline(FakeTranscriber(), with_model=True)
        outputs = self.timed.DiffusionGemmaTimedLyricsAnalyzer().analyze(
            final,
            "Lyrics + lip sync",
            "We light the night",
            self.timed._waveform_sha256(final),
            12.0,
            0.0,
            12.0,
            "en",
            0.55,
            None,
            pipeline,
        )
        report = json.loads(outputs[0])
        self.assertIn("vocal_stem_missing", report["warnings"])
        self.assertEqual(pipeline["model"].moves[-1], "cpu")

    def test_success_is_cached_by_audio_excerpt_lyrics_and_model(self) -> None:
        final, stem = audio_fixture()
        transcriber = FakeTranscriber()
        pipeline = fake_pipeline(transcriber)
        first = self.run_lyrics(final, stem, pipeline)
        second = self.run_lyrics(final, stem, pipeline)
        self.assertTrue(json.loads(first[0])["timing_ready"])
        cached_report = json.loads(second[0])
        self.assertTrue(cached_report["timing_ready"])
        self.assertTrue(cached_report["diagnostics"]["cache_hit"])
        self.assertEqual(len(transcriber.calls), 1)
        self.assertIn("cache", second[1].casefold())

    def test_nonfinite_or_misaligned_stem_is_never_a_blocker(self) -> None:
        final, stem = audio_fixture()
        bad = {"waveform": stem["waveform"].clone(), "sample_rate": stem["sample_rate"]}
        bad["waveform"][..., 7] = math.nan
        outputs = self.run_lyrics(final, bad, fake_pipeline(FakeTranscriber()))
        report = json.loads(outputs[0])
        self.assertEqual(report["analysis_status"], "natural_fallback")
        self.assertTrue(outputs[2])
        self.assertIn("vocal_stem_invalid_or_misaligned", report["warnings"])

        shortened = {
            "waveform": stem["waveform"][..., :-1000].clone(),
            "sample_rate": stem["sample_rate"],
        }
        outputs = self.run_lyrics(final, shortened, fake_pipeline(FakeTranscriber()))
        report = json.loads(outputs[0])
        self.assertEqual(report["analysis_status"], "natural_fallback")
        self.assertTrue(outputs[2])
        self.assertIn("vocal_stem_invalid_or_misaligned", report["warnings"])

    def test_package_registers_analyzer(self) -> None:
        package = load_package()
        self.assertIs(
            package.NODE_CLASS_MAPPINGS["DiffusionGemmaTimedLyricsAnalyzer"],
            sys.modules[f"{package.__name__}.timed_lyrics_nodes"].DiffusionGemmaTimedLyricsAnalyzer,
        )
        self.assertEqual(
            package.NODE_DISPLAY_NAME_MAPPINGS["DiffusionGemmaTimedLyricsAnalyzer"],
            "DiffusionGemma Timed Lyrics Analyzer",
        )


if __name__ == "__main__":
    unittest.main()
