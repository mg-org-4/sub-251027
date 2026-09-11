from __future__ import annotations

import importlib.util
import json
import re
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]


def load_nodes_module():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    package_name = f"diffusiongemma_dialogue_patch_integration_{uuid.uuid4().hex}"
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
    return sys.modules[f"{package_name}.nodes"]


class MiniMaxH3DialoguePatchIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()

    def runtime(self):
        return self.nodes.RuntimeConfig(
            model_path="unused-local-checkpoint",
            backend="transformers_inprocess",
            dtype="auto",
            quantization="none",
            local_files_only=True,
            unload_policy="unload_after_run",
            max_memory_gb=20.0,
            status={"ready": True, "supports_pixels": False},
        )

    def context(self):
        return self.nodes.GemmaContext(
            user_prompt="Create exactly two shots in a quiet train compartment.",
            images=None,
            source="none",
            media_metadata={"source": "none"},
        )

    def target(self):
        return self.nodes.TargetProfileConfig(
            target_profile="minimax_h3",
            audio_mode="auto_scene_audio",
            target_duration_seconds=12.0,
            minimax_h3_mode="t2va",
            minimax_h3_shot_count="2",
            minimax_h3_dialogue_mode="required",
            minimax_h3_dialogue_line_count=2,
            minimax_h3_dialogue_guidance=(
                "Mara makes a concise departure announcement, then Ivo confirms the plan."
            ),
        )

    @staticmethod
    def speechless_native_prompt() -> str:
        return (
            "integrated_multimodal_description: [Shot 1] A locked medium two-shot frames Mara (S1) on the left "
            "and Ivo (S2) on the right in a quiet train compartment. Soft dawn light crosses the wall while the "
            "camera remains static and Mara rests both hands on a suitcase. "
            "[Shot 2] At 00:06.000, the camera cuts to a static close-up of Ivo (S2) as he closes the suitcase in "
            "one visible motion. The final frame holds on Ivo seated beside the closed suitcase in the static "
            "composition.\n\n"
            "overall_soundscape: Low train rumble, cloth movement, and a synchronized suitcase-latch click align "
            "with the visible actions.\n\n"
            "non_diegetic_music: N/A"
        )

    @classmethod
    def compiler_output(cls) -> str:
        return (
            f"{cls.speechless_native_prompt()}\n\n"
            "GROUNDING_EVIDENCE_REPORT_ID: dggr-dialogue-test\n"
            "USED_GROUNDING_FACT_IDS: fact-train-1"
        )

    @staticmethod
    def valid_patch_output() -> str:
        return json.dumps(
            {
                "schema": "dg-h3-dialogue-patch/1",
                "dialogue_patch": [
                    {
                        "shot": 1,
                        "subject_tag": "",
                        "speaker_id": "S1",
                        "language": "English",
                        "delivery": "quiet resolve",
                        "text": "We leave at dawn.",
                    },
                    {
                        "shot": 2,
                        "subject_tag": "",
                        "speaker_id": "S2",
                        "language": "English",
                        "delivery": "calm confidence",
                        "text": "Then pack light.",
                    },
                ],
            },
            separators=(",", ":"),
        )

    def run_with_patch_output(self, dialogue_patch_output: str):
        stages: list[str] = []
        prompts: list[str] = []

        def backend(
            _config,
            prompt_text,
            _media_context,
            _max_tokens,
            _node_id,
            options,
            _backend_results,
        ):
            stage = str(options.stage)
            stages.append(stage)
            prompts.append(prompt_text)
            if stage == "compiler":
                return self.compiler_output()
            if stage.startswith("dialogue_patch_"):
                return dialogue_patch_output
            self.fail(f"unexpected full-prompt repair stage: {stage}")

        options = self.nodes.BackendRunOptions(stage="compiler")
        with patch.object(self.nodes, "_run_backend_for_packet", side_effect=backend):
            result = self.nodes._run_generation_packet_legacy(
                self.runtime(),
                self.context(),
                self.target(),
                runtime_required=True,
                max_new_tokens=768,
                max_output_chars=8000,
                backend_options=options,
                backend_results=[],
                manage_unload=False,
                max_refinement_attempts_override=2,
                native_h3_output=True,
            )
        return result, stages, prompts

    def test_count_only_failure_uses_one_strict_dialogue_patch_without_rewriting_storyboard(self) -> None:
        initial = self.speechless_native_prompt()
        initial_reasons = self.nodes._minimax_h3_prompt_validation_reasons(
            initial,
            duration_seconds=12.0,
            user_prompt=self.context().user_prompt,
            audio_mode="auto_scene_audio",
            max_prompt_chars=8000,
            minimax_h3_mode="t2va",
            minimax_h3_shot_count="2",
            minimax_h3_dialogue_mode="required",
            minimax_h3_dialogue_line_count=2,
        )
        self.assertEqual(initial_reasons, ["minimax_h3_dialogue_count_mismatch"])

        result, stages, prompts = self.run_with_patch_output(self.valid_patch_output())
        packet, metadata = result[0], result[-1]
        patched = packet["minimax_h3_prompt"]

        self.assertEqual(stages, ["compiler", "dialogue_patch_1"])
        self.assertEqual(stages.count("dialogue_patch_1"), 1)
        self.assertFalse(any(stage.startswith("target_repair_") for stage in stages))
        self.assertIn("The storyboard below is immutable", prompts[1])
        self.assertIn("Do not rewrite, summarize, or return it", prompts[1])

        self.assertEqual(re.findall(r"\[Shot \d+\]", patched), ["[Shot 1]", "[Shot 2]"])
        self.assertEqual(re.findall(r"\b\d\d:\d\d\.\d{3}\b", patched), ["00:06.000"])
        self.assertIn(
            "A locked medium two-shot frames Mara (S1) on the left and Ivo (S2) on the right",
            patched,
        )
        self.assertIn(
            "the camera cuts to a static close-up of Ivo (S2) as he closes the suitcase in one visible motion",
            patched,
        )
        self.assertIn(
            "overall_soundscape: Low train rumble, cloth movement, and a synchronized suitcase-latch click align "
            "with the visible actions.\n\nnon_diegetic_music: N/A",
            patched,
        )
        self.assertEqual(patched.count("<d>"), 2)
        self.assertEqual(patched.count("</d>"), 2)

        self.assertEqual(metadata["grounding_evidence_report_id"], "dggr-dialogue-test")
        self.assertEqual(metadata["used_grounding_fact_ids"], ["fact-train-1"])
        self.assertEqual(
            metadata["compiler_provenance_source"],
            "model_declared_native_footer",
        )
        self.assertTrue(metadata["ready_for_generation"])
        self.assertTrue(metadata["minimax_h3_dialogue_contract_satisfied"])
        self.assertEqual(
            metadata["minimax_h3_refinement"]["repair_mode"],
            "dialogue_patch",
        )
        self.assertTrue(metadata["minimax_h3_refinement"]["accepted"])

    def test_invalid_dialogue_patch_fails_closed_without_any_full_prompt_repair(self) -> None:
        invalid_patch = json.dumps(
            {
                "schema": "dg-h3-dialogue-patch/1",
                "dialogue_patch": [
                    {
                        "shot": 1,
                        "subject_tag": "",
                        "speaker_id": "S1",
                        "language": "English",
                        "delivery": "quiet resolve",
                        "text": "We leave at dawn.",
                    }
                ],
            }
        )

        result, stages, _prompts = self.run_with_patch_output(invalid_patch)
        packet, metadata = result[0], result[-1]

        self.assertEqual(stages, ["compiler", "dialogue_patch_1", "dialogue_patch_2"])
        self.assertEqual(
            [stage for stage in stages if stage.startswith("dialogue_patch_")],
            ["dialogue_patch_1", "dialogue_patch_2"],
        )
        self.assertFalse(any(stage.startswith("target_repair_") for stage in stages))
        self.assertEqual(packet["minimax_h3_prompt"], self.speechless_native_prompt())
        self.assertFalse(metadata["ready_for_generation"])
        self.assertFalse(metadata["minimax_h3_dialogue_contract_satisfied"])
        self.assertIn(
            "minimax_h3_dialogue_count_mismatch",
            metadata["blocked_reasons"],
        )
        refinement = metadata["minimax_h3_refinement"]
        self.assertEqual(refinement["repair_mode"], "dialogue_patch")
        self.assertFalse(refinement["accepted"])
        self.assertEqual(
            refinement["dialogue_patch"]["patch_validation_reasons"],
            ["dialogue_patch_count_invalid"],
        )


if __name__ == "__main__":
    unittest.main()
