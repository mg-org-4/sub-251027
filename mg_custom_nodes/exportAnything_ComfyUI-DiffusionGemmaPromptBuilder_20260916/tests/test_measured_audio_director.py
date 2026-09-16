from __future__ import annotations

import importlib.util
import json
import sys
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]


def load_nodes_module():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    package_name = f"diffusiongemma_measured_audio_fixture_{uuid.uuid4().hex}"
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


class MeasuredAudioDirectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()

    def test_cot_exposes_optional_measured_audio_report_input(self) -> None:
        optional = self.nodes.DiffusionGemmaCoTGenerator.INPUT_TYPES()["optional"]
        self.assertIn("measured_audio_report_json", optional)
        input_type, spec = optional["measured_audio_report_json"]
        self.assertEqual(input_type, "STRING")
        self.assertTrue(spec["forceInput"])

    def test_context_enrichment_is_bounded_and_does_not_mutate_caller(self) -> None:
        original = self.nodes.GemmaContext(
            user_prompt="One continuous dance shot",
            media_metadata={"source": "image"},
        )
        report = {
            "schema": "diffusiongemma.music_audition_report",
            "status": "ready",
            "selected_audio_sha256": "abc123",
            "selected_excerpt": {
                "start_seconds": 44.0,
                "duration_seconds": 20.0,
                "onset_rate_hz": 4.2,
                "tonal_family_stability": 0.86,
            },
        }
        enriched = self.nodes._context_with_measured_audio_report(
            original,
            json.dumps(report),
        )
        self.assertNotIn("ltx_measured_audio_report", original.media_metadata)
        self.assertEqual(
            enriched.media_metadata["ltx_measured_audio_report"]["selected_audio_sha256"],
            "abc123",
        )

    def test_invalid_measured_audio_report_fails_before_director(self) -> None:
        for value in ("[]", "not json"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "valid JSON object"):
                    self.nodes._normalize_measured_audio_report_json(value)

    def test_ltx_compiler_treats_rendered_audio_as_authoritative(self) -> None:
        metadata = {
            "source": "none",
            "ltx_measured_audio_report": {
                "status": "ready",
                "selected_excerpt": {
                    "start_seconds": 44.0,
                    "duration_seconds": 20.0,
                    "onset_rate_hz": 4.2,
                    "stable_windows": [[44.0, 50.0], [56.0, 64.0]],
                },
            },
        }
        prompt = self.nodes._build_model_prompt(
            "A singer dances in one continuous shot",
            self.nodes.DEFAULT_MASTER_PROMPT,
            "ltx",
            metadata,
            audio_mode="explicit_sound_design",
            audio_guidance="Perform to the connected selected song.",
            target_duration_seconds=20.0,
            ltx_generation_mode="text_to_video",
        )
        self.assertIn("exact rendered song excerpt", prompt)
        self.assertIn("requested BPM, key, energy, or meter are not facts", prompt)
        self.assertIn("stable or lower-density spans", prompt)
        self.assertIn("Vocal-activity proxies do not by themselves authorize visible singing", prompt)
        self.assertIn("performance-mode control decides whether the performer lip-syncs", prompt)
        self.assertIn('"ltx_measured_audio_report"', prompt)


if __name__ == "__main__":
    unittest.main()
