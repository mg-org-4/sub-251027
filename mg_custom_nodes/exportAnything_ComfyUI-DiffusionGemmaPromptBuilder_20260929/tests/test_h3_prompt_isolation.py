from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]


def load_nodes_module():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    package_name = f"diffusiongemma_h3_prompt_isolation_{uuid.uuid4().hex}"
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


class H3PromptIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()

    def context(self):
        return self.nodes.GemmaContext(
            user_prompt="Create a concise ten-second scene.",
            source="none",
            media_metadata={"source": "none", "duration_seconds": 10.0},
        )

    def h3_target(self):
        return self.nodes.TargetProfileConfig(
            target_profile="minimax_h3",
            minimax_h3_mode="t2va",
        )

    def test_bare_thought_only_output_is_reasoning_not_candidate_answer(self) -> None:
        raw = (
            "thought\n"
            "The user wants a MiniMax H3 video prompt from an image.\n"
            "- `ltx_prompt`: Empty.\n"
            "- `ideogram_prompt`: An alternate still-image prompt.\n"
            "- `minimax_h3_prompt`: Draft the requested timeline.\n"
            "*Refining&#x20;**`minimax_h3_prompt`**&#x20;content:*\n"
            "[Shot 1] A close-up holds on the subject.\n"
            "`non_diegetic_music`: Somber strings transition to fast-"
        )

        reasoning, answer = self.nodes._split_reasoning_and_answer(raw)

        self.assertEqual(answer, "")
        self.assertIn("The user wants a MiniMax H3 video prompt", reasoning)
        self.assertIn("minimax_h3_prompt", reasoning)
        self.assertIn("Somber strings transition to fast-", reasoning)

    def test_minimal_h3_json_is_expanded_to_the_host_packet(self) -> None:
        h3_prompt = (
            "integrated_multimodal_description: [Shot 1] A locked medium shot "
            "holds on a performer beneath a single spotlight. The final frame "
            "holds on the performer centered beneath the unchanged light.\n\n"
            "overall_soundscape: Quiet room tone and soft fabric movement.\n\n"
            "non_diegetic_music: N/A"
        )
        packet, parse_valid, warning = self.nodes.repair_or_salvage_prompt_packet(
            json.dumps({"minimax_h3_prompt": h3_prompt}),
            self.context(),
            self.h3_target(),
            deterministic_transport_repair=True,
        )

        self.assertTrue(parse_valid)
        self.assertEqual(warning, "")
        self.assertEqual(packet["minimax_h3_prompt"], h3_prompt)
        self.assertEqual(packet["ltx_prompt"], "")
        self.assertEqual(packet["ideogram_prompt"], "")
        self.assertEqual(packet["negative_prompt"], "")
        self.assertEqual(packet["scene_segments"], [])
        self.assertEqual(
            packet["metadata"]["host_structural_repairs"],
            [
                "completed_inactive_packet_field:ltx_prompt",
                "completed_inactive_packet_field:ideogram_prompt",
                "completed_inactive_packet_field:negative_prompt",
                "completed_inactive_packet_field:scene_segments",
            ],
        )
        self.assertTrue(packet["metadata"]["model_output_json_parse_valid"])

    def test_unambiguous_three_digit_cut_timestamp_is_canonicalized(self) -> None:
        prompt = (
            "integrated_multimodal_description: [Shot 1] A locked wide shot "
            "establishes the subject. [Shot 2] At 04:500, the camera cuts to "
            "a close-up and remains locked-off in a stable view. The final frame "
            "holds on the centered subject.\n\n"
            "overall_soundscape: Quiet room tone.\n\n"
            "non_diegetic_music: N/A"
        )

        repaired, repairs = self.nodes._repair_minimax_h3_t2va_contract_transport(
            prompt,
            "t2va",
        )

        self.assertIn("[Shot 2] At 00:04.500, the camera cuts to", repaired)
        self.assertNotIn("At 04:500", repaired)
        self.assertIn("canonicalized_timed_cut_timestamps", repairs)

    def test_h3_model_prompt_excludes_inactive_target_contracts_and_controls(self) -> None:
        sentinel_values = (
            "DO_NOT_LEAK_LTX_CONTROL",
            "DO_NOT_LEAK_IDEOGRAM_CONTROL",
            "DO_NOT_LEAK_LTX_METADATA",
            "DO_NOT_LEAK_IDEOGRAM_METADATA",
        )
        model_prompt = self.nodes._build_model_prompt(
            "A performer turns toward a spotlight.",
            self.nodes.DEFAULT_MASTER_PROMPT,
            "minimax_h3",
            {
                "source": "none",
                "duration_seconds": 10.0,
                "minimax_h3_mode": "t2va",
                "ltx_generation_mode": sentinel_values[2],
                "ideogram_render_style": sentinel_values[3],
            },
            ltx_style=sentinel_values[0],
            ideogram_render_style=sentinel_values[1],
            ideogram_exact_text="DO_NOT_LEAK_IDEOGRAM_EXACT_TEXT",
            thinking_mode="off",
            minimax_h3_mode="t2va",
        )

        self.assertIn('"minimax_h3_prompt":', model_prompt)
        self.assertIn(
            "The JSON object must contain minimax_h3_prompt, and no other top-level keys.",
            model_prompt,
        )
        for forbidden in (
            "ltx_prompt",
            "ideogram_prompt",
            "scene_segments",
            '"ltx_',
            '"ltx25_',
            '"ideogram_',
            *sentinel_values,
            "DO_NOT_LEAK_IDEOGRAM_EXACT_TEXT",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, model_prompt)

    def test_h3_refinement_prompt_excludes_inactive_target_contracts(self) -> None:
        refinement_prompt = self.nodes._build_minimax_h3_refinement_prompt(
            "A silent two-shot orbital rescue scene.",
            "candidate",
            10.0,
            "auto_scene_audio",
            ["minimax_h3_ref_sections_invalid"],
            "ref2va",
            "<Picture 1>: sole visual reference",
        )

        self.assertIn('"minimax_h3_prompt"', refinement_prompt)
        for inactive_field in (
            '"ltx_prompt"',
            '"ideogram_prompt"',
            '"negative_prompt"',
            '"scene_segments"',
            '"metadata"',
        ):
            with self.subTest(inactive_field=inactive_field):
                self.assertNotIn(inactive_field, refinement_prompt)

    def runtime(self):
        return self.nodes.RuntimeConfig(
            model_path="unused-local-checkpoint",
            backend="transformers_inprocess",
            dtype="auto",
            quantization="none",
            local_files_only=True,
            unload_policy="unload_after_run",
            max_memory_gb=20.0,
            status={"ready": True, "supports_pixels": True},
        )

    def image_context(self):
        return self.nodes.GemmaContext(
            user_prompt="Describe the reference faithfully.",
            images=torch.zeros((1, 8, 8, 3), dtype=torch.float32),
            source="image",
            media_metadata={
                "source": "image",
                "reference_image_count": 1,
                "image_role": "primary_image_source",
            },
        )

    def grounded_ledger(self, registry: dict) -> dict:
        asset = registry["assets"][0]
        sample = asset["samples"][0]
        return {
            "schema": "dg-grounding-ledger/1",
            "analysis_status": "grounded",
            "observed_facts": [
                {
                    "fact_id": "fact-1",
                    "claim": "A directly visible subject is present.",
                    "confidence": "high",
                    "categories": ["object"],
                    "evidence": [
                        {
                            "asset_id": asset["asset_id"],
                            "sample_ordinal": sample["sample_ordinal"],
                            "source_frame_index": sample["source_frame_index"],
                            "timecode_seconds": sample["timecode_seconds"],
                        }
                    ],
                    "typed_claims": [],
                }
            ],
            "inferred_facts": [],
            "creative_additions": [],
            "uncertainties": [],
            "grounding_failure_reasons": [],
        }

    def backend_result(self, ledger: dict, stage: str):
        return self.nodes.BackendRunResult(
            decoded_text=json.dumps(ledger),
            transport={
                "pixel_transport_confirmed": True,
                "not_applicable": False,
                "mismatch_reasons": [],
            },
            telemetry={
                "schema_version": "dg-denoising-telemetry/1",
                "forward_count": 2,
                "snapshots": [{"canvas": 1, "step": 1}],
                "stream_ended": True,
                "final_output": {"text": [], "token_count": [1]},
                "errors": [],
                "refusal": {"detected": False},
            },
            effective_sampling={"profile": "checkpoint_defaults"},
            timing={"total_seconds": 0.01},
            stage=stage,
        )

    @staticmethod
    def packet(prompt: str, ready: bool) -> dict:
        return {
            "ltx_prompt": "",
            "ideogram_prompt": "",
            "minimax_h3_prompt": prompt,
            "negative_prompt": "",
            "scene_segments": [],
            "metadata": {
                "target_profile": "minimax_h3",
                "ready_for_generation": ready,
                "blocked_reasons": [] if ready else ["json_parse_invalid"],
            },
        }

    def test_audit_retry_forces_thinking_off_in_call_and_backend_options(self) -> None:
        context = self.image_context()
        registry = self.nodes._grounding_asset_registry(context)
        ledger = self.grounded_ledger(registry)
        invalid_packet = self.packet("salvaged invalid H3 output", False)
        valid_packet = self.packet("strict JSON retry result", True)
        observed_calls: list[tuple[str, str, str]] = []

        def audit_legacy(*args, **kwargs):
            options = kwargs["backend_options"]
            options.call_budget.claim()
            observed_calls.append((args[9], options.thinking_mode, options.stage))
            kwargs["backend_results"].append(
                self.backend_result(ledger, options.stage)
            )
            if options.stage == "audit_director":
                return (
                    copy.deepcopy(invalid_packet),
                    "thought\nunfinished planning",
                    "unfinished planning",
                    "",
                    False,
                    "salvaged malformed output",
                    copy.deepcopy(invalid_packet["metadata"]),
                )
            packet = copy.deepcopy(valid_packet)
            return packet, "compiler raw", "", "", True, "", packet["metadata"]

        with patch.object(
            self.nodes,
            "_run_generation_packet_legacy",
            side_effect=audit_legacy,
        ) as legacy, patch.object(self.nodes, "_maybe_unload"):
            result = self.nodes._run_generation_packet(
                self.runtime(),
                context,
                self.h3_target(),
                thinking_mode="on",
                grounding_guard_config={
                    "mode": "audit",
                    "retry_on_uncertain": True,
                },
            )

        self.assertEqual(legacy.call_count, 2)
        self.assertEqual(
            observed_calls,
            [
                ("on", "on", "audit_director"),
                ("off", "off", "audit_director_retry"),
            ],
        )
        self.assertEqual(result[0]["minimax_h3_prompt"], "strict JSON retry result")


if __name__ == "__main__":
    unittest.main()
