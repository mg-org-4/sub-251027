from __future__ import annotations

import importlib.util
import json
import re
import sys
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
MANIFEST_PATH = ROOT / "benchmarks" / "ltx25_long_horizon_v1" / "manifest.json"


def load_nodes_module():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    package_name = f"diffusiongemma_ltx25_long_horizon_{uuid.uuid4().hex}"
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


class Ltx25LongHorizonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()
        cls.contract = sys.modules[f"{cls.nodes.__package__}.ltx25_contract"]
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_target_profile_appends_backward_compatible_control(self) -> None:
        inputs = self.nodes.DiffusionGemmaLTX25TargetProfile.INPUT_TYPES()
        self.assertEqual(list(inputs["required"])[-1], "long_horizon_mode")
        choices, options = inputs["required"]["long_horizon_mode"]
        self.assertEqual(choices, ["Off", "Auto (>20 seconds)", "On"])
        self.assertEqual(options["default"], "Off")

        # A call saved before the new final widget existed keeps historical
        # behavior instead of silently enabling experimental prompt planning.
        legacy_config, legacy_json = self.nodes.DiffusionGemmaLTX25TargetProfile().build(
            generation_mode="Image to video",
            target_duration_seconds=60.0,
            style_guidance="restrained surrealism",
            audio_mode="explicit_sound_design",
            audio_guidance="wind and fireplace crackle",
            negative_prompt_mode="custom",
            negative_prompt_guidance="no captions",
        )
        self.assertEqual(legacy_config.ltx_long_horizon_mode, "off")
        self.assertEqual(json.loads(legacy_json)["ltx_long_horizon_mode"], "off")

        auto_config, auto_json = self.nodes.DiffusionGemmaLTX25TargetProfile().build(
            generation_mode="Image to video",
            target_duration_seconds=60.0,
            style_guidance="restrained surrealism",
            audio_mode="explicit_sound_design",
            audio_guidance="wind and fireplace crackle",
            negative_prompt_mode="custom",
            negative_prompt_guidance="no captions",
            long_horizon_mode="Auto (>20 seconds)",
        )
        self.assertEqual(auto_config.ltx_long_horizon_mode, "auto")
        self.assertEqual(json.loads(auto_json)["ltx_long_horizon_mode"], "auto")

    def test_auto_activation_boundary_and_forced_modes(self) -> None:
        cases = (
            ("off", 60.0, False),
            ("auto", 20.0, False),
            ("auto", 20.001, True),
            ("on", 5.0, True),
        )
        for mode, duration, expected_active in cases:
            with self.subTest(mode=mode, duration=duration):
                plan = self.contract.ltx25_long_horizon_plan(mode, duration)
                self.assertEqual(plan["requested_mode"], mode)
                self.assertEqual(plan["active"], expected_active)
                self.assertEqual(plan["phase_count"], 4 if expected_active else 0)
                self.assertEqual(len(plan["phases"]), 4 if expected_active else 0)

        at_boundary = self.contract.ltx25_long_horizon_plan("auto", 20.0)
        above_boundary = self.contract.ltx25_long_horizon_plan("auto", 20.001)
        self.assertIn("at_or_below_20", at_boundary["activation_reason"])
        self.assertIn("above_20", above_boundary["activation_reason"])

    def test_active_plan_is_four_contiguous_phases_not_three_chapters(self) -> None:
        plan = self.contract.ltx25_long_horizon_plan("on", 60.0)
        phases = plan["phases"]
        self.assertEqual(
            [phase["name"] for phase in phases],
            ["establish", "commit", "sustain_reveal", "settle_hold"],
        )
        self.assertEqual(plan["phase_count"], 4)
        self.assertNotEqual(plan["phase_count"], 3)
        self.assertEqual(phases[0]["start_seconds"], 0.0)
        self.assertEqual(phases[-1]["end_seconds"], 60.0)
        for left, right in zip(phases, phases[1:]):
            self.assertEqual(left["end_seconds"], right["start_seconds"])
        for phase in phases:
            self.assertLess(phase["start_seconds"], phase["end_seconds"])
            self.assertIsInstance(phase["purpose"], str)
            self.assertTrue(phase["purpose"].strip())
        durations = {
            phase["name"]: phase["end_seconds"] - phase["start_seconds"]
            for phase in phases
        }
        self.assertEqual(max(durations, key=durations.get), "sustain_reveal")

    def test_invalid_duration_uses_a_safe_five_second_plan(self) -> None:
        plan = self.contract.ltx25_long_horizon_plan("on", 0)
        self.assertTrue(plan["active"])
        self.assertEqual(plan["phases"][0]["start_seconds"], 0.0)
        self.assertEqual(plan["phases"][-1]["end_seconds"], 5.0)

    def test_active_compiler_contract_requests_compact_continuous_output(self) -> None:
        active = self.contract.ltx25_compiler_contract(
            "image_to_video",
            60.0,
            long_horizon_mode="on",
        ).casefold()
        inactive = self.contract.ltx25_compiler_contract(
            "image_to_video",
            60.0,
            long_horizon_mode="off",
        ).casefold()

        self.assertIn("silent", active)
        self.assertIn("four-phase", active)
        for label in ("establish", "commit", "sustain/reveal", "settle/hold"):
            self.assertIn(label, active)
        self.assertRegex(active, r"4\s*[-\u2013]\s*8(?:\s+\w+){0,2}\s+sentences")
        self.assertRegex(
            active,
            r"(?:at most|no more than|maximum of|never more than)\s+200\s+words",
        )
        self.assertIn("minimal critical anchor set", active)
        self.assertIn("one continuous take", active)
        self.assertRegex(active, r"requested transformations?\b")
        for prohibited_output in ("headings", "shot labels", "timecodes", "cuts"):
            self.assertIn(prohibited_output, active)
        self.assertNotRegex(active, r"\b(?:three|3)\s+chapters?\b")

        self.assertIn("150-220 words", inactive)
        self.assertNotIn("sustain/reveal", inactive)

    def test_long_horizon_t2v_is_one_take_and_visual_only_is_silent(self) -> None:
        budget = self.contract.ltx25_complexity_budget(
            "text_to_video",
            60.0,
            "on",
        )
        self.assertEqual(budget["max_shots"], 1)
        self.assertEqual(budget["max_cuts"], 0)

        silent_contract = self.contract.ltx25_compiler_contract(
            "text_to_video",
            60.0,
            long_horizon_mode="on",
            audio_enabled=False,
        ).casefold()
        self.assertIn("one continuous take with no cuts", silent_contract)
        self.assertNotIn("a cut is allowed", silent_contract)
        for positive_audio_instruction in (
            "integrate sound",
            "continuous audio identity",
            "synchronized transient sounds",
            "requested speech",
            "audible facts",
        ):
            self.assertNotIn(positive_audio_instruction, silent_contract)

        model_prompt = self.nodes._build_model_prompt(
            user_prompt="A camera slowly crosses a silent empty gallery.",
            master_prompt="",
            target_profile="ltx",
            media_metadata={"source": "none"},
            audio_mode="visual_only",
            target_duration_seconds=60.0,
            ltx_generation_mode="text_to_video",
            ltx_long_horizon_mode="on",
        ).casefold()
        self.assertIn(
            "do not include sound, music, speech, narration, voiceover",
            model_prompt,
        )
        for positive_audio_instruction in (
            "integrate sound beside",
            "one continuous audio identity",
            "synchronized transient sounds",
            "weave concrete sound into the timeline",
        ):
            self.assertNotIn(positive_audio_instruction, model_prompt)

    def test_long_horizon_diagnostics_are_advisory_only(self) -> None:
        metadata = {
            "source": "image",
            "ltx25_context_schema": self.nodes.LTX25_CONTRACT_SCHEMA,
            "ltx_first_frame_attached": True,
            "ltx_last_frame_attached": False,
            "ltx_frame_pair_attached": False,
        }
        mode_report = self.contract.resolve_ltx25_generation_mode(
            "image_to_video", metadata, 1
        )
        sparse_prompt = (
            "From the exact supplied first-frame viewpoint, a medium shot remains static "
            "while the woman slowly turns her head and room tone continues."
        )
        report = self.contract.validate_ltx25_prompt(
            sparse_prompt,
            mode_report,
            metadata,
            60.0,
            long_horizon_mode="on",
        )

        long_horizon_warnings = [
            warning
            for warning in report["warnings"]
            if warning.startswith("ltx_long_horizon_")
        ]
        self.assertTrue(report["ready"], report["reasons"])
        self.assertTrue(long_horizon_warnings)
        self.assertFalse(
            [reason for reason in report["reasons"] if reason.startswith("ltx_long_horizon_")]
        )

    def test_long_horizon_mode_survives_salvage_and_template_fallback(self) -> None:
        metadata = {
            "source": "image",
            "ltx25_context_schema": self.nodes.LTX25_CONTRACT_SCHEMA,
            "ltx_first_frame_attached": True,
            "ltx_last_frame_attached": False,
            "ltx_frame_pair_attached": False,
        }
        context = self.nodes.GemmaContext(
            user_prompt="Continue from the supplied first frame in one long take.",
            source="image",
            media_metadata=metadata,
        )
        target = self.nodes.TargetProfileConfig(
            target_profile="ltx",
            target_duration_seconds=60.0,
            ltx_generation_mode="image_to_video",
            ltx_long_horizon_mode="on",
        )

        cases = (
            ("", False),
            (
                "A wide shot at eye level begins from the supplied frame and slowly "
                "moves backward while steady wind continues throughout.",
                True,
            ),
        )
        for raw_output, expected_salvage in cases:
            with self.subTest(expected_salvage=expected_salvage):
                packet, parse_valid, warning = (
                    self.nodes.repair_or_salvage_prompt_packet(
                        raw_output,
                        context,
                        target,
                    )
                )
                self.assertFalse(parse_valid)
                self.assertTrue(warning)
                self.assertEqual(
                    bool(packet["metadata"].get("plain_text_salvage")),
                    expected_salvage,
                )
                controls = packet["metadata"]["controls"]
                self.assertEqual(
                    controls["ltx_generation_mode_effective"],
                    "image_to_video",
                )
                self.assertEqual(controls["ltx_long_horizon_mode"], "on")
                self.assertTrue(controls["ltx_long_horizon_active"])
                self.assertEqual(
                    controls["ltx_long_horizon_plan"]["phase_count"],
                    4,
                )

    def test_duration_benchmark_has_fixed_controls_and_paired_seeds(self) -> None:
        manifest = self.manifest
        self.assertEqual(
            manifest["schema_version"], "dg-ltx25-long-horizon-benchmark/1"
        )
        self.assertEqual(manifest["execution_status"], "protocol_only_unexecuted")
        self.assertEqual(manifest["results"], [])
        self.assertEqual(len(manifest["asset_requirements"]), 4)
        self.assertEqual(
            manifest["variant_order"],
            ["bare", "current_director", "long_horizon", "long_horizon_structural"],
        )
        self.assertEqual(
            {(pair["stage_1"], pair["stage_2"]) for pair in manifest["seed_pairs"]},
            {(17, 42), (101, 420), (809, 421)},
        )
        self.assertEqual(
            [case["duration_seconds"] for case in manifest["cases"]],
            [5.0, 20.0, 60.0],
        )
        self.assertEqual(
            [case["expected_auto_long_horizon_active"] for case in manifest["cases"]],
            [False, False, True],
        )

        for case in manifest["cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(
                    case["frames"],
                    int(case["duration_seconds"] * case["fps"]) + 1,
                )
                self.assertEqual(case["frames"] % 8, 1)
                self.assertEqual(list(case["variants"]), manifest["variant_order"])
                self.assertEqual(case["mode"], "image_to_video")
                self.assertEqual(
                    case["fixed_generation_control"]["sampler"],
                    "euler_ancestral",
                )
                self.assertEqual(
                    case["fixed_generation_control"]["scheduler"],
                    "bong_tangent",
                )
                self.assertEqual(
                    case["fixed_generation_control"]["stage_1_steps"],
                    12,
                )
                self.assertLessEqual(len(case["critical_anchors"]), 3)
                self.assertEqual(
                    case["rubric"]["phase_names"],
                    ["establish", "commit", "sustain_reveal", "settle_hold"],
                )
                self.assertFalse(case["rubric"]["diagnostics_block_generation"])

                auto_plan = self.contract.ltx25_long_horizon_plan(
                    "auto", case["duration_seconds"]
                )
                self.assertEqual(
                    auto_plan["active"], case["expected_auto_long_horizon_active"]
                )

                long_prompt = case["variants"]["long_horizon"]["prompt"]
                structural_prompt = case["variants"]["long_horizon_structural"]["prompt"]
                self.assertEqual(long_prompt, structural_prompt)
                self.assertEqual(
                    case["variants"]["long_horizon"]["structural_conditioning"],
                    "first_frame",
                )
                self.assertEqual(
                    case["variants"]["long_horizon_structural"][
                        "structural_conditioning"
                    ],
                    "first_frame_plus_depth",
                )

    def test_checked_in_long_horizon_prompts_meet_output_shape_contract(self) -> None:
        word_re = re.compile(r"\b[\w'-]+\b")
        sentence_re = re.compile(r"[^.!?]+(?:[.!?]+|$)")
        timecode_re = re.compile(r"\b\d{1,2}:\d{2}(?:\.\d+)?\b|\[\s*shot\b", re.I)
        cut_re = re.compile(r"\b(?:hard\s+cut|jump\s+cut|cut|cuts|dissolve|wipe)\b", re.I)

        for case in self.manifest["cases"]:
            prompt = case["variants"]["long_horizon"]["prompt"]
            with self.subTest(case=case["id"]):
                word_count = len(word_re.findall(prompt))
                sentence_count = sum(
                    1 for match in sentence_re.finditer(prompt) if match.group(0).strip()
                )
                self.assertGreaterEqual(word_count, case["rubric"]["min_words"])
                self.assertLessEqual(word_count, case["rubric"]["max_words"])
                self.assertGreaterEqual(sentence_count, case["rubric"]["sentence_range"][0])
                self.assertLessEqual(sentence_count, case["rubric"]["sentence_range"][1])
                self.assertNotIn("\n", prompt)
                self.assertIsNone(timecode_re.search(prompt))
                self.assertIsNone(cut_re.search(prompt))
                diagnostics = self.contract.analyze_ltx25_long_horizon_prompt(
                    prompt,
                    "on",
                    case["duration_seconds"],
                    case["mode"],
                )
                self.assertEqual(diagnostics["warnings"], [])


if __name__ == "__main__":
    unittest.main()
