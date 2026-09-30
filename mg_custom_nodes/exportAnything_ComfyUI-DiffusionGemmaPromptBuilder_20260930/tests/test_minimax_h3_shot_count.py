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
    package_name = f"diffusiongemma_shot_count_{uuid.uuid4().hex}"
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


class MiniMaxH3ShotCountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()

    @staticmethod
    def target_profile_required_args() -> tuple:
        return (
            "minimax_h3",
            "auto_scene_audio",
            "",
            20.0,
            "",
            "16:9",
            "",
            "",
            False,
            "auto",
            "",
        )

    @staticmethod
    def t2va_prompt() -> str:
        return (
            "integrated_multimodal_description: [Shot 1] A locked wide camera holds on a runner at a rain-darkened starting line. "
            "[Shot 2] At 00:06.000, the camera cuts to a low tracking view as the same runner accelerates around the bend. "
            "[Shot 3] At 00:13.000, the camera cuts to a static close view as the runner stops beneath the finish light. "
            "The final frame holds on the runner standing still beneath the light.\n\n"
            "overall_soundscape: Rain, measured footfalls, and distant crowd ambience surround the track.\n\n"
            "non_diegetic_music: N/A"
        )

    @classmethod
    def ref2va_prompt(cls) -> str:
        return (
            "subject_definitions:\n"
            "<Subject 1>: A runner with a dark rain jacket and silver shoes.\n\n"
            "summary:\n"
            "[reference generation] A runner completes one measured lap in the rain.\n\n"
            "retention_analysis:\n"
            "<Subject 1>: fully_preserved - The dark jacket, silver shoes, and athletic proportions remain unchanged.\n\n"
            "detailed_description:\n"
            "Naturalistic live-action sports photography uses cool rain-muted color, soft stadium light, and restrained film grain. "
            + cls.t2va_prompt().split("integrated_multimodal_description: ", 1)[1].split("\n\noverall_soundscape:", 1)[0]
            + "\n\noverall_soundscape:\nRain, measured footfalls, and distant crowd ambience surround the track.\n\n"
            "non_diegetic_music:\nN/A"
        )

    def test_ui_contract_keeps_presets_and_appends_typed_custom_count(self) -> None:
        optional = self.nodes.DiffusionGemmaTargetProfile.INPUT_TYPES()["optional"]
        self.assertEqual(
            list(optional),
            [
                "minimax_h3_mode",
                "minimax_h3_shot_count",
                "minimax_h3_custom_shot_count",
                "minimax_h3_dialogue_mode",
                "minimax_h3_dialogue_line_count",
                "minimax_h3_dialogue_guidance",
                "ltx_generation_mode",
                "ltx_long_horizon_mode",
                "ltx_camera_capability",
            ],
        )
        choices, config = optional["minimax_h3_shot_count"]
        self.assertEqual(choices, ["auto", *(str(value) for value in range(1, 13)), "custom"])
        self.assertEqual(config["default"], "auto")
        input_type, custom_config = optional["minimax_h3_custom_shot_count"]
        self.assertEqual(input_type, "INT")
        self.assertEqual(custom_config["default"], 12)
        self.assertEqual(custom_config["min"], 1)
        self.assertEqual(custom_config["max"], 99)

    def test_config_defaults_and_legacy_positional_calls_remain_compatible(self) -> None:
        profile_node = self.nodes.DiffusionGemmaTargetProfile()

        legacy_config, legacy_json = profile_node.build(*self.target_profile_required_args())
        self.assertEqual(legacy_config.minimax_h3_mode, "t2va")
        self.assertEqual(legacy_config.minimax_h3_shot_count, "auto")
        self.assertEqual(json.loads(legacy_json)["minimax_h3_shot_count"], "auto")

        ref_config, ref_json = profile_node.build(
            *self.target_profile_required_args(),
            "ref2va",
        )
        self.assertEqual(ref_config.minimax_h3_mode, "ref2va")
        self.assertEqual(ref_config.minimax_h3_shot_count, "auto")
        self.assertEqual(json.loads(ref_json)["minimax_h3_mode"], "ref2va")
        self.assertEqual(json.loads(ref_json)["minimax_h3_shot_count"], "auto")

        explicit_config, explicit_json = profile_node.build(
            *self.target_profile_required_args(),
            "ref2va",
            "12",
        )
        self.assertEqual(explicit_config.minimax_h3_mode, "ref2va")
        self.assertEqual(explicit_config.minimax_h3_shot_count, "12")
        self.assertEqual(json.loads(explicit_json)["minimax_h3_shot_count"], "12")

        custom_config, custom_json = profile_node.build(
            *self.target_profile_required_args(),
            "ref2va",
            "custom",
            17,
        )
        self.assertEqual(custom_config.minimax_h3_shot_count, "17")
        self.assertEqual(json.loads(custom_json)["minimax_h3_shot_count"], "17")
        self.assertNotIn("minimax_h3_custom_shot_count", json.loads(custom_json))

        self.assertEqual(self.nodes._normalize_minimax_h3_shot_count("13"), "13")
        self.assertEqual(self.nodes._normalize_minimax_h3_shot_count("099"), "99")
        for invalid in (None, "", "0", "100", "invalid", "custom"):
            with self.subTest(invalid=invalid):
                self.assertEqual(self.nodes._normalize_minimax_h3_shot_count(invalid), "auto")

    def test_custom_count_is_used_only_when_the_selector_is_custom(self) -> None:
        self.assertEqual(self.nodes._resolve_minimax_h3_shot_count("5", 17), "5")
        self.assertEqual(self.nodes._resolve_minimax_h3_shot_count("auto", 17), "auto")
        self.assertEqual(self.nodes._resolve_minimax_h3_shot_count("custom", 17), "17")
        self.assertEqual(self.nodes._resolve_minimax_h3_shot_count("custom", 100), "auto")

    def test_auto_honors_brief_count_and_explicit_setting_overrides_it(self) -> None:
        brief = (
            "Create three hard-cut shots. Shot 1 establishes the runner. "
            "Shot 2 follows the turn. Shot 3 holds on the finish."
        )
        self.assertEqual(self.nodes._minimax_h3_effective_shot_count(brief, "auto"), 3)
        self.assertEqual(self.nodes._minimax_h3_effective_shot_count(brief, "5"), 5)

        automatic_prompt = self.nodes._build_model_prompt(
            brief,
            self.nodes.DEFAULT_MASTER_PROMPT,
            "minimax_h3",
            {"source": "none"},
            target_duration_seconds=20.0,
            minimax_h3_mode="t2va",
            minimax_h3_shot_count="auto",
        )
        self.assertIn(
            "Use exactly 3 consecutively numbered shots as requested in the user brief.",
            automatic_prompt,
        )

        explicit_prompt = self.nodes._build_model_prompt(
            brief,
            self.nodes.DEFAULT_MASTER_PROMPT,
            "minimax_h3",
            {"source": "none"},
            target_duration_seconds=20.0,
            minimax_h3_mode="t2va",
            minimax_h3_shot_count="5",
        )
        self.assertIn("Use exactly 5 consecutively numbered shots.", explicit_prompt)
        self.assertIn("overrides any conflicting shot-count wording", explicit_prompt)

    def test_t2va_and_ref2va_validation_receive_explicit_count(self) -> None:
        brief = "Create three hard-cut shots from start to finish."
        cases = (
            ("t2va", self.t2va_prompt()),
            ("ref2va", self.ref2va_prompt()),
        )
        for mode, candidate in cases:
            with self.subTest(mode=mode, selected="3"):
                reasons = self.nodes._minimax_h3_prompt_validation_reasons(
                    candidate,
                    20.0,
                    brief,
                    minimax_h3_mode=mode,
                    minimax_h3_shot_count="3",
                )
                self.assertNotIn("minimax_h3_requested_shot_count_mismatch", reasons)
            with self.subTest(mode=mode, selected="4"):
                reasons = self.nodes._minimax_h3_prompt_validation_reasons(
                    candidate,
                    20.0,
                    brief,
                    minimax_h3_mode=mode,
                    minimax_h3_shot_count="4",
                )
                self.assertIn("minimax_h3_requested_shot_count_mismatch", reasons)

    def test_transition_repair_covers_double_digit_shots_without_inventing_a_view(self) -> None:
        shots = [
            "[Shot 1] A locked wide shot follows a runner beginning a measured walk."
        ]
        for shot in range(2, 13):
            timestamp = f"00:{shot - 1:02d}.000"
            if shot == 10:
                body = "The camera tracks beside the runner as the measured walk continues."
            elif shot == 11:
                body = "A static close-up shows the runner looking toward the finish."
            else:
                body = f"the camera cuts to a locked wide shot of the runner completing beat {shot}."
            separator = " " if shot == 11 else ", "
            shots.append(f"[Shot {shot}] At {timestamp}{separator}{body}")

        repaired = self.nodes._repair_minimax_h3_ref_detailed_timeline(
            " ".join(shots),
            15.0,
        )
        self.assertIn(
            "[Shot 10] At 00:09.000, the shot cuts to: The camera tracks beside",
            repaired,
        )
        self.assertIn(
            "[Shot 11] At 00:10.000, the shot cuts to: A static close-up",
            repaired,
        )
        self.assertEqual(
            repaired.count(
                "[Shot 12] At 00:11.000, the camera cuts to a locked wide shot"
            ),
            1,
        )
        self.assertNotIn("static wide view as", repaired)

        for shot in (20, 99):
            with self.subTest(later_shot=shot):
                high_number_repaired = self.nodes._repair_minimax_h3_ref_detailed_timeline(
                    f"[Shot 1] A locked wide shot holds. [Shot {shot}] At 00:01.000, A tracking view continues.",
                    15.0,
                )
                self.assertIn(
                    f"[Shot {shot}] At 00:01.000, the shot cuts to: A tracking view continues.",
                    high_number_repaired,
                )

        candidate = (
            f"integrated_multimodal_description: {repaired}\n\n"
            "overall_soundscape: N/A\n\n"
            "non_diegetic_music: N/A"
        )
        reasons = self.nodes._minimax_h3_prompt_validation_reasons(
            candidate,
            15.0,
            "Create a 15-second sequence.",
            minimax_h3_mode="t2va",
            minimax_h3_shot_count="12",
        )
        self.assertNotIn("minimax_h3_cut_transition_invalid", reasons)
        self.assertNotIn("minimax_h3_requested_shot_count_mismatch", reasons)

    def test_refinement_uses_target_profile_count_for_both_h3_modes(self) -> None:
        brief = "Create three hard-cut shots of a runner completing a lap."
        for mode, candidate in (
            ("t2va", self.t2va_prompt()),
            ("ref2va", self.ref2va_prompt()),
        ):
            with self.subTest(mode=mode):
                refinement = self.nodes._build_minimax_h3_refinement_prompt(
                    brief,
                    candidate,
                    20.0,
                    "auto_scene_audio",
                    ["minimax_h3_requested_shot_count_mismatch"],
                    minimax_h3_mode=mode,
                    minimax_h3_shot_count="5",
                )
                self.assertIn("Use exactly 5 consecutively numbered shots.", refinement)
                self.assertIn("Shot-count selector: 5; effective required count: 5; source: target_profile", refinement)
                self.assertNotIn("Use exactly 0 consecutively numbered shots", refinement)

    def test_initial_and_refinement_contracts_allow_expressive_h3_camera_paths(self) -> None:
        brief = "Create four measured shots of a performer moving to the music."
        exact_fallback = "The camera remains locked-off in a stable view."
        for mode, candidate in (
            ("t2va", self.t2va_prompt()),
            ("ref2va", self.ref2va_prompt()),
        ):
            with self.subTest(mode=mode, stage="initial"):
                model_prompt = self.nodes._build_model_prompt(
                    brief,
                    self.nodes.DEFAULT_MASTER_PROMPT,
                    "minimax_h3",
                    {"source": "none"},
                    target_duration_seconds=20.0,
                    minimax_h3_mode=mode,
                    minimax_h3_shot_count="4",
                )
                self.assertIn(exact_fallback, model_prompt)
                self.assertIn("immediately after its [Shot N] marker", model_prompt)
                self.assertIn("An incoming cut, shot size, angle, subject framing", model_prompt)
                self.assertIn("never omit it to save words in a dense multi-shot timeline", model_prompt)
                self.assertNotIn("LTX Stable / base-model Camera Capability", model_prompt)
                self.assertIn("Honor explicit user and reference camera choreography exactly", model_prompt)
                self.assertIn("orbit or arc, roll or rotation, spin or swirl", model_prompt)
                self.assertIn("sweeping or whip movement", model_prompt)
                self.assertIn("pronounced or layered parallax", model_prompt)
                self.assertIn("coherent compound/multi-axis path", model_prompt)
                self.assertIn("proportional to creative strength", model_prompt)
                self.assertNotIn("Never invent an orbit, arc, roll, rotation, spin, swirl", model_prompt)
                self.assertNotIn('"ltx_camera_capability"', model_prompt)
                self.assertNotIn('"ltx_complexity_limits"', model_prompt)
                native_model_prompt = self.nodes._build_model_prompt(
                    brief,
                    self.nodes.DEFAULT_MASTER_PROMPT,
                    "minimax_h3",
                    {"source": "none"},
                    target_duration_seconds=20.0,
                    minimax_h3_mode=mode,
                    minimax_h3_shot_count="4",
                    native_h3_output=True,
                )
                self.assertNotIn('"ltx_camera_capability"', native_model_prompt)
                self.assertNotIn('"ltx_complexity_limits"', native_model_prompt)
                self.assertIn("coherent compound/multi-axis path", native_model_prompt)

            with self.subTest(mode=mode, stage="refinement"):
                refinement = self.nodes._build_minimax_h3_refinement_prompt(
                    brief,
                    candidate,
                    20.0,
                    "auto_scene_audio",
                    ["minimax_h3_shot_camera_unspecified"],
                    minimax_h3_mode=mode,
                    minimax_h3_shot_count="4",
                )
                self.assertIn(exact_fallback, refinement)
                self.assertIn("Repair every action-only or framing-only shot", refinement)
                self.assertIn("Orbiting, swirling, sweeping or whip movement", refinement)
                self.assertIn("pronounced parallax", refinement)
                self.assertIn("coherent compound paths are valid H3 choices", refinement)
                self.assertNotIn("do not rewrite a safe hold as a more ambitious move", refinement)

    def test_h3_camera_contract_examples_are_validator_recognizable(self) -> None:
        camera_sentences = (
            "The camera remains locked-off in a stable view.",
            "The camera slowly pushes in, then settles.",
            "The camera slowly pulls back, then settles.",
            "The camera slowly pans left, then settles.",
            "The camera slowly tilts down, then settles.",
            "The camera slowly tracks right, then settles.",
            "The camera orbits clockwise around the dancer, accelerating into a close profile view.",
            "The camera swirls upward around the performer and resolves in an overhead view.",
            "The camera rolls thirty degrees counterclockwise while tracking the singer.",
            "The camera whip-pans right to catch the drummer on the downbeat.",
            "The viewpoint rotates around the pair before settling behind them.",
            "A lateral truck past foreground lights creates pronounced layered parallax.",
        )
        for sentence in camera_sentences:
            with self.subTest(sentence=sentence):
                self.assertTrue(self.nodes._minimax_h3_shot_has_camera_spec(sentence))

        non_h3_prompt = self.nodes._build_model_prompt(
            "Create a quiet portrait.",
            self.nodes.DEFAULT_MASTER_PROMPT,
            "ltx",
            {"source": "none"},
        )
        self.assertNotIn("The camera remains locked-off in a stable view.", non_h3_prompt)
        self.assertIn('"ltx_camera_capability"', non_h3_prompt)

    def test_director_tooltips_explain_h3_camera_creativity_is_not_ltx_gated(self) -> None:
        required = self.nodes.DiffusionGemmaCoTGenerator.INPUT_TYPES()["required"]
        mode_tooltip = required["creativity_mode"][1]["tooltip"]
        strength_tooltip = required["creative_strength"][1]["tooltip"]
        self.assertIn("MiniMax H3 and LTX", mode_tooltip)
        self.assertIn("expressive, physically coherent per-shot choreography", mode_tooltip)
        self.assertIn("without inheriting LTX's Camera Capability", strength_tooltip)

    def test_five_shot_fifteen_second_prompts_make_the_endpoint_exclusive(self) -> None:
        expected_chain = "0 < t2 < t3 < t4 < t5 < 00:15.000"
        brief = "Create a 15-second five-shot animated sequence."
        model_prompt = self.nodes._build_model_prompt(
            brief,
            self.nodes.DEFAULT_MASTER_PROMPT,
            "minimax_h3",
            {"source": "none"},
            target_duration_seconds=15.0,
            minimax_h3_mode="ref2va",
            minimax_h3_shot_count="5",
            native_h3_output=True,
        )
        self.assertIn(expected_chain, model_prompt)
        self.assertIn("00:15.000 is exclusive", model_prompt)
        self.assertIn("never a valid shot-start timestamp", model_prompt)

        refinement = self.nodes._build_minimax_h3_refinement_prompt(
            brief,
            self.ref2va_prompt(),
            15.0,
            "auto_scene_audio",
            ["minimax_h3_cut_timestamp_out_of_range"],
            minimax_h3_mode="ref2va",
            minimax_h3_shot_count="5",
            native_prompt_only=True,
        )
        self.assertIn(expected_chain, refinement)
        self.assertIn("strictly earlier than 00:15.000", refinement)
        self.assertIn("not a valid start for the last shot", refinement)

    def test_cut_timestamp_endpoint_boundary_remains_fail_closed(self) -> None:
        brief = "Create exactly three shots in 15 seconds."
        before_endpoint = self.t2va_prompt().replace("00:13.000", "00:14.999")
        at_endpoint = self.t2va_prompt().replace("00:13.000", "00:15.000")
        after_endpoint = self.t2va_prompt().replace("00:13.000", "00:15.001")

        def reasons(prompt: str) -> list[str]:
            return self.nodes._minimax_h3_prompt_validation_reasons(
                prompt,
                15.0,
                brief,
                minimax_h3_mode="t2va",
                minimax_h3_shot_count="3",
            )

        self.assertNotIn("minimax_h3_cut_timestamp_out_of_range", reasons(before_endpoint))
        self.assertIn("minimax_h3_cut_timestamp_out_of_range", reasons(at_endpoint))
        self.assertIn("minimax_h3_cut_timestamp_out_of_range", reasons(after_endpoint))

    def test_shot_count_instruction_is_isolated_to_h3_model_prompts(self) -> None:
        brief = "Create a quiet portrait with a slow camera move."
        for profile in ("ltx", "ideogram4"):
            with self.subTest(profile=profile):
                prompt = self.nodes._build_model_prompt(
                    brief,
                    self.nodes.DEFAULT_MASTER_PROMPT,
                    profile,
                    {"source": "none"},
                    minimax_h3_shot_count="7",
                )
                self.assertNotIn("Use exactly 7 consecutively numbered shots", prompt)
                self.assertNotIn('"minimax_h3_shot_count"', prompt)

        for mode in ("t2va", "ref2va"):
            with self.subTest(profile="minimax_h3", mode=mode):
                prompt = self.nodes._build_model_prompt(
                    brief,
                    self.nodes.DEFAULT_MASTER_PROMPT,
                    "minimax_h3",
                    {"source": "none"},
                    minimax_h3_mode=mode,
                    minimax_h3_shot_count="7",
                )
                self.assertIn("Use exactly 7 consecutively numbered shots.", prompt)
                self.assertIn('"minimax_h3_shot_count": "7"', prompt)


if __name__ == "__main__":
    unittest.main()
