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
    package_name = f"diffusiongemma_camera_capability_{uuid.uuid4().hex}"
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


class LTX25CameraCapabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()
        cls.contract = sys.modules[cls.nodes.__package__ + ".ltx25_contract"]
        cls.mode_report = {
            "resolved_mode": "text_to_video",
            "first_frame_attached": False,
            "last_frame_attached": False,
            "warnings": [],
        }

    def validate(
        self,
        prompt: str,
        capability: str = "stable",
        duration: float = 12.0,
        long_horizon_mode: str = "off",
    ) -> dict:
        return self.contract.validate_ltx25_prompt(
            prompt,
            self.mode_report,
            {},
            duration,
            long_horizon_mode,
            capability,
        )

    def test_target_profile_missing_field_normalizes_to_stable(self) -> None:
        normalized = self.nodes._target_profile_config_to_dict(
            {
                "target_profile": "ltx",
                "ltx_generation_mode": "Image to video",
            }
        )
        self.assertEqual(normalized["ltx_camera_capability"], "stable")

        config = self.nodes._make_target_profile_config(target_profile="ltx")
        self.assertEqual(config.ltx_camera_capability, "stable")
        payload = json.loads(self.nodes._target_profile_node_result(config)[1])
        self.assertEqual(payload["ltx_camera_capability"], "stable")

    def test_ltx_target_profile_defaults_stable_and_requires_explicit_advanced_opt_in(self) -> None:
        node = self.nodes.DiffusionGemmaLTX25TargetProfile()
        stable, stable_json = node.build(
            generation_mode="Image to video",
            target_duration_seconds=25.0,
            style_guidance="",
            audio_mode="auto_scene_audio",
            audio_guidance="",
            negative_prompt_mode="auto",
            negative_prompt_guidance="",
            long_horizon_mode="On",
        )
        advanced, advanced_json = node.build(
            generation_mode="Image to video",
            target_duration_seconds=25.0,
            style_guidance="",
            audio_mode="auto_scene_audio",
            audio_guidance="",
            negative_prompt_mode="auto",
            negative_prompt_guidance="",
            long_horizon_mode="On",
            camera_capability="Advanced / controlled camera",
        )
        self.assertEqual(stable.ltx_camera_capability, "stable")
        self.assertEqual(json.loads(stable_json)["ltx_camera_capability"], "stable")
        self.assertEqual(advanced.ltx_camera_capability, "advanced")
        self.assertEqual(json.loads(advanced_json)["ltx_camera_capability"], "advanced")

    def test_stable_compiler_removes_ambitious_unspecified_camera_policy(self) -> None:
        stable = self.contract.ltx25_compiler_contract(
            "image_to_video",
            25.0,
            long_horizon_mode="on",
            camera_capability="stable",
        ).casefold()
        advanced = self.contract.ltx25_compiler_contract(
            "image_to_video",
            25.0,
            long_horizon_mode="on",
            camera_capability="advanced",
        ).casefold()
        self.assertIn("stable / base model", stable)
        self.assertIn("one restrained single-axis move", stable)
        self.assertIn("rather than sustaining camera travel", stable)
        self.assertNotIn("fill time with sustained movement, parallax", stable)
        self.assertIn("one dominant physically continuous camera intention", advanced)
        self.assertIn("fill time with sustained movement, parallax", advanced)

    def test_stable_creativity_keeps_art_direction_but_caps_camera_ambition(self) -> None:
        for mode in ("editorial", "cinematic", "concept_art", "wild"):
            instruction = self.nodes._ltx_creativity_instruction(
                mode,
                1.0,
                "image_to_video",
                "stable",
            ).casefold()
            with self.subTest(mode=mode):
                self.assertIn(mode.replace("_", "-"), instruction)
                self.assertIn("stable / base model", instruction)
                self.assertIn("one restrained single-axis move", instruction)
                self.assertIn("camera swirl", instruction)
                self.assertNotIn("sweeping arc or orbit", instruction)

        advanced = self.nodes._ltx_creativity_instruction(
            "wild",
            1.0,
            "image_to_video",
            "advanced",
        ).casefold()
        self.assertIn("sweeping arc or orbit", advanced)
        self.assertIn("compound camera", advanced)

    def test_stable_validation_rejects_each_high_risk_camera_family(self) -> None:
        cases = {
            "ltx_stable_camera_orbit_or_rotation": [
                "The camera begins a slow dramatic circular orbit around the subject.",
                "The camera slows its rotation.",
                "The camera spins around her.",
                "The camera swirls around her.",
            ],
            "ltx_stable_camera_sweeping_or_whip_motion": [
                "The camera makes a sweeping move around the performer.",
                "The camera performs a whip-pan to the right.",
            ],
            "ltx_stable_camera_dolly_zoom": [
                "The camera performs a dolly zoom toward her face.",
            ],
            "ltx_stable_camera_shake_or_handheld_pursuit": [
                "The camera shakes violently beside the runner.",
                "A handheld pursuit follows the runner down the alley.",
            ],
            "ltx_stable_camera_compound_or_reversal": [
                "The camera uses a compound multi-axis path around the performer.",
                "The camera pushes in and then pulls back before the end.",
            ],
            "ltx_stable_camera_pronounced_parallax": [
                "The shot creates dizzying layered parallax around the subject.",
            ],
        }
        for expected_reason, clauses in cases.items():
            for clause in clauses:
                prompt = f"A medium shot at eye level frames a performer. {clause}"
                with self.subTest(reason=expected_reason, clause=clause):
                    stable = self.validate(prompt, "stable")
                    advanced = self.validate(prompt, "advanced")
                    self.assertIn(expected_reason, stable["reasons"])
                    self.assertNotIn(expected_reason, advanced["reasons"])

    def test_user_nightmare_prompt_is_hard_failure_only_in_stable_mode(self) -> None:
        prompt = (
            "A medium shot at eye level frames the performer. Through most of the shot, "
            "the camera continues its sweeping arc, creating a dizzying parallax effect "
            "with the bills that flutter occasionally as if caught in a faint draft. "
            "Near the end, the camera slows its rotation and settles on her face."
        )
        stable = self.validate(prompt, "stable", 25.0, "on")
        advanced = self.validate(prompt, "advanced", 25.0, "on")
        self.assertFalse(stable["ready"])
        self.assertIn("ltx_stable_camera_orbit_or_rotation", stable["reasons"])
        self.assertIn("ltx_stable_camera_sweeping_or_whip_motion", stable["reasons"])
        self.assertIn("ltx_stable_camera_pronounced_parallax", stable["reasons"])
        self.assertIn("ltx_stable_camera_sustained_travel", stable["reasons"])
        self.assertFalse(
            any(reason.startswith("ltx_stable_camera_") for reason in advanced["reasons"])
        )

    def test_negated_risky_language_is_not_treated_as_positive_camera_direction(self) -> None:
        prompt = (
            "A medium shot at eye level frames the performer while the camera remains locked "
            "and static. No sweeping arc, whip-pan, dolly zoom, camera shake, handheld pursuit, "
            "circular orbit, rotation, spin, swirl, roll, compound multi-axis movement, direction "
            "reversal, pronounced layered dizzying parallax, or sustained camera travel is used."
        )
        report = self.validate(prompt, "stable", 25.0, "on")
        self.assertFalse(
            any(reason.startswith("ltx_stable_camera_") for reason in report["reasons"]),
            report,
        )

    def test_subject_swirl_is_not_misclassified_as_camera_swirl(self) -> None:
        prompt = (
            "A medium shot at eye level frames a dancer while the camera remains locked and "
            "static. Her skirt swirls around her ankles as she turns once."
        )
        report = self.validate(prompt, "stable")
        self.assertNotIn("ltx_stable_camera_orbit_or_rotation", report["reasons"])

    def test_subject_and_environment_motion_do_not_bind_to_stable_camera(self) -> None:
        safe_prompts = (
            "A medium shot at eye level frames her; the camera stays static while coins circle around the performer.",
            "A medium shot at eye level frames her; the camera is stabilized as ribbons orbit the dancer.",
            "A medium shot at eye level frames her; the camera is locked while a sweeping curtain crosses the background.",
            "A medium shot at eye level frames her; the camera is locked as the performer begins a sweeping arc with her arm.",
            "A medium shot at eye level frames her; the shot stays steady while the dancer performs a sweeping arc.",
            "A medium shot at eye level frames her; the camera is locked while the dancer performs an orbit around the stage.",
            "A medium shot at eye level frames her; the camera is locked while the dancer moves throughout.",
            "A static medium shot at eye level frames her while the performer moves continuously throughout.",
            "A medium shot at eye level holds with framing fixed while bills travel throughout the background.",
            "A medium shot at eye level holds; the camera is locked while fog moves for the duration.",
        )
        for prompt in safe_prompts:
            with self.subTest(prompt=prompt):
                report = self.validate(prompt, "stable", 25.0, "on")
                self.assertFalse(
                    any(
                        reason.startswith("ltx_stable_camera_")
                        for reason in report["reasons"]
                    ),
                    report,
                )

    def test_negating_a_safe_hold_does_not_negate_the_risky_move_that_follows(self) -> None:
        prompts = (
            "A medium shot at eye level frames her. Without a locked camera, the shot begins a slow circular orbit.",
            "A medium shot at eye level frames her. No longer locked, the camera begins a slow circular orbit.",
            "A medium shot at eye level frames her. The camera does not stop as it orbits her.",
            "A medium shot at eye level frames her. The camera keeps moving as it orbits her.",
            "A medium shot at eye level frames her. A circular tracking arc surrounds the subject.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertIn(
                    "ltx_stable_camera_orbit_or_rotation",
                    self.validate(prompt, "stable")["reasons"],
                )

    def test_short_range_camera_pronouns_preserve_risky_motion_ownership(self) -> None:
        risky = (
            "A medium shot at eye level frames her. The camera begins locked. Then it orbits.",
            "A medium shot at eye level frames her. The shot starts static. It then spins.",
            "A medium shot at eye level frames her. The camera starts stable, but it soon swirls.",
            "A medium shot at eye level frames her. The camera begins in a stable view. Near the end, it slows its rotation.",
            "A medium shot at eye level frames her. The camera begins with a stable view. Near the end, it slows its rotation.",
            "A medium shot at eye level frames her while the camera begins locked. Then it orbits around her.",
            "A medium shot at eye level frames her while the camera starts static. Near the end, it spins.",
            "A medium shot frames her and the camera remains stable. Then it rotates around her.",
        )
        for prompt in risky:
            with self.subTest(prompt=prompt):
                self.assertIn(
                    "ltx_stable_camera_orbit_or_rotation",
                    self.validate(prompt, "stable")["reasons"],
                )
        subject_pronoun = (
            "A medium shot at eye level frames a robot while the camera stays static. "
            "The robot lifts one hand. It spins once."
        )
        self.assertNotIn(
            "ltx_stable_camera_orbit_or_rotation",
            self.validate(subject_pronoun, "stable")["reasons"],
        )
        ambiguous_subject_pronouns = (
            "A medium shot at eye level frames a toy top while the camera remains locked. Then it spins.",
            "A medium shot at eye level frames a coin while the shot remains static. It then rotates.",
            "A medium shot at eye level frames a robot while the camera remains locked on it. Then it spins once.",
        )
        for prompt in ambiguous_subject_pronouns:
            with self.subTest(prompt=prompt):
                self.assertNotIn(
                    "ltx_stable_camera_orbit_or_rotation",
                    self.validate(prompt, "stable")["reasons"],
                )

    def test_negated_stop_or_static_state_does_not_hide_persistent_camera_rotation(self) -> None:
        prompts = (
            "A medium shot at eye level frames her. The camera never stops rotating around her.",
            "A medium shot at eye level frames her. The camera does not stop rotating around her.",
            "A medium shot at eye level frames her. The camera never stops its orbit around her.",
            "A medium shot at eye level frames her. The camera does not cease to orbit her.",
            "A medium shot at eye level frames her. The camera is never static and instead orbits her.",
            "A medium shot at eye level frames her. The camera does not slow its sweeping arc around her.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertIn(
                    "ltx_stable_camera_orbit_or_rotation",
                    self.validate(prompt, "stable")["reasons"],
                )

    def test_simple_stable_camera_states_and_single_axis_moves_pass_risk_validation(self) -> None:
        clauses = (
            "the camera remains locked and static",
            "the stabilized camera makes one short gentle push-in and settles",
            "the stabilized camera makes one short gentle pull-back and settles",
            "the stabilized camera makes one small pan left and settles",
            "the stabilized camera makes one small tilt up and settles",
            "the stabilized camera makes one short lateral track right and settles",
        )
        for clause in clauses:
            prompt = f"A medium shot at eye level frames a performer while {clause}."
            with self.subTest(clause=clause):
                report = self.validate(prompt, "stable")
                self.assertFalse(
                    any(
                        reason.startswith("ltx_stable_camera_")
                        for reason in report["reasons"]
                    ),
                    report,
                )

    def test_long_horizon_stable_rejects_sustained_travel_and_advanced_allows_it(self) -> None:
        prompt = (
            "A medium shot at eye level frames the performer as the camera tracks right "
            "throughout the shot and finally settles on the same subject."
        )
        stable = self.validate(prompt, "stable", 25.0, "on")
        advanced = self.validate(prompt, "advanced", 25.0, "on")
        self.assertIn("ltx_stable_camera_sustained_travel", stable["reasons"])
        self.assertNotIn("ltx_stable_camera_sustained_travel", advanced["reasons"])

    def test_low_level_legacy_defaults_remain_advanced_but_target_path_is_stable(self) -> None:
        prompt = (
            "A medium shot at eye level frames the performer while the camera orbits her."
        )
        legacy = self.contract.validate_ltx25_prompt(
            prompt,
            self.mode_report,
            {},
            12.0,
        )
        self.assertEqual(legacy["camera_capability"], "advanced")
        self.assertNotIn("ltx_stable_camera_orbit_or_rotation", legacy["reasons"])

        context = self.nodes.GemmaContext(user_prompt="A performer dances.")
        target = self.nodes._target_profile_config_to_dict({"target_profile": "ltx"})
        diagnostics = self.nodes._ltx25_contract_diagnostics(
            prompt,
            context,
            target,
            {},
        )
        self.assertEqual(diagnostics["camera_capability"], "stable")
        self.assertIn("ltx_stable_camera_orbit_or_rotation", diagnostics["reasons"])

    def test_stable_model_prompt_has_no_unconditional_advanced_camera_authorization(self) -> None:
        metadata = {
            "source": "image",
            "reference_image_count": 1,
            "ltx_first_frame_attached": True,
            "ltx_last_frame_attached": False,
        }
        for long_horizon_mode in ("off", "on"):
            stable = self.nodes._build_model_prompt(
                "A performer dances among loose bills.",
                self.nodes.DEFAULT_MASTER_PROMPT,
                "ltx",
                metadata,
                creativity_mode="wild",
                creative_strength=1.0,
                ltx_generation_mode="image_to_video",
                ltx_long_horizon_mode=long_horizon_mode,
                ltx_camera_capability="stable",
            ).casefold()
            with self.subTest(long_horizon_mode=long_horizon_mode):
                self.assertIn("camera capability is stable / base model", stable)
                self.assertIn("creativity-driven camera ambition", stable)
                self.assertNotIn("zooming out while rotating 180 degrees", stable)
                self.assertNotIn("may author mode-appropriate camera choreography", stable)
                self.assertNotIn("may be assertive, fast, or compound", stable)
                self.assertNotIn("fill the long middle with sustained travel, parallax", stable)
                self.assertNotIn("one dominant continuous camera-and-subject path", stable)

        advanced = self.nodes._build_model_prompt(
            "A performer dances among loose bills.",
            self.nodes.DEFAULT_MASTER_PROMPT,
            "ltx",
            metadata,
            creativity_mode="wild",
            creative_strength=1.0,
            ltx_generation_mode="image_to_video",
            ltx_long_horizon_mode="on",
            ltx_camera_capability="advanced",
        ).casefold()
        self.assertIn("zooming out while rotating 180 degrees", advanced)
        self.assertIn("fill the long middle with sustained travel, parallax", advanced)


if __name__ == "__main__":
    unittest.main()
