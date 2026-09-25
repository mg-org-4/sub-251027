from __future__ import annotations

import importlib.util
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_contract_module():
    """Load the pure contract without importing ComfyUI, Torch, or nodes.py."""

    module_name = f"diffusiongemma_ltx25_camera_contract_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "ltx25_contract.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the LTX-2.5 contract module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Ltx25CameraMotionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract_module()

    @staticmethod
    def _conditioned_metadata(mode: str) -> dict:
        first_last = mode == "first_last_frame"
        return {
            "source": "image",
            "ltx25_context_schema": "dg-ltx25-contract/1",
            "ltx_first_frame_attached": True,
            "ltx_last_frame_attached": first_last,
            "ltx_frame_pair_attached": first_last,
        }

    def _validate_conditioned(self, prompt: str, mode: str) -> dict:
        metadata = self._conditioned_metadata(mode)
        mode_report = self.contract.resolve_ltx25_generation_mode(mode, metadata, 2 if mode == "first_last_frame" else 1)
        return self.contract.validate_ltx25_prompt(prompt, mode_report, metadata, 15.0)

    def test_i2v_and_flf_still_hard_block_a_cut(self) -> None:
        prompt = (
            "A static medium shot at eye level holds on the subject. "
            "A hard cut to a locked wide shot at eye level reveals the street."
        )

        for mode in ("image_to_video", "first_last_frame"):
            with self.subTest(mode=mode):
                report = self._validate_conditioned(prompt, mode)
                self.assertFalse(report["ready"])
                self.assertIn(
                    "ltx_conditioned_mode_requires_single_continuous_take",
                    report["reasons"],
                )
                self.assertGreater(report["complexity"]["observed"]["cut_count"], 0)

    def test_compatible_camera_phases_do_not_block_conditioned_readiness(self) -> None:
        prompt = (
            "A medium shot at eye level follows one continuous take. The camera slowly pans right "
            "to follow the subject, smoothly dollies forward as the subject approaches a doorway, "
            "and gently tilts upward to finish on the sign above it."
        )

        for mode in ("image_to_video", "first_last_frame"):
            with self.subTest(mode=mode):
                report = self._validate_conditioned(prompt, mode)
                self.assertTrue(report["ready"], report["reasons"])
                self.assertNotIn(
                    "ltx_camera_complexity_exceeds_duration_budget",
                    report["reasons"],
                )

    def test_brisk_compound_i2v_camera_path_is_one_valid_continuous_take(self) -> None:
        prompt = (
            "From the exact supplied first-frame viewpoint, a medium shot follows the cyclist in "
            "one unbroken take. The camera rapidly trucks right beside her; then the camera "
            "orbits energetically as she rounds the corner; finally the camera cranes upward to finish in a "
            "high-angle wide composition over the same alley, with no cut and no added actor."
        )
        report = self._validate_conditioned(prompt, "image_to_video")

        self.assertTrue(report["ready"], report["reasons"])
        observed = report["complexity"]["observed"]
        self.assertEqual(observed["estimated_shots"], 1)
        self.assertEqual(observed["cut_count"], 0)
        self.assertGreaterEqual(observed["camera_motion_phase_count"], 3)
        self.assertTrue(
            {"track", "orbit", "vertical"}.issubset(
                {match["family"] for match in observed["camera_motion_matches"]}
            )
        )
        self.assertNotIn(
            "ltx_conditioned_mode_requires_single_continuous_take",
            report["reasons"],
        )
        self.assertNotIn(
            "ltx_camera_complexity_exceeds_duration_budget",
            report["reasons"],
        )

    def test_negated_camera_movements_count_as_zero_positive_moves(self) -> None:
        prompt = (
            "A medium shot at eye level remains static, with no pan, tilt, zoom, or dolly "
            "throughout the continuous take."
        )
        report = self.contract.analyze_ltx25_prompt_complexity(prompt, "image_to_video", 15.0)

        self.assertEqual(report["observed"]["distinct_camera_moves"], 0)
        self.assertEqual(report["observed"]["camera_motion_matches"], [])
        self.assertEqual(report["observed"]["negated_camera_move_count"], 4)
        self.assertNotIn("ltx_camera_complexity_exceeds_duration_budget", report["reasons"])

    def test_contracted_and_plain_negations_do_not_become_positive_motion(self) -> None:
        prompt = (
            "A medium shot at eye level remains locked off. The camera isn't panning, doesn't "
            "tilt, and cannot zoom during the continuous take."
        )
        report = self.contract.analyze_ltx25_prompt_complexity(prompt, "image_to_video", 15.0)

        self.assertEqual(report["observed"]["distinct_camera_moves"], 0)
        self.assertEqual(report["observed"]["camera_motion_matches"], [])
        self.assertGreaterEqual(report["observed"]["negated_camera_move_count"], 3)

    def test_subject_actions_and_scene_nouns_are_not_camera_movements(self) -> None:
        prompt = (
            "A medium shot at eye level stays steady as the astronaut tilts his head while a candy "
            "truck rolls past and a construction crane rotates overhead."
        )
        report = self.contract.analyze_ltx25_prompt_complexity(prompt, "image_to_video", 15.0)

        self.assertEqual(report["observed"]["distinct_camera_moves"], 0)
        self.assertEqual(report["observed"]["camera_motion_matches"], [])
        self.assertNotIn("ltx_camera_complexity_exceeds_duration_budget", report["reasons"])

    def test_pan_inflections_canonicalize_to_one_motion_family(self) -> None:
        prompt = (
            "A medium shot at eye level begins as the camera pans right, continues panning across "
            "the street, and then pans toward the doorway in one unbroken take."
        )
        report = self.contract.analyze_ltx25_prompt_complexity(prompt, "image_to_video", 15.0)
        matches = report["observed"]["camera_motion_matches"]

        self.assertEqual(report["observed"]["distinct_camera_moves"], 1)
        self.assertTrue(matches)
        self.assertEqual({match["family"] for match in matches}, {"pan"})

    def test_camera_support_systems_are_not_movement_families(self) -> None:
        prompt = (
            "A medium shot at eye level uses gimbal and Steadicam support while the camera remains "
            "steady throughout the continuous take."
        )
        report = self.contract.analyze_ltx25_prompt_complexity(prompt, "image_to_video", 15.0)

        self.assertEqual(report["observed"]["distinct_camera_moves"], 0)
        self.assertEqual(report["observed"]["camera_motion_matches"], [])
        self.assertNotIn("ltx_camera_complexity_exceeds_duration_budget", report["reasons"])

        support_only = self.contract.analyze_ltx25_prompt_complexity(
            "A medium shot at eye level uses gimbal and Steadicam support throughout.",
            "image_to_video",
            15.0,
        )
        self.assertIn("ltx_camera_state_missing", support_only["reasons"])

    def test_tracking_shot_and_generic_camera_move_are_recognized(self) -> None:
        prompt = (
            "A medium shot at eye level begins as a tracking shot, and then the camera moves "
            "forward toward the doorway in one continuous take."
        )
        report = self.contract.analyze_ltx25_prompt_complexity(prompt, "image_to_video", 15.0)

        self.assertEqual(
            {match["family"] for match in report["observed"]["camera_motion_matches"]},
            {"track", "move"},
        )
        self.assertNotIn("ltx_camera_state_missing", report["reasons"])

    def test_nominal_tracking_arc_is_a_continuous_camera_path(self) -> None:
        prompt = (
            "A medium shot starts from the exact supplied first-frame viewpoint. "
            "The camera begins a slow, handheld-style tracking arc around her, gaining kinetic "
            "energy as she starts to dance. As the camera decelerates, it settles into a tight "
            "close-up and holds."
        )
        report = self.contract.analyze_ltx25_prompt_complexity(
            prompt,
            "image_to_video",
            25.0,
            "auto",
        )

        self.assertIn(
            "tracking arc",
            {match["text"].casefold() for match in report["observed"]["camera_motion_matches"]},
        )
        self.assertNotIn("ltx_camera_state_missing", report["reasons"])

    def test_replayed_25_second_candidate_passes_camera_state_validation(self) -> None:
        prompt = (
            "A medium shot captures a woman with curly blonde hair, wearing a green floral print "
            "bikini, holding a green game controller aloft in a dimly lit, crowded party setting "
            "with harsh flash lighting. Starting from this anchored frame, she lets out a joyful "
            "shout as the muffled thumping of an alt-pop beat and ambient crowd chatter fills the "
            "room. The camera begins a slow, handheld-style tracking arc around her, gaining "
            "kinetic energy as she starts to dance rhythmically. Through most of the shot, she "
            "spins slowly, the game controller held steady in her raised hand while the background "
            "figures of partygoers blur in a bokeh of movement and light. The direct flash creates "
            "glints on her skin and deep shadows behind her, emphasizing the late90s aesthetic. "
            "As the camera decelerates, it settles into a tight close-up on her expressive face as "
            "she laughs and looks directly into the lens. The clip ends holding on this vibrant, "
            "high-energy expression as the music reaches a brief crescendo and the sound of a "
            "cheering crowd fades into a rhythmic hum."
        )
        metadata = self._conditioned_metadata("image_to_video")
        mode_report = self.contract.resolve_ltx25_generation_mode(
            "image_to_video",
            metadata,
            1,
        )
        report = self.contract.validate_ltx25_prompt(
            prompt,
            mode_report,
            metadata,
            25.0,
            "auto",
        )

        self.assertTrue(report["ready"], report["reasons"])
        self.assertNotIn("ltx_camera_state_missing", report["reasons"])

    def test_camera_pronoun_carries_across_a_compound_path(self) -> None:
        prompt = (
            "A medium shot at eye level follows the camera as it pans right, then tilts up, "
            "and slowly pulls back in one continuous take."
        )
        report = self.contract.analyze_ltx25_prompt_complexity(prompt, "image_to_video", 15.0)

        self.assertEqual(
            {match["family"] for match in report["observed"]["camera_motion_matches"]},
            {"pan", "tilt", "dolly"},
        )
        self.assertNotIn("ltx_camera_state_missing", report["reasons"])

    def test_diagnostics_expose_each_positive_match_family_and_source_span(self) -> None:
        prompt = (
            "A medium shot at eye level begins as the camera pans right and then smoothly dollies "
            "forward in one continuous take."
        )
        report = self.contract.analyze_ltx25_prompt_complexity(prompt, "image_to_video", 15.0)
        matches = report["observed"]["camera_motion_matches"]

        self.assertEqual({match["family"] for match in matches}, {"pan", "dolly"})
        for match in matches:
            self.assertIsInstance(match["text"], str)
            self.assertTrue(match["text"])
            self.assertEqual(len(match["span"]), 2)
            start, end = match["span"]
            self.assertIsInstance(start, int)
            self.assertIsInstance(end, int)
            self.assertLess(start, end)
            self.assertEqual(prompt[start:end], match["text"])

    def test_three_positive_phases_never_emit_the_obsolete_three_vs_one_hard_reason(self) -> None:
        prompt = (
            "A medium shot at eye level uses one continuous camera path: the camera pans right, "
            "dollies forward, and tilts upward without any cut."
        )
        report = self.contract.analyze_ltx25_prompt_complexity(prompt, "image_to_video", 15.0)

        self.assertEqual(report["observed"]["distinct_camera_moves"], 3)
        self.assertNotIn(
            "ltx_camera_complexity_exceeds_duration_budget",
            report["reasons"],
        )

    def test_equipment_and_capture_nouns_are_not_motion_or_camera_state(self) -> None:
        equipment_prompts = (
            "The camera uses rolling shutter capture for a medium shot at eye level.",
            "The camera uses a tracking marker for a medium shot at eye level.",
            "The camera uses a zoom lens for a medium shot at eye level.",
            "The camera uses a boom arm for a medium shot at eye level.",
            "The camera uses a dolly track for a medium shot at eye level.",
        )

        for prompt in equipment_prompts:
            with self.subTest(prompt=prompt):
                report = self.contract.analyze_ltx25_prompt_complexity(
                    prompt, "image_to_video", 15.0
                )
                self.assertEqual(report["observed"]["distinct_camera_moves"], 0)
                self.assertEqual(report["observed"]["distinct_camera_states"], 0)
                self.assertEqual(report["observed"]["camera_motion_matches"], [])
                self.assertIn("ltx_camera_state_missing", report["reasons"])

    def test_common_positive_camera_phrasings_satisfy_camera_state(self) -> None:
        motion_phrases = (
            "A gentle pan right follows the subject",
            "A slow push-in approaches the subject",
            "A panning camera frames the subject",
            "The camera follows the subject",
            "The camera glides beside the subject",
        )

        for motion_phrase in motion_phrases:
            with self.subTest(motion_phrase=motion_phrase):
                prompt = f"A medium shot at eye level holds the composition. {motion_phrase}."
                report = self.contract.analyze_ltx25_prompt_complexity(
                    prompt, "image_to_video", 15.0
                )
                self.assertGreater(report["observed"]["distinct_camera_moves"], 0)
                self.assertTrue(report["observed"]["camera_motion_matches"])
                self.assertNotIn("ltx_camera_state_missing", report["reasons"])

    def test_negated_static_states_do_not_satisfy_camera_requirement(self) -> None:
        negated_states = (
            "The camera is not static",
            "There is no static camera",
            "The shot proceeds without a locked-off camera",
            "The camera never stays steady",
        )

        for state_phrase in negated_states:
            with self.subTest(state_phrase=state_phrase):
                prompt = f"A medium shot at eye level shows the subject. {state_phrase}."
                report = self.contract.analyze_ltx25_prompt_complexity(
                    prompt, "image_to_video", 15.0
                )
                self.assertEqual(report["observed"]["distinct_camera_states"], 0)
                self.assertIn("ltx_camera_state_missing", report["reasons"])

    def test_another_subject_stays_steady_does_not_bind_to_the_camera(self) -> None:
        prompt = (
            "A medium shot at eye level shows a cyclist crossing the frame while another subject "
            "stays steady beside the road."
        )
        report = self.contract.analyze_ltx25_prompt_complexity(
            prompt, "image_to_video", 15.0
        )

        self.assertEqual(report["observed"]["distinct_camera_states"], 0)
        self.assertIn("ltx_camera_state_missing", report["reasons"])

    def test_camera_directions_do_not_bleed_and_zoom_in_reports_direction(self) -> None:
        prompt = (
            "A medium shot at eye level uses one continuous path as the camera pans left, then "
            "dollies forward and zooms in."
        )
        report = self.contract.analyze_ltx25_prompt_complexity(
            prompt, "image_to_video", 15.0
        )
        matches = report["observed"]["camera_motion_matches"]
        by_family = {match["family"]: match for match in matches}

        self.assertEqual(by_family["pan"]["direction"], "left")
        self.assertEqual(by_family["dolly"]["direction"], "forward")
        self.assertIn(by_family["zoom"]["direction"], {"in", "forward"})

    def test_conditioned_modes_block_transitions_that_create_a_new_shot(self) -> None:
        transition_phrases = (
            "A cut reveals",
            "After a cut, the image reveals",
            "A cross-dissolve reveals",
            "A wipe transitions to",
            "The image fades into a new",
        )

        for mode in ("image_to_video", "first_last_frame"):
            for transition_phrase in transition_phrases:
                with self.subTest(mode=mode, transition_phrase=transition_phrase):
                    prompt = (
                        "A static medium shot at eye level holds on the subject. "
                        f"{transition_phrase} locked wide shot at eye level showing the street."
                    )
                    report = self._validate_conditioned(prompt, mode)
                    self.assertFalse(report["ready"])
                    self.assertGreater(report["complexity"]["observed"]["cut_count"], 0)
                    self.assertIn(
                        "ltx_conditioned_mode_requires_single_continuous_take",
                        report["reasons"],
                    )

    def test_conditioned_modes_allow_terminal_fade_or_cut_to_black(self) -> None:
        terminal_transitions = (
            "then the image fades to black",
            "then the image cuts to black",
        )

        for mode in ("image_to_video", "first_last_frame"):
            for transition in terminal_transitions:
                with self.subTest(mode=mode, transition=transition):
                    prompt = (
                        "A static medium shot at eye level holds on the subject, "
                        f"{transition}."
                    )
                    report = self._validate_conditioned(prompt, mode)
                    self.assertTrue(report["ready"], report["reasons"])
                    self.assertEqual(report["complexity"]["observed"]["cut_count"], 0)
                    self.assertNotIn(
                        "ltx_conditioned_mode_requires_single_continuous_take",
                        report["reasons"],
                    )

    def test_negated_editing_instructions_do_not_create_phantom_cuts(self) -> None:
        negated_edits = (
            "with no cuts, wipes, or dissolves",
            "with no hard cuts",
            "and the image does not cut to another shot",
        )

        for mode in ("image_to_video", "first_last_frame"):
            for instruction in negated_edits:
                with self.subTest(mode=mode, instruction=instruction):
                    prompt = (
                        "A static medium shot at eye level holds on the subject "
                        f"{instruction}."
                    )
                    report = self._validate_conditioned(prompt, mode)
                    self.assertEqual(report["complexity"]["observed"]["cut_count"], 0)
                    self.assertNotIn(
                        "ltx_conditioned_mode_requires_single_continuous_take",
                        report["reasons"],
                    )

    def test_modified_negated_camera_motion_has_no_active_matches(self) -> None:
        negated_motion_phrases = (
            "The camera never slowly pans",
            "The camera doesn't gently tilt",
            "The shot proceeds without a slow pan",
            "The camera does not use a slow pan",
        )

        for motion_phrase in negated_motion_phrases:
            with self.subTest(motion_phrase=motion_phrase):
                prompt = f"A medium shot at eye level shows the subject. {motion_phrase}."
                report = self.contract.analyze_ltx25_prompt_complexity(
                    prompt, "image_to_video", 15.0
                )
                self.assertEqual(report["observed"]["distinct_camera_moves"], 0)
                self.assertEqual(report["observed"]["camera_motion_matches"], [])

    def test_physical_rig_and_optical_actions_are_not_camera_motion(self) -> None:
        prompts = (
            "A medium shot at eye level uses a moving platform beneath the actor.",
            "The camera lens uses a rotating polarizer for a medium shot at eye level.",
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                report = self.contract.analyze_ltx25_prompt_complexity(
                    prompt, "image_to_video", 15.0
                )
                self.assertEqual(report["observed"]["distinct_camera_moves"], 0)
                self.assertEqual(report["observed"]["camera_motion_matches"], [])
                self.assertIn("ltx_camera_state_missing", report["reasons"])

    def test_camera_uses_motion_grammar_remains_recognized(self) -> None:
        cases = (
            ("The camera uses a gentle pan right", "pan"),
            ("The camera uses a tracking shot beside the subject", "track"),
        )

        for motion_phrase, family in cases:
            with self.subTest(motion_phrase=motion_phrase):
                prompt = f"A medium shot at eye level shows the subject. {motion_phrase}."
                report = self.contract.analyze_ltx25_prompt_complexity(
                    prompt, "image_to_video", 15.0
                )
                matches = report["observed"]["camera_motion_matches"]
                self.assertIn(family, {match["family"] for match in matches})
                self.assertNotIn("ltx_camera_state_missing", report["reasons"])

    def test_explicit_no_camera_motion_counts_as_a_static_state(self) -> None:
        static_phrases = (
            "There is no camera movement",
            "There is no camera motion",
            "The camera does not move",
        )

        for static_phrase in static_phrases:
            with self.subTest(static_phrase=static_phrase):
                prompt = f"A medium shot at eye level shows the subject. {static_phrase}."
                report = self.contract.analyze_ltx25_prompt_complexity(
                    prompt, "image_to_video", 15.0
                )
                self.assertEqual(report["observed"]["distinct_camera_moves"], 0)
                self.assertGreater(report["observed"]["distinct_camera_states"], 0)
                self.assertNotIn("ltx_camera_state_missing", report["reasons"])

    def test_physical_cut_language_is_not_an_edit(self) -> None:
        physical_cut_phrases = (
            "The cut reveals blood beneath the skin",
            "A cut shows pale wood beneath the paint",
            "A bandage covers the cut to his hand",
            "After a cut blood beads on the skin",
            "Another cut reveals hidden wire in the cable",
            "She cuts to the left side of the fabric",
        )

        for physical_phrase in physical_cut_phrases:
            with self.subTest(physical_phrase=physical_phrase):
                prompt = (
                    "A static medium shot at eye level holds on the subject. "
                    f"{physical_phrase}."
                )
                report = self.contract.analyze_ltx25_prompt_complexity(
                    prompt, "image_to_video", 15.0
                )
                self.assertEqual(report["observed"]["cut_count"], 0)
                self.assertNotIn(
                    "ltx_conditioned_mode_requires_single_continuous_take",
                    report["reasons"],
                )

    def test_shot_bearing_cut_language_remains_an_edit(self) -> None:
        editorial_prompts = (
            (
                "A static medium shot at eye level holds on the subject. "
                "A cut reveals a locked wide shot at eye level showing the street."
            ),
            (
                "A static medium shot at eye level holds on the subject. "
                "After a cut, a wide shot at eye level shows the street with a locked camera."
            ),
        )

        for mode in ("image_to_video", "first_last_frame"):
            for prompt in editorial_prompts:
                with self.subTest(mode=mode, prompt=prompt):
                    report = self._validate_conditioned(prompt, mode)
                    self.assertGreater(report["complexity"]["observed"]["cut_count"], 0)
                    self.assertIn(
                        "ltx_conditioned_mode_requires_single_continuous_take",
                        report["reasons"],
                    )

    def test_editorial_cut_imperatives_block_conditioned_modes(self) -> None:
        editorial_prompts = (
            "A static medium shot at eye level holds on her. Then, cut to the street.",
            "A static medium shot at eye level holds on the street. Cut back to her.",
            (
                "A static medium shot at eye level holds on the subject. "
                "Cutting between the street and the room."
            ),
        )

        for mode in ("image_to_video", "first_last_frame"):
            for prompt in editorial_prompts:
                with self.subTest(mode=mode, prompt=prompt):
                    report = self._validate_conditioned(prompt, mode)
                    self.assertGreater(report["complexity"]["observed"]["cut_count"], 0)
                    self.assertIn(
                        "ltx_conditioned_mode_requires_single_continuous_take",
                        report["reasons"],
                    )

    def test_physical_cut_subjects_do_not_become_editorial_commands(self) -> None:
        physical_prompts = (
            (
                "A static medium shot at eye level shows the tailor as she cuts to the left side "
                "of fabric."
            ),
            (
                "A static medium shot at eye level shows scissors cut between two layers of "
                "fabric."
            ),
        )

        for prompt in physical_prompts:
            with self.subTest(prompt=prompt):
                report = self.contract.analyze_ltx25_prompt_complexity(
                    prompt, "image_to_video", 15.0
                )
                self.assertEqual(report["observed"]["cut_count"], 0)
                self.assertNotIn(
                    "ltx_conditioned_mode_requires_single_continuous_take",
                    report["reasons"],
                )

    def test_camera_control_verbs_do_not_bind_equipment_actions_as_motion(self) -> None:
        equipment_actions = (
            "The camera makes a moving platform wobble",
            "The camera executes a tracking calibration",
            "The camera performs a rotating filter calibration",
        )

        for action in equipment_actions:
            with self.subTest(action=action):
                prompt = f"A medium shot at eye level shows the subject. {action}."
                report = self.contract.analyze_ltx25_prompt_complexity(
                    prompt, "image_to_video", 15.0
                )
                self.assertEqual(report["observed"]["distinct_camera_moves"], 0)
                self.assertEqual(report["observed"]["camera_motion_matches"], [])
                self.assertIn("ltx_camera_state_missing", report["reasons"])

    def test_camera_control_verbs_preserve_real_motion(self) -> None:
        cases = (
            ("The camera performs a gentle pan right", "pan"),
            ("The camera executes a slow orbit around the subject", "orbit"),
            ("The camera makes a dolly-in toward the subject", "dolly"),
            ("A slowly rotating camera frames the subject", "roll"),
        )

        for action, family in cases:
            with self.subTest(action=action):
                prompt = f"A medium shot at eye level shows the subject. {action}."
                report = self.contract.analyze_ltx25_prompt_complexity(
                    prompt, "image_to_video", 15.0
                )
                matches = report["observed"]["camera_motion_matches"]
                self.assertIn(family, {match["family"] for match in matches})
                self.assertNotIn("ltx_camera_state_missing", report["reasons"])

    def test_not_only_coordination_keeps_both_camera_moves_positive(self) -> None:
        prompt = (
            "A medium shot at eye level shows the subject while the camera is not only panning "
            "but also tilting."
        )
        report = self.contract.analyze_ltx25_prompt_complexity(
            prompt, "image_to_video", 15.0
        )

        self.assertEqual(
            {match["family"] for match in report["observed"]["camera_motion_matches"]},
            {"pan", "tilt"},
        )
        self.assertEqual(
            report["observed"]["camera_diagnostics"]["negated_motion_mentions"],
            [],
        )

    def test_contrastive_negation_only_negates_the_first_camera_move(self) -> None:
        prompt = (
            "A medium shot at eye level shows the subject while the camera does not pan but instead "
            "tilts upward."
        )
        report = self.contract.analyze_ltx25_prompt_complexity(
            prompt, "image_to_video", 15.0
        )
        active_families = {
            match["family"] for match in report["observed"]["camera_motion_matches"]
        }
        negated_families = {
            match["family"]
            for match in report["observed"]["camera_diagnostics"]["negated_motion_mentions"]
        }

        self.assertEqual(active_families, {"tilt"})
        self.assertEqual(negated_families, {"pan"})
        self.assertNotIn("ltx_camera_state_missing", report["reasons"])

    def test_gerund_bearing_camera_negations_have_no_active_motion(self) -> None:
        negated_actions = (
            "The shot proceeds without using a slow pan",
            "The camera avoids using a gentle tilt",
            "The shot continues without executing a slow orbit",
            "The shot continues without making a slow camera move",
        )

        for action in negated_actions:
            with self.subTest(action=action):
                prompt = f"A medium shot at eye level shows the subject. {action}."
                report = self.contract.analyze_ltx25_prompt_complexity(
                    prompt, "image_to_video", 15.0
                )
                self.assertEqual(report["observed"]["distinct_camera_moves"], 0)
                self.assertEqual(report["observed"]["camera_motion_matches"], [])

    def test_article_bearing_camera_negations_have_no_active_motion(self) -> None:
        negated_actions = (
            "The shot proceeds without the camera slowly panning",
            "The shot proceeds without any camera gently tilting",
            "The shot proceeds without a camera slowly panning",
            "The shot proceeds without the camera performing a slow pan",
            "The shot proceeds without the camera using a slow pan",
        )

        for action in negated_actions:
            with self.subTest(action=action):
                prompt = f"A medium shot at eye level shows the subject. {action}."
                report = self.contract.analyze_ltx25_prompt_complexity(
                    prompt, "image_to_video", 15.0
                )
                self.assertEqual(report["observed"]["distinct_camera_moves"], 0)
                self.assertEqual(report["observed"]["camera_motion_matches"], [])

    def test_control_verb_camera_negations_are_classified_as_negated(self) -> None:
        negated_actions = (
            "The director does not allow the camera to pan",
            "The director does not let the camera pan",
            "The director never permits the camera to pan",
            "Do not let the camera pan",
            "The director forbids the camera from panning",
        )

        for action in negated_actions:
            with self.subTest(action=action):
                prompt = f"A medium shot at eye level shows the subject. {action}."
                report = self.contract.analyze_ltx25_prompt_complexity(
                    prompt, "image_to_video", 15.0
                )
                self.assertEqual(report["observed"]["distinct_camera_moves"], 0)
                self.assertEqual(report["observed"]["camera_motion_matches"], [])
                self.assertGreater(report["observed"]["negated_camera_move_count"], 0)
                self.assertEqual(
                    {
                        match["family"]
                        for match in report["observed"]["camera_diagnostics"][
                            "negated_motion_mentions"
                        ]
                    },
                    {"pan"},
                )

    def test_modified_control_verb_negations_keep_the_motion_negated(self) -> None:
        cases = (
            ("The director does not allow camera to slowly pan", "pan"),
            ("The director does not let camera slowly pan", "pan"),
            ("The director never permits camera to gently tilt", "tilt"),
            ("The shot proceeds without allowing camera to gently tilt", "tilt"),
        )

        for action, family in cases:
            with self.subTest(action=action):
                prompt = f"A medium shot at eye level shows the subject. {action}."
                report = self.contract.analyze_ltx25_prompt_complexity(
                    prompt, "image_to_video", 15.0
                )
                self.assertEqual(report["observed"]["distinct_camera_moves"], 0)
                self.assertEqual(report["observed"]["camera_motion_matches"], [])
                self.assertEqual(
                    {
                        match["family"]
                        for match in report["observed"]["camera_diagnostics"][
                            "negated_motion_mentions"
                        ]
                    },
                    {family},
                )

    def test_gerund_bearing_cut_negations_do_not_create_edits(self) -> None:
        negated_edits = (
            "without making a cut",
            "without using a hard cut",
            "and avoids making any cuts",
            "and does not make a cut",
        )

        for mode in ("image_to_video", "first_last_frame"):
            for instruction in negated_edits:
                with self.subTest(mode=mode, instruction=instruction):
                    prompt = (
                        "A static medium shot at eye level holds on the subject "
                        f"{instruction}."
                    )
                    report = self._validate_conditioned(prompt, mode)
                    self.assertEqual(report["complexity"]["observed"]["cut_count"], 0)
                    self.assertNotIn(
                        "ltx_conditioned_mode_requires_single_continuous_take",
                        report["reasons"],
                    )

    def test_yet_coordination_only_negates_the_first_camera_move(self) -> None:
        prompt = (
            "A medium shot at eye level shows the subject while the camera does not pan, yet tilts "
            "upward."
        )
        report = self.contract.analyze_ltx25_prompt_complexity(
            prompt, "image_to_video", 15.0
        )
        active_families = {
            match["family"] for match in report["observed"]["camera_motion_matches"]
        }
        negated_families = {
            match["family"]
            for match in report["observed"]["camera_diagnostics"]["negated_motion_mentions"]
        }

        self.assertEqual(active_families, {"tilt"})
        self.assertEqual(negated_families, {"pan"})
        self.assertNotIn("ltx_camera_state_missing", report["reasons"])


if __name__ == "__main__":
    unittest.main()
