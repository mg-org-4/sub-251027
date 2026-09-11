from __future__ import annotations

import importlib.util
import difflib
import json
import re
import sys
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]


def load_nodes_module():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    package_name = f"diffusiongemma_dialogue_{uuid.uuid4().hex}"
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


class MiniMaxH3DialogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()

    @staticmethod
    def target_profile_required_args() -> tuple:
        return (
            "minimax_h3",
            "auto_scene_audio",
            "",
            12.0,
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
            "integrated_multimodal_description: [Shot 1] A locked medium two-shot frames Mara (S1) on the left and "
            "Ivo (S2) on the right in a quiet train compartment. Mara (S1) looks directly at Ivo and says "
            "<d>[English] We leave at dawn.</d> while Ivo listens. "
            "[Shot 2] At 00:06.000, the camera cuts to a static close-up of Ivo (S2) as he replies "
            "<d>[English] Then pack light.</d> and closes the suitcase. The final frame holds on Ivo seated beside the "
            "closed suitcase in the static composition.\n\n"
            "overall_soundscape: Low train rumble, cloth movement, and the suitcase latch align with the visible "
            "actions without repeating speech.\n\n"
            "non_diegetic_music: N/A"
        )

    @staticmethod
    def one_shot_t2va_prompt() -> str:
        return (
            "integrated_multimodal_description: [Shot 1] A locked medium two-shot holds Mara (S1) on the left and "
            "Ivo (S2) on the right in a quiet train compartment. Mara (S1) says <d>[English] We leave at dawn.</d> "
            "and Ivo (S2) replies <d>[English] Then pack light.</d> before closing the suitcase. The final frame holds "
            "on both travelers beside the closed suitcase in the same locked composition.\n\n"
            "overall_soundscape: Low train rumble, cloth movement, and a synchronized suitcase-latch click.\n\n"
            "non_diegetic_music: N/A"
        )

    @staticmethod
    def reference_manifest() -> str:
        return (
            "<Picture 1>: [dg:identity,appearance,object,color] Mara identity, blue coat, and brass suitcase."
        )

    @staticmethod
    def ref2va_prompt() -> str:
        return (
            "subject_definitions:\n"
            "<Subject 1>: Mara, a traveler whose face, dark bobbed hair, blue coat, and brass suitcase derive from "
            "<Picture 1>.\n\n"
            "summary:\n"
            "[reference generation] <Subject 1> shares a decision before closing the suitcase.\n\n"
            "retention_analysis:\n"
            "<Subject 1>: fully_preserved - Her face, dark bobbed hair, blue coat, proportions, and brass suitcase "
            "remain stable in both shots.\n\n"
            "detailed_description:\n"
            "Realistic live-action practical film photography uses soft window light, restrained blue-brown color, "
            "shallow depth of field, and fine grain throughout. [Shot 1] A locked medium shot holds <Subject 1> (S1) "
            "on the left side of a quiet train compartment, matching the face, dark bobbed hair, blue coat, and brass "
            "suitcase derived from <Picture 1>. Passing dawn light slides across the wall while she keeps both hands on "
            "the suitcase. <Subject 1> (S1) looks toward an unseen companion and says <d>[English] We leave at "
            "dawn.</d> in a steady voice. The camera remains static while her mouth forms the words and her right hand "
            "moves to the latch. Low train vibration subtly moves the hanging curtain without changing her position or "
            "identity. [Shot 2] At 00:06.000, the camera cuts to a static close-up of <Subject 1> (S1) beside the brass "
            "latch. She answers her own uncertainty with <d>[English] Then we are ready.</d> and closes the suitcase in "
            "one visible motion. The latch clicks only when her thumb presses it; her blue sleeve settles against the "
            "brass edge. The camera holds her resolved expression and the same soft window light as the train continues "
            "forward. The final frame holds on <Subject 1> seated calmly beside the closed suitcase in the static close "
            "composition.\n\n"
            "overall_soundscape:\n"
            "Low train rumble, curtain movement, cloth rustle, and the synchronized suitcase-latch click surround the "
            "compartment without repeating speech.\n\n"
            "non_diegetic_music:\n"
            "N/A"
        )

    def validation_reasons(
        self,
        prompt: str,
        *,
        mode: str,
        line_count: int = 2,
        h3_mode: str = "t2va",
        audio_mode: str = "auto_scene_audio",
        shot_count: str = "2",
        dialogue_guidance: str = "",
    ) -> list[str]:
        return self.nodes._minimax_h3_prompt_validation_reasons(
            prompt,
            duration_seconds=12.0,
            user_prompt="Create exactly two shots in a quiet train compartment.",
            audio_mode=audio_mode,
            minimax_h3_mode=h3_mode,
            reference_manifest=self.reference_manifest() if h3_mode == "ref2va" else "",
            minimax_h3_shot_count=shot_count,
            minimax_h3_dialogue_mode=mode,
            minimax_h3_dialogue_line_count=line_count,
            minimax_h3_dialogue_guidance=dialogue_guidance,
        )

    def test_ui_contract_appends_dialogue_controls_without_shifting_existing_widgets(self) -> None:
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
        choices, mode_config = optional["minimax_h3_dialogue_mode"]
        self.assertEqual(choices, ["auto", "required", "off"])
        self.assertEqual(mode_config["default"], "auto")
        count_type, count_config = optional["minimax_h3_dialogue_line_count"]
        self.assertEqual(count_type, "INT")
        self.assertEqual(count_config["default"], 2)
        self.assertEqual(count_config["min"], 1)
        self.assertEqual(count_config["max"], 12)
        guidance_type, guidance_config = optional["minimax_h3_dialogue_guidance"]
        self.assertEqual(guidance_type, "STRING")
        self.assertTrue(guidance_config["multiline"])

    def test_legacy_target_profile_calls_default_to_auto_dialogue(self) -> None:
        profile_node = self.nodes.DiffusionGemmaTargetProfile()
        legacy_config, legacy_json = profile_node.build(*self.target_profile_required_args())
        legacy_data = json.loads(legacy_json)
        self.assertEqual(legacy_config.minimax_h3_dialogue_mode, "auto")
        self.assertEqual(legacy_config.minimax_h3_dialogue_line_count, 2)
        self.assertEqual(legacy_config.minimax_h3_dialogue_guidance, "")
        self.assertEqual(legacy_data["minimax_h3_dialogue_mode"], "auto")
        self.assertEqual(legacy_data["minimax_h3_dialogue_line_count"], 2)

        configured, configured_json = profile_node.build(
            *self.target_profile_required_args(),
            minimax_h3_mode="ref2va",
            minimax_h3_shot_count="3",
            minimax_h3_custom_shot_count=12,
            minimax_h3_dialogue_mode="required",
            minimax_h3_dialogue_line_count=3,
            minimax_h3_dialogue_guidance="Mara: We leave at dawn.",
        )
        configured_data = json.loads(configured_json)
        self.assertEqual(configured.minimax_h3_dialogue_mode, "required")
        self.assertEqual(configured.minimax_h3_dialogue_line_count, 3)
        self.assertEqual(configured.minimax_h3_dialogue_guidance, "Mara: We leave at dawn.")
        self.assertEqual(configured_data["minimax_h3_dialogue_mode"], "required")
        self.assertEqual(configured_data["minimax_h3_dialogue_line_count"], 3)

    def test_required_dialogue_controls_reach_compiler_and_refinement_prompts(self) -> None:
        guidance = "Mara says exactly: We leave at dawn. Ivo answers: Then pack light."
        model_prompt = self.nodes._build_model_prompt(
            "Create a quiet train-compartment scene.",
            self.nodes.DEFAULT_MASTER_PROMPT,
            "minimax_h3",
            {"source": "none"},
            target_duration_seconds=12.0,
            minimax_h3_mode="t2va",
            minimax_h3_shot_count="2",
            minimax_h3_dialogue_mode="required",
            minimax_h3_dialogue_line_count=2,
            minimax_h3_dialogue_guidance=guidance,
        )
        self.assertIn('"minimax_h3_dialogue_mode": "required"', model_prompt)
        self.assertIn('"minimax_h3_dialogue_line_count": 2', model_prompt)
        self.assertIn(guidance, model_prompt)
        self.assertRegex(model_prompt, r"(?is)exactly\s+2.+?<d>")

        refinement_prompt = self.nodes._build_minimax_h3_refinement_prompt(
            "Create a quiet train-compartment scene.",
            self.t2va_prompt().replace("<d>", "", 1).replace("</d>", "", 1),
            12.0,
            "auto_scene_audio",
            ["minimax_h3_dialogue_count_mismatch"],
            minimax_h3_mode="t2va",
            minimax_h3_shot_count="2",
            minimax_h3_dialogue_mode="required",
            minimax_h3_dialogue_line_count=2,
            minimax_h3_dialogue_guidance=guidance,
        )
        self.assertRegex(
            refinement_prompt,
            r"(?i)Dialogue mode:\s*required;\s*required line count:\s*2",
        )
        self.assertIn(guidance, refinement_prompt)
        self.assertRegex(refinement_prompt, r"(?is)exactly\s+2.+?<d>")

    def test_dialogue_off_does_not_override_the_separate_music_vocal_contract(self) -> None:
        instruction = self.nodes._minimax_h3_dialogue_instruction(
            "off",
            minimax_h3_mode="ref2va",
        )
        self.assertIn("Spoken dialogue is disabled", instruction)
        self.assertIn("governed separately by the music-performance contract", instruction)
        self.assertNotIn("or singing", instruction)
        self.assertNotIn("no singing", instruction.casefold())

    def test_ref2va_required_dialogue_repeats_explicit_subject_speaker_binding(self) -> None:
        model_prompt = self.nodes._build_model_prompt(
            "Create a two-image reference sequence with two spoken lines.",
            self.nodes.DEFAULT_MASTER_PROMPT,
            "minimax_h3",
            {
                "source": "images",
                "verified_grounding_ledger": {"observed_facts": []},
                "grounding_evidence_report_id": "dggr-dialogue-instruction-test",
            },
            target_duration_seconds=12.0,
            minimax_h3_mode="ref2va",
            minimax_h3_reference_manifest=self.reference_manifest(),
            minimax_h3_shot_count="2",
            native_h3_output=True,
            minimax_h3_dialogue_mode="required",
            minimax_h3_dialogue_line_count=2,
        )
        self.assertIn("<Subject 1> (S1)", model_prompt)
        self.assertIn("repeat the speaking '<Subject N> (S1)' pair", model_prompt)
        self.assertIn("never output literal (Sx)", model_prompt)
        self.assertIn(
            "never output literal (Sx) or use a bare numbered cue after multiple Subject tags",
            model_prompt,
        )

        refinement_prompt = self.nodes._build_minimax_h3_refinement_prompt(
            "Create a two-image reference sequence with two spoken lines.",
            self.ref2va_prompt().replace("<Subject 1> (S1) looks", "<Subject 1> looks", 1),
            12.0,
            "auto_scene_audio",
            ["minimax_h3_dialogue_speaker_invalid"],
            minimax_h3_mode="ref2va",
            reference_manifest=self.reference_manifest(),
            minimax_h3_shot_count="2",
            minimax_h3_dialogue_mode="required",
            minimax_h3_dialogue_line_count=2,
            native_prompt_only=True,
        )
        self.assertIn("'<Subject N> (S1)'", refinement_prompt)
        self.assertIn("never output literal (Sx)", refinement_prompt)
        self.assertIn("repeat that pair for every line", refinement_prompt)
        self.assertIn(
            "never output literal (Sx) or use a bare numbered cue after multiple Subject tags",
            refinement_prompt,
        )

    def test_t2va_required_off_and_auto_modes(self) -> None:
        prompt = self.t2va_prompt()
        required_reasons = self.validation_reasons(prompt, mode="required")
        self.assertNotIn("minimax_h3_dialogue_count_mismatch", required_reasons)
        self.assertNotIn("minimax_h3_dialogue_forbidden", required_reasons)
        self.assertNotIn("minimax_h3_dialogue_audio_mode_conflict", required_reasons)

        missing_one = prompt.replace(
            "<d>[English] Then pack light.</d>",
            "the requested reply",
        )
        self.assertIn(
            "minimax_h3_dialogue_count_mismatch",
            self.validation_reasons(missing_one, mode="required"),
        )
        self.assertIn(
            "minimax_h3_dialogue_forbidden",
            self.validation_reasons(prompt, mode="off"),
        )

        # Auto deliberately preserves the legacy acceptance contract.
        auto_reasons = self.validation_reasons(prompt, mode="auto")
        self.assertNotIn("minimax_h3_dialogue_count_mismatch", auto_reasons)
        self.assertNotIn("minimax_h3_dialogue_forbidden", auto_reasons)

    def test_required_dialogue_is_counted_only_inside_the_timeline(self) -> None:
        prompt = self.t2va_prompt()
        without_timeline_dialogue = re.sub(r"<d>.*?</d>", "the requested line", prompt)
        outside = without_timeline_dialogue.replace(
            "without repeating speech.",
            "with <d>[English] We leave at dawn.</d> and <d>[English] Then pack light.</d>.",
        )
        reasons = self.validation_reasons(outside, mode="required")
        self.assertIn("minimax_h3_dialogue_outside_timeline", reasons)
        self.assertNotIn("minimax_h3_dialogue_count_mismatch", reasons)

    def test_required_dialogue_does_not_add_or_require_extra_shots(self) -> None:
        reasons = self.validation_reasons(
            self.one_shot_t2va_prompt(),
            mode="required",
            line_count=2,
            shot_count="1",
        )
        self.assertNotIn("minimax_h3_dialogue_count_mismatch", reasons)
        self.assertNotIn("minimax_h3_requested_shot_count_mismatch", reasons)

    def test_required_dialogue_conflicts_with_visual_only_even_when_output_is_silent(self) -> None:
        silent = re.sub(r"<d>.*?</d>", "a silent look", self.t2va_prompt())
        silent = re.sub(
            r"overall_soundscape:.*?\n\nnon_diegetic_music:.*$",
            "overall_soundscape: N/A\n\nnon_diegetic_music: N/A",
            silent,
            flags=re.DOTALL,
        )
        reasons = self.validation_reasons(
            silent,
            mode="required",
            audio_mode="visual_only",
        )
        self.assertIn("minimax_h3_dialogue_audio_mode_conflict", reasons)

        off_reasons = self.validation_reasons(
            silent,
            mode="off",
            audio_mode="visual_only",
        )
        self.assertNotIn("minimax_h3_dialogue_audio_mode_conflict", off_reasons)
        self.assertNotIn("minimax_h3_dialogue_forbidden", off_reasons)

    def test_ref2va_dialogue_is_validated_in_detailed_description(self) -> None:
        prompt = self.ref2va_prompt()
        required_reasons = self.validation_reasons(
            prompt,
            mode="required",
            h3_mode="ref2va",
        )
        self.assertNotIn("minimax_h3_dialogue_count_mismatch", required_reasons)
        self.assertNotIn("minimax_h3_dialogue_forbidden", required_reasons)
        self.assertNotIn("minimax_h3_dialogue_speaker_invalid", required_reasons)

        self.assertIn(
            "minimax_h3_dialogue_count_mismatch",
            self.validation_reasons(
                prompt,
                mode="required",
                line_count=3,
                h3_mode="ref2va",
            ),
        )
        self.assertIn(
            "minimax_h3_dialogue_forbidden",
            self.validation_reasons(prompt, mode="off", h3_mode="ref2va"),
        )

    def test_ref2va_analysis_section_dialogue_does_not_satisfy_required_count(self) -> None:
        prompt = re.sub(r"<d>.*?</d>", "the requested line", self.ref2va_prompt())
        prompt = prompt.replace(
            "[reference generation] <Subject 1> shares a decision before closing the suitcase.",
            "[reference generation] <Subject 1> says <d>[English] We leave at dawn.</d> and "
            "<d>[English] Then we are ready.</d> before closing the suitcase.",
        )
        reasons = self.validation_reasons(
            prompt,
            mode="required",
            h3_mode="ref2va",
        )
        self.assertIn("minimax_h3_dialogue_count_mismatch", reasons)
        self.assertIn("minimax_h3_dialogue_outside_timeline", reasons)

    def test_dialogue_transport_repairs_only_unambiguous_variants_in_both_modes(self) -> None:
        for h3_mode, canonical in (
            ("t2va", self.t2va_prompt()),
            ("ref2va", self.ref2va_prompt()),
        ):
            variants = {
                "case_and_spacing": canonical.replace("<d>", "< D >  ", 1).replace(
                    "</d>", "< / D >", 1
                ),
                "self_closer": canonical.replace("</d>", "<d/>", 1),
                "repeated_open": canonical.replace("</d>", "<d>", 1),
            }
            for variant_name, variant in variants.items():
                with self.subTest(h3_mode=h3_mode, variant=variant_name):
                    repaired, repairs = self.nodes._repair_minimax_h3_dialogue_transport(
                        variant,
                        h3_mode,
                    )
                    self.assertEqual(repaired, canonical)
                    self.assertTrue(repairs)
                    repaired_again, second_repairs = (
                        self.nodes._repair_minimax_h3_dialogue_transport(
                            repaired,
                            h3_mode,
                        )
                    )
                    self.assertEqual(repaired_again, canonical)
                    self.assertEqual(second_repairs, [])

    def test_t2va_dialogue_transport_repairs_live_lang_attribute_and_duplicate_close(self) -> None:
        canonical = self.one_shot_t2va_prompt()
        live_candidate = canonical.replace(
            "<d>[English]",
            '<d lang="English">',
        ).replace("</d>", "</</d>")

        normalized = self.nodes._normalize_minimax_h3_prompt(live_candidate, "t2va")
        contract_repaired, contract_repairs = (
            self.nodes._repair_minimax_h3_t2va_contract_transport(
                normalized,
                "t2va",
            )
        )
        repaired, repairs = self.nodes._repair_minimax_h3_dialogue_transport(
            contract_repaired,
            "t2va",
        )

        self.assertEqual(contract_repairs, [])
        self.assertEqual(repaired, canonical)
        self.assertEqual(
            repairs,
            [
                "canonicalized_dialogue_language_attribute",
                "canonicalized_dialogue_duplicate_prefix_close",
            ],
        )
        self.assertEqual(repaired.count("(S1)"), canonical.count("(S1)"))
        self.assertEqual(repaired.count("(S2)"), canonical.count("(S2)"))
        self.assertEqual(
            self.validation_reasons(repaired, mode="required", shot_count="1"),
            [],
        )

        repaired_again, second_repairs = self.nodes._repair_minimax_h3_dialogue_transport(
            repaired,
            "t2va",
        )
        self.assertEqual(repaired_again, canonical)
        self.assertEqual(second_repairs, [])

    def test_dialogue_transport_rejects_conflicting_attribute_and_body_languages(self) -> None:
        canonical = self.one_shot_t2va_prompt()
        ambiguous = canonical.replace(
            "<d>[English]",
            '<d lang="English">[French]',
            1,
        )

        repaired, repairs = self.nodes._repair_minimax_h3_dialogue_transport(
            ambiguous,
            "t2va",
        )

        self.assertEqual(repaired, ambiguous)
        self.assertEqual(repairs, [])

    def test_dialogue_transport_leaves_ambiguous_unbalanced_and_nested_markup_unchanged(self) -> None:
        for h3_mode, canonical in (
            ("t2va", self.t2va_prompt()),
            ("ref2va", self.ref2va_prompt()),
        ):
            first_block = re.search(r"<d>.*?</d>", canonical)
            self.assertIsNotNone(first_block)
            assert first_block is not None
            first_text = first_block.group(0)
            ambiguous = {
                "orphan_close": canonical.replace("<d>", "</d>", 1),
                "opening_self_closer": canonical.replace("<d>", "<d/>", 1),
                "unbalanced": canonical.replace("</d>", "", 1),
                "nested": canonical.replace(
                    first_text,
                    "<d>[English] We <d>[English] should not nest.</d></d>",
                    1,
                ),
            }
            for variant_name, variant in ambiguous.items():
                with self.subTest(h3_mode=h3_mode, variant=variant_name):
                    repaired, repairs = self.nodes._repair_minimax_h3_dialogue_transport(
                        variant,
                        h3_mode,
                    )
                    self.assertEqual(repaired, variant)
                    self.assertEqual(repairs, [])

    def test_dialogue_transport_never_changes_non_timeline_bytes(self) -> None:
        for h3_mode, canonical in (
            ("t2va", self.t2va_prompt()),
            ("ref2va", self.ref2va_prompt()),
        ):
            variant = canonical.replace("<d>", "&lt; D &gt;  ", 1).replace(
                "</d>", "&lt; / D &gt;", 1
            )
            variant = variant.replace(
                "without repeating speech.",
                "without repeating speech; &lt;D&gt;[English] outside&lt;/D&gt; remains literal.",
            )
            original_bounds = self.nodes._minimax_h3_timeline_bounds(variant, h3_mode)
            self.assertIsNotNone(original_bounds)
            assert original_bounds is not None

            repaired, repairs = self.nodes._repair_minimax_h3_dialogue_transport(
                variant,
                h3_mode,
            )
            repaired_bounds = self.nodes._minimax_h3_timeline_bounds(repaired, h3_mode)
            self.assertIsNotNone(repaired_bounds)
            assert repaired_bounds is not None
            self.assertTrue(repairs)
            self.assertEqual(
                variant[: original_bounds[0]],
                repaired[: repaired_bounds[0]],
            )
            self.assertEqual(
                variant[original_bounds[1] :],
                repaired[repaired_bounds[1] :],
            )
            self.assertIn(
                "&lt;D&gt;[English] outside&lt;/D&gt; remains literal.",
                repaired,
            )

    def test_invalid_dialogue_structure_suppresses_secondary_count_mismatch(self) -> None:
        for h3_mode, canonical in (
            ("t2va", self.t2va_prompt()),
            ("ref2va", self.ref2va_prompt()),
        ):
            nested = canonical.replace(
                "<d>[English] We leave at dawn.</d>",
                "<d>[English] We <d>[English] should not nest.</d></d>",
                1,
            )
            repaired, repairs = self.nodes._repair_minimax_h3_dialogue_transport(
                nested,
                h3_mode,
            )
            self.assertEqual(repaired, nested)
            self.assertEqual(repairs, [])
            reasons = self.validation_reasons(
                repaired,
                mode="required",
                h3_mode=h3_mode,
            )
            self.assertIn("minimax_h3_dialogue_tags_invalid", reasons)
            self.assertNotIn("minimax_h3_dialogue_count_mismatch", reasons)

    def test_t2va_contract_transport_repairs_the_attached_failure_without_rewriting_it(self) -> None:
        brief = (
            "2d anime of a young girl holding a teddy bear surrounded by enemies.  She states: "
            "\"maybe you haven't met my friend yet...\"  The camera cuts to a her enemies - they are laughing "
            "at the girl.  Smash Cut to a wide shot of the same girl holding her teddy bear - she screams: "
            "\"TEDDY BEAR ACTIVATE\" as she throws him up into the sky.  Just then, the teddy bear morphs in "
            "mid-air into a teddy bear-battle cyborg and lands heavy on it's metallic feet just behind the girl. "
            "No subtitles or words."
        )
        candidate = (
            "integrated_multimodal_description: [Shot 1] 2D hand-drawn cel animation, preserve drawn linework, "
            "cel shading, stylized proportions, and design continuity across every cut, no live action, "
            "photorealism, or 3D CGI. A young girl with determined eyes stands in a dark urban alleyway, "
            "clutching a tattered teddy bear to her chest. Shadowy, menacing enemies surround her in a circle. "
            "The girl looks up with a slight smirk and says (Sx: Girl) <d>English>maybe you haven't met my friend "
            "yet...</d>. The camera performs a slow dramatic zoom-in on her face.\n"
            "[Shot 2] At 00:03.500, the camera cuts to a medium shot of the enemies; they are jagged figures with "
            "glowing eyes, laughing hysterically and pointing at the small girl. The camera uses a handheld-style "
            "shake to emphasize their chaotic energy.\n"
            "[Shot 3] At 00:06.000, the camera cuts to a wide shot of the girl in the center of the frame. She "
            "raises the teddy bear high and screams (Sx: Girl) <d>English>TEDDY BEAR ACTIVATE</d> as she flurls the "
            "toy into the night sky. The camera tilts upward rapidly to follow the bear's arc.\n"
            "[Shot 4] At 00:07.500, the teddy bear morphs in mid-air, unfolding into mechanical plates and glowing "
            "blue circuits, transforming into a massive metallic teddy bear-battle cyborg. The cyborg descends and "
            "lands with a heavy metallic thud on its feet directly behind the girl, causing the ground to crack. "
            "The final frame holds on a wide shot of the girl standing confidently in front of the towering cyborg "
            "as the enemies recoil.\n\n"
            "overall_soundscape: Wind whistling through the alley, the mocking laughter of the enemies, the girl's "
            "defiant scream, mechanical whirring and clicking during transformation, and a thunderous metallic "
            "impact as the cyborg hits the ground.\n\n"
            "non_diegetic_music: A tense orchestral swell that builds in intensity, culminating in a heavy, "
            "cinematic brass drop when the bear transforms and lands."
        )
        normalized = self.nodes._normalize_minimax_h3_prompt(candidate, "t2va")
        repaired, repairs = self.nodes._repair_minimax_h3_t2va_contract_transport(
            normalized,
            "t2va",
        )
        repaired, dialogue_repairs = self.nodes._repair_minimax_h3_dialogue_transport(
            repaired,
            "t2va",
        )

        self.assertEqual(
            set(repairs),
            {
                "canonicalized_timed_cut_openers",
                "restored_dialogue_language_brackets",
                "canonicalized_dialogue_speaker_ids",
            },
        )
        self.assertEqual(dialogue_repairs, [])
        self.assertEqual(repaired.count("(S1)"), 2)
        self.assertNotIn("(Sx", repaired)
        self.assertIn("<d>[English]maybe you haven't met my friend yet...</d>", repaired)
        self.assertIn(
            "[Shot 4] At 00:07.500, the shot cuts to: the teddy bear morphs",
            repaired,
        )
        self.assertIn("teddy bear-battle cyborg", repaired)
        self.assertFalse(self.nodes._minimax_h3_battle_intent_requested(brief))
        self.assertEqual(
            self.nodes._minimax_h3_prompt_validation_reasons(
                repaired,
                10.0,
                brief,
            ),
            [],
        )

        repaired_again, second_repairs = (
            self.nodes._repair_minimax_h3_t2va_contract_transport(repaired, "t2va")
        )
        self.assertEqual(repaired_again, repaired)
        self.assertEqual(second_repairs, [])

    def test_t2va_contract_transport_leaves_ambiguous_bare_speakers_unchanged(self) -> None:
        ambiguous = self.one_shot_t2va_prompt().replace("(S1)", "(Sx)").replace(
            "(S2)",
            "(Sx)",
        )
        repaired, repairs = self.nodes._repair_minimax_h3_t2va_contract_transport(
            ambiguous,
            "t2va",
        )
        self.assertEqual(repaired, ambiguous)
        self.assertEqual(repairs, [])

    def test_t2va_contract_transport_never_rewrites_cue_like_spoken_text(self) -> None:
        prompt = self.one_shot_t2va_prompt().replace(
            "We leave at dawn.",
            "The literal token (Sx: Stage note) stays in my words.",
        ).replace(
            "and Ivo (S2) replies",
            "and Ivo replies",
        )
        repaired, _repairs = self.nodes._repair_minimax_h3_t2va_contract_transport(
            prompt,
            "t2va",
        )
        self.assertIn("<d>[English] The literal token (Sx: Stage note) stays", repaired)

    def test_timed_smash_cut_alias_is_canonical_and_idempotent(self) -> None:
        timeline = (
            "[Shot 1] A locked wide shot holds. "
            "[Shot 2] At 00:04.000, SMASH CUT TO: A close-up reveals the switch."
        )
        repaired = self.nodes._repair_minimax_h3_timed_cut_openers(timeline)
        self.assertIn(
            "[Shot 2] At 00:04.000, the camera hard cuts to A close-up",
            repaired,
        )
        self.assertEqual(
            self.nodes._repair_minimax_h3_timed_cut_openers(repaired),
            repaired,
        )

    def test_live_t2va_smash_cut_without_comma_is_canonical_and_idempotent(self) -> None:
        timeline = (
            "[Shot 1] A slow zoom holds on the girl. "
            "[Shot 2] At 00:06.000 the smash cuts to a wide shot as she throws the bear."
        )
        repaired = self.nodes._repair_minimax_h3_timed_cut_openers(timeline)
        self.assertIn(
            "[Shot 2] At 00:06.000, the camera hard cuts to a wide shot",
            repaired,
        )
        self.assertEqual(
            self.nodes._repair_minimax_h3_timed_cut_openers(repaired),
            repaired,
        )

    def test_battle_intent_ignores_character_labels_but_keeps_real_combat(self) -> None:
        self.assertFalse(
            self.nodes._minimax_h3_battle_intent_requested(
                "A teddy bear-battle cyborg lands behind the girl."
            )
        )
        self.assertFalse(
            self.nodes._minimax_h3_battle_intent_requested(
                "A portrait introduces the battle cyborg design."
            )
        )
        self.assertTrue(
            self.nodes._minimax_h3_battle_intent_requested(
                "Two knights battle in the courtyard."
            )
        )

    @staticmethod
    def dialogue_patch_payload(records: list[dict], **extra: object) -> str:
        return json.dumps(
            {
                "schema": "dg-h3-dialogue-patch/1",
                "dialogue_patch": records,
                **extra,
            },
            separators=(",", ":"),
        )

    @staticmethod
    def t2va_patch_records() -> list[dict]:
        return [
            {
                "shot": 1,
                "subject_tag": "",
                "speaker_id": "S1",
                "language": "English",
                "delivery": "quiet awe",
                "text": "The clouds are moving.",
            },
            {
                "shot": 2,
                "subject_tag": "",
                "speaker_id": "S2",
                "language": "English",
                "delivery": "firm resolve",
                "text": "Then pack light.",
            },
        ]

    @staticmethod
    def ref2va_patch_records() -> list[dict]:
        return [
            {
                "shot": 1,
                "subject_tag": "<Subject 1>",
                "speaker_id": "S1",
                "language": "English",
                "delivery": "quiet awe",
                "text": "The clouds are moving.",
            },
            {
                "shot": 2,
                "subject_tag": "<Subject 1>",
                "speaker_id": "S1",
                "language": "English",
                "delivery": "firm resolve",
                "text": "Then we are ready.",
            },
        ]

    def test_strict_dialogue_patch_validation_accepts_t2va_and_ref2va_records(self) -> None:
        cases = (
            (
                "t2va",
                re.sub(r"<d>.*?</d>", "the requested line", self.t2va_prompt()),
                self.t2va_patch_records(),
            ),
            (
                "ref2va",
                re.sub(r"<d>.*?</d>", "the requested line", self.ref2va_prompt()),
                self.ref2va_patch_records(),
            ),
        )
        for h3_mode, candidate, expected_records in cases:
            with self.subTest(h3_mode=h3_mode):
                records, reasons = self.nodes._validated_minimax_h3_dialogue_patch_records(
                    self.dialogue_patch_payload(expected_records),
                    candidate,
                    h3_mode,
                    len(expected_records),
                )
                self.assertEqual(reasons, [])
                self.assertEqual(records, expected_records)

    def test_strict_dialogue_patch_validation_rejects_unsafe_payloads(self) -> None:
        t2va_candidate = re.sub(
            r"<d>.*?</d>",
            "the requested line",
            self.t2va_prompt(),
        )
        valid_record = self.t2va_patch_records()[0]
        invalid_cases = (
            (
                "extra_top_level_key",
                self.dialogue_patch_payload([valid_record], extra=True),
                t2va_candidate,
                "t2va",
                1,
                "dialogue_patch_schema_invalid",
            ),
            (
                "extra_record_key",
                self.dialogue_patch_payload([{**valid_record, "camera": "pan"}]),
                t2va_candidate,
                "t2va",
                1,
                "dialogue_patch_record_shape_invalid",
            ),
            (
                "duplicate_json_key",
                '{"schema":"dg-h3-dialogue-patch/1","schema":"dg-h3-dialogue-patch/1","dialogue_patch":[]}',
                t2va_candidate,
                "t2va",
                0,
                "dialogue_patch_not_strict_json",
            ),
            (
                "duplicate_record",
                self.dialogue_patch_payload([valid_record, dict(valid_record)]),
                t2va_candidate,
                "t2va",
                2,
                "dialogue_patch_duplicate_record",
            ),
            (
                "out_of_range_shot",
                self.dialogue_patch_payload([{**valid_record, "shot": 99}]),
                t2va_candidate,
                "t2va",
                1,
                "dialogue_patch_shot_invalid",
            ),
            (
                "injected_shot_marker",
                self.dialogue_patch_payload(
                    [{**valid_record, "text": "Wait [Shot 9] now."}]
                ),
                t2va_candidate,
                "t2va",
                1,
                "dialogue_patch_text_invalid",
            ),
            (
                "injected_control_content",
                self.dialogue_patch_payload(
                    [
                        {
                            **valid_record,
                            "text": "GROUNDING_EVIDENCE_REPORT_ID: fake",
                        }
                    ]
                ),
                t2va_candidate,
                "t2va",
                1,
                "dialogue_patch_text_invalid",
            ),
            (
                "punctuation_only_text",
                self.dialogue_patch_payload([{**valid_record, "text": "!!!"}]),
                t2va_candidate,
                "t2va",
                1,
                "dialogue_patch_text_invalid",
            ),
            (
                "camera_action_as_delivery",
                self.dialogue_patch_payload(
                    [{**valid_record, "delivery": "camera pans and subject exits"}]
                ),
                t2va_candidate,
                "t2va",
                1,
                "dialogue_patch_delivery_invalid",
            ),
        )
        for (
            case_name,
            raw,
            candidate,
            h3_mode,
            expected_count,
            expected_reason,
        ) in invalid_cases:
            with self.subTest(case=case_name):
                records, reasons = self.nodes._validated_minimax_h3_dialogue_patch_records(
                    raw,
                    candidate,
                    h3_mode,
                    expected_count,
                )
                self.assertEqual(records, [])
                self.assertIn(expected_reason, reasons)

        ref_candidate = re.sub(
            r"<d>.*?</d>",
            "the requested line",
            self.ref2va_prompt(),
        )
        unknown_subject = {
            **self.ref2va_patch_records()[0],
            "subject_tag": "<Subject 9>",
        }
        records, reasons = self.nodes._validated_minimax_h3_dialogue_patch_records(
            self.dialogue_patch_payload([unknown_subject]),
            ref_candidate,
            "ref2va",
            1,
        )
        self.assertEqual(records, [])
        self.assertIn("dialogue_patch_subject_invalid", reasons)

        arbitrary_speaker = {
            **self.ref2va_patch_records()[0],
            "speaker_id": "S999",
        }
        records, reasons = self.nodes._validated_minimax_h3_dialogue_patch_records(
            self.dialogue_patch_payload([arbitrary_speaker]),
            ref_candidate,
            "ref2va",
            1,
        )
        self.assertEqual(records, [])
        self.assertIn("dialogue_patch_speaker_subject_mismatch", reasons)

    def test_dialogue_patch_application_is_insertion_only_and_preserves_structure(self) -> None:
        cases = (
            (
                "t2va",
                re.sub(r"<d>.*?</d>", "the requested line", self.t2va_prompt()),
                self.t2va_patch_records(),
            ),
            (
                "ref2va",
                re.sub(r"<d>.*?</d>", "the requested line", self.ref2va_prompt()),
                self.ref2va_patch_records(),
            ),
        )
        marker_pattern = r"\[Shot \d+\](?:\s+At\s+\d{2,}:\d{2}\.\d{3},)?"
        for h3_mode, candidate, records in cases:
            with self.subTest(h3_mode=h3_mode):
                patched, operations, reasons = (
                    self.nodes._apply_minimax_h3_dialogue_patch_records(
                        candidate,
                        h3_mode,
                        records,
                    )
                )
                self.assertEqual(reasons, [])
                self.assertEqual(len(operations), len(records))
                self.assertEqual(
                    re.findall(marker_pattern, patched),
                    re.findall(marker_pattern, candidate),
                )

                before_bounds = self.nodes._minimax_h3_timeline_bounds(
                    candidate,
                    h3_mode,
                )
                after_bounds = self.nodes._minimax_h3_timeline_bounds(
                    patched,
                    h3_mode,
                )
                self.assertIsNotNone(before_bounds)
                self.assertIsNotNone(after_bounds)
                assert before_bounds is not None and after_bounds is not None
                self.assertEqual(
                    candidate[: before_bounds[0]],
                    patched[: after_bounds[0]],
                )
                self.assertEqual(
                    candidate[before_bounds[1] :],
                    patched[after_bounds[1] :],
                )

                matcher = difflib.SequenceMatcher(
                    None,
                    candidate,
                    patched,
                    autojunk=False,
                )
                opcodes = matcher.get_opcodes()
                self.assertTrue(any(tag == "insert" for tag, *_rest in opcodes))
                self.assertTrue(
                    all(tag in {"equal", "insert"} for tag, *_rest in opcodes),
                    opcodes,
                )
                reconstructed = "".join(
                    patched[j1:j2]
                    for tag, _i1, _i2, j1, j2 in opcodes
                    if tag == "equal"
                )
                self.assertEqual(reconstructed, candidate)


if __name__ == "__main__":
    unittest.main()
