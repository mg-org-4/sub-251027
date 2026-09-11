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
    package_name = f"diffusiongemma_manifest_presets_{uuid.uuid4().hex}"
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


class MiniMaxH3ReferenceManifestPresetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()

    @staticmethod
    def reference_manifest(picture_count: int = 1) -> str:
        lines = [
            "<Picture 1>: Mara identity, dark bobbed hair, blue coat, and brass suitcase."
        ]
        if picture_count == 2:
            lines.append(
                "<Picture 2>: Quiet train-compartment environment, dawn window light, and restrained blue-brown palette."
            )
        return "\n".join(lines)

    @classmethod
    def valid_ref2va_prompt(
        cls,
        subject_count: int = 1,
        picture_count: int = 1,
    ) -> str:
        picture_sources = " and ".join(
            f"<Picture {ordinal}>" for ordinal in range(1, picture_count + 1)
        )
        definitions = [
            "<Subject 1>: Mara, a traveler whose face, dark bobbed hair, blue coat, and brass suitcase derive from "
            f"{picture_sources}."
        ]
        retention = [
            "<Subject 1>: fully_preserved - Her face, dark bobbed hair, blue coat, proportions, and brass suitcase "
            "remain stable in both shots."
        ]
        timeline_extras: list[str] = []
        for ordinal in range(2, subject_count + 1):
            definitions.append(
                f"<Subject {ordinal}>: A recurring blue geometric compartment fixture with stable brass trim derived from <Picture 1>."
            )
            retention.append(
                f"<Subject {ordinal}>: fully_preserved - Its blue geometry and brass trim remain fixed beside Mara in both shots."
            )
            timeline_extras.append(
                f"<Subject {ordinal}> remains visible beside her with unchanged blue geometry and brass trim."
            )

        return (
            "subject_definitions:\n"
            + "\n".join(definitions)
            + "\n\nsummary:\n"
            "[reference generation] <Subject 1> shares a decision before closing the suitcase.\n\n"
            "retention_analysis:\n"
            + "\n".join(retention)
            + "\n\ndetailed_description:\n"
            "Realistic live-action practical film photography uses soft window light, restrained blue-brown color, "
            "shallow depth of field, and fine grain throughout. [Shot 1] A locked medium shot holds <Subject 1> (S1) "
            "on the left side of a quiet train compartment, matching the face, dark bobbed hair, blue coat, and brass "
            f"suitcase derived from {picture_sources}. "
            + " ".join(timeline_extras)
            + " Passing dawn light slides across the wall while she keeps both hands on the suitcase. <Subject 1> (S1) "
            "looks toward an unseen companion and says <d>[English] We leave at dawn.</d> in a steady voice. The camera "
            "remains static while her mouth forms the words and her right hand moves to the latch. Low train vibration "
            "subtly moves the hanging curtain without changing her position or identity. [Shot 2] At 00:06.000, the "
            "camera cuts to a static close-up of <Subject 1> (S1) beside the brass latch. She answers her own uncertainty "
            "with <d>[English] Then we are ready.</d> and closes the suitcase in one visible motion. The latch clicks only "
            "when her thumb presses it; her blue sleeve settles against the brass edge. The camera holds her resolved "
            "expression and the same soft window light as the train continues forward. The final frame holds on "
            "<Subject 1> seated calmly beside the closed suitcase in the static close composition.\n\n"
            "overall_soundscape:\n"
            "Low train rumble, curtain movement, cloth rustle, and the synchronized suitcase-latch click surround the "
            "compartment without repeating speech.\n\n"
            "non_diegetic_music:\n"
            "N/A"
        )

    def ref2va_validation_reasons(
        self,
        prompt: str,
        *,
        expected_subject_count: int = 0,
        picture_count: int = 1,
    ) -> list[str]:
        return self.nodes._minimax_h3_ref2va_validation_reasons(
            prompt,
            12.0,
            "Create exactly two shots in a quiet train compartment.",
            "auto_scene_audio",
            8000,
            self.reference_manifest(picture_count),
            "2",
            "required",
            2,
            "Mara says We leave at dawn, then says Then we are ready.",
            minimax_h3_expected_subject_count=expected_subject_count,
        )

    def test_ui_contract_appends_preset_after_existing_optional_inputs(self) -> None:
        optional = self.nodes.DiffusionGemmaH3ReferenceContext.INPUT_TYPES()["optional"]
        self.assertEqual(
            list(optional),
            [
                "reference_images",
                "reference_video",
                "visual_description",
                "reference_manifest_preset",
                "expected_subject_count",
            ],
        )
        choices, config = optional["reference_manifest_preset"]
        self.assertEqual(
            choices,
            [
                "custom",
                "1 image - all visual attributes",
                "2 images - subject + environment/style",
                "2 images - main subject + same-subject contact sheet",
                "2 images - main subject + multi-entity contact sheet",
            ],
        )
        self.assertEqual(config["default"], "custom")
        count_type, count_config = optional["expected_subject_count"]
        self.assertEqual(count_type, "INT")
        self.assertEqual(
            count_config,
            {
                "default": 0,
                "min": 0,
                "max": 99,
                "step": 1,
                "tooltip": count_config["tooltip"],
            },
        )

    def test_custom_preset_preserves_the_typed_manifest(self) -> None:
        typed = "<image 1>: [dg:identity,appearance] custom subject identity and wardrobe"
        expected = "<Picture 1>: [dg:identity,appearance] custom subject identity and wardrobe"
        self.assertEqual(
            self.nodes._resolve_minimax_h3_reference_manifest(typed, "custom"),
            expected,
        )
        self.assertEqual(
            self.nodes._resolve_minimax_h3_reference_manifest(typed, "unknown preset"),
            expected,
        )

    def test_one_and_two_image_presets_are_strict_valid(self) -> None:
        typed = "<Picture 9>: this text must be ignored by a selected preset"
        one = self.nodes._resolve_minimax_h3_reference_manifest(
            typed,
            "1 image - all visual attributes",
        )
        two = self.nodes._resolve_minimax_h3_reference_manifest(
            typed,
            "2 images - subject + environment/style",
        )

        self.assertEqual(self.nodes._minimax_h3_reference_tags(one), ["<Picture 1>"])
        self.assertEqual(
            self.nodes._minimax_h3_reference_tags(two),
            ["<Picture 1>", "<Picture 2>"],
        )
        self.assertEqual(self.nodes._minimax_h3_reference_manifest_validation_reasons(one), [])
        self.assertEqual(self.nodes._minimax_h3_reference_manifest_validation_reasons(two), [])
        self.assertIn("[dg:identity,appearance,color]", two)
        self.assertIn("[dg:environment,lighting,color,composition]", two)

    def test_context_build_uses_effective_preset_in_metadata_and_preview(self) -> None:
        context, context_json, preview = self.nodes.DiffusionGemmaH3ReferenceContext().build(
            "Create a short sequence.",
            "<Picture 9>: ignored custom text",
            reference_manifest_preset="2 images - subject + environment/style",
        )
        metadata = context.media_metadata
        serialized = json.loads(context_json)["media"]

        self.assertEqual(
            metadata["minimax_h3_reference_manifest_preset"],
            "2 images - subject + environment/style",
        )
        self.assertEqual(metadata["minimax_h3_reference_tags"], ["<Picture 1>", "<Picture 2>"])
        self.assertEqual(metadata["minimax_h3_reference_manifest_reasons"], [])
        self.assertEqual(serialized["minimax_h3_reference_manifest"], metadata["minimax_h3_reference_manifest"])
        self.assertIn("<Picture 2>", preview)
        self.assertNotIn("<Picture 9>", preview)

    def test_expected_subject_count_defaults_to_auto_for_saved_workflow_compatibility(self) -> None:
        context, context_json, _preview = self.nodes.DiffusionGemmaH3ReferenceContext().build(
            "Create a short sequence.",
            self.reference_manifest(),
        )
        serialized = json.loads(context_json)["media"]

        self.assertEqual(context.media_metadata["minimax_h3_expected_subject_count"], 0)
        self.assertEqual(serialized["minimax_h3_expected_subject_count"], 0)

    def test_expected_subject_count_is_normalized_into_context_metadata(self) -> None:
        context, context_json, _preview = self.nodes.DiffusionGemmaH3ReferenceContext().build(
            "Create a short sequence.",
            self.reference_manifest(),
            expected_subject_count=2,
        )
        serialized = json.loads(context_json)["media"]

        self.assertEqual(context.media_metadata["minimax_h3_expected_subject_count"], 2)
        self.assertEqual(serialized["minimax_h3_expected_subject_count"], 2)

    def test_auto_expected_subject_count_accepts_multiple_semantic_subjects(self) -> None:
        reasons = self.ref2va_validation_reasons(
            self.valid_ref2va_prompt(subject_count=5),
            expected_subject_count=0,
        )
        self.assertEqual(reasons, [])

    def test_exact_expected_subject_counts_one_and_two_are_enforced(self) -> None:
        one_subject = self.valid_ref2va_prompt(subject_count=1)
        two_subjects = self.valid_ref2va_prompt(subject_count=2)

        self.assertEqual(
            self.ref2va_validation_reasons(
                one_subject,
                expected_subject_count=1,
            ),
            [],
        )
        self.assertEqual(
            self.ref2va_validation_reasons(
                two_subjects,
                expected_subject_count=2,
            ),
            [],
        )
        self.assertIn(
            "minimax_h3_ref_subject_count_mismatch",
            self.ref2va_validation_reasons(
                one_subject,
                expected_subject_count=2,
            ),
        )
        self.assertIn(
            "minimax_h3_ref_subject_count_mismatch",
            self.ref2va_validation_reasons(
                two_subjects,
                expected_subject_count=1,
            ),
        )

    def test_expected_subject_count_is_independent_of_picture_count(self) -> None:
        one_subject_two_pictures = self.ref2va_validation_reasons(
            self.valid_ref2va_prompt(subject_count=1, picture_count=2),
            expected_subject_count=1,
            picture_count=2,
        )
        two_subjects_one_picture = self.ref2va_validation_reasons(
            self.valid_ref2va_prompt(subject_count=2, picture_count=1),
            expected_subject_count=2,
            picture_count=1,
        )

        self.assertEqual(one_subject_two_pictures, [])
        self.assertEqual(two_subjects_one_picture, [])

    def test_subject_count_refinement_requests_exact_rewrite_without_host_deletion(self) -> None:
        candidate = self.valid_ref2va_prompt(subject_count=2)
        refinement = self.nodes._build_minimax_h3_refinement_prompt(
            "Create exactly two shots in a quiet train compartment.",
            candidate,
            12.0,
            "auto_scene_audio",
            ["minimax_h3_ref_subject_count_mismatch"],
            "ref2va",
            self.reference_manifest(),
            minimax_h3_shot_count="2",
            minimax_h3_dialogue_mode="required",
            minimax_h3_dialogue_line_count=2,
            minimax_h3_dialogue_guidance=(
                "Mara says We leave at dawn, then says Then we are ready."
            ),
            minimax_h3_expected_subject_count=1,
        )

        self.assertIn(
            "Create exactly 1 consecutive semantic Subject label",
            refinement,
        )
        self.assertIn("and no other <Subject N> labels", refinement)
        self.assertIn("<Subject 1>", refinement)
        self.assertIn("<Subject 2>", refinement)

    def test_dangling_grounding_annotation_fragment_is_rejected(self) -> None:
        prompt = self.valid_ref2va_prompt().replace(
            "derive from <Picture 1>.",
            "derive from <Picture 1>: dg:identity,appearance,color] leaked host annotation.",
            1,
        )
        self.assertIn(
            "minimax_h3_prompt_contains_grounding_role_annotation",
            self.ref2va_validation_reasons(prompt),
        )

    def test_malformed_non_diegetic_music_sentinel_is_rejected(self) -> None:
        prompt = self.valid_ref2va_prompt().removesuffix("N/A") + "/A"
        self.assertIn(
            "minimax_h3_non_diegetic_music_invalid",
            self.ref2va_validation_reasons(prompt),
        )

    def test_multi_entity_contact_sheet_enforces_subject_source_bindings(self) -> None:
        manifest = self.nodes._resolve_minimax_h3_reference_manifest(
            "",
            "2 images - main subject + multi-entity contact sheet",
        )
        invalid = self.valid_ref2va_prompt(subject_count=2, picture_count=2)
        invalid_reasons = self.nodes._minimax_h3_ref2va_validation_reasons(
            invalid,
            12.0,
            "Create exactly two shots in a quiet train compartment.",
            "auto_scene_audio",
            8000,
            manifest,
            "2",
            "required",
            2,
            "Mara says We leave at dawn, then says Then we are ready.",
            minimax_h3_expected_subject_count=2,
        )
        self.assertIn(
            "minimax_h3_ref_contact_sheet_subject_source_invalid",
            invalid_reasons,
        )

        valid = invalid.replace(
            "derive from <Picture 1> and <Picture 2>",
            "derive from <Picture 1>",
        ).replace(
            "derived from <Picture 1> and <Picture 2>",
            "derived from <Picture 1>",
        ).replace(
            "A recurring blue geometric compartment fixture with stable brass trim derived from <Picture 1>.",
            "A recurring blue geometric compartment fixture with stable brass trim derived from <Picture 2>.",
        )
        valid_reasons = self.nodes._minimax_h3_ref2va_validation_reasons(
            valid,
            12.0,
            "Create exactly two shots in a quiet train compartment.",
            "auto_scene_audio",
            8000,
            manifest,
            "2",
            "required",
            2,
            "Mara says We leave at dawn, then says Then we are ready.",
            minimax_h3_expected_subject_count=2,
        )
        self.assertEqual(valid_reasons, [])

    def test_contact_sheet_layout_language_is_rejected_from_timeline(self) -> None:
        manifest = self.nodes._resolve_minimax_h3_reference_manifest(
            "",
            "2 images - main subject + multi-entity contact sheet",
        )
        prompt = self.valid_ref2va_prompt(subject_count=2, picture_count=2).replace(
            "derive from <Picture 1> and <Picture 2>",
            "derive from <Picture 1>",
        ).replace(
            "derived from <Picture 1> and <Picture 2>",
            "derived from <Picture 1>",
        ).replace(
            "A recurring blue geometric compartment fixture with stable brass trim derived from <Picture 1>.",
            "A recurring blue geometric compartment fixture with stable brass trim derived from <Picture 2>.",
        ).replace(
            "Realistic live-action practical film photography",
            "A contact-sheet grid remains visible behind them. Realistic live-action practical film photography",
        )
        reasons = self.nodes._minimax_h3_ref2va_validation_reasons(
            prompt,
            12.0,
            "Create exactly two shots in a quiet train compartment.",
            "auto_scene_audio",
            8000,
            manifest,
            "2",
            "required",
            2,
            "Mara says We leave at dawn, then says Then we are ready.",
            minimax_h3_expected_subject_count=2,
        )
        self.assertIn("minimax_h3_ref_contact_sheet_layout_transfer_invalid", reasons)


if __name__ == "__main__":
    unittest.main()
