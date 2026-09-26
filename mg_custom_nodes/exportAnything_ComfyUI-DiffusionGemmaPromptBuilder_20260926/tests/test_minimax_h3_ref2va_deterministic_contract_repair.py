from __future__ import annotations

import copy
import importlib.util
import json
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
    package_name = f"diffusiongemma_h3_deterministic_repair_{uuid.uuid4().hex}"
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


class MiniMaxH3Ref2VADeterministicContractRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()

    @staticmethod
    def user_prompt() -> str:
        return (
            "Create a rap music video; keep the black-and-white high-fashion "
            "photography aesthetic."
        )

    @staticmethod
    def manifest() -> str:
        return (
            "<Picture 1>: sole visual reference for visible subject, wardrobe, scene, palette, and composition; "
            "preserve source-grounded identity and appearance across every shot.\n"
            "<Audio 1>: locked selected soundtrack excerpt; copy its exact timing, vocal phrasing, beat, dynamics, "
            "and dance synchronization without replacing or reinterpreting the composition."
        )

    @staticmethod
    def valid_prompt() -> str:
        return (
            "subject_definitions:\n"
            "<Subject 1>: A high-fashion female model with short dark hair, wearing a black sleeveless bodysuit with "
            "thin straps and sheer tights, featuring a confident pose as seen in <Picture 1>.\n"
            "<Audio 1>: A high-energy rap soundtrack with rhythmic vocal phrasing and heavy beats, referenced as "
            "<Audio 1>.\n\n"
            "summary:\n"
            "[reference generation + audio reference] <Subject 1> performs a rhythmic rap performance in a high-contrast "
            "black-and-white environment synchronized to the beats of <Audio 1>.\n\n"
            "retention_analysis:\n"
            "<Subject 1>: fully_preserved - The model's face, hair, bodysuit, and high-fashion aesthetic are maintained "
            "across all shots.\n"
            "<Audio 1>: fully_copy - The exact locked soundtrack timing and vocal dynamics dictate the performance "
            "movements and camera cuts.\n\n"
            "detailed_description:\n"
            "The video is rendered in a high-contrast black-and-white fashion photography aesthetic, characterized by "
            "deep shadows, blown highlights, and sharp film grain. The setting is a minimalist space with a tiled wall.\n\n"
            "[Shot 1] <Subject 1> stands in a wide-stanced power pose derived from <Picture 1>, one hand raised to her "
            "head and the other on her hip. The camera is a static medium shot, capturing her silhouette against the "
            "white tiles as she begins to move rhythmically to the opening beat of <Audio 1>. Her shoulders settle on "
            "the downbeat while the tiled geometry and hard side light remain fixed behind her.\n\n"
            "[Shot 2] At 00:04.500, the camera cuts to a static close-up on <Subject 1>'s face. She leans toward the lens "
            "and marks the rap cadence with a controlled head turn. The harsh lighting creates dramatic shadows across "
            "her cheekbones and neck while the same hairstyle, wardrobe, and facial proportions remain clear.\n\n"
            "[Shot 3] At 00:09.000, the camera cuts to a low-angle tracking shot. <Subject 1> performs dynamic dance "
            "movements, shifting her weight and gesturing with her hands to the double-time pulse of <Audio 1>. The "
            "camera tracks upward slightly, emphasizing her height and the geometry of the bodysuit without changing "
            "the monochrome tiled setting or her screen direction.\n\n"
            "[Shot 4] At 00:12.500, the camera cuts to a locked medium-wide shot. <Subject 1> resumes her original pose, "
            "looking directly into the lens with an intense gaze as the music reaches its final accent. Her feet stop, "
            "her hands settle, and the camera holds the resolved symmetrical composition through the final frame.\n\n"
            "overall_soundscape:\n"
            "Faint floor vibration, measured foot contacts, and clothing rustle stay synchronized with the visible "
            "movement without adding speech.\n\n"
            "non_diegetic_music:\n"
            "The exact locked fast-paced rap track from <Audio 1>, with its driving bassline, crisp percussion, and "
            "sharp vocal phrasing preserved unchanged."
        )

    def reasons(self, prompt: str) -> list[str]:
        return self.nodes._minimax_h3_ref2va_validation_reasons(
            prompt,
            15.0,
            self.user_prompt(),
            "auto_scene_audio",
            20_000,
            self.manifest(),
            "4",
            "off",
            2,
            "",
            0,
        )

    def exact_trio_candidate(self) -> str:
        sections = self.nodes._minimax_h3_ref_sections(self.valid_prompt())
        self.assertIsNotNone(sections)
        sections = dict(sections)
        sections["retention_analysis"] = sections["retention_analysis"].replace(
            "<Audio 1>: fully_copy",
            "<Audio 1>: fully_preserved",
            1,
        )
        sections["detailed_description"] = sections["detailed_description"].replace(
            "<Subject 1>",
            "the performer",
        ).replace(
            "[Shot 3] At 00:09.000,",
            "[Shot 3]",
            1,
        )
        return self.nodes._rebuild_minimax_h3_ref_prompt(sections)

    def test_exact_three_reason_candidate_repairs_before_model_retry(self) -> None:
        candidate = self.exact_trio_candidate()
        self.assertEqual(
            self.reasons(candidate),
            [
                "minimax_h3_ref_subject_usage_invalid",
                "minimax_h3_ref_retention_invalid",
                "minimax_h3_cut_timestamp_invalid",
            ],
        )

        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            self.user_prompt(),
            self.manifest(),
            15.0,
        )

        self.assertEqual(self.reasons(repaired), [])
        self.assertIn("<Audio 1>: fully_copy - The exact locked soundtrack", repaired)
        self.assertIn("[Shot 2] At 00:04.500,", repaired)
        self.assertIn("[Shot 3] At 00:08.500,", repaired)
        self.assertIn("[Shot 4] At 00:12.500,", repaired)
        self.assertIn("<Subject 1> (derived from <Picture 1>)", repaired)
        self.assertEqual(repaired.count("[Shot "), 4)
        self.assertIn("performs dynamic dance movements", repaired)
        self.assertEqual(
            self.nodes._repair_minimax_h3_ref2va_structure(
                repaired,
                self.user_prompt(),
                self.manifest(),
                15.0,
            ),
            repaired,
        )

        packet = {
            "ltx_prompt": "",
            "ideogram_prompt": "",
            "minimax_h3_prompt": candidate,
            "negative_prompt": "",
            "scene_segments": [],
            "metadata": {},
        }
        context = self.nodes.GemmaContext(
            user_prompt=self.user_prompt(),
            images=None,
            source="minimax_h3_ref2va",
            media_metadata={
                "source": "minimax_h3_ref2va",
                "duration_seconds": 15.0,
                "reference_image_count": 1,
                "minimax_h3_reference_image_batch_count": 1,
                "minimax_h3_reference_manifest": self.manifest(),
                "minimax_h3_expected_subject_count": 0,
            },
        )
        target = self.nodes.TargetProfileConfig(
            target_profile="minimax_h3",
            audio_mode="auto_scene_audio",
            target_duration_seconds=15.0,
            minimax_h3_mode="ref2va",
            minimax_h3_shot_count="4",
            minimax_h3_dialogue_mode="off",
        )
        runtime = self.nodes.RuntimeConfig(
            model_path="unused-local-checkpoint",
            backend="transformers_inprocess",
            dtype="auto",
            quantization="none",
            local_files_only=True,
            unload_policy="unload_after_run",
            max_memory_gb=20.0,
            status={"ready": True, "supports_pixels": False},
        )
        with patch.object(
            self.nodes,
            "_run_backend_for_packet",
            return_value=json.dumps(packet),
        ) as backend:
            result = self.nodes._run_generation_packet_legacy(
                runtime,
                context,
                target,
                runtime_required=True,
                max_output_chars=20_000,
                manage_unload=False,
                max_refinement_attempts_override=2,
            )
        self.assertEqual(backend.call_count, 1)
        self.assertTrue(result[-1]["ready_for_generation"])
        self.assertEqual(self.reasons(result[0]["minimax_h3_prompt"]), [])

    def test_missing_detailed_heading_preserves_all_fifteen_authored_shots(self) -> None:
        style_opening = (
            "The target video uses an immutable black-and-white high-fashion photography aesthetic with hard contrast, "
            "controlled film grain, and precise editorial composition."
        )
        shot_blocks: list[str] = []
        for shot in range(1, 16):
            if shot == 1:
                opening = "[Shot 1]"
            else:
                milliseconds = (shot - 1) * 1500
                minutes, remainder = divmod(milliseconds, 60_000)
                seconds, millis = divmod(remainder, 1000)
                opening = (
                    f"[Shot {shot}] At {minutes:02d}:{seconds:02d}.{millis:03d}, "
                    "the camera cuts to a"
                )
            ending = (
                "The final frame holds on the resolved centered pose while the camera remains completely static."
                if shot == 15
                else "The camera remains completely static while the pose resolves before the next measured cut."
            )
            shot_blocks.append(
                f"{opening} high-contrast fashion portrait of <Subject 1> derived from <Picture 1>, marking the exact "
                f"rap cadence of <Audio 1> with a distinct controlled gesture for beat {shot}. {ending}"
            )
        misplaced_timeline = "\n\n".join((style_opening, *shot_blocks))
        candidate = (
            "subject_definitions:\n"
            "<Picture 1>: The sole visual reference for the visible performer, wardrobe, monochrome set, lighting, and composition.\n"
            "<Audio 1>: The exact locked rap soundtrack supplies beat timing, vocal phrasing, dynamics, and performance pacing.\n\n"
            "summary:\n"
            "[reference generation + audio reference] <Subject 1> performs a black-and-white fashion rap video derived "
            "from <Picture 1> and synchronized to <Audio 1>.\n\n"
            "retention_analysis:\n"
            "<Picture 1>: fully_preserved - Preserve the performer, wardrobe, monochrome set, lighting, and composition.\n"
            "<Audio 1>: fully_copy - Preserve the exact song, timing, vocal phrasing, dynamics, and cadence.\n"
            f"{misplaced_timeline}\n\n"
            "overall_soundscape:\n"
            "Measured foot contacts and clothing movement remain subordinate to the exact locked song.\n\n"
            "non_diegetic_music:\n"
            "Use the exact supplied <Audio 1> composition without replacement or regeneration."
        )
        initial_reasons = self.nodes._minimax_h3_ref2va_validation_reasons(
            candidate,
            25.0,
            self.user_prompt(),
            "auto_scene_audio",
            20_000,
            self.manifest(),
            "15",
            "off",
            2,
            "",
            0,
        )
        self.assertIn("minimax_h3_ref_missing_detailed_description", initial_reasons)
        self.assertEqual(candidate.count("[Shot "), 15)

        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            self.user_prompt(),
            self.manifest(),
            25.0,
        )
        repaired_sections = self.nodes._minimax_h3_ref_sections(repaired)
        self.assertIsNotNone(repaired_sections)
        self.assertEqual(repaired_sections["detailed_description"].count("[Shot "), 15)
        self.assertIn("[Shot 15] At 00:21.000", repaired_sections["detailed_description"])
        self.assertIn("<Subject 1>:", repaired_sections["subject_definitions"])
        self.assertIn("Reference source: <Picture 1>.", repaired_sections["subject_definitions"])
        self.assertIn("<Subject 1>: fully_preserved -", repaired_sections["retention_analysis"])
        self.assertEqual(
            self.nodes._minimax_h3_ref2va_validation_reasons(
                repaired,
                25.0,
                self.user_prompt(),
                "auto_scene_audio",
                20_000,
                self.manifest(),
                "15",
                "off",
                2,
                "",
                0,
            ),
            [],
        )
        self.assertEqual(
            self.nodes._repair_minimax_h3_ref2va_structure(
                repaired,
                self.user_prompt(),
                self.manifest(),
                25.0,
            ),
            repaired,
        )

    def test_canonical_picture_only_performer_definition_is_promoted_safely(self) -> None:
        sections = self.nodes._minimax_h3_ref_sections(self.valid_prompt())
        self.assertIsNotNone(sections)
        sections = dict(sections)
        sections["subject_definitions"] = sections["subject_definitions"].replace(
            "<Subject 1>: A high-fashion female model",
            "<Picture 1>: A high-fashion female model",
            1,
        )
        sections["retention_analysis"] = sections["retention_analysis"].replace(
            "<Subject 1>: fully_preserved",
            "<Picture 1>: fully_preserved",
            1,
        )
        candidate = self.nodes._rebuild_minimax_h3_ref_prompt(sections)
        self.assertIn("minimax_h3_ref_subject_definition_invalid", self.reasons(candidate))

        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            self.user_prompt(),
            self.manifest(),
            15.0,
        )

        self.assertEqual(self.reasons(repaired), [])
        self.assertIn("<Subject 1>: A high-fashion female model", repaired)
        self.assertIn("Reference source: <Picture 1>.", repaired)
        self.assertIn("<Subject 1>: fully_preserved -", repaired)

    def test_environment_picture_is_never_promoted_to_performer_subject(self) -> None:
        plate_descriptions = (
            "An empty monochrome dressing room with wardrobe racks, hard side lighting, and no visible inhabitants.",
            "An empty tiled room with no person, human, performer, or model visible anywhere in the frame.",
            "A fashion style board showing wardrobe palette, fabric swatches, lighting references, and set textures.",
        )
        for plate_description in plate_descriptions:
            with self.subTest(plate_description=plate_description):
                sections = self.nodes._minimax_h3_ref_sections(self.valid_prompt())
                self.assertIsNotNone(sections)
                sections = dict(sections)
                sections["subject_definitions"] = (
                    f"<Picture 1>: {plate_description}\n"
                    "<Audio 1>: A high-energy rap soundtrack with rhythmic vocal phrasing and heavy beats."
                )
                sections["retention_analysis"] = (
                    "<Picture 1>: attribute_transfer - Preserve the authored plate attributes without inventing a performer.\n"
                    "<Audio 1>: fully_copy - Preserve the exact locked soundtrack."
                )
                candidate = self.nodes._rebuild_minimax_h3_ref_prompt(sections)

                repaired = self.nodes._repair_minimax_h3_ref2va_structure(
                    candidate,
                    self.user_prompt(),
                    self.manifest(),
                    15.0,
                )

                repaired_sections = self.nodes._minimax_h3_ref_sections(repaired)
                self.assertIsNotNone(repaired_sections)
                self.assertNotIn("<Subject 1>:", repaired_sections["subject_definitions"])
                self.assertIn("minimax_h3_ref_subject_definition_invalid", self.reasons(repaired))

    def test_salvaged_picture_timeline_receives_promoted_subject_label(self) -> None:
        sections = self.nodes._minimax_h3_ref_sections(self.valid_prompt())
        self.assertIsNotNone(sections)
        sections = dict(sections)
        sections["subject_definitions"] = sections["subject_definitions"].replace(
            "<Subject 1>: A high-fashion female model",
            "<Picture 1>: A high-fashion female model",
            1,
        )
        sections["retention_analysis"] = sections["retention_analysis"].replace(
            "<Subject 1>: fully_preserved",
            "<Picture 1>: fully_preserved",
            1,
        )
        sections["detailed_description"] = sections["detailed_description"].replace(
            "<Subject 1>",
            "<Picture 1>",
        )
        candidate = (
            f"subject_definitions:\n{sections['subject_definitions']}\n\n"
            f"summary:\n{sections['summary']}\n\n"
            f"retention_analysis:\n{sections['retention_analysis']}\n"
            f"{sections['detailed_description']}\n\n"
            f"overall_soundscape:\n{sections['overall_soundscape']}\n\n"
            f"non_diegetic_music:\n{sections['non_diegetic_music']}"
        )

        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            self.user_prompt(),
            self.manifest(),
            15.0,
        )

        repaired_sections = self.nodes._minimax_h3_ref_sections(repaired)
        self.assertIsNotNone(repaired_sections)
        timeline = repaired_sections["detailed_description"]
        self.assertIn("<Subject 1> (derived from <Picture 1>)", timeline)
        self.assertNotIn("minimax_h3_ref_subject_usage_invalid", self.reasons(repaired))

    def test_live_dropped_n_music_alias_with_empty_canonical_footer_is_repaired(self) -> None:
        music_heading = "non_diegetic_music:\n"
        candidate = self.valid_prompt().replace(
            music_heading,
            "on_diegetic_music:\n",
            1,
        ) + "\n\nnon_diegetic_music:"

        self.assertIn("minimax_h3_extra_top_level_field", self.reasons(candidate))
        self.assertIn("minimax_h3_ref_missing_non_diegetic_music", self.reasons(candidate))

        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            self.user_prompt(),
            self.manifest(),
            15.0,
        )

        self.assertEqual(repaired.count("non_diegetic_music:"), 1)
        self.assertNotRegex(repaired, r"(?m)^on_diegetic_music:")
        self.assertIn("The exact locked fast-paced rap track", repaired)
        self.assertEqual(self.reasons(repaired), [])

    def test_duplicate_empty_music_footer_preserves_complete_music_section(self) -> None:
        candidate = self.valid_prompt() + "\n\nnon_diegetic_music:"

        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            self.user_prompt(),
            self.manifest(),
            15.0,
        )

        self.assertEqual(repaired.count("non_diegetic_music:"), 1)
        self.assertIn("The exact locked fast-paced rap track", repaired)
        self.assertEqual(self.reasons(repaired), [])

    def test_live_cosmic_environment_subject_gets_a_timeline_binding(self) -> None:
        user_prompt = (
            "No dialogue. Superman holds the lifeless Lois Lane in space, overcome with guilt and grief before his "
            "sadness turns to anger and he rushes back to Earth as a streak of energy."
        )
        manifest = (
            "<Picture 1>: [dg:identity,appearance,color,environment,lighting,composition] sole visual reference for "
            "Superman, Lois Lane, their visible appearance, colors, spatial relationship, space environment, lighting, "
            "and composition; preserve visible identities and scene attributes throughout."
        )
        candidate = (
            "subject_definitions:\n"
            "<Subject 1>: (Picture 1): Superman in his classic blue suit with a red cape and the yellow 'S' shield, "
            "expressing intense grief and burgeoning rage.\n"
            "<Subject 2>: (Picture 1): Lois Lane, with long wavy hair, wearing a white maternity dress, appearing limp "
            "and lifeless in space.\n"
            "<Subject 3>: The void of space above the curved horizon of Earth, with clouds and distant stars visible.\n"
            "<Picture 1>: sole visual reference for all reusable visible content and scene attributes; picture count "
            "does not determine semantic Subject count.\n\n"
            "summary:\n"
            "[reference generation] The scene depicts Superman holding the lifeless Lois Lane in the orbit of Earth, "
            "overcome by guilt and grief before transitioning into a furious, high-speed descent back to Earth.\n\n"
            "retention_analysis:\n"
            "<Subject 1>: fully_preserved - Superman's appearance and emotional state are maintained from Picture 1.\n"
            "<Subject 2>: fully_preserved - Lois Lane's appearance and white dress are maintained from Picture 1.\n"
            "<Subject 3>: fully_preserved - The space environment and Earth horizon are consistent with Picture 1.\n"
            "<Picture 1>: attribute_transfer - sole visual reference for all reusable visible content and scene "
            "attributes; picture count does not determine semantic Subject count.\n\n"
            "detailed_description:\n"
            "The visual style is cinematic with high-contrast lighting, utilizing deep shadows of space and the "
            "brilliant glow from the Earth's atmosphere. [Shot 1] In the silent void of space, <Subject 1> holds the "
            "limp <Subject 2> in his arms; <Subject 1>'s face is contorted in silent anguish as he looks down at "
            "<Subject 2>, whose white dress drifts slightly in zero-gravity. The camera performs a slow, tragic orbit "
            "around the pair, capturing their isolation against the vast, curved Earth below. [Shot 2] At 00:05.000, "
            "the shot cuts to: <Subject 1>'s expression shifts from sobbing to a fierce snarl; he releases <Subject 2> "
            "and ignites toward the camera. The camera executes a rapid tracking shot following him as he transforms "
            "into a blurred streak of blue and red energy, creating heavy motion blur as he breaks into the upper "
            "atmosphere.\n\n"
            "overall_soundscape:\n"
            "The low, rhythmic hum of space vacuum, followed by the sudden sonic boom and rushing roar of air friction "
            "as Superman enters the atmosphere.\n\n"
            "non_diegetic_music:\n"
            "A dramatic, orchestral score starting with somber, mourn cello notes building into a fast, aggressive "
            "percussion and brass crescendo as he descends."
        )

        def live_reasons(prompt: str) -> list[str]:
            return self.nodes._minimax_h3_ref2va_validation_reasons(
                prompt,
                10.0,
                user_prompt,
                "auto_scene_audio",
                20_000,
                manifest,
                "auto",
                "auto",
                2,
                "",
                0,
            )

        self.assertEqual(
            live_reasons(candidate),
            ["minimax_h3_ref_subject_usage_invalid"],
        )

        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            user_prompt,
            manifest,
            10.0,
        )

        repaired_sections = self.nodes._minimax_h3_ref_sections(repaired)
        self.assertIsNotNone(repaired_sections)
        self.assertIn(
            "[Shot 1] The setting is <Subject 3> as defined above. In the silent void of space",
            repaired_sections["detailed_description"],
        )
        self.assertEqual(live_reasons(repaired), [])
        self.assertEqual(
            self.nodes._repair_minimax_h3_ref2va_structure(
                repaired,
                user_prompt,
                manifest,
                10.0,
            ),
            repaired,
        )

    def test_multiple_implicit_environment_subjects_remain_fail_closed(self) -> None:
        sections = self.nodes._minimax_h3_ref_sections(self.valid_prompt())
        self.assertIsNotNone(sections)
        sections = copy.deepcopy(sections)
        sections["subject_definitions"] = sections["subject_definitions"].replace(
            "<Audio 1>:",
            "<Subject 2>: The minimalist tiled room environment with deep shadows and hard side lighting.\n"
            "<Subject 3>: A separate exterior city location under blue evening light and drifting fog.\n"
            "<Audio 1>:",
            1,
        )
        sections["retention_analysis"] = sections["retention_analysis"].replace(
            "<Audio 1>:",
            "<Subject 2>: fully_preserved - The tiled room and hard side lighting remain stable.\n"
            "<Subject 3>: fully_preserved - The city exterior and blue evening fog remain stable.\n"
            "<Audio 1>:",
            1,
        )
        candidate = self.nodes._rebuild_minimax_h3_ref_prompt(sections)

        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            self.user_prompt(),
            self.manifest(),
            15.0,
        )

        repaired_sections = self.nodes._minimax_h3_ref_sections(repaired)
        self.assertIsNotNone(repaired_sections)
        self.assertNotIn("<Subject 2>", repaired_sections["detailed_description"])
        self.assertNotIn("<Subject 3>", repaired_sections["detailed_description"])
        self.assertIn("minimax_h3_ref_subject_usage_invalid", self.reasons(repaired))

    def test_implicit_style_subject_is_not_recast_as_the_setting(self) -> None:
        sections = self.nodes._minimax_h3_ref_sections(self.valid_prompt())
        self.assertIsNotNone(sections)
        sections = copy.deepcopy(sections)
        sections["subject_definitions"] = sections["subject_definitions"].replace(
            "<Audio 1>:",
            "<Subject 2>: A visual style palette with sharp film grain and hard monochrome lighting.\n"
            "<Audio 1>:",
            1,
        )
        sections["retention_analysis"] = sections["retention_analysis"].replace(
            "<Audio 1>:",
            "<Subject 2>: fully_preserved - The visual style, palette, film grain, and lighting remain stable.\n"
            "<Audio 1>:",
            1,
        )
        candidate = self.nodes._rebuild_minimax_h3_ref_prompt(sections)

        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            self.user_prompt(),
            self.manifest(),
            15.0,
        )

        repaired_sections = self.nodes._minimax_h3_ref_sections(repaired)
        self.assertIsNotNone(repaired_sections)
        self.assertNotIn("<Subject 2>", repaired_sections["detailed_description"])
        self.assertIn("minimax_h3_ref_subject_usage_invalid", self.reasons(repaired))

    def test_salvage_keeps_locked_audio_as_fully_copy(self) -> None:
        sections = self.nodes._minimax_h3_ref_sections(self.valid_prompt())
        self.assertIsNotNone(sections)
        sections = dict(sections)
        retention = sections["retention_analysis"].replace(
            "<Audio 1>: fully_copy",
            "<Audio 1>: fully_preserved",
            1,
        )
        candidate = (
            f"subject_definitions:\n{sections['subject_definitions']}\n\n"
            f"summary:\n{sections['summary']}\n\n"
            f"retention_analysis:\n{retention}\n"
            f"{sections['detailed_description']}\n\n"
            f"overall_soundscape:\n{sections['overall_soundscape']}\n\n"
            f"non_diegetic_music:\n{sections['non_diegetic_music']}"
        )

        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            self.user_prompt(),
            self.manifest(),
            15.0,
        )

        repaired_sections = self.nodes._minimax_h3_ref_sections(repaired)
        self.assertIsNotNone(repaired_sections)
        self.assertIn("<Audio 1>: fully_copy -", repaired_sections["retention_analysis"])
        self.assertNotIn("<Audio 1>: reference -", repaired_sections["retention_analysis"])
        self.assertEqual(self.reasons(repaired), [])

    def test_refinement_prompt_explicitly_rebuilds_the_four_failed_contract_parts(self) -> None:
        prompt = self.nodes._build_minimax_h3_refinement_prompt(
            self.user_prompt(),
            "candidate",
            25.0,
            "auto_scene_audio",
            [
                "minimax_h3_ref_missing_detailed_description",
                "minimax_h3_ref_subject_definition_invalid",
                "minimax_h3_ref_style_opening_missing",
                "minimax_h3_ref_detailed_description_short",
            ],
            "ref2va",
            self.manifest(),
            strict_retry=True,
            minimax_h3_shot_count="15",
            minimax_h3_dialogue_mode="off",
        )
        self.assertIn("Rebuild detailed_description", prompt)
        self.assertIn("exactly 15 consecutive [Shot N] blocks", prompt)
        self.assertIn("at least 120 words", prompt)
        self.assertIn("about 20 to 35 words", prompt)
        self.assertIn("Rebuild subject_definitions", prompt)
        self.assertIn("at least four descriptive words", prompt)
        self.assertIn("replace every blank or invalid listed section", prompt)

    def test_short_one_shot_detail_budget_scales_without_weakening_dense_work(self) -> None:
        self.assertEqual(
            self.nodes._minimax_h3_ref_detail_word_budget(5.0, 1),
            (60, 60, 100),
        )
        self.assertEqual(
            self.nodes._minimax_h3_ref_detail_word_budget(15.0, 1),
            (90, 90, 130),
        )
        self.assertEqual(
            self.nodes._minimax_h3_ref_detail_word_budget(25.0, 15),
            (120, 320, 500),
        )
        for shot_count in (25, 99):
            with self.subTest(shot_count=shot_count):
                budget = self.nodes._minimax_h3_ref_detail_word_budget(
                    25.0,
                    shot_count,
                )
                self.assertEqual(budget, (120, 500, 500))
                self.assertLessEqual(budget[1], budget[2])

        refinement = self.nodes._build_minimax_h3_refinement_prompt(
            self.user_prompt(),
            "candidate with [Shot 1] and a static medium shot",
            5.0,
            "auto_scene_audio",
            ["minimax_h3_ref_detailed_description_short"],
            "ref2va",
            self.manifest(),
            minimax_h3_shot_count="1",
            minimax_h3_dialogue_mode="off",
        )
        self.assertIn("at least 60 words", refinement)
        self.assertIn("target about 60 to 100 words", refinement)
        self.assertNotIn("at least 120 words", refinement)
        self.assertIn("do not pad it with extra cuts", refinement)

        model_prompt = self.nodes._build_model_prompt(
            self.user_prompt(),
            self.nodes.DEFAULT_MASTER_PROMPT,
            "minimax_h3",
            {
                "source": "minimax_h3_ref2va",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_manifest": self.manifest(),
            },
            target_duration_seconds=5.0,
            minimax_h3_mode="ref2va",
            minimax_h3_reference_manifest=self.manifest(),
            minimax_h3_shot_count="1",
            native_h3_output=True,
            minimax_h3_dialogue_mode="off",
        )
        self.assertIn("about 60 to 100 words", model_prompt)
        self.assertIn("never fewer than 60 words", model_prompt)

    def test_initial_prompt_detail_budget_uses_resolved_duration_with_zero_override(self) -> None:
        cases = (
            (
                self.user_prompt(),
                {
                    "source": "minimax_h3_ref2va",
                    "minimax_h3_mode": "ref2va",
                    "minimax_h3_reference_manifest": self.manifest(),
                    "duration_seconds": 25.0,
                },
                "media duration",
            ),
            (
                f"{self.user_prompt()} Create a 25-second video.",
                {
                    "source": "minimax_h3_ref2va",
                    "minimax_h3_mode": "ref2va",
                    "minimax_h3_reference_manifest": self.manifest(),
                },
                "user-brief duration",
            ),
        )
        for user_prompt, media_metadata, source in cases:
            with self.subTest(source=source):
                model_prompt = self.nodes._build_model_prompt(
                    user_prompt,
                    self.nodes.DEFAULT_MASTER_PROMPT,
                    "minimax_h3",
                    media_metadata,
                    target_duration_seconds=0.0,
                    minimax_h3_mode="ref2va",
                    minimax_h3_reference_manifest=self.manifest(),
                    minimax_h3_shot_count="1",
                    native_h3_output=True,
                    minimax_h3_dialogue_mode="off",
                )
                self.assertIn("planned for exactly 25.000 seconds", model_prompt)
                self.assertIn("about 120 to 160 words", model_prompt)
                self.assertIn("never fewer than 120 words", model_prompt)
                self.assertNotIn("about 60 to 100 words", model_prompt)

    def test_short_one_shot_prompt_no_longer_hits_dense_timeline_floor(self) -> None:
        sections = self.nodes._minimax_h3_ref_sections(self.valid_prompt())
        self.assertIsNotNone(sections)
        sections = dict(sections)
        sections["detailed_description"] = (
            "The video remains a high-contrast black-and-white high-fashion photography study with hard side light, deep shadows, "
            "blown highlights, and fine monochrome grain. "
            "[Shot 1] <Subject 1> holds the source-derived stance from <Picture 1> in a static medium shot, then marks the exact "
            "rap cadence of <Audio 1> with one controlled shoulder drop and a small head turn. Her short dark hair, feminine facial "
            "structure, black bodysuit, screen position, and tiled background remain stable as the movement settles into a resolved "
            "direct gaze on the final beat."
        )
        one_shot_prompt = self.nodes._rebuild_minimax_h3_ref_prompt(sections)

        short_reasons = self.nodes._minimax_h3_ref2va_validation_reasons(
            one_shot_prompt,
            5.0,
            self.user_prompt(),
            "auto_scene_audio",
            20_000,
            self.manifest(),
            "1",
            "off",
            2,
            "",
            0,
        )
        self.assertNotIn("minimax_h3_ref_detailed_description_short", short_reasons)

        long_reasons = self.nodes._minimax_h3_ref2va_validation_reasons(
            one_shot_prompt,
            25.0,
            self.user_prompt(),
            "auto_scene_audio",
            20_000,
            self.manifest(),
            "1",
            "off",
            2,
            "",
            0,
        )
        self.assertIn("minimax_h3_ref_detailed_description_short", long_reasons)

    def test_best_parse_valid_rejected_refinement_replaces_stale_initial_candidate(self) -> None:
        initial_prompt = "initial malformed candidate"
        improved_prompt = "improved candidate with one remaining camera issue"
        worse_prompt = "second candidate with two remaining issues"

        def packet(prompt: str) -> str:
            return json.dumps(
                {
                    "ltx_prompt": "",
                    "ideogram_prompt": "",
                    "minimax_h3_prompt": prompt,
                    "negative_prompt": "",
                    "scene_segments": [],
                    "metadata": {},
                }
            )

        initial_reasons = [
            "minimax_h3_ref_missing_detailed_description",
            "minimax_h3_ref_subject_definition_invalid",
            "minimax_h3_ref_style_opening_missing",
            "minimax_h3_ref_detailed_description_short",
        ]

        def validation_reasons(candidate: str, *_args, **_kwargs) -> list[str]:
            if candidate == improved_prompt:
                return ["minimax_h3_shot_camera_unspecified"]
            if candidate == worse_prompt:
                return [
                    "minimax_h3_shot_camera_unspecified",
                    "minimax_h3_cut_transition_invalid",
                ]
            return list(initial_reasons)

        context = self.nodes.GemmaContext(
            user_prompt=self.user_prompt(),
            images=None,
            source="minimax_h3_ref2va",
            media_metadata={
                "source": "minimax_h3_ref2va",
                "duration_seconds": 25.0,
                "reference_image_count": 1,
                "minimax_h3_reference_image_batch_count": 1,
                "minimax_h3_reference_manifest": self.manifest(),
                "minimax_h3_expected_subject_count": 0,
            },
        )
        target = self.nodes.TargetProfileConfig(
            target_profile="minimax_h3",
            audio_mode="auto_scene_audio",
            target_duration_seconds=25.0,
            minimax_h3_mode="ref2va",
            minimax_h3_shot_count="15",
            minimax_h3_dialogue_mode="off",
        )
        runtime = self.nodes.RuntimeConfig(
            model_path="unused-local-checkpoint",
            backend="transformers_inprocess",
            dtype="auto",
            quantization="none",
            local_files_only=True,
            unload_policy="unload_after_run",
            max_memory_gb=20.0,
            status={"ready": True, "supports_pixels": False},
        )
        with patch.object(
            self.nodes,
            "_run_backend_for_packet",
            side_effect=[packet(initial_prompt), packet(improved_prompt), packet(worse_prompt)],
        ), patch.object(
            self.nodes,
            "_minimax_h3_prompt_validation_reasons",
            side_effect=validation_reasons,
        ):
            result = self.nodes._run_generation_packet_legacy(
                runtime,
                context,
                target,
                runtime_required=True,
                max_output_chars=20_000,
                manage_unload=False,
                max_refinement_attempts_override=2,
            )

        self.assertEqual(result[0]["minimax_h3_prompt"], improved_prompt)
        metadata = result[-1]
        self.assertFalse(metadata["ready_for_generation"])
        self.assertEqual(metadata["blocked_reasons"], ["minimax_h3_shot_camera_unspecified"])
        refinement = metadata["minimax_h3_refinement"]
        self.assertFalse(refinement["accepted"])
        self.assertTrue(refinement["applied_best_rejected_candidate"])
        self.assertEqual(refinement["applied_best_rejected_candidate_attempt"], 1)
        self.assertEqual(
            refinement["applied_best_rejected_candidate_reasons"],
            ["minimax_h3_shot_camera_unspecified"],
        )

    def test_valid_contract_is_byte_stable(self) -> None:
        prompt = self.valid_prompt()
        self.assertEqual(self.reasons(prompt), [])
        self.assertEqual(
            self.nodes._repair_minimax_h3_ref2va_structure(
                prompt,
                self.user_prompt(),
                self.manifest(),
                15.0,
            ),
            prompt,
        )

    def test_missing_manifest_assets_are_restored_without_model_retry(self) -> None:
        sections = self.nodes._minimax_h3_ref_sections(self.valid_prompt())
        self.assertIsNotNone(sections)
        sections = dict(sections)
        original_subject = sections["subject_definitions"].split("\n<Audio 1>:", 1)[0]
        sections["subject_definitions"] = original_subject.replace(
            " as seen in <Picture 1>",
            "",
            1,
        )
        candidate = self.nodes._rebuild_minimax_h3_ref_prompt(sections)
        self.assertIn("minimax_h3_ref_missing_asset_tag", self.reasons(candidate))

        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            self.user_prompt(),
            self.manifest(),
            15.0,
        )

        repaired_sections = self.nodes._minimax_h3_ref_sections(repaired)
        self.assertIsNotNone(repaired_sections)
        self.assertTrue(
            repaired_sections["subject_definitions"].startswith(
                sections["subject_definitions"]
            )
        )
        self.assertIn(
            "<Picture 1>: sole visual reference for visible subject",
            repaired_sections["subject_definitions"],
        )
        self.assertIn(
            "<Audio 1>: locked selected soundtrack excerpt",
            repaired_sections["subject_definitions"],
        )
        self.assertIn("<Picture 1>: attribute_transfer -", repaired)
        self.assertIn("<Audio 1>: fully_copy -", repaired)
        self.assertEqual(self.reasons(repaired), [])
        self.assertEqual(
            self.nodes._repair_minimax_h3_ref2va_structure(
                repaired,
                self.user_prompt(),
                self.manifest(),
                15.0,
            ),
            repaired,
        )

    def test_manifest_completion_does_not_legalize_unknown_prompt_assets(self) -> None:
        candidate = self.valid_prompt().replace(
            "against the white tiles",
            "beside the undeclared prop from <Picture 2> against the white tiles",
            1,
        )
        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            self.user_prompt(),
            self.manifest(),
            15.0,
        )
        self.assertIn("minimax_h3_ref_undefined_tag", self.reasons(repaired))

    def test_missing_locked_audio_retention_defaults_to_fully_copy(self) -> None:
        sections = self.nodes._minimax_h3_ref_sections(self.valid_prompt())
        self.assertIsNotNone(sections)
        sections = dict(sections)
        sections["retention_analysis"] = sections["retention_analysis"].split(
            "\n<Audio 1>:",
            1,
        )[0]
        candidate = self.nodes._rebuild_minimax_h3_ref_prompt(sections)
        self.assertIn("minimax_h3_ref_retention_invalid", self.reasons(candidate))

        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            self.user_prompt(),
            self.manifest(),
            15.0,
        )

        repaired_sections = self.nodes._minimax_h3_ref_sections(repaired)
        self.assertIsNotNone(repaired_sections)
        self.assertIn(
            "<Audio 1>: fully_copy - A high-energy rap soundtrack",
            repaired_sections["retention_analysis"],
        )
        self.assertNotIn("minimax_h3_ref_retention_invalid", self.reasons(repaired))

    def test_nonmonotonic_cut_repairs_only_conflicting_timestamp(self) -> None:
        candidate = self.valid_prompt().replace(
            "[Shot 3] At 00:09.000,",
            "[Shot 3] At 00:04.500,",
            1,
        )
        self.assertIn("minimax_h3_cut_timestamp_invalid", self.reasons(candidate))

        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            self.user_prompt(),
            self.manifest(),
            15.0,
        )

        self.assertIn("[Shot 2] At 00:04.500,", repaired)
        self.assertIn("[Shot 3] At 00:08.500,", repaired)
        self.assertIn("[Shot 4] At 00:12.500,", repaired)
        self.assertEqual(repaired.count("performs dynamic dance movements"), 1)
        self.assertEqual(self.reasons(repaired), [])

    def test_out_of_range_authored_cut_remains_fail_closed(self) -> None:
        candidate = self.valid_prompt().replace(
            "[Shot 3] At 00:09.000,",
            "[Shot 3] At 00:15.000,",
            1,
        )
        self.assertIn("minimax_h3_cut_timestamp_out_of_range", self.reasons(candidate))

        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            self.user_prompt(),
            self.manifest(),
            15.0,
        )

        self.assertEqual(repaired, candidate)
        self.assertIn("minimax_h3_cut_timestamp_out_of_range", self.reasons(repaired))

    def test_ambiguous_multi_subject_usage_remains_fail_closed(self) -> None:
        sections = self.nodes._minimax_h3_ref_sections(self.valid_prompt())
        self.assertIsNotNone(sections)
        sections = copy.deepcopy(sections)
        sections["subject_definitions"] += (
            "\n<Subject 2>: A newly designed mirrored mannequin that remains separate from the referenced performer."
        )
        sections["retention_analysis"] += (
            "\n<Subject 2>: fully_preserved - Its mirrored surface and mannequin proportions remain stable."
        )
        sections["detailed_description"] = sections["detailed_description"].replace(
            "<Subject 1>",
            "the performer",
        )
        candidate = self.nodes._rebuild_minimax_h3_ref_prompt(sections)

        repaired = self.nodes._repair_minimax_h3_ref2va_structure(
            candidate,
            self.user_prompt(),
            self.manifest(),
            15.0,
        )

        repaired_sections = self.nodes._minimax_h3_ref_sections(repaired)
        self.assertIsNotNone(repaired_sections)
        self.assertNotIn("<Subject 1>", repaired_sections["detailed_description"])
        self.assertNotIn("<Subject 2>", repaired_sections["detailed_description"])
        self.assertIn("minimax_h3_ref_subject_usage_invalid", self.reasons(repaired))


if __name__ == "__main__":
    unittest.main()
