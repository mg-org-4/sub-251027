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
    package_name = f"diffusiongemma_model_targets_{uuid.uuid4().hex}"
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


class ModelTargetProfileNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()

    def test_each_model_node_exposes_only_relevant_controls(self) -> None:
        ltx = self.nodes.DiffusionGemmaLTX25TargetProfile.INPUT_TYPES()
        self.assertEqual(
            list(ltx["required"]),
            [
                "generation_mode",
                "target_duration_seconds",
                "style_guidance",
                "audio_mode",
                "audio_guidance",
                "negative_prompt_mode",
                "negative_prompt_guidance",
                "long_horizon_mode",
            ],
        )
        self.assertEqual(list(ltx["optional"]), ["camera_capability"])
        camera_choices, camera_config = ltx["optional"]["camera_capability"]
        self.assertEqual(
            camera_choices,
            ["Stable / base model", "Advanced / controlled camera"],
        )
        self.assertEqual(camera_config["default"], "Stable / base model")

        h3 = self.nodes.DiffusionGemmaMiniMaxH3TargetProfile.INPUT_TYPES()
        self.assertEqual(
            list(h3["required"]),
            [
                "generation_mode",
                "target_duration_seconds",
                "audio_mode",
                "audio_guidance",
                "shot_count",
                "custom_shot_count",
                "dialogue_mode",
                "dialogue_line_count",
                "dialogue_guidance",
                "negative_prompt_mode",
                "negative_prompt_guidance",
            ],
        )
        self.assertEqual(list(h3["optional"]), ["shot_count_override"])
        override_type, override_config = h3["optional"]["shot_count_override"]
        self.assertEqual(override_type, "INT")
        self.assertEqual(override_config["default"], 0)
        self.assertIn("native [Shot N]", override_config["tooltip"])
        self.assertIn("never connect a Project Master generation-lane count", override_config["tooltip"])

        ideogram = self.nodes.DiffusionGemmaIdeogram4TargetProfile.INPUT_TYPES()
        self.assertEqual(
            list(ideogram["required"]),
            [
                "aspect_ratio",
                "render_style",
                "exact_text",
                "json_output",
                "negative_prompt_mode",
                "negative_prompt_guidance",
            ],
        )
        self.assertNotIn("optional", ideogram)
        self.assertNotIn("audio_mode", ideogram["required"])
        self.assertNotIn("audio_guidance", ideogram["required"])
        self.assertNotIn("target_duration_seconds", ideogram["required"])

    def test_ltx_node_builds_the_same_canonical_config_as_the_legacy_surface(self) -> None:
        config, payload_json = self.nodes.DiffusionGemmaLTX25TargetProfile().build(
            generation_mode="Image to video",
            target_duration_seconds=8.0,
            style_guidance="restrained live-action thriller",
            audio_mode="explicit_sound_design",
            audio_guidance="dry footsteps and one exact off-screen line",
            negative_prompt_mode="custom",
            negative_prompt_guidance="no captions",
        )
        legacy, legacy_json = self.nodes.DiffusionGemmaTargetProfile().build(
            target_profile="ltx",
            audio_mode="explicit_sound_design",
            audio_guidance="dry footsteps and one exact off-screen line",
            target_duration_seconds=8.0,
            ltx_style="restrained live-action thriller",
            ideogram_aspect_ratio="1:1",
            ideogram_render_style="",
            ideogram_exact_text="",
            ideogram_json_output=True,
            negative_prompt_mode="custom",
            negative_prompt_guidance="no captions",
            ltx_generation_mode="Image to video",
        )
        self.assertEqual(config, legacy)
        self.assertEqual(json.loads(payload_json), json.loads(legacy_json))
        self.assertEqual(config.target_profile, "ltx")
        self.assertEqual(config.ltx_generation_mode, "image_to_video")
        self.assertEqual(config.ltx_camera_capability, "stable")

    def test_h3_node_preserves_audio_shot_dialogue_and_negative_controls(self) -> None:
        config, payload_json = self.nodes.DiffusionGemmaMiniMaxH3TargetProfile().build(
            generation_mode="ref2va",
            target_duration_seconds=20.0,
            audio_mode="explicit_sound_design",
            audio_guidance="wind and measured footsteps",
            shot_count="custom",
            custom_shot_count=17,
            dialogue_mode="required",
            dialogue_line_count=3,
            dialogue_guidance="English; protagonist S1; three terse lines",
            negative_prompt_mode="custom",
            negative_prompt_guidance="no subtitles",
        )
        payload = json.loads(payload_json)
        self.assertEqual(config.target_profile, "minimax_h3")
        self.assertEqual(config.minimax_h3_mode, "ref2va")
        self.assertEqual(config.minimax_h3_shot_count, "17")
        self.assertEqual(config.minimax_h3_dialogue_mode, "required")
        self.assertEqual(config.minimax_h3_dialogue_line_count, 3)
        self.assertEqual(payload["negative_prompt_guidance"], "no subtitles")

        overridden, overridden_json = self.nodes.DiffusionGemmaMiniMaxH3TargetProfile().build(
            generation_mode="ref2va",
            target_duration_seconds=25.0,
            audio_mode="auto_scene_audio",
            audio_guidance="locked soundtrack",
            shot_count="auto",
            custom_shot_count=12,
            dialogue_mode="off",
            dialogue_line_count=2,
            dialogue_guidance="",
            negative_prompt_mode="auto",
            negative_prompt_guidance="",
            shot_count_override=2,
        )
        self.assertEqual(overridden.minimax_h3_shot_count, "2")
        self.assertEqual(json.loads(overridden_json)["minimax_h3_shot_count"], "2")

        with self.assertRaisesRegex(ValueError, "conflicts with audio_mode=visual_only"):
            self.nodes.DiffusionGemmaMiniMaxH3TargetProfile().build(
                generation_mode="t2va",
                target_duration_seconds=5.0,
                audio_mode="visual_only",
                audio_guidance="",
                shot_count="auto",
                custom_shot_count=12,
                dialogue_mode="required",
                dialogue_line_count=1,
                dialogue_guidance="",
                negative_prompt_mode="auto",
                negative_prompt_guidance="",
            )

    def test_ideogram_node_has_visual_defaults_and_no_hidden_audio_request(self) -> None:
        config, payload_json = self.nodes.DiffusionGemmaIdeogram4TargetProfile().build(
            aspect_ratio="4:3",
            render_style="editorial poster illustration",
            exact_text="DIFFUSION GEMMA",
            json_output=True,
            negative_prompt_mode="custom",
            negative_prompt_guidance="no extra text or logos",
        )
        payload = json.loads(payload_json)
        self.assertEqual(config.target_profile, "ideogram4")
        self.assertEqual(config.audio_mode, "visual_only")
        self.assertEqual(config.audio_guidance, "")
        self.assertEqual(config.target_duration_seconds, 0.0)
        self.assertEqual(config.ideogram_aspect_ratio, "4:3")
        self.assertEqual(config.ideogram_exact_text, "DIFFUSION GEMMA")
        self.assertEqual(payload["ideogram_render_style"], "editorial poster illustration")

    def test_dedicated_nodes_share_the_downstream_type_and_legacy_stays_loadable(self) -> None:
        node_ids = {
            "DiffusionGemmaLTX25TargetProfile",
            "DiffusionGemmaMiniMaxH3TargetProfile",
            "DiffusionGemmaIdeogram4TargetProfile",
        }
        for node_id in node_ids:
            with self.subTest(node_id=node_id):
                node_class = self.nodes.NODE_CLASS_MAPPINGS[node_id]
                self.assertEqual(node_class.RETURN_TYPES, (self.nodes.TARGET_PROFILE_TYPE, "STRING"))
                self.assertEqual(
                    node_class.RETURN_NAMES,
                    ("target_profile_config", "target_profile_json"),
                )
                self.assertIn(node_id, self.nodes.NODE_DISPLAY_NAME_MAPPINGS)

        self.assertIn("DiffusionGemmaTargetProfile", self.nodes.NODE_CLASS_MAPPINGS)
        self.assertTrue(self.nodes.DiffusionGemmaTargetProfile.DEPRECATED)
        self.assertIn("Legacy", self.nodes.NODE_DISPLAY_NAME_MAPPINGS["DiffusionGemmaTargetProfile"])

    def test_ideogram_resolution_ignores_splitter_widget_override(self) -> None:
        context = self.nodes.GemmaContext()
        target = self.nodes._target_profile_config_to_dict(
            self.nodes.TargetProfileConfig(
                target_profile="ideogram4",
                ideogram_aspect_ratio="4:3",
            )
        )

        resolved = self.nodes._splitter_resolution_aspect_ratio(
            "16:9 (Widescreen)",
            context,
            target,
            "4:3",
        )

        self.assertEqual(resolved, ("4:3", "ideogram_target_profile"))

    def test_h3_auto_resolution_does_not_follow_reference_media_dimensions(self) -> None:
        context = self.nodes.GemmaContext(
            media_metadata={
                "width": 1920,
                "height": 1080,
                "ltx_first_frame_width": 1080,
                "ltx_first_frame_height": 1920,
            }
        )
        target = self.nodes._target_profile_config_to_dict(
            self.nodes.TargetProfileConfig(target_profile="minimax_h3")
        )

        resolved = self.nodes._splitter_resolution_aspect_ratio(
            self.nodes.RESOLUTION_SELECTOR_AUTO,
            context,
            target,
            "4:3",
        )

        self.assertEqual(resolved, ("4:3", "legacy_target_profile"))

    def test_ltx_auto_aliases_and_invalid_values_do_not_silently_select_square(self) -> None:
        context = self.nodes.GemmaContext()
        target = self.nodes._target_profile_config_to_dict(
            self.nodes.TargetProfileConfig(target_profile="ltx")
        )

        for requested in (
            "auto",
            "AUTO (TARGET/SOURCE)",
            "not-a-resolution",
        ):
            with self.subTest(requested=requested):
                aspect_ratio, source = self.nodes._splitter_resolution_aspect_ratio(
                    requested,
                    context,
                    target,
                    "1:1",
                )
                self.assertEqual((aspect_ratio, source), ("16:9", "ltx_default"))
                self.assertEqual(
                    self.nodes._resolution_selector_dimensions(aspect_ratio, 0.5, 32),
                    (960, 544),
                )


if __name__ == "__main__":
    unittest.main()
