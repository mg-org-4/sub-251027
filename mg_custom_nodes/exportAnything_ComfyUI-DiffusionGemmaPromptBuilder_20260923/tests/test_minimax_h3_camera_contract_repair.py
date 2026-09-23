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
    package_name = f"diffusiongemma_h3_camera_repair_{uuid.uuid4().hex}"
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


class MiniMaxH3CameraContractRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()

    @staticmethod
    def multi_shot_t2va_prompt() -> str:
        return (
            "integrated_multimodal_description: [Shot 1] A runner waits at a rain-darkened starting line. "
            "[Shot 2] At 00:06.000, the camera cuts to a low tracking view as the runner accelerates around the bend. "
            "[Shot 3] At 00:13.000, the camera cuts to the runner stopping beneath the finish light. "
            "The final frame holds on the runner standing still beneath the light.\n\n"
            "overall_soundscape: Rain, measured footfalls, and distant crowd ambience surround the track.\n\n"
            "non_diegetic_music: N/A"
        )

    @staticmethod
    def fully_specified_t2va_prompt() -> str:
        return (
            "integrated_multimodal_description: [Shot 1] A locked-off medium shot holds on a runner at the starting line. "
            "[Shot 2] At 00:06.000, the camera cuts to a low tracking view and tracks beside the accelerating runner. "
            "[Shot 3] At 00:13.000, the camera cuts to a static close view as the runner stops beneath the finish light.\n\n"
            "overall_soundscape: Rain and measured footfalls surround the track.\n\n"
            "non_diegetic_music: N/A"
        )

    @classmethod
    def ref2va_prompt(cls) -> str:
        timeline = cls.multi_shot_t2va_prompt().split(
            "integrated_multimodal_description: ", 1
        )[1].split("\n\noverall_soundscape:", 1)[0]
        return (
            "subject_definitions:\n"
            "<Subject 1>: A runner wearing a dark rain jacket and silver shoes.\n\n"
            "summary:\n"
            "[reference generation] A runner completes a measured lap in the rain.\n\n"
            "retention_analysis:\n"
            "<Subject 1>: fully_preserved - The face, dark jacket, silver shoes, and proportions remain unchanged.\n\n"
            "detailed_description:\n"
            "Naturalistic live-action sports photography uses cool rain-muted color and restrained film grain. "
            f"{timeline}\n\n"
            "overall_soundscape:\n"
            "Rain, measured footfalls, and distant crowd ambience surround the track.\n\n"
            "non_diegetic_music:\n"
            "N/A"
        )

    def test_multi_shot_repair_adds_only_safe_locked_holds_and_is_idempotent(self) -> None:
        prompt = self.multi_shot_t2va_prompt()
        repaired, repaired_shots = (
            self.nodes._repair_minimax_h3_unspecified_shot_cameras(prompt, "t2va")
        )

        self.assertEqual(repaired_shots, [1, 3])
        clause = self.nodes._MINIMAX_H3_SAFE_CAMERA_HOLD
        self.assertEqual(repaired.count(clause), 2)
        self.assertEqual(repaired.replace(f" {clause}", ""), prompt)
        self.assertIn(
            "the camera cuts to a low tracking view as the runner accelerates",
            repaired,
        )
        self.assertNotIn(
            "minimax_h3_shot_camera_unspecified",
            self.nodes._minimax_h3_prompt_validation_reasons(
                repaired,
                20.0,
                "A runner completes a measured lap in the rain.",
            ),
        )
        second_pass, second_pass_shots = (
            self.nodes._repair_minimax_h3_unspecified_shot_cameras(
                repaired,
                "t2va",
            )
        )
        self.assertEqual(second_pass, repaired)
        self.assertEqual(second_pass_shots, [])

    def test_already_specified_camera_choreography_is_byte_stable(self) -> None:
        prompt = self.fully_specified_t2va_prompt()
        repaired, repaired_shots = (
            self.nodes._repair_minimax_h3_unspecified_shot_cameras(prompt, "t2va")
        )

        self.assertEqual(repaired, prompt)
        self.assertEqual(repaired_shots, [])

    def test_expressive_h3_camera_choreography_is_recognized_and_byte_stable(self) -> None:
        prompt = (
            "integrated_multimodal_description: "
            "[Shot 1] A dancer crosses a mirrored floor. The camera swirls clockwise around her and rises into a high view. "
            "[Shot 2] At 00:05.000, the camera cuts to the dancer turning toward the lights. The camera rolls thirty degrees counterclockwise while tracking her. "
            "[Shot 3] At 00:10.000, the camera cuts to a line of foreground prisms. The viewpoint rotates around the dancer as pronounced layered parallax reveals the stage. "
            "[Shot 4] At 00:15.000, the camera cuts to the final beat. The camera whip-pans right, then cranes up along one continuous path to the ending wide frame.\n\n"
            "overall_soundscape: Footfalls and fabric movement follow the performance.\n\n"
            "non_diegetic_music: Percussion drives the four camera beats."
        )

        repaired, repaired_shots = (
            self.nodes._repair_minimax_h3_unspecified_shot_cameras(prompt, "t2va")
        )

        self.assertEqual(repaired, prompt)
        self.assertEqual(repaired_shots, [])
        self.assertNotIn(self.nodes._MINIMAX_H3_SAFE_CAMERA_HOLD, repaired)
        self.assertNotIn(
            "minimax_h3_shot_camera_unspecified",
            self.nodes._minimax_h3_prompt_validation_reasons(
                repaired,
                20.0,
                "A dynamic four-shot dance with orbiting, rolling, whip movement, and pronounced parallax.",
                minimax_h3_mode="t2va",
                minimax_h3_shot_count="4",
            ),
        )

    def test_empty_shot_body_stays_fail_closed(self) -> None:
        prompt = self.multi_shot_t2va_prompt().replace(
            "[Shot 1] A runner waits at a rain-darkened starting line. ",
            "[Shot 1] ",
        )
        repaired, repaired_shots = (
            self.nodes._repair_minimax_h3_unspecified_shot_cameras(prompt, "t2va")
        )

        self.assertNotIn(1, repaired_shots)
        self.assertIn(
            "minimax_h3_empty_visual_timeline",
            self.nodes._minimax_h3_prompt_validation_reasons(
                repaired,
                20.0,
                "A runner completes a measured lap in the rain.",
            ),
        )

    def test_ref2va_repair_is_confined_to_detailed_description(self) -> None:
        prompt = self.ref2va_prompt()
        original_sections = self.nodes._minimax_h3_ref_sections(prompt)
        self.assertIsNotNone(original_sections)

        repaired, repaired_shots = (
            self.nodes._repair_minimax_h3_unspecified_shot_cameras(
                prompt,
                "ref2va",
            )
        )
        repaired_sections = self.nodes._minimax_h3_ref_sections(repaired)
        self.assertIsNotNone(repaired_sections)
        self.assertEqual(repaired_shots, [1, 3])
        for field_name in self.nodes._MINIMAX_H3_REF_FIELDS:
            if field_name != "detailed_description":
                self.assertEqual(
                    repaired_sections[field_name],
                    original_sections[field_name],
                )
        self.assertTrue(
            repaired_sections["detailed_description"].startswith(
                "Naturalistic live-action sports photography"
            )
        )

    def test_strict_packet_splitter_repairs_before_readiness_validation(self) -> None:
        prompt = self.multi_shot_t2va_prompt()
        packet = {
            "ltx_prompt": "",
            "ideogram_prompt": "",
            "minimax_h3_prompt": prompt,
            "negative_prompt": "",
            "scene_segments": [],
            "metadata": {},
        }
        context = self.nodes.GemmaContext(
            user_prompt="A runner completes a measured lap in the rain.",
            images=None,
            source="text",
            media_metadata={"source": "text", "duration_seconds": 20.0},
        )
        target = self.nodes.TargetProfileConfig(
            target_profile="minimax_h3",
            audio_mode="auto_scene_audio",
            target_duration_seconds=20.0,
            minimax_h3_mode="t2va",
            minimax_h3_dialogue_mode="off",
        )

        result = self.nodes.DiffusionGemmaJSONSplitter().split(
            json.dumps(packet),
            context,
            target,
        )
        metadata = json.loads(result[4])
        repaired_prompt = result[13]

        self.assertTrue(result[12], metadata)
        self.assertEqual(
            metadata["minimax_h3_camera_contract_repair"],
            {
                "strategy": "locked_off_stable_view",
                "shots": [1, 3],
                "repair_count": 2,
            },
        )
        self.assertEqual(
            repaired_prompt.count(self.nodes._MINIMAX_H3_SAFE_CAMERA_HOLD),
            2,
        )
        self.assertNotIn(
            "minimax_h3_shot_camera_unspecified",
            metadata["blocked_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
