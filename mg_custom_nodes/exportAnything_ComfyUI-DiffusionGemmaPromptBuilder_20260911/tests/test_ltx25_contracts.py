from __future__ import annotations

import importlib.util
import json
import re
import sys
import unittest
import uuid
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]


def load_nodes_module():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    package_name = f"diffusiongemma_ltx25_contracts_{uuid.uuid4().hex}"
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


class Ltx25ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()
        cls.contract = sys.modules[f"{cls.nodes.__package__}.ltx25_contract"]

    @staticmethod
    def frame_ledger(*categories_and_claims: tuple[list[str], str]) -> dict:
        facts = []
        for index, (categories, claim) in enumerate(categories_and_claims, start=1):
            facts.append(
                {
                    "fact_id": f"fact-{index}",
                    "claim": claim,
                    "confidence": "high",
                    "categories": categories,
                    "evidence": [{"asset_id": "image:1", "sample_ordinal": 1}],
                }
            )
        return {
            "schema": "dg-grounding-ledger/1",
            "analysis_status": "grounded",
            "observed_facts": facts,
            "inferred_facts": [],
            "creative_additions": [],
            "uncertainties": [],
            "grounding_failure_reasons": [],
        }

    @classmethod
    def conditioned_metadata(cls, *, first: bool = True, last: bool = False, ledger=None) -> dict:
        metadata = {
            "source": "image" if first or last else "none",
            "ltx25_context_schema": cls.nodes.LTX25_CONTRACT_SCHEMA,
            "ltx_first_frame_attached": first,
            "ltx_last_frame_attached": last,
            "ltx_frame_pair_attached": first and last,
        }
        if ledger is not None:
            metadata["verified_grounding_ledger"] = ledger
        return metadata

    @staticmethod
    def prompt_packet(*, ltx_prompt: str = "", ideogram_prompt: str = "") -> str:
        return json.dumps(
            {
                "ltx_prompt": ltx_prompt,
                "ideogram_prompt": ideogram_prompt,
                "minimax_h3_prompt": "",
                "negative_prompt": "",
                "scene_segments": [],
                "metadata": {"json_parse_valid": True},
            }
        )

    def test_mode_normalization_accepts_ui_labels_aliases_and_legacy_video(self) -> None:
        cases = {
            None: "auto",
            "Auto (recommended)": "auto",
            "T2V": "text_to_video",
            "Text to video": "text_to_video",
            "I2V": "image_to_video",
            "Image-to-video": "image_to_video",
            "FLF": "first_last_frame",
            "First + last frame": "first_last_frame",
            "legacy video": "legacy_video",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(self.nodes.normalize_ltx25_generation_mode(value), expected)

    def test_auto_mode_resolution_distinguishes_t2v_i2v_flf_and_legacy_video(self) -> None:
        cases = (
            ({"source": "none"}, 0, "text_to_video", "no_conditioning_frames"),
            (
                self.conditioned_metadata(first=True),
                1,
                "image_to_video",
                "attached_frames",
            ),
            (
                self.conditioned_metadata(first=True, last=True),
                2,
                "first_last_frame",
                "attached_frames",
            ),
            ({"source": "video"}, 8, "legacy_video", "legacy_video_context"),
        )
        for metadata, image_count, expected_mode, expected_source in cases:
            with self.subTest(expected_mode=expected_mode):
                report = self.nodes.resolve_ltx25_generation_mode("auto", metadata, image_count)
                self.assertEqual(report["resolved_mode"], expected_mode)
                self.assertEqual(report["resolution_source"], expected_source)

        generic_image = self.nodes.resolve_ltx25_generation_mode(
            "auto", {"source": "image"}, image_count=1
        )
        self.assertEqual(generic_image["resolved_mode"], "image_to_video")
        self.assertIn("ltx_mode_inferred_from_generic_image_context", generic_image["warnings"])

    def test_explicit_mode_overrides_context_hint_but_reports_the_mismatch(self) -> None:
        metadata = self.conditioned_metadata(first=True)
        metadata["ltx_generation_mode_hint"] = "image_to_video"
        report = self.nodes.resolve_ltx25_generation_mode("Text to video", metadata, 1)
        self.assertEqual(report["resolved_mode"], "text_to_video")
        self.assertEqual(report["resolution_source"], "target_profile")
        self.assertIn("ltx_explicit_mode_overrides_context_hint", report["warnings"])

    def test_context_hub_appends_last_frame_socket_without_reordering_legacy_widgets(self) -> None:
        inputs = self.nodes.DiffusionGemmaContextHub.INPUT_TYPES()
        self.assertEqual(
            list(inputs["required"]),
            [
                "user_prompt",
                "sample_fps",
                "max_duration_seconds",
                "max_frames",
                "overlong_policy",
                "include_image_with_video",
            ],
        )
        self.assertEqual(
            list(inputs["optional"]),
            ["image", "video", "visual_description", "media_synthesis_mode", "last_frame_image"],
        )

        # This is the complete positional signature saved before last_frame_image
        # was appended. It must continue to bind media_synthesis_mode correctly.
        context, context_json, _preview = self.nodes.DiffusionGemmaContextHub().build(
            "A person waits beside a window.",
            1.0,
            60.0,
            60,
            "trim",
            True,
            None,
            None,
            "",
            "video_recreation",
        )
        payload = json.loads(context_json)
        self.assertEqual(context.user_prompt, "A person waits beside a window.")
        self.assertEqual(context.media_metadata["ltx_generation_mode_hint"], "text_to_video")
        self.assertFalse(context.media_metadata["ltx_first_frame_attached"])
        self.assertFalse(context.media_metadata["ltx_last_frame_attached"])
        self.assertEqual(payload["media"]["ltx_generation_mode_hint"], "text_to_video")

    def test_context_hub_emits_first_and_last_frame_roles_and_geometry(self) -> None:
        first = torch.zeros((1, 12, 20, 3), dtype=torch.float32)
        last = torch.ones((1, 8, 14, 3), dtype=torch.float32)

        i2v_context, _json, _preview = self.nodes.DiffusionGemmaContextHub().build(
            "She turns toward the doorway.", image=first
        )
        self.assertEqual(i2v_context.media_metadata["ltx_generation_mode_hint"], "image_to_video")
        self.assertEqual(i2v_context.media_metadata["ltx_frame_roles"], ["first_frame"])
        self.assertEqual(i2v_context.media_metadata["ltx_first_frame_width"], 20)
        self.assertEqual(i2v_context.media_metadata["ltx_first_frame_height"], 12)
        self.assertEqual(tuple(i2v_context.images.shape), (1, 12, 20, 3))

        flf_context, _json, _preview = self.nodes.DiffusionGemmaContextHub().build(
            "She crosses the room and reaches the doorway.",
            image=first,
            last_frame_image=last,
        )
        metadata = flf_context.media_metadata
        self.assertEqual(metadata["ltx_generation_mode_hint"], "first_last_frame")
        self.assertTrue(metadata["ltx_frame_pair_attached"])
        self.assertEqual(metadata["ltx_frame_roles"], ["first_frame", "last_frame"])
        self.assertEqual(metadata["reference_image_count"], 2)
        self.assertTrue(metadata["reference_image_backend_attached"])
        self.assertEqual(metadata["ltx_first_frame_width"], 20)
        self.assertEqual(metadata["ltx_first_frame_height"], 12)
        self.assertEqual(metadata["ltx_last_frame_width"], 14)
        self.assertEqual(metadata["ltx_last_frame_height"], 8)
        self.assertEqual(tuple(flf_context.images.shape), (2, 12, 20, 3))

    def test_each_mode_has_a_unique_compiler_contract_in_the_model_prompt(self) -> None:
        modes_and_markers = {
            "text_to_video": "This is text-to-video with no conditioning frame.",
            "image_to_video": "This is image-to-video.",
            "first_last_frame": "This is first-and-last-frame video.",
            "legacy_video": "This is a legacy source-video recreation",
        }
        contracts = {}
        for mode, marker in modes_and_markers.items():
            with self.subTest(mode=mode):
                contract = self.nodes.ltx25_compiler_contract(mode, 8.0)
                contracts[mode] = contract
                self.assertIn(marker, contract)
                self.assertIn("The clip lasts 8.000 seconds.", contract)
                self.assertRegex(
                    contract,
                    r"\bphysically continuous(?: subject-relative)? path\b",
                )
                self.assertIn("Compatible simultaneous or sequential motion phases", contract)
                self.assertNotIn("camera move(s)", contract)

                metadata = {"source": "none"}
                if mode == "image_to_video":
                    metadata = self.conditioned_metadata(first=True)
                elif mode == "first_last_frame":
                    metadata = self.conditioned_metadata(first=True, last=True)
                elif mode == "legacy_video":
                    metadata = {"source": "video", "sampled_frame_count": 4}
                model_prompt = self.nodes._build_model_prompt(
                    "A concise test scene.",
                    "Return only the requested JSON.",
                    "ltx",
                    metadata,
                    target_duration_seconds=8.0,
                    ltx_generation_mode=mode,
                )
                self.assertIn(marker, model_prompt)
                self.assertNotIn("max_camera_moves", model_prompt)
        self.assertEqual(len(set(contracts.values())), len(contracts))

    def test_ltx_model_output_shape_keeps_host_owned_fields_compact(self) -> None:
        model_prompt = self.nodes._build_model_prompt(
            "A concise test scene.",
            "Return only the requested JSON.",
            "ltx",
            self.conditioned_metadata(first=True),
            target_duration_seconds=5.0,
            ltx_generation_mode="image_to_video",
        )
        shape_text = model_prompt.split(
            "Return the final answer as JSON matching this shape:\n", 1
        )[1].split("\n\nCritical output rules:", 1)[0]
        output_shape = json.loads(shape_text)

        self.assertIsInstance(output_shape["ltx_prompt"], str)
        self.assertTrue(output_shape["ltx_prompt"])
        self.assertEqual(output_shape["ideogram_prompt"], "")
        self.assertEqual(output_shape["minimax_h3_prompt"], "")
        self.assertEqual(output_shape["scene_segments"], [])
        self.assertEqual(output_shape["metadata"], {})
        self.assertNotIn('"controls"', shape_text)

        verified_metadata = self.conditioned_metadata(
            first=True,
            ledger=self.frame_ledger(
                (["identity", "appearance"], "An officer in a dark uniform is visible."),
            ),
        )
        verified_metadata.update(
            {
                "grounding_evidence_report_id": "dggr-test",
                "grounding_required_asset_ids": ["image:1"],
            }
        )
        verified_prompt = self.nodes._build_model_prompt(
            "The officer takes one measured step.",
            "Return only the requested JSON.",
            "ltx",
            verified_metadata,
            target_duration_seconds=5.0,
            ltx_generation_mode="image_to_video",
        )
        verified_shape = json.loads(
            verified_prompt.split(
                "Return the final answer as JSON matching this shape:\n", 1
            )[1].split("\n\nCritical output rules:", 1)[0]
        )
        self.assertEqual(
            set(verified_shape["metadata"]),
            {"grounding_evidence_report_id", "used_grounding_fact_ids"},
        )

    def test_audit_contract_scales_fact_limit_to_attached_assets(self) -> None:
        one_image = {
            "assets": [
                {
                    "asset_id": "image:1",
                    "kind": "image",
                    "samples": [{"sample_ordinal": 1}],
                }
            ]
        }
        prompt = self.nodes._audit_grounding_master_prompt(one_image)
        self.assertIn("Limit observed facts to the 4 most useful facts", prompt)
        self.assertNotIn("24 most useful facts", prompt)

        four_images = {
            "assets": [
                {
                    "asset_id": f"image:{index}",
                    "kind": "image",
                    "samples": [{"sample_ordinal": index}],
                }
                for index in range(1, 5)
            ]
        }
        self.assertIn(
            "Limit observed facts to the 8 most useful facts",
            self.nodes._audit_grounding_master_prompt(four_images),
        )

    def test_i2v_context_image_roles_are_sanitized_for_the_model_prompt(self) -> None:
        first = torch.zeros((1, 12, 20, 3), dtype=torch.float32)
        context, _context_json, _preview = self.nodes.DiffusionGemmaContextHub().build(
            "She slowly turns toward the doorway.",
            image=first,
        )
        self.assertIn("[dg:", context.media_metadata["image_roles"][0])

        model_prompt = self.nodes._build_model_prompt(
            context.user_prompt,
            "Return only the requested JSON.",
            "ltx",
            context.media_metadata,
            target_duration_seconds=5.0,
            ltx_generation_mode="auto",
        )

        self.assertIn("LTX first-frame anchor", model_prompt)
        self.assertNotIn("[dg:", model_prompt)
        self.assertIn("[dg:", context.media_metadata["image_roles"][0])

    def test_conditioned_modes_report_missing_anchors(self) -> None:
        i2v_report = self.nodes.resolve_ltx25_generation_mode(
            "image_to_video", self.conditioned_metadata(first=False), 0
        )
        i2v = self.nodes.validate_ltx25_prompt(
            "A static medium shot holds on the empty room.", i2v_report, {}, 5.0
        )
        self.assertIn("ltx_i2v_first_frame_missing", i2v["reasons"])
        self.assertFalse(i2v["ready"])

        flf_report = self.nodes.resolve_ltx25_generation_mode(
            "first_last_frame", self.conditioned_metadata(first=False, last=False), 0
        )
        flf = self.nodes.validate_ltx25_prompt(
            "A static medium shot holds on the empty room.", flf_report, {}, 5.0
        )
        self.assertIn("ltx_flf_first_frame_missing", flf["reasons"])
        self.assertIn("ltx_flf_last_frame_missing", flf["reasons"])

        first_only_metadata = self.conditioned_metadata(first=True, last=False)
        first_only_report = self.nodes.resolve_ltx25_generation_mode(
            "first_last_frame", first_only_metadata, 1
        )
        first_only = self.nodes.validate_ltx25_prompt(
            "A static close-up holds on the subject.", first_only_report, first_only_metadata, 5.0
        )
        self.assertNotIn("ltx_flf_first_frame_missing", first_only["reasons"])
        self.assertIn("ltx_flf_last_frame_missing", first_only["reasons"])

    def test_i2v_rejects_cuts_and_multiple_shots(self) -> None:
        metadata = self.conditioned_metadata(first=True)
        mode_report = self.nodes.resolve_ltx25_generation_mode("image_to_video", metadata, 1)
        validation = self.nodes.validate_ltx25_prompt(
            "A static close-up holds on the woman. A hard cut to a wide shot reveals the street.",
            mode_report,
            metadata,
            8.0,
        )
        self.assertIn("ltx_conditioned_mode_requires_single_continuous_take", validation["reasons"])
        self.assertIn("ltx_shot_count_exceeds_duration_budget", validation["reasons"])
        self.assertGreater(validation["complexity"]["observed"]["cut_count"], 0)

    def test_duration_budgets_scale_t2v_but_keep_conditioned_modes_to_one_take(self) -> None:
        expected_t2v_shots = {4.0: 1, 8.0: 2, 12.0: 3, 15.0: 4}
        for duration, expected_shots in expected_t2v_shots.items():
            with self.subTest(duration=duration):
                budget = self.nodes.ltx25_complexity_budget("text_to_video", duration)
                self.assertEqual(budget["max_shots"], expected_shots)
                self.assertEqual(budget["max_cuts"], expected_shots - 1)

        for mode in ("image_to_video", "first_last_frame"):
            budget = self.nodes.ltx25_complexity_budget(mode, 20.0)
            self.assertEqual(budget["max_shots"], 1)
            self.assertEqual(budget["max_cuts"], 0)
            self.assertNotIn("max_camera_moves", budget)
            self.assertEqual(budget["recommended_camera_motion_phases"], 4)

        terminal_black = self.contract.analyze_ltx25_prompt_complexity(
            "A static medium shot at eye level holds on the woman, then the image cuts to black.",
            "text_to_video",
            4.0,
        )
        self.assertEqual(terminal_black["observed"]["cut_count"], 0)
        self.assertEqual(terminal_black["observed"]["estimated_shots"], 1)
        self.assertNotIn("ltx_shot_scale_missing", terminal_black["reasons"])
        self.assertNotIn("ltx_camera_state_missing", terminal_black["reasons"])
        self.assertNotIn("ltx_viewpoint_missing", terminal_black["reasons"])

        overloaded = self.contract.analyze_ltx25_prompt_complexity(
            (
                "A wide shot shows a runner. A hard cut to a medium shot follows as the camera pans, "
                "then dollies, then tilts. She turns, walks, runs, jumps, falls, opens a gate, enters, "
                "exits, and disappears. Music, footsteps, heartbeat, wind, rain, and a siren overlap. "
                'A narrator says, "There is far too much dialogue here to fit clearly in four seconds."'
            ),
            "text_to_video",
            4.0,
        )
        self.assertTrue(
            {
                "ltx_shot_count_exceeds_duration_budget",
                "ltx_action_density_exceeds_duration_budget",
                "ltx_audio_density_exceeds_duration_budget",
                "ltx_speech_exceeds_duration_budget",
            }.issubset(set(overloaded["reasons"]))
        )
        self.assertNotIn("ltx_camera_complexity_exceeds_duration_budget", overloaded["reasons"])
        self.assertIn("ltx_camera_complexity_at_duration_limit", overloaded["warnings"])

    def test_conditioned_viewpoint_continuity_is_not_mistaken_for_missing_angle(self) -> None:
        live_caption = (
            "In a medium close-up, an officer stands by a roadside during daylight. "
            "The first frame shows him centered beneath soft overcast light. "
            "The camera starts with this static framing and begins a subtle stabilized push. "
            "He shifts his weight and takes one measured step while roadside ambience continues."
        )
        i2v = self.contract.analyze_ltx25_prompt_complexity(
            live_caption, "image_to_video", 5.0
        )
        self.assertEqual(i2v["observed"]["conditioned_viewpoint_anchor_count"], 1)
        self.assertNotIn("ltx_viewpoint_missing", i2v["reasons"])

        t2v = self.contract.analyze_ltx25_prompt_complexity(
            live_caption, "text_to_video", 5.0
        )
        self.assertEqual(t2v["observed"]["conditioned_viewpoint_anchor_count"], 0)
        self.assertIn("ltx_viewpoint_missing", t2v["reasons"])

        unanchored_i2v = self.contract.analyze_ltx25_prompt_complexity(
            "A static medium close-up centers an officer beside a road.",
            "image_to_video",
            5.0,
        )
        self.assertNotIn("ltx_viewpoint_missing", unanchored_i2v["reasons"])

    def test_conditioned_viewpoint_is_inherited_but_t2v_stays_strict(self) -> None:
        caption = (
            "A medium shot frames an astronaut holding a lollipop while the camera remains static. "
            "He raises the candy as nearby creatures sway gently and soft music continues."
        )
        i2v_report = self.contract.resolve_ltx25_generation_mode(
            "image_to_video",
            self.conditioned_metadata(first=True),
            1,
        )
        i2v = self.contract.validate_ltx25_prompt(
            caption,
            i2v_report,
            self.conditioned_metadata(first=True),
            5.0,
        )
        self.assertNotIn("ltx_viewpoint_missing", i2v["reasons"])
        self.assertIn(
            "ltx_conditioned_viewpoint_inherited_from_frame",
            i2v["warnings"],
        )
        self.assertTrue(
            i2v["complexity"]["observed"][
                "conditioned_viewpoint_inherited_from_frame"
            ]
        )

        t2v_report = self.contract.resolve_ltx25_generation_mode(
            "text_to_video",
            {},
            0,
        )
        t2v = self.contract.validate_ltx25_prompt(caption, t2v_report, {}, 5.0)
        self.assertIn("ltx_viewpoint_missing", t2v["reasons"])

    def test_exact_first_frame_viewpoint_anchors_are_i2v_only(self) -> None:
        source_relative_prompts = (
            (
                "A static medium close-up preserves the exact supplied first-frame "
                "camera viewpoint while the officer watches the road."
            ),
            (
                "A medium shot preserves the same supplied first-frame angle as the "
                "camera makes one gentle push-in toward the officer."
            ),
            (
                "A medium close-up holds on the officer. The camera starts from the "
                "exact supplied first-frame framing and remains static."
            ),
        )

        for prompt in source_relative_prompts:
            with self.subTest(prompt=prompt, mode="image_to_video"):
                i2v = self.contract.analyze_ltx25_prompt_complexity(
                    prompt,
                    "image_to_video",
                    5.0,
                )
                self.assertEqual(
                    i2v["observed"]["conditioned_viewpoint_anchor_count"],
                    1,
                )
                self.assertNotIn("ltx_viewpoint_missing", i2v["reasons"])

            with self.subTest(prompt=prompt, mode="text_to_video"):
                t2v = self.contract.analyze_ltx25_prompt_complexity(
                    prompt,
                    "text_to_video",
                    5.0,
                )
                self.assertEqual(
                    t2v["observed"]["conditioned_viewpoint_anchor_count"],
                    0,
                )
                self.assertIn("ltx_viewpoint_missing", t2v["reasons"])

    def test_medium_close_shot_is_a_valid_ltx_shot_scale(self) -> None:
        live_prompt = (
            "The shot opens with a medium close shot of a middle-aged male officer with short light "
            "brown hair, wearing a professional black police uniform with a radio microphone clipped "
            "to his shoulder. He stands on a paved roadside with a dirt lot and residential houses in "
            "the far background under soft, natural daylight. The camera is static at eye level, "
            "focusing on him from the chest up. The officer shifts his weight from one foot to the "
            "other, his expression turning tense as he turns his head toward something just off-camera "
            "to the right. He takes a single, measured step backward on the gravel, his eyes widening "
            "as he tracks an unseen threat. Suddenly, a man-sized, fuzzy muppet lunges into the frame "
            "from the right, colliding with the officer. The officer recoils in surprise as the "
            "muppet's arms grab his uniform. The sound of gravel crunching under boots is followed by "
            "a muffled fabric-like impact and the officer's grunt of surprise."
        )
        analysis = self.contract.analyze_ltx25_prompt_complexity(
            live_prompt,
            "image_to_video",
            10.0,
        )

        self.assertEqual(analysis["observed"]["estimated_shots"], 1)
        self.assertEqual(analysis["observed"]["explicit_shot_phrases"], 1)
        self.assertEqual(analysis["observed"]["shot_scale_count"], 1)
        self.assertNotIn("ltx_shot_scale_missing", analysis["reasons"])

        vague = self.contract.analyze_ltx25_prompt_complexity(
            "The camera is static at eye level in a medium close framing.",
            "image_to_video",
            10.0,
        )
        self.assertIn("ltx_shot_scale_missing", vague["reasons"])

    def test_i2v_continuous_scale_evolution_has_an_explicit_opening_scale(self) -> None:
        metadata = self.conditioned_metadata(first=True)
        mode_report = self.nodes.resolve_ltx25_generation_mode(
            "image_to_video",
            metadata,
            1,
        )
        prompt = (
            "One continuous take starts as a wide eye-level shot from the exact supplied "
            "first-frame camera viewpoint. The camera performs a slow dolly zoom toward the "
            "subject while preserving continuous space and motion, ending in a medium-close "
            "shot without a cut."
        )

        validation = self.nodes.validate_ltx25_prompt(
            prompt,
            mode_report,
            metadata,
            8.0,
        )

        self.assertNotIn("ltx_shot_scale_missing", validation["reasons"])
        self.assertTrue(validation["ready"], validation["reasons"])
        self.assertEqual(validation["complexity"]["observed"]["estimated_shots"], 1)

    def test_i2v_missing_opening_scale_is_inherited_only_with_attached_frame(self) -> None:
        metadata = self.conditioned_metadata(first=True)
        mode_report = self.nodes.resolve_ltx25_generation_mode(
            "image_to_video",
            metadata,
            1,
        )
        prompt = (
            "One continuous take starts at eye level from the exact supplied first-frame "
            "camera viewpoint. The camera slowly dollies toward the subject and ends in a "
            "close-up without a cut."
        )

        analysis = self.contract.analyze_ltx25_prompt_complexity(
            prompt,
            "image_to_video",
            8.0,
        )
        self.assertIn("ltx_shot_scale_missing", analysis["reasons"])

        validation = self.nodes.validate_ltx25_prompt(
            prompt,
            mode_report,
            metadata,
            8.0,
        )

        self.assertNotIn("ltx_shot_scale_missing", validation["reasons"])
        self.assertIn(
            "ltx_conditioned_shot_scale_inherited_from_frame",
            validation["warnings"],
        )
        self.assertTrue(
            validation["complexity"]["observed"][
                "conditioned_shot_scale_inherited_from_frame"
            ]
        )
        self.assertTrue(validation["ready"], validation["reasons"])

        missing_metadata = self.conditioned_metadata(first=False)
        missing_mode_report = self.nodes.resolve_ltx25_generation_mode(
            "image_to_video",
            missing_metadata,
            0,
        )
        missing_frame = self.nodes.validate_ltx25_prompt(
            prompt,
            missing_mode_report,
            missing_metadata,
            8.0,
        )
        self.assertIn("ltx_i2v_first_frame_missing", missing_frame["reasons"])
        self.assertIn("ltx_shot_scale_missing", missing_frame["reasons"])
        self.assertNotIn(
            "ltx_conditioned_shot_scale_inherited_from_frame",
            missing_frame["warnings"],
        )
        self.assertFalse(missing_frame["ready"])

    def test_i2v_wide_angle_lens_wording_inherits_the_frame_scale(self) -> None:
        metadata = self.conditioned_metadata(first=True)
        mode_report = self.nodes.resolve_ltx25_generation_mode(
            "image_to_video",
            metadata,
            1,
        )
        prompt = (
            "One continuous take opens on a wide-angle, eye-level shot from the exact supplied "
            "first-frame camera viewpoint. The camera performs a slow dolly zoom toward the "
            "subject and ends in a medium close-up without a cut."
        )

        analysis = self.contract.analyze_ltx25_prompt_complexity(
            prompt,
            "image_to_video",
            8.0,
        )
        self.assertEqual(analysis["observed"]["opening_shot_scale_coverage_count"], 0)
        self.assertIn("ltx_shot_scale_missing", analysis["reasons"])

        validation = self.nodes.validate_ltx25_prompt(
            prompt,
            mode_report,
            metadata,
            8.0,
        )
        self.assertNotIn("ltx_shot_scale_missing", validation["reasons"])
        self.assertIn(
            "ltx_conditioned_shot_scale_inherited_from_frame",
            validation["warnings"],
        )
        self.assertTrue(
            validation["complexity"]["observed"][
                "conditioned_shot_scale_inherited_from_frame"
            ]
        )
        self.assertTrue(validation["ready"], validation["reasons"])

    def test_wide_comma_eye_level_shot_is_an_explicit_opening_scale(self) -> None:
        analysis = self.contract.analyze_ltx25_prompt_complexity(
            (
                "A wide, eye-level shot opens on the subject as the camera slowly dollies "
                "forward through one continuous take."
            ),
            "image_to_video",
            8.0,
        )

        self.assertEqual(analysis["observed"]["opening_shot_scale_coverage_count"], 1)
        self.assertNotIn("ltx_shot_scale_missing", analysis["reasons"])

    def test_hyphenated_scale_shot_aliases_are_explicit_opening_scales(self) -> None:
        cases = (
            ("A medium-shot at eye level", "medium shot"),
            ("A wide-shot at eye level", "wide shot"),
            ("A tighter medium-shot at eye level", "medium shot"),
        )
        for opening, expected_scale in cases:
            with self.subTest(opening=opening):
                analysis = self.contract.analyze_ltx25_prompt_complexity(
                    f"{opening} frames the subject while the camera remains static.",
                    "text_to_video",
                    5.0,
                )

                observed = analysis["observed"]
                self.assertEqual(observed["shot_scale_count"], 1)
                self.assertEqual(observed["opening_shot_scale_coverage_count"], 1)
                self.assertEqual(
                    observed["shot_scale_mentions"][0]["canonical_scale"],
                    expected_scale,
                )
                self.assertNotIn("ltx_shot_scale_missing", analysis["reasons"])

    def test_explicit_scale_can_precede_ots_or_pov_composition(self) -> None:
        cases = (
            ("A wide over-the-shoulder shot", "wide shot"),
            ("A medium POV shot", "medium shot"),
        )
        for opening, expected_scale in cases:
            with self.subTest(opening=opening):
                analysis = self.contract.analyze_ltx25_prompt_complexity(
                    f"{opening} at eye level frames the subject while the camera remains static.",
                    "text_to_video",
                    5.0,
                )

                observed = analysis["observed"]
                self.assertEqual(observed["shot_scale_count"], 1)
                self.assertEqual(observed["opening_shot_scale_coverage_count"], 1)
                self.assertEqual(
                    observed["shot_scale_mentions"][0]["canonical_scale"],
                    expected_scale,
                )
                self.assertNotIn("ltx_shot_scale_missing", analysis["reasons"])

    def test_continuous_scale_destination_aliases_are_recognized(self) -> None:
        for destination in (
            "medium close-up",
            "medium close shot",
            "medium-close shot",
        ):
            with self.subTest(destination=destination):
                analysis = self.contract.analyze_ltx25_prompt_complexity(
                    (
                        "In one continuous take at eye level, the camera dollies from a wide "
                        f"shot into a {destination} while the subject remains centered."
                    ),
                    "image_to_video",
                    8.0,
                )

                self.assertEqual(analysis["observed"]["estimated_shots"], 1)
                self.assertEqual(analysis["observed"]["shot_scale_count"], 2)
                self.assertNotIn("ltx_shot_scale_missing", analysis["reasons"])

    def test_tighter_medium_shot_is_destination_after_medium_wide_opening(self) -> None:
        metadata = self.conditioned_metadata(first=True)
        mode_report = self.nodes.resolve_ltx25_generation_mode(
            "image_to_video",
            metadata,
            1,
        )
        prompt = (
            "A medium-wide shot at eye level frames the subject while the camera slowly dollies "
            "forward in one continuous take. The shot ends in a tighter medium-shot at eye level."
        )

        validation = self.nodes.validate_ltx25_prompt(
            prompt,
            mode_report,
            metadata,
            8.0,
        )

        observed = validation["complexity"]["observed"]
        self.assertEqual(observed["shot_scale_count"], 2)
        self.assertEqual(
            [
                (item["canonical_scale"], item["role"])
                for item in observed["shot_scale_mentions"]
            ],
            [("medium-wide shot", "opening"), ("medium shot", "destination")],
        )
        self.assertEqual(observed["opening_shot_scale_coverage_count"], 1)
        self.assertEqual(observed["opening_shot_scale_shot_indices"], [1])
        self.assertNotIn("ltx_shot_scale_missing", validation["reasons"])
        self.assertTrue(validation["ready"], validation["reasons"])

    def test_scale_mentions_before_a_cut_do_not_cover_the_next_shot(self) -> None:
        mode_report = self.nodes.resolve_ltx25_generation_mode(
            "text_to_video",
            {},
            0,
        )
        prompt = (
            "A wide shot at eye level holds as the camera slowly pushes into a close-up of a "
            "runner. A hard cut reveals the street at eye level. The camera remains static, "
            "but the second shot has no opening scale."
        )

        validation = self.nodes.validate_ltx25_prompt(
            prompt,
            mode_report,
            {},
            8.0,
        )

        self.assertEqual(validation["complexity"]["observed"]["estimated_shots"], 2)
        self.assertIn("ltx_shot_scale_missing", validation["reasons"])
        self.assertFalse(validation["ready"])

    def test_composition_type_alone_does_not_satisfy_shot_scale(self) -> None:
        for composition in (
            "An over-the-shoulder shot",
            "A POV",
            "An insert shot",
        ):
            with self.subTest(composition=composition):
                analysis = self.contract.analyze_ltx25_prompt_complexity(
                    f"{composition} at eye level remains static on the subject.",
                    "text_to_video",
                    5.0,
                )

                self.assertEqual(analysis["observed"]["shot_scale_count"], 0)
                self.assertIn("ltx_shot_scale_missing", analysis["reasons"])

    def test_editorial_category_after_cut_counts_as_destination_not_scale(self) -> None:
        for destination in (
            "a POV of the street",
            "an over-the-shoulder shot of the street",
            "an insert shot of a street sign",
        ):
            with self.subTest(destination=destination):
                analysis = self.contract.analyze_ltx25_prompt_complexity(
                    (
                        "A wide shot at eye level holds while the camera remains static. "
                        f"A cut reveals {destination} at eye level while the camera remains static."
                    ),
                    "text_to_video",
                    8.0,
                )

                observed = analysis["observed"]
                self.assertEqual(observed["cut_count"], 1)
                self.assertEqual(observed["estimated_shots"], 2)
                self.assertEqual(observed["shot_scale_count"], 1)
                self.assertEqual(observed["opening_shot_scale_shot_indices"], [1])
                self.assertEqual(
                    observed["missing_opening_shot_scale_shot_indices"],
                    [2],
                )
                self.assertIn("ltx_shot_scale_missing", analysis["reasons"])

    def test_truncated_ltx_packet_remains_salvage_only_and_fails_closed(self) -> None:
        context = self.nodes.GemmaContext(
            user_prompt="Continue naturally from the supplied first frame.",
            source="image",
            media_metadata=self.conditioned_metadata(first=True),
        )
        target = self.nodes.TargetProfileConfig(
            target_profile="ltx",
            ltx_generation_mode="image_to_video",
        )
        raw = (
            '<|final_json|>{"ltx_prompt":"A static medium close-up at eye level holds on '
            'the officer.","metadata":{"controls":{"ideogram_'
        )
        packet, parse_valid, warning = self.nodes.repair_or_salvage_prompt_packet(
            raw, context, target
        )
        self.assertFalse(parse_valid)
        self.assertTrue(warning)
        self.assertTrue(packet["metadata"]["plain_text_salvage"])
        self.assertTrue(packet["metadata"]["template_used_for_missing_fields"])

        packet["metadata"].update(
            {
                "json_parse_valid": False,
                "salvage_warning": warning,
            }
        )
        ready, reasons = self.nodes._packet_generation_readiness(
            packet,
            context,
            self.nodes._target_profile_config_to_dict(target),
            packet["ltx_prompt"],
            context.media_metadata,
            8000,
        )
        self.assertFalse(ready)
        self.assertTrue(
            {"json_parse_invalid", "salvaged_output", "template_filled_missing_fields"}
            .issubset(set(reasons))
        )

    def test_opening_conflict_detects_time_of_day_and_framing_contradictions(self) -> None:
        ledger = self.frame_ledger(
            (["identity", "appearance"], "A woman with red hair fills the frame."),
            (["environment", "lighting"], "Bright daylight illuminates the outdoor street."),
            (["camera", "composition"], "The source is a tight close-up."),
        )
        metadata = self.conditioned_metadata(first=True, ledger=ledger)
        mode_report = self.nodes.resolve_ltx25_generation_mode("image_to_video", metadata, 1)
        validation = self.nodes.validate_ltx25_prompt(
            "At night, a wide shot shows the woman beneath moonlight. She remains still.",
            mode_report,
            metadata,
            8.0,
        )
        self.assertIn("ltx_first_frame_prompt_conflict", validation["reasons"])
        self.assertIn("daylight_to_night", validation["first_frame_conflicts"])
        self.assertIn("closeup_to_wide", validation["first_frame_conflicts"])

    def test_continuous_scale_destination_is_not_an_opening_frame_conflict(self) -> None:
        cases = (
            (
                "wide_to_closeup_same_sentence",
                "A wide shot frames the woman at eye level.",
                (
                    "A wide shot at eye level frames the woman while the camera slowly dollies "
                    "inward, ending in a close-up without a cut."
                ),
            ),
            (
                "wide_to_closeup_second_sentence",
                "A wide shot frames the woman at eye level.",
                (
                    "A wide shot at eye level frames the woman. The camera slowly dollies inward "
                    "and ends in a close-up without a cut."
                ),
            ),
            (
                "closeup_to_wide_same_sentence",
                "A tight close-up frames the woman at eye level.",
                (
                    "A close-up at eye level frames the woman while the camera slowly pulls back, "
                    "ending in a wide shot without a cut."
                ),
            ),
            (
                "closeup_to_wide_second_sentence",
                "A tight close-up frames the woman at eye level.",
                (
                    "A close-up at eye level frames the woman. The camera slowly pulls back and "
                    "ends in a wide shot without a cut."
                ),
            ),
        )

        for name, source_composition, prompt in cases:
            with self.subTest(case=name):
                ledger = self.frame_ledger(
                    (["identity", "appearance"], "A woman with red hair is visible."),
                    (["environment", "lighting"], "She stands in a daylight studio."),
                    (["camera", "composition"], source_composition),
                )
                metadata = self.conditioned_metadata(first=True, ledger=ledger)
                mode_report = self.nodes.resolve_ltx25_generation_mode(
                    "image_to_video",
                    metadata,
                    1,
                )
                validation = self.nodes.validate_ltx25_prompt(
                    prompt,
                    mode_report,
                    metadata,
                    8.0,
                )

                self.assertNotIn(
                    "ltx_first_frame_prompt_conflict",
                    validation["reasons"],
                )
                self.assertEqual(validation["first_frame_conflicts"], [])
                self.assertTrue(validation["ready"], validation["reasons"])

    def test_genuine_opening_scale_conflict_still_blocks_i2v(self) -> None:
        cases = (
            (
                "wide_source_closeup_opening",
                "A wide shot frames the woman at eye level.",
                "A static close-up at eye level holds on the woman.",
                "wide_to_closeup",
            ),
            (
                "closeup_source_wide_opening",
                "A tight close-up frames the woman at eye level.",
                "A static wide shot at eye level holds on the woman.",
                "closeup_to_wide",
            ),
        )

        for name, source_composition, prompt, expected_conflict in cases:
            with self.subTest(case=name):
                ledger = self.frame_ledger(
                    (["identity", "appearance"], "A woman with red hair is visible."),
                    (["environment", "lighting"], "She stands in a daylight studio."),
                    (["camera", "composition"], source_composition),
                )
                metadata = self.conditioned_metadata(first=True, ledger=ledger)
                mode_report = self.nodes.resolve_ltx25_generation_mode(
                    "image_to_video",
                    metadata,
                    1,
                )
                validation = self.nodes.validate_ltx25_prompt(
                    prompt,
                    mode_report,
                    metadata,
                    8.0,
                )

                self.assertIn(
                    "ltx_first_frame_prompt_conflict",
                    validation["reasons"],
                )
                self.assertIn(expected_conflict, validation["first_frame_conflicts"])
                self.assertFalse(validation["ready"])

    def test_source_scale_conflicts_use_only_camera_composition_claims(self) -> None:
        ledger = self.frame_ledger(
            (
                ["appearance", "object"],
                "A close-up portrait is printed on her shirt.",
            ),
            (["environment", "lighting"], "She stands in a daylight studio."),
            (["camera", "composition"], "The source is a wide shot at eye level."),
        )
        metadata = self.conditioned_metadata(first=True, ledger=ledger)
        mode_report = self.nodes.resolve_ltx25_generation_mode(
            "image_to_video",
            metadata,
            1,
        )

        matching = self.nodes.validate_ltx25_prompt(
            "A static wide shot at eye level frames the woman wearing the printed shirt.",
            mode_report,
            metadata,
            8.0,
        )
        self.assertNotIn("ltx_first_frame_prompt_conflict", matching["reasons"])
        self.assertEqual(matching["first_frame_conflicts"], [])
        self.assertTrue(matching["ready"], matching["reasons"])

        conflicting = self.nodes.validate_ltx25_prompt(
            "A static close-up at eye level frames the woman wearing the printed shirt.",
            mode_report,
            metadata,
            8.0,
        )
        self.assertIn("ltx_first_frame_prompt_conflict", conflicting["reasons"])
        self.assertIn("wide_to_closeup", conflicting["first_frame_conflicts"])
        self.assertFalse(conflicting["ready"])

    def test_object_modifier_closeup_does_not_override_prompt_opening_scale(self) -> None:
        ledger = self.frame_ledger(
            (["identity", "appearance"], "A woman wearing a printed shirt is visible."),
            (["environment", "lighting"], "She stands in a daylight studio."),
            (["camera", "composition"], "The source is a wide shot at eye level."),
        )
        metadata = self.conditioned_metadata(first=True, ledger=ledger)
        mode_report = self.nodes.resolve_ltx25_generation_mode(
            "image_to_video",
            metadata,
            1,
        )
        prompt = (
            "A close-up portrait is printed on her shirt as a wide shot frames her at eye "
            "level while the camera remains static."
        )

        analysis = self.contract.analyze_ltx25_prompt_complexity(
            prompt,
            "image_to_video",
            8.0,
        )
        observed = analysis["observed"]
        self.assertEqual(observed["shot_scale_count"], 1)
        self.assertEqual(observed["opening_shot_scale_coverage_count"], 1)
        self.assertEqual(
            [
                (item["canonical_scale"], item["role"])
                for item in observed["shot_scale_mentions"]
            ],
            [("wide shot", "opening")],
        )
        self.assertEqual(
            [
                (item["canonical_scale"], item["role"])
                for item in observed["ignored_shot_scale_mentions"]
            ],
            [("close-up", "object_modifier")],
        )
        self.assertEqual(
            [
                item["canonical_scale"]
                for item in observed["shot_scale_mentions"]
                if item["covers_opening"]
            ],
            ["wide shot"],
        )

        validation = self.nodes.validate_ltx25_prompt(
            prompt,
            mode_report,
            metadata,
            8.0,
        )
        self.assertNotIn("ltx_first_frame_prompt_conflict", validation["reasons"])
        self.assertEqual(validation["first_frame_conflicts"], [])
        self.assertTrue(validation["ready"], validation["reasons"])

    def test_first_frame_grounding_coverage_accepts_multiple_fact_families(self) -> None:
        complete_ledger = self.frame_ledger(
            (["identity", "appearance"], "A red-haired woman wears a dark jacket."),
            (["environment", "lighting"], "She stands in a sunlit exterior courtyard."),
            (["camera", "composition"], "A static medium shot frames her at eye level."),
        )
        complete_metadata = self.conditioned_metadata(first=True, ledger=complete_ledger)
        complete_mode = self.nodes.resolve_ltx25_generation_mode(
            "image_to_video", complete_metadata, 1
        )
        complete = self.nodes.validate_ltx25_prompt(
            (
                "A static medium shot at eye level holds on the red-haired woman in her dark jacket "
                "inside the sunlit exterior courtyard. She slowly turns toward the doorway."
            ),
            complete_mode,
            complete_metadata,
            8.0,
        )
        coverage = complete["first_frame_grounding"]
        self.assertEqual(coverage["status"], "verified")
        self.assertEqual(set(coverage["families"]), {"subject", "setting_light", "composition"})
        self.assertNotIn("ltx_first_frame_grounding_incomplete", complete["reasons"])

        incomplete_ledger = self.frame_ledger(
            (["identity"], "A woman is visible."),
        )
        incomplete_metadata = self.conditioned_metadata(first=True, ledger=incomplete_ledger)
        incomplete_mode = self.nodes.resolve_ltx25_generation_mode(
            "image_to_video", incomplete_metadata, 1
        )
        incomplete = self.nodes.validate_ltx25_prompt(
            "A static medium shot holds on the woman as she turns her head.",
            incomplete_mode,
            incomplete_metadata,
            8.0,
        )
        self.assertEqual(incomplete["first_frame_grounding"]["status"], "incomplete")
        self.assertIn("ltx_first_frame_grounding_incomplete", incomplete["reasons"])

    def test_generation_gate_translates_contract_reasons_into_actionable_language(self) -> None:
        metadata = {
            "blocked_reasons": ["ltx_shot_count_exceeds_duration_budget"],
            "ltx25_contract": {
                "complexity": {
                    "observed": {"estimated_shots": 3},
                    "budget": {"max_shots": 1},
                }
            },
        }
        with self.assertRaises(ValueError) as caught:
            self.nodes.DiffusionGemmaGenerationGate().gate(
                "A prompt that should not pass.",
                False,
                json.dumps(metadata),
            )
        message = str(caught.exception)
        self.assertIn("more shots than fit the selected duration", message)
        self.assertIn("Detected 3; limit 1", message)
        self.assertIn("Remove cuts or increase duration", message)
        self.assertNotIn("ltx_shot_count_exceeds_duration_budget", message)

    def test_splitter_explicit_resolution_ratio_controls_dimensions_preset_and_metadata(self) -> None:
        context = self.nodes.GemmaContext(
            user_prompt="A woman raises one hand in a quiet studio.",
            source="none",
            media_metadata={"source": "none"},
        )
        target = self.nodes.TargetProfileConfig(target_profile="ltx")
        result = self.nodes.DiffusionGemmaJSONSplitter().split(
            self.prompt_packet(
                ltx_prompt=(
                    "A static medium shot at eye level holds on a woman in a quiet studio. "
                    "She slowly raises one hand while the locked camera remains still."
                )
            ),
            context,
            target,
            "",
            0.5,
            32,
            "16:9 (Widescreen)",
        )
        metadata = json.loads(result[4])

        self.assertEqual(result[3], "16:9")
        self.assertEqual(result[9], "16:9 (Widescreen)")
        self.assertEqual(result[10:12], (960, 544))
        self.assertEqual(metadata["aspect_ratio"], "16:9")
        self.assertEqual(metadata["resolution_selector_preset"], "16:9 (Widescreen)")
        self.assertEqual(metadata["resolution_width"], 960)
        self.assertEqual(metadata["resolution_height"], 544)
        self.assertEqual(metadata["resolution_aspect_ratio_requested"], "16:9 (Widescreen)")
        self.assertEqual(metadata["resolution_aspect_ratio_source"], "splitter_widget")

    def test_splitter_auto_i2v_uses_first_frame_ratio_with_legacy_positional_call(self) -> None:
        media_metadata = self.conditioned_metadata(first=True)
        media_metadata.update(
            {
                "ltx_first_frame_width": 1080,
                "ltx_first_frame_height": 1920,
            }
        )
        context = self.nodes.GemmaContext(
            user_prompt="Continue naturally from the supplied portrait first frame.",
            source="image",
            media_metadata=media_metadata,
        )
        target = self.nodes.TargetProfileConfig(
            target_profile="ltx",
            ltx_generation_mode="image_to_video",
        )

        # This is the complete positional splitter call from before
        # resolution_aspect_ratio was appended. The omitted value must still
        # bind to the new Auto default instead of shifting an older widget.
        result = self.nodes.DiffusionGemmaJSONSplitter().split(
            self.prompt_packet(
                ltx_prompt=(
                    "The supplied portrait first-frame viewpoint holds in a static medium shot at eye level. "
                    "The subject slowly turns her head in one continuous take."
                )
            ),
            context,
            target,
            "",
            0.5,
            32,
        )
        metadata = json.loads(result[4])

        self.assertEqual(result[3], "9:16")
        self.assertEqual(result[9], "9:16 (Portrait Widescreen)")
        self.assertEqual(result[10:12], (544, 960))
        self.assertEqual(metadata["resolution_aspect_ratio_requested"], self.nodes.RESOLUTION_SELECTOR_AUTO)
        self.assertEqual(metadata["resolution_aspect_ratio_source"], "ltx_first_frame")

    def test_splitter_auto_ideogram_preserves_target_profile_aspect_ratio(self) -> None:
        context = self.nodes.GemmaContext(
            user_prompt="Create an editorial portrait poster.",
            source="none",
            media_metadata={"source": "none"},
        )
        target = self.nodes.TargetProfileConfig(
            target_profile="ideogram4",
            ideogram_aspect_ratio="3:4",
            ideogram_json_output=False,
        )
        result = self.nodes.DiffusionGemmaJSONSplitter().split(
            self.prompt_packet(
                ideogram_prompt=(
                    "An editorial portrait poster with a centered subject, restrained typography, "
                    "soft studio lighting, and a muted blue palette."
                )
            ),
            context,
            target,
            resolution_megapixels=0.5,
            resolution_multiple=32,
        )
        metadata = json.loads(result[4])

        self.assertEqual(result[3], "3:4")
        self.assertEqual(result[9], "3:4 (Portrait Standard)")
        self.assertEqual(metadata["resolution_aspect_ratio_source"], "ideogram_target_profile")
        self.assertEqual(metadata["resolution_aspect_ratio_requested"], self.nodes.RESOLUTION_SELECTOR_AUTO)

    def test_splitter_resolution_selector_is_appended_after_legacy_optional_inputs(self) -> None:
        optional = self.nodes.DiffusionGemmaJSONSplitter.INPUT_TYPES()["optional"]
        self.assertEqual(
            list(optional),
            [
                "gemma_context",
                "target_profile_config",
                "fallback_prompt",
                "resolution_megapixels",
                "resolution_multiple",
                "resolution_aspect_ratio",
                "resolution_aspect_ratio_override",
            ],
        )
        choices, config = optional["resolution_aspect_ratio"]
        self.assertEqual(choices, self.nodes.RESOLUTION_SELECTOR_ASPECT_CHOICES)
        self.assertEqual(config["default"], self.nodes.RESOLUTION_SELECTOR_AUTO)
        override_type, override_config = optional["resolution_aspect_ratio_override"]
        self.assertEqual(override_type, "STRING")
        self.assertTrue(override_config["forceInput"])

    def test_splitter_project_aspect_override_is_authoritative(self) -> None:
        context = self.nodes.GemmaContext(
            user_prompt="Create a portrait music video.",
            source="none",
            media_metadata={"source": "none"},
        )
        target = self.nodes.TargetProfileConfig(target_profile="minimax_h3")
        result = self.nodes.DiffusionGemmaJSONSplitter().split(
            self.prompt_packet(),
            context,
            target,
            resolution_megapixels=0.4,
            resolution_multiple=32,
            resolution_aspect_ratio="16:9 (Widescreen)",
            resolution_aspect_ratio_override="9:16",
        )
        metadata = json.loads(result[4])

        self.assertEqual(result[3], "9:16")
        self.assertEqual(metadata["resolution_aspect_ratio_requested"], "9:16")
        self.assertEqual(
            metadata["resolution_aspect_ratio_source"], "project_master_contract"
        )
        self.assertLess(result[10], result[11])

    def test_splitter_rejects_invalid_project_aspect_override(self) -> None:
        context = self.nodes.GemmaContext(
            user_prompt="Create a music video.",
            source="none",
            media_metadata={"source": "none"},
        )
        target = self.nodes.TargetProfileConfig(target_profile="minimax_h3")
        with self.assertRaisesRegex(ValueError, "supported canonical W:H ratios"):
            self.nodes.DiffusionGemmaJSONSplitter().split(
                self.prompt_packet(),
                context,
                target,
                resolution_aspect_ratio_override="5:7",
            )

    def test_ltx_controls_are_appended_to_config_and_node_positional_signatures(self) -> None:
        legacy_values = (
            "minimax_h3",
            "explicit_sound_design",
            "measured footsteps",
            12.0,
            "cinematic",
            "16:9",
            "photo",
            "TITLE",
            False,
            "custom",
            "no blur",
            "ref2va",
            "3",
            "required",
            1,
            "One short line.",
        )
        direct = self.nodes.TargetProfileConfig(*legacy_values)
        self.assertEqual(direct.minimax_h3_dialogue_guidance, "One short line.")
        self.assertEqual(direct.ltx_generation_mode, "auto")
        self.assertEqual(direct.ltx_long_horizon_mode, "off")
        self.assertEqual(direct.ltx_camera_capability, "stable")

        made = self.nodes._make_target_profile_config(*legacy_values)
        self.assertEqual(made.minimax_h3_dialogue_guidance, "One short line.")
        self.assertEqual(made.ltx_generation_mode, "auto")
        self.assertEqual(made.ltx_long_horizon_mode, "off")
        self.assertEqual(made.ltx_camera_capability, "stable")

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
        choices, config = optional["ltx_generation_mode"]
        self.assertEqual(
            choices,
            ["Auto (recommended)", "Text to video", "Image to video", "First + last frame"],
        )
        self.assertEqual(config["default"], "Auto (recommended)")
        long_choices, long_config = optional["ltx_long_horizon_mode"]
        self.assertEqual(long_choices, ["Off", "Auto (>20 seconds)", "On"])
        self.assertEqual(long_config["default"], "Off")
        camera_choices, camera_config = optional["ltx_camera_capability"]
        self.assertEqual(
            camera_choices,
            ["Stable / base model", "Advanced / controlled camera"],
        )
        self.assertEqual(camera_config["default"], "Stable / base model")

        required = (
            "ltx",
            "auto_scene_audio",
            "",
            8.0,
            "",
            "16:9",
            "",
            "",
            True,
            "auto",
            "",
        )
        # Six legacy optional values end at dialogue guidance. The newly added
        # Appended LTX controls must retain their defaults instead of shifting
        # old widgets.
        legacy_node_config, legacy_json = self.nodes.DiffusionGemmaTargetProfile().build(
            *required,
            "t2va",
            "auto",
            12,
            "off",
            2,
            "",
        )
        self.assertEqual(legacy_node_config.ltx_generation_mode, "auto")
        self.assertEqual(legacy_node_config.ltx_long_horizon_mode, "off")
        self.assertEqual(legacy_node_config.ltx_camera_capability, "stable")
        self.assertEqual(json.loads(legacy_json)["ltx_generation_mode"], "auto")
        self.assertEqual(json.loads(legacy_json)["ltx_long_horizon_mode"], "off")
        self.assertEqual(json.loads(legacy_json)["ltx_camera_capability"], "stable")

        explicit_node_config, _json = self.nodes.DiffusionGemmaTargetProfile().build(
            *required,
            "t2va",
            "auto",
            12,
            "off",
            2,
            "",
            "First + last frame",
        )
        self.assertEqual(explicit_node_config.ltx_generation_mode, "first_last_frame")
        self.assertEqual(explicit_node_config.ltx_long_horizon_mode, "off")

        long_node_config, _json = self.nodes.DiffusionGemmaTargetProfile().build(
            *required,
            "t2va",
            "auto",
            12,
            "off",
            2,
            "",
            "Image to video",
            "Auto (>20 seconds)",
        )
        self.assertEqual(long_node_config.ltx_generation_mode, "image_to_video")
        self.assertEqual(long_node_config.ltx_long_horizon_mode, "auto")

        advanced_node_config, advanced_json = self.nodes.DiffusionGemmaTargetProfile().build(
            *required,
            "t2va",
            "auto",
            12,
            "off",
            2,
            "",
            "Image to video",
            "Off",
            "Advanced / controlled camera",
        )
        self.assertEqual(advanced_node_config.ltx_camera_capability, "advanced")
        self.assertEqual(json.loads(advanced_json)["ltx_camera_capability"], "advanced")


if __name__ == "__main__":
    unittest.main()
