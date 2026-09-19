from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import torch

from comfy_test_stubs import install_comfy_stubs

install_comfy_stubs()

from ltx_reframe import (
    LTXReframeCustomAudioLoader,
    LTXReframeRectilinearCorrection,
    LTXReframeTargetFit,
    NODE_CLASS_MAPPINGS,
    build_reframe_masks,
    compute_reframe_placement,
    ensure_audio_track,
    fit_audio_track,
    prepare_reframe_canvas,
    rectilinear_correct_images,
    resolve_frame_range,
    snap_dimension,
)


class LTXReframeTests(unittest.TestCase):
    @staticmethod
    def _write_test_wav(path: Path, *, sample_rate: int = 8000, samples: int = 80):
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(b"\x00\x00" * samples)

    def test_custom_audio_loader_is_registered_and_accepts_an_empty_lazy_branch(self):
        self.assertIs(
            NODE_CLASS_MAPPINGS["LTXReframeCustomAudioLoader"],
            LTXReframeCustomAudioLoader,
        )
        self.assertTrue(LTXReframeCustomAudioLoader.VALIDATE_INPUTS(""))
        with self.assertRaisesRegex(ValueError, "no custom audio file is selected"):
            LTXReframeCustomAudioLoader().load_audio("")

    def test_custom_audio_loader_lists_audio_uploads_with_a_blank_default(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("ltx_reframe.folder_paths.get_input_directory", return_value=directory),
                patch(
                    "ltx_reframe.folder_paths.filter_files_content_types",
                    return_value=["song.wav"],
                ),
            ):
                spec = LTXReframeCustomAudioLoader.INPUT_TYPES()

        options, config = spec["required"]["audio_file"]
        self.assertEqual(options, ["", "song.wav"])
        self.assertTrue(config["audio_upload"])

    def test_custom_audio_loader_decodes_and_fingerprints_an_input_file(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "clip.wav"
            self._write_test_wav(audio_path)
            with (
                patch("ltx_reframe.folder_paths.get_input_directory", return_value=directory),
                patch(
                    "ltx_reframe.folder_paths.annotated_filepath",
                    side_effect=lambda name: (name, None),
                ),
            ):
                audio, = LTXReframeCustomAudioLoader().load_audio("clip.wav")
                first_fingerprint = LTXReframeCustomAudioLoader.IS_CHANGED("clip.wav")
                second_fingerprint = LTXReframeCustomAudioLoader.IS_CHANGED("clip.wav")
                with audio_path.open("ab") as handle:
                    handle.write(b"changed")
                changed_fingerprint = LTXReframeCustomAudioLoader.IS_CHANGED("clip.wav")

        self.assertEqual(audio["sample_rate"], 8000)
        self.assertEqual(tuple(audio["waveform"].shape), (1, 1, 80))
        self.assertEqual(first_fingerprint, second_fingerprint)
        self.assertNotEqual(first_fingerprint, changed_fingerprint)
        self.assertEqual(len(first_fingerprint), 64)

    def test_custom_audio_loader_reports_a_missing_input_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("ltx_reframe.folder_paths.get_input_directory", return_value=directory),
                patch(
                    "ltx_reframe.folder_paths.annotated_filepath",
                    side_effect=lambda name: (name, None),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "file was not found"):
                    LTXReframeCustomAudioLoader().load_audio("missing.wav")

    def test_custom_audio_loader_rejects_paths_outside_comfyui_input(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("ltx_reframe.folder_paths.get_input_directory", return_value=directory),
                patch(
                    "ltx_reframe.folder_paths.annotated_filepath",
                    return_value=("../outside.wav", None),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "input directory"):
                    LTXReframeCustomAudioLoader().load_audio("../outside.wav")

    def test_custom_audio_example_workflow_has_consistent_links_and_safe_defaults(self):
        workflow_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "gemmaREFRAME_Experimental_audio_on-off.json"
        )
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}

        self.assertEqual(len(nodes), len(workflow["nodes"]))
        self.assertEqual(len(links), len(workflow["links"]))
        for link_id, (_, source_id, source_slot, target_id, target_slot, _type) in links.items():
            self.assertIn(source_id, nodes)
            self.assertIn(target_id, nodes)
            self.assertIn(link_id, nodes[source_id]["outputs"][source_slot].get("links") or [])
            self.assertEqual(nodes[target_id]["inputs"][target_slot].get("link"), link_id)

        custom_loader = nodes[9086]
        selector = nodes[9112]
        reframe = nodes[5553]
        stage_one = nodes[9055]
        create_video = nodes[5539]
        preview = nodes[9088]

        self.assertEqual(custom_loader["type"], "LTXReframeCustomAudioLoader")
        self.assertEqual(custom_loader["widgets_values"], [""])
        self.assertEqual(selector["type"], "LazySwitchKJ")
        self.assertFalse(selector["widgets_values"][0])
        self.assertEqual(set(selector["outputs"][0]["links"]), {14666, 14818, 14918})
        self.assertEqual(reframe["outputs"][4]["links"], [14990, 14991])
        self.assertEqual(stage_one["inputs"][10]["link"], 14818)
        self.assertEqual(preview["inputs"][0]["link"], 14918)
        self.assertEqual(create_video["inputs"][1]["link"], 14999)
        self.assertEqual(create_video["inputs"][2]["link"], 14666)

    def test_existing_audio_is_preserved(self):
        audio = {
            "waveform": torch.ones((1, 1, 100), dtype=torch.float32),
            "sample_rate": 16000,
        }

        result, synthesized = ensure_audio_track(audio, frame_count=24, fps=24.0)

        self.assertIs(result, audio)
        self.assertFalse(synthesized)

    def test_audio_less_video_gets_duration_matched_stereo_silence(self):
        audio, synthesized = ensure_audio_track(None, frame_count=121, fps=24.0)

        self.assertTrue(synthesized)
        self.assertEqual(audio["sample_rate"], 44100)
        self.assertEqual(tuple(audio["waveform"].shape), (1, 2, 222338))
        self.assertEqual(torch.count_nonzero(audio["waveform"]).item(), 0)

    def test_silent_audio_fallback_rejects_invalid_fps(self):
        with self.assertRaisesRegex(ValueError, "valid video frame rate"):
            ensure_audio_track(None, frame_count=121, fps=0.0)

    def test_audio_is_cropped_and_padded_to_selected_duration(self):
        cropped, synthesized = fit_audio_track(
            {"waveform": torch.ones((1, 2, 2000)), "sample_rate": 1000},
            frame_count=25,
            fps=25.0,
        )
        padded, _ = fit_audio_track(
            {"waveform": torch.ones((1, 2, 500)), "sample_rate": 1000},
            frame_count=25,
            fps=25.0,
        )

        self.assertFalse(synthesized)
        self.assertEqual(cropped["waveform"].shape[-1], 1000)
        self.assertEqual(padded["waveform"].shape[-1], 1000)
        self.assertEqual(torch.count_nonzero(padded["waveform"][..., 500:]).item(), 0)

    def test_full_video_range_is_not_aligned_or_trimmed(self):
        selected = resolve_frame_range(70, 25.0, False, 8, 16)

        self.assertFalse(selected.range_enabled)
        self.assertEqual(selected.effective_first_frame, 0)
        self.assertEqual(selected.effective_last_frame, 69)
        self.assertEqual(selected.selected_frame_count, 70)

    def test_valid_sixty_five_frame_range_is_preserved(self):
        selected = resolve_frame_range(241, 25.0, True, 0, 64)

        self.assertEqual(selected.effective_first_frame, 0)
        self.assertEqual(selected.effective_last_frame, 64)
        self.assertEqual(selected.selected_frame_count, 65)

    def test_range_alignment_preserves_first_frame_and_trims_end(self):
        zero_based = resolve_frame_range(241, 25.0, True, 0, 70)
        offset = resolve_frame_range(241, 25.0, True, 100, 199)

        self.assertEqual((zero_based.effective_first_frame, zero_based.effective_last_frame), (0, 64))
        self.assertEqual(zero_based.selected_frame_count, 65)
        self.assertEqual((offset.effective_first_frame, offset.effective_last_frame), (100, 196))
        self.assertEqual(offset.selected_frame_count, 97)

    def test_last_frame_minus_one_resolves_to_source_end_then_aligns(self):
        selected = resolve_frame_range(100, 25.0, True, 10, -1)

        self.assertEqual(selected.requested_last_frame, 99)
        self.assertEqual(selected.effective_last_frame, 98)
        self.assertEqual(selected.selected_frame_count, 89)

    def test_invalid_frame_ranges_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            resolve_frame_range(100, 25.0, True, 30, 20)
        with self.assertRaisesRegex(ValueError, "outside"):
            resolve_frame_range(100, 25.0, True, 0, 100)
        with self.assertRaisesRegex(ValueError, "at least nine"):
            resolve_frame_range(100, 25.0, True, 0, 7)

    def test_single_frame_range_is_allowed(self):
        selected = resolve_frame_range(100, 25.0, True, 42, 42)

        self.assertEqual(selected.selected_frame_count, 1)
        self.assertEqual(selected.effective_last_frame, 42)

    def test_dimensions_snap_to_ltx_multiple(self):
        self.assertEqual(snap_dimension(1919), 1920)
        self.assertEqual(snap_dimension(1050), 1024)
        self.assertEqual(snap_dimension(1), 256)

    def test_portrait_canvas_centers_aspect_locked_landscape_source(self):
        placement = compute_reframe_placement(1920, 1080, 1088, 1920, 0.5, 0.5, 0.7)
        self.assertEqual(placement.target_width, 1088)
        self.assertEqual(placement.target_height, 1920)
        self.assertLess(abs(placement.box_width / placement.box_height - 16 / 9), 0.01)
        self.assertLessEqual(abs(placement.box_x * 2 + placement.box_width - placement.target_width), 1)
        self.assertLessEqual(abs(placement.box_y * 2 + placement.box_height - placement.target_height), 1)

    def test_canvas_and_masks_preserve_source_region(self):
        frames = torch.ones((2, 8, 16, 3), dtype=torch.float32)
        placement = compute_reframe_placement(16, 8, 256, 256, 0.5, 0.5, 0.5)
        canvas = prepare_reframe_canvas(frames, placement)
        outpaint, source = build_reframe_masks(placement, feather=0)

        self.assertEqual(tuple(canvas.shape), (2, 256, 256, 3))
        self.assertEqual(tuple(outpaint.shape), (1, 256, 256))
        self.assertEqual(tuple(source.shape), (1, 256, 256))
        self.assertTrue(torch.allclose(outpaint + source, torch.ones_like(source)))
        self.assertEqual(canvas[:, placement.box_y, placement.box_x].mean().item(), 1.0)
        self.assertEqual(canvas[:, 0, 0].sum().item(), 0.0)

    def test_feather_ramps_only_outside_source_box(self):
        placement = compute_reframe_placement(16, 9, 256, 256, 0.5, 0.5, 0.5)
        outpaint, source = build_reframe_masks(placement, feather=16)
        inside_y = placement.box_y + placement.box_height // 2
        inside_x = placement.box_x + placement.box_width // 2
        self.assertEqual(outpaint[0, inside_y, inside_x].item(), 0.0)
        self.assertEqual(source[0, inside_y, inside_x].item(), 1.0)
        self.assertEqual(outpaint[0, 0, 0].item(), 1.0)

    def test_zoom_in_can_extend_source_beyond_target(self):
        placement = compute_reframe_placement(1920, 1080, 1088, 1920, 0.5, 0.5, 2.0)

        self.assertEqual(placement.box_width, 2176)
        self.assertEqual(placement.box_height, 1224)
        self.assertEqual(placement.box_x, -544)
        self.assertEqual(placement.box_y, 348)
        self.assertEqual(placement.source_scale, 2.0)

    def test_zoomed_source_is_cropped_safely_to_target_canvas(self):
        frames = torch.ones((2, 8, 16, 3), dtype=torch.float32)
        placement = compute_reframe_placement(16, 8, 256, 256, 0.5, 0.5, 4.0)
        canvas = prepare_reframe_canvas(frames, placement)
        outpaint, source = build_reframe_masks(placement, feather=32)

        self.assertEqual(tuple(canvas.shape), (2, 256, 256, 3))
        self.assertTrue(torch.allclose(canvas, torch.ones_like(canvas)))
        self.assertTrue(torch.allclose(source, torch.ones_like(source)))
        self.assertTrue(torch.allclose(outpaint, torch.zeros_like(outpaint)))

    def test_target_fit_returns_exact_dimensions(self):
        images = torch.rand((2, 96, 160, 3), dtype=torch.float32)

        fitted, = LTXReframeTargetFit().fit(images, target_width=128, target_height=128)

        self.assertEqual(tuple(fitted.shape), (2, 128, 128, 3))
        self.assertGreaterEqual(fitted.min().item(), 0.0)
        self.assertLessEqual(fitted.max().item(), 1.0)

    def test_rectilinear_correction_is_temporally_deterministic(self):
        horizontal = torch.linspace(0.0, 1.0, 65).view(1, 1, 65, 1)
        frame = horizontal.expand(1, 33, 65, 3)
        images = frame.expand(2, -1, -1, -1).clone()

        corrected, correction_mask = rectilinear_correct_images(
            images,
            correction_strength=0.1,
            center_x=0.5,
            center_y=0.5,
            protect_source=0.0,
            protection_feather=0,
            interpolation="bilinear",
            chunk_size=1,
        )

        self.assertEqual(tuple(corrected.shape), tuple(images.shape))
        self.assertEqual(tuple(correction_mask.shape), (1, 33, 65))
        self.assertTrue(torch.allclose(corrected[0], corrected[1]))
        self.assertGreater(corrected[0, 16, 0, 0].item(), images[0, 16, 0, 0].item())
        self.assertLess(corrected[0, 16, -1, 0].item(), images[0, 16, -1, 0].item())
        self.assertAlmostEqual(
            corrected[0, 16, 32, 0].item(),
            images[0, 16, 32, 0].item(),
            places=5,
        )

    def test_rectilinear_source_mask_can_fully_protect_original_pixels(self):
        images = torch.rand((2, 24, 40, 3), dtype=torch.float32)
        source_mask = torch.ones((1, 24, 40), dtype=torch.float32)

        corrected, correction_mask = rectilinear_correct_images(
            images,
            correction_strength=0.15,
            center_x=0.5,
            center_y=0.5,
            protect_source=1.0,
            protection_feather=0,
            interpolation="bicubic",
            chunk_size=2,
            source_mask=source_mask,
        )

        self.assertTrue(torch.allclose(corrected, images))
        self.assertEqual(torch.count_nonzero(correction_mask).item(), 0)

    def test_rectilinear_node_bypass_is_pixel_exact(self):
        images = torch.rand((3, 16, 20, 3), dtype=torch.float32)

        bypassed, correction_mask = LTXReframeRectilinearCorrection().correct(
            images,
            enabled=False,
            correction_strength=0.05,
            center_x=0.5,
            center_y=0.5,
            protect_source=1.0,
            protection_feather=96,
            interpolation="bicubic",
            chunk_size=8,
        )

        self.assertIs(bypassed, images)
        self.assertEqual(tuple(correction_mask.shape), (1, 16, 20))
        self.assertEqual(torch.count_nonzero(correction_mask).item(), 0)


if __name__ == "__main__":
    unittest.main()
