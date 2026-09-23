import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from comfy_api.latest import _io


spec = importlib.util.spec_from_file_location("parallax_test", Path(__file__).parents[1] / "nodes/vfx/FL_LayeredParallax.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def plate(color, depth, count=3):
    image = torch.zeros(count, 32, 64, 4)
    image[:, 8:24, 24:40, :3] = torch.tensor(color)
    image[:, 8:24, 24:40, 3] = 1
    return m.FL_ParallaxLayer.execute(image, depth, 1, 0, 0, 1).result[0]


def render(layers, **kwargs):
    values = dict(background=torch.zeros(3, 32, 64, 3), width=64, height=32, motion="glide", travel_x=.3,
                  travel_y=0, push_in=0, background_depth=12, overscan=1, device="cpu", layers=layers)
    values.update(kwargs)
    with patch.object(m, "preview_layers", return_value=[]):
        return m.FL_LayeredParallax.execute(**values).result


class ParallaxTests(unittest.TestCase):
    def test_rgba_batch_stack_preserves_order_alpha_and_variable_count(self):
        for count in (1, 2, 4, 8):
            images = torch.rand(count, 16, 16, 4)
            stack = m.FL_ParallaxStackFromBatch.execute(images, 2, 10).result[0]
            self.assertEqual(len(stack["layers"]), count - 1)
            torch.testing.assert_close(stack["background"], images[:1])
            for i, layer in enumerate(stack["layers"]):
                torch.testing.assert_close(layer["images"], images[i + 1:i + 2])
                self.assertNotEqual(layer["images"].untyped_storage().data_ptr(), images.untyped_storage().data_ptr())
            depths = [p["depth"] for p in stack["layers"]]
            self.assertEqual(depths, sorted(depths, reverse=True))
            if depths:
                self.assertEqual(depths[-1], 2)

    def test_rgba_batch_stack_rejects_rgb_and_inverted_depths(self):
        for images, near, far in ((torch.ones(2, 16, 16, 3), 2, 10), (torch.ones(2, 16, 16, 4), 10, 2)):
            with self.assertRaises(ValueError):
                m.FL_ParallaxStackFromBatch.execute(images, near, far)

    def test_rgba_batch_stack_connects_to_depth_and_compositor(self):
        images = torch.zeros(3, 32, 64, 4)
        images[0, ..., 3] = 1
        images[1, 4:20, 4:20, 0] = 1
        images[1, 4:20, 4:20, 3] = 1
        images[2, 12:28, 12:28, 1] = 1
        images[2, 12:28, 12:28, 3] = 1
        stack = m.FL_ParallaxStackFromBatch.execute(images, 2, 10).result[0]
        analysis = m.FL_ParallaxDepthSources.execute(stack, 126).result[0]
        self.assertEqual(len(analysis), 3)
        _, neutral, _ = render({}, layer_stack=stack, frames=3, motion="locked")
        expected = images[0:1, ..., :3]
        for layer in images[1:]:
            expected = layer[None, ..., :3] * layer[None, ..., 3:4] + expected * (1 - layer[None, ..., 3:4])
        torch.testing.assert_close(neutral[:1], expected)

    def test_animated_sources_keep_per_frame_neutral_composites(self):
        background = torch.rand(9, 32, 64, 3)
        p = plate([.2, .5, .8], 2, count=9)
        p["images"][:, 8:24, 24:40, 0] = torch.arange(9)[:, None, None] / 9
        _, locked, _ = render({"p": p}, background=background)
        self.assertNotEqual(locked.stride(0), 0)
        for i in (0, 4, 8):
            still = dict(p, images=p["images"][i:i+1])
            expected = render({"p": still}, background=background[i:i+1], frames=1, motion="locked")[1]
            torch.testing.assert_close(locked[i:i+1], expected)

    def test_static_locked_output_shares_one_frame(self):
        p = plate([1, 0, 0], 2, count=1)
        moving, locked, _ = render({"p": p}, background=torch.zeros(1, 32, 64, 3), frames=13)
        self.assertEqual(locked.stride(0), 0)
        self.assertEqual(locked.untyped_storage().nbytes(), 32 * 64 * 3 * 4)
        for frame in locked:
            torch.testing.assert_close(frame, locked[0])
        self.assertFalse(torch.equal(moving[0], moving[-1]))

    def test_contain_preserves_mismatched_cutout_aspect(self):
        image = torch.ones(1, 64, 64, 4)
        p = m.FL_ParallaxLayer.execute(image, 2, 1, 0, 0, 1).result[0]
        cover = render({"p": p}, motion="locked", layer_fit="cover")[0]
        contain = render({"p": p}, motion="locked", layer_fit="contain")[0]
        self.assertEqual(float(cover[0, 16, 2, 0]), 1)
        self.assertEqual(float(contain[0, 16, 2, 0]), 0)
        self.assertEqual(float(contain[0, 16, 32, 0]), 1)

    def test_thumbnail_metadata_and_explicit_alpha(self):
        p = plate([1, 0, 0], 2, count=1)
        p["name"] = "Subject"
        with patch.object(m.nodes.SaveImage, "save_images", return_value={"ui": {"images": [{"filename": "thumb.png", "type": "output"}]}}) as save:
            previews = m.preview_layers(torch.zeros(1, 512, 1024, 3), [p])
        self.assertEqual(len(previews), 2)
        self.assertEqual(previews[1]["name"], "Subject")
        self.assertEqual(save.call_args_list[0].args[0].shape, (1, 128, 256, 4))
        self.assertTrue(previews[0]["background"])

    def test_dynamic_stack_matches_explicit_layers(self):
        p = plate([1, 0, 0], 2)
        background = torch.zeros(3, 32, 64, 3)
        expected = render({"layer": p}, background=background)
        actual = render({}, background=None, layer_stack={"background": background, "layers": [p]})
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_autogrow_wire_names_normalize_to_layers(self):
        values = {"layers.layer_0": "first", "layers.layer_1": "second"}
        _, _, data = _io.get_finalized_class_inputs(m.FL_LayeredParallax.INPUT_TYPES(), values)
        self.assertEqual(_io.build_nested_inputs(values, data), {"layers": {"layer_0": "first", "layer_1": "second"}})

    def test_depth_sort_and_locked_identity(self):
        out, flat, depth = render({"near": plate([1, 0, 0], 2), "far": plate([0, 1, 0], 5)}, motion="locked")
        torch.testing.assert_close(out, flat)
        torch.testing.assert_close(out[:, 16, 32], torch.tensor([[1., 0, 0]]).expand(3, -1))
        self.assertEqual(depth.shape, (1, 32, 64, 3))

    def test_near_layer_moves_faster(self):
        def displacement(depth):
            out = render({"layer": plate([1, 0, 0], depth)})[0]
            weights = out[..., 0].sum(1)
            center = (weights * torch.arange(64)).sum(1) / weights.sum(1)
            return float((center[-1] - center[0]).abs())
        self.assertAlmostEqual(displacement(2) / displacement(4), 2, places=3)

    def test_input_unchanged_and_output_finite(self):
        p = plate([.2, .5, .9], 2)
        original = p["images"].clone()
        out, flat, _ = render({"a": p}, push_in=.1)
        torch.testing.assert_close(p["images"], original)
        self.assertTrue(torch.isfinite(out).all() and (out >= 0).all() and (out <= 1).all())
        self.assertFalse(torch.equal(out, flat))

    def test_still_layer_broadcasts(self):
        out = render({"a": plate([1, 0, 0], 2, count=1)})[0]
        self.assertEqual(len(out), 3)

    def test_still_plates_generate_requested_video_length(self):
        out, flat, _ = render({"a": plate([1, 0, 0], 2, count=1)}, background=torch.zeros(1, 32, 64, 3), frames=12)
        self.assertEqual(len(out), 12)
        torch.testing.assert_close(flat[0], flat[-1])
        self.assertFalse(torch.equal(out[0], out[-1]))

    def test_explicit_length_rejects_mismatched_video(self):
        with self.assertRaisesRegex(ValueError, "equal frame counts"):
            render({}, frames=12)

    def test_frame_mismatch_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "equal frame counts"):
            render({"a": plate([1, 0, 0], 2, count=2)})

    def test_explicit_mask_white_is_foreground(self):
        rgb = torch.ones(1, 32, 64, 3)
        mask = torch.zeros(1, 32, 64)
        mask[:, 8:24, 24:40] = .5
        p = m.FL_ParallaxLayer.execute(rgb, 2, 1, 0, 0, 1, mask=mask).result[0]
        out = render({"a": p}, motion="locked")[0]
        self.assertAlmostEqual(float(out[0, 16, 32, 0]), .5)
        self.assertEqual(float(out[0, 0, 0, 0]), 0)

    def test_mask_override_does_not_multiply_embedded_alpha(self):
        rgba = torch.ones(1, 32, 64, 4) * .5
        mask = torch.ones(1, 32, 64)
        p = m.FL_ParallaxLayer.execute(rgba, 2, 1, 0, 0, 1, mask=mask).result[0]
        self.assertAlmostEqual(float(render({"a": p}, motion="locked")[0][0, 16, 32, 0]), .5)

    def test_empty_layers_and_single_frame(self):
        bg = torch.rand(1, 32, 64, 3)
        out = render({}, background=bg)[0]
        torch.testing.assert_close(out, bg, atol=1e-6, rtol=1e-6)

    def test_premultiplied_sampling_has_no_white_fringe(self):
        rgba = torch.ones(1, 32, 64, 4)
        rgba[..., 3] = 0
        rgba[:, 8:24, 24:40, :3] = torch.tensor([1., 0, 0])
        rgba[:, 8:24, 24:40, 3] = 1
        p = m.FL_ParallaxLayer.execute(rgba, 2, 1.1, .017, 0, 1).result[0]
        out = render({"a": p})[0]
        self.assertEqual(float(out[..., 1:].max()), 0)

    def test_no_alpha_and_invalid_mask_rejected(self):
        with self.assertRaisesRegex(ValueError, "no cutout alpha"):
            m.FL_ParallaxLayer.execute(torch.ones(1, 32, 64, 3), 2, 1, 0, 0, 1)
        with self.assertRaisesRegex(ValueError, "match image size"):
            m.FL_ParallaxLayer.execute(torch.ones(1, 32, 64, 3), 2, 1, 0, 0, 1, mask=torch.ones(1, 8, 8))

    def test_camera_paths(self):
        self.assertEqual(m.camera_path(1, "bursts"), [0])
        self.assertEqual(m.camera_path(124, "bursts")[0], -1)
        self.assertEqual(m.camera_path(124, "bursts")[-1], 1)
        self.assertAlmostEqual(m.camera_path(124, "loop")[0], m.camera_path(124, "loop")[-1])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_gpu_cpu_agree(self):
        p = plate([.2, .4, .9], 2)
        cpu = render({"a": p})[0]
        gpu = render({"a": p}, device="auto")[0]
        torch.testing.assert_close(cpu, gpu, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
