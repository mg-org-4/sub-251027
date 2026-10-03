import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import comfy.latent_formats
import comfy.sample


package = ModuleType("fl_plus_test")
package.__path__ = [str(Path(__file__).parents[1] / "nodes/ksamplers")]
sys.modules[package.__name__] = package
m = importlib.import_module("fl_plus_test.FL_KsamplerPlus")


class KsamplerPlusTests(unittest.TestCase):
    def sample(self, source, latent_format, x_slices=1, y_slices=1, batch_size=1, overlap=0, **metadata):
        objects = {"latent_format": latent_format, "diffusion_model.vae_scale_factors": (4, 8, 8)}
        model = SimpleNamespace(get_model_object=objects.__getitem__)
        calls = []

        def sampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, **kwargs):
            calls.append(latent["samples"].shape)
            native = comfy.sample.fix_empty_latent_channels(model, latent["samples"])
            return ({"samples": native + 1},)

        with patch.object(m.comfy.model_management, "get_torch_device", return_value=torch.device("cpu")), \
             patch.object(m, "common_ksampler", side_effect=sampler):
            result = m.FL_KsamplerPlus().sample(
                model, [], [], 42, 8, 1, "euler", "simple", 1, "latent",
                x_slices, y_slices, overlap, batch_size, False,
                latent_image={"samples": source, **metadata})[3]["samples"]
        return result, calls

    def test_empty_image_latent_uses_native_krea_shape_before_sampling(self):
        source = torch.zeros(1, 4, 128, 128)
        result, calls = self.sample(source, comfy.latent_formats.Wan21(), downscale_ratio_spacial=8)
        self.assertEqual(calls, [torch.Size((1, 16, 1, 128, 128))])
        self.assertEqual(result.shape, (1, 16, 1, 128, 128))
        torch.testing.assert_close(result, torch.ones_like(result))
        self.assertEqual(source.shape, (1, 4, 128, 128))
        self.assertEqual(source.count_nonzero(), 0)

    def test_tiles_preserve_native_dimensions_and_image_batches(self):
        for shape, latent_format in (((2, 16, 16, 16), comfy.latent_formats.Flux()),
                                     ((2, 16, 16, 16), comfy.latent_formats.Wan21()),
                                     ((2, 16, 3, 16, 16), comfy.latent_formats.Wan21())):
            for batch_size in (1, 4):
                with self.subTest(shape=shape, batch_size=batch_size):
                    source = torch.rand(shape) + 1
                    expected = comfy.sample.fix_empty_latent_channels(
                        SimpleNamespace(get_model_object=lambda name: latent_format), source) + 1
                    result, calls = self.sample(source, latent_format, 2, 2, batch_size, .25)
                    torch.testing.assert_close(result, expected)
                    self.assertEqual(len(calls), 4 // batch_size)
                    self.assertTrue(all(shape[-2:] == (10, 10) for shape in calls))

    def test_empty_latent_scale_metadata_is_applied_before_tiling(self):
        source = torch.zeros(1, 4, 16, 16)
        result, calls = self.sample(source, comfy.latent_formats.Wan21(), 2, 2,
                                    downscale_ratio_spacial=16, downscale_ratio_temporal=8)
        self.assertEqual(result.shape, (1, 16, 2, 32, 32))
        self.assertEqual(calls, [torch.Size((1, 16, 2, 16, 16))] * 4)
        torch.testing.assert_close(result, torch.ones_like(result))

    def test_image_noise_mask_can_be_cropped_with_native_5d_latent(self):
        source = torch.ones(1, 16, 16, 16)
        mask = torch.ones(1, 1, 16, 16)
        result, calls = self.sample(source, comfy.latent_formats.Wan21(), 2, 2, noise_mask=mask)
        self.assertEqual(result.shape, (1, 16, 1, 16, 16))
        torch.testing.assert_close(result, torch.full_like(result, 2))


if __name__ == "__main__":
    unittest.main()
