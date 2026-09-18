import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


root = Path(__file__).parents[1] / "nodes/ksamplers"
package = ModuleType("fl_seg_test")
package.__path__ = [str(root)]
sys.modules[package.__name__] = package


def load(name):
    spec = importlib.util.spec_from_file_location(f"fl_seg_test.{name}", root / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load("FL_KsamplerSEG")
r = load("FL_KsamplerSEG_Regions")


def model(ratio=8):
    objects = {"latent_format": SimpleNamespace(spacial_downscale_ratio=ratio, latent_dimensions=2, latent_channels=16),
               "model_sampling": object(), "process_latent_in": lambda x: x * 2 - 1}
    return SimpleNamespace(load_device=torch.device("cpu"), model_options={}, get_model_object=objects.__getitem__)


class SegSamplerTests(unittest.TestCase):
    def advanced_args(self):
        masks = torch.zeros(2, 128, 256)
        masks[0, :, :128] = 1
        masks[1, :, 128:] = 1
        regions = r.make_regions_dict(masks, masks, masks, [(0, 0, 128, 128), (0, 128, 128, 256)], (128, 256), 8, 1)
        regions['conditioning_per_region'] = [('first', 'negative'), ('second', 'negative')]
        return dict(model=model(), regions=regions, latent_image={'samples': torch.ones(1, 16, 16, 32), 'metadata': 'keep'},
                    positive='default', negative='negative', noise_seed=10, steps=8, cfg=1,
                    sampler_name='euler', scheduler='simple', start_at_step=2, end_at_step=5,
                    add_noise='enable', return_with_leftover_noise='enable', cascade_start_corner='top_left')

    def test_advanced_uses_native_sigma_window_noise_controls_and_progress(self):
        for add_noise in ('enable', 'disable'):
            for leftover in ('enable', 'disable'):
                args = self.advanced_args()
                args.update(add_noise=add_noise, return_with_leftover_noise=leftover)
                calls, progress = [], []

                def sample(model, noise, positive, negative, cfg, device, sampler, sigmas, options, **kwargs):
                    calls.append((noise.clone(), positive, sigmas.clone(), kwargs['seed']))
                    latent = kwargs['latent_image']
                    for step in range(len(sigmas)):
                        kwargs['callback'](step, latent, latent, len(sigmas) - 1)
                    return latent + 1

                with patch.object(m.comfy.samplers, 'calculate_sigmas', side_effect=lambda sampling, scheduler, steps: torch.linspace(1, 0, steps + 1)), \
                     patch.object(m.comfy.samplers, 'sample', side_effect=sample), \
                     patch.object(m.latent_preview, 'prepare_callback', return_value=lambda step, x0, x, total: progress.append((step, total))) as prepare:
                    result = m.FL_KsamplerSEGAdvanced().sample(**args)[0]
                expected = torch.tensor([.75, .625, .5, .375 if leftover == 'enable' else 0])
                self.assertEqual([c[1] for c in calls], ['first', 'second'])
                self.assertEqual([c[3] for c in calls], [10, 11])
                for noise, _, sigmas, _ in calls:
                    torch.testing.assert_close(sigmas, expected)
                    self.assertEqual(bool(noise.count_nonzero()), add_noise == 'enable')
                prepare.assert_called_once_with(args['model'], 6)
                self.assertEqual(progress, [(i, 6) for i in (0, 1, 2, 2, 3, 4, 5, 5)])
                torch.testing.assert_close(result['samples'], torch.full_like(result['samples'], 2))
                torch.testing.assert_close(args['latent_image']['samples'], torch.ones_like(result['samples']))
                self.assertEqual(result['metadata'], 'keep')

    def test_advanced_empty_windows_do_not_sample_or_add_noise(self):
        for start, end in ((5, 5), (6, 3), (8, 10000), (0, 0)):
            args = self.advanced_args()
            args.update(start_at_step=start, end_at_step=end)
            with patch.object(m.comfy.sample, 'prepare_noise') as noise, patch.object(m.comfy.sample, 'sample') as sample:
                result = m.FL_KsamplerSEGAdvanced().sample(**args)
            self.assertIs(result[0], args['latent_image'])
            noise.assert_not_called()
            sample.assert_not_called()

    def test_advanced_window_matches_regular_denoise_schedule_and_noise(self):
        args = self.advanced_args()
        calls = []

        def sample(model, noise, positive, negative, cfg, device, sampler, sigmas, options, **kwargs):
            calls.append((noise.clone(), sigmas.clone()))
            return kwargs['latent_image']

        with patch.object(m.comfy.samplers, 'calculate_sigmas', side_effect=lambda sampling, scheduler, steps: torch.linspace(1, 0, steps + 1)), \
             patch.object(m.comfy.samplers, 'sample', side_effect=sample), \
             patch.object(m.latent_preview, 'prepare_callback', return_value=lambda *args: None):
            regular = {k: args[k] for k in ('model', 'regions', 'latent_image', 'positive', 'negative', 'cfg', 'sampler_name', 'scheduler', 'cascade_start_corner')}
            m.FL_KsamplerSEG().sample(**regular, seed=10, steps=4, denoise=.2)
            args.update(steps=20, start_at_step=16, end_at_step=10000, return_with_leftover_noise='disable')
            m.FL_KsamplerSEGAdvanced().sample(**args)
        self.assertEqual(len(calls), 4)
        for regular, advanced in zip(calls[:2], calls[2:]):
            torch.testing.assert_close(regular[0], advanced[0])
            torch.testing.assert_close(regular[1], advanced[1])
            self.assertEqual(len(advanced[1]), 5)

    def test_advanced_schema_has_step_controls_without_denoise(self):
        required = m.FL_KsamplerSEGAdvanced.INPUT_TYPES()['required']
        self.assertNotIn('denoise', required)
        self.assertNotIn('seed', required)
        for name in ('add_noise', 'noise_seed', 'start_at_step', 'end_at_step', 'return_with_leftover_noise'):
            self.assertIn(name, required)
        self.assertIn('denoise', m.FL_KsamplerSEG.INPUT_TYPES()['required'])

    def test_pixel_overlap_has_bounded_support_and_full_coverage(self):
        hard = torch.zeros(2, 8, 20)
        hard[0, :, :10] = 1
        hard[1, :, 10:] = 1
        for pixel in (1, 8):
            for feather in (0, pixel, 2 * pixel):
                masks = r.FL_KsamplerSEG_Regions._pixel_masks(hard, 2 * pixel, feather, pixel, pixel)
                self.assertTrue(torch.all(masks.sum(0) >= 1))
                self.assertTrue(torch.all(masks[0, :, 12:] == 0))
                self.assertTrue(torch.all(masks[1, :, :8] == 0))
                self.assertTrue(torch.all(masks[:, :, 8:12] > 0))
        torch.testing.assert_close(r.FL_KsamplerSEG_Regions._pixel_masks(hard, 0, 0, 8, 8), hard)

    def test_feather_does_not_grow_crop_and_context_does_not_grow_mask(self):
        node = r.FL_KsamplerSEG_Regions()
        args = dict(num_regions=4, relaxation_iterations=2, region_overlap_factor=.15,
                    edge_softness=.1, context_padding_factor=.2, safe_zone_feather_px=0,
                    downscale_ratio=8, seed=12, show_preview=False, preview_mode="sampler_crops",
                    image=torch.zeros(1, 256, 256, 3), margin_mode="pixels", overlap_width_px=32)
        a = node.build(**args, feather_width_px=0, context_padding_px=0)[0]
        b = node.build(**args, feather_width_px=16, context_padding_px=0)[0]
        c = node.build(**args, feather_width_px=16, context_padding_px=24)[0]
        self.assertEqual(a["padded_bboxes"], b["padded_bboxes"])
        self.assertNotEqual(b["padded_bboxes"], c["padded_bboxes"])
        torch.testing.assert_close(b["write_masks"], c["write_masks"])
        zero = node.build(**dict(args, overlap_width_px=0), feather_width_px=0, context_padding_px=0)[0]
        self.assertTrue(torch.all((zero["write_masks"] > 0).sum(0) == 1))
        with self.assertRaisesRegex(ValueError, "half the overlap"):
            node.build(**args, feather_width_px=24)

    def test_wrong_image_size_fails_instead_of_guessing_downscale(self):
        with self.assertRaisesRegex(ValueError, "same resized image"):
            m.FL_KsamplerSEG._resolve_downscale(model(), {}, 512, 412, 1152, 928)
        self.assertEqual(m.FL_KsamplerSEG._resolve_downscale(model(), {}, 512, 412, 4096, 3296), 8)

    def test_model_ratio_and_vae_rounding(self):
        self.assertEqual(m.FL_KsamplerSEG._resolve_downscale(model(32), {}, 32, 24, 1025, 769), 32)
        self.assertEqual(m.FL_KsamplerSEG._resolve_downscale(model(32), {}, 33, 25, 1025, 769), 32)
        with self.assertRaises(ValueError):
            m.FL_KsamplerSEG._resolve_downscale(model(32), {}, 34, 25, 1025, 769)

    def test_full_canvas_preview_progress_and_latent_are_independent(self):
        for shape in ((1, 16, 24, 32), (1, 16, 1, 24, 32)):
            with self.subTest(shape=shape):
                source = torch.ones(shape)
                original = source.clone()
                spec = dict(latent_bbox=(4, 8, 20, 24), seed=3, region_index=2,
                            write_lat=torch.ones(16, 16), comp_lat=torch.full((16, 16), .5),
                            pos_cond=object(), neg_cond=object())
                frames = []

                def preview(step, x0, x, total):
                    frames.append((step, x0.clone(), total))

                def sample(model, noise, steps, cfg, sampler, scheduler, positive, negative, latent, **kwargs):
                    self.assertIs(positive, spec["pos_cond"])
                    self.assertIs(negative, spec["neg_cond"])
                    for step in range(steps):
                        kwargs["callback"](step, torch.full_like(latent, 5 + step), latent, steps)
                    return torch.full_like(latent, 9)

                with patch.object(m.comfy.sample, "sample", side_effect=sample):
                    result = m.FL_KsamplerSEG()._sample_one_region_full(
                        model=model(), source_latent=source, spec=spec, steps=2, cfg=1,
                        sampler_name="euler", scheduler="simple", denoise=.3,
                        preview_callback=preview, step_offset=4, total_steps=8)
                self.assertEqual([(i, total) for i, _, total in frames], [(4, 8), (5, 8)])
                for i, (_, frame, _) in enumerate(frames):
                    expected = torch.ones_like(source)
                    expected[..., 4:20, 8:24] = 3 + i * .5
                    torch.testing.assert_close(frame, expected)
                torch.testing.assert_close(source, original)
                torch.testing.assert_close(result, torch.full_like(result, 9))

    def test_raw_regions_keep_reference_conditioning(self):
        marker = ("reference_0", .42, (1, 1, 0, 0))
        positive = [[torch.zeros(1), {"fl_krea_reference": marker}]]
        negative = [[torch.zeros(1), {}]]
        regions = dict(padded_bboxes=[(0, 0, 128, 128)], write_masks=torch.ones(1, 128, 128),
                       composite_masks=torch.ones(1, 128, 128))
        spec = m.FL_KsamplerSEG()._build_region_spec(
            regions=regions, region_index=0, downscale=8, latent_h=16, latent_w=16,
            device="cpu", per_region_cond=None, cond_pos_default=positive,
            cond_neg_default=negative, base_seed=3)
        self.assertIs(spec["pos_cond"], positive)
        self.assertEqual(spec["pos_cond"][0][1]["fl_krea_reference"], marker)

    def test_crop_preview_does_not_change_masks(self):
        image = np.full((64, 64, 3), 160, dtype=np.uint8)
        masks = torch.zeros(2, 64, 64)
        masks[0, :, :32] = 1
        masks[1, :, 32:] = 1
        original = masks.clone()
        preview = r.FL_KsamplerSEG_Regions._build_crop_viz(image, masks, [(0, 0, 64, 48), (0, 16, 64, 64)])
        self.assertEqual(preview.shape, (350, 640, 3))
        torch.testing.assert_close(masks, original)


if __name__ == "__main__":
    unittest.main()
