import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


package = ModuleType("fl_seg_krea_test")
package.__path__ = [str(Path(__file__).parents[1] / "nodes")]
sys.modules[package.__name__] = package
m = importlib.import_module("fl_seg_krea_test.ksamplers.FL_KsamplerSEG_Krea")
common = importlib.import_module("fl_seg_krea_test.ksamplers.FL_KsamplerSEG_common")


def model():
    objects = {"latent_format": SimpleNamespace(spacial_downscale_ratio=8),
               "model_sampling": SimpleNamespace(percent_to_sigma=lambda p: 1 - p)}
    result = SimpleNamespace(model=Mock(spec=m.comfy.model_base.Krea2), model_options={},
                             is_dynamic=lambda: False, get_model_object=objects.__getitem__)
    result.clone = model
    result.set_model_sampler_calc_cond_batch_function = lambda fn: result.model_options.update(sampler_calc_cond_batch_function=fn)
    return result


class SegKreaTests(unittest.TestCase):
    def test_each_reference_matches_sampler_crop_without_changing_source(self):
        source = torch.rand(1, 64, 96, 3)
        original = source.clone()
        masks = torch.ones(2, 64, 96)
        regions = common.make_regions_dict(masks, masks, masks, [(1, 3, 31, 47), (25, 41, 64, 96)], (64, 96), 8, 1)
        clip = Mock(tokenizer=Mock(spec=m.Krea2Tokenizer))
        baseline = [[torch.ones(1, 2, 3), {}]]
        clip.encode_from_tokens_scheduled.return_value = baseline
        for strength in (0, .5, 1):
            clip.reset_mock()
            with patch.object(m, "encode_reference", return_value=[[torch.zeros(1, 2, 3), {}]]) as encode:
                original_model = model()
                patched, encoded, cond = m.FL_KsamplerSEG_Krea().encode(original_model, clip, regions, source, "preserve", strength)
            self.assertEqual(clip.encode_from_tokens_scheduled.call_count, 1)
            self.assertEqual(encode.call_count, 2 if strength else 0)
            self.assertIs(cond, baseline)
            self.assertEqual(original_model.model_options, {})
            self.assertIs(patched.model_options["sampler_calc_cond_batch_function"], m.reference_cond_batch)
            self.assertIsNone(regions["conditioning_per_region"])
            for i, call in enumerate(encode.call_args_list):
                by0, bx0, by1, bx1 = common.latent_bbox_from_image_bbox(regions["padded_bboxes"][i], 8, 8, 12)
                torch.testing.assert_close(call.args[2]["image"], source[:, by0 * 8:by1 * 8, bx0 * 8:bx1 * 8])
                self.assertEqual(encoded["conditioning_per_region"][i][0][-1][1]["fl_krea_reference"][1], strength)
        torch.testing.assert_close(source, original)
        with self.assertRaisesRegex(ValueError, "same size"):
            m.FL_KsamplerSEG_Krea().encode(model(), clip, regions, source[:, :32], "preserve")


if __name__ == "__main__":
    unittest.main()
