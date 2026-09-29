import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import comfy.sampler_helpers


spec = importlib.util.spec_from_file_location("fl_krea_test", Path(__file__).parents[1] / "nodes/conditioning/FL_KreaReference.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def model():
    result = SimpleNamespace(model=Mock(spec=m.comfy.model_base.Krea2), model_options={}, is_dynamic=lambda: False,
                             get_model_object=lambda name: SimpleNamespace(percent_to_sigma=lambda p: 1 - p))
    result.clone = model
    result.set_model_sampler_calc_cond_batch_function = lambda fn: result.model_options.update(sampler_calc_cond_batch_function=fn)
    return result


def conditioning(value):
    return [[torch.full((1, 2, 30720), float(value)), {"value": value}]]


def reference(**kwargs):
    kwargs.setdefault("reference_mode", "full")
    return m.FL_KreaReference.execute(torch.zeros(1, 48, 96, 3), **kwargs).result[0]


class KreaReferenceTests(unittest.TestCase):
    def predict(self, branches, influence=.7, sigma=.5):
        guider = m.KreaReferenceGuider(model(), conditioning(2), branches, influence)
        guider.conds = guider.original_conds
        guider.inner_model = object()
        calls = []

        def sample(inner, x, timestep, negative, positive, cfg, **kwargs):
            calls.append(positive[0]["value"])
            return torch.full_like(x, positive[0]["value"])

        x = torch.zeros(1, 16, 2, 2)
        with patch.object(m.comfy.samplers, "sampling_function", side_effect=sample):
            result = guider.predict_noise(x, torch.tensor([sigma]))
        return result, calls

    def test_zero_influence_and_no_references_use_baseline(self):
        for branches, strength in (([], .7), ([(reference(), conditioning(8))], 0)):
            output, calls = self.predict(branches, strength)
            self.assertEqual(calls, [2])
            torch.testing.assert_close(output, torch.full_like(output, 2))

    def test_single_reference_endpoint_skips_baseline(self):
        output, calls = self.predict([(reference(), conditioning(8))], 1)
        self.assertEqual(calls, [8])
        torch.testing.assert_close(output, torch.full_like(output, 8))

    def test_blend_formula_order_and_weight_scaling(self):
        branches = [(reference(weight=.25), conditioning(4)), (reference(weight=.75), conditioning(8))]
        a, calls = self.predict(branches, .5)
        b, _ = self.predict(list(reversed(branches)), .5)
        c, _ = self.predict([(dict(r, weight=r["weight"] * .5), cond) for r, cond in branches], .5)
        self.assertEqual(calls, [2, 4, 8])
        torch.testing.assert_close(a, torch.full_like(a, 4.5))
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(c, torch.full_like(c, 3.25))

    def test_single_reference_slider_scales_contribution(self):
        for weight in (0, .05, .25, .5, 1):
            with self.subTest(weight=weight):
                output, _ = self.predict([(reference(weight=weight), conditioning(8))], .7)
                torch.testing.assert_close(output, torch.full_like(output, 2 + .7 * weight * (8 - 2)))

    def test_removing_reference_does_not_boost_remaining_reference(self):
        a = (reference(weight=.25), conditioning(4))
        b = (reference(weight=.5), conditioning(8))
        both, _ = self.predict([a, b], 1)
        only_b, _ = self.predict([b], 1)
        zero_a, _ = self.predict([(reference(weight=0), conditioning(4)), b], 1)
        torch.testing.assert_close(both - only_b, torch.full_like(both, .25 * (4 - 2)))
        torch.testing.assert_close(only_b, zero_a)

    def test_combined_weights_above_one_keep_unit_prediction_gain(self):
        output, _ = self.predict([(reference(weight=1), conditioning(4)), (reference(weight=1), conditioning(4))], 1)
        torch.testing.assert_close(output, torch.full_like(output, 6))
        same, _ = self.predict([(reference(weight=1), conditioning(2)), (reference(weight=1), conditioning(2))], 1)
        torch.testing.assert_close(same, torch.full_like(same, 2))

    def test_schedule_does_not_renormalize_other_reference(self):
        branches = [(reference(weight=.5, start=.6), conditioning(4)), (reference(weight=.5), conditioning(8))]
        a, calls = self.predict(branches, 1, sigma=.7)
        self.assertEqual(calls, [2, 8])
        torch.testing.assert_close(a, torch.full_like(a, 5))

    def test_envelope_boundaries_and_fades(self):
        bounds = (.8, .6, .4, .2)
        for sigma, expected in ((1, 0), (.8, 0), (.7, .5), (.6, 1), (.5, 1), (.3, .5), (.2, 0), (.1, 0)):
            self.assertAlmostEqual(m.sigma_envelope(sigma, bounds), expected)
        self.assertEqual(m.sigma_envelope(1, (1, 1, 0, 0)), 1)
        self.assertEqual(m.sigma_envelope(0, (1, 1, 0, 0)), 1)

    def test_repeat_evaluations_follow_sigma_not_call_count(self):
        branches = [(reference(start=.25, end=.75), conditioning(8))]
        for sigma in (.9, .5, .5, .1, .9):
            output, _ = self.predict(branches, 1, sigma)
            torch.testing.assert_close(output, torch.full_like(output, 8 if sigma == .5 else 2))

    def test_reference_validation(self):
        for kwargs in (dict(start=.5, end=.5), dict(start=-.1), dict(end=1.1), dict(fade=.6),
                       dict(weight=-1), dict(weight=1.01), dict(weight=10), dict(weight=float("nan")),
                       dict(weight=float("inf")), dict(role="unknown"), dict(resolution=999)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                reference(**kwargs)
        with self.assertRaisesRegex(ValueError, "one RGB image"):
            m.FL_KreaReference.execute(torch.zeros(2, 32, 32, 3))

    def clip(self):
        clip = Mock()
        clip.tokenizer = Mock(spec=m.Krea2Tokenizer)
        clip.tokenize.side_effect = lambda text, **kwargs: (text, kwargs)
        clip.encode_from_tokens_scheduled.side_effect = lambda tokens: conditioning(len(tokens[0]))
        return clip

    def test_disabled_and_zero_weight_references_are_not_encoded(self):
        clip = self.clip()
        with patch.object(m, "encode_reference", return_value=conditioning(8)) as encode:
            guider = m.FL_KreaReferenceGuider.execute(model(), clip, "dog", references={
                "reference_0": reference(enabled=False), "reference_1": reference(weight=0),
                "reference_2": reference(),
            }).result[0]
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(len(guider.references), 1)
        self.assertEqual(guider.references[0][1], 1)

    def test_zero_strength_skips_reference_encoding(self):
        clip = self.clip()
        with patch.object(m, "encode_reference") as encode:
            m.FL_KreaReferenceGuider.execute(model(), clip, "dog", influence=0, references={"reference_0": reference()})
        encode.assert_not_called()
        clip.tokenize.assert_called_once_with("dog")

    def test_same_shape_images_and_clip_changes_are_reencoded(self):
        clip = self.clip()
        ref = reference()
        encoded_images = []

        def tokenize(text, **kwargs):
            if "images" in kwargs:
                encoded_images.append(kwargs["images"][0].clone())
            return text, kwargs

        clip.tokenize.side_effect = tokenize
        for image in (torch.zeros_like(ref["image"]), torch.ones_like(ref["image"])):
            m.FL_KreaReferenceGuider.execute(model(), clip, "dog", references={"reference_0": dict(ref, image=image)})
        other_clip = self.clip()
        m.FL_KreaReferenceGuider.execute(model(), other_clip, "dog", references={"reference_0": ref})
        self.assertEqual(len(encoded_images), 2)
        self.assertFalse(torch.equal(*encoded_images))
        self.assertEqual(other_clip.encode_from_tokens_scheduled.call_count, 2)

    def test_each_branch_sees_one_image_and_preserves_taps(self):
        clip = self.clip()
        refs = {"reference_0": reference(), "reference_1": reference(role="palette")}
        guider = m.FL_KreaReferenceGuider.execute(model(), clip, "dog", references=refs).result[0]
        calls = clip.tokenize.call_args_list
        self.assertEqual(len(calls), 3)
        for call in calls[1:]:
            self.assertEqual(len(call.kwargs["images"]), 1)
            self.assertEqual(call.args[0].count("<|image_pad|>"), 1)
            self.assertIn("dog", call.args[0])
        self.assertIn(m.ROLES["palette"], calls[2].args[0])
        for cond in guider.original_conds.values():
            self.assertEqual(cond[0]["cross_attn"].shape[-1], 30720)
            self.assertTrue(torch.all(cond[0]["cross_attn"] != 0))

    def test_model_and_guidance_mismatch_rejected_before_encoding(self):
        clip = self.clip()
        bad = model()
        bad.model = object()
        with self.assertRaisesRegex(ValueError, "Krea 2"):
            m.FL_KreaReferenceGuider.execute(bad, clip, "dog")
        bad = model()
        bad.model_options["sampler_cfg_function"] = lambda args: None
        with self.assertRaisesRegex(ValueError, "guidance patches"):
            m.FL_KreaReferenceGuider.execute(bad, clip, "dog")
        clip.encode_from_tokens_scheduled.assert_not_called()

    def test_schema_has_optional_growing_references(self):
        schema = m.FL_KreaReferenceGuider.INPUT_TYPES()
        self.assertEqual(schema["optional"]["references"][1]["template"]["min"], 0)

    def test_ksampler_outputs_preserve_guider_and_original_model(self):
        original = model()
        guider, patched, cond = m.FL_KreaReferenceGuider.execute(original, self.clip(), "dog", references={
            "reference_0": reference(weight=.25), "reference_1": reference(weight=.5),
        }).result
        self.assertIsInstance(guider, m.KreaReferenceGuider)
        self.assertEqual(original.model_options, {})
        self.assertIs(patched.model_options["sampler_calc_cond_batch_function"], m.reference_cond_batch)
        self.assertEqual(len(cond), 3)
        self.assertNotIn("fl_krea_reference", cond[0][1])
        self.assertEqual(cond[1][1]["fl_krea_reference"][1], .7 * .25)

    def test_ksampler_matches_guider_weights_schedules_and_cfg(self):
        x = torch.zeros(1, 16, 2, 2)

        def batch(inner, conds, x, sigma, options):
            return [torch.full_like(x, cond[0]["value"] if cond else 0) for cond in conds]

        for weight in (0, .05, .5, 1):
            for sigma in (.9, .5, .1):
                refs = [reference(weight=weight, start=.2, end=.8, fade=.25), reference(weight=.3)]
                branches = list(zip(refs, (conditioning(8), conditioning(4))))
                expected, _ = self.predict(branches, .7, sigma)
                clip = self.clip()
                clip.encode_from_tokens_scheduled.side_effect = lambda tokens: conditioning(2)
                with patch.object(m, "encode_reference", side_effect=[cond for ref, cond in branches if ref["weight"] > 0]):
                    _, patched, cond = m.FL_KreaReferenceGuider.execute(model(), clip, "dog", references=dict(enumerate(refs))).result
                prepared = comfy.sampler_helpers.convert_cond(cond)
                negative = comfy.sampler_helpers.convert_cond(conditioning(-2))
                with patch.object(m.comfy.samplers, "calc_cond_batch", side_effect=batch):
                    for cfg in (1., 2.):
                        actual = m.comfy.samplers.sampling_function(object(), x, torch.tensor([sigma]), negative, prepared, cfg, model_options=patched.model_options)
                        torch.testing.assert_close(actual, -2 + (expected + 2) * cfg)

    def test_batch_hook_leaves_plain_and_missing_conditioning_unchanged(self):
        cond = [{"value": 2}]
        with patch.object(m.comfy.samplers, "calc_cond_batch", side_effect=lambda model, conds, *args: conds):
            out = m.reference_cond_batch(dict(model=object(), conds=[cond, None], input=None, sigma=torch.tensor([.5]), model_options={}))
        self.assertEqual(out, [cond, None])

    def test_average_blends_predictions_identically_for_both_sampler_outputs(self):
        cases = [
            ([], [], 2, 2),
            ([reference()], [8], 8, 8),
            ([reference() for _ in range(4)], [4, 6, 8, 10], 22, 7),
            ([reference(), reference(weight=.5)], [4, 8], 7, 4.5),
            ([reference(weight=0), reference()], [4, 8], 8, 5),
            ([reference(enabled=False), reference()], [4, 8], 8, 8),
            ([reference(start=.6), reference()], [4, 8], 8, 5),
            ([reference(start=.25, end=.75, fade=.5), reference()], [4, 8], 10, 6),
        ]
        x = torch.zeros(1, 16, 2, 2)
        for refs, values, additive, average in cases:
            for mode in ("add", "average"):
                for amount in (0, .5, 1):
                    for influence in (0, .7, 1):
                        with self.subTest(values=values, mode=mode, amount=amount, influence=influence):
                            clip = self.clip()
                            clip.encode_from_tokens_scheduled.side_effect = lambda tokens: conditioning(2)
                            encoded = [conditioning(value) for ref, value in zip(refs, values) if ref["enabled"] and ref["weight"] > 0]
                            with patch.object(m, "encode_reference", side_effect=encoded):
                                guider, patched, cond = m.FL_KreaReferenceGuider.execute(
                                    model(), clip, "dog", influence, dict(enumerate(refs)), mode, amount).result
                            guider.conds = guider.original_conds
                            guider.inner_model = object()
                            with patch.object(m.comfy.samplers, "sampling_function", side_effect=lambda inner, x, t, neg, pos, cfg, **kwargs: torch.full_like(x, pos[0]["value"])):
                                guided = guider.predict_noise(x, torch.tensor([.5]))
                            with patch.object(m.comfy.samplers, "calc_cond_batch", side_effect=lambda inner, conds, x, sigma, options: [torch.full_like(x, cond[0]["value"] if cond else 0) for cond in conds]):
                                sampled = m.comfy.samplers.sampling_function(
                                    object(), x, torch.tensor([.5]), None, comfy.sampler_helpers.convert_cond(cond), 1,
                                    model_options=patched.model_options)
                            blended = additive + (average - additive) * amount if mode == "average" else additive
                            expected = torch.full_like(x, 2 + influence * (blended - 2))
                            torch.testing.assert_close(guided, expected)
                            torch.testing.assert_close(sampled, expected)

    def test_invalid_blend_controls_fail_before_encoding(self):
        clip = self.clip()
        for kwargs in (dict(blend_mode="unknown"), dict(average_amount=-.1), dict(average_amount=1.1), dict(average_amount=float("nan"))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                m.FL_KreaReferenceGuider.execute(model(), clip, "dog", **kwargs)
        clip.encode_from_tokens_scheduled.assert_not_called()

    def test_context_mode_removes_visual_prefix_and_copies_suffix(self):
        clip = self.clip()
        clip.tokenize.side_effect = None
        clip.tokenize.return_value = {"qwen3vl_4b": [[(151652, 1), ({"type": "image"}, 1), (151653, 1), (12, 1), (13, 1)]]}
        tensor = torch.randn(1, 12, 30720)
        mask = torch.ones(1, 12)
        clip.encode_from_tokens_scheduled.side_effect = None
        clip.encode_from_tokens_scheduled.return_value = [[tensor, {"attention_mask": mask, "other": 7}]]
        result = m.encode_reference(clip, "dog", reference(reference_mode="context"))
        torch.testing.assert_close(result[0][0], tensor[:, -2:])
        self.assertEqual(result[0][0].shape[-1], 30720)
        self.assertNotEqual(result[0][0].untyped_storage().data_ptr(), tensor.untyped_storage().data_ptr())
        self.assertEqual(result[0][1]["attention_mask"].shape, (1, 2))
        self.assertEqual(result[0][1]["other"], 7)
        self.assertEqual(mask.shape, (1, 12))


if __name__ == "__main__":
    unittest.main()
