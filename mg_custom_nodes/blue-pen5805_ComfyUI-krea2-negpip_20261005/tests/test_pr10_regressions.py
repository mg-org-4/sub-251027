"""CPU tensor regressions; ComfyUI imports are stubbed, torch math is real.

Run: python -m unittest discover -s tests -v
Full model/API validation is separate and requires a ComfyUI installation.
"""
import copy
import importlib.util
import inspect
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


def load_node():
    modules = {name: types.ModuleType(name) for name in (
        "comfy", "comfy.sd1_clip", "comfy.model_management", "comfy.samplers",
        "comfy.patcher_extension", "comfy.text_encoders", "comfy.text_encoders.qwen_vl")}
    for name, module in modules.items():
        if "." in name:
            parent, attr = name.rsplit(".", 1)
            setattr(modules[parent], attr, module)
    extension = modules["comfy.patcher_extension"]
    extension.WrappersMP = types.SimpleNamespace(CALC_COND_BATCH="calc", DIFFUSION_MODEL="diffusion")
    def add_wrapper(kind, key, wrapper, options):
        options.setdefault("wrappers", {}).setdefault(kind, {})[key] = [wrapper]
    extension.add_wrapper_with_key = add_wrapper
    spec = importlib.util.spec_from_file_location("negpip_under_test", Path(__file__).resolve().parents[1] / "krea2_negpip.py")
    node = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**modules, spec.name: node}):
        spec.loader.exec_module(node)
    return node


n = load_node()


def options(uuids, metadata, **config):
    return {"uuids": uuids, n.WRAPPER_KEY: {n.NEGATIVE_METADATA_BY_UUID_KEY: metadata, **config}}


def item(positions, weights=None, source_length=4, sidecar_tokens=1):
    return {"positions": positions, "weights": weights, "source_length": source_length,
            "sidecar_tokens": sidecar_tokens}


class BatchWeights(unittest.TestCase):
    def test_condition_chunks_and_image_batches(self):
        for batch in (1, 2, 4):
            for count in (1, 2, 3):
                for strength in (0.25, 1.0, 8.0):
                    with self.subTest(batch=batch, conditions=count, strength=strength):
                        uuids = [str(i) for i in range(count)]
                        metadata = {key: item([0, 2], [i + 1., i + 2.]) for i, key in enumerate(uuids)}
                        positions = [[0, 2] for _ in range(batch * count)]
                        weights = n._resolve_negative_weights(options(uuids, metadata), positions)
                        expected_weights = [metadata[key]["weights"] for key in uuids for _ in range(batch)]
                        self.assertEqual(weights, expected_weights)
                        v = torch.ones(batch * count, 2, 6, 3)
                        self.assertEqual(n._flip_v_rows_(v, positions, strength, 4, weights), 2 * batch * count)
                        expected = torch.ones_like(v)
                        for row, row_weights in enumerate(expected_weights):
                            expected[row, :, 0, :] = -row_weights[0] * strength
                            expected[row, :, 2, :] = -row_weights[1] * strength
                        torch.testing.assert_close(v, expected, rtol=0, atol=0)

    def test_partial_missing_metadata_keeps_its_chunk(self):
        opts = options(["A", "missing", "B"], {"A": item([1], [2]), "B": item([1], [3])})
        self.assertEqual(n._resolve_negative_weights(opts, [[1]] * 6), [[2], [2], [1], [1], [3], [3]])

    def test_missing_and_mismatched_weights_fall_back_to_unit(self):
        for opts in ({}, options(["A"], {}), options(["A"], {"A": item([1], [2, 3])})):
            with self.subTest(options=opts):
                weights = n._resolve_negative_weights(opts, [[1], [1]])
                self.assertIsNone(weights)
                v = torch.ones(2, 1, 4, 1)
                n._flip_v_rows_(v, [[1], [1]], 0.5, 3, weights)
                self.assertEqual(v[:, 0, 1, 0].tolist(), [-0.5, -0.5])

    def test_untrustworthy_batch_cardinality_is_not_guessed(self):
        opts = options(["A", "B"], {"A": item([1], [2]), "B": item([2], [3])})
        for positions in ([], [[1]], [[1], [1], [2]]):
            self.assertIsNone(n._resolve_negative_weights(opts, positions))
        for batch in (0, 1, 3):
            self.assertIsNone(n._negative_metadata_from_transformer_options(opts, 4, batch))

    def test_bounds_and_signed_metadata(self):
        positions = [[-1, 0, 4]]
        weights = n._resolve_negative_weights(options(["A"], {"A": item(positions[0], [-1, -2, -3])}), positions)
        v = torch.ones(1, 1, 6, 1)
        n._flip_v_rows_(v, positions, 0.5, 4, weights)
        self.assertEqual(v.flatten().tolist(), [-1, 1, 1, 1, 1, 1])


class WrapperPaths(unittest.TestCase):
    def run_wrapper(self, context, opts, attention_mask=None):
        class Executor:
            class_obj = types.SimpleNamespace(txtfusion=None, txtmlp=None, blocks=[], _unpack_context=None,
                                              txtlayers=n.KREA2_TAP_LAYERS, txtdim=n.KREA2_TAP_DIM)
            def __call__(self, x, timesteps, context, mask, refs, transformer_options, **kwargs):
                v = torch.ones(context.shape[0], 1, context.shape[1] + 2, 1)
                for hook in transformer_options.get("patches", {}).get("attn1_patch", []):
                    v = hook(v, v, v, extra_options={"block_index": 0, "img_slice": [context.shape[1], v.shape[2]]})["v"]
                return context, mask, v, transformer_options
        return n.krea2_negpip_wrapper(Executor(), None, None, context, attention_mask, transformer_options=opts)

    def test_sidecar_and_metadata_fallback_match_contiguous_chunks(self):
        for batch in (1, 2, 4):
            for fallback in (False, True):
                for trim in (False, True):
                    with self.subTest(batch=batch, fallback=fallback, trim=trim):
                        opts = options(["A", "missing", "B"], {"A": item([1], [2]), "B": item([2], [3])})
                        plain = torch.zeros(batch, 4, 32)
                        if fallback:
                            context = torch.zeros(3 * batch, 5 if trim else 4, 32)
                        else:
                            # Keep equal-length chunks, with an empty metadata token in the unweighted one.
                            context = torch.cat([n._make_sidecar(plain, [1]), torch.zeros(batch, 5, 32), n._make_sidecar(plain, [2])])
                        mask = torch.ones(context.shape[:2])
                        stripped, stripped_mask, v, active = self.run_wrapper(context, opts, mask)
                        self.assertEqual(stripped.shape[1], 4)
                        self.assertEqual(stripped_mask.shape, (3 * batch, 4))
                        self.assertEqual(active[n.WRAPPER_KEY]["_negative_positions"], [[1]] * batch + [[]] * batch + [[2]] * batch)
                        expected = torch.ones_like(v)
                        expected[:batch, :, 1, :] = -2
                        expected[2 * batch:, :, 2, :] = -3
                        torch.testing.assert_close(v, expected, rtol=0, atol=0)

    def test_no_metadata_no_sidecar_is_noop(self):
        context = torch.zeros(2, 4, 32)
        output, _, v, _ = self.run_wrapper(context, options(["A"], {}))
        self.assertIs(output, context)
        self.assertTrue(torch.all(v == 1))

    def test_zero_strength_strips_sidecar_but_does_not_flip(self):
        context = n._make_sidecar(torch.zeros(2, 4, 32), [1])
        stripped, _, v, _ = self.run_wrapper(context, options(["A"], {"A": item([1], [2])}, value_strength=0))
        self.assertEqual(stripped.shape[1], 4)
        self.assertTrue(torch.all(v == 1))

    def test_zero_v_debug_does_not_crash(self):
        self.assertIn("flipped=0", n._v_row_magnitudes(torch.zeros(2, 1, 4, 1), [1], 3))
        cfg = {"_active": True, "_negative_positions": [[1], [1]], "_debug": True,
               "_stats": {"blocks": set(), "rows": 0, "probed": False}}
        v = torch.zeros(2, 1, 4, 1)
        result = n._make_attn1_v_flip_patch(cfg)(v, v, v, extra_options={"block_index": 0, "img_slice": [3, 4]})
        torch.testing.assert_close(result["v"], v)


class Compatibility(unittest.TestCase):
    def test_legacy_api_schema_and_widget_order(self):
        schema = n.ApplyKrea2NegPiP.INPUT_TYPES()
        self.assertEqual(list(schema["required"]), ["model", "clip", "value_strength", "patch_txtfusion_refiners"])
        self.assertEqual(list(schema["optional"]), ["block_start", "block_end", "block_stride", "debug"])
        self.assertEqual(schema["optional"]["debug"], (n.DEBUG_MODES, {"default": "off"}))
        widgets = [key for group in ("required", "optional") for key in schema[group] if key not in ("model", "clip")]
        self.assertEqual(dict(zip(widgets, [1.5, False, 2, 15, 3])), {
            "value_strength": 1.5, "patch_txtfusion_refiners": False, "block_start": 2, "block_end": 15, "block_stride": 3})

    def test_legacy_positional_apply_and_debug_modes(self):
        class Model:
            model_options = {}
            def clone(self):
                cloned = Model()
                cloned.model_options = copy.deepcopy(self.model_options)
                return cloned
            def remove_wrappers_with_key(self, *args): pass
            def add_wrapper_with_key(self, *args): pass
        node = n.ApplyKrea2NegPiP()
        clip = object()
        with patch.object(n, "_patch_clip_for_krea2_negpip", lambda value: value):
            model, returned_clip = node.apply(Model(), clip, 1.5, False, 2, 15, 3)
            cfg = model.model_options["transformer_options"][n.WRAPPER_KEY]
            self.assertIs(returned_clip, clip)
            self.assertEqual((cfg["block_start"], cfg["block_end"], cfg["block_stride"], cfg["debug"]), (2, 15, 3, "off"))
            for mode in n.DEBUG_MODES:
                model, _ = node.apply(Model(), clip, debug=mode)
                self.assertEqual(model.model_options["transformer_options"][n.WRAPPER_KEY]["debug"], mode)

    def test_embedding_weight_interpolation_preserved(self):
        sample = torch.tensor([[[[5.], [7.], [9.]]]])
        reference = torch.tensor([[[[1.], [2.], [3.]]]])
        output, positions, weights = n._apply_krea2_token_magnitudes(sample, reference, [(10, -2.), (11, .5), (12, 1.)], 0, 0)
        torch.testing.assert_close(output, torch.tensor([[[[9.], [4.5], [9.]]]]), rtol=0, atol=0)
        self.assertEqual((positions, weights), ([0], [2.]))


if __name__ == "__main__":
    unittest.main()
