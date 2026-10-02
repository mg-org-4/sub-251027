import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from comfy_api.latest import _io


spec = importlib.util.spec_from_file_location("poster_test", Path(__file__).parents[1] / "nodes/vfx/FL_PosterLayers.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def plan(count):
    return m.parse_plan(json.dumps([dict(name=f"Layer {i}", prompt=f"Object {i}", kind="art" if i else "background", depth=2 if i else 12) for i in range(count)]), count)


def graph(entries, edits=None):
    m.GraphBuilder.set_default_prefix("test", 0)
    settings = json.dumps({"plan_key": m.plan_key(entries), "layers": edits or {}})
    return m.extraction_graph(entries, ["model", 0], ["clip", 0], ["vae", 0], ["image", 0], 777, 30, 4, "euler", "simple", settings).expand


class PosterTests(unittest.TestCase):
    def test_auto_merges_duplicate_targets(self):
        entries = plan(4)
        entries[1]["prompt"] = "black speakers beside the woman"
        entries[2]["prompt"] = "black speakers"
        with patch.object(m.TextGenerate, "execute", return_value=m.io.NodeOutput(json.dumps(entries))):
            result = m.FL_PosterLayerPlanner.execute(torch.zeros(1, 8, 8, 3), "auto", 4, "balanced", "[]", object())
        self.assertEqual(len(result.result[0]), 3)
        self.assertEqual([r["id"] for r in result.result[0]], ["layer_0", "layer_1", "layer_2"])
        self.assertIn(" / ", result.result[0][1]["name"])

    def test_auto_targets_do_not_include_other_object_anchors(self):
        entries = plan(4)
        entries[0]["prompt"] = "studio wall behind a table"
        entries[1]["prompt"] = "large yellow disk behind the woman"
        entries[2]["prompt"] = "silver stars positioned above the speakers"
        entries[3]["prompt"] = "brass lamp beside a sofa"
        with patch.object(m.TextGenerate, "execute", return_value=m.io.NodeOutput(json.dumps(entries))):
            result = m.FL_PosterLayerPlanner.execute(torch.zeros(1, 8, 8, 3), "auto", 4, "balanced", "[]", object())
        self.assertEqual([p["prompt"] for p in result.result[0]], ["studio wall behind a table", "large yellow disk", "silver stars", "brass lamp"])
        manual = m.FL_PosterLayerPlanner.execute(torch.zeros(1, 8, 8, 3), "manual", 4, "balanced", json.dumps(entries))
        self.assertEqual(manual.result[0][1]["prompt"], entries[1]["prompt"])

    def test_auto_repairs_over_budget_plan_once(self):
        responses = [m.io.NodeOutput(json.dumps(plan(4))), m.io.NodeOutput(json.dumps(plan(3)))]
        with patch.object(m.TextGenerate, "execute", side_effect=responses) as generate:
            result = m.FL_PosterLayerPlanner.execute(torch.zeros(1, 8, 8, 3), "auto", 3, "balanced", "[]", object(), "photography")
        self.assertEqual(len(result.result[0]), 3)
        self.assertEqual(generate.call_count, 2)
        self.assertIn("photography", generate.call_args_list[0].args[1])
        self.assertIn("Do not invent a headline", generate.call_args_list[0].args[1])
        self.assertIn("returned 4 layers", generate.call_args_list[1].args[1])

    def test_auto_repair_is_bounded(self):
        with patch.object(m.TextGenerate, "execute", return_value=m.io.NodeOutput("[]")) as generate:
            with self.assertRaises(ValueError):
                m.FL_PosterLayerPlanner.execute(torch.zeros(1, 8, 8, 3), "auto", 3, "balanced", "[]", object())
        self.assertEqual(generate.call_count, 2)

    def test_asset_defers_extraction_until_after_cache_lookup(self):
        image_input = m.FL_PosterLayerAsset.INPUT_TYPES()["required"]["image"]
        self.assertTrue(image_input[1]["lazy"])
        self.assertEqual(m.FL_PosterLayerAsset.check_lazy_status(), ["image"])
        self.assertEqual(m.FL_PosterLayerAsset.check_lazy_status(image=torch.zeros(1, 8, 8, 4)), [])

    def test_variable_layer_counts(self):
        for count in (1, 3, 6, 12, 32):
            g = graph(plan(count))
            self.assertEqual(sum(n["class_type"] == "KSampler" for n in g.values()), count)
            self.assertEqual(sum(n["class_type"] == "VAEEncode" for n in g.values()), 1)
            self.assertEqual(sum(n["class_type"] == "EmptyQwenImageLayeredLatentImage" for n in g.values()), 1)

    def test_layout_edits_do_not_change_sampling_graph(self):
        p = plan(6)
        old, new = graph(p), graph(p, {"layer_2": {"depth": 7, "visible": False, "scale": 1.2}})
        changed = [key for key in old if old[key] != new[key]]
        self.assertEqual(changed, ["test.0.0.stack"])

    def test_reroll_changes_only_one_sampler(self):
        p = plan(5)
        old, new = graph(p), graph(p, {"layer_3": {"revision": 1}})
        changed = [key for key in old if old[key] != new[key] and old[key]["class_type"] == "KSampler"]
        self.assertEqual(changed, ["test.0.0.layer_3_sample"])

    def test_prompt_edit_changes_only_target_conditioning(self):
        p = plan(5)
        old, new = graph(p), graph(p, {"layer_4": {"prompt": 'The red "AFTER HOURS" headline'}})
        changed = [key for key in old if old[key] != new[key]]
        self.assertEqual(set(changed), {"test.0.0.layer_4_text", "test.0.0.stack"})

    def test_stale_overrides_not_applied_to_new_plan(self):
        p = plan(3)
        edits = json.dumps({"plan_key": m.plan_key(p), "layers": {"layer_1": {"prompt": "wrong old object"}}})
        p[1]["prompt"] = "New artwork"
        self.assertEqual(m.apply_overrides(p, edits)[1]["prompt"], "New artwork")

    def test_bad_plan_rejected(self):
        for value in ('not json', '{}', '[]', '[{"kind":"art","name":"a","prompt":"a"}]', '[{"kind":"background","name":"a","prompt":"a","depth":NaN}]'):
            with self.assertRaises(ValueError):
                m.parse_plan(value, 6)
        with self.assertRaises(ValueError):
            m.parse_plan(json.dumps(plan(4)), 3)

    def test_fenced_thinking_output(self):
        value = '<think>reasoning</think>\n```json\n' + json.dumps(plan(2)) + '\n```'
        self.assertEqual(len(m.parse_plan(value, 6)), 2)

    def test_bad_overrides_rejected(self):
        p = plan(2)
        for edit in ({"depth": float("nan")}, {"scale": 0}, {"prompt": ""}, {"visible": "false"}, {"revision": -1}):
            with self.assertRaises(ValueError):
                m.apply_overrides(p, json.dumps({"plan_key": m.plan_key(p), "layers": {"layer_1": edit}}))

    def test_asset_path_validation_before_save(self):
        with self.assertRaises(ValueError):
            m.FL_PosterLayerAsset.execute(torch.ones(1, 8, 8, 4), "../../outside")

    def test_autogrow_accepts_more_than_ten_assets(self):
        values = {f"assets.asset_{i}": i for i in range(12)}
        _, _, data = _io.get_finalized_class_inputs(m.FL_PosterLayerStack.INPUT_TYPES(), values)
        self.assertEqual(len(_io.build_nested_inputs(values, data)["assets"]), 12)

    def test_stack_visibility_and_review(self):
        p = plan(3)
        entries = m.apply_overrides(p, json.dumps({"plan_key": m.plan_key(p), "layers": {"layer_1": {"visible": False}}}))
        assets = {f"asset_{i}": dict(image=torch.ones(1, 8, 8, 4), file={}, thumbnail={}, coverage=1) for i in range(3)}
        out = m.FL_PosterLayerStack.execute(json.dumps(entries), m.plan_key(p), assets)
        self.assertEqual(len(out.result[0]["layers"]), 1)
        self.assertEqual(len(out.ui["poster_layers"]), 3)
        self.assertEqual(out.result[0]["background"].shape, (1, 8, 8, 4))

    def test_manual_mode_does_not_request_vision(self):
        self.assertEqual(m.FL_PosterLayerPlanner.check_lazy_status(mode="manual"), [])
        out = m.FL_PosterLayerPlanner.execute(torch.zeros(1, 8, 8, 3), "manual", 3, "balanced", json.dumps(plan(3)))
        self.assertEqual(len(out.result[0]), 3)


if __name__ == "__main__":
    unittest.main()
