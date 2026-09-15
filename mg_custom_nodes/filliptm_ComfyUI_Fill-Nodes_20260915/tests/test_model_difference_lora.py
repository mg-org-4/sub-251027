import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import load_file


spec = importlib.util.spec_from_file_location("fl_lora", Path(__file__).parents[1] / "nodes/utility/FL_ModelDifferenceLoraSave.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def model(weight, bias=None):
    network = torch.nn.Module()
    network.diffusion_model = torch.nn.Module()
    network.diffusion_model.linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    network.diffusion_model.linear.weight = torch.nn.Parameter(weight.clone())
    if bias is not None:
        network.diffusion_model.linear.bias = torch.nn.Parameter(bias.clone())
    return SimpleNamespace(model=network, backup={}, hook_backup={}, patches={})


class ModelDifferenceLoraTests(unittest.TestCase):
    def test_export_roundtrip_sign_bias_and_no_input_mutation(self):
        base = model(torch.randn(4, 5), torch.randn(4))
        tuned = model(base.model.diffusion_model.linear.weight.detach() + torch.randn(4, 5),
                      base.model.diffusion_model.linear.bias.detach() + 0.25)
        before = tuned.model.diffusion_model.linear.weight.detach().clone()
        with TemporaryDirectory() as root, patch.object(m.folder_paths, "get_folder_paths", return_value=[root]):
            path, = m.FL_ModelDifferenceLoraSave().save(tuned, base, "Krea2/test", 4, "cpu")
            sd = load_file(path)
            key = "diffusion_model.linear"
            delta = sd[key + ".lora_up.weight"].float() @ sd[key + ".lora_down.weight"].float()
            torch.testing.assert_close(base.model.diffusion_model.linear.weight + delta, before, atol=.003, rtol=.003)
            torch.testing.assert_close(sd[key + ".bias.diff"].float(), torch.full((4,), .25))
            key_map = {key: key + ".weight", key + ".bias": key + ".bias"}
            patches = m.comfy.lora.load_lora(sd, key_map)
            self.assertEqual(set(patches), set(key_map.values()))
            with safe_open(path, framework="pt") as f:
                self.assertAlmostEqual(json.loads(f.metadata()["fl_extraction"])["retained_weight_energy"], 1, places=5)
            second, = m.FL_ModelDifferenceLoraSave().save(tuned, base, "Krea2/test", 4, "cpu")
            self.assertNotEqual(path, second)
        torch.testing.assert_close(tuned.model.diffusion_model.linear.weight, before)

    def test_lowrank_and_convolution_reconstruction(self):
        for shape in ((96, 80), (96, 5, 4, 4)):
            diff = (torch.randn(96, 3) @ torch.randn(3, 80)).reshape(shape)
            up, down = m.factorize_difference(diff, 3)
            torch.testing.assert_close(up.flatten(1) @ down.flatten(1), diff.flatten(1), atol=1e-4, rtol=1e-4)

    def test_patches_and_backup_are_applied_before_subtraction(self):
        target = model(torch.full((3, 4), 99.))
        key = "diffusion_model.linear.weight"
        target.backup[key] = SimpleNamespace(weight=torch.ones(3, 4))
        target.patches[key] = [(0.5, (torch.full((3, 4), 4.),), 1., None, None)]
        torch.testing.assert_close(m.materialize_weight(target, key, torch.device("cpu")), torch.full((3, 4), 3.))

    def test_invalid_models_and_paths_do_not_save(self):
        base = model(torch.zeros(4, 5))
        with TemporaryDirectory() as root, patch.object(m.folder_paths, "get_folder_paths", return_value=[root]):
            for tuned, prefix in ((base, "same"), (model(torch.zeros(3, 5)), "shape"),
                                  (model(torch.full((4, 5), float("nan"))), "nan"),
                                  (base, "../escape"), (base, "C:/escape")):
                with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                    m.FL_ModelDifferenceLoraSave().save(tuned, base, prefix, 4, "cpu")
            self.assertEqual(list(Path(root).rglob("*.safetensors")), [])


if __name__ == "__main__":
    unittest.main()
