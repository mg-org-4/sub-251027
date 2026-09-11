from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import advertising_runtime_nodes as runtime


class AdvertisementRuntimeNodeTests(unittest.TestCase):
    def test_passthrough_policy_preserves_exact_object_without_comfy_runtime(self):
        value = {"timing_report": "locked"}
        result = runtime.AdvertisementMemoryBarrier().release(
            value, "Passthrough (no release)"
        )
        self.assertIs(result[0], value)
        self.assertIn("without releasing", result[1])

    def test_release_calls_only_advertisement_owned_comfy_memory_boundary(self):
        calls: list[object] = []
        management = types.ModuleType("comfy.model_management")
        management.unload_all_models = lambda: calls.append("unload")
        management.soft_empty_cache = lambda force: calls.append(("soft", force))
        comfy = types.ModuleType("comfy")
        comfy.model_management = management
        value = object()
        with mock.patch.dict(
            sys.modules,
            {"comfy": comfy, "comfy.model_management": management},
        ), mock.patch("advertising_runtime_nodes.gc.collect") as collect, mock.patch(
            "torch.cuda.is_available", return_value=False
        ):
            result = runtime.AdvertisementMemoryBarrier().release(
                value, "Unload models + clear CUDA cache"
            )
        self.assertIs(result[0], value)
        self.assertEqual(calls, ["unload", ("soft", True)])
        collect.assert_called_once_with()

    def test_unknown_policy_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            runtime.AdvertisementMemoryBarrier().release("value", "mystery")


if __name__ == "__main__":
    unittest.main()
