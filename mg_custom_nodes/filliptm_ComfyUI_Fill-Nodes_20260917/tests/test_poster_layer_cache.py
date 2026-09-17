import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from comfy_execution.cache_provider import CacheContext, CacheValue
from comfy_execution.cache_provider import register_cache_provider, unregister_cache_provider
from comfy_execution.caching import RAMPressureCache, CacheKeySetInputSignature
from comfy_execution.graph import DynamicPrompt, ExecutionList
import nodes


spec = importlib.util.spec_from_file_location("poster_cache_test", Path(__file__).parents[1] / "nodes/vfx/poster_layer_cache.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PosterCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_expanded_asset_recovers_before_scheduling_ancestors(self):
        poster_spec = importlib.util.spec_from_file_location("poster_integration_test", Path(__file__).parents[1] / "nodes/vfx/FL_PosterLayers.py")
        poster = importlib.util.module_from_spec(poster_spec)
        poster_spec.loader.exec_module(poster)
        class Unchanged:
            async def get(self, node_id):
                return False
        dyn = DynamicPrompt({"parent": {"class_type": "EmptyImage", "inputs": {}}})
        dyn.add_ephemeral_node("extract", {"class_type": "EmptyImage", "inputs": {}}, "parent", "parent")
        dyn.add_ephemeral_node("asset", {"class_type": "FL_PosterLayerAsset", "inputs": {"image": ["extract", 0], "layer_id": "layer_0"}}, "parent", "parent")
        with tempfile.TemporaryDirectory() as root, patch.dict(nodes.NODE_CLASS_MAPPINGS, FL_PosterLayerAsset=poster.FL_PosterLayerAsset):
            cache = RAMPressureCache(CacheKeySetInputSignature, enable_providers=True)
            await cache.set_prompt(dyn, ["parent"], Unchanged())
            await cache.ensure_subcache_for("parent", ["extract", "asset"])
            provider = m.PosterLayerCache(root)
            context = cache._build_context("asset", cache.cache_key_set.get_data_key("asset"))
            expected = torch.rand(1, 8, 8, 4)
            await provider.on_store(context, CacheValue(outputs=[[dict(image=expected)]]))
            register_cache_provider(provider)
            try:
                execution = ExecutionList(dyn, cache)
                execution.add_node("asset")
                self.assertEqual(list(execution.pendingNodes), ["asset"])
                restored = await cache.get("asset")
                torch.testing.assert_close(restored.outputs[0][0]["image"], expected, atol=0, rtol=0)
                self.assertTrue(execution.is_cached("asset"))
            finally:
                unregister_cache_provider(provider)

    async def test_exact_disk_round_trip_and_key_invalidation(self):
        with tempfile.TemporaryDirectory() as root:
            cache = m.PosterLayerCache(root)
            context = CacheContext("1", "FL_PosterLayerAsset", "a" * 64)
            image = torch.rand(1, 8, 8, 4)
            asset = dict(image=image, file={"filename": "test.png"}, thumbnail={}, coverage=.6)
            self.assertIsNone(await cache.on_lookup(context))
            await cache.on_store(context, CacheValue(outputs=[[asset]]))
            restored = await m.PosterLayerCache(root).on_lookup(context)
            torch.testing.assert_close(restored.outputs[0][0]["image"], image, rtol=0, atol=0)
            self.assertEqual(restored.outputs[0][0]["file"], asset["file"])
            self.assertIsNone(await cache.on_lookup(CacheContext("1", "FL_PosterLayerAsset", "b" * 64)))

    async def test_scope_and_path_validation(self):
        with tempfile.TemporaryDirectory() as root:
            cache = m.PosterLayerCache(root)
            for context in (CacheContext("1", "KSampler", "a" * 64), CacheContext("1", "FL_PosterLayerAsset", "../outside")):
                self.assertFalse(cache.should_cache(context))
                self.assertIsNone(await cache.on_lookup(context))

    async def test_bounded_eviction_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as root:
            unrelated = Path(root) / "keep.txt"
            unrelated.touch()
            cache = m.PosterLayerCache(root, maximum_bytes=2000)
            image = torch.zeros(1, 8, 8, 4)
            for char in ("a", "b", "c"):
                await cache.on_store(CacheContext("1", "FL_PosterLayerAsset", char * 64), CacheValue(outputs=[[dict(image=image)]]))
            self.assertEqual(len(list(Path(root).glob("*.safetensors"))), 1)
            self.assertTrue(unrelated.exists())
            self.assertIsNotNone(await cache.on_lookup(CacheContext("1", "FL_PosterLayerAsset", "c" * 64)))


if __name__ == "__main__":
    unittest.main()
