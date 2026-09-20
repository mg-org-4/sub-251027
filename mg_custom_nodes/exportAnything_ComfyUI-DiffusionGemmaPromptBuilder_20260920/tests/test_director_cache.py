from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]


def load_nodes_module():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    package_name = f"diffusiongemma_director_cache_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        package_name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load DiffusionGemma Prompt Builder")
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    return sys.modules[f"{package_name}.nodes"]


class DirectorCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()

    def runtime(self, *, unload_policy: str = "keep_loaded", model_path: str = "unused"):
        return self.nodes.RuntimeConfig(
            model_path=model_path,
            backend="transformers_inprocess",
            dtype="auto",
            quantization="modelopt_nvfp4",
            local_files_only=True,
            unload_policy=unload_policy,
            max_memory_gb=20.0,
            status={"ready": True, "supports_pixels": True},
        )

    def context(self, images=None):
        return self.nodes.GemmaContext(
            user_prompt="A short verified scene.",
            images=images,
            source="image" if images is not None else "none",
            media_metadata={
                "source": "image" if images is not None else "none",
                "reference_image_count": 1 if images is not None else 0,
            },
        )

    def target(self):
        return self.nodes.TargetProfileConfig(target_profile="ltx")

    @staticmethod
    def successful_result():
        report = {
            "schema": "dg-grounding-report/1",
            "config": {"mode": "strict"},
            "analysis_status": "not_applicable",
            "decision": "not_applicable",
            "would_block": False,
            "grounding_guard_would_block": False,
            "blocked_reasons": [],
            "attempt_count": 0,
            "retry_reasons": [],
            "model_call_count": 1,
        }
        metadata = {
            "ready_for_generation": True,
            "json_parse_valid": True,
            "used_template_fallback": False,
            "salvage_warning": "",
            "grounding_guard": report,
        }
        packet = {
            "ltx_prompt": "A complete verified local video prompt.",
            "ideogram_prompt": "",
            "minimax_h3_prompt": "",
            "negative_prompt": "",
            "scene_segments": [],
            "metadata": metadata,
        }
        return packet, "raw", "reasoning", "", True, "", metadata

    def generate(self, node, runtime, *, guard=None, cache_mode="reuse"):
        return node.generate(
            runtime,
            self.context(),
            self.target(),
            temperature=0.45,
            creativity_mode="editorial",
            creative_strength=0.6,
            thinking_mode="off",
            max_new_tokens=2048,
            director_cache_mode=cache_mode,
            grounding_guard_config=guard or {"mode": "strict"},
            unique_id="node-does-not-affect-disk-key",
        )

    def test_second_call_hits_disk_without_running_director_again(self) -> None:
        fixed_key = "a" * 64
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {"DG_DIRECTOR_CACHE_DIR": temporary},
        ), patch.object(
            self.nodes,
            "_director_cache_key",
            return_value=fixed_key,
        ), patch.object(
            self.nodes,
            "_run_generation_packet",
            side_effect=lambda *_args, **_kwargs: copy.deepcopy(self.successful_result()),
        ) as director:
            node = self.nodes.DiffusionGemmaCoTGenerator()
            first = self.generate(node, self.runtime())
            second = self.generate(node, self.runtime())

        self.assertEqual(director.call_count, 1)
        self.assertEqual(json.loads(first[3])["director_cache"]["status"], "stored")
        second_metadata = json.loads(second[3])
        self.assertTrue(second_metadata["director_cache"]["hit"])
        self.assertTrue(second_metadata["director_runtime"]["cache_hit"])
        self.assertEqual(second_metadata["director_runtime"]["generation_seconds"], 0.0)
        self.assertEqual(second[1], "reasoning")
        self.assertEqual(second[2], "raw")

    def test_cache_hit_honors_unload_after_run(self) -> None:
        fixed_key = "b" * 64
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {"DG_DIRECTOR_CACHE_DIR": temporary},
        ), patch.object(
            self.nodes,
            "_director_cache_key",
            return_value=fixed_key,
        ), patch.object(
            self.nodes,
            "_run_generation_packet",
            side_effect=lambda *_args, **_kwargs: copy.deepcopy(self.successful_result()),
        ), patch.object(self.nodes, "_release_transformers_runtime") as release:
            node = self.nodes.DiffusionGemmaCoTGenerator()
            runtime = self.runtime(unload_policy="unload_after_run")
            self.generate(node, runtime)
            self.generate(node, runtime)

        # The mocked first Director call does not own the real wrapper finally;
        # the second, cached call explicitly enforces the lifecycle policy.
        release.assert_called_once_with()

    def test_corrupt_entry_causes_safe_recomputation(self) -> None:
        fixed_key = "c" * 64
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {"DG_DIRECTOR_CACHE_DIR": temporary},
        ), patch.object(
            self.nodes,
            "_director_cache_key",
            return_value=fixed_key,
        ), patch.object(
            self.nodes,
            "_run_generation_packet",
            side_effect=lambda *_args, **_kwargs: copy.deepcopy(self.successful_result()),
        ) as director:
            node = self.nodes.DiffusionGemmaCoTGenerator()
            self.generate(node, self.runtime())
            self.nodes._director_disk_cache_path(fixed_key).write_text("{", encoding="utf-8")
            result = self.generate(node, self.runtime())

        self.assertEqual(director.call_count, 2)
        self.assertFalse(json.loads(result[3])["director_cache"]["hit"])

    def test_detailed_trace_bypasses_cache_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {"DG_DIRECTOR_CACHE_DIR": temporary},
        ), patch.object(
            self.nodes,
            "_run_generation_packet",
            side_effect=lambda *_args, **_kwargs: copy.deepcopy(self.successful_result()),
        ) as director:
            node = self.nodes.DiffusionGemmaCoTGenerator()
            guard = {"mode": "strict", "save_detailed_trace": True}
            first = self.generate(node, self.runtime(), guard=guard)
            second = self.generate(node, self.runtime(), guard=guard)

        self.assertEqual(director.call_count, 2)
        self.assertEqual(
            json.loads(first[3])["director_cache"]["status"],
            "bypassed_for_detailed_trace",
        )
        self.assertEqual(
            json.loads(second[3])["director_cache"]["status"],
            "bypassed_for_detailed_trace",
        )
        self.assertEqual(list(Path(temporary).glob("**/*.json")), [])

    def test_reuse_is_stable_while_explicit_rerun_modes_bypass_native_cache(self) -> None:
        first = self.nodes.DiffusionGemmaCoTGenerator.IS_CHANGED(
            director_cache_mode="reuse"
        )
        second = self.nodes.DiffusionGemmaCoTGenerator.IS_CHANGED(
            director_cache_mode="reuse"
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            f"director-reuse:{self.nodes.DIRECTOR_EXECUTION_CONTRACT_REVISION}",
        )
        for mode in ("refresh", "off"):
            with self.subTest(director_cache_mode=mode):
                value = self.nodes.DiffusionGemmaCoTGenerator.IS_CHANGED(
                    director_cache_mode=mode
                )
                self.assertTrue(math.isnan(value))

    def test_detailed_trace_guard_bypasses_native_cache(self) -> None:
        stable = self.nodes.DiffusionGemmaGroundingGuardSettings.IS_CHANGED(
            save_detailed_trace=False
        )
        self.assertEqual(stable, "grounding-guard-config-v1")
        changed = self.nodes.DiffusionGemmaGroundingGuardSettings.IS_CHANGED(
            save_detailed_trace=True
        )
        self.assertTrue(math.isnan(changed))
        linked = self.nodes.DiffusionGemmaGroundingGuardSettings.IS_CHANGED(
            save_detailed_trace=None
        )
        self.assertTrue(math.isnan(linked))

    def test_comfy_signature_cache_isolates_an_ltx_only_edit_from_h3(self) -> None:
        # Other test modules install lightweight Comfy stubs during discovery.
        # Temporarily replace that module family with the real local ComfyUI
        # implementation, then restore the discovery-time modules afterward.
        module_prefixes = (
            "comfy",
            "comfy_api",
            "comfy_execution",
            "execution",
            "nodes",
        )
        saved_modules = {
            name: module
            for name, module in tuple(sys.modules.items())
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in module_prefixes)
        }
        for name in saved_modules:
            sys.modules.pop(name, None)

        def restore_modules() -> None:
            for name in tuple(sys.modules):
                if any(
                    name == prefix or name.startswith(f"{prefix}.")
                    for prefix in module_prefixes
                ):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)

        self.addCleanup(restore_modules)
        import nodes as comfy_nodes
        from comfy_execution.caching import CacheKeySetInputSignature, LRUCache
        from comfy_execution.graph import DynamicPrompt
        from execution import CacheEntry, IsChangedCache

        class StableSource:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"value": ("STRING",)}}

            RETURN_TYPES = ("STRING",)
            FUNCTION = "run"

        class StablePass:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"source": ("STRING",)}}

            RETURN_TYPES = ("STRING",)
            FUNCTION = "run"

        class StableMerge:
            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {
                        "left": ("STRING",),
                        "right": ("STRING",),
                    }
                }

            RETURN_TYPES = ("STRING",)
            FUNCTION = "run"

        class StableOutput(StablePass):
            OUTPUT_NODE = True

        class_types = {
            "DGCacheTestSource": StableSource,
            "DGCacheTestPass": StablePass,
            "DGCacheTestMerge": StableMerge,
            "DGCacheTestOutput": StableOutput,
            "DiffusionGemmaCoTGenerator": self.nodes.DiffusionGemmaCoTGenerator,
            "DiffusionGemmaGroundingGuardSettings": (
                self.nodes.DiffusionGemmaGroundingGuardSettings
            ),
        }
        base_prompt = {
            "shared": {
                "class_type": "DGCacheTestSource",
                "inputs": {"value": "shared creative brief"},
            },
            "510": {
                "class_type": "DGCacheTestSource",
                "inputs": {"value": "LTX profile A"},
            },
            "186": {
                "class_type": "DGCacheTestMerge",
                "inputs": {
                    "left": ["shared", 0],
                    "right": ["510", 0],
                },
            },
            "ltx_output": {
                "class_type": "DGCacheTestOutput",
                "inputs": {"source": ["186", 0]},
            },
            "182": {
                "class_type": "DGCacheTestSource",
                "inputs": {"value": "model config"},
            },
            "513": {
                "class_type": "DGCacheTestPass",
                "inputs": {"source": ["shared", 0]},
            },
            "515": {
                "class_type": "DiffusionGemmaGroundingGuardSettings",
                "inputs": {"save_detailed_trace": False},
            },
            "518": {
                "class_type": "DGCacheTestSource",
                "inputs": {"value": "H3 target profile"},
            },
            "516": {
                "class_type": "DiffusionGemmaCoTGenerator",
                "inputs": {
                    "model_config": ["182", 0],
                    "gemma_context": ["513", 0],
                    "target_profile_config": ["518", 0],
                    "grounding_guard_config": ["515", 0],
                    "temperature": 0.45,
                    "creativity_mode": "wild",
                    "creative_strength": 0.2,
                    "thinking_mode": "off",
                    "max_new_tokens": 2048,
                    "director_cache_mode": "reuse",
                },
            },
            "517": {
                "class_type": "DGCacheTestPass",
                "inputs": {"source": ["516", 0]},
            },
            "h3_output": {
                "class_type": "DGCacheTestOutput",
                "inputs": {"source": ["517", 0]},
            },
        }

        async def cache_hits_after(changed_prompt):
            cache = LRUCache(CacheKeySetInputSignature, max_size=1)
            first = DynamicPrompt(copy.deepcopy(base_prompt))
            await cache.set_prompt(
                first,
                base_prompt.keys(),
                IsChangedCache("first", first, cache),
            )
            cache.clean_unused()
            for node_id in base_prompt:
                cache.set_local(node_id, CacheEntry(ui=None, outputs=[node_id]))

            second = DynamicPrompt(changed_prompt)
            await cache.set_prompt(
                second,
                changed_prompt.keys(),
                IsChangedCache("second", second, cache),
            )
            cache.clean_unused()
            return {
                node_id: cache.get_local(node_id) is not None
                for node_id in changed_prompt
            }

        with patch.dict(comfy_nodes.NODE_CLASS_MAPPINGS, class_types):
            ltx_edit = copy.deepcopy(base_prompt)
            ltx_edit["510"]["inputs"]["value"] = "LTX profile B"
            ltx_hits = asyncio.run(cache_hits_after(ltx_edit))
            self.assertFalse(ltx_hits["510"])
            self.assertFalse(ltx_hits["186"])
            self.assertFalse(ltx_hits["ltx_output"])
            for node_id in ("shared", "513", "515", "516", "517", "h3_output"):
                self.assertTrue(ltx_hits[node_id], node_id)

            shared_edit = copy.deepcopy(base_prompt)
            shared_edit["shared"]["inputs"]["value"] = "changed shared brief"
            shared_hits = asyncio.run(cache_hits_after(shared_edit))
            for node_id in ("shared", "513", "516", "517", "h3_output"):
                self.assertFalse(shared_hits[node_id], node_id)

            trace_edit = copy.deepcopy(base_prompt)
            trace_edit["515"]["inputs"]["save_detailed_trace"] = True
            trace_hits = asyncio.run(cache_hits_after(trace_edit))
            for node_id in ("515", "516", "517", "h3_output"):
                self.assertFalse(trace_hits[node_id], node_id)

            refresh_edit = copy.deepcopy(base_prompt)
            refresh_edit["516"]["inputs"]["director_cache_mode"] = "refresh"
            refresh_hits = asyncio.run(cache_hits_after(refresh_edit))
            for node_id in ("516", "517", "h3_output"):
                self.assertFalse(refresh_hits[node_id], node_id)

    def test_cache_key_tracks_pixels_and_semantics_but_not_unload_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.gguf"
            checkpoint.write_bytes(b"small-test-checkpoint")
            images = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
            context = self.context(images)
            target = self.target()
            common = dict(
                temperature=0.45,
                creativity_mode="editorial",
                creative_strength=0.6,
                thinking_mode="off",
                max_new_tokens=2048,
            )
            key = self.nodes._director_cache_key(
                self.runtime(model_path=str(checkpoint)),
                context,
                target,
                {"mode": "strict", "seed": 7},
                **common,
            )
            unload_key = self.nodes._director_cache_key(
                self.runtime(
                    unload_policy="unload_after_run",
                    model_path=str(checkpoint),
                ),
                context,
                target,
                {"mode": "strict", "seed": 7},
                **common,
            )
            changed_images = images.clone()
            changed_images[0, 0, 0, 0] = 1.0
            pixel_key = self.nodes._director_cache_key(
                self.runtime(model_path=str(checkpoint)),
                self.context(changed_images),
                target,
                {"mode": "strict", "seed": 7},
                **common,
            )
            duration_target = self.nodes.TargetProfileConfig(
                target_profile="ltx",
                target_duration_seconds=15.0,
            )
            target_key = self.nodes._director_cache_key(
                self.runtime(model_path=str(checkpoint)),
                context,
                duration_target,
                {"mode": "strict", "seed": 7},
                **common,
            )

        self.assertEqual(key, unload_key)
        self.assertNotEqual(key, pixel_key)
        self.assertNotEqual(key, target_key)

    def test_rank_three_tensor_hashes_as_one_image(self) -> None:
        rank_three = torch.zeros((2, 3, 3), dtype=torch.float32)
        rank_four = torch.zeros((2, 2, 3, 3), dtype=torch.float32)
        self.assertEqual(len(self.nodes._ordered_tensor_content_hashes(rank_three)), 1)
        self.assertEqual(len(self.nodes._ordered_tensor_content_hashes(rank_four)), 2)


if __name__ == "__main__":
    unittest.main()
