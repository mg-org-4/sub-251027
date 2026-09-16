from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_harness_module():
    module_name = f"grounding_gpu_acceptance_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts" / "grounding_gpu_acceptance.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load grounding GPU acceptance harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_telemetry_record(harness, *, seed=0, seconds=1.0, peak=1_000_000):
    return {
        "seed": seed,
        "telemetry_enabled": True,
        "generation_seconds": seconds,
        "wall_seconds": seconds,
        "peak_vram_allocated_bytes": peak,
        "peak_vram_reserved_bytes": peak,
        "_token_ids": [[10, 20, 30]],
        "_decoded_text": "same output",
        "capture_error": "",
        "transport": {"pixel_transport_confirmed": True},
        "effective_sampling": {
            "max_denoising_steps": 4,
            "adaptive_stopping": False,
        },
        "telemetry": {
            "schema_version": "dg-denoising-telemetry/1",
            "max_denoising_steps": 4,
            "forward_count": 4,
            "canvas_forward_counts": {"1": 4},
            "errors": [],
            "final_output": {"token_count": [3]},
        },
    }


def disabled_record(*, seed=0, seconds=1.0, peak=1_000_000):
    return {
        "seed": seed,
        "telemetry_enabled": False,
        "generation_seconds": seconds,
        "wall_seconds": seconds,
        "peak_vram_allocated_bytes": peak,
        "peak_vram_reserved_bytes": peak,
        "_token_ids": [[10, 20, 30]],
        "_decoded_text": "same output",
        "capture_error": "",
        "transport": {"pixel_transport_confirmed": True},
        "effective_sampling": {
            "max_denoising_steps": 4,
            "adaptive_stopping": False,
        },
        "telemetry": {},
    }


class GroundingGpuAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.harness = load_harness_module()

    def test_required_tiers_are_automatically_selected(self):
        selected, required = self.harness._normalize_tiers([1], [3, 2, 3])
        self.assertEqual(selected, [1, 2, 3])
        self.assertEqual(required, [2, 3])

    def test_default_cli_is_processor_only_and_report_only(self):
        args = self.harness._parse_args(["local-model"])
        self.assertEqual(args.tiers, [1])
        self.assertEqual(args.require, [])

    def test_report_only_failures_return_zero_but_required_failures_do_not(self):
        report = {
            "requested_tiers": [1],
            "tiers": {"1": {"passed": False}},
        }
        self.harness._finalize_report(report, [])
        self.assertEqual(self.harness._exit_code(report), 0)

        self.harness._finalize_report(report, [1])
        self.assertEqual(self.harness._exit_code(report), 1)

    def test_counterfactual_comparison_requires_same_shapes_and_different_pixels(self):
        base = {
            "input_keys": ["input_ids", "pixel_values"],
            "tensor_summary": {
                "input_ids": {"shape": [1, 8], "dtype": "torch.int64"},
                "pixel_values": {"shape": [1, 16, 32], "dtype": "torch.float32"},
            },
            "pixel_hashes": {"pixel_values": "aaa"},
            "transport_proof": {"pixel_transport_confirmed": True},
        }
        counterfactual = {
            **base,
            "pixel_hashes": {"pixel_values": "bbb"},
        }
        comparison = self.harness._compare_same_shape_cases(base, counterfactual)
        self.assertTrue(comparison["passed"])

        same_pixels = {**counterfactual, "pixel_hashes": {"pixel_values": "aaa"}}
        self.assertFalse(self.harness._compare_same_shape_cases(base, same_pixels)["passed"])

    def test_pair_evaluator_accepts_five_identical_pairs_within_limits(self):
        pairs = []
        for seed in self.harness.FIXED_PAIR_SEEDS:
            pairs.append(
                {
                    "seed": seed,
                    "execution_order": ["disabled", "enabled"],
                    "disabled": disabled_record(seed=seed, seconds=1.0, peak=1_000_000),
                    "enabled": valid_telemetry_record(
                        self.harness,
                        seed=seed,
                        seconds=1.05,
                        peak=1_000_000 + 64 * 1024 * 1024,
                    ),
                }
            )
        evaluation = self.harness._evaluate_pair_records(pairs)
        self.assertTrue(evaluation["passed"])
        self.assertEqual(evaluation["pair_count"], 5)
        self.assertAlmostEqual(evaluation["median_telemetry_overhead_percent"], 5.0)

    def test_pair_evaluator_rejects_token_changes_overhead_and_vram(self):
        pairs = []
        for index, seed in enumerate(self.harness.FIXED_PAIR_SEEDS):
            enabled = valid_telemetry_record(
                self.harness,
                seed=seed,
                seconds=1.2,
                peak=1_000_000 + 600 * 1024 * 1024,
            )
            if index == 0:
                enabled["_token_ids"] = [[99]]
            pairs.append(
                {
                    "seed": seed,
                    "execution_order": ["enabled", "disabled"],
                    "disabled": disabled_record(seed=seed),
                    "enabled": enabled,
                }
            )
        evaluation = self.harness._evaluate_pair_records(pairs)
        self.assertFalse(evaluation["passed"])
        self.assertFalse(evaluation["all_token_ids_and_text_identical"])
        self.assertFalse(evaluation["overhead_passed"])
        self.assertFalse(evaluation["vram_passed"])

    def test_runtime_patch_restores_loader_and_sampling(self):
        original_loader = lambda _config: "original"
        original_sampling = lambda _profile: "sampling"
        fake_nodes = SimpleNamespace(
            _load_transformers_model=original_loader,
            _guarded_sampling_kwargs=original_sampling,
        )
        processor = object()
        model = object()
        with self.harness._patched_nodes_runtime(fake_nodes, processor, model):
            self.assertEqual(fake_nodes._load_transformers_model(None), (processor, model))
            self.assertIs(fake_nodes._guarded_sampling_kwargs, self.harness._four_step_sampling_kwargs)
        self.assertIs(fake_nodes._load_transformers_model, original_loader)
        self.assertIs(fake_nodes._guarded_sampling_kwargs, original_sampling)

    def test_runtime_lifecycle_loads_once_and_releases_once(self):
        runtime_config = object()
        processor = object()
        model = SimpleNamespace(config=SimpleNamespace(canvas_length=256))
        fake_nodes = SimpleNamespace(
            RuntimeConfig=Mock(return_value=runtime_config),
            _model_path_info=Mock(
                return_value={"path_kind": "nvfp4_hf_repo", "quant_algo": "NVFP4"}
            ),
            _load_transformers_model=Mock(return_value=(processor, model)),
            _release_transformers_runtime=Mock(),
            _guarded_sampling_kwargs=Mock(),
        )
        args = argparse.Namespace(
            model_path=str(ROOT),
            tiers=[2, 3],
            require=[],
            max_memory_gb=20.0,
        )
        pass_tier2 = {"name": self.harness.TIER_NAMES[2], "status": "pass", "passed": True}
        pass_tier3 = {"name": self.harness.TIER_NAMES[3], "status": "pass", "passed": True}
        with patch.object(self.harness, "_environment_info", return_value={}), patch.object(
            self.harness, "_run_tier2", return_value=pass_tier2
        ), patch.object(self.harness, "_run_tier3", return_value=pass_tier3):
            report = self.harness.run_acceptance(args, dg_nodes=fake_nodes)

        fake_nodes._load_transformers_model.assert_called_once_with(runtime_config)
        fake_nodes._release_transformers_runtime.assert_called_once_with()
        self.assertEqual(report["runtime_lifecycle"]["model_load_attempts"], 1)
        self.assertEqual(report["runtime_lifecycle"]["release_attempts"], 1)
        self.assertTrue(report["runtime_lifecycle"]["release_succeeded"])

    def test_pair_seed_set_is_exactly_five_fixed_values(self):
        self.assertEqual(len(self.harness.FIXED_PAIR_SEEDS), 5)
        self.assertEqual(len(set(self.harness.FIXED_PAIR_SEEDS)), 5)

    def test_main_keeps_stdout_as_one_machine_readable_json_document(self):
        expected = {
            "schema": self.harness.SCHEMA_ID,
            "overall": {"required_passed": True},
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(self.harness, "run_acceptance", return_value=expected), redirect_stdout(
            stdout
        ), redirect_stderr(stderr):
            exit_code = self.harness.main(["local-model"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), expected)


if __name__ == "__main__":
    unittest.main()
