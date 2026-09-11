from __future__ import annotations

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts import benchmark_ltx25_prompts


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "benchmarks"
    / "ltx25_prompt_contract_v1"
    / "manifest.json"
)


class Ltx25PromptBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.manifest = benchmark_ltx25_prompts.validate_manifest(cls.raw_manifest)

    def test_fixture_covers_all_modes_and_three_prompt_conditions(self):
        self.assertEqual(self.manifest["schema_version"], "dg-ltx25-prompt-benchmark/1")
        self.assertEqual(self.manifest["seeds"], [17, 101, 809])
        self.assertEqual({case["mode"] for case in self.manifest["cases"]}, {"t2v", "i2v", "flf"})
        for case in self.manifest["cases"]:
            self.assertEqual(
                tuple(case["candidates"]),
                benchmark_ltx25_prompts.CANDIDATE_IDS,
                case["id"],
            )
            self.assertEqual(len(set(case["candidates"].values())), 3, case["id"])
            control = case["generation_control"]
            self.assertEqual(control["frames"] % 8, 1, case["id"])
            self.assertEqual(
                control["duration_seconds"],
                (control["frames"] - 1) / control["fps"],
                case["id"],
            )
            self.assertFalse(control["prompt_enhancer"], case["id"])
            self.assertEqual(
                control["conditioning"]["kind"],
                benchmark_ltx25_prompts.CONDITIONING_KINDS[case["mode"]],
                case["id"],
            )

    def test_offline_report_is_deterministic_and_revised_beats_current(self):
        first = benchmark_ltx25_prompts.evaluate_manifest(MANIFEST_PATH)
        second = benchmark_ltx25_prompts.evaluate_manifest(MANIFEST_PATH)

        self.assertEqual(first, second)
        self.assertEqual(
            first["execution"],
            {"offline": True, "model_loaded": False, "network_used": False},
        )
        self.assertEqual(first["runtime_results"]["status"], "not_supplied")
        self.assertGreater(first["revised_vs_current_mean_delta"], 0)
        for case in first["cases"]:
            self.assertGreater(
                case["candidates"]["revised"]["score"],
                case["candidates"]["current"]["score"],
                case["case_id"],
            )
            self.assertGreater(case["revised_vs_current_delta"], 0, case["case_id"])
            self.assertTrue(case["downstream_control_is_fixed_across_candidates"], case["case_id"])
        self.assertEqual(first["primary_ab"]["A"]["prompt_producer"], "native_ltx25_prompt_enhancer")
        self.assertEqual(first["primary_ab"]["B"]["prompt_producer"], "diffusiongemma_ltx25_mode_contract")

    def test_legacy_prompt_fails_mode_specific_feasibility_and_complexity(self):
        report = benchmark_ltx25_prompts.evaluate_manifest(MANIFEST_PATH)
        cases = {case["mode"]: case for case in report["cases"]}

        t2v_failures = set(cases["t2v"]["candidates"]["current"]["failed_rule_ids"])
        self.assertTrue(
            {
                "budget.cuts",
                "budget.camera_operations",
                "budget.action_beats",
                "required.two_visible_subjects",
            }.issubset(t2v_failures)
        )

        i2v_failures = set(cases["i2v"]["candidates"]["current"]["failed_rule_ids"])
        self.assertIn("mode.i2v_first_frame_feasible", i2v_failures)
        self.assertIn("prohibited.scene_replacement", i2v_failures)
        self.assertIn("required.first_frame_daylight", i2v_failures)

        flf_failures = set(cases["flf"]["candidates"]["current"]["failed_rule_ids"])
        self.assertIn("mode.flf_endpoint_transition", flf_failures)
        self.assertIn("prohibited.vehicle_transformation", flf_failures)
        self.assertIn("required.last_frame_state", flf_failures)

    def test_generation_plan_pairs_native_enhancer_and_diffusiongemma_with_fixed_downstream_controls(self):
        manifest_bytes = MANIFEST_PATH.read_bytes()
        plan = benchmark_ltx25_prompts.build_generation_plan(
            self.manifest, hashlib.sha256(manifest_bytes).hexdigest()
        )

        expected_runs = (
            len(self.manifest["cases"])
            * len(benchmark_ltx25_prompts.CANDIDATE_IDS)
            * len(self.manifest["seeds"])
        )
        self.assertEqual(len(plan["runs"]), expected_runs)
        self.assertFalse(plan["execution"]["performed"])
        for case in self.manifest["cases"]:
            runs = [run for run in plan["runs"] if run["case_id"] == case["id"]]
            self.assertEqual(
                {run["candidate"] for run in runs},
                set(benchmark_ltx25_prompts.CANDIDATE_IDS),
                case["id"],
            )
            self.assertEqual({run["seed"] for run in runs}, set(self.manifest["seeds"]), case["id"])
            self.assertEqual(len({run["downstream_control_sha256"] for run in runs}), 1, case["id"])
            native_runs = [run for run in runs if run["candidate"] == "raw"]
            revised_runs = [run for run in runs if run["candidate"] == "revised"]
            self.assertTrue(all(run["generation_control"]["prompt_enhancer"] for run in native_runs))
            self.assertTrue(all(run["prompt"] == case["brief"] for run in native_runs))
            self.assertTrue(all(not run["generation_control"]["prompt_enhancer"] for run in revised_runs))
            self.assertTrue(all(run["prompt"] == case["candidates"]["revised"] for run in revised_runs))
            self.assertEqual(
                {run["prompt_producer"] for run in native_runs + revised_runs},
                {"native_ltx25_prompt_enhancer", "diffusiongemma_ltx25_mode_contract"},
            )
            self.assertEqual(len({run["prompt_sha256"] for run in runs}), 3, case["id"])

    def test_manifest_rejects_conditioning_mode_mismatch_and_candidate_overrides(self):
        wrong_conditioning = copy.deepcopy(self.raw_manifest)
        wrong_conditioning["cases"][0]["generation_control"]["conditioning"] = {
            "kind": "first_frame",
            "asset_ids": ["unexpected-image"],
            "strength": 0.7,
        }
        with self.assertRaisesRegex(benchmark_ltx25_prompts.ManifestError, "text_only"):
            benchmark_ltx25_prompts.validate_manifest(wrong_conditioning)

        candidate_override = copy.deepcopy(self.raw_manifest)
        candidate_override["cases"][0]["candidates"]["raw"] = {
            "prompt": "A prompt that attempts to change controls.",
            "video_cfg": 9.0,
        }
        with self.assertRaisesRegex(benchmark_ltx25_prompts.ManifestError, "non-empty string"):
            benchmark_ltx25_prompts.validate_manifest(candidate_override)

    def test_optional_runtime_ratings_are_merged_without_running_a_model(self):
        manifest_sha256 = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
        runtime = {
            "schema_version": benchmark_ltx25_prompts.RUNTIME_RESULTS_SCHEMA_VERSION,
            "manifest_sha256": manifest_sha256,
            "runs": [
                {
                    "case_id": "t2v_single_exchange_4s",
                    "candidate": "revised",
                    "seed": 17,
                    "output": "C:/ComfyUI/app/output/controlled-run.mp4",
                    "ratings": {
                        "prompt_adherence": 4.5,
                        "motion_coherence": 4.0,
                        "visual_continuity": 3.5,
                        "audio_alignment": 4.0,
                    },
                    "notes": "Unit-test annotation.",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            runtime_path = Path(directory) / "runtime.json"
            runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
            report = benchmark_ltx25_prompts.evaluate_manifest(
                MANIFEST_PATH, runtime_results_path=runtime_path
            )

        merged = report["runtime_results"]
        self.assertEqual(merged["status"], "partial")
        self.assertEqual(merged["run_count"], 1)
        self.assertEqual(merged["runs"][0]["mean_rating"], 4.0)
        self.assertEqual(
            merged["candidate_metrics"]["revised"]["mean_by_dimension"]["prompt_adherence"],
            4.5,
        )
        self.assertTrue(report["execution"]["offline"])
        self.assertFalse(report["execution"]["model_loaded"])

    def test_cli_defaults_to_checked_in_fixture_and_writes_nothing(self):
        output = io.StringIO()
        with redirect_stdout(output):
            return_code = benchmark_ltx25_prompts.main([])

        self.assertEqual(return_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["manifest"]["case_count"], 3)
        self.assertTrue(payload["execution"]["offline"])


if __name__ == "__main__":
    unittest.main()
