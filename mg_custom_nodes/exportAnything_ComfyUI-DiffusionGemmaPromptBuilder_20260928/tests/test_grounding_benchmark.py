from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts import benchmark_grounding


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fact(fact_id: str, metric_tag: str = "supported_fact") -> dict:
    return {
        "id": fact_id,
        "claim": fact_id.replace("_", " "),
        "aliases": [],
        "frame_indices": [],
        "timecodes": [],
        "metric_tags": [metric_tag],
    }


def _case(
    case_id: str,
    *,
    category: str = "synthetic_image",
    private: bool = False,
    expected: tuple[str, ...] = ("red_cube",),
    prohibited: tuple[str, ...] = (),
    group: str = "",
    variant: str = "",
) -> dict:
    return {
        "id": case_id,
        "category": category,
        "fixture": f"fixtures/{case_id}.png",
        "fixture_sha256": hashlib.sha256(b"fixture").hexdigest() if not private else "0" * 64,
        "result": f"results/{case_id}.json",
        "private": private,
        "holdout": False,
        "target_profile": "ltx",
        "seeds": [7],
        "expected_facts": [_fact(item) for item in expected],
        "prohibited_facts": [_fact(item, "unsupported_fact") for item in prohibited],
        "ambiguity_labels": [],
        "counterfactual_group": group,
        "counterfactual_variant": variant,
        "notes": "",
    }


def _manifest(cases: list[dict], seeds: list[int] | None = None) -> dict:
    chosen_seeds = seeds or [7]
    for case in cases:
        case["seeds"] = list(chosen_seeds)
    return {
        "schema_version": benchmark_grounding.MANIFEST_SCHEMA_VERSION,
        "name": "unit benchmark",
        "description": "offline fixture",
        "seeds": list(chosen_seeds),
        "cases": cases,
    }


def _condition(condition_id: str = "unit", **overrides) -> dict:
    condition = {
        "condition_id": condition_id,
        "prompting": "guard_strict",
        "sampling_profile": "checkpoint_defaults",
        "media_variant": "case_default",
        "precision": "nvfp4",
        "telemetry_enabled": True,
        "frame_budget": 0,
        "visual_token_budget": 256,
        "model_revision": "unit-test-revision",
        "release_candidate": False,
    }
    condition.update(overrides)
    return condition


def _result(
    case_id: str,
    reported: list[str],
    unsupported: list[str] | None = None,
    *,
    seeds: list[int] | None = None,
    condition: dict | None = None,
    mode: str = "strict",
) -> dict:
    chosen_seeds = seeds or [7]
    chosen_condition = condition or _condition()
    return {
        "schema_version": benchmark_grounding.RESULT_SCHEMA_VERSION,
        "case_id": case_id,
        "runs": [
            {
                "seed": seed,
                "mode": mode,
                "condition": dict(chosen_condition),
                "guard_decision": "pass",
                "analysis_status": "grounded",
                "reported_fact_ids": reported,
                "unsupported_fact_ids": unsupported or [],
                "metadata": {},
            }
            for seed in chosen_seeds
        ],
    }


class GroundingBenchmarkTests(unittest.TestCase):
    def test_scores_facts_and_counterfactual_pairs_from_recorded_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [
                _case(
                    "real",
                    expected=("red_cube",),
                    prohibited=("dragon",),
                    group="image-swap",
                    variant="real",
                ),
                _case(
                    "unrelated",
                    expected=("blue_sphere",),
                    prohibited=("red_cube",),
                    group="image-swap",
                    variant="unrelated",
                ),
            ]
            manifest_path = root / "manifest.json"
            _write_json(manifest_path, _manifest(cases))
            for case in cases:
                fixture = root / case["fixture"]
                fixture.parent.mkdir(parents=True, exist_ok=True)
                fixture.write_bytes(b"fixture")
            _write_json(root / "results/real.json", _result("real", ["red_cube", "dragon"], ["dragon"]))
            _write_json(root / "results/unrelated.json", _result("unrelated", ["blue_sphere"]))

            report = benchmark_grounding.evaluate_manifest(manifest_path)

        self.assertEqual(report["metrics"]["supported_fact_recall"], 1.0)
        self.assertAlmostEqual(report["metrics"]["supported_fact_precision"], 2 / 3)
        self.assertAlmostEqual(report["metrics"]["unsupported_fact_rate"], 1 / 3)
        self.assertEqual(report["metrics"]["silent_hallucination_run_count"], 1)
        self.assertEqual(report["counterfactual"]["pair_count"], 1)
        self.assertEqual(report["counterfactual"]["discrimination_rate"], 1.0)

    def test_missing_private_fixture_is_a_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            _write_json(
                manifest_path,
                _manifest(
                    [
                        _case(
                            "private_failure",
                            category="private_known_failure",
                            private=True,
                        )
                    ]
                ),
            )

            report = benchmark_grounding.evaluate_manifest(manifest_path)

        self.assertEqual(report["metrics"]["scored_run_count"], 0)
        self.assertEqual(
            report["skipped_cases"],
            [{"case_id": "private_failure", "reason": "private_fixture_missing"}],
        )

    def test_missing_public_fixture_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            _write_json(manifest_path, _manifest([_case("public")]))

            with self.assertRaisesRegex(benchmark_grounding.ManifestError, "Public fixture"):
                benchmark_grounding.evaluate_manifest(manifest_path)

    def test_fixture_hash_mismatch_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _case("changed")
            manifest_path = root / "manifest.json"
            _write_json(manifest_path, _manifest([case]))
            fixture = root / case["fixture"]
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_bytes(b"different bytes")

            with self.assertRaisesRegex(benchmark_grounding.ManifestError, "hash mismatch"):
                benchmark_grounding.evaluate_manifest(manifest_path)

    def test_manifest_rejects_fact_ids_that_are_expected_and_prohibited(self):
        case = _case("overlap", expected=("same",), prohibited=("same",))

        with self.assertRaisesRegex(benchmark_grounding.ManifestError, "both expected and prohibited"):
            benchmark_grounding.validate_manifest(_manifest([case]))

    def test_conditions_can_coexist_per_seed_but_each_must_cover_every_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _case("matrix")
            manifest_path = root / "manifest.json"
            _write_json(manifest_path, _manifest([case], seeds=[7, 11]))
            fixture = root / case["fixture"]
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_bytes(b"fixture")
            first = _result(
                "matrix",
                ["red_cube"],
                seeds=[7, 11],
                condition=_condition("defaults"),
            )
            second = _result(
                "matrix",
                ["red_cube"],
                seeds=[7, 11],
                condition=_condition(
                    "diagnostic",
                    sampling_profile="full_48_diagnostic",
                    telemetry_enabled=False,
                ),
            )
            first["runs"].extend(second["runs"])
            _write_json(root / case["result"], first)

            report = benchmark_grounding.evaluate_manifest(manifest_path)
            self.assertEqual(report["metrics"]["scored_run_count"], 4)

            first["runs"].pop()
            _write_json(root / case["result"], first)
            with self.assertRaisesRegex(benchmark_grounding.ResultError, "incomplete"):
                benchmark_grounding.evaluate_manifest(manifest_path)

    def test_result_rejects_duplicate_seed_condition_and_invalid_typed_condition(self):
        case = benchmark_grounding.validate_manifest(_manifest([_case("duplicate")]))["cases"][0]
        result = _result("duplicate", ["red_cube"])
        result["runs"].append(dict(result["runs"][0]))
        with self.assertRaisesRegex(benchmark_grounding.ResultError, "duplicates seed"):
            benchmark_grounding._validate_result(result, case, Path("result.json"))

        malformed = _result("duplicate", ["red_cube"])
        malformed["runs"][0]["condition"]["precision"] = "mystery"
        with self.assertRaisesRegex(benchmark_grounding.ResultError, "precision"):
            benchmark_grounding._validate_result(malformed, case, Path("result.json"))

    def test_release_candidate_requires_strict_mode(self):
        case = benchmark_grounding.validate_manifest(_manifest([_case("release")]))["cases"][0]
        result = _result(
            "release",
            ["red_cube"],
            condition=_condition("release", release_candidate=True),
            mode="audit",
        )
        result["runs"][0]["guard_decision"] = "pass"
        with self.assertRaisesRegex(benchmark_grounding.ResultError, "strict mode"):
            benchmark_grounding._validate_result(result, case, Path("result.json"))

    def test_manifest_rejects_private_category_spoofing_and_path_traversal(self):
        spoofed = _case("spoofed")
        spoofed["private"] = True
        with self.assertRaisesRegex(benchmark_grounding.ManifestError, "private must be false"):
            benchmark_grounding.validate_manifest(_manifest([spoofed]))

        escaped = _case("escaped")
        escaped["fixture"] = "../outside.png"
        with self.assertRaisesRegex(benchmark_grounding.ManifestError, "confined relative path"):
            benchmark_grounding.validate_manifest(_manifest([escaped]))

    def test_small_or_missing_release_dataset_never_gets_a_passing_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _case("not_release_scale")
            manifest_path = root / "manifest.json"
            _write_json(manifest_path, _manifest([case]))
            fixture = root / case["fixture"]
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_bytes(b"fixture")
            _write_json(
                root / case["result"],
                _result(
                    case["id"],
                    ["red_cube"],
                    condition=_condition("release", release_candidate=True),
                ),
            )
            report = benchmark_grounding.evaluate_manifest(manifest_path)

        self.assertEqual(report["release_candidate"]["status"], "incomplete")
        self.assertFalse(report["release_candidate"]["eligible_for_verdict"])
        self.assertFalse(report["release_candidate"]["passed"])

    def test_cli_writes_no_file_without_output_option(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _case("ready")
            manifest_path = root / "manifest.json"
            _write_json(manifest_path, _manifest([case]))
            fixture = root / case["fixture"]
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_bytes(b"fixture")
            _write_json(root / case["result"], _result("ready", ["red_cube"]))
            before = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())

            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = benchmark_grounding.main([str(manifest_path)])
            after = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())

        self.assertEqual(exit_code, 0)
        self.assertEqual(before, after)
        self.assertEqual(json.loads(output.getvalue())["metrics"]["scored_run_count"], 1)


if __name__ == "__main__":
    unittest.main()
