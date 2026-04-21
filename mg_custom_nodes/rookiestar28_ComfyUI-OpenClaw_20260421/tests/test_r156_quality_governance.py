import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.quality_governance_test_utils import (
    sample_policy_payload,
    write_governance_baseline_fixture,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_quality_governance.py"


class TestR156QualityGovernance(unittest.TestCase):
    def _run_script(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            check=False,
        )

    def test_repo_governance_baseline_passes(self):
        result = self._run_script()
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("GOVERNANCE-PASS", result.stdout)

    def test_missing_coverage_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fixture = write_governance_baseline_fixture(tmp)

            result = self._run_script(
                "--pyproject",
                str(fixture["pyproject"]),
                "--adversarial-gate",
                str(fixture["adversarial_gate"]),
                "--test-sop",
                str(fixture["test_sop"]),
                "--mutation-survivor-allowlist",
                str(fixture["survivor_allowlist"]),
                "--release-policy-doc",
                str(fixture["release_policy_doc"]),
                "--coverage-policy",
                str(tmp / "missing_policy.json"),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing coverage governance policy", result.stdout)

    def test_non_monotonic_policy_thresholds_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fixture = write_governance_baseline_fixture(
                tmp,
                coverage_policy_payload=sample_policy_payload(
                    stages=[
                        {"id": "baseline-35", "min_fail_under": 35.0},
                        {"id": "ratchet-30", "min_fail_under": 30.0},
                    ],
                    required_hotspot_families=["safe_io"],
                    hotspot_families=[
                        {"id": "safe_io", "paths": ["services/safe_io.py"]}
                    ],
                ),
            )

            result = self._run_script(
                "--pyproject",
                str(fixture["pyproject"]),
                "--adversarial-gate",
                str(fixture["adversarial_gate"]),
                "--test-sop",
                str(fixture["test_sop"]),
                "--mutation-survivor-allowlist",
                str(fixture["survivor_allowlist"]),
                "--release-policy-doc",
                str(fixture["release_policy_doc"]),
                "--coverage-policy",
                str(fixture["coverage_policy"]),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("coverage stages must increase strictly", result.stdout)

    def test_missing_required_hotspot_family_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fixture = write_governance_baseline_fixture(
                tmp,
                coverage_policy_payload=sample_policy_payload(
                    required_hotspot_families=["safe_io", "connector_config"],
                    hotspot_families=[
                        {"id": "safe_io", "paths": ["services/safe_io.py"]}
                    ],
                ),
            )

            result = self._run_script(
                "--pyproject",
                str(fixture["pyproject"]),
                "--adversarial-gate",
                str(fixture["adversarial_gate"]),
                "--test-sop",
                str(fixture["test_sop"]),
                "--mutation-survivor-allowlist",
                str(fixture["survivor_allowlist"]),
                "--release-policy-doc",
                str(fixture["release_policy_doc"]),
                "--coverage-policy",
                str(fixture["coverage_policy"]),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing required hotspot families", result.stdout)

    def test_missing_fail_under_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fixture = write_governance_baseline_fixture(tmp, fail_under=None)

            result = self._run_script(
                "--pyproject",
                str(fixture["pyproject"]),
                "--adversarial-gate",
                str(fixture["adversarial_gate"]),
                "--test-sop",
                str(fixture["test_sop"]),
                "--mutation-survivor-allowlist",
                str(fixture["survivor_allowlist"]),
                "--release-policy-doc",
                str(fixture["release_policy_doc"]),
                "--coverage-policy",
                str(fixture["coverage_policy"]),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing coverage fail_under", result.stdout)

    def test_stale_exception_review_date_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fixture = write_governance_baseline_fixture(
                tmp,
                coverage_policy_payload=sample_policy_payload(
                    exceptions=[
                        {
                            "id": "stale-hotspot-gap",
                            "family": "connector_config",
                            "paths": ["connector/config.py"],
                            "reason": "temporary uplift gap",
                            "review_by": "2000-01-01",
                        }
                    ],
                ),
            )
            review_evidence = tmp / "coverage_promotion_reviews.json"
            review_evidence.write_text(
                json.dumps({"schema_version": 1, "reviews": []}, indent=2) + "\n",
                encoding="utf-8",
            )

            result = self._run_script(
                "--pyproject",
                str(fixture["pyproject"]),
                "--adversarial-gate",
                str(fixture["adversarial_gate"]),
                "--test-sop",
                str(fixture["test_sop"]),
                "--mutation-survivor-allowlist",
                str(fixture["survivor_allowlist"]),
                "--release-policy-doc",
                str(fixture["release_policy_doc"]),
                "--coverage-policy",
                str(fixture["coverage_policy"]),
                "--coverage-review-evidence",
                str(review_evidence),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stale exception review date", result.stdout)

    def test_promoted_stage_requires_two_previous_stage_reviews(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fixture = write_governance_baseline_fixture(
                tmp,
                fail_under=45.0,
                coverage_policy_payload=sample_policy_payload(
                    current_stage="ratchet-45",
                    stages=[
                        {
                            "id": "baseline-35",
                            "min_fail_under": 35.0,
                            "promotion_requires": ["reviewed hotspots"],
                            "rollback_triggers": ["coverage regression"],
                        },
                        {
                            "id": "ratchet-45",
                            "min_fail_under": 45.0,
                            "promotion_requires": ["two consecutive reviews"],
                            "rollback_triggers": ["coverage regression"],
                        },
                    ],
                ),
            )
            review_evidence = tmp / "coverage_promotion_reviews.json"
            review_evidence.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "reviews": [
                            {
                                "cycle_id": "baseline-cycle-1",
                                "stage_id": "baseline-35",
                                "reviewed_at": "2026-04-20",
                                "overall_percent_covered": 68.14,
                                "reviewed_hotspot_families": [
                                    "connector_config",
                                    "config_bootstrap",
                                ],
                                "hotspot_percent_covered": {
                                    "connector_config": 58.0,
                                    "config_bootstrap": 65.0,
                                },
                                "artifact_reference": ".tmp/coverage/cycle1.json",
                            }
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            result = self._run_script(
                "--pyproject",
                str(fixture["pyproject"]),
                "--adversarial-gate",
                str(fixture["adversarial_gate"]),
                "--test-sop",
                str(fixture["test_sop"]),
                "--mutation-survivor-allowlist",
                str(fixture["survivor_allowlist"]),
                "--release-policy-doc",
                str(fixture["release_policy_doc"]),
                "--coverage-policy",
                str(fixture["coverage_policy"]),
                "--coverage-review-evidence",
                str(review_evidence),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires at least 2 promotion review cycles", result.stdout)

    def test_promoted_stage_with_two_previous_stage_reviews_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fixture = write_governance_baseline_fixture(
                tmp,
                fail_under=45.0,
                coverage_policy_payload=sample_policy_payload(
                    current_stage="ratchet-45",
                    stages=[
                        {
                            "id": "baseline-35",
                            "min_fail_under": 35.0,
                            "promotion_requires": ["reviewed hotspots"],
                            "rollback_triggers": ["coverage regression"],
                        },
                        {
                            "id": "ratchet-45",
                            "min_fail_under": 45.0,
                            "promotion_requires": ["two consecutive reviews"],
                            "rollback_triggers": ["coverage regression"],
                        },
                    ],
                ),
            )
            review_evidence = tmp / "coverage_promotion_reviews.json"
            review_evidence.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "reviews": [
                            {
                                "cycle_id": "baseline-cycle-1",
                                "stage_id": "baseline-35",
                                "reviewed_at": "2026-04-19",
                                "overall_percent_covered": 67.25,
                                "reviewed_hotspot_families": [
                                    "connector_config",
                                    "config_bootstrap",
                                ],
                                "hotspot_percent_covered": {
                                    "connector_config": 56.5,
                                    "config_bootstrap": 64.0,
                                },
                                "artifact_reference": ".tmp/coverage/cycle1.json",
                            },
                            {
                                "cycle_id": "baseline-cycle-2",
                                "stage_id": "baseline-35",
                                "reviewed_at": "2026-04-20",
                                "overall_percent_covered": 68.14,
                                "reviewed_hotspot_families": [
                                    "connector_config",
                                    "config_bootstrap",
                                ],
                                "hotspot_percent_covered": {
                                    "connector_config": 58.0,
                                    "config_bootstrap": 65.0,
                                },
                                "artifact_reference": ".tmp/coverage/cycle2.json",
                            },
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            result = self._run_script(
                "--pyproject",
                str(fixture["pyproject"]),
                "--adversarial-gate",
                str(fixture["adversarial_gate"]),
                "--test-sop",
                str(fixture["test_sop"]),
                "--mutation-survivor-allowlist",
                str(fixture["survivor_allowlist"]),
                "--release-policy-doc",
                str(fixture["release_policy_doc"]),
                "--coverage-policy",
                str(fixture["coverage_policy"]),
                "--coverage-review-evidence",
                str(review_evidence),
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertIn("GOVERNANCE-PASS", result.stdout)
