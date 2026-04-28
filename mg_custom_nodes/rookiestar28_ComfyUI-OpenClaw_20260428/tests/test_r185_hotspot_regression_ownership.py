import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from services.csrf_protection import require_same_origin_if_no_token
from services.safe_io import SSRFError, safe_fetch, validate_outbound_url
from tests.quality_governance_test_utils import (
    sample_policy_payload,
    write_governance_baseline_fixture,
)

ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_SCRIPT = ROOT / "scripts" / "verify_quality_governance.py"


class TestR185HotspotRegressionOwnership(unittest.TestCase):
    def _run_governance_script(self, *args):
        return subprocess.run(
            [sys.executable, str(GOVERNANCE_SCRIPT), *args],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            check=False,
        )

    def test_ratchet55_critical_families_require_targeted_regression_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fixture = write_governance_baseline_fixture(
                tmp,
                fail_under=45.0,
                coverage_policy_payload=sample_policy_payload(
                    current_stage="ratchet-45",
                    stages=[
                        {"id": "baseline-35", "min_fail_under": 35.0},
                        {"id": "ratchet-45", "min_fail_under": 45.0},
                        {"id": "ratchet-55", "min_fail_under": 55.0},
                    ],
                    hotspot_families=[
                        {"id": "safe_io", "paths": ["services/safe_io.py"]},
                        {
                            "id": "security_boundary",
                            "paths": ["services/security_gate.py"],
                        },
                        {
                            "id": "connector_config",
                            "paths": ["connector/config.py"],
                        },
                        {"id": "config_bootstrap", "paths": ["config.py"]},
                    ],
                ),
            )

            result = self._run_governance_script(
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
        self.assertIn("ratchet55_readiness", result.stdout)
        self.assertIn("safe_io", result.stdout)
        self.assertIn("security_boundary", result.stdout)

    def test_safe_io_redirect_revalidates_second_hop_before_connect(self):
        class RedirectResponse:
            headers = {"Location": "http://169.254.169.254/latest/meta-data"}

            def getcode(self):
                return 302

            def close(self):
                pass

        opener = MagicMock()
        opener.open.return_value = RedirectResponse()

        with (
            patch("services.safe_io.socket.getaddrinfo") as getaddrinfo,
            patch("services.safe_io._build_pinned_opener", return_value=opener),
        ):
            getaddrinfo.side_effect = [
                [
                    (
                        None,
                        None,
                        None,
                        None,
                        ("93.184.216.34", 443),
                    )
                ],
                [
                    (
                        None,
                        None,
                        None,
                        None,
                        ("169.254.169.254", 80),
                    )
                ],
            ]

            with self.assertRaises(SSRFError) as ctx:
                safe_fetch(
                    "https://example.com/start",
                    allow_hosts={"example.com"},
                    max_redirects=1,
                )

        self.assertIn("Host not in allowlist", str(ctx.exception))
        opener.open.assert_called_once()

    def test_security_boundary_denies_cross_origin_convenience_request(self):
        request = SimpleNamespace(
            path="/openclaw/config",
            headers={"Origin": "https://attacker.example"},
        )

        with patch("services.csrf_protection.logger.warning"):
            response = require_same_origin_if_no_token(
                request,
                admin_token_configured=False,
            )

        self.assertIsNotNone(response)
        self.assertEqual(response.status, 403)
        self.assertIn("csrf_protection", response.text)

    def test_security_gate_fails_closed_on_fatal_error(self):
        from services.security_gate import enforce_startup_gate

        with (
            patch(
                "services.security_gate.SecurityGate.verify_mandatory_controls",
                return_value=(False, [], ["fatal boundary violation"]),
            ),
            patch("services.security_gate.logger"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                enforce_startup_gate()

        self.assertIn("fatal boundary violation", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
