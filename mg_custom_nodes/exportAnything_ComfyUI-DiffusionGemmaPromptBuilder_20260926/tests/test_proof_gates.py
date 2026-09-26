from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

import proof_gates


def _probe_result(**overrides):
    result = {
        "processor_loads": False,
        "short_video_processor_smoke": {"passed": False},
        "path_kind": "missing",
        "comfy_nvfp4_bridge_supported": False,
        "nvfp4_bridge_smoke": {"passed": False},
        "telemetry_probe": {"passed": False},
        "torch_cuda_available": False,
    }
    result.update(overrides)
    return result


class ProofGateRequirementTests(unittest.TestCase):
    def test_all_expands_once_in_stable_order(self):
        self.assertEqual(
            proof_gates._normalize_requirements([["gpu"], ["all"], ["processor"]]),
            ["gpu", "processor", "video", "nvfp4", "telemetry"],
        )

    def test_requirement_evaluation_keeps_processor_and_video_separate(self):
        evaluation = proof_gates._evaluate_required_proofs(
            _probe_result(processor_loads=True),
            ["processor", "video"],
        )

        self.assertTrue(evaluation["available"]["processor"])
        self.assertFalse(evaluation["available"]["video"])
        self.assertEqual(evaluation["failed"], ["video"])
        self.assertFalse(evaluation["passed"])

    def test_default_mode_remains_report_only(self):
        output = io.StringIO()
        with patch.object(proof_gates, "probe", return_value=_probe_result()), redirect_stdout(output):
            exit_code = proof_gates.main(["--model-path", "missing"])

        self.assertEqual(exit_code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["required_proofs"]["requested"], [])
        self.assertTrue(report["required_proofs"]["passed"])

    def test_requested_failure_returns_nonzero(self):
        output = io.StringIO()
        with patch.object(
            proof_gates,
            "probe",
            return_value=_probe_result(torch_cuda_available=True),
        ), redirect_stdout(output):
            exit_code = proof_gates.main(
                ["--model-path", "missing", "--require", "gpu", "telemetry"]
            )

        self.assertEqual(exit_code, 1)
        report = json.loads(output.getvalue())
        self.assertEqual(report["required_proofs"]["failed"], ["telemetry"])

    def test_requested_success_returns_zero(self):
        output = io.StringIO()
        with patch.object(
            proof_gates,
            "probe",
            return_value=_probe_result(
                processor_loads=True,
                short_video_processor_smoke={"passed": True},
            ),
        ), redirect_stdout(output):
            exit_code = proof_gates.main(
                ["--model-path", "model", "--require", "processor", "video"]
            )

        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
