from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grounding_guard as guard


def _required_report(ledger: dict, trajectory: dict | None = None) -> dict:
    return {
        "schema": guard.GROUNDING_REPORT_SCHEMA_ID,
        "config": {"mode": "strict", "retry_on_uncertain": True},
        "analysis_status": "grounded",
        "decision": "pass",
        "would_block": False,
        "grounding_guard_would_block": False,
        "blocked_reasons": [],
        "asset_registry": {
            "schema": guard.ASSET_REGISTRY_SCHEMA_ID,
            "expected_image_count": 1,
            "assigned_image_count": 1,
            "assets": [
                {
                    "asset_id": "image:1",
                    "kind": "image",
                    "role": "primary image",
                    "tensor_batch_positions": [0],
                }
            ],
        },
        "transport": {
            "pixel_transport_confirmed": True,
            "expected_asset_count": 1,
            "processor_observed_asset_count": 1,
        },
        "validated_ledger": ledger,
        "attempt_count": 1,
        "retry_reasons": ["first evidence attempt was uncertain"],
        "trajectory_summary": trajectory or {"calls": [], "snapshots": []},
        "effective_sampling": {
            "max_denoising_steps": 48,
            "temperature_start": 0.8,
            "temperature_end": 0.4,
        },
        "timings": {"evidence_seconds": 1.25, "total_seconds": 2.5},
        "forward_call_count": 48,
        "warnings": ["diagnostic warning"],
        "verification_level": "transport+structured_self_report",
    }


def _large_schema_valid_ledger() -> dict:
    evidence = [{"asset_id": "image:1"} for _ in range(64)]
    typed_claims = [
        {
            "asset_id": "image:1",
            "claim_type": f"manual.label{index}",
            # Typed string values intentionally have no schema maxLength.
            "value": f"typed-{index}-" + ("v" * 4096),
        }
        for index in range(4)
    ]
    facts = [
        {
            "fact_id": f"fact-{index}",
            "claim": f"fact {index} " + ("c" * 2038),
            "confidence": "high",
            "categories": ["object", "color"],
            "evidence": copy.deepcopy(evidence),
            "typed_claims": copy.deepcopy(typed_claims),
        }
        for index in range(256)
    ]
    long_list = [f"item {index} " + ("x" * 2038) for index in range(256)]
    return {
        "schema": guard.GROUNDING_LEDGER_SCHEMA_ID,
        "analysis_status": "grounded",
        "observed_facts": facts,
        "inferred_facts": copy.deepcopy(long_list),
        "creative_additions": copy.deepcopy(long_list),
        "uncertainties": copy.deepcopy(long_list),
        "grounding_failure_reasons": copy.deepcopy(long_list),
    }


class GroundingReportCompactionTests(unittest.TestCase):
    def test_maximal_schema_valid_ledger_keeps_all_public_report_sections(self) -> None:
        ledger = _large_schema_valid_ledger()
        Draft202012Validator(guard.load_grounding_evidence_schema()).validate(ledger)
        report = _required_report(
            ledger,
            {
                "snapshots": [
                    {"step": index, "draft": "telemetry " + ("t" * 4096)}
                    for index in range(12)
                ],
                "calls": [{"call": index, "detail": "d" * 4096} for index in range(8)],
            },
        )

        compact = guard.compact_grounding_report(report)

        self.assertLessEqual(
            len(guard.grounding_report_json(compact).encode("utf-8")),
            guard.MAX_COMPACT_REPORT_BYTES,
        )
        self.assertTrue(compact["report_truncated"])
        self.assertTrue(
            {
                "asset_registry",
                "transport",
                "validated_ledger",
                "analysis_status",
                "decision",
                "retry_reasons",
                "trajectory_summary",
                "effective_sampling",
                "verification_level",
                "timings",
                "forward_call_count",
                "warnings",
                "would_block",
                "grounding_guard_would_block",
            }.issubset(compact)
        )
        compact_ledger = compact["validated_ledger"]
        self.assertTrue(
            {
                "schema",
                "analysis_status",
                "observed_facts",
                "inferred_facts",
                "creative_additions",
                "uncertainties",
                "grounding_failure_reasons",
            }.issubset(compact_ledger)
        )
        self.assertGreater(len(compact_ledger["observed_facts"]), 0)
        self.assertEqual(compact_ledger["observed_facts"][0]["fact_id"], "fact-0")
        self.assertEqual(compact_ledger["observed_facts"][-1]["fact_id"], "fact-255")
        self.assertIn(
            "...[truncated]",
            compact_ledger["observed_facts"][0]["typed_claims"][0]["value"],
        )
        self.assertEqual(
            compact["compaction"]["validated_ledger"]["observed_facts_original"],
            256,
        )
        self.assertEqual(len(report["validated_ledger"]["observed_facts"]), 256)
        self.assertEqual(
            len(
                report["validated_ledger"]["observed_facts"][0]["typed_claims"][0][
                    "value"
                ]
            ),
            len("typed-0-") + 4096,
        )

    def test_compaction_is_deterministic(self) -> None:
        ledger = _large_schema_valid_ledger()
        report = _required_report(ledger)
        first = guard.compact_grounding_report(report)
        second = guard.compact_grounding_report(report)
        self.assertEqual(first, second)

    def test_intermediate_telemetry_is_removed_before_ledger_content(self) -> None:
        ledger = {
            "schema": guard.GROUNDING_LEDGER_SCHEMA_ID,
            "analysis_status": "grounded",
            "observed_facts": [
                {
                    "fact_id": "fact-1",
                    "claim": "a red square",
                    "confidence": "high",
                    "categories": ["object", "color"],
                    "evidence": [{"asset_id": "image:1"}],
                }
            ],
            "inferred_facts": [],
            "creative_additions": [],
            "uncertainties": [],
            "grounding_failure_reasons": [],
        }
        report = _required_report(
            ledger,
            {
                "snapshots": [
                    {"step": index, "draft": "x" * 5000} for index in range(10)
                ]
            },
        )

        compact = guard.compact_grounding_report(report, max_bytes=20_000)

        self.assertEqual(compact["validated_ledger"], ledger)
        self.assertEqual(
            [item["step"] for item in compact["trajectory_summary"]["snapshots"]],
            [0, 9],
        )
        self.assertEqual(compact["trajectory_summary"]["snapshots_truncated"], 8)
        self.assertNotIn("compaction", compact)

    def test_mapping_key_flood_is_bounded_and_keeps_validation(self) -> None:
        ledger = {
            "schema": guard.GROUNDING_LEDGER_SCHEMA_ID,
            "analysis_status": "uncertain",
            "observed_facts": [],
            "inferred_facts": [],
            "creative_additions": [],
            "uncertainties": ["not verified"],
            "grounding_failure_reasons": ["insufficient evidence"],
        }
        for key_count, byte_limit in ((50_000, guard.MAX_COMPACT_REPORT_BYTES), (10_000, 2048)):
            with self.subTest(key_count=key_count, byte_limit=byte_limit):
                report = _required_report(ledger)
                report["transport"] = {f"key-{index:05d}": "value" for index in range(key_count)}
                report["validation"] = {
                    "schema_valid": False,
                    "blocking_reasons": ["insufficient_coverage"],
                }
                compact = guard.compact_grounding_report(report, max_bytes=byte_limit)
                self.assertLessEqual(
                    len(guard.grounding_report_json(compact).encode("utf-8")),
                    byte_limit,
                )
                self.assertIn("transport", compact)
                self.assertIn("validation", compact)
                self.assertTrue(
                    any(
                        key.startswith("__dg_omitted_mapping_keys")
                        for key in compact["transport"]
                    )
                )

    def test_single_extreme_mapping_key_is_truncated(self) -> None:
        ledger = {
            "schema": guard.GROUNDING_LEDGER_SCHEMA_ID,
            "analysis_status": "uncertain",
            "observed_facts": [],
            "inferred_facts": [],
            "creative_additions": [],
            "uncertainties": ["not verified"],
            "grounding_failure_reasons": ["insufficient evidence"],
        }
        report = _required_report(ledger)
        report["transport"] = {"K" * 300_000: "value"}
        report["validation"] = {"schema_valid": False}
        compact = guard.compact_grounding_report(report)
        self.assertLessEqual(
            len(guard.grounding_report_json(compact).encode("utf-8")),
            guard.MAX_COMPACT_REPORT_BYTES,
        )
        self.assertIn("validation", compact)
        self.assertLess(max(len(key) for key in compact["transport"]), 300_000)


if __name__ == "__main__":
    unittest.main()
