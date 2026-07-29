"""Deterministic scale baselines for jobs and connector hot paths."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import random
import re
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from connector.config import ConnectorConfig
from connector.contract import CommandRequest
from connector.router import CommandRouter
from services import jobs_read_model
from services.jobs_security import normalize_jobs_query

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "tests" / "performance_baseline_policy.json"
EXPECTED_WORKLOAD_IDS = {
    "backend_jobs_history",
    "connector_jobs_dispatch",
    "frontend_history_outputs",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_performance_policy(policy: object) -> list[str]:
    """Return deterministic schema, budget, and privacy violations."""

    errors: list[str] = []
    if not isinstance(policy, dict):
        return ["policy must be an object"]
    if set(policy) != {
        "schema_version",
        "policy_id",
        "reviewed_on",
        "timing",
        "workloads",
    }:
        errors.append("root keys must match the versioned schema")
    if policy.get("schema_version") != 1:
        errors.append("schema_version must equal 1")

    timing = policy.get("timing")
    if not isinstance(timing, dict) or set(timing) != {
        "clock",
        "enforcement",
        "samples",
    }:
        errors.append("timing keys must match the advisory schema")
    elif (
        timing.get("clock") != "monotonic_high_resolution"
        or timing.get("enforcement") != "advisory_only"
        or timing.get("samples") != 2
    ):
        errors.append("timing must be two advisory monotonic samples")

    workloads = policy.get("workloads")
    if not isinstance(workloads, list):
        errors.append("workloads must be a list")
        workloads = []
    ids = [item.get("id") for item in workloads if isinstance(item, dict)]
    if set(ids) != EXPECTED_WORKLOAD_IDS or len(ids) != len(set(ids)):
        errors.append("workload ids must be unique and complete")

    expected_inputs = {
        "backend_jobs_history": ({"history_records"}, 10_001),
        "connector_jobs_dispatch": ({"returned_jobs"}, 200),
        "frontend_history_outputs": ({"nodes", "refs_per_node"}, 512),
    }
    expected_outputs = {
        "backend_jobs_history": (
            {
                "source_records",
                "examined",
                "total",
                "returned",
                "truncated",
                "queue_snapshot_calls",
                "history_snapshot_calls",
                "upstream_calls",
                "upstream_records",
            },
            "max_payload_bytes",
        ),
        "connector_jobs_dispatch": (
            {"returned_jobs", "client_calls", "fallback_calls", "visible_job_lines"},
            "max_summary_chars",
        ),
        "frontend_history_outputs": (
            {"input_refs", "normalized_outputs", "image_outputs"},
            "max_serialized_bytes",
        ),
    }
    for item in workloads:
        if not isinstance(item, dict) or set(item) != {
            "id",
            "seed",
            "owner",
            "review_after",
            "input",
            "expected",
        }:
            errors.append("workload keys must match the versioned schema")
            continue
        workload_id = item["id"]
        seed = item["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
            errors.append(f"{workload_id}: seed must be a positive integer")
        if not isinstance(item["owner"], str) or not item["owner"]:
            errors.append(f"{workload_id}: owner is required")
        if not isinstance(item["review_after"], str) or not item["review_after"]:
            errors.append(f"{workload_id}: review date is required")

        inputs = item["input"]
        input_schema = expected_inputs.get(workload_id)
        if (
            not isinstance(inputs, dict)
            or input_schema is None
            or set(inputs) != input_schema[0]
        ):
            errors.append(f"{workload_id}: input keys are invalid")
        else:
            for name, value in inputs.items():
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    errors.append(f"{workload_id}: {name} must be positive")
                elif value > input_schema[1]:
                    errors.append(f"{workload_id}: {name} exceeds its safe test bound")
            if workload_id == "frontend_history_outputs" and (
                inputs["nodes"] * inputs["refs_per_node"] > 4096
            ):
                errors.append(
                    "frontend_history_outputs: total refs exceed the safe test bound"
                )

        expected = item["expected"]
        output_schema = expected_outputs.get(workload_id)
        if (
            not isinstance(expected, dict)
            or output_schema is None
            or set(expected) != {"exact", output_schema[1], "digest_sha256"}
            or not isinstance(expected.get("exact"), dict)
            or set(expected["exact"]) != output_schema[0]
        ):
            errors.append(f"{workload_id}: expected budgets are missing")
        else:
            digest = expected.get("digest_sha256")
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                errors.append(f"{workload_id}: canonical digest is invalid")
            maxima = [
                value for key, value in expected.items() if key.startswith("max_")
            ]
            if len(maxima) != 1 or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in maxima
            ):
                errors.append(f"{workload_id}: one positive maximum budget is required")

    serialized = json.dumps(policy, sort_keys=True).lower()
    for forbidden in (
        "latency_threshold",
        "max_seconds",
        "b:\\",
        "/home/",
        "prompt",
        "token",
        "secret",
    ):
        if forbidden in serialized:
            errors.append(f"policy contains forbidden content marker: {forbidden}")
    return errors


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _workload(policy: dict, workload_id: str) -> dict:
    return next(item for item in policy["workloads"] if item["id"] == workload_id)


def _history_record(prompt_id: str, status: str) -> dict:
    return {
        "prompt": (0, prompt_id, {}, {"openclaw": {"tenant_id": "default"}}, []),
        "synthetic_status": status,
    }


class _CountingPromptQueue:
    def __init__(self, history: dict[str, dict]) -> None:
        self.history = history
        self.queue_calls = 0
        self.history_calls = 0

    def get_current_queue_volatile(self):
        self.queue_calls += 1
        return ([], [])

    def get_history(self):
        self.history_calls += 1
        return self.history


def _run_backend_probe(workload: dict) -> tuple[dict, float]:
    size = workload["input"]["history_records"]
    rng = random.Random(workload["seed"])
    statuses = ("completed", "failed", "cancelled")
    history = {
        f"job-{index:05d}": _history_record(
            f"job-{index:05d}", statuses[rng.randrange(len(statuses))]
        )
        for index in range(size)
    }
    queue = _CountingPromptQueue(history)
    upstream_calls = 0
    upstream_records = 0

    def get_all_jobs(running, queued, bounded_history, **_kwargs):
        nonlocal upstream_calls, upstream_records
        upstream_calls += 1
        upstream_records = len(running) + len(queued) + len(bounded_history)
        jobs = [
            {"id": prompt_id, "status": record["synthetic_status"]}
            for prompt_id, record in bounded_history.items()
        ]
        return jobs, len(jobs)

    started = time.perf_counter()
    with (
        patch.object(
            jobs_read_model, "_resolve_get_all_jobs", return_value=get_all_jobs
        ),
        patch.object(jobs_read_model, "_resolve_prompt_queue", return_value=queue),
        patch.object(jobs_read_model, "is_multi_tenant_enabled", return_value=False),
    ):
        body = jobs_read_model.read_jobs(normalize_jobs_query({}), tenant_id="default")
    elapsed = time.perf_counter() - started
    deterministic = {
        "source_records": size,
        "examined": body["scan"]["examined"],
        "total": body["pagination"]["total"],
        "returned": len(body["jobs"]),
        "truncated": body["scan"]["truncated"],
        "queue_snapshot_calls": queue.queue_calls,
        "history_snapshot_calls": queue.history_calls,
        "upstream_calls": upstream_calls,
        "upstream_records": upstream_records,
        "payload_bytes": len(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "digest": _canonical_digest(body),
    }
    return deterministic, elapsed


def _connector_request() -> CommandRequest:
    return CommandRequest(
        platform="test",
        sender_id="admin-user",
        channel_id="scale-channel",
        username="operator",
        message_id="scale-message",
        text="/jobs",
        timestamp=0,
    )


def _run_connector_probe(workload: dict) -> tuple[dict, float]:
    count = workload["input"]["returned_jobs"]
    rng = random.Random(workload["seed"])
    statuses = ("pending", "in_progress", "completed", "failed", "cancelled")
    jobs = [
        {"id": f"job-{index:03d}", "status": statuses[rng.randrange(len(statuses))]}
        for index in range(count)
    ]
    response = {
        "ok": True,
        "status": 200,
        "data": {
            "ok": True,
            "contract_version": 1,
            "jobs": jobs,
            "pagination": {
                "offset": 0,
                "limit": count,
                "total": count,
                "has_more": False,
                "warnings": [],
            },
            "source": {"adapter": "comfy_execution.jobs", "authority": "in_process"},
            "scan": {
                "window": 10_000,
                "examined": count,
                "excluded": 0,
                "malformed": 0,
                "truncated": False,
            },
        },
    }
    config = ConnectorConfig()
    config.admin_users = ["admin-user"]
    config.admin_token = "configured-test-value"
    client = MagicMock()
    client.get_jobs = AsyncMock(return_value=response)
    client.get_prompt_queue = AsyncMock()
    router = CommandRouter(config, client)

    started = time.perf_counter()
    rendered = asyncio.run(router.handle(_connector_request())).text
    elapsed = time.perf_counter() - started
    deterministic = {
        "returned_jobs": count,
        "client_calls": client.get_jobs.await_count,
        "fallback_calls": client.get_prompt_queue.await_count,
        "visible_job_lines": sum(
            line.startswith("- ") for line in rendered.splitlines()
        ),
        "summary_chars": len(rendered),
        "digest": _canonical_digest(rendered),
    }
    return deterministic, elapsed


class TestR218PerformanceBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = _load_policy()

    def test_policy_schema_is_bounded_advisory_and_content_free(self):
        policy = self.policy
        self.assertEqual(policy["schema_version"], 1)
        self.assertEqual(policy["timing"]["enforcement"], "advisory_only")
        self.assertEqual(policy["timing"]["samples"], 2)
        ids = [item["id"] for item in policy["workloads"]]
        self.assertEqual(set(ids), EXPECTED_WORKLOAD_IDS)
        self.assertEqual(len(ids), len(set(ids)))
        serialized = json.dumps(policy, sort_keys=True).lower()
        for forbidden in (
            "latency_threshold",
            "max_seconds",
            "b:\\",
            "/home/",
            "prompt",
            "token",
            "secret",
        ):
            self.assertNotIn(forbidden, serialized)
        for item in policy["workloads"]:
            self.assertIsInstance(item["seed"], int)
            self.assertGreater(item["seed"], 0)
            self.assertTrue(item["owner"])
            self.assertTrue(item["review_after"])

    def test_policy_validator_rejects_schema_budget_and_privacy_drift(self):
        cases = []
        unknown = copy.deepcopy(self.policy)
        unknown["unexpected"] = True
        cases.append(unknown)
        duplicate = copy.deepcopy(self.policy)
        duplicate["workloads"][1]["id"] = duplicate["workloads"][0]["id"]
        cases.append(duplicate)
        timed_gate = copy.deepcopy(self.policy)
        timed_gate["timing"]["max_seconds"] = 1
        cases.append(timed_gate)
        unsafe_size = copy.deepcopy(self.policy)
        unsafe_size["workloads"][0]["input"]["history_records"] = 100_000
        cases.append(unsafe_size)
        unsafe_content = copy.deepcopy(self.policy)
        unsafe_content["policy_id"] = "B:\\private\\scale"
        cases.append(unsafe_content)
        missing_counter = copy.deepcopy(self.policy)
        del missing_counter["workloads"][0]["expected"]["exact"]["examined"]
        cases.append(missing_counter)
        unknown_budget = copy.deepcopy(self.policy)
        unknown_budget["workloads"][1]["expected"]["unexpected"] = 1
        cases.append(unknown_budget)

        self.assertEqual(validate_performance_policy(self.policy), [])
        for invalid in cases:
            with self.subTest(invalid=invalid):
                self.assertTrue(validate_performance_policy(invalid))

    def test_backend_jobs_history_matches_deterministic_budgets(self):
        workload = _workload(self.policy, "backend_jobs_history")
        result, elapsed = _run_backend_probe(workload)
        expected = workload["expected"]
        self.assertEqual(
            {key: result[key] for key in expected["exact"]}, expected["exact"]
        )
        self.assertLessEqual(result["payload_bytes"], expected["max_payload_bytes"])
        self.assertEqual(result["digest"], expected["digest_sha256"])
        self.assertGreaterEqual(elapsed, 0.0)

    def test_connector_dispatch_matches_deterministic_budgets(self):
        workload = _workload(self.policy, "connector_jobs_dispatch")
        result, elapsed = _run_connector_probe(workload)
        expected = workload["expected"]
        self.assertEqual(
            {key: result[key] for key in expected["exact"]}, expected["exact"]
        )
        self.assertLessEqual(result["summary_chars"], expected["max_summary_chars"])
        self.assertEqual(result["digest"], expected["digest_sha256"])
        self.assertGreaterEqual(elapsed, 0.0)

    def test_repeated_runs_compare_deterministic_results_not_timing(self):
        for workload_id, probe in (
            ("backend_jobs_history", _run_backend_probe),
            ("connector_jobs_dispatch", _run_connector_probe),
        ):
            workload = _workload(self.policy, workload_id)
            first, first_elapsed = probe(workload)
            second, second_elapsed = probe(workload)
            self.assertEqual(first, second)
            self.assertGreaterEqual(first_elapsed, 0.0)
            self.assertGreaterEqual(second_elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
