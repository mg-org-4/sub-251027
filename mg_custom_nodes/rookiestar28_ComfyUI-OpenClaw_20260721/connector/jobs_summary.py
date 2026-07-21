"""Strict, bounded formatter for the connector's authoritative jobs view."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

JOBS_CONTRACT_VERSION = 1
MAX_RETURNED_JOBS = 200
MAX_SNAPSHOT_TOTAL = 10_000
MAX_JOB_ID_LENGTH = 128
MAX_DISPLAY_JOB_ID_LENGTH = 24
MAX_JOBS_SUMMARY_LENGTH = 1_000
MAX_QUEUE_REMAINING = 1_000_000
MAX_NORMALIZATION_WARNINGS = 2

JOB_STATUSES = (
    "pending",
    "in_progress",
    "completed",
    "failed",
    "cancelled",
)
STATUS_LABELS = {
    "pending": "pending",
    "in_progress": "in progress",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
}
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class JobsContractError(ValueError):
    """Raised when a connector jobs payload is not safe to render."""


def format_jobs_summary(payload: Any) -> str:
    """Validate contract version 1 and return a deterministic operator summary."""

    jobs, pagination = _parse_jobs_payload(payload)
    total = pagination["total"]
    if total == 0:
        return "[Jobs] No jobs in the authoritative snapshot."

    counts = Counter(job["status"] for job in jobs)
    active = counts["pending"] + counts["in_progress"]
    terminal = counts["completed"] + counts["failed"] + counts["cancelled"]
    lines = [
        "[Jobs] Authoritative snapshot",
        f"Snapshot total: {total}; returned page: {len(jobs)}",
        (
            f"Page states: Active {active} (pending {counts['pending']}, "
            f"in progress {counts['in_progress']}); "
            f"Terminal {terminal} (completed {counts['completed']}, "
            f"failed {counts['failed']}, cancelled {counts['cancelled']})"
        ),
    ]
    if jobs:
        for job in jobs[:5]:
            lines.append(
                f"- {_short_job_id(job['id'])} — {STATUS_LABELS[job['status']]}"
            )
        if len(jobs) > 5:
            lines.append(f"Showing 5 of {len(jobs)} returned jobs.")
    else:
        lines.append("No jobs are present on this page.")

    summary = "\n".join(lines)
    if len(summary) > MAX_JOBS_SUMMARY_LENGTH:
        raise JobsContractError("jobs summary exceeds the safe display bound")
    return summary


def format_queue_fallback(response: Any) -> str:
    """Render only a bounded coarse queue count from the legacy fallback seam."""

    remaining = _queue_remaining(response)
    if remaining is None:
        return "[Jobs fallback] Coarse queue count is unavailable."
    return (
        f"[Jobs fallback] Queue remaining: {remaining} "
        "(coarse count; not an authoritative jobs snapshot)."
    )


def _parse_jobs_payload(payload: Any) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        raise JobsContractError("jobs response must be a successful mapping")
    version = payload.get("contract_version")
    if isinstance(version, bool) or version != JOBS_CONTRACT_VERSION:
        raise JobsContractError("unsupported jobs contract version")
    raw_jobs = payload.get("jobs")
    pagination = payload.get("pagination")
    if not isinstance(raw_jobs, list) or len(raw_jobs) > MAX_RETURNED_JOBS:
        raise JobsContractError("jobs list is malformed or oversized")
    if not isinstance(pagination, Mapping):
        raise JobsContractError("jobs pagination is missing")
    if not isinstance(payload.get("source"), Mapping) or not isinstance(
        payload.get("scan"), Mapping
    ):
        raise JobsContractError("jobs source diagnostics are missing")

    parsed_jobs = [_parse_job(item) for item in raw_jobs]
    parsed_pagination = _parse_pagination(pagination, returned=len(parsed_jobs))
    return parsed_jobs, parsed_pagination


def _parse_job(item: Any) -> dict[str, str]:
    if not isinstance(item, Mapping):
        raise JobsContractError("job summary must be a mapping")
    job_id = item.get("id")
    status = item.get("status")
    if (
        not isinstance(job_id, str)
        or not job_id
        or len(job_id) > MAX_JOB_ID_LENGTH
        or _SAFE_JOB_ID.fullmatch(job_id) is None
    ):
        raise JobsContractError("job id is outside the safe display contract")
    if not isinstance(status, str) or status not in JOB_STATUSES:
        raise JobsContractError("job status is unsupported")
    return {"id": job_id, "status": status}


def _parse_pagination(
    pagination: Mapping[str, Any], *, returned: int
) -> dict[str, Any]:
    offset = _bounded_int(pagination.get("offset"), minimum=0, maximum=10_000)
    limit = _bounded_int(pagination.get("limit"), minimum=1, maximum=MAX_RETURNED_JOBS)
    total = _bounded_int(pagination.get("total"), minimum=0, maximum=MAX_SNAPSHOT_TOTAL)
    has_more = pagination.get("has_more")
    warnings = pagination.get("warnings")
    if not isinstance(has_more, bool):
        raise JobsContractError("jobs has_more must be boolean")
    if not isinstance(warnings, list) or len(warnings) > MAX_NORMALIZATION_WARNINGS:
        raise JobsContractError("jobs warnings are malformed")
    if returned > limit or offset + returned > total:
        raise JobsContractError("jobs pagination counts are inconsistent")
    if has_more != (offset + returned < total):
        raise JobsContractError("jobs has_more is inconsistent")
    return {
        "offset": offset,
        "limit": limit,
        "total": total,
        "has_more": has_more,
    }


def _bounded_int(value: Any, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise JobsContractError("jobs count must be an integer")
    if value < minimum or value > maximum:
        raise JobsContractError("jobs count is outside the safe bound")
    return value


def _short_job_id(job_id: str) -> str:
    if len(job_id) <= MAX_DISPLAY_JOB_ID_LENGTH:
        return job_id
    return job_id[: MAX_DISPLAY_JOB_ID_LENGTH - 3] + "..."


def _queue_remaining(response: Any) -> int | None:
    if not isinstance(response, Mapping) or response.get("ok") is not True:
        return None
    data = response.get("data")
    if not isinstance(data, Mapping):
        return None
    exec_info = data.get("exec_info")
    if not isinstance(exec_info, Mapping):
        return None
    remaining = exec_info.get("queue_remaining")
    if (
        isinstance(remaining, bool)
        or not isinstance(remaining, int)
        or remaining < 0
        or remaining > MAX_QUEUE_REMAINING
    ):
        return None
    return remaining
