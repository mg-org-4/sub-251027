"""Bounded in-process adapter for the authoritative ComfyUI jobs read contract."""

from __future__ import annotations

import importlib
import itertools
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .jobs_security import (
    MAX_JOBS_SOURCE_WINDOW,
    JobsQuery,
    filter_visible_job_records,
    project_job_summary,
)
from .tenant_context import is_multi_tenant_enabled

JOBS_CONTRACT_VERSION = 1
JOBS_SOURCE = {
    "adapter": "comfy_execution.jobs",
    "authority": "in_process",
}


class JobsReadModelError(RuntimeError):
    """Base class for bounded jobs adapter failures."""

    code = "jobs_backend_unavailable"
    safe_message = "Jobs backend is unavailable."

    def __init__(self) -> None:
        super().__init__(self.safe_message)


class JobsHostContractUnsupported(JobsReadModelError):
    """The active host does not expose the required in-process jobs contract."""

    code = "jobs_host_contract_unsupported"
    safe_message = "Jobs host contract is unsupported."


class JobsBackendUnavailable(JobsReadModelError):
    """The supported host contract exists but cannot provide a safe snapshot."""


def read_jobs(query: JobsQuery, *, tenant_id: str) -> dict[str, Any]:
    """Read one bounded authoritative jobs snapshot without external IO or mutation."""

    try:
        return _read_jobs(query, tenant_id=tenant_id)
    except JobsReadModelError:
        raise
    except Exception as exc:
        # IMPORTANT: host objects and upstream helpers are runtime extension seams.
        # Never leak an arbitrary host exception through the HTTP boundary.
        raise JobsBackendUnavailable() from exc


def _read_jobs(query: JobsQuery, *, tenant_id: str) -> dict[str, Any]:
    get_all_jobs = _resolve_get_all_jobs()
    prompt_queue = _resolve_prompt_queue()
    running, queued, history, examined, truncated = _read_bounded_snapshot(prompt_queue)
    multi_tenant = is_multi_tenant_enabled()

    running_ready, running_excluded, running_malformed = _prepare_queue_records(
        running,
        tenant_id=tenant_id,
        multi_tenant=multi_tenant,
    )
    queued_ready, queued_excluded, queued_malformed = _prepare_queue_records(
        queued,
        tenant_id=tenant_id,
        multi_tenant=multi_tenant,
    )
    history_ready, history_excluded, history_malformed = _prepare_history_records(
        history,
        tenant_id=tenant_id,
        multi_tenant=multi_tenant,
    )

    jobs = _normalize_and_project(
        get_all_jobs,
        running=running_ready,
        queued=queued_ready,
        history=history_ready,
        query=query,
    )
    total = len(jobs)
    page = jobs[query.offset : query.offset + query.limit]
    pagination = query.to_pagination()
    pagination.update(
        {
            "total": total,
            "has_more": query.offset + len(page) < total,
        }
    )

    return {
        "ok": True,
        "contract_version": JOBS_CONTRACT_VERSION,
        "jobs": page,
        "pagination": pagination,
        "source": dict(JOBS_SOURCE),
        "scan": {
            "window": MAX_JOBS_SOURCE_WINDOW,
            "examined": examined,
            "excluded": running_excluded + queued_excluded + history_excluded,
            "malformed": running_malformed + queued_malformed + history_malformed,
            "truncated": truncated,
        },
    }


def _resolve_get_all_jobs() -> Callable[..., Any]:
    try:
        jobs_module = importlib.import_module("comfy_execution.jobs")
    except ModuleNotFoundError as exc:
        if exc.name in {"comfy_execution", "comfy_execution.jobs"}:
            raise JobsHostContractUnsupported() from exc
        raise JobsBackendUnavailable() from exc
    except Exception as exc:
        raise JobsBackendUnavailable() from exc

    get_all_jobs = getattr(jobs_module, "get_all_jobs", None)
    if not callable(get_all_jobs):
        raise JobsHostContractUnsupported()
    return get_all_jobs


def _resolve_prompt_queue() -> Any:
    # CRITICAL: use the already-loaded host module only. Importing a request-selected
    # module or calling the local HTTP route would bypass the in-process trust boundary.
    server_module = sys.modules.get("server")
    if server_module is None:
        raise JobsBackendUnavailable()
    prompt_server = getattr(server_module, "PromptServer", None)
    if prompt_server is None:
        raise JobsHostContractUnsupported()
    instance = getattr(prompt_server, "instance", None)
    if instance is None:
        raise JobsBackendUnavailable()
    prompt_queue = getattr(instance, "prompt_queue", None)
    if prompt_queue is None:
        raise JobsBackendUnavailable()
    for name in ("get_current_queue_volatile", "get_history"):
        if not callable(getattr(prompt_queue, name, None)):
            raise JobsHostContractUnsupported()
    return prompt_queue


def _read_bounded_snapshot(
    prompt_queue: Any,
) -> tuple[list[Any], list[Any], dict[Any, Any], int, bool]:
    try:
        queue_snapshot = prompt_queue.get_current_queue_volatile()
        history_snapshot = prompt_queue.get_history()
        if not isinstance(queue_snapshot, (list, tuple)) or len(queue_snapshot) != 2:
            raise TypeError("queue snapshot must contain running and pending lists")
        running_source, queued_source = queue_snapshot
        if not isinstance(running_source, (list, tuple)) or not isinstance(
            queued_source, (list, tuple)
        ):
            raise TypeError("queue snapshot sources must be sequences")
        if not isinstance(history_snapshot, Mapping):
            raise TypeError("history snapshot must be a mapping")

        available = len(running_source) + len(queued_source) + len(history_snapshot)
        remaining = MAX_JOBS_SOURCE_WINDOW
        running = list(itertools.islice(running_source, remaining))
        remaining -= len(running)
        queued = list(itertools.islice(queued_source, remaining))
        remaining -= len(queued)
        history_items = list(itertools.islice(history_snapshot.items(), remaining))
        history = dict(history_items)
        examined = len(running) + len(queued) + len(history)
    except Exception as exc:
        raise JobsBackendUnavailable() from exc

    return running, queued, history, examined, available > examined


def _prepare_queue_records(
    records: Sequence[Any],
    *,
    tenant_id: str,
    multi_tenant: bool,
) -> tuple[list[tuple[Any, ...]], int, int]:
    try:
        visibility = filter_visible_job_records(
            records,
            source="queue",
            tenant_id=tenant_id,
            multi_tenant=multi_tenant,
        )
    except Exception as exc:
        raise JobsBackendUnavailable() from exc

    ready: list[tuple[Any, ...]] = []
    additional_malformed = 0
    for record in visibility.records:
        if (
            not isinstance(record, (list, tuple))
            or len(record) < 5
            or not isinstance(record[3], Mapping)
        ):
            additional_malformed += 1
            continue
        # Upstream normalizers require the sensitive queue tuple to be reduced to
        # its canonical five-element list shape before delegation.
        ready.append(tuple(record[:5]))

    return (
        ready,
        visibility.excluded_count + additional_malformed,
        visibility.malformed_count + additional_malformed,
    )


def _prepare_history_records(
    history: Mapping[Any, Any],
    *,
    tenant_id: str,
    multi_tenant: bool,
) -> tuple[dict[Any, dict[str, Any]], int, int]:
    items = list(history.items())
    records = [record for _, record in items]
    try:
        visibility = filter_visible_job_records(
            records,
            source="history",
            tenant_id=tenant_id,
            multi_tenant=multi_tenant,
        )
    except Exception as exc:
        raise JobsBackendUnavailable() from exc

    visible_id_counts = Counter(id(record) for record in visibility.records)
    ready: dict[Any, dict[str, Any]] = {}
    additional_malformed = 0
    for prompt_id, record in items:
        record_id = id(record)
        if visible_id_counts[record_id] <= 0:
            continue
        visible_id_counts[record_id] -= 1
        if not isinstance(record, Mapping):
            additional_malformed += 1
            continue
        prompt = record.get("prompt")
        if (
            not isinstance(prompt, (list, tuple))
            or len(prompt) < 5
            or not isinstance(prompt[3], Mapping)
        ):
            additional_malformed += 1
            continue
        sanitized = dict(record)
        sanitized["prompt"] = tuple(prompt[:5])
        ready[prompt_id] = sanitized

    return (
        ready,
        visibility.excluded_count + additional_malformed,
        visibility.malformed_count + additional_malformed,
    )


def _normalize_and_project(
    get_all_jobs: Callable[..., Any],
    *,
    running: list[tuple[Any, ...]],
    queued: list[tuple[Any, ...]],
    history: dict[Any, dict[str, Any]],
    query: JobsQuery,
) -> list[dict[str, Any]]:
    try:
        normalized = get_all_jobs(
            running,
            queued,
            history,
            status_filter=[query.status] if query.status else None,
            workflow_id=query.workflow_id,
            sort_by=query.sort_by,
            sort_order=query.sort_order,
            limit=None,
            offset=0,
        )
        if not isinstance(normalized, (list, tuple)) or len(normalized) != 2:
            raise TypeError("get_all_jobs must return jobs and total")
        raw_jobs, upstream_total = normalized
        if not isinstance(raw_jobs, (list, tuple)):
            raise TypeError("normalized jobs must be a sequence")
        if (
            isinstance(upstream_total, bool)
            or not isinstance(upstream_total, int)
            or upstream_total < 0
            or upstream_total != len(raw_jobs)
            or len(raw_jobs) > MAX_JOBS_SOURCE_WINDOW
        ):
            raise ValueError("normalized jobs total is inconsistent")
        return [project_job_summary(job) for job in raw_jobs]
    except JobsReadModelError:
        raise
    except Exception as exc:
        raise JobsBackendUnavailable() from exc
