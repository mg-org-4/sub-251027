"""Security boundary for jobs ownership filtering and list projections."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Iterator, Literal

from .management_query import normalize_limit_offset
from .tenant_context import (
    DEFAULT_TENANT_ID,
    TenantBoundaryError,
    extract_tenant_from_headers,
    is_multi_tenant_enabled,
    normalize_tenant_id,
    request_tenant_scope,
)

JobSource = Literal["queue", "history"]

ALLOWED_JOB_STATUSES = frozenset(
    {"pending", "in_progress", "completed", "failed", "cancelled"}
)
SAFE_JOB_SUMMARY_FIELDS = frozenset(
    {
        "id",
        "status",
        "priority",
        "create_time",
        "execution_start_time",
        "execution_end_time",
        "outputs_count",
        "workflow_id",
    }
)
MAX_JOB_IDENTIFIER_LENGTH = 128
MAX_OUTPUT_COUNT = 1_000_000
MAX_ABSOLUTE_NUMBER = 1_000_000_000_000_000_000
DEFAULT_JOBS_LIMIT = 50
MAX_JOBS_LIMIT = 200
MAX_JOBS_OFFSET = 10_000
MAX_JOBS_SOURCE_WINDOW = 10_000
ALLOWED_JOB_SORT_FIELDS = frozenset({"created_at", "execution_duration"})
ALLOWED_JOB_SORT_ORDERS = frozenset({"asc", "desc"})
ALLOWED_JOB_QUERY_FIELDS = frozenset(
    {"status", "workflow_id", "sort_by", "sort_order", "limit", "offset"}
)
SAFE_JOB_AUDIT_OUTCOMES = frozenset(
    {"allow", "deny", "rate_limit", "unsupported", "error"}
)
SAFE_JOB_AUDIT_REASONS = frozenset(
    {
        "stub",
        "jobs_listed",
        "jobs_admin_required",
        "jobs_rate_limited",
        "jobs_query_invalid",
        "jobs_host_contract_unsupported",
        "jobs_backend_unavailable",
        "tenant_required",
        "tenant_mismatch",
        "tenant_invalid",
        "jobs_error",
    }
)
SAFE_JOB_AUDIT_COUNT_FIELDS = frozenset(
    {"returned_count", "excluded_count", "malformed_count"}
)
_BOOTSTRAP_TOKEN_IDS = frozenset({"env-admin", "local-admin", "local-internal"})


class JobsSecurityError(ValueError):
    """Raised when an upstream jobs value cannot cross the list boundary safely."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class VisibilityFilterResult:
    """Immutable result of filtering raw queue/history records by ownership."""

    records: tuple[Any, ...]
    excluded_count: int
    malformed_count: int


@dataclass(frozen=True)
class JobsQueryWarning:
    """Bounded pagination warning without echoing raw request input."""

    code: str
    field: str
    normalized: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "field": self.field,
            "normalized": self.normalized,
        }


@dataclass(frozen=True)
class JobsQuery:
    """Normalized immutable jobs list query."""

    status: str | None
    workflow_id: str | None
    sort_by: str
    sort_order: str
    limit: int
    offset: int
    warnings: tuple[JobsQueryWarning, ...]

    def to_pagination(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "offset": self.offset,
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


@contextlib.contextmanager
def jobs_request_tenant_scope(request: Any, token_info: Any) -> Iterator[Any]:
    """Bind an explicit jobs tenant context without bootstrap default fallback."""

    if is_multi_tenant_enabled():
        headers = getattr(request, "headers", None)
        header_tenant = (
            extract_tenant_from_headers(headers)
            if isinstance(headers, Mapping)
            else None
        )
        token_id = str(getattr(token_info, "token_id", "") or "")
        token_tenant = str(
            getattr(token_info, "tenant_id", DEFAULT_TENANT_ID) or DEFAULT_TENANT_ID
        )

        # CRITICAL: bootstrap/local token resolution uses `default` when the
        # tenant header is absent. Jobs must not reinterpret that fallback as
        # an explicit cross-job tenant authorization.
        if (
            header_tenant is None
            and token_id in _BOOTSTRAP_TOKEN_IDS
            and token_tenant == DEFAULT_TENANT_ID
        ):
            raise TenantBoundaryError(
                "tenant_required",
                "Explicit tenant context is required for jobs in multi-tenant mode.",
            )

    with request_tenant_scope(
        request=request,
        token_info=token_info,
        allow_default_when_missing=False,
    ) as context:
        yield context


def filter_visible_job_records(
    records: Iterable[Any],
    *,
    source: JobSource,
    tenant_id: str,
    multi_tenant: bool,
) -> VisibilityFilterResult:
    """Filter raw records before upstream normalization or pagination."""

    if source not in {"queue", "history"}:
        raise JobsSecurityError("jobs_source_invalid", "Unsupported jobs source.")

    materialized = tuple(records)
    if not multi_tenant:
        return VisibilityFilterResult(
            records=materialized,
            excluded_count=0,
            malformed_count=0,
        )

    expected_tenant = normalize_tenant_id(tenant_id)
    visible: list[Any] = []
    malformed_count = 0

    for record in materialized:
        owner, malformed = _extract_owner(record, source=source)
        if malformed:
            malformed_count += 1
        if owner == expected_tenant:
            visible.append(record)

    return VisibilityFilterResult(
        records=tuple(visible),
        excluded_count=len(materialized) - len(visible),
        malformed_count=malformed_count,
    )


def normalize_jobs_query(query: Mapping[str, Any]) -> JobsQuery:
    """Normalize the frozen jobs filter/sort/pagination contract."""

    if not isinstance(query, Mapping):
        raise JobsSecurityError("jobs_query_invalid", "Jobs query must be a mapping.")

    unknown = set(query) - ALLOWED_JOB_QUERY_FIELDS
    if unknown:
        raise JobsSecurityError(
            "jobs_query_invalid", "Jobs query contains unsupported fields."
        )

    status = _optional_query_value(query.get("status"), field="status")
    if status is not None and status not in ALLOWED_JOB_STATUSES:
        raise JobsSecurityError("jobs_query_invalid", "Unsupported jobs status.")

    workflow_id = _optional_query_value(query.get("workflow_id"), field="workflow_id")
    sort_by = (
        _optional_query_value(query.get("sort_by"), field="sort_by") or "created_at"
    )
    if sort_by not in ALLOWED_JOB_SORT_FIELDS:
        raise JobsSecurityError("jobs_query_invalid", "Unsupported jobs sort field.")
    sort_order = (
        _optional_query_value(query.get("sort_order"), field="sort_order") or "desc"
    )
    if sort_order not in ALLOWED_JOB_SORT_ORDERS:
        raise JobsSecurityError("jobs_query_invalid", "Unsupported jobs sort order.")

    page = normalize_limit_offset(
        dict(query),
        default_limit=DEFAULT_JOBS_LIMIT,
        max_limit=MAX_JOBS_LIMIT,
        default_offset=0,
        max_offset=MAX_JOBS_OFFSET,
    )
    warnings = tuple(
        JobsQueryWarning(
            code=str(warning.get("code") or "jobs_query_normalized"),
            field=str(warning.get("field") or "query"),
            normalized=int(warning.get("normalized") or 0),
        )
        for warning in page.warnings
    )
    return JobsQuery(
        status=status,
        workflow_id=workflow_id,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=page.limit,
        offset=page.offset,
        warnings=warnings,
    )


def build_jobs_audit_details(reason: Any, **counts: Any) -> dict[str, Any]:
    """Build content-free jobs audit details from safe codes and aggregate counts."""

    safe_reason = str(reason or "")
    if safe_reason not in SAFE_JOB_AUDIT_REASONS:
        safe_reason = "jobs_error"
    details: dict[str, Any] = {"reason": safe_reason}
    for field in SAFE_JOB_AUDIT_COUNT_FIELDS:
        value = counts.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        details[field] = max(0, min(value, MAX_JOBS_SOURCE_WINDOW))
    return details


def project_job_summary(job: Mapping[str, Any]) -> dict[str, Any]:
    """Project an upstream normalized job onto the frozen list allowlist."""

    if not isinstance(job, Mapping):
        raise JobsSecurityError("jobs_record_invalid", "Job must be an object.")

    projected: dict[str, Any] = {
        "id": _bounded_identifier(job.get("id"), field="id"),
        "status": _validated_status(job.get("status")),
    }

    for field in (
        "priority",
        "create_time",
        "execution_start_time",
        "execution_end_time",
    ):
        if field in job and job[field] is not None:
            projected[field] = _bounded_number(job[field], field=field)

    if "outputs_count" in job and job["outputs_count"] is not None:
        outputs_count = job["outputs_count"]
        if (
            isinstance(outputs_count, bool)
            or not isinstance(outputs_count, int)
            or outputs_count < 0
            or outputs_count > MAX_OUTPUT_COUNT
        ):
            raise JobsSecurityError(
                "jobs_record_invalid", "outputs_count is outside the safe bound."
            )
        projected["outputs_count"] = outputs_count

    if "workflow_id" in job and job["workflow_id"] is not None:
        projected["workflow_id"] = _bounded_identifier(
            job["workflow_id"], field="workflow_id"
        )

    return projected


def _extract_owner(record: Any, *, source: JobSource) -> tuple[str | None, bool]:
    extra_data: Any
    if source == "queue":
        if not isinstance(record, (list, tuple)) or len(record) < 4:
            return None, True
        extra_data = record[3]
    else:
        if not isinstance(record, Mapping):
            return None, True
        prompt = record.get("prompt")
        if not isinstance(prompt, (list, tuple)) or len(prompt) < 4:
            return None, True
        extra_data = prompt[3]

    if not isinstance(extra_data, Mapping):
        return None, True
    openclaw = extra_data.get("openclaw")
    if openclaw is None:
        return None, False
    if not isinstance(openclaw, Mapping):
        return None, True
    owner = openclaw.get("tenant_id")
    if owner is None:
        return None, False
    try:
        return normalize_tenant_id(owner), False
    except TenantBoundaryError:
        return None, True


def _bounded_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise JobsSecurityError(
            "jobs_record_invalid", f"{field} must be a non-empty string."
        )
    if len(value) > MAX_JOB_IDENTIFIER_LENGTH:
        raise JobsSecurityError(
            "jobs_record_invalid", f"{field} exceeds the safe length bound."
        )
    return value


def _optional_query_value(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise JobsSecurityError(
            "jobs_query_invalid", f"{field} must be a non-empty string."
        )
    if len(value) > MAX_JOB_IDENTIFIER_LENGTH:
        raise JobsSecurityError(
            "jobs_query_invalid", f"{field} exceeds the safe length bound."
        )
    return value


def _validated_status(value: Any) -> str:
    if not isinstance(value, str) or value not in ALLOWED_JOB_STATUSES:
        raise JobsSecurityError(
            "jobs_record_invalid", "status is outside the jobs lifecycle contract."
        )
    return str(value)


def _bounded_number(value: Any, *, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JobsSecurityError("jobs_record_invalid", f"{field} must be numeric.")
    # Check integer magnitude before float conversion; hostile large integers
    # can otherwise raise OverflowError inside math.isfinite().
    if isinstance(value, int):
        valid = abs(value) <= MAX_ABSOLUTE_NUMBER
    else:
        valid = math.isfinite(value) and abs(value) <= MAX_ABSOLUTE_NUMBER
    if not valid:
        raise JobsSecurityError(
            "jobs_record_invalid", f"{field} is outside the safe numeric bound."
        )
    return value
