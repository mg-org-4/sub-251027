"""
Legacy compatibility registry and helpers (R149).

Centralizes alias metadata so deprecation handling is not re-implemented ad hoc
across unrelated modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

OPENCLAW_API_PREFIX = "/openclaw"
LEGACY_API_PREFIX = "/moltbot"
API_PREFIXES = (OPENCLAW_API_PREFIX, LEGACY_API_PREFIX)


@dataclass(frozen=True)
class HeaderAlias:
    primary: str
    legacy: str


@dataclass(frozen=True)
class LegacyCompatibilityEntry:
    key: str
    surface: str
    legacy: str
    canonical: str
    status: str
    review_cadence_days: int
    telemetry_signal: str
    removal_criteria: Tuple[str, ...]
    review_trigger: str
    operator_notice: str


ADMIN_TOKEN_HEADERS = HeaderAlias(
    primary="X-OpenClaw-Admin-Token",
    legacy="X-Moltbot-Admin-Token",
)
OBS_TOKEN_HEADERS = HeaderAlias(
    primary="X-OpenClaw-Obs-Token",
    legacy="X-Moltbot-Obs-Token",
)
TRACE_ID_HEADERS = HeaderAlias(
    primary="X-OpenClaw-Trace-Id",
    legacy="X-Moltbot-Trace-Id",
)
WEBHOOK_SIGNATURE_HEADERS = HeaderAlias(
    primary="X-OpenClaw-Signature",
    legacy="X-Moltbot-Signature",
)
WEBHOOK_TIMESTAMP_HEADERS = HeaderAlias(
    primary="X-OpenClaw-Timestamp",
    legacy="X-Moltbot-Timestamp",
)
WEBHOOK_NONCE_HEADERS = HeaderAlias(
    primary="X-OpenClaw-Nonce",
    legacy="X-Moltbot-Nonce",
)

DEPRECATED_OBSERVED = "deprecated-observed"
RETAINED_COMPATIBILITY = "retained-compatibility"

LEGACY_ROUTE_COMPATIBILITY_KEY = "api-path-moltbot-prefix"

_STANDARD_REMOVAL_CRITERIA = (
    "No observed compatibility usage for two consecutive review windows.",
    "A public migration path exists for the canonical OpenClaw surface.",
    "Removal is covered by targeted regression tests and release notes.",
)

_LEGACY_COMPATIBILITY_ENTRIES = (
    LegacyCompatibilityEntry(
        key=LEGACY_ROUTE_COMPATIBILITY_KEY,
        surface="api_path",
        legacy="/moltbot/* and /api/moltbot/*",
        canonical="/openclaw/* and /api/openclaw/*",
        status=DEPRECATED_OBSERVED,
        review_cadence_days=90,
        telemetry_signal="legacy_api_hits",
        removal_criteria=_STANDARD_REMOVAL_CRITERIA,
        review_trigger="Review whenever legacy_api_hits remains zero for the review window or rises after a release.",
        operator_notice="Use canonical /openclaw routes; legacy routes emit deprecation headers.",
    ),
    LegacyCompatibilityEntry(
        key="header-x-moltbot-aliases",
        surface="header",
        legacy="X-Moltbot-* request headers",
        canonical="X-OpenClaw-* request headers",
        status=DEPRECATED_OBSERVED,
        review_cadence_days=90,
        telemetry_signal="legacy_api_hits",
        removal_criteria=_STANDARD_REMOVAL_CRITERIA,
        review_trigger="Review with API-path telemetry and any legacy-header warning logs.",
        operator_notice="Prefer X-OpenClaw-* headers; legacy headers log deprecation warnings.",
    ),
    LegacyCompatibilityEntry(
        key="environment-moltbot-prefix",
        surface="environment",
        legacy="MOLTBOT_* environment variables",
        canonical="OPENCLAW_* environment variables",
        status=RETAINED_COMPATIBILITY,
        review_cadence_days=180,
        telemetry_signal="configuration diagnostics and deprecation warning logs",
        removal_criteria=_STANDARD_REMOVAL_CRITERIA,
        review_trigger="Review when config diagnostics show no legacy env usage across supported deployment profiles.",
        operator_notice="Prefer OPENCLAW_* variables; legacy MOLTBOT_* fallbacks remain compatibility-only.",
    ),
    LegacyCompatibilityEntry(
        key="ui-class-moltbot-prefix",
        surface="ui_class",
        legacy="moltbot-* CSS classes and local UI keys",
        canonical="openclaw-* CSS classes and local UI keys",
        status=RETAINED_COMPATIBILITY,
        review_cadence_days=180,
        telemetry_signal="frontend compatibility helper tests and user-reported extension compatibility",
        removal_criteria=_STANDARD_REMOVAL_CRITERIA,
        review_trigger="Review when canonical frontend markup has shipped through two stable release windows.",
        operator_notice="Use openclaw-* selectors for new integrations; moltbot-* aliases are generated for older extensions.",
    ),
    LegacyCompatibilityEntry(
        key="workflow-node-moltbot-classes",
        surface="workflow_node",
        legacy="Moltbot* node class aliases and moltbot node category",
        canonical="OpenClaw* node classes and openclaw node category",
        status=RETAINED_COMPATIBILITY,
        review_cadence_days=180,
        telemetry_signal="workflow portability diagnostics and node-registration regression tests",
        removal_criteria=_STANDARD_REMOVAL_CRITERIA,
        review_trigger="Review after node metadata migration proves older workflows keep deterministic replacement hints.",
        operator_notice="Keep canonical OpenClaw node names in new workflows; legacy Moltbot names remain for older workflow loads.",
    ),
)

_LEGACY_COMPATIBILITY_BY_KEY: Dict[str, LegacyCompatibilityEntry] = {
    entry.key: entry for entry in _LEGACY_COMPATIBILITY_ENTRIES
}


def _header_value(headers: Mapping[str, str], name: str) -> str:
    value = headers.get(name)
    if value is None:
        return ""
    return str(value).strip()


def _increment_legacy_api_hits() -> None:
    try:
        from .metrics import metrics
    except ImportError:
        metrics = None
    if metrics:
        metrics.inc("legacy_api_hits")


def emit_legacy_header_warning(
    alias: HeaderAlias,
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    _increment_legacy_api_hits()
    active_logger = logger or logging.getLogger(
        "ComfyUI-OpenClaw.services.legacy_compat"
    )
    active_logger.warning(
        "DEPRECATION WARNING: Legacy header %s used. Please migrate to %s.",
        alias.legacy,
        alias.primary,
    )


def get_header_alias_value(
    headers: Mapping[str, str],
    alias: HeaderAlias,
    *,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, bool]:
    """
    Return the canonical header value, falling back to the legacy alias.

    Returns `(value, used_legacy)` and emits the standard deprecation warning
    exactly when the legacy header supplied the effective value.
    """
    value = _header_value(headers, alias.primary)
    if value:
        return value, False

    legacy_value = _header_value(headers, alias.legacy)
    if legacy_value:
        emit_legacy_header_warning(alias, logger=logger)
        return legacy_value, True

    return "", False


def iter_legacy_compatibility_entries() -> Tuple[LegacyCompatibilityEntry, ...]:
    return _LEGACY_COMPATIBILITY_ENTRIES


def get_legacy_compatibility_entry(
    key: str,
) -> Optional[LegacyCompatibilityEntry]:
    return _LEGACY_COMPATIBILITY_BY_KEY.get(key)


def canonicalize_legacy_api_path(path: str) -> str:
    if path.startswith("/api" + LEGACY_API_PREFIX + "/"):
        return path.replace(LEGACY_API_PREFIX, OPENCLAW_API_PREFIX, 1)
    if path.startswith(LEGACY_API_PREFIX + "/"):
        return path.replace(LEGACY_API_PREFIX, OPENCLAW_API_PREFIX, 1)
    return path


def build_legacy_route_deprecation_headers(path: str) -> Dict[str, str]:
    canonical_path = canonicalize_legacy_api_path(path)
    if canonical_path == path:
        return {}

    entry = _LEGACY_COMPATIBILITY_BY_KEY[LEGACY_ROUTE_COMPATIBILITY_KEY]
    return {
        "Deprecation": "true",
        "X-OpenClaw-Compatibility-Key": entry.key,
        "X-OpenClaw-Compatibility-Status": entry.status,
        "X-OpenClaw-Compatibility-Telemetry": entry.telemetry_signal,
        "X-OpenClaw-Canonical-Path": canonical_path,
    }


def get_api_path_candidates(path: str) -> Tuple[str, ...]:
    if path.startswith(OPENCLAW_API_PREFIX + "/"):
        return (path, path.replace(OPENCLAW_API_PREFIX, LEGACY_API_PREFIX, 1))
    if path.startswith(LEGACY_API_PREFIX + "/"):
        return (path, path.replace(LEGACY_API_PREFIX, OPENCLAW_API_PREFIX, 1))
    return (path,)
