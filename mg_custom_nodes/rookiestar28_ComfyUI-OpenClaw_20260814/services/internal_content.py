"""Explicit internal-only content markers and sanitizer."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

INTERNAL_CONTENT_FLAG = "openclaw_internal"
INTERNAL_CONTENT_ORIGIN = "openclaw.internal_maintenance"

_DROP = object()


def mark_internal_content(
    text: str,
    *,
    kind: str = "maintenance_prompt",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an explicit internal-only payload block."""
    return {
        INTERNAL_CONTENT_FLAG: True,
        "origin": INTERNAL_CONTENT_ORIGIN,
        "visibility": "internal",
        "kind": str(kind or "maintenance_prompt"),
        "content": str(text or ""),
        "metadata": dict(metadata or {}),
    }


def is_internal_content(value: Any) -> bool:
    """Return True only for explicit OpenClaw internal-content markers."""
    if not isinstance(value, Mapping):
        return False
    if value.get(INTERNAL_CONTENT_FLAG) is True:
        return True
    return str(value.get("origin") or "") == INTERNAL_CONTENT_ORIGIN


def sanitize_internal_content(value: Any) -> Any:
    """Remove explicitly marked internal-only content recursively."""
    cleaned = _sanitize(value)
    if cleaned is _DROP:
        return {}
    return cleaned


def _sanitize(value: Any) -> Any:
    if is_internal_content(value):
        return _DROP

    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        for key, raw_child in value.items():
            child = _sanitize(raw_child)
            if child is _DROP:
                continue
            result[key] = child
        return result

    if isinstance(value, list):
        result_list = []
        for item in value:
            child = _sanitize(item)
            if child is _DROP:
                continue
            result_list.append(child)
        return result_list

    return copy.deepcopy(value)
