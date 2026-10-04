"""
Canonical scheduler delivery contract.

Schedules persist a stable delivery shape so CRUD, manual run, and background
execution do not depend on loose connector-specific dictionaries.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

from ..chatops.targets import TargetValidationError, parse_target
from ..chatops.transport_contract import TransportType

DELIVERY_MALFORMED = "delivery_malformed"
DELIVERY_AMBIGUOUS = "delivery_ambiguous"
DELIVERY_UNSUPPORTED = "delivery_unsupported"

SUPPORTED_PLATFORMS = {
    "custom",
    "discord",
    "feishu",
    "lark",
    "slack",
    "telegram",
    "webhook",
}

_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.:@/-]{1,200}$")
_FEISHU_TARGET_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
_FEISHU_THREAD_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,128}$")
_WEBHOOK_SAFE_TARGET_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,128}$")


class DeliveryContractError(ValueError):
    """Raised when schedule delivery cannot be normalized safely."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def normalize_schedule_delivery(value: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize loose schedule delivery data into the canonical persisted contract.

    Patch semantics are handled by callers:
    - omitted delivery: caller preserves existing value
    - ``None``: clear delivery
    - ``{"enabled": false}`` or ``{"mode": "none"}``: explicit no-delivery
    """
    if value is None:
        return None

    if not isinstance(value, dict):
        raise DeliveryContractError(DELIVERY_MALFORMED, "delivery must be an object")

    raw = dict(value)
    if _is_explicit_no_delivery(raw):
        return {"enabled": False, "mode": "none"}

    platform = _resolve_platform(raw)
    target_id = _resolve_alias(
        raw, ("target_id", "channel_id", "chat_id", "room_id", "callback_url", "url")
    )
    thread_id = _resolve_alias(
        raw, ("thread_id", "thread_ts", "message_thread_id", "topic_id")
    )
    workspace_id = _resolve_alias(raw, ("workspace_id", "team_id"))
    account_id = _resolve_alias(raw, ("account_id",))
    mode = _normalize_mode(raw.get("mode", "reply"))
    failure_alert = _normalize_bool(raw.get("failure_alert", True), "failure_alert")

    if not target_id:
        raise DeliveryContractError(
            DELIVERY_MALFORMED, "delivery.target_id is required"
        )

    _validate_platform_target(platform, target_id, thread_id, mode)
    _validate_optional_safe_id(workspace_id, "workspace_id")
    _validate_optional_safe_id(account_id, "account_id")

    normalized: Dict[str, Any] = {
        "enabled": True,
        "platform": platform,
        "target_id": target_id,
    }
    if thread_id:
        normalized["thread_id"] = thread_id
    if workspace_id:
        normalized["workspace_id"] = workspace_id
    if account_id:
        normalized["account_id"] = account_id
    normalized["mode"] = mode
    normalized["failure_alert"] = failure_alert
    return normalized


def _is_explicit_no_delivery(raw: Dict[str, Any]) -> bool:
    enabled = raw.get("enabled")
    if isinstance(enabled, bool) and not enabled:
        return True
    mode = raw.get("mode")
    return isinstance(mode, str) and mode.strip().lower() == "none"


def _resolve_platform(raw: Dict[str, Any]) -> str:
    platform = _resolve_alias(raw, ("platform", "transport"))
    if not platform and "url" in raw:
        platform = "webhook"
    if not platform:
        raise DeliveryContractError(DELIVERY_MALFORMED, "delivery.platform is required")

    platform = platform.lower()
    if platform not in SUPPORTED_PLATFORMS:
        raise DeliveryContractError(
            DELIVERY_UNSUPPORTED, f"unsupported delivery platform: {platform}"
        )
    return "feishu" if platform == "lark" else platform


def _resolve_alias(raw: Dict[str, Any], aliases: Iterable[str]) -> Optional[str]:
    values: list[tuple[str, str]] = []
    for key in aliases:
        if key not in raw:
            continue
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            raise DeliveryContractError(
                DELIVERY_MALFORMED, f"delivery.{key} must be a string"
            )
        text = str(value).strip()
        if not text:
            continue
        values.append((key, text))

    unique_values = {text for _, text in values}
    if len(unique_values) > 1:
        fields = ", ".join(key for key, _ in values)
        raise DeliveryContractError(
            DELIVERY_AMBIGUOUS,
            f"delivery fields conflict: {fields}",
        )
    return values[0][1] if values else None


def _normalize_mode(value: Any) -> str:
    if value is None:
        return "reply"
    if not isinstance(value, str):
        raise DeliveryContractError(
            DELIVERY_MALFORMED, "delivery.mode must be a string"
        )
    mode = value.strip().lower()
    if mode == "none":
        raise DeliveryContractError(
            DELIVERY_MALFORMED,
            "delivery.mode=none must be represented as explicit no-delivery",
        )
    return mode or "reply"


def _normalize_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise DeliveryContractError(
        DELIVERY_MALFORMED, f"delivery.{field_name} must be boolean"
    )


def _validate_platform_target(
    platform: str, target_id: str, thread_id: Optional[str], mode: str
) -> None:
    if platform == "feishu":
        if not _FEISHU_TARGET_PATTERN.match(target_id):
            raise DeliveryContractError(
                DELIVERY_MALFORMED, "invalid delivery.target_id for feishu"
            )
        if thread_id and not _FEISHU_THREAD_PATTERN.match(thread_id):
            raise DeliveryContractError(
                DELIVERY_MALFORMED, "invalid delivery.thread_id for feishu"
            )
        _validate_mode_for_scheduler(mode)
        return

    if platform == "webhook":
        if target_id.startswith(("http://", "https://")):
            return
        if not _WEBHOOK_SAFE_TARGET_PATTERN.match(target_id):
            raise DeliveryContractError(
                DELIVERY_MALFORMED, "invalid delivery.target_id for webhook"
            )
        _validate_mode_for_scheduler(mode)
        return

    try:
        parse_target(TransportType(platform), target_id, thread_id, mode)
    except (TargetValidationError, ValueError) as exc:
        raise DeliveryContractError(DELIVERY_MALFORMED, str(exc)) from exc


def _validate_mode_for_scheduler(mode: str) -> None:
    if mode not in {"reply", "new_thread", "dm", "broadcast"}:
        raise DeliveryContractError(
            DELIVERY_MALFORMED, f"invalid delivery.mode: {mode}"
        )


def _validate_optional_safe_id(value: Optional[str], field_name: str) -> None:
    if value is None:
        return
    if not _SAFE_ID_PATTERN.match(value):
        raise DeliveryContractError(
            DELIVERY_MALFORMED, f"invalid delivery.{field_name}"
        )
