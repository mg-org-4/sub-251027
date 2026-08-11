"""Shared connector reply visibility decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

VISIBLE = "visible"
SUPPRESS_TEXT = "suppress_text"
TOOL_ONLY = "tool_only"
INTERNAL = "internal"
AUTO = "auto"

_VISIBLE_VALUES = {"", AUTO, VISIBLE, "public", "reply", "send"}
_SUPPRESS_VALUES = {SUPPRESS_TEXT, "suppress", "silent", "no_text", "none"}
_TOOL_ONLY_VALUES = {TOOL_ONLY, "tool-only", "tool", "action_only", "action-only"}
_INTERNAL_VALUES = {INTERNAL, "internal_only", "internal-only", "private"}
_TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class ReplyVisibilityDecision:
    visible: bool
    mode: str
    reason: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def suppressed(self) -> bool:
        return not self.visible


def normalize_reply_visibility_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in _VISIBLE_VALUES:
        return VISIBLE
    if text in _SUPPRESS_VALUES:
        return SUPPRESS_TEXT
    if text in _TOOL_ONLY_VALUES:
        return TOOL_ONLY
    if text in _INTERNAL_VALUES:
        return INTERNAL
    return VISIBLE


def decide_reply_visibility(
    *,
    delivery_context: Optional[Dict[str, Any]] = None,
    platform: str = "",
    channel_kind: str = "",
    mentioned: Optional[bool] = None,
    in_thread: bool = False,
    text: str = "",
    has_buttons: bool = False,
    has_files: bool = False,
) -> ReplyVisibilityDecision:
    ctx = dict(delivery_context or {})
    explicit_mode = _extract_mode(ctx)
    mode = normalize_reply_visibility_mode(explicit_mode)
    normalized_channel_kind = (
        str(ctx.get("channel_kind") or ctx.get("chat_type") or channel_kind or "")
        .strip()
        .lower()
    )
    threaded = bool(in_thread or str(ctx.get("thread_id", "") or "").strip())

    diagnostics = {
        "platform": str(platform or ctx.get("platform", "") or "").strip(),
        "mode": mode,
        "channel_kind": normalized_channel_kind,
        "in_thread": threaded,
        "has_text": bool(str(text or "").strip()),
        "has_buttons": bool(has_buttons),
        "has_files": bool(has_files),
    }

    # Approval/action replies must stay visible; hiding them can strand operators.
    if has_buttons:
        return ReplyVisibilityDecision(
            True, VISIBLE, "interactive_action_required", diagnostics
        )

    if mode == INTERNAL or _truthy(ctx.get("internal_delivery")):
        return ReplyVisibilityDecision(
            False, INTERNAL, "internal_delivery", diagnostics
        )

    if (
        mode in {SUPPRESS_TEXT, TOOL_ONLY}
        or _truthy(ctx.get("tool_only"))
        or _truthy(ctx.get("silent"))
    ):
        if has_files:
            return ReplyVisibilityDecision(
                True, VISIBLE, "file_delivery_preserved", diagnostics
            )
        return ReplyVisibilityDecision(
            False, mode, "text_reply_suppressed", diagnostics
        )

    if normalized_channel_kind in {"group", "supergroup", "channel"}:
        if mentioned is None and "mentioned" in ctx:
            mentioned = _truthy(ctx.get("mentioned"))
        if mentioned is None and "mentioned_bot" in ctx:
            mentioned = _truthy(ctx.get("mentioned_bot"))
        if mentioned is False and not threaded:
            return ReplyVisibilityDecision(
                False, SUPPRESS_TEXT, "group_no_mention", diagnostics
            )

    return ReplyVisibilityDecision(True, VISIBLE, "visible", diagnostics)


def _extract_mode(ctx: Dict[str, Any]) -> Any:
    for key in ("reply_visibility", "visibility", "reply_visibility_mode"):
        if key in ctx:
            return ctx.get(key)
    policy = ctx.get("delivery_policy")
    if isinstance(policy, dict):
        for key in ("reply_visibility", "visibility", "reply_visibility_mode"):
            if key in policy:
                return policy.get(key)
    return AUTO


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in _TRUTHY_VALUES
