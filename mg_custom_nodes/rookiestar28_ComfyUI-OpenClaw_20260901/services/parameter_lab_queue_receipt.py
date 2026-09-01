"""Authoritative, transient prompt-ID carrier for Parameter Lab submissions."""

from __future__ import annotations

import re
from typing import Any

PARAMETER_LAB_RECEIPT_KEY = "__openclaw_parameter_lab_receipt__"
PARAMETER_LAB_RECEIPT_VERSION = 1

_CANONICAL_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-" r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _valid_marker(marker: Any) -> str | None:
    if not isinstance(marker, dict):
        return None
    if set(marker) != {"version", "prompt_id"}:
        return None
    if marker.get("version") != PARAMETER_LAB_RECEIPT_VERSION:
        return None
    prompt_id = marker.get("prompt_id")
    if not isinstance(prompt_id, str) or not _CANONICAL_UUID_RE.fullmatch(prompt_id):
        return None
    return prompt_id


def consume_parameter_lab_queue_receipt(json_data: Any) -> Any:
    """Strip one transient marker and promote its UUID to native ``prompt_id``.

    The function is copy-on-write so other prompt handlers never observe an in-place
    mutation of their input. Invalid markers are still stripped, but never gain
    identifier authority.
    """

    if not isinstance(json_data, dict):
        return json_data
    extra_data = json_data.get("extra_data")
    if not isinstance(extra_data, dict):
        return json_data
    extra_pnginfo = extra_data.get("extra_pnginfo")
    if not isinstance(extra_pnginfo, dict):
        return json_data
    workflow = extra_pnginfo.get("workflow")
    if not isinstance(workflow, dict):
        return json_data
    workflow_extra = workflow.get("extra")
    if (
        not isinstance(workflow_extra, dict)
        or PARAMETER_LAB_RECEIPT_KEY not in workflow_extra
    ):
        return json_data

    marker = workflow_extra.get(PARAMETER_LAB_RECEIPT_KEY)
    prompt_id = _valid_marker(marker)

    # CRITICAL: the carrier is transient; never retain it in queue/history/image metadata.
    next_workflow_extra = dict(workflow_extra)
    next_workflow_extra.pop(PARAMETER_LAB_RECEIPT_KEY, None)
    next_workflow = dict(workflow)
    next_workflow["extra"] = next_workflow_extra
    next_pnginfo = dict(extra_pnginfo)
    next_pnginfo["workflow"] = next_workflow
    next_extra_data = dict(extra_data)
    next_extra_data["extra_pnginfo"] = next_pnginfo
    result = dict(json_data)
    result["extra_data"] = next_extra_data

    if prompt_id is not None:
        # CRITICAL: the frontend assigns this exact UUID only after promptQueued.
        # Preserving a different earlier handler value would cross-assign lifecycle
        # events to the wrong Parameter Lab run.
        result["prompt_id"] = prompt_id
    return result


def register_parameter_lab_queue_receipt_handler(server: Any) -> bool:
    """Register the official ComfyUI on-prompt handler exactly once."""

    handlers = getattr(server, "on_prompt_handlers", ())
    if isinstance(handlers, (list, tuple)) and (
        consume_parameter_lab_queue_receipt in handlers
    ):
        return False
    add_handler = getattr(server, "add_on_prompt_handler", None)
    if not callable(add_handler):
        raise RuntimeError("ComfyUI host does not expose add_on_prompt_handler")
    add_handler(consume_parameter_lab_queue_receipt)
    return True
