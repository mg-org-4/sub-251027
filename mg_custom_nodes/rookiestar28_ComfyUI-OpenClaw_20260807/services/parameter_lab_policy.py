"""Versioned, dependency-light validation policy for Parameter Lab creation."""

from __future__ import annotations

import json
import math
from types import MappingProxyType
from typing import Any

PARAMETER_LAB_POLICY_VERSION = "1.0"
MAX_PARAMETER_LAB_REQUEST_BYTES = 5 * 1024 * 1024
MAX_PARAMETER_LAB_WORKFLOW_UTF8_BYTES = 4 * 1024 * 1024
MAX_SWEEP_DIMENSIONS = 8
MAX_VALUES_PER_DIMENSION = 50
MAX_NODE_ID_UTF8_BYTES = 128
MAX_WIDGET_NAME_UTF8_BYTES = 256
MAX_SCALAR_STRING_UTF8_BYTES = 16 * 1024
MAX_PARAMETER_LAB_PLAN_UTF8_BYTES = 8 * 1024 * 1024
MAX_SWEEP_COMBINATIONS = 50
MAX_COMPARE_ITEMS = 8

PARAMETER_LAB_POLICY = MappingProxyType(
    {
        "version": PARAMETER_LAB_POLICY_VERSION,
        "max_request_bytes": MAX_PARAMETER_LAB_REQUEST_BYTES,
        "max_workflow_utf8_bytes": MAX_PARAMETER_LAB_WORKFLOW_UTF8_BYTES,
        "max_sweep_dimensions": MAX_SWEEP_DIMENSIONS,
        "max_values_per_dimension": MAX_VALUES_PER_DIMENSION,
        "max_node_id_utf8_bytes": MAX_NODE_ID_UTF8_BYTES,
        "max_widget_name_utf8_bytes": MAX_WIDGET_NAME_UTF8_BYTES,
        "max_scalar_string_utf8_bytes": MAX_SCALAR_STRING_UTF8_BYTES,
        "max_plan_utf8_bytes": MAX_PARAMETER_LAB_PLAN_UTF8_BYTES,
        "max_sweep_combinations": MAX_SWEEP_COMBINATIONS,
        "max_compare_items": MAX_COMPARE_ITEMS,
    }
)

_ERROR_MESSAGES = MappingProxyType(
    {
        "payload_too_large": "Parameter Lab request exceeds the byte limit",
        "invalid_json": "Request body must be valid JSON",
        "invalid_payload": "Request payload must be an object",
        "workflow_required": "workflow_json is required",
        "workflow_too_large": "workflow_json exceeds the byte limit",
        "params_must_be_list": "params must be a list",
        "items_must_be_list": "items must be a non-empty list",
        "dimensions_required": "At least one sweep dimension is required",
        "too_many_dimensions": "Too many sweep dimensions",
        "invalid_dimension": "Each sweep dimension must be an object",
        "node_id_required": "node_id is required",
        "invalid_node_id": "node_id is not a supported identifier",
        "node_id_too_large": "node_id exceeds the byte limit",
        "widget_name_required": "widget_name is required",
        "invalid_widget_name": "widget_name is not a supported identifier",
        "widget_name_too_large": "widget_name exceeds the byte limit",
        "values_required": "values must be a non-empty list",
        "too_many_values": (
            f"Values per dimension exceeds limit {MAX_VALUES_PER_DIMENSION}"
        ),
        "invalid_scalar_value": "items must contain only scalar values",
        "scalar_string_too_large": "A scalar string exceeds the byte limit",
        "duplicate_ambiguous_value": "Values contain a presentation-ambiguous duplicate",
        "duplicate_dimension": "Duplicate node/widget dimension",
        "invalid_strategy": "Only grid sweep strategy is supported",
        "sweep_too_large": (f"Sweep size exceeds limit {MAX_SWEEP_COMBINATIONS}"),
        "plan_too_large": "Serialized Parameter Lab plan exceeds the byte limit",
    }
)


class ParameterLabValidationError(ValueError):
    """Content-free creation validation error with a stable public reason code."""

    def __init__(self, code: str, *, status: int = 400, message: str | None = None):
        self.code = code
        self.status = status
        super().__init__(
            message or _ERROR_MESSAGES.get(code, "Invalid Parameter Lab request")
        )


def utf8_size(value: str) -> int:
    return len(value.encode("utf-8"))


def _contains_control(value: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in value)


def validate_workflow(workflow: Any) -> str:
    if not isinstance(workflow, str) or not workflow.strip():
        raise ParameterLabValidationError("workflow_required")
    if utf8_size(workflow) > MAX_PARAMETER_LAB_WORKFLOW_UTF8_BYTES:
        raise ParameterLabValidationError("workflow_too_large", status=413)
    return workflow


def normalize_node_id(node_id: Any) -> str:
    if node_id is None:
        raise ParameterLabValidationError("node_id_required")
    if isinstance(node_id, bool):
        raise ParameterLabValidationError("invalid_node_id")
    if isinstance(node_id, int):
        normalized = str(node_id)
    elif isinstance(node_id, str):
        normalized = node_id
    else:
        raise ParameterLabValidationError("invalid_node_id")
    if not normalized.strip():
        raise ParameterLabValidationError("node_id_required")
    # IMPORTANT: run override keys use the first "." as the node/widget separator.
    if "." in normalized or _contains_control(normalized):
        raise ParameterLabValidationError("invalid_node_id")
    if utf8_size(normalized) > MAX_NODE_ID_UTF8_BYTES:
        raise ParameterLabValidationError("node_id_too_large")
    return normalized


def normalize_widget_name(widget_name: Any) -> str:
    if not isinstance(widget_name, str) or not widget_name.strip():
        raise ParameterLabValidationError("widget_name_required")
    if _contains_control(widget_name):
        raise ParameterLabValidationError("invalid_widget_name")
    if utf8_size(widget_name) > MAX_WIDGET_NAME_UTF8_BYTES:
        raise ParameterLabValidationError("widget_name_too_large")
    return widget_name


def _normalize_scalar(value: Any, *, allow_empty_string: bool) -> Any:
    if isinstance(value, str):
        if not allow_empty_string and not value.strip():
            raise ParameterLabValidationError("invalid_scalar_value")
        if utf8_size(value) > MAX_SCALAR_STRING_UTF8_BYTES:
            raise ParameterLabValidationError("scalar_string_too_large")
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ParameterLabValidationError("invalid_scalar_value")


def _presentation_key(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if value == 0:
        return "0"
    if float(value).is_integer():
        return str(int(value))
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def validate_scalar_values(
    values: Any,
    *,
    max_values: int = MAX_VALUES_PER_DIMENSION,
    allow_empty_string: bool = True,
) -> list[Any]:
    if not isinstance(values, list) or not values:
        raise ParameterLabValidationError("values_required")
    if len(values) > max_values:
        raise ParameterLabValidationError("too_many_values")
    normalized: list[Any] = []
    seen_presentations = set()
    for value in values:
        scalar = _normalize_scalar(value, allow_empty_string=allow_empty_string)
        presentation = _presentation_key(scalar)
        if presentation in seen_presentations:
            raise ParameterLabValidationError("duplicate_ambiguous_value")
        seen_presentations.add(presentation)
        normalized.append(scalar)
    return normalized


def validate_sweep_dimensions(params: Any) -> list[dict[str, Any]]:
    if not isinstance(params, list):
        raise ParameterLabValidationError("params_must_be_list")
    if not params:
        raise ParameterLabValidationError("dimensions_required")
    if len(params) > MAX_SWEEP_DIMENSIONS:
        raise ParameterLabValidationError("too_many_dimensions")

    normalized: list[dict[str, Any]] = []
    seen_dimensions = set()
    combinations = 1
    for raw_dimension in params:
        if not isinstance(raw_dimension, dict):
            raise ParameterLabValidationError("invalid_dimension")
        node_id = normalize_node_id(raw_dimension.get("node_id"))
        widget_name = normalize_widget_name(raw_dimension.get("widget_name"))
        dimension_key = (node_id, widget_name)
        if dimension_key in seen_dimensions:
            raise ParameterLabValidationError("duplicate_dimension")
        seen_dimensions.add(dimension_key)

        strategy = raw_dimension.get("strategy", "grid")
        if strategy != "grid":
            raise ParameterLabValidationError("invalid_strategy")
        values = validate_scalar_values(raw_dimension.get("values"))
        combinations *= len(values)
        if combinations > MAX_SWEEP_COMBINATIONS:
            raise ParameterLabValidationError("sweep_too_large")
        normalized.append(
            {
                "node_id": node_id,
                "widget_name": widget_name,
                "values": values,
                "strategy": "grid",
                "count": 0,
            }
        )
    return normalized


def validate_compare_input(
    items: Any, node_id: Any, widget_name: Any
) -> tuple[list[Any], str, str]:
    if not isinstance(items, list) or not items:
        raise ParameterLabValidationError("items_must_be_list")
    if len(items) > MAX_COMPARE_ITEMS:
        raise ParameterLabValidationError(
            "too_many_values",
            message=f"Too many items for comparison (max {MAX_COMPARE_ITEMS})",
        )
    normalized_node_id = normalize_node_id(node_id)
    normalized_widget_name = normalize_widget_name(widget_name)
    normalized_items = validate_scalar_values(
        items,
        max_values=MAX_COMPARE_ITEMS,
        allow_empty_string=False,
    )
    return normalized_items, normalized_node_id, normalized_widget_name


def serialize_plan_payload(payload: Any) -> str:
    try:
        serialized = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ParameterLabValidationError("invalid_scalar_value") from exc
    if utf8_size(serialized) > MAX_PARAMETER_LAB_PLAN_UTF8_BYTES:
        raise ParameterLabValidationError("plan_too_large", status=413)
    return serialized
