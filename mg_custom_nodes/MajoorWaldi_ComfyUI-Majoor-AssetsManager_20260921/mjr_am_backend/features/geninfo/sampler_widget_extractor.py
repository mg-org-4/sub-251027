"""Sampler widget fallback helpers extracted from parser_impl."""

from __future__ import annotations

from typing import Any

from .graph_converter import _inputs, _is_link, _lower, _node_type
from .sampler_tracer import _scalar

_PIPELINE_HINT_KEYWORDS = ("detailer", "highres", "upscale", "refine")


def _has_pipeline_hint(node: dict[str, Any]) -> bool:
    hint = f"{_lower(_node_type(node))} {_lower(node.get('title'))}"
    return any(k in hint for k in _PIPELINE_HINT_KEYWORDS)


def _has_pipeline_inputs(ins: dict[str, Any]) -> bool:
    has_model = _is_link(ins.get("model")) or _is_link(ins.get("model_1"))
    has_conditioning = any(
        _is_link(ins.get(k)) for k in ("positive", "positive_1", "negative", "negative_1")
    )
    has_image_or_pixel = any(_is_link(ins.get(k)) for k in ("image", "images", "pixels"))
    return has_model or has_conditioning or has_image_or_pixel


def _looks_like_pipeline_pass_node(node: dict[str, Any]) -> bool:
    if not _has_pipeline_hint(node):
        return False
    return _has_pipeline_inputs(_inputs(node))


def _node_has_detailer_signals(node: dict[str, Any]) -> bool:
    if not isinstance(node, dict):
        return False
    node_type = _lower(_node_type(node))
    title = _lower(node.get("title"))
    hint = f"{node_type} {title}"
    if any(token in hint for token in ("detailer", "facedetailer", "inpaint")):
        return True
    ins = _inputs(node)
    return any(key in ins for key in ("mask", "masks", "bbox", "segs", "sam_mask"))


_PROXY_WIDGET_FIELDS = ("sampler_name", "scheduler", "steps", "cfg", "denoise", "seed_val")


def _apply_proxy_field(values: dict[str, Any], proxy_values: dict[str, Any], key: str) -> None:
    if values.get(key) is None and proxy_values.get(key) is not None:
        values[key] = _scalar(proxy_values.get(key))


def _apply_proxy_widget_sampler_values(values: dict[str, Any], sampler_node: dict[str, Any]) -> None:
    if not isinstance(values, dict) or not isinstance(sampler_node, dict):
        return
    proxy_values = _extract_proxy_widget_sampler_values(sampler_node)
    if not proxy_values:
        return
    for field in _PROXY_WIDGET_FIELDS:
        _apply_proxy_field(values, proxy_values, field)


def _map_proxy_key(name: str) -> str | None:
    key = _lower(name)
    if key in ("sampler_name", "sampler"):
        return "sampler_name"
    if key == "scheduler":
        return "scheduler"
    if key == "steps":
        return "steps"
    if key in ("cfg", "cfg_scale"):
        return "cfg"
    if key == "denoise":
        return "denoise"
    if key in ("seed", "noise_seed"):
        return "seed_val"
    return None


def _extract_raw_name_from_proxy_item(proxy_item: list[Any]) -> str | None:
    for candidate in reversed(proxy_item):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _proxy_widget_pairs(proxy_widgets: list[Any], widgets_values: list[Any]) -> list[tuple[int, list[Any]]]:
    pairs: list[tuple[int, list[Any]]] = []
    for idx, proxy_item in enumerate(proxy_widgets):
        if idx >= len(widgets_values):
            break
        if isinstance(proxy_item, list) and proxy_item:
            pairs.append((idx, proxy_item))
    return pairs


def _extract_proxy_widget_value(proxy_item: list[Any], value: Any, out: dict[str, Any]) -> None:
    raw_name = _extract_raw_name_from_proxy_item(proxy_item)
    if not raw_name:
        return
    mapped = _map_proxy_key(raw_name)
    if not mapped or mapped in out:
        return
    out[mapped] = value


def _extract_proxy_widget_sampler_values(node: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    props = node.get("properties")
    if not isinstance(props, dict):
        return out

    proxy_widgets = props.get("proxyWidgets")
    widgets_values = node.get("widgets_values")
    if not isinstance(proxy_widgets, list) or not isinstance(widgets_values, list):
        return out

    for idx, proxy_item in _proxy_widget_pairs(proxy_widgets, widgets_values):
        _extract_proxy_widget_value(proxy_item, widgets_values[idx], out)

    return out
