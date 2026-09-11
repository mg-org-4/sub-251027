# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_WORKFLOW = Path("LTX-2.3-INPAINTING-EDIT-ANY-VIDEO.json")
DEFAULT_OUTPUT_WORKFLOW = Path("LTX-2.3-INPAINTING-EDIT-ANY-VIDEO_DG_PROMPT_INJECTED_v0.json")
DEFAULT_MANIFEST = Path("LTX-2.3-INPAINTING-EDIT-ANY-VIDEO_DG_PROMPT_INJECTED_v0.manifest.json")
REPAIR_SCOPE = "localized_masked_inpaint"


@dataclass(frozen=True)
class FieldSpec:
    field_id: str
    source_key: str
    scope: str
    node_id: int
    class_type: str
    title: str
    widget_key: int | str
    role: str
    optional: bool = False


FIELD_SPECS = [
    FieldSpec(
        field_id="video_input",
        source_key="video_path",
        scope="top",
        node_id=5496,
        class_type="VHS_LoadVideo",
        title="VHS_LoadVideo",
        widget_key="video",
        role="video input field",
        optional=True,
    ),
    FieldSpec(
        field_id="reference_image_input",
        source_key="reference_image_path",
        scope="subgraph:fe45e814-d005-4dd9-97dc-e31c4c6e9773:I2V Image",
        node_id=5038,
        class_type="LoadImage",
        title="Reference Object",
        widget_key=0,
        role="reference image input field",
        optional=True,
    ),
    FieldSpec(
        field_id="sam3_target_primary",
        source_key="sam_target",
        scope="subgraph:ba181d97-b80f-49a8-ae30-e552bd7c8f49:Mask",
        node_id=5433,
        class_type="SAM3Segment",
        title="SAM3 Segmentation (RMBG) - 1 Pass",
        widget_key=0,
        role="SAM3 target/prompt field",
    ),
    FieldSpec(
        field_id="sam3_target_pass2_a",
        source_key="sam_target",
        scope="subgraph:f935d44a-3a0d-41cf-b9bc-b2261f8ba60f:New Subgraph",
        node_id=5676,
        class_type="SAM3Segment",
        title="SAM3 Segmentation (RMBG) - 2 Pass",
        widget_key=0,
        role="SAM3 target/prompt field",
    ),
    FieldSpec(
        field_id="sam3_target_pass2_b",
        source_key="sam_target",
        scope="subgraph:a0619931-10e4-48af-9d29-114a59e7c821:New Subgraph",
        node_id=5745,
        class_type="SAM3Segment",
        title="SAM3 Segmentation (RMBG) - 2 Pass",
        widget_key=0,
        role="SAM3 target/prompt field",
    ),
    FieldSpec(
        field_id="ltx_positive_prompt",
        source_key="composed_positive_prompt",
        scope="subgraph:483656b1-20e1-4b50-a7e2-3b6126ad4bb7:Prompt",
        node_id=5630,
        class_type="CLIPTextEncode",
        title="Manual Prompt",
        widget_key=0,
        role="manual LTX positive prompt field",
    ),
    FieldSpec(
        field_id="ltx_negative_prompt",
        source_key="negative_prompt",
        scope="subgraph:483656b1-20e1-4b50-a7e2-3b6126ad4bb7:Prompt",
        node_id=5626,
        class_type="CLIPTextEncode",
        title="CLIPTextEncode",
        widget_key=0,
        role="negative prompt field",
    ),
]


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object at {path}")
    return parsed


def _scope_parts(scope: str) -> tuple[str, str | None, str | None]:
    if scope == "top":
        return "top", None, None
    if not scope.startswith("subgraph:"):
        raise ValueError(f"unsupported field scope: {scope}")
    _, subgraph_id, subgraph_name = scope.split(":", 2)
    return "subgraph", subgraph_id, subgraph_name


def _node_title(node: dict[str, Any]) -> str:
    return str(node.get("title") or node.get("properties", {}).get("Node name for S&R") or "")


def _find_spec_node(workflow: dict[str, Any], spec: FieldSpec) -> tuple[dict[str, Any], str]:
    scope_kind, subgraph_id, _subgraph_name = _scope_parts(spec.scope)
    if scope_kind == "top":
        nodes = workflow.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError("workflow top-level nodes must be a list")
        for index, node in enumerate(nodes):
            if isinstance(node, dict) and node.get("id") == spec.node_id:
                return node, f"/nodes/{index}/widgets_values/{spec.widget_key}"
        raise ValueError(f"missing top-level node {spec.node_id} for {spec.field_id}")

    subgraphs = workflow.get("definitions", {}).get("subgraphs")
    if not isinstance(subgraphs, list):
        raise ValueError("workflow definitions.subgraphs must be a list")
    for subgraph_index, subgraph in enumerate(subgraphs):
        if not isinstance(subgraph, dict) or str(subgraph.get("id")) != str(subgraph_id):
            continue
        nodes = subgraph.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError(f"subgraph {subgraph_id} nodes must be a list")
        for node_index, node in enumerate(nodes):
            if isinstance(node, dict) and node.get("id") == spec.node_id:
                return node, f"/definitions/subgraphs/{subgraph_index}/nodes/{node_index}/widgets_values/{spec.widget_key}"
        raise ValueError(f"missing node {spec.node_id} in subgraph {subgraph_id} for {spec.field_id}")
    raise ValueError(f"missing subgraph {subgraph_id} for {spec.field_id}")


def _get_widget(node: dict[str, Any], widget_key: int | str) -> Any:
    widgets = node.get("widgets_values")
    if isinstance(widget_key, int):
        if not isinstance(widgets, list) or widget_key >= len(widgets):
            raise ValueError(f"widget index {widget_key} is unavailable on node {node.get('id')}")
        return widgets[widget_key]
    if not isinstance(widgets, dict) or widget_key not in widgets:
        raise ValueError(f"widget key {widget_key} is unavailable on node {node.get('id')}")
    return widgets[widget_key]


def _set_widget(node: dict[str, Any], widget_key: int | str, value: str) -> None:
    widgets = node.get("widgets_values")
    if isinstance(widget_key, int):
        if not isinstance(widgets, list) or widget_key >= len(widgets):
            raise ValueError(f"widget index {widget_key} is unavailable on node {node.get('id')}")
        widgets[widget_key] = value
        return
    if not isinstance(widgets, dict) or widget_key not in widgets:
        raise ValueError(f"widget key {widget_key} is unavailable on node {node.get('id')}")
    widgets[widget_key] = value


def _field_inventory(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for spec in FIELD_SPECS:
        node, pointer = _find_spec_node(workflow, spec)
        class_type = str(node.get("type") or "")
        if class_type != spec.class_type:
            raise ValueError(f"{spec.field_id} expected {spec.class_type}, found {class_type}")
        inventory.append(
            {
                "field_id": spec.field_id,
                "role": spec.role,
                "scope": spec.scope,
                "node_id": spec.node_id,
                "class_type": class_type,
                "title": _node_title(node),
                "widget_key": spec.widget_key,
                "json_pointer": pointer,
                "current_value": _get_widget(node, spec.widget_key),
            }
        )
    return inventory


def _as_text(value: Any, key: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value.strip()


def _as_text_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return [item.strip() for item in value if item.strip()]


def _first_text(payload: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = _as_text(payload.get(key), key)
        if value:
            return value
    return ""


def _compose_positive_prompt(payload: dict[str, Any]) -> str:
    positive = _as_text(payload.get("positive_prompt"), "positive_prompt")
    if not positive:
        raise ValueError("positive_prompt is required")
    parts = [positive]
    preservation = _as_text_list(payload, "preservation_constraints")
    if preservation:
        parts.append("Preservation constraints: " + "; ".join(preservation) + ".")
    do_not_change = _as_text_list(payload, "do_not_change")
    if do_not_change:
        parts.append("Do not change: " + "; ".join(do_not_change) + ".")
    repair_scope = _as_text(payload.get("repair_scope"), "repair_scope")
    if repair_scope != REPAIR_SCOPE:
        raise ValueError(f"repair_scope must be {REPAIR_SCOPE!r}")
    parts.append("Repair scope: localized masked inpaint.")
    return " ".join(parts)


def _dg_values(payload: dict[str, Any], video_path: str | None = None, reference_image_path: str | None = None) -> dict[str, str]:
    sam_target = _as_text(payload.get("sam_target"), "sam_target")
    if not sam_target:
        raise ValueError("sam_target is required")
    values = {
        "sam_target": sam_target,
        "composed_positive_prompt": _compose_positive_prompt(payload),
        "negative_prompt": _as_text(payload.get("negative_prompt"), "negative_prompt"),
        "video_path": _as_text(video_path, "video_path") or _first_text(payload, ["video_path", "video_input", "source_video", "video"]),
        "reference_image_path": _as_text(reference_image_path, "reference_image_path")
        or _first_text(payload, ["reference_image_path", "reference_image", "ref_image"]),
    }
    return values


def _diff_leaf_paths(left: Any, right: Any, path: str = "") -> list[str]:
    if type(left) is not type(right):
        return [path or "/"]
    if isinstance(left, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}/{key}"
            if key not in left or key not in right:
                paths.append(child_path)
            else:
                paths.extend(_diff_leaf_paths(left[key], right[key], child_path))
        return paths
    if isinstance(left, list):
        paths = []
        for index in range(max(len(left), len(right))):
            child_path = f"{path}/{index}"
            if index >= len(left) or index >= len(right):
                paths.append(child_path)
            else:
                paths.extend(_diff_leaf_paths(left[index], right[index], child_path))
        return paths
    return [] if left == right else [path or "/"]


def build_patched_workflow(
    workflow_path: Path,
    dg_prompt_json_path: Path,
    output_workflow_path: Path,
    manifest_path: Path,
    *,
    video_path: str | None = None,
    reference_image_path: str | None = None,
) -> dict[str, Any]:
    source_workflow = _load_json(workflow_path)
    patched_workflow = copy.deepcopy(source_workflow)
    dg_payload = _load_json(dg_prompt_json_path)
    values = _dg_values(dg_payload, video_path=video_path, reference_image_path=reference_image_path)
    before_inventory = _field_inventory(source_workflow)
    allowed_pointers = set()
    changes: list[dict[str, Any]] = []
    skipped: list[str] = []

    for spec in FIELD_SPECS:
        value = values.get(spec.source_key, "")
        if not value and spec.optional:
            skipped.append(spec.field_id)
            continue
        node, pointer = _find_spec_node(patched_workflow, spec)
        old_value = _get_widget(node, spec.widget_key)
        _set_widget(node, spec.widget_key, value)
        allowed_pointers.add(pointer)
        if old_value != value:
            changes.append(
                {
                    "field_id": spec.field_id,
                    "role": spec.role,
                    "scope": spec.scope,
                    "node_id": spec.node_id,
                    "class_type": spec.class_type,
                    "title": spec.title,
                    "widget_key": spec.widget_key,
                    "json_pointer": pointer,
                    "old_value": old_value,
                    "new_value": value,
                }
            )

    changed_paths = set(_diff_leaf_paths(source_workflow, patched_workflow))
    forbidden_changes = sorted(changed_paths - allowed_pointers)
    if forbidden_changes:
        raise RuntimeError("topology or non-approved workflow fields changed: " + ", ".join(forbidden_changes))

    output_workflow_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_workflow_path.write_text(_json_dumps(patched_workflow) + "\n", encoding="utf-8")
    reloaded = _load_json(output_workflow_path)
    after_inventory = _field_inventory(reloaded)
    manifest = {
        "schema_version": 1,
        "tool": "dg_to_ltx_prompt_injector_v0",
        "source_workflow_path": str(workflow_path),
        "patched_workflow_path": str(output_workflow_path),
        "dg_prompt_json_path": str(dg_prompt_json_path),
        "manifest_path": str(manifest_path),
        "source_workflow_modified": False,
        "topology_source_of_truth": "LTX-2.3-INPAINTING-EDIT-ANY-VIDEO.json",
        "repair_scope": REPAIR_SCOPE,
        "field_inventory_before": before_inventory,
        "field_inventory_after": after_inventory,
        "changed_widget_count": len(changes),
        "changes": changes,
        "skipped_optional_fields": skipped,
        "changed_json_pointers": sorted(changed_paths),
        "allowed_changed_json_pointers": sorted(allowed_pointers),
        "forbidden_changes": forbidden_changes,
        "only_prompt_path_widgets_changed": not forbidden_changes,
        "sam3_node_count": sum(1 for item in after_inventory if item["class_type"] == "SAM3Segment"),
        "sam3_remains_in_workflow": any(item["class_type"] == "SAM3Segment" for item in after_inventory),
        "ltx_graph_topology_unchanged": not forbidden_changes,
        "patched_workflow_reloaded": bool(reloaded.get("nodes") and reloaded.get("definitions")),
        "generation_run": False,
        "semantic_repair_success_claimed": False,
        "bad_soldier_touched": False,
    }
    manifest_path.write_text(_json_dumps(manifest) + "\n", encoding="utf-8")
    return manifest


def inspect_workflow(workflow_path: Path) -> dict[str, Any]:
    workflow = _load_json(workflow_path)
    return {
        "schema_version": 1,
        "workflow_path": str(workflow_path),
        "node_count": len(workflow.get("nodes", [])),
        "link_count": len(workflow.get("links", [])),
        "field_inventory": _field_inventory(workflow),
        "generation_run": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Patch DG prompt JSON into the known-good LTX inpainting workflow without changing topology.")
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--dg-json", "--prompt-json", dest="dg_json", type=Path)
    parser.add_argument("--output-workflow", type=Path, default=DEFAULT_OUTPUT_WORKFLOW)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video", default=None)
    parser.add_argument("--reference-image", default=None)
    parser.add_argument("--inspect", nargs="?", const="", default=None, metavar="WORKFLOW")
    args = parser.parse_args(argv)
    if args.inspect is not None:
        workflow_path = Path(args.inspect) if args.inspect else args.workflow
        print(_json_dumps(inspect_workflow(workflow_path)))
        return 0
    if args.dg_json is None:
        parser.error("--dg-json is required unless --inspect is used")
    manifest = build_patched_workflow(
        args.workflow,
        args.dg_json,
        args.output_workflow,
        args.manifest,
        video_path=args.video,
        reference_image_path=args.reference_image,
    )
    print(_json_dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
