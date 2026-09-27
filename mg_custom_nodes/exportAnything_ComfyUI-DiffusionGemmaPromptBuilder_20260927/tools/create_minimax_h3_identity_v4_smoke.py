#!/usr/bin/env python3
"""Create an isolated five-second smoke sibling from the verified V4 graph.

The source is never overwritten.  The smoke graph changes exactly four
execution widgets, keeps relay Off, receives a fresh workflow UUID, and is
written byte-identically to Desktop and ComfyUI's user workflow directory.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import uuid
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_MIGRATION_PATH = REPO_ROOT / "tools" / "migrate_minimax_h3_identity_relay.py"
SPEC = importlib.util.spec_from_file_location(
    "minimax_h3_identity_relay_base", BASE_MIGRATION_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot load base migration: {BASE_MIGRATION_PATH}")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

SMOKE_SCHEMA = "diffusiongemma.minimax_h3_identity_relay_smoke"
SMOKE_VERSION = 1
SOURCE_V4_SHA256 = (
    "fe4261c2dd4fb16df6acf05d8735beb274a8e2d80821936dcda0271d2978d6fa"
)
SOURCE_V4 = Path(
    r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_DUAL_IDENTITY_P3_RELAY_GUARDED_V4_20260822.json"
)
DEFAULT_DESKTOP_OUTPUT = Path(
    r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_DUAL_IDENTITY_V4_SMOKE_5S_20260822.json"
)
DEFAULT_USER_OUTPUT = Path(
    r"C:\ComfyUI\app\user\default\workflows\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_DUAL_IDENTITY_V4_SMOKE_5S_20260822.json"
)

SMOKE_DURATION_SECONDS = 5.0
SMOKE_TARGET_WIDGET = "1"
SMOKE_RESOLUTION_SCALE = 0.2
SMOKE_OUTPUT_PREFIX = "minimax_h3/smoke_dual_identity_v4_5s"

ALLOWED_WIDGET_EDITS: dict[int, dict[int, Any]] = {
    178: {0: SMOKE_DURATION_SECONDS},
    673: {4: SMOKE_TARGET_WIDGET},
    187: {1: SMOKE_RESOLUTION_SCALE},
    651: {0: SMOKE_OUTPUT_PREFIX},
}


def _node(workflow: dict[str, Any], node_id: int) -> dict[str, Any]:
    matches = [
        item
        for item in workflow.get("nodes", [])
        if int(item.get("id", -1)) == int(node_id)
    ]
    if len(matches) != 1:
        raise BASE.WorkflowError(f"Expected one main node {node_id}; found {len(matches)}.")
    return matches[0]


def _require_source(source: dict[str, Any], source_hash: str) -> None:
    if source_hash.lower() != SOURCE_V4_SHA256:
        raise BASE.WorkflowError(
            f"Smoke source must be verified V4 SHA256 {SOURCE_V4_SHA256}; observed {source_hash}."
        )
    if SMOKE_SCHEMA in source.get("extra", {}):
        raise BASE.WorkflowError("Smoke source is already a smoke artifact.")
    BASE.validate_workflow(source)
    secondary = _node(source, 721)
    if (secondary.get("widgets_values") or [None])[0] != BASE.SECONDARY_IDENTITY_RELATIVE_PATH:
        raise BASE.WorkflowError("Verified V4 does not use the root-level Picture 2 basename.")
    if Path(BASE.SECONDARY_IDENTITY_RELATIVE_PATH).parent != Path("."):
        raise BASE.WorkflowError("Picture 2 must remain at the ComfyUI input root.")


def create_smoke_workflow(
    source: dict[str, Any],
    *,
    source_file_sha256: str,
    source_path: Path | None = None,
) -> dict[str, Any]:
    _require_source(source, source_file_sha256)
    smoke = copy.deepcopy(source)

    for node_id, edits in ALLOWED_WIDGET_EDITS.items():
        node = _node(smoke, node_id)
        widgets = list(node.get("widgets_values") or [])
        for index, value in edits.items():
            if index >= len(widgets):
                raise BASE.WorkflowError(
                    f"Smoke node {node_id} has no widget index {index}."
                )
            widgets[index] = value
        node["widgets_values"] = widgets

    source_canonical = BASE._canonical_hash(source)
    fresh_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{SMOKE_SCHEMA}/v{SMOKE_VERSION}/{source_canonical}",
        )
    )
    if fresh_id == str(source.get("id", "")):
        raise BASE.WorkflowError("Smoke workflow UUID did not diverge from V4.")
    smoke["id"] = fresh_id
    smoke["revision"] = int(source.get("revision", 0) or 0) + 1

    marker = {
        "schema": SMOKE_SCHEMA,
        "version": SMOKE_VERSION,
        "source_workflow_path": str(source_path.resolve()) if source_path else "",
        "source_workflow_file_sha256": source_file_sha256.lower(),
        "source_workflow_canonical_sha256": source_canonical,
        "source_workflow_id": str(source.get("id", "")),
        "smoke_workflow_id": fresh_id,
        "duration_seconds": SMOKE_DURATION_SECONDS,
        "target_profile_widget_4": SMOKE_TARGET_WIDGET,
        "target_profile_widget_5_preserved": 15,
        "target_profile_widget_11_preserved": 0,
        "resolution_scale": SMOKE_RESOLUTION_SCALE,
        "output_prefix": SMOKE_OUTPUT_PREFIX,
        "continuity_relay_mode": "Off",
        "exact_widget_edits": {
            "178.widget_0": SMOKE_DURATION_SECONDS,
            "673.widget_4": SMOKE_TARGET_WIDGET,
            "187.widget_1": SMOKE_RESOLUTION_SCALE,
            "651.widget_0": SMOKE_OUTPUT_PREFIX,
        },
        "source_is_never_overwritten": True,
        "generation_was_not_queued_by_creator": True,
    }
    smoke.setdefault("extra", {})[SMOKE_SCHEMA] = marker
    smoke["extra"]["workflow_note"] = (
        "Isolated five-second V4 dual-identity smoke artifact. It preserves the verified V4 topology, "
        "root-level Picture 2 asset, identity-first relay-Off policy, and b430 execution path while "
        "changing only duration, target widget 4, resolution scale, and output prefix."
    )

    validate_smoke_workflow(smoke, source=source)
    return smoke


def validate_smoke_workflow(
    smoke: dict[str, Any],
    *,
    source: dict[str, Any],
) -> None:
    BASE.validate_workflow(smoke)
    marker = smoke.get("extra", {}).get(SMOKE_SCHEMA)
    if not isinstance(marker, dict) or int(marker.get("version", 0)) != SMOKE_VERSION:
        raise BASE.WorkflowError("Smoke provenance marker is missing or unsupported.")
    if marker.get("source_workflow_file_sha256") != SOURCE_V4_SHA256:
        raise BASE.WorkflowError("Smoke marker does not identify the verified V4 source hash.")
    try:
        uuid.UUID(str(smoke.get("id", "")))
    except ValueError as exc:
        raise BASE.WorkflowError("Smoke workflow id is not a UUID.") from exc
    if str(smoke.get("id")) == str(source.get("id")):
        raise BASE.WorkflowError("Smoke workflow must have an independent UUID.")
    if int(smoke.get("revision", 0)) != int(source.get("revision", 0)) + 1:
        raise BASE.WorkflowError("Smoke workflow revision is not exactly one after V4.")

    if smoke.get("links") != source.get("links"):
        raise BASE.WorkflowError("Smoke creation changed main-graph links.")
    if smoke.get("definitions") != source.get("definitions"):
        raise BASE.WorkflowError("Smoke creation changed subgraphs or runtime topology.")
    for key in ("last_node_id", "last_link_id", "groups", "config", "version"):
        if smoke.get(key) != source.get(key):
            raise BASE.WorkflowError(f"Smoke creation changed protected workflow field {key!r}.")

    source_nodes = {int(item["id"]): item for item in source["nodes"]}
    smoke_nodes = {int(item["id"]): item for item in smoke["nodes"]}
    if source_nodes.keys() != smoke_nodes.keys():
        raise BASE.WorkflowError("Smoke creation added or removed nodes.")
    for node_id, source_node in source_nodes.items():
        comparable = copy.deepcopy(smoke_nodes[node_id])
        if node_id in ALLOWED_WIDGET_EDITS:
            widgets = list(comparable.get("widgets_values") or [])
            for index in ALLOWED_WIDGET_EDITS[node_id]:
                widgets[index] = source_node["widgets_values"][index]
            comparable["widgets_values"] = widgets
        if comparable != source_node:
            raise BASE.WorkflowError(
                f"Smoke node {node_id} changed outside the authorized widget edits."
            )

    if _node(smoke, 178)["widgets_values"][0] != SMOKE_DURATION_SECONDS:
        raise BASE.WorkflowError("Smoke master duration is not 5.0 seconds.")
    target_widgets = _node(smoke, 673)["widgets_values"]
    if target_widgets[4] != SMOKE_TARGET_WIDGET:
        raise BASE.WorkflowError("Smoke target widget 4 is not the requested string '1'.")
    if target_widgets[5] != 15 or target_widgets[11] != 0:
        raise BASE.WorkflowError("Smoke target widgets 5 or 11 changed unexpectedly.")
    if _node(smoke, 187)["widgets_values"][1] != SMOKE_RESOLUTION_SCALE:
        raise BASE.WorkflowError("Smoke resolution scale is not 0.2.")
    if _node(smoke, 651)["widgets_values"][0] != SMOKE_OUTPUT_PREFIX:
        raise BASE.WorkflowError("Smoke output prefix is incorrect.")
    if _node(smoke, 677)["widgets_values"][-1] != "Off":
        raise BASE.WorkflowError("Smoke relay must remain Off.")
    if _node(smoke, 721)["widgets_values"][0] != BASE.SECONDARY_IDENTITY_RELATIVE_PATH:
        raise BASE.WorkflowError("Smoke Picture 2 path is not the root-level guarded basename.")

    source_extra = copy.deepcopy(source.get("extra", {}))
    smoke_extra = copy.deepcopy(smoke.get("extra", {}))
    smoke_extra.pop(SMOKE_SCHEMA, None)
    source_extra.pop("workflow_note", None)
    smoke_extra.pop("workflow_note", None)
    if smoke_extra != source_extra:
        raise BASE.WorkflowError("Smoke changed non-smoke provenance metadata.")


def _preflight_outputs(paths: tuple[Path, ...]) -> None:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise BASE.WorkflowError("Smoke output paths must be distinct.")
    existing = [str(path) for path in resolved if path.exists()]
    if existing:
        raise FileExistsError(
            "Smoke output already exists; refusing overwrite: " + ", ".join(existing)
        )


def write_smoke_copies(
    smoke: dict[str, Any],
    *,
    desktop_output: Path,
    user_output: Path,
) -> tuple[str, str]:
    _preflight_outputs((desktop_output, user_output))
    desktop_hash = BASE._exclusive_json_write(desktop_output, smoke)
    user_hash = BASE._exclusive_json_write(user_output, smoke)
    if desktop_hash != user_hash:
        raise BASE.WorkflowError("Desktop and user smoke copies are not byte-identical.")
    if desktop_output.resolve().read_bytes() != user_output.resolve().read_bytes():
        raise BASE.WorkflowError("Desktop and user smoke bytes differ despite matching hashes.")
    return desktop_hash, user_hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_V4)
    parser.add_argument("--desktop-output", type=Path, default=DEFAULT_DESKTOP_OUTPUT)
    parser.add_argument("--user-output", type=Path, default=DEFAULT_USER_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    source, source_hash = BASE.load_workflow(args.source)
    smoke = create_smoke_workflow(
        source,
        source_file_sha256=source_hash,
        source_path=args.source,
    )
    result_hash = BASE._file_hash(BASE._json_bytes(smoke))
    canonical_hash = BASE._canonical_hash(smoke)
    if args.check:
        print(
            f"validated smoke in memory source_sha256={source_hash} "
            f"output_sha256={result_hash} canonical_sha256={canonical_hash} "
            f"workflow_id={smoke['id']}"
        )
        return 0

    desktop_hash, user_hash = write_smoke_copies(
        smoke,
        desktop_output=args.desktop_output,
        user_output=args.user_output,
    )
    print(
        f"created desktop={args.desktop_output.resolve()} user={args.user_output.resolve()} "
        f"sha256={desktop_hash} canonical_sha256={canonical_hash} "
        f"workflow_id={smoke['id']}"
    )
    if desktop_hash != result_hash or user_hash != result_hash:
        raise BASE.WorkflowError("Written smoke hash changed after validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
