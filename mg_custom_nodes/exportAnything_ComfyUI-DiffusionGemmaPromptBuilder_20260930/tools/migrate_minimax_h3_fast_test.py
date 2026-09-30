#!/usr/bin/env python3
"""Create a non-destructive MiniMax-H3 Ref2VA Turbo fast-test sibling.

The source workflow is never rewritten.  The tool fails closed when the H3
runtime topology is ambiguous, applies the official four-step Turbo contract,
adds a conditioning-side VRAM cleanup seam to every lazy H3 lane, validates the
result, and writes a new sibling with exclusive-create semantics.

The fast-test defaults (five seconds and 0.4 megapixels) remain ordinary user
controls.  Reopening the untouched source workflow is the rollback path; its
exact SHA-256 is persisted in the migrated workflow.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


MIGRATION_SCHEMA = "diffusiongemma.minimax_h3_fast_test"
MIGRATION_VERSION = 1

DEFAULT_SOURCE = Path(
    r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_LIVE_UNSAVED_CAPTURE_20260822_0057.json"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_BASELINE_FAST_H3.json"
)

SOURCE_CAPTURE_SHA256 = (
    "c4dc698ebd05d07c8422333c3e4a498e52e47270207231b6067d7fca33fb30e0"
)
FAST_MODEL = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
FAST_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
FAST_LORA_SHA256 = (
    "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c"
)
FAST_LORA_BYTES = 1_956_193_000
FAST_SAMPLER = "euler"
FAST_SCHEDULER = "simple"
FAST_STEPS = 4
FAST_VIDEO_SHIFT = 12.0
FAST_AUDIO_SHIFT = 3.0
FAST_DURATION_SECONDS = 5.0
FAST_RESOLUTION_MEGAPIXELS = 0.4
LANE_COUNT = 4
CLEANUP_NODE_TYPE = "FL_UnloadAllModels"
CLEANUP_INPUT = "value"
CLEANUP_OUTPUT = "*"
CLEANUP_BACKEND_OUTPUT_NODE = False


class WorkflowError(RuntimeError):
    """Raised when a workflow cannot be changed without guessing."""


def _file_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _file_hash(payload)


def _sensitive_key_paths(value: Any, prefix: str = "") -> list[str]:
    hits: list[str] = []
    names = {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "password",
        "secret",
        "client_secret",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).casefold() in names and item not in (None, "", False):
                hits.append(path)
            hits.extend(_sensitive_key_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(_sensitive_key_paths(item, f"{prefix}[{index}]"))
    return hits


def _subgraphs(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    definitions = workflow.get("definitions")
    if not isinstance(definitions, dict):
        return []
    values = definitions.get("subgraphs")
    return values if isinstance(values, list) else []


def _all_nodes(workflow: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield from workflow.get("nodes", [])
    for subgraph in _subgraphs(workflow):
        yield from subgraph.get("nodes", [])


class IdAllocator:
    def __init__(self, workflow: dict[str, Any]) -> None:
        node_ids = [int(node["id"]) for node in _all_nodes(workflow)]
        link_ids = [int(link[0]) for link in workflow.get("links", [])]
        for subgraph in _subgraphs(workflow):
            link_ids.extend(int(link["id"]) for link in subgraph.get("links", []))
            state = subgraph.get("state") or {}
            node_ids.append(int(state.get("lastNodeId", 0) or 0))
            link_ids.append(int(state.get("lastLinkId", 0) or 0))
        node_ids.append(int(workflow.get("last_node_id", 0) or 0))
        link_ids.append(int(workflow.get("last_link_id", 0) or 0))
        self._node = max(node_ids or [0])
        self._link = max(link_ids or [0])

    def node(self) -> int:
        self._node += 1
        return self._node

    def link(self) -> int:
        self._link += 1
        return self._link

    @property
    def last_node(self) -> int:
        return self._node

    @property
    def last_link(self) -> int:
        return self._link


def _socket(node: dict[str, Any], direction: str, name: str) -> int:
    matches = [
        index
        for index, item in enumerate(node.get(direction, []))
        if item.get("name") == name
    ]
    if len(matches) != 1:
        raise WorkflowError(
            f"Node {node.get('id')} ({node.get('type')}) requires one "
            f"{direction[:-1]} {name!r}; found {len(matches)}."
        )
    return matches[0]


def _one(values: Iterable[dict[str, Any]], predicate, label: str) -> dict[str, Any]:
    matches = [value for value in values if predicate(value)]
    if len(matches) != 1:
        raise WorkflowError(f"Expected one {label}; found {len(matches)}.")
    return matches[0]


class MainGraph:
    def __init__(self, workflow: dict[str, Any], ids: IdAllocator) -> None:
        self.workflow = workflow
        self.nodes: list[dict[str, Any]] = workflow["nodes"]
        self.links: list[list[Any]] = workflow["links"]
        self.ids = ids

    def node(self, node_id: int) -> dict[str, Any]:
        return _one(
            self.nodes,
            lambda node: int(node["id"]) == int(node_id),
            f"main-graph node {node_id}",
        )

    def input_origin(
        self, target: dict[str, Any], input_name: str
    ) -> tuple[dict[str, Any], int]:
        input_slot = _socket(target, "inputs", input_name)
        link_id = target["inputs"][input_slot].get("link")
        if link_id is None:
            raise WorkflowError(
                f"Node {target.get('id')} input {input_name!r} is unconnected."
            )
        link = _one(
            self.links,
            lambda item: int(item[0]) == int(link_id),
            f"main-graph link {link_id}",
        )
        return self.node(int(link[1])), int(link[2])

    def remove_link(self, link_id: int) -> None:
        matches = [link for link in self.links if int(link[0]) == int(link_id)]
        if not matches:
            return
        if len(matches) != 1:
            raise WorkflowError(f"Duplicate main-graph link id {link_id}.")
        link = matches[0]
        _, origin_id, origin_slot, target_id, target_slot, _ = link
        origin = self.node(int(origin_id))
        target = self.node(int(target_id))
        output = origin["outputs"][int(origin_slot)]
        output["links"] = [
            value
            for value in (output.get("links") or [])
            if int(value) != int(link_id)
        ]
        target_input = target["inputs"][int(target_slot)]
        if target_input.get("link") is not None and int(target_input["link"]) == int(
            link_id
        ):
            target_input["link"] = None
        self.links.remove(link)

    def disconnect(self, node: dict[str, Any]) -> None:
        node_id = int(node["id"])
        for link in list(self.links):
            if int(link[1]) == node_id or int(link[3]) == node_id:
                self.remove_link(int(link[0]))

    def add(self, node: dict[str, Any]) -> dict[str, Any]:
        if any(int(value["id"]) == int(node["id"]) for value in _all_nodes(self.workflow)):
            raise WorkflowError(f"Duplicate node id {node['id']}.")
        node["order"] = max(
            (int(value.get("order", 0)) for value in self.nodes), default=0
        ) + 1
        self.nodes.append(node)
        return node

    def connect(
        self,
        origin: dict[str, Any],
        output_name: str,
        target: dict[str, Any],
        input_name: str,
        link_type: str,
    ) -> int:
        output_slot = _socket(origin, "outputs", output_name)
        input_slot = _socket(target, "inputs", input_name)
        existing = target["inputs"][input_slot].get("link")
        if existing is not None:
            self.remove_link(int(existing))
        link_id = self.ids.link()
        self.links.append(
            [
                link_id,
                int(origin["id"]),
                output_slot,
                int(target["id"]),
                input_slot,
                link_type,
            ]
        )
        target["inputs"][input_slot]["link"] = link_id
        output = origin["outputs"][output_slot]
        output.setdefault("links", []).append(link_id)
        return link_id


class SubgraphEditor:
    def __init__(self, subgraph: dict[str, Any], ids: IdAllocator) -> None:
        self.subgraph = subgraph
        self.nodes: list[dict[str, Any]] = subgraph["nodes"]
        self.links: list[dict[str, Any]] = subgraph["links"]
        self.ids = ids

    def node(self, node_id: int) -> dict[str, Any]:
        return _one(
            self.nodes,
            lambda node: int(node["id"]) == int(node_id),
            f"subgraph node {node_id}",
        )

    def input_origin(
        self, target: dict[str, Any], input_name: str
    ) -> tuple[dict[str, Any], int]:
        slot = _socket(target, "inputs", input_name)
        link_id = target["inputs"][slot].get("link")
        if link_id is None:
            raise WorkflowError(
                f"Subgraph node {target.get('id')} input {input_name!r} is unconnected."
            )
        link = _one(
            self.links,
            lambda value: int(value["id"]) == int(link_id),
            f"subgraph link {link_id}",
        )
        origin_id = int(link["origin_id"])
        if origin_id < 0:
            raise WorkflowError(
                f"Subgraph node {target.get('id')} input {input_name!r} comes from "
                "the subgraph boundary, not a node."
            )
        return self.node(origin_id), int(link["origin_slot"])

    def remove_link(self, link_id: int) -> None:
        matches = [link for link in self.links if int(link["id"]) == int(link_id)]
        if not matches:
            return
        if len(matches) != 1:
            raise WorkflowError(f"Duplicate subgraph link id {link_id}.")
        link = matches[0]
        origin_id = int(link["origin_id"])
        target_id = int(link["target_id"])
        if origin_id < 0 or target_id < 0:
            raise WorkflowError(
                "The fast migration never rewires subgraph boundary links."
            )
        origin = self.node(origin_id)
        target = self.node(target_id)
        output = origin["outputs"][int(link["origin_slot"])]
        output["links"] = [
            value
            for value in (output.get("links") or [])
            if int(value) != int(link_id)
        ]
        target_input = target["inputs"][int(link["target_slot"])]
        if target_input.get("link") is not None and int(target_input["link"]) == int(
            link_id
        ):
            target_input["link"] = None
        self.links.remove(link)

    def add(self, node: dict[str, Any]) -> dict[str, Any]:
        if any(int(value["id"]) == int(node["id"]) for value in self.nodes):
            raise WorkflowError(f"Duplicate subgraph node id {node['id']}.")
        node["order"] = max(
            (int(value.get("order", 0)) for value in self.nodes), default=0
        ) + 1
        self.nodes.append(node)
        return node

    def connect(
        self,
        origin: dict[str, Any],
        output_name: str,
        target: dict[str, Any],
        input_name: str,
        link_type: str,
    ) -> int:
        output_slot = _socket(origin, "outputs", output_name)
        input_slot = _socket(target, "inputs", input_name)
        existing = target["inputs"][input_slot].get("link")
        if existing is not None:
            self.remove_link(int(existing))
        link_id = self.ids.link()
        self.links.append(
            {
                "id": link_id,
                "origin_id": int(origin["id"]),
                "origin_slot": output_slot,
                "target_id": int(target["id"]),
                "target_slot": input_slot,
                "type": link_type,
            }
        )
        target["inputs"][input_slot]["link"] = link_id
        origin["outputs"][output_slot].setdefault("links", []).append(link_id)
        return link_id


def _input(name: str, type_name: str, *, widget: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "localized_name": name,
        "name": name,
        "type": type_name,
        "link": None,
    }
    if widget:
        result["widget"] = {"name": name}
    return result


def _output(name: str, type_name: str) -> dict[str, Any]:
    return {
        "localized_name": name,
        "name": name,
        "type": type_name,
        "links": [],
    }


def _sigma_shift_node(ids: IdAllocator, pos: list[float]) -> dict[str, Any]:
    return {
        "id": ids.node(),
        "type": "MiniMaxH3SigmaShift",
        "pos": [float(pos[0]), float(pos[1])],
        "size": [270.0, 82.0],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            _input("model", "MODEL"),
            _input("shift_video", "FLOAT", widget=True),
            _input("shift_audio", "FLOAT", widget=True),
        ],
        "outputs": [_output("MODEL", "MODEL")],
        "title": "H3 Turbo sigma shifts — video 12 / audio 3",
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.30.0",
            "Node name for S&R": "MiniMaxH3SigmaShift",
        },
        "widgets_values": [FAST_VIDEO_SHIFT, FAST_AUDIO_SHIFT],
        "color": "#100a73",
        "bgcolor": "#252b3e",
    }


def _cleanup_node(ids: IdAllocator, lane: int, pos: list[float]) -> dict[str, Any]:
    return {
        "id": ids.node(),
        "type": CLEANUP_NODE_TYPE,
        "pos": [float(pos[0]), float(pos[1])],
        "size": [285.0, 58.0],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [_input(CLEANUP_INPUT, "*")],
        "outputs": [_output(CLEANUP_OUTPUT, "*")],
        "title": (
            f"H3 LANE {lane} — release encoder/VAEs after conditioning"
        ),
        "properties": {
            "Node name for S&R": CLEANUP_NODE_TYPE,
            "cnr_id": "comfyui_fill-nodes",
            "ver": "2.9.2",
        },
        "widgets_values": [],
    }


def _find_h3_subgraph(workflow: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for subgraph in _subgraphs(workflow):
        nodes = subgraph.get("nodes", [])
        if (
            sum(node.get("type") == "MiniMaxH3ReferenceToVideo" for node in nodes)
            == LANE_COUNT
            and sum(node.get("type") == "BasicScheduler" for node in nodes) == 1
            and sum(node.get("type") == "KSamplerSelect" for node in nodes) == 1
        ):
            candidates.append(subgraph)
    if len(candidates) != 1:
        raise WorkflowError(
            f"Expected one four-lane H3 sampling subgraph; found {len(candidates)}."
        )
    return candidates[0]


def _main_nodes(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    values = workflow.get("nodes")
    if not isinstance(values, list):
        raise WorkflowError("The workflow has no editable main graph.")
    if not isinstance(workflow.get("links"), list):
        raise WorkflowError("The workflow has no editable main-graph links.")
    return values


def _find_runtime(
    workflow: dict[str, Any], ids: IdAllocator
) -> dict[str, Any]:
    nodes = _main_nodes(workflow)
    graph = MainGraph(workflow, ids)
    subgraph = _find_h3_subgraph(workflow)
    subeditor = SubgraphEditor(subgraph, ids)
    outer = _one(
        nodes,
        lambda node: str(node.get("type")) == str(subgraph.get("id")),
        "main-graph H3 subgraph node",
    )
    loader = _one(
        nodes,
        lambda node: node.get("type") == "UNETLoader"
        and "minimax_h3_ref2va" in str((node.get("widgets_values") or [""])[0]),
        "MiniMax-H3 Ref2VA UNET loader",
    )
    lora = _one(
        nodes,
        lambda node: node.get("type") == "LoraLoaderModelOnly"
        and "minimax_h3_ref2v_turbo" in str(
            (node.get("widgets_values") or [""])[0]
        ),
        "MiniMax-H3 Turbo LoRA loader",
    )
    sage = _one(
        nodes,
        lambda node: node.get("type") == "PathchSageAttentionKJ",
        "KJ SageAttention patch",
    )
    cache = _one(
        nodes,
        lambda node: node.get("type") == "EasyCache",
        "EasyCache experiment",
    )
    compile_node = _one(
        nodes,
        lambda node: node.get("type") == "TorchCompileModel",
        "TorchCompile experiment",
    )
    sigma_nodes = [
        node for node in nodes if node.get("type") == "MiniMaxH3SigmaShift"
    ]
    sampler = _one(
        subeditor.nodes,
        lambda node: node.get("type") == "KSamplerSelect",
        "H3 KSamplerSelect",
    )
    scheduler = _one(
        subeditor.nodes,
        lambda node: node.get("type") == "BasicScheduler",
        "H3 BasicScheduler",
    )
    h3_nodes = sorted(
        [
            node
            for node in subeditor.nodes
            if node.get("type") == "MiniMaxH3ReferenceToVideo"
        ],
        key=lambda node: float((node.get("pos") or [0.0, 0.0])[1]),
    )
    guiders = sorted(
        [node for node in subeditor.nodes if node.get("type") == "BasicGuider"],
        key=lambda node: float((node.get("pos") or [0.0, 0.0])[1]),
    )
    samplers = sorted(
        [
            node
            for node in subeditor.nodes
            if node.get("type") == "SamplerCustomAdvanced"
        ],
        key=lambda node: float((node.get("pos") or [0.0, 0.0])[1]),
    )
    if not (len(h3_nodes) == len(guiders) == len(samplers) == LANE_COUNT):
        raise WorkflowError(
            "The H3 subgraph must contain four reference nodes, guiders, and samplers."
        )
    return {
        "graph": graph,
        "subgraph": subgraph,
        "subeditor": subeditor,
        "outer": outer,
        "loader": loader,
        "lora": lora,
        "sage": sage,
        "cache": cache,
        "compile": compile_node,
        "sigma_nodes": sigma_nodes,
        "sampler": sampler,
        "scheduler": scheduler,
        "h3_nodes": h3_nodes,
        "guiders": guiders,
        "samplers": samplers,
    }


def _require_source_chain(runtime: dict[str, Any]) -> None:
    graph: MainGraph = runtime["graph"]
    expected = (
        (runtime["lora"], "model", runtime["loader"]),
        (runtime["sage"], "model", runtime["lora"]),
        (runtime["cache"], "model", runtime["sage"]),
        (runtime["compile"], "model", runtime["cache"]),
        (runtime["outer"], "model", runtime["compile"]),
    )
    for target, input_name, wanted_origin in expected:
        origin, _ = graph.input_origin(target, input_name)
        if int(origin["id"]) != int(wanted_origin["id"]):
            raise WorkflowError(
                "The live H3 acceleration chain changed after capture; refusing "
                f"to guess at node {target.get('id')} input {input_name!r}."
            )
    if runtime["sigma_nodes"]:
        raise WorkflowError(
            "An unmarked MiniMaxH3SigmaShift already exists; refusing to guess "
            "whether a partial fast migration should be replaced."
        )
    subeditor: SubgraphEditor = runtime["subeditor"]
    for lane, (h3, guider, sampler) in enumerate(
        zip(runtime["h3_nodes"], runtime["guiders"], runtime["samplers"]), start=1
    ):
        origin, slot = subeditor.input_origin(guider, "conditioning")
        if int(origin["id"]) != int(h3["id"]) or slot != _socket(
            h3, "outputs", "positive"
        ):
            raise WorkflowError(
                f"H3 lane {lane} conditioning is no longer a direct reference-node output."
            )
        origin, slot = subeditor.input_origin(sampler, "latent_image")
        if int(origin["id"]) != int(h3["id"]) or slot != _socket(
            h3, "outputs", "LATENT"
        ):
            raise WorkflowError(
                f"H3 lane {lane} latent path is no longer direct and cannot be patched safely."
            )


def _set_widget(node: dict[str, Any], index: int, value: Any, label: str) -> None:
    widgets = node.get("widgets_values")
    if not isinstance(widgets, list) or len(widgets) <= index:
        raise WorkflowError(
            f"Node {node.get('id')} is missing the expected {label} widget."
        )
    widgets[index] = value


def _apply_fast_defaults(workflow: dict[str, Any]) -> None:
    nodes = _main_nodes(workflow)
    duration = _one(
        nodes,
        lambda node: int(node.get("id", -1)) == 178
        and node.get("type") == "PrimitiveFloat",
        "master video duration control",
    )
    splitter = _one(
        nodes,
        lambda node: int(node.get("id", -1)) == 187
        and node.get("type") == "DiffusionGemmaJSONSplitter",
        "H3 resolution control",
    )
    _set_widget(duration, 0, FAST_DURATION_SECONDS, "duration")
    _set_widget(splitter, 1, FAST_RESOLUTION_MEGAPIXELS, "megapixel")
    duration["title"] = (
        "2. FAST TEST VIDEO duration — 5s default; configurable for production"
    )
    splitter["title"] = (
        "7. Route H3 prompt + resolution — FAST TEST defaults to 0.4 MP"
    )

    project = [
        node
        for node in nodes
        if node.get("type") == "DiffusionGemmaProjectMasterContract"
    ]
    if len(project) == 1:
        values = project[0].get("widgets_values")
        if isinstance(values, list) and len(values) >= 4:
            values[0] = FAST_DURATION_SECONDS
            values[3] = FAST_DURATION_SECONDS


def migrate_workflow(
    workflow: dict[str, Any], *, source_file_sha256: str | None = None
) -> dict[str, Any]:
    """Return a migrated deep copy without modifying the source object."""

    if not isinstance(workflow, dict):
        raise WorkflowError("Workflow root must be a JSON object.")
    sensitive = _sensitive_key_paths(workflow)
    if sensitive:
        raise WorkflowError(
            "Refusing to process non-empty sensitive-looking fields at "
            + ", ".join(sensitive)
            + "."
        )
    result = copy.deepcopy(workflow)
    marker = result.get("extra", {}).get(MIGRATION_SCHEMA)
    if marker is not None:
        if not isinstance(marker, dict) or int(marker.get("version", 0)) != MIGRATION_VERSION:
            raise WorkflowError("The MiniMax-H3 fast-test marker is unsupported.")
        validate_workflow(result)
        return result

    _validate_link_integrity(result)
    ids = IdAllocator(result)
    runtime = _find_runtime(result, ids)
    _require_source_chain(runtime)
    graph: MainGraph = runtime["graph"]
    subeditor: SubgraphEditor = runtime["subeditor"]

    _set_widget(runtime["loader"], 0, FAST_MODEL, "H3 model")
    runtime["loader"]["title"] = "MiniMax-H3 Ref2VA — pruned INT8 fast-test model"
    _set_widget(runtime["lora"], 0, FAST_LORA, "Turbo LoRA")
    _set_widget(runtime["lora"], 1, 1.0, "Turbo LoRA strength")
    runtime["lora"]["title"] = "Official Ref2VA Turbo 4-step LoRA — ComfyUI format"
    runtime["sage"]["widgets_values"] = ["auto", False]
    runtime["sage"]["title"] = "SageAttention auto — compile disabled"

    graph.disconnect(runtime["cache"])
    graph.disconnect(runtime["compile"])
    for node, title, y in (
        (
            runtime["cache"],
            "EXPERIMENT DISABLED — EasyCache is bypassed for four-step Turbo",
            -2890.0,
        ),
        (
            runtime["compile"],
            "EXPERIMENT DISABLED — Torch compile is outside the fast baseline",
            -2890.0,
        ),
    ):
        node["mode"] = 4
        node["title"] = title
        if isinstance(node.get("pos"), list) and len(node["pos"]) >= 2:
            node["pos"][1] = y

    sigma = graph.add(
        _sigma_shift_node(ids, [4246.0, -2668.0])
    )
    graph.connect(runtime["sage"], "MODEL", sigma, "model", "MODEL")
    graph.connect(sigma, "MODEL", runtime["outer"], "model", "MODEL")

    runtime["sampler"]["widgets_values"] = [FAST_SAMPLER]
    runtime["sampler"]["title"] = "Shared H3 sampler — Euler Turbo baseline"
    runtime["scheduler"]["widgets_values"] = [
        FAST_SCHEDULER,
        FAST_STEPS,
        1.0,
    ]
    runtime["scheduler"]["title"] = "Shared H3 schedule — simple / 4-step Turbo"
    runtime["subgraph"]["name"] = "MiniMax H3 Ref2VA — FAST 4-step"
    runtime["outer"]["title"] = "FAST H3 Ref2VA — pruned INT8 / Turbo 4-step"

    cleanup_nodes: list[dict[str, Any]] = []
    for lane, (h3, guider) in enumerate(
        zip(runtime["h3_nodes"], runtime["guiders"]), start=1
    ):
        h3_pos = h3.get("pos") or [4198.0, float(lane - 1) * 970.0]
        cleanup = subeditor.add(
            _cleanup_node(
                ids,
                lane,
                [4198.0, float(h3_pos[1]) - 70.0],
            )
        )
        subeditor.connect(
            h3, "positive", cleanup, CLEANUP_INPUT, "CONDITIONING"
        )
        subeditor.connect(
            cleanup, CLEANUP_OUTPUT, guider, "conditioning", "CONDITIONING"
        )
        cleanup_nodes.append(cleanup)

    _apply_fast_defaults(result)

    result["last_node_id"] = ids.last_node
    result["last_link_id"] = ids.last_link
    result["revision"] = int(result.get("revision", 0) or 0) + 1
    for subgraph in _subgraphs(result):
        state = subgraph.setdefault("state", {})
        state["lastNodeId"] = ids.last_node
        state["lastLinkId"] = ids.last_link
    runtime["subgraph"]["revision"] = int(
        runtime["subgraph"].get("revision", 0) or 0
    ) + 1

    extra = result.setdefault("extra", {})
    extra[MIGRATION_SCHEMA] = {
        "version": MIGRATION_VERSION,
        "profile": "ref2va_turbo_4step_fast_test",
        "source": {
            "file_sha256": source_file_sha256,
            "workflow_id": workflow.get("id"),
            "workflow_revision": workflow.get("revision"),
            "rollback_policy": "reopen the unchanged source capture",
        },
        "assets": {
            "diffusion_model": {
                "filename": FAST_MODEL,
                "folder": "diffusion_models",
                "format": "pruned_int8_convrot",
                "required": True,
            },
            "turbo_lora": {
                "filename": FAST_LORA,
                "folder": "loras",
                "format": "comfyui_bf16",
                "required": True,
                "expected_bytes": FAST_LORA_BYTES,
                "expected_sha256": FAST_LORA_SHA256,
            },
        },
        "sampling_contract": {
            "sampler": FAST_SAMPLER,
            "scheduler": FAST_SCHEDULER,
            "steps": FAST_STEPS,
            "denoise": 1.0,
            "shift_video": FAST_VIDEO_SHIFT,
            "shift_audio": FAST_AUDIO_SHIFT,
            "sage_attention": "auto",
            "sage_allow_compile": False,
            "easycache_active": False,
            "torch_compile_active": False,
        },
        "benchmark_defaults": {
            "duration_seconds": FAST_DURATION_SECONDS,
            "resolution_megapixels": FAST_RESOLUTION_MEGAPIXELS,
            "controls_remain_configurable": True,
        },
        "generation_semantics": {
            "lazy_generation_lane_capacity": LANE_COUNT,
            "project_and_native_shot_contract_preserved": True,
            "conditioning_cleanup_per_lane": True,
            "cleanup_node_type": CLEANUP_NODE_TYPE,
            "cleanup_backend_output_node": CLEANUP_BACKEND_OUTPUT_NODE,
            "latent_path_unchanged": True,
        },
        "nodes": {
            "loader": int(runtime["loader"]["id"]),
            "lora": int(runtime["lora"]["id"]),
            "sage": int(runtime["sage"]["id"]),
            "sigma_shift": int(sigma["id"]),
            "disabled_easycache": int(runtime["cache"]["id"]),
            "disabled_torch_compile": int(runtime["compile"]["id"]),
            "h3_subgraph": str(runtime["subgraph"]["id"]),
            "h3_outer": int(runtime["outer"]["id"]),
            "conditioning_cleanup": [int(node["id"]) for node in cleanup_nodes],
        },
    }
    validate_workflow(result, require_benchmark_defaults=True)
    sensitive_after = _sensitive_key_paths(result)
    if sensitive_after:
        raise WorkflowError(
            "Migration introduced non-empty sensitive-looking fields at "
            + ", ".join(sensitive_after)
            + "."
        )
    return result


def _validate_main_links(workflow: dict[str, Any]) -> None:
    nodes = {int(node["id"]): node for node in _main_nodes(workflow)}
    link_ids: set[int] = set()
    expected_outputs: dict[tuple[int, int], list[int]] = {}
    expected_inputs: dict[tuple[int, int], int] = {}
    for link in workflow["links"]:
        if not isinstance(link, list) or len(link) != 6:
            raise WorkflowError("Malformed main-graph link.")
        link_id, origin_id, origin_slot, target_id, target_slot, _ = link
        link_id = int(link_id)
        origin_id = int(origin_id)
        target_id = int(target_id)
        origin_slot = int(origin_slot)
        target_slot = int(target_slot)
        if link_id in link_ids:
            raise WorkflowError(f"Duplicate main-graph link id {link_id}.")
        link_ids.add(link_id)
        if origin_id not in nodes or target_id not in nodes:
            raise WorkflowError(f"Main-graph link {link_id} references a missing node.")
        if origin_slot >= len(nodes[origin_id].get("outputs", [])):
            raise WorkflowError(f"Main-graph link {link_id} has an invalid origin slot.")
        if target_slot >= len(nodes[target_id].get("inputs", [])):
            raise WorkflowError(f"Main-graph link {link_id} has an invalid target slot.")
        key = (target_id, target_slot)
        if key in expected_inputs:
            raise WorkflowError(f"Main-graph input {key} has multiple incoming links.")
        expected_inputs[key] = link_id
        expected_outputs.setdefault((origin_id, origin_slot), []).append(link_id)
    for node_id, node in nodes.items():
        for slot, item in enumerate(node.get("inputs", [])):
            actual = item.get("link")
            expected = expected_inputs.get((node_id, slot))
            if actual != expected:
                raise WorkflowError(
                    f"Main-graph input backlink mismatch at node {node_id} slot {slot}."
                )
        for slot, item in enumerate(node.get("outputs", [])):
            actual = sorted(int(value) for value in (item.get("links") or []))
            expected = sorted(expected_outputs.get((node_id, slot), []))
            if actual != expected:
                raise WorkflowError(
                    f"Main-graph output backlink mismatch at node {node_id} slot {slot}."
                )


def _validate_subgraph_links(subgraph: dict[str, Any]) -> None:
    nodes = {int(node["id"]): node for node in subgraph.get("nodes", [])}
    input_node_id = int((subgraph.get("inputNode") or {}).get("id", -10))
    output_node_id = int((subgraph.get("outputNode") or {}).get("id", -20))
    expected_node_inputs: dict[tuple[int, int], int] = {}
    expected_node_outputs: dict[tuple[int, int], list[int]] = {}
    expected_boundary_inputs: dict[int, list[int]] = {}
    expected_boundary_outputs: dict[int, list[int]] = {}
    seen: set[int] = set()
    for link in subgraph.get("links", []):
        link_id = int(link["id"])
        if link_id in seen:
            raise WorkflowError(f"Duplicate subgraph link id {link_id}.")
        seen.add(link_id)
        origin_id = int(link["origin_id"])
        target_id = int(link["target_id"])
        origin_slot = int(link["origin_slot"])
        target_slot = int(link["target_slot"])
        if origin_id == input_node_id:
            if origin_slot >= len(subgraph.get("inputs", [])):
                raise WorkflowError(f"Subgraph link {link_id} has an invalid boundary input.")
            expected_boundary_inputs.setdefault(origin_slot, []).append(link_id)
        elif origin_id in nodes:
            if origin_slot >= len(nodes[origin_id].get("outputs", [])):
                raise WorkflowError(f"Subgraph link {link_id} has an invalid origin slot.")
            expected_node_outputs.setdefault((origin_id, origin_slot), []).append(link_id)
        else:
            raise WorkflowError(f"Subgraph link {link_id} has a missing origin.")
        if target_id == output_node_id:
            if target_slot >= len(subgraph.get("outputs", [])):
                raise WorkflowError(f"Subgraph link {link_id} has an invalid boundary output.")
            expected_boundary_outputs.setdefault(target_slot, []).append(link_id)
        elif target_id in nodes:
            if target_slot >= len(nodes[target_id].get("inputs", [])):
                raise WorkflowError(f"Subgraph link {link_id} has an invalid target slot.")
            key = (target_id, target_slot)
            if key in expected_node_inputs:
                raise WorkflowError(f"Subgraph input {key} has multiple incoming links.")
            expected_node_inputs[key] = link_id
        else:
            raise WorkflowError(f"Subgraph link {link_id} has a missing target.")
    for node_id, node in nodes.items():
        for slot, item in enumerate(node.get("inputs", [])):
            if item.get("link") != expected_node_inputs.get((node_id, slot)):
                raise WorkflowError(
                    f"Subgraph input backlink mismatch at node {node_id} slot {slot}."
                )
        for slot, item in enumerate(node.get("outputs", [])):
            actual = sorted(int(value) for value in (item.get("links") or []))
            expected = sorted(expected_node_outputs.get((node_id, slot), []))
            if actual != expected:
                raise WorkflowError(
                    f"Subgraph output backlink mismatch at node {node_id} slot {slot}."
                )
    for slot, item in enumerate(subgraph.get("inputs", [])):
        actual = sorted(int(value) for value in (item.get("linkIds") or []))
        expected = sorted(expected_boundary_inputs.get(slot, []))
        if actual != expected:
            raise WorkflowError(f"Subgraph boundary input backlink mismatch at slot {slot}.")
    for slot, item in enumerate(subgraph.get("outputs", [])):
        actual = sorted(int(value) for value in (item.get("linkIds") or []))
        expected = sorted(expected_boundary_outputs.get(slot, []))
        if actual != expected:
            raise WorkflowError(f"Subgraph boundary output backlink mismatch at slot {slot}.")


def _validate_link_integrity(workflow: dict[str, Any]) -> None:
    _validate_main_links(workflow)
    all_node_ids = [int(node["id"]) for node in _all_nodes(workflow)]
    if len(all_node_ids) != len(set(all_node_ids)):
        raise WorkflowError("Node ids must be unique across the workflow and subgraphs.")
    all_link_ids = [int(link[0]) for link in workflow.get("links", [])]
    for subgraph in _subgraphs(workflow):
        _validate_subgraph_links(subgraph)
        all_link_ids.extend(int(link["id"]) for link in subgraph.get("links", []))
    if len(all_link_ids) != len(set(all_link_ids)):
        raise WorkflowError("Link ids must be unique across the workflow and subgraphs.")


def _assert_origin(
    editor: MainGraph | SubgraphEditor,
    target: dict[str, Any],
    input_name: str,
    origin: dict[str, Any],
    output_name: str,
) -> None:
    actual, actual_slot = editor.input_origin(target, input_name)
    if int(actual["id"]) != int(origin["id"]) or actual_slot != _socket(
        origin, "outputs", output_name
    ):
        raise WorkflowError(
            f"Node {target.get('id')} input {input_name!r} has the wrong origin."
        )


def validate_workflow(
    workflow: dict[str, Any], *, require_benchmark_defaults: bool = False
) -> None:
    """Validate the complete fast-test runtime and rollback metadata."""

    _validate_link_integrity(workflow)
    sensitive = _sensitive_key_paths(workflow)
    if sensitive:
        raise WorkflowError(
            "Workflow contains non-empty sensitive-looking fields at "
            + ", ".join(sensitive)
            + "."
        )
    marker = workflow.get("extra", {}).get(MIGRATION_SCHEMA)
    if not isinstance(marker, dict) or int(marker.get("version", 0)) != MIGRATION_VERSION:
        raise WorkflowError("MiniMax-H3 fast-test marker is missing or stale.")

    ids = IdAllocator(workflow)
    runtime = _find_runtime(workflow, ids)
    nodes_marker = marker.get("nodes") or {}
    if int(nodes_marker.get("loader", -1)) != int(runtime["loader"]["id"]):
        raise WorkflowError("Fast-test loader marker does not match the graph.")
    if int(nodes_marker.get("lora", -1)) != int(runtime["lora"]["id"]):
        raise WorkflowError("Fast-test LoRA marker does not match the graph.")
    if int(nodes_marker.get("sage", -1)) != int(runtime["sage"]["id"]):
        raise WorkflowError("Fast-test Sage marker does not match the graph.")
    sigma = _one(
        runtime["sigma_nodes"],
        lambda _node: True,
        "active MiniMaxH3SigmaShift",
    )
    if int(nodes_marker.get("sigma_shift", -1)) != int(sigma["id"]):
        raise WorkflowError("Fast-test sigma-shift marker does not match the graph.")

    if (runtime["loader"].get("widgets_values") or [None])[0] != FAST_MODEL:
        raise WorkflowError("Fast-test workflow does not select the pruned INT8 model.")
    if runtime["lora"].get("widgets_values") != [FAST_LORA, 1.0]:
        raise WorkflowError("Fast-test workflow does not select the official ComfyUI LoRA.")
    if runtime["sage"].get("widgets_values") != ["auto", False]:
        raise WorkflowError("SageAttention must be Auto with compilation disabled.")
    if sigma.get("widgets_values") != [FAST_VIDEO_SHIFT, FAST_AUDIO_SHIFT]:
        raise WorkflowError("MiniMax-H3 sigma shifts must be video 12 / audio 3.")
    if runtime["sampler"].get("widgets_values") != [FAST_SAMPLER]:
        raise WorkflowError("The fast-test sampler must be Euler.")
    if runtime["scheduler"].get("widgets_values") != [
        FAST_SCHEDULER,
        FAST_STEPS,
        1.0,
    ]:
        raise WorkflowError("The fast-test schedule must be simple / four steps.")

    graph: MainGraph = runtime["graph"]
    _assert_origin(graph, runtime["lora"], "model", runtime["loader"], "MODEL")
    _assert_origin(graph, runtime["sage"], "model", runtime["lora"], "MODEL")
    _assert_origin(graph, sigma, "model", runtime["sage"], "MODEL")
    _assert_origin(graph, runtime["outer"], "model", sigma, "MODEL")
    for disabled in (runtime["cache"], runtime["compile"]):
        if int(disabled.get("mode", 0)) != 4:
            raise WorkflowError(f"Disabled experiment node {disabled.get('id')} is active.")
        disabled_id = int(disabled["id"])
        if any(
            int(link[1]) == disabled_id or int(link[3]) == disabled_id
            for link in workflow["links"]
        ):
            raise WorkflowError(
                f"Disabled experiment node {disabled_id} remains in the model path."
            )

    cleanup_ids = [int(value) for value in nodes_marker.get("conditioning_cleanup", [])]
    if len(cleanup_ids) != LANE_COUNT or len(set(cleanup_ids)) != LANE_COUNT:
        raise WorkflowError("The fast-test marker must name four cleanup seams.")
    cleanups = [runtime["subeditor"].node(node_id) for node_id in cleanup_ids]
    if any(node.get("type") != CLEANUP_NODE_TYPE for node in cleanups):
        raise WorkflowError("A conditioning cleanup marker references the wrong node type.")
    if CLEANUP_BACKEND_OUTPUT_NODE:
        raise WorkflowError("The configured cleanup seam must not be an execution root.")
    subeditor: SubgraphEditor = runtime["subeditor"]
    for lane, (h3, cleanup, guider, sampler) in enumerate(
        zip(
            runtime["h3_nodes"],
            cleanups,
            runtime["guiders"],
            runtime["samplers"],
        ),
        start=1,
    ):
        _assert_origin(subeditor, cleanup, CLEANUP_INPUT, h3, "positive")
        _assert_origin(subeditor, guider, "conditioning", cleanup, CLEANUP_OUTPUT)
        _assert_origin(subeditor, sampler, "latent_image", h3, "LATENT")
        cleanup_output = cleanup["outputs"][
            _socket(cleanup, "outputs", CLEANUP_OUTPUT)
        ]
        if len(cleanup_output.get("links") or []) != 1:
            raise WorkflowError(
                f"H3 lane {lane} cleanup must feed only its matching guider."
            )

    semantics = marker.get("generation_semantics") or {}
    if semantics != {
        "lazy_generation_lane_capacity": LANE_COUNT,
        "project_and_native_shot_contract_preserved": True,
        "conditioning_cleanup_per_lane": True,
        "cleanup_node_type": CLEANUP_NODE_TYPE,
        "cleanup_backend_output_node": CLEANUP_BACKEND_OUTPUT_NODE,
        "latent_path_unchanged": True,
    }:
        raise WorkflowError("Fast-test generation-semantics metadata is stale.")
    source = marker.get("source") or {}
    if not source.get("file_sha256"):
        raise WorkflowError("Fast-test rollback source SHA-256 is missing.")
    if source.get("rollback_policy") != "reopen the unchanged source capture":
        raise WorkflowError("Fast-test rollback policy is missing or stale.")

    if require_benchmark_defaults:
        duration = _one(
            _main_nodes(workflow),
            lambda node: int(node.get("id", -1)) == 178,
            "duration control",
        )
        splitter = _one(
            _main_nodes(workflow),
            lambda node: int(node.get("id", -1)) == 187,
            "resolution control",
        )
        if (duration.get("widgets_values") or [None])[0] != FAST_DURATION_SECONDS:
            raise WorkflowError("The initial fast benchmark must default to five seconds.")
        if (splitter.get("widgets_values") or [None, None])[1] != FAST_RESOLUTION_MEGAPIXELS:
            raise WorkflowError("The initial fast benchmark must default to 0.4 MP.")


def _exclusive_json_write(path: Path, workflow: dict[str, Any]) -> None:
    """Create a new JSON file and refuse to overwrite any existing sibling."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(workflow, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise WorkflowError(
            f"Refusing to overwrite existing fast sibling: {path}"
        ) from exc
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if created and path.exists() and path.stat().st_size != len(payload):
            path.unlink()
        raise


def migrate_file(
    source_path: Path,
    *,
    output_path: Path,
    write: bool,
    expected_source_sha256: str | None = None,
) -> tuple[dict[str, Any], str, Path | None]:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if source_path == output_path:
        raise WorkflowError("Fast-test output must be a new sibling, never the source.")
    if not source_path.is_file():
        raise WorkflowError(f"Workflow does not exist: {source_path}")
    source_bytes = source_path.read_bytes()
    source_hash = _file_hash(source_bytes)
    if expected_source_sha256 and source_hash.casefold() != expected_source_sha256.casefold():
        raise WorkflowError(
            f"Source SHA-256 mismatch: {source_hash} != {expected_source_sha256}."
        )
    try:
        source = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Workflow is not valid UTF-8 JSON: {source_path}") from exc
    migrated = migrate_workflow(source, source_file_sha256=source_hash)
    validate_workflow(migrated, require_benchmark_defaults=True)
    written: Path | None = None
    if write:
        if _file_hash(source_path.read_bytes()) != source_hash:
            raise WorkflowError("Source changed during migration; no sibling was written.")
        if output_path.exists():
            existing_bytes = output_path.read_bytes()
            try:
                existing = json.loads(existing_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WorkflowError(
                    f"Refusing to overwrite invalid existing sibling: {output_path}"
                ) from exc
            validate_workflow(existing)
            if existing != migrated:
                raise WorkflowError(
                    f"Refusing to overwrite user-edited fast sibling: {output_path}"
                )
        else:
            _exclusive_json_write(output_path, migrated)
            written = output_path
        if _file_hash(source_path.read_bytes()) != source_hash:
            raise WorkflowError("Source changed while writing the sibling.")
    return migrated, source_hash, written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Migrate and validate in memory without writing a sibling.",
    )
    parser.add_argument(
        "--expected-source-sha256",
        default=SOURCE_CAPTURE_SHA256,
        help="Exact source guard; defaults to the captured unsaved workflow hash.",
    )
    args = parser.parse_args(argv)
    migrated, source_hash, written = migrate_file(
        args.source,
        output_path=args.output,
        write=not args.check,
        expected_source_sha256=args.expected_source_sha256,
    )
    action = "checked" if args.check else ("created" if written else "unchanged")
    print(
        f"{action} source={args.source.resolve()} source_sha256={source_hash} "
        f"result_sha256={_canonical_hash(migrated)} output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
