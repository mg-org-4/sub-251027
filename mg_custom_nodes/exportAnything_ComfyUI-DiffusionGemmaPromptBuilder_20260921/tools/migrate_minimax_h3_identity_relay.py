#!/usr/bin/env python3
"""Create a guarded dual-identity MiniMax-H3 sibling from a proven workflow.

The migration is intentionally narrow:

* the source workflow is never overwritten;
* Picture 1 and Picture 2 become clean, persistent identity anchors;
* the prior-lane tail moves to a gated Picture 3 socket;
* the audio-aware planner owns the relay toggle and defaults this sibling to
  identity-first (relay Off);
* the ACE song path, executed H3 sampling path, latent path, and final audio mux
  remain unchanged.

The CLI can extract the exact frontend workflow embedded in a completed
ComfyUI history prompt before producing the migrated sibling.  Both writes use
exclusive creation, so an existing capture or sibling is never replaced.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


SOURCE_EXPANSION_SCHEMA = "diffusiongemma.minimax_music_video_expansion"
IDENTITY_RELAY_SCHEMA = "diffusiongemma.minimax_h3_identity_relay"
IDENTITY_RELAY_VERSION = 1
SOURCE_EXPANSION_VERSION = 2
FAST_SUBGRAPH_ID = "7e490d92-ff82-4226-8969-3deddf16b717"

PLANNER_TYPE = "DiffusionGemmaAudioAwareMultiShotPlanner"
CONTEXT_TYPE = "DiffusionGemmaH3ReferenceContext"
PAIR_PREP_TYPE = "DiffusionGemmaH3ReferencePairPrep"
RELAY_GATE_TYPE = "DiffusionGemmaH3RelayReferenceGate"
NATIVE_H3_TYPE = "MiniMaxH3ReferenceToVideo"

DEFAULT_BASE_URL = "http://127.0.0.1:8188"
DEFAULT_HISTORY_PROMPT_ID = "b43059cc-e1db-4a51-8d2d-5ab95519e271"
DEFAULT_CAPTURE = Path(
    r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_EXECUTED_423S_CAPTURE_b43059cc_20260822.json"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_DUAL_IDENTITY_P3_RELAY_GUARDED_V4_20260822.json"
)
SECONDARY_IDENTITY_RELATIVE_PATH = "img_221_identity_sheet_gpt-image-2_v1.png"
SECONDARY_IDENTITY_SHA256 = (
    "e95f6c119cf4889d5dd83670fbe413b8c9467d9eee9added8ffa863f84195945"
)
DEFAULT_COMFY_INPUT_ROOT = Path(__file__).resolve().parents[3] / "input"

DUAL_IDENTITY_MANIFEST = (
    "<Picture 1>: [dg:identity,appearance,object,color,composition] primary identity, body, wardrobe, "
    "and target-composition authority for the same woman.\n"
    "<Picture 2>: [dg:identity,appearance] multi-panel identity evidence for the same woman only; "
    "do not transfer its layout, grid, seams, backgrounds, or pose sequence, and its panels are not "
    "separate people.\n"
    "<Audio 1>: [dg:audio,rhythm] locked soundtrack timing reference."
)

RELAY_REFERENCE_MEBIPIXELS = 0.064
RELAY_REFERENCE_RESOLUTION_STEPS = 16

START_NOTE = """# DiffusionGemma + ACE-Step + MiniMax-H3 — dual identity / guarded relay

1. **Picture 1** is the primary full-body/wardrobe reference. **Picture 2** is the generated multi-view identity sheet for that same sole performer. If either input is replaced, keep Picture 2 as a same-person multi-panel sheet rather than a second character. The two pictures jointly define one semantic Subject; they are not two people.
2. **Identity picture count** is fixed to 2 for this sibling. The shared manifest is the single contract source for both Director validation and generation-lane planning.
3. **Continuity relay** defaults to **Off (identity-first)**. Every lane therefore starts from the same two clean identity anchors and may execute independently. Select **Previous lane tail** only when boundary pose/spatial continuity matters more than the risk of propagating a degraded face.
4. When relay is enabled, the immediately previous retained tail is routed lazily as **Picture 3** for lanes 2–4. Picture 1 and Picture 2 remain the identity authorities; Picture 3 is opening continuity evidence only.
5. The two clean references share one downscale-only reference-pixel budget: Picture 1 receives 40% and the multi-panel Picture 2 receives 60%. When enabled, Picture 3 is capped near 0.068 megapixels (roughly 192×352 for a 544×960 source) before entering H3. Native H3 stays on `ref_image_size=match`; every model, LoRA-strength, sampler, scheduler, sigma-shift, Sage, audio-slice, latent, assembly, and pristine ACE-mux setting from the successful b430 execution remains unchanged.
6. One H3 generation lane remains one model invocation of at most 15 seconds. Native `[Shot N]` blocks remain inside the planned lanes, and the Project Master remains authoritative for duration and lane ceiling.
7. Leave Director cache on **reuse**. **refresh** is a one-run diagnostic and must immediately return to **reuse**.

The dependable path is: two clean identity anchors → shared strict manifest → measured-audio Director → dual-identity lane plan → optional lazy Picture 3 relay → unchanged FAST H3 samplers → exact-frame assembly → pristine-song mux.
"""


class WorkflowError(RuntimeError):
    """Raised when the workflow cannot be migrated without guessing."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


def _file_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _file_hash(payload)


def _require_secondary_identity_asset(
    input_root: Path = DEFAULT_COMFY_INPUT_ROOT,
) -> Path:
    asset = input_root / Path(SECONDARY_IDENTITY_RELATIVE_PATH)
    if not asset.is_file():
        raise WorkflowError(
            f"Generated Picture 2 identity sheet is missing: {asset.resolve()}"
        )
    observed = _file_hash(asset.read_bytes())
    if observed.lower() != SECONDARY_IDENTITY_SHA256:
        raise WorkflowError(
            "Generated Picture 2 identity sheet SHA256 changed: "
            f"expected {SECONDARY_IDENTITY_SHA256}, observed {observed}."
        )
    return asset.resolve()


def _exclusive_json_write(path: Path, value: Any) -> str:
    """Create one JSON artifact without ever replacing an existing path."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    written = path.read_bytes()
    if written != payload:
        raise WorkflowError(f"Exclusive write verification failed for {path}.")
    return _file_hash(written)


def fetch_history_workflow(
    prompt_id: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/history/{prompt_id}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Could not read ComfyUI history prompt {prompt_id}: {exc}") from exc
    entry = payload.get(prompt_id)
    if not isinstance(entry, dict):
        raise WorkflowError(f"ComfyUI history has no prompt {prompt_id}.")
    prompt = entry.get("prompt")
    try:
        workflow = prompt[3]["extra_pnginfo"]["workflow"]
    except (IndexError, KeyError, TypeError) as exc:
        raise WorkflowError(
            f"History prompt {prompt_id} has no embedded extra_pnginfo.workflow."
        ) from exc
    if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list):
        raise WorkflowError(f"History prompt {prompt_id} contains an invalid workflow object.")
    return copy.deepcopy(workflow)


def capture_history_workflow(
    prompt_id: str,
    capture_path: Path,
    *,
    base_url: str = DEFAULT_BASE_URL,
) -> tuple[dict[str, Any], str, str]:
    workflow = fetch_history_workflow(prompt_id, base_url=base_url)
    file_hash = _exclusive_json_write(capture_path, workflow)
    return workflow, file_hash, _canonical_hash(workflow)


class IdAllocator:
    def __init__(self, workflow: dict[str, Any]) -> None:
        nodes = [int(node["id"]) for node in workflow.get("nodes", [])]
        links = [int(link[0]) for link in workflow.get("links", [])]
        for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
            nodes.extend(int(node["id"]) for node in subgraph.get("nodes", []))
            links.extend(int(link["id"]) for link in subgraph.get("links", []))
        self.last_node = max(nodes + [int(workflow.get("last_node_id", 0) or 0)])
        self.last_link = max(links + [int(workflow.get("last_link_id", 0) or 0)])

    def node(self) -> int:
        self.last_node += 1
        return self.last_node

    def link(self) -> int:
        self.last_link += 1
        return self.last_link


def _node_by_id(nodes: list[dict[str, Any]], node_id: int) -> dict[str, Any]:
    matches = [node for node in nodes if int(node.get("id", -1)) == int(node_id)]
    if len(matches) != 1:
        raise WorkflowError(f"Expected one node {node_id}; found {len(matches)}.")
    return matches[0]


def _one(nodes: list[dict[str, Any]], node_type: str) -> dict[str, Any]:
    matches = [node for node in nodes if node.get("type") == node_type]
    if len(matches) != 1:
        raise WorkflowError(f"Expected one {node_type}; found {len(matches)}.")
    return matches[0]


def _input_slot(node: dict[str, Any], name: str) -> int:
    matches = [
        index for index, item in enumerate(node.get("inputs", []))
        if item.get("name") == name
    ]
    if len(matches) != 1:
        raise WorkflowError(
            f"Node {node.get('id')} ({node.get('type')}) requires one input {name!r}; found {len(matches)}."
        )
    return matches[0]


def _output_slot(node: dict[str, Any], name: str) -> int:
    matches = [
        index for index, item in enumerate(node.get("outputs", []))
        if item.get("name") == name
    ]
    if len(matches) != 1:
        raise WorkflowError(
            f"Node {node.get('id')} ({node.get('type')}) requires one output {name!r}; found {len(matches)}."
        )
    return matches[0]


class MainGraph:
    def __init__(self, workflow: dict[str, Any], ids: IdAllocator) -> None:
        self.workflow = workflow
        self.nodes: list[dict[str, Any]] = workflow["nodes"]
        self.links: list[list[Any]] = workflow["links"]
        self.ids = ids

    def node(self, node_id: int) -> dict[str, Any]:
        return _node_by_id(self.nodes, node_id)

    def add(self, node: dict[str, Any]) -> dict[str, Any]:
        if any(int(item["id"]) == int(node["id"]) for item in self.nodes):
            raise WorkflowError(f"Duplicate main-graph node id {node['id']}.")
        node["order"] = max((int(item.get("order", 0)) for item in self.nodes), default=0) + 1
        self.nodes.append(node)
        return node

    def link(self, link_id: int) -> list[Any]:
        matches = [link for link in self.links if int(link[0]) == int(link_id)]
        if len(matches) != 1:
            raise WorkflowError(f"Expected one main link {link_id}; found {len(matches)}.")
        return matches[0]

    def remove_link(self, link_id: int) -> None:
        matches = [link for link in self.links if int(link[0]) == int(link_id)]
        if not matches:
            return
        if len(matches) != 1:
            raise WorkflowError(f"Duplicate main link id {link_id}.")
        link = matches[0]
        _, origin_id, origin_slot, target_id, target_slot, _link_type = link
        origin = self.node(int(origin_id))
        target = self.node(int(target_id))
        origin_output = origin["outputs"][int(origin_slot)]
        origin_output["links"] = [
            item for item in (origin_output.get("links") or [])
            if int(item) != int(link_id)
        ]
        target_input = target["inputs"][int(target_slot)]
        if target_input.get("link") == link_id:
            target_input["link"] = None
        self.links.remove(link)

    def connect(
        self,
        origin: dict[str, Any],
        output_name: str,
        target: dict[str, Any],
        input_name: str,
        link_type: str | None = None,
    ) -> int:
        output_slot = _output_slot(origin, output_name)
        input_slot = _input_slot(target, input_name)
        existing = target["inputs"][input_slot].get("link")
        if existing is not None:
            self.remove_link(int(existing))
        resolved_type = str(
            link_type
            or target["inputs"][input_slot].get("type")
            or origin["outputs"][output_slot].get("type")
        )
        link_id = self.ids.link()
        self.links.append(
            [link_id, int(origin["id"]), output_slot, int(target["id"]), input_slot, resolved_type]
        )
        target["inputs"][input_slot]["link"] = link_id
        output_links = origin["outputs"][output_slot].get("links")
        if not isinstance(output_links, list):
            output_links = []
            origin["outputs"][output_slot]["links"] = output_links
        output_links.append(link_id)
        return link_id


class SubgraphGraph:
    def __init__(self, subgraph: dict[str, Any], ids: IdAllocator) -> None:
        self.subgraph = subgraph
        self.nodes: list[dict[str, Any]] = subgraph["nodes"]
        self.links: list[dict[str, Any]] = subgraph["links"]
        self.inputs: list[dict[str, Any]] = subgraph["inputs"]
        self.ids = ids
        self.input_node_id = int((subgraph.get("inputNode") or {}).get("id", -10))

    def node(self, node_id: int) -> dict[str, Any]:
        return _node_by_id(self.nodes, node_id)

    def add(self, node: dict[str, Any]) -> dict[str, Any]:
        if any(int(item["id"]) == int(node["id"]) for item in self.nodes):
            raise WorkflowError(f"Duplicate subgraph node id {node['id']}.")
        node["order"] = max((int(item.get("order", 0)) for item in self.nodes), default=0) + 1
        self.nodes.append(node)
        return node

    def remove_link(self, link_id: int) -> None:
        matches = [link for link in self.links if int(link["id"]) == int(link_id)]
        if not matches:
            return
        if len(matches) != 1:
            raise WorkflowError(f"Duplicate subgraph link id {link_id}.")
        link = matches[0]
        target = self.node(int(link["target_id"]))
        target_input = target["inputs"][int(link["target_slot"])]
        if target_input.get("link") == link_id:
            target_input["link"] = None
        if int(link["origin_id"]) == self.input_node_id:
            source_input = self.inputs[int(link["origin_slot"])]
            source_input["linkIds"] = [
                item for item in (source_input.get("linkIds") or [])
                if int(item) != int(link_id)
            ]
        else:
            origin = self.node(int(link["origin_id"]))
            origin_output = origin["outputs"][int(link["origin_slot"])]
            origin_output["links"] = [
                item for item in (origin_output.get("links") or [])
                if int(item) != int(link_id)
            ]
        self.links.remove(link)

    def connect(
        self,
        origin: dict[str, Any],
        output_name: str,
        target: dict[str, Any],
        input_name: str,
        link_type: str | None = None,
    ) -> int:
        output_slot = _output_slot(origin, output_name)
        input_slot = _input_slot(target, input_name)
        existing = target["inputs"][input_slot].get("link")
        if existing is not None:
            self.remove_link(int(existing))
        resolved_type = str(
            link_type
            or target["inputs"][input_slot].get("type")
            or origin["outputs"][output_slot].get("type")
        )
        link_id = self.ids.link()
        self.links.append(
            {
                "id": link_id,
                "origin_id": int(origin["id"]),
                "origin_slot": output_slot,
                "target_id": int(target["id"]),
                "target_slot": input_slot,
                "type": resolved_type,
            }
        )
        target["inputs"][input_slot]["link"] = link_id
        output_links = origin["outputs"][output_slot].get("links")
        if not isinstance(output_links, list):
            output_links = []
            origin["outputs"][output_slot]["links"] = output_links
        output_links.append(link_id)
        return link_id

    def append_boundary_input(
        self,
        *,
        name: str,
        type_name: str,
        label: str | None = None,
        optional: bool = False,
    ) -> int:
        index = len(self.inputs)
        input_node = self.subgraph["inputNode"]
        x = float(input_node["bounding"][0]) + 104.0
        y = float(input_node["bounding"][1]) + 24.0 + 20.0 * index
        item: dict[str, Any] = {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{FAST_SUBGRAPH_ID}/input/{name}")),
            "name": name,
            "type": type_name,
            "linkIds": [],
            "localized_name": name,
            "pos": [x, y],
        }
        if label:
            item["label"] = label
        if optional:
            item["shape"] = 7
        self.inputs.append(item)
        input_node["bounding"][3] = max(float(input_node["bounding"][3]), 28.0 + 20.0 * len(self.inputs))
        return index

    def connect_boundary(
        self,
        boundary_slot: int,
        target: dict[str, Any],
        input_name: str,
        link_type: str,
    ) -> int:
        input_slot = _input_slot(target, input_name)
        existing = target["inputs"][input_slot].get("link")
        if existing is not None:
            self.remove_link(int(existing))
        link_id = self.ids.link()
        self.links.append(
            {
                "id": link_id,
                "origin_id": self.input_node_id,
                "origin_slot": int(boundary_slot),
                "target_id": int(target["id"]),
                "target_slot": input_slot,
                "type": link_type,
            }
        )
        target["inputs"][input_slot]["link"] = link_id
        self.inputs[int(boundary_slot)].setdefault("linkIds", []).append(link_id)
        return link_id


def _input(name: str, type_name: str, *, optional: bool = False, widget: bool = False) -> dict[str, Any]:
    item: dict[str, Any] = {"name": name, "type": type_name, "link": None}
    if optional:
        item["shape"] = 7
    if widget:
        item["widget"] = {"name": name}
    return item


def _output(name: str, type_name: str) -> dict[str, Any]:
    return {"name": name, "type": type_name, "links": []}


def _types_compatible(left: Any, right: Any) -> bool:
    """Accept Comfy wildcard and comma-union socket types."""

    left_types = {item.strip() for item in str(left).split(",") if item.strip()}
    right_types = {item.strip() for item in str(right).split(",") if item.strip()}
    return "*" in left_types or "*" in right_types or bool(left_types & right_types)


def _node(
    ids: IdAllocator,
    node_type: str,
    title: str,
    pos: tuple[float, float],
    size: tuple[float, float],
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    widgets: list[Any] | None = None,
    *,
    cnr_id: str = "diffusiongemma-prompt-builder",
) -> dict[str, Any]:
    return {
        "id": ids.node(),
        "type": node_type,
        "pos": [float(pos[0]), float(pos[1])],
        "size": [float(size[0]), float(size[1])],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "title": title,
        "properties": {"cnr_id": cnr_id, "Node name for S&R": node_type},
        "widgets_values": list(widgets or []),
    }


def _batch_node(ids: IdAllocator) -> dict[str, Any]:
    node = _node(
        ids,
        "BatchImagesNode",
        "Director analysis batch — Picture 1 then Picture 2",
        (-1655.0, 1375.0),
        (620.0, 150.0),
        [
            {"label": "image0", "localized_name": "images.image0", "name": "images.image0", "type": "IMAGE", "link": None},
            {"label": "image1", "localized_name": "images.image1", "name": "images.image1", "type": "IMAGE", "link": None},
        ],
        [{"localized_name": "IMAGE", "name": "IMAGE", "type": "IMAGE", "links": []}],
        [],
        cnr_id="comfy-core",
    )
    node["widgets_values"] = None
    return node


def _pair_prep_node(ids: IdAllocator) -> dict[str, Any]:
    return _node(
        ids,
        PAIR_PREP_TYPE,
        "H3 dual identity budget — Pictures 1+2 share one frame area",
        (-965.0, 1375.0),
        (700.0, 330.0),
        [
            _input("reference_image_1", "IMAGE"),
            _input("reference_image_2", "IMAGE"),
            _input("generation_width", "INT", widget=True),
            _input("generation_height", "INT", widget=True),
        ],
        [
            _output("reference_image_1", "IMAGE"),
            _output("reference_image_2", "IMAGE"),
            _output("prep_metadata_json", "STRING"),
            _output("reference_1_width", "INT"),
            _output("reference_1_height", "INT"),
            _output("reference_2_width", "INT"),
            _output("reference_2_height", "INT"),
            _output("combined_reference_megapixels", "FLOAT"),
            _output("estimated_packed_reference_rows", "INT"),
        ],
        [1376, 768, 1.0, 0.40],
    )


def _manifest_node(ids: IdAllocator) -> dict[str, Any]:
    return _node(
        ids,
        "PrimitiveStringMultiline",
        "H3 REFERENCE MANIFEST — shared Director + relay contract",
        (-250.0, 1375.0),
        (820.0, 385.0),
        [],
        [_output("STRING", "STRING")],
        [DUAL_IDENTITY_MANIFEST],
        cnr_id="comfy-core",
    )


def _relay_gate_node(ids: IdAllocator, lane: int, pos: tuple[float, float]) -> dict[str, Any]:
    return _node(
        ids,
        RELAY_GATE_TYPE,
        f"H3 LANE {lane} — optional prior-tail Picture 3 gate",
        pos,
        (520.0, 190.0),
        [
            _input("shot_plan_json", "STRING"),
            _input("relay_image", "IMAGE", optional=True),
        ],
        [
            _output("relay_reference", "IMAGE"),
            _output("relay_picture_tag", "STRING"),
            _output("enabled", "BOOLEAN"),
            _output("status", "STRING"),
        ],
        [lane],
    )


def _relay_scale_node(ids: IdAllocator, lane: int, pos: tuple[float, float]) -> dict[str, Any]:
    """Bound a degraded prior-lane frame before it can influence the next lane."""

    return _node(
        ids,
        "ImageScaleToTotalPixels",
        f"H3 LANE {lane} — cap optional Picture 3 relay near 0.068 MP",
        pos,
        (520.0, 150.0),
        [_input("image", "IMAGE")],
        [_output("IMAGE", "IMAGE")],
        ["area", RELAY_REFERENCE_MEBIPIXELS, RELAY_REFERENCE_RESOLUTION_STEPS],
        cnr_id="comfy-core",
    )


def _require_source(source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if IDENTITY_RELAY_SCHEMA in source.get("extra", {}):
        raise WorkflowError("Source is already an identity-relay sibling; use the proven pre-migration capture.")
    if not isinstance(source.get("nodes"), list) or not isinstance(source.get("links"), list):
        raise WorkflowError("Source has no editable main graph.")
    marker = source.get("extra", {}).get(SOURCE_EXPANSION_SCHEMA)
    if not isinstance(marker, dict) or int(marker.get("version", 0)) != SOURCE_EXPANSION_VERSION:
        raise WorkflowError("Source is not the stabilized MiniMax music-video expansion v2.")
    expected = {
        116: "MarkdownNote",
        179: "LoadImage",
        187: "DiffusionGemmaJSONSplitter",
        674: CONTEXT_TYPE,
        677: PLANNER_TYPE,
        710: FAST_SUBGRAPH_ID,
        713: "LoraLoaderModelOnly",
        716: "MiniMaxH3SigmaShift",
    }
    for node_id, node_type in expected.items():
        node = _node_by_id(source["nodes"], node_id)
        if node.get("type") != node_type:
            raise WorkflowError(f"Source node {node_id} is {node.get('type')}, expected {node_type}.")
    subgraphs = [
        item for item in source.get("definitions", {}).get("subgraphs", [])
        if str(item.get("id")) == FAST_SUBGRAPH_ID
    ]
    if len(subgraphs) != 1:
        raise WorkflowError("The FAST H3 subgraph is missing or ambiguous.")
    subgraph = subgraphs[0]
    runtime = marker.get("runtime_lane_node_ids")
    if not isinstance(runtime, dict):
        raise WorkflowError("Source has no runtime lane marker.")
    expected_runtime = {
        "h3": [667, 684, 692, 700],
        "tails": [682, 690, 698],
        "tail_math": [665, 689, 697],
        "decoders": [656, 688, 696, 704],
        "trims": [681, 683, 691, 699],
    }
    for key, value in expected_runtime.items():
        if [int(item) for item in runtime.get(key, [])] != value:
            raise WorkflowError(f"Source runtime lane marker {key!r} changed unexpectedly.")
    h3_nodes = [_node_by_id(subgraph["nodes"], item) for item in expected_runtime["h3"]]
    if any(node.get("type") != NATIVE_H3_TYPE for node in h3_nodes):
        raise WorkflowError("Source FAST subgraph no longer uses native MiniMaxH3ReferenceToVideo nodes.")
    if str((h3_nodes[0].get("widgets_values") or [None])[-1]) != "match":
        raise WorkflowError("Source H3 reference sizing is no longer the verified match baseline.")
    sampler = _node_by_id(subgraph["nodes"], 657)
    scheduler = _node_by_id(subgraph["nodes"], 658)
    lora = _node_by_id(source["nodes"], 713)
    if sampler.get("type") != "KSamplerSelect" or sampler.get("widgets_values") != [
        "res_multistep"
    ]:
        raise WorkflowError("Source b430 sampler is no longer res_multistep.")
    if scheduler.get("type") != "BasicScheduler" or scheduler.get(
        "widgets_values"
    ) != ["simple", 8, 1]:
        raise WorkflowError("Source b430 scheduler is no longer simple / 8 steps / denoise 1.")
    if lora.get("type") != "LoraLoaderModelOnly" or float(
        (lora.get("widgets_values") or [None, -1])[1]
    ) != 0.60:
        raise WorkflowError("Source b430 LoRA strength is no longer 0.60.")
    for lane, node in enumerate(h3_nodes, start=1):
        p1 = node["inputs"][_input_slot(node, "ref_images.ref_image_0")]
        if p1.get("link") is None:
            raise WorkflowError(f"Source H3 lane {lane} has no Picture 1 input.")
        p2_slot = _input_slot(node, "ref_images.ref_image_1")
        if lane == 1 and node["inputs"][p2_slot].get("link") is not None:
            raise WorkflowError("Source lane 1 unexpectedly has a Picture 2 input.")
        if lane > 1 and node["inputs"][p2_slot].get("link") is None:
            raise WorkflowError(f"Source lane {lane} has no prior-tail Picture 2 input.")
    return marker, subgraph


def migrate_workflow(
    source: dict[str, Any],
    *,
    source_file_sha256: str = "",
    source_history_prompt_id: str = "",
) -> dict[str, Any]:
    source_marker, _source_subgraph = _require_source(source)
    secondary_asset = _require_secondary_identity_asset()
    migrated = copy.deepcopy(source)
    ids = IdAllocator(migrated)
    main = MainGraph(migrated, ids)
    marker = migrated["extra"][SOURCE_EXPANSION_SCHEMA]
    subgraph = next(
        item for item in migrated["definitions"]["subgraphs"]
        if str(item.get("id")) == FAST_SUBGRAPH_ID
    )
    inner = SubgraphGraph(subgraph, ids)
    outer = main.node(710)
    context = main.node(674)
    planner = main.node(677)
    splitter = main.node(187)
    primary = main.node(179)

    # Picture 2 is a separately generated multi-view identity sheet for the
    # exact same performer.  Its hash is guarded so a stale or substituted
    # image cannot silently change the reference contract.
    secondary = copy.deepcopy(primary)
    secondary["id"] = ids.node()
    secondary["title"] = "3b. Picture 2 — generated multi-view identity sheet (same performer)"
    secondary["pos"] = [-3590.0, 520.0]
    secondary["size"] = [1150.0, 1450.0]
    secondary["order"] = 0
    secondary_widgets = list(secondary.get("widgets_values") or [])
    if len(secondary_widgets) < 1:
        raise WorkflowError("Primary LoadImage widget contract changed unexpectedly.")
    secondary_widgets[0] = SECONDARY_IDENTITY_RELATIVE_PATH
    secondary["widgets_values"] = secondary_widgets
    for output in secondary.get("outputs", []):
        output["links"] = []
    main.add(secondary)

    batch = main.add(_batch_node(ids))
    pair = main.add(_pair_prep_node(ids))
    manifest = main.add(_manifest_node(ids))

    # The successful history graph carried stale FAST4/Euler labels even
    # though its executed widgets were res_multistep, simple/8, and LoRA 0.60.
    # Correct only those human-readable labels; widgets and links stay exact.
    lora = main.node(713)
    sampler = inner.node(657)
    scheduler = inner.node(658)
    lora["title"] = "Official Ref2VA Turbo LoRA — preserved b430 strength 0.60"
    sampler["title"] = "Shared H3 sampler — preserved b430 res_multistep"
    scheduler["title"] = "Shared H3 schedule — preserved b430 simple / 8 steps"
    subgraph["name"] = (
        "MiniMax H3 Ref2VA — preserved b430 fast path (res_multistep / simple / 8)"
    )

    primary["title"] = "3a. Picture 1 — primary full-body / wardrobe identity authority"
    main.connect(primary, "IMAGE", batch, "images.image0", "IMAGE")
    main.connect(secondary, "IMAGE", batch, "images.image1", "IMAGE")
    main.connect(primary, "IMAGE", pair, "reference_image_1", "IMAGE")
    main.connect(secondary, "IMAGE", pair, "reference_image_2", "IMAGE")
    main.connect(splitter, "resolution_width", pair, "generation_width", "INT")
    main.connect(splitter, "resolution_height", pair, "generation_height", "INT")

    # One manifest primitive feeds both the Director context and the planner.
    # Keep the old widget value as a readable fallback, but the typed link is
    # authoritative at runtime.
    context["title"] = "5a. MiniMax-H3 ordered reference context — Pictures 1+2 + Audio 1"
    context.setdefault("inputs", []).append(
        {"name": "reference_manifest", "type": "STRING", "widget": {"name": "reference_manifest"}, "link": None}
    )
    main.connect(manifest, "STRING", context, "reference_manifest", "STRING")
    main.connect(batch, "IMAGE", context, "reference_images", "IMAGE")
    context_widgets = list(context.get("widgets_values") or [])
    if len(context_widgets) < 8:
        raise WorkflowError("H3 Reference Context widget contract changed unexpectedly.")
    context_widgets[0] = DUAL_IDENTITY_MANIFEST
    context_widgets[-2] = "custom"
    context_widgets[-1] = 1
    context["widgets_values"] = context_widgets

    planner.setdefault("inputs", []).append(
        {"name": "reference_manifest", "shape": 7, "type": "STRING", "link": None}
    )
    main.connect(manifest, "STRING", planner, "reference_manifest", "STRING")
    planner_widgets = list(planner.get("widgets_values") or [])
    if len(planner_widgets) != 5:
        raise WorkflowError("Audio-aware planner legacy widget contract changed unexpectedly.")
    planner["widgets_values"] = [*planner_widgets, 2, "Off"]
    planner["title"] = "AUDIO-AWARE H3 MULTI-LANE PLAN — dual identity / optional Picture 3 relay"

    # Append outer/subgraph inputs instead of reordering the proven input list;
    # every existing slot and link therefore keeps its historical number.
    picture_2_boundary = inner.append_boundary_input(
        name="identity_picture_2",
        type_name="IMAGE",
        label="Picture 2 clean identity",
        optional=True,
    )
    plan_boundary = inner.append_boundary_input(
        name="shot_plan_json",
        type_name="STRING",
        label="identity / relay policy",
    )
    outer["inputs"].append(
        {"label": "Picture 2 clean identity", "name": "identity_picture_2", "shape": 7, "type": "IMAGE", "link": None}
    )
    outer["inputs"].append(
        {"label": "identity / relay policy", "name": "shot_plan_json", "type": "STRING", "link": None}
    )
    if isinstance(outer.get("size"), list) and len(outer["size"]) == 2:
        outer["size"][1] = float(outer["size"][1]) + 50.0
    main.connect(pair, "reference_image_1", outer, "ref_images.ref_image_0", "IMAGE")
    main.connect(pair, "reference_image_2", outer, "identity_picture_2", "IMAGE")
    main.connect(planner, "plan_json", outer, "shot_plan_json", "STRING")
    outer["title"] = (
        "FAST H3 Ref2VA — dual identity / optional P3 relay / "
        "b430 res_multistep + simple/8"
    )

    runtime = marker["runtime_lane_node_ids"]
    h3_nodes = [inner.node(int(node_id)) for node_id in runtime["h3"]]
    tails = [inner.node(int(node_id)) for node_id in runtime["tails"]]
    for lane, h3 in enumerate(h3_nodes, start=1):
        inner.connect_boundary(
            picture_2_boundary,
            h3,
            "ref_images.ref_image_1",
            "IMAGE",
        )
        h3["title"] = (
            f"H3 GENERATION LANE {lane} — Pictures 1+2 identity + Audio 1"
            if lane == 1
            else f"H3 GENERATION LANE {lane} — Pictures 1+2 identity + optional Picture 3 relay + Audio 1"
        )

    gate_positions = ((3190.0, 1120.0), (3190.0, 2090.0), (3190.0, 3060.0))
    scale_positions = ((5790.0, 390.0), (5790.0, 1330.0), (5790.0, 2300.0))
    gates: list[dict[str, Any]] = []
    relay_scalers: list[dict[str, Any]] = []
    for lane, (tail, h3, gate_pos, scale_pos) in enumerate(
        zip(tails, h3_nodes[1:], gate_positions, scale_positions), start=2
    ):
        tail["title"] = f"LANE {lane - 1} TAIL — optional continuity-only Picture 3 source"
        scaler = inner.add(_relay_scale_node(ids, lane, scale_pos))
        gate = inner.add(_relay_gate_node(ids, lane, gate_pos))
        inner.connect_boundary(plan_boundary, gate, "shot_plan_json", "STRING")
        inner.connect(tail, "IMAGE", scaler, "image", "IMAGE")
        inner.connect(scaler, "IMAGE", gate, "relay_image", "IMAGE")
        inner.connect(gate, "relay_reference", h3, "ref_images.ref_image_2", "IMAGE")
        relay_scalers.append(scaler)
        gates.append(gate)

    main.node(116)["widgets_values"] = [START_NOTE]
    main.node(116)["title"] = "START HERE — two clean identity pictures / relay Off by default"
    for group in migrated.get("groups", []):
        if int(group.get("id", -1)) == 1:
            group["title"] = "1. Inputs + two clean identity anchors"
        elif int(group.get("id", -1)) == 38:
            group["title"] = "Per-lane Audio 1 + optional prior-tail Picture 3 relay"

    marker["reference_contract"] = {
        "picture_1": "primary identity, body, wardrobe, and target-composition authority for the same sole Subject",
        "picture_2": "multi-panel identity evidence for that same Subject only; never layout, grid, seams, backgrounds, pose sequence, or separate people",
        "picture_3": "optional capped prior retained-lane tail for lanes 2–4; continuity-only and never identity authority; omitted when relay is Off",
        "audio_1": "relative per-lane sync-safe guide slice; pristine master is muxed only after assembly",
    }
    marker["identity_relay_extension"] = {
        "schema": IDENTITY_RELAY_SCHEMA,
        "version": IDENTITY_RELAY_VERSION,
        "identity_picture_count": 2,
        "default_continuity_relay_mode": "Off",
        "continuity_picture_tag_when_enabled": "<Picture 3>",
        "manifest_node": int(manifest["id"]),
        "reference_pair_prep": int(pair["id"]),
        "director_analysis_batch": int(batch["id"]),
        "secondary_identity_loader": int(secondary["id"]),
        "secondary_identity_asset": {
            "relative_input_path": SECONDARY_IDENTITY_RELATIVE_PATH,
            "sha256": SECONDARY_IDENTITY_SHA256,
            "verified_absolute_path": str(secondary_asset),
        },
        "relay_gate_nodes": [int(item["id"]) for item in gates],
        "relay_scale_nodes": [int(item["id"]) for item in relay_scalers],
        "relay_reference_cap": {
            "node_type": "ImageScaleToTotalPixels",
            "upscale_method": "area",
            "mebipixels": RELAY_REFERENCE_MEBIPIXELS,
            "approx_decimal_megapixels": 0.068,
            "resolution_steps": RELAY_REFERENCE_RESOLUTION_STEPS,
            "approx_generation_frame_area_ratio": 0.15,
        },
        "observed_executed_fast_controls": {
            "sampler_node": 657,
            "sampler": "res_multistep",
            "scheduler_node": 658,
            "scheduler": "simple",
            "steps": 8,
            "denoise": 1.0,
            "lora_node": 713,
            "lora_name": str((lora.get("widgets_values") or [""])[0]),
            "lora_strength": 0.60,
        },
    }
    migrated["extra"][IDENTITY_RELAY_SCHEMA] = {
        "version": IDENTITY_RELAY_VERSION,
        "source_workflow_id": str(source.get("id", "")),
        "source_workflow_revision": int(source.get("revision", 0) or 0),
        "source_workflow_file_sha256": source_file_sha256,
        "source_workflow_canonical_sha256": _canonical_hash(source),
        "source_history_prompt_id": source_history_prompt_id,
        "source_execution_seconds": 423.22,
        "preserved_fast_sampling_contract": copy.deepcopy(
            source.get("extra", {})
            .get("diffusiongemma.minimax_h3_fast_test", {})
            .get("sampling_contract", {})
        ),
        "identity_picture_tags": ["<Picture 1>", "<Picture 2>"],
        "picture_2_asset": {
            "relative_input_path": SECONDARY_IDENTITY_RELATIVE_PATH,
            "sha256": SECONDARY_IDENTITY_SHA256,
        },
        "continuity_picture_tag_when_enabled": "<Picture 3>",
        "default_continuity_relay_mode": "Off",
        "relay_gate_is_lazy": True,
        "relay_reference_cap_mebipixels": RELAY_REFERENCE_MEBIPIXELS,
        "relay_reference_resolution_steps": RELAY_REFERENCE_RESOLUTION_STEPS,
        "observed_executed_fast_controls": copy.deepcopy(
            marker["identity_relay_extension"]["observed_executed_fast_controls"]
        ),
        "source_is_never_overwritten": True,
    }
    migrated["extra"]["workflow_note"] = (
        "MiniMax-H3 Ref2VA dual-identity sibling: two clean Picture identity anchors share one reference budget; "
        "the audio-aware planner defaults relay Off and can lazily expose the prior retained tail as Picture 3; "
        "the exact successful b430 H3 settings, lane audio slices, latent sampling, exact-frame assembly, and pristine ACE mux remain unchanged."
    )
    migrated["id"] = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{IDENTITY_RELAY_SCHEMA}/{_canonical_hash(source)}",
        )
    )
    migrated["revision"] = int(source.get("revision", 0) or 0) + 1
    subgraph["revision"] = int(subgraph.get("revision", 0) or 0) + 1
    subgraph.setdefault("state", {})["lastNodeId"] = ids.last_node
    subgraph.setdefault("state", {})["lastLinkId"] = ids.last_link
    migrated["last_node_id"] = ids.last_node
    migrated["last_link_id"] = ids.last_link

    validate_workflow(migrated, source=source)
    return migrated


def _origin_main(workflow: dict[str, Any], node: dict[str, Any], input_name: str) -> tuple[dict[str, Any], str]:
    slot = _input_slot(node, input_name)
    link_id = node["inputs"][slot].get("link")
    if link_id is None:
        raise WorkflowError(f"Main node {node['id']}.{input_name} is unconnected.")
    link = next((item for item in workflow["links"] if int(item[0]) == int(link_id)), None)
    if link is None:
        raise WorkflowError(f"Main node {node['id']}.{input_name} references missing link {link_id}.")
    origin = _node_by_id(workflow["nodes"], int(link[1]))
    return origin, str(origin["outputs"][int(link[2])]["name"])


def _origin_subgraph(subgraph: dict[str, Any], node: dict[str, Any], input_name: str) -> tuple[int, int, str]:
    slot = _input_slot(node, input_name)
    link_id = node["inputs"][slot].get("link")
    if link_id is None:
        raise WorkflowError(f"Subgraph node {node['id']}.{input_name} is unconnected.")
    link = next((item for item in subgraph["links"] if int(item["id"]) == int(link_id)), None)
    if link is None:
        raise WorkflowError(f"Subgraph node {node['id']}.{input_name} references missing link {link_id}.")
    return int(link["origin_id"]), int(link["origin_slot"]), str(link["type"])


def _validate_main_links(workflow: dict[str, Any]) -> None:
    ids = [int(link[0]) for link in workflow.get("links", [])]
    if len(ids) != len(set(ids)):
        raise WorkflowError("Main graph contains duplicate link ids.")
    nodes = {int(node["id"]): node for node in workflow["nodes"]}
    for link in workflow["links"]:
        link_id, origin_id, origin_slot, target_id, target_slot, link_type = link
        if int(origin_id) not in nodes or int(target_id) not in nodes:
            raise WorkflowError(f"Main link {link_id} escapes the graph.")
        origin = nodes[int(origin_id)]
        target = nodes[int(target_id)]
        if int(link_id) not in (origin["outputs"][int(origin_slot)].get("links") or []):
            raise WorkflowError(f"Main link {link_id} is absent from its origin output.")
        if target["inputs"][int(target_slot)].get("link") != link_id:
            raise WorkflowError(f"Main link {link_id} is absent from its target input.")
        target_type = str(target["inputs"][int(target_slot)].get("type"))
        if not _types_compatible(target_type, link_type):
            raise WorkflowError(f"Main link {link_id} type does not match its target input.")


def _validate_subgraph_links(subgraph: dict[str, Any]) -> None:
    ids = [int(link["id"]) for link in subgraph.get("links", [])]
    if len(ids) != len(set(ids)):
        raise WorkflowError("FAST subgraph contains duplicate link ids.")
    nodes = {int(node["id"]): node for node in subgraph["nodes"]}
    input_id = int(subgraph["inputNode"]["id"])
    output_id = int(subgraph["outputNode"]["id"])
    for link in subgraph["links"]:
        link_id = int(link["id"])
        if int(link["target_id"]) == output_id:
            target_boundary = subgraph["outputs"][int(link["target_slot"])]
            if link_id not in (target_boundary.get("linkIds") or []):
                raise WorkflowError(
                    f"Subgraph output link {link_id} is absent from its boundary descriptor."
                )
        else:
            target = nodes.get(int(link["target_id"]))
            if target is None:
                raise WorkflowError(f"Subgraph link {link_id} has no target node.")
            if target["inputs"][int(link["target_slot"])].get("link") != link_id:
                raise WorkflowError(f"Subgraph link {link_id} is absent from its target input.")
        if int(link["origin_id"]) == input_id:
            boundary = subgraph["inputs"][int(link["origin_slot"])]
            if link_id not in (boundary.get("linkIds") or []):
                raise WorkflowError(f"Subgraph boundary link {link_id} is absent from its input descriptor.")
        else:
            origin = nodes.get(int(link["origin_id"]))
            if origin is None or link_id not in (origin["outputs"][int(link["origin_slot"])].get("links") or []):
                raise WorkflowError(f"Subgraph link {link_id} is absent from its origin output.")


def validate_workflow(
    workflow: dict[str, Any],
    *,
    source: dict[str, Any] | None = None,
) -> None:
    marker = workflow.get("extra", {}).get(IDENTITY_RELAY_SCHEMA)
    if not isinstance(marker, dict) or int(marker.get("version", 0)) != IDENTITY_RELAY_VERSION:
        raise WorkflowError("Identity-relay marker is missing or unsupported.")
    if marker.get("default_continuity_relay_mode") != "Off":
        raise WorkflowError("Guarded sibling must default to identity-first relay Off.")
    _validate_main_links(workflow)
    subgraphs = [
        item for item in workflow.get("definitions", {}).get("subgraphs", [])
        if str(item.get("id")) == FAST_SUBGRAPH_ID
    ]
    if len(subgraphs) != 1:
        raise WorkflowError("Identity sibling has no unique FAST H3 subgraph.")
    subgraph = subgraphs[0]
    _validate_subgraph_links(subgraph)
    nodes = workflow["nodes"]
    context = _one(nodes, CONTEXT_TYPE)
    planner = _one(nodes, PLANNER_TYPE)
    pair = _one(nodes, PAIR_PREP_TYPE)
    manifest = next(
        node for node in nodes
        if node.get("type") == "PrimitiveStringMultiline"
        and node.get("title", "").startswith("H3 REFERENCE MANIFEST")
    )
    batch = _one(nodes, "BatchImagesNode")
    outer = _node_by_id(nodes, 710)
    extension = workflow["extra"][SOURCE_EXPANSION_SCHEMA].get("identity_relay_extension")
    if not isinstance(extension, dict):
        raise WorkflowError("Expansion marker has no identity-relay extension metadata.")
    secondary = _node_by_id(nodes, int(extension["secondary_identity_loader"]))
    secondary_asset = extension.get("secondary_identity_asset")
    if secondary.get("type") != "LoadImage":
        raise WorkflowError("Picture 2 identity asset is not loaded by LoadImage.")
    if (secondary.get("widgets_values") or [None])[0] != SECONDARY_IDENTITY_RELATIVE_PATH:
        raise WorkflowError("Picture 2 loader no longer points to the generated identity sheet.")
    if not isinstance(secondary_asset, dict) or secondary_asset.get(
        "relative_input_path"
    ) != SECONDARY_IDENTITY_RELATIVE_PATH or str(secondary_asset.get("sha256", "")).lower() != SECONDARY_IDENTITY_SHA256:
        raise WorkflowError("Picture 2 identity-sheet provenance is missing or changed.")
    _require_secondary_identity_asset()
    lora = _node_by_id(nodes, 713)
    sampler = _node_by_id(subgraph["nodes"], 657)
    scheduler = _node_by_id(subgraph["nodes"], 658)
    observed_controls = extension.get("observed_executed_fast_controls")
    expected_controls = {
        "sampler_node": 657,
        "sampler": "res_multistep",
        "scheduler_node": 658,
        "scheduler": "simple",
        "steps": 8,
        "denoise": 1.0,
        "lora_node": 713,
        "lora_name": str((lora.get("widgets_values") or [""])[0]),
        "lora_strength": 0.60,
    }
    if observed_controls != expected_controls:
        raise WorkflowError("Observed b430 execution controls are missing or inaccurate.")
    if sampler.get("widgets_values") != ["res_multistep"] or "res_multistep" not in str(
        sampler.get("title", "")
    ):
        raise WorkflowError("Sampler title does not agree with its executed res_multistep widget.")
    if scheduler.get("widgets_values") != ["simple", 8, 1] or "simple / 8" not in str(
        scheduler.get("title", "")
    ):
        raise WorkflowError("Scheduler title does not agree with its executed simple/8 widgets.")
    if float((lora.get("widgets_values") or [None, -1])[1]) != 0.60 or "0.60" not in str(
        lora.get("title", "")
    ):
        raise WorkflowError("LoRA title does not agree with its executed 0.60 strength.")
    if not all(
        token in str(subgraph.get("name", ""))
        for token in ("res_multistep", "simple", "8")
    ):
        raise WorkflowError("FAST subgraph label does not report the observed b430 controls.")

    context_origin, context_output = _origin_main(workflow, context, "reference_images")
    if (context_origin.get("type"), context_output) != ("BatchImagesNode", "IMAGE"):
        raise WorkflowError("Director reference_images must receive the ordered two-picture batch.")
    manifest_origin, manifest_output = _origin_main(workflow, context, "reference_manifest")
    if (manifest_origin["id"], manifest_output) != (manifest["id"], "STRING"):
        raise WorkflowError("Director does not use the shared identity manifest.")
    planner_manifest_origin, planner_manifest_output = _origin_main(workflow, planner, "reference_manifest")
    if (planner_manifest_origin["id"], planner_manifest_output) != (manifest["id"], "STRING"):
        raise WorkflowError("Planner does not use the shared identity manifest.")
    if list(planner.get("widgets_values") or [])[-2:] != [2, "Off"]:
        raise WorkflowError("Planner must default to two identity pictures and relay Off.")
    if str((manifest.get("widgets_values") or [""])[0]) != DUAL_IDENTITY_MANIFEST:
        raise WorkflowError("Shared identity manifest text changed unexpectedly.")
    if int((context.get("widgets_values") or [0])[-1]) != 1:
        raise WorkflowError("Dual identity Director must require exactly one semantic Subject.")

    pair_p1, pair_p1_output = _origin_main(workflow, pair, "reference_image_1")
    pair_p2, pair_p2_output = _origin_main(workflow, pair, "reference_image_2")
    if (int(pair_p1["id"]), pair_p1_output) != (179, "IMAGE"):
        raise WorkflowError("Reference Pair Prep Picture 1 does not use the proven primary loader.")
    if (int(pair_p2["id"]), pair_p2_output) != (int(secondary["id"]), "IMAGE"):
        raise WorkflowError("Reference Pair Prep Picture 2 does not use the clean supporting loader.")
    if list(pair.get("widgets_values") or [])[-2:] != [1.0, 0.40]:
        raise WorkflowError("Dual identity references no longer share the guarded one-frame budget.")
    width_origin, width_output = _origin_main(workflow, pair, "generation_width")
    height_origin, height_output = _origin_main(workflow, pair, "generation_height")
    if (width_origin.get("type"), width_output) != ("DiffusionGemmaJSONSplitter", "resolution_width"):
        raise WorkflowError("Reference budget width is not the native H3 generation width.")
    if (height_origin.get("type"), height_output) != ("DiffusionGemmaJSONSplitter", "resolution_height"):
        raise WorkflowError("Reference budget height is not the native H3 generation height.")

    p1_origin, p1_output = _origin_main(workflow, outer, "ref_images.ref_image_0")
    p2_origin, p2_output = _origin_main(workflow, outer, "identity_picture_2")
    plan_origin, plan_output = _origin_main(workflow, outer, "shot_plan_json")
    if (p1_origin["id"], p1_output) != (pair["id"], "reference_image_1"):
        raise WorkflowError("FAST H3 Picture 1 does not use Reference Pair Prep.")
    if (p2_origin["id"], p2_output) != (pair["id"], "reference_image_2"):
        raise WorkflowError("FAST H3 Picture 2 does not use Reference Pair Prep.")
    if (plan_origin["id"], plan_output) != (planner["id"], "plan_json"):
        raise WorkflowError("FAST H3 relay gates do not receive the authoritative shot plan.")

    runtime = workflow["extra"][SOURCE_EXPANSION_SCHEMA]["runtime_lane_node_ids"]
    h3_nodes = [_node_by_id(subgraph["nodes"], int(item)) for item in runtime["h3"]]
    gates = [_node_by_id(subgraph["nodes"], int(item)) for item in extension["relay_gate_nodes"]]
    relay_scalers = [
        _node_by_id(subgraph["nodes"], int(item))
        for item in extension["relay_scale_nodes"]
    ]
    tails = [_node_by_id(subgraph["nodes"], int(item)) for item in runtime["tails"]]
    boundary_p2 = next(
        index for index, item in enumerate(subgraph["inputs"])
        if item.get("name") == "identity_picture_2"
    )
    boundary_plan = next(
        index for index, item in enumerate(subgraph["inputs"])
        if item.get("name") == "shot_plan_json"
    )
    for lane, h3 in enumerate(h3_nodes, start=1):
        if str((h3.get("widgets_values") or [None])[-1]) != "match":
            raise WorkflowError(f"H3 lane {lane} changed ref_image_size away from match.")
        p2_origin_id, p2_origin_slot, p2_type = _origin_subgraph(subgraph, h3, "ref_images.ref_image_1")
        if (p2_origin_id, p2_origin_slot, p2_type) != (-10, boundary_p2, "IMAGE"):
            raise WorkflowError(f"H3 lane {lane} Picture 2 is not the clean supporting identity input.")
        if lane > 1:
            p3_origin_id, _p3_slot, p3_type = _origin_subgraph(subgraph, h3, "ref_images.ref_image_2")
            if p3_origin_id != int(gates[lane - 2]["id"]) or p3_type != "IMAGE":
                raise WorkflowError(f"H3 lane {lane} Picture 3 does not come from its relay gate.")
    for lane, (gate, scaler, tail) in enumerate(
        zip(gates, relay_scalers, tails), start=2
    ):
        if gate.get("type") != RELAY_GATE_TYPE or list(gate.get("widgets_values") or []) != [lane]:
            raise WorkflowError(f"Relay gate {lane} has the wrong type or lane selector.")
        if (
            scaler.get("type") != "ImageScaleToTotalPixels"
            or list(scaler.get("widgets_values") or [])
            != ["area", RELAY_REFERENCE_MEBIPIXELS, RELAY_REFERENCE_RESOLUTION_STEPS]
        ):
            raise WorkflowError(f"Relay scaler {lane} no longer enforces the guarded Picture 3 cap.")
        plan_source, plan_slot, _ = _origin_subgraph(subgraph, gate, "shot_plan_json")
        if (plan_source, plan_slot) != (-10, boundary_plan):
            raise WorkflowError(f"Relay gate {lane} does not use the shot plan boundary input.")
        scale_source, _scale_slot, _ = _origin_subgraph(subgraph, gate, "relay_image")
        if scale_source != int(scaler["id"]):
            raise WorkflowError(f"Relay gate {lane} does not use its bounded relay scaler.")
        tail_source, _tail_slot, _ = _origin_subgraph(subgraph, scaler, "image")
        if tail_source != int(tail["id"]):
            raise WorkflowError(f"Relay scaler {lane} does not use the immediately previous retained tail.")

    # Compare all performance-critical fast-path controls and audio/latent links
    # against the exact executed source.  Only reference routing may change.
    if source is not None:
        source_fast = source.get("extra", {}).get("diffusiongemma.minimax_h3_fast_test", {})
        migrated_fast = workflow.get("extra", {}).get("diffusiongemma.minimax_h3_fast_test", {})
        if migrated_fast != source_fast:
            raise WorkflowError("FAST H3 marker changed; the 423-second sampling path was not preserved exactly.")
        for node_id in (651, 653, 654, 661, 662, 664, 679, 713, 715, 716):
            before = _node_by_id(source["nodes"], node_id)
            after = _node_by_id(workflow["nodes"], node_id)
            if before.get("widgets_values") != after.get("widgets_values"):
                raise WorkflowError(f"Performance-critical node {node_id} widgets changed.")
        source_subgraph = next(
            item for item in source["definitions"]["subgraphs"]
            if str(item.get("id")) == FAST_SUBGRAPH_ID
        )
        for node_id in (656, 657, 658, 659, 660, 663, 685, 686, 687, 688, 693, 694, 695, 696, 701, 702, 703, 704, 717, 718, 719, 720):
            before = _node_by_id(source_subgraph["nodes"], node_id)
            after = _node_by_id(subgraph["nodes"], node_id)
            before_comparable = copy.deepcopy(before)
            after_comparable = copy.deepcopy(after)
            if node_id in {657, 658}:
                before_comparable.pop("title", None)
                after_comparable.pop("title", None)
            if before_comparable != after_comparable:
                raise WorkflowError(
                    f"Unrelated FAST subgraph node {node_id} changed; latent/sampling cleanup must remain exact."
                )


def load_workflow(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    if not path.is_file():
        raise WorkflowError(f"Workflow does not exist: {path}")
    payload = path.read_bytes()
    try:
        workflow = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Workflow is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(workflow, dict):
        raise WorkflowError(f"Workflow root must be a JSON object: {path}")
    return workflow, _file_hash(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source", type=Path, help="Existing proven workflow/capture JSON.")
    source.add_argument(
        "--history-prompt-id",
        default=DEFAULT_HISTORY_PROMPT_ID,
        help="Completed ComfyUI prompt whose extra_pnginfo.workflow is authoritative.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--source-history-prompt-id",
        default="",
        help="Optional provenance prompt id when --source is an extracted history capture.",
    )
    parser.add_argument("--capture-output", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--check", action="store_true", help="Validate in memory without writing the migrated sibling.")
    args = parser.parse_args(argv)

    history_prompt_id = ""
    if args.source is not None:
        source_workflow, source_hash = load_workflow(args.source)
        source_path = args.source.resolve()
        source_canonical = _canonical_hash(source_workflow)
        history_prompt_id = str(args.source_history_prompt_id or "")
        print(
            f"loaded {source_path} file_sha256={source_hash} canonical_sha256={source_canonical}"
        )
    else:
        history_prompt_id = str(args.history_prompt_id)
        if args.check:
            source_workflow = fetch_history_workflow(history_prompt_id, base_url=args.base_url)
            source_hash = _file_hash(_json_bytes(source_workflow))
            source_canonical = _canonical_hash(source_workflow)
            print(
                f"checked-history {history_prompt_id} reconstructed_file_sha256={source_hash} "
                f"canonical_sha256={source_canonical}"
            )
        else:
            source_workflow, source_hash, source_canonical = capture_history_workflow(
                history_prompt_id,
                args.capture_output,
                base_url=args.base_url,
            )
            print(
                f"captured {args.capture_output.resolve()} prompt_id={history_prompt_id} "
                f"file_sha256={source_hash} canonical_sha256={source_canonical}"
            )
    if args.capture_only:
        return 0

    migrated = migrate_workflow(
        source_workflow,
        source_file_sha256=source_hash,
        source_history_prompt_id=history_prompt_id,
    )
    result_hash = _file_hash(_json_bytes(migrated))
    result_canonical = _canonical_hash(migrated)
    if args.check:
        print(
            f"validated in memory output={args.output.resolve()} reconstructed_file_sha256={result_hash} "
            f"canonical_sha256={result_canonical}"
        )
    else:
        written_hash = _exclusive_json_write(args.output, migrated)
        if written_hash != result_hash:
            raise WorkflowError("Migrated sibling hash changed during exclusive write.")
        print(
            f"created {args.output.resolve()} file_sha256={written_hash} canonical_sha256={result_canonical}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
