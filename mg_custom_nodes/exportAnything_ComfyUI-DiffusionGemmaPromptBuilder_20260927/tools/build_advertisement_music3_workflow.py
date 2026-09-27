#!/usr/bin/env python3
"""Build the separate, user-editable Advertisement + Music 3 workflow.

The production graph is deliberately flattened: it does not reuse either V6
subgraph UUID, so edits to this Advertisement workflow cannot mutate the
known-good music-video workflow.  Node socket metadata is read from a live
ComfyUI ``/object_info`` endpoint only while generating the checked-in UI
artifact; the resulting workflow is self-contained.

The native Music 3 route is the default.  The uploaded-song and legacy ACE
branches are also present and connected through the lazy Advertisement source
router, but remain dormant unless selected.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping
from urllib.request import urlopen
import uuid


REPOSITORY = Path(__file__).resolve().parents[1]
KNOWN_GOOD_UI = REPOSITORY / "examples" / "15_minimax_h3_ref2va_music_video_v6.json"
KNOWN_GOOD_UI_SHA256 = "594f6aef94531f87b74659e1e8004cccdff00716d0cfe263cee6bf2149545c83"
FIELD_TEST_API = (
    REPOSITORY / "examples" / "api" / "16_minimax_h3_ref2va_advertisement_glowbloom_v1_api.json"
)
FIELD_TEST_API_CANONICAL_SHA256 = "6296d86a7640ce1babe307f68c3ddab845341e5e1185cb2216924097e9b4f743"
MUSIC3_API = (
    REPOSITORY / "examples" / "api" / "17_minimax_h3_ref2va_advertisement_music3_glowbloom_v1_api.json"
)
MUSIC3_API_CANONICAL_SHA256 = "d6f8d8de431f233c00a1172afca39815b8bd8bd619b633e8e5162db310fffc82"
OUTPUT = REPOSITORY / "examples" / "16_minimax_h3_ref2va_advertisement_music3_v1.json"

SCHEMA = "diffusiongemma.advertisement_music3_workflow"
VERSION = 1
WORKFLOW_UUID = "d9118b98-cc97-5cf5-81d8-86eff1336120"
FORBIDDEN_RUNTIME_CLASSES = frozenset(
    {
        "EasyCache",
        "TorchCompileModel",
        "PathchSageAttentionKJ",
        "PatchSageAttentionKJ",
        "easy cleanGpuUsed",
        "FL_UnloadAllModels",
    }
)


class WorkflowError(RuntimeError):
    """Raised when workflow lineage or live node metadata is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


def _load_guarded(path: Path, *, byte_hash: str = "", canonical_hash: str = "") -> Any:
    data = path.read_bytes()
    if byte_hash and _sha256_bytes(data) != byte_hash:
        raise WorkflowError(f"Immutable source changed: {path}")
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Invalid UTF-8 JSON source: {path}") from exc
    if canonical_hash and _canonical_sha256(parsed) != canonical_hash:
        raise WorkflowError(f"Immutable canonical graph changed: {path}")
    return parsed


def _object_info(url: str) -> dict[str, Any]:
    endpoint = url.rstrip("/") + "/object_info"
    try:
        with urlopen(endpoint, timeout=30) as response:  # noqa: S310 - explicit local endpoint
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - environment-specific transport
        raise WorkflowError(f"Could not read live ComfyUI node metadata from {endpoint}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WorkflowError("ComfyUI /object_info did not return a JSON object.")
    return payload


def _is_connection(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _api_node(class_type: str, title: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def _copy_ancestors(
    source: Mapping[str, Any], target: dict[str, Any], roots: list[str]
) -> None:
    pending = list(roots)
    while pending:
        node_id = pending.pop()
        if node_id in target:
            continue
        node = source.get(node_id)
        if not isinstance(node, Mapping):
            raise WorkflowError(f"Legacy fallback source is missing node {node_id}.")
        target[node_id] = copy.deepcopy(node)
        for value in node.get("inputs", {}).values():
            if _is_connection(value):
                pending.append(value[0])


def _with_lazy_fallbacks(
    music3: dict[str, Any], field_test: Mapping[str, Any]
) -> dict[str, Any]:
    graph = copy.deepcopy(music3)
    # Reference preparation must not wait for Director resolution because the
    # Director itself consumes the prepared reference batch.  The saved 9:16
    # advertisement baseline uses the contract node's native 480x864 defaults;
    # H3's final generation dimensions remain Director-controlled downstream.
    graph["803"]["inputs"]["generation_width"] = 480
    graph["803"]["inputs"]["generation_height"] = 864

    ace_candidates = ["623:567", "623:597", "623:602", "623:607"]
    _copy_ancestors(field_test, graph, ace_candidates + ["562"])
    upload = field_test.get("736")
    if not isinstance(upload, Mapping):
        raise WorkflowError("Field-test graph has no uploaded-song loader.")
    graph["736"] = copy.deepcopy(upload)
    graph["736"]["inputs"]["audio_file"] = ""
    graph["736"]["_meta"]["title"] = "UPLOAD SONG — used only when Upload song is selected"

    # Force-input primitives make fallback controls visible and editable in the
    # UI while keeping the router's selected branch truly lazy.
    graph["846"] = _api_node("SeedNode", "Music 3 candidate count — one", seed=1)
    graph["847"] = _api_node("PrimitiveFloat", "Uploaded-song expected BPM — zero means measure", value=0.0)
    graph["848"] = _api_node("PrimitiveStringMultiline", "Uploaded-song lyrics — optional outside Lyrics mode", value="")
    graph["850"] = _api_node("SeedNode", "Legacy ACE candidate count — four", seed=4)
    graph["851"] = _api_node(
        "PrimitiveStringMultiline",
        "Advertisement performance guidance — music-only H3 motion authority",
        value=graph["675"]["inputs"]["base_audio_guidance"],
    )
    graph["675"]["inputs"]["base_audio_guidance"] = ["851", 0]

    # The old ACE audition plan depended on the H3 Director context, while the
    # Director waits for the selected soundtrack.  Reuse only the proven ACE
    # model/sampler branch and feed it from the already-governed Advertisement
    # Soundtrack contract.  This keeps ACE a real lazy fallback without a
    # circular song<->Director dependency.
    for node_id in ("623:561", "623:594", "623:599", "623:604"):
        inputs = graph[node_id]["inputs"]
        inputs["tags"] = ["805", 0]
        inputs["lyrics"] = ["805", 1]
        inputs["bpm"] = ["805", 5]
        inputs["timesignature"] = "4"
        inputs["language"] = "unknown"
        inputs["keyscale"] = "C major"

    # These nodes existed only to derive the former Director-dependent ACE
    # blueprint or to preview intermediate audio.  They are not production
    # outputs and must not leak into the independent Advertisement workflow.
    for node_id in ("183", "556", "578", "609"):
        graph.pop(node_id, None)

    router = graph["814"]["inputs"]
    router.update(
        {
            "music3_candidate_count": ["846", 0],
            "uploaded_audio": ["736", 0],
            "uploaded_duration_seconds": ["736", 1],
            "uploaded_expected_bpm": ["847", 0],
            "uploaded_lyrics": ["848", 0],
            "uploaded_waveform_sha256": ["736", 2],
            "uploaded_status": ["736", 3],
            "uploaded_ready": ["736", 4],
            "ace_candidate_count": ["850", 0],
            "ace_expected_bpm": ["805", 3],
            "ace_lyrics": ["805", 1],
            "ace_duration_seconds": ["562", 0],
            "ace_candidate_1": [ace_candidates[0], 0],
            "ace_candidate_2": [ace_candidates[1], 0],
            "ace_candidate_3": [ace_candidates[2], 0],
            "ace_candidate_4": [ace_candidates[3], 0],
        }
    )
    graph["818"]["inputs"]["source_policy"] = ["814", 8]
    inherited = sorted(
        (node_id, str(node.get("class_type", "")))
        for node_id, node in graph.items()
        if str(node.get("class_type", "")) in FORBIDDEN_RUNTIME_CLASSES
    )
    if inherited:
        raise WorkflowError(f"Forbidden inherited runtime nodes entered Advertisement UI source: {inherited}")
    return graph


def _definition(info: Mapping[str, Any], name: str) -> tuple[Any, dict[str, Any]] | None:
    for section in ("required", "optional"):
        raw = info.get("input", {}).get(section, {}).get(name)
        if isinstance(raw, list) and raw:
            config = raw[1] if len(raw) > 1 and isinstance(raw[1], Mapping) else {}
            return raw[0], dict(config)
    if "." in name:
        group, _child = name.split(".", 1)
        raw = info.get("input", {}).get("optional", {}).get(group)
        if isinstance(raw, list) and len(raw) > 1 and isinstance(raw[1], Mapping):
            template = raw[1].get("template", {})
            required = template.get("input", {}).get("required", {})
            if isinstance(required, Mapping) and required:
                first = next(iter(required.values()))
                if isinstance(first, list) and first:
                    config = first[1] if len(first) > 1 and isinstance(first[1], Mapping) else {}
                    return first[0], dict(config)
    return None


def _ordered_inputs(info: Mapping[str, Any], api_inputs: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    order = info.get("input_order", {})
    for section in ("required", "optional"):
        for name in order.get(section, []):
            if name in api_inputs:
                values.append(str(name))
            elif any(str(item).startswith(str(name) + ".") for item in api_inputs):
                values.extend(
                    str(item) for item in api_inputs if str(item).startswith(str(name) + ".")
                )
    for name in api_inputs:
        if str(name) not in values:
            values.append(str(name))
    return values


def _widget_compatible(type_name: Any, config: Mapping[str, Any]) -> bool:
    if bool(config.get("forceInput")):
        return False
    if isinstance(type_name, list):
        return True
    return str(type_name) in {"STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"}


def _default_widget(type_name: Any, config: Mapping[str, Any]) -> Any:
    if "default" in config:
        return copy.deepcopy(config["default"])
    if isinstance(type_name, list):
        return copy.deepcopy(type_name[0]) if type_name else ""
    if str(type_name) == "COMBO":
        options = config.get("options", [])
        return copy.deepcopy(options[0]) if options else ""
    return {"STRING": "", "INT": 0, "FLOAT": 0.0, "BOOLEAN": False}.get(str(type_name), "")


def _socket_type(type_name: Any, config: Mapping[str, Any]) -> str:
    if isinstance(type_name, list) or str(type_name) == "COMBO":
        return "COMBO"
    return str(type_name)


def _cnr_id(info: Mapping[str, Any]) -> str:
    module = str(info.get("python_module", ""))
    if "ComfyUI-DiffusionGemmaPromptBuilder" in module:
        return "diffusiongemma-prompt-builder"
    if "easy" in module.casefold():
        return "comfyui-easy-use"
    if module.startswith("comfy") or module == "nodes":
        return "comfy-core"
    return module.rsplit(".", 1)[-1] or "comfy-core"


def _size_for(class_type: str, widget_count: int, input_count: int) -> list[float]:
    explicit = {
        "LoadImage": [430.0, 520.0],
        "DiffusionGemmaAdvertisementCampaignContract": [620.0, 720.0],
        "DiffusionGemmaAdvertisementReferenceContract": [620.0, 600.0],
        "DiffusionGemmaAdvertisementReferenceAssetPrep": [620.0, 500.0],
        "DiffusionGemmaAdvertisementSoundtrackContract": [620.0, 660.0],
        "DiffusionGemmaAdvertisementSoundtrackSourceRouter": [650.0, 850.0],
        "DiffusionGemmaAdvertisementMultiShotPlanner": [700.0, 880.0],
        "MiniMaxH3ReferenceToVideo": [560.0, 620.0],
        "DiffusionGemmaCoTGenerator": [520.0, 420.0],
        "DiffusionGemmaH3ReferenceContext": [620.0, 620.0],
        "DiffusionGemmaAdvertisementMediaQAGate": [620.0, 430.0],
        "SaveVideo": [420.0, 170.0],
    }
    if class_type in explicit:
        return explicit[class_type]
    return [380.0, float(max(90, min(620, 70 + widget_count * 28 + input_count * 22)))]


def _position(api_id: str, class_type: str, index: int) -> list[float]:
    fixed: dict[str, tuple[float, float]] = {
        "179": (-3850, -2600), "721": (-3850, -1450), "800": (-3850, -300),
        "801": (-3200, -2600), "802": (-3200, -1450), "803": (-2450, -1600),
        "804": (-3200, 650), "805": (-2450, 650), "806": (-2450, 1120),
        "807": (-2450, 1280), "808": (-2450, 1440), "809": (-1900, 900),
        "810": (-1400, 1120), "811": (-1400, 1280), "812": (-980, 980),
        "813": (-520, 1020), "814": (0, 650), "815": (760, 800),
        "816": (1450, 900), "817": (1900, 760), "818": (2440, 800),
        "819": (2950, 850), "844": (3400, 650), "845": (3400, 900),
        "183": (-1850, -2600), "674": (-1150, -2600), "185": (-1850, -1800),
        "182": (-1850, -1400), "673": (-1150, -1850), "675": (-500, -1850),
        "186": (-350, -2600), "820": (300, -2600), "187": (950, -2600), "188": (1550, -2600),
        "676": (1450, -2600), "821": (2200, -2600), "822": (2950, -2600),
        "823": (3700, -2600), "418": (3050, -1750), "678": (3650, -1650),
        "661": (2800, -1250), "713": (3250, -1250), "715": (3700, -1250),
        "716": (4150, -1250), "662": (2800, -1050), "653": (2800, -850),
        "654": (2800, -650), "710:657": (4550, -1250), "710:658": (4550, -1050),
        "831": (6600, -2500), "832": (6600, -1750), "833": (7300, -2300),
        "834": (7300, -1600), "835": (7950, -2300), "836": (8650, -2300),
        "837": (8650, -1700), "838": (9150, -1700), "839": (8650, -1350),
        "840": (9150, -1350), "841": (8650, -1000), "842": (9150, -1000),
        "843": (9300, -2300), "844": (9300, 650), "845": (9300, 900),
        "900": (1400, 1450), "901": (900, -2050), "902": (200, -2050),
        "736": (-3200, 1600), "846": (-2450, 1850), "847": (-3200, 1850),
        "848": (-3200, 2020), "851": (-500, -1450),
        "849": (-2350, -3000),
    }
    if api_id in fixed:
        return [float(fixed[api_id][0]), float(fixed[api_id][1])]
    lane_map = {
        "710:667": 0, "710:663": 0, "710:660": 0, "710:717": 0,
        "710:659": 0, "710:656": 0,
        "710:684": 1, "710:685": 1, "710:686": 1, "710:718": 1,
        "710:687": 1, "710:688": 1,
        "710:692": 2, "710:693": 2, "710:694": 2, "710:719": 2,
        "710:695": 2, "710:696": 2,
        "710:700": 3, "710:701": 3, "710:702": 3, "710:720": 3,
        "710:703": 3, "710:704": 3,
    }
    if api_id in lane_map:
        lane = lane_map[api_id]
        y = -500 + lane * 1050
        column = {
            "MiniMaxH3ReferenceToVideo": 4300,
            "RandomNoise": 4900,
            "BasicGuider": 4900,
            "FL_UnloadAllModels": 4900,
            "SamplerCustomAdvanced": 5350,
            "VAEDecode": 5750,
        }.get(class_type, 5200)
        if class_type in {"BasicGuider", "FL_UnloadAllModels"}:
            y += 260 if class_type == "BasicGuider" else 500
        return [float(column), float(y)]
    relay_positions = {
        "681": (3800, -300), "683": (3800, 750), "691": (3800, 1800), "699": (3800, 2850),
        "825": (6150, 250), "826": (6600, 250), "827": (6150, 1300), "828": (6600, 1300),
        "829": (6150, 2350), "830": (6600, 2350),
    }
    if api_id in relay_positions:
        x, y = relay_positions[api_id]
        return [float(x), float(y)]
    if api_id.startswith("623:"):
        try:
            inner = int(api_id.split(":", 1)[1])
        except ValueError:
            inner = index
        return [-1600.0 + float((inner % 4) * 420), 1650.0 + float(((inner // 4) % 6) * 280)]
    if api_id in {"556", "558", "559", "560", "562", "565", "578", "589", "590", "591", "592", "609"}:
        slot = sorted(["556", "558", "559", "560", "562", "565", "578", "589", "590", "591", "592", "609"]).index(api_id)
        return [-2450.0 + float((slot % 4) * 430), 1650.0 + float((slot // 4) * 310)]
    return [float(9600 + (index % 4) * 420), float(-2500 + (index // 4) * 260)]


def _api_sort_key(value: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"(\d+)(?::(\d+))?", value)
    if match:
        return int(match.group(1)), int(match.group(2) or -1), value
    return 1_000_000, 0, value


def _topological_order(graph: Mapping[str, Any]) -> list[str]:
    pending = set(graph)
    ordered: list[str] = []
    while pending:
        ready = sorted(
            [
                node_id
                for node_id in pending
                if all(
                    not _is_connection(value) or value[0] not in pending
                    for value in graph[node_id].get("inputs", {}).values()
                )
            ],
            key=_api_sort_key,
        )
        if not ready:
            raise WorkflowError("API graph contains a dependency cycle.")
        ordered.extend(ready)
        pending.difference_update(ready)
    return ordered


def build_workflow(graph: dict[str, Any], object_info: Mapping[str, Any]) -> dict[str, Any]:
    api_ids = sorted(graph, key=_api_sort_key)
    numeric_ids = {api_id: index + 1 for index, api_id in enumerate(api_ids)}
    topo = _topological_order(graph)
    topo_order = {api_id: index for index, api_id in enumerate(topo)}
    nodes: list[dict[str, Any]] = []
    records: dict[str, dict[str, Any]] = {}

    for index, api_id in enumerate(api_ids):
        api_node = graph[api_id]
        class_type = str(api_node.get("class_type", ""))
        info = object_info.get(class_type)
        if not isinstance(info, Mapping):
            raise WorkflowError(f"Live ComfyUI has no node metadata for {class_type} ({api_id}).")
        api_inputs = api_node.get("inputs", {})
        input_records: list[dict[str, Any]] = []
        widget_values: list[Any] = []
        ordered_names = _ordered_inputs(info, api_inputs)
        for name in ordered_names:
            definition = _definition(info, name)
            if definition is None:
                # Some custom nodes persist frontend-only scalar widgets that
                # are intentionally absent from the execution contract.  A
                # connected socket is different: ComfyUI forwards it to the
                # node function, so an unknown connected name is a runtime
                # TypeError waiting to happen.  Fail that case closed while
                # preserving established frontend-only widget state.
                if _is_connection(api_inputs.get(name)):
                    raise WorkflowError(
                        f"Live ComfyUI {class_type} has no connected input {name} ({api_id})."
                    )
                if name in api_inputs:
                    widget_values.append(copy.deepcopy(api_inputs[name]))
                continue
            type_name, config = definition
            value = api_inputs.get(name)
            socket_type = _socket_type(type_name, config)
            if _is_connection(value):
                item: dict[str, Any] = {"name": name, "type": socket_type, "link": None}
                if _widget_compatible(type_name, config):
                    item["widget"] = {"name": name}
                if "." in name:
                    item["localized_name"] = name
                    item["label"] = name.split(".", 1)[1]
                input_records.append(item)
            elif bool(config.get("forceInput")):
                # Guarded above by explicit primitive nodes in this workflow.
                if value is not None:
                    raise WorkflowError(f"Force-input scalar {api_id}:{name} needs a primitive connection.")
            if _widget_compatible(type_name, config):
                widget_values.append(
                    copy.deepcopy(value) if value is not None and not _is_connection(value) else _default_widget(type_name, config)
                )
                if bool(config.get("control_after_generate")):
                    widget_values.append("fixed")

        # Current ComfyUI serializes both LoadImage's ordinary image widget and
        # its frontend-only upload widget as input descriptors.  Widget values
        # alone preview correctly but the media validator reports a false
        # missing-image error unless these exact descriptors are also present.
        if class_type == "LoadImage":
            input_records = [
                {
                    "localized_name": "image",
                    "name": "image",
                    "type": "COMBO",
                    "widget": {"name": "image"},
                    "link": None,
                },
                {
                    "localized_name": "choose file to upload",
                    "name": "upload",
                    "type": "IMAGEUPLOAD",
                    "widget": {"name": "upload"},
                    "link": None,
                },
            ]
            if len(widget_values) == 1:
                widget_values.append("image")

        output_types = list(info.get("output", []))
        output_names = list(info.get("output_name", []))
        outputs = []
        for slot, output_type in enumerate(output_types):
            name = str(output_names[slot] if slot < len(output_names) else output_type)
            outputs.append({"name": name, "type": _socket_type(output_type, {}), "links": []})
        size = _size_for(class_type, len(widget_values), len(input_records))
        title = str(api_node.get("_meta", {}).get("title") or info.get("display_name") or class_type)
        node = {
            "id": numeric_ids[api_id],
            "type": class_type,
            "pos": _position(api_id, class_type, index),
            "size": size,
            "flags": {},
            "order": topo_order[api_id],
            "mode": 0,
            "inputs": input_records,
            "outputs": outputs,
            "title": title,
            "properties": {
                "cnr_id": _cnr_id(info),
                "Node name for S&R": class_type,
                "diffusiongemma_api_node_id": api_id,
            },
            "widgets_values": widget_values,
        }
        nodes.append(node)
        records[api_id] = node

    links: list[list[Any]] = []
    next_link = 0
    for target_api_id in api_ids:
        target = records[target_api_id]
        api_inputs = graph[target_api_id].get("inputs", {})
        for input_name, value in api_inputs.items():
            if not _is_connection(value):
                continue
            origin_api_id, origin_slot = value
            if origin_api_id not in records:
                raise WorkflowError(f"Dangling API origin {origin_api_id} -> {target_api_id}:{input_name}.")
            origin = records[origin_api_id]
            if int(origin_slot) >= len(origin["outputs"]):
                raise WorkflowError(f"Invalid output slot {origin_api_id}:{origin_slot}.")
            target_slots = [
                slot for slot, item in enumerate(target["inputs"]) if item.get("name") == input_name
            ]
            if len(target_slots) != 1:
                raise WorkflowError(f"UI socket reconstruction failed for {target_api_id}:{input_name}.")
            target_slot = target_slots[0]
            next_link += 1
            link_type = str(origin["outputs"][int(origin_slot)].get("type") or target["inputs"][target_slot].get("type") or "*")
            links.append(
                [
                    next_link,
                    int(origin["id"]),
                    int(origin_slot),
                    int(target["id"]),
                    int(target_slot),
                    link_type,
                ]
            )
            target["inputs"][target_slot]["link"] = next_link
            origin["outputs"][int(origin_slot)]["links"].append(next_link)

    note_id = len(nodes) + 1
    note = {
        "id": note_id,
        "type": "MarkdownNote",
        "pos": [-3850.0, -3300.0],
        "size": [1850.0, 560.0],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "title": "START HERE — separate Advertisement system / MiniMax Music 3 default",
        "properties": {},
        "widgets_values": [
            "# DiffusionGemma Advertisement — MiniMax Music 3 + MiniMax H3\n\n"
            "This is a new, flattened workflow with its own workflow UUID. The known-good V6 music-video workflow is not a parent graph and is never overwritten.\n\n"
            "1. Picture 1 is the performer hero. Picture 2 is a same-performer identity sheet. Picture 3 is a product-only contact sheet. Picture 4 is created automatically from the exact retained prior-lane tail and is continuity evidence only.\n"
            "2. Edit CAMPAIGN, REFERENCES, and SOUNDTRACK. MiniMax Music 3 is the default. SONG SOURCE also exposes Upload song and Legacy ACE-Step; unselected branches remain lazy.\n"
            "3. Use ADVERTISEMENT CONTROLS as the single duration, native-shot, and performance authority. The default 30-second master uses eight native H3 shots inside two 15-second generation lanes. EasyCache and TorchCompile are absent from the baseline because they provided no measured gain.\n"
            "4. The motion guide is music-only. Optional non-diegetic VO is mixed only into the delivery master and never fed to H3.\n"
            "5. Exact campaign copy is rendered in the deterministic three-second end card. The workflow saves review drafts of the master plus real 15s and 6s cutdowns. Requested 1:1/16:9 adaptations remain explicitly unrendered until a semantic reframe renderer is connected.\n"
            "6. QA defaults visual identity, copy legibility, and sync to not_measured. Saved files are review drafts; every delivery pass requires named evidence."
        ],
    }
    nodes.append(note)

    groups = [
        {"id": 1, "title": "1. Campaign + independent performer/product references", "bounding": [-3950, -3400, 4200, 3250], "color": "#d66b3d", "font_size": 36, "flags": {}},
        {"id": 2, "title": "2. MiniMax Music 3 default + lazy Upload / ACE fallbacks", "bounding": [-3300, 500, 7300, 3400], "color": "#4f8cc9", "font_size": 36, "flags": {}},
        {"id": 3, "title": "3. Advertisement Director + typed H3 plan", "bounding": [-1950, -2800, 6500, 1750], "color": "#8064b4", "font_size": 36, "flags": {}},
        {"id": 4, "title": "4. Four-reference H3 lanes + persistent retained-tail relay", "bounding": [2700, -1400, 4300, 5200], "color": "#b88632", "font_size": 36, "flags": {}},
        {"id": 5, "title": "5. Deterministic end card, exact master, real cutdowns, honest QA", "bounding": [6500, -2700, 3400, 2050], "color": "#4d9a75", "font_size": 36, "flags": {}},
    ]
    workflow = {
        "id": WORKFLOW_UUID,
        "revision": 1,
        "last_node_id": note_id,
        "last_link_id": next_link,
        "nodes": nodes,
        "links": links,
        "groups": groups,
        "config": {},
        "extra": {
            SCHEMA: {
                "version": VERSION,
                "known_good_v6_file_sha256": KNOWN_GOOD_UI_SHA256,
                "known_good_v6_is_never_overwritten": True,
                "field_test_api_canonical_sha256": FIELD_TEST_API_CANONICAL_SHA256,
                "music3_api_canonical_sha256": MUSIC3_API_CANONICAL_SHA256,
                "workflow_uuid_is_fresh": True,
                "graph_form": "flattened_independent_no_shared_subgraph_uuid",
                "audio_source_default": "MiniMax Music 3",
                "fallbacks": ["Upload song", "Legacy ACE-Step"],
                "easycache_default": "absent_measured_1.00x",
                "torchcompile_default": "absent_outside_baseline",
                "advertisement_memory_barriers": 5,
            },
            "workflow_note": (
                "Separate Advertisement workflow: MiniMax Music 3 default; independent P1/P2 performer and P3 product roles; "
                "persistent P4 retained-tail relay; deterministic copy finishing; exact 30/15/6 deliverables; V6 untouched."
            ),
        },
        "version": 0.4,
    }
    validate_workflow(workflow)
    return workflow


def _origin(workflow: Mapping[str, Any], node: Mapping[str, Any], input_name: str) -> Mapping[str, Any]:
    slots = [index for index, item in enumerate(node.get("inputs", [])) if item.get("name") == input_name]
    if len(slots) != 1:
        raise WorkflowError(f"Expected one input {node.get('id')}:{input_name}.")
    link_id = node["inputs"][slots[0]].get("link")
    link = next((item for item in workflow.get("links", []) if int(item[0]) == int(link_id)), None)
    if link is None:
        raise WorkflowError(f"Missing link for {node.get('id')}:{input_name}.")
    origin = next((item for item in workflow.get("nodes", []) if int(item["id"]) == int(link[1])), None)
    if origin is None:
        raise WorkflowError("Workflow link has a missing origin node.")
    return origin


def validate_workflow(workflow: Mapping[str, Any]) -> None:
    if workflow.get("id") != WORKFLOW_UUID or int(workflow.get("revision", 0)) != 1:
        raise WorkflowError("Advertisement workflow must keep its fresh revision-1 UUID.")
    if workflow.get("id") == _load_guarded(KNOWN_GOOD_UI, byte_hash=KNOWN_GOOD_UI_SHA256).get("id"):
        raise WorkflowError("Advertisement workflow reused the known-good V6 UUID.")
    nodes = list(workflow.get("nodes", []))
    types = [str(node.get("type", "")) for node in nodes]
    for required in (
        "DiffusionGemmaAdvertisementWorkflowControls",
        "DiffusionGemmaAdvertisementCampaignContract",
        "DiffusionGemmaAdvertisementReferenceAssetPrep",
        "DiffusionGemmaAdvertisementSoundtrackSourceRouter",
        "MiniMaxMusic3TextEncode",
        "DiffusionGemmaAdvertisementMultiShotPlanner",
        "DiffusionGemmaAdvertisementRelayArtifact",
        "DiffusionGemmaAdvertisementEndCardRenderer",
        "DiffusionGemmaAdvertisementCutdownRenderer",
        "DiffusionGemmaAdvertisementMediaQAGate",
    ):
        if required not in types:
            raise WorkflowError(f"Advertisement UI workflow is missing {required}.")
    forbidden = sorted(set(types).intersection(FORBIDDEN_RUNTIME_CLASSES))
    if forbidden:
        raise WorkflowError(f"Forbidden acceleration/convenience runtime nodes entered the Advertisement UI baseline: {forbidden}")
    if types.count("DiffusionGemmaAdvertisementMemoryBarrier") != 5:
        raise WorkflowError("Advertisement UI workflow must retain five self-contained memory barriers.")
    if types.count("SaveVideo") != 3:
        raise WorkflowError("Advertisement UI workflow must save master, 15s, and 6s videos.")
    if types.count("DiffusionGemmaAdvertisementWorkflowControls") != 1:
        raise WorkflowError("Advertisement UI workflow must have one shared settings authority.")
    by_api_id = {
        str(node.get("properties", {}).get("diffusiongemma_api_node_id", "")): node
        for node in nodes
    }
    by_id = {int(node["id"]): node for node in nodes}
    for link in workflow.get("links", []):
        origin = by_id.get(int(link[1]))
        target = by_id.get(int(link[3]))
        if origin is None or target is None:
            raise WorkflowError("Advertisement UI link references a missing node.")
        origin_type = str(origin["outputs"][int(link[2])].get("type", "*"))
        target_type = str(target["inputs"][int(link[4])].get("type", "*"))
        link_type = str(link[5])
        if origin_type != "*" and target_type != "*" and origin_type != target_type:
            raise WorkflowError(
                f"Advertisement UI link type mismatch: {origin_type} -> {target_type} "
                f"at {origin.get('type')}:{link[2]} -> {target.get('type')}:{link[4]}."
            )
        if link_type != origin_type and origin_type != "*":
            raise WorkflowError("Advertisement UI link metadata does not match its origin socket type.")
    for api_id, input_name in (
        ("801", "production_duration_seconds"),
        ("804", "target_duration_seconds"),
        ("673", "target_duration_seconds"),
        ("676", "production_duration_seconds"),
        ("815", "excerpt_duration_seconds"),
        ("816", "duration"),
        ("817", "target_duration_seconds"),
        ("818", "excerpt_duration_seconds"),
        ("673", "custom_shot_count"),
        ("821", "native_shot_count"),
        ("675", "performance_mode"),
        ("823", "performance_mode"),
        ("835", "cutdown_15_start_seconds"),
        ("835", "cutdown_6_start_seconds"),
        ("801", "deliverables_json"),
        ("676", "deliverables_json"),
        ("801", "aspect_ratio"),
        ("676", "aspect_ratio"),
        ("673", "generation_mode"),
        ("673", "shot_count"),
        ("673", "shot_count_override"),
        ("673", "audio_mode"),
        ("673", "dialogue_mode"),
        ("676", "excerpt_start_seconds"),
        ("676", "generation_model"),
        ("676", "max_h3_shot_seconds"),
        ("187", "resolution_aspect_ratio_override"),
    ):
        if api_id not in by_api_id or _origin(workflow, by_api_id[api_id], input_name).get("type") != "DiffusionGemmaAdvertisementWorkflowControls":
            raise WorkflowError(f"Advertisement UI shared control drifted at {api_id}:{input_name}.")
    router = next(node for node in nodes if node.get("type") == "DiffusionGemmaAdvertisementSoundtrackSourceRouter")
    if not router.get("widgets_values") or router["widgets_values"][0] != "MiniMax Music 3":
        raise WorkflowError("MiniMax Music 3 must remain the saved soundtrack default.")
    for name in ("music3_candidate_1", "uploaded_audio", "ace_candidate_1"):
        _origin(workflow, router, name)
    h3_nodes = [node for node in nodes if node.get("type") == "MiniMaxH3ReferenceToVideo"]
    if len(h3_nodes) != 4:
        raise WorkflowError("Advertisement UI workflow must retain four lazy H3 generation lanes.")
    h3_nodes.sort(key=lambda node: float(node.get("pos", [0, 0])[1]))
    for lane, node in enumerate(h3_nodes, start=1):
        for name in ("ref_images.ref_image_0", "ref_images.ref_image_1", "ref_images.ref_image_2"):
            _origin(workflow, node, name)
        relay_slots = [item for item in node.get("inputs", []) if item.get("name") == "ref_images.ref_image_3"]
        if lane == 1 and relay_slots:
            raise WorkflowError("Lane 1 cannot consume relay Picture 4.")
        if lane > 1 and len(relay_slots) != 1:
            raise WorkflowError(f"Lane {lane} must consume persistent relay Picture 4.")
    marker = workflow.get("extra", {}).get(SCHEMA, {})
    if marker.get("known_good_v6_file_sha256") != KNOWN_GOOD_UI_SHA256 or not marker.get("known_good_v6_is_never_overwritten"):
        raise WorkflowError("Advertisement UI lineage guard is incomplete.")


def _write_guarded(path: Path, payload: bytes, *, force: bool) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    if existed:
        current = path.read_bytes()
        if current == payload:
            return "verified"
        if not force:
            raise WorkflowError(f"Refusing to overwrite a different Advertisement workflow: {path}")
        try:
            existing = json.loads(current.decode("utf-8"))
            marker = existing.get("extra", {}).get(SCHEMA, {})
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError) as exc:
            raise WorkflowError(f"Refusing to replace an unrecognized artifact: {path}") from exc
        if marker.get("version") != VERSION or existing.get("id") != WORKFLOW_UUID:
            raise WorkflowError(f"Refusing to replace an artifact without Advertisement ownership: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return "replaced" if existed else "created"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-info-url", default="http://127.0.0.1:8191")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    _load_guarded(KNOWN_GOOD_UI, byte_hash=KNOWN_GOOD_UI_SHA256)
    field_test = _load_guarded(FIELD_TEST_API, canonical_hash=FIELD_TEST_API_CANONICAL_SHA256)
    music3 = _load_guarded(MUSIC3_API, canonical_hash=MUSIC3_API_CANONICAL_SHA256)
    graph = _with_lazy_fallbacks(music3, field_test)
    workflow = build_workflow(graph, _object_info(args.object_info_url))
    payload = _json_bytes(workflow)
    summary = {
        "workflow_id": workflow["id"],
        "nodes": len(workflow["nodes"]),
        "links": len(workflow["links"]),
        "sha256": _sha256_bytes(payload),
        "canonical_sha256": _canonical_sha256(workflow),
        "known_good_v6_sha256": KNOWN_GOOD_UI_SHA256,
    }
    if args.check:
        print(json.dumps(summary, sort_keys=True))
        return 0
    summary["action"] = _write_guarded(args.output, payload, force=args.force)
    summary["path"] = str(args.output.resolve())
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
