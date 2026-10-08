#!/usr/bin/env python3
"""Repair and expand the organized ACE-Step/MiniMax-H3 music-video graph.

This migration is deliberately graph-aware.  It preserves the existing ACE
audition/QC subgraph and all user controls, repairs the H3 Ref2VA reference
contract, builds four lazily evaluated H3 generation lanes, and muxes the
untouched ACE master only after exact-frame assembly.  A sibling backup is
created before the first write, writes are atomic, and a source digest check
prevents the tool from overwriting a workflow that changed while it was being
migrated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


MIGRATION_SCHEMA = "diffusiongemma.minimax_music_video_expansion"
LEGACY_MIGRATION_VERSION = 1
MIGRATION_VERSION = 2
BACKUP_SUFFIX = ".pre_minimax_expansion_v2.bak"
DEFAULT_WORKFLOW = Path(
    r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized.json"
)

PROJECT_NODE_TYPE = "DiffusionGemmaProjectMasterContract"
PLANNER_NODE_TYPE = "DiffusionGemmaAudioAwareMultiShotPlanner"
SEED_NODE_TYPE = "DiffusionGemmaH3ShotSeedFanout"
ASSEMBLER_NODE_TYPE = "DiffusionGemmaH3ShotAssembler"
DELIVERY_NODE_TYPE = "DiffusionGemmaMultiFormatDeliveryPlanner"
H3_CONTEXT_NODE_TYPE = "DiffusionGemmaH3ReferenceContext"
PERFORMANCE_MODE_NODE_TYPE = "DiffusionGemmaMusicVideoPerformanceMode"
PERFORMANCE_MODES = (
    "Dance / music sync",
    "Lyrics + lip sync",
    "Natural / audio-led sync",
)
# Dance and Lyrics retain their historical serialized indices; newly authored
# controls fall back to the appended audio-led mode.
DEFAULT_PERFORMANCE_MODE = PERFORMANCE_MODES[-1]
PERFORMANCE_MODE_MARKER = {
    "node_type": PERFORMANCE_MODE_NODE_TYPE,
    "default": DEFAULT_PERFORMANCE_MODE,
    "modes": list(PERFORMANCE_MODES),
    "existing_selection_policy": "preserve any supported saved selection",
    "lyrics_policy": (
        "audio-gated lexical/pronunciation hints only; lyrics are not a timing schedule"
    ),
}
BOUNDARY_POLICY_MARKER = {
    "revision": 2,
    "priority": [
        "native_measured",
        "native_only",
        "mixed_native_or_measured",
        "duration_balanced",
    ],
    "missing_ideal_cut_evidence": "advisory_not_blocking",
    "non_native_seam_policy": (
        "carry_active_source_shot_and_preserve_later_timestamps"
    ),
    "hard_failures": [
        "invalid_project_or_excerpt_clock",
        "malformed_or_nonconsecutive_native_shot_blocks",
        "non_monotonic_or_out_of_range_native_timestamps",
        "stale_or_mismatched_measured_audio_report",
        "invalid_identity_reference_manifest",
        "duration_outside_generation_lane_capacity",
        "required_generation_lane_count_exceeds_four",
    ],
}

MAX_GENERATION_LANES = 4
# Serialized node sockets retain their historical ``shot_*`` names so saved
# workflows and API prompts do not shift.  In v2 those sockets represent
# separate H3 generation lanes, while native ``[Shot N]`` blocks remain inside
# each lane's prompt.
MAX_SHOTS = MAX_GENERATION_LANES
FPS = 24.0
REFERENCE_MANIFEST = (
    "<Picture 1>: [dg:identity,appearance,color,environment,lighting,composition] "
    "sole visual reference for visible subject, wardrobe, scene, palette, and "
    "composition; preserve source-grounded identity and appearance across every shot.\n"
    "<Audio 1>: locked selected ACE soundtrack excerpt; use its timing, vocal "
    "phrasing, beat, dynamics, and dance or lip synchronization as performance "
    "authority; do not replace or reinterpret the composition."
)

START_NOTE_V1 = """# DiffusionGemma + ACE-Step + MiniMax-H3 — locked-song multi-shot production

1. Replace the reference image and edit the creative brief. **Project Master Contract** is the production source of truth for duration, master aspect, shot budget, continuity, and delivery intent.
2. **Song plan / audition root** changes only music candidates. The separate **H3 root seed** changes only video sampling. Changing duration or the H3 seed does not alter a hash-locked song.
3. ACE-Step composes and auditions up to four candidates, then waveform QC selects a musically coherent winner and a safe excerpt. The exact decoded excerpt remains the delivery soundtrack.
4. **H3 performance mode** defaults to dance/music synchronization. Lyrics are sent to the audio-aware shot planner only in lyric/lip-sync mode; each shot receives a bounded excerpt rather than the full song's lyric density.
5. The H3 planner divides longer productions into native 5–15 second Ref2VA shots, aligns cuts to measured musical recovery points when available, and emits H3-valid frame counts. Shot timestamps are relative to the locked excerpt; the plan also records absolute song time.
6. Every H3 shot uses `<Picture 1>` as identity/source authority and `<Audio 1>` as the shot's sync-safe audio guide. From shot 2 onward, `<Picture 2>` is the previous shot's last frame and is continuity guidance only.
7. H3's jointly generated audio is intentionally discarded. **Locked-song assembly** concatenates and trims the generated frames to the exact master frame count, then muxes the pristine ACE excerpt unchanged.
8. **Multi-format delivery** reports honest framing/export instructions. It does not claim that alternate aspect renders exist until those variants are actually generated.

The dependable path is: project contract → ACE audition/QC → measured-audio H3 Director → audio-aware shot plan → lazy H3 shots → exact-frame assembly → pristine-song mux.
"""

START_NOTE = """# DiffusionGemma + ACE-Step + MiniMax-H3 — locked-song multi-shot production

1. Replace the reference image and edit the creative brief. **Project Master Contract** is the production source of truth for duration, master aspect, H3 generation-lane budget, continuity, and delivery intent.
2. **Song plan / audition root** changes only music candidates. The separate **H3 root seed** changes only video sampling. Changing duration or the H3 seed does not alter a hash-locked song.
3. ACE-Step composes and auditions up to four candidates, then waveform QC selects a musically coherent winner and a safe excerpt. The exact decoded excerpt remains the delivery soundtrack.
4. **H3 performance mode** defaults to **Natural / audio-led sync**, which lets the locked soundtrack lead motion without adding lyrics or a forced mouth schedule. **Dance / music sync** requests deliberate whole-body beat and phrase synchronization. **Lyrics + lip sync** supplies bounded words as lexical and pronunciation hints only; lyrics are not a timing schedule, and verified audio timing remains authoritative.
5. One H3 generation lane is one model invocation of at most 15 seconds. A lane may preserve several native `[Shot N]` blocks and their internal cut timestamps; native shot count is controlled by the H3 Target Profile, independently from the number of generation lanes.
6. Productions longer than one H3 invocation are divided into up to four generation lanes. Seams prefer native shot boundaries with measured musical recovery, then native-only or mixed evidence; if those grids cannot satisfy the hard duration limits, deterministic balanced seams are used. A non-native seam carries the active source shot into the next lane so later timestamps remain truthful. Missing ideal cut evidence is advisory and never blocks generation. Lane starts are relative to the locked excerpt, while the plan also records absolute song time.
7. Every lane uses `<Picture 1>` as identity/source authority and `<Audio 1>` as its sync-safe audio guide. From generation lane 2 onward, `<Picture 2>` is the previous lane's retained final frame and is continuity guidance only.
8. H3's jointly generated audio is intentionally discarded. **Locked-song assembly** concatenates and trims the generated frames to the exact master frame count, then muxes the pristine ACE excerpt unchanged.
9. **Multi-format delivery** reports honest framing/export instructions. It does not claim that alternate aspect renders exist until those variants are actually generated.

**Director cache:** Leave it on **reuse**. **refresh** is a one-run diagnostic and must be returned to **reuse** after that queue.

The dependable path is: project contract → ACE audition/QC → measured-audio H3 Director → audio-aware generation-lane plan → lazy H3 lanes → exact-frame assembly → pristine-song mux.
"""


class WorkflowError(RuntimeError):
    """Raised when the workflow cannot be changed without guessing."""


class IdAllocator:
    def __init__(self, workflow: dict[str, Any]) -> None:
        node_ids = [int(node["id"]) for node in workflow.get("nodes", [])]
        link_ids = [int(link[0]) for link in workflow.get("links", [])]
        for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
            node_ids.extend(int(node["id"]) for node in subgraph.get("nodes", []))
            link_ids.extend(int(link["id"]) for link in subgraph.get("links", []))
        self._node = max(node_ids + [int(workflow.get("last_node_id", 0) or 0)])
        self._link = max(link_ids + [int(workflow.get("last_link_id", 0) or 0)])

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


class MainGraph:
    def __init__(self, workflow: dict[str, Any], ids: IdAllocator) -> None:
        self.workflow = workflow
        self.nodes: list[dict[str, Any]] = workflow["nodes"]
        self.links: list[list[Any]] = workflow["links"]
        self.ids = ids

    def node(self, node_id: int) -> dict[str, Any]:
        matches = [node for node in self.nodes if int(node["id"]) == int(node_id)]
        if len(matches) != 1:
            raise WorkflowError(f"Expected one node {node_id}; found {len(matches)}.")
        return matches[0]

    def one(self, node_type: str) -> dict[str, Any]:
        matches = [node for node in self.nodes if node.get("type") == node_type]
        if len(matches) != 1:
            raise WorkflowError(f"Expected one {node_type}; found {len(matches)}.")
        return matches[0]

    @staticmethod
    def input_slot(node: dict[str, Any], name: str) -> int:
        matches = [
            index for index, item in enumerate(node.get("inputs", []))
            if item.get("name") == name
        ]
        if len(matches) != 1:
            raise WorkflowError(
                f"Node {node.get('id')} ({node.get('type')}) requires one input {name!r}."
            )
        return matches[0]

    @staticmethod
    def output_slot(node: dict[str, Any], name: str) -> int:
        matches = [
            index for index, item in enumerate(node.get("outputs", []))
            if item.get("name") == name
        ]
        if len(matches) != 1:
            raise WorkflowError(
                f"Node {node.get('id')} ({node.get('type')}) requires one output {name!r}."
            )
        return matches[0]

    def remove_link(self, link_id: int) -> None:
        matches = [link for link in self.links if int(link[0]) == int(link_id)]
        if not matches:
            return
        if len(matches) != 1:
            raise WorkflowError(f"Duplicate link id {link_id} prevents safe removal.")
        link = matches[0]
        _, origin_id, origin_slot, target_id, target_slot, _ = link
        origin = self.node(int(origin_id))
        target = self.node(int(target_id))
        output = origin["outputs"][int(origin_slot)]
        output["links"] = [
            value for value in (output.get("links") or [])
            if int(value) != int(link_id)
        ]
        target_input = target["inputs"][int(target_slot)]
        if target_input.get("link") == link_id:
            target_input["link"] = None
        self.links.remove(link)

    def remove_node(self, node: dict[str, Any]) -> None:
        node_id = int(node["id"])
        for link in list(self.links):
            if int(link[1]) == node_id or int(link[3]) == node_id:
                self.remove_link(int(link[0]))
        self.nodes.remove(node)

    def replace(self, old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        if int(old["id"]) != int(new["id"]):
            raise WorkflowError("Replacement nodes must retain the original id.")
        for link in list(self.links):
            if int(link[1]) == int(old["id"]) or int(link[3]) == int(old["id"]):
                self.remove_link(int(link[0]))
        index = self.nodes.index(old)
        self.nodes[index] = new
        return new

    def add(self, node: dict[str, Any]) -> dict[str, Any]:
        if any(int(item["id"]) == int(node["id"]) for item in self.nodes):
            raise WorkflowError(f"Duplicate node id {node['id']}.")
        node["order"] = max(
            (int(item.get("order", 0)) for item in self.nodes), default=0
        ) + 1
        self.nodes.append(node)
        return node

    def connect(
        self,
        origin: dict[str, Any],
        output_name: str,
        target: dict[str, Any],
        input_name: str,
        link_type: str | None = None,
    ) -> int:
        output_slot = self.output_slot(origin, output_name)
        input_slot = self.input_slot(target, input_name)
        existing = target["inputs"][input_slot].get("link")
        if existing is not None:
            self.remove_link(int(existing))
        if link_type is None:
            link_type = str(
                target["inputs"][input_slot].get("type")
                or origin["outputs"][output_slot].get("type")
            )
        link_id = self.ids.link()
        link = [
            link_id,
            int(origin["id"]),
            output_slot,
            int(target["id"]),
            input_slot,
            link_type,
        ]
        self.links.append(link)
        target["inputs"][input_slot]["link"] = link_id
        output_links = origin["outputs"][output_slot].get("links")
        if not isinstance(output_links, list):
            output_links = []
            origin["outputs"][output_slot]["links"] = output_links
        output_links.append(link_id)
        return link_id


def _input(
    name: str,
    type_name: str,
    *,
    widget: bool = False,
    optional: bool = False,
) -> dict[str, Any]:
    value: dict[str, Any] = {"name": name, "type": type_name, "link": None}
    if widget:
        value["widget"] = {"name": name}
    if optional:
        value["shape"] = 7
    return value


def _output(name: str, type_name: str) -> dict[str, Any]:
    return {"name": name, "type": type_name, "links": []}


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
    node_id: int | None = None,
    cnr_id: str | None = "diffusiongemma-prompt-builder",
) -> dict[str, Any]:
    properties: dict[str, Any] = {"Node name for S&R": node_type}
    if cnr_id:
        properties["cnr_id"] = cnr_id
    return {
        "id": ids.node() if node_id is None else int(node_id),
        "type": node_type,
        "pos": [float(pos[0]), float(pos[1])],
        "size": [float(size[0]), float(size[1])],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "title": title,
        "properties": properties,
        "widgets_values": list(widgets or []),
    }


def _clone_reset(
    ids: IdAllocator,
    template: dict[str, Any],
    *,
    title: str,
    pos: tuple[float, float],
) -> dict[str, Any]:
    node = copy.deepcopy(template)
    node["id"] = ids.node()
    node["title"] = title
    node["pos"] = [float(pos[0]), float(pos[1])]
    node["order"] = 0
    for item in node.get("inputs", []):
        item["link"] = None
    for item in node.get("outputs", []):
        item["links"] = []
    return node


def _preview_node(
    ids: IdAllocator,
    title: str,
    pos: tuple[float, float],
) -> dict[str, Any]:
    return _node(
        ids,
        "PreviewAny",
        title,
        pos,
        (420.0, 180.0),
        [_input("source", "*", optional=True)],
        [],
        [],
        cnr_id="comfyui-easy-use",
    )


def _note_node(
    ids: IdAllocator,
    title: str,
    text: str,
    pos: tuple[float, float],
    size: tuple[float, float],
) -> dict[str, Any]:
    return _node(
        ids,
        "MarkdownNote",
        title,
        pos,
        size,
        [],
        [],
        [text],
        cnr_id="comfy-core",
    )


def _sensitive_key_paths(value: Any, prefix: str = "") -> list[str]:
    hits: list[str] = []
    sensitive_names = {
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
            if str(key).casefold() in sensitive_names and item not in (None, "", False):
                hits.append(path)
            hits.extend(_sensitive_key_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(_sensitive_key_paths(item, f"{prefix}[{index}]"))
    return hits


def _all_nodes(workflow: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield from workflow.get("nodes", [])
    for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
        yield from subgraph.get("nodes", [])


def _node_in_any_scope(workflow: dict[str, Any], node_id: int) -> dict[str, Any]:
    matches = [
        node for node in _all_nodes(workflow) if int(node.get("id", -1)) == int(node_id)
    ]
    if len(matches) != 1:
        raise WorkflowError(
            f"Expected one node {node_id} across the workflow; found {len(matches)}."
        )
    return matches[0]


def _subgraph_containing_node(
    workflow: dict[str, Any], node_id: int
) -> dict[str, Any] | None:
    matches = [
        subgraph
        for subgraph in workflow.get("definitions", {}).get("subgraphs", [])
        if any(int(node.get("id", -1)) == int(node_id) for node in subgraph.get("nodes", []))
    ]
    if len(matches) > 1:
        raise WorkflowError(f"Node {node_id} is duplicated across subgraphs.")
    return matches[0] if matches else None


def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reset_node(
    graph: MainGraph,
    node: dict[str, Any],
    *,
    title: str,
    pos: tuple[float, float],
) -> dict[str, Any]:
    node_id = int(node["id"])
    for link in list(graph.links):
        if int(link[1]) == node_id or int(link[3]) == node_id:
            graph.remove_link(int(link[0]))
    for item in node.get("inputs", []):
        item["link"] = None
    for item in node.get("outputs", []):
        item["links"] = []
    node["title"] = title
    node["pos"] = [float(pos[0]), float(pos[1])]
    node["mode"] = 0
    return node


def _require_baseline(workflow: dict[str, Any]) -> None:
    """Fail closed when the attached graph is no longer recognizable."""

    if not isinstance(workflow.get("nodes"), list) or not isinstance(
        workflow.get("links"), list
    ):
        raise WorkflowError("The workflow has no editable main graph.")
    expected_by_id = {
        116: "MarkdownNote",
        177: "PrimitiveStringMultiline",
        178: "PrimitiveFloat",
        179: "LoadImage",
        182: "DiffusionGemmaModelLoader",
        183: "DiffusionGemmaContextHub",
        185: "DiffusionGemmaGroundingGuardSettings",
        186: "DiffusionGemmaCoTGenerator",
        187: "DiffusionGemmaJSONSplitter",
        188: "DiffusionGemmaBranchGenerationGate",
        418: "SeedNode",
        556: "SplatStageBlueprintRouter",
        562: "PrimitiveFloat",
        569: "TrimAudioDuration",
        578: "DiffusionGemmaSplatStagePlanner",
        589: "DiffusionGemmaMusicProductionConcept",
        610: "DiffusionGemmaAudioCandidateSelector",
        611: "DiffusionGemmaLTXAudioGuide",
        612: "easy cleanGpuUsed",
        613: "PreviewAudio",
        621: "DiffusionGemmaLTXPerformancePrompt",
        622: "PreviewAny",
        624: "PrimitiveStringMultiline",
        651: "SaveVideo",
        653: "VAELoader",
        654: "VAELoader",
        655: "VAEDecodeAudio",
        656: "VAEDecode",
        657: "KSamplerSelect",
        658: "BasicScheduler",
        659: "SamplerCustomAdvanced",
        660: "BasicGuider",
        661: "UNETLoader",
        662: "CLIPLoader",
        663: "RandomNoise",
        664: "CreateVideo",
        665: "ComfyMathExpression",
        667: "MiniMaxH3ReferenceToVideo",
        669: "PrimitiveStringMultiline",
        670: "RHMiniMaxH3SageAttentionPatch",
        672: "Reroute",
        673: "DiffusionGemmaMiniMaxH3TargetProfile",
    }
    nodes_by_id = {int(node["id"]): node for node in workflow["nodes"]}
    for node_id, expected_type in expected_by_id.items():
        actual = nodes_by_id.get(node_id, {}).get("type")
        if actual != expected_type:
            raise WorkflowError(
                f"Baseline preflight expected node {node_id} to be {expected_type}; "
                f"found {actual!r}."
            )
    marker = workflow.get("extra", {}).get("diffusiongemma.music_video_production")
    if not isinstance(marker, dict) or int(marker.get("version", 0)) != 10:
        raise WorkflowError(
            "The attached workflow must first satisfy music-video production v10."
        )
    if any(node.get("type") == PROJECT_NODE_TYPE for node in workflow["nodes"]):
        raise WorkflowError(
            "An unmarked Project Master Contract already exists; refusing to guess "
            "whether a partial migration should be overwritten."
        )


def _normalize_h3_reference_node(node: dict[str, Any]) -> None:
    """Keep one Picture 1, optional Picture 2, and exactly one Audio 1 lane."""

    wanted = {
        "clip",
        "vae",
        "audio_vae",
        "ref_images.ref_image_0",
        "ref_images.ref_image_1",
        "ref_videos.ref_video_0",
        "ref_video_audios.ref_video_audio_0",
        "ref_audios.ref_audio_0",
        "prompt",
        "width",
        "height",
        "length",
    }
    node["inputs"] = [
        item for item in node.get("inputs", []) if item.get("name") in wanted
    ]
    actual = [item.get("name") for item in node["inputs"]]
    expected = [
        "clip",
        "vae",
        "audio_vae",
        "ref_images.ref_image_0",
        "ref_images.ref_image_1",
        "ref_videos.ref_video_0",
        "ref_video_audios.ref_video_audio_0",
        "ref_audios.ref_audio_0",
        "prompt",
        "width",
        "height",
        "length",
    ]
    if actual != expected:
        raise WorkflowError(
            f"MiniMaxH3ReferenceToVideo inputs changed; expected {expected}, found {actual}."
        )
    for item in node["inputs"]:
        item["link"] = None
    for item in node.get("outputs", []):
        item["links"] = []
    node["widgets_values"] = ["", 1344, 768, 124, "match"]


def _trim_node(
    ids: IdAllocator,
    template: dict[str, Any],
    shot: int,
    pos: tuple[float, float],
) -> dict[str, Any]:
    node = _clone_reset(
        ids,
        template,
        title=f"H3 GENERATION LANE {shot} AUDIO — relative sync-safe guide slice",
        pos=pos,
    )
    node["widgets_values"] = [0.0, 15.0]
    return node


def _tail_node(
    ids: IdAllocator,
    shot: int,
    pos: tuple[float, float],
) -> dict[str, Any]:
    return _node(
        ids,
        "ImageFromBatch",
        f"LANE {shot} TAIL — Picture 2 continuity for generation lane {shot + 1}",
        pos,
        (330.0, 110.0),
        [
            _input("image", "IMAGE"),
            _input("batch_index", "INT", widget=True),
            _input("length", "INT", widget=True),
        ],
        [_output("IMAGE", "IMAGE")],
        [0, 1],
        cnr_id="comfy-core",
    )


def _tail_index_node(
    ids: IdAllocator,
    template: dict[str, Any],
    shot: int,
    pos: tuple[float, float],
    *,
    reuse_template_id: bool,
) -> dict[str, Any]:
    node = copy.deepcopy(template)
    if not reuse_template_id:
        node["id"] = ids.node()
    node["title"] = f"LANE {shot} RETAINED CUT — continuity frame index"
    node["pos"] = [float(pos[0]), float(pos[1])]
    node["mode"] = 0
    node["widgets_values"] = [
        "max(0, round((a + b) * 24) - round(a * 24) - 1)"
    ]
    for item in node.get("inputs", []):
        item["link"] = None
    for item in node.get("outputs", []):
        item["links"] = []
    return node


def _context_node(ids: IdAllocator) -> dict[str, Any]:
    return _node(
        ids,
        H3_CONTEXT_NODE_TYPE,
        "5a. MiniMax-H3 ordered reference context — Picture 1 + Audio 1",
        (-170.0, -1160.0),
        (520.0, 650.0),
        [
            _input("user_prompt", "STRING"),
            _input("reference_images", "IMAGE", optional=True),
            _input("reference_video", "VIDEO", optional=True),
        ],
        [
            _output("gemma_context", "DG_CONTEXT"),
            _output("context_json", "STRING"),
            _output("context_preview", "STRING"),
        ],
        [REFERENCE_MANIFEST, 1.0, 60, 60, "trim", "", "custom", 0],
    )


def _ensure_splitter_aspect_override(splitter: dict[str, Any]) -> None:
    matches = [
        item for item in splitter.get("inputs", [])
        if item.get("name") == "resolution_aspect_ratio_override"
    ]
    if not matches:
        splitter.setdefault("inputs", []).append(
            _input("resolution_aspect_ratio_override", "STRING", optional=True)
        )
    elif len(matches) != 1:
        raise WorkflowError("JSON Splitter has duplicate aspect override inputs.")


def _ensure_target_shot_override(target: dict[str, Any]) -> None:
    """Keep the compatibility socket without assigning generation-lane count to it."""

    matches = [
        item for item in target.get("inputs", [])
        if item.get("name") == "shot_count_override"
    ]
    if not matches:
        target.setdefault("inputs", []).append(
            _input("shot_count_override", "INT", widget=True, optional=True)
        )
    elif len(matches) != 1:
        raise WorkflowError("H3 target profile has duplicate shot-count override inputs.")


def _require_target_shot_override_unconnected(target: dict[str, Any]) -> None:
    _ensure_target_shot_override(target)
    slot = MainGraph.input_slot(target, "shot_count_override")
    if target["inputs"][slot].get("link") is not None:
        raise WorkflowError(
            "H3 Target Profile shot_count_override controls native [Shot N] blocks "
            "and must not be driven by the Project Master's generation-lane count."
        )
    widgets = list(target.get("widgets_values") or [])
    if len(widgets) < 11:
        raise WorkflowError("H3 target profile is missing required widget state.")
    if len(widgets) == 11:
        # Connected v1 workflows did not serialize the hidden override widget.
        # Once disconnected, materialize its neutral value without changing any
        # visible native-shot choice.
        widgets.append(0)
        target["widgets_values"] = widgets


def _ensure_performance_mode_output(performance: dict[str, Any]) -> None:
    matches = [
        item for item in performance.get("outputs", [])
        if item.get("name") == "performance_mode"
    ]
    if not matches:
        performance.setdefault("outputs", []).append(
            _output("performance_mode", "STRING")
        )
    elif len(matches) != 1:
        raise WorkflowError("Performance selector has duplicate mode outputs.")


def _ensure_performance_mode_override_input(performance: dict[str, Any]) -> None:
    matches = [
        item for item in performance.get("inputs", [])
        if item.get("name") == "performance_mode_override"
    ]
    if not matches:
        performance.setdefault("inputs", []).append(
            _input("performance_mode_override", "STRING", optional=True)
        )
    elif len(matches) != 1:
        raise WorkflowError("Performance lyric compiler has duplicate mode-override inputs.")


def _validate_link_integrity(workflow: dict[str, Any]) -> None:
    nodes = workflow.get("nodes", [])
    links = workflow.get("links", [])
    node_by_id: dict[int, dict[str, Any]] = {}
    for node in nodes:
        node_id = int(node["id"])
        if node_id in node_by_id:
            raise WorkflowError(f"Duplicate main-graph node id {node_id}.")
        node_by_id[node_id] = node

    link_by_id: dict[int, list[Any]] = {}
    expected_output_backlinks: dict[tuple[int, int], list[int]] = {}
    expected_input_backlinks: dict[tuple[int, int], int] = {}
    for link in links:
        if not isinstance(link, list) or len(link) != 6:
            raise WorkflowError(f"Malformed main-graph link: {link!r}.")
        link_id, origin_id, origin_slot, target_id, target_slot, link_type = link
        link_id = int(link_id)
        origin_id = int(origin_id)
        origin_slot = int(origin_slot)
        target_id = int(target_id)
        target_slot = int(target_slot)
        if link_id in link_by_id:
            raise WorkflowError(f"Duplicate main-graph link id {link_id}.")
        if origin_id not in node_by_id or target_id not in node_by_id:
            raise WorkflowError(f"Dangling main-graph link {link_id}.")
        origin = node_by_id[origin_id]
        target = node_by_id[target_id]
        if not 0 <= origin_slot < len(origin.get("outputs", [])):
            raise WorkflowError(f"Link {link_id} has invalid origin slot {origin_slot}.")
        if not 0 <= target_slot < len(target.get("inputs", [])):
            raise WorkflowError(f"Link {link_id} has invalid target slot {target_slot}.")
        origin_type = str(origin["outputs"][origin_slot].get("type", ""))
        target_type = str(target["inputs"][target_slot].get("type", ""))
        serialized_type = str(link_type)

        def type_tokens(value: str) -> set[str]:
            return {token.strip() for token in value.split(",") if token.strip()}

        origin_tokens = type_tokens(origin_type)
        target_tokens = type_tokens(target_type)
        link_tokens = type_tokens(serialized_type)
        wildcard = "*" in origin_tokens | target_tokens | link_tokens
        if not wildcard and (
            not origin_tokens.intersection(link_tokens)
            or not target_tokens.intersection(link_tokens)
        ):
            raise WorkflowError(
                f"Link {link_id} type {serialized_type!r} is incompatible with "
                f"{origin_type!r} -> {target_type!r}."
            )
        target_key = (target_id, target_slot)
        if target_key in expected_input_backlinks:
            raise WorkflowError(
                f"Node {target_id} input {target_slot} has multiple incoming links."
            )
        link_by_id[link_id] = link
        expected_input_backlinks[target_key] = link_id
        expected_output_backlinks.setdefault((origin_id, origin_slot), []).append(link_id)

    for node_id, node in node_by_id.items():
        for slot, item in enumerate(node.get("inputs", [])):
            actual = item.get("link")
            expected = expected_input_backlinks.get((node_id, slot))
            if actual != expected:
                raise WorkflowError(
                    f"Node {node_id} input {slot} backlink is {actual}; expected {expected}."
                )
        for slot, item in enumerate(node.get("outputs", [])):
            actual = [int(value) for value in (item.get("links") or [])]
            expected = expected_output_backlinks.get((node_id, slot), [])
            if len(actual) != len(set(actual)) or sorted(actual) != sorted(expected):
                raise WorkflowError(
                    f"Node {node_id} output {slot} backlinks {actual} do not match {expected}."
                )

    highest_node = max(node_by_id, default=0)
    highest_link = max(link_by_id, default=0)
    if int(workflow.get("last_node_id", 0) or 0) < highest_node:
        raise WorkflowError("last_node_id is lower than an active node id.")
    if int(workflow.get("last_link_id", 0) or 0) < highest_link:
        raise WorkflowError("last_link_id is lower than an active link id.")


def _input_origin(
    workflow: dict[str, Any], node: dict[str, Any], input_name: str
) -> tuple[dict[str, Any], str, list[Any]]:
    graph = MainGraph(workflow, IdAllocator(workflow))
    slot = graph.input_slot(node, input_name)
    link_id = node["inputs"][slot].get("link")
    if link_id is None:
        raise WorkflowError(
            f"Node {node['id']} ({node['type']}) input {input_name!r} is unconnected."
        )
    matches = [link for link in workflow["links"] if int(link[0]) == int(link_id)]
    if len(matches) != 1:
        raise WorkflowError(f"Input link {link_id} is absent or duplicated.")
    link = matches[0]
    origin = graph.node(int(link[1]))
    output = origin["outputs"][int(link[2])]
    return origin, str(output.get("name")), link


def _assert_origin(
    workflow: dict[str, Any],
    node: dict[str, Any],
    input_name: str,
    origin_type: str,
    output_name: str,
) -> dict[str, Any]:
    origin, actual_output, _link = _input_origin(workflow, node, input_name)
    if origin.get("type") != origin_type or actual_output != output_name:
        raise WorkflowError(
            f"Node {node['id']} input {input_name!r} must come from "
            f"{origin_type}.{output_name}; found {origin.get('type')}.{actual_output}."
        )
    return origin


def _first_widget(node: dict[str, Any], default: Any = None) -> Any:
    widgets = node.get("widgets_values")
    if isinstance(widgets, list) and widgets:
        return widgets[0]
    return default


def _aspect_from_splitter(splitter: dict[str, Any]) -> str:
    widgets = splitter.get("widgets_values") or []
    preset = str(widgets[3]) if len(widgets) > 3 else "9:16"
    token = preset.strip().split(" ", 1)[0]
    return token if token in {"16:9", "9:16", "1:1", "4:3", "3:4"} else "9:16"


def _update_visible_copy(workflow: dict[str, Any], graph: MainGraph) -> None:
    note = graph.node(116)
    note["title"] = "START HERE — lock the song, plan H3 generation lanes, then assemble"
    note["widgets_values"] = [START_NOTE]

    title_updates = {
        178: "2. Master VIDEO duration — Director + H3 + locked song excerpt",
        179: "3. Reference image — H3 Picture 1 identity authority",
        182: "4a. Load DiffusionGemma Director — unload before native H3",
        183: "4. Song-planning Context Hub — source image context",
        185: "5b. H3 Grounding Guard — audit comparison baseline",
        186: "6. Build grounded MiniMax-H3 Ref2VA production prompt",
        187: "7. Route H3 prompt + select master output resolution",
        188: "8. Isolated MiniMax-H3 prompt generation gate",
        189: "MiniMax-H3 branch gate status",
        418: "H3 root seed — video only; does not regenerate the locked song",
        557: "LEGACY SPLATSTAGE REPORT — full-song clock does not drive H3 duration",
        577: "Unload ACE models before Director and H3 conditioning",
        610: "MUSIC AUDITION + QC — choose, verify, and lock before H3",
        611: "H3 AUDIO GUIDE — sync-safe conditioning / pristine final master",
        612: "Unload ACE after QC — pass compact measurements to H3 Director",
        613: "FINAL SOUNDTRACK EXCERPT — pristine assembly audio",
        621: "INTERNAL LYRIC CANDIDATE WINDOW — mode comes from shared H3 control",
        622: "H3 PERFORMANCE LYRIC CANDIDATES — empty unless Lyrics + lip sync is selected",
        651: "FINAL MINIMAX-H3 MUSIC VIDEO — locked ACE master",
        653: "MiniMax-H3 video VAE",
        654: "MiniMax-H3 audio VAE — conditioning encoder only",
        657: "Shared H3 sampler — res_multistep",
        658: "Shared H3 schedule — 20-step baseline",
        661: "MiniMax-H3 Ref2VA diffusion model",
        662: "MiniMax-H3 Qwen vision-language encoder",
        664: "Mux exact ACE master with exact-length assembled H3 frames",
        670: "Bundled H3 SageAttention patch — safe passthrough when unavailable",
        673: "H3 TARGET PROFILE — native shots inside each generated video",
    }
    for node_id, title in title_updates.items():
        graph.node(node_id)["title"] = title

    audio_guidance = graph.node(624)
    audio_guidance["title"] = "1a. Creative brief — H3 audio/performance guidance"
    audio_guidance["widgets_values"] = [
        "Use the generated ACE-Step song as the exact soundtrack and timing "
        "authority. Synchronize body motion, scene changes, cuts, and restrained "
        "camera accents to measured rhythm, phrasing, dynamics, and recovery "
        "intervals. The audio-aware H3 generation-lane planner applies the selected Natural, "
        "Dance, or Lyrics performance contract after validation; lyric text is never a timing schedule."
    ]

    group_titles = {
        1: "1. Inputs + Picture 1 identity anchor",
        2: "2. DiffusionGemma MiniMax-H3 Ref2VA Director",
        3: "3. Verified H3 project + audio-aware generation-lane plan",
        4: "4. Native MiniMax-H3 Ref2VA generation lanes",
        35: "ACE audition, waveform QC + locked soundtrack",
        36: "Shared MiniMax-H3 models",
        37: "Lazy H3 generation-lane samplers (1–4)",
        38: "Per-lane Audio 1 + prior-tail Picture 2 continuity",
        39: "Exact-frame locked-song assembly + delivery",
    }
    group_bounds = {
        1: [-1600.0, -1750.0, 1550.0, 3000.0],
        2: [-800.0, -1100.0, 2450.0, 2100.0],
        3: [1550.0, -1550.0, 1050.0, 2250.0],
        4: [2500.0, -1050.0, 4250.0, 4650.0],
        35: [-760.0, 160.0, 3206.0, 1540.0],
        36: [2500.0, 600.0, 760.0, 900.0],
        37: [3200.0, -900.0, 1900.0, 4400.0],
        38: [2500.0, -900.0, 1050.0, 4400.0],
        39: [5050.0, -900.0, 1650.0, 2100.0],
    }
    for group in workflow.get("groups", []):
        group_id = int(group.get("id", -1))
        if group_id in group_titles:
            group["title"] = group_titles[group_id]
        if group_id in group_bounds:
            group["bounding"] = group_bounds[group_id]


def _refresh_performance_contract(workflow: dict[str, Any]) -> bool:
    """Refresh generated contract metadata/copy while preserving user choices."""

    marker = workflow.setdefault("extra", {}).get(MIGRATION_SCHEMA)
    if not isinstance(marker, dict):
        raise WorkflowError("The MiniMax expansion marker is missing or malformed.")
    changed = False
    if marker.get("performance_mode_contract") != PERFORMANCE_MODE_MARKER:
        marker["performance_mode_contract"] = copy.deepcopy(PERFORMANCE_MODE_MARKER)
        changed = True
    if marker.get("generation_lane_boundary_policy") != BOUNDARY_POLICY_MARKER:
        marker["generation_lane_boundary_policy"] = copy.deepcopy(
            BOUNDARY_POLICY_MARKER
        )
        changed = True

    graph = MainGraph(workflow, IdAllocator(workflow))
    performance_control = graph.one(PERFORMANCE_MODE_NODE_TYPE)
    previous_values = list(performance_control.get("widgets_values") or [])
    values = list(previous_values)
    if not values:
        values = [DEFAULT_PERFORMANCE_MODE]
    elif values[0] not in PERFORMANCE_MODES:
        values[0] = DEFAULT_PERFORMANCE_MODE
    if values != previous_values:
        performance_control["widgets_values"] = values
        changed = True

    legacy_note_line = (
        "4. **H3 performance mode** defaults to dance/music synchronization. Lyrics "
        "are sent to the audio-aware lane planner only in lyric/lip-sync mode; each "
        "generated lane receives a bounded excerpt rather than the full song's lyric density."
    )
    natural_note_line = (
        "4. **H3 performance mode** defaults to **Natural / audio-led sync**, which "
        "lets the locked soundtrack lead motion without adding lyrics or a forced mouth "
        "schedule. **Dance / music sync** requests deliberate whole-body beat and phrase "
        "synchronization. **Lyrics + lip sync** supplies bounded words as lexical and "
        "pronunciation hints only; lyrics are not a timing schedule, and verified audio "
        "timing remains authoritative."
    )
    note = graph.node(116)
    note_values = list(note.get("widgets_values") or [])
    refreshed_note_values = [
        value.replace(legacy_note_line, natural_note_line).replace(
            "6. Productions longer than one H3 invocation are divided into up to four generation lanes at measured musical recovery points. Lane starts are relative to the locked excerpt, while the plan also records absolute song time.",
            "6. Productions longer than one H3 invocation are divided into up to four generation lanes. Seams prefer native shot boundaries with measured musical recovery, then native-only or mixed evidence; if those grids cannot satisfy the hard duration limits, deterministic balanced seams are used. A non-native seam carries the active source shot into the next lane so later timestamps remain truthful. Missing ideal cut evidence is advisory and never blocks generation. Lane starts are relative to the locked excerpt, while the plan also records absolute song time.",
        )
        if isinstance(value, str)
        else value
        for value in note_values
    ]
    if refreshed_note_values != note_values:
        note["widgets_values"] = refreshed_note_values
        changed = True

    planner = graph.one(PLANNER_NODE_TYPE)
    if planner.get("title") in {
        "AUDIO-AWARE H3 MULTI-LANE PLAN — native shots preserved / measured lane cuts",
        "AUDIO-AWARE H3 MULTI-SHOT PLAN — measured recovery cuts",
    }:
        planner["title"] = (
            "AUDIO-AWARE H3 MULTI-LANE PLAN — evidence-preferred / deterministic fallback"
        )
        changed = True

    extra = workflow.setdefault("extra", {})
    old_workflow_notes = {
        (
            "MiniMax-H3 Ref2VA production graph: versioned Project Master Contract, "
            "locked ACE song/QC, measured-audio generation-lane planning, four lazy "
            "5–15 second H3 invocations that preserve native multi-shot prompts, "
            "retained-tail Picture 2 continuity between lanes, exact-frame assembly, "
            "pristine ACE mux audio, and honest multi-format delivery planning."
        ),
        (
            "MiniMax-H3 Ref2VA production graph: versioned Project Master Contract, "
            "locked ACE song/QC, measured-audio shot planning, four lazy 5–15 second "
            "H3 lanes with retained-cut Picture 2 continuity, exact-frame assembly, "
            "pristine ACE mux audio, and honest multi-format delivery planning."
        ),
    }
    if extra.get("workflow_note") in old_workflow_notes:
        extra["workflow_note"] = (
            "MiniMax-H3 Ref2VA production graph: versioned Project Master Contract, "
            "locked ACE song/QC, evidence-preferred generation-lane planning with a "
            "deterministic balanced fallback, four lazy 5–15 second H3 invocations "
            "that preserve native multi-shot prompts and carry active shots across "
            "non-native seams, retained-tail Picture 2 continuity between lanes, "
            "exact-frame assembly, pristine ACE mux audio, and honest multi-format "
            "delivery planning."
        )
        changed = True

    title_replacements = {
        "PERFORMANCE LYRIC WINDOW — follows shared H3 mode": (
            "INTERNAL LYRIC CANDIDATE WINDOW — mode comes from shared H3 control"
        ),
        "H3 PERFORMANCE LYRIC WINDOW — empty in dance/music-sync mode": (
            "H3 PERFORMANCE LYRIC CANDIDATES — empty unless Lyrics + lip sync is selected"
        ),
    }
    for node in workflow.get("nodes", []):
        title = str(node.get("title", ""))
        if title in title_replacements:
            node["title"] = title_replacements[title]
            changed = True

    guidance = graph.node(624)
    guidance_values = list(guidance.get("widgets_values") or [])
    refreshed_guidance_values = []
    for value in guidance_values:
        if isinstance(value, str):
            value = value.replace(
                "the selected dance or lyric/lip-sync performance contract after validation.",
                "the selected Natural, Dance, or Lyrics performance contract after "
                "validation; lyric text is never a timing schedule.",
            )
        refreshed_guidance_values.append(value)
    if refreshed_guidance_values != guidance_values:
        guidance["widgets_values"] = refreshed_guidance_values
        changed = True
    return changed


def _replace_exact(container: dict[str, Any], key: str, old: Any, new: Any) -> None:
    """Upgrade generated copy without overwriting a user's later customization."""

    if container.get(key) == old:
        container[key] = new


def _link_by_id(graph: MainGraph, link_id: int) -> list[Any]:
    matches = [link for link in graph.links if int(link[0]) == int(link_id)]
    if len(matches) != 1:
        raise WorkflowError(f"Expected one main-graph link {link_id}; found {len(matches)}.")
    return matches[0]


def _is_unmodified_legacy_lane_count_reroute(node: dict[str, Any]) -> bool:
    """Recognize only the anonymous muted reroute emitted by the v1 graph.

    This intentionally rejects titled, active, widget-bearing, or structurally
    customized reroutes.  Those may be user-authored even when they happen to
    consume the Project Master's legacy count socket.
    """

    if node.get("type") != "Reroute" or int(node.get("mode", 0) or 0) != 4:
        return False
    if node.get("title") not in (None, ""):
        return False
    if node.get("widgets_values") not in (None, []):
        return False
    inputs = node.get("inputs") or []
    outputs = node.get("outputs") or []
    if len(inputs) != 1 or len(outputs) != 1:
        return False
    if str(inputs[0].get("name", "")) or str(outputs[0].get("name", "")):
        return False
    if str(inputs[0].get("type", "")) not in {"*", "INT"}:
        return False
    if str(outputs[0].get("type", "")) != "INT":
        return False
    properties = node.get("properties")
    if properties not in (
        None,
        {},
        {"horizontal": False, "showOutputText": False},
    ):
        return False
    return True


def _remove_legacy_lane_count_override(
    graph: MainGraph,
    project: dict[str, Any],
    target: dict[str, Any],
) -> None:
    """Remove only the v1 lane-count/native-shot conflation.

    Supported v1 shapes are the direct Project-to-Target link and the exact
    anonymous muted Reroute used by the captured FAST graph.  The latter may
    still feed the Target or may have a disconnected output.  Shared or
    customized reroutes fail closed instead of being guessed away.
    """

    _ensure_target_shot_override(target)
    override_slot = graph.input_slot(target, "shot_count_override")
    project_slot = graph.output_slot(project, "recommended_h3_shot_count")
    override_link_id = target["inputs"][override_slot].get("link")

    def is_project_lane_count_link(link: list[Any], reroute: dict[str, Any]) -> bool:
        return (
            int(link[1]) == int(project["id"])
            and int(link[2]) == project_slot
            and int(link[3]) == int(reroute["id"])
            and int(link[4]) == 0
        )

    def require_removable_reroute(
        reroute: dict[str, Any], *, expected_target_link: int | None
    ) -> None:
        if not _is_unmodified_legacy_lane_count_reroute(reroute):
            raise WorkflowError(
                "Refusing to remove a customized/user-authored lane-count reroute."
            )
        input_link_id = (reroute.get("inputs") or [{}])[0].get("link")
        if input_link_id is None:
            raise WorkflowError("The legacy lane-count reroute has no Project input.")
        input_link = _link_by_id(graph, int(input_link_id))
        if not is_project_lane_count_link(input_link, reroute):
            raise WorkflowError(
                "Refusing to remove a reroute not sourced solely from Project "
                "Master recommended_h3_shot_count."
            )
        output_links = [
            int(value) for value in ((reroute.get("outputs") or [{}])[0].get("links") or [])
        ]
        if expected_target_link is None:
            if output_links:
                raise WorkflowError(
                    "Refusing to remove a shared/user-authored lane-count reroute."
                )
        elif output_links != [int(expected_target_link)]:
            raise WorkflowError(
                "Refusing to remove a shared/user-authored lane-count reroute."
            )

    if override_link_id is not None:
        override_link = _link_by_id(graph, int(override_link_id))
        origin = graph.node(int(override_link[1]))
        if (
            int(origin["id"]) == int(project["id"])
            and int(override_link[2]) == project_slot
            and int(override_link[3]) == int(target["id"])
            and int(override_link[4]) == override_slot
        ):
            graph.remove_link(int(override_link_id))
            return
        if origin.get("type") != "Reroute":
            raise WorkflowError(
                "Refusing to remove a user-authored shot-count override; the v1 "
                "upgrade only removes Project Master recommended_h3_shot_count."
            )
        if (
            int(override_link[3]) != int(target["id"])
            or int(override_link[4]) != override_slot
        ):
            raise WorkflowError("The v1 shot-count override target is malformed.")
        require_removable_reroute(
            origin, expected_target_link=int(override_link_id)
        )
        graph.remove_node(origin)
        return

    disconnected_candidates: list[dict[str, Any]] = []
    unsafe_candidates: list[dict[str, Any]] = []
    for reroute in [node for node in graph.nodes if node.get("type") == "Reroute"]:
        inputs = reroute.get("inputs") or []
        if len(inputs) != 1 or inputs[0].get("link") is None:
            continue
        input_link = _link_by_id(graph, int(inputs[0]["link"]))
        if not is_project_lane_count_link(input_link, reroute):
            continue
        outputs = reroute.get("outputs") or []
        output_links = outputs[0].get("links") if len(outputs) == 1 else None
        if not output_links and _is_unmodified_legacy_lane_count_reroute(reroute):
            disconnected_candidates.append(reroute)
        else:
            unsafe_candidates.append(reroute)

    if unsafe_candidates:
        raise WorkflowError(
            "Refusing to remove a shared/user-authored lane-count reroute."
        )
    if len(disconnected_candidates) > 1:
        raise WorkflowError(
            "Multiple disconnected legacy lane-count reroutes are ambiguous."
        )
    if disconnected_candidates:
        reroute = disconnected_candidates[0]
        require_removable_reroute(reroute, expected_target_link=None)
        graph.remove_node(reroute)


def _upgrade_v1_to_v2(workflow: dict[str, Any]) -> None:
    """Disentangle native H3 shots from duration-bounded generation lanes.

    The v1 graph connected Project Master's duration-derived count to Target
    Profile's native ``[Shot N]`` override.  V2 removes exactly that known link,
    retains every socket and widget value, and updates only unmodified v1 labels
    plus the versioned marker.
    """

    marker = workflow.get("extra", {}).get(MIGRATION_SCHEMA)
    if not isinstance(marker, dict) or int(marker.get("version", 0)) != LEGACY_MIGRATION_VERSION:
        raise WorkflowError("MiniMax expansion v1 marker is missing or malformed.")
    graph = MainGraph(workflow, IdAllocator(workflow))
    project = graph.one(PROJECT_NODE_TYPE)
    planner = graph.one(PLANNER_NODE_TYPE)
    seed_fanout = graph.one(SEED_NODE_TYPE)
    assembler = graph.one(ASSEMBLER_NODE_TYPE)
    performance = graph.one(PERFORMANCE_MODE_NODE_TYPE)
    target_id = int(marker.get("nodes", {}).get("h3_target_profile", 673))
    target = graph.node(target_id)
    if target.get("type") != "DiffusionGemmaMiniMaxH3TargetProfile":
        raise WorkflowError("MiniMax expansion v1 target-profile marker is invalid.")

    _remove_legacy_lane_count_override(graph, project, target)
    _require_target_shot_override_unconnected(target)

    node_title_updates = (
        (
            project,
            "PROJECT MASTER CONTRACT — duration, song hash, format + H3 shot budget",
            "PROJECT MASTER CONTRACT — duration, song hash, format + H3 generation-lane budget",
        ),
        (
            planner,
            "AUDIO-AWARE H3 MULTI-SHOT PLAN — measured recovery cuts",
            "AUDIO-AWARE H3 MULTI-LANE PLAN — evidence-preferred / deterministic fallback",
        ),
        (
            seed_fanout,
            "H3 SHOT SEEDS — deterministic root/project/lane fanout",
            "H3 GENERATION-LANE SEEDS — deterministic root/project/lane fanout",
        ),
        (
            performance,
            "H3 PERFORMANCE MODE — shared before Director and shot planning",
            "H3 PERFORMANCE MODE — shared before Director and generation-lane planning",
        ),
        (
            target,
            "H3 TARGET PROFILE — Project Contract controls shot count",
            "H3 TARGET PROFILE — native shots inside each generated video",
        ),
    )
    for node, old, new in node_title_updates:
        _replace_exact(node, "title", old, new)

    note = graph.node(116)
    _replace_exact(
        note,
        "title",
        "START HERE — lock the song, plan H3 shots, then assemble",
        "START HERE — lock the song, plan H3 generation lanes, then assemble",
    )
    if note.get("widgets_values") == [START_NOTE_V1]:
        note["widgets_values"] = [START_NOTE]

    guidance = graph.node(624)
    old_guidance = (
        "Use the generated ACE-Step song as the exact soundtrack and timing authority. "
        "Synchronize body motion, scene changes, cuts, and restrained camera accents "
        "to measured rhythm, phrasing, dynamics, and recovery intervals. The "
        "audio-aware H3 shot planner applies the selected dance or lyric/lip-sync "
        "performance contract after validation."
    )
    new_guidance = (
        "Use the generated ACE-Step song as the exact soundtrack and timing authority. "
        "Synchronize body motion, scene changes, cuts, and expressive, physically coherent H3 camera choreography "
        "to measured rhythm, phrasing, dynamics, and recovery intervals. Preserve explicitly requested orbiting, "
        "swirling, sweeping or whip movement, pronounced parallax, and coherent compound paths rather than "
        "downgrading them. When camera direction is unspecified, let the active Director creativity mode choose "
        "camera energy proportional to creative strength. The audio-aware H3 generation-lane planner applies the "
        "selected dance or lyric/lip-sync performance contract after validation."
    )
    if guidance.get("widgets_values") == [old_guidance]:
        guidance["widgets_values"] = [new_guidance]

    group_title_updates = {
        "3. Verified H3 project + audio-aware shot plan": (
            "3. Verified H3 project + audio-aware generation-lane plan"
        ),
        "4. Native MiniMax-H3 Ref2VA multi-shot render": (
            "4. Native MiniMax-H3 Ref2VA generation lanes"
        ),
        "Lazy H3 shot samplers (1–4)": "Lazy H3 generation-lane samplers (1–4)",
        "Per-shot Audio 1 + Picture 2 continuity conditioning": (
            "Per-lane Audio 1 + prior-tail Picture 2 continuity"
        ),
    }
    for group in workflow.get("groups", []):
        title = group.get("title")
        if title in group_title_updates:
            group["title"] = group_title_updates[title]

    runtime_ids = marker.get("runtime_lane_node_ids")
    if not isinstance(runtime_ids, dict):
        raise WorkflowError("MiniMax expansion v1 runtime-lane marker is missing.")
    runtime_title_templates = {
        "h3": (
            lambda lane: "H3 SHOT 1 — Picture 1 + Audio 1"
            if lane == 1
            else f"H3 SHOT {lane} — Picture 1 + prior-tail Picture 2 + Audio 1",
            lambda lane: f"H3 GENERATION LANE {lane} — Picture 1 + Audio 1"
            if lane == 1
            else f"H3 GENERATION LANE {lane} — Picture 1 + prior-tail Picture 2 + Audio 1",
        ),
        "noise": (
            lambda lane: f"H3 SHOT {lane} deterministic noise",
            lambda lane: f"H3 GENERATION LANE {lane} deterministic noise",
        ),
        "guiders": (
            lambda lane: f"H3 SHOT {lane} guider",
            lambda lane: f"H3 GENERATION LANE {lane} guider",
        ),
        "samplers": (
            lambda lane: f"H3 SHOT {lane} sampler",
            lambda lane: f"H3 GENERATION LANE {lane} sampler",
        ),
        "decoders": (
            lambda lane: f"H3 SHOT {lane} video frames",
            lambda lane: f"H3 GENERATION LANE {lane} video frames",
        ),
        "trims": (
            lambda lane: f"H3 SHOT {lane} AUDIO — relative slice of sync-safe guide",
            lambda lane: f"H3 GENERATION LANE {lane} AUDIO — relative sync-safe guide slice",
        ),
        "tail_math": (
            lambda lane: f"SHOT {lane} RETAINED CUT — continuity frame index",
            lambda lane: f"LANE {lane} RETAINED CUT — continuity frame index",
        ),
        "tails": (
            lambda lane: f"SHOT {lane} TAIL — Picture 2 continuity for shot {lane + 1}",
            lambda lane: f"LANE {lane} TAIL — Picture 2 continuity for generation lane {lane + 1}",
        ),
    }
    for key, (old_title, new_title) in runtime_title_templates.items():
        values = runtime_ids.get(key)
        if not isinstance(values, list):
            raise WorkflowError(f"MiniMax expansion v1 runtime marker {key!r} is missing.")
        for lane, node_id in enumerate(values, start=1):
            node = _node_in_any_scope(workflow, int(node_id))
            _replace_exact(node, "title", old_title(lane), new_title(lane))

    extra = workflow.setdefault("extra", {})
    old_workflow_note = (
        "MiniMax-H3 Ref2VA production graph: versioned Project Master Contract, "
        "locked ACE song/QC, measured-audio shot planning, four lazy 5–15 second "
        "H3 lanes with retained-cut Picture 2 continuity, exact-frame assembly, "
        "pristine ACE mux audio, and honest multi-format delivery planning."
    )
    new_workflow_note = (
        "MiniMax-H3 Ref2VA production graph: versioned Project Master Contract, "
        "locked ACE song/QC, measured-audio generation-lane planning, four lazy "
        "5–15 second H3 invocations that preserve native multi-shot prompts, "
        "retained-tail Picture 2 continuity between lanes, exact-frame assembly, "
        "pristine ACE mux audio, and honest multi-format delivery planning."
    )
    _replace_exact(extra, "workflow_note", old_workflow_note, new_workflow_note)

    upgraded = copy.deepcopy(marker)
    upgraded["version"] = MIGRATION_VERSION
    upgraded["master_video_duration_seconds"] = float(
        _first_widget(graph.node(178), 15.0)
    )
    upgraded["master_aspect_ratio"] = _aspect_from_splitter(graph.node(187))
    upgraded["max_h3_generation_lanes"] = int(
        upgraded.get("max_h3_shots", MAX_GENERATION_LANES)
    )
    upgraded["max_h3_generation_lane_seconds"] = float(
        upgraded.get("max_h3_shot_seconds", 15.0)
    )
    upgraded["count_semantics"] = {
        "target_profile_shot_count": "native_[Shot_N]_blocks_inside_each_generated_video",
        "project_master_recommended_h3_shot_count": "legacy_socket_name_for_duration_bounded_generation_lanes",
        "planner_effective_shot_count": "legacy_socket_name_for_effective_generation_lanes",
        "project_master_to_target_shot_override_connected": False,
    }
    upgraded["performance_mode_contract"] = copy.deepcopy(PERFORMANCE_MODE_MARKER)
    upgraded["generation_lane_boundary_policy"] = copy.deepcopy(
        BOUNDARY_POLICY_MARKER
    )
    reference = upgraded.get("reference_contract")
    if isinstance(reference, dict):
        _replace_exact(
            reference,
            "picture_2",
            "prior retained-cut tail; continuity guidance only for shots 2–4",
            "prior retained-lane tail; continuity guidance only for generation lanes 2–4",
        )
        _replace_exact(
            reference,
            "audio_1",
            "relative sync-safe guide slice; pristine master is muxed only after assembly",
            "relative per-lane sync-safe guide slice; pristine master is muxed only after assembly",
        )
    nodes = upgraded.get("nodes")
    if not isinstance(nodes, dict):
        raise WorkflowError("MiniMax expansion v1 node-id marker is missing.")
    nodes["generation_lane_planner"] = int(planner["id"])
    nodes["generation_lane_seed_fanout"] = int(seed_fanout["id"])
    nodes["generation_lane_assembler"] = int(assembler["id"])
    extra[MIGRATION_SCHEMA] = upgraded


def _install_marker(
    workflow: dict[str, Any],
    *,
    original: dict[str, Any],
    project: dict[str, Any],
    planner: dict[str, Any],
    seed_fanout: dict[str, Any],
    assembler: dict[str, Any],
    delivery: dict[str, Any],
    h3_context: dict[str, Any],
    performance_control: dict[str, Any],
    raw_source_sha256: str | None,
) -> None:
    extra = workflow.setdefault("extra", {})

    duration = float(_first_widget(next(
        node for node in workflow["nodes"] if int(node["id"]) == 178
    ), 15.0))
    splitter = next(node for node in workflow["nodes"] if int(node["id"]) == 187)
    aspect = _aspect_from_splitter(splitter)
    extra["workflow_note"] = (
        "MiniMax-H3 Ref2VA production graph: versioned Project Master Contract, "
        "locked ACE song/QC, evidence-preferred generation-lane planning with a "
        "deterministic balanced fallback, four lazy 5–15 second H3 invocations that "
        "preserve native multi-shot prompts and carry active shots across non-native "
        "seams, retained-tail Picture 2 continuity between lanes, exact-frame assembly, "
        "pristine ACE mux audio, and honest multi-format delivery planning."
    )
    extra[MIGRATION_SCHEMA] = {
        "version": MIGRATION_VERSION,
        "source_workflow_canonical_sha256": _canonical_json_hash(original),
        "source_workflow_file_sha256": raw_source_sha256 or "not supplied",
        "master_video_duration_seconds": duration,
        "master_aspect_ratio": aspect,
        "fps": FPS,
        "max_h3_generation_lanes": MAX_GENERATION_LANES,
        "max_h3_generation_lane_seconds": 15.0,
        # Compatibility aliases retained for consumers of the v1 marker.
        "max_h3_shots": MAX_SHOTS,
        "max_h3_shot_seconds": 15.0,
        "count_semantics": {
            "target_profile_shot_count": "native_[Shot_N]_blocks_inside_each_generated_video",
            "project_master_recommended_h3_shot_count": "legacy_socket_name_for_duration_bounded_generation_lanes",
            "planner_effective_shot_count": "legacy_socket_name_for_effective_generation_lanes",
            "project_master_to_target_shot_override_connected": False,
        },
        "reference_contract": {
            "picture_1": "source identity, appearance, scene, palette, composition authority",
            "picture_2": "prior retained-lane tail; continuity guidance only for generation lanes 2–4",
            "audio_1": "relative per-lane sync-safe guide slice; pristine master is muxed only after assembly",
        },
        "locked_song_sha256_source": "DiffusionGemmaAudioCandidateSelector.waveform_sha256",
        "hard_cuts_default": True,
        "alternate_formats_are_planned_not_claimed_as_rendered": True,
        "performance_mode_contract": copy.deepcopy(PERFORMANCE_MODE_MARKER),
        "generation_lane_boundary_policy": copy.deepcopy(BOUNDARY_POLICY_MARKER),
        "nodes": {
            "project_contract": int(project["id"]),
            "shot_planner": int(planner["id"]),
            "shot_seed_fanout": int(seed_fanout["id"]),
            "shot_assembler": int(assembler["id"]),
            "generation_lane_planner": int(planner["id"]),
            "generation_lane_seed_fanout": int(seed_fanout["id"]),
            "generation_lane_assembler": int(assembler["id"]),
            "delivery_planner": int(delivery["id"]),
            "h3_context": int(h3_context["id"]),
            "performance_control": int(performance_control["id"]),
            "h3_target_profile": 673,
            "create_video": 664,
            "save_video": 651,
        },
    }


def _project_contract_node(
    ids: IdAllocator, aspect: str, duration: float
) -> dict[str, Any]:
    return _node(
        ids,
        PROJECT_NODE_TYPE,
        "PROJECT MASTER CONTRACT — duration, song hash, format + H3 generation-lane budget",
        (1610.0, -1490.0),
        (660.0, 570.0),
        [
            _input("creative_brief", "STRING"),
            _input("production_duration_seconds", "FLOAT", widget=True),
            _input("master_audio_sha256", "STRING"),
            _input("lyrics", "STRING"),
            _input("excerpt_start_seconds", "FLOAT", widget=True),
            _input("excerpt_duration_seconds", "FLOAT", widget=True),
        ],
        [
            _output("project_manifest_json", "STRING"),
            _output("project_id", "STRING"),
            _output("master_aspect_ratio", "STRING"),
            _output("recommended_h3_shot_count", "INT"),
            _output("brief_sha256", "STRING"),
            _output("lyrics_sha256", "STRING"),
            _output("status", "STRING"),
            _output("ready", "BOOLEAN"),
            _output("max_h3_shot_seconds", "FLOAT"),
        ],
        [
            float(duration),
            aspect,
            0.0,
            float(duration),
            "MiniMax H3 Ref2VA",
            json.dumps(
                {
                    "primary": f"{aspect} master",
                    "adaptations": [value for value in ("9:16", "16:9", "1:1") if value != aspect],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            15.0,
        ],
    )


def _performance_mode_node(ids: IdAllocator, mode: str) -> dict[str, Any]:
    selected = mode if mode in PERFORMANCE_MODES else DEFAULT_PERFORMANCE_MODE
    return _node(
        ids,
        PERFORMANCE_MODE_NODE_TYPE,
        "H3 PERFORMANCE MODE — shared before Director and generation-lane planning",
        (820.0, -1460.0),
        (620.0, 230.0),
        [
            _input("base_audio_guidance", "STRING"),
            _input("performance_mode", "COMBO", widget=True),
        ],
        [
            _output("performance_mode", "STRING"),
            _output("target_audio_guidance", "STRING"),
            _output("status", "STRING"),
        ],
        [selected],
    )


def _shot_planner_node(ids: IdAllocator, duration: float) -> dict[str, Any]:
    outputs = [
        _output("plan_json", "STRING"),
        _output("status", "STRING"),
        _output("ready", "BOOLEAN"),
        _output("effective_shot_count", "INT"),
    ]
    for shot in range(1, MAX_SHOTS + 1):
        outputs.extend(
            [
                _output(f"shot_{shot}_prompt", "STRING"),
                _output(f"shot_{shot}_start", "FLOAT"),
                _output(f"shot_{shot}_duration", "FLOAT"),
                _output(f"shot_{shot}_frames", "INT"),
                _output(f"shot_{shot}_ready", "BOOLEAN"),
            ]
        )
    return _node(
        ids,
        PLANNER_NODE_TYPE,
        "AUDIO-AWARE H3 MULTI-LANE PLAN — evidence-preferred / deterministic fallback",
        (1610.0, -830.0),
        (770.0, 740.0),
        [
            _input("measured_audio_report_json", "STRING"),
            _input("base_h3_prompt", "STRING"),
            _input("performance_mode", "STRING"),
            _input("excerpt_start_seconds", "FLOAT", widget=True),
            _input("excerpt_duration_seconds", "FLOAT", widget=True),
            _input("max_shot_seconds", "FLOAT", widget=True),
            _input("project_manifest_json", "STRING"),
            _input("lyrics", "STRING", optional=True),
        ],
        outputs,
        [0.0, float(duration), 12.0, 5.0, 15.0],
    )


def _seed_fanout_node(ids: IdAllocator) -> dict[str, Any]:
    return _node(
        ids,
        SEED_NODE_TYPE,
        "H3 GENERATION-LANE SEEDS — deterministic root/project/lane fanout",
        (2600.0, -990.0),
        (500.0, 190.0),
        [
            _input("root_seed", "INT", widget=True),
            _input("project_manifest_json", "STRING"),
        ],
        [
            _output("shot_1_seed", "INT"),
            _output("shot_2_seed", "INT"),
            _output("shot_3_seed", "INT"),
            _output("shot_4_seed", "INT"),
            _output("seed_report_json", "STRING"),
        ],
        [42],
    )


def _assembler_node(ids: IdAllocator, duration: float) -> dict[str, Any]:
    return _node(
        ids,
        ASSEMBLER_NODE_TYPE,
        "LOCKED-SONG H3 ASSEMBLY — lazy lanes, exact frames, pristine audio",
        (5100.0, -650.0),
        (650.0, 410.0),
        [
            _input("final_audio", "AUDIO"),
            _input("plan_json", "STRING"),
            _input("target_duration_seconds", "FLOAT", widget=True),
            _input("shot_1_images", "IMAGE", optional=True),
            _input("shot_2_images", "IMAGE", optional=True),
            _input("shot_3_images", "IMAGE", optional=True),
            _input("shot_4_images", "IMAGE", optional=True),
        ],
        [
            _output("assembled_images", "IMAGE"),
            _output("final_audio", "AUDIO"),
            _output("assembly_report_json", "STRING"),
            _output("status", "STRING"),
            _output("ready", "BOOLEAN"),
        ],
        [float(duration), FPS],
    )


def _delivery_node(ids: IdAllocator) -> dict[str, Any]:
    return _node(
        ids,
        DELIVERY_NODE_TYPE,
        "MULTI-FORMAT DELIVERY — honest master/adaptation/cutdown plan",
        (5100.0, 40.0),
        (650.0, 310.0),
        [_input("project_manifest_json", "STRING")],
        [
            _output("delivery_manifest_json", "STRING"),
            _output("deliverable_count", "INT"),
            _output("status", "STRING"),
            _output("ready", "BOOLEAN"),
        ],
        ["Master + social adaptations", "Plan 15s + 6s cutdowns", 10.0],
    )


def _wire_h3_runtime(
    graph: MainGraph,
    *,
    planner: dict[str, Any],
    seed_fanout: dict[str, Any],
    assembler: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Build four H3 generation lanes and join them only through the lazy assembler."""

    ids = graph.ids
    source_image = graph.node(179)
    splitter = graph.node(187)
    audio_guide = graph.node(611)
    video_vae = graph.node(653)
    audio_vae = graph.node(654)
    sampler_select = graph.node(657)
    scheduler = graph.node(658)
    patched_model = graph.node(670)
    clip = graph.node(662)

    trim_template = copy.deepcopy(graph.node(569))
    h3_template = copy.deepcopy(graph.node(667))
    noise_template = copy.deepcopy(graph.node(663))
    guider_template = copy.deepcopy(graph.node(660))
    sampler_template = copy.deepcopy(graph.node(659))
    decoder_template = copy.deepcopy(graph.node(656))
    math_template = copy.deepcopy(graph.node(665))

    # H3's decoded audio is deliberately not part of delivery.  The old master
    # duration math/manual prompt/reroute are also replaced by the lane plan.
    for node_id in (655, 669, 672):
        graph.remove_node(graph.node(node_id))

    shot_h3: list[dict[str, Any]] = []
    shot_noise: list[dict[str, Any]] = []
    shot_guiders: list[dict[str, Any]] = []
    shot_samplers: list[dict[str, Any]] = []
    shot_decoders: list[dict[str, Any]] = []
    shot_trims: list[dict[str, Any]] = []
    tail_math: list[dict[str, Any]] = []
    tail_images: list[dict[str, Any]] = []

    prior_tail: dict[str, Any] | None = None
    for shot in range(1, MAX_SHOTS + 1):
        row_y = -650.0 + (shot - 1) * 970.0
        trim = graph.add(
            _trim_node(ids, trim_template, shot, (2650.0, row_y + 40.0))
        )
        shot_trims.append(trim)

        if shot == 1:
            h3 = _reset_node(
                graph,
                graph.node(667),
                title="H3 GENERATION LANE 1 — Picture 1 + Audio 1",
                pos=(3180.0, row_y),
            )
            _normalize_h3_reference_node(h3)
            noise = _reset_node(
                graph,
                graph.node(663),
                title="H3 GENERATION LANE 1 deterministic noise",
                pos=(3630.0, row_y),
            )
            guider = _reset_node(
                graph,
                graph.node(660),
                title="H3 GENERATION LANE 1 guider",
                pos=(3630.0, row_y + 150.0),
            )
            sampler = _reset_node(
                graph,
                graph.node(659),
                title="H3 GENERATION LANE 1 sampler",
                pos=(4050.0, row_y + 70.0),
            )
            decoder = _reset_node(
                graph,
                graph.node(656),
                title="H3 GENERATION LANE 1 video frames",
                pos=(4380.0, row_y + 70.0),
            )
        else:
            h3 = graph.add(
                _clone_reset(
                    ids,
                    h3_template,
                    title=(
                        f"H3 GENERATION LANE {shot} — Picture 1 + prior-tail Picture 2 + Audio 1"
                    ),
                    pos=(3180.0, row_y),
                )
            )
            _normalize_h3_reference_node(h3)
            noise = graph.add(
                _clone_reset(
                    ids,
                    noise_template,
                    title=f"H3 GENERATION LANE {shot} deterministic noise",
                    pos=(3630.0, row_y),
                )
            )
            guider = graph.add(
                _clone_reset(
                    ids,
                    guider_template,
                    title=f"H3 GENERATION LANE {shot} guider",
                    pos=(3630.0, row_y + 150.0),
                )
            )
            sampler = graph.add(
                _clone_reset(
                    ids,
                    sampler_template,
                    title=f"H3 GENERATION LANE {shot} sampler",
                    pos=(4050.0, row_y + 70.0),
                )
            )
            decoder = graph.add(
                _clone_reset(
                    ids,
                    decoder_template,
                    title=f"H3 GENERATION LANE {shot} video frames",
                    pos=(4380.0, row_y + 70.0),
                )
            )

        shot_h3.append(h3)
        shot_noise.append(noise)
        shot_guiders.append(guider)
        shot_samplers.append(sampler)
        shot_decoders.append(decoder)

        graph.connect(clip, "CLIP", h3, "clip", "CLIP")
        graph.connect(video_vae, "VAE", h3, "vae", "VAE")
        graph.connect(audio_vae, "VAE", h3, "audio_vae", "VAE")
        graph.connect(source_image, "IMAGE", h3, "ref_images.ref_image_0", "IMAGE")
        if prior_tail is not None:
            graph.connect(prior_tail, "IMAGE", h3, "ref_images.ref_image_1", "IMAGE")
        graph.connect(trim, "AUDIO", h3, "ref_audios.ref_audio_0", "AUDIO")
        graph.connect(planner, f"shot_{shot}_prompt", h3, "prompt", "STRING")
        graph.connect(splitter, "resolution_width", h3, "width", "INT")
        graph.connect(splitter, "resolution_height", h3, "height", "INT")
        graph.connect(planner, f"shot_{shot}_frames", h3, "length", "INT")

        graph.connect(
            audio_guide, "conditioning_audio", trim, "audio", "AUDIO"
        )
        graph.connect(planner, f"shot_{shot}_start", trim, "start_index", "FLOAT")
        graph.connect(planner, f"shot_{shot}_duration", trim, "duration", "FLOAT")

        graph.connect(seed_fanout, f"shot_{shot}_seed", noise, "noise_seed", "INT")
        graph.connect(patched_model, "model", guider, "model", "MODEL")
        graph.connect(h3, "positive", guider, "conditioning", "CONDITIONING")
        graph.connect(noise, "NOISE", sampler, "noise", "NOISE")
        graph.connect(guider, "GUIDER", sampler, "guider", "GUIDER")
        graph.connect(sampler_select, "SAMPLER", sampler, "sampler", "SAMPLER")
        graph.connect(scheduler, "SIGMAS", sampler, "sigmas", "SIGMAS")
        graph.connect(h3, "LATENT", sampler, "latent_image", "LATENT")
        graph.connect(sampler, "output", decoder, "samples", "LATENT")
        graph.connect(video_vae, "VAE", decoder, "vae", "VAE")
        graph.connect(
            decoder, "IMAGE", assembler, f"shot_{shot}_images", "IMAGE"
        )

        if shot < MAX_SHOTS:
            if shot == 1:
                replacement = _tail_index_node(
                    ids,
                    math_template,
                    shot,
                    (4690.0, row_y),
                    reuse_template_id=True,
                )
                tail_index = graph.replace(graph.node(665), replacement)
            else:
                tail_index = graph.add(
                    _tail_index_node(
                        ids,
                        math_template,
                        shot,
                        (4690.0, row_y),
                        reuse_template_id=False,
                    )
                )
            tail = graph.add(_tail_node(ids, shot, (4690.0, row_y + 160.0)))
            tail_math.append(tail_index)
            tail_images.append(tail)
            graph.connect(
                planner,
                f"shot_{shot}_start",
                tail_index,
                "values.a",
                "FLOAT,INT,BOOLEAN",
            )
            graph.connect(
                planner,
                f"shot_{shot}_duration",
                tail_index,
                "values.b",
                "FLOAT,INT,BOOLEAN",
            )
            graph.connect(decoder, "IMAGE", tail, "image", "IMAGE")
            graph.connect(tail_index, "INT", tail, "batch_index", "INT")
            prior_tail = tail

    create_video = _reset_node(
        graph,
        graph.node(664),
        title="Mux exact ACE master with exact-length assembled H3 frames",
        pos=(5480.0, -520.0),
    )
    create_video["widgets_values"] = [24, 8]
    save_video = graph.node(651)
    save_video["pos"] = [5850.0, -520.0]
    save_video["title"] = "FINAL MINIMAX-H3 MUSIC VIDEO — locked ACE master"

    graph.connect(planner, "plan_json", assembler, "plan_json", "STRING")
    graph.connect(
        audio_guide, "final_audio", assembler, "final_audio", "AUDIO"
    )
    graph.connect(
        graph.node(178), "FLOAT", assembler, "target_duration_seconds", "FLOAT"
    )
    graph.connect(assembler, "assembled_images", create_video, "images", "IMAGE")
    graph.connect(assembler, "final_audio", create_video, "audio", "AUDIO")
    graph.connect(create_video, "VIDEO", save_video, "video", "VIDEO")

    return {
        "h3": shot_h3,
        "noise": shot_noise,
        "guiders": shot_guiders,
        "samplers": shot_samplers,
        "decoders": shot_decoders,
        "trims": shot_trims,
        "tail_math": tail_math,
        "tails": tail_images,
    }


def migrate_workflow(
    workflow: dict[str, Any], *, source_file_sha256: str | None = None
) -> dict[str, Any]:
    """Return a migrated deep copy without writing the user-owned file."""

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
    existing = result.get("extra", {}).get(MIGRATION_SCHEMA)
    if existing is not None:
        if not isinstance(existing, dict):
            raise WorkflowError("The MiniMax expansion marker is present but unsupported.")
        existing_version = int(existing.get("version", 0))
        _validate_link_integrity(result)
        if existing_version == LEGACY_MIGRATION_VERSION:
            _upgrade_v1_to_v2(result)
        elif existing_version != MIGRATION_VERSION:
            raise WorkflowError("The MiniMax expansion marker is present but unsupported.")
        _refresh_performance_contract(result)
        validate_workflow(result)
        return result

    _validate_link_integrity(result)
    _require_baseline(result)
    original = copy.deepcopy(result)
    ids = IdAllocator(result)
    graph = MainGraph(result, ids)
    duration = float(_first_widget(graph.node(178), 15.0))
    if not 0.1 <= duration <= 60.0:
        raise WorkflowError(
            "The four-lane MiniMax-H3 expansion supports a master duration from 0.1 to 60 seconds."
        )
    aspect = _aspect_from_splitter(graph.node(187))

    context = graph.add(_context_node(ids))
    graph.connect(graph.node(177), "STRING", context, "user_prompt", "STRING")
    graph.connect(graph.node(179), "IMAGE", context, "reference_images", "IMAGE")
    graph.connect(context, "gemma_context", graph.node(186), "gemma_context", "DG_CONTEXT")
    graph.connect(context, "gemma_context", graph.node(187), "gemma_context", "DG_CONTEXT")

    splitter = graph.node(187)
    target = graph.node(673)
    _ensure_splitter_aspect_override(splitter)
    _require_target_shot_override_unconnected(target)
    _ensure_performance_mode_output(graph.node(621))
    _ensure_performance_mode_override_input(graph.node(621))
    target_values = list(target.get("widgets_values") or [])
    if len(target_values) < 7:
        raise WorkflowError("H3 target profile is missing its dialogue-mode widget.")
    target_values[6] = "off"
    target["widgets_values"] = target_values

    # Remove only the two disconnected LTX diagnostics.  The performance node
    # remains as the user's Natural/Dance/Lyrics selector, but its LTX-formatted prompt
    # output is never routed into H3.
    for node_id, title_fragment in (
        (511, "Candidate LTX negative prompt"),
        (549, "Candidate LTX positive prompt"),
    ):
        matches = [node for node in graph.nodes if int(node["id"]) == node_id]
        if matches:
            if title_fragment not in str(matches[0].get("title", "")):
                raise WorkflowError(
                    f"Refusing to remove node {node_id}: its diagnostic role changed."
                )
            graph.remove_node(matches[0])

    current_performance_mode = str(
        _first_widget(graph.node(621), DEFAULT_PERFORMANCE_MODE)
    )
    performance_control = graph.add(
        _performance_mode_node(ids, current_performance_mode)
    )
    project = graph.add(_project_contract_node(ids, aspect, duration))
    planner = graph.add(_shot_planner_node(ids, duration))
    seed_fanout = graph.add(_seed_fanout_node(ids))
    assembler = graph.add(_assembler_node(ids, duration))
    delivery = graph.add(_delivery_node(ids))

    graph.connect(graph.node(177), "STRING", project, "creative_brief", "STRING")
    graph.connect(
        graph.node(178), "FLOAT", project, "production_duration_seconds", "FLOAT"
    )
    graph.connect(
        graph.node(610), "waveform_sha256", project, "master_audio_sha256", "STRING"
    )
    graph.connect(graph.node(556), "lyrics", project, "lyrics", "STRING")
    graph.connect(
        graph.node(610),
        "suggested_start_seconds",
        project,
        "excerpt_start_seconds",
        "FLOAT",
    )
    graph.connect(
        graph.node(178), "FLOAT", project, "excerpt_duration_seconds", "FLOAT"
    )
    graph.connect(
        project,
        "master_aspect_ratio",
        splitter,
        "resolution_aspect_ratio_override",
        "STRING",
    )

    graph.connect(
        graph.node(612),
        "output",
        planner,
        "measured_audio_report_json",
        "STRING",
    )
    graph.connect(
        graph.node(188), "prompt", planner, "base_h3_prompt", "STRING"
    )
    graph.connect(
        performance_control, "performance_mode", planner, "performance_mode", "STRING"
    )
    graph.connect(
        graph.node(610),
        "suggested_start_seconds",
        planner,
        "excerpt_start_seconds",
        "FLOAT",
    )
    graph.connect(
        graph.node(178), "FLOAT", planner, "excerpt_duration_seconds", "FLOAT"
    )
    graph.connect(
        project,
        "max_h3_shot_seconds",
        planner,
        "max_shot_seconds",
        "FLOAT",
    )
    graph.connect(
        graph.node(621), "selected_lyrics", planner, "lyrics", "STRING"
    )
    graph.connect(
        project,
        "project_manifest_json",
        planner,
        "project_manifest_json",
        "STRING",
    )

    graph.connect(
        graph.node(624),
        "STRING",
        performance_control,
        "base_audio_guidance",
        "STRING",
    )
    graph.connect(
        performance_control,
        "target_audio_guidance",
        target,
        "audio_guidance",
        "STRING",
    )
    graph.connect(
        performance_control,
        "performance_mode",
        graph.node(621),
        "performance_mode_override",
        "STRING",
    )

    graph.connect(graph.node(418), "seed", seed_fanout, "root_seed", "INT")
    graph.connect(
        project,
        "project_manifest_json",
        seed_fanout,
        "project_manifest_json",
        "STRING",
    )
    graph.connect(
        project,
        "project_manifest_json",
        delivery,
        "project_manifest_json",
        "STRING",
    )

    runtime = _wire_h3_runtime(
        graph,
        planner=planner,
        seed_fanout=seed_fanout,
        assembler=assembler,
    )

    preview_specs = (
        (
            project,
            "status",
            "PROJECT CONTRACT STATUS",
            (2310.0, -1490.0),
        ),
        (
            planner,
            "plan_json",
            "AUDIO-AWARE H3 GENERATION-LANE PLAN",
            (2420.0, -760.0),
        ),
        (
            planner,
            "status",
            "H3 GENERATION-LANE PLAN STATUS",
            (2420.0, -520.0),
        ),
        (
            assembler,
            "status",
            "LOCKED-SONG ASSEMBLY STATUS",
            (5100.0, -160.0),
        ),
        (
            delivery,
            "delivery_manifest_json",
            "MULTI-FORMAT DELIVERY MANIFEST — plan only",
            (5800.0, 30.0),
        ),
    )
    for origin, output_name, title, pos in preview_specs:
        preview = graph.add(_preview_node(ids, title, pos))
        graph.connect(origin, output_name, preview, "source", "STRING")

    _update_visible_copy(result, graph)
    result["last_node_id"] = ids.last_node
    result["last_link_id"] = ids.last_link
    result["revision"] = int(result.get("revision", 0) or 0) + 1
    _install_marker(
        result,
        original=original,
        project=project,
        planner=planner,
        seed_fanout=seed_fanout,
        assembler=assembler,
        delivery=delivery,
        h3_context=context,
        performance_control=performance_control,
        raw_source_sha256=source_file_sha256,
    )
    # Persist node ids for tests and future surgical upgrades without making
    # lane inference depend on canvas positions.
    result["extra"][MIGRATION_SCHEMA]["runtime_lane_node_ids"] = {
        key: [int(node["id"]) for node in values] for key, values in runtime.items()
    }
    validate_workflow(result)
    sensitive_after = _sensitive_key_paths(result)
    if sensitive_after:
        raise WorkflowError(
            "Migration introduced non-empty sensitive-looking fields at "
            + ", ".join(sensitive_after)
            + "."
        )
    return result


def validate_workflow(workflow: dict[str, Any]) -> None:
    """Validate link backlinks plus every MiniMax expansion invariant."""

    _validate_link_integrity(workflow)
    sensitive = _sensitive_key_paths(workflow)
    if sensitive:
        raise WorkflowError(
            "Workflow contains non-empty sensitive-looking fields at "
            + ", ".join(sensitive)
            + "."
        )
    marker = workflow.get("extra", {}).get(MIGRATION_SCHEMA)
    if not marker:
        return
    if not isinstance(marker, dict) or int(marker.get("version", 0)) != MIGRATION_VERSION:
        raise WorkflowError("MiniMax expansion marker is stale or malformed.")
    if marker.get("performance_mode_contract") != PERFORMANCE_MODE_MARKER:
        raise WorkflowError("MiniMax performance-mode marker is stale or malformed.")
    if marker.get("generation_lane_boundary_policy") != BOUNDARY_POLICY_MARKER:
        raise WorkflowError("MiniMax generation-lane boundary marker is stale or malformed.")
    legacy = workflow.get("extra", {}).get("diffusiongemma.music_video_production")
    if not isinstance(legacy, dict) or int(legacy.get("version", 0)) != 10:
        raise WorkflowError("The preserved music-video production v10 marker is missing.")

    graph = MainGraph(workflow, IdAllocator(workflow))
    all_nodes = list(_all_nodes(workflow))
    all_nodes_by_id: dict[int, dict[str, Any]] = {}
    for node in all_nodes:
        node_id = int(node["id"])
        if node_id in all_nodes_by_id:
            raise WorkflowError(f"Node id {node_id} is duplicated across graph scopes.")
        all_nodes_by_id[node_id] = node

    def nodes_of_type(node_type: str) -> list[dict[str, Any]]:
        return [node for node in graph.nodes if node.get("type") == node_type]

    def all_nodes_of_type(node_type: str) -> list[dict[str, Any]]:
        return [node for node in all_nodes if node.get("type") == node_type]

    def exactly_one(node_type: str) -> dict[str, Any]:
        matches = nodes_of_type(node_type)
        if len(matches) != 1:
            raise WorkflowError(f"Expected one {node_type}; found {len(matches)}.")
        return matches[0]

    project = exactly_one(PROJECT_NODE_TYPE)
    planner = exactly_one(PLANNER_NODE_TYPE)
    seed_fanout = exactly_one(SEED_NODE_TYPE)
    assembler = exactly_one(ASSEMBLER_NODE_TYPE)
    delivery = exactly_one(DELIVERY_NODE_TYPE)
    h3_context = exactly_one(H3_CONTEXT_NODE_TYPE)
    performance_control = exactly_one(PERFORMANCE_MODE_NODE_TYPE)
    performance_values = list(performance_control.get("widgets_values") or [])
    if len(performance_values) != 1 or performance_values[0] not in PERFORMANCE_MODES:
        raise WorkflowError("H3 performance control has an unsupported mode widget.")
    start_note = graph.node(116)
    start_copy = " ".join(str(value) for value in start_note.get("widgets_values", []))
    generated_start_note = (
        str(start_note.get("title", ""))
        == "START HERE — lock the song, plan H3 generation lanes, then assemble"
    )
    if generated_start_note and (
        "defaults to **Natural / audio-led sync**" not in start_copy
        or "**Dance / music sync**" not in start_copy
        or "**Lyrics + lip sync**" not in start_copy
        or "lyrics are not a timing schedule" not in start_copy
    ):
        raise WorkflowError("H3 start note omits the tri-mode audio-led contract.")
    h3_nodes = all_nodes_of_type("MiniMaxH3ReferenceToVideo")
    if len(h3_nodes) != MAX_SHOTS:
        raise WorkflowError(f"Expected four H3 generation-lane nodes; found {len(h3_nodes)}.")

    node_ids = marker.get("nodes")
    expected_marker_nodes = {
        "project_contract": int(project["id"]),
        "shot_planner": int(planner["id"]),
        "shot_seed_fanout": int(seed_fanout["id"]),
        "shot_assembler": int(assembler["id"]),
        "generation_lane_planner": int(planner["id"]),
        "generation_lane_seed_fanout": int(seed_fanout["id"]),
        "generation_lane_assembler": int(assembler["id"]),
        "delivery_planner": int(delivery["id"]),
        "h3_context": int(h3_context["id"]),
        "performance_control": int(performance_control["id"]),
        "h3_target_profile": 673,
        "create_video": 664,
        "save_video": 651,
    }
    if node_ids != expected_marker_nodes:
        raise WorkflowError("Expansion node-id marker does not match the graph.")

    context_widgets = h3_context.get("widgets_values") or []
    manifest = str(context_widgets[0]) if context_widgets else ""
    if "<Picture 1>:" not in manifest or "<Audio 1>:" not in manifest:
        raise WorkflowError("H3 context must define Picture 1 and Audio 1.")
    _assert_origin(workflow, graph.node(578), "gemma_context", "DiffusionGemmaContextHub", "gemma_context")
    _assert_origin(workflow, graph.node(186), "gemma_context", H3_CONTEXT_NODE_TYPE, "gemma_context")
    _assert_origin(workflow, graph.node(187), "gemma_context", H3_CONTEXT_NODE_TYPE, "gemma_context")

    _assert_origin(workflow, project, "creative_brief", "PrimitiveStringMultiline", "STRING")
    _assert_origin(workflow, project, "production_duration_seconds", "PrimitiveFloat", "FLOAT")
    _assert_origin(workflow, project, "master_audio_sha256", "DiffusionGemmaAudioCandidateSelector", "waveform_sha256")
    _assert_origin(workflow, project, "lyrics", "SplatStageBlueprintRouter", "lyrics")
    _assert_origin(workflow, project, "excerpt_start_seconds", "DiffusionGemmaAudioCandidateSelector", "suggested_start_seconds")
    _assert_origin(workflow, project, "excerpt_duration_seconds", "PrimitiveFloat", "FLOAT")

    _require_target_shot_override_unconnected(graph.node(673))
    _assert_origin(workflow, graph.node(187), "resolution_aspect_ratio_override", PROJECT_NODE_TYPE, "master_aspect_ratio")
    _assert_origin(workflow, planner, "measured_audio_report_json", "easy cleanGpuUsed", "output")
    _assert_origin(workflow, planner, "base_h3_prompt", "DiffusionGemmaBranchGenerationGate", "prompt")
    _assert_origin(workflow, planner, "performance_mode", PERFORMANCE_MODE_NODE_TYPE, "performance_mode")
    _assert_origin(workflow, planner, "excerpt_start_seconds", "DiffusionGemmaAudioCandidateSelector", "suggested_start_seconds")
    _assert_origin(workflow, planner, "excerpt_duration_seconds", "PrimitiveFloat", "FLOAT")
    _assert_origin(workflow, planner, "max_shot_seconds", PROJECT_NODE_TYPE, "max_h3_shot_seconds")
    _assert_origin(workflow, planner, "lyrics", "DiffusionGemmaLTXPerformancePrompt", "selected_lyrics")
    _assert_origin(workflow, planner, "project_manifest_json", PROJECT_NODE_TYPE, "project_manifest_json")
    _assert_origin(workflow, seed_fanout, "root_seed", "SeedNode", "seed")
    _assert_origin(workflow, seed_fanout, "project_manifest_json", PROJECT_NODE_TYPE, "project_manifest_json")
    _assert_origin(workflow, delivery, "project_manifest_json", PROJECT_NODE_TYPE, "project_manifest_json")

    performance = exactly_one("DiffusionGemmaLTXPerformancePrompt")
    performance_prompt_output = performance["outputs"][graph.output_slot(performance, "ltx_prompt")]
    if performance_prompt_output.get("links"):
        raise WorkflowError("The LTX-formatted performance prompt must not feed H3.")
    _assert_origin(workflow, performance, "ltx_prompt", "DiffusionGemmaBranchGenerationGate", "prompt")
    _assert_origin(workflow, performance, "performance_mode_override", PERFORMANCE_MODE_NODE_TYPE, "performance_mode")
    _assert_origin(workflow, performance_control, "base_audio_guidance", "PrimitiveStringMultiline", "STRING")
    _assert_origin(workflow, graph.node(673), "audio_guidance", PERFORMANCE_MODE_NODE_TYPE, "target_audio_guidance")
    target_values = list(graph.node(673).get("widgets_values") or [])
    if len(target_values) < 7 or target_values[6] != "off":
        raise WorkflowError("H3 spoken-dialogue mode must be Off for the music-performance contract.")

    if int(marker.get("max_h3_generation_lanes", 0)) != MAX_GENERATION_LANES:
        raise WorkflowError("Expansion generation-lane capacity metadata is stale.")
    if float(marker.get("max_h3_generation_lane_seconds", 0.0)) != 15.0:
        raise WorkflowError("Expansion generation-lane duration metadata is stale.")
    semantics = marker.get("count_semantics")
    if not isinstance(semantics, dict) or semantics.get(
        "project_master_to_target_shot_override_connected"
    ) is not False:
        raise WorkflowError("Expansion count-semantics marker is missing or stale.")

    runtime_ids = marker.get("runtime_lane_node_ids")
    expected_counts = {
        "h3": 4,
        "noise": 4,
        "guiders": 4,
        "samplers": 4,
        "decoders": 4,
        "trims": 4,
        "tail_math": 3,
        "tails": 3,
    }
    if not isinstance(runtime_ids, dict):
        raise WorkflowError("Runtime-lane node-id marker is missing.")
    nodes_by_id = all_nodes_by_id
    expected_types = {
        "h3": "MiniMaxH3ReferenceToVideo",
        "noise": "RandomNoise",
        "guiders": "BasicGuider",
        "samplers": "SamplerCustomAdvanced",
        "decoders": "VAEDecode",
        "trims": "TrimAudioDuration",
        "tail_math": "ComfyMathExpression",
        "tails": "ImageFromBatch",
    }
    lane_nodes: dict[str, list[dict[str, Any]]] = {}
    for key, count in expected_counts.items():
        values = runtime_ids.get(key)
        if not isinstance(values, list) or len(values) != count:
            raise WorkflowError(f"Runtime marker {key!r} must contain {count} node ids.")
        try:
            resolved = [nodes_by_id[int(node_id)] for node_id in values]
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(f"Runtime marker {key!r} references a missing node.") from exc
        if any(node.get("type") != expected_types[key] for node in resolved):
            raise WorkflowError(f"Runtime marker {key!r} references the wrong node type.")
        lane_nodes[key] = resolved

    if nodes_of_type("VAEDecodeAudio"):
        raise WorkflowError("H3-generated audio decode must not exist in the delivery graph.")
    for forbidden_id in (655, 669, 672):
        if forbidden_id in nodes_by_id:
            raise WorkflowError(f"Dead baseline node {forbidden_id} was not removed.")

    runtime_node_ids = [
        int(node_id)
        for values in runtime_ids.values()
        if isinstance(values, list)
        for node_id in values
    ]
    collapsed_runtime = any(
        _subgraph_containing_node(workflow, node_id) is not None
        for node_id in runtime_node_ids
    )
    if collapsed_runtime:
        fast_marker = workflow.get("extra", {}).get(
            "diffusiongemma.minimax_h3_fast_test"
        )
        if not isinstance(fast_marker, dict):
            raise WorkflowError(
                "Subgraph-collapsed H3 lanes require the FAST migration marker."
            )
        fast_nodes = fast_marker.get("nodes")
        if not isinstance(fast_nodes, dict):
            raise WorkflowError("The FAST migration node marker is missing.")
        subgraph_id = str(fast_nodes.get("h3_subgraph", ""))
        matching_subgraphs = [
            item
            for item in workflow.get("definitions", {}).get("subgraphs", [])
            if str(item.get("id", "")) == subgraph_id
        ]
        if len(matching_subgraphs) != 1:
            raise WorkflowError("The FAST H3 subgraph marker is missing or ambiguous.")
        subgraph = matching_subgraphs[0]
        if any(
            node.get("type") == "VAEDecodeAudio"
            for node in subgraph.get("nodes", [])
        ):
            raise WorkflowError("The FAST H3 subgraph must not decode generated audio.")
        outer = graph.node(int(fast_nodes.get("h3_outer", -1)))
        if str(outer.get("type", "")) != subgraph_id:
            raise WorkflowError("The FAST H3 outer node does not match its subgraph.")
        for key in (
            "h3",
            "noise",
            "guiders",
            "samplers",
            "decoders",
            "tail_math",
            "tails",
        ):
            for node in lane_nodes[key]:
                if _subgraph_containing_node(workflow, int(node["id"])) is not subgraph:
                    raise WorkflowError(
                        f"FAST runtime marker {key!r} escapes the declared H3 subgraph."
                    )
        for trim in lane_nodes["trims"]:
            if _subgraph_containing_node(workflow, int(trim["id"])) is not None:
                raise WorkflowError("FAST audio trims must remain in the main graph.")

        output_node_id = int((subgraph.get("outputNode") or {}).get("id", -20))
        for lane in range(1, MAX_SHOTS + 1):
            index = lane - 1
            trim = lane_nodes["trims"][index]
            _assert_origin(
                workflow,
                trim,
                "audio",
                "DiffusionGemmaLTXAudioGuide",
                "conditioning_audio",
            )
            _assert_origin(
                workflow,
                trim,
                "start_index",
                PLANNER_NODE_TYPE,
                f"shot_{lane}_start",
            )
            _assert_origin(
                workflow,
                trim,
                "duration",
                PLANNER_NODE_TYPE,
                f"shot_{lane}_duration",
            )
            assembler_slot = graph.input_slot(assembler, f"shot_{lane}_images")
            assembler_link_id = assembler["inputs"][assembler_slot].get("link")
            if assembler_link_id is None:
                raise WorkflowError(f"Assembler generation lane {lane} is unconnected.")
            assembler_link = _link_by_id(graph, int(assembler_link_id))
            if (
                int(assembler_link[1]) != int(outer["id"])
                or int(assembler_link[2]) != index
            ):
                raise WorkflowError(
                    f"Assembler generation lane {lane} does not come from FAST outer output {index}."
                )
            boundary_links = [
                link
                for link in subgraph.get("links", [])
                if int(link.get("target_id", 0)) == output_node_id
                and int(link.get("target_slot", -1)) == index
            ]
            if len(boundary_links) != 1 or int(boundary_links[0]["origin_id"]) != int(
                lane_nodes["decoders"][index]["id"]
            ):
                raise WorkflowError(
                    f"FAST outer output {index} is not sourced by generation lane {lane}'s decoder."
                )

    for shot in range(1, MAX_SHOTS + 1):
        if collapsed_runtime:
            break
        index = shot - 1
        h3 = lane_nodes["h3"][index]
        names = [item.get("name") for item in h3.get("inputs", [])]
        if "ref_audios.ref_audio_0" not in names or any(
            name in {"ref_audios.ref_audio_1", "ref_audios.ref_audio_2"}
            for name in names
        ):
            raise WorkflowError(f"H3 generation lane {shot} must expose exactly Audio 1.")
        _assert_origin(workflow, h3, "clip", "CLIPLoader", "CLIP")
        _assert_origin(workflow, h3, "vae", "VAELoader", "VAE")
        _assert_origin(workflow, h3, "audio_vae", "VAELoader", "VAE")
        _assert_origin(workflow, h3, "ref_images.ref_image_0", "LoadImage", "IMAGE")
        _assert_origin(workflow, h3, "ref_audios.ref_audio_0", "TrimAudioDuration", "AUDIO")
        _assert_origin(workflow, h3, "prompt", PLANNER_NODE_TYPE, f"shot_{shot}_prompt")
        _assert_origin(workflow, h3, "width", "DiffusionGemmaJSONSplitter", "resolution_width")
        _assert_origin(workflow, h3, "height", "DiffusionGemmaJSONSplitter", "resolution_height")
        _assert_origin(workflow, h3, "length", PLANNER_NODE_TYPE, f"shot_{shot}_frames")
        if shot == 1:
            slot = graph.input_slot(h3, "ref_images.ref_image_1")
            if h3["inputs"][slot].get("link") is not None:
                raise WorkflowError("H3 generation lane 1 must not have a Picture 2 continuity input.")
        else:
            origin = _assert_origin(workflow, h3, "ref_images.ref_image_1", "ImageFromBatch", "IMAGE")
            if int(origin["id"]) != int(lane_nodes["tails"][shot - 2]["id"]):
                raise WorkflowError(f"H3 generation lane {shot} uses the wrong prior-tail continuity frame.")

        trim = lane_nodes["trims"][index]
        _assert_origin(workflow, trim, "audio", "DiffusionGemmaLTXAudioGuide", "conditioning_audio")
        _assert_origin(workflow, trim, "start_index", PLANNER_NODE_TYPE, f"shot_{shot}_start")
        _assert_origin(workflow, trim, "duration", PLANNER_NODE_TYPE, f"shot_{shot}_duration")
        _assert_origin(workflow, lane_nodes["noise"][index], "noise_seed", SEED_NODE_TYPE, f"shot_{shot}_seed")
        _assert_origin(workflow, lane_nodes["guiders"][index], "conditioning", "MiniMaxH3ReferenceToVideo", "positive")
        _assert_origin(workflow, lane_nodes["samplers"][index], "latent_image", "MiniMaxH3ReferenceToVideo", "LATENT")
        _assert_origin(workflow, lane_nodes["decoders"][index], "samples", "SamplerCustomAdvanced", "output")
        origin = _assert_origin(workflow, assembler, f"shot_{shot}_images", "VAEDecode", "IMAGE")
        if int(origin["id"]) != int(lane_nodes["decoders"][index]["id"]):
            raise WorkflowError(f"Assembler lane {shot} input uses the wrong decoder.")

        if shot < MAX_SHOTS:
            tail_index = lane_nodes["tail_math"][index]
            tail = lane_nodes["tails"][index]
            expression = str(_first_widget(tail_index, ""))
            if expression != "max(0, round((a + b) * 24) - round(a * 24) - 1)":
                raise WorkflowError("Continuity index must use the retained master cut.")
            _assert_origin(workflow, tail_index, "values.a", PLANNER_NODE_TYPE, f"shot_{shot}_start")
            _assert_origin(workflow, tail_index, "values.b", PLANNER_NODE_TYPE, f"shot_{shot}_duration")
            origin = _assert_origin(workflow, tail, "image", "VAEDecode", "IMAGE")
            if int(origin["id"]) != int(lane_nodes["decoders"][index]["id"]):
                raise WorkflowError(f"Generation lane {shot} tail uses the wrong decoded batch.")
            _assert_origin(workflow, tail, "batch_index", "ComfyMathExpression", "INT")
            tail_widgets = tail.get("widgets_values") or []
            if len(tail_widgets) < 2 or int(tail_widgets[1]) != 1:
                raise WorkflowError("Continuity extractor must retain exactly one frame.")

    _assert_origin(workflow, assembler, "plan_json", PLANNER_NODE_TYPE, "plan_json")
    _assert_origin(workflow, assembler, "final_audio", "DiffusionGemmaLTXAudioGuide", "final_audio")
    _assert_origin(workflow, assembler, "target_duration_seconds", "PrimitiveFloat", "FLOAT")
    create_video = graph.node(664)
    _assert_origin(workflow, create_video, "images", ASSEMBLER_NODE_TYPE, "assembled_images")
    _assert_origin(workflow, create_video, "audio", ASSEMBLER_NODE_TYPE, "final_audio")
    _assert_origin(workflow, graph.node(651), "video", "CreateVideo", "VIDEO")
    _assert_origin(workflow, graph.node(613), "audio", "DiffusionGemmaLTXAudioGuide", "final_audio")

    gate = graph.node(188)
    _assert_origin(workflow, gate, "prompt", "DiffusionGemmaJSONSplitter", "minimax_h3_prompt")
    if float(marker.get("master_video_duration_seconds", -1)) != float(
        _first_widget(graph.node(178), -2)
    ):
        raise WorkflowError("Expansion duration metadata is stale.")
    if marker.get("master_aspect_ratio") != _aspect_from_splitter(graph.node(187)):
        raise WorkflowError("Expansion aspect metadata is stale.")


def _atomic_json_write(
    path: Path,
    workflow: dict[str, Any],
    *,
    expected_source_sha256: str,
) -> Path:
    """Write atomically after a race check and preserve the exact source once."""

    current = path.read_bytes()
    current_hash = _file_hash(current)
    if current_hash != expected_source_sha256:
        raise WorkflowError(
            f"Refusing to overwrite {path}: it changed after migration began "
            f"({current_hash} != {expected_source_sha256})."
        )
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if backup.exists():
        backup_hash = _file_hash(backup.read_bytes())
        if backup_hash != expected_source_sha256:
            raise WorkflowError(
                f"Refusing to reuse {backup}: its contents do not match the source "
                "being replaced. The existing recovery copy was left untouched."
            )
    else:
        shutil.copy2(path, backup)
        if _file_hash(backup.read_bytes()) != expected_source_sha256:
            raise WorkflowError("The recovery backup did not verify byte-for-byte.")

    payload = json.dumps(workflow, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return backup


def migrate_file(
    path: Path,
    *,
    write: bool,
    expected_source_sha256: str | None = None,
) -> tuple[dict[str, Any], str, Path | None]:
    path = path.resolve()
    if not path.is_file():
        raise WorkflowError(f"Workflow does not exist: {path}")
    source_bytes = path.read_bytes()
    source_hash = _file_hash(source_bytes)
    if expected_source_sha256 and source_hash.casefold() != expected_source_sha256.casefold():
        raise WorkflowError(
            f"Source SHA-256 mismatch for {path}: {source_hash} != {expected_source_sha256}."
        )
    try:
        source = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Workflow is not valid UTF-8 JSON: {path}") from exc
    migrated = migrate_workflow(source, source_file_sha256=source_hash)
    validate_workflow(migrated)
    backup: Path | None = None
    if write and migrated != source:
        backup = _atomic_json_write(
            path,
            migrated,
            expected_source_sha256=source_hash,
        )
    return migrated, source_hash, backup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_WORKFLOW])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Migrate in memory and validate without writing or creating a backup.",
    )
    parser.add_argument(
        "--expected-source-sha256",
        help="Optional exact SHA-256 guard for a single source workflow.",
    )
    args = parser.parse_args(argv)
    if args.expected_source_sha256 and len(args.paths) != 1:
        parser.error("--expected-source-sha256 requires exactly one workflow path")
    for path in args.paths:
        migrated, source_hash, backup = migrate_file(
            path,
            write=not args.check,
            expected_source_sha256=args.expected_source_sha256,
        )
        result_hash = _canonical_json_hash(migrated)
        action = "checked" if args.check else ("migrated" if backup else "unchanged")
        backup_text = f" backup={backup}" if backup else ""
        print(
            f"{action} {path.resolve()} source_sha256={source_hash} "
            f"result_sha256={result_hash}{backup_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
