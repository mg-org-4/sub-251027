#!/usr/bin/env python3
"""Create the guarded V5 MiniMax-H3 audio-led/timed-lyrics workflow sibling.

The verified V4 workflow is treated as immutable evidence.  This migration
copies it in memory, adds a lazy vocal-separation/Whisper timing branch, wires
the resulting hash-locked report into the existing H3 lane planner, and writes
new V5 siblings only when their exact deterministic contents are known.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


MIGRATION_SCHEMA = "diffusiongemma.minimax_h3_timed_lyrics"
MIGRATION_VERSION = 1
TIMED_REPORT_SCHEMA = "diffusiongemma.timed_lyrics_report"
TIMED_REPORT_VERSION = 1
NATURAL_MODE = "Natural / audio-led sync"
DANCE_MODE = "Dance / music sync"
LYRICS_MODE = "Lyrics + lip sync"
PERFORMANCE_MODES = (DANCE_MODE, LYRICS_MODE, NATURAL_MODE)
AUDIO_SELECTOR_POLICY_MARKER = {
    "revision": 4,
    "aligned_half_time": "advisory_when_canonical_bpm_within_5_percent",
    "aligned_double_time": "mandatory_alignment_pressure_vocal_plus_two_of_four_quality_signals",
    "double_time": "hard_failure_outside_evidence_bundle",
    "supporting_signals_required": 2,
    "supporting_signals": [
        "visual_recovery_coverage",
        "excerpt_score",
        "tonal_family_consistency",
        "transition_stability",
    ],
    "auto_selection_order": "tempo_class_then_recovery_then_excerpt_then_full_song_then_index",
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
LEGACY_AUDIO_QC_COPY = (
    "A canonically aligned double-time subdivision is accepted only when the selected excerpt proves low onset pressure, "
    "at least 70% visual-recovery coverage, strong excerpt quality, tonal stability, and vocal activity; otherwise it remains blocking."
)
CURRENT_AUDIO_QC_COPY = (
    "A canonically aligned double-time subdivision is advisory when canonical error is within 5%, excerpt onset pressure is at most "
    "4.0/s, vocal activity is present, and at least two of visual-recovery coverage, excerpt score, tonal-family consistency, or "
    "transition stability pass their strict thresholds; weakly supported, dense, or misaligned double-time remains blocking."
)
CURRENT_BOUNDARY_POLICY_COPY = (
    "Generation-lane seams prefer native shot boundaries with measured recovery, "
    "then native-only or mixed evidence. If those grids cannot satisfy the hard "
    "duration limits, deterministic balanced seams are used. A non-native seam "
    "carries the active source shot into the next lane so later timestamps remain "
    "truthful. The plan records why each seam was selected separately from the "
    "evidence present there. Missing ideal cut evidence never blocks generation; "
    "invalid clocks, malformed shot blocks, timestamps, identity references, stale "
    "audio locks, and impossible lane capacity remain hard failures."
)

SOURCE_V4 = Path(
    r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_DUAL_IDENTITY_P3_RELAY_GUARDED_V4_20260822.json"
)
SOURCE_V4_SHA256 = (
    "fe4261c2dd4fb16df6acf05d8735beb274a8e2d80821936dcda0271d2978d6fa"
)
V5_FILENAME = (
    "14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_"
    "DUAL_IDENTITY_P3_RELAY_TIMED_LYRICS_V5_20260822.json"
)
DEFAULT_DESKTOP_OUTPUT = Path(r"C:\Users\danrh\Desktop") / V5_FILENAME
DEFAULT_USER_OUTPUT = (
    Path(r"C:\ComfyUI\app\user\default\workflows") / V5_FILENAME
)
WHISPER_MODEL_RELATIVE_PATH = "models/whisper/whisper-large-v3-turbo/model.safetensors"
WHISPER_MODEL_BYTES = 1_617_824_864
WHISPER_MODEL_SHA256 = (
    "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1"
)
DEMUCS_CHECKPOINT_PATH = (
    r"C:\Users\danrh\.cache\torch\hub\torchaudio\models\hdemucs_high_trained.pt"
)
DEMUCS_CHECKPOINT_BYTES = 334_697_255
DEMUCS_CHECKPOINT_SHA256 = (
    "a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816"
)

START_NODE_ID = 116
DIRECTOR_NODE_ID = 186
VIDEO_DURATION_NODE_ID = 178
LYRICS_NODE_ID = 556
SONG_DURATION_NODE_ID = 562
AUDIO_SELECTOR_NODE_ID = 610
AUDIO_GUIDE_NODE_ID = 611
LEGACY_LYRIC_WINDOW_NODE_ID = 621
PERFORMANCE_MODE_NODE_ID = 675
PLANNER_NODE_ID = 677

SEPARATOR_TYPE = "FL_Audio_Separation"
WHISPER_LOADER_TYPE = "Load Whisper (mtb)"
ANALYZER_TYPE = "DiffusionGemmaTimedLyricsAnalyzer"
PLANNER_TYPE = "DiffusionGemmaAudioAwareMultiShotPlanner"


class WorkflowError(RuntimeError):
    """Raised when a guarded migration invariant is not satisfied."""


def _file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _file_sha256(payload)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


def _all_nodes(workflow: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield from workflow.get("nodes", [])
    for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
        yield from subgraph.get("nodes", [])


class IdAllocator:
    def __init__(self, workflow: dict[str, Any]) -> None:
        node_ids = [int(node["id"]) for node in _all_nodes(workflow)]
        link_ids = [int(link[0]) for link in workflow.get("links", [])]
        for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
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
        matches = [node for node in self.nodes if int(node.get("id", -1)) == node_id]
        if len(matches) != 1:
            raise WorkflowError(f"Expected one main-graph node {node_id}; found {len(matches)}.")
        return matches[0]

    def one(self, node_type: str) -> dict[str, Any]:
        matches = [node for node in self.nodes if node.get("type") == node_type]
        if len(matches) != 1:
            raise WorkflowError(f"Expected one {node_type}; found {len(matches)}.")
        return matches[0]

    @staticmethod
    def input_slot(node: dict[str, Any], name: str) -> int:
        matches = [
            index
            for index, item in enumerate(node.get("inputs", []))
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
            index
            for index, item in enumerate(node.get("outputs", []))
            if item.get("name") == name
        ]
        if len(matches) != 1:
            raise WorkflowError(
                f"Node {node.get('id')} ({node.get('type')}) requires one output {name!r}."
            )
        return matches[0]

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
        if target["inputs"][input_slot].get("link") is not None:
            raise WorkflowError(
                f"Refusing to replace existing link on {target.get('id')}:{input_name}."
            )
        if link_type is None:
            link_type = str(
                target["inputs"][input_slot].get("type")
                or origin["outputs"][output_slot].get("type")
            )
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
        output_links = origin["outputs"][output_slot].get("links")
        if not isinstance(output_links, list):
            output_links = []
            origin["outputs"][output_slot]["links"] = output_links
        output_links.append(link_id)
        return link_id


def _input(name: str, type_name: str, *, optional: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {"name": name, "type": type_name, "link": None}
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
    cnr_id: str | None = "diffusiongemma-prompt-builder",
) -> dict[str, Any]:
    properties: dict[str, Any] = {"Node name for S&R": node_type}
    if cnr_id:
        properties["cnr_id"] = cnr_id
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
        "properties": properties,
        "widgets_values": list(widgets or []),
    }


def _preview_node(
    ids: IdAllocator, title: str, pos: tuple[float, float], size: tuple[float, float]
) -> dict[str, Any]:
    return _node(
        ids,
        "PreviewAny",
        title,
        pos,
        size,
        [_input("source", "*", optional=True)],
        [],
        [],
        cnr_id="comfyui-easy-use",
    )


def _set_first_widget(node: dict[str, Any], value: Any) -> None:
    widgets = node.get("widgets_values")
    if not isinstance(widgets, list) or not widgets:
        raise WorkflowError(
            f"Node {node.get('id')} ({node.get('type')}) has no first widget to update."
        )
    widgets[0] = value


def _set_director_cache_reuse(node: dict[str, Any]) -> None:
    widgets = node.get("widgets_values")
    if not isinstance(widgets, list) or len(widgets) != 6:
        raise WorkflowError(
            f"Node {node.get('id')} ({node.get('type')}) has an unexpected Director widget layout."
        )
    widgets[-1] = "reuse"


def _assert_node_type(node: dict[str, Any], expected: str) -> None:
    if node.get("type") != expected:
        raise WorkflowError(
            f"Node {node.get('id')} is {node.get('type')!r}, expected {expected!r}."
        )


def _link_by_id(workflow: dict[str, Any], link_id: int) -> list[Any]:
    matches = [link for link in workflow.get("links", []) if int(link[0]) == link_id]
    if len(matches) != 1:
        raise WorkflowError(f"Expected one main-graph link {link_id}; found {len(matches)}.")
    return matches[0]


def _origin(
    workflow: dict[str, Any], target: dict[str, Any], input_name: str
) -> tuple[dict[str, Any], str]:
    graph = MainGraph(workflow, IdAllocator(workflow))
    input_slot = graph.input_slot(target, input_name)
    link_id = target["inputs"][input_slot].get("link")
    if link_id is None:
        raise WorkflowError(f"{target.get('id')}:{input_name} is not connected.")
    link = _link_by_id(workflow, int(link_id))
    origin = graph.node(int(link[1]))
    output = origin["outputs"][int(link[2])]
    return origin, str(output.get("name"))


def _assert_origin(
    workflow: dict[str, Any],
    target: dict[str, Any],
    input_name: str,
    origin_id: int,
    output_name: str,
) -> None:
    origin, actual_output = _origin(workflow, target, input_name)
    if int(origin["id"]) != origin_id or actual_output != output_name:
        raise WorkflowError(
            f"{target.get('id')}:{input_name} must come from "
            f"{origin_id}:{output_name}, got {origin.get('id')}:{actual_output}."
        )


def _sensitive_key_paths(value: Any, prefix: str = "") -> list[str]:
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
    hits: list[str] = []
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


def _append_start_note(node: dict[str, Any]) -> None:
    widgets = node.get("widgets_values")
    if not isinstance(widgets, list) or not widgets or not isinstance(widgets[0], str):
        raise WorkflowError("The START HERE note is missing its text widget.")
    if LEGACY_AUDIO_QC_COPY in widgets[0]:
        widgets[0] = widgets[0].replace(LEGACY_AUDIO_QC_COPY, CURRENT_AUDIO_QC_COPY)
    addition = f"""

## Music audition + QC (policy revision 4)

{CURRENT_AUDIO_QC_COPY}

## H3 generation-lane boundary policy (revision 2)

{CURRENT_BOUNDARY_POLICY_COPY}

## Music-performance choice (V5)

- **Natural / audio-led sync** is the default. The soundtrack alone decides whether the subject naturally articulates; no lyric text is used as a mouth-motion schedule.
- **Dance / music sync** deliberately suppresses singing/lip-sync and concentrates on whole-body choreography, expressive and physically coherent H3 camera work, and beat/dynamic synchronization.
- **Lyrics + lip sync** first separates the vocal stem and aligns the audible vocal intervals to the authored lyrics. Only verified, hash-locked timing events reach the H3 lane prompts. Instrumental gaps remain explicitly non-vocal. Missing, stale, weak, or failed analysis safely becomes Natural/audio-led behavior instead of forcing singing.

The lyric-window node below remains an internal lexical-candidate helper. Its old local dropdown is informational only; the single authoritative control is **H3 PERFORMANCE MODE**.
""".rstrip()
    if "## Music-performance choice (V5)" not in widgets[0]:
        widgets[0] = widgets[0].rstrip() + addition + "\n"
    node["title"] = "START HERE — dual identity / guarded relay / Natural audio-led default"


def _add_timing_branch(
    workflow: dict[str, Any], graph: MainGraph, ids: IdAllocator
) -> dict[str, int]:
    selector = graph.node(AUDIO_SELECTOR_NODE_ID)
    guide = graph.node(AUDIO_GUIDE_NODE_ID)
    lyrics = graph.node(LYRICS_NODE_ID)
    song_duration = graph.node(SONG_DURATION_NODE_ID)
    video_duration = graph.node(VIDEO_DURATION_NODE_ID)
    performance = graph.node(PERFORMANCE_MODE_NODE_ID)
    planner = graph.node(PLANNER_NODE_ID)

    separator = graph.add(
        _node(
            ids,
            SEPARATOR_TYPE,
            "VOCAL STEM — Demucs separation (Lyrics mode only)",
            (1788.0, -3680.0),
            (760.0, 240.0),
            [_input("audio", "AUDIO")],
            [
                _output("bass", "AUDIO"),
                _output("drums", "AUDIO"),
                _output("other", "AUDIO"),
                _output("vocals", "AUDIO"),
            ],
            [10.0, 0.5, "half_sine"],
            cnr_id="comfyui_fill-nodes",
        )
    )
    whisper = graph.add(
        _node(
            ids,
            WHISPER_LOADER_TYPE,
            "WHISPER LARGE-V3-TURBO — local / Lyrics mode only",
            (1788.0, -3410.0),
            (760.0, 160.0),
            [],
            [_output("pipeline", "WHISPER_PIPELINE")],
            ["large-v3-turbo", False],
            cnr_id="comfy-mtb",
        )
    )
    analyzer = graph.add(
        _node(
            ids,
            ANALYZER_TYPE,
            "TIMED LYRICS — audio-gated alignment / safe Natural fallback",
            (1788.0, -3220.0),
            (760.0, 500.0),
            [
                _input("final_audio", "AUDIO"),
                _input("performance_mode", "STRING"),
                _input("lyrics", "STRING"),
                _input("master_audio_sha256", "STRING"),
                _input("song_duration_seconds", "FLOAT"),
                _input("excerpt_start_seconds", "FLOAT"),
                _input("excerpt_duration_seconds", "FLOAT"),
                _input("vocal_stem", "AUDIO", optional=True),
                _input("whisper_pipeline", "WHISPER_PIPELINE", optional=True),
            ],
            [
                _output("timed_lyrics_report_json", "STRING"),
                _output("status", "STRING"),
                _output("safe_ready", "BOOLEAN"),
                _output("aligned_preview", "STRING"),
            ],
            [90.0, 0.0, 25.0, "en", 0.55],
        )
    )
    status_preview = graph.add(
        _preview_node(
            ids,
            "TIMED LYRICS STATUS — timing-ready or explicit Natural fallback",
            (1788.0, -2690.0),
            (760.0, 180.0),
        )
    )
    alignment_preview = graph.add(
        _preview_node(
            ids,
            "ALIGNED VOCAL EVENTS — excerpt-relative, never a lyric guess",
            (1788.0, -2480.0),
            (760.0, 260.0),
        )
    )

    planner_inputs = planner.get("inputs")
    if not isinstance(planner_inputs, list):
        raise WorkflowError("Planner inputs are not serialized as a list.")
    if any(item.get("name") == "timed_lyrics_report_json" for item in planner_inputs):
        raise WorkflowError("The V4 source already contains a timed-lyrics planner input.")
    planner_inputs.append(_input("timed_lyrics_report_json", "STRING", optional=True))

    graph.connect(guide, "final_audio", separator, "audio")
    graph.connect(guide, "final_audio", analyzer, "final_audio")
    graph.connect(performance, "performance_mode", analyzer, "performance_mode")
    graph.connect(lyrics, "lyrics", analyzer, "lyrics")
    graph.connect(selector, "waveform_sha256", analyzer, "master_audio_sha256")
    graph.connect(song_duration, "FLOAT", analyzer, "song_duration_seconds")
    graph.connect(selector, "suggested_start_seconds", analyzer, "excerpt_start_seconds")
    graph.connect(video_duration, "FLOAT", analyzer, "excerpt_duration_seconds")
    graph.connect(separator, "vocals", analyzer, "vocal_stem")
    graph.connect(whisper, "pipeline", analyzer, "whisper_pipeline")
    graph.connect(analyzer, "timed_lyrics_report_json", planner, "timed_lyrics_report_json")
    graph.connect(analyzer, "status", status_preview, "source", "*")
    graph.connect(analyzer, "aligned_preview", alignment_preview, "source", "*")

    return {
        "separator": int(separator["id"]),
        "whisper_loader": int(whisper["id"]),
        "analyzer": int(analyzer["id"]),
        "status_preview": int(status_preview["id"]),
        "alignment_preview": int(alignment_preview["id"]),
        "planner": PLANNER_NODE_ID,
        "performance_control": PERFORMANCE_MODE_NODE_ID,
        "legacy_lyric_window": LEGACY_LYRIC_WINDOW_NODE_ID,
    }


def migrate_workflow(
    source: dict[str, Any], *, source_file_sha256: str
) -> dict[str, Any]:
    if source_file_sha256.casefold() != SOURCE_V4_SHA256:
        raise WorkflowError(
            "Timed-lyrics V5 must be derived from the exact verified V4 source: "
            f"{source_file_sha256} != {SOURCE_V4_SHA256}."
        )
    if not isinstance(source.get("nodes"), list) or not isinstance(
        source.get("links"), list
    ):
        raise WorkflowError("Source workflow is missing main-graph nodes or links.")
    if MIGRATION_SCHEMA in (source.get("extra") or {}):
        raise WorkflowError("The supplied source is already a timed-lyrics V5 workflow.")

    workflow = copy.deepcopy(source)
    ids = IdAllocator(workflow)
    graph = MainGraph(workflow, ids)

    expected_types = {
        START_NODE_ID: "MarkdownNote",
        DIRECTOR_NODE_ID: "DiffusionGemmaCoTGenerator",
        VIDEO_DURATION_NODE_ID: "PrimitiveFloat",
        LYRICS_NODE_ID: "SplatStageBlueprintRouter",
        SONG_DURATION_NODE_ID: "PrimitiveFloat",
        AUDIO_SELECTOR_NODE_ID: "DiffusionGemmaAudioCandidateSelector",
        AUDIO_GUIDE_NODE_ID: "DiffusionGemmaLTXAudioGuide",
        LEGACY_LYRIC_WINDOW_NODE_ID: "DiffusionGemmaLTXPerformancePrompt",
        PERFORMANCE_MODE_NODE_ID: "DiffusionGemmaMusicVideoPerformanceMode",
        PLANNER_NODE_ID: PLANNER_TYPE,
    }
    for node_id, node_type in expected_types.items():
        _assert_node_type(graph.node(node_id), node_type)

    _append_start_note(graph.node(START_NODE_ID))
    _set_director_cache_reuse(graph.node(DIRECTOR_NODE_ID))
    graph.node(PLANNER_NODE_ID)["title"] = (
        "AUDIO-AWARE H3 MULTI-LANE PLAN — evidence-preferred / deterministic fallback"
    )
    performance = graph.node(PERFORMANCE_MODE_NODE_ID)
    _set_first_widget(performance, NATURAL_MODE)
    performance["title"] = "H3 PERFORMANCE MODE — one authoritative control / Natural default"

    legacy = graph.node(LEGACY_LYRIC_WINDOW_NODE_ID)
    _set_first_widget(legacy, NATURAL_MODE)
    legacy["title"] = (
        "INTERNAL LYRIC CANDIDATE WINDOW — mode comes only from H3 PERFORMANCE MODE"
    )
    legacy["flags"] = {**(legacy.get("flags") or {}), "collapsed": True}

    added_nodes = _add_timing_branch(workflow, graph, ids)

    planning_groups = [
        group
        for group in workflow.get("groups", [])
        if str(group.get("title", "")).startswith(
            "3. Verified H3 project + audio-aware generation-lane plan"
        )
    ]
    if len(planning_groups) != 1:
        raise WorkflowError(
            "Expected one verified H3 planning group to extend for timed lyrics."
        )
    planning_group = planning_groups[0]
    bounding = planning_group.get("bounding")
    if not isinstance(bounding, list) or len(bounding) != 4:
        raise WorkflowError("Verified H3 planning group has invalid bounds.")
    existing_bottom = float(bounding[1]) + float(bounding[3])
    bounding[1] = -3730.0
    bounding[3] = existing_bottom - float(bounding[1])
    planning_group["title"] = (
        "3. Verified H3 project + lazy timed lyrics + audio-aware generation-lane plan"
    )

    extra = workflow.setdefault("extra", {})
    if not isinstance(extra, dict):
        raise WorkflowError("Workflow extra metadata is not an object.")
    extra["workflow_note"] = (
        "MiniMax-H3 Ref2VA production graph: versioned Project Master Contract, "
        "locked ACE song/QC, evidence-preferred generation-lane planning with a "
        "deterministic balanced fallback, four lazy 5–15 second H3 invocations that "
        "preserve native multi-shot prompts and carry active shots across non-native "
        "seams, retained-tail Picture 2 continuity between lanes, exact-frame assembly, "
        "pristine ACE mux audio, and honest multi-format delivery planning."
    )
    music_marker = extra.get("diffusiongemma.music_video_production")
    if isinstance(music_marker, dict):
        music_marker["selector_policy"] = copy.deepcopy(AUDIO_SELECTOR_POLICY_MARKER)
        performance_marker = music_marker.setdefault("ltx_performance_mode", {})
        if isinstance(performance_marker, dict):
            performance_marker.update(
                {
                    "default": NATURAL_MODE,
                    "modes": list(PERFORMANCE_MODES),
                    "lyrics_policy": (
                        "authored text is lexical/pronunciation evidence only; "
                        "visible articulation requires hash-locked vocal timing"
                    ),
                    "natural_policy": (
                        "connected audio alone governs articulation; no written-lyric schedule"
                    ),
                }
            )
    expansion_marker = extra.get("diffusiongemma.minimax_music_video_expansion")
    if isinstance(expansion_marker, dict):
        expansion_marker["generation_lane_boundary_policy"] = copy.deepcopy(
            BOUNDARY_POLICY_MARKER
        )
        expansion_marker["performance_mode_contract"] = {
            "default": NATURAL_MODE,
            "modes": list(PERFORMANCE_MODES),
            "authoritative_node": PERFORMANCE_MODE_NODE_ID,
            "lyrics_require_timing_report": True,
        }

    previous_revision = int(source.get("revision", 0) or 0)
    workflow["revision"] = previous_revision + 1
    workflow["last_node_id"] = ids.last_node
    workflow["last_link_id"] = ids.last_link
    extra[MIGRATION_SCHEMA] = {
        "version": MIGRATION_VERSION,
        "source_workflow_id": str(source.get("id", "")),
        "source_workflow_revision": previous_revision,
        "source_workflow_file_sha256": source_file_sha256,
        "source_workflow_canonical_sha256": _canonical_sha256(source),
        "source_is_never_overwritten": True,
        "default_performance_mode": NATURAL_MODE,
        "performance_modes": list(PERFORMANCE_MODES),
        "timing_report": {
            "schema": TIMED_REPORT_SCHEMA,
            "version": TIMED_REPORT_VERSION,
            "audio_hash_locked": True,
            "lyrics_hash_locked": True,
            "excerpt_clock_locked": True,
            "events_are_excerpt_relative": True,
            "lyrics_are_not_a_timing_schedule": True,
        },
        "analysis": {
            "vocal_separator": "torchaudio HDEMUCS_HIGH_MUSDB_PLUS via FL_Audio_Separation",
            "demucs_checkpoint": {
                "path": DEMUCS_CHECKPOINT_PATH,
                "bytes": DEMUCS_CHECKPOINT_BYTES,
                "sha256": DEMUCS_CHECKPOINT_SHA256,
            },
            "whisper_model": "openai/whisper-large-v3-turbo",
            "whisper_local_directory": "models/whisper/whisper-large-v3-turbo",
            "whisper_model_file": {
                "relative_path": WHISPER_MODEL_RELATIVE_PATH,
                "bytes": WHISPER_MODEL_BYTES,
                "sha256": WHISPER_MODEL_SHA256,
            },
            "transcription_chunk_seconds_max": 15,
            "minimum_alignment_confidence": 0.55,
            "lazy_for_non_lyrics_modes": True,
        },
        "fallback_policy": {
            "mode": NATURAL_MODE,
            "separator_failure": "safe fallback; never original-mix vocal evidence",
            "missing_or_stale_report": "safe fallback",
            "weak_alignment": "safe fallback",
            "instrumental_intervals": "explicit non-vocal behavior",
            "generation_blocked_by_analysis_failure": False,
        },
        "director_cache_mode": {
            "node": DIRECTOR_NODE_ID,
            "default": "reuse",
            "refresh_is_one_run_only": True,
        },
        "audio_selector_policy": copy.deepcopy(AUDIO_SELECTOR_POLICY_MARKER),
        "generation_lane_boundary_policy": copy.deepcopy(BOUNDARY_POLICY_MARKER),
        "nodes": added_nodes,
    }

    validate_workflow(workflow, source=source)
    return workflow


def validate_workflow(
    workflow: dict[str, Any], *, source: dict[str, Any] | None = None
) -> None:
    graph = MainGraph(workflow, IdAllocator(workflow))
    marker = (workflow.get("extra") or {}).get(MIGRATION_SCHEMA)
    if not isinstance(marker, dict) or marker.get("version") != MIGRATION_VERSION:
        raise WorkflowError("Timed-lyrics migration marker is missing or invalid.")
    nodes = marker.get("nodes")
    if not isinstance(nodes, dict):
        raise WorkflowError("Timed-lyrics node map is missing.")
    if marker.get("audio_selector_policy") != AUDIO_SELECTOR_POLICY_MARKER:
        raise WorkflowError("Timed-lyrics audio selector policy marker is stale.")
    if marker.get("generation_lane_boundary_policy") != BOUNDARY_POLICY_MARKER:
        raise WorkflowError("Timed-lyrics generation-lane boundary policy is stale.")
    music_marker = (workflow.get("extra") or {}).get(
        "diffusiongemma.music_video_production"
    )
    if not isinstance(music_marker, dict) or music_marker.get(
        "selector_policy"
    ) != AUDIO_SELECTOR_POLICY_MARKER:
        raise WorkflowError("Inherited music-production selector policy is stale.")
    expansion_marker = (workflow.get("extra") or {}).get(
        "diffusiongemma.minimax_music_video_expansion"
    )
    if not isinstance(expansion_marker, dict) or expansion_marker.get(
        "generation_lane_boundary_policy"
    ) != BOUNDARY_POLICY_MARKER:
        raise WorkflowError("Inherited generation-lane boundary policy is stale.")

    performance = graph.node(PERFORMANCE_MODE_NODE_ID)
    legacy = graph.node(LEGACY_LYRIC_WINDOW_NODE_ID)
    director = graph.node(DIRECTOR_NODE_ID)
    _assert_node_type(director, "DiffusionGemmaCoTGenerator")
    if director.get("widgets_values", [None])[-1:] != ["reuse"]:
        raise WorkflowError("The V5 Director must reopen in reusable cache mode.")
    if performance.get("widgets_values", [None])[0] != NATURAL_MODE:
        raise WorkflowError("The authoritative performance control must default to Natural.")
    if legacy.get("widgets_values", [None])[0] != NATURAL_MODE:
        raise WorkflowError("The legacy hidden dropdown must visually agree with Natural.")
    if not bool((legacy.get("flags") or {}).get("collapsed")):
        raise WorkflowError("The legacy lyric-window node must be collapsed.")
    start_copy = " ".join(
        str(value) for value in graph.node(START_NODE_ID).get("widgets_values", [])
    )
    if CURRENT_AUDIO_QC_COPY not in start_copy or LEGACY_AUDIO_QC_COPY in start_copy:
        raise WorkflowError("START HERE does not describe audio selector policy revision 4.")
    if CURRENT_BOUNDARY_POLICY_COPY not in start_copy:
        raise WorkflowError("START HERE does not describe boundary policy revision 2.")

    separator = graph.node(int(nodes["separator"]))
    whisper = graph.node(int(nodes["whisper_loader"]))
    analyzer = graph.node(int(nodes["analyzer"]))
    status_preview = graph.node(int(nodes["status_preview"]))
    alignment_preview = graph.node(int(nodes["alignment_preview"]))
    planner = graph.node(PLANNER_NODE_ID)
    _assert_node_type(separator, SEPARATOR_TYPE)
    _assert_node_type(whisper, WHISPER_LOADER_TYPE)
    _assert_node_type(analyzer, ANALYZER_TYPE)
    if planner.get("title") != (
        "AUDIO-AWARE H3 MULTI-LANE PLAN — evidence-preferred / deterministic fallback"
    ):
        raise WorkflowError("The H3 planner boundary-policy title is stale.")
    if separator.get("widgets_values") != [10.0, 0.5, "half_sine"]:
        raise WorkflowError("Vocal separation must retain the guarded chunk/fade settings.")
    if whisper.get("widgets_values") != ["large-v3-turbo", False]:
        raise WorkflowError("Whisper must use the local large-v3-turbo model without downloads.")
    if analyzer.get("widgets_values") != [90.0, 0.0, 25.0, "en", 0.55]:
        raise WorkflowError(
            "Timed-lyrics analyzer connected-widget defaults/language/confidence changed."
        )

    planner_names = [item.get("name") for item in planner.get("inputs", [])]
    if planner_names[-1:] != ["timed_lyrics_report_json"]:
        raise WorkflowError("Timed report must append at the end of planner inputs.")
    timed_input = planner["inputs"][-1]
    if timed_input.get("type") != "STRING" or timed_input.get("shape") != 7:
        raise WorkflowError("Planner timed-report socket must remain optional STRING shape 7.")

    _assert_origin(workflow, separator, "audio", AUDIO_GUIDE_NODE_ID, "final_audio")
    _assert_origin(workflow, analyzer, "final_audio", AUDIO_GUIDE_NODE_ID, "final_audio")
    _assert_origin(
        workflow, analyzer, "performance_mode", PERFORMANCE_MODE_NODE_ID, "performance_mode"
    )
    _assert_origin(workflow, analyzer, "lyrics", LYRICS_NODE_ID, "lyrics")
    _assert_origin(
        workflow, analyzer, "master_audio_sha256", AUDIO_SELECTOR_NODE_ID, "waveform_sha256"
    )
    _assert_origin(
        workflow, analyzer, "song_duration_seconds", SONG_DURATION_NODE_ID, "FLOAT"
    )
    _assert_origin(
        workflow,
        analyzer,
        "excerpt_start_seconds",
        AUDIO_SELECTOR_NODE_ID,
        "suggested_start_seconds",
    )
    _assert_origin(
        workflow,
        analyzer,
        "excerpt_duration_seconds",
        VIDEO_DURATION_NODE_ID,
        "FLOAT",
    )
    _assert_origin(workflow, analyzer, "vocal_stem", int(separator["id"]), "vocals")
    _assert_origin(
        workflow, analyzer, "whisper_pipeline", int(whisper["id"]), "pipeline"
    )
    _assert_origin(
        workflow,
        planner,
        "timed_lyrics_report_json",
        int(analyzer["id"]),
        "timed_lyrics_report_json",
    )
    _assert_origin(workflow, status_preview, "source", int(analyzer["id"]), "status")
    _assert_origin(
        workflow, alignment_preview, "source", int(analyzer["id"]), "aligned_preview"
    )

    if int(workflow.get("last_node_id", -1)) != max(
        int(node["id"]) for node in _all_nodes(workflow)
    ):
        raise WorkflowError("last_node_id is stale.")
    main_link_ids = [int(link[0]) for link in workflow.get("links", [])]
    nested_link_ids = [
        int(link["id"])
        for subgraph in workflow.get("definitions", {}).get("subgraphs", [])
        for link in subgraph.get("links", [])
    ]
    if int(workflow.get("last_link_id", -1)) != max(main_link_ids + nested_link_ids):
        raise WorkflowError("last_link_id is stale.")
    node_ids = [int(node["id"]) for node in _all_nodes(workflow)]
    if len(node_ids) != len(set(node_ids)):
        raise WorkflowError("Workflow contains duplicate node ids.")
    if len(main_link_ids + nested_link_ids) != len(set(main_link_ids + nested_link_ids)):
        raise WorkflowError("Workflow contains duplicate link ids.")
    sensitive = _sensitive_key_paths(workflow)
    if sensitive:
        raise WorkflowError("Workflow contains sensitive values: " + ", ".join(sensitive))

    if source is not None:
        if workflow.get("id") != source.get("id"):
            raise WorkflowError("V5 must preserve the verified workflow id.")
        if int(workflow.get("revision", 0)) != int(source.get("revision", 0)) + 1:
            raise WorkflowError("V5 revision must be exactly V4 revision + 1.")
        if workflow.get("definitions") != source.get("definitions"):
            raise WorkflowError("The H3 subgraph/definitions must remain byte-semantically unchanged.")


def _write_new_or_verify(
    path: Path, payload: bytes, *, replace_owned: bool = False
) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    if existed:
        existing = path.read_bytes()
        if existing != payload:
            if not replace_owned:
                raise WorkflowError(
                    f"Refusing to overwrite a different existing V5 artifact: {path}"
                )
            try:
                owned = json.loads(existing.decode("utf-8"))
                owned_marker = owned["extra"][MIGRATION_SCHEMA]
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise WorkflowError(
                    f"Refusing to replace unrecognized artifact: {path}"
                ) from exc
            if (
                owned_marker.get("version") != MIGRATION_VERSION
                or owned_marker.get("source_workflow_file_sha256") != SOURCE_V4_SHA256
                or not bool(owned_marker.get("source_is_never_overwritten"))
            ):
                raise WorkflowError(
                    f"Refusing to replace artifact without matching V5 ownership: {path}"
                )
        else:
            return "verified"
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
    return "replaced" if existed else "created"


def build_artifact(source_path: Path = SOURCE_V4) -> tuple[dict[str, Any], bytes, str]:
    source_path = source_path.resolve()
    source_bytes = source_path.read_bytes()
    source_hash = _file_sha256(source_bytes)
    if source_hash != SOURCE_V4_SHA256:
        raise WorkflowError(
            f"Verified V4 source hash changed: {source_hash} != {SOURCE_V4_SHA256}."
        )
    try:
        source = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkflowError("Verified V4 source is not valid UTF-8 JSON.") from exc
    migrated = migrate_workflow(source, source_file_sha256=source_hash)
    return migrated, _json_bytes(migrated), source_hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_V4)
    parser.add_argument("--desktop-output", type=Path, default=DEFAULT_DESKTOP_OUTPUT)
    parser.add_argument("--user-output", type=Path, default=DEFAULT_USER_OUTPUT)
    parser.add_argument(
        "--check", action="store_true", help="Validate and print hashes without writing."
    )
    parser.add_argument(
        "--replace-owned",
        action="store_true",
        help="Atomically replace only an existing artifact carrying this migration's exact ownership marker.",
    )
    args = parser.parse_args(argv)
    migrated, payload, source_hash = build_artifact(args.source)
    result_hash = _file_sha256(payload)
    if args.check:
        print(
            f"checked source_sha256={source_hash} result_sha256={result_hash} "
            f"canonical_sha256={_canonical_sha256(migrated)}"
        )
        return 0
    actions = []
    for path in (args.desktop_output, args.user_output):
        action = _write_new_or_verify(
            path, payload, replace_owned=bool(args.replace_owned)
        )
        actions.append(f"{action}={path.resolve()}")
    print(
        " ".join(actions)
        + f" source_sha256={source_hash} result_sha256={result_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
