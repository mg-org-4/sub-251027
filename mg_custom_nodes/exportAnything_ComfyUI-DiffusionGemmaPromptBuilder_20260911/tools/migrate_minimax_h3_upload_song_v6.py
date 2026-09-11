#!/usr/bin/env python3
"""Create the guarded V6 MiniMax-H3 workflow with a lazy uploaded-song path.

The user-supplied timed-lyrics V5 export is immutable source evidence.  This
migration adds a dedicated upload loader and one lazy song-source router before
the existing decoded-waveform selector.  Generated ACE candidates and uploaded
audio therefore share the same QC, excerpt, hash-lock, guide, planner, and final
mux path without changing Compose, Cover, or production-concept semantics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


def _load_tool(filename: str, module_name: str) -> ModuleType:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import guard
        raise RuntimeError(f"Cannot load migration dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TIMED = _load_tool(
    "migrate_minimax_h3_timed_lyrics_v5.py", "diffusiongemma_timed_v5_dependency"
)
PRODUCTION = _load_tool(
    "migrate_music_video_production.py", "diffusiongemma_music_production_dependency"
)
EXPANSION = _load_tool(
    "migrate_minimax_music_video_expansion.py",
    "diffusiongemma_minimax_expansion_dependency",
)

WorkflowError = TIMED.WorkflowError
IdAllocator = TIMED.IdAllocator
MainGraph = TIMED.MainGraph
_input = TIMED._input
_output = TIMED._output
_node = TIMED._node
_origin = TIMED._origin

MIGRATION_SCHEMA = "diffusiongemma.minimax_h3_upload_song"
MIGRATION_VERSION = 1
GENERATE_MODE = "Generate with ACE-Step"
UPLOAD_MODE = "Upload song"
SOURCE_MODES = (GENERATE_MODE, UPLOAD_MODE)
UPLOAD_LOADER_TYPE = "DiffusionGemmaUploadSong"
SOURCE_ROUTER_TYPE = "DiffusionGemmaSongSourceRouter"

_SOURCE_V5_FILENAME = (
    "14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_"
    "DUAL_IDENTITY_P3_RELAY_TIMED_LYRICS_V5_20260822 (1).json"
)
_SOURCE_V5_CANDIDATES = (
    Path(r"C:\Users\danrh\Desktop") / _SOURCE_V5_FILENAME,
    Path(r"C:\Users\danrh\Desktop\diffuGemmWkflow") / _SOURCE_V5_FILENAME,
)
SOURCE_V5 = next(
    (candidate for candidate in _SOURCE_V5_CANDIDATES if candidate.is_file()),
    _SOURCE_V5_CANDIDATES[0],
)
SOURCE_V5_SHA256 = (
    "eff3e5a15bcb00b7f70290df5ac972b7f81be85fc9ea6e065f7ca1feb37ed6c3"
)
V6_FILENAME = (
    "14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_"
    "DUAL_IDENTITY_P3_RELAY_TIMED_LYRICS_UPLOAD_SONG_V6_20260823.json"
)
DEFAULT_DESKTOP_OUTPUT = Path(r"C:\Users\danrh\Desktop") / V6_FILENAME
DEFAULT_USER_OUTPUT = Path(r"C:\ComfyUI\app\user\default\workflows") / V6_FILENAME
REPOSITORY_EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "15_minimax_h3_ref2va_music_video_v6.json"
)
REPOSITORY_EXAMPLE_CANONICAL_SHA256 = (
    "e3fd1889891fd92f75580a34210fb94e90ae3468d01c8275f06e425e4d46a875"
)

START_NODE_ID = 116
BLUEPRINT_ROUTER_NODE_ID = 556
GENERATED_DURATION_NODE_ID = 562
PRODUCTION_CONCEPT_NODE_ID = 589
ACE_REFERENCE_MODE_NODE_ID = 592
EXPECTED_BPM_NODE_ID = 609
AUDIO_SELECTOR_NODE_ID = 610
LYRIC_WINDOW_NODE_ID = 621
ACE_CANDIDATES_NODE_ID = 623
PROJECT_CONTRACT_NODE_ID = 676
H3_PLANNER_NODE_ID = 677
TIMED_ANALYZER_NODE_ID = 733
AUDIO_GUIDANCE_NODE_ID = 624

LEGACY_DANCE_HELP = (
    "- **Dance / music sync** deliberately suppresses singing/lip-sync and concentrates "
    "on body, camera-safe scene, and beat/dynamic synchronization."
)
H3_DANCE_HELP = (
    "- **Dance / music sync** deliberately suppresses singing/lip-sync and concentrates "
    "on whole-body choreography, expressive and physically coherent H3 camera work, "
    "and beat/dynamic synchronization."
)
LEGACY_H3_AUDIO_GUIDANCE = (
    "Use the generated ACE-Step song as the exact soundtrack and timing authority. "
    "Synchronize body motion, scene changes, cuts, and restrained camera accents "
    "to measured rhythm, phrasing, dynamics, and recovery intervals. The "
    "audio-aware H3 generation-lane planner applies the selected dance or lyric/lip-sync "
    "performance contract after validation."
)
H3_AUDIO_GUIDANCE = (
    "Use the selected song as the exact soundtrack and timing authority. Synchronize body motion, "
    "scene changes, cuts, and expressive, physically coherent H3 camera choreography to measured "
    "rhythm, phrasing, dynamics, and recovery intervals. Preserve explicitly requested orbiting, "
    "swirling, sweeping or whip movement, pronounced parallax, and coherent compound paths rather "
    "than downgrading them. When camera direction is unspecified, let the active Director creativity "
    "mode choose camera energy proportional to creative strength. The audio-aware H3 generation-lane "
    "planner applies the selected dance or lyric/lip-sync performance contract after validation."
)
H3_CAMERA_POLICY_MARKER = {
    "revision": 1,
    "ltx_camera_capability_applies": False,
    "dynamic_paths_allowed": [
        "orbit",
        "swirl",
        "sweep_or_whip",
        "pronounced_parallax",
        "coherent_compound_path",
    ],
    "creative_energy_source": "director_creativity_mode_and_strength",
}

UPLOAD_SOURCE_POLICY_MARKER = {
    "revision": 1,
    "modes": list(SOURCE_MODES),
    "default": GENERATE_MODE,
    "production_concept_independent": True,
    "source_passthrough_semantics": (
        "preserves one authored ACE blueprint and still generates audio; "
        "it is not waveform passthrough"
    ),
    "ace_mode_semantics": {
        "Compose new": "generate a new ACE-Step waveform",
        "Cover reference": "generate ACE-Step audio from lazy reference conditioning",
    },
    "upload_semantics": {
        "candidate_count": 1,
        "expected_bpm_default": 0.0,
        "expected_bpm_zero_means": "signal-derived tempo analysis",
        "lyrics": "user supplied; empty is valid outside Lyrics + lip sync",
        "duration": "measured from the decoded upload",
        "file_is_embedded_in_workflow": False,
        "selector_policy": "source_locked_uploaded_song",
        "blocking_qc": [
            "valid decoded waveform",
            "requested excerpt duration",
            "non-silent signal",
            "clipping limit",
            "broadband artifact guard",
        ],
        "advisory_qc": [
            "tempo interpretation",
            "vocal presence proxy",
            "onset pressure",
            "tonal movement",
            "generated-candidate production score",
        ],
    },
    "lazy_contract": {
        "upload_mode_requests_generated_metadata_or_candidates": False,
        "generate_mode_requests_uploaded_audio_or_metadata": False,
        "generate_mode_requests_candidates_only_through_effective_count": True,
    },
    "shared_downstream_contract": {
        "decoded_waveform_qc": "DiffusionGemmaAudioCandidateSelector",
        "selector_waveform_sha256_is_authoritative": True,
        "selector_excerpt_start_is_authoritative": True,
        "router_source_token_drives_selector_policy": True,
        "audio_guide_and_pristine_mux_unchanged": True,
    },
}

UPLOAD_START_COPY = """## Custom soundtrack upload (V6)

- **Source passthrough** is an ACE planning choice: it still generates one ACE song and does not load an audio file.
- Set **SONG SOURCE** to **Upload song**, choose a file in **UPLOAD SONG**, and optionally enter its known BPM and matching lyrics. BPM `0` uses signal-derived tempo analysis.
- Upload mode requests only the uploaded waveform and its measured metadata. ACE candidates remain lazy. Generate mode requests only the existing ACE branch, so an unselected upload never affects Compose or Cover.
- An uploaded song is source-locked: malformed, too-short, silent, clipped, or artifact-like audio still blocks, while tempo aliases, vocal presence, onset density, tonal movement, and generated-candidate score are reported as advisories instead of rejecting your chosen music. The existing selector remains the authority for waveform SHA-256, excerpt start, sync-safe guide construction, and the pristine soundtrack mux.
- Matching uploaded lyrics are needed only for **Lyrics + lip sync**. Natural/audio-led and Dance/music-sync modes do not require lyric text.
""".strip()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"


def _all_nodes(workflow: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield from workflow.get("nodes", [])
    for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
        yield from subgraph.get("nodes", [])


def _widget_input(name: str, type_name: str) -> dict[str, Any]:
    value = _input(name, type_name)
    value["widget"] = {"name": name}
    return value


def _lazy_input(name: str, type_name: str) -> dict[str, Any]:
    value = _input(name, type_name, optional=True)
    return value


def _disconnect_input(
    workflow: dict[str, Any], graph: MainGraph, target: dict[str, Any], input_name: str
) -> list[Any]:
    slot = graph.input_slot(target, input_name)
    link_id = target["inputs"][slot].get("link")
    if link_id is None:
        raise WorkflowError(
            f"Cannot reroute unconnected input {target.get('id')}:{input_name}."
        )
    matches = [
        link for link in workflow.get("links", []) if int(link[0]) == int(link_id)
    ]
    if len(matches) != 1:
        raise WorkflowError(f"Expected one link {link_id}; found {len(matches)}.")
    link = matches[0]
    origin = graph.node(int(link[1]))
    output_slot = int(link[2])
    links = origin["outputs"][output_slot].get("links")
    if isinstance(links, list):
        origin["outputs"][output_slot]["links"] = [
            item for item in links if int(item) != int(link_id)
        ]
    workflow["links"] = [
        item for item in workflow.get("links", []) if int(item[0]) != int(link_id)
    ]
    graph.links = workflow["links"]
    target["inputs"][slot]["link"] = None
    return link


def _assert_node_type(node: dict[str, Any], expected: str) -> None:
    if node.get("type") != expected:
        raise WorkflowError(
            f"Node {node.get('id')} is {node.get('type')!r}; expected {expected!r}."
        )


def _source_node_contract(graph: MainGraph) -> None:
    expected = {
        START_NODE_ID: "MarkdownNote",
        BLUEPRINT_ROUTER_NODE_ID: "SplatStageBlueprintRouter",
        GENERATED_DURATION_NODE_ID: "PrimitiveFloat",
        PRODUCTION_CONCEPT_NODE_ID: "DiffusionGemmaMusicProductionConcept",
        ACE_REFERENCE_MODE_NODE_ID: "DiffusionGemmaACEReferenceMode",
        EXPECTED_BPM_NODE_ID: "CM_IntToFloat",
        AUDIO_SELECTOR_NODE_ID: "DiffusionGemmaAudioCandidateSelector",
        LYRIC_WINDOW_NODE_ID: "DiffusionGemmaLTXPerformancePrompt",
        PROJECT_CONTRACT_NODE_ID: "DiffusionGemmaProjectMasterContract",
        H3_PLANNER_NODE_ID: "DiffusionGemmaAudioAwareMultiShotPlanner",
        TIMED_ANALYZER_NODE_ID: "DiffusionGemmaTimedLyricsAnalyzer",
        AUDIO_GUIDANCE_NODE_ID: "PrimitiveStringMultiline",
    }
    for node_id, node_type in expected.items():
        _assert_node_type(graph.node(node_id), node_type)
    candidates = graph.node(ACE_CANDIDATES_NODE_ID)
    if [item.get("name") for item in candidates.get("outputs", [])][:4] != [
        "AUDIO",
        "AUDIO_1",
        "AUDIO_2",
        "AUDIO_3",
    ]:
        raise WorkflowError("The four-output ACE candidate group contract changed.")


def _normalize_inherited_contracts(workflow: dict[str, Any], graph: MainGraph) -> None:
    extra = workflow.setdefault("extra", {})
    if not isinstance(extra, dict):
        raise WorkflowError("Workflow extra metadata is not an object.")
    timed_marker = extra.get(TIMED.MIGRATION_SCHEMA)
    if not isinstance(timed_marker, dict) or timed_marker.get("version") != 1:
        raise WorkflowError("The source is not a recognized timed-lyrics V5 workflow.")
    if timed_marker.get("source_workflow_file_sha256") != TIMED.SOURCE_V4_SHA256:
        raise WorkflowError("The timed-lyrics V5 source lineage is not the verified V4.")

    music_marker = extra.get(PRODUCTION.MIGRATION_SCHEMA)
    if not isinstance(music_marker, dict):
        raise WorkflowError("The inherited music-production marker is missing.")
    if int(music_marker.get("version", 0)) not in {10, PRODUCTION.MIGRATION_VERSION}:
        raise WorkflowError("The inherited music-production marker is unsupported.")
    music_marker["version"] = PRODUCTION.MIGRATION_VERSION
    music_marker["selector_policy"] = copy.deepcopy(
        PRODUCTION.SELECTOR_POLICY_MARKER
    )

    expansion_marker = extra.get(EXPANSION.MIGRATION_SCHEMA)
    if not isinstance(expansion_marker, dict) or int(
        expansion_marker.get("version", 0)
    ) != EXPANSION.MIGRATION_VERSION:
        raise WorkflowError("The inherited MiniMax expansion marker is unsupported.")
    expansion_marker["performance_mode_contract"] = copy.deepcopy(
        EXPANSION.PERFORMANCE_MODE_MARKER
    )
    expansion_marker["generation_lane_boundary_policy"] = copy.deepcopy(
        EXPANSION.BOUNDARY_POLICY_MARKER
    )

    timed_marker["audio_selector_policy"] = copy.deepcopy(
        TIMED.AUDIO_SELECTOR_POLICY_MARKER
    )
    timed_marker["generation_lane_boundary_policy"] = copy.deepcopy(
        TIMED.BOUNDARY_POLICY_MARKER
    )

    planner = graph.node(H3_PLANNER_NODE_ID)
    planner["title"] = (
        "AUDIO-AWARE H3 MULTI-LANE PLAN — evidence-preferred / deterministic fallback"
    )

    start = graph.node(START_NODE_ID)
    widgets = start.get("widgets_values")
    if not isinstance(widgets, list) or not widgets or not isinstance(widgets[0], str):
        raise WorkflowError("START HERE is missing its text widget.")
    copy_text = widgets[0]
    if TIMED.LEGACY_AUDIO_QC_COPY in copy_text:
        copy_text = copy_text.replace(
            TIMED.LEGACY_AUDIO_QC_COPY, TIMED.CURRENT_AUDIO_QC_COPY
        )
    if TIMED.CURRENT_AUDIO_QC_COPY not in copy_text:
        copy_text = (
            copy_text.rstrip()
            + "\n\n## Music audition + QC (policy revision 4)\n\n"
            + TIMED.CURRENT_AUDIO_QC_COPY
        )
    if TIMED.CURRENT_BOUNDARY_POLICY_COPY not in copy_text:
        copy_text = (
            copy_text.rstrip()
            + "\n\n## H3 generation-lane boundary policy (revision 2)\n\n"
            + TIMED.CURRENT_BOUNDARY_POLICY_COPY
        )
    if UPLOAD_START_COPY not in copy_text:
        copy_text = copy_text.rstrip() + "\n\n" + UPLOAD_START_COPY + "\n"
    copy_text = copy_text.replace(LEGACY_DANCE_HELP, H3_DANCE_HELP)
    widgets[0] = copy_text
    start["title"] = (
        "START HERE — upload or ACE song / dual identity / Natural audio-led default"
    )

    guidance = graph.node(AUDIO_GUIDANCE_NODE_ID)
    guidance_widgets = guidance.get("widgets_values")
    if (
        not isinstance(guidance_widgets, list)
        or len(guidance_widgets) != 1
        or not isinstance(guidance_widgets[0], str)
    ):
        raise WorkflowError("H3 audio/performance guidance is missing its text widget.")
    if guidance_widgets[0] == LEGACY_H3_AUDIO_GUIDANCE:
        guidance_widgets[0] = H3_AUDIO_GUIDANCE
    elif guidance_widgets[0] != H3_AUDIO_GUIDANCE:
        raise WorkflowError("The inherited H3 audio/performance guidance is unrecognized.")


def _add_upload_branch(
    workflow: dict[str, Any], graph: MainGraph, ids: IdAllocator
) -> dict[str, int]:
    loader = graph.add(
        _node(
            ids,
            UPLOAD_LOADER_TYPE,
            "UPLOAD SONG — choose a local audio file (Upload mode only)",
            (235.0, 118.0),
            (535.0, 135.0),
            [_widget_input("audio_file", "COMBO")],
            [
                _output("audio", "AUDIO"),
                _output("duration_seconds", "FLOAT"),
                _output("waveform_sha256", "STRING"),
                _output("status", "STRING"),
                _output("ready", "BOOLEAN"),
            ],
            [""],
        )
    )
    router = graph.add(
        _node(
            ids,
            SOURCE_ROUTER_TYPE,
            "SONG SOURCE — Generate with ACE-Step / Upload song",
            (230.0, 285.0),
            (545.0, 625.0),
            [
                _widget_input("source_mode", "COMBO"),
                _widget_input("uploaded_expected_bpm", "FLOAT"),
                _widget_input("uploaded_lyrics", "STRING"),
                _lazy_input("generated_candidate_count", "INT"),
                _lazy_input("generated_expected_bpm", "FLOAT"),
                _lazy_input("generated_lyrics", "STRING"),
                _lazy_input("generated_duration_seconds", "FLOAT"),
                _lazy_input("generated_candidate_1", "AUDIO"),
                _lazy_input("generated_candidate_2", "AUDIO"),
                _lazy_input("generated_candidate_3", "AUDIO"),
                _lazy_input("generated_candidate_4", "AUDIO"),
                _lazy_input("uploaded_audio", "AUDIO"),
                _lazy_input("uploaded_duration_seconds", "FLOAT"),
                _lazy_input("uploaded_waveform_sha256", "STRING"),
                _lazy_input("uploaded_status", "STRING"),
                _lazy_input("uploaded_ready", "BOOLEAN"),
            ],
            [
                _output("candidate_1", "AUDIO"),
                _output("candidate_2", "AUDIO"),
                _output("candidate_3", "AUDIO"),
                _output("candidate_4", "AUDIO"),
                _output("effective_candidate_count", "INT"),
                _output("effective_expected_bpm", "FLOAT"),
                _output("selected_lyrics", "STRING"),
                _output("selected_duration_seconds", "FLOAT"),
                _output("source_token", "STRING"),
                _output("status", "STRING"),
                _output("ready", "BOOLEAN"),
            ],
            [GENERATE_MODE, 0.0, ""],
        )
    )

    concept = graph.node(PRODUCTION_CONCEPT_NODE_ID)
    bpm = graph.node(EXPECTED_BPM_NODE_ID)
    lyrics = graph.node(BLUEPRINT_ROUTER_NODE_ID)
    duration = graph.node(GENERATED_DURATION_NODE_ID)
    candidates = graph.node(ACE_CANDIDATES_NODE_ID)
    selector = graph.node(AUDIO_SELECTOR_NODE_ID)
    lyric_window = graph.node(LYRIC_WINDOW_NODE_ID)
    project = graph.node(PROJECT_CONTRACT_NODE_ID)
    timed = graph.node(TIMED_ANALYZER_NODE_ID)

    generated_links = [
        (concept, "candidate_count", "generated_candidate_count", "INT"),
        (bpm, "FLOAT", "generated_expected_bpm", "FLOAT"),
        (lyrics, "lyrics", "generated_lyrics", "STRING"),
        (duration, "FLOAT", "generated_duration_seconds", "FLOAT"),
        (candidates, "AUDIO", "generated_candidate_1", "AUDIO"),
        (candidates, "AUDIO_1", "generated_candidate_2", "AUDIO"),
        (candidates, "AUDIO_2", "generated_candidate_3", "AUDIO"),
        (candidates, "AUDIO_3", "generated_candidate_4", "AUDIO"),
    ]
    for origin, output_name, input_name, link_type in generated_links:
        graph.connect(origin, output_name, router, input_name, link_type)

    upload_links = [
        ("audio", "uploaded_audio", "AUDIO"),
        ("duration_seconds", "uploaded_duration_seconds", "FLOAT"),
        ("waveform_sha256", "uploaded_waveform_sha256", "STRING"),
        ("status", "uploaded_status", "STRING"),
        ("ready", "uploaded_ready", "BOOLEAN"),
    ]
    for output_name, input_name, link_type in upload_links:
        graph.connect(loader, output_name, router, input_name, link_type)

    for input_name in (
        "candidate_1",
        "candidate_2",
        "candidate_3",
        "candidate_4",
        "candidate_count",
        "expected_bpm",
    ):
        _disconnect_input(workflow, graph, selector, input_name)
    if "source_policy" not in [item.get("name") for item in selector.get("inputs", [])]:
        selector.setdefault("inputs", []).append(_lazy_input("source_policy", "STRING"))
    selector_links = [
        ("candidate_1", "candidate_1", "AUDIO"),
        ("candidate_2", "candidate_2", "AUDIO"),
        ("candidate_3", "candidate_3", "AUDIO"),
        ("candidate_4", "candidate_4", "AUDIO"),
        ("effective_candidate_count", "candidate_count", "INT"),
        ("effective_expected_bpm", "expected_bpm", "FLOAT"),
        ("source_token", "source_policy", "STRING"),
    ]
    for output_name, input_name, link_type in selector_links:
        graph.connect(router, output_name, selector, input_name, link_type)

    for target, input_name in (
        (lyric_window, "lyrics"),
        (project, "lyrics"),
        (timed, "lyrics"),
    ):
        _disconnect_input(workflow, graph, target, input_name)
        graph.connect(router, "selected_lyrics", target, input_name, "STRING")
    for target in (lyric_window, timed):
        _disconnect_input(workflow, graph, target, "song_duration_seconds")
        graph.connect(
            router,
            "selected_duration_seconds",
            target,
            "song_duration_seconds",
            "FLOAT",
        )

    selector["title"] = (
        "MUSIC INPUT + QC — ACE candidates or uploaded-song verification"
    )
    for group in workflow.get("groups", []):
        if int(group.get("id", -1)) == 35:
            bounds = group.get("bounding")
            if not isinstance(bounds, list) or len(bounds) != 4:
                raise WorkflowError("The ACE/QC group has invalid bounds.")
            bottom = max(float(bounds[1]) + float(bounds[3]), 950.0)
            bounds[3] = bottom - float(bounds[1])
            group["title"] = (
                "ACE generation or uploaded song → waveform QC + locked soundtrack"
            )
            break
    else:
        raise WorkflowError("The ACE/QC workflow group is missing.")

    return {"upload_loader": int(loader["id"]), "source_router": int(router["id"])}


def migrate_workflow(
    source: dict[str, Any], *, source_file_sha256: str
) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise WorkflowError("Workflow root must be a JSON object.")
    existing = (source.get("extra") or {}).get(MIGRATION_SCHEMA)
    if existing is not None:
        if not isinstance(existing, dict) or existing.get("version") != MIGRATION_VERSION:
            raise WorkflowError("The upload-song migration marker is unsupported.")
        result = copy.deepcopy(source)
        graph = MainGraph(result, IdAllocator(result))
        _source_node_contract(graph)
        _normalize_inherited_contracts(result, graph)
        result["extra"][MIGRATION_SCHEMA]["h3_camera_policy"] = copy.deepcopy(
            H3_CAMERA_POLICY_MARKER
        )
        validate_workflow(result)
        return result
    if source_file_sha256.casefold() != SOURCE_V5_SHA256:
        raise WorkflowError(
            "Upload-song V6 must be derived from the exact attached V5 export: "
            f"{source_file_sha256} != {SOURCE_V5_SHA256}."
        )
    if not isinstance(source.get("nodes"), list) or not isinstance(
        source.get("links"), list
    ):
        raise WorkflowError("Source workflow is missing nodes or links.")

    workflow = copy.deepcopy(source)
    ids = IdAllocator(workflow)
    graph = MainGraph(workflow, ids)
    _source_node_contract(graph)
    _normalize_inherited_contracts(workflow, graph)
    added = _add_upload_branch(workflow, graph, ids)

    extra = workflow["extra"]
    extra["workflow_note"] = (
        "MiniMax-H3 Ref2VA V6: Generate with ACE-Step and Upload song are lazy, "
        "mutually exclusive sources feeding the same decoded-waveform selector. "
        "Compose, Cover, Source passthrough, dual identity, timed-lyrics fallback, "
        "generation-lane planning, relay, assembly, and pristine soundtrack mux are preserved."
    )
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
        "policy": copy.deepcopy(UPLOAD_SOURCE_POLICY_MARKER),
        "h3_camera_policy": copy.deepcopy(H3_CAMERA_POLICY_MARKER),
        "nodes": added,
        "preserved_controls": {
            "production_concept": PRODUCTION_CONCEPT_NODE_ID,
            "ace_reference_mode": ACE_REFERENCE_MODE_NODE_ID,
            "audio_selector": AUDIO_SELECTOR_NODE_ID,
            "performance_mode": TIMED.PERFORMANCE_MODE_NODE_ID,
            "project_contract": PROJECT_CONTRACT_NODE_ID,
        },
    }

    validate_workflow(workflow, source=source)
    return workflow


def _assert_origin(
    workflow: dict[str, Any], target: dict[str, Any], input_name: str,
    origin_id: int, output_name: str,
) -> None:
    origin, actual_output = _origin(workflow, target, input_name)
    if int(origin["id"]) != int(origin_id) or actual_output != output_name:
        raise WorkflowError(
            f"{target.get('id')}:{input_name} must come from "
            f"{origin_id}:{output_name}, got {origin.get('id')}:{actual_output}."
        )


def _validate_integrity(workflow: dict[str, Any]) -> None:
    node_ids = [int(node["id"]) for node in _all_nodes(workflow)]
    if len(node_ids) != len(set(node_ids)):
        raise WorkflowError("Workflow contains duplicate node ids.")
    main_links = [int(link[0]) for link in workflow.get("links", [])]
    nested_links = [
        int(link["id"])
        for subgraph in workflow.get("definitions", {}).get("subgraphs", [])
        for link in subgraph.get("links", [])
    ]
    if len(main_links + nested_links) != len(set(main_links + nested_links)):
        raise WorkflowError("Workflow contains duplicate link ids.")
    if int(workflow.get("last_node_id", -1)) != max(node_ids):
        raise WorkflowError("last_node_id is stale.")
    if int(workflow.get("last_link_id", -1)) != max(main_links + nested_links):
        raise WorkflowError("last_link_id is stale.")
    link_by_id = {int(link[0]): link for link in workflow.get("links", [])}
    main_nodes = {int(node["id"]): node for node in workflow.get("nodes", [])}
    for link_id, link in link_by_id.items():
        origin = main_nodes.get(int(link[1]))
        target = main_nodes.get(int(link[3]))
        if origin is None or target is None:
            raise WorkflowError(f"Main link {link_id} references a missing node.")
        if int(link[2]) >= len(origin.get("outputs", [])) or int(link[4]) >= len(
            target.get("inputs", [])
        ):
            raise WorkflowError(f"Main link {link_id} references a missing socket.")
        if int(target["inputs"][int(link[4])].get("link", -1)) != link_id:
            raise WorkflowError(f"Main link {link_id} target backlink is stale.")
        output_links = origin["outputs"][int(link[2])].get("links") or []
        if link_id not in [int(value) for value in output_links]:
            raise WorkflowError(f"Main link {link_id} origin backlink is stale.")


def validate_workflow(
    workflow: dict[str, Any], *, source: dict[str, Any] | None = None
) -> None:
    graph = MainGraph(workflow, IdAllocator(workflow))
    marker = (workflow.get("extra") or {}).get(MIGRATION_SCHEMA)
    if not isinstance(marker, dict) or marker.get("version") != MIGRATION_VERSION:
        raise WorkflowError("Upload-song V6 marker is missing or invalid.")
    if marker.get("policy") != UPLOAD_SOURCE_POLICY_MARKER:
        raise WorkflowError("Upload-song source policy marker is stale.")
    if marker.get("h3_camera_policy") != H3_CAMERA_POLICY_MARKER:
        raise WorkflowError("MiniMax H3 camera policy marker is missing or stale.")
    nodes = marker.get("nodes")
    if not isinstance(nodes, dict):
        raise WorkflowError("Upload-song node map is missing.")
    loader = graph.node(int(nodes.get("upload_loader", -1)))
    router = graph.node(int(nodes.get("source_router", -1)))
    _assert_node_type(loader, UPLOAD_LOADER_TYPE)
    _assert_node_type(router, SOURCE_ROUTER_TYPE)
    if [item.get("name") for item in loader.get("inputs", [])] != ["audio_file"]:
        raise WorkflowError("Upload loader input contract changed.")
    if [item.get("name") for item in loader.get("outputs", [])] != [
        "audio", "duration_seconds", "waveform_sha256", "status", "ready"
    ]:
        raise WorkflowError("Upload loader output contract changed.")
    expected_router_inputs = [
        "source_mode", "uploaded_expected_bpm", "uploaded_lyrics",
        "generated_candidate_count", "generated_expected_bpm", "generated_lyrics",
        "generated_duration_seconds", "generated_candidate_1", "generated_candidate_2",
        "generated_candidate_3", "generated_candidate_4", "uploaded_audio",
        "uploaded_duration_seconds", "uploaded_waveform_sha256", "uploaded_status",
        "uploaded_ready",
    ]
    if [item.get("name") for item in router.get("inputs", [])] != expected_router_inputs:
        raise WorkflowError("Song-source router input contract changed.")
    router_widgets = router.get("widgets_values")
    if not isinstance(router_widgets, list) or len(router_widgets) != 3:
        raise WorkflowError("Song-source router widgets are malformed.")
    if router_widgets[0] not in SOURCE_MODES:
        raise WorkflowError("Song-source router has an unsupported saved source mode.")
    try:
        uploaded_bpm = float(router_widgets[1])
    except (TypeError, ValueError) as exc:
        raise WorkflowError("Uploaded-song BPM must be numeric.") from exc
    if not 0.0 <= uploaded_bpm <= 300.0:
        raise WorkflowError("Uploaded-song BPM must be from 0 to 300.")
    if not isinstance(router_widgets[2], str):
        raise WorkflowError("Uploaded-song lyrics must remain a string widget.")
    expected_router_outputs = [
        "candidate_1", "candidate_2", "candidate_3", "candidate_4",
        "effective_candidate_count", "effective_expected_bpm", "selected_lyrics",
        "selected_duration_seconds", "source_token", "status", "ready",
    ]
    if [item.get("name") for item in router.get("outputs", [])] != expected_router_outputs:
        raise WorkflowError("Song-source router output contract changed.")
    for item in router.get("inputs", [])[3:]:
        if item.get("shape") != 7:
            raise WorkflowError(
                f"Song-source input {item.get('name')} must remain optional/lazy shape 7."
            )

    generated_origins = {
        "generated_candidate_count": (PRODUCTION_CONCEPT_NODE_ID, "candidate_count"),
        "generated_expected_bpm": (EXPECTED_BPM_NODE_ID, "FLOAT"),
        "generated_lyrics": (BLUEPRINT_ROUTER_NODE_ID, "lyrics"),
        "generated_duration_seconds": (GENERATED_DURATION_NODE_ID, "FLOAT"),
        "generated_candidate_1": (ACE_CANDIDATES_NODE_ID, "AUDIO"),
        "generated_candidate_2": (ACE_CANDIDATES_NODE_ID, "AUDIO_1"),
        "generated_candidate_3": (ACE_CANDIDATES_NODE_ID, "AUDIO_2"),
        "generated_candidate_4": (ACE_CANDIDATES_NODE_ID, "AUDIO_3"),
    }
    for input_name, (origin_id, output_name) in generated_origins.items():
        _assert_origin(workflow, router, input_name, origin_id, output_name)
    for input_name, output_name in (
        ("uploaded_audio", "audio"),
        ("uploaded_duration_seconds", "duration_seconds"),
        ("uploaded_waveform_sha256", "waveform_sha256"),
        ("uploaded_status", "status"),
        ("uploaded_ready", "ready"),
    ):
        _assert_origin(workflow, router, input_name, int(loader["id"]), output_name)

    selector = graph.node(AUDIO_SELECTOR_NODE_ID)
    selector_origins = {
        "candidate_1": "candidate_1", "candidate_2": "candidate_2",
        "candidate_3": "candidate_3", "candidate_4": "candidate_4",
        "candidate_count": "effective_candidate_count",
        "expected_bpm": "effective_expected_bpm",
        "source_policy": "source_token",
    }
    for input_name, output_name in selector_origins.items():
        _assert_origin(workflow, selector, input_name, int(router["id"]), output_name)
    _assert_origin(
        workflow, graph.node(LYRIC_WINDOW_NODE_ID), "lyrics", int(router["id"]),
        "selected_lyrics",
    )
    _assert_origin(
        workflow, graph.node(PROJECT_CONTRACT_NODE_ID), "lyrics", int(router["id"]),
        "selected_lyrics",
    )
    _assert_origin(
        workflow, graph.node(TIMED_ANALYZER_NODE_ID), "lyrics", int(router["id"]),
        "selected_lyrics",
    )
    for target_id in (LYRIC_WINDOW_NODE_ID, TIMED_ANALYZER_NODE_ID):
        _assert_origin(
            workflow, graph.node(target_id), "song_duration_seconds", int(router["id"]),
            "selected_duration_seconds",
        )
    # The H3 planner must continue receiving the bounded lyric window rather than
    # an unbounded full-song upload.
    _assert_origin(
        workflow, graph.node(H3_PLANNER_NODE_ID), "lyrics", LYRIC_WINDOW_NODE_ID,
        "selected_lyrics",
    )

    extra = workflow.get("extra") or {}
    music = extra.get(PRODUCTION.MIGRATION_SCHEMA)
    if not isinstance(music, dict) or int(music.get("version", 0)) != PRODUCTION.MIGRATION_VERSION:
        raise WorkflowError("Inherited music-production marker is not current.")
    if music.get("selector_policy") != PRODUCTION.SELECTOR_POLICY_MARKER:
        raise WorkflowError("Inherited selector policy is stale.")
    expansion = extra.get(EXPANSION.MIGRATION_SCHEMA)
    if not isinstance(expansion, dict) or expansion.get(
        "generation_lane_boundary_policy"
    ) != EXPANSION.BOUNDARY_POLICY_MARKER:
        raise WorkflowError("Inherited boundary policy is stale.")
    timed = extra.get(TIMED.MIGRATION_SCHEMA)
    if not isinstance(timed, dict) or timed.get(
        "audio_selector_policy"
    ) != TIMED.AUDIO_SELECTOR_POLICY_MARKER:
        raise WorkflowError("Timed-lyrics selector policy is stale.")
    if timed.get("generation_lane_boundary_policy") != TIMED.BOUNDARY_POLICY_MARKER:
        raise WorkflowError("Timed-lyrics boundary policy is stale.")
    start_text = " ".join(
        str(value) for value in graph.node(START_NODE_ID).get("widgets_values", [])
    )
    for required in (
        TIMED.CURRENT_AUDIO_QC_COPY,
        TIMED.CURRENT_BOUNDARY_POLICY_COPY,
        UPLOAD_START_COPY,
    ):
        if required not in start_text:
            raise WorkflowError("START HERE upload/current-policy help is incomplete.")
    if H3_DANCE_HELP not in start_text or LEGACY_DANCE_HELP in start_text:
        raise WorkflowError("START HERE still carries the retired H3 camera-safe guidance.")
    guidance_text = " ".join(
        str(value)
        for value in graph.node(AUDIO_GUIDANCE_NODE_ID).get("widgets_values", [])
    )
    if guidance_text != H3_AUDIO_GUIDANCE:
        raise WorkflowError("The expressive H3 audio/performance camera guidance is missing.")
    if graph.node(H3_PLANNER_NODE_ID).get("title") != (
        "AUDIO-AWARE H3 MULTI-LANE PLAN — evidence-preferred / deterministic fallback"
    ):
        raise WorkflowError("The current H3 boundary-policy title is missing.")

    _validate_integrity(workflow)
    sensitive = TIMED._sensitive_key_paths(workflow)
    if sensitive:
        raise WorkflowError("Workflow contains sensitive values: " + ", ".join(sensitive))

    if source is not None:
        if workflow.get("id") != source.get("id"):
            raise WorkflowError("V6 must preserve the attached V5 workflow id.")
        if int(workflow.get("revision", 0)) != int(source.get("revision", 0)) + 1:
            raise WorkflowError("V6 revision must be exactly attached V5 revision + 1.")
        if workflow.get("definitions") != source.get("definitions"):
            raise WorkflowError("The H3 subgraphs/definitions must remain unchanged.")
        for node_id in (
            PRODUCTION_CONCEPT_NODE_ID,
            ACE_REFERENCE_MODE_NODE_ID,
            AUDIO_SELECTOR_NODE_ID,
            TIMED.PERFORMANCE_MODE_NODE_ID,
            PROJECT_CONTRACT_NODE_ID,
        ):
            if graph.node(node_id).get("widgets_values") != MainGraph(
                source, IdAllocator(source)
            ).node(node_id).get("widgets_values"):
                raise WorkflowError(f"Saved user control widgets changed on node {node_id}.")


def _write_new_or_verify(
    path: Path, payload: bytes, *, replace_owned: bool = False
) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    if existed:
        current = path.read_bytes()
        if current == payload:
            return "verified"
        if not replace_owned:
            raise WorkflowError(f"Refusing to overwrite a different V6 artifact: {path}")
        try:
            owned = json.loads(current.decode("utf-8"))
            marker = owned["extra"][MIGRATION_SCHEMA]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise WorkflowError(f"Refusing to replace unrecognized artifact: {path}") from exc
        if (
            marker.get("version") != MIGRATION_VERSION
            or marker.get("source_workflow_file_sha256") != SOURCE_V5_SHA256
            or not bool(marker.get("source_is_never_overwritten"))
        ):
            raise WorkflowError(
                f"Refusing to replace artifact without matching V6 ownership: {path}"
            )
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


def build_artifact(
    source_path: Path = SOURCE_V5,
) -> tuple[dict[str, Any], bytes, str]:
    source_path = source_path.resolve()
    source_bytes = source_path.read_bytes()
    source_hash = _sha256_bytes(source_bytes)
    if source_hash != SOURCE_V5_SHA256:
        raise WorkflowError(
            f"Attached V5 source hash changed: {source_hash} != {SOURCE_V5_SHA256}."
        )
    try:
        source = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkflowError("Attached V5 source is not valid UTF-8 JSON.") from exc
    migrated = migrate_workflow(source, source_file_sha256=source_hash)
    return migrated, _json_bytes(migrated), source_hash


def upgrade_owned_artifact(path: Path) -> tuple[str, str]:
    """Upgrade one recognized V6 artifact when its immutable V5 source is unavailable."""

    path = path.resolve()
    try:
        current = json.loads(path.read_bytes().decode("utf-8"))
        marker = current["extra"][MIGRATION_SCHEMA]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise WorkflowError(f"Cannot read a recognized V6 artifact: {path}") from exc
    if (
        marker.get("version") != MIGRATION_VERSION
        or marker.get("source_workflow_file_sha256") != SOURCE_V5_SHA256
        or not bool(marker.get("source_is_never_overwritten"))
    ):
        raise WorkflowError(f"Refusing to upgrade an unrecognized V6 artifact: {path}")
    upgraded = migrate_workflow(current, source_file_sha256="already-owned-v6")
    payload = _json_bytes(upgraded)
    result = _write_new_or_verify(path, payload, replace_owned=True)
    return result, _sha256_bytes(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_V5)
    parser.add_argument("--desktop-output", type=Path, default=DEFAULT_DESKTOP_OUTPUT)
    parser.add_argument("--user-output", type=Path, default=DEFAULT_USER_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--replace-owned", action="store_true")
    parser.add_argument(
        "--upgrade-owned-v6",
        action="store_true",
        help="Upgrade both recognized active V6 copies without requiring the immutable V5 source.",
    )
    args = parser.parse_args(argv)
    if args.upgrade_owned_v6:
        desktop_result, desktop_hash = upgrade_owned_artifact(args.desktop_output)
        user_result, user_hash = upgrade_owned_artifact(args.user_output)
        if desktop_hash != user_hash:
            raise WorkflowError("Upgraded Desktop and ComfyUI V6 copies are not byte-identical.")
        print(
            json.dumps(
                {
                    "desktop": desktop_result,
                    "user": user_result,
                    "sha256": desktop_hash,
                },
                sort_keys=True,
            )
        )
        return 0
    migrated, payload, source_hash = build_artifact(args.source)
    result_hash = _sha256_bytes(payload)
    if args.check:
        print(
            f"checked source_sha256={source_hash} result_sha256={result_hash} "
            f"canonical_sha256={_canonical_sha256(migrated)}"
        )
        return 0
    actions = []
    for path in (args.desktop_output, args.user_output):
        actions.append(
            f"{_write_new_or_verify(path, payload, replace_owned=args.replace_owned)}="
            f"{path.resolve()}"
        )
    print(
        f"source_sha256={source_hash} result_sha256={result_hash} "
        + " ".join(actions)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
