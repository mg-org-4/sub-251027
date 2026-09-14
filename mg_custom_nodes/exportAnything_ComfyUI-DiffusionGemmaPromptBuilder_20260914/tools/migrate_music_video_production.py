#!/usr/bin/env python3
"""Migrate an ACE-Step/LTX-2.5 production workflow to production v11.

The migration is intentionally graph-aware: it updates node backlinks, keeps the
two embedded LTX stages intact, and validates every edited link before writing.
It is safe to run more than once. A revision-specific sibling backup is made
before the first write so each user-owned workflow remains recoverable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable


MIGRATION_SCHEMA = "diffusiongemma.music_video_production"
MIGRATION_VERSION = 11
PERFORMANCE_NODE_TYPE = "DiffusionGemmaLTXPerformancePrompt"
PERFORMANCE_MODES = (
    "Dance / music sync",
    "Lyrics + lip sync",
    "Natural / audio-led sync",
)
# Keep the two historical indices stable for saved COMBO widgets while making
# newly authored controls prefer an audio-led performance with no forced mouth
# schedule.
DEFAULT_PERFORMANCE_MODE = PERFORMANCE_MODES[-1]
PERFORMANCE_WIDGET_DEFAULTS = [DEFAULT_PERFORMANCE_MODE, 90.0, 0.0, 20.0]
CAMERA_CAPABILITY_INPUT = "camera_capability"
CAMERA_CAPABILITIES = (
    "Stable / base model",
    "Advanced / controlled camera",
)
DEFAULT_CAMERA_CAPABILITY = CAMERA_CAPABILITIES[0]
BACKUP_SUFFIX = ".pre_camera_capability_v10.bak"
CAMERA_CAPABILITY_MARKER = {
    "input": CAMERA_CAPABILITY_INPUT,
    "default": DEFAULT_CAMERA_CAPABILITY,
    "options": list(CAMERA_CAPABILITIES),
    "stable_contract": "base-model-safe camera motion; no unrequested orbit, roll, swirl, or pronounced parallax",
    "advanced_contract": "explicitly requested controlled-camera path; requires matching control or LoRA capability",
}
LEGACY_PERFORMANCE_PREVIEW_TITLE = (
    "LTX PERFORMANCE LYRIC WINDOW — empty in dance/music-sync mode"
)
PERFORMANCE_PREVIEW_TITLE = (
    "LTX PERFORMANCE LYRIC CANDIDATES — empty unless Lyrics + lip sync is selected"
)
LEGACY_PERFORMANCE_NOTE_LINE = (
    "6. **LTX performance mode** defaults to **Dance / music sync**: it sends no "
    "lyric text and asks for whole-body beat/phrase synchronization without visible "
    "phoneme articulation. Switch to **Lyrics + lip sync** to append the section-aware "
    "lyric window for the selected excerpt; the connected audio remains timing "
    "authority, and the visible lyric-window preview shows exactly what LTX receives. "
    "Changing this control does not regenerate the locked song."
)
PERFORMANCE_NOTE_LINE = (
    "6. **LTX performance mode** defaults to **Natural / audio-led sync**: it adds "
    "neither lyric text nor a forced mouth schedule and lets the connected soundtrack "
    "lead motion naturally. Choose **Dance / music sync** for deliberate whole-body "
    "beat/phrase synchronization without visible phoneme articulation. Choose "
    "**Lyrics + lip sync** to append bounded section-aware words as lexical and "
    "pronunciation hints only; lyrics are not a timing schedule, and connected audio "
    "remains timing authority. Changing this control does not regenerate the locked song."
)
PERFORMANCE_LYRICS_POLICY = (
    "audio-gated lexical/pronunciation hints only; lyrics are not a timing schedule"
)
ACE_4B_LANGUAGE_MODEL = "qwen_4b_ace15.safetensors"
ACE_4B_LANGUAGE_MODEL_URL = (
    "https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/resolve/main/"
    "split_files/text_encoders/qwen_4b_ace15.safetensors"
)
ACE_4B_LANGUAGE_MODEL_SHA256 = (
    "ffe5ffb855086c2ab55e467e9859fb01894781020a0376484dd19de166b79873"
)
ACE_4B_METADATA_MARKER = {
    "second_language_model": ACE_4B_LANGUAGE_MODEL,
    "second_language_model_url": ACE_4B_LANGUAGE_MODEL_URL,
    "second_language_model_sha256": ACE_4B_LANGUAGE_MODEL_SHA256,
    "second_language_model_parameter_class": "4B",
    "second_language_model_role": "ACE-Step 1.5 language model",
}
SELECTOR_POLICY_MARKER = {
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
DEFAULT_WORKFLOWS = (
    Path(r"C:\ComfyUI\app\user\default\workflows\14_ltx25_i2v+AUDIO_IN_PROCESS_SYNC.json"),
    Path(r"C:\Users\danrh\Desktop\14_ltx25_i2v+AUDIO_IN_PROCESS_SYNC.json"),
)
LTX_SUBGRAPH_ID = "d9d4c7f2-3c3b-4c2a-9f8b-2a1e3d4c5b6a"

LEGACY_FORCED_LIP_SYNC_GUIDANCE = (
    "Use the generated ACE-Step song as the exact soundtrack. If a visible performer "
    "is present, they sing or lip-sync the lead vocal in exact time and dance to its "
    "beat; preserve continuous audiovisual synchronization."
)
NEUTRAL_PERFORMANCE_GUIDANCE = (
    "Use the generated ACE-Step song as the exact soundtrack. Synchronize body motion, "
    "scene changes, and camera accents to its measured rhythm, phrasing, dynamics, and "
    "recovery intervals. Leave visible vocal performance unspecified; the downstream "
    "LTX Performance Prompt chooses it after validation."
)


class WorkflowError(RuntimeError):
    """Raised when a workflow cannot be migrated without guessing."""


class IdAllocator:
    def __init__(self, workflow: dict[str, Any]) -> None:
        subgraphs = workflow.get("definitions", {}).get("subgraphs", [])
        node_ids = [int(node["id"]) for node in workflow.get("nodes", []) if int(node["id"]) >= 0]
        link_ids = [int(link[0]) for link in workflow.get("links", [])]
        for subgraph in subgraphs:
            node_ids.extend(
                int(node["id"])
                for node in subgraph.get("nodes", [])
                if int(node["id"]) >= 0
            )
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
        for node in self.nodes:
            if int(node["id"]) == int(node_id):
                return node
        raise WorkflowError(f"Missing required node {node_id}.")

    def one(self, node_type: str) -> dict[str, Any]:
        matches = [node for node in self.nodes if node.get("type") == node_type]
        if len(matches) != 1:
            raise WorkflowError(
                f"Expected one {node_type} node, found {len(matches)}."
            )
        return matches[0]

    @staticmethod
    def input_slot(node: dict[str, Any], name: str) -> int:
        for index, item in enumerate(node.get("inputs", [])):
            if item.get("name") == name:
                return index
        raise WorkflowError(f"Node {node['id']} ({node['type']}) has no input {name!r}.")

    @staticmethod
    def output_slot(node: dict[str, Any], name: str) -> int:
        for index, item in enumerate(node.get("outputs", [])):
            if item.get("name") == name:
                return index
        raise WorkflowError(f"Node {node['id']} ({node['type']}) has no output {name!r}.")

    def remove_link(self, link_id: int) -> None:
        match = next((link for link in self.links if int(link[0]) == int(link_id)), None)
        if match is None:
            return
        _, origin_id, origin_slot, target_id, target_slot, _ = match
        origin = self.node(int(origin_id))
        target = self.node(int(target_id))
        output = origin["outputs"][int(origin_slot)]
        if isinstance(output.get("links"), list):
            output["links"] = [value for value in output["links"] if int(value) != int(link_id)]
        if target["inputs"][int(target_slot)].get("link") == link_id:
            target["inputs"][int(target_slot)]["link"] = None
        self.links.remove(match)

    def disconnect_input(self, node: dict[str, Any], name: str) -> None:
        item = node["inputs"][self.input_slot(node, name)]
        if item.get("link") is not None:
            self.remove_link(int(item["link"]))

    def remove_incident_links(self, node_ids: Iterable[int]) -> None:
        wanted = {int(value) for value in node_ids}
        for link in list(self.links):
            if int(link[1]) in wanted or int(link[3]) in wanted:
                self.remove_link(int(link[0]))

    def connect(
        self,
        origin: dict[str, Any],
        output_name: str,
        target: dict[str, Any],
        input_name: str,
        link_type: str | None = None,
    ) -> int:
        origin_slot = self.output_slot(origin, output_name)
        target_slot = self.input_slot(target, input_name)
        existing = target["inputs"][target_slot].get("link")
        if existing is not None:
            self.remove_link(int(existing))
        if link_type is None:
            link_type = str(target["inputs"][target_slot].get("type") or origin["outputs"][origin_slot].get("type"))
        link_id = self.ids.link()
        self.links.append(
            [link_id, int(origin["id"]), origin_slot, int(target["id"]), target_slot, link_type]
        )
        target["inputs"][target_slot]["link"] = link_id
        links = origin["outputs"][origin_slot].get("links")
        if not isinstance(links, list):
            links = []
            origin["outputs"][origin_slot]["links"] = links
        links.append(link_id)
        return link_id

    def add(self, node: dict[str, Any]) -> dict[str, Any]:
        node["order"] = max(
            (int(item.get("order", 0)) for item in self.nodes), default=0
        ) + 1
        self.nodes.append(node)
        return node

    def clone_reset(
        self,
        template: dict[str, Any],
        *,
        node_id: int | None = None,
        title: str | None = None,
        pos: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        node = copy.deepcopy(template)
        node["id"] = self.ids.node() if node_id is None else int(node_id)
        node["order"] = max((int(item.get("order", 0)) for item in self.nodes), default=0) + 1
        if title is not None:
            node["title"] = title
        if pos is not None:
            node["pos"] = [float(pos[0]), float(pos[1])]
        for item in node.get("inputs", []):
            item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = []
        return node


def _input(
    name: str,
    type_name: str,
    *,
    widget: bool = False,
    optional: bool = False,
    label: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "localized_name": label or name,
        "name": name,
        "type": type_name,
        "link": None,
    }
    if widget:
        item["widget"] = {"name": name}
    if optional:
        item["shape"] = 7
    return item


def _output(name: str, type_name: str) -> dict[str, Any]:
    return {
        "localized_name": name,
        "name": name,
        "type": type_name,
        "links": [],
    }


def _custom_node(
    ids: IdAllocator,
    node_type: str,
    title: str,
    pos: tuple[float, float],
    size: tuple[float, float],
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    widgets: list[Any],
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
        "properties": {
            "cnr_id": "diffusiongemma-prompt-builder",
            "Node name for S&R": node_type,
        },
        "widgets_values": widgets,
    }


def _replace_node(graph: MainGraph, replacement: dict[str, Any]) -> dict[str, Any]:
    node_id = int(replacement["id"])
    for index, existing in enumerate(graph.nodes):
        if int(existing["id"]) == node_id:
            graph.nodes[index] = replacement
            return replacement
    raise WorkflowError(f"Cannot replace absent node {node_id}.")


def _ensure_input(
    node: dict[str, Any],
    name: str,
    type_name: str,
    *,
    widget: bool = False,
    optional: bool = False,
    label: str | None = None,
) -> None:
    if any(item.get("name") == name for item in node.get("inputs", [])):
        return
    node.setdefault("inputs", []).append(
        _input(name, type_name, widget=widget, optional=optional, label=label)
    )


def _seed_from_router(node: dict[str, Any]) -> int:
    values = node.get("widgets_values", [])
    if values and isinstance(values[0], (int, float)):
        return int(values[0])
    return 26_073_001


def _all_workflow_nodes(workflow: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield main-graph and embedded-subgraph nodes without changing topology."""
    yield from workflow.get("nodes", [])
    for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
        yield from subgraph.get("nodes", [])


def _upgrade_ace_4b_contract(workflow: dict[str, Any], marker: dict[str, Any]) -> None:
    """Select the official ACE 4B LM and refresh only its visible metadata."""
    clip_loaders = [
        node for node in workflow.get("nodes", []) if node.get("type") == "DualCLIPLoader"
    ]
    if len(clip_loaders) != 1:
        raise WorkflowError(
            f"ACE 4B upgrade requires one DualCLIPLoader; found {len(clip_loaders)}."
        )
    clip_values = list(clip_loaders[0].get("widgets_values") or [])
    if len(clip_values) < 2:
        raise WorkflowError("ACE 4B upgrade found malformed DualCLIPLoader widgets.")
    clip_values[1] = ACE_4B_LANGUAGE_MODEL
    clip_loaders[0]["widgets_values"] = clip_values
    loader_properties = clip_loaders[0].setdefault("properties", {})
    model_assets = loader_properties.get("models")
    if not isinstance(model_assets, list) or len(model_assets) < 2:
        raise WorkflowError("ACE 4B upgrade found malformed DualCLIPLoader model metadata.")
    second_asset = dict(model_assets[1])
    second_asset.update(
        {
            "name": ACE_4B_LANGUAGE_MODEL,
            "url": ACE_4B_LANGUAGE_MODEL_URL,
            "directory": "text_encoders",
        }
    )
    model_assets[1] = second_asset

    for node in _all_workflow_nodes(workflow):
        title = str(node.get("title", ""))
        if "ACE Candidate" not in title:
            continue
        if "ACE 1.7B LM" in title:
            node["title"] = title.replace("ACE 1.7B LM", "ACE 4B LM")
        elif "— 4B LM" in title:
            # Normalize the earliest production-v2 label to the explicit ACE role.
            node["title"] = title.replace("— 4B LM", "— ACE 4B LM")

    canonical = marker.setdefault("canonical_ace", {})
    canonical.update(copy.deepcopy(ACE_4B_METADATA_MARKER))
    canonical.pop("ace_4b_upgrade", None)


def _neutralize_legacy_forced_lip_sync(workflow: dict[str, Any]) -> bool:
    """Remove only the known production-v8 forced-lip-sync profile sentence."""

    matches = [
        node
        for node in workflow.get("nodes", [])
        if node.get("type") == "DiffusionGemmaLTX25TargetProfile"
    ]
    if len(matches) != 1:
        raise WorkflowError(
            "Performance-mode migration requires one LTX target profile; "
            f"found {len(matches)}."
        )
    values = list(matches[0].get("widgets_values") or [])
    if len(values) > 4 and values[4] == LEGACY_FORCED_LIP_SYNC_GUIDANCE:
        values[4] = NEUTRAL_PERFORMANCE_GUIDANCE
        matches[0]["widgets_values"] = values
        return True
    return False


def _ensure_camera_capability_widget(workflow: dict[str, Any]) -> bool:
    """Persist the appended LTX camera-capability COMBO without inventing a socket.

    Unlinked ComfyUI widgets are serialized only in ``widgets_values``.  The
    target-profile node's ``inputs`` list therefore remains untouched; adding a
    fake COMBO input there would change graph topology and reload incorrectly.
    """

    matches = [
        node
        for node in workflow.get("nodes", [])
        if node.get("type") == "DiffusionGemmaLTX25TargetProfile"
    ]
    if len(matches) != 1:
        raise WorkflowError(
            "Camera-capability migration requires one LTX target profile; "
            f"found {len(matches)}."
        )
    target = matches[0]
    values = list(target.get("widgets_values") or [])
    changed = False

    # Seven values are accepted only as the pre-long-horizon serialization.
    # Restore that known default before appending the new final widget.
    if len(values) == 7:
        values.append("Off")
        changed = True
    if len(values) == 8:
        if values[-1] in CAMERA_CAPABILITIES:
            values.insert(7, "Off")
        else:
            values.append(DEFAULT_CAMERA_CAPABILITY)
        changed = True
    if len(values) != 9:
        raise WorkflowError(
            "LTX target profile camera migration expected 8 legacy widgets or "
            f"9 current widgets; found {len(values)}."
        )
    if values[-1] not in CAMERA_CAPABILITIES:
        raise WorkflowError(
            f"LTX target profile has unsupported camera capability {values[-1]!r}."
        )
    target["widgets_values"] = values

    # The organized workflow has room below this node. Grow only when the new
    # widget was appended so it remains visible without moving any node.
    size = target.get("size")
    if changed and isinstance(size, list) and len(size) >= 2:
        try:
            height = float(size[1])
        except (TypeError, ValueError):
            pass
        else:
            if height < 459.375:
                size[1] = 459.375
    return changed


def _normalize_performance_widget_serialization(control: dict[str, Any]) -> None:
    """Persist linked numeric widgets in the shape ComfyUI restores reliably."""

    widget_inputs = {
        "performance_mode",
        "song_duration_seconds",
        "excerpt_start_seconds",
        "excerpt_duration_seconds",
    }
    for item in control.get("inputs", []):
        name = str(item.get("name", ""))
        if name in widget_inputs:
            item["widget"] = {"name": name}

    values = list(control.get("widgets_values") or [])
    if len(values) > len(PERFORMANCE_WIDGET_DEFAULTS):
        raise WorkflowError("LTX performance control has extra serialized widgets.")
    values.extend(PERFORMANCE_WIDGET_DEFAULTS[len(values) :])
    if values[0] not in PERFORMANCE_MODES:
        raise WorkflowError(f"Performance control has unsupported mode {values[0]!r}.")
    control["widgets_values"] = values


def _ensure_performance_mode_control(
    workflow: dict[str, Any], graph: MainGraph, ids: IdAllocator
) -> dict[str, Any]:
    """Insert the one prompt-only Natural/Dance/Lyrics control at the LTX seam."""

    matches = [
        node for node in graph.nodes if node.get("type") == PERFORMANCE_NODE_TYPE
    ]
    if len(matches) > 1:
        raise WorkflowError(
            f"Expected at most one {PERFORMANCE_NODE_TYPE}; found {len(matches)}."
        )
    if matches:
        control = matches[0]
    else:
        control = graph.add(
            _custom_node(
                ids,
                PERFORMANCE_NODE_TYPE,
                "LTX PERFORMANCE MODE — Natural audio-led, Dance sync, or Lyrics hints",
                (1665.0, -855.0),
                (700.0, 275.0),
                [
                    _input("ltx_prompt", "STRING"),
                    _input("performance_mode", "COMBO", widget=True),
                    _input("song_duration_seconds", "FLOAT", widget=True),
                    _input("excerpt_start_seconds", "FLOAT", widget=True),
                    _input("excerpt_duration_seconds", "FLOAT", widget=True),
                    _input("lyrics", "STRING", optional=True),
                ],
                [
                    _output("ltx_prompt", "STRING"),
                    _output("selected_lyrics", "STRING"),
                    _output("status", "STRING"),
                    _output("performance_report_json", "STRING"),
                    _output("performance_mode", "STRING"),
                ],
                copy.deepcopy(PERFORMANCE_WIDGET_DEFAULTS),
            )
        )
    _normalize_performance_widget_serialization(control)

    gate = graph.node(188)
    router = graph.node(556)
    song_duration = graph.node(562)
    master_duration = graph.node(178)
    selector = graph.one("DiffusionGemmaAudioCandidateSelector")
    outer = next(
        (node for node in graph.nodes if node.get("type") == LTX_SUBGRAPH_ID),
        None,
    )
    if outer is None:
        raise WorkflowError("Performance-mode migration cannot find the outer LTX group.")

    graph.connect(gate, "prompt", control, "ltx_prompt", "STRING")
    graph.connect(router, "lyrics", control, "lyrics", "STRING")
    graph.connect(
        song_duration, "FLOAT", control, "song_duration_seconds", "FLOAT"
    )
    graph.connect(
        selector,
        "suggested_start_seconds",
        control,
        "excerpt_start_seconds",
        "FLOAT",
    )
    graph.connect(
        master_duration, "FLOAT", control, "excerpt_duration_seconds", "FLOAT"
    )
    graph.connect(control, "ltx_prompt", outer, "value", "STRING")

    previews = [
        node
        for node in graph.nodes
        if str(node.get("title", "")) == PERFORMANCE_PREVIEW_TITLE
    ]
    if len(previews) > 1:
        raise WorkflowError(
            f"Expected at most one performance lyric preview; found {len(previews)}."
        )
    if previews:
        preview = previews[0]
    else:
        preview = graph.add(
            graph.clone_reset(
                graph.node(189),
                title=PERFORMANCE_PREVIEW_TITLE,
                pos=(1665.0, -515.0),
            )
        )
        preview["size"] = [700.0, 220.0]
    graph.connect(control, "selected_lyrics", preview, "source", "STRING")
    return control


def _update_start_note(workflow: dict[str, Any]) -> None:
    notes = [node for node in workflow["nodes"] if node.get("type") == "MarkdownNote"]
    if not notes:
        return
    note = next((node for node in notes if int(node.get("id", -1)) == 116), notes[0])
    note["title"] = "START HERE — audition, lock, then synchronize ACE-Step + LTX-2.5"
    note["size"] = [1325.0, 420.0]
    note["widgets_values"] = [
        "# DiffusionGemma + ACE-Step + LTX-2.5 — production music-video test\n\n"
        "1. Replace the first-frame image and edit the creative brief. Leave the production concept on **Audition and select** for the reliable path. The visible candidate count defaults to **2**; choose 3 or 4 only when the extra ACE render time is worthwhile. An explicit soundtrack genre outranks the visual setting; **juxtaposition is not musical fusion** unless the brief explicitly requests a fusion or hybrid.\n"
        "2. The **Song plan / audition root** controls only music candidates. The separate **LTX seed** controls video sampling. Changing video duration or the LTX seed does not change the locked song.\n"
        "3. ACE uses the official ACE-specific **4B language model** (`qwen_4b_ace15.safetensors`), LM temperature 0.72, shift 3, 8 sampling steps, and CFG 1. The router passes the jointly authored caption, lyrics, BPM, key, and meter through without post-hoc rewriting. This is the ACE-Step 1.5 language model—not the unrelated `qwen_3_4b` Z-Image encoder.\n"
        "4. **Music Audition + QC** measures each decoded candidate. Expected-aligned half-time is advisory. A canonically aligned double-time subdivision is advisory when canonical error is within 5%, excerpt onset pressure is at most 4.0/s, vocal activity is present, and at least two of visual-recovery coverage, excerpt score, tonal-family consistency, or transition stability pass their strict thresholds. Canonically unrelated tempo, weakly supported or dense double-time, unstable tonality, clipping, artifacts, and unsafe excerpt density still block. The minimum score is only one numeric gate. Direct/aligned-half candidates rank first; accepted double-time candidates rank by recovery coverage, then excerpt and whole-song quality. Use a lock-candidate option to compare cached choices; use lock-by-hash plus the locked start field to freeze a winner.\n"
        "5. **LTX audio guide** defaults to sync-safe: its reduced-transient copy conditions both frozen LTX audio latents, while the pristine selected excerpt is muxed into the final video unchanged. Full mix is the A/B baseline. Vocal only fails closed until a vocal stem is connected.\n"
        f"{PERFORMANCE_NOTE_LINE}\n"
        "7. **A2V influence** inside the LTX group is 1.0 (neutral) by default. Raise it cautiously toward 2.0 only after the song passes QC; values above 1 add an extra cross-modal prediction pass.\n"
        "8. Optional ACE Cover mode is lazy. Compose new works with no reference. Cover reference stops with a clear instruction until a VAE-encoded reference-audio latent is connected to the four cover-conditioning nodes.\n\n"
        "The final soundtrack is never the processed guide: the original selected waveform remains the delivery audio."
    ]


def _performance_mode_marker() -> dict[str, Any]:
    return {
        "node_type": PERFORMANCE_NODE_TYPE,
        "default": DEFAULT_PERFORMANCE_MODE,
        "modes": list(PERFORMANCE_MODES),
        "prompt_seam": "post-validation gate, pre-LTX",
        "lyrics_policy": PERFORMANCE_LYRICS_POLICY,
    }


def _refresh_current_performance_contract(
    workflow: dict[str, Any], marker: dict[str, Any]
) -> bool:
    """Upgrade the shipped v10 two-mode metadata without changing a saved choice."""

    changed = False
    legacy_marker = {
        "node_type": PERFORMANCE_NODE_TYPE,
        "default": "Dance / music sync",
        "modes": list(PERFORMANCE_MODES[:2]),
        "prompt_seam": "post-validation gate, pre-LTX",
        "lyrics_policy": "bounded section-aware excerpt window; audio is timing authority",
    }
    current_marker = marker.get("ltx_performance_mode")
    expected_marker = _performance_mode_marker()
    if current_marker == legacy_marker:
        marker["ltx_performance_mode"] = expected_marker
        changed = True

    for node in workflow.get("nodes", []):
        title = str(node.get("title", ""))
        if title == "LTX PERFORMANCE MODE — dance sync or lyrics + lip sync":
            node["title"] = (
                "LTX PERFORMANCE MODE — Natural audio-led, Dance sync, or Lyrics hints"
            )
            changed = True
        elif title == LEGACY_PERFORMANCE_PREVIEW_TITLE:
            node["title"] = PERFORMANCE_PREVIEW_TITLE
            changed = True

        if int(node.get("id", -1)) != 116:
            continue
        values = node.get("widgets_values")
        if not isinstance(values, list):
            continue
        refreshed_values = [
            value.replace(LEGACY_PERFORMANCE_NOTE_LINE, PERFORMANCE_NOTE_LINE)
            if isinstance(value, str)
            else value
            for value in values
        ]
        if refreshed_values != values:
            node["widgets_values"] = refreshed_values
            changed = True

    extra = workflow.get("extra")
    if isinstance(extra, dict):
        workflow_note = extra.get("workflow_note")
        if isinstance(workflow_note, str):
            refreshed_note = workflow_note.replace(
                "a post-validation dance/lip-sync performance switch",
                "a post-validation Natural/Dance/Lyrics performance control",
            )
            if refreshed_note != workflow_note:
                extra["workflow_note"] = refreshed_note
                changed = True
    return changed


def _multimodal_guider_node(
    node: dict[str, Any], *, parameters_link: int | None = None
) -> dict[str, Any]:
    old_inputs = {item["name"]: item.get("link") for item in node.get("inputs", [])}
    node["type"] = "MultimodalGuider"
    node["title"] = "LTX multimodal guider — neutral CFG + adjustable A2V"
    node["size"] = [310.0, 185.0]
    node["inputs"] = [
        {"localized_name": "model", "name": "model", "type": "MODEL", "link": old_inputs.get("model")},
        {"localized_name": "positive", "name": "positive", "type": "CONDITIONING", "link": old_inputs.get("positive")},
        {"localized_name": "negative", "name": "negative", "type": "CONDITIONING", "link": old_inputs.get("negative")},
        {"localized_name": "parameters", "name": "parameters", "type": "GUIDER_PARAMETERS", "link": parameters_link},
        {"localized_name": "skip_blocks", "name": "skip_blocks", "type": "STRING", "widget": {"name": "skip_blocks"}, "link": None},
    ]
    node["properties"] = {
        "cnr_id": "ComfyUI-LTXVideo",
        "Node name for S&R": "MultimodalGuider",
    }
    node["widgets_values"] = [""]
    return node


def _guider_parameters_node(
    node_id: int,
    *,
    modality: str,
    pos: tuple[float, float],
    order: int,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "GuiderParameters",
        "pos": [float(pos[0]), float(pos[1])],
        "size": [315.0, 250.0],
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": [
            _input("modality", "COMBO", widget=True),
            _input("cfg", "FLOAT", widget=True),
            _input("stg", "FLOAT", widget=True),
            _input("perturb_attn", "BOOLEAN", widget=True),
            _input("rescale", "FLOAT", widget=True),
            _input("modality_scale", "FLOAT", widget=True),
            _input("skip_step", "INT", widget=True),
            _input("cross_attn", "BOOLEAN", widget=True),
            _input("parameters", "GUIDER_PARAMETERS", optional=True),
        ],
        "outputs": [_output("GUIDER_PARAMETERS", "GUIDER_PARAMETERS")],
        "title": f"{modality.title()} guider parameters — CFG 1 / STG 0",
        "properties": {
            "cnr_id": "ComfyUI-LTXVideo",
            "Node name for S&R": "GuiderParameters",
        },
        "widgets_values": [modality, 1.0, 0.0, False, 0.0, 1.0, 0, True],
    }


def _subgraph_by_id(workflow: dict[str, Any], subgraph_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in workflow.get("definitions", {}).get("subgraphs", [])
        if item.get("id") == subgraph_id
    ]
    if len(matches) != 1:
        raise WorkflowError(f"Expected one embedded LTX subgraph {subgraph_id}, found {len(matches)}.")
    return matches[0]


def _sg_node(subgraph: dict[str, Any], node_id: int) -> dict[str, Any]:
    for node in subgraph.get("nodes", []):
        if int(node["id"]) == int(node_id):
            return node
    raise WorkflowError(f"Embedded LTX subgraph is missing node {node_id}.")


def _sg_link(subgraph: dict[str, Any], link_id: int) -> dict[str, Any]:
    for link in subgraph.get("links", []):
        if int(link["id"]) == int(link_id):
            return link
    raise WorkflowError(f"Embedded LTX subgraph is missing link {link_id}.")


def _sg_add_link(
    subgraph: dict[str, Any],
    ids: IdAllocator,
    origin_id: int,
    origin_slot: int,
    target_id: int,
    target_slot: int,
    type_name: str,
) -> int:
    link_id = ids.link()
    subgraph["links"].append(
        {
            "id": link_id,
            "origin_id": origin_id,
            "origin_slot": origin_slot,
            "target_id": target_id,
            "target_slot": target_slot,
            "type": type_name,
        }
    )
    if origin_id >= 0:
        output = _sg_node(subgraph, origin_id)["outputs"][origin_slot]
        links = output.get("links")
        if not isinstance(links, list):
            links = []
            output["links"] = links
        links.append(link_id)
    if target_id >= 0:
        _sg_node(subgraph, target_id)["inputs"][target_slot]["link"] = link_id
    return link_id


def _migrate_ltx_subgraph(
    workflow: dict[str, Any], graph: MainGraph, ids: IdAllocator
) -> tuple[dict[str, Any], int, int]:
    subgraph = _subgraph_by_id(workflow, LTX_SUBGRAPH_ID)
    outer = next(
        (node for node in graph.nodes if node.get("type") == LTX_SUBGRAPH_ID),
        None,
    )
    if outer is None:
        raise WorkflowError("Missing outer LTX group node.")
    inputs = subgraph["inputs"]
    if len(inputs) != 16 or inputs[15].get("type") != "AUDIO":
        raise WorkflowError("Unexpected LTX subgraph interface; expected the legacy single audio slot 15.")

    # The Desktop copy predates frontend materialization of two unlinked COMBO
    # sockets. Rebuild the outer interface from the canonical embedded input
    # definition and retarget existing link slot indices by input name.
    outer_by_name = {item.get("name"): copy.deepcopy(item) for item in outer.get("inputs", [])}
    widget_names = {
        "value",
        "value_2",
        "value_3",
        "value_4",
        "value_5",
        "negative_prompt",
        "unet_name",
        "vae_name",
        "vae_name_1",
        "clip_name",
        "model_name",
    }
    rebuilt_outer_inputs: list[dict[str, Any]] = []
    for definition in inputs:
        name = str(definition["name"])
        item = outer_by_name.get(name)
        if item is None:
            item = {
                "name": name,
                "type": definition["type"],
                "link": None,
            }
            if definition.get("label"):
                item["label"] = definition["label"]
            if name in widget_names:
                item["widget"] = {"name": name}
        rebuilt_outer_inputs.append(item)
    outer["inputs"] = rebuilt_outer_inputs
    for target_slot, item in enumerate(outer["inputs"]):
        link_id = item.get("link")
        if link_id is None:
            continue
        link = next((value for value in graph.links if int(value[0]) == int(link_id)), None)
        if link is None:
            raise WorkflowError(f"Outer LTX input {item['name']} references absent link {link_id}.")
        link[4] = target_slot

    # Slot 15 conditions LTX only; slot 16 is the pristine mux soundtrack.
    conditioning = inputs[15]
    conditioning["name"] = "conditioning_audio"
    conditioning["label"] = "conditioning_audio"
    conditioning["linkIds"] = [985]
    input_y = float(conditioning.get("pos", [0, 0])[1])
    final_input_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{LTX_SUBGRAPH_ID}:final_soundtrack"))
    a2v_input_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{LTX_SUBGRAPH_ID}:a2v_influence"))
    inputs.append(
        {
            "id": final_input_id,
            "name": "final_soundtrack",
            "type": "AUDIO",
            "linkIds": [967],
            "label": "final_soundtrack",
            "pos": [float(conditioning.get("pos", [0, 0])[0]), input_y + 20.0],
        }
    )
    inputs.append(
        {
            "id": a2v_input_id,
            "name": "a2v_influence",
            "type": "FLOAT",
            "linkIds": [],
            "label": "A2V influence (1 neutral; >1 stronger / slower)",
            "pos": [float(conditioning.get("pos", [0, 0])[0]), input_y + 40.0],
        }
    )
    _sg_link(subgraph, 985)["origin_slot"] = 15
    _sg_link(subgraph, 967)["origin_slot"] = 16

    outer_input = outer["inputs"][15]
    outer_input["name"] = "conditioning_audio"
    outer_input["label"] = "conditioning_audio"
    outer_input["link"] = None
    outer["inputs"].append(
        {
            "label": "final_soundtrack",
            "name": "final_soundtrack",
            "type": "AUDIO",
            "link": None,
        }
    )
    outer["inputs"].append(
        {
            "label": "A2V influence (1 neutral; >1 stronger / slower)",
            "name": "a2v_influence",
            "type": "FLOAT",
            "widget": {"name": "a2v_influence"},
            "link": None,
        }
    )
    outer.setdefault("widgets_values", []).append(1.0)
    outer["size"] = [float(outer["size"][0]), float(outer["size"][1]) + 50.0]
    outer["title"] = "9. Native two-stage LTX-2.5 — separate guide + pristine soundtrack"
    if isinstance(subgraph.get("inputNode"), dict):
        bounding = subgraph["inputNode"].get("bounding")
        if isinstance(bounding, list) and len(bounding) == 4:
            bounding[3] = float(bounding[3]) + 40.0

    # Replace both misleading dual-CFG nodes with explicit multimodal parameters.
    max_order = max(int(node.get("order", 0)) for node in subgraph["nodes"])
    for stage, guider_id in enumerate((434, 447), start=1):
        guider = _sg_node(subgraph, guider_id)
        old_parameter_links = [
            link
            for link in list(subgraph["links"])
            if int(link["target_id"]) == guider_id and int(link["target_slot"]) >= 3
        ]
        for link in old_parameter_links:
            subgraph["links"].remove(link)
        _multimodal_guider_node(guider)
        audio_id = ids.node()
        video_id = ids.node()
        x, y = (float(guider["pos"][0]), float(guider["pos"][1]))
        audio_params = _guider_parameters_node(
            audio_id,
            modality="AUDIO",
            pos=(x - 420.0, y + 190.0),
            order=max_order + stage * 2 - 1,
        )
        video_params = _guider_parameters_node(
            video_id,
            modality="VIDEO",
            pos=(x - 420.0, y - 100.0),
            order=max_order + stage * 2,
        )
        subgraph["nodes"].extend([audio_params, video_params])
        audio_to_video = _sg_add_link(
            subgraph, ids, audio_id, 0, video_id, 8, "GUIDER_PARAMETERS"
        )
        parameters_to_guider = _sg_add_link(
            subgraph, ids, video_id, 0, guider_id, 3, "GUIDER_PARAMETERS"
        )
        a2v_link = _sg_add_link(
            subgraph, ids, -10, 17, video_id, 5, "FLOAT"
        )
        inputs[17]["linkIds"].append(a2v_link)
        # The assignments are redundant with _sg_add_link but document the chain.
        video_params["inputs"][8]["link"] = audio_to_video
        guider["inputs"][3]["link"] = parameters_to_guider

    return outer, 15, 16


def _sensitive_key_paths(value: Any, prefix: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).casefold() in {
                "api_key",
                "apikey",
                "authorization",
                "access_token",
                "refresh_token",
                "password",
            } and item not in (None, "", False):
                hits.append(path)
            hits.extend(_sensitive_key_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(_sensitive_key_paths(item, f"{prefix}[{index}]"))
    return hits


def _remove_stale_disabled_lut_branch(workflow: dict[str, Any]) -> bool:
    """Remove a dead disabled experiment that references an unavailable node."""

    subgraph = _subgraph_by_id(workflow, LTX_SUBGRAPH_ID)
    expected = {
        590: "VAEDecodeTiled",
        591: "VHS_VideoCombine",
        592: "OlmLUT",
        593: "VAEEncode",
        595: "Film Grain",
    }
    present = {
        int(node["id"]): node
        for node in subgraph.get("nodes", [])
        if int(node.get("id", -1)) in expected
    }
    if not present:
        return False
    for node_id, node in present.items():
        if node.get("type") != expected[node_id] or int(node.get("mode", 0)) != 4:
            raise WorkflowError(
                f"Refusing to remove node {node_id}: it is no longer the expected disabled LUT experiment."
            )
    branch_ids = set(present)
    for link in subgraph.get("links", []):
        if int(link["origin_id"]) in branch_ids and int(link["target_id"]) not in branch_ids:
            raise WorkflowError(
                "Refusing to remove the disabled LUT experiment because it now feeds an active branch."
            )
    removed_link_ids = {
        int(link["id"])
        for link in subgraph.get("links", [])
        if int(link["origin_id"]) in branch_ids or int(link["target_id"]) in branch_ids
    }
    subgraph["nodes"] = [
        node for node in subgraph.get("nodes", []) if int(node.get("id", -1)) not in branch_ids
    ]
    subgraph["links"] = [
        link for link in subgraph.get("links", []) if int(link["id"]) not in removed_link_ids
    ]
    for node in subgraph.get("nodes", []):
        for input_value in node.get("inputs", []):
            if input_value.get("link") in removed_link_ids:
                input_value["link"] = None
        for output in node.get("outputs", []):
            if isinstance(output.get("links"), list):
                output["links"] = [
                    int(link_id)
                    for link_id in output["links"]
                    if int(link_id) not in removed_link_ids
                ]
    subgraph["groups"] = [
        group
        for group in subgraph.get("groups", [])
        if str(group.get("title", "")).strip().casefold() != "lut test (disabled)"
    ]
    return True


def migrate_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return a migrated deep copy of one saved workflow."""
    result = copy.deepcopy(workflow)
    stale_branch_removed = _remove_stale_disabled_lut_branch(result)
    existing_marker = result.get("extra", {}).get(MIGRATION_SCHEMA)
    if existing_marker and int(existing_marker.get("version", 0)) == MIGRATION_VERSION:
        # v8 was first shipped with the active 4B widget corrected but one
        # stale 1.7B download hint in ``properties.models``. Repair that hidden
        # frontend asset metadata for already-v8 files as an idempotent hygiene
        # step, then validate the widget, marker, and asset as one contract.
        before_ace_contract = copy.deepcopy(result)
        _upgrade_ace_4b_contract(result, existing_marker)
        performance_nodes = [
            node
            for node in result.get("nodes", [])
            if node.get("type") == PERFORMANCE_NODE_TYPE
        ]
        if len(performance_nodes) != 1:
            raise WorkflowError(
                f"Current production workflow requires one {PERFORMANCE_NODE_TYPE}; "
                f"found {len(performance_nodes)}."
            )
        _normalize_performance_widget_serialization(performance_nodes[0])
        _ensure_camera_capability_widget(result)
        performance_contract_refreshed = _refresh_current_performance_contract(
            result, existing_marker
        )
        if (
            stale_branch_removed
            or performance_contract_refreshed
            or result != before_ace_contract
        ):
            result["revision"] = int(result.get("revision", 0) or 0) + 1
        validate_workflow(result)
        return result
    marker_version = int(existing_marker.get("version", 0)) if existing_marker else 0
    if existing_marker and marker_version in {2, 3, 4, 5, 6, 7, 8, 9, 10}:
        if marker_version == 2:
            clip_loader = next(
                (node for node in result["nodes"] if node.get("type") == "DualCLIPLoader"),
                None,
            )
            if clip_loader is None:
                raise WorkflowError("Cannot upgrade production v2: ACE DualCLIPLoader is missing.")
            values = list(clip_loader.get("widgets_values") or [])
            if len(values) < 2:
                raise WorkflowError("Cannot upgrade production v2: ACE loader widgets are malformed.")
            values[1] = "qwen_1.7b_ace15.safetensors"
            clip_loader["widgets_values"] = values
            for node in result["nodes"]:
                title = str(node.get("title", ""))
                if "ACE Candidate" in title and "4B LM" in title:
                    node["title"] = title.replace("4B LM", "ACE 1.7B LM")
            _update_start_note(result)
            existing_marker.setdefault("canonical_ace", {})[
                "second_language_model"
            ] = "qwen_1.7b_ace15.safetensors"
            existing_marker["canonical_ace"]["ace_4b_upgrade"] = (
                "not installed; requires qwen_4b_ace15.safetensors"
            )

        # v4 makes the fixed full-song clock explicit in both saved variants.
        ids = IdAllocator(result)
        graph = MainGraph(result, ids)
        planner = graph.node(578)
        song_duration = graph.node(562)
        current_link = next(
            (
                link
                for link in result.get("links", [])
                if int(link[3]) == int(planner["id"])
                and int(link[4]) == graph.input_slot(planner, "duration_seconds")
            ),
            None,
        )
        if current_link is None or int(current_link[1]) != int(song_duration["id"]):
            graph.connect(song_duration, "FLOAT", planner, "duration_seconds", "FLOAT")

        # v5 repairs the widget container shape. ComfyUI requires an ordered
        # widget list; the old scalar reloads as a null API input. v6/v7 update
        # selector policy/help copy. v8 selects the official ACE-specific 4B
        # language model and refreshes metadata without changing topology or
        # candidate controls.
        aura_nodes = [
            node
            for node in _all_workflow_nodes(result)
            if node.get("type") == "ModelSamplingAuraFlow"
        ]
        if len(aura_nodes) != 1:
            raise WorkflowError("Cannot upgrade production workflow: ACE AuraFlow sampler is missing.")
        if marker_version <= 8:
            aura_nodes[0]["widgets_values"] = [3.0]
        _upgrade_ace_4b_contract(result, existing_marker)
        if marker_version <= 8:
            _update_start_note(result)
            _neutralize_legacy_forced_lip_sync(result)
            _ensure_performance_mode_control(result, graph, ids)
        else:
            # A UI-resaved v9 workflow omits the unlinked performance-mode
            # COMBO from ``inputs`` and keeps it only in ``widgets_values``.
            # Preserve that canonical shape and all of its existing link IDs.
            performance_nodes = [
                node
                for node in result.get("nodes", [])
                if node.get("type") == PERFORMANCE_NODE_TYPE
            ]
            if len(performance_nodes) != 1:
                raise WorkflowError(
                    f"Production v9 requires one {PERFORMANCE_NODE_TYPE}; "
                    f"found {len(performance_nodes)}."
                )
            _normalize_performance_widget_serialization(performance_nodes[0])
        # v11 refreshes the visible selector help and governed policy marker
        # without changing audio or video topology.
        _update_start_note(result)
        _ensure_camera_capability_widget(result)
        for node in result.get("nodes", []):
            if str(node.get("title", "")).startswith("FULL MUSIC QC AUDIT"):
                node["title"] = (
                    "FULL MUSIC QC AUDIT — candidate measurements, rejection reasons + advisories"
                )
        result["last_node_id"] = ids.last_node
        result["last_link_id"] = ids.last_link
        subgraph = _subgraph_by_id(result, LTX_SUBGRAPH_ID)
        subgraph.setdefault("state", {})["lastNodeId"] = ids.last_node
        subgraph.setdefault("state", {})["lastLinkId"] = ids.last_link
        existing_marker["version"] = MIGRATION_VERSION
        existing_marker["planner_duration_source"] = "ACE full-song duration node 562"
        existing_marker["auraflow_shift_serialization"] = "widget_list_[3.0]"
        existing_marker["selector_policy"] = copy.deepcopy(SELECTOR_POLICY_MARKER)
        existing_marker["ltx_performance_mode"] = _performance_mode_marker()
        existing_marker["ltx_camera_capability"] = copy.deepcopy(
            CAMERA_CAPABILITY_MARKER
        )
        _refresh_current_performance_contract(result, existing_marker)
        result["revision"] = int(result.get("revision", 0) or 0) + 1
        validate_workflow(result)
        return result
    new_types = {
        "DiffusionGemmaMusicProductionConcept",
        "DiffusionGemmaSongSeedFanout",
        "DiffusionGemmaAudioCandidateSelector",
        "DiffusionGemmaLTXAudioGuide",
        "DiffusionGemmaACEReferenceMode",
        "DiffusionGemmaACECoverConditioning",
        PERFORMANCE_NODE_TYPE,
    }
    partial = [node["type"] for node in result.get("nodes", []) if node.get("type") in new_types]
    if partial:
        raise WorkflowError(
            "Workflow contains production nodes without its migration marker; "
            f"refusing to guess around a partial edit: {sorted(set(partial))}."
        )

    ids = IdAllocator(result)
    graph = MainGraph(result, ids)
    planner = graph.node(578)
    router = graph.node(556)
    cot = graph.node(186)
    text_template = graph.node(561)
    zero_template = graph.node(564)
    sampler_template = graph.node(566)
    decode_template = graph.node(567)
    trim = graph.node(569)
    cleanup_audio = graph.node(577)
    full_preview = graph.node(568)
    preview_template = graph.node(189)
    audio_preview_template = full_preview
    ltx_seed = graph.node(418)
    clip_loader = graph.node(559)
    vae_loader = graph.node(560)
    model_sampler = graph.node(563)
    empty_audio = graph.node(565)
    song_duration = graph.node(562)
    master_duration = graph.node(178)

    # One workflow copy was saved before ComfyUI materialized widget-backed
    # sockets into its JSON. Normalize only the known target nodes so both
    # copies migrate to the same live schema without touching user content.
    for name, type_name in (
        ("root_seed", "INT"),
        ("duration_seconds", "FLOAT"),
        ("aspect_ratio", "STRING"),
        ("temperature", "FLOAT"),
        ("max_new_tokens", "INT"),
    ):
        _ensure_input(planner, name, type_name, widget=True)
    for name, type_name in (
        ("generate_audio_codes", "BOOLEAN"),
        ("cfg_scale", "FLOAT"),
        ("temperature", "FLOAT"),
        ("top_p", "FLOAT"),
        ("top_k", "INT"),
        ("min_p", "FLOAT"),
    ):
        _ensure_input(text_template, name, type_name, widget=True)
    for name, type_name in (
        ("steps", "INT"),
        ("cfg", "FLOAT"),
        ("sampler_name", "COMBO"),
        ("scheduler", "COMBO"),
        ("denoise", "FLOAT"),
    ):
        _ensure_input(sampler_template, name, type_name, widget=True)
    _ensure_input(model_sampler, "shift", "FLOAT", widget=True)
    _ensure_input(empty_audio, "batch_size", "INT", widget=True)
    _ensure_input(full_preview, "audioUI", "AUDIO_UI", widget=True)

    # Normalize TrimAudioDuration's older two-socket serialization before new
    # links are allocated. Duration is reconnected from the master control.
    for name in ("audio", "start_index", "duration"):
        if any(item.get("name") == name for item in trim.get("inputs", [])):
            graph.disconnect_input(trim, name)
    trim["inputs"] = [
        _input("audio", "AUDIO"),
        _input("start_index", "FLOAT", widget=True),
        _input("duration", "FLOAT", widget=True),
    ]

    legacy_mode = str((router.get("widgets_values") or [0, "default"])[1])
    song_root_value = _seed_from_router(router)
    old_trim_start = float((trim.get("widgets_values") or [0.0])[0])
    master_values = master_duration.get("widgets_values")
    master_duration_value = float(master_values[0] if isinstance(master_values, list) else master_values)

    _update_start_note(result)
    planner["title"] = "SONG + VIDEO BLUEPRINT — shared production concept / fixed full-song clock"
    router["title"] = "SONG BLUEPRINT ROUTER — exact passthrough + audition provenance"
    full_preview["title"] = "SELECTED FULL SONG — decoded QC winner"
    trim["title"] = "QC-SELECTED EXCERPT — suggested start + master video duration"
    cleanup_audio["title"] = "Unload ACE models before Director and LTX conditioning"

    # Canonical ACE-Step 1.5 Turbo arm with its official ACE-specific 4B LM.
    clip_values = list(clip_loader.get("widgets_values") or [])
    if len(clip_values) < 2:
        raise WorkflowError("ACE DualCLIPLoader has an unexpected widget layout.")
    clip_values[1] = ACE_4B_LANGUAGE_MODEL
    clip_loader["widgets_values"] = clip_values
    # ComfyUI serializes widget-backed inputs as an ordered list. A scalar here
    # reloads as ``shift=None`` even though it looks numerically correct on disk.
    model_sampler["widgets_values"] = [3.0]

    # Shared production concept and song seed controls.
    concept = graph.add(
        _custom_node(
            ids,
            "DiffusionGemmaMusicProductionConcept",
            "PRODUCTION CONCEPT — audition 2–4, or one source/joint song",
            (-1440.0, 510.0),
            (520.0, 130.0),
            [
                _input("production_concept", "COMBO", widget=True),
                _input("audition_candidate_count", "INT", widget=True),
            ],
            [
                _output("planner_mode", "COMBO"),
                _output("router_mode", "COMBO"),
                _output("candidate_count", "INT"),
            ],
            ["Audition and select", 2],
        )
    )
    song_root = graph.add(
        graph.clone_reset(
            ltx_seed,
            title="Song plan / audition root — independent from LTX seed",
            pos=(-1440.0, 700.0),
        )
    )
    song_root["widgets_values"] = [song_root_value, "fixed"]
    fanout = graph.add(
        _custom_node(
            ids,
            "DiffusionGemmaSongSeedFanout",
            "ACE candidate seeds — deterministic / concept-aware",
            (-900.0, 1260.0),
            (520.0, 210.0),
            [
                _input("root_seed", "INT", widget=True),
                _input("production_concept", "COMBO", widget=True),
            ],
            [
                _output("candidate_1_seed", "INT"),
                _output("candidate_2_seed", "INT"),
                _output("candidate_3_seed", "INT"),
                _output("candidate_4_seed", "INT"),
                _output("seed_report_json", "STRING"),
            ],
            [song_root_value, "Audition and select"],
        )
    )

    # Bind the one visible concept to Planner, Router, selector, and provenance.
    if not any(item.get("name") == "production_mode" for item in planner["inputs"]):
        planner["inputs"].append(_input("production_mode", "COMBO", widget=True))
        planner.setdefault("widgets_values", []).append("Audition and select")
    if not any(item.get("name") == "candidate_count" for item in router["inputs"]):
        router["inputs"].append(
            _input("candidate_count", "INT", widget=True, optional=True, label="effective ACE candidates")
        )
        router.setdefault("widgets_values", []).append(2)
    router_values = list(router.get("widgets_values") or [])
    if len(router_values) >= 2:
        router_values[1] = "Audition and select"
    if len(router_values) < 5:
        router_values.extend([2] * (5 - len(router_values)))
    router_values[4] = 2
    router["widgets_values"] = router_values
    planner_values = list(planner.get("widgets_values") or [])
    if len(planner_values) < 6:
        planner_values.append("Audition and select")
    else:
        planner_values[5] = "Audition and select"
    planner["widgets_values"] = planner_values
    graph.connect(song_root, "seed", planner, "root_seed", "INT")
    graph.connect(song_duration, "FLOAT", planner, "duration_seconds", "FLOAT")
    graph.connect(song_root, "seed", router, "root_seed", "INT")
    graph.connect(song_root, "seed", fanout, "root_seed", "INT")
    graph.connect(concept, "planner_mode", planner, "production_mode", "COMBO")
    graph.connect(concept, "router_mode", router, "rhythm_mode", "COMBO")
    graph.connect(concept, "planner_mode", fanout, "production_concept", "COMBO")
    graph.connect(concept, "candidate_count", router, "candidate_count", "INT")

    # Compose/Cover is a lazy gate. Compose new is the runnable default.
    reference_mode = graph.add(
        _custom_node(
            ids,
            "DiffusionGemmaACEReferenceMode",
            "ACE SOURCE MODE — Compose new / optional Cover reference",
            (390.0, 930.0),
            (440.0, 105.0),
            [_input("mode", "COMBO", widget=True)],
            [_output("generate_audio_codes", "BOOLEAN"), _output("mode_token", "STRING")],
            ["Compose new"],
        )
    )

    # Replace the original candidate-A chain and clone three independent lanes.
    lane_ids = {561, 564, 566, 567}
    graph.remove_incident_links(lane_ids)
    lane_positions = [250.0, 1030.0, 1810.0, 2590.0]
    lane_letters = "ABCD"
    lane_chains: list[dict[str, dict[str, Any]]] = []
    for index, (letter, y) in enumerate(zip(lane_letters, lane_positions), start=1):
        if index == 1:
            encode = _replace_node(
                graph,
                graph.clone_reset(
                    text_template,
                    node_id=561,
                    title="ACE Candidate A — ACE 4B LM / temperature 0.72",
                    pos=(860.0, y),
                ),
            )
            zero = _replace_node(
                graph,
                graph.clone_reset(
                    zero_template,
                    node_id=564,
                    title="Candidate A negative conditioning",
                    pos=(1840.0, y + 170.0),
                ),
            )
            sampler = _replace_node(
                graph,
                graph.clone_reset(
                    sampler_template,
                    node_id=566,
                    title="ACE Candidate A — 8-step Turbo / CFG 1",
                    pos=(2070.0, y),
                ),
            )
            decode = _replace_node(
                graph,
                graph.clone_reset(
                    decode_template,
                    node_id=567,
                    title="Decode ACE Candidate A",
                    pos=(2460.0, y + 80.0),
                ),
            )
        else:
            encode = graph.add(
                graph.clone_reset(
                    text_template,
                    title=f"ACE Candidate {letter} — ACE 4B LM / temperature 0.72",
                    pos=(860.0, y),
                )
            )
            zero = graph.add(
                graph.clone_reset(
                    zero_template,
                    title=f"Candidate {letter} negative conditioning",
                    pos=(1840.0, y + 170.0),
                )
            )
            sampler = graph.add(
                graph.clone_reset(
                    sampler_template,
                    title=f"ACE Candidate {letter} — 8-step Turbo / CFG 1",
                    pos=(2070.0, y),
                )
            )
            decode = graph.add(
                graph.clone_reset(
                    decode_template,
                    title=f"Decode ACE Candidate {letter}",
                    pos=(2460.0, y + 80.0),
                )
            )
        encode_values = list(encode.get("widgets_values") or [])
        if len(encode_values) < 15:
            raise WorkflowError("ACE TextEncode node has an unexpected widget layout.")
        encode_values[9] = True
        encode_values[10] = 2.0
        encode_values[11] = 0.72
        encode_values[12] = 0.9
        encode_values[13] = 0
        encode_values[14] = 0.0
        encode["widgets_values"] = encode_values
        sampler_values = list(sampler.get("widgets_values") or [])
        if len(sampler_values) < 7:
            raise WorkflowError("ACE KSampler has an unexpected widget layout.")
        sampler_values[2:] = [8, 1.0, "euler", "simple", 1.0]
        sampler["widgets_values"] = sampler_values
        cover = graph.add(
            _custom_node(
                ids,
                "DiffusionGemmaACECoverConditioning",
                f"Candidate {letter} — lazy Cover reference gate",
                (1530.0, y + 30.0),
                (410.0, 145.0),
                [
                    _input("conditioning", "CONDITIONING"),
                    _input("mode_token", "STRING"),
                    _input("reference_latent", "LATENT", optional=True),
                ],
                [
                    _output("conditioning", "CONDITIONING"),
                    _output("status", "STRING"),
                    _output("ready", "BOOLEAN"),
                ],
                [],
            )
        )
        lane_chains.append(
            {"encode": encode, "cover": cover, "zero": zero, "sampler": sampler, "decode": decode}
        )

    # The router remains the single source of authored music fields.
    common_encode_links = (
        (clip_loader, "CLIP", "clip", "CLIP"),
        (router, "ace_tags", "tags", "STRING"),
        (router, "lyrics", "lyrics", "STRING"),
        (router, "bpm", "bpm", "INT"),
        (song_duration, "FLOAT", "duration", "FLOAT"),
        (router, "time_signature", "timesignature", "COMBO"),
        (router, "language", "language", "COMBO"),
        (router, "key", "keyscale", "COMBO"),
        (reference_mode, "generate_audio_codes", "generate_audio_codes", "BOOLEAN"),
    )
    for index, chain in enumerate(lane_chains, start=1):
        encode = chain["encode"]
        cover = chain["cover"]
        zero = chain["zero"]
        sampler = chain["sampler"]
        decode = chain["decode"]
        for origin, output_name, input_name, type_name in common_encode_links:
            graph.connect(origin, output_name, encode, input_name, type_name)
        graph.connect(fanout, f"candidate_{index}_seed", encode, "seed", "INT")
        graph.connect(encode, "CONDITIONING", cover, "conditioning", "CONDITIONING")
        graph.connect(reference_mode, "mode_token", cover, "mode_token", "STRING")
        graph.connect(cover, "conditioning", zero, "conditioning", "CONDITIONING")
        graph.connect(model_sampler, "MODEL", sampler, "model", "MODEL")
        graph.connect(cover, "conditioning", sampler, "positive", "CONDITIONING")
        graph.connect(zero, "CONDITIONING", sampler, "negative", "CONDITIONING")
        graph.connect(empty_audio, "LATENT", sampler, "latent_image", "LATENT")
        graph.connect(fanout, f"candidate_{index}_seed", sampler, "seed", "INT")
        graph.connect(sampler, "LATENT", decode, "samples", "LATENT")
        graph.connect(vae_loader, "VAE", decode, "vae", "VAE")

    bpm_converter = graph.add(
        _custom_node(
            ids,
            "CM_IntToFloat",
            "Requested BPM → numeric QC expectation",
            (2680.0, 650.0),
            (320.0, 86.0),
            [_input("a", "INT", widget=True)],
            [_output("FLOAT", "FLOAT")],
            [120],
        )
    )
    bpm_converter["properties"] = {
        "cnr_id": "ComfyMath",
        "Node name for S&R": "CM_IntToFloat",
    }
    selector = graph.add(
        _custom_node(
            ids,
            "DiffusionGemmaAudioCandidateSelector",
            "MUSIC AUDITION + QC — choose, verify, and lock before LTX",
            (2860.0, 930.0),
            (650.0, 500.0),
            [
                _input("candidate_count", "INT", widget=True),
                _input("selection_mode", "COMBO", widget=True),
                _input("expected_bpm", "FLOAT", widget=True),
                _input("excerpt_duration_seconds", "FLOAT", widget=True),
                _input("minimum_score", "FLOAT", widget=True),
                _input("locked_waveform_sha256", "STRING", widget=True, optional=True),
                _input("locked_start_seconds", "FLOAT", widget=True, optional=True),
                _input("candidate_1", "AUDIO", optional=True),
                _input("candidate_2", "AUDIO", optional=True),
                _input("candidate_3", "AUDIO", optional=True),
                _input("candidate_4", "AUDIO", optional=True),
            ],
            [
                _output("selected_audio", "AUDIO"),
                _output("suggested_start_seconds", "FLOAT"),
                _output("waveform_sha256", "STRING"),
                _output("director_report_json", "STRING"),
                _output("audit_report_json", "STRING"),
                _output("status", "STRING"),
                _output("ready", "BOOLEAN"),
            ],
            [2, "auto_select", 0.0, master_duration_value, 0.52, "", -1.0],
        )
    )
    graph.connect(concept, "candidate_count", selector, "candidate_count", "INT")
    graph.connect(router, "bpm", bpm_converter, "a", "INT")
    graph.connect(bpm_converter, "FLOAT", selector, "expected_bpm", "FLOAT")
    graph.connect(master_duration, "FLOAT", selector, "excerpt_duration_seconds", "FLOAT")
    for index, chain in enumerate(lane_chains, start=1):
        graph.connect(chain["decode"], "AUDIO", selector, f"candidate_{index}", "AUDIO")

    # Selector -> pristine trim -> cleanup -> separate guide/final paths.
    graph.disconnect_input(trim, "audio")
    graph.disconnect_input(trim, "start_index")
    graph.disconnect_input(cleanup_audio, "anything")
    graph.disconnect_input(full_preview, "audio")
    graph.connect(selector, "selected_audio", trim, "audio", "AUDIO")
    graph.connect(selector, "suggested_start_seconds", trim, "start_index", "FLOAT")
    graph.connect(master_duration, "FLOAT", trim, "duration", "FLOAT")
    graph.connect(selector, "selected_audio", full_preview, "audio", "AUDIO")
    graph.connect(trim, "AUDIO", cleanup_audio, "anything", "AUDIO")
    trim_values = list(trim.get("widgets_values") or [])
    if len(trim_values) < 2:
        trim_values = [old_trim_start, master_duration_value]
    trim_values[0] = old_trim_start
    trim["widgets_values"] = trim_values

    guide = graph.add(
        _custom_node(
            ids,
            "DiffusionGemmaLTXAudioGuide",
            "LTX AUDIO GUIDE — sync-safe conditioning / pristine final master",
            (3640.0, 940.0),
            (610.0, 230.0),
            [
                _input("final_audio", "AUDIO"),
                _input("mode", "COMBO", widget=True),
                _input("vocal_stem", "AUDIO", optional=True),
            ],
            [
                _output("conditioning_audio", "AUDIO"),
                _output("final_audio", "AUDIO"),
                _output("conditioning_sha256", "STRING"),
                _output("guide_report_json", "STRING"),
                _output("status", "STRING"),
                _output("ready", "BOOLEAN"),
            ],
            ["sync_safe"],
        )
    )
    graph.connect(cleanup_audio, "output", guide, "final_audio", "AUDIO")

    # The compact measured report is a hard scheduling dependency: all requested
    # ACE candidates decode and ACE memory is released before the 26B Director.
    report_cleanup = graph.add(
        graph.clone_reset(
            cleanup_audio,
            title="Unload ACE after QC — then pass compact measurements to Director",
            pos=(3630.0, 1270.0),
        )
    )
    if not any(item.get("name") == "measured_audio_report_json" for item in cot["inputs"]):
        cot["inputs"].append(
            _input("measured_audio_report_json", "STRING", optional=True)
        )
    graph.connect(selector, "director_report_json", report_cleanup, "anything", "STRING")
    graph.connect(report_cleanup, "output", cot, "measured_audio_report_json", "STRING")

    # Reuse/create only selected downstream previews. Never force a candidate lane
    # merely for listening; selector lazy status controls which candidates run.
    final_previews = [
        node
        for node in graph.nodes
        if int(node.get("id", -1)) == 589 and node.get("type") == "PreviewAudio"
    ]
    if final_previews:
        final_preview = final_previews[0]
        graph.disconnect_input(final_preview, "audio")
        final_preview["title"] = "FINAL SOUNDTRACK EXCERPT — pristine mux audio"
        final_preview["pos"] = [4300.0, 1130.0]
    else:
        final_preview = graph.add(
            graph.clone_reset(
                audio_preview_template,
                title="FINAL SOUNDTRACK EXCERPT — pristine mux audio",
                pos=(4300.0, 1130.0),
            )
        )
    graph.connect(guide, "final_audio", final_preview, "audio", "AUDIO")
    full_preview["pos"] = [3630.0, 680.0]

    audit_preview = graph.add(
        graph.clone_reset(
            preview_template,
            title="FULL MUSIC QC AUDIT — candidate measurements, rejection reasons + advisories",
            pos=(2860.0, 1500.0),
        )
    )
    status_preview = graph.add(
        graph.clone_reset(
            preview_template,
            title="MUSIC QC STATUS — must pass before Director + LTX",
            pos=(3520.0, 1500.0),
        )
    )
    hash_preview = graph.add(
        graph.clone_reset(
            preview_template,
            title="SELECTED SONG SHA-256 — copy when using lock-by-hash",
            pos=(3520.0, 1750.0),
        )
    )
    graph.connect(selector, "audit_report_json", audit_preview, "source", "STRING")
    graph.connect(selector, "status", status_preview, "source", "STRING")
    graph.connect(selector, "waveform_sha256", hash_preview, "source", "STRING")

    # Split the embedded LTX interface and replace dual-CFG labels with real A2V.
    outer = next(node for node in graph.nodes if node.get("type") == LTX_SUBGRAPH_ID)
    graph.disconnect_input(outer, "audio")
    outer, conditioning_slot, final_slot = _migrate_ltx_subgraph(result, graph, ids)
    graph.connect(guide, "conditioning_audio", outer, "conditioning_audio", "AUDIO")
    graph.connect(guide, "final_audio", outer, "final_soundtrack", "AUDIO")
    _neutralize_legacy_forced_lip_sync(result)
    _ensure_performance_mode_control(result, graph, ids)
    _ensure_camera_capability_widget(result)

    # Keep the fixed full-song clock separate from video duration. Existing
    # per-file prompt, image, target duration, LTX seed, and old trim value remain.
    song_duration["title"] = "ACE FULL SONG DURATION — independent from video duration"
    empty_audio["title"] = "Shared full-song ACE latent — candidates are seed-independent"
    ltx_seed["title"] = "LTX seed — video only; does not regenerate the locked song"
    result["revision"] = int(result.get("revision", 0) or 0) + 1
    result["last_node_id"] = ids.last_node
    result["last_link_id"] = ids.last_link
    subgraph = _subgraph_by_id(result, LTX_SUBGRAPH_ID)
    subgraph.setdefault("state", {})["lastNodeId"] = ids.last_node
    subgraph.setdefault("state", {})["lastLinkId"] = ids.last_link
    extra = result.setdefault("extra", {})
    extra.setdefault("seed_widgets", {})[str(song_root["id"])] = 0
    extra["workflow_note"] = (
        "DiffusionGemma production music-video graph: upstream joint planning, lazy 2–4 ACE audition, "
        "decoded-waveform QC/locking, compact measured-audio Director evidence, separate sync-safe LTX guide "
        "and pristine final soundtrack, a post-validation Natural/Dance/Lyrics performance control, plus "
        "neutral-by-default real A2V influence."
    )
    extra[MIGRATION_SCHEMA] = {
        "version": MIGRATION_VERSION,
        "production_concept": "Audition and select",
        "audition_candidate_count": 2,
        "legacy_router_mode": legacy_mode,
        "song_root_seed": song_root_value,
        "ltx_seed_node_id": int(ltx_seed["id"]),
        "master_video_duration_seconds": master_duration_value,
        "preserved_trim_widget_seconds": old_trim_start,
        "planner_duration_source": "ACE full-song duration node 562",
        "canonical_ace": {
            **copy.deepcopy(ACE_4B_METADATA_MARKER),
            "lm_temperature": 0.72,
            "sampling_shift": 3.0,
            "steps": 8,
            "cfg": 1.0,
        },
        "selector_policy": copy.deepcopy(SELECTOR_POLICY_MARKER),
        "ltx_performance_mode": _performance_mode_marker(),
        "ltx_camera_capability": copy.deepcopy(CAMERA_CAPABILITY_MARKER),
        "ltx_group_slots": {
            "conditioning_audio": conditioning_slot,
            "final_soundtrack": final_slot,
            "a2v_influence": 17,
        },
    }
    validate_workflow(result)
    return result


def validate_workflow(workflow: dict[str, Any]) -> None:
    """Validate main-graph links and the production-specific invariants."""
    node_by_id = {int(node["id"]): node for node in workflow.get("nodes", [])}
    link_ids: set[int] = set()
    for link in workflow.get("links", []):
        link_id, origin_id, origin_slot, target_id, target_slot, _ = link
        if int(link_id) in link_ids:
            raise WorkflowError(f"Duplicate main-graph link id {link_id}.")
        link_ids.add(int(link_id))
        if int(origin_id) not in node_by_id or int(target_id) not in node_by_id:
            raise WorkflowError(f"Dangling main-graph link {link_id}.")
        origin = node_by_id[int(origin_id)]
        target = node_by_id[int(target_id)]
        if int(origin_slot) >= len(origin.get("outputs", [])):
            raise WorkflowError(f"Link {link_id} has an invalid origin slot.")
        if int(target_slot) >= len(target.get("inputs", [])):
            raise WorkflowError(f"Link {link_id} has an invalid target slot.")
        if int(link_id) not in (origin["outputs"][int(origin_slot)].get("links") or []):
            raise WorkflowError(f"Link {link_id} is missing from its origin backlink.")
        if target["inputs"][int(target_slot)].get("link") != link_id:
            raise WorkflowError(f"Link {link_id} is missing from its target backlink.")

    for node in workflow.get("nodes", []):
        for index, item in enumerate(node.get("inputs", [])):
            link_id = item.get("link")
            if link_id is not None and int(link_id) not in link_ids:
                raise WorkflowError(f"Node {node['id']} input {index} references absent link {link_id}.")
        for index, item in enumerate(node.get("outputs", [])):
            for link_id in item.get("links") or []:
                if int(link_id) not in link_ids:
                    raise WorkflowError(f"Node {node['id']} output {index} references absent link {link_id}.")

    marker = workflow.get("extra", {}).get(MIGRATION_SCHEMA)
    if not marker or int(marker.get("version", 0)) != MIGRATION_VERSION:
        return

    links_by_id = {int(link[0]): link for link in workflow["links"]}

    def nodes_of_type(node_type: str) -> list[dict[str, Any]]:
        return [node for node in workflow["nodes"] if node.get("type") == node_type]

    def exactly_one(node_type: str) -> dict[str, Any]:
        matches = nodes_of_type(node_type)
        if len(matches) != 1:
            raise WorkflowError(f"Production graph requires one {node_type}; found {len(matches)}.")
        return matches[0]

    def input_item(node: dict[str, Any], name: str) -> dict[str, Any]:
        matches = [item for item in node.get("inputs", []) if item.get("name") == name]
        if len(matches) != 1:
            raise WorkflowError(f"Node {node['id']} requires one input {name!r}.")
        return matches[0]

    def input_origin(node: dict[str, Any], name: str) -> tuple[dict[str, Any], int, list[Any]]:
        link_id = input_item(node, name).get("link")
        if link_id is None:
            raise WorkflowError(f"Node {node['id']} input {name!r} must be connected.")
        link = links_by_id[int(link_id)]
        return node_by_id[int(link[1])], int(link[2]), link

    selector_policy = marker.get("selector_policy", {})
    if selector_policy != SELECTOR_POLICY_MARKER:
        raise WorkflowError("Workflow selector-policy marker is stale or malformed.")
    expected_performance_marker = _performance_mode_marker()
    if marker.get("ltx_performance_mode") != expected_performance_marker:
        raise WorkflowError("Workflow LTX performance-mode marker is stale or malformed.")
    if marker.get("ltx_camera_capability") != CAMERA_CAPABILITY_MARKER:
        raise WorkflowError("Workflow LTX camera-capability marker is stale or malformed.")
    canonical_ace = marker.get("canonical_ace", {})
    for key, expected in ACE_4B_METADATA_MARKER.items():
        if canonical_ace.get(key) != expected:
            raise WorkflowError(
                f"Workflow ACE 4B metadata {key!r} is missing or stale."
            )
    if "ace_4b_upgrade" in canonical_ace:
        raise WorkflowError("Workflow still describes ACE 4B as a future upgrade.")
    start_note = node_by_id.get(116)
    start_copy = " ".join(str(value) for value in (start_note or {}).get("widgets_values", []))
    if "at least two of visual-recovery coverage" not in start_copy:
        raise WorkflowError("Workflow start note omits audio selector policy revision 4.")
    if (
        "official ACE-specific **4B language model**" not in start_copy
        or ACE_4B_LANGUAGE_MODEL not in start_copy
        or "future 4B upgrade" in start_copy
    ):
        raise WorkflowError("Workflow start note does not describe the active ACE 4B model.")
    if "juxtaposition is not musical fusion" not in start_copy:
        raise WorkflowError("Workflow start note omits the explicit genre-authority rule.")
    if (
        "**LTX performance mode** defaults to **Natural / audio-led sync**" not in start_copy
        or "**Dance / music sync**" not in start_copy
        or "**Lyrics + lip sync**" not in start_copy
        or "lyrics are not a timing schedule" not in start_copy
    ):
        raise WorkflowError("Workflow start note omits the LTX performance-mode control.")
    audit_titles = [
        str(node.get("title", ""))
        for node in workflow.get("nodes", [])
        if str(node.get("title", "")).startswith("FULL MUSIC QC AUDIT")
    ]
    if len(audit_titles) != 1 or "advisories" not in audit_titles[0]:
        raise WorkflowError("Full music QC preview must expose rejection reasons and advisories.")

    concept = exactly_one("DiffusionGemmaMusicProductionConcept")
    fanout = exactly_one("DiffusionGemmaSongSeedFanout")
    selector = exactly_one("DiffusionGemmaAudioCandidateSelector")
    guide = exactly_one("DiffusionGemmaLTXAudioGuide")
    reference_mode = exactly_one("DiffusionGemmaACEReferenceMode")
    planner = exactly_one("DiffusionGemmaSplatStagePlanner")
    router = exactly_one("SplatStageBlueprintRouter")
    cot = exactly_one("DiffusionGemmaCoTGenerator")
    gate = exactly_one("DiffusionGemmaBranchGenerationGate")
    performance = exactly_one(PERFORMANCE_NODE_TYPE)
    target_profile = exactly_one("DiffusionGemmaLTX25TargetProfile")
    if list(concept.get("widgets_values") or [])[:2] != ["Audition and select", 2]:
        raise WorkflowError("Migrated workflow must default to two-candidate Audition and select.")
    if input_origin(planner, "production_mode")[:2] != (concept, 0):
        raise WorkflowError("Planner production mode is not driven by the shared concept.")
    song_duration = node_by_id.get(562)
    if song_duration is None or input_origin(planner, "duration_seconds")[:2] != (
        song_duration,
        0,
    ):
        raise WorkflowError(
            "Planner duration must be driven by node 562's fixed ACE full-song clock."
        )
    if input_origin(router, "rhythm_mode")[:2] != (concept, 1):
        raise WorkflowError("Router production mode is not driven by the shared concept.")
    if input_origin(router, "candidate_count")[:2] != (concept, 2):
        raise WorkflowError("Router candidate count is not driven by the shared concept.")
    if input_origin(selector, "candidate_count")[:2] != (concept, 2):
        raise WorkflowError("Selector candidate count is not driven by the shared concept.")
    if input_origin(fanout, "production_concept")[:2] != (concept, 0):
        raise WorkflowError("Song seed fanout is not concept-aware.")

    authored_performance_inputs = [
        ("ltx_prompt", "STRING"),
        ("performance_mode", "COMBO"),
        ("song_duration_seconds", "FLOAT"),
        ("excerpt_start_seconds", "FLOAT"),
        ("excerpt_duration_seconds", "FLOAT"),
        ("lyrics", "STRING"),
    ]
    ui_resaved_performance_inputs = [
        ("ltx_prompt", "STRING"),
        ("lyrics", "STRING"),
        ("song_duration_seconds", "FLOAT"),
        ("excerpt_start_seconds", "FLOAT"),
        ("excerpt_duration_seconds", "FLOAT"),
    ]
    ui_resaved_performance_inputs_with_mode = [
        ("ltx_prompt", "STRING"),
        ("lyrics", "STRING"),
        ("performance_mode", "COMBO"),
        ("song_duration_seconds", "FLOAT"),
        ("excerpt_start_seconds", "FLOAT"),
        ("excerpt_duration_seconds", "FLOAT"),
    ]
    expected_performance_outputs = [
        ("ltx_prompt", "STRING"),
        ("selected_lyrics", "STRING"),
        ("status", "STRING"),
        ("performance_report_json", "STRING"),
    ]
    current_performance_outputs = [
        (str(item.get("name")), str(item.get("type")))
        for item in performance.get("outputs", [])
    ]
    accepted_performance_outputs = (
        expected_performance_outputs,
        [*expected_performance_outputs, ("performance_mode", "STRING")],
    )
    serialized_performance_inputs = [
        (str(item.get("name")), str(item.get("type")))
        for item in performance.get("inputs", [])
    ]
    if serialized_performance_inputs not in (
        authored_performance_inputs,
        ui_resaved_performance_inputs,
        ui_resaved_performance_inputs_with_mode,
    ):
        raise WorkflowError("LTX performance control inputs do not match its node contract.")
    if current_performance_outputs not in accepted_performance_outputs:
        raise WorkflowError("LTX performance control outputs do not match its node contract.")
    performance_values = list(performance.get("widgets_values") or [])
    if (
        len(performance_values) != len(PERFORMANCE_WIDGET_DEFAULTS)
        or performance_values[0] not in PERFORMANCE_MODES
    ):
        raise WorkflowError("LTX performance control has a malformed mode widget.")
    for input_name in (
        "song_duration_seconds",
        "excerpt_start_seconds",
        "excerpt_duration_seconds",
    ):
        if input_item(performance, input_name).get("widget", {}).get("name") != input_name:
            raise WorkflowError(
                f"LTX performance control input {input_name!r} lacks widget serialization."
            )
    if serialized_performance_inputs in (
        authored_performance_inputs,
        ui_resaved_performance_inputs_with_mode,
    ):
        if input_item(performance, "performance_mode").get("widget", {}).get("name") != "performance_mode":
            raise WorkflowError(
                "Authored LTX performance control lacks mode-widget serialization."
            )
    master_duration = node_by_id.get(178)
    if master_duration is None:
        raise WorkflowError("LTX performance control requires master duration node 178.")
    if input_origin(performance, "ltx_prompt")[:2] != (gate, 0):
        raise WorkflowError("LTX performance control must receive the validated gate prompt.")
    if input_origin(performance, "lyrics")[:2] != (router, 4):
        raise WorkflowError("LTX performance control must receive exact router lyrics.")
    if input_origin(performance, "song_duration_seconds")[:2] != (song_duration, 0):
        raise WorkflowError("LTX performance control has the wrong full-song clock.")
    if input_origin(performance, "excerpt_start_seconds")[:2] != (selector, 1):
        raise WorkflowError("LTX performance control must use the QC-selected excerpt start.")
    if input_origin(performance, "excerpt_duration_seconds")[:2] != (master_duration, 0):
        raise WorkflowError("LTX performance control must use the master video duration.")
    outer_for_prompt = next(
        (node for node in workflow["nodes"] if node.get("type") == LTX_SUBGRAPH_ID),
        None,
    )
    if outer_for_prompt is None or input_origin(outer_for_prompt, "value")[:2] != (
        performance,
        0,
    ):
        raise WorkflowError("Only the performance-controlled prompt may feed LTX.")
    lyric_previews = [
        node
        for node in workflow.get("nodes", [])
        if str(node.get("title", "")) == PERFORMANCE_PREVIEW_TITLE
    ]
    if len(lyric_previews) != 1 or input_origin(lyric_previews[0], "source")[:2] != (
        performance,
        1,
    ):
        raise WorkflowError("The selected LTX lyric window must remain visibly previewable.")
    profile_values = list(target_profile.get("widgets_values") or [])
    if len(profile_values) > 4 and profile_values[4] == LEGACY_FORCED_LIP_SYNC_GUIDANCE:
        raise WorkflowError("LTX target profile still forces lip sync ahead of the new toggle.")
    if len(profile_values) != 9 or profile_values[-1] not in CAMERA_CAPABILITIES:
        raise WorkflowError(
            "LTX target profile must persist one supported camera-capability widget."
        )
    if any(
        item.get("name") == CAMERA_CAPABILITY_INPUT
        for item in target_profile.get("inputs", [])
    ):
        raise WorkflowError(
            "Unlinked camera capability must be widget-only, not a synthetic graph socket."
        )

    def candidate_lane_nodes(
        nodes: Iterable[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        values = list(nodes)
        encoders = [
            node for node in values if node.get("type") == "TextEncodeAceStepAudio1.5"
        ]
        lane_samplers = [
            node
            for node in values
            if node.get("type") == "KSampler"
            and str(node.get("title", "")).startswith("ACE Candidate")
        ]
        lane_decoders = [
            node
            for node in values
            if node.get("type") == "VAEDecodeAudio"
            and str(node.get("title", "")).startswith("Decode ACE Candidate")
        ]
        lane_covers = [
            node
            for node in values
            if node.get("type") == "DiffusionGemmaACECoverConditioning"
        ]
        for lane in (encoders, lane_samplers, lane_decoders, lane_covers):
            lane.sort(key=lambda node: str(node.get("title", "")))
        return encoders, lane_samplers, lane_decoders, lane_covers

    def validate_candidate_widgets(
        encoders: list[dict[str, Any]], lane_samplers: list[dict[str, Any]]
    ) -> None:
        if len(encoders) != 4 or len(lane_samplers) != 4:
            raise WorkflowError("Production graph must contain four complete ACE candidate lanes.")
        for index, (encode, sampler) in enumerate(
            zip(encoders, lane_samplers), start=1
        ):
            encode_values = list(encode.get("widgets_values") or [])
            sampler_values = list(sampler.get("widgets_values") or [])
            if "ACE 4B LM" not in str(encode.get("title", "")):
                raise WorkflowError(
                    f"ACE candidate {index} title does not identify the active 4B LM."
                )
            if len(encode_values) < 15 or float(encode_values[11]) != 0.72:
                raise WorkflowError(
                    f"ACE candidate {index} does not use LM temperature 0.72."
                )
            if (
                len(sampler_values) < 7
                or int(sampler_values[2]) != 8
                or float(sampler_values[3]) != 1.0
            ):
                raise WorkflowError(
                    f"ACE candidate {index} does not use 8-step CFG-1 Turbo sampling."
                )

    text_encoders, samplers, decoders, covers = candidate_lane_nodes(
        workflow.get("nodes", [])
    )
    flat_lane_counts = tuple(
        len(values) for values in (text_encoders, samplers, decoders, covers)
    )
    if flat_lane_counts == (4, 4, 4, 4):
        validate_candidate_widgets(text_encoders, samplers)
        for index, (encode, sampler, decoder, cover) in enumerate(
            zip(text_encoders, samplers, decoders, covers), start=1
        ):
            if input_origin(encode, "seed")[:2] != (fanout, index - 1):
                raise WorkflowError(f"ACE candidate {index} TextEncode seed provenance is wrong.")
            if input_origin(sampler, "seed")[:2] != (fanout, index - 1):
                raise WorkflowError(f"ACE candidate {index} KSampler seed provenance is wrong.")
            if input_origin(encode, "generate_audio_codes")[:2] != (reference_mode, 0):
                raise WorkflowError(f"ACE candidate {index} is not governed by Compose/Cover mode.")
            if input_origin(cover, "mode_token")[:2] != (reference_mode, 1):
                raise WorkflowError(f"ACE candidate {index} Cover gate has the wrong mode source.")
            if input_origin(selector, f"candidate_{index}")[:2] != (decoder, 0):
                raise WorkflowError(f"Selector candidate {index} is not fed by its decoded lane.")
            decoder_targets = [
                node_by_id[int(links_by_id[int(link_id)][3])]
                for link_id in decoder["outputs"][0].get("links") or []
            ]
            if decoder_targets != [selector]:
                raise WorkflowError(
                    f"ACE candidate {index} has a forced Preview/Save output and is no longer lazy."
                )
    elif flat_lane_counts == (0, 0, 0, 0):
        definitions = {
            str(item.get("id")): item
            for item in workflow.get("definitions", {}).get("subgraphs", [])
        }
        candidate_banks: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for wrapper in workflow.get("nodes", []):
            definition = definitions.get(str(wrapper.get("type")))
            if definition is None:
                continue
            inner_lanes = candidate_lane_nodes(definition.get("nodes", []))
            if tuple(len(values) for values in inner_lanes) == (4, 4, 4, 4):
                candidate_banks.append((wrapper, definition))
        if len(candidate_banks) != 1:
            raise WorkflowError(
                "Production graph must contain one four-lane ACE candidate bank."
            )
        bank, bank_definition = candidate_banks[0]
        text_encoders, samplers, decoders, covers = candidate_lane_nodes(
            bank_definition.get("nodes", [])
        )
        validate_candidate_widgets(text_encoders, samplers)

        seed_inputs = ("seed", "seed_1", "seed_2", "seed_3")
        for index, seed_name in enumerate(seed_inputs, start=1):
            if input_origin(bank, seed_name)[:2] != (fanout, index - 1):
                raise WorkflowError(
                    f"Grouped ACE candidate {index} seed provenance is wrong."
                )
            if input_origin(selector, f"candidate_{index}")[:2] != (bank, index - 1):
                raise WorkflowError(
                    f"Selector candidate {index} is not fed by the grouped ACE bank."
                )
        if input_origin(bank, "generate_audio_codes")[:2] != (reference_mode, 0):
            raise WorkflowError("Grouped ACE bank is not governed by Compose/Cover mode.")
        if input_origin(bank, "mode_token")[:2] != (reference_mode, 1):
            raise WorkflowError("Grouped ACE bank Cover gates have the wrong mode source.")

        inner_links = {
            int(link["id"]): link for link in bank_definition.get("links", [])
        }
        bank_input_slots = {
            str(item.get("name")): index
            for index, item in enumerate(bank.get("inputs", []))
        }

        def inner_origin(node: dict[str, Any], input_name: str) -> tuple[int, int]:
            matches = [
                item for item in node.get("inputs", []) if item.get("name") == input_name
            ]
            if len(matches) != 1 or matches[0].get("link") is None:
                raise WorkflowError(
                    f"Grouped node {node.get('id')} requires connected input {input_name!r}."
                )
            link = inner_links[int(matches[0]["link"])]
            return int(link["origin_id"]), int(link["origin_slot"])

        for index, (encode, sampler, decoder, cover) in enumerate(
            zip(text_encoders, samplers, decoders, covers), start=1
        ):
            seed_slot = bank_input_slots[seed_inputs[index - 1]]
            if inner_origin(encode, "seed") != (-10, seed_slot):
                raise WorkflowError(
                    f"Grouped ACE candidate {index} TextEncode seed route is wrong."
                )
            if inner_origin(sampler, "seed") != (-10, seed_slot):
                raise WorkflowError(
                    f"Grouped ACE candidate {index} KSampler seed route is wrong."
                )
            if inner_origin(encode, "generate_audio_codes") != (
                -10,
                bank_input_slots["generate_audio_codes"],
            ):
                raise WorkflowError(
                    f"Grouped ACE candidate {index} Compose/Cover route is wrong."
                )
            if inner_origin(cover, "mode_token") != (
                -10,
                bank_input_slots["mode_token"],
            ):
                raise WorkflowError(
                    f"Grouped ACE candidate {index} Cover mode route is wrong."
                )
            decoder_links = decoder.get("outputs", [])[0].get("links") or []
            decoder_targets = [
                int(inner_links[int(link_id)]["target_id"])
                for link_id in decoder_links
            ]
            if decoder_targets != [-20]:
                raise WorkflowError(
                    f"Grouped ACE candidate {index} has a forced internal Preview/Save output."
                )
    else:
        raise WorkflowError(
            "Production graph contains a partial mixture of grouped and expanded ACE lanes."
        )

    clip_loader = exactly_one("DualCLIPLoader")
    if list(clip_loader.get("widgets_values") or [None, None])[1] != ACE_4B_LANGUAGE_MODEL:
        raise WorkflowError(
            f"ACE loader is not using the official ACE-specific {ACE_4B_LANGUAGE_MODEL}."
        )
    loader_assets = clip_loader.get("properties", {}).get("models")
    if not isinstance(loader_assets, list) or len(loader_assets) < 2:
        raise WorkflowError("ACE loader model-download metadata is missing or malformed.")
    second_asset = loader_assets[1]
    expected_asset = {
        "name": ACE_4B_LANGUAGE_MODEL,
        "url": ACE_4B_LANGUAGE_MODEL_URL,
        "directory": "text_encoders",
    }
    if not isinstance(second_asset, dict) or any(
        second_asset.get(key) != expected for key, expected in expected_asset.items()
    ):
        raise WorkflowError("ACE loader's second model-download asset is not the official 4B LM.")
    aura_nodes = [
        node
        for node in _all_workflow_nodes(workflow)
        if node.get("type") == "ModelSamplingAuraFlow"
    ]
    if len(aura_nodes) != 1:
        raise WorkflowError(
            "Production graph requires one ACE AuraFlow sampler across all scopes; "
            f"found {len(aura_nodes)}."
        )
    aura = aura_nodes[0]
    aura_value = aura.get("widgets_values")
    if not (
        isinstance(aura_value, list)
        and len(aura_value) == 1
        and float(aura_value[0]) == 3.0
    ):
        raise WorkflowError("ACE AuraFlow shift must be serialized as the widget list [3.0].")

    report_origin, report_slot, _ = input_origin(cot, "measured_audio_report_json")
    if report_origin.get("type") != "easy cleanGpuUsed":
        raise WorkflowError("Director measurements must pass through the post-ACE cleanup node.")
    if input_origin(report_origin, "anything")[:2] != (selector, 3):
        raise WorkflowError("Director must receive selector output 3 (compact report), not the full audit.")
    full_audit_targets = [
        node_by_id[int(links_by_id[int(link_id)][3])]
        for link_id in selector["outputs"][4].get("links") or []
    ]
    if not any(node.get("type") == "PreviewAny" for node in full_audit_targets):
        raise WorkflowError("Full selector audit must remain visibly previewable.")

    outer = next(
        (node for node in workflow["nodes"] if node.get("type") == LTX_SUBGRAPH_ID),
        None,
    )
    if outer is None:
        raise WorkflowError("Production graph is missing the embedded LTX group.")
    if input_origin(outer, "conditioning_audio")[:2] != (guide, 0):
        raise WorkflowError("LTX conditioning slot is not fed by the processed guide.")
    if input_origin(outer, "final_soundtrack")[:2] != (guide, 1):
        raise WorkflowError("LTX final mux slot is not fed by the pristine soundtrack.")

    subgraph = _subgraph_by_id(workflow, LTX_SUBGRAPH_ID)
    sg_nodes = {int(node["id"]): node for node in subgraph.get("nodes", [])}
    if any(node.get("type") == "Film Grain" for node in sg_nodes.values()):
        raise WorkflowError("Embedded LTX subgraph still contains the unavailable Film Grain test node.")
    if any(
        str(group.get("title", "")).strip().casefold() == "lut test (disabled)"
        for group in subgraph.get("groups", [])
    ):
        raise WorkflowError("Embedded LTX subgraph still contains the stale disabled LUT test group.")
    sg_links = {int(link["id"]): link for link in subgraph.get("links", [])}
    if len(sg_links) != len(subgraph.get("links", [])):
        raise WorkflowError("Embedded LTX subgraph has duplicate link IDs.")
    interface_links = {
        int(link_id)
        for item in subgraph.get("inputs", [])
        for link_id in item.get("linkIds", [])
    }
    output_links = {
        int(link_id)
        for item in subgraph.get("outputs", []) if isinstance(item, dict)
        for link_id in item.get("linkIds", [])
    }
    for link_id, link in sg_links.items():
        origin_id = int(link["origin_id"])
        target_id = int(link["target_id"])
        origin_slot = int(link["origin_slot"])
        target_slot = int(link["target_slot"])
        if origin_id == -10:
            if link_id not in interface_links:
                raise WorkflowError(f"Subgraph input link {link_id} lacks its interface backlink.")
        else:
            if origin_id not in sg_nodes or origin_slot >= len(sg_nodes[origin_id].get("outputs", [])):
                raise WorkflowError(f"Subgraph link {link_id} has an invalid origin.")
            if link_id not in (sg_nodes[origin_id]["outputs"][origin_slot].get("links") or []):
                raise WorkflowError(f"Subgraph link {link_id} lacks its origin backlink.")
        if target_id == -20:
            if link_id not in output_links:
                raise WorkflowError(f"Subgraph output link {link_id} lacks its interface backlink.")
        else:
            if target_id not in sg_nodes or target_slot >= len(sg_nodes[target_id].get("inputs", [])):
                raise WorkflowError(f"Subgraph link {link_id} has an invalid target.")
            if sg_nodes[target_id]["inputs"][target_slot].get("link") != link_id:
                raise WorkflowError(f"Subgraph link {link_id} lacks its target backlink.")

    sg_inputs = subgraph.get("inputs", [])
    if len(sg_inputs) != 18:
        raise WorkflowError("Embedded LTX interface must expose conditioning, final, and A2V slots.")
    if sg_inputs[15].get("name") != "conditioning_audio" or sg_inputs[15].get("linkIds") != [985]:
        raise WorkflowError("Embedded LTX conditioning audio must route only to link 985.")
    if sg_inputs[16].get("name") != "final_soundtrack" or sg_inputs[16].get("linkIds") != [967]:
        raise WorkflowError("Embedded LTX final soundtrack must route only to link 967.")
    if _sg_link(subgraph, 985).get("target_id") != 586 or _sg_link(subgraph, 985).get("origin_slot") != 15:
        raise WorkflowError("Conditioning audio no longer feeds only LTX AudioVAEEncode.")
    if _sg_link(subgraph, 967).get("target_id") != 453 or _sg_link(subgraph, 967).get("origin_slot") != 16:
        raise WorkflowError("Pristine soundtrack no longer feeds only CreateVideo.")
    mask = sg_nodes.get(588)
    if mask is None or set(mask["outputs"][0].get("links") or []) != {666, 966}:
        raise WorkflowError("Frozen audio mask must continue feeding both LTX sampling stages.")
    if any(node.get("type") == "LTXVDualCFGGuider" for node in sg_nodes.values()):
        raise WorkflowError("Misleading LTX dual-CFG nodes remain in the embedded workflow.")
    multimodal = [node for node in sg_nodes.values() if node.get("type") == "MultimodalGuider"]
    parameters = [node for node in sg_nodes.values() if node.get("type") == "GuiderParameters"]
    if len(multimodal) != 2 or len(parameters) != 4:
        raise WorkflowError("Both LTX stages require AUDIO+VIDEO GuiderParameters and MultimodalGuider.")
    video_params = [node for node in parameters if list(node.get("widgets_values") or [None])[0] == "VIDEO"]
    audio_params = [node for node in parameters if list(node.get("widgets_values") or [None])[0] == "AUDIO"]
    if len(video_params) != 2 or len(audio_params) != 2:
        raise WorkflowError("A2V topology requires two AUDIO and two VIDEO parameter nodes.")
    a2v_links = set(int(value) for value in sg_inputs[17].get("linkIds", []))
    expected_a2v_links: set[int] = set()
    for node in video_params:
        modality_scale = [
            item for item in node.get("inputs", []) if item.get("name") == "modality_scale"
        ]
        if len(modality_scale) != 1 or modality_scale[0].get("link") is None:
            raise WorkflowError("VIDEO GuiderParameters requires one linked modality_scale input.")
        expected_a2v_links.add(int(modality_scale[0]["link"]))
    if len(a2v_links) != 2 or a2v_links != expected_a2v_links:
        raise WorkflowError("One outer A2V control must drive VIDEO modality_scale in both stages.")


def _atomic_json_write(path: Path, workflow: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
    payload = json.dumps(workflow, ensure_ascii=False, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=list(DEFAULT_WORKFLOWS))
    parser.add_argument("--check", action="store_true", help="Validate without writing.")
    args = parser.parse_args(argv)
    for path in args.paths:
        workflow = _load(path)
        # Always pass through the idempotent migrator.  This also applies
        # narrowly scoped hygiene repairs added after the schema version was
        # first written, without rewriting already-current production nodes.
        migrated = migrate_workflow(workflow)
        validate_workflow(migrated)
        sensitive = _sensitive_key_paths(migrated)
        if sensitive:
            raise WorkflowError(
                f"Refusing to write {path}: non-empty sensitive-looking fields at {sensitive}."
            )
        if not args.check:
            _atomic_json_write(path, migrated)
        digest = hashlib.sha256(
            json.dumps(migrated, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        print(f"{'checked' if args.check else 'migrated'} {path} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
