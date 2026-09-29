# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

"""Versioned advertisement planning and persistent H3 relay contracts.

These nodes intentionally live beside the established music-video production
nodes.  They import generic parsing/timing helpers but do not change any legacy
node class, return tuple, or saved-workflow contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

import torch
import torch.nn.functional as torch_functional

try:
    from .production_planning_nodes import (
        H3_FPS,
        MAX_H3_SHOTS,
        MAX_PROJECT_SECONDS,
        _carry_in_source_block,
        _h3_frames,
        _leading_shot_timestamp_seconds,
        _render_sections,
        _section_map,
        _segment_prompt,
        _shot_blocks,
        _validate_source_timestamps,
    )
except ImportError:  # Standalone repository tests.
    from production_planning_nodes import (
        H3_FPS,
        MAX_H3_SHOTS,
        MAX_PROJECT_SECONDS,
        _carry_in_source_block,
        _h3_frames,
        _leading_shot_timestamp_seconds,
        _render_sections,
        _section_map,
        _segment_prompt,
        _shot_blocks,
        _validate_source_timestamps,
    )


CATEGORY = "prompt/diffusiongemma/advertising"
ADVERTISEMENT_MASTER_SCHEMA = "diffusiongemma.advertisement_master_contract"
ADVERTISEMENT_PLAN_SCHEMA = "diffusiongemma.advertisement_h3_plan"
ADVERTISEMENT_RELAY_SCHEMA = "diffusiongemma.advertisement_relay_artifact"
VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SLUG_RE = re.compile(r"^[a-z0-9]+(?:[a-z0-9_-]*[a-z0-9])?$")
PRODUCT_PRESENCE = ("absent", "background", "supporting", "hero", "consumed")
ADVERTISEMENT_PERFORMANCE_CARRIER_MODES = (
    "Dance / music sync",
    "Lyrics + lip sync",
    "Natural / audio-led sync",
)
BOUNDARY_MODES = (
    "hard_cut",
    "match_state",
    "relay_continuity",
    "practical_transition",
)
EXPORTED_BOUNDARY_MODES = ("relay_continuity",)
EXPORTED_SEAM_STYLES = ("Campaign default",)
STATE_FIELDS = (
    "composition",
    "performer_state",
    "product_state",
    "environment",
    "transition_token",
)


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_object(value: Any, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def _parse_array(value: Any, label: str) -> list[Any]:
    try:
        parsed = json.loads(str(value or ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON.") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{label} must be a JSON array.")
    return parsed


def _positive(value: Any, label: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(number) or number < 0 or (not allow_zero and number <= 0):
        raise ValueError(f"{label} must be a positive finite number.")
    return number


def _hash(value: Any, label: str) -> str:
    digest = str(value or "").strip().casefold()
    if not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} must be an exact 64-character SHA-256 value.")
    return digest


def _required_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must not be empty.")
    return text


def _contract(value: Any) -> dict[str, Any]:
    contract = _parse_object(value, "advertisement_contract_json")
    if contract.get("schema") != ADVERTISEMENT_MASTER_SCHEMA or contract.get("version") != VERSION:
        raise ValueError("advertisement_contract_json has an unsupported schema or version.")
    if not contract.get("campaign_id"):
        raise ValueError("advertisement_contract_json has no campaign_id.")
    supplied_contract_hash = _hash(contract.get("contract_sha256"), "advertisement contract_sha256")
    contract_identity = dict(contract)
    contract_identity.pop("contract_sha256", None)
    if _sha_text(_json(contract_identity)) != supplied_contract_hash:
        raise ValueError("advertisement_contract_json contract_sha256 does not match its canonical content.")
    supplied_campaign_id = _hash(contract.get("campaign_id"), "advertisement campaign_id")
    campaign_identity = dict(contract_identity)
    campaign_identity.pop("campaign_id", None)
    if _sha_text(_json(campaign_identity)) != supplied_campaign_id:
        raise ValueError("advertisement_contract_json campaign_id does not match its canonical identity.")
    return contract


def _plan(value: Any) -> dict[str, Any]:
    plan = _parse_object(value, "advertisement_plan_json")
    if plan.get("schema") != ADVERTISEMENT_PLAN_SCHEMA or plan.get("version") != VERSION:
        raise ValueError("advertisement_plan_json has an unsupported schema or version.")
    if not plan.get("ready") or not isinstance(plan.get("lanes"), list):
        raise ValueError("advertisement_plan_json is not ready.")
    supplied_hash = _hash(plan.get("plan_sha256"), "advertisement plan_sha256")
    identity = dict(plan)
    identity.pop("plan_sha256", None)
    if _sha_text(_json(identity)) != supplied_hash:
        raise ValueError("advertisement_plan_json plan_sha256 does not match its canonical content.")
    return plan


def _legacy_project(value: Any) -> dict[str, Any]:
    project = _parse_object(value, "project_manifest_json")
    if project.get("schema") != "diffusiongemma.project_master_contract":
        raise ValueError("project_manifest_json must come from Project Master Contract.")
    if not project.get("project_id"):
        raise ValueError("project_manifest_json has no project_id.")
    return project


class AdvertisementMasterContract:
    """Compile the typed ad contracts and legacy timing lock into one master."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "campaign_contract_json": ("STRING", {"forceInput": True, "multiline": True}),
                "reference_contract_json": ("STRING", {"forceInput": True, "multiline": True}),
                "soundtrack_contract_json": ("STRING", {"forceInput": True, "multiline": True}),
                "project_manifest_json": ("STRING", {"forceInput": True, "multiline": True}),
                "performer_hero_sha256": ("STRING", {"default": ""}),
                "performer_contact_sheet_sha256": ("STRING", {"default": ""}),
                "product_package_sha256": ("STRING", {"default": ""}),
                "master_audio_sha256": ("STRING", {"default": "", "forceInput": True}),
                "native_shot_count": ("INT", {"default": 8, "min": 1, "max": 99}),
                "boundary_mode": (
                    list(EXPORTED_BOUNDARY_MODES),
                    {"default": EXPORTED_BOUNDARY_MODES[0]},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN", "INT", "STRING")
    RETURN_NAMES = (
        "advertisement_contract_json",
        "campaign_id",
        "status",
        "ready",
        "generation_lane_count",
        "reference_manifest",
    )
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Bridges Campaign, Reference, and Soundtrack contracts to a validated Project Master, three immutable "
        "clean-reference hashes, independent performer/product retention, and deterministic delivery intent."
    )

    def build(
        self,
        campaign_contract_json: str,
        reference_contract_json: str,
        soundtrack_contract_json: str,
        project_manifest_json: str,
        performer_hero_sha256: str,
        performer_contact_sheet_sha256: str,
        product_package_sha256: str,
        master_audio_sha256: str,
        native_shot_count: int,
        boundary_mode: str,
    ):
        campaign = _parse_object(campaign_contract_json, "campaign_contract_json")
        references = _parse_object(reference_contract_json, "reference_contract_json")
        soundtrack = _parse_object(soundtrack_contract_json, "soundtrack_contract_json")
        if campaign.get("schema") != "diffusiongemma.advertisement_campaign_contract" or campaign.get("version") != VERSION or not campaign.get("ready"):
            raise ValueError("campaign_contract_json is not a ready supported advertisement campaign contract.")
        if references.get("schema") != "diffusiongemma.advertisement_reference_contract" or references.get("version") != VERSION or not references.get("ready"):
            raise ValueError("reference_contract_json is not a ready supported advertisement reference contract.")
        if soundtrack.get("schema") != "diffusiongemma.advertisement_soundtrack_contract" or soundtrack.get("version") != VERSION or not soundtrack.get("ready"):
            raise ValueError("soundtrack_contract_json is not a ready supported advertisement soundtrack contract.")
        project = _legacy_project(project_manifest_json)
        brand_block = campaign.get("brand")
        timeline = campaign.get("timeline")
        mandatory_copy = campaign.get("copy")
        if not isinstance(brand_block, Mapping) or not isinstance(timeline, Mapping) or not isinstance(mandatory_copy, Mapping):
            raise ValueError("Advertisement campaign contract is missing brand, timeline, or exact-copy fields.")
        brand = _required_text(brand_block.get("brand_name"), "campaign brand_name")
        product = _required_text(brand_block.get("product_name"), "campaign product_name")
        objective = _required_text(campaign.get("objective"), "campaign objective")
        audience_text = _required_text(campaign.get("audience"), "campaign audience")
        message_text = _required_text(mandatory_copy.get("headline"), "campaign headline")
        cta = str(mandatory_copy.get("call_to_action", "")).strip()
        slug_source = str(campaign.get("campaign_name") or f"{brand}-{product}").casefold()
        slug = re.sub(r"[^a-z0-9_-]+", "-", slug_source).strip("-_") or f"campaign-{_sha_text(slug_source)[:12]}"
        if not SLUG_RE.fullmatch(slug):
            raise ValueError("Derived advertisement campaign_slug is invalid.")
        slots = references.get("picture_slots")
        if not isinstance(slots, list):
            raise ValueError("Advertisement reference contract has no picture_slots.")
        role_by_tag = {
            str(item.get("picture_tag", "")): str(item.get("role", ""))
            for item in slots
            if isinstance(item, Mapping)
        }
        expected_roles = {
            "<Picture 1>": "performer_primary",
            "<Picture 2>": "performer_supplemental",
            "<Picture 3>": "product_package",
        }
        if any(role_by_tag.get(tag) != role for tag, role in expected_roles.items()):
            raise ValueError(
                "Advertisement H3 planning requires the Reference Contract policy with performer hero Picture 1, "
                "performer contact sheet Picture 2, and product/package Picture 3."
            )
        relay = references.get("relay")
        if (
            references.get("slot_policy") != "performer_2_product_1_relay_1"
            or not isinstance(relay, Mapping)
            or relay.get("enabled") is not True
            or relay.get("picture_tag") != "<Picture 4>"
        ):
            raise ValueError(
                "Advertisement H3 planning requires the shipped two-performer, one-product, Picture 4 relay policy."
            )
        subjects = references.get("subjects")
        if not isinstance(subjects, list) or len(subjects) != 2:
            raise ValueError("Advertisement reference contract must contain exactly performer Subject 1 and product Subject 2.")
        reference_hashes = {
            "performer_hero": _hash(performer_hero_sha256, "performer_hero_sha256"),
            "performer_contact_sheet": _hash(
                performer_contact_sheet_sha256, "performer_contact_sheet_sha256"
            ),
            "product_package": _hash(product_package_sha256, "product_package_sha256"),
        }
        if len(set(reference_hashes.values())) != 3:
            raise ValueError("The performer hero, performer contact sheet, and product package must be three distinct assets.")
        audio_hash = _hash(master_audio_sha256, "master_audio_sha256")
        project_audio = project.get("audio_lock")
        if not isinstance(project_audio, Mapping):
            raise ValueError("Project Master has no audio_lock.")
        if str(project_audio.get("waveform_sha256", "")).casefold() != audio_hash:
            raise ValueError("Advertisement and Project Master audio SHA-256 values disagree.")
        duration = _positive(project.get("production_duration_seconds"), "production_duration_seconds")
        campaign_duration = _positive(timeline.get("duration_seconds"), "campaign timeline duration_seconds")
        if abs(campaign_duration - duration) > 1.0e-6:
            raise ValueError("Campaign Contract and Project Master durations disagree.")
        campaign_aspect = str(timeline.get("aspect_ratio", ""))
        if campaign_aspect != str(project.get("master_aspect_ratio", "")):
            raise ValueError("Campaign Contract and Project Master aspect ratios disagree.")
        if duration > MAX_PROJECT_SECONDS:
            raise ValueError(f"Advertisement H3 planning supports at most {MAX_PROJECT_SECONDS:g} seconds.")
        segmentation = project.get("h3_segmentation")
        if not isinstance(segmentation, Mapping):
            raise ValueError("Project Master has no h3_segmentation contract.")
        lane_ceiling = _positive(segmentation.get("maximum_shot_seconds"), "maximum_shot_seconds")
        if lane_ceiling > 15.0:
            raise ValueError("One H3 advertisement generation lane cannot exceed 15 seconds.")
        lane_count = int(math.ceil(duration / lane_ceiling))
        if not 1 <= lane_count <= MAX_H3_SHOTS:
            raise ValueError("Advertisement duration requires more than four H3 generation lanes.")
        shots = int(native_shot_count)
        if not 1 <= shots <= 99:
            raise ValueError("native_shot_count must be from 1 to 99.")
        mode = str(boundary_mode or "")
        if mode not in BOUNDARY_MODES:
            raise ValueError("boundary_mode is unsupported.")
        end_card = timeline.get("end_card")
        if not isinstance(end_card, Mapping):
            raise ValueError("Campaign Contract has no end-card timing.")
        end_seconds = _positive(end_card.get("duration_seconds"), "end_card duration_seconds", allow_zero=True)
        if end_seconds > duration + 1.0e-6:
            raise ValueError("Campaign end-card duration cannot exceed the advertisement duration.")
        claims = campaign.get("claims", [])
        if not isinstance(claims, list):
            raise ValueError("Campaign Contract claims must be a JSON array.")
        source_reference_manifest = _required_text(references.get("reference_manifest"), "reference_manifest")
        reference_rows = [source_reference_manifest]
        if "<Picture 4>" not in source_reference_manifest:
            reference_rows.append(
                "<Picture 4>: [dg:spatial,lighting,motion,temporal] optional prior-lane retained tail for opening continuity only; never performer or product identity authority."
            )
        reference_rows.append("<Audio 1>: [dg:audio,rhythm] exact locked soundtrack and timing authority.")
        reference_manifest = "\n".join(reference_rows)
        contract: dict[str, Any] = {
            "schema": ADVERTISEMENT_MASTER_SCHEMA,
            "version": VERSION,
            "parent_project_id": str(project["project_id"]),
            "parent_project_sha256": _sha_text(_json(project)),
            "upstream_contracts": {
                "campaign_contract_sha256": _sha_text(_json(campaign)),
                "reference_contract_sha256": _sha_text(_json(references)),
                "soundtrack_contract_sha256": _sha_text(_json(soundtrack)),
            },
            "campaign_slug": slug,
            "brand_name": brand,
            "product_name": product,
            "campaign_objective": objective,
            "audience": audience_text,
            "message": message_text,
            "call_to_action": cta,
            "production_duration_seconds": round(duration, 6),
            "fps": H3_FPS,
            "master_aspect_ratio": str(project.get("master_aspect_ratio", "")),
            "audio_lock": {
                "waveform_sha256": audio_hash,
                "excerpt_start_seconds": float(project_audio.get("excerpt_start_seconds", 0.0)),
                "excerpt_duration_seconds": float(project_audio.get("excerpt_duration_seconds", duration)),
                "soundtrack_content_mode": str(soundtrack.get("effective_content_mode", "")),
            },
            "reference_assets": {
                "picture_1_performer_hero_sha256": reference_hashes["performer_hero"],
                "picture_2_performer_contact_sheet_sha256": reference_hashes[
                    "performer_contact_sheet"
                ],
                "picture_3_product_package_sha256": reference_hashes["product_package"],
                "picture_4_role": "optional_previous_lane_tail_continuity_only",
            },
            "reference_manifest": reference_manifest,
            "subject_retention": {
                "<Subject 1>": {
                    "role": "performer",
                    "mode": "fully_preserved",
                    "authority": ["<Picture 1>", "<Picture 2>"],
                },
                "<Subject 2>": {
                    "role": "product_package",
                    "mode": "fully_preserved",
                    "authority": ["<Picture 3>"],
                },
            },
            "h3_planning": {
                "native_shot_count": shots,
                "generation_lane_count": lane_count,
                "maximum_lane_seconds": round(lane_ceiling, 6),
                "boundary_mode": mode,
                "maximum_generation_lanes": MAX_H3_SHOTS,
            },
            "end_card": {
                "duration_seconds": round(end_seconds, 6),
                "frame_count": int(round(end_seconds * H3_FPS)),
                "mode": "replace_master_tail",
            },
            "mandatory_copy": mandatory_copy,
            "claims": claims,
            "requested_deliverables": campaign.get("deliverables", {}),
        }
        campaign_id = _sha_text(_json(contract))
        contract["campaign_id"] = campaign_id
        contract["contract_sha256"] = _sha_text(_json(contract))
        status = (
            f"Advertisement {slug} locked to {duration:g}s, {shots} native shots, "
            f"{lane_count} H3 generation lane{'s' if lane_count != 1 else ''}, three clean "
            "hash-addressed references, and independent performer/product retention."
        )
        return _json(contract), campaign_id, status, True, lane_count, reference_manifest


def _state(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    result: dict[str, str] = {}
    for field in STATE_FIELDS:
        text = str(value.get(field, "")).strip()
        if field != "transition_token" and not text:
            raise ValueError(f"{label}.{field} must not be empty.")
        result[field] = text
    return result


def _lane_states(raw: list[Any], count: int, default_mode: str) -> list[dict[str, Any]]:
    if len(raw) != count:
        raise ValueError(f"lane_states_json must contain exactly {count} lane entries.")
    states: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping) or int(item.get("lane_index", 0)) != index:
            raise ValueError("lane_states_json entries must be consecutively indexed from 1.")
        boundary = str(item.get("boundary_mode", "project_start" if index == 1 else default_mode))
        if index == 1:
            if boundary != "project_start":
                raise ValueError("Lane 1 boundary_mode must be project_start.")
        elif boundary not in BOUNDARY_MODES:
            raise ValueError(f"Lane {index} boundary_mode is unsupported.")
        states.append(
            {
                "lane_index": index,
                "boundary_mode": boundary,
                "entry_state": _state(item.get("entry_state"), f"lane {index} entry_state"),
                "exit_state": _state(item.get("exit_state"), f"lane {index} exit_state"),
            }
        )
    for index in range(1, count):
        previous = states[index - 1]["exit_state"]
        following = states[index]["entry_state"]
        mode = states[index]["boundary_mode"]
        if mode in {"match_state", "relay_continuity"} and previous != following:
            raise ValueError(
                f"Advertisement lane seam mismatch between lanes {index} and {index + 1}: "
                f"{mode} requires identical typed exit/entry state."
            )
        if mode == "practical_transition":
            if not previous["transition_token"] or previous["transition_token"] != following["transition_token"]:
                raise ValueError(
                    f"Advertisement lane seam mismatch between lanes {index} and {index + 1}: "
                    "practical_transition requires one matching non-empty transition_token."
                )
    return states


def _shot_metadata(raw: list[Any], count: int) -> list[dict[str, Any]]:
    if len(raw) != count:
        raise ValueError(f"shot_metadata_json must contain exactly {count} native-shot entries.")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping) or int(item.get("shot_index", 0)) != index:
            raise ValueError("shot_metadata_json entries must be consecutively indexed from 1.")
        presence = str(item.get("product_presence", ""))
        if presence not in PRODUCT_PRESENCE:
            raise ValueError(f"Shot {index} product_presence must be one of {', '.join(PRODUCT_PRESENCE)}.")
        result.append(
            {
                "shot_index": index,
                "product_presence": presence,
                "product_action": str(item.get("product_action", "")).strip(),
                "mandatory_copy_visible": bool(item.get("mandatory_copy_visible", False)),
            }
        )
    if not any(item["product_presence"] == "hero" for item in result):
        raise ValueError("At least one native shot must declare hero product presence.")
    return result


class AdvertisementPlanningDefaults:
    """Emit valid, deterministic non-technical defaults for advertisement planning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "advertisement_contract_json": ("STRING", {"forceInput": True, "multiline": True}),
                "product_use": (
                    "STRING",
                    {
                        "default": "The performer naturally carries, opens, presents, and enjoys the product while keeping its package recognizable.",
                        "multiline": True,
                    },
                ),
                "seam_style": (
                    list(EXPORTED_SEAM_STYLES),
                    {"default": EXPORTED_SEAM_STYLES[0]},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("shot_metadata_json", "lane_states_json", "planning_brief", "status", "ready")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Creates production-safe product-presence and typed lane-seam defaults, so an advertisement can "
        "run without asking users to hand-author JSON. Advanced users may override either JSON output."
    )

    def build(self, advertisement_contract_json: str, product_use: str, seam_style: str):
        contract = _contract(advertisement_contract_json)
        planning = contract.get("h3_planning", {})
        shot_count = int(planning.get("native_shot_count", 0))
        lane_count = int(planning.get("generation_lane_count", 0))
        if shot_count < 1 or not 1 <= lane_count <= MAX_H3_SHOTS:
            raise ValueError("Advertisement contract has invalid native-shot or generation-lane counts.")
        action = _required_text(product_use, "product_use")
        hero_indices = {max(1, (shot_count + 1) // 2), shot_count}
        shots: list[dict[str, Any]] = []
        for index in range(1, shot_count + 1):
            presence = "hero" if index in hero_indices else "supporting"
            shots.append(
                {
                    "shot_index": index,
                    "product_presence": presence,
                    "product_action": action,
                    "mandatory_copy_visible": False,
                }
            )
        mode_names = {
            "Campaign default": str(planning.get("boundary_mode", "hard_cut")),
            "Hard cut": "hard_cut",
            "Match state": "match_state",
            "Relay continuity": "relay_continuity",
            "Practical transition": "practical_transition",
        }
        if seam_style not in mode_names:
            raise ValueError("seam_style is unsupported.")
        boundary = mode_names[seam_style]
        if boundary not in BOUNDARY_MODES:
            raise ValueError("Advertisement contract has an unsupported default boundary mode.")

        def state(lane: int, phase: str, token: str = "") -> dict[str, str]:
            product_name = str(contract.get("product_name", "product"))
            return {
                "composition": f"lane {lane} {phase} campaign composition",
                "performer_state": f"performer pose and facing at lane {lane} {phase}",
                "product_state": f"{product_name} package held visibly at lane {lane} {phase}",
                "environment": f"campaign environment state at lane {lane} {phase}",
                "transition_token": token,
            }

        states: list[dict[str, Any]] = []
        previous_exit: dict[str, str] | None = None
        for lane in range(1, lane_count + 1):
            lane_boundary = "project_start" if lane == 1 else boundary
            transition_token = "product_wipe_transition" if boundary == "practical_transition" else ""
            if lane == 1 or boundary == "hard_cut":
                entry = state(lane, "entry", transition_token if lane > 1 else "")
            else:
                if previous_exit is None:
                    raise AssertionError("Advertisement planning defaults lost prior lane state.")
                entry = dict(previous_exit)
            exit_state = state(lane, "exit", transition_token)
            states.append(
                {
                    "lane_index": lane,
                    "boundary_mode": lane_boundary,
                    "entry_state": entry,
                    "exit_state": exit_state,
                }
            )
            previous_exit = exit_state
        brief = (
            f"{contract['brand_name']} {contract['product_name']} advertisement: preserve performer Subject 1 "
            f"from Pictures 1 and 2 and product Subject 2 from Picture 3 in all {shot_count} native shots. "
            f"Product behavior: {action} Use {boundary.replace('_', ' ')} between the {lane_count} H3 generation lane(s). "
            "Readable campaign copy is reserved for deterministic finishing, not invented inside generated frames."
        )
        return (
            _json(shots),
            _json(states),
            brief,
            f"Prepared {shot_count} product-aware shots and {lane_count} typed lane states with no hand-authored JSON required.",
            True,
        )


def _canonicalize_advertisement_subject_picture_citations(
    sections: dict[str, str],
) -> None:
    """Repair only unambiguous Picture citations in governed subject rows.

    H3 sometimes returns ``Picture 1`` where its own native contract spelling is
    ``<Picture 1>``.  Normalizing those exact authority citations before strict
    validation keeps the fail-closed contract while avoiding a meaningless
    retry.  No free-form description or unexpected Picture ordinal is changed.
    """

    subject_text = str(sections.get("subject_definitions", ""))
    governed = {
        "1": ("1", "2"),
        "2": ("3",),
    }
    for subject_ordinal, picture_ordinals in governed.items():
        row_pattern = re.compile(
            rf"(?im)^(\s*<Subject\s+{subject_ordinal}>\s*:\s*)(\S.*)$"
        )
        rows = list(row_pattern.finditer(subject_text))
        if len(rows) != 1:
            continue
        body = rows[0].group(2)
        for picture_ordinal in picture_ordinals:
            body = re.sub(
                rf"(?<!<)\bPicture\s+{picture_ordinal}\b(?!\s*>)",
                f"<Picture {picture_ordinal}>",
                body,
                flags=re.IGNORECASE,
            )
        subject_text = (
            subject_text[: rows[0].start()]
            + rows[0].group(1)
            + body
            + subject_text[rows[0].end() :]
        )
    sections["subject_definitions"] = subject_text


def _validate_advertisement_subjects(sections: Mapping[str, str]) -> None:
    subject_text = str(sections.get("subject_definitions", ""))
    retention = str(sections.get("retention_analysis", ""))
    for subject, pictures in (("<Subject 1>", ("<Picture 1>", "<Picture 2>")), ("<Subject 2>", ("<Picture 3>",))):
        rows = re.findall(rf"(?im)^\s*{re.escape(subject)}\s*:\s*(\S.*)$", subject_text)
        if len(rows) != 1:
            raise ValueError(f"Advertisement prompt must define {subject} exactly once.")
        if any(picture not in rows[0] for picture in pictures):
            raise ValueError(f"{subject} must cite its authoritative advertisement Picture references.")
        retained = re.findall(
            rf"(?im)^\s*{re.escape(subject)}(?:\s*\([^\n)]*\))?\s*:\s*fully_preserved\s*-\s*(\S.*)$",
            retention,
        )
        if len(retained) != 1:
            raise ValueError(f"Advertisement prompt must fully preserve {subject} exactly once.")
    if "<Picture 4>" in subject_text or "<Picture 4>" in retention:
        raise ValueError("Picture 4 is reserved for host-injected relay continuity and cannot be a source identity authority.")


def _strict_director_packet_json(value: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"Director packet contains non-finite JSON number {token}.")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"Director packet contains duplicate JSON key {key!r}.")
            result[key] = item
        return result

    try:
        payload = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Director packet remains invalid JSON after transport repair.") from exc
    pending: list[Any] = [payload]
    while pending:
        item = pending.pop()
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Director packet contains a non-finite JSON number.")
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    if not isinstance(payload, dict):
        raise ValueError("Director packet must decode to one JSON object.")
    return payload


def _repair_director_json_string_escapes(value: str) -> tuple[str, int]:
    """Escape only invalid backslashes occurring inside JSON strings."""

    output: list[str] = []
    in_string = False
    repairs = 0
    index = 0
    while index < len(value):
        character = value[index]
        if not in_string:
            output.append(character)
            if character == '"':
                in_string = True
            index += 1
            continue
        if character == '"':
            output.append(character)
            in_string = False
            index += 1
            continue
        if character != "\\":
            output.append(character)
            index += 1
            continue
        if index + 1 >= len(value):
            output.append(character)
            index += 1
            continue
        escaped = value[index + 1]
        if escaped in {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}:
            output.extend((character, escaped))
            index += 2
            continue
        output.extend(("\\", "\\"))
        repairs += 1
        index += 1
    return "".join(output), repairs


def _advertisement_reference_authorities(reference_contract_json: str) -> dict[str, tuple[str, ...]]:
    if len(str(reference_contract_json or "")) > 20_000:
        raise ValueError("reference_contract_json exceeds the Advertisement repair limit.")
    reference = _strict_director_packet_json(str(reference_contract_json or ""))
    if (
        reference.get("schema") != "diffusiongemma.advertisement_reference_contract"
        or type(reference.get("version")) is not int
        or reference.get("version") != VERSION
        or reference.get("ready") is not True
    ):
        raise ValueError("reference_contract_json is not a ready Advertisement reference contract.")
    manifest = str(reference.get("reference_manifest", ""))
    if not manifest or _sha_text(manifest) != str(reference.get("reference_manifest_sha256", "")):
        raise ValueError("reference_contract_json has a stale reference manifest hash.")
    subjects = reference.get("subjects")
    slots = reference.get("picture_slots")
    if not isinstance(subjects, list) or not isinstance(slots, list):
        raise ValueError("reference_contract_json has no governed subjects or Picture slots.")
    subject_tags = [str(item.get("subject_tag", "")) for item in subjects if isinstance(item, dict)]
    expected_count_value = reference.get("expected_subject_count")
    expected_count = expected_count_value if type(expected_count_value) is int else -1
    if (
        subject_tags != [f"<Subject {index}>" for index in range(1, len(subject_tags) + 1)]
        or expected_count != len(subject_tags)
        or len(subject_tags) != 2
    ):
        raise ValueError("reference_contract_json must define the two consecutive Advertisement subjects.")
    subject_descriptions = {
        str(item.get("subject_tag", "")): str(item.get("description", ""))
        for item in subjects
        if isinstance(item, dict)
    }
    policy = str(reference.get("slot_policy", ""))
    expected_layouts = {
        "performer_2_product_1_relay_1": (
            ("<Picture 1>", "performer_primary", "<Subject 1>"),
            ("<Picture 2>", "performer_supplemental", "<Subject 1>"),
            ("<Picture 3>", "product_package", "<Subject 2>"),
        ),
        "performer_1_product_1_relay_1": (
            ("<Picture 1>", "performer_primary", "<Subject 1>"),
            ("<Picture 2>", "product_package", "<Subject 2>"),
        ),
        "performer_2_product_1_relay_off": (
            ("<Picture 1>", "performer_primary", "<Subject 1>"),
            ("<Picture 2>", "performer_supplemental", "<Subject 1>"),
            ("<Picture 3>", "product_package", "<Subject 2>"),
        ),
    }
    expected_layout = expected_layouts.get(policy)
    if expected_layout is None:
        raise ValueError("reference_contract_json has an unsupported slot policy.")
    manifest_rows: dict[str, tuple[str, str, str]] = {}
    manifest_pattern = re.compile(
        r"(?im)^\s*(<Picture ([1-9]\d*)>)\s*:\s*\[dg:[^\]\n]+\]\s*"
        r"\[ad:role=([a-z0-9_]+);subject=(subject_[1-9]\d*);retention=([a-z0-9_]+)\]\s+\S.*$"
    )
    for row in manifest_pattern.finditer(manifest):
        picture_tag, _ordinal, role, subject_id, retention = row.groups()
        if picture_tag in manifest_rows:
            raise ValueError("reference_contract_json manifest repeats a Picture authority.")
        manifest_rows[picture_tag] = (role, subject_id, retention)
    manifest_lines = [line.strip() for line in manifest.splitlines() if line.strip()]
    if len(manifest_rows) != len(slots) or len(manifest_lines) != len(slots):
        raise ValueError("reference_contract_json manifest/slot authority counts differ.")
    authorities: dict[str, list[str]] = {tag: [] for tag in subject_tags}
    observed_pictures: set[str] = set()
    observed_layout: list[tuple[str, str, str]] = []
    for slot in slots:
        if not isinstance(slot, dict):
            raise ValueError("reference_contract_json contains a malformed Picture slot.")
        subject_tag = str(slot.get("subject_tag", ""))
        picture_tag = str(slot.get("picture_tag", ""))
        role = str(slot.get("role", ""))
        retention = str(slot.get("retention", ""))
        if subject_tag not in authorities or not re.fullmatch(r"<Picture [1-9]\d*>", picture_tag):
            raise ValueError("reference_contract_json contains an unsupported subject/Picture authority.")
        if picture_tag in observed_pictures or picture_tag == "<Picture 4>":
            raise ValueError("reference_contract_json contains a duplicate or relay-only Picture authority.")
        observed_pictures.add(picture_tag)
        subject_ordinal = int(re.search(r"\d+", subject_tag).group())
        manifest_authority = manifest_rows.get(picture_tag)
        if manifest_authority != (role, f"subject_{subject_ordinal}", retention):
            raise ValueError("reference_contract_json Picture slots disagree with the hashed manifest authority.")
        if retention != "fully_preserved" or str(slot.get("description", "")) != subject_descriptions.get(subject_tag):
            raise ValueError("reference_contract_json Picture slot retention/description is inconsistent.")
        observed_layout.append((picture_tag, role, subject_tag))
        authorities[subject_tag].append(picture_tag)
    if tuple(observed_layout) != expected_layout:
        raise ValueError("reference_contract_json Picture layout disagrees with its exported slot policy.")
    if any(not pictures for pictures in authorities.values()):
        raise ValueError("Every Advertisement subject needs at least one source Picture authority.")
    return {subject: tuple(pictures) for subject, pictures in authorities.items()}


def _complete_advertisement_subject_definition_rows(
    prompt: str,
    reference_contract_json: str,
) -> tuple[str, list[str]]:
    """Complete delimiters/citations only for the two governed subject rows."""

    authorities = _advertisement_reference_authorities(reference_contract_json)
    match = re.search(
        r"(?ms)\Asubject_definitions\s*:\s*\n(?P<body>.*?)\n\nsummary\s*:",
        str(prompt or ""),
    )
    if match is None:
        raise ValueError("Recovered Advertisement prompt has no exact subject_definitions/summary boundary.")
    body = match.group("body")
    lines = body.splitlines()
    repairs: list[str] = []
    seen_subjects: set[str] = set()
    for index, line in enumerate(lines):
        tag_match = re.match(r"^[ \t]*(<Subject ([1-9]\d*)>)(?P<rest>.*)$", line)
        if tag_match is None:
            if re.search(r"<\s*Subject\s+\d+\s*>", line, flags=re.IGNORECASE):
                raise ValueError("Recovered Advertisement prompt contains a non-canonical Subject tag.")
            continue
        tag = tag_match.group(1)
        if tag not in authorities or tag in seen_subjects:
            raise ValueError("Recovered Advertisement prompt contains an unknown or duplicate Subject definition.")
        seen_subjects.add(tag)
        rest = tag_match.group("rest").strip()
        description = re.sub(r"^(?::|=|[-–—](?:\s|$)|is\b)\s*", "", rest, count=1, flags=re.IGNORECASE).strip()
        if len(re.findall(r"[A-Za-z0-9]{2,}", description)) < 4:
            raise ValueError(f"Recovered Advertisement {tag} description is too short for narrow completion.")
        authored_pictures = {
            f"<Picture {int(ordinal)}>"
            for ordinal in re.findall(r"<\s*Picture\s+([1-9]\d*)\s*>", description, flags=re.IGNORECASE)
        }
        expected_pictures = set(authorities[tag])
        if authored_pictures - expected_pictures:
            raise ValueError(f"Recovered Advertisement {tag} cites an unauthorized Picture authority.")
        citation_suffix = ""
        if not expected_pictures.issubset(authored_pictures):
            citation_suffix = f" Reference sources: {', '.join(authorities[tag])}."
            repairs.append(f"completed_subject_authorities:{tag}")
        if not re.match(r"^\s*:", tag_match.group("rest")):
            repairs.append(f"inserted_subject_definition_colon:{tag}")
        lines[index] = f"{tag}: {description}{citation_suffix}"
    if seen_subjects != set(authorities):
        raise ValueError("Recovered Advertisement prompt does not define every governed Subject exactly once.")
    repaired_body = "\n".join(lines)
    repaired_prompt = str(prompt)[: match.start("body")] + repaired_body + str(prompt)[match.end("body") :]
    return repaired_prompt, repairs


def _grounding_report_would_block(report: Mapping[str, Any]) -> bool:
    return bool(
        str(report.get("decision", "")).strip().casefold() == "block"
        or report.get("would_block") is True
        or report.get("grounding_guard_would_block") is True
    )


class AdvertisementDirectorPacketRepair:
    """Recover one transport-only Director packet without bypassing host governance."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "director_final_json": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                ),
                "director_raw_response": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                ),
                "director_metadata_json": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                ),
                "grounding_status": (
                    "STRING",
                    {"default": "", "multiline": False, "forceInput": True},
                ),
                "grounding_report_json": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                ),
                "reference_contract_json": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("final_json", "repair_report_json", "status", "ready")
    FUNCTION = "repair"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Uses the host final packet normally. Only for a proven JSON-transport failure, it repairs invalid "
        "backslashes, completes governed Advertisement subject authorities, and replaces only the H3 creative "
        "field while preserving and cross-checking host grounding, runtime, cache, and block metadata."
    )

    def repair(
        self,
        director_final_json: str,
        director_raw_response: str,
        director_metadata_json: str,
        grounding_status: str,
        grounding_report_json: str,
        reference_contract_json: str,
    ):
        host_final = str(director_final_json or "").strip()
        raw = str(director_raw_response or "").strip()
        host_metadata_text = str(director_metadata_json or "").strip()
        grounding_report_text = str(grounding_report_json or "").strip()
        if not host_final or not host_metadata_text or not grounding_report_text:
            raise ValueError("Director host packet, metadata, and grounding report are required.")
        if any(len(value) > 100_000 for value in (host_final, raw, host_metadata_text, grounding_report_text)):
            raise ValueError("Director packet input exceeds the Advertisement transport-repair limit.")
        host_packet = _strict_director_packet_json(host_final)
        host_metadata = _strict_director_packet_json(host_metadata_text)
        grounding_report = _strict_director_packet_json(grounding_report_text)
        packet_metadata = host_packet.get("metadata")
        if not isinstance(packet_metadata, dict) or _json(packet_metadata) != _json(host_metadata):
            raise ValueError("Director final packet metadata does not match its host metadata output.")
        host_grounding = host_metadata.get("grounding_guard")
        if not isinstance(host_grounding, dict) or _json(host_grounding) != _json(grounding_report):
            raise ValueError("Director grounding report is missing or differs from host-authoritative metadata.")
        if grounding_report.get("schema") != "dg-grounding-report/1":
            raise ValueError("Director grounding report has an unsupported schema.")
        if (
            type(grounding_report.get("would_block")) is not bool
            or type(grounding_report.get("grounding_guard_would_block")) is not bool
        ):
            raise ValueError("Director grounding report block flags must be JSON booleans.")
        if not isinstance(grounding_report.get("decision"), str) or not isinstance(
            grounding_report.get("analysis_status"), str
        ):
            raise ValueError("Director grounding report decision/status must be strings.")
        if str(grounding_status or "").strip() != str(grounding_report.get("analysis_status", "")).strip():
            raise ValueError("Director grounding status differs from its host report.")

        for flag_name in (
            "json_parse_valid",
            "model_output_json_parse_valid",
            "plain_text_salvage",
            "used_template_fallback",
            "template_used_for_missing_fields",
        ):
            if flag_name in host_metadata and type(host_metadata[flag_name]) is not bool:
                raise ValueError(f"Director host metadata {flag_name} must be a JSON boolean.")
        for text_name in ("fallback_reason", "salvage_warning", "json_parse_warning"):
            if text_name in host_metadata and not isinstance(host_metadata[text_name], str):
                raise ValueError(f"Director host metadata {text_name} must be a string.")

        model_parse_key_present = "model_output_json_parse_valid" in host_metadata
        model_parse_state = host_metadata.get("model_output_json_parse_valid")
        host_parse_valid = bool(
            host_metadata.get("json_parse_valid") is True
            and (not model_parse_key_present or model_parse_state is True)
            and host_metadata.get("plain_text_salvage") is not True
            and not host_metadata.get("salvage_warning")
            and not host_metadata.get("json_parse_warning")
            and host_metadata.get("template_used_for_missing_fields") is not True
        )
        guard_blocked = _grounding_report_would_block(grounding_report)
        route = "host_final_json_passthrough"
        repair_count = 0
        contract_repairs: list[str] = []
        creative_prompt_replaced = False
        resolved_parse_state: dict[str, Any] = {}

        if not host_parse_valid and not guard_blocked:
            if (
                host_metadata.get("json_parse_valid") is not False
                or (model_parse_key_present and model_parse_state is not False)
                or host_metadata.get("plain_text_salvage") is not True
                or host_metadata.get("used_template_fallback") is True
                or host_metadata.get("fallback_reason")
            ):
                raise ValueError("Director host packet is not eligible for narrow transport recovery.")
            if not raw:
                raise ValueError("director_raw_response is empty during transport recovery.")
            route = "raw_transport_recovery"
            opening = "<|final_json|>"
            closing = "<|end_final_json|>"
            if raw.startswith(opening) and raw.endswith(closing):
                inner = raw[len(opening) : -len(closing)].strip()
                markers = "native_final_json_markers"
            elif raw.startswith("{") and raw.endswith("}"):
                inner = raw
                markers = "bare_json_object"
            else:
                raise ValueError(
                    "Director response must be one bare JSON object or one exact <|final_json|> packet."
                )
            repaired, repair_count = _repair_director_json_string_escapes(inner)
            if not 1 <= repair_count <= 32:
                raise ValueError("Director response is not a bounded invalid-backslash transport failure.")
            raw_payload = _strict_director_packet_json(repaired)
            prompt = raw_payload.get("minimax_h3_prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Director packet has no non-empty minimax_h3_prompt.")
            prompt, contract_repairs = _complete_advertisement_subject_definition_rows(
                prompt,
                reference_contract_json,
            )
            host_packet["minimax_h3_prompt"] = prompt
            creative_prompt_replaced = True
            resolved_keys = (
                "json_parse_valid",
                "model_output_json_parse_valid",
                "plain_text_salvage",
                "salvage_warning",
                "json_parse_warning",
                "template_used_for_missing_fields",
            )
            resolved_parse_state = {
                key: host_metadata.get(key)
                for key in resolved_keys
                if key in host_metadata
            }
            for key in (
                "plain_text_salvage",
                "salvage_warning",
                "json_parse_warning",
                "template_used_for_missing_fields",
            ):
                host_metadata.pop(key, None)
            host_metadata["json_parse_valid"] = True
            host_metadata["model_output_json_parse_valid"] = True
        else:
            markers = "not_evaluated"
            if guard_blocked:
                route = "host_grounding_block_preserved"

        transport_evidence = {
            "strategy": "host_packet_first_invalid_backslashes_only",
            "route": route,
            "marker_policy": markers,
            "invalid_escape_repair_count": repair_count,
            "contract_completions": contract_repairs,
            "original_transport_json_valid": host_parse_valid,
            "creative_prompt_replaced": creative_prompt_replaced,
            "creative_content_synthesized": False,
            "grounding_block_preserved": guard_blocked,
            "resolved_host_parse_state": resolved_parse_state,
            "raw_response_sha256": _sha_text(raw),
            "host_final_json_sha256": _sha_text(host_final),
            "host_metadata_sha256": _sha_text(host_metadata_text),
            "grounding_report_sha256": _sha_text(grounding_report_text),
        }
        host_metadata["advertisement_director_transport_repair"] = transport_evidence
        host_packet["metadata"] = host_metadata
        encoded = _json(host_packet)
        ready = not guard_blocked
        report = {
            "schema": "diffusiongemma.advertisement_director_transport_repair",
            "version": VERSION,
            **transport_evidence,
            "repaired_packet_sha256": _sha_text(encoded),
            "ready": ready,
        }
        if guard_blocked:
            status = "Advertisement Director host grounding block was preserved; raw recovery was not applied."
        elif creative_prompt_replaced:
            status = (
                f"Advertisement Director host packet recovered with {repair_count} invalid transport "
                f"escape{'s' if repair_count != 1 else ''}; host grounding/runtime/cache metadata preserved."
            )
        else:
            status = "Advertisement Director host final packet passed through with governance preserved."
        return encoded, _json(report), status, ready


def _source_starts(blocks: list[str], duration: float) -> list[float]:
    starts = [0.0]
    first = _leading_shot_timestamp_seconds(blocks[0])
    if first is not None and abs(first) > 1.0e-6:
        raise ValueError("Native [Shot 1] must begin at 00:00.000.")
    for index, block in enumerate(blocks[1:], start=2):
        value = _leading_shot_timestamp_seconds(block)
        if value is None:
            raise ValueError(f"Native [Shot {index}] needs a leading At MM:SS.mmm timestamp.")
        if value <= starts[-1] + 1.0e-6 or value >= duration - 1.0e-6:
            raise ValueError("Native advertisement shot timestamps must be strictly increasing inside the master duration.")
        starts.append(value)
    return starts


def _boundaries(duration: float, count: int, maximum: float) -> list[float]:
    values = [duration * index / count for index in range(count + 1)]
    lengths = [values[index + 1] - values[index] for index in range(count)]
    if any(value < 5.0 - 1.0e-6 or value > maximum + 1.0e-6 for value in lengths):
        raise ValueError("Advertisement duration cannot be divided into 5-15 second H3 generation lanes.")
    return values


class AdvertisementMultiShotPlanner:
    """Compile typed advertisement state into up to four native multi-shot lanes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "advertisement_contract_json": ("STRING", {"forceInput": True, "multiline": True}),
                "base_h3_prompt": ("STRING", {"forceInput": True, "multiline": True}),
                "shot_metadata_json": ("STRING", {"default": "[]", "multiline": True}),
                "lane_states_json": ("STRING", {"default": "[]", "multiline": True}),
                "performance_mode": (
                    list(ADVERTISEMENT_PERFORMANCE_CARRIER_MODES),
                    {"default": "Natural / audio-led sync"},
                ),
            }
        }

    _lane_types: list[str] = []
    _lane_names: list[str] = []
    for _index in range(1, MAX_H3_SHOTS + 1):
        _lane_types.extend(("STRING", "FLOAT", "FLOAT", "INT", "BOOLEAN"))
        _lane_names.extend(
            (
                f"lane_{_index}_prompt",
                f"lane_{_index}_start",
                f"lane_{_index}_duration",
                f"lane_{_index}_frames",
                f"lane_{_index}_ready",
            )
        )

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN", "INT", *_lane_types)
    RETURN_NAMES = ("advertisement_plan_json", "status", "ready", "generation_lane_count", *_lane_names)
    FUNCTION = "plan"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Preserves every native H3 shot while dividing a campaign into at most four 15-second lanes. "
        "It governs performer Pictures 1/2, product Picture 3, relay Picture 4, product presence, and typed seam state."
    )

    def plan(
        self,
        advertisement_contract_json: str,
        base_h3_prompt: str,
        shot_metadata_json: str,
        lane_states_json: str,
        performance_mode: str,
    ):
        contract = _contract(advertisement_contract_json)
        if (
            str(contract.get("audio_lock", {}).get("soundtrack_content_mode", "")) == "instrumental"
            and performance_mode == "Lyrics + lip sync"
        ):
            raise ValueError("An instrumental advertisement soundtrack cannot use Lyrics + lip sync performance mode.")
        planning = contract["h3_planning"]
        duration = float(contract["production_duration_seconds"])
        lane_count = int(planning["generation_lane_count"])
        maximum = float(planning["maximum_lane_seconds"])
        native_count = int(planning["native_shot_count"])
        mode, fields, sections = _section_map(base_h3_prompt)
        if mode != "ref2va":
            raise ValueError("Advertisement planning requires the native six-section Ref2VA prompt.")
        _canonicalize_advertisement_subject_picture_citations(sections)
        _validate_advertisement_subjects(sections)
        detail = sections["detailed_description"]
        prefix, blocks = _shot_blocks(detail)
        if len(blocks) != native_count:
            raise ValueError(
                f"Advertisement contract requires {native_count} native shots, but the validated prompt contains {len(blocks)}."
            )
        _validate_source_timestamps(blocks, duration)
        starts = _source_starts(blocks, duration)
        metadata = _shot_metadata(_parse_array(shot_metadata_json, "shot_metadata_json"), native_count)
        lane_state = _lane_states(
            _parse_array(lane_states_json, "lane_states_json"),
            lane_count,
            str(planning["boundary_mode"]),
        )
        boundary_values = _boundaries(duration, lane_count, maximum)
        lanes: list[dict[str, Any]] = []
        outputs: list[Any] = []
        for lane_index in range(lane_count):
            lane_start = boundary_values[lane_index]
            lane_end = boundary_values[lane_index + 1]
            active_index = max(index for index, value in enumerate(starts) if value <= lane_start + 1.0e-6)
            selected = [
                index for index, value in enumerate(starts) if index >= active_index and value < lane_end - 1.0e-6
            ]
            if not selected:
                raise ValueError(f"Advertisement lane {lane_index + 1} has no native source shot.")
            carry = starts[active_index] < lane_start - 1.0e-6
            local_blocks: list[str] = []
            per_shot: list[dict[str, Any]] = []
            for local_position, source_index in enumerate(selected):
                block = blocks[source_index]
                if local_position == 0 and carry:
                    block = _carry_in_source_block(block, starts[source_index], lane_start)
                shot_meta = dict(metadata[source_index])
                directive = (
                    f"Product presence contract for source [Shot {source_index + 1}]: "
                    f"{shot_meta['product_presence']}."
                )
                if shot_meta["product_action"]:
                    directive += f" Product action: {shot_meta['product_action']}."
                block = f"{block.rstrip()} {directive}".strip()
                local_blocks.append(block)
                per_shot.append(shot_meta)
            state = lane_state[lane_index]
            state_directive = (
                "Typed lane state contract. Opening: "
                + _json(state["entry_state"])
                + ". Required retained exit: "
                + _json(state["exit_state"])
                + f". Boundary mode: {state['boundary_mode']}."
            )
            local_blocks[0] = f"{state_directive} {local_blocks[0]}"
            relay_required = lane_index > 0 and state["boundary_mode"] == "relay_continuity"
            prompt = _segment_prompt(
                mode,
                fields,
                sections,
                prefix,
                local_blocks,
                lane_index + 1,
                lane_count,
                lane_start,
                lane_end - lane_start,
                performance_mode,
                [],
                "natural_audio_led",
                ["<Picture 1>", "<Picture 2>"],
                "<Picture 4>" if relay_required else "",
                "<Subject 1>",
            )
            if "<Picture 3>" not in prompt or "<Subject 2>" not in prompt:
                raise ValueError("Advertisement lane prompt lost the governed product/package reference contract.")
            master_frame_start = int(round(lane_start * H3_FPS))
            master_frame_end = int(round(lane_end * H3_FPS))
            retained_frames = master_frame_end - master_frame_start
            lane = {
                "lane_index": lane_index + 1,
                "start_seconds": round(lane_start, 6),
                "duration_seconds": round(lane_end - lane_start, 6),
                "master_frame_start": master_frame_start,
                "master_frame_end_exclusive": master_frame_end,
                "retained_master_frames": retained_frames,
                "generated_h3_frames": _h3_frames(lane_end - lane_start),
                "source_native_shot_indices": [value + 1 for value in selected],
                "source_native_shot_count": len(selected),
                "source_shot_continuation_at_lane_start": carry,
                "product_presence": per_shot,
                "entry_state": state["entry_state"],
                "exit_state": state["exit_state"],
                "boundary_mode": state["boundary_mode"],
                "reference_picture_tags": [
                    "<Picture 1>",
                    "<Picture 2>",
                    "<Picture 3>",
                    *(["<Picture 4>"] if relay_required else []),
                ],
                "relay_required": relay_required,
                "prompt": prompt,
                "prompt_sha256": _sha_text(prompt),
                "ready": True,
            }
            lanes.append(lane)
            outputs.extend(
                (
                    prompt,
                    float(lane_start),
                    float(lane_end - lane_start),
                    int(lane["generated_h3_frames"]),
                    True,
                )
            )
        unique_indices = sorted(
            {
                shot_index
                for lane in lanes
                for shot_index in lane["source_native_shot_indices"]
            }
        )
        if unique_indices != list(range(1, native_count + 1)):
            raise ValueError("Advertisement lane partition did not preserve every native shot exactly once as source material.")
        while len(outputs) < MAX_H3_SHOTS * 5:
            outputs.extend(("", 0.0, 0.0, 0, False))
        plan: dict[str, Any] = {
            "schema": ADVERTISEMENT_PLAN_SCHEMA,
            "version": VERSION,
            "campaign_id": contract["campaign_id"],
            "advertisement_contract_sha256": contract["contract_sha256"],
            "production_duration_seconds": duration,
            "fps": H3_FPS,
            "source_native_shot_count": native_count,
            "effective_generation_lane_count": lane_count,
            "reference_policy": {
                "performer_subject": "<Subject 1>",
                "product_subject": "<Subject 2>",
                "performer_pictures": ["<Picture 1>", "<Picture 2>"],
                "product_picture": "<Picture 3>",
                "relay_picture": "<Picture 4>",
                "independent_fully_preserved_subjects": True,
            },
            "lanes": lanes,
            "ready": True,
        }
        plan["plan_sha256"] = _sha_text(_json(plan))
        status = (
            f"Advertisement {contract['campaign_slug']} preserves {native_count} native shots across "
            f"{lane_count} H3 lanes with persistent Pictures 1-3 and Picture 4 reserved only for typed relay seams."
        )
        return _json(plan), status, True, lane_count, *outputs


def _default_relay_root() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory()) / "diffusiongemma_advertisement" / "relay_artifacts"
    except ModuleNotFoundError:
        return Path.cwd() / "output" / "diffusiongemma_advertisement" / "relay_artifacts"


def _tail_tensor(images: Any, max_megapixels: float, retained_frames: int | None = None) -> torch.Tensor:
    tensor = images if isinstance(images, torch.Tensor) else torch.as_tensor(images)
    if tensor.ndim != 4 or int(tensor.shape[-1]) != 3 or int(tensor.shape[0]) < 1:
        raise ValueError("lane_images must be a non-empty ComfyUI RGB IMAGE batch.")
    if retained_frames is None:
        tail_index = int(tensor.shape[0]) - 1
    else:
        retained = int(retained_frames)
        if retained < 1 or int(tensor.shape[0]) < retained:
            raise ValueError(
                f"lane_images must contain at least {retained} frames to persist the exact retained relay tail."
            )
        tail_index = retained - 1
    tail = tensor[tail_index : tail_index + 1].detach().to(device="cpu", dtype=torch.float32).contiguous()
    height, width = int(tail.shape[1]), int(tail.shape[2])
    target_pixels = max(32 * 32, int(float(max_megapixels) * 1_000_000))
    if height * width > target_pixels:
        scale = math.sqrt(target_pixels / float(height * width))
        out_width = max(32, int(math.floor(width * scale / 16.0)) * 16)
        out_height = max(32, int(math.floor(height * scale / 16.0)) * 16)
        tail = torch_functional.interpolate(
            tail.permute(0, 3, 1, 2),
            size=(out_height, out_width),
            mode="area",
        ).permute(0, 2, 3, 1).contiguous()
    return tail


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    header = _json({"shape": list(value.shape), "dtype": "float32"}).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    payload = memoryview(value.view(torch.uint8).numpy()).cast("B")
    for offset in range(0, len(payload), 1024 * 1024):
        digest.update(payload[offset : offset + 1024 * 1024])
    return digest.hexdigest()


class AdvertisementRelayArtifact:
    """Persist one small, safe, hash-addressed previous-lane tail."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lane_images": ("IMAGE",),
                "advertisement_plan_json": ("STRING", {"forceInput": True, "multiline": True}),
                "campaign_id": ("STRING", {"forceInput": True}),
                "completed_lane_index": ("INT", {"default": 1, "min": 1, "max": 3}),
                "next_lane_index": ("INT", {"default": 2, "min": 2, "max": 4}),
                "max_megapixels": ("FLOAT", {"default": 0.064, "min": 0.01, "max": 0.25, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("relay_artifact_json", "artifact_id", "status", "ready")
    FUNCTION = "persist"
    CATEGORY = CATEGORY
    DESCRIPTION = "Persists one hash-addressed relay tail independently from ComfyUI's in-memory cache."

    def persist(
        self,
        lane_images: Any,
        advertisement_plan_json: str,
        campaign_id: str,
        completed_lane_index: int,
        next_lane_index: int,
        max_megapixels: float = 0.064,
        artifact_root: str | Path | None = None,
    ):
        plan = _plan(advertisement_plan_json)
        campaign = str(campaign_id or "").strip()
        if campaign != str(plan["campaign_id"]):
            raise ValueError("Relay artifact campaign_id does not match the advertisement plan.")
        completed = int(completed_lane_index)
        next_lane = int(next_lane_index)
        if next_lane != completed + 1:
            raise ValueError("Relay artifact must connect immediately consecutive generation lanes.")
        lanes = plan["lanes"]
        if completed < 1 or next_lane > len(lanes):
            raise ValueError("Relay artifact lane indices are outside the advertisement plan.")
        if not bool(lanes[next_lane - 1].get("relay_required")):
            raise ValueError("The destination advertisement lane does not request relay continuity.")
        retained_frames = int(lanes[completed - 1].get("retained_master_frames", 0))
        tail = _tail_tensor(lane_images, float(max_megapixels), retained_frames)
        pixel_hash = _tensor_sha256(tail)
        identity = {
            "schema": ADVERTISEMENT_RELAY_SCHEMA,
            "version": VERSION,
            "campaign_id": campaign,
            "plan_sha256": str(plan["plan_sha256"]),
            "completed_lane_index": completed,
            "next_lane_index": next_lane,
            "picture_tag": "<Picture 4>",
            "pixel_sha256": pixel_hash,
            "shape": list(tail.shape),
            "source_retained_frame_index": retained_frames - 1,
        }
        artifact_id = _sha_text(_json(identity))
        root = Path(artifact_root) if artifact_root is not None else _default_relay_root()
        root.mkdir(parents=True, exist_ok=True)
        filename = f"{artifact_id}.safetensors"
        path = root / filename
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as exc:
            raise RuntimeError("Advertisement relay persistence requires safetensors.") from exc
        if not path.exists():
            temporary = root / f".{artifact_id}.{os.getpid()}.tmp.safetensors"
            save_file({"image": tail}, str(temporary), metadata={"manifest": _json(identity)})
            os.replace(temporary, path)
        file_hash = _sha_bytes(path.read_bytes())
        manifest = {
            **identity,
            "artifact_id": artifact_id,
            "filename": filename,
            "file_sha256": file_hash,
            "persistent": True,
        }
        return (
            _json(manifest),
            artifact_id,
            f"Persisted relay Picture 4 for advertisement lane {next_lane} as {artifact_id[:12]}….",
            True,
        )


class AdvertisementRelayGate:
    """Load and verify a persistent Picture 4 relay artifact."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "relay_artifact_json": ("STRING", {"forceInput": True, "multiline": True}),
                "advertisement_plan_json": ("STRING", {"forceInput": True, "multiline": True}),
                "campaign_id": ("STRING", {"forceInput": True}),
                "lane_index": ("INT", {"default": 2, "min": 2, "max": 4}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("relay_reference", "relay_picture_tag", "status", "ready")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Verifies campaign, plan, lane, content hash, and file hash before emitting persistent Picture 4."

    def load(
        self,
        relay_artifact_json: str,
        advertisement_plan_json: str,
        campaign_id: str,
        lane_index: int,
        enabled: bool = True,
        artifact_root: str | Path | None = None,
    ):
        if not bool(enabled):
            return None, "", "Advertisement relay is disabled.", True
        plan = _plan(advertisement_plan_json)
        manifest = _parse_object(relay_artifact_json, "relay_artifact_json")
        campaign = str(campaign_id or "").strip()
        lane = int(lane_index)
        if manifest.get("schema") != ADVERTISEMENT_RELAY_SCHEMA or manifest.get("version") != VERSION:
            raise ValueError("relay_artifact_json has an unsupported schema or version.")
        if campaign != str(plan["campaign_id"]) or campaign != str(manifest.get("campaign_id", "")):
            raise ValueError("Relay artifact campaign identity mismatch.")
        if str(manifest.get("plan_sha256", "")) != str(plan["plan_sha256"]):
            raise ValueError("Relay artifact belongs to a different advertisement plan.")
        if int(manifest.get("next_lane_index", 0)) != lane:
            raise ValueError("Relay artifact destination lane mismatch.")
        if lane > len(plan["lanes"]) or not bool(plan["lanes"][lane - 1].get("relay_required")):
            raise ValueError("The requested advertisement lane does not permit relay Picture 4.")
        artifact_id = str(manifest.get("artifact_id", ""))
        if not SHA256_RE.fullmatch(artifact_id):
            raise ValueError("Relay artifact_id is invalid.")
        expected_filename = f"{artifact_id}.safetensors"
        if str(manifest.get("filename", "")) != expected_filename:
            raise ValueError("Relay artifact filename is not hash-addressed by artifact_id.")
        root = Path(artifact_root) if artifact_root is not None else _default_relay_root()
        path = root / expected_filename
        if not path.is_file():
            raise ValueError("Persistent relay artifact file is missing.")
        if _sha_bytes(path.read_bytes()) != str(manifest.get("file_sha256", "")):
            raise ValueError("Persistent relay artifact file SHA-256 mismatch.")
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise RuntimeError("Advertisement relay loading requires safetensors.") from exc
        tensors = load_file(str(path), device="cpu")
        image = tensors.get("image")
        if image is None or _tensor_sha256(image) != str(manifest.get("pixel_sha256", "")):
            raise ValueError("Persistent relay artifact pixel SHA-256 mismatch.")
        return (
            image,
            "<Picture 4>",
            f"Verified persistent Picture 4 for advertisement lane {lane}.",
            True,
        )


NODE_CLASS_MAPPINGS = {
    "DiffusionGemmaAdvertisementDirectorPacketRepair": AdvertisementDirectorPacketRepair,
    "DiffusionGemmaAdvertisementMasterContract": AdvertisementMasterContract,
    "DiffusionGemmaAdvertisementPlanningDefaults": AdvertisementPlanningDefaults,
    "DiffusionGemmaAdvertisementMultiShotPlanner": AdvertisementMultiShotPlanner,
    "DiffusionGemmaAdvertisementRelayArtifact": AdvertisementRelayArtifact,
    "DiffusionGemmaAdvertisementRelayGate": AdvertisementRelayGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionGemmaAdvertisementDirectorPacketRepair": "DiffusionGemma Advertisement Director Packet Repair",
    "DiffusionGemmaAdvertisementMasterContract": "DiffusionGemma Advertisement Master Contract",
    "DiffusionGemmaAdvertisementPlanningDefaults": "DiffusionGemma Advertisement Planning Defaults",
    "DiffusionGemmaAdvertisementMultiShotPlanner": "DiffusionGemma Advertisement Multi-Shot Planner",
    "DiffusionGemmaAdvertisementRelayArtifact": "DiffusionGemma Advertisement Relay Artifact",
    "DiffusionGemmaAdvertisementRelayGate": "DiffusionGemma Advertisement Relay Gate",
}


__all__ = [
    "AdvertisementDirectorPacketRepair",
    "AdvertisementMasterContract",
    "AdvertisementPlanningDefaults",
    "AdvertisementMultiShotPlanner",
    "AdvertisementRelayArtifact",
    "AdvertisementRelayGate",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
