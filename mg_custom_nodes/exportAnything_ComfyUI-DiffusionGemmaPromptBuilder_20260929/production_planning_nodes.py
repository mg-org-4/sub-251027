# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

"""Project, generation-segment, assembly, and delivery contracts for long-form H3 work.

The nodes in this module are deliberately lightweight.  They do not call a
model or mutate media.  They turn an approved soundtrack, a validated H3
prompt, and decoded-audio evidence into an auditable project/shot contract.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from typing import Any, Mapping

import torch

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ModuleNotFoundError:  # Standalone repository tests.
    class ExecutionBlocker:  # type: ignore[no-redef]
        def __init__(self, message: str | None) -> None:
            self.message = message


CATEGORY = "prompt/diffusiongemma/production-planning"
PROJECT_SCHEMA = "diffusiongemma.project_master_contract"
SHOT_PLAN_SCHEMA = "diffusiongemma.audio_aware_h3_shot_plan"
IDENTITY_RELAY_POLICY_SCHEMA = "diffusiongemma.h3_identity_relay_policy"
ASSEMBLY_SCHEMA = "diffusiongemma.h3_shot_assembly"
DELIVERY_SCHEMA = "diffusiongemma.multi_format_delivery_plan"
VERSION = 1
BOUNDARY_SELECTION_REVISION = 2
H3_FPS = 24.0
MAX_H3_SHOTS = 4
MAX_PROJECT_SECONDS = 60.0
ASPECT_RATIOS = ("9:16", "16:9", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHOT_RE = re.compile(
    r"(?is)\[Shot\s+(\d+)\]\s*(.*?)(?=(?:\[Shot\s+\d+\])|\Z)"
)
_EMBEDDED_TIMESTAMP_RE = re.compile(
    r"\bAt\s+(?P<minutes>\d{2,}):(?P<seconds>\d{2})\.(?P<milliseconds>\d{3})\b",
    flags=re.IGNORECASE,
)
_REF_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_BASE_FIELDS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)
PERFORMANCE_MODES = (
    "Dance / music sync",
    "Lyrics + lip sync",
    "Natural / audio-led sync",
)
CONTINUITY_RELAY_MODES = ("Previous lane tail", "Off")
MUSIC_REPORT_SCHEMA = "diffusiongemma.music_audition_report"
MUSIC_REPORT_VERSION = 1
TIMED_LYRICS_REPORT_SCHEMA = "diffusiongemma.timed_lyrics_report"
TIMED_LYRICS_REPORT_VERSION = 1


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _positive_finite(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(number) or number < 0.0 or (not allow_zero and number <= 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a finite {qualifier} number.")
    return number


def _parse_object(text: Any, label: str) -> dict[str, Any]:
    try:
        value = json.loads(str(text or ""))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _canonical_aspect(value: Any) -> str:
    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", str(value or ""))
    if not match:
        raise ValueError("aspect_ratio must be a supported W:H ratio.")
    result = f"{int(match.group(1))}:{int(match.group(2))}"
    if result not in ASPECT_RATIOS:
        raise ValueError("aspect_ratio must be one of " + ", ".join(ASPECT_RATIOS) + ".")
    return result


def _h3_frames(duration_seconds: float, fps: float = H3_FPS) -> int:
    requested = max(5, int(math.ceil(float(duration_seconds) * float(fps) - 1e-9)))
    remainder = (requested - 5) % 17
    return requested if remainder == 0 else requested + (17 - remainder)


def _section_map(prompt: str) -> tuple[str, tuple[str, ...], dict[str, str]]:
    text = str(prompt or "").strip()
    if not text:
        raise ValueError("base_h3_prompt is empty.")
    candidates = (("ref2va", _REF_FIELDS), ("base", _BASE_FIELDS))
    for mode, fields in candidates:
        matches: list[tuple[str, int, int]] = []
        for field in fields:
            match = re.search(rf"(?m)^\s*{re.escape(field)}\s*:\s*", text)
            if match is None:
                matches = []
                break
            matches.append((field, match.start(), match.end()))
        if not matches or [item[1] for item in matches] != sorted(item[1] for item in matches):
            continue
        if text[: matches[0][1]].strip():
            continue
        sections: dict[str, str] = {}
        for index, (field, _start, body_start) in enumerate(matches):
            body_end = matches[index + 1][1] if index + 1 < len(matches) else len(text)
            sections[field] = text[body_start:body_end].strip()
        return mode, fields, sections
    raise ValueError("base_h3_prompt does not have the native H3 three- or six-section structure.")


def _render_sections(fields: tuple[str, ...], sections: Mapping[str, str]) -> str:
    return "\n\n".join(f"{field}:\n{str(sections.get(field, '')).strip()}" for field in fields)


def _canonical_reference_tag(kind: Any, ordinal: Any) -> str:
    return f"<{str(kind).title()} {int(ordinal)}>"


def _reference_tags(text: Any, *, include_subjects: bool = False) -> list[str]:
    kinds = "Subject|Picture|Video|Audio" if include_subjects else "Picture|Video|Audio"
    tags: list[str] = []
    for kind, ordinal in re.findall(
        rf"<\s*({kinds})\s+([1-9]\d*)\s*>",
        str(text or ""),
        flags=re.IGNORECASE,
    ):
        tag = _canonical_reference_tag(kind, ordinal)
        if tag not in tags:
            tags.append(tag)
    return tags


def _manifest_reference_definitions(manifest: Any) -> list[tuple[str, str]]:
    definitions: list[tuple[str, str]] = []
    for match in re.finditer(
        r"(?m)^[ \t]*<\s*(Picture|Video|Audio)\s+([1-9]\d*)\s*>"
        r"[ \t]*(?::|=|[-\u2013\u2014])[ \t]*(\S.*)$",
        str(manifest or ""),
        flags=re.IGNORECASE,
    ):
        definitions.append(
            (_canonical_reference_tag(match.group(1), match.group(2)), match.group(3).strip())
        )
    return definitions


def _subject_definition_items(text: Any) -> list[tuple[str, str]]:
    return [
        (_canonical_reference_tag("Subject", ordinal), description.strip())
        for ordinal, description in re.findall(
            r"(?im)^[ \t]*<\s*Subject\s+([1-9]\d*)\s*>"
            r"[ \t]*(?::|=|[-\u2013\u2014]|\bis\b)[ \t]*(\S.*)$",
            str(text or ""),
        )
    ]


def _replace_subject_row(
    text: Any,
    subject_tag: str,
    replacement: str,
    *,
    retention_row: bool = False,
) -> str:
    """Replace one semantic Subject row without accepting duplicate bindings."""

    ordinal_match = re.fullmatch(r"<Subject ([1-9]\d*)>", str(subject_tag or ""))
    if ordinal_match is None:
        raise ValueError("Internal identity policy received an invalid Subject tag.")
    qualifier = r"(?:[ \t]*\([^\n)]*\))?" if retention_row else ""
    separator = r"[ \t]*:" if retention_row else r"[ \t]*(?::|=|[-\u2013\u2014]|\bis\b)"
    pattern = re.compile(
        rf"(?im)^[ \t]*<\s*Subject\s+{ordinal_match.group(1)}\s*>"
        rf"{qualifier}{separator}[ \t]*\S.*$"
    )
    source = str(text or "").strip()
    matches = list(pattern.finditer(source))
    if len(matches) > 1:
        raise ValueError(f"{subject_tag} is defined more than once.")
    if not matches:
        return (source + "\n" + replacement).strip()
    return pattern.sub(lambda _match: replacement, source, count=1)


def _has_unnegated_distinct_identity(text: Any) -> bool:
    distinct = re.compile(
        r"\b(?:(?:different|another|second|third|additional|other|new|unrelated|distinct)\s+"
        r"(?:subjects?|persons?|people|women|woman|men|man|performers?|characters?|"
        r"models?|individuals?|identit(?:y|ies))|separate\s+"
        r"(?:subject|person|woman|man|performer|character|model|individual|identity)|"
        r"(?:multiple|two)\s+(?:subjects?|persons?|people|women|men|performers?|"
        r"characters?|models?|individuals?|identit(?:y|ies)))\b",
        flags=re.IGNORECASE,
    )
    negated = re.compile(
        r"\b(?:not|never|no|is\s+not|are\s+not|was\s+not|were\s+not|"
        r"do\s+not|does\s+not|don't|doesn't|cannot|can't|must\s+not|mustn't|"
        r"without|avoid|avoids|avoiding)\b[^.!?;]{0,40}$",
        flags=re.IGNORECASE,
    )
    source = str(text or "")
    return any(
        negated.search(source[max(0, match.start() - 64) : match.start()]) is None
        for match in distinct.finditer(source)
    )


def _dual_identity_subject_contract(
    sections: dict[str, str], identity_tags: list[str]
) -> tuple[str, list[str]]:
    """Bind same-person identity anchors using the already-validated host manifest.

    DiffusionGemma is allowed to vary prose and retention vocabulary.  The host
    owns the fact that Picture 1 and Picture 2 are complementary evidence for
    one person, so harmless omissions are canonicalized here.  Explicit anchor
    bindings to multiple semantic Subjects still fail closed.
    """

    subject_text = str(sections.get("subject_definitions", ""))
    subject_items = _subject_definition_items(subject_text)
    if not subject_items:
        raise ValueError(
            "Dual identity pictures require exactly one identifiable semantic <Subject N> binding in subject_definitions."
        )
    subject_tags = [tag for tag, _description in subject_items]
    duplicates = sorted({tag for tag in subject_tags if subject_tags.count(tag) > 1})
    if duplicates:
        raise ValueError(
            "Duplicate semantic Subject definition(s) are not allowed: "
            + ", ".join(duplicates)
            + "."
        )

    anchor_set = set(identity_tags)
    anchor_bindings: dict[str, set[str]] = {
        tag: set(_reference_tags(description)).intersection(anchor_set)
        for tag, description in subject_items
    }
    same_identity = re.compile(
        r"\b(?:same|one)\s+"
        r"(?:subject|person|woman|man|performer|character|model|individual)\b",
        flags=re.IGNORECASE,
    )
    for _tag, description in subject_items:
        if "<Picture 2>" in _reference_tags(
            description
        ) and _has_unnegated_distinct_identity(description):
            raise ValueError(
                "<Picture 2> is described as a different identity, which conflicts with the validated same-person reference_manifest."
            )
    # A Picture definition may explicitly name its semantic Subject even when
    # the Subject row itself omits the reciprocal Picture citation.
    picture_definitions: dict[str, list[str]] = {
        "<Picture 1>": [],
        "<Picture 2>": [],
    }
    for line in subject_text.splitlines():
        raw_picture_match = re.match(
            r"^[ \t]*<\s*Picture\s+([12])\s*>",
            line,
            flags=re.IGNORECASE,
        )
        if raw_picture_match is None:
            continue
        picture_tag = _canonical_reference_tag("Picture", raw_picture_match.group(1))
        picture_match = re.match(
            r"^[ \t]*<\s*Picture\s+[12]\s*>"
            r"[ \t]*(?::|=|[-\u2013\u2014]|\bis\b)[ \t]*(\S.*)$",
            line,
            flags=re.IGNORECASE,
        )
        picture_description = picture_match.group(1) if picture_match else ""
        picture_definitions[picture_tag].append(picture_description)
        if picture_match is None:
            continue
        if picture_tag == "<Picture 2>" and _has_unnegated_distinct_identity(
            picture_description
        ):
            raise ValueError(
                "<Picture 2> is described as a different identity, which conflicts with the validated same-person reference_manifest."
            )
        for referenced_tag in _reference_tags(
            picture_description, include_subjects=True
        ):
            if not referenced_tag.startswith("<Subject "):
                continue
            if referenced_tag not in anchor_bindings:
                raise ValueError(
                    f"{picture_tag} names undefined semantic {referenced_tag} in subject_definitions."
                )
            anchor_bindings[referenced_tag].add(picture_tag)
    duplicate_picture_definitions = [
        tag for tag, descriptions in picture_definitions.items() if len(descriptions) > 1
    ]
    if duplicate_picture_definitions:
        raise ValueError(
            "Duplicate identity Picture definition row(s) are not allowed: "
            + ", ".join(duplicate_picture_definitions)
            + "."
        )
    incomplete_picture_definitions = [
        tag
        for tag, descriptions in picture_definitions.items()
        if descriptions and not descriptions[0]
    ]
    if incomplete_picture_definitions:
        raise ValueError(
            "Identity Picture definition row(s) are incomplete: "
            + ", ".join(incomplete_picture_definitions)
            + "."
        )

    jointly_bound = [
        tag for tag, bindings in anchor_bindings.items() if anchor_set.issubset(bindings)
    ]
    partially_bound = [tag for tag, bindings in anchor_bindings.items() if bindings]
    if len(jointly_bound) == 1:
        identity_subject_tag = jointly_bound[0]
        conflicting = [tag for tag in partially_bound if tag != identity_subject_tag]
        if conflicting:
            raise ValueError(
                "Identity Picture tags cannot be split across multiple semantic Subjects: "
                + ", ".join([identity_subject_tag, *conflicting])
                + "."
            )
    elif not jointly_bound:
        primary_candidates = [
            tag
            for tag, bindings in anchor_bindings.items()
            if bindings == {"<Picture 1>"}
        ]
        picture_2_rows = picture_definitions["<Picture 2>"]
        if (
            len(primary_candidates) == 1
            and partially_bound == primary_candidates
            and len(picture_2_rows) == 1
            and same_identity.search(picture_2_rows[0]) is not None
        ):
            identity_subject_tag = primary_candidates[0]
        else:
            involved = partially_bound or subject_tags
            raise ValueError(
                "Dual identity pictures have conflicting or ambiguous semantic Subject bindings: "
                + ", ".join(involved)
                + "."
            )
    else:
        involved = partially_bound or subject_tags
        raise ValueError(
            "Dual identity pictures have conflicting or ambiguous semantic Subject bindings: "
            + ", ".join(involved)
            + "."
        )

    repairs: list[str] = []
    subject_description = dict(subject_items)[identity_subject_tag]
    if not anchor_set.issubset(set(_reference_tags(subject_description))):
        punctuation = "" if re.search(r"[.!?]\s*$", subject_description) else "."
        canonical_binding = (
            f"{identity_subject_tag}: {subject_description}{punctuation} "
            "The same performer is jointly identified by <Picture 1> and <Picture 2>; "
            "<Picture 1> remains the body, wardrobe, and target-composition authority, "
            "while <Picture 2> supplies complementary identity evidence only."
        )
        sections["subject_definitions"] = _replace_subject_row(
            subject_text, identity_subject_tag, canonical_binding
        )
        repairs.append("joint_identity_subject_binding")

    retention_text = str(sections.get("retention_analysis", ""))
    retention_pattern = re.compile(
        rf"(?im)^\s*{re.escape(identity_subject_tag)}(?:\s*\([^\n)]*\))?\s*:\s*"
        r"([A-Za-z][A-Za-z0-9_-]*)\s*-\s*(\S.*)$"
    )
    subject_ordinal = re.fullmatch(
        r"<Subject ([1-9]\d*)>", identity_subject_tag
    ).group(1)
    raw_retention_pattern = re.compile(
        rf"(?im)^\s*<\s*Subject\s+{subject_ordinal}\s*>"
        r"(?:\s*\([^\n)]*\))?\s*:.*$"
    )
    raw_retention_rows = list(raw_retention_pattern.finditer(retention_text))
    if len(raw_retention_rows) != 1:
        raise ValueError(
            f"{identity_subject_tag} must have exactly one retention row for the dual identity anchors."
        )
    retention_rows = list(retention_pattern.finditer(retention_text))
    if len(retention_rows) != 1:
        raise ValueError(
            f"{identity_subject_tag} must have exactly one retention row for the dual identity anchors."
        )
    retention_mode = retention_rows[0].group(1).casefold()
    if retention_mode == "attribute_transfer":
        canonical_retention = (
            f"{identity_subject_tag}: fully_preserved - Preserve the same identity, face, "
            "gender presentation, hair, body/build, and wardrobe jointly established by "
            "<Picture 1> and <Picture 2> across every generated lane."
        )
        sections["retention_analysis"] = _replace_subject_row(
            retention_text,
            identity_subject_tag,
            canonical_retention,
            retention_row=True,
        )
        repairs.append("fully_preserved_identity_retention")
    elif retention_mode != "fully_preserved":
        raise ValueError(
            f"{identity_subject_tag} dual-anchor retention must be fully_preserved or the repairable attribute_transfer form; received {retention_mode}."
        )

    return identity_subject_tag, repairs


def _grounding_categories(description: Any) -> set[str]:
    categories: set[str] = set()
    for body in re.findall(r"\[\s*dg\s*:\s*([^\]]+)\]", str(description or ""), flags=re.IGNORECASE):
        categories.update(
            token.casefold()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", body)
        )
    return categories


def _normalize_relay_mode(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().casefold()).strip("_")
    aliases = {
        "": "previous_lane_tail",
        "previous_lane_tail": "previous_lane_tail",
        "previous_tail": "previous_lane_tail",
        "on": "previous_lane_tail",
        "enabled": "previous_lane_tail",
        "off": "off",
        "disabled": "off",
        "none": "off",
    }
    if normalized not in aliases:
        raise ValueError(
            "continuity_relay_mode must be Previous lane tail or Off."
        )
    return aliases[normalized]


def _identity_relay_policy(
    sections: dict[str, str],
    base_h3_prompt: str,
    identity_picture_count: Any,
    continuity_relay_mode: Any,
    reference_manifest: Any,
) -> dict[str, Any]:
    try:
        identity_count = int(identity_picture_count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("identity_picture_count must be 1 or 2.") from exc
    if identity_count not in {1, 2}:
        raise ValueError("identity_picture_count must be 1 or 2.")
    relay_mode = _normalize_relay_mode(continuity_relay_mode)
    identity_tags = [f"<Picture {index}>" for index in range(1, identity_count + 1)]
    continuity_tag = (
        f"<Picture {identity_count + 1}>"
        if relay_mode == "previous_lane_tail"
        else ""
    )

    manifest = str(reference_manifest or "").strip()
    manifest_definitions = _manifest_reference_definitions(manifest)
    if identity_count == 2 and not manifest:
        raise ValueError(
            "Dual identity pictures require reference_manifest so Picture 1 and Picture 2 roles can be validated before H3 runs."
        )
    if manifest:
        nonempty_lines = [line for line in manifest.splitlines() if line.strip()]
        if len(nonempty_lines) != len(manifest_definitions):
            raise ValueError(
                "reference_manifest must contain exactly one complete <Picture N>, <Video N>, or <Audio N> definition per non-empty line."
            )
        manifest_tags = [tag for tag, _description in manifest_definitions]
        if len(manifest_tags) != len(set(manifest_tags)):
            raise ValueError("reference_manifest contains a duplicate reference tag.")
        for kind, maximum in (("Picture", 9), ("Video", 3), ("Audio", 6)):
            ordinals = [
                int(match.group(1))
                for tag in manifest_tags
                if (match := re.fullmatch(rf"<{kind} (\d+)>", tag))
            ]
            if ordinals and sorted(ordinals) != list(range(1, max(ordinals) + 1)):
                raise ValueError(
                    f"reference_manifest {kind} tags must be consecutive from <{kind} 1>."
                )
            if ordinals and max(ordinals) > maximum:
                raise ValueError(f"reference_manifest exceeds the supported {kind} limit.")
        undefined = set(_reference_tags(manifest)) - set(manifest_tags)
        if undefined:
            raise ValueError(
                "reference_manifest mentions undefined reference tag(s): "
                + ", ".join(sorted(undefined))
                + "."
            )
        manifest_picture_tags = [tag for tag in manifest_tags if tag.startswith("<Picture ")]
        if manifest_picture_tags != identity_tags:
            raise ValueError(
                "reference_manifest Picture tags must exactly match the selected identity anchors "
                + ", ".join(identity_tags)
                + "; the relay Picture is injected only for later lanes."
            )
        by_tag = dict(manifest_definitions)
        for tag in identity_tags:
            categories = _grounding_categories(by_tag[tag])
            if not {"identity", "appearance"}.issubset(categories):
                raise ValueError(
                    f"{tag} must declare [dg:identity,appearance] in reference_manifest."
                )
        if identity_count == 2:
            picture_1_categories = _grounding_categories(by_tag["<Picture 1>"])
            if "composition" not in picture_1_categories:
                raise ValueError(
                    "<Picture 1> must include composition in its [dg:...] role because it remains the body, wardrobe, and target-composition authority."
                )
            picture_1_role = by_tag["<Picture 1>"].casefold()
            if not all(
                re.search(pattern, picture_1_role)
                for pattern in (
                    r"\b(?:body|build|proportions|silhouette)\b",
                    r"\b(?:wardrobe|clothing|outfit|costume)\b",
                    r"\b(?:composition|framing)\b",
                )
            ):
                raise ValueError(
                    "<Picture 1> must explicitly declare body/build, wardrobe/outfit, and composition/framing authority."
                )
            picture_2_categories = _grounding_categories(by_tag["<Picture 2>"])
            forbidden_picture_2_categories = picture_2_categories.intersection(
                {
                    "action",
                    "camera",
                    "composition",
                    "environment",
                    "lighting",
                    "motion",
                    "spatial",
                    "style",
                    "temporal",
                }
            )
            if forbidden_picture_2_categories:
                raise ValueError(
                    "<Picture 2> is identity evidence only and cannot declare panel-derived "
                    + ", ".join(sorted(forbidden_picture_2_categories))
                    + " roles."
                )
            picture_2_role = by_tag["<Picture 2>"].casefold()
            required_picture_2_role_patterns = (
                r"\b(?:multi[ -]?panel|contact[ -]?sheet|reference[ -]?sheet)\b",
                r"\b(?:same|one)\s+(?:subject|person|woman|performer|character)\b",
                r"\b(?:not|never|do\s+not)\b",
                r"\b(?:layout|grid)\b",
                r"\bseams?\b",
                r"\bbackgrounds?\b",
                r"\bpose\s+sequence\b",
                r"\bseparate\s+(?:people|persons|subjects|characters)\b",
            )
            if not all(
                re.search(pattern, picture_2_role)
                for pattern in required_picture_2_role_patterns
            ):
                raise ValueError(
                    "<Picture 2> must explicitly describe multi-panel identity evidence for the same Subject only and state that its layout/grid, seams, backgrounds, pose sequence, and panels-as-separate-people do not transfer."
                )

    identity_subject_tag = ""
    contract_repairs: list[str] = []
    if identity_count == 2:
        identity_subject_tag, contract_repairs = _dual_identity_subject_contract(
            sections, identity_tags
        )

    # Validate the effective prompt after deterministic host normalization.  A
    # repaired reciprocal citation is part of the prompt H3 will actually see.
    effective_prompt = _render_sections(_REF_FIELDS, sections)
    prompt_picture_tags = {
        tag for tag in _reference_tags(effective_prompt) if tag.startswith("<Picture ")
    }
    unexpected_prompt_tags = prompt_picture_tags - set(identity_tags)
    if unexpected_prompt_tags:
        raise ValueError(
            "base_h3_prompt contains Picture tag(s) outside the identity-anchor contract: "
            + ", ".join(sorted(unexpected_prompt_tags))
            + ". The relay Picture is reserved for deterministic later-lane injection."
        )
    missing_prompt_tags = set(identity_tags) - prompt_picture_tags
    if missing_prompt_tags:
        raise ValueError(
            "base_h3_prompt is missing selected identity Picture tag(s): "
            + ", ".join(sorted(missing_prompt_tags))
            + "."
        )

    return {
        "schema": IDENTITY_RELAY_POLICY_SCHEMA,
        "version": VERSION,
        "identity_picture_count": identity_count,
        "identity_picture_tags": identity_tags,
        "identity_subject_tag": identity_subject_tag,
        "subject_binding_validation": (
            "host_normalized" if contract_repairs else "strict"
        ),
        "contract_repairs": contract_repairs,
        "identity_picture_roles": {
            "<Picture 1>": "primary_body_wardrobe_target_composition_authority",
            **(
                {
                    "<Picture 2>": "same_subject_multi_panel_identity_evidence_only"
                }
                if identity_count == 2
                else {}
            ),
        },
        "continuity_relay_mode": relay_mode,
        "continuity_picture_tag": continuity_tag,
        "continuity_picture_role": (
            "previous_lane_pose_spatial_lighting_motion_only_not_identity"
            if continuity_tag
            else "off"
        ),
        "reference_manifest_sha256": _sha(manifest) if manifest else "",
        "manifest_validation": "strict" if manifest else "legacy_implicit_single_identity",
    }


def _shot_blocks(description: str) -> tuple[str, list[str]]:
    matches = list(_SHOT_RE.finditer(description))
    if not matches:
        body = description.strip()
        if not body:
            raise ValueError("The H3 detailed description has no renderable source material.")
        return "", [body]
    authored_numbers = [int(match.group(1)) for match in matches]
    expected_numbers = list(range(1, len(matches) + 1))
    if authored_numbers != expected_numbers:
        raise ValueError(
            "Native H3 shot blocks must be consecutively numbered from [Shot 1] "
            "without duplicates or gaps."
        )
    prefix = description[: matches[0].start()].strip()
    blocks = [match.group(2).strip() for match in matches]
    empty_numbers = [
        authored_numbers[index]
        for index, block in enumerate(blocks)
        if not block
    ]
    if empty_numbers:
        rendered = ", ".join(str(value) for value in empty_numbers)
        raise ValueError(
            f"Native H3 shot block(s) {rendered} have no renderable description."
        )
    return prefix, blocks


def _leading_shot_timestamp_seconds(block: str) -> float | None:
    """Return a native H3 shot's leading absolute timestamp, if present."""

    match = re.match(
        r"(?is)^\s*At\s+(?P<minutes>\d{2,}):(?P<seconds>\d{2})\.(?P<milliseconds>\d{3})\b",
        str(block or ""),
    )
    if match is None:
        return None
    seconds = int(match.group("seconds"))
    if seconds >= 60:
        return None
    return (
        int(match.group("minutes")) * 60.0
        + seconds
        + int(match.group("milliseconds")) / 1000.0
    )


def _embedded_timestamp_seconds(match: re.Match[str]) -> float:
    seconds = int(match.group("seconds"))
    if seconds >= 60:
        raise ValueError("H3 source timestamps must use MM:SS.mmm with SS below 60.")
    return (
        int(match.group("minutes")) * 60.0
        + seconds
        + int(match.group("milliseconds")) / 1000.0
    )


def _validate_source_timestamps(source_blocks: list[str], duration: float) -> None:
    """Reject timestamps outside the production clock before lane-local clipping."""

    for shot_index, block in enumerate(source_blocks, start=1):
        for match in _EMBEDDED_TIMESTAMP_RE.finditer(str(block or "")):
            absolute = _embedded_timestamp_seconds(match)
            if absolute < -1.0e-6 or absolute >= duration - 1.0e-6:
                raise ValueError(
                    f"A timestamp inside native [Shot {shot_index}] falls outside "
                    f"the {duration:g}s production clock."
                )


def _partition_native_shot_blocks(
    source_blocks: list[str],
    boundaries: list[tuple[float, str]],
    *,
    has_native_markers: bool,
) -> list[list[str]]:
    """Assign native ``[Shot]`` blocks to duration-bounded H3 render lanes.

    A render lane is one H3 invocation. It is not a camera-shot count: H3 may
    execute several native shot blocks during that invocation. The legacy
    one-block-per-lane path is retained exactly, while additional blocks are
    assigned by their validated absolute start instants.
    """

    lane_count = len(boundaries) - 1
    if lane_count < 1 or not source_blocks:
        raise ValueError("The H3 shot plan has no renderable source material.")
    if not has_native_markers:
        # Preserve the legacy compatibility path for an unlabelled description.
        return [[source_blocks[0]] for _ in range(lane_count)]
    if lane_count == 1:
        return [list(source_blocks)]
    if len(source_blocks) == lane_count:
        # This was the original planner contract. Keep its selection behavior
        # and resulting prompts byte-for-byte compatible.
        return [[block] for block in source_blocks]
    if len(source_blocks) < lane_count:
        raise ValueError(
            f"The validated H3 prompt has {len(source_blocks)} native shot blocks, but "
            f"the {lane_count} duration-bounded generation lanes each need source material."
        )

    starts: list[float] = [0.0]
    for index, block in enumerate(source_blocks[1:], start=2):
        timestamp = _leading_shot_timestamp_seconds(block)
        if timestamp is None:
            raise ValueError(
                f"Native [Shot {index}] needs a leading At MM:SS.mmm instant so it can "
                "be assigned to the correct duration-bounded H3 generation lane."
            )
        starts.append(timestamp)

    groups: list[list[str]] = [[] for _ in range(lane_count)]
    duration = boundaries[-1][0]
    for index, (block, start) in enumerate(zip(source_blocks, starts), start=1):
        if start < -1.0e-6 or start >= duration - 1.0e-6:
            raise ValueError(
                f"Native [Shot {index}] begins at {start:g}s, outside the {duration:g}s excerpt."
            )
        lane_index = lane_count - 1
        for candidate in range(lane_count - 1):
            if start < boundaries[candidate + 1][0] - 1.0e-6:
                lane_index = candidate
                break
        groups[lane_index].append(block)

    empty_lanes = [index + 1 for index, blocks in enumerate(groups) if not blocks]
    if empty_lanes:
        lanes = ", ".join(str(value) for value in empty_lanes)
        raise ValueError(
            "The native H3 shot timestamps leave generation lane(s) "
            f"{lanes} without source material; move a native shot boundary onto each "
            "planned generation-lane seam or reduce the duration segmentation."
        )
    return groups


def _carry_in_source_block(block: str, source_start: float, lane_start: float) -> str:
    """Keep an already-active native shot alive across a non-native lane seam."""

    directive = (
        "At this generation lane's local 00:00.000 opening, re-establish and continue "
        f"the same source shot that began at {source_start:g}s; the {lane_start:g}s "
        "generation boundary must not skip ahead to the next native shot before its "
        "rebased timestamp."
    )
    return f"{str(block or '').rstrip()} {directive}".strip()


def _partition_native_shot_blocks_with_carry(
    source_blocks: list[str],
    boundaries: list[tuple[float, str]],
    *,
    has_native_markers: bool,
    allow_non_native_carry: bool,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Partition source shots and preserve the shot active at a fallback seam.

    The strict/native-only paths retain the established byte-compatible partition.
    Recovery-only and duration-balanced fallbacks may place a render seam between
    authored native shot starts.  In that case the active source block is repeated
    as lane-local carry-in material, and later native timestamps remain exact after
    rebasing instead of being shifted early or silently omitted.
    """

    lane_count = len(boundaries) - 1
    if not allow_non_native_carry or not has_native_markers:
        groups = _partition_native_shot_blocks(
            source_blocks,
            boundaries,
            has_native_markers=has_native_markers,
        )
        return groups, [
            {
                "required": False,
                "source_native_shot_index": 0,
                "source_native_shot_start_seconds": 0.0,
            }
            for blocks in groups
        ]

    duration = boundaries[-1][0]
    starts = [0.0, *(value for value, _reason in _native_shot_points(source_blocks, duration))]
    groups: list[list[str]] = []
    carry_ins: list[dict[str, Any]] = []
    for lane_index in range(lane_count):
        lane_start = boundaries[lane_index][0]
        lane_end = boundaries[lane_index + 1][0]
        active_index = max(
            index for index, value in enumerate(starts) if value <= lane_start + 1.0e-6
        )
        selected_indices = [
            index
            for index in range(active_index, len(source_blocks))
            if starts[index] < lane_end - 1.0e-6
        ]
        if not selected_indices:
            raise ValueError(
                f"Generation lane {lane_index + 1} has no source shot active within "
                f"{lane_start:g}-{lane_end:g}s."
            )
        carry_required = starts[active_index] < lane_start - 1.0e-6
        blocks = [source_blocks[index] for index in selected_indices]
        if carry_required:
            blocks[0] = _carry_in_source_block(
                blocks[0], starts[active_index], lane_start
            )
        groups.append(blocks)
        carry_ins.append(
            {
                "required": carry_required,
                "source_native_shot_index": active_index + 1 if carry_required else 0,
                "source_native_shot_start_seconds": (
                    round(starts[active_index], 6) if carry_required else 0.0
                ),
            }
        )
    return groups, carry_ins


def _remove_leading_timestamp(text: str) -> str:
    return re.sub(
        r"(?is)^\s*(?:at\s+)?(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?\s*[,;:\-]?\s*",
        "",
        str(text or ""),
        count=1,
    ).strip()


def _normalize_leading_edit_transition(text: str) -> str:
    """Turn an edit-boundary cue into the opening frame of a local H3 shot.

    Each planned lane is rendered independently, so a source storyboard's
    leading ``hard cut`` is already represented by the lane boundary.  Leaving
    that edit verb in the local prompt can make H3 create a second cut inside
    what must be one continuous shot.  Only the leading boundary phrase is
    rewritten; later action and camera language remain untouched.
    """

    value = str(text or "").strip()
    patterns = (
        r"^(?:the\s+)?camera\s+cuts?\s+to\s+",
        r"^(?:a\s+)?(?:hard|smash|match)\s+cut\s+(?:to|reveals?)\s+",
        r"^cut\s+to\s+",
    )
    for pattern in patterns:
        if re.search(pattern, value, flags=re.IGNORECASE):
            return re.sub(pattern, "The shot opens on ", value, count=1, flags=re.IGNORECASE).strip()
    return value


def _dance_visible_vocal_cue(text: str) -> str | None:
    """Return an active visible-vocal cue that conflicts with Dance mode."""

    subject_source = (
        r"(?:(?:the|a|any|visible|lead)\s+){0,3}"
        r"(?:person|performer|subject|singer|dancer|woman|man|girl|boy|she|he|they)"
    )
    action_source = (
        r"(?:sings?|singing|speaks?|speaking|says?|saying|"
        r"lip[ -]?syncs?|lip[ -]?syncing|mouths?|mouthing)"
    )
    subject = re.compile(rf"\b{subject_source}\b", flags=re.IGNORECASE)
    action = re.compile(rf"\b{action_source}\b", flags=re.IGNORECASE)
    action_item = (
        rf"\b{action_source}\b(?:\s+(?:words?|the\s+lyrics))?"
        r"(?:\s+(?:visibly|audibly|actually|openly))?"
    )
    separator = r"(?:\s*,\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+)"
    prior_vocal_list = re.compile(
        rf"^(?:{action_item}{separator})*$",
        flags=re.IGNORECASE,
    )
    negator_patterns = tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            rf"\bno\s+{subject_source}\s+(?:(?:is|are)\s+)?",
            rf"\b{subject_source}\s+(?:(?:does\s+not|do\s+not|doesn't|don't|never|is\s+not|are\s+not|was\s+not|were\s+not)\s+(?:(?:visibly|audibly|actually|openly|ever)\s+)*|(?:avoid(?:s|ing)?|refrain(?:s|ing)?\s+from)\s+)",
            r"\b(?:(?:does\s+not|do\s+not|doesn't|don't|never|is\s+not|are\s+not|was\s+not|were\s+not)\s+(?:(?:visibly|audibly|actually|openly|ever)\s+)*|(?:avoid(?:s|ing)?|refrain(?:s|ing)?\s+from)\s+)",
            rf"\bwithout\s+(?:{subject_source}\s+)?",
        )
    )
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", str(text or "")):
        for match in action.finditer(sentence):
            prefix = sentence[: match.start()]
            negator_ends = [
                negator.end()
                for pattern in negator_patterns
                for negator in pattern.finditer(prefix)
            ]
            if any(
                prior_vocal_list.fullmatch(prefix[end:]) is not None
                for end in negator_ends
            ):
                continue
            window_start = max(0, match.start() - 96)
            window_end = min(len(sentence), match.end() + 48)
            window = sentence[window_start:window_end]
            if not subject.search(window):
                continue
            return match.group(0)
    return None


def _timed_lyrics_fallback(
    warning: str,
    *,
    analysis_status: str = "missing",
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    messages = list(warnings or [])
    if warning and warning not in messages:
        messages.append(warning)
    return {
        "state": "natural_fallback",
        "analysis_status": analysis_status,
        "timing_ready": False,
        "events": [],
        "vocal_intervals": [],
        "instrumental_intervals": [],
        "warnings": messages,
    }


def _timed_report_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _timed_report_intervals(
    report: Mapping[str, Any], key: str, duration: float
) -> tuple[list[dict[str, float]], str | None]:
    source = report.get(key, [])
    if source is None:
        source = []
    if not isinstance(source, list):
        return [], f"Timed-lyrics {key} must be an array."
    intervals: list[dict[str, float]] = []
    for index, item in enumerate(source):
        if not isinstance(item, Mapping):
            return [], f"Timed-lyrics {key}[{index}] is not an object."
        start = _timed_report_number(item.get("start_seconds"))
        end = _timed_report_number(item.get("end_seconds"))
        if (
            start is None
            or end is None
            or start < -0.05
            or end <= start
            or end > duration + 0.05
        ):
            return [], f"Timed-lyrics {key}[{index}] has an invalid excerpt-relative interval."
        intervals.append(
            {
                "start_seconds": round(max(0.0, start), 6),
                "end_seconds": round(min(duration, end), 6),
            }
        )
    return intervals, None


def _timed_lyrics_contract(
    timed_lyrics_report_json: Any,
    project: Mapping[str, Any],
    lyrics: Any,
    excerpt_start: float,
    duration: float,
) -> dict[str, Any]:
    """Validate optional alignment evidence without making generation brittle.

    Lyrics are lexical evidence only after a hash-locked, excerpt-locked timing
    report validates.  Any stale, weak, missing, or malformed report deliberately
    returns Natural/audio-led behavior instead of guessing a lyric schedule.
    """

    source = str(timed_lyrics_report_json or "").strip()
    if not source:
        return _timed_lyrics_fallback(
            "Timed lyrics are missing; using Natural / audio-led sync."
        )
    try:
        report = json.loads(source)
    except (TypeError, ValueError, json.JSONDecodeError):
        return _timed_lyrics_fallback(
            "Timed lyrics are not valid JSON; using Natural / audio-led sync.",
            analysis_status="invalid",
        )
    if not isinstance(report, Mapping):
        return _timed_lyrics_fallback(
            "Timed lyrics are not a JSON object; using Natural / audio-led sync.",
            analysis_status="invalid",
        )
    analysis_status = str(report.get("analysis_status", "missing") or "missing")
    if (
        report.get("schema") != TIMED_LYRICS_REPORT_SCHEMA
        or report.get("version") != TIMED_LYRICS_REPORT_VERSION
    ):
        return _timed_lyrics_fallback(
            "Timed lyrics use an unsupported schema or version; using Natural / audio-led sync.",
            analysis_status=analysis_status,
        )
    if report.get("timing_ready") is not True or analysis_status != "timing_ready":
        report_warnings = report.get("warnings")
        warnings = (
            [str(item) for item in report_warnings if str(item).strip()]
            if isinstance(report_warnings, list)
            else []
        )
        return _timed_lyrics_fallback(
            "Timed lyric alignment is not ready; using Natural / audio-led sync.",
            analysis_status=analysis_status,
            warnings=warnings,
        )
    audio_lock = project.get("audio_lock")
    if not isinstance(audio_lock, Mapping):
        return _timed_lyrics_fallback(
            "Project audio-lock evidence is unavailable; using Natural / audio-led sync.",
            analysis_status=analysis_status,
        )
    expected_audio_hash = str(audio_lock.get("waveform_sha256", "")).strip().lower()
    expected_lyrics_hash = str(audio_lock.get("lyrics_sha256", "")).strip().lower()
    report_audio_hash = str(report.get("master_audio_sha256", "")).strip().lower()
    report_lyrics_hash = str(report.get("lyrics_sha256", "")).strip().lower()
    if report_audio_hash != expected_audio_hash:
        return _timed_lyrics_fallback(
            "Timed lyrics belong to a different master audio file; using Natural / audio-led sync.",
            analysis_status=analysis_status,
        )
    if report_lyrics_hash != expected_lyrics_hash or _sha(str(lyrics or "")) != expected_lyrics_hash:
        return _timed_lyrics_fallback(
            "Timed lyrics are stale for the selected authored lyrics; using Natural / audio-led sync.",
            analysis_status=analysis_status,
        )
    excerpt = report.get("excerpt")
    if not isinstance(excerpt, Mapping):
        return _timed_lyrics_fallback(
            "Timed lyrics are missing excerpt timing; using Natural / audio-led sync.",
            analysis_status=analysis_status,
        )
    report_start = _timed_report_number(excerpt.get("start_seconds"))
    report_duration = _timed_report_number(excerpt.get("duration_seconds"))
    report_end = _timed_report_number(excerpt.get("end_seconds"))
    if (
        report_start is None
        or report_duration is None
        or abs(report_start - excerpt_start) > 0.01
        or abs(report_duration - duration) > 0.01
        or (
            report_end is not None
            and abs(report_end - (excerpt_start + duration)) > 0.01
        )
    ):
        return _timed_lyrics_fallback(
            "Timed lyrics target a different excerpt clock; using Natural / audio-led sync.",
            analysis_status=analysis_status,
        )
    minimum_confidence = _timed_report_number(
        report.get("minimum_alignment_confidence", 0.0)
    )
    if minimum_confidence is None or not 0.0 <= minimum_confidence <= 1.0:
        return _timed_lyrics_fallback(
            "Timed lyrics have an invalid confidence threshold; using Natural / audio-led sync.",
            analysis_status=analysis_status,
        )
    raw_events = report.get("events")
    if not isinstance(raw_events, list):
        return _timed_lyrics_fallback(
            "Timed lyrics events are missing or invalid; using Natural / audio-led sync.",
            analysis_status=analysis_status,
        )
    events: list[dict[str, Any]] = []
    weak_event_count = 0
    for index, item in enumerate(raw_events):
        if not isinstance(item, Mapping):
            return _timed_lyrics_fallback(
                f"Timed lyric event {index + 1} is invalid; using Natural / audio-led sync.",
                analysis_status=analysis_status,
            )
        start = _timed_report_number(item.get("start_seconds"))
        end = _timed_report_number(item.get("end_seconds"))
        confidence = _timed_report_number(item.get("confidence"))
        authored_source = item.get("authored_lines", [])
        authored_lines = (
            [re.sub(r"\s+", " ", str(line)).strip() for line in authored_source]
            if isinstance(authored_source, list)
            else []
        )
        authored_lines = [line for line in authored_lines if line]
        supplied_text = re.sub(
            r"\s+",
            " ",
            str(item.get("text", "") or ""),
        ).strip()
        transcript = re.sub(
            r"\s+", " ", str(item.get("transcript", "") or "")
        ).strip()
        text = " / ".join(authored_lines) or supplied_text or transcript
        if (
            start is None
            or end is None
            or confidence is None
            or not 0.0 <= confidence <= 1.0
            or start < -0.05
            or end <= start
            or end > duration + 0.05
            or not text
        ):
            return _timed_lyrics_fallback(
                f"Timed lyric event {index + 1} has invalid timing, confidence, or text; using Natural / audio-led sync.",
                analysis_status=analysis_status,
            )
        if confidence + 1.0e-9 < minimum_confidence:
            weak_event_count += 1
            continue
        events.append(
            {
                "start_seconds": round(max(0.0, start), 6),
                "end_seconds": round(min(duration, end), 6),
                "text": text,
                "authored_lines": authored_lines,
                "transcript": transcript,
                "confidence": round(confidence, 6),
            }
        )
    vocal_intervals, interval_error = _timed_report_intervals(
        report, "vocal_intervals", duration
    )
    if interval_error:
        return _timed_lyrics_fallback(
            interval_error + " Using Natural / audio-led sync.",
            analysis_status=analysis_status,
        )
    instrumental_intervals, interval_error = _timed_report_intervals(
        report, "instrumental_intervals", duration
    )
    if interval_error:
        return _timed_lyrics_fallback(
            interval_error + " Using Natural / audio-led sync.",
            analysis_status=analysis_status,
        )
    warnings = []
    if weak_event_count:
        warnings.append(
            f"Ignored {weak_event_count} lyric event{'s' if weak_event_count != 1 else ''} below the report confidence threshold."
        )
    if not events and not instrumental_intervals:
        return _timed_lyrics_fallback(
            "Timed lyric evidence is too weak to schedule lexical hints; using Natural / audio-led sync.",
            analysis_status=analysis_status,
            warnings=warnings,
        )
    return {
        "state": "timing_ready",
        "analysis_status": analysis_status,
        "timing_ready": True,
        "events": events,
        "vocal_intervals": vocal_intervals,
        "instrumental_intervals": instrumental_intervals,
        "minimum_alignment_confidence": round(minimum_confidence, 6),
        "warnings": warnings,
    }


def _lane_timed_events(
    events: list[dict[str, Any]], lane_start: float, lane_end: float
) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for event in events:
        source_start = float(event["start_seconds"])
        source_end = float(event["end_seconds"])
        overlap_start = max(lane_start, source_start)
        overlap_end = min(lane_end, source_end)
        if overlap_end <= overlap_start + 1.0e-9:
            continue
        local_start = overlap_start - lane_start
        local_end = overlap_end - lane_start
        mapped.append(
            {
                "start_seconds": round(local_start, 6),
                "end_seconds": round(local_end, 6),
                "lane_local_start_seconds": round(local_start, 6),
                "lane_local_end_seconds": round(local_end, 6),
                "excerpt_relative_start_seconds": round(source_start, 6),
                "excerpt_relative_end_seconds": round(source_end, 6),
                "text": str(event["text"]),
                "authored_lines": list(event["authored_lines"]),
                "transcript": str(event.get("transcript", "")),
                "confidence": float(event["confidence"]),
                "clipped_at_lane_start": overlap_start > source_start + 1.0e-9,
                "clipped_at_lane_end": overlap_end < source_end - 1.0e-9,
            }
        )
    return mapped


def _intervals_cover_lane(
    intervals: list[dict[str, float]], lane_start: float, lane_end: float
) -> bool:
    cursor = lane_start
    for interval in sorted(intervals, key=lambda item: item["start_seconds"]):
        start = max(lane_start, float(interval["start_seconds"]))
        end = min(lane_end, float(interval["end_seconds"]))
        if end <= start + 1.0e-9:
            continue
        if start > cursor + 0.05:
            return False
        cursor = max(cursor, end)
        if cursor >= lane_end - 0.05:
            return True
    return cursor >= lane_end - 0.05


def _timed_lyrics_prompt_clock(seconds: float) -> str:
    milliseconds_total = int(round(max(0.0, seconds) * 1000.0))
    minutes, remainder = divmod(milliseconds_total, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _continuity_sections(
    mode: str,
    sections: dict[str, str],
    lane: int,
    identity_picture_tags: list[str] | None = None,
    continuity_picture_tag: str = "<Picture 2>",
) -> None:
    if mode != "ref2va" or lane <= 1 or not continuity_picture_tag:
        return
    identity_tags = list(identity_picture_tags or ["<Picture 1>"])
    continuity_definition = (
        f"{continuity_picture_tag}: [reference] retained final frame of the immediately previous generated segment; "
        "use only for pose, spatial, lighting, and motion continuity at this segment opening."
    )
    if continuity_picture_tag not in sections["subject_definitions"]:
        sections["subject_definitions"] = (
            sections["subject_definitions"].rstrip() + "\n" + continuity_definition
        )
    if identity_tags == ["<Picture 1>"] and continuity_picture_tag == "<Picture 2>":
        # Preserve the original single-anchor prompt byte-for-byte.
        retention = (
            "<Picture 2>: weak_reference - Use the prior segment's retained final frame only for "
            "opening pose, spatial, lighting, and motion continuity; <Picture 1> remains the "
            "identity and art-direction authority."
        )
    else:
        retention = (
            f"{continuity_picture_tag}: weak_reference - Use the prior segment's retained final frame only for "
            "opening pose, spatial, lighting, and motion continuity; it is not an identity authority. "
            f"{identity_tags[0]} remains the body, wardrobe, and target-composition authority, while "
            f"{' and '.join(identity_tags)} jointly identify the same performer."
        )
    if continuity_picture_tag not in sections["retention_analysis"]:
        sections["retention_analysis"] = sections["retention_analysis"].rstrip() + "\n" + retention


def _segment_prompt(
    mode: str,
    fields: tuple[str, ...],
    source_sections: Mapping[str, str],
    prefix: str,
    blocks: list[str],
    lane: int,
    lane_count: int,
    segment_start: float,
    segment_duration: float,
    performance_mode: str,
    timed_lyrics_events: list[dict[str, Any]],
    lyric_timing_state: str,
    identity_picture_tags: list[str] | None = None,
    continuity_picture_tag: str = "<Picture 2>",
    identity_subject_tag: str = "",
) -> str:
    sections = {key: str(value) for key, value in source_sections.items()}
    detail_field = "detailed_description" if mode == "ref2va" else "integrated_multimodal_description"
    if not blocks:
        raise ValueError("An H3 generation lane cannot have zero native shot blocks.")
    bodies: list[str] = []
    for index, block in enumerate(blocks):
        source = str(block or "").strip()
        if index == 0:
            source = _normalize_leading_edit_transition(_remove_leading_timestamp(source))
        bodies.append(
            _rebase_embedded_timestamps(source, segment_start, segment_duration)
        )
    if performance_mode == "Lyrics + lip sync" and timed_lyrics_events:
        hints = "; ".join(
            f'{_timed_lyrics_prompt_clock(float(event["start_seconds"]))}-'
            f'{_timed_lyrics_prompt_clock(float(event["end_seconds"]))} '
            f'"{str(event["text"])}" (confidence {float(event["confidence"]):.2f})'
            for event in timed_lyrics_events
        )
        performance_directive = (
            "<Audio 1> alone is the timing authority. Use these audio-gated lexical hints only "
            "inside their measured local vocal intervals: "
            f"{hints}. These lexical hints are not a timing schedule: never create facial or mouth "
            "articulation when the matching vocal is not audible in <Audio 1>, and do not compress, "
            "repeat, invent, or pantomime words."
        )
    elif performance_mode == "Lyrics + lip sync" and lyric_timing_state == "confirmed_instrumental":
        performance_directive = (
            "<Audio 1> confirms that this entire generated segment is instrumental. No visible "
            "person sings, speaks, mouths words, or lip-syncs in this segment."
        )
    elif performance_mode == "Dance / music sync":
        performance_directive = (
            "<Audio 1> is the timing authority for dance, body motion, scene accents, and musical "
            "phrasing only. No visible person sings, speaks, mouths words, or lip-syncs."
        )
    elif performance_mode == "Natural / audio-led sync" or lyric_timing_state == "natural_audio_led":
        performance_directive = (
            "<Audio 1> alone decides whether and when visible facial or mouth articulation occurs. "
            "Do not invent or pantomime words, do not schedule lip movement from written lyrics, "
            "and do not force singing or blanket closed-mouth behavior; follow only vocal evidence "
            "actually audible in <Audio 1>."
        )
    else:  # Defensive: callers validate the enum before reaching this helper.
        performance_directive = ""
    if len(blocks) == 1:
        segment_note = f"This is generated segment {lane} of {lane_count}; it is one locally continuous H3 shot."
        bodies[0] = f"{bodies[0].rstrip()} {performance_directive}".strip()
        scoped_directive = ""
    else:
        segment_note = (
            f"This is generated segment {lane} of {lane_count}; it contains "
            f"{len(blocks)} consecutively numbered native H3 shots on one local clock."
        )
        scoped_directive = (
            f"{segment_note} Across all {len(blocks)} native shots in this generated segment, "
            f"{performance_directive}"
        )
    rendered_blocks = []
    for index, body in enumerate(bodies, start=1):
        note = f"{segment_note} " if len(blocks) == 1 and index == 1 else ""
        rendered_blocks.append(f"[Shot {index}] {note}{body}".strip())
    local_description = "\n".join(rendered_blocks)
    if scoped_directive:
        local_description = f"{scoped_directive}\n{local_description}"
    identity_tags = list(identity_picture_tags or ["<Picture 1>"])
    reference_directives: list[str] = []
    if len(identity_tags) > 1:
        subject = identity_subject_tag or "the same semantic Subject"
        reference_directives.append(
            f"{identity_tags[1]} is multi-panel identity evidence for the same {subject} only. "
            "Do not reproduce its panel layout, grid, seams, source backgrounds, or pose sequence, "
            "and do not interpret its panels as separate people. "
            f"{identity_tags[0]} remains the body, wardrobe, and target-composition authority."
        )
    if lane > 1 and continuity_picture_tag:
        reference_directives.append(
            f"At this segment opening, {continuity_picture_tag} supplies only pose, spatial, lighting, "
            "and motion continuity from the previous lane; it is not an identity authority."
        )
    if reference_directives:
        local_description = "\n".join([*reference_directives, local_description])
    if prefix:
        local_description = f"{prefix}\n{local_description}"
    sections[detail_field] = local_description
    _continuity_sections(
        mode,
        sections,
        lane,
        identity_tags,
        continuity_picture_tag,
    )
    return _render_sections(fields, sections)


def _recovery_points(report: Mapping[str, Any], duration: float) -> list[tuple[float, str]]:
    excerpt = report.get("selected_excerpt")
    if not isinstance(excerpt, Mapping):
        raise ValueError("measured_audio_report_json is missing selected_excerpt evidence.")
    points: list[tuple[float, str]] = []
    intervals = excerpt.get("low_density_visual_recovery_intervals")
    if isinstance(intervals, list):
        for item in intervals:
            if not isinstance(item, Mapping):
                continue
            start = float(item.get("relative_start_seconds", -1))
            end = float(item.get("relative_end_seconds", -1))
            if 0 <= start < end <= duration + 0.05:
                points.extend(((start, "recovery_interval_edge"), ((start + end) / 2.0, "recovery_interval_midpoint"), (end, "recovery_interval_edge")))
    timeline = excerpt.get("timeline")
    entries = timeline.get("entries") if isinstance(timeline, Mapping) else None
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, Mapping) or not item.get("low_density_visual_recovery_proxy"):
                continue
            start = float(item.get("relative_start_seconds", -1))
            end = float(item.get("relative_end_seconds", -1))
            if 0 <= start < end <= duration + 0.05:
                points.append(((start + end) / 2.0, "recovery_timeline_window"))
    unique: dict[float, str] = {}
    for value, reason in points:
        if 0.0 < value < duration:
            unique[round(value, 6)] = reason
    return sorted(unique.items())


def _recovery_reason_at(
    report: Mapping[str, Any],
    duration: float,
    instant: float,
    recovery_points: list[tuple[float, str]] | None = None,
) -> str | None:
    """Return the literal measured-recovery evidence covering one instant."""

    excerpt = report.get("selected_excerpt")
    intervals = excerpt.get("low_density_visual_recovery_intervals") if isinstance(excerpt, Mapping) else None
    timeline = excerpt.get("timeline") if isinstance(excerpt, Mapping) else None
    entries = timeline.get("entries") if isinstance(timeline, Mapping) else None
    if isinstance(intervals, list):
        for item in intervals:
            if not isinstance(item, Mapping):
                continue
            start = float(item.get("relative_start_seconds", -1))
            end = float(item.get("relative_end_seconds", -1))
            if 0 <= start <= instant <= end <= duration + 0.05:
                return "recovery_interval"
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, Mapping) or not item.get("low_density_visual_recovery_proxy"):
                continue
            start = float(item.get("relative_start_seconds", -1))
            end = float(item.get("relative_end_seconds", -1))
            if 0 <= start <= instant <= end <= duration + 0.05:
                return "recovery_timeline_window"
    nearby = [
        (abs(point - instant), reason)
        for point, reason in (recovery_points or [])
        if abs(point - instant) <= 0.05 + 1.0e-9
    ]
    return min(nearby)[1] if nearby else None


def _native_shot_points(
    source_blocks: list[str], duration: float
) -> list[tuple[float, str]]:
    """Validate and return every native inter-shot start instant."""

    native: dict[float, str] = {}
    aligned: dict[float, str] = {}
    first_start = _leading_shot_timestamp_seconds(source_blocks[0])
    if first_start is not None and abs(first_start) > 1.0e-6:
        raise ValueError("Native [Shot 1] must open at 00:00.000 on the production clock.")
    previous_start = 0.0
    for index, block in enumerate(source_blocks[1:], start=2):
        native_start = _leading_shot_timestamp_seconds(block)
        if native_start is None:
            raise ValueError(
                f"Native [Shot {index}] needs a leading At MM:SS.mmm instant before "
                "a measured inter-lane cut can be planned."
            )
        if not 0.0 < native_start < duration:
            raise ValueError(
                f"Native [Shot {index}] begins at {native_start:g}s, outside the {duration:g}s excerpt."
            )
        if native_start <= previous_start + 1.0e-6:
            raise ValueError("Native H3 shot timestamps must be strictly increasing.")
        previous_start = native_start
        native[round(native_start, 6)] = "native_h3_shot_boundary"
    return sorted(native.items())


def _native_recovery_points(
    source_blocks: list[str],
    recovery_points: list[tuple[float, str]],
    report: Mapping[str, Any],
    duration: float,
) -> list[tuple[float, str]]:
    """Return native cut instants that also have measured recovery evidence."""

    aligned: dict[float, str] = {}
    for native_start, _native_reason in _native_shot_points(source_blocks, duration):
        reason = _recovery_reason_at(
            report, duration, native_start, recovery_points
        )
        if reason is None:
            continue
        aligned[round(native_start, 6)] = (
            f"{reason}+native_h3_shot_boundary"
        )
    return sorted(aligned.items())


def _merge_boundary_points(
    native_points: list[tuple[float, str]],
    recovery_points: list[tuple[float, str]],
    native_recovery_points: list[tuple[float, str]],
) -> list[tuple[float, str]]:
    """Build a deterministic union while retaining the strongest provenance."""

    merged: dict[float, str] = {}
    for value, reason in recovery_points:
        merged[round(value, 6)] = reason
    for value, reason in native_points:
        key = round(value, 6)
        merged[key] = (
            f"{merged[key]}+native_h3_shot_boundary"
            if key in merged
            else reason
        )
    for value, reason in native_recovery_points:
        merged[round(value, 6)] = reason
    return sorted(merged.items())


def _boundary_reason_penalty(reason: str) -> int:
    has_native = "native_h3_shot_boundary" in reason
    has_recovery = "recovery_" in reason
    if has_native and has_recovery:
        return 0
    if has_native:
        return 1
    if has_recovery:
        return 2
    return 3


def _choose_boundaries(
    duration: float,
    count: int,
    minimum: float,
    maximum: float,
    points: list[tuple[float, str]],
    *,
    prefer_evidence: bool = False,
) -> list[tuple[float, str]]:
    if count == 1:
        return [(0.0, "excerpt_start"), (duration, "excerpt_end")]
    best: tuple[tuple[Any, ...], tuple[tuple[float, str], ...]] | None = None
    for candidate_cuts in itertools.combinations(points, count - 1):
        values = (0.0, *(item[0] for item in candidate_cuts), duration)
        segment_lengths = [values[index + 1] - values[index] for index in range(count)]
        if any(length < minimum - 1e-6 or length > maximum + 1e-6 for length in segment_lengths):
            continue
        distance = sum(
            abs(candidate_cuts[index - 1][0] - duration * index / count)
            for index in range(1, count)
        )
        evidence_penalty = (
            sum(_boundary_reason_penalty(item[1]) for item in candidate_cuts)
            if prefer_evidence
            else 0
        )
        rank = (
            evidence_penalty,
            round(distance, 9),
            tuple(item[0] for item in candidate_cuts),
        )
        if best is None or rank < best[0]:
            best = (rank, candidate_cuts)
    if best is None:
        raise ValueError(
            "No complete candidate cut sequence satisfies every H3 generation-lane duration limit."
        )
    return [(0.0, "excerpt_start"), *best[1], (duration, "excerpt_end")]


def _duration_balanced_boundaries(
    duration: float, count: int, minimum: float, maximum: float
) -> list[tuple[float, str]]:
    """Return the always-feasible final fallback after hard-limit validation."""

    boundaries = [
        (duration * index / count, "duration_limit_fallback")
        for index in range(count + 1)
    ]
    boundaries[0] = (0.0, "excerpt_start")
    boundaries[-1] = (duration, "excerpt_end")
    lengths = [
        boundaries[index + 1][0] - boundaries[index][0]
        for index in range(count)
    ]
    if any(length < minimum - 1.0e-6 or length > maximum + 1.0e-6 for length in lengths):
        raise ValueError(
            "The requested duration cannot be divided within the hard H3 generation-lane limits."
        )
    return boundaries


def _boundary_evidence_records(
    boundaries: list[tuple[float, str]],
    *,
    strategy: str,
    native_points: list[tuple[float, str]],
    recovery_points: list[tuple[float, str]],
    report: Mapping[str, Any],
    duration: float,
) -> tuple[list[tuple[float, str]], list[dict[str, Any]]]:
    """Separate how a cut was selected from evidence that happens to cover it."""

    native_values = [value for value, _reason in native_points]
    enriched = [boundaries[0]]
    records: list[dict[str, Any]] = []
    for value, candidate_reason in boundaries[1:-1]:
        is_native = any(abs(value - point) <= 1.0e-6 for point in native_values)
        recovery_reason = _recovery_reason_at(
            report, duration, value, recovery_points
        )
        has_recovery = recovery_reason is not None
        if strategy == "duration_balanced_fallback":
            selection_origin = "duration_balanced"
        elif "native" in candidate_reason and "recovery" in candidate_reason:
            selection_origin = "native_measured"
        elif "native" in candidate_reason:
            selection_origin = "native"
        elif "recovery" in candidate_reason:
            selection_origin = "measured_recovery"
        else:
            selection_origin = strategy
        provenance: list[str] = []
        if selection_origin == "duration_balanced":
            provenance.append("duration_limit_fallback")
        if has_recovery:
            provenance.append(str(recovery_reason))
        if is_native:
            provenance.append("native_h3_shot_boundary")
        evidence = "+".join(provenance) or str(candidate_reason)
        enriched.append((value, evidence))
        records.append(
            {
                "seconds": round(value, 6),
                "selection_origin": selection_origin,
                "provenance": evidence,
                "native_shot_boundary": is_native,
                "measured_low_density_recovery": has_recovery,
            }
        )
    enriched.append(boundaries[-1])
    return enriched, records


def _rebase_embedded_timestamps(text: str, segment_start: float, segment_duration: float) -> str:
    """Slice timestamped sub-cues onto one lane's local clock.

    A non-native seam can pass through the middle of an authored source shot.
    The repeated source block therefore contains elapsed and future cues as well
    as cues that belong to this lane.  Keep only the latest elapsed cue as
    explicit opening-state history, rebase in-lane cues, and defer future cues
    to the carried copy rendered by the later lane.
    """

    raw = str(text or "")
    matches = list(_EMBEDDED_TIMESTAMP_RE.finditer(raw))
    if not matches:
        return raw

    def local_marker(absolute: float) -> str:
        local = max(0.0, absolute - segment_start)
        minutes = int(local // 60)
        seconds = local - minutes * 60
        whole_seconds = int(seconds)
        milliseconds = int(round((seconds - whole_seconds) * 1000.0))
        if milliseconds == 1000:
            whole_seconds += 1
            milliseconds = 0
        if whole_seconds == 60:
            minutes += 1
            whole_seconds = 0
        return f"At {minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"

    prefix = raw[: matches[0].start()].rstrip()
    latest_elapsed_context = ""
    in_lane_chunks: list[str] = []
    lane_end = segment_start + segment_duration
    for index, match in enumerate(matches):
        chunk_end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        suffix = raw[match.end() : chunk_end]
        absolute = _embedded_timestamp_seconds(match)
        if absolute < segment_start - 0.001:
            context = suffix.strip().lstrip(",;:- ").strip()
            if context:
                latest_elapsed_context = context
            continue
        if absolute >= lane_end - 1.0e-6:
            continue
        in_lane_chunks.append(f"{local_marker(absolute)}{suffix}".strip())

    rendered: list[str] = []
    if prefix:
        rendered.append(prefix)
    if latest_elapsed_context:
        rendered.append(
            "Opening-state history from before this generation lane (do not re-enact "
            f"as a new timed action): {latest_elapsed_context}"
        )
    rendered.extend(chunk for chunk in in_lane_chunks if chunk)
    return " ".join(rendered).strip()


def _validate_music_report(
    report: Mapping[str, Any],
    project: Mapping[str, Any],
    excerpt_start: float,
    duration: float,
) -> None:
    if report.get("schema") != MUSIC_REPORT_SCHEMA or report.get("version") != MUSIC_REPORT_VERSION:
        raise ValueError("measured_audio_report_json is not a supported music audition report.")
    if report.get("ready") is not True:
        raise ValueError("measured_audio_report_json is not ready for generation.")
    selected_hash = str(report.get("selected_audio_sha256", "")).strip().lower()
    if not _SHA256_RE.fullmatch(selected_hash):
        raise ValueError("measured_audio_report_json is missing the selected waveform SHA-256.")
    settings = report.get("settings")
    excerpt = report.get("selected_excerpt")
    if not isinstance(settings, Mapping) or not isinstance(excerpt, Mapping):
        raise ValueError("measured_audio_report_json is missing selected excerpt evidence.")
    report_duration = _positive_finite(
        excerpt.get("duration_seconds"), "measured selected excerpt duration"
    )
    requested_duration = _positive_finite(
        settings.get("excerpt_duration_seconds"), "measured requested excerpt duration"
    )
    report_start = _positive_finite(
        excerpt.get("start_seconds"), "measured selected excerpt start", allow_zero=True
    )
    if abs(report_duration - duration) > 0.05 or abs(requested_duration - duration) > 0.05:
        raise ValueError("Measured audio report and requested excerpt durations disagree.")
    if abs(report_start - excerpt_start) > 0.05:
        raise ValueError("Measured audio report and requested excerpt starts disagree.")
    if project.get("schema") != PROJECT_SCHEMA or project.get("version") != VERSION:
        raise ValueError("project_manifest_json has an unsupported schema or version.")
    audio_lock = project.get("audio_lock")
    if not isinstance(audio_lock, Mapping):
        raise ValueError("project_manifest_json is missing its audio lock.")
    if str(audio_lock.get("waveform_sha256", "")).strip().lower() != selected_hash:
        raise ValueError("Measured audio SHA-256 does not match the Project Master audio lock.")
    if abs(float(audio_lock.get("excerpt_start_seconds", -1)) - excerpt_start) > 0.05:
        raise ValueError("Project Master and measured excerpt starts disagree.")
    if abs(float(audio_lock.get("excerpt_duration_seconds", -1)) - duration) > 0.05:
        raise ValueError("Project Master and measured excerpt durations disagree.")
    if abs(float(project.get("production_duration_seconds", -1)) - duration) > 0.05:
        raise ValueError("Project Master production duration and shot-plan duration disagree.")


class DiffusionGemmaProjectMasterContract:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "creative_brief": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "production_duration_seconds": ("FLOAT", {"default": 15.0, "min": 0.1, "max": MAX_PROJECT_SECONDS, "step": 0.1}),
                "aspect_ratio": (list(ASPECT_RATIOS), {"default": "9:16"}),
                "master_audio_sha256": ("STRING", {"default": "", "forceInput": True}),
                "lyrics": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "excerpt_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
                "excerpt_duration_seconds": ("FLOAT", {"default": 15.0, "min": 0.1, "step": 0.01}),
                "generation_model": ("STRING", {"default": "MiniMax H3 Ref2VA", "multiline": False}),
                "deliverables_json": ("STRING", {"default": '{"primary":"9:16 master","adaptations":["16:9","1:1"]}', "multiline": True}),
                "max_h3_shot_seconds": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "min": 5.0,
                        "max": 15.0,
                        "step": 0.5,
                        "tooltip": "Legacy name: this is the hard maximum duration of one H3 generation pass. It does not limit native [Shot N] blocks inside that pass. If preferred native/recovery seams cannot satisfy the cap, the planner uses deterministic duration-balanced seams and carries an active source shot into the next lane.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING", "STRING", "STRING", "BOOLEAN", "FLOAT")
    # Legacy socket name retained: this count is ceil(duration / lane ceiling),
    # not the number of native `[Shot N]` blocks inside an H3 prompt.
    RETURN_NAMES = ("project_manifest_json", "project_id", "master_aspect_ratio", "recommended_h3_shot_count", "brief_sha256", "lyrics_sha256", "status", "ready", "max_h3_shot_seconds")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "Freeze a selected song, brief, duration, format, generation-lane budget, and delivery intent into one versioned production contract. Native H3 [Shot] count remains prompt-controlled."

    def build(self, creative_brief: str, production_duration_seconds: float, aspect_ratio: str, master_audio_sha256: str, lyrics: str, excerpt_start_seconds: float, excerpt_duration_seconds: float, generation_model: str, deliverables_json: str, max_h3_shot_seconds: float):
        brief = str(creative_brief or "").strip()
        if not brief:
            raise ValueError("creative_brief must not be empty.")
        duration = _positive_finite(production_duration_seconds, "production_duration_seconds")
        if duration > MAX_PROJECT_SECONDS:
            raise ValueError(f"This four-lane H3 contract supports at most {MAX_PROJECT_SECONDS:.0f} seconds.")
        excerpt_start = _positive_finite(excerpt_start_seconds, "excerpt_start_seconds", allow_zero=True)
        excerpt_duration = _positive_finite(excerpt_duration_seconds, "excerpt_duration_seconds")
        if abs(excerpt_duration - duration) > 0.05:
            raise ValueError("excerpt_duration_seconds must match production_duration_seconds within 0.05 seconds.")
        max_shot = _positive_finite(max_h3_shot_seconds, "max_h3_shot_seconds")
        if not 5.0 <= max_shot <= 15.0:
            raise ValueError("max_h3_shot_seconds must be between 5 and 15 seconds.")
        audio_hash = str(master_audio_sha256 or "").strip().lower()
        if not audio_hash:
            raise ValueError(
                "Project Master has no selected audio SHA-256. Check MUSIC QC STATUS: "
                "at least one decoded song candidate must pass before project planning."
            )
        if not _SHA256_RE.fullmatch(audio_hash):
            raise ValueError("master_audio_sha256 must be an exact 64-character lowercase SHA-256 value.")
        aspect = _canonical_aspect(aspect_ratio)
        try:
            deliverables = json.loads(str(deliverables_json or "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError("deliverables_json must be valid JSON.") from exc
        if not isinstance(deliverables, (dict, list)):
            raise ValueError("deliverables_json must contain a JSON object or array.")
        shot_count = int(math.ceil(duration / max_shot))
        if not 1 <= shot_count <= MAX_H3_SHOTS:
            raise ValueError("The requested duration needs more than four H3 generation lanes.")
        contract = {
            "schema": PROJECT_SCHEMA,
            "version": VERSION,
            "creative_brief": brief,
            "brief_sha256": _sha(brief),
            "production_duration_seconds": round(duration, 6),
            "master_aspect_ratio": aspect,
            "generation_model": str(generation_model or "MiniMax H3 Ref2VA").strip(),
            "audio_lock": {
                "waveform_sha256": audio_hash,
                "excerpt_start_seconds": round(excerpt_start, 6),
                "excerpt_duration_seconds": round(excerpt_duration, 6),
                "lyrics_sha256": _sha(str(lyrics or "")),
            },
            "h3_segmentation": {
                "maximum_shot_seconds": round(max_shot, 6),
                "recommended_shot_count": shot_count,
                "maximum_supported_shots": MAX_H3_SHOTS,
            },
            "requested_deliverables": deliverables,
        }
        project_id = _sha(_json(contract))
        contract["project_id"] = project_id
        status = f"Project {project_id[:12]}… locked to {duration:g}s {aspect}, exact audio {audio_hash[:12]}…, and {shot_count} H3 generation lane{'s' if shot_count != 1 else ''}; native [Shot] blocks remain prompt-controlled."
        return (_json(contract), project_id, aspect, shot_count, contract["brief_sha256"], contract["audio_lock"]["lyrics_sha256"], status, True, max_shot)


class DiffusionGemmaAudioAwareMultiShotPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "measured_audio_report_json": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "base_h3_prompt": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "performance_mode": ("STRING", {"default": "Dance / music sync", "forceInput": True}),
                "excerpt_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
                "excerpt_duration_seconds": ("FLOAT", {"default": 15.0, "min": 0.1, "max": MAX_PROJECT_SECONDS, "step": 0.01}),
                "preferred_shot_seconds": (
                    "FLOAT",
                    {
                        "default": 12.0,
                        "min": 5.0,
                        "max": 15.0,
                        "step": 0.5,
                        "tooltip": "Legacy non-binding hint for one H3 generation pass. It is automatically limited to the active minimum/maximum range and never overrides the Project Master ceiling.",
                    },
                ),
                "min_shot_seconds": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 1.0,
                        "max": 15.0,
                        "step": 0.5,
                        "tooltip": "Legacy name: minimum duration of one separate H3 generation pass.",
                    },
                ),
                "max_shot_seconds": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "min": 5.0,
                        "max": 15.0,
                        "step": 0.5,
                        "tooltip": "Legacy name: hard maximum duration of one H3 generation pass. Multiple native [Shot N] blocks may remain inside. The planner prefers evidence-aligned seams but always honors this cap with a truthful deterministic fallback when needed.",
                    },
                ),
                "project_manifest_json": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            },
            "optional": {
                "lyrics": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "identity_picture_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 2,
                        "step": 1,
                        "tooltip": "1 preserves the legacy Picture 1 identity anchor. 2 keeps Picture 1 as body/wardrobe/composition authority and uses Picture 2 only as multi-panel evidence for the same semantic Subject.",
                    },
                ),
                "continuity_relay_mode": (
                    list(CONTINUITY_RELAY_MODES),
                    {
                        "default": "Previous lane tail",
                        "tooltip": "Previous lane tail preserves the established sequential relay. Off removes the continuity Picture from both prompts and native conditioning so generation lanes may run independently.",
                    },
                ),
                "reference_manifest": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "The exact host reference manifest used by the Ref2VA Director. Dual identity mode requires consecutive Picture 1/Picture 2 rows, each tagged [dg:identity,appearance].",
                    },
                ),
                "timed_lyrics_report_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Optional hash-locked lyric alignment evidence. Missing, stale, weak, or invalid evidence safely falls back to Natural / audio-led sync.",
                    },
                ),
            },
        }

    _lane_types: list[str] = []
    _lane_names: list[str] = []
    for _lane in range(1, MAX_H3_SHOTS + 1):
        _lane_types.extend(("STRING", "FLOAT", "FLOAT", "INT", "BOOLEAN"))
        _lane_names.extend((f"shot_{_lane}_prompt", f"shot_{_lane}_start", f"shot_{_lane}_duration", f"shot_{_lane}_frames", f"shot_{_lane}_ready"))
    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN", "INT", *_lane_types)
    # `effective_shot_count` and shot_N_* are saved-workflow compatibility
    # names. Their values identify duration-bounded generation lanes.
    RETURN_NAMES = ("plan_json", "status", "ready", "effective_shot_count", *_lane_names)
    FUNCTION = "plan"
    CATEGORY = CATEGORY
    DESCRIPTION = "Divide one validated H3 prompt into up to four duration-bounded generation lanes. Native plus measured-recovery seams are preferred; native-only and mixed evidence follow, with a deterministic balanced fallback that preserves mid-shot timing through explicit carry-in fragments."

    def plan(
        self,
        measured_audio_report_json: str,
        base_h3_prompt: str,
        performance_mode: str,
        excerpt_start_seconds: float,
        excerpt_duration_seconds: float,
        preferred_shot_seconds: float,
        min_shot_seconds: float,
        max_shot_seconds: float,
        project_manifest_json: str,
        lyrics: str = "",
        identity_picture_count: int = 1,
        continuity_relay_mode: str = "Previous lane tail",
        reference_manifest: str = "",
        timed_lyrics_report_json: str = "",
    ):
        report = _parse_object(measured_audio_report_json, "measured_audio_report_json")
        project = _parse_object(project_manifest_json, "project_manifest_json")
        excerpt_start = _positive_finite(excerpt_start_seconds, "excerpt_start_seconds", allow_zero=True)
        duration = _positive_finite(excerpt_duration_seconds, "excerpt_duration_seconds")
        minimum = _positive_finite(min_shot_seconds, "min_shot_seconds")
        maximum = _positive_finite(max_shot_seconds, "max_shot_seconds")
        requested_preferred = _positive_finite(
            preferred_shot_seconds, "preferred_shot_seconds"
        )
        mode_value = str(performance_mode or "").strip()
        if mode_value not in PERFORMANCE_MODES:
            raise ValueError(
                "performance_mode must be Dance / music sync, Lyrics + lip sync, or Natural / audio-led sync."
            )
        if minimum > maximum:
            raise ValueError(
                "Generation-lane limits must satisfy min_shot_seconds <= max_shot_seconds."
            )
        if maximum > 15.0:
            raise ValueError("max_shot_seconds cannot exceed H3's 15-second segment limit.")
        project_segmentation = project.get("h3_segmentation")
        if not isinstance(project_segmentation, dict):
            raise ValueError("project_manifest_json.h3_segmentation must be a JSON object.")
        project_maximum = _positive_finite(
            project_segmentation.get("maximum_shot_seconds", -1),
            "project_manifest_json.h3_segmentation.maximum_shot_seconds",
        )
        if abs(maximum - project_maximum) > 1.0e-6:
            raise ValueError(
                "max_shot_seconds must come from the Project Master Contract."
            )
        # Saved workflows expose this legacy preference as an independent widget,
        # while the authoritative ceiling is linked from the Project Master.  A
        # user can therefore lower the ceiling without changing the stale hint.
        # Preferences must not make an otherwise valid project un-runnable.
        effective_preferred = min(max(requested_preferred, minimum), maximum)
        preferred_was_normalized = (
            abs(effective_preferred - requested_preferred) > 1.0e-6
        )
        count = int(math.ceil(duration / maximum))
        if not 1 <= count <= MAX_H3_SHOTS:
            raise ValueError("The excerpt requires more than four H3 generation lanes.")
        if count > 1 and duration < count * minimum:
            raise ValueError("The excerpt is too short for the required number of minimum-duration generation lanes.")
        mode, fields, sections = _section_map(base_h3_prompt)
        if mode != "ref2va":
            raise ValueError(
                "The locked-song H3 multi-shot planner requires a native Ref2VA six-section prompt."
            )
        reference_policy = _identity_relay_policy(
            sections,
            base_h3_prompt,
            identity_picture_count,
            continuity_relay_mode,
            reference_manifest,
        )
        if re.search(r"</?d>", base_h3_prompt, flags=re.IGNORECASE):
            raise ValueError(
                "The music-performance shot plan cannot combine H3 spoken-dialogue blocks with its Dance/Lyrics contract; set H3 dialogue mode to Off or remove requested dialogue."
            )
        if mode_value == "Dance / music sync":
            visible_vocal_cue = _dance_visible_vocal_cue(
                sections["detailed_description"]
            )
            if visible_vocal_cue:
                raise ValueError(
                    "Dance / music sync cannot accept an active visible singing, speaking, mouthing, or lip-sync instruction in the H3 source prompt; remove the cue or choose Lyrics + lip sync."
                )
        detail_field = "detailed_description" if mode == "ref2va" else "integrated_multimodal_description"
        prefix, source_blocks = _shot_blocks(sections[detail_field])
        has_native_markers = _SHOT_RE.search(sections[detail_field]) is not None
        _validate_source_timestamps(source_blocks, duration)
        _validate_music_report(report, project, excerpt_start, duration)
        recovery_points = _recovery_points(report, duration)
        native_points = (
            _native_shot_points(source_blocks, duration)
            if has_native_markers
            else []
        )
        native_recovery_points: list[tuple[float, str]] = []
        strategy = "single_lane"
        fallback_reason = ""
        boundaries: list[tuple[float, str]] | None = None
        if count == 1:
            boundaries = _choose_boundaries(
                duration, count, minimum, maximum, recovery_points
            )
        else:
            attempts: list[tuple[str, list[tuple[float, str]], bool]] = []
            if has_native_markers:
                native_recovery_points = _native_recovery_points(
                    source_blocks, recovery_points, report, duration
                )
                attempts.extend(
                    (
                        ("native_measured_preferred", native_recovery_points, False),
                        ("native_only_fallback", native_points, False),
                        (
                            "mixed_native_or_recovery_fallback",
                            _merge_boundary_points(
                                native_points,
                                recovery_points,
                                native_recovery_points,
                            ),
                            True,
                        ),
                    )
                )
            else:
                attempts.append(
                    ("measured_recovery_preferred", recovery_points, False)
                )
            for candidate_strategy, candidate_points, prefer_evidence in attempts:
                try:
                    boundaries = _choose_boundaries(
                        duration,
                        count,
                        minimum,
                        maximum,
                        candidate_points,
                        prefer_evidence=prefer_evidence,
                    )
                except ValueError:
                    continue
                strategy = candidate_strategy
                break
            if boundaries is None:
                boundaries = _duration_balanced_boundaries(
                    duration, count, minimum, maximum
                )
                strategy = "duration_balanced_fallback"
            if strategy != "native_measured_preferred" and strategy != "measured_recovery_preferred":
                capacity_saturated = abs(duration - count * maximum) <= 1.0e-6
                fallback_reason = (
                    "saturated_lane_capacity_incompatible_with_available_cut_grid"
                    if strategy == "duration_balanced_fallback" and capacity_saturated
                    else "no_complete_preferred_cut_sequence_within_generation_lane_limits"
                )
        boundaries, inter_lane_cuts = _boundary_evidence_records(
            boundaries,
            strategy=strategy,
            native_points=native_points,
            recovery_points=recovery_points,
            report=report,
            duration=duration,
        )
        allow_non_native_carry = has_native_markers and any(
            not bool(item["native_shot_boundary"]) for item in inter_lane_cuts
        )
        lane_blocks, lane_carry_ins = _partition_native_shot_blocks_with_carry(
            source_blocks,
            boundaries,
            has_native_markers=has_native_markers,
            allow_non_native_carry=allow_non_native_carry,
        )
        if mode_value == "Lyrics + lip sync":
            lyric_timing = _timed_lyrics_contract(
                timed_lyrics_report_json,
                project,
                lyrics,
                excerpt_start,
                duration,
            )
            effective_performance_mode = (
                "Lyrics + lip sync"
                if lyric_timing["timing_ready"]
                else "Natural / audio-led sync"
            )
        else:
            lyric_timing = {
                "state": (
                    "dance_non_vocal"
                    if mode_value == "Dance / music sync"
                    else "natural_audio_led"
                ),
                "analysis_status": "not_required",
                "timing_ready": False,
                "events": [],
                "vocal_intervals": [],
                "instrumental_intervals": [],
                "warnings": [],
            }
            effective_performance_mode = mode_value
        shots: list[dict[str, Any]] = []
        outputs: list[Any] = []
        for index in range(count):
            start = boundaries[index][0]
            end = boundaries[index + 1][0]
            shot_duration = end - start
            master_frame_start = int(round(start * H3_FPS))
            master_frame_end = int(round(end * H3_FPS))
            retained_master_frames = master_frame_end - master_frame_start
            if retained_master_frames <= 0:
                raise ValueError("A planned H3 segment has no retained master frames.")
            lane_timing_events = _lane_timed_events(
                list(lyric_timing["events"]), start, end
            )
            if lane_timing_events:
                lane_timing_state = "timed_lyrics"
            elif lyric_timing["timing_ready"] and _intervals_cover_lane(
                list(lyric_timing["instrumental_intervals"]), start, end
            ):
                lane_timing_state = "confirmed_instrumental"
            elif effective_performance_mode == "Dance / music sync":
                lane_timing_state = "dance_non_vocal"
            else:
                lane_timing_state = "natural_audio_led"
            lane_lyrics: list[str] = []
            for event in lane_timing_events:
                lexical_items = list(event["authored_lines"]) or [str(event["text"])]
                for item in lexical_items:
                    if item not in lane_lyrics:
                        lane_lyrics.append(item)
            prompt = _segment_prompt(
                mode,
                fields,
                sections,
                prefix,
                lane_blocks[index],
                index + 1,
                count,
                start,
                shot_duration,
                effective_performance_mode,
                lane_timing_events,
                lane_timing_state,
                list(reference_policy["identity_picture_tags"]),
                str(reference_policy["continuity_picture_tag"]),
                str(reference_policy["identity_subject_tag"]),
            )
            frames = _h3_frames(shot_duration)
            carry_in = lane_carry_ins[index]
            native_shot_count = len(lane_blocks[index]) - int(
                bool(carry_in["required"])
            )
            shot = {
                "index": index + 1,
                "relative_start_seconds": round(start, 6),
                "absolute_song_start_seconds": round(excerpt_start + start, 6),
                "duration_seconds": round(shot_duration, 6),
                "generated_h3_frames": frames,
                "master_frame_start": master_frame_start,
                "master_frame_end_exclusive": master_frame_end,
                "retained_master_frames": retained_master_frames,
                "cut_in_evidence": boundaries[index][1],
                "cut_out_evidence": boundaries[index + 1][1],
                "lyrics": lane_lyrics,
                "lyrics_timing_state": lane_timing_state,
                "timed_lyrics_events": lane_timing_events,
                "timing_state": lane_timing_state,
                "timing_events": lane_timing_events,
                "lyrics_not_a_timing_schedule": True,
                "requested_performance_mode": mode_value,
                "effective_performance_mode": effective_performance_mode,
                "native_shot_count": native_shot_count,
                "rendered_source_fragment_count": len(lane_blocks[index]),
                "source_shot_continuation_at_lane_start": bool(
                    carry_in["required"]
                ),
                "carry_in_source_native_shot_index": int(
                    carry_in["source_native_shot_index"]
                ),
                "carry_in_source_native_shot_start_seconds": float(
                    carry_in["source_native_shot_start_seconds"]
                ),
                "identity_picture_tags": list(reference_policy["identity_picture_tags"]),
                "continuity_reference_required": bool(
                    index > 0
                    and reference_policy["continuity_relay_mode"] == "previous_lane_tail"
                ),
                "continuity_picture_tag": (
                    str(reference_policy["continuity_picture_tag"])
                    if index > 0
                    and reference_policy["continuity_relay_mode"] == "previous_lane_tail"
                    else ""
                ),
                "reference_picture_tags": [
                    *list(reference_policy["identity_picture_tags"]),
                    *(
                        [str(reference_policy["continuity_picture_tag"])]
                        if index > 0
                        and reference_policy["continuity_relay_mode"] == "previous_lane_tail"
                        else []
                    ),
                ],
                "prompt_sha256": _sha(prompt),
                "prompt": prompt,
                "ready": True,
            }
            shots.append(shot)
            outputs.extend((prompt, float(start), float(shot_duration), int(frames), True))
        while len(outputs) < MAX_H3_SHOTS * 5:
            outputs.extend(("", 0.0, 0.0, 0, False))
        realized_lane_seconds = [
            round(float(shot["duration_seconds"]), 6) for shot in shots
        ]
        realized_mean_seconds = round(
            sum(realized_lane_seconds) / len(realized_lane_seconds), 6
        )
        preferred_realized = all(
            abs(value - effective_preferred) <= 0.05
            for value in realized_lane_seconds
        )
        fallback_used = strategy.endswith("_fallback")
        all_cuts_native = all(
            bool(item["native_shot_boundary"]) for item in inter_lane_cuts
        )
        all_cuts_measured = all(
            bool(item["measured_low_density_recovery"]) for item in inter_lane_cuts
        )
        capacity_saturated = count > 1 and abs(duration - count * maximum) <= 1.0e-6
        cut_policy = (
            "inter-lane generation cuts use native H3 shot boundaries with measured "
            "low-density visual-recovery evidence; in-lane native shot cuts follow the "
            "validated source prompt"
            if strategy == "native_measured_preferred"
            else "inter-lane cuts prefer native H3 shot boundaries with measured recovery, "
            "then native-only and mixed native/recovery sequences; when none satisfies the "
            "hard lane-duration limits, deterministic duration-balanced seams are used. "
            "Non-native seams carry the already-active source shot into the next lane, and "
            "every cut records literal provenance without claiming absent evidence"
        )
        plan = {
            "schema": SHOT_PLAN_SCHEMA,
            "version": VERSION,
            "project_id": str(project.get("project_id", "")),
            "prompt_mode": mode,
            "excerpt_start_seconds": round(excerpt_start, 6),
            "excerpt_duration_seconds": round(duration, 6),
            "fps": H3_FPS,
            "effective_shot_count": count,  # Legacy key: generation-lane count.
            "effective_generation_lane_count": count,
            "source_native_shot_count": len(source_blocks),
            "performance_mode": mode_value,
            "requested_performance_mode": mode_value,
            "effective_performance_mode": effective_performance_mode,
            "lyrics_timing_state": str(lyric_timing["state"]),
            "timed_lyrics_events": list(lyric_timing["events"]),
            "timing_state": str(lyric_timing["state"]),
            "timing_events": list(lyric_timing["events"]),
            "lyrics_not_a_timing_schedule": True,
            "timed_lyrics_report": {
                "schema": TIMED_LYRICS_REPORT_SCHEMA,
                "version": TIMED_LYRICS_REPORT_VERSION,
                "analysis_status": str(lyric_timing["analysis_status"]),
                "timing_ready": bool(lyric_timing["timing_ready"]),
                "warnings": list(lyric_timing["warnings"]),
            },
            "reference_policy": reference_policy,
            "generation_lane_duration_policy": {
                "minimum_seconds": round(minimum, 6),
                "requested_preferred_seconds": round(requested_preferred, 6),
                "effective_preferred_seconds": round(effective_preferred, 6),
                "maximum_seconds": round(maximum, 6),
                "preferred_was_normalized": preferred_was_normalized,
                "preferred_realized": preferred_realized,
                "realized_lane_seconds": realized_lane_seconds,
                "realized_mean_seconds": realized_mean_seconds,
                "capacity_saturated": capacity_saturated,
                "maximum_source": "project_master_contract",
            },
            "boundary_selection_revision": BOUNDARY_SELECTION_REVISION,
            "generation_lane_boundary_policy": {
                "revision": BOUNDARY_SELECTION_REVISION,
                "strategy": strategy,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "priority": [
                    "native_measured",
                    "native_only",
                    "mixed_native_or_measured",
                    "duration_balanced",
                ],
                "all_inter_lane_cuts_native": all_cuts_native,
                "all_inter_lane_cuts_measured_recovery": all_cuts_measured,
                "capacity_saturated": capacity_saturated,
            },
            "inter_lane_cuts": inter_lane_cuts,
            "cut_policy": cut_policy,
            "shots": shots,
            "ready": True,
        }
        native_count = len(source_blocks)
        realized_text = "/".join(f"{value:g}" for value in realized_lane_seconds)
        if preferred_was_normalized:
            preference_status = (
                f" The non-binding preferred lane duration was limited from "
                f"{requested_preferred:g}s to {effective_preferred:g}s by the active "
                f"{minimum:g}-{maximum:g}s range; realized lanes are {realized_text}s "
                f"(mean {realized_mean_seconds:g}s)."
            )
        elif not preferred_realized:
            preference_status = (
                f" The non-binding {effective_preferred:g}s preference cannot be realized "
                f"by this {count}-lane hard-limit solution; realized lanes are "
                f"{realized_text}s (mean {realized_mean_seconds:g}s), with the "
                f"{maximum:g}s ceiling supplied by the Project Master."
            )
        else:
            preference_status = (
                f" The non-binding {effective_preferred:g}s preference is realized within "
                f"the active {minimum:g}-{maximum:g}s range."
            )
        identity_status = (
            "one Picture identity anchor"
            if reference_policy["identity_picture_count"] == 1
            else "two Picture identity anchors jointly bound to "
            + str(reference_policy["identity_subject_tag"])
        )
        relay_status = (
            "previous-tail continuity relay "
            + str(reference_policy["continuity_picture_tag"])
            if reference_policy["continuity_relay_mode"] == "previous_lane_tail"
            else "continuity relay off"
        )
        timing_status = ""
        if mode_value == "Lyrics + lip sync":
            if lyric_timing["timing_ready"]:
                timing_status = (
                    f" Timed lyrics are hash-locked and excerpt-locked with "
                    f"{len(lyric_timing['events'])} audio-gated lexical event"
                    f"{'s' if len(lyric_timing['events']) != 1 else ''}; written lyrics are not a timing schedule."
                )
            else:
                timing_status = (
                    " Lyrics mode safely fell back to Natural / audio-led sync: "
                    + " ".join(str(item) for item in lyric_timing["warnings"])
                )
        if not inter_lane_cuts:
            boundary_status = "one lane requires no inter-lane cut"
        elif strategy == "native_measured_preferred":
            boundary_status = (
                "every inter-lane cut is both a native shot boundary and measured "
                "recovery evidence"
            )
        else:
            native_hits = sum(
                bool(item["native_shot_boundary"]) for item in inter_lane_cuts
            )
            recovery_hits = sum(
                bool(item["measured_low_density_recovery"])
                for item in inter_lane_cuts
            )
            carry_count = sum(
                bool(item["source_shot_continuation_at_lane_start"])
                for item in shots
            )
            strategy_label = strategy.replace("_", " ")
            boundary_status = (
                f"{strategy_label} selected {native_hits}/{len(inter_lane_cuts)} native "
                f"and {recovery_hits}/{len(inter_lane_cuts)} measured-recovery cuts; "
                f"{carry_count} non-native seam{'s' if carry_count != 1 else ''} retain "
                "the active source shot as a lane-local continuation"
            )
        status = f"Audio-aware H3 plan covers {duration:g}s with {count} generation lane{'s' if count != 1 else ''} carrying {native_count} unique native H3 shot block{'s' if native_count != 1 else ''} in {effective_performance_mode} mode, {identity_status}, and {relay_status}; {boundary_status}, and starts are relative to the selected excerpt.{timing_status}{preference_status}"
        return (_json(plan), status, True, count, *outputs)


class DiffusionGemmaH3RelayReferenceGate:
    """Expose the prior-lane tail only when the shot plan enables continuity relay."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_plan_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Connect plan_json from the Audio-Aware H3 Multi-Shot Planner. The embedded identity/relay policy is authoritative.",
                    },
                ),
                "lane_index": (
                    "INT",
                    {
                        "default": 2,
                        "min": 2,
                        "max": MAX_H3_SHOTS,
                        "step": 1,
                        "tooltip": "The later generation lane that may consume the immediately previous lane's retained tail.",
                    },
                ),
            },
            "optional": {
                "relay_image": (
                    "IMAGE",
                    {
                        "lazy": True,
                        "tooltip": "One retained final frame from the immediately previous lane. It is not evaluated when relay mode is Off or this lane is not planned.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("relay_reference", "relay_picture_tag", "enabled", "status")
    FUNCTION = "route"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Lazily route one prior-lane tail into the next native MiniMax H3 image-reference socket. "
        "Dual identity anchors reserve Picture 1 and Picture 2, so relay becomes Picture 3. "
        "Off returns None, which stock MiniMaxH3ReferenceToVideo skips as an absent optional image."
    )

    @staticmethod
    def _policy(
        shot_plan_json: str,
        lane_index: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        plan = _parse_object(shot_plan_json, "shot_plan_json")
        shots = plan.get("shots")
        if (
            plan.get("schema") != SHOT_PLAN_SCHEMA
            or plan.get("version") != VERSION
            or not plan.get("ready")
            or not isinstance(shots, list)
        ):
            raise ValueError("shot_plan_json is not a ready audio-aware H3 shot plan.")
        try:
            lane = int(lane_index)
            count = int(plan.get("effective_generation_lane_count", 0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("shot_plan_json has an invalid generation-lane count.") from exc
        if not 2 <= lane <= MAX_H3_SHOTS:
            raise ValueError(f"lane_index must be between 2 and {MAX_H3_SHOTS}.")
        if not 1 <= count <= MAX_H3_SHOTS or len(shots) != count:
            raise ValueError("shot_plan_json has an invalid generation-lane list.")

        policy = plan.get("reference_policy")
        if (
            not isinstance(policy, dict)
            or policy.get("schema") != IDENTITY_RELAY_POLICY_SCHEMA
            or policy.get("version") != VERSION
        ):
            raise ValueError("shot_plan_json has no supported identity/relay policy.")
        try:
            identity_count = int(policy.get("identity_picture_count", 0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("shot_plan_json identity picture count is invalid.") from exc
        if identity_count not in {1, 2}:
            raise ValueError("shot_plan_json identity picture count is invalid.")
        expected_identity_tags = [
            f"<Picture {index}>" for index in range(1, identity_count + 1)
        ]
        if policy.get("identity_picture_tags") != expected_identity_tags:
            raise ValueError("shot_plan_json identity Picture tags are inconsistent.")
        relay_mode = str(policy.get("continuity_relay_mode", ""))
        if relay_mode not in {"previous_lane_tail", "off"}:
            raise ValueError("shot_plan_json continuity relay mode is invalid.")
        expected_continuity_tag = (
            f"<Picture {identity_count + 1}>"
            if relay_mode == "previous_lane_tail"
            else ""
        )
        if str(policy.get("continuity_picture_tag", "")) != expected_continuity_tag:
            raise ValueError("shot_plan_json continuity Picture tag is inconsistent.")

        active = lane <= count
        if active:
            shot = shots[lane - 1]
            if not isinstance(shot, dict) or int(shot.get("index", 0)) != lane:
                raise ValueError(f"shot_plan_json lane {lane} metadata is invalid.")
            expected_required = relay_mode == "previous_lane_tail"
            if bool(shot.get("continuity_reference_required")) != expected_required:
                raise ValueError(
                    f"shot_plan_json lane {lane} continuity requirement conflicts with its reference policy."
                )
            expected_shot_tag = expected_continuity_tag if expected_required else ""
            if str(shot.get("continuity_picture_tag", "")) != expected_shot_tag:
                raise ValueError(
                    f"shot_plan_json lane {lane} continuity Picture tag conflicts with its reference policy."
                )
        return plan, policy, active

    def check_lazy_status(
        self,
        shot_plan_json: str,
        lane_index: int,
        relay_image: Any = None,
    ) -> list[str]:
        try:
            _plan, policy, active = self._policy(shot_plan_json, lane_index)
        except ValueError:
            return []
        needs_relay = (
            active and policy["continuity_relay_mode"] == "previous_lane_tail"
        )
        return ["relay_image"] if needs_relay and relay_image is None else []

    def route(
        self,
        shot_plan_json: str,
        lane_index: int,
        relay_image: Any = None,
    ):
        _plan, policy, active = self._policy(shot_plan_json, lane_index)
        lane = int(lane_index)
        if not active:
            return (
                None,
                "",
                False,
                f"Generation lane {lane} is not planned; no relay reference is emitted.",
            )
        if policy["continuity_relay_mode"] == "off":
            return (
                None,
                "",
                False,
                f"Generation lane {lane} runs independently because continuity relay is Off.",
            )
        if not isinstance(relay_image, torch.Tensor) or relay_image.ndim != 4:
            raise ValueError(
                "relay_image must be one ComfyUI IMAGE tensor from the immediately previous lane."
            )
        if int(relay_image.shape[0]) != 1:
            raise ValueError(
                f"relay_image must contain exactly one retained tail frame; got {int(relay_image.shape[0])}."
            )
        if int(relay_image.shape[-1]) != 3:
            raise ValueError("relay_image must be an RGB IMAGE with 3 channels.")
        tag = str(policy["continuity_picture_tag"])
        return (
            relay_image,
            tag,
            True,
            f"Generation lane {lane} receives the immediately previous retained tail as {tag} for continuity only, never identity; {policy['identity_picture_tags'][0]} remains body/wardrobe/composition authority and the configured identity pictures jointly identify the same Subject.",
        )


class DiffusionGemmaMultiFormatDeliveryPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_manifest_json": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "format_policy": (["Master only", "Master + social adaptations"], {"default": "Master + social adaptations"}),
                "cutdown_policy": (["None", "Plan 15s + 6s cutdowns"], {"default": "Plan 15s + 6s cutdowns"}),
                "safe_area_percent": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 25.0, "step": 1.0}),
            }
        }
    RETURN_TYPES = ("STRING", "INT", "STRING", "BOOLEAN")
    RETURN_NAMES = ("delivery_manifest_json", "deliverable_count", "status", "ready")
    FUNCTION = "plan"
    CATEGORY = CATEGORY
    DESCRIPTION = "Create an honest plan-only manifest for master, reframe, and cutdown deliverables. It does not claim those adaptations were rendered."

    def plan(self, project_manifest_json: str, format_policy: str, cutdown_policy: str, safe_area_percent: float):
        project = _parse_object(project_manifest_json, "project_manifest_json")
        if project.get("schema") != PROJECT_SCHEMA:
            raise ValueError("project_manifest_json has an unsupported schema.")
        duration = float(project["production_duration_seconds"])
        master_aspect = str(project["master_aspect_ratio"])
        safe_area = max(0.0, min(25.0, float(safe_area_percent)))
        requested_deliverables = project.get("requested_deliverables", {})
        items: list[dict[str, Any]] = [
            {
                "id": "primary_master",
                "aspect_ratio": master_aspect,
                "duration_seconds": duration,
                "status": "rendered_by_primary_workflow_when_run",
                "audio_policy": "exact locked master excerpt",
            }
        ]
        if format_policy == "Master + social adaptations":
            requested_aspects: list[str] = []
            if isinstance(requested_deliverables, Mapping):
                raw_adaptations = requested_deliverables.get("adaptations", [])
                if isinstance(raw_adaptations, list):
                    for value in raw_adaptations:
                        match = re.search(
                            r"\b(?:21:9|16:9|9:16|4:3|3:4|3:2|2:3|1:1)\b",
                            str(value),
                        )
                        if match and match.group(0) not in requested_aspects:
                            requested_aspects.append(match.group(0))
            if not requested_aspects:
                requested_aspects = ["9:16", "16:9", "1:1"]
            for aspect in requested_aspects:
                if aspect == master_aspect:
                    continue
                items.append({
                    "id": f"adapt_{aspect.replace(':', 'x')}",
                    "aspect_ratio": aspect,
                    "duration_seconds": duration,
                    "status": "planned_only_not_rendered",
                    "reframe_policy": "semantic reframe required; do not claim a destructive center crop is a finished adaptation",
                })
        if cutdown_policy == "Plan 15s + 6s cutdowns":
            for seconds in (15.0, 6.0):
                if duration > seconds + 0.05:
                    items.append({
                        "id": f"cutdown_{int(seconds)}s",
                        "aspect_ratio": master_aspect,
                        "duration_seconds": seconds,
                        "status": "planned_only_not_rendered",
                        "selection_policy": "choose a complete musical phrase and preserve the exact waveform window",
                    })
        manifest = {
            "schema": DELIVERY_SCHEMA,
            "version": VERSION,
            "project_id": project.get("project_id", ""),
            "safe_area_percent": safe_area,
            "plan_only": True,
            "requested_deliverables": requested_deliverables,
            "deliverables": items,
            "ready": True,
        }
        status = f"Delivery manifest contains {len(items)} item(s); only the primary master is wired to render, while adaptations remain explicitly plan-only."
        return (_json(manifest), len(items), status, True)


class DiffusionGemmaH3ShotSeedFanout:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root_seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "project_manifest_json": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            }
        }
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("shot_1_seed", "shot_2_seed", "shot_3_seed", "shot_4_seed", "seed_report_json")
    FUNCTION = "fanout"
    CATEGORY = CATEGORY
    DESCRIPTION = "Derive deterministic per-generation-lane H3 seeds. Legacy shot_1..shot_4 socket names are retained for saved-workflow compatibility."

    def fanout(self, root_seed: int, project_manifest_json: str):
        project = _parse_object(project_manifest_json, "project_manifest_json")
        if project.get("schema") != PROJECT_SCHEMA or not project.get("project_id"):
            raise ValueError("project_manifest_json is not a valid project master contract.")
        root = int(root_seed) & 0xFFFFFFFFFFFFFFFF
        project_id = str(project["project_id"])
        seeds: list[int] = []
        for index in range(1, MAX_H3_SHOTS + 1):
            digest = hashlib.sha256(f"diffusiongemma.h3-shot-seed@1\0{root}\0{project_id}\0{index}".encode("utf-8")).digest()
            seeds.append(int.from_bytes(digest[:8], "little", signed=False))
        report = {"schema": "diffusiongemma.h3_shot_seed_fanout", "version": VERSION, "root_seed": root, "project_id": project_id, "seeds": seeds}
        return (*seeds, _json(report))


class DiffusionGemmaH3ShotAssembler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_audio": ("AUDIO",),
                "plan_json": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "target_duration_seconds": ("FLOAT", {"default": 15.0, "min": 0.1, "max": MAX_PROJECT_SECONDS, "step": 0.01}),
                "fps": ("FLOAT", {"default": H3_FPS, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
            "optional": {
                **{
                    f"shot_{index}_images": ("IMAGE", {"lazy": True})
                    for index in range(1, MAX_H3_SHOTS + 1)
                }
            },
        }
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("assembled_images", "final_audio", "assembly_report_json", "status", "ready")
    FUNCTION = "assemble"
    CATEGORY = CATEGORY
    DESCRIPTION = "Lazily join only planned H3 image batches, trim every segment to its master timing, and pass the exact locked audio object unchanged."

    @staticmethod
    def _plan(plan_json: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        plan = _parse_object(plan_json, "plan_json")
        shots = plan.get("shots")
        if plan.get("schema") != SHOT_PLAN_SCHEMA or not plan.get("ready") or not isinstance(shots, list):
            raise ValueError("plan_json is not a ready audio-aware H3 shot plan.")
        count = int(plan.get("effective_shot_count", 0))
        if not 1 <= count <= MAX_H3_SHOTS or len(shots) != count:
            raise ValueError("plan_json has an invalid effective generation-lane count.")
        return plan, [dict(item) for item in shots]

    def check_lazy_status(self, final_audio: Any, plan_json: str, target_duration_seconds: float, fps: float, **kwargs: Any):
        try:
            _plan, shots = self._plan(plan_json)
        except ValueError:
            return []
        requested: list[str] = []
        for index in range(1, len(shots) + 1):
            key = f"shot_{index}_images"
            if key in kwargs and kwargs[key] is None:
                requested.append(key)
        return requested

    @staticmethod
    def _failure(message: str, final_audio: Any):
        blocker = ExecutionBlocker(message)
        report = _json({"schema": ASSEMBLY_SCHEMA, "version": VERSION, "ready": False, "failure": message})
        return (blocker, final_audio, report, message, False)

    def assemble(self, final_audio: Any, plan_json: str, target_duration_seconds: float, fps: float, **kwargs: Any):
        try:
            _plan, shots = self._plan(plan_json)
            target_duration = _positive_finite(target_duration_seconds, "target_duration_seconds")
            frame_rate = _positive_finite(fps, "fps")
        except ValueError as exc:
            return self._failure(str(exc), final_audio)
        plan_duration = float(_plan.get("excerpt_duration_seconds", -1))
        plan_fps = float(_plan.get("fps", -1))
        if abs(plan_duration - target_duration) > 0.001:
            return self._failure(
                "Assembler target duration does not match the audio-aware shot plan.",
                final_audio,
            )
        if abs(plan_fps - frame_rate) > 1.0e-6:
            return self._failure(
                "Assembler fps does not match the audio-aware shot plan.", final_audio
            )
        if not isinstance(final_audio, Mapping):
            return self._failure("final_audio is not a ComfyUI AUDIO object.", final_audio)
        waveform = final_audio.get("waveform")
        sample_rate = final_audio.get("sample_rate")
        if not isinstance(waveform, torch.Tensor) or waveform.ndim < 1:
            return self._failure("final_audio has no valid waveform tensor.", final_audio)
        try:
            audio_duration = int(waveform.shape[-1]) / float(sample_rate)
        except (TypeError, ValueError, ZeroDivisionError):
            return self._failure("final_audio has an invalid sample rate.", final_audio)
        if not math.isfinite(audio_duration) or abs(audio_duration - target_duration) > 0.05:
            return self._failure(
                "The exact final audio duration does not match the shot-plan master clock within 0.05 seconds.",
                final_audio,
            )
        retained: list[torch.Tensor] = []
        shot_reports: list[dict[str, Any]] = []
        spatial_shape: tuple[int, ...] | None = None
        for index, shot in enumerate(shots, start=1):
            key = f"shot_{index}_images"
            images = kwargs.get(key)
            if not isinstance(images, torch.Tensor) or images.ndim != 4 or int(images.shape[0]) <= 0:
                return self._failure(f"{key} is missing or is not a non-empty ComfyUI IMAGE batch.", final_audio)
            shape = tuple(int(value) for value in images.shape[1:])
            if spatial_shape is None:
                spatial_shape = shape
            elif shape != spatial_shape:
                return self._failure(f"{key} dimensions {shape} do not match earlier shots {spatial_shape}.", final_audio)
            try:
                keep = int(shot["retained_master_frames"])
                frame_start = int(shot["master_frame_start"])
                frame_end = int(shot["master_frame_end_exclusive"])
            except (KeyError, TypeError, ValueError):
                return self._failure(
                    f"{key} has no valid cumulative master-frame allocation in the plan.",
                    final_audio,
                )
            if frame_end - frame_start != keep:
                return self._failure(
                    f"{key} has an inconsistent cumulative master-frame allocation.",
                    final_audio,
                )
            if keep <= 0 or int(images.shape[0]) < keep:
                return self._failure(f"{key} has {int(images.shape[0])} frames but its plan requires {keep} retained frames.", final_audio)
            retained.append(images[:keep])
            shot_reports.append({"index": index, "generated_frames": int(images.shape[0]), "retained_frames": keep, "trimmed_padding_frames": int(images.shape[0]) - keep})
        assembled = torch.cat(retained, dim=0)
        target_frames = int(round(target_duration * frame_rate))
        if int(assembled.shape[0]) != target_frames:
            return self._failure(
                f"Assembled shots contain {int(assembled.shape[0])} retained frames but the master requires exactly {target_frames}.",
                final_audio,
            )
        report = {
            "schema": ASSEMBLY_SCHEMA,
            "version": VERSION,
            "ready": True,
            "fps": frame_rate,
            "target_duration_seconds": target_duration,
            "target_frames": target_frames,
            "assembled_frames": int(assembled.shape[0]),
            "hard_cuts": True,
            "audio_identity_preserved": True,
            "shots": shot_reports,
        }
        status = f"Assembled {len(shots)} H3 shot{'s' if len(shots) != 1 else ''} into {target_frames} frames at {frame_rate:g} fps; the exact locked audio object is unchanged."
        return (assembled, final_audio, _json(report), status, True)


NODE_CLASS_MAPPINGS = {
    "DiffusionGemmaProjectMasterContract": DiffusionGemmaProjectMasterContract,
    "DiffusionGemmaAudioAwareMultiShotPlanner": DiffusionGemmaAudioAwareMultiShotPlanner,
    "DiffusionGemmaH3RelayReferenceGate": DiffusionGemmaH3RelayReferenceGate,
    "DiffusionGemmaMultiFormatDeliveryPlanner": DiffusionGemmaMultiFormatDeliveryPlanner,
    "DiffusionGemmaH3ShotSeedFanout": DiffusionGemmaH3ShotSeedFanout,
    "DiffusionGemmaH3ShotAssembler": DiffusionGemmaH3ShotAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionGemmaProjectMasterContract": "DiffusionGemma Project Master Contract",
    "DiffusionGemmaAudioAwareMultiShotPlanner": "DiffusionGemma Audio-Aware H3 Multi-Shot Planner",
    "DiffusionGemmaH3RelayReferenceGate": "DiffusionGemma H3 Relay Reference Gate",
    "DiffusionGemmaMultiFormatDeliveryPlanner": "DiffusionGemma Multi-Format Delivery Planner",
    "DiffusionGemmaH3ShotSeedFanout": "DiffusionGemma H3 Shot Seed Fanout",
    "DiffusionGemmaH3ShotAssembler": "DiffusionGemma H3 Shot Assembler",
}
