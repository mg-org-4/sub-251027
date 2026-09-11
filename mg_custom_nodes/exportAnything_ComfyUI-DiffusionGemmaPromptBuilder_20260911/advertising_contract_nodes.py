# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

"""Typed, additive contracts for advertisement-specific workflows.

This module deliberately does not alter the generic music-video contracts.  It
turns commercial intent into small versioned JSON envelopes that an
advertisement workflow can validate before an expensive Director or H3 lane is
queued.  Natural-language descriptions are never used as machine-readable
reference roles; the generated manifest carries explicit ``[ad:...]`` role
annotations instead.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import torch
import torch.nn.functional as torch_functional


CATEGORY = "prompt/diffusiongemma/advertising"
CAMPAIGN_SCHEMA = "diffusiongemma.advertisement_campaign_contract"
REFERENCE_SCHEMA = "diffusiongemma.advertisement_reference_contract"
SOUNDTRACK_SCHEMA = "diffusiongemma.advertisement_soundtrack_contract"
VERSION = 1
ASPECT_RATIOS = ("9:16", "16:9", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9")
REFERENCE_SLOT_POLICIES = (
    "2 performer + 1 product + relay",
    "1 performer + 1 product + relay",
    "2 performer + 1 product; relay off",
)
EXPORTED_REFERENCE_SLOT_POLICIES = (REFERENCE_SLOT_POLICIES[0],)
ADVERTISEMENT_PERFORMANCE_MODES = (
    "Natural / audio-led sync",
    "Dance / music sync",
)
ADVERTISEMENT_PERFORMANCE_CARRIER_MODES = (
    "Dance / music sync",
    "Lyrics + lip sync",
    "Natural / audio-led sync",
)
H3_GENERATION_MODE_CARRIER = ("t2va", "ref2va")
H3_SHOT_COUNT_CARRIER = ("auto", *(str(value) for value in range(1, 13)), "custom")
H3_AUDIO_MODE_CARRIER = ("auto_scene_audio", "explicit_sound_design", "visual_only")
H3_DIALOGUE_MODE_CARRIER = ("auto", "required", "off")
ADVERTISEMENT_FPS = 24
PRODUCT_REFERENCE_KINDS = ("single packshot", "product contact sheet")
SOUNDTRACK_CONTENT_MODES = ("Instrumental", "Vocal", "Auto")
VOICE_OVER_POLICIES = ("None", "Separate non-diegetic VO")
EXPORTED_VOICE_OVER_POLICIES = (VOICE_OVER_POLICIES[0],)
TIME_SIGNATURES = ("2", "3", "4", "6")
LANGUAGES = (
    "unknown", "en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh",
    "yue", "hi", "ar", "tr", "ru", "uk", "pl", "nl", "sv", "no",
    "da", "fi", "id", "vi", "th", "tl",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REFERENCE_ROW_RE = re.compile(
    r"^\s*<\s*(Picture|Video|Audio)\s+([1-9]\d*)\s*>\s*:\s*(\S.*)$",
    flags=re.IGNORECASE,
)
_AD_ANNOTATION_RE = re.compile(r"\[\s*ad\s*:\s*([^\]]+)\]", flags=re.IGNORECASE)
_LYRIC_SECTION_RE = re.compile(
    r"(?im)^\s*\[(Intro|Verse(?:\s+\d+)?|Pre-Chorus|Chorus|Post-Chorus|Bridge|Outro|Instrumental|Break|Interlude)\]\s*$"
)


def _image_tensor(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError(f"{label} must be a ComfyUI IMAGE tensor [batch,height,width,channels].")
    if int(value.shape[0]) != 1:
        raise ValueError(f"{label} must contain exactly one image.")
    if int(value.shape[-1]) not in {3, 4}:
        raise ValueError(f"{label} must have RGB or RGBA channels.")
    result = value.detach().to(device="cpu", dtype=torch.float32)
    if int(result.shape[-1]) == 4:
        alpha = result[..., 3:4].clamp(0.0, 1.0)
        result = result[..., :3] * alpha + 0.5 * (1.0 - alpha)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} contains NaN or infinite pixels.")
    return result.clamp(0.0, 1.0).contiguous()


def _budgeted_image(
    image: torch.Tensor,
    *,
    maximum_area: float,
    multiple: int = 8,
) -> torch.Tensor:
    source_height = int(image.shape[1])
    source_width = int(image.shape[2])
    source_area = source_width * source_height
    if source_area <= maximum_area + 1.0e-6:
        return image
    scale = math.sqrt(maximum_area / float(source_area))
    target_width = max(multiple, int(math.floor(source_width * scale / multiple)) * multiple)
    target_height = max(multiple, int(math.floor(source_height * scale / multiple)) * multiple)
    channels_first = image.permute(0, 3, 1, 2)
    resized = torch_functional.interpolate(
        channels_first,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.permute(0, 2, 3, 1).contiguous()


def _image_sha256(image: torch.Tensor, role: str) -> str:
    value = image.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(b"diffusiongemma-ad-reference-image@1\0")
    digest.update(str(role).encode("utf-8"))
    digest.update(b"\0")
    for size in value.shape:
        digest.update(int(size).to_bytes(8, "little", signed=False))
    payload = memoryview(value.view(torch.uint8).numpy()).cast("B")
    for offset in range(0, len(payload), 1024 * 1024):
        digest.update(payload[offset : offset + 1024 * 1024])
    return digest.hexdigest()


def _image_content_sha256(image: torch.Tensor) -> str:
    """Hash normalized source pixels without a role salt for distinctness."""

    value = image.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(b"diffusiongemma-ad-reference-source-image@1\0")
    for size in value.shape:
        digest.update(int(size).to_bytes(8, "little", signed=False))
    payload = memoryview(value.view(torch.uint8).numpy()).cast("B")
    for offset in range(0, len(payload), 1024 * 1024):
        digest.update(payload[offset : offset + 1024 * 1024])
    return digest.hexdigest()


def _director_canvas(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    source_height = int(image.shape[1])
    source_width = int(image.shape[2])
    scale = min(width / float(source_width), height / float(source_height))
    target_width = max(1, min(width, int(round(source_width * scale))))
    target_height = max(1, min(height, int(round(source_height * scale))))
    resized = torch_functional.interpolate(
        image.permute(0, 3, 1, 2),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).permute(0, 2, 3, 1)
    canvas = torch.full((1, height, width, 3), 0.5, dtype=torch.float32)
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    canvas[:, top : top + target_height, left : left + target_width, :] = resized
    return canvas.contiguous()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _strict_json_object(value: Any, label: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON number {token}.")

    try:
        parsed = json.loads(str(value or ""), parse_constant=reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid strict JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def _strict_json_array(value: Any, label: str) -> list[Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON number {token}.")

    try:
        parsed = json.loads(str(value or ""), parse_constant=reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid strict JSON.") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{label} must be a JSON array.")
    return parsed


def _text(value: Any, label: str, *, maximum: int, required: bool = True) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ValueError(f"{label} is required.")
    if "\x00" in result:
        raise ValueError(f"{label} cannot contain a NUL character.")
    if len(result) > maximum:
        raise ValueError(f"{label} exceeds {maximum} characters.")
    return result


def _copy_text(value: Any, label: str, *, maximum: int, required: bool = True) -> str:
    """Normalize exact copy once so the finishing renderer can preserve it verbatim."""

    result = _text(value, label, maximum=maximum, required=required)
    if not result:
        return ""
    normalized = "\n".join(" ".join(line.split()) for line in result.splitlines())
    if len(normalized) > maximum:
        raise ValueError(f"{label} exceeds {maximum} characters after whitespace normalization.")
    return normalized


def _finite(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{label} must be from {minimum:g} to {maximum:g}.")
    return number


def _frame_aligned_seconds(value: float, label: str) -> float:
    frames = value * ADVERTISEMENT_FPS
    if abs(frames - round(frames)) > 1.0e-6:
        raise ValueError(f"{label} must align exactly to the {ADVERTISEMENT_FPS} fps advertisement clock.")
    return value


def _schema_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / "schemas" / filename


def _validate_schema(payload: Mapping[str, Any], filename: str, label: str) -> None:
    try:
        from jsonschema import Draft202012Validator
    except Exception as exc:  # pragma: no cover - dependency is declared by the package.
        raise RuntimeError("Advertisement contracts require jsonschema.") from exc
    schema = json.loads(_schema_path(filename).read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    path = "$" + "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}"
        for part in error.absolute_path
    )
    raise ValueError(f"{label} failed schema validation at {path}: {error.message}")


def _canonical_aspect(value: Any) -> str:
    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", str(value or ""))
    if not match:
        raise ValueError("aspect_ratio must be a supported W:H ratio.")
    aspect = f"{int(match.group(1))}:{int(match.group(2))}"
    if aspect not in ASPECT_RATIOS:
        raise ValueError("aspect_ratio must be one of " + ", ".join(ASPECT_RATIOS) + ".")
    return aspect


def _normalize_claims(claims_json: Any) -> tuple[list[dict[str, Any]], list[str]]:
    raw = _strict_json_array(claims_json, "claims_json")
    normalized: list[dict[str, Any]] = []
    approved: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"claims_json[{index}] must be an object.")
        unknown = set(item) - {"text", "status", "evidence"}
        if unknown:
            raise ValueError(
                f"claims_json[{index}] has unknown field(s): {', '.join(sorted(unknown))}."
            )
        claim = _text(item.get("text"), f"claims_json[{index}].text", maximum=300)
        status = str(item.get("status", "")).strip().casefold()
        if status not in {"approved", "draft", "prohibited"}:
            raise ValueError(
                f"claims_json[{index}].status must be approved, draft, or prohibited."
            )
        evidence = _text(
            item.get("evidence"),
            f"claims_json[{index}].evidence",
            maximum=1000,
            required=False,
        )
        if status == "approved" and not evidence:
            raise ValueError(
                f"claims_json[{index}] is approved but has no substantiation/evidence reference."
            )
        row = {"text": claim, "status": status, "evidence": evidence}
        normalized.append(row)
        if status == "approved":
            approved.append(claim)
    return normalized, approved


def _normalize_deliverables(deliverables_json: Any, master_aspect: str) -> dict[str, Any]:
    payload = _strict_json_object(deliverables_json, "deliverables_json")
    unknown = set(payload) - {"primary", "adaptations", "cutdowns_seconds"}
    if unknown:
        raise ValueError(
            "deliverables_json has unknown field(s): " + ", ".join(sorted(unknown)) + "."
        )
    primary = str(payload.get("primary", master_aspect)).strip()
    primary_match = re.search(r"\b(?:21:9|16:9|9:16|4:3|3:4|3:2|2:3|1:1)\b", primary)
    primary_aspect = primary_match.group(0) if primary_match else _canonical_aspect(primary)
    if primary_aspect != master_aspect:
        raise ValueError("deliverables_json.primary must match aspect_ratio.")
    adaptations: list[str] = []
    for index, item in enumerate(payload.get("adaptations", [])):
        aspect = _canonical_aspect(item)
        if aspect != master_aspect and aspect not in adaptations:
            adaptations.append(aspect)
    cutdowns: list[float] = []
    for index, item in enumerate(payload.get("cutdowns_seconds", [])):
        seconds = _finite(item, f"deliverables_json.cutdowns_seconds[{index}]", minimum=0.1, maximum=60.0)
        if seconds not in cutdowns:
            cutdowns.append(seconds)
    return {
        "primary": master_aspect,
        "adaptations": adaptations,
        "cutdowns_seconds": sorted(cutdowns, reverse=True),
        "render_status": "requested_not_rendered",
    }


def _annotation_fields(description: str) -> dict[str, str]:
    matches = _AD_ANNOTATION_RE.findall(str(description or ""))
    if len(matches) != 1:
        raise ValueError("Each advertisement reference row must contain exactly one [ad:...] annotation.")
    fields: dict[str, str] = {}
    for part in matches[0].split(";"):
        if "=" not in part:
            raise ValueError("Advertisement reference annotations use key=value fields separated by semicolons.")
        key, value = (token.strip().casefold() for token in part.split("=", 1))
        if not key or not value or key in fields:
            raise ValueError("Advertisement reference annotations contain an empty or duplicate field.")
        fields[key] = value
    required = {"role", "subject", "retention"}
    if set(fields) != required:
        raise ValueError("Advertisement reference annotations require exactly role, subject, and retention fields.")
    if not re.fullmatch(r"subject_[1-9]\d*", fields["subject"]):
        raise ValueError("Advertisement reference annotation subject must be subject_N.")
    return fields


def parse_advertisement_reference_manifest(manifest: Any) -> list[dict[str, Any]]:
    """Parse explicit reference semantics without inspecting prose keywords."""

    rows: list[dict[str, Any]] = []
    for line_index, raw in enumerate(str(manifest or "").splitlines(), start=1):
        if not raw.strip():
            continue
        match = _REFERENCE_ROW_RE.fullmatch(raw)
        if match is None:
            raise ValueError(f"reference_manifest line {line_index} is not a complete reference row.")
        tag = f"<{match.group(1).title()} {int(match.group(2))}>"
        fields = _annotation_fields(match.group(3))
        rows.append(
            {
                "tag": tag,
                "kind": match.group(1).casefold(),
                "ordinal": int(match.group(2)),
                **fields,
            }
        )
    if not rows:
        raise ValueError("reference_manifest has no advertisement reference rows.")
    tags = [row["tag"] for row in rows]
    if len(tags) != len(set(tags)):
        raise ValueError("reference_manifest contains duplicate reference tags.")
    return rows


class DiffusionGemmaAdvertisementWorkflowControls:
    """One authoritative control surface for settings shared across the saved graph."""

    # Advertisement v1 is a governed 30/15/6 deliverable system.  Other master
    # lengths need duration-aware campaign copy and arrangement presets before
    # they can be exposed without reintroducing split-setting failures.
    DURATION_PRESETS = ("30 seconds",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "master_duration": (list(cls.DURATION_PRESETS), {"default": "30 seconds"}),
                "native_shot_count": ("INT", {"default": 8, "min": 1, "max": 99}),
                "performance_mode": (
                    list(ADVERTISEMENT_PERFORMANCE_MODES),
                    {"default": ADVERTISEMENT_PERFORMANCE_MODES[0]},
                ),
            }
        }

    RETURN_TYPES = (
        "FLOAT",
        "INT",
        list(ADVERTISEMENT_PERFORMANCE_CARRIER_MODES),
        "FLOAT",
        "FLOAT",
        "STRING",
        list(ASPECT_RATIOS),
        list(H3_GENERATION_MODE_CARRIER),
        list(H3_SHOT_COUNT_CARRIER),
        "INT",
        list(H3_AUDIO_MODE_CARRIER),
        list(H3_DIALOGUE_MODE_CARRIER),
        "FLOAT",
        "STRING",
        "FLOAT",
        "STRING",
        "STRING",
        "BOOLEAN",
    )
    RETURN_NAMES = (
        "master_duration_seconds",
        "native_shot_count",
        "performance_mode",
        "cutdown_15_start_seconds",
        "cutdown_6_start_seconds",
        "deliverables_json",
        "master_aspect_ratio",
        "h3_generation_mode",
        "h3_shot_count_mode",
        "h3_shot_count_override",
        "h3_audio_mode",
        "h3_dialogue_mode",
        "excerpt_start_seconds",
        "generation_model",
        "max_h3_lane_seconds",
        "master_aspect_ratio_override",
        "status",
        "ready",
    )
    FUNCTION = "resolve"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Provides the single duration, native-shot, and performance authority used by every Advertisement "
        "consumer. Cutdown windows are derived from the master tail so governed end-card copy is retained."
    )

    def resolve(self, master_duration, native_shot_count, performance_mode):
        preset = str(master_duration)
        if preset not in self.DURATION_PRESETS:
            raise ValueError("master_duration is unsupported.")
        duration = float(preset.split(" ", 1)[0])
        shots = int(native_shot_count)
        if not 1 <= shots <= 99:
            raise ValueError("native_shot_count must be from 1 to 99.")
        mode = str(performance_mode)
        if mode not in ADVERTISEMENT_PERFORMANCE_MODES:
            raise ValueError("performance_mode is unsupported.")
        cutdown_15_start = duration - 15.0
        cutdown_6_start = duration - 6.0
        deliverables_json = _json(
            {
                "primary": "9:16",
                "adaptations": ["1:1", "16:9"],
                "cutdowns_seconds": [15, 6],
            }
        )
        return (
            duration,
            shots,
            mode,
            cutdown_15_start,
            cutdown_6_start,
            deliverables_json,
            "9:16",
            "ref2va",
            "custom",
            0,
            "auto_scene_audio",
            "off",
            0.0,
            "MiniMax H3 Ref2VA",
            15.0,
            "9:16",
            f"Advertisement controls locked: {duration:g}s, {shots} native shots, {mode}.",
            True,
        )


class DiffusionGemmaAdvertisementCampaignContract:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "brand_name": ("STRING", {"default": "", "multiline": False}),
                "product_name": ("STRING", {"default": "", "multiline": False}),
                "variant_name": ("STRING", {"default": "", "multiline": False}),
                "audience": ("STRING", {"default": "", "multiline": True}),
                "campaign_objective": ("STRING", {"default": "", "multiline": True}),
                "headline": ("STRING", {"default": "", "multiline": True}),
                "call_to_action": ("STRING", {"default": "", "multiline": True}),
                "claims_json": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": "Each claim is {text,status,evidence}. Only substantiated approved claims enter production guidance.",
                    },
                ),
                "production_duration_seconds": (
                    "FLOAT",
                    {"default": 30.0, "min": 1.0 / ADVERTISEMENT_FPS, "max": 60.0, "step": 1.0 / ADVERTISEMENT_FPS},
                ),
                "aspect_ratio": (list(ASPECT_RATIOS), {"default": "9:16"}),
                "end_card_duration_seconds": (
                    "FLOAT",
                    {"default": 3.0, "min": 1.0 / ADVERTISEMENT_FPS, "max": 10.0, "step": 1.0 / ADVERTISEMENT_FPS},
                ),
                "deliverables_json": (
                    "STRING",
                    {
                        "default": '{"primary":"9:16","adaptations":["1:1","16:9"],"cutdowns_seconds":[15,6]}',
                        "multiline": True,
                    },
                ),
            },
            "optional": {
                "campaign_name": ("STRING", {"default": "", "multiline": False}),
                "legal_line": ("STRING", {"default": "", "multiline": True}),
                "offer_text": ("STRING", {"default": "", "multiline": True}),
                "price_text": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = (
        "campaign_contract_json",
        "director_brief",
        "exact_copy_json",
        "contract_sha256",
        "status",
        "ready",
    )
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "Freeze typed ad identity, exact copy, governed claims, timing, end card, and requested deliverables without claiming that variants were rendered."

    def build(
        self,
        brand_name,
        product_name,
        variant_name,
        audience,
        campaign_objective,
        headline,
        call_to_action,
        claims_json,
        production_duration_seconds,
        aspect_ratio,
        end_card_duration_seconds,
        deliverables_json,
        campaign_name="",
        legal_line="",
        offer_text="",
        price_text="",
    ):
        brand = _copy_text(brand_name, "brand_name", maximum=120)
        product = _copy_text(product_name, "product_name", maximum=160)
        variant = _copy_text(variant_name, "variant_name", maximum=120, required=False)
        target_audience = _text(audience, "audience", maximum=1000)
        objective = _text(campaign_objective, "campaign_objective", maximum=1500)
        exact_headline = _copy_text(headline, "headline", maximum=300)
        cta = _copy_text(call_to_action, "call_to_action", maximum=240)
        legal = _copy_text(legal_line, "legal_line", maximum=1000, required=False)
        offer = _copy_text(offer_text, "offer_text", maximum=300, required=False)
        price = _copy_text(price_text, "price_text", maximum=120, required=False)
        campaign = _text(campaign_name, "campaign_name", maximum=160, required=False)
        duration = _frame_aligned_seconds(_finite(
            production_duration_seconds,
            "production_duration_seconds",
            minimum=1.0 / ADVERTISEMENT_FPS,
            maximum=60.0,
        ), "production_duration_seconds")
        end_duration = _frame_aligned_seconds(_finite(
            end_card_duration_seconds,
            "end_card_duration_seconds",
            minimum=1.0 / ADVERTISEMENT_FPS,
            maximum=10.0,
        ), "end_card_duration_seconds")
        if end_duration >= duration:
            raise ValueError("end_card_duration_seconds must be shorter than the production duration.")
        aspect = _canonical_aspect(aspect_ratio)
        claims, approved_claims = _normalize_claims(claims_json)
        deliverables = _normalize_deliverables(deliverables_json, aspect)
        if any(seconds >= duration for seconds in deliverables["cutdowns_seconds"]):
            raise ValueError("Every requested cutdown must be shorter than the primary duration.")

        exact_copy = {
            "brand_name": brand,
            "product_name": product,
            "variant_name": variant,
            "headline": exact_headline,
            "call_to_action": cta,
            "legal_line": legal,
            "offer_text": offer,
            "price_text": price,
            "approved_claims": approved_claims,
            "render_policy": "deterministic_finishing_overlay",
        }
        exact_copy_json = _json(exact_copy)
        payload = {
            "schema": CAMPAIGN_SCHEMA,
            "version": VERSION,
            "campaign_name": campaign or f"{brand} {product}",
            "brand": {
                "brand_name": brand,
                "product_name": product,
                "variant_name": variant,
            },
            "audience": target_audience,
            "objective": objective,
            "copy": exact_copy,
            "claims": claims,
            "timeline": {
                "duration_seconds": round(duration, 6),
                "aspect_ratio": aspect,
                "end_card": {
                    "start_seconds": round(duration - end_duration, 6),
                    "duration_seconds": round(end_duration, 6),
                    "render_status": "contracted_not_rendered",
                },
            },
            "deliverables": deliverables,
            "exact_copy_sha256": _sha_text(exact_copy_json),
            "ready": True,
        }
        _validate_schema(payload, "advertisement_campaign.schema.json", "campaign contract")
        encoded = _json(payload)
        digest = _sha_text(encoded)
        approved_text = (
            "\nApproved substantiated claims: " + " | ".join(approved_claims)
            if approved_claims
            else "\nNo product claim is authorized for generated scene text or narration."
        )
        director_brief = (
            f"ADVERTISEMENT — {payload['campaign_name']}\n"
            f"Brand/product: {brand} {product}{(' — ' + variant) if variant else ''}.\n"
            f"Audience: {target_audience}\nObjective: {objective}\n"
            f"Create a {duration:g}-second {aspect} visual story. Reserve the final "
            f"{end_duration:g} seconds for a deterministic end card added after generation. "
            "Do not invent readable packaging, headlines, prices, legal wording, offers, or claims inside generated video."
            f"{approved_text}\n"
            f"Finishing-only exact copy: headline={exact_headline!r}; CTA={cta!r}."
        )
        excluded = sum(row["status"] != "approved" for row in claims)
        status = (
            f"Advertisement campaign contract locked for {duration:g}s {aspect}; "
            f"{len(approved_claims)} approved claim(s), {excluded} draft/prohibited claim(s) excluded, "
            f"and {len(deliverables['adaptations']) + len(deliverables['cutdowns_seconds'])} requested variant(s) remain unrendered."
        )
        return (encoded, director_brief, exact_copy_json, digest, status, True)


class DiffusionGemmaAdvertisementReferenceContract:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "slot_policy": (
                    list(EXPORTED_REFERENCE_SLOT_POLICIES),
                    {"default": EXPORTED_REFERENCE_SLOT_POLICIES[0]},
                ),
                "performer_description": ("STRING", {"default": "", "multiline": True}),
                "product_description": ("STRING", {"default": "", "multiline": True}),
                "product_retention_attributes": ("STRING", {"default": "", "multiline": True}),
                "product_reference_kind": (list(PRODUCT_REFERENCE_KINDS), {"default": "product contact sheet"}),
            },
            "optional": {
                "performer_retention_attributes": ("STRING", {"default": "identity, face, hair, body, wardrobe", "multiline": True}),
                "package_copy_json": ("STRING", {"default": "[]", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = (
        "reference_contract_json",
        "reference_manifest",
        "expected_subject_count",
        "performer_picture_count",
        "product_picture_tag",
        "continuity_picture_tag",
        "status",
        "ready",
    )
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "Assign explicit performer, product/package, and optional relay roles to H3 Picture slots without natural-language wording passwords."

    def build(
        self,
        slot_policy,
        performer_description,
        product_description,
        product_retention_attributes,
        product_reference_kind,
        performer_retention_attributes="identity, face, hair, body, wardrobe",
        package_copy_json="[]",
    ):
        policy = str(slot_policy)
        if policy not in REFERENCE_SLOT_POLICIES:
            raise ValueError("slot_policy is unsupported.")
        performer = _text(performer_description, "performer_description", maximum=2000)
        product = _text(product_description, "product_description", maximum=2000)
        performer_retention = _text(
            performer_retention_attributes,
            "performer_retention_attributes",
            maximum=1000,
        )
        product_retention = _text(
            product_retention_attributes,
            "product_retention_attributes",
            maximum=1000,
        )
        reference_kind = str(product_reference_kind)
        if reference_kind not in PRODUCT_REFERENCE_KINDS:
            raise ValueError("product_reference_kind is unsupported.")
        raw_package_copy = _strict_json_array(package_copy_json, "package_copy_json")
        package_copy: list[str] = []
        for index, item in enumerate(raw_package_copy):
            value = _text(item, f"package_copy_json[{index}]", maximum=160)
            if value not in package_copy:
                package_copy.append(value)

        two_performer = policy != "1 performer + 1 product + relay"
        relay_enabled = policy != "2 performer + 1 product; relay off"
        performer_tags = ["<Picture 1>", "<Picture 2>"] if two_performer else ["<Picture 1>"]
        product_tag = "<Picture 3>" if two_performer else "<Picture 2>"
        relay_tag = (
            ("<Picture 4>" if two_performer else "<Picture 3>")
            if relay_enabled
            else ""
        )
        slots: list[dict[str, Any]] = []
        manifest_rows: list[str] = []
        for index, tag in enumerate(performer_tags):
            role = "performer_primary" if index == 0 else "performer_supplemental"
            slots.append(
                {
                    "picture_tag": tag,
                    "role": role,
                    "subject_tag": "<Subject 1>",
                    "retention": "fully_preserved",
                    "description": performer,
                    "non_transfer": [] if index == 0 else ["layout", "background", "pose sequence"],
                }
            )
            categories = "identity,appearance,composition" if index == 0 else "identity,appearance"
            manifest_rows.append(
                f"{tag}: [dg:{categories}] [ad:role={role};subject=subject_1;retention=fully_preserved] "
                + performer
            )

        product_non_transfer = (
            ["grid", "seams", "backgrounds", "panel multiplicity", "viewpoint sequence"]
            if reference_kind == "product contact sheet"
            else []
        )
        slots.append(
            {
                "picture_tag": product_tag,
                "role": "product_package",
                "subject_tag": "<Subject 2>",
                "retention": "fully_preserved",
                "description": product,
                "reference_kind": reference_kind.replace(" ", "_"),
                "non_transfer": product_non_transfer,
            }
        )
        non_transfer_text = (
            " The contact-sheet grid, seams, backgrounds, repeated panels, and viewpoint sequence do not transfer."
            if product_non_transfer
            else ""
        )
        manifest_rows.append(
            f"{product_tag}: [dg:object,color,text,appearance] "
            "[ad:role=product_package;subject=subject_2;retention=fully_preserved] "
            f"{product}{non_transfer_text}"
        )
        manifest = "\n".join(manifest_rows)
        parsed_roles = parse_advertisement_reference_manifest(manifest)
        if [row["role"] for row in parsed_roles] != [slot["role"] for slot in slots]:
            raise ValueError("Internal advertisement reference-role serialization is inconsistent.")

        payload = {
            "schema": REFERENCE_SCHEMA,
            "version": VERSION,
            "slot_policy": (
                "performer_2_product_1_relay_1"
                if two_performer and relay_enabled
                else (
                    "performer_1_product_1_relay_1"
                    if relay_enabled
                    else "performer_2_product_1_relay_off"
                )
            ),
            "subjects": [
                {
                    "subject_tag": "<Subject 1>",
                    "role": "performer",
                    "description": performer,
                    "retention_attributes": performer_retention,
                },
                {
                    "subject_tag": "<Subject 2>",
                    "role": "product_package",
                    "description": product,
                    "retention_attributes": product_retention,
                    "exact_package_copy": package_copy,
                    "copy_fidelity_policy": "qa_required_not_guaranteed_by_generation",
                },
            ],
            "picture_slots": slots,
            "relay": {
                "enabled": bool(relay_tag),
                "picture_tag": relay_tag,
                "role": "previous_lane_pose_spatial_lighting_motion_only_not_identity_or_product",
                "manifest_reservation": (
                    f"{relay_tag}: [dg:action,motion,camera,temporal,spatial,lighting] "
                    "[ad:role=continuity_relay;subject=subject_1;retention=continuity_only] "
                    "Reserved for the immediately previous lane tail; never performer identity or product/package authority."
                    if relay_tag
                    else ""
                ),
            },
            "reference_manifest": manifest,
            "reference_manifest_sha256": _sha_text(manifest),
            "expected_subject_count": 2,
            "ready": True,
        }
        _validate_schema(payload, "advertisement_reference.schema.json", "reference contract")
        encoded = _json(payload)
        status = (
            f"Advertisement references reserve {len(performer_tags)} performer Picture(s), "
            f"{product_tag} for the independent product package, and "
            + (f"{relay_tag} for later-lane continuity." if relay_tag else "no relay slot.")
        )
        return (
            encoded,
            manifest,
            2,
            len(performer_tags),
            product_tag,
            relay_tag,
            status,
            True,
        )


class DiffusionGemmaAdvertisementReferenceAssetPrep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "performer_hero_image": (
                    "IMAGE",
                    {"tooltip": "Picture 1: primary performer identity, body, wardrobe, and composition authority."},
                ),
                "performer_sheet_image": (
                    "IMAGE",
                    {"tooltip": "Picture 2: same-performer supplemental identity sheet. It never carries product authority."},
                ),
                "product_sheet_image": (
                    "IMAGE",
                    {"tooltip": "Picture 3: independent product/package reference or product-only contact sheet."},
                ),
                "generation_width": ("INT", {"default": 480, "min": 256, "max": 8192, "step": 8}),
                "generation_height": ("INT", {"default": 864, "min": 256, "max": 8192, "step": 8}),
                "performer_area_ratio": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.25,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Combined pixel-area budget for Pictures 1 and 2 relative to one generation frame.",
                    },
                ),
                "performer_hero_share": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.2,
                        "max": 0.8,
                        "step": 0.05,
                        "tooltip": "Share of the performer-only budget assigned to Picture 1; Picture 2 receives the remainder.",
                    },
                ),
                "product_area_ratio": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 1.5,
                        "step": 0.05,
                        "tooltip": "Independent Picture 3 product budget relative to one generation frame; it is never borrowed by performer assets.",
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "STRING",
        "STRING",
        "STRING",
        "IMAGE",
        "STRING",
        "STRING",
        "BOOLEAN",
    )
    RETURN_NAMES = (
        "performer_hero_image",
        "performer_sheet_image",
        "product_sheet_image",
        "performer_hero_sha256",
        "performer_sheet_sha256",
        "product_sheet_sha256",
        "director_reference_batch",
        "asset_manifest_json",
        "status",
        "ready",
    )
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = "Downscale-only prep and deterministic hashing for separate performer hero, same-performer sheet, and product sheet roles, plus a padded Director-analysis batch."

    def prepare(
        self,
        performer_hero_image,
        performer_sheet_image,
        product_sheet_image,
        generation_width,
        generation_height,
        performer_area_ratio,
        performer_hero_share,
        product_area_ratio,
    ):
        width = int(generation_width)
        height = int(generation_height)
        if width < 256 or height < 256 or width > 8192 or height > 8192:
            raise ValueError("generation_width and generation_height must be from 256 to 8192.")
        performer_ratio = _finite(
            performer_area_ratio, "performer_area_ratio", minimum=0.25, maximum=2.0
        )
        hero_share = _finite(
            performer_hero_share, "performer_hero_share", minimum=0.2, maximum=0.8
        )
        product_ratio = _finite(
            product_area_ratio, "product_area_ratio", minimum=0.1, maximum=1.5
        )
        source_images = (
            _image_tensor(performer_hero_image, "performer_hero_image"),
            _image_tensor(performer_sheet_image, "performer_sheet_image"),
            _image_tensor(product_sheet_image, "product_sheet_image"),
        )
        source_hashes = tuple(_image_content_sha256(image) for image in source_images)
        if len(set(source_hashes)) != 3:
            raise ValueError(
                "Performer hero, performer sheet, and product sheet must be three distinct source images."
            )
        frame_area = float(width * height)
        performer_area = frame_area * performer_ratio
        budgets = (
            performer_area * hero_share,
            performer_area * (1.0 - hero_share),
            frame_area * product_ratio,
        )
        prepared = tuple(
            _budgeted_image(image, maximum_area=budget)
            for image, budget in zip(source_images, budgets)
        )
        roles = ("performer_hero", "performer_sheet", "product_sheet")
        prepared_role_hashes = tuple(
            _image_sha256(image, role) for image, role in zip(prepared, roles)
        )
        director_batch = torch.cat(
            [_director_canvas(image, width, height) for image in prepared], dim=0
        )
        assets = []
        for index, (role, image, source_digest, prepared_digest, budget) in enumerate(
            zip(roles, prepared, source_hashes, prepared_role_hashes, budgets), start=1
        ):
            assets.append(
                {
                    "picture_tag": f"<Picture {index}>",
                    "role": role,
                    "sha256": source_digest,
                    "prepared_role_sha256": prepared_digest,
                    "width": int(image.shape[2]),
                    "height": int(image.shape[1]),
                    "pixel_area_budget": int(math.floor(budget)),
                    "upscaled": False,
                }
            )
        manifest = {
            "schema": "diffusiongemma.advertisement_reference_assets",
            "version": 1,
            "generation_width": width,
            "generation_height": height,
            "performer_budget": {
                "area_ratio": performer_ratio,
                "hero_share": hero_share,
                "roles": ["performer_hero", "performer_sheet"],
            },
            "product_budget": {
                "area_ratio": product_ratio,
                "independent_from_performer_budget": True,
                "role": "product_sheet",
            },
            "assets": assets,
            "director_batch": {
                "count": 3,
                "width": width,
                "height": height,
                "aspect_policy": "preserve_then_neutral_pad",
                "role_order": list(roles),
            },
            "identity_product_combined_sheet": False,
            "ready": True,
        }
        _validate_schema(
            manifest,
            "advertisement_reference_assets.schema.json",
            "advertisement reference assets",
        )
        status = (
            "Prepared and hash-locked Picture 1 performer hero, Picture 2 same-performer sheet, "
            "and Picture 3 independent product sheet; the product received a separate pixel budget."
        )
        return (
            prepared[0],
            prepared[1],
            prepared[2],
            source_hashes[0],
            source_hashes[1],
            source_hashes[2],
            director_batch,
            _json(manifest),
            status,
            True,
        )


def _normalize_tagged_lyrics(value: Any, effective_mode: str) -> str:
    lyrics = str(value or "").strip()
    if effective_mode == "instrumental":
        if lyrics and lyrics.casefold() not in {"instrumental", "[instrumental]"}:
            raise ValueError("Instrumental soundtrack mode cannot contain sung lyrics.")
        # Music 3 may emit an early end token when given only a single
        # instrumental sentinel, even when max_duration is substantially
        # longer. A sparse, wordless section scaffold preserves the strict
        # no-vocal contract while giving the duration model enough structure
        # to realize a commercial-length intro, development, and outro.
        return "[Intro]\n\n[Instrumental]\n\n[Bridge]\n\n[Instrumental]\n\n[Outro]"
    if not lyrics or lyrics.casefold() in {"instrumental", "[instrumental]"}:
        raise ValueError("Vocal soundtrack mode requires real lyrics.")
    if not _LYRIC_SECTION_RE.search(lyrics):
        lyrics = "[Verse]\n" + lyrics
    if len(lyrics) > 4000:
        raise ValueError("tagged lyrics exceed 4000 characters.")
    return lyrics


class DiffusionGemmaAdvertisementSoundtrackContract:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_mode": (list(SOUNDTRACK_CONTENT_MODES), {"default": "Instrumental"}),
                "genre_style": ("STRING", {"default": "", "multiline": True}),
                "mood": ("STRING", {"default": "", "multiline": True}),
                "instrumentation": ("STRING", {"default": "", "multiline": True}),
                "target_duration_seconds": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 0.1,
                        "max": 60.0,
                        "step": 0.1,
                        "tooltip": "Selectable advertisement master duration. Music3 generation receives five seconds of selection headroom when the 60-second cap allows it.",
                    },
                ),
                "bpm": ("FLOAT", {"default": 105.0, "min": 0.0, "max": 300.0, "step": 0.1}),
                "time_signature": (list(TIME_SIGNATURES), {"default": "4"}),
                "language": (list(LANGUAGES), {"default": "unknown"}),
                "lyrics": ("STRING", {"default": "", "multiline": True}),
                "voice_over_policy": (
                    list(EXPORTED_VOICE_OVER_POLICIES),
                    {"default": EXPORTED_VOICE_OVER_POLICIES[0]},
                ),
            },
            "optional": {
                "arrangement_notes": ("STRING", {"default": "", "multiline": True}),
                "do_not_sound_like": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "FLOAT", "FLOAT", "STRING", "BOOLEAN")
    RETURN_NAMES = (
        "soundtrack_contract_json",
        "music3_caption",
        "tagged_lyrics",
        "effective_content_mode",
        "expected_bpm",
        "generation_duration_seconds",
        "status",
        "ready",
    )
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "Create a structured Music3/ACE-style caption and tagged lyrics while keeping soundtrack vocals distinct from non-diegetic advertisement VO."

    def build(
        self,
        content_mode,
        genre_style,
        mood,
        instrumentation,
        target_duration_seconds,
        bpm,
        time_signature,
        language,
        lyrics,
        voice_over_policy,
        arrangement_notes="",
        do_not_sound_like="",
    ):
        requested = str(content_mode)
        if requested not in SOUNDTRACK_CONTENT_MODES:
            raise ValueError("content_mode is unsupported.")
        style = _text(genre_style, "genre_style", maximum=1000)
        mood_text = _text(mood, "mood", maximum=600)
        instruments = _text(instrumentation, "instrumentation", maximum=1000)
        arrangement = _text(arrangement_notes, "arrangement_notes", maximum=1500, required=False)
        exclusions = _text(do_not_sound_like, "do_not_sound_like", maximum=600, required=False)
        master_duration = _finite(
            target_duration_seconds,
            "target_duration_seconds",
            minimum=0.1,
            maximum=60.0,
        )
        generation_duration = min(60.0, master_duration + 5.0)
        selection_headroom = generation_duration - master_duration
        expected_bpm = _finite(bpm, "bpm", minimum=0.0, maximum=300.0)
        meter = str(time_signature)
        if meter not in TIME_SIGNATURES:
            raise ValueError("time_signature is unsupported.")
        lang = str(language)
        if lang not in LANGUAGES:
            raise ValueError("language is unsupported.")
        vo_policy = str(voice_over_policy)
        if vo_policy not in VOICE_OVER_POLICIES:
            raise ValueError("voice_over_policy is unsupported.")
        raw_lyrics = str(lyrics or "").strip()
        effective = (
            "vocal"
            if requested == "Vocal" or (requested == "Auto" and raw_lyrics.casefold() not in {"", "instrumental", "[instrumental]"})
            else "instrumental"
        )
        tagged = _normalize_tagged_lyrics(raw_lyrics, effective)
        vocal_sentence = (
            f"Vocal content: sung lyrics in language {lang}."
            if effective == "vocal"
            else "Vocal content: instrumental, with no intelligible sung or spoken words."
        )
        tempo_sentence = (
            f"Tempo: {expected_bpm:g} BPM in {meter}/4."
            if expected_bpm > 0.0
            else f"Tempo: signal-selected in {meter}/4."
        )
        caption_parts = [
            "Global Metadata:",
            f"Generation duration: {generation_duration:g} seconds. Selectable advertisement master: {master_duration:g} seconds. Selection headroom: {selection_headroom:g} seconds.",
            f"Style: {style}. Mood: {mood_text}. {tempo_sentence}",
            "Vocal Details:",
            vocal_sentence,
            "Arrangement:",
            f"Instrumentation: {instruments}.",
            (
                arrangement
                if arrangement
                else "Build one complete ad-ready musical arc inside the selectable master and keep clean edit points for picture synchronization."
            ),
        ]
        if exclusions:
            caption_parts.append(f"Avoid: {exclusions}.")
        caption = " ".join(caption_parts)
        payload = {
            "schema": SOUNDTRACK_SCHEMA,
            "version": VERSION,
            "requested_content_mode": requested.casefold(),
            "effective_content_mode": effective,
            "music3": {
                "caption": caption,
                "tagged_lyrics": tagged,
                "bpm": expected_bpm,
                "time_signature": meter,
                "language": lang,
                "target_duration_seconds": master_duration,
                "generation_duration_seconds": generation_duration,
                "selection_headroom_seconds": selection_headroom,
            },
            "voice_over": {
                "policy": "separate_non_diegetic" if vo_policy == VOICE_OVER_POLICIES[1] else "none",
                "included_in_motion_guide": False,
                "included_in_final_mix": vo_policy == VOICE_OVER_POLICIES[1],
            },
            "content_evidence_policy": (
                "vocal_proxy_required"
                if effective == "vocal"
                else "vocal_proxy_not_required"
            ),
            "ready": True,
        }
        _validate_schema(payload, "advertisement_soundtrack.schema.json", "soundtrack contract")
        encoded = _json(payload)
        status = (
            f"Advertisement soundtrack locked as {effective}; Music3 will generate {generation_duration:g}s "
            f"for a {master_duration:g}s selectable advertisement master. "
            + (
                "Voice-over remains a separate final-mix input and is excluded from the motion guide."
                if vo_policy == VOICE_OVER_POLICIES[1]
                else "No voice-over is contracted."
            )
        )
        return (
            encoded,
            caption,
            tagged,
            effective,
            expected_bpm,
            generation_duration,
            status,
            True,
        )


class DiffusionGemmaAdvertisementMusic3PromptAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "soundtrack_contract_json": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                )
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT", "FLOAT", "INT", "STRING", "BOOLEAN")
    RETURN_NAMES = (
        "caption",
        "tagged_lyrics",
        "content_mode",
        "bpm",
        "generation_duration_seconds",
        "ace_bpm_integer",
        "status",
        "ready",
    )
    FUNCTION = "adapt"
    CATEGORY = CATEGORY
    DESCRIPTION = "Validate and expose only the structured Music3 caption and tagged lyrics from an advertisement soundtrack contract."

    def adapt(self, soundtrack_contract_json):
        payload = _strict_json_object(soundtrack_contract_json, "soundtrack_contract_json")
        _validate_schema(payload, "advertisement_soundtrack.schema.json", "soundtrack contract")
        if payload.get("schema") != SOUNDTRACK_SCHEMA or payload.get("version") != VERSION or not payload.get("ready"):
            raise ValueError("soundtrack_contract_json is not a ready supported advertisement soundtrack contract.")
        music = payload["music3"]
        bpm = float(music["bpm"])
        ace_bpm = int(round(bpm))
        if ace_bpm < 10:
            # Zero means signal-selected/unspecified to the shared soundtrack
            # contract, but the optional ACE encoder requires an integer in
            # [10, 300].  Keep the default Music3/upload paths independent and
            # give the explicitly selected ACE fallback a neutral valid tempo.
            ace_bpm = 120
        if ace_bpm > 300:
            raise ValueError("Music3 BPM exceeds the ACE fallback integer carrier.")
        status = (
            f"Music3 prompt adapter emitted the governed {payload['effective_content_mode']} caption without changing it; "
            f"ACE fallback BPM carrier={ace_bpm}."
        )
        return (
            str(music["caption"]),
            str(music["tagged_lyrics"]),
            str(payload["effective_content_mode"]),
            bpm,
            float(music["generation_duration_seconds"]),
            ace_bpm,
            status,
            True,
        )


NODE_CLASS_MAPPINGS = {
    "DiffusionGemmaAdvertisementWorkflowControls": DiffusionGemmaAdvertisementWorkflowControls,
    "DiffusionGemmaAdvertisementCampaignContract": DiffusionGemmaAdvertisementCampaignContract,
    "DiffusionGemmaAdvertisementReferenceContract": DiffusionGemmaAdvertisementReferenceContract,
    "DiffusionGemmaAdvertisementReferenceAssetPrep": DiffusionGemmaAdvertisementReferenceAssetPrep,
    "DiffusionGemmaAdvertisementSoundtrackContract": DiffusionGemmaAdvertisementSoundtrackContract,
    "DiffusionGemmaAdvertisementMusic3PromptAdapter": DiffusionGemmaAdvertisementMusic3PromptAdapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionGemmaAdvertisementWorkflowControls": "DiffusionGemma Advertisement Workflow Controls",
    "DiffusionGemmaAdvertisementCampaignContract": "DiffusionGemma Advertisement Campaign Contract",
    "DiffusionGemmaAdvertisementReferenceContract": "DiffusionGemma Advertisement Reference Contract",
    "DiffusionGemmaAdvertisementReferenceAssetPrep": "DiffusionGemma Advertisement Reference Asset Prep & Hash",
    "DiffusionGemmaAdvertisementSoundtrackContract": "DiffusionGemma Advertisement Soundtrack Contract",
    "DiffusionGemmaAdvertisementMusic3PromptAdapter": "DiffusionGemma Advertisement Music3 Prompt Adapter",
}


__all__ = [
    "CAMPAIGN_SCHEMA",
    "REFERENCE_SCHEMA",
    "SOUNDTRACK_SCHEMA",
    "ADVERTISEMENT_PERFORMANCE_MODES",
    "ADVERTISEMENT_PERFORMANCE_CARRIER_MODES",
    "DiffusionGemmaAdvertisementWorkflowControls",
    "DiffusionGemmaAdvertisementCampaignContract",
    "DiffusionGemmaAdvertisementReferenceContract",
    "DiffusionGemmaAdvertisementReferenceAssetPrep",
    "DiffusionGemmaAdvertisementSoundtrackContract",
    "DiffusionGemmaAdvertisementMusic3PromptAdapter",
    "parse_advertisement_reference_manifest",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
