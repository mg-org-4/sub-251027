# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

"""Deterministic assembly, delivery, and QA nodes for advertisement workflows.

The implementation is deliberately separate from the established music-video
nodes.  It accepts only the versioned advertisement planning contracts and
does not alter any legacy output or saved-workflow interface.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont

try:
    from .advertising_planning_nodes import (
        ADVERTISEMENT_MASTER_SCHEMA,
        ADVERTISEMENT_PLAN_SCHEMA,
        H3_FPS,
        VERSION,
        _contract,
        _json,
        _plan,
        _tensor_sha256,
    )
    from .audio_production_nodes import waveform_sha256
except ImportError:  # Standalone repository tests.
    from advertising_planning_nodes import (
        ADVERTISEMENT_MASTER_SCHEMA,
        ADVERTISEMENT_PLAN_SCHEMA,
        H3_FPS,
        VERSION,
        _contract,
        _json,
        _plan,
        _tensor_sha256,
    )
    from audio_production_nodes import waveform_sha256


CATEGORY = "prompt/diffusiongemma/advertising"
ADVERTISEMENT_ASSEMBLY_SCHEMA = "diffusiongemma.advertisement_master_assembly"
ADVERTISEMENT_END_CARD_SCHEMA = "diffusiongemma.advertisement_end_card"
ADVERTISEMENT_DELIVERY_SCHEMA = "diffusiongemma.advertisement_delivery_manifest"
ADVERTISEMENT_ADAPTATION_SCHEMA = "diffusiongemma.advertisement_adaptation_status"
ADVERTISEMENT_MEDIA_QA_SCHEMA = "diffusiongemma.advertisement_media_qa"
ASPECT_RE = re.compile(r"^(?:21:9|16:9|9:16|4:3|3:4|3:2|2:3|1:1)$")
QA_STATES = ("pass", "fail", "not_measured")
MAX_END_CARD_TENSOR_BYTES = 2 * 1024 * 1024 * 1024
MANDATORY_MEDIA_QA_CHECKS = (
    "product_identity",
    "performer_identity",
    "copy_legibility",
    "audio_sync",
    "technical_integrity",
)


def _parse_object(value: Any, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def _rgb_images(value: Any, label: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim != 4 or int(tensor.shape[0]) < 1 or int(tensor.shape[-1]) != 3:
        raise ValueError(f"{label} must be a non-empty ComfyUI RGB IMAGE batch.")
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{label} contains non-finite pixels.")
    return tensor


def _audio_parts(audio: Any, label: str = "audio") -> tuple[torch.Tensor, int]:
    if not isinstance(audio, Mapping):
        raise ValueError(f"{label} must be a ComfyUI AUDIO mapping.")
    waveform = audio.get("waveform")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError(f"{label} waveform must have shape [batch, channels, samples].")
    try:
        sample_rate = int(audio.get("sample_rate", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} sample_rate must be a positive integer.") from exc
    if sample_rate <= 0 or int(waveform.shape[-1]) < 1:
        raise ValueError(f"{label} must contain samples at a positive sample rate.")
    if not torch.isfinite(waveform).all():
        raise ValueError(f"{label} contains non-finite samples.")
    return waveform, sample_rate


def _validate_identity(contract: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != ADVERTISEMENT_PLAN_SCHEMA or plan.get("version") != VERSION:
        raise ValueError("advertisement_plan_json has an unsupported schema or version.")
    if str(plan.get("campaign_id", "")) != str(contract.get("campaign_id", "")):
        raise ValueError("Advertisement plan and master contract campaign identities disagree.")
    if str(plan.get("advertisement_contract_sha256", "")) != str(contract.get("contract_sha256", "")):
        raise ValueError("Advertisement plan was compiled from a different master contract revision.")
    if abs(float(plan.get("production_duration_seconds", -1.0)) - float(contract["production_duration_seconds"])) > 1.0e-6:
        raise ValueError("Advertisement plan and master contract durations disagree.")
    if int(plan.get("fps", 0)) != int(contract.get("fps", 0)):
        raise ValueError("Advertisement plan and master contract frame rates disagree.")


def _expected_master_frames(contract: Mapping[str, Any]) -> int:
    fps = int(contract.get("fps", H3_FPS))
    if fps != H3_FPS:
        raise ValueError(f"Advertisement H3 masters must use exactly {H3_FPS} fps.")
    return int(round(float(contract["production_duration_seconds"]) * fps))


def _validate_pixel_aspect(
    contract: Mapping[str, Any], width: int, height: int
) -> dict[str, Any]:
    expected = str(contract.get("master_aspect_ratio", ""))
    if not ASPECT_RE.fullmatch(expected):
        raise ValueError("Advertisement contract has an unsupported master aspect ratio.")
    if width <= 0 or height <= 0:
        raise ValueError("Advertisement master pixel dimensions must be positive.")
    numerator, denominator = (int(value) for value in expected.split(":"))
    expected_ratio = numerator / denominator
    actual_ratio = width / height
    relative_error = abs(actual_ratio - expected_ratio) / expected_ratio
    # H3's native dimensions are rounded to model-compatible multiples.  The
    # 480x864 9:16 preset differs by ~1.2%, so permit that quantization while
    # rejecting a genuinely different shape such as 1:1 or 16:9.
    tolerance = 0.04
    if relative_error > tolerance:
        raise ValueError(
            f"Advertisement master pixels are {width}:{height}, which does not match "
            f"the contracted {expected} aspect ratio within {tolerance:.0%}."
        )
    return {
        "contracted_aspect_ratio": expected,
        "pixel_width": width,
        "pixel_height": height,
        "actual_ratio": round(actual_ratio, 9),
        "relative_error": round(relative_error, 9),
        "tolerance": tolerance,
        "matched": True,
    }


def _verify_audio(contract: Mapping[str, Any], audio: Any, *, exact_duration: bool) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = _audio_parts(audio, "master_audio")
    expected_hash = str(contract.get("audio_lock", {}).get("waveform_sha256", ""))
    actual_hash = waveform_sha256(audio)
    if actual_hash != expected_hash:
        raise ValueError("Master audio SHA-256 does not match the advertisement campaign lock.")
    if exact_duration:
        expected_samples = int(round(float(contract["production_duration_seconds"]) * sample_rate))
        if int(waveform.shape[-1]) != expected_samples:
            raise ValueError(
                f"Master audio must contain exactly {expected_samples} samples for this campaign; "
                f"received {int(waveform.shape[-1])}."
            )
    return waveform, sample_rate


def _manifest_sha(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json(manifest).encode("utf-8")).hexdigest()


def _verify_embedded_hash(
    payload: Mapping[str, Any], field: str, label: str
) -> str:
    supplied = str(payload.get(field, "")).strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", supplied):
        raise ValueError(f"{label} has no valid {field}.")
    identity = dict(payload)
    identity.pop(field, None)
    if _manifest_sha(identity) != supplied:
        raise ValueError(f"{label} {field} does not match its canonical content.")
    return supplied


class AdvertisementMasterAssembler:
    """Assemble retained H3 lane frames under exact campaign/audio identity."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "advertisement_contract_json": ("STRING", {"forceInput": True, "multiline": True}),
                "advertisement_plan_json": ("STRING", {"forceInput": True, "multiline": True}),
                "master_audio": ("AUDIO",),
            },
            "optional": {
                "lane_1_images": ("IMAGE", {"lazy": True}),
                "lane_2_images": ("IMAGE", {"lazy": True}),
                "lane_3_images": ("IMAGE", {"lazy": True}),
                "lane_4_images": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("assembled_master", "locked_master_audio", "assembly_report_json", "status", "ready")
    FUNCTION = "assemble"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Trims each H3 lane to its exact retained-frame contract and assembles one campaign- and "
        "audio-locked master. Extra H3 padding frames are never allowed into the delivery master."
    )

    @staticmethod
    def _required_inputs(advertisement_plan_json: str) -> list[str]:
        plan = _plan(advertisement_plan_json)
        return [f"lane_{index}_images" for index in range(1, len(plan["lanes"]) + 1)]

    def check_lazy_status(
        self,
        advertisement_contract_json: str,
        advertisement_plan_json: str,
        master_audio: Any,
        lane_1_images: Any = None,
        lane_2_images: Any = None,
        lane_3_images: Any = None,
        lane_4_images: Any = None,
    ) -> list[str]:
        supplied = (lane_1_images, lane_2_images, lane_3_images, lane_4_images)
        required = self._required_inputs(advertisement_plan_json)
        return [name for index, name in enumerate(required) if supplied[index] is None]

    def assemble(
        self,
        advertisement_contract_json: str,
        advertisement_plan_json: str,
        master_audio: Any,
        lane_1_images: Any = None,
        lane_2_images: Any = None,
        lane_3_images: Any = None,
        lane_4_images: Any = None,
    ):
        contract = _contract(advertisement_contract_json)
        plan = _plan(advertisement_plan_json)
        _validate_identity(contract, plan)
        _verify_audio(contract, master_audio, exact_duration=True)
        supplied = (lane_1_images, lane_2_images, lane_3_images, lane_4_images)
        lane_count = len(plan["lanes"])
        if not 1 <= lane_count <= 4:
            raise ValueError("Advertisement assembly requires from one to four generation lanes.")
        retained: list[torch.Tensor] = []
        lane_reports: list[dict[str, Any]] = []
        dimensions: tuple[int, int] | None = None
        running_frame = 0
        for index, lane in enumerate(plan["lanes"]):
            images = supplied[index]
            if images is None:
                raise ValueError(f"Advertisement lane {index + 1} images are missing.")
            tensor = _rgb_images(images, f"lane_{index + 1}_images")
            current_dimensions = (int(tensor.shape[1]), int(tensor.shape[2]))
            if dimensions is None:
                dimensions = current_dimensions
            elif current_dimensions != dimensions:
                raise ValueError("All advertisement H3 lanes must have identical pixel dimensions.")
            frame_count = int(lane.get("retained_master_frames", 0))
            if frame_count <= 0 or int(tensor.shape[0]) < frame_count:
                raise ValueError(
                    f"Advertisement lane {index + 1} needs {frame_count} retained frames, "
                    f"but received {int(tensor.shape[0])}."
                )
            expected_start = int(lane.get("master_frame_start", -1))
            expected_end = int(lane.get("master_frame_end_exclusive", -1))
            if expected_start != running_frame or expected_end - expected_start != frame_count:
                raise ValueError(f"Advertisement lane {index + 1} has a non-contiguous master-frame contract.")
            segment = tensor[:frame_count]
            retained.append(segment)
            lane_reports.append(
                {
                    "lane_index": index + 1,
                    "received_frames": int(tensor.shape[0]),
                    "retained_frames": frame_count,
                    "discarded_padding_frames": int(tensor.shape[0]) - frame_count,
                    "master_frame_start": expected_start,
                    "master_frame_end_exclusive": expected_end,
                }
            )
            running_frame = expected_end
        master = torch.cat(retained, dim=0)
        expected_frames = _expected_master_frames(contract)
        if running_frame != expected_frames or int(master.shape[0]) != expected_frames:
            raise ValueError(
                f"Advertisement assembly must contain exactly {expected_frames} frames; "
                f"the lane plan produced {int(master.shape[0])}."
            )
        pixel_aspect_evidence = _validate_pixel_aspect(
            contract, int(master.shape[2]), int(master.shape[1])
        )
        report: dict[str, Any] = {
            "schema": ADVERTISEMENT_ASSEMBLY_SCHEMA,
            "version": VERSION,
            "campaign_id": contract["campaign_id"],
            "advertisement_contract_sha256": contract["contract_sha256"],
            "advertisement_plan_sha256": plan["plan_sha256"],
            "master_audio_sha256": waveform_sha256(master_audio),
            "fps": H3_FPS,
            "exact_master_frames": expected_frames,
            "pixel_dimensions": [int(master.shape[2]), int(master.shape[1])],
            "pixel_aspect_evidence": pixel_aspect_evidence,
            "assembled_pixel_sha256": _tensor_sha256(master),
            "lanes": lane_reports,
            "ready": True,
        }
        report["report_sha256"] = _manifest_sha(report)
        return (
            master,
            master_audio,
            _json(report),
            f"Assembled exact {expected_frames}-frame advertisement master from {lane_count} H3 lanes.",
            True,
        )


def _color(value: str, label: str) -> tuple[int, int, int]:
    try:
        rgb = ImageColor.getrgb(str(value or ""))
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid CSS color or #RRGGBB value.") from exc
    if len(rgb) != 3:
        raise ValueError(f"{label} must be an opaque RGB color.")
    return tuple(int(component) for component in rgb)


def _font(size: int, *, bold: bool) -> ImageFont.ImageFont:
    names = ("DejaVuSans-Bold.ttf", "arialbd.ttf") if bold else ("DejaVuSans.ttf", "arial.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size=max(4, int(size)))
        except OSError:
            continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text or " ", font=font)
    return int(box[2] - box[0])


def _break_token(
    draw: ImageDraw.ImageDraw,
    token: str,
    font: ImageFont.ImageFont,
    maximum_width: int,
) -> list[str]:
    """Break an unspaced token deterministically so no line can be clipped."""

    pieces: list[str] = []
    current = ""
    for character in token:
        candidate = current + character
        if current and _text_width(draw, candidate, font) > maximum_width:
            pieces.append(current)
            current = character
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces or [""]


def _wrap_copy(
    draw: ImageDraw.ImageDraw,
    value: str,
    font: ImageFont.ImageFont,
    maximum_width: int,
) -> list[str]:
    """Pixel-wrap exact copy while preserving authored paragraph boundaries."""

    lines: list[str] = []
    for paragraph in str(value).splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            fragments = (
                [word]
                if _text_width(draw, word, font) <= maximum_width
                else _break_token(draw, word, font, maximum_width)
            )
            for fragment in fragments:
                candidate = fragment if not current else f"{current} {fragment}"
                if current and _text_width(draw, candidate, font) > maximum_width:
                    lines.append(current)
                    current = fragment
                else:
                    current = candidate
        if current:
            lines.append(current)
    return lines


def _copy_layout(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    fields: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Find the first deterministic, hierarchy-preserving layout that fits."""

    short_edge = min(width, height)
    margin = max(3, int(round(short_edge * 0.055)))
    content_width = width - 2 * margin
    content_height = height - 2 * margin
    if content_width < 8 or content_height < 8:
        raise ValueError("End-card safe area is too small for deterministic copy rendering.")
    scales = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
    for scale in scales:
        layout: list[dict[str, Any]] = []
        for field in fields:
            requested_size = max(
                int(field["minimum_size"]),
                int(round(short_edge * float(field["size_ratio"]) * scale)),
            )
            font = _font(requested_size, bold=bool(field["bold"]))
            lines = _wrap_copy(draw, str(field["text"]), font, content_width)
            if any(_text_width(draw, line, font) > content_width for line in lines):
                layout = []
                break
            line_spacing = max(0, int(round(requested_size * 0.15)))
            rendered_text = "\n".join(lines)
            block_box = draw.multiline_textbbox(
                (0, 0), rendered_text, font=font, spacing=line_spacing, align="center"
            )
            block_width = max(1, int(block_box[2] - block_box[0]))
            block_height = max(1, int(block_box[3] - block_box[1]))
            layout.append(
                {
                    **field,
                    "font": font,
                    "font_size": requested_size,
                    "lines": lines,
                    "rendered_text": rendered_text,
                    "line_spacing": line_spacing,
                    "width": block_width,
                    "height": block_height,
                    "bbox_top": int(block_box[1]),
                }
            )
        if not layout:
            continue
        gap = max(1, int(round(short_edge * 0.012 * scale)))
        total_height = sum(int(item["height"]) for item in layout) + gap * (len(layout) - 1)
        if total_height <= content_height:
            return layout, margin, gap
    raise ValueError(
        "Locked advertisement end-card copy does not fit the requested dimensions at the minimum "
        "legible hierarchy. Increase the canvas size or shorten the governed copy; clipping is prohibited."
    )


class EndCardRenderer:
    """Render a deterministic campaign-derived tail card for an exact frame count."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "advertisement_contract_json": ("STRING", {"forceInput": True, "multiline": True}),
                "width": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 8}),
                "background_color": ("STRING", {"default": "#10141f"}),
                "brand_color": ("STRING", {"default": "#ffffff"}),
                "accent_color": ("STRING", {"default": "#76f7c5"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("end_card_images", "end_card_report_json", "status", "ready")
    FUNCTION = "render"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Renders all campaign-locked finishing copy in a deterministic legible hierarchy and the exact "
        "contracted end-card frame count. Copy that cannot fit is rejected instead of clipped."
    )

    def render(
        self,
        advertisement_contract_json: str,
        width: int,
        height: int,
        background_color: str,
        brand_color: str,
        accent_color: str,
    ):
        contract = _contract(advertisement_contract_json)
        w, h = int(width), int(height)
        if w < 64 or h < 64:
            raise ValueError("End-card dimensions must each be at least 64 pixels.")
        end_card = contract.get("end_card", {})
        frame_count = int(end_card.get("frame_count", -1))
        seconds = float(end_card.get("duration_seconds", -1.0))
        if frame_count != int(round(seconds * H3_FPS)) or frame_count <= 0:
            raise ValueError("Advertisement contract has no positive exact end-card frame count.")
        estimated_tensor_bytes = frame_count * w * h * 3 * 4
        if estimated_tensor_bytes > MAX_END_CARD_TENSOR_BYTES:
            raise ValueError(
                "Advertisement end-card tensor would exceed the 2 GiB allocation ceiling; "
                "reduce dimensions or contracted end-card duration."
            )
        background = _color(background_color, "background_color")
        brand_color_rgb = _color(brand_color, "brand_color")
        accent = _color(accent_color, "accent_color")
        exact_copy = contract.get("mandatory_copy")
        if not isinstance(exact_copy, Mapping):
            raise ValueError("Advertisement contract has no exact deterministic finishing copy.")
        brand = str(exact_copy.get("brand_name", "")).strip()
        product = str(exact_copy.get("product_name", "")).strip()
        variant = str(exact_copy.get("variant_name", "")).strip()
        headline = str(exact_copy.get("headline", "")).strip()
        cta = str(exact_copy.get("call_to_action", "")).strip()
        offer = str(exact_copy.get("offer_text", "")).strip()
        price = str(exact_copy.get("price_text", "")).strip()
        legal = str(exact_copy.get("legal_line", "")).strip()
        raw_claims = exact_copy.get("approved_claims", [])
        if not isinstance(raw_claims, list) or any(not isinstance(value, str) for value in raw_claims):
            raise ValueError("Advertisement exact-copy approved_claims must be a JSON array of strings.")
        claims = [value.strip() for value in raw_claims if value.strip()]
        if not brand or not product or not headline:
            raise ValueError("Advertisement exact finishing copy requires brand_name, product_name, and headline.")
        if brand != str(contract.get("brand_name", "")) or product != str(contract.get("product_name", "")):
            raise ValueError("Advertisement root identity and exact finishing copy disagree.")
        if cta != str(contract.get("call_to_action", "")).strip():
            raise ValueError("Advertisement root CTA and exact finishing copy disagree.")
        rendered_copy: dict[str, Any] = {
            "brand_name": brand,
            "product_name": product,
            "headline": headline,
        }
        for key, value in (
            ("variant_name", variant),
            ("offer_text", offer),
            ("price_text", price),
            ("call_to_action", cta),
            ("legal_line", legal),
        ):
            if value:
                rendered_copy[key] = value
        if claims:
            rendered_copy["approved_claims"] = claims
        product_line = product + (f" — {variant}" if variant else "")
        fields: list[dict[str, Any]] = [
            {
                "field": "brand_name",
                "text": brand,
                "size_ratio": 0.052,
                "minimum_size": 4,
                "bold": True,
                "fill": brand_color_rgb,
            },
            {
                "field": "product_name" + ("+variant_name" if variant else ""),
                "text": product_line,
                "size_ratio": 0.042,
                "minimum_size": 4,
                "bold": True,
                "fill": accent,
            },
            {
                "field": "headline",
                "text": headline,
                "size_ratio": 0.075,
                "minimum_size": 5,
                "bold": True,
                "fill": brand_color_rgb,
            },
        ]
        for field, text_value, size_ratio, minimum_size, bold, fill in (
            ("offer_text", offer, 0.040, 4, False, accent),
            ("price_text", price, 0.060, 5, True, accent),
            ("call_to_action", cta, 0.050, 4, True, brand_color_rgb),
            ("approved_claims", "\n".join(claims), 0.025, 4, False, brand_color_rgb),
            ("legal_line", legal, 0.022, 4, False, brand_color_rgb),
        ):
            if text_value:
                fields.append(
                    {
                        "field": field,
                        "text": text_value,
                        "size_ratio": size_ratio,
                        "minimum_size": minimum_size,
                        "bold": bold,
                        "fill": fill,
                    }
                )
        canvas = Image.new("RGB", (w, h), background)
        draw = ImageDraw.Draw(canvas)
        layout, margin, gap = _copy_layout(draw, w, h, fields)
        draw.rounded_rectangle(
            (max(1, margin // 3), max(1, margin // 3), w - max(1, margin // 3) - 1, h - max(1, margin // 3) - 1),
            radius=max(2, margin // 2),
            outline=accent,
            width=max(1, margin // 10),
        )
        total_height = sum(int(item["height"]) for item in layout) + gap * (len(layout) - 1)
        y = max(margin, (h - total_height) // 2)
        hierarchy: list[dict[str, Any]] = []
        for item in layout:
            text_value = str(item["rendered_text"])
            x = (w - int(item["width"])) // 2
            draw.multiline_text(
                (x, y - int(item["bbox_top"])),
                text_value,
                font=item["font"],
                fill=item["fill"],
                spacing=int(item["line_spacing"]),
                align="center",
            )
            hierarchy.append(
                {
                    "field": item["field"],
                    "font_size": int(item["font_size"]),
                    "line_count": len(item["lines"]),
                    "top": y,
                    "bottom_exclusive": y + int(item["height"]),
                }
            )
            y += int(item["height"]) + gap
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        frame = torch.from_numpy(array).unsqueeze(0)
        images = frame.repeat(frame_count, 1, 1, 1).contiguous()
        report: dict[str, Any] = {
            "schema": ADVERTISEMENT_END_CARD_SCHEMA,
            "version": VERSION,
            "campaign_id": contract["campaign_id"],
            "advertisement_contract_sha256": contract["contract_sha256"],
            "frame_count": frame_count,
            "duration_seconds": seconds,
            "fps": H3_FPS,
            "pixel_dimensions": [w, h],
            "estimated_tensor_bytes": estimated_tensor_bytes,
            "pixel_sha256": _tensor_sha256(images),
            "rendered_copy": rendered_copy,
            "copy_hierarchy": hierarchy,
            "copy_fit_policy": "deterministic_scale_and_pixel_wrap_fail_closed_no_clipping",
            "deterministic": True,
            "ready": True,
        }
        report["report_sha256"] = _manifest_sha(report)
        return images, _json(report), f"Rendered exact {frame_count}-frame deterministic advertisement end card.", True


class AdvertisementMasterFinisher:
    """Replace the exact contracted master tail with a verified end card."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "assembled_master": ("IMAGE",),
                "master_audio": ("AUDIO",),
                "end_card_images": ("IMAGE",),
                "end_card_report_json": ("STRING", {"forceInput": True, "multiline": True}),
                "advertisement_contract_json": ("STRING", {"forceInput": True, "multiline": True}),
                "assembly_report_json": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("finished_master", "finished_audio", "finish_report_json", "status", "ready")
    FUNCTION = "finish"
    CATEGORY = CATEGORY
    DESCRIPTION = "Replaces, rather than appends, the exact contracted master tail so delivery duration never drifts."

    def finish(
        self,
        assembled_master: Any,
        master_audio: Any,
        end_card_images: Any,
        end_card_report_json: str,
        advertisement_contract_json: str,
        assembly_report_json: str,
    ):
        contract = _contract(advertisement_contract_json)
        master = _rgb_images(assembled_master, "assembled_master")
        card = _rgb_images(end_card_images, "end_card_images")
        assembly = _parse_object(assembly_report_json, "assembly_report_json")
        end_report = _parse_object(end_card_report_json, "end_card_report_json")
        campaign = str(contract["campaign_id"])
        if assembly.get("schema") != ADVERTISEMENT_ASSEMBLY_SCHEMA or assembly.get("campaign_id") != campaign:
            raise ValueError("Assembly report does not belong to this advertisement campaign.")
        if end_report.get("schema") != ADVERTISEMENT_END_CARD_SCHEMA or end_report.get("campaign_id") != campaign:
            raise ValueError("End-card report does not belong to this advertisement campaign.")
        if assembly.get("advertisement_contract_sha256") != contract["contract_sha256"] or end_report.get("advertisement_contract_sha256") != contract["contract_sha256"]:
            raise ValueError("Advertisement finishing inputs were produced from different contract revisions.")
        _verify_embedded_hash(assembly, "report_sha256", "Assembly report")
        _verify_embedded_hash(end_report, "report_sha256", "End-card report")
        _verify_audio(contract, master_audio, exact_duration=True)
        expected_frames = _expected_master_frames(contract)
        end_frames = int(contract["end_card"]["frame_count"])
        if int(master.shape[0]) != expected_frames:
            raise ValueError(f"Assembled advertisement master must contain exactly {expected_frames} frames.")
        if int(card.shape[0]) != end_frames or int(end_report.get("frame_count", -1)) != end_frames:
            raise ValueError(f"Advertisement end card must contain exactly {end_frames} frames.")
        if master.shape[1:] != card.shape[1:]:
            raise ValueError("Advertisement end-card and master pixel dimensions must match exactly.")
        if str(assembly.get("assembled_pixel_sha256", "")) != _tensor_sha256(master):
            raise ValueError("Assembled advertisement pixels do not match the assembly report.")
        if str(end_report.get("pixel_sha256", "")) != _tensor_sha256(card):
            raise ValueError("Advertisement end-card pixels do not match the end-card report.")
        if end_frames > expected_frames:
            raise ValueError("Advertisement end card cannot exceed the master duration.")
        finished = torch.cat((master[:-end_frames], card), dim=0)
        if int(finished.shape[0]) != expected_frames:
            raise AssertionError("Advertisement finishing changed the master frame count.")
        report: dict[str, Any] = {
            "schema": "diffusiongemma.advertisement_master_finish",
            "version": VERSION,
            "campaign_id": campaign,
            "advertisement_contract_sha256": contract["contract_sha256"],
            "assembly_report_sha256": str(assembly.get("report_sha256", "")),
            "end_card_report_sha256": str(end_report.get("report_sha256", "")),
            "exact_master_frames": expected_frames,
            "end_card_replaced_tail_frames": end_frames,
            "duration_seconds": float(contract["production_duration_seconds"]),
            "fps": H3_FPS,
            "finished_pixel_sha256": _tensor_sha256(finished),
            "master_audio_sha256": waveform_sha256(master_audio),
            "duration_changed": False,
            "ready": True,
        }
        report["report_sha256"] = _manifest_sha(report)
        return finished, master_audio, _json(report), "Finished advertisement master without changing duration or audio.", True


def _slice_audio(audio: Mapping[str, Any], start_seconds: float, duration_seconds: float) -> tuple[dict[str, Any], int, int]:
    waveform, sample_rate = _audio_parts(audio)
    start = int(round(start_seconds * sample_rate))
    count = int(round(duration_seconds * sample_rate))
    end = start + count
    if start < 0 or end > int(waveform.shape[-1]):
        raise ValueError("Requested cutdown audio window is outside the locked master waveform.")
    return {"waveform": waveform[..., start:end].clone(), "sample_rate": sample_rate}, start, count


def _requested_adaptations(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    requested = contract.get("requested_deliverables", {})
    raw: list[Any] = []
    if isinstance(requested, Mapping):
        value = requested.get("adaptations", [])
        if isinstance(value, list):
            raw = value
    elif isinstance(requested, list):
        raw = requested
    master = str(contract.get("master_aspect_ratio", ""))
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item).strip()
        match = re.search(r"(?:21:9|16:9|9:16|4:3|3:4|3:2|2:3|1:1)", text)
        aspect = match.group(0) if match else text
        if aspect in seen:
            continue
        seen.add(aspect)
        if aspect == master:
            status = "not_applicable_same_as_master"
            reason = "The requested aspect is already the rendered master; no separate adaptation was produced."
        elif not ASPECT_RE.fullmatch(aspect):
            status = "impossible_invalid_aspect"
            reason = "No supported aspect ratio could be parsed, so no pixels were produced."
        else:
            status = "planned_not_rendered"
            reason = "A semantic reframe renderer was not supplied; a center crop is not represented as a finished adaptation."
        results.append({"request": text, "aspect_ratio": aspect, "status": status, "reason": reason})
    return results


class AdvertisementAdaptationStatus:
    """Expose requested format adaptations without pretending a reframe was rendered."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "advertisement_contract_json": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "BOOLEAN", "STRING", "BOOLEAN")
    RETURN_NAMES = (
        "adaptation_status_json",
        "requested_adaptation_count",
        "all_requested_rendered",
        "status",
        "ready",
    )
    FUNCTION = "evaluate"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Reports every requested aspect adaptation as planned, inapplicable, or impossible. "
        "It cannot emit a rendered status because this node receives no adapted pixels."
    )

    def evaluate(self, advertisement_contract_json: str):
        contract = _contract(advertisement_contract_json)
        adaptations = _requested_adaptations(contract)
        report: dict[str, Any] = {
            "schema": ADVERTISEMENT_ADAPTATION_SCHEMA,
            "version": VERSION,
            "campaign_id": contract["campaign_id"],
            "advertisement_contract_sha256": contract["contract_sha256"],
            "master_aspect_ratio": contract["master_aspect_ratio"],
            "adaptations": adaptations,
            "all_requested_rendered": False,
            "pixel_inputs_received": False,
            "ready": True,
        }
        report["report_sha256"] = _manifest_sha(report)
        message = (
            f"Reported {len(adaptations)} requested advertisement adaptation(s); none is claimed rendered."
            if adaptations
            else "No separate advertisement aspect adaptation was requested."
        )
        return _json(report), len(adaptations), False, message, True


class AdvertisementCutdownRenderer:
    """Render real, sample-accurate 15-second and 6-second cutdowns."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "finished_master": ("IMAGE",),
                "finished_audio": ("AUDIO",),
                "advertisement_contract_json": ("STRING", {"forceInput": True, "multiline": True}),
                "cutdown_15_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 1.0 / H3_FPS}),
                "cutdown_6_start_seconds": ("FLOAT", {"default": 24.0, "min": 0.0, "step": 1.0 / H3_FPS}),
            },
            "optional": {
                "adaptation_status_json": ("STRING", {"forceInput": True, "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "AUDIO", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = (
        "cutdown_15_images",
        "cutdown_15_audio",
        "cutdown_6_images",
        "cutdown_6_audio",
        "delivery_manifest_json",
        "status",
        "ready",
    )
    FUNCTION = "render"
    CATEGORY = CATEGORY
    DESCRIPTION = "Slices actual master frames and exact waveform samples for honest 15s and 6s rendered deliverables."

    def render(
        self,
        finished_master: Any,
        finished_audio: Any,
        advertisement_contract_json: str,
        cutdown_15_start_seconds: float,
        cutdown_6_start_seconds: float,
        adaptation_status_json: str = "",
    ):
        contract = _contract(advertisement_contract_json)
        master = _rgb_images(finished_master, "finished_master")
        _verify_audio(contract, finished_audio, exact_duration=True)
        expected_frames = _expected_master_frames(contract)
        if int(master.shape[0]) != expected_frames:
            raise ValueError(f"Finished advertisement master must contain exactly {expected_frames} frames.")
        master_seconds = float(contract["production_duration_seconds"])
        requested = contract.get("requested_deliverables", {})
        requested_cutdowns = (
            requested.get("cutdowns_seconds", []) if isinstance(requested, Mapping) else []
        )
        if requested_cutdowns not in ([15, 6], [15.0, 6.0]):
            raise ValueError(
                "Advertisement v1 Cutdown Renderer requires the governed requested cutdowns [15, 6]."
            )
        rendered: list[tuple[torch.Tensor, dict[str, Any], dict[str, Any]]] = []
        for seconds, raw_start in ((15.0, cutdown_15_start_seconds), (6.0, cutdown_6_start_seconds)):
            start_seconds = float(raw_start)
            if not math.isfinite(start_seconds) or start_seconds < 0:
                raise ValueError("Cutdown start times must be non-negative finite numbers.")
            frame_start_float = start_seconds * H3_FPS
            frame_start = int(round(frame_start_float))
            if abs(frame_start_float - frame_start) > 1.0e-6:
                raise ValueError("Cutdown start times must align exactly to a 24 fps master frame.")
            frame_count = int(round(seconds * H3_FPS))
            if start_seconds + seconds > master_seconds + 1.0e-6 or frame_start + frame_count > expected_frames:
                raise ValueError(f"The requested {seconds:g}s cutdown window is outside the advertisement master.")
            images = master[frame_start : frame_start + frame_count].clone()
            audio, sample_start, sample_count = _slice_audio(finished_audio, start_seconds, seconds)
            report = {
                "id": f"cutdown_{int(seconds)}s",
                "status": "rendered",
                "aspect_ratio": contract["master_aspect_ratio"],
                "start_seconds": round(start_seconds, 6),
                "duration_seconds": seconds,
                "frame_start": frame_start,
                "frame_count": frame_count,
                "audio_sample_start": sample_start,
                "audio_sample_count": sample_count,
                "pixel_sha256": _tensor_sha256(images),
                "audio_sha256": waveform_sha256(audio),
            }
            rendered.append((images, audio, report))
        adaptations = _requested_adaptations(contract)
        if str(adaptation_status_json or "").strip():
            adaptation_report = _parse_object(adaptation_status_json, "adaptation_status_json")
            if adaptation_report.get("schema") != ADVERTISEMENT_ADAPTATION_SCHEMA or adaptation_report.get("version") != VERSION:
                raise ValueError("adaptation_status_json has an unsupported schema or version.")
            if adaptation_report.get("campaign_id") != contract["campaign_id"] or adaptation_report.get("advertisement_contract_sha256") != contract["contract_sha256"]:
                raise ValueError("Adaptation status does not belong to this advertisement contract revision.")
            _verify_embedded_hash(adaptation_report, "report_sha256", "Adaptation status")
            if adaptation_report.get("adaptations") != adaptations:
                raise ValueError("Adaptation status does not match the contract's requested deliverables.")
        manifest: dict[str, Any] = {
            "schema": ADVERTISEMENT_DELIVERY_SCHEMA,
            "version": VERSION,
            "campaign_id": contract["campaign_id"],
            "advertisement_contract_sha256": contract["contract_sha256"],
            "master": {
                "status": "rendered",
                "aspect_ratio": contract["master_aspect_ratio"],
                "duration_seconds": master_seconds,
                "frame_count": expected_frames,
                "pixel_sha256": _tensor_sha256(master),
                "audio_sha256": waveform_sha256(finished_audio),
            },
            "cutdowns": [entry[2] for entry in rendered],
            "adaptations": adaptations,
            "honesty_policy": "Only returned frame/audio tensors are marked rendered; format adaptations remain explicit non-renders.",
            "ready": True,
        }
        manifest["manifest_sha256"] = _manifest_sha(manifest)
        return (
            rendered[0][0],
            rendered[0][1],
            rendered[1][0],
            rendered[1][1],
            _json(manifest),
            "Rendered exact 15s and 6s advertisement cutdowns; unrendered adaptations remain explicitly labeled.",
            True,
        )


class MediaQAGate:
    """Aggregate measured media checks without converting unknowns into passes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "advertisement_contract_json": ("STRING", {"forceInput": True, "multiline": True}),
                "delivery_manifest_json": ("STRING", {"forceInput": True, "multiline": True}),
                "checks_json": ("STRING", {"default": "{}", "multiline": True}),
                "required_checks_json": (
                    "STRING",
                    {
                        "default": '["product_identity","performer_identity","copy_legibility","audio_sync","technical_integrity"]',
                        "multiline": True,
                        "tooltip": "Host-required identity, product, copy, sync, and integrity checks cannot be removed; this list may only add requirements.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", ["pass", "fail", "not_measured"], "STRING", "BOOLEAN")
    RETURN_NAMES = ("media_qa_report_json", "qa_status", "status", "ready")
    FUNCTION = "evaluate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Emits pass, fail, or not_measured; absent evidence can never silently pass delivery QA."

    def evaluate(
        self,
        advertisement_contract_json: str,
        delivery_manifest_json: str,
        checks_json: str,
        required_checks_json: str,
    ):
        contract = _contract(advertisement_contract_json)
        delivery = _parse_object(delivery_manifest_json, "delivery_manifest_json")
        if delivery.get("schema") != ADVERTISEMENT_DELIVERY_SCHEMA or delivery.get("version") != VERSION:
            raise ValueError("delivery_manifest_json has an unsupported schema or version.")
        if delivery.get("campaign_id") != contract["campaign_id"] or delivery.get("advertisement_contract_sha256") != contract["contract_sha256"]:
            raise ValueError("Delivery manifest does not belong to this advertisement contract revision.")
        _verify_embedded_hash(delivery, "manifest_sha256", "Delivery manifest")
        checks = _parse_object(checks_json, "checks_json")
        try:
            required = json.loads(str(required_checks_json or "[]"))
        except json.JSONDecodeError as exc:
            raise ValueError("required_checks_json must be valid JSON.") from exc
        if not isinstance(required, list) or any(not str(value).strip() for value in required):
            raise ValueError("required_checks_json must be a JSON array of check names.")
        requested = [str(value).strip() for value in required]
        if len(requested) != len(set(requested)):
            raise ValueError("required_checks_json cannot contain duplicate check names.")
        # The editable list is additive only.  A saved-workflow edit cannot
        # remove the host-owned delivery baseline and silently approve media.
        required = list(MANDATORY_MEDIA_QA_CHECKS)
        required.extend(name for name in requested if name not in MANDATORY_MEDIA_QA_CHECKS)
        evaluated: list[dict[str, Any]] = []
        states: list[str] = []
        for raw_name in required:
            name = str(raw_name).strip()
            raw = checks.get(name, "not_measured")
            evidence = ""
            if isinstance(raw, Mapping):
                state = str(raw.get("status", "not_measured")).strip().casefold()
                evidence = str(raw.get("evidence", "")).strip()
            else:
                state = str(raw).strip().casefold()
            if state not in QA_STATES:
                raise ValueError(f"Media QA check {name!r} must be pass, fail, or not_measured.")
            if state == "pass" and not evidence:
                state = "not_measured"
                evidence = "A pass status was supplied without evidence and was downgraded to not_measured."
            states.append(state)
            evaluated.append({"check": name, "status": state, "evidence": evidence})
        if "fail" in states:
            overall = "fail"
        elif "not_measured" in states:
            overall = "not_measured"
        else:
            overall = "pass"
        ready = overall == "pass"
        report: dict[str, Any] = {
            "schema": ADVERTISEMENT_MEDIA_QA_SCHEMA,
            "version": VERSION,
            "campaign_id": contract["campaign_id"],
            "advertisement_contract_sha256": contract["contract_sha256"],
            "delivery_manifest_sha256": str(delivery.get("manifest_sha256", "")),
            "host_required_checks": list(MANDATORY_MEDIA_QA_CHECKS),
            "checks": evaluated,
            "overall_status": overall,
            "ready_for_delivery": ready,
        }
        report["report_sha256"] = _manifest_sha(report)
        if overall == "pass":
            message = "All required advertisement media checks passed with supplied evidence."
        elif overall == "fail":
            message = "Advertisement media QA failed; delivery remains blocked."
        else:
            message = "Advertisement media QA is not measured completely; delivery remains blocked."
        return _json(report), overall, message, ready


NODE_CLASS_MAPPINGS = {
    "DiffusionGemmaAdvertisementMasterAssembler": AdvertisementMasterAssembler,
    "DiffusionGemmaAdvertisementEndCardRenderer": EndCardRenderer,
    "DiffusionGemmaAdvertisementMasterFinisher": AdvertisementMasterFinisher,
    "DiffusionGemmaAdvertisementAdaptationStatus": AdvertisementAdaptationStatus,
    "DiffusionGemmaAdvertisementCutdownRenderer": AdvertisementCutdownRenderer,
    "DiffusionGemmaAdvertisementMediaQAGate": MediaQAGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionGemmaAdvertisementMasterAssembler": "DiffusionGemma Advertisement Master Assembler",
    "DiffusionGemmaAdvertisementEndCardRenderer": "DiffusionGemma Advertisement End Card Renderer",
    "DiffusionGemmaAdvertisementMasterFinisher": "DiffusionGemma Advertisement Master Finisher",
    "DiffusionGemmaAdvertisementAdaptationStatus": "DiffusionGemma Advertisement Adaptation Status",
    "DiffusionGemmaAdvertisementCutdownRenderer": "DiffusionGemma Advertisement Cutdown Renderer",
    "DiffusionGemmaAdvertisementMediaQAGate": "DiffusionGemma Advertisement Media QA Gate",
}


__all__ = [
    "AdvertisementMasterAssembler",
    "EndCardRenderer",
    "AdvertisementMasterFinisher",
    "AdvertisementAdaptationStatus",
    "AdvertisementCutdownRenderer",
    "MediaQAGate",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
