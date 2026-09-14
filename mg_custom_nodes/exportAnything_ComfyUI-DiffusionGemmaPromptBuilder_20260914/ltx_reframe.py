# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

"""Interactive source-video layout preparation for LTX outpainting workflows."""

from __future__ import annotations

import json
import hashlib
import math
import os
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

import folder_paths
import torch
from aiohttp import web
from comfy_api.latest import InputImpl, Types
from comfy_extras.nodes_audio import load as load_audio_file
import comfy.utils


CATEGORY = "Lightricks/video"
ASPECT_PRESETS = ["16:9", "9:16", "1:1", "4:5", "5:4", "4:3", "3:4", "21:9", "Custom"]
DIMENSION_MULTIPLE = 64
MIN_SOURCE_SCALE = 0.1
MAX_SOURCE_SCALE = 4.0
SILENT_AUDIO_SAMPLE_RATE = 44100
SILENT_AUDIO_CHANNELS = 2


@dataclass(frozen=True)
class ReframePlacement:
    target_width: int
    target_height: int
    source_width: int
    source_height: int
    box_x: int
    box_y: int
    box_width: int
    box_height: int
    position_x: float
    position_y: float
    source_scale: float


@dataclass(frozen=True)
class ReframeFrameRange:
    range_enabled: bool
    source_frame_count: int
    requested_first_frame: int
    requested_last_frame: int
    effective_first_frame: int
    effective_last_frame: int
    selected_frame_count: int
    start_time: float
    duration: float


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def snap_dimension(value: int, multiple: int = DIMENSION_MULTIPLE) -> int:
    """Snap a target dimension to an LTX-friendly multiple."""

    value = max(256, min(8192, int(value)))
    return max(multiple, int(math.floor((value + multiple / 2) / multiple) * multiple))


def resolve_frame_range(
    source_frame_count: int,
    fps: float,
    limit_frames: bool,
    first_frame: int,
    last_frame: int,
) -> ReframeFrameRange:
    """Resolve an inclusive UI range to an LTX-compatible ``8n+1`` window."""

    source_frame_count = int(source_frame_count)
    fps = float(fps)
    if source_frame_count < 1:
        raise ValueError("The selected video does not contain any frames.")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("The selected video does not have a valid frame rate.")

    if not bool(limit_frames):
        selected_frame_count = source_frame_count
        return ReframeFrameRange(
            range_enabled=False,
            source_frame_count=source_frame_count,
            requested_first_frame=0,
            requested_last_frame=source_frame_count - 1,
            effective_first_frame=0,
            effective_last_frame=source_frame_count - 1,
            selected_frame_count=selected_frame_count,
            start_time=0.0,
            duration=selected_frame_count / fps,
        )

    requested_first = int(first_frame)
    requested_last = source_frame_count - 1 if int(last_frame) == -1 else int(last_frame)
    if requested_first < 0 or requested_first >= source_frame_count:
        raise ValueError(
            f"First frame {requested_first} is outside this video's 0-{source_frame_count - 1} range."
        )
    if requested_last < 0 or requested_last >= source_frame_count:
        raise ValueError(
            f"Last frame {requested_last} is outside this video's 0-{source_frame_count - 1} range."
        )
    if requested_last < requested_first:
        raise ValueError("Last frame must be greater than or equal to first frame.")

    requested_count = requested_last - requested_first + 1
    if 2 <= requested_count <= 8:
        raise ValueError(
            "LTX frame-limited video ranges must contain one frame or at least nine frames."
        )
    selected_frame_count = (
        1 if requested_count == 1 else ((requested_count - 1) // 8) * 8 + 1
    )
    effective_last = requested_first + selected_frame_count - 1
    return ReframeFrameRange(
        range_enabled=True,
        source_frame_count=source_frame_count,
        requested_first_frame=requested_first,
        requested_last_frame=requested_last,
        effective_first_frame=requested_first,
        effective_last_frame=effective_last,
        selected_frame_count=selected_frame_count,
        start_time=requested_first / fps,
        duration=selected_frame_count / fps,
    )


def ensure_audio_track(
    audio: Any,
    frame_count: int,
    fps: float,
    *,
    sample_rate: int = SILENT_AUDIO_SAMPLE_RATE,
    channels: int = SILENT_AUDIO_CHANNELS,
) -> tuple[Any, bool]:
    """Return source audio, or duration-matched silence for an audio-less video."""

    if audio is not None:
        return audio, False

    frame_count = max(1, int(frame_count))
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Cannot synthesize fallback audio without a valid video frame rate.")

    sample_rate = max(1, int(sample_rate))
    channels = max(1, min(2, int(channels)))
    duration = frame_count / fps
    num_samples = max(1, int(round(duration * sample_rate)))
    return (
        {
            "waveform": torch.zeros((1, channels, num_samples), dtype=torch.float32),
            "sample_rate": sample_rate,
        },
        True,
    )


def fit_audio_track(
    audio: Any,
    frame_count: int,
    fps: float,
    *,
    sample_rate: int = SILENT_AUDIO_SAMPLE_RATE,
    channels: int = SILENT_AUDIO_CHANNELS,
) -> tuple[Any, bool]:
    """Crop or right-pad audio to the exact selected video duration."""

    if audio is None:
        return ensure_audio_track(
            None,
            frame_count,
            fps,
            sample_rate=sample_rate,
            channels=channels,
        )

    waveform = audio.get("waveform")
    source_sample_rate = int(audio.get("sample_rate", 0))
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError("Source audio must contain a [batch, channels, samples] waveform.")
    if source_sample_rate <= 0:
        raise ValueError("Source audio must have a valid sample rate.")

    target_samples = max(1, int(round((int(frame_count) / float(fps)) * source_sample_rate)))
    if waveform.shape[-1] < target_samples:
        waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[-1]))
    else:
        waveform = waveform[..., :target_samples]
    return {"waveform": waveform, "sample_rate": source_sample_rate}, False


def _resolve_input_audio_path(audio_file: str) -> str:
    """Resolve a custom audio selection strictly inside ComfyUI's input folder."""

    selection = str(audio_file or "").strip()
    if not selection:
        raise ValueError(
            "Custom audio is enabled, but no custom audio file is selected. "
            "Upload or select an audio file in LTX Reframe Custom Audio Loader."
        )

    relative_name, annotated_base = folder_paths.annotated_filepath(selection)
    input_dir = os.path.realpath(folder_paths.get_input_directory())
    if annotated_base is not None and os.path.realpath(annotated_base) != input_dir:
        raise ValueError("Custom audio must be selected from ComfyUI's input directory.")

    audio_path = os.path.realpath(os.path.join(input_dir, relative_name))
    try:
        inside_input = os.path.commonpath((input_dir, audio_path)) == input_dir
    except ValueError:
        inside_input = False
    if not inside_input:
        raise ValueError("Custom audio must be selected from ComfyUI's input directory.")
    if not os.path.isfile(audio_path):
        raise ValueError(f"Custom audio file was not found: {relative_name}")
    return audio_path


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LTXReframeCustomAudioLoader:
    """Lazy-safe custom audio loader for optional reframe workflow branches."""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        files = folder_paths.filter_files_content_types(
            os.listdir(input_dir),
            ["audio", "video"],
        )
        return {
            "required": {
                "audio_file": (
                    [""] + sorted(files),
                    {"audio_upload": True},
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_audio"
    CATEGORY = "Lightricks/audio"
    DESCRIPTION = (
        "Loads custom audio only when its lazy workflow branch is selected. "
        "An empty selection remains valid while source-audio mode is active."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, audio_file):
        # ComfyUI validates both sides of a lazy switch. File validation is
        # intentionally deferred until this branch actually executes.
        return True

    @classmethod
    def IS_CHANGED(cls, audio_file):
        try:
            return _sha256_file(_resolve_input_audio_path(audio_file))
        except (OSError, ValueError):
            return f"unselected:{str(audio_file or '').strip()}"

    def load_audio(self, audio_file):
        audio_path = _resolve_input_audio_path(audio_file)
        try:
            waveform, sample_rate = load_audio_file(audio_path)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(f"Custom audio could not be decoded: {error}") from error
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)


def _resolve_input_video_path(source_file: str) -> str:
    """Resolve a source name while preventing metadata reads outside ComfyUI input."""

    relative_name, annotated_base = folder_paths.annotated_filepath(str(source_file))
    input_dir = os.path.realpath(folder_paths.get_input_directory())
    if annotated_base is not None and os.path.realpath(annotated_base) != input_dir:
        raise ValueError("Only ComfyUI input videos may be inspected.")
    video_path = os.path.realpath(os.path.join(input_dir, relative_name))
    try:
        inside_input = os.path.commonpath((input_dir, video_path)) == input_dir
    except ValueError:
        inside_input = False
    if not inside_input or not os.path.isfile(video_path):
        raise ValueError("The selected input video does not exist.")
    return video_path


def probe_video_metadata(source_file: str) -> dict[str, Any]:
    video_path = _resolve_input_video_path(source_file)
    source_video = InputImpl.VideoFromFile(video_path)
    width, height = source_video.get_dimensions()
    fps = float(source_video.get_frame_rate())
    frame_count = int(source_video.get_frame_count())
    duration = float(source_video.get_duration())
    return {
        "source_file": source_file,
        "width": int(width),
        "height": int(height),
        "duration": duration,
        "fps": fps,
        "frame_count": frame_count,
    }


def compute_reframe_placement(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    position_x: float,
    position_y: float,
    source_scale: float,
) -> ReframePlacement:
    """Resolve normalized UI controls into an aspect-locked pixel box."""

    source_width = max(1, int(source_width))
    source_height = max(1, int(source_height))
    target_width = snap_dimension(target_width)
    target_height = snap_dimension(target_height)
    position_x = _clamp(position_x, 0.0, 1.0)
    position_y = _clamp(position_y, 0.0, 1.0)
    source_scale = _clamp(source_scale, MIN_SOURCE_SCALE, MAX_SOURCE_SCALE)

    fit = min(target_width / source_width, target_height / source_height)
    box_width = max(2, int(round(source_width * fit * source_scale)))
    box_height = max(2, int(round(source_height * fit * source_scale)))
    box_x = int(round(position_x * (target_width - box_width)))
    box_y = int(round(position_y * (target_height - box_height)))

    return ReframePlacement(
        target_width=target_width,
        target_height=target_height,
        source_width=source_width,
        source_height=source_height,
        box_x=box_x,
        box_y=box_y,
        box_width=box_width,
        box_height=box_height,
        position_x=position_x,
        position_y=position_y,
        source_scale=source_scale,
    )


def build_reframe_masks(
    placement: ReframePlacement,
    feather: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an LTX outpaint mask and a binary source-preservation mask."""

    height = placement.target_height
    width = placement.target_width
    x0 = placement.box_x
    y0 = placement.box_y
    x1 = x0 + placement.box_width
    y1 = y0 + placement.box_height

    source_mask = torch.zeros((1, height, width), dtype=dtype, device=device)
    visible_x0 = max(0, x0)
    visible_y0 = max(0, y0)
    visible_x1 = min(width, x1)
    visible_y1 = min(height, y1)
    if visible_x1 > visible_x0 and visible_y1 > visible_y0:
        source_mask[:, visible_y0:visible_y1, visible_x0:visible_x1] = 1.0

    feather = max(0, int(feather))
    if feather == 0:
        return 1.0 - source_mask, source_mask

    yy = torch.arange(height, dtype=dtype, device=device).view(height, 1)
    xx = torch.arange(width, dtype=dtype, device=device).view(1, width)
    dx = torch.maximum(torch.maximum(x0 - xx, xx - (x1 - 1)), torch.zeros_like(xx))
    dy = torch.maximum(torch.maximum(y0 - yy, yy - (y1 - 1)), torch.zeros_like(yy))
    distance = torch.sqrt(dx.square() + dy.square())
    outpaint_mask = torch.clamp(distance / float(feather), 0.0, 1.0).unsqueeze(0)
    return outpaint_mask, source_mask


def prepare_reframe_canvas(
    frames: torch.Tensor,
    placement: ReframePlacement,
) -> torch.Tensor:
    """Sample the visible source region into the target canvas without distortion."""

    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError("The selected video did not decode any frames.")
    if frames.shape[-1] < 3:
        raise ValueError("Video frames must have at least three color channels.")

    source = frames[..., :3]
    canvas = torch.zeros(
        (
            source.shape[0],
            placement.target_height,
            placement.target_width,
            3,
        ),
        dtype=source.dtype,
        device=source.device,
    )

    if (
        placement.box_x >= 0
        and placement.box_y >= 0
        and placement.box_x + placement.box_width <= placement.target_width
        and placement.box_y + placement.box_height <= placement.target_height
    ):
        resized = torch.nn.functional.interpolate(
            source.movedim(-1, 1),
            size=(placement.box_height, placement.box_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).movedim(1, -1)
        canvas[
            :,
            placement.box_y : placement.box_y + placement.box_height,
            placement.box_x : placement.box_x + placement.box_width,
            :,
        ] = resized
        return canvas

    x0 = max(0, placement.box_x)
    y0 = max(0, placement.box_y)
    x1 = min(placement.target_width, placement.box_x + placement.box_width)
    y1 = min(placement.target_height, placement.box_y + placement.box_height)
    if x1 <= x0 or y1 <= y0:
        return canvas

    # Sampling only the visible intersection prevents a zoomed-in placement from
    # allocating a potentially enormous intermediate image before it is cropped.
    destination_x = torch.arange(x0, x1, dtype=torch.float32, device=source.device)
    destination_y = torch.arange(y0, y1, dtype=torch.float32, device=source.device)
    grid_x = 2.0 * (
        (destination_x + 0.5 - placement.box_x) / float(placement.box_width)
    ) - 1.0
    grid_y = 2.0 * (
        (destination_y + 0.5 - placement.box_y) / float(placement.box_height)
    ) - 1.0
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(source.shape[0], -1, -1, -1)
    sampled = torch.nn.functional.grid_sample(
        source.movedim(-1, 1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).movedim(1, -1)
    canvas[:, y0:y1, x0:x1, :] = sampled
    return canvas


def _normalize_source_mask(
    source_mask: torch.Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize a ComfyUI MASK to ``[1|batch, 1, height, width]``."""

    if not isinstance(source_mask, torch.Tensor):
        raise ValueError("Source protection requires a MASK tensor.")
    if source_mask.ndim == 2:
        source_mask = source_mask.unsqueeze(0)
    elif source_mask.ndim == 4:
        if source_mask.shape[1] == 1:
            source_mask = source_mask[:, 0]
        elif source_mask.shape[-1] == 1:
            source_mask = source_mask[..., 0]
        else:
            raise ValueError("Source protection MASK must have one channel.")
    if source_mask.ndim != 3:
        raise ValueError("Source protection MASK must have shape [batch, height, width].")
    if source_mask.shape[0] not in (1, batch_size):
        raise ValueError(
            "Source protection MASK batch must contain one mask or match the IMAGE batch."
        )

    normalized = source_mask.to(device=device, dtype=torch.float32).unsqueeze(1)
    if normalized.shape[-2:] != (height, width):
        normalized = torch.nn.functional.interpolate(
            normalized,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    return normalized.clamp(0.0, 1.0)


def _feather_mask(mask: torch.Tensor, feather: int) -> torch.Tensor:
    """Apply a separable Gaussian feather without allocating a 2-D kernel."""

    radius = max(0, int(feather))
    if radius == 0:
        return mask

    sigma = max(float(radius) / 3.0, 0.5)
    coordinates = torch.arange(
        -radius,
        radius + 1,
        device=mask.device,
        dtype=torch.float32,
    )
    kernel = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    horizontal = kernel.view(1, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1)
    blurred = torch.nn.functional.conv2d(
        torch.nn.functional.pad(mask, (radius, radius, 0, 0), mode="replicate"),
        horizontal,
    )
    blurred = torch.nn.functional.conv2d(
        torch.nn.functional.pad(blurred, (0, 0, radius, radius), mode="replicate"),
        vertical,
    )
    return blurred.clamp(0.0, 1.0)


def rectilinear_correct_images(
    images: torch.Tensor,
    correction_strength: float,
    center_x: float,
    center_y: float,
    protect_source: float,
    protection_feather: int,
    interpolation: str,
    chunk_size: int,
    source_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one temporally stable barrel-to-rectilinear remap to an IMAGE batch."""

    if images.ndim != 4 or images.shape[0] == 0 or images.shape[-1] < 1:
        raise ValueError("Rectilinear correction requires a non-empty IMAGE batch.")

    batch_size, height, width, _channels = images.shape
    correction_strength = _clamp(correction_strength, 0.0, 0.25)
    center_x = _clamp(center_x, 0.0, 1.0)
    center_y = _clamp(center_y, 0.0, 1.0)
    protect_source = _clamp(protect_source, 0.0, 1.0)
    chunk_size = max(1, min(64, int(chunk_size)))
    interpolation = str(interpolation).lower()
    if interpolation not in {"bilinear", "bicubic"}:
        raise ValueError("Rectilinear correction interpolation must be bilinear or bicubic.")

    device = images.device
    center_pixel_x = center_x * max(0, width - 1)
    center_pixel_y = center_y * max(0, height - 1)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    delta_x = xx - center_pixel_x
    delta_y = yy - center_pixel_y
    corner_distances = torch.tensor(
        [
            center_pixel_x**2 + center_pixel_y**2,
            (width - 1 - center_pixel_x) ** 2 + center_pixel_y**2,
            center_pixel_x**2 + (height - 1 - center_pixel_y) ** 2,
            (width - 1 - center_pixel_x) ** 2
            + (height - 1 - center_pixel_y) ** 2,
        ],
        device=device,
        dtype=torch.float32,
    )
    maximum_radius_squared = corner_distances.max().clamp_min(1.0)
    radial_weight = ((delta_x.square() + delta_y.square()) / maximum_radius_squared).clamp(
        0.0, 1.0
    )

    # Positive strength samples progressively nearer the optical center. This
    # counteracts barrel/fisheye expansion while naturally cropping the outermost
    # pixels, so the remap never creates empty borders.
    radial_scale = 1.0 - correction_strength * radial_weight
    sample_x = center_pixel_x + delta_x * radial_scale
    sample_y = center_pixel_y + delta_y * radial_scale
    grid_x = (2.0 * (sample_x + 0.5) / float(width)) - 1.0
    grid_y = (2.0 * (sample_y + 0.5) / float(height)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    protection = None
    if source_mask is not None and protect_source > 0.0:
        protection = _normalize_source_mask(
            source_mask,
            batch_size=batch_size,
            height=height,
            width=width,
            device=device,
        )
        protection = _feather_mask(protection, protection_feather) * protect_source

    correction_mask = radial_weight.unsqueeze(0)
    if protection is not None:
        correction_mask = correction_mask * (1.0 - protection[:, 0])
    if correction_strength == 0.0:
        correction_mask = torch.zeros_like(correction_mask)

    source = images.movedim(-1, 1)
    corrected_chunks: list[torch.Tensor] = []
    for start in range(0, batch_size, chunk_size):
        end = min(batch_size, start + chunk_size)
        source_chunk = source[start:end].to(dtype=torch.float32)
        corrected_chunk = torch.nn.functional.grid_sample(
            source_chunk,
            grid.expand(end - start, -1, -1, -1),
            mode=interpolation,
            padding_mode="border",
            align_corners=False,
        )
        if protection is not None:
            protection_chunk = protection if protection.shape[0] == 1 else protection[start:end]
            corrected_chunk = (
                corrected_chunk * (1.0 - protection_chunk)
                + source_chunk * protection_chunk
            )
        corrected_chunks.append(corrected_chunk.to(dtype=images.dtype).movedim(1, -1))

    corrected = torch.cat(corrected_chunks, dim=0).clamp(0.0, 1.0)
    return corrected, correction_mask.to(dtype=images.dtype)


class LTXReframeLayout:
    """Upload, position, and mask a source video for downstream LTX outpainting."""

    DESCRIPTION = (
        "Preview, zoom, and place a source video on an LTX-compatible canvas. "
        "Outputs deterministic padded frames and masks; connect them to "
        "LTXVInpaintPreprocess and the LTX IC-LoRA outpaint workflow."
    )

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            name
            for name in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, name))
        ]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return {
            "required": {
                "source_file": (sorted(files), {"video_upload": True}),
                "aspect_ratio": (ASPECT_PRESETS, {"default": "9:16"}),
                "target_width": (
                    "INT",
                    {"default": 1088, "min": 256, "max": 8192, "step": DIMENSION_MULTIPLE},
                ),
                "target_height": (
                    "INT",
                    {"default": 1920, "min": 256, "max": 8192, "step": DIMENSION_MULTIPLE},
                ),
                "position_x": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001},
                ),
                "position_y": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001},
                ),
                "source_scale": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": MIN_SOURCE_SCALE,
                        "max": MAX_SOURCE_SCALE,
                        "step": 0.001,
                        "round": 0.001,
                    },
                ),
                "feather": (
                    "INT",
                    {"default": 32, "min": 0, "max": 512, "step": 1},
                ),
            },
            "optional": {
                "limit_frames": ("BOOLEAN", {"default": False}),
                "first_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2147483647, "step": 1},
                ),
                "last_frame": (
                    "INT",
                    {"default": -1, "min": -1, "max": 2147483647, "step": 1},
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "MASK",
        "IMAGE",
        "AUDIO",
        "FLOAT",
        "VIDEO",
        "INT",
        "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "padded_frames",
        "outpaint_mask",
        "source_mask",
        "source_frames",
        "audio",
        "fps",
        "source_video",
        "target_width",
        "target_height",
        "placement_json",
    )
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(
        self,
        source_file: str,
        aspect_ratio: str,
        target_width: int,
        target_height: int,
        position_x: float,
        position_y: float,
        source_scale: float,
        feather: int,
        limit_frames: bool = False,
        first_frame: int = 0,
        last_frame: int = -1,
    ) -> tuple[Any, ...]:
        video_path = _resolve_input_video_path(source_file)
        original_source_video = InputImpl.VideoFromFile(video_path)
        source_width, source_height = original_source_video.get_dimensions()
        fps_fraction = Fraction(original_source_video.get_frame_rate())
        fps = float(fps_fraction)
        frame_range = resolve_frame_range(
            original_source_video.get_frame_count(),
            fps,
            limit_frames,
            first_frame,
            last_frame,
        )

        if frame_range.range_enabled:
            decode_source = InputImpl.VideoFromFile(
                video_path,
                start_time=frame_range.start_time,
                duration=frame_range.duration,
            )
            decoded = decode_source.get_components()
            decoded_count = int(decoded.images.shape[0])
            if decoded_count < frame_range.selected_frame_count:
                raise ValueError(
                    "The selected video range decoded fewer frames than expected "
                    f"({decoded_count} of {frame_range.selected_frame_count})."
                )
            images = decoded.images[: frame_range.selected_frame_count]
            alpha = (
                decoded.alpha[: frame_range.selected_frame_count]
                if decoded.alpha is not None
                else None
            )
            audio, synthesized_silent_audio = fit_audio_track(
                decoded.audio,
                frame_range.selected_frame_count,
                fps,
            )
            components = Types.VideoComponents(
                images=images,
                alpha=alpha,
                audio=audio,
                frame_rate=fps_fraction,
                metadata=decoded.metadata,
            )
            source_video = InputImpl.VideoFromComponents(components)
        else:
            components = original_source_video.get_components()
            audio, synthesized_silent_audio = ensure_audio_track(
                components.audio,
                components.images.shape[0],
                components.frame_rate,
            )
            source_video = original_source_video

        placement = compute_reframe_placement(
            source_width,
            source_height,
            target_width,
            target_height,
            position_x,
            position_y,
            source_scale,
        )
        padded_frames = prepare_reframe_canvas(components.images, placement)
        outpaint_mask, source_mask = build_reframe_masks(
            placement,
            feather,
            device=padded_frames.device,
            dtype=padded_frames.dtype,
        )
        metadata = {
            "schema": "ltx_reframe_layout/v2",
            "source_file": source_file,
            "aspect_ratio_preset": aspect_ratio,
            "feather": int(feather),
            "frame_count": int(components.images.shape[0]),
            "fps": float(components.frame_rate),
            "source_frame_count": frame_range.source_frame_count,
            "range_enabled": frame_range.range_enabled,
            "requested_first_frame": frame_range.requested_first_frame,
            "requested_last_frame": frame_range.requested_last_frame,
            "effective_first_frame": frame_range.effective_first_frame,
            "effective_last_frame": frame_range.effective_last_frame,
            "selected_frame_count": frame_range.selected_frame_count,
            "start_time": frame_range.start_time,
            "duration": frame_range.duration,
            "source_has_audio": not synthesized_silent_audio,
            "audio_fallback": "duration_matched_silence" if synthesized_silent_audio else "none",
            "placement": asdict(placement),
        }
        return (
            padded_frames,
            outpaint_mask,
            source_mask,
            components.images,
            audio,
            float(components.frame_rate),
            source_video,
            placement.target_width,
            placement.target_height,
            json.dumps(metadata, indent=2, sort_keys=True),
        )

    @classmethod
    def IS_CHANGED(cls, source_file: str, **_kwargs):
        video_path = _resolve_input_video_path(source_file)
        return os.path.getmtime(video_path)

    @classmethod
    def VALIDATE_INPUTS(cls, source_file: str, **_kwargs):
        try:
            _resolve_input_video_path(source_file)
        except ValueError as error:
            return f"Invalid video file: {source_file}. {error}"
        return True


class LTXReframeTargetFit:
    """Center-fit an upscaled frame batch to the reframe node's exact canvas."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "target_width": ("INT", {"forceInput": True}),
                "target_height": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "fit"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Center-crop and Lanczos-resize a learned LTX upscale to the exact "
        "canvas dimensions produced by LTX Reframe Layout."
    )

    def fit(
        self,
        images: torch.Tensor,
        target_width: int,
        target_height: int,
    ) -> tuple[torch.Tensor]:
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[-1] < 3:
            raise ValueError("LTX Reframe Target Fit requires a non-empty IMAGE batch.")
        target_width = max(1, int(target_width))
        target_height = max(1, int(target_height))
        if images.shape[2] == target_width and images.shape[1] == target_height:
            return (images,)
        fitted = comfy.utils.common_upscale(
            images.movedim(-1, 1),
            target_width,
            target_height,
            "lanczos",
            "center",
        ).movedim(1, -1)
        return (fitted.clamp(0.0, 1.0),)


class LTXReframeRectilinearCorrection:
    """Temporally stable, mask-aware correction for generated fisheye curvature."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "correction_strength": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 0.25,
                        "step": 0.005,
                        "round": 0.005,
                    },
                ),
                "center_x": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "center_y": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "protect_source": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": 0.05,
                    },
                ),
                "protection_feather": (
                    "INT",
                    {"default": 96, "min": 0, "max": 512, "step": 8},
                ),
                "interpolation": (["bicubic", "bilinear"], {"default": "bicubic"}),
                "chunk_size": (
                    "INT",
                    {"default": 8, "min": 1, "max": 64, "step": 1},
                ),
            },
            "optional": {
                "source_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "correction_mask")
    FUNCTION = "correct"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Apply a deterministic barrel-to-rectilinear correction after the final "
        "Stage 2 fit. Connect LTX Reframe Layout's source_mask to protect original "
        "pixels. Positive strength removes fisheye curvature and crops edges "
        "without creating empty borders. Start at 0.04-0.06."
    )

    def correct(
        self,
        images: torch.Tensor,
        enabled: bool,
        correction_strength: float,
        center_x: float,
        center_y: float,
        protect_source: float,
        protection_feather: int,
        interpolation: str,
        chunk_size: int,
        source_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[-1] < 1:
            raise ValueError("Rectilinear correction requires a non-empty IMAGE batch.")
        if not bool(enabled):
            return (
                images,
                torch.zeros(
                    (1, images.shape[1], images.shape[2]),
                    device=images.device,
                    dtype=images.dtype,
                ),
            )
        return rectilinear_correct_images(
            images,
            correction_strength,
            center_x,
            center_y,
            protect_source,
            protection_feather,
            interpolation,
            chunk_size,
            source_mask,
        )


NODE_CLASS_MAPPINGS = {
    "LTXReframeLayout": LTXReframeLayout,
    "LTXReframeTargetFit": LTXReframeTargetFit,
    "LTXReframeRectilinearCorrection": LTXReframeRectilinearCorrection,
    "LTXReframeCustomAudioLoader": LTXReframeCustomAudioLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXReframeLayout": "LTX Reframe Layout",
    "LTXReframeTargetFit": "LTX Reframe Target Fit",
    "LTXReframeRectilinearCorrection": "LTX Reframe Rectilinear Correction",
    "LTXReframeCustomAudioLoader": "LTX Reframe Custom Audio Loader",
}


def _register_video_metadata_route() -> None:
    try:
        from server import PromptServer
    except ImportError:
        return

    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None or getattr(prompt_server, "_ltx_reframe_metadata_route", False):
        return

    @prompt_server.routes.get("/ltx_reframe/video_metadata")
    async def ltx_reframe_video_metadata(request):
        source_file = request.query.get("source_file", "")
        if not source_file:
            return web.json_response({"error": "source_file is required"}, status=400)
        try:
            return web.json_response(probe_video_metadata(source_file))
        except (OSError, RuntimeError, ValueError) as error:
            return web.json_response({"error": str(error)}, status=400)

    prompt_server._ltx_reframe_metadata_route = True


_register_video_metadata_route()
