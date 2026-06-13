from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch

try:
    import comfy.model_management as model_management
except Exception:  # pragma: no cover - only available inside ComfyUI
    model_management = None

try:
    from .pid_decode import (
        BACKBONE_CHOICES,
        PID_BASE_RESOLUTIONS,
        PID_BACKBONES,
        PiDNodeError,
        _pid_base_resolution_labels,
        _pid_base_resolution_size_map,
    )
except ImportError:  # Allows `python -c "import pid_empty_latent"` from this folder.
    from pid_decode import (
        BACKBONE_CHOICES,
        PID_BASE_RESOLUTIONS,
        PID_BACKBONES,
        PiDNodeError,
        _pid_base_resolution_labels,
        _pid_base_resolution_size_map,
    )


PID_EMPTY_LATENT_RESOLUTIONS: Dict[str, List[Tuple[str, int, int]]] = {
    mode: list(entries) for mode, entries in PID_BASE_RESOLUTIONS.items()
}

# Default preserves older workflows that omitted the new backbone widget.
LATENT_CHANNELS = 16
LATENT_DOWNSCALE = 8

_RESOLUTION_BY_MODE_AND_LABEL = {
    mode: _pid_base_resolution_size_map(mode)
    for mode in PID_EMPTY_LATENT_RESOLUTIONS
}

ALL_RESOLUTION_LABELS: List[str] = []
for _mode in PID_EMPTY_LATENT_RESOLUTIONS:
    for _label in _pid_base_resolution_labels(_mode):
        if _label not in ALL_RESOLUTION_LABELS:
            ALL_RESOLUTION_LABELS.append(_label)

_RESOLUTION_RE = re.compile(r"^\s*(\d+)\s*[xX×]\s*(\d+)")


def _parse_resolution_label(label: str) -> Tuple[int, int]:
    match = _RESOLUTION_RE.match(str(label))
    if not match:
        raise PiDNodeError(f"Could not parse PiD empty latent resolution: {label!r}")
    return int(match.group(1)), int(match.group(2))


def _intermediate_device():
    if model_management is not None:
        try:
            return model_management.intermediate_device()
        except Exception:
            pass
    return torch.device("cpu")


class PiDEmptyLatentImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pid_ckpt_type": (["2k", "2kto4k"], {"default": "2k"}),
                "resolution": (ALL_RESOLUTION_LABELS, {"default": "512x512 (1:1)"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "backbone": (BACKBONE_CHOICES, {"default": "sd3"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "PiD"

    def generate(
        self,
        pid_ckpt_type: str = "2k",
        resolution: str = "512x512 (1:1)",
        batch_size: int = 1,
        backbone: str = "sd3",
    ):
        pid_ckpt_type = str(pid_ckpt_type).strip()
        backbone = str(backbone or "sd3").strip()

        if pid_ckpt_type not in PID_EMPTY_LATENT_RESOLUTIONS:
            raise PiDNodeError("Unknown PiD checkpoint type for empty latent: " f"{pid_ckpt_type!r}")
        if backbone not in PID_BACKBONES:
            raise PiDNodeError(f"Unknown PiD empty latent backbone={backbone!r}; expected one of {BACKBONE_CHOICES}")
        backbone_info = PID_BACKBONES[backbone]
        if pid_ckpt_type not in backbone_info.ckpt_types:
            supported = ", ".join(backbone_info.ckpt_types)
            raise PiDNodeError(
                f"{backbone_info.label} empty latents do not support pid_ckpt_type={pid_ckpt_type!r}. "
                f"Supported values: {supported}."
            )

        choices_for_mode = _RESOLUTION_BY_MODE_AND_LABEL[pid_ckpt_type]
        if resolution in choices_for_mode:
            width, height = choices_for_mode[resolution]
        else:
            width, height = _parse_resolution_label(resolution)
            valid = ", ".join(choices_for_mode.keys())
            raise PiDNodeError(
                f"Resolution {resolution!r} is not one of the {pid_ckpt_type} PiD empty latent presets. "
                f"Choose one of: {valid}."
            )

        latent_downscale = int(backbone_info.latent_downscale)
        latent_channels = int(backbone_info.latent_channels)
        if width % latent_downscale or height % latent_downscale:
            raise PiDNodeError(
                f"Resolution {width}x{height} is not divisible by {latent_downscale}, "
                f"which is required for {backbone_info.label} PiD empty latents."
            )

        batch_size = max(1, int(batch_size))
        latent_width = width // latent_downscale
        latent_height = height // latent_downscale
        samples = torch.zeros(
            [batch_size, latent_channels, latent_height, latent_width],
            device=_intermediate_device(),
        )
        return (
            {
                "samples": samples,
                "downscale_ratio_spacial": latent_downscale,
                "pid_source_backbone": backbone,
            },
        )


NODE_CLASS_MAPPINGS = {
    "PiDEmptyLatentImage": PiDEmptyLatentImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDEmptyLatentImage": "PiD Empty Latent Image",
}
