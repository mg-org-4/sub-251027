from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch

try:
    import comfy.model_management as model_management
except Exception:  # pragma: no cover - only available inside ComfyUI
    model_management = None

try:
    from .pid_decode import PiDNodeError
except ImportError:  # Allows `python -c "import pid_empty_latent"` from this folder.
    from pid_decode import PiDNodeError


# These are the base/LDM image sizes that PiD will decode from.
# Final PiD output is normally 4x these dimensions for the released LDM checkpoints.
#
# 2k presets follow NVIDIA's 2K training aspect-ratio family where available:
#   512x512 -> 2048x2048
#   576x432 -> 2304x1728
#   432x576 -> 1728x2304
#   672x384 -> 2688x1536
#   384x672 -> 1536x2688
# 3:2 and 21:9 are area-matched, exact-ratio additions near the same 0.25MP base.
#
# 2kto4k presets keep a 4K-class long edge for non-square aspect ratios. This
# matches NVIDIA's 4096x3072 / 1024x768 4:3 example and avoids extremely wide
# or tall >4K long-edge ultrawide presets by default.
PID_EMPTY_LATENT_RESOLUTIONS: Dict[str, List[Tuple[str, int, int]]] = {
    "2k": [
        ("512x512 (1:1)", 512, 512),
        ("576x432 (4:3)", 576, 432),
        ("432x576 (3:4)", 432, 576),
        ("624x416 (3:2)", 624, 416),
        ("416x624 (2:3)", 416, 624),
        ("672x384 (16:9)", 672, 384),
        ("384x672 (9:16)", 384, 672),
        ("784x336 (21:9)", 784, 336),
        ("336x784 (9:21)", 336, 784),
    ],
    "2kto4k": [
        ("1024x1024 (1:1)", 1024, 1024),
        ("1024x768 (4:3)", 1024, 768),
        ("768x1024 (3:4)", 768, 1024),
        ("1008x672 (3:2)", 1008, 672),
        ("672x1008 (2:3)", 672, 1008),
        ("1024x576 (16:9)", 1024, 576),
        ("576x1024 (9:16)", 576, 1024),
        ("1008x432 (21:9)", 1008, 432),
        ("432x1008 (9:21)", 432, 1008),
    ],
}

# This node intentionally mirrors EmptySD3LatentImage-style output:
#   [batch, 16, height / 8, width / 8]
# It only adds PiD-specific resolution presets and the 2k / 2kto4k switch.
LATENT_CHANNELS = 16
LATENT_DOWNSCALE = 8

_RESOLUTION_BY_MODE_AND_LABEL = {
    mode: {label: (width, height) for label, width, height in entries}
    for mode, entries in PID_EMPTY_LATENT_RESOLUTIONS.items()
}

ALL_RESOLUTION_LABELS: List[str] = []
for _mode_entries in PID_EMPTY_LATENT_RESOLUTIONS.values():
    for _label, _width, _height in _mode_entries:
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
    ):
        pid_ckpt_type = str(pid_ckpt_type).strip()

        if pid_ckpt_type not in PID_EMPTY_LATENT_RESOLUTIONS:
            raise PiDNodeError("Unknown PiD checkpoint type for empty latent: " f"{pid_ckpt_type!r}")

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

        if width % LATENT_DOWNSCALE or height % LATENT_DOWNSCALE:
            raise PiDNodeError(
                f"Resolution {width}x{height} is not divisible by {LATENT_DOWNSCALE}, "
                f"which is required for SD3-style PiD empty latents."
            )

        batch_size = max(1, int(batch_size))
        latent_width = width // LATENT_DOWNSCALE
        latent_height = height // LATENT_DOWNSCALE
        samples = torch.zeros(
            [batch_size, LATENT_CHANNELS, latent_height, latent_width],
            device=_intermediate_device(),
        )
        return ({"samples": samples},)


NODE_CLASS_MAPPINGS = {
    "PiDEmptyLatentImage": PiDEmptyLatentImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDEmptyLatentImage": "PiD Empty Latent Image",
}
