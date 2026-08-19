"""
Histogram scope node for ComfyUI-Darkroom.
Renders per-channel or luma histogram with graticule + clip warnings.
"""

from __future__ import annotations

import numpy as np
import torch

from ..utils.scopes import render_histogram


class Histogram:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["RGB", "Luma", "R", "G", "B"], {
                    "default": "RGB",
                    "tooltip": "RGB = overlaid channels. Luma = Rec.709 luminance. R/G/B = single channel."
                }),
                "width": ("INT", {
                    "default": 512, "min": 128, "max": 2048, "step": 32,
                    "tooltip": "Scope width in pixels"
                }),
                "height": ("INT", {
                    "default": 256, "min": 96, "max": 1024, "step": 16,
                    "tooltip": "Scope height in pixels"
                }),
                "log_scale": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Log-scale the counts (useful when highlights swamp midtones)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("scope",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Scopes"

    def execute(self, image, mode, width, height, log_scale):
        # ComfyUI IMAGE: (B, H, W, C). Score only the first image in the batch.
        arr = image[0].detach().cpu().numpy().astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        scope = render_histogram(
            arr, mode=mode, width=int(width), height=int(height),
            log_scale=bool(log_scale),
        )
        out = torch.from_numpy(scope[None, ...])
        return (out,)


NODE_CLASS_MAPPINGS = {"DarkroomHistogram": Histogram}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomHistogram": "Histogram"}
