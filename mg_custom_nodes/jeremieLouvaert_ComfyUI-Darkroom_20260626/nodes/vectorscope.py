"""
Vectorscope node for ComfyUI-Darkroom.
Renders Rec.709 YCbCr chroma density with primary target boxes + skin line.
"""

from __future__ import annotations

import numpy as np
import torch

from ..utils.scopes import render_vectorscope


class Vectorscope:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("INT", {
                    "default": 512, "min": 192, "max": 2048, "step": 32,
                    "tooltip": "Scope canvas size in pixels (square)"
                }),
                "gain": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                    "tooltip": "Radial scaling of the trace. 1.0 = normal, >1 zooms in on low saturation"
                }),
                "log_scale": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Log-scale the density (single-hue scenes read better with log)"
                }),
                "show_skin_line": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Draw the I-line at 123 degrees (natural skin-tone axis)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("scope",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Scopes"

    def execute(self, image, size, gain, log_scale, show_skin_line):
        arr = image[0].detach().cpu().numpy().astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        scope = render_vectorscope(
            arr, size=int(size), gain=float(gain),
            log_scale=bool(log_scale), show_skin_line=bool(show_skin_line),
        )
        out = torch.from_numpy(scope[None, ...])
        return (out,)


NODE_CLASS_MAPPINGS = {"DarkroomVectorscope": Vectorscope}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomVectorscope": "Vectorscope"}
