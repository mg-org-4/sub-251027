"""
Chromatic Aberration node for ComfyUI-Darkroom.

Simulates lateral chromatic aberration -- the color fringing caused by a lens
failing to focus all wavelengths to the same point.

Select a real lens profile or use Custom for manual control.

GPU-accelerated via torch. No CPU roundtrip.
"""

import math
import torch

from ..data.lens_profiles import LENS_PROFILES_FLAT, LENS_PROFILE_NAMES
from ..utils.torch_ops import pixel_to_grid_coords, grid_sample_channel


PRESET_NAMES = ["Custom"] + LENS_PROFILE_NAMES


def _apply_lateral_ca(img, shift_r, shift_b):
    """
    Apply lateral CA by radially scaling R and B channels on GPU.
    Green channel stays fixed (reference). Shift in pixels at the image edge.
    img: (H, W, C) tensor on device.
    """
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    max_r = math.sqrt(cx * cx + cy * cy)
    device = img.device

    yy = torch.arange(h, dtype=torch.float32, device=device)
    xx = torch.arange(w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(yy, xx, indexing='ij')
    dy = yy - cy
    dx = xx - cx
    r = torch.sqrt(dy * dy + dx * dx) / max_r

    result = img.clone()

    for c, shift in enumerate([shift_r, 0.0, shift_b]):
        if abs(shift) < 0.01:
            continue

        scale = 1.0 + (shift / max_r) * r
        new_y = cy + dy * scale
        new_x = cx + dx * scale

        grid = pixel_to_grid_coords(new_y, new_x, h, w)
        result[..., c] = grid_sample_channel(img[..., c], grid,
                                             padding_mode='reflection')

    return result.clamp(0.0, 1.0)


class ChromaticAberration:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lens": (PRESET_NAMES, {
                    "default": "Custom",
                    "tooltip": "Select a real lens or 'Custom' for manual CA control"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "Overall CA intensity multiplier"
                }),
            },
            "optional": {
                "shift_r": ("FLOAT", {
                    "default": -1.0, "min": -5.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Red channel shift in pixels at image edge. Negative = inward"
                }),
                "shift_b": ("FLOAT", {
                    "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Blue channel shift in pixels at image edge. Positive = outward"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Lens"

    def execute(self, image, lens, strength, shift_r=-1.0, shift_b=1.0):
        if strength <= 0.0:
            return (image,)

        if lens != "Custom" and lens in LENS_PROFILES_FLAT:
            p = LENS_PROFILES_FLAT[lens]
            shift_r = p.ca_r
            shift_b = p.ca_b

        results = []
        for i in range(image.shape[0]):
            img = image[i]
            h, w = img.shape[:2]
            scale = min(h, w) / 1024.0
            sr = shift_r * strength * scale
            sb = shift_b * strength * scale

            print(f"[Darkroom] Chromatic Aberration: {lens}, shift_r={sr:.2f}px, shift_b={sb:.2f}px")
            results.append(_apply_lateral_ca(img, sr, sb))

        return (torch.stack(results, dim=0),)


NODE_CLASS_MAPPINGS = {
    "DarkroomChromaticAberration": ChromaticAberration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomChromaticAberration": "Chromatic Aberration",
}
