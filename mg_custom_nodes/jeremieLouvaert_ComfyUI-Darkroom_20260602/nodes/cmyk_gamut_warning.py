"""
CMYK Gamut Warning node. Highlights pixels that shift more than `threshold`
on a sRGB -> CMYK -> sRGB round-trip against the chosen print profile.
The overlaid pixels are the ones that cannot be accurately reproduced by the
chosen print condition.
"""

from __future__ import annotations

import numpy as np

from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor
from ..utils.cmyk import discover_profiles, gamut_warning, INTENT_NAMES


_PROFILES = discover_profiles()
_LABELS = [lbl for lbl, _, _ in _PROFILES]
_LABEL_TO_PATH = {lbl: str(p) for lbl, p, _ in _PROFILES}


class CMYKGamutWarning:

    @classmethod
    def INPUT_TYPES(cls):
        choices = _LABELS if _LABELS else ["(no CMYK profiles found)"]
        return {
            "required": {
                "image": ("IMAGE",),
                "target_profile": (choices, {
                    "default": choices[0],
                    "tooltip": "CMYK print stock to check gamut against"
                }),
                "intent": (list(INTENT_NAMES.keys()), {
                    "default": "perceptual",
                    "tooltip": "Rendering intent used when the image gets printed"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.03, "min": 0.001, "max": 0.2, "step": 0.005,
                    "tooltip": "RGB distance above which a pixel is flagged out-of-gamut"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Print"

    def execute(self, image, target_profile, intent, threshold):
        if not _LABELS or target_profile not in _LABEL_TO_PATH:
            return (image,)

        path = _LABEL_TO_PATH[target_profile]
        arrs = tensor_to_numpy_batch(image)
        out = [gamut_warning(np.clip(a, 0.0, 1.0), path, intent, threshold=float(threshold))
               for a in arrs]

        # Compute fraction out-of-gamut for the first image for the log.
        if arrs:
            from ..utils.cmyk import _softproof_transform
            from PIL import ImageCms, Image as PImage
            first = np.clip(arrs[0], 0.0, 1.0)
            u8 = (first * 255.0).astype(np.uint8)
            pil = PImage.fromarray(u8, mode="RGB")
            proofed = np.asarray(
                ImageCms.applyTransform(pil, _softproof_transform(path, intent)),
                dtype=np.float32
            ) / 255.0
            diff = np.linalg.norm(first - proofed, axis=-1)
            frac = float((diff > threshold).mean()) * 100.0
            print(f"[Darkroom] CMYK Gamut Warning: {target_profile} ({intent}) "
                  f"-- {frac:.1f}% out-of-gamut at threshold {threshold}")

        return (numpy_batch_to_tensor(out),)


NODE_CLASS_MAPPINGS = {"DarkroomCMYKGamutWarning": CMYKGamutWarning}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomCMYKGamutWarning": "CMYK Gamut Warning"}
