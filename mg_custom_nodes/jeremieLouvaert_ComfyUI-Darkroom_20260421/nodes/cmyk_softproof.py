"""
CMYK Soft-Proof node. Simulates how the image will look after being
converted to the chosen CMYK print stock + viewed on an sRGB display.
Does not modify the image's actual colour space -- output stays RGB for
continued editing downstream.
"""

from __future__ import annotations

import numpy as np

from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor
from ..utils.cmyk import discover_profiles, soft_proof, INTENT_NAMES


_PROFILES = discover_profiles()
_LABELS = [lbl for lbl, _, _ in _PROFILES]
_LABEL_TO_PATH = {lbl: str(p) for lbl, p, _ in _PROFILES}


class CMYKSoftProof:

    @classmethod
    def INPUT_TYPES(cls):
        choices = _LABELS if _LABELS else ["(no CMYK profiles found)"]
        return {
            "required": {
                "image": ("IMAGE",),
                "target_profile": (choices, {
                    "default": choices[0],
                    "tooltip": "CMYK print stock to simulate"
                }),
                "intent": (list(INTENT_NAMES.keys()), {
                    "default": "perceptual",
                    "tooltip": "perceptual for photos, relative colorimetric for logos, "
                               "saturation for charts, absolute for pre-press proofing"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Print"

    def execute(self, image, target_profile, intent):
        if not _LABELS or target_profile not in _LABEL_TO_PATH:
            print("[Darkroom] CMYK Soft-Proof: no profile selected / found, passing through")
            return (image,)

        path = _LABEL_TO_PATH[target_profile]
        arrs = tensor_to_numpy_batch(image)
        out = [soft_proof(np.clip(a, 0.0, 1.0), path, intent) for a in arrs]
        print(f"[Darkroom] CMYK Soft-Proof: {target_profile} ({intent})")
        return (numpy_batch_to_tensor(out),)


NODE_CLASS_MAPPINGS = {"DarkroomCMYKSoftProof": CMYKSoftProof}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomCMYKSoftProof": "CMYK Soft-Proof"}
