"""
CMYK TAC (Total Area Coverage) Check node. Flags pixels whose C+M+Y+K ink
sum exceeds the target stock's TAC limit. Useful before sending a file to
print -- exceeding TAC causes ink adhesion problems, drying failures, and
show-through on the back of the sheet.
"""

from __future__ import annotations

import numpy as np

from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor
from ..utils.cmyk import discover_profiles, tac_check, INTENT_NAMES, TAC_LIMITS


_PROFILES = discover_profiles()
_LABELS = [lbl for lbl, _, _ in _PROFILES]
_LABEL_TO_PATH = {lbl: str(p) for lbl, p, _ in _PROFILES}


class CMYKTACCheck:

    @classmethod
    def INPUT_TYPES(cls):
        choices = _LABELS if _LABELS else ["(no CMYK profiles found)"]
        tac_presets = list(TAC_LIMITS.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "target_profile": (choices, {
                    "default": choices[0],
                    "tooltip": "CMYK print stock"
                }),
                "intent": (list(INTENT_NAMES.keys()), {
                    "default": "perceptual",
                }),
                "tac_preset": (tac_presets, {
                    "default": tac_presets[0],
                    "tooltip": "TAC limit by stock type. 'custom' uses the numeric field below."
                }),
            },
            "optional": {
                "custom_tac": ("INT", {
                    "default": 330, "min": 150, "max": 400, "step": 5,
                    "tooltip": "Only used when tac_preset is 'custom'"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Print"

    def execute(self, image, target_profile, intent, tac_preset, custom_tac=330):
        if not _LABELS or target_profile not in _LABEL_TO_PATH:
            return (image,)

        preset_limit = TAC_LIMITS.get(tac_preset)
        tac_limit = int(preset_limit) if preset_limit is not None else int(custom_tac)

        path = _LABEL_TO_PATH[target_profile]
        arrs = tensor_to_numpy_batch(image)
        out = []
        first_stats = None
        for a in arrs:
            overlay, frac, tac_max = tac_check(
                np.clip(a, 0.0, 1.0), path, intent, tac_limit=tac_limit,
            )
            out.append(overlay)
            if first_stats is None:
                first_stats = (frac, tac_max)
        if first_stats is not None:
            frac, tac_max = first_stats
            print(f"[Darkroom] CMYK TAC Check: {target_profile} ({intent}) "
                  f"-- limit {tac_limit}%, peak {tac_max:.1f}%, "
                  f"{frac*100:.2f}% of pixels exceeded")
        return (numpy_batch_to_tensor(out),)


NODE_CLASS_MAPPINGS = {"DarkroomCMYKTACCheck": CMYKTACCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomCMYKTACCheck": "CMYK TAC Check"}
