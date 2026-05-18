"""
Color Space Transform node for ComfyUI-Darkroom.
Convert between sRGB, Linear sRGB, ACEScg, ACEScct, Rec.2020, and DCI-P3.
Makes Darkroom the only ACES-aware toolset in ComfyUI.
"""

import numpy as np

from ..utils.color import blend
from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor
from ..utils.colorspace import convert_colorspace, SPACE_NAMES


class ColorSpaceTransform:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_space": (SPACE_NAMES, {
                    "default": "sRGB",
                    "tooltip": "Color space of the input image"
                }),
                "target_space": (SPACE_NAMES, {
                    "default": "ACEScg",
                    "tooltip": "Color space for the output image"
                }),
            },
            "optional": {
                "gamut_clip": (["Clip", "Soft Compress"], {
                    "default": "Clip",
                    "tooltip": "How to handle out-of-gamut values. Clip = hard clamp to [0,1]. "
                               "Soft Compress = gently roll off values approaching gamut boundary"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and transformed (1)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Pipeline"

    def execute(self, image, source_space="sRGB", target_space="ACEScg",
                gamut_clip="Clip", strength=1.0):

        if strength <= 0.0 or source_space == target_space:
            return (image,)

        print(f"[Darkroom] Color Space Transform: {source_space} → {target_space}")

        images = tensor_to_numpy_batch(image)
        results = []

        for img in images:
            original = img.copy()
            converted = convert_colorspace(img, source_space, target_space)

            # Gamut handling
            if gamut_clip == "Soft Compress":
                # Soft compression: smoothly map values outside [0,1] back in
                # Using a simple knee function at the boundaries
                converted = self._soft_compress(converted)
            else:
                converted = np.clip(converted, 0.0, 1.0)

            results.append(blend(original, converted.astype(np.float32), strength))

        return (numpy_batch_to_tensor(results),)

    @staticmethod
    def _soft_compress(img, knee=0.9):
        """
        Soft-compress values outside [0, 1] using a smooth knee.
        Values below knee/above (1-knee) pass through linearly.
        Values beyond are compressed asymptotically toward the boundary.
        """
        result = img.copy()

        # Compress highlights (values above knee toward 1.0)
        mask_hi = result > knee
        if np.any(mask_hi):
            excess = result[mask_hi] - knee
            compressed = knee + (1.0 - knee) * (1.0 - np.exp(-excess / (1.0 - knee + 1e-10)))
            result[mask_hi] = compressed

        # Compress shadows (values below 1-knee toward 0.0)
        neg_knee = 1.0 - knee  # 0.1 for knee=0.9
        mask_lo = result < neg_knee
        if np.any(mask_lo):
            deficit = neg_knee - result[mask_lo]
            compressed = neg_knee - neg_knee * (1.0 - np.exp(-deficit / (neg_knee + 1e-10)))
            result[mask_lo] = compressed

        return np.clip(result, 0.0, 1.0).astype(np.float32)


NODE_CLASS_MAPPINGS = {"DarkroomColorSpaceTransform": ColorSpaceTransform}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomColorSpaceTransform": "Color Space Transform"}
