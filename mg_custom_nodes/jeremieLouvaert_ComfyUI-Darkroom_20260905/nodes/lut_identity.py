"""
LUT Identity Generator node for ComfyUI-Darkroom.
Outputs a neutral identity lattice image — connect your Darkroom chain after this,
then feed the result into LUT Export to bake a .cube file.
"""

import numpy as np
import torch

from ..utils.lut import generate_identity_lut


class LUTIdentity:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lut_size": (["17", "33", "65"], {
                    "default": "33",
                    "tooltip": "LUT resolution — 33 is standard for most tools (DaVinci, Premiere, Photoshop). "
                               "17 is lighter, 65 is maximum precision"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("identity_lattice", "lut_size")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Pipeline"

    def execute(self, lut_size="33"):
        size = int(lut_size)
        print(f"[Darkroom] LUT Identity: generating {size}x{size}x{size} lattice "
              f"({size}x{size * size} image, {size ** 3:,} color samples)")

        identity = generate_identity_lut(size)

        # Convert to ComfyUI tensor (B, H, W, C)
        tensor = torch.from_numpy(identity).unsqueeze(0)

        return (tensor, size)


NODE_CLASS_MAPPINGS = {"DarkroomLUTIdentity": LUTIdentity}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomLUTIdentity": "LUT Identity Generator"}
