"""
CMYK Export TIFF node. Converts the incoming sRGB image to CMYK via the
chosen ICC profile + rendering intent and writes a 4-channel TIFF with the
ICC profile embedded. The file is what goes to the printer.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from ..utils.image import tensor_to_numpy_batch
from ..utils.cmyk import discover_profiles, export_cmyk_tiff, INTENT_NAMES


_PROFILES = discover_profiles()
_LABELS = [lbl for lbl, _, _ in _PROFILES]
_LABEL_TO_PATH = {lbl: str(p) for lbl, p, _ in _PROFILES}


def _default_output_dir() -> str:
    """
    Default to ComfyUI's output/ if we can find it, else current cwd/output.
    """
    try:
        import folder_paths  # ComfyUI runtime module
        return os.path.join(folder_paths.get_output_directory(), "cmyk")
    except Exception:
        return os.path.join(os.getcwd(), "output", "cmyk")


class CMYKExportTIFF:

    @classmethod
    def INPUT_TYPES(cls):
        choices = _LABELS if _LABELS else ["(no CMYK profiles found)"]
        return {
            "required": {
                "image": ("IMAGE",),
                "target_profile": (choices, {
                    "default": choices[0],
                    "tooltip": "CMYK profile embedded in the output TIFF"
                }),
                "intent": (list(INTENT_NAMES.keys()), {
                    "default": "perceptual",
                }),
                "filename_prefix": ("STRING", {
                    "default": "darkroom_cmyk",
                    "tooltip": "Prefix for the output filename. A timestamp is appended."
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory to save to. Blank = ComfyUI output/cmyk/"
                }),
                "dpi": ("INT", {
                    "default": 300, "min": 72, "max": 1200, "step": 12,
                    "tooltip": "DPI metadata stored in the TIFF"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Print"
    OUTPUT_NODE = True

    def execute(self, image, target_profile, intent, filename_prefix, output_dir, dpi):
        if not _LABELS or target_profile not in _LABEL_TO_PATH:
            raise RuntimeError(
                "[Darkroom] CMYK Export: no CMYK profile available. "
                "Ensure system ICC profiles exist, or drop a .icc into data/icc_profiles/"
            )

        out_dir = output_dir.strip() or _default_output_dir()
        os.makedirs(out_dir, exist_ok=True)

        path = _LABEL_TO_PATH[target_profile]
        arrs = tensor_to_numpy_batch(image)
        ts = time.strftime("%Y%m%d_%H%M%S")

        saved_paths: list[str] = []
        for idx, arr in enumerate(arrs):
            suffix = f"_{idx:02d}" if len(arrs) > 1 else ""
            out_name = f"{filename_prefix}_{ts}{suffix}.tif"
            out_path = os.path.join(out_dir, out_name)
            saved = export_cmyk_tiff(
                np.clip(arr, 0.0, 1.0),
                target_path=path,
                intent=intent,
                out_path=out_path,
                dpi=int(dpi),
            )
            saved_paths.append(saved)
            print(f"[Darkroom] CMYK Export: wrote {saved} "
                  f"(profile {Path(path).name}, {intent}, {dpi} DPI)")

        return (saved_paths[0] if len(saved_paths) == 1 else "\n".join(saved_paths),)


NODE_CLASS_MAPPINGS = {"DarkroomCMYKExportTIFF": CMYKExportTIFF}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomCMYKExportTIFF": "CMYK Export TIFF"}
