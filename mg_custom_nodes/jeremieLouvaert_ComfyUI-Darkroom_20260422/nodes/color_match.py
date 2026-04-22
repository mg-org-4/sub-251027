"""
Color Match node for ComfyUI-Darkroom.
Grade a target image toward a reference's colour distribution.

Four methods, all operating in LAB space (classical colour-science approach,
not neural):
  * reinhard     -- mean/std transfer. Fast, safe default. Best on single-
                    subject scenes with similar composition.
  * wasserstein  -- sliced optimal transport via iterative advection. Handles
                    multi-modal distributions where Reinhard's linear shift
                    flattens. 20 iterations by default; more is not always
                    better (quantile noise overshoots past ~50).
  * forgy        -- K-means palette matching with Gaussian-weighted soft
                    cluster assignment. Preserves local colour regions
                    better than Reinhard on scenes with distinct palettes.
                    Requires scikit-learn (ships with ComfyUI).
  * kantorovich  -- closed-form Gaussian linear transport. Monge map under
                    Gaussian assumption. Requires the POT library
                    (pip install POT). Falls back to reinhard if POT
                    missing.

Attribution: algorithm implementations from rajawski/gradia (MIT).
"""

from __future__ import annotations

import numpy as np

from ..utils.color import blend
from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor
from ..utils.colormatch import (
    color_match_reinhard,
    color_match_wasserstein,
    color_match_forgy,
    color_match_kantorovich,
    is_method_available,
)


class ColorMatch:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "The image to grade (target)"
                }),
                "reference": ("IMAGE", {
                    "tooltip": "The image whose colour character to match"
                }),
                "method": (["reinhard", "wasserstein", "forgy", "kantorovich"], {
                    "default": "reinhard",
                    "tooltip": "reinhard: fast mean/std. wasserstein: multi-modal OT. "
                               "forgy: palette K-means (sklearn). kantorovich: Gaussian OT (POT)."
                }),
                "intensity": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blend between original (0) and full match (1)"
                }),
            },
            "optional": {
                "n_colors": ("INT", {
                    "default": 8, "min": 4, "max": 16, "step": 1,
                    "tooltip": "Forgy only: palette size"
                }),
                "n_slices": ("INT", {
                    "default": 20, "min": 4, "max": 200, "step": 2,
                    "tooltip": "Wasserstein only: advection iterations"
                }),
                "sample_size": ("INT", {
                    "default": 50000, "min": 5000, "max": 500000, "step": 5000,
                    "tooltip": "Wasserstein/Kantorovich/Forgy: pixels sampled for fitting"
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "Random seed (determinism)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Grading"

    def execute(self, image, reference, method, intensity,
                n_colors=8, n_slices=20, sample_size=50000, seed=42):
        if intensity <= 0.0:
            return (image,)

        available, reason = is_method_available(method)
        if not available:
            print(f"[Darkroom] Color Match: {method} unavailable ({reason}); falling back to reinhard")
            method = "reinhard"

        # Reference is a single image; target may be a batch. Use the first
        # frame of the reference for everything.
        ref_arrs = tensor_to_numpy_batch(reference)
        ref = ref_arrs[0]

        tgt_arrs = tensor_to_numpy_batch(image)
        out = []
        for tgt in tgt_arrs:
            if method == "reinhard":
                graded = color_match_reinhard(tgt, ref)
            elif method == "wasserstein":
                graded = color_match_wasserstein(
                    tgt, ref, n_slices=int(n_slices),
                    sample_size=int(sample_size), seed=int(seed),
                )
            elif method == "forgy":
                graded = color_match_forgy(
                    tgt, ref, n_colors=int(n_colors),
                    sample_size=int(sample_size), seed=int(seed),
                )
            elif method == "kantorovich":
                graded = color_match_kantorovich(
                    tgt, ref, sample_size=int(sample_size), seed=int(seed),
                )
            else:
                graded = tgt

            out.append(blend(tgt, graded, float(intensity)))

        print(f"[Darkroom] Color Match: method={method} intensity={intensity:.2f} "
              f"(batch={len(tgt_arrs)})")
        return (numpy_batch_to_tensor(out),)


NODE_CLASS_MAPPINGS = {"DarkroomColorMatch": ColorMatch}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomColorMatch": "Color Match (Reference)"}
