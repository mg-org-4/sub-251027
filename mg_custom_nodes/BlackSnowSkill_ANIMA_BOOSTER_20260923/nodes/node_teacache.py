"""
© 2026 blacksnowskill (BSS). All rights reserved.
Developed by: blacksnowskill (BSS)

nodes/node_teacache.py
AnimaTeaCache — Adaptive Timestep-aware Cache for Anima DiT (BSS).

Caches transformer block outputs between denoising steps when the
timestep embedding changes very little. Skips expensive recomputation
when it's not needed.

Key innovation: ADAPTIVE threshold — varies by position in denoising:
  • Early steps (structure forming) → low threshold (rarely skip)
  • Late steps  (details stable)    → high threshold (skip aggressively)
"""

import logging
from ..core.teacache_engine import patch_model_with_teacache

logger = logging.getLogger("ANIMA_BOOSTER.teacache_node")


class AnimaTeaCache:
    """
    Adaptive TeaCache for Anima DiT models.

    Speeds up inference by skipping redundant transformer computations
    between similar denoising steps. Typically provides 1.5–2.0× speedup.

    The ADAPTIVE mode is the key innovation:
      - Uses a higher threshold (skips more) for late denoising steps
        where image details are already stable
      - Uses a lower threshold (computes more) for early steps
        where global structure is still being established

    Place this node AFTER AnimaBoosterLoader (or any model loader)
    and BEFORE the KSampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "teacache_version": (
                    ["v1 (Legacy Fast)", "v2 (Standard Precise)"],
                    {
                        "default": "v1 (Legacy Fast)",
                    },
                ),
                "adaptive_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
                "early_steps_factor": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                    },
                ),
                "late_steps_factor": (
                    "FLOAT",
                    {
                        "default": 1.8,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
            },
            "optional": {
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "cache_device": (
                    ["cuda", "cpu"],
                    {
                        "default": "cuda",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "BSS/AnimaBooster"
    DESCRIPTION = (
        "Адаптивный TeaCache: значительно ускоряет генерацию Anima DiT, пропуская избыточные вычисления "
        "между близкими шагами сэмплинга. Адаптивный режим гибко настраивает качество под фазы генерации."
    )

    def apply(
        self,
        model,
        threshold: float,
        teacache_version: str,
        adaptive_mode: bool,
        early_steps_factor: float,
        late_steps_factor: float,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        cache_device: str = "cuda",
    ):
        patched = patch_model_with_teacache(
            model=model,
            threshold=threshold,
            version=teacache_version,
            adaptive=adaptive_mode,
            early_factor=early_steps_factor,
            late_factor=late_steps_factor,
            start_percent=start_percent,
            end_percent=end_percent,
            cache_device=cache_device,
        )
        return (patched,)
