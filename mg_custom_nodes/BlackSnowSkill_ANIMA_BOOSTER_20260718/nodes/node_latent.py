"""
© 2026 blacksnowskill (BSS). All rights reserved.
Developed by: blacksnowskill (BSS)

nodes/node_latent.py
AnimaLatentImage — Latent image node for Anima DiT with auto-valid dimensions (BSS).

Anima requires image dimensions divisible by 16:
  - VAE spatial compression: ÷8
  - Anima spatial_patch_size: ÷2
  - Total: must be divisible by 8×2 = 16

Using non-divisible-by-16 dimensions causes:
  AssertionError: H,W should be divisible by spatial_patch_size 2
"""

import logging
import torch

logger = logging.getLogger("ANIMA_BOOSTER.latent")

ANIMA_ALIGN = 16  # 8 (VAE) × 2 (spatial_patch_size)

# Common Anima-compatible resolutions (W×H)
PRESET_RESOLUTIONS = [
    "Custom",
    "512×512 (1:1)",
    "768×512 (3:2 landscape)",
    "512×768 (2:3 portrait)",
    "768×768 (1:1 medium)",
    "1024×576 (16:9 landscape)",
    "576×1024 (9:16 portrait)",
    "1024×768 (4:3 landscape)",
    "768×1024 (3:4 portrait)",
    "1024×1024 (1:1 large)",
    "1280×720 (16:9 HD)",
    "720×1280 (9:16 HD)",
    "1024×1280 (4:5 portrait)",
    "1280×1024 (5:4 landscape)",
]

PRESET_MAP = {
    "512×512 (1:1)":          (512, 512),
    "768×512 (3:2 landscape)": (768, 512),
    "512×768 (2:3 portrait)":  (512, 768),
    "768×768 (1:1 medium)":    (768, 768),
    "1024×576 (16:9 landscape)":(1024, 576),
    "576×1024 (9:16 portrait)": (576, 1024),
    "1024×768 (4:3 landscape)": (1024, 768),
    "768×1024 (3:4 portrait)":  (768, 1024),
    "1024×1024 (1:1 large)":    (1024, 1024),
    "1280×720 (16:9 HD)":       (1280, 720),
    "720×1280 (9:16 HD)":       (720, 1280),
    "1024×1280 (4:5 portrait)": (1024, 1280),
    "1280×1024 (5:4 landscape)":(1280, 1024),
}


def align_to(value: int, align: int = ANIMA_ALIGN) -> int:
    """Round value UP to nearest multiple of align."""
    return ((value + align - 1) // align) * align


class AnimaLatentImage:
    """
    Empty latent image generator for Anima DiT.

    Automatically ensures width and height are divisible by 16
    (required by Anima's VAE × spatial_patch_size constraint).

    If you enter a non-divisible value, it is rounded UP to the
    nearest valid size and a warning is logged.

    Use presets for common aspect ratios, or Custom for manual entry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    PRESET_RESOLUTIONS,
                    {
                        "default": "1024×1024 (1:1 large)",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 16,
                        "display": "number",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 16,
                        "display": "number",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "generate"
    CATEGORY = "BSS/AnimaBooster"
    DESCRIPTION = (
        "Создает пустой латент с автовыравниванием размеров до кратных 16, как требует Anima DiT. "
        "Позволяет выбирать готовые пресеты соотношений сторон или вводить размер вручную."
    )

    def generate(
        self,
        preset: str,
        width: int,
        height: int,
        batch_size: int,
    ):
        # Use preset if not Custom
        if preset != "Custom" and preset in PRESET_MAP:
            width, height = PRESET_MAP[preset]

        # Auto-align
        aligned_w = align_to(width)
        aligned_h = align_to(height)

        if aligned_w != width or aligned_h != height:
            logger.warning(
                f"[AnimaLatentImage] Dimensions {width}×{height} not divisible by {ANIMA_ALIGN}. "
                f"Rounded to {aligned_w}×{aligned_h}."
            )
            width, height = aligned_w, aligned_h

        # Anima VAE compresses by 8×
        latent_w = width // 8
        latent_h = height // 8

        # Anima latent format: (B, C, H, W) with C=16
        latent = torch.zeros(
            [batch_size, 16, latent_h, latent_w],
            dtype=torch.float32,
        )

        logger.info(
            f"[AnimaLatentImage] {width}×{height} → latent {latent_w}×{latent_h} "
            f"| batch={batch_size}"
        )
        return ({"samples": latent}, width, height)
