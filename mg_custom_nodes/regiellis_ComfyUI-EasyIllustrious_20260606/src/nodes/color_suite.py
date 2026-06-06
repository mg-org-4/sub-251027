"""
Unified Illustrious Color Suite node

Combines Auto Color Corrector and Smart Cache Corrector into a single node.
Keeps the effective behavior while simplifying the node surface area.

Inputs cover both simple correction and cache-aware options. Output returns:
- corrected_images (IMAGE)
- report (STRING)
- cache_used (BOOLEAN)
"""

from .smart_cache_corrector import IllustriousSmartCacheCorrector


class IllustriousColorSuite(IllustriousSmartCacheCorrector):
    """Single, unified color correction node for Illustrious"""

    CATEGORY = "Easy Illustrious / Image Enhancement"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN")
    RETURN_NAMES = ("corrected_images", "report", "cache_used")

    @classmethod
    def INPUT_TYPES(cls):
        base = IllustriousSmartCacheCorrector.INPUT_TYPES()
        # Rename category for clarity in the palette and provide helpful tooltips
        # Ensure all relevant widgets exist and are documented
        req = base.get("required", {})
        opt = base.get("optional", {})

        # Ensure images is required and clearly documented
        req = {
            "images": ("IMAGE", {"tooltip": "Input images to correct (NHWC, 0..1)."}),
        }

        # Merge simple, auto, and smart cache controls
        # Note: We rely on parent implementation to interpret these correctly
        opt.update(
            {
                # Core auto correct controls
                "model_version": (
                    ["auto", "v0.5", "v0.75", "v1.0", "v1.1", "v2.0", "v3.x"],
                    {"default": "auto", "tooltip": "Illustrious model version for tailored fixes."},
                ),
                "correction_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.3,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Overall strength multiplier for corrections.",
                    },
                ),
                "auto_detect_issues": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Analyze the first image to auto-tune corrections."},
                ),
                "fix_oversaturation": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Reduce excessive saturation safely."},
                ),
                "preserve_anime_aesthetic": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Favor anime-friendly look while correcting."},
                ),
                "enhance_details": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Subtle detail enhancement where safe."},
                ),
                "balance_colors": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Neutralize color casts and balance channels."},
                ),
                "adjust_contrast": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Tweak brightness/contrast gently."},
                ),
                "custom_preset": (
                    [
                        "none",
                        "character_portrait",
                        "detailed_scene",
                        "soft_illustration",
                        "vibrant_anime",
                        "natural_colors",
                    ],
                    {"default": "none", "tooltip": "Preset recipe (may override some toggles)."},
                ),
                "show_corrections": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Include a bullet list of applied fixes in the report."},
                ),
                # Smart cache controls
                "cache_mode": (
                    ["auto", "always_cache", "never_cache"],
                    {
                        "default": "auto",
                        "tooltip": "Use cache when safe (auto), or force cache usage/avoidance.",
                    },
                ),
                "adjustment_mode": (
                    ["hybrid", "cached_only", "force_recalculate"],
                    {
                        "default": "hybrid",
                        "tooltip": "Hybrid tries fast cached tweaks; cached_only avoids full recompute.",
                    },
                ),
                "force_recalculate": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Ignore cache and recompute from originals."},
                ),
            }
        )

        return {"required": req, "optional": opt}

    # Keep logic in parent; provide a user-friendly FUNCTION name
    def run(self, images, **kwargs):
        # Filter kwargs to only those accepted by IllustriousSmartCacheCorrector.auto_correct_colors
        allowed = {
            "model_version",
            "correction_strength",
            "auto_detect_issues",
            "preserve_anime_aesthetic",
            "fix_oversaturation",
            "enhance_details",
            "balance_colors",
            "adjust_contrast",
            "custom_preset",
            "show_corrections",
            "cache_mode",
            "adjustment_mode",
            "force_recalculate",
            "enable_preview",
            "auto_analyze",
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return super().auto_correct_colors(images, **filtered)
