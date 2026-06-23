"""
Illustrious Smart Cache Color Corrector
Advanced caching system with real-time adjustments without workflow regeneration
"""

import torch
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2
from .color_corrector import IllustriousColorCorrector




class IllustriousSmartCacheCorrector(IllustriousColorCorrector):
    """Enhanced Illustrious corrector with intelligent caching for real-time adjustments (Color Suite version)"""

    def __init__(self):
        super().__init__()
        # Cache system
        self.cache = {
            "original_images": None,
            "analysis_data": None,
            "last_settings": None,
            "corrected_result": None,
            "timestamp": None,
            "cache_valid": False,
        }

    @classmethod
    def INPUT_TYPES(cls):
        # Define base required/optional inputs
        base_inputs = {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to correct (NHWC, 0..1)."}),
            },
            "optional": {
                "cache_mode": (
                    ["auto", "always_cache", "never_cache"],
                    {"default": "auto", "tooltip": "Auto uses cache when safe; always/never override behavior."},
                ),
                "adjustment_mode": (
                    ["hybrid", "cached_only", "force_recalculate"],
                    {"default": "hybrid", "tooltip": "Hybrid tries fast cached tweaks; force recalculates fully."},
                ),
                "force_recalculate": ("BOOLEAN", {"default": False, "tooltip": "Ignore cache and recompute from originals."}),
                "enable_preview": ("BOOLEAN", {"default": True, "tooltip": "Enable live preview (if supported by UI)."}),
                "auto_analyze": ("BOOLEAN", {"default": True, "tooltip": "Analyze image to auto-tune corrections."}),
            }
        }
        return base_inputs

    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN")
    RETURN_NAMES = ("corrected_images", "correction_report", "cache_used")

    CATEGORY = "Easy Illustrious / Smart Cache"

    def auto_correct_colors(
        self,
        images,
        model_version="auto",
        correction_strength=1.0,
        auto_detect_issues=True,
        preserve_anime_aesthetic=True,
        fix_oversaturation=True,
        enhance_details=True,
        balance_colors=True,
        adjust_contrast=True,
        custom_preset="none",
        show_corrections=False,
        cache_mode="auto",
        adjustment_mode="hybrid",
        force_recalculate=False,
        enable_preview=True,
        auto_analyze=True,
    ):

        current_settings = {
            "model_version": model_version,
            "correction_strength": correction_strength,
            "auto_detect_issues": auto_detect_issues,
            "preserve_anime_aesthetic": preserve_anime_aesthetic,
            "fix_oversaturation": fix_oversaturation,
            "enhance_details": enhance_details,
            "balance_colors": balance_colors,
            "adjust_contrast": adjust_contrast,
            "custom_preset": custom_preset,
            "show_corrections": show_corrections,
        }

        # Check if we can use cached results
        cache_used = False
        should_use_cache = self._should_use_cache(
            images, current_settings, cache_mode, adjustment_mode, force_recalculate
        )

        if should_use_cache and adjustment_mode == "cached_only":
            # Apply adjustments to cached images without full recalculation
            corrected_images, report = self._apply_cached_adjustments(current_settings)
            cache_used = True

            print(
                f"🧠 [Illustrious Smart Cache] Used cached adjustments (mode: {adjustment_mode})"
            )

        elif should_use_cache and adjustment_mode == "hybrid":
            # Use hybrid approach: fast adjustments when possible, full recalc when needed
            if self._can_apply_fast_adjustment(current_settings):
                corrected_images, report = self._apply_cached_adjustments(
                    current_settings
                )
                cache_used = True
                print(f"🧠 [Illustrious Smart Cache] Applied fast hybrid adjustment")
            else:
                # Need full recalculation
                corrected_images, report = self._perform_full_correction(
                    images, current_settings
                )
                cache_used = False
                print(
                    f"🧠 [Illustrious Smart Cache] Full recalculation required for hybrid mode"
                )

        else:
            # Force full recalculation or cache not available
            corrected_images, report = self._perform_full_correction(
                images, current_settings
            )
            cache_used = False
            print(
                f"🧠 [Illustrious Smart Cache] Full correction applied (cache_mode: {cache_mode})"
            )

        # Update cache for future use
        if cache_mode != "never_cache":
            self._update_cache(images, current_settings, corrected_images, report)

        return corrected_images, report, cache_used

    def _should_use_cache(
        self, images, settings, cache_mode, adjustment_mode, force_recalculate
    ):
        """Determine if cache should be used"""
        if force_recalculate:
            return False

        if cache_mode == "never_cache":
            return False

        if not self.cache["cache_valid"]:
            return False

        if adjustment_mode == "force_recalculate":
            return False

        # Check if input images match cached images
        if not self._images_match_cache(images):
            return False

        # For auto mode, decide based on settings similarity
        if cache_mode == "auto":
            return self._settings_allow_cache_use(settings)

        # always_cache mode
        return True

    def _images_match_cache(self, images):
        """Check if current images match cached images"""
        if self.cache["original_images"] is None:
            return False

        if len(images) != len(self.cache["original_images"]):
            return False

        # Quick tensor comparison
        for img, cached_img in zip(images, self.cache["original_images"]):
            if not torch.equal(img, cached_img):
                return False

        return True

    def _settings_allow_cache_use(self, settings):
        """Check if settings changes allow cache use"""
        if self.cache["last_settings"] is None:
            return False

        cached_settings = self.cache["last_settings"]

        # Settings that require full recalculation
        critical_settings = ["model_version", "auto_detect_issues", "custom_preset"]

        for setting in critical_settings:
            if settings.get(setting) != cached_settings.get(setting):
                return False

        return True

    def _can_apply_fast_adjustment(self, settings):
        """Check if we can apply fast adjustments to cached results"""
        if not self.cache["cache_valid"]:
            return False

        # Settings that can be adjusted post-processing
        fast_adjustable = [
            "correction_strength",
            "fix_oversaturation",
            "enhance_details",
            "balance_colors",
            "adjust_contrast",
        ]

        cached_settings = self.cache["last_settings"]

        # Check if only fast-adjustable settings changed
        for key, value in settings.items():
            if key in fast_adjustable:
                continue  # These can be adjusted quickly
            elif cached_settings.get(key) != value:
                return False  # This setting requires full recalculation

        return True

    def _apply_cached_adjustments(self, settings):
        """Apply adjustments to cached images"""
        if not self.cache["cache_valid"] or self.cache["corrected_result"] is None:
            raise ValueError("No valid cache available for adjustment")

        cached_images = self.cache["corrected_result"]
        cached_settings = self.cache["last_settings"]

        # Apply differential adjustments
        adjusted_images = self._apply_differential_corrections(
            cached_images, cached_settings, settings
        )

        # Generate adjustment report
        report = self._generate_adjustment_report(cached_settings, settings)

        return adjusted_images, report

    def _apply_differential_corrections(
        self, cached_images, old_settings, new_settings
    ):
        """Apply differential corrections to cached images based on setting changes"""
        result_images = []

        for img_tensor in cached_images:
            img = img_tensor.clone()

            # Apply strength adjustment
            strength_diff = new_settings.get(
                "correction_strength", 1.0
            ) - old_settings.get("correction_strength", 1.0)
            if abs(strength_diff) > 0.01:  # Threshold for meaningful change
                img = self._adjust_correction_strength(img, strength_diff)

            # Apply oversaturation fix adjustment
            if new_settings.get("fix_oversaturation") != old_settings.get(
                "fix_oversaturation"
            ):
                img = self._adjust_oversaturation_fix(
                    img, new_settings.get("fix_oversaturation", True)
                )

            # Apply detail enhancement adjustment
            if new_settings.get("enhance_details") != old_settings.get(
                "enhance_details"
            ):
                img = self._adjust_detail_enhancement(
                    img, new_settings.get("enhance_details", True)
                )

            # Apply color balance adjustment
            if new_settings.get("balance_colors") != old_settings.get("balance_colors"):
                img = self._adjust_color_balance(
                    img, new_settings.get("balance_colors", True)
                )

            # Apply contrast adjustment
            if new_settings.get("adjust_contrast") != old_settings.get(
                "adjust_contrast"
            ):
                img = self._adjust_contrast_correction(
                    img, new_settings.get("adjust_contrast", True)
                )

            result_images.append(img)

        return result_images

    def _adjust_correction_strength(self, img, strength_diff):
        """Adjust overall correction strength"""
        # Simple strength adjustment by blending with original
        if strength_diff > 0:
            # Increase correction (move further from neutral)
            neutral = torch.ones_like(img) * 0.5
            img = img + (img - neutral) * strength_diff * 0.5
        elif strength_diff < 0:
            # Decrease correction (move towards neutral)
            neutral = torch.ones_like(img) * 0.5
            img = img + (neutral - img) * abs(strength_diff) * 0.3

        return torch.clamp(img, 0, 1)

    def _adjust_oversaturation_fix(self, img, enable_fix):
        """Toggle oversaturation fix"""
        if enable_fix:
            # Apply desaturation to highly saturated areas
            img_hsv = self._rgb_to_hsv(img)
            saturation = img_hsv[:, :, :, 1]
            mask = saturation > 0.8
            img_hsv[:, :, :, 1] = torch.where(mask, saturation * 0.85, saturation)
            img = self._hsv_to_rgb(img_hsv)
        else:
            # Restore some saturation
            img_hsv = self._rgb_to_hsv(img)
            img_hsv[:, :, :, 1] = torch.clamp(img_hsv[:, :, :, 1] * 1.05, 0, 1)
            img = self._hsv_to_rgb(img_hsv)

        return torch.clamp(img, 0, 1)

    def _adjust_detail_enhancement(self, img, enhance_details):
        """Toggle detail enhancement"""
        if enhance_details:
            # Apply subtle sharpening
            kernel = torch.tensor(
                [[[[-0.1, -0.1, -0.1], [-0.1, 1.8, -0.1], [-0.1, -0.1, -0.1]]]],
                device=img.device,
                dtype=img.dtype,
            )
            kernel = kernel.repeat(3, 1, 1, 1)

            # Apply convolution for each channel
            enhanced = torch.nn.functional.conv2d(
                img.permute(0, 3, 1, 2), kernel, padding=1, groups=3
            ).permute(0, 2, 3, 1)

            # Blend with original
            img = 0.7 * img + 0.3 * enhanced
        else:
            # Apply subtle blur to remove enhancement
            kernel = torch.ones(1, 1, 3, 3, device=img.device, dtype=img.dtype) / 9
            kernel = kernel.repeat(3, 1, 1, 1)

            blurred = torch.nn.functional.conv2d(
                img.permute(0, 3, 1, 2), kernel, padding=1, groups=3
            ).permute(0, 2, 3, 1)

            img = 0.9 * img + 0.1 * blurred

        return torch.clamp(img, 0, 1)

    def _adjust_color_balance(self, img, balance_colors):
        """Toggle color balance correction"""
        if balance_colors:
            # Apply color balance correction
            img_lab = self._rgb_to_lab(img)
            # Center the a and b channels
            img_lab[:, :, :, 1] = (
                img_lab[:, :, :, 1] - img_lab[:, :, :, 1].mean()
            ) * 0.95 + img_lab[:, :, :, 1].mean()
            img_lab[:, :, :, 2] = (
                img_lab[:, :, :, 2] - img_lab[:, :, :, 2].mean()
            ) * 0.95 + img_lab[:, :, :, 2].mean()
            img = self._lab_to_rgb(img_lab)

        return torch.clamp(img, 0, 1)

    def _adjust_contrast_correction(self, img, adjust_contrast):
        """Toggle contrast adjustment"""
        if adjust_contrast:
            # Increase contrast slightly
            img = torch.clamp((img - 0.5) * 1.1 + 0.5, 0, 1)
        else:
            # Decrease contrast slightly
            img = torch.clamp((img - 0.5) * 0.95 + 0.5, 0, 1)

        return img

    def _perform_full_correction(self, images, settings):
        """Perform full color correction using parent method"""
        # Call parent method with all settings
        return super().auto_correct_colors(
            images,
            model_version=settings.get("model_version", "auto"),
            correction_strength=settings.get("correction_strength", 1.0),
            auto_detect_issues=settings.get("auto_detect_issues", True),
            preserve_anime_aesthetic=settings.get("preserve_anime_aesthetic", True),
            fix_oversaturation=settings.get("fix_oversaturation", True),
            enhance_details=settings.get("enhance_details", True),
            balance_colors=settings.get("balance_colors", True),
            adjust_contrast=settings.get("adjust_contrast", True),
            custom_preset=settings.get("custom_preset", "none"),
            show_corrections=settings.get("show_corrections", False),
        )

    def _update_cache(self, images, settings, corrected_images, report):
        """Update internal cache"""
        self.cache.update(
            {
                "original_images": [img.clone() for img in images],
                "last_settings": settings.copy(),
                "corrected_result": [img.clone() for img in corrected_images],
                "timestamp": time.time(),
                "cache_valid": True,
            }
        )

        print(f"🧠 [Illustrious Smart Cache] Cache updated with {len(images)} images")

    def _generate_adjustment_report(self, old_settings, new_settings):
        """Generate report for cached adjustments"""
        changes = []

        for key, new_value in new_settings.items():
            old_value = old_settings.get(key)
            if old_value != new_value:
                changes.append(f"• {key}: {old_value} → {new_value}")

        if changes:
            report = f"🧠 Illustrious Smart Cache Adjustments Applied:\n\n" + "\n".join(
                changes
            )
            report += f"\n\nAdjustment Speed: ⚡ Fast (cached)"
            report += f"\nCache Age: {time.time() - self.cache['timestamp']:.1f}s"
        else:
            report = "🧠 Illustrious Smart Cache: No changes detected"

        return report

    def _apply_corrections_to_cached(self, original_images, analysis, settings):
        """Apply corrections to cached images (used by server)"""
        try:
            # This method is called by the server for real-time preview
            current_cache_settings = self.cache.get("last_settings", {})

            if self._can_apply_fast_adjustment(settings):
                # Apply fast adjustments
                cached_result = self.cache.get("corrected_result")
                if cached_result:
                    adjusted = self._apply_differential_corrections(
                        cached_result, current_cache_settings, settings
                    )
                    report = self._generate_adjustment_report(
                        current_cache_settings, settings
                    )
                    return adjusted, report

            # Fall back to parent correction method
            return super().auto_correct_colors(original_images, **settings)

        except Exception as e:
            print(f"[Illustrious Smart Cache] Error in cached correction: {e}")
            # Fallback to parent method
            return super().auto_correct_colors(original_images, **settings)

    def clear_cache(self):
        """Clear the internal cache"""
        self.cache = {
            "original_images": None,
            "analysis_data": None,
            "last_settings": None,
            "corrected_result": None,
            "timestamp": None,
            "cache_valid": False,
        }
        print("🧠 [Illustrious Smart Cache] Cache cleared")

    def get_cache_info(self):
        """Get cache information"""
        if not self.cache["cache_valid"]:
            return "No cache available"

        age = time.time() - (self.cache.get("timestamp", 0))
        image_count = len(self.cache.get("original_images", []))

        return f"Cache: {image_count} images, {age:.1f}s old"

    # Helper methods for color space conversions (production versions)
    def _rgb_to_hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB (0..1) torch NHWC to HSV (0..1 for S,V; H in 0..1 normalized)."""
        was_batched = rgb.dim() == 4
        if not was_batched:
            rgb = rgb.unsqueeze(0)
        arr = (rgb.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
        out = []
        for img in arr:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            # Normalize H (0..179) -> 0..1, S/V (0..255) -> 0..1
            h = (hsv[:, :, 0] / 179.0)
            s = (hsv[:, :, 1] / 255.0)
            v = (hsv[:, :, 2] / 255.0)
            out.append(np.stack([h, s, v], axis=-1))
        hsv_t = torch.from_numpy(np.stack(out, axis=0)).to(dtype=rgb.dtype, device=rgb.device)
        if not was_batched:
            hsv_t = hsv_t.squeeze(0)
        return hsv_t.clamp(0.0, 1.0)

    def _hsv_to_rgb(self, hsv: torch.Tensor) -> torch.Tensor:
        """Convert HSV (0..1) torch NHWC back to RGB (0..1)."""
        was_batched = hsv.dim() == 4
        if not was_batched:
            hsv = hsv.unsqueeze(0)
        arr = hsv.detach().cpu().numpy().astype(np.float32)
        out = []
        for img in arr:
            h = (img[:, :, 0] * 179.0).astype(np.float32)
            s = (img[:, :, 1] * 255.0).astype(np.float32)
            v = (img[:, :, 2] * 255.0).astype(np.float32)
            hsv_cv = np.stack([h, s, v], axis=-1).astype(np.uint8)
            rgb = cv2.cvtColor(hsv_cv, cv2.COLOR_HSV2RGB)
            out.append(np.clip(rgb, 0, 255).astype(np.uint8))
        rgb_arr = (np.stack(out, axis=0).astype(np.float32) / 255.0)
        rgb_t = torch.from_numpy(rgb_arr).to(dtype=hsv.dtype, device=hsv.device)
        if not was_batched:
            rgb_t = rgb_t.squeeze(0)
        return rgb_t.clamp(0.0, 1.0)

    def _rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB (0..1) torch NHWC to LAB in float32 with L in 0..100 range.

        Accepts [B,H,W,C] or [H,W,C]. Returns same shape as input.
        """
        was_batched = rgb.dim() == 4
        if not was_batched:
            rgb = rgb.unsqueeze(0)
        # to Numpy uint8
        arr = (rgb.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
        out_list = []
        for img in arr:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
            # Normalize L to 0..1 if desired? Keep OpenCV's 0..255 for A/B centered 128.
            out_list.append(lab)
        lab_arr = np.stack(out_list, axis=0)
        lab_t = torch.from_numpy(lab_arr).to(dtype=rgb.dtype, device=rgb.device)
        if not was_batched:
            lab_t = lab_t.squeeze(0)
        return lab_t

    def _lab_to_rgb(self, lab: torch.Tensor) -> torch.Tensor:
        """Convert LAB back to RGB (0..1) as torch NHWC float.

        Accepts [B,H,W,C] or [H,W,C]. Returns same shape with values in 0..1.
        """
        was_batched = lab.dim() == 4
        if not was_batched:
            lab = lab.unsqueeze(0)
        arr = lab.detach().cpu().numpy().astype(np.float32)
        out_list = []
        for img in arr:
            rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
            out_list.append(np.clip(rgb, 0, 255).astype(np.uint8))
        rgb_arr = np.stack(out_list, axis=0).astype(np.float32) / 255.0
        rgb_t = torch.from_numpy(rgb_arr).to(dtype=lab.dtype, device=lab.device)
        if not was_batched:
            rgb_t = rgb_t.squeeze(0)
        return rgb_t.clamp(0.0, 1.0)
