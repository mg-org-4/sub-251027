import torch
import numpy as np
import cv2
from PIL import Image, ImageEnhance


class IllustriousColorCorrector:
    """Simple, effective color correction for Illustrious AI models"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to correct (NHWC, 0..1)."}),
            },
            "optional": {
                "mode": (
                    ["auto", "gentle", "balanced", "strong"],
                    {"default": "balanced", "tooltip": "Preset strength and behavior for corrections."},
                ),
                "saturation_fix": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05, "tooltip": "Vibrance-like saturation change (protects strong colors)."},
                ),
                "brightness_adjust": (
                    "FLOAT",
                    {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05, "tooltip": "Brightness adjustment (small recommended)."},
                ),
                "contrast_boost": (
                    "FLOAT",
                    {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05, "tooltip": "Contrast adjustment (small recommended)."},
                ),
                "preserve_anime": ("BOOLEAN", {"default": True, "tooltip": "Protect lineart and edges during correction."}),
                "overall_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Blend between original (0) and corrected (1)",
                    },
                ),
                # accepted but ignored by base node (used by preview subclass)
                "enable_preview": ("BOOLEAN", {"default": False, "tooltip": "Enable live preview (if supported by UI)."}),
                "auto_analyze": ("BOOLEAN", {"default": False, "tooltip": "Analyze first image and suggest tweaks (server mode)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("corrected_images",)
    FUNCTION = "correct_colors"
    CATEGORY = "Easy Illustrious / Image Enhancement"
    OUTPUT_NODE = False

    def correct_colors(
        self,
        images,
        mode="balanced",
        saturation_fix=0.0,
        brightness_adjust=0.0,
        contrast_boost=0.0,
        preserve_anime=True,
        overall_strength=1.0,
        **_,  # accept and ignore extras like enable_preview, auto_analyze
    ):
        device = images.device if isinstance(images, torch.Tensor) else "cpu"

        # Normalize input to torch tensor [B,H,W,C], float32 0..1
        if isinstance(images, torch.Tensor):
            batch = images
        else:
            # Fallback if someone passes a list of tensors
            batch = torch.stack(list(images))

        if batch.dtype != torch.float32:
            batch = batch.float()
        # Some custom nodes pass 0..255; normalize safely
        if batch.max() > 1.5:
            batch = batch / 255.0

        batch = batch.clamp(0.0, 1.0).to("cpu")

        corrected_images = []

        for i in range(batch.shape[0]):
            img_np = (batch[i].numpy() * 255.0).astype(np.uint8)  # HWC uint8 RGB
            pil_image = Image.fromarray(img_np, mode="RGB")

            corrected_pil = self.apply_simple_correction(
                pil_image,
                mode=mode,
                saturation_fix=float(saturation_fix),
                brightness_adjust=float(brightness_adjust),
                contrast_boost=float(contrast_boost),
                preserve_anime=bool(preserve_anime),
            )

            # Global mix to prevent over-processing washout
            s = float(np.clip(overall_strength, 0.0, 1.0))
            if s < 1.0:
                orig_np = np.asarray(pil_image, dtype=np.float32)
                corr_np = np.asarray(corrected_pil, dtype=np.float32)
                mix_np = np.clip(orig_np * (1.0 - s) + corr_np * s, 0, 255).astype(
                    np.uint8
                )
                corrected_pil = Image.fromarray(mix_np)

            corrected_np = (
                np.asarray(corrected_pil, dtype=np.uint8).astype(np.float32) / 255.0
            )
            corrected_images.append(torch.from_numpy(corrected_np))

        corrected_batch = torch.stack(corrected_images).to(device).clamp(0.0, 1.0)
        return (corrected_batch,)

    def apply_simple_correction(
        self,
        image,
        mode,
        saturation_fix,
        brightness_adjust,
        contrast_boost,
        preserve_anime,
    ):
        """Apply simple, effective color corrections without over-processing"""
        corrected = image.copy()

        # Mode-based defaults
        cast_strength = 0.25  # LAB color-cast neutralization (balanced)
        if mode == "auto":
            corrected = self.auto_fix_common_issues(corrected)
            cast_strength = 0.4
        elif mode == "gentle":
            saturation_fix = np.clip(saturation_fix * 0.5, -0.15, 0.15)
            brightness_adjust = np.clip(brightness_adjust * 0.5, -0.1, 0.1)
            contrast_boost = np.clip(contrast_boost * 0.5, -0.1, 0.1)
            cast_strength = 0.15
        elif mode == "strong":
            saturation_fix = np.clip(saturation_fix * 1.5, -0.4, 0.35)
            brightness_adjust = np.clip(brightness_adjust * 1.5, -0.3, 0.3)
            contrast_boost = np.clip(contrast_boost * 1.5, -0.3, 0.3)
            cast_strength = 0.6

        # Keep baseline stats for safety checks
        base_np = np.array(image, dtype=np.uint8)
        base_v_mean = self._v_mean(base_np)

        # 1) Neutralize color cast in LAB (keeps luminance, fixes tint)
        corrected = self.fix_color_cast_lab(corrected, strength=cast_strength)

        # 2) Brightness / Contrast (ImageEnhance operates in sRGB; gentle factors)
        if abs(brightness_adjust) > 0.005:
            corrected = ImageEnhance.Brightness(corrected).enhance(
                1.0 + float(brightness_adjust)
            )
        if abs(contrast_boost) > 0.005:
            corrected = ImageEnhance.Contrast(corrected).enhance(
                1.0 + float(contrast_boost)
            )

        # 3) Vibrance instead of raw saturation (protects already-saturated areas)
        if abs(saturation_fix) > 0.005:
            corrected = self.adjust_vibrance(corrected, amount=float(saturation_fix))

        # 4) Preserve anime aesthetic if requested (edge-aware blend)
        if preserve_anime:
            corrected = self.preserve_anime_characteristics(image, corrected)

        # 5) Safety: prevent blowouts (unexpected large brightness rise)
        corr_np = np.array(corrected, dtype=np.uint8)
        corr_v_mean = self._v_mean(corr_np)
        if corr_v_mean > base_v_mean * 1.15 and float(brightness_adjust) <= 0.0:
            corrected = self._reduce_brightness_to_match(
                corrected, target_ratio=base_v_mean * 1.08 / max(corr_v_mean, 1e-6)
            )

        return corrected

    def auto_fix_common_issues(self, image):
        """Automatically detect and fix common issues without over-processing"""
        img_array = np.array(image, dtype=np.uint8)

        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        avg_saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
        avg_brightness = float(np.mean(hsv[:, :, 2]) / 255.0)

        # Gentle desaturation for oversaturated images
        if avg_saturation > 0.75:
            factor = max(0.85, 1.0 - (avg_saturation - 0.75) * 0.5)
            image = ImageEnhance.Color(image).enhance(factor)

        # Brightness adjustments
        if avg_brightness < 0.33:
            factor = min(1.12, 1.0 + (0.33 - avg_brightness) * 0.45)
            image = ImageEnhance.Brightness(image).enhance(factor)
        elif avg_brightness > 0.88:
            factor = max(0.9, 1.0 - (avg_brightness - 0.88) * 0.6)
            image = ImageEnhance.Brightness(image).enhance(factor)

        # Mild color cast neutralization
        image = self.fix_color_cast_lab(image, strength=0.25)
        return image

    def fix_color_cast_lab(self, image, strength=0.5):
        """
        Neutralize color cast by shifting mean a/b toward neutral in LAB.
        strength: 0..1 (how much of the detected cast to remove)
        """
        img = np.array(image, dtype=np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)

        # OpenCV LAB a/b are centered at 128 (neutral)
        mean_a = float(np.mean(A))
        mean_b = float(np.mean(B))
        da = (mean_a - 128.0) * float(strength)
        db = (mean_b - 128.0) * float(strength)

        A_corr = np.clip(A - da, 0, 255).astype(np.uint8)
        B_corr = np.clip(B - db, 0, 255).astype(np.uint8)
        lab_corr = cv2.merge([L, A_corr, B_corr])
        rgb_corr = cv2.cvtColor(lab_corr, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_corr)

    def adjust_vibrance(self, image, amount=0.0):
        """
        Vibrance-like saturation: affects low-sat pixels more than high-sat.
        amount: negative to reduce, positive to boost.
        """
        img = np.array(image, dtype=np.uint8)
        # Work in float32 to compute a scale that affects low saturation more
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        H, S, V = cv2.split(hsv)  # all float32
        Sn = (S / 255.0).astype(np.float32)

        eff = np.clip(float(amount), -1.0, 1.0) * 0.6
        scale = (1.0 + eff * (1.0 - Sn)).astype(np.float32)  # ensure float32

        # Scale S channel and clamp to 0..255
        S2 = np.clip(S * scale, 0, 255).astype(np.float32)

        # Merge back, then convert to uint8 HSV before converting to RGB
        hsv2 = cv2.merge([H.astype(np.float32), S2, V.astype(np.float32)]).astype(
            np.uint8
        )
        rgb = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb)

    def preserve_anime_characteristics(self, original, corrected):
        """Edge-aware blending to protect lineart and details while applying color changes"""
        orig = np.asarray(original, dtype=np.uint8)
        corr = np.asarray(corrected, dtype=np.uint8)

        # Edge mask from luminance
        gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        mask_edges = edges.astype(np.float32) / 255.0
        mask_edges = cv2.GaussianBlur(mask_edges, (0, 0), 1.0)

        # Invert and soften: stronger correction in flat areas, weaker on edges
        flat_mask = 1.0 - mask_edges
        flat_mask = np.clip(flat_mask, 0.0, 1.0)
        flat_mask = cv2.GaussianBlur(flat_mask, (0, 0), 0.8)
        flat_mask = flat_mask[..., None]  # broadcast to RGB

        # Compute a convex blending weight w in [0.3, 0.8]
        w = 0.3 + 0.5 * flat_mask
        out = orig.astype(np.float32) * (1.0 - w) + corr.astype(np.float32) * w
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    # -------- Helpers --------
    def _v_mean(self, rgb_uint8):
        hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
        return float(np.mean(hsv[:, :, 2])) + 1e-6

    def _reduce_brightness_to_match(self, image, target_ratio=0.95):
        """Reduce brightness by scaling V channel to avoid blowouts."""
        rgb = np.array(image, dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        H, S, V = cv2.split(hsv)

        V2 = np.clip(V * float(target_ratio), 0, 255).astype(np.float32)
        hsv2 = cv2.merge([H.astype(np.float32), S.astype(np.float32), V2]).astype(
            np.uint8
        )
        out = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)
        return Image.fromarray(out)

    # -------- Server compatibility: basic analysis + auto API --------
    def detect_model_version(self, image_pil):
        """Heuristic Illustrious model version guess based on image traits.

        Not exact, but helps choose sensible defaults when users leave version to auto.
        """
        try:
            img = np.asarray(image_pil.convert("RGB"), dtype=np.uint8)
            h, w = img.shape[:2]
            longer = max(h, w)

            # Resolution bands (typical training crops):
            if longer >= 2048:
                return "v2.0"
            if longer >= 1536:
                return "v1.1"
            if longer >= 1024:
                return "v1.0"

            # Fall back to color-stat based tiering
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            avg_s = float(np.mean(hsv[:, :, 1])) / 255.0
            if avg_s > 0.55:
                return "v1.0"
            return "v0.75"
        except Exception:
            return "v1.0"

    def analyze_image_characteristics(self, image_pil, version="auto"):
        img = np.asarray(image_pil, dtype=np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(hsv)
        Sn = S.astype(np.float32) / 255.0
        Vn = V.astype(np.float32) / 255.0

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        _, A, B = cv2.split(lab)
        mean_a = float(np.mean(A) - 128.0)
        mean_b = float(np.mean(B) - 128.0)

        Ln = lab[:, :, 0].astype(np.float32) / 255.0
        contrast = float(np.std(Ln))

        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 100, 200)
        edge_density = float(np.mean(edges > 0))

        return {
            "avg_saturation": float(np.mean(Sn)),
            "max_saturation": float(np.max(Sn)),
            "avg_brightness": float(np.mean(Vn)),
            "contrast": contrast,
            "color_cast": {
                "red_green_axis": mean_a / 128.0,
                "blue_yellow_axis": mean_b / 128.0,
            },
            "highlight_clipping": float(np.mean(V >= 250)),
            "shadow_clipping": float(np.mean(V <= 5)),
            "illustration_style": (
                "anime" if contrast < 0.22 and np.mean(Sn) > 0.45 else "mixed"
            ),
            "is_character_focused": edge_density < 0.12,
            "has_detailed_background": edge_density > 0.18,
            "anime_score": float(
                np.clip(0.5 * (np.mean(Sn) + (0.25 - contrast) * 2.0), 0.0, 1.0)
            ),
            "oversaturation_score": float(np.mean(Sn > 0.8)),
            "color_bias": {
                "overall_bias_strength": float(
                    min(1.0, (abs(mean_a) + abs(mean_b)) / 80.0)
                ),
                "a_bias": mean_a,
                "b_bias": mean_b,
            },
        }

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
        **_,
    ):
        # Pick mode from strength
        mode = (
            "gentle"
            if correction_strength <= 0.7
            else ("balanced" if correction_strength < 1.2 else "strong")
        )

        # Defaults
        sat = 0.0
        bri = 0.0
        con = 0.0

        # Auto tune based on the first image if requested
        if (
            auto_detect_issues
            and isinstance(images, torch.Tensor)
            and images.shape[0] > 0
        ):
            np0 = images[0].detach().cpu().numpy()
            if np0.max() <= 1.0:
                np0 = (np0 * 255).astype(np.uint8)
            else:
                np0 = np0.astype(np.uint8)
            pil0 = Image.fromarray(np0[:, :, :3])
            a = self.analyze_image_characteristics(pil0, model_version)

            if fix_oversaturation and (
                a["avg_saturation"] > 0.65 or a.get("oversaturation_score", 0) > 0.15
            ):
                sat -= 0.15 * correction_strength
            elif a["avg_saturation"] < 0.35:
                sat += 0.12 * correction_strength

            if a["avg_brightness"] < 0.3:
                bri += 0.08 * correction_strength
            elif a["avg_brightness"] > 0.85:
                bri -= 0.08 * correction_strength

            if adjust_contrast:
                if a["contrast"] < 0.18:
                    con += 0.12 * correction_strength
                elif a["contrast"] > 0.38:
                    con -= 0.08 * correction_strength

        (batch,) = self.correct_colors(
            images,
            mode=mode,
            saturation_fix=sat,
            brightness_adjust=bri,
            contrast_boost=con,
            preserve_anime=preserve_anime_aesthetic,
        )

        report = (
            f"Illustrious Auto Color Corrector\n"
            f"- Version: {model_version}\n"
            f"- Mode: {mode}\n"
            f"Corrections Applied:\n"
            f"• Vibrance: {sat:+.2f}\n"
            f"• Brightness: {bri:+.2f}\n"
            f"• Contrast: {con:+.2f}\n"
            f"• Preserve Anime: {preserve_anime_aesthetic}\n"
            f"• LAB Color Cast Neutralization: on\n"
        )
        return (batch, report)
