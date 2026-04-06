import torch
import math
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Optional
import comfy.model_management
import comfy.utils
from .. import RESOLUTIONS


class IllustriousEmptyLatentImage:
    """Custom Empty Latent Image node optimized for Illustrious models"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
        "resolution": (
                    list(RESOLUTIONS.keys()),
                    {
                        "default": "1:1 - (1024x1024)",
            "tooltip": "Select resolution/aspect ratio optimized for Illustrious models",
                    },
                ),
        "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "How many latents to generate."}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Optional: provide model to auto-detect EPS/VPred."}),
                "model_version": (
                    [
                        "auto",
                        "Illustrious v0.1",
                        "Illustrious v1.0",
                        "Illustrious v1.1",
                        "Illustrious v2.0",
                        "Illustrious v3.0 EPS",
                        "Illustrious v3.0 VPred",
                        "Illustrious v3.5 VPred",
                    ],
                    {"default": "auto", "tooltip": "Auto: infer from provided model (EPS/VPred) when connected, else fall back to a resolution-based heuristic."},
                ),
                "optimization_mode": (
                    ["auto", "quality", "speed", "compatibility"],
                    {"default": "auto", "tooltip": "Quality: more detail; Speed: faster; Compatibility: conservative."},
                ),
                "noise_pattern": (
                    [
                        "standard",
                        "illustrious_optimized",
                        "high_frequency",
                        "low_frequency",
                    ],
                    {"default": "illustrious_optimized", "tooltip": "Noise flavor tuned for Illustrious anime/illustration."},
                ),
                "enable_native_resolution": ("BOOLEAN", {"default": True, "tooltip": "Snap sizes to native-friendly dimensions for stability."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Base random seed for noise generation."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "resolution_info")
    FUNCTION = "generate_illustrious_latent"
    CATEGORY = "Easy Illustrious / Latent Image"

    def __init__(self):
        # Illustrious optimal resolution tiers and characteristics
        self.resolution_tiers = {
            "512": {
                "native_sizes": [(512, 512), (512, 768), (768, 512)],
                "max_size": 768,
                "optimal_versions": ["Illustrious v0.1"],
                "characteristics": "Basic compatibility, good for testing and quick generations",
            },
            "768": {
                "native_sizes": [(768, 768), (512, 1024), (1024, 512)],
                "max_size": 1024,
                "optimal_versions": ["Illustrious v0.1"],
                "characteristics": "Good balance, works well for most anime content",
            },
            "1024": {
                "native_sizes": [(1024, 1024), (768, 1280), (1280, 768)],
                "max_size": 1280,
                "optimal_versions": ["Illustrious v0.1", "Illustrious v1.0"],
                "characteristics": "Standard SDXL resolution, reliable anime/illustration results",
            },
            "1536": {
                "native_sizes": [
                    (1536, 1536),
                    (1024, 2048),
                    (2048, 1024),
                    (1248, 1824),
                    (1824, 1248),
                ],
                "max_size": 2048,
                "optimal_versions": ["Illustrious v1.0", "Illustrious v1.1"],
                "characteristics": "Illustrious native resolution, excellent for detailed anime art",
            },
            "2048": {
                "native_sizes": [(2048, 2048), (1536, 2560), (2560, 1536)],
                "max_size": 2560,
                "optimal_versions": ["Illustrious v1.1", "Illustrious v2.0"],
                "characteristics": "Ultra high-res, Illustrious v2.0 optimized, best for masterpiece quality",
            },
        }

        # Aspect ratio categories for optimization
        self.aspect_categories = {
            "square": (0.9, 1.1),
            "portrait": (0.6, 0.9),
            "landscape": (1.1, 1.7),
            "ultra_wide": (1.7, 3.0),
            "ultra_tall": (0.3, 0.6),
        }

    def generate_illustrious_latent(
        self,
        resolution,
        batch_size,
        model=None,
        model_version="auto",
        optimization_mode="auto",
        noise_pattern="illustrious_optimized",
        enable_native_resolution=True,
        seed=0,
    ):
        # Parse resolution from dropdown selection
        resolution_str = RESOLUTIONS[resolution]
        width, height = map(int, resolution_str.split("x"))

        # Determine effective version: prefer explicit, else model introspection, else resolution heuristic
        if model_version != "auto":
            inferred_version = model_version
        else:
            inferred_version = self.detect_model_version_from_model(model) if model is not None else "auto"
            if inferred_version == "auto":
                inferred_version = self.detect_optimal_version(width, height)

        resolution_tier = self.detect_resolution_tier(width, height)
        aspect_ratio_mode = self.detect_aspect_ratio_mode(width, height)

        # Optimize dimensions for Illustrious if enabled
        if enable_native_resolution:
            width, height, size_adjusted = self.optimize_for_native_resolution(
                width, height, resolution_tier, inferred_version
            )
        else:
            size_adjusted = False

        # Calculate latent dimensions
        latent_width = width // 8
        latent_height = height // 8

        # Generate optimized noise pattern
        latent_image = self.create_optimized_noise(
            batch_size,
            latent_height,
            latent_width,
            noise_pattern,
            aspect_ratio_mode,
            inferred_version,
            seed,
        )

        # Apply version-specific optimizations
        if optimization_mode != "compatibility":
            latent_image = self.apply_version_optimizations(
                latent_image, inferred_version, optimization_mode, aspect_ratio_mode
            )

        # Generate information report
        resolution_info = self.generate_resolution_report(
            width,
            height,
            latent_width,
            latent_height,
            inferred_version,
            resolution_tier,
            aspect_ratio_mode,
            size_adjusted,
            noise_pattern,
        )

        print(
            f"🎨 Generated Illustrious latent: {width}x{height} ({resolution_tier}) for {inferred_version}"
        )

        return ({"samples": latent_image}, resolution_info)

    def _detect_pred_mode(self, model):
        """Return 'vpred', 'eps', or None by inspecting the model object."""
        try:
            candidates = [model]
            m = getattr(model, "model", None)
            if m is not None:
                candidates.append(m)
                dm = getattr(m, "diffusion_model", None)
                if dm is not None:
                    candidates.append(dm)
            for obj in candidates:
                if obj is None:
                    continue
                for attr in ("parameterization", "prediction_type"):
                    if hasattr(obj, attr):
                        val = getattr(obj, attr)
                        if isinstance(val, str):
                            low = val.lower()
                            if low.startswith("v") or "vpred" in low:
                                return "vpred"
                            if low.startswith("eps") or "epsilon" in low:
                                return "eps"
                config = getattr(obj, "model_config", None) or getattr(obj, "config", None)
                if isinstance(config, dict):
                    val = (
                        config.get("parameterization")
                        or config.get("prediction_type")
                        or config.get("param")
                    )
                    if isinstance(val, str):
                        low = val.lower()
                        if low.startswith("v") or "vpred" in low:
                            return "vpred"
                        if low.startswith("eps") or "epsilon" in low:
                            return "eps"
        except Exception:
            pass
        return None

    def detect_model_version_from_model(self, model):
        mode = self._detect_pred_mode(model)
        if mode == "vpred":
            return "Illustrious v3.5 VPred"
        if mode == "eps":
            return "Illustrious v3.0 EPS"
        return "auto"

    def detect_optimal_version(self, width, height):
        """Detect optimal model version based on resolution (Illustrious-focused)"""
        max_dim = max(width, height)
        total_pixels = width * height

        if max_dim >= 3072 or total_pixels >= 3072 * 2048:  # ~6MP+
            return "Illustrious v3.0 VPred"
        if total_pixels >= 2048 * 1536:  # 3MP+
            return "Illustrious v2.0"
        elif max_dim >= 1536:  # High-res (native Illustrious)
            return "Illustrious v1.1"
        elif max_dim >= 1024:  # Standard SDXL
            return "Illustrious v1.0"
        else:
            return "Illustrious v0.1"

    def detect_resolution_tier(self, width, height):
        """Detect resolution tier"""
        max_dim = max(width, height)

        if max_dim >= 1792:
            return "2048"
        elif max_dim >= 1280:
            return "1536"
        elif max_dim >= 896:
            return "1024"
        elif max_dim >= 640:
            return "768"
        else:
            return "512"

    def detect_aspect_ratio_mode(self, width, height):
        """Detect aspect ratio mode"""
        aspect_ratio = width / height

        for mode, (min_ar, max_ar) in self.aspect_categories.items():
            if min_ar <= aspect_ratio <= max_ar:
                return mode

        return "landscape" if aspect_ratio > 1 else "portrait"

    def optimize_for_native_resolution(self, width, height, tier, version):
        """Optimize dimensions for Illustrious native resolutions"""
        tier_info = self.resolution_tiers.get(tier, self.resolution_tiers["1024"])
        native_sizes = tier_info["native_sizes"]

        # Find closest native size
        aspect_ratio = width / height
        best_match = None
        best_diff = float("inf")

        for native_w, native_h in native_sizes:
            native_ar = native_w / native_h
            ar_diff = abs(aspect_ratio - native_ar)

            if ar_diff < best_diff:
                best_diff = ar_diff
                best_match = (native_w, native_h)

        if best_match and version in tier_info["optimal_versions"]:
            # Use native size if it's close to requested size
            native_w, native_h = best_match
            size_diff = abs(width - native_w) + abs(height - native_h)

            # If the difference is reasonable, use native size
            if size_diff < max(width, height) * 0.2:  # Within 20% difference
                return native_w, native_h, True

        # Otherwise, round to multiples that work well with Illustrious models
        optimized_width = self.round_to_optimal_size(width, tier)
        optimized_height = self.round_to_optimal_size(height, tier)

        size_adjusted = (optimized_width != width) or (optimized_height != height)
        return optimized_width, optimized_height, size_adjusted

    def round_to_optimal_size(self, dimension, tier):
        """Round dimension to size optimal for Illustrious"""
        # Illustrious models work best with sizes divisible by 64; high-res prefers 128
        if tier in ["1536", "2048"]:
            # High-res tiers prefer multiples of 128
            return round(dimension / 128) * 128
        else:
            # Lower tiers use standard 64 multiples
            return round(dimension / 64) * 64

    def create_optimized_noise(
        self, batch_size, height, width, pattern, aspect_mode, version, seed
    ):
        """Create noise pattern optimized for Illustrious characteristics"""

        # Set seed for reproducibility
        if seed != 0:
            torch.manual_seed(seed)

        # Base noise tensor
        latent_image = torch.randn(batch_size, 4, height, width)

        if pattern == "illustrious_optimized":
            # Apply Illustrious-specific noise optimizations
            latent_image = self.apply_illustrious_noise_pattern(
                latent_image, aspect_mode, version
            )
        elif pattern == "high_frequency":
            # Add high-frequency details for intricate anime illustrations
            latent_image = self.add_high_frequency_noise(latent_image)
        elif pattern == "low_frequency":
            # Smoother noise for cleaner anime compositions
            latent_image = self.add_low_frequency_bias(latent_image)

        return latent_image

    def apply_illustrious_noise_pattern(self, latent_image, aspect_mode, version):
        """Apply Illustrious-specific noise patterns"""
        batch_size, channels, height, width = latent_image.shape

        # Version-specific optimizations (Illustrious-focused)
        if "Illustrious v2.0" in version:
            # v2.0 handles ultra high-res better with structured initial noise
            latent_image = self.add_multi_scale_structure(latent_image, strength=0.08)
        elif "Illustrious v1.1" in version:
            # v1.1 is very stable, needs minimal structure
            latent_image = self.add_multi_scale_structure(latent_image, strength=0.06)
        elif "Illustrious v1.0" in version:
            # v1.0 benefits from moderate structure
            latent_image = self.add_multi_scale_structure(latent_image, strength=0.1)
        elif "Illustrious v0.1" in version:
            # v0.1 benefits from more aggressive initial structure
            latent_image = self.add_multi_scale_structure(latent_image, strength=0.12)

        # Aspect ratio specific optimizations
        if aspect_mode == "portrait":
            # Encourage vertical coherence for character portraits
            latent_image = self.add_vertical_coherence(latent_image, strength=0.08)

        elif aspect_mode == "landscape":
            # Encourage horizontal coherence for landscape scenes
            latent_image = self.add_horizontal_coherence(latent_image, strength=0.08)

        elif aspect_mode == "ultra_wide":
            # Strong horizontal structure for ultra-wide anime scenes
            latent_image = self.add_horizontal_coherence(latent_image, strength=0.12)
            latent_image = self.reduce_vertical_noise(latent_image)

        # Add subtle anime/illustration bias optimized for Illustrious
        latent_image = self.add_anime_illustration_bias(latent_image, version)

        return latent_image

    def add_multi_scale_structure(self, latent_image, strength=0.1):
        """Add multi-scale structure to initial noise"""
        batch_size, channels, height, width = latent_image.shape

        # Create structure at different scales
        for scale in [2, 4, 8]:
            if height // scale > 1 and width // scale > 1:
                # Generate low-res structure
                low_res_noise = torch.randn(
                    batch_size, channels, height // scale, width // scale
                )

                # Upscale to full resolution
                upscaled = F.interpolate(
                    low_res_noise,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )

                # Blend with main noise
                scale_strength = strength / scale
                latent_image = (
                    latent_image * (1 - scale_strength) + upscaled * scale_strength
                )

        return latent_image

    def add_vertical_coherence(self, latent_image, strength=0.08):
        """Add vertical coherence for portrait orientations"""
        # Apply vertical smoothing
        # Create kernel with shape [out_channels=4, in_channels/groups=1, H=3, W=1]
        kernel = torch.tensor([[[[1], [2], [1]]]], dtype=latent_image.dtype) / 4
        kernel = kernel.repeat(4, 1, 1, 1)  # Shape: [4, 1, 3, 1]
        kernel = kernel.to(latent_image.device)

        # Latent has shape [batch, 4, H, W] - process all 4 channels at once
        smoothed = F.conv2d(
            latent_image,
            kernel,
            padding=(1, 0),
            groups=4,  # Process each channel independently
        )

        return latent_image * (1 - strength) + smoothed * strength

    def add_horizontal_coherence(self, latent_image, strength=0.08):
        """Add horizontal coherence for landscape orientations"""
        # Apply horizontal smoothing
        # Create kernel with shape [out_channels=4, in_channels/groups=1, H=1, W=3]
        kernel = torch.tensor([[[[1, 2, 1]]]], dtype=latent_image.dtype) / 4
        kernel = kernel.repeat(4, 1, 1, 1)  # Shape: [4, 1, 1, 3]
        kernel = kernel.to(latent_image.device)

        # Latent has shape [batch, 4, H, W] - process all 4 channels at once
        smoothed = F.conv2d(
            latent_image,
            kernel,
            padding=(0, 1),
            groups=4,  # Process each channel independently
        )

        return latent_image * (1 - strength) + smoothed * strength

    def reduce_vertical_noise(self, latent_image):
        """Reduce vertical frequency for ultra-wide compositions"""
        # Apply gentle vertical blur
        kernel = torch.ones(4, 1, 3, 1, dtype=latent_image.dtype) / 3
        kernel = kernel.to(latent_image.device)

        # Latent has shape [batch, 4, H, W] - process all 4 channels at once
        blurred = F.conv2d(
            latent_image,
            kernel,  # Kernel already has 4 output channels
            padding=(1, 0),
            groups=4,  # Process each channel independently
        )

        return latent_image * 0.7 + blurred * 0.3

    def add_anime_illustration_bias(self, latent_image, version):
        """Add subtle bias toward anime/illustration characteristics"""
        # This is optimized for Illustrious anime/illustration training
        # Add slight bias toward certain latent space regions that improve anime results
        if "Illustrious v1.1" in version or "Illustrious v2.0" in version:
            bias_strength = 0.08
        elif "Illustrious v1.0" in version:
            bias_strength = 0.06
        else:
            bias_strength = 0.04

        # Apply frequency-based bias optimized for anime content
        fft = torch.fft.fft2(latent_image)
        h, w = latent_image.shape[2], latent_image.shape[3]
        freq_bias = torch.ones_like(fft)

        # Create radial frequency mask
        center_h, center_w = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        radius = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
        mid_freq_mask = torch.exp(
            -((radius - min(h, w) * 0.3) ** 2) / (min(h, w) * 0.12) ** 2
        )
        freq_bias = 1 + mid_freq_mask * bias_strength

        fft_biased = fft * freq_bias.to(fft.device)
        latent_image = torch.real(torch.fft.ifft2(fft_biased))
        return latent_image

    def add_high_frequency_noise(self, latent_image):
        """Add high-frequency noise for detailed anime illustrations"""
        high_freq_noise = torch.randn_like(latent_image) * 0.1

        # Apply high-pass filter
        # Create kernel with shape [out_channels=4, in_channels/groups=1, H=3, W=3]
        kernel = torch.tensor(
            [[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=latent_image.dtype
        )
        kernel = kernel.repeat(4, 1, 1, 1)  # Shape: [4, 1, 3, 3]
        kernel = kernel.to(latent_image.device)

        # Latent has shape [batch, 4, H, W] - process all 4 channels at once
        high_freq = F.conv2d(
            high_freq_noise,
            kernel,
            padding=1,
            groups=4,  # Process each channel independently
        )

        return latent_image + high_freq * 0.06  # Slightly higher for anime details

    def add_low_frequency_bias(self, latent_image):
        """Add low-frequency bias for smoother anime compositions"""
        # Apply Gaussian smoothing
        kernel_size = 5
        sigma = 1.2  # Slightly higher sigma for smoother anime style
        kernel_1d = torch.exp(
            -0.5 * (torch.arange(kernel_size) - kernel_size // 2) ** 2 / sigma**2
        )
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).to(latent_image.device)

        # Latent has shape [batch, 4, H, W] - process all 4 channels at once
        smoothed = F.conv2d(
            latent_image,
            kernel.repeat(4, 1, 1, 1),  # Repeat kernel for each channel
            padding=kernel_size // 2,
            groups=4,  # Process each channel independently
        )

        return latent_image * 0.75 + smoothed * 0.25  # More smoothing for anime style

    def apply_version_optimizations(self, latent_image, version, mode, aspect_mode):
        """Apply version-specific optimizations"""

        if (
            "Illustrious v1.0" in version
            or "Illustrious v1.1" in version
            or "Illustrious v2.0" in version
        ) and mode == "quality":
            # Illustrious quality mode: Add subtle structure for better high-res anime results
            latent_image = self.add_quality_structure(latent_image)

        elif ("Illustrious v0.1" in version) and mode == "compatibility":
            # Older version compatibility: Reduce potential instabilities
            latent_image = self.apply_stability_smoothing(latent_image)

        elif mode == "speed":
            # Speed mode: Simplify noise pattern for faster convergence
            latent_image = self.simplify_noise_pattern(latent_image)

        return latent_image

    def add_quality_structure(self, latent_image):
        """Add structure for quality-focused anime generation"""
        # Add very subtle large-scale structure
        batch_size, channels, height, width = latent_image.shape

        # Large-scale structure for anime composition
        large_structure = F.interpolate(
            torch.randn(batch_size, channels, height // 16, width // 16),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        return (
            latent_image * 0.94 + large_structure * 0.06
        )  # Slightly stronger for anime

    def apply_stability_smoothing(self, latent_image):
        """Apply smoothing for stability with older model versions"""
        # Gentle overall smoothing
        kernel = torch.ones(4, 1, 3, 3, dtype=latent_image.dtype) / 9
        kernel = kernel.to(latent_image.device)

        # Latent has shape [batch, 4, H, W] - process all 4 channels at once
        smoothed = F.conv2d(
            latent_image,
            kernel,  # Kernel already has 4 output channels
            padding=1,
            groups=4,  # Process each channel independently
        )

        return latent_image * 0.88 + smoothed * 0.12  # More smoothing for v0.5

    def simplify_noise_pattern(self, latent_image):
        """Simplify noise for faster convergence in anime generation"""
        # Reduce high-frequency components for faster anime generation
        return F.avg_pool2d(
            F.interpolate(
                latent_image, scale_factor=0.5, mode="bilinear", align_corners=False
            ),
            kernel_size=1,
            stride=1,
        )

    def generate_resolution_report(
        self,
        width,
        height,
        latent_w,
        latent_h,
        version,
        tier,
        aspect_mode,
        size_adjusted,
        noise_pattern,
    ):
        """Generate detailed resolution report"""
        aspect_ratio = width / height
        total_mp = (width * height) / 1000000
        tier_info = self.resolution_tiers.get(tier, {})

        report = f"🎨 Illustrious Latent Generation Report\n"
        report += f"═" * 45 + "\n"
        report += f"Resolution: {width}x{height} ({total_mp:.1f}MP)\n"
        report += f"Latent Size: {latent_w}x{latent_h}\n"
        report += f"Aspect Ratio: {aspect_ratio:.2f}:1 ({aspect_mode})\n"
        report += f"Resolution Tier: {tier}\n"
        report += f"Model Version: {version}\n"
        report += f"Noise Pattern: {noise_pattern}\n"

        if size_adjusted:
            report += f"✨ Size optimized for native resolution\n"

        if tier_info:
            report += f"\n📊 Tier Characteristics:\n"
            report += (
                f"  {tier_info.get('characteristics', 'Standard resolution tier')}\n"
            )
            report += f"  Optimal for: {', '.join(tier_info.get('optimal_versions', [version]))}\n"

        return report


class IllustriousLatentUpscale:
    """Specialized latent upscaling for Illustrious high-resolution capabilities"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Latent tensor to upscale."}),
                "upscale_method": (
                    ["nearest-exact", "bilinear", "area", "bicubic"],
                    {"default": "bilinear", "tooltip": "Interpolation method for latent upscaling."},
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.25,
                        "max": 8.0,
                        "step": 0.05,
                        "tooltip": "Scale multiplier (e.g., 2.0 doubles latent size).",
                    },
                ),
            },
            "optional": {
                "model_version": (
                    [
                        "auto",
                        "Illustrious v0.1",
                        "Illustrious v1.0",
                        "Illustrious v1.1",
                        "Illustrious v2.0",
                    ],
                    {"default": "auto", "tooltip": "Leave auto unless you need version-specific tweaks."},
                ),
                "preserve_anime_details": ("BOOLEAN", {"default": True, "tooltip": "Preserve anime-style edges and textures post-scale."}),
                "crop": (["disabled", "center"], {"default": "disabled", "tooltip": "Optionally center-crop to exact size."}),
                "noise_augmentation": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.01, "tooltip": "Add tiny noise to enhance detail after upscale."},
                ),
                # Deprecated inputs kept for backward compatibility with older workflows
                "width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 64,
                        "tooltip": "Deprecated: use 'scale' instead. When > 0 (with height), overrides scale.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8192,
                        "step": 64,
                        "tooltip": "Deprecated: use 'scale' instead. When > 0 (with width), overrides scale.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale_illustrious"
    CATEGORY = "Easy Illustrious / Latent Image"

    def upscale_illustrious(
        self,
        samples,
        upscale_method,
        scale=2.0,
        model_version="auto",
        preserve_anime_details=True,
        crop="disabled",
        noise_augmentation=0.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):

        latent_samples = samples["samples"]
        current_h, current_w = latent_samples.shape[2], latent_samples.shape[3]

        # Determine target size: prefer deprecated absolute width/height when provided (>0), otherwise use scale
        target_h: int
        target_w: int
        if width is not None and height is not None and width > 0 and height > 0:
            # Backward compatibility path (deprecated): interpret as pixel size, convert to latent grid
            target_w = max(1, int(width // 8))
            target_h = max(1, int(height // 8))
            print(
                "[IllustriousLatentUpscale] Using deprecated width/height inputs; please switch to 'scale'."
            )
        else:
            # Multiplier path (preferred)
            # Round to nearest integer latent size, ensure minimum 1
            target_h = max(1, int(round(current_h * float(scale))))
            target_w = max(1, int(round(current_w * float(scale))))

        # Upscale the latent
        upscaled = F.interpolate(
            latent_samples,
            size=(target_h, target_w),
            mode=upscale_method,
            align_corners=False,
        )

        # Apply Illustrious-specific enhancements
        if preserve_anime_details and (
            model_version == "auto" or "Illustrious" in model_version
        ):
            # Add detail preservation for anime content in high-res capable versions
            upscaled = self.preserve_anime_details(latent_samples, upscaled)

        # Add noise augmentation if enabled (good for anime detail enhancement)
        if noise_augmentation > 0:
            noise = torch.randn_like(upscaled) * noise_augmentation
            upscaled = upscaled + noise

        # Crop if requested
        if crop == "center":
            upscaled = self.center_crop_latent(upscaled, target_w, target_h)

        return ({"samples": upscaled},)

    def preserve_anime_details(self, original, upscaled):
        """Preserve anime-specific details during upscaling"""
        # This is optimized for anime/illustration content
        # Add subtle high-frequency preservation for anime details like hair, eyes, clothing patterns

        # Calculate high-frequency components from original
        if original.shape[2] < upscaled.shape[2]:  # Only if actually upscaling
            # Apply edge enhancement filter to preserve anime line art
            # Create kernel with shape [out_channels=4, in_channels/groups=1, H=3, W=3]
            edge_kernel = torch.tensor(
                [[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]],
                dtype=upscaled.dtype,
                device=upscaled.device,
            )
            edge_kernel = edge_kernel.repeat(4, 1, 1, 1)  # Shape: [4, 1, 3, 3]

            # Latent has shape [batch, 4, H, W] - process all 4 channels at once
            edges = F.conv2d(
                upscaled, edge_kernel, padding=1, groups=4
            )  # Process each channel independently

            # Subtle edge enhancement for anime content
            upscaled = upscaled + edges * 0.03

        return upscaled

    def center_crop_latent(self, latent, target_w, target_h):
        """Center crop latent to target dimensions"""
        current_h, current_w = latent.shape[2], latent.shape[3]

        if current_h == target_h and current_w == target_w:
            return latent

        # Calculate crop coordinates
        start_h = (current_h - target_h) // 2
        start_w = (current_w - target_w) // 2

        return latent[:, :, start_h : start_h + target_h, start_w : start_w + target_w]
