import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional
import comfy.model_management
import comfy.utils
from server import PromptServer


class IllustriousVAEDecode:
    """VAE Decode optimized for Illustrious models with high-resolution support (neutral by default)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Latent tensor to decode to images."}),
                "vae": ("VAE", {"tooltip": "Variational Autoencoder used to decode latents."}),
            },
            "optional": {
                "illustrious_version": (
                    ["auto", "v0.1", "v1.0", "v1.1", "v2.0"],
                    {"default": "auto", "tooltip": "Auto-detect from latent size; choose specific for nudged settings."},
                ),
                "optimization_mode": (
                    ["auto", "quality", "speed", "memory"],
                    {"default": "auto", "tooltip": "Adjusts tile size/overlap and enhancements."},
                ),
                "enable_tiling": ("BOOLEAN", {"default": True, "tooltip": "Enable when decoding very large images to avoid VRAM issues."}),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 256, "max": 1024, "step": 64, "tooltip": "Decode tile size in image pixels."},
                ),
                "overlap": ("INT", {"default": 64, "min": 32, "max": 128, "step": 16, "tooltip": "Overlap between tiles to hide seams."}),
                # Neutral path + safer defaults
                "neutral_decode": ("BOOLEAN", {"default": True, "tooltip": "Decode only; no enhancements unless enabled below."}),
                "color_fix": ("BOOLEAN", {"default": False, "tooltip": "Apply gentle Illustrious color correction when needed."}),
                "seamless_tiling": ("BOOLEAN", {"default": True, "tooltip": "Blend tiles with feathering to avoid seams."}),
                "preserve_details": ("BOOLEAN", {"default": False, "tooltip": "Apply subtle detail preservation in standard/tiled paths."}),
                "memory_efficient": ("BOOLEAN", {"default": True, "tooltip": "Lower thresholds to reduce VRAM usage."}),
                # New highlight tone mapping control
                "Highlight Roll-Off": (
                    ["Clamped", "Soft", "Filmic"],
                    {"default": "Clamped", "tooltip": "Control highlight compression to avoid harsh clipping."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "decode_info")
    FUNCTION = "illustrious_decode"
    CATEGORY = "Easy Illustrious / VAE"

    def __init__(self):
        # Illustrious-specific VAE characteristics
        self.version_profiles = {
            "v0.1": {
                "optimal_tile_size": 512,
                "color_correction_needed": True,
                "memory_multiplier": 1.2,
                "detail_preservation": "medium",
            },
            "v1.0": {
                "optimal_tile_size": 512,
                "color_correction_needed": False,
                "memory_multiplier": 1.0,
                "detail_preservation": "high",
            },
            "v1.1": {
                "optimal_tile_size": 640,
                "color_correction_needed": False,
                "memory_multiplier": 0.9,
                "detail_preservation": "high",
            },
            "v2.0": {
                "optimal_tile_size": 768,
                "color_correction_needed": False,
                "memory_multiplier": 0.8,
                "detail_preservation": "very_high",
            },
        }

    def send_vae_guidance_toast(self, severity, summary, detail):
        """Send VAE guidance toast to frontend"""
        try:
            if hasattr(PromptServer, "instance") and PromptServer.instance:
                PromptServer.instance.send_sync(
                    "illustrious.guidance",
                    {
                        "severity": severity,
                        "summary": summary,
                        "detail": detail,
                        "life": 4000,
                    },
                )
        except Exception as e:
            print(f"[VAE] Toast notification failed: {e}")

    def illustrious_decode(
        self,
        samples,
        vae,
        illustrious_version="auto",
        optimization_mode="auto",
        enable_tiling=True,
        tile_size=512,
        overlap=64,
        neutral_decode=True,
        color_fix=False,
        seamless_tiling=True,
        preserve_details=False,
        memory_efficient=True,
        **extra,
    ):

        highlight_rolloff = extra.get("Highlight Roll-Off", "Clamped")

        # Auto-detect version if needed
        if illustrious_version == "auto":
            detected_version = self.detect_version_from_latent(samples)
            self.send_vae_guidance_toast(
                "info",
                "Auto-Detection",
                f"Detected {detected_version} settings based on {samples['samples'].shape[2]*8}x{samples['samples'].shape[3]*8} resolution",
            )
            illustrious_version = detected_version

        # Get version profile
        profile = self.version_profiles.get(
            illustrious_version, self.version_profiles["v1.1"]
        )

        # Optimize settings based on latent size and version
        optimized_settings = self.optimize_decode_settings(
            samples, profile, optimization_mode, tile_size, overlap, memory_efficient
        )

        # Neutral fast path: decode only (tiling if needed), no enhancements/compensation
        if neutral_decode:
            if optimized_settings["use_tiling"] and enable_tiling:
                decoded_images = self.tiled_decode(
                    samples,
                    vae,
                    optimized_settings,
                    seamless_tiling,
                    preserve_details=False,
                )
            else:
                decoded_images = self.standard_decode(
                    samples, vae, optimized_settings, preserve_details=False
                )
            # Apply highlight roll-off even in neutral mode
            decoded_images = self.apply_highlight_rolloff(
                decoded_images, highlight_rolloff
            )
            info = self.generate_decode_info(
                optimized_settings, profile, illustrious_version
            )
            info += f"Highlight Roll-Off: {highlight_rolloff}\n"
            return (torch.clamp(decoded_images, 0, 1), info)

        # Show optimization results
        if optimized_settings["use_tiling"] and enable_tiling:
            total_pixels = (
                samples["samples"].shape[2] * samples["samples"].shape[3] * 64
            )  # Convert to pixel count
            self.send_vae_guidance_toast(
                "success",
                "Tiling Enabled",
                f"Using {optimized_settings['tile_size']*8}px tiles for {total_pixels//1000000:.1f}MP image. This prevents VRAM overflow.",
            )
        elif optimization_mode == "quality":
            self.send_vae_guidance_toast(
                "info", "Quality Mode", "Enhanced processing enabled."
            )
        elif optimization_mode == "speed":
            self.send_vae_guidance_toast(
                "info",
                "Speed Mode",
                "Fast processing mode - some enhancements disabled for quicker results.",
            )

        print(
            f"🎨 Illustrious VAE Decode: {illustrious_version} | "
            f"{optimized_settings['method']} | {samples['samples'].shape}"
        )

        # Choose decoding method based on settings
        if optimized_settings["use_tiling"] and enable_tiling:
            decoded_images = self.tiled_decode(
                samples, vae, optimized_settings, seamless_tiling, preserve_details
            )
        else:
            decoded_images = self.standard_decode(
                samples, vae, optimized_settings, preserve_details
            )

        # Store original value range for *gentle* compensation
        original_min = decoded_images.min()
        original_max = decoded_images.max()
        original_mean = decoded_images.mean()

        # Post-processing corrections (optional, float-safe)
        enhancement_count = 0
        if color_fix and profile["color_correction_needed"]:
            decoded_images = self.apply_color_correction(
                decoded_images, illustrious_version
            )
            enhancement_count += 1

        if preserve_details:
            decoded_images = self.apply_detail_enhancement(decoded_images, profile)
            enhancement_count += 1

        # Enhancement summary
        if enhancement_count > 0:
            if enhancement_count == 1:
                self.send_vae_guidance_toast(
                    "success",
                    "Enhancement Applied",
                    f"Applied {'color correction' if color_fix else 'detail enhancement'} optimized for {illustrious_version}.",
                )
            else:
                self.send_vae_guidance_toast(
                    "success",
                    "Full Enhancement",
                    f"Applied color correction + detail enhancement for {illustrious_version}.",
                )

        # Gentle compensation only for pathological cases (no flattening)
        decoded_images = self.apply_highlight_rolloff(decoded_images, highlight_rolloff)
        decoded_images = self.compensate_value_range(
            decoded_images, original_min, original_max, original_mean
        )

        # Generate decode information
        decode_info = self.generate_decode_info(
            optimized_settings, profile, illustrious_version
        )
        decode_info += f"Highlight Roll-Off: {highlight_rolloff}\n"

        return (decoded_images, decode_info)

    def detect_version_from_latent(self, samples):
        """Auto-detect Illustrious version from latent characteristics"""
        latent_samples = samples["samples"]
        h, w = latent_samples.shape[2], latent_samples.shape[3]
        pixel_h, pixel_w = h * 8, w * 8

        # Heuristics based on typical resolution usage
        if max(pixel_h, pixel_w) >= 1792:
            return "v2.0"  # Likely v2.0 for ultra high-res
        elif max(pixel_h, pixel_w) >= 1536:
            return "v1.1"  # v1.0+ native high-res
        elif max(pixel_h, pixel_w) >= 1024:
            return "v1.0"
        else:
            return "v0.1"

    def optimize_decode_settings(
        self, samples, profile, optimization_mode, tile_size, overlap, memory_efficient
    ):
        """Optimize decode settings based on input and target"""
        latent_samples = samples["samples"]
        batch_size, channels, h, w = latent_samples.shape
        pixel_h, pixel_w = h * 8, w * 8
        total_pixels = pixel_h * pixel_w * batch_size

        settings = {
            "use_tiling": False,
            "tile_size": tile_size,
            "overlap": overlap,
            "method": "standard",
            "memory_strategy": "normal",
        }

        # Determine if tiling is needed
        memory_threshold = 2048 * 2048  # 4MP per image
        if memory_efficient:
            memory_threshold = 1536 * 1536  # 2.3MP per image

        if total_pixels > memory_threshold:
            settings["use_tiling"] = True
            settings["method"] = "tiled"

            # Optimize tile size based on version
            optimal_tile = profile["optimal_tile_size"]
            settings["tile_size"] = min(optimal_tile, max(256, tile_size))

            # Warn about tile size extremes
            if tile_size > 768:
                self.send_vae_guidance_toast(
                    "warn",
                    "Large Tile Size",
                    f"{tile_size}px tiles may cause VRAM issues. Auto-limiting to {settings['tile_size']}px for stability.",
                )
            elif tile_size < 384:
                self.send_vae_guidance_toast(
                    "warn",
                    "Small Tile Size",
                    f"{tile_size}px tiles may create visible seams. Consider 512px+ for better quality.",
                )

            # Adjust overlap based on image size
            if max(pixel_h, pixel_w) > 2048:
                settings["overlap"] = max(
                    overlap, 96
                )  # Larger overlap for very high-res
                if overlap < 96:
                    self.send_vae_guidance_toast(
                        "info",
                        "Overlap Increased",
                        f"Overlap increased to {settings['overlap']}px for ultra-high-res processing. This reduces seams.",
                    )

        # Optimization mode adjustments
        if optimization_mode == "speed":
            settings["tile_size"] = min(settings["tile_size"], 512)
            settings["overlap"] = max(32, settings["overlap"] // 2)
        elif optimization_mode == "quality":
            settings["tile_size"] = max(settings["tile_size"], 640)
            settings["overlap"] = max(settings["overlap"], 80)
        elif optimization_mode == "memory":
            settings["use_tiling"] = True
            settings["tile_size"] = 384
            settings["overlap"] = 48
            settings["memory_strategy"] = "conservative"

        return settings

    def tiled_decode(self, samples, vae, settings, seamless_tiling, preserve_details):
        """Perform tiled VAE decode for large images (NHWC-safe)."""
        latent_samples = samples["samples"]
        batch_size, channels, h, w = latent_samples.shape

        tile_size = settings["tile_size"] // 8  # Convert to latent space
        overlap = settings["overlap"] // 8

        # Calculate tile positions
        tiles = self.calculate_tile_positions(h, w, tile_size, overlap)

        decoded_tiles = []
        total_tiles = len(tiles)

        print(f"🧩 Decoding {total_tiles} tiles ({tile_size*8}x{tile_size*8} each)")

        # Process each tile
        for i, (y_start, y_end, x_start, x_end) in enumerate(tiles):
            # Extract tile from latent (NCHW)
            tile_latent = latent_samples[:, :, y_start:y_end, x_start:x_end]
            tile_sample = {"samples": tile_latent}

            # Decode tile -> NHWC image in Comfy
            tile_decoded = vae.decode(tile_sample["samples"])  # [B, H, W, C]

            # Check for processing interruption
            comfy.model_management.throw_exception_if_processing_interrupted()

            # Optional per-tile enhancement (reduced strength, no range remap)
            if preserve_details:
                tile_decoded = self.enhance_tile_details(tile_decoded, i, total_tiles)

            # Keep in [0,1] without compressing contrast
            tile_decoded = torch.clamp(tile_decoded, 0, 1)

            decoded_tiles.append(
                {
                    "image": tile_decoded,  # NHWC
                    "position": (y_start * 8, y_end * 8, x_start * 8, x_end * 8),
                    "tile_id": i,
                }
            )

        # Combine tiles into final image with improved blending (NHWC)
        if seamless_tiling:
            combined_image = self.seamless_tile_combine(
                decoded_tiles, h * 8, w * 8, settings["overlap"]
            )
        else:
            combined_image = self.standard_tile_combine(decoded_tiles, h * 8, w * 8)

        return combined_image

    def standard_decode(self, samples, vae, settings, preserve_details):
        """Enhanced standard VAE decode with Illustrious optimizations (NHWC)."""
        latent_samples = samples["samples"]

        # Apply standard VAE decode
        decoded = vae.decode(latent_samples)  # [B, H, W, C]
        comfy.model_management.throw_exception_if_processing_interrupted()

        # Optional subtle enhancements (disabled if preserve_details=False or speed)
        if preserve_details and settings.get("method") != "speed":
            # Store original range for possible guard
            original_min = decoded.min()
            original_max = decoded.max()
            original_mean = decoded.mean()

            decoded = self.apply_standard_enhancement(decoded)

            # Gentle range guard only if needed
            decoded = self.compensate_value_range(
                decoded, original_min, original_max, original_mean
            )

        return decoded

    def apply_standard_enhancement(self, images):
        """Apply subtle enhancements for standard (non-tiled) decoding (NHWC)."""
        # Gentle sharpening kernel
        sharpen_kernel = torch.tensor(
            [[[[-0.05, -0.10, -0.05], [-0.10, 1.40, -0.10], [-0.05, -0.10, -0.05]]]],
            dtype=images.dtype,
            device=images.device,
        )

        # Process each RGB channel separately (NHWC -> NCHW per-channel)
        sharpened_channels = []
        for c in range(3):  # RGB channels last
            channel = images[:, :, :, c].unsqueeze(1)  # [N,1,H,W]
            sharpened_channel = torch.nn.functional.conv2d(
                channel, sharpen_kernel, padding=1
            )
            sharpened_channels.append(sharpened_channel.squeeze(1))  # [N,H,W]

        # Stack channels back together [N,H,W,3]
        sharpened = torch.stack(sharpened_channels, dim=-1)

        # Gentle color enhancement in HSV
        enhanced = self.apply_gentle_anime_enhancement(images)

        # Much lighter mixing to avoid flattening
        combined = images * 0.92 + sharpened * 0.05 + enhanced * 0.03

        return torch.clamp(combined, 0, 1)

    def apply_gentle_anime_enhancement(self, images):
        """Apply gentle anime-specific color enhancement with white protection (NHWC)."""
        hsv = self.rgb_to_hsv_torch(images)

        hue = hsv[:, :, :, 0]
        saturation = hsv[:, :, :, 1]
        luminance = hsv[:, :, :, 2]

        # Protect whites and light yellow highlights
        white_mask = (luminance > 0.85) & (saturation < 0.20)
        light_yellow_mask = (hue > 0.13) & (hue < 0.19) & (luminance > 0.75)
        protected_mask = white_mask | light_yellow_mask

        # Safe mid-tones
        safe_mid_tone_mask = (luminance > 0.25) & (luminance < 0.75) & (~protected_mask)

        # Very gentle enhancement
        hsv[:, :, :, 1][safe_mid_tone_mask] = torch.clamp(
            hsv[:, :, :, 1][safe_mid_tone_mask] * 1.05, 0, 1
        )

        return self.hsv_to_rgb_torch(hsv)

    def calculate_tile_positions(self, h, w, tile_size, overlap):
        """Calculate optimal tile positions with overlap (latent space)."""
        tiles = []

        # Y positions
        y_positions = []
        y = 0
        while y < h:
            y_end = min(y + tile_size, h)
            y_positions.append((y, y_end))
            if y_end >= h:
                break
            y += tile_size - overlap

        # X positions
        x_positions = []
        x = 0
        while x < w:
            x_end = min(x + tile_size, w)
            x_positions.append((x, x_end))
            if x_end >= w:
                break
            x += tile_size - overlap

        # Combine into tile coordinates
        for y_start, y_end in y_positions:
            for x_start, x_end in x_positions:
                tiles.append((y_start, y_end, x_start, x_end))

        return tiles

    def seamless_tile_combine(self, decoded_tiles, final_h, final_w, overlap):
        """Combine tiles with seamless blending (NHWC)."""
        # Create output tensor (NHWC)
        batch_size = decoded_tiles[0]["image"].shape[0]
        channels = decoded_tiles[0]["image"].shape[3]
        combined = torch.zeros(
            (batch_size, final_h, final_w, channels),
            dtype=decoded_tiles[0]["image"].dtype,
            device=decoded_tiles[0]["image"].device,
        )

        weight_map = torch.zeros((final_h, final_w), device=combined.device)

        for tile_data in decoded_tiles:
            tile_image = tile_data["image"]  # [N,Ht,Wt,C]
            y_start, y_end, x_start, x_end = tile_data["position"]

            # Feather mask NHWC: [1,Ht,Wt,1]
            tile_h, tile_w = tile_image.shape[1], tile_image.shape[2]
            feather_mask = self.create_feather_mask(tile_h, tile_w, overlap).to(
                tile_image.device
            )  # [1,H,W,1]

            # Apply tile to combined image
            combined[:, y_start:y_end, x_start:x_end, :] += tile_image * feather_mask
            weight_map[y_start:y_end, x_start:x_end] += feather_mask[0, :, :, 0]

        # Normalize by weight map with safe clamp
        weight_map = torch.clamp(weight_map, min=1e-6)
        combined = combined / weight_map[None, :, :, None]

        return combined

    def standard_tile_combine(self, decoded_tiles, final_h, final_w):
        """Standard tile combination without blending (NHWC)."""
        batch_size = decoded_tiles[0]["image"].shape[0]
        channels = decoded_tiles[0]["image"].shape[3]
        combined = torch.zeros(
            (batch_size, final_h, final_w, channels),
            dtype=decoded_tiles[0]["image"].dtype,
            device=decoded_tiles[0]["image"].device,
        )

        for tile_data in decoded_tiles:
            tile_image = tile_data["image"]  # [N,Ht,Wt,C]
            y_start, y_end, x_start, x_end = tile_data["position"]
            combined[:, y_start:y_end, x_start:x_end, :] = tile_image

        return combined

    def create_feather_mask(self, h, w, overlap):
        """Create feathering mask for seamless tile blending (NHWC mask [1,H,W,1])."""
        mask = torch.ones((1, h, w, 1), dtype=torch.float32)

        if overlap > 0:
            fade_size = min(overlap, h // 4, w // 4)
            if fade_size > 0:
                # vertical fades
                for i in range(fade_size):
                    factor = (i + 1) / fade_size
                    mask[:, i, :, :] *= factor
                    mask[:, h - 1 - i, :, :] *= factor
                # horizontal fades
                for i in range(fade_size):
                    factor = (i + 1) / fade_size
                    mask[:, :, i, :] *= factor
                    mask[:, :, w - 1 - i, :] *= factor
        return mask

    def enhance_tile_details(self, tile_image, tile_id, total_tiles):
        """Apply reduced-strength detail enhancement (NHWC)."""
        # Fine detail enhancement kernel
        fine_kernel = torch.tensor(
            [[[[-0.10, -0.20, -0.10], [-0.20, 2.60, -0.20], [-0.10, -0.20, -0.10]]]],
            dtype=tile_image.dtype,
            device=tile_image.device,
        )

        # Per-channel conv (NHWC)
        fine_channels = []
        for c in range(3):
            channel = tile_image[:, :, :, c].unsqueeze(1)  # [N,1,H,W]
            fine_channel = torch.nn.functional.conv2d(channel, fine_kernel, padding=1)
            fine_channels.append(fine_channel.squeeze(1))
        fine_enhanced = torch.stack(fine_channels, dim=-1)  # NHWC

        # Edge enhancement kernel (softer)
        edge_kernel = (
            torch.tensor(
                [[[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]]],
                dtype=tile_image.dtype,
                device=tile_image.device,
            )
            / 2.0
        )
        edge_channels = []
        for c in range(3):
            channel = tile_image[:, :, :, c].unsqueeze(1)
            edge_channel = torch.nn.functional.conv2d(channel, edge_kernel, padding=1)
            edge_channels.append(edge_channel.squeeze(1))
        edge_enhanced = torch.stack(edge_channels, dim=-1)

        # Contrast enhancement (mild)
        contrast_enhanced = torch.clamp((tile_image - 0.5) * 1.08 + 0.5, 0, 1)

        # Reduced strengths
        fine_strength = 0.10
        edge_strength = 0.08
        contrast_strength = 0.05

        enhanced = (
            tile_image * (1 - fine_strength - edge_strength - contrast_strength)
            + fine_enhanced * fine_strength
            + edge_enhanced * edge_strength
            + contrast_enhanced * contrast_strength
        )

        return torch.clamp(enhanced, 0, 1)

    def apply_color_correction(self, images, version):
        """Apply color correction optimized for Illustrious versions (float-safe)."""
        if version in ["v0.1", "v1.0"]:
            images = self.apply_illustrious_color_enhancement(images, version)
        # Optionally, you could add a tiny vibrance here—kept neutral.
        return images

    def apply_detail_enhancement(self, images, profile):
        """Apply luminance-only sharpening to avoid color shifts."""
        if profile["detail_preservation"] in ["medium", "high", "very_high"]:
            images = self.apply_luminance_sharpening(images, profile)
        return images

    def apply_safe_sharpening(self, images, profile):
        """Simple sharpening that won't cause color shifts (NHWC)."""
        strength = {"medium": 0.10, "high": 0.15, "very_high": 0.20}[
            profile["detail_preservation"]
        ]

        kernel = torch.tensor(
            [[[[-0.05, -0.10, -0.05], [-0.10, 1.40, -0.10], [-0.05, -0.10, -0.05]]]],
            dtype=images.dtype,
            device=images.device,
        )

        sharpened_channels = []
        for c in range(3):
            channel = images[:, :, :, c].unsqueeze(1)
            sharpened_channel = torch.nn.functional.conv2d(channel, kernel, padding=1)
            sharpened_channels.append(sharpened_channel.squeeze(1))
        sharpened = torch.stack(sharpened_channels, dim=-1)

        enhanced = images * (1 - strength) + sharpened * strength
        return torch.clamp(enhanced, 0, 1)

    def apply_luminance_sharpening(self, images, profile):
        """Sharpen only luminance channel to avoid color shifts (NHWC)."""
        strength = {"medium": 0.15, "high": 0.25, "very_high": 0.35}[
            profile["detail_preservation"]
        ]

        hsv = self.rgb_to_hsv_torch(images)
        luminance = hsv[:, :, :, 2]  # V channel

        kernel = torch.tensor(
            [[[[-0.10, -0.20, -0.10], [-0.20, 2.20, -0.20], [-0.10, -0.20, -0.10]]]],
            dtype=images.dtype,
            device=images.device,
        )

        sharpened_luma = torch.nn.functional.conv2d(
            luminance.view(-1, 1, luminance.shape[1], luminance.shape[2]),
            kernel,
            padding=1,
        ).view_as(luminance)

        enhanced_luma = torch.clamp(
            luminance * (1 - strength) + sharpened_luma * strength, 0, 1
        )
        hsv[:, :, :, 2] = enhanced_luma

        return self.hsv_to_rgb_torch(hsv)

    def generate_decode_info(self, settings, profile, version):
        """Generate decode information report."""
        info = f"🎨 Illustrious VAE Decode Report\n"
        info += f"{'═'*40}\n"
        info += f"Version: {version}\n"
        info += f"Method: {settings['method']}\n"

        if settings["use_tiling"]:
            info += (
                f"Tile Size: {settings['tile_size'] * 8}x{settings['tile_size'] * 8}\n"
            )
            info += f"Overlap: {settings['overlap'] * 8}px\n"

        info += f"Memory Strategy: {settings['memory_strategy']}\n"
        info += f"Detail Preservation: {profile['detail_preservation']}\n"
        info += f"Color Correction: {'Needed' if profile['color_correction_needed'] else 'Not needed'}\n"

        return info

    def apply_illustrious_color_enhancement(self, images, version):
        """Illustrious color tweaks in float32 (no 8-bit round-trip), NHWC."""
        import cv2

        imgs = images.detach().to(torch.float32)
        b, h, w, c = imgs.shape

        out = []
        for i in range(b):
            img = imgs[i].cpu().numpy()  # float32 0..1
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)  # float32, 0..1-ish in OpenCV

            # Map to tweakable space: L ~ [0..100], a/b ~ [-128..127]
            L = lab[..., 0] * 100.0
            a = (lab[..., 1] - 0.5) * 255.0
            bch = (lab[..., 2] - 0.5) * 255.0

            white_mask = L > 85.0
            midtone_mask = (L >= 25.0) & (L <= 75.0)

            # Gentle luminance adjustment
            L = np.where(white_mask, L, L * 1.02)

            # Midtone a/b tweaks
            if version == "v0.1":
                a = np.where(midtone_mask, a * 1.08, a)
                bch = np.where(midtone_mask, bch * 1.06, bch)
            else:
                a = np.where(midtone_mask, a * 1.05, a)
                bch = np.where(midtone_mask, bch * 1.04, bch)

            # Neutralize whites slightly
            a[white_mask] *= 0.95
            bch[white_mask] *= 0.95

            # Clamp to valid-ish ranges
            L = np.clip(L, 0.0, 100.0)
            a = np.clip(a, -127.0, 127.0)
            bch = np.clip(bch, -127.0, 127.0)

            # Back to OpenCV float LAB 0..1-ish
            lab_float = np.empty_like(lab, dtype=np.float32)
            lab_float[..., 0] = L / 100.0
            lab_float[..., 1] = (a / 255.0) + 0.5
            lab_float[..., 2] = (bch / 255.0) + 0.5

            rgb = cv2.cvtColor(lab_float, cv2.COLOR_LAB2RGB)
            out.append(rgb.astype(np.float32))

        return torch.from_numpy(np.stack(out, axis=0)).to(images.device)

    def apply_anime_color_grading(self, images):
        """(Unused in neutral defaults) Stronger grading removed to avoid washout."""
        hsv = self.rgb_to_hsv_torch(images)
        hue = hsv[:, :, :, 0]
        saturation = hsv[:, :, :, 1]
        luminance = hsv[:, :, :, 2]

        white_mask = (luminance > 0.85) & (saturation < 0.2)
        yellow_mask = (hue > 0.13) & (hue < 0.19) & (luminance > 0.75)
        safe_mask = ~(white_mask | yellow_mask)
        mid_tone_mask = (luminance > 0.25) & (luminance < 0.75) & safe_mask

        hsv[:, :, :, 1][mid_tone_mask] = torch.clamp(
            hsv[:, :, :, 1][mid_tone_mask] * 1.10, 0, 1
        )
        return self.hsv_to_rgb_torch(hsv)

    def apply_multi_scale_sharpening(self, images, profile):
        """Multi-scale sharpening (kept but not used by default)."""
        strength = {"medium": 0.15, "high": 0.25, "very_high": 0.35}[
            profile["detail_preservation"]
        ]

        # Fine kernel
        fine_kernel = torch.tensor(
            [[[[-0.10, -0.20, -0.10], [-0.20, 2.20, -0.20], [-0.10, -0.20, -0.10]]]],
            dtype=images.dtype,
            device=images.device,
        )
        fine_channels = []
        for c in range(3):
            channel = images[:, :, :, c].unsqueeze(1)
            fine_channel = torch.nn.functional.conv2d(channel, fine_kernel, padding=1)
            fine_channels.append(fine_channel.squeeze(1))
        fine_sharpened = torch.stack(fine_channels, dim=-1)

        # Medium kernel
        medium_kernel = torch.tensor(
            [[[[-0.05, -0.10, -0.05], [-0.10, 1.50, -0.10], [-0.05, -0.10, -0.05]]]],
            dtype=images.dtype,
            device=images.device,
        )
        medium_channels = []
        for c in range(3):
            channel = images[:, :, :, c].unsqueeze(1)
            medium_channel = torch.nn.functional.conv2d(
                channel, medium_kernel, padding=1
            )
            medium_channels.append(medium_channel.squeeze(1))
        medium_sharpened = torch.stack(medium_channels, dim=-1)

        enhanced = (
            images * (1 - strength)
            + fine_sharpened * (strength * 0.6)
            + medium_sharpened * (strength * 0.4)
        )
        return torch.clamp(enhanced, 0, 1)

    def apply_anime_edge_enhancement(self, images, profile):
        """Anime-specific edge enhancement (kept but not used by default)."""
        strength = 0.2 if profile["detail_preservation"] == "very_high" else 0.15

        edge_kernel = (
            torch.tensor(
                [[[[-1, -2, -1], [-2, 12, -2], [-1, -2, -1]]]],
                dtype=images.dtype,
                device=images.device,
            )
            / 8.0
        )
        edge_channels = []
        for c in range(3):
            channel = images[:, :, :, c].unsqueeze(1)
            edge_channel = torch.nn.functional.conv2d(channel, edge_kernel, padding=1)
            edge_channels.append(edge_channel.squeeze(1))
        edges = torch.stack(edge_channels, dim=-1)

        edge_magnitude = torch.abs(edges)
        edge_mask = edge_magnitude > 0.02
        enhanced = images.clone()
        enhanced[edge_mask] = enhanced[edge_mask] + edges[edge_mask] * strength
        return torch.clamp(enhanced, 0, 1)

    def apply_frequency_enhancement(self, images):
        """Frequency domain enhancement (kept but not used by default)."""
        batch_size, height, width, channels = images.shape
        enhanced_batch = []

        for b in range(batch_size):
            enhanced_channels = []
            for c in range(channels):
                channel = images[b, :, :, c]
                fft = torch.fft.fft2(channel)
                fft_shifted = torch.fft.fftshift(fft)

                center_h, center_w = height // 2, width // 2
                y, x = torch.meshgrid(
                    torch.arange(height), torch.arange(width), indexing="ij"
                )
                y, x = y.to(images.device), x.to(images.device)

                radius = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
                max_radius = min(height, width) // 2
                freq_mask = torch.exp(
                    -((radius - max_radius * 0.3) ** 2) / (max_radius * 0.15) ** 2
                )
                enhancement_factor = 1.0 + freq_mask * 0.2

                fft_enhanced = fft_shifted * enhancement_factor
                fft_unshifted = torch.fft.ifftshift(fft_enhanced)
                enhanced_channel = torch.real(torch.fft.ifft2(fft_unshifted))
                enhanced_channels.append(enhanced_channel)
            enhanced_image = torch.stack(enhanced_channels, dim=-1)
            enhanced_batch.append(enhanced_image)
        return torch.clamp(torch.stack(enhanced_batch), 0, 1)

    def rgb_to_hsv_torch(self, rgb):
        """Convert RGB to HSV using PyTorch operations (NHWC)."""
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = torch.max(rgb, dim=-1)[0]
        minc = torch.min(rgb, dim=-1)[0]
        deltac = maxc - minc

        h = torch.zeros_like(maxc)
        mask = deltac != 0

        r_mask = (maxc == r) & mask
        g_mask = (maxc == g) & mask
        b_mask = (maxc == b) & mask

        h[r_mask] = ((g[r_mask] - b[r_mask]) / deltac[r_mask]) % 6
        h[g_mask] = ((b[g_mask] - r[g_mask]) / deltac[g_mask]) + 2
        h[b_mask] = ((r[b_mask] - g[b_mask]) / deltac[b_mask]) + 4
        h = h / 6.0

        s = torch.zeros_like(maxc)
        s[maxc != 0] = deltac[maxc != 0] / maxc[maxc != 0]
        v = maxc
        return torch.stack([h, s, v], dim=-1)

    def hsv_to_rgb_torch(self, hsv):
        """Convert HSV to RGB using PyTorch operations (NHWC)."""
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = h * 6.0

        c = v * s
        x = c * (1 - torch.abs((h % 2) - 1))
        m = v - c

        zeros = torch.zeros_like(h)
        rp = torch.where(
            (0 <= h) & (h < 1),
            c,
            torch.where(
                (1 <= h) & (h < 2),
                x,
                torch.where(
                    (2 <= h) & (h < 3),
                    zeros,
                    torch.where(
                        (3 <= h) & (h < 4), zeros, torch.where((4 <= h) & (h < 5), x, c)
                    ),
                ),
            ),
        )
        gp = torch.where(
            (0 <= h) & (h < 1),
            x,
            torch.where(
                (1 <= h) & (h < 2),
                c,
                torch.where(
                    (2 <= h) & (h < 3),
                    c,
                    torch.where(
                        (3 <= h) & (h < 4),
                        x,
                        torch.where((4 <= h) & (h < 5), zeros, zeros),
                    ),
                ),
            ),
        )
        bp = torch.where(
            (0 <= h) & (h < 1),
            zeros,
            torch.where(
                (1 <= h) & (h < 2),
                zeros,
                torch.where(
                    (2 <= h) & (h < 3),
                    x,
                    torch.where(
                        (3 <= h) & (h < 4), c, torch.where((4 <= h) & (h < 5), c, x)
                    ),
                ),
            ),
        )

        rgb = torch.stack([rp + m, gp + m, bp + m], dim=-1)
        return rgb

    def compensate_value_range(
        self, processed_images, original_min, original_max, original_mean
    ):
        """Only correct pathological shifts; don't squash healthy contrast (NHWC)."""
        x = torch.clamp(processed_images, 0.0, 1.0)

        cur_min = x.amin()
        cur_max = x.amax()
        cur_mean = x.mean()
        cur_range = cur_max - cur_min
        orig_range = original_max - original_min + 1e-6

        # If the range collapsed severely, gently expand around mean
        if cur_range < 0.25:
            scale = 0.25 / max(cur_range.item(), 1e-6)
            x = torch.clamp((x - cur_mean) * scale + cur_mean, 0.0, 1.0)
            return x

        # If mean drifted a lot, nudge halfway back
        if torch.abs(cur_mean - original_mean) > 0.10:
            x = torch.clamp(x + (original_mean - cur_mean) * 0.5, 0.0, 1.0)

        return x

    def apply_highlight_rolloff(self, images, mode):
        """
        Selective highlight roll-off (no global wash). Operates only on luminance > pivot.
        Modes:
          Clamped: return clamp(images).
          Soft: mild shoulder (slight compression of top ~20%).
          Filmic: stronger shoulder using filmic curve only in highlight band.
        """
        if mode == "Clamped":
            return torch.clamp(images, 0.0, 1.0)

        x = torch.clamp(images, 0.0, 1.0)

        # Compute luminance (ITU-R BT.709)
        luma = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
        pivot = 0.80  # start of highlight region
        eps = 1e-6

        mask = luma > pivot
        if not mask.any():
            return x

        # Normalized highlight fraction t in [0,1]
        t = (luma[mask] - pivot) / (1.0 - pivot + eps)

        if mode == "Soft":
            # Gentle compression: y = pivot + (1 - (1 - t)^k)*(1 - pivot)
            k = 1.35  # >1 => softer shoulder
            compressed = pivot + (1.0 - (1.0 - t) ** k) * (1.0 - pivot)
            strength = 0.55  # blend amount
        elif mode == "Filmic":
            # Filmic-like curve on t: f(t) = (t*(1+a))/(t+a)
            a = 0.55
            filmic_t = (t * (1 + a)) / (t + a + eps)
            compressed = pivot + filmic_t * (1.0 - pivot)
            strength = 0.85
        else:
            return x

        # Blend original vs compressed luminance
        new_luma = luma[mask] * (1 - strength) + compressed * strength

        # Avoid over-flatten: keep a minimum micro-contrast
        micro = luma[mask] - pivot
        new_luma = new_luma - 0.08 * (1 - t) * micro * (strength * 0.5)

        # Compute per-pixel scaling factor (keep chroma)
        scale = (new_luma / (luma[mask] + eps)).clamp(
            0.75 if mode == "Filmic" else 0.85, 1.04 if mode == "Filmic" else 1.02
        )

        # Apply scaling to RGB channels only where mask
        for c in range(3):
            chan = x[..., c]
            chan_masked = chan[mask] * scale
            x[..., c][mask] = chan_masked

        return torch.clamp(x, 0.0, 1.0)


class IllustriousVAEEncode:
    """VAE Encode optimized for Illustrious workflows."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE", {"tooltip": "Images to encode into latents."}),
                "vae": ("VAE", {"tooltip": "Variational Autoencoder used to encode images."}),
            },
            "optional": {
                # Mode: standard, inpaint (mask + optional reference), regional (pass mask downstream)
                "mode": (["standard", "inpaint", "regional"], {"default": "standard", "tooltip": "Encoding mode. Inpaint uses a mask and optional reference image; Regional forwards mask for downstream guidance."}),
                "illustrious_version": (
                    ["auto", "v0.1", "v1.0", "v1.1", "v2.0"],
                    {"default": "auto", "tooltip": "Auto-detect from image size; choose specific to nudge prep."},
                ),
                "optimization_mode": (
                    ["auto", "quality", "speed", "memory"],
                    {"default": "auto", "tooltip": "Adjusts tile size and memory strategy."},
                ),
                "enable_tiling": ("BOOLEAN", {"default": True, "tooltip": "Enable for large images to avoid VRAM issues."}),
                "tile_size": (
                    "INT",
                    {"default": 512, "min": 256, "max": 1024, "step": 64, "tooltip": "Encode tile size in image pixels."},
                ),
                "overlap": ("INT", {"default": 64, "min": 32, "max": 128, "step": 16, "tooltip": "Overlap between tiles to hide seams."}),
                "prepare_for_illustrious": ("BOOLEAN", {"default": True, "tooltip": "Apply minor tweaks before encode for Illustrious."}),
                "enhance_for_anime": ("BOOLEAN", {"default": True, "tooltip": "Light enhancement that suits anime/illustration."}),
                # Inpainting/regional inputs
                "mask": ("MASK", {"tooltip": "White = region to affect. Will be downscaled to latent size and used as noise mask."}),
                "reference_image": ("IMAGE", {"tooltip": "Optional reference image to fill masked area prior to sampling (inpaint mode)."}),
                "mask_blur": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1, "tooltip": "Feather mask edges (pixels)."}),
                "mask_dilate": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1, "tooltip": "Grow mask (pixels) before blur."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "encode_info")
    FUNCTION = "illustrious_encode"
    CATEGORY = "Easy Illustrious / VAE"

    def illustrious_encode(
        self,
        pixels,
        vae,
    mode="standard",
        illustrious_version="auto",
        optimization_mode="auto",
        enable_tiling=True,
        tile_size=512,
        overlap=64,
        prepare_for_illustrious=True,
        enhance_for_anime=True,
    mask=None,
    reference_image=None,
    mask_blur=4,
    mask_dilate=2,
    ):

        # Auto-detect version from image characteristics
        if illustrious_version == "auto":
            illustrious_version = self.detect_version_from_image(pixels)

        # Pre-process image for Illustrious if enabled
        if prepare_for_illustrious:
            pixels = self.prepare_image_for_illustrious(
                pixels, illustrious_version, enhance_for_anime
            )

        # Determine encoding strategy
        should_tile = self.should_use_tiling(pixels, enable_tiling, optimization_mode)

        if should_tile:
            latent = self.tiled_encode(
                pixels, vae, tile_size, overlap, optimization_mode
            )
            method = "tiled"
        else:
            latent = self.standard_encode(pixels, vae, optimization_mode)
            method = "standard"

        # Post-process latent for Illustrious compatibility
        if prepare_for_illustrious:
            latent = self.post_process_latent(latent, illustrious_version)

        # Optional inpaint/regional handling: prepare noise mask and optional reference composition
        try:
            if mode in ("inpaint", "regional") and mask is not None:
                mask_img = self._ensure_mask_image_space(mask, pixels)
                mask_img = self._refine_mask(mask_img, blur_px=int(mask_blur), dilate_px=int(mask_dilate)) if (mask_blur or mask_dilate) else mask_img
                # Downscale to latent resolution (B,1,H/8,W/8)
                b, ih, iw, _ = pixels.shape
                lh, lw = ih // 8, iw // 8
                mask_latent = torch.nn.functional.interpolate(
                    mask_img, size=(lh, lw), mode="bilinear", align_corners=False
                )
                mask_latent = torch.clamp(mask_latent, 0.0, 1.0)

                # Compose with reference image in masked area for inpaint mode (latent space)
                if mode == "inpaint" and reference_image is not None:
                    # Encode reference (match strategy used for main image)
                    if should_tile:
                        ref_latent = self.tiled_encode(reference_image, vae, tile_size, overlap, optimization_mode)
                    else:
                        ref_latent = self.standard_encode(reference_image, vae, optimization_mode)
                    ref_samples = ref_latent["samples"].to(latent["samples"].device)

                    # Broadcast mask to 4 channels for NCHW samples
                    mask_b = mask_latent.to(ref_samples.device)
                    if mask_b.dim() == 4 and mask_b.shape[1] != 1:
                        # If mask came as NHWC, transpose to NCHW 1-channel
                        # We expect mask_b as [B,1,H,W]
                        pass
                    # Ensure mask is [B,1,H,W]
                    if mask_b.shape[1] != 1:
                        # If shape is [B,H,W,1] -> permute
                        if mask_b.dim() == 4 and mask_b.shape[-1] == 1:
                            mask_b = mask_b.permute(0, 3, 1, 2).contiguous()
                        else:
                            mask_b = mask_b.mean(dim=1, keepdim=True) if mask_b.dim() == 4 else mask_b.unsqueeze(1)
                    # Compose in latent: base*(1-m)+ref*m
                    base = latent["samples"].to(ref_samples.device)
                    m4 = mask_b
                    if m4.shape[2:] != base.shape[2:]:
                        m4 = torch.nn.functional.interpolate(m4, size=base.shape[2:], mode="bilinear", align_corners=False)
                    m4c = m4.expand(-1, base.shape[1], -1, -1)
                    composed = base * (1.0 - m4c) + ref_samples * m4c
                    latent["samples"] = composed

                # Attach noise_mask to latent dict for downstream samplers (common_ksampler-compatible)
                # Ensure mask is NCHW [B,1,H,W]
                noise_mask = mask_latent
                if noise_mask.dim() == 4 and noise_mask.shape[-1] == 1:
                    noise_mask = noise_mask.permute(0, 3, 1, 2).contiguous()
                if noise_mask.dim() == 3:
                    noise_mask = noise_mask.unsqueeze(1)
                latent["noise_mask"] = torch.clamp(noise_mask, 0.0, 1.0)
        except Exception as e:
            print(f"[IllustriousVAEEncode] Inpaint/Regional mask handling failed: {e}")

        # Generate encode info
        encode_info = self.generate_encode_info(
            method, pixels.shape, illustrious_version
        )
        if mode in ("inpaint", "regional") and mask is not None:
            encode_info += f"Mode: {mode} (mask applied)\n"
        elif mode in ("inpaint", "regional"):
            encode_info += f"Mode: {mode} (no mask provided)\n"

        return (latent, encode_info)

    # ----- Mask helpers -----
    def _ensure_mask_image_space(self, mask, pixels):
        """Return mask as image-space float tensor [B,1,H,W] matching pixels batch/size.
        Accepts MASK tensors from ComfyUI in common shapes.
        """
        # Pixels: [B,H,W,C]
        b, h, w, c = pixels.shape
        device = pixels.device if hasattr(pixels, 'device') else None
        # Convert mask to torch tensor on same device
        m = mask
        try:
            if not torch.is_tensor(m):
                m = torch.tensor(m)
        except Exception:
            pass
        if device is not None and hasattr(m, 'to'):
            m = m.to(device=device, dtype=torch.float32)
        else:
            m = m.float() if hasattr(m, 'float') else m

        # Normalize shape to [B,1,H,W]
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:
            # Could be [H,W,1] or [1,H,W]
            if m.shape[-1] == 1:
                m = m.permute(2, 0, 1).unsqueeze(0)  # -> [1,1,H,W]
            else:
                m = m.unsqueeze(0)  # assume [1,H,W]
                if m.shape[1] != 1:
                    m = m[:, :1, :, :]
        elif m.dim() == 4:
            # Could be [B,H,W,1] or [B,1,H,W]
            if m.shape[-1] == 1:
                m = m.permute(0, 3, 1, 2).contiguous()
            elif m.shape[1] != 1:
                m = m[:, :1, :, :]

        # Match batch/size
        if m.shape[0] != b:
            if m.shape[0] == 1:
                m = m.expand(b, -1, -1, -1).clone()
            else:
                m = m[:b]
        if m.shape[2] != h or m.shape[3] != w:
            m = torch.nn.functional.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)

        # Clamp 0..1
        m = torch.clamp(m, 0.0, 1.0)
        return m

    def _refine_mask(self, mask_bchw, blur_px=0, dilate_px=0):
        """Simple morphological refine on [B,1,H,W] masks using max/avg pooling approximations."""
        m = mask_bchw
        if dilate_px and dilate_px > 0:
            k = max(1, int(dilate_px))
            # Use max pool as dilation
            pad = k // 2
            m = torch.nn.functional.max_pool2d(m, kernel_size=k, stride=1, padding=pad)
        if blur_px and blur_px > 0:
            k = max(1, int(blur_px) | 1)  # make odd
            pad = k // 2
            weight = torch.ones((1, 1, k, k), device=m.device, dtype=m.dtype) / float(k * k)
            m = torch.nn.functional.conv2d(m, weight, padding=pad)
        return torch.clamp(m, 0.0, 1.0)

    def detect_version_from_image(self, pixels):
        """Auto-detect optimal Illustrious version from image characteristics."""
        batch_size, h, w, channels = pixels.shape
        total_pixels = h * w

        if total_pixels >= 2048 * 2048:
            return "v2.0"
        elif total_pixels >= 1536 * 1536:
            return "v1.1"
        elif total_pixels >= 1024 * 1024:
            return "v1.0"
        else:
            return "v0.1"

    def prepare_image_for_illustrious(self, pixels, version, enhance_for_anime):
        """Pre-process image for optimal Illustrious encoding."""
        processed = pixels.clone()
        if enhance_for_anime:
            processed = self.enhance_anime_characteristics(processed)
        if version == "v0.1":
            processed = torch.clamp(processed * 1.05, 0, 1)
        elif version in ["v1.1", "v2.0"]:
            processed = self.expand_dynamic_range(processed)
        return processed

    def enhance_anime_characteristics(self, pixels):
        """Enhance image characteristics that work well with anime models."""
        enhanced = torch.clamp(pixels * 1.02, 0, 1)
        return enhanced

    def expand_dynamic_range(self, pixels):
        """Expand dynamic range for newer Illustrious versions."""
        s_curve = 0.5 * (torch.sin((pixels - 0.5) * math.pi) + 1)
        adjusted = 0.8 * pixels + 0.2 * s_curve
        return torch.clamp(adjusted, 0, 1)

    def should_use_tiling(self, pixels, enable_tiling, optimization_mode):
        """Determine if tiling should be used."""
        batch_size, h, w, channels = pixels.shape
        total_pixels = h * w * batch_size

        if optimization_mode == "memory":
            return total_pixels > 1024 * 1024  # 1MP threshold
        elif optimization_mode == "speed":
            return total_pixels > 2048 * 2048
        else:
            return enable_tiling and total_pixels > 1536 * 1536

    def tiled_encode(self, pixels, vae, tile_size, overlap, optimization_mode):
        """Encode image using tiling strategy."""
        batch_size, h, w, channels = pixels.shape
        tiles = self.calculate_encode_tile_positions(h, w, tile_size, overlap)
        encoded_tiles = []

        for i, (y_start, y_end, x_start, x_end) in enumerate(tiles):
            tile_pixels = pixels[:, y_start:y_end, x_start:x_end, :]
            tile_latent = vae.encode(tile_pixels)  # NCHW latent
            comfy.model_management.throw_exception_if_processing_interrupted()
            encoded_tiles.append(
                {
                    "latent": tile_latent,
                    "position": (y_start // 8, y_end // 8, x_start // 8, x_end // 8),
                }
            )

        combined_latent = self.combine_encoded_tiles(encoded_tiles, h // 8, w // 8)
        return {"samples": combined_latent}

    def standard_encode(self, pixels, vae, optimization_mode):
        """Standard VAE encode with memory management."""
        if optimization_mode == "memory":
            batch_size = pixels.shape[0]
            if batch_size > 1:
                encoded_batches = []
                for i in range(batch_size):
                    single_pixel = pixels[i : i + 1]
                    encoded = vae.encode(single_pixel)
                    encoded_batches.append(encoded)
                    comfy.model_management.throw_exception_if_processing_interrupted()
                combined = torch.cat(encoded_batches, dim=0)
            else:
                combined = vae.encode(pixels)
                comfy.model_management.throw_exception_if_processing_interrupted()
        else:
            combined = vae.encode(pixels)
            comfy.model_management.throw_exception_if_processing_interrupted()
        return {"samples": combined}

    def calculate_encode_tile_positions(self, h, w, tile_size, overlap):
        """Calculate tile positions for encoding (image space)."""
        tiles = []
        y = 0
        while y < h:
            y_end = min(y + tile_size, h)
            x = 0
            while x < w:
                x_end = min(x + tile_size, w)
                tiles.append((y, y_end, x, x_end))
                if x_end >= w:
                    break
                x += tile_size - overlap
            if y_end >= h:
                break
            y += tile_size - overlap
        return tiles

    def combine_encoded_tiles(self, encoded_tiles, final_h, final_w):
        """Combine encoded tiles into single latent (NCHW)."""
        first_tile = encoded_tiles[0]["latent"]
        batch_size, channels = first_tile.shape[0], first_tile.shape[1]
        combined = torch.zeros(
            (batch_size, channels, final_h, final_w),
            dtype=first_tile.dtype,
            device=first_tile.device,
        )
        for tile_data in encoded_tiles:
            tile_latent = tile_data["latent"]
            y_start, y_end, x_start, x_end = tile_data["position"]
            combined[:, :, y_start:y_end, x_start:x_end] = tile_latent
        return combined

    def post_process_latent(self, latent, version):
        """Post-process latent for Illustrious compatibility (NCHW)."""
        samples = latent["samples"]
        if version == "v0.1":
            samples = samples * 0.98
        elif version == "v2.0":
            samples = samples * 1.01
        return {"samples": samples}

    def generate_encode_info(self, method, pixel_shape, version):
        """Generate encoding information."""
        batch_size, h, w, channels = pixel_shape
        info = f"🎨 Illustrious VAE Encode Report\n"
        info += "=" * 40 + "\n"
        info += f"Version: {version}\n"
        info += f"Method: {method}\n"
        info += f"Input Size: {w}x{h} ({channels} channels)\n"
        info += f"Output Size: {w//8}x{h//8} (latent)\n"
        info += f"Batch Size: {batch_size}\n"
        return info
