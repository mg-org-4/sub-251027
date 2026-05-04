"""
Server routes for Illustrious Color Corrector
Provides API endpoints for real-time analysis and correction
"""

import json
import base64
import io
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
from server import PromptServer
from aiohttp import web

# Import the main corrector class
from .color_corrector import IllustriousColorCorrector


class IllustriousColorCorrectorServer:
    """Server component for Illustrious color correction with real-time preview"""

    def __init__(self):
        self.corrector = IllustriousColorCorrector()

        # Create presets directory if it doesn't exist
        self.presets_dir = (
            Path(__file__).parent.parent.parent / "presets" / "color_corrector"
        )
        self.presets_dir.mkdir(parents=True, exist_ok=True)

        # Illustrious routes
        PromptServer.instance.routes.post("/illustrious/color_corrector/analyze")(
            self.analyze_image
        )
        PromptServer.instance.routes.post("/illustrious/color_corrector/correct")(
            self.correct_image
        )
        PromptServer.instance.routes.get("/illustrious/color_corrector/presets")(
            self.get_presets
        )
        PromptServer.instance.routes.post("/illustrious/color_corrector/save_preset")(
            self.save_preset
        )
        PromptServer.instance.routes.delete("/illustrious/color_corrector/preset")(
            self.delete_preset
        )
        PromptServer.instance.routes.get("/illustrious/color_corrector/load_preset")(
            self.load_preset
        )

        print("🎨 Illustrious Color Corrector server routes registered")

    async def analyze_image(self, request):
        """Analyze image for color correction needs"""
        try:
            data = await request.json()
            image_data = data.get("image_data")
            node_id = data.get("node_id")
            model_version = data.get("model_version", "auto")

            if not image_data:
                return web.json_response(
                    {"error": "No image data provided"}, status=400
                )

            # Decode image
            image_bytes = base64.b64decode(image_data)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Auto-detect version if needed
            if model_version == "auto":
                detected_version = self.corrector.detect_model_version(pil_image)
            else:
                detected_version = model_version

            # Analyze image characteristics
            analysis = self.corrector.analyze_image_characteristics(
                pil_image, detected_version
            )

            analysis_result = {
                "detected_version": detected_version,
                "avg_saturation": analysis["avg_saturation"],
                "max_saturation": analysis["max_saturation"],
                "avg_brightness": analysis["avg_brightness"],
                "contrast": analysis["contrast"],
                "color_cast": analysis["color_cast"],
                "highlight_clipping": analysis["highlight_clipping"],
                "shadow_clipping": analysis["shadow_clipping"],
                "illustration_style": analysis["illustration_style"],
                "is_character_focused": analysis["is_character_focused"],
                "has_detailed_background": analysis["has_detailed_background"],
                "anime_score": analysis.get("anime_score", 0),
                "oversaturation_score": analysis.get("oversaturation_score", 0),
                "color_bias": analysis.get("color_bias", {}),
                "issues_detected": self.detect_issues_list(analysis),
                "suggested_preset": self.suggest_preset(analysis),
                "recommended_strength": self.recommend_strength(
                    analysis, detected_version
                ),
            }

            # Emit preview of the original image for the frontend to latch onto
            try:
                if hasattr(PromptServer, "instance") and node_id is not None:
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    PromptServer.instance.send_sync(
                        "illustrious.color_corrector_preview",
                        {
                            "node_id": node_id,
                            "image_data": b64,
                        },
                    )
            except Exception as e:
                print(f"[IllustriousColorCorrector] Analyze preview emit error: {e}")

            return web.json_response(analysis_result)

        except Exception as e:
            print(f"[IllustriousColorCorrector] Analysis error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def correct_image(self, request):
        """Apply color correction to image with given settings"""
        try:
            data = await request.json()
            image_data = data.get("image_data")
            settings = data.get("settings", {})
            node_id = data.get("node_id")

            if not image_data:
                return web.json_response(
                    {"error": "No image data provided"}, status=400
                )

            # Decode image
            image_bytes = base64.b64decode(image_data)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Convert to tensor format expected by the corrector
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)

            # Map new/legacy version field
            version = settings.get("model_version", "auto")

            # Apply correction
            corrected_batch, report = self.corrector.auto_correct_colors(
                image_tensor,
                model_version=version,
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

            # Convert result back to base64
            corrected_tensor = corrected_batch[0]
            corrected_np = (corrected_tensor.cpu().numpy() * 255).astype(np.uint8)
            corrected_pil = Image.fromarray(corrected_np)

            buffer = io.BytesIO()
            corrected_pil.save(buffer, format="PNG")
            corrected_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            corrections_applied = self.parse_corrections_from_report(report)

            # Emit original preview so UI has a source even before corrected loads
            try:
                if hasattr(PromptServer, "instance") and node_id is not None:
                    buf2 = io.BytesIO()
                    pil_image.save(buf2, format="PNG")
                    b64src = base64.b64encode(buf2.getvalue()).decode("utf-8")
                    PromptServer.instance.send_sync(
                        "illustrious.color_corrector_preview",
                        {
                            "node_id": node_id,
                            "image_data": b64src,
                        },
                    )
            except Exception as e:
                print(f"[IllustriousColorCorrector] Correct preview emit error: {e}")

            return web.json_response(
                {
                    "corrected_image": corrected_b64,
                    "corrections_applied": corrections_applied,
                    "report": report,
                    "model_version": version,
                }
            )

        except Exception as e:
            print(f"[IllustriousColorCorrector] Correction error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def get_presets(self, request):
        """Get available correction presets"""
        try:
            presets = {
                "Character Portrait (Anime)": {
                    "description": "Optimized for character-focused illustrations with anime enhancement",
                    "settings": {
                        "custom_preset": "character_portrait",
                        "correction_strength": 1.0,
                        "preserve_anime_aesthetic": True,
                        "enhance_details": True,
                        "fix_oversaturation": True,
                    },
                },
                "Detailed Scene": {
                    "description": "Balanced correction for complex illustrated scenes",
                    "settings": {
                        "custom_preset": "detailed_scene",
                        "correction_strength": 1.1,
                        "preserve_anime_aesthetic": True,
                        "enhance_details": False,
                        "balance_colors": True,
                    },
                },
                "Soft Illustration": {
                    "description": "Gentle correction preserving soft anime aesthetics",
                    "settings": {
                        "custom_preset": "soft_illustration",
                        "correction_strength": 0.8,
                        "preserve_anime_aesthetic": True,
                        "enhance_details": False,
                        "fix_oversaturation": False,
                    },
                },
                "Vibrant Anime": {
                    "description": "Enhanced colors for vibrant anime style outputs",
                    "settings": {
                        "custom_preset": "vibrant_anime",
                        "correction_strength": 1.3,
                        "preserve_anime_aesthetic": True,
                        "fix_oversaturation": False,
                        "enhance_details": True,
                    },
                },
                "Natural Colors": {
                    "description": "Realistic color grading for semi-realistic outputs",
                    "settings": {
                        "custom_preset": "natural_colors",
                        "correction_strength": 1.1,
                        "preserve_anime_aesthetic": False,
                        "enhance_details": True,
                        "balance_colors": True,
                    },
                },
                "Legacy v0.5 Fix": {
                    "description": "Specialized correction for legacy v0.5 oversaturation issues",
                    "settings": {
                        "model_version": "v0.5",
                        "correction_strength": 1.4,
                        "fix_oversaturation": True,
                        "preserve_anime_aesthetic": True,
                        "adjust_contrast": True,
                    },
                },
            }

            return web.json_response({"presets": presets})

        except Exception as e:
            print(f"[IllustriousColorCorrector] Get presets error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def save_preset(self, request):
        """Save a custom correction preset to filesystem"""
        try:
            data = await request.json()
            preset_name = data.get("preset_name")
            preset_data = data.get("preset_data")

            if not preset_name or not preset_data:
                return web.json_response(
                    {"error": "Missing preset name or data"}, status=400
                )

            safe_name = "".join(
                c for c in preset_name if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()
            if not safe_name:
                return web.json_response({"error": "Invalid preset name"}, status=400)

            preset_file = self.presets_dir / f"{safe_name}.json"

            # Add metadata (use os.path for timestamps)
            if preset_file.exists():
                import os, time

                preset_data["created_at"] = preset_data.get("created_at") or str(
                    int(os.path.getctime(preset_file))
                )
                preset_data["modified_at"] = str(int(os.path.getmtime(preset_file)))
            else:
                import time

                preset_data["created_at"] = preset_data.get("created_at") or str(
                    int(time.time())
                )
                preset_data["modified_at"] = preset_data["created_at"]

            with open(preset_file, "w") as f:
                json.dump(preset_data, f, indent=2)

            print(
                f"[IllustriousColorCorrector] Saved preset '{preset_name}' to {preset_file}"
            )

            return web.json_response(
                {
                    "success": True,
                    "message": f"Preset '{preset_name}' saved successfully",
                    "filename": f"{safe_name}.json",
                }
            )

        except Exception as e:
            print(f"[IllustriousColorCorrector] Save preset error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    def detect_issues_list(self, analysis):
        """Convert analysis data to list of detected issues"""
        issues = []

        # Oversaturation detection (more sensitive)
        if analysis["avg_saturation"] > 0.65:
            issues.append("Oversaturation detected")
        oversaturation_score = analysis.get("oversaturation_score", 0)
        if oversaturation_score > 0.15:
            issues.append("High oversaturation areas")

        if analysis["max_saturation"] > 0.85:
            issues.append("Extreme saturation peaks")

        if analysis["highlight_clipping"] > 0.015:
            issues.append("Highlight clipping")

        if analysis["shadow_clipping"] > 0.015:
            issues.append("Shadow clipping")

        # Color bias detection
        bias = analysis.get("color_bias", {})
        if bias.get("overall_bias_strength", 0) > 0.08:
            issues.append("Color bias detected")

        # General color cast issues
        color_cast = analysis["color_cast"]
        for color, b in color_cast.items():
            if abs(b) > 0.025:
                issues.append(f"{color.title()} color cast")

        # Contrast issues
        if analysis["contrast"] < 0.18:
            issues.append("Low contrast (anime typical)")
        elif analysis["contrast"] > 0.38:
            issues.append("High contrast")

        # Brightness issues
        if analysis["avg_brightness"] < 0.25:
            issues.append("Image too dark")
        elif analysis["avg_brightness"] > 0.85:
            issues.append("Image too bright")

        # Anime-specific
        anime_score = analysis.get("anime_score", 0)
        if anime_score > 0.8 and analysis["avg_saturation"] > 0.7:
            issues.append("Anime oversaturation")

        return issues

    def suggest_preset(self, analysis):
        """Suggest appropriate preset based on analysis"""
        anime_score = analysis.get("anime_score", 0)
        oversaturation_score = analysis.get("oversaturation_score", 0)

        if oversaturation_score > 0.2:
            return "character_portrait"
        elif analysis["is_character_focused"] and anime_score > 0.7:
            return "character_portrait"
        elif analysis["has_detailed_background"] and anime_score > 0.5:
            return "detailed_scene"
        elif analysis["avg_saturation"] > 0.7 and anime_score > 0.8:
            return "vibrant_anime"
        elif anime_score < 0.4:
            return "natural_colors"
        else:
            return "soft_illustration"

    def recommend_strength(self, analysis, version):
        """Recommend correction strength based on detected issues and version"""
        issue_count = len(self.detect_issues_list(analysis))
        base_strength = 0.8

        if version == "v0.5":
            base_strength = 1.1
        elif version == "v0.75":
            base_strength = 0.9
        elif version == "v1.0":
            base_strength = 0.7

        oversaturation_score = analysis.get("oversaturation_score", 0)
        if oversaturation_score > 0.25:
            base_strength += 0.3
        elif oversaturation_score > 0.15:
            base_strength += 0.2

        if issue_count == 0:
            return max(0.4, base_strength - 0.3)
        elif issue_count <= 2:
            return base_strength
        elif issue_count <= 4:
            return min(1.8, base_strength + 0.2)
        else:
            return min(2.0, base_strength + 0.4)

    def parse_corrections_from_report(self, report):
        """Extract applied corrections from Illustrious report text"""
        corrections = []
        lines = report.split("\n")

        in_corrections_section = False
        for line in lines:
            if "Corrections Applied:" in line:
                in_corrections_section = True
                continue
            elif in_corrections_section and line.strip().startswith("•"):
                # Extract correction from bullet point
                correction = line.strip()[1:].strip()
                corrections.append(correction)
            elif in_corrections_section and (
                line.strip().startswith(
                    ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")
                )
            ):
                # Extract numbered corrections
                correction = line.strip()[2:].strip()  # Remove number and space
                corrections.append(correction)
            elif in_corrections_section and line.strip() and not line.startswith(" "):
                # End of corrections section
                break

        return corrections

    async def delete_preset(self, request):
        """Delete a saved preset"""
        try:
            preset_name = request.query.get("name")
            if not preset_name:
                return web.json_response({"error": "Missing preset name"}, status=400)

            # Sanitize filename
            safe_name = "".join(
                c for c in preset_name if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()
            preset_file = self.presets_dir / f"{safe_name}.json"

            if not preset_file.exists():
                return web.json_response(
                    {"error": f"Preset '{preset_name}' not found"}, status=404
                )

            preset_file.unlink()
            print(f"[IllustriousColorCorrector] Deleted preset '{preset_name}'")

            return web.json_response(
                {
                    "success": True,
                    "message": f"Preset '{preset_name}' deleted successfully",
                }
            )

        except Exception as e:
            print(f"[IllustriousColorCorrector] Delete preset error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def load_preset(self, request):
        """Load a saved preset from filesystem"""
        try:
            preset_name = request.query.get("name")
            if not preset_name:
                return web.json_response({"error": "Missing preset name"}, status=400)

            # Sanitize filename
            safe_name = "".join(
                c for c in preset_name if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()
            preset_file = self.presets_dir / f"{safe_name}.json"

            if not preset_file.exists():
                # Try loading from custom presets list
                custom_presets = await self.get_custom_presets()
                for preset in custom_presets:
                    if preset["name"] == preset_name:
                        return web.json_response({"success": True, "preset": preset})

                return web.json_response(
                    {"error": f"Preset '{preset_name}' not found"}, status=404
                )

            with open(preset_file, "r") as f:
                preset_data = json.load(f)

            return web.json_response({"success": True, "preset": preset_data})

        except Exception as e:
            print(f"[IllustriousColorCorrector] Load preset error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def get_custom_presets(self):
        """Get list of custom saved presets"""
        custom_presets = []

        try:
            for preset_file in self.presets_dir.glob("*.json"):
                try:
                    with open(preset_file, "r") as f:
                        preset_data = json.load(f)
                        custom_presets.append(
                            {
                                "name": preset_data.get("name", preset_file.stem),
                                "filename": preset_file.name,
                                "custom": True,
                                "settings": preset_data,
                            }
                        )
                except Exception as e:
                    print(
                        f"[IllustriousColorCorrector] Error loading preset {preset_file}: {e}"
                    )
                    continue
        except Exception as e:
            print(f"[IllustriousColorCorrector] Error reading presets directory: {e}")

        return custom_presets


# Initialize the server
illustrious_corrector_server = IllustriousColorCorrectorServer()


# Enhanced node with server integration
class IllustriousColorCorrectorWithPreview(IllustriousColorCorrector):
    """Illustrious version with real-time preview support"""

    # Expose auto mode + report so it can be used as a standalone node
    FUNCTION = "auto_correct_colors"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("corrected_images", "report")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(s):
        base = super().INPUT_TYPES()
        # Replace simple optional set with full interactive set expected by JS
        base_opt = base["optional"]
        # Keep existing simple controls
        # Add advanced / auto controls (names must match frontend)
        advanced = {
            "model_version": (
                ["auto", "v0.5", "v0.75", "v1.0", "v1.1", "v2.0", "v3.x"],
                {
                    "default": "auto",
                    "tooltip": "Model version for tailored corrections (auto uses heuristics).",
                },
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
                {"default": True, "tooltip": "Analyze image to enable relevant fixes automatically."},
            ),  # renamed from auto_analyze
            "fix_oversaturation": ("BOOLEAN", {"default": True, "tooltip": "Reduce excessive saturation safely."}),
            "preserve_anime_aesthetic": ("BOOLEAN", {"default": True, "tooltip": "Favor anime-friendly look while correcting."}),
            "enhance_details": ("BOOLEAN", {"default": True, "tooltip": "Subtle detail sharpening where safe."}),
            "balance_colors": ("BOOLEAN", {"default": True, "tooltip": "Neutralize color casts and balance channels."}),
            "adjust_contrast": ("BOOLEAN", {"default": True, "tooltip": "Tweak brightness/contrast gently."}),
            "custom_preset": (
                [
                    "none",
                    "character_portrait",
                    "detailed_scene",
                    "soft_illustration",
                    "vibrant_anime",
                    "natural_colors",
                ],
                {"default": "none", "tooltip": "Apply a preset recipe (overrides specific toggles)."},
            ),
            "show_corrections": ("BOOLEAN", {"default": False, "tooltip": "Include bullet list of applied fixes in report."}),
            # preview toggles
            "enable_preview": ("BOOLEAN", {"default": True, "tooltip": "Emit preview event for the UI."}),
        }
        # Remove legacy auto_analyze if present
        base_opt.pop("auto_analyze", None)
        # Merge
        base_opt.update(advanced)
        return base

    def auto_correct_colors(self, images, **kwargs):
        """
        Wrapper that maps widget names to base implementation and emits preview.
        """
        enable_preview = kwargs.pop("enable_preview", True)
        # Map model_version -> illustrious_version internal param
        model_version = kwargs.pop("model_version", "auto")
        # JS uses auto_detect_issues
        auto_detect_issues = kwargs.pop("auto_detect_issues", True)

        result = super().auto_correct_colors(
            images,
            model_version=model_version,
            auto_detect_issues=auto_detect_issues,
            **kwargs,
        )

        if enable_preview and hasattr(PromptServer, "instance"):
            try:
                first = images[0]
                np_img = first.cpu().numpy()
                if np_img.max() <= 1.0:
                    np_img = (np_img * 255).astype(np.uint8)
                else:
                    np_img = np_img.astype(np.uint8)
                pil = Image.fromarray(np_img)
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                PromptServer.instance.send_sync(
                    "illustrious.color_corrector_preview",
                    {"node_id": getattr(self, "_node_id", None), "image_data": b64},
                )
                # Illustrious-only event
            except Exception as e:
                print(f"[IllustriousColorCorrector] Preview error: {e}")

        return result


print("🎨 Illustrious Color Corrector server integration loaded")
