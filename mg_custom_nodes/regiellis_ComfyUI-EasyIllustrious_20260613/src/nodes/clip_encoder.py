import torch
import re
import comfy.sd
import comfy.clip_model
import comfy.model_management
from transformers import CLIPTokenizer
import json
from pathlib import Path


class IllustriousCLIPTextEncoder:
    """Custom CLIP Text Encoder optimized for Illustrious models with Danbooru tag handling"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP text encoder from your base model."}),
                "text": (
                    "STRING",
                    {"multiline": True, "default": "masterpiece, best quality, 1girl", "tooltip": "Prompt text (comma-separated tags work well)."},
                ),
            },
            "optional": {
                "illustrious_version": (
                    ["auto", "v0.1", "v1.0", "v1.1", "v2.0"],
                    {"default": "auto", "tooltip": "Used for tag optimization hints; auto is fine."},
                ),
                "enable_tag_optimization": ("BOOLEAN", {"default": True, "tooltip": "Fix common tag variants and add helpful tags."}),
                "enable_padding_fix": ("BOOLEAN", {"default": True, "tooltip": "Trim excessive padding tokens to improve conditioning."}),
                "auto_tag_ordering": ("BOOLEAN", {"default": True, "tooltip": "Reorder tags for Illustrious-friendly hierarchy."}),
                "danbooru_weighting": ("BOOLEAN", {"default": True, "tooltip": "Apply frequency-based weights to common tags."}),
                "clip_skip": ("INT", {"default": 2, "min": 1, "max": 12, "tooltip": "Number of CLIP layers to skip from the end."}),
                "quality_boost": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "tooltip": "Multiplier for quality tags (masterpiece, best_quality)."},
                ),
                "character_boost": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "tooltip": "Multiplier for character tags (1girl, long_hair)."},
                ),
                "composition_boost": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "tooltip": "Multiplier for composition tags (portrait, close-up)."},
                ),
                "enable_hybrid_mode": ("BOOLEAN", {"default": False, "tooltip": "Convert some phrases to tags for v1.1+ models."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "processed_prompt")
    FUNCTION = "encode_illustrious"
    CATEGORY = "Easy Illustrious / Encoder"

    def __init__(self):
        self.danbooru_categories = {
            "quality": [
                "masterpiece",
                "best quality",
                "amazing quality",
                "very aesthetic",
                "newest",
                "absurdres",
                "highres",
                "ultra-detailed",
                "extremely detailed",
                "perfect quality",
                "absolutely eye-catching",
                "high_quality",
                "best_quality",
            ],
            "rating": ["safe", "sensitive", "questionable", "explicit", "general"],
            "character_count": [
                "solo",
                "1girl",
                "1boy",
                "2girls",
                "2boys",
                "3girls",
                "3boys",
                "multiple girls",
                "multiple boys",
                "crowd",
            ],
            "composition": [
                "portrait",
                "upper body",
                "cowboy shot",
                "full body",
                "close-up",
                "wide shot",
                "from above",
                "from below",
                "from side",
                "looking at viewer",
                "looking away",
            ],
            "character_features": [
                "long hair",
                "short hair",
                "blue eyes",
                "brown eyes",
                "blonde hair",
                "black hair",
                "large breasts",
                "small breasts",
                "school uniform",
            ],
            "background": [
                "simple background",
                "white background",
                "detailed background",
                "outdoors",
                "indoors",
                "scenery",
                "nature",
                "city",
            ],
            "lighting": [
                "soft lighting",
                "dramatic lighting",
                "rim lighting",
                "backlighting",
                "cinematic lighting",
                "natural lighting",
            ],
            "artist_style": [
                "official art",
                "concept art",
                "promotional art",
                "anime coloring",
                "cel shading",
                "traditional media",
                "digital art",
                "anime_style",
            ],
            "illustrious_specialty": [
                "anime",
                "illustration",
                "detailed_eyes",
                "clean_lineart",
                "vibrant_colors",
                "anime_style",
                "official_art",
                "high_resolution",
            ],
            "negative_quality": [
                "worst quality",
                "bad quality",
                "low quality",
                "bad anatomy",
                "bad hands",
                "extra digits",
                "jpeg artifacts",
                "watermark",
                "signature",
                "username",
                "lowres",
                "sketch",
                "oldest",
                "early",
                "worst_quality",
                "bad_quality",
                "poorly_detailed",
            ],
        }

        # Load Danbooru tag database if available
        self.tag_weights = self.load_tag_weights()

    def load_tag_weights(self):
        """Load tag weights based on Illustrious training and Danbooru frequency"""
        # Optimized weights for Illustrious models
        return {
            "masterpiece": 1.3,
            "best_quality": 1.2,
            "high_quality": 1.2,
            "amazing_quality": 1.15,
            "very_aesthetic": 1.1,
            "anime_style": 1.15,
            "official_art": 1.1,
            "illustration": 1.1,
            "detailed_eyes": 1.05,
            "clean_lineart": 1.1,
            "1girl": 1.0,
            "1boy": 1.0,
            "looking_at_viewer": 0.95,
            "detailed_background": 0.85,
            "vibrant_colors": 1.08,
            "high_resolution": 1.05,
        }

    def encode_illustrious(
        self,
        clip,
        text,
        illustrious_version="auto",
        enable_tag_optimization=True,
        enable_padding_fix=True,
        auto_tag_ordering=True,
        danbooru_weighting=True,
        clip_skip=2,
        quality_boost=1.0,
        character_boost=1.0,
        composition_boost=1.0,
        enable_hybrid_mode=False,
    ):

        # Process the input text
        processed_text = text
        processing_log = []

        if enable_tag_optimization:
            processed_text, tag_log = self.optimize_tags(
                processed_text, illustrious_version
            )
            processing_log.extend(tag_log)

        if auto_tag_ordering:
            processed_text, order_log = self.reorder_tags(processed_text)
            processing_log.extend(order_log)

        if danbooru_weighting:
            processed_text, weight_log = self.apply_danbooru_weights(
                processed_text, quality_boost, character_boost, composition_boost
            )
            processing_log.extend(weight_log)

        if enable_hybrid_mode:
            processed_text, hybrid_log = self.enable_hybrid_processing(
                processed_text, illustrious_version
            )
            processing_log.extend(hybrid_log)

        # Apply padding fix if enabled
        if enable_padding_fix:
            tokens = clip.tokenize(processed_text)
            tokens, padding_log = self.fix_padding_tokens(tokens)
            processing_log.extend(padding_log)

            # Encode with fixed tokens and clip skip
            cond = self.encode_with_clip_skip(clip, tokens, clip_skip)
        else:
            # Standard encoding with clip skip
            cond = self.encode_with_clip_skip_text(clip, processed_text, clip_skip)

        # Generate processing report
        report = self.generate_processing_report(
            text, processed_text, processing_log, illustrious_version
        )

        return (cond, report)

    def optimize_tags(self, text, version):
        """Optimize tags for Illustrious version-specific behavior"""
        log = []

        # Handle both underscore and space formats (Illustrious prefers underscores for consistency)
        # Convert common variations to Illustrious preferred format
        illustrious_fixes = {
            "best quality": "best_quality",
            "high quality": "high_quality",
            "amazing quality": "amazing_quality",
            "very aesthetic": "very_aesthetic",
            "worst quality": "worst_quality",
            "bad quality": "bad_quality",
            "low quality": "low_quality",
            "long hair": "long_hair",
            "short hair": "short_hair",
            "blue eyes": "blue_eyes",
            "brown eyes": "brown_eyes",
            "looking at viewer": "looking_at_viewer",
            "school uniform": "school_uniform",
            "large breasts": "large_breasts",
            "small breasts": "small_breasts",
            "detailed eyes": "detailed_eyes",
            "anime style": "anime_style",
            "vibrant colors": "vibrant_colors",
            "clean lineart": "clean_lineart",
        }

        for wrong, correct in illustrious_fixes.items():
            if wrong in text:
                text = text.replace(wrong, correct)
                log.append(f"Illustrious format fix: '{wrong}' → '{correct}'")

        # Version-specific optimizations
        if version == "v0.1":
            # v0.1 needs more explicit quality tags
            if "masterpiece" not in text:
                text = "masterpiece, " + text
                log.append("Added 'masterpiece' for v0.1 compatibility")

        elif version in ["v1.0", "v1.1"]:
            # Newer versions benefit from aesthetic tags
            if "very_aesthetic" not in text and "best_quality" not in text:
                text = "very_aesthetic, " + text
                log.append("Added 'very_aesthetic' for newer Illustrious version")

            # Add anime_style for better results
            if "anime" not in text and "anime_style" not in text:
                text = text + ", anime_style"
                log.append("Added 'anime_style' for Illustrious specialty")

        elif version == "v2.0":
            # v2.0 benefits from high resolution tags
            if "high_resolution" not in text:
                text = text + ", high_resolution"
                log.append("Added 'high_resolution' for v2.0 capability")

        return text, log

    def reorder_tags(self, text):
        """Reorder tags according to optimal Illustrious hierarchy"""
        log = []
        tags = [tag.strip() for tag in text.split(",")]

        # Categorize tags
        categorized = {category: [] for category in self.danbooru_categories}
        uncategorized = []

        for tag in tags:
            assigned = False
            for category, category_tags in self.danbooru_categories.items():
                if tag in category_tags or any(
                    cat_tag in tag for cat_tag in category_tags
                ):
                    categorized[category].append(tag)
                    assigned = True
                    break
            if not assigned:
                uncategorized.append(tag)

        # Optimal order for Illustrious (prioritizes quality and anime style)
        optimal_order = [
            "quality",
            "illustrious_specialty",
            "rating",
            "character_count",
            "character_features",
            "composition",
            "background",
            "lighting",
            "artist_style",
        ]

        # Reconstruct prompt in optimal order
        reordered_tags = []
        for category in optimal_order:
            reordered_tags.extend(categorized[category])

        # Add uncategorized tags (likely specific descriptors)
        reordered_tags.extend(uncategorized)

        reordered_text = ", ".join(reordered_tags)

        if reordered_text != text:
            log.append(f"Reordered tags for optimal Illustrious hierarchy")
            log.append(
                f"Order: Quality → Illustrious Specialty → Rating → Character → Composition → Background → Lighting → Style"
            )

        return reordered_text, log

    def apply_danbooru_weights(
        self, text, quality_boost, character_boost, composition_boost
    ):
        """Apply Danbooru-based tag weighting optimized for Illustrious"""
        log = []

        # Parse existing weights and tags
        weighted_tags = []
        for tag_part in text.split(","):
            tag_part = tag_part.strip()

            # Check if tag already has weight
            weight_match = re.search(r"\\(([^)]+):([0-9.]+)\\)", tag_part)
            if weight_match:
                tag_name = weight_match.group(1)
                current_weight = float(weight_match.group(2))
            else:
                tag_name = tag_part
                current_weight = 1.0

            # Apply category-based boosts
            boost_applied = False

            # Quality boost
            if tag_name in self.danbooru_categories["quality"]:
                new_weight = current_weight * quality_boost
                weighted_tags.append(f"({tag_name}:{new_weight:.2f})")
                if quality_boost != 1.0:
                    log.append(
                        f"Applied quality boost to '{tag_name}': {new_weight:.2f}"
                    )
                    boost_applied = True

            # Illustrious specialty boost
            elif tag_name in self.danbooru_categories["illustrious_specialty"]:
                new_weight = (
                    current_weight * quality_boost
                )  # Use quality boost for Illustrious specialties
                weighted_tags.append(f"({tag_name}:{new_weight:.2f})")
                if quality_boost != 1.0:
                    log.append(
                        f"Applied Illustrious specialty boost to '{tag_name}': {new_weight:.2f}"
                    )
                    boost_applied = True

            # Character boost
            elif (
                tag_name in self.danbooru_categories["character_count"]
                or tag_name in self.danbooru_categories["character_features"]
            ):
                new_weight = current_weight * character_boost
                weighted_tags.append(f"({tag_name}:{new_weight:.2f})")
                if character_boost != 1.0:
                    log.append(
                        f"Applied character boost to '{tag_name}': {new_weight:.2f}"
                    )
                    boost_applied = True

            # Composition boost
            elif tag_name in self.danbooru_categories["composition"]:
                new_weight = current_weight * composition_boost
                weighted_tags.append(f"({tag_name}:{new_weight:.2f})")
                if composition_boost != 1.0:
                    log.append(
                        f"Applied composition boost to '{tag_name}': {new_weight:.2f}"
                    )
                    boost_applied = True

            # Apply Illustrious-specific weights
            elif tag_name in self.tag_weights:
                new_weight = current_weight * self.tag_weights[tag_name]
                weighted_tags.append(f"({tag_name}:{new_weight:.2f})")
                log.append(
                    f"Applied Illustrious weight to '{tag_name}': {new_weight:.2f}"
                )
                boost_applied = True

            if not boost_applied:
                weighted_tags.append(tag_name)

        return ", ".join(weighted_tags), log

    def enable_hybrid_processing(self, text, version):
        """Enable hybrid Danbooru + Natural Language processing for newer versions"""
        log = []

        if version not in ["v1.1", "v2.0"]:
            return text, ["Hybrid mode only supported for v1.1+ versions"]

        # Convert some natural language phrases to tag + description hybrid
        hybrid_conversions = {
            "beautiful girl": "1girl, beautiful",
            "cute character": "cute",
            "detailed artwork": "extremely_detailed, official_art",
            "high quality": "high_quality, very_aesthetic",
            "dramatic scene": "dramatic_lighting, cinematic_lighting",
            "anime art": "anime_style, illustration",
            "masterwork": "masterpiece, very_aesthetic",
            "clean art": "clean_lineart, high_resolution",
        }

        for phrase, replacement in hybrid_conversions.items():
            if phrase in text.lower():
                text = text.replace(phrase, replacement)
                log.append(
                    f"Illustrious hybrid conversion: '{phrase}' → '{replacement}'"
                )

        return text, log

    def fix_padding_tokens(self, tokens):
        """Fix padding token issues for Illustrious-compatible models"""
        log = []

        # Count actual tokens vs padding
        for batch_idx, token_batch in enumerate(tokens):
            if hasattr(token_batch, "input_ids"):
                input_ids = token_batch.input_ids

                # Find padding tokens (usually 0 or a specific padding token ID)
                non_pad_count = torch.sum(input_ids != 0).item()
                total_count = len(input_ids)
                pad_count = total_count - non_pad_count

                if pad_count > 0:
                    log.append(
                        f"Detected {pad_count} padding tokens out of {total_count} total"
                    )

                    # If too many padding tokens (>70%), trim
                    if pad_count > total_count * 0.7:
                        # Trim to reduce padding token impact
                        trim_length = max(non_pad_count + 5, total_count // 2)
                        token_batch.input_ids = input_ids[:trim_length]
                        if hasattr(token_batch, "attention_mask"):
                            token_batch.attention_mask = token_batch.attention_mask[
                                :trim_length
                            ]
                        log.append(
                            f"Trimmed tokens to reduce padding impact: {total_count} → {trim_length}"
                        )

        return tokens, log

    def encode_with_clip_skip(self, clip, tokens, clip_skip):
        """Encode with CLIP skip optimization for Illustrious"""
        # Set CLIP skip layer if supported
        if hasattr(clip, "layer_idx"):
            original_layer = clip.layer_idx
            clip.layer_idx = -clip_skip

            try:
                # Use ComfyUI's built-in CLIP encoding with pooled output
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                return [[cond, {"pooled_output": pooled}]]
            finally:
                # Restore original layer
                clip.layer_idx = original_layer
        else:
            # Fallback encoding with pooled output
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            return [[cond, {"pooled_output": pooled}]]

    def encode_with_clip_skip_text(self, clip, text, clip_skip):
        """Encode text with CLIP skip"""
        tokens = clip.tokenize(text)
        return self.encode_with_clip_skip(clip, tokens, clip_skip)

    def generate_processing_report(self, original_text, processed_text, log, version):
        """Generate processing report"""
        report = f"🎨 Illustrious CLIP Encoder Report\n"
        report += f"═" * 40 + "\n"
        report += f"Version: {version}\n"
        report += f"Original length: {len(original_text.split(','))} tags\n"
        report += f"Processed length: {len(processed_text.split(','))} tags\n\n"

        if log:
            report += "🔧 Optimizations Applied:\n"
            for i, entry in enumerate(log, 1):
                report += f"  {i}. {entry}\n"

        report += f"\n📝 Final Prompt:\n{processed_text}\n"

        return report


class IllustriousNegativeCLIPEncoder:
    """Specialized negative prompt encoder for Illustrious"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
        "clip": ("CLIP", {"tooltip": "CLIP text encoder from your base model."}),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "worst_quality, bad_quality, lowres",
            "tooltip": "Negative prompt text (comma-separated tags).",
                    },
                ),
            },
            "optional": {
                "preset": (
                    [
                        "basic",
                        "comprehensive",
                        "ultra_clean",
                        "illustrious_optimized",
                        "custom",
                    ],
                    {"default": "illustrious_optimized", "tooltip": "Prebuilt negative tag lists tuned for Illustrious."},
                ),
                "illustrious_version": (
                    ["auto", "v0.1", "v1.0", "v1.1", "v2.0"],
                    {"default": "auto", "tooltip": "Used only for minor negative prompt hints."},
                ),
                "clip_skip": ("INT", {"default": 2, "min": 1, "max": 12, "tooltip": "Number of CLIP layers to skip from the end."}),
                "strength_multiplier": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "tooltip": "Weight multiplier applied to the whole negative prompt."},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode_negative"
    CATEGORY = "Easy Illustrious / Encoder"

    def __init__(self):
        self.negative_presets = {
            "basic": "worst_quality, bad_quality, lowres, bad_anatomy, sketch, jpeg_artifacts, signature, watermark, artist_name",
            "comprehensive": "worst_quality, bad_quality, lowres, bad_anatomy, bad_hands, extra_digits, multiple_views, fewer_digits, extra_limbs, missing_limbs, text, error, jpeg_artifacts, watermark, unfinished, signature, artistic_error, username, scan",
            "ultra_clean": "worst_quality, bad_quality, lowres, bad_anatomy, bad_hands, extra_digits, fewer_digits, cropped, jpeg_artifacts, signature, watermark, username, blurry, artist_name, multiple_views, text, error, extra_limbs, missing_limbs, fused_fingers, too_many_fingers, long_neck, cross-eyed, mutation, poorly_drawn, bad_proportions, gross_proportions, malformed, mutated_hands, poorly_drawn_hands, poorly_drawn_face, malformed_hands, extra_digit, pixelated, grainy, bad_art, amateur_drawing, cell_shading, inaccurate_limb, bad_composition, inaccurate_eyes, multiple_breasts, cloned_face, long_fingers, moved_limbs, merged_fingers, bad_feet",
            "illustrious_optimized": "worst_quality, bad_quality, poorly_detailed, jpeg_artifacts, extra_fingers, malformed_hands, blurry, compression_artifacts, pixelated, bad_anatomy, bad_hands, text, watermark, signature, username, lowres, normal_quality, oldest, early, very_displeasing, displeasing, bad_proportions",
        }

    def encode_negative(
        self,
        clip,
        text,
        preset="illustrious_optimized",
        illustrious_version="auto",
        clip_skip=2,
        strength_multiplier=1.0,
    ):

        # Use preset if text is default or empty
        if text.strip() in ["worst_quality, bad_quality, lowres", ""]:
            processed_text = self.negative_presets[preset]
        else:
            processed_text = text

        # Apply strength multiplier if not 1.0
        if strength_multiplier != 1.0:
            # Apply weighting to the entire negative prompt
            processed_text = f"({processed_text}:{strength_multiplier:.2f})"

        # Encode with clip skip
        encoder = IllustriousCLIPTextEncoder()
        tokens = clip.tokenize(processed_text)
        cond = encoder.encode_with_clip_skip(clip, tokens, clip_skip)

        return (cond,)
