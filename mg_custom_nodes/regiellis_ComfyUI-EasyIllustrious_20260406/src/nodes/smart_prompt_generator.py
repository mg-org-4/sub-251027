"""
Smart Anime Scene Generator Node

Generates comprehensive anime scene descriptions using extended vocabulary systems.
Combines the user's chain token system with extensive anime genre vocabulary.

- Anime Scene Vocabulary System: Comprehensive scene generation with 8+ categories
- Anime Composition System: Enhanced composition elements and framing
- Focuses on environments, situations, composition, and atmosphere
- Complements character selection nodes without overlap
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
import re

try:
    import yaml
except ImportError:
    yaml = None

from ..core.anime_scene_system import ScenesPlus as IllustriousScenesPlus
from .emotions import EMOTION_TOKENS as ILL_EMOTION_TOKENS
from .emotions import EMOTION_GROUPS as ILL_EMOTION_GROUPS


class IllustriousSmartSceneGenerator:
    """
    Smart Prompt Generator for anime scenes.
    Creates contextual scene descriptions that work with any character.
    """

    def __init__(self):
        self.scene_system_plus = IllustriousScenesPlus()

        # Stats tracking
        self.stats_file = (
            Path(__file__).parent.parent.parent / "data" / "smart_prompt_stats.yaml"
        )
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        # Get comprehensive categories from the enhanced scene system
        try:
            categories = list(IllustriousScenesPlus.CATEGORIES.keys())
        except:
            categories = [
                "Daily Life",
                "Outdoor",
                "Indoor",
                "Seasonal",
                "Atmospheric",
                "Action",
                "Emotional",
                "School Life",
                "Fantasy Adventure",
                "Romance",
            ]

        return {
            "required": {
                # Core scene configuration
                "Category": (
                    categories,
                    {
                        "default": "Outdoor",
                        "tooltip": "Select anime scene category with extensive vocabulary",
                    },
                ),
                "Complexity": (
                    ["simple", "medium", "detailed"],
                    {
                        "default": "medium",
                        "tooltip": "Control number of scene elements generated",
                    },
                ),
                # Environment elements
                "Include Time/Weather": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Add time of day and weather conditions",
                    },
                ),
                "Include Ambience": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Add atmospheric mood and feeling"},
                ),
                "Include Event": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Add specific events or activities"},
                ),
                "Include Prop": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Add scene props and objects"},
                ),
                "Include Density": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Add crowd density and population info",
                    },
                ),
                # Character elements (optional for scene completion)
                "Include Person Description": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Add basic person description (use only if no character nodes)",
                    },
                ),
                "Include Pose/Action": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Add pose and action descriptions"},
                ),
                "Include Clothing": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Add clothing descriptions (use only if no clothing nodes)",
                    },
                ),
                # Clothing refinement
                "Outfits": (
                    ["-"] + IllustriousScenesPlus.CLOTHING_OUTFIT,
                    {"default": "-", "tooltip": "Pick a full outfit or leave '-' for random."},
                ),
                "Top": (
                    ["-"] + IllustriousScenesPlus.CLOTHING_TOP,
                    {"default": "-", "tooltip": "Pick a top; used if no Outfit is chosen."},
                ),
                "Bottoms": (
                    ["-"] + IllustriousScenesPlus.CLOTHING_BOTTOM,
                    {"default": "-", "tooltip": "Pick bottoms; used if no Outfit is chosen."},
                ),
                "General Style": (
                    ["-"] + IllustriousScenesPlus.GENERAL_STYLES,
                    {"default": "-", "tooltip": "Overall clothing style/aesthetic."},
                ),
                "Headwear": (
                    ["-"] + IllustriousScenesPlus.HEADWEAR,
                    {"default": "-", "tooltip": "Specific headwear item to include."},
                ),
                # Expressions/Emotes (optional)
                "Emotion/Expression": (
                    ["-"] + ILL_EMOTION_TOKENS,
                    {"default": "-", "tooltip": "Optional expression/emote to include."},
                ),
                "Emotion Group": (
                    ["-"] + list(ILL_EMOTION_GROUPS.keys()),
                    {"default": "-", "tooltip": "Pick a group to browse or randomize when token is '-'"},
                ),
                "Emotion Weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Weight applied to emotion (tag:weight)."},
                ),
                "Wrap Emotion": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Wrap emotion as (tag:weight)."},
                ),
                # Safety and generation options
                "Safe Adult Subject": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Use adult-safe subject terms"},
                ),
                "Use Chain Insert": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable chain token integration with other nodes",
                    },
                ),
                "Strict Tags (no phrases)": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use individual tags instead of phrases for better compatibility",
                    },
                ),
                "De-duplicate With Prefix/Suffix": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Remove duplicate terms when chaining with other nodes",
                    },
                ),
                "Danbooru Tag Style": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Format output as lowercase, underscore-separated tags (community style)",
                    },
                ),
                # Token budget (maps to a soft character cap)
                "Token Count": (
                    ["-", "77", "150", "250", "300", "500"],
                    {
                        "default": "77",
                        "tooltip": "Approximate token budget. Drives a soft character cap and selection size.",
                    },
                ),
                # TIPO (text pre-sampling and ranking)
                "Enable TIPO Optimization": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Generate prompt candidates and pick the best for Illustrious",
                    },
                ),
                "TIPO Candidates": (
                    "INT",
                    {
                        "default": 8,
                        "min": 3,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Number of variants to consider",
                    },
                ),
                "TIPO Flavor": (
                    ["balanced", "vibrant", "soft", "natural"],
                    {
                        "default": "balanced",
                        "tooltip": "Illustrious-oriented flavor to bias scoring",
                    },
                ),
                "TIPO Max Length": (
                    "INT",
                    {
                        "default": 320,
                        "min": 80,
                        "max": 800,
                        "step": 10,
                        "tooltip": "Soft cap for final prompt length (characters)",
                    },
                ),
                "TIPO Seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "Randomization seed (0 = auto)",
                    },
                ),
                # Baseline / upscaling / negative helpers
                "Subject Fallback": (
                    ["none", "1girl", "1boy", "solo", "1girl, solo"],
                    {"default": "none", "tooltip": "Inject baseline subject tokens when person description is off."},
                ),
                "Upscale Tags": (
                    ["none", "anime_hd", "ultra_hd", "photo_8k"],
                    {"default": "none", "tooltip": "Prepend additional high-resolution/upscale tags after quality block."},
                ),
                "Negative Preset": (
                    ["none", "standard", "aggressive", "anime_clean", "photoreal", "custom"],
                    {"default": "none", "tooltip": "Generate a negative prompt using a curated or custom preset."},
                ),
                "Generate Negative Output": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "If true, second output returns the negative prompt string."},
                ),
                # Quality block controls
                "Use Quality Block": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Automatically prepend a quality/emphasis block unless already present."},
                ),
                "Quality Block Text": (
                    "STRING",
                    {"default": "masterpiece, best quality, ultra-detailed, highres", "multiline": False, "tooltip": "Comma separated quality tokens. Will be auto-weighted if enabled."},
                ),
                "Quality Weight": (
                    "FLOAT",
                    {"default": 1.2, "min": 0.1, "max": 2.5, "step": 0.05, "tooltip": "Weight applied per token or to group depending on scheme."},
                ),
                "Quality Weight Scheme": (
                    ["auto", "comfy", "A1111"],
                    {"default": "auto", "tooltip": "How to format weighted quality tokens (grouped vs per-token)."},
                ),
                "Per-Token Quality Weights": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Force apply weight to every quality token even in comfy/auto grouped modes."},
                ),
                "Subject Fallback Order": (
                    ["after_upscale", "before_upscale"],
                    {"default": "after_upscale", "tooltip": "Choose whether subject fallback tokens are inserted before or after Upscale Tags."},
                ),
            },
            "optional": {
                # Prompt weighting preferences (used for how we express optional emphasis)
                "Weight Interpretation": (
                    ["comfy", "A1111", "compel", "comfy++", "down_weight"],
                    {
                        "default": "comfy",
                        "tooltip": "How to hint weights in text; actual enforcement happens at encoding stage.",
                    },
                ),
                "Token Normalization": (
                    ["none", "mean", "length", "length+mean"],
                    {
                        "default": "none",
                        "tooltip": "Normalization hint for downstream encoders; used to shape emphasis strength.",
                    },
                ),
                "prefix": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Input from previous node in chain",
                    },
                ),
                "suffix": (
                    "STRING",
                    {"forceInput": True, "tooltip": "Additional suffix text"},
                ),
                "tipo_extra_negatives": (
                    "STRING",
                    {
                        "forceInput": False,
                        "tooltip": "Comma-separated terms to downweight/remove",
                    },
                ),
                "Custom Negative Preset": (
                    "STRING",
                    {"forceInput": False, "tooltip": "If Negative Preset is 'custom', use this string as the negative prompt."},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative", "metadata")
    FUNCTION = "generate_smart_prompt"
    CATEGORY = "Easy Illustrious / Generators"

    def generate_smart_prompt(self, **kwargs) -> Tuple[str, str, str]:
        start_time = time.time()
        try:
            # Use the enhanced scene system's construct method
            result = self.scene_system_plus.construct(**kwargs)

            # Extract the generated prompt
            if isinstance(result, tuple) and len(result) > 0:
                prompt = result[0]
            else:
                prompt = str(result)

            # --- Quality Block Configuration ---
            use_quality_block = bool(kwargs.get("Use Quality Block", True))
            qb_tokens_raw = kwargs.get("Quality Block Text", "masterpiece, best quality, ultra-detailed, highres") or ""
            quality_weight = float(kwargs.get("Quality Weight", 1.2) or 1.2)
            quality_scheme = kwargs.get("Quality Weight Scheme", "auto") or "auto"
            qb_tokens = [t.strip() for t in qb_tokens_raw.split(",") if t.strip()]
            per_token_force = bool(kwargs.get("Per-Token Quality Weights", False))
            # Detect if user already has any of the quality tokens (case-insensitive) present
            base_lower = (prompt or "").lower()
            q_present = any(t.lower() in base_lower for t in qb_tokens) if qb_tokens else False

            def build_quality_block(tokens: List[str]) -> str:
                if not tokens:
                    return ""
                # Apply weighting only if weight != 1.0
                if per_token_force:
                    scheme = "per"
                else:
                    if quality_scheme == "auto":
                        # Heuristic: if len tokens > 4 use per-token for readability else grouped
                        scheme = "per" if len(tokens) > 4 else "group"
                    elif quality_scheme.lower() == "a1111":
                        scheme = "per"
                    else:  # comfy
                        scheme = "group"
                if quality_weight == 1.0:
                    if scheme == "group":
                        return f"({', '.join(tokens)})"
                    else:
                        return ", ".join(f"({t})" for t in tokens)
                if scheme == "group":
                    # Attach weight to final token (common comfy convention)
                    if tokens:
                        tokens_mod = tokens[:-1] + [f"{tokens[-1]}:{quality_weight:.2f}"]
                    else:
                        tokens_mod = []
                    return f"({', '.join(tokens_mod)})"
                else:  # per-token weighting (A1111 style)
                    return ", ".join(f"({t}:{quality_weight:.2f})" for t in tokens)

            quality_block = build_quality_block(qb_tokens) if use_quality_block else ""

            tipo_enabled = kwargs.get("Enable TIPO Optimization", True)
            tipo_meta = {}
            strict_tags_flag = kwargs.get("Strict Tags (no phrases)", True)

            # Map token count to a soft character cap
            token_count_sel = str(kwargs.get("Token Count", "77") or "-")
            token_to_char = {"77": 320, "150": 650, "250": 1100, "300": 1300, "500": 2200}
            mapped_cap = token_to_char.get(token_count_sel)

            if tipo_enabled and strict_tags_flag:
                prompt, tipo_meta = self._optimize_prompt_tipo(
                    prompt=prompt,
                    category=kwargs.get("Category", "Outdoor"),
                    flavor=kwargs.get("TIPO Flavor", "balanced"),
                    k=int(kwargs.get("TIPO Candidates", 8)),
                    strict_tags=strict_tags_flag,
                    max_len=(int(mapped_cap) if mapped_cap else int(kwargs.get("TIPO Max Length", 320))),
                    seed=(int(kwargs.get("TIPO Seed", 0)) or (int(time.time()) & 0x7FFFFFFF)),
                    include_time_weather=kwargs.get("Include Time/Weather", True),
                    include_ambience=kwargs.get("Include Ambience", True),
                    extra_negatives=(kwargs.get("tipo_extra_negatives", "") or ""),
                    weight_interpretation=kwargs.get("Weight Interpretation", "comfy"),
                    token_normalization=kwargs.get("Token Normalization", "none"),
                )

            # Re-check presence after optimization (it may have reordered tokens)
            if not q_present and qb_tokens:
                pl = (prompt or "").lower()
                if any(t.lower() in pl for t in qb_tokens):
                    q_present = True
            if strict_tags_flag and use_quality_block and quality_block and not q_present:
                prompt = f"{quality_block}, {prompt}" if prompt else quality_block

            # Inject Upscale Tags (after quality block) if selected
            upscale_map = {
                "none": [],
                "anime_hd": ["highres", "detailed background"],
                "ultra_hd": ["8k", "ultra-detailed", "extremely detailed"],
                "photo_8k": ["RAW photo", "8k", "highres", "ultra-detailed"],
            }
            upscale_choice = kwargs.get("Upscale Tags", "none")
            pruning_meta = {"upscale_original": [], "upscale_final": [], "pruned": False, "dropped": []}
            if strict_tags_flag and upscale_choice in upscale_map and upscale_choice != "none":
                current_lower = (prompt or "").lower()
                up_tokens_all = [t for t in upscale_map[upscale_choice]]
                up_tokens = [t for t in up_tokens_all if t.lower() not in current_lower]
                pruning_meta["upscale_original"] = up_tokens_all
                if mapped_cap and up_tokens:
                    base_len = len(prompt)
                    # Progressive pruning
                    while len(up_tokens) > 1 and (base_len + len(", ".join(up_tokens)) > mapped_cap * 1.05):
                        dropped = up_tokens.pop()  # drop last
                        pruning_meta["dropped"].append(dropped)
                        pruning_meta["pruned"] = True
                if up_tokens:
                    pruning_meta["upscale_final"] = up_tokens[:]
                    if quality_block and prompt.startswith(quality_block):
                        rest = prompt[len(quality_block):].lstrip(", ")
                        prompt = f"{quality_block}, {', '.join(up_tokens)}" + (f", {rest}" if rest else "")
                    else:
                        prompt = ", ".join(up_tokens + ([prompt] if prompt else []))

            # Subject fallback injection (only if person description absent and tokens missing)
            subj_fb = kwargs.get("Subject Fallback", "none")
            subj_order = kwargs.get("Subject Fallback Order", "after_upscale")
            if strict_tags_flag and subj_fb and subj_fb != "none":
                fb_tokens = [t.strip() for t in subj_fb.split(",") if t.strip()]
                lower_prompt = (prompt or "").lower()
                inject = [t for t in fb_tokens if t.lower() not in lower_prompt]
                if inject:
                    # Placement: before_upscale means insert right after quality block (before any upscale tokens we may have injected)
                    if subj_order == "before_upscale" and quality_block and prompt.startswith(quality_block):
                        # Separate quality block then reassemble with subject tokens placed immediately after
                        remainder = prompt[len(quality_block):].lstrip(", ")
                        prompt = f"{quality_block}, {', '.join(inject)}" + (f", {remainder}" if remainder else "")
                    else:
                        # Default (after_upscale) or no quality block present -> standard logic
                        if quality_block and prompt.startswith(quality_block):
                            after = prompt[len(quality_block):].lstrip(", ")
                            prompt = f"{quality_block}, {', '.join(inject)}" + (f", {after}" if after else "")
                        else:
                            prompt = ", ".join(inject + ([prompt] if prompt else []))

            # Optionally augment with an explicit Emotion/Expression (if provided)
            emotion = kwargs.get("Emotion/Expression", "-")
            if (not emotion or emotion == "-"):
                group = kwargs.get("Emotion Group", "-")
                if group and group != "-":
                    try:
                        choices = ILL_EMOTION_GROUPS.get(group, [])
                        if choices:
                            import random
                            emotion = random.choice(choices)
                    except Exception:
                        pass
            if emotion and emotion != "-":
                if strict_tags_flag:
                    e_w = float(kwargs.get("Emotion Weight", 1.0) or 1.0)
                    e_wrap = bool(kwargs.get("Wrap Emotion", True))
                    e_token = f"({emotion}:{e_w:.2f})" if e_wrap else f"{emotion}:{e_w:.2f}"
                    prompt = f"{prompt}, {e_token}" if prompt else e_token
                else:
                    # Sentence mode: add a natural phrase
                    phrase = f"with a {emotion} expression"
                    prompt = f"{prompt}. {phrase}." if prompt and not prompt.strip().endswith(".") else f"{prompt} {phrase}." if prompt else f"{phrase}."

            # If Danbooru tag style is ON we should not destroy the weighted parentheses block.
            # We'll temporarily remove the quality block, format remaining tags, then restore it.
            if kwargs.get("Danbooru Tag Style", True) and strict_tags_flag:
                if quality_block and prompt.startswith(quality_block):
                    rest = prompt[len(quality_block):].lstrip(", ")
                    rest_fmt = self._format_danbooru_tags(rest)
                    prompt = f"{quality_block}, {rest_fmt}" if rest_fmt else quality_block
                else:
                    prompt = self._format_danbooru_tags(prompt)

            # Negative prompt helper generation
            neg_preset = kwargs.get("Negative Preset", "none")
            negative_prompt = ""
            if neg_preset and neg_preset != "none":
                if neg_preset == "custom":
                    negative_prompt = kwargs.get("Custom Negative Preset", "") or ""
                else:
                    neg_map = {
                        "standard": [
                            "(worst quality:1.2)", "(low quality:1.2)", "(normal quality:1.2)",
                            "lowres", "bad anatomy", "bad hands", "text", "error", "extra digits",
                            "fewer digits", "cropped", "blurry", "jpeg artifacts", "signature", "watermark",
                            "username", "artist name",
                        ],
                        "aggressive": [
                            "(worst quality:1.3)", "(low quality:1.3)", "(normal quality:1.3)",
                            "lowres", "bad anatomy", "bad hands", "text", "error", "extra digits", "fewer digits",
                            "cropped", "blurry", "jpeg artifacts", "signature", "watermark", "username",
                            "artist name", "deformed", "mutated", "long neck", "long body", "bad proportions",
                            "poorly drawn face", "poorly drawn hands",
                        ],
                        "anime_clean": [
                            "(worst quality:1.2)", "(low quality:1.2)", "(normal quality:1.2)",
                            "lowres", "bad anatomy", "bad hands", "text", "error", "extra digits",
                            "fewer digits", "blurry", "jpeg artifacts", "signature", "watermark", "nsfw",
                        ],
                        "photoreal": [
                            "(worst quality:1.2)", "(low quality:1.2)", "(normal quality:1.2)",
                            "lowres", "bad anatomy", "bad hands", "text", "error", "extra digits", "fewer digits",
                            "cropped", "blurry", "jpeg artifacts", "signature", "watermark", "unnatural skin",
                            "overprocessed", "grain", "deformed", "mutated",
                        ],
                    }
                    toks = neg_map.get(neg_preset, [])
                    # Dedupe + join
                    seen = set()
                    deduped = []
                    for t in toks:
                        k = t.lower()
                        if k not in seen:
                            seen.add(k)
                            deduped.append(t)
                    negative_prompt = ", ".join(deduped)
            gen_neg_out = bool(kwargs.get("Generate Negative Output", False))

            # ---- Absolute Token Count Enforcement ----
            token_target_meta = {}
            token_count_sel_int = None
            try:
                if token_count_sel and token_count_sel.isdigit():
                    token_count_sel_int = int(token_count_sel)
            except Exception:
                token_count_sel_int = None

            def _split_tokens(s: str) -> List[str]:
                return [t.strip() for t in (s or "").split(",") if t.strip()]

            def _rebuild_from_tokens(tokens: List[str]) -> str:
                return ", ".join(tokens)

            if token_count_sel_int and token_count_sel_int > 0:
                # Enforce after all structural insertions (and after Danbooru formatting if used) to reflect final output tokens.
                # If Danbooru style is ON, re-run formatting prior to enforcement to maintain style.
                enforce_after_format = kwargs.get("Danbooru Tag Style", True) and strict_tags_flag
                if enforce_after_format:
                    # Already formatted earlier if style on; no action needed here.
                    pass
                original_tokens = _split_tokens(prompt)
                original_len = len(original_tokens)
                target = token_count_sel_int
                added = []
                truncated = []
                if original_len > target:
                    truncated = original_tokens[target:]
                    working = original_tokens[:target]
                elif original_len < target:
                    working = original_tokens[:]
                    # Build a filler pool from flavor boosts + ambience + generic detail terms
                    filler_pool = set([
                        "detailed", "finely detailed", "crisp lines", "balanced colors", "clean lineart",
                        "dynamic lighting", "dramatic lighting", "soft lighting", "atmospheric depth",
                        "subtle shading", "volumetric light", "color harmony", "sharp focus",
                    ])
                    # Extend pool with flavor boosts heuristically
                    for fb_list in [
                        ["vibrant colors", "high saturation control", "dynamic lighting"],
                        ["soft lighting", "pastel tones", "gentle shading"],
                        ["natural color grading", "realistic lighting", "neutral tones"],
                    ]:
                        filler_pool.update(fb_list)
                    lower_existing = set(t.lower() for t in working)
                    for f in list(filler_pool):
                        if len(working) >= target:
                            break
                        if f.lower() not in lower_existing:
                            working.append(f)
                            added.append(f)
                            lower_existing.add(f.lower())
                else:
                    working = original_tokens
                prompt = _rebuild_from_tokens(working)
                token_target_meta = {
                    "enabled": True,
                    "target": target,
                    "original_length": original_len,
                    "final_length": len(working),
                    "added": added,
                    "truncated": truncated,
                }
            else:
                token_target_meta = {"enabled": False}

            # Generate comprehensive metadata
            generation_time = time.time() - start_time
            category = kwargs.get("Category", "Outdoor")

            metadata = {
                "generator": "Illustrious Smart Scene Generator",
                "version": "2.7 - Absolute Token Count Enforcement",
                "system": "Anime Scene Vocabulary System + Anime Composition System",
                "category": category,
                "complexity": kwargs.get("Complexity", "medium"),
                "generation_time": round(generation_time, 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                # Feature usage tracking
                "features_used": {
                    "time_weather": kwargs.get("Include Time/Weather", True),
                    "ambience": kwargs.get("Include Ambience", True),
                    "events": kwargs.get("Include Event", False),
                    "props": kwargs.get("Include Prop", True),
                    "density": kwargs.get("Include Density", False),
                    "person_description": kwargs.get(
                        "Include Person Description", False
                    ),
                    "pose_action": kwargs.get("Include Pose/Action", True),
                    "clothing": kwargs.get("Include Clothing", False),
                },
                # Generation settings
                "settings": {
                    "safe_adult_subject": kwargs.get("Safe Adult Subject", True),
                    "use_chain_insert": kwargs.get("Use Chain Insert", True),
                    "strict_tags": kwargs.get("Strict Tags (no phrases)", True),
                    "deduplicate": kwargs.get("De-duplicate With Prefix/Suffix", True),
                    "weight_interpretation": kwargs.get("Weight Interpretation", "comfy"),
                    "token_normalization": kwargs.get("Token Normalization", "none"),
                    "token_count": token_count_sel,
                },
                # Chain information
                "chain_info": {
                    "has_prefix": bool(kwargs.get("prefix", "").strip()),
                    "has_suffix": bool(kwargs.get("suffix", "").strip()),
                    "chain_enabled": kwargs.get("Use Chain Insert", True),
                },
                # Vocabulary stats
                "vocabulary_scope": {
                    "category_environments": len(
                        IllustriousScenesPlus.CATEGORIES.get(category, {}).get(
                            "env", []
                        )
                    ),
                    "category_events": len(
                        IllustriousScenesPlus.CATEGORIES.get(category, {}).get(
                            "events", []
                        )
                    ),
                    "total_categories": len(IllustriousScenesPlus.CATEGORIES),
                    "total_props": len(IllustriousScenesPlus.PROPS),
                    "total_poses": len(IllustriousScenesPlus.POSES),
                    "total_ambience_options": len(IllustriousScenesPlus.AMBIENCE),
                    "total_time_weather_options": len(
                        IllustriousScenesPlus.TIME_WEATHER
                    ),
                    "total_headwear": len(getattr(IllustriousScenesPlus, "HEADWEAR", [])),
                    "total_general_styles": len(getattr(IllustriousScenesPlus, "GENERAL_STYLES", [])),
                    "total_emotions": len(ILL_EMOTION_TOKENS),
                },
                # TIPO metadata
                "tipo": tipo_meta,
                "negative": {
                    "preset": neg_preset,
                    "prompt": negative_prompt,
                },
                "quality_block": {
                    "enabled": use_quality_block,
                    "tokens": qb_tokens,
                    "weight": quality_weight,
                    "scheme": quality_scheme,
                    "forced_per_token": per_token_force,
                    "present_preexisting": q_present,
                },
                "pruning": pruning_meta,
                "subject_fallback": {
                    "value": subj_fb,
                    "order": subj_order,
                },
                "token_enforcement": token_target_meta,
            }

            # Update statistics
            self._update_stats(
                category, kwargs.get("Complexity", "medium"), generation_time, kwargs
            )

            # Outputs: prompt, (negative or metadata), metadata (third)
            if gen_neg_out:
                return prompt, (negative_prompt or ""), json.dumps(metadata, indent=2)
            else:
                # Maintain backward-ish compatibility: second output empty when not used
                return prompt, (negative_prompt or ""), json.dumps(metadata, indent=2)
        except Exception as e:
            # Return detailed error information
            error_metadata = {
                "generator": "Illustrious Smart Scene Generator",
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "category": kwargs.get("Category", "Outdoor"),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "inputs_received": list(kwargs.keys()),
                "system_info": {
                    "scene_system_available": hasattr(self, "scene_system_plus"),
                    "composition_system_available": hasattr(self, "scene_system"),
                },
            }

            return (
                f"Error generating anime scene prompt: {str(e)}",
                "",
                json.dumps(error_metadata, indent=2),
            )

    def _update_stats(
        self,
        category: str,
        complexity: str,
        generation_time: float,
        kwargs: Dict[str, Any] = None,
    ):
        """Update comprehensive generation statistics."""
        try:
            # Load existing stats
            stats = self._load_stats()

            # Update basic counters
            stats["total_generations"] = stats.get("total_generations", 0) + 1
            stats["total_time"] = stats.get("total_time", 0.0) + generation_time
            stats["avg_time"] = stats["total_time"] / stats["total_generations"]

            # Update usage patterns
            category_stats = stats.setdefault("category_usage", {})
            category_stats[category] = category_stats.get(category, 0) + 1

            complexity_stats = stats.setdefault("complexity_usage", {})
            complexity_stats[complexity] = complexity_stats.get(complexity, 0) + 1

            # Track feature usage if kwargs provided
            if kwargs:
                feature_stats = stats.setdefault("feature_usage", {})
                features = {
                    "time_weather": kwargs.get("Include Time/Weather", False),
                    "ambience": kwargs.get("Include Ambience", False),
                    "events": kwargs.get("Include Event", False),
                    "props": kwargs.get("Include Prop", False),
                    "density": kwargs.get("Include Density", False),
                    "person_description": kwargs.get(
                        "Include Person Description", False
                    ),
                    "pose_action": kwargs.get("Include Pose/Action", False),
                    "clothing": kwargs.get("Include Clothing", False),
                }

                for feature, enabled in features.items():
                    if enabled:
                        feature_stats[feature] = feature_stats.get(feature, 0) + 1

                # Track setting preferences
                setting_stats = stats.setdefault("setting_preferences", {})
                settings = {
                    "safe_adult_subject": kwargs.get("Safe Adult Subject", True),
                    "use_chain_insert": kwargs.get("Use Chain Insert", True),
                    "strict_tags": kwargs.get("Strict Tags (no phrases)", True),
                    "deduplicate": kwargs.get("De-duplicate With Prefix/Suffix", True),
                }

                for setting, value in settings.items():
                    setting_group = setting_stats.setdefault(
                        setting, {"true": 0, "false": 0}
                    )
                    setting_group["true" if value else "false"] += 1

            # Performance tracking
            perf_stats = stats.setdefault("performance", {})
            if generation_time < 0.1:
                perf_stats["fast_generations"] = (
                    perf_stats.get("fast_generations", 0) + 1
                )
            elif generation_time > 1.0:
                perf_stats["slow_generations"] = (
                    perf_stats.get("slow_generations", 0) + 1
                )

            # Update version info
            stats["version"] = "2.0 - Enhanced Vocabulary"
            stats["system"] = "Anime Scene Vocabulary System + Anime Composition System"
            stats["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

            # Save stats
            self._save_stats(stats)

        except Exception as e:
            # Don't fail generation if stats update fails
            print(f"[Illustrious SmartSceneGenerator] Stats update failed: {e}")

    def _load_stats(self) -> Dict[str, Any]:
        """Load statistics from file."""
        if not self.stats_file.exists():
            return {}

        try:
            content = self.stats_file.read_text(encoding="utf-8")
            if yaml:
                return yaml.safe_load(content) or {}
            else:
                return json.loads(content)
        except Exception:
            return {}

    def _save_stats(self, stats: Dict[str, Any]):
        """Save statistics to file."""
        try:
            if yaml:
                content = yaml.safe_dump(
                    stats, default_flow_style=False, sort_keys=False
                )
            else:
                content = json.dumps(stats, indent=2)

            # Atomic write
            temp_file = self.stats_file.with_suffix(".tmp")
            temp_file.write_text(content, encoding="utf-8")
            temp_file.replace(self.stats_file)

        except Exception as e:
            print(f"[SmartPromptGenerator] Failed to save stats: {e}")

    def get_current_stats(self) -> Dict[str, Any]:
        """Get comprehensive current statistics for API access."""
        stats = self._load_stats()
        stats["status"] = "ok"
        stats["stats_file"] = str(self.stats_file)

        # Add system information
        stats["system_info"] = {
            "generator_name": "Illustrious Smart Scene Generator",
            "version": "2.0 - Enhanced Vocabulary",
            "vocabulary_systems": [
                "Anime Scene Vocabulary System",
                "Anime Composition System",
            ],
            "total_categories": len(IllustriousScenesPlus.CATEGORIES),
            "total_vocabulary_size": {
                "props": len(IllustriousScenesPlus.PROPS),
                "poses": len(IllustriousScenesPlus.POSES),
                "ambience_options": len(IllustriousScenesPlus.AMBIENCE),
                "time_weather_options": len(IllustriousScenesPlus.TIME_WEATHER),
                "density_options": len(IllustriousScenesPlus.DENSITY),
            },
            "available_categories": list(IllustriousScenesPlus.CATEGORIES.keys()),
        }

        # Calculate usage percentages if we have data
        if "category_usage" in stats and "total_generations" in stats:
            total = stats["total_generations"]
            stats["category_percentages"] = {
                cat: round((count / total) * 100, 1)
                for cat, count in stats["category_usage"].items()
            }

        return stats

    # ---- TIPO internals (text-only, Illustrious-flavored) ----
    def _optimize_prompt_tipo(
        self,
        prompt: str,
        category: str,
        flavor: str,
        k: int,
        strict_tags: bool,
        max_len: int,
        seed: int,
        include_time_weather: bool,
        include_ambience: bool,
        extra_negatives: str,
        weight_interpretation: str,
        token_normalization: str,
    ) -> Tuple[str, Dict[str, Any]]:
        rng = random.Random(seed)
        base_tokens = self._tipo_split(prompt, strict=strict_tags)
        base_tokens = self._tipo_dedupe(base_tokens)

        flavor_boosts = {
            "balanced": ["balanced colors", "clean lineart", "consistent shading"],
            "vibrant": [
                "vibrant colors",
                "high saturation control",
                "dynamic lighting",
            ],
            "soft": ["soft lighting", "pastel tones", "gentle shading"],
            "natural": ["natural color grading", "realistic lighting", "neutral tones"],
        }
        banned = set(
            t.strip().lower() for t in self._tipo_split(extra_negatives, strict=False)
        )

        candidates = []
        for _ in range(max(3, k)):
            cand = self._tipo_perturb(
                rng,
                base_tokens,
                flavor,
                flavor_boosts,
                include_time_weather,
                include_ambience,
                banned,
                weight_interpretation,
                token_normalization,
                strict_tags,
            )
            cand_s = ", ".join(cand)
            score, breakdown = self._tipo_score(
                cand, flavor, category, strict_tags, max_len, flavor_boosts, banned
            )
            candidates.append((score, cand_s, breakdown))

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_prompt, best_bd = candidates[0]

        meta = {
            "enabled": True,
            "seed": seed,
            "selected_score": round(best_score, 4),
            "selected_breakdown": best_bd,
            "top3": [
                {"score": round(s, 4), "prompt": p[:max_len], "why": bd}
                for s, p, bd in candidates[:3]
            ],
            "candidate_count": len(candidates),
            "flavor": flavor,
            "category": category,
        }
        if len(best_prompt) > max_len:
            best_prompt = best_prompt[:max_len].rstrip(", ")

        return best_prompt, meta

    def _tipo_split(self, text: str, strict: bool) -> List[str]:
        parts = [t.strip() for t in (text or "").split(",")]
        parts = [p for p in parts if p]
        if strict:
            tokens = []
            for p in parts:
                tokens.extend([t for t in re.split(r"[;|/]+", p) if t.strip()])
            parts = [t.strip() for t in tokens if t.strip()]
        return parts

    def _tipo_dedupe(self, tokens: List[str]) -> List[str]:
        seen, out = set(), []
        for t in tokens:
            key = re.sub(r"\s+", " ", t.lower())
            if key not in seen:
                seen.add(key)
                out.append(t)
        return out

    def _tipo_perturb(
        self,
        rng: random.Random,
        tokens: List[str],
        flavor: str,
        flavor_boosts: Dict[str, List[str]],
        include_time_weather: bool,
        include_ambience: bool,
        banned: set,
        weight_interpretation: str,
        token_normalization: str,
        strict_tags: bool,
    ) -> List[str]:
        cand = tokens[:]
        if cand:
            head = cand[0:1]
            tail = cand[1:]
            rng.shuffle(tail)
            cand = head + tail

        for _ in range(min(3, max(1, len(cand) // 6))):
            i = rng.randrange(0, len(cand)) if cand else 0
            j = rng.randrange(0, len(cand)) if cand else 0
            if i < len(cand) and j < len(cand):
                cand[i], cand[j] = cand[j], cand[i]

        boosts = flavor_boosts.get(flavor, [])
        for b in boosts:
            if b not in cand and rng.random() < 0.6:
                cand.append(b)

        if include_time_weather and not any(
            ("sunset" in t or "night" in t or "morning" in t or "weather" in t)
            for t in cand
        ):
            if rng.random() < 0.5:
                cand.append(
                    rng.choice(
                        ["sunset", "golden hour", "overcast", "night city lights"]
                    )
                )
        if include_ambience and not any(
            ("ambient" in t or "mood" in t or "atmosphere" in t) for t in cand
        ):
            if rng.random() < 0.5:
                cand.append(
                    rng.choice(
                        ["cinematic atmosphere", "serene mood", "dramatic atmosphere"]
                    )
                )

        def emphasize(t: str) -> str:
            if not strict_tags:
                return t
            if "(" in t or ")" in t:
                return t
            # pick a gentle weight depending on preference
            if len(t) <= 40:
                if weight_interpretation == "down_weight":
                    w = 0.9
                else:
                    # Slightly modulate by normalization preference
                    w = 1.1 if token_normalization in ("none", "mean") else 1.07
                return f"({t}:{w})"
            return t

        for idx in range(min(2, len(cand))):
            if rng.random() < 0.6:
                cand[idx] = emphasize(cand[idx])

        cand = [t for t in cand if t.strip().lower() not in banned]
        return self._tipo_dedupe(cand)

    def _tipo_score(
        self,
        tokens: List[str],
        flavor: str,
        category: str,
        strict_tags: bool,
        max_len: int,
        flavor_boosts: Dict[str, List[str]],
        banned: set,
    ) -> Tuple[float, Dict[str, Any]]:
        text = ", ".join(tokens)
        score = 0.0
        why = {}

        boosts = flavor_boosts.get(flavor, [])
        present = sum(1 for b in boosts if any(b.lower() in t.lower() for t in tokens))
        score += present * 1.2
        why["flavor_hits"] = present

        cat_hits = sum(1 for t in tokens if category.lower() in t.lower())
        score += cat_hits * 0.5
        why["category_hits"] = cat_hits

        uniq = len(set(t.lower() for t in tokens))
        dup_penalty = max(0, len(tokens) - uniq) * 0.6
        score -= dup_penalty
        why["dup_penalty"] = round(dup_penalty, 3)

        length_pen = max(0, len(text) - max_len) / 40.0
        score -= length_pen
        why["length_penalty"] = round(length_pen, 3)

        ban_pen = sum(1 for t in tokens if t.strip().lower() in banned) * 0.8
        score -= ban_pen
        why["banned_penalty"] = round(ban_pen, 3)

        ambience_hits = sum(
            1
            for t in tokens
            if any(k in t.lower() for k in ["atmosphere", "ambient", "mood"])
        )
        tw_hits = sum(
            1
            for t in tokens
            if t.lower()
            in [
                "sunset",
                "golden hour",
                "overcast",
                "night city lights",
                "night",
                "morning",
            ]
        )
        score += min(1, ambience_hits) * 0.4 + min(1, tw_hits) * 0.4
        why["context_bonus"] = round(
            min(1, ambience_hits) * 0.4 + min(1, tw_hits) * 0.4, 3
        )

        il_bonus = (
            sum(
                1
                for k in ["clean lineart", "balanced colors", "natural color grading"]
                if any(k in t.lower() for t in tokens)
            )
            * 0.5
        )
        score += il_bonus
        why["illustrious_bonus"] = round(il_bonus, 3)

        return score, why

    def _format_danbooru_tags(self, text: str) -> str:
        """Convert a comma-separated list into danbooru-style tags: lowercase, underscores, no weights."""
        parts = [t.strip() for t in (text or "").split(",") if t.strip()]
        out = []
        seen = set()
        for p in parts:
            # strip weights like (tag:1.1)
            if p.startswith("(") and p.endswith(")") and ":" in p:
                try:
                    p = p[1:-1].split(":", 1)[0]
                except Exception:
                    pass
            p = p.replace("(", "").replace(")", "")
            p = re.sub(r"\s+", "_", p).lower()
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return ", ".join(out)
