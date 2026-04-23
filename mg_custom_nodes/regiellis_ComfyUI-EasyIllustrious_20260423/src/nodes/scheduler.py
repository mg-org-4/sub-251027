import torch
import numpy as np
import math
import comfy.model_management
from comfy.k_diffusion import sampling as k_diffusion_sampling


class IllustriousAwareScheduler:
    """Custom scheduler that accounts for Illustrious XL training patterns"""

    def __init__(self):
        # Common content patterns that benefit from tailored scheduling for Illustrious
        self.content_patterns = {
            "character_focus": ["1girl", "1boy", "solo", "portrait", "close-up"],
            "complex_scene": [
                "multiple_girls",
                "multiple_boys",
                "crowd",
                "detailed_background",
            ],
            "artistic_style": [
                "official_art",
                "concept_art",
                "promotional_art",
                "cover_art",
            ],
            "high_detail": ["extremely_detailed", "intricate", "ornate", "decorated"],
            "lighting_focus": [
                "dramatic_lighting",
                "rim_lighting",
                "backlighting",
                "volumetric_lighting",
            ],
            "illustrious_specialty": [
                "anime",
                "illustration",
                "high_quality",
                "masterpiece",
                "detailed",
            ],
        }

    def create_illustrious_schedule(
        self, steps, content_analysis=None, model_version="v1.0"
    ):
        """Create a schedule optimized for Illustrious XL models"""

        # Base schedule using cosine annealing (proven effective for Illustrious)
        base_schedule = self.cosine_annealing_schedule(steps)

        # Adjust based on content analysis if provided
        if content_analysis:
            schedule = self.adjust_for_content_patterns(
                base_schedule, content_analysis, model_version
            )
        else:
            schedule = base_schedule

        # Apply Illustrious-specific modifications
        schedule = self.apply_illustrious_modifications(schedule, model_version)

        return schedule

    def cosine_annealing_schedule(self, steps, min_noise=0.0, max_noise=1.0):
        """Cosine annealing schedule for Illustrious"""
        schedule = []
        for i in range(steps):
            # Cosine decay from max_noise to min_noise
            progress = i / (steps - 1) if steps > 1 else 0
            noise_level = (
                min_noise
                + (max_noise - min_noise) * (1 + math.cos(math.pi * progress)) / 2
            )
            schedule.append(noise_level)
        return torch.tensor(schedule)

    def adjust_for_content_patterns(
        self, base_schedule, content_analysis, model_version
    ):
        """Adjust schedule based on detected content patterns"""
        schedule = base_schedule.clone()

        # Character-focused images benefit from more careful early denoising
        if content_analysis.get("character_focus", 0) > 0.5:
            # Emphasize early steps for better character structure
            early_boost = torch.exp(
                -torch.arange(len(schedule)) / (len(schedule) * 0.3)
            )
            schedule = schedule * (0.8 + 0.2 * early_boost)

        # Complex scenes need more gradual denoising
        if content_analysis.get("complex_scene", 0) > 0.3:
            # More linear progression for complex compositions
            linear_factor = torch.linspace(1.0, 0.8, len(schedule))
            schedule = schedule * linear_factor

        # High-detail content benefits from extended fine denoising
        if content_analysis.get("high_detail", 0) > 0.4:
            # Extend the tail for better detail resolution
            tail_steps = len(schedule) // 3
            if tail_steps > 0:
                tail_modifier = torch.exp(-torch.arange(tail_steps) / tail_steps * 2)
                schedule[-tail_steps:] = schedule[-tail_steps:] * (
                    0.7 + 0.3 * tail_modifier
                )

        # Illustrious specialty content (anime/illustration) gets optimized scheduling
        if content_analysis.get("illustrious_specialty", 0) > 0.6:
            # Apply gentle smoothing for anime-style content
            smoothed = torch.zeros_like(schedule)
            smoothed[0] = schedule[0]
            smoothed[-1] = schedule[-1]
            for i in range(1, len(schedule) - 1):
                smoothed[i] = (
                    0.2 * schedule[i - 1] + 0.6 * schedule[i] + 0.2 * schedule[i + 1]
                )
            schedule = smoothed

        return schedule

    def apply_illustrious_modifications(self, schedule, model_version):
        """Apply version-specific Illustrious modifications"""

        if model_version == "v0.5":
            # v0.5 needs more aggressive early denoising
            schedule = schedule * 1.1
            schedule = torch.clamp(schedule, 0, 1)

        elif model_version in ["v0.75", "v1.0"]:
            # Newer versions work better with gentler curves
            # Apply slight smoothing
            smoothed = torch.zeros_like(schedule)
            smoothed[0] = schedule[0]
            smoothed[-1] = schedule[-1]
            for i in range(1, len(schedule) - 1):
                smoothed[i] = (
                    0.25 * schedule[i - 1] + 0.5 * schedule[i] + 0.25 * schedule[i + 1]
                )
            schedule = smoothed

        return schedule


class AdaptiveContentScheduler:
    """Scheduler that adapts step sizing based on content complexity for Illustrious"""

    def __init__(self):
        self.complexity_cache = {}

    def create_adaptive_schedule(
        self, steps, latent_input, positive_conditioning, model_version="v1.0"
    ):
        """Create schedule that adapts to content complexity"""

        # Analyze content complexity from conditioning and latent
        complexity_score = self.analyze_content_complexity(
            positive_conditioning, latent_input
        )

        # Create base schedule
        if complexity_score > 0.7:  # High complexity
            schedule = self.high_complexity_schedule(steps)
        elif complexity_score < 0.3:  # Low complexity
            schedule = self.low_complexity_schedule(steps)
        else:  # Medium complexity
            schedule = self.medium_complexity_schedule(steps)

        # Apply adaptive step sizing
        schedule = self.apply_adaptive_step_sizing(schedule, complexity_score)

        return schedule

    def analyze_content_complexity(self, conditioning, latent_input):
        """Analyze complexity from conditioning and latent input"""
        complexity_score = 0.5  # Base complexity

        # This is a simplified version - in practice, you'd analyze:
        # 1. Token count and diversity in conditioning
        # 2. Presence of complex descriptors
        # 3. Latent image complexity (if img2img)
        # 4. Aspect ratio and resolution

        # For now, return medium complexity
        return complexity_score

    def high_complexity_schedule(self, steps):
        """Schedule optimized for complex scenes/compositions"""
        # More steps in the middle range for complex content
        t = torch.linspace(0, 1, steps)
        # Sigmoid-like curve with emphasis on middle denoising
        schedule = 1.0 - torch.sigmoid(8 * (t - 0.5))
        return schedule

    def low_complexity_schedule(self, steps):
        """Schedule optimized for simple content"""
        # More aggressive early denoising for simple content
        t = torch.linspace(0, 1, steps)
        schedule = (1.0 - t) ** 1.5
        return schedule

    def medium_complexity_schedule(self, steps):
        """Balanced schedule for medium complexity content"""
        # Standard cosine schedule works well for medium complexity
        t = torch.linspace(0, 1, steps)
        schedule = (1 + torch.cos(math.pi * t)) / 2
        return schedule

    def apply_adaptive_step_sizing(self, schedule, complexity_score):
        """Apply non-uniform step sizing based on complexity"""
        # For high complexity, use smaller steps in critical regions
        if complexity_score > 0.6:
            # Compress early steps, expand middle steps
            indices = torch.arange(len(schedule), dtype=torch.float32)
            # Non-linear index mapping
            new_indices = torch.pow(indices / (len(schedule) - 1), 0.8) * (
                len(schedule) - 1
            )
            # Interpolate to get new schedule
            schedule = torch.nn.functional.interpolate(
                schedule.unsqueeze(0).unsqueeze(0),
                size=len(schedule),
                mode="linear",
                align_corners=True,
            ).squeeze()

        return schedule


class IllustriousDistanceScheduler:
    """Implementation of Distance-like scheduler optimized for Illustrious"""

    def create_distance_schedule(self, steps, distance_power=1.0, model_version="v1.0"):
        """Create a distance-based schedule similar to the Distance sampler"""

        # Base distance schedule
        t = torch.linspace(0, 1, steps)

        # Distance-based noise schedule
        # Higher distance_power = more emphasis on early steps
        schedule = torch.pow(1.0 - t, distance_power)

        # Illustrious-specific modifications
        if model_version == "v1.0":
            # v1.0 handles high-resolution better with gentler distance curve
            distance_power = min(distance_power * 0.9, 1.0)
            schedule = torch.pow(1.0 - t, distance_power)

        # Add slight randomization to prevent artifacts (like Distance sampler)
        if len(schedule) > 10:
            noise_factor = 0.02  # Small randomization
            random_offset = torch.randn_like(schedule) * noise_factor
            schedule = schedule + random_offset
            schedule = torch.clamp(schedule, 0, 1)
            # Re-sort to maintain monotonic decrease
            schedule = torch.sort(schedule, descending=True)[0]

        return schedule


class IllustriousSchedulerNode:
    """ComfyUI node for custom Illustrious schedulers"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": (
                    "INT",
                    {
                        "default": 24,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Number of sampling steps to generate a sigma curve for.",
                    },
                ),
                "scheduler_type": (
                    [
                        "aware",
                        "adaptive_content",
                        "distance",
                        "cosine_annealing",
                        "hybrid_optimized",
                    ],
                    {
                        "default": "aware",
                        "tooltip": "Which strategy to use when shaping the sigma schedule.",
                    },
                ),
            },
            "optional": {
                "model_version": (
                    ["auto", "v0.5", "v0.75", "v1.0", "v1.1", "v2.0", "v3.x"],
                    {
                        "default": "v1.0",
                        "tooltip": "Illustrious model version to tailor schedule behavior (auto uses simple heuristics).",
                    },
                ),
                "complexity_bias": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": "Bias for adaptive_content: 0=simple scenes, 1=highly complex scenes.",
                    },
                ),
                "distance_power": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Curve strength for distance schedule. Higher values emphasize early denoising.",
                    },
                ),
                "content_analysis": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional hints (e.g., tags/notes). Affects aware/hybrid schedules if provided.",
                    },
                ),
                "adaptive_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable non-uniform step sizing for complex content (future-facing; safe to leave on).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "create_schedule"
    CATEGORY = "Easy Illustrious "

    def create_schedule(
        self,
        steps,
        scheduler_type,
        model_version="v1.0",
        complexity_bias=0.5,
        distance_power=1.0,
        content_analysis="",
        adaptive_mode=True,
    ):

        # Initialize schedulers
        illustrious_scheduler = IllustriousAwareScheduler()
        adaptive_scheduler = AdaptiveContentScheduler()
        distance_scheduler = IllustriousDistanceScheduler()

        # Parse content analysis if provided
        content_patterns = (
            self.parse_content_analysis(content_analysis) if content_analysis else None
        )

        if scheduler_type == "aware":
            schedule = illustrious_scheduler.create_illustrious_schedule(
                steps, content_patterns, model_version
            )

        elif scheduler_type == "adaptive_content":
            # For adaptive content, we need latent input - using default for now
            schedule = adaptive_scheduler.medium_complexity_schedule(steps)
            if complexity_bias != 0.5:
                # Adjust for bias
                bias_factor = 1.0 + (complexity_bias - 0.5) * 0.4
                schedule = torch.pow(schedule, bias_factor)

        elif scheduler_type == "distance":
            schedule = distance_scheduler.create_distance_schedule(
                steps, distance_power, model_version
            )

        elif scheduler_type == "cosine_annealing":
            schedule = illustrious_scheduler.cosine_annealing_schedule(steps)

        elif scheduler_type == "hybrid_optimized":
            # Combine multiple approaches
            aware_sched = illustrious_scheduler.create_illustrious_schedule(
                steps, content_patterns, model_version
            )
            distance_sched = distance_scheduler.create_distance_schedule(
                steps, distance_power, model_version
            )
            # Weighted combination
            schedule = 0.6 * aware_sched + 0.4 * distance_sched

        else:
            # Fallback to cosine
            schedule = illustrious_scheduler.cosine_annealing_schedule(steps)

        # Convert to sigmas (this is simplified - actual implementation would need model sampling info)
        # In practice, you'd integrate this with ComfyUI's sigma generation
        sigmas = self.schedule_to_sigmas(schedule)

        print(
            f"Created {scheduler_type} schedule for Illustrious {model_version}: {len(sigmas)} sigmas"
        )

        return (sigmas,)

    def parse_content_analysis(self, content_text):
        """Parse content analysis string into pattern weights"""
        patterns = {}
        content_lower = content_text.lower()

        # Simple parsing - in practice, you'd use NLP or token analysis
        if any(
            term in content_lower for term in ["character", "portrait", "girl", "boy"]
        ):
            patterns["character_focus"] = 0.8
        if any(term in content_lower for term in ["complex", "detailed", "multiple"]):
            patterns["complex_scene"] = 0.7
        if any(term in content_lower for term in ["art", "style", "official"]):
            patterns["artistic_style"] = 0.6
        if any(
            term in content_lower
            for term in ["anime", "illustration", "masterpiece", "high_quality"]
        ):
            patterns["illustrious_specialty"] = 0.8
        if any(
            term in content_lower for term in ["lighting", "dramatic", "volumetric"]
        ):
            patterns["lighting_focus"] = 0.6
        if any(
            term in content_lower
            for term in ["intricate", "ornate", "extremely_detailed"]
        ):
            patterns["high_detail"] = 0.7

        return patterns

    def schedule_to_sigmas(self, schedule):
        """Convert a normalized 0..1 schedule to descending sigma values.

        Uses a log-space interpolation over a typical SDXL sigma range to better
        reflect perceptual noise scaling, and appends 0 at the end per ComfyUI.
        """
        # Typical SDXL sigma range; values align with ComfyUI defaults
        min_sigma, max_sigma = 0.0292, 14.6146

        # Ensure tensor on CPU float32 for math ops
        sched = schedule.detach().float().cpu().clamp(0.0, 1.0)

        # Map 1->max_sigma, 0->min_sigma using log interpolation
        # Interpolate in log space: sigma = exp( log(min) + t*(log(max)-log(min)) )
        log_min = math.log(min_sigma)
        log_max = math.log(max_sigma)
        # Invert because higher schedule value implies earlier, noisier step
        t = sched  # 0..1
        log_sigma = log_min + (1.0 - t) * (log_max - log_min)
        sigmas = torch.exp(log_sigma)

        # Enforce strictly descending order (safety)
        sigmas, _ = torch.sort(sigmas, descending=True)

        # Append terminal zero sigma
        sigmas = torch.cat([sigmas, torch.zeros(1, dtype=sigmas.dtype)])
        return sigmas
