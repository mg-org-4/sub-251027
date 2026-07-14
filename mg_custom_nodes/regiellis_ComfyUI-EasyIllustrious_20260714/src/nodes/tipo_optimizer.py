import json
import random
import re
from typing import List, Tuple, Dict, Any


class TIPOPromptOptimizer:
    """
    Text-to-Image Prompt Optimizer (TIPO)
    - Generates K prompt variants via reordering, weighting, and flavor anchors
    - Ranks using text-only heuristics; optional CLIP ranking can be added later
    """

    CATEGORY = "Easy Illustrious / Utils / Prompt Optimization"
    FUNCTION = "optimize"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("optimized_prompt", "candidate_prompts", "diagnostics")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Base prompt to optimize (comma-separated tags recommended).",
                    },
                ),
                "candidates": (
                    "INT",
                    {
                        "default": 8,
                        "min": 3,
                        "max": 32,
                        "step": 1,
                        "tooltip": "How many prompt variants to generate and rank.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "tooltip": "Random seed (0 = auto).",
                    },
                ),
                "enable_synonyms": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Allow synonym swaps for variation."},
                ),
                "flavor": (
                    ["balanced", "vibrant", "soft", "natural"],
                    {"default": "balanced", "tooltip": "Flavor bias for ranking."},
                ),
                "max_length": (
                    "INT",
                    {
                        "default": 320,
                        "min": 80,
                        "max": 800,
                        "step": 10,
                        "tooltip": "Soft cap for final prompt length (characters).",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Terms to avoid (downweighted/removed).",
                    },
                ),
                "style_anchors": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "balanced colors, clean lineart, consistent shading",
                        "tooltip": "Anchor tags to encourage in the optimized prompt.",
                    },
                ),
                # Future: "clip" (CLIP), "ref_image" (IMAGE)
            },
        }

    def optimize(
        self,
        prompt: str,
        candidates: int,
        seed: int,
        enable_synonyms: bool,
        flavor: str,
        max_length: int,
        negative_prompt: str = "",
        style_anchors: str = "",
    ):
        rng = random.Random(seed or 1)
        anchors = self._split_tags(style_anchors, strict=False)
        base_tokens = self._split_tags(prompt, strict=True)
        base_tokens = self._dedupe(base_tokens)

        flavor_boosts = {
            "balanced": ["balanced colors", "clean lineart", "consistent shading"],
            "vibrant": ["vibrant colors", "dynamic lighting", "saturation control"],
            "soft": ["soft lighting", "pastel tones", "gentle shading"],
            "natural": ["natural color grading", "neutral tones", "realistic lighting"],
        }
        synonym_map = {
            "cinematic": ["filmic", "movie-like"],
            "dramatic": ["intense", "powerful"],
            "soft lighting": ["diffused light", "gentle light"],
            "vibrant": ["colorful", "lively"],
            "detailed": ["intricate", "elaborate"],
        }

        banned = set(
            t.strip().lower()
            for t in self._split_tags(negative_prompt or "", strict=False)
        )

        cands: List[Tuple[float, str, Dict[str, Any]]] = []
        for _ in range(max(3, int(candidates))):
            tokens = self._perturb(
                rng,
                base_tokens,
                flavor,
                flavor_boosts,
                enable_synonyms,
                synonym_map,
                banned,
            )
            cand_text = ", ".join(tokens)
            score, why = self._score(
                tokens, anchors, flavor, flavor_boosts, banned, max_length
            )
            cands.append((score, cand_text, why))

        cands.sort(key=lambda x: x[0], reverse=True)
        best_score, best_text, best_why = cands[0]
        if len(best_text) > max_length:
            best_text = best_text[:max_length].rstrip(", ")

        diagnostics = {
            "selected_score": round(best_score, 4),
            "selected_why": best_why,
            "candidate_count": len(cands),
            "top3": [
                {"score": round(s, 4), "prompt": p[:max_length], "why": w}
                for s, p, w in cands[:3]
            ],
            "flavor": flavor,
        }

        return (
            best_text,
            json.dumps(
                [{"score": s, "prompt": p, "why": w} for s, p, w in cands], indent=2
            ),
            json.dumps(diagnostics, indent=2),
        )

    # ---------------- internals ----------------
    def _split_tags(self, text: str, strict: bool) -> List[str]:
        parts = [t.strip() for t in (text or "").split(",")]
        parts = [p for p in parts if p]
        if strict:
            tokens = []
            for p in parts:
                tokens.extend([t for t in re.split(r"[;|/]+", p) if t.strip()])
            parts = [t.strip() for t in tokens if t.strip()]
        return parts

    def _dedupe(self, tokens: List[str]) -> List[str]:
        seen, out = set(), []
        for t in tokens:
            key = re.sub(r"\s+", " ", t.lower())
            if key not in seen:
                seen.add(key)
                out.append(t)
        return out

    def _perturb(
        self,
        rng: random.Random,
        tokens: List[str],
        flavor: str,
        flavor_boosts: Dict[str, List[str]],
        enable_synonyms: bool,
        synonym_map: Dict[str, List[str]],
        banned: set,
    ) -> List[str]:
        cand = tokens[:]
        if cand:
            head, tail = cand[:1], cand[1:]
            rng.shuffle(tail)
            cand = head + tail
        for _ in range(min(3, max(1, len(cand) // 6))):
            i = rng.randrange(len(cand)) if cand else 0
            j = rng.randrange(len(cand)) if cand else 0
            if i < len(cand) and j < len(cand):
                cand[i], cand[j] = cand[j], cand[i]

        if enable_synonyms and rng.random() < 0.5:
            idx = rng.randrange(len(cand)) if cand else 0
            if idx < len(cand):
                key = cand[idx].lower()
                for k, vs in synonym_map.items():
                    if k in key and vs:
                        cand[idx] = vs[rng.randrange(len(vs))]
                        break

        for idx in range(min(2, len(cand))):
            if rng.random() < 0.6 and "(" not in cand[idx]:
                cand[idx] = f"({cand[idx]}:1.1)"

        boosts = flavor_boosts.get(flavor, [])
        for b in boosts:
            if rng.random() < 0.6 and b not in cand:
                cand.append(b)

        cand = [t for t in cand if t.strip().lower() not in banned]
        return self._dedupe(cand)

    def _score(
        self,
        tokens: List[str],
        anchors: List[str],
        flavor: str,
        flavor_boosts: Dict[str, List[str]],
        banned: set,
        max_len: int,
    ) -> Tuple[float, Dict[str, Any]]:
        text = ", ".join(tokens)
        score = 0.0
        why = {}

        # Anchor overlap score
        anchor_hits = 0
        alist = [a.lower() for a in anchors]
        for t in tokens:
            tl = t.lower()
            if any(a in tl for a in alist):
                anchor_hits += 1
        score += anchor_hits * 0.9
        why["anchor_hits"] = anchor_hits

        # Flavor presence
        flavor_hits = sum(
            1
            for b in flavor_boosts.get(flavor, [])
            if any(b.lower() in t.lower() for t in tokens)
        )
        score += flavor_hits * 1.1
        why["flavor_hits"] = flavor_hits

        # Penalties
        uniq = len(set(t.lower() for t in tokens))
        dup_pen = max(0, len(tokens) - uniq) * 0.6
        score -= dup_pen
        why["dup_penalty"] = round(dup_pen, 3)

        length_pen = max(0, len(text) - max_len) / 40.0
        score -= length_pen
        why["length_penalty"] = round(length_pen, 3)

        ban_pen = sum(1 for t in tokens if t.strip().lower() in banned) * 0.8
        score -= ban_pen
        why["banned_penalty"] = round(ban_pen, 3)

        # Context bonuses
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
        ctx_bonus = min(1, ambience_hits) * 0.4 + min(1, tw_hits) * 0.4
        score += ctx_bonus
        why["context_bonus"] = round(ctx_bonus, 3)

        illustrious_bonus = (
            sum(
                1
                for k in ["clean lineart", "balanced colors", "natural color grading"]
                if any(k in t.lower() for t in tokens)
            )
            * 0.5
        )
        score += illustrious_bonus
        why["illustrious_bonus"] = round(illustrious_bonus, 3)

        return score, why
