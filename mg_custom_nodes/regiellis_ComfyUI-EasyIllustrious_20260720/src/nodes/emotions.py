import re
from typing import Dict, List, Tuple


class IllustriousEmotions:
    """
    Emotions/Expressions injector node.
    Inserts a selected facial expression or emote into the prompt chain,
    optionally weighted and wrapped in parentheses, with chain insert support.
    """

    CHAIN_INSERT_TOKEN = "[EN122112_CHAIN]"

    # Grouped vocabularies (deduplicated)
    EMOTION_GROUPS: Dict[str, List[str]] = {
        "Emotions": [
            "angry",
            "anger vein",
            "annoyed",
            "clenched teeth",
            "scowl",
            "blush",
            "blush stickers",
            "embarrassed",
            "full-face blush",
            "nose blush",
            "bored",
            "closed eyes",
            "confused",
            "crazy",
            "despair",
            "determined",
            "disappointed",
            "disdain",
            "disgust",
            "drunk",
            "envy",
            "excited",
            "exhausted",
            "expressionless",
            "facepalm",
            "flustered",
            "frustrated",
            "furrowed brow",
            "grimace",
            "guilt",
            "happy",
            "kubrick stare",
            "lonely",
            "nervous",
            "nosebleed",
            "one eye closed (winking)",
            "round mouth",
            "open mouth",
            "parted lips",
            "pain",
            "pout",
            "raised eyebrow",
            "raised inner eyebrows",
            "rape face",
            "rolling eyes",
            "sad",
            "depressed",
            "frown",
            "gloom (expression)",
            "tears",
            "scared",
            "panicking",
            "worried",
            "serious",
            "sigh",
            "sleepy",
            "sulking",
            "surprised",
            "thinking",
            "pensive",
            "v-shaped eyebrows",
            "wince",
        ],
        "Sexual": [
            "afterglow",
            "ahegao",
            "aroused",
            "in heat",
            "naughty face",
            "ohhoai",
            "seductive smile",
            "torogao",
        ],
        "Smile": [
            ":d (:D, open-mouthed smile)",
            "crazy smile",
            "evil smile",
            "fingersmile",
            "forced smile",
            "glasgow smile",
            "grin",
            "evil grin",
            "light smile",
            "sad smile",
            "seductive smile",
            "smile (smile with mouth close)",
            "stifled laugh",
        ],
        "Smug": [
            "doyagao (self-satisfaction / smugness)",
            "smirk",
            "smug",
            "troll face",
        ],
        "Surprised / Scared / Sad": [
            "color drain",
            "depressed",
            "despair",
            "gloom (expression)",
            "horrified",
            "screaming",
            "sobbing",
            "traumatized",
            "turn pale",
            "wavy mouth",
        ],
        "Emotes": [
            ";)",
            ":d",
            ";d",
            "xd",
            "d:",
            ":3",
            ";3",
            "x3",
            "3:",
            "0w0",
            "uwu",
            ":p",
            ";p",
            ":q",
            ";q",
            ">:)",
            ">:(",
            ":t",
            ":i",
            ":/",
            ":|",
            "x mouth",
            ":c",
            "c:",
            ":<",
            ";<",
            "diamond mouth",
            ":>",
            ":>=",
            ":o",
            ";o",
            "o3o",
            ">3<",
            "o_o",
            "0_0",
            "|_|",
            "._.",
            "solid circle eyes",
            "heart-shaped eyes",
            "^_^",
            "\(^o^)/",
            "^q^",
            ">_<",
            ">o<",
            "@_@",
            ">_@",
            "+_+",
            "+_-",
            "=_=",
            "t_t",
            "<o>_<o>",
            "<|>_<|>",
        ],
    }

    # Flatten unique tokens for convenience in UIs that can’t be dynamic
    EMOTION_TOKENS: List[str] = sorted(
        {
            token.strip()
            for group in EMOTION_GROUPS.values()
            for token in group
            if token and token.strip()
        }
    )

    @classmethod
    def INPUT_TYPES(cls) -> Dict:
        return {
            "optional": {
                "prefix": (
                    "STRING",
                    {"forceInput": True, "tooltip": "Prefix text (chain-aware)"},
                ),
                "suffix": (
                    "STRING",
                    {"forceInput": True, "tooltip": "Suffix text (after content)"},
                ),
            },
            "required": {
                "Emotion Group": (
                    ["-"] + list(cls.EMOTION_GROUPS.keys()),
                    {"default": "-", "tooltip": "Optional grouping for reference."},
                ),
                "Emotion/Expression": (
                    ["-"] + cls.EMOTION_TOKENS,
                    {"default": "-", "tooltip": "Select an expression/emote to insert."},
                ),
                "Weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05},
                ),
                "Wrap With Parentheses": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Output as (tag:weight)."},
                ),
                "Add Chain Insert Point": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": f"Insert '{IllustriousEmotions.CHAIN_INSERT_TOKEN}' so next node can inject its content.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PROMPT",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "construct"
    CATEGORY = "Easy Illustrious / Generators"

    def _fmt(self, token: str, weight: float, wrap: bool) -> str:
        token = token.strip()
        if not token:
            return ""
        if wrap:
            return f"({token}:{weight:.2f})"
        return f"{token}:{weight:.2f}"

    def _join(self, *parts: str) -> str:
        items = [p.strip().strip(',') for p in parts if p and p.strip()]
        s = ", ".join(items)
        s = re.sub(r"\s*,\s*", ", ", s).strip(", ").strip()
        return s

    def construct(self, **kwargs) -> Tuple[str]:
        prefix = kwargs.get("prefix", "").strip()
        suffix = kwargs.get("suffix", "").strip()
        token = kwargs.get("Emotion/Expression", "-")
        group = kwargs.get("Emotion Group", "-")
        weight = float(kwargs.get("Weight", 1.0) or 1.0)
        wrap = bool(kwargs.get("Wrap With Parentheses", True))
        add_chain = bool(kwargs.get("Add Chain Insert Point", False))

        # If no explicit token but a group is provided, pick randomly from that group
        if token == "-" and group and group != "-":
            try:
                options = self.EMOTION_GROUPS.get(group, [])
                if options:
                    import random
                    token = random.choice(options)
            except Exception:
                token = "-"

        if token == "-":
            # No change; optionally just add chain token
            out = self._join(prefix, suffix)
            return (out if out else " ",)

        expr = self._fmt(token, weight, wrap)

        if self.CHAIN_INSERT_TOKEN in prefix:
            head, tail = prefix.split(self.CHAIN_INSERT_TOKEN, 1)
            base = self._join(head, expr, tail)
        else:
            base = self._join(prefix, expr)
            base = self._join(base, suffix)

        if add_chain:
            base = self._join(base, self.CHAIN_INSERT_TOKEN)

        return (base if base else " ",)


# Export the token list for other nodes (e.g., Smart Prompt UI)
EMOTION_TOKENS = IllustriousEmotions.EMOTION_TOKENS
EMOTION_GROUPS = IllustriousEmotions.EMOTION_GROUPS
