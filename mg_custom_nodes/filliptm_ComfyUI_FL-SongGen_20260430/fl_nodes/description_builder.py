"""
FL Song Gen Description Builder Node.
Helps users build style description strings from components.
"""

from typing import Tuple


class FL_SongGen_DescriptionBuilder:
    """
    Build style description string from individual components.

    This helper node makes it easy to construct a description string
    that controls the musical style of generated songs.

    Output format: "female, warm, pop, emotional, piano and drums, the bpm is 120"
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "build_description"
    CATEGORY = "FL Song Gen"

    # Available options for each category
    VOICE_TYPES = ["female", "male", "mixed", "none"]
    TIMBRES = ["bright", "dark", "warm", "soft", "none"]
    GENRES = [
        "pop", "rock", "hip hop", "jazz", "blues", "electronic",
        "folk", "r&b", "metal", "reggae", "classical", "country",
        "indie", "soul", "funk", "disco", "none"
    ]
    EMOTIONS = [
        "happy", "sad", "romantic", "melancholic", "uplifting",
        "emotional", "passionate", "introspective", "energetic",
        "calm", "intense", "nostalgic", "hopeful", "none"
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice_type": (
                    cls.VOICE_TYPES,
                    {
                        "default": "female",
                        "tooltip": "Vocal type/gender"
                    }
                ),
                "timbre": (
                    cls.TIMBRES,
                    {
                        "default": "warm",
                        "tooltip": "Vocal timbre/tone quality"
                    }
                ),
                "genre": (
                    cls.GENRES,
                    {
                        "default": "pop",
                        "tooltip": "Musical genre"
                    }
                ),
                "emotion": (
                    cls.EMOTIONS,
                    {
                        "default": "emotional",
                        "tooltip": "Emotional tone of the song"
                    }
                ),
            },
            "optional": {
                "instruments": (
                    "STRING",
                    {
                        "default": "piano and drums",
                        "tooltip": "Primary instruments (e.g., 'piano and drums', 'guitar and strings')"
                    }
                ),
                "bpm": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 300,
                        "step": 5,
                        "tooltip": "Beats per minute (0 = auto/not specified)"
                    }
                ),
                "custom_tags": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Additional style tags (comma-separated)"
                    }
                ),
                "raw_description": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "OR: Provide complete description directly (overrides above fields)"
                    }
                ),
            }
        }

    def build_description(
        self,
        voice_type: str,
        timbre: str,
        genre: str,
        emotion: str,
        instruments: str = "piano and drums",
        bpm: int = 0,
        custom_tags: str = "",
        raw_description: str = "",
    ) -> Tuple[str]:
        """
        Build description string from components.

        Args:
            voice_type: Vocal type (female, male, mixed)
            timbre: Tonal quality (bright, dark, warm, soft)
            genre: Musical genre
            emotion: Emotional tone
            instruments: Primary instruments
            bpm: Beats per minute (0 for auto)
            custom_tags: Additional comma-separated tags
            raw_description: Pre-built description (overrides other fields)

        Returns:
            Description string for conditioning
        """
        # If raw description provided, use it
        if raw_description.strip():
            print(f"[FL SongGen Description] Using raw description input")
            return (raw_description.strip(),)

        parts = []

        # Add main attributes (skip if "none")
        if voice_type != "none":
            parts.append(voice_type)

        if timbre != "none":
            parts.append(timbre)

        if genre != "none":
            parts.append(genre)

        if emotion != "none":
            parts.append(emotion)

        # Add instruments
        if instruments.strip():
            parts.append(instruments.strip())

        # Add BPM if specified
        if bpm > 0:
            parts.append(f"the bpm is {bpm}")

        # Add custom tags
        if custom_tags.strip():
            custom = [t.strip() for t in custom_tags.split(",") if t.strip()]
            parts.extend(custom)

        result = ", ".join(parts)

        print(f"[FL SongGen Description] Built: {result}")

        return (result,)
