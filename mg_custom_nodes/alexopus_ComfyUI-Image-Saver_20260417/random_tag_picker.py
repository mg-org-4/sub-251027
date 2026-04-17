import csv
import os
import random


class RandomTagPicker:
    """Pick N random tags from a CSV file (first column) and join them with a delimiter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
                "count": ("INT", {"default": 5, "min": 1, "max": 1000, "step": 1}),
                "delimiter": ("STRING", {"default": ", ", "multiline": False}),
                "replace_underscore": ("BOOLEAN", {"default": False}),
                "trailing_comma": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "pick_random_tags"
    CATEGORY = "utils"

    def pick_random_tags(self, file_path: str, count: int, delimiter: str, replace_underscore: bool, trailing_comma: bool, seed: int) -> tuple[str]:
        with open(os.path.expanduser(file_path), newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if rows and rows[0][0].strip().lower() == "tag":
                rows = rows[1:]

        tags = [row[0] for row in rows if row and row[0].strip()]
        sample_size = min(count, len(tags))
        rng = random.Random(seed)
        selected = rng.sample(tags, sample_size)
        if replace_underscore:
            selected = [t.replace("_", " ") for t in selected]
        escaped = [t.replace("(", "\\(").replace(")", "\\)") for t in selected]
        result = delimiter.join(escaped)
        if trailing_comma:
            result += ","
        return (result,)
