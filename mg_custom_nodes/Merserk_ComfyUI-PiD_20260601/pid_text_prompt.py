from __future__ import annotations


class PiDTextPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "caption")
    FUNCTION = "build"
    CATEGORY = "PiD"

    def build(self, prompt: str = ""):
        text = prompt or ""
        return (text, text)


NODE_CLASS_MAPPINGS = {
    "PiDTextPrompt": PiDTextPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDTextPrompt": "PiD Text Prompt",
}
