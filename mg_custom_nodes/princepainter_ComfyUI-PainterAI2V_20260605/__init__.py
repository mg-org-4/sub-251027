from .PainterAI2V import PainterAI2V
from .PainterAV2V import PainterAV2V

__version__ = "1.0.0"

NODE_CLASS_MAPPINGS = {
    "PainterAI2V": PainterAI2V,
    "PainterAV2V": PainterAV2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterAI2V": "Painter AI2V",
    "PainterAV2V": "Painter AV2V",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
