"""
ComfyUI Realtime LoRA Trainer

Trains LoRAs on-the-fly from images during generation.
Supports Z-Image, FLUX, Wan models via AI-Toolkit.
Also supports SDXL and SD 1.5 via kohya sd-scripts.
"""

from .realtime_lora_trainer import RealtimeLoraTrainer, ApplyTrainedLora
from .sdxl_lora_trainer import SDXLLoraTrainer
from .sd15_lora_trainer import SD15LoraTrainer

# Web directory for JavaScript extensions
WEB_DIRECTORY = "./web/js"

NODE_CLASS_MAPPINGS = {
    "RealtimeLoraTrainer": RealtimeLoraTrainer,
    "ApplyTrainedLora": ApplyTrainedLora,
    "SDXLLoraTrainer": SDXLLoraTrainer,
    "SD15LoraTrainer": SD15LoraTrainer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RealtimeLoraTrainer": "Realtime LoRA Trainer",
    "ApplyTrainedLora": "Apply Trained LoRA",
    "SDXLLoraTrainer": "Realtime LoRA Trainer (SDXL - sd-scripts)",
    "SD15LoraTrainer": "Realtime LoRA Trainer (SD 1.5 - sd-scripts)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
