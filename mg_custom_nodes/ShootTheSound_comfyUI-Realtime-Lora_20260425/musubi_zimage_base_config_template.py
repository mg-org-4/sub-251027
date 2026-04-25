"""
Musubi Tuner Z-Image Base LoRA Training Config Template

Generates TOML dataset configuration files for zimage_train_network.py
This is for the non-distilled Z-Image Base model (vs Z-Image Turbo).
"""

import os

# Import shared dataset config generation from existing Z-Image module
from .musubi_zimage_config_template import generate_dataset_config, save_config


# VRAM mode presets for Z-Image Base (Musubi Tuner)
# Same structure as Z-Image Turbo but may have different defaults
# Max blocks_to_swap is 28 for Z-Image models
MUSUBI_ZIMAGE_BASE_VRAM_PRESETS = {
    "Max (1256px)": {
        "optimizer": "adamw8bit",
        "mixed_precision": "bf16",
        "batch_size": 1,
        "gradient_checkpointing": False,
        "fp8_scaled": False,
        "fp8_llm": False,
        "blocks_to_swap": 0,
        "resolution": 1256,
    },
    "Max (1256px) fp8": {
        "optimizer": "adamw8bit",
        "mixed_precision": "bf16",
        "batch_size": 1,
        "gradient_checkpointing": False,
        "fp8_scaled": True,
        "fp8_llm": True,
        "blocks_to_swap": 0,
        "resolution": 1256,
    },
    "Max (1256px) fp8 offload": {
        "optimizer": "adamw8bit",
        "mixed_precision": "bf16",
        "batch_size": 1,
        "gradient_checkpointing": True,
        "fp8_scaled": True,
        "fp8_llm": True,
        "blocks_to_swap": 14,
        "resolution": 1256,
    },
    "Medium (1024px)": {
        "optimizer": "adamw8bit",
        "mixed_precision": "bf16",
        "batch_size": 1,
        "gradient_checkpointing": True,
        "fp8_scaled": False,
        "fp8_llm": False,
        "blocks_to_swap": 0,
        "resolution": 1024,
    },
    "Medium (1024px) fp8": {
        "optimizer": "adamw8bit",
        "mixed_precision": "bf16",
        "batch_size": 1,
        "gradient_checkpointing": True,
        "fp8_scaled": True,
        "fp8_llm": True,
        "blocks_to_swap": 0,
        "resolution": 1024,
    },
    "Medium (1024px) fp8 offload": {
        "optimizer": "adamw8bit",
        "mixed_precision": "bf16",
        "batch_size": 1,
        "gradient_checkpointing": True,
        "fp8_scaled": True,
        "fp8_llm": True,
        "blocks_to_swap": 14,
        "resolution": 1024,
    },
    "Low (768px)": {
        "optimizer": "adamw8bit",
        "mixed_precision": "bf16",
        "batch_size": 1,
        "gradient_checkpointing": True,
        "fp8_scaled": True,
        "fp8_llm": True,
        "blocks_to_swap": 14,
        "resolution": 768,
    },
    "Min (512px)": {
        "optimizer": "adamw8bit",
        "mixed_precision": "bf16",
        "batch_size": 1,
        "gradient_checkpointing": True,
        "fp8_scaled": True,
        "fp8_llm": True,
        "blocks_to_swap": 28,
        "resolution": 512,
    },
}
