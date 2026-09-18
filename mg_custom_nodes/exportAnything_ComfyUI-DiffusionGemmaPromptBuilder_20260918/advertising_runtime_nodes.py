# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

"""Self-contained runtime memory barriers for Advertisement workflows.

The established music-video workflows use convenience nodes from several
third-party packs to release text encoders and VAEs between expensive stages.
Advertisement workflows keep the same useful scheduling boundary without
depending on those unrelated packs.
"""

from __future__ import annotations

import gc
from typing import Any


CATEGORY = "prompt/diffusiongemma/advertising"
POLICIES = (
    "Unload models + clear CUDA cache",
    "Unload models + soft cache",
    "Passthrough (no release)",
)


class _AnyType(str):
    """ComfyUI wildcard socket compatible with arbitrary pass-through values."""

    def __ne__(self, value: object) -> bool:
        return False


ANY_TYPE = _AnyType("*")


class AdvertisementMemoryBarrier:
    """Pass a value through after an explicit, observable model-memory release."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY_TYPE,),
                "policy": (list(POLICIES), {"default": POLICIES[0]}),
            }
        }

    RETURN_TYPES = (ANY_TYPE, "STRING")
    RETURN_NAMES = ("value", "memory_status")
    FUNCTION = "release"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Advertisement-owned scheduling barrier. It preserves its input exactly while releasing loaded "
        "ComfyUI models before the next high-memory stage; no third-party cleanup pack is required."
    )

    def release(self, value: Any, policy: str):
        selected = str(policy or "").strip()
        if selected not in POLICIES:
            raise ValueError("Advertisement memory policy is unsupported.")
        if selected == POLICIES[2]:
            return value, "Advertisement memory barrier passed through without releasing models."

        try:
            import comfy.model_management as model_management
        except ImportError as exc:  # pragma: no cover - exercised only outside ComfyUI.
            raise RuntimeError(
                "Advertisement memory release requires the ComfyUI runtime."
            ) from exc

        model_management.unload_all_models()
        model_management.soft_empty_cache(True)
        gc.collect()

        cuda_cleared = False
        if selected == POLICIES[0]:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    cuda_cleared = True
            except (ImportError, RuntimeError):
                cuda_cleared = False

        suffix = " CUDA cache cleared." if cuda_cleared else " Soft cache cleared."
        return value, "Advertisement memory barrier released all loaded models." + suffix


NODE_CLASS_MAPPINGS = {
    "DiffusionGemmaAdvertisementMemoryBarrier": AdvertisementMemoryBarrier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionGemmaAdvertisementMemoryBarrier": "DiffusionGemma Advertisement Memory Barrier",
}


__all__ = [
    "AdvertisementMemoryBarrier",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
