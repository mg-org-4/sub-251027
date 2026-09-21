"""
© 2026 blacksnowskill (BSS). All rights reserved.
Developed by: blacksnowskill (BSS)

core/attention_sage.py
SageAttention integration through the unified BSS attention router.
"""

import logging
import torch
from .attention_helper import register_bss_attention_router

logger = logging.getLogger("BSS_ANIMA_BOOSTER.sage")

SAGE_MODES = [
    "disabled",
    "auto",
]


def check_sageattention_available() -> bool:
    try:
        import sageattention  # noqa: F401
        return True
    except ImportError:
        return False


def apply_sage_attention_to_model(model, mode: str) -> tuple[any, bool]:
    """
    Patches model to use SageAttention via the unified attention router.
    """
    if mode == "disabled":
        return model, False

    if not check_sageattention_available():
        return model, False

    # Check CUDA device capability to prevent crashes on pre-Ampere GPUs (e.g., Turing RTX 20xx, Pascal, etc.)
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            if major < 8:
                logger.warning(
                    f"[SageAttention] SageAttention is auto-disabled for your GPU (compute capability {major}.{minor}, "
                    f"Turing/RTX 20xx or older). SageAttention officially requires Ampere (RTX 30xx, compute capability 8.0+) "
                    f"or newer GPUs to run stably. Falling back to native PyTorch Scaled Dot Product Attention (SDPA) "
                    f"to prevent Triton compilation crashes and performance degradation."
                )
                return model, False
        except Exception as device_err:
            logger.error(f"[SageAttention] Failed to query CUDA device capability: {device_err}")

    try:
        # Register the BSS unified router
        m = register_bss_attention_router(model)
        
        # Set the option flags in transformer_options
        m.model_options["transformer_options"]["bss_sage_mode"] = mode
        
        logger.info(f"[SageAttention] Configured mode '{mode}' in model options | Developed by blacksnowskill (BSS)")
        return m, True

    except Exception as e:
        logger.error(f"[SageAttention] Failed to apply: {e}", exc_info=True)
        return model, False

