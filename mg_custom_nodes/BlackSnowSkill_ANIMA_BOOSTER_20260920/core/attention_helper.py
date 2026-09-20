"""
© 2026 blacksnowskill (BSS). All rights reserved.
Developed by: blacksnowskill (BSS)

core/attention_helper.py
Unified attention router to safely intercept and redirect attention calls
without mutating the underlying model blocks in memory (which causes artifact issues).
"""

import logging
import torch
from comfy.ldm.modules.attention import attention_sage, attention_pytorch

logger = logging.getLogger("BSS_ANIMA_BOOSTER.helper")


def bss_optimized_attention_override(
    func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs
):
    """
    Unified attention router registered as model_options["transformer_options"]["optimized_attention_override"].
    
    Routes attention requests to:
      1. SageAttention if bss_sage_mode is active.
      2. Default ComfyUI attention otherwise (func).
    """
    transformer_options = kwargs.get("transformer_options", {})

    # 1. Try Sage Attention if configured
    sage_mode = transformer_options.get("bss_sage_mode", "disabled")
    if sage_mode != "disabled":
        try:
            return attention_sage(
                q=q,
                k=k,
                v=v,
                heads=heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs
            )
        except Exception as e:
            logger.error(f"[BSS_ANIMA_BOOSTER] SageAttention routing failed: {e}. Falling back to default.", exc_info=False)

    # 3. Default ComfyUI attention fallback
    return func(
        q,
        k,
        v,
        heads,
        mask=mask,
        attn_precision=attn_precision,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        **kwargs
    )


def register_bss_attention_router(model) -> object:
    """
    Safely registers the BSS optimized attention router into model options
    without mutating the model block layers in memory.
    """
    m = model.clone()
    if "transformer_options" not in m.model_options:
        m.model_options["transformer_options"] = {}
    
    # Register the router as ComfyUI's standard attention override hook
    m.model_options["transformer_options"]["optimized_attention_override"] = bss_optimized_attention_override
    return m
