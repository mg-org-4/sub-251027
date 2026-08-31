"""
© 2026 blacksnowskill (BSS). All rights reserved.
Developed by: blacksnowskill (BSS)

core/compile_utils.py
torch.compile utilities for Anima/Cosmos DiT model blocks.

Detects the block type (Anima uses .blocks, Flux uses .double_blocks/.single_blocks)
and applies set_torch_compile_wrapper from ComfyUI's torch_helpers.
"""

import logging
import torch

logger = logging.getLogger("ANIMA_BOOSTER.compile")

# Known block container attribute names across different architectures
_KNOWN_BLOCK_ATTRS = [
    "blocks",           # Anima (MiniTrainDIT / Cosmos Predict-2)
    "double_blocks",    # Flux
    "single_blocks",    # Flux
    "transformer_blocks",  # Wan2.1, LTX-V
    "layers",           # some older models
    "visual_transformer_blocks",
    "text_transformer_blocks",
]


def detect_and_compile_blocks(
    model,
    backend: str = "inductor",
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool | None = None,
    dynamo_cache_limit: int = 64,
) -> object:
    """
    Detects transformer block attributes in the diffusion model and applies
    torch.compile to each block individually.

    Individual block compilation is preferred over full-model compilation because:
    1. Faster initial compilation (one block at a time)
    2. Less prone to graph breaks from dynamic control flow
    3. LoRA and other patches can still work on non-compiled parts
    4. More predictable VRAM usage

    Returns cloned compiled model.
    """
    try:
        from comfy_api.torch_helpers import set_torch_compile_wrapper
    except ImportError:
        logger.error("[Compile] comfy_api.torch_helpers not available — "
                     "please update ComfyUI")
        return model

    # Inductor backend requires Triton to compile models on GPU.
    # We check for its presence beforehand to prevent runtime ModuleNotFoundError crashes.
    if backend == "inductor" and torch.cuda.is_available():
        try:
            import triton
        except ImportError:
            logger.error(
                "[Compile] 'triton' module is missing! PyTorch's 'inductor' JIT backend "
                "requires the Triton compiler for CUDA acceleration. "
                "To prevent ComfyUI from crashing during KSampler execution, "
                "torch.compile has been SAFELY DISABLED. "
                "Tip: Install 'triton-windows' or look for 'ComfyUI-Sage-EasyInstall' in ComfyUI Manager."
            )
            return model

    m = model.clone()
    diffusion_model = m.get_model_object("diffusion_model")

    torch._dynamo.config.cache_size_limit = dynamo_cache_limit

    compile_keys = []
    for attr in _KNOWN_BLOCK_ATTRS:
        if hasattr(diffusion_model, attr):
            blocks = getattr(diffusion_model, attr)
            for i in range(len(blocks)):
                compile_keys.append(f"diffusion_model.{attr}.{i}")

    if not compile_keys:
        logger.warning("[Compile] No known block attributes found. "
                       "Compiling full diffusion_model (may be slow or unstable)")
        compile_keys = ["diffusion_model"]

    logger.info(f"[Compile] Compiling {len(compile_keys)} blocks | "
                f"backend={backend} | mode={mode} | fullgraph={fullgraph}")

    try:
        set_torch_compile_wrapper(
            model=m,
            keys=compile_keys,
            backend=backend,
            mode=mode,
            dynamic=dynamic,
            fullgraph=fullgraph,
        )
        logger.info(f"[Compile] Successfully set up torch.compile wrapper "
                    f"for {len(compile_keys)} components")
    except Exception as e:
        logger.error(f"[Compile] Failed to apply torch.compile: {e}")
        return model

    return m
