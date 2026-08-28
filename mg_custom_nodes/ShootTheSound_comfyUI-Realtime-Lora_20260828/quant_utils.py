"""
Quantized-model detection helpers.

Strength scheduling in this pack uses ComfyUI's hook system
(``create_hook_lora`` + ``register_all_hook_patches``). At sample time ComfyUI
walks the model's weights via ``ModelPatcher.get_key_patches`` ->
``get_key_weight`` -> ``get_attr``, which assumes plain ``nn.Linear`` layers.
On GGUF / fp8-quantized diffusion models the Linear layers are different
objects (GGUF ``GGMLOps`` linears with a ``.temp`` recompute buffer, scaled-fp8
linears with a ``.weight_scale`` / ``.scale_weight``), so that walk raises
``AttributeError: 'Linear' object has no attribute 'temp' / 'weight_scale'`` or
hangs. See issues #26, #41, #51, #33.

``is_quantized_model`` lets the scheduling nodes detect this up front and fall
back to a flat-strength apply instead of crashing mid-sample.
"""

import torch


# float8 dtypes vary by torch build; collect whatever this build exposes.
_FP8_DTYPES = tuple(
    dt for dt in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    ) if dt is not None
)


def _diffusion_module(model_patcher):
    """Return the diffusion model nn.Module from a ModelPatcher (or None)."""
    inner = getattr(model_patcher, "model", None)
    if inner is None:
        return None
    dm = getattr(inner, "diffusion_model", None)
    return dm if dm is not None else inner


def is_quantized_model(model_patcher):
    """Best-effort detection of GGUF / fp8 quantized diffusion models.

    Returns a ``(is_quantized: bool, label: str)`` tuple. ``label`` is a short
    human-readable tag ("GGUF", "fp8", "fp8 (scaled)") for messaging, or "".

    Detection is intentionally conservative and defensive — any failure returns
    ``(False, "")`` so a detection bug can never break a normal (bf16/fp16) run.
    """
    try:
        target = _diffusion_module(model_patcher)
        if target is None:
            return (False, "")

        # Module-level signals: GGUF ops and scaled-fp8 linears.
        for module in target.modules():
            cls = type(module).__name__.lower()
            if "ggml" in cls or "gguf" in cls:
                return (True, "GGUF")
            # scaled-fp8 linears carry a per-tensor weight scale
            if hasattr(module, "scale_weight") or hasattr(module, "weight_scale"):
                return (True, "fp8 (scaled)")

        # Parameter-level signals: GGMLTensor-wrapped weights or fp8 dtypes.
        # Sample a bounded number of params — no need to scan every tensor.
        for i, p in enumerate(target.parameters()):
            tname = type(p).__name__.lower()
            if "ggml" in tname or hasattr(p, "tensor_type"):
                return (True, "GGUF")
            if _FP8_DTYPES and p.dtype in _FP8_DTYPES:
                return (True, "fp8")
            if i >= 400:
                break

        return (False, "")
    except Exception:
        return (False, "")
