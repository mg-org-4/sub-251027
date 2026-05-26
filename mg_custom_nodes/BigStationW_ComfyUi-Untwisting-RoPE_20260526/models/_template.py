"""Template for a drop-in UntwistingRoPE model adapter.

Copy this file to ``models/new_model.py`` and remove the leading underscore.
The registry auto-loads non-private Python files in this folder on ComfyUI reload.
"""

from __future__ import annotations

from typing import Any

ARCHITECTURE = "new_model"
DISPLAY_NAME = "New Model"
PRIORITY = 0  # Higher values are tried first when multiple adapters can match.


def matches_model(model_info: dict[str, Any]) -> bool:
    """Fast metadata check. Return True when model_info identifies this architecture."""
    # Example:
    # return str(model_info.get("image_model", "")).lower() == "new_model"
    return False


def looks_like_diffusion_model(obj: Any) -> bool:
    """Structural check for the actual diffusion object."""
    # Example:
    # return obj is not None and hasattr(obj, "blocks") and hasattr(obj, "patchify")
    return False


def find_diffusion_model(model_patcher: Any) -> Any:
    """Return the underlying diffusion model or raise RuntimeError."""
    roots = []
    if hasattr(model_patcher, "model"):
        roots.append(model_patcher.model)
    roots.append(model_patcher)

    attr_paths = (
        "diffusion_model",
        "model.diffusion_model",
        "model.model.diffusion_model",
        "inner_model.diffusion_model",
        "model.inner_model.diffusion_model",
    )

    for root in roots:
        for attr_path in attr_paths:
            obj = root
            ok = True
            for part in attr_path.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok and looks_like_diffusion_model(obj):
                return obj

    seen: set[int] = set()
    stack = list(roots)
    while stack and len(seen) < 256:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if looks_like_diffusion_model(obj):
            return obj
        for name in ("model", "inner_model", "diffusion_model", "unet", "wrapped"):
            if hasattr(obj, name):
                try:
                    stack.append(getattr(obj, name))
                except Exception:
                    pass

    raise RuntimeError(f"Could not find {DISPLAY_NAME} diffusion model.")


def default_runtime_cfg(dm: Any | None = None) -> dict[str, Any]:
    """Optional: add architecture-specific defaults to the generic runtime config."""
    return {}


def is_joint_attention(module: Any) -> bool:
    """Optional: identify attention modules when using the generic attention patch."""
    return False


def is_attention_name(name: str, min_layer: int = 0, max_layer: int = 999) -> bool:
    """Optional: identify target module names when using the generic attention patch."""
    return False


def prepare_reference_conditioning(
    ref_conditioning: Any,
    dm: Any,
    device: Any,
    dtype: Any,
    stats: Any,
    label: str = "",
    helpers: dict[str, Any] | None = None,
):
    """Optional: convert reference conditioning into the architecture's expected format."""
    return ref_conditioning, "not-applicable"


def patch_attention_modules(dm: Any, stats: Any, helpers: dict[str, Any] | None = None):
    """Optional: fully custom attention patching for this architecture.

    Omit this function to let the top-level machinery use the generic patch path.
    """
    if helpers:
        helpers["patch_context_refiner_mask_modules"](dm, stats)
        helpers["patch_patchify_and_embed"](dm, stats)
        helpers["patch_joint_attention_modules"](dm, stats)


def uses_reference_branch_kv() -> bool:
    """Optional: used only for debug logging."""
    return False
