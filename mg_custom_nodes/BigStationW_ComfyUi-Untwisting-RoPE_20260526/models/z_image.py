from __future__ import annotations

from typing import Any

ARCHITECTURE = "zimage_nextdit"
DISPLAY_NAME = "Z-Image/NextDiT"
CONFIG_KEY = "untwisting_rope"

# Paths commonly used by ComfyUI model patcher wrappers to reach the diffusion object.
DIFFUSION_ATTR_PATHS = (
    "diffusion_model",
    "model.diffusion_model",
    "model.model.diffusion_model",
    "inner_model.diffusion_model",
    "model.inner_model.diffusion_model",
)

SEARCH_CHILD_ATTRS = ("model", "inner_model", "diffusion_model", "unet", "wrapped")


def looks_like_diffusion_model(obj: Any) -> bool:
    """Return True for the Z-Image/NextDiT diffusion object used by this patch."""
    return (
        obj is not None
        and hasattr(obj, "patchify_and_embed")
        and hasattr(obj, "layers")
    )


def _roots(model_patcher: Any) -> list[Any]:
    roots: list[Any] = []
    if hasattr(model_patcher, "model"):
        roots.append(model_patcher.model)
    roots.append(model_patcher)
    return roots


def _get_attr_path(root: Any, attr_path: str) -> tuple[Any, bool]:
    obj = root
    for part in attr_path.split("."):
        if not hasattr(obj, part):
            return None, False
        obj = getattr(obj, part)
    return obj, True


def find_diffusion_model(model_patcher: Any) -> Any:
    """Best-effort lookup for the Z-Image/NextDiT diffusion model inside ComfyUI wrappers."""
    roots = _roots(model_patcher)
    for root in roots:
        for path in DIFFUSION_ATTR_PATHS:
            obj, ok = _get_attr_path(root, path)
            if ok and looks_like_diffusion_model(obj):
                return obj

    seen: set[int] = set()
    stack = roots[:]
    while stack and len(seen) < 256:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if looks_like_diffusion_model(obj):
            return obj
        for name in SEARCH_CHILD_ATTRS:
            if hasattr(obj, name):
                try:
                    stack.append(getattr(obj, name))
                except Exception:
                    pass
    raise RuntimeError("Could not find Z-Image/NextDiT diffusion model.")


def is_joint_attention(module: Any) -> bool:
    """Return True for the Z-Image/NextDiT joint-attention module shape."""
    return (
        hasattr(module, "qkv") and hasattr(module, "out")
        and hasattr(module, "q_norm") and hasattr(module, "k_norm")
        and hasattr(module, "n_local_heads") and hasattr(module, "n_local_kv_heads")
        and hasattr(module, "head_dim")
        and callable(getattr(module, "forward", None))
    )


def is_main_layers_attention_name(name: str, min_layer: int = 0, max_layer: int = 29) -> bool:
    """Z-Image/NextDiT attention modules are named layers.N.attention."""
    parts = str(name).split(".")
    if len(parts) != 3:
        return False
    if parts[0] != "layers" or parts[2] != "attention":
        return False
    try:
        idx = int(parts[1])
    except Exception:
        return False
    return int(min_layer) <= idx <= int(max_layer)


def default_runtime_cfg(dm: Any | None = None) -> dict[str, Any]:
    """Architecture-specific cfg fields merged into the main runtime cfg."""
    return {"architecture": ARCHITECTURE}


def matches_model(model_info: dict[str, Any]) -> bool:
    """This adapter is normally selected by structural diffusion-model lookup."""
    return False


def is_attention_name(name: str, min_layer: int = 0, max_layer: int = 29) -> bool:
    return is_main_layers_attention_name(name, min_layer, max_layer)


def prepare_reference_conditioning(ref_conditioning: Any, dm: Any, device: Any, dtype: Any, stats: Any, label: str = "", helpers: dict[str, Any] | None = None):
    return ref_conditioning, "not-applicable"


def patch_attention_modules(dm: Any, stats: Any, helpers: dict[str, Any] | None = None):
    helpers = helpers or {}
    if callable(helpers.get("patch_context_refiner_mask_modules")):
        helpers["patch_context_refiner_mask_modules"](dm, stats)
    if callable(helpers.get("patch_patchify_and_embed")):
        helpers["patch_patchify_and_embed"](dm, stats)
    if callable(helpers.get("patch_joint_attention_modules")):
        return helpers["patch_joint_attention_modules"](dm, stats)
    return None


def uses_reference_branch_kv() -> bool:
    return True
