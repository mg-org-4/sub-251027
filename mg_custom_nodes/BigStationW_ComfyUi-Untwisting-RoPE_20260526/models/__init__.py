"""Auto-discovered model architecture adapters for UntwistingRoPE.

The top-level node code imports this package only. Every model-specific name,
shape check, module lookup, and optional preprocessing routine should live in an adapter module in this folder.

Drop-in adapter rule:
    Create ``models/new_model.py`` and expose at least:
        ARCHITECTURE = "new_model"
        DISPLAY_NAME = "New Model"
        def matches_model(model_info: dict) -> bool: ...
        def find_diffusion_model(model_patcher): ...

Optional hooks are discovered automatically when present:
        default_runtime_cfg(dm=None) -> dict
        prepare_reference_conditioning(...)
        patch_attention_modules(dm, stats, helpers=None)
        is_joint_attention(module) -> bool
        is_attention_name(name, min_layer=0, max_layer=999) -> bool
        uses_reference_branch_kv() -> bool

No edit to this registry is needed for a new adapter module, unless two adapters
match the same model and you want to resolve ordering by setting PRIORITY.
Higher PRIORITY values are tried first.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Any

CONFIG_KEY = "untwisting_rope"

_IMPORT_ERRORS: dict[str, str] = {}


def _is_adapter_module(module: ModuleType) -> bool:
    """Return True when a discovered module exposes the minimum adapter API."""
    return callable(getattr(module, "find_diffusion_model", None))


def _adapter_priority(module: ModuleType) -> int:
    try:
        return int(getattr(module, "PRIORITY", 0))
    except Exception:
        return 0


def _load_adapters() -> tuple[ModuleType, ...]:
    """Import every non-private Python module in this package as an adapter."""
    adapters: list[ModuleType] = []
    _IMPORT_ERRORS.clear()

    package_path = __path__  # type: ignore[name-defined]
    package_name = __name__

    for module_info in pkgutil.iter_modules(package_path):
        name = module_info.name
        if name.startswith("_"):
            continue
        if module_info.ispkg:
            continue

        qualified_name = f"{package_name}.{name}"
        try:
            module = importlib.import_module(qualified_name)
        except Exception as exc:
            _IMPORT_ERRORS[name] = repr(exc)
            continue

        if _is_adapter_module(module):
            adapters.append(module)

    adapters.sort(
        key=lambda module: (
            _adapter_priority(module),
            str(getattr(module, "ARCHITECTURE", module.__name__)),
        ),
        reverse=True,
    )
    return tuple(adapters)


# Loaded once at package import. A freshly added adapter file is picked up on the
# next Python/ComfyUI reload, which is the normal plugin-development workflow.
REGISTERED_ADAPTERS = _load_adapters()


def refresh() -> tuple[ModuleType, ...]:
    """Reload the adapter list after files are added during a live session."""
    global REGISTERED_ADAPTERS
    REGISTERED_ADAPTERS = _load_adapters()
    return REGISTERED_ADAPTERS


def identify(model_patcher: Any, model_info: dict[str, Any] | None = None) -> ModuleType:
    """Return the first discovered adapter that recognizes the ComfyUI model wrapper."""
    model_info = model_info or {}

    for adapter in REGISTERED_ADAPTERS:
        matches = getattr(adapter, "matches_model", None)
        if callable(matches):
            try:
                if matches(model_info):
                    return adapter
            except Exception:
                pass

    for adapter in REGISTERED_ADAPTERS:
        try:
            adapter.find_diffusion_model(model_patcher)
            return adapter
        except Exception:
            continue

    details = ""
    if _IMPORT_ERRORS:
        details = " Adapter import errors: " + "; ".join(
            f"{name}: {error}" for name, error in sorted(_IMPORT_ERRORS.items())
        )
    raise RuntimeError(f"Could not resolve a supported diffusion architecture.{details}")


def adapter_label(adapter: Any) -> str:
    return str(getattr(adapter, "DISPLAY_NAME", getattr(adapter, "ARCHITECTURE", type(adapter).__name__)))


def adapter_key(adapter: Any) -> str:
    return str(getattr(adapter, "ARCHITECTURE", type(adapter).__name__))


__all__ = [
    "CONFIG_KEY",
    "REGISTERED_ADAPTERS",
    "refresh",
    "identify",
    "adapter_key",
    "adapter_label",
]
