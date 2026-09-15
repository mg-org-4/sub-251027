"""Accelerator detection from build metadata alone.

Nothing here calls `torch.cuda.is_available()`, `torch.xpu.is_available()` or
any other probe that would spin up a device context: picking a wheel must not
cost a CUDA init, and must still work on a machine whose driver is broken.
"""

import os
import platform
import re
import sys
from typing import NamedTuple

from . import scheme


class Accelerator(NamedTuple):
    family: str
    version: "tuple[int, int] | None"
    torch_device: str

    def describe(self):
        if self.version is None:
            return self.family
        return f"{self.family} {self.version[0]}.{self.version[1]}"


def _torch():
    module = sys.modules.get("torch")
    if module is not None:
        return module
    try:
        import torch

        return torch
    except Exception:
        return None


def _version_pair(text):
    parts = re.findall(r"\d+", str(text or ""))
    if not parts:
        return None
    if len(parts) == 1:
        return (int(parts[0]), 0)
    return (int(parts[0]), int(parts[1]))


def _apple_silicon():
    return sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")


def _platform_only():
    if _apple_silicon():
        return Accelerator("metal", None, "mps")
    return Accelerator("cpu", None, "cpu")


def detect() -> Accelerator:
    torch = _torch()
    if torch is None:
        return _platform_only()

    version = getattr(torch, "version", None)

    # ROCm builds populate torch.version.cuda too, so HIP has to be read first.
    hip = getattr(version, "hip", None)
    if hip:
        family = "rocm" if sys.platform.startswith("linux") else "hip"
        return Accelerator(family, _version_pair(hip), "cuda")

    cuda = getattr(version, "cuda", None)
    if cuda:
        return Accelerator("cuda", _version_pair(cuda), "cuda")

    if getattr(version, "xpu", None):
        return Accelerator("xpu", None, "xpu")

    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and mps.is_built() and _apple_silicon():
        return Accelerator("metal", None, "mps")

    if "torch_directml" in sys.modules:
        return Accelerator("vulkan", None, "cpu")

    return _platform_only()


def forced_variant():
    value = os.environ.get(scheme.ENV_VARIANT, "").strip()
    if not value:
        return None
    return "" if value.lower() == "cpu" else value


def torch_devices(accelerator: Accelerator = None):
    """Device choices to offer in the UI, best first, always including cpu."""
    accelerator = accelerator or detect()
    if accelerator.torch_device == "cpu":
        return ["cpu"]
    return [accelerator.torch_device, "cpu"]
