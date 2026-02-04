import os
import torch


def _version_tuple(version_str):
    version_str = version_str.split("+", 1)[0]
    parts = version_str.split(".")
    nums = []
    for part in parts:
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits == "":
            break
        nums.append(int(digits))
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])


def _is_safetensors_path(path):
    if not isinstance(path, (str, os.PathLike)):
        return False
    return str(path).lower().endswith(".safetensors")


def safe_torch_load(path, *args, **kwargs):
    if not _is_safetensors_path(path):
        if _version_tuple(torch.__version__) < (2, 6, 0):
            raise RuntimeError(
                "torch>=2.6 is required to use torch.load on non-safetensors files "
                "due to CVE-2025-32434. Upgrade torch or use safetensors."
            )
    return torch.load(path, *args, **kwargs)
