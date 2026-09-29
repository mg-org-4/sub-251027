"""Content keys for already-selected decode entries; no model or storage mutation."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import threading
import weakref
from collections.abc import Mapping
from typing import Any, Iterator

import torch

CONTRACT = "h3-decode-helper-v1"
BLOCK_BYTES = 8 * 1024 * 1024


class CacheBypass(Exception):
    """The native decoder may run, but cache identity is not safely available."""


class ObjectTokens:
    """Monotonic process-local identities, without retaining model objects."""

    def __init__(self):
        self._refs = {}
        self._next = itertools.count(1)
        self._lock = threading.RLock()

    def token(self, obj):
        address = id(obj)
        with self._lock:
            record = self._refs.get(address)
            if record is not None and record[0]() is obj:
                return record[1]
            token = next(self._next)

            def expired(ref, address=address):
                with self._lock:
                    current = self._refs.get(address)
                    if current is not None and current[0] is ref:
                        self._refs.pop(address, None)

            try:
                ref = weakref.ref(obj, expired)
            except TypeError as exc:
                raise CacheBypass("object does not support a weak identity") from exc
            self._refs[address] = (ref, token)
            return token


TOKENS = ObjectTokens()


def tensor_blocks(tensor: torch.Tensor, block_bytes=BLOCK_BYTES) -> Iterator[memoryview]:
    """Yield C-order bytes, including BF16, with bounded contiguous scratch space."""
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise CacheBypass("only dense strided tensors are cacheable")
    if tensor.device.type != "cpu" or tensor.is_nested or tensor.is_quantized:
        raise CacheBypass("non-CPU/nested/quantized payload is delegated without cache")
    tensor = tensor.detach()
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    if tensor.is_neg():
        tensor = tensor.resolve_neg()
    size = tensor.numel() * tensor.element_size()
    if size == 0:
        return
    if size <= block_bytes:
        raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
        yield memoryview(raw).cast("B")
        return
    # All earlier dimensions are singleton, preserving logical C-order.
    for axis, length in enumerate(tensor.shape):
        if length > 1:
            per_slice = size // length
            step = max(1, block_bytes // max(1, per_slice))
            for start in range(0, length, step):
                yield from tensor_blocks(tensor.narrow(axis, start, min(step, length-start)), block_bytes)
            return
    raise CacheBypass("unrepresentable tensor block")


def tensor_digest(tensor, interrupt=None):
    h = hashlib.sha256()
    for block in tensor_blocks(tensor):
        if interrupt is not None:
            interrupt()
        h.update(block)
    return h.hexdigest()


def canonical(value: Any, interrupt=None, depth=0):
    """Reject opaque objects instead of using an unreliable repr or pointer key."""
    if depth > 16:
        raise CacheBypass("metadata nesting limit")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CacheBypass("non-finite metadata")
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, torch.Tensor):
        return ["tensor", str(value.dtype), list(value.shape), tensor_digest(value, interrupt)]
    if isinstance(value, Mapping):
        if len(value) > 256 or any(not isinstance(k, str) for k in value):
            raise CacheBypass("unsupported metadata mapping")
        return {k: canonical(value[k], interrupt, depth+1) for k in sorted(value)}
    if isinstance(value, (tuple, list)) and len(value) <= 4096:
        return [canonical(x, interrupt, depth+1) for x in value]
    raise CacheBypass(f"opaque metadata: {type(value).__name__}")


def selected_latent(samples, stream):
    latent = samples["samples"]
    if getattr(latent, "is_nested", False):
        parts = latent.unbind()
        latent = parts[0 if stream == "video" else -1]
    if not isinstance(latent, torch.Tensor):
        raise CacheBypass("unknown latent payload")
    return latent


def _callable_token(function):
    function = getattr(function, "__func__", function)
    return TOKENS.token(function)


def _empty_option_tree(value):
    if value is None:
        return True
    if isinstance(value, Mapping):
        return all(_empty_option_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_empty_option_tree(v) for v in value)
    return False


def vae_signature(vae, stream, delegate):
    """Supported native H3 inference VAE only; unknown wrappers still decode normally.

    This is not a model-file hash. Unsupported in-place .data changes require
    reset_token or a reloaded VAE. No weights are copied or read onto the CPU.
    """
    model = getattr(vae, "first_stage_model", None)
    patcher = getattr(vae, "patcher", None)
    if type(vae).__module__ != "comfy.sd" or not type(model).__module__.startswith("comfy.ldm.minimax."):
        raise CacheBypass("unrecognized VAE identity; native decode only")
    if patcher is None or not hasattr(patcher, "patches_uuid"):
        raise CacheBypass("VAE patch version unavailable")
    if bool(getattr(model, "training", True)):
        raise CacheBypass("training-mode VAE")
    if "decode" in vars(vae) or "decode" in vars(model):
        raise CacheBypass("instance-patched decoder")
    if getattr(patcher, "object_patches", None):
        raise CacheBypass("VAE object patches require cache bypass")

    if not _empty_option_tree(getattr(patcher, "model_options", None)):
        raise CacheBypass("non-default VAE patcher options")
    if any(getattr(patcher, field, None) for field in ("forced_hooks", "hook_patches")):
        raise CacheBypass("VAE hook state requires cache bypass")
    if any(getattr(part, "_forward_hooks", None) or getattr(part, "_forward_pre_hooks", None)
           for part in model.modules()):
        raise CacheBypass("VAE forward hooks require cache bypass")

    config = {}
    for name in ("vae_dtype", "device", "output_device", "audio_sample_rate", "audio_sample_rate_output",
                 "latent_channels", "disable_offload", "output_channels"):
        if hasattr(vae, name):
            config[name] = canonical(getattr(vae, name))
    # Native H3 tiling/temporal configuration is scalar, not model weights.
    model_config = {k: v for k, v in vars(model).items()
                    if not k.startswith("_") and type(v) in (str, bool, int, float, type(None))}
    # Model residency can replace buffer objects without changing their values.
    # Logical names/versions + Core patch UUID are used, not buffer pointers.
    # Reloading weights through raw .data (outside Core) requires reset_token.
    versions = []
    for name, value in itertools.chain(model.named_parameters(), model.named_buffers()):
        try:
            version = value._version
        except RuntimeError:  # inference tensors intentionally have no counter
            version = None
        versions.append((name, version, str(value.dtype), tuple(value.shape)))
    identity = (
        TOKENS.token(vae), TOKENS.token(model), TOKENS.token(patcher),
        str(patcher.patches_uuid), _callable_token(vae.decode), _callable_token(model.decode),
        _callable_token(delegate),
        _callable_token(vae.process_output) if callable(getattr(vae, "process_output", None)) else None,
        config, model_config, versions,
        str(torch.get_default_dtype()), torch.get_float32_matmul_precision(),
        bool(torch.backends.cuda.matmul.allow_tf32), bool(torch.backends.cudnn.allow_tf32),
        bool(torch.backends.cudnn.benchmark), bool(torch.backends.cudnn.deterministic),
        bool(torch.are_deterministic_algorithms_enabled()),
        bool(torch.is_grad_enabled()), bool(torch.is_inference_mode_enabled()),
        bool(torch.is_autocast_enabled("cpu")), str(torch.get_autocast_dtype("cpu")),
        bool(torch.is_autocast_enabled("cuda")), str(torch.get_autocast_dtype("cuda")),
    )
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest(), (weakref.ref(vae), weakref.ref(model), weakref.ref(patcher))


def make_key(samples, stream, signature, reset_token=0, interrupt=None):
    latent = selected_latent(samples, stream)
    # Only the chosen AV stream is hashed. Other decode-relevant metadata,
    # notably samples['sample_rate'], is included, conservatively, as well.
    metadata = {k: v for k, v in samples.items() if k != "samples"}
    payload = [CONTRACT, stream, signature, int(reset_token), str(latent.dtype), list(latent.shape),
               tensor_digest(latent, interrupt), canonical(metadata, interrupt)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
