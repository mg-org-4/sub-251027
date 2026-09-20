"""Native decoder delegation. Cache errors fail open; decoder/cancellation errors do not."""
from __future__ import annotations

import importlib
import json
import logging
import time


from .identity import make_key, vae_signature
from .store import DecodeStore

LOG = logging.getLogger("h3_decode_cache_helper")


def interrupt_check():
    try:
        mm = importlib.import_module("comfy.model_management")
    except ImportError:
        return
    mm.throw_exception_if_processing_interrupted()


def reraise_control(exc):
    # ComfyUI's interruption is an Exception, unlike KeyboardInterrupt.
    # Never convert an interrupted Queue into another Core Decode attempt.
    if any(cls.__name__ == "InterruptProcessingException" for cls in type(exc).__mro__):
        raise exc


def native_delegate(stream):
    if stream == "video":
        core = importlib.import_module("nodes").VAEDecode
        method = core.decode

        def execute(vae, samples):
            return core().decode(vae=vae, samples=samples)[0]
        return method, execute
    helper = importlib.import_module("comfy_extras.nodes_audio").vae_decode_audio
    return helper, lambda vae, samples: helper(vae, samples)


class DecodeService:
    def __init__(self, store=None, delegate_resolver=native_delegate,
                 signature_fn=vae_signature, interrupt=interrupt_check):
        self.store = store or DecodeStore(interrupt=interrupt)
        self.delegate_resolver = delegate_resolver
        self.signature_fn = signature_fn
        self.interrupt = interrupt

    def decode(self, samples, vae, stream, reset_token=0):
        self.interrupt()
        delegate_identity, native = self.delegate_resolver(stream)
        event = {"stream": stream, "status": "miss", "hash_s": 0.0,
                 "read_s": 0.0, "decode_s": 0.0, "write_s": 0.0}
        signature = key = guards = None
        epoch = self.store.epoch
        if self.store.mode != "Off":
            start = time.perf_counter()
            try:
                signature, guards = self.signature_fn(vae, stream, delegate_identity)
                key = make_key(samples, stream, signature, reset_token, self.interrupt)
            except Exception as exc:
                reraise_control(exc)
                event["status"] = "bypass"
                event["reason"] = str(exc)[:160]
            event["hash_s"] = time.perf_counter() - start
        else:
            event["status"] = "off"
        if key is not None:
            start = time.perf_counter()
            try:
                cached = self.store.get(key)
                if cached is not None:
                    tensor, extra = cached
                    event.update(status="hit", read_s=time.perf_counter()-start)
                    return (tensor if stream == "video" else {"waveform": tensor, **extra}), event
            except Exception as exc:
                reraise_control(exc)
                event["cache_read_error"] = str(exc)[:160]
                LOG.warning("Decode cache read failed; using native decoder: %s", exc)
            event["read_s"] = time.perf_counter() - start

        # Outside cache exception handlers. No blind retry of OOM, invalid
        # inputs, VAE errors, or interrupted Sampling/Decode.
        self.interrupt()
        start = time.perf_counter()
        decoded = native(vae, samples)
        event["decode_s"] = time.perf_counter() - start
        self.interrupt()
        if key is not None:
            start = time.perf_counter()
            try:
                after, _ = self.signature_fn(vae, stream, delegate_identity)
                if after != signature:
                    event["store_skipped"] = "VAE identity changed during native decode"
                else:
                    if stream == "video":
                        tensor, extra = decoded, {}
                    else:
                        tensor = decoded["waveform"]
                        extra = {k: v for k, v in decoded.items() if k != "waveform"}
                        # Only JSON-compatible metadata is retained; store defensively copies it.
                        json.dumps(extra, allow_nan=False)
                    replacement = self.store.put(key, tensor, stream, extra, guards, epoch)
                    event["stored"] = self.store.contains(key)
                    if replacement is not None:
                        tensor, extra = replacement
                        decoded = tensor if stream == "video" else {"waveform": tensor, **extra}
            except Exception as exc:
                reraise_control(exc)
                event["cache_write_error"] = str(exc)[:160]
                LOG.warning("Decode cache write failed; returning native result: %s", exc)
            event["write_s"] = time.perf_counter() - start
        return decoded, event
