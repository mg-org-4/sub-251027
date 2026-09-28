"""One independent helper; existing Continuum node IDs and schemas are untouched."""
from __future__ import annotations

import json
import logging
import threading

from .service import DecodeService, reraise_control
from .store import MiB, GiB


class H3DecodeCacheHelper:
    CATEGORY = "MiniMax H3/Continuum/Helpers"
    FUNCTION = "decode"
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "report")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True, False)
    DESCRIPTION = (
        "Optional V3.8X2 helper: reuse complete physical VAE decode entries. "
        "Auto stores video in a private process-local disk cache, audio in bounded RAM. "
        "Connect the unchanged Assembly Plan directly to Finalize. No model unload or sampling changes."
    )
    SEARCH_ALIASES = ["Decode Cache Helper", "H3 cached decode", "video audio cache"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_mode": (["Auto", "RAM", "Off"], {"default": "Auto", "tooltip":
                    "Auto: disk-backed video + small RAM audio. RAM: bounded RAM only. Off: native decoding, clear helper cache."}),
                "ram_budget_mb": ("INT", {"default": 256, "min": 0, "max": 8192, "step": 128, "advanced": True,
                    "tooltip": "Additional private RAM retention limit (MiB), not total process memory. Low headroom disables RAM retention."}),
                "disk_budget_gb": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1, "advanced": True,
                    "tooltip": "Process-local cache quota in GiB. Set 0 for no cache file writes. Auto does not reuse files after restart."}),
                "reset_token": ("INT", {"default": 0, "min": 0, "max": 2147483647, "advanced": True,
                    "tooltip": "Increase to discard this helper's cache, especially after unsupported in-place VAE weight edits."}),
            },
            "optional": {
                "video_samples": ("LATENT", {"tooltip": "Continuum video_latents LIST; each entry is decoded whole, before trim/seam."}),
                "video_vae": ("VAE", {"tooltip": "Same native Video VAE as the existing VAE Decode node."}),
                "audio_samples": ("LATENT", {"tooltip": "Continuum audio_latents LIST; optional, decoded independently."}),
                "audio_vae": ("VAE", {"tooltip": "Same native Audio VAE as VAE Decode Audio. Required only with audio_samples."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Review must still execute; only this helper's content cache skips decode.
        return float("nan")

    def __init__(self):
        self.service = DecodeService()
        self._lock = threading.RLock()

    @staticmethod
    def _scalar(value, name):
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"{name} expects one value, not a list batch")
            return value[0]
        return value

    @staticmethod
    def _list(value):
        return value if isinstance(value, list) else ([] if value is None else [value])

    def _stream(self, samples, vaes, stream, reset_token):
        entries = self._list(samples)
        vaes = self._list(vaes)
        if not entries:
            return [], []
        outputs, events = [], []
        for index, entry in enumerate(entries):
            if entry is None:
                # Same missing-output position; do not duplicate the previous entry.
                outputs.append(None)
                events.append({"stream": stream, "entry": index, "status": "unconnected"})
                continue
            vae = vaes[min(index, len(vaes)-1)] if vaes else None
            result, event = self.service.decode(entry, vae, stream, reset_token)
            outputs.append(result)
            events.append({"entry": index, **event})
        return outputs, events

    def decode(self, cache_mode, ram_budget_mb, disk_budget_gb, reset_token,
               video_samples=None, video_vae=None, audio_samples=None, audio_vae=None):
        with self._lock:
            mode = self._scalar(cache_mode, "cache_mode")
            ram = int(self._scalar(ram_budget_mb, "ram_budget_mb"))
            disk = int(self._scalar(disk_budget_gb, "disk_budget_gb"))
            reset = int(self._scalar(reset_token, "reset_token"))
            if mode not in ("Auto", "RAM", "Off") or ram < 0 or disk < 0:
                raise ValueError("invalid cache settings")
            config_error = None
            try:
                self.service.store.configure(mode, ram*MiB, disk*GiB, reset)
            except Exception as exc:
                reraise_control(exc)
                self.service.store.mode = "Off"
                config_error = str(exc)[:160]
                logging.getLogger("h3_decode_cache_helper").warning("Cache unavailable; native decode: %s", exc)
            images, video_events = self._stream(video_samples, video_vae, "video", reset)
            audio, audio_events = self._stream(audio_samples, audio_vae, "audio", reset)
            try:
                stats = self.service.store.stats()
            except Exception as exc:
                reraise_control(exc)
                stats = {"diagnostic_error": str(exc)[:160]}
            report = {"helper_version": "3.8.3", "mode": mode,
                      "video": video_events, "audio": audio_events, "cache": stats}
            if config_error:
                report["cache_disabled_reason"] = config_error
            return images, audio, json.dumps(report, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {"H3DecodeCacheHelper": H3DecodeCacheHelper}
NODE_DISPLAY_NAME_MAPPINGS = {"H3DecodeCacheHelper": "Decode Cache Helper"}
