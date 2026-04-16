"""
RAW Load node for ComfyUI-Darkroom.
Decodes camera RAW files into linear scene-referred IMAGE tensors via rawpy/LibRaw.
Outputs metadata as a RAW_METADATA custom type consumed by RAW Metadata Split.
"""

import os

import torch

from ..utils.raw_loader import (
    load_raw,
    DEMOSAIC_OPTIONS,
    COLORSPACE_OPTIONS,
    HIGHLIGHT_OPTIONS,
    WHITE_BALANCE_OPTIONS,
)


OUTPUT_MODE_OPTIONS = ["sRGB display", "Linear scene"]


class DarkroomRAWLoad:

    # Simple in-process decode cache — RAW decode is expensive (10s+ on a 100MP file).
    _cache = {}
    _cache_max = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_file": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to a camera RAW file "
                               "(.raf .cr3 .nef .arw .dng ...). Use the Browse button to pick one."
                }),
                "demosaic": (DEMOSAIC_OPTIONS, {
                    "default": DEMOSAIC_OPTIONS[0],
                    "tooltip": "Demosaic algorithm. Auto uses DHT for Bayer sensors and "
                               "LibRaw's 3-pass Markesteijn for Fuji X-Trans."
                }),
                "output_colorspace": (COLORSPACE_OPTIONS, {
                    "default": "Linear sRGB",
                    "tooltip": "Output colorspace primaries. Linear sRGB is the standard choice."
                }),
                "white_balance": (WHITE_BALANCE_OPTIONS, {
                    "default": "As shot",
                    "tooltip": "As shot = use the camera's recorded WB (what you saw in the viewfinder). "
                               "Auto = gray-world estimate. Daylight = fixed ~5500K."
                }),
                "highlight_mode": (HIGHLIGHT_OPTIONS, {
                    "default": "Rebuild (default)",
                    "tooltip": "How to handle blown highlights. Rebuild reconstructs partial-channel "
                               "data (Adobe default). Clip discards. Blend is a middle ground."
                }),
                "half_size": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Half-resolution decode. Skips one demosaic step per axis. "
                               "~10x faster. Great for iteration, disable for final output."
                }),
                "output_mode": (OUTPUT_MODE_OPTIONS, {
                    "default": "sRGB display",
                    "tooltip": "sRGB display = gamma-encoded — matches Camera Raw's default open view "
                               "(flatter than JPEG, same brightness). Drop-in compatible with all "
                               "Darkroom grading nodes. "
                               "Linear scene = pure linear HDR for ACES pipelines."
                }),
                "baseline_exposure": ("FLOAT", {
                    "default": 0.0, "min": -4.0, "max": 4.0, "step": 0.1,
                    "tooltip": "Extra exposure offset in stops, applied in linear space before the default "
                               "tone curve and sRGB encoding. The default is 0 — the baked-in Camera Raw-like "
                               "tone curve already lifts midtones to match ACR's default view. Use this to "
                               "brighten or darken from that starting point. Ignored when output_mode is Linear scene."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "RAW_METADATA")
    RETURN_NAMES = ("image", "metadata")
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/RAW"

    def execute(self, raw_file, demosaic, output_colorspace,
                white_balance, highlight_mode, half_size, output_mode,
                baseline_exposure=0.0):
        raw_file = (raw_file or "").strip().replace("\\", "/")
        if not raw_file:
            raise ValueError("[Darkroom RAW Load] no file path provided")
        if not os.path.isfile(raw_file):
            raise FileNotFoundError(f"[Darkroom RAW Load] file not found: {raw_file}")

        # Defensive: saved workflows from earlier node versions may pass None
        # or a missing value for the baseline_exposure widget.
        if baseline_exposure is None:
            baseline_exposure = 0.0
        baseline_exposure = float(baseline_exposure)

        mtime = os.path.getmtime(raw_file)
        key = (raw_file, mtime, demosaic, output_mode, output_colorspace,
               white_balance, highlight_mode, bool(half_size),
               round(baseline_exposure, 3))

        cached = DarkroomRAWLoad._cache.get(key)
        if cached is not None:
            img, meta = cached
            print(f"[Darkroom RAW Load] cache hit: {os.path.basename(raw_file)}")
        else:
            print(f"[Darkroom RAW Load] decoding {os.path.basename(raw_file)} "
                  f"(mode={output_mode}, exposure={baseline_exposure:+.1f}EV, "
                  f"demosaic={demosaic}, wb={white_balance}, half={half_size})")
            img, meta = load_raw(
                raw_file,
                demosaic=demosaic,
                colorspace=output_colorspace,
                white_balance=white_balance,
                highlight_mode=highlight_mode,
                output_mode=output_mode,
                baseline_exposure=baseline_exposure,
                half_size=half_size,
            )
            DarkroomRAWLoad._cache[key] = (img, meta)
            if len(DarkroomRAWLoad._cache) > DarkroomRAWLoad._cache_max:
                oldest = next(iter(DarkroomRAWLoad._cache))
                del DarkroomRAWLoad._cache[oldest]

            sensor = meta.get("sensor_type", "?")
            model = meta.get("camera_model", "?")
            print(f"[Darkroom RAW Load] decoded: {model} "
                  f"{meta['image_width']}x{meta['image_height']} ({sensor})")
            if meta.get("film_simulation"):
                print(f"[Darkroom RAW Load] film sim: {meta['film_simulation']}")

        tensor = torch.from_numpy(img).unsqueeze(0).contiguous()
        return (tensor, meta)


NODE_CLASS_MAPPINGS = {"DarkroomRAWLoad": DarkroomRAWLoad}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomRAWLoad": "RAW Load"}
