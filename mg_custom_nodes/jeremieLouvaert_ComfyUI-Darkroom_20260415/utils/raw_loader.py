"""
RAW file loading utilities for ComfyUI-Darkroom.
Wraps rawpy / LibRaw to decode camera RAW files into linear scene-referred data.
Metadata comes from the embedded JPEG's EXIF, with Fujifilm film simulation
parsed from the maker note at tag 0x1401.

This is separate from utils/raw.py, which is Wave 2 Camera Raw Tools math
(white balance, parametric tone masks, etc.). This module is the file I/O
layer for Wave 6 RAW Pipeline.
"""

import io
import os

import numpy as np

from .color import linear_to_srgb
from .grading import cubic_spline_curve


# "Camera Raw-like default" tone curve, applied in linear light before sRGB encoding.
# Approximates the combination of Adobe Standard profile's value lift (from the DCP
# LookTable — which we don't yet parse) + Camera Raw's default Medium Contrast user
# tone curve (applied globally on top of every profile). Lifts shadows and quarter-
# tones aggressively, preserves midtone character, gentle on highlights.
#
# Session 2 will replace this with a real DCP LookTable application (per-camera).
# For now this is camera-agnostic and gives ~70% of the visual match to Camera Raw.
CAMERA_RAW_DEFAULT_CURVE = [
    (0.000, 0.000),
    (0.045, 0.110),   # deep shadow lift
    (0.180, 0.340),   # middle gray pushed to ~34% (matches ACR display baseline)
    (0.500, 0.680),   # strong midtone lift
    (0.800, 0.920),   # gentle highlight rolloff
    (1.000, 1.000),
]

try:
    import rawpy
    _HAS_RAWPY = True
except ImportError:
    _HAS_RAWPY = False

try:
    from PIL import Image as _PILImage
    from PIL import ExifTags as _PILExifTags
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import exifread
    _HAS_EXIFREAD = True
except ImportError:
    _HAS_EXIFREAD = False


RAW_EXTENSIONS = (
    ".cr3", ".cr2", ".crw",             # Canon
    ".nef", ".nrw",                     # Nikon
    ".arw", ".sr2", ".srf",             # Sony
    ".raf",                             # Fujifilm
    ".rw2", ".raw",                     # Panasonic
    ".dng", ".rwl",                     # Leica / generic DNG
    ".3fr", ".fff",                     # Hasselblad
    ".orf",                             # Olympus / OM
    ".pef", ".ptx",                     # Pentax / Ricoh
    ".x3f",                             # Sigma
    ".iiq", ".mos", ".eip",             # Phase One / Leaf
)


# Fujifilm FilmMode maker note (tag 0x1401) → human name.
# Source: ExifTool's Image::ExifTool::FujiFilm FilmMode table.
FUJI_FILM_MODE = {
    0x000: "F0 / Standard (Provia)",
    0x100: "F1 / Studio Portrait",
    0x110: "F1a / Enhanced Saturation",
    0x120: "F1b / Smooth Skin (Astia)",
    0x130: "F1c / Increased Sharpness",
    0x200: "F2 / Fujichrome (Velvia)",
    0x300: "F3 / Studio Portrait Ex",
    0x400: "F4 / Velvia",
    0x500: "Pro Neg Std",
    0x501: "Pro Neg Hi",
    0x600: "Classic Chrome",
    0x700: "Eterna",
    0x800: "Classic Negative",
    0x900: "Bleach Bypass",
    0xa00: "Nostalgic Neg.",
    0xb00: "Reala ACE",
}


# Only the demosaic algorithms present in the pip wheel LGPL build of LibRaw.
# AMAZE / LMMSE / MODIFIED_AHD / AFD / VCD / VCD_MODIFIED_AHD need GPL packs.
_DEMOSAIC_MAP = {
    "Auto (best for sensor)": None,           # DHT for Bayer, LibRaw-auto for X-Trans
    "DHT (high quality)": "DHT",
    "AAHD (high quality)": "AAHD",
    "AHD (Adobe default)": "AHD",
    "DCB (balanced)": "DCB",
    "PPG (fast)": "PPG",
    "VNG (fast)": "VNG",
    "Linear (fastest)": "LINEAR",
}
DEMOSAIC_OPTIONS = list(_DEMOSAIC_MAP.keys())


_COLORSPACE_MAP = {
    "Linear sRGB": "sRGB",
    "Linear Rec.2020": "Rec2020",
    "Linear ProPhoto": "ProPhoto",
    "Wide Gamut RGB": "Wide",
    "XYZ": "XYZ",
}
COLORSPACE_OPTIONS = list(_COLORSPACE_MAP.keys())


_HIGHLIGHT_MAP = {
    "Clip": "Clip",
    "Ignore": "Ignore",
    "Blend": "Blend",
    "Rebuild (default)": "ReconstructDefault",
}
HIGHLIGHT_OPTIONS = list(_HIGHLIGHT_MAP.keys())


_WHITE_BALANCE_MAP = {
    "As shot": "camera",
    "Auto (gray world)": "auto",
    "Daylight": "daylight",
}
WHITE_BALANCE_OPTIONS = list(_WHITE_BALANCE_MAP.keys())


def is_xtrans(raw_handle):
    """X-Trans sensors have a 6x6 CFA pattern. Bayer is 2x2."""
    pat = getattr(raw_handle, "raw_pattern", None)
    if pat is None:
        return False
    return tuple(pat.shape) == (6, 6)


def _rawpy_demosaic(name, xtrans):
    if not _HAS_RAWPY or xtrans:
        return None
    key = _DEMOSAIC_MAP.get(name) or "DHT"
    return getattr(rawpy.DemosaicAlgorithm, key)


def _rawpy_colorspace(name):
    if not _HAS_RAWPY:
        return None
    key = _COLORSPACE_MAP.get(name, "sRGB")
    cs = rawpy.ColorSpace
    return {
        "sRGB": cs.sRGB,
        "Rec2020": getattr(cs, "rec2020", cs.sRGB),
        "ProPhoto": cs.ProPhoto,
        "Wide": cs.Wide,
        "XYZ": cs.XYZ,
    }.get(key, cs.sRGB)


def _rawpy_highlight(name):
    if not _HAS_RAWPY:
        return None
    key = _HIGHLIGHT_MAP.get(name, "ReconstructDefault")
    return getattr(rawpy.HighlightMode, key, rawpy.HighlightMode.ReconstructDefault)


def _extract_thumb_exif(raw_handle):
    """Return (PIL_named_exif, exifread_tags) from the embedded JPEG, or ({}, {})."""
    if not _HAS_RAWPY:
        return {}, {}
    try:
        thumb = raw_handle.extract_thumb()
    except Exception:
        return {}, {}
    if thumb.format != rawpy.ThumbFormat.JPEG:
        return {}, {}

    pil_named = {}
    if _HAS_PIL:
        try:
            pil_img = _PILImage.open(io.BytesIO(thumb.data))
            raw_exif = pil_img._getexif() or {}
            pil_named = {_PILExifTags.TAGS.get(k, k): v for k, v in raw_exif.items()}
        except Exception as e:
            print(f"[Darkroom RAW] PIL EXIF parse failed: {e}")

    exifread_tags = {}
    if _HAS_EXIFREAD:
        try:
            exifread_tags = exifread.process_file(io.BytesIO(thumb.data), details=True)
        except Exception as e:
            print(f"[Darkroom RAW] exifread parse failed: {e}")

    return pil_named, exifread_tags


def _safe_int(v, default=0):
    if v is None:
        return default
    if isinstance(v, (tuple, list)) and v:
        v = v[0]
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_float(v, default=0.0):
    if v is None:
        return default
    if isinstance(v, (tuple, list)) and v:
        v = v[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _shutter_string(seconds):
    if seconds <= 0:
        return ""
    if seconds >= 1.0:
        return f"{seconds:.1f}s"
    return f"1/{round(1.0 / seconds)}"


def extract_metadata(path, raw_handle):
    """
    Combine rawpy sensor info with embedded-JPEG EXIF + Fuji maker notes.
    """
    meta = {
        "file_path": path.replace("\\", "/"),
        "file_size": os.path.getsize(path),
        "sensor_type": "X-Trans" if is_xtrans(raw_handle) else "Bayer",
        "raw_height": int(raw_handle.sizes.raw_height),
        "raw_width": int(raw_handle.sizes.raw_width),
        "image_height": int(raw_handle.sizes.height),
        "image_width": int(raw_handle.sizes.width),
        "flip": int(raw_handle.sizes.flip),
        "camera_whitebalance": [float(x) for x in raw_handle.camera_whitebalance],
        "daylight_whitebalance": [float(x) for x in raw_handle.daylight_whitebalance],
        "black_level": [int(x) for x in raw_handle.black_level_per_channel],
        "white_level": int(raw_handle.white_level),
        "camera_make": "",
        "camera_model": "",
        "lens_make": "",
        "lens_model": "",
        "iso": 0,
        "aperture": 0.0,
        "shutter_seconds": 0.0,
        "shutter_string": "",
        "focal_length": 0.0,
        "focal_length_35mm": 0.0,
        "datetime": "",
        "orientation": 1,
        "film_simulation": "",
    }

    pil_named, exifread_tags = _extract_thumb_exif(raw_handle)

    if pil_named:
        meta["camera_make"] = str(pil_named.get("Make", "")).strip().strip("\x00")
        meta["camera_model"] = str(pil_named.get("Model", "")).strip().strip("\x00")
        meta["lens_make"] = str(pil_named.get("LensMake", "")).strip().strip("\x00")
        meta["lens_model"] = str(pil_named.get("LensModel", "")).strip().strip("\x00")
        meta["iso"] = _safe_int(pil_named.get("ISOSpeedRatings"))
        meta["aperture"] = _safe_float(pil_named.get("FNumber"))
        meta["shutter_seconds"] = _safe_float(pil_named.get("ExposureTime"))
        meta["shutter_string"] = _shutter_string(meta["shutter_seconds"])
        meta["focal_length"] = _safe_float(pil_named.get("FocalLength"))
        meta["focal_length_35mm"] = _safe_float(pil_named.get("FocalLengthIn35mmFilm"))
        meta["datetime"] = str(pil_named.get("DateTimeOriginal", "")).strip().strip("\x00")
        meta["orientation"] = _safe_int(pil_named.get("Orientation"), 1)

    if "FUJI" in meta["camera_make"].upper() and exifread_tags:
        tag = exifread_tags.get("MakerNote Tag 0x1401")
        if tag is not None:
            try:
                vals = tag.values
                code = int(vals[0] if isinstance(vals, (list, tuple)) else vals)
                meta["film_simulation"] = FUJI_FILM_MODE.get(
                    code, f"Unknown (0x{code:x})"
                )
            except Exception as e:
                print(f"[Darkroom RAW] Fuji film sim decode failed: {e}")

    return meta


def load_raw(path,
             demosaic="Auto (best for sensor)",
             colorspace="Linear sRGB",
             white_balance="As shot",
             highlight_mode="Rebuild (default)",
             output_mode="sRGB display",
             baseline_exposure=1.5,
             half_size=False):
    """
    Decode a RAW file to (image_float32_hwc in [0,1], metadata_dict).

    output_mode:
        "sRGB display"    — linear decode + baseline_exposure (in stops) +
                            sRGB gamma encoding + hard clip to [0,1]. Matches
                            Camera Raw's default view on open (flatter than
                            JPEG, same brightness). Drop-in compatible with
                            every Darkroom grading node.
        "Linear scene"    — pure linear light, no gamma, no exposure boost,
                            no clip. For ACES pipelines. Mean around 0.05 for
                            a typical outdoor shot — looks very dark in a raw
                            preview and needs a tonemap to view.

    baseline_exposure is applied in linear space, before sRGB encoding.
    Ignored when output_mode is "Linear scene".
    """
    if not _HAS_RAWPY:
        raise RuntimeError(
            "[Darkroom RAW] rawpy is not installed. Run: pip install rawpy"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[Darkroom RAW] file not found: {path}")

    with rawpy.imread(path) as raw:
        xtrans = is_xtrans(raw)
        pp_kwargs = dict(
            output_color=_rawpy_colorspace(colorspace),
            gamma=(1, 1),                # always decode linearly; we encode ourselves
            no_auto_bright=True,         # always disable; we control exposure explicitly
            output_bps=16,
            half_size=bool(half_size),
            highlight_mode=_rawpy_highlight(highlight_mode),
        )
        demo_enum = _rawpy_demosaic(demosaic, xtrans)
        if demo_enum is not None:
            pp_kwargs["demosaic_algorithm"] = demo_enum

        wb_mode = _WHITE_BALANCE_MAP.get(white_balance, "camera")
        if wb_mode == "camera":
            pp_kwargs["use_camera_wb"] = True
        elif wb_mode == "auto":
            pp_kwargs["use_auto_wb"] = True
        elif wb_mode == "daylight":
            pp_kwargs["user_wb"] = list(raw.daylight_whitebalance)

        img_u16 = raw.postprocess(**pp_kwargs)
        meta = extract_metadata(path, raw)

    img = np.ascontiguousarray(img_u16).astype(np.float32) / 65535.0

    if output_mode == "sRGB display":
        # 1. Baseline exposure in linear light (user override, 0 by default).
        if abs(baseline_exposure) > 0.001:
            img = img * (2.0 ** baseline_exposure)

        # 2. Camera Raw-like default tone curve (lifts mids, approximates Adobe
        #    Standard LookTable + Medium Contrast combo until session 2 DCP Apply).
        img = np.clip(img, 0.0, 1.0)
        img[..., 0] = cubic_spline_curve(img[..., 0], CAMERA_RAW_DEFAULT_CURVE)
        img[..., 1] = cubic_spline_curve(img[..., 1], CAMERA_RAW_DEFAULT_CURVE)
        img[..., 2] = cubic_spline_curve(img[..., 2], CAMERA_RAW_DEFAULT_CURVE)

        # 3. sRGB OETF + clip to [0, 1].
        img = linear_to_srgb(np.clip(img, 0.0, None))
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
    # Linear scene mode: return raw normalized linear data, no further processing.

    return img, meta
