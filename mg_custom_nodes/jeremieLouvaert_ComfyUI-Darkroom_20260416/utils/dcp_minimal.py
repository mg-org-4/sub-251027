"""
Minimal DCP (DNG Color Profile) reader for ComfyUI-Darkroom.

This module is a stopgap for Wave 6 session 1 — it extracts only the
BaselineExposure tag so RAW Load can auto-apply the correct per-camera
exposure baseline matching Camera Raw. Session 2 will replace this with
a full DCP parser that handles HueSatMap, LookTable, ProfileToneCurve, etc.

DCP files are TIFF/IFD containers with DNG-specific tags. We walk the
IFDs looking for tag 0xC7A5 (BaselineExposure, SRATIONAL, count 1).
Pure Python, no dependencies beyond the stdlib.
"""

import os
import struct

# DNG tag numbers (hex)
TAG_BASELINE_EXPOSURE = 0xC7A5        # 51109

# TIFF data types
TYPE_SRATIONAL = 10                   # two signed int32: numerator, denominator

# Default location on Windows where Adobe Camera Raw ships camera profiles.
# The "Adobe Standard" subfolder holds per-camera baseline calibration DCPs
# for almost every supported body, including Fuji / Hasselblad / Leica which
# don't have film-sim DCPs in the sibling "Camera/" folder.
ADOBE_STANDARD_DIR = r"C:\ProgramData\Adobe\CameraRaw\CameraProfiles\Adobe Standard"


# Cache: (make, model) → (dcp_path, baseline_ev) so a workflow that loads
# multiple files from the same camera doesn't re-scan the directory.
_lookup_cache = {}


def read_dcp_baseline_exposure(path):
    """
    Parse a DCP file and return BaselineExposure as a float in stops.
    Returns None if the file is missing, not a valid TIFF, or the tag is absent.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None

    if len(data) < 8:
        return None

    bo = data[:2]
    if bo == b"II":
        e = "<"
    elif bo == b"MM":
        e = ">"
    else:
        return None

    magic = struct.unpack_from(f"{e}H", data, 2)[0]
    if magic != 42:
        return None

    first_ifd = struct.unpack_from(f"{e}I", data, 4)[0]
    if first_ifd == 0 or first_ifd >= len(data):
        return None

    # Walk all IFDs; BaselineExposure is usually in IFD0 but profile subfiles exist.
    ifd_offset = first_ifd
    visited = set()
    while ifd_offset != 0 and ifd_offset not in visited and ifd_offset < len(data):
        visited.add(ifd_offset)
        if ifd_offset + 2 > len(data):
            break
        n_entries = struct.unpack_from(f"{e}H", data, ifd_offset)[0]
        entries_start = ifd_offset + 2
        entries_end = entries_start + n_entries * 12
        if entries_end + 4 > len(data):
            break

        for i in range(n_entries):
            off = entries_start + i * 12
            tag, dtype, count = struct.unpack_from(f"{e}HHI", data, off)
            if tag != TAG_BASELINE_EXPOSURE:
                continue
            if dtype != TYPE_SRATIONAL or count != 1:
                # Unexpected type — skip rather than crash
                continue
            # 8 bytes of data > 4 byte field → field holds an offset
            value_offset = struct.unpack_from(f"{e}I", data, off + 8)[0]
            if value_offset + 8 > len(data):
                return None
            num, den = struct.unpack_from(f"{e}ii", data, value_offset)
            if den == 0:
                return None
            return float(num) / float(den)

        next_ifd = struct.unpack_from(f"{e}I", data, entries_end)[0]
        ifd_offset = next_ifd

    return None


def _normalize_camera_key(make, model):
    make = (make or "").strip()
    model = (model or "").strip()
    return (make.lower(), model.lower())


def find_adobe_standard_dcp(make, model, root=ADOBE_STANDARD_DIR):
    """
    Locate the Adobe Standard DCP for a camera make/model, or None.
    Case-insensitive directory scan — handles Adobe's inconsistent capitalisation
    (Leica profiles are sometimes "LEICA ...", Fuji profiles are "Fujifilm ...").
    """
    if not os.path.isdir(root):
        return None
    make_clean = (make or "").strip()
    model_clean = (model or "").strip()
    if not model_clean:
        return None

    try:
        entries = os.listdir(root)
    except OSError:
        return None

    # Adobe's filename pattern is "{Make} {Model} Adobe Standard.dcp"
    target = f"{make_clean} {model_clean} adobe standard.dcp".lower()
    target_no_make = f"{model_clean} adobe standard.dcp".lower()

    # Pass 1: exact match on the full pattern (case insensitive)
    for fn in entries:
        if fn.lower() == target:
            return os.path.join(root, fn)

    # Pass 2: substring match (handles odd brand capitalisation like "FUJIFILM" EXIF
    # mapping to "Fujifilm" in the filename, or model names with trailing variants)
    model_lower = model_clean.lower()
    make_lower = make_clean.lower()
    for fn in entries:
        low = fn.lower()
        if not low.endswith(" adobe standard.dcp"):
            continue
        if model_lower and model_lower in low and (not make_lower or make_lower in low or "fuji" in make_lower and "fuji" in low):
            return os.path.join(root, fn)

    # Pass 3: model-only fallback (some profiles drop the brand prefix, e.g. "LEICA M11")
    for fn in entries:
        if target_no_make in fn.lower():
            return os.path.join(root, fn)

    return None


def resolve_auto_baseline(make, model):
    """
    Return (dcp_path, baseline_exposure_in_stops) for a camera, or (None, None).
    Results are cached per (make, model) pair so repeated loads are free.
    """
    key = _normalize_camera_key(make, model)
    if key in _lookup_cache:
        return _lookup_cache[key]

    dcp_path = find_adobe_standard_dcp(make, model)
    if dcp_path is None:
        _lookup_cache[key] = (None, None)
        return None, None

    ev = read_dcp_baseline_exposure(dcp_path)
    result = (dcp_path, ev)
    _lookup_cache[key] = result
    return result
