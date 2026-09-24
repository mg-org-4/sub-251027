"""
Build per-body Fujifilm B&W Camera Look DCPs (Acros and Monochrome, each
with no-filter / +R / +Y / +G variants) as our own approximation of Fuji's
in-camera B&W sims.

This is OUR approximation, not a calibrated 1:1 match to Fuji's in-camera
output. abpy/FujifilmCameraProfiles deliberately skips B&W sims because
Fuji's B&W rendering is a channel-mix + tone-curve operation that DCP's
HSV-based LookTable expresses awkwardly. We encode it anyway by:

  - sat_scale = 0 everywhere in the LookTable (output is grayscale)
  - val_scale varies per hue bin, matching a luma-weighted mono mix
    derived from published Neopan 100 Acros channel weights (for Acros)
    or BT.709 (for Monochrome)
  - filter variants (+R / +Y / +G) pre-multiply the base weights by the
    filter's transmission spectrum before building the LUT
  - ToneCurve: a Pchip-interpolated shape matching Acros's characteristic
    H&D curve (soft toe, smooth highlight roll-off) for Acros variants;
    near-identity for Monochrome variants

Base matrices (ColorMatrix1/2, ForwardMatrix, illuminants) are cloned
from the body's Adobe Standard DCP so the upstream color calibration
stays intact. HSM is dropped (Camera Look DCPs don't ship an HSM).

Usage:
    python build_fuji_bw_dcps.py \\
        --body "Fujifilm GFX 50S" \\
        --out  "F:/.../ComfyUI/models/camera_profiles/Fujifilm GFX 50S"
"""

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from dataclasses import replace

from utils.dcp import (
    read_dcp_profile,
    write_dcp_profile,
    build_bw_look_table,
    build_tone_curve_from_points,
    find_adobe_standard_dcp,
)


# Base luma weights before any filter is applied.
#   ACROS: Fuji Neopan 100 Acros weights from Capture One's channel-mix
#     style data (parsed into Darkroom's data/bw_stocks.py). Film-accurate
#     panchromatic with elevated blue response -- the hallmark of real
#     Acros emulsion and, by extension, Fuji's sim derived from it.
#   MONOCHROME: BT.709 luma (digital perceptual luma). Flatter, more
#     "neutral" B&W than Acros.
BASE_WEIGHTS = {
    "Acros":      (0.21, 0.32, 0.47),
    "Monochrome": (0.2126, 0.7152, 0.0722),
}

# Classic darkroom filter transmissions (approximate pass-band fractions
# per RGB channel). Pre-multiplied onto the base weights and renormalized.
FILTERS = {
    "":   (1.00, 1.00, 1.00),   # no filter
    "+R": (1.00, 0.15, 0.02),   # Wratten 25
    "+Y": (1.00, 0.85, 0.20),   # Wratten 8
    "+G": (0.30, 1.00, 0.30),   # Wratten 11
}


def _effective_weights(base, flt):
    r = base[0] * flt[0]
    g = base[1] * flt[1]
    b = base[2] * flt[2]
    s = r + g + b
    return (r / s, g / s, b / s)


# Tone curves. Control points interpolated via Pchip to 128 points.
# Acros: slight shadow compression, linear mid, smooth highlight shoulder.
# Monochrome: near-identity with tiny S-curve for a hair of contrast.
ACROS_CURVE_POINTS = [
    (0.00, 0.00),
    (0.03, 0.015),
    (0.10, 0.07),
    (0.25, 0.23),
    (0.50, 0.52),
    (0.75, 0.80),
    (0.90, 0.93),
    (1.00, 1.00),
]

MONOCHROME_CURVE_POINTS = [
    (0.00, 0.00),
    (0.10, 0.09),
    (0.50, 0.50),
    (0.90, 0.91),
    (1.00, 1.00),
]


# "Base sim" -> "display name", in the order we want them written.
VARIANTS = [
    ("Acros", ""),
    ("Acros", "+R"),
    ("Acros", "+Y"),
    ("Acros", "+G"),
    ("Monochrome", ""),
    ("Monochrome", "+R"),
    ("Monochrome", "+Y"),
    ("Monochrome", "+G"),
]


def _curve_for(base):
    pts = ACROS_CURVE_POINTS if base == "Acros" else MONOCHROME_CURVE_POINTS
    return build_tone_curve_from_points(pts, n=128)


def build_one(base_profile, base_sim, filter_tag, out_path, body):
    weights = _effective_weights(BASE_WEIGHTS[base_sim], FILTERS[filter_tag])
    lut, dims = build_bw_look_table(weights)
    curve = _curve_for(base_sim)
    display = base_sim + filter_tag

    new = replace(
        base_profile,
        path="",
        profile_name=f"Camera {display}",
        unique_camera_model=base_profile.unique_camera_model or body,
        # Drop HSM: Camera Look DCPs are LookTable + ToneCurve only.
        hsm_dims=None,
        hsm_data_1=None,
        hsm_data_2=None,
        hsm_encoding=0,
        lut_dims=dims,
        lut_data=lut,
        lut_encoding=1,
        tone_curve=curve,
    )
    write_dcp_profile(out_path, new)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True,
                    help='Body folder name, e.g. "Fujifilm GFX 50S"')
    ap.add_argument("--out", required=True,
                    help="Destination folder (typically ComfyUI/models/camera_profiles/<body>/)")
    ap.add_argument("--base", default=None,
                    help="Path to base Adobe Standard DCP (auto-resolved if omitted)")
    args = ap.parse_args()

    if args.base:
        base_path = args.base
    else:
        parts = args.body.strip().split(" ", 1)
        make = parts[0]
        model = parts[1] if len(parts) > 1 else ""
        base_path = find_adobe_standard_dcp(make, model)
        if base_path is None:
            print(f"[build] No Adobe Standard DCP found for {args.body}.")
            print("[build] Pass --base /path/to/Body Adobe Standard.dcp explicitly.")
            sys.exit(1)

    print(f"[build] Base: {base_path}")
    base = read_dcp_profile(base_path)

    os.makedirs(args.out, exist_ok=True)
    written = []
    for base_sim, filter_tag in VARIANTS:
        display = base_sim + filter_tag
        out_path = os.path.join(args.out, f"{args.body} Camera {display}.dcp")
        build_one(base, base_sim, filter_tag, out_path, args.body)
        size_kb = os.path.getsize(out_path) // 1024
        print(f"[build] wrote {os.path.basename(out_path)}  ({size_kb} KB)")
        written.append(out_path)

    print(f"[build] done: {len(written)} DCPs in {args.out}")


if __name__ == "__main__":
    main()
