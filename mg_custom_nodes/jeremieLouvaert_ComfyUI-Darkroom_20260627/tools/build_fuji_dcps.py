"""
Build per-body Fujifilm Camera Look DCPs from abpy/FujifilmCameraProfiles
XML tables, by splicing the LookTable + ToneCurve onto a body's Adobe
Standard DCP base.

Usage:
    python build_fuji_dcps.py \\
        --body "Fujifilm GFX 50S" \\
        --abpy "C:/Users/Jeremie/comfyUI-DEV-Tools/third_party/FujifilmCameraProfiles" \\
        --out  "F:/.../ComfyUI/models/camera_profiles/Fujifilm GFX 50S"

The base DCP is found in C:/ProgramData/Adobe/CameraRaw/CameraProfiles/Adobe Standard
unless --base is given explicitly.

Each output is named "<body> Camera <SimDisplayName>.dcp" so it slots into
the brand/look dropdown automatically once placed in
ComfyUI/models/camera_profiles/<body>/.
"""

import argparse
import os
import sys

# Make sibling utils/ importable when run as a script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from utils.dcp import (
    read_dcp_profile,
    write_dcp_profile,
    parse_abpy_xml_table,
    find_adobe_standard_dcp,
)


# Short XML filename -> Adobe-style display name. Matches Fuji's official
# in-camera labels (used by ACR's profile dropdown).
SIM_NAMES = {
    "provia":         "Provia",
    "velvia":         "Velvia",
    "astia":          "Astia",
    "classic chrome": "Classic Chrome",
    "pro neg hi":     "Pro Neg Hi",
    "pro neg std":    "Pro Neg Std",
    "eterna":         "Eterna",
    "reala ace":      "Reala Ace",
}


def build_one(base_profile, xml_path, out_path, sim_display, body):
    text = open(xml_path, "r", encoding="utf-8").read()
    lut, lut_dims, tone_curve = parse_abpy_xml_table(text)

    # Clone the base, then replace the look-defining stages.
    p = base_profile
    # Start from a fresh dataclass instance to avoid mutating the cached one.
    from dataclasses import replace
    new = replace(
        p,
        path="",
        profile_name=f"Camera {sim_display}",
        unique_camera_model=p.unique_camera_model or body,
        # Drop HSM — Camera Look DCPs are LUT + ToneCurve only.
        hsm_dims=None,
        hsm_data_1=None,
        hsm_data_2=None,
        hsm_encoding=0,
        # Replace LUT with abpy's per-sim LookTable.
        lut_dims=lut_dims,
        lut_data=lut,
        lut_encoding=1,                 # abpy: ProfileLookTableEncoding = 1
        tone_curve=tone_curve,
    )
    write_dcp_profile(out_path, new)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True,
                    help='Body folder name, e.g. "Fujifilm GFX 50S"')
    ap.add_argument("--abpy", required=True,
                    help="Path to cloned abpy/FujifilmCameraProfiles repo")
    ap.add_argument("--out", required=True,
                    help="Destination folder (typically ComfyUI/models/camera_profiles/<body>/)")
    ap.add_argument("--base", default=None,
                    help="Path to base Adobe Standard DCP (auto-resolved if omitted)")
    args = ap.parse_args()

    if args.base:
        base_path = args.base
    else:
        # Body folder is usually "<Make> <Model>"; split into make/model for the resolver.
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

    xml_dir = os.path.join(args.abpy, "xml tables")
    if not os.path.isdir(xml_dir):
        print(f"[build] Missing xml tables/ under {args.abpy}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    written = []
    for stem, display in SIM_NAMES.items():
        xml_path = os.path.join(xml_dir, f"{stem}.txt")
        if not os.path.isfile(xml_path):
            print(f"[build] skip {display}: {xml_path} not found")
            continue
        out_path = os.path.join(args.out, f"{args.body} Camera {display}.dcp")
        build_one(base, xml_path, out_path, display, args.body)
        size_kb = os.path.getsize(out_path) // 1024
        print(f"[build] wrote {os.path.basename(out_path)}  ({size_kb} KB)")
        written.append(out_path)

    print(f"[build] done: {len(written)} DCPs in {args.out}")


if __name__ == "__main__":
    main()
