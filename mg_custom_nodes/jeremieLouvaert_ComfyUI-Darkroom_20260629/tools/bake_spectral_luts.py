"""
CLI baker for Spectral Film LUTs.

Runs JanLohse/spectral_film_lut (vendored MIT) to produce .cube files under
data/spectral_luts/. Ships a small numba shim because ComfyUI's embedded
Python has numba broken against NumPy 2.4; the shim lets their code run in
pure-Python mode (slower but functional for offline baking).

Usage:
  python bake_spectral_luts.py --list
  python bake_spectral_luts.py --all
  python bake_spectral_luts.py --name portra_400_endura_premier
  python bake_spectral_luts.py --lut-size 33 --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path


# --- numba shim --------------------------------------------------------------
# Must be installed BEFORE spectral_film_lut imports numba via utils.py.
def _install_numba_shim() -> None:
    shim = types.ModuleType("numba")

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    class _Cuda:
        @staticmethod
        def jit(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]

            def decorator(func):
                return func

            return decorator

        @staticmethod
        def is_available() -> bool:
            return False

    shim.njit = njit
    shim.prange = range
    shim.cuda = _Cuda()
    sys.modules["numba"] = shim


_install_numba_shim()


# --- path setup --------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_DARKROOM_ROOT = _HERE.parent
_VENDORED = _DARKROOM_ROOT / "third_party" / "spectral_film_lut" / "src"
_LUT_DIR = _DARKROOM_ROOT / "data" / "spectral_luts"

if str(_VENDORED) not in sys.path:
    sys.path.insert(0, str(_VENDORED))


# --- preset definitions ------------------------------------------------------
# Curated negative x print combinations. Each produces one .cube.
# Parameters beyond (negative, print) use JanLohse defaults -- add kwargs to
# override projection_kelvin, color_masking, etc. as needed.

# Bake-time defaults applied to every preset. JanLohse's engine defaults to
# ARRI Wide Gamut 4 / Gamma 2.4 on input (cinema log workflow). ComfyUI
# tensors are sRGB-encoded Rec.709, so we target sRGB on both ends to make
# the LUTs drop-in for our pipeline.
BAKE_DEFAULTS = {
    "input_colourspace": "sRGB",
    "output_gamut": "Rec. 709",
    "gamma_func": "sRGB",
}


def _cat(name: str, neg: str, print_: str, category: str, **extra) -> dict:
    return {"name": name, "neg": neg, "print": print_, "category": category, **extra}


PRESETS: list[dict] = [
    # --- C41 STILLS on Endura / Supra / Portra Endura papers ---
    _cat("portra_160_endura_premier", "Kodak Portra 160", "Kodak Endura Premier Paper", "C41 Still"),
    _cat("portra_400_endura_premier", "Kodak Portra 400", "Kodak Endura Premier Paper", "C41 Still"),
    _cat("portra_800_endura_premier", "Kodak Portra 800", "Kodak Endura Premier Paper", "C41 Still"),
    _cat("portra_400_supra_endura", "Kodak Portra 400", "Kodak Supra Endura Paper", "C41 Still"),
    _cat("portra_400_portra_endura", "Kodak Portra 400", "Kodak Portra Endura Paper", "C41 Still"),
    _cat("ektar_100_endura_premier", "Kodak Ektar 100", "Kodak Endura Premier Paper", "C41 Still"),
    _cat("gold_200_endura_premier", "Kodak Gold 200", "Kodak Endura Premier Paper", "C41 Still"),
    _cat("ultramax_500_endura_premier", "Kodak Ultramax 500", "Kodak Endura Premier Paper", "C41 Still"),
    _cat("vericolor_iii_endura_premier", "Kodak Vericolor III", "Kodak Endura Premier Paper", "C41 Still"),
    # --- FUJI STILLS on Fuji Crystal Archive papers ---
    _cat("fuji_pro_400h_ca_maxima", "Fuji Pro 400H", "Fuji Crystal Archive Maxima", "C41 Still"),
    _cat("fuji_pro_160c_ca_maxima", "Fuji Pro 160C", "Fuji Crystal Archive Maxima", "C41 Still"),
    _cat("fuji_pro_160s_ca_dpii", "Fuji Pro 160S", "Fuji Crystal Archive DPII", "C41 Still"),
    _cat("fuji_c200_ca_dpii", "Fuji C200", "Fuji Crystal Archive DPII", "C41 Still"),
    _cat("fuji_superia_reala_ca_pro_pdii", "Fuji Superia Reala", "Fuji Crystal Archive Pro PDII", "C41 Still"),
    _cat("fuji_superia_xtra_400_ca_dpii", "Fuji Superia X-Tra 400", "Fuji Crystal Archive DPII", "C41 Still"),
    _cat("fuji_natura_1600_ca_dpii", "Fuji Natura 1600", "Fuji Crystal Archive DPII", "C41 Still"),
    # --- CINEMA: Vision3 neg + modern print stocks ---
    _cat("vision3_50d_2383", "Kodak Vision3 50D 5203", "Kodak Vision 2383", "Cinema"),
    _cat("vision3_200t_2383", "Kodak Vision3 200T 5213", "Kodak Vision 2383", "Cinema"),
    _cat("vision3_250d_2383", "Kodak Vision3 250D 5207", "Kodak Vision 2383", "Cinema"),
    _cat("vision3_250d_2393", "Kodak Vision3 250D 5207", "Kodak Vision Premier 2393", "Cinema"),
    _cat("vision3_500t_2383", "Kodak Vision3 500T 5219", "Kodak Vision 2383", "Cinema"),
    _cat("vision3_500t_2393", "Kodak Vision3 500T 5219", "Kodak Vision Premier 2393", "Cinema"),
    # --- REVERSAL slides printed to Ilfochrome / Ektachrome Radiance ---
    _cat("velvia_50_ilfochrome_m", "Fuji Velvia 50", "Ilfochrome Micrographic M", "Reversal"),
    _cat("provia_100f_ilfochrome_m", "Fuji Provia 100F", "Ilfochrome Micrographic M", "Reversal"),
    _cat("ektachrome_100d_radiance_iii", "Kodak Ektachrome 100D", "Kodak Ektachrome Radiance III Paper", "Reversal"),
    _cat("kodachrome_64_radiance_iii", "Kodachrome 64", "Kodak Ektachrome Radiance III Paper", "Reversal"),
    # --- INSTANT / SPECIALTY ---
    _cat("fp100c_fujiflex_new", "Fuji FP-100C", "Fujiflex Crystal Archive New Version", "Instant"),
    _cat("instax_color_fujiflex_new", "Fuji Instax color", "Fujiflex Crystal Archive New Version", "Instant"),
    # --- AEROCOLOR / Niche ---
    _cat("aerocolor_endura_premier", "Kodak Aerocolor IV 2460", "Kodak Endura Premier Paper", "Niche"),
    _cat("aerocolor_high_endura_premier", "Kodak Aerocolor IV 2460 High", "Kodak Endura Premier Paper", "Niche"),
    _cat("agfa_vista_100_ca_dpii", "Agfa Vista 100", "Fuji Crystal Archive DPII", "Niche"),
    # --- B&W ---
    _cat("trix_400_polymax_grade_2", "Kodak Tri-X 400", "Kodak Polymax Fine-Art Paper Grade 2", "B&W"),
    _cat("trix_400_polymax_grade_3", "Kodak Tri-X 400", "Kodak Polymax Fine-Art Paper Grade 3", "B&W"),
    _cat("kodak_5222_polymax_grade_2", "Kodak 5222", "Kodak Polymax Fine-Art Paper Grade 2", "B&W"),
    _cat("kodak_5222_2302_dev_5", "Kodak 5222", "Kodak 2302 Dev 5", "B&W"),
]


_RESERVED_KEYS = {"name", "neg", "print", "category"}


def _bake_one(preset: dict, create_lut_fn, stocks: dict, lut_size: int) -> Path:
    neg = stocks.get(preset["neg"])
    prn = stocks.get(preset["print"])
    if neg is None:
        raise KeyError(f"negative stock not found: {preset['neg']!r}")
    if prn is None:
        raise KeyError(f"print stock not found: {preset['print']!r}")

    out_path = _LUT_DIR / f"{preset['name']}.cube"
    # Preset-specific overrides win over bake defaults.
    kwargs = {**BAKE_DEFAULTS, **{k: v for k, v in preset.items() if k not in _RESERVED_KEYS}}

    t0 = time.time()
    import colour
    import numpy as np

    lut = colour.LUT3D(size=lut_size, name=preset["name"])
    transform = neg.generate_conversion(neg, prn, **kwargs)
    table = transform(lut.table)
    if table.shape[-1] == 1:
        table = table.repeat(3, -1)
    lut.table = np.asarray(table)
    colour.io.write_LUT(lut, str(out_path))
    elapsed = time.time() - t0
    print(f"[bake] {preset['name']}: {elapsed:.1f}s -> {out_path.name}")
    return out_path


def _list_presets() -> None:
    print(f"{len(PRESETS)} presets:")
    for p in PRESETS:
        print(f"  {p['name']:<42}  {p['neg']}  ->  {p['print']}")


def _load_engine():
    from spectral_film_lut import FILM_STOCKS
    from spectral_film_lut.film_spectral import FilmSpectral
    from spectral_film_lut.utils import create_lut

    print(f"[bake] loading {len(FILM_STOCKS)} film stocks (precomputing)...")
    t0 = time.time()
    stocks = {}
    for fd in FILM_STOCKS:
        instance = FilmSpectral(fd)
        if instance.stage == "print" and instance.density_measure == "status_a":
            first_neg = next(
                (s for s in stocks.values() if s.stage == "camera"), None
            )
            if first_neg is not None:
                instance.set_color_checker(negative=first_neg)
            else:
                instance.set_color_checker()
        else:
            instance.set_color_checker()
        stocks[instance.name] = instance
    print(f"[bake] stocks ready in {time.time() - t0:.1f}s")
    return create_lut, stocks


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list presets and exit")
    parser.add_argument("--all", action="store_true", help="bake every preset")
    parser.add_argument("--name", help="bake one preset by name")
    parser.add_argument("--lut-size", type=int, default=33, help="LUT cube size (default 33)")
    args = parser.parse_args(argv[1:])

    if args.list:
        _list_presets()
        return 0
    if not args.all and not args.name:
        parser.print_help()
        return 2

    _LUT_DIR.mkdir(parents=True, exist_ok=True)
    create_lut_fn, stocks = _load_engine()

    to_bake: list[dict]
    if args.name:
        match = [p for p in PRESETS if p["name"] == args.name]
        if not match:
            print(f"[bake] unknown preset: {args.name}")
            return 2
        to_bake = match
    else:
        to_bake = PRESETS

    failures: list[tuple[str, str]] = []
    t_all = time.time()
    for i, preset in enumerate(to_bake, 1):
        print(f"[bake] ({i}/{len(to_bake)}) {preset['name']}")
        try:
            _bake_one(preset, create_lut_fn, stocks, args.lut_size)
        except Exception as e:
            failures.append((preset["name"], str(e)))
            print(f"[bake] FAILED: {preset['name']}: {e}")
    print(f"[bake] done in {time.time() - t_all:.1f}s total")

    _write_manifest(args.lut_size)

    if failures:
        print(f"[bake] {len(failures)} failures:")
        for name, err in failures:
            print(f"  {name}: {err}")
        return 1
    return 0


def _write_manifest(lut_size: int) -> None:
    manifest = {"lut_size": lut_size, "presets": []}
    for p in PRESETS:
        cube = _LUT_DIR / f"{p['name']}.cube"
        if not cube.exists():
            continue
        manifest["presets"].append(
            {
                "name": p["name"],
                "category": p.get("category", "Other"),
                "negative": p["neg"],
                "print": p["print"],
                "file": cube.name,
            }
        )
    manifest_path = _LUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[bake] wrote manifest: {manifest_path.name} ({len(manifest['presets'])} entries)")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
