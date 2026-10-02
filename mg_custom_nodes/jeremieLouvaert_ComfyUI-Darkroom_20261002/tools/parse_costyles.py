"""
Parse Capture One .costyle XML files and extract film stock parameters.

Outputs:
1. A cross-reference report: what we have vs what C1 has
2. Extracted curve + color data ready to refine our stocks

Usage:
    python parse_costyles.py <costyle_dir> [--report] [--export]
"""

import xml.etree.ElementTree as ET
import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict


# ── Capture One parameter extraction ──────────────────────────────────────────

@dataclass
class CostyleParsed:
    """All useful parameters extracted from a single .costyle file."""
    name: str
    filename: str
    brand: str  # folder name (Agfa, Fuji, Kodak, etc.)

    # Gradation curves: list of (x, y) control points, 0-1 normalized
    curve_master: List[Tuple[float, float]] = field(default_factory=list)
    curve_red: List[Tuple[float, float]] = field(default_factory=list)
    curve_green: List[Tuple[float, float]] = field(default_factory=list)
    curve_blue: List[Tuple[float, float]] = field(default_factory=list)
    curve_luma: List[Tuple[float, float]] = field(default_factory=list)

    # Color balance (RGB multipliers)
    color_balance_shadow: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    color_balance_midtone: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    color_balance_highlight: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Tone adjustments
    contrast: float = 0.0
    saturation: float = 0.0
    shadow_recovery: float = 0.0
    highlight_recovery: float = 0.0

    # Film grain
    grain_amount: float = 0.0
    grain_density: float = 0.0
    grain_granularity: float = 0.0
    grain_type: int = 0

    # Sharpening
    usm_amount: float = 0.0
    usm_radius: float = 0.0

    # B&W specific
    is_bw: bool = False
    bw_red: float = 0.0
    bw_green: float = 0.0
    bw_blue: float = 0.0
    bw_cyan: float = 0.0
    bw_magenta: float = 0.0
    bw_yellow: float = 0.0

    # Color corrections (the complex multi-entry field)
    color_corrections_raw: str = ""

    # Variant type
    is_variant: bool = False
    variant_type: str = ""  # "", "Cool", "Warm", "+", "++", "-", "--"
    base_stock_name: str = ""


def _parse_curve(value: str) -> List[Tuple[float, float]]:
    """Parse 'x1,y1;x2,y2;...' into list of (x, y) tuples."""
    if not value or value.strip() == "":
        return []
    points = []
    for pair in value.split(";"):
        parts = pair.strip().split(",")
        if len(parts) == 2:
            points.append((float(parts[0]), float(parts[1])))
    return points


def _parse_rgb(value: str) -> Tuple[float, float, float]:
    """Parse 'r;g;b' into (r, g, b) tuple."""
    parts = value.split(";")
    if len(parts) >= 3:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    return (1.0, 1.0, 1.0)


def _detect_variant(filename: str) -> Tuple[bool, str, str]:
    """Detect if a file is a variant and extract base name + variant type."""
    stem = Path(filename).stem
    variants = ["- -", "++", "--", "+", "-", "Cool", "Warm",
                 "Portrait", "Vibrant", "Landscape", "HC",
                 "Soft Highs", "Alt", "Green", "Negative", "Cold"]
    for v in variants:
        if stem.endswith(f" {v}"):
            base = stem[:-(len(v) + 1)].strip()
            return True, v, base
        # Handle "(2)" variant
        if stem.endswith(f" ({v})"):
            base = stem[:-(len(v) + 3)].strip()
            return True, v, base
    if stem.endswith(" (2)"):
        return True, "(2)", stem[:-4].strip()
    return False, "", stem


def parse_costyle(filepath: str, brand: str) -> CostyleParsed:
    """Parse a single .costyle XML file."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    params = {}
    for elem in root.findall("E"):
        params[elem.get("K")] = elem.get("V", "")

    filename = os.path.basename(filepath)
    is_variant, variant_type, base_name = _detect_variant(filename)

    parsed = CostyleParsed(
        name=params.get("Name", Path(filepath).stem),
        filename=filename,
        brand=brand,
        is_variant=is_variant,
        variant_type=variant_type,
        base_stock_name=base_name,
    )

    # Curves
    parsed.curve_master = _parse_curve(params.get("GradationCurve", ""))
    parsed.curve_red = _parse_curve(params.get("GradationCurveRed", ""))
    parsed.curve_green = _parse_curve(params.get("GradationCurveGreen", ""))
    parsed.curve_blue = _parse_curve(params.get("GradationCurveBlue", ""))
    parsed.curve_luma = _parse_curve(params.get("GradationCurveY", ""))

    # Color balance
    if "ColorBalanceShadow" in params:
        parsed.color_balance_shadow = _parse_rgb(params["ColorBalanceShadow"])
    if "ColorBalanceMidtone" in params:
        parsed.color_balance_midtone = _parse_rgb(params["ColorBalanceMidtone"])
    if "ColorBalanceHighlight" in params:
        parsed.color_balance_highlight = _parse_rgb(params["ColorBalanceHighlight"])

    # Tone
    parsed.contrast = float(params.get("Contrast", 0))
    parsed.saturation = float(params.get("Saturation", 0))
    parsed.shadow_recovery = float(params.get("ShadowRecovery", 0))
    parsed.highlight_recovery = float(params.get("HighlightRecovery", 0))

    # Grain
    parsed.grain_amount = float(params.get("FilmGrainAmount", 0))
    parsed.grain_density = float(params.get("FilmGrainDensity", 0))
    parsed.grain_granularity = float(params.get("FilmGrainGranularity", 0))
    parsed.grain_type = int(params.get("FilmGrainType", 0))

    # Sharpening
    parsed.usm_amount = float(params.get("UsmAmount", 0))
    parsed.usm_radius = float(params.get("UsmRadius", 0))

    # B&W
    if params.get("BwEnabled") == "1":
        parsed.is_bw = True
        parsed.bw_red = float(params.get("BwRed", 0))
        parsed.bw_green = float(params.get("BwGreen", 0))
        parsed.bw_blue = float(params.get("BwBlue", 0))
        parsed.bw_cyan = float(params.get("BwCyan", 0))
        parsed.bw_magenta = float(params.get("BwMagenta", 0))
        parsed.bw_yellow = float(params.get("BwYellow", 0))

    # Raw color corrections
    parsed.color_corrections_raw = params.get("ColorCorrections", "")

    return parsed


def parse_all_costyles(styles_dir: str) -> List[CostyleParsed]:
    """Parse all .costyle files in the styles directory."""
    results = []
    for brand_dir in sorted(Path(styles_dir).iterdir()):
        if not brand_dir.is_dir():
            continue
        brand = brand_dir.name
        for f in sorted(brand_dir.glob("*.costyle")):
            try:
                parsed = parse_costyle(str(f), brand)
                results.append(parsed)
            except Exception as e:
                print(f"  [WARN] Failed to parse {f.name}: {e}")
    return results


# ── Analysis: curve → our CurveParams ────────────────────────────────────────

def curve_to_params(points: List[Tuple[float, float]]) -> Dict:
    """
    Analyze a C1 gradation curve and estimate our CurveParams equivalent.

    C1 curves are multi-point. Our model uses: toe_power, shoulder_power, slope.
    We estimate by measuring:
    - slope: midtone steepness around 0.5
    - toe: how compressed the shadows are (how far below diagonal)
    - shoulder: how compressed the highlights are (how far below diagonal)
    """
    if len(points) < 3:
        return {"toe": 1.5, "shoulder": 1.5, "slope": 1.0, "note": "too few points"}

    # Sort by x
    pts = sorted(points, key=lambda p: p[0])

    # Find midtone slope: derivative around x=0.4-0.6
    mid_slopes = []
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x1 - x0 > 0.001 and 0.25 < (x0 + x1) / 2 < 0.75:
            mid_slopes.append((y1 - y0) / (x1 - x0))
    slope = sum(mid_slopes) / len(mid_slopes) if mid_slopes else 1.0

    # Toe: shadow compression. Compare actual y vs diagonal y in low range.
    toe_deviation = 0.0
    toe_count = 0
    for x, y in pts:
        if 0.05 < x < 0.35:
            toe_deviation += (x - y)  # positive = compressed (below diagonal)
            toe_count += 1
    avg_toe_dev = toe_deviation / toe_count if toe_count else 0.0
    # Map deviation to toe_power: more deviation = stronger toe
    toe_power = 1.5 + avg_toe_dev * 8.0  # scale factor

    # Shoulder: highlight compression
    shoulder_deviation = 0.0
    shoulder_count = 0
    for x, y in pts:
        if 0.65 < x < 0.95:
            shoulder_deviation += (x - y)  # positive = compressed
            shoulder_count += 1
    avg_shoulder_dev = shoulder_deviation / shoulder_count if shoulder_count else 0.0
    shoulder_power = 1.5 + avg_shoulder_dev * 8.0

    return {
        "toe": round(max(1.0, toe_power), 3),
        "shoulder": round(max(1.0, shoulder_power), 3),
        "slope": round(slope, 4),
    }


def analyze_color_balance(parsed: CostyleParsed) -> Dict:
    """Extract shadow/highlight tint from color balance values."""
    s = parsed.color_balance_shadow
    m = parsed.color_balance_midtone
    h = parsed.color_balance_highlight

    # C1 color balance: 1.0 = neutral, >1 = more, <1 = less
    # Convert to our tint format: offset from neutral
    shadow_tint = (
        round(s[0] - 1.0, 4),
        round(s[1] - 1.0, 4),
        round(s[2] - 1.0, 4),
    )
    highlight_tint = (
        round(h[0] - 1.0, 4),
        round(h[1] - 1.0, 4),
        round(h[2] - 1.0, 4),
    )

    return {
        "shadow_tint": shadow_tint,
        "highlight_tint": highlight_tint,
        "midtone_balance": (round(m[0], 4), round(m[1], 4), round(m[2], 4)),
    }


# ── Cross-reference with our existing stocks ──────────────────────────────────

# Our existing color stocks (by base name, without category prefix)
OUR_COLOR_STOCKS = {
    # Negative
    "Kodak Portra 160", "Kodak Portra 160NC", "Kodak Portra 160VC",
    "Kodak Portra 400", "Kodak Portra 400NC", "Kodak Portra 400VC", "Kodak Portra 800",
    "Kodak Ektar 100",
    "Kodak Gold 100", "Kodak Gold 200", "Kodak Gold 400",
    "Kodak UltraMax 400", "Kodak ColorPlus 200", "Kodak Ultra Color 100UC",
    "Fuji Pro 160C", "Fuji Pro 160S", "Fuji Pro 400H", "Fuji Pro 800Z",
    "Fuji Reala 100", "Fuji Superia 200", "Fuji Superia 400", "Fuji Superia 800", "Fuji Superia 1600", "Fuji C200",
    "Agfa Vista 200", "Agfa Vista 400",
    "Cinestill 50D", "Cinestill 800T",
    "Lomography 100", "Lomography 400", "Lomography 800",
    # Slide
    "Fuji Velvia 50", "Fuji Velvia 100", "Fuji Velvia 100F",
    "Fuji Provia 100F", "Fuji Provia 400F", "Fuji Astia 100F",
    "Kodak Ektachrome 100G", "Kodak Ektachrome 100GX", "Kodak Ektachrome 100VS",
    "Kodak Ektachrome E100", "Kodak Ektachrome EES",
    "Kodak Kodachrome 25", "Kodak Kodachrome 64", "Kodak Kodachrome 200",
    "Agfa RSX II 100", "Agfachrome 1000 RS",
    "GAF 500",
    # Cinema
    "Kodak Vision3 50D", "Kodak Vision3 250D", "Kodak Vision3 200T", "Kodak Vision3 500T",
    "Kodak Vision2 500T", "Fuji Eterna 250D", "Fuji Eterna 500T", "Fuji Eterna Vivid 500",
    # Instant
    "Polaroid 600", "Polaroid SX-70", "Polaroid Spectra", "Fuji Instax",
    # Aged
    "Expired Kodak Gold", "Expired Kodak Portra", "Expired Fuji", "Expired Ektachrome",
    "Faded Kodachrome", "1970s Warm", "1980s Cool",
}

# Name mapping: C1 name → our name (where they differ)
C1_TO_OURS = {
    "Kodak Portra 160": "Kodak Portra 160",
    "Kodak Portra 160 NC": "Kodak Portra 160NC",
    "Kodak Portra 160 VC": "Kodak Portra 160VC",
    "Kodak Portra 400": "Kodak Portra 400",
    "Kodak Portra 400 NC": "Kodak Portra 400NC",
    "Kodak Portra 400 VC": "Kodak Portra 400VC",
    "Kodak Portra 400 UC": "Kodak Portra 400",  # UC maps to base
    "Kodak Portra 800": "Kodak Portra 800",
    "Kodak Portra 100T": "Kodak Portra 100T",  # we don't have this
    "Kodak Ektar 100": "Kodak Ektar 100",
    "Kodak Ektar 25": "Kodak Ektar 25",  # we don't have this
    "Kodak Gold 100": "Kodak Gold 100",
    "Kodak Gold 200": "Kodak Gold 200",
    "Kodak UltraMax 400": "Kodak UltraMax 400",
    "Kodak UltraMax 800": "Kodak UltraMax 800",  # we don't have this
    "Kodak Max 800": "Kodak Max 800",  # we don't have this
    "Kodak Royal Gold 400": "Kodak Royal Gold 400",  # we don't have this
    "Kodak BW400CN": "Kodak BW400CN",
    "Kodak E100G": "Kodak Ektachrome 100G",
    "Kodak E100VS": "Kodak Ektachrome 100VS",
    "Kodak E200": "Kodak E200",  # we don't have this
    "Kodak Ektachrome 64": "Kodak Ektachrome 64",  # we don't have this
    "Kodak Ektachrome 64T": "Kodak Ektachrome 64T",  # we don't have this
    "Kodak Elite 50 II": "Kodak Elite 50 II",  # we don't have this
    "Kodak Elite Chrome 160T": "Kodak Elite Chrome 160T",  # we don't have this
    "Fuji 160C": "Fuji Pro 160C",
    "Fuji 160S": "Fuji Pro 160S",
    "Fuji 400H": "Fuji Pro 400H",
    "Fuji 800Z": "Fuji Pro 800Z",
    "Fuji Velvia 50": "Fuji Velvia 50",
    "Fuji Velvia 100": "Fuji Velvia 100",
    "Fuji Velvia 100F": "Fuji Velvia 100F",
    "Fuji Provia 100F": "Fuji Provia 100F",
    "Fuji Provia 400X": "Fuji Provia 400F",  # close match
    "Fuji Astia 100F": "Fuji Astia 100F",
    "Fuji Sensia 100": "Fuji Sensia 100",  # we don't have this
    "Fuji Fortia SP": "Fuji Fortia SP",  # we don't have this
    "Fuji T64": "Fuji T64",  # we don't have this
    "Fuji Superia 100": "Fuji Superia 100",  # we don't have close match
    "Fuji Superia 400": "Fuji Superia 400",
    "Fuji Superia 800": "Fuji Superia 800",
    "Fuji Superia 1600": "Fuji Superia 1600",
    "Fuji Neopan 400": "Fuji Neopan 400",  # B&W
    "Fuji Neopan 1600": "Fuji Neopan 1600",  # B&W
    "Agfa Optima 100 II": "Agfa Optima 100",  # we don't have this
    "Agfa Portrait XPS 160": "Agfa Portrait XPS 160",  # we don't have this
    "Agfa RSX 200 II": "Agfa RSX II 100",  # close
    "Agfa Scala 200": "Agfa Scala 200",  # B&W
    "Agfa Ultra 50": "Agfa Ultra 50",  # we don't have this
    "Agfa Ultra 100": "Agfa Ultra 100",  # we don't have this
    "Agfa Vista 100": "Agfa Vista 100",  # we don't have this
    "Agfa Vista 400": "Agfa Vista 400",
    "Agfa Vista 800": "Agfa Vista 800",  # we don't have this
    "Polaroid 665": "Polaroid 665",  # we don't have this
    "Polaroid 669": "Polaroid 669",  # we don't have this
    "Polaroid 690": "Polaroid 690",  # we don't have this
}


def generate_report(styles: List[CostyleParsed]) -> str:
    """Generate a cross-reference report."""
    lines = []
    lines.append("=" * 80)
    lines.append("CAPTURE ONE → DARKROOM CROSS-REFERENCE REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Split into base stocks and variants
    base_stocks = [s for s in styles if not s.is_variant]
    variants = [s for s in styles if s.is_variant]

    lines.append(f"Total .costyle files: {len(styles)}")
    lines.append(f"Base stocks: {len(base_stocks)}")
    lines.append(f"Variants: {len(variants)} (Cool/Warm/+/-/Portrait/etc.)")
    lines.append("")

    # Split by color vs B&W
    color_bases = [s for s in base_stocks if not s.is_bw]
    bw_bases = [s for s in base_stocks if s.is_bw]

    lines.append(f"Color base stocks: {len(color_bases)}")
    lines.append(f"B&W base stocks: {len(bw_bases)}")
    lines.append("")

    # ── MATCH REPORT ──
    lines.append("-" * 80)
    lines.append("STOCKS WE HAVE — with C1 curve data available for refinement")
    lines.append("-" * 80)

    matched = []
    c1_names = {s.base_stock_name if s.is_variant else s.name: s for s in base_stocks}

    for c1_name, our_name in sorted(C1_TO_OURS.items()):
        if our_name in OUR_COLOR_STOCKS and c1_name in c1_names:
            matched.append((c1_name, our_name))
            s = c1_names[c1_name]
            params_r = curve_to_params(s.curve_red)
            params_g = curve_to_params(s.curve_green)
            params_b = curve_to_params(s.curve_blue)
            cb = analyze_color_balance(s)
            lines.append(f"\n  ✓ C1: {c1_name}  →  Ours: {our_name}")
            lines.append(f"    R curve: toe={params_r['toe']}, shoulder={params_r['shoulder']}, slope={params_r['slope']}")
            lines.append(f"    G curve: toe={params_g['toe']}, shoulder={params_g['shoulder']}, slope={params_g['slope']}")
            lines.append(f"    B curve: toe={params_b['toe']}, shoulder={params_b['shoulder']}, slope={params_b['slope']}")
            lines.append(f"    Shadow tint: {cb['shadow_tint']}")
            lines.append(f"    Highlight tint: {cb['highlight_tint']}")
            lines.append(f"    Saturation: {s.saturation}")
            lines.append(f"    Grain: amount={s.grain_amount}, type={s.grain_type}, granularity={s.grain_granularity}")

    lines.append(f"\n  Total matched: {len(matched)}")
    lines.append("")

    # ── MISSING FROM US ──
    lines.append("-" * 80)
    lines.append("C1 BASE STOCKS WE DON'T HAVE (candidates to add)")
    lines.append("-" * 80)

    c1_only = []
    for s in color_bases:
        name = s.name
        our_match = C1_TO_OURS.get(name)
        if our_match is None or our_match not in OUR_COLOR_STOCKS:
            c1_only.append(s)

    for s in sorted(c1_only, key=lambda x: (x.brand, x.name)):
        params_r = curve_to_params(s.curve_red)
        cb = analyze_color_balance(s)
        lines.append(f"\n  ✗ {s.brand} / {s.name}")
        lines.append(f"    R curve: toe={params_r['toe']}, shoulder={params_r['shoulder']}, slope={params_r['slope']}")
        lines.append(f"    Sat: {s.saturation}, Grain: amount={s.grain_amount} type={s.grain_type}")
        lines.append(f"    Shadow balance: {s.color_balance_shadow}")
        lines.append(f"    Highlight balance: {s.color_balance_highlight}")

    lines.append(f"\n  Total missing from us: {len(c1_only)}")
    lines.append("")

    # ── B&W STOCKS ──
    lines.append("-" * 80)
    lines.append("B&W STOCKS IN C1")
    lines.append("-" * 80)
    for s in bw_bases:
        lines.append(f"\n  {s.name}")
        lines.append(f"    Channel mix: R={s.bw_red}, G={s.bw_green}, B={s.bw_blue}")
        lines.append(f"    CMY: C={s.bw_cyan}, M={s.bw_magenta}, Y={s.bw_yellow}")
        lines.append(f"    Grain: amount={s.grain_amount}, type={s.grain_type}")

    # ── VARIANT STYLES ──
    lines.append("")
    lines.append("-" * 80)
    lines.append("VARIANT TYPES AVAILABLE")
    lines.append("-" * 80)
    variant_types = {}
    for v in variants:
        vt = v.variant_type
        if vt not in variant_types:
            variant_types[vt] = []
        variant_types[vt].append(v.base_stock_name)
    for vt, stocks in sorted(variant_types.items()):
        lines.append(f"\n  {vt}: {len(stocks)} stocks")
        for st in sorted(set(stocks))[:5]:
            lines.append(f"    - {st}")
        if len(set(stocks)) > 5:
            lines.append(f"    ... and {len(set(stocks)) - 5} more")

    return "\n".join(lines)


def export_refined_data(styles: List[CostyleParsed]) -> Dict:
    """Export all parsed data as a structured dict for further processing."""
    base_stocks = [s for s in styles if not s.is_variant]
    export = {}

    for s in base_stocks:
        entry = {
            "brand": s.brand,
            "is_bw": s.is_bw,
            "curves": {
                "master": s.curve_master,
                "red": s.curve_red,
                "green": s.curve_green,
                "blue": s.curve_blue,
                "luma": s.curve_luma,
            },
            "curve_params": {
                "red": curve_to_params(s.curve_red),
                "green": curve_to_params(s.curve_green),
                "blue": curve_to_params(s.curve_blue),
            },
            "color_balance": analyze_color_balance(s),
            "saturation": s.saturation,
            "contrast": s.contrast,
            "shadow_recovery": s.shadow_recovery,
            "highlight_recovery": s.highlight_recovery,
            "grain": {
                "amount": s.grain_amount,
                "density": s.grain_density,
                "granularity": s.grain_granularity,
                "type": s.grain_type,
            },
        }
        if s.is_bw:
            entry["bw_mix"] = {
                "red": s.bw_red, "green": s.bw_green, "blue": s.bw_blue,
                "cyan": s.bw_cyan, "magenta": s.bw_magenta, "yellow": s.bw_yellow,
            }
        export[s.name] = entry

    return export


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    styles_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "C:/Users/Jeremie/Downloads/CaptureOne_FilmStyles-master/CaptureOne_FilmStyles-master/styles"

    print(f"Parsing styles from: {styles_dir}")
    styles = parse_all_costyles(styles_dir)
    print(f"Parsed {len(styles)} styles")

    # Always generate report
    report = generate_report(styles)
    report_path = os.path.join(os.path.dirname(__file__), "costyle_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to: {report_path}")

    # Always export JSON
    data = export_refined_data(styles)
    json_path = os.path.join(os.path.dirname(__file__), "costyle_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Data exported to: {json_path}")

    # Print report to console too
    print("\n" + report)
