"""
Scope rendering utilities.
Pure numpy + PIL drawing for Histogram and Vectorscope nodes.
"""

from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont


_BG = (18, 18, 22)
_GRATICULE = (64, 68, 78)
_GRATICULE_STRONG = (110, 115, 128)
_LABEL = (180, 185, 200)

# Rec.709 luma weights
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _font(size: int = 11) -> ImageFont.ImageFont:
    # Default PIL bitmap font — no file dependency. Size arg ignored on bitmap.
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _to_uint8(canvas: np.ndarray) -> np.ndarray:
    return np.clip(canvas * 255.0, 0, 255).astype(np.uint8)


def _new_canvas(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), _BG)
    return img


def render_histogram(
    rgb: np.ndarray,
    *,
    mode: str = "RGB",
    width: int = 512,
    height: int = 256,
    bins: int = 256,
    log_scale: bool = False,
) -> np.ndarray:
    """
    Draw a waveform-style histogram. rgb is HxWx3 float32 in [0,1].
    mode: 'RGB' (overlay), 'Luma', 'R', 'G', 'B'.
    Returns HxWx3 float32 in [0,1].
    """
    img = _new_canvas(width, height)
    d = ImageDraw.Draw(img, "RGBA")

    pad_l, pad_r, pad_t, pad_b = 36, 12, 14, 22
    plot_x0 = pad_l
    plot_x1 = width - pad_r
    plot_y0 = pad_t
    plot_y1 = height - pad_b
    pw = plot_x1 - plot_x0
    ph = plot_y1 - plot_y0

    # Graticule: vertical lines at 0, 25, 50, 75, 100% input, horizontal
    # guidelines at 25 / 50 / 75 % of count.
    for pct in (0, 25, 50, 75, 100):
        x = plot_x0 + int(round(pct / 100 * pw))
        col = _GRATICULE_STRONG if pct in (0, 50, 100) else _GRATICULE
        d.line([(x, plot_y0), (x, plot_y1)], fill=col, width=1)
    for pct in (25, 50, 75):
        y = plot_y1 - int(round(pct / 100 * ph))
        d.line([(plot_x0, y), (plot_x1, y)], fill=_GRATICULE, width=1)
    # Frame
    d.rectangle([(plot_x0, plot_y0), (plot_x1, plot_y1)], outline=_GRATICULE_STRONG, width=1)

    def _hist(values: np.ndarray) -> np.ndarray:
        h_counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
        return h_counts.astype(np.float64)

    channels: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    if mode == "RGB":
        channels = [
            (_hist(rgb[..., 0]), (240, 70, 70)),
            (_hist(rgb[..., 1]), (70, 220, 90)),
            (_hist(rgb[..., 2]), (80, 140, 255)),
        ]
    elif mode == "Luma":
        luma = (rgb * _LUMA).sum(axis=-1)
        channels = [(_hist(luma), (230, 230, 230))]
    elif mode in ("R", "G", "B"):
        idx = {"R": 0, "G": 1, "B": 2}[mode]
        col = {"R": (240, 70, 70), "G": (70, 220, 90), "B": (80, 140, 255)}[mode]
        channels = [(_hist(rgb[..., idx]), col)]
    else:
        raise ValueError(f"unknown histogram mode: {mode}")

    # Normalise all channels by the global max so they share a vertical scale.
    all_max = max((c[0].max() for c in channels), default=1.0)
    if all_max <= 0:
        all_max = 1.0

    for counts, color in channels:
        if log_scale:
            counts = np.log1p(counts)
            scale_max = np.log1p(all_max)
        else:
            scale_max = all_max
        scale_max = max(scale_max, 1.0)

        pts: list[tuple[int, int]] = []
        for i in range(bins):
            x = plot_x0 + int(round((i + 0.5) / bins * pw))
            v = counts[i] / scale_max
            y = plot_y1 - int(round(v * ph))
            pts.append((x, y))
        # Fill area under the curve with translucent channel colour.
        poly = [(plot_x0, plot_y1), *pts, (plot_x1, plot_y1)]
        d.polygon(poly, fill=(*color, 60))
        d.line(pts, fill=(*color, 220), width=1)

    # Clipped-pixel warning stripes at the edges (black/white clipping).
    total = rgb.shape[0] * rgb.shape[1]
    if total > 0:
        if mode in ("RGB", "R", "G", "B"):
            ch_idx = {"R": 0, "G": 1, "B": 2}.get(mode, None)
            if ch_idx is None:
                ch_data = rgb
            else:
                ch_data = rgb[..., ch_idx:ch_idx + 1]
            clip_lo = float(np.mean(np.any(ch_data <= 0.0, axis=-1))) if ch_data.ndim == 3 else float(np.mean(ch_data <= 0.0))
            clip_hi = float(np.mean(np.any(ch_data >= 1.0, axis=-1))) if ch_data.ndim == 3 else float(np.mean(ch_data >= 1.0))
        else:
            luma = (rgb * _LUMA).sum(axis=-1)
            clip_lo = float(np.mean(luma <= 0.0))
            clip_hi = float(np.mean(luma >= 1.0))

        if clip_lo > 0.001:
            d.rectangle([(plot_x0 - 3, plot_y0), (plot_x0 - 1, plot_y1)], fill=(220, 30, 30))
        if clip_hi > 0.001:
            d.rectangle([(plot_x1 + 1, plot_y0), (plot_x1 + 3, plot_y1)], fill=(220, 30, 30))

    # Axis labels
    font = _font(10)
    for pct in (0, 50, 100):
        x = plot_x0 + int(round(pct / 100 * pw))
        d.text((x - 6, plot_y1 + 4), f"{pct}%", fill=_LABEL, font=font)
    d.text((4, (plot_y0 + plot_y1) // 2 - 5), mode, fill=_LABEL, font=font)

    return np.asarray(img, dtype=np.float32) / 255.0


def render_vectorscope(
    rgb: np.ndarray,
    *,
    size: int = 512,
    gain: float = 1.0,
    log_scale: bool = True,
    show_skin_line: bool = True,
) -> np.ndarray:
    """
    Draw a vectorscope in Rec.709 YCbCr space.
    rgb is HxWx3 float32 in [0,1]. Returns SIZExSIZEx3 float32 in [0,1].
    """
    w = h = int(size)
    img = _new_canvas(w, h)
    d = ImageDraw.Draw(img, "RGBA")

    cx, cy = w / 2.0, h / 2.0
    # 100% Cb/Cr reach for a fully-saturated BT.709 primary is ~0.5 in
    # normalized YCbCr; scale so that the 100% ring touches ~90% of the
    # canvas radius for legibility.
    r_100 = min(w, h) * 0.45
    r_75 = r_100 * 0.75

    # Rec.709 YCbCr (analog, full range). Maps RGB in [0,1] to (Y, Cb, Cr)
    # where Cb, Cr are in approximately [-0.5, 0.5].
    m = np.array(
        [[0.2126, 0.7152, 0.0722],
         [-0.1146, -0.3854, 0.5000],
         [0.5000, -0.4542, -0.0458]],
        dtype=np.float32,
    )
    ycbcr = rgb.reshape(-1, 3) @ m.T
    cb = ycbcr[:, 1]
    cr = ycbcr[:, 2]

    # Accumulate into a density grid.
    # Cb maps to x (right = +Cb = blue), Cr maps to y (up = +Cr = red).
    # Display convention: +Cr should point toward red (upper-left area on
    # standard broadcast scopes; but for simplicity we put red at top).
    x_coord = cx + cb * r_100 * 2.0 * gain
    y_coord = cy - cr * r_100 * 2.0 * gain  # subtract because image y grows down

    xs = np.clip(np.round(x_coord).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(y_coord).astype(np.int32), 0, h - 1)
    density = np.zeros((h, w), dtype=np.float64)
    np.add.at(density, (ys, xs), 1.0)

    if density.max() > 0:
        vis = np.log1p(density) if log_scale else density
        vis = vis / vis.max()
    else:
        vis = density

    # Colour map the density: warm cyan->green->yellow-white heatmap.
    cm = _density_colormap(vis)  # HxWx3 uint8
    trace_img = Image.fromarray(cm, mode="RGB").convert("RGBA")
    # Mask the trace to where density > 0 so background stays clean.
    alpha = (vis > 0).astype(np.uint8) * 230
    trace_img.putalpha(Image.fromarray(alpha, mode="L"))
    img.paste(trace_img, (0, 0), trace_img)

    # Re-grab the draw object after paste (PIL quirk: paste keeps the draw
    # bound to the same surface, but safer to reassign).
    d = ImageDraw.Draw(img, "RGBA")

    # --- Graticule ---
    # 100% ring
    d.ellipse(
        [(cx - r_100, cy - r_100), (cx + r_100, cy + r_100)],
        outline=_GRATICULE_STRONG, width=1,
    )
    # 75% ring
    d.ellipse(
        [(cx - r_75, cy - r_75), (cx + r_75, cy + r_75)],
        outline=_GRATICULE, width=1,
    )
    # Cross-hairs
    d.line([(cx - r_100, cy), (cx + r_100, cy)], fill=_GRATICULE, width=1)
    d.line([(cx, cy - r_100), (cx, cy + r_100)], fill=_GRATICULE, width=1)

    # Target boxes for 75% primaries.
    # Rec.709 @ 75%: RGB corners of the 75% cube map to fixed YCbCr targets.
    primaries = {
        "R":  (1.0, 0.0, 0.0),
        "Yl": (1.0, 1.0, 0.0),
        "G":  (0.0, 1.0, 0.0),
        "Cy": (0.0, 1.0, 1.0),
        "B":  (0.0, 0.0, 1.0),
        "Mg": (1.0, 0.0, 1.0),
    }
    target_colors = {
        "R": (240, 70, 70), "Yl": (235, 215, 80), "G": (70, 220, 90),
        "Cy": (70, 220, 220), "B": (80, 140, 255), "Mg": (220, 90, 220),
    }
    font = _font(10)
    for label, rgb_primary in primaries.items():
        p = np.array(rgb_primary, dtype=np.float32) * 0.75
        ycbcr_p = m @ p
        tx = cx + ycbcr_p[1] * r_100 * 2.0
        ty = cy - ycbcr_p[2] * r_100 * 2.0
        # Target box
        box = 6
        d.rectangle(
            [(tx - box, ty - box), (tx + box, ty + box)],
            outline=target_colors[label], width=1,
        )
        # Label just outside the box
        d.text((tx + box + 2, ty - box - 1), label, fill=target_colors[label], font=font)

    # Skin-tone line (I-line at ~123 degrees in the Cb/Cr plane)
    if show_skin_line:
        angle = math.radians(123)
        dx = math.cos(angle) * r_100
        dy = -math.sin(angle) * r_100
        d.line(
            [(cx - dx, cy - dy), (cx + dx, cy + dy)],
            fill=(200, 160, 120, 180), width=1,
        )
        d.text((cx + dx * 0.82 + 6, cy + dy * 0.82 - 6), "skin", fill=(200, 160, 120), font=font)

    # Title
    d.text((8, 6), "VECTORSCOPE  (Rec.709 YCbCr)", fill=_LABEL, font=font)

    return np.asarray(img, dtype=np.float32) / 255.0


def _density_colormap(vis: np.ndarray) -> np.ndarray:
    """Cold-to-warm colormap for vectorscope density."""
    # Stops: 0 -> very dark blue, 0.3 -> cyan, 0.6 -> yellow-green,
    # 0.85 -> orange, 1.0 -> near-white
    stops = np.array([
        [10, 20, 60],
        [20, 100, 160],
        [60, 200, 180],
        [230, 220, 100],
        [255, 240, 220],
    ], dtype=np.float32)
    xs = np.array([0.0, 0.3, 0.6, 0.85, 1.0], dtype=np.float32)

    out = np.zeros((*vis.shape, 3), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(vis, xs, stops[:, c])
    return np.clip(out, 0, 255).astype(np.uint8)
