"""
LUT utilities for ComfyUI-Darkroom.
Identity lattice generation, .cube file parsing/writing, trilinear interpolation.
"""

import os
import numpy as np


def generate_identity_lut(size=33):
    """
    Generate a 3D identity LUT as a 2D image.

    Layout: width = size, height = size * size.
    Pixel at (x, y): R = x/(size-1), G = (y%size)/(size-1), B = (y//size)/(size-1).
    Reading row-by-row, left-to-right yields .cube ordering (R fastest, then G, then B).

    Returns: (size*size, size, 3) float32 array in [0, 1].
    """
    r = np.linspace(0.0, 1.0, size, dtype=np.float32)
    g = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b = np.linspace(0.0, 1.0, size, dtype=np.float32)

    # Build the image row by row
    # y = b_idx * size + g_idx, x = r_idx
    img = np.empty((size * size, size, 3), dtype=np.float32)
    for b_idx in range(size):
        for g_idx in range(size):
            y = b_idx * size + g_idx
            img[y, :, 0] = r                    # R varies across columns
            img[y, :, 1] = g[g_idx]             # G constant per row within a B-block
            img[y, :, 2] = b[b_idx]             # B constant per block

    return img


def image_to_lut_3d(image, size):
    """
    Convert a processed identity lattice image back to a 3D LUT array.

    image: (size*size, size, 3) float32 — the processed lattice.
    Returns: (size, size, size, 3) float32 — lut_3d[r, g, b] = output RGB.
    """
    lut_3d = np.empty((size, size, size, 3), dtype=np.float32)
    for b_idx in range(size):
        for g_idx in range(size):
            y = b_idx * size + g_idx
            lut_3d[:, g_idx, b_idx, :] = image[y, :, :]  # row = all R values
    return lut_3d


def write_cube_file(filepath, lut_3d, size, title="Darkroom Grade"):
    """
    Write a 3D LUT in Adobe .cube format.

    lut_3d: (size, size, size, 3) float32 — lut_3d[r, g, b] = output RGB.
    Order: R changes fastest, then G, then B.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(f"# Created by ComfyUI-Darkroom (AKURATE)\n")
        f.write(f"TITLE \"{title}\"\n")
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write(f"DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write(f"DOMAIN_MAX 1.0 1.0 1.0\n\n")

        for b_idx in range(size):
            for g_idx in range(size):
                for r_idx in range(size):
                    r, g, b = lut_3d[r_idx, g_idx, b_idx]
                    f.write(f"{r:.6f} {g:.6f} {b:.6f}\n")


def parse_cube_file(filepath):
    """
    Parse an Adobe .cube 3D LUT file.

    Returns: (lut_3d, size) where lut_3d is (size, size, size, 3) float32.
    """
    size = None
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    data_lines = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('TITLE'):
                continue
            if line.startswith('LUT_3D_SIZE'):
                size = int(line.split()[-1])
                continue
            if line.startswith('LUT_1D_SIZE'):
                raise ValueError("1D LUTs are not supported — use a 3D .cube file")
            if line.startswith('DOMAIN_MIN'):
                domain_min = np.array([float(x) for x in line.split()[1:4]], dtype=np.float32)
                continue
            if line.startswith('DOMAIN_MAX'):
                domain_max = np.array([float(x) for x in line.split()[1:4]], dtype=np.float32)
                continue

            # Data line — three floats
            parts = line.split()
            if len(parts) >= 3:
                try:
                    data_lines.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue

    if size is None:
        raise ValueError(f"No LUT_3D_SIZE found in {filepath}")

    expected = size ** 3
    if len(data_lines) != expected:
        raise ValueError(
            f"Expected {expected} entries for size {size}, got {len(data_lines)} in {filepath}"
        )

    # Normalize to 0-1 if domain is non-standard
    data = np.array(data_lines, dtype=np.float32)
    domain_range = domain_max - domain_min
    domain_range[domain_range == 0] = 1.0
    data = (data - domain_min) / domain_range

    # Reshape: .cube order is R fastest, then G, then B
    lut_3d = np.empty((size, size, size, 3), dtype=np.float32)
    idx = 0
    for b_idx in range(size):
        for g_idx in range(size):
            for r_idx in range(size):
                lut_3d[r_idx, g_idx, b_idx] = data[idx]
                idx += 1

    return lut_3d, size


def apply_lut_trilinear(image, lut_3d, lut_size):
    """
    Apply a 3D LUT to an image using trilinear interpolation.

    image: (H, W, 3) float32 in [0, 1].
    lut_3d: (size, size, size, 3) float32.
    Returns: (H, W, 3) float32.
    """
    h, w, _ = image.shape
    img = np.clip(image, 0.0, 1.0)

    # Scale to LUT grid coordinates
    scale = float(lut_size - 1)
    coords = img * scale

    # Floor and ceil indices
    floor_idx = np.floor(coords).astype(np.int32)
    floor_idx = np.clip(floor_idx, 0, lut_size - 2)
    ceil_idx = floor_idx + 1

    # Fractional parts
    frac = coords - floor_idx.astype(np.float32)

    r0 = floor_idx[..., 0]
    g0 = floor_idx[..., 1]
    b0 = floor_idx[..., 2]
    r1 = ceil_idx[..., 0]
    g1 = ceil_idx[..., 1]
    b1 = ceil_idx[..., 2]
    fr = frac[..., 0:1]
    fg = frac[..., 1:2]
    fb = frac[..., 2:3]

    # 8 corners of the interpolation cube
    c000 = lut_3d[r0, g0, b0]
    c100 = lut_3d[r1, g0, b0]
    c010 = lut_3d[r0, g1, b0]
    c110 = lut_3d[r1, g1, b0]
    c001 = lut_3d[r0, g0, b1]
    c101 = lut_3d[r1, g0, b1]
    c011 = lut_3d[r0, g1, b1]
    c111 = lut_3d[r1, g1, b1]

    # Trilinear interpolation
    c00 = c000 * (1.0 - fr) + c100 * fr
    c10 = c010 * (1.0 - fr) + c110 * fr
    c01 = c001 * (1.0 - fr) + c101 * fr
    c11 = c011 * (1.0 - fr) + c111 * fr

    c0 = c00 * (1.0 - fg) + c10 * fg
    c1 = c01 * (1.0 - fg) + c11 * fg

    result = c0 * (1.0 - fb) + c1 * fb

    return np.clip(result, 0.0, 1.0).astype(np.float32)
