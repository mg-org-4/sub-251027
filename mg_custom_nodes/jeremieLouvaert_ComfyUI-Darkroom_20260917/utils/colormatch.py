"""
Reference-driven colour-matching algorithms.

Adapted from Gradia (rajawski/gradia, MIT) -- four LAB-space transport methods
wrapped to take numpy float32 RGB arrays in [0, 1] instead of cv2 BGR uint8
files. All operations internally round-trip through uint8 LAB for the cv2
colour-space conversion; this matches the source implementation exactly
(including its subtle uint8-LAB-range convention) at the cost of 8-bit
precision on the LAB conversion step.

Attribution: https://github.com/rajawski/gradia (MIT)
"""

from __future__ import annotations

import numpy as np
import cv2


def _rgb_float_to_bgr_u8(rgb: np.ndarray) -> np.ndarray:
    x = np.clip(rgb, 0.0, 1.0) * 255.0
    x = x.astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def _bgr_u8_to_rgb_float(bgr_u8: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _to_lab_u8(bgr_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)


def _from_lab_u8(lab_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Reinhard - statistical transfer in LAB
# ---------------------------------------------------------------------------

def color_match_reinhard(target_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    """
    Per-channel LAB mean/std transfer (Reinhard et al. 2001).
    Safe default; works well on single-subject scenes.
    """
    ref_bgr = _rgb_float_to_bgr_u8(reference_rgb)
    tgt_bgr = _rgb_float_to_bgr_u8(target_rgb)

    ref_lab = _to_lab_u8(ref_bgr).astype(np.float32)
    tgt_lab = _to_lab_u8(tgt_bgr).astype(np.float32)

    result_lab = tgt_lab.copy()
    for i in range(3):
        ref_mean, ref_std = ref_lab[:, :, i].mean(), ref_lab[:, :, i].std()
        tgt_mean, tgt_std = tgt_lab[:, :, i].mean(), tgt_lab[:, :, i].std()
        if tgt_std < 1e-6:
            result_lab[:, :, i] = tgt_lab[:, :, i] - tgt_mean + ref_mean
        else:
            result_lab[:, :, i] = (
                (tgt_lab[:, :, i] - tgt_mean) * (ref_std / tgt_std) + ref_mean
            )

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return _bgr_u8_to_rgb_float(_from_lab_u8(result_lab))


# ---------------------------------------------------------------------------
# Wasserstein - sliced OT via iterative advection
# ---------------------------------------------------------------------------

def color_match_wasserstein(
    target_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    *,
    n_slices: int = 20,
    sample_size: int = 50000,
    seed: int = 42,
    step_factor: float = 0.5,
) -> np.ndarray:
    """
    Sliced Wasserstein optimal transport via iterative advection in LAB.
    Handles multi-modal distributions that Reinhard's linear shift misses.
    """
    ref_bgr = _rgb_float_to_bgr_u8(reference_rgb)
    tgt_bgr = _rgb_float_to_bgr_u8(target_rgb)

    ref_lab = _to_lab_u8(ref_bgr).astype(np.float32) / 255.0
    tgt_lab = _to_lab_u8(tgt_bgr).astype(np.float32) / 255.0

    h, w = tgt_lab.shape[:2]
    ref_flat = ref_lab.reshape(-1, 3)
    tgt_flat = tgt_lab.reshape(-1, 3)

    rng = np.random.default_rng(seed)
    ref_sample = ref_flat[
        rng.choice(len(ref_flat), min(sample_size, len(ref_flat)), replace=False)
    ]

    result_flat = tgt_flat.copy()
    for _ in range(n_slices):
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)

        ref_proj = ref_sample @ direction
        tgt_proj = result_flat @ direction

        ref_sorted = np.sort(ref_proj)
        n_ref = len(ref_sorted)
        n_tgt = len(tgt_proj)

        tgt_argsort = np.argsort(tgt_proj)
        tgt_ranks = np.empty(n_tgt, dtype=np.int64)
        tgt_ranks[tgt_argsort] = np.arange(n_tgt)

        ref_indices = np.clip(
            (tgt_ranks * (n_ref - 1) / max(n_tgt - 1, 1)).astype(np.int64),
            0, n_ref - 1,
        )

        displacement = ref_sorted[ref_indices] - tgt_proj
        result_flat += step_factor * np.outer(displacement, direction)

    result_flat = np.clip(result_flat, 0, 1)
    # Critical: cast back to uint8 LAB before cv2.COLOR_LAB2BGR. The cv2
    # colour-conversion expects [0,255] uint8 LAB; feeding it float32 values
    # with the same numerical magnitudes triggers a silent range
    # reinterpretation that wrecks the a/b channels.
    result_lab = (result_flat.reshape(h, w, 3) * 255.0).astype(np.uint8)
    return _bgr_u8_to_rgb_float(_from_lab_u8(result_lab))


# ---------------------------------------------------------------------------
# Forgy - K-means palette matching with soft Gaussian assignment
# ---------------------------------------------------------------------------

def color_match_forgy(
    target_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    *,
    n_colors: int = 8,
    sample_size: int = 50000,
    seed: int = 42,
) -> np.ndarray:
    """
    Palette-based matching. Fit K-means to both images, map target
    centres to nearest reference centres, apply Gaussian-weighted delta
    per pixel. Requires sklearn.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise RuntimeError(
            "ColorMatch 'forgy' method requires scikit-learn. "
            "Install with: pip install scikit-learn"
        ) from e

    ref_bgr = _rgb_float_to_bgr_u8(reference_rgb)
    tgt_bgr = _rgb_float_to_bgr_u8(target_rgb)

    ref_lab = _to_lab_u8(ref_bgr).astype(np.float32)
    tgt_lab = _to_lab_u8(tgt_bgr).astype(np.float32)

    h, w = tgt_lab.shape[:2]
    rng = np.random.default_rng(seed)

    def _subsample(lab_img: np.ndarray) -> np.ndarray:
        flat = lab_img.reshape(-1, 3)
        if len(flat) > sample_size:
            idx = rng.choice(len(flat), sample_size, replace=False)
            return flat[idx]
        return flat

    ref_samples = _subsample(ref_lab)
    tgt_samples = _subsample(tgt_lab)

    km_ref = KMeans(n_clusters=n_colors, random_state=seed, n_init="auto").fit(ref_samples)
    km_tgt = KMeans(n_clusters=n_colors, random_state=seed, n_init="auto").fit(tgt_samples)

    ref_centers = km_ref.cluster_centers_
    tgt_centers = km_tgt.cluster_centers_

    tgt_flat = tgt_lab.reshape(-1, 3)
    result_flat = tgt_flat.copy()

    deltas = np.zeros((n_colors, 3), dtype=np.float32)
    for ci in range(n_colors):
        tgt_color = tgt_centers[ci]
        dists = np.linalg.norm(ref_centers - tgt_color, axis=1)
        ref_match = ref_centers[np.argmin(dists)]
        deltas[ci] = ref_match - tgt_color

    center_dists = np.linalg.norm(
        tgt_centers[:, None, :] - tgt_centers[None, :, :], axis=2
    )
    upper = center_dists[np.triu_indices(n_colors, k=1)]
    sigma = (np.median(upper) + 1e-6) * 0.75

    pixel_to_centers = np.linalg.norm(
        tgt_flat[:, None, :] - tgt_centers[None, :, :], axis=2
    )
    weights = np.exp(-0.5 * (pixel_to_centers / sigma) ** 2)
    weights /= weights.sum(axis=1, keepdims=True) + 1e-12

    blended_delta = weights @ deltas
    result_flat += blended_delta

    result_lab = np.clip(result_flat.reshape(h, w, 3), 0, 255).astype(np.uint8)
    return _bgr_u8_to_rgb_float(_from_lab_u8(result_lab))


# ---------------------------------------------------------------------------
# Kantorovich - linear Gaussian optimal transport (requires POT)
# ---------------------------------------------------------------------------

def color_match_kantorovich(
    target_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    *,
    sample_size: int = 50000,
    seed: int = 42,
) -> np.ndarray:
    """
    Monge-Kantorovich closed-form Gaussian OT in LAB. Requires the POT
    library (pip install POT).
    """
    try:
        import ot  # noqa: F401
        from ot.da import LinearTransport
    except ImportError as e:
        raise RuntimeError(
            "ColorMatch 'kantorovich' method requires the POT library. "
            "Install with: pip install POT"
        ) from e

    ref_bgr = _rgb_float_to_bgr_u8(reference_rgb)
    tgt_bgr = _rgb_float_to_bgr_u8(target_rgb)

    ref_lab = _to_lab_u8(ref_bgr).astype(np.float32) / 255.0
    tgt_lab = _to_lab_u8(tgt_bgr).astype(np.float32) / 255.0

    h, w = tgt_lab.shape[:2]
    ref_flat = ref_lab.reshape(-1, 3)
    tgt_flat = tgt_lab.reshape(-1, 3)

    rng = np.random.default_rng(seed)
    ref_sample = ref_flat[
        rng.choice(len(ref_flat), min(sample_size, len(ref_flat)), replace=False)
    ]
    tgt_sample = tgt_flat[
        rng.choice(len(tgt_flat), min(sample_size, len(tgt_flat)), replace=False)
    ]

    ot_map = LinearTransport()
    ot_map.fit(Xs=tgt_sample, Xt=ref_sample)

    result_flat = ot_map.transform(Xs=tgt_flat)
    result_flat = np.clip(result_flat, 0, 1)

    result_lab = (result_flat.reshape(h, w, 3) * 255.0).astype(np.uint8)
    return _bgr_u8_to_rgb_float(_from_lab_u8(result_lab))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

METHODS = {
    "reinhard": color_match_reinhard,
    "wasserstein": color_match_wasserstein,
    "forgy": color_match_forgy,
    "kantorovich": color_match_kantorovich,
}


def is_method_available(method: str) -> tuple[bool, str]:
    """Return (available, reason) for a method."""
    if method in ("reinhard", "wasserstein"):
        return True, ""
    if method == "forgy":
        try:
            import sklearn  # noqa: F401
            return True, ""
        except ImportError:
            return False, "scikit-learn not installed"
    if method == "kantorovich":
        try:
            import ot  # noqa: F401
            return True, ""
        except ImportError:
            return False, "POT not installed"
    return False, f"unknown method: {method}"
