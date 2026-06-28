"""
Perspective Correction node for ComfyUI-Darkroom.

Corrects converging verticals (keystone) and horizontal perspective shift
common in architectural and product photography.

Uses projective (homography) transformation with bilinear interpolation.
"""

import numpy as np
from scipy.ndimage import map_coordinates

from ..utils.image import tensor_to_numpy_batch, numpy_batch_to_tensor


def _apply_perspective(img, vertical, horizontal, rotation):
    """
    Apply perspective correction via projective transform.

    vertical: keystone correction (-1 to 1). Positive tilts top away.
    horizontal: horizontal keystone (-1 to 1). Positive tilts right away.
    rotation: rotation in degrees (-15 to 15).
    """
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0

    # Build coordinate grid (normalized -1 to 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ny = (yy - cy) / cy
    nx = (xx - cx) / cx

    # Apply rotation first
    if abs(rotation) > 0.01:
        theta = np.radians(rotation)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rnx = nx * cos_t - ny * sin_t
        rny = nx * sin_t + ny * cos_t
        nx, ny = rnx, rny

    # Apply vertical perspective (keystone)
    # Top-to-bottom scaling: at ny=-1 (top), scale is (1 + v), at ny=1 (bottom), scale is (1 - v)
    if abs(vertical) > 0.001:
        v_scale = 1.0 / (1.0 + vertical * ny)
        nx = nx * v_scale

    # Apply horizontal perspective
    if abs(horizontal) > 0.001:
        h_scale = 1.0 / (1.0 + horizontal * nx)
        ny = ny * h_scale

    # Convert back to pixel coordinates
    src_x = nx * cx + cx
    src_y = ny * cy + cy

    # Remap
    result = np.empty_like(img)
    for c in range(img.shape[2]):
        result[..., c] = map_coordinates(
            img[..., c], [src_y, src_x],
            order=1, mode='constant', cval=0.0
        ).astype(np.float32)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


class PerspectiveCorrect:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vertical": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01,
                    "tooltip": "Vertical keystone. Positive corrects converging verticals (building lean-back)"
                }),
            },
            "optional": {
                "horizontal": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01,
                    "tooltip": "Horizontal keystone. Corrects left-right convergence"
                }),
                "rotation": ("FLOAT", {
                    "default": 0.0, "min": -15.0, "max": 15.0, "step": 0.1,
                    "tooltip": "Rotation correction in degrees"
                }),
                "auto_crop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically crop to remove black borders from the transform"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/Lens"

    def execute(self, image, vertical, horizontal=0.0, rotation=0.0, auto_crop=True):
        if abs(vertical) < 0.001 and abs(horizontal) < 0.001 and abs(rotation) < 0.01:
            return (image,)

        print(f"[Darkroom] Perspective: vertical={vertical}, horizontal={horizontal}, rotation={rotation}")

        arrays = tensor_to_numpy_batch(image)
        processed = []

        for img in arrays:
            result = _apply_perspective(img, vertical, horizontal, rotation)

            if auto_crop:
                result = _auto_crop(result)

            processed.append(result)

        return (numpy_batch_to_tensor(processed),)


def _auto_crop(img):
    """
    Crop out black borders introduced by perspective transform.
    Finds the largest inner rectangle with no black pixels.
    """
    # Sum across channels — black pixels have sum ~0
    gray = img.sum(axis=2)

    # Find rows and columns that are mostly non-black
    row_sums = (gray > 0.01).sum(axis=1)
    col_sums = (gray > 0.01).sum(axis=0)

    h, w = img.shape[:2]
    threshold_row = w * 0.9  # at least 90% of width should be non-black
    threshold_col = h * 0.9

    valid_rows = np.where(row_sums > threshold_row)[0]
    valid_cols = np.where(col_sums > threshold_col)[0]

    if len(valid_rows) < 10 or len(valid_cols) < 10:
        return img  # can't crop meaningfully

    top = valid_rows[0]
    bottom = valid_rows[-1] + 1
    left = valid_cols[0]
    right = valid_cols[-1] + 1

    cropped = img[top:bottom, left:right]

    # Resize back to original dimensions using simple area sampling
    from scipy.ndimage import zoom
    scale_y = h / cropped.shape[0]
    scale_x = w / cropped.shape[1]
    result = zoom(cropped, (scale_y, scale_x, 1.0), order=1)

    return np.clip(result, 0.0, 1.0).astype(np.float32)


NODE_CLASS_MAPPINGS = {
    "DarkroomPerspectiveCorrect": PerspectiveCorrect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DarkroomPerspectiveCorrect": "Perspective Correct",
}
