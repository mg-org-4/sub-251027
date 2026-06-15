"""
Image processing utilities for ComfyUI-Darkroom.
Handles tensor↔numpy conversion and blur operations.
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve


def tensor_to_numpy_batch(tensor):
    """
    Convert ComfyUI IMAGE tensor (B, H, W, C) to list of numpy (H, W, C) float32 arrays.
    """
    arrays = []
    for i in range(tensor.shape[0]):
        arr = tensor[i].cpu().numpy().astype(np.float32)
        arrays.append(arr)
    return arrays


def numpy_batch_to_tensor(arrays):
    """
    Convert list of numpy (H, W, C) float32 arrays to ComfyUI IMAGE tensor (B, H, W, C).
    """
    tensors = [torch.from_numpy(arr).unsqueeze(0) for arr in arrays]
    return torch.cat(tensors, dim=0)


def process_batch(tensor, func, **kwargs):
    """
    Apply a per-image processing function to each image in a batch tensor.
    func signature: func(img_numpy_hwc, **kwargs) -> img_numpy_hwc
    Returns: ComfyUI IMAGE tensor (B, H, W, C)
    """
    arrays = tensor_to_numpy_batch(tensor)
    processed = [func(arr, **kwargs) for arr in arrays]
    return numpy_batch_to_tensor(processed)


def disk_kernel(radius):
    """
    Generate a normalized disk (circle) blur kernel.
    Returns (2*radius+1, 2*radius+1) float32 array.
    """
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = (x * x + y * y) <= (radius * radius)
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[mask] = 1.0
    kernel /= kernel.sum()
    return kernel


def apply_gaussian_blur(img, radius, sigma=None):
    """
    Apply Gaussian blur per-channel.
    img: (H, W, C) float32. radius: approximate kernel radius. sigma: defaults to radius/3.
    """
    if radius <= 0:
        return img
    if sigma is None:
        sigma = max(radius / 3.0, 0.5)
    result = np.empty_like(img)
    for c in range(img.shape[2]):
        result[..., c] = gaussian_filter(img[..., c], sigma=sigma)
    return result.astype(np.float32)


def apply_disk_blur(img, radius):
    """
    Apply disk (uniform circle) blur per-channel via FFT convolution.
    Physically accurate for halation. Slower than Gaussian.
    """
    if radius <= 0:
        return img
    kernel = disk_kernel(radius)
    result = np.empty_like(img)
    for c in range(img.shape[2]):
        result[..., c] = fftconvolve(img[..., c], kernel, mode='same')
    return np.clip(result, 0.0, 1.0).astype(np.float32)
