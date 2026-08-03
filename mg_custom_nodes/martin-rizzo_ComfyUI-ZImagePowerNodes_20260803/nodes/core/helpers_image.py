"""
File    : helpers_image.py
Purpose : Helpers functions for image color filtering using torch, torchvision and kornia
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jul 27, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
         ComfyUI nodes designed specifically for the "Z-Image" model.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import kornia
from typing import Final


def adjust_hsv_components(images: torch.Tensor,
                          *,
                          saturation_scurve_factor: float = 0.0,
                          contrast_scurve_factor  : float = 0.0,
                          saturation_target       : float = -1.0,
                          brightness_target       : float = -1.0,
                          hue_shift_factor        : float = 0.0,
                          images_color_space      : str = "rgb",
                          ) -> torch.Tensor:
    """
    Adjust HSV color space properties of RGB images using S-curve mapping and target normalization.

    Args:
        images                  : Input tensor of RGB images with shape [B, 3, H, W] and range [0, 1].
        saturation_scurve_factor: Power factor to apply an S-curve adjustment to saturation.
        contrast_scurve_factor  : Power factor to apply an S-curve adjustment to value/contrast.
        saturation_target       : Target mean saturation value (if < 0, the parameter is ignored.)
        brightness_target       : Target mean brightness value (if < 0, the parameter is ignored.)
        hue_shift_factor        : Factor to shift the hue based on the value component.
        images_color_space      : The color space that `image` is in. Can be either "rgb" or "hsv".

    Returns:
        A tensor of RGB images with the applied HSV adjustments, in the range.
    """
    if images_color_space.lower() == "rgb":
        transform_color_space = True
    elif images_color_space.lower() == "hsv":
        transform_color_space = False
    else:
        raise ValueError(f"Invalid color space. Must be either 'rgb' or 'hsv'. Got \"{images_color_space}\".")


    # transform from RGB to HSV
    hsv = kornia.color.rgb_to_hsv(images) if transform_color_space else images
    h = hsv[:, 0:1, :, :]
    s = torch.clamp(hsv[:, 1:2, :, :], 0.0, 1.0)
    v = torch.clamp(hsv[:, 2:3, :, :], 0.0, 1.0)

    # apply s-curve adjustment to saturation
    if saturation_scurve_factor > 0.0 and saturation_scurve_factor != 1.0:
        s = torch.sign(s) * torch.pow(torch.abs(s), 1.0 / saturation_scurve_factor)

    # apply s-curve adjustment to value (contrast)
    if contrast_scurve_factor > 0.0 and contrast_scurve_factor != 1.0:
        v = torch.sign(v) * torch.pow(torch.abs(v), 1.0 / contrast_scurve_factor)

    # normalize saturation to a target mean
    if saturation_target >= 0.0:
        s_mean = torch.mean(s, dim=(2, 3), keepdim=True) #< result: [B, 1, 1, 1]
        s_mean = torch.clamp(s_mean, min=1e-5)
        s = s * (saturation_target / s_mean)
        s = torch.clamp(s, 0.0, 1.0)

    # normalize brightness to a target mean
    if brightness_target >= 0.0:
        v_mean = torch.mean(v, dim=(2, 3), keepdim=True) #< result: [B, 1, 1, 1]
        v_mean = torch.clamp(v_mean, min=1e-5)
        v = v * (brightness_target / v_mean)
        v = torch.clamp(v, 0.0, 1.0)

    # apply hue shift
    if hue_shift_factor != 0.0:
        hue_shift = hue_shift_factor * (v - 0.5)
        h = torch.remainder(h + hue_shift, 2 * 3.14159)

    # reconstruct HSV and convert back to RGB
    hsv = torch.cat([h, s, v], dim=1)
    images = kornia.color.hsv_to_rgb(hsv) if transform_color_space else hsv
    return images



def stretch_histogram(images: torch.Tensor,
                      *,
                      q_lower: float = 0.001,
                      q_upper: float = 0.999
                      ) -> torch.Tensor:
    """
    Expands the dynamic range of a batch of images by stretching the histogram.

    Args:
        images : A tensor containing a batch of images. Shape [B, C, H, W]
                    with values normalized in the range [0.0, 1.0].
        q_lower: The lower quantile to ignore (e.g., 0.001 for bottom 0.1%).
        q_upper: The upper quantile to ignore (e.g., 0.999 for top 0.1%).

    Returns:
        A tensor containing the batch of images with the dynamic range expanded.
    """
    b, c, h, w = images.shape

    # flatten H, W, C dimensions per batch image -> [B, C*H*W]
    flattened = images.view(b, -1)

    # calculate quantiles for each image independently;
    # torch.quantile returns the value at which a specific percentage of data falls;
    # dim=1 ensures independent calculation per image; reshape to [B,1,1,1] allows broadcasting
    low_vals  = torch.quantile(flattened, q_lower, dim=1, keepdim=True).view (b, 1, 1, 1)
    high_vals = torch.quantile(flattened, q_upper, dim=1, keepdim=True).view (b, 1, 1, 1)

    # calculate the range of values for each image and stretch it to [0,1]
    dynamic_range = (high_vals - low_vals).clamp(min=1e-7)
    images = (images - low_vals) / dynamic_range

    # clamp as outliers outside the quantiles will exceed the [0,1] range
    return torch.clamp(images, 0.0, 1.0)



def apply_dithering(images: torch.Tensor,
                    *,
                    amplitude_bits: float = 1.0
                    ) -> torch.Tensor:
    """
    Applies dithering to reduce banding by adding noise.

    Args:
        images        : A tensor containing a batch of images. Shape [B, C, H, W]
                        with values normalized in the range [0.0, 1.0].
        amplitude_bits: Amplitude of the noise relative to an 8-bit color step.
    Returns:

        A tensor containing the batch of images with dithering applied.
    """
    # the value of a single color step in an 8-bit space (1/255 ≈ 0.00392)
    STEP_SIZE: Final = 1.0 / 255.0

    # add uniform noise centered at zero with the desired amplitude
    noise = (torch.rand_like(images) - 0.5) * (amplitude_bits * STEP_SIZE)
    return torch.clamp(images + noise, 0.0, 1.0)

