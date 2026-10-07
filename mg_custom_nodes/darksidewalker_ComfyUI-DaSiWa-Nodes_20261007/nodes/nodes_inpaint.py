import torch
import torch.nn.functional as F


def _extract_bounds(mask, grow):
    """Extract bounding box from mask with grow padding.

    mask: [H, W] or [B, H, W] float tensor in [0, 1].
    Returns: (x, y, w, h) clamped to image bounds.
    """
    m = mask[0] if mask.ndim >= 3 else mask
    coords = torch.nonzero(m > 0.5, as_tuple=True)
    if len(coords[0]) == 0:
        raise ValueError("DaSiWa Inpaint: mask is empty (no pixels > 0.5).")

    y_min = int(coords[0].min())
    y_max = int(coords[0].max())
    x_min = int(coords[1].min())
    x_max = int(coords[1].max())

    h, w = m.shape
    grow = max(0, int(grow))

    x = max(0, x_min - grow)
    y = max(0, y_min - grow)
    bw = min(w - x, x_max - x_min + 1 + 2 * grow)
    bh = min(h - y, y_max - y_min + 1 + 2 * grow)

    return x, y, bw, bh


def _gaussian_blur_2d(tensor, sigma):
    """Separable Gaussian blur without torchvision.

    tensor: [B, C, H, W]
    sigma: float (standard deviation)
    Returns: blurred [B, C, H, W]
    """
    if sigma <= 0:
        return tensor

    radius = max(1, int(sigma * 3))
    kernel_size = radius * 2 + 1
    x = torch.arange(kernel_size, device=tensor.device, dtype=tensor.dtype) - radius
    kernel_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2)).unsqueeze(0)
    kernel_1d /= kernel_1d.sum()

    # Horizontal pass
    out = F.conv2d(tensor, kernel_1d.unsqueeze(0).unsqueeze(0), padding=(radius, 0), groups=tensor.shape[1])
    # Vertical pass
    out = F.conv2d(out, kernel_1d.unsqueeze(0).unsqueeze(0).transpose(2, 3), padding=(0, radius), groups=out.shape[1])
    return out


def _apply_mask_blur_and_threshold(mask, blur_radius, min_val, max_val):
    """Gaussian-blur mask then clamp to [min_val, max_val].

    mask: [B, H, W] or [H, W].
    Returns: [H, W] blurred and clamped mask.
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)

    padded = mask.unsqueeze(1)  # [B, 1, H, W]
    blurred = _gaussian_blur_2d(padded, blur_radius)
    result = blurred.squeeze(1).squeeze(0)
    return result.clamp(min_val, max_val)


def _color_correct(source, destination, method):
    """Apply color correction to source based on destination statistics.

    source: [B, H, W, C] inpainted patch
    destination: [B, H, W, C] original image at same resolution
    method: 'None', 'Match Channels', 'Histogram'
    Returns: color-corrected source
    """
    if method == "None":
        return source

    src = source[..., :3]
    dst = destination[..., :3]

    src_mean = src.mean(dim=[1, 2], keepdim=True)
    src_std = src.std(dim=[1, 2], keepdim=True).clamp_min(1e-6)
    dst_mean = dst.mean(dim=[1, 2], keepdim=True)
    dst_std = dst.std(dim=[1, 2], keepdim=True).clamp_min(1e-6)

    corrected = ((src - src_mean) / src_std) * dst_std + dst_mean

    if source.shape[-1] == 4:
        corrected = torch.cat([corrected, source[..., 3:]], dim=-1)

    return corrected.clamp(0.0, 1.0)


class DaSiWa_InpaintCropPrep:
    """Prepare image and mask for inpainting: extract bounds, crop, scale."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "mask_blur": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 64.0, "step": 0.5}),
                "mask_min": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grow_px": ("INT", {"default": 32, "min": 0, "max": 2048}),
                "can_shrink": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "bbox_x", "bbox_y", "bbox_w", "bbox_h")
    FUNCTION = "execute"
    CATEGORY = "DaSiWa/Inpaint"

    def execute(self, image, mask, target_width, target_height, mask_blur, mask_min, mask_max, grow_px, can_shrink):
        processed_mask = _apply_mask_blur_and_threshold(mask, mask_blur, mask_min, mask_max)

        bx, by, bw, bh = _extract_bounds(processed_mask, grow_px)

        cropped_img = image[:, by:by + bh, bx:bx + bw, :].clone()
        cropped_mask = processed_mask[by:by + bh, bx:bx + bw]

        tw, th = int(target_width), int(target_height)
        if not can_shrink:
            tw = max(tw, bw)
            th = max(th, bh)

        nchw = cropped_img.movedim(-1, 1)
        scaled = F.interpolate(nchw, size=(th, tw), mode="bicubic", align_corners=False, antialias=True)
        scaled = scaled.movedim(1, -1).clamp(0.0, 1.0)

        mask_nchw = cropped_mask.unsqueeze(0).unsqueeze(0)
        scaled_mask = F.interpolate(mask_nchw, size=(th, tw), mode="bilinear", align_corners=False)
        scaled_mask = scaled_mask.squeeze(0).squeeze(0).clamp(0.0, 1.0)

        return (scaled, scaled_mask, bx, by, bw, bh)


class DaSiWa_InpaintComposite:
    """Composite inpainted patch back onto original image with color correction."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "mask": ("MASK",),
                "x": ("INT", {"default": 0}),
                "y": ("INT", {"default": 0}),
                "w": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "h": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "correction_method": (["None", "Match Channels", "Histogram"], {"default": "Match Channels"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "DaSiWa/Inpaint"

    def execute(self, destination, source, mask, x, y, w, h, correction_method):
        bx, by = int(x), int(y)
        crop_w, crop_h = int(w), int(h)

        if source.shape[0] != destination.shape[0]:
            source = source[:destination.shape[0]]

        sh, sw = source.shape[1], source.shape[2]
        if sh != crop_h or sw != crop_w:
            nchw = source.movedim(-1, 1)
            scaled = F.interpolate(nchw, size=(crop_h, crop_w), mode="bicubic", align_corners=False, antialias=True)
            source = scaled.movedim(1, -1).clamp(0.0, 1.0)

        # Scale mask back to original crop size too
        mh, mw = mask.shape[0], mask.shape[1]
        if mh != crop_h or mw != crop_w:
            mask_nchw = mask.unsqueeze(0).unsqueeze(0)
            scaled_mask = F.interpolate(mask_nchw, size=(crop_h, crop_w), mode="bilinear", align_corners=False)
            mask = scaled_mask.squeeze(0).squeeze(0).clamp(0.0, 1.0)

        if correction_method != "None":
            region = destination[:, by:by + crop_h, bx:bx + crop_w, :]
            source = _color_correct(source, region, correction_method)

        result = destination.clone()
        m = mask.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, destination.shape[-1])

        result[:, by:by + crop_h, bx:bx + crop_w, :] = (
            m * source + (1 - m) * result[:, by:by + crop_h, bx:bx + crop_w, :]
        )

        return (result.clamp(0.0, 1.0),)
