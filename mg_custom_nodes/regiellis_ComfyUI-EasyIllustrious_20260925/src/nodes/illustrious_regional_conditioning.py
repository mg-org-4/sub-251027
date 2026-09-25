from __future__ import annotations

from typing import List
import torch


class IllustriousRegionalConditioning:
    """
    Compose regional (mask-scoped) prompts into a CONDITIONING object.
    - Takes a base/global conditioning and appends region-specific entries with masks at latent resolution.
    - Compatible with samplers that support masked/area conditioning (extras include: mask, timestep_percent_range, weight).
    - Does not touch the latent; only modifies CONDITIONING.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP used to encode regional prompts."}),
                "base_cond": ("CONDITIONING", {"tooltip": "Base/global conditioning to start from."}),
                "latent": ("LATENT", {"tooltip": "Latent (for sizing masks to H/8 x W/8)."}),
                "normalize_overlap": ("BOOLEAN", {"default": True, "tooltip": "Normalize overlapping masks so combined weights per pixel don't exceed 1."}),
                "mask_blur": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1}),
                "mask_dilate": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1}),
            },
            "optional": {
                # Unlimited regions via builder nodes
                "regions": ("ILLUSTRIOUS_REGIONS", {"tooltip": "Optional list of regions built via region builder nodes."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "info")
    FUNCTION = "build"
    CATEGORY = "Easy Illustrious / Conditioning"

    # ---- helpers ----
    def _refine_mask(self, m_bchw, blur_px=0, dilate_px=0):
        m = m_bchw
        if dilate_px and dilate_px > 0:
            k = max(1, int(dilate_px))
            pad = k // 2
            m = torch.nn.functional.max_pool2d(m, kernel_size=k, stride=1, padding=pad)
        if blur_px and blur_px > 0:
            k = max(1, int(blur_px) | 1)
            pad = k // 2
            w = torch.ones((1, 1, k, k), device=m.device, dtype=m.dtype) / float(k * k)
            m = torch.nn.functional.conv2d(m, w, padding=pad)
        return torch.clamp(m, 0.0, 1.0)

    def _mask_to_latent_res(self, mask, latent_hw, device):
        # Convert mask to [B,1,lh,lw]
        m = mask
        if not torch.is_tensor(m):
            m = torch.tensor(m)
        m = m.to(device=device, dtype=torch.float32)

        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        elif m.dim() == 3:
            if m.shape[-1] == 1:
                m = m.permute(2, 0, 1).unsqueeze(0)  # [1,1,H,W]
            else:
                m = m.unsqueeze(0)  # [1,H,W] -> [1,1,H,W] slice first channel
                if m.shape[1] != 1:
                    m = m[:, :1, :, :]
        elif m.dim() == 4:
            if m.shape[-1] == 1:
                m = m.permute(0, 3, 1, 2).contiguous()
            elif m.shape[1] != 1:
                m = m[:, :1, :, :]

        m = torch.nn.functional.interpolate(m, size=latent_hw, mode="bilinear", align_corners=False)
        return torch.clamp(m, 0.0, 1.0)

    # no inline region collector anymore – use builder nodes instead

    # ---- main ----
    def build(self, clip, base_cond, latent,
              normalize_overlap=True, mask_blur=4, mask_dilate=2, regions=None):

        device = latent["samples"].device
        _, _, lh, lw = latent["samples"].shape
        latent_hw = (lh, lw)

        regions_tuples = []
        # Collect regions from builder chain if provided
        if isinstance(regions, dict) and isinstance(regions.get("regions", None), list):
            for r in regions["regions"]:
                mk = r.get("mask")
                pk = str(r.get("prompt", ""))
                if mk is None or len(pk.strip()) == 0:
                    continue
                regions_tuples.append((
                    mk, pk,
                    float(r.get("weight", 1.0)),
                    float(r.get("start", 0.0)),
                    float(r.get("end", 1.0)),
                ))

        cond = list(base_cond)  # shallow copy

        # Refine and collect masks
        refined_masks: List[torch.Tensor] = []
        for (mask, prompt, weight, start, end) in regions_tuples:
            m = self._mask_to_latent_res(mask, latent_hw, device)
            m = self._refine_mask(m, blur_px=mask_blur, dilate_px=mask_dilate)
            refined_masks.append(m)

        # Normalize overlaps if requested
        if normalize_overlap and len(refined_masks) > 0:
            stack = torch.stack(refined_masks, dim=0)  # [R,B,1,lh,lw]
            denom = torch.clamp(stack.sum(dim=0), min=1e-6)  # [B,1,lh,lw]
            refined_masks = [m / denom for m in refined_masks]

        info_lines: List[str] = []
        for idx, ((mask, prompt, weight, start, end), m_bchw) in enumerate(zip(regions_tuples, refined_masks), start=1):
            # Encode prompt using CLIP
            tokens = clip.tokenize(prompt)
            region_cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

            # Use [B,lh,lw] mask (no channel dim) to match Comfy's area-conditioning expectations
            mask_3d = m_bchw.squeeze(1) if m_bchw.dim() == 4 else m_bchw
            extras = {
                "pooled_output": pooled,
                "mask": mask_3d,  # [B,lh,lw]
                "weight": float(weight),
                "timestep_percent_range": (float(start), float(end)),
            }
            cond.append([region_cond, extras])
            info_lines.append(
                f"R{idx}: w={weight:.2f}, t=[{start:.2f},{end:.2f}], prompt={prompt[:64]}{'...' if len(prompt)>64 else ''}"
            )

        info = "Illustrious Regional Conditioning\n" + ("\n".join(info_lines) if info_lines else "No regions added.")
        return (cond, info)


class IllustriousMakeRegion:
    """Create a single region descriptor to be collected later."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Region mask (white = apply)."}),
                "prompt": ("STRING", {"multiline": True}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("ILLUSTRIOUS_REGION",)
    RETURN_NAMES = ("region",)
    FUNCTION = "make"
    CATEGORY = "Easy Illustrious / Conditioning"

    def make(self, mask, prompt, weight=1.0, start=0.0, end=1.0):
        return ({
            "mask": mask,
            "prompt": str(prompt),
            "weight": float(weight),
            "start": float(start),
            "end": float(end),
        },)


class IllustriousEmptyRegions:
    """Create an empty regions container."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("ILLUSTRIOUS_REGIONS",)
    RETURN_NAMES = ("regions",)
    FUNCTION = "empty"
    CATEGORY = "Easy Illustrious / Conditioning"

    def empty(self):
        return ({"regions": []},)


class IllustriousAppendRegion:
    """Append a region to a regions list (chain this to collect unlimited regions)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "regions": ("ILLUSTRIOUS_REGIONS", {}),
                "region": ("ILLUSTRIOUS_REGION", {}),
            }
        }

    RETURN_TYPES = ("ILLUSTRIOUS_REGIONS",)
    RETURN_NAMES = ("regions",)
    FUNCTION = "append"
    CATEGORY = "Easy Illustrious / Conditioning"

    def append(self, regions, region):
        out = {"regions": list(regions.get("regions", []))}
        out["regions"].append(region)
        return (out,)
