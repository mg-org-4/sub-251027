# FL_KsamplerSEG: per-region segmented diffusion sampler.
#
# Cascade-paint compositing (single mode):
#   - Regions are painted onto a working canvas in a corner-anchored
#     nearest-neighbor traversal (each new region paints adjacent to
#     already-painted territory).
#   - Each region samples from the IN-PROGRESS canvas, so the model sees
#     previously-painted regions as part of its context. The model itself
#     does the seam-blending in feature space rather than us doing it
#     mechanically in pixel space.
#   - Per-region results are alpha-painted on top using the soft mask:
#     canvas[bbox] = result * mask + canvas[bbox] * (1 - mask)
#
# The soft masks come from FL_KsamplerSEG_Regions; they overlap and feather
# already, so coverage is guaranteed.

import base64
import io
import logging
import math

import torch
import torch.nn.functional as F
from PIL import Image

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import latent_preview
from server import PromptServer

from .FL_KsamplerSEG_common import unwrap_regions, latent_bbox_from_image_bbox
from ._latent_helpers import primary_only_noise_mask, primary_tensor, replace_primary_tensor


CASCADE_START_CORNERS = ["top_left", "top_right", "bottom_left", "bottom_right", "center"]


class FL_KsamplerSEG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "regions": ("SEG_REGIONS", {"tooltip": "Encoded regions override positive and negative. Use raw Regions to preserve connected reference conditioning."}),
                "latent_image": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cascade_start_corner": (CASCADE_START_CORNERS, {"default": "top_left"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "🏵️Fill Nodes/Ksamplers"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def sample(self, model, regions, latent_image, positive, negative,
               seed, steps, cfg, sampler_name, scheduler, denoise,
               cascade_start_corner, unique_id=None):
        return self._sample_regions(model, regions, latent_image, positive, negative,
                                    seed, steps, cfg, sampler_name, scheduler, denoise,
                                    cascade_start_corner, unique_id)

    def _sample_regions(self, model, regions, latent_image, positive, negative,
                        seed, steps, cfg, sampler_name, scheduler, denoise,
                        cascade_start_corner, unique_id=None, *, disable_noise=False,
                        start_step=None, last_step=None, force_full_denoise=False):
        regions = unwrap_regions(regions)
        window_start = 0 if start_step is None else start_step
        window_end = steps if last_step is None else min(steps, last_step)
        region_steps = max(0, window_end - window_start)
        if region_steps == 0:
            return (latent_image,)

        latent_samples = latent_image["samples"]
        primary_samples = primary_tensor(latent_samples)

        is_zero = bool(torch.count_nonzero(primary_samples) == 0)
        if is_zero and denoise < 0.99:
            raise ValueError(
                "FL_KsamplerSEG: per-region refinement requires denoise=1.0 with "
                "an empty latent, or a real latent input with denoise<1.0. "
                "An empty latent at low denoise produces noise."
            )

        device = model.load_device

        latent_full = comfy.sample.fix_empty_latent_channels(
            model, latent_samples.to(device=device),
            latent_image.get("downscale_ratio_spacial", None),
        )
        H, W = regions["image_size"]
        latent_h, latent_w = primary_tensor(latent_full).shape[-2:]
        downscale = self._resolve_downscale(model, regions, latent_h, latent_w, H, W)

        N = regions["shape_masks"].shape[0]
        per_region_cond = regions.get("conditioning_per_region")
        if per_region_cond is not None and len(per_region_cond) != N:
            logging.warning(
                f"[FL_KsamplerSEG] conditioning_per_region length {len(per_region_cond)} "
                f"!= region count {N}; falling back to widget conditioning."
            )
            per_region_cond = None

        # Spatial-locality ordering: corner-anchored nearest-neighbor.
        # Each new region paints adjacent to already-painted territory so the
        # cascade flows naturally from the chosen corner outward.
        write_areas = regions["write_masks"].sum(dim=(1, 2))
        order = self._cascade_order(
            regions=regions,
            write_areas=write_areas,
            start_corner=cascade_start_corner,
        )

        # Pre-compute per-region (latent_bbox, write_mask_lat, comp_mask_lat,
        # pos_cond, neg_cond, seed). Skips regions too small for stable attention.
        region_specs = []
        for n in order:
            spec = self._build_region_spec(
                regions=regions, region_index=int(n), downscale=downscale,
                latent_h=latent_h, latent_w=latent_w, device=device,
                per_region_cond=per_region_cond,
                cond_pos_default=positive, cond_neg_default=negative,
                base_seed=int(seed),
            )
            if spec is not None:
                region_specs.append(spec)

        if not region_specs:
            logging.warning("[FL_KsamplerSEG] no usable regions; returning input latent unchanged.")
            return (latent_image,)

        # Run the cascade.
        canvas = primary_tensor(latent_full).clone()
        # Pre-compute the broadcast shape for masks. For a 4D latent (B,C,H,W)
        # the mask broadcasts as (1,1,H,W). For a 5D video latent (B,C,T,H,W)
        # it must broadcast as (1,1,1,H,W) -- one extra leading dim per
        # non-spatial axis. PyTorch won't auto-align mismatched-rank tensors.
        latent_ndim = canvas.ndim
        total_steps = region_steps * len(region_specs)
        preview_callback = latent_preview.prepare_callback(model, total_steps)
        for region_step, spec in enumerate(region_specs):
            preview_state = {
                "node": str(unique_id), "region": spec["region_index"],
                "position": region_step + 1, "count": len(region_specs),
                "step": 0, "steps": region_steps, "state": "sampling",
                "start_step": start_step, "end_step": window_end, "schedule_steps": steps,
                "crop": list(spec["latent_bbox"]), "size": [latent_w, latent_h],
                "mask": self._preview_mask(spec["write_lat"]),
            }

            def send_status():
                if unique_id is not None and PromptServer.instance is not None:
                    PromptServer.instance.send_sync("fl_seg_sampling", dict(preview_state), PromptServer.instance.client_id)

            send_status()
            samples = self._sample_one_region_full(
                model=model,
                source_latent=replace_primary_tensor(latent_full, canvas),
                spec=spec,
                steps=steps, cfg=cfg, sampler_name=sampler_name,
                scheduler=scheduler, denoise=denoise,
                latent_ndim=latent_ndim,
                preview_callback=preview_callback, step_offset=region_step * region_steps,
                total_steps=total_steps, preview_state=preview_state, send_status=send_status,
                disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                force_full_denoise=force_full_denoise,
            )
            by0, bx0, by1, bx1 = spec["latent_bbox"]
            comp_b = self._reshape_mask_for_broadcast(
                spec["comp_lat"], latent_ndim,
            ).to(dtype=canvas.dtype)
            samples_dev = samples.to(device=canvas.device, dtype=canvas.dtype)
            existing = canvas[..., by0:by1, bx0:bx1]
            canvas[..., by0:by1, bx0:bx1] = samples_dev * comp_b + existing * (1.0 - comp_b)

        preview_state["state"] = "complete"
        send_status()

        out = replace_primary_tensor(latent_full, canvas).to(
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        result = dict(latent_image)
        result["samples"] = out
        result.pop("noise_mask", None)
        return (result,)

    # ------------------------------------------------------------------
    # Per-region helpers
    # ------------------------------------------------------------------

    def _build_region_spec(self, *, regions, region_index, downscale,
                           latent_h, latent_w, device, per_region_cond,
                           cond_pos_default, cond_neg_default, base_seed):
        """Resolve geometry + masks + conditioning for one region. Returns None
        if the region is too small to sample."""
        bbox = regions["padded_bboxes"][region_index]
        latent_bbox = latent_bbox_from_image_bbox(bbox, downscale, latent_h, latent_w)
        by0, bx0, by1, bx1 = latent_bbox

        if (by1 - by0) < 8 or (bx1 - bx0) < 8:
            logging.warning(
                f"[FL_KsamplerSEG] region {region_index} latent bbox too small "
                f"({by1-by0}x{bx1-bx0}); skipping."
            )
            return None

        write_full = regions["write_masks"][region_index].to(device=device, dtype=torch.float32)
        comp_full = regions["composite_masks"][region_index].to(device=device, dtype=torch.float32)

        iy0 = by0 * downscale
        ix0 = bx0 * downscale
        iy1 = min(by1 * downscale, write_full.shape[0])
        ix1 = min(bx1 * downscale, write_full.shape[1])
        write_img_crop = write_full[iy0:iy1, ix0:ix1]
        comp_img_crop = comp_full[iy0:iy1, ix0:ix1]

        target_h_img = (by1 - by0) * downscale
        target_w_img = (bx1 - bx0) * downscale
        write_img_crop = self._pad_to(write_img_crop, target_h_img, target_w_img)
        comp_img_crop = self._pad_to(comp_img_crop, target_h_img, target_w_img)

        write_lat = F.avg_pool2d(
            write_img_crop.unsqueeze(0).unsqueeze(0), kernel_size=downscale,
        ).squeeze(0).squeeze(0)
        comp_lat = F.avg_pool2d(
            comp_img_crop.unsqueeze(0).unsqueeze(0), kernel_size=downscale,
        ).squeeze(0).squeeze(0)

        if write_lat.sum() < 1e-6:
            return None

        if per_region_cond is not None:
            pos_cond, neg_cond = per_region_cond[region_index]
        else:
            pos_cond, neg_cond = cond_pos_default, cond_neg_default

        return {
            "region_index": region_index,
            "latent_bbox": latent_bbox,
            "write_lat": write_lat,
            "comp_lat": comp_lat,
            "pos_cond": pos_cond,
            "neg_cond": neg_cond,
            "seed": base_seed + region_index,
        }

    def _sample_one_region_full(self, *, model, source_latent, spec,
                                steps, cfg, sampler_name, scheduler, denoise,
                                latent_ndim=None, preview_callback=None, step_offset=0,
                                total_steps=None, preview_state=None, send_status=None,
                                disable_noise=False, start_step=None, last_step=None,
                                force_full_denoise=False):
        """Run a complete sampler call (all steps) for one region, sourcing the
        crop from `source_latent` (the in-progress canvas)."""
        by0, bx0, by1, bx1 = spec["latent_bbox"]
        source_primary = primary_tensor(source_latent)
        primary_crop = source_primary[..., by0:by1, bx0:bx1].contiguous()
        latent_crop = replace_primary_tensor(source_latent, primary_crop)

        noise = (comfy.sample.prepare_empty_noise(latent_crop) if disable_noise
                 else comfy.sample.prepare_noise(latent_crop.cpu(), spec["seed"]))

        # Mask must match the latent rank so it broadcasts. For 4D latent
        # (image): (1,1,H,W). For 5D latent (video): (1,1,1,H,W).
        if latent_ndim is None:
            latent_ndim = source_latent.ndim
        noise_mask = self._reshape_mask_for_broadcast(
            spec["write_lat"], latent_ndim,
        ).to(dtype=primary_crop.dtype)
        noise_mask = primary_only_noise_mask(latent_crop, noise_mask)

        window_start = 0 if start_step is None else start_step
        window_end = steps if last_step is None else min(steps, last_step)
        region_steps = max(0, window_end - window_start)
        if preview_callback is None:
            preview_callback = latent_preview.prepare_callback(model, region_steps)
        if total_steps is None:
            total_steps = region_steps
        preview_canvas = None
        preview_base = None

        def callback(step, x0, x, _total_steps):
            nonlocal preview_canvas, preview_base
            if preview_canvas is None:
                preview_canvas = model.get_model_object("process_latent_in")(source_primary).clone()
                preview_base = preview_canvas[..., by0:by1, bx0:bx1].clone()
            comp = self._reshape_mask_for_broadcast(spec["comp_lat"], preview_canvas.ndim)
            preview_canvas[..., by0:by1, bx0:bx1] = primary_tensor(x0) * comp + preview_base * (1.0 - comp)
            # Some samplers emit one more preview after the final diffusion step.
            completed_steps = min(step + 1, region_steps)
            preview_callback(step_offset + completed_steps - 1, preview_canvas, x, total_steps)
            if preview_state is not None:
                preview_state["step"] = completed_steps
                preview_state.pop("mask", None)
                send_status()

        try:
            samples = comfy.sample.sample(
                model, noise, steps, cfg, sampler_name, scheduler,
                spec["pos_cond"], spec["neg_cond"], latent_crop,
                denoise=denoise, noise_mask=noise_mask,
                disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                force_full_denoise=force_full_denoise,
                callback=callback, disable_pbar=True, seed=spec["seed"],
            )
        except Exception as e:
            if preview_state is not None:
                preview_state["state"] = "stopped"
                send_status()
            logging.error(
                f"[FL_KsamplerSEG] region {spec['region_index']} sample failed: {e}"
            )
            raise

        return primary_tensor(samples)

    @staticmethod
    def _preview_mask(mask):
        mask = mask.detach().float().cpu()
        image = Image.fromarray((mask.clamp(0, 1).numpy() * 255).astype("uint8"))
        image.thumbnail((160, 160))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _cascade_order(*, regions, write_areas, start_corner):
        """Corner-anchored nearest-neighbor traversal of regions for cascade_paint.

        Computes each region's centroid (mass-weighted) from its write_mask,
        picks the region nearest the chosen corner as the starting point, then
        greedily appends the unprocessed region whose centroid is closest to
        any already-processed region's centroid. Size is used as a tiebreaker
        (larger regions preferred when distances are within 1% of each other).

        Returns a list of region indices in painting order.
        """
        write_masks = regions["write_masks"]  # (N, H, W) on CPU
        N, H, W = write_masks.shape

        ys = torch.arange(H, dtype=torch.float32).view(1, H, 1)
        xs = torch.arange(W, dtype=torch.float32).view(1, 1, W)
        masses = write_masks.sum(dim=(1, 2)).clamp(min=1e-6)
        cy = (write_masks * ys).sum(dim=(1, 2)) / masses
        cx = (write_masks * xs).sum(dim=(1, 2)) / masses
        centroids = torch.stack([cx, cy], dim=1)  # (N, 2) in (x, y) order

        if start_corner == "top_left":
            anchor = torch.tensor([0.0, 0.0])
        elif start_corner == "top_right":
            anchor = torch.tensor([float(W - 1), 0.0])
        elif start_corner == "bottom_left":
            anchor = torch.tensor([0.0, float(H - 1)])
        elif start_corner == "bottom_right":
            anchor = torch.tensor([float(W - 1), float(H - 1)])
        else:  # center
            anchor = torch.tensor([(W - 1) / 2.0, (H - 1) / 2.0])

        d_to_anchor = ((centroids - anchor) ** 2).sum(dim=1)
        first = int(d_to_anchor.argmin().item())

        order = [first]
        unvisited = [i for i in range(N) if i != first]
        sizes = write_areas.tolist() if isinstance(write_areas, torch.Tensor) else list(write_areas)

        c_np = centroids.numpy()
        nearest_d = {}
        for i in unvisited:
            dx = c_np[i, 0] - c_np[first, 0]
            dy = c_np[i, 1] - c_np[first, 1]
            nearest_d[i] = dx * dx + dy * dy

        while unvisited:
            min_d = min(nearest_d[i] for i in unvisited)
            tie_threshold = min_d * 1.01 + 1e-6
            candidates = [i for i in unvisited if nearest_d[i] <= tie_threshold]
            if len(candidates) == 1:
                pick = candidates[0]
            else:
                pick = max(candidates, key=lambda i: sizes[i])

            order.append(pick)
            unvisited.remove(pick)

            for i in unvisited:
                dx = c_np[i, 0] - c_np[pick, 0]
                dy = c_np[i, 1] - c_np[pick, 1]
                d = dx * dx + dy * dy
                if d < nearest_d[i]:
                    nearest_d[i] = d

        return order

    @staticmethod
    def _resolve_downscale(model, regions, latent_h, latent_w, image_h, image_w):
        downscale = int(model.get_model_object("latent_format").spacial_downscale_ratio)
        if not (image_h // downscale <= latent_h <= math.ceil(image_h / downscale)
                and image_w // downscale <= latent_w <= math.ceil(image_w / downscale)):
            raise ValueError(
                f"SEG Regions describe {image_w}x{image_h}, but the latent represents "
                f"{latent_w * downscale}x{latent_h * downscale}. "
                "Connect Regions and VAE Encode to the same resized image."
            )
        return downscale

    @staticmethod
    def _reshape_mask_for_broadcast(mask_2d, target_ndim):
        """Take a 2D (h, w) mask and add leading 1-dims to match target_ndim.

        For target_ndim=4 (image B,C,H,W): returns (1, 1, h, w).
        For target_ndim=5 (video B,C,T,H,W): returns (1, 1, 1, h, w).
        """
        if mask_2d.ndim != 2:
            # Defensive: if already higher-dim, leave it.
            return mask_2d
        out = mask_2d
        leading = max(0, target_ndim - 2)
        for _ in range(leading):
            out = out.unsqueeze(0)
        return out

    @staticmethod
    def _pad_to(t, h, w):
        ch = t.shape[0]
        cw = t.shape[1]
        if ch == h and cw == w:
            return t
        pad_h = max(0, h - ch)
        pad_w = max(0, w - cw)
        return F.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=0.0)


class FL_KsamplerSEGAdvanced(FL_KsamplerSEG):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        required = inputs["required"]
        seed = required.pop("seed")
        required.pop("denoise")
        corner = required.pop("cascade_start_corner")
        inputs["required"] = {
            **{name: required[name] for name in ("model", "regions", "latent_image", "positive", "negative")},
            "add_noise": (["enable", "disable"], {"default": "enable", "tooltip": "Enable for a source latent. Disable when continuing a latent that already contains noise."}),
            "noise_seed": (seed[0], {**seed[1], "control_after_generate": True}),
            **{name: required[name] for name in ("steps", "cfg", "sampler_name", "scheduler")},
            "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Start index in the full schedule, applied to every region. Zero starts at the beginning."}),
            "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000, "tooltip": "Stop index in the full schedule. Values above steps run to the end. An empty step window returns the input unchanged."}),
            "return_with_leftover_noise": (["disable", "enable"], {"default": "disable", "tooltip": "Disable to finish at zero noise even when stopping early. Enable to retain noise for a following sampler."}),
            "cascade_start_corner": corner,
        }
        return inputs

    DESCRIPTION = "Sample each SEG region over an explicit window of the full diffusion schedule. Uses KSampler Advanced noise and end-step behavior. Overlapping regions are composited in cascade order, so splitting across nodes is not identical to one uninterrupted pass."

    def sample(self, model, regions, latent_image, positive, negative, add_noise,
               noise_seed, steps, cfg, sampler_name, scheduler, start_at_step,
               end_at_step, return_with_leftover_noise, cascade_start_corner, unique_id=None):
        return self._sample_regions(
            model, regions, latent_image, positive, negative, noise_seed, steps, cfg,
            sampler_name, scheduler, 1.0, cascade_start_corner, unique_id,
            disable_noise=add_noise == "disable", start_step=start_at_step,
            last_step=end_at_step, force_full_denoise=return_with_leftover_noise == "disable")
