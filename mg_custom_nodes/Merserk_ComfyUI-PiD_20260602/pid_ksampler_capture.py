from __future__ import annotations

from typing import Optional

import torch

try:
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview
except Exception:  # pragma: no cover - only available inside ComfyUI
    comfy = None
    latent_preview = None


def _sampler_names():
    if comfy is not None:
        try:
            return comfy.samplers.KSampler.SAMPLERS
        except Exception:
            pass
    return ["euler"]


def _scheduler_names():
    if comfy is not None:
        try:
            return comfy.samplers.KSampler.SCHEDULERS
        except Exception:
            pass
    return ["beta"]


def _copy_latent(latent: dict, samples: torch.Tensor, sigma: Optional[float] = None) -> dict:
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    if sigma is not None:
        out["pid_sigma"] = float(sigma)
    return out


class PiDKSamplerCapture:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (_sampler_names(), {"default": "euler"}),
                "scheduler": (_scheduler_names(), {"default": "beta"}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "capture_step": ("INT", {"default": 46, "min": 1, "max": 10000, "step": 1}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "FLOAT")
    RETURN_NAMES = ("final_latent", "pid_latent", "pid_sigma")
    FUNCTION = "sample"
    CATEGORY = "PiD/Staged"

    def sample(
        self,
        model,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive,
        negative,
        latent_image,
        denoise: float = 1.0,
        capture_step: int = 46,
    ):
        if comfy is None:
            raise RuntimeError("PiD KSampler Capture must run inside ComfyUI.")

        latent_samples = latent_image["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(
            model,
            latent_samples,
            latent_image.get("downscale_ratio_spacial", None),
        )

        batch_inds = latent_image.get("batch_index")
        noise = comfy.sample.prepare_noise(latent_samples, int(seed), batch_inds)
        noise_mask = latent_image.get("noise_mask")

        sampler = comfy.samplers.KSampler(
            model,
            steps=int(steps),
            device=model.load_device,
            sampler=str(sampler_name),
            scheduler=str(scheduler),
            denoise=float(denoise),
            model_options=model.model_options,
        )
        sigmas = sampler.sigmas
        effective_steps = max(0, int(sigmas.shape[0]) - 1)
        target_step = min(max(1, int(capture_step)), max(1, effective_steps))

        captured = {"samples": None, "sigma": None}
        preview_callback = None
        if latent_preview is not None:
            try:
                preview_callback = latent_preview.prepare_callback(model, effective_steps)
            except Exception:
                preview_callback = None

        def callback(step, x0, x, total_steps):
            if preview_callback is not None:
                preview_callback(step, x0, x, total_steps)
            one_based_step = int(step) + 1
            if one_based_step == target_step:
                captured["samples"] = x.detach().to("cpu").contiguous()
                try:
                    captured["sigma"] = float(sigmas[int(step)].detach().float().cpu().item())
                except Exception:
                    captured["sigma"] = 0.0

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(
            model,
            noise,
            int(steps),
            float(cfg),
            str(sampler_name),
            str(scheduler),
            positive,
            negative,
            latent_samples,
            denoise=float(denoise),
            disable_noise=False,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=int(seed),
        )

        if captured["samples"] is None:
            captured["samples"] = samples.detach().to("cpu").contiguous()
            captured["sigma"] = 0.0

        final_latent = _copy_latent(latent_image, samples)
        pid_latent = _copy_latent(latent_image, captured["samples"], captured["sigma"])
        return (final_latent, pid_latent, float(captured["sigma"]))


NODE_CLASS_MAPPINGS = {
    "PiDKSamplerCapture": PiDKSamplerCapture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDKSamplerCapture": "PiD KSampler Capture",
}
