"""
IllustriousKSamplerPro — production-ready sampler tuned for Illustrious/SDXL.
Delegates the denoising loop to ComfyUI's common_ksampler so previews & interrupts work.
"""

from __future__ import annotations
import torch, numpy as np, inspect
import comfy.samplers, comfy.sample, comfy.utils, comfy.model_management as model_management
from nodes import common_ksampler


class IllustriousKSamplerPro:
    """
    Production-ready KSampler tuned for Illustrious models.
    Defaults are color-safe and contrast-true. Users normally only tweak seed/steps.
    """

    SAMPLER_ALIASES = {
        "euler_a": "euler_ancestral",
        "euler_ancestral": "euler_ancestral",
        "dpm_2s_a": "dpm_2s_ancestral",
        "dpm_2m": "dpm_2_ancestral",
        "dpm_2_ancestral": "dpm_2_ancestral",
        "dpm_2s_ancestral": "dpm_2s_ancestral",
        "dpm_fast": "dpm_fast",
        "dpm_adaptive": "dpm_adaptive",
    }

    @classmethod
    def INPUT_TYPES(cls):
        base = {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
            },
            "optional": {
                "steps": ("INT", {"default": 26, "min": 4, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.2, "min": 3.0, "max": 12.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Model Version": ([
                    "Auto Detect","Illustrious v0.1","Illustrious v1.0","Illustrious v1.1",
                    "Illustrious v2.0","Illustrious v3.0 EPS","Illustrious v3.0 VPred","Illustrious v3.5 VPred"
                ], {"default": "Auto Detect"}),
                "Resolution Adaptive": ("BOOLEAN", {"default": True}),
                "Auto CFG": ("BOOLEAN", {"default": True}),
                "Color Safe Mode": ("BOOLEAN", {"default": True}),
                "Seed Offset": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "Advanced Logs": ("BOOLEAN", {"default": False}),
                # Accept direct SIGMAS from scheduler nodes.
                "sigmas": ("SIGMAS", {"tooltip": "Direct schedule tensor (SIGMAS) to override scheduler logic."}),
                # Seam control (off by default for crisp results)
                "Edge Feather (px)": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1, "tooltip": "Feather mask edges in latent px before reblend; 0 = off."}),
                "Edge Tone Match": ("BOOLEAN", {"default": False, "tooltip": "Match tone near mask edge before reblend (reduces seams, can soften)."}),
                "Tone Strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Blend weight of tone match when enabled."}),
            },
        }
        return base

    RETURN_TYPES = ("LATENT", "MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("LATENT", "model", "positive", "negative")
    FUNCTION = "sample"
    CATEGORY = "Easy Illustrious / KSampler"

    # ---- helpers ----
    def _normalize_sampler(self, name: str) -> str:
        try: key = str(name or "").lower()
        except Exception: key = ""
        return self.SAMPLER_ALIASES.get(key, key or "euler_ancestral")

    def _preferred_samplers_for_version(self, v):
        v = (v or "").lower()
        if "v3.5" in v or ("v3.0" in v and "vpred" in v):
            return ["dpmpp_2m_sde","dpmpp_sde","dpmpp_2m","euler_ancestral","dpm_2_ancestral","dpm_fast"]
        if "v3.0" in v and "eps" in v:
            return ["dpmpp_2m","euler_ancestral","dpm_2_ancestral","dpm_fast"]
        if "v2.0" in v:
            return ["euler_ancestral","dpm_2_ancestral","dpm_fast"]
        return ["euler_ancestral","dpm_2_ancestral","dpm_fast"]

    def _preferred_schedulers_for_version(self, v):
        v = (v or "").lower()
        if "v3.5" in v or ("v3.0" in v and "vpred" in v): return ["sgm_uniform","karras","exponential","normal"]
        if "v3.0" in v and "eps" in v: return ["karras","exponential","normal"]
        if "v2.0" in v: return ["karras","exponential","normal"]
        return ["karras","normal"]

    def _auto_steps(self, steps,h,w):
        total = h*w
        if total>=2048*2048: return max(28,int(steps*1.10))
        if total>=1536*1536: return max(26,int(steps*1.05))
        if total<=768*768: return max(18,int(steps*0.95))
        return steps

    def _auto_cfg(self,cfg,h,w,version):
        band_min, band_max = 4.6,6.0
        if "v0.1" in version: band_min, band_max=5.0,6.2
        elif "v1.1" in version or "v2.0" in version: band_min, band_max=4.6,5.8
        elif "v3.0" in version or "v3.5" in version: band_min, band_max=4.4,5.6
        total=h*w
        if total>=1536*1536: band_min, band_max=max(4.5,band_min),min(6.0,band_max)
        return float(np.clip(cfg,band_min,band_max))

    def _detect_version(self,h,w):
        longer=max(h,w)
        if longer>=3072: return "Illustrious v3.0 VPred"
        if longer>=2048: return "Illustrious v2.0"
        if longer>=1536: return "Illustrious v1.1"
        if longer>=1024: return "Illustrious v1.0"
        return "Illustrious v0.1"

    def _ksampler_params(self):
        if not hasattr(self,"_ks_params"):
            try: self._ks_params=set(inspect.signature(common_ksampler).parameters.keys())
            except Exception: self._ks_params=set()
        return self._ks_params

    def _call_core_sampler(self,model,seed,steps,cfg,sampler_name,scheduler,
                           positive,negative,latent_dict,denoise=1.0,noise_mask=None,disable_noise=None,
                           sigmas=None):
        args=(model,int(seed),int(steps),float(cfg),sampler_name,scheduler,positive,negative,latent_dict)
        ks_params=self._ksampler_params(); kwargs={"denoise":float(denoise)}
        if noise_mask is not None:
            # Prefer core inpaint 'mask' if available for per-step protection
            # Comfy's core 'mask' expects 1.0 = protect/keep. Our semantics: white = change.
            # Invert for the core call to match Comfy expectations.
            if "mask" in ks_params:
                core_mask = noise_mask
                try:
                    core_mask = 1.0 - torch.clamp(core_mask, 0.0, 1.0)
                except Exception:
                    pass
                kwargs["mask"] = core_mask
            elif "denoise_mask" in ks_params:
                kwargs["denoise_mask"] = noise_mask
            elif "noise_mask" in ks_params:
                kwargs["noise_mask"] = noise_mask
        if disable_noise is not None and "disable_noise" in ks_params: kwargs["disable_noise"]=bool(disable_noise)
        # Wire through sigmas if supported by this ComfyUI build
        if sigmas is not None and "sigmas" in ks_params:
            try:
                if hasattr(sigmas, "detach"):
                    sigmas = sigmas.detach().float().cpu()
            except Exception:
                pass
            kwargs["sigmas"] = sigmas
        out=common_ksampler(*args,**kwargs)
        if isinstance(out,dict) and "samples" in out: return {"samples":out["samples"]}
        if isinstance(out,torch.Tensor): return {"samples":out}
        if isinstance(out,(list,tuple)):
            for it in out:
                if isinstance(it,torch.Tensor): return {"samples":it}
                if isinstance(it,dict) and "samples" in it and isinstance(it["samples"],torch.Tensor):
                    return {"samples":it["samples"]}
        maybe=getattr(out,"samples",None)
        if isinstance(maybe,torch.Tensor): return {"samples":maybe}
        raise TypeError(f"Unsupported sampler return type: {type(out)}")

    # ---- main ----
    def _soften_mask(self, m: torch.Tensor, k: int = 9) -> torch.Tensor:
        """Box-blur the single-channel latent mask [B,1,H,W] to soften edges."""
        try:
            if m is None or not isinstance(m, torch.Tensor):
                return m
            k = max(1, int(k) | 1)
            pad = k // 2
            weight = torch.ones((1, 1, k, k), device=m.device, dtype=m.dtype) / float(k * k)
            mb = torch.nn.functional.conv2d(m, weight, padding=pad)
            return torch.clamp(mb, 0.0, 1.0)
        except Exception:
            return m

    def _smoothstep(self, edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
        t = torch.clamp((x - edge0) / max(1e-6, (edge1 - edge0)), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _weighted_stats(self, x: torch.Tensor, w: torch.Tensor):
        # x: [B,C,H,W], w: [B,1,H,W]
        ws = torch.clamp(w, 0.0, 1.0)
        ws_sum = ws.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        mean = (x * ws).sum(dim=(2, 3), keepdim=True) / ws_sum
        var = ((x - mean) ** 2 * ws).sum(dim=(2, 3), keepdim=True) / ws_sum
        std = torch.sqrt(var + 1e-6)
        return mean, std

    def _match_edge_tone(self, s: torch.Tensor, x0: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # Build an edge band weight around the mask transition
        # band ~ in (0.1..0.9) with a soft profile
        inner = self._smoothstep(0.05, 0.35, m)
        outer = 1.0 - self._smoothstep(0.65, 0.95, m)
        band = (inner * outer).clamp(0.0, 1.0)  # [B,1,H,W]
        if band.max() <= 0:
            return s
        # Compute weighted stats in band for both s and x0
        mean_s, std_s = self._weighted_stats(s, band)
        mean_x0, std_x0 = self._weighted_stats(x0, band)
        # Tone match inside masked region only; clamp ratio
        ratio = (std_x0 / std_s).clamp(0.5, 2.0)
        s_adj = (s - mean_s) * ratio + mean_x0
        # Blend adjustment stronger near edge, lighter deep inside
        edge_strength = band  # already peaked at edge
        s = s * (1.0 - edge_strength) + s_adj * edge_strength
        return s

    def sample(self, model, seed, positive, negative, latent_image,
               steps=26, cfg=5.2, sampler_name="euler_ancestral", scheduler="karras",
               denoise=1.0, sigmas=None, **kw):

        mv = kw.get("Model Version", "Auto Detect")
        res_adapt = kw.get("Resolution Adaptive", True)
        auto_cfg = kw.get("Auto CFG", True)
        color_safe = kw.get("Color Safe Mode", True)
        seed_offset = kw.get("Seed Offset", 0)
        adv_logs = kw.get("Advanced Logs", False)
        edge_feather_px = kw.get("Edge Feather (px)", 0)
        do_tone_match = kw.get("Edge Tone Match", False)
        tone_strength = float(kw.get("Tone Strength", 0.5))

        device = model_management.get_torch_device()
        # Preserve full latent dict (samples, noise_mask, etc.) and move tensors to device
        latent_in = dict(latent_image) if isinstance(latent_image, dict) else {"samples": latent_image}
        latent = latent_in.get("samples")
        if not isinstance(latent, torch.Tensor):
            raise TypeError("latent_image['samples'] must be a torch.Tensor")
        latent = latent.to(device=device)
        # Ensure noise_mask (if present) is on device and clamped
        nm = latent_in.get("noise_mask", None)
        if isinstance(nm, torch.Tensor):
            nm = nm.to(device=device)
            try:
                nm = nm.clamp(0.0, 1.0)
            except Exception:
                pass
            latent_in["noise_mask"] = nm
        # Update samples tensor in the latent dict after moving to device
        latent_in["samples"] = latent
        h, w = latent.shape[2] * 8, latent.shape[3] * 8

        sampler_name = self._normalize_sampler(sampler_name)
        if scheduler == "normal":
            scheduler = "karras"
        if mv == "Auto Detect":
            mv = self._detect_version(h, w)

        if not color_safe:
            if res_adapt:
                steps = self._auto_steps(steps, h, w)
            if auto_cfg:
                cfg = self._auto_cfg(cfg, h, w, mv)
            try:
                sampler_name = next(
                    (s for s in self._preferred_samplers_for_version(mv) if s in comfy.samplers.KSampler.SAMPLERS),
                    sampler_name,
                )
            except Exception:
                pass
            try:
                scheduler = next(
                    (s for s in self._preferred_schedulers_for_version(mv) if s in comfy.samplers.KSampler.SCHEDULERS),
                    scheduler,
                )
            except Exception:
                pass

        final_seed = (int(seed) + int(seed_offset)) & 0xFFFFFFFFFFFFFFFF
        if adv_logs:
            print(
                f"[IllustriousKSamplerPro] v={mv} res={w}x{h} steps={steps} cfg={cfg:.2f} "
                f"sampler={sampler_name}/{scheduler} denoise={denoise} seed={final_seed}"
            )

        # If we will pass mask via kwargs, drop any mask keys from latent dict to avoid duplication
        if isinstance(nm, torch.Tensor):
            try:
                latent_in.pop("noise_mask", None)
                latent_in.pop("denoise_mask", None)
            except Exception:
                pass
        # Choose schedule if provided
        effective_sigmas = sigmas
        if effective_sigmas is not None:
            sampled = self._call_core_sampler(
                model,
                final_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_in,
                denoise,
                noise_mask=nm,
                disable_noise=None,
                sigmas=effective_sigmas,
            )
        else:
            sampled = self._call_core_sampler(
                model,
                final_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_in,
                denoise,
                noise_mask=nm,
            )

        # Strongly protect unmasked area: optional color-match near edges, then re-blend original latent back into (1 - mask)
        try:
            if isinstance(nm, torch.Tensor):
                s = sampled["samples"] if isinstance(sampled, dict) else getattr(sampled, "samples", None)
                if s is not None and isinstance(s, torch.Tensor):
                    m = nm.to(device=s.device, dtype=s.dtype)
                    # Optional mask feather in latent space
                    if isinstance(edge_feather_px, (int, float)) and edge_feather_px and edge_feather_px > 0:
                        m = self._soften_mask(m, k=int(edge_feather_px))
                    if m.shape[1] != 1:
                        m = m[:, :1, :, :]
                    keep = 1.0 - m  # unmasked area
                    # orig latent is in latent_in["samples"]
                    x0 = latent_in.get("samples", None)
                    if isinstance(x0, torch.Tensor):
                        x0 = x0.to(device=s.device, dtype=s.dtype)
                        # Optional edge color/tone harmonization inside masked area before final reblend
                        if do_tone_match and tone_strength > 0.0:
                            s_matched = self._match_edge_tone(s, x0, m)
                            s = s * (1.0 - tone_strength) + s_matched * tone_strength
                        s = s * m + x0 * keep
                        if isinstance(sampled, dict):
                            sampled["samples"] = s
                        else:
                            # if object-like, try attribute
                            try:
                                setattr(sampled, "samples", s)
                            except Exception:
                                sampled = {"samples": s}
        except Exception:
            pass
        return (sampled, model, positive, negative)



# ------------------------------
# IllustriousKSamplerPresets
# ------------------------------

class IllustriousKSamplerPresets:
    """
    One-click presets for IllustriousKSamplerPro.
    Curated params, no hidden overrides. Users typically only touch seed/prompt.
    """

    PRESETS = {
        # Balanced pop + stability (great default)
        "Balanced": {
            "steps": 26,
            "cfg": 5.2,
            "sampler": "euler_ancestral",
            "scheduler": "karras",
            "denoise": 1.0,
        },
        # Extra refinement for close-ups / faces
        "High Quality": {
            "steps": 32,
            "cfg": 4.9,
            "sampler": "dpm_2s_ancestral",
            "scheduler": "karras",
            "denoise": 1.0,
        },
        # Quicker iteration while staying crisp
        "Fast": {
            "steps": 18,
            "cfg": 5.4,
            "sampler": "euler_ancestral",
            "scheduler": "karras",
            "denoise": 1.0,
        },
        # Slightly more guidance and steps for portrait punch
        "Portrait": {
            "steps": 28,
            "cfg": 5.3,
            "sampler": "dpm_2_ancestral",
            "scheduler": "karras",
            "denoise": 1.0,
        },
        # Softer guidance & Karras for scenic gradients
        "Landscape": {
            "steps": 26,
            "cfg": 4.7,
            "sampler": "euler_ancestral",
            "scheduler": "karras",
            "denoise": 1.0,
        },
        # A tad more guidance for saturated anime palettes
        "Anime": {
            "steps": 24,
            "cfg": 5.6,
            "sampler": "euler_ancestral",
            "scheduler": "karras",
            "denoise": 1.0,
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "UNet diffusion model to sample with (SDXL / Illustrious)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Base random seed; use different seeds for different outputs."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning (what you want to see)."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning (what to avoid)."}),
                "latent_image": ("LATENT", {"tooltip": "Starting latent tensor (shape [B,4,H/8,W/8])."}),
                "Preset": (list(cls.PRESETS.keys()), {"default": "Balanced", "tooltip": "Curated configuration presets for Illustrious XL."}),
            },
            "optional": {
                # Power users can flip these; defaults are production-safe and transparent.
                "Model Version": (
                    [
                        "Auto Detect",
                        "Illustrious v0.1",
                        "Illustrious v1.0",
                        "Illustrious v1.1",
                        "Illustrious v2.0",
                        "Illustrious v3.0 EPS",
                        "Illustrious v3.0 VPred",
                        "Illustrious v3.5 VPred",
                    ],
                    {"default": "Auto Detect", "tooltip": "Pick your Illustrious version or let the node auto-detect by resolution."},
                ),
                "Color Safe Mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When ON, use preset values exactly with minimal auto-tweaks.",
                    },
                ),
                "Resolution Adaptive": ("BOOLEAN", {"default": True, "tooltip": "Gently adjusts steps for larger/smaller images."}),
                "Auto CFG": ("BOOLEAN", {"default": True, "tooltip": "Keeps CFG within a sweet spot for the selected version."}),
                "Seed Offset": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Added to seed to vary results while keeping the base seed."}),
                "Advanced Logs": ("BOOLEAN", {"default": False, "tooltip": "Print extra diagnostics to the console."}),
            },
        }

    # Match Pro node outputs without the legacy 'input_latent' passthrough.
    RETURN_TYPES = ("LATENT", "MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("LATENT", "model", "positive", "negative")
    FUNCTION = "sample_with_preset"
    CATEGORY = "Easy Illustrious / KSampler"

    def sample_with_preset(self, model, seed, positive, negative, latent_image, Preset, **kw):
        # Pull preset; copy to avoid accidental mutation
        if Preset not in self.PRESETS:
            raise ValueError(f"Unknown preset '{Preset}'. Available: {', '.join(self.PRESETS.keys())}")
        p = dict(self.PRESETS[Preset])

        # Wire optional controls straight through (names match Pro node)
        passthrough = {
            "Model Version": kw.get("Model Version", "Auto Detect"),
            "Color Safe Mode": kw.get("Color Safe Mode", True),
            "Resolution Adaptive": kw.get("Resolution Adaptive", True),
            "Auto CFG": kw.get("Auto CFG", True),
            "Seed Offset": kw.get("Seed Offset", 0),
            "Advanced Logs": kw.get("Advanced Logs", False),
        }

        # Delegate to Pro node (which delegates to common_ksampler → native preview/interrupt)
        pro = IllustriousKSamplerPro()
        sampled, out_model, out_pos, out_neg = pro.sample(
            model=model,
            seed=seed,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            steps=p["steps"],
            cfg=p["cfg"],
            sampler_name=p["sampler"],
            scheduler=p["scheduler"],
            denoise=p["denoise"],
            **passthrough,
        )
        return sampled, out_model, out_pos, out_neg
