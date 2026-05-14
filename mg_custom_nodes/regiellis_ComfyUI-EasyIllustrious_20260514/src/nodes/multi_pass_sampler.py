import torch
import comfy.samplers
import comfy.utils
import comfy.model_management
import numpy as np
import inspect
from nodes import common_ksampler

# ---------- Common helpers ----------


def _safe_clamp(t, clamp_val):
    # Only clamp when values look off the rails
    # Typical latents rarely exceed |5|.
    needs_clamp = (t.abs() > clamp_val).any()
    if needs_clamp:
        t = torch.clamp(
            torch.nan_to_num(t, nan=0.0, posinf=clamp_val, neginf=-clamp_val),
            -clamp_val,
            clamp_val,
        )
    else:
        # Still scrub NaNs/Infs if any slipped in
        t = torch.nan_to_num(t, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
    return t


def _latent_std(x: torch.Tensor) -> float:
    # Per-sample mean then average to reduce batch spikes
    with torch.no_grad():
        s = x.float().std(dim=[1, 2, 3]).mean().item()
    return float(s)


def _lerp(a, b, w):
    return a * (1.0 - w) + b * w


def _soften_mask(m: torch.Tensor, k: int = 11) -> torch.Tensor:
    try:
        if m is None or not isinstance(m, torch.Tensor):
            return m
        k = max(1, int(k) | 1)
        pad = k // 2
        w = torch.ones((1, 1, k, k), device=m.device, dtype=m.dtype) / float(k * k)
        mb = torch.nn.functional.conv2d(m, w, padding=pad)
        return torch.clamp(mb, 0.0, 1.0)
    except Exception:
        return m


# ---------- Version-aware presets (EPS vs VPred) ----------

def _preferred_samplers_for_mode(mode: str):
    m = (mode or "").lower()
    if m == "vpred":
        return [
            "dpmpp_2m_sde",
            "dpmpp_sde",
            "dpmpp_2m",
            "euler_ancestral",
            "dpm_2_ancestral",
            "dpm_fast",
        ]
    # EPS or fallback
    return ["dpmpp_2m", "euler_ancestral", "dpm_2_ancestral", "dpm_fast"]


def _preferred_schedulers_for_mode(mode: str):
    m = (mode or "").lower()
    if m == "vpred":
        return ["sgm_uniform", "karras", "exponential", "normal"]
    return ["karras", "exponential", "normal"]


def _pick_supported(name_list, available):
    for n in name_list:
        if n in available:
            return n
    return next(iter(available), None)


def _maybe_apply_version_presets(current_sampler: str, current_scheduler: str, desired_mode: str, default_sampler: str, default_scheduler: str):
    """Only override when the current values equal the node defaults to avoid clobbering user choices."""
    try:
        samplers_avail = set(getattr(comfy.samplers.KSampler, "SAMPLERS", []))
        sched_avail = set(getattr(comfy.samplers.KSampler, "SCHEDULERS", []))
    except Exception:
        samplers_avail, sched_avail = set(), set()

    out_sampler, out_sched = current_sampler, current_scheduler
    if desired_mode and current_sampler == default_sampler:
        prefs = _preferred_samplers_for_mode(desired_mode)
        pick = _pick_supported(prefs, samplers_avail)
        if pick:
            out_sampler = pick
    if desired_mode and current_scheduler == default_scheduler:
        prefs_s = _preferred_schedulers_for_mode(desired_mode)
        pick_s = _pick_supported(prefs_s, sched_avail)
        if pick_s:
            out_sched = pick_s
    return out_sampler, out_sched


# ---------- MultiPass (two stages) ----------


class IllustriousMultiPassSampler:
    """Multi-pass sampler optimized for Illustrious XL models"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "UNet diffusion model to sample with (SDXL / Illustrious)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Base random seed; use different seeds for different outputs."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning (what you want to see)."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning (what to avoid)."}),
                "latent_image": ("LATENT", {"tooltip": "Starting latent tensor (shape [B,4,H/8,W/8])."}),
            },
            "optional": {
                "structure_steps": ("INT", {"default": 12, "min": 5, "max": 50, "tooltip": "Steps for the structure pass (coarse layout)."}),
                "structure_cfg": ("FLOAT", {"default": 6.5, "min": 3.0, "max": 8.0, "step": 0.1, "tooltip": "CFG for structure (higher = obey prompt more)."}),
                "structure_denoise": ("FLOAT", {"default": 0.7, "min": 0.4, "max": 1.0, "step": 0.05, "tooltip": "Denoise strength for structure pass."}),
                "structure_sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpm_2", "tooltip": "Sampler for structure pass."}),
                "structure_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras", "tooltip": "Scheduler for structure pass."}),
                "detail_steps": ("INT", {"default": 10, "min": 5, "max": 50, "tooltip": "Steps for the detail pass (fine features)."}),
                "detail_cfg": ("FLOAT", {"default": 4.2, "min": 2.0, "max": 7.0, "step": 0.1, "tooltip": "CFG for detail (higher = more literal to prompt)."}),
                "detail_denoise": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.7, "step": 0.05, "tooltip": "Denoise strength for detail pass."}),
                "detail_sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "Sampler for detail pass."}),
                "detail_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal", "tooltip": "Scheduler for detail pass."}),
                "adaptive_transition": ("BOOLEAN", {"default": True, "tooltip": "Gently re-anchor structure if denoise was high."}),
                "preserve_composition": ("BOOLEAN", {"default": True, "tooltip": "Keep the base composition stable across passes."}),
                "reuse_noise": ("BOOLEAN", {"default": False, "tooltip": "Reuse structure noise for detail for tighter coupling."}),
                "detail_seed_offset": ("INT", {"default": 1, "min": 0, "max": 10000, "tooltip": "Seed offset for the detail pass."}),
                "model_version": ([
                    "auto",
                    "Illustrious v0.1",
                    "Illustrious v1.0",
                    "Illustrious v1.1",
                    "Illustrious v2.0",
                    "Illustrious v3.0 EPS",
                    "Illustrious v3.0 VPred",
                    "Illustrious v3.5 VPred",
                ], {"default": "auto", "tooltip": "Explicit version to nudge CFG if desired (leave on auto for neutrality)."}),
                "Version Preset Override": (["disabled", "EPS", "VPred"], {"default": "disabled", "tooltip": "Force EPS or VPred-optimized presets regardless of model detection"}),
                # Accept direct scheduler output
                "sigmas": ("SIGMAS", {"tooltip": "Direct schedule tensor (SIGMAS) from a scheduler node."}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("structure_latent", "final_latent")
    FUNCTION = "multi_pass_sample"
    CATEGORY = "Easy Illustrious / Sampling"

    # ---- common_ksampler bridge ----
    def _ksampler_params(self):
        if not hasattr(self, "_ks_params"):
            try:
                self._ks_params = set(inspect.signature(common_ksampler).parameters.keys())
            except Exception:
                self._ks_params = set()
        return self._ks_params

    def _run_common(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_in, denoise=1.0, sigmas=None):
        # Extract mask if present, then remove from latent dict to avoid duplication
        nm = None
        orig = None
        li = latent_in
        if isinstance(li, dict):
            nm = li.get("noise_mask") or li.get("denoise_mask")
            orig = li.get("samples") if isinstance(li.get("samples"), torch.Tensor) else None
            li = dict(li)
            li.pop("noise_mask", None)
            li.pop("denoise_mask", None)

        # Use the cleaned latent dict (without mask keys)
        args = (model, int(seed), int(steps), float(cfg), sampler_name, scheduler, positive, negative, li)
        kwargs = {"denoise": float(denoise)}

        # Pass-through custom sigmas if the core supports it
        try:
            ks_params = self._ksampler_params()
        except Exception:
            ks_params = set()

        # Wire mask if available (prefer core 'mask' param when supported). Core expects 1.0=protect.
        if nm is not None:
            if "mask" in ks_params:
                try:
                    kwargs["mask"] = 1.0 - torch.clamp(nm, 0.0, 1.0)
                except Exception:
                    kwargs["mask"] = nm
            elif "denoise_mask" in ks_params:
                kwargs["denoise_mask"] = nm
            elif "noise_mask" in ks_params:
                kwargs["noise_mask"] = nm

        if sigmas is not None and "sigmas" in ks_params:
            try:
                if hasattr(sigmas, "detach"):
                    sigmas = sigmas.detach().float().cpu()
            except Exception:
                pass
            kwargs["sigmas"] = sigmas

        out = common_ksampler(*args, **kwargs)

        # normalize to {"samples": tensor}
        if isinstance(out, dict) and "samples" in out and isinstance(out["samples"], torch.Tensor):
            result = {"samples": out["samples"]}
        elif isinstance(out, torch.Tensor):
            result = {"samples": out}
        elif isinstance(out, (list, tuple)):
            for it in out:
                if isinstance(it, torch.Tensor):
                    result = {"samples": it}
                    break
                if isinstance(it, dict) and "samples" in it and isinstance(it["samples"], torch.Tensor):
                    result = {"samples": it["samples"]}
                    break
            else:
                maybe = getattr(out, "samples", None)
                if isinstance(maybe, torch.Tensor):
                    result = {"samples": maybe}
                else:
                    raise TypeError(f"Unsupported sampler return type: {type(out)}")
        else:
            maybe = getattr(out, "samples", None)
            if isinstance(maybe, torch.Tensor):
                result = {"samples": maybe}
            else:
                raise TypeError(f"Unsupported sampler return type: {type(out)}")

    # Protect keep/black area by re-blending original latent where mask==0 (no extra softening by default)
        try:
            if nm is not None and isinstance(result.get("samples"), torch.Tensor) and isinstance(orig, torch.Tensor):
                s = result["samples"]
                m = nm.to(device=s.device, dtype=s.dtype)
                if m.shape[1] != 1:
                    m = m[:, :1, :, :]
                keep = 1.0 - m  # black protects
                x0 = orig.to(device=s.device, dtype=s.dtype)
                result["samples"] = s * m + x0 * keep
        except Exception:
            pass

        return result

    def multi_pass_sample(
        self,
        model,
        seed,
        positive,
        negative,
        latent_image,
        structure_steps=12,
        structure_cfg=6.5,
        structure_denoise=0.7,
        structure_sampler="dpm_2",
        structure_scheduler="karras",
        detail_steps=10,
        detail_cfg=4.2,
        detail_denoise=0.2,
        detail_sampler="euler",
        detail_scheduler="normal",
        adaptive_transition=True,
        preserve_composition=True,
        reuse_noise=False,
    detail_seed_offset=1,
    model_version="auto",
    sigmas=None,
    **kw,
    ):

        device = comfy.model_management.get_torch_device()
        base_latent = latent_image["samples"].to(device)

        # Only adjust for *explicit* version picks; "auto" means hands-off
        if model_version != "auto":
            structure_cfg, detail_cfg = self.adjust_cfg_for_version(
                structure_cfg, detail_cfg, model_version
            )

        structure_cfg = float(np.clip(structure_cfg, 3.0, 8.0))
        detail_cfg = float(np.clip(detail_cfg, 2.0, 7.0))

        print(f"[MultiPass] Model: {model_version}")
        print(
            f"[MultiPass] Structure pass: steps={structure_steps} cfg={structure_cfg:.2f} denoise={structure_denoise:.2f}"
        )
        print(
            f"[MultiPass] Detail pass:   steps={detail_steps} cfg={detail_cfg:.2f} denoise={detail_denoise:.2f}"
        )

        # Determine desired mode (EPS/VPred) for presets
        override = kw.get("Version Preset Override", "disabled").lower()
        desired_mode = None
        if override in ("eps", "vpred"):
            desired_mode = override
        else:
            mv = (model_version or "").lower()
            if "vpred" in mv or "v3.5" in mv:
                desired_mode = "vpred"
            elif "eps" in mv:
                desired_mode = "eps"
            # Fallback: if still unknown and user left 'auto', try to infer from the model
            if desired_mode is None and (model_version or "auto").lower() == "auto":
                inferred = (self.detect_model_version(model) or "auto").lower()
                if "vpred" in inferred:
                    desired_mode = "vpred"
                elif "eps" in inferred:
                    desired_mode = "eps"

        # Apply presets gently (only if user left defaults)
        structure_sampler, structure_scheduler = _maybe_apply_version_presets(
            structure_sampler, structure_scheduler, desired_mode, default_sampler="dpm_2", default_scheduler="karras"
        )
        detail_sampler, detail_scheduler = _maybe_apply_version_presets(
            detail_sampler, detail_scheduler, desired_mode, default_sampler="euler", default_scheduler="normal"
        )

        # Resolve effective schedule if provided
        effective_sigmas = sigmas

        with torch.inference_mode():
            # ---- STRUCTURE ----
            s_dict = self._run_common(
                model,
                seed,
                structure_steps,
                structure_cfg,
                structure_sampler,
                structure_scheduler,
                positive,
                negative,
                {"samples": base_latent},
                denoise=structure_denoise,
                sigmas=effective_sigmas,
            )
            s = s_dict["samples"]
            s = _safe_clamp(s, 20.0)  # guard only; higher ceiling

            structure_latent = {"samples": s}

            # ---- Adaptive tweak for detail based on latent std ----
            if adaptive_transition:
                detail_cfg, detail_denoise = self.adaptive_detail_adjustment(
                    s, detail_cfg, detail_denoise, model_version
                )

            # Optional soft re-anchor if structure denoise was high
            if adaptive_transition and structure_denoise >= 0.75:
                blend = float(np.clip((structure_denoise - 0.75) / 0.25, 0.0, 0.25))
                if blend > 0:
                    s = _lerp(s, base_latent, blend)
                    print(f"[MultiPass] Composition re-anchor blend={blend:.2f}")

            # ---- DETAIL ----
            # (preserve_composition could in the future alter conditioning; left as-pass)
            dp, dn = self.prepare_detail_conditioning(
                positive, negative, preserve_composition, model_version
            )
            # Emulate noise reuse via seeds
            dseed = seed if reuse_noise else (seed + detail_seed_offset)
            f_dict = self._run_common(
                model,
                dseed,
                detail_steps,
                detail_cfg,
                detail_sampler,
                detail_scheduler,
                dp,
                dn,
                {"samples": s},
                denoise=detail_denoise,
                sigmas=effective_sigmas,
            )
            f = f_dict["samples"]
            f = _safe_clamp(f, 20.0)

        return (structure_latent, {"samples": f})

    def _detect_pred_mode(self, model):
        """Return 'vpred', 'eps', or None by inspecting the model object."""
        try:
            candidates = [model]
            m = getattr(model, "model", None)
            if m is not None:
                candidates.append(m)
                dm = getattr(m, "diffusion_model", None)
                if dm is not None:
                    candidates.append(dm)
            for obj in candidates:
                if obj is None:
                    continue
                # Direct attribute on UNet or wrapper
                for attr in ("parameterization", "prediction_type"):
                    if hasattr(obj, attr):
                        val = getattr(obj, attr)
                        if isinstance(val, str):
                            low = val.lower()
                            if low.startswith("v") or "vpred" in low:
                                return "vpred"
                            if low.startswith("eps") or "epsilon" in low:
                                return "eps"
                # Config dicts some variants carry
                config = getattr(obj, "model_config", None) or getattr(obj, "config", None)
                if isinstance(config, dict):
                    val = (
                        config.get("parameterization")
                        or config.get("prediction_type")
                        or config.get("param")
                    )
                    if isinstance(val, str):
                        low = val.lower()
                        if low.startswith("v") or "vpred" in low:
                            return "vpred"
                        if low.startswith("eps") or "epsilon" in low:
                            return "eps"
        except Exception:
            pass
        return None

    def detect_model_version(self, model):
        mode = self._detect_pred_mode(model)
        if mode == "vpred":
            return "Illustrious v3.5 VPred"
        if mode == "eps":
            return "Illustrious v3.0 EPS"
        return "auto"
                # (all input definitions are now in INPUT_TYPES above)
    def adjust_cfg_for_version(self, structure_cfg, detail_cfg, version):
        if "Illustrious v0.1" in version:
            structure_cfg = min(7.0, structure_cfg + 0.2)
            detail_cfg = min(5.5, detail_cfg + 0.3)
        elif "Illustrious v1.0" in version:
            structure_cfg = max(5.5, structure_cfg - 0.2)
            detail_cfg = max(4.0, detail_cfg - 0.2)
        elif "Illustrious v1.1" in version or "Illustrious v2.0" in version:
            structure_cfg = max(5.0, structure_cfg - 0.5)
            detail_cfg = max(3.5, detail_cfg - 0.3)

        return structure_cfg, detail_cfg

    def adaptive_detail_adjustment(
        self, structure_result, detail_cfg, detail_denoise, version
    ):
        std_val = _latent_std(structure_result)
        # Latent-space oriented thresholds
        if std_val < 0.18:
            detail_cfg = min(5.5, detail_cfg + 0.3)
            detail_denoise = min(0.6, detail_denoise + 0.1)
        elif std_val > 0.35:
            detail_cfg = max(3.5, detail_cfg - 0.2)
            detail_denoise = max(0.2, detail_denoise - 0.05)
        # Gentle model-specific nudge if explicitly selected
        if version != "auto":
            if (
                "Illustrious v1.0" in version or "Illustrious v1.1" in version
            ) and std_val > 0.28:
                detail_cfg = max(3.8, detail_cfg - 0.2)

        print(
            f"[MultiPass] latent std={std_val:.3f} -> detail cfg={detail_cfg:.2f} denoise={detail_denoise:.2f}"
        )
        return detail_cfg, detail_denoise

    def prepare_detail_conditioning(
        self, positive, negative, preserve_composition, version
    ):
        # Hook to actually alter conditioning later (e.g., lower weight on low-freq terms)
        return positive, negative


# ---------- TriplePass (three stages) ----------


class IllustriousTriplePassSampler:
    """Stabilized Triple-pass sampler: Composition -> Structure -> Details (Illustrious)"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "UNet diffusion model to sample with (SDXL / Illustrious)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Base random seed; use different seeds for different outputs."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning (what you want to see)."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning (what to avoid)."}),
                "latent_image": ("LATENT", {"tooltip": "Starting latent tensor (shape [B,4,H/8,W/8])."}),
            },
            "optional": {
                # Composition stage
                "comp_steps": ("INT", {"default": 8, "min": 5, "max": 50, "tooltip": "Steps for composition (global layout)."}),
                "comp_cfg": ("FLOAT", {"default": 7.0, "min": 3.0, "max": 10.0, "step": 0.1, "tooltip": "CFG for composition stage."}),
                "comp_denoise": ("FLOAT", {"default": 0.9, "min": 0.5, "max": 1.0, "step": 0.01, "tooltip": "Denoise strength for composition."}),
                "comp_sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpm_2", "tooltip": "Sampler for composition pass."}),
                "comp_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras", "tooltip": "Scheduler for composition pass."}),

                # Structure stage
                "struct_steps": ("INT", {"default": 12, "min": 5, "max": 50, "tooltip": "Steps for structure (coarse forms)."}),
                "struct_cfg": ("FLOAT", {"default": 5.5, "min": 3.0, "max": 8.0, "step": 0.1, "tooltip": "CFG for structure stage."}),
                "struct_denoise": ("FLOAT", {"default": 0.7, "min": 0.4, "max": 1.0, "step": 0.05, "tooltip": "Denoise strength for structure pass."}),
                "struct_sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "Sampler for structure pass."}),
                "struct_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal", "tooltip": "Scheduler for structure pass."}),

                # Detail stage
                "detail_steps": ("INT", {"default": 16, "min": 5, "max": 50, "tooltip": "Steps for detail (fine features)."}),
                "detail_cfg": ("FLOAT", {"default": 4.5, "min": 2.5, "max": 7.0, "step": 0.1, "tooltip": "CFG for detail stage."}),
                "detail_denoise": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.8, "step": 0.05, "tooltip": "Denoise strength for detail pass."}),
                "detail_sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler", "tooltip": "Sampler for detail pass."}),
                "detail_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal", "tooltip": "Scheduler for detail pass."}),

                # Behavior and noise control
                "progressive_cfg_decay": ("BOOLEAN", {"default": True, "tooltip": "Gently lower CFG across stages to stabilize."}),
                "adaptive_steps": ("BOOLEAN", {"default": True, "tooltip": "Adapt steps based on latent complexity."}),
                "adaptive_refinement": ("BOOLEAN", {"default": True, "tooltip": "Refine detail settings based on structure output."}),
                "preserve_composition": ("BOOLEAN", {"default": True, "tooltip": "Keep base composition stable if denoise is high."}),
                "separate_noises": ("BOOLEAN", {"default": True, "tooltip": "Use different seeds per stage."}),
                "reuse_comp_noise_for_struct": ("BOOLEAN", {"default": False, "tooltip": "Force structure to reuse composition seed."}),
                "reuse_struct_noise_for_detail": ("BOOLEAN", {"default": False, "tooltip": "Force detail to reuse structure seed."}),
                "comp_seed_offset": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Seed offset for composition pass."}),
                "struct_seed_offset": ("INT", {"default": 1, "min": 0, "max": 10000, "tooltip": "Seed offset for structure pass."}),
                "detail_seed_offset": ("INT", {"default": 2, "min": 0, "max": 10000, "tooltip": "Seed offset for detail pass."}),
                "nan_guard": ("BOOLEAN", {"default": True, "tooltip": "Scrub NaN/Inf and clamp extreme values defensively."}),
                "stability_clamp": ("FLOAT", {"default": 20.0, "min": 5.0, "max": 50.0, "step": 0.5, "tooltip": "Clamp magnitude of latents to this bound if unstable."}),

                # Version awareness
                "model_version": ([
                    "auto",
                    "Illustrious v0.1",
                    "Illustrious v1.0",
                    "Illustrious v1.1",
                    "Illustrious v2.0",
                    "Illustrious v3.0 EPS",
                    "Illustrious v3.0 VPred",
                    "Illustrious v3.5 VPred",
                ], {"default": "auto", "tooltip": "Explicit version to nudge behavior; leave auto for neutrality."}),
                "Version Preset Override": (["disabled", "EPS", "VPred"], {"default": "disabled", "tooltip": "Force EPS or VPred-optimized presets regardless of detection."}),
            },
        }
    RETURN_TYPES = ("LATENT", "LATENT", "LATENT")
    RETURN_NAMES = ("composition_latent", "structure_latent", "final_latent")
    FUNCTION = "triple_pass_sample"
    CATEGORY = "Easy Illustrious / Sampling"

    # ---- common_ksampler bridge (duplicate to avoid dependency on MultiPass) ----
    def _run_common(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_in, denoise=1.0):
        # Extract mask if present, then remove from latent dict to avoid duplication
        nm = None
        orig = None
        li = latent_in
        if isinstance(li, dict):
            nm = li.get("noise_mask") or li.get("denoise_mask")
            orig = li.get("samples") if isinstance(li.get("samples"), torch.Tensor) else None
            li = dict(li)
            li.pop("noise_mask", None)
            li.pop("denoise_mask", None)
        args = (model, int(seed), int(steps), float(cfg), sampler_name, scheduler, positive, negative, li)
        kwargs = {"denoise": float(denoise)}
        # Prefer core 'mask' param when supported
        try:
            ks_params = set(inspect.signature(common_ksampler).parameters.keys())
        except Exception:
            ks_params = set()
        if nm is not None:
            if "mask" in ks_params:
                try:
                    kwargs["mask"] = 1.0 - torch.clamp(nm, 0.0, 1.0)
                except Exception:
                    kwargs["mask"] = nm
            elif "denoise_mask" in ks_params:
                kwargs["denoise_mask"] = nm
            elif "noise_mask" in ks_params:
                kwargs["noise_mask"] = nm

        out = common_ksampler(*args, **kwargs)
        # normalize to {"samples": tensor}
        if isinstance(out, dict) and "samples" in out and isinstance(out["samples"], torch.Tensor):
            result = {"samples": out["samples"]}
        elif isinstance(out, torch.Tensor):
            result = {"samples": out}
        elif isinstance(out, (list, tuple)):
            for it in out:
                if isinstance(it, torch.Tensor):
                    result = {"samples": it}
                    break
                if isinstance(it, dict) and "samples" in it and isinstance(it["samples"], torch.Tensor):
                    result = {"samples": it["samples"]}
                    break
            else:
                maybe = getattr(out, "samples", None)
                if isinstance(maybe, torch.Tensor):
                    result = {"samples": maybe}
                else:
                    raise TypeError(f"Unsupported sampler return type: {type(out)}")
        else:
            maybe = getattr(out, "samples", None)
            if isinstance(maybe, torch.Tensor):
                result = {"samples": maybe}
            else:
                raise TypeError(f"Unsupported sampler return type: {type(out)}")

    # Protect keep/black area by re-blending original latent where mask==0 (no extra softening by default)
        try:
            if nm is not None and isinstance(result.get("samples"), torch.Tensor) and isinstance(orig, torch.Tensor):
                s = result["samples"]
                m = nm.to(device=s.device, dtype=s.dtype)
                if m.shape[1] != 1:
                    m = m[:, :1, :, :]
                keep = 1.0 - m  # black protects
                x0 = orig.to(device=s.device, dtype=s.dtype)
                result["samples"] = s * m + x0 * keep
        except Exception:
            pass

        return result

    def triple_pass_sample(
        self,
        model,
        seed,
        positive,
        negative,
        latent_image,
        comp_steps=8,
        comp_cfg=7.0,
        comp_denoise=0.9,
        comp_sampler="dpm_2",
        comp_scheduler="karras",
        struct_steps=12,
        struct_cfg=5.5,
        struct_denoise=0.7,
        struct_sampler="euler",
        struct_scheduler="normal",
        detail_steps=16,
        detail_cfg=4.5,
        detail_denoise=0.35,
        detail_sampler="euler",
        detail_scheduler="normal",
        progressive_cfg_decay=True,
        adaptive_steps=True,
        adaptive_refinement=True,
        preserve_composition=True,
        separate_noises=True,
        reuse_comp_noise_for_struct=False,
        reuse_struct_noise_for_detail=False,
        comp_seed_offset=0,
        struct_seed_offset=1,
        detail_seed_offset=2,
        nan_guard=True,
        stability_clamp=20.0,
        model_version="auto",
        **kw,
    ):
        device = comfy.model_management.get_torch_device()
        base_latent = latent_image["samples"].to(device)

        if model_version != "auto":
            comp_cfg, struct_cfg, detail_cfg = self.adjust_cfg_for_version_triple(
                comp_cfg, struct_cfg, detail_cfg, model_version
            )

        # Mild decay, but respect user intent
        if progressive_cfg_decay:
            comp_cfg = max(comp_cfg, struct_cfg + 0.5, detail_cfg + 1.0)
            struct_cfg = max(struct_cfg, detail_cfg + 0.25)

        comp_cfg = float(np.clip(comp_cfg, 4.0, 10.0))
        struct_cfg = float(np.clip(struct_cfg, 3.0, 8.0))
        detail_cfg = float(np.clip(detail_cfg, 2.5, 7.0))

        print(f"[TriplePass] Model={model_version}")
        print(
            f"[TriplePass] Comp: steps={comp_steps} cfg={comp_cfg:.2f} denoise={comp_denoise:.2f}"
        )
        print(
            f"[TriplePass] Struct: steps={struct_steps} cfg={struct_cfg:.2f} denoise={struct_denoise:.2f}"
        )
        print(
            f"[TriplePass] Detail: steps={detail_steps} cfg={detail_cfg:.2f} denoise={detail_denoise:.2f}"
        )

        # Determine desired mode (EPS/VPred) for presets
        override = kw.get("Version Preset Override", "disabled").lower()
        desired_mode = None
        if override in ("eps", "vpred"):
            desired_mode = override
        else:
            mv = (model_version or "").lower()
            if "vpred" in mv or "v3.5" in mv:
                desired_mode = "vpred"
            elif "eps" in mv:
                desired_mode = "eps"
            # Fallback: if still unknown and user left 'auto', try to infer from the model
            if desired_mode is None and (model_version or "auto").lower() == "auto":
                inferred = (self.detect_model_version(model) or "auto").lower()
                if "vpred" in inferred:
                    desired_mode = "vpred"
                elif "eps" in inferred:
                    desired_mode = "eps"

        # Apply presets gently (only if user left defaults)
        comp_sampler, comp_scheduler = _maybe_apply_version_presets(
            comp_sampler, comp_scheduler, desired_mode, default_sampler="dpm_2", default_scheduler="karras"
        )
        struct_sampler, struct_scheduler = _maybe_apply_version_presets(
            struct_sampler, struct_scheduler, desired_mode, default_sampler="euler", default_scheduler="normal"
        )
        detail_sampler, detail_scheduler = _maybe_apply_version_presets(
            detail_sampler, detail_scheduler, desired_mode, default_sampler="euler", default_scheduler="normal"
        )

        with torch.inference_mode():
            # Resolve seeds to emulate noise control
            comp_seed = seed + comp_seed_offset
            if not separate_noises:
                struct_seed = comp_seed
                detail_seed = comp_seed
            else:
                struct_seed = comp_seed if reuse_comp_noise_for_struct else (seed + struct_seed_offset)
                detail_seed = struct_seed if reuse_struct_noise_for_detail else (seed + detail_seed_offset)

            # ---- Composition ----
            c_dict = self._run_common(
                model,
                comp_seed,
                comp_steps,
                comp_cfg,
                comp_sampler,
                comp_scheduler,
                positive,
                negative,
                {"samples": base_latent},
                denoise=comp_denoise,
            )
            c = c_dict["samples"]
            c = _safe_clamp(c, stability_clamp)
            if preserve_composition and comp_denoise > 0.9:
                # much lighter re-anchor
                blend = float(np.clip((comp_denoise - 0.9) / 0.1, 0.0, 0.15))
                if blend > 0:
                    c = _lerp(c, base_latent, blend)
                    print(f"[TriplePass] Comp blend applied={blend:.2f}")
            comp_latent = {"samples": c}

            # ---- Structure ----
            if adaptive_steps:
                struct_steps = self.adapt_structure_steps(c, struct_steps)
            s_dict = self._run_common(
                model,
                struct_seed,
                struct_steps,
                struct_cfg,
                struct_sampler,
                struct_scheduler,
                positive,
                negative,
                {"samples": c},
                denoise=struct_denoise,
            )
            s = s_dict["samples"]
            s = _safe_clamp(s, stability_clamp)
            struct_latent = {"samples": s}

            # ---- Adaptive refinement ----
            if adaptive_refinement:
                detail_cfg, detail_denoise = self.adaptive_detail_adjustment_triple(
                    s, detail_cfg, detail_denoise, model_version
                )
                print(
                    f"[TriplePass] Adaptive detail -> cfg={detail_cfg:.2f} denoise={detail_denoise:.2f}"
                )
            if adaptive_steps:
                detail_steps = self.adapt_detail_steps(s, detail_steps)
            if adaptive_refinement and struct_denoise > 0.8:
                blend = float(np.clip((struct_denoise - 0.8) / 0.15, 0.0, 0.25))
                if blend > 0:
                    s = _lerp(s, c, blend)
                    print(f"[TriplePass] Struct re-anchor blend={blend:.2f}")

            # ---- Detail ----
            f_dict = self._run_common(
                model,
                detail_seed,
                detail_steps,
                detail_cfg,
                detail_sampler,
                detail_scheduler,
                positive,
                negative,
                {"samples": s},
                denoise=detail_denoise,
            )
            f = f_dict["samples"]
            f = _safe_clamp(f, stability_clamp)
            final_latent = {"samples": f}

        return (comp_latent, struct_latent, final_latent)

    # --- Helpers ---
    def _validate_and_clamp(self, tensor, fallback, nan_guard, clamp_val, tag=""):
        # Kept for compatibility if something calls it elsewhere
        if nan_guard and (
            torch.isnan(tensor).any()
            or torch.isinf(tensor).any()
            or (tensor.abs() > clamp_val).any()
        ):
            print(
                f"[TriplePass][WARN] {tag} pass produced invalid values. Scrubbing & clamping."
            )
        return _safe_clamp(tensor, clamp_val)

    def _detect_pred_mode(self, model):
        """Return 'vpred', 'eps', or None by inspecting the model object."""
        try:
            candidates = [model]
            m = getattr(model, "model", None)
            if m is not None:
                candidates.append(m)
                dm = getattr(m, "diffusion_model", None)
                if dm is not None:
                    candidates.append(dm)
            for obj in candidates:
                if obj is None:
                    continue
                # Direct attribute on UNet or wrapper
                for attr in ("parameterization", "prediction_type"):
                    if hasattr(obj, attr):
                        val = getattr(obj, attr)
                        if isinstance(val, str):
                            low = val.lower()
                            if low.startswith("v") or "vpred" in low:
                                return "vpred"
                            if low.startswith("eps") or "epsilon" in low:
                                return "eps"
                # Config dicts some variants carry
                config = getattr(obj, "model_config", None) or getattr(obj, "config", None)
                if isinstance(config, dict):
                    val = (
                        config.get("parameterization")
                        or config.get("prediction_type")
                        or config.get("param")
                    )
                    if isinstance(val, str):
                        low = val.lower()
                        if low.startswith("v") or "vpred" in low:
                            return "vpred"
                        if low.startswith("eps") or "epsilon" in low:
                            return "eps"
        except Exception:
            pass
        return None

    def detect_model_version(self, model):
        mode = self._detect_pred_mode(model)
        if mode == "vpred":
            return "Illustrious v3.5 VPred"
        if mode == "eps":
            return "Illustrious v3.0 EPS"
        return "auto"

    def adjust_cfg_for_version_triple(self, comp_cfg, struct_cfg, detail_cfg, version):
        if "Illustrious v0.1" in version:
            comp_cfg += 0.3
            struct_cfg += 0.2
            detail_cfg += 0.2
        elif "Illustrious v1.0" in version:
            comp_cfg -= 0.2
            struct_cfg -= 0.2
            detail_cfg -= 0.1
        elif "Illustrious v1.1" in version or "Illustrious v2.0" in version:
            comp_cfg -= 0.4
            struct_cfg -= 0.4
            detail_cfg -= 0.2

        return comp_cfg, struct_cfg, detail_cfg

    def adaptive_detail_adjustment_triple(
        self, struct_result, detail_cfg, detail_denoise, version
    ):
        # FIX: used to read undefined "structure_result"
        std_val = _latent_std(struct_result)
        if std_val < 0.18:
            detail_cfg = min(detail_cfg + 0.3, 6.8)
            detail_denoise = min(detail_denoise + 0.1, 0.6)
        elif std_val > 0.35:
            detail_cfg = max(detail_cfg - 0.2, 3.0)
            detail_denoise = max(detail_denoise - 0.05, 0.25)
        if version != "auto":
            if (
                "Illustrious v1.1" in version or "Illustrious v2.0" in version
            ) and std_val > 0.28:
                detail_cfg = max(detail_cfg - 0.2, 3.2)
        return detail_cfg, detail_denoise

    def adapt_structure_steps(self, comp_result, base_steps):
        # Use smaller thresholds for latent edges
        grad_x = torch.abs(comp_result[:, :, :, 1:] - comp_result[:, :, :, :-1])
        grad_y = torch.abs(comp_result[:, :, 1:, :] - comp_result[:, :, :-1, :])
        complexity = (torch.mean(grad_x) + torch.mean(grad_y)).item()
        if complexity > 0.12:
            return min(base_steps + 4, 40)
        elif complexity < 0.04:
            return max(base_steps - 2, 6)
        return base_steps

    def adapt_detail_steps(self, struct_result, base_steps):
        # Laplacian kernel for edge/sharpness detection, 4 channels
        lap = torch.tensor(
            [[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]],
            dtype=struct_result.dtype,
            device=struct_result.device,
        ).repeat(4, 1, 1, 1)  # [4, 1, 3, 3]
        sharpness = torch.mean(
            torch.abs(torch.nn.functional.conv2d(struct_result, lap, padding=1, groups=4))
        ).item()
        if sharpness < 0.06:
            return min(base_steps + 6, 50)
        elif sharpness > 0.18:
            return max(base_steps - 3, 6)
        return base_steps
