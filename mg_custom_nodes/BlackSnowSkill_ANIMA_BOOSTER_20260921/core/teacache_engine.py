"""
© 2026 blacksnowskill (BSS). All rights reserved.
Developed by: blacksnowskill (BSS)

core/teacache_engine.py
Adaptive Timestep-aware Cache (TeaCache) engine for Anima DiT.

Uses ComfyUI's model_patcher callback system (set_model_unet_function_wrapper)
to intercept each sampling step cleanly without monkey-patching model.forward.

The key insight: instead of patching the model's internal forward, we wrap
the outer UNet call that ComfyUI's sampler makes. This gives us access to
the timestep and lets us decide to skip or compute.

Innovation: ADAPTIVE threshold by timestep position
  - Early steps (structure forming)  → low threshold  → compute accurately
  - Late steps  (details stable)     → high threshold → skip aggressively
"""

import logging
import torch
import numpy as np

logger = logging.getLogger("ANIMA_BOOSTER.teacache")


# ─────────────────────────────────────────────
#  Math helpers
# ─────────────────────────────────────────────

@torch.compiler.disable()
def relative_l1_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L1 — measures how much the timestep embedding changed."""
    return (
        (a - b).abs().mean() / a.abs().mean().clamp(min=1e-8)
    ).to(torch.float32).item()


def adaptive_threshold(
    step_percent: float,
    base: float,
    early_factor: float,
    late_factor: float,
) -> float:
    """
    step_percent: 0.0 = first step (max noise), 1.0 = last step (clean)
    """
    if step_percent < 0.35:
        return base * early_factor   # structure forming — be careful
    elif step_percent > 0.70:
        return base * late_factor    # details stable — cache aggressively
    else:
        return base


# ─────────────────────────────────────────────
#  Cache state
# ─────────────────────────────────────────────

class _StreamState:
    __slots__ = ("prev_x", "prev_out", "prev_t", "accumulated", "skips", "max_t")

    def __init__(self):
        self.prev_x: torch.Tensor | None = None
        self.prev_out: torch.Tensor | None = None
        self.prev_t: float = -1.0
        self.accumulated: float = 0.0
        self.skips: int = 0
        self.max_t: float = 1000.0


# ─────────────────────────────────────────────
#  Model patching
# ─────────────────────────────────────────────

def patch_model_with_teacache(
    model,
    threshold: float = 0.15,
    version: str = "v1 (Legacy Fast)",
    adaptive: bool = True,
    early_factor: float = 0.4,
    late_factor: float = 1.8,
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    cache_device: str = "cuda",
) -> object:
    """
    Patches the model to use TeaCache via ComfyUI's unet_function_wrapper.

    The wrapper intercepts every sampler call to the model and decides
    whether to recompute or return the cached output.
    """
    m = model.clone()

    cfg = {
        "threshold": threshold,
        "version": version,
        "adaptive": adaptive,
        "early_factor": early_factor,
        "late_factor": late_factor,
        "start_percent": start_percent,
        "end_percent": end_percent,
        "cache_device": cache_device,
    }

    # Per-run cache state (cond / uncond)
    state = {
        "cond": _StreamState(),
        "uncond": _StreamState(),
        "step": 0,
        "total_steps": 20,  # will be updated
    }

    def unet_wrapper(apply_model, args: dict):
        """
        Called by ComfyUI sampler for each model evaluation.
        apply_model: the actual model call  lambda: model(x, t, **cond)
        args: dict with 'input', 'timestep', 'c' (conditioning), 'cond_or_uncond'
        """
        x = args["input"]
        t = args["timestep"]
        cond_or_uncond = args.get("cond_or_uncond", [0])

        # Determine stream (cond vs uncond)
        stream_key = "uncond" if 1 in cond_or_uncond else "cond"
        st = state[stream_key]

        # Get current timestep float
        t_val = float(t[0]) if t.numel() > 0 else 500.0

        # Detect new generation run (if resolution changed or timestep went UP instead of down)
        if st.prev_x is not None:
            if x.shape != st.prev_x.shape or t_val > st.prev_t + 1e-4:
                # Reset state
                st.prev_x = None
                st.prev_out = None
                st.accumulated = 0.0
                st.skips = 0
                st.max_t = max(1e-4, t_val)

        if st.prev_x is None:
            st.max_t = max(1e-4, t_val)

        # Estimate step position (0.0 = start, 1.0 = end)
        if cfg["version"] == "v1 (Legacy Fast)":
            # Legacy mode: fixed normalizer of 1000.0, which causes aggressive early caching on SDE samplers
            step_pct = max(0.0, min(1.0, 1.0 - t_val / 1000.0))
        else:
            # Precise mode: dynamic normalization adapting exactly to any timestep range (sigmas, 1000..0, 1..0)
            step_pct = max(0.0, min(1.0, 1.0 - t_val / st.max_t))

        st.prev_t = t_val

        # Check if we're in the active range
        in_range = cfg["start_percent"] <= step_pct <= cfg["end_percent"]

        if not in_range or st.prev_x is None:
            # Always compute on first step or outside range
            out = apply_model(args["input"], args["timestep"], **args["c"])
            st.prev_x = x.detach()
            st.prev_out = out.detach().to(cache_device)
            st.accumulated = 0.0
            return out

        # Compute how much the input changed since last step
        delta = relative_l1_distance(st.prev_x, x)

        # Adaptive threshold
        thr = adaptive_threshold(step_pct, cfg["threshold"],
                                  cfg["early_factor"], cfg["late_factor"]) if cfg["adaptive"] \
              else cfg["threshold"]

        st.accumulated += delta

        if st.accumulated < thr:
            # Skip — return cached output
            st.skips += 1
            logger.debug(
                f"[TeaCache] {stream_key} step_pct={step_pct:.2f} "
                f"acc={st.accumulated:.4f} < thr={thr:.4f} → SKIP (total skips: {st.skips})"
            )
            return st.prev_out.to(x.device).to(x.dtype)
        else:
            # Compute
            st.accumulated = 0.0
            out = apply_model(args["input"], args["timestep"], **args["c"])
            st.prev_x = x.detach()
            st.prev_out = out.detach().to(cache_device)
            return out

    # Register with ComfyUI's model patcher
    m.set_model_unet_function_wrapper(unet_wrapper)

    logger.info(
        f"[TeaCache] Patched | version={version} | threshold={threshold} | adaptive={adaptive} | "
        f"early={early_factor} | late={late_factor} | "
        f"range=[{start_percent:.0%}–{end_percent:.0%}]"
    )
    return m
