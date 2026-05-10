import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IllustriousAttentionCouple:
    """
    Spatial attention coupling for Illustrious (v0.1–v2.0) with dynamic block discovery.

    What it does
    - Finds cross-attention modules ("attn2"/CrossAttention) dynamically across Down/Mid/Up stages.
    - Scales cross-attention output per spatial position using an input mask, resized per block.
    - Safe: dtype/device aligned, NaN/Inf guarded, idempotent patching and unpatching.

    Notes
    - This is model-architecture aware but not hardcoded to SD1.5/SDXL indices.
    - If a block's spatial grid cannot be inferred (non-square token count), that block is skipped.
    - Mask is optional; if absent or empty, the node is a no-op and unpatches previous hooks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
            "optional": {
                # Provide a spatial mask (IMAGE: [B,H,W,C] in 0..1). If omitted, patch is removed.
                "mask": (
                    "IMAGE",
                    {
                        "tooltip": "Spatial mask (NHWC 0..1). White strengthens attention; black leaves unchanged.",
                    },
                ),
                # Optionally pass the latent to infer target HxW; helpful for perfect mapping.
                "latent_image": (
                    "LATENT",
                    {
                        "tooltip": "Optional latent to infer UNet grid size for precise mask alignment.",
                    },
                ),
                # Which stages to affect
                "affect_down": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Apply coupling to Down blocks."},
                ),
                "affect_mid": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Apply coupling to Mid block."},
                ),
                "affect_up": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Apply coupling to Up blocks."},
                ),
                # Coupling strength (per-position blend)
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Blend strength per position: 0=no change, 1=full masked coupling.",
                    },
                ),
                # Reserved for future token-based gating (unused for now)
                "token_filter": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Reserved: token-level gating (not used yet). Leave empty.",
                    },
                ),
                # Diagnostics verbosity
                "advanced_logs": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Print patching details to console."},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "report")
    FUNCTION = "apply"
    CATEGORY = "Easy Illustrious / Advanced"

    # ---- helpers ----
    def _get_diffusion_root(self, model):
        """Try to locate the UNet/diffusion root module within Comfy's model wrapper."""
        # Common candidates across Comfy releases
        for attr in ("model", "diffusion_model", "inner_model", "unet", "module"):
            if hasattr(model, attr):
                obj = getattr(model, attr)
                # Some wrappers nest the diffusion model one level deeper
                for inner in (
                    "diffusion_model",
                    "model",
                    "inner_model",
                    "unet",
                    "module",
                ):
                    if hasattr(obj, inner) and isinstance(
                        getattr(obj, inner), nn.Module
                    ):
                        return getattr(obj, inner)
                if isinstance(obj, nn.Module):
                    return obj
        # Fallback to model itself if it's a nn.Module
        return model if isinstance(model, nn.Module) else None

    def _named_modules_safe(self, root: nn.Module):
        try:
            return list(root.named_modules())
        except Exception:
            # Some wrappers restrict iteration; try direct children
            out = []
            for name, child in root._modules.items():
                out.append((name, child))
                try:
                    for subname, sub in child.named_modules():
                        out.append((f"{name}.{subname}", sub))
                except Exception:
                    pass
            return out

    def _infer_stage_from_name(self, full_name: str) -> str:
        lname = full_name.lower()
        if any(k in lname for k in ["down", "input_blocks", "down_blocks"]):
            return "down"
        if any(k in lname for k in ["mid", "middle_block", "mid_block"]):
            return "mid"
        if any(k in lname for k in ["up", "output_blocks", "up_blocks"]):
            return "up"
        return "unknown"

    def _to_hw_from_tokens(
        self, tokens: int, latent_hw: Optional[Tuple[int, int]] = None
    ) -> Optional[Tuple[int, int]]:
        # If latent size known, propagate typical scale factors (handle most UNet stages)
        if latent_hw is not None:
            H, W = latent_hw
            candidates = [
                (H, W),
                (H // 2, W // 2),
                (H // 4, W // 4),
                (H // 8, W // 8),
                (H // 16, W // 16),
            ]
            for h, w in candidates:
                if h > 0 and w > 0 and h * w == tokens:
                    return (h, w)
        # Fallback: square grid if perfect square
        side = int(math.sqrt(tokens))
        if side * side == tokens:
            return (side, side)
        return None

    def _prepare_mask(
        self, mask: torch.Tensor, target_hw: Tuple[int, int], device, dtype
    ) -> torch.Tensor:
        # Expect IMAGE tensor in NHWC 0..1; reduce to single channel and resize to target HxW
        # Accept shapes: [B,H,W,C] or [H,W,C] or [H,W]
        if mask.dim() == 4:
            # take first batch
            mask = mask[0]
        if mask.dim() == 3:
            # average channels
            mask = mask.mean(dim=-1)
        mask = mask.to(device=device, dtype=torch.float32)
        mask = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
        # Resize to target using bilinear
        h, w = target_hw
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        mask = mask.clamp(0.0, 1.0).to(dtype)
        return mask

    def _unpatch_existing(self, model) -> int:
        restored = 0
        patches: List[Tuple[nn.Module, str, callable]] = getattr(
            model, "_ill_attn_couple_patches", []
        )
        for mod, attr, orig_fn in patches:
            try:
                if hasattr(mod, attr):
                    setattr(mod, attr, orig_fn)
                    restored += 1
            except Exception:
                pass
        if hasattr(model, "_ill_attn_couple_patches"):
            model._ill_attn_couple_patches = []
        if hasattr(model, "_ill_attn_couple_state"):
            delattr(model, "_ill_attn_couple_state")
        return restored

    def _discover_cross_attention_modules(
        self, root: nn.Module
    ) -> List[Tuple[str, nn.Module, Optional[nn.Module]]]:
        """Return list of (name, module, attn2_or_none). If module has attribute 'attn2', return that as third; else None for generic wrapper."""
        found = []
        for name, m in self._named_modules_safe(root):
            try:
                # SDXL-style BasicTransformerBlock with .attn2
                if hasattr(m, "attn2") and isinstance(getattr(m, "attn2"), nn.Module):
                    found.append((name, m, getattr(m, "attn2")))
                    continue
                # A generic cross attention module - heuristic: has forward with encoder_hidden_states kw or param count > 1
                fwd = getattr(m, "forward", None)
                if callable(fwd):
                    code = fwd.__code__ if hasattr(fwd, "__code__") else None
                    if code and (
                        "encoder_hidden_states" in code.co_varnames
                        or code.co_argcount >= 2
                    ):
                        # Avoid capturing too-generic containers; prefer modules that have projections
                        if any(
                            hasattr(m, attr)
                            for attr in (
                                "to_q",
                                "to_k",
                                "to_v",
                                "query",
                                "key",
                                "value",
                            )
                        ):
                            found.append((name, m, None))
            except Exception:
                continue
        return found

    # ---- main ----
    def apply(
        self,
        model,
        positive,
        negative,
        mask=None,
        latent_image=None,
        affect_down=True,
        affect_mid=True,
        affect_up=True,
        strength=0.5,
        token_filter="",
        advanced_logs=False,
    ):

        # Unpatch first to keep idempotent behavior (also acts as no-op if nothing to restore)
        restored = self._unpatch_existing(model)

        report_lines = []
        if restored:
            report_lines.append(f"Restored {restored} previous attention hooks.")

        # Early exit if no mask or zero strength
        if mask is None or strength <= 0.0:
            report_lines.append("No mask or zero strength; model left unmodified.")
            return (model, positive, negative, "\n".join(report_lines))

        diff_root = self._get_diffusion_root(model)
        if diff_root is None:
            report_lines.append("Could not locate diffusion model; skipping patch.")
            return (model, positive, negative, "\n".join(report_lines))

        # Infer latent HxW if provided (Comfy LATENT stores [B,C,H/8,W/8])
        latent_hw = None
        try:
            if (
                latent_image is not None
                and isinstance(latent_image, dict)
                and "samples" in latent_image
            ):
                samples = latent_image["samples"]
                if isinstance(samples, torch.Tensor) and samples.dim() == 4:
                    # Convert latent spatial to UNet token grid at current resolution (base is latent)
                    latent_hw = (int(samples.shape[2]), int(samples.shape[3]))
        except Exception:
            pass

        # Cache state on model for use in patched forwards
        stages_enabled = {"down": affect_down, "mid": affect_mid, "up": affect_up}
        model._ill_attn_couple_state = {
            "strength": float(max(0.0, min(1.0, strength))),
            "latent_hw": latent_hw,
            "stages_enabled": stages_enabled,
            "advanced_logs": bool(advanced_logs),
        }

        # Discover and patch
        targets = self._discover_cross_attention_modules(diff_root)
        patched = 0

        # Prepare a CPU-side ref; per-layer we will resize to specific HxW/dtype/device
        base_mask = mask

        for full_name, module, attn2 in targets:
            stage = self._infer_stage_from_name(full_name)
            if stage in stages_enabled and not stages_enabled[stage]:
                continue

            # Determine which callable to patch: attn2.forward if present, else module.forward
            target_mod = attn2 if attn2 is not None else module
            if not hasattr(target_mod, "forward"):
                continue

            orig_forward = target_mod.forward

            def make_wrapper(orig_fn):
                def wrapped_forward(hidden_states, *args, **kwargs):
                    out = orig_fn(hidden_states, *args, **kwargs)
                    try:
                        state = getattr(model, "_ill_attn_couple_state", None)
                        if state is None:
                            return out

                        # hidden_states and out usually share shape [B, N, C]
                        if not (isinstance(out, torch.Tensor) and out.dim() == 3):
                            return out

                        B, N, C = out.shape
                        # Infer HxW from tokens and latent hint
                        hw = self._to_hw_from_tokens(N, state.get("latent_hw"))
                        if hw is None:
                            return out
                        h, w = hw

                        # Get device/dtype from out to ensure alignment
                        mdev = out.device
                        mdtype = out.dtype

                        # Prepare and resize mask to h x w
                        m = base_mask
                        if isinstance(m, torch.Tensor):
                            m_resized = self._prepare_mask(
                                m, (h, w), device=mdev, dtype=mdtype
                            )
                        else:
                            return out

                        # Broadcast: [B,h*w,1]
                        pos_scale = m_resized.view(1, h * w, 1)
                        if pos_scale.shape[1] != N:
                            # As a safety, skip if mismatch
                            return out

                        s = state.get("strength", 0.5)
                        # Per-position blend: out = hidden + (out - hidden) * (s * pos)
                        # If orig input is available, use hidden_states; else use out as both
                        base = (
                            hidden_states
                            if isinstance(hidden_states, torch.Tensor)
                            and hidden_states.shape == out.shape
                            else out
                        )
                        delta = out - base
                        # Guard NaNs/Infs
                        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                        out = base + delta * (s * pos_scale)
                        return out
                    except Exception:
                        return out

                return wrapped_forward

            try:
                wrapper = make_wrapper(orig_forward)
                target_mod.forward = wrapper
                # Track patch to restore later
                if not hasattr(model, "_ill_attn_couple_patches"):
                    model._ill_attn_couple_patches = []
                model._ill_attn_couple_patches.append(
                    (target_mod, "forward", orig_forward)
                )
                patched += 1
            except Exception:
                continue

        report_lines = []
        if patched == 0:
            # Clean state if nothing was patched
            self._unpatch_existing(model)
            report_lines.append(
                "No compatible cross-attention modules found; no changes applied."
            )
        else:
            report_lines.append(
                f"Patched {patched} cross-attention call(s) across stages."
            )
            report_lines.append(
                f"Stages: down={affect_down}, mid={affect_mid}, up={affect_up}"
            )
            if latent_hw is not None:
                report_lines.append(f"Latent HW: {latent_hw}")

        return (model, positive, negative, "\n".join(report_lines))
