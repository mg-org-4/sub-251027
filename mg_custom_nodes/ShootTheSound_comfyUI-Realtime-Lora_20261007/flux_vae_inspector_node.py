"""
ComfyUI Flux VAE Inspector — Individual Tensor Analysis

For Flux 2 Klein 9B's VAE (Flux.1 Autoencoder variant, 32ch latent).
Analyzes each of the 125 tensor units to understand their contribution.
"""

import re
import os
import json
import gc
import torch
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple

# Import block definitions from the debiaser
from .flux_vae_debiaser_node import BLOCKS, ALL_BLOCK_IDS, DEC_BLOCK_IDS, ENC_BLOCK_IDS, _KEY_TO_BLOCK


class FluxVAEInspector:
    """
    Analyzes every tensor unit in Flux VAE.
    125 units — per-tensor weight norms, distributions, and optional decode ablation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {
                    "tooltip": "VAE model to analyze"
                }),
                "analysis_mode": (["weight_analysis", "decode_ablation_HEAVY"], {
                    "default": "weight_analysis",
                    "tooltip": "weight_analysis = fast stats. "
                               "decode_ablation = test actual decode impact per tensor unit (SLOW)."
                }),
                "ablation_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "How much to weaken each unit during ablation (0.5 = 50%)"
                }),
            },
            "optional": {
                "latent": ("LATENT", {
                    "tooltip": "Required for decode_ablation mode"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "json_data")
    OUTPUT_TOOLTIPS = (
        "Human-readable analysis report",
        "JSON data for further processing",
    )
    FUNCTION = "inspect"
    CATEGORY = "model_patches/analysis"
    OUTPUT_NODE = True
    DESCRIPTION = """Analyzes every tensor unit in Flux VAE (125 units).

⚡ WEIGHT ANALYSIS (default): Fast
  - Weight norms, distributions, max values per tensor unit
  - Identifies which convs/norms dominate

⚠️ DECODE ABLATION: Slow, needs latent
  - Weakens each decoder unit individually and decodes
  - Measures pixel-level + per-channel (R/G/B) impact
  - Precise importance rankings for color tuning

Run weight analysis first, then ablation with a sample latent."""

    def inspect(self, vae, analysis_mode="weight_analysis", ablation_strength=0.5, latent=None):
        print(f"[Flux VAE Inspector] Starting {analysis_mode}...")

        state_dict = vae.first_stage_model.state_dict()

        # Map keys
        block_key_map = defaultdict(list)
        for key in state_dict.keys():
            bid = _KEY_TO_BLOCK.get(key)
            if bid is not None:
                block_key_map[bid].append(key)

        results = {
            "architecture": self._analyze_architecture(state_dict),
            "blocks": {},
            "summary": {},
            "recommendations": [],
        }

        # Per-unit analysis
        max_norm = 0.0
        for bid in ALL_BLOCK_IDS:
            keys = block_key_map.get(bid, [])
            if not keys:
                continue
            block_data = self._analyze_unit(state_dict, bid, keys)
            results["blocks"][bid] = block_data
            if block_data["weight_norm"] > max_norm:
                max_norm = block_data["weight_norm"]

        # Normalize scores
        if max_norm > 0:
            for bid in results["blocks"]:
                results["blocks"][bid]["score"] = \
                    (results["blocks"][bid]["weight_norm"] / max_norm) * 100.0

        # Ablation
        if analysis_mode == "decode_ablation_HEAVY":
            if latent is not None:
                results["ablation"] = self._run_ablation(
                    vae, latent, block_key_map, state_dict, ablation_strength
                )
            else:
                results["ablation"] = {"error": "No latent provided — connect a latent sample"}

        results["summary"] = self._compute_summary(results)
        results["recommendations"] = self._generate_recommendations(results)

        report = self._format_report(results)
        json_data = self._create_ui_json(results)

        print(f"[Flux VAE Inspector] Done.")
        return {"ui": {"analysis_json": [json_data]}, "result": (report, json_data)}

    def _analyze_architecture(self, state_dict):
        total_params = 0
        dec_params = 0
        enc_params = 0
        dtypes = set()
        for key, tensor in state_dict.items():
            total_params += tensor.numel()
            dtypes.add(str(tensor.dtype))
            if key.startswith("decoder."):
                dec_params += tensor.numel()
            elif key.startswith("encoder."):
                enc_params += tensor.numel()
        return {
            "total_params": total_params,
            "decoder_params": dec_params,
            "encoder_params": enc_params,
            "dtypes": list(dtypes),
            "tensor_count": len(state_dict),
            "unit_count": len(ALL_BLOCK_IDS),
        }

    def _analyze_unit(self, state_dict, bid, keys):
        total_params = 0
        total_bytes = 0
        total_norm = 0.0
        max_val = 0.0
        mean_abs_sum = 0.0
        shapes = []

        for key in keys:
            tensor = state_dict[key]
            numel = tensor.numel()
            total_params += numel
            total_bytes += numel * tensor.element_size()
            shapes.append(list(tensor.shape))

            t = tensor.float()
            norm = t.norm().item()
            total_norm += norm
            mv = t.abs().max().item()
            if mv > max_val:
                max_val = mv
            mean_abs_sum += t.abs().mean().item()

        return {
            "param_count": total_params,
            "memory_mb": total_bytes / (1024 * 1024),
            "weight_norm": total_norm,
            "max_val": max_val,
            "mean_abs": mean_abs_sum / len(keys) if keys else 0,
            "tensor_count": len(keys),
            "shapes": shapes,
            "label": BLOCKS[bid][0],
        }

    def _run_ablation(self, vae, latent, block_key_map, state_dict, abl_strength):
        """Weaken each decoder unit individually and measure decode difference."""
        ablation_results = {}
        samples = latent["samples"]

        # Load VAE to GPU
        try:
            from comfy import model_management
            if hasattr(vae, 'patcher'):
                model_management.load_models_gpu([vae.patcher])
        except Exception:
            pass

        # Baseline decode
        print(f"[Flux VAE Inspector] Baseline decode...")
        try:
            with torch.no_grad():
                baseline = vae.decode(samples).cpu()
        except Exception as e:
            return {"error": f"Baseline decode failed: {e}"}

        # Only ablate decoder units (encoder doesn't affect decode)
        dec_units = [bid for bid in DEC_BLOCK_IDS if bid in block_key_map]
        total = len(dec_units)

        for idx, bid in enumerate(dec_units):
            print(f"[Flux VAE Inspector] Ablating {bid} ({idx + 1}/{total})...")
            keys = block_key_map[bid]

            # Save originals
            originals = {}
            for key in keys:
                tensor = state_dict[key]
                if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    originals[key] = tensor.detach().clone()
                    tensor.mul_(abl_strength)

            if not originals:
                continue

            try:
                vae.first_stage_model.load_state_dict(state_dict, strict=False)
                with torch.no_grad():
                    ablated = vae.decode(samples).cpu()

                diff = (baseline - ablated).abs()
                mean_diff = diff.mean().item()
                max_diff = diff.max().item()

                # Per-channel RGB
                if diff.dim() == 4 and diff.shape[1] >= 3:
                    r_diff = diff[:, 0].mean().item()
                    g_diff = diff[:, 1].mean().item()
                    b_diff = diff[:, 2].mean().item()
                else:
                    r_diff = g_diff = b_diff = mean_diff

                impact_score = min(100, mean_diff * 1000)

                ablation_results[bid] = {
                    "impact_score": impact_score,
                    "mean_diff": mean_diff,
                    "max_diff": max_diff,
                    "r_impact": r_diff,
                    "g_impact": g_diff,
                    "b_impact": b_diff,
                }
                del ablated

            except Exception as e:
                ablation_results[bid] = {"error": str(e), "impact_score": 0}

            # Restore
            for key, orig in originals.items():
                state_dict[key].copy_(orig)
            del originals

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Restore full model
        try:
            vae.first_stage_model.load_state_dict(state_dict, strict=False)
        except Exception:
            pass

        del baseline
        gc.collect()
        return ablation_results

    def _compute_summary(self, results):
        dec_norms = []
        enc_norms = []
        for bid, data in results["blocks"].items():
            norm = data["weight_norm"]
            if bid.startswith("d_"):
                dec_norms.append((bid, norm, data["label"]))
            elif bid.startswith("e_"):
                enc_norms.append((bid, norm, data["label"]))

        dec_norms.sort(key=lambda x: x[1], reverse=True)
        enc_norms.sort(key=lambda x: x[1], reverse=True)

        return {
            "top_decoder": dec_norms[:10],
            "top_encoder": enc_norms[:10],
            "decoder_total_norm": sum(n for _, n, _ in dec_norms),
            "encoder_total_norm": sum(n for _, n, _ in enc_norms),
        }

    def _generate_recommendations(self, results):
        recs = []
        ablation = results.get("ablation", {})

        if ablation and "error" not in ablation:
            sorted_abl = sorted(
                [(bid, d) for bid, d in ablation.items() if isinstance(d, dict) and "impact_score" in d],
                key=lambda x: x[1]["impact_score"],
                reverse=True
            )
            if sorted_abl:
                recs.append("🎯 HIGHEST DECODE IMPACT (top 5):")
                for bid, data in sorted_abl[:5]:
                    score = data["impact_score"]
                    r, g, b = data.get("r_impact", 0), data.get("g_impact", 0), data.get("b_impact", 0)
                    max_ch = max(r, g, b)
                    ch = "R" if r == max_ch else ("G" if g == max_ch else "B")
                    label = BLOCKS.get(bid, (bid,))[0]
                    recs.append(f"  {label}: score={score:.1f} (strongest: {ch} channel)")
                recs.append("")

        recs.append("📋 TUNING GUIDE:")
        recs.append("• d_conv_out — final RGB output (VERY sensitive to changes)")
        recs.append("• d_u0b*_conv* — fine detail & color precision (128ch)")
        recs.append("• d_u1b*_conv* — color gradients (256ch)")
        recs.append("• d_u2b*/d_u3b* — coarse structure & tone (512ch)")
        recs.append("• d_mid_attn_* — global spatial color coherence")
        recs.append("• d_*_norm* — normalization (affects contrast curves)")
        recs.append("• d_norm_out — overall brightness/contrast")
        recs.append("")
        recs.append("⚠️ Encoder units only affect img2img/inpainting.")
        return recs

    def _format_report(self, results):
        lines = [
            "=" * 70,
            "FLUX VAE ANALYSIS — 125 Individual Tensor Units",
            "=" * 70,
        ]
        arch = results.get("architecture", {})
        lines.append(f"Total: {arch.get('total_params', 0) / 1e6:.1f}M params, "
                     f"{arch.get('tensor_count', 0)} tensors, {arch.get('unit_count', 0)} units")
        lines.append(f"  Dec: {arch.get('decoder_params', 0) / 1e6:.1f}M | "
                     f"Enc: {arch.get('encoder_params', 0) / 1e6:.1f}M")
        lines.append(f"  Dtypes: {', '.join(arch.get('dtypes', []))}")
        lines.append("")

        # Ablation
        ablation = results.get("ablation", {})
        if ablation and "error" not in ablation:
            lines.append("-" * 70)
            lines.append("DECODE ABLATION IMPACT")
            lines.append("-" * 70)
            sorted_abl = sorted(
                [(bid, d) for bid, d in ablation.items() if isinstance(d, dict) and "impact_score" in d],
                key=lambda x: x[1]["impact_score"],
                reverse=True
            )
            for bid, data in sorted_abl:
                score = data.get("impact_score", 0)
                bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
                r = data.get("r_impact", 0)
                g = data.get("g_impact", 0)
                b = data.get("b_impact", 0)
                label = BLOCKS.get(bid, (bid,))[0]
                lines.append(f"  {label:<24} [{bar}] {score:5.1f}  "
                             f"R:{r:.4f} G:{g:.4f} B:{b:.4f}")
            lines.append("")

        # Weight norms — decoder
        lines.append("-" * 70)
        lines.append("DECODER TENSOR UNIT NORMS")
        lines.append("-" * 70)
        dec_data = [(bid, results["blocks"][bid]) for bid in DEC_BLOCK_IDS if bid in results["blocks"]]
        if dec_data:
            max_norm = max(d["weight_norm"] for _, d in dec_data)
            for bid, data in dec_data:
                norm = data["weight_norm"]
                blen = int((norm / max_norm) * 25) if max_norm > 0 else 0
                bar = "█" * blen + "░" * (25 - blen)
                label = data["label"]
                lines.append(f"  {label:<24} [{bar}] {norm:8.1f}  "
                             f"({data['param_count']:>8} params)")

        lines.append("")

        # Weight norms — encoder
        lines.append("-" * 70)
        lines.append("ENCODER TENSOR UNIT NORMS")
        lines.append("-" * 70)
        enc_data = [(bid, results["blocks"][bid]) for bid in ENC_BLOCK_IDS if bid in results["blocks"]]
        if enc_data:
            max_norm = max(d["weight_norm"] for _, d in enc_data)
            for bid, data in enc_data:
                norm = data["weight_norm"]
                blen = int((norm / max_norm) * 25) if max_norm > 0 else 0
                bar = "█" * blen + "░" * (25 - blen)
                label = data["label"]
                lines.append(f"  {label:<24} [{bar}] {norm:8.1f}  "
                             f"({data['param_count']:>8} params)")

        lines.append("")
        lines.append("=" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 70)
        for rec in results.get("recommendations", []):
            lines.append(rec)

        return "\n".join(lines)

    def _create_ui_json(self, results):
        ui_data = {
            "type": "vae_inspector_125",
            "blocks": {},
            "ablation": results.get("ablation", {}),
            "recommendations": results.get("recommendations", []),
        }
        for bid, data in results["blocks"].items():
            ui_data["blocks"][bid] = {
                "weight_norm": data["weight_norm"],
                "score": data.get("score", 50.0),
                "param_count": data["param_count"],
                "tensor_count": data["tensor_count"],
                "max_val": data["max_val"],
                "label": data["label"],
            }
        return json.dumps(ui_data)


NODE_CLASS_MAPPINGS = {
    "FluxVAEInspector": FluxVAEInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxVAEInspector": "VAE Inspector (Flux 2 Klein — 125 Tensors)",
}
