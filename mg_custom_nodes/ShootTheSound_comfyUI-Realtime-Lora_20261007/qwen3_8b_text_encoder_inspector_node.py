"""
ComfyUI Qwen3-8B Text Encoder Inspector — Layer Analysis Tool

For Flux 2 Klein 9B's Qwen3-8B text encoder.
Analyzes which layers and sub-components contribute most to prompt encoding.
"""

import re
import os
import json
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Tuple

NUM_LAYERS = 36
LAYER_SUBS = ["input_norm", "attn", "attn_norm", "mlp", "post_norm"]


class ActivationCapture:
    """Context manager to capture activations from specific layers."""
    
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def capture_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if out is not None and hasattr(out, 'detach'):
                self.activations[name] = out.detach().cpu()
        return hook
    
    def register(self, module, name):
        hook = module.register_forward_hook(self.capture_hook(name))
        self.hooks.append(hook)
    
    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}


def compute_activation_stats(activation: torch.Tensor) -> Dict:
    """Compute statistics for an activation tensor."""
    if activation is None:
        return {"magnitude": 0, "variance": 0, "sparsity": 0, "max": 0}
    
    act = activation.float()
    
    if act.dim() == 2:
        act = act.unsqueeze(0)
    
    magnitude = act.abs().mean().item()
    variance = act.var().item()
    sparsity = (act.abs() < 0.01).float().mean().item()
    max_val = act.abs().max().item()
    l2_norm = act.norm(dim=-1).mean().item()
    
    return {
        "magnitude": magnitude,
        "variance": variance,
        "sparsity": sparsity,
        "max": max_val,
        "l2_norm": l2_norm,
    }


class Qwen3_8BTextEncoderInspector:
    """
    Analyzes text encoder layers to understand their contribution to prompt encoding.
    For Flux 2 Klein 9B's Qwen3-8B.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "CLIP/Text Encoder to analyze (Qwen3-8B)"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "a photo of a cat sitting on a table",
                    "tooltip": "Test prompt to analyze"
                }),
                "analysis_mode": (["quick", "ablation_HEAVY"], {
                    "default": "quick",
                    "tooltip": "quick = activation stats (fast, low VRAM). ablation_HEAVY = layer importance via weakening (SLOW, 3x VRAM!)"
                }),
                "ablation_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Only used in ablation mode. How much to weaken layers (0.5 = 50%)"
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
    DESCRIPTION = """Analyzes Qwen3-8B text encoder layers (Flux 2 Klein).

⚡ QUICK MODE (default): Fast, low VRAM
  - Captures activation magnitudes per layer
  - Shows which layers respond strongest to your prompt

⚠️ ABLATION MODE: Slow, HIGH VRAM (3x model size!)
  - Tests what happens when each region is weakened
  - Gives precise importance scores

Use QUICK mode first, then ABLATION if needed."""

    def inspect(self, clip, prompt, analysis_mode="quick", ablation_strength=0.5):
        print(f"[Qwen3-8B TE Inspector] Starting analysis...")
        print(f"[Qwen3-8B TE Inspector] Prompt: {prompt[:50]}...")
        
        results = {
            "prompt": prompt,
            "mode": analysis_mode,
            "layers": {},
            "summary": {},
            "recommendations": [],
        }
        
        cond_stage_model = clip.cond_stage_model
        tokens_data = clip.tokenize(prompt)
        
        token_ids = []
        for key in tokens_data:
            if hasattr(tokens_data[key], '__iter__'):
                for batch in tokens_data[key]:
                    if hasattr(batch, '__iter__'):
                        for item in batch:
                            if isinstance(item, (list, tuple)) and len(item) >= 1:
                                token_ids.append(item[0])
                            elif isinstance(item, int):
                                token_ids.append(item)
                        break
                break
        
        print(f"[Qwen3-8B TE Inspector] Token count: {len(token_ids)}")
        
        state_dict = cond_stage_model.state_dict()
        layer_info = self._analyze_architecture(state_dict)
        
        capturer = ActivationCapture()
        
        search_target = cond_stage_model
        for attr in ['qwen3_8b', 'qwen3_4b', 'transformer', 'model', 'text_model', 'encoder']:
            if hasattr(search_target, attr):
                search_target = getattr(search_target, attr)
        
        layers_module = None
        if hasattr(search_target, 'layers'):
            layers_module = search_target.layers
        elif hasattr(search_target, 'model') and hasattr(search_target.model, 'layers'):
            layers_module = search_target.model.layers
        
        if layers_module is not None:
            for i, layer in enumerate(layers_module):
                if i >= NUM_LAYERS:
                    break
                capturer.register(layer, f"layer_{i}")
                
                if hasattr(layer, 'self_attn'):
                    capturer.register(layer.self_attn, f"layer_{i}_attn")
                if hasattr(layer, 'mlp'):
                    capturer.register(layer.mlp, f"layer_{i}_mlp")
                if hasattr(layer, 'input_layernorm'):
                    capturer.register(layer.input_layernorm, f"layer_{i}_input_norm")
                if hasattr(layer, 'post_attention_layernorm'):
                    capturer.register(layer.post_attention_layernorm, f"layer_{i}_post_norm")
        
        try:
            clip.cond_stage_model.reset_clip_options()
            
            from comfy import model_management
            model_management.load_models_gpu([clip.patcher])
            
            with torch.no_grad():
                cond = clip.encode_from_tokens(tokens_data, return_pooled=True)
        except Exception as e:
            print(f"[Qwen3-8B TE Inspector] Forward pass error: {e}")
            capturer.clear()
            return self._error_report(str(e))
        
        print(f"[Qwen3-8B TE Inspector] Captured {len(capturer.activations)} activation points")
        
        layer_stats = {}
        for name, act in capturer.activations.items():
            stats = compute_activation_stats(act)
            layer_stats[name] = stats
        
        capturer.clear()
        
        for i in range(NUM_LAYERS):
            layer_key = f"layer_{i}"
            layer_data = {
                "index": i,
                "total": layer_stats.get(layer_key, {}),
                "attn": layer_stats.get(f"{layer_key}_attn", {}),
                "mlp": layer_stats.get(f"{layer_key}_mlp", {}),
                "input_norm": layer_stats.get(f"{layer_key}_input_norm", {}),
                "post_norm": layer_stats.get(f"{layer_key}_post_norm", {}),
            }
            results["layers"][f"L{i}"] = layer_data
        
        results["architecture"] = layer_info
        
        if analysis_mode == "ablation_HEAVY":
            print(f"[Qwen3-8B TE Inspector] ⚠️ Running ablation analysis (HEAVY - 3x VRAM)...")
            ablation_results = self._run_ablation(clip, tokens_data, ablation_strength)
            results["ablation"] = ablation_results
        else:
            print(f"[Qwen3-8B TE Inspector] Quick mode - skipping ablation")
        
        results["summary"] = self._compute_summary(results)
        results["recommendations"] = self._generate_recommendations(results)
        
        report = self._format_report(results)
        json_data = json.dumps(results, indent=2, default=str)
        analysis_json = self._create_ui_json(results)
        
        print(f"[Qwen3-8B TE Inspector] Analysis complete.")
        return {"ui": {"analysis_json": [analysis_json]}, "result": (report, json_data)}
    
    def _analyze_architecture(self, state_dict) -> Dict:
        info = {
            "total_params": 0,
            "layer_count": 0,
            "hidden_size": 0,
            "quantized_count": 0,
        }
        
        layer_indices = set()
        for key in state_dict.keys():
            info["total_params"] += state_dict[key].numel()
            
            if 'comfy_quant' in key or 'weight_scale' in key:
                info["quantized_count"] += 1
            
            match = re.search(r'layers\.(\d+)\.', key)
            if match:
                layer_indices.add(int(match.group(1)))
            
            if 'embed_tokens' in key or 'q_proj' in key:
                shape = state_dict[key].shape
                if len(shape) >= 2:
                    info["hidden_size"] = max(info["hidden_size"], min(shape))
        
        info["layer_count"] = len(layer_indices)
        return info
    
    def _run_ablation(self, clip, tokens_data, strength: float) -> Dict:
        import gc
        ablation_results = {}
        
        try:
            with torch.no_grad():
                baseline_cond, baseline_pooled = clip.encode_from_tokens(tokens_data, return_pooled=True)
                baseline_norm = baseline_cond.float().norm().item()
                baseline_cond_cpu = baseline_cond.cpu()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            layer_groups = {
                "early (0-11)": list(range(0, 12)),
                "middle (12-23)": list(range(12, 24)),
                "late (24-35)": list(range(24, 36)),
            }
            
            for group_name, layer_indices in layer_groups.items():
                print(f"[Qwen3-8B TE Inspector] Ablating {group_name}...")
                
                state_dict = clip.cond_stage_model.state_dict()
                patches = {}
                
                for key in state_dict.keys():
                    if 'comfy_quant' in key or 'weight_scale' in key:
                        continue
                    tensor = state_dict[key]
                    if tensor.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
                        continue
                    
                    for idx in layer_indices:
                        if f'layers.{idx}.' in key:
                            diff = tensor.detach().cpu() * (strength - 1.0)
                            patches[key] = (diff,)
                            break
                
                if patches:
                    test_clip = clip.clone()
                    test_clip.add_patches(patches, strength_patch=1.0)
                    
                    with torch.no_grad():
                        test_cond, test_pooled = test_clip.encode_from_tokens(tokens_data, return_pooled=True)
                        test_cond_cpu = test_cond.cpu()
                    
                    diff_norm = (baseline_cond_cpu - test_cond_cpu).float().norm().item()
                    relative_change = diff_norm / (baseline_norm + 1e-8)
                    
                    ablation_results[group_name] = {
                        "relative_change": relative_change,
                        "absolute_change": diff_norm,
                        "impact_score": min(100, relative_change * 100),
                    }
                    
                    print(f"[Qwen3-8B TE Inspector] Ablation {group_name}: {relative_change:.4f} relative change")
                    
                    del test_clip, test_cond, test_cond_cpu, patches
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[Qwen3-8B TE Inspector] Ablation error: {e}")
            ablation_results["error"] = str(e)
        
        return ablation_results
    
    def _compute_summary(self, results: Dict) -> Dict:
        summary = {
            "most_active_layers": [],
            "highest_variance_layers": [],
            "avg_magnitude_by_region": {},
        }
        
        layer_mags = []
        layer_vars = []
        
        for layer_key, data in results.get("layers", {}).items():
            total_stats = data.get("total", {})
            if total_stats:
                mag = total_stats.get("magnitude", 0)
                var = total_stats.get("variance", 0)
                layer_mags.append((layer_key, mag))
                layer_vars.append((layer_key, var))
        
        layer_mags.sort(key=lambda x: x[1], reverse=True)
        layer_vars.sort(key=lambda x: x[1], reverse=True)
        
        summary["most_active_layers"] = layer_mags[:5]
        summary["highest_variance_layers"] = layer_vars[:5]
        
        regions = {
            "early (0-11)": [],
            "middle (12-23)": [],
            "late (24-35)": [],
        }
        
        for layer_key, mag in layer_mags:
            idx = int(layer_key.replace("L", ""))
            if idx < 12:
                regions["early (0-11)"].append(mag)
            elif idx < 24:
                regions["middle (12-23)"].append(mag)
            else:
                regions["late (24-35)"].append(mag)
        
        for region, mags in regions.items():
            if mags:
                summary["avg_magnitude_by_region"][region] = sum(mags) / len(mags)
        
        return summary
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        recommendations = []
        
        summary = results.get("summary", {})
        ablation = results.get("ablation", {})
        
        if ablation and "error" not in ablation:
            max_impact = 0
            max_region = None
            for region, data in ablation.items():
                impact = data.get("impact_score", 0)
                if impact > max_impact:
                    max_impact = impact
                    max_region = region
            
            if max_region:
                recommendations.append(
                    f"🎯 HIGHEST IMPACT REGION: {max_region} (score: {max_impact:.1f})"
                )
                
                if "middle" in max_region:
                    recommendations.append(
                        "→ Middle layers store semantic knowledge. "
                        "KEEP THESE STRONG (1.0) for good prompt adherence."
                    )
                elif "late" in max_region:
                    recommendations.append(
                        "→ Late layers affect style/interpretation. "
                        "Try WEAKENING late attn (0.85-0.95) to reduce creative drift."
                    )
                elif "early" in max_region:
                    recommendations.append(
                        "→ Early layers handle token basics. "
                        "Usually safe to keep at 1.0."
                    )
        else:
            recommendations.append("📊 ACTIVATION-BASED ANALYSIS (run ablation_HEAVY for precise data)")
            recommendations.append("")
            
            by_region = summary.get("avg_magnitude_by_region", {})
            if by_region:
                sorted_regions = sorted(by_region.items(), key=lambda x: x[1], reverse=True)
                top_region = sorted_regions[0][0] if sorted_regions else None
                
                if top_region:
                    recommendations.append(f"🔥 Most active region: {top_region}")
        
        recommendations.append("")
        recommendations.append("📋 GENERAL GUIDELINES FOR PROMPT ADHERENCE:")
        recommendations.append("• Boost embed_tokens (1.5-2.0) for stronger prompt signal")
        recommendations.append("• Keep L12-L24 mlp at 1.0 (concept knowledge)")
        recommendations.append("• Try L28-L35 attn at 0.85-0.95 (reduce interpretation)")
        recommendations.append("• ⚠️ Quantized layers cannot be modified - only float tensors")
        
        return recommendations
    
    def _format_report(self, results: Dict) -> str:
        lines = [
            "=" * 70,
            "QWEN3-8B TEXT ENCODER ANALYSIS REPORT (Flux 2 Klein)",
            "=" * 70,
            f"Prompt: {results['prompt'][:60]}...",
            "",
        ]
        
        arch = results.get("architecture", {})
        lines.append(f"Model: {arch.get('layer_count', '?')} layers, "
                     f"{arch.get('total_params', 0) / 1e9:.2f}B params, "
                     f"hidden={arch.get('hidden_size', '?')}")
        lines.append(f"Quantized tensors: {arch.get('quantized_count', 0)}")
        lines.append("")
        
        ablation = results.get("ablation", {})
        if ablation and "error" not in ablation:
            lines.append("-" * 70)
            lines.append("ABLATION ANALYSIS (Impact when weakened to 50%)")
            lines.append("-" * 70)
            for region, data in sorted(ablation.items(), 
                                        key=lambda x: x[1].get("impact_score", 0), 
                                        reverse=True):
                score = data.get("impact_score", 0)
                bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
                lines.append(f"  {region:<20} [{bar}] {score:.1f}")
            lines.append("")
        
        lines.append("-" * 70)
        lines.append("LAYER ACTIVATION MAGNITUDES")
        lines.append("-" * 70)
        
        for region_name, (start, end) in [("Early (0-11)", (0, 12)), 
                                           ("Middle (12-23)", (12, 24)), 
                                           ("Late (24-35)", (24, 36))]:
            lines.append(f"\n{region_name}:")
            for i in range(start, min(end, 36)):
                layer_key = f"L{i}"
                data = results.get("layers", {}).get(layer_key, {})
                total = data.get("total", {})
                attn = data.get("attn", {})
                mlp = data.get("mlp", {})
                
                mag = total.get("magnitude", 0)
                attn_mag = attn.get("magnitude", 0)
                mlp_mag = mlp.get("magnitude", 0)
                
                bar_len = min(30, int(mag * 100))
                bar = "█" * bar_len + "░" * (30 - bar_len)
                
                lines.append(f"  L{i:02d}: [{bar}] mag={mag:.4f} "
                            f"(attn={attn_mag:.4f}, mlp={mlp_mag:.4f})")
        
        lines.append("")
        lines.append("=" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 70)
        for rec in results.get("recommendations", []):
            lines.append(rec)
        
        return "\n".join(lines)
    
    def _create_ui_json(self, results: Dict) -> str:
        ui_data = {
            "type": "te_inspector_8b",
            "layers": {},
            "ablation": results.get("ablation", {}),
            "recommendations": results.get("recommendations", []),
        }
        
        for layer_key, data in results.get("layers", {}).items():
            total = data.get("total", {})
            ui_data["layers"][layer_key] = {
                "magnitude": total.get("magnitude", 0),
                "variance": total.get("variance", 0),
                "attn_mag": data.get("attn", {}).get("magnitude", 0),
                "mlp_mag": data.get("mlp", {}).get("magnitude", 0),
            }
        
        return json.dumps(ui_data)
    
    def _error_report(self, error_msg: str) -> Tuple[str, str]:
        report = f"ERROR: Analysis failed\n{error_msg}"
        json_data = json.dumps({"error": error_msg})
        return (report, json_data)


NODE_CLASS_MAPPINGS = {
    "Qwen3_8BTextEncoderInspector": Qwen3_8BTextEncoderInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3_8BTextEncoderInspector": "Text Encoder Inspector (Qwen3-8B / Flux 2 Klein)",
}
