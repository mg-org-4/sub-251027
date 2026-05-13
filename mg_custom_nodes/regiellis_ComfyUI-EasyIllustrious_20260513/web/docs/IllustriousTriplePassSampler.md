# IllustriousTriplePassSampler

A three-stage sampler for artists who want maximum control: composition, structure, and detail passes, each with their own settings.

## What does it do?

- Runs three denoising passes: composition (global layout), structure (shapes), and detail (edges/textures).
- Lets you tune steps, CFG, denoise, samplers, and schedulers for each pass.
- Supports adaptive step/CFG/denoise tweaks for sharpness and stability.
- Version-aware: auto-detects EPS/VPred models and applies best presets.

## Inputs

- **model**: Your UNet diffusion model (SDXL/Illustrious).
- **seed**: Random seed for reproducibility.
- **positive/negative**: Conditioning prompts.
- **latent_image**: Starting latent tensor.
- **comp/struct/detail steps/cfg/denoise**: Control each pass.
- **sampler/scheduler**: Choose your favorite sampler for each pass.
- **progressive_cfg_decay**: Gradually reduce CFG for stability.
- **adaptive_steps/refinement**: Auto-tune steps and sharpness.
- **preserve_composition**: Keeps the initial layout stable.
- **model_version**: Explicitly set model version or leave on auto.
- **Version Preset Override**: Force EPS/VPred presets if needed.

## Outputs

- **composition_latent**: After composition pass.
- **structure_latent**: After structure pass.
- **final_latent**: After detail pass (ready for decode).

## Artist Tips

- Use high comp denoise for wild layouts, then rein in with structure/detail.
- Try different samplers for each pass to experiment with style.
- Adaptive steps/refinement can help with sharpness in complex scenes.
- Combine with Smart Scene Generator for rich, multi-layered prompts.
- Use with Empty Latent for blank-canvas workflows or with outpaint for expansion.

## Example Workflow

1. Generate a latent with Empty Latent node.
2. Use Smart Scene Generator for a creative prompt.
3. Run TriplePassSampler with adaptive settings for cinematic results.
4. Decode and color correct as desired.

---
For simpler workflows, see [IllustriousMultiPassSampler](IllustriousMultiPassSampler.md).
