# IllustriousMultiPassSampler

Multi-pass sampler optimized for Illustrious XL models. Designed for artists who want more control over structure/detail passes and creative composition.

## What does it do?
- Runs two denoising passes: first for structure/composition, second for fine details.
- Lets you tune steps, CFG, denoise, samplers, and schedulers for each pass.
- Supports adaptive tweaks for composition stability and detail sharpness.
- Version-aware: auto-detects EPS/VPred models and applies best presets.

## Inputs
- **model**: Your UNet diffusion model (SDXL/Illustrious).
- **seed**: Random seed for reproducibility.
- **positive/negative**: Conditioning prompts.
- **latent_image**: Starting latent tensor.
- **structure/detail steps/cfg/denoise**: Control how much each pass shapes the image.
- **sampler/scheduler**: Choose your favorite sampler for each pass.
- **adaptive_transition**: Keeps composition stable if denoise is high.
- **preserve_composition**: Ensures base layout stays consistent.
- **reuse_noise**: Reuses noise for tighter coupling (advanced).
- **model_version**: Explicitly set model version or leave on auto.
- **Version Preset Override**: Force EPS/VPred presets if needed.

## Outputs
- **structure_latent**: Latent after structure pass.
- **final_latent**: Latent after detail pass (ready for decode).

## Artist Tips
- Use higher structure denoise for loose, creative compositions; lower for tight control.
- Try different samplers for each pass (e.g., Euler for structure, DPM for detail).
- Adaptive transition is great for keeping the subject anchored when experimenting.
- "Auto" model version works best if your model is properly tagged; override if you know your model type.
- Combine with Smart Scene Generator for rich, layered prompts.

## Example Workflow
1. Generate a latent with Empty Latent node.
2. Use Smart Scene Generator for a creative prompt.
3. Run MultiPassSampler with adaptive settings for a painterly look.
4. Decode and color correct as desired.

---
For more advanced tricks, see the [IllustriousTriplePassSampler](IllustriousTriplePassSampler.md) for three-stage sampling.
