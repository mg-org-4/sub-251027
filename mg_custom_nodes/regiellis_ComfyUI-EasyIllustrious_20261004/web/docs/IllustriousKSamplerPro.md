# IllustriousKSamplerPro

Production-ready sampler for Illustrious/SDXL models. Designed for artists who want fast, reliable, and color-safe results.

## What does it do?

- Delegates denoising to ComfyUI's core sampler for live previews and interrupts.
- Defaults are color-safe and contrast-true.
- Lets you tweak seed, steps, CFG, sampler, scheduler, and more.
- Version-aware: auto-detects model type and applies best presets.

## Inputs

- **model**: Your UNet diffusion model (SDXL/Illustrious).
- **seed**: Random seed for reproducibility.
- **positive/negative**: Conditioning prompts.
- **latent_image**: Starting latent tensor.
- **steps/cfg/sampler/scheduler/denoise**: Main sampling controls.
- **Model Version**: Explicitly set or auto-detect.
- **Resolution Adaptive/Auto CFG/Color Safe Mode**: Advanced options for power users.

## Outputs

- **LATENT**: The sampled latent, ready for decode.
- **model/positive/negative**: For chaining in advanced graphs.

## Artist Tips

- For most workflows, just set your prompt and seed, and let the defaults handle the rest.
- Use Color Safe Mode for vibrant, artifact-free anime colors.
- Try different samplers for subtle style changes.
- Combine with MultiPass or TriplePass for more control.

## Example Workflow

1. Generate a latent with Empty Latent node.
2. Use KSamplerPro with your favorite prompt and settings.
3. Decode and color correct as needed.

---
For multi-stage sampling, see [IllustriousMultiPassSampler](IllustriousMultiPassSampler.md).
