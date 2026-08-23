# IllustriousLatentUpscale

Makes your image canvas bigger while keeping it in the AI's working format. Essential for creating high-resolution artwork without starting over.

## What does it do?

- Enlarges your working canvas (latent) before the final image is created
- Maintains the AI-friendly format so you can add more detail
- Much more efficient than upscaling after the image is done
- Preserves your composition while allowing for enhanced details

## Inputs

- **samples**: Your current latent/canvas from previous nodes
- **upscale_method**: How to resize (nearest, bilinear, area, bicubic)
- **scale_by**: Multiplication factor (2.0 = double size, 1.5 = 50% larger)
- **width/height**: (Alternative) Set exact dimensions instead of scaling

## Outputs

- **LATENT**: Larger canvas ready for detail enhancement

## Artist Tips

- Upscale BEFORE final sampling for best quality
- Use 1.5-2x for most cases (higher can lose coherence)
- After upscaling, sample with low denoise (0.3-0.5) to add details
- Different methods work better for different styles:
  - **bilinear**: Smooth, good for paintings
  - **nearest**: Sharp, good for pixel-perfect styles
  - **bicubic**: Balanced, good general choice

## When to Use

- **High-res from low-res**: Generate at 512 → Upscale 2x → Refine at 1024
- **Detail enhancement**: Upscale → Low denoise sample → Crispy details
- **Memory saving**: Start small, upscale, rather than starting huge
- **Wallpaper/prints**: Progressive upscaling for massive resolutions

## Common Workflows

**Standard Hi-Res Fix:**
1. Generate at 832x1216 → LatentUpscale 1.5x →
2. KSampler (denoise 0.4) → VAEDecode →
3. Final image at 1248x1824 with enhanced details

**Progressive Upscaling:**
1. Start 512x768 → Sample →
2. LatentUpscale 1.5x → Sample (denoise 0.5) →
3. LatentUpscale 1.5x → Sample (denoise 0.3) →
4. Ultra high-res with maintained coherence

**Quick 2K/4K:**
1. Generate at optimal size → LatentUpscale 2x →
2. KSampler (denoise 0.35) → Instant high-res

## Settings Guide

- **Scale 1.5x + Denoise 0.4-0.5**: Safe, reliable enhancement
- **Scale 2x + Denoise 0.3-0.4**: Maximum size, some detail refinement
- **Scale 2x+ + Denoise 0.2-0.3**: Risky but can work for some styles
- **Never**: Scale >3x in one step (do progressive instead)

---
For initial canvas creation, see [IllustriousEmptyLatentImage](IllustriousEmptyLatentImage.md)
For converting to viewable image, see [IllustriousVAEDecode](IllustriousVAEDecode.md)
