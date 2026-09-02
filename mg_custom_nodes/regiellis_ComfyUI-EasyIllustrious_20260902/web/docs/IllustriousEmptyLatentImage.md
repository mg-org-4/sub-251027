# IllustriousEmptyLatentImage

Creates your blank canvas for the AI to paint on. Pick your size, and you're ready to create!

## What it does

- Sets up a blank starting point at your chosen size
- Like choosing canvas size before painting
- Optimized for anime and illustration ratios
- Smart detection for best quality with your model

## Inputs

- **resolution**: Canvas size and shape
  - **Square (1:1)**: 1024x1024 - Instagram, profile pics
  - **Portrait (2:3)**: 832x1216 - Character art, posters  
  - **Landscape (3:2)**: 1216x832 - Scenes, wallpapers
  - **Widescreen (16:9)**: 1344x768 - Cinematic, banners
  - **Tall (9:16)**: 768x1344 - Phone wallpapers, Pinterest
  
- **batch_size**: How many images to create at once (1-8)
- **model**: (Optional) Connect for smart optimization
- **seed**: Random number for variations

## Outputs

- **LATENT**: Your blank canvas, ready for the sampler
- **info**: Helpful tips about your chosen size

## Why Size Matters

Different sizes work better for different art:
- **Square**: Balanced, good for everything
- **Portrait**: Natural for characters, people
- **Landscape**: Perfect for environments, scenes
- **Widescreen**: Cinematic feel, dramatic scenes
- **Tall**: Mobile-friendly, full body shots

## Resolution Sweet Spots

**For Best Quality:**
- Illustrious v0.1: 1024x1024 base
- Pony models: 832x1216 or 1216x832
- SDXL base: 1024x1024

**For Speed:**
- Start at 768x768 or 832x832
- Upscale later for details

**For Memory:**
- 6GB VRAM: Max 1024x1024
- 8GB VRAM: Max 1216x832
- 12GB+ VRAM: Any size

## Batch Size Tips

- **1 image**: Testing, perfecting single piece
- **2-4 images**: Variations, choosing best
- **5-8 images**: Exploration, finding style
- More images = more VRAM needed

## Common Resolutions Explained

**Instagram/Social:**
- Square: 1024x1024
- Portrait: 832x1216 (4:5 ratio)

**Desktop Wallpaper:**
- 1920x1080 → Start at 1344x768, upscale
- 2560x1440 → Start at 1216x832, upscale

**Print/Posters:**
- 11x17 → 1216x832 then upscale 2x
- 18x24 → 832x1216 then upscale 2x

## Noise Patterns (Advanced)

Different starting noise creates different feels:
- **Gaussian**: Standard, balanced (default)
- **Uniform**: Smoother, less contrast
- **Brownian**: More natural, organic
- Keep default unless experimenting

## Pro Tips

- Wider images → Better for landscapes
- Taller images → Better for full body
- Square → Most flexible, works for anything
- Batch of 4 → Perfect for testing variations
- Non-standard sizes may cause issues

## Memory Management

**Getting "Out of Memory"?**
1. Reduce resolution
2. Lower batch size
3. Use 832x832 instead of 1024x1024
4. Upscale later instead

## Example Workflows

**Single Perfect Image:**
1. Resolution: 1024x1024
2. Batch: 1
3. Generate → Perfect → Done

**Finding Best Variation:**
1. Resolution: 832x1216
2. Batch: 4
3. Generate → Pick best → Refine

**Memory-Friendly Hi-Res:**
1. Start: 768x768, Batch: 1
2. Generate base image
3. Upscale to 1536x1536
4. Refine with low denoise

**Production Batch:**
1. Resolution: Your target
2. Batch: Maximum your GPU allows
3. Generate many → Cherry pick best

## Quick Reference

| Purpose | Resolution | Why |
|---------|------------|-----|
| Testing | 768x768 | Fast, low memory |
| Characters | 832x1216 | Natural proportions |
| Landscapes | 1216x832 | Wide view |
| Social Media | 1024x1024 | Universal format |
| Wallpaper | 1344x768 | 16:9 widescreen |

---
For upscaling: [IllustriousLatentUpscale](IllustriousLatentUpscale.md)
For sampling: [IllustriousKSamplerPro](IllustriousKSamplerPro.md)