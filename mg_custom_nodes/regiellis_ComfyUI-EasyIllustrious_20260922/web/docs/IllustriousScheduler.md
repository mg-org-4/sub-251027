# IllustriousScheduler

Controls how your image develops from noise to final art. Think of it as the "cooking temperature" - different schedules create different artistic styles.

## What does it do?

- Determines how quickly the AI refines your image at each step
- Different schedules produce different artistic qualities
- Some preserve fine details, others enhance creativity
- Content-aware options adapt to your specific image

## Inputs

- **scheduler_type**: The noise reduction pattern
  - **normal**: Standard, balanced progression
  - **karras**: Smoother, better for high-resolution
  - **exponential**: Fast start, careful finish
  - **sgm_uniform**: Even steps throughout
  - **simple**: Linear, predictable results
  - **ddim_uniform**: Good for animations
  - **beta**: Experimental, unique results

- **steps**: How many refinement iterations (10-50 typical)
- **denoise**: How much to change (1.0 = from scratch, 0.5 = half change)

## Outputs

- **sigmas**: The schedule data for your sampler

## Artist Tips

- **Karras**: Best all-rounder, especially for detailed work
- **Normal**: Classic choice, reliable results
- **Exponential**: Good for artistic/painterly styles
- More steps = smoother but slower (20-30 is usually perfect)
- Different schedulers can dramatically change the art style!

## When to Use Which

**For Anime/Illustrations:**
- karras or normal: Clean lines, consistent colors

**For Painterly/Artistic:**
- exponential or beta: More texture, artistic interpretation

**For High-Resolution:**
- karras: Handles fine details best

**For Quick Tests:**
- simple or ddim_uniform: Fast, predictable

**For Animations:**
- ddim_uniform: Frame consistency

## Common Settings

- **Standard Quality**: karras, 20-30 steps
- **High Quality**: karras, 30-50 steps  
- **Quick Preview**: simple, 10-15 steps
- **Artistic**: exponential, 25-35 steps

## Pro Combinations

**Crisp Anime:**
- Scheduler: karras
- Sampler: DPM++ 2M or Euler a
- Steps: 25

**Soft Painting:**
- Scheduler: exponential  
- Sampler: DPM++ SDE
- Steps: 30

**Fast Testing:**
- Scheduler: ddim_uniform
- Sampler: Euler
- Steps: 15

## Example Workflows

**Finding Your Style:**
1. Same prompt + seed
2. Try different schedulers
3. Compare artistic differences

**Quality vs Speed:**
1. Test with simple (10 steps) for composition
2. Final with karras (30 steps) for quality

---
For sampling, see [IllustriousKSamplerPro](IllustriousKSamplerPro.md)
For multi-pass control, see [IllustriousMultiPassSampler](IllustriousMultiPassSampler.md)
