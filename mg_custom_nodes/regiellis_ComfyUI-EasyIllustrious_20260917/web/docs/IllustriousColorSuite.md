# IllustriousColorSuite

Unified color correction node for Illustrious workflows. Combines auto color correction and smart cache correction for easy, artist-friendly enhancement.

## What does it do?

- Automatically fixes color, saturation, and contrast issues in anime/illustration images.
- Offers both simple and advanced controls (cache, presets, detail enhancement, etc).
- Returns corrected images, a report, and whether cache was used.

## Inputs

- **images**: Input images to correct (NHWC, 0..1).
- **model_version**: Pick your Illustrious model version or leave on auto.
- **correction_strength**: Overall strength multiplier.
- **auto_detect_issues**: Analyze image to auto-tune corrections.
- **fix_oversaturation**: Reduce excessive saturation.
- **preserve_anime_aesthetic**: Keep anime look while correcting.
- **enhance_details**: Subtle detail enhancement.
- **balance_colors**: Neutralize color casts.
- **adjust_contrast**: Gentle brightness/contrast tweak.
- **custom_preset**: Choose a preset for quick results.
- **cache_mode/adjustment_mode/force_recalculate**: Advanced cache controls for real-time tweaks.

## Outputs

- **corrected_images**: The enhanced images.
- **report**: A summary of what was fixed.
- **cache_used**: Whether the cache was used for this correction.

## Artist Tips

- For most images, just drop them in and let auto mode do the work.
- Use cache for fast, real-time tweaks during workflow exploration.
- Try different presets for different art styles or models.
- Combine with samplers and outpaint for a full creative pipeline.

## Example Workflow

1. Generate or load your image.
2. Run through IllustriousColorSuite for instant enhancement.
3. Continue with further nodes (sampler, outpaint, etc).

---
For more advanced color workflows, see [IllustriousSmartCacheCorrector](IllustriousSmartCacheCorrector.md) or [IllustriousColorCorrector](IllustriousColorCorrector.md).
