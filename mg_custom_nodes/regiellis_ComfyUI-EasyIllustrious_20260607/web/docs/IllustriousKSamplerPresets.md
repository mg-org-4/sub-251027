# IllustriousKSamplerPresets

One-click quality presets. Pick your style - anime, realistic, or artistic - and go!

## What does it do?

- Provides tested preset combinations for instant results
- Each preset includes sampler, scheduler, steps, and CFG settings
- Optimized specifically for different art styles
- No guessing - just pick and create

## Inputs

- **preset_name**: Choose your style
  - **Anime Clean**: Crisp lines, vibrant colors
  - **Anime Soft**: Gentle, dreamy atmosphere  
  - **Realistic**: Photo-like quality
  - **Artistic**: Painterly, creative
  - **Fast Draft**: Quick previews
  - **Ultra Quality**: Maximum detail (slow)
  
- **model**: Your loaded model
- **positive/negative**: Your prompts
- **latent_image**: Starting canvas
- **seed**: For reproducibility
- **denoise**: How much to change (1.0 = full)

## Outputs

- **LATENT**: Ready for VAE decode
- **settings_used**: What settings were applied

## Artist Tips

- Start with presets to find your style
- Note which presets you prefer for different subjects
- Use "Fast Draft" for testing compositions
- Switch to "Ultra Quality" for final renders
- Each preset can be a starting point - adjust as needed

## Preset Breakdown

**Anime Clean** (Most popular):
- Sampler: DPM++ 2M Karras
- Steps: 20, CFG: 7
- Perfect for: Characters, clean illustrations

**Anime Soft**:
- Sampler: Euler a
- Steps: 28, CFG: 6
- Perfect for: Romantic, dreamy scenes

**Realistic**:
- Sampler: DPM++ SDE Karras  
- Steps: 30, CFG: 7
- Perfect for: Portraits, photo-style

**Artistic**:
- Sampler: DPM++ 2S a
- Steps: 25, CFG: 8
- Perfect for: Creative, painterly styles

**Fast Draft**:
- Sampler: Euler
- Steps: 10, CFG: 7
- Perfect for: Quick tests, iterations

**Ultra Quality**:
- Sampler: DPM++ 3M SDE
- Steps: 50, CFG: 7
- Perfect for: Final renders, prints

## When to Use

- **Starting out**: Use presets exclusively
- **Learning**: Try all presets with same prompt
- **Production**: Use your favorite preset consistently
- **Experimentation**: Start with preset, then tweak

## Workflow Examples

**Finding Your Style:**
1. Same prompt + seed
2. Try each preset
3. Pick your favorite
4. Note it for future use

**Quick to Final:**
1. Fast Draft for composition
2. Adjust prompt if needed
3. Ultra Quality for final
4. Perfect workflow efficiency

**Batch Generation:**
1. Pick best preset for your style
2. Generate multiple seeds
3. Consistent quality across all

## Pro Tips

- Anime models → Anime presets
- Realistic models → Realistic preset
- Mixed models → Try Artistic
- Low VRAM → Avoid Ultra Quality
- Each preset tested on thousands of images

---
For fine control: [IllustriousKSamplerPro](IllustriousKSamplerPro.md)
For multi-pass: [IllustriousMultiPassSampler](IllustriousMultiPassSampler.md)
