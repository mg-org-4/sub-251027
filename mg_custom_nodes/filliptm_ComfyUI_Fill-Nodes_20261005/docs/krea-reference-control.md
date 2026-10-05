# Krea reference control

Use **FL Krea Reference** for each image and connect its output to **FL Krea Reference Guider**. Connect the guider to **SamplerCustomAdvanced**, with standard noise, sampler, scheduler, and latent nodes. The example uses Krea 2 Turbo, eight steps, Euler/simple, and a fixed seed.

For **KSampler** or **KSampler Advanced**, connect the guider node's **MODEL** output to `model` and **CONDITIONING** output to `positive`. Both connections are required to preserve independent reference weights and timing. Connect an ordinary empty CLIP Text Encode to `negative`. Use CFG **1** to match the GUIDER output; higher CFG applies normal negative guidance after the reference blend. The original GUIDER output remains available. This changes sampler compatibility, not the reference blending method or its potential for overlapping subjects.

The guider encodes each reference separately, preserves all twelve Krea conditioning layers, and blends model predictions during sampling. It does not place the reference images into a shared picture grid.

## Controls

- **Role:** select what to borrow. Style targets medium and texture; palette targets colors; subject targets appearance; composition targets arrangement. Custom uses only the main scene prompt. These are model instructions, not guaranteed identity or geometry controls.
- **Weight:** direct reference strength. 0.05 applies 5% of that reference?s contribution before overall influence; 1 applies the full contribution. Zero or disabled excludes it. Weights are not normalized against other references.
- **Influence:** overall reference strength after blending. Zero gives the plain scene prompt. Start at 0.7 and compare using the same seed.
- **Blend mode / average amount:** Add preserves the additive blend. In Average mode, amount 0 is additive and amount 1 divides contributions by the number of enabled references; intermediate amounts blend those results. Four enabled references at weight 1 contribute 25% each at full averaging. Individual weights still apply. Disabled references are excluded from the count; zero-weight or faded references retain their share, which returns to the scene prompt. Existing workflows default to Add.
- **Resolution:** reference longest-side limit. Start at 512; use 1024 for fine detail. Larger settings increase encoding cost. Images are not enlarged except to meet the minimum vision patch dimensions.
- **Reference mode:** Context (default) lets Qwen see the image, then supplies only image-informed text states to Krea. This reduces direct copying of the source layout. Full supplies visual tokens as well, for stronger source resemblance. Both preserve all twelve conditioning layers.
- **Start / end / fade:** advanced timing controls in full diffusion progress. Start must be below end. Fade is a fraction of the window at each edge, up to 0.5. Fade interpolation occurs between converted sigma boundaries. Shortened denoise runs use the corresponding part of this full schedule.

Reference sockets grow as you connect them, up to ComfyUI's 100-entry Autogrow ceiling. Each reference accepts one RGB image; select a frame first when the source is an image batch.

For a dog rendered like an ink illustration, describe the scene in the main prompt and select a style reference. A second palette reference can supply its colors independently.

## Cost and behavior

Each active reference normally adds one model evaluation per sampling evaluation. Two references plus the text baseline require three evaluations. The baseline evaluation is skipped when the combined active reference contribution is exactly 1. Timed-out branches are skipped. This is slower than encoding all pictures together; it makes their numerical contributions independent.

Weights directly scale each reference?s difference from the scene baseline. Fading one reference returns its contribution to the scene baseline without boosting the other references. The blend is:

`baseline + influence * blend_scale * sum(weight * envelope * (reference_prediction - baseline))`

`blend_scale` is 1 in Add mode. In Average mode it is `1 - average_amount + average_amount / N`, where `N` is the number of enabled references. No enabled references uses the baseline. Both GUIDER and MODEL/CONDITIONING outputs use this blend.

When active weights multiplied by influence sum above 1, the guider extrapolates beyond the baseline instead of silently normalizing or clipping the sliders. Lower overall influence or the reference weights for a gentler result.

The visual result is nonlinear. A larger reference weight does not guarantee proportionally stronger style. References containing panels can still suggest panels, especially in Full mode, and unrelated subject references can compete. Compare a reference alone before mixing it with others. Context can weaken source identity and geometry; use Full when those matter more than avoiding copied layouts.

The nodes use normal ComfyUI execution caching, with no private image/embedding cache. Changes to reference settings can re-encode references. Changing same-size image contents or the text encoder is not hidden by a shape-only cache.

Use a Krea 2 model and CLIPLoader type `krea2`. The guider uses CFG 1 and rejects custom CFG/pre-CFG/post-CFG/batch-guidance functions. Ordinary model patches and optimized attention remain managed by ComfyUI. Other guidance or cross-call caching patches have not been validated with independent reference branches.

This version does not supply reference latents, destination masks, identity training, or automatic captioning. Use existing image nodes to crop a source reference.
