# IllustriousAttentionCouple

Spatial attention coupling for Illustrious models (v0.1–v2.0) with dynamic block discovery.

## What does it do?

- Dynamically finds and scales cross-attention modules in the UNet.
- Lets you apply a spatial mask to control attention per position (e.g., focus on a region).
- Safe: dtype/device aligned, NaN/Inf guarded, idempotent patching/unpatching.

## Inputs

- **model**: The UNet model to patch.
- **positive/negative**: Conditioning prompts.
- **mask**: (Optional) Spatial mask (NHWC 0..1). White = more attention, black = unchanged.
- **latent_image**: (Optional) For precise mask alignment.
- **affect_down/mid/up**: Which UNet stages to affect.
- **strength**: Blend strength per position (0 = no change, 1 = full masked coupling).
- **token_filter**: (Reserved for future use.)
- **advanced_logs**: Enable for debug info.

## Outputs

- (Patched model, conditioning, etc. as per node design.)

## Artist Tips

- Use a mask to focus attention on a character, face, or region.
- Try different strengths for subtle or strong effects.
- Combine with samplers for region-specific style or detail.
- Use with latent_image for perfect mask-to-latent alignment.

## Example Workflow

1. Prepare a mask image (white = focus, black = ignore).
2. Connect your model and prompts.
3. Use AttentionCouple to direct attention where you want it.

---
For more advanced prompt control, see Smart Scene Generator and Prompt nodes.
