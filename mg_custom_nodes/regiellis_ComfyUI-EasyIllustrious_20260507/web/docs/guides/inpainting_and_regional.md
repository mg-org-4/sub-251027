# Inpainting vs Regional Prompting (Artist Guide)

This short guide explains what each feature does, when to use it, and how to set it up in ComfyUI with the Easy-Illustrious nodes.

## Inpainting

- What it is: Constrain changes to a mask. Optionally pre-fill the masked area from a reference image.
- When to use: Fix/replace part of an image (hands, face, background section) while keeping the rest untouched.
- How to use:
  1. Drop "VAE Encode (Illustrious)" and set mode = inpaint.
  2. Connect your mask (black = protect, white = change). Use generous feathering for soft, seam-free edges.
  3. Optional: connect a reference image to guide the masked area.
  4. Sample as usual. Only the masked area will be denoised.

## Regional Prompting

- What it is: Different prompts in different areas using masks, each with its own weight and time range.
- When to use: You want distinct subjects/styles per area (e.g., skylines vs. foreground flowers) in one image.
- How to use:
  1. Encode your base prompt (positive/negative) as usual.
  1. Add "Regional Conditioning (Illustrious)".
  1. Connect: CLIP, base CONDITIONING, and the LATENT (for sizing masks).
  1. Create regions using nodes:
  - "Empty Regions (Illustrious)" → one or more "Make Region (Illustrious)" → chain with "Append Region (Illustrious)"
  - Connect the resulting ILLUSTRIOUS_REGIONS to the "regions" input of Regional Conditioning.
  1. Pipe the Regional Conditioning output CONDITIONING to your sampler.

## Do I need VAE "regional" mode for Regional Prompting?

No. Keep VAE in standard mode unless you are explicitly doing inpainting. Regional Conditioning operates on conditioning, not the latent noise mask.

## Tips

- Masks: Black = protect/keep, White = change/apply. Use stronger feathering (blur 8–16px at image size) for smoother blends.
- CFG Scale: ~5.0 is a good starting point for Illustrious models.
- Steps: Increase for complex multi-region scenes.
- Overlaps: If region masks overlap, weights will be normalized to avoid over-driving any pixel.

## Auto Outpaint

- The Auto Outpaint node builds its own border mask and handles sampling.
- If you want regional prompting with outpainting:
  - Outpaint first, then run a second pass with Regional Conditioning.
  - Or make a manual outpaint pipeline and insert Regional Conditioning before the sampler stage.
