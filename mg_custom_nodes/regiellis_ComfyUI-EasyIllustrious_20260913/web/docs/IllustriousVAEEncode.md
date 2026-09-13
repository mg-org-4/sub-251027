# IllustriousVAEEncode

Converts your existing image into the AI's working format (latent). Like scanning a painting so the AI can understand and modify it.

## What does it do?

- Takes any image and converts it to the AI's internal canvas format
- Enables you to start from existing artwork instead of noise
- Preserves the structure and composition of your original image
- Essential for img2img workflows, inpainting, and style transfer

## Inputs

- **pixels**: Your source image (from Load Image, previous generation, etc.)
- **vae**: The encoder model (usually loaded with your main model)
- **mask**: (Optional) Select specific areas to regenerate (for inpainting)
- **grow_mask_by**: Expand masked area for smoother blending

## Outputs

- **LATENT**: The encoded canvas ready for sampling/modification

## Artist Tips

- Perfect for "variations" - encode your image then sample with low denoise (0.3-0.5)
- Higher denoise = more changes, lower = stays closer to original
- Use mask for selective editing (fix hands, change background, etc.)
- The VAE quality affects how well details are preserved

## When to Use

- **Style Transfer**: Encode photo → Sample with art style prompt
- **Variations**: Make similar versions of existing art
- **Inpainting**: Fix or change parts of an image
- **Upscaling**: Encode → Upscale latent → Sample → Better details

## Common Settings

- **No mask**: Full image gets processed
- **With mask**: Only masked areas change (great for corrections)
- **Denoise 0.3**: Subtle variations, keeps most original
- **Denoise 0.7**: Major changes, uses original as guide
- **Denoise 1.0**: Complete regeneration (why encode then?)

## Example Workflows

**Quick Variation:**
1. Load Image → VAEEncode → 
2. KSampler (denoise 0.4) → 
3. VAEDecode → Similar but unique version

**Style Transfer:**
1. Load Photo → VAEEncode →
2. Add style prompt → KSampler (denoise 0.6) →
3. VAEDecode → Stylized version

---
For the reverse process (latent to image), see [IllustriousVAEDecode](IllustriousVAEDecode.md)
