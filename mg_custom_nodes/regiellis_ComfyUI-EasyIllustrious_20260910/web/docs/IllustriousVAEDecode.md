# IllustriousVAEDecode

Converts your AI's working canvas (latent) into a viewable image. Think of it as the "develop" button that turns your invisible blueprint into actual pixels.

## What does it do?

- Takes the AI's internal representation and converts it to a real image you can see
- Automatically handles large images by processing them in tiles (like assembling a puzzle)
- Preserves all the details and colors from your generation process
- Works seamlessly with Illustrious models for best quality

## Inputs

- **samples**: The latent/canvas from your sampler (the AI's working version)
- **vae**: The decoder model (usually loaded with your main model)
- **tile_size**: For huge images, breaks them into smaller chunks (default: auto)

## Outputs

- **IMAGE**: Your final artwork, ready to save or process further

## Artist Tips

- This is always the final step before seeing your image
- If you get memory errors with large images, try enabling tiling
- The VAE quality affects final colors - use the VAE that came with your model
- For best colors with Illustrious, use this instead of generic VAE decode nodes

## When to Use

- **Always**: After any sampler to see your results
- **Image2Image**: Decode → Edit → Encode → Sample again
- **Upscaling**: After latent upscale to see the larger image

## Example Workflow

1. Sampler creates latent → 
2. VAEDecode converts to image → 
3. Save or apply color corrections

---
For the reverse process (image to latent), see [IllustriousVAEEncode](IllustriousVAEEncode.md)
