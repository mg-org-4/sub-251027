# IllustriousMasterModel

Loads your AI art model. This is your starting point - pick your model and you're ready to create!

## What does it do?

- Loads Illustrious XL and compatible checkpoint files
- Automatically configures optimal settings for the model
- Provides model, CLIP, and VAE outputs for your workflow
- Detects model version (EPS/VPred) for best quality

## Inputs

- **ckpt_name**: Your model file (Illustrious XL, Pony, etc.)
- **vae_name**: VAE file (usually "Baked VAE" is best)
- **clip_name**: CLIP model (usually "ViT-L" or leave default)
- **clip_strength**: How strongly CLIP influences (1.0 = normal)
- **optimization**: Memory/speed settings (auto is usually best)

## Outputs

- **MODEL**: The main AI model for sampling
- **CLIP**: Text understanding model for prompts
- **VAE**: Image encoder/decoder
- **model_info**: Details about loaded model

## Artist Tips

- Illustrious XL v0.1 models: Great for anime/manga style
- NovelAI models: Excellent for detailed illustrations
- Pony models: Good for varied styles and characters
- Always use the VAE that came with your model for best colors
- Higher CLIP strength = more literal prompt following

## Choosing Models

**For Anime/Manga:**
- Illustrious-XL-v0.1
- Any Illustrious-based merges
- NovelAI-based models

**For Semi-Realistic:**
- Illustrious merged with realistic
- Pony-based merges

**For Artistic Styles:**
- Models trained on specific artists
- Style-focused checkpoints

## Memory Optimization

- **Auto**: Let ComfyUI decide (recommended)
- **Low VRAM**: For 6-8GB cards
- **Normal**: For 8-12GB cards  
- **High**: For 12GB+ cards
- **CPU**: Emergency mode (very slow)

## Common Issues & Fixes

**"Out of Memory":**
- Switch to "Low VRAM" mode
- Reduce batch size
- Use smaller resolution

**"Wrong Colors":**
- Check VAE selection
- Use "Baked VAE" if available
- Try different VAE file

**"Model Not Found":**
- Check models/checkpoints folder
- Refresh model list
- Ensure .safetensors extension

## Example Workflows

**Basic Setup:**
1. Select your Illustrious model
2. Keep VAE on "Baked VAE"
3. Connect to samplers and encoders

**Model Comparison:**
1. Load multiple MasterModel nodes
2. Use same prompt/seed
3. Compare different models side-by-side

**Memory-Conscious:**
1. Set optimization to "Low VRAM"
2. Use smaller resolutions initially
3. Upscale only final selections

## Pro Tips

- Download models from Civitai or Hugging Face
- Keep models organized in subfolders
- Test new models with same seed for consistency
- Some models need specific negative prompts
- Read model cards for optimal settings

---
Next steps: [IllustriousPrompt](IllustriousPrompt.md) for text input
For sampling: [IllustriousKSamplerPro](IllustriousKSamplerPro.md)
