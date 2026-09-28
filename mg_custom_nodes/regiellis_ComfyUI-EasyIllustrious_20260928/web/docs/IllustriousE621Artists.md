# IllustriousE621Artists

Specialized artist styles for furry and anthro art. Access E621's massive artist database!

## What does it do?

- Provides E621 platform's recognized artist tags
- Optimized for furry, anthro, and creature art
- Properly formats artist names with E621 conventions
- Allows mixing multiple artists for hybrid styles
- Works best with models trained on E621 data

## Inputs

- **artist_1 to artist_5**: E621 artist names
- **weight_1 to weight_5**: Style strength (0.5-1.5 typical)
- **species_compatibility**: Auto-adjust for species
- **rating_filter**: SFW/NSFW style filtering
- **style_mode**: How to apply the style

## Outputs

- **prompt**: E621-formatted artist tags
- **compatibility_info**: Model compatibility notes
- **chain_output**: For node chaining

## Artist Tips

- E621 artists specialize in anthropomorphic art
- These work best with Pony or E621-trained models
- Start with single artist to test compatibility
- Mix 2-3 artists max for stable results
- Check artist's actual gallery for style reference

## Popular E621 Artists & Styles

**Cute/Cartoon:**
- "glacierclear": Soft, adorable style
- "falvie": Vibrant, playful characters
- "zackary911": Clean, appealing designs

**Detailed/Realistic:**
- "ruaidri": High detail, realistic fur
- "zaush": Professional, polished art
- "keeltheequine": Painterly, atmospheric

**Stylized/Unique:**
- "slugbox": Bold, graphic style
- "glitchedpuppet": Distinctive, colorful
- "zeta-haru": Anime-influenced furry

## Weight Guidelines

- **0.3-0.5**: Light style touch
- **0.6-0.8**: Moderate influence
- **0.9-1.1**: Standard strength
- **1.2-1.5**: Strong style
- **Avoid >1.5**: Can break proportions

## Species Compatibility

**Best matches:**
- Canine artists → Wolf, dog, fox characters
- Feline artists → Cat, lion, tiger characters
- Avian artists → Bird, gryphon characters
- Reptile artists → Dragon, lizard characters

## Model Compatibility

**Best with:**
- Pony Diffusion models
- E621-trained models
- Furry-specific fine-tunes

**May not work with:**
- Pure anime models
- Photorealistic models
- Base SDXL without furry training

## Common Issues & Fixes

**"Wrong anatomy":**
- Check artist specializes in your species
- Reduce weight to 0.7-0.9
- Add species tags to prompt

**"Style not appearing":**
- Verify model has E621 training
- Check exact artist tag spelling
- Some artists need underscores (not spaces)

**"Too extreme":**
- Lower weight to 0.5
- Check artist's typical content
- Use rating_filter if available

## Example Workflows

**Character Design:**
1. Choose species (e.g., "wolf")
2. Add matching artist (e.g., "zackary911")
3. Set weight to 1.0
4. Generate variations

**Style Study:**
1. Keep same character/seed
2. Try different E621 artists
3. Compare style differences
4. Mix favorites

**Commission Reference:**
1. Client wants "E621 style"
2. Use 2-3 popular artists
3. Blend at 0.8 weight each
4. Achieve platform-authentic look

## Pro Tips

- Browse E621 to preview artist styles
- Tag format: use underscores, not spaces
- Some artists have "signature poses" 
- Combine with species-specific prompts
- Not all E621 artists work in all models

## Finding Artists

1. Visit E621.net (if appropriate)
2. Search by style you like
3. Note artist tags exactly
4. Test in your workflow

---
For general artists: [IllustriousArtists](IllustriousArtists.md)
For characters: [IllustriousE621Characters](IllustriousE621Characters.md)
