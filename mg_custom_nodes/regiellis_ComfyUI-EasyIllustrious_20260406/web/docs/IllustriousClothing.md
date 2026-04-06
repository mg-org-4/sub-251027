# IllustriousClothing

Quick clothing selector with hundreds of pre-tagged outfits. No more googling "what's that jacket called" - just pick and create!

## What does it do?

- Provides organized categories of clothing with proper anime/illustration tags
- Automatically formats clothing descriptions for best model understanding
- Includes full outfits, individual pieces, and accessories
- Smart placement in your prompt for consistent results

## Inputs

- **category**: Type of clothing (tops, bottoms, dresses, outfits, etc.)
- **style**: Specific item within category
- **color**: (Optional) Override default colors
- **material**: (Optional) Add texture details (leather, silk, denim, etc.)
- **details**: (Optional) Extra descriptors (torn, oversized, fitted, etc.)

## Outputs

- **prompt**: Ready-to-use clothing description
- **chain_output**: For connecting to other Illustrious nodes

## Artist Tips

- Start with full outfits for quick results, then customize
- Layer multiple clothing nodes for complex costumes
- The model knows fashion terms better than generic descriptions
- Use with Characters node for complete character designs

## Common Clothing Combos

**School Uniform:**
- Category: "outfits" → Style: "school_uniform"
- Auto-includes: blazer, skirt/pants, proper accessories

**Fantasy Armor:**
- Category: "outfits" → Style: "armor_fantasy"
- Auto-includes: plates, chainmail, appropriate details

**Casual Modern:**
- Category: "tops" → Style: "t-shirt" + 
- Category: "bottoms" → Style: "jeans"

## When to Adjust Settings

- **Default**: Use preset colors and materials (most consistent)
- **Custom color**: When you need specific color schemes
- **Material override**: For unique textures (wet clothes, metallic, etc.)
- **Details**: Add wear, fit, or style modifiers

## Example Workflows

**Quick Character:**
1. Character node → Clothing node → 
2. Prompt → Sampler → Instant dressed character

**Fashion Design:**
1. Multiple Clothing nodes (top, bottom, accessories) →
2. Combine → Unique outfit combinations

**Costume Variations:**
1. Same character + different Clothing selections →
2. Batch generate → Fashion lineup

---
For hairstyles, see [IllustriousHairstyles](IllustriousHairstyles.md)
For full characters, see [IllustriousCharacters](IllustriousCharacters.md)
