# IllustriousE621Characters

Furry and anthro character database. Access thousands of E621-tagged characters!

## What does it do?

- Provides E621's extensive character database
- Specializes in furry, anthro, and creature characters
- Properly formats character tags for E621-trained models
- Includes species and trait information
- Allows character mixing and hybrids

## Inputs

- **character_1 to character_5**: E621 character names
- **weight_1 to weight_5**: Character prominence (0.5-1.5)
- **species_tag**: Override or specify species
- **include_traits**: Add character-specific features
- **hybrid_mode**: How to mix characters
  - **Separate**: Distinct characters
  - **Hybrid**: Blend species/traits
  - **Transform**: Character as different species

## Outputs

- **prompt**: E621-formatted character tags
- **species_info**: Species and trait details
- **chain_output**: For node chaining

## Artist Tips

- E621 characters work best with furry models
- Include species for better accuracy
- Use exact E621 tag format (underscores)
- Popular characters generate better
- Check character's species compatibility

## Popular E621 Characters

**Canonical Characters:**
- "loona_(helluva_boss)" - Hellhound
- "nick_wilde" - Fox from Zootopia
- "judy_hopps" - Rabbit from Zootopia
- "legoshi" - Wolf from Beastars
- "isabelle_(animal_crossing)" - Dog

**Original Characters (OCs):**
- "seasalt" - Popular dalmatian
- "reggie_(whygena)" - Mouse character
- "ankha" - Cat from Animal Crossing
- "krystal" - Fox from Star Fox

**Pokemon:**
- "lucario" - Fighting/Steel type
- "gardevoir" - Psychic type
- "lopunny" - Normal type
- "zoroark" - Dark type

## Species Compatibility

**Common species tags:**
- Canine: dog, wolf, fox, husky
- Feline: cat, lion, tiger, lynx
- Reptile: dragon, lizard, snake
- Avian: bird, gryphon, phoenix
- Hybrid: Multiple species mixed

## Weight Guidelines

- **0.5-0.7**: Background/secondary
- **0.8-1.0**: Standard presence
- **1.1-1.3**: Primary focus
- **1.4-1.5**: Dominant (careful!)
- **Multiple**: Balance weights carefully

## Character Mixing

**Species Swap:**
- Character: "nick_wilde" (1.0)
- Species: "dragon"
= Nick as a dragon

**Hybrid Creation:**
- Character 1: "loona" (1.0)
- Character 2: "isabelle" (0.7)
- Mode: Hybrid
= Mixed character design

**Crossover Scene:**
- Multiple characters at 0.8-1.0
- Mode: Separate
= Multiple characters together

## Common Issues & Fixes

**"Human instead of anthro":**
- Add "anthro" tag explicitly
- Check model has furry training
- Increase character weight

**"Wrong species":**
- Specify species in prompt
- Use species_tag override
- Check character's canon species

**"Generic appearance":**
- Use more specific tags
- Add character traits
- Consider using LoRA

## Model Requirements

**Best results with:**
- Pony Diffusion V6
- E621-trained models
- Furry-specific checkpoints

**Won't work well with:**
- Pure anime models
- Realistic human models
- Base SDXL

## Example Workflows

**Canon Character Art:**
1. Select character (e.g., "loona")
2. Set weight to 1.0
3. Add pose/scene tags
4. Generate!

**Species Transformation:**
1. Pick character
2. Override species tag
3. Adjust weight for balance
4. Create alternate versions

**OC Creation:**
1. Mix 2-3 characters at low weights
2. Specify unique species
3. Add custom colors/markings
4. Generate unique character

## Pro Tips

- Browse E621 for exact character tags
- Include copyright tags for series
- Species affects pose capabilities
- Some characters have multiple versions
- Combine with E621Artists for consistent style

## Tag Format Rules

- Use underscores: "nick_wilde" not "nick wilde"
- Add series: "loona_(helluva_boss)"
- Species separate: "fox anthro"
- Order matters: character, then species

---
For human/anime characters: [IllustriousCharacters](IllustriousCharacters.md)
For art styles: [IllustriousE621Artists](IllustriousE621Artists.md)
