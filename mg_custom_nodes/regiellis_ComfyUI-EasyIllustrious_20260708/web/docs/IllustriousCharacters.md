# IllustriousCharacters

Instant character library with 10,000+ anime, game, and manga characters. Just pick and generate!

## What does it do?

- Provides a massive database of recognized characters
- Automatically formats character names and traits
- Lets you mix multiple characters or create crossovers
- Includes appearance details for accurate generation
- Works with popular anime, games, and manga series

## Inputs

- **character_1 to character_5**: Select or type character name
- **weight_1 to weight_5**: Character prominence (0.5-1.5)
- **include_outfit**: Add canonical outfit (default: yes)
- **include_features**: Add hair/eye colors (default: yes)
- **crossover_mode**: How to blend multiple characters
  - **Separate**: Each character distinct
  - **Fusion**: Blend characteristics
  - **Cosplay**: Character 1 in outfit of Character 2

## Outputs

- **prompt**: Complete character description
- **character_tags**: Individual character traits
- **chain_output**: For connecting to other nodes

## Artist Tips

- Single character = most accurate
- Use underscores for names: "hatsune_miku"
- Popular characters work better
- Newer anime may need more weight
- Check spelling - it matters!

## Popular Characters That Work Great

**Anime Classics:**
- "hatsune_miku" - Vocaloid icon
- "rem_(re:zero)" - Specify series in parentheses
- "asuka_langley" - Evangelion
- "zero_two" - Darling in the Franxx
- "saber" - Fate series

**Game Characters:**
- "2b_(nier)" - NieR Automata
- "ganyu_(genshin)" - Genshin Impact
- "tifa_lockhart" - Final Fantasy
- "princess_zelda" - Legend of Zelda

**Manga/Light Novel:**
- "nezuko_kamado" - Demon Slayer
- "mikasa_ackerman" - Attack on Titan
- "megumin" - Konosuba
- "albedo_(overlord)" - Overlord

## Weight Guidelines

- **0.5-0.7**: Background character, subtle
- **0.8-1.0**: Standard prominence
- **1.1-1.3**: Enhanced, focal character
- **1.4-1.5**: Very strong (may override scene)
- **Multiple characters**: Keep total ≤2.5

## Character Mixing Techniques

**Crossover Scene:**
- Character 1: "naruto" (1.0)
- Character 2: "goku" (1.0)
- Mode: Separate
= Both characters in same image

**Fusion/Hybrid:**
- Character 1: "sailor_moon" (1.0)
- Character 2: "cardcaptor_sakura" (0.7)
- Mode: Fusion
= Magical girl hybrid design

**Cosplay Style:**
- Character 1: "miku" (1.2)
- Character 2: "saber" (0.8)
- Mode: Cosplay
= Miku wearing Saber's armor

## Common Issues & Fixes

**"Wrong character appearing":**
- Add series name: "rem_(re:zero)"
- Increase weight to 1.2+
- Check exact character tag

**"Mixed up features":**
- Use Separate mode for distinct characters
- Lower secondary character weights
- Add "solo" for single character

**"Missing outfit":**
- Enable include_outfit
- Add outfit tags manually
- Some characters need outfit specified

## Series-Specific Tips

**Genshin Impact:** Add "_(genshin)"
**Fate Series:** Specify class (saber, archer)
**Vtubers:** Use agency tag (hololive, nijisanji)
**Pokemon:** Add "_pokemon" suffix

## Example Workflows

**Fan Art Creation:**
1. Select favorite character
2. Set weight to 1.0
3. Add scene/pose description
4. Generate variations

**Crossover Art:**
1. Pick 2 characters from different series
2. Use Separate mode
3. Add interaction prompts
4. Create unique crossovers

**OC with Inspiration:**
1. Mix 2-3 characters at 0.5-0.7 weight
2. Use Fusion mode
3. Add unique details
4. Create original character

## Pro Tips

- Download character LoRAs for better accuracy
- Recent characters may need higher weights
- Use negative prompts to exclude wrong characters
- Combine with Clothing node for outfit variations
- Series tags help disambiguation

## Finding Character Tags

- Check Danbooru for exact tags
- Use underscores, not spaces
- Include series for common names
- Test with single character first

---
For furry characters: [IllustriousE621Characters](IllustriousE621Characters.md)
For outfits: [IllustriousClothing](IllustriousClothing.md)
