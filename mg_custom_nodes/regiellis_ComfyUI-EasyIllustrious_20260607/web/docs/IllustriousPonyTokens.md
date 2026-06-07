# IllustriousPonyTokens

Makes Pony/MLP models understand your prompts perfectly. Essential for getting authentic pony art!

## What it does

- Fixes prompts so Pony Diffusion models work properly
- Adds the right character names and style tags
- Ensures your pony art looks authentic to the show
- Prevents generic or off-model results

## Why You Need This

Pony models are picky about how you write prompts:
- **Wrong**: "purple unicorn with magic"
- **Right**: "twilight sparkle, unicorn, magic aura, pony"

This node automatically fixes your prompts!

## Inputs

- **base_prompt**: Your description (can be normal English)
- **character**: Which pony you want
  - Twilight Sparkle, Rainbow Dash, Fluttershy, etc.
  - Leave empty for original characters
- **style_override**: Art style
  - "show style" = Official MLP look
  - "anthro" = Humanoid ponies
  - "realistic" = Photo-style ponies
- **mode**: How to fix the prompt
  - **Auto**: Smart fixing (recommended)
  - **Prepend**: Add pony tags at start
  - **Append**: Add pony tags at end
  - **Replace**: Full pony-style conversion

## Outputs

- **prompt**: Fixed prompt that works with pony models
- **tags_added**: Shows what was changed

## Popular Characters

**Mane 6:**
- twilight sparkle (purple unicorn, magic)
- rainbow dash (blue pegasus, speed)
- pinkie pie (pink earth pony, party)
- fluttershy (yellow pegasus, animals)
- rarity (white unicorn, fashion) 
- applejack (orange earth pony, farm)

**Princesses:**
- princess celestia (white alicorn, sun)
- princess luna (blue alicorn, moon)
- princess cadance (pink alicorn, love)

## Style Guide

**Show Style** (most popular):
- Cartoon look from the TV show
- Bright colors, simple shading
- Cute, friendly expressions

**Anthro Style**:
- Human-like body with pony features
- Standing upright, hands instead of hooves
- More mature/adult themes possible

**Realistic Style**:
- Actual horse proportions
- Detailed fur and anatomy
- Photo-realistic rendering

## Mode Explained

**Auto Mode** (easiest):
- Figures out what you want
- Adds missing pony tags
- Fixes character names
- Best for beginners

**Prepend Mode**:
- Adds pony stuff at the beginning
- Good for style consistency
- "pony, twilight sparkle, YOUR_PROMPT"

**Replace Mode**:
- Completely rewrites for pony models
- Most accurate to show
- Use when other modes fail

## Common Fixes Applied

- "horse" → "pony"
- "purple unicorn" → "twilight sparkle"
- "flying horse" → "rainbow dash, pegasus"
- Adds "pony" tag if missing
- Fixes character name spelling
- Adds species tags (unicorn, pegasus, earth pony)

## Example Transformations

**Input:** "cute purple horse reading a book"
**Output:** "twilight sparkle, unicorn, pony, reading, book, cute, show style"

**Input:** "rainbow pony flying fast"
**Output:** "rainbow dash, pegasus, pony, flying, fast, dynamic pose, show style"

## Pro Tips

- Always use with Pony Diffusion models
- "Show style" works best for authentic look
- Character names are case-sensitive
- Add "score_9" for highest quality
- Use with E621 tags for best results

## Common Issues & Fixes

**"Doesn't look like MLP":**
- Enable "show style" override
- Use Replace mode
- Check you're using a pony model

**"Wrong character":**
- Check character name spelling
- Try Replace mode
- Add character's cutie mark to prompt

**"Generic pony":**
- Specify exact character name
- Add unique features to prompt
- Use higher prompt strength

## Model Compatibility

**Works great with:**
- Pony Diffusion v6
- Any pony-trained models
- MLP-specific checkpoints

**Won't help with:**
- Pure anime models
- Realistic photo models
- Models without pony training

## Example Workflows

**Canon Character:**
1. Type: "purple unicorn studying magic"
2. Character: "twilight sparkle"
3. Style: "show style"
4. Mode: Auto
5. Perfect MLP art!

**Original Character:**
1. Type: "green earth pony farmer"
2. Character: (leave empty)
3. Style: "show style"
4. Mode: Prepend
5. Custom pony in MLP style!

---
For general characters: [IllustriousCharacters](IllustriousCharacters.md)
For furry characters: [IllustriousE621Characters](IllustriousE621Characters.md)
