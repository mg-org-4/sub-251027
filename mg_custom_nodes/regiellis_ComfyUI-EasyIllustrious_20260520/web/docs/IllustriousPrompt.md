# IllustriousPrompt

Your main creative control panel. Write what you want, add quality boosts, and connect other nodes for complex prompts.

## What does it do?

- Takes your text description and formats it for best results
- Adds quality tags automatically (masterpiece, best quality, etc.)
- Connects with other Illustrious nodes to build complex prompts
- Handles tag formatting and weight syntax for you

## Inputs

- **text**: Your main description (characters, scene, style, etc.)
- **add_quality_tags**: Auto-add "masterpiece, best quality" (recommended ON)
- **add_year_tag**: Adds "year 2024" for latest art styles
- **add_chain_insert**: Connect other Illustrious nodes here
- **prompt_strength**: Overall intensity (1.0 = normal, 1.2 = stronger)
- **use_danbooru_format**: Format tags with underscores (better for anime)

## Outputs

- **prompt**: Complete formatted text for encoders
- **raw_prompt**: Unformatted version for debugging

## Artist Tips

- Start simple: "1girl, blue hair, sunset, beach"
- Let quality tags do their work (keep them ON)
- Use parentheses for emphasis: (beautiful eyes)
- Use square brackets to reduce: [background]
- Connect Character/Artist/Style nodes for instant complexity

## Prompt Writing Basics

**Good Prompt Structure:**
1. Subject: "1girl" or "landscape" 
2. Details: "long hair, red eyes, smile"
3. Outfit/Scene: "school uniform, classroom"
4. Style: "digital art, soft lighting"
5. Quality: (auto-added if enabled)

**Weight Control:**
- Normal: red hair
- Stronger: (red hair) or (red hair:1.2)
- Weaker: [red hair] or (red hair:0.8)
- Very strong: ((red hair)) or (red hair:1.5)

## Common Patterns

**Portrait:**
```
1girl, close-up, detailed eyes, soft smile, 
bokeh background, professional lighting
```

**Full Scene:**
```
1girl, standing, city street, sunset, 
urban environment, cinematic composition
```

**Action Shot:**
```
1boy, running, dynamic pose, speed lines,
action scene, dramatic lighting
```

## Chain Insert Magic

Connect these nodes to the chain_insert:
- Characters → Instant character details
- Artists → Apply art styles
- Clothing → Quick outfit selection
- Hairstyles → Perfect hair every time
- Scene Generator → Complex backgrounds

## Example Workflows

**Simple Portrait:**
1. Type: "1girl, portrait, smile"
2. Keep quality tags ON
3. Encode → Sample → Beautiful portrait

**Complex Character:**
1. Connect: Character node → chain_insert
2. Connect: Clothing node → chain_insert  
3. Add your scene description
4. Everything merges automatically!

**Style Mixing:**
1. Connect: Artist node → chain_insert
2. Type your subject
3. Instant style transfer!

---
For encoding, see [IllustriousCLIPTextEncoder](IllustriousCLIPTextEncoder.md)
For characters, see [IllustriousCharacters](IllustriousCharacters.md)
