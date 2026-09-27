# IllustriousScenesPlus

Your personal scene designer! Click a few buttons and get rich, detailed backgrounds and settings for your art.

## What it does

- Builds complete scene descriptions automatically
- Mixes time of day, weather, atmosphere, and props
- Creates backgrounds that match your art style
- Saves you from writing long scene descriptions

## Inputs

- **category**: What kind of place
  - **Outdoor**: Forests, beaches, mountains, gardens
  - **Indoor**: Rooms, cafes, libraries, studios
  - **Urban**: Streets, rooftops, parks, stations
  - **Fantasy**: Magical forests, castles, floating islands
  - **Seasonal**: Spring gardens, snowy scenes, autumn
  - **Action**: Battlefields, racing, sports scenes
  - **Romance**: Romantic settings, cozy spots
  
- **complexity**: How much detail to include
  - **Simple**: Basic scene (fast, clean)
  - **Medium**: Good detail (recommended)
  - **Detailed**: Rich description (cinematic)
  
- **include options**: What to add to your scene
  - **Time/Weather**: Sunset, rain, foggy morning
  - **Ambience**: Cozy, mysterious, epic, dreamy
  - **Events**: Festivals, markets, concerts
  - **Props**: Books, flowers, lanterns, crystals
  - **Population**: Crowded, empty, bustling
  - **Pose/Action**: What characters might do
  - **Person**: Add a character to the scene
  - **Clothing**: Outfit that fits the scene

## Outputs

- **scene_prompt**: Complete scene description ready to use
- **scene_summary**: What was created (for reference)

## Why This Saves Time

Instead of writing:
```
"cherry blossom garden in spring with pink petals falling, 
warm afternoon light filtering through trees, 
stone path winding through, peaceful atmosphere"
```

Just pick:
- Category: Seasonal
- Complexity: Medium  
- Include: Time/Weather, Ambience
- Done!

## Scene Categories Explained

**Outdoor**: Nature scenes
- Forests, lakes, mountains, fields
- Perfect for: Fantasy, adventure, peaceful scenes

**Indoor**: Inside spaces  
- Bedrooms, kitchens, workshops, libraries
- Perfect for: Intimate moments, cozy vibes

**Urban**: City life
- Streets, cafes, rooftops, subways
- Perfect for: Modern life, cyberpunk, slice-of-life

**Fantasy**: Magical places
- Enchanted forests, floating cities, crystal caves
- Perfect for: Magic, adventure, otherworldly

**Seasonal**: Time-specific scenes
- Spring gardens, autumn forests, winter snow
- Perfect for: Mood setting, holidays

## Complexity Guide

**Simple** = Basic scene elements
- Good for: Character focus, clean backgrounds
- Example: "library with books and warm lighting"

**Medium** = Balanced detail (most popular)
- Good for: Complete scenes, illustrations
- Example: "cozy library with ancient books, fireplace crackling, warm golden light, comfortable reading chair"

**Detailed** = Rich, cinematic
- Good for: Wallpapers, dramatic scenes
- Example: "Grand library with towering shelves of leather-bound books, ornate reading desks, golden afternoon light streaming through tall windows, dust motes dancing in sunbeams, comfortable wingback chairs, crackling fireplace"

## Smart Include Options

**Time/Weather** adds atmosphere:
- "at sunset", "during a storm", "foggy morning"

**Ambience** sets the mood:
- Cozy, mysterious, epic, romantic, peaceful

**Events** add life:
- Festival lights, market stalls, concert crowds

**Props** add detail:
- Flowers, lanterns, books, magical crystals

**Population** sets the energy:
- Busy crowds, peaceful solitude, bustling activity

## Pro Scene Combos

**Cozy Coffee Shop:**
- Category: Indoor
- Complexity: Medium
- Include: Ambience, Props, Time/Weather
- Result: Perfect cafe scene with atmosphere

**Epic Fantasy Battle:**
- Category: Fantasy  
- Complexity: Detailed
- Include: Events, Ambience, Props
- Result: Dramatic battlefield with details

**Romantic Date:**
- Category: Romance
- Complexity: Medium  
- Include: Time/Weather, Ambience, Person
- Result: Perfect romantic setting

## Using with Other Nodes

**Complete Character Scene:**
1. Character node → Character details
2. ScenesPlus → Background setting  
3. Clothing → Outfit that fits scene
4. Combine → Perfect illustrated story

**Style Consistency:**
1. Artist node → Art style
2. ScenesPlus → Scene description
3. Same style applied to detailed scene

## Example Workflows

**Quick Scene Generation:**
1. Pick category you like
2. Set complexity to Medium
3. Enable 2-3 include options
4. Instant detailed scene!

**Story Illustration:**
1. Think of story moment
2. Pick matching category
3. Add relevant includes
4. Get scene that tells story

**Mood Board Creation:**
1. Same settings, different categories
2. Generate multiple scenes
3. Find perfect mood/atmosphere

## Pro Tips

- Medium complexity works for 90% of scenes
- Don't enable ALL includes (overwhelming)
- Match scene to your character's story
- Urban + Time/Weather = great city vibes
- Fantasy + Props = magical details
- Romance + Ambience = perfect mood

## Chain Integration

Connect to IllustriousPrompt with chain enabled:
1. Your character prompt
2. + ScenesPlus scene
3. = Complete scene with character

Perfect for building complex prompts without writing everything yourself!

---
For characters: [IllustriousCharacters](IllustriousCharacters.md)
For prompts: [IllustriousPrompt](IllustriousPrompt.md)