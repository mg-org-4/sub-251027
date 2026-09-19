# IllustriousArtists

Instantly apply famous art styles to your creations. Mix up to 5 artists for unique hybrid styles!

## What does it do?

- Provides a curated list of 1000+ recognized artists
- Automatically formats artist names for best model recognition
- Lets you blend multiple art styles with custom weights
- Creates unique hybrid styles by mixing artists
- Optimized for anime, illustration, and fine art styles

## Inputs

- **artist_1 to artist_5**: Select from dropdown or type artist name
- **weight_1 to weight_5**: Influence strength (0.5 = subtle, 1.0 = normal, 1.5 = strong)
- **mix_mode**: How to blend styles
  - **Balanced**: Equal influence from all
  - **Primary**: First artist dominates
  - **Gradient**: Smooth transition between styles
- **style_intensity**: Overall style strength (1.0 = normal)

## Outputs

- **prompt**: Artist style tags ready for your prompt
- **style_description**: Human-readable style summary
- **chain_output**: For connecting to other nodes

## Artist Tips

- **Single artist**: Pure style, most predictable
- **2 artists**: Interesting blend, still stable
- **3+ artists**: Experimental, unique results
- Start with weight 1.0, adjust if needed
- Popular anime artists work best with Illustrious

## Style Combinations That Work

**Dreamy Illustration:**
- Artist 1: "makoto_shinkai" (1.0)
- Artist 2: "james_jean" (0.7)
= Ethereal backgrounds with delicate details

**Bold Anime:**
- Artist 1: "yusuke_murata" (1.2)
- Artist 2: "hirohiko_araki" (0.8)
= Dynamic poses with strong lines

**Soft Fantasy:**
- Artist 1: "yoshitaka_amano" (1.0)
- Artist 2: "alphonse_mucha" (0.6)
= Flowing, ornate fantasy art

**Modern Digital:**
- Artist 1: "ross_tran" (1.0)
- Artist 2: "ilya_kuvshinov" (0.8)
= Clean, contemporary illustration

## Weight Guidelines

- **0.3-0.5**: Subtle influence, background style
- **0.6-0.8**: Noticeable but not dominant
- **0.9-1.1**: Standard influence
- **1.2-1.5**: Strong, dominant style
- **1.5+**: Very strong (may overpower content)

## Finding Artists

**Anime/Manga:** Look for pixiv, twitter artists
**Classical:** Old masters, art movements
**Digital:** ArtStation, DeviantArt popular artists
**Game Art:** Concept artists from favorite games

## Common Issues & Fixes

**"Style not showing":**
- Increase weight to 1.2-1.5
- Check artist name spelling
- Some artists need specific model versions

**"Too stylized":**
- Reduce weight to 0.5-0.7
- Use fewer artists (1-2 max)
- Lower style_intensity

**"Conflicting styles":**
- Choose artists from similar genres
- Adjust weights so one dominates
- Try "Primary" mix mode

## Example Workflows

**Style Exploration:**
1. Same prompt + seed
2. Try different single artists
3. Find your favorites
4. Mix top 2 for unique style

**Commission Work:**
1. Client requests "Studio Ghibli style"
2. Use "hayao_miyazaki" (1.0)
3. Add "kazuo_oga" (0.5) for backgrounds
4. Perfect studio style achieved

**Personal Style:**
1. Pick 3 favorite artists
2. Experiment with weights
3. Find your signature blend
4. Save weights for consistency

## Pro Tips

- Research artist's actual work first
- Similar art periods blend better
- Anime + realism = interesting results
- Some artists have "signature elements" (learn them!)
- Weight ratios matter more than absolute values

---
For furry/anthro styles: [IllustriousE621Artists](IllustriousE621Artists.md)
For prompts: [IllustriousPrompt](IllustriousPrompt.md)
