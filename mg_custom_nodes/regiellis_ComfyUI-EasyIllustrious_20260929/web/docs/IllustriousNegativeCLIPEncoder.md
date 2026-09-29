# IllustriousNegativeCLIPEncoder

Tells the AI what you DON'T want in your image. Like saying "no pickles" on your burger order!

## What it does

- Takes your list of things to avoid and makes the AI understand them
- Prevents common problems like extra fingers, bad anatomy, or ugly colors
- Works together with your positive prompt for complete control
- Essential for getting clean, professional results

## Inputs

- **text**: What you want to avoid (bad hands, blurry, etc.)
- **clip**: Text understanding model (usually auto is fine)
- **normalize**: Balances the strength (keep ON for stability)

## Outputs

- **CONDITIONING**: Instructions for what to avoid, ready for sampler

## Why You Need This

Think of it like quality control:
- **Without negatives**: AI might add weird artifacts
- **With negatives**: Clean, professional artwork
- **Too many negatives**: Confused AI, weird results

## The Golden Negative Prompts

**For Anime/Illustration:**
```
low quality, bad anatomy, bad hands, text, error, 
missing fingers, extra digit, fewer digits, cropped, 
worst quality, jpeg artifacts, signature, watermark
```

**For Realistic:**
```
deformed, ugly, mutilated, disfigured, text, extra limbs, 
face cut, head cut, extra fingers, poorly drawn hands, 
mutation, bad proportions, cropped head, malformed limbs
```

**For Clean Art:**
```
blurry, lowres, bad anatomy, bad hands, error, 
missing fingers, extra digit, cropped, worst quality, 
low quality, normal quality, jpeg artifacts
```

## Common Negatives Explained

- **"bad hands"**: Fixes finger problems
- **"text/watermark"**: Removes unwanted text
- **"low quality"**: Improves overall quality
- **"blurry"**: Sharper images
- **"jpeg artifacts"**: Cleaner output
- **"extra fingers"**: Correct anatomy
- **"cropped"**: Full composition

## When to Add More

**Getting specific problems?**
- Add that specific issue to negatives
- Example: Seeing doubles? Add "duplicate"
- Wrong style? Add "realistic" (for anime) or vice versa

## When to Use Less

**Image looking bland?**
- You might be over-restricting
- Remove some negatives
- Keep only essential ones

## Pro Tips

- Start with basic negative template
- Add specific issues as they appear
- Less is often more
- Some models have recommended negatives
- Save your perfect negative as preset

## Common Mistakes to Avoid

❌ **Don't do this:**
- "not ugly, not bad, not horrible" (redundant)
- Super long lists (confuses AI)
- Contradicting your positive prompt

✅ **Do this:**
- Short, specific negatives
- Target actual problems you see
- Use proven templates

## Example Workflows

**Basic Setup:**
1. Copy a golden negative prompt
2. Connect to sampler's negative input
3. Generate cleaner art instantly

**Problem Solving:**
1. See a specific issue?
2. Add it to negatives
3. Regenerate - problem solved!

**Style Control:**
1. Don't want realistic in anime?
2. Add "realistic, photo" to negatives
3. Get pure anime style

---
For positive prompts: [IllustriousCLIPTextEncoder](IllustriousCLIPTextEncoder.md)
For complete prompting: [IllustriousPrompt](IllustriousPrompt.md)