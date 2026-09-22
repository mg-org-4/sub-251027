# IllustriousPoses

Makes characters copy poses from reference images. Like having a model pose for your AI!

## What it does

- Takes any pose from a photo or drawing
- Makes your AI character copy that exact pose
- Perfect for consistent character positioning
- Works with any style while keeping the pose

## Inputs

- **pose_image**: Reference image showing the pose you want
- **strength**: How exactly to copy the pose
  - **0.5**: Loose interpretation, natural variation
  - **0.7**: Good balance (recommended)
  - **1.0**: Exact copy, very strict
  - **1.2**: Force the pose even if awkward
- **prompt**: What character/style you want in that pose

## Outputs

- **CONDITIONING**: Pose instructions for your sampler

## Why Use This?

**Instead of:**
"1girl, standing, hand on hip, looking back"
= Generic, might not match what you imagined

**With poses:**
Find perfect reference photo → Extract pose → Get exact positioning
= Precise control over body language

## Getting Pose References

**From Photos:**
1. Find any photo with good pose
2. Use ControlNet OpenPose to extract skeleton
3. Use that skeleton as your pose_image

**From Art:**
1. Screenshot pose from artwork
2. Extract pose lines/skeleton
3. Apply to your character

**Stock Poses:**
1. Search "pose reference" online
2. Save poses you like
3. Build your pose library

## Strength Guidelines

**0.3-0.5**: Suggestion only
- Good for: Natural variation, loose reference
- Risk: Might ignore pose entirely

**0.6-0.8**: Balanced control (best for most cases)
- Good for: Clear pose guidance with flexibility
- Risk: None, sweet spot

**0.9-1.0**: Strict copying
- Good for: Exact replication, animation frames
- Risk: Might look stiff or unnatural

**1.1+**: Force the pose
- Good for: Difficult poses, stubborn models
- Risk: Broken anatomy, weird proportions

## Common Use Cases

**Character Sheets:**
- Same character, different poses
- Consistent style, varied positioning
- Perfect for reference sheets

**Animation Frames:**
- Key poses for movement
- Consistent character between frames
- Smooth animation planning

**Comic Panels:**
- Dramatic poses for storytelling
- Consistent character positioning
- Dynamic action scenes

**Portrait Variations:**
- Different angles of same character
- Various expressions with set pose
- Professional headshot variations

## Pro Tips

- Clean, clear pose references work best
- Simple backgrounds in pose images
- Strength 0.7 is the magic number for most cases
- Combine with negative prompts to avoid copying unwanted details
- Test different strengths with same seed

## Common Issues & Fixes

**"Pose not copying":**
- Increase strength to 0.8-1.0
- Use clearer pose reference
- Check pose extraction quality

**"Looks stiff/unnatural":**
- Reduce strength to 0.5-0.6
- Add "natural pose" to prompt
- Use more dynamic reference

**"Wrong body parts":**
- Use better pose extraction
- Clean up skeleton lines
- Try different reference angle

## Example Workflows

**Basic Pose Copy:**
1. Find reference photo
2. Extract pose with OpenPose
3. Set strength to 0.7
4. Add your character prompt
5. Generate!

**Character Sheet:**
1. Collect 5-10 pose references
2. Extract all poses
3. Same character prompt for all
4. Batch generate consistent sheet

**Animation Planning:**
1. Plan key poses for movement
2. Extract each pose
3. Generate frames with poses
4. Plan your animation sequence

---
For characters: [IllustriousCharacters](IllustriousCharacters.md)
For sampling: [IllustriousKSamplerPro](IllustriousKSamplerPro.md)
