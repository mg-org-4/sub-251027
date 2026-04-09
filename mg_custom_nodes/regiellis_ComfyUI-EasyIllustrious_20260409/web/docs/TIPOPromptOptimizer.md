# TIPOPromptOptimizer

Smart prompt improver that automatically organizes and enhances your text. See your improvements in real-time!

## What does it do?

- Automatically reorganizes your prompt for better results
- Adds intelligent emphasis to important elements
- Removes redundant or conflicting tags
- Shows you exactly what changed with before/after preview
- Optimizes specifically for Illustrious models

## Inputs

- **prompt**: Your original text (messy is OK!)
- **optimization_level**: 
  - **Light**: Minor cleanup, keeps your style
  - **Standard**: Balanced improvement (recommended)
  - **Heavy**: Major restructuring for best quality
- **preserve_style**: Keep your writing style vs full optimization
- **auto_weight**: Add emphasis automatically to key elements
- **preview_changes**: Show what changed (great for learning!)

## Outputs

- **optimized_prompt**: Your improved prompt
- **change_report**: What was changed and why
- **weight_map**: Visual emphasis guide

## Artist Tips

- Perfect for cleaning up prompts from other sources
- Use preview to learn optimal prompt structure
- Great for beginners - write naturally, let it optimize
- Combine with prompt builder nodes for best results

## When to Use

- **Messy prompts**: Copied from web, mixed styles
- **Learning**: See how pros structure prompts
- **Consistency**: Ensure all prompts follow best practices
- **Quick fixes**: Clean up without manual editing

## Optimization Examples

**Before:** 
```
girl with blue hair smiling, wearing red dress, 
beautiful, in a garden, sunset, anime style, 
masterpiece, detailed
```

**After (Optimized):**
```
(masterpiece:1.2), best quality, 1girl, 
(blue hair:1.1), (smile), red dress, 
beautiful, garden, sunset lighting, 
anime style, highly detailed
```

## Common Improvements

- Moves quality tags to front
- Groups related concepts
- Adds appropriate weights
- Removes duplicates
- Fixes common typos
- Standardizes formatting

## Example Workflows

**Quick Cleanup:**
1. Paste any prompt → TIPOOptimizer →
2. Review changes → Use optimized version

**Learning Mode:**
1. Write your prompt naturally
2. Set preview ON
3. See what changes for better results
4. Learn and improve!

**Batch Processing:**
1. Multiple prompts → Optimizer →
2. Consistent quality across all

---
For manual control: [IllustriousPrompt](IllustriousPrompt.md)
For encoding: [IllustriousCLIPTextEncoder](IllustriousCLIPTextEncoder.md)
