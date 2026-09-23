# IllustriousSmartCacheCorrector

Fixes cache issues and speeds up your workflow. When things get stuck or slow, this is your reset button!

## What does it do?

- Detects and fixes stale cached data that's slowing you down
- Refreshes specific parts without clearing everything
- Optimizes memory usage during long sessions
- Prevents the "why isn't my prompt changing?" problem
- Auto-clears cache when switching between projects

## Inputs

- **cache_mode**: How aggressive to be
  - **Smart**: Only clear what's changed
  - **Selective**: Clear specific items
  - **Full**: Complete refresh (nuclear option)
  - **Auto**: Let it decide based on usage
  
- **target**: What to clear
  - **Prompts**: Text/conditioning cache
  - **Images**: Generated image cache
  - **Models**: Model state cache
  - **All**: Everything
  
- **threshold_mb**: Clear when cache exceeds this size
- **age_minutes**: Clear items older than this
- **force_refresh**: Bypass smart detection

## Outputs

- **status**: What was cleared and why
- **cache_size**: Current cache usage
- **performance_gain**: Estimated speedup

## Artist Tips

- Run this when generations feel sluggish
- Use Smart mode for everyday clearing
- Full clear when switching projects
- Set threshold to 50% of your VRAM
- Age-based clearing good for long sessions

## When to Use

**Prompt not updating?**
- Target: Prompts
- Mode: Selective
- Your changes will finally show!

**Out of memory warnings?**
- Target: All
- Mode: Full
- Frees up VRAM instantly

**Workflow getting slow?**
- Mode: Smart
- Threshold: 2048 MB
- Automatic optimization

**Switching projects?**
- Mode: Full
- Clean slate for new work

## Common Issues & Fixes

**"Cache keeps refilling":**
- Lower threshold_mb
- Enable age-based clearing
- Check for memory leaks

**"Lost my cached prompts":**
- Use Selective mode
- Only target specific types
- Smart mode preserves recent items

**"No performance improvement":**
- Try Full clear
- Restart ComfyUI if needed
- Check system memory too

## Pro Tips

- Schedule clears between batches
- Monitor cache_size output
- Use before long generations
- Combine with prompt cache nodes
- Clear models when switching styles

## Example Workflows

**Regular Maintenance:**
1. Every 10 generations
2. Run Smart clear
3. Continue working smoothly

**Project Switch:**
1. Finish current project
2. Full cache clear
3. Load new project
4. Fresh, fast performance

**Memory Recovery:**
1. Getting VRAM warnings
2. Target: Images + Models
3. Mode: Full
4. Instantly recover memory

---
For caching prompts: [IllustriousSmartPromptCache](IllustriousSmartPromptCache.md)
For performance: [IllustriousKSamplerPro](IllustriousKSamplerPro.md)
