# IllustriousSmartPromptCache

A node for caching and reusing prompt conditioning in Illustrious workflows.

## What does it do?

- Stores prompt-to-conditioning mappings for fast reuse.
- Reduces recomputation when prompts are unchanged.
- Supports smart invalidation when upstream nodes change.

## Inputs

- **prompt**: The text prompt to cache.
- **conditioning**: The computed conditioning to store.
- **mode**: Cache mode (auto, force, manual).
- **key**: (Optional) Custom cache key.
- **log**: Enable for debug info.

## Outputs

- Cached conditioning for use in samplers or other nodes.

## Artist Tips

- Use in large workflows to speed up repeated prompt runs.
- Auto mode is best for most cases; force for guaranteed refresh.
- Manual mode lets you control cache keys for advanced setups.
- Combine with Smart Scene Generator for complex prompt pipelines.

## Example Workflow

1. Connect your prompt and conditioning.
2. Choose cache mode.
3. Use the cached output in your sampler or prompt node.

---
For more on prompt caching, see the Smart Scene and workflow optimization guides.
