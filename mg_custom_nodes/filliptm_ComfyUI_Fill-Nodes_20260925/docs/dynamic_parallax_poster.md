# Dynamic parallax posters

`FL Image Layer Planner` (existing node ID `FL_PosterLayerPlanner`) inspects one finished image using a vision-capable CLIP, or accepts a manual JSON layer list. Auto mode chooses up to the requested budget, including one filled background. Grouping controls whether related objects stay together; `artwork_type` guides photography, illustration, graphic design, products and mixed media. Text layers are only planned when lettering is visible. Auto can correct an invalid plan once and removes other-object location clauses from cutout targets; Manual prompts remain unchanged. Typography is extracted from the generated artwork, not rendered as an overlay.

Manual plans use this form:

```json
[
  {"name":"Paper", "kind":"background", "prompt":"Cream paper without objects or lettering", "depth":12},
  {"name":"Headline", "kind":"text", "prompt":"The large black AFTER HOURS lettering at the top", "depth":8},
  {"name":"Hero", "kind":"art", "prompt":"Woman wearing a silver jacket and headphones", "depth":4}
]
```

`FL Poster Layers` expands the plan into native ComfyUI conditioning, sampling, and decode nodes. Use **Qwen Image Layered Control**, its text encoder, and the layered RGBA VAE. The image reference is encoded once; the Control latent has `layers=0`, meaning one RGBA result per extraction. Each layer has its own stable sampler ID and seed.

The review panel shows small thumbnails and full-resolution RGBA on click. Edit a basic-English extraction prompt, depth, scale, offset or visibility, then Run. Reroll changes one layer's seed and queues the workflow. Reset removes that layer's edits for the next Run. Edits are scoped to the plan; changed plans do not inherit stale object prompts.

Connect the resulting `FL_PARALLAX_STACK` to **FL Layered Parallax**. It supplies the background and a variable-length list of cutouts. Existing individual layer inputs still work. Lower depths are nearer and move faster; keep the camera background depth above every cutout depth.

The compositor now includes [Parallax Studio](parallax_editor.md), a visual camera editor with live low-resolution previews, presets, drag controls and cover/contain framing. Width, height and frames remain standard widgets.

Layout changes only affect stack assembly/composition. Prompt edits or rerolls invalidate the affected extraction path, not every sampler. A node-scoped cache provider recovers exact RGBA tensors when ComfyUI's RAM cache evicts them. It uses ComfyUI's native dependency signatures, writes safetensors atomically under `user/__cache/fl_poster_layers_v1`, and keeps at most 1 GB by removing its oldest cache files. No large tensor store is retained by the provider. Clearing that folder forces fresh extraction; exported PNGs remain untouched. RGBA assets and 192-pixel thumbnails are saved under `output/Dynamic_Parallax_Poster/layers` and `thumbnails`, so saved reviews survive a server restart.

These models generate their extraction: lettering can change, small items can be omitted, and hidden regions can be invented. Check the source spelling, alpha previews, and reassembled still before exporting motion. Auto planning is not a guarantee of perfect semantic separation.
