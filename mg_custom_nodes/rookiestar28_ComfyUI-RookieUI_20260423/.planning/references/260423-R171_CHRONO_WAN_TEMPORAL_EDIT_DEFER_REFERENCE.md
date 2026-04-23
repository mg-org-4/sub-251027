# Chrono/Wan Temporal Edit Defer Reference

Date: 2026-04-23
Workspace: `C:\Users\Ray\Documents\我的專案\ComfyUI-RookieUI`
Purpose: freeze the first-wave defer contract for `Chrono Edit 14B` and the broader Wan-style temporal/video-like edit lineage so the initial RookieUI image-edit rollout stays scoped to static image-edit models.

## 1. Why this needs an explicit defer contract

The current official image-edit inventory now mixes two materially different groups:

- static image-edit templates that stay in the normal image-to-image domain
- temporal/video-like edit templates that start from an image but generate a time-extended latent/output path

Without an explicit defer contract, later `img2img` request/manifest/runtime work would either:

- overfit the static-image contract to a temporal graph
- or quietly omit `Chrono Edit 14B` without recording why

## 2. Concrete `Chrono Edit 14B` graph facts

From `reference/workflow_templates/imageEdit/Chrono Edit 14B.json`:

- image input starts with `LoadImage`
- temporal/video path is introduced by `WanImageToVideo`
- model behavior is modified by `ScaleROPE`
- vision conditioning uses `CLIPVisionLoader` + `CLIPVisionEncode`
- output selection uses `ImageFromBatch`
- text path uses `CLIPLoader(type="wan")`
- the graph still carries a fixed template-owned `LoraLoaderModelOnly(chronoedit_distill_lora.safetensors)`

Practical implication:

- this is not just "another static edit model with a different encoder"
- it crosses into Wan temporal/video semantics and should not be treated as a drop-in member of the first-wave static image-edit matrix

## 3. Host-side node evidence for the temporal split

Relevant host-side references:

- `reference/ComfyUI/comfy_extras/nodes_rope.py`
  - `ScaleROPE`
- `reference/ComfyUI/comfy_extras/nodes_wan.py`
  - `WanImageToVideo`
- `reference/ComfyUI/comfy_extras/nodes_images.py`
  - `ImageFromBatch`

Practical implication:

- the official host itself models this path with dedicated temporal/video-like nodes rather than the normal static image-edit latent/reference helpers

## 4. Why this should be deferred instead of squeezed into first-wave `img2img`

The first-wave image-edit contract already needs to absorb:

- ordered multi-reference input
- no-mask image-edit semantics
- manifest/profile expansion
- Qwen-family encoder variants
- Flux/Kontext/Klein/Longcat multi-reference latent helpers

Adding `Chrono Edit 14B` to that same first-wave scope would also require freezing:

- temporal/video output expectations
- Wan-specific prompt/conditioning semantics
- frame/batch extraction ownership
- possible duration/length/output-selection UX contracts

Planning implication:

- `Chrono Edit 14B` should be explicitly deferred so the first-wave chain can finish the static-image contract first

## 5. Defer rule being frozen

For the first-wave RookieUI image-edit rollout:

- `Chrono Edit 14B` is out of scope
- other Wan-style temporal/video-like image-edit graphs are also out of scope by category, not only by current file name
- first-wave acceptance must not imply support for these temporal/video-like edit graphs

## 6. Non-goals protected by this defer

- no temporal/video output UI in the first-wave `img2img` surface
- no Wan-specific duration/length/public parameter contract yet
- no temporal/video live-smoke claim in first-wave acceptance
- no attempt to normalize `Chrono Edit 14B` into the same static builder/helper seam used for Qwen or Flux-family image-edit

## 7. Re-entry condition for future work

The defer is not a rejection. Future work may reopen this chain when there is explicit scope for:

- Wan-style temporal edit request contracts
- temporal/video output handling in RookieUI
- dedicated runtime builders and live-smoke coverage for temporal edit graphs

## 8. Reference list for roadmap / plan citation

- `reference/workflow_templates/imageEdit/Chrono Edit 14B.json`
- `reference/ComfyUI/comfy_extras/nodes_rope.py`
- `reference/ComfyUI/comfy_extras/nodes_wan.py`
- `reference/ComfyUI/comfy_extras/nodes_images.py`
- `.planning/references/260423-R170_OFFICIAL_IMAGEEDIT_REBASELINE_REFERENCE.md`
