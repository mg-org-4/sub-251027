# Reference Memo - Multi-Reference ImageEdit Conditioning Foundation

Date: 2026-04-23
Scope: `F164` shared image-edit conditioning seam

## Authoritative reference facts

- `reference/workflow_templates/imageEdit/Qwen-image edit.json`
  - current shipped baseline stays on `TextEncodeQwenImageEdit`
  - single reference image only
- `reference/workflow_templates/imageEdit/Firered image edit.json`
  - uses `TextEncodeQwenImageEditPlus`
  - confirms the Qwen-family expansion path moves beyond the older single-image encoder seam
- `reference/workflow_templates/imageEdit/Qwen-image edit-multi-lora.json`
  - confirms Qwen-family edit delivery must preserve template-owned multi-LoRA ordering in later items
- `reference/workflow_templates/imageEdit/Flux.2 Klein 9b KV image edit.json`
  - uses multiple chained `ReferenceLatent` nodes
  - shows ordered multi-reference latents are a first-class runtime concern, not a UI-only concern
- `reference/workflow_templates/imageEdit/Longcat image edit.json`
  - uses `FluxKontextMultiReferenceLatentMethod`
  - confirms Flux/Kontext-family edit workflows need an explicit reference-latent method seam

## Built-in node documentation facts

- `reference/docs/zh/built-in-nodes/TextEncodeQwenImageEditPlus.mdx`
  - `TextEncodeQwenImageEditPlus` supports up to three input images
  - optional VAE input produces reference latents from all supplied images
- `reference/docs/zh/built-in-nodes/ReferenceLatent.mdx`
  - `ReferenceLatent` accepts `conditioning` plus an optional `latent`
  - multiple `ReferenceLatent` nodes can be chained for multiple reference images
- `reference/docs/zh/built-in-nodes/FluxKontextMultiReferenceLatentMethod.mdx`
  - `FluxKontextMultiReferenceLatentMethod` appends a `reference_latents_method` value to conditioning
  - documented methods include `offset`, `index`, and `uxo/uno`
- `reference/ComfyUI/comfy_extras/nodes_flux.py`
  - actual node schema options are `offset`, `index`, `uxo/uno`, and `index_timestep_zero`
  - legacy `uxo` / `uso` style values normalize to the `uxo` runtime branch internally
- `reference/ComfyUI/comfy_extras/nodes_edit_model.py`
  - `ReferenceLatent` appends each latent to conditioning with `append=True`
  - chaining order is therefore semantically meaningful

## ComfyUI-EditUtils reference facts

- `reference/ComfyUI-EditUtils/README.md`
  - EditUtils explicitly targets both Qwen and Flux2Klein edit workflows
  - multi-reference image editing is treated as a first-class workflow, not an edge case
- `reference/ComfyUI-EditUtils/nodes_doc.md`
  - `QwenEditTextEncode_EditUtils` supports `image1`, `image2`, `image3`
  - `Flux2KleinEditTextEncode_EditUtils` supports `image1`, `image2`, `image3`
  - `QwenEditOutputExtractor_EditUtils` exposes:
    - `full_refs_cond`
    - `main_ref_cond`
    - `ref_latents`
    - `no_refs_cond`
  - `Flux2KleinOutputExtractor_EditUtils` exposes:
    - `ref_latents`
    - `no_refs_cond`
- `reference/ComfyUI-EditUtils/nodes.py`
  - both simplified EditUtils encoders treat the first image as the default main reference
  - both support a bounded three-image direct-input contract
  - Qwen and Flux2Klein helper seams both center on reusable config preparation plus reusable reference-latent extraction

## Implementation guidance for RookieUI

- First-wave shared foundation should preserve ordered reference assets plus explicit `main_reference_index`.
- Shared builder helpers should support:
  - ordered image loading
  - optional per-reference resize ownership
  - reusable VAE latent creation
  - ordered `ReferenceLatent` chaining
  - optional `FluxKontextMultiReferenceLatentMethod` wrapping
- `qwen_image_edit` must keep its current single-reference public behavior in `F164`; the new helper seam is for reuse, not premature adapter expansion.
- Profile-specific direct-reference limits should become enforceable from manifest truth so the backend stops silently accepting unsupported reference counts.
