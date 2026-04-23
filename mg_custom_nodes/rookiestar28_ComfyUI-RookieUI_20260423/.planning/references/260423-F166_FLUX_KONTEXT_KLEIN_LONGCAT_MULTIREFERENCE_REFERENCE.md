# Reference Memo - Flux / Kontext / Klein / Longcat Multi-Reference Foundation

Date: 2026-04-23
Scope: `F166` shared image-edit foundation for Flux-family adapters

## Authoritative workflow-template facts

- `reference/workflow_templates/imageEdit/Flux.1 Kontext Dev .json`
  - uses two direct `LoadImage` nodes, then `ImageStitch`, then `FluxKontextImageScale`
  - encodes the stitched/scaled image with `VAEEncode`
  - applies one `ReferenceLatent` onto positive `CLIPTextEncode` conditioning
  - derives the negative branch from `ConditioningZeroOut`
  - uses classic `KSampler`, not the Flux2 advanced sampler stack
  - does not use a user mask
- `reference/workflow_templates/imageEdit/Flux.2 image edit.json`
  - uses one direct image
  - rescales the reference with `ImageScaleToTotalPixels`
  - derives latent canvas size from `GetImageSize` plus `EmptyFlux2LatentImage`
  - encodes the reference image with `VAEEncode`
  - applies one `ReferenceLatent` after `FluxGuidance`
  - uses `RandomNoise` + `KSamplerSelect` + `Flux2Scheduler` + `SamplerCustomAdvanced`
  - uses `BasicGuider`, not `CFGGuider`
  - includes an optional template-owned `Flux_2-Turbo-LoRA_comfyui.safetensors` switch branch
- `reference/workflow_templates/imageEdit/Flux.2 Klein 9b KV image edit.json`
  - uses two direct images and rescales both with `ImageScaleToTotalPixels`
  - builds `GetImageSize` + `EmptyFlux2LatentImage` from the main reference image
  - encodes both references with `VAEEncode`
  - mirrors chained `ReferenceLatent` application onto both positive and negative conditioning branches
  - patches the model through `FluxKVCache` before sampling
  - uses the same Flux2 advanced sampler stack as `Flux.2 image edit`
  - uses `CFGGuider`, not `BasicGuider`
- `reference/workflow_templates/imageEdit/Longcat image edit.json`
  - uses `TextEncodeQwenImageEdit` for both positive and negative conditioning
  - applies `FluxGuidance` to both branches
  - applies `FluxKontextMultiReferenceLatentMethod` to both branches after guidance
  - uses classic `KSampler`
  - does not add standalone `ReferenceLatent` nodes because the Qwen edit encoder already owns the reference-latent payload

## Built-in host-node facts

- `reference/ComfyUI/comfy_extras/nodes_edit_model.py`
  - `ReferenceLatent` appends `reference_latents` with `append=True`
  - multiple reference images are represented by chaining `ReferenceLatent` nodes in order
- `reference/ComfyUI/comfy_extras/nodes_flux.py`
  - `FluxKontextImageScale` resizes to a preferred Kontext resolution set
  - `FluxKontextMultiReferenceLatentMethod` supports `offset`, `index`, `uxo/uno`, and `index_timestep_zero`
  - `FluxKVCache` patches the model and sets default ref method to `index_timestep_zero`
  - `Flux2Scheduler` computes sigmas from width, height, and step count
- `reference/ComfyUI/comfy_extras/nodes_images.py`
  - `ImageStitch` is the upstream node for horizontal/vertical concatenation and already supports chained pairwise composition

## ComfyUI-EditUtils reference facts

- `reference/ComfyUI-EditUtils/nodes.py`
  - `Flux2KleinEditTextEncode_EditUtils` accepts ordered `image1` / `image2` / `image3`
  - the first image is the default main reference unless the config marks another main image
  - the helper uses one ordered config list rather than special-casing each template
  - `ref_longest_edge` is the shared scaling knob for Klein-style multi-reference encode preparation
  - masks remain optional helper inputs, not required workflow ownership

## Implementation guidance for RookieUI

- Keep `image_edit_foundation.py` as the single shared seam for ordered reference-image loading and Flux-family edit helper nodes.
- Add shared helpers for:
  - pairwise `ImageStitch` chaining
  - `FluxKontextImageScale`
  - mirrored `ReferenceLatent` chaining across positive/negative branches
  - `FluxKVCache`
  - Flux2 advanced sampler assembly (`GetImageSize` / `EmptyFlux2LatentImage` / `RandomNoise` / `KSamplerSelect` / `Flux2Scheduler` / guider selection / `SamplerCustomAdvanced`)
- Preserve ordered reference ownership and main-reference indexing from `F164`; do not invent a second image-order contract for Flux-family templates.
- Do not expose new public profiles in `F166`; adapter delivery belongs to `F167`.
