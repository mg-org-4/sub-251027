# Reference Memo - Flux / Kontext / Klein / Longcat ImageEdit Adapter Delivery

Date: 2026-04-23
Scope: `F167` first-wave bounded image-edit adapter delivery

## Authoritative workflow-template facts

- `reference/workflow_templates/imageEdit/Flux.1 Kontext Dev .json`
  - model: `flux1-dev-kontext_fp8_scaled.safetensors`
  - text encoders: `clip_l.safetensors` + `t5xxl_fp8_e4m3fn_scaled.safetensors`
  - VAE: `ae.safetensors`
  - prompt path: `CLIPTextEncode`
  - image path: ordered `LoadImage` -> `ImageStitch` -> `FluxKontextImageScale` -> `VAEEncode`
  - conditioning path: positive `ReferenceLatent` then `FluxGuidance`, negative `ConditioningZeroOut`
  - sampler defaults: `steps=20`, `cfg=1`, `sampler=euler`, `scheduler=simple`, `guidance=2.5`
- `reference/workflow_templates/imageEdit/Flux.2 image edit.json`
  - model: `flux2_dev_fp8mixed.safetensors`
  - text encoder: `mistral_3_small_flux2_bf16.safetensors`
  - VAE: `full_encoder_small_decoder.safetensors`
  - image path: `ImageScaleToTotalPixels(megapixels=1)` -> `GetImageSize` -> `EmptyFlux2LatentImage` and `VAEEncode`
  - conditioning path: `CLIPTextEncode` -> `FluxGuidance(guidance=4)` -> `ReferenceLatent`
  - sampler path: `RandomNoise` + `KSamplerSelect` + `Flux2Scheduler` + `BasicGuider` + `SamplerCustomAdvanced`
  - template includes an optional turbo-LoRA branch, but the default branch is the non-LoRA 20-step path
- `reference/workflow_templates/imageEdit/Flux.2 Klein 9b KV image edit.json`
  - model: `flux-2-klein-9b-kv-fp8.safetensors`
  - text encoder: `qwen_3_8b_fp8mixed.safetensors`
  - VAE: `flux2-vae.safetensors`
  - image path: two `ImageScaleToTotalPixels(megapixels=1)` refs, each with its own `VAEEncode`
  - conditioning path: `CLIPTextEncode` positive, `ConditioningZeroOut` negative, then mirrored chained `ReferenceLatent` application on both branches
  - model path: `FluxKVCache`
  - sampler defaults: `steps=4`, `cfg=1`, `sampler=euler`
- `reference/workflow_templates/imageEdit/Longcat image edit.json`
  - model: `longcat_image_edit_bf16.safetensors`
  - text encoder: `qwen_2.5_vl_7b_fp8_scaled.safetensors`
  - VAE: `ae.safetensors`
  - image path: `ImageScaleToTotalPixels(megapixels=1, resolution_steps=16)` -> `VAEEncode`
  - conditioning path: `TextEncodeQwenImageEdit` on positive and negative, then `FluxGuidance(guidance=4.5)` on both, then `FluxKontextMultiReferenceLatentMethod(index)` on both
  - sampler defaults: `steps=50`, `cfg=4.5`, `sampler=euler`, `scheduler=simple`

## Existing registry and inventory constraints

- `rookieui/contracts/family_template_manifest.py`
  - image-edit profile metadata already supports:
    - `request_contract_surface`
    - `reference_input_mode`
    - `max_direct_references`
    - `encoder_family`
    - `template_lora_chain_mode`
    - `runtime_adapter_id`
    - `official_template_path`
- `rookieui/services/model_inventory.py`
  - profile-specific selector resolution is driven by hint/priority matrices, so new adapters should be added by manifest metadata rather than custom inventory code
- `rookieui/services/img2img.py`
  - official image-edit profiles already normalize onto the canonical `img2img` request contract
  - direct-reference cap enforcement already comes from manifest metadata

## Delivery guidance for RookieUI

- Ship four bounded edit profiles in `F167`:
  - `flux_kontext_dev_edit`
  - `flux2_image_edit`
  - `klein_9b_kv_image_edit`
  - `longcat_image_edit`
- Keep all four on the transitional `available_surface_flows=("edit",)` until `F168` merges image-edit back into the public `img2img` UI surface.
- First-wave direct-reference limits should stay explicit:
  - `flux2_image_edit`: single-reference
  - `longcat_image_edit`: single-reference
  - `klein_9b_kv_image_edit`: bounded multi-reference
  - `flux_kontext_dev_edit`: bounded multi-reference
- Reuse `image_edit_foundation.py` for all new graph construction; do not reintroduce template-by-template helper sprawl inside `non_sd_templates.py`.
