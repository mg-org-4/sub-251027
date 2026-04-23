# Reference Memo - Qwen / Qwen+ ImageEdit Runtime Adapter Expansion

Date: 2026-04-23
Scope: `F165` official Qwen-family image-edit adapters

## Authoritative workflow-template facts

- `reference/workflow_templates/imageEdit/Qwen-image edit.json`
  - uses `TextEncodeQwenImageEdit`
  - uses one direct source image and no user mask
  - uses `qwen_image_edit_fp8_e4m3fn.safetensors`
  - uses `qwen_2.5_vl_7b_fp8_scaled.safetensors`
  - uses `qwen_image_vae.safetensors`
  - applies one template-owned `LoraLoaderModelOnly` with `Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors`
  - sampler defaults stay on `steps=4`, `cfg=1`, `sampler=euler`, `scheduler=simple`, `shift=3.0`
- `reference/workflow_templates/imageEdit/Qwen-image edit-multi-lora.json`
  - keeps the same Qwen edit UNet, CLIP, VAE, and encoder class as the base Qwen edit template
  - stacks three `LoraLoaderModelOnly` nodes in sequence before `ModelSamplingAuraFlow`
  - all three template-owned LoRA nodes use the same official file: `Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors`
- `reference/workflow_templates/imageEdit/Firered image edit.json`
  - uses `TextEncodeQwenImageEditPlus`
  - official template only wires `image1`, but the host node supports `image1`, `image2`, `image3`
  - uses `FireRed-Image-Edit-1.1-transformer.safetensors`
  - uses `qwen_2.5_vl_7b_fp8_scaled.safetensors`
  - uses `qwen_image_vae.safetensors`
  - includes an optional lightning branch with `FireRed-Image-Edit-1.0-Lightning-8steps-v1.0.safetensors`
  - template defaults split into two objective lanes:
    - base lane: `steps=40`, `cfg=4`, no template LoRA
    - lightning lane: `steps=8`, `cfg=1`, one template-owned model-only LoRA
  - sampler defaults stay on `sampler=euler`, `scheduler=simple`, `shift=3.1`

## Built-in host-node facts

- `reference/ComfyUI/comfy_extras/nodes_qwen.py`
  - `TextEncodeQwenImageEdit`
    - accepts `clip`, `prompt`, optional `vae`, optional `image`
    - internally rescales the image to roughly `1024 * 1024` total pixels for conditioning
    - appends one `reference_latents` entry when `vae` is supplied
  - `TextEncodeQwenImageEditPlus`
    - accepts `clip`, `prompt`, optional `vae`, and optional `image1`, `image2`, `image3`
    - internally rescales VL images to roughly `384 * 384` total pixels
    - when `vae` is supplied, also builds reference latents for all supplied images at a rounded `1024 * 1024` latent scale
    - appends all reference latents with `append=True`
- `reference/docs/built-in-nodes/TextEncodeQwenImageEditPlus.mdx`
  - documents direct support for up to three input images
  - explicitly states that reference latents are emitted for all supplied images when `vae` is present

## ComfyUI-EditUtils reference facts

- `reference/ComfyUI-EditUtils/nodes.py`
  - the Qwen-family helper path treats ordered direct images as a first-class input list
  - the first image remains the default main image unless a workflow chooses another main-reference index
  - multi-image edit helpers expose reusable reference-latent and no-reference conditioning outputs rather than hiding them inside ad-hoc graphs
- `reference/ComfyUI-EditUtils/nodes_doc.md`
  - Qwen edit helper nodes document direct `image1` / `image2` / `image3` ownership
  - confirms that bounded three-image direct input is a strong first-wave contract for Qwen+ style edit models

## Implementation guidance for RookieUI

- `qwen_image_edit` remains the base single-image lane.
- `qwen_image_edit_multi_lora` should be a distinct shipped profile because the official template objectively changes template-owned LoRA chain depth.
- FireRed should ship as two bounded manifest-backed lanes:
  - base profile without template-owned LoRA
  - lightning profile with the official one-LoRA branch defaults
- FireRed family adapters should use `TextEncodeQwenImageEditPlus` and a truthful direct-reference cap of `3`.
- All Qwen-family image-edit profiles remain:
  - `img2img` request-contract owned
  - maskless
  - host-prerequisite truthful for required diffusion model, VAE, text encoder, and template-owned LoRA where applicable
