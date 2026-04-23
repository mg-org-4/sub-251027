# Official ImageEdit Re-baseline Reference

Date: 2026-04-23
Workspace: `C:\Users\Ray\Documents\我的專案\ComfyUI-RookieUI`
Purpose: replace the older "dedicated `Edit` surface + single-reference baseline" assumption with an authoritative image-edit reference synthesis grounded in the current `reference/workflow_templates/imageEdit` set, host-side ComfyUI nodes, and the newly added `reference/ComfyUI-EditUtils` project.

## 1. Current accepted RookieUI baseline that is now too narrow

Current shipped facts in the accepted repo:

- `rookieui/contracts/family_template_manifest.py`
  - `qwen_image_edit` is the only shipped official image-edit profile.
  - `flow_kind="edit"` and `available_surface_flows=("edit",)` still place it on a dedicated surface.
  - notes explicitly say the current shipped path is single-reference only.
- `rookieui/services/img2img.py`
  - `execution_mode == "edit"` still maps to a dedicated `edit` surface gate.
- `rookieui/services/workflow_builders/non_sd_templates.py`
  - `build_non_sd_edit_workflow()` only supports `qwen_image_edit`.
  - `_build_qwen_image_edit_workflow()` only consumes one source image and one template-owned LoRA default.
- `web/sidebar_tabs/rookieui_img2img_pane.js`
  - `Edit` is still rendered as a distinct visible mode.
  - the UI still partitions `img2img` presets vs `edit` presets.

Practical implication:

- the accepted runtime/UI contract still reflects phase-79 assumptions rather than the current authoritative image-edit template inventory.

## 2. Authoritative official image-edit template inventory now present

Current official templates under `reference/workflow_templates/imageEdit/`:

- `Chrono Edit 14B.json`
- `Firered image edit.json`
- `Flux.1 Kontext Dev .json`
- `Flux.2 image edit.json`
- `Flux.2 Klein 9b KV image edit.json`
- `Longcat image edit.json`
- `Qwen-image edit.json`
- `Qwen-image edit-multi-lora.json`

Practical implication:

- RookieUI is no longer planning against one explicit edit template.
- The authoritative host-side inventory now spans multiple runtime families and topologies.

## 3. Common graph facts shared across the reviewed official image-edit templates

Shared facts from the current official image-edit set:

- they are all image-input workflows and therefore belong to RookieUI's image-input chain rather than txt2img
- none of the reviewed official templates require a user mask to operate
- multiple reference images are first-class in the official graph set, not an exotic extension case

Planning rule derived from the current user instruction plus template evidence:

- all image-edit workflows must be treated as `img2img` subtypes
- image-edit workflows must not require a mask
- multiple reference images must be considered a normal supported contract on this chain

## 4. Official family groups by real runtime topology

### 4.1 Qwen-family edit group

Templates:

- `Qwen-image edit.json`
- `Qwen-image edit-multi-lora.json`
- `Firered image edit.json`

Key facts:

- `Qwen-image edit.json`
  - uses `CLIPLoader(type="qwen_image")`
  - uses `TextEncodeQwenImageEdit`
  - uses `VAEEncode`
  - uses one template-owned `LoraLoaderModelOnly`
- `Qwen-image edit-multi-lora.json`
  - preserves the same Qwen edit seam
  - adds a chained multi-template-LoRA path instead of a single template-owned LoRA
- `Firered image edit.json`
  - uses `TextEncodeQwenImageEditPlus`
  - preserves image-edit semantics but changes encoder class and template-owned LoRA wiring

Practical implication:

- RookieUI needs a Qwen-family adapter group, not a single hard-coded `qwen_image_edit` builder.
- Template-owned LoRA handling must support ordered multi-LoRA chains.

### 4.2 Flux / Kontext / Klein / Longcat static-image edit group

Templates:

- `Flux.1 Kontext Dev .json`
- `Flux.2 image edit.json`
- `Flux.2 Klein 9b KV image edit.json`
- `Longcat image edit.json`

Key facts:

- `Flux.2 image edit.json`
  - uses `ReferenceLatent`
  - uses advanced sampler/scheduler nodes rather than the basic Qwen builder seam
- `Flux.2 Klein 9b KV image edit.json`
  - uses multiple input images
  - chains `ReferenceLatent`
  - adds `FluxKVCache`
- `Flux.1 Kontext Dev .json`
  - uses multiple input images
  - uses `ImageStitch`
  - uses `FluxKontextImageScale`
- `Longcat image edit.json`
  - uses Qwen-style text encoding but still routes image ownership through Flux-family latent/reference mechanics

Practical implication:

- these templates need a shared multi-reference latent/context helper layer before per-profile adapters land.

### 4.3 Temporal / video-like edit group

Template:

- `Chrono Edit 14B.json`

Key facts:

- uses `ScaleROPE`
- uses `CLIPVisionLoader` / `CLIPVisionEncode`
- uses `WanImageToVideo`
- diverges materially from the first-wave static image-edit templates

Practical implication:

- this template should be explicitly deferred from the first-wave static image-edit rollout instead of being silently omitted later.

## 5. Host-side ComfyUI node facts relevant to implementation

Primary node references:

- `reference/ComfyUI/comfy_extras/nodes_qwen.py`
  - contains `TextEncodeQwenImageEdit`
  - contains `TextEncodeQwenImageEditPlus`
- `reference/ComfyUI/comfy_extras/nodes_edit_model.py`
  - contains `ReferenceLatent`
- `reference/ComfyUI/comfy_extras/nodes_flux.py`
  - contains `FluxKontextImageScale`
  - contains `FluxKontextMultiReferenceLatentMethod`
  - contains `FluxKVCache`
- `reference/ComfyUI/comfy_extras/nodes_rope.py`
  - contains `ScaleROPE`
- `reference/ComfyUI/comfy_extras/nodes_wan.py`
  - contains `WanImageToVideo`

Practical implication:

- RookieUI should prefer host-native execution seams that align to these nodes rather than inventing parallel semantics.

## 6. `ComfyUI-EditUtils` reference value and limits

Primary files:

- `reference/ComfyUI-EditUtils/README.md`
- `reference/ComfyUI-EditUtils/nodes.py`
- `reference/ComfyUI-EditUtils/nodes_doc.md`

Key facts:

- the project explicitly targets advanced image editing for both Qwen and Flux2Klein
- it provides a unified `EditTextEncode_EditUtils` interface plus model-specific helpers
- the simplified Qwen / Flux2Klein nodes support up to three direct image inputs
- the project documents a config-driven path that can scale beyond three images by chaining configs

Planning rule:

- `ComfyUI-EditUtils` is a strong implementation reference for conditioning ergonomics, multi-image encoder design, and config decomposition
- it must not override official workflow-template topology when the official graphs disagree

## 7. Concrete re-baseline rules for RookieUI planning

### 7.1 Flow classification rule

- all image-edit workflows are `img2img`-owned image-input flows
- they must no longer be modeled as a separate public `Edit` surface

### 7.2 Mask rule

- image-edit flows do not require mask input
- inpaint-only mask contracts must stay scoped to the inpaint branch

### 7.3 Reference-image rule

- multiple reference images are a first-class contract on this chain
- first-wave UI may cap direct inputs, but the internal contract must not assume one image forever

### 7.4 Adapter-family rule

- later implementation must be grouped by real runtime topology:
  - Qwen / Qwen+
  - Flux / Kontext / Klein / Longcat
  - Chrono / Wan temporal edit (deferred)

## 8. Reference list for roadmap / plan citation

- `reference/workflow_templates/imageEdit/Chrono Edit 14B.json`
- `reference/workflow_templates/imageEdit/Firered image edit.json`
- `reference/workflow_templates/imageEdit/Flux.1 Kontext Dev .json`
- `reference/workflow_templates/imageEdit/Flux.2 image edit.json`
- `reference/workflow_templates/imageEdit/Flux.2 Klein 9b KV image edit.json`
- `reference/workflow_templates/imageEdit/Longcat image edit.json`
- `reference/workflow_templates/imageEdit/Qwen-image edit.json`
- `reference/workflow_templates/imageEdit/Qwen-image edit-multi-lora.json`
- `reference/ComfyUI/comfy_extras/nodes_qwen.py`
- `reference/ComfyUI/comfy_extras/nodes_edit_model.py`
- `reference/ComfyUI/comfy_extras/nodes_flux.py`
- `reference/ComfyUI/comfy_extras/nodes_rope.py`
- `reference/ComfyUI/comfy_extras/nodes_wan.py`
- `reference/ComfyUI-EditUtils/README.md`
- `reference/ComfyUI-EditUtils/nodes.py`
- `reference/ComfyUI-EditUtils/nodes_doc.md`
