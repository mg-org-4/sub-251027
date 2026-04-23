# F168 Reference Summary - Img2Img ImageEdit UI Surface Integration

Date: 2026-04-23
Item: F168
Branch: `dev`

## Primary sources reviewed

- `reference/workflow_templates/imageEdit/Qwen-image edit.json`
- `reference/workflow_templates/imageEdit/Qwen-image edit-multi-lora.json`
- `reference/workflow_templates/imageEdit/Firered image edit.json`
- `reference/workflow_templates/imageEdit/Flux.1 Kontext Dev .json`
- `reference/workflow_templates/imageEdit/Flux.2 image edit.json`
- `reference/workflow_templates/imageEdit/Flux.2 Klein 9b KV image edit.json`
- `reference/workflow_templates/imageEdit/Longcat image edit.json`
- `reference/ComfyUI/comfy_extras/nodes_edit_model.py`
- `reference/ComfyUI/comfy_extras/nodes_flux.py`
- `reference/ComfyUI-EditUtils/README.md`
- `reference/ComfyUI-EditUtils/nodes.py`

## Observed implementation facts that affect RookieUI UI design

1. Official image-edit templates are not a separate public request surface.
   - RookieUI manifest entries already mark these profiles with `request_contract_surface="img2img"`.
   - The remaining split was a UI/bootstrap exposure artifact, not a backend request-type fact.

2. Official image-edit templates do not share one encoder/runtime topology.
   - `Qwen-image edit*.json` and `Longcat image edit.json` use `TextEncodeQwenImageEdit`.
   - `Firered image edit.json` uses `TextEncodeQwenImageEditPlus`.
   - `Flux.1 Kontext Dev .json`, `Flux.2 image edit.json`, and `Flux.2 Klein 9b KV image edit.json` center image ownership on `ReferenceLatent`.
   - `Longcat image edit.json` additionally sets `FluxKontextMultiReferenceLatentMethod`.
   - UI therefore must follow profile metadata rather than assuming one generic “edit mode” behavior.

3. Multiple reference images are real first-wave behavior, not future-only behavior.
   - `reference/ComfyUI/comfy_extras/nodes_edit_model.py` documents that `ReferenceLatent` can be chained for multiple reference images.
   - `reference/ComfyUI-EditUtils/README.md` explicitly documents multiple-image input support and states the simple workflow supports up to 3 images.
   - `reference/ComfyUI-EditUtils/nodes.py` exposes per-image config with `to_ref` and `ref_main_image`.

4. Main-reference selection is an explicit contract in reference implementations.
   - `reference/ComfyUI-EditUtils/nodes.py` scans configs for `ref_main_image`, enforces only one main image, and auto-falls back to the first image when none is selected.
   - RookieUI should expose ordered reference state plus a main-reference selector instead of pretending all image-edit profiles are single-source.

5. Mask input is not required for the accepted static-image edit chain.
   - User constraint for this chain: all image-edit flows do not require masks.
   - The shipped official first-wave templates above route edit ownership through source/reference images rather than inpaint-mask UX.
   - Any mask-oriented affordance on the image-edit branch would therefore be misleading.

6. First-wave truthful UI cap should stay bounded at 3 ordered references.
   - Current shipped RookieUI image-edit profiles cap at 1 or 3 direct references.
   - `ComfyUI-EditUtils` also treats 3-image simple workflows as a practical bounded UI pattern.
   - F168 can therefore expose slot 1 (primary source canvas) plus two additional ordered references without inventing an unbounded UI before it is needed.

## F168 implementation implications

- Remove the dedicated `Edit` generation subtab from the visible Img2Img mode rail.
- Expose image-edit profiles on the canonical `img2img` surface in manifest/capabilities/fallback metadata.
- Make profile selection, not mode selection, determine:
  - whether mask controls are hidden/disabled,
  - whether non-`img2img` execution modes are allowed,
  - whether ordered reference-image controls are shown,
  - how many ordered reference slots are visible.
- Serialize image-edit requests through `reference_images` plus `main_reference_index`, with slot 1 backed by the existing source image/canvas state.
