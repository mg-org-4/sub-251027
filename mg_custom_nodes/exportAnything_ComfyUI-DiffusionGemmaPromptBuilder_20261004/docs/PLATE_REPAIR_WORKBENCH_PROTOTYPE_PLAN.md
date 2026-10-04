# DiffusionGemma Plate Repair Workbench Prototype Plan

Status: research-backed prototype plan, not implementation proof.

Implementation target: `C:\ComfyUI\app\custom_nodes\ComfyUI-DiffusionGemma-LTX-Motion-Planner`.

Prompt Builder role: keep the stable Director Assistant surface. The Prompt Builder should supply the local DiffusionGemma NVFP4 runtime and structured decision packets, but the video repair workbench belongs in the sibling Motion Planner pack.

Research method: this plan was informed by three focused subagent probes:

- LTX/Lightricks lane: confirmed LTX-2.3 IC-LoRA In/Outpainting and mask-aware LTXVideo nodes are the best LTX-native repair primitives; Motion Brush is timeline/control support, not the raw plate-mask bridge.
- Wan/VACE lane: confirmed `WanVaceToVideo` and official VACE semantics fit masked repair plates; local VACE model availability still needs verification.
- ComfyUI glue lane: confirmed local IO, SAM3, crop/uncrop, batch composite, alpha, and pyramid-blend primitives already cover most non-model work; missing pieces are thin deterministic adapters and QA.

## Thesis

The repair loop should not behave like a serial full-video retake agent. It should behave like a small VFX plate system:

```text
Immutable source plate
-> DiffusionGemma issue inventory
-> plate plan
-> per-issue temporal masks
-> per-issue repair plates from LTX or Wan
-> deterministic masked composite over source
-> critic verifies the composite and updates issue status
```

This is scientifically plausible because it converts an underconstrained generative loop into a sequence of explicit hypotheses:

1. A defect exists at a named time window and layer.
2. The defect has a maskable target.
3. A mask covers the intended damaged pixels and not stable pixels.
4. A backend can generate a candidate for only that plate.
5. The compositor preserves unmasked source pixels.
6. The critic evaluates the composite, not a full-video regeneration chain.

The architecture should prefer visible schema-bound nodes over hidden autonomous loops. DiffusionGemma is the orchestrator; ComfyUI nodes are the subagents.

## Evidence Summary

### LTX/Lightricks

LTX provides relevant primitives for this architecture, especially through IC-LoRA in/outpainting and ComfyUI-LTXVideo utility nodes.

External evidence:

- `https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting`
  - The model card describes inpainting and outpainting on top of LTX-2.3-22B.
  - It names the training type as video-to-video, mask-conditioned IC-LoRA.
  - It uses a reference video plus binary mask, generating the masked region while leaving the unmasked area intact.
  - It describes a two-stage pipeline with boundary refinement and Laplacian blending.
  - It warns to describe only the masked/outpainted region, not the full scene.
- `https://docs.ltx.video/open-source-model/usage-guides/ic-lo-ra`
  - IC-LoRAs are task-specific, reference-driven adapters for structural control, VFX, restoration, and creative transforms.
  - The guide names `LTX Add Video IC-LoRA Guide` and `LTX Add Video IC-LoRA Guide Advanced`; the advanced path includes mask-aware conditioning.
- `https://docs.ltx.video/open-source-model/integration-tools/ltx-comfy-ui-nodes`
  - The official LTX ComfyUI node documentation names the same inpainting and blend primitives used by the local node pack.
- `https://github.com/Lightricks/ComfyUI-LTXVideo`
  - The repository hosts LTX-2 ComfyUI custom nodes and workflows, including IC-LoRA workflows and LTX-2.3 adapter assets.

Local evidence:

- `C:\ComfyUI\app\custom_nodes\ComfyUI-LTXVideo\vanish_nodes.py:8` registers `LTXVDilateVideoMask`.
- `C:\ComfyUI\app\custom_nodes\ComfyUI-LTXVideo\vanish_nodes.py:96` registers `LTXVInpaintPreprocess`.
- `C:\ComfyUI\app\custom_nodes\ComfyUI-LTXVideo\masks.py:11` registers `LTXVPreprocessMasks`.
- `C:\ComfyUI\app\custom_nodes\ComfyUI-LTXVideo\latents.py:209` registers `LTXVSetVideoLatentNoiseMasks`.
- `C:\ComfyUI\app\custom_nodes\ComfyUI-LTXVideo\pyramid_blending.py:178` registers `LTXVLaplacianPyramidBlend`.
- `C:\ComfyUI\app\custom_nodes\ComfyUI-LTXVideo\guide.py:258` registers `LTXVAddGuideAdvancedAttention`.
- `C:\ComfyUI\app\custom_nodes\ComfyUI-LTXVideo\iclora.py:261` registers `LTXAddVideoICLoRAGuideAdvanced`.
- `C:\ComfyUI\app\custom_nodes\ComfyUI-LTXVideo\example_workflows\2.3\LTX-2.3_ICLoRA_Inpaint_Two_Stage_Distilled.json` exists locally.
- `C:\ComfyUI\app\custom_nodes\ComfyUI-LTXVideo\example_workflows\2.3\LTX-2.3_ICLoRA_Outpaint_Two_Stage_Distilled.json` exists locally.
- `C:\ComfyUI\app\custom_nodes\ComfyUI-LTXVideo\example_workflows\2.3\LTX-2.3_ICLoRA_Motion_Track_Distilled.json` exists locally.
- `C:\ComfyUI\app\custom_nodes\ComfyUI-LTXVideo\example_workflows\2.3\LTX-2.3_ICLoRA_Motion_Brush_Distilled.json` exists locally.

Important limitation:

`ComfyUI-LTX-Director-Motion-Brush` is not the raw mask compositor. Motion Brush is useful for timeline and guide payloads, but the direct mask seam is in LTXVideo guide, IC-LoRA, mask, latent-mask, and blend nodes.

Keep three mask concepts separate:

- `attention_mask`: controls guide/conditioning influence.
- latent or denoise/noise mask: controls where generation may rewrite latent content.
- composite or blend mask: controls final pixel replacement over the source plate.

### Wan/VACE

Wan VACE is also a strong backend candidate for repair plates. It may be better than LTX for some masked object replacement and region-regeneration tasks.

External evidence:

- `https://github.com/ali-vilab/VACE`
  - VACE describes itself as all-in-one video creation and editing.
  - It explicitly includes reference-to-video generation, video-to-video editing, and masked video-to-video editing.
  - Its preprocessing path produces `src_video`, `src_mask`, and `src_ref_images`, which maps cleanly onto a plate manifest.
- `https://docs.comfy.org/built-in-nodes/conditioning/video-models/wan-vace-to-video`
  - The ComfyUI built-in `WanVaceToVideo` node accepts `control_video`, `control_masks`, and `reference_image`.
- `https://docs.comfy.org/tutorials/video/wan/vace`
  - ComfyUI's Wan VACE page describes VACE as supporting text, images, video, masks, and control signals; it names local replacement through masks.
- `https://github.com/stuttlepress/ComfyUI-Wan-VACE-Prep`
  - Its `Wan VACE Inpaint` node prepares a control video and mask for `WanVaceToVideo.control_video` and `control_masks`, replacing masked pixels with neutral gray and preserving the mask polarity.

Local evidence:

- `C:\ComfyUI\app\comfy_extras\nodes_wan.py:285` defines `WanVaceToVideo`.
- `C:\ComfyUI\app\comfy_extras\nodes_wan.py:301` exposes `control_video`.
- `C:\ComfyUI\app\comfy_extras\nodes_wan.py:302` exposes `control_masks`.
- `C:\ComfyUI\app\comfy_extras\nodes_wan.py:303` exposes `reference_image`.
- `C:\ComfyUI\app\comfy_extras\nodes_wan.py:365` and `:366` attach `vace_frames`, `vace_mask`, and `vace_strength` into conditioning.
- `C:\ComfyUI\app\custom_nodes\comfyui_fill-nodes\nodes\wip\FL_WanVaceToVideoMultiRef.py:17` provides a local multi-reference adapter.

Important limitation:

Wan VACE should be treated as a repair backend, not as the overall autonomous planner. The plate contract should be independent of the backend so LTX and Wan can be compared on the same issue packet.

Local model status:

- Current local scan found Wan diffusion models in `C:\ComfyUI\app\models\diffusion_models`, specifically `wan2.2_bernini_r_low_noise_mxfp8.safetensors` and `wan2.2_bernini_r_high_noise_mxfp8.safetensors`.
- The same scan did not find an obvious VACE diffusion model. Wan/VACE should therefore be designed into the prototype, but LTX is the likely first runnable backend unless VACE weights are added.
- `ComfyUI-Wan-VACE-Prep` is useful prior art, but it is not installed locally. A tiny Wan prep node may be justified if we want to avoid an external custom-node dependency.

### ComfyUI Glue

The non-model pieces also mostly exist.

Local evidence:

- `C:\ComfyUI\app\custom_nodes\comfyui-videohelpersuite\videohelpersuite\load_video_nodes.py:476` provides local video loading into frame batches.
- `C:\ComfyUI\app\custom_nodes\comfyui-videohelpersuite\videohelpersuite\nodes.py:235` provides `VHS_VideoCombine`.
- `C:\ComfyUI\app\custom_nodes\comfyui-kjnodes\nodes\image_nodes.py:1962` provides `GetImageRangeFromBatch`.
- `C:\ComfyUI\app\comfy_extras\nodes_sam3.py:88` provides native SAM3 node definitions.
- `C:\ComfyUI\app\custom_nodes\ComfyUI-DiffusionGemma-LTX-Motion-Planner\nodes.py:3471` defines `DiffusionGemmaLTXQualityCritic`.
- `C:\ComfyUI\app\custom_nodes\ComfyUI-DiffusionGemma-LTX-Motion-Planner\nodes.py:3666` defines `DiffusionGemmaLTXIssueScheduler`.
- `C:\ComfyUI\app\custom_nodes\ComfyUI-DiffusionGemma-LTX-Motion-Planner\nodes.py:3771` rejects scheduled issues unless they are both time-localized and maskable.
- `C:\ComfyUI\app\custom_nodes\ComfyUI-DiffusionGemma-LTX-Motion-Planner\nodes.py:3933` defines `DiffusionGemmaLTXTimelineMaskPad`.
- `C:\ComfyUI\app\custom_nodes\comfyui_essentials\image.py:206` defines `ImageCompositeFromMaskBatch`.
- `C:\ComfyUI\app\custom_nodes\comfyui_fill-nodes\nodes\utility\FL_PasteByMask.py:65` defines `FL_PasteByMask`.
- `C:\ComfyUI\app\custom_nodes\comfyui_fill-nodes\nodes\utility\FL_VideoCropNStitch.py:6` defines `FL_VideoCropMask`.
- `C:\ComfyUI\app\custom_nodes\comfyui_fill-nodes\nodes\utility\FL_VideoCropNStitch.py:135` defines `FL_VideoRecompose`.
- `C:\ComfyUI\app\custom_nodes\comfyui-kjnodes\nodes\batchcrop_nodes.py:26` defines `BatchCropFromMask`.
- `C:\ComfyUI\app\custom_nodes\comfyui-kjnodes\nodes\batchcrop_nodes.py:156` defines `BatchUncrop`.
- `C:\ComfyUI\app\custom_nodes\comfyui-kjnodes\nodes\batchcrop_nodes.py:253` defines `BatchCropFromMaskAdvanced`.
- `C:\ComfyUI\app\custom_nodes\comfyui-kjnodes\nodes\batchcrop_nodes.py:535` defines `BatchUncropAdvanced`.

Composite caution:

`FL_VideoRecompose` is useful, but it replaces the whole crop rectangle. It should not be the scientific preservation proof. The final proof should still use an explicit mask composite over the immutable source plate.

## Proposed Node Pack Shape

### 1. DiffusionGemma Plate Critic

Can start as an extension of `DiffusionGemmaLTXQualityCritic`.

Inputs:

- original source evidence
- generated/composited candidate video
- user intent
- optional current plate manifest
- optional previous issue statuses

Outputs:

- `issue_inventory_json`
- `plate_plan_json`
- `blocked_or_advisory_reasons`
- `verification_level`

Each issue should include:

- `issue_id`
- `category`
- `severity`
- `confidence`
- `time_window`
- `target_layer`
- `z_order`
- `maskable_target`
- `sam_prompt`
- `bbox_hint`
- `repair_instruction`
- `preserve_prompt`
- `backend_preference`
- `evidence_notes`
- `status`

### 2. Plate Plan Compiler

New thin deterministic node.

Purpose:

Convert issue inventory into executable plate jobs. This is the place to enforce scientific constraints before any generator runs.

Required invariants:

- every executable job has `time_localized=true`
- every executable job has `maskable_target=true`
- every executable job references the immutable `base_plate_id`
- every executable job declares mask polarity
- every executable job declares backend and prompt scope
- no job may use a previous generated/composited video as `base_plate`

Outputs:

- `plate_manifest_json`
- `next_plate_job_json`
- `remaining_plate_jobs_json`
- `source_plate_file`
- `repair_prompt`
- `preserve_prompt`
- `backend_kind`
- `z_order`
- `composite_mode`

### 3. Mask Acquisition Lane

Reuse existing nodes:

```text
Plate job
-> Issue Scheduler or Plate Plan Compiler lane output
-> SAM3 Detect / SAM3 Video Track / SAM3 Track to Mask
-> DiffusionGemma LTX Timeline Mask Pad
-> mask processors or LTXVDilateVideoMask
```

This lane should produce:

- `window_mask`
- `timeline_mask`
- `mask_debug_json`
- `nonzero_frame_count`
- `mask_quality_flags`

### 4. Repair Backend Adapter

New thin routing node or two separate backend-specific adapter nodes.

Backend A: LTX IC-LoRA in/outpainting.

Likely node chain:

```text
base source frames + timeline mask
-> LTXVInpaintPreprocess
-> LTX IC-LoRA loader / Add Video IC-LoRA Guide Advanced
-> sampler/decode
-> repair_plate_frames
```

Alternative LTX guide path:

```text
timeline_mask
-> LTXVAddGuideAdvancedAttention.attention_mask
or LTXAddVideoICLoRAGuideAdvanced.attention_mask
```

Backend B: Wan VACE.

Likely node chain:

```text
base source frames + timeline mask
-> VACE control video prep
-> WanVaceToVideo.control_video
-> WanVaceToVideo.control_masks
-> Wan sampler/decode
-> repair_plate_frames
```

If `ComfyUI-Wan-VACE-Prep` is not installed, the first Wan adapter can be tiny:

- input `IMAGE` source frames and `MASK`
- output `control_video` where masked pixels become neutral 0.5 gray
- output `control_masks` with white indicating regenerate
- output width, height, length

Backend C: no-generator deterministic proof.

Before either LTX or Wan generation, the router should support a synthetic or passthrough repair plate. This lets tests prove mask polarity, frame alignment, and unmasked preservation without confusing generator errors with orchestration errors.

### 5. Plate Composite

Start with existing nodes:

- `ImageCompositeFromMaskBatch`
- `LTXVLaplacianPyramidBlend`
- `FL_PasteByMask`
- `FL_VideoRecompose`

Add a custom node only if existing nodes fail frame-count, z-order, or holdout semantics.

Potential new node:

`DiffusionGemmaPlateComposite`

Inputs:

- `base_frames`
- `repair_frames`
- `timeline_mask`
- `plate_manifest_json`
- `z_order`
- `feather_or_blend_mode`

Outputs:

- `composited_frames`
- `composite_debug_json`
- `unmasked_pixel_delta`
- `mask_coverage_stats`

Non-negotiable test:

Pixels outside the active mask must match the immutable source plate exactly, or within a declared tolerance if a blend operation intentionally softens boundaries.

Recommended convention:

- project mask polarity: `1 = repair/generate`, `0 = preserve source`
- exact composite: `ImageCompositeFromMaskBatch(image_from=source, image_to=repair, mask=timeline_mask)`
- LTX Laplacian blend: `image_a=repair`, `image_b=source`, `mask=timeline_mask`
- feathered/pyramid modes may allow tolerance only inside a declared dilated boundary band

### 6. Plate Verifier

Can start as a mode of `DiffusionGemmaLTXQualityCritic`.

It should evaluate:

- issue fixed or not fixed
- new artifacts introduced or not
- unmasked preservation respected or not
- mask leakage at boundary
- temporal consistency inside the repaired window
- whether remaining issues should proceed

Outputs:

- `issue_status_json`
- `fixed_issue_ids`
- `failed_issue_ids`
- `regression_issue_ids`
- `continue_next_plate`
- `verification_level`

## Minimal Prototype Workflow

### Phase 0: Deterministic Composite Proof

No generator.

Use synthetic tensors or a short loaded clip:

1. Load base video frames.
2. Create a simple moving mask or use SAM3 output.
3. Create a synthetic repair plate, for example solid color or blurred patch.
4. Composite repair over source.
5. Assert unmasked pixels are preserved.
6. Assert frame counts and mask windows match.

This proves the plate invariant before any model gets blamed.

### Phase 1: One-Issue LTX Plate

1. Source evidence from original video.
2. DiffusionGemma emits one issue with time window, layer, mask prompt, and repair instruction.
3. SAM3 generates the window mask.
4. Timeline Mask Pad expands it to full frame count.
5. LTX in/outpainting or IC-LoRA guide generates one repair plate.
6. Composite repair over immutable source.
7. Critic verifies only that issue.

Success condition:

The final output is source pixels plus masked repair, not generation two of generation one.

### Phase 2: Three Manual Sequential Lanes

Expose three lanes for `issue_index=0`, `1`, and `2`.

This keeps debugging visible and avoids pretending the automatic loop is solved.

Each lane writes:

- selected issue
- mask preview
- repair plate preview
- composite preview
- issue status

### Phase 3: LTX vs Wan Backend Comparison

For the same issue packet and same mask:

1. Generate repair plate with LTX.
2. Generate repair plate with Wan VACE.
3. Composite both over the same immutable source.
4. Critic compares composites using the same criteria.

This creates a scientific backend-selection loop. The orchestrator can later learn that LTX is better for some classes and Wan is better for others.

## Backend Selection Hypotheses

Start with these priors and let evidence update them:

- LTX IC-LoRA In-Outpainting: best first choice for masked fill, boundary-aware repair, canvas extension, and repairs that should preserve LTX timing/style.
- LTX attention-mask guide path: useful when the repair should bias conditioning but not fully replace pixels.
- Wan VACE: strong alternate for local replacement, background replacement, and larger masked video-to-video edits.
- Motion Brush: useful for timeline, retake, and motion-control payloads, but not the raw mask-to-pixel compositor.
- Generic image/composite nodes: should own final pixel preservation and plate merge whenever possible.

## Scientific Acceptance Criteria

The prototype is not accepted until these are true:

1. Source plate immutability is explicit in every plate manifest.
2. Every executable repair job has a time window and maskable target.
3. Timeline masks fail fast on frame-count mismatch.
4. Unmasked source pixels are preserved by deterministic composite.
5. Prompts for repair backends describe only the repaired region.
6. The critic updates per-issue status rather than one overall retake score.
7. A generated or composited output is never reused as the next base plate.
8. LTX and Wan backends can be tested against the same plate job.
9. Fallback or metadata-only critic reports cannot green-light execution.
10. The graph exposes intermediate plate, mask, and composite previews.

## What Not To Build

Do not build:

- a hidden all-in-one retake agent
- a new SAM implementation
- a custom compositor before testing `ImageCompositeFromMaskBatch`, `LTXVLaplacianPyramidBlend`, and `FL_PasteByMask`
- a custom Wan VACE inpaint implementation if a small control-video prep adapter is enough
- a prompt-only multi-issue repair chain that collapses layers into one dense retake prompt
- a loop where pass two treats pass one's defects as new source truth

## Required New Artifacts

Likely needed:

- `schemas/plate_plan.schema.json`
- `schemas/plate_job.schema.json`
- `schemas/plate_result.schema.json`
- `schemas/issue_status.schema.json`
- smoke test for synthetic composite preservation
- smoke test for scheduler sequential lanes
- smoke test for vague issue refusal
- smoke test for strict mask/window mismatch
- optional example workflow: `examples/layered_repair_workbench_poc.json`

Likely new nodes:

- `DiffusionGemma Source Plate Manifest`
- `DiffusionGemma LTX Plate Plan Compiler`
- `DiffusionGemma LTX Plate Job Router`
- `DiffusionGemma Plate Composite QA`
- `DiffusionGemma Plate Composite` only if existing composites do not satisfy z-order/holdout/testing needs
- `DiffusionGemma Wan VACE Prep` only if installing or relying on external Wan prep nodes is undesirable

## Open Verification Tasks

Before implementation:

1. Load the official LTX in/outpainting workflow locally and record exact node sequence.
2. Confirm local `ComfyUI-LTXVideo` in/outpainting workflows run with the installed model files, not only that the JSON exists.
3. Confirm whether local ComfyUI has native SAM3 model files available, not just node classes.
4. Confirm whether Wan VACE model files are installed and runnable in this ComfyUI environment.
5. Decide whether the first prototype should target LTX only, or LTX plus a Wan A/B backend.
6. Build a no-generator composite smoke before touching generation.

## Handoff Summary

The architecture is not vibes:

- LTX/Lightricks provides mask-conditioned in/outpainting and blend primitives.
- Wan VACE provides masked video-to-video/control-mask primitives.
- ComfyUI already provides SAM3 masks, crop/recompose, batch compositing, and LTX/Wan sockets.
- DiffusionGemma's unique role is not to invent pixels. Its role is to produce verified issue inventory, plate plans, backend routing, and critic status updates.

The next implementation should be a visible plate workbench, not a closed-loop full-video retake chain.
