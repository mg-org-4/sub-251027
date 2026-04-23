# R172 ImageEdit Live-Host Acceptance Reference

Date: 2026-04-23
Branch target: `dev`
Source of truth priority:
1. User instruction in current session
2. `.planning/ROADMAP.md`
3. Existing accepted code and validation tooling

## Objective

Record the authoritative reference facts needed to close `R172` without overstating host support.

## Key reference facts

### 1. Accepted closure contract

Phase 89 in `.planning/ROADMAP.md` defines `R172` as the final image-edit closure item and requires:

- targeted backend/frontend/live-host evidence
- truthful live-host catalog / execute proof for the asset-ready subset
- final repository SOP validation on `dev`

This means `R172` is an acceptance/closure item, not a new product-scope intake item.

### 2. Existing live host on `127.0.0.1:8188` is invalid acceptance evidence

`scripts/run_host_embedded_e2e.py --validation-mode image-edit --skip-execute` failed before lane execution because the live host fingerprint did not match the workspace fingerprint.

Observed failure:

- host fingerprint: `sha256:b5b1d91a6647827c8365c1ad9287638498db564633434147e25bd5340a0cbf40`
- workspace fingerprint: `sha256:621abb47b338762ff588a3b52c3ad3c6d3409d55a6cc3470d49ecfc43256cabf`

Existing accepted planning references also show the active `8188` process is tied to an external ComfyUI deployment rooted outside this workspace and therefore cannot be edited from this task.

### 3. Workspace-safe host baseline exists inside `reference/ComfyUI`

The workspace contains a full `reference/ComfyUI` tree with:

- `reference/ComfyUI/main.py`
- `reference/ComfyUI/folder_paths.py`
- `reference/ComfyUI/comfy/cli_args.py`

Relevant host facts:

- `reference/ComfyUI/custom_nodes` is loaded from the host base path.
- `--port` is supported.
- `--extra-model-paths-config` is supported.

This provides a workspace-safe path to start a temporary validation host without modifying the external deployment.

### 4. Official image-edit node classes are present in the reference host baseline

Reference inspection shows the required first-wave image-edit node classes already exist inside `reference/ComfyUI/comfy_extras`, including:

- `TextEncodeQwenImageEdit`
- `TextEncodeQwenImageEditPlus`
- `FluxKontextImageScale`
- `FluxKontextMultiReferenceLatentMethod`

This means first-wave image-edit validation does not require accepting a claim that `ComfyUI-EditUtils` must be installed on the live validation host for the currently shipped adapters.

`reference/ComfyUI-EditUtils` remains an implementation reference for edit-family CLIP encoding behavior, not a mandatory runtime dependency for the accepted first-wave matrix.

### 5. Asset-ready subset must be claimed narrowly

Current model inventory evidence under `A:\ComfyUI\models` supports a narrower first-wave acceptance subset than the full manifest:

Asset-ready subset confirmed by filesystem evidence:

- `qwen_image_edit`
- `qwen_image_edit_multi_lora`
- `klein_9b_kv_image_edit`
- `longcat_image_edit`

Observed supporting assets include:

- Qwen-family edit diffusion models under `A:\ComfyUI\models\diffusion_models\Qwen_Image`
- `A:\ComfyUI\models\diffusion_models\Flux\flux-2-klein-9b-kv-fp8.safetensors`
- `A:\ComfyUI\models\diffusion_models\Longcat\longcat_image_edit_bf16.safetensors`
- `A:\ComfyUI\models\text_encoders\qwen_2.5_vl_7b_fp8_scaled.safetensors`
- `A:\ComfyUI\models\text_encoders\qwen_3_8b_fp8mixed.safetensors`
- `A:\ComfyUI\models\vae\qwen_image_vae.safetensors`
- `A:\ComfyUI\models\vae\ae.safetensors`
- `A:\ComfyUI\models\vae\flux2-vae.safetensors`
- Qwen edit lightning LoRAs under `A:\ComfyUI\models\loras`

Observed not-ready or not-proven-ready for this closure:

- `flux_kontext_dev_edit`
- `flux2_image_edit`
- deferred temporal/video edit families already frozen by `R171`

`flux_kontext_dev_edit` is not part of the truthful live-host acceptance subset because the inspected external model tree did not show the expected Kontext diffusion/text-encoder asset pairing.

`flux2_image_edit` is not part of the truthful live-host acceptance subset because the inspected external model tree did not show the expected Mistral/text-encoder pairing.

### 6. Acceptance language must stay truthful

If the workspace-safe host can be started and the subset passes report + execute, `R172` may close for the asset-ready subset only.

If the workspace-safe host cannot be started with the available environment, the record must classify the live-host lane as blocked by external environment/runtime prerequisites rather than silently claiming full live-host acceptance.

## Intended execution path

1. Create a workspace-safe temporary ComfyUI host from `reference/ComfyUI`.
2. Mount the current RookieUI workspace into that host via a workspace-local custom-node link.
3. Point the reference host at `A:\ComfyUI\models` through an extra-model-paths config.
4. Run `scripts/run_host_embedded_e2e.py` in `image-edit` mode against only the asset-ready subset.
5. Re-run the full Windows repository SOP gate.
6. Update roadmap and acceptance record with the final truthful outcome.
