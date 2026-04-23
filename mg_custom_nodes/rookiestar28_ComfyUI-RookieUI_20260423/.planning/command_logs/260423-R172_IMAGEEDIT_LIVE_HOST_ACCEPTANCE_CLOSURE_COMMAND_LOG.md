# R172 ImageEdit Live-Host Acceptance Closure Command Log

Date: 2026-04-23
Environment:

- OS: Windows
- Shell: PowerShell
- Repo branch: `dev`
- Repo root: `C:\Users\Ray\Documents\我的專案\ComfyUI-RookieUI`
- Validation host base URL: `http://127.0.0.1:8191`
- Validation host runtime fingerprint after restart: `sha256:bf0ed0f7bcf974374702bc8569ef3cba3adabb20d08e39c953e124b4ebb43ce0`

## Pre-fix reproduction

### 1. External deployed host rejected as stale

Command:

```powershell
.\.venv\Scripts\python.exe scripts/run_host_embedded_e2e.py --validation-mode image-edit --skip-execute
```

Result:

- failed before lane execution
- active `8188` host fingerprint did not match the workspace fingerprint
- per project policy, no outside-workspace deployment edits were made

### 2. Workspace-safe host report-only proof for the intended asset-ready subset

Command:

```powershell
.\.venv\Scripts\python.exe scripts/run_host_embedded_e2e.py --base-url http://127.0.0.1:8191 --validation-mode image-edit --profiles klein_9b_kv_image_edit,longcat_image_edit --skip-execute
```

Result:

- PASS

### 3. Workspace-safe host execute failure before the code fix

Command:

```powershell
.\.venv\Scripts\python.exe scripts/run_host_embedded_e2e.py --base-url http://127.0.0.1:8191 --validation-mode image-edit --profiles klein_9b_kv_image_edit,longcat_image_edit
```

Result:

- report pass succeeded
- execute pass failed
- host stderr showed:
  - `RookieUILoadAssetImage.VALIDATE_INPUTS() missing 1 required positional argument: 'asset_handle'`
  - `Flux2Scheduler ... steps, received_type(IMAGE) mismatch input_type(INT)`

Interpretation:

- the accepted workspace-safe host proved this was a repo-side execute drift, not an external host sync problem

## Fix validation

### 4. Targeted backend regression tests after the fix

Command:

```powershell
$env:MOLTBOT_STATE_DIR = (Resolve-Path 'moltbot_state\_local_unit').Path
.\.venv\Scripts\python.exe -m unittest tests.test_image_edit_foundation tests.test_img2img_translation
```

Result:

- PASS

### 5. Workspace-safe host restart

Representative actions:

- stopped the temporary `reference/ComfyUI` host on port `8191`
- relaunched `reference/ComfyUI` from the workspace with:
  - current repo mounted as `reference/ComfyUI/custom_nodes/comfyui-rookieui`
  - extra model paths pointed at `A:\ComfyUI\models`

Readiness proof:

- `GET http://127.0.0.1:8191/rookieui/bootstrap` returned 200
- runtime fingerprint matched the updated workspace build fingerprint

### 6. Workspace-safe host report + execute proof after the fix

Command:

```powershell
.\.venv\Scripts\python.exe scripts/run_host_embedded_e2e.py --base-url http://127.0.0.1:8191 --validation-mode image-edit --profiles klein_9b_kv_image_edit,longcat_image_edit
```

Result:

- PASS
- report lane: PASS
- execute lane: PASS

### 7. Truthful prerequisite proof for Qwen edit variants on the same restarted host

Command:

```powershell
.\.venv\Scripts\python.exe scripts/run_host_embedded_e2e.py --base-url http://127.0.0.1:8191 --validation-mode image-edit --profiles qwen_image_edit,qwen_image_edit_multi_lora --skip-execute
```

Result:

- FAIL as expected for truthful host-prerequisite classification
- reported mismatch:
  - resolved host template LoRA: `加速與功能性\Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors`
  - manifest official label: `Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors`

## Final full-gate sweep

### 8. Full Windows SOP gate with image-edit live-host lane enabled

Command:

```powershell
$env:ROOKIEUI_RUN_LIVE_SMOKE='1'
$env:ROOKIEUI_LIVE_BASE_URL='http://127.0.0.1:8191'
$env:ROOKIEUI_HOST_EMBEDDED_VALIDATION_MODE='image-edit'
$env:ROOKIEUI_LIVE_SMOKE_PROFILES='klein_9b_kv_image_edit,longcat_image_edit'
powershell -File scripts\run_full_tests_windows.ps1
```

Result:

- PASS
- `detect-secrets`: PASS
- `pre-commit --all-files`: PASS
- backend unit tests: PASS
- frontend type validation: PASS
- frontend unit tests: PASS
- Playwright E2E: PASS
- optional host-embedded image-edit lane: PASS
