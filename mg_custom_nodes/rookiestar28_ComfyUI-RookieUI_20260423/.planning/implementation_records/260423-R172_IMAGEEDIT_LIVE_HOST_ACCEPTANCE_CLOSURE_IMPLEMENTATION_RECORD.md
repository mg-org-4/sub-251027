# R172 Official ImageEdit Regression and Live-Host Acceptance Closure Implementation Record

Date: 2026-04-23
Branch: `dev`
Related command log: `.planning/command_logs/260423-R172_IMAGEEDIT_LIVE_HOST_ACCEPTANCE_CLOSURE_COMMAND_LOG.md`

## What Changed

Code-level changes:

- updated `rookieui/services/workflow_builders/image_edit_foundation.py`
  - `RookieUILoadAssetImage` workflow nodes now emit `asset_handle`, matching the declared runtime node input contract
  - `Flux2Scheduler.steps` now stays a scalar integer when the builder is given literal step counts
- expanded backend regression coverage
  - `tests/test_image_edit_foundation.py` now asserts the image-edit asset-loader input key and the scalar `Flux2Scheduler.steps` wiring
  - `tests/test_img2img_translation.py` now asserts host-compatible `asset_handle` wiring plus scalar `Flux2Scheduler.steps` in shipped image-edit workflows

Planning / governance changes:

- updated `.planning/ROADMAP.md` to close `R172`
- added a dedicated phase-89 command log and acceptance record
- added a dedicated phase-89 reference memo and plan file

## Why Changed

`R172` uncovered execute-only workflow drift that earlier translation-only and dry-run coverage did not catch.

The workspace-safe restarted host proved two concrete defects:

1. `RookieUILoadAssetImage` validation was invoked with `asset_handle`, but the generated image-edit workflow still serialized the input as `asset`.
2. `Flux2Scheduler.steps` was incorrectly serialized through the node-reference helper, turning a literal step count into an invalid linked-image input.

Those defects prevented truthful live-host execute acceptance for the otherwise asset-ready image-edit subset. The fix was therefore required to complete `R172`.

## Full Verification Evidence

Date: 2026-04-23
Environment: Windows PowerShell on `dev`
Evidence source: `.planning/command_logs/260423-R172_IMAGEEDIT_LIVE_HOST_ACCEPTANCE_CLOSURE_COMMAND_LOG.md`

Pre-fix reproduction evidence:

- stale external `8188` host failed the runtime fingerprint freshness gate
- workspace-safe `8191` host reproduced the execute bug with:
  - missing `asset_handle` on `RookieUILoadAssetImage`
  - invalid `Flux2Scheduler.steps` input type

Post-fix targeted regression evidence:

- `.\.venv\Scripts\python.exe -m unittest tests.test_image_edit_foundation tests.test_img2img_translation`
- result: PASS

Live-host acceptance evidence:

- `.\.venv\Scripts\python.exe scripts/run_host_embedded_e2e.py --base-url http://127.0.0.1:8191 --validation-mode image-edit --profiles klein_9b_kv_image_edit,longcat_image_edit`
- result: PASS
- report and execute both passed on a restarted workspace-safe host whose runtime fingerprint matched the current workspace

Truthful host-prerequisite evidence:

- `.\.venv\Scripts\python.exe scripts/run_host_embedded_e2e.py --base-url http://127.0.0.1:8191 --validation-mode image-edit --profiles qwen_image_edit,qwen_image_edit_multi_lora --skip-execute`
- result: FAIL as expected because the validation host exposed only the `2509` Qwen edit lightning label, not the manifest's official template label

Final full-gate evidence:

- `powershell -File scripts\run_full_tests_windows.ps1` with the image-edit host lane enabled via environment variables
- result: PASS

## Known Limitations

- `R172` live-host execute proof is intentionally limited to the truthful asset-ready subset proven on the validation host:
  - `klein_9b_kv_image_edit`
  - `longcat_image_edit`
- `qwen_image_edit` and `qwen_image_edit_multi_lora` remain validation-host prerequisites on this environment because the available Qwen edit lightning LoRA label drifted from the manifest's official template label.
- `R171`-deferred temporal/video edit graphs remain explicitly out of scope for this closure.

## Follow-up Items

- If the validation host later receives the exact official Qwen edit lightning asset label, rerun the image-edit host lane to expand the proven execute subset without changing the accepted repo contract.
