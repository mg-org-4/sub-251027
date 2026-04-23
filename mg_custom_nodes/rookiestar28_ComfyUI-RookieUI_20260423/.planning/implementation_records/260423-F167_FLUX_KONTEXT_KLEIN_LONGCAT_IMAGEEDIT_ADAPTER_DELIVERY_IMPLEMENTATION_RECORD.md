# Phase 87 Implementation Record - Flux / Kontext / Klein / Longcat ImageEdit Adapter Delivery

## What Changed

- Expanded `rookieui/contracts/family_template_manifest.py`
  - added first-wave image-edit profiles for:
    - `flux_kontext_dev_edit`
    - `flux2_image_edit`
    - `klein_9b_kv_image_edit`
    - `longcat_image_edit`
  - updated the manifest contract version to `f167-20260423`
  - added truthful selector hints, encoder-family metadata, direct-reference caps, and official-template paths
- Expanded `rookieui/services/workflow_builders/non_sd_templates.py`
  - added dedicated runtime builders for the four new image-edit profiles
  - routed `build_non_sd_edit_workflow()` through a multi-profile adapter map instead of the previous qwen-only branch
  - matched official first-wave topology:
    - Kontext stitched references plus `FluxKontextImageScale`
    - Flux2 advanced sampler with `BasicGuider`
    - Klein KV mirrored `ReferenceLatent` chains plus `FluxKVCache`
    - Longcat `TextEncodeQwenImageEdit` plus `FluxKontextMultiReferenceLatentMethod`
- Expanded `rookieui/services/workflow_builders/image_edit_foundation.py`
  - added configurable `resolution_steps` support for `ImageScaleToTotalPixels`
  - used that new seam to match the official Longcat image-edit template scaling contract
- Expanded backend tests:
  - `tests/test_image_edit_foundation.py`
  - `tests/test_img2img_translation.py`
  - `tests/test_model_family_registry.py`
  - `tests/test_capabilities.py`
  - `tests/test_model_inventory.py`
  - `tests/test_parity_matrix.py`
  - `tests/test_live_smoke_tests.py`
- Expanded frontend and bootstrap fixtures:
  - `web/rookieui_api.js`
  - `web/tests/rookieui_api.test.js`
  - `tests/e2e/boot.mjs`
  - `tests/e2e/specs/bootstrap.spec.js`
  - `web/rookieui_asset_revision.js`
- Added/updated planning artifacts:
  - `.planning/references/260423-F167_FLUX_KONTEXT_KLEIN_LONGCAT_IMAGEEDIT_ADAPTER_REFERENCE.md`
  - `.planning/plans/260423-F167_FLUX_KONTEXT_KLEIN_LONGCAT_IMAGEEDIT_ADAPTER_DELIVERY_PLAN.md`
  - `.planning/command_logs/260423-F167_FLUX_KONTEXT_KLEIN_LONGCAT_IMAGEEDIT_ADAPTER_DELIVERY_COMMAND_LOG.md`

## Why Changed

- `F166` closed the shared Flux-family helper seam, but RookieUI still only shipped the qwen-family edit adapters.
- The authoritative image-edit templates now show a broader first-wave matrix with distinct runtime shapes that cannot be represented by the qwen-only adapter:
  - Flux Kontext stitched multi-reference edits
  - Flux2 advanced sampler edits
  - Klein KV cached multi-reference edits
  - Longcat reference-method edits
- `F167` delivers those four adapters on the accepted `img2img`-owned, no-mask image-edit contract while keeping selector truth manifest-backed instead of scattering special cases across inventory/runtime code.

## Full Verification Evidence

- Date/environment: 2026-04-23, Windows PowerShell, branch `dev`, repo-local `.venv`
- Command log reference:
  - `.planning/command_logs/260423-F167_FLUX_KONTEXT_KLEIN_LONGCAT_IMAGEEDIT_ADAPTER_DELIVERY_COMMAND_LOG.md`

### Targeted regression proof

- `.\.venv\Scripts\python.exe -m unittest tests.test_image_edit_foundation tests.test_model_family_registry tests.test_capabilities tests.test_model_inventory tests.test_parity_matrix tests.test_img2img_translation tests.test_live_smoke_tests`
  - passed
- `npm run test:unit -- web/tests/rookieui_api.test.js web/tests/rookieui_extension.test.js web/tests/rookieui_frontend_architecture.test.js`
  - passed
- `npx playwright test tests/e2e/specs/bootstrap.spec.js`
  - passed

### Final full-gate evidence

- `powershell -File scripts/run_full_tests_windows.ps1`
  - passed
  - `detect-secrets`: pass
  - `pre-commit --all-files`: pass
  - backend unit tests: pass
  - frontend `npm run test:types`: pass
  - frontend `npm test`: pass
  - Playwright E2E: pass
  - optional host-embedded live-smoke lane: skipped because `ROOKIEUI_RUN_LIVE_SMOKE` was not enabled
  - rerun after roadmap / record synchronization: passed again

## Known Limitations

- Public image-edit UX is still the transitional `Edit` lane; `F168` must fold the broader profile matrix into the accepted `img2img` surface and add first-class ordered multi-reference controls.
- The server-backed `web/tests/rookieui_extension.test.js` fixture remains intentionally narrower than the fallback matrix; it still validates the pre-`F168` shell behavior rather than the full first-wave edit catalog.
- Live-host asset readiness and execute proof for the newly shipped adapters remain a later acceptance concern for `F169` and `R172`.

## Follow-up Items

- `F168` must complete the public `img2img` UI integration for the expanded image-edit contract.
- `F169` must extend smoke/fixture coverage around the broader first-wave image-edit matrix.
- `R172` must close the chain with full live-host acceptance evidence plus another green repository-wide SOP gate.
