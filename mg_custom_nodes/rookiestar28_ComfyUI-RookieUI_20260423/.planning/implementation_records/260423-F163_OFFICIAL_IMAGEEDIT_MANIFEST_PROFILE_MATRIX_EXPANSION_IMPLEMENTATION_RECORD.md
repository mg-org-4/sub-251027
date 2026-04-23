# Phase 85 Implementation Record - Official ImageEdit Manifest/Profile Matrix Expansion

## What Changed

- Updated `rookieui/contracts/family_template_manifest.py`
  - bumped the model-family registry contract version to `f163-20260423`
  - added explicit image-edit metadata fields to `FamilyTemplateManifestEntry`:
    - `image_edit_profile`
    - `request_contract_surface`
    - `reference_input_mode`
    - `max_direct_references`
    - `encoder_family`
    - `template_lora_chain_mode`
  - propagated those fields through registry and preset payload builders
  - annotated `qwen_image_edit` with the first shipped image-edit metadata set while keeping `available_surface_flows=("edit",)` unchanged for sequencing safety
- Updated `rookieui/contracts/models.py`
  - extended `PresetDefinition` so manifest-derived image-edit metadata survives preset serialization
- Updated `rookieui/services/capabilities.py`
  - extended model-family payload normalization so backend capabilities do not drop the new image-edit fields
- Updated `web/rookieui_api.js`
  - bumped the frontend fallback contract version to `f163-20260423`
  - added fallback image-edit metadata for `qwen_image_edit`
  - added default registry/preset fallback fields for the new manifest contract
- Updated `web/rookieui_asset_revision.js`
  - refreshed the shipped frontend asset revision token after the fallback bootstrap contract changed
- Updated regression coverage:
  - `tests/test_model_family_registry.py`
  - `tests/test_capabilities.py`
  - `web/tests/rookieui_api.test.js`

## Why Changed

- `F162` moved official image-edit requests onto the canonical backend `img2img` contract, but the manifest/profile/bootstrap layer still described image-edit almost entirely through the older dedicated `Edit` surface framing.
- Without explicit manifest-backed metadata, later items would have needed to infer image-edit behavior from profile ids or ad-hoc frontend fallbacks, which would reopen the same parallel-truth problem phase 59 deliberately removed.
- `F163` makes the image-edit profile matrix explicit without prematurely changing public UI exposure; later runtime and UI work can now consume one consistent source of truth.

## Full Verification Evidence

- Date/environment: 2026-04-23, Windows PowerShell, repo-local `.venv`, branch `dev`
- Command log reference:
  - `.planning/command_logs/260423-F163_OFFICIAL_IMAGEEDIT_MANIFEST_PROFILE_MATRIX_EXPANSION_COMMAND_LOG.md`

### Targeted contract proof

- `.venv\Scripts\python.exe -m unittest tests.test_model_family_registry tests.test_capabilities tests.test_img2img_translation`
  - passed
- `npm run test:unit -- web/tests/rookieui_api.test.js`
  - passed

### Intermediate regression caught and fixed within this item

- `powershell -File scripts/run_full_tests_windows.ps1`
  - initial run failed
  - root cause: `web/rookieui_asset_revision.js` still pointed at the pre-change shipped frontend fingerprint, so `web/tests/rookieui_frontend_architecture.test.js` correctly rejected the stale cache-busting token
- `npm run test:unit -- web/tests/rookieui_frontend_architecture.test.js web/tests/rookieui_api.test.js`
  - passed after refreshing the asset revision token

### Final full-gate evidence

- `powershell -File scripts/run_full_tests_windows.ps1`
  - passed
  - `detect-secrets`: pass
  - `pre-commit --all-files`: pass
  - backend unit tests: pass
  - frontend `npm run test:types`: pass
  - frontend `npm test`: pass
  - Playwright E2E: pass

## Known Limitations

- Public `available_surface_flows` remain unchanged in this item, so `qwen_image_edit` still advertises the dedicated `edit` surface until `F168` rewires the UI exposure.
- Only the currently shipped `qwen_image_edit` lane is annotated with first-wave image-edit metadata here; broader adapter families will be added by later runtime delivery items.
- This item is metadata/bootstrap focused, so it does not provide new live-host execute evidence by itself.

## Follow-up Items

- `F164` must land the shared multi-reference image-edit conditioning foundation that consumes the new manifest metadata.
- `F165-F167` must extend the per-family runtime adapters now that encoder and reference-count truth is explicit.
- `F168` must later align the public frontend surface with the canonical backend `img2img` ownership already established in `F162`.
