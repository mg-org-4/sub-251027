# Phase 86 Implementation Record - Shared Multi-Reference ImageEdit Conditioning Foundation

## What Changed

- Added `rookieui/services/workflow_builders/image_edit_foundation.py`
  - introduced `ImageEditReferenceBundle`
  - added reusable helpers for:
    - ordered image-edit reference loading
    - optional `main_only` / `all` reference scaling
    - reusable reference VAE latent creation
    - ordered `ReferenceLatent` chaining
    - Flux multi-reference latent method wrapping and normalization
- Updated `rookieui/services/workflow_builders/non_sd_templates.py`
  - migrated `qwen_image_edit` to the new shared image-edit foundation seam
  - reused shared reference bundle plus shared latent creation instead of the previous inline single-image helper stack
- Updated `rookieui/services/img2img.py`
  - enforced manifest-backed `max_direct_references` for official image-edit profiles during normalization
  - `qwen_image_edit` now rejects multi-reference payloads that exceed its declared first-wave direct-input contract
- Updated `rookieui/contracts/extensibility.py`
  - registered `image_edit_foundation` as a workflow-builder target module
- Added/updated backend regression coverage:
  - `tests/test_image_edit_foundation.py`
  - `tests/test_img2img_translation.py`

## Why Changed

- The roadmap requires a reusable image-edit conditioning seam before family-specific adapters expand.
- Prior to this change, `qwen_image_edit` still embedded a narrow, inline single-image stack inside `non_sd_templates.py`, and the backend accepted multi-reference payload shapes that the manifest already declared unsupported for the shipped qwen path.
- `F164` creates the reusable foundation that later Qwen+ and Flux/Kontext/Klein/Longcat items can build on while keeping current public behavior stable.

## Full Verification Evidence

- Date/environment: 2026-04-23, Windows PowerShell, repo-local `.venv`, branch `dev`
- Command log reference:
  - `.planning/command_logs/260423-F164_SHARED_MULTI_REFERENCE_IMAGEEDIT_CONDITIONING_FOUNDATION_COMMAND_LOG.md`

### Targeted contract proof

- `.venv\Scripts\python.exe -m unittest tests.test_image_edit_foundation tests.test_img2img_translation tests.test_workflow_builder_modules`
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

## Known Limitations

- The new `ReferenceLatent` and Flux multi-reference method helpers are foundation-only in this item; shipped adapters beyond qwen still arrive in later items.
- `qwen_image_edit` intentionally remains a single-reference public path here; richer multi-reference qwen delivery is deferred to `F165`.
- No new live-host execute lane was added in this item because runtime-family expansion has not shipped yet.

## Follow-up Items

- `F165` must consume the shared foundation for Qwen-family multi-LoRA and Qwen-Edit-Plus style encoder expansion.
- `F166-F167` must consume the same foundation for Flux/Kontext/Klein/Longcat adapter delivery.
- `F168` must later align the public frontend surface with the already-established backend image-edit contract.
