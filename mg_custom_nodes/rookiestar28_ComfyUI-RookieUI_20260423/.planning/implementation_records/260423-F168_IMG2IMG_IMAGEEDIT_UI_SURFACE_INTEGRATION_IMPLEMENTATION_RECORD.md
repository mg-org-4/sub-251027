# Phase 88 Implementation Record - Img2Img ImageEdit UI Surface Integration

## What Changed

- Updated the image-edit public contract to live on the accepted `img2img` surface instead of a separate `edit` lane:
  - `rookieui/contracts/family_template_manifest.py`
  - `web/rookieui_api.js`
  - `tests/test_model_family_registry.py`
  - `tests/test_capabilities.py`
  - `web/tests/rookieui_api.test.js`
- Folded first-wave image-edit UX into the img2img shell and pane:
  - `web/rookieui_sidebar_shell.js`
  - `web/sidebar_tabs/rookieui_img2img_pane.js`
  - `web/sidebar_tabs/rookieui_img2img_mask_editor.js`
  - `web/sidebar_tabs/rookieui_img2img_mode_router.js`
  - removed the public `Edit` mode button
  - added ordered multi-reference inputs and main-reference selection
  - hid and disabled mask-only affordances for image-edit profiles
  - validated source/reference payload rules before submit
- Extracted and reused shared helper seams for profile lookup plus ordered img2img references:
  - `web/rookieui_sidebar_shell_utils.js`
  - `web/rookieui_sidebar_shell_deps.js`
  - `web/sidebar_tabs/rookieui_txt2img_pane.js` and `web/sidebar_tabs/rookieui_img2img_pane.js` now consume the same injected helper seam
- Fixed the post-refactor frontend cache fingerprint so the shipped asset token matches the current module graph:
  - `web/rookieui_asset_revision.js`
- Expanded regression coverage around the merged surface:
  - `web/tests/rookieui_extension.test.js`
  - `web/tests/rookieui_modularization_regression.test.js`
  - `web/tests/rookieui_img2img_mode_router.test.js`
  - `tests/e2e/specs/bootstrap.spec.js`
- Synchronized planning artifacts:
  - `.planning/ROADMAP.md`
  - `.planning/references/260423-F168_IMG2IMG_IMAGEEDIT_UI_SURFACE_INTEGRATION_REFERENCE.md`
  - `.planning/plans/260423-F168_IMG2IMG_IMAGEEDIT_UI_SURFACE_INTEGRATION_PLAN.md`
  - `.planning/command_logs/260423-F168_IMG2IMG_IMAGEEDIT_UI_SURFACE_INTEGRATION_COMMAND_LOG.md`

## Why Changed

- The accepted roadmap direction for image-edit is:
  - no mask
  - img2img-owned surface
  - multi-reference capable by default
- `F168` was the phase that had to make that public contract real in the shipped UI, not just in backend adapters.
- The first full-gate run also exposed a concrete modularization regression:
  - the helper extraction from `rookieui_sidebar_shell.js` was incomplete because the barrel module did not re-export `buildProfileLookup`
- The second full-gate run exposed the standard frontend fingerprint tripwire:
  - the shipped module graph changed after the seam fix, but the asset revision token had not been updated yet
- This item closes both the planned feature work and the real regression fixes needed to get the integrated tree back to SOP-green.

## Full Verification Evidence

- Date/environment: 2026-04-23, Windows PowerShell, branch `dev`, repo-local `.venv`
- Command log reference:
  - `.planning/command_logs/260423-F168_IMG2IMG_IMAGEEDIT_UI_SURFACE_INTEGRATION_COMMAND_LOG.md`

### Pre-fix reproduction evidence

- `powershell -File scripts/run_full_tests_windows.ps1`
  - failed with `buildProfileLookup is not a function` in `web/tests/rookieui_extension.test.js` and `web/tests/rookieui_modularization_regression.test.js`
- `powershell -File scripts/run_full_tests_windows.ps1`
  - failed with frontend asset fingerprint mismatch in `web/tests/rookieui_frontend_architecture.test.js`

### Targeted regression evidence

- `.\.venv\Scripts\python.exe -m unittest tests.test_model_family_registry tests.test_capabilities`
  - passed
- `npm run test:unit -- web/tests/rookieui_api.test.js web/tests/rookieui_extension.test.js`
  - passed
- `npx playwright test tests/e2e/specs/bootstrap.spec.js`
  - passed
- `npm run test:unit -- web/tests/rookieui_frontend_architecture.test.js`
  - passed
- `npm run test:unit -- web/tests/rookieui_extension.test.js web/tests/rookieui_modularization_regression.test.js web/tests/rookieui_api.test.js web/tests/rookieui_img2img_mode_router.test.js`
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
- `powershell -File scripts/run_full_tests_windows.ps1`
  - passed
  - rerun after `.planning` command log / implementation record synchronization so the final accepted tree is covered by the same SOP gate
  - `detect-secrets`: pass
  - `pre-commit --all-files`: pass
  - backend unit tests: pass
  - frontend `npm run test:types`: pass
  - frontend `npm test`: pass
  - Playwright E2E: pass
  - optional host-embedded live-smoke lane: skipped because `ROOKIEUI_RUN_LIVE_SMOKE` was not enabled

## Known Limitations

- The current shipped image-edit surface exposes up to three ordered direct references because that is the accepted first-wave UI contract; broader multi-reference counts remain later backlog work.
- Live-host proof for each image-edit profile remains a later acceptance concern for `F169` and `R172`; this item closes the repo-side surface integration, not the external asset-readiness problem.

## Follow-up Items

- `F169` must extend fixture, smoke, and regression coverage around the merged img2img image-edit surface.
- `R172` must close the chain with live-host acceptance evidence plus another full SOP sweep on the final integrated tree.
