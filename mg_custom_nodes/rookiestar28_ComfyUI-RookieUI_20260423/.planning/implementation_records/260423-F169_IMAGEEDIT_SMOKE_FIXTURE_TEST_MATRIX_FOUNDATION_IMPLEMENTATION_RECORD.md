# Implementation Record - ImageEdit Smoke / Fixture / Test Matrix Foundation

## What Changed

1. Extended `scripts/run_live_smoke_tests.py` with a dedicated `image-edit` validation lane.
   - Added manifest-driven default profile selection for the first-wave image-edit matrix.
   - Replaced legacy single-image edit smoke payloads with canonical ordered `reference_images` plus `main_reference_index`.
   - Added dry-run validation for:
     - `workflow_kind`
     - canonical public `img2img` mode with runtime `execution_mode="edit"`
     - normalized reference count and main reference slot
     - no-mask topology
     - template-owned `LoraLoaderModelOnly` depth

2. Expanded `tests/test_live_smoke_tests.py`.
   - Pinned the new `image-edit` default profile order.
   - Added ordered multi-reference payload assertions for `flux_kontext_dev_edit`.
   - Added qwen multi-LoRA catalog and dry-run coverage.
   - Added execute-lane assertions proving image-edit submissions route through `/rookieui/generate/img2img` without mask payloads.

3. Tightened frontend image-edit serialization.
   - `web/rookieui_sidebar_shell.js` now deletes stale `mask_asset` / `mask_data` before sending image-edit requests so previous inpaint state cannot leak into edit submissions.

4. Expanded frontend regression proof.
   - `web/tests/rookieui_extension.test.js` now includes a real multi-reference `flux_kontext_dev_edit` submit assertion.
   - The unit-test bootstrap fixture was extended with the minimum `flux_kontext_dev_edit` model/preset/profile metadata required to exercise the shipped multi-reference path.
   - `tests/e2e/specs/bootstrap.spec.js` now captures and asserts the ordered multi-reference image-edit request payload in the browser harness.

5. Bumped `web/rookieui_asset_revision.js`.
   - Updated the shipped frontend revision token after the `F169` frontend changes so the asset fingerprint tripwire and cache-busting token remain aligned.

## Why Changed

`F168` shipped image-edit on the canonical `img2img` surface, but the repository still lacked direct smoke/fixture proof for the most failure-prone parts of the new contract:

- ordered multi-reference serialization
- non-zero main-reference selection
- template-owned multi-LoRA chain depth
- no-mask image-edit payload hygiene
- truthful first-wave image-edit readiness coverage in the live-smoke runner

`F169` closes that proof gap so the final `R172` acceptance item can focus on truthful live-host evidence instead of inventing the first reusable regression foundation there.

## Full Verification Evidence

Date: 2026-04-23
Environment: Windows (PowerShell), repo-local `.venv`, repository-managed Node/Playwright toolchain
Command log: `.planning/command_logs/260423-F169_IMAGEEDIT_SMOKE_FIXTURE_TEST_MATRIX_FOUNDATION_COMMAND_LOG.md`

Targeted regression evidence:

1. `.\.venv\Scripts\python.exe -m unittest tests.test_live_smoke_tests tests.test_host_embedded_e2e`
   Result: PASS (`80` tests)

2. `npm run test:unit -- web/tests/rookieui_extension.test.js`
   Result: PASS (`7` tests)

3. `npx playwright test tests/e2e/specs/bootstrap.spec.js`
   Result: PASS (`1` spec)

Full gate evidence:

1. `powershell -File scripts/run_full_tests_windows.ps1`
   First run: FAIL because the frontend fingerprint tripwire caught a stale `ROOKIEUI_ASSET_REVISION`.

2. `powershell -File scripts/run_full_tests_windows.ps1`
   Final run: PASS
   - detect-secrets PASS
   - pre-commit all hooks PASS
   - backend unit suite PASS
   - `npm run test:types` PASS
   - `npm test` PASS

## Known Limitations

1. `F169` stops at deterministic repo-side regression and smoke foundation.
2. The optional host-embedded live-smoke lane was intentionally not promoted to acceptance evidence here; truthful live-host catalog/execute proof remains part of `R172`.

## Follow-up Items

1. Advance `R172` with restarted-host catalog / execute evidence for the asset-ready image-edit subset.
2. Use the `F169` smoke lane and frontend request assertions as the regression floor for that final acceptance closure.
