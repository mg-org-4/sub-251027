# Command Log - F168 Img2Img ImageEdit UI Surface Integration

Date: 2026-04-23
Environment: Windows PowerShell, branch `dev`, repo-local `.venv`

## Pre-fix reproduction commands

1. `powershell -File scripts/run_full_tests_windows.ps1`
   - Result: failed
   - Failure summary:
     - `web/tests/rookieui_extension.test.js`: `buildProfileLookup is not a function`
     - `web/tests/rookieui_modularization_regression.test.js`: `buildProfileLookup is not a function`
   - Root cause confirmed during follow-up inspection:
     - `web/rookieui_sidebar_shell_deps.js` did not re-export the newly extracted `buildProfileLookup` helper from `web/rookieui_sidebar_shell_utils.js`
2. `powershell -File scripts/run_full_tests_windows.ps1`
   - Result: failed
   - Failure summary:
     - `web/tests/rookieui_frontend_architecture.test.js`: asset revision fingerprint mismatch
   - Root cause confirmed during follow-up inspection:
     - shipped frontend fingerprint advanced to `h4f7d746de7`, but `web/rookieui_asset_revision.js` still pinned `h44445a4fbe`

## Targeted regression commands

1. `.\.venv\Scripts\python.exe -m unittest tests.test_model_family_registry tests.test_capabilities`
   - Result: passed
2. `npm run test:unit -- web/tests/rookieui_api.test.js web/tests/rookieui_extension.test.js`
   - Result: passed
3. `npx playwright test tests/e2e/specs/bootstrap.spec.js`
   - Result: passed
4. `npm run test:unit -- web/tests/rookieui_frontend_architecture.test.js`
   - Result: passed
5. `npm run test:unit -- web/tests/rookieui_extension.test.js web/tests/rookieui_modularization_regression.test.js web/tests/rookieui_api.test.js web/tests/rookieui_img2img_mode_router.test.js`
   - Result: passed

## Final full-gate command

1. `powershell -File scripts/run_full_tests_windows.ps1`
   - Result: passed
   - `detect-secrets`: pass
   - `pre-commit --all-files`: pass
   - backend unit tests: pass
   - frontend `npm run test:types`: pass
   - frontend `npm test`: pass
   - Playwright E2E: pass
   - optional host-embedded live-smoke lane: skipped because `ROOKIEUI_RUN_LIVE_SMOKE` was not enabled
2. `powershell -File scripts/run_full_tests_windows.ps1`
   - Result: passed
   - Purpose: rerun after `.planning` command log / implementation record synchronization so the final accepted tree is fully covered by the SOP gate
   - `detect-secrets`: pass
   - `pre-commit --all-files`: pass
   - backend unit tests: pass
   - frontend `npm run test:types`: pass
   - frontend `npm test`: pass
   - Playwright E2E: pass
   - optional host-embedded live-smoke lane: skipped because `ROOKIEUI_RUN_LIVE_SMOKE` was not enabled
