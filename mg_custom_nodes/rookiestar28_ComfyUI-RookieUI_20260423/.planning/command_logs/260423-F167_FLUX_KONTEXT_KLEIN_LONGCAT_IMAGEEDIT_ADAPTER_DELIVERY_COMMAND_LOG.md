# Command Log - F167 Flux / Kontext / Klein / Longcat ImageEdit Adapter Delivery

Date: 2026-04-23
Environment: Windows PowerShell, branch `dev`, repo-local `.venv`

## Targeted regression commands

1. `.\.venv\Scripts\python.exe -m unittest tests.test_image_edit_foundation tests.test_model_family_registry tests.test_capabilities tests.test_model_inventory tests.test_parity_matrix tests.test_img2img_translation tests.test_live_smoke_tests`
   - Result: passed
2. `npm run test:unit -- web/tests/rookieui_api.test.js web/tests/rookieui_extension.test.js web/tests/rookieui_frontend_architecture.test.js`
   - Result: passed
3. `npx playwright test tests/e2e/specs/bootstrap.spec.js`
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
   - Purpose: rerun after `.planning/ROADMAP.md` plus implementation-record synchronization so the final accepted tree is fully covered by the SOP gate
   - `detect-secrets`: pass
   - `pre-commit --all-files`: pass
   - backend unit tests: pass
   - frontend `npm run test:types`: pass
   - frontend `npm test`: pass
   - Playwright E2E: pass
   - optional host-embedded live-smoke lane: skipped because `ROOKIEUI_RUN_LIVE_SMOKE` was not enabled
