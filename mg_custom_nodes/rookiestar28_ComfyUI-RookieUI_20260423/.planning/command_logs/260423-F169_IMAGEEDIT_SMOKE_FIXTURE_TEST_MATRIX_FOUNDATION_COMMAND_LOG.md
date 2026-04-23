# Command Log - ImageEdit Smoke / Fixture / Test Matrix Foundation

Date: 2026-04-23
Environment: Windows (PowerShell), repo-local `.venv`, Node/Playwright via repository SOP

## Targeted Regression Commands

1. `.\.venv\Scripts\python.exe -m unittest tests.test_live_smoke_tests`
   Result: PASS (`76` tests)

2. `npm run test:unit -- web/tests/rookieui_extension.test.js`
   Result: FAIL
   Note: initial image-edit payload assertion exposed missing multi-reference profile metadata in the unit-test bootstrap fixture.

3. `npx playwright test tests/e2e/specs/bootstrap.spec.js`
   Result: PASS (`1` spec)
   Note: browser harness already proved the multi-reference image-edit submit path was correct.

4. `npm run test:unit -- web/tests/rookieui_extension.test.js`
   Result: FAIL
   Note: iterative fixture alignment surfaced additional expectation drift (diffusion-model list, serialized `image_data` fields, preserved `txt2img` fetch history).

5. `.\.venv\Scripts\python.exe -m unittest tests.test_live_smoke_tests tests.test_host_embedded_e2e`
   Result: PASS (`80` tests)

6. `npm run test:unit -- web/tests/rookieui_extension.test.js`
   Result: PASS (`7` tests)

7. `npx playwright test tests/e2e/specs/bootstrap.spec.js`
   Result: PASS (`1` spec)

## SOP Reading

1. `Get-Content tests/TEST_SOP.md`
   Result: reviewed

2. `Get-Content tests/E2E_TESTING_NOTICE.md`
   Result: reviewed

3. `Get-Content tests/E2E_TESTING_SOP.md`
   Result: reviewed

## Full Gate

1. `powershell -File scripts/run_full_tests_windows.ps1`
   Result: FAIL
   Root cause: `web/tests/rookieui_frontend_architecture.test.js` rejected stale `ROOKIEUI_ASSET_REVISION` because the shipped frontend fingerprint changed after the `F169` frontend edits.

2. `npm run test:unit -- web/tests/rookieui_frontend_architecture.test.js web/tests/rookieui_extension.test.js`
   Result: PASS (`11` tests)

3. `npx playwright test tests/e2e/specs/bootstrap.spec.js`
   Result: PASS (`1` spec)

4. `powershell -File scripts/run_full_tests_windows.ps1`
   Result: PASS
   Gate details:
   - `pre-commit run detect-secrets --all-files` PASS
   - `pre-commit run --all-files --show-diff-on-failure` PASS
   - backend unit tests PASS (`540` tests, `10` skipped)
   - `npm run test:types` PASS
   - `npm test` PASS (`19` frontend unit files, `112` tests; `5` Playwright specs)
   - optional host-embedded lane skipped by wrapper because `ROOKIEUI_RUN_LIVE_SMOKE` was not enabled for `F169`
