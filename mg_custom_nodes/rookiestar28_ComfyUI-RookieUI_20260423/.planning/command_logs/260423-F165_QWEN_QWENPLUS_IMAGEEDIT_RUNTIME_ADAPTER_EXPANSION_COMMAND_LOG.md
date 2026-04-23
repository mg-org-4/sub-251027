# Command Log - F165 Qwen / Qwen+ ImageEdit Runtime Adapter Expansion

Date: 2026-04-23
Environment: Windows PowerShell, repo-local `.venv`, branch `dev`

## Targeted regression commands

1. Backend targeted regression

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_model_family_registry tests.test_capabilities tests.test_model_inventory tests.test_img2img_translation
```

Result: passed

2. Frontend targeted unit regression

```powershell
npm run test:unit -- web/tests/rookieui_api.test.js web/tests/rookieui_extension.test.js
```

Result: passed

3. Playwright targeted bootstrap regression

```powershell
npx playwright test tests/e2e/specs/bootstrap.spec.js
```

Result: passed

4. Frontend architecture fingerprint regression

```powershell
npm run test:unit -- web/tests/rookieui_frontend_architecture.test.js
```

Result: passed

## Final full-gate command

```powershell
powershell -File scripts/run_full_tests_windows.ps1
```

Result: passed

Observed final gate outcomes:

- `detect-secrets`: pass
- `pre-commit --all-files --show-diff-on-failure`: pass
- backend unit suite: pass
- `npm run test:types`: pass
- `npm test`:
  - Vitest unit suite: pass
  - Playwright E2E suite: pass
- optional host-embedded live-smoke lane:
  - skipped by script because `ROOKIEUI_RUN_LIVE_SMOKE` was not enabled

## Intermediate corrections resolved before final pass

- Updated `tests/test_live_smoke_tests.py` to match the manifest-derived non-SD catalog ordering after the new Qwen-family edit profiles were added.
- Updated `web/rookieui_asset_revision.js` to match the new shipped frontend fingerprint after the fallback bootstrap matrix changed.
