# Command Log - F166 Flux / Kontext / Klein / Longcat Multi-Reference Foundation

Date: 2026-04-23
Environment: Windows PowerShell, branch `dev`, repo-local `.venv`

## Targeted regression commands

1. `.\.venv\Scripts\python.exe -m unittest tests.test_image_edit_foundation`
   - Result: passed
2. `.\.venv\Scripts\python.exe -m unittest tests.test_image_edit_foundation tests.test_img2img_translation`
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
