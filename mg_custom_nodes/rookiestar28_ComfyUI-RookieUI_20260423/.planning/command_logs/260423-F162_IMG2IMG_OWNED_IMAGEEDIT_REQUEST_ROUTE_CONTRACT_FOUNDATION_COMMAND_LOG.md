# Command Log - Img2Img-Owned ImageEdit Request and Route Contract Foundation

Date: 2026-04-23
Environment: Windows PowerShell, repo-local `.venv`, branch `dev`

## Commands

1. `Get-Content rookieui/services/workflow_translation.py | Select-Object -First 220`
2. `Get-Content tests/test_img2img_translation.py | Select-Object -First 1160`
3. `Get-Content tests/test_model_family_registry.py | Select-Object -First 200`
4. `Get-Content tests/test_capabilities.py | Select-Object -First 140`
5. `Get-Content scripts/run_live_smoke_tests.py | Select-Object -Skip 3248 -First 70`
6. `.venv\Scripts\python.exe -m unittest tests.test_img2img_translation`
7. `.venv\Scripts\python.exe -m unittest tests.test_live_smoke_tests`
8. `powershell -File scripts/run_full_tests_windows.ps1`
9. `$env:ROOKIEUI_LIVE_BASE_URL='http://127.0.0.1:8188'; .venv\Scripts\python.exe scripts/run_host_embedded_e2e.py --skip-execute`

## Manual review checkpoints

- Verified the new backend contract preserves legacy single-image callers through reference slot `0`.
- Verified official image-edit profiles now normalize onto canonical `mode="img2img"` while keeping internal `execution_mode="edit"` for transitional runtime compatibility.
- Verified the optional host-embedded lane failed only because the active host fingerprint was stale, so it cannot be used as acceptance evidence for this item.
