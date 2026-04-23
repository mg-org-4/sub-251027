# Command Log - Shared Multi-Reference ImageEdit Conditioning Foundation

Date: 2026-04-23
Environment: Windows PowerShell, repo-local `.venv`, branch `dev`

## Commands

1. `.venv\Scripts\python.exe -m unittest tests.test_image_edit_foundation tests.test_img2img_translation tests.test_workflow_builder_modules`
2. `powershell -File scripts/run_full_tests_windows.ps1`

## Manual review checkpoints

- Verified the new helper seam keeps qwen-edit on the same single-reference public behavior while moving image loading and latent creation onto shared infrastructure.
- Verified manifest-declared direct-reference limits now reject unsupported qwen multi-reference payloads instead of silently accepting them.
- Verified the foundation module carries the later `ReferenceLatent` and Flux method helpers without forcing premature adapter delivery in this item.
