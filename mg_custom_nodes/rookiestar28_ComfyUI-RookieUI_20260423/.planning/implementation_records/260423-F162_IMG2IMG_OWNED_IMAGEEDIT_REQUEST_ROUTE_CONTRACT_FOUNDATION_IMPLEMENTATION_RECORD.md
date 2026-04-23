# Phase 85 Implementation Record - Img2Img-Owned ImageEdit Request and Route Contract Foundation

## What Changed

- Updated `rookieui/contracts/generation.py`
  - added raw `reference_images` plus `main_reference_index` to `Img2ImgRequest`
  - added normalized `reference_image_assets` plus `main_reference_index` to `NormalizedImg2ImgRequest`
- Updated `rookieui/services/img2img.py`
  - added ordered reference-image normalization with legacy `image_asset` / `image_data` fallback to reference slot `0`
  - canonicalized official image-edit profiles onto the public `img2img` contract
  - kept legacy `mode="edit"` accepted as a compatibility alias
  - preserved internal `execution_mode="edit"` for official image-edit profiles so the existing runtime seam still works during the transition
  - made the normalized `image_asset` alias point at the selected main reference image
- Updated `rookieui/services/workflow_translation.py`
  - changed official image-edit workflow kinds from `edit-<profile>` to `img2img-<profile>`
- Updated `scripts/run_live_smoke_tests.py`
  - switched the qwen-image-edit live-smoke payload builder to the canonical `mode="img2img"` contract
- Updated backend tests:
  - `tests/test_img2img_translation.py`
    - pinned canonical `img2img` mode for official image-edit profiles
    - added ordered `reference_images` normalization coverage
    - updated route/translation expectations to `img2img-qwen_image_edit`
  - `tests/test_live_smoke_tests.py`
    - pinned canonical `img2img` edit payloads for smoke helpers

## Why Changed

- The planning baseline is now explicit: image-edit belongs to the `img2img` chain, does not require masks, and must support ordered reference images.
- Before this change, the backend still treated official image-edit as a dedicated public `Edit` surface and had no canonical multi-reference request seam.
- `F162` establishes the backend contract foundation without prematurely changing manifest/bootstrap exposure or frontend UI.

## Full Verification Evidence

- Date/environment: 2026-04-23, Windows PowerShell, repo-local `.venv`, branch `dev`
- Command log reference:
  - `.planning/command_logs/260423-F162_IMG2IMG_OWNED_IMAGEEDIT_REQUEST_ROUTE_CONTRACT_FOUNDATION_COMMAND_LOG.md`

### Targeted contract proof

- `.venv\Scripts\python.exe -m unittest tests.test_img2img_translation`
  - passed
- `.venv\Scripts\python.exe -m unittest tests.test_live_smoke_tests`
  - passed

### Final full-gate evidence

- `powershell -File scripts/run_full_tests_windows.ps1`
  - passed
  - `detect-secrets`: pass
  - `pre-commit --all-files`: pass
  - backend unit tests: pass
  - frontend `npm run test:types`: pass
  - frontend `npm test`: pass

### Optional host-embedded E2E evidence

- `$env:ROOKIEUI_LIVE_BASE_URL='http://127.0.0.1:8188'; .venv\Scripts\python.exe scripts/run_host_embedded_e2e.py --skip-execute`
  - failed
  - reason: live host runtime build fingerprint mismatch (stale host)
  - classification: external host freshness blocker; not accepted as live-host evidence and not treated as a repo-code failure for this item

## Known Limitations

- Manifest/bootstrap exposure still reflects the old dedicated `Edit` surface until `F163` and later UI work land.
- The runtime builder seam is still the older dedicated edit path internally; this item only moved the public request/route contract to `img2img`.
- No valid live-host evidence was available because the current host instance was stale relative to the workspace fingerprint.

## Follow-up Items

- `F163` must align manifest/profile metadata to the new backend contract.
- `F168` must later remove the dedicated public `Edit` surface from the frontend.
