# Phase 86 Implementation Record - Qwen / Qwen+ ImageEdit Runtime Adapter Expansion

## What Changed

- Expanded `rookieui/contracts/family_template_manifest.py`
  - added `qwen_image_edit_multi_lora`
  - added `firered_image_edit`
  - added `firered_image_edit_lightning`
  - refreshed Qwen edit manifest notes and selector hints
  - bumped the registry contract version to `f165-20260423`
- Updated `rookieui/services/img2img.py`
  - generalized template-owned LoRA host-prerequisite enforcement from a qwen-specific hard-code into a manifest-driven check
- Updated `rookieui/services/workflow_builders/non_sd_templates.py`
  - generalized the current qwen edit builder into a Qwen-family image-edit builder
  - added `TextEncodeQwenImageEditPlus` workflow construction
  - added manifest-driven template-owned LoRA chain depth handling
  - added FireRed multi-reference plus-encoder support while keeping the accepted no-mask image-edit contract
- Updated frontend fallback/bootstrap artifacts:
  - `web/rookieui_api.js`
  - `web/rookieui_asset_revision.js`
  - `tests/e2e/boot.mjs`
  - `tests/e2e/specs/bootstrap.spec.js`
- Added and updated regression coverage for:
  - manifest/capability metadata
  - selector resolution
  - qwen triple-LoRA chaining
  - FireRed multi-reference plus-encoder translation
  - frontend fallback/edit-preset exposure

## Why Changed

- `F165` is the first runtime-delivery step that broadens the accepted image-edit contract beyond the initial single-reference `qwen_image_edit` path.
- The official reference set now proves three objective Qwen-family runtime shapes:
  - base Qwen single-image edit
  - Qwen multi-LoRA chain
  - FireRed/Qwen+ multi-reference edit
- Prior accepted code still shipped only the narrow single-reference Qwen path, which left the roadmap-backed Qwen-family runtime scope materially incomplete.

## Full Verification Evidence

- Date/environment: 2026-04-23, Windows PowerShell, repo-local `.venv`, branch `dev`
- Command log reference:
  - `.planning/command_logs/260423-F165_QWEN_QWENPLUS_IMAGEEDIT_RUNTIME_ADAPTER_EXPANSION_COMMAND_LOG.md`

### Targeted regression proof

- `.venv\Scripts\python.exe -m unittest tests.test_model_family_registry tests.test_capabilities tests.test_model_inventory tests.test_img2img_translation`
  - passed
- `npm run test:unit -- web/tests/rookieui_api.test.js web/tests/rookieui_extension.test.js`
  - passed
- `npx playwright test tests/e2e/specs/bootstrap.spec.js`
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

## Known Limitations

- The public UI still exposes these profiles through the transitional `Edit` lane; `F168` is still responsible for folding image-edit profiles back into the main `img2img` workspace.
- This item does not yet add Flux/Kontext/Klein/Longcat edit adapters.
- This item does not yet expand live-smoke execute coverage for the new FireRed/Qwen-family subset; that belongs to later image-edit acceptance work.

## Follow-up Items

- `F166-F167` must deliver the Flux/Kontext/Klein/Longcat image-edit foundation and adapter set on top of the accepted ordered-reference seam.
- `F168` must remove the transitional separate `Edit` UI assumptions and expose ordered multi-reference input state on `img2img`.
- `F169-R172` must extend smoke/live-host evidence so the broader image-edit runtime matrix is proven against truthful host readiness, not only local translation tests.
