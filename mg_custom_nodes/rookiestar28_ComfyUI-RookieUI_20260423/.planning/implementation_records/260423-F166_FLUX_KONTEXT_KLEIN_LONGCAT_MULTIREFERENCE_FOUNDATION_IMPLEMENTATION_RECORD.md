# Phase 87 Implementation Record - Flux / Kontext / Klein / Longcat Multi-Reference Foundation

## What Changed

- Expanded `rookieui/services/workflow_builders/image_edit_foundation.py`
  - added structured bundle dataclasses for Kontext reference ownership and Flux2 sampler assembly
  - added shared helpers for chained `ImageStitch`
  - added `FluxKontextImageScale` + VAE encode bundling
  - added mirrored positive/negative `ReferenceLatent` chain helpers
  - added `FluxGuidance` plus Flux reference-method branch helpers
  - added `FluxKVCache` model wrapper construction
  - added Flux2 advanced sampler helpers covering:
    - `GetImageSize`
    - `EmptyFlux2LatentImage`
    - `RandomNoise`
    - `KSamplerSelect`
    - `Flux2Scheduler`
    - `BasicGuider` vs `CFGGuider`
    - `SamplerCustomAdvanced`
- Expanded `tests/test_image_edit_foundation.py`
  - added coverage for stitch ordering
  - added Kontext scale/encode bundle coverage
  - added mirrored positive/negative latent-chain coverage
  - added Flux reference-method branch coverage
  - added KV-cache node coverage
  - added Flux2 advanced sampler bundle coverage for both guider variants
- Added planning artifacts:
  - `.planning/references/260423-F166_FLUX_KONTEXT_KLEIN_LONGCAT_MULTIREFERENCE_REFERENCE.md`
  - `.planning/plans/260423-F166_FLUX_KONTEXT_KLEIN_LONGCAT_MULTIREFERENCE_FOUNDATION_PLAN.md`
  - `.planning/command_logs/260423-F166_FLUX_KONTEXT_KLEIN_LONGCAT_MULTIREFERENCE_FOUNDATION_COMMAND_LOG.md`
- Updated `.planning/ROADMAP.md`
  - marked `F166` completed
  - moved the open image-edit chain start to `F167`

## Why Changed

- The current image-edit chain already had ordered reference loading from `F164`, but the official Flux-family templates prove that later adapters need more than raw `ReferenceLatent` chaining.
- `Flux.1 Kontext Dev`, `Flux.2 image edit`, `Flux.2 Klein 9b KV image edit`, and `Longcat image edit` share repeatable graph shapes around stitched references, mirrored latent ownership, Flux-specific reference-method metadata, KV cache, and Flux2 sampler assembly.
- `F166` closes that shared-infrastructure gap first so `F167` can ship bounded public adapters without re-implementing each template as a bespoke graph.

## Full Verification Evidence

- Date/environment: 2026-04-23, Windows PowerShell, repo-local `.venv`, branch `dev`
- Command log reference:
  - `.planning/command_logs/260423-F166_FLUX_KONTEXT_KLEIN_LONGCAT_MULTIREFERENCE_FOUNDATION_COMMAND_LOG.md`

### Targeted regression proof

- `.venv\Scripts\python.exe -m unittest tests.test_image_edit_foundation`
  - passed
- `.venv\Scripts\python.exe -m unittest tests.test_image_edit_foundation tests.test_img2img_translation`
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

- No new public image-edit profile was exposed by this item; adapter delivery still belongs to `F167`.
- The public UI remains unchanged; `F168` is still responsible for folding image-edit profiles into the primary `img2img` surface.
- Live-host execute proof for Flux/Kontext/Klein/Longcat adapters is still pending because the adapters themselves are not yet shipped.

## Follow-up Items

- `F167` must consume the new foundation helpers to deliver the first bounded Flux/Kontext/Klein/Longcat image-edit adapters.
- `F168` must rework the visible `img2img` surface around the accepted image-edit contract.
- `F169-R172` must extend fixture/live-host proof so the broader image-edit matrix is accepted on direct evidence rather than helper-level unit coverage alone.
