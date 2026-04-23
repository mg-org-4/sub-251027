# Phase 86 Plan - Shared Multi-Reference ImageEdit Conditioning Foundation

## Branch Execution Note

- Target branch: `dev`
- Reason/source: `.planning/ROADMAP.md` phase 86 defines this as the reusable runtime-foundation step before family-specific image-edit adapter delivery.
- Merge condition: only after full `tests/TEST_SOP.md` validation plus review.

## Source-of-Truth Override

- Explicit current-session user instruction requires roadmap/plan/reference/record artifacts to be written in `.planning/` and accepted through sequential commits.
- This overrides the repo-standard `.gitignore` non-tracking rule for `.planning/` artifacts for this execution chain only.

## Scope

### In scope

- Add a shared workflow-builder seam for ordered image-edit reference handling.
- Land reusable helpers for:
  - ordered reference image loading
  - main-reference selection
  - optional reference-image scaling ownership
  - reusable VAE latent creation for references
  - ordered `ReferenceLatent` chaining
  - optional `FluxKontextMultiReferenceLatentMethod` wrapping
- Enforce manifest-declared direct-reference limits for official image-edit profiles during request normalization.
- Migrate the existing `qwen_image_edit` builder to the new helper seam without changing its public single-reference behavior.
- Add regression coverage for the new helper seam and qwen single-reference limit enforcement.

### Out of scope

- Shipping new Qwen-family edit adapters beyond the current `qwen_image_edit` baseline.
- Shipping Flux/Kontext/Klein/Longcat edit adapters.
- UI changes for multi-reference upload controls.
- Any temporal / Chrono / Wan edit work.

## Design Changes

### Shared builder module

- Add `rookieui/services/workflow_builders/image_edit_foundation.py` as the reusable image-edit conditioning module.
- Move image-edit-specific loader / scaling / latent helper logic out of `non_sd_templates.py` into this shared seam.

### Reference contract enforcement

- Use manifest-backed `max_direct_references` truth during `img2img` normalization for official image-edit profiles.
- Reject payloads that exceed the profile-declared direct reference count instead of silently accepting unsupported shapes.

### Transitional runtime behavior

- Keep `qwen_image_edit` on its current public single-reference exposure.
- Keep `available_surface_flows` unchanged; this item is internal runtime-foundation work only.
- Preserve current qwen graph topology wherever possible while swapping to the shared helper seam underneath.

## Security Implications

- Positive: manifest-declared reference limits reduce backend/runtime drift and avoid silently constructing unsupported edit graphs.
- No new external I/O or permission surface is introduced; all changes stay within existing asset-handle and workflow-construction boundaries.

## Failure Modes and Rollback

- Failure mode: helper abstraction changes qwen graph topology unintentionally.
  - Mitigation: keep topology regressions pinned in `tests/test_img2img_translation.py`.
- Failure mode: reference-limit enforcement rejects previously accepted but unsupported payload shapes.
  - Mitigation: scope enforcement to official image-edit profiles whose manifest already declares explicit limits.
- Failure mode: Flux/Kontext helper wiring bakes in the wrong method normalization.
  - Mitigation: pin helper behavior against the documented built-in node options from `nodes_flux.py`.
- Rollback:
  - revert the shared helper module, remove the normalization gate, restore the previous qwen builder wiring, and reset roadmap status.

## Test Plan

Reference sources:
- `tests/TEST_SOP.md`
- `tests/E2E_TESTING_NOTICE.md`
- `tests/E2E_TESTING_SOP.md`
- `.planning/references/260423-F164_MULTI_REFERENCE_IMAGEEDIT_CONDITIONING_REFERENCE.md`

### Targeted contract proof

- backend/tests:
  - `tests.test_image_edit_foundation`
  - `tests.test_img2img_translation`
  - `tests.test_workflow_builder_modules`

### Final full-gate sweep

1. `pre-commit run detect-secrets --all-files`
2. `pre-commit run --all-files --show-diff-on-failure`
3. backend unit tests via `powershell -File scripts/run_full_tests_windows.ps1`
4. frontend Playwright harness per `tests/E2E_TESTING_SOP.md`
5. `npm run test:types` via the same full-gate wrapper

## Acceptance Criteria

- Shared image-edit builder helpers exist in a dedicated module and cover ordered references, optional scaling, latent creation, `ReferenceLatent` chaining, and Flux reference-method wrapping.
- Official image-edit profiles with manifest-declared direct reference limits reject unsupported reference counts during normalization.
- `qwen_image_edit` still translates successfully through the new helper seam without introducing mask requirements.
- Targeted tests pass.
- Full repository validation gate passes and is recorded in the implementation record.
