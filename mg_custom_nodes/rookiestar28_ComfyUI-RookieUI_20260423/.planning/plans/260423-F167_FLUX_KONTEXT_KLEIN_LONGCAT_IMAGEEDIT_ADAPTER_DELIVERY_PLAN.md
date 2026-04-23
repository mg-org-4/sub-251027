# Phase 87 Plan - Flux / Kontext / Klein / Longcat ImageEdit Adapter Delivery

## Branch Execution Note

- Target branch: `dev`
- Reason/source: `.planning/ROADMAP.md` marks `F167` as the first public adapter-delivery step after the accepted Flux-family foundation and requires `dev`-first validation.
- Merge condition: only after full `tests/TEST_SOP.md` validation plus review.

## Source-of-Truth Override

- Explicit current-session user instruction requires roadmap/plan/reference/record artifacts to be written in `.planning/` and accepted through sequential commits.
- This overrides the repo-standard `.gitignore` non-tracking rule for `.planning/` artifacts for this execution chain only.

## Scope

### In scope

- Add four new manifest-backed official image-edit profiles:
  - `flux_kontext_dev_edit`
  - `flux2_image_edit`
  - `klein_9b_kv_image_edit`
  - `longcat_image_edit`
- Add truthful selector hints/defaults for the above profiles.
- Expand `build_non_sd_edit_workflow` to dispatch the four new adapter families on top of `image_edit_foundation.py`.
- Keep all four profiles on the accepted image-edit contract:
  - `img2img` request ownership
  - no user mask
  - ordered reference images
  - manifest-derived direct-reference caps
- Sync backend capability/preset/registry payloads plus frontend fallback/bootstrap data.
- Add targeted backend/frontend/E2E regressions for:
  - registry metadata
  - inventory selector resolution
  - workflow topology
  - transitional edit-lane preset exposure

### Out of scope

- Public UI merge of image-edit into the main `img2img` pane (`F168`)
- Additional smoke/live-host matrix hardening (`F169`, `R172`)
- Temporal/video edit graphs already deferred by `R171`
- Any turbo/lightning second-wave variants not explicitly called for by the current roadmap item

## Design Changes

### Manifest/profile matrix

- Extend the model-family manifest with four first-wave edit profiles and a new contract version.
- Keep metadata explicit per profile:
  - base family
  - encoder family
  - direct-reference mode/count
  - visible template-owned LoRA requirements
  - official template path
  - runtime adapter id

### Runtime builders

- Add dedicated builder functions for:
  - Flux Kontext edit
  - Flux2 edit
  - Klein 9b KV edit
  - Longcat edit
- Each builder must reuse `image_edit_foundation.py` helpers for image ownership, latent chains, reference-method wiring, and advanced sampler assembly.
- Preserve template-specific truth:
  - Kontext uses stitched references plus classic `KSampler`
  - Flux2 uses `BasicGuider` advanced sampling
  - Klein 9b KV uses mirrored `ReferenceLatent` chains plus `FluxKVCache`
  - Longcat uses `TextEncodeQwenImageEdit` plus `FluxKontextMultiReferenceLatentMethod`

### Frontend/bootstrap synchronization

- Update shipped fallback registry entries and preset metadata to include the new edit profiles while the UI still exposes the transitional `Edit` lane.
- Refresh the shipped asset revision token after fallback registry expansion.

## Security Implications

- Positive: manifest-driven prerequisites keep selector resolution truthful and prevent silent routing to the wrong host assets.
- Positive: direct-reference caps remain explicit per profile and avoid uncontrolled image fan-in on newly shipped adapters.
- No new network, filesystem, or privilege surfaces are introduced.

## Failure Modes and Rollback

- Failure mode: edit profiles resolve to the wrong diffusion model or text encoder when multiple family-adjacent assets coexist.
  - Mitigation: add selector-resolution tests with family-adjacent inventories.
- Failure mode: Flux2/Klein builders collapse `BasicGuider` and `CFGGuider` paths.
  - Mitigation: pin workflow topology per profile in translation tests.
- Failure mode: frontend fallback registry drifts from backend profile exposure.
  - Mitigation: update backend capability tests, frontend API tests, and Playwright bootstrap expectations together.
- Rollback:
  - remove the new profiles, revert builder dispatch to the prior qwen-only state, restore roadmap status, and remove associated tests/docs.

## Test Plan

Reference sources:
- `tests/TEST_SOP.md`
- `tests/E2E_TESTING_NOTICE.md`
- `tests/E2E_TESTING_SOP.md`
- `.planning/references/260423-F167_FLUX_KONTEXT_KLEIN_LONGCAT_IMAGEEDIT_ADAPTER_REFERENCE.md`

### Targeted regression proof

- backend/tests:
  - `tests.test_model_family_registry`
  - `tests.test_capabilities`
  - `tests.test_model_inventory`
  - `tests.test_img2img_translation`
  - `tests.test_parity_matrix`
- frontend/tests:
  - `web/tests/rookieui_api.test.js`
  - `web/tests/rookieui_extension.test.js`
  - `tests/e2e/specs/bootstrap.spec.js`

### Final full-gate sweep

1. `pre-commit run detect-secrets --all-files`
2. `pre-commit run --all-files --show-diff-on-failure`
3. backend unit tests via `powershell -File scripts/run_full_tests_windows.ps1`
4. frontend Playwright harness per `tests/E2E_TESTING_SOP.md`
5. `npm run test:types` via the same full-gate wrapper

## Acceptance Criteria

- The registry/preset/capability matrix exposes the four new edit profiles with truthful metadata and selector hints.
- `translate_img2img_request` builds:
  - Kontext edit with stitched/scaled references
  - Flux2 edit with `BasicGuider`
  - Klein 9b KV edit with mirrored `ReferenceLatent` chains plus `FluxKVCache`
  - Longcat edit with `TextEncodeQwenImageEdit` plus Flux reference-method nodes
- No shipped image-edit profile in `F167` requires a user mask.
- Transitional frontend bootstrap/edit-lane tests reflect the expanded edit profile set without breaking existing shells.
- The full repository validation gate passes and is recorded in the implementation record.
