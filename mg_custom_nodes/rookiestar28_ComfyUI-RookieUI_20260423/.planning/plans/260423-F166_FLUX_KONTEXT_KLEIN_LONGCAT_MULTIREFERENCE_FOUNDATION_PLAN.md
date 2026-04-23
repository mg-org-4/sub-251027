# Phase 87 Plan - Flux / Kontext / Klein / Longcat Multi-Reference Foundation

## Branch Execution Note

- Target branch: `dev`
- Reason/source: `.planning/ROADMAP.md` marks `F166` as a phase-87 runtime-foundation item that must land on `dev` before adapter delivery.
- Merge condition: only after full `tests/TEST_SOP.md` validation plus review.

## Source-of-Truth Override

- Explicit current-session user instruction requires roadmap/plan/reference/record artifacts to be written in `.planning/` and accepted through sequential commits.
- This overrides the repo-standard `.gitignore` non-tracking rule for `.planning/` artifacts for this execution chain only.

## Scope

### In scope

- Expand `rookieui/services/workflow_builders/image_edit_foundation.py` into the shared helper seam for Flux-family image-edit workflows.
- Add reusable helper nodes for:
  - chained `ImageStitch`
  - `FluxKontextImageScale`
  - mirrored `ReferenceLatent` chains
  - `FluxKVCache`
  - Flux2 advanced sampler assembly and guider selection
- Add dataclasses or equivalent structured return values where the helper output is multi-node and consumed as one logical unit.
- Add focused backend unit coverage for the new foundation helpers.
- Sync `.planning/ROADMAP.md` so `F166` moves from planned to completed when the work lands.

### Out of scope

- Adding new public image-edit profiles or selector hints (`F167`)
- Reworking the public `img2img` UI and removing the transitional separate edit lane (`F168`)
- Expanding smoke/live-host execution matrices (`F169`, `R172`)
- Temporal/video image-edit templates already deferred by `R171`

## Design Changes

### Shared foundation expansion

- Keep `image_edit_foundation.py` as the canonical ordered-reference seam introduced by `F164`.
- Add three groups of helpers:
  - Kontext helpers for chained `ImageStitch` and `FluxKontextImageScale`
  - latent-conditioning helpers for mirrored `ReferenceLatent` chains and optional Flux reference-method application
  - Flux2 sampler helpers that hide the repeated `GetImageSize` / latent-canvas / guider / scheduler / sampler-custom wiring

### Structured helper outputs

- Introduce dataclass-backed bundles for multi-node helper outputs where downstream builders need more than one returned node id.
- Candidate bundles:
  - Kontext reference-image bundle
  - Flux2 latent-canvas bundle
  - Flux2 advanced-sampler bundle
- Keep bundle fields limited to node ids and ordered references so later adapters stay transparent and testable.

### Non-functional delivery boundary

- `F166` is infrastructure-only.
- Existing shipped behavior should remain unchanged after this item; public adapter expansion begins in `F167`.
- The new helpers must be directly tested so the item has objective acceptance evidence even before `F167` consumes them.

## Security Implications

- Positive: centralized helper ownership reduces the risk of template-by-template graph drift when more Flux-family edit profiles land.
- Positive: ordered reference handling stays bounded and explicit, which reduces accidental fan-in or hidden latent-order changes.
- No new network, filesystem, or privilege surfaces are introduced.

## Failure Modes and Rollback

- Failure mode: mirrored reference-latent helpers reverse the latent order between positive and negative branches.
  - Mitigation: pin both branch chains in dedicated unit tests.
- Failure mode: Kontext stitch helpers silently change image ordering or stitch direction defaults.
  - Mitigation: pin pairwise stitch chaining and Kontext scale wiring in unit tests.
- Failure mode: Flux2 sampler helpers build the wrong guider type for BasicGuider vs CFGGuider templates.
  - Mitigation: add separate unit tests for both guider paths.
- Rollback:
  - revert the new helper layer, restore the prior minimal foundation file, revert roadmap status, and remove associated tests/docs.

## Test Plan

Reference sources:
- `tests/TEST_SOP.md`
- `tests/E2E_TESTING_NOTICE.md`
- `tests/E2E_TESTING_SOP.md`
- `.planning/references/260423-F166_FLUX_KONTEXT_KLEIN_LONGCAT_MULTIREFERENCE_REFERENCE.md`

### Targeted regression proof

- backend/tests:
  - `tests.test_image_edit_foundation`
  - `tests.test_img2img_translation`
- frontend/tests:
  - no functional frontend change expected for `F166`; rely on the required full gate to prove no regression

### Final full-gate sweep

1. `pre-commit run detect-secrets --all-files`
2. `pre-commit run --all-files --show-diff-on-failure`
3. backend unit tests via `powershell -File scripts/run_full_tests_windows.ps1`
4. frontend Playwright harness per `tests/E2E_TESTING_SOP.md`
5. `npm run test:types` via the same full-gate wrapper

## Acceptance Criteria

- `image_edit_foundation.py` exposes shared helpers for Kontext stitch/scale, mirrored reference-latent chains, `FluxKVCache`, and Flux2 advanced sampler assembly.
- The new helper outputs are structured enough for later adapters to consume without re-parsing raw workflow dictionaries.
- `tests.test_image_edit_foundation` covers:
  - chained stitch ordering
  - Kontext scale bundle
  - mirrored positive/negative reference-latent chains
  - BasicGuider vs CFGGuider Flux2 sampler assembly
  - KV-cache model patch node construction
- Existing shipped behavior remains unchanged and the full repository validation gate passes.
