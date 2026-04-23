# Phase 88 Plan - ImageEdit Smoke / Fixture / Test Matrix Foundation

## Branch Execution Note

- Target branch: `dev`
- Reason/source: `.planning/ROADMAP.md` marks `F169` as a `dev`-first regression/live-smoke foundation item that must land before final image-edit acceptance closure.
- Merge condition: only after full `tests/TEST_SOP.md` validation plus review.

## Source-of-Truth Override

- Explicit current-session user instruction requires roadmap/plan/reference/record artifacts to be written in `.planning/` and accepted through sequential commits.
- This overrides the repo-standard `.gitignore` non-tracking rule for `.planning/` artifacts for this execution chain only.

## Scope

### In scope

- Add a dedicated image-edit validation lane in the live-smoke runner.
- Make image-edit smoke payloads use the canonical ordered-reference contract:
  - `reference_images`
  - `main_reference_index`
  - no mask
  - canonical public `img2img` mode
- Add manifest-driven dry-run validation for first-wave image-edit profiles:
  - reference-count expectations
  - no-mask workflow proof
  - template-owned LoRA chain depth expectations
- Extend frontend regression coverage so shipped UI tests assert submitted multi-reference image-edit payloads instead of only checking visibility.
- Extend host-embedded runner tests and live-smoke unit tests to pin the new lane.

### Out of scope

- New runtime adapters or model-family additions already covered by `F165` / `F167`
- Final restarted-host acceptance closure and claim language (`R172`)
- Temporal/video edit graphs already deferred by `R171`
- Any change to non-image-edit validation lanes beyond the minimal integration needed to host the new image-edit lane

## Design Changes

### Live-smoke runner

- Extend `scripts/run_live_smoke_tests.py` with a dedicated `image-edit` validation mode.
- Add a manifest-driven image-edit case builder that derives:
  - profile id
  - direct-reference count
  - selected main-reference slot
  - expected template-owned LoRA chain depth
  - expected workflow kind
- Keep the lane two-step:
  - catalog/readiness validation first
  - image-edit dry-run validation second
  - optional execute lane third

### Payload and fixture contract

- `_build_edit_payload()` should emit ordered `reference_images` and `main_reference_index` instead of relying on a single legacy source image.
- Multi-reference cases should intentionally use a non-zero `main_reference_index` where the manifest allows it so ordering is actually tested.
- Single-reference edit profiles should still use the same canonical `reference_images` contract to avoid split smoke semantics.

### Dry-run assertions

- Validate route payloads returned by `/rookieui/generate/img2img` for image-edit profiles:
  - `workflow_kind`
  - `normalized_request.mode == "img2img"`
  - `normalized_request.execution_mode == "edit"`
  - `normalized_request.reference_image_assets` length
  - `normalized_request.main_reference_index`
  - absence of mask nodes
  - template-owned `LoraLoaderModelOnly` depth derived from `template_lora_chain_mode`

### Frontend regression proof

- Extend `web/tests/rookieui_extension.test.js` to submit a multi-reference image-edit profile and assert:
  - serialized `reference_images`
  - serialized `main_reference_index`
  - mask fields are not submitted
- Extend `tests/e2e/specs/bootstrap.spec.js` with the same multi-reference request proof using the browser harness request capture.

## Security Implications

- Positive: manifest-derived smoke expectations reduce the chance of false host-readiness claims for edit profiles.
- Positive: explicit ordered-reference validation prevents a regression back to ambiguous single-image fallback semantics.
- No new network, filesystem, or privilege surfaces are introduced; this work strengthens validation truth rather than expanding runtime capability.

## Failure Modes and Rollback

- Failure mode: the new smoke lane duplicates stale assumptions instead of following manifest metadata.
  - Mitigation: derive case expectations from `family_template_manifest.py`, not a parallel handwritten matrix.
- Failure mode: multi-reference smoke payloads pass because of legacy single-image fallback rather than true ordered-reference handling.
  - Mitigation: require `reference_images` plus a non-zero `main_reference_index` for bounded multi-reference profiles.
- Failure mode: frontend request tests become brittle because hidden reference fields are manipulated inconsistently.
  - Mitigation: write assertions against the captured request payload rather than DOM-only visibility state.
- Rollback:
  - remove the dedicated image-edit smoke lane, revert the new image-edit-specific tests/docs, and restore the prior catalog-only behavior.

## Test Plan

Reference sources:
- `tests/TEST_SOP.md`
- `tests/E2E_TESTING_NOTICE.md`
- `tests/E2E_TESTING_SOP.md`
- `.planning/references/260423-F169_IMAGEEDIT_SMOKE_FIXTURE_TEST_MATRIX_REFERENCE.md`

### Targeted regression proof

- backend/tests:
  - `tests.test_live_smoke_tests`
  - `tests.test_host_embedded_e2e`
- frontend/tests:
  - `web/tests/rookieui_extension.test.js`
  - `tests/e2e/specs/bootstrap.spec.js`

### Final full-gate sweep

1. `pre-commit run detect-secrets --all-files`
2. `pre-commit run --all-files --show-diff-on-failure`
3. backend unit tests via `powershell -File scripts/run_full_tests_windows.ps1`
4. frontend Playwright harness per `tests/E2E_TESTING_SOP.md`
5. `npm run test:types` via the same full-gate wrapper

## Acceptance Criteria

- `scripts/run_live_smoke_tests.py` supports a dedicated image-edit validation lane with manifest-driven defaults.
- Image-edit live-smoke payloads submit ordered `reference_images` and `main_reference_index` using the canonical `img2img` surface contract.
- Dry-run validation pins no-mask image-edit topology and template-owned LoRA chain depth for the selected profiles.
- Frontend unit/E2E coverage proves that multi-reference image-edit submissions serialize the expected payload instead of only toggling UI visibility.
- The full repository validation gate passes and is recorded in the implementation record.
