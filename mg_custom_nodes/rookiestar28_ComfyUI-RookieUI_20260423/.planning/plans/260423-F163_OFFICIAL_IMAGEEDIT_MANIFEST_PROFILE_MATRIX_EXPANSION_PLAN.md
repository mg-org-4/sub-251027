# Phase 85 Plan - Official ImageEdit Manifest/Profile Matrix Expansion

## Branch Execution Note

- Target branch: `dev`
- Reason/source: `.planning/ROADMAP.md` phase 85 defines this as the manifest/profile truth-alignment step that follows `F162` and precedes runtime/UI delivery.
- Merge condition: only after full `tests/TEST_SOP.md` validation plus review.

## Source-of-Truth Override

- Explicit current-session user instruction requires roadmap/plan/reference/record artifacts to be written in `.planning/` and accepted through sequential commits.
- This overrides the repo-standard `.gitignore` non-tracking rule for `.planning/` artifacts for this execution chain only.

## Scope

### In scope

- Expand manifest-derived profile metadata for official image-edit models.
- Add manifest fields that describe:
  - image-edit identity
  - canonical request/route contract surface
  - reference-input mode / reference-count expectations
  - encoder family
  - template-owned LoRA chain mode
- Expose those fields through registry/preset/capabilities payloads.
- Keep existing public UI exposure stable until `F168`.

### Out of scope

- Frontend `img2img` pane behavior changes.
- Replacing the current dedicated UI `Edit` surface.
- New runtime builders or additional official image-edit profiles.
- Any Wan/temporal runtime work.

## Design Changes

### Manifest schema

- Extend `FamilyTemplateManifestEntry` with explicit image-edit metadata rather than overloading `flow_kind` alone.
- Preserve compatibility for existing helper functions and runtime routing.

### Registry / preset payloads

- Include the new image-edit metadata in registry payloads so later frontend/runtime steps can consume the same source of truth.
- Include the relevant image-edit metadata in preset payloads for later UI integration.

### Backward-compatibility / sequencing rule

- `available_surface_flows` remains unchanged in this item so the current frontend surface does not jump ahead of `F168`.
- The canonical backend `img2img` route ownership introduced in `F162` remains the accepted contract even while UI exposure is still transitional.

## Security Implications

- None directly beyond metadata truthfulness.
- Indirectly positive because later UI/runtime work will read one explicit source of truth instead of inferring image-edit semantics from ad-hoc profile ids.

## Failure Modes and Rollback

- Failure mode: changing `available_surface_flows` too early unintentionally exposes unfinished UI paths.
  - Mitigation: do not change public surface exposure in this item.
- Failure mode: helper functions that currently depend on `flow_kind == "edit"` regress.
  - Mitigation: keep helper compatibility or migrate them to explicit image-edit metadata in the same change.
- Rollback:
  - revert the manifest/schema/payload changes and restore the previous registry contract version

## Test Plan

Reference sources:
- `tests/TEST_SOP.md`
- `tests/E2E_TESTING_NOTICE.md`
- `tests/E2E_TESTING_SOP.md`
- `.planning/references/260423-R170_OFFICIAL_IMAGEEDIT_REBASELINE_REFERENCE.md`
- `.planning/references/260423-R171_CHRONO_WAN_TEMPORAL_EDIT_DEFER_REFERENCE.md`

### Targeted contract proof

- backend/tests:
  - `tests.test_model_family_registry`
  - `tests.test_capabilities`
  - `tests.test_img2img_translation`
- frontend unit tests if fallback registry data changes:
  - `web/tests/rookieui_api.test.js`

### Final full-gate sweep

1. `pre-commit run detect-secrets --all-files`
2. `pre-commit run --all-files --show-diff-on-failure`
3. backend unit tests via `powershell -File scripts/run_full_tests_windows.ps1`
4. frontend Playwright harness per `tests/E2E_TESTING_SOP.md`
5. `npm run test:types` via the same full-gate wrapper

## Acceptance Criteria

- Manifest entries for official image-edit profiles expose explicit image-edit metadata beyond the older `flow_kind` assumption.
- Registry/capabilities/preset payloads expose the new metadata.
- Existing UI surface exposure remains unchanged in this item.
- Targeted tests pass.
- Full repository validation gate passes and is recorded in the implementation record.
