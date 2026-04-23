# F168 Img2Img ImageEdit UI Surface Integration Plan

Date: 2026-04-23
Item: F168
Branch execution note:
- Target branch: `dev`
- Reason/source: `.planning/ROADMAP.md` Phase 88 marks F168 as `dev`-first because it changes visible `img2img` UX and removes invalid edit-surface assumptions.
- Merge condition: only after full `tests/TEST_SOP.md` validation plus review evidence are complete.

## 1. Scope

In scope:
- Re-expose first-wave image-edit profiles on the canonical `img2img` surface.
- Remove the dedicated visible `Edit` generation subtab from the Img2Img pane.
- Make Img2Img mode availability profile-aware so official image-edit profiles stay on `img2img` only.
- Replace old edit-mode mask semantics with profile-driven image-edit semantics.
- Add ordered image-edit reference-image UI state for current first-wave limits:
  - slot 1 uses the existing source image/canvas contract,
  - slots 2-3 are additional ordered references,
  - UI selects a main reference and serializes `reference_images` + `main_reference_index`.
- Update fallback/bootstrap metadata, tests, roadmap status, implementation record, and acceptance commit.

Out of scope:
- Chrono/Wan temporal edit delivery.
- New backend workflow builders or new image-edit families beyond the shipped first wave.
- Unlimited reference-image UI.
- Masked edit/inpaint hybrids for image-edit profiles.

## 2. Design Changes

Manifest / capability contract:
- Change first-wave image-edit `available_surface_flows` truth from `edit` to `img2img`.
- Keep `flow_kind="edit"` and `request_contract_surface="img2img"` so runtime classification and request contract remain distinct.
- Bump manifest/fallback contract versions to reflect the surface-contract change.

Frontend metadata / lookup:
- Extend `buildProfileLookup()` so the pane can read `image_edit_profile`, `request_contract_surface`, `reference_input_mode`, and `max_direct_references`.

Img2Img UI / state:
- Remove the user-facing `Edit` generation mode button and option.
- Normalize any legacy imported `mode="edit"` payload back to `img2img`.
- Keep mode switching for `img2img`, `sketch`, `inpaint`, `inpaint_sketch`, `inpaint_upload`, `batch`.
- When the active preset/profile is an official image-edit profile:
  - force `mode` to `img2img`,
  - disable unsupported mode buttons,
  - hide/disable mask controls,
  - hide batch-only affordances,
  - keep edit-specific advanced parameters visible based on manifest truth,
  - show ordered reference-image controls with truthful per-profile slot limits.

Payload serialization:
- Add hidden pane state for additional ordered reference images and `main_reference_index`.
- Serialize image-edit requests through:
  - `reference_images`
  - `main_reference_index`
- Preserve legacy `image_asset` / `image_data` source fields for non-image-edit profiles and as slot 1 for image-edit profiles.

Testing:
- Update backend capability/registry assertions for `available_surface_flows`.
- Update frontend unit tests and E2E tests from “separate edit mode” expectations to “img2img profile-driven image-edit branch” expectations.

## 3. Security Implications

- No new network surface is introduced.
- Uploaded reference images remain local data URLs or existing asset handles, following current Img2Img upload behavior.
- The UI must clear or ignore stale mask fields when an image-edit profile is active so hidden stale inputs cannot leak misleading payload state.
- The UI must validate main-reference selection so an empty selected slot cannot silently degrade into a wrong request.

## 4. Failure Modes And Rollback

Failure modes:
- Legacy queue/history payloads that still store `mode="edit"` may fail to restore correctly if not normalized.
- Hidden stale mask state may remain attached to image-edit submissions if profile-aware cleanup is incomplete.
- Main-reference selection may become invalid when users switch presets or clear a reference slot.
- E2E may still pin removed `generation-mode-edit` selectors.

Rollback:
- Revert the F168 commit on `dev`.
- Restore prior manifest surface-flow metadata, edit-mode UI buttons, and pre-F168 tests if the integrated branch cannot pass full SOP validation.

## 5. Test Plan

Required reading order before final sweep:
1. `tests/TEST_SOP.md`
2. `tests/E2E_TESTING_NOTICE.md`
3. `tests/E2E_TESTING_SOP.md`

Targeted verification before full sweep:
- Backend:
  - `.\.venv\Scripts\python.exe -m unittest tests.test_model_family_registry tests.test_capabilities`
- Frontend unit:
  - `npm run test:unit -- web/tests/rookieui_api.test.js web/tests/rookieui_extension.test.js`
- Targeted E2E:
  - `npx playwright test tests/e2e/specs/bootstrap.spec.js`

Final acceptance sweep per `tests/TEST_SOP.md`:
- `powershell -File scripts/run_full_tests_windows.ps1`

## 6. Acceptance Criteria

- All first-wave image-edit profiles are visible on the Img2Img preset surface without a separate `Edit` generation subtab.
- Selecting an image-edit profile forces truthful Img2Img-only execution behavior and removes mask-first edit affordances.
- Ordered reference-image UI supports the first-wave cap and submits `reference_images` plus `main_reference_index`.
- Legacy `mode="edit"` payload restore paths normalize back to the canonical Img2Img surface without breaking pane state.
- Backend/frontend/E2E assertions are updated and the full Windows SOP gate passes.
- `.planning/ROADMAP.md`, reference summary, implementation record, command log, and acceptance commit are complete.
