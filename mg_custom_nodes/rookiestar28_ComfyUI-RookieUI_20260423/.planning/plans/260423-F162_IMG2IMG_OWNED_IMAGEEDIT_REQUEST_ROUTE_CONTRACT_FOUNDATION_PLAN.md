# Phase 85 Plan - Img2Img-Owned ImageEdit Request and Route Contract Foundation

## Branch Execution Note

- Target branch: `dev`
- Reason/source: `.planning/ROADMAP.md` phase 85 marks this contract change as a `dev`-only image-edit chain item because it changes canonical `img2img` request normalization and translation behavior.
- Merge condition: only after full `tests/TEST_SOP.md` validation plus review.

## Source-of-Truth Override

- Explicit current-session user instruction requires roadmap/plan/reference/record artifacts to be written in `.planning/` and accepted through sequential commits.
- This overrides the repo-standard `.gitignore` non-tracking rule for `.planning/` artifacts for this execution chain only.

## Scope

### In scope

- Add a canonical ordered-reference payload seam for image-edit requests.
- Keep legacy `image_asset` / `image_data` input compatible by mapping it to reference slot `0`.
- Add explicit `main_reference_index` support.
- Re-baseline image-edit normalization so official image-edit profiles are accepted through the `img2img` route/surface contract rather than a dedicated public `Edit` surface.
- Keep internal runtime routing compatible with the existing dedicated edit builder seam until later runtime items land.
- Update translation output and targeted tests to reflect the new contract.

### Out of scope

- Manifest/profile exposure changes in bootstrap/capabilities/presets (`F163`).
- Frontend `img2img` pane changes (`F168`).
- New family adapters or multi-reference runtime builders (`F164+`).
- Temporal/Wan runtime support (explicitly deferred by `R171`).

## Design Changes

### API / payload contract

- Extend `Img2ImgRequest` with:
  - `reference_images`: ordered list of objects using the same field names as the existing single-image payload seam (`image_asset`, `image_data`)
  - `main_reference_index`
- Extend `NormalizedImg2ImgRequest` with:
  - `reference_image_assets`
  - `main_reference_index`
- Compatibility rule:
  - if `reference_images` is absent, legacy `image_asset` / `image_data` become reference `0`
  - if `reference_images` is present, it is the canonical ordered source list

### Flow / execution semantics

- Public/canonical `mode` for image-edit profiles becomes `img2img`.
- Legacy `mode="edit"` remains accepted as an alias during the transition.
- Internal `execution_mode` may remain `edit` for official image-edit profiles so the existing builder seam continues to function until later phases replace it.

### Translation / routing

- `translate_img2img_request()` should emit `img2img-<profile>` workflow kinds for official image-edit profiles because the public contract is now `img2img`-owned.
- Backend route acceptance should allow official image-edit profiles on the `img2img` contract even before manifest/bootstrap exposure is updated in `F163`.

## Security Implications

- Ordered reference images increase input surface area.
- Guardrails required:
  - each reference image must still pass asset/data validation through the existing input-asset guard
  - enforce a bounded maximum reference count for the raw route contract
  - reject negative or out-of-range `main_reference_index`
- No filesystem-path bypasses or raw local paths may be introduced.

## Failure Modes and Rollback

- Failure mode: existing legacy callers sending `mode="edit"` break immediately.
  - Mitigation: keep `mode="edit"` as a compatibility alias in this phase.
- Failure mode: route contract and capabilities/bootstrap drift temporarily disagree.
  - Mitigation: document that `F162` is backend-only contract foundation; public exposure stays for `F163/F168`.
- Failure mode: reference-image normalization changes the existing single-image path.
  - Mitigation: keep `image_asset` as the normalized primary reference alias so the existing runtime builders remain stable.
- Rollback:
  - revert request/normalization/translation changes
  - preserve the existing single-image edit path until a corrected contract lands

## Test Plan

Reference sources:
- `tests/TEST_SOP.md`
- `tests/E2E_TESTING_NOTICE.md`
- `tests/E2E_TESTING_SOP.md`
- `.planning/references/260423-R170_OFFICIAL_IMAGEEDIT_REBASELINE_REFERENCE.md`
- `.planning/references/260423-R171_CHRONO_WAN_TEMPORAL_EDIT_DEFER_REFERENCE.md`

### Targeted contract proof

- Add or update targeted backend tests that directly prove the new contract:
  - `qwen_image_edit` accepted through `mode="img2img"`
  - canonical normalized `mode` becomes `img2img`
  - ordered `reference_images` normalize into a deterministic list with a valid `main_reference_index`

### Targeted regression coverage

- Backend:
  - `python -m unittest tests.test_img2img_translation`
  - any directly affected capability/registry tests if request/translation surfaces require it
- Frontend unit tests only if a touched shared JS contract file requires them

### Final full-gate sweep

1. `pre-commit run detect-secrets --all-files`
2. `pre-commit run --all-files --show-diff-on-failure`
3. backend unit tests via `powershell -File scripts/run_full_tests_windows.ps1`
4. frontend Playwright harness per `tests/E2E_TESTING_SOP.md`
5. `npm run test:types` via the same full-gate wrapper

## Acceptance Criteria

- Official image-edit profiles can be normalized through the `img2img` request contract without requiring mask input.
- The normalized contract exposes ordered reference images plus a main-reference index.
- Legacy single-image callers still work.
- Legacy `mode="edit"` input remains accepted as a compatibility alias in this phase.
- Translation output reflects the new `img2img`-owned contract.
- Targeted regression tests pass.
- Full repository validation gate passes and is recorded in the implementation record.
