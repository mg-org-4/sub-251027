# Phase 86 Plan - Qwen / Qwen+ ImageEdit Runtime Adapter Expansion

## Branch Execution Note

- Target branch: `dev`
- Reason/source: `.planning/ROADMAP.md` phase 86 marks `F165` as the family-adapter delivery step that follows the shared image-edit foundation and must land on `dev` first.
- Merge condition: only after full `tests/TEST_SOP.md` validation plus review.

## Source-of-Truth Override

- Explicit current-session user instruction requires roadmap/plan/reference/record artifacts to be written in `.planning/` and accepted through sequential commits.
- This overrides the repo-standard `.gitignore` non-tracking rule for `.planning/` artifacts for this execution chain only.

## Scope

### In scope

- Expand the current Qwen-family image-edit delivery from the single shipped `qwen_image_edit` lane into four bounded official profiles:
  - `qwen_image_edit`
  - `qwen_image_edit_multi_lora`
  - `firered_image_edit`
  - `firered_image_edit_lightning`
- Keep all four profiles on the accepted image-edit contract:
  - `img2img`-owned request surface
  - no user mask
  - ordered direct reference images
- Extend manifest/profile metadata, selector hints, and fallback bootstrap data for the new Qwen-family edit lanes.
- Expand the non-SD edit builder to cover:
  - Qwen single-image encoder path
  - Qwen template-owned triple-LoRA chain path
  - Qwen-Edit-Plus / FireRed multi-reference encoder path
- Enforce truthful host prerequisites:
  - FireRed base profile does not require a template-owned LoRA
  - FireRed lightning profile does require the official lightning LoRA
  - Qwen single and Qwen triple-chain lanes require the official Qwen edit lightning LoRA
- Add targeted backend/frontend regressions for manifest metadata, selector resolution, builder topology, and edit-surface preset exposure.

### Out of scope

- Flux/Kontext/Klein/Longcat image-edit adapters (`F166-F167`)
- `img2img` pane redesign and removal of the separate `Edit` UI lane (`F168`)
- live-smoke catalog expansion and broader execute-matrix work (`F169`)
- Chrono/Wan temporal edit delivery

## Design Changes

### Manifest and selector matrix

- Add new manifest-backed edit profiles for:
  - Qwen multi-LoRA chain
  - FireRed base
  - FireRed lightning
- Keep public surface exposure on `available_surface_flows=("edit",)` until `F168`.
- Use profile metadata to declare:
  - encoder family
  - direct-reference mode/count
  - template-owned LoRA chain mode
  - profile-specific official template files and selector hints

### Qwen-family runtime builder

- Generalize the current qwen edit builder into a manifest-aware Qwen-family edit builder.
- Use `template_lora_chain_mode` to drive template-owned LoRA stack depth instead of hard-coding one specific chain.
- Add a `TextEncodeQwenImageEditPlus` encode path for FireRed and other Qwen+ style profiles.
- Keep ordered `reference_image_assets` and `main_reference_index` as the source of truth for direct input ordering.

### Truthful host prerequisite handling

- Continue failing fast when a profile declares a required template-owned LoRA and the host inventory does not expose it.
- Do not require template-owned LoRA for FireRed base because the official template ships a non-lightning baseline branch.
- Keep diffusion-model, VAE, and text-encoder resolution manifest-derived and profile-aware.

### Frontend/bootstrap synchronization

- Update the shipped frontend fallback registry/preset metadata so the browser bootstrap contract matches the backend profile matrix.
- Update current edit-surface expectations to show the new Qwen-family profiles on the existing `Edit` lane until `F168` merges them into `img2img`.

## Security Implications

- Positive: stronger profile-specific prerequisite checks reduce the risk of silently constructing the wrong official graph when host assets are missing.
- Positive: bounded multi-reference limits keep first-wave Qwen+ support explicit and prevent accidental unbounded image fan-in.
- No new network, filesystem, or privileged execution surface is introduced.

## Failure Modes and Rollback

- Failure mode: FireRed base and lightning lanes collapse into the same runtime defaults.
  - Mitigation: pin base-vs-lightning defaults and template-owned LoRA presence in translation tests.
- Failure mode: Qwen multi-LoRA chain is applied in the wrong order relative to inline LoRAs.
  - Mitigation: pin exact `LoraLoaderModelOnly` chain topology in backend tests.
- Failure mode: frontend fallback and backend registry drift, causing edit preset mismatches.
  - Mitigation: update backend capability tests, frontend API tests, extension tests, and Playwright edit preset expectations together.
- Rollback:
  - remove the new profiles, restore the prior single-path qwen edit builder, revert roadmap status, and remove associated fallback/test changes.

## Test Plan

Reference sources:
- `tests/TEST_SOP.md`
- `tests/E2E_TESTING_NOTICE.md`
- `tests/E2E_TESTING_SOP.md`
- `.planning/references/260423-F165_QWEN_QWENPLUS_IMAGEEDIT_REFERENCE.md`

### Targeted regression proof

- backend/tests:
  - `tests.test_model_family_registry`
  - `tests.test_capabilities`
  - `tests.test_model_inventory`
  - `tests.test_img2img_translation`
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

- The registry/preset/capability matrix exposes the new Qwen-family edit profiles with truthful metadata.
- `qwen_image_edit_multi_lora` builds the official three-node template-owned model-only LoRA chain before any inline LoRAs.
- FireRed base and FireRed lightning both translate through the non-SD edit runtime using `TextEncodeQwenImageEditPlus`.
- FireRed supports up to three direct reference images and rejects higher direct counts.
- Current frontend bootstrap/edit-surface tests reflect the expanded Qwen-family profile list without breaking the existing edit-mode maskless behavior.
- The full repository validation gate passes and is recorded in the implementation record.
