# Phase 84 Plan - Official ImageEdit Re-baseline and Inventory Freeze

## Branch Execution Note

- Target branch: `dev`
- Reason/source: `.planning/ROADMAP.md` phase 84 marks this chain as `dev`-only because it resets public flow classification and future runtime/UI ownership for official image-edit models.
- Merge condition: only after full `tests/TEST_SOP.md` validation plus review; this specific item is documentation/planning-only, so the documentation-only test exception applies unless code changes are introduced.

## Source-of-Truth Override

- Explicit current-session user instruction requires roadmap/plan/reference/record artifacts to be written in `.planning/` and accepted through sequential commits.
- This overrides the repo-standard `.gitignore` non-tracking rule for `.planning/` artifacts for this execution chain only.

## Scope

### In scope

- Create a fresh image-edit reference synthesis grounded in:
  - `reference/workflow_templates/imageEdit/*`
  - `reference/ComfyUI/comfy_extras/*`
  - `reference/ComfyUI-EditUtils/*`
- Replace the old planning assumption that official edit models belong to a separate single-reference `Edit` surface.
- Record the authoritative planning rules:
  - image-edit belongs to `img2img`
  - image-edit does not require mask input
  - multiple reference images are a first-class contract
- Freeze the real adapter-family grouping for later implementation:
  - Qwen / Qwen+
  - Flux / Kontext / Klein / Longcat
  - Chrono / Wan temporal edit
- Synchronize the new planning artifacts with the already-updated roadmap phase ordering.

### Out of scope

- Any runtime code changes.
- Any request-contract or manifest code changes.
- Any UI changes.
- Any host-smoke/runtime acceptance claims.

## Design Changes

### API / config / data-flow

- No runtime/API/config changes for this item.
- Planning artifacts added:
  - reference memo
  - implementation plan
  - implementation record
  - command log

### Planning decisions being frozen

- `qwen_image_edit` is the current accepted baseline, but it is not the long-term contract.
- Official image-edit flows must be planned as `img2img` subtypes, not a dedicated public `Edit` surface.
- Multi-reference support must be treated as part of the default architecture rather than an optional afterthought.

## Security Implications

- None for runtime behavior because this item is planning/documentation-only.
- Indirectly positive: the new freeze reduces later implementation risk by preventing a false single-reference / dedicated-surface design from being expanded further.

## Failure Modes and Rollback

- Failure mode: reference memo overstates what `ComfyUI-EditUtils` can define.
  - Mitigation: explicitly state that official workflow templates remain the primary source of truth and `EditUtils` is implementation guidance only.
- Failure mode: later phases inherit ambiguous family grouping.
  - Mitigation: freeze the topology groups explicitly in the reference memo and plan.
- Rollback:
  - revert the new planning artifacts if the reference synthesis is found to be incorrect
  - restore the previous roadmap interpretation only with explicit user approval

## Test Plan

Reference sources:
- `tests/TEST_SOP.md`
- `tests/E2E_TESTING_NOTICE.md`
- `tests/E2E_TESTING_SOP.md`

### Documentation-only applicability

- This item touches planning/reference documents only.
- Per `tests/TEST_SOP.md`, full automated test execution is optional for documentation-only changes.

### Verification steps for this item

1. Review the created reference memo for:
   - authoritative template inventory coverage
   - explicit `img2img` / no-mask / multi-reference rules
   - correct adapter-family grouping
2. Confirm roadmap linkage remains consistent with the new memo.
3. Record the documentation-only test exception in the implementation record.

## Acceptance Criteria

- A dated reference memo exists under `.planning/references/` for this item.
- The memo explicitly supersedes the old dedicated-`Edit` / single-reference planning assumption.
- The memo freezes the three main planning rules:
  - image-edit is `img2img`
  - image-edit does not require mask input
  - multiple reference images are first-class
- The memo freezes the adapter-family grouping used by later phases.
- A dated implementation record and command log exist and cite the documentation-only verification path.
