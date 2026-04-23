# Phase 84 Plan - Chrono/Wan Temporal Edit Scope Split and Defer Contract

## Branch Execution Note

- Target branch: `dev`
- Reason/source: `.planning/ROADMAP.md` phase 84 requires this defer contract before any request/manifest/runtime expansion starts on the new image-edit chain.
- Merge condition: only after full `tests/TEST_SOP.md` validation plus review; this specific item is documentation/planning-only, so the documentation-only exception applies unless code changes are introduced.

## Source-of-Truth Override

- Explicit current-session user instruction requires roadmap/plan/reference/record artifacts to be written in `.planning/` and accepted through sequential commits.
- This overrides the repo-standard `.gitignore` non-tracking rule for `.planning/` artifacts for this execution chain only.

## Scope

### In scope

- Freeze the rule that `Chrono Edit 14B` is not part of the first-wave static image-edit rollout.
- Generalize the defer to Wan-style temporal/video-like edit graphs by category.
- Record why the defer exists using current official template and host-node evidence.
- Synchronize roadmap status and working references for `R171`.

### Out of scope

- Any runtime changes.
- Any request-contract or UI changes.
- Designing the future temporal/video edit feature set.
- Revising first-wave static image-edit family priorities.

## Design Changes

### API / config / data-flow

- No runtime/API/config changes for this item.
- Planning output only:
  - dedicated defer reference
  - implementation plan
  - implementation record
  - command log
  - roadmap synchronization

### Planning rule being frozen

- First-wave image-edit delivery covers static image-edit templates only.
- Temporal/video-like Wan lineage remains explicitly deferred and must not be implied by first-wave acceptance.

## Security Implications

- None for runtime behavior because this is a planning-only defer contract.
- Indirectly positive because it narrows first-wave scope and reduces the risk of shipping an underspecified temporal/media contract.

## Failure Modes and Rollback

- Failure mode: the defer is written too narrowly and later only excludes `Chrono Edit 14B` by filename.
  - Mitigation: phrase the contract in categorical terms around Wan-style temporal/video-like image-edit graphs.
- Failure mode: later acceptance claims forget to exclude deferred temporal edit.
  - Mitigation: state explicit acceptance implications in the roadmap item and reference memo.
- Rollback:
  - adjust or remove the defer only when a dedicated temporal-edit chain is explicitly opened

## Test Plan

Reference sources:
- `tests/TEST_SOP.md`
- `tests/E2E_TESTING_NOTICE.md`
- `tests/E2E_TESTING_SOP.md`

### Documentation-only applicability

- This item touches planning/reference documents only.
- Per `tests/TEST_SOP.md`, full automated test execution is optional for documentation-only changes.

### Verification steps for this item

1. Review the defer memo and confirm it cites:
   - `Chrono Edit 14B` workflow facts
   - host-side Wan/ROPE/image-batch nodes
   - explicit first-wave exclusion language
2. Confirm roadmap status is synchronized:
   - `R171` marked completed in phase 84
   - `R171` removed from the open backlog board
3. Record the documentation-only exception in the implementation record.

## Acceptance Criteria

- A dated defer reference exists under `.planning/references/`.
- The defer is expressed categorically for Wan-style temporal/video-like image-edit graphs, not only as a filename note.
- The roadmap marks `R171` completed and keeps later static-image items open.
- A dated implementation record and command log exist and cite the documentation-only verification path.
