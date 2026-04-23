# Phase 84 Implementation Record - Official ImageEdit Re-baseline and Inventory Freeze

## What Changed

- Added `.planning/references/260423-R170_OFFICIAL_IMAGEEDIT_REBASELINE_REFERENCE.md`
  - captures the current official image-edit template inventory
  - records the now-authoritative `img2img` / no-mask / multi-reference planning rules
  - groups future implementation into Qwen-family, Flux-family, and deferred Chrono/Wan topology buckets
- Added `.planning/plans/260423-R170_OFFICIAL_IMAGEEDIT_REBASELINE_AND_INVENTORY_FREEZE_PLAN.md`
  - freezes scope, non-goals, risks, rollback, and verification rules for `R170`
- Added `.planning/implementation_records/260423-R170_OFFICIAL_IMAGEEDIT_REBASELINE_AND_INVENTORY_FREEZE_IMPLEMENTATION_RECORD.md`
  - records this item's completion and verification basis
- Added `.planning/command_logs/260423-R170_OFFICIAL_IMAGEEDIT_REBASELINE_AND_INVENTORY_FREEZE_COMMAND_LOG.md`
  - captures the commands and manual review steps used during this item

## Why Changed

- The accepted repo still reflects an older assumption: official image-edit models live on a dedicated `Edit` surface and the shipped path is essentially single-reference.
- The current workspace references now prove that this is no longer a safe planning baseline.
- Without a fresh inventory freeze, later implementation items would risk extending the wrong public contract and the wrong runtime seams.

## Source-of-Truth Override

- Explicit current-session user instruction required planning/reference artifacts to be stored under `.planning/` and committed as part of the sequential execution chain.
- That instruction was treated as higher priority than the repo-standard `.gitignore` non-tracking rule for `.planning/`.

## Full Verification Evidence

- Date/environment: 2026-04-23, Windows PowerShell, `dev` branch
- Command log reference:
  - `.planning/command_logs/260423-R170_OFFICIAL_IMAGEEDIT_REBASELINE_AND_INVENTORY_FREEZE_COMMAND_LOG.md`

### Verification evidence

- Reviewed current accepted code seams:
  - `rookieui/contracts/family_template_manifest.py`
  - `rookieui/services/img2img.py`
  - `rookieui/services/workflow_builders/non_sd_templates.py`
  - `web/sidebar_tabs/rookieui_img2img_pane.js`
- Reviewed authoritative reference inputs:
  - `reference/workflow_templates/imageEdit/*`
  - `reference/ComfyUI/comfy_extras/*`
  - `reference/ComfyUI-EditUtils/*`
- Confirmed the roadmap already tracks this chain on `dev` and that the new memo aligns with that sequence.

### Automated test evidence

- No automated tests executed.
- Reason: this item is documentation/planning-only and qualifies for the documentation-only exception in `tests/TEST_SOP.md`.

## Known Limitations

- This item does not change runtime behavior yet.
- The repo still ships the older dedicated-`Edit` / single-reference implementation baseline until later items land.

## Follow-up Items

- `R171` must freeze the explicit Chrono/Wan defer contract before runtime expansion starts.
- `F162` must convert the planning rules into the canonical `img2img` request/route contract.
