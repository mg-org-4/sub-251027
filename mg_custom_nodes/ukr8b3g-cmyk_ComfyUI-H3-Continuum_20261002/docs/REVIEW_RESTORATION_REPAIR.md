# Review restoration after workflow-tab switching

## What changed

Returning to a workflow could hide the Review actions or disable Take application while the saved Take data still existed. The frontend could record a settings snapshot before upstream nodes, links and deferred configuration had finished restoring, then mistake the completed restoration for a user edit.

History loading now waits for graph restoration and the existing deferred configuration passes. Graph/node request guards discard callbacks and responses from obsolete or detached nodes. Initial history reads preserve saved Take-selection fields. Loading and failed-history states have distinct summaries. Transient Review widgets remain excluded from workflow serialization.

Real generation-setting edits still hide the three Review actions and disable Use This Take / Continue From Here. Returning to the original settings restores these controls. No automatic Take application or generation is performed during restoration.

## Verified locally

- Real production JavaScript fixture: 60/60 cases (48 existing plus 12 lifecycle cases). The previous code fails seven cases in the expanded fixture.
- Focused CPU tests: 108 passed, no failures; one existing pynvml deprecation warning. This is not a new full CPU suite.
- Chrome: five V3.9-copy / V3.8X2-copy / V3.9-copy round trips, with the V3.9 Review actions restored on return to the V3.9 copy. Render History retained its one compatible Take.
- Chrome real-edit guard: change Base Seed by one; confirm the three Review actions disappear and Use This Take / Continue From Here are disabled. Restore the original seed; confirm all five controls become enabled again.
- Saved workflow and Run project.json hashes remained unchanged after the guard test; no Take was applied and no generation was queued.

The browser fixture uses a historical Run whose stored generation settings differ from the loaded diagnostic workflow. It validates frontend restoration and edit guarding only, not backend Take compatibility or GPU continuation. Previous/Next remained disabled because the fixture had only one compatible Take.

## Scope and update

The runtime change is frontend-only in web/project_id.js. Sampling, Run Storage schemas, State/Session/Assembly contracts, official workflows, package version, historical Releases and Registry publication are unchanged. New regression tests and integrity manifests accompany the fix.

Update the custom node from main and refresh the browser (hard-refresh if necessary) before testing. If the Review controls still disappear, report the ComfyUI/frontend versions and the exact tab-switch sequence, including whether settings were edited. Do not delete saved Runs or Takes to work around this UI issue.

No GPU continuation or new GitHub Release / Registry publication was performed for this repair.
