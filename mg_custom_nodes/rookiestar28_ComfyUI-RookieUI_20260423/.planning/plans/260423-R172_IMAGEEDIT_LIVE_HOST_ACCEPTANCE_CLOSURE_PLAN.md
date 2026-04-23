# R172 Official ImageEdit Regression and Live-Host Acceptance Closure Plan

Date: 2026-04-23
Target branch: `dev`
Reason/source: `.planning/ROADMAP.md` phase 89 marks `R172` as a `dev`-branch closure item.
Merge condition: only after full `tests/TEST_SOP.md` validation and review evidence are recorded.

## Scope

In scope:

- close `R172` using the already shipped image-edit surface
- capture truthful live-host evidence for the asset-ready subset only
- keep the workspace boundary intact by avoiding edits to the external deployed ComfyUI tree
- update roadmap, reference memo, command log, and implementation record
- run the final Windows full-gate sweep per repository SOP

Out of scope:

- new product-scope image-edit families beyond the accepted phase-84 to phase-88 chain
- claiming live-host support for profiles that are not asset-ready on the validation environment
- modifying `A:\ComfyUI\custom_nodes\comfyui-rookieui` or any other outside-workspace deployment path
- reopening deferred temporal/video edit graphs frozen by `R171`

## Design Changes

API/config/data-flow changes planned for this closure:

- prefer a workspace-safe validation host rooted at `reference/ComfyUI` instead of the stale external `8188` deployment
- provide the reference host with external model access via `--extra-model-paths-config`
- mount the current repository as the loaded RookieUI custom node in the reference host `custom_nodes` folder
- run `scripts/run_host_embedded_e2e.py` against a profile-restricted asset-ready subset:
  - `qwen_image_edit`
  - `qwen_image_edit_multi_lora`
  - `klein_9b_kv_image_edit`
  - `longcat_image_edit`
- if validation reveals a subset member is still not runnable, narrow the claim further and record the reason explicitly

No user-facing API contract changes are planned unless a validation-only defect is discovered during closure.

## Security Implications

- no edits may be made outside the workspace boundary
- temporary host startup must bind only to localhost
- validation may read external model assets, but must not mutate external ComfyUI deployment files
- acceptance language must not overclaim runtime support beyond the proven subset

## Failure Modes and Rollback

Failure modes:

- reference host fails to boot in the available Python environment
- reference host boots but fails to load RookieUI or required edit nodes
- subset profile dry-run fails because a supposedly asset-ready model, VAE, text encoder, or LoRA is missing
- execute lane fails due host/runtime constraints
- final repository SOP gate fails after live-host acceptance work

Rollback / containment:

- stop and terminate the temporary reference host
- keep the external deployment untouched
- record the exact blocking cause in the implementation record
- if code/config support files were added only for the failed path, either remove them before acceptance or keep them only if still useful and fully validated

## Test Plan

Required reading order before final acceptance sweep:

1. `tests/TEST_SOP.md`
2. `tests/E2E_TESTING_NOTICE.md`
3. `tests/E2E_TESTING_SOP.md`

Validation stages:

1. Reuse the already green targeted regression evidence from the shipped image-edit chain and add any new targeted checks required by closure support changes.
2. Start a workspace-safe reference ComfyUI host on localhost with the current workspace RookieUI node loaded.
3. Run report-only host validation:
   - `.\.venv\Scripts\python.exe scripts/run_host_embedded_e2e.py --base-url http://127.0.0.1:<port> --validation-mode image-edit --profiles qwen_image_edit,qwen_image_edit_multi_lora,klein_9b_kv_image_edit,longcat_image_edit --skip-execute`
4. Run execute host validation on the same proven subset:
   - `.\.venv\Scripts\python.exe scripts/run_host_embedded_e2e.py --base-url http://127.0.0.1:<port> --validation-mode image-edit --profiles qwen_image_edit,qwen_image_edit_multi_lora,klein_9b_kv_image_edit,longcat_image_edit`
5. Run the full Windows repository gate per `tests/TEST_SOP.md`:
   - `powershell -File scripts/run_full_tests_windows.ps1`

If any stage fails, fix the root cause when it is inside the workspace; otherwise classify it as an external blocker and record truthful evidence.

## Acceptance Criteria

`R172` is accepted only if all of the following are true:

- a reference memo captures the authoritative host/runtime facts for this closure
- live-host evidence is captured against a host whose runtime fingerprint matches the current workspace
- acceptance language claims only the asset-ready subset actually proven by report + execute evidence
- roadmap status is updated to reflect the final truthful result
- command log and implementation record include dated evidence and exact commands
- the final Windows full repository SOP gate passes
