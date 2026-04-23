# ROADMAP

> Internal planning artifact. Not for public distribution.
>
> The material in this file, `reference/`, and `.planning/` is for internal planning and implementation guidance only.

Last updated: 2026-04-23 (`R172` is now closed on `dev` after fixing host-compatible image-edit execute drift, capturing workspace-safe restarted-host report/execute proof for the asset-ready `klein_9b_kv_image_edit` + `longcat_image_edit` subset, classifying `qwen_image_edit` variants as validation-host prerequisites because only the `2509` lightning label was available, and rerunning the full SOP gate green.)

Local note 2026-04-23: `F170` completed with frontend-only retirement of the visible `qwen_image_edit_multi_lora` `Img2Img` preset, preservation of hidden-profile compatibility for existing apply-back/runtime seams, synchronized README wording, refreshed shipped frontend asset fingerprint, targeted frontend regression proof, and a green full repository SOP gate on `dev`.
Local note 2026-04-23: `F168` completed with canonical `img2img` exposure for first-wave image-edit profiles, removal of the dedicated visible `Edit` mode, profile-aware mask suppression and mode gating, ordered multi-reference UI state plus `reference_images` / `main_reference_index` payload serialization, synchronized fallback/contract metadata, targeted backend/frontend/Playwright coverage, and a refreshed shipped-asset revision token.
Local note 2026-04-23: `R172` completed with a workspace-safe `reference/ComfyUI` validation host after the external `8188` deployment failed the runtime fingerprint freshness gate; the closure fixed two execute-only workflow drifts (`RookieUILoadAssetImage.asset_handle` alignment and `Flux2Scheduler.steps` scalar wiring), captured green report + execute evidence for `klein_9b_kv_image_edit` and `longcat_image_edit`, recorded `qwen_image_edit` / `qwen_image_edit_multi_lora` as validation-host prerequisites because the host only exposed `Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors`, and reran the full Windows SOP gate with the image-edit host lane enabled.
Local note 2026-04-23: `F166` completed with a significantly expanded `image_edit_foundation` seam: structured Kontext stitch/scale/encode bundles, mirrored `ReferenceLatent` chaining, Flux reference-method branch helpers, `FluxKVCache` model wrapping, Flux2 advanced sampler assembly bundles, dedicated backend unit coverage, and a green full repository gate without changing the currently shipped public image-edit profile set.
Local note 2026-04-23: `F165` completed with a broader Qwen-family image-edit runtime matrix: manifest-backed `qwen_image_edit_multi_lora`, `firered_image_edit`, and `firered_image_edit_lightning` profiles; a generalized Qwen/Qwen+ edit builder covering triple template-owned LoRA chaining plus `TextEncodeQwenImageEditPlus`; truthful FireRed base-vs-lightning prerequisite handling; synchronized frontend bootstrap fallbacks; targeted backend/frontend/Playwright regressions; and a green full repository gate.
Local note 2026-04-23: `F164` completed with a dedicated `image_edit_foundation` builder module for ordered reference bundles, reusable VAE latent creation, ordered `ReferenceLatent` chaining, Flux multi-reference method wrapping, manifest-backed direct-reference limit enforcement, and a qwen-edit builder migration onto the shared seam while keeping single-reference public behavior unchanged.
Local note 2026-04-23: `F163` completed with manifest-backed image-edit metadata (`image_edit_profile`, canonical request surface, reference-mode/count, encoder family, template-owned LoRA chain mode) propagated through presets, capabilities normalization, frontend bootstrap fallbacks, and a refreshed shipped-asset revision token; public `available_surface_flows` intentionally remain unchanged until `F168`.
Local note 2026-04-23: `F162` completed with canonical `img2img`-owned image-edit request normalization, ordered reference-image support, legacy single-image compatibility, updated `img2img-<profile>` translation naming, and a green full repository gate; the optional host-embedded report-only lane remained invalid because the active ComfyUI host fingerprint was stale.
Local note 2026-04-23: `R171` completed with a dedicated defer memo for `Chrono Edit 14B` and Wan-style temporal/video-like image-edit graphs, explicitly keeping them out of the first-wave static image-edit rollout and later acceptance claim.
Local note 2026-04-23: `R170` completed with a fresh authoritative image-edit reference memo that supersedes the older dedicated-`Edit` / single-reference planning baseline and freezes the new `img2img` / no-mask / multi-reference-first rules for later implementation.
Local note 2026-04-22: future RookieUI image-edit work now explicitly treats all image-edit workflows as `img2img` subtypes, forbids mask requirements on that chain, assumes multi-reference image input is common rather than exceptional, and adds `reference/ComfyUI-EditUtils` as a primary implementation reference for edit conditioning / CLIP-encoding design.
Local note 2026-04-15: frontend test-gap hardening completed with shipped-asset revision fingerprint guard, action-level fullscreen E2E assertions, and isolated bootstrap-state cleanup coverage.
Local note 2026-04-16: `R115` completed with green in-sync deployed-host prompt-parity dry-run plus execute evidence after the restarted ComfyUI host was synchronized to the accepted workspace commits.
Local note 2026-04-16: `R109` completed after phase-55 prompt-parity closure items (`F104`, `R114`, `R115`) were fully accepted and re-aggregated into the final SD-family parity evidence claim.
Local note 2026-04-17: `F107` completed with a profile-aware ControlNet live-host lane that now proves detect, dry-run, and execute behavior against a family-compatible shipped checkpoint/control-model pairing instead of assuming an SD15-only execute environment.
Local note 2026-04-17: phase 59 completed with `R122` guardrail closeout, including manifest-backed target-module path proof, facade size-budget/import-cycle regression coverage, and final live-host `full-pipeline` dry-run + execute validation on `dev`.
Local note 2026-04-17: `F108` completed with a dedicated ADetailer live-host lane that validates detector/runtime truthfulness, passthrough/custom ControlNet refinement topology, `img2img` skip-vs-append behavior, and fallback-safe execute evidence against the restarted ComfyUI host.
Local note 2026-04-17: `F109` completed with an auxiliary-pipeline live-host lane that proves synchronous Extras execution, PNG Info parse/inspect/apply-back behavior, and queue/history assertions against a real RookieUI-origin job while remaining truthful about inspect-only and missing-input contracts.
Local note 2026-04-17: `R120` completed with shared queue/post-state closure across the phase-58 live-host lanes, explicit reusable-output/job-lookup validation, and a green aggregate `full-pipeline` dry-run plus execute mode on the restarted host.
Local note 2026-04-17: `DOCSYNC` completed with README last-update and feature-section synchronization for the accepted live-host validation expansion, including explicit documentation of shipped `controlnet` / `adetailer` / auxiliary / `full-pipeline` smoke coverage.
Local note 2026-04-17: phase-59 planning intake added an extensibility-focused refactor chain for the remaining backend/service bottlenecks, centered on workflow-builder extraction, ControlNet/ADetailer vertical splits, integrated-feature registry consolidation, and refactor-specific hardening.
Local note 2026-04-17: the global backlog board was re-audited against current accepted code and live-host evidence; stale planned items already closed by later waves were removed from the open board, and phase-27 items `R62`/`R64` were closed as later absorbed work rather than left as phantom backlog.
Local note 2026-04-17: phase-61 planning intake added an A1111 `XYZ Plot` migration chain for RookieUI-native axis registry, queue-backed sweep sessions, grid asset assembly, integrated frontend delivery, and final live-host closure.
Local note 2026-04-17: `F120` completed with shipped OpenAI-compatible AI Assist execution, prompt-workbench language/theme-style delivery, integrated assist-pane apply-back behavior, and full-gate acceptance on `dev`.
Local note 2026-04-17: `R124` completed with dedicated prompt-workbench live-host route/state validation, truthful translate/AI-assist execute coverage, stale-host route detection, and final phase-60 closure on `dev`.
Local note 2026-04-17: `R125` completed with a frozen `xyz_plot` contract module covering route-family ownership, axis truthfulness tiers, queue-backed session/grid delivery models, and explicit A1111-to-RookieUI adaptation rules ahead of phase-61 runtime work.
Local note 2026-04-17: `F121` completed with extracted XYZ axis/value/estimate service modules, internal `/xyz-plot/axes` and `/xyz-plot/estimate` routes, registry-driven bootstrap exposure, targeted regression coverage, and full-gate acceptance on `dev`.
Local note 2026-04-17: `F122` completed with a queue-backed XYZ session runner, persistent session/cell ownership, internal run/list/detail/cancel routes, explicit prompt-submission metadata tagging, truthful non-runner-ready axis rejection, and full-gate acceptance on `dev`.
Local note 2026-04-17: `F123` completed with RookieUI-owned XYZ grid assembly, PNG metadata delivery, cached session-result materialization, optional sub-grid/lone-image delivery, and full-gate acceptance on `dev`.
Local note 2026-04-17: phase 62 intake added a bounded runtime robustness hardening chain focused on retained open issues only: ADetailer cache/cascade guardrails, ControlNet PromptServer shim concurrency, prompt nesting depth limits, and ControlNet tensor range normalization truthfulness.
Local note 2026-04-17: phase 63 intake added an XYZ Plot choice-axis parity follow-up for A1111-style multi-select dropdown behavior, CSV-compatible payload serialization, and regression closure on top of the shipped phase-61 surface.
Local note 2026-04-17: `R129`, `F129`, and `R130` completed with A1111-referenced XYZ choice-axis multi-select dropdown delivery, CSV-compatible serialization over the accepted estimate/run payload seam, and full-gate regression closure on `dev`.
Local note 2026-04-17: phase 64 intake added an XYZ Plot choice-panel visual hardening follow-up for inherited font sizing, wider wrapped value presentation, and tooltip-safe full-name access on long choice-backed checkpoint labels.
Local note 2026-04-17: `R131`, `F131`, and `R132` completed with CSS/DOM hardening that lets XYZ choice panels inherit surrounding UI font sizing, expand beyond trigger width for long names, wrap filenames safely, and retain full-gate acceptance on `dev`.
Local note 2026-04-17: phase 65 intake added an XYZ Plot interaction hotfix follow-up for outside-click collapse behavior and Fill-button select-all toggle symmetry on choice-backed axes.
Local note 2026-04-17: `R133`, `F133`, and `R134` completed with choice-dropdown outside-click / Escape collapse behavior, Fill toggle symmetry between select-all and clear-all, and full-gate bugfix acceptance on `dev`.
Local note 2026-04-17: phase 66 intake added an XYZ Plot results-parity hotfix follow-up for shared preview fullscreen wiring, explicit axis-descriptor framing in assembled grids, and automatic host-output mirroring for completed grid/sub-grid artifacts.
Local note 2026-04-17: `R135`, `F135`, and `R136` completed with shared preview fullscreen/zoom parity for XYZ results, explicit `X` / `Y` / `Z` descriptor framing in assembled grids, automatic host-output mirroring for main/sub-grid artifacts, and green live-host `xyz-plot --execute` confirmation on `dev`.
Local note 2026-04-17: phase 67 intake added an XYZ Plot primary-preview/progress hotfix follow-up for running-session partial grid previews, shared top-preview synchronization, focused debug-hotspot guard comments, and larger assembled-grid label typography.
Local note 2026-04-17: `R137`, `F137`, and `R138` completed with running-session partial `main_grid` preview delivery, shared txt2img/img2img primary-preview synchronization, focused guard comments at the new preview hotfix seams, larger assembled-grid axis-label typography, and green live-host `xyz-plot --execute` confirmation on `dev`.
Local note 2026-04-18: phase 68 intake added an XYZ Plot seed-policy parity follow-up for A1111-style `Keep -1 for seeds` behavior, per-axis `Vary seeds for X/Y/Z` toggles, and truthful fixed-seed metadata on top of the shipped XYZ session runner.
Local note 2026-04-18: phase 69 intake added an XYZ Plot control-surface visual hotfix follow-up for smaller Plot Options typography, action-row color parity, equal-width action buttons, and explicit button-row spacing alignment.
Local note 2026-04-18: phase 73 intake added a live-host freshness hardening chain after stale pre-restart host evidence was incorrectly accepted; the new objective is an import-time backend runtime fingerprint plus a smoke-runner hard gate that fails stale or fingerprint-less hosts before any validation lane executes.
Local note 2026-04-18: `F146` completed with import-time backend runtime fingerprint metadata exposed on `/rookieui/bootstrap` and `/rookieui/capabilities`, plus a top-level smoke-runner freshness hard gate that now classifies the currently stale pre-restart host as invalid live-host evidence before lane execution.
Local note 2026-04-18: `R150` completed after the restarted ComfyUI host reloaded the accepted workspace code and the `full-pipeline` live-smoke report/execute lanes both passed on the new freshness-gated validation baseline.
Local note 2026-04-18: phase 76 intake added a full official non-SD workflow-template alignment chain so every text-to-image template under `reference/workflow_templates` can be represented as a truthful RookieUI preset/profile and mapped to official topology/parameter semantics instead of generic SDXL-adapted defaults.
Local note 2026-04-18: phase 77 intake added a deferred i2i backlog freeze for explicit `Edit`-marked official templates, currently limited to `Chrono Edit 14B.json` until the broader edit-template set is provided.
Local note 2026-04-19: phase 78 intake added a manifest-driven family/template extensibility chain so future official-template families and edit-intake expansion can land through canonical manifest data plus bounded adapters, rather than parallel hand-edited registry/preset/UI/runtime/live-smoke maps.
Local note 2026-04-19: phase 80 completed with a shared non-SD model-only inline LoRA helper reused across shipped official txt2img/edit builders, plus targeted regression coverage proving template-owned-first ordering and truthful warning behavior for clip/TE drift on non-SD inline LoRA activations.
Local note 2026-04-19: phase 81 completed as a release-branch hotfix after Windows Git-Bash `pre-push` failed in the Playwright harness on a denied `4173` bind; pre-push now pins `ROOKIEUI_E2E_PYTHON` to the repo `.venv`, resolves a bindable fallback port through a shared Python helper, and has direct full-gate proof under an occupied default port.
Local note 2026-04-20: phase 82 completed with explicit SD-family `img2img` multi-inline LoRA chaining regression coverage, proving multiple prompt `<lora:...>` activations plus a selected LoRA continue to serially chain through the shared SD `LoraLoader(model+clip)` path without relying on inference from txt2img-only evidence.
Local note 2026-04-20: phase 83 completed with XYZ Plot UI polish on both txt2img/img2img surfaces: the redundant helper note was removed, the section border reverted to a solid rule, Estimate/Refresh gained distinct green/blue action treatments, the bottom action order now places Refresh before Run, and both shell-level and Playwright parity checks were updated to pin the new control-surface contract.

Detailed roadmap plan:
- `.planning/roadmap/260409-S01S02R01R02R03F01F02F03F04F05F06F07F08_ROOKIEUI_REPO_ROADMAP_PLAN.md`

Current working references for Phase 5 parity:
- `.planning/references/260410-R11F22R12F23F24R13F25_A1111_FORGE_NEO_RUNTIME_UI_REFERENCE.md`
- `.planning/plans/260410-R14F26_HOST_MODEL_INVENTORY_AND_FAMILY_ALIGNMENT_PLAN.md`
- `.planning/references/260410-R15F27_A1111_PROMPT_DSL_AND_INLINE_LORA_REFERENCE.md`

Current working references for Phase 26 ControlNet parity:
- `.planning/references/260412-R60F69F70F71R61_CONTROLNET_A1111_PARITY_REFERENCE.md`
- `.planning/plans/260412-R60F69F70F71R61_CONTROLNET_A1111_PARITY_PLAN.md`

Current working references for Phase 27-29 architecture modernization:
- `.planning/references/260412-R62R63R64R65R66R67F72R68R69_ARCHITECTURE_MODERNIZATION_REFERENCE.md`
- `.planning/plans/260412-R62R63R64R65R66R67F72R68R69_ARCHITECTURE_MODERNIZATION_PLAN.md`

Current working references for Phase 30 diffusion-model loader routing hotfix:
- `.planning/references/260412-R70F73R71_DIFFUSION_LOADER_ROUTING_REFERENCE.md`
- `.planning/plans/260412-R70_DIFFUSION_LOADER_CONTRACT_FREEZE_PLAN.md`
- `.planning/plans/260412-F73_DIFFUSION_MODEL_LOADER_ROUTING_IMPLEMENTATION_PLAN.md`
- `.planning/plans/260412-R71_LOADER_AUDIT_AND_REGRESSION_HARDENING_PLAN.md`
- `.planning/plans/260412-R71_TEST_GAP_HARDENING_PLAN.md`

Current working references for Phase 31 Forge-Neo ControlNet Integrated upgrade:
- `.planning/references/260412-R72F74F75F76R73_FORGE_NEO_CONTROLNET_INTEGRATED_REFERENCE.md`
- `.planning/plans/260412-R72F74F75F76R73_FORGE_NEO_CONTROLNET_INTEGRATED_PLAN.md`

Current working references for Phase 32 Adetailer planning re-baseline:
- `.planning/references/260414-R74F77F78F79F80F81R75_ADETAILER_A1111_FORGE_PARITY_REFERENCE.md`
- `.planning/references/260414-LOCALHOST_7860_ADETAILER_UI_PARITY_REFERENCE.md`
- `.planning/plans/260414-R74F77F78F79F80F81R75_ADETAILER_A1111_FORGE_PARITY_PLAN.md`

Current working references for Phase 33 non-Lightning default alignment hotfix:
- `.planning/references/260412-R76_QWEN_WAN_NON_LIGHTNING_DEFAULTS_REFERENCE.md`
- `.planning/plans/260412-R76_QWEN_WAN_NON_LIGHTNING_DEFAULTS_AND_SELECTOR_HARDENING_PLAN.md`

Current working references for Phase 34 ControlNet run-preprocessor and layout parity hotfix:
- `.planning/references/260412-R77F82R78_CONTROLNET_RUN_PREPROCESSOR_LAYOUT_PARITY_HOTFIX_REVISED_REFERENCE.md`
- `.planning/plans/260412-R77F82R78_CONTROLNET_RUN_PREPROCESSOR_LAYOUT_PARITY_HOTFIX_REVISED_PLAN.md`
- `.planning/references/260412-R77F82R78_CONTROLNET_LAYOUT_PARITY_VISUAL_EVIDENCE.md`

Current working references for Phase 35 Forge-Neo canvas parity (Img2Img full tabs + ControlNet):
- `.planning/references/260412-R79F83F84R80_FORGE_NEO_CANVAS_PARITY_IMG2IMG_FULL_TABS_CONTROLNET_REFERENCE.md`
- `.planning/plans/260412-R79F83F84R80_FORGE_NEO_CANVAS_PARITY_IMG2IMG_FULL_TABS_CONTROLNET_PLAN.md`
- `.planning/references/260412-R79F83F84R80_CANVAS_PARITY_VISUAL_EVIDENCE.md`

Current working references for Phase 36 canvas-brush and ControlNet preview/preprocessor hotfix:
- `.planning/references/260412-R81F85R82R83F86R84_CONTROLNET_CANVAS_BRUSH_PREVIEW_HOTFIX_REFERENCE.md`
- `.planning/plans/260412-R81F85R82R83F86R84_CONTROLNET_CANVAS_BRUSH_PREVIEW_HOTFIX_CHAIN_PLAN.md`

Current working references for Phase 37 RookieUI run-preprocessor non-response and canvas fidelity hotfix:
- `.planning/references/260413-R85F87F88R86_ROOKIEUI_CONTROLNET_RUN_PREPROCESSOR_CANVAS_HOTFIX_REFERENCE.md`
- `.planning/plans/260413-R85F87F88R86_ROOKIEUI_CONTROLNET_RUN_PREPROCESSOR_CANVAS_HOTFIX_PLAN.md`
- `.planning/plans/260413-R85_RUN_PREPROCESSOR_RUNTIME_SCOPE_AND_FAILURE_CONTRACT_FREEZE_PLAN.md`
- `.planning/implementation_records/260413-R85_RUN_PREPROCESSOR_RUNTIME_SCOPE_AND_FAILURE_CONTRACT_FREEZE_IMPLEMENTATION_RECORD.md`
- `.planning/plans/260413-F87_SOURCE_CANVAS_FIDELITY_AND_BRUSH_INDICATOR_PARITY_PLAN.md`
- `.planning/implementation_records/260413-F87_SOURCE_CANVAS_FIDELITY_AND_BRUSH_INDICATOR_PARITY_IMPLEMENTATION_RECORD.md`
- `.planning/plans/260413-F88_RUN_PREPROCESSOR_FEEDBACK_AND_PREVIEW_GATING_UX_PLAN.md`
- `.planning/implementation_records/260413-F88_RUN_PREPROCESSOR_FEEDBACK_AND_PREVIEW_GATING_UX_IMPLEMENTATION_RECORD.md`
- `.planning/plans/260413-R86_RUN_PREPROCESSOR_BACKEND_ROUTE_VERIFICATION_AND_REGRESSION_HARDENING_PLAN.md`
- `.planning/implementation_records/260413-R86_RUN_PREPROCESSOR_BACKEND_ROUTE_VERIFICATION_AND_REGRESSION_HARDENING_IMPLEMENTATION_RECORD.md`

Current working references for Phase 38 ControlNet extension-first detect backend alignment:
- `.planning/references/260413-R87F89R88_CONTROLNET_EXTENSION_DETECT_BACKEND_ALIGNMENT_REFERENCE.md`
- `.planning/plans/260413-R87F89R88_CONTROLNET_EXTENSION_DETECT_BACKEND_ALIGNMENT_PLAN.md`
- `.planning/implementation_records/260413-R87F89R88_CONTROLNET_EXTENSION_DETECT_BACKEND_ALIGNMENT_IMPLEMENTATION_RECORD.md`

Current working references for Phase 39 ControlNet fullscreen zoom and header parity hotfix:
- `.planning/plans/260413-R89F90R90_CONTROLNET_FULLSCREEN_ZOOM_HOVER_HEADER_PARITY_PLAN.md`
- `.planning/implementation_records/260413-R89F90R90_CONTROLNET_FULLSCREEN_ZOOM_HOVER_HEADER_PARITY_IMPLEMENTATION_RECORD.md`
- `.planning/references/260413-R89F90R90_CONTROLNET_FULLSCREEN_ZOOM_HOVER_HEADER_PARITY_VISUAL_EVIDENCE.md`

Current working references for Phase 40 ControlNet detect endpoint de-hardcoding hotfix:
- `.planning/references/260413-R91F91R92_CONTROLNET_EXTERNAL_ENDPOINT_DEHARDCODING_REFERENCE.md`
- `.planning/plans/260413-R91F91R92_CONTROLNET_7860_REMOVAL_AND_DETECT_ENDPOINT_GATING_PLAN.md`
- `.planning/implementation_records/260413-R91F91R92_CONTROLNET_7860_REMOVAL_AND_DETECT_ENDPOINT_GATING_IMPLEMENTATION_RECORD.md`

Current working references for Phase 41 Forge-native ControlNet host preprocessor alignment:
- `.planning/references/260413-R93F92R94_FORGE_NATIVE_CONTROLNET_BACKEND_ALIGNMENT_REFERENCE.md`
- `.planning/plans/260413-R93F92R94_FORGE_NATIVE_CONTROLNET_HOST_PREPROCESSOR_ALIGNMENT_PLAN.md`
- `.planning/implementation_records/260413-R93F92R94_FORGE_NATIVE_CONTROLNET_HOST_PREPROCESSOR_ALIGNMENT_IMPLEMENTATION_RECORD.md`

Current working references for Phase 42 ControlNet fullscreen zoom visibility and sync hotfix:
- `.planning/references/260413-R95F93R96_CONTROLNET_FULLSCREEN_ZOOM_VISIBILITY_SYNC_REFERENCE.md`
- `.planning/plans/260413-R95F93R96_CONTROLNET_FULLSCREEN_ZOOM_VISIBILITY_SYNC_HOTFIX_PLAN.md`
- `.planning/implementation_records/260413-R95F93R96_CONTROLNET_FULLSCREEN_ZOOM_VISIBILITY_SYNC_HOTFIX_IMPLEMENTATION_RECORD.md`

Current working references for Phase 43 ControlNet depth preprocessor deterministic hotfix:
- `.planning/references/260413-R97F94R98_CONTROLNET_DEPTH_PREPROCESSOR_DETERMINISTIC_REFERENCE.md`
- `.planning/plans/260413-R97F94R98_CONTROLNET_DEPTH_PREPROCESSOR_DETERMINISTIC_HOTFIX_PLAN.md`
- `.planning/implementation_records/260413-R97F94R98_CONTROLNET_DEPTH_PREPROCESSOR_DETERMINISTIC_HOTFIX_IMPLEMENTATION_RECORD.md`

Current working references for Phase 45 ControlNet preprocessor variant filter and dispatch parity:
- `.planning/references/260413-R101F96R102_CONTROLNET_PREPROCESSOR_VARIANT_FILTER_AND_DISPATCH_REFERENCE.md`
- `.planning/plans/260413-R101F96R102_CONTROLNET_PREPROCESSOR_VARIANT_FILTER_AND_DISPATCH_PLAN.md`
- `.planning/implementation_records/260413-R101F96R102_CONTROLNET_PREPROCESSOR_VARIANT_FILTER_AND_DISPATCH_IMPLEMENTATION_RECORD.md`

Current working references for Phase 47 Forge-style OpenPose-family execution hotfix:
- `.planning/plans/260413-R104_FORGE_STYLE_CONTROLNET_PREPROCESSOR_EXECUTION_HOTFIX_PLAN.md`
- `.planning/implementation_records/260413-R104_FORGE_STYLE_CONTROLNET_PREPROCESSOR_EXECUTION_HOTFIX_IMPLEMENTATION_RECORD.md`

Current working references for Phase 48 OpenPose-family schema-aware host flag coercion hotfix:
- `.planning/plans/260413-R105_OPENPOSE_SCHEMA_AWARE_FLAG_COERCION_HOTFIX_PLAN.md`
- `.planning/implementation_records/260413-R105_OPENPOSE_SCHEMA_AWARE_FLAG_COERCION_HOTFIX_IMPLEMENTATION_RECORD.md`

Current working references for Phase 49-52 A1111-native prompt parity rearchitecture:
- `.planning/references/260413-R106F97R107F98F99R108F100R109_A1111_PROMPT_PARITY_REARCHITECTURE_REFERENCE.md`
- `.planning/plans/260413-R106F97R107F98F99R108F100R109_A1111_PROMPT_PARITY_REARCHITECTURE_PLAN.md`
- `.planning/plans/260413-R107F98_SD15_PROMPT_PARITY_NODE_DELIVERY_PLAN.md`
- `.planning/implementation_records/260413-R107F98_SD15_PROMPT_PARITY_NODE_DELIVERY_IMPLEMENTATION_RECORD.md`
- `.planning/plans/260414-F99_SDXL_PROMPT_PARITY_NODE_DELIVERY_PLAN.md`
- `.planning/implementation_records/260414-F99_SDXL_PROMPT_PARITY_NODE_DELIVERY_IMPLEMENTATION_RECORD.md`

Current working references for Phase 53 host-default selector sentinel hotfix:
- `.planning/plans/260415-HOTFIX_GENERATE_INVALID_REQUEST_SENTINEL_DEFAULTS_PLAN.md`
- `.planning/implementation_records/260415-HOTFIX_GENERATE_INVALID_REQUEST_SENTINEL_DEFAULTS_IMPLEMENTATION_RECORD.md`
- `.planning/command_logs/260415-HOTFIX_GENERATE_INVALID_REQUEST_SENTINEL_DEFAULTS_COMMAND_LOG.md`

Current working references for Phase 54 native ADetailer detector + advanced ControlNet runtime:
- `.planning/references/260415-R111F101F102R112_NATIVE_ADETAILER_CONTROLNET_RUNTIME_REFERENCE.md`
- `.planning/plans/260415-R111F101F102R112_NATIVE_ADETAILER_CONTROLNET_RUNTIME_PLAN.md`
- `.planning/plans/260415-R111_NATIVE_ADETAILER_CONTROLNET_CONTRACT_FREEZE_PLAN.md`
- `.planning/implementation_records/260415-R111_NATIVE_ADETAILER_CONTROLNET_CONTRACT_FREEZE_IMPLEMENTATION_RECORD.md`
- `.planning/command_logs/260415-R111_NATIVE_ADETAILER_CONTROLNET_CONTRACT_FREEZE_COMMAND_LOG.md`
- `.planning/plans/260415-F101_NATIVE_ADETAILER_DETECTOR_RUNTIME_PLAN.md`
- `.planning/implementation_records/260415-F101_NATIVE_ADETAILER_DETECTOR_RUNTIME_IMPLEMENTATION_RECORD.md`
- `.planning/command_logs/260415-F101_NATIVE_ADETAILER_DETECTOR_RUNTIME_COMMAND_LOG.md`
- `.planning/plans/260415-F102_NATIVE_ADVANCED_CONTROLNET_RUNTIME_PLAN.md`
- `.planning/implementation_records/260415-F102_NATIVE_ADVANCED_CONTROLNET_RUNTIME_IMPLEMENTATION_RECORD.md`
- `.planning/command_logs/260415-F102_NATIVE_ADVANCED_CONTROLNET_RUNTIME_COMMAND_LOG.md`
- `.planning/plans/260415-R112_NATIVE_RUNTIME_CHAIN_HARDENING_PLAN.md`
- `.planning/implementation_records/260415-R112_NATIVE_RUNTIME_CHAIN_HARDENING_IMPLEMENTATION_RECORD.md`
- `.planning/command_logs/260415-R112_NATIVE_RUNTIME_CHAIN_HARDENING_COMMAND_LOG.md`

Current working references for Phase 55 SD-family prompt parity closure expansion:
- `.planning/references/260416-R113F103F104R114_SD_FAMILY_PROMPT_PARITY_CLOSURE_EXPANSION_REFERENCE.md`
- `.planning/plans/260416-R113F103F104R114_SD_FAMILY_PROMPT_PARITY_CLOSURE_EXPANSION_PLAN.md`
- `.planning/references/260416-R114_SD_FAMILY_PROMPT_PARITY_LIVE_HOST_SMOKE_REFERENCE.md`
- `.planning/plans/260416-R114_SD_FAMILY_PROMPT_PARITY_LIVE_HOST_SMOKE_LANE_PLAN.md`
- `.planning/references/260416-R115_IN_SYNC_DEPLOYED_HOST_PROMPT_PARITY_EVIDENCE_REFERENCE.md`
- `.planning/plans/260416-R115_IN_SYNC_DEPLOYED_HOST_PROMPT_PARITY_EVIDENCE_CAPTURE_PLAN.md`
- `.planning/references/260416-R109_SD_FAMILY_PROMPT_PARITY_CLOSURE_REFERENCE.md`
- `.planning/plans/260416-R109_SD_FAMILY_PROMPT_PARITY_CLOSURE_PLAN.md`
- `.planning/plans/260415-HOTFIX_ENTER_SHORTCUT_AND_SUMMARY_ROW_PLAN.md`
- `.planning/implementation_records/260415-HOTFIX_ENTER_SHORTCUT_AND_SUMMARY_ROW_IMPLEMENTATION_RECORD.md`
- `.planning/command_logs/260415-HOTFIX_ENTER_SHORTCUT_AND_SUMMARY_ROW_COMMAND_LOG.md`
- `.planning/plans/260414-F100_PROMPT_CAPABILITY_UI_API_TRUTHFULNESS_REALIGNMENT_PLAN.md`
- `.planning/implementation_records/260414-F100_PROMPT_CAPABILITY_UI_API_TRUTHFULNESS_REALIGNMENT_IMPLEMENTATION_RECORD.md`

Current working references for Phase 56 SD-family prompt parity maximal continuation:
- `.planning/references/260416-R116F105F106R117_SD_FAMILY_PROMPT_PARITY_MAXIMAL_CONTINUATION_REFERENCE.md`
- `.planning/plans/260416-R116F105F106R117_SD_FAMILY_PROMPT_PARITY_MAXIMAL_CONTINUATION_PLAN.md`
- `.planning/plans/260416-R116_REFERENCE_BACKED_SD_FAMILY_PROMPT_CONTINUATION_CONTRACT_FREEZE_PLAN.md`
- `.planning/plans/260416-F105_ALTERNATE_PROMPT_SCHEDULING_DELIVERY_PLAN.md`

Current working references for Phase 58 auxiliary live-host validation expansion:
- `.planning/references/260417-R119F107F108F109R120_LIVE_HOST_VALIDATION_CHAIN_REFERENCE.md`
- `.planning/plans/260417-R119F107F108F109R120_LIVE_HOST_VALIDATION_CHAIN_PLAN.md`

Current working references for Phase 59 extensibility refactor planning:
- `.planning/references/260417-R121F110F111F112F113R122_EXTENSIBILITY_REFACTOR_REFERENCE.md`
- `.planning/plans/260417-R121F110F111F112F113R122_EXTENSIBILITY_REFACTOR_PLAN.md`

Current working references for Phase 60 prompt-workbench migration planning:
- `.planning/references/260417-R123F114F115F116F117F118F119F120R124_PROMPT_WORKBENCH_MIGRATION_REFERENCE.md`
- `.planning/plans/260417-R123F114F115F116F117F118F119F120R124_PROMPT_WORKBENCH_MIGRATION_PLAN.md`

Current working references for Phase 61 A1111 XYZ Plot migration planning:
- `.planning/references/260417-R125F121F122F123F124R126_A1111_XYZ_PLOT_MIGRATION_REFERENCE.md`
- `.planning/plans/260417-R125F121F122F123F124R126_A1111_XYZ_PLOT_MIGRATION_PLAN.md`

Current working references for Phase 62 runtime robustness hardening:
- `.planning/references/260417-R127F125F126F127F128R128_RUNTIME_ROBUSTNESS_HARDENING_REFERENCE.md`
- `.planning/plans/260417-R127F125F126F127F128R128_RUNTIME_ROBUSTNESS_HARDENING_PLAN.md`

Current working references for Phase 63 XYZ Plot choice-axis parity follow-up:
- `.planning/references/260417-R129F129R130_A1111_XYZ_MULTISELECT_PARITY_REFERENCE.md`
- `.planning/plans/260417-R129F129R130_A1111_XYZ_MULTISELECT_PARITY_PLAN.md`

Current working references for Phase 64 XYZ Plot choice-panel visual hardening:
- `.planning/plans/260417-R131F131R132_XYZ_CHOICE_PANEL_VISUAL_HARDENING_PLAN.md`

Current working references for Phase 65 XYZ Plot choice-dropdown interaction hotfix:
- `.planning/plans/260417-R133F133R134_XYZ_DROPDOWN_INTERACTION_HOTFIX_PLAN.md`

Current working references for Phase 66 XYZ Plot results parity hotfix:
- `.planning/plans/260417-R135F135R136_XYZ_PREVIEW_LEGEND_PARITY_HOTFIX_PLAN.md`

Current working references for Phase 68 XYZ Plot seed-policy parity follow-up:
- `.planning/references/260418-R139F138R140_A1111_XYZ_SEED_POLICY_PARITY_REFERENCE.md`
- `.planning/plans/260418-R139F138R140_XYZ_SEED_POLICY_PARITY_PLAN.md`

Current working references for Phase 69 XYZ Plot control-surface visual hotfix:
- `.planning/plans/260418-R141F139R142_XYZ_CONTROL_SURFACE_VISUAL_HOTFIX_PLAN.md`

Current working references for Phase 76 official non-SD T2I workflow-template alignment:
- `.planning/references/260418-R155F150F151R156_OFFICIAL_NON_SD_T2I_TEMPLATE_ALIGNMENT_REFERENCE.md`
- `.planning/plans/260418-R155F150F151R156_OFFICIAL_NON_SD_T2I_TEMPLATE_ALIGNMENT_PLAN.md`

Current working references for Phase 77 official edit-template i2i backlog freeze:
- `.planning/references/260418-R157_OFFICIAL_EDIT_TEMPLATE_I2I_INTAKE_REFERENCE.md`
- `.planning/plans/260418-R157_OFFICIAL_EDIT_TEMPLATE_I2I_INTAKE_PLAN.md`

Current working references for Phase 78 manifest-driven family/template extensibility:
- `.planning/references/260419-R158F152F153F154R159_MANIFEST_DRIVEN_FAMILY_TEMPLATE_EXTENSIBILITY_REFERENCE.md`
- `.planning/plans/260419-R158F152F153F154R159_MANIFEST_DRIVEN_FAMILY_TEMPLATE_EXTENSIBILITY_PLAN.md`

Accepted implementation records:
- `.planning/implementation_records/260413-R106F97_A1111_PROMPT_PARITY_FOUNDATION_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260413-R107F98_SD15_PROMPT_PARITY_NODE_DELIVERY_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260414-F99_SDXL_PROMPT_PARITY_NODE_DELIVERY_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260414-F100_PROMPT_CAPABILITY_UI_API_TRUTHFULNESS_REALIGNMENT_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260412-R57F67F68R58_A1111_PROMPT_PARITY_AND_IMG2IMG_COMPLETION_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260412-R60F69F70F71R61_CONTROLNET_A1111_PARITY_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260412-R72_FORGE_NEO_CONTROLNET_CONTRACT_AND_ROUTE_FREEZE_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260412-R71_TEST_GAP_HARDENING_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260412-R76_QWEN_WAN_NON_LIGHTNING_DEFAULTS_AND_SELECTOR_HARDENING_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260412-R77F82R78_CONTROLNET_RUN_PREPROCESSOR_LAYOUT_PARITY_HOTFIX_REVISED_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260412-R79_FORGE_NEO_CANVAS_PARITY_CONTRACT_FREEZE_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260412-R79F83F84R80_FORGE_NEO_CANVAS_PARITY_IMG2IMG_FULL_TABS_CONTROLNET_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260412-R81_SOURCE_CANVAS_INTERACTION_CONTRACT_FREEZE_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260412-F85_SHARED_SOURCE_CANVAS_BRUSH_CONTROLS_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260412-R82_SOURCE_CANVAS_INTERACTION_REGRESSION_AND_ROLLBACK_HARDENING_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260412-R83_CONTROLNET_PREVIEW_PREPROCESSOR_CONTRACT_FREEZE_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260412-F86_CONTROLNET_DUAL_PANE_PREVIEW_AND_RUN_PREPROCESSOR_UI_LAYOUT_PARITY_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260413-R84_CONTROLNET_PREVIEW_PREPROCESSOR_REGRESSION_AND_ACCEPTANCE_HARDENING_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260413-R85_RUN_PREPROCESSOR_RUNTIME_SCOPE_AND_FAILURE_CONTRACT_FREEZE_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260413-F87_SOURCE_CANVAS_FIDELITY_AND_BRUSH_INDICATOR_PARITY_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260413-F88_RUN_PREPROCESSOR_FEEDBACK_AND_PREVIEW_GATING_UX_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260413-R86_RUN_PREPROCESSOR_BACKEND_ROUTE_VERIFICATION_AND_REGRESSION_HARDENING_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260413-R87F89R88_CONTROLNET_EXTENSION_DETECT_BACKEND_ALIGNMENT_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260413-R89F90R90_CONTROLNET_FULLSCREEN_ZOOM_HOVER_HEADER_PARITY_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260413-R91F91R92_CONTROLNET_7860_REMOVAL_AND_DETECT_ENDPOINT_GATING_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260413-R93F92R94_FORGE_NATIVE_CONTROLNET_HOST_PREPROCESSOR_ALIGNMENT_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260413-R95F93R96_CONTROLNET_FULLSCREEN_ZOOM_VISIBILITY_SYNC_HOTFIX_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260413-R97F94R98_CONTROLNET_DEPTH_PREPROCESSOR_DETERMINISTIC_HOTFIX_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260413-R101F96R102_CONTROLNET_PREPROCESSOR_VARIANT_FILTER_AND_DISPATCH_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260413-R104_FORGE_STYLE_CONTROLNET_PREPROCESSOR_EXECUTION_HOTFIX_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260413-R105_OPENPOSE_SCHEMA_AWARE_FLAG_COERCION_HOTFIX_IMPLEMENTATION_RECORD.md`

## Goal

Build ComfyUI-RookieUI as a ComfyUI custom node package that adds an A1111-style sidebar workflow to the ComfyUI frontend while keeping inference inside the ComfyUI backend and sharing the host ComfyUI model paths/runtime. The primary product target is full reproduction of A1111 inference behavior for the Stable Diffusion model family, especially SD1.5, SDXL, and SDXL-derived ecosystems such as Pony, Illustrious, and Noob, with a rookie-friendly presentation layered on top.

## Reference Baseline

- `reference/ComfyUI`: host backend, custom node loading, `WEB_DIRECTORY`, prompt execution.
- `reference/ComfyUI_frontend`: official sidebar extension API and host frontend behavior.
- `reference/desktop`: desktop-hosted frontend/runtime surface that must be kept compatible.
- `reference/stable-diffusion-webui`: baseline A1111 UX, parameter semantics, and generation flow.
- `reference/stable-diffusion-webui-forge`: newer model-family defaults, runtime/resource optimizations, and implementation ideas worth porting selectively.
- `reference/stable-diffusion-webui-reForge`: proof that A1111-style UX can run on a Comfy-derived backend, plus additional optimization patterns and newer-model support references.
- `reference/adetailer`: canonical Adetailer option semantics, detector/mask/refine behavior, and ControlNet coupling direction for integrated-pack planning.
- `reference/ComfyUI-OpenClaw`: best structural reference for sidebar packaging, route bootstrap, and tests.
- `reference/ComfyUI_smZNodes`: targeted parity fallback for prompt parsing and A1111-like behavior gaps.

## Priority Clarification

- First priority: reproduce A1111 inference behavior for the Stable Diffusion family, not just the A1111 visual layout.
- Primary parity scope: SD1.5, SDXL, and SDXL-derived ecosystems such as Pony, Illustrious, and Noob.
- Required parity areas include prompt parsing, conditioning semantics, sampler/scheduler behavior, seed/noise behavior, CFG-related behavior, img2img/inpaint semantics, hires-like workflows, PNG info round-trip, and other A1111-specific Stable Diffusion workflow features.
- SDXL authoring rule: preserve the A1111-style single prompt surface for SDXL-family workflows; independent user-facing `text_g` / `text_l` authoring is explicitly out of scope for parity planning unless product direction changes beyond A1111 reproduction.
- Forge and reForge are reference sources for:
  - A1111 optimizations worth porting into a Comfy-backed implementation
  - newer-model-era design ideas and runtime policy
  - selective support for Flux, Qwen-Image, and later families when complexity is justified
- Non-SD-family models such as Flux, Qwen-Image, Wan, ZiT, Klein, Lumina, and Anima are explicitly secondary scope.
  - Default rule: reuse existing ComfyUI-native inference and conditioning engines when practical instead of recreating Forge/reForge internals or forcing A1111-style parity layers onto newer families without a stable canonical reference.
  - Adoption rule: add or deepen support only when the implementation cost is justified and does not slow SD-family parity work.
- Optimization intake rule:
  - early and mandatory when it directly preserves SD-family A1111 behavior on a Comfy backend
  - later and opt-in when it mainly affects performance, memory policy, experimental schedulers, low-bit loading, or newer-family expansion
- Early optimization intake focus:
  - Forge-style shared checkpoint, VAE, and text-encoder discovery plus model-family presets
  - reForge-style A1111-conditioning to Comfy-backend translation patterns
  - reForge evaluation of dynamic clip-skip, NGMS, and cond/uncond padding only where they improve A1111 SD-family fidelity
- Later optimization intake focus:
  - runtime memory and offload controls, dtype policy, scheduler catalog expansion, and complexity-justified newer-family features
- Flux, Qwen-Image, Wan, ZiT, Klein, Lumina, Anima, and similar newer families remain in scope, but exact Forge/reForge feature parity for them is a later, complexity-sensitive decision rather than the immediate acceptance target.

### Execution Strategy Update (2026-04-11)

- Primary delivery route: execute frontend modularization chain first (`R51 -> R52 -> F61 -> F62 -> F63 -> R53`).
- Exception rule: blocker-level regression fixes that break active user flows (especially `img2img` / `inpaint` / `mask` paths) may interrupt the chain and are handled as fast-track hotfixes.
- Resume rule: after each fast-track hotfix closes and passes the full gate, immediately resume the modularization chain at the next unfinished item.

## Architecture Direction

- Package type: standard ComfyUI custom node extension with `WEB_DIRECTORY`.
- Frontend shape: one RookieUI sidebar shell with modular tabs and capability-gated panels.
- Backend shape: typed request normalization plus translation from A1111-style Stable Diffusion state into ComfyUI prompt graphs.
- Compatibility rule: A1111 defines the primary Stable Diffusion behavior contract; Forge and reForge provide optimization and extension references; ComfyUI defines the execution host.
- Reference naming rule: outside internal planning/reference documents, avoid exposing reference-repo names in shipped code/comments/metadata whenever practical. Exceptions are the host `ComfyUI` name plus the canonical A1111 / `stable-diffusion-webui` lineage when needed for truthful behavior scope.
- Native extension rule: all newly added extension features must be RookieUI-owned and shipped as native capability within this repo. Do not require users to install separate external custom-node packs for those features to function.
- Parity rule: Stable Diffusion family behavior parity is a core requirement starting in Phase 1, not a late polish layer.
- Model-path rule: RookieUI shares the host ComfyUI backend and model path configuration instead of creating a parallel model inventory.
- Newer-family rule: for non-SD families, prefer thin UI/state adapters on top of existing ComfyUI workflows before considering deeper Forge/reForge-style runtime ports.

## Branch Policy

- Default branch target for normal roadmap work: `dev`.
- Reason: workflow translation, host integration, and compatibility work carry high regression risk across both standalone and desktop hosts.
- Merge condition to `main`: full validation per `tests/TEST_SOP.md` plus review.

## Item Code Legend

- `Sxx`: Security
- `Rxx`: Robustness
- `Fxx`: Functionality

Status values:
- `Planned`
- `In Progress`
- `Blocked`
- `Done`

Priority values (open backlog):
- `P0`: Immediate dependency-critical implementation item, ready to execute, no gating dependency.
- `P1`: High-priority quality/safety baseline item required before expansion chains.
- `P2`: Feature-expansion item with explicit upstream dependency closure requirements.
- `P3`: Decision-gated or feasibility-spike item deferred behind delivery-critical chains.

Open-backlog priority ranking rules (objective-only):
- Primary sorter: dependency blocking effect on other open items.
- Secondary sorter: delivery value vs. exploration nature (shipping behavior first, spikes later).
- Tertiary sorter: explicit gated/deferred flags in phase execution policy.
- Tie-breaker: existing `Order` value.

## Phase 0 - Foundation

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 1 | R01 | Done | Extension Bootstrap and Host Surface Baseline | `dev` | Create the custom node package skeleton, route bootstrap pattern, host detection, and stable load path. |
| 2 | S01 | Done | Internal Route and Asset Boundary Hardening | `dev` | Keep RookieUI routes internal, constrain file/path handling, and prevent unsafe asset or metadata access. |
| 3 | F01 | Done | Rookie Sidebar Shell and Capability Bootstrap | `dev` | Register the sidebar tab, load capabilities, and mount a modular rookie-first UI shell. |

## Phase 1 - MVP Generation

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 4 | R02 | Done | SD-Family Parity Matrix and Workflow Translation | `dev` | Define typed request contracts, explicit A1111 parity targets for SD1.5/SDXL derivatives, the A1111-state to Comfy-graph translation layer, and parity-critical Forge/reForge bridge patterns. |
| 5 | F02 | Done | Txt2Img MVP with A1111 SD Parity | `dev` | Deliver end-to-end txt2img with A1111-oriented prompt/state handling, size/steps/CFG/sampler/scheduler/seed/batch behavior, and Stable Diffusion-family parity targets, with later prompt-DSL gaps tracked explicitly in Phase 5. |
| 6 | F03 | Done | Shared Model Discovery and SD-Family Profiles | `dev` | Normalize host ComfyUI checkpoint, VAE, and text-encoder metadata; add Forge-inspired model-family presets; and provide SD1.5, SDXL, Pony, Illustrious, and Noob-oriented defaults before expanding newer families. |

## Phase 2 - Workflow Expansion

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 7 | F04 | Done | Img2Img and Inpaint Workflow | `dev` | Add image-to-image and mask-driven workflow paths with rookie-safe controls. |
| 8 | F05 | Done | PNG Info and Parameter Round-Trip | `dev` | Parse infotext or PNG metadata back into RookieUI state and normalize unsupported fields. |
| 9 | F07 | Done | Queue, History, and Result Reuse | `dev` | Surface queue/progress/history inside the sidebar and support sending results back into follow-up flows. |

## Phase 3 - Hardening and Parity

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 10 | S02 | Done | Request Validation and Generation Safety Guardrails | `dev` | Enforce request sanitization, parameter bounds, and defensive handling around uploads, metadata import, and parity-sensitive generation inputs. |
| 11 | R03 | Done | Regression Harness and Desktop Parity Sweep | `dev` | Build repeatable backend/frontend/E2E coverage for standalone and desktop hosts, including SD-family parity lanes and opt-in optimization regression lanes. |
| 12 | F06 | Done | Extended A1111/Forge Compatibility and Optimization Layer | `dev` | Isolate advanced parity helpers and selectively port Forge/reForge optimizations such as runtime memory policy, offload and dtype controls, scheduler expansion, and complexity-justified newer-family support without destabilizing the core shell. |

## Phase 4 - Guided Expansion

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 13 | F08 | Done | Guided Advanced Controls | `dev` | Add curated advanced panels such as hires-like flows and model-family-specific controls with progressive disclosure. |

## Post-Roadmap Hotfixes

| Date | Code | Status | Title | Summary |
| --- | --- | --- | --- | --- |
| 2026-04-09 | R04 | Done | Sidebar Minimum Width Guard | Constrain RookieUI sidebar host width and collapse form grids safely before controls clip in narrow ComfyUI sidebar states. |
| 2026-04-09 | F09 | Done | A1111 Classic Sidebar Relayout | Collapse visible preset/profile duplication, switch to tabbed A1111-style panes, and simplify the header to title, version, and GitHub entrypoint. |
| 2026-04-09 | R05 | Done | Frontend Asset Cache-Busting Hotfix | Add revisioned frontend asset URLs so live ComfyUI does not keep serving stale pre-relayout RookieUI shell resources from browser or host cache. |
| 2026-04-09 | R06 | Done | Tab Pane Visibility Fix | Replace pane visibility toggling with explicit active-pane classes so tab clicks switch real content instead of only restyling the active button. |
| 2026-04-09 | F10 | Done | A1111 Width and Layout Refinement | Raise the sidebar width floor, compact the tabs/header, switch to orange A1111-style accents, and keep the form layout fixed instead of responsive reflow. |
| 2026-04-09 | R07 | Done | Live Tab Pane Isolation Hardening | Add DOM-level pane hiding and revisioned asset loading so top tabs switch real visible content reliably in the live ComfyUI host. |
| 2026-04-09 | F11 | Done | Remove Shared Model Inventory Surface | Remove the visible Shared Model Inventory summary panel while keeping model selectors inside generation panes. |
| 2026-04-09 | R08 | Done | Numeric Input Contract Hardening | Add explicit decimal-capable numeric input metadata only where RookieUI fields semantically allow floats, while preserving integer-only contracts elsewhere. |
| 2026-04-09 | F12 | Done | Optional Hires Validation Fix | Stop disabled hires controls from blocking `txt2img` submission by gating backend hires validation on `hires_enabled`. |
| 2026-04-09 | F13 | Done | Inline Hires Checkbox Layout Fix | Replace the oversized generic checkbox field with a compact inline `Enable Hires` control aligned to A1111-style expectations. |
| 2026-04-09 | R09 | Done | Forge-Neo Sidebar Layout Contract | Adapt the `sd-webui-forge-neo` main page skeleton into a host-safe ComfyUI sidebar layout contract, including backend-sync rules for newly surfaced controls. |
| 2026-04-09 | F14 | Done | Forge-Neo Quicksettings and Shell Relayout | Add a Forge-Neo-inspired quicksettings strip and slimmer main tab bar while preserving RookieUI control wiring and the required header identity. |
| 2026-04-09 | F15 | Done | Forge-Neo Generation Workspace Relayout | Rebuild `txt2img` and `img2img` panes around a prompt band, generation sub-tabs, split left-parameter/right-preview workspace, and synced backend seams for low-bit and single-LoRA intake. |
| 2026-04-09 | R10 | Done | Host Selector and A1111 Seed Semantics Hardening | Preserve exact host checkpoint selectors instead of rewriting path separators, and convert A1111 `seed=-1` semantics into valid ComfyUI execution seeds before prompt submission. |
| 2026-04-09 | F16 | Done | Forge-Neo Density and Prompt-Band Refinement | Compact oversized quicksetting controls, tighten select/button density, and rebalance the prompt band so prompt textareas gain more usable space while keeping A1111/Forge-Neo proportions proportionally downscaled for the sidebar host. |
| 2026-04-09 | F17 | Done | Sampler and Scheduler Selector Restoration | Replace text inputs with real dropdown selectors for generation sampling controls and back them with explicit backend catalog payloads. |
| 2026-04-09 | F18 | Done | Forge-Neo Visual Detail Parity Intake | Port the next layer of visual parity from the live Forge-Neo UI, including normal/dark background-tone distinction, compact icon actions, badges/counters, footer-detail cues, and A1111-style slider-backed parameter controls that fit within the RookieUI sidebar shell. |
| 2026-04-10 | F19 | Done | Quicksettings Density Refinement | Further reduce the top quicksettings selector height and spacing so the Forge-Neo quicksettings row remains proportionally smaller than prompt/workspace controls inside the sidebar host. |
| 2026-04-10 | F20 | Done | Generate Rail and Preview Icon Parity | Add A1111-style compact icon controls beneath the hero Generate button and below the preview box while keeping every new control wired to a real internal RookieUI action. |
| 2026-04-10 | F21 | Done | A1111 Slider Color Refinement | Replace the current orange slider palette with a sidebar-safe white-track and blue-accent slider treatment aligned with A1111 expectations. |
| 2026-04-10 | F34 | Done | Extras Hero Rail Top-Edge Alignment Fix | Align Extras `Generate` hero button top edge with the left `Single Image` frame top edge under live host rendering. |
| 2026-04-10 | F35 | Done | Global Slider/Selector Density Reduction (Pass 2) | Reduce slider and dropdown chrome size again to better match compact A1111/Forge-Neo sidebar proportions. |
| 2026-04-10 | R19 | Done | Img2Img Asset Handle Preflight Guard | Reject missing or unknown img2img/inpaint asset handles at RookieUI request-normalization time instead of letting host queue jobs fail later. |
| 2026-04-10 | R20 | Done | PNG Info Apply Button Height Lock | Harden PNG Info `Apply to...` button dimensions against host CSS interference so controls do not stretch into full-height columns. |

## Phase 5 - Sidebar Semantic Cleanup and Postprocessing Completion

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 22 | R14 | Done | Host Model Inventory Coverage and Family-Aware Loader Baseline | `dev` | Expand RookieUI beyond the current checkpoint-first host inventory so it can read the relevant ComfyUI model folders, including `checkpoints`, `clip`, `clip_vision`, `controlnet`, `diffusion_models`, `embeddings`, `loras`, `text_encoders`, `ultralytics`, `unet`, `upscale_models`, and `vae`, while establishing a family-aware loader baseline. |
| 23 | F26 | Done | Host Model Catalog Surface Expansion | `dev` | Expand frontend/backend catalog payloads to consume the broader host inventory while keeping the visible controls compact and family-aware. |
| 24 | R15 | Done | A1111 Prompt DSL and Extra-Network Parsing Baseline | `dev` | Add a dedicated prompt preprocessing seam for A1111 prompt attention, scheduling, `AND`/`BREAK`, and generic `<name:...>` extra-network extraction, using A1111 as the canonical source and ComfyUI_smZNodes only as a partial fallback. |
| 25 | F27 | Done | Inline LoRA Syntax and Prompt Token Parity | `dev` | Make prompt-side A1111 syntax such as `<lora:name:0.8>` execute as real backend behavior by stripping inline tags from encoded prompt text, resolving host LoRA selectors, and injecting deterministic workflow-side LoRA activation. |
| 26 | R11 | Done | SD-Family Control Semantics and RookieUI-Origin Queue Hardening | `dev` | Make control exposure family-aware, remove decorative SD1.5 text-encoder behavior, and tag/filter RookieUI-origin jobs while still using the host ComfyUI queue. |
| 27 | F22 | Done | Sidebar Control Density and Typography Correction | `dev` | Restore readable quicksetting labels, shrink the control boxes instead of the label text, normalize prompt and slider typography, align the Generate button height to the prompt field, and follow the visual/runtime primitives captured in the Phase 5 reference note. |
| 28 | R12 | Done | PNG Info Dual-Metadata Ingest and Asset-Handle Bridge | `dev` | Rebuild PNG Info backend around image-first inspection for A1111 and ComfyUI metadata, with safe internal asset handles for downstream apply flows. |
| 29 | F23 | Done | PNG Info Image-First Surface and A1111 Apply Flow | `dev` | Replace the prefilled infotext pane with an image-first inspector, show warnings only when present, and allow one-click apply into txt2img and img2img for A1111 metadata images only while adopting the upload/preview primitives recorded in the Phase 5 reference note. |
| 30 | F24 | Done | Hide Dormant Settings Surface and Remove Empty Warning Chrome | `dev` | Hide the current non-functional Settings tab and remove compatibility/catalog chrome that has no meaningful user-facing action. |
| 31 | R13 | Done | A1111 Extras Postprocessing Pipeline Contract | `dev` | Introduce a dedicated A1111-style postprocessing contract for Extras, covering single-image, batch, restoration, upscaling, and metadata-preserving output behavior. |
| 32 | F25 | Done | A1111 Extras Surface and Backend Execution | `dev` | Add the visible Extras tab and execute it through a dedicated RookieUI postprocessing backend rather than the generation workflow translator, using the A1111/Forge-Neo runtime UI reference note for tab, dropzone, rail, preview, and theme-detail parity. |

## Phase 6 - Pre-Release UI and Runtime Correction Intake

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 33 | R16 | Done | RookieUI Queue Boundary and Session Isolation | `dev` | Clarify and harden the boundary between RookieUI submissions and host-native canvas workflows by enforcing RookieUI-scoped metadata and client/session isolation through the host queue lifecycle. |
| 34 | F28 | Done | Generate and Prompt Top-Edge Alignment Fix | `dev` | Resolve txt2img prompt-band misalignment so the Generate hero rail aligns to the prompt input top edge under real host rendering conditions. |
| 35 | F29 | Done | A1111/Forge-Neo Colored Tool Icon Parity | `dev` | Port compact color-coded icon semantics for tool rails so action icons match the A1111/Forge-Neo visual language instead of monochrome placeholders. |
| 36 | F30 | Done | Prompt Header Chrome Removal | `dev` | Remove the redundant `Prompt` and `Negative Prompt` captions above textarea controls while preserving counter and placeholder behavior. |
| 37 | R17 | Done | PNG Info Apply-Rail Layout Hardening | `dev` | Fix PNG Info action-button deformation by introducing a dedicated, host-safe action rail primitive for `Apply to` controls. |
| 38 | F31 | Done | UI Preset Taxonomy Refresh (SD/Flux/Qwen) | `dev` | Replace Noob/Illustrious as standalone presets with SDXL-derived handling and add explicit Flux/Qwen preset lanes aligned with reference `sd-webui-forge-neo` capability direction. |
| 39 | F32 | Done | A1111-Style Generation Progress and Live Preview | `dev` | Add real generation runtime feedback with queue/executing progress, in-flight preview, and final image presentation using ComfyUI host runtime channels. |
| 40 | F33 | Done | PNG Info Auto-Inspect on Image Load | `dev` | Trigger metadata inspection immediately after image import and remove the separate `Inspect Metadata` action from the PNG Info flow. |
| 41 | R18 | Done | Family-Aware Text Encoder Control Lock | `dev` | Keep SD1.5 and SDXL on an A1111-style single prompt authoring surface by hiding/ignoring separate text encoder selectors for those family presets. |

## Phase 7 - Secondary Model-Family Expansion Backlog

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 42 | F36 | Done | Forge-Neo Secondary Family Preset Intake | `dev` | Add a secondary-priority UI Preset expansion plan (reference: `reference/sd-webui-forge-neo/modules_forge/presets.py`) for `klein` (Flux.2), `lumina`, `zit` (Z-Image-Turbo), `wan`, and `anima`, while keeping SD-family parity work as primary scope. |

## Phase 8 - Release Candidate UX/Runtime Optimization Intake

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 43 | F37 | Done | Slim White Slider Track Refinement | `dev` | Apply a dedicated slider-only visual pass so horizontal range controls become thinner while forcing the unfilled track/base color to pure white under host rendering. |
| 44 | F38 | Done | PNG Info Prompt Auto-Extraction (Image-First) | `dev` | Align PNG Info ingest with `reference/comfyui-openclaw` image-first behavior and always extract positive/negative prompts automatically from imported image metadata when available. |
| 45 | F39 | Done | PNG Info Infotext Surface Retirement | `dev` | Disable legacy infotext-based PNG Info flow and remove the Infotext input UI so PNG Info remains image-driven only. |
| 46 | R21 | Done | Live Preview Host-Event Compatibility Hardening | `dev` | Harden runtime preview event adaptation so real-time preview renders reliably across host event payload variants while preserving RookieUI queue/session boundaries. |
| 47 | F40 | Done | Primary Tab Scale-Up and Workspace Divider Parity | `dev` | Increase top-level tab button size to at least current selector dimensions and add an explicit divider line between tab rail and workspace content to better match A1111 shell framing. |
| 48 | F41 | Done | A1111 Hires.fix Chrome Parity (Header/Border/Toggle) | `dev` | Rebuild Hires.fix presentation with A1111-style collapsible header chrome (triangle indicator, framed border, quick enable/disable checkbox) and remove non-A1111 helper copy. |
| 49 | R22 | Done | Img2Img Hires.fix Contract Restoration | `dev` | Restore missing img2img Hires.fix controls and backend contract path (request normalization + workflow translation) so img2img supports a true two-pass hires flow again. |
| 50 | F42 | Done | Extras Hires.fix Surface Recovery | `dev` | Restore a visible Hires.fix surface on Extras and bind it to the existing postprocessing upscale execution path so the control set is no longer missing from the Extras UI. |
| 51 | F43 | Done | Generate Hero Typography Softening | `dev` | Reduce `Generate` hero-label typography by two size steps and remove bold weight so button text matches target A1111 visual density. |
| 52 | F44 | Done | A1111 Native Emoji Tool-Icon Parity | `dev` | Replace current PrimeIcons-based mini action icons with A1111-style native emoji tool symbols (e.g. 📂 💾 🗃️ 🖼️ 🎨 📐 ✨ 🖌️) aligned with reference `stable-diffusion-webui` / `forge-neo` `ToolButton` semantics. |
| 53 | F45 | Done | v0.1.0 Release Metadata and OSS Packaging | `dev` | Finalize release-facing metadata by confirming the header `View on GitHub` destination, adding AGPL-3.0 licensing, hardening ignore rules for internal materials (`.planning/`, `reference/`, runtime/output artifacts), and replacing the minimal README with a detailed project/architecture/test guide. |

## Phase 9 - Post-Release Bug Intake Wave B (User-Reported)

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 54 | F46 | Done | Hires.fix Collapsible Triangle Icon Recovery | `dev` | Restore the A1111-style collapsed/expanded triangle indicator in the Hires.fix header so the state is visually explicit. |
| 55 | F47 | Done | Hires.fix Helper Copy Removal Completion | `dev` | Remove remaining `Second latent pass with bounded rookie-safe defaults.` text from Hires.fix surfaces. |
| 56 | F48 | Done | Hires.fix Header-Edge Enable Checkbox Behavior | `dev` | Move `Enable Hires` checkbox to the outer-left header edge and keep it interactive while the section body is collapsed. |
| 57 | R23 | Done | Img2Img Hires.fix Surface/Contract Completion | `dev` | Ensure Img2Img tab exposes Hires.fix controls and the backend translation path executes the configured second-pass hires flow. |
| 58 | F49 | Done | PNG Info Infotext Input Surface Removal | `dev` | Remove PNG Info `Infotext` label and textarea so PNG Info remains image-driven without legacy text-entry path. |
| 59 | R24 | Done | PNG Info Dual-Prompt Auto-Extraction Completion | `dev` | Complete automatic extraction/mapping of both positive and negative prompts from imported metadata, aligned with `reference/comfyui-openclaw` behavior. |
| 60 | R25 | Done | Sidebar Live Preview Completion Hardening | `dev` | Complete in-sidebar real-time preview refresh path and close remaining runtime gaps that block reliable preview updates during generation. |
| 61 | F50 | Done | Generate Button Width Unification | `dev` | Normalize Generate button width across tabs, using Extras tab width as the visual baseline. |
| 62 | R26 | Done | Sidebar Minimum Width Increase | `dev` | Raise the sidebar minimum width baseline so RookieUI has more consistent readable horizontal space under host layouts. |
| 63 | R27 | Done | Cross-Tab Parameter Persistence Lock | `dev` | Prevent model/parameter drift after tab switches by persisting and restoring per-pane state deterministically across tab navigation. |

## Phase 10 - Post-Release Bug Intake Wave C (User-Reported)

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 64 | F51 | Done | Generate Hero Height Reduction (85%) | `dev` | Reduce Generate button height to 85% of current visual height while preserving top-edge alignment and button row position. |
| 65 | F52 | Done | A1111 Seed Mode Emoji and Extras Toggle Expansion | `dev` | Add A1111-style seed controls (random/fixed emoji actions) and wire an explicit `Extra` feature-enable checkbox with backend-consumed state. |
| 66 | R28 | Done | Send-to-Img2Img Payload Transfer Integrity Fix | `dev` | Fix preview toolbar `Send to Img2Img` behavior so tab switch also transfers the generated image payload into Img2Img source state deterministically, including state-lock-safe apply ordering and data-URL fallback. |
| 67 | R29 | Done | Img2Img A1111 Feature-Surface Gap Closure | `dev` | Analyze reference implementation and complete missing Img2Img controls and semantics, including explicit upload area, resize mode behavior, inpaint masked-content handling, and soft inpainting-adjacent options with backend translation support. |
| 68 | R30 | Done | Live Preview Runtime Completion Rework | `dev` | Rework generation-time preview transport/update path to eliminate remaining no-preview runtime failures across host payload variants. |
| 69 | F53 | Done | PNG Info Layout Space Utilization and Reflow Recovery | `dev` | Resolve PNG Info right-column overgrowth/left-column underuse by redesigning layout flow to use available width effectively without excessive vertical stretching. |

## Phase 11 - Post-Release Bug Intake Wave D (User-Reported)

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 70 | F54 | Done | Primary/Functional Tab Scale-Up and Divider Reinforcement | `dev` | Increase primary tabs and functional subtab label/button size to approximately 140% of current baseline and add explicit separator lines between tab rails and parameter workspace. |
| 71 | F55 | Done | Extras Generate-Rail Emoji Action Recovery | `dev` | Restore the missing emoji action rail directly beneath Extras `Generate` so Queue/PNG Info quick actions remain visible and A1111 ToolButton semantics stay consistent. |
| 72 | R31 | Done | Img2Img Multi-Mode Surface Expansion (Sketch/Batch Lanes) | `dev` | Extend RookieUI Img2Img mode surface toward A1111 parity by adding sketch and batch-oriented lanes, mode-aware asset validation, and backend mode normalization/translation aliases for the new surfaces. |

## Phase 12 - Post-Release Bug Intake Wave E (User-Reported)

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 73 | R32 | Done | Flux/Qwen Text Encoder Selector Activation and Contract Lock | `dev` | Ensure Flux and Qwen-Image presets expose selectable Text Encoder UI in txt2img/img2img, keep backend normalization from clearing those profiles, and add profile-contract regression checks to prevent fallback to SDXL-locked behavior. |
| 74 | F56 | Done | Generate Hero Height Pass-2 Reduction (-15%) | `dev` | Further reduce hero `Generate` button height by 15% from the current F51 baseline while preserving top-edge alignment and cross-tab width/rail alignment contracts. |

## Phase 13 - PNG Info Visual Parity Follow-up (User-Reported)

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 75 | F57 | Done | PNG Info Metadata Card Layout Parity (OpenClaw Reference) | `dev` | Re-layout PNG Info post-inspection view into a summary rail, two-column metadata cards, and prompt/negative-prompt panels with copy actions, matching the visual density/style direction from `reference/comfyui-openclaw`. |
| 76 | R33 | Done | Clip Skip Input Freeze Remediation | `dev` | Remove profile-driven UI hard-disable that froze Clip Skip number/slider input, keep controls editable, and annotate unsupported profiles as execution-level ignore behavior. |

## Phase 14 - Regression Harness Hardening (User-Reported)

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 77 | R34 | Done | Clip Skip Regression-Capture Test Hardening | `dev` | Expand unit + Playwright harness coverage with SD1.5/SDXL/Flux/Qwen preset/profile transition matrices so Clip Skip lock regressions fail deterministically across txt2img/img2img and tab-restore seams. |

## Phase 15 - Prompt Usability Regression Follow-up (User-Reported)

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 78 | R35 | Done | Prompt Placeholder Visibility Recovery | `dev` | Normalize whitespace-only prompt textarea values to empty string so placeholder guidance remains visible, and add unit/E2E regressions for txt2img prompt hint continuity. |

## Phase 16 - Live Preview Stability Follow-up (User-Reported)

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 79 | R36 | Done | Sidebar Flicker Mitigation for Live Preview Frames | `dev` | Reduce sidebar-wide flicker during generation preview by avoiding per-frame preview DOM replacement, delaying blob URL revoke until post-frame swap, throttling frame updates, and isolating preview paint scope in CSS. |

## Phase 17 - PNG Info Layout Follow-up (User-Reported)

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 80 | R37 | Done | PNG Info Preview Top-Anchor Alignment | `dev` | Keep PNG Info preview section fixed at the top of the left input column so image preview does not drop below status/action blocks after metadata load. |

## Phase 18 - Audit-Driven Reliability and Maintainability Intake (Wave F)

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 81 | R38 | Done | Extras Boolean Numeric-Coercion Consistency Fix | `dev` | Align `rookieui/services/extras.py` boolean coercion with txt2img/img2img/pnginfo behavior by accepting numeric `0/1` and preventing cross-endpoint payload inconsistency for Extras requests. |
| 82 | R39 | Done | Img2Img Denoise Safe-Coercion Error-Surface Alignment | `dev` | Replace bare `float()` conversion for `denoise_strength` with field-aware coercion to keep validation semantics and error messages consistent with other normalized numeric fields. |
| 83 | R40 | Done | Shared Coercion Utility Extraction and Adoption | `dev` | Extract duplicated `_coerce_bool/_coerce_int/_coerce_float` helpers into a shared service module and migrate txt2img/img2img/pnginfo/extras/routes callers to a single contract to reduce drift regressions. |
| 84 | R41 | Done | Alias Constant Single-Source Consolidation | `dev` | Consolidate duplicated alias maps and locked-profile/upscale constant sets into one canonical module so img2img/pnginfo/txt2img normalization paths no longer diverge silently. |
| 85 | R42 | Done | Model Inventory Discovery TTL Cache | `dev` | Add short-lived TTL caching for host model inventory discovery to avoid repeated full folder scans on every normalization call while keeping inventory freshness acceptable for active sessions. |
| 86 | R43 | Done | Defensive Exception Logging Baseline | `dev` | Add structured debug/warn logging on non-fatal `except Exception` fallback paths so runtime degradation remains diagnosable without changing current fail-safe behavior. |
| 87 | R44 | Done | Runtime Asset Directory Cleanup Guard | `dev` | Add bounded cleanup policy for `.rookieui_runtime/input` and `.rookieui_runtime/output` (age/count-based) to prevent unbounded disk growth from img2img/pnginfo/extras usage. |
| 88 | R45 | Done | Workflow Node-ID Allocation Unification | `dev` | Replace mixed hardcoded/dynamic node-id assignment in workflow translation with a unified allocator seam to reduce collision risk as img2img/inpaint/hires graph surfaces continue expanding. |
| 89 | R46 | Done | Sidebar Shell Utility Extraction (Phase 1 Split) | `dev` | Extract pure helper utilities from `web/rookieui_sidebar_shell.js` into dedicated modules to reduce monolith pressure and prepare stable seams for later per-pane decomposition. |
| 90 | R47 | Done | Sidebar Per-Tab Module Split (Phase 2) | `dev` | Split sidebar tab-pane builders into per-tab modules (`txt2img/img2img/extras/pnginfo/queue`) while preserving current behavior and host-compatible lifecycle wiring. |
| 91 | R48 | Done | Frontend Debug Flag and Guarded Warning Telemetry | `dev` | Add a frontend debug flag and guarded warning output for runtime/network/state-fallback paths so silent UI failures become traceable without polluting normal production console output. |

## Phase 19 - Img2Img In-App Mask Canvas Expansion (User-Reported)

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 92 | R49 | Done | Img2Img Mask Canvas Contract and State Bridge | `dev` | Define a stable in-app mask drawing contract for Img2Img by formalizing source-image binding, mask serialization into existing `mask_data`, apply-order semantics, mode-transition guards, and backend translation compatibility without introducing a new generation API. |
| 93 | F58 | Done | Img2Img Embedded Mask Canvas Core Tools | `dev` | Implement an embedded Img2Img mask canvas with core tools (brush/eraser, size/opacity, undo/redo, clear/invert, zoom/pan/fit) and an explicit `Apply Mask` action so users can draw masks directly inside RookieUI. |
| 94 | F59 | Done | Img2Img Mask Canvas Advanced Tooling and Parity Pass | `dev` | Add advanced mask operations (selection/fill/transform-oriented tooling as feasible) and complete A1111-oriented inpaint usability parity checks against reference mask-editor behavior under sidebar constraints. |

Stage sequencing (implementation order):
- Stage 1: `R49` contract-first foundation and guard rails.
- Stage 2: `F58` core embedded drawing surface and apply workflow.
- Stage 3: `F59` advanced tooling and parity closure with regression hardening.

## Phase 20 - Img2Img Multi-Tier Generation Subtab Parity (User-Reported)

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 95 | R50 | Done | Img2Img Generation Subtab Router and State Contract | `dev` | Formalize an Img2Img Generation-level nested-subtab contract that synchronizes visible subtab selection, internal mode value, and existing mode alias/validation logic without backend API changes. |
| 96 | F60 | Done | Img2Img A1111-Style Multi-Tier Generation Subtabs | `dev` | Replace dropdown-first mode UX with A1111-style second-level subtabs (`img2img`, `Sketch`, `Inpaint`, `Inpaint sketch`, `Inpaint upload`, `Batch`) under Img2Img Generation while reusing current mode-specific asset surfaces and backend translation contracts. |

Stage sequencing (implementation order):
- Stage 1: `R50` nested-subtab router/state bridge.
- Stage 2: `F60` visible multi-tier mode rail and mode-pane parity.

## Phase 21 - Frontend Modularization Continuation (Wave G)

Execution policy: this phase is the current highest-priority implementation chain, subject to the blocker-hotfix exception rule in `Execution Strategy Update (2026-04-11)`.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 97 | R51 | Done | Frontend Module Boundary Contract (Shell vs Pane Ownership) | `dev` | Added canonical top-level tab contract metadata + shared adapter helper + shell runtime tab-definition validation to enforce explicit pane ownership boundaries before render. |
| 98 | R52 | Done | Shell State/Event Contract Extraction | `dev` | Added shared shell state/event contract module and migrated shell paths to contract-backed tab activation, pane-state lock registration, and cross-pane payload apply dispatch with deterministic fallback behavior. |
| 99 | F61 | Done | Img2Img Pane Module Extraction (Phase 3) | `dev` | Extracted Img2Img pane implementation into a dedicated module and wired an explicit shell helper-context bridge that preserves existing generation-subtab and mask-canvas behavior contracts. |
| 100 | F62 | Done | Txt2Img/Extras/PNGInfo/Queue Pane Module Extraction Completion | `dev` | Extracted txt2img, pnginfo, extras, and queue pane bodies into dedicated modules and switched shell to centralized explicit pane-context injection wrappers. |
| 101 | F63 | Done | Frontend Style Ownership Split and CSS Modular Cleanup | `dev` | Split monolithic sidebar styling into `rookieui_shell_foundation.css` and `rookieui_panes.css`, kept stable entry CSS import wiring, and aligned asset revision loading paths. |
| 102 | R53 | Done | Modularization Regression Harness Hardening | `dev` | Added dedicated unit + E2E seam-regression suites for activation lifecycle visibility, cross-pane apply routing, and pane state persistence after modular extraction. |

Stage sequencing (implementation order):
- Stage 1: `R51` boundary contract.
- Stage 2: `R52` state/event contract extraction.
- Stage 3: `F61` Img2Img module extraction.
- Stage 4: `F62` remaining pane extraction.
- Stage 5: `F63` CSS ownership split.
- Stage 6: `R53` regression hardening sweep.

## Phase 22 - Test Automation and SOP Reality Alignment (User-Requested)

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 103 | R54 | Done | Full-Test Script Chain, Pre-Push Hook, and Tests SOP Alignment | `dev` | Add repository-managed full-test scripts and `pre-push` hook wiring in OpenClaw-style structure, while rewriting `tests/` SOP documents to match RookieUI's actual scripts, ports, and executable verification surface. |

## Phase 23 - Generation Panel Layout Consolidation (User-Requested)

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 104 | F64 | Done | Integrate Hires.fix into Generation Parameter Panel | `dev` | Move the standalone `Hires. fix` block into the `Generation` parameter section directly below the `Clip Skip` and `Seed` row while preserving `Enable Hires` checkbox visibility, framed border chrome, expand/collapse behavior, and backend request semantics. |

## Phase 24 - A1111 Prompt Parity and Img2Img Completion (User-Requested)

Execution policy: this phase is the next high-priority continuity chain for A1111-compatible behavior parity on ComfyUI.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 105 | R55 | Done | A1111 Prompt Capability Matrix and Contract Freeze | `dev` | Freeze a reference-backed capability matrix for `AND`/`BREAK`/schedule/attention/extra-network semantics and lock the behavior contract for SD-family-first parity execution in ComfyUI. |
| 106 | F65 | Done | Prompt DSL Parser v2 (AND/BREAK/Schedule/Attention) | `dev` | Upgrade prompt preprocessing from warning-only detection to structured parsing of A1111 prompt semantics while preserving deterministic inline LoRA extraction and merge behavior. |
| 107 | F66 | Done | Conditioning Compiler for ComfyUI Graph Translation | `dev` | Compile parsed prompt semantics into ComfyUI-executable conditioning composition (weighted and timestep-ranged) and integrate it into txt2img/img2img workflow translation paths. |
| 108 | R56 | Done | Parity Guardrails, Warning Codes, and Rollback Switch | `dev` | Add bounded parser/compiler guardrails, stable warning-code diagnostics, and a reversible legacy fallback switch so parity rollout remains controllable under host/runtime variance. |
| 109 | R57 | Done | Img2Img Source/Mask Bridge Integrity Hardening | `dev` | Fix deterministic source-image and mask-state carryover across send-to-img2img, mode switches, and tab transitions, including mask upload event integrity. |
| 110 | F67 | Done | Img2Img Mode Surface Completion (A1111 Lanes) | `dev` | Complete functional mode surfaces for `img2img`, `sketch`, `inpaint`, `inpaint sketch`, `inpaint upload`, and `batch` with mode-aware validation and backend translation coverage. |
| 111 | F68 | Done | Multi-Tier Generation/Layout Parity Stabilization | `dev` | Stabilize A1111-style multi-tier generation and mode subtab layout/state behavior without introducing placeholder-only UI controls. |
| 112 | R58 | Done | Prompt/Img2Img Parity Regression Harness Expansion | `dev` | Add parser/compiler fixtures and end-to-end parity regressions (Reproduce -> Pin -> Sweep) so known prompt-semantic and img2img state-bridge regressions are caught deterministically. |

Stage sequencing (implementation order):
- Stage 1: `R55` capability matrix and parity contract freeze.
- Stage 2: `F65` prompt parser v2.
- Stage 3: `F66` conditioning compiler integration.
- Stage 4: `R56` guardrails and rollback controls.
- Stage 5: `R57` img2img source/mask bridge hardening.
- Stage 6: `F67` img2img mode completion.
- Stage 7: `F68` multi-tier layout/state parity stabilization.
- Stage 8: `R58` regression harness hardening sweep.

## Phase 25 - Img2Img Mask Canvas Source-Placeholder False-Positive Hotfix (User-Reported)

Execution policy: blocker-hotfix exception lane on `dev`; run before broader Phase 24 parity chain continues.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 113 | R59 | Done | Mask Canvas `No source image` Placeholder Visibility Contract Fix | `dev` | Fixed mask-canvas placeholder visibility contract so `No source image` stays hidden whenever source binding is valid (including txt2img -> send-to-img2img handoff), eliminating false-positive overlay after successful image transfer. |

Stage sequencing (implementation order):
- Stage 1: `R59` placeholder visibility + source-binding regression capture and fix.

## Phase 26 - ControlNet A1111-Native Parity Intake (User-Requested)

Execution policy: implement on `dev` with contract-first sequencing; keep ComfyUI host-native execution path and avoid direct Forge runtime patcher porting.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 114 | R60 | Done | ControlNet Contract and A1111 Compatibility Freeze | `dev` | Freeze RookieUI ControlNet unit contract, add dual payload compatibility (`controlnet_units` + `alwayson_scripts.controlnet`), and lock warning-code semantics for downgrade/error paths. |
| 115 | F69 | Done | ControlNet Core Graph Integration (Txt2Img/Img2Img/Inpaint) | `dev` | Integrate ControlNet generation flow into ComfyUI graph translation for `txt2img`, `img2img`, and `inpaint` using host-native loader/apply nodes with deterministic multi-unit ordering. |
| 116 | F70 | Done | A1111-Style ControlNet Unit Surface in Generation Panes | `dev` | Add A1111-style per-unit ControlNet UI groups with mode-safe source/mask binding and state-lock compatible persistence across subtab and tab transitions. |
| 117 | F71 | Done | ControlNet API Surface and Optional Preprocessor/Detect Pipeline | `dev` | Add canonical `/rookieui/controlnet/*` routes plus A1111-compatible `/controlnet/*` aliases and provide module/model/control-types/detect endpoints with optional dependency downgrade behavior. |
| 118 | R61 | Done | ControlNet Regression Harness, Rollback Switches, and Guardrail Hardening | `dev` | Expand targeted and end-to-end regression suites, add feature-flag rollback seams, and harden high-risk compatibility points with explicit diagnostics and guard comments. |

Stage sequencing (implementation order):
- Stage 1: `R60` contract and compatibility freeze.
- Stage 2: `F69` backend graph integration across generation modes.
- Stage 3: `F70` frontend unit-surface integration.
- Stage 4: `F71` API + preprocessor/detect capability surface.
- Stage 5: `R61` regression/rollback hardening sweep.

## Open Backlog Priority Board (Global, Open Items Only)

Re-audited on 2026-04-22 after the authoritative `reference/workflow_templates/imageEdit` inventory expanded and `reference/ComfyUI-EditUtils` was added to the workspace. The repo now tracks a new open image-edit delivery chain because the accepted code still ships only `qwen_image_edit` on a dedicated single-reference edit seam, while the authoritative references now show broader `img2img`-owned, no-mask, multi-reference edit topology. External host-asset gaps discovered by truthful live-smoke validation remain classified as host prerequisites rather than repo blockers.

| Index | Item | Priority | Status | Title | Branch | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 314 | F169 | P1 | Done | ImageEdit Smoke/Fixture/Test Matrix Foundation | `dev` | Added the dedicated `image-edit` live-smoke lane, manifest-driven dry-run validation, multi-reference frontend request assertions, and a green full SOP gate on `dev`. |
| 315 | R172 | P0 | Planned | Official ImageEdit Regression and Live-Host Acceptance Closure | `dev` | Close the chain only after targeted regressions, frontend interaction proof, truthful live-host catalog / execute evidence for the asset-ready subset, and a full repository SOP gate all pass on the accepted `dev` branch. |

## Phase 27 - Architecture Modernization Foundation (Review-Driven)

Execution policy: historical architecture-modernization intake on `dev`; phase 27 is now effectively closed after later accepted waves absorbed the remaining builder/coercion work without reopening public-contract risk.

| Order | Code | Priority | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| 119 | R62 | P0 | Completed (2026-04-17) | Workflow Translation Graph Builder Consolidation | `dev` | Closed by later accepted phase-59 work (`F110`/`R122`): `workflow_translation.py` is now a stable orchestration facade over extracted `workflow_builders/*`, which is the substantive graph-builder consolidation outcome originally targeted here. |
| 120 | R63 | P0 | Completed (2026-04-15) | Frontend Revision Token Single-Source Contract | `dev` | Replaced scattered per-import `?v=` ownership with canonical revision-token helpers plus revisioned loader modules for shipped sidebar/frontend assets. |
| 121 | R64 | P0 | Completed (2026-04-17) | Prompt/Metadata Coercion Utility Single-Source Completion | `dev` | Closed by later shared-coercion convergence: `prompt_dsl.py`, `pnginfo.py`, and multiple runtime/normalization services now rely on `rookieui/services/coercion.py`, leaving only thin local wrappers for field-specific defaults/error wording rather than duplicated coercion behavior. |

Stage sequencing (implementation order):
- Historical stage 1: `R62` graph-builder seam consolidation. Closed on 2026-04-17 via later accepted phase-59 extraction/hardening work.
- Stage 2: `R63` revision-token single-source migration.
- Historical stage 3: `R64` coercion utility convergence. Closed on 2026-04-17 after shared coercion-core adoption across prompt/pnginfo/runtime services.

## Phase 28 - Validation and Type-Safety Baseline (Review-Driven)

Execution policy: keep runtime contract stable on `dev`; introduce verification/type-safety improvements without bundler-default changes.

| Order | Code | Priority | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| 122 | R65 | P1 | Completed | Frontend TypeScript-First Migration Foundation | `dev` | Completed with repo-native `tsconfig` + `allowJs`/`checkJs` coverage, typed declaration seams for bootstrap/host surfaces, and test-script/SOP integration without bundler or runtime entrypoint swaps. |
| 123 | R66 | P1 | Completed | Real-Host Embedded E2E Lane and CI Parity Contract | `dev` | Completed with `scripts/run_host_embedded_e2e.py`, wrapper-script/SOP alignment, unit coverage for report/execute ordering, and green restarted-host full-pipeline evidence through the new contract runner. |

Stage sequencing (implementation order):
- Stage 1: `R65` TS-first foundation with no bundler swap.
- Stage 2: `R66` host-embedded E2E and CI parity lane.

## Phase 29 - Modularization Continuation and Decision-Gated Spikes (Review-Driven)

Execution policy: complete structural backlog first; keep high-risk modernization tracks (`R68`, `R69`) as gated feasibility items until explicit acceptance criteria are met.

| Order | Code | Priority | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| 124 | R67 | P1 | Completed (2026-04-15) | Sidebar Shell/Pane Service Extraction and Size-Budget Enforcement | `dev` | Extracted remaining high-churn sidebar shell services, reduced `rookieui_sidebar_shell.js` to 1577 lines, and added automated size-budget regression coverage while preserving existing tab/state contracts. |
| 125 | F72 | P1 | Completed | Model-Family Capability Registry Surface | `dev` | Completed with a canonical backend family-registry module plus read-only capability/bootstrap exposure; parity, presets, compatibility newer-family entries, and primary-model category routing now derive from the same contract. |
| 126 | R68 | P3 | Completed | Vite Build-Path Feasibility Spike and Compatibility Gate | `dev` | Completed with `spikes/vite/` plus `scripts/run_vite_spike.mjs`: build/preview checks proved the current frontend can be built under Vite, but preview runtime compatibility remains insufficient for a low-risk default-path switch because revision-token dynamic imports request raw asset names instead of Vite-emitted hashed chunks. |
| 127 | R69 | P3 | Completed | Vue Host-Adapter Feasibility Spike and Coexistence Contract | `dev` | Completed with `spikes/vue/` plus `scripts/run_vue_spike.mjs`: the Vue adapter consumed the same RookieUI bootstrap contract via injected loaders, coexisted with a custom extension slot lifecycle proof, and produced a `keep-exploring` decision without changing shipped entrypoints. |

Stage sequencing (implementation order):
- Stage 1: `R67` service extraction and budget enforcement.
- Stage 2: `F72` capability registry baseline.
- Stage 3 (`P3`, decision-gated): `R68` Vite feasibility spike completed with a `defer` decision; keep artifacts for future exploration, but preserve the shipped non-bundled frontend path as the production default.
- Stage 4 (`P3`, decision-gated): `R69` Vue host-adapter feasibility spike completed with a `keep-exploring` coexistence result; future migration remains optional and still requires explicit product-value justification before any production rewrite.

## Phase 30 - Diffusion-Model Loader Routing Hotfix and Regression Sweep (User-Requested)

Execution policy: blocker-hotfix lane on `dev`; run `Reproduce -> Pin -> Sweep` for each item and keep compatibility with existing SD-family checkpoint flow.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 128 | R70 | Done | Diffusion-Model Loader Root-Cause Freeze and Contract Definition | `dev` | Lock root cause for preset-switched `diffusion_models` selections incorrectly routed into `CheckpointLoaderSimple`, define loader-selection contract by primary model category, and capture fail-fast behavior for missing auxiliary loaders. |
| 129 | F73 | Done | Diffusion-Model Loader Graph Routing Implementation | `dev` | Implement workflow translation routing for `diffusion_models` via `UNETLoader` plus explicit text-encoder/VAE loader seams while preserving existing checkpoint-based SD-family graph behavior. |
| 130 | R71 | Done | Loader Surface Audit and Regression Harness Expansion | `dev` | Audit other loader seams (checkpoint, text encoder, VAE, ControlNet model path coupling) for category-routing drift and add targeted regression coverage for preset-switch and execution-path consistency. |

Stage sequencing (implementation order):
- Stage 1: `R70` root-cause freeze and loader contract capture.
- Stage 2: `F73` workflow loader routing implementation.
- Stage 3: `R71` cross-loader audit and regression hardening sweep.

## Phase 31 - Forge-Neo ControlNet Integrated Upgrade (User-Requested)

Execution policy: deliver in ordered `dev` chain and keep host-native ComfyUI execution; align UI/API/runtime semantics without direct Forge runtime patcher porting.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 131 | R72 | Done | Forge-Neo ControlNet Integrated Contract and Route Freeze | `dev` | Freeze target UX/contract for Forge-Neo style `ControlNet Integrated` tabs, lock compatibility boundaries, and publish implementation reference matrix across Forge/Forge-Neo/reForge. |
| 132 | F74 | Done | ControlNet Integrated Tabs UI Rebuild | `dev` | Rebuild RookieUI ControlNet unit surface to Forge-Neo style integrated tabs with grouped options, image/mask integrated controls, and state-safe serialization. |
| 133 | F75 | Done | Dynamic Control Type/Module/Model API Upgrade | `dev` | Replace static ControlNet module/type behavior with dynamic `module_list/control_types/model_list` routing and module-dispatch detect behavior. |
| 134 | F76 | Done | ControlNet Runtime Alignment and Mask Wiring Hardening | `dev` | Ensure selector changes from the integrated UI are reflected in workflow translation/runtime behavior, including mask path wiring and per-unit guidance semantics. |
| 135 | R73 | Done | ControlNet Integrated Regression and Rollback Hardening | `dev` | Expand regression harness and rollback switches for integrated ControlNet UI/API/runtime path, including preset-switch and send-to-img2img continuity lanes. |

Stage sequencing (implementation order):
- Stage 1: `R72` contract freeze and implementation matrix.
- Stage 2: `F74` integrated tabbed UI delivery.
- Stage 3: `F75` dynamic API surface delivery.
- Stage 4: `F76` runtime/translation alignment and mask wiring hardening.
- Stage 5: `R73` regression sweep and rollback-readiness closure.

## Phase 32 - ADetailer A1111 / Forge / Forge-Neo Parity Re-Baseline (User-Requested)

Execution policy: execute on `dev` after prompt-parity closure reaches an acceptable stopping point for current user work; use `reference/adetailer` as the canonical behavior source, treat Forge / Forge-Neo as host-UX compatibility targets, preserve host-native ComfyUI execution, and do not embed A1111 ScriptRunner runtime. Execution override for the 2026-04-14 stage-1/stage-2 delivery: implementation proceeded from the active `main` workspace because local branch switching remains blocked by the previously recorded Git index permission fault; keep `dev` as the target branch for later phase-32 stages.

| Order | Code | Priority | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- | --- |
| 136 | R74 | P1 | Completed (2026-04-14) | ADetailer A1111 / Forge Contract and UX Matrix Freeze | `dev` | Completed the English-source re-baseline against `reference/adetailer`, Forge / Forge-Neo, and healthy localhost `7860` capture; froze the canonical unit schema, prompt tokens, skip-img2img rules, ControlNet mode semantics, and component-level UI parity matrix for implementation. |
| 137 | F77 | P1 | Completed (2026-04-14) | ADetailer Request / Capability Foundation | `dev` | Shipped the canonical `adetailer` request block, normalized four-unit refinement-context contract, capability/bootstrap metadata, `/rookieui/adetailer/catalog` route, detector/model inventory reuse, and warning-backed foundation seams without importing A1111 script runtime. |
| 138 | F78 | P1 | Completed (2026-04-14) | ADetailer Integrated UI Parity Delivery | `dev` | Delivered the integrated multi-unit ADetailer surface on top of the ControlNet integrated panel primitives, with 4-unit tabs, A1111/Forge-style grouped layout, gated override controls, catalog-backed request serialization, and visual evidence captured against the healthy Forge-Neo/A1111 host reference. |
| 139 | F79 | P1 | Completed (2026-04-14) | ADetailer Detect-Mask-Refine Runtime Pipeline | `dev` | Implemented the host-native secondary refinement chain with detector-mask seam, mask preprocessing inputs, prompt token fallback, sampler overrides, seed ordering, and final-save binding to the last refinement decode. |
| 140 | F80 | P1 | Completed (2026-04-14) | ADetailer ControlNet None/Passthrough/Custom Coupling | `dev` | Reused RookieUI's integrated ControlNet runtime inside the ADetailer refinement context, snapshotting primary units for `passthrough` and isolating custom unit-local ControlNet state from the base generation path. |
| 141 | F81 | P1 | Completed (2026-04-14) | ADetailer Diagnostics and Availability Guidance Surface | `dev` | Added stable warning codes, normalized diagnostics, catalog/capability availability guidance, and fallback API metadata so degraded Adetailer behavior remains visible instead of silently disappearing. |
| 142 | R75 | P1 | Completed (2026-04-14) | ADetailer Regression, Visual Parity, and Rollback Hardening | `dev` | Added chain-level dry-run regression coverage for ADetailer runtime, passthrough ControlNet coupling, final-save rebinding, disabled-state rollback, diagnostics preservation, current harness visual evidence, and full Windows SOP gate closure. |

Stage sequencing (implementation order):
- Stage 1: `R74` contract freeze against ADetailer source semantics and healthy-host UI matrix. Completed 2026-04-14.
- Stage 2: `F77` request/capability foundation and refinement-context isolation. Completed 2026-04-14.
- Stage 3: `F78` UI parity delivery with mandatory multi-unit / grouped-option / interactive-gating fidelity. Completed 2026-04-14.
- Stage 4: `F79` detect-mask-refine runtime delivery with prompt/seed/mask-order semantics pinned. Completed 2026-04-14.
- Stage 5: `F80` ControlNet `none` / `passthrough` / `custom` coupling and context isolation. Completed 2026-04-14.
- Stage 6: `F81` diagnostics, warning-code, and availability-guidance surface. Completed 2026-04-14.
- Stage 7: `R75` regression sweep, healthy-host visual evidence, and rollback-readiness closure. Completed 2026-04-14.

## Phase 33 - Qwen/Wan Non-Lightning Defaults and Selector Hardening Hotfix (User-Requested)

Execution policy: blocker-hotfix lane on `dev`; enforce non-Lightning baseline defaults for RookieUI preset behavior and keep accelerated LoRA behavior explicit opt-in.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 143 | R76 | Done | Non-Lightning Preset Baseline and Selector Hardening for Qwen/Wan/ZiT | `dev` | Correct Qwen/Wan/ZiT preset/profile defaults to non-Lightning baselines, align ZiT sampler/scheduler with official template values, and harden default selector resolution to avoid auto-picking accelerated/distilled variants. |

Stage sequencing (implementation order):
- Stage 1: `R76` root-cause freeze using reference docs/templates.
- Stage 2: `R76` parity/preset/default-selector implementation.
- Stage 3: `R76` targeted regression + full-gate sweep and record closeout.

## Phase 34 - ControlNet Run-Preprocessor and Layout Parity Hotfix (User-Requested)

Execution policy: blocker-first UI parity hotfix lane on `dev`; restore Forge-style control-row semantics before broader canvas parity expansion.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 144 | R77 | Completed | ControlNet Run-Preprocessor Visibility and Layout Contract Freeze | `dev` | Contract freeze completed: `💥` run icon and `⤴` placeholder icon semantics pinned; `img2img` now hides run-preprocessor by default and auto-shows only when independent control image data is present; full SOP gate passed on 2026-04-12. |
| 145 | F82 | Completed | ControlNet Preprocessor/Model Row Geometry and Slider Alignment Implementation | `dev` | Completed on 2026-04-12: normalized preprocessor/model selector geometry, placed `💥` between both selectors, and aligned `Control Weight` lane width with `Timestep Range`; full SOP gate passed. |
| 146 | R78 | Completed | ControlNet Layout-Parity Regression and Rollback Hardening | `dev` | Completed on 2026-04-12: added targeted unit/E2E regression coverage for icon semantics, visibility toggling, selector-row order, and slider-lane alignment; full SOP gate passed. |

Stage sequencing (implementation order):
- Stage 1: `R77` revised visibility/layout contract freeze.
- Stage 2: `F82` UI geometry and interaction implementation.
- Stage 3: `R78` regression sweep and rollback-readiness closure.

## Phase 35 - Forge-Neo Canvas Parity (Img2Img Full Tabs + ControlNet) (User-Requested)

Execution policy: continue on `dev` after Phase 34 hotfix baseline; prioritize host-safe canvas-first interaction parity without changing backend execution ownership.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 147 | R79 | Completed | Forge-Neo Canvas Parity Contract and Integration Boundary Freeze | `dev` | Completed on 2026-04-12: introduced shared canvas contract helpers (`normalize source`, `source-presence`, `fullscreen request`) and adopted them across Img2Img + ControlNet seams to lock payload-sync and unit-isolation boundaries; full SOP gate passed. |
| 148 | F83 | Completed | Img2Img Full Tabs Canvas-First Interaction Parity Delivery | `dev` | Completed on 2026-04-12: delivered canvas-first source stage with in-canvas toolbar actions (`fullscreen/upload/remove/reset/undo/redo`), deterministic source-history rollback, and payload-safe synchronization for Img2Img full-tab workflows; full SOP gate passed. |
| 149 | F84 | Completed | ControlNet Integrated Canvas Surface Parity Delivery | `dev` | Completed on 2026-04-12: delivered integrated ControlNet canvas source toolbar with click-upload/drag-drop, per-unit rollback history, legacy source-row retirement, and run-preprocessor visibility compatibility; full SOP gate passed. |
| 150 | R80 | Completed | Canvas Parity Regression and Rollback Hardening | `dev` | Completed on 2026-04-12: expanded unit/E2E regression coverage for remove/undo/redo rollback behavior and run-preprocessor visibility transitions, plus final full-gate sweep for rollback-readiness closure. |

Stage sequencing (implementation order):
- Stage 1: `R79` contract freeze and behavior matrix lock.
- Stage 2: `F83` Img2Img full-tabs canvas parity delivery.
- Stage 3: `F84` ControlNet integrated canvas parity delivery.
- Stage 4: `R80` regression hardening and rollback-readiness closure.

## Phase 36 - Canvas Brush + ControlNet Preview/Preprocessor Hotfix Chain (User-Requested)

Execution policy: blocker-first parity hotfix lane on `dev`; prioritize interaction correctness and deterministic regression proof over visual-only adjustments.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 151 | R81 | Completed | Source Canvas Interaction Contract Freeze (Brush-First Upload-State Switching) | `dev` | Completed on 2026-04-12: introduced shared interaction-mode helpers (`upload`/`edit`), gated source-stage click/keyboard upload routing to upload mode only, and pinned Img2Img + ControlNet stage mode transitions via unit/E2E regression coverage; full SOP gate passed. |
| 152 | F85 | Completed | Shared Source Canvas Brush Controls and Edit Interaction Delivery | `dev` | Completed on 2026-04-12: shipped shared brush overlay controller (`size`, `opacity`, `softness`) for Img2Img source stage and ControlNet source stages, wired brush commits through existing snapshot-history paths, and validated with full SOP gate plus updated unit/E2E assertions. |
| 153 | R82 | Completed | Source Canvas Interaction Regression and Rollback Hardening | `dev` | Completed on 2026-04-12: hardened Img2Img source-stage regression seams with explicit brush-default, interaction-mode transition, and remove/undo/redo assertions in integration + E2E layers; full SOP gate passed. |
| 154 | R83 | Completed | ControlNet Preview/Preprocessor Contract Freeze (Allow Preview + Dual-Pane Semantics) | `dev` | Completed on 2026-04-12: run-preprocessor output now writes into per-unit generated-preview state, source fields remain immutable, and generated preview visibility is governed by `Allow Preview`; full SOP gate passed. |
| 155 | F86 | Completed | ControlNet Dual-Pane Preview + Run-Preprocessor UI/Layout Parity Delivery | `dev` | Completed on 2026-04-12: introduced dual-pane preview container with generated-preview lane visibility routing, added run-preprocessor hover tooltip discoverability, and preserved selector-row geometry + control-weight lane parity under full SOP gate validation. |
| 156 | R84 | Completed | ControlNet Preview/Preprocessor Regression and Acceptance Hardening | `dev` | Completed on 2026-04-13: expanded unit + E2E assertions for run-preprocessor tooltip discoverability, generated preview visibility routing, source-state non-mutation guarantees, and preview reset on source updates; full SOP gate passed. |

Stage sequencing (implementation order):
- Stage 1: `R81` source-canvas interaction contract freeze.
- Stage 2: `F85` source-canvas brush controls and edit interactions.
- Stage 3: `R82` source-canvas regression and rollback hardening.
- Stage 4: `R83` ControlNet preview/preprocessor contract freeze.
- Stage 5: `F86` dual-pane preview + run-preprocessor/layout parity delivery.
- Stage 6: `R84` regression hardening and full acceptance closure.

## Phase 37 - RookieUI Run-Preprocessor Non-Response and Canvas Fidelity Hotfix (User-Requested)

Execution policy: blocker-first hotfix lane on `dev`; keep Forge-Neo UI-reference analysis host separate from RookieUI runtime acceptance (`127.0.0.1:8188`) and close interaction/runtime regressions with deterministic evidence.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 157 | R85 | Completed (2026-04-13) | Runtime Scope and Failure Contract Freeze for Run-Preprocessor | `dev` | Freeze the diagnosis contract for \"Run Preprocessor has no response\" by pinning environment boundaries (reference host for UI analysis vs `8188` runtime truth), failure taxonomy, and explicit behavior expectations for `Allow Preview`-gated generated-lane rendering. |
| 158 | F87 | Completed (2026-04-13) | Source Canvas Fidelity and Brush Indicator Parity Delivery | `dev` | Fix source-preview stretch and pointer/brush geometry mismatch on integrated source stages, add brush-radius indicator parity (`crosshair + circle`), and enforce toolbar/slider bounds within preview-stage layout. |
| 159 | F88 | Completed (2026-04-13) | Run-Preprocessor Feedback and Preview-Gating UX Delivery | `dev` | Preserve source immutability and dual-pane generated-lane routing, and add explicit success/error/hidden-preview status feedback so run-preprocessor outcomes are always user-visible and actionable. |
| 160 | R86 | Completed (2026-04-13) | Run-Preprocessor Backend Route Verification and Regression Hardening | `dev` | Harden route-readiness verification and regression coverage for `/rookieui/controlnet/detect` behavior in live host runtime, including deterministic tests for `Allow Preview` visibility contracts and failure-path diagnostics. |

Stage sequencing (implementation order):
- Stage 1: `R85` runtime scope and failure-contract freeze (completed 2026-04-13).
- Stage 2: `F87` canvas fidelity and brush-indicator parity implementation (completed 2026-04-13).
- Stage 3: `F88` run-preprocessor feedback and preview-gating UX implementation (completed 2026-04-13).
- Stage 4: `R86` backend-route verification and regression hardening closure (completed 2026-04-13).

## Phase 38 - ControlNet Extension-First Detect Backend Alignment (User-Requested)

Execution policy: backend correctness-first on `dev`; align run-preprocessor behavior with reference `sd-webui-controlnet` / `forge-neo` detect semantics and make degraded runtime states explicit.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 161 | R87 | Completed (2026-04-13) | Detect Runtime Contract Freeze and Reference API Matrix | `dev` | Freeze extension-first detect contract against reference APIs, lock endpoint-probing strategy and warning-code semantics, and publish initial runtime boundary assumptions for extension host vs `8188` RookieUI runtime (later overridden by `R91`). |
| 162 | F89 | Completed (2026-04-13) | Extension-First Detect Backend Integration and Schema Alignment | `dev` | Implement multi-endpoint external detect probing with reference-compatible payload forwarding (`controlnet_masks`, `low_vram`), add deterministic `detect_backend` metadata, and gate internal fallback behind explicit env configuration. |
| 163 | R88 | Completed (2026-04-13) | Detect Route Diagnostics and Regression Hardening | `dev` | Expand backend/frontend regression coverage for extension request forwarding, extension-unavailable warning paths, explicit internal fallback behavior, and route-level diagnostics visibility in runtime logs. |

Stage sequencing (implementation order):
- Stage 1: `R87` detect contract freeze and reference matrix lock (completed 2026-04-13).
- Stage 2: `F89` extension-first detect implementation and schema alignment (completed 2026-04-13).
- Stage 3: `R88` regression hardening and acceptance closure (completed 2026-04-13).

## Phase 39 - ControlNet Fullscreen Zoom, Hover, and Header Parity Hotfix (User-Requested)

Execution policy: frontend parity-first on `dev`; preserve phase-38 extension-first detect backend path while fixing fullscreen image-fit, zoom interaction, hover chrome reveal, and header/caret parity.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 164 | R89 | Completed (2026-04-13) | Fullscreen and Header Parity Contract Freeze | `dev` | Freeze UI parity contract for fullscreen contain-fit behavior, fullscreen-only zoom interaction, preview hover-reveal behavior, and Controlnet header/caret alignment against Forge-Neo reference direction. |
| 165 | F90 | Completed (2026-04-13) | Fullscreen Zoom and Hover/Header UI Delivery | `dev` | Implement fullscreen auto-fit correction, fullscreen zoom slider control, hover-reveal presentation updates, `Controlnet` title rename, and mirrored/aligned collapse caret behavior. |
| 166 | R90 | Completed (2026-04-13) | Fullscreen Zoom/Parity Regression Hardening | `dev` | Add targeted regression coverage for fullscreen zoom interaction and CSS parity contracts, then close with full-gate validation per repository SOP. |

Stage sequencing (implementation order):
- Stage 1: `R89` parity contract freeze (completed 2026-04-13).
- Stage 2: `F90` fullscreen/hover/header parity implementation (completed 2026-04-13).
- Stage 3: `R90` regression hardening and full-gate acceptance closure (completed 2026-04-13).

## Phase 40 - ControlNet Detect Endpoint De-Hardcoding and 7860 Routing Removal (User-Requested)

Execution policy: backend safety-first on `dev`; override phase-38 hard-coded external host assumption and keep default runtime fully self-contained unless explicit external endpoint configuration is provided.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 167 | R91 | Completed (2026-04-13) | External Detect Endpoint Policy Override and Contract Freeze | `dev` | Freeze policy override: RookieUI must not hard-code `127.0.0.1:7860` or any implicit external detect host; external detect routing is explicit config-only behavior. |
| 168 | F91 | Completed (2026-04-13) | Hardcoded 7860 Removal and Explicit Endpoint Gating Delivery | `dev` | Remove hard-coded detect endpoint defaults from runtime code, require explicit endpoint env config for `a1111` provider calls, and preserve internal/self-contained default detect behavior. |
| 169 | R92 | Completed (2026-04-13) | Endpoint-Gating Regression and Full-Gate Hardening | `dev` | Update detect-route regression coverage for explicit endpoint-only forwarding and run full repository acceptance gate to verify no runtime/UI regressions. |

Stage sequencing (implementation order):
- Stage 1: `R91` endpoint-policy override contract freeze (completed 2026-04-13).
- Stage 2: `F91` hard-coded endpoint removal and runtime implementation (completed 2026-04-13).
- Stage 3: `R92` targeted regression + full-gate closure (completed 2026-04-13).

## Phase 41 - Forge-Native ControlNet Host Preprocessor Alignment (User-Requested)

Execution policy: backend correctness-first on `dev`; implement Forge-style in-process preprocessor routing inside Comfy host runtime and eliminate any remaining external detect assumptions from execution behavior.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 170 | R93 | Completed (2026-04-13) | Forge-Native Backend Contract Freeze and Root-Cause Capture | `dev` | Freeze backend contract for host-local preprocessor execution, capture pre-fix degraded detect behavior, and lock architecture direction to in-process host routing with no external detect dependency. |
| 171 | F92 | Completed (2026-04-13) | Host Preprocessor Dispatcher and Mask-Aware Detect Delivery | `dev` | Deliver host-runtime preprocessor dispatcher with explicit + dynamic candidate selection, AIO choice heuristics, mask forwarding for detect requests, and deterministic detect backend metadata/warnings. |
| 172 | R94 | Completed (2026-04-13) | Host-Dispatcher Regression Hardening and Full-Gate Closure | `dev` | Add regression coverage for AIO selector behavior and mask propagation, then close with full repository SOP gate validation. |

Stage sequencing (implementation order):
- Stage 1: `R93` backend contract freeze and root-cause capture (completed 2026-04-13).
- Stage 2: `F92` host dispatcher + detect implementation (completed 2026-04-13).
- Stage 3: `R94` targeted regression and full-gate closure (completed 2026-04-13).

## Phase 43 - ControlNet Depth Preprocessor Deterministic Dispatch and Black-Preview Hotfix (User-Requested)

Execution policy: backend/runtime correctness-first on `dev`; keep detect behavior self-contained in Comfy host, reduce probe side effects, and close depth-preview diagnostics gaps with full acceptance evidence.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 173 | R97 | Completed (2026-04-13) | Depth Preprocessor Root-Cause Freeze and Dispatch Policy Lock | `dev` | Freeze root-cause findings for depth detect mismatch/download churn, lock deterministic host-probe policy (`priority + probe limit`), and codify AIO avoidance for depth/normal fallback lane. |
| 174 | F94 | Completed (2026-04-13) | Deterministic Runtime Dispatch and Control-Model Diagnostics Delivery | `dev` | Deliver depth/normal deterministic host dispatch constraints, tighten dynamic preprocessor discovery, add `controlnet_model` echo diagnostics, and clarify frontend status text for generation-model vs preprocessor-annotator scope. |
| 175 | R98 | Completed (2026-04-13) | Black-Preview/Caret Regression Hardening and Full-Gate Closure | `dev` | Add targeted runtime/frontend regression coverage for range normalization, probe-limit behavior, and detect payload diagnostics; keep Hires/ControlNet caret parity and close with full repository SOP gate. |

Stage sequencing (implementation order):
- Stage 1: `R97` root-cause freeze and deterministic dispatch policy lock (completed 2026-04-13).
- Stage 2: `F94` runtime/frontend implementation delivery (completed 2026-04-13).
- Stage 3: `R98` targeted regression and full-gate closure (completed 2026-04-13).

## Phase 44 - ControlNet All-Module Deterministic Dispatch and AIO Gating Hardening (User-Requested)

Execution policy: backend/runtime stability-first on `dev`; apply deterministic host preprocessor dispatch across all supported modules, keep host-native execution preferred, and gate AIO probing behind explicit opt-in to avoid broad annotator side effects.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 176 | R99 | Completed (2026-04-13) | All-Module Dispatch Policy Freeze and Root-Cause Expansion | `dev` | Expand root-cause scope from depth-only to all ControlNet preprocessor modules, and freeze deterministic probe policy (`single host candidate attempt per module`) as the default runtime contract. |
| 177 | F95 | Completed (2026-04-13) | Global Deterministic Host Probe + Explicit AIO Opt-In Delivery | `dev` | Implement global single-attempt host probing with debug hotspot annotations and explicit `ROOKIEUI_CONTROLNET_AIO_PREPROCESSOR_ENABLED` opt-in gating for AIO execution paths. |
| 178 | R100 | Completed (2026-04-13) | Global Probe/AIO Regression Hardening and Full-Gate Closure | `dev` | Add targeted runtime regression tests for non-depth modules and AIO gate behavior, then close with full repository SOP acceptance gate (detect-secrets, pre-commit, backend unit suite, frontend unit+E2E). |

Stage sequencing (implementation order):
- Stage 1: `R99` root-cause expansion and policy freeze (completed 2026-04-13).
- Stage 2: `F95` runtime implementation for global deterministic probing and AIO opt-in gate (completed 2026-04-13).
- Stage 3: `R100` targeted regression + full-gate closure (completed 2026-04-13).

## Phase 45 - ControlNet Preprocessor Variant Filter and Dispatch Parity (User-Requested)

Execution policy: integrated UI and backend correctness-first on `dev`; align Control Type filtering with Forge-style preprocessor variants and ensure host dispatch follows selected annotator intent.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 179 | R101 | Completed (2026-04-13) | Variant Catalog Contract Freeze and Root-Cause Capture | `dev` | Freeze variant-capable preprocessor contract (`control_type -> module_list`), capture coarse-option mismatch root cause, and lock backward-compatible alias normalization scope. |
| 180 | F96 | Completed (2026-04-13) | Forge-Style Variant Filtering and Variant-Aware Host Dispatch Delivery | `dev` | Deliver variant-capable preprocessor catalog/filtering plus runtime dispatch preference binding so selected preprocessor variants bias host annotator choice and status visibility. |
| 181 | R102 | Completed (2026-04-13) | Variant Filtering/Dispatch Regression Hardening and Full-Gate Closure | `dev` | Add backend/frontend regression coverage for variant filtering and dispatch forwarding, then close with full repository SOP acceptance gate. |

Stage sequencing (implementation order):
- Stage 1: `R101` contract freeze and root-cause capture (completed 2026-04-13).
- Stage 2: `F96` variant filtering + runtime dispatch implementation (completed 2026-04-13).
- Stage 3: `R102` targeted regression + full-gate closure (completed 2026-04-13).

## Phase 46 - OpenPose Variant Dispatch Isolation Hotfix (User-Requested)

Execution policy: correctness-first hotfix on `dev`; preserve explicit preprocessor-variant intent during host dispatch, prevent silent DensePose fallback for OpenPose-family defaults, and audit whether other preprocessor families share the same variant-collapse defect.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 182 | R103 | Completed (2026-04-13) | OpenPose Variant Dispatch Isolation and Cross-Family Fallback Audit | `dev` | Fixed variant-stable host candidate resolution for explicit preprocessors, bounded generic OpenPose fallback to OpenPose/DW lanes, corrected final-result empty-output warning aggregation, and audited other variant families for the same drift pattern. |

Stage sequencing (implementation order):
- Stage 1: `R103` pre-fix reproduction, OpenPose-family dispatch isolation fix, cross-family audit, targeted regression coverage, and full-gate closure (completed 2026-04-13).

## Phase 47 - Forge-Style OpenPose-Family Exact Execution Hotfix (User-Requested)

Execution policy: correctness-first hotfix on the active branch while preserving repository SOP acceptance. Align RookieUI OpenPose-family execution with Forge / Forge-Neo semantics: exact selected preprocessor routing, variant-aware host parameter binding, truthful fallback preview behavior, and removal of dense-map-only visual-empty rejection from sparse pose outputs.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 183 | R104 | Completed (2026-04-13) | Forge-Style OpenPose-Family Exact Execution and Acceptance Closure | `main` (documented hotfix-chain override) | Reworked OpenPose-family runtime execution to preserve exact selected variant semantics, added explicit host detect-flag overrides, removed the generic visual-empty gate for sparse pose maps, replaced source-image fallback echo with truthful blank-map failure output, cleaned hotspot guard comments, and closed with targeted regression evidence plus full repository SOP gate. |

Stage sequencing (implementation order):
- Stage 1: `R104` pre-fix reproduction, Forge-style exact-execution retrofit, variant-aware host-parameter binding, truthful fallback preview correction, hotspot annotation cleanup, targeted regression coverage, and full-gate closure (completed 2026-04-13).

## Phase 48 - OpenPose-Family Schema-Aware Host Flag Coercion Hotfix (User-Requested)

Execution policy: correctness-first hotfix on the active branch while preserving repository SOP acceptance. Repair the remaining OpenPose-family host-wrapper boundary mismatch by coercing detect flags against real Comfy host node schemas so OpenPose/DWPose wrappers receive `"enable"` / `"disable"` values instead of Python booleans.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 184 | R105 | Completed (2026-04-13) | OpenPose-Family Schema-Aware Host Flag Coercion and Regression Closure | `main` (documented hotfix-chain override) | Added schema-aware host parameter coercion for combo inputs, fixed OpenPose/DWPose detect-flag type mismatches that produced successful-but-black previews, aligned runtime tests with the real local `comfyui_controlnet_aux` wrapper contract, and closed with targeted regression evidence plus full repository SOP gate. |

Stage sequencing (implementation order):
- Stage 1: `R105` wrapper-contract reproduction, schema-aware detect-flag coercion, local-contract regression updates, hotspot guard comment hardening, and full-gate closure (completed 2026-04-13).

## Phase 49 - A1111-Native Prompt Parity Rearchitecture Foundation (User-Requested)

Execution policy: parity-first on `dev`; treat shipped `R55/F65/F66` as an approximation baseline, then re-baseline SD-family prompt behavior against A1111 / Forge tokenizer-side semantics before further feature-surface expansion. Secondary/newer model families remain on ComfyUI-native prompt/conditioning engines unless a separate non-parity product decision is approved.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 185 | R106 | Completed (2026-04-13) | A1111-Native Prompt Parity Re-Baseline and Cutover Contract Freeze | `dev` | Captured the exact semantic gap between current prompt-DSL/compiler behavior and A1111-native prompt processing, froze the SD-family cutover contract, and redefined capability truthfulness for exact vs approximate prompt support. |
| 186 | F97 | Completed (2026-04-13) | Reference-Backed A1111 Prompt Engine Core Port | `dev` | Added a dedicated RookieUI A1111 prompt engine for attention parsing, `BREAK`, schedule timelines, and `AND` branch splitting, backed by reference-style regression coverage and isolated from the current runtime path pending phase-50 cutover. |

Stage sequencing (implementation order):
- Stage 1: `R106` root-cause capture, parity re-baseline, and cutover contract freeze.
- Stage 2: `F97` A1111 prompt engine core port with rollback-ready isolation from the current graph-only compiler path.

## Phase 50 - Comfy Host Prompt-Encoder Boundary and SD-Family Parity Nodes (User-Requested)

Execution policy: adapter-first on `dev`; move prompt semantics to the CLIP/text-encoder boundary through RookieUI-owned Comfy nodes instead of continuing graph-only emulation for SD-family routes. IMPORTANT: this phase preserves A1111-style SDXL single-prompt authoring and does not introduce independent user-facing `text_g` / `text_l` prompt lanes as a parity objective.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 187 | R107 | Completed (2026-04-14) | Comfy Host Text-Encoder Boundary Contract Freeze | `dev` | Froze the single-encoder adapter seam for CLIP/tokenizer access, conditioning payload metadata, rollback gating, and hires-pass ownership so SD15 parity-node delivery remains deterministic and host-compatible. |
| 188 | F98 | Completed (2026-04-14) | SD1.x / SD2.x A1111 Parity Text-Encoder Node Delivery | `dev` | Delivered RookieUI-owned parity encoder node(s) for single-encoder SD-family models, rewired SD15 txt2img/img2img/inpaint translation to use them by default, preserved the legacy rollback path, and added SDXL hires non-regression coverage while `F99` remains pending. |
| 189 | F99 | Completed (2026-04-14) | SDXL A1111 Parity Dual-Encoder Node Delivery | `dev` | Delivered RookieUI-owned SDXL parity encoder node(s), rewired SDXL txt2img/img2img/inpaint translation to use them by default when dual-channel host support is present, preserved A1111-style single-prompt authoring at the surface layer, preserved legacy rollback, and hardened single-encoder fallback detection. |

Stage sequencing (implementation order):
- Stage 1: `R107` encoder-boundary contract freeze (completed 2026-04-14).
- Stage 2: `F98` SD1.x / SD2.x parity-node delivery (completed 2026-04-14).
- Stage 3: `F99` SDXL dual-encoder parity-node delivery (completed 2026-04-14).

## Phase 51 - Workflow Translator Cutover and Prompt Capability Truthfulness (User-Requested)

Execution policy: correctness-first on `dev`; cut SD-family workflow translation over to parity nodes, demote the current graph-only semantic compiler to an explicit fallback path, and realign all capability messaging with actual behavior. Secondary/newer families are intentionally excluded from exact A1111 prompt-parity claims and remain on ComfyUI-native backend execution.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 190 | R108 | Completed (2026-04-16) | Workflow Translator Cutover and Legacy-Path Demotion | `dev` | Completed the shipped SD-family cutover to RookieUI-owned prompt encode nodes across txt2img/img2img and ADetailer-local conditioning, moved A1111 attention handling onto the encoder seam, preserved rollback controls, and added regression coverage for stock-encoder avoidance plus de-emphasis forwarding. |
| 191 | F100 | Completed (2026-04-14) | Prompt Capability / UI / API Truthfulness Realignment | `main` (documented user-requested override) | Realigned backend/frontend prompt capability messaging to the shipped SD-family parity-node default, added explicit fallback/unsupported warning metadata, corrected legacy warning copy, and closed with full SOP validation. |

Stage sequencing (implementation order):
- Stage 1: `R108` translator cutover and legacy-path demotion (completed 2026-04-16).
- Stage 2: `F100` capability / UI / API truthfulness realignment (completed 2026-04-14).

## Phase 52 - Golden Prompt Parity Regression and Host-Embedded Closure (User-Requested)

Execution policy: evidence-first on `dev`; no A1111-native prompt parity claim is accepted until golden parser/conditioning fixtures and host-embedded execution evidence prove the SD-family path end to end. This closure gate applies to SD-family parity only, not to secondary/newer families that intentionally follow ComfyUI-native engines.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 192 | R109 | Completed (2026-04-16) | Golden Prompt Parity Regression Harness and Host-Embedded Validation Closure | `dev` | Closed after the standalone golden fixture corpus (`F104`), prompt-parity live smoke lane (`R114`), green in-sync deployed-host dry-run plus execute evidence (`R115`), and full repository SOP acceptance were all delivered on the accepted SD-family parity path. |

Stage sequencing (implementation order):
- Stage 1: `R109` golden-fixture coverage, host-embedded parity evidence, and full-gate closure. Completed 2026-04-16.

## Phase 55 - SD-Family Prompt Parity Closure Expansion (User-Requested Backlog Correction)

Execution policy: complete on `dev` in closure order, not broad backlog order. This phase decomposes the oversized `R109` closure target into acceptance-sized items and separates prompt-parity smoke-lane delivery from external deployed-host evidence when the active host is out of sync with the workspace. Product scope remains SD-family-only A1111-style prompt parity.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 193 | R113 | Completed (2026-04-16) | Embeddings / Textual Inversion Prompt Contract Freeze | `dev` | Completed with prompt contract freeze, explicit payload shape, staged capability truthfulness, and full-gate acceptance ahead of runtime delivery. |
| 194 | F103 | Completed (2026-04-16) | Embeddings / Textual Inversion Prompt-Path Delivery | `dev` | Completed with inventory-aware prompt normalization, canonical host embedding tokens, missing-reference fallback diagnostics, and full-gate SD-family regression coverage. |
| 195 | F104 | Completed (2026-04-16) | Golden Prompt Parity Fixture Matrix and Standalone Harness | `dev` | Delivered a shared SD-family golden fixture corpus plus standalone harness coverage across parser semantics and txt2img/img2img translation topology, including attention, `BREAK`, scheduling, `AND` / weighted multi-cond, embeddings/textual inversion, and mixed negative-prompt symmetry. |
| 196 | R114 | Completed (2026-04-16) | SD-Family Prompt-Parity Live Host Smoke Lane and Deployment-Drift Detection | `dev` | Delivered prompt-parity-specific live-host smoke coverage for SD-family dry-run assertions, healthy host checkpoint override selection, report-only evidence capture, and explicit stale/out-of-sync deployment classification. |
| 197 | R115 | Completed (2026-04-16) | In-Sync Deployed-Host Prompt-Parity Evidence Capture | `dev` | Re-ran the prompt-parity smoke lane against the restarted, synchronized ComfyUI deployment, fixed the remaining runner-side live-fixture rewrite bug, and captured green dry-run plus execute host evidence required for final parity closure. |

Stage sequencing (implementation order):
- Stage 1: `R113` embeddings/textual inversion prompt contract freeze.
- Stage 2: `F103` embeddings/textual inversion prompt-path delivery.
- Stage 3: `F104` golden fixture matrix and standalone harness. Completed 2026-04-16.
- Stage 4: `R114` SD-family prompt-parity live host smoke lane and deployment-drift detection. Completed 2026-04-16.
- Stage 5: `R115` in-sync deployed-host prompt-parity evidence capture. Completed 2026-04-16.

## Phase 56 - SD-Family Prompt Parity Maximal Continuation (Same-Scope Hardening)

Execution policy: complete on `dev` in strict sequence. This phase stays inside the existing SD-family-only A1111-style prompt parity scope and targets the highest remaining reference-backed gaps after `R109` closure: explicit alternate scheduling parity, tokenizer-side chunk hardening, and refreshed reference/live-host evidence.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 198 | R116 | Completed (2026-04-16) | Reference-Backed SD-Family Prompt Continuation Contract Freeze | `dev` | Completed with a reference-backed phase-56 intake that freezes the next same-scope parity targets after `R109`: alternate prompt scheduling, tokenizer-side chunk hardening, and reference/live-host closure. |
| 199 | F105 | Completed (2026-04-16) | Alternate Prompt Scheduling Delivery | `dev` | Completed with recursive dynamic-group prompt expansion for A1111-style `[a|b]` alternates, truthful capability-surface updates, step-aware txt2img/img2img normalization, golden-fixture delivery, and prompt-parity smoke-case coverage. |
| 200 | F106 | Completed (2026-04-16) | Token-Chunk / Comma-Backtrack / Textual-Inversion Boundary Hardening | `dev` | Completed with a RookieUI-owned SD-family token rebatching seam that uses host word-id payloads when available, applies recent-comma backtrack behavior, preserves grouped textual-inversion boundaries, and falls back safely on hosts without word-id support. |
| 201 | R117 | Completed (2026-04-16) | Reference Differential and Live-Host Maximal Parity Closure | `dev` | Completed with a reference-backed token-chunk differential harness, a long comma-heavy SD15 golden/live case, host-embedding execute coverage, and refreshed live-host dry-run plus execute evidence after fixing a smoke-runner step-count false negative for compiled parity cases. |

Stage sequencing (implementation order):
- Stage 1: `R116` reference-backed continuation contract freeze. Completed 2026-04-16.
- Stage 2: `F105` alternate prompt scheduling delivery. Completed 2026-04-16.
- Stage 3: `F106` token-chunk / comma-backtrack / textual-inversion boundary hardening. Completed 2026-04-16.
- Stage 4: `R117` reference differential and live-host maximal parity closure. Completed 2026-04-16.

## Phase 57 - README Synchronization for Shipped Prompt-Parity Facts

Execution policy: complete on `dev` after phase 56 closure. This phase is documentation-only and exists to keep the public README truthful about the currently shipped SD-family prompt-parity surface, installation behavior, and live-host validation evidence.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 202 | R118 | Completed (2026-04-16) | README Prompt-Parity and Installation Truthfulness Sync | `dev` | Completed with a refreshed README last-update block, explicit SD-family prompt-parity coverage notes, ToC synchronization, and public-facing wording aligned to the shipped phase 55-56 behavior and validation evidence. |

Stage sequencing (implementation order):
- Stage 1: `R118` README prompt-parity and installation truthfulness sync. Completed 2026-04-16.

## Phase 58 - Live-Host Validation Expansion for Shipped Auxiliary Pipelines

Execution policy: complete on `dev` in strict sequence after phase 57. This phase extends host-embedded validation beyond SD-family prompt parity to the shipped `ControlNet`, `ADetailer`, `Extras`, `PNG Info`, and `Queue` surfaces. Validation remains truthfulness-first: fallback-capable runtime seams must be validated as shipped behavior contracts, but must not be upgraded into exact-native-equivalence claims.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 203 | R119 | Completed | Auxiliary Surface Live-Host Contract Freeze and Shared Smoke Framework Baseline | `dev` | Freeze route-level contract/version anchors for live-host validation, extend the smoke-runner framework beyond prompt parity, and establish shared host-context / execute / post-state validation seams for auxiliary surfaces. |
| 204 | F107 | Completed | ControlNet Live-Host Validation Lane | `dev` | Deliver a dedicated live-host lane for integrated ControlNet normalization, detect-route behavior, workflow dry-run assertions, and execute-level evidence on the shipped main generation pipeline. |
| 205 | F108 | Completed | ADetailer Live-Host Validation Lane | `dev` | Deliver a dedicated live-host lane for ADetailer catalog/runtime validation, dry-run refinement-chain topology checks, fallback-safe detector execution evidence, and ADetailer-local ControlNet coupling assertions. |
| 206 | F109 | Completed | Extras / PNG Info / Queue Live-Host Validation Lane | `dev` | Deliver live-host validation for synchronous Extras execution, PNG Info parse/inspect/apply-back behavior, and explicit queue/history route assertions tied to real RookieUI-origin jobs. |
| 207 | R120 | Completed | Full-Pipeline Shared Queue/Post-State Closure | `dev` | Close the chain with shared queue/post-state assertions, reusable-output validation, and an aggregate full-pipeline live-host mode spanning the newly added auxiliary validation lanes. |

Stage sequencing (implementation order):
- Stage 1: `R119` auxiliary-surface live-host contract freeze and shared smoke framework baseline.
- Stage 2: `F107` ControlNet live-host validation lane.
- Stage 3: `F108` ADetailer live-host validation lane.
- Stage 4: `F109` Extras / PNG Info / Queue live-host validation lane.
- Stage 5: `R120` full-pipeline shared queue/post-state closure.

## Phase 59 - Extensibility Refactor for High-Churn Integrated Packs

Execution policy: complete on `dev` in strict sequence after phase 58. This phase is maintainability-first and optimizes RookieUI for sustained expansion of high-churn integrated surfaces (`ControlNet`, `ADetailer`, and adjacent host-native auxiliary flows) by reducing service monoliths, extracting graph feature builders, and consolidating integrated-feature bootstrap ownership without changing shipped public route/payload contracts.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 208 | R121 | Completed | Extensibility Refactor Contract and Module-Boundary Freeze | `dev` | Freeze the target ownership map for workflow builders, ControlNet/ADetailer vertical services, and integrated-feature registry seams while explicitly preserving current public route/payload behavior. |
| 209 | F110 | Completed | Workflow Translation Feature-Builder Extraction | `dev` | Reduce `workflow_translation.py` to an orchestration façade by extracting deterministic shared graph builders for prompt conditioning, ControlNet application, ADetailer refinement, and output/persistence lanes. |
| 210 | F111 | Completed | ControlNet Vertical Service Split | `dev` | Split the oversized ControlNet service into focused catalog, normalization, detect/runtime-adapter, and warning/contract modules behind a stable façade. |
| 211 | F112 | Completed | ADetailer Vertical Service Split | `dev` | Split the oversized ADetailer service into focused catalog/availability, normalization, refinement-planning, and warning/diagnostic modules behind a stable façade. |
| 212 | F113 | Completed | Integrated-Feature Registry and Bootstrap Consolidation | `dev` | Introduce a lightweight internal registry/manifest for integrated feature contracts, bootstrap fetch ownership, and validation-lane linkage so future feature growth does not require more one-off bootstrap wiring. |
| 213 | R122 | Completed | Refactor Regression, Import-Cycle, and Live-Host Hardening | `dev` | Close the chain with topology regression proof, import-cycle/size-budget checks, and mandatory live-host validation for the touched integrated surfaces. |

Stage sequencing (implementation order):
- Stage 1: `R121` extensibility refactor contract and module-boundary freeze.
- Stage 2: `F110` workflow translation feature-builder extraction.
- Stage 3: `F111` ControlNet vertical service split.
- Stage 4: `F112` ADetailer vertical service split.
- Stage 5: `F113` integrated-feature registry and bootstrap consolidation.
- Stage 6: `R122` refactor regression, import-cycle, and live-host hardening.

Phase 59 progress notes:
- `R121` completed on 2026-04-17 with a tracked extensibility-boundary manifest, stable-facade guard annotations, roadmap/plan synchronization, and full-gate acceptance on `dev`.
- `F110` completed on 2026-04-17 with `workflow_builders/*` extraction, a reduced `workflow_translation.py` façade, targeted builder import smoke coverage, and full-gate acceptance on `dev`.
- `F111` completed on 2026-04-17 with `controlnet_*` vertical service modules, a stable façade preserved for route/test patch seams, targeted ControlNet regression coverage, and live-host `controlnet` dry-run + execute validation on `dev`.
- `F112` completed on 2026-04-17 with `adetailer_*` vertical service modules, a stable façade preserved for route/capability/normalization seams, targeted ADetailer regression coverage, and live-host `adetailer` dry-run + execute validation on `dev`.
- `F113` completed on 2026-04-17 with backend/frontend integrated-feature registries, registry-driven sidebar bootstrap loading, revision-token synchronization, and live-host `full-pipeline` validation on `dev`.
- `R122` completed on 2026-04-17 with manifest-backed target-module path proof, explicit phase-59 facade size-budget/import-cycle guardrails, a final `workflow_translation.py` facade compaction back to budget, and green live-host `full-pipeline` dry-run + execute evidence on `dev`.

## Phase 60 - Prompt Workbench Migration from `sd-webui-prompt-all-in-one`

Execution policy: complete on `dev` in strict sequence after phase 59. This phase plans the full migration route for the feature family in `reference/sd-webui-prompt-all-in-one/`, but adapts it to RookieUI-native architecture: registry-aware prompt-workbench services, versioned persistent state, host-inventory-backed prompt catalogs, integrated editor surfaces inside the sidebar prompt bands, and truthfulness-first external provider handling.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 214 | R123 | Completed | Prompt Workbench Migration Contract and Reference Freeze | `dev` | Completed with a tracked prompt-workbench contract module that freezes route-family ownership, per-surface namespaces, schema versioning, bootstrap defaults, and provider-secret masking rules before backend route/state work lands. |
| 215 | F114 | Completed | Prompt Workbench Persistent State and Route Substrate | `dev` | Completed with a RookieUI-owned versioned prompt-workbench state store plus coherent `/rookieui/prompt-tools/*` routes for config, state, history, favorites, and blacklist, along with lightweight frontend bootstrap preload hooks. |
| 216 | F115 | Completed | Translation Provider and Secret-Handling Surface | `dev` | Delivered a truthful translation-provider catalog, masked provider configuration state, single/batch translation execution routes, and provider availability diagnostics suitable for future prompt-workbench translation and AI-assist features. |
| 217 | F116 | Completed | Prompt Catalog, Token Analysis, Group Tags, and Extra-Network Intake | `dev` | Delivered grouped quick-add tags, prompt-library catalogs, token-analysis services, and extra-network prompt metadata through RookieUI-owned catalog/analyzer services aligned with the shipped SD-family prompt runtime and host inventory seams. |
| 218 | F117 | Completed | Frontend Prompt Workbench Shell Foundation | `dev` | Delivered a registry-aware prompt-workbench shell seam mounted into txt2img/img2img prompt bands, backed by lazy prompt-tools request helpers, debounced namespace-state persistence, dedicated frontend module coverage, and no pane-monolith regression. |
| 219 | F118 | Completed | Editor Actions, History/Favorites, Blacklist, and Formatting Delivery | `dev` | Delivered tokenized prompt editing, enable/disable/delete/reorder flows, history/favorites persistence and apply actions, blacklist-aware prompt cleanup, formatting-rule controls, and dedicated frontend/API regression coverage without regrowing the phase-59 bootstrap monolith. |
| 220 | F119 | Completed | Translation-Aware Editing, Grouped Quick-Add, and Extra-Network Quick-Insert Delivery | `dev` | Delivered translation-driven editing flows, grouped quick-add libraries, prompt-library append/replace browsing, and extra-network quick-insert behavior inside the RookieUI prompt workbench without re-expanding the phase-59 bootstrap monolith. |
| 221 | F120 | Completed | AI Prompt Assist, Language, and Theme/Style Delivery | `dev` | Completed with shipped OpenAI-compatible AI-assist execution, prompt-workbench language/theme-style controls, provider configuration delivery, and integrated prompt apply-back inside the workbench shell. |
| 222 | R124 | Completed | Prompt Workbench Regression, Live-Host, and Truthfulness Closure | `dev` | Completed with targeted regression coverage, prompt-workbench live-host route/state/apply validation, truthful translate and AI-assist execute paths, stale-host route detection, and final closure of the migrated prompt-workbench feature surface. |

Stage sequencing (implementation order):
- Stage 1: `R123` prompt-workbench migration contract and reference freeze.
- Stage 2: `F114` prompt-workbench persistent state and route substrate.
- Stage 3: `F115` translation-provider and secret-handling surface.
- Stage 4: `F116` prompt catalog / token-analysis / group-tag / extra-network intake.
- Stage 5: `F117` frontend prompt-workbench shell foundation. Completed 2026-04-17.
- Stage 6: `F118` editor actions / history-favorites / blacklist / formatting delivery. Completed 2026-04-17.
- Stage 7: `F119` translation-aware editing / grouped quick-add / extra-network quick-insert delivery. Completed 2026-04-17.
- Stage 8: `F120` AI prompt assist / language / theme-style delivery.
- Stage 9: `R124` regression, live-host, and truthfulness closure.

## Phase 61 - A1111 Native XYZ Plot Migration

Execution policy: complete on `dev` after phase 59 and in coordination with the accepted live-host queue baseline. This phase migrates A1111 native `X/Y/Z plot` behavior into RookieUI as a host-native feature: explicit axis registry, value parser, queue-backed sweep sessions, RookieUI-owned grid asset assembly, and an integrated frontend surface with truthful axis support.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 223 | R125 | Completed | XYZ Plot Migration Contract and Reference Freeze | `dev` | Completed with a tracked `rookieui/contracts/xyz_plot.py` module that freezes supported-axis truthfulness, route/session ownership, grid-output delivery, and the RookieUI-native adaptation rules replacing A1111 script-hijack behavior. |
| 224 | F121 | Completed | Axis Registry, Value Parser, and Sweep-Estimate Substrate | `dev` | Completed with extracted XYZ axis/value/estimate modules, contract-versioned `GET /rookieui/xyz-plot/axes` plus `POST /rookieui/xyz-plot/estimate`, bootstrap registry exposure, and full-gate proof across route/registry/parser/estimate regressions. |
| 225 | F122 | Completed | Queue-Backed Sweep Session Runner and Cell Execution Orchestrator | `dev` | Completed with a RookieUI-owned multi-cell session runner that mutates base requests per axis cell, reuses current normalize/translate/submit seams, records explicit session/cell metadata, reconstructs progress through queue/history, and truthfully rejects non-runner-ready axes from the queue-backed execution path. |
| 226 | F123 | Completed | Grid Assembly, Metadata, and XYZ Asset Delivery | `dev` | Completed with RookieUI-owned sub-grid/main-grid assembly, annotations/margins, XYZ metadata payloads, runtime asset persistence, and cached session-result delivery so the feature returns coherent grid outputs rather than only raw per-cell jobs. |
| 227 | F124 | Completed | Frontend XYZ Plot Surface and Session UX | `dev` | Completed with an extracted `rookieui_xyz_plot_shell` ownership seam, integrated txt2img/img2img mounting below ControlNet and ADetailer, XYZ API/bootstrap bindings, and frontend regressions that pin estimate/run/session/result flow plus bottom-placement behavior. |
| 228 | R126 | Completed | XYZ Plot Regression, Live-Host, and Truthfulness Closure | `dev` | Completed with dedicated XYZ route/session/live-host validation, aggregate `full-pipeline` coverage, and closure evidence that the shipped axis/support matrix and grid delivery behavior match the current workspace truth. |

Stage sequencing (implementation order):
- Stage 1: `R125` XYZ Plot migration contract and reference freeze.
- Stage 2: `F121` axis registry, value parser, and sweep-estimate substrate.
- Stage 3: `F122` queue-backed sweep session runner and cell execution orchestrator.
- Stage 4: `F123` grid assembly, metadata, and XYZ asset delivery.
- Stage 5: `F124` frontend XYZ Plot surface and session UX.
- Stage 6: `R126` regression, live-host, and truthfulness closure.

## Phase 62 - Runtime Robustness Hardening from Rebased Audit Findings

Execution policy: complete on `dev` after phases 58-61. This phase intentionally excludes already absorbed architecture findings and focuses only on retained open runtime risks with concrete user-facing stability value: ADetailer cache/cascade guardrails, ControlNet shared-state hardening, prompt nesting limits, and ControlNet tensor normalization truthfulness.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 229 | R127 | Completed | Runtime Robustness Hardening Contract Freeze | `dev` | Freeze the retained runtime hardening targets from the rebased robustness audit, pin pre-fix reproduction expectations where practical, and lock the proof model before implementation changes land. |
| 230 | F125 | Completed | ADetailer Runtime Cache and Cascade Guardrails | `dev` | Introduce a bounded Ultralytics model-cache policy plus synchronized OpenCV face-cascade initialization for the shipped native ADetailer runtime, preserving current detector availability semantics while reducing long-running host risk. |
| 231 | F126 | Completed | ControlNet PromptServer Shim Concurrency Hardening | `dev` | Completed with a serialized refcounted shim lifecycle for `PromptServer.instance.last_prompt_id`, preventing overlapping detect requests from tearing down shared compatibility state. |
| 232 | F127 | Completed | Prompt DSL Nesting Guard and Descriptive Failure Path | `dev` | Completed with an explicit maximum nesting depth on balanced attention-group rewriting so pathological prompts now fail descriptively instead of recursing into interpreter limits. |
| 233 | F128 | Completed | ControlNet Tensor Range Normalization Hardening | `dev` | Completed with stricter integer-like `0..255` detection plus min/max fallback for fractional low-range tensors to preserve truthful ControlNet preview normalization. |
| 234 | R128 | Completed | Runtime Robustness Regression and Closure | `dev` | Completed with targeted seam regressions, repeated full SOP-gate validation, and live-host `controlnet`, `adetailer`, and `full-pipeline` confirmation on the final runtime-hardening baseline. |

Stage sequencing (implementation order):
- Stage 1: `R127` runtime hardening contract freeze.
- Stage 2: `F125` ADetailer runtime cache and cascade guardrails.
- Stage 3: `F126` ControlNet PromptServer shim concurrency hardening.
- Stage 4: `F127` prompt DSL nesting guard and descriptive failure path.
- Stage 5: `F128` ControlNet tensor range normalization hardening.
- Stage 6: `R128` regression and closure.

## Phase 63 - XYZ Plot Choice-Axis Multi-Select Parity Follow-Up

Execution policy: complete on `dev` because this is a direct follow-up on the accepted phase-61 XYZ Plot delivery surface. The implementation must preserve the shipped backend contract while upgrading the frontend value-entry behavior to match A1111's choice-axis semantics more closely.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 235 | R129 | Completed | XYZ Plot Choice-Axis Parity Reference Freeze | `dev` | Completed with a focused A1111 reference freeze documenting that choice-backed XYZ axes use multiselect dropdowns by default, textbox mode is fallback rather than primary, and fill-button behavior can materialize the full choice set. |
| 236 | F129 | Completed | XYZ Plot Multi-Select Dropdown Delivery and Payload Serialization | `dev` | Completed with a RookieUI-owned custom dropdown + checkbox list for whitelisted choice axes, select-all fill behavior, preserved X/Y/Z swap semantics, and CSV serialization into the unchanged estimate/run payload contract. |
| 237 | R130 | Completed | XYZ Plot Choice-Axis Regression and Acceptance Closure | `dev` | Completed with targeted unit + Playwright proof, frontend fingerprint synchronization, full SOP gate validation, and synchronized plan/record evidence for the accepted multi-select parity follow-up. |

Stage sequencing (implementation order):
- Stage 1: `R129` XYZ Plot choice-axis parity reference freeze.
- Stage 2: `F129` multi-select dropdown delivery and payload serialization.
- Stage 3: `R130` regression and acceptance closure.

## Phase 64 - XYZ Plot Choice-Panel Visual Hardening

Execution policy: complete on `dev` because this is a direct bugfix follow-up on the accepted phase-63 choice-axis delivery surface. The implementation must preserve the shipped estimate/run payload contract while fixing long-label readability and custom-control font inheritance inside the XYZ Plot UI.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 238 | R131 | Completed | XYZ Plot Choice-Panel Visual Bug Reproduction and Contract Freeze | `dev` | Completed with a bugfix intake that pins the current defect: long choice-backed labels can be truncated inside the custom multiselect panel, and the custom `details/summary` surface must explicitly inherit surrounding RookieUI font sizing instead of relying on browser defaults. |
| 239 | F131 | Completed | XYZ Plot Choice-Panel Width, Wrap, and Tooltip Hardening | `dev` | Completed with a widened max-content choice panel, inherited typography on the custom dropdown shell, wrapped long option labels, and summary/option tooltips that preserve full filename visibility without changing payload serialization. |
| 240 | R132 | Completed | XYZ Plot Choice-Panel Regression and Acceptance Closure | `dev` | Completed with targeted unit + Playwright visual-style assertions, frontend revision synchronization, and final full SOP gate acceptance on `dev`. |

Stage sequencing (implementation order):
- Stage 1: `R131` XYZ Plot choice-panel visual bug reproduction and contract freeze.
- Stage 2: `F131` choice-panel width, wrap, and tooltip hardening.
- Stage 3: `R132` regression and acceptance closure.

## Phase 65 - XYZ Plot Choice-Dropdown Interaction Hotfix

Execution policy: complete on `dev` because this is a direct hotfix follow-up on the accepted phase-63/64 XYZ Plot choice-dropdown surface. The implementation must preserve the shipped estimate/run payload contract while fixing collapse behavior and restoring symmetric Fill-button interaction for choice-backed axes.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 241 | R133 | Completed | XYZ Plot Choice-Dropdown Interaction Bug Reproduction and Contract Freeze | `dev` | Completed with a bugfix intake that pins the current interaction defects: the custom choice dropdown stays open after outside clicks, and the `Fill` action only selects all values instead of toggling back to an empty selection on subsequent clicks. |
| 242 | F133 | Completed | XYZ Plot Choice-Dropdown Collapse and Fill-Toggle Delivery | `dev` | Completed with RookieUI-owned outside-click / Escape collapse behavior for open choice dropdowns, closure on axis changes, and `Fill` toggle symmetry that switches between select-all and clear-all without changing the accepted CSV payload contract. |
| 243 | R134 | Completed | XYZ Plot Choice-Dropdown Regression and Acceptance Closure | `dev` | Completed with targeted unit + Playwright interaction proof, final SOP-gate validation, and synchronized bugfix evidence on `dev`. |

Stage sequencing (implementation order):
- Stage 1: `R133` choice-dropdown interaction bug reproduction and contract freeze.
- Stage 2: `F133` collapse behavior and Fill-toggle delivery.
- Stage 3: `R134` regression and acceptance closure.

## Phase 66 - XYZ Plot Results Parity Hotfix

Execution policy: complete on `dev` because this is a direct hotfix follow-up on the accepted phase-61/63/65 XYZ Plot surface. The implementation must preserve the shipped session/grid payload contract while restoring A1111-referenced result behavior around preview inspection, descriptor framing, and grid persistence.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 244 | R135 | Completed | XYZ Plot Results Parity Bug Reproduction and Contract Freeze | `dev` | Completed with a bugfix intake that pins the current results-surface defects: XYZ results preview lacks the shared fullscreen/zoom affordance used by generation preview, assembled grids do not surface explicit `X` / `Y` / `Z` descriptor framing, and completed grids remain runtime-only instead of mirroring into the normal host output flow. |
| 245 | F135 | Completed | XYZ Plot Results Preview, Descriptor Framing, and Host-Output Delivery | `dev` | Completed with shared preview-viewer wiring for XYZ results, explicit axis-descriptor corner framing during grid/sub-grid assembly, and host-output mirroring for assembled main/sub-grid artifacts while preserving the accepted runtime asset contract. |
| 246 | R136 | Completed | XYZ Plot Results Parity Regression and Acceptance Closure | `dev` | Completed with targeted backend/frontend regressions, synchronized frontend fingerprinting, final SOP-gate validation, green live-host `xyz-plot --execute` confirmation, and synchronized bugfix evidence on `dev`. |

Stage sequencing (implementation order):
- Stage 1: `R135` results-parity bug reproduction and contract freeze.
- Stage 2: `F135` preview parity, descriptor framing, and host-output delivery.
- Stage 3: `R136` regression and acceptance closure.

## Phase 67 - XYZ Plot Primary Preview and Progress Hotfix

Execution policy: complete on `dev` because this is a direct hotfix follow-up on the accepted phase-61/65/66 XYZ Plot surface. The implementation must preserve the shipped session/grid payload contract while restoring top-pane preview behavior, in-progress partial grid preview delivery, and readable assembled-grid label typography.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 247 | R137 | Completed | XYZ Plot Primary Preview Bug Reproduction and Contract Freeze | `dev` | Completed with a bugfix intake that pins the current defects: running XYZ sessions expose no partial `main_grid` preview, the shared top preview box stays empty even when the local Results card has an assembled grid, and assembled-grid legend typography is too small for practical checkpoint-axis sweeps. |
| 248 | F137 | Completed | XYZ Plot Running Preview, Primary Preview Sync, and Grid Typography Delivery | `dev` | Completed with running-session partial grid preview materialization, explicit synchronization from XYZ session payloads into the shared txt2img/img2img preview box, focused guard comments at the new debug hotspots, and larger assembled-grid axis label typography while preserving ready-state asset delivery. |
| 249 | R138 | Completed | XYZ Plot Primary Preview Regression and Acceptance Closure | `dev` | Completed with targeted backend/frontend regressions, synchronized frontend fingerprinting, final SOP-gate validation, and green live-host `xyz-plot --execute` confirmation on `dev`. |

Stage sequencing (implementation order):
- Stage 1: `R137` primary-preview/progress bug reproduction and contract freeze.
- Stage 2: `F137` running partial preview, primary preview sync, and typography delivery.
- Stage 3: `R138` regression and acceptance closure.

## Phase 68 - XYZ Plot Seed-Policy Parity Follow-Up

Execution policy: complete on `dev` because this is a direct parity follow-up on the accepted phase-61/63/65/66/67 XYZ Plot surface. The implementation must preserve the shipped session/grid payload shape while adding A1111-referenced `Keep -1 for seeds` semantics, per-axis seed variation toggles, and truthful fixed-seed metadata.

References:

- `.planning/references/260418-R139F138R140_A1111_XYZ_SEED_POLICY_PARITY_REFERENCE.md`
- `.planning/plans/260418-R139F138R140_XYZ_SEED_POLICY_PARITY_PLAN.md`
- `reference/stable-diffusion-webui/scripts/xyz_grid.py`

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 250 | R139 | Completed (2026-04-18) | XYZ Plot Seed-Policy Parity Reference Freeze | `dev` | Completed via accepted reference-freeze checkpoint and roadmap reprioritization: the A1111 gap for `Keep -1 for seeds`, literal seed-axis overrides, per-axis `Vary seeds for X/Y/Z`, and truthful fixed-seed metadata is now frozen as the source-of-truth contract. |
| 251 | F138 | Completed (2026-04-18) | XYZ Plot Seed-Policy Delivery | `dev` | Added RookieUI-native XYZ Plot seed-policy controls, session-level seed materialization/preservation logic, coordinate-based seed variation toggles, and truthful session/grid metadata while keeping accepted axis/session/grid interfaces stable. |
| 252 | R140 | Completed (2026-04-18) | XYZ Plot Seed-Policy Regression and Closure | `dev` | Closed with targeted backend/frontend regressions, full SOP-gate validation, and live-host `xyz-plot --execute` confirmation proving the accepted seed-policy seam matches the planned A1111-style contract. |

Stage sequencing (implementation order):
- Stage 1: `R139` XYZ Plot seed-policy parity reference freeze.
- Stage 2: `F138` seed-policy delivery across backend session logic, UI controls, and metadata.
- Stage 3: `R140` regression and acceptance closure.

## Phase 69 - XYZ Plot Control-Surface Visual Hotfix

Execution policy: complete on `dev` because this is a direct frontend parity/hotfix follow-up on the accepted phase-61/63/65/68 XYZ Plot surface. The implementation must preserve the shipped estimate/run/session payload contract while refining option typography and aligning the action-row button treatment with the accepted Generate-button visual language.

Current working references:

- `.planning/plans/260418-R141F139R142_XYZ_CONTROL_SURFACE_VISUAL_HOTFIX_PLAN.md`

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 253 | R141 | Completed (2026-04-18) | XYZ Plot Control-Surface Visual Bug Reproduction and Contract Freeze | `dev` | Completed with a bugfix intake that pins the current UI defects: Plot Options typography reads larger than adjacent `Grid margin` labeling, the cancel action lacks danger emphasis, and the bottom action row does not yet match the accepted Generate-button accent treatment or equal-width spacing. |
| 254 | F139 | Completed (2026-04-18) | XYZ Plot Option Typography and Action-Row Styling Delivery | `dev` | Reduced Plot Options typography to the `field-label` scale, added explicit accent/danger button treatment for the XYZ action row, and enforced equal-width action buttons with distributed row alignment while preserving current behavior and payload semantics. |
| 255 | R142 | Completed (2026-04-18) | XYZ Plot Control-Surface Visual Regression and Acceptance Closure | `dev` | Closed the hotfix with targeted frontend regressions, automated Playwright before/after control-surface captures, final SOP-gate validation, and synchronized acceptance records on `dev`. |

Stage sequencing (implementation order):

- Stage 1: `R141` XYZ Plot control-surface bug reproduction and contract freeze.
- Stage 2: `F139` typography and action-row styling delivery.
- Stage 3: `R142` regression and acceptance closure.

## Phase 70 - XYZ Plot Cancel Button Ferrari-Red Hotfix

Execution policy: complete on `dev` because this is a direct frontend parity/hotfix follow-up on the accepted phase-69 XYZ Plot action-row surface. The implementation must preserve the shipped XYZ action-row sizing, payload semantics, and session controls while correcting the cancel-button danger color to a vivid Ferrari-red lane.

Current working references:

- `.planning/plans/260418-R143F140R144_XYZ_CANCEL_BUTTON_FERRARI_RED_HOTFIX_PLAN.md`

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 256 | R143 | Completed (2026-04-18) | XYZ Plot Cancel Button Color Bug Reproduction and Contract Freeze | `dev` | Completed with a bugfix intake that pins the accepted parity target: `Cancel Session` must remain a danger action, but the current pink treatment is visually incorrect for the shipped XYZ action row and needs a vivid Ferrari-red override. |
| 257 | F140 | Completed (2026-04-18) | XYZ Plot Ferrari-Red Cancel Button Delivery | `dev` | Added an XYZ-scoped cancel-button modifier and Ferrari-red gradient override so the action remains visually distinct from the orange Generate lane without changing other RookieUI danger surfaces or XYZ behavior. |
| 258 | R144 | Completed (2026-04-18) | XYZ Plot Cancel Button Regression and Acceptance Closure | `dev` | Closed the hotfix with before/after visual evidence, targeted Playwright computed-style regression coverage, final SOP-gate validation, and synchronized acceptance records on `dev`. |

Stage sequencing (implementation order):

- Stage 1: `R143` XYZ Plot cancel-button bug reproduction and contract freeze.
- Stage 2: `F140` Ferrari-red cancel-button delivery.
- Stage 3: `R144` regression and acceptance closure.

## Phase 71 - Prompt Workbench Danbooru Upsampler Editor-Toolbar Integration

Execution policy: complete on `dev` because this is a direct Prompt Workbench follow-up on the accepted phase-60 surface and requires coordinated contract, route, backend-host, and frontend-editor changes on a live integrated feature family. The implementation must preserve the shipped Prompt Workbench translation, AI-assist, history/favorites, and formatting behavior while adding a truthful host-native prompt-expansion action.

Current working references:

- `.planning/references/260418-R145F141F142R146_PROMPT_WORKBENCH_DANBOORU_UPSAMPLER_INTEGRATION_REFERENCE.md`
- `.planning/plans/260418-R145F141F142R146_PROMPT_WORKBENCH_DANBOORU_UPSAMPLER_INTEGRATION_PLAN.md`

Current execution note (2026-04-18): phase-71 is now fully closed on `dev`. Targeted regressions, the full SOP gate, prompt-workbench live-host report/execute, and restarted-host `full-pipeline` report/execute all passed against the current workspace fingerprint.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 259 | R145 | Completed (2026-04-18) | Danbooru Upsampler Prompt Workbench Contract Freeze and Host-Integration Intake | `dev` | Frozen product stance: the host-installed Danbooru upsampler lands as a Prompt Workbench editor-toolbar host action with explicit host-detection and MVP parameter boundaries rather than as a translation-provider or current AI-assist extension. |
| 260 | F141 | Completed (2026-04-18) | Prompt Workbench Danbooru Host-Action Substrate and Route Delivery | `dev` | Delivered backend contract/capability updates, host-node detection and delegation adapter, and the dedicated `/rookieui/prompt-tools/upsample` route for bounded Danbooru upsampler execution on the active ComfyUI host. |
| 261 | F142 | Completed (2026-04-18) | Prompt Workbench Danbooru Editor-Toolbar Action and Apply-Back Delivery | `dev` | Mounted a truthful `Upsample Tags` editor-toolbar lane in Prompt Workbench, including disabled-state/host-missing messaging and apply-back into the active workbench draft and bound prompt input without regressing existing Prompt Workbench actions. |
| 262 | R146 | Completed (2026-04-18) | Prompt Workbench Danbooru Integration Regression and Acceptance Closure | `dev` | Closed with targeted backend/frontend regressions, full SOP-gate validation, prompt-workbench live-host report/execute, and restarted-host `full-pipeline` report/execute acceptance evidence on `dev`. |

Stage sequencing (implementation order):

- Stage 1: `R145` Danbooru upsampler Prompt Workbench contract freeze and host-integration intake.
- Stage 2: `F141` backend host-action substrate and route delivery.
- Stage 3: `F142` editor-toolbar integration and apply-back delivery.
- Stage 4: `R146` regression and acceptance closure.

## Phase 72 - Stateful Surface Persistence and XYZ Plot Session Robustness

Execution policy: complete on `dev` after phase 62 because this chain hardens user-visible stateful surfaces whose current seams were confirmed during the 2026-04-18 robustness review. The implementation must preserve the shipped Prompt Workbench and XYZ Plot product contracts while tightening async state coordination, persistence durability, and long-running runtime retention.

Current working references:

- `.planning/references/260418-R147F143F144F145R148_STATEFUL_SURFACE_ROBUSTNESS_REFERENCE.md`
- `.planning/plans/260418-R147F143F144F145R148_STATEFUL_SURFACE_ROBUSTNESS_PLAN.md`
- `.planning/ROBUSTNESS_IMPROVEMENT_REPORT.md`

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 263 | R147 | Completed | Stateful Surface Robustness Contract Freeze | `dev` | Completed with a frozen stateful-surface proof model plus local `.planning` reference/plan artifacts covering async XYZ coordination, atomic persistence, and bounded retained session history. |
| 264 | F143 | Completed | XYZ Plot Async Session-State Concurrency Hardening | `dev` | Completed with per-event-loop `asyncio.Lock` coordination across run/list/detail/cancel session mutations, preventing coroutine interleaving while awaited host refresh work is in flight. |
| 265 | F144 | Completed | Atomic JSON Persistence for Prompt Workbench and XYZ Plot | `dev` | Completed with shared atomic temp-write/replace persistence and corrupt-file quarantine so torn writes or partial JSON loads no longer silently reset state. |
| 266 | F145 | Completed | XYZ Plot Session Retention and Pruning Guardrails | `dev` | Completed with retention-hour and terminal-count guardrails that prune only stale terminal sessions while preserving active runs and the newest retained results. |
| 267 | R148 | Completed | Stateful Surface Robustness Regression and Closure | `dev` | Completed with targeted persistence/concurrency/retention regressions, full SOP-gate sweeps, and green XYZ/full-pipeline live-host closure evidence recorded for the accepted stateful-surface baseline. |

Stage sequencing (implementation order):

- Stage 1: `R147` stateful-surface robustness contract freeze.
- Stage 2: `F143` XYZ Plot async session-state concurrency hardening.
- Stage 3: `F144` atomic JSON persistence for Prompt Workbench and XYZ Plot.
- Stage 4: `F145` XYZ Plot session retention and pruning guardrails.
- Stage 5: `R148` regression and acceptance closure.

## Phase 73 - Live-Host Freshness Hard Gate and Restart-Truthful Closure

Execution policy: complete on `dev` after phases 58-72 because this chain extends the already-accepted live-host smoke framework rather than reopening product scope. Acceptance requires two distinct proofs: stale pre-restart hosts must fail the freshness gate before lane execution, and only a restarted in-sync ComfyUI host may produce final green live-host evidence.

Current working references:

- `.planning/references/260418-R149F146R150_LIVE_HOST_FRESHNESS_GATE_REFERENCE.md`
- `.planning/plans/260418-R149F146R150_LIVE_HOST_FRESHNESS_GATE_PLAN.md`
- `.planning/ROBUSTNESS_IMPROVEMENT_REPORT.md`

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 268 | R149 | Completed | Live-Host Freshness Contract Freeze | `dev` | Freeze the corrected validation contract after the stale pre-restart host incident: live-host evidence is invalid unless the host exposes a current runtime build fingerprint that matches the workspace. |
| 269 | F146 | Completed | Runtime Build Fingerprint Exposure and Smoke Hard Gate | `dev` | Expose import-time backend runtime metadata on bootstrap/capabilities and hard-gate the smoke runner on that fingerprint before any validation lane can run. |
| 270 | R150 | Completed | Live-Host Freshness Regression and Restarted-Host Closure | `dev` | Completed with targeted freshness-gate regressions, full SOP-gate validation, explicit stale-host classification proof, and restarted-host `full-pipeline` report/execute confirmation after the external ComfyUI process reloaded the accepted workspace code. |

Stage sequencing (implementation order):

- Stage 1: `R149` live-host freshness contract freeze.
- Stage 2: `F146` runtime build fingerprint exposure and smoke hard gate.
- Stage 3: `R150` live-host freshness regression and restarted-host closure.

## Phase 74 - ERNIE-Image Host-Family UI Preset Intake

Execution policy: complete on `dev` because this chain changes canonical model-family registry, inventory-routing, preset/bootstrap, and normalized runtime-facing family truthfulness after a host-side `reference/comfyui` update introduced a new first-class `ERNIE-Image` family. The implementation must preserve the shipped SD-family parity contract and the current newer-family diffusion-model seam while adding ERNIE-Image as a truthful experimental family-adapted preset rather than overstating SDXL or A1111 semantic compatibility.

Current working references:

- `.planning/references/260418-R151F147F148R152_ERNIE_IMAGE_UI_PRESET_INTAKE_REFERENCE.md`
- `.planning/plans/260418-R151F147F148R152_ERNIE_IMAGE_UI_PRESET_INTAKE_PLAN.md`

Planned implementation note (2026-04-18): `reference/comfyui` now exposes `ERNIE-Image` through dedicated host detection, supported-model registration, model-base wiring, and ERNIE / Ministral3 text-encoder loading. RookieUI should therefore intake it through the canonical family-registry and diffusion-model preset path first, while clearly documenting that initial preset defaults are RookieUI policy and not yet host-declared preset truth.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 271 | R151 | Completed (2026-04-18) | ERNIE-Image Host Contract Freeze and Preset Intake Reference | `dev` | Completed with a frozen host-fact reference, implementation plan, command log, and roadmap synchronization that pin ERNIE-Image as a truthful experimental family-adapted intake on the existing diffusion-model seam. |
| 272 | F147 | Completed (2026-04-18) | ERNIE-Image Family Registry and Inventory Hint Foundation | `dev` | Added `ernie_image` to the canonical family-registry/capability surface and extended diffusion-model / text-encoder / VAE hint routing so host inventories resolve ERNIE selectors through the accepted non-checkpoint family seam. |
| 273 | F148 | Completed (2026-04-18) | ERNIE-Image UI Preset and Diffusion-Path Compatibility Delivery | `dev` | Exposed an ERNIE-Image preset in RookieUI bootstrap/UI payloads, kept Text Encoder visible, applied conservative first-wave defaults, refreshed frontend fallback/E2E fixtures, and preserved strict explicit-selector enforcement for diffusion-model families on txt2img/img2img normalization. |
| 274 | R152 | Completed (2026-04-18) | ERNIE-Image Regression and Host-Readiness Closure | `dev` | Closed with targeted registry/inventory/preset/normalization regressions and a full repository SOP gate; no live-host ERNIE execution proof was possible because the active host did not expose dedicated ERNIE assets/lane coverage during this chain. |

Stage sequencing (implementation order):

- Stage 1: `R151` ERNIE-Image host contract freeze and preset intake reference.
- Stage 2: `F147` family registry and inventory hint foundation.
- Stage 3: `F148` UI preset and diffusion-path compatibility delivery.
- Stage 4: `R152` regression and host-readiness closure.

## Phase 75 - ERNIE-Image Live-Host Execution Proof

Execution policy: complete on `dev` because this chain changes accepted live-host validation truth for the newly shipped ERNIE-Image preset surface and touches the runtime-facing smoke runner used for restarted-host acceptance evidence.

Current working references:

- `.planning/references/260418-R153F149R154_ERNIE_IMAGE_LIVE_HOST_EXECUTION_PROOF_REFERENCE.md`
- `.planning/plans/260418-R153F149R154_ERNIE_IMAGE_LIVE_HOST_EXECUTION_PROOF_PLAN.md`

Planned implementation note (2026-04-18): phase 74 intentionally closed without ERNIE live-host execution proof because the smoke runner still omitted ERNIE-Image from non-SD diffusion coverage. This follow-up phase is limited to validation truthfulness: add ERNIE to the existing non-SD diffusion catalog/execute lane, pin that coverage in tests, and close with restarted-host report/execute evidence.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 275 | R153 | Completed (2026-04-18) | ERNIE-Image Live-Host Proof Contract Freeze | `dev` | Completed with a dedicated phase-75 reference/plan/command-log chain that freezes ERNIE proof onto the existing non-SD diffusion smoke lane and requires restarted-host report/execute evidence for closure. |
| 276 | F149 | Completed (2026-04-18) | ERNIE-Image Live-Smoke Runner Coverage Delivery | `dev` | Added ERNIE to the non-SD diffusion smoke lane, taught execute payloads to resolve family-aware selectors from host inventory when presets leave them blank, and tightened catalog truth checks so non-ERNIE checkpoints or missing ERNIE assets fail at report time. |
| 277 | R154 | Completed (2026-04-19) | ERNIE-Image Live-Host Validation Closure | `dev` | Closed later by phase-76 restarted-host asset-ready proof once the host exposed truthful ERNIE assets; the earlier 2026-04-18 failure was an external host snapshot limitation, not a repo blocker. |

Stage sequencing (implementation order):

- Stage 1: `R153` ERNIE live-host proof contract freeze.
- Stage 2: `F149` live-smoke runner coverage delivery.
- Stage 3: `R154` restarted-host validation closure.

## Phase 76 - Official Non-SD T2I Workflow Template Alignment

Execution policy: complete on `dev` because this chain changes canonical preset/profile truth, family-specific runtime parameter surfaces, and workflow-translation topology for official non-SD families after the host-side `reference/workflow_templates` inventory was expanded.

Current working references:

- `.planning/references/260418-R155F150F151R156_OFFICIAL_NON_SD_T2I_TEMPLATE_ALIGNMENT_REFERENCE.md`
- `.planning/plans/260418-R155F150F151R156_OFFICIAL_NON_SD_T2I_TEMPLATE_ALIGNMENT_PLAN.md`

Planned implementation note (2026-04-18): the official `reference/workflow_templates` folder now contains the authoritative non-SD T2I template set for current host support. RookieUI must therefore stop treating non-SD families as a small secondary-default bucket and instead expose every official T2I template as a truthful preset/profile while aligning runtime graph topology and family-specific parameters to the official template source. `Edit`-marked templates remain out of scope for this phase, and the current user instruction also freezes `Flux.2 Dev.json` into the future i2i/edit chain because the official graph includes `LoadImage` / `VAEEncode`.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 278 | R155 | Completed (2026-04-18) | Official Non-SD T2I Template Coverage and Contract Freeze | `dev` | Completed with a frozen official non-SD T2I template inventory, an explicit `Edit`/`LoadImage`-driven i2i segregation rule, a `Flux.2 Dev.json -> i2i/edit` override, synchronized roadmap/reference/plan artifacts, and a per-item full SOP gate. |
| 279 | F150 | Completed (2026-04-18) | Official Non-SD T2I UI Preset Matrix Expansion | `dev` | Completed with canonical template-backed preset/profile entries for every official non-SD T2I workflow, refreshed model-inventory hint routing, synchronized fallback/bootstrap/catalog payloads, official-template E2E fixtures, and a full SOP gate after the shipped frontend fingerprint refresh. |
| 280 | F151 | Completed (2026-04-18) | Official Non-SD Workflow Topology and Parameter Alignment Delivery | `dev` | Completed with official non-SD topology builders for template-backed txt2img profiles, family-aware CLIP loader typing, composite and quadruple encoder routing, shift/flux-guidance/prompt-enhancement parameter surfaces, advanced guider-scheduler-sampler graphs, truthful fallback gating for ControlNet/ADetailer/Hires, and a full SOP gate after the shipped frontend fingerprint refresh. |
| 281 | R156 | Completed (2026-04-19) | Official Non-SD Template Alignment Regression and Live-Host Closure | `dev` | Completed with tightened family-aware live-smoke catalog validation, targeted regression coverage, a rerun full SOP gate, truthful host-prerequisite reporting for official families not yet asset-ready on the active host (`chroma`, `klein_*`, `hidream_*`, `longcat_image`, `qwen_image`), and restarted-host catalog+execute proof for the current asset-ready subset (`anima`, `ernie_image`, `ernie_image_turbo`, `flux`, `z_image`, `z_image_turbo`). |

Stage sequencing (implementation order):

- Stage 1: `R155` official-template coverage and contract freeze.
- Stage 2: `F150` UI preset matrix expansion.
- Stage 3: `F151` workflow topology and parameter alignment delivery.
- Stage 4: `R156` regression and live-host closure.

## Phase 77 - Official Edit-Template I2I Intake Backlog Freeze

Execution policy: keep on `dev` and defer implementation. This phase records the current explicit `Edit`-marked official template inventory for future i2i/edit work, but does not start runtime delivery until the broader edit-template set is available.

Current working references:

- `.planning/references/260418-R157_OFFICIAL_EDIT_TEMPLATE_I2I_INTAKE_REFERENCE.md`
- `.planning/plans/260418-R157_OFFICIAL_EDIT_TEMPLATE_I2I_INTAKE_PLAN.md`

Planned implementation note (2026-04-18): explicit `Edit`-marked official templates in `reference/workflow_templates` belong to the future RookieUI image-editing/i2i chain. The current user override also places `Flux.2 Dev.json` into that future chain because the official graph includes image-input semantics. More edit templates are expected later, so this phase intentionally freezes backlog ownership without beginning implementation.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 282 | R157 | Completed (2026-04-19) | Official Edit-Template I2I Intake Backlog Freeze | `dev` | Recorded the explicit `Edit` template inventory plus the `Flux.2 Dev.json` i2i override, froze the future edit/i2i ownership rule, and closed with full SOP validation so later edit-template intake can start from a stable classification contract. |

## Phase 78 - Manifest-Driven Family/Template Extensibility Foundation

Execution policy: complete on `dev` because this chain changes the canonical source of truth for family/template identity, preset/bootstrap truth, workflow-template compilation, and live-smoke prerequisite reporting across the shipped non-SD and future edit-template expansion paths.

Current working references:

- `.planning/references/260419-R158F152F153F154R159_MANIFEST_DRIVEN_FAMILY_TEMPLATE_EXTENSIBILITY_REFERENCE.md`
- `.planning/plans/260419-R158F152F153F154R159_MANIFEST_DRIVEN_FAMILY_TEMPLATE_EXTENSIBILITY_PLAN.md`

Planned implementation note (2026-04-19): phase 76 proved RookieUI can align official non-SD templates, but it also exposed the scaling cost of adding each new family/template across multiple parallel surfaces (`model_family_registry`, `presets`, `model_inventory`, frontend family-aware UI wiring, workflow builders, live-smoke catalog rules, and host-prerequisite checks). The next extensibility investment is therefore not a fantasy zero-code hot-plug system; it is a manifest-driven + adapter-based architecture where canonical family/template facts live in one manifest surface and new official-template intake mostly requires manifest data plus a bounded adapter, not repo-wide hand edits.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 283 | R158 | Completed (2026-04-19) | Manifest-Driven Family/Template Extensibility Contract Freeze | `dev` | Froze the canonical manifest taxonomy for family/template identity, flow kind, UI-parameter schema, host prerequisites, and adapter ownership, and explicitly locked the non-goal that future intake targets minimal code-change adapters rather than impossible zero-code hot-plugging. |
| 284 | F152 | Completed (2026-04-19) | Canonical Family Manifest and Derived Registry/Preset Foundation | `dev` | Landed a canonical manifest source and moved family-registry, preset/bootstrap, compatibility, frontend fallback, and family-aware parameter truth onto manifest-derived outputs. |
| 285 | F153 | Completed (2026-04-19) | Template Compiler and Family-Adapter Runtime Delivery | `dev` | Introduced a manifest-backed family-adapter runtime seam so official-template graph ownership now routes through bounded adapters instead of profile-id condition sprawl. |
| 286 | F154 | Completed (2026-04-19) | Host-Prerequisite and Live-Smoke Matrix Derivation Delivery | `dev` | Moved host prerequisite declaration and non-SD smoke profile metadata onto the canonical manifest so truthful host-asset gating expands without hand-maintained per-family smoke tables. |
| 287 | R159 | Completed (2026-04-19) | Manifest-Driven Extensibility Regression and Acceptance Closure | `dev` | Closed with regression coverage proving manifest-derived UI/preset/runtime/live-smoke truth, a full SOP gate pass, and restarted-host catalog/execute evidence for the asset-ready subset while leaving external host-asset gaps classified as prerequisites rather than repo blockers. |

Stage sequencing (implementation order):

- Stage 1: `R158` manifest/extensibility contract freeze.
- Stage 2: `F152` canonical manifest and derived registry/preset foundation.
- Stage 3: `F153` template compiler and family-adapter runtime delivery.
- Stage 4: `F154` host-prerequisite and live-smoke derivation delivery.
- Stage 5: `R159` regression and acceptance closure.

## Phase 79 - Official Img2Img and Edit-Flow Separation

Execution policy: complete on `dev` because this chain changes the public `img2img` preset surface, flow-kind filtering rules, and runtime-builder ownership for image-input models. The implementation must preserve truthful preset visibility and must not continue exposing non-SD profiles through the legacy generic i2i graph.

Current working references:

- `.planning/references/260419-R160F155F156F157R161_OFFICIAL_IMG2IMG_EDIT_FLOW_SEPARATION_REFERENCE.md`
- `.planning/plans/260419-R160F155F156F157R161_OFFICIAL_IMG2IMG_EDIT_FLOW_SEPARATION_PLAN.md`

Implementation note (2026-04-19): post-phase-78 inspection confirmed that `img2img` normalized many non-SD family selectors correctly but still routed them through the generic SD-family i2i graph (`_build_sd15_img2img_graph` / `_build_sdxl_img2img_graph`) rather than the official family/template builders. Phase 79 closed that user-safety gap by hiding unaligned presets from the generic `img2img` surface, introducing a real official edit runtime seam, and partitioning future edit models into a dedicated flow-aware surface. A same-day follow-up also aligned shipped fixed-LoRA official templates (`flux`, `qwen_image`, `qwen_image_edit`) to a template-owned LoRA contract with truthful override messaging and host-readiness validation. Remaining work is limited to restarted-host closure for `R161`.

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 288 | R160 | Completed (2026-04-19) | Official Img2Img/Edit Flow Scope and Safety Gate Freeze | `dev` | Froze the classification and UX rule that the current generic i2i surface is SD-family-first, while non-SD official-template families stay hidden there until aligned i2i/edit builders exist; also froze the later `Edit` split so official edit graphs are not reintroduced as generic img2img presets. |
| 289 | F155 | Completed (2026-04-19) | Img2Img Preset Safety Filter for Unaligned Non-SD Profiles | `dev` | Hid known non-SD official-template presets from the generic `img2img` tab so users can no longer route them into the legacy diffusion-model i2i graph and assume unsupported official-template alignment. |
| 290 | F156 | Completed (2026-04-19) | Official Non-SD Img2Img Template Runtime Delivery | `dev` | Delivered a dedicated official non-SD image-input runtime seam, including the first shipped official edit adapter (`qwen_image_edit`) plus truthful image-input/no-mask ownership and template-owned fixed-LoRA handling for shipped official templates. |
| 291 | F157 | Completed (2026-04-19) | Edit-Flow Surface Split and Preset Partition Delivery | `dev` | Introduced flow-kind-aware preset partitioning and a dedicated `Edit` model surface inside the image-input workspace so official edit templates and future image-editing models no longer compete with the existing denoise-based i2i modes. |
| 292 | R161 | Completed (2026-04-19) | Img2Img/Edit Separation Regression and Acceptance Closure | `dev` | Closed with a green full SOP gate plus restarted-host live-smoke truthfulness proof: the repo now rejects missing or non-official template-owned LoRA states for `flux`, `qwen_image`, and `qwen_image_edit` as external host prerequisites instead of silently claiming official-template parity. |

Stage sequencing (implementation order):

- Stage 1: `R160` i2i/edit scope and safety gate freeze.
- Stage 2: `F155` immediate `img2img` preset safety filter.
- Stage 3: `F156` official non-SD img2img runtime delivery.
- Stage 4: `F157` edit-flow surface split and preset partition.
- Stage 5: `R161` regression and acceptance closure.

## Phase 80 - Non-SD Inline LoRA Model-Only Chain

Execution policy: implement on `dev` with contract-first sequencing; preserve the shipped template-owned LoRA contract for official non-SD templates while appending user inline `<lora:...>` activations as model-only `Load LoRA` nodes after the official chain on both txt2img and edit runtimes. Do not reuse the SD-family `LoraLoader(model+clip)` seam on these official non-SD paths.

Current working references for Phase 80 non-SD inline LoRA chain:
- `.planning/references/260419-R162F158R163_NON_SD_INLINE_LORA_MODEL_ONLY_CHAIN_REFERENCE.md`
- `.planning/plans/260419-R162F158R163_NON_SD_INLINE_LORA_MODEL_ONLY_CHAIN_PLAN.md`

| Order | Code | Status | Title | Branch | Summary |
| --- | --- | --- | --- | --- | --- |
| 293 | R162 | Completed (2026-04-19) | Non-SD Inline LoRA Contract Freeze and Ordering Rule | `dev` | Froze the non-SD inline LoRA contract so A1111-style `<lora:...>` prompt activations resolve against host LoRA inventory, preserve any shipped template-owned LoRA first, and append user activations afterward as ordered model-only `Load LoRA` nodes on official txt2img/edit runtime paths. |
| 294 | F158 | Completed (2026-04-19) | Non-SD Model-Only Inline LoRA Chain Delivery | `dev` | Delivered the shared non-SD model-only LoRA chain helper, wired it into shipped official txt2img/edit builders, and kept prompt-side clip/TE strength drift truthful instead of pretending SD-style `model+clip` parity exists on non-SD templates. |
| 295 | R163 | Completed (2026-04-19) | Non-SD Inline LoRA Regression and Acceptance Closure | `dev` | Closed with targeted regression coverage for template-owned-first ordering, non-template non-SD inline LoRA delivery, clip/TE drift warnings, and a green full SOP gate; no dedicated live-host lane exists yet for inline prompt LoRA execution proof. |

Stage sequencing (implementation order):
- Stage 1: `R162` contract freeze and reference capture.
- Stage 2: `F158` shared non-SD model-only inline LoRA runtime delivery.
- Stage 3: `R163` regression, SOP gate, and truthful live-host closure.

## Item Dependency Map

- `R01` before all implementation items.
- `S01` before any upload, metadata, or route-heavy feature.
- `F01` depends on `R01`.
- `R02` depends on `R01` and defines the acceptance target for `F02`, `F04`, `F05`, and `F06`.
- `F02` depends on `F01` and `R02`.
- `F03` depends on `R02` and feeds `F02`, `F04`, `F06`, and `F08`.
- `F04` depends on `F02`.
- `F05` depends on `F02` and `F03`.
- `F07` depends on `F01` and `F02`.
- `S02` depends on `R02` and must be complete before broadening external inputs.
- `R03` spans all shipped items and becomes mandatory before promotion to `main`.
- `R87` depends on `R86`, `R72`, and `F75` because detect backend alignment must preserve established integrated ControlNet contract metadata and dynamic module/model routing behavior.
- `F89` depends on `R87` because extension-first detect execution must implement the frozen warning, schema, and runtime-boundary contract.
- `R88` depends on `F89` and `R03` because regression hardening must pin the new detect policy while retaining full-suite acceptance guarantees.
- `R89` depends on `R88`, `F84`, and `R03` because fullscreen/hover/header parity contract freeze must preserve phase-38 detect behavior, existing canvas parity primitives, and regression governance.
- `F90` depends on `R89`, `F87`, and `R77` because fullscreen zoom and header/hover implementation must follow the frozen contract while remaining consistent with source-canvas fidelity and run-preprocessor layout contracts.
- `R90` depends on `F90` and `R54` because regression hardening must validate the new fullscreen/hover/header UI behavior under repository full-gate automation.
- `R91` depends on `R87`, `R90`, and `R03` because detect endpoint policy override must preserve established detect diagnostics and recent UI/runtime hardening while removing invalid implicit host coupling assumptions.
- `F91` depends on `R91`, `F89`, and `R54` because hard-coded endpoint removal must follow the policy override, keep detect-contract compatibility, and remain acceptance-gated.
- `R92` depends on `F91` and `R03` because endpoint-gating hardening must pin the new routing policy with targeted and full-gate regression proof.
- `R93` depends on `R92`, `R90`, and `R03` because Forge-native host-preprocessor contract freeze must preserve endpoint de-hardcoding policy and recent integrated ControlNet UI/runtime behavior.
- `F92` depends on `R93`, `F75`, and `R54` because host dispatcher delivery must align with existing module/control-type contract surfaces and remain acceptance-gated.
- `R94` depends on `F92` and `R03` because regression hardening must pin host dispatcher behavior with targeted and full-gate verification.
- `R97` depends on `R94`, `R96`, and `R03` because depth deterministic-dispatch policy must preserve the shipped host-preprocessor baseline and fullscreen/caret parity contracts.
- `F94` depends on `R97`, `F92`, and `R54` because deterministic runtime delivery and control-model diagnostics must build on existing host dispatcher seams and remain acceptance-gated.
- `R98` depends on `F94` and `R03` because final hardening must pin black-preview/caret/diagnostic behavior with targeted and full-gate evidence.
- `R99` depends on `R98`, `F92`, and `R03` because all-module policy expansion must preserve the shipped host-preprocessor baseline and prior depth-only hardening.
- `F95` depends on `R99`, `F94`, and `R54` because global deterministic probe/AIO gate delivery extends existing runtime seams and remains acceptance-gated.
- `R100` depends on `F95` and `R03` because full closure requires targeted non-depth/AIO regression evidence plus full SOP gate proof.
- `R101` depends on `R100`, `F75`, and `R03` because variant contract freeze must extend the integrated ControlNet catalog surface while preserving existing acceptance governance.
- `F96` depends on `R101`, `F92`, and `R54` because variant filtering and dispatch delivery must align with current host-preprocessor runtime seams and remain acceptance-gated.
- `R102` depends on `F96` and `R03` because final closure requires targeted backend/frontend variant regression evidence plus full SOP gate proof.
- `F06` depends on `F02`, `F03`, `F04`, and `F05`, but core SD-family parity requirements are already owned by `R02` and `F02`.
- `F08` depends on `F03` and `F06`.
- `R07` follows `F10` and hardens live-host tab behavior for the post-roadmap sidebar shell.
- `F11` depends on `R07` so the simplified `Settings` pane lands on top of the corrected pane-isolation behavior.
- `R08` follows `F10` and hardens numeric-entry contracts for the post-roadmap sidebar shell.
- `F12` depends on `R08` because optional-hires behavior must be validated on top of the corrected decimal-entry seam.
- `F13` depends on `F12` so the final `Enable Hires` row layout lands on top of the corrected optional-hires behavior and DOM structure.
- `R09` follows `F13` and defines the next host-safe visual contract for the post-roadmap sidebar shell.
- `F14` depends on `R09` because the quicksettings strip and top-level shell frame must be established before pane internals move.
- `F15` depends on `F14` so the `txt2img` and `img2img` workspace relayout lands inside the new Forge-Neo-inspired shell frame.
- `R10` follows `F15` because live-host generation is currently blocked by host selector canonicalization and A1111 `seed=-1` semantics not being translated into valid ComfyUI execution inputs.
- `F16` depends on `R10` so the next UI density pass lands on top of a generation path that can actually validate and run in the host.
- `F17` depends on `R10` and `F16` because real sampler/scheduler dropdowns need both backend catalog support and the refined compact generation layout.
- `F18` depends on `F16` and `F17` because the fine-grained Forge-Neo visual layer, including slider-backed parameter rows, should land after the compacted shell and real control surfaces are in place.
- `F19` depends on `F18` because the quicksettings density follow-up must tune the already-tokenized Forge-Neo shell rather than reworking pre-parity sizing rules.
- `F20` depends on `F19` because the action rail and preview-strip parity should land on top of the final compact quicksettings proportions.
- `F21` depends on `F20` because slider recoloring is the last visual parity pass after the action-rail surface is stabilized.
- `R14` follows `F21` because the next chain must correct the host model inventory foundation before more family-aware UI, PNG Info apply, or Extras work can be safely expanded.
- `F26` depends on `R14` so the sidebar catalog surface consumes the corrected host inventory model instead of extending the current checkpoint-first payload.
- `R15` depends on `R14` and `F26` because prompt-side extra-network syntax cannot be resolved safely until the host inventory baseline covers LoRAs, embeddings, and related selector categories correctly.
- `F27` depends on `R15` because inline LoRA syntax should only ship after prompt preprocessing, cleaned-prompt output, and structured extra-network extraction exist.
- `R11` depends on `R14`, `F26`, and `F27` because family-aware control gating should sit on top of the expanded host inventory baseline and the corrected prompt-side activation semantics.
- `F22` depends on `R11` so the control-density pass lands on top of the corrected family-aware behavior and submit/queue semantics.
- `R12` depends on `R14`, `F27`, and `R11` because PNG Info apply flow and asset bridging must align with the corrected host inventory model, prompt-side activation semantics, and the RookieUI-origin request semantics.
- `F23` depends on `R12` so the image-first PNG Info surface lands only after dual-metadata inspection and safe asset-handle plumbing exist.
- `F24` depends on `F23` because the dormant Settings surface should disappear only after PNG Info becomes a real, self-contained inspector/apply pane.
- `R13` depends on `R14` and `F24` and remains lower priority inside the chain so Extras does not delay the earlier inventory, prompt-DSL, SD-family semantic, and PNG Info corrections.
- `F25` depends on `R13` because Extras must ship as a dedicated postprocessing subsystem rather than an improvised extension of the generation translator.
- `R16` follows `R11` and `F25` to harden RookieUI-vs-host queue ownership before new live preview/runtime coupling expands.
- `F28` depends on `F22` and targets the remaining prompt-band alignment regression in live host rendering.
- `F29` depends on `F20` and `F22` so icon-color parity lands on top of the stabilized compact action rails and typography pass.
- `F30` depends on `F22` and removes redundant prompt chrome without reintroducing label-density regressions.
- `R17` depends on `R12` and `F23` because PNG Info apply controls must be corrected on top of the image-first inspection architecture.
- `F31` depends on `R14`, `F26`, and `R15` so preset taxonomy refresh uses the expanded host catalog and prompt/extra-network baseline.
- `F32` depends on `R16`, `F07`, and `R03` because live generation feedback must remain isolated to RookieUI-origin jobs while preserving queue/history regression coverage.
- `F33` depends on `R12` and `R17` so auto-inspect ships with the hardened PNG Info action rail and metadata bridge.
- `R18` depends on `R11` and `F31` to keep family-aware control exposure consistent after preset taxonomy updates.
- `R145` depends on `R124` because the Danbooru upsampler intake must build on the accepted Prompt Workbench phase-60 contract rather than reopening the base workbench architecture.
- `R151` depends on `F72` and `R03` because the ERNIE-Image intake must start from the accepted canonical family-registry surface and repository-wide truthfulness/acceptance governance rather than adding another ad-hoc family path.
- `F147` depends on `R151` and `F72` because ERNIE-Image registry and selector-hint delivery must follow the frozen host contract and continue using the canonical family-registry source of truth.
- `F148` depends on `F147` and `R18` because the ERNIE-Image preset must land on top of the accepted family-aware text-encoder visibility rules while preserving explicit selector enforcement for diffusion-model families.
- `R152` depends on `F148` and `R03` because final closure requires targeted regression evidence plus full SOP-gate validation after the new family appears in registry/preset/bootstrap surfaces.
- `R153` depends on `R152` and `R66` because ERNIE execution proof only becomes meaningful after the preset intake is shipped and the repo already has a restart-aware live-host validation framework.
- `F149` depends on `R153` and `R66` because ERNIE smoke coverage must follow the frozen proof contract and extend the accepted live-host runner rather than inventing a separate validation path.
- `R154` depends on `F149`, `R03`, and host-side ERNIE asset availability because final closure requires targeted regressions, a full SOP gate, and restarted-host ERNIE report/execute evidence against truthful host selectors.
- `R147` depends on `R128`, `R124`, and `R03` because the stateful-surface hardening chain should start from the accepted runtime hardening baseline, the shipped Prompt Workbench contract, and repository-wide acceptance governance.
- `F143` depends on `R147`, `R126`, and `R03` because XYZ session-state concurrency hardening must preserve the accepted queue-backed XYZ session runner semantics while tightening only the async mutation boundary.
- `F144` depends on `R147`, `F114`, `F124`, and `R03` because atomic persistence must preserve the shipped Prompt Workbench and XYZ Plot state contracts rather than introducing a new storage model.
- `F145` depends on `R147`, `F123`, and `R03` because XYZ retention/pruning guardrails should build on the shipped persisted session/grid delivery path without weakening active-session observability.
- `R148` depends on `F143`, `F144`, `F145`, and `R03` because final closure requires targeted persistence/concurrency proof plus full acceptance-gate validation after all new hardening changes land.
- `F141` depends on `R145` and `R54` because the host-action backend and route delivery must follow the frozen Prompt Workbench integration target and remain acceptance-gated.
- `F142` depends on `F141` and `R145` because the editor-toolbar action should land only after the host capability and route substrate are stable.
- `R146` depends on `F142` and `R03` because final closure requires targeted Prompt Workbench regression evidence plus the repository-wide SOP gate.
- `F34` follows `F25` and `R09` because Extras button alignment is a host-rendering layout correction on top of the shipped Extras surface.
- `F35` follows `F22` and `F21` because control-density reduction must tune the stabilized compact shell instead of pre-parity sizing.
- `R19` follows `F04` and `R11` because img2img asset handles must be validated inside RookieUI boundaries before host queue submission.
- `R20` follows `R17` because PNG Info action-rail hardening needs an additional host-safe button height lock.
- `F36` depends on `R14`, `F26`, and `F31` and is explicitly lower priority so SD-family parity and release-stability fixes remain first.
- `F37` follows `F35` and `F21` because slider micro-geometry and base-color tuning should only target the already-stabilized compact slider contract.
- `F38` depends on `R12`, `F23`, and `F33` because automatic positive/negative extraction should remain inside the existing image-first metadata bridge and auto-inspect flow.
- `F39` depends on `F38` and `R12` because infotext-surface retirement must land after image-first extraction is complete and validated end-to-end.
- `R21` depends on `R16`, `F32`, and `F07` because host-event live preview hardening must preserve RookieUI-origin queue isolation and existing queue/history contracts.
- `F40` follows `F22`, `F35`, and `R09` because top-tab scale and divider framing should extend the stabilized shell spacing contract instead of introducing a parallel layout primitive.
- `F41` depends on `F30` and `F18` because Hires.fix chrome parity builds on the current compact typography and icon/chrome token system.
- `R22` depends on `R11`, `F04`, and `F41` because img2img Hires.fix restoration must preserve family-aware control exposure and execute through validated img2img/inpaint translation seams.
- `F42` depends on `F25`, `R13`, and `F41` because Extras Hires.fix surface recovery should piggyback on the shipped Extras postprocessing subsystem and shared Hires chrome pattern.
- `F43` follows `F40` and `F20` because hero-label typography tuning is a final visual pass after action-rail and top-tab sizing stabilize.
- `F44` follows `F29`, `F20`, and `F41` because native emoji icon replacement should land after color/icon rail semantics and Hires chrome framing are settled.
- `F45` follows `F44` and `R03` because public release metadata/docs packaging should happen after the main release-candidate UI/runtime lanes and full acceptance governance are in place.
- `F46` depends on `F41` because triangle-state recovery must reuse the established Hires.fix header/chrome contract.
- `F47` depends on `F41` and `F46` because helper-copy retirement should be finalized after Hires chrome/state rendering is stabilized.
- `F48` depends on `F41` and `F46` because header-edge checkbox placement and collapsed-state interaction are tied to the same Hires header layout seam.
- `R23` depends on `R22`, `F46`, and `F48` because img2img Hires completion requires both backend second-pass contract restoration and finalized shared Hires header behavior.
- `F49` depends on `F39` because legacy Infotext UI retirement should remain aligned with the image-first PNG Info direction.
- `R24` depends on `F38`, `F49`, and `R12` because dual-prompt extraction completion must run on top of image-first ingest and explicit infotext-surface retirement.
- `R25` depends on `R21` and `F32` because live preview completion hardening extends the existing host-event compatibility and runtime progress contracts.
- `F50` depends on `F20`, `F34`, and `F43` because cross-tab Generate width unification should inherit the stabilized hero/button rail geometry.
- `R26` depends on `R04`, `F10`, and `F50` because sidebar minimum-width expansion must build on prior width-guard work and final button baseline alignment.
- `R27` depends on `R16`, `R07`, and `R11` because cross-tab state locking must preserve pane isolation, queue boundaries, and family-aware parameter semantics.
- `F51` depends on `F50`, `F34`, and `F28` because hero-height reduction must preserve previously stabilized top-edge alignment and cross-tab hero geometry.
- `F52` depends on `R10`, `R11`, and `F44` because A1111-style seed emoji behavior and Extra feature toggles must remain consistent with seed normalization and current emoji-centric action surfaces.
- `R28` depends on `R19`, `R27`, and `F07` because send-to-img2img transfer integrity requires stable asset-handle semantics plus deterministic cross-tab state transfer.
- `R29` depends on `F04`, `R23`, and `R14` because the remaining Img2Img feature-surface expansion must build on the shipped img2img/inpaint baseline, restored hires seam, and host inventory contracts.
- `R30` depends on `R25`, `R21`, and `R16` because preview-runtime completion rework extends the existing host-event adaptation while preserving RookieUI queue/session boundaries.
- `F53` depends on `F49`, `R24`, and `R26` because PNG Info layout reflow must land on top of image-first metadata behavior and the newer sidebar width baseline.
- `F54` depends on `F40`, `F50`, and `R26` because enlarged tab chrome and divider framing must inherit the stabilized shell sizing and sidebar width baseline.
- `F55` depends on `F44`, `F25`, and `F50` because Extras action emoji recovery must stay aligned with the native emoji ToolButton contract and finalized Generate-rail geometry.
- `R31` depends on `R29`, `R23`, and `R19` because sketch/batch lane expansion should build on the current Img2Img parity lane, existing hires restoration, and asset-handle validation seam.
- `R32` depends on `F31`, `R18`, and `R14` because Flux/Qwen text-encoder visibility must remain consistent with preset taxonomy, family-aware control gating, and host model catalog/text-encoder inventory contracts.
- `F56` depends on `F51`, `F50`, and `F34` because the second-pass Generate-height reduction must preserve previously stabilized height, width, and top-edge alignment rails across tabs.
- `R34` depends on `R33`, `R27`, and `R03` because regression-capture hardening must pin the fixed Clip Skip behavior across preset-switch, tab-restore, and cross-surface parity lanes.
- `R35` depends on `F30`, `R27`, and `R03` because prompt usability recovery must preserve streamlined prompt chrome, cross-tab state behavior, and regression harness guarantees.
- `R36` depends on `R30`, `R25`, and `R03` because preview flicker mitigation extends live-preview runtime seams and must preserve existing runtime completion and regression guarantees.
- `R37` depends on `F57`, `R12`, and `R03` because PNG Info preview alignment must preserve image-first ingest architecture, metadata apply workflow wiring, and UI regression coverage.
- `R38` depends on `F25`, `R13`, and `R03` because Extras coercion consistency must preserve the established postprocessing contract and regression guarantees.
- `R39` depends on `R29`, `R31`, and `R03` because img2img denoise coercion alignment must remain compatible with the expanded multi-mode img2img normalization surface.
- `R40` depends on `R38`, `R39`, and `R03` because shared coercion extraction should land after immediate parity bugfixes and retain broad regression coverage across service modules.
- `R41` depends on `R24`, `R29`, and `R03` because alias consolidation must preserve PNG Info dual-prompt ingest and img2img feature-surface semantics across normalization seams.
- `R42` depends on `R14`, `F26`, and `R03` because model inventory caching must build on the expanded host catalog baseline and remain regression-safe for family-aware selector behavior.
- `R43` depends on `R16`, `R30`, and `R03` because defensive logging should instrument existing fallback seams without altering queue/runtime behavior or acceptance guarantees.
- `R44` depends on `R19`, `R24`, and `R03` because runtime asset cleanup must preserve valid img2img/pnginfo asset-handle workflows while bounding disk growth safely.
- `R45` depends on `R29`, `R31`, and `R03` because node-id allocation unification must preserve the currently shipped img2img/inpaint/hires graph translation behavior.
- `R46` depends on `F54`, `F57`, and `R03` because utility extraction must preserve current tab/layout parity outcomes and remain covered by existing regression lanes.
- `R47` depends on `R46`, `R27`, and `R03` because per-tab split should build on extracted utilities while preserving cross-tab state-lock behavior and overall acceptance guarantees.
- `R48` depends on `R47`, `R36`, and `R03` because debug-flag telemetry should instrument the modularized sidebar and existing live-preview/runtime fallback seams without behavior regression.
- `R49` depends on `R29`, `R31`, and `R47` because mask-canvas state and payload bridge work must build on the current img2img multi-mode contract and the modularized per-tab shell seam.
- `F58` depends on `R49`, `R36`, and `R48` because embedded mask drawing UX should land on top of the finalized contract while preserving live preview paint stability and debug traceability.
- `F59` depends on `F58`, `R45`, and `R03` because advanced mask tooling must extend the core canvas safely across expanded img2img workflow translation paths and remain regression-proven.
- `R50` depends on `F59`, `R27`, and `R47` because generation-subtab routing must sit on top of the latest img2img mode/canvas surface while preserving pane-state lock and modular tab-shell seams.
- `F60` depends on `R50`, `R31`, and `F54` because visible multi-tier mode rails should be delivered after router-state hardening, current img2img multi-mode behavior, and tab-scale visual contracts are stable.
- `R51` depends on `F60`, `R47`, and `R48` because module-boundary formalization should follow the latest img2img subtab/canvas shape and existing modular shell/debug seams.
- `R52` depends on `R51`, `R27`, and `R16` because state/event extraction must preserve pane-state lock and queue/session ownership guarantees.
- `F61` depends on `R52`, `F60`, and `R49` because img2img extraction requires stabilized router/state contracts plus current subtab and mask-canvas integration behavior.
- `F62` depends on `F61`, `R46`, and `F25` because remaining pane extraction should follow the hardest img2img split and preserve existing utility/postprocessing seams.
- `F63` depends on `F62`, `F54`, and `F57` because CSS ownership split should land after pane-module boundaries stabilize and must preserve tab/frame and PNG Info layout contracts.
- `R53` depends on `F63`, `R34`, and `R03` because modularization regression hardening extends existing deterministic harness guarantees across newly extracted seams.
- `R54` depends on `R03`, `R53`, and `R44` because test-automation scripts and pre-push hook enforcement must track the current regression-harness baseline, respect modularized frontend seams, and preserve runtime-artifact guardrails during local full-gate execution.
- `F64` depends on `F41`, `F56`, and `R47` because Hires.fix-in-Generation layout integration must preserve existing Hires header/toggle chrome, current hero/control spacing outcomes, and per-pane module split contracts.
- `R55` depends on `R15`, `R29`, and `R03` because prompt parity contract freeze must align with existing prompt DSL baseline, current img2img parity surface, and regression-governed acceptance rules.
- `F65` depends on `R55`, `F27`, and `R03` because parser v2 extends prompt DSL behavior only after capability boundaries are frozen and inline-LoRA semantics are preserved under regression control.
- `F66` depends on `F65`, `R45`, and `R03` because conditioning compilation must consume structured prompt semantics and remain compatible with unified node-id allocation and regression guarantees.
- `R56` depends on `F66`, `R48`, and `R03` because parity guardrails and warning-code fallback instrumentation should land on top of compiler integration and existing debug/telemetry seams.
- `R57` depends on `R49`, `R50`, and `R28` because source/mask bridge hardening must preserve current mask-canvas and multi-tier mode-router contracts while keeping send-to-img2img transfer integrity.
- `F67` depends on `R57`, `R31`, and `R29` because full img2img mode completion should build on hardened source/mask state bridges and current mode-surface/backend normalization contracts.
- `F68` depends on `F67`, `F60`, and `R27` because multi-tier layout parity stabilization must preserve completed mode behavior plus existing nested-tab and cross-tab persistence contracts.
- `R58` depends on `F68`, `R34`, and `R03` because parity regression expansion should harden the newly completed prompt/img2img chain using existing deterministic regression governance.
- `R59` depends on `F58`, `F59`, and `R28` because placeholder false-positive remediation must preserve shipped mask-editor UX/tooling and send-to-img2img transfer semantics while correcting source-visibility state behavior.
- `R60` depends on `R14`, `R55`, and `R03` because ControlNet intake must align with expanded host model inventory, frozen parity-contract governance, and deterministic acceptance coverage.
- `F69` depends on `R60`, `R45`, and `R03` because ControlNet graph integration must build on finalized contract mapping and preserve unified node-id allocation/regression guarantees.
- `F70` depends on `F69`, `R50`, and `R27` because A1111-style unit UX must reflect real backend semantics while preserving current multi-tier mode routing and cross-tab state persistence.
- `F71` depends on `R60`, `F69`, and `R48` because ControlNet API and detect pipeline must consume frozen unit contracts, execute against shipped graph behavior, and expose diagnosable guarded telemetry.
- `R61` depends on `F70`, `F71`, and `R03` because ControlNet hardening must validate final UI/API/runtime seams with rollback-ready guardrails and full acceptance harness parity.
- `R62` depends on `F66`, `R45`, and `R03` because workflow-translation seam consolidation must preserve current conditioning compiler behavior, deterministic node-id allocation, and regression-governed parity.
- `R63` depends on `R48`, `F63`, and `R03` because revision-token centralization must preserve modularized frontend asset boundaries, guarded telemetry seams, and existing regression guarantees.
- `R64` depends on `R40`, `R41`, and `R03` because coercion single-source completion should extend the shared utility/alignment baseline while preserving prompt/pnginfo parity contracts.
- `R65` depends on `R47`, `R53`, and `R03` because TS-first foundation must build on established module boundaries and hardened regression seams without changing runtime behavior.
- `R66` depends on `R54`, `R65`, and `R03` because host-embedded E2E lane expansion should extend the existing SOP/automation baseline after TS-first seams are in place.
- `R67` depends on `R53`, `R63`, and `R03` because continued service extraction should follow stabilized module boundaries and shared revision-token ownership while preserving regression guarantees.
- `F72` depends on `R14`, `F31`, and `R60` because capability registry surfacing must align host inventory breadth, current family-preset taxonomy, and ControlNet-era family-aware execution contracts.
- `R68` depends on `R65`, `R66`, and `R03` because Vite feasibility evaluation requires type-safe/module-stable baseline plus real-host validation governance before any migration decision.
- `R69` depends on `R67`, `F72`, and `R03` because Vue host-adapter feasibility must build on stabilized frontend service boundaries and explicit capability contracts under full regression governance.
- `R106` depends on `R55`, `F65`, `F66`, and `R03` because parity re-baselining must start from the shipped prompt contract, current parser/compiler behavior, and existing regression governance before any cutover is planned.
- `F97` depends on `R106`, `R15`, `F27`, and `R03` because the A1111-native prompt engine must preserve the existing prompt-side extra-network / inline-LoRA semantics while replacing only the incorrect semantic-ownership layer.
- `R107` depends on `F97` and `R03` because the host encoder-boundary contract can only be frozen after the parity engine core exists and remains governed by current acceptance discipline.
- `F98` depends on `R107`, `F97`, and `R03` because SD1.x / SD2.x parity-node delivery must follow the frozen adapter seam and the new prompt engine while remaining rollback-safe.
- `F99` depends on `R107`, `F97`, `F98`, and `R03` because SDXL parity-node delivery should extend the proven single-encoder adapter pattern while accounting for dual-encoder / pooled-output constraints.
- `R108` depends on `F98`, `F99`, and `R03` because workflow translator cutover must only happen after both primary SD-family parity-node lanes exist and remain regression-governed.
- `F100` depends on `R108`, `F72`, and `R03` because capability / UI / API truthfulness should be updated after the new runtime path exists and in alignment with the planned model-family capability registry direction.
- `R109` depends on `R58`, `R114`, `R115`, `F100`, and `R03` because final parity closure now requires the shipped golden-regression chain, the delivered prompt-parity live-smoke lane, and a green in-sync deployed-host evidence pass before exact prompt-parity claims can be accepted.
- `R113` depends on `R108`, `F100`, `R14`, and `R03` because embeddings/textual inversion contract freeze must build on the shipped SD-family encoder seam, truthful capability wording, the existing host embeddings inventory baseline, and repository-wide acceptance governance.
- `F103` depends on `R113`, `R108`, `R14`, and `R03` because embeddings/textual inversion delivery must follow the frozen contract, stay on the shipped SD-family encoder seam, use the established host inventory baseline, and remain full-gate accepted.
- `F104` depends on `F103`, `R108`, and `R03` because golden prompt fixtures must reflect the shipped prompt path, not a pre-delivery abstraction, and must close with repository-wide regression governance.
- `R114` depends on `F104`, `R54`, and `R03` because prompt-parity host smoke must sit on the current automated test baseline, validate the delivered fixture contract, and classify live-host/workspace drift truthfully before any broader `R66` generalization claim.
- `R115` depends on `R114`, `F104`, and external host deployment synchronization outside the workspace boundary because the green deployed-host evidence pass is only meaningful after the smoke lane exists and the live ComfyUI install is updated to the accepted workspace code.
- `R116` depends on `R109`, the accepted local reference corpus, and `R03` because the next same-scope parity chain must start from the already-closed SD-family baseline and remain governed by the repository acceptance discipline.
- `F105` depends on `R116`, `R108`, and `R03` because alternate scheduling delivery must follow the frozen continuation contract, stay on the shipped SD-family encoder/compiler seam, and remain full-gate accepted.
- `F106` depends on `R116`, `F105`, and `R03` because tokenizer-side chunk hardening should build on the newly frozen continuation scope and the shipped alternate-scheduling-aware prompt path while remaining rollback-safe.
- `R117` depends on `F105`, `F106`, `R114`, `R115`, and `R03` because maximal same-scope closure requires the strengthened prompt path plus refreshed standalone and live-host evidence before any stronger parity claim can be accepted.
- `R118` depends on `R117`, `F45`, and `R03` because the public README should only be updated after the shipped prompt-parity continuation is accepted, while still following the repository documentation-packaging baseline and full-gate governance.
- `R119` depends on `R54`, `R117`, and `R03` because auxiliary live-host validation must build on the current smoke-lane/test-automation baseline, reuse the accepted prompt-parity host-validation framework, and remain under full repository acceptance governance.
- `F107` depends on `R119`, `R61`, and `R03` because ControlNet live-host validation should extend the shared smoke framework only after the shipped ControlNet contract/UI/API/runtime parity chain is already accepted and regression-governed.
- `F108` depends on `R119`, `R112`, and `R03` because ADetailer live-host validation should extend the shared smoke framework only after the shipped native detector/runtime and hardening chain is already accepted and regression-governed.
- `F109` depends on `R119`, `F05`, `F07`, and `R03` because the Extras / PNG Info / Queue live-host lane must build on the current auxiliary-surface contracts and remain under the repository-wide acceptance baseline.
- `R120` depends on `F107`, `F108`, `F109`, and `R03` because aggregate queue/post-state closure is only meaningful after the individual live-host auxiliary lanes exist and pass full repository acceptance.
- `R121` depends on `R120`, `R62`, and `R03` because the extensibility-first refactor should start from the accepted live-host validation baseline, reuse the earlier graph-builder modernization direction, and remain under repository-wide acceptance governance.
- `F110` depends on `R121`, `R108`, `R120`, and `R03` because workflow feature-builder extraction must preserve the shipped SD-family prompt/controlnet/adetailer execution path and remain live-host regression-governed.
- `F111` depends on `R121`, `F110`, `R61`, and `R03` because the ControlNet vertical split should land on top of the new builder seam while preserving the accepted ControlNet contract/UI/API/runtime baseline.
- `F112` depends on `R121`, `F110`, `R112`, and `R03` because the ADetailer vertical split should reuse the extracted refinement/builder seam while preserving the accepted native detector/runtime and ADetailer-local ControlNet baseline.
- `F113` depends on `F111`, `F112`, `R119`, and `R03` because integrated-feature registry/bootstrap consolidation should reflect already-split service ownership while remaining compatible with the shipped live-host validation framework.
- `R122` depends on `F110`, `F111`, `F112`, `F113`, and `R03` because final refactor closure requires targeted topology/import-cycle/size-budget regression proof plus full repository validation and live-host evidence.
- `R123` depends on `R122`, `R108`, and `R03` because prompt-workbench migration should start from the accepted extensibility seams, preserve the shipped SD-family prompt runtime baseline, and remain under repository-wide acceptance governance.
- `F114` depends on `R123`, `F113`, and `R03` because persistent prompt-workbench state/routes should build on the frozen migration contract and the existing integrated-feature registry seam instead of introducing new ad-hoc bootstrap ownership.
- `F115` depends on `R123`, `F114`, and `R03` because translation-provider and secret-handling behavior requires the frozen prompt-workbench contract plus the shipped persistent-state substrate.
- `F116` depends on `R123`, `F114`, `R14`, and `R03` because prompt catalogs, token analysis, grouped tags, and extra-network prompt metadata should build on the stable prompt-workbench substrate and reuse the existing host inventory baseline.
- `F117` depends on `F114`, `F113`, and `R03` because the frontend prompt-workbench shell should plug into registry-driven bootstrap and the new backend substrate rather than growing the pane façades into new monoliths.
- `F118` depends on `F117`, `F114`, and `R03` because editor actions, history/favorites, blacklist, and formatting delivery require both the mounted workbench shell and the shipped persistent-state routes.
- `F119` depends on `F115`, `F116`, `F117`, and `R03` because translation-aware editing and quick-add libraries require provider/catalog backends plus the integrated prompt-workbench shell.
- `F120` depends on `F115`, `F117`, and `R03` because AI prompt assist plus language/theme-style controls require the provider/secret-handling contract and the mounted prompt-workbench shell.
- `R124` depends on `F114`, `F115`, `F116`, `F117`, `F118`, `F119`, `F120`, and `R03` because final closure requires regression proof, live-host validation, frontend interaction evidence, and truthful capability/documentation synchronization across the full migrated prompt-workbench surface.
- `R125` depends on `R122`, `R120`, and `R03` because XYZ Plot migration should start from the accepted extensibility seams plus the existing live-host queue/post-state baseline and remain under repository-wide acceptance governance.
- `F121` depends on `R125`, `R108`, and `R03` because axis registry and value parsing must preserve the shipped SD-family runtime baseline while freezing truthful supported-axis scope before any multi-cell execution path lands.
- `F122` depends on `R125`, `F121`, `R120`, and `R03` because queue-backed XYZ sessions require a frozen axis/session contract, the accepted queue/post-state validation baseline, and repository-wide regression governance.
- `F123` depends on `F121`, `F122`, and `R03` because grid assembly and metadata output only become meaningful after axis labels and cell execution/session reconstruction already exist.
- `F124` depends on `F121`, `F122`, `F123`, `F113`, and `R03` because the frontend XYZ surface should plug into registry-driven bootstrap and completed session/grid backends instead of growing new pane-local monoliths.
- `R126` depends on `F121`, `F122`, `F123`, `F124`, and `R03` because final XYZ closure requires targeted parser/session/grid/frontend proof plus dedicated live-host queue/grid evidence.
- `R127` depends on `R122`, `R120`, and `R03` because the rebased runtime hardening chain should start from the accepted extensibility seams, existing live-host runtime baseline, and repository-wide acceptance governance.
- `F125` depends on `R127`, `F101`, and `R03` because ADetailer runtime guardrails should extend the shipped native detector/runtime path without reopening unfrozen feature semantics.
- `F126` depends on `R127`, `F107`, and `R03` because ControlNet shim hardening should preserve the accepted live-host detect/execute seam and remain regression-governed.
- `F127` depends on `R127`, `R108`, and `R03` because prompt nesting limits must preserve the shipped SD-family prompt-parity runtime contract while adding only bounded guard behavior.
- `F128` depends on `R127`, `F107`, and `R03` because tensor-range hardening should build on the accepted ControlNet detect/runtime seam and remain validated under the current live-host proof model.
- `R128` depends on `F125`, `F126`, `F127`, `F128`, and `R03` because final closure requires targeted seam-specific regression evidence plus full acceptance-gate proof after all retained runtime fixes land.
- `R139` depends on `R126`, `R138`, and `R03` because the next XYZ parity follow-up must start from the accepted queue-backed XYZ session/grid surface, the accepted preview hotfix baseline, and repository-wide bugfix acceptance governance.
- `F138` depends on `R139`, `F122`, `F123`, and `R03` because seed-policy delivery must extend the shipped session-runner and grid-metadata seams without regressing accepted XYZ queue/grid ownership.
- `R140` depends on `F138` and `R03` because final closure requires targeted seed-policy proof plus the full acceptance-gate sweep after the new backend/frontend/live-host seams land.
- `R149` depends on `R120`, `R124`, `R126`, and `R03` because the freshness hard gate extends the accepted shared live-host validation framework across multiple shipped surfaces and must remain under repository-wide acceptance governance.
- `F146` depends on `R149`, `F113`, and `R03` because runtime fingerprint exposure touches the consolidated bootstrap/capabilities seams and must enforce truthful stale-host classification without weakening the accepted host-validation framework.
- `R150` depends on `F146`, `R03`, and external host restart synchronization outside the workspace boundary because final live-host closure is only meaningful after the current ComfyUI process reloads the accepted workspace code.
- `R72` depends on `R60`, `R61`, and `R03` because Forge-Neo integrated adoption must start from the shipped ControlNet baseline and remain regression-governed.
- `F74` depends on `R72`, `F70`, and `R47` because integrated tabs UI rebuild should preserve current ControlNet semantics and existing modular shell boundaries.
- `F75` depends on `R72`, `F71`, and `R14` because dynamic module/type/model APIs require frozen contract boundaries, existing route surface baseline, and host inventory consistency.
- `F76` depends on `F74`, `F75`, and `F73` because runtime alignment must land after integrated UI and dynamic API surfaces are in place and remain consistent with the recent loader-routing fix.
- `R73` depends on `F76`, `R54`, and `R03` because final hardening requires full regression automation governance plus complete UI/API/runtime path validation.
- `R74` depends on `R72`, `F75`, and `R03` because Adetailer contract freeze must align with the integrated ControlNet baseline, dynamic catalog semantics, and existing regression governance.
- `F77` depends on `R74`, `R67`, and `R03` because reusable integrated-pack infrastructure should follow explicit contract boundaries and existing modular service seams.
- `F78` depends on `F77`, `F74`, and `F41` because Adetailer UI delivery must reuse integrated-pack primitives, stay compatible with the ControlNet integrated shell, and enforce the same Hires.fix-style collapsible framing contract.
- `F79` depends on `R74`, `F73`, and `R03` because detect-mask-refine runtime delivery must follow frozen contract semantics and remain consistent with current loader-routing/runtime governance.
- `F80` depends on `F79`, `F75`, and `F76` because Adetailer ControlNet coupling should land only after secondary runtime pipeline and integrated ControlNet API/runtime alignment are in place.
- `F81` depends on `F79`, `R14`, and `R03` because alternative-model diagnostics require real detector/runtime execution seams plus host inventory consistency and regression-safe behavior.
- `R75` depends on `F80`, `F81`, and `R54` because final hardening must validate ControlNet coupling and fallback recommendation lanes under full repository acceptance automation.
- `R76` depends on `R71`, `F36`, and `R54` because non-Lightning baseline correction must preserve the recent diffusion-loader/selector hardening, align with secondary family preset contracts, and pass full acceptance gate coverage.
- `R77` depends on `R73`, `F75`, and `R03` because the revised run-preprocessor/layout contract must start from the shipped ControlNet integrated baseline and remain regression-governed.
- `F82` depends on `R77`, `F74`, and `R54` because row-geometry and slider-width fixes must follow the frozen contract, preserve integrated UI structure, and pass the current test-governance baseline.
- `R78` depends on `F82`, `R54`, and `R03` because layout-parity hardening must validate the implemented geometry/visibility behavior under full acceptance automation.
- `R79` depends on `R78`, `R50`, and `R03` because canvas parity contract freeze should follow the immediate ControlNet row hotfix and align with existing Img2Img subtab-router semantics.
- `F83` depends on `R79`, `F67`, and `R57` because Img2Img canvas-first delivery must preserve current mode-surface behavior and hardened source/mask bridge integrity.
- `F84` depends on `F83`, `R77`, and `F75` because ControlNet canvas parity should land after Img2Img canvas primitives and maintain revised row/visibility contracts plus dynamic API semantics.
- `R80` depends on `F84`, `R54`, and `R03` because final canvas-parity hardening must cover both Img2Img and ControlNet interactions under full regression-governed acceptance.
- `R81` depends on `R80`, `F84`, and `R03` because source-canvas interaction mode freeze must build on the shipped parity baseline and preserve rollback-governed behavior.
- `F85` depends on `R81`, `F83`, and `R57` because shared source-brush controls should extend existing canvas primitives while preserving source/mask bridge integrity.
- `R82` depends on `F85`, `R54`, and `R03` because source-canvas regression hardening must pin upload-mode switching and rollback behavior under full acceptance governance.
- `R83` depends on `R77`, `R80`, and `R03` because preview/preprocessor contract freeze should align with existing run-preprocessor semantics and completed canvas rollback baseline.
- `F86` depends on `R83`, `F84`, and `F75` because dual-pane preview and layout parity delivery should build on integrated ControlNet canvas/runtime/catalog seams.
- `R84` depends on `F86`, `R54`, and `R03` because final preview/preprocessor hardening must validate source non-mutation and visibility routing under full repository gates.
- `R85` depends on `R84`, `F86`, and `R03` because run-preprocessor non-response diagnosis must start from the shipped dual-pane/source-immutability contract and remain regression-governed.
- `F87` depends on `R85`, `F85`, and `R57` because source-canvas fidelity fixes should extend the existing brush-controller and source/mask bridge baselines under the frozen runtime boundary contract.
- `F88` depends on `R85`, `R83`, and `F86` because explicit run-preprocessor feedback must preserve generated-preview gating and source immutability semantics from the existing preview contract.
- `R86` depends on `F87`, `F88`, and `R54` because final hardening must validate both canvas/runtime UX fixes and detect-route diagnostics under repository acceptance automation.

## Forge/reForge Intake Map

- `R02` owns parity-critical bridge patterns taken from Forge/reForge analysis: conditioning translation, sampler and scheduler aliasing, and evaluation of clip-skip or CFG-adjacent behaviors that affect SD-family fidelity.
- `F03` owns Forge-style shared inventory patterns: checkpoint, VAE, and text-encoder discovery plus model-family preset defaults.
- `R03` owns the regression proof that any adopted optimization does not break SD1.5, SDXL, Pony, Illustrious, or Noob behavior expectations.
- `F06` owns later opt-in optimization ports: memory and offload policy, dtype policy, scheduler expansion, extension-like helpers, and selective newer-family support.
- `F36` owns the next secondary preset-intake sweep for Forge-Neo families (`klein`, `lumina`, `zit`, `wan`, `anima`) after release-stability fixes.
- `F38` and `F39` follow `reference/comfyui-openclaw` PNG Info image-first metadata parsing direction for prompt extraction while keeping RookieUI-specific apply-target and safety checks.
- `R24` extends the same `reference/comfyui-openclaw` image-first metadata direction by enforcing positive/negative dual-prompt extraction as default behavior.
- `F44` follows `reference/stable-diffusion-webui/modules/ui_common.py` and `reference/sd-webui-forge-neo/modules/ui_common.py` `ToolButton` emoji-symbol conventions for output/action rails.
- `R49`, `F58`, and `F59` follow `reference/ComfyUI_frontend` mask-editor and painter implementation direction (`extensions/core/maskeditor.ts`, `components/painter/WidgetPainter.vue`, `composables/maskeditor/*`, `composables/painter/usePainter.ts`) plus `reference/ComfyUI/comfy_extras/nodes_mask.py` and `nodes_painter.py` semantics for mask payload compatibility.
- `R50` and `F60` follow `reference/stable-diffusion-webui/modules/ui.py` and `reference/sd-webui-forge-neo/modules/ui.py` nested Img2Img mode-tab structure under `Generation`, while reusing RookieUI's existing tab-shell primitive and mode translation contracts.
- `R51`, `R52`, `F61`, `F62`, `F63`, and `R53` follow `reference/ComfyUI_frontend` composable/module separation direction while preserving RookieUI's host-safe shell lifecycle and tab-state compatibility rules.
- `R55`, `F65`, and `F66` follow `reference/stable-diffusion-webui/modules/prompt_parser.py`, `modules/extra_networks.py`, and `modules/processing.py` as primary behavior sources for prompt semantics and conditioning composition, with `reference/ComfyUI/nodes.py` and `comfy/samplers.py` as execution-feasibility constraints.
- `R56` follows guardrail patterns from existing RookieUI runtime hardening items while enforcing explicit warning-code diagnostics for any parity downgrade paths.
- `R57`, `F67`, and `F68` follow `reference/stable-diffusion-webui/modules/ui.py` and `modules/img2img.py` mode-lane semantics plus `reference/sd-webui-forge-neo/modules/ui.py` nested-tab behavior, while preserving RookieUI's existing mask-canvas and mode-router contracts.
- `R58` follows the repository bugfix acceptance model (`Reproduce -> Pin -> Sweep`) and extends prompt/img2img parity fixtures to keep A1111 behavior regressions observable.
- `R59` follows `reference/ComfyUI-OpenClaw` and host-frontend visibility semantics where source-placeholder text must not remain visible once source image binding succeeds, and adds a frontend regression guard specifically for txt2img -> img2img handoff.
- `R60` follows `reference/stable-diffusion-webui/modules/api/models.py`, `modules/api/api.py`, and `modules/scripts.py` for `alwayson_scripts`-compatible ControlNet intake shape, while freezing RookieUI-native unit contract as canonical internal representation.
- `F69` follows `reference/ComfyUI/nodes.py` (`ControlNetLoader`, `DiffControlNetLoader`, `ControlNetApplyAdvanced`) for host-native execution, and uses Forge/Forge-Neo unit semantics only as compatibility mapping guidance.
- `F70` follows `reference/stable-diffusion-webui-forge/extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/controlnet_ui_group.py` and `reference/sd-webui-forge-neo/modules/ui.py` for A1111-style multi-unit UX direction under sidebar constraints.
- `F71` follows `reference/sd-webui-forge-neo/extensions-builtin/sd_forge_controlnet/lib_controlnet/api.py` and `global_state.py` for `model_list/module_list/control_types/detect` surface design, with optional dependency downgrade policy adapted to RookieUI host constraints.
- `R61` follows existing `R03` acceptance discipline and expands ControlNet-specific `Reproduce -> Pin -> Sweep` regression lanes to keep dual-payload/API/state-bridge regressions observable.
- `R72` follows `reference/sd-webui-forge-neo/extensions-builtin/sd_forge_controlnet/scripts/controlnet.py` and `controlnet_ui_group.py` to freeze the Forge-Neo integrated tabs target and compatibility route.
- `F74` follows Forge-Neo `controlnet_ui_group.py`, `javascript/active_units.js`, and `style.css` for integrated tab UX patterns while preserving RookieUI host-safe shell/state contracts.
- `F75` follows Forge/Forge-Neo `global_state.py` and `api.py` for dynamic `module_list/control_types/model_list` behavior and module-dispatch detect semantics.
- `F76` follows Forge-family per-unit semantic mapping from `scripts/controlnet.py` and `external_code.py`, but applies it through RookieUI host-native graph translation instead of direct patcher runtime porting.
- `R73` follows `reference/stable-diffusion-webui-reForge/extensions-builtin/sd_forge_controlnet/tests/web_api/*` test-lane direction together with repository `Reproduce -> Pin -> Sweep` acceptance governance.
- `R74` follows `reference/adetailer/adetailer/args.py`, `reference/adetailer/scripts/!adetailer.py`, and `.planning/references/260414-R74F77F78F79F80F81R75_ADETAILER_A1111_FORGE_PARITY_REFERENCE.md` for canonical Adetailer unit semantics, prompt-token behavior, skip-img2img rules, and the healthy-host UI matrix.
- `F77` follows the same 2026-04-14 Adetailer parity reference and reuses RookieUI inventory/capability seams so the feature lands on a stable request and refinement-context foundation instead of on ad-hoc script emulation.
- `F78` follows `reference/adetailer/aaaaaa/ui.py` together with `.planning/references/260414-LOCALHOST_7860_ADETAILER_UI_PARITY_REFERENCE.md` for unit tabs, group ordering, label wording, and interactive-gating parity requirements.
- `F79` follows `reference/adetailer/adetailer/mask.py`, `mediapipe.py`, `ultralytics.py`, `opts.py`, and `scripts/!adetailer.py` for detect/mask preprocessing order, prompt/seed behavior, and inpaint override handling, adapted to host-native Comfy execution seams.
- `F80` follows `reference/adetailer/controlnet_ext/common.py` and `controlnet_ext_forge.py` for ADetailer ControlNet `none` / `passthrough` / explicit-model semantics, but reuses RookieUI integrated ControlNet runtime instead of A1111 script patcher hooks.
- `F81` follows `reference/adetailer/README.md` and the 2026-04-14 parity reference for degraded-run expectations, detector/model availability surfacing, and actionable guidance behavior.
- `R75` follows existing `R03` and `R73` regression governance patterns, extending `Reproduce -> Pin -> Sweep` to Adetailer plus integrated ControlNet coupling paths while requiring healthy-host visual evidence.
- `R76` follows `reference/docs/tutorials/image/qwen/qwen-image-2512.mdx`, `reference/docs/tutorials/image/qwen/qwen-image-layered.mdx`, `reference/docs/tutorials/video/wan/wan2-2-s2v.mdx`, and official `workflow_templates` baseline values to keep non-Lightning defaults authoritative and acceleration LoRA behavior explicit opt-in.
- `R155` depends on `F72`, `R76`, and `R151` because official-template re-baselining must start from the canonical family-registry surface, the accepted non-Lightning selector/default hardening work, and the recent ERNIE intake facts before expanding to the full official template inventory.
- `F150` depends on `R155`, `F72`, and `R14` because template-backed preset exposure needs the frozen official inventory policy, canonical family-registry ownership, and the existing host model-inventory/category baseline.
- `F151` depends on `R155`, `F150`, and `R122` because official-template topology alignment should build on the frozen preset/profile matrix and the current modular workflow-builder surface rather than reopening pre-refactor monolithic graph edits.
- `R156` depends on `F150`, `F151`, and `R03` because final acceptance requires both catalog expansion and runtime-topology delivery under full repository regression governance.
- `R157` depends on `R155` and the explicit `Edit`-marked template inventory because future i2i intake must reuse the same official-template classification rules while staying deferred until the edit-template set is broader than the current `Chrono Edit 14B` baseline.
- `R158` depends on `R155`, `F151`, and `R157` because the manifest taxonomy must reuse the accepted official-template classification rules (`txt2img` vs. future `edit/i2i`) and the current family-specific topology/parameter evidence before it freezes a broader extensibility contract.
- `F152` depends on `R158` because the canonical manifest shape, derivation boundaries, and non-goals must be frozen before registry/preset/bootstrap truth is rehomed.
- `F153` depends on `F152` and `F151` because template compilation and bounded adapter ownership can only be introduced after the canonical manifest exists and the currently shipped official-template topology differences are treated as the baseline source of truth.
- `F154` depends on `F152`, `R156`, and `R154` because host-prerequisite/live-smoke derivation must reuse the already accepted truthful host-gating semantics, including external host-asset absence not being treated as a repo blocker.
- `R159` depends on `F152`, `F153`, `F154`, and `R03` because acceptance requires manifest-derived UI/preset/runtime/live-smoke truth plus the full repository SOP gate.
- `R77` and `F82` follow `reference/sd-webui-forge-neo/extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/controlnet_ui_group.py` row composition and `Run Preprocessor` (`💥`) semantics, including img2img independent-image visibility gating.
- `R79`, `F83`, and `F84` follow `reference/sd-webui-forge-neo/modules/ui.py` plus `modules_forge/forge_canvas/canvas.js|canvas.html|canvas.css` for canvas-first Img2Img and integrated ControlNet interaction direction under host-safe RookieUI constraints.
- `R80` follows existing regression governance from `R73` and repository-wide `Reproduce -> Pin -> Sweep` acceptance policy for high-risk UI/runtime seams.
- `R81` and `F85` follow Forge-Neo canvas interaction semantics from `modules_forge/forge_canvas/canvas.js|canvas.html|canvas.css` for source-present edit behavior, explicit upload-button routing, and in-surface brush-control affordances.
- `R83` and `F86` follow `reference/sd-webui-forge-neo/extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/controlnet_ui_group.py` generated-preview semantics where preprocessor output is shown in an independent preview lane controlled by `Allow Preview`.
- `R82` and `R84` follow repository-wide `Reproduce -> Pin -> Sweep` acceptance policy and extend existing ControlNet/Img2Img rollback-governed UI regression lanes.
- `R85`, `F87`, `F88`, and `R86` follow Forge-Neo canvas and ControlNet preview semantics from `modules_forge/forge_canvas/canvas.js|canvas.html|canvas.css` and `controlnet_ui_group.py`, while enforcing runtime-validation boundary rules (reference host for UI sampling vs `8188` RookieUI behavior acceptance).
- `R91`, `F91`, and `R92` explicitly override hard-coded extension-host assumptions from earlier detect intake: external detect routing is now configuration-only and no runtime code path is allowed to hard-code `127.0.0.1:7860`.
- `R93`, `F92`, and `R94` follow Forge-native detect execution architecture from `lib_controlnet/api.py`, `global_state.py`, and `controlnet_ui_group.py`: preprocessors are selected and executed in-process against host-available runtime nodes, with deterministic fallback diagnostics when host signatures/models are unavailable.
- `R97`, `F94`, and `R98` continue the same Forge-native direction by reducing heuristic fan-out (single-attempt depth/normal host probing), surfacing processor/control-model diagnostics, and hardening preview normalization against signed/non-8bit host outputs.
- `R99`, `F95`, and `R100` extend deterministic-dispatch hardening to all modules and explicitly gate AIO probing behind opt-in so host-side annotator execution remains stable without cross-family bootstrap side effects.
- `R101`, `F96`, and `R102` follow Forge/Forge-Neo ControlNet preprocessor UX direction (`controlnet_ui_group.py`) by narrowing preprocessor options per selected Control Type and ensuring selected preprocessor variants bias host annotator dispatch order with explicit status diagnostics.
- `R62` follows `reference/ComfyUI/nodes.py` graph-construction primitives plus existing RookieUI workflow translator contracts to split oversized builder seams without semantic drift.
- `R63` follows `reference/ComfyUI-OpenClaw` and `reference/ComfyUI_frontend` module ownership direction by centralizing asset-revision ownership instead of scattered per-import querystrings.
- `R64` follows existing shared-coercion direction (`R40` lineage) and A1111 prompt/metadata normalization expectations from `reference/stable-diffusion-webui/modules/prompt_parser.py` and related metadata ingest behavior.
- `R65` follows gradual typing patterns compatible with current ComfyUI frontend extension loading, prioritizing incremental TS safety over immediate bundler/runtime replacement.
- `R66` follows repository SOP governance (`tests/TEST_SOP.md` and `tests/E2E_TESTING_SOP.md`) and extends parity proof to host-embedded execution lanes.
- `R119`, `F107`, `F108`, `F109`, and `R120` follow the same host-embedded validation direction as `R114/R115`, but expand it to shipped non-prompt auxiliary surfaces; `reference/stable-diffusion-webui/modules/infotext_utils.py`, `reference/stable-diffusion-webui/scripts/postprocessing_upscale.py`, `reference/adetailer/adetailer/args.py`, and existing ControlNet/ADetailer internal reference memos remain the primary behavior sources.
- `R121`, `F110`, `F111`, `F112`, `F113`, and `R122` follow the earlier architecture-modernization direction from `reference/ComfyUI/nodes.py` and existing RookieUI modularization work, plus the service-separation patterns visible in `reference/stable-diffusion-webui-forge/extensions-builtin/sd_forge_controlnet/lib_controlnet/{api.py,external_code.py,global_state.py,controlnet_ui/controlnet_ui_group.py}` and `reference/adetailer/adetailer/{args.py,mask.py,mediapipe.py,ultralytics.py}` with `reference/adetailer/controlnet_ext/{common.py,controlnet_ext_forge.py}`. The adaptation rule is to keep RookieUI host-native execution and public route contracts stable while splitting monolithic services into feature builders, vertical service modules, and a lightweight integrated-feature registry.
- `R123`, `F114`, `F115`, `F116`, `F117`, `F118`, `F119`, `F120`, and `R124` follow `reference/sd-webui-prompt-all-in-one/{README.MD,scripts/on_app_started.py,scripts/physton_prompt/storage.py,scripts/physton_prompt/history.py,scripts/physton_prompt/get_token_counter.py,scripts/physton_prompt/get_group_tags.py,scripts/physton_prompt/get_translate_apis.py,scripts/physton_prompt/get_extra_networks.py,scripts/physton_prompt/gen_openai.py,src/src/App.vue,src/src/components/phystonPrompt.vue,src/src/components/promptFormat.vue}` as the primary migration source. The adaptation rule is to migrate the feature family into a RookieUI-native prompt-workbench architecture: no Gradio textarea hijack, no parallel prompt runtime, no forced in-app package installer, and explicit truthfulness-first handling for translation/AI providers.
- `R125`, `F121`, `F122`, `F123`, `F124`, and `R126` follow `reference/stable-diffusion-webui/scripts/xyz_grid.py`, `modules/images.py`, `modules/processing.py`, `modules/shared_options.py`, and `extensions-builtin/hypertile/scripts/hypertile_script.py` as the primary migration source. The adaptation rule is to migrate the feature family into a RookieUI-native XYZ architecture: no A1111 script-slot port, no direct mutable-processing-object patch path, no automatic import of third-party extension axes, and explicit truthfulness-first axis support over a queue-backed sweep/session runner plus RookieUI-owned grid asset delivery.
- `R67` follows the same modularization direction as `R51-R53/F61-F63`, continuing shell/pane ownership reduction with explicit size-budget controls.
- `F72` follows `reference/stable-diffusion-webui-forge/modules_forge/presets.py` and `reference/sd-webui-forge-neo/modules_forge/presets.py` family-aware capability expression patterns while keeping RookieUI contracts explicit and host-safe.
- `R155`, `F150`, `F151`, and `R157` follow the official `reference/workflow_templates/*.json` graphs as the primary source of truth for non-SD template coverage, family-specific parameter exposure, topology alignment, and `Edit` vs. T2I classification. Fixed template internals must not be upgraded into generic user-facing controls without stronger official evidence.
- `R158`, `F152`, `F153`, `F154`, and `R159` must preserve the same official-template source-of-truth rule while reducing future intake churn; manifest derivation is not allowed to weaken topology fidelity or truthful host-prerequisite reporting.
- `R160` depends on `R157`, `F153`, and `R159` because the img2img/edit safety freeze must reuse the already accepted template classification rules, the manifest-backed runtime ownership seams, and the truthful host/prerequisite reporting baseline.
- `F155` depends on `R160` and `F152` because hiding unaligned i2i presets safely requires a frozen flow-kind visibility rule plus manifest-derived preset metadata rather than ad-hoc frontend filtering.
- `F156` depends on `R160`, `F153`, and `R159` because official non-SD img2img delivery must extend the accepted bounded-adapter runtime architecture and preserve the same truthful runtime/host semantics already closed for txt2img.
- `F157` depends on `R160`, `F156`, and `R157` because the future `Edit` surface must build on the frozen edit-template ownership rule and on a real non-SD image-input runtime seam instead of the legacy generic i2i graph.
- `R161` depends on `F155`, `F156`, `F157`, and `R03` because acceptance requires both user-safety filtering and the full repository validation sweep after the new flow split lands.
- `R162` depends on `F156`, `F157`, and `R159` because the new inline-LoRA contract must extend the accepted official txt2img/edit runtime seams and preserve the manifest-backed truthful host-prerequisite baseline rather than reopening generic SD-style loader assumptions.
- `F158` depends on `R162`, `F153`, and `R161` because the model-only inline LoRA chain must reuse the bounded non-SD adapter architecture, preserve the current template-owned LoRA ownership rules, and remain compatible with the already accepted img2img/edit split.
- `R163` depends on `F158` and `R03` because acceptance requires targeted ordering/drift regressions plus the full repository SOP gate and truthful host-evidence capture where current assets permit execution.

## Phase 81 - Windows Pre-Push E2E Harness Port Guardrail

Current working references for Phase 81 Windows pre-push E2E guardrail:
- `.planning/references/260419-R164F159R165_WINDOWS_PRE_PUSH_E2E_PORT_GUARDRAIL_REFERENCE.md`
- `.planning/plans/260419-R164F159R165_WINDOWS_PRE_PUSH_E2E_PORT_GUARDRAIL_PLAN.md`

Planned implementation note (2026-04-19): the active bug is a push-blocking repository-tooling regression on Windows Git-Bash rather than a host-runtime/product feature. The failing path is `.githooks/pre-push -> scripts/pre_push_checks.sh -> npm test -> playwright test`, where the harness still assumes the default `4173` port and an arbitrary `python` command. This phase closes that gap by mirroring the accepted Windows full-gate behavior on the actual pre-push path: pin the harness Python to the repo-local `.venv`, auto-select a bindable localhost port, add targeted regression coverage for the port-selection helper, and prove the real Git-Bash pre-push gate passes while `4173` is occupied.

| Index | Item | Priority | Status | Title | Branch | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 296 | R164 | P1 | Completed (2026-04-19) | Windows Pre-Push E2E Harness Contract Freeze | `main` (documented hotfix override) | Frozen as a release-branch tooling hotfix: Git-Bash `pre-push` must not rely on the default `4173` bind or arbitrary global Python when launching the Playwright harness. |
| 297 | F159 | P1 | Completed (2026-04-19) | Shared Bindable-Port Helper and Pre-Push Harness Delivery | `main` (documented hotfix override) | Added a shared Python helper to resolve the first bindable localhost port and wired Git-Bash `pre-push` to export `ROOKIEUI_E2E_PYTHON` from the project `.venv` plus `ROOKIEUI_E2E_PORT` from that helper before `npm test`. |
| 298 | R165 | P1 | Completed (2026-04-19) | Windows Pre-Push E2E Regression and Acceptance Closure | `main` (documented hotfix override) | Closed with pre-fix failing regression evidence, post-fix targeted helper tests, and a full real Git-Bash `scripts/pre_push_checks.sh` pass while `127.0.0.1:4173` was intentionally occupied. |

Stage sequencing:

- Stage 1: `R164` freeze the exact failing Git-Bash pre-push path and the accepted parity target with the Windows full-gate wrapper.
- Stage 2: `F159` land a shared bindable-port helper plus pre-push environment pinning for `ROOKIEUI_E2E_PYTHON` / `ROOKIEUI_E2E_PORT`.
- Stage 3: `R165` close with targeted regression coverage and a real occupied-port Git-Bash full-gate proof.

Dependencies and rationale:

- `R164` depends on `R03` because this is a bugfix chain governed by the repository-wide `Reproduce -> Pin -> Sweep` rule rather than a feature backlog expansion.
- `F159` depends on `R164` because the shell-path fix must follow a frozen statement of the exact push failure mode and accepted Windows parity target.
- `R165` depends on `F159` and `R03` because acceptance requires both helper-level regression proof and a final full repository gate using the same Git-Bash pre-push entrypoint that originally failed.
- `R166` depends on `R03` because this is a regression-hardening proof chain for an already shipped SD-family behavior and acceptance still follows the repository-wide SOP gate.
- `F160` depends on `R166` because the new regression must pin the exact multi-inline `img2img` LoRA chaining contract before future runtime refactors.
- `R167` depends on `F160` and `R03` because closure requires both the dedicated regression proof and the full repository validation sweep.

## Phase 82 - SD Img2Img Inline LoRA Regression Hardening

Current working references for Phase 82 SD img2img inline LoRA regression:
- `.planning/plans/260420-R166F160R167_SD_IMG2IMG_INLINE_LORA_REGRESSION_HARDENING_PLAN.md`

Planned implementation note (2026-04-20): SD-family `txt2img` already had explicit multi-inline LoRA regression coverage, but `img2img` relied on the same shared `_resolve_model_sources()` seam without its own dedicated test. This phase closes that evidence gap by pinning the `img2img` behavior directly: multiple inline `<lora:...>` activations plus a selected UI LoRA must serially chain into `LoraLoader` nodes on SD-family requests, and the full repository SOP gate must stay green afterward.

| Index | Item | Priority | Status | Title | Branch | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 299 | R166 | P2 | Completed (2026-04-20) | SD Img2Img Inline LoRA Regression Scope Freeze | `main` | Frozen as a release-branch test-hardening follow-up on the active branch: the goal is to add direct regression proof for existing SD-family `img2img` multi-inline LoRA chaining rather than change runtime behavior. |
| 300 | F160 | P2 | Completed (2026-04-20) | SD Img2Img Multi-Inline LoRA Regression Delivery | `main` | Added a dedicated `img2img` regression asserting two prompt inline LoRAs plus one selected LoRA chain into serial `LoraLoader` nodes and feed the final SD sampler input. |
| 301 | R167 | P2 | Completed (2026-04-20) | SD Img2Img Inline LoRA Regression Acceptance Closure | `main` | Closed with direct targeted regression execution and a green full repository SOP gate on Windows PowerShell using the repo-local `.venv`. |

Stage sequencing:

- Stage 1: `R166` freeze the exact SD-family `img2img` multi-inline LoRA behavior that already exists in the shared builder seam.
- Stage 2: `F160` land a dedicated regression that proves chaining order and final sampler wiring.
- Stage 3: `R167` close with targeted proof and the full repository validation sweep.

## Phase 83 - XYZ Plot Control-Surface Polish

Current working references for Phase 83 XYZ Plot UI polish:
- `.planning/plans/260420-R168F161R169_XYZ_PLOT_CONTROL_SURFACE_POLISH_PLAN.md`

Planned implementation note (2026-04-20): this is a focused frontend parity/polish follow-up on the active branch. The target is the shipped XYZ Plot control surface in both txt2img and img2img: remove the redundant explanatory note, restore a solid section border, assign distinct semantic colors to `Estimate` and `Refresh`, and swap the positions of `Refresh` and `Run XYZ Plot`. Acceptance requires direct shell regression coverage, Playwright parity evidence, and the full repository SOP gate.

| Index | Item | Priority | Status | Title | Branch | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 302 | R168 | P2 | Completed (2026-04-20) | XYZ Plot Control-Surface Parity Freeze | `main` | Frozen as a release-branch frontend polish follow-up on the active branch; the accepted parity checklist is: no helper note, solid border, green Estimate, blue Refresh, and action order `Estimate -> Refresh -> Run XYZ Plot -> Cancel`. |
| 303 | F161 | P2 | Completed (2026-04-20) | XYZ Plot Shell and CSS Polish Delivery | `main` | Removed the redundant note from the shell, added explicit estimate/refresh action modifiers, reordered the action row, restored the section border to solid, and refreshed the shipped frontend asset revision token. |
| 304 | R169 | P2 | Completed (2026-04-20) | XYZ Plot UI Regression and Acceptance Closure | `main` | Closed with a red-to-green shell regression, a targeted Playwright XYZ Plot spec, and a green full SOP gate on Windows PowerShell using the repo-local `.venv`. |

Stage sequencing:

- Stage 1: `R168` freeze the exact component-level parity checklist for the XYZ Plot control surface.
- Stage 2: `F161` deliver the shell/CSS updates and refresh the shipped frontend asset revision token.
- Stage 3: `R169` close with targeted shell + Playwright parity proof and the full repository validation sweep.

## Phase 84 - Official ImageEdit Re-baseline and Scope Freeze

Execution policy: implement on `dev` because this chain redefines public flow classification, `img2img` payload semantics, and future runtime/UI ownership for all official image-edit models. The implementation must explicitly override the older phase-77/79 assumption that official edit models live on a separate single-reference `Edit` surface.

Current working references for Phase 84 image-edit re-baseline:
- `.planning/references/260423-R170_OFFICIAL_IMAGEEDIT_REBASELINE_REFERENCE.md`
- `.planning/references/260423-R171_CHRONO_WAN_TEMPORAL_EDIT_DEFER_REFERENCE.md`
- `.planning/plans/260423-R170_OFFICIAL_IMAGEEDIT_REBASELINE_AND_INVENTORY_FREEZE_PLAN.md`
- `.planning/plans/260423-R171_CHRONO_WAN_TEMPORAL_EDIT_DEFER_PLAN.md`
- `.planning/implementation_records/260423-R170_OFFICIAL_IMAGEEDIT_REBASELINE_AND_INVENTORY_FREEZE_IMPLEMENTATION_RECORD.md`
- `.planning/implementation_records/260423-R171_CHRONO_WAN_TEMPORAL_EDIT_DEFER_IMPLEMENTATION_RECORD.md`
- `reference/workflow_templates/imageEdit/Chrono Edit 14B.json`
- `reference/workflow_templates/imageEdit/Firered image edit.json`
- `reference/workflow_templates/imageEdit/Flux.1 Kontext Dev .json`
- `reference/workflow_templates/imageEdit/Flux.2 image edit.json`
- `reference/workflow_templates/imageEdit/Flux.2 Klein 9b KV image edit.json`
- `reference/workflow_templates/imageEdit/Longcat image edit.json`
- `reference/workflow_templates/imageEdit/Qwen-image edit.json`
- `reference/workflow_templates/imageEdit/Qwen-image edit-multi-lora.json`
- `reference/ComfyUI/comfy_extras/nodes_qwen.py`
- `reference/ComfyUI/comfy_extras/nodes_edit_model.py`
- `reference/ComfyUI/comfy_extras/nodes_flux.py`
- `reference/ComfyUI/comfy_extras/nodes_rope.py`
- `reference/ComfyUI/comfy_extras/nodes_wan.py`
- `reference/ComfyUI-EditUtils/README.md`
- `reference/ComfyUI-EditUtils/nodes.py`
- `reference/ComfyUI-EditUtils/nodes_doc.md`

Implementation note (2026-04-22): authoritative reference review now shows that RookieUI's current `qwen_image_edit` delivery is only a first-wave subset. All reviewed official image-edit workflows are `img2img`-owned image-input graphs, none of the reviewed templates require a user mask, and multiple references are already first-class in both the official host graph set and the new `ComfyUI-EditUtils` reference project. This phase therefore re-baselines future work around `img2img`-owned image-edit subtypes, bounded family adapters, and truthful first-wave deferral of temporal/video-like edit graphs (`Chrono Edit 14B` / `WanImageToVideo` lineage).

| Index | Item | Priority | Status | Title | Branch | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 305 | R170 | P0 | Completed (2026-04-23) | Official ImageEdit Contract Re-baseline and Inventory Freeze | `dev` | Closed with a dated reference synthesis, plan, implementation record, and command log that supersede the old dedicated-`Edit` / single-reference planning baseline and freeze the authoritative `img2img` / no-mask / multi-reference-first rules plus real adapter-family grouping. |
| 306 | R171 | P0 | Completed (2026-04-23) | Chrono/Wan Temporal Edit Scope Split and Defer Contract | `dev` | Closed with a dated defer reference, plan, implementation record, and command log that classify `Chrono Edit 14B` and Wan-style temporal/video-like image-edit graphs as out of scope for the first-wave static image-edit rollout and later acceptance claim. |

Stage sequencing:

- Stage 1: `R170` freeze authoritative template inventory, replace the old dedicated-`Edit` assumption, and classify the adapter families that later phases must implement.
- Stage 2: `R171` freeze the temporal/video-like non-goal so the first-wave static-image rollout has a hard scope boundary before any adapter work starts.

Dependencies and rationale:

- `R170` depends on `R157`, `R160`, and `R161` because the new chain must consciously supersede the earlier backlog-freeze and edit-surface split assumptions rather than silently drifting away from accepted history.
- `R171` depends on `R170` because temporal/video-like deferral has to be resolved against the same authoritative inventory freeze rather than by ad-hoc omission during runtime work.
- `R170` follows `reference/workflow_templates/imageEdit/*.json` as the primary source of truth for shipped official image-edit topology, while `reference/ComfyUI-EditUtils` is a secondary implementation reference for encoder/config ergonomics rather than a behavior source that can override official graphs.

## Phase 85 - Img2Img ImageEdit Contract and Manifest Expansion

Execution policy: implement on `dev` because this phase changes canonical request normalization, manifest-derived profile truth, route payload semantics, and frontend bootstrap metadata for the public image-input surface.

Current working references for Phase 85 image-edit contract expansion:
- `.planning/plans/260423-F162_IMG2IMG_OWNED_IMAGEEDIT_REQUEST_ROUTE_CONTRACT_FOUNDATION_PLAN.md`
- `.planning/implementation_records/260423-F162_IMG2IMG_OWNED_IMAGEEDIT_REQUEST_ROUTE_CONTRACT_FOUNDATION_IMPLEMENTATION_RECORD.md`
- `.planning/plans/260423-F163_OFFICIAL_IMAGEEDIT_MANIFEST_PROFILE_MATRIX_EXPANSION_PLAN.md`
- `.planning/implementation_records/260423-F163_OFFICIAL_IMAGEEDIT_MANIFEST_PROFILE_MATRIX_EXPANSION_IMPLEMENTATION_RECORD.md`
- `rookieui/contracts/generation.py`
- `rookieui/services/img2img.py`
- `rookieui/contracts/family_template_manifest.py`
- `rookieui/contracts/model_family_registry.py`
- `web/sidebar_tabs/rookieui_img2img_pane.js`
- `web/rookieui_feature_registry.js`
- `reference/docs/zh/built-in-nodes/TextEncodeQwenImageEditPlus.mdx`
- `reference/docs/zh/built-in-nodes/ReferenceLatent.mdx`
- `reference/docs/zh/built-in-nodes/FluxKontextMultiReferenceLatentMethod.mdx`
- `reference/ComfyUI-EditUtils/README.md`
- `reference/ComfyUI-EditUtils/nodes_doc.md`

Implementation note (2026-04-22): before more adapters can ship, the canonical image-input contract must stop assuming one source image plus an optional mask. RookieUI needs an `img2img`-owned image-edit subtype with ordered reference assets, explicit main-reference ownership, per-profile supported-reference-count metadata, and truthful first-wave limits. The first implementation target should preserve an extensible list contract internally while the initial UI likely caps direct attachment count to a bounded number that matches strong reference evidence (`3` direct images from `TextEncodeQwenImageEditPlus` / `ComfyUI-EditUtils`, with chained-expansion left to later work).

| Index | Item | Priority | Status | Title | Branch | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 307 | F162 | P0 | Completed (2026-04-23) | Img2Img-Owned ImageEdit Request and Route Contract Foundation | `dev` | Closed with canonical `img2img`-owned image-edit request normalization, ordered `reference_images` plus `main_reference_index`, legacy single-image compatibility, updated `img2img-<profile>` translation naming, targeted backend regressions, and a green full repository gate; the optional host-embedded lane stayed invalid because the active host fingerprint was stale. |
| 308 | F163 | P0 | Completed (2026-04-23) | Official ImageEdit Manifest/Profile Matrix Expansion | `dev` | Closed with manifest-backed image-edit metadata for `qwen_image_edit` (`image_edit_profile`, canonical `img2img` request surface, single-reference contract, encoder family, template-owned LoRA chain mode) propagated through presets, capabilities normalization, frontend bootstrap fallbacks, targeted contract tests, and a green full repository gate; public `available_surface_flows` intentionally remain unchanged until `F168`. |

Stage sequencing:

- Stage 1: `F162` land the canonical request/route/data-model expansion for ordered reference images and no-mask image-edit semantics.
- Stage 2: `F163` move the public profile matrix, bootstrap payloads, and host-prerequisite truth onto manifest-backed image-edit metadata.

Dependencies and rationale:

- `F162` depends on `R170` because request/route expansion must follow a frozen statement of what counts as image-edit and which first-wave template families are in scope.
- `F163` depends on `F152`, `F153`, `F154`, and `F162` because manifest growth must reuse the accepted extensibility architecture rather than reopen hand-maintained parallel registries.

## Phase 86 - Shared ImageEdit Conditioning and Qwen-Family Runtime Delivery

Execution policy: implement on `dev` because this phase changes runtime graph compilation, host-node selection, and template-owned LoRA ordering for official Qwen-family image-edit models.

Current working references for Phase 86 Qwen-family delivery:
- `rookieui/services/workflow_builders/non_sd_templates.py`
- `reference/workflow_templates/imageEdit/Qwen-image edit.json`
- `reference/workflow_templates/imageEdit/Qwen-image edit-multi-lora.json`
- `reference/workflow_templates/imageEdit/Firered image edit.json`
- `reference/ComfyUI/comfy_extras/nodes_qwen.py`
- `reference/docs/zh/built-in-nodes/TextEncodeQwenImageEditPlus.mdx`
- `reference/ComfyUI-EditUtils/README.md`
- `reference/ComfyUI-EditUtils/nodes.py`
- `reference/ComfyUI-EditUtils/nodes_doc.md`
- `.planning/references/260423-F164_MULTI_REFERENCE_IMAGEEDIT_CONDITIONING_REFERENCE.md`
- `.planning/plans/260423-F164_SHARED_MULTI_REFERENCE_IMAGEEDIT_CONDITIONING_FOUNDATION_PLAN.md`
- `.planning/implementation_records/260423-F164_SHARED_MULTI_REFERENCE_IMAGEEDIT_CONDITIONING_FOUNDATION_IMPLEMENTATION_RECORD.md`

Implementation note (2026-04-22): the current `qwen_image_edit` builder is intentionally narrow: one reference image, one template-owned LoRA, one encoder class. The official reference set already proves a broader family: base Qwen edit, chained template-owned multi-LoRA variants, and Qwen-Edit-Plus style encoding paths (`Firered`) with richer multi-image conditioning. RookieUI should therefore add a shared image-edit conditioning foundation first, then land bounded Qwen-family adapters on top instead of copy-pasting per-template builders.

| Index | Item | Priority | Status | Title | Branch | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 309 | F164 | P1 | Completed (2026-04-23) | Shared Multi-Reference ImageEdit Conditioning Foundation | `dev` | Closed with a dedicated `image_edit_foundation` builder module for ordered reference bundles, optional scaling ownership, reusable VAE latent creation, ordered `ReferenceLatent` chaining, Flux multi-reference method wrapping, manifest-backed direct-reference limit enforcement, qwen-edit builder migration, targeted backend regressions, and a green full repository gate. |
| 310 | F165 | P1 | Completed (2026-04-23) | Qwen/Qwen+ ImageEdit Runtime Adapter Expansion | `dev` | Closed with manifest-backed `qwen_image_edit_multi_lora`, `firered_image_edit`, and `firered_image_edit_lightning` profiles; a generalized Qwen/Qwen+ edit builder that now covers triple template-owned LoRA chaining and `TextEncodeQwenImageEditPlus`; truthful FireRed base-vs-lightning prerequisite handling; synchronized frontend bootstrap fallbacks; targeted backend/frontend/Playwright regressions; and a green full repository gate. |

Stage sequencing:

- Stage 1: `F164` land the reusable image-edit builder helpers shared by later family adapters.
- Stage 2: `F165` deliver bounded Qwen-family adapters on the new helper seam, including truthful handling of template-owned multi-LoRA chains and richer encoder variants.

Dependencies and rationale:

- `F164` depends on `F162` and `F163` because shared builder seams need the new ordered-reference contract and manifest-declared adapter/parameter metadata.
- `F165` depends on `F164`, `R162`, and `F158` because Qwen-family edit expansion must preserve the accepted non-SD model-only LoRA ordering rule while broadening template-owned chain depth.

## Phase 87 - Flux/Kontext/Klein/Longcat ImageEdit Runtime Delivery

Execution policy: implement on `dev` because this phase introduces new multi-reference runtime helpers, expands bounded family-adapter coverage, and changes truthful host-prerequisite reporting for official Flux-family image-edit profiles.

Current working references for Phase 87 Flux-family image-edit delivery:
- `reference/workflow_templates/imageEdit/Flux.1 Kontext Dev .json`
- `reference/workflow_templates/imageEdit/Flux.2 image edit.json`
- `reference/workflow_templates/imageEdit/Flux.2 Klein 9b KV image edit.json`
- `reference/workflow_templates/imageEdit/Longcat image edit.json`
- `reference/ComfyUI/comfy_extras/nodes_edit_model.py`
- `reference/ComfyUI/comfy_extras/nodes_flux.py`
- `reference/docs/zh/built-in-nodes/ReferenceLatent.mdx`
- `reference/docs/zh/built-in-nodes/FluxKontextMultiReferenceLatentMethod.mdx`
- `reference/ComfyUI-EditUtils/README.md`
- `reference/ComfyUI-EditUtils/nodes.py`

Implementation note (2026-04-22): the official Flux-family image-edit templates are clearly multi-reference-capable but they are not one topology. `Flux.2 image edit` centers on `ReferenceLatent` plus advanced sampler wiring; `Flux.2 Klein 9b KV image edit` adds chained references and KV-cache ownership; `Flux.1 Kontext Dev` introduces stitched/context-conditioned multi-image semantics; `Longcat image edit` reuses Qwen-style encoding for prompt conditioning but still routes image ownership through Flux-family latent/reference nodes. RookieUI should implement these through one shared foundation plus bounded profile adapters, not per-template ad-hoc graph sprawl.

| Index | Item | Priority | Status | Title | Branch | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 311 | F166 | P1 | Completed (2026-04-23) | Flux/Kontext/Klein/Longcat Multi-Reference Foundation | `dev` | Closed with a shared `image_edit_foundation` expansion covering chained `ImageStitch` plus `FluxKontextImageScale`, mirrored `ReferenceLatent` helpers, Flux reference-method branch wiring, `FluxKVCache`, structured Flux2 sampler bundles, dedicated backend regression tests, and a green full repository gate without yet exposing new public adapters. |
| 312 | F167 | P1 | Completed (2026-04-23) | Flux/Kontext/Klein/Longcat ImageEdit Adapter Delivery | `dev` | Closed with four shipped first-wave edit adapters (`flux_kontext_dev_edit`, `flux2_image_edit`, `klein_9b_kv_image_edit`, `longcat_image_edit`), manifest/fallback contract sync, targeted backend/frontend/E2E regressions, and a green full repository gate on `dev`. |

Stage sequencing:

- Stage 1: `F166` land the shared multi-reference latent/context/KV helpers required by the Flux-family templates.
- Stage 2: `F167` exposed the first-wave official Flux/Kontext/Klein/Longcat image-edit profiles on top of the new helper seam and closed on 2026-04-23 with a green full repository gate.

Dependencies and rationale:

- `F166` depends on `F164` because the Flux-family helper layer should reuse the same ordered-reference asset / latent normalization foundation rather than fork its own pre-processing contract.
- `F167` depends on `F166` and `F163` because profile delivery must stay manifest-backed and must not hard-code selector/prerequisite truth outside the accepted extensibility system.

## Phase 88 - Img2Img Surface Integration and Test-Matrix Expansion

Execution policy: implement on `dev` because this phase changes the visible `img2img` user experience, removes now-invalid mask-first edit assumptions, formalizes first-wave exclusions, and extends the regression/live-smoke harness for image-edit behavior.

Current working references for Phase 88 UI + temporal split:
- `web/sidebar_tabs/rookieui_img2img_pane.js`
- `web/rookieui_sidebar_shell.js`
- `web/rookieui_feature_registry.js`
- `rookieui/services/smoke_profiles.py`
- `reference/workflow_templates/imageEdit/Chrono Edit 14B.json`
- `reference/ComfyUI/comfy_extras/nodes_rope.py`
- `reference/ComfyUI/comfy_extras/nodes_wan.py`
- `reference/ComfyUI-EditUtils/README.md`

Implementation note (2026-04-22): user-facing integration should happen only after the runtime families are real and after the temporal/video-like defer contract is already frozen. Once the static-image adapters exist, the `img2img` pane must stop presenting edit models as a separate flow, stop implying mask ownership on that branch, and expose ordered multi-reference inputs as first-class state.

| Index | Item | Priority | Status | Title | Branch | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 313 | F168 | P1 | Completed (2026-04-23) | Img2Img ImageEdit UI Surface Integration | `dev` | Closed with canonical `img2img` surface exposure for first-wave image-edit profiles, removal of the visible `Edit` mode split, profile-aware mask suppression plus mode gating, ordered multi-reference UI state and payload serialization, synchronized fallback metadata/tests, and a green targeted backend/frontend/Playwright sweep. |
| 314 | F169 | P1 | Done | ImageEdit Smoke/Fixture/Test Matrix Foundation | `dev` | Closed with ordered-reference live-smoke coverage, multi-LoRA/template-depth assertions, frontend multi-reference payload proof, and a green repository SOP gate on `dev`. |

Stage sequencing:

- Stage 1: `F168` reworked the `img2img` UI / bootstrap / request-building surface around the accepted image-edit contract and closed on 2026-04-23.
- Stage 2: `F169` expanded regression and live-smoke infrastructure so later acceptance is grounded in direct image-edit evidence rather than inference from txt2img or legacy edit behavior.
- Outcome (2026-04-23): `F169` closed on `dev` after targeted backend/frontend/Playwright proof plus a green `powershell -File scripts/run_full_tests_windows.ps1` sweep.

Dependencies and rationale:

- `F168` depends on `F165` and `F167` because the public UI surface must not expose profile families that still lack real runtime builders.
- `F169` depends on `F163`, `F165`, `F167`, `F168`, and `R166-R169` lineage because image-edit test expansion must follow both the manifest-derived profile matrix and the shipped `img2img` interaction contract rather than pinning stale pre-integration behavior.

## Phase 89 - Official ImageEdit Regression and Live-Host Acceptance Closure

Execution policy: close on `dev` only after the full image-edit chain is implemented, the repository SOP gate passes, and truthful live-host evidence proves the asset-ready subset without overstating host support.

Current working references for Phase 89 acceptance closure:
- `tests/TEST_SOP.md`
- `tests/E2E_TESTING_NOTICE.md`
- `tests/E2E_TESTING_SOP.md`
- `rookieui/services/smoke_profiles.py`
- `rookieui/services/model_inventory.py`
- `rookieui/contracts/family_template_manifest.py`

Implementation note (2026-04-22): image-edit acceptance must be stricter than the earlier single-profile proof. Closure requires targeted regressions for ordered reference-image payloads, manifest-derived per-profile visibility, multi-LoRA template ownership, UI interaction proof for reference-image state, and restarted-host catalog / execute evidence for whichever first-wave image-edit families are actually asset-ready on the validation host. Missing official models, LoRAs, VAEs, text encoders, or custom nodes remain truthful host prerequisites rather than repo blockers.

| Index | Item | Priority | Status | Title | Branch | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 315 | R172 | P0 | Completed (2026-04-23) | Official ImageEdit Regression and Live-Host Acceptance Closure | `dev` | Closed with execute-safe image-edit workflow wiring, targeted regression proof, workspace-safe restarted-host report/execute evidence for `klein_9b_kv_image_edit` + `longcat_image_edit`, truthful prerequisite classification for the current Qwen edit LoRA-label drift on the validation host, and a green full repository SOP gate. |

Stage sequencing:

- Stage 1: `R172` run targeted regression proof, restarted-host catalog / execute evidence for the asset-ready subset, and the final repository SOP gate before any `main` promotion.

Dependencies and rationale:

- `R172` depends on `F169` and `R03` because acceptance requires both direct image-edit regression proof and the repository-wide `Reproduce -> Pin -> Sweep` validation model.
- `R172` also depends on `R171` because first-wave closure must explicitly exclude deferred temporal/video edit graphs rather than silently omitting them from the acceptance claim.

Additional existing global dependency notes:

- Outcome (2026-04-23): `R172` closed on `dev` after a workspace-safe `reference/ComfyUI` host replaced the stale external deployment for live validation. The accepted runtime fix was limited to host-compatible execute drift inside the shipped image-edit builders: `RookieUILoadAssetImage` now emits the declared `asset_handle` input name, and `Flux2Scheduler.steps` now stays scalar instead of being serialized as a bogus node reference. With those fixes in place, report + execute evidence passed for the truthful asset-ready subset `klein_9b_kv_image_edit,longcat_image_edit`; `qwen_image_edit` and `qwen_image_edit_multi_lora` remained validation-host prerequisites because the host resolved only `加速與功能性\\Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors`, not the manifest's official template label.

## Phase 90 - Qwen Image-Edit Multi-LoRA Visible Preset Retirement

Execution policy: implement on `dev` because this change alters shipped `img2img` UI behavior while intentionally preserving backend profile compatibility. The accepted path must keep the manifest/profile/runtime/live-smoke chain intact and only retire the separate visible preset exposure.

Current working references for Phase 90 visible-preset retirement:

- Process override (2026-04-23): the repository `.gitignore` prohibits adding new `.planning/*` files, so the phase-90 plan/verification summary is embedded directly in this roadmap section instead of new tracked plan/record artifacts.

| Roadmap ID | Item | Priority | Status | Title | Branch | Description |
| --- | --- | --- | --- | --- | --- | --- |
| 316 | F170 | P1 | Completed (2026-04-23) | Qwen Image-Edit Multi-LoRA Visible Preset Retirement | `dev` | Closed with frontend-only `img2img` preset filtering that retires the separate visible `Qwen-Image Edit Multi-LoRA` dropdown entry, preserves hidden-profile compatibility for existing backend/runtime seams, refreshes the shipped frontend asset fingerprint, updates README truthfulness, and passes the full repository SOP gate on `dev`. |

- Outcome (2026-04-23): `F170` closed on `dev` after `web/sidebar_tabs/rookieui_img2img_pane.js` began filtering `qwen_image_edit_multi_lora` out of the visible `Img2Img` preset list while still appending a hidden selected option when that exact compatibility profile is already active. Frontend unit and Playwright checks were updated to pin the retired visible preset contract, README now documents the canonical `Qwen-Image Edit` plus prompt-inline multi-`<lora:...>` path, the shipped frontend asset fingerprint was refreshed, and the final `powershell -File scripts/run_full_tests_windows.ps1` gate passed green.
- Embedded plan summary:
  - Scope in: retire the visible `Img2Img` dropdown entry only; preserve backend/runtime/profile compatibility.
  - Scope out: do not remove the `qwen_image_edit_multi_lora` manifest/profile/runtime seams.
  - Design: frontend-only preset filtering plus hidden selected-option preservation for compatibility-driven apply-back/runtime paths.
  - Security: no new trust boundary; change only reduces duplicate UI exposure.
  - Failure/rollback: revert the frontend-only filter if compatibility apply-back drifts or the visible preset must be restored.
  - Verification: targeted Vitest + Playwright proof, then full `tests/TEST_SOP.md` gate via `powershell -File scripts/run_full_tests_windows.ps1`.

- `R68` uses `reference/ComfyUI_frontend` build/runtime constraints as feasibility baseline for Vite-path evaluation, with rollback-first decision governance.
- `R69` uses `reference/ComfyUI_frontend` Vue composition/runtime patterns as host-adapter feasibility baseline, with coexistence-first and no-forced-rewrite policy.
- `R106`, `F97`, `R107`, `F98`, `F99`, `R108`, `F100`, and `R109` follow `reference/stable-diffusion-webui/modules/prompt_parser.py`, `modules/sd_hijack_clip.py`, and Forge equivalents as primary semantic sources, with `reference/ComfyUI_smZNodes/modules/text_processing/prompt_parser.py`, `smZNodes.py`, and `nodes.py` as the primary Comfy adaptation references.
- `R113`, `F103`, `F104`, and `R114` continue the same primary prompt-parity source lineage and additionally treat `reference/ComfyUI_smZNodes/modules/text_processing/textual_inversion.py`, `reference/ComfyUI/server.py`, and `reference/ComfyUI/folder_paths.py` as required embeddings/textual-inversion and host-inventory references.
- The same prompt-parity chain may optionally consult secondary external Comfy-side references such as `asagi4/comfyui-prompt-control` and `BlenderNeko/ComfyUI_ADV_CLIP_emb`, but only as implementation supplements; behavior disputes are resolved in favor of A1111 / Forge source files.
- `F100` explicitly corrects the over-broad parity language introduced by the shipped `R55/F65/F66` baseline so capability surfaces distinguish exact support from approximation or fallback.
- `R110` follows repository-wide host inventory contracts and the `Reproduce -> Pin -> Sweep` bugfix acceptance model to ensure canonical request-default sentinels (`__host_default__`, `Automatic`) are resolved before strict host inventory matching and that generate failures expose actionable diagnostics.
- `R111`, `F101`, `F102`, and `R112` follow `reference/ComfyUI-Impact-Pack`, `reference/ComfyUI-Impact-Subpack`, and `reference/ComfyUI-Advanced-ControlNet` as architectural references only. Selective migration of key nodes or core algorithms is allowed when it materially improves RookieUI, but the resulting runtime must be RookieUI-owned and must not depend on those external packs being installed.
- Adoption rule: no Forge/reForge optimization becomes a default RookieUI behavior until it passes the SD-family parity lanes defined by `R03`.
- Newer-family support rule: support for Flux, Qwen-Image, Wan, ZiT, Klein, Lumina, Anima, and similar families should prefer existing ComfyUI execution paths unless a deeper custom port is clearly justified.

## Global Acceptance Gate For Each Item

Every roadmap item is considered incomplete until the corresponding dated plan and implementation record exist and the final validation passes:

1. `pre-commit run detect-secrets --all-files`
2. `pre-commit run --all-files --show-diff-on-failure`
3. Backend full unit tests per `tests/TEST_SOP.md`
4. Frontend E2E per `tests/TEST_SOP.md`, `tests/E2E_TESTING_NOTICE.md`, and `tests/E2E_TESTING_SOP.md`

## Document Governance

- `.planning/ROADMAP.md` is the status-tracking index for all roadmap items.
- `.planning/roadmap/260409-S01S02R01R02R03F01F02F03F04F05F06F07F08_ROOKIEUI_REPO_ROADMAP_PLAN.md` is the detailed internal roadmap plan and reference synthesis.
- Future implementation plans and implementation records must use the required date-prefixed naming convention and reference the roadmap item codes they execute.
