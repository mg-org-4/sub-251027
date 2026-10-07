# Changelog

All notable changes to DiffusionGemma Director are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); version numbers continue to follow the package metadata in `pyproject.toml`.

## [Unreleased]

### Added

- Added a documented real-world 30-second beverage-ad field test covering GPT Image 2 campaign references, two-lane MiniMax H3 execution, instrumental upload routing, identity retention, relay comparison, rendered review masters, and a prioritized commercial-production hardening backlog.
- Added a separate twenty-two-node Advertisement contract, shared-control, audio, Director-transport repair, planning, relay, runtime-memory, finishing, and QA surface, bringing the registered ComfyUI total to 55 nodes while leaving the established 33-node surface intact.
- Added versioned Advertisement schemas for campaign fields and claims, ordered performer/product references, soundtrack policy, locked audio, H3 planning, retained-tail relay, final mix, delivery status, adaptations, and media QA.
- Added a native MiniMax Music 3 Advertisement route with structured commercial timing, five-second candidate headroom, the locally verified pruned INT8/ConvRot text-encoder path, and lazy MiniMax Music 3 / Upload song / Legacy ACE-Step source selection.
- Added `examples/16_minimax_h3_ref2va_advertisement_music3_v1.json` as a separate user-editable Advertisement workflow with a fresh graph identity, flattened independent nodes, and immutable V6 lineage guards, plus a governed API-format companion fixture.
- Added deterministic exact-copy end-card rendering, exact 30-second master finishing, real 15-second and 6-second frame/audio cutdowns, and explicit status-only reporting for unrendered 1:1 and 16:9 adaptations.
- Added an Advertisement-owned model-memory barrier so Music 3 resources can be released before Director and H3 without inheriting unrelated cleanup-node packs.
- Added end-to-end runtime evidence for the exact release API graph on isolated ComfyUI `0.33.1`: two 15-second H3 lanes, persistent Picture 4 relay, every diagnostic root, a 720-frame/30-second master, and real 360-frame/15-second and 144-frame/6-second cutdowns with exact matching audio clocks. Visual spot review found stable performer/product continuity; formal QA remains blocked pending evidence-backed identity, copy, and audio-sync approval.
- Added 17 exported audio, timing, and production-planning nodes, bringing the established non-Advertisement ComfyUI surface to 33 nodes.
- Added decoded-song audition and locking with measured tempo, onset, tonal, clipping, silence, artifact, vocal-activity, recovery-window, excerpt, and whole-song diagnostics. Candidate and waveform-hash locks now fail closed instead of silently selecting different music.
- Added a custom-song path with a lazy upload node and independent **Generate with ACE-Step / Upload song** router. Uploaded songs reuse the established selector, excerpt, hash-lock, audio-guide, planner, and pristine final-mux path without waking dormant ACE candidate lanes.
- Added a single **Natural / audio-led sync**, **Dance / music sync**, or **Lyrics + lip sync** performance control shared by Director guidance, lyric selection, timed analysis, and H3 planning.
- Added lazy timed-lyrics analysis using local vocal separation and Whisper alignment. Only verified, soundtrack-hash-locked vocal events guide articulation; instrumental, missing, stale, weak, or failed evidence safely falls back to Natural/audio-led behavior.
- Added experimental ACE **Compose new / Cover reference** conditioning. Cover mode remains lazy and fails closed until compatible reference-audio conditioning is connected.
- Added a deterministic Project Master contract, audio-aware H3 generation-lane planner, separate H3 video-seed fanout, optional previous-lane relay reference, exact-frame assembler, and plan-only multi-format delivery manifest.
- Added dual-reference identity preparation and relay support. Two clean references can remain the identity authorities while a downscaled previous-lane tail is used only as optional continuity evidence.
- Added Experimental Long Horizon planning for LTX with four contiguous duration-scaled pacing phases, compact output budgets, advisory diagnostics, and an explicitly unexecuted offline 5-, 20-, and 60-second benchmark protocol and manifest.
- Added hash-guarded workflow migrations for the synchronized music-video graph, MiniMax-H3 expansion, Turbo/fast test path, dual-identity relay, timed lyrics V5, uploaded-song V6, and an isolated identity-relay smoke graph.
- Added the validated MiniMax-H3 synchronized music-video V6 graph as a checked-in example with a repository-local integrity and contract test, so the working production artifact is preserved independently of mutable Desktop and ComfyUI copies.
- Pinned the canonical name, description, scope boundaries, and activation criteria for a future `diffusiongemma-director` agent skill without making the unfinished skill discoverable.
- Added regression coverage for audio QC and routing, timed lyrics, production planning, exact assembly, H3 dialogue and shot counts, reference-contract repair, identity relay, uploaded songs, camera capability, and long-horizon planning.

### Changed

- Centralized Advertisement v1's 30-second duration, 9:16 aspect, native-shot count, Natural/Dance performance, Ref2VA mode, H3 audio/dialogue policies, excerpt start, generation model, 15-second lane ceiling, splitter aspect, and governed 30/15/6 deliverables in one shared control. Unsupported reference layouts, seam modes, Lyrics-without-evidence, and VO-without-input choices are hidden from the shipped UI.
- Labeled automatically saved Advertisement media as review drafts. QA approval remains separate, requires evidence for every pass, and never treats queue completion as delivery approval.
- Assigned Advertisement references by role: Picture 1 performer hero, Picture 2 same-performer contact sheet, Picture 3 independent product/package sheet, and Picture 4 persistent retained-tail relay for continuity only. Product and performer authority remain separate across every H3 lane.
- Split Advertisement audio into a soundtrack-only H3 motion guide and an exact final delivery mix. Optional non-diegetic VO, ducking, mix hashes, and clipping diagnostics affect only the final mix, never H3 conditioning, and cannot become an accidental lip-sync instruction.
- Made Advertisement soundtrack QC content-aware: instrumental generations do not require a vocal proxy, vocal generations do, and uploaded or legacy sources retain their meaningful technical-integrity and locking policies.
- Added explicit `pass`, `fail`, and `not_measured` Advertisement media-QA states plus an unremovable host baseline for product identity, performer identity, copy legibility, audio sync, and technical integrity. Queue completion, widget edits, and missing evidence can no longer be presented as identity, typography, product, or sync approval.
- Separated MiniMax-H3 native `[Shot N]` blocks from duration-bounded generation lanes. A single H3 invocation may now retain several native shots, and up to four planned lanes can cover a 60-second project when the configured 15-second lane ceiling and capacity constraints permit it.
- Reworked H3 lane boundaries to prefer native cuts with measured recovery, then native or mixed evidence, with deterministic balanced seams as a safe fallback. Non-native seams carry the active source shot and rebase timestamped cues rather than changing the master timeline.
- Made Project Master duration, aspect, soundtrack hash, excerpt, lane ceiling, and deliverables authoritative while leaving native shot choreography to the Director and H3 target profile.
- Made MiniMax-H3 camera policy model-specific. H3 may preserve requested orbiting, swirling, sweeping, whip movement, pronounced parallax, and coherent compound paths; unspecified cameras inherit an explicit creativity-scaled direction instead of being rejected.
- Added **Stable** and **Advanced** camera capability to the LTX target contract. LTX keeps its conservative motion safeguards where needed without imposing those restrictions on H3.
- Updated SplatStage planning so requested genre, subgenre, era, instrumentation, groove, and production style override visual nightclub or dance cues. Scene evidence may shape visuals and lyrics but no longer forces unrelated music into EDM.
- Extended SplatStage duration and aspect handling, measured-audio Director context, source-passthrough semantics, production concepts, and reproducible song-seed fanout.
- Revised Director caching and workflow fixtures so unchanged verified branches can reuse their packet while explicit refresh remains a one-run diagnostic choice.
- Codified the tested MiniMax-H3 fast path around the pruned INT8 model, official four-step Turbo LoRA, separate video/audio sigma shifts, and Euler/simple sampling. EasyCache and TorchCompile remain disconnected bypasses in that workflow because the measured configuration reported no effective acceleration.

### Fixed

- Preserved the host Director packet's grounding, runtime, cache, and reference-authority evidence during bounded Advertisement transport recovery. Raw output is considered only after an explicit host JSON-parse failure with successful plain salvage and no true template fallback; duplicate keys, non-finite values, malformed Unicode, metadata disagreement, excess repairs, and authority drift remain blocked.
- Removed invalid `model` inputs from all four native `MiniMaxH3ReferenceToVideo` conditioning nodes while retaining model links on the scheduler and guiders. The UI builder now rejects every connected API input that is absent from live ComfyUI node metadata, preventing the same unexpected-keyword failure class from reaching execution.
- Corrected the saved Grounding Guard title to state its intentional `off` baseline instead of implying audit mode; the independent Advertisement reference and generation contracts remain active.
- Corrected the four H3 lane labels to distinguish persistent Picture 3 product authority from Picture 4 relay continuity, corrected the scheduler label to its actual seven-step setting, made the Director checkpoint path application-relative, and made the API manifest enumerate only its real source branch, all required Director/H3/Music 3 assets, and real reference hashes.
- Kept a runtime-observed Music 3 tempo miss honest: the technically passing candidate's `87.890625` BPM signal estimate versus requested `122` remains a visible advisory, while H3 continues to follow the selected waveform and exact-tempo campaigns can use a locked upload.
- Fixed the MiniMax Music 3 selection boundary where time-to-sample rounding could leave an otherwise valid excerpt one sample short. The mixer records and right-pads only deficits up to one millisecond; material shortages still fail.
- Fixed Advertisement control-link combo and ACE integer-BPM typing, selected-source BPM provenance and mismatch evidence, exact brand-case rendering, canonical report/tensor hash verification, frame-clock and pixel-aspect alignment, retained-tail cutdown selection, and false LoadImage missing-input warnings in the generated UI workflow.
- Canonicalized only unambiguous `Picture N` citations inside governed Advertisement subject-definition rows before strict validation, eliminating meaningless Director retries without relaxing free-form prompt validation.
- Prevented Advertisement product references from being parsed as a second performer identity, and prevented the retained-tail Picture 4 relay from becoming a new source identity or product authority.
- Prevented added voice-over from entering the H3 motion guide, generated instrumental candidates from failing for lack of vocal evidence, and incomplete visual/audio QA from becoming a false pass.
- Repaired MiniMax-H3 Ref2VA packets deterministically when model output contains malformed subject definitions, reference usage or retention rows, misplaced or short detailed descriptions, invalid cut timestamps, or omitted per-shot camera direction.
- Removed the one-shot Project Master mismatch that incorrectly compared H3 native shot blocks with the number of generation invocations.
- Prevented generic camera-safety text from making H3 output creatively anemic while retaining fail-closed validation for genuinely malformed camera contracts.
- Prevented Dance mode from forcing singing or mouth motion, and prevented raw written lyrics from acting as an unverified timing schedule. Lyrics mode now supplies articulation cues only during measured vocal intervals.
- Preserved soundtrack timing across lane-local lyric windows and exact-frame assembly, including explicit non-vocal handling for instrumental gaps.
- Reduced long-video identity drift by keeping clean identity references authoritative across lanes and making relay continuity optional, bounded, and independently lazy.
- Made missing ideal H3 seam evidence advisory rather than fatal while continuing to block invalid clocks, malformed shot structures, stale audio locks, impossible lane capacity, and bad identity manifests.
- Hardened workflow migrations with immutable-source hashes, additive policy markers, exact wiring checks, and regression tests for the validated Desktop and ComfyUI copies when those local artifacts are available.

### Repository hygiene

- Excluded generated runtime videos and identity/reference frames from Git and ComfyUI package archives. These files remain local and are not required by the source or tests.
- Kept migration tools and offline benchmark manifests in Git while excluding them from the install archive, and updated package metadata and the release checklist for the 55-node surface.
- Made the legacy external V5 Desktop/ComfyUI byte-equality audit explicitly opt-in; the default release suite now relies on the checked-in V6 workflow's hermetic hash and contract validation.

## [0.2.0] - 2026-08-07

### Added

- Introduced the model-specific DiffusionGemma Director core, grounding guard, fail-closed generation gates, LTX and MiniMax-H3 target profiles, and packaged example workflows.
