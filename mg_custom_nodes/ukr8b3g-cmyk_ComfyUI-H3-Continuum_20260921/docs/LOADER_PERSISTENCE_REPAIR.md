# Loader persistence repair — 2026-09-21

Base: `3d7c2fd6928a545b12b84968b09dc455e7f9da65`.

Image keeps its Core-backed implementation and Enable convenience toggle. The mode descriptor lookup walks the prototype chain; a tracked Core accessor is called rather than shadowed by a detached closure. Draw/serialize/configure functions and widget collections are not replaced. The former drawing swap could truncate a stable Core widget view, separately from Issue #23's mode-store divergence. New regression models both failures.

The current paired official V3.8X2 workflows replace H3EasyLoadAudio with Core LoadAudio and H3ContinuumLoadVideo with Core LoadVideo + H3ContinuumVideoAdapter. Core titles, original media names, disabled input branches, video 24fps setting, and all unrelated node values and original links are preserved. Video/audio outputs remain wired through the old output node ID. Core LoadVideo and the converter are kept in the video input group. Disable the converter or entire group, not only the Core input while its adapter remains active.

Legacy H3EasyLoadAudio/H3ContinuumLoadVideo remain registered/deprecated. Their old backend input contracts remain intact. The new converter has no Enable, file selector, upload handler or custom JavaScript. It uses the established _resample_video_frames() and passes existing audio through without resampling. Source rate 24 is a frame-object no-op; 30/29.97 to 24 uses the existing duration-preserving selection. This does not claim VFR timestamp-aware conversion or motion interpolation.

No user Windows worktree, installed runtime, GPU, queue, saved workflow, Run Storage or Takes were accessed. A Core 0.36.0/frontend 1.53.6 live-browser save/reopen, A->B->A, direct/right-click Bypass and upload check remains pending. The fixed H3情報チェック handoff is PENDING: cross-task transport is unavailable in this session. No Release/tag or Registry publication.

Files already saved without their selection/OFF state require one explicit user reselection; never reconstruct them from defaults. Upstream graph replacement may correctly create a new Run Storage revision, so retain an old workflow for resuming its saved run.

## Validation result

- Original-code regression: 6 cases; 3 expected failures and 0 collection/runtime errors.
- Focused CPU/Node suite: 46 passed, zero failures/errors/skips.
- JavaScript module syntax, changed Python compilation, workflow JSON/ZIP parity, endpoint consistency and git diff whitespace checks passed.
- Official workflow structural guard preserved every non-loader node, each original link, and non-layout workflow metadata.
- Scope: isolated GitHub Actions CPU runner, not Windows deployment or a real-browser/GPU test.

## Loader repair full-suite follow-up (2026-09-21)

- Main repair `f8eb40d8b44e54386bcf3c5197c48f0f71daf6df` passed the 46 focused tests. Its first full CI returned 1376 passed, 3 skipped, and one stale pre-migration workflow fingerprint failure in `test_v38x2_decode_cache_integration.py`.
- Updated only that expected fingerprint to the user-approved migrated workflow (1B2212B7511FAC0B63DFA6E9F006CD8644C72D08C6945AB0E4ED0E722C97A730); all Decode Cache route and output assertions remain. No runtime or workflow bytes changed in this follow-up.
- Full isolated CPU/Node suite: 1377 passed, 3 skipped, zero failures/errors. Actions run 35558988542. Source was snapshotted with `tools/snapshot.ps1` before edits; source/Registry manifest hashes verified.
- Windows deployment, live-browser checks and GPU generation were not performed; fixed H3情報チェック handoff remains pending. No Release/tag/Registry publication.
