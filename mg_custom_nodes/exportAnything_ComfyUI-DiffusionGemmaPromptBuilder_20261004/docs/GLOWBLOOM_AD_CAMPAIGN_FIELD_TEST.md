# GLOWBLOOM 30-second ad campaign field test

Date: 2026-08-23

This note records a real end-to-end ad use-case trial of the synchronized MiniMax H3 Ref2VA V6 workflow. It separates capabilities that produced media from behaviors that were only planned or that failed during the trial.

## Brief and assets

The fictional product was **GLOWBLOOM YUZU PEAR**, a nonalcoholic sparkling botanical drink for trend-conscious adults in their twenties. The campaign idea was **Brighten the In-Between**.

GPT Image 2 produced two external campaign references:

- `input/diffusiongemma_campaigns/glowbloom_20260823/glowbloom_hero_identity.png`
- `input/diffusiongemma_campaigns/glowbloom_20260823/glowbloom_identity_product_contact_sheet.png`

Both images show the same adult woman, wardrobe, and orange can. The contact sheet is a same-person identity sheet; it is not an independently governed product reference.

The final in-contract configuration used:

- 30 seconds, 9:16, 480x864, 24 fps
- eight native H3 shot blocks split into two 15-second generation lanes
- a native-shot and measured-recovery lane boundary at 15.000 seconds
- `Natural / audio-led sync`
- one semantic Subject: the performer
- two same-person identity pictures
- the can as an ordinary recurring prop, not a separately retained Subject
- uploaded-source instrumental at measured 104.17 BPM
- Director thinking mode off, temperature 0.35, cinematic creativity 0.7, cache reuse
- dialogue off and no timed-lyrics analysis

## Rendered outputs

Two exact 30-second masters were rendered successfully:

| Version | Relay | Prompt ID | Output |
| --- | --- | --- | --- |
| V1 | Off | `0cf74d13-0c05-4031-96ae-dcfbc17c9cf4` | `output/diffusiongemma_campaigns/glowbloom_20260823/glowbloom_30s_ad_v1_00001_.mp4` |
| V2 | Previous lane tail | `d9fe3260-c297-4004-ab6b-bfd85b5a2651` | `output/diffusiongemma_campaigns/glowbloom_20260823/glowbloom_30s_ad_v2_relay_00001_.mp4` |

Both masters contain H.264 video at 480x864/24 fps and stereo AAC at 48 kHz. V2 is the preferred review master because its later product sequence and final hero composition are stronger.

Visual sampling showed unusually good performer consistency and unusually legible physical can text, including `GLOWBLOOM` and `YUZU PEAR`. This is an observed success, not a deterministic logo or OCR guarantee.

## Confirmed capabilities

- A 30-second vertical master can run as two 15-second H3 lanes while retaining eight unique native shots.
- The Project Master, measured audio, lane planner, exact-frame assembler, and final pristine-audio mux agree on the 30-second clock.
- Natural mode avoids timed-lyrics analysis and does not install a written lyric schedule.
- Upload song accepts an instrumental track, treats missing vocal proxy as advisory, and preserves a waveform hash lock.
- Two references can jointly retain one performer identity when Picture 2 is explicitly declared as same-Subject multi-panel evidence.
- The can can remain visually prominent as a prop even without an independent product identity contract.
- Previous-tail relay is callable and injects Picture 3 into lane 2 as pose/spatial/lighting/motion evidence rather than identity authority.
- Changing only continuity policy leaves the lane-1 prompt hash unchanged.
- The delivery node honestly plans 16:9 and 1:1 adaptations plus 15-second and 6-second cutdowns without falsely claiming that it rendered them.

## Gaps and failure evidence

### P0: generated instrumental QC rejects the requested music

ACE-Step generated the requested instrumental and measured it at 104.17 BPM, but generated-candidate QC rejected it solely for `no_non_silent_vocal_active_proxy_window`. The same waveform passed technical QC after re-entry through Upload song, where missing vocals are advisory.

Needed hardening: carry an explicit instrumental/vocal-content contract from the blueprint into `DiffusionGemmaAudioCandidateSelector`, and never require a vocal-active window for an authored instrumental.

### P0: no independent product identity/retention role

With expected Subject count set to Auto, Director correctly defined the woman as Subject 1 and the can as Subject 2. The planner then rejected the packet because dual identity pictures are required to bind to one identity Subject. The failure was:

`Dual identity pictures have conflicting or ambiguous semantic Subject bindings: <Subject 1>, <Subject 2>.`

The successful render therefore demoted the can to an ordinary prop. Its observed label stability is not protected.

Needed hardening: add a product/package reference role and product-retention contract independent from the two-picture performer identity contract. A clean extension would reserve a separate product picture and permit product Subject binding without treating it as a second performer.

### P0: shared manifest semantics are not actually shared

The planner rejected `same adult woman only` even though it was semantically equivalent to the required same-Subject declaration. It accepted wording only after the manifest included regex-compatible phrases for multi-panel, same Subject, layout/grid, seams, backgrounds, pose sequence, and separate people.

Needed hardening: centralize reference-manifest parsing and validation, return field-specific diagnostics, and avoid component-specific wording passwords.

### P1: Director transport and repair latency are disproportionate

Observed Director behavior during this test:

- Grounding Guard audit performed a 4 minute 44 second guarded pass and began a second comparison pass.
- Thinking mode on took 22 minutes for a 1,321-token first draft, then approximately eight minutes per repair attempt.
- Thinking mode off reduced comparable first drafts to roughly 2 minutes 10 seconds.
- Stochastic drafts still produced `minimax_h3_extra_top_level_field`, wrong Subject count, and `json_parse_invalid`/salvaged empty templates.
- Failed retries do not surface their rule codes until all expensive generation attempts finish.

Needed hardening: add a concise commercial preflight profile with thinking off, a typed ad/shot schema, bounded response size, early streaming validation, and immediate visibility into repair reasons.

### P1: lane-boundary creative state does not obey the planned wipe

The planner selected a correct 15.000-second native/recovery boundary. In both V1 and V2, lane 1 had already reached the rooftop before 15 seconds, while lane 2 restarted from the studio composition at the boundary. V2 relay carried continuity evidence but did not prevent the initial studio reset; a short blended transition appeared later in lane 2.

Needed hardening: add an explicit end-state/start-state handshake at each generation boundary, make the relay state authoritative for lane-2 opening composition when requested, and validate boundary frames before committing the full assembly.

### P1: relay iteration does not reuse large lane output under current cache policy

Although V1 and V2 had identical lane-1 prompt hashes, ComfyUI was launched with `--cache-lru 1`, so the decoded lane-1 frames had been evicted. V2 regenerated lane 1 to recover the relay tail.

Needed hardening: persist or explicitly save small relay-tail artifacts independently from the full decoded-frame cache, so continuity reruns do not repay an unchanged lane.

### P1: EasyCache provides no acceleration in this graph

Every H3 lane reported `skipped 0/7 steps (1.00x speedup)` at the tested threshold and window. The approximately 2 minute 24 second lane sampler time came from the Turbo path, not EasyCache.

### P2: commercial finishing is plan-only or absent

The workflow does not currently provide deterministic:

- logo/package OCR QA or package-label repair
- headline, CTA, legal line, price, offer, or end-card compositing
- product claims and legal approval fields
- non-diegetic voice-over generation, ducking, mixing, or loudness delivery
- rendered 16:9/1:1 adaptations or 15-second/6-second cutdowns
- campaign variant batching and review manifests
- contact-sheet creation inside the ComfyUI graph

The checked-in V6 workflow is also UI-format only and has no checked-in API-format execution companion. Its saved UI copy contains bypassed core H3/audio nodes, while known-good API execution graphs currently come from history.

## Recommended build order

1. Make generated-instrumental QC content-aware.
2. Add a governed product/package reference and retention role.
3. Share one semantic manifest parser across Director, gate, and planner.
4. Add a concise commercial preflight/shot schema with thinking off.
5. Make lane boundary state and relay opening composition testable and enforceable.
6. Persist relay-tail artifacts independently from the large-frame cache.
7. Check in an active API-format V6 companion and verify that every final-output ancestor is enabled and class-resolvable.
8. Add finishing nodes for brand text, CTA/legal, OCR QA, voice-over/mix, and actual adaptations/cutdowns.

## Honest conclusion

The present skeleton can already produce a visually credible 30-second lifestyle beverage ad with synchronized instrumental music and stable performer identity. It cannot yet claim governed product identity, deterministic commercial copy, or reliable cross-lane story-state continuity. Those are the primary differences between a successful concept spot and a repeatable ad-campaign production system.
