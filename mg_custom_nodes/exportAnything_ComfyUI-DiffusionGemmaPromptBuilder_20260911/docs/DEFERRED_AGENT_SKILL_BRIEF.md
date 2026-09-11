# Deferred DiffusionGemma Agent Skill Brief

Status: **Pinned for post-hardening implementation; inactive.** This document is intentionally outside `.agents/skills`, so Codex and other compatible agents do not discover or invoke an operating contract that is not finished yet. The separate Advertisement workflow expands the future scope but does not activate or install this Skill.

Pinned: 2026-08-23
Last reviewed: 2026-08-24

## Canonical skill identity

```yaml
name: diffusiongemma-director
description: Configure, run, validate, and troubleshoot DiffusionGemma Director LTX, MiniMax-H3 music-video, and separate Advertisement workflows in a user's local ComfyUI through an available local operations interface. Use for this package's target profiles, typed campaign and reference contracts, soundtrack source and performance routing, native-shot and generation-lane planning, identity/product continuity, deterministic finishing, media QA, contract failures, queueing, or output verification; do not use for unrelated ComfyUI graphs, generic prompt writing, or ungoverned advertising claims.
```

Treat that `name` and `description` as the canonical starting point when the active skill is created. Revise them only if the hardened public surface materially changes the skill's scope or trigger boundary.

## Intended outcome

The eventual skill should translate an ordinary-language production request plus user-supplied images, video, or audio into a correctly configured DiffusionGemma workflow; operate it through an available local ComfyUI interface; diagnose contract or runtime failures; and verify actual branch media output rather than treating a completed queue as proof of generation. For Advertisement requests it must also preserve exact approved copy and claims, distinguish performer and product authority, separate the H3 motion guide from final voice-over, and report delivery evidence honestly.

The skill supplies operating knowledge. It does not itself grant filesystem, browser, model-download, or ComfyUI execution access. Those capabilities must come from explicitly available tools, ideally a local ComfyUI MCP server declared as a plugin dependency.

## Boundaries to preserve

- Keep local ComfyUI as the default execution target. Use cloud generation only when the user explicitly requests it.
- Never bypass the Director's fail-closed gate or silently weaken a Project Master, identity, soundtrack-hash, timing, or prompt-validation contract.
- Distinguish MiniMax-H3 native `[Shot N]` blocks from duration-bounded H3 generation lanes.
- Treat Natural, Dance, and Lyrics as mutually exclusive performance authorities. Written lyrics are not a timing schedule without verified timed-lyrics evidence.
- Keep clean identity references authoritative. Previous-lane relay is optional continuity evidence, not a replacement identity source.
- Preserve the selected soundtrack source, pristine final audio, exact retained-frame clock, and user-selected camera intent.
- Keep the known-good V6 workflow and generic contracts separate from Advertisement contracts. Never migrate or overwrite V6 to satisfy an ad request.
- For Advertisement references, preserve Picture 1 as performer hero, Picture 2 as same-performer sheet, Picture 3 as independent product/package sheet, and Picture 4 as retained-tail continuity only. Never make Picture 4 a new identity or product authority, and never combine performer and product sheets to save a reference slot.
- Distinguish the three Advertisement soundtrack origins—MiniMax Music 3, Upload song, and Legacy ACE-Step—from the `Instrumental` / `Vocal` / `Auto` content policy. Do not require vocal evidence for a governed instrumental.
- Send only the selected soundtrack to H3 as the motion guide. Optional non-diegetic VO belongs only in the final delivery mix and must not become dialogue or lip-sync conditioning.
- Keep exact brand, CTA, and legal copy in the deterministic end card. Never invent or strengthen a product claim, and never represent generated in-scene typography as exact.
- Treat the 30-second master plus rendered 15-second and 6-second temporal cutdowns as media outputs. Treat 1:1 or 16:9 adaptations as unrendered until an actual semantic reframe renderer produces them.
- Preserve `pass`, `fail`, and `not_measured` as distinct QA states. A completed queue, valid graph, or successful Music 3 canary is not evidence that performer identity, product identity, copy legibility, sync, or the complete H3 campaign passed.
- Require fresh authorization before downloads, installs, external publication, destructive changes, or other actions not already placed in scope.

## Hardening exit criteria

Promote this brief to `.agents/skills/diffusiongemma-director/SKILL.md`, or package it in an installable plugin, only after:

- the exported node ids, widget contracts, and workflow migration policy are intentionally versioned;
- the checked-in production workflow passes a clean-install graph and dependency audit;
- the local ComfyUI operations/MCP interface needed by the skill is stable and documented;
- validation errors, cache refresh/reuse behavior, lane assembly, and media-output checks have repeatable tests;
- required versus optional models and custom-node packs are expressed without machine-specific paths;
- representative Natural, Dance, Lyrics, upload-song, identity-relay, long-duration, and failure-recovery requests have been forward-tested.
- the Advertisement workflow has clean-install coverage on ComfyUI `>=0.33.1`, including the locally verified MiniMax Music 3 INT8/ConvRot text-encoder path and all three soundtrack origins;
- Picture 1/2 performer authority, Picture 3 product authority, Picture 4 retained-tail relay, soundtrack-only H3 guidance, optional final-mix VO, deterministic end cards, actual 15-second/6-second cutdowns, and honest QA states have repeatable integration tests;
- at least one complete Advertisement H3 render—not only an audio canary or serialized fixture—has been reviewed for performer/product retention, copy legibility, audio sync, and exact delivery clocks;
- model-generated package text, stochastic reference drift, missing semantic reframing, and unmeasured QA evidence remain documented as limitations rather than silently repaired or marked passed.

Current hardening evidence: the exact release API graph has completed a full two-lane 30-second Advertisement render with retained-tail relay, deterministic end card, exact 30/15/6 media clocks, and all diagnostic roots. A sampled visual contact-sheet review found stable performer/wardrobe and recognizable product continuity. The active Skill is still deferred because evidence-backed listening/audio-sync approval, representative mode/source coverage, and the remaining clean-install matrix above are not complete; the workflow QA result remains `not_measured`, not promoted to `pass`.

## Expected active-skill structure

```text
diffusiongemma-director/
|-- SKILL.md
|-- agents/
|   `-- openai.yaml
`-- references/
    |-- advertisement-workflow.md
    |-- workflow-modes.md
    |-- node-contracts.md
    `-- troubleshooting.md
```

Keep the final `SKILL.md` concise. Route only mode-specific contracts and troubleshooting details into the references, and add a deterministic script only if repeated workflow inspection cannot be expressed reliably through the ComfyUI operations interface itself.
