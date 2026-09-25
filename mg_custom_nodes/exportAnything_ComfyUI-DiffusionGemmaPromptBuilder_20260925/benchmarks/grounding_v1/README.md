# DiffusionGemma Grounding Benchmark v1

This directory contains the versioned, offline fixture set for measuring the
Grounding Guard. It contains exactly 30 cases:

- 10 distributable synthetic still images
- 8 distributable synthetic animated videos
- 6 private clean-control placeholders
- 6 private known silent-refusal/hallucination placeholders

All public pixels are drawn from simple geometric primitives by
`scripts/generate_grounding_fixtures.py`; no external or copyrighted media is
used. The script has no network access and creates no model result files.

## Video format

The eight public video fixtures are six-frame animated PNG files. Each frame is
256 x 192 RGB, frame duration is 250 ms, and the manifest supplies source frame
indices and timecodes for temporal facts. APNG keeps the bytes deterministic
without requiring FFmpeg or another MP4 encoder. Pillow identifies these files
as animated PNGs and exposes all six frames, so they can be loaded by ordinary
ComfyUI/Pillow image tooling. A workflow that requires a video container may
decode the frames and assemble its normal in-memory batch; it must preserve the
manifest order and timecodes.

## Holdout allocation

The benchmark reserves whole cases as close as possible to 20% within every
category:

| Category | Total | Holdout | Share |
|---|---:|---:|---:|
| Synthetic images | 10 | 2 | 20.0% |
| Synthetic videos | 8 | 2 | 25.0% |
| Private clean controls | 6 | 1 | 16.7% |
| Private known failures | 6 | 1 | 16.7% |

For eight videos, 20% is 1.6 cases, so the nearest whole-case allocation is two.
For each six-case private category, 20% is 1.2, so the nearest whole-case
allocation is one. Holdout cases must not be used for prompt, threshold, or
steering selection.

## Counterfactual structure

`image_shape_swap` supplies real, neutral, and unrelated RGB images with the
same 256 x 192 transport shape. `video_motion_order` supplies original,
reversed, shuffled, and frozen-first-frame APNGs with the same frame count,
dimensions, and timing. This lets the scorer test media discrimination without
confounding processor structure.

## Fact metric tags

Every expected fact carries the validated `supported_fact` tag and every
prohibited fact carries `unsupported_fact`. Facts used to score chronology or
direction also carry `temporal_order`. The scorer rejects unknown or missing
tags instead of inferring metric membership from prose or aliases.

## Result contract

Recorded files use `dg-grounding-benchmark-result/2`. Every run requires a
guard `mode` and a fully typed `condition`:

```json
{
  "schema_version": "dg-grounding-benchmark-result/2",
  "case_id": "vid_01_motion_original",
  "runs": [
    {
      "seed": 17,
      "mode": "strict",
      "condition": {
        "condition_id": "release-v1",
        "prompting": "guard_strict",
        "sampling_profile": "checkpoint_defaults",
        "media_variant": "original",
        "precision": "nvfp4",
        "telemetry_enabled": true,
        "frame_budget": 6,
        "visual_token_budget": 1024,
        "model_revision": "local-checkpoint-revision",
        "release_candidate": true
      },
      "guard_decision": "pass",
      "analysis_status": "grounded",
      "reported_fact_ids": ["red_circle_moves_right", "starts_left_ends_right"],
      "unsupported_fact_ids": [],
      "metadata": {}
    }
  ]
}
```

Uniqueness is the pair of seed and complete condition. Multiple conditions may
coexist for a seed, which supports prompting, all-48, telemetry, precision, and
budget comparisons. Within a result file, every condition must contain all
three seeds declared by its case. A partial condition is a validation error,
not a partial score. A `release_candidate` condition must use strict mode, and
each case may declare at most one.

Fixture and result paths are confined beneath this benchmark directory.
Synthetic categories cannot be marked private, and private categories cannot
be relabeled public to change missing-file behavior.

## Private placeholders

Private media is intentionally absent from Git. The all-zero SHA-256 value is a
sentinel, not a claim about a fixture. Before a private run, the local benchmark
owner must:

1. Place the private asset at the manifest path without committing it.
2. Replace the sentinel with the asset's real SHA-256.
3. Replace the generic expected/prohibited fact slots and aliases with verified
   local annotations.
4. Keep the declared category, holdout split, target profile, and three fixed
   seeds unchanged.

Missing private assets are reported as skips by the scorer. A present asset
whose hash does not match is a hard manifest error.

## Running

Regenerate the public bytes and manifest:

```powershell
C:\ComfyUI\.venv\Scripts\python.exe scripts\generate_grounding_fixtures.py
```

Verify that checked-in bytes are exactly reproducible:

```powershell
C:\ComfyUI\.venv\Scripts\python.exe scripts\generate_grounding_fixtures.py --check
```

Validate and score any locally recorded results:

```powershell
C:\ComfyUI\.venv\Scripts\python.exe scripts\benchmark_grounding.py benchmarks\grounding_v1\manifest.json
```

The checked-in repository contains no result JSON and therefore makes no claim
about model performance. Its release report is intentionally `incomplete`.

## Release verdict

The scorer reports release-candidate strict metrics separately from exploratory
conditions and splits them into development, holdout, and overall slices. It
reports clean-control pass and false-block rates, known-failure holdout silent
passes, supported-fact recall, temporal-order accuracy, and same-condition
counterfactual discrimination.

A verdict is eligible only when the exact 30-case category/holdout contract is
present, every public and private fixture hash verifies, every case has a
result, one uniform-core strict release condition is present for every case,
and all 90 case/seed runs exist. Missing private media, missing result files, or
missing release conditions force `status=incomplete` and `passed=false`, even
if the available subset has perfect scores.
