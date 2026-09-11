# DiffusionGemma Grounding Guard

The Grounding Guard detects visual transport and grounding failures before DiffusionGemma-authored descriptions are allowed to drive LTX, MiniMax H3, or Ideogram. It separates observations from inferences and creative additions, exposes suspected refusal trajectories, and can stop unverified prompts at the existing Generation Gate. It does not alter model weights, patch Transformers, ban refusal words, or claim to remove the model's internal safety behavior.

Everything runs locally. The guard makes no network calls and does not require a cloud verifier.

## Settings Node

`DiffusionGemma Grounding Guard Settings` produces `DG_GROUNDING_GUARD_CONFIG` plus a human-readable `config_json` output. Connect the config to the optional `grounding_guard_config` input on `DiffusionGemma CoT Generator`.

| Input | Default | Meaning |
| --- | --- | --- |
| `mode` | `audit` | `off`, `audit`, or `strict`; see the mode contract below. |
| `retry_on_uncertain` | `true` | Allows one focused evidence retry in strict mode. In audit mode it also allows one fresh combined retry when an LTX or H3 packet is malformed, salvaged, or contains an invalid grounding ledger. It never permits an open-ended loop. |
| `evidence_token_budget` | `auto` | Advanced strict-mode evidence-output budget. `auto` uses 768 tokens initially and up to 1024 on retry; `768`, `1024`, and `1280` apply that fixed budget to every evidence attempt. |
| `sampling_profile` | `checkpoint_defaults` | Selects the native in-process DiffusionGemma denoising profile. `full_48_diagnostic` retains all 48 steps by disabling adaptive stopping. |
| `seed` | `0` | Unsigned 64-bit seed used in a scoped RNG context. The caller's CPU and CUDA RNG state is restored afterward. |
| `save_detailed_trace` | `false` | Opts into a local JSON trace. Compact graph metadata is returned whether this is enabled or not. |

`evidence_token_budget` affects only the Guard's structured evidence passes. MiniMax H3 storyboard length is controlled separately by `minimax_h3_shot_count` on the Target Profile (`auto`, a preset from `1` through `12`, or `custom` plus a typed value from `1` through `99`).
| `trace_subfolder` | `diffusiongemma_grounding` | Relative folder beneath `ComfyUI/output`; absolute paths and traversal are rejected. |
| `external_evidence_json` | empty | Optional typed evidence from a local OCR, detector, pose, tracking, motion, SigLIP, or manual process. |

Both sampling profiles use at most 48 denoising steps, a `0.8` to `0.4` temperature schedule, an entropy bound of `0.1`, token-stability threshold `1`, and confidence threshold `0.005`. `checkpoint_defaults` permits native adaptive stopping; `full_48_diagnostic` disables it. The scalar `temperature` on the CoT Generator remains for GGUF compatibility, but in-process DiffusionGemma guard calls use the selected native sampling profile.

The evidence token budget applies only to strict evidence passes and is independent of the CoT Generator's `max_new_tokens` when an explicit value is selected. Larger budgets can take longer, but they do not relax schema validation, grounding coverage, or fail-closed behavior. The setting is ignored in `off` and `audit` modes because those modes do not run a separate strict evidence pass.

## Mode Contract

### Disconnected input: implicit audit

Existing workflows do not need to be rewired. A disconnected `grounding_guard_config` input is interpreted as `audit` with the defaults above. Audit inspects transport, the evidence ledger, coverage, and the passive denoising trajectory, but it never clears prompts or changes the packet's existing readiness result.

### `off`: exact compatibility

An explicitly connected Settings node with `mode=off` selects the legacy generation path. Prompt construction, generation and repair calls, readiness, random-number behavior, and the original four output positions remain unchanged. The generated packet and metadata are augmented with `final_json.metadata.grounding_guard`, while the appended guard outputs report `not_run` / `disabled`; consumers that compare the first four serialized values byte-for-byte should account for that added metadata.

### `audit`: observe without blocking

Audit normally uses one combined Director pass to produce the evidence ledger and generator prompts together; it does not add a separate evidence-only pass. A narrow deterministic transport repair may recover an unambiguous packet wrapper or closing delimiter without changing prompt semantics. When `retry_on_uncertain=true`, an LTX or H3 response that is still malformed, salvaged, or carries a schema-invalid grounding ledger receives exactly one fresh combined retry under the same hard call budget. Existing target-validation repairs, such as the established H3 repair step, remain available so audit does not reduce legacy prompt readiness. No salvage is promoted to ready: if the retry remains invalid, the splitter and Generation Gate still fail closed. If the same evidence would pass strict verification, the decision is `pass`. Otherwise the decision is `warn`, and the report sets `grounding_guard_would_block=true`. Current prompt readiness and downstream output remain unchanged.

Telemetry is passive: the per-call processor returns the model's original logits tensor unchanged. Telemetry failures warn and fail open in audit mode.

### `strict`: verify, then compile

Strict mode uses a two-stage boundary:

1. An evidence-only pass sees the original pixels and returns strict `dg-grounding-ledger/1` JSON.
2. The host verifies visual transport, schema, citations, asset-role coverage, uncertainty, refusal evidence, frame/time references, and typed external conflicts.
3. When the evidence is uncertain, refused, malformed, or insufficiently covered, the guard may retry once from the original pixels. The retry removes creative instructions, asks for an asset-by-asset factual checklist, and reduces video evidence to opening, middle, and closing frames while retaining required still references.
4. Only a verified ledger reaches the Director compiler. The compiler and any target-specific repairs receive the ledger and user intent, not the pixels or previous model-generated observations.
5. If evidence remains unverified, compilation is skipped. The returned packet is empty, `ready_for_generation=false`, and the existing splitter and Generation Gate prevent downstream execution.

Strict telemetry errors fail closed. A successful first evidence pass leaves room for the compiler and up to two target repairs. If the one evidence retry is used, only one target repair remains. A failed evidence stage makes no compiler or repair call. The hard ceiling is four model calls per strict run. When the loader uses `unload_after_run`, the wrapper performs exactly one final release on success, block, or exception; `keep_loaded` intentionally performs no unload.

The pixel evidence ledger remains exact strict JSON: fences, prose wrappers, duplicate keys, trailing commas, and truncated ledgers are rejected and can trigger the one full multimodal evidence retry. The later pixel-free compiler packet has a narrower deterministic transport-repair lane. Python may remove a recognized outer answer/fence wrapper, remove unambiguous trailing commas, remove one extra outer opening brace, append one provably missing root closer, and supply inactive packet-envelope fields. Duplicate keys, evidence provenance, H3 sections, claims, reference ordinals/roles, retention choices, timeline events, camera, audio, and visual semantics are never repaired this way. The complete packet, provenance, H3 validators, readiness calculation, splitter, Grounding Guard, and Generation Gate still decide whether the result can run. Applied repairs are recorded in packet metadata and the grounding report.

Strict mode requires at least one high- or medium-confidence observed fact for every required still/reference role. Multi-frame video must cite distinct temporal regions. Identity-plus-control inputs must cite identity evidence from the still and action or camera evidence from the video. Unknown assets, impossible frame/time references, empty or low-confidence-only observations, role cross-wiring, and exact typed evidence conflicts cannot pass.

For H3 Ref2VA, strict mode requires every connected Picture and Video role to carry an explicit host category annotation such as `[dg:identity,appearance]`, `[dg:action,motion,camera,temporal]`, or `[dg:object,color,text]`. These annotations use the ledger category vocabulary and are authoritative. Audit mode may infer categories from unannotated role prose only for compatibility diagnostics; strict mode fails such a role closed. Annotations are removed before the compiler and can never appear in a generated H3 prompt. Keep observations role-isolated (for example, describe video action separately from the pictured actor's appearance). The host enforces declared categories against the role contract and screens obvious prose/category mismatches, but the prose screen is intentionally heuristic. The report therefore says `transport+structured_self_report`, not independent visual verification, unless recognized external evidence was supplied.

## Outputs and Compatibility

The CoT Generator keeps its original output positions unchanged:

1. `final_json`
2. `reasoning_text`
3. `raw_response`
4. `metadata_json`
5. `grounding_status` (appended)
6. `grounding_report_json` (appended)

The same compact report is stored under `final_json.metadata.grounding_guard`. No sockets are added to the JSON Splitter or Generation Gate.

The CoT metadata also carries `director_runtime` and `director_cache`. Runtime telemetry reports per-call and aggregate load/bridge/preprocess/prefill/generation/validation/unload timing, generated length, first-pass acceptance, retry counts, peak allocated VRAM, and cache timing. In `reuse` mode, ComfyUI's in-memory graph cache keeps unchanged Director branches idle; the content-addressed disk cache stores only successful, ready, non-fallback packets whose guard decision is `pass`, `not_applicable`, or `disabled`. Strict blocks and audit warnings are never stored on disk. An unchanged blocked result can be retried by selecting `refresh` once, queueing, and returning the node to `reuse`. Cache hits still honor `unload_after_run`, and `save_detailed_trace=true` bypasses both reuse layers so every requested trace is written.

Analysis status is one of:

- `not_run`
- `not_applicable`
- `grounded`
- `uncertain`
- `refused`
- `transport_error`

The host's guard decision is one of:

- `disabled`
- `not_applicable`
- `pass`
- `warn`
- `block`

The host owns both the final status and decision. Model-supplied asset IDs, packet metadata, and claimed grounding status are validated or replaced. A complete refusal clause in the final evidence output, or in at least two of the last three late telemetry snapshots, counts as refusal evidence; isolated words and early noisy fragments do not block.

Strict failures include the umbrella reason `visual_grounding_unverified` plus the most specific available reason:

- `visual_transport_error`
- `visual_grounding_schema_invalid`
- `visual_grounding_refused`
- `visual_grounding_uncertain`
- `visual_grounding_insufficient_coverage`
- `visual_grounding_telemetry_error`

## Evidence Ledger

The evidence pass must return strict JSON matching `schemas/grounding_evidence.schema.json`. Its `dg-grounding-ledger/1` ledger keeps these categories separate:

- `observed_facts`: fact ID, factual claim, `high`, `medium`, or `low` confidence, categories, and evidence references.
- `inferred_facts`: deductions that were not directly observed.
- `creative_additions`: target-oriented invention that is allowed but must not masquerade as source evidence.
- `uncertainties`: unresolved visual ambiguity.
- `grounding_failure_reasons`: reasons the evidence could not be grounded.

Evidence references use host-created IDs such as `image:1`, `picture:1`, and `video:1`. Video references carry a sample ordinal, source-frame index, and timecode. A still may omit all three sample fields, but frame or time fields are never accepted without a sample ordinal. The host registry determines whether every reference is real, in bounds, and connected to the intended asset role.

Optional model-authored typed claims must cite an asset—and, when present, the exact sample ordinal—already cited by their enclosing observed fact. Their value is a string, Boolean, null, or signed 64-bit integer; encode an exact decimal measurement as a string. Contradictory values for the same typed key make the ledger malformed rather than allowing the compiler to choose one silently. Accepted frame and time references are replaced with the authoritative host-registry values before the validated ledger is published.

## Optional External Evidence

External evidence uses the `dg-external-evidence/1` typed-claim envelope defined by `schemas/external_evidence.schema.json`. Typed values follow the same signed-64-bit-integer-or-string rule, so exact decimal measurements should be encoded as strings. For example:

```json
{
  "schema": "dg-external-evidence/1",
  "provider": "ocr",
  "claims": [
    {
      "asset_id": "image:1",
      "claim_type": "visible_text",
      "value": "OPEN",
      "confidence": "high"
    },
    {
      "asset_id": "video:1",
      "sample_ordinal": 3,
      "source_frame_index": 48,
      "timecode_seconds": 2.0,
      "claim_type": "motion_direction",
      "value": "left_to_right"
    }
  ]
}
```

Recognized providers are `manual`, `ocr`, `detector`, `pose`, `tracking`, `motion`, and `siglip`. In v1, only an exact typed conflict on the same asset, claim type, and sample ordinal is blocking. Unknown providers, unknown schemas, or unstructured evidence stay informational and generate a warning instead of being treated as proof.

## Reports and Private Traces

`grounding_report_json` is a compact, graph-safe report capped at 256 KiB. It includes the asset registry, actual processor transport proof, host decision, validated ledger, attempts and retry reasons, trajectory summary, effective denoising settings, timings, forward counts, warnings, and the verification level. The initial verification label is `transport+structured_self_report`; recognized independent typed evidence can raise the evidence level without turning the model's own prose into independent proof.

When `save_detailed_trace=true`, the guard writes JSON only beneath:

```text
ComfyUI/output/<trace_subfolder>
```

The detailed trace may contain asset hashes, raw evidence/compiler responses, detailed trajectory summaries, model/environment revision data, and timing information. It never copies images or videos and never persists pixel tensors or logits. Trace writing performs no network calls. The saved path is recorded in the compact report.

## Proof Gates

From the repository root, the normal proof command remains report-only and exits zero even when optional hardware checks are unavailable:

```powershell
C:\ComfyUI\.venv\Scripts\python.exe proof_gates.py --model-path C:\ComfyUI\app\models\LLM\diffusiongemma-26B-A4B-it-NVFP4
```

Use opt-in required mode in CI or before a GPU acceptance run:

```powershell
C:\ComfyUI\.venv\Scripts\python.exe proof_gates.py --model-path C:\ComfyUI\app\models\LLM\diffusiongemma-26B-A4B-it-NVFP4 --require processor video nvfp4 telemetry gpu
```

Requested proof failures then exit nonzero. Valid proof names are `processor`, `video`, `nvfp4`, `telemetry`, `gpu`, and `all`.

Run the discoverable tests with:

```powershell
C:\ComfyUI\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### Opt-in local GPU acceptance

The acceptance harness is offline and uses only the local checkpoint path you pass. Its default is the processor-only tier and report-only exit behavior:

```powershell
C:\ComfyUI\.venv\Scripts\python.exe scripts\grounding_gpu_acceptance.py C:\ComfyUI\app\models\LLM\diffusiongemma-26B-A4B-it-NVFP4
```

Run all three tiers after the normal proof gates pass:

```powershell
C:\ComfyUI\.venv\Scripts\python.exe scripts\grounding_gpu_acceptance.py C:\ComfyUI\app\models\LLM\diffusiongemma-26B-A4B-it-NVFP4 --tiers 1 2 3
```

Tier 1 proves image/video processor transport, same-shape media counterfactuals, and frame-count shape sensitivity. Tier 2 performs one four-denoising-step, one-canvas NVFP4 telemetry smoke. Tier 3 performs five fixed-seed telemetry-disabled/enabled pairs and requires exact token-ID and decoded-text equality, median telemetry overhead no greater than 10%, and extra peak VRAM no greater than 512 MiB. Tier 3 makes ten short generation calls, so it is deliberately not part of the default command.

The harness prints `dg-grounding-gpu-acceptance/1` JSON and does not save media or logits. A failed tier returns nonzero only when explicitly required, for example:

```powershell
C:\ComfyUI\.venv\Scripts\python.exe scripts\grounding_gpu_acceptance.py C:\ComfyUI\app\models\LLM\diffusiongemma-26B-A4B-it-NVFP4 --tiers 1 2 3 --require 1 2 3
```

## Offline Grounding Benchmark

The benchmark scaffold reads a versioned `dg-grounding-benchmark/1` manifest and pre-recorded `dg-grounding-benchmark-result/2` files. It scores exact fact IDs locally, verifies fixture hashes, treats missing private fixtures as explicit incompleteness, and never calls a model or network service:

```powershell
C:\ComfyUI\.venv\Scripts\python.exe scripts\benchmark_grounding.py C:\path\to\grounding_manifest.json --output C:\path\to\grounding_report.json
```

Keep private clean controls and known silent-refusal media outside Git. Manifests can retain hashes, expected and prohibited fact IDs, aliases, frame/time facts, target profile, fixed seeds, ambiguity labels, and counterfactual variants without publishing the media itself.

Each result run records a typed experiment condition covering prompting, sampling profile, media variant, precision, telemetry, frame/token budgets, model revision, and release-candidate status. Every condition must include every declared seed. Release metrics use only strict release-candidate runs and are split into development, holdout, and overall slices; missing media, results, conditions, or seeds make the verdict `incomplete` rather than scoring a favorable subset.

The release program targets complete blocking of injected transport and invalid-ledger faults, no silent-hallucination pass on the known-failure holdout, at least 85% strict pass rate on clean controls, supported-fact recall of at least 0.80, clean false-block rate at most 15%, counterfactual discrimination of at least 90%, and temporal-order accuracy of at least 80%. The scorer returns an explicit threshold verdict and cannot pass while the 30-case/90-run release set is incomplete.

## Gated Future Research

Visual-contrastive logit steering is not a production mode and is not exposed by the Settings node. It remains gated future research until passive telemetry demonstrates meaningful real-versus-null media differences, measurable drift toward refusal or fabrication, and held-out refusal-detector performance of at least 80% recall with at most 10% false positives.

If those gates are met, the research implementation will live in a local generation engine rather than a site-packages patch. It will test paired real-media and seeded Gaussian-distorted-media branches using `L_guided = L_real + alpha(L_real - L_null)`, without refusal-word penalties or phrase banning. It must meet the benchmark's quality, confidence-interval, runtime, VRAM, schema, and no-new-silent-hallucination graduation criteria before becoming eligible for a production interface.
