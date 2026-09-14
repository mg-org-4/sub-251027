# DiffusionGemma Example Workflows

The established character-motion release workflow is:

```text
07_ltx23_character_motion_transfer.json
```

It demonstrates LTX 2.3 character motion transfer with:

- a reference image for character identity and appearance
- a source video for pose, action, timing, camera choreography, depth, composition, Canny/edge layout, and scene geometry
- DiffusionGemma prompt synthesis through `image_identity_video_control`
- Canny, Depth, and DWPose control branches for comparison

The workflow uses these DiffusionGemma nodes:

- `DiffusionGemmaModelLoader`
- `DiffusionGemmaContextHub`
- `DiffusionGemmaLTX25TargetProfile`
- `DiffusionGemmaGroundingGuardSettings` in `strict` mode
- `DiffusionGemmaCoTGenerator`
- `DiffusionGemmaJSONSplitter`
- `DiffusionGemmaGenerationGate`

Bundled media lives under:

```text
examples/assets/character_motion_transfer
```

Before running the workflow, copy that folder to:

```text
ComfyUI/input/character_motion_transfer
```

Expected workflow inputs after copy:

```text
character_motion_transfer/character_reference.png
character_motion_transfer/motion_control_video.mp4
```

Included demo output:

```text
examples/assets/character_motion_transfer/diffusiongemma_ltx_character_demo.mp4
```

The LTX prompt now passes through `DiffusionGemmaGenerationGate` before prompt encoding. A strict transport, evidence, refusal, uncertainty, or coverage failure therefore cannot reach the LTX generation core.

## MiniMax H3 Director T2VA

The native text-to-video/audio example is:

```text
09_minimax_h3_t2va_director.json
```

It converts a normal creative brief into Director's dedicated `minimax_h3_prompt`, wires the splitter's width and height outputs into native MiniMax-H3, and shares one editable duration between the Director profile and H3 frame calculation. The strict settings node is connected to the CoT Generator, and the H3 prompt, readiness flag, and metadata all pass through `DiffusionGemmaGenerationGate` before the native generator. Parallel previews show the generated H3 prompt and the splitter metadata/blocked reasons. The included preset is 20 seconds, 481 frames at 24 fps, 0.4 MP at multiple-of-32 resolution, and 25 sampling steps. This T2VA fixture has no source image/video connected, so visual grounding is `not_applicable`; connect media to its Context Hub when using it as a grounded analysis workflow.

The workflow uses the native `MiniMaxH3ImageToVideo` path plus `RHMiniMaxH3SageAttentionPatch`; install `ComfyUI_RH_MinMaxH3` and place the four model files selected in the embedded H3 group in their normal ComfyUI model folders. DiffusionGemma is set to unload after prompt generation so H3 can reclaim VRAM before sampling.

The CoT Generator uses `director_cache_mode=reuse`. Re-queueing the same Director brief/reference/profile after a restart can reuse a verified disk packet without loading the full DiffusionGemma model; the splitter and fail-closed gate still run. Select `refresh` for a deliberate new Director pass, but change the Grounding Guard/Director seed when you want a different stochastic variant—refresh with a fixed seed is reproducible. Candidate prompt preview sockets are diagnostic only and remain separate from the gated native-generation path.

The dedicated MiniMax-H3 Target Profile in either H3 example exposes `dialogue_mode`: `auto` preserves brief-requested speech without inventing it, `required` authorizes dialogue and enforces the exact `dialogue_line_count` (`1` through `12`, default `2`), and `off` prohibits speech while leaving other enabled audio available. Use `dialogue_guidance` for language, speaker, delivery, story purpose, or exact quoted lines. Native syntax keeps the speaker outside the block, for example `(S1) says: <d>[English] Exact words.</d>`; dialogue must remain in the T2VA or Ref2VA shot timeline. `required` plans speech before decorative camera coverage and is incompatible with `audio_mode=visual_only`.

For example, set `dialogue_mode=required`, `dialogue_line_count=2`, and guidance such as `English; on-screen protagonist S1; exact lines: "We made it this far." and "Now let's finish it."`. A 12-shot, 15-second target leaves only 1.25 seconds per shot on average, so use short lines and reserve longer locked or gently moving dialogue shots; reduce the shot or line count for a longer exchange.

Required-dialogue repair is surgical: unambiguous tag spelling/spacing is normalized locally, while genuinely missing lines use a small text-only patch whose model-authored words are inserted into existing shots with canonical H3 markup. The patch cannot replace the storyboard, timestamps, camera directions, reference analysis, audio sections, or provenance; invalid patches fail closed.

## MiniMax H3 Director Ref2VA

The ordered-reference example is:

```text
10_minimax_h3_ref2va_director.json
```

It uses two images in explicit `<Picture 1>` / `<Picture 2>` order, analyzes that same ordered batch with `DiffusionGemmaH3ReferenceContext`, and sends the dedicated Ref2VA prompt through `DiffusionGemmaGenerationGate` to native `MiniMaxH3ReferenceToVideo`. The strict settings node is connected to the CoT Generator. The sample creative brief explicitly requests 2D hand-drawn animation and assigns protagonist identity to Picture 1 and environment/style to Picture 2, preventing vague “inspired” wording from drifting into generic CGI subjects.

One editable duration is shared by Director planning and native frame calculation; the included setting is 20 seconds, 481 frames at 24 fps, with 25 sampling steps. The Director uses its full 2048-token output budget because the official Ref2VA representation includes subject definitions, retention analysis, and a detailed timeline in addition to audio sections. That Director budget is separate from the Grounding Guard's advanced evidence token budget, which defaults to `auto`. Shot count is also a separate Target Profile/compiler control: leave `shot_count` at `auto`, select `1` through `12`, or select `custom` and type `1` through `99` in `custom_shot_count`. Increasing `evidence_token_budget` cannot add shots. Replace both `example.png` placeholders before running and keep their socket order aligned with the reference manifest. The context node also provides one-image and two-image manifest presets; the two-image preset assigns Picture 1 to identity and Picture 2 to environment/style. The Director loader unloads before H3 sampling, and both original images feed the native Ref2VA node separately.

The CoT Generator is preset to `director_cache_mode=reuse`; reference tensor hashes, order/roles, duration, shot count, audio/negative policy, guard settings, checkpoint identity, and Director implementation all participate in the key. The 2048-token maximum is unchanged. Inspect `final_json.metadata.director_runtime` to compare cold load, NVFP4 bridge, media preparation, prefill, generation, validation/retry, and unload costs before changing the budget or runtime kernels.

## Ideogram 4 strict prompt routing

The local, generator-independent Ideogram acceptance example is:

```text
11_ideogram4_strict_grounded_prompt.json
```

Replace `example.png` with a source image before queueing. The workflow uses the dedicated `DiffusionGemmaIdeogram4TargetProfile`, whose surface contains only image-relevant aspect ratio, render style, exact-text, JSON/prose, and negative-prompt controls—there are no audio or duration widgets. It uses strict grounding, routes `ideogram_prompt`, `ready_for_generation`, and splitter metadata through the Generation Gate, and intentionally ends at a harmless prompt preview. It performs no cloud call and bundles no Ideogram generator; connect the gated output to a separately validated local consumer if desired.

These strict workflows are serialization-checked acceptance fixtures. They are not claims that a live GPU render was completed on the current machine.

## MiniMax H3 Ref2VA Grounding Guard drop-in

For an existing MiniMax H3 Ref2VA graph, load:

```text
12_minimax_h3_ref2va_grounding_guard_drop_in.json
```

This compact workflow ends at a harmless preview and is designed to be copied into another graph. Connect only the `DiffusionGemmaGenerationGate` output to the native MiniMax H3 `prompt` input. Keep the original reference images connected to MiniMax in the same order used by the Grounding Guard analysis batch, and keep the manifest's `<Picture N>` entries synchronized with the attached assets. The included note explains optional identity-plus-control-video wiring and which duration/resolution values can be shared with the native graph.

## LTX-2.5 I2V Director

The clean single-model image-to-video workflow is:

```text
13_ltx25_i2v_director.json
```

It connects one source image to both `DiffusionGemmaContextHub` and native LTX first-frame conditioning, leaves the Context Hub video and last-frame sockets empty, and lets the dedicated LTX target resolve Auto mode to I2V. The splitter controls native width and height; the included 16:9 / 0.5 MP baseline resolves to 960×544. The positive prompt passes through the standard fail-closed `DiffusionGemmaGenerationGate`, while the negative prompt comes directly from the splitter's fail-closed negative output. Candidate prompt previews are diagnostic only and never feed native generation.

## LTX-2.5 I2V vs MiniMax-H3 Ref2VA comparison

The side-by-side comparison workflow is:

```text
14_ltx25_i2v_vs_minimax_h3_ref2va_comparison.json
```

It shares one creative brief, duration, source image, and DiffusionGemma loader while keeping two independent model-specific Director paths. LTX uses `DiffusionGemmaContextHub` with the image as its first-frame anchor. MiniMax-H3 uses `DiffusionGemmaH3ReferenceContext` with the one-image Ref2VA manifest preset, and the same image is also connected to native `ref_image_0`. Each splitter passes through a `DiffusionGemma Branch Generation Gate`: an invalid branch remains fail-closed, its status preview explains why, and a valid comparison branch can still render instead of being canceled by the other gate's exception. Gate status describes compiled-packet readiness; the separate Grounding status/report previews describe evidence auditing.

All LTX creativity modes work in the I2V branch. The first frame fixes its opening pose, framing, and camera geometry, not the future camera trajectory. Explicit camera motion or a static/locked request always wins. With unspecified motion, faithful chooses the least-invasive suitable path; editorial, cinematic, concept-art, and wild may author progressively more expressive physically continuous choreography without adding cuts or violating the anchor. Raise `creative_strength` above the faithful band to increase camera-path ambition.

For one-branch tuning, first right-click that branch's gate-status `Preview Any` and choose **Run Branch** as a cheap preflight. If it reports ready, right-click the matching terminal `SaveVideo` and choose **Run Branch** (or Ctrl-select the status preview and save node, then use the selection-toolbar play button). Queue LTX and H3 as two separate jobs when CUDA, OOM, or sampler-failure isolation matters. The normal global Run button selects all active outputs in both branches. Partial targeting excludes the sibling branch; with each CoT Generator on `director_cache_mode=reuse`, the completed preflight also stays cached for the subsequent render. `refresh` and `off` deliberately rerun the selected Director every queue.

The comparison baseline sets both Grounding Guards to `audit`, which keeps evidence diagnostics visible without blocking either native render; switch both to `strict` only for a later compliance comparison. The H3 Director uses a 2048-token budget because Ref2VA's six required sections are substantially larger than the LTX caption contract.

The H3 Reference Context sets `expected_subject_count=1`, so its Ref2VA packet must contain exactly one semantic `<Subject 1>` definition. This is independent of the single `<Picture 1>` asset: Picture labels count ordered reference inputs, while Subject labels count independently tracked semantic units. The included briefs keep one hero as the only independently tracked Subject and fold ordinary props, environment, lighting, composition, style, actions, and effects into that Subject and its timeline.

The included raster settings are intentionally model-specific: LTX is 1.0 MP at 16:9 and H3 is 0.4 MP at 16:9. Match them manually if the hardware and comparison design require identical output dimensions. The conditioning is still not identical: LTX I2V anchors an opening frame, while H3 Ref2VA may recompose its full-sequence visual reference.

## MiniMax-H3 synchronized music-video production V6

The checked-in operational production workflow is:

```text
15_minimax_h3_ref2va_music_video_v6.json
```

It combines the accumulated MiniMax-H3 production upgrades in one graph: two same-person identity references, optional previous-lane continuity relay, one to four duration-bounded H3 generation lanes, multiple native `[Shot N]` blocks inside each lane, exact-frame assembly, measured soundtrack timing, Natural/Dance/Lyrics performance modes, local timed-lyrics analysis, and the pristine final soundtrack mux. The graph also includes the tested H3 Turbo/fast path and the H3-specific expressive camera contract.

`SONG SOURCE` selects either the existing lazy ACE-Step generation path or a custom file from `UPLOAD SONG`. Source passthrough under the separate production-concept control still means ACE blueprint passthrough; it is not the file-upload choice. Uploaded music reuses the same technical-integrity checks, excerpt selection, waveform hash, audio guide, lane planner, and final mux.

Replace both saved `LoadImage` choices with your own primary identity image and same-person contact sheet. Keep Picture 1 and Picture 2 as identity authorities for the same semantic subject. Continuity relay is off by default; enable Previous lane tail only when pose/spatial continuity is worth the additional identity-drift risk. The saved upload widget is blank and Generate with ACE-Step is selected, so loading the example does not require a custom song.

The repository copy is the validated V6 graph used by the Desktop and ComfyUI deployments at the time of this changelog. Its migration marker and canonical JSON SHA-256 are regression-tested without depending on those mutable external copies or platform-specific line endings.

## Separate MiniMax-H3 Advertisement production path

Advertisement production is additive and does not replace or migrate V6. The user-editable workflow has a fresh revision-1 graph identity and no shared mutable subgraph UUID with V6:

```text
16_minimax_h3_ref2va_advertisement_music3_v1.json
```

Its governed API-format fixture is:

```text
api/17_minimax_h3_ref2va_advertisement_music3_glowbloom_v1_api.json
```

Its companion manifest records graph hashes, model selections, reference roles, deliverables, adaptation status, and the default honest-QA posture. The earlier `api/16_minimax_h3_ref2va_advertisement_glowbloom_v1_api.json` remains as field-test lineage; new Advertisement work should use the Music 3 fixture rather than treating the earlier graph as the hardened contract.

The saved reference contract is:

```text
<Picture 1> performer hero identity
<Picture 2> same-performer contact sheet
<Picture 3> independent product/package contact sheet
<Picture 4> hash-verified, downscaled copy of the exact retained prior-lane final frame; continuity only
```

Select three images in that order. `DiffusionGemmaAdvertisementReferenceAssetPrep` normalizes them and computes the three SHA-256 values automatically. Picture 3 retains a separate product budget and must not be merged with either performer reference. Picture 4 is reserved by the contract and is populated only by a hash-verified, downscaled copy of the previous H3 lane's exact retained final frame; it is never a source identity.

`ADVERTISEMENT CONTROLS` is the only shared duration, native-shot, performance, aspect, Ref2VA mode, H3 audio/dialogue policy, excerpt-start, generation-model, lane-ceiling, and deliverable authority in Advertisement v1. The release intentionally locks a 30-second 9:16 master with derived tail-selected 15-second and 6-second cutdowns, keeps one shared shot count, drives the splitter aspect, and exposes Natural or Dance performance. Alternate reference layouts, non-relay seam modes, Lyrics without timed evidence, and VO without an input path are hidden instead of offered as broken combinations.

`SOUNDTRACK SOURCE` has three lazy origins:

- **MiniMax Music 3** is the default generated path.
- **Upload song** accepts a user-selected soundtrack without evaluating either generator.
- **Legacy ACE-Step** preserves the earlier generated-music route as a fallback.

The separate soundtrack content policy is `Instrumental`, `Vocal`, or `Auto`. Instrumental Music 3 output is not rejected for lacking a vocal proxy; Vocal mode requires vocal evidence and governed tagged lyrics. The 30-second preset requests 35 seconds from Music 3 so the selector can lock an exact 30-second candidate window. Its verified local model selections are:

```text
models/text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors
models/diffusion_models/minimax_music3_dit_fp16.safetensors
models/vae/minimax_music3_dav.safetensors
```

The combined editable graph retains the Legacy ACE-Step fallback. ComfyUI may validate linked lazy loader/model choices before executing the selected source, so the saved ACE assets and `qwen_4b_ace15.safetensors` text encoder must still be discoverable even when MiniMax Music 3 is selected. The Music3-only API fixture avoids this fallback dependency.

The exact 99-node API fixture is runtime-proven end to end on an isolated ComfyUI `0.33.1` environment: both 15-second H3 lanes, retained-tail Picture 4 relay, deterministic end card, all diagnostic roots, and exact 30-second/15-second/6-second review media completed without node errors. The master was 720 frames at 480x864/24 fps with exactly 30.000 seconds of stereo audio; the cutdowns were 360 and 144 frames with exact matching audio clocks. A sampled visual contact-sheet review found stable performer/wardrobe and recognizable package continuity. The formal QA status correctly remains blocked because identity, package, copy, and audio sync have not all received evidence-backed human or measured approval.

The API fixture is intentionally MiniMax Music 3-only; Upload song and Legacy ACE-Step exist only in the combined editable UI workflow. Both artifacts also require the native H3 files below:

```text
models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
models/vae/minimax_h3_video_vae_fp16.safetensors
models/vae/minimax_h3_audio_vae_fp32.safetensors
models/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
```

The Director loader is saved as `models/LLM/diffusiongemma-26B-A4B-it-NVFP4`, relative to the ComfyUI application directory. Edit that field to a full local repo-folder path if the process uses a different working directory or model layout.

H3 receives only the selected soundtrack as its motion guide. The reusable mixer supports separate non-diegetic voice-over only in the final delivery mix and never in H3 conditioning, but Advertisement v1 locks VO to `None` until an actual upload/loader path is connected. Finishing replaces the governed master tail with a deterministic exact-copy end card, then emits actual 15-second and 6-second frame/audio cutdowns alongside the 30-second 9:16 master. Requested 1:1 and 16:9 adaptations remain explicitly unrendered until a semantic reframe renderer exists.

The runtime-proven MiniMax Music 3 candidate passed technical QC but produced a signal-estimated `87.890625` BPM instead of the requested `122`; the selector records this as an advisory rather than pretending the generative tempo is exact. H3 still follows the selected waveform itself. Use Upload song or a stricter candidate lock when the numerical BPM is a hard campaign requirement.

The QA node distinguishes `pass`, `fail`, and `not_measured`; a pass without evidence is downgraded to `not_measured`. Automatic MP4 and soundtrack saves are labeled review drafts rather than delivery approvals. Technical assembly can pass while performer identity, product identity, copy legibility, or audio sync remains unmeasured. Generated in-scene packaging text is not guaranteed, exact copy is governed only on the deterministic end card, and stochastic performer/product drift still requires human or measured review. The fictional GlowBloom inputs and rendered campaign media are not bundled.

## LTX 2.3 Reframe with selectable audio

The provisional, unexported repository prototype is:

```text
gemmaREFRAME_Experimental_audio_on-off.json
```

The current package entry point does not register the `LTXReframe*` nodes, and the retired layout frontend is not shipped. Keep these reframe artifacts separate from the stable Director examples when preparing a Registry release.

Its **Use Custom Vocal Audio?** switch defaults to off:

- **Off:** the source video's audio is used for both LTX audio conditioning and the final video. Videos without an audio track use duration-matched silence.
- **On:** upload a custom audio track, choose its start time in **Custom Track Window**, and use the MelBandRoFormer vocal stem. Long tracks are trimmed and short tracks are padded with silence to match the effective video range.

Required custom-node dependencies:

- `ComfyUI-MelBandRoFormer`
- `comfyui-kjnodes`
- `comfy-mtb`

Required model:

```text
ComfyUI/models/diffusion_models/MelBandRoformer_fp16.safetensors
```

No custom music is bundled. Source-audio mode can remain selected without choosing a custom file; custom-audio mode reports a clear error until an audio file is uploaded or selected.
