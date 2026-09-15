# FL Krea Reference Control — proposed scope

Status: implemented as FL Krea Reference and FL Krea Reference Guider. See [usage](krea-reference-control.md). Visual validation results are recorded separately; the initial scope below describes the design and acceptance targets.

Implementation finding: independent full-image prediction blending still copied source portraits and panel layouts in the three-seed comparison. The delivered node therefore adds Context mode (default), which retains image-informed text states after the vision block, and Full mode, which retains visual tokens as well. Both retain all twelve layers. Context removed the unwanted portraits/panels in that test set; this is not a universal guarantee.

## Goal

Generate one coherent image from a scene prompt and independently controlled references. Let the user request the linework from one image, the palette from another, and the subject or composition from another, with adjustable contribution and timing.

Initial target: the installed Krea 2 Turbo model, Qwen3-VL-4B Krea encoder, and the user's 8-step Euler/simple workflow. Role instructions express intent; they are not trained, disentangled style/identity controls.

## Findings driving the design

- Rebalance's `compile_edit` serializes multiple images into one Qwen conversation as `Picture 1`, `Picture 2`, etc. It does not build a pixel grid or independently mix image contributions. A generated grid is a possible model interpretation, not an explicit image assembly operation in this code.
- `Krea2EditRebalance` retains only one of twelve conditioning taps. Its complementary reference branch has no effect under the current per-band projection math. This was reproduced with synthetic tensors.
- Its private pass cache omits image content and CLIP identity. Different same-size images can return identical cached conditioning. This was also reproduced.
- Both existing nodes use vision-language conditioning, without reference latents. The new node must not advertise exact identity, spatial, or pixel preservation on that basis.
- Krea already uses ComfyUI's optimized attention. A custom guider can combine model predictions using existing sampling interfaces without replacing attention implementations or changing core model code.

## User-facing nodes

Two node types keep individual reference settings beside each image while keeping the sampling controls centralized.

### FL Krea Reference

Produces an `FL_KREA_REFERENCE` descriptor containing the image and its interpretation settings. It performs no model encoding itself.

| Input | Meaning |
| --- | --- |
| image | One reference image. A batch with more than one image raises a clear error; use existing batch selection/splitting nodes. |
| enabled | Exclude this reference completely when disabled. |
| role | Style, palette, subject, composition, or custom. Default: style. |
| instruction | Optional description of what to borrow and what to leave behind. Editable text supplements the role. |
| weight | Nonnegative direct strength of the reference contribution. Zero excludes the reference. Initial neutral value: 1. |
| resolution | Longest-side limit: 256, 512, 1024, or 1280. Initial default: 512; preserve aspect ratio. |
| start / end | Advanced activation window in diffusion progress, from 0 to 1. Default: full run. |
| fade | Advanced fade fraction at each edge of the activation window; capped to prevent overlapping ramps. Default: no fade. |

No arbitrary four-reference limit. Practical cost is shown in documentation. Initial performance validation covers one, two, and four references.

Roles generate explicit instructions, not hidden numerical layer presets. For example, style requests medium, linework, shading, and texture while the scene prompt defines the subject. Composition requests broad arrangement without claiming pose locking. Custom uses the user's instruction directly.

Source cropping uses existing image nodes. Output-region masks and automatic subject extraction are outside the first version.

### FL Krea Reference Guider

Inputs: MODEL, CLIP, scene prompt, an optional `io.Autogrow` collection of reference descriptors with minimum zero, and overall reference influence.

Output: GUIDER, for the existing `SamplerCustomAdvanced` node. Noise seed, sampler, sigma schedule, latent size, and VAE decode stay in standard ComfyUI nodes. No model or image pass-through outputs.

Overall influence initially spans 0 to 1, with a provisional default of 0.7. Zero is the plain text baseline; one applies the entered reference weights wherever references are fully active. Combined contributions above one extrapolate beyond the baseline.

The example workflow replaces KSampler with RandomNoise, KSamplerSelect, BasicScheduler, and SamplerCustomAdvanced, retaining Euler/simple, eight steps, fixed seed, and existing latent/decode nodes. This is an intentional sampling interface change; a CONDITIONING-only output cannot provide this independent prediction blend to a stock KSampler.

## Proposed sampling behavior

1. Encode the scene prompt as a text-only baseline using Krea's native template.
2. Independently encode the same scene prompt with each enabled reference and its role instruction. Each reference branch sees only one image. Preserve all twelve conditioning taps and their metadata.
3. At each model evaluation, calculate the baseline prediction and each active reference prediction at the same latent and sigma.
4. Combine predictions in their shared latent coordinates. Do not average text or visual embeddings by token index: different sequences do not have reliable token correspondence.

Let `D0` be the baseline prediction, `Di` each reference prediction, `S` overall influence, `wi` direct enabled weights, and `ei(t)` the reference activation envelope:

`D = D0 + S * sum(wi * ei(t) * (Di - D0))`

This is a blend of denoiser predictions, not an image crossfade or a blend of random seeds. The whole reference-conditioned branch contributes, including its role instruction. It is not a mathematically isolated image-only effect.

Apply positive enabled weights directly, without normalization. Fading one reference therefore reduces its contribution without automatically boosting another. With no references, all weights zero, or S=0, skip reference encoding/evaluation and follow the ordinary text-only CFG-1 path. Skip baseline evaluation when its coefficient is exactly zero. Disabled/zero-weight entries never influence other weights or prompts.

Schedules use ComfyUI's model sampling conversion from progress boundaries to sigma. Interpolate fades between those converted boundaries, documenting that interpolation occurs in sigma space. Do not count Python callbacks as diffusion steps. Fractional denoise and samplers with multiple evaluations must follow the same sigma-based contract.

Independent branches remove the shared multi-picture input as a source of panel layout. They cannot guarantee that no grid will be generated, especially when a reference itself contains panels. The effect of weighting on visual style is not guaranteed to be linear or monotonic.

## Implementation boundaries and cost

- Add one Python module under `nodes/conditioning/`, register two nodes through the pack's existing mappings, and add focused tests plus an example workflow. Reuse the installed V3 schema patterns for Autogrow.
- Use the existing CFGGuider lifecycle and `calc_cond_batch` contract. Register every conditioning branch through the guider so ComfyUI owns hook preparation, devices, loading, offloading, and cleanup.
- Evaluate branches sequentially initially and accumulate predictions to avoid retaining N full prediction tensors. No custom attention kernels, backend selection, raw model mutation, extra model copies, dependencies, downloads, or internet access.
- Keep the twelve-tap encoder layout intact. Validate the supported Krea model/encoder combination at the node boundary, with an actionable mismatch error.
- Scope is CFG-1 Krea Turbo. Additional negative-prompt guidance, other Krea variants, and interactions with guidance-modifying third-party patches need separate validation. Preserve supported standard hooks and report unsupported combinations explicitly.
- Worst-case denoiser work is N+1 evaluations per sampler evaluation: four references plus the baseline can cost approximately five evaluations. Wall-clock time and VRAM must be measured, not inferred from this count alone.
- No private persistent tensor cache. Ordinary ComfyUI graph caching applies; changing settings that invalidate the guider can initially re-encode its references. Future encoding reuse requires a separate measured need and correct ownership/invalidation.
- The descriptor and guider outputs may hold their execution data under normal graph caching. Do not introduce an additional store that retains images, embeddings, or predictions across executions.

## Prototype and acceptance gates

First establish whether independent prediction blending visibly improves reference mixing. Do this before polishing controls or adding presets.

Use a small fixed-seed comparison set: the current dog/illustration case, two clearly different visual styles, palette plus style, subject plus style, and references that contain panels. Compare native text-only, current joint Encode Rebalance, and the proposed blend, holding prompt, seed, size, model, and sampler constant. Use at least three seeds for each selected comparison.

Review prompt adherence, contribution from both references, unwanted panels/duplicates, source-subject leakage, recognizable requested style features, artifacts, latency, and peak VRAM. Record comparisons for user review; do not label a subjective quality improvement as proven from synthetic tests.

Required numerical and integration checks:

- No references and S=0 match the native text-only baseline within the established numerical tolerance.
- One reference at S=1 with a full envelope matches that branch alone; the baseline can be skipped.
- Removing a zero-weight or disabled reference leaves the result unchanged.
- Swapping same-size image contents invalidates upstream execution and changes encoded conditioning when the images encode differently; changing CLIP invalidates encoding as well.
- Reordering complete reference descriptors preserves the mixture within floating-point tolerance.
- Doubling weights doubles the reference difference from the baseline at a fixed latent and sigma. A lone reference at 0.05 contributes 5% as much as at 1, before overall influence.
- Weights, schedules, and short/fractional-denoise runs follow the documented formula. Validate zero-width windows and out-of-range inputs at the boundary.
- Multiple differently sized images and prompts with different sequence lengths work without token-index averaging or padding-based correspondence assumptions.
- Repeated execution, cancellation, and sampler failure leave no additional model-owned cache or patch behind.
- Existing optimized attention, model patches, supported dtype/device handling, and low-VRAM execution continue through ComfyUI's lifecycle. Run available hardware checks; explicitly record untested backends.

If the blend produces washed-out styles, weak identities, or persistent competing compositions, revisit the conditioning strategy before release. Do not compensate with unexplained layer multipliers or promise a grid-prevention switch.

## Deferred work

Per-region destination masks, trained identity preservation, latent-reference control, automatic image captioning, per-layer expert knobs, extrapolating reference strength, a custom canvas editor, and a faster attention-based mode are outside v1. Each needs its own evidence and implementation contract.

## Deliverables

Two registered nodes, focused regression tests, a concise usage guide, an example replacing the user's current sampler path, and fixed-seed visual comparisons with timing/VRAM measurements. No existing Rebalance code or user workflow is modified as part of this scope document.
