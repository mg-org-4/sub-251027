# IllustriousAutoOutpaint

Single-node, Illustrious-first outpainting for SDXL-family models. Expands an image outward with soft, mask-based denoising, optional **two-stage growth**, optional **two-phase schedule (grow → refine)**, optional **refiner pass**, and a subtle **seam-toning micro-pass** to hide borders.

---

## What does it do?

* Pads your latent on any side(s) and **adds noise only in the new border**, preserving the original interior.
* Uses a **feathered denoise mask** to avoid hard seams.
* Optionally grows big canvases in two **stages** (half pad → full pad) to reduce stretching.
* Optionally runs two **phases** per stage:

  * **Grow:** unmasked, high-denoise pass to let structure appear in the border.
  * **Refine:** masked, lower-denoise pass to stabilize and clean the seam while protecting the interior.
* Optional **refiner model** pass after outpainting for final polish.
* Optional **seam-toning** post-decode to equalize brightness/color along the transition band.
* Deterministic border noise seeded by the node’s `seed` for reproducible outpaints.

---

## Inputs

### Core

* **image** *(IMAGE)*: The starting image to outpaint.
* **model** *(MODEL)*: UNet diffusion model (Illustrious/SDXL).
* **clip** *(CLIP)*: Matching CLIP for text conditioning.
* **vae** *(VAE)*: VAE for encode/decode.
* **prompt / negative** *(STRING)*: Positive and negative prompts.

### Canvas expansion (pixels, image space)

* **expand\_left / right / top / bottom** *(INT)*: How much to grow each edge. Internally mapped to latent space (\~÷8) with ceil.

### Sampling & quality

* **steps** *(INT)*: Denoising steps for single-pass or per-stage passes.
* **cfg** *(FLOAT)*: Classifier-free guidance.
* **sampler\_name** *(CHOICE)*: e.g., `euler_ancestral`, `dpmpp_2m`, etc.
* **scheduler** *(CHOICE)*: `normal`, `karras`, `exponential`, `sgm_uniform`.
* **seed** *(INT)*: Random seed (also used for deterministic border noise).
* **denoise** *(FLOAT)*: Strength for masked/pass-through modes (0–1).

### Masking & noise

* **noise\_outside** *(FLOAT)*: Scales how much noise is added/allowed in the border (0–1).
* **feather\_px** *(INT)*: Feather width (image-space) for the soft border mask (converted to latent scale).

### Scene hint (optional)

* **scene\_hint** *(CHOICE: none/landscape/interior/street/portrait)*: Appends a small, safe scene bias to your prompt to encourage coherent environmental context.

### Growth strategy

* **two\_stage\_growth** *(BOOLEAN)*: If enabled and pads are large, runs half-pad then full-pad (two stages) to reduce context shock.

### Two-phase schedule (per stage)

* **two\_phase\_schedule** *(BOOLEAN)*: Enable the **grow → refine** sequence.
* **grow\_steps\_frac** *(FLOAT)*: Fraction of `steps` for the grow phase (unmasked).
* **grow\_denoise** *(FLOAT)*: Denoise strength for grow phase (high, e.g., 0.9).
* **refine\_denoise** *(FLOAT)*: Denoise strength for refine phase (lower, e.g., 0.55).

### Seam-toning (post-decode)

* **seam\_toning** *(BOOLEAN)*: Enable feathered luminance/chroma equalization on the seam.
* **seam\_tone\_strength** *(FLOAT)*: Strength of the seam adjustment.
* **seam\_feather\_px** *(INT)*: Feather width (image space) for seam blending.

### Refiner (optional)

* **use\_refiner** *(BOOLEAN)*: Run a final low-denoise pass with a refiner model.
* **refiner\_model** *(MODEL)*: Refiner UNet.
* **refiner\_steps** *(INT)*, **refiner\_denoise** *(FLOAT)*: Refiner settings (denoise typically \~0.2).

### Advanced text weighting (optional; BNK encoder if present)

* **weight\_interpretation** *(CHOICE)*: `comfy`, `A1111`, `compel`, `comfy++`, `down_weight`.
* **token\_normalization** *(CHOICE)*: `none`, `mean`, `length`, `length+mean`.

---

## Outputs

* **image** *(IMAGE)*: The outpainted image (post-decode; seam-toned if enabled).
* **latent** *(LATENT)*: The final latent (useful for further refinement or decode variants).

---

## How it works (under the hood)

1. **Encode**: Input image → latent.
2. **Pad**: Create a larger latent and place the original in the center.
3. **Border mask**: Build a **1×1×H×W soft mask** that’s 1 in new borders, 0 inside; feather to avoid seams.
4. **Seed new noise**: Deterministically add Gaussian noise **only in the border** (interior remains the original latent).
5. **Denoise**:

   * **Single pass** *(default)*: Masked denoising with optional two-stage growth for large pads.
   * **Two-phase** *(optional)*:
     **Grow** (unmasked, high denoise) → **Refine** (masked, lower denoise).
6. **(Optional) Refiner**: Low-denoise pass on the result using a second model.
7. **Decode** via VAE.
8. **(Optional) Seam-toning**: Feathered luma/chroma equalization across the seam band to hide residual edges.

---

## When to enable the toggles

* **two\_stage\_growth = ON**:
  Big expansions (≥ \~384 px latent-equivalent per side). Reduces stretching/warping by giving the model an intermediate context.

* **two\_phase\_schedule = ON**:
  Large, especially **one-sided** outpaints or blank-canvas growth. Grow phase invents structure; refine phase locks it to the original content.

* **seam\_toning = ON (default)**:
  Low-risk quality polish; recommended always. It only touches a thin band around the seam.

---

## Artist Tips

* For cinematic panorama extensions, try:
  `sampler_name = euler_ancestral`, `scheduler = karras`, `cfg ~ 5–6.5`, `two_phase_schedule = ON`, `grow_steps_frac ≈ 0.45`, `grow_denoise ≈ 0.9`, `refine_denoise ≈ 0.55`, `feather_px 96–128`.

* If borders look too “empty”, **raise `noise_outside`** and/or **increase `grow_denoise`**.

* If the interior shifts more than you like, ensure **masked passes** are used (default), keep `refine_denoise ≤ 0.6`, and let **seam\_toning** stay on.

* Try a **refiner model** to add micro-detail after structure is set—keep refiner denoise modest (≈0.2).

* Pair with **regional conditioning** nodes to place different prompts in left/right/top/bottom bands for guided world-building.

---

## Performance Notes

* Two-stage growth and two-phase schedule add passes → more compute.
  Use them when quality matters (big or tricky expansions).

* The node sends masks as **\[B,1,H,W]** and seeds border noise deterministically from `seed`. Most Comfy variants will respect `denoise_mask` directly.

---

## Troubleshooting

* **“The border doesn’t fill / looks flat.”**
  Make sure `noise_outside > 0`. Turn **two\_phase\_schedule ON** and keep **grow\_denoise high (\~0.9)**. Increase `steps` modestly.

* **“I see a faint seam.”**
  Increase `feather_px` (e.g., 96–128) and keep **seam\_toning ON**. Slightly raise `seam_tone_strength` (0.3–0.45).

* **“Interior changed too much.”**
  Ensure refine pass is **masked** and lower `refine_denoise` (0.4–0.6). Avoid very high CFG in refine.

* **“Banding or mismatch with refiner.”**
  Use the same sampler/scheduler for refiner as main, or reduce refiner CFG slightly.

---

## Example Workflows

### Quick panorama widen

1. Load image → **AutoOutpaint**:

   * `expand_left=0, expand_right=1024, expand_top=0, expand_bottom=0`
   * `two_phase_schedule=ON`, `grow_steps_frac=0.45`, `grow_denoise=0.9`, `refine_denoise=0.55`
   * `sampler=euler_ancestral`, `scheduler=karras`, `steps=30`, `cfg=5.5`
2. Decode result (seam-toning ON by default).

### Big canvas in two stages

1. Set `expand_*` large (e.g., 768–1280).
2. Enable **two\_stage\_growth=ON** + **two\_phase\_schedule=ON**.
3. Run; optional **refiner** at the end.

### Character to environment

1. Start with a portrait.
2. `expand_left/right=900`, `scene_hint=portrait` (or `street/interior` depending on context).
3. Two-phase ON; higher `feather_px` (96–128).
4. Seam-toning ON.

---
