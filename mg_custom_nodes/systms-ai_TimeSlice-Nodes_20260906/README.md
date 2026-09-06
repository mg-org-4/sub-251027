# TimeSlice Nodes — SYSTMS

Temporal slit-scan / time-displacement effects for ComfyUI. A frame is divided into parallel bands ("slices") and each band is sampled from a *different frame* in the sequence — like a rolling shutter you can art-direct.

The pack contains three nodes built on a shared slicing engine, so angle, spacing, blending and masking behave identically across all of them:

| Node | What it does |
|---|---|
| **TimeSlice Effect** | Classic cascading slit-scan — each slice is offset further in time than the last |
| **TimeSlice Wave** | Animated variant — slice offsets ripple over time following a travelling sine wave |
| **TimeSlice Dual** | Two-input variant — interleaves slices from two clips, each with independent time offsets |

All three nodes can be found under **Systms → effects**.

---

## Installation

**ComfyUI Manager** — search for "TimeSlice Nodes" and click Install.

**Manual (git)** — clone into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/systms-ai/TimeSlice-Nodes.git
```

Then restart ComfyUI. The only dependency is `numpy` (already present in ComfyUI); `torch` is provided by the ComfyUI runtime.

---

## Shared concepts

### Slicing geometry

- **num_slices** — how many bands the frame is divided into. More slices = finer, smoother displacement; fewer = bold, blocky steps.
- **angle** — slice orientation in degrees. `0` = vertical slices, `90` = horizontal, `45` = diagonal. Any angle works.
- **spacing** — how slice widths are distributed: `linear` (equal widths), `ease_in` / `ease_out` / `ease_in_out` (slices compress toward one side or the middle), or `random` (seeded irregular widths).

### Time handling

- **loop_mode** — what happens when a slice's offset reaches past the ends of the clip:
  - `loop` — wraps around to the other end (seamless for looping clips)
  - `ping_pong` — bounces back and forth
  - `trim` — drops frames at the ends so no slice ever has to wrap. Output is shorter than the input.

### Finishing

- **blend_width** — soft cross-fade (in pixels) across slice boundaries. `0` = hard edges. Fully vectorised, so high slice counts and wide blends cost the same as low ones.
- **mask** (optional) — restricts the effect to the white area of a mask; the rest of the frame stays untouched original footage.
- **seed** — drives `random` spacing and slice jitter, for repeatable results.

### Outputs

Every node returns four sockets:

| Output | Type | Description |
|---|---|---|
| `images` | IMAGE | The sliced frames |
| `mask` | MASK | The input mask passed through (or solid white if none) |
| `num_frames` | INT | Output frame count — useful downstream when using `trim` |
| `slice_edges` | MASK | The boundary lines as a separate mask, for feeding glow / glitch overlays |

---

## TimeSlice Effect

The core cascade. Slice 0 shows the current frame, and each subsequent slice is pushed further back (or forward) in time.

**Key parameters**

- **offset** — frame step between neighbouring slices. `1` = subtle smear, higher = stretched time trails. Negative values reverse the time direction.
- **direction** — which slices carry the most displacement: `forward`, `reverse`, `center_out` (still in the middle, displaced at the edges) or `edges_in` (the inverse).
- **cascade_curve** — how the offset grows across slices: `linear`, `ease_in`, `ease_out`, `ease_in_out` or `exponential`. Exponential keeps most of the frame coherent and whips the far slices away.
- **slice_jitter** — seeded random ± frames added per slice, for a glitchier, less mechanical cascade.

**Animation (optional)**

Enable **animate** to interpolate `num_slices`, `offset` and `angle` from their start values to `num_slices_end` / `offset_end` / `angle_end` over the clip, shaped by **temporal_curve**. By default the animation spans the full clip length; set **frame_count** to complete it over a specific number of frames instead.

---

## TimeSlice Wave

Instead of a fixed cascade, slice offsets follow a sine wave that travels across the frame over time — an undulating, liquid temporal ripple.

**Key parameters**

- **max_offset** — peak frame displacement at the wave crest.
- **wave_count** — number of full sine cycles across the slices (spatial frequency: how many ripples are visible at once).
- **wave_speed** — number of full wave cycles over the whole clip (temporal frequency: how fast the ripple travels). Higher = faster. Whole-number values loop seamlessly with `loop_mode: loop`.

**Note on trim mode:** because the wave swings both backwards and forwards in time, `trim` removes `max_offset` frames from *both* ends of the clip, so the output is `input − 2 × max_offset` frames.

---

## TimeSlice Dual

Slices alternate between two video inputs — built for transitions, split-reality looks and A/B mashups. Input B is automatically resized to match A.

**Key parameters**

- **offset_a / offset_b** — independent cascade step for slices drawn from each source.
- **pattern** — how slices are assigned to sources: `alternate` (ABAB), `pairs` (AABB repeating), `aba`, `aabb`, `abba`.
- **mix** — balance between sources. `0.5` keeps the pattern as-is; pushing toward `0` or `1` reassigns slices (seeded) so one source dominates.
- Supports the same `direction`, `cascade_curve`, `slice_jitter`, masking and start→end animation as TimeSlice Effect, with independent `offset_a_end` / `offset_b_end` targets.

---

## Quick recipes

- **Clean classic slit-scan** — Effect: `num_slices 40+`, `offset 1`, `angle 90`, everything else default.
- **Glitch smear** — Effect: `spacing random`, `slice_jitter 4–10`, with `slice_edges` fed into a glow.
- **Liquid ripple loop** — Wave: `wave_count 1–2`, `wave_speed` a whole number, `loop_mode loop`, `blend_width 20–60` for a smooth undulation that cycles perfectly.
- **Two-clip transition** — Dual: animate from `mix`-heavy A to B-dominant offsets, or keyframe `num_slices` low→high for a shatter-style handover.
- **Effect only on the subject** — feed a subject mask into any node's `mask` input; the background stays clean.

## Performance notes

The slicing and blending are fully vectorised — blend cost does not scale with `num_slices` or `blend_width`, so even 2000 slices with wide blends runs fine. Memory is driven by clip size (`frames × height × width`), the same as any video node: very long, high-resolution clips need RAM regardless of settings.

---

## License

MIT © 2026 SYSTMS. See [LICENSE](LICENSE).
