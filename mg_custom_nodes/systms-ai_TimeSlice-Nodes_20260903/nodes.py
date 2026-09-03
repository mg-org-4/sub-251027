import math

import numpy as np
import torch


# ════════════════════════════════════════════════════════════════════
#  Shared helpers (no torch needed for the pure-maths ones)
# ════════════════════════════════════════════════════════════════════

def _ease(t, mode: str):
    """Apply an easing curve to t (float or np.ndarray), t in [0, 1]."""
    if mode == "linear":
        return t
    if mode == "ease_in":
        return t * t
    if mode == "ease_out":
        return 1.0 - (1.0 - t) ** 2
    if mode == "ease_in_out":
        return np.where(t < 0.5, 2.0 * t * t, 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0)
    if mode == "exponential":
        k = 3.0
        return np.expm1(k * t) / np.expm1(k)
    return t


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _boundary_ts(num_slices: int, spacing: str, seed: int) -> np.ndarray:
    """Normalised cut positions in [0, 1], length num_slices + 1."""
    t = np.linspace(0.0, 1.0, num_slices + 1)

    if spacing == "ease_in":
        t = _ease(t, "ease_in")
    elif spacing == "ease_out":
        t = _ease(t, "ease_out")
    elif spacing == "ease_in_out":
        t = _ease(t, "ease_in_out")
    elif spacing == "random":
        rng = np.random.RandomState(seed)
        interior = np.sort(rng.uniform(0.0, 1.0, max(0, num_slices - 1)))
        t = np.concatenate([[0.0], interior, [1.0]])

    t = np.asarray(t, dtype=np.float64)
    t[0], t[-1] = 0.0, 1.0
    # enforce monotonic non-decreasing
    np.maximum.accumulate(t, out=t)
    return t


def _proj_extent(W: int, H: int, dx: float, dy: float):
    """Min / max of the projection x*dx + y*dy over the image rectangle."""
    xmin = 0.0 if dx >= 0 else (W - 1) * dx
    xmax = (W - 1) * dx if dx >= 0 else 0.0
    ymin = 0.0 if dy >= 0 else (H - 1) * dy
    ymax = (H - 1) * dy if dy >= 0 else 0.0
    return xmin + ymin, xmax + ymax


def _build_geometry(W: int, H: int, angle_deg: float, num_slices: int,
                    spacing: str, seed: int, device, xs_t, ys_t):
    """
    Build the per-pixel slice map for an arbitrary slice angle.

    Returns:
        slice_id_map : (H, W) long   slice index per pixel, in [0, num_slices-1]
        proj         : (H, W) float  projection coordinate per pixel
        bpos_t       : (num_slices+1,) float  boundary positions in proj units
    """
    th = math.radians(angle_deg)
    dx, dy = math.cos(th), math.sin(th)

    # proj = x*dx + y*dy   (gradient magnitude is 1, so proj distance == pixel
    # distance perpendicular to the slice direction)
    proj = xs_t * dx + ys_t * dy  # (H, W) via broadcasting

    pmin, pmax = _proj_extent(W, H, dx, dy)
    span = (pmax - pmin) if (pmax - pmin) != 0 else 1.0

    ts = _boundary_ts(num_slices, spacing, seed)
    bpos = pmin + ts * span
    bpos_t = torch.tensor(bpos, device=device, dtype=proj.dtype)

    interior = bpos_t[1:-1].contiguous()
    if interior.numel() == 0:
        slice_id = torch.zeros_like(proj, dtype=torch.long)
    else:
        slice_id = torch.bucketize(proj, interior, right=True)
    slice_id = slice_id.clamp_(0, num_slices - 1)
    return slice_id, proj, bpos_t


def _slice_offsets(num_slices: int, offset: int, direction: str,
                   cascade_curve: str, slice_jitter: int,
                   seed: int) -> np.ndarray:
    """
    Per-slice frame offset.

    Cumulative cascade semantics:
        linear + forward  ->  offset[i] = i * offset
    direction reshapes which slices carry the most displacement; cascade_curve
    reshapes how fast the displacement grows across slices.
    """
    N = int(num_slices)
    if N <= 1:
        base = np.zeros(max(N, 1), dtype=np.float64)
    else:
        idx = np.arange(N, dtype=np.float64)
        u = idx / (N - 1)  # 0..1

        if direction == "reverse":
            a = 1.0 - u
        elif direction == "center_out":
            a = np.abs(2.0 * u - 1.0)
        elif direction == "edges_in":
            a = 1.0 - np.abs(2.0 * u - 1.0)
        else:  # forward
            a = u

        shaped = _ease(a, cascade_curve)
        base = np.round(shaped * (N - 1) * offset)

    base = base.astype(np.int64)

    if slice_jitter > 0:
        rng = np.random.RandomState((int(seed) + 1234) & 0xFFFFFFFF)
        base = base + rng.randint(-slice_jitter, slice_jitter + 1, size=base.shape)

    return base.astype(np.int64)


def _resolve_index(raw: int, total: int, loop_mode: str) -> int:
    """Resolve a (possibly out-of-range) frame index for a single slice."""
    if total <= 1:
        return 0
    if loop_mode == "loop":
        return int(raw % total)
    if loop_mode == "ping_pong":
        cycle = 2 * total - 2
        mod = int(raw % cycle)
        return mod if mod < total else 2 * total - 2 - mod
    # trim -> clamp
    return int(max(0, min(raw, total - 1)))


def _compute_edges(proj: torch.Tensor, slice_id_map: torch.Tensor,
                   bpos_t: torch.Tensor, num_slices: int,
                   thickness: float = 1.0) -> torch.Tensor:
    """
    Compute the interior slice-boundary mask used for the `slice_edges` output.
    Outer edges are never marked. Returns edge_mask (H, W) float.

    `thickness` controls the line width (px) of the returned mask only; it does
    not alter the image frames.
    """
    half = thickness / 2.0
    left_b = bpos_t[slice_id_map]
    right_b = bpos_t[(slice_id_map + 1).clamp(max=num_slices)]
    dL = proj - left_b
    dR = right_b - proj

    is_left_interior = slice_id_map >= 1
    is_right_interior = slice_id_map <= (num_slices - 2)
    edge = (is_left_interior & (dL <= half)) | (is_right_interior & (dR <= half))
    return edge.to(proj.dtype)


def _apply_blend(frame: torch.Tensor, proj: torch.Tensor,
                 slice_id_map: torch.Tensor, bpos_t: torch.Tensor,
                 num_slices: int, blend_width: int, frame_for):
    """
    Vectorised soft blend across interior slice boundaries.

    `frame_for(sid_map)` must return the (H, W, C) frame gathered as if every
    pixel belonged to the slice ids in `sid_map`. Cost is constant in
    num_slices (two extra gathers), unlike a per-boundary loop.

    Within blend_width of an interior boundary, a pixel mixes its own slice's
    frame with the neighbouring slice's frame; weight ramps from 0.5 at the
    boundary to 1.0 (fully its own) at half the blend width. Where a slice is
    thinner than the blend width, the nearest boundary wins.
    """
    if blend_width <= 1 or num_slices <= 1:
        return frame

    hw = blend_width / 2.0
    left_b = bpos_t[slice_id_map]
    right_b = bpos_t[(slice_id_map + 1).clamp(max=num_slices)]
    dL = proj - left_b
    dR = right_b - proj

    in_left = (slice_id_map >= 1) & (dL <= hw)
    in_right = (slice_id_map <= num_slices - 2) & (dR <= hw)
    use_left = in_left & (~in_right | (dL <= dR))
    use_right = in_right & ~use_left

    if bool(use_left.any()):
        neighbour = frame_for((slice_id_map - 1).clamp(min=0))
        own_w = ((dL + hw) / (2.0 * hw)).clamp(0.0, 1.0).unsqueeze(-1)
        mixed = neighbour * (1.0 - own_w) + frame * own_w
        frame = torch.where(use_left.unsqueeze(-1), mixed, frame)

    if bool(use_right.any()):
        neighbour = frame_for((slice_id_map + 1).clamp(max=num_slices - 1))
        own_w = ((dR + hw) / (2.0 * hw)).clamp(0.0, 1.0).unsqueeze(-1)
        mixed = neighbour * (1.0 - own_w) + frame * own_w
        frame = torch.where(use_right.unsqueeze(-1), mixed, frame)

    return frame


# ════════════════════════════════════════════════════════════════════
#  Node 1 — TimeSlice Effect
# ════════════════════════════════════════════════════════════════════

class TimeSliceEffect:
    """
    Divides frames into angled slices, each offset in time from its neighbour,
    creating a cascading slit-scan style effect.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Source video frames to slice.",
                }),
                "num_slices": ("INT", {
                    "default": 20, "min": 1, "max": 2000, "step": 1,
                    "tooltip": "Number of slices to divide the frame into",
                }),
                "offset": ("INT", {
                    "default": 1, "min": -120, "max": 120, "step": 1,
                    "tooltip": "Frame offset step between slices. Negative reverses time.",
                }),
                "angle": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Slice angle in degrees. 0 = vertical, 90 = horizontal, 45 = diagonal.",
                }),
                "spacing": (["linear", "ease_in", "ease_out", "ease_in_out", "random"], {
                    "tooltip": "How slice widths are distributed across the frame: "
                               "even (linear), eased toward one side, or random.",
                }),
                "direction": (["forward", "reverse", "center_out", "edges_in"], {
                    "tooltip": "Which slices carry the most time offset: building from "
                               "one edge (forward/reverse), from the centre, or from both edges.",
                }),
                "loop_mode": (["loop", "ping_pong", "trim"], {
                    "tooltip": "How out-of-range frame indices are handled: wrap around "
                               "(loop), bounce back (ping_pong), or shorten the clip (trim).",
                }),
                "blend_width": ("INT", {
                    "default": 0, "min": 0, "max": 400, "step": 1,
                    "tooltip": "Soft blend width (px) across boundaries. 0 = hard edges.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1,
                    "tooltip": "Seed for 'random' spacing and slice jitter",
                }),
            },
            "optional": {
                # ── per-slice cascade shaping ──
                "cascade_curve": (["linear", "ease_in", "ease_out", "ease_in_out", "exponential"], {
                    "tooltip": "How the per-slice offset grows across slices.",
                }),
                "slice_jitter": ("INT", {
                    "default": 0, "min": 0, "max": 60, "step": 1,
                    "tooltip": "Random +/- frames added per slice (seeded). Adds glitchy irregularity.",
                }),
                # ── temporal animation (advanced) ──
                "animate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Animate slices/offset/angle from start -> end over the clip.",
                    "advanced": True,
                }),
                "num_slices_end": ("INT", {
                    "default": 60, "min": 1, "max": 2000, "step": 1,
                    "tooltip": "Target slice count reached at the end of the animation.",
                    "advanced": True,
                }),
                "offset_end": ("INT", {
                    "default": 6, "min": -120, "max": 120, "step": 1,
                    "tooltip": "Target per-slice frame offset at the end of the animation.",
                    "advanced": True,
                }),
                "angle_end": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Target slice angle (degrees) at the end of the animation.",
                    "advanced": True,
                }),
                "temporal_curve": (["linear", "ease_in", "ease_out", "ease_in_out", "exponential"], {
                    "tooltip": "Easing applied to the start -> end animation over time.",
                    "advanced": True,
                }),
                "frame_count": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "Frames over which the animation completes. 0 = use clip length.",
                    "advanced": True,
                }),
                "mask": ("MASK", {
                    "tooltip": "Optional mask. Slicing is applied only inside the white "
                               "area; outside keeps the original frame.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "MASK")
    RETURN_NAMES = ("images", "mask", "num_frames", "slice_edges")
    FUNCTION = "apply_time_slice"
    CATEGORY = "Systms/effects"
    DESCRIPTION = (
        "Angled temporal slit-scan. Splits frames into slices offset in time, "
        "with cascade shaping and start->end animation."
    )

    def apply_time_slice(self, images, num_slices, offset, angle, spacing,
                         direction, loop_mode, blend_width, seed,
                         cascade_curve="linear", slice_jitter=0,
                         animate=False, num_slices_end=60, offset_end=6,
                         angle_end=0.0, temporal_curve="linear",
                         frame_count=0, mask=None):

        B, H, W, C = images.shape
        device = images.device
        dtype = images.dtype

        # pixel coordinate grids (shared)
        xs_t = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
        ys_t = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
        xs_t = xs_t.expand(H, W)
        ys_t = ys_t.expand(H, W)

        # ── normalisation length for animation ──
        L_norm = frame_count if (animate and frame_count > 0) else B
        L_norm = max(1, L_norm)

        geometry_animates = animate and (int(num_slices_end) != int(num_slices)
                                         or float(angle_end) != float(angle))

        def params_at(t: int):
            if not animate:
                return int(num_slices), int(offset), float(angle)
            tau = min(1.0, t / max(1, (L_norm - 1)))
            tau = float(_ease(np.array(tau), temporal_curve))
            ns = max(1, int(round(_lerp(num_slices, num_slices_end, tau))))
            off = int(round(_lerp(offset, offset_end, tau)))
            ang = _lerp(angle, angle_end, tau)
            return ns, off, ang

        # ── precompute per-frame offsets to size trim & normalise base ──
        offsets_per_frame = []
        for t in range(B):
            ns, off, _ = params_at(t)
            offsets_per_frame.append(
                _slice_offsets(ns, off, direction, cascade_curve,
                               slice_jitter, seed)
            )
        global_min = min(int(a.min()) for a in offsets_per_frame)
        global_max = max(int(a.max()) for a in offsets_per_frame)
        offset_range = global_max - global_min

        num_output = max(1, B - offset_range) if loop_mode == "trim" else B

        # ── static geometry (computed once when nothing geometric animates) ──
        static_geo = None
        if not geometry_animates:
            ns0, _, ang0 = params_at(0)
            static_geo = _build_geometry(W, H, ang0, ns0, spacing, seed,
                                         device, xs_t, ys_t)

        output = torch.zeros(num_output, H, W, C, dtype=dtype, device=device)
        edges = torch.zeros(num_output, H, W, dtype=dtype, device=device)

        for t in range(num_output):
            ns, off, ang = params_at(t)

            if geometry_animates:
                slice_id_map, proj, bpos_t = _build_geometry(
                    W, H, ang, ns, spacing, seed, device, xs_t, ys_t)
            else:
                slice_id_map, proj, bpos_t = static_geo

            off_arr = offsets_per_frame[t]

            # per-slice resolved source frame index
            src_idx = np.array(
                [_resolve_index(t + int(off_arr[i]) - global_min, B, loop_mode)
                 for i in range(ns)],
                dtype=np.int64,
            )
            src_idx_t = torch.from_numpy(src_idx).to(device)
            ys_l, xs_l = ys_t.long(), xs_t.long()

            def frame_for(sid_map):
                return images[src_idx_t[sid_map], ys_l, xs_l]

            # ── hard gather: each pixel from its slice's source frame ──
            frame = frame_for(slice_id_map)  # (H, W, C)

            # ── soft blend across interior boundaries (vectorised) ──
            frame = _apply_blend(frame, proj, slice_id_map, bpos_t, ns,
                                 blend_width, frame_for)

            # ── slice-boundary mask (for the slice_edges output) ──
            edges[t] = _compute_edges(proj, slice_id_map, bpos_t, ns)

            output[t] = frame

        # ── mask compositing (slicing only inside the mask) ──
        if mask is not None:
            for t in range(num_output):
                m_idx = min(t, mask.shape[0] - 1) if mask.shape[0] > 1 else 0
                m = mask[m_idx].unsqueeze(-1)
                orig_idx = min(t, B - 1)
                output[t] = output[t] * m + images[orig_idx] * (1.0 - m)
                edges[t] = edges[t] * mask[m_idx]

        # ── output mask ──
        if mask is not None:
            if mask.shape[0] >= num_output:
                output_mask = mask[:num_output].clone()
            else:
                output_mask = mask[-1:].expand(num_output, -1, -1).clone()
        else:
            output_mask = torch.ones(num_output, H, W, dtype=dtype, device=device)

        return (output, output_mask, num_output, edges)


# ════════════════════════════════════════════════════════════════════
#  Node 2 — TimeSlice Wave
# ════════════════════════════════════════════════════════════════════

class TimeSliceWave:
    """
    Animated variant — slice offsets oscillate over time following a sine wave,
    producing a rippling / undulating temporal displacement. Now angle-aware,
    with optional blending.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Source video frames to slice.",
                }),
                "num_slices": ("INT", {
                    "default": 30, "min": 1, "max": 2000, "step": 1,
                    "tooltip": "Number of slices to divide the frame into",
                }),
                "max_offset": ("INT", {
                    "default": 8, "min": 1, "max": 120, "step": 1,
                    "tooltip": "Peak frame offset at the wave crest",
                }),
                "wave_speed": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 50.0, "step": 0.25,
                    "tooltip": "Full wave cycles over the clip. Higher = faster. "
                               "Whole numbers loop seamlessly with loop_mode 'loop'.",
                }),
                "wave_count": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 20.0, "step": 0.25,
                    "tooltip": "Number of full sine cycles across all slices",
                }),
                "angle": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Slice angle in degrees. 0 = vertical, 90 = horizontal.",
                }),
                "loop_mode": (["loop", "ping_pong", "trim"], {
                    "tooltip": "How out-of-range frame indices are handled: wrap around "
                               "(loop), bounce back (ping_pong), or shorten the clip (trim).",
                }),
            },
            "optional": {
                "blend_width": ("INT", {
                    "default": 0, "min": 0, "max": 400, "step": 1,
                    "tooltip": "Soft blend width (px) across boundaries. 0 = hard edges.",
                }),
                "spacing": (["linear", "ease_in", "ease_out", "ease_in_out", "random"], {
                    "tooltip": "How slice widths are distributed across the frame: "
                               "even (linear), eased toward one side, or random.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1,
                    "tooltip": "Seed for 'random' spacing",
                }),
                "mask": ("MASK", {
                    "tooltip": "Optional mask. Slicing is applied only inside the white "
                               "area; outside keeps the original frame.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "MASK")
    RETURN_NAMES = ("images", "mask", "num_frames", "slice_edges")
    FUNCTION = "apply_wave_slice"
    CATEGORY = "Systms/effects"
    DESCRIPTION = (
        "Angled animated slit-scan with sinusoidal time displacement that "
        "ripples across the frame, creating an undulating temporal wave."
    )

    def apply_wave_slice(self, images, num_slices, max_offset, wave_speed,
                         wave_count, angle, loop_mode, blend_width=0,
                         spacing="linear", seed=0, mask=None):

        B, H, W, C = images.shape
        device = images.device
        dtype = images.dtype

        xs_t = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)
        ys_t = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)

        slice_id_map, proj, bpos_t = _build_geometry(
            W, H, angle, num_slices, spacing, seed, device, xs_t, ys_t)

        # trim must pad both ends: the sine swings +/- max_offset, so the
        # first/last frames would otherwise clamp and freeze at the edges.
        if loop_mode == "trim":
            num_output = max(1, B - 2 * max_offset)
            pad = max_offset
        else:
            num_output = B
            pad = 0

        output = torch.zeros(num_output, H, W, C, dtype=dtype, device=device)
        edges = torch.zeros(num_output, H, W, dtype=dtype, device=device)

        for t in range(num_output):
            # wave_speed = full cycles travelled over the whole output clip,
            # so the phase per frame stays well below 1 cycle (no temporal
            # aliasing) and whole-number speeds loop seamlessly.
            phase_t = wave_speed * t / max(1, num_output)
            off_arr = np.array([
                round(max_offset * math.sin(
                    2.0 * math.pi * (wave_count * i / max(1, num_slices)
                                     + phase_t)))
                for i in range(num_slices)
            ], dtype=np.int64)

            src_idx = np.array(
                [_resolve_index(t + pad + int(off_arr[i]), B, loop_mode)
                 for i in range(num_slices)],
                dtype=np.int64,
            )
            src_idx_t = torch.from_numpy(src_idx).to(device)
            ys_l, xs_l = ys_t.long(), xs_t.long()

            def frame_for(sid_map):
                return images[src_idx_t[sid_map], ys_l, xs_l]

            frame = frame_for(slice_id_map)

            frame = _apply_blend(frame, proj, slice_id_map, bpos_t,
                                 num_slices, blend_width, frame_for)

            edges[t] = _compute_edges(proj, slice_id_map, bpos_t, num_slices)
            output[t] = frame

        if mask is not None:
            for t in range(num_output):
                m_idx = min(t, mask.shape[0] - 1) if mask.shape[0] > 1 else 0
                m = mask[m_idx].unsqueeze(-1)
                orig_idx = min(t + pad, B - 1)
                output[t] = output[t] * m + images[orig_idx] * (1.0 - m)
                edges[t] = edges[t] * mask[m_idx]

        if mask is not None:
            if mask.shape[0] >= num_output:
                output_mask = mask[:num_output].clone()
            else:
                output_mask = mask[-1:].expand(num_output, -1, -1).clone()
        else:
            output_mask = torch.ones(num_output, H, W, dtype=dtype, device=device)

        return (output, output_mask, num_output, edges)


# ════════════════════════════════════════════════════════════════════
#  Node 3 — TimeSlice Dual
# ════════════════════════════════════════════════════════════════════

class TimeSliceDual:
    """
    Dual-source TimeSlice — alternates slices between two video inputs, each
    with independent temporal offsets. Angle-aware, with cascade shaping
    and start->end animation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE", {
                    "tooltip": "First source clip (A).",
                }),
                "images_b": ("IMAGE", {
                    "tooltip": "Second source clip (B). Resized to match A if their "
                               "dimensions differ.",
                }),
                "num_slices": ("INT", {
                    "default": 20, "min": 1, "max": 2000, "step": 1,
                    "tooltip": "Number of slices to divide the frame into",
                }),
                "offset_a": ("INT", {
                    "default": 1, "min": -120, "max": 120, "step": 1,
                    "tooltip": "Frame offset step for A-source slices",
                }),
                "offset_b": ("INT", {
                    "default": 1, "min": -120, "max": 120, "step": 1,
                    "tooltip": "Frame offset step for B-source slices",
                }),
                "angle": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Slice angle in degrees. 0 = vertical, 90 = horizontal.",
                }),
                "spacing": (["linear", "ease_in", "ease_out", "ease_in_out", "random"], {
                    "tooltip": "How slice widths are distributed across the frame: "
                               "even (linear), eased toward one side, or random.",
                }),
                "direction": (["forward", "reverse", "center_out", "edges_in"], {
                    "tooltip": "Which slices carry the most time offset: building from "
                               "one edge (forward/reverse), from the centre, or from both edges.",
                }),
                "pattern": (["alternate", "pairs", "aba", "aabb", "abba"], {
                    "tooltip": "How slices are assigned to the A and B clips across the frame.",
                }),
                "loop_mode": (["loop", "ping_pong", "trim"], {
                    "tooltip": "How out-of-range frame indices are handled: wrap around "
                               "(loop), bounce back (ping_pong), or shorten the clip (trim).",
                }),
                "blend_width": ("INT", {
                    "default": 0, "min": 0, "max": 400, "step": 1,
                    "tooltip": "Soft blend width (px) across boundaries. 0 = hard edges.",
                }),
                "mix": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Balance between A and B. 0.5 = equal.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1,
                    "tooltip": "Seed for 'random' spacing, slice jitter and mix balancing",
                }),
            },
            "optional": {
                "cascade_curve": (["linear", "ease_in", "ease_out", "ease_in_out", "exponential"], {
                    "tooltip": "How the per-slice offset grows across slices.",
                }),
                "slice_jitter": ("INT", {
                    "default": 0, "min": 0, "max": 60, "step": 1,
                    "tooltip": "Random +/- frames added per slice (seeded). Adds glitchy irregularity.",
                }),
                # ── temporal animation (advanced) ──
                "animate": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Animate slices/offsets/angle from start -> end over the clip.",
                    "advanced": True,
                }),
                "num_slices_end": ("INT", {
                    "default": 60, "min": 1, "max": 2000, "step": 1,
                    "tooltip": "Target slice count reached at the end of the animation.",
                    "advanced": True,
                }),
                "offset_a_end": ("INT", {
                    "default": 6, "min": -120, "max": 120, "step": 1,
                    "tooltip": "Target A-source offset at the end of the animation.",
                    "advanced": True,
                }),
                "offset_b_end": ("INT", {
                    "default": 6, "min": -120, "max": 120, "step": 1,
                    "tooltip": "Target B-source offset at the end of the animation.",
                    "advanced": True,
                }),
                "angle_end": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "Target slice angle (degrees) at the end of the animation.",
                    "advanced": True,
                }),
                "temporal_curve": (["linear", "ease_in", "ease_out", "ease_in_out", "exponential"], {
                    "tooltip": "Easing applied to the start -> end animation over time.",
                    "advanced": True,
                }),
                "frame_count": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "Frames over which the animation completes. 0 = use clip length.",
                    "advanced": True,
                }),
                "mask": ("MASK", {
                    "tooltip": "Optional mask. Slicing is applied only inside the white "
                               "area; outside keeps the original A frame.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "MASK")
    RETURN_NAMES = ("images", "mask", "num_frames", "slice_edges")
    FUNCTION = "apply_dual_slice"
    CATEGORY = "Systms/effects"
    DESCRIPTION = (
        "Dual-source angled slit-scan. Interleaves slices from two clips, each "
        "with its own time offset. Great for transitions and split-reality looks."
    )

    @staticmethod
    def _get_source_pattern(num_slices: int, pattern: str) -> list:
        if pattern == "alternate":
            return [i % 2 for i in range(num_slices)]
        if pattern == "pairs":
            return [(i // 2) % 2 for i in range(num_slices)]
        if pattern == "aba":
            cycle = [0, 1, 0]
            return [cycle[i % 3] for i in range(num_slices)]
        if pattern == "aabb":
            cycle = [0, 0, 1, 1]
            return [cycle[i % 4] for i in range(num_slices)]
        if pattern == "abba":
            cycle = [0, 1, 1, 0]
            return [cycle[i % 4] for i in range(num_slices)]
        return [i % 2 for i in range(num_slices)]

    @staticmethod
    def _biased_source_map(num_slices, pattern, mix, seed):
        source_map = TimeSliceDual._get_source_pattern(num_slices, pattern)
        if mix != 0.5:
            b_count = sum(source_map)
            target_b = max(0, min(num_slices, round(num_slices * mix)))
            rng = np.random.RandomState((int(seed) + 9999) & 0xFFFFFFFF)
            if b_count < target_b:
                a_idx = [i for i, s in enumerate(source_map) if s == 0]
                rng.shuffle(a_idx)
                for idx in a_idx[:target_b - b_count]:
                    source_map[idx] = 1
            elif b_count > target_b:
                b_idx = [i for i, s in enumerate(source_map) if s == 1]
                rng.shuffle(b_idx)
                for idx in b_idx[:b_count - target_b]:
                    source_map[idx] = 0
        return source_map

    def apply_dual_slice(self, images_a, images_b, num_slices, offset_a,
                         offset_b, angle, spacing, direction, pattern,
                         loop_mode, blend_width, mix, seed,
                         cascade_curve="linear", slice_jitter=0,
                         animate=False, num_slices_end=60, offset_a_end=6,
                         offset_b_end=6, angle_end=0.0,
                         temporal_curve="linear", frame_count=0, mask=None):

        B_a, H, W, C = images_a.shape
        device = images_a.device
        dtype = images_a.dtype

        # match B to A's spatial size
        if images_b.shape[1] != H or images_b.shape[2] != W:
            b_perm = images_b.permute(0, 3, 1, 2)
            b_perm = torch.nn.functional.interpolate(
                b_perm, size=(H, W), mode="bilinear", align_corners=False)
            images_b = b_perm.permute(0, 2, 3, 1)
        B_b = images_b.shape[0]

        source_frames = [B_a, B_b]

        xs_t = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)
        ys_t = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)

        L_norm = frame_count if (animate and frame_count > 0) else min(B_a, B_b)
        L_norm = max(1, L_norm)

        geometry_animates = animate and (int(num_slices_end) != int(num_slices)
                                         or float(angle_end) != float(angle))

        def params_at(t):
            if not animate:
                return int(num_slices), int(offset_a), int(offset_b), float(angle)
            tau = min(1.0, t / max(1, (L_norm - 1)))
            tau = float(_ease(np.array(tau), temporal_curve))
            ns = max(1, int(round(_lerp(num_slices, num_slices_end, tau))))
            oa = int(round(_lerp(offset_a, offset_a_end, tau)))
            ob = int(round(_lerp(offset_b, offset_b_end, tau)))
            ang = _lerp(angle, angle_end, tau)
            return ns, oa, ob, ang

        # precompute per-frame source maps + offsets to size trim
        base_len = min(B_a, B_b)
        per_frame = []  # (source_map, frame_offsets, min_a, min_b)
        glob_min_a = glob_min_b = 0
        a_lo = b_lo = 10 ** 9
        a_hi = b_hi = -10 ** 9
        for t in range(base_len):
            ns, oa, ob, _ = params_at(t)
            smap = self._biased_source_map(ns, pattern, mix, seed)
            off_a = _slice_offsets(ns, oa, direction, cascade_curve,
                                   slice_jitter, seed)
            off_b = _slice_offsets(ns, ob, direction, cascade_curve,
                                   slice_jitter, seed + 7)
            foff = [int(off_a[i]) if smap[i] == 0 else int(off_b[i])
                    for i in range(ns)]
            a_vals = [foff[i] for i in range(ns) if smap[i] == 0] or [0]
            b_vals = [foff[i] for i in range(ns) if smap[i] == 1] or [0]
            a_lo, a_hi = min(a_lo, min(a_vals)), max(a_hi, max(a_vals))
            b_lo, b_hi = min(b_lo, min(b_vals)), max(b_hi, max(b_vals))
            per_frame.append((smap, foff))
        glob_min_a, glob_min_b = a_lo, b_lo
        a_range, b_range = a_hi - a_lo, b_hi - b_lo

        if loop_mode == "trim":
            num_output = min(max(1, B_a - a_range), max(1, B_b - b_range))
        else:
            num_output = base_len

        static_geo = None
        if not geometry_animates:
            ns0, _, _, ang0 = params_at(0)
            static_geo = _build_geometry(W, H, ang0, ns0, spacing, seed,
                                         device, xs_t, ys_t)

        output = torch.zeros(num_output, H, W, C, dtype=dtype, device=device)
        edges = torch.zeros(num_output, H, W, dtype=dtype, device=device)

        for t in range(num_output):
            ns, _, _, ang = params_at(t)
            smap, foff = per_frame[t]

            if geometry_animates:
                slice_id_map, proj, bpos_t = _build_geometry(
                    W, H, ang, ns, spacing, seed, device, xs_t, ys_t)
            else:
                slice_id_map, proj, bpos_t = static_geo

            # resolved frame index + chosen source per slice
            src_sel = np.array(smap, dtype=np.int64)
            frame_idx = np.empty(ns, dtype=np.int64)
            for i in range(ns):
                sid = smap[i]
                total = source_frames[sid]
                min_off = glob_min_a if sid == 0 else glob_min_b
                frame_idx[i] = _resolve_index(t + foff[i] - min_off, total, loop_mode)

            src_sel_t = torch.from_numpy(src_sel).to(device)
            fidx_t = torch.from_numpy(frame_idx).to(device)
            ys_l, xs_l = ys_t.long(), xs_t.long()

            def frame_for(sid_map):
                sel = src_sel_t[sid_map]                    # (H,W) 0/1
                fidx = fidx_t[sid_map]                      # (H,W)
                ga = images_a[fidx.clamp(max=B_a - 1), ys_l, xs_l]
                gb = images_b[fidx.clamp(max=B_b - 1), ys_l, xs_l]
                return torch.where(sel.unsqueeze(-1) == 1, gb, ga)

            frame = frame_for(slice_id_map)

            frame = _apply_blend(frame, proj, slice_id_map, bpos_t, ns,
                                 blend_width, frame_for)

            edges[t] = _compute_edges(proj, slice_id_map, bpos_t, ns)
            output[t] = frame

        if mask is not None:
            for t in range(num_output):
                m_idx = min(t, mask.shape[0] - 1) if mask.shape[0] > 1 else 0
                m = mask[m_idx].unsqueeze(-1)
                orig_idx = min(t, B_a - 1)
                output[t] = output[t] * m + images_a[orig_idx] * (1.0 - m)
                edges[t] = edges[t] * mask[m_idx]

        if mask is not None:
            if mask.shape[0] >= num_output:
                output_mask = mask[:num_output].clone()
            else:
                output_mask = mask[-1:].expand(num_output, -1, -1).clone()
        else:
            output_mask = torch.ones(num_output, H, W, dtype=dtype, device=device)

        return (output, output_mask, num_output, edges)


# ── Node registration ──────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "TimeSliceEffect": TimeSliceEffect,
    "TimeSliceWave": TimeSliceWave,
    "TimeSliceDual": TimeSliceDual,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TimeSliceEffect": "TimeSlice Effect",
    "TimeSliceWave": "TimeSlice Wave",
    "TimeSliceDual": "TimeSlice Dual",
}