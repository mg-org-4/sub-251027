"""Beat-driven editing and cursor-drag reveals of aligned scan layers."""
import json
import math
import random

import cv2
import numpy as np
import torch

from comfy.utils import ProgressBar
from ..audio.audio_envelope import load_audio_envelope


def edit_plan(lengths, kick, snare, min_cut, max_cut, rate, seed):
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("Scan Edit shot lengths must be positive frame counts.")
    if min_cut < 2 or max_cut < min_cut:
        raise ValueError("Scan Edit maximum cut length must be at least the minimum (2 frames or more).")
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Scan Edit playback rate must be positive.")
    count = len(kick)
    triggers = [max(k, s) for k, s in zip(kick, snare)]
    boundaries, reasons = [0], ["start"]
    for frame in range(1, count):
        age = frame - boundaries[-1]
        hit = triggers[frame] >= 0.65 and triggers[frame - 1] < 0.65
        if age >= min_cut and (hit or age >= max_cut):
            boundaries.append(frame)
            reasons.append("audio_onset" if hit else "maximum_hold")
    boundaries.append(count)
    rng = np.random.default_rng(seed)
    offsets = [sum(lengths[:index]) for index in range(len(lengths))]
    indices, segments, cycle = [], [], []
    previous_shot = None
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        jump = index % 5 == 4
        if not cycle:
            cycle = rng.permutation(len(lengths)).tolist()
            if len(cycle) > 1 and cycle[0] == previous_shot:
                cycle[0], cycle[1] = cycle[1], cycle[0]
        shot = previous_shot if jump else cycle.pop(0)
        previous_shot = shot
        step = min(rate, (lengths[shot] - 1) / max(1, end - start - 1))
        travel = round((end - start - 1) * step)
        seek = int(rng.integers(0, lengths[shot] - travel))
        if jump:
            last = indices[-1] - offsets[shot]
            if abs(seek - last) < 6:
                seek = max((0, lengths[shot] - travel - 1), key=lambda value: abs(value - last))
        indices.extend(offsets[shot] + seek + round(local * step) for local in range(end - start))
        segments.append({"start_frame": start, "end_frame": end, "shot": shot + 1,
                         "source_start": seek, "playback_rate": step, "trigger": reasons[index],
                         "edit_type": "jump_cut" if jump else "shot_cut"})
    return indices, segments


def draw_cursor(image, point, color, scale):
    shape = np.array([[0, 0], [0, 23], [6, 17], [11, 28], [16, 25], [10, 15], [20, 15]], np.float32)
    polygon = np.rint(shape * scale + point).astype(np.int32)
    cv2.fillPoly(image, [polygon], (0.02, 0.025, 0.03), cv2.LINE_AA)
    cv2.polylines(image, [polygon], True, (0.02, 0.025, 0.03), max(2, round(scale * 3)), cv2.LINE_AA)
    cv2.fillPoly(image, [polygon], color, cv2.LINE_AA)


def order_windows(events, mode, ranks):
    if mode == "random_on_snare":
        return sorted(events, key=lambda e: ranks[e["cursor"]])
    priority = {"voxel_on_top": 0, "edge_on_top": 1, "depth_on_top": 2}.get(mode)
    if priority is not None:
        return sorted(events, key=lambda e: e["effect"] == priority)
    return events


def blend_reveal(region, layer, alpha, mode):
    if mode == "screen":
        layer = 1 - (1-region) * (1-layer)
    elif mode == "add":
        layer = (region + layer).clip(0,1)
    region *= 1-alpha
    region += layer * alpha


def reveal_fade(frame, event, fps, fade_in, fade_out):
    drag_start = event["start"] + .12 * max(1, event["end"]-event["start"]-1)
    attack = min(1, max(0, (frame-drag_start)/(fps*fade_in))) if fade_in else 1
    release = min(1, (event["end"]-1-frame)/(fps*fade_out)) if fade_out else 1
    return attack * max(0,release)


def cursor_plan(envelopes, count, seed, fps):
    rng = np.random.default_rng(seed)
    available = [0] * count
    events = []
    for frame in range(len(envelopes[0])):
        hits = [band for band, values in enumerate(envelopes)
                if values[frame] >= 0.5 and (frame == 0 or values[frame - 1] < 0.5)]
        for band in hits:
            free = [slot for slot, end in enumerate(available) if end <= frame]
            if not free:
                break
            slot = int(rng.choice(free))
            strength = envelopes[band][frame]
            duration = int(rng.integers(max(4, round(fps * .3)), max(5, round(fps * .9))))
            anchor = rng.uniform(.12, .88, 2)
            direction = np.where(anchor > .5, -1, 1)
            end = np.clip(anchor + direction * rng.uniform(.16, .48, 2) * (.7 + .3 * strength), .05, .95)
            events.append({"start": frame, "end": frame + duration, "cursor": slot, "band": band,
                           "anchor": anchor.tolist(), "target": end.tolist(),
                           "approach": np.clip(anchor + rng.uniform(-.18, .18, 2), .02, .98).tolist(),
                           "bend": float(rng.uniform(-.12, .12)), "effect": int(rng.integers(0, 2))})
            available[slot] = frame + duration + int(rng.integers(1, max(2, round(fps * .15))))
    return events


def colorize_depth(depth, style):
    depth = depth.clip(0,1)
    if style == "grayscale":
        return np.repeat(depth[:,:,None],3,axis=2)
    if style == "false_color":
        color = cv2.applyColorMap((depth*255).astype(np.uint8),cv2.COLORMAP_TURBO)[:,:,::-1].astype(np.float32)/255
        color[depth == 0] = 0
        return color
    levels = np.floor(depth*12)/12
    color = np.repeat(levels[:,:,None],3,axis=2)
    edges = (depth*12 % 1 < .12) & (depth > 0)
    color[edges] = (.2,.8,1)
    return color


def assign_reveal_layers(events, values, seed):
    rng = np.random.default_rng(seed+1729)
    result = []
    for event in events:
        frame = event["start"]
        activity = values["cursor_activity"][frame]
        if activity < 1 and rng.random() >= activity:
            continue
        weights = np.array([values[k][frame] for k in ("voxel_weight","edge_weight","depth_weight")])
        if weights[2] != 0 or weights[0] != weights[1]:
            event["effect"] = int(rng.choice(3,p=weights/weights.sum()))
        size = values["reveal_size"][frame]
        if size != 1:
            anchor = np.array(event["anchor"])
            event["target"] = np.clip(anchor+(np.array(event["target"])-anchor)*size,.02,.98).tolist()
        result.append(event)
    return result


class FL_ScanAudioEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",), "scan_images": ("IMAGE",), "normal_images": ("IMAGE",),
            "kick_envelope": ("FL_AUDIO_ENVELOPE",), "snare_envelope": ("FL_AUDIO_ENVELOPE",),
            "hihat_envelope": ("FL_AUDIO_ENVELOPE",),
            "shot_lengths": ("STRING", {"default": "66,66,66,66"}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            "seed": ("INT", {"default": 73, "min": 0, "max": 2147483647}),
            "min_cut_frames": ("INT", {"default": 10, "min": 2, "max": 120}),
            "max_cut_frames": ("INT", {"default": 20, "min": 2, "max": 240}),
            "playback_rate": ("FLOAT", {"default": 1.7, "min": 0.25, "max": 5, "step": 0.05}),
            "cursor_count": ("INT", {"default": 2, "min": 0, "max": 3}),
            "cursor_scale": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 3, "step": 0.05}),
            "reveal_strength": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
            "motion_strength": ("FLOAT", {"default": 0.8, "min": 0, "max": 2, "step": 0.05}),
        }, "optional": {
            "timing_mode": (["remix", "audio_locked"], {"default": "remix",
                "tooltip": "Audio locked preserves chronological source frames and uses camera cuts instead of seeking or retiming."}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("final", "edited_original", "reveal_mask", "edit_report")
    FUNCTION = "render"
    CATEGORY = "🏵️Fill Nodes/VFX"
    DESCRIPTION = "FL drum envelopes drive jump cuts, punch-ins and cursor-drag layer reveals. All source banks must use identical frame order. Outputs last exactly as long as the envelopes. Maximum cut length caps holds between detected onsets."

    def render(self, images, scan_images, normal_images, kick_envelope, snare_envelope, hihat_envelope,
               shot_lengths, fps, seed, min_cut_frames, max_cut_frames, playback_rate, cursor_count,
               cursor_scale, reveal_strength, motion_strength, timing_mode="remix"):
        return self.render_mapped(images,scan_images,normal_images,kick_envelope,snare_envelope,hihat_envelope,
            shot_lengths,fps,seed,min_cut_frames,max_cut_frames,playback_rate,cursor_count,cursor_scale,reveal_strength,
            motion_strength,timing_mode)

    def render_mapped(self, images, scan_images, normal_images, kick_envelope, snare_envelope, hihat_envelope,
               shot_lengths, fps, seed, min_cut_frames, max_cut_frames, playback_rate, cursor_count,
               cursor_scale, reveal_strength, motion_strength, timing_mode="remix", frame_values=None,
               depth_images=None, depth_style="grayscale", window_order="newest_on_top", window_blend="normal",
               window_fade_in=0, window_fade_out=0):
        if images.shape != scan_images.shape or images.shape != normal_images.shape or images.shape[-1] != 3:
            raise ValueError("Scan Edit needs matching RGB original, scan and projected-normal frame banks.")
        lengths = [int(value.strip()) for value in shot_lengths.split(",")]
        if sum(lengths) != len(images):
            raise ValueError("Scan Edit shot lengths must add up to the source bank's frame count.")
        envelopes = [load_audio_envelope(value) for value in (kick_envelope, snare_envelope, hihat_envelope)]
        count = envelopes[0]["total_frames"]
        if any(e["total_frames"] != count or not math.isclose(e["fps"], fps) for e in envelopes):
            raise ValueError("Scan Edit envelopes must have equal frame counts and match the edit FPS.")
        kick, snare, hat = [e["values"] for e in envelopes]
        indices, segments = edit_plan(lengths, kick, snare, min_cut_frames, max_cut_frames, playback_rate, seed)
        if timing_mode == "audio_locked":
            if len(images) != count:
                raise ValueError("Audio locked editing requires one chronological source frame per envelope frame.")
            indices = list(range(count))
            boundaries = {s["start_frame"]: s["trigger"] for s in segments}
            offsets = [sum(lengths[:i]) for i in range(len(lengths))]
            boundaries.update({offset: "authored_shot" for offset in offsets})
            starts = sorted(boundaries)
            segments = [{"start_frame": start, "end_frame": end,
                         "shot": max(i + 1 for i, offset in enumerate(offsets) if offset <= start),
                         "source_start": start, "playback_rate": 1, "trigger": boundaries[start],
                         "edit_type": "shot_cut" if start in offsets else "camera_cut"}
                        for start, end in zip(starts, starts[1:] + [count])]
        elif timing_mode != "remix":
            raise ValueError("Scan Edit timing mode must be remix or audio_locked.")
        events = cursor_plan((kick, snare, hat), cursor_count, seed + 811, fps)
        if frame_values is not None:
            events = assign_reveal_layers(events,frame_values,seed)
        height, width = images.shape[1:3]
        final = np.empty((count, height, width, 3), np.float32)
        original = np.empty_like(final)
        masks = np.zeros((count, height, width), np.float32)
        progress_bar = ProgressBar(count)
        order_rng = random.Random(seed + 2909)
        ranks = list(range(cursor_count))
        order_rng.shuffle(ranks)
        for cut, segment in enumerate(segments):
            rng = np.random.default_rng(seed + cut * 997)
            camera_zoom = float(rng.uniform(0, .22)) if timing_mode == "audio_locked" else 0
            camera_angle = float(rng.uniform(-3, 3)) if timing_mode == "audio_locked" else 0
            for frame in range(segment["start_frame"], segment["end_frame"]):
                if frame_values is not None:
                    motion_strength,cursor_scale,reveal_strength = [frame_values[k][frame] for k in
                        ("motion_strength","cursor_scale","reveal_strength")]
                source = indices[frame]
                original[frame] = images[source].cpu().float().numpy()
                scan = scan_images[source].cpu().float().numpy()
                normal = normal_images[source].cpu().float().numpy()
                zoom = 1 + motion_strength * (camera_zoom + 0.16 * kick[frame] + 0.055 * snare[frame])
                angle = motion_strength * (camera_angle + snare[frame] * (1.8 if cut % 2 else -1.8))
                transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, zoom)
                transform[0, 2] += math.sin(frame * 2.4) * hat[frame] * motion_strength * width * 0.004
                composite = cv2.warpAffine(scan, transform, (width, height), borderMode=cv2.BORDER_REFLECT_101)
                normal = cv2.warpAffine(normal, transform, (width, height), borderMode=cv2.BORDER_REFLECT_101)
                effect = None
                active = [e for e in events if e["start"] <= frame < e["end"]]
                if window_order == "random_on_snare" and snare[frame] >= .5 and (frame == 0 or snare[frame-1] < .5):
                    order_rng.shuffle(ranks)
                active = order_windows(active, window_order, ranks)
                if any(e["effect"] == 1 for e in active):
                    gray = cv2.cvtColor((composite * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 55, 135).astype(np.float32) / 255
                    effect = composite[:, :, ::-1].copy() * 0.55
                    effect += edges[:, :, None] * np.array([0.2, 0.9, 1], np.float32)
                    effect = effect.clip(0, 1)
                depth_layer = None
                if any(e["effect"] == 2 for e in active):
                    depth_layer = colorize_depth(cv2.warpAffine(depth_images[source].cpu().numpy(),transform,(width,height),
                        borderMode=cv2.BORDER_REFLECT_101),depth_style)
                for event in active:
                    anchor = np.array(event["anchor"]) * [width, height]
                    end = np.array(event["target"]) * [width, height]
                    age = (frame - event["start"]) / max(1, event["end"] - event["start"] - 1)
                    drag = min(1, max(0, (age - 0.12) / 0.58))
                    drag = 1 - (1 - drag) ** 2
                    tip = anchor + (end - anchor) * drag
                    if age < 0.12:
                        approach = np.array(event["approach"]) * [width, height]
                        tip = approach + (anchor - approach) * age / .12
                    else:
                        tip[0] += math.sin(drag * math.pi) * event["bend"] * width
                    tip = np.clip(tip, [0, 0], [width - 1, height - 1])
                    normals = event["effect"] == 0
                    is_depth = event["effect"] == 2
                    color = (1,.8,.2) if is_depth else ((1.0, 1.0, 1.0) if normals else (0.25, 1.0, 0.88))
                    x1, y1 = np.rint(np.minimum(anchor, tip)).astype(int)
                    x2, y2 = np.rint(np.maximum(anchor, tip)).astype(int)
                    if drag > 0 and x2 > x1 and y2 > y1:
                        layer = depth_layer if is_depth else (normal if normals else effect)
                        alpha = reveal_strength * (0.8 + 0.2 * snare[frame] if normals or is_depth else 0.55 + 0.45 * hat[frame])
                        opacity_key = ("voxel_opacity","edge_opacity","depth_opacity")[event["effect"]]
                        if frame_values is not None and opacity_key in frame_values:
                            alpha *= frame_values[opacity_key][frame]
                        alpha *= reveal_fade(frame,event,fps,window_fade_in,window_fade_out)
                        region = composite[y1:y2, x1:x2]
                        blend_reveal(region,layer[y1:y2, x1:x2],alpha,window_blend)
                        masks[frame, y1:y2, x1:x2] = 1 - (1 - masks[frame, y1:y2, x1:x2]) * (1 - alpha)
                        cv2.rectangle(composite, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
                        label = "DEPTH / DRAG" if is_depth else ("NORMALS / DRAG" if normals else "EDGE SCAN / DRAG")
                        cv2.putText(composite, label, (x1, y1 - 7), cv2.FONT_HERSHEY_PLAIN, max(0.65, width / 750), color, 1, cv2.LINE_AA)
                        for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                            cv2.rectangle(composite, (x - 2, y - 2), (x + 2, y + 2), color, -1)
                    if 0.12 <= age < 0.35 or age > 0.9:
                        center = anchor if age < 0.35 else tip
                        radius = max(4, round(width * 0.015 * (1 + kick[frame])))
                        cv2.circle(composite, tuple(np.rint(center).astype(int)), radius, color, 1, cv2.LINE_AA)
                    draw_cursor(composite, tip, color, cursor_scale * max(0.5, width / 640))
                final[frame] = composite.clip(0, 1)
                progress_bar.update(1)
        report = {"fps": fps, "frames": count, "duration": count / fps, "cuts": len(segments) - 1,
                  "audio_triggered_cuts": sum(s["trigger"] == "audio_onset" for s in segments),
                  "same_shot_jump_cuts": sum(s["edit_type"] == "jump_cut" for s in segments),
                  "timing_mode": timing_mode, "cursor_events": events,
                  "segments": segments, "source_indices": indices}
        return torch.from_numpy(final), torch.from_numpy(original), torch.from_numpy(masks), json.dumps(report, indent=2)
