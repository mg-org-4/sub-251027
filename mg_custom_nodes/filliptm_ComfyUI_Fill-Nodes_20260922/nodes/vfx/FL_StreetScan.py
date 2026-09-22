"""Tracked detections and a depth-reprojected street-scan composite."""
import math

import cv2
import numpy as np
import torch

from comfy.utils import ProgressBar


class FL_ScanVideoDetections:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "person_detector": ("SEGM_DETECTOR",),
                             "threshold": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 1.0, "step": 0.01})},
                "optional": {"face_detector": ("BBOX_DETECTOR",), "hand_detector": ("BBOX_DETECTOR",)}}

    RETURN_TYPES = ("MASK", "FL_SCAN_TRACKS")
    RETURN_NAMES = ("subject_masks", "detections")
    FUNCTION = "detect"
    CATEGORY = "🏵️Fill Nodes/VFX"
    DESCRIPTION = "Run connected Impact detectors on each frame. Produces person masks and nearest-neighbor tracked boxes; no fabricated object labels."

    def detect(self, images, person_detector, threshold, face_detector=None, hand_detector=None):
        count, height, width = images.shape[:3]
        masks = np.zeros((count, height, width), dtype=np.float32)
        frames, previous, next_id = [], [], 1
        progress = ProgressBar(count)
        for frame in range(count):
            detections = []
            for label, detector in (("person", person_detector), ("face", face_detector), ("hand", hand_detector)):
                if detector is None:
                    continue
                _, segments = detector.detect(images[frame:frame+1], threshold, 0, 1.0, drop_size=8)
                for segment in segments:
                    x1, y1, x2, y2 = map(int, segment.bbox)
                    detections.append({"label": label, "box": [x1, y1, x2, y2], "confidence": float(segment.confidence)})
                    if label == "person":
                        left, top, right, bottom = map(int, segment.crop_region)
                        mask = np.asarray(segment.cropped_mask, dtype=np.float32).squeeze()
                        masks[frame, top:bottom, left:right] = np.maximum(masks[frame, top:bottom, left:right], mask)
            used = set()
            for detection in detections:
                box = detection["box"]
                center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                candidates = [(np.linalg.norm(center - np.array([(old["box"][0] + old["box"][2]) / 2, (old["box"][1] + old["box"][3]) / 2])), old)
                              for old in previous if old["label"] == detection["label"] and old["id"] not in used]
                distance, closest = min(candidates, key=lambda pair: pair[0]) if candidates else (float("inf"), None)
                if distance < 0.18 * width:
                    detection["id"] = closest["id"]
                else:
                    detection["id"] = next_id
                    next_id += 1
                used.add(detection["id"])
            frames.append(detections)
            previous = detections
            progress.update(1)
        return torch.from_numpy(masks), {"width": width, "height": height, "frames": frames}


def scene_fragment(depth, subject, seed, raggedness):
    height, width = depth.shape
    rng = np.random.default_rng(seed)
    coarse = cv2.resize(rng.uniform(-1, 1, (11, 11)).astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    chips = cv2.resize(rng.uniform(-1, 1, (48, 48)).astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    nx, ny = (xx - width / 2) / (width * 0.5), (yy - height / 2) / (height * 0.5)
    radius = (np.abs(nx)**4 + np.abs(ny)**4)**0.25
    field = radius + raggedness * (coarse * 0.55 + chips * 0.15) + (0.5 - depth) * 0.16
    keep = (field < 0.87).astype(np.float32)
    subject = cv2.dilate((subject > 0.4).astype(np.uint8), np.ones((5, 5), np.uint8))
    return np.maximum(keep, subject).astype(bool)


def project_depth(depth, phase, orbit, relief, scale):
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    focal = width * 0.9
    z = 1.5 + (1 - depth) * relief
    x, y = (xx - width / 2) * z / focal, (yy - height / 2) * z / focal
    angle = math.radians(orbit) * math.sin(phase * 2 * math.pi)
    pivot = 1.5 + relief * 0.5
    rotated_x = math.cos(angle) * x + math.sin(angle) * (z - pivot)
    rotated_z = -math.sin(angle) * x + math.cos(angle) * (z - pivot) + pivot
    px = rotated_x / rotated_z * focal * scale + width / 2
    py = y / rotated_z * focal * scale + height / 2
    return px, py, rotated_z


def project_parallax(depth, phase, orbit, relief, scale, strength, offset_x, offset_y, dolly, steady):
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    focal = width * .9
    z = 1.5 + (1-depth)*relief
    x,y = (xx-width/2)*z/focal, (yy-height/2)*z/focal
    pivot = 1.5 + (1-steady)*relief
    angle = math.radians(orbit)*math.sin(phase*math.tau)
    rx = math.cos(angle)*x + math.sin(angle)*(z-pivot)
    rz = -math.sin(angle)*x + math.cos(angle)*(z-pivot) + pivot
    travel = dolly*strength
    dx,dy = offset_x*strength*pivot, offset_y*strength*pivot
    # Counter-shift and counter-zoom anchor the selected depth plane.
    zoom = (pivot-travel)/pivot
    denominator = np.maximum(.1,rz-travel)
    px = ((rx-dx)/denominator*zoom + dx/pivot)*focal*scale+width/2
    py = ((y-dy)/denominator*zoom + dy/pivot)*focal*scale+height/2
    return px,py,denominator


def splat(rgb, keep, px, py, z, background):
    height, width = keep.shape
    channels = rgb.shape[-1]
    xi, yi = np.rint(px).astype(np.int32), np.rint(py).astype(np.int32)
    valid = keep & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    source = np.flatnonzero(valid)
    target = yi.flat[source] * width + xi.flat[source]
    order = np.lexsort((z.flat[source], target))
    target, source = target[order], source[order]
    _, first = np.unique(target, return_index=True)
    target, source = target[first], source[first]
    output = np.full((height * width, channels), background, np.float32)
    output[target] = rgb.reshape(-1, channels)[source]
    matte = np.zeros(height * width, np.uint8)
    matte[target] = 255
    return output.reshape(height, width, channels), matte.reshape(height, width)


def draw_box(image, box, color, text, opacity):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = np.rint(box).astype(int)
    x1, x2 = np.clip([x1, x2], 2, width - 3)
    y1, y2 = np.clip([y1, y2], 14, height - 3)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    overlay = image.copy()
    length = min(10, (x2 - x1) // 3, (y2 - y1) // 3)
    for x, y, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(overlay, (x, y), (x + dx * length, y), color, 1, cv2.LINE_AA)
        cv2.line(overlay, (x, y), (x, y + dy * length), color, 1, cv2.LINE_AA)
    text_width = min(width - x1 - 2, max(44, len(text) * 4))
    cv2.rectangle(overlay, (x1, y1 - 11), (x1 + text_width, y1 - 1), color, -1)
    cv2.putText(overlay, text, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_PLAIN, 0.55, (0.03, 0.04, 0.04), 1, cv2.LINE_AA)
    image[:] = image * (1 - opacity) + overlay * opacity


def fill_projected_gaps(layers, matte):
    height, width = matte.shape
    points = cv2.findNonZero(matte)
    if points is None:
        return layers, matte
    hull = np.zeros_like(matte)
    cv2.fillConvexPoly(hull, cv2.convexHull(points), 255)
    holes = (hull > 0) & (matte == 0)
    if holes.any():
        _, labels = cv2.distanceTransformWithLabels((matte == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
        layers[holes] = layers[matte > 0][labels[holes] - 1]
    layers[hull == 0] = 0
    return layers, hull


def digital_layers(layers, matte, phase, relief, count=4, spacing=1, x=1, y=1, rotation=0, opacity=1, palette="cobalt"):
    height, width = matte.shape
    if not matte.any():
        return layers,matte,layers[:,:,:3].copy()
    layers,hull = fill_projected_gaps(layers,matte)
    composite = np.zeros((height, width, 3), np.float32)
    step = max(2, round(width * (.009 + .004 * relief))) * spacing
    bright = {"cobalt": (.035,.02,.9), "cyan": (.02,.7,.85), "magenta": (.8,.02,.6), "mono": (.65,.65,.7)}[palette]
    for level in range(count if opacity > 0 else 0, 0, -1):
        shift = cv2.getRotationMatrix2D((width/2,height/2), level * rotation, 1).astype(np.float32)
        shift[0,2] += level * step * x
        shift[1,2] += level * step * y * (.65 + .2 * math.sin(phase * math.tau))
        plate = cv2.warpAffine(hull, shift, (width, height), flags=cv2.INTER_NEAREST)
        color = bright if level % 2 == 0 else (.04, .04, .055)
        target = composite if opacity == 1 else composite.copy()
        target[plate > 0] = color
        contours, _ = cv2.findContours(plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(target, contours, -1, (.85, .9, 1), max(1, width // 640), cv2.LINE_8)
        if opacity != 1:
            cv2.addWeighted(target, opacity, composite, 1-opacity, 0, dst=composite)
    composite[hull > 0] = layers[:, :, :3][hull > 0]
    contours, _ = cv2.findContours(hull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(composite, contours, -1, (1, 1, 1), max(1, width // 320), cv2.LINE_8)
    return layers, hull, composite


class FL_StreetScanComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",), "depth": ("IMAGE",), "normals": ("IMAGE",),
            "subject_masks": ("MASK",), "detections": ("FL_SCAN_TRACKS",),
            "fps": ("FLOAT", {"default": 24, "min": 1, "max": 120}),
            "seed": ("INT", {"default": 41, "min": 0, "max": 2147483647}),
            "orbit_degrees": ("FLOAT", {"default": 9, "min": 0, "max": 25, "step": 0.5}),
            "depth_relief": ("FLOAT", {"default": 1.4, "min": 0, "max": 3, "step": 0.05}),
            "scene_scale": ("FLOAT", {"default": 0.88, "min": 0.5, "max": 1.0, "step": 0.01}),
            "raggedness": ("FLOAT", {"default": 0.42, "min": 0, "max": 1, "step": 0.01}),
            "normal_mix": ("FLOAT", {"default": 0.85, "min": 0, "max": 1, "step": 0.01}),
            "hud_opacity": ("FLOAT", {"default": 0.85, "min": 0, "max": 1, "step": 0.01}),
            "echo_strength": ("FLOAT", {"default": 0.45, "min": 0, "max": 1, "step": 0.01}),
        }, "optional": {"pose_keypoints": ("POSE_KEYPOINT",),
                          "pose_opacity": ("FLOAT", {"default": 0.65, "min": 0, "max": 1, "step": 0.01}),
                          "edge_style": (["fragment", "digital_layers"], {"default": "fragment",
                              "tooltip": "Digital layers keeps the full scene on stacked cobalt panels. Raggedness and temporal echoes apply only to fragment mode."})}}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("composite", "scene_matte", "projected_normals")
    FUNCTION = "render"
    CATEGORY = "🏵️Fill Nodes/VFX"
    DESCRIPTION = "2.5D depth reprojection with a fragmented boundary or clean stacked digital panels, masked normal-map flashes and detection HUD. Run separately per shot to reset temporal state. This is not a reconstructed multi-view 3D mesh."

    def render(self, images, depth, normals, subject_masks, detections, fps, seed, orbit_degrees, depth_relief,
               scene_scale, raggedness, normal_mix, hud_opacity, echo_strength, pose_keypoints=None, pose_opacity=0.65,
               edge_style="fragment"):
        return self.render_layers(images,depth,normals,subject_masks,detections,fps,seed,orbit_degrees,depth_relief,
            scene_scale,raggedness,normal_mix,hud_opacity,echo_strength,pose_keypoints,pose_opacity,edge_style)[:3]

    def render_layers(self, images, depth, normals, subject_masks, detections, fps, seed, orbit_degrees, depth_relief,
               scene_scale, raggedness, normal_mix, hud_opacity, echo_strength, pose_keypoints=None, pose_opacity=0.65,
               edge_style="fragment", frame_values=None, motion_mode="current", parallax_scope="whole_scene",
               stack_count=4, stack_palette="cobalt"):
        if edge_style not in ("fragment", "digital_layers"):
            raise ValueError("Street Scan edge style must be fragment or digital_layers.")
        count, height, width = images.shape[:3]
        if depth.shape[:3] != images.shape[:3] or normals.shape[:3] != images.shape[:3] or tuple(subject_masks.shape) != (count, height, width):
            raise ValueError("Street Scan requires one aligned depth, normal and subject mask for every source frame.")
        if len(detections["frames"]) != count or (detections["height"], detections["width"]) != (height, width):
            raise ValueError("Street Scan detections must come from the same frame batch and resolution.")
        if pose_keypoints is not None and len(pose_keypoints) != count:
            raise ValueError("Street Scan needs one DWPose keypoint frame for every source frame.")
        output = np.empty((count, height, width, 3), np.float32)
        projected_normals = np.empty_like(output)
        mattes = np.empty((count, height, width), np.float32)
        depth_output = np.empty((count,height,width),np.float32) if frame_values is not None else None
        history = []
        progress = ProgressBar(count)
        colors = {"person": (0.45, 1.0, 0.07), "face": (0.28, 0.48, 1.0), "hand": (1.0, 0.82, 0.12)}
        previous_depth = None
        previous_gray, anchors = None, None
        for frame in range(count):
            if frame_values is not None:
                orbit_degrees,depth_relief,scene_scale,normal_mix,hud_opacity,pose_opacity = [frame_values[k][frame] for k in
                    ("orbit_degrees","depth_relief","scene_scale","normal_mix","hud_opacity","pose_opacity")]
            rgb = images[frame, :, :, :3].cpu().float().numpy().copy()
            d = cv2.GaussianBlur(depth[frame, :, :, 0].cpu().float().numpy(), (0, 0), 1.5)
            if previous_depth is not None:
                d = d * 0.85 + previous_depth * 0.15
            previous_depth = d
            normal = normals[frame, :, :, :3].cpu().float().numpy()
            subject = subject_masks[frame].cpu().float().numpy()
            keep = np.ones((height, width), bool) if edge_style == "digital_layers" else scene_fragment(d, subject, seed, raggedness)
            phase = frame / max(1, count - 1)
            pulse = (frame + seed) % max(1, round(fps * 1.2))
            mix = normal_mix if pulse < 5 else normal_mix * 0.06
            rgb = rgb * (1 - subject[:, :, None] * mix) + normal * subject[:, :, None] * mix
            normal_window = None
            if 8 <= pulse < 19 and normal_mix > 0:
                left = 0.12 if seed % 2 else 0.6
                x1, x2 = round(width * left), round(width * (left + 0.26))
                y1, y2 = round(height * 0.19), round(height * 0.52)
                normal_window = (x1, y1, x2, y2)
                alpha = (1 - subject[y1:y2, x1:x2, None]) * normal_mix
                rgb[y1:y2, x1:x2] = rgb[y1:y2, x1:x2] * (1 - alpha) + normal[y1:y2, x1:x2] * alpha
            px, py, z = project_depth(d, phase, orbit_degrees, depth_relief, scene_scale)
            alternate = None
            if motion_mode == "depth_parallax":
                alternate = project_parallax(d,phase,orbit_degrees,depth_relief,scene_scale,
                    *[frame_values[k][frame] for k in ("parallax_strength","offset_x","offset_y","dolly","steady_depth")])
                if parallax_scope == "whole_scene":
                    px,py,z = alternate
            packed = np.concatenate((rgb,normal,d[:,:,None]),axis=-1) if depth_output is not None else np.concatenate((rgb,normal),axis=-1)
            background = (0.18,0.19,0.20)*2 + ((0,) if depth_output is not None else ())
            layers, matte = splat(packed, keep, px, py, z, background)
            if edge_style == "digital_layers":
                stack = {key.removeprefix("stack_"): frame_values[key][frame] for key in
                    ("stack_spacing","stack_x","stack_y","stack_rotation","stack_opacity") if frame_values is not None and key in frame_values}
                layers, matte, digital = digital_layers(layers, matte, phase, depth_relief, count=stack_count, palette=stack_palette, **stack)
            projected = layers[:, :, :3].copy()
            projected_normals[frame] = layers[:, :, 3:6]
            if depth_output is not None:
                depth_output[frame] = layers[:,:,6]
            if alternate is not None and parallax_scope == "reveals_only":
                reveal_layers,reveal_matte = splat(np.concatenate((normal,d[:,:,None]),axis=-1),keep,*alternate,(0,0,0,0))
                reveal_layers,_ = fill_projected_gaps(reveal_layers,reveal_matte)
                projected_normals[frame] = reveal_layers[:,:,:3]
                depth_output[frame] = reveal_layers[:,:,3]
            composite = digital if edge_style == "digital_layers" else projected.copy()
            for age, (old, old_matte) in enumerate(reversed(history) if edge_style == "fragment" else (), 1):
                transform = np.float32([[1, 0, age * 3], [0, 1, age * 4]])
                ghost = cv2.warpAffine(old, transform, (width, height), borderValue=(0.18, 0.19, 0.20))
                ghost_mask = cv2.warpAffine(old_matte, transform, (width, height))
                alpha = ((matte == 0) & (ghost_mask > 128)).astype(np.float32)[:, :, None] * echo_strength / age
                composite = composite * (1 - alpha) + ghost * alpha
            if edge_style == "fragment":
                history.append((projected, matte))
                history = history[-2:]
            centers = []
            if normal_window is not None:
                x1, y1, x2, y2 = normal_window
                box = [px[y1, x1], py[y1, x1], px[y2, x2], py[y2, x2]]
                draw_box(composite, box, (1.0, 0.52, 0.16), "normal_field", hud_opacity)
            for detection in detections["frames"][frame]:
                x1, y1, x2, y2 = detection["box"]
                xs = np.clip([x1, x2 - 1, x1, x2 - 1], 0, width - 1)
                ys = np.clip([y1, y1, y2 - 1, y2 - 1], 0, height - 1)
                box = [float(px[ys, xs].min()), float(py[ys, xs].min()), float(px[ys, xs].max()), float(py[ys, xs].max())]
                color = colors[detection["label"]]
                text = f"{detection['label']}_{detection['id']:02} {detection['confidence']:.2f}"
                draw_box(composite, box, color, text, hud_opacity)
                centers.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
            if pose_keypoints is not None and pulse >= 19 and pose_opacity > 0:
                pose_frame = pose_keypoints[frame]
                overlay = composite.copy()
                for person in pose_frame["people"]:
                    for part in ("pose_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"):
                        values = person.get(part)
                        if not values:
                            continue
                        points = np.asarray(values, dtype=np.float32).reshape(-1, 3)
                        xs = np.clip(np.rint(points[:, 0] * width / pose_frame["canvas_width"]).astype(int), 0, width - 1)
                        ys = np.clip(np.rint(points[:, 1] * height / pose_frame["canvas_height"]).astype(int), 0, height - 1)
                        projected_points = np.stack([px[ys, xs], py[ys, xs]], axis=-1).astype(int)
                        edges = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (0, 15)] if part == "pose_keypoints_2d" else [(0 if index % 4 == 1 else index - 1, index) for index in range(1, 21)]
                        for index, (a, b) in enumerate(edges):
                            if b >= len(points) or a >= len(points) or min(points[a, 2], points[b, 2]) < 0.3:
                                continue
                            color = ((0.2, 1.0, 0.65), (1.0, 0.75, 0.2), (0.55, 0.45, 1.0))[index % 3]
                            cv2.line(overlay, tuple(projected_points[a]), tuple(projected_points[b]), color, 1, cv2.LINE_AA)
                            cv2.circle(overlay, tuple(projected_points[b]), 2, color, -1, cv2.LINE_AA)
                composite = composite * (1 - pose_opacity) + overlay * pose_opacity
            source_rgb = images[frame, :, :, :3].cpu().float().numpy()
            gray = cv2.cvtColor((source_rgb * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            if previous_gray is not None and anchors is not None:
                moved, valid, error = cv2.calcOpticalFlowPyrLK(previous_gray, gray, anchors, None, winSize=(21, 21), maxLevel=2)
                anchors = moved[(valid[:, 0] == 1) & (error[:, 0] < 30)]
                if not len(anchors):
                    anchors = None
            if anchors is None or len(anchors) < 3:
                anchors = cv2.goodFeaturesToTrack(gray, 5, 0.05, width * 0.16, mask=(keep & (subject < 0.2)).astype(np.uint8) * 255)
            previous_gray = gray
            if anchors is not None and hud_opacity > 0:
                overlay = composite.copy()
                for index, point in enumerate(anchors[:, 0]):
                    x, y = np.clip(np.rint(point).astype(int), [0, 0], [width - 1, height - 1])
                    target = (int(px[y, x]), int(py[y, x]))
                    if centers:
                        cv2.line(overlay, centers[index % len(centers)], target, (0.47, 0.73, 0.27), 1, cv2.LINE_AA)
                    cv2.drawMarker(overlay, target, (0.7, 0.95, 0.2), cv2.MARKER_CROSS, 7, 1)
                    cv2.putText(overlay, f"Z_REL {d[y,x]:.2f}", (target[0] + 4, target[1] - 4), cv2.FONT_HERSHEY_PLAIN, 0.5, (0.75, 0.95, 0.3), 1, cv2.LINE_AA)
                composite = composite * (1 - hud_opacity) + overlay * hud_opacity
            output[frame] = composite.clip(0, 1)
            mattes[frame] = matte / 255.0
            progress.update(1)
        return (torch.from_numpy(output), torch.from_numpy(mattes), torch.from_numpy(projected_normals),
                torch.from_numpy(depth_output) if depth_output is not None else None)
