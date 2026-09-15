import math

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import nodes
from comfy.utils import ProgressBar
from comfy_api.latest import io


Layer = io.Custom("FL_PARALLAX_LAYER")


class FL_ParallaxStackFromBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_ParallaxStackFromBatch", display_name="FL Parallax Stack From RGBA Batch", category="Fill Nodes/VFX", inputs=[
            io.Image.Input("images", tooltip="RGBA layers ordered back to front, background first. Exclude any reconstruction/composite image."),
            io.Float.Input("near_depth", default=2, min=.25, max=100, step=.25),
            io.Float.Input("far_depth", default=10, min=.25, max=100, step=.25),
        ], outputs=[io.Custom("FL_PARALLAX_STACK").Output(display_name="layer_stack")])

    @classmethod
    def execute(cls, images, near_depth, far_depth):
        if images.ndim != 4 or not len(images) or images.shape[-1] != 4:
            raise ValueError("Parallax stack needs a nonempty RGBA layer batch, background first.")
        if not all(math.isfinite(d) and d > 0 for d in (near_depth, far_depth)) or near_depth > far_depth:
            raise ValueError("Parallax stack depths must be positive, with near depth <= far depth.")
        count = len(images) - 1
        layers = []
        for i in range(count):
            depth = far_depth - (far_depth - near_depth) * i / max(1, count - 1) if count > 1 else near_depth
            layers.append(dict(images=images[i + 1:i + 2].clone(), mask=None, depth=depth,
                               scale=1, offset_x=0, offset_y=0, opacity=1, name=f"Layer {i + 1}", kind="art"))
        return io.NodeOutput(dict(background=images[:1].clone(), layers=layers))


class FL_ParallaxDepthSources(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_ParallaxDepthSources", display_name="FL Parallax Depth Sources", category="Fill Nodes/VFX", inputs=[
            io.Custom("FL_PARALLAX_STACK").Input("layer_stack"),
            io.Int.Input("resolution", default=518, min=126, max=1024, step=14),
        ], outputs=[io.Image.Output(display_name="depth_analysis_images")])

    @classmethod
    def execute(cls, layer_stack, resolution):
        images = [layer_stack["background"]] + [p["images"] for p in layer_stack["layers"]]
        h, w = images[0].shape[1:3]
        size = (max(14, round(h * resolution / max(h, w) / 14) * 14), max(14, round(w * resolution / max(h, w) / 14) * 14))
        sources = []
        for image in images:
            if len(image) != 1:
                raise ValueError("Parallax depth analysis currently supports still-image layers only.")
            rgb = image[..., :3]
            if image.shape[-1] == 4:
                alpha = image[..., 3:4].clamp(0, 1)
                rgb = rgb * alpha + .5 * (1 - alpha)
            sources.append(F.interpolate(rgb.movedim(-1, 1), size=size, mode="bilinear", align_corners=False).movedim(1, -1))
        return io.NodeOutput(torch.cat(sources))


def prepare_relief(depth_map, invert, smoothing, device):
    if depth_map.ndim != 4 or depth_map.shape[-1] < 1 or not torch.isfinite(depth_map).all():
        raise ValueError("Parallax relief needs a finite IMAGE depth batch, background first then cutouts in stack order.")
    depth = depth_map[..., :3].mean(-1, keepdim=True).movedim(-1, 1).to(device=device, dtype=torch.float32).clamp(0, 1)
    if invert:
        depth = 1 - depth
    radius = int(smoothing)
    if radius:
        depth = F.avg_pool2d(F.pad(depth, (radius,) * 4, mode="replicate"), 2 * radius + 1, stride=1)
    return depth


class FL_ParallaxLayer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_ParallaxLayer", display_name="FL Parallax Layer", category="Fill Nodes/VFX", inputs=[
            io.Image.Input("images", tooltip="RGBA from a matting node, or RGB with an explicit foreground mask."),
            io.Float.Input("depth", default=4, min=0.25, max=100, step=0.25, tooltip="Camera distance. Smaller values move faster and draw in front."),
            io.Float.Input("scale", default=1, min=0.1, max=4, step=0.01),
            io.Float.Input("offset_x", default=0, min=-2, max=2, step=0.01, tooltip="Fraction of output width. Positive moves right."),
            io.Float.Input("offset_y", default=0, min=-2, max=2, step=0.01, tooltip="Fraction of output height. Positive moves down."),
            io.Float.Input("opacity", default=1, min=0, max=1, step=0.01),
            io.Mask.Input("mask", optional=True, tooltip="White = opaque foreground. Overrides embedded alpha."),
        ], outputs=[Layer.Output(display_name="layer"), io.Image.Output(display_name="matte_check")])

    @classmethod
    def execute(cls, images, depth, scale, offset_x, offset_y, opacity, mask=None):
        if images.ndim != 4 or images.shape[-1] not in (3, 4) or len(images) == 0:
            raise ValueError("Parallax Layer needs a nonempty RGB or RGBA image batch.")
        if not all(math.isfinite(v) for v in (depth, scale, offset_x, offset_y, opacity)) or depth <= 0 or scale <= 0 or not 0 <= opacity <= 1:
            raise ValueError("Parallax Layer needs positive depth/scale and opacity between zero and one.")
        if mask is not None and (mask.ndim != 3 or mask.shape[1:3] != images.shape[1:3] or len(mask) not in (1, len(images))):
            raise ValueError("Parallax mask must match image size, with one frame or one mask per image.")
        if mask is None and images.shape[-1] != 4:
            raise ValueError("Connect RGBA images or a foreground MASK. RGB alone has no cutout alpha.")
        layer = dict(images=images, mask=mask, depth=depth, scale=scale, offset_x=offset_x, offset_y=offset_y, opacity=opacity)
        i = len(images) // 2
        rgb = images[i:i+1, ..., :3]
        alpha = (mask[0:1] if len(mask) == 1 else mask[i:i+1]) if mask is not None else images[i:i+1, ..., 3]
        h, w = images.shape[1:3]
        y = torch.arange(h, device=rgb.device)[:, None] // 16
        x = torch.arange(w, device=rgb.device)[None, :] // 16
        checker = (0.22 + ((x + y) % 2) * 0.12).to(rgb.dtype)[None, ..., None]
        alpha = alpha.to(rgb.device)[..., None].clamp(0, 1) * opacity
        preview = rgb * alpha + checker * (1 - alpha)
        return io.NodeOutput(layer, preview)


def camera_path(count, motion):
    if count == 1 or motion == "locked":
        return [0.0] * count
    values = []
    for i in range(count):
        t = i / (count - 1)
        if motion == "loop":
            value = math.sin(t * math.tau)
        elif motion == "glide":
            value = 2 * (t * t * (3 - 2 * t)) - 1
        else:
            keys = ((0, -1), (.10, -1), (.28, .4), (.45, .4), (.61, -.3), (.73, -.3), (.92, 1), (1, 1))
            for (ta, va), (tb, vb) in zip(keys, keys[1:]):
                if ta <= t <= tb:
                    u = (t - ta) / (tb - ta)
                    value = va + (vb - va) * (u * u * (3 - 2 * u))
                    break
        values.append(value)
    return values


def project_plate(images, mask, start, stop, grid, depth, scale, offset_x, offset_y, opacity, xs, ys, push, width, height, opaque=False, fit="cover", relief_map=None, relief_strength=0, relief_anchor=.5):
    batch = images[0:1] if len(images) == 1 else images[start:stop]
    batch = batch.to(device=grid.device, dtype=torch.float32)
    rgb = batch[..., :3].movedim(-1, 1)
    if opaque:
        alpha = torch.ones_like(rgb[:, :1])
    elif mask is not None:
        alpha = mask[0:1].expand(len(batch), -1, -1) if len(mask) == 1 else mask[start:stop]
        alpha = alpha.to(device=grid.device, dtype=torch.float32).unsqueeze(1).clamp(0, 1)
    else:
        alpha = batch[..., 3:4].movedim(-1, 1).clamp(0, 1)
    alpha = alpha * opacity
    source = torch.cat((rgb * alpha, alpha), dim=1)
    if len(images) == 1:
        source = source.expand(stop-start, -1, -1, -1)
    ih, iw = images.shape[1:3]
    cover = (min if fit == "contain" else max)(width / iw, height / ih)
    sx, sy = iw * cover / width * scale, ih * cover / height * scale
    zoom = depth / (depth - push)
    # Inverse pinhole projection of fronto-parallel planes; neutral framing is independent of depth.
    gx = (grid[..., 0] / zoom[:, None, None] + 2 * xs[:, None, None] / depth - 2 * offset_x) / sx
    gy = (grid[..., 1] / zoom[:, None, None] + 2 * ys[:, None, None] / depth - 2 * offset_y) / sy
    sample_grid = torch.stack((gx, gy), -1)
    if relief_map is not None and relief_strength:
        depth_source = relief_map.expand(stop-start, -1, -1, -1)
        # Two inverse-warp refinements keep RGB and premultiplied alpha on the same surface.
        for _ in range(2):
            sampled = F.grid_sample(depth_source, sample_grid, padding_mode="border", align_corners=False)[:, 0]
            inv_depth = (1 + relief_strength * (sampled - relief_anchor)) / depth
            gx = (grid[..., 0] * (1 - push[:, None, None] * inv_depth) + 2 * xs[:, None, None] * inv_depth - 2 * offset_x) / sx
            gy = (grid[..., 1] * (1 - push[:, None, None] * inv_depth) + 2 * ys[:, None, None] * inv_depth - 2 * offset_y) / sy
            sample_grid = torch.stack((gx, gy), -1)
    return F.grid_sample(source, sample_grid, mode="bilinear", padding_mode="border" if opaque else "zeros", align_corners=False)


def preview_layers(background, plates):
    previews = []
    for i, p in enumerate([dict(images=background, mask=None)] + plates):
        image = p["images"][:1].detach().cpu()
        rgb = image[..., :3]
        alpha = p["mask"][:1].detach().cpu()[..., None] if p["mask"] is not None else image[..., 3:4]
        if i == 0 or image.shape[-1] == 3 and p["mask"] is None:
            alpha = torch.ones_like(rgb[..., :1])
        rgba = torch.cat((rgb, alpha.clamp(0, 1)), -1)
        h, w = rgba.shape[1:3]
        rgba = F.interpolate(rgba.movedim(-1, 1), size=(max(1, round(h * 256 / max(h, w))), max(1, round(w * 256 / max(h, w)))), mode="area").movedim(1, -1)
        file = nodes.SaveImage().save_images(rgba, "Parallax/previews/plate")["ui"]["images"][0]
        previews.append(dict(image=file, width=w, height=h, background=i == 0, name=p.get("name", "Background" if i == 0 else f"Layer {i}"),
                             animated=len(p["images"]) > 1, kind=p.get("kind", "art"), **{k: p.get(k, v) for k, v in dict(depth=12, scale=1, offset_x=0, offset_y=0, opacity=1).items()}))
    return previews


class FL_LayeredParallax(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_LayeredParallax", display_name="FL Layered Parallax", category="Fill Nodes/VFX", inputs=[
            io.Image.Input("background", optional=True, tooltip="Opaque distant plate. Optional when a poster layer stack supplies the background."),
            io.Autogrow.Input("layers", optional=True, template=io.Autogrow.TemplatePrefix(input=Layer.Input("layer"), prefix="layer_", min=0, max=100)),
            io.Int.Input("width", default=640, min=64, max=4096, step=8),
            io.Int.Input("height", default=360, min=64, max=4096, step=8),
            io.Combo.Input("motion", options=["bursts", "glide", "loop", "locked"], default="bursts"),
            io.Float.Input("travel_x", default=0.35, min=-2, max=2, step=0.01, tooltip="Camera excursion in output widths at depth 1. Actual displacement divides by layer depth."),
            io.Float.Input("travel_y", default=0.035, min=-2, max=2, step=0.005),
            io.Float.Input("push_in", default=0.06, min=-0.2, max=0.2, step=0.01, tooltip="Forward camera travel in depth units; zero disables dolly scaling."),
            io.Float.Input("background_depth", default=12, min=0.25, max=100, step=0.25),
            io.Float.Input("overscan", default=1.12, min=1, max=2, step=0.01, tooltip="Extra background coverage. Border extension is used beyond the source; no generated hidden scenery."),
            io.Combo.Input("device", options=["auto", "cpu"], default="auto"),
            io.Int.Input("frames", optional=True, default=0, min=0, max=4096, tooltip="Output frames for still-image plates. Zero uses the source video length. Animated plates must match this length."),
            io.Custom("FL_PARALLAX_STACK").Input("layer_stack", optional=True, tooltip="Dynamic poster layers, including their background. Stack background takes precedence over the separate image input."),
            io.Combo.Input("layer_fit", optional=True, options=["cover", "contain"], default="cover", tooltip="Cover fills the output aspect ratio; contain preserves the whole cutout when source and output shapes differ. Background always covers the canvas."),
            io.Image.Input("depth_maps", optional=True, tooltip="One background map, or a depth batch from FL Parallax Depth Sources in unchanged stack order. White means near."),
            io.Combo.Input("relief_scope", optional=True, options=["off", "background", "background + artwork"], default="off"),
            io.Float.Input("relief_strength", optional=True, default=.2, min=0, max=.5, step=.01),
            io.Float.Input("relief_anchor", optional=True, default=.5, min=0, max=1, step=.01),
            io.Boolean.Input("depth_invert", optional=True, default=False),
            io.Int.Input("depth_smoothing", optional=True, default=2, min=0, max=16),
        ], outputs=[io.Image.Output(display_name="parallax"), io.Image.Output(display_name="locked_composite"), io.Image.Output(display_name="depth_preview")])

    @classmethod
    def execute(cls, background=None, width=640, height=360, motion="bursts", travel_x=.35, travel_y=.035, push_in=.06, background_depth=12, overscan=1.12, device="auto", layers=None, frames=0, layer_stack=None, layer_fit="cover", depth_maps=None, relief_scope="off", relief_strength=.2, relief_anchor=.5, depth_invert=False, depth_smoothing=2):
        if layer_stack is not None:
            background = layer_stack["background"]
        if background is None or background.ndim != 4 or background.shape[-1] < 3 or len(background) == 0:
            raise ValueError("Layered Parallax needs a nonempty background image batch.")
        plates = [v for v in (layers or {}).values() if v is not None]
        if layer_stack is not None:
            plates.extend(layer_stack["layers"])
        plates = [dict(p) for p in plates]
        for i, p in enumerate(plates):
            p["depth_index"] = i + 1
        active_relief = relief_scope != "off" and relief_strength > 0
        render_device = mm.get_torch_device() if device == "auto" else torch.device("cpu")
        background_relief = None
        if active_relief:
            if not 0 <= relief_strength <= .5 or not 0 <= relief_anchor <= 1 or not 0 <= depth_smoothing <= 16:
                raise ValueError("Parallax relief: strength must be 0–0.5, anchor 0–1, smoothing 0–16.")
            if depth_maps is None or len(depth_maps) not in (1, len(plates) + 1):
                raise ValueError("Parallax relief needs one background depth map or one map for every plate, background first.")
            if len(background) != 1 or any(len(p["images"]) != 1 for p in plates):
                raise ValueError("Parallax relief currently supports still-image layers only; use Off for animated plates.")
            if relief_scope == "background + artwork" and plates and len(depth_maps) == 1:
                raise ValueError("Artwork relief needs the complete depth batch from FL Parallax Depth Sources.")
            prepared = prepare_relief(depth_maps, depth_invert, depth_smoothing, render_device)
            background_relief = prepared[:1]
            for i, p in enumerate(plates):
                if relief_scope == "background + artwork" and p.get("kind") != "text":
                    p["relief_map"] = prepared[i+1:i+2]
        plates.sort(key=lambda p: p["depth"], reverse=True)
        count = frames or max([len(background)] + [len(p["images"]) for p in plates])
        if len(background) not in (1, count) or any(len(p["images"]) not in (1, count) for p in plates):
            raise ValueError("Parallax videos must have equal frame counts; single images can be held for the full clip. Trim or resample mismatched videos first.")
        if not all(math.isfinite(v) for v in (travel_x, travel_y, push_in, background_depth, overscan)) or background_depth <= abs(push_in) or any(p["depth"] <= abs(push_in) for p in plates):
            raise ValueError("Camera push must stay in front of every layer; use positive depths greater than the push distance.")
        if any(p["depth"] >= background_depth for p in plates):
            raise ValueError("Background depth must be greater than each cutout layer depth.")
        if active_relief and any(p["depth"] / (1 + relief_strength) <= abs(push_in) for p in [dict(depth=background_depth)] + plates):
            raise ValueError("Reduce camera push or relief: the camera must stay in front of the relieved surface.")
        x = (torch.arange(width, device=render_device, dtype=torch.float32) + .5) * (2 / width) - 1
        y = (torch.arange(height, device=render_device, dtype=torch.float32) + .5) * (2 / height) - 1
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xx, yy), -1).unsqueeze(0)
        path = camera_path(count, motion)
        result = torch.empty((count, height, width, 3), dtype=torch.float32)
        static = len(background) == 1 and all(len(p["images"]) == 1 for p in plates)
        locked = torch.empty((1 if static else count, height, width, 3), dtype=torch.float32)
        depth_preview = None
        progress = ProgressBar(count)
        for start in range(0, count, 4):
            mm.throw_exception_if_processing_interrupted()
            stop = min(count, start + 4)
            t = torch.tensor(path[start:stop], device=render_device, dtype=torch.float32)
            xs, ys, push = t * travel_x, t * travel_y, t * push_in
            neutral = not static or start == 0
            flat_start, flat_stop = (0, 1) if static else (start, stop)
            zero = torch.zeros(flat_stop-flat_start, device=render_device, dtype=torch.float32)
            out = project_plate(background, None, start, stop, grid, background_depth, overscan, 0, 0, 1, xs, ys, push, width, height, True, relief_map=background_relief, relief_strength=relief_strength, relief_anchor=relief_anchor)[:, :3]
            if neutral:
                flat = project_plate(background, None, flat_start, flat_stop, grid, background_depth, overscan, 0, 0, 1, zero, zero, zero, width, height, True)[:, :3]
            middle = count // 2 - start
            depth = torch.zeros_like(out[:1]) if 0 <= middle < stop-start else None
            for p in plates:
                args = (p["images"], p["mask"], start, stop, grid, p["depth"], p["scale"], p["offset_x"], p["offset_y"], p["opacity"])
                rgba = project_plate(*args, xs, ys, push, width, height, fit=layer_fit, relief_map=p.get("relief_map"), relief_strength=relief_strength, relief_anchor=relief_anchor)
                out = out * (1 - rgba[:, 3:4]) + rgba[:, :3]
                if neutral:
                    still = project_plate(p["images"], p["mask"], flat_start, flat_stop, grid, p["depth"], p["scale"], p["offset_x"], p["offset_y"], p["opacity"], zero, zero, zero, width, height, fit=layer_fit)
                    flat = flat * (1 - still[:, 3:4]) + still[:, :3]
                if depth is not None:
                    alpha = rgba[middle:middle+1, 3:4]
                    depth = depth * (1 - alpha) + (1 - p["depth"] / background_depth) * alpha
            result[start:stop] = out.movedim(1, -1).clamp(0, 1).cpu()
            if neutral:
                locked[flat_start:flat_stop] = flat.movedim(1, -1).clamp(0, 1).cpu()
            if depth is not None:
                depth_preview = depth.movedim(1, -1).cpu()
            progress.update(stop-start)
        if static:
            locked = locked.expand(count, -1, -1, -1)
        previews = preview_layers(background, plates)
        for preview, index in zip(previews, [0] + [p["depth_index"] for p in plates]):
            if depth_maps is not None and index < len(depth_maps):
                raw = prepare_relief(depth_maps[index:index+1], False, depth_smoothing, torch.device("cpu"))
                small = F.interpolate(raw, size=(32, 32), mode="bilinear", align_corners=False)[0, 0].cpu().clamp(0, 1)
                preview["relief"] = small.flatten().tolist()
        return io.NodeOutput(result, locked, depth_preview, ui={"parallax_layers": previews,
            "parallax_settings": [dict(width=width, height=height, motion=motion, travel_x=travel_x, travel_y=travel_y, push_in=push_in, background_depth=background_depth, overscan=overscan, device=device, layer_fit=layer_fit, relief_scope=relief_scope, relief_strength=relief_strength, relief_anchor=relief_anchor, depth_invert=depth_invert, depth_smoothing=depth_smoothing)]})
