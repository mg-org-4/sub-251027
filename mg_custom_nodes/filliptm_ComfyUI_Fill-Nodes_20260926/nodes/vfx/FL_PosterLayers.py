import hashlib
import json
import math
import re

import torch.nn.functional as F

import comfy.samplers
import nodes
from comfy_api.latest import io
from comfy_execution.graph_utils import GraphBuilder
from comfy_extras.nodes_textgen import TextGenerate


Plan = io.Custom("FL_POSTER_LAYER_PLAN")
Asset = io.Custom("FL_POSTER_LAYER_ASSET")
Stack = io.Custom("FL_PARALLAX_STACK")


def parse_plan(text, maximum):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        entries = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Poster planner: expected a JSON layer list. Review the generated text or use Manual mode.") from error
    if not isinstance(entries, list):
        raise ValueError("Image layer planner: return a JSON array, not an enclosing object.")
    if not 1 <= len(entries) <= maximum:
        raise ValueError(f"Image layer planner returned {len(entries)} layers; use between 1 and {maximum}, including the background.")
    result = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("kind") not in ("background", "art", "text"):
            raise ValueError("Poster planner: each layer needs kind background, art or text.")
        if (entry["kind"] == "background") != (i == 0):
            raise ValueError("Poster planner: the first layer must be the only background.")
        name, prompt = entry.get("name"), entry.get("prompt")
        if not isinstance(name, str) or not name.strip() or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Poster planner: each layer needs a name and a basic-English extraction prompt.")
        depth = entry.get("depth", 12 if i == 0 else max(1, 10 - i))
        if not isinstance(depth, (int, float)) or not math.isfinite(depth) or not .25 <= depth <= 12 or (i and depth > 11.75):
            raise ValueError("Poster planner: cutout depths must be between 0.25 and 11.75; background depth is 12.")
        result.append(dict(id=f"layer_{i}", name=name.strip(), prompt=prompt.strip(), kind=entry["kind"], depth=12 if i == 0 else depth))
    return result


def plan_key(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:20]


def apply_overrides(plan, text):
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Poster layers: overrides must be a JSON object.")
    overrides = data.get("layers", {}) if data.get("plan_key") == plan_key(plan) else {}
    if not isinstance(overrides, dict):
        raise ValueError("Poster layers: layer edits must be keyed by layer ID.")
    result = []
    for entry in plan:
        item = dict(entry, visible=True, scale=1.04, offset_x=0, offset_y=0, revision=0)
        item["defaults"] = dict(item)
        edit = overrides.get(entry["id"], {})
        if not isinstance(edit, dict):
            raise ValueError("Poster layers: each override must be an object.")
        for key in ("prompt", "depth", "visible", "scale", "offset_x", "offset_y", "revision"):
            if key in edit:
                item[key] = edit[key]
        if not isinstance(item["prompt"], str) or not item["prompt"].strip():
            raise ValueError("Poster layers: extraction prompts cannot be empty.")
        for key, low, high in (("depth", .25, 11.75), ("scale", .1, 4), ("offset_x", -2, 2), ("offset_y", -2, 2)):
            if entry["kind"] == "background" and key == "depth":
                item[key] = 12
                continue
            value = item[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Poster layers: {key} must be between {low} and {high}.")
        if not isinstance(item["visible"], bool) or not isinstance(item["revision"], int) or not 0 <= item["revision"] <= 1000000:
            raise ValueError("Poster layers: visibility must be boolean and reroll revision a nonnegative integer.")
        if entry["kind"] == "background":
            item["visible"] = True
        result.append(item)
    return result


class FL_PosterLayerPlanner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_PosterLayerPlanner", display_name="FL Image Layer Planner", category="Fill Nodes/VFX/Poster", inputs=[
            io.Image.Input("image"), io.Clip.Input("vision_clip", optional=True, lazy=True),
            io.Combo.Input("mode", options=["auto", "manual"], default="auto"),
            io.Int.Input("maximum_layers", default=6, min=1, max=32, tooltip="Extraction budget including the background; Auto may choose fewer layers."),
            io.Combo.Input("grouping", options=["broad groups", "balanced", "individual elements"], default="balanced"),
            io.String.Input("manual_plan", default='[{"name":"Background","kind":"background","prompt":"Plain paper background without text or objects","depth":12}]', multiline=True),
            io.Combo.Input("artwork_type", optional=True, options=["auto", "photography", "illustration", "graphic design", "product / 3D", "painting / collage"], default="auto", tooltip="Guides layer grouping, not style transfer. Auto follows the actual image; text layers are only planned when text is visible."),
        ], outputs=[Plan.Output(display_name="plan"), io.String.Output(display_name="plan_json")])

    @classmethod
    def check_lazy_status(cls, mode, vision_clip=None, **kwargs):
        return ["vision_clip"] if mode == "auto" and vision_clip is None else []

    @classmethod
    def execute(cls, image, mode, maximum_layers, grouping, manual_plan, vision_clip=None, artwork_type="auto"):
        if len(image) != 1:
            raise ValueError("Poster planner: select one poster image, not a video or batch.")
        if mode == "auto":
            if vision_clip is None:
                raise ValueError("Poster planner: connect a vision-capable CLIP for Auto mode.")
            prompt = f'''Inspect this image and plan its extraction into independently moving 2.5D layers. Artwork type: {artwork_type}; auto means infer the medium from the actual image, not from a presumed poster template.
Return ONLY a valid JSON array, with at most {maximum_layers} entries INCLUDING one background. Choose fewer when appropriate. Grouping preference: {grouping}.
Each entry must have: "name" (short label), "kind" ("background", "art", or "text"), "prompt" (a short basic English noun phrase identifying the visible material to extract), "depth" (number).
The first entry must be the only background, depth 12: describe ALL permanent scenery and support surfaces remaining after the cutouts are removed, not just the farthest wall or sky. Include visible floors, ground, tabletops and room surfaces together so objects do not float in an empty backdrop. Preserve their material, perspective and lighting. Do not replace a landscape, room or photographic setting with paper. For graphic layouts, the background is the full canvas or paper ONLY: floating disks, panels and color blocks must be separate art cutouts, never mixed into the background prompt. Group related graphics or matching props into a shared cutout when the budget is tight. Other depths must be 1 through 10; larger depths are behind smaller ones.
Only if lettering is actually visible, include meaningful text groups as text layers, preserving their exact visible words. Do not invent a headline or add text to a text-free image. Describe lettering style and location. Only name a lettering color if clearly visible; do not confuse the ink color with its label background. Never rewrite words.
Name whole subjects with their worn accessories. Group matching decorations and identify all their positions, without guessing their count. In each extraction prompt, name ONLY the target objects. Use image directions for location (upper right, lower left), never other objects as location anchors: write "silver four-point stars along the left and right edges", NOT "stars around the woman and speakers". Do not assign the same object to multiple layers. Do not invent invisible objects. Keep at least a background and the principal foreground material when your budget permits.
Preserve the source medium, photographic realism, brushwork, line work, texture and lighting. For photos and products, keep attached shadows, reflections and fine hair with the relevant subject where separable extraction would break them. Keep continuous surfaces together. For landscapes, prefer coherent foreground, middle ground and distant scenery instead of individual leaves. For graphic design, group related lettering and shapes. Do not split a subject's body into pieces. Fine transparency and reflections may not survive extraction perfectly; prefer fewer coherent layers over fragile tiny fragments.
Use all layers together to reconstruct the complete image. Depth follows visible occlusion, not a fixed person-and-headline template. No Markdown, commentary, bounding boxes or coordinate arrays.'''
            prompt += '\nAssign depth from visible occlusion, NOT list order: distant material 9, middle ground 6, foreground 2 are useful starting points. Worn accessories belong with their subject, not separate layers. Stay within the total layer budget by grouping related elements.'
            text = TextGenerate.execute(vision_clip, prompt, max(1536, maximum_layers * 192), {"sampling_mode": "off"}, image=image, thinking=True).result[0]
            try:
                plan = parse_plan(text, maximum_layers)
            except ValueError as error:
                correction = f"{prompt}\nRevise this draft. Validation failed: {error}\nMerge related objects or lettering to respect the budget, preserving all visible material. Keep subjects and their worn accessories together. Return only the corrected JSON array.\nDraft:\n{text}"
                text = TextGenerate.execute(vision_clip, correction, max(1536, maximum_layers * 192), {"sampling_mode": "off"}, image=image, thinking=True).result[0]
                plan = parse_plan(text, maximum_layers)
        else:
            text = manual_plan
            plan = parse_plan(text, maximum_layers)
        if mode == "auto":
            targets = {}
            for entry in plan:
                if entry["kind"] != "background":
                    # Other-object location anchors make Control extract those objects too.
                    entry["prompt"] = re.split(r"\s+(?:(?:positioned|located|placed)\s+)?(?:behind|beside|around|surrounding|above|below|in front of|(?:on\s+)?(?:either|both|the left|the right)\s+sides?\s+of|to the (?:left|right) of)\b", entry["prompt"], maxsplit=1, flags=re.I)[0].rstrip(" ,;")
                key = (entry["kind"], " ".join(entry["prompt"].casefold().split()))
                if key in targets:
                    targets[key]["name"] += " / " + entry["name"]
                else:
                    targets[key] = entry
            plan = [dict(entry, id=f"layer_{i}") for i, entry in enumerate(targets.values())]
        rendered = json.dumps(plan, ensure_ascii=False, indent=2)
        return io.NodeOutput(plan, rendered, ui={"poster_plan": [plan]})


def extraction_graph(plan, model, clip, vae, image, seed, steps, cfg, sampler_name, scheduler, overrides, display_id=None):
    entries = apply_overrides(plan, overrides)
    graph = GraphBuilder()
    size = graph.node("GetImageSize", "size", image=image)
    init = graph.node("EmptyQwenImageLayeredLatentImage", "init", width=size.out(0), height=size.out(1), layers=0, batch_size=1)
    reference = graph.node("VAEEncode", "reference", pixels=image, vae=vae)
    negative = graph.node("CLIPTextEncode", "negative", clip=clip, text="")
    negative_ref = graph.node("ReferenceLatent", "negative_ref", conditioning=negative.out(0), latent=reference.out(0))
    assets = {}
    for i, entry in enumerate(entries):
        key = entry["id"]
        positive = graph.node("CLIPTextEncode", key + "_text", clip=clip, text=entry["prompt"])
        cond = graph.node("ReferenceLatent", key + "_ref", conditioning=positive.out(0), latent=reference.out(0))
        sample = graph.node("KSampler", key + "_sample", model=model, positive=cond.out(0), negative=negative_ref.out(0), latent_image=init.out(0),
                            seed=(seed + i + entry["revision"] * 100003) % (2**64), steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, denoise=1)
        decode = graph.node("VAEDecode", key + "_decode", samples=sample.out(0), vae=vae)
        asset = graph.node("FL_PosterLayerAsset", key + "_asset", image=decode.out(0), layer_id=key)
        assets["assets.asset_" + str(i)] = asset.out(0)
    stack = graph.node("FL_PosterLayerStack", "stack", layout=json.dumps(entries), plan_key=plan_key(plan), **assets)
    if display_id is not None:
        stack.set_override_display_id(display_id)
    return io.NodeOutput(stack.out(0), expand=graph.finalize())


class FL_PosterLayers(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_PosterLayers", display_name="FL Poster Layers", category="Fill Nodes/VFX/Poster", enable_expand=True, inputs=[
            Plan.Input("plan"), io.Model.Input("model", raw_link=True), io.Clip.Input("clip", raw_link=True),
            io.Vae.Input("vae", raw_link=True), io.Image.Input("image", raw_link=True),
            io.Int.Input("seed", default=777, min=0, max=2**64-1, control_after_generate=True),
            io.Int.Input("steps", default=30, min=1, max=100), io.Float.Input("cfg", default=4, min=0, max=20, step=.1),
            io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
            io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple"),
            io.String.Input("overrides", default="{}", multiline=True, tooltip="Per-layer prompt, depth, placement, visibility and reroll edits. Applied on the next Run."),
        ], outputs=[Stack.Output(display_name="layer_stack")], hidden=[io.Hidden.unique_id])

    @classmethod
    def execute(cls, plan, model, clip, vae, image, seed, steps, cfg, sampler_name, scheduler, overrides):
        return extraction_graph(plan, model, clip, vae, image, seed, steps, cfg, sampler_name, scheduler, overrides, cls.hidden.unique_id)


class FL_PosterLayerAsset(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_PosterLayerAsset", category="Fill Nodes/VFX/Poster/Internal", is_output_node=True, inputs=[
            io.Image.Input("image", lazy=True), io.String.Input("layer_id", default="layer_0"),
        ], outputs=[Asset.Output()])

    @classmethod
    def check_lazy_status(cls, image=None, **kwargs):
        # Give the disk cache a lookup before scheduling extraction ancestors.
        return ["image"] if image is None else []

    @classmethod
    def execute(cls, image, layer_id):
        if not re.fullmatch(r"layer_\d+", layer_id):
            raise ValueError("Poster asset: invalid layer ID.")
        if image.ndim != 4 or len(image) != 1 or image.shape[-1] != 4:
            raise ValueError("Poster extraction needs one RGBA image per layer. Use Qwen Layered Control and its RGBA VAE.")
        saved = nodes.SaveImage().save_images(image, "Dynamic_Parallax_Poster/layers/" + layer_id)["ui"]["images"][0]
        h, w = image.shape[1:3]
        thumb = F.interpolate(image.movedim(-1, 1), size=(max(1, round(h * 192 / max(h, w))), max(1, round(w * 192 / max(h, w)))), mode="area").movedim(1, -1)
        thumbnail = nodes.SaveImage().save_images(thumb, "Dynamic_Parallax_Poster/thumbnails/" + layer_id)["ui"]["images"][0]
        coverage = float((image[..., 3] > .05).float().mean())
        return io.NodeOutput(dict(image=image, file=saved, thumbnail=thumbnail, coverage=coverage),
                             ui={"poster_layer_ready": [dict(id=layer_id, file=saved, thumbnail=thumbnail)]})


class FL_PosterLayerStack(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="FL_PosterLayerStack", category="Fill Nodes/VFX/Poster/Internal", inputs=[
            io.String.Input("layout", default="[]"), io.String.Input("plan_key", default=""),
            io.Autogrow.Input("assets", template=io.Autogrow.TemplatePrefix(input=Asset.Input("asset"), prefix="asset_", min=1, max=100)),
        ], outputs=[Stack.Output()])

    @classmethod
    def execute(cls, layout, plan_key, assets):
        entries = json.loads(layout)
        if len(entries) != len(assets) or not entries or entries[0]["kind"] != "background":
            raise ValueError("Poster stack: the assets must match the plan, starting with its background.")
        layers, review = [], []
        for i, entry in enumerate(entries):
            asset = assets["asset_" + str(i)]
            review.append(dict(entry, file=asset["file"], thumbnail=asset["thumbnail"], coverage=asset["coverage"]))
            if i and entry["visible"]:
                layers.append(dict(images=asset["image"], mask=None, depth=entry["depth"], scale=entry["scale"], offset_x=entry["offset_x"], offset_y=entry["offset_y"], opacity=1, name=entry["name"], kind=entry["kind"]))
        return io.NodeOutput(dict(background=assets["asset_0"]["image"], layers=layers), ui={"poster_layers": review, "poster_plan_key": [plan_key]})
