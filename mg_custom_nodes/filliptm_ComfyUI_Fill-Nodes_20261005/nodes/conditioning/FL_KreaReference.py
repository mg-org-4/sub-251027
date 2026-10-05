import math

import comfy.model_base
import comfy.samplers
import comfy.utils
from comfy.text_encoders.krea2 import KREA2_TEMPLATE, Krea2Tokenizer
from comfy_api.latest import io


Reference = io.Custom("FL_KREA_REFERENCE")
ROLES = {
    "style": "Borrow the reference's medium, linework, shading, and texture. Use the scene below for the subject and composition, not the reference's subjects or panel layout.",
    "palette": "Borrow the reference's colors and color relationships. Use the scene below for the subjects, composition, and rendering style.",
    "subject": "Use the appearance of the reference's main subject in the scene below. Follow the scene for the setting and composition.",
    "composition": "Borrow the reference's broad arrangement, framing, and spatial relationships. Use the scene below for the subjects and appearance.",
    "custom": "",
}
RESOLUTIONS = (256, 512, 1024, 1280)


def encode_reference(clip, prompt, reference):
    image = reference["image"]
    height, width = image.shape[1:3]
    scale = min(1.0, reference["resolution"] / max(height, width))
    # Qwen3-VL merges 2x2 patches of 16 pixels; preserve at least one merged patch per axis.
    width, height = max(32, round(width * scale)), max(32, round(height * scale))
    image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "area", "disabled").movedim(1, -1)
    instruction = ROLES[reference["role"]]
    text = f"<|vision_start|><|image_pad|><|vision_end|>\n{instruction}\nCreate one coherent image of this scene:\n{prompt}"
    tokens = clip.tokenize(text, images=[image], llama_template=KREA2_TEMPLATE)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if reference["reference_mode"] == "context":
        pairs = tokens["qwen3vl_4b"][0]
        image_end = next(i for i, pair in enumerate(pairs) if isinstance(pair[0], int) and pair[0] == 151653)
        suffix_length = len(pairs) - image_end - 1
        # Keep text states that attended to the image, without passing the image's spatial tokens to the DiT.
        result = []
        for tensor, metadata in conditioning:
            metadata = metadata.copy()
            if "attention_mask" in metadata:
                metadata["attention_mask"] = metadata["attention_mask"][:, -suffix_length:].clone()
            result.append([tensor[:, -suffix_length:].clone(), metadata])
        conditioning = result
    return conditioning


def sigma_envelope(sigma, bounds):
    start, fade_in, fade_out, end = bounds
    if sigma > start or sigma < end:
        return 0.0
    gain = 1.0
    if start > fade_in and sigma > fade_in:
        gain = (start - sigma) / (start - fade_in)
    if fade_out > end and sigma < fade_out:
        gain = min(gain, (sigma - end) / (fade_out - end))
    return gain


class KreaReferenceGuider(comfy.samplers.CFGGuider):
    def __init__(self, model, baseline, branches, influence):
        super().__init__(model)
        self.influence = influence
        sampling = model.get_model_object("model_sampling")
        self.references = []
        conditions = {"positive": baseline}
        for i, (reference, conditioning) in enumerate(branches):
            name = f"reference_{i}"
            start, end = reference["start"], reference["end"]
            fade = (end - start) * reference["fade"]
            bounds = tuple(float(sampling.percent_to_sigma(t)) for t in (start, start + fade, end - fade, end))
            self.references.append((name, reference["weight"], bounds))
            conditions[name] = conditioning
        self.inner_set_conds(conditions)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        if not self.references or self.influence == 0:
            return super().predict_noise(x, timestep, model_options, seed)
        sigma = float(timestep[0])
        weighted = [(name, self.influence * weight * sigma_envelope(sigma, bounds)) for name, weight, bounds in self.references]
        baseline_weight = 1.0 - math.fsum(weight for _, weight in weighted)
        weighted.insert(0, ("positive", baseline_weight))
        result = None
        for name, weight in weighted:
            if weight == 0:
                continue
            prediction = comfy.samplers.sampling_function(
                self.inner_model, x, timestep, None, self.conds[name], 1.0, model_options=model_options, seed=seed)
            if result is None:
                result = prediction if weight == 1 else prediction * weight
            else:
                result = result + prediction * weight
        return result


def reference_cond_batch(args):
    outputs = []
    sigma = float(args["sigma"][0])
    for conditioning in args["conds"]:
        baseline, branches = [], {}
        for item in conditioning or []:
            marker = item.get("fl_krea_reference")
            if marker is None:
                baseline.append(item)
            else:
                branches.setdefault(marker, []).append(item)
        if not branches:
            outputs.append(comfy.samplers.calc_cond_batch(args["model"], [conditioning], args["input"], args["sigma"], args["model_options"])[0])
            continue
        weighted = [(cond, weight * sigma_envelope(sigma, bounds)) for (_, weight, bounds), cond in branches.items()]
        weighted.insert(0, (baseline, 1.0 - math.fsum(weight for _, weight in weighted)))
        result = None
        for cond, weight in weighted:
            if weight == 0:
                continue
            prediction = comfy.samplers.calc_cond_batch(args["model"], [cond], args["input"], args["sigma"], args["model_options"])[0]
            if result is None:
                result = prediction if weight == 1 else prediction * weight
            else:
                result = result + prediction * weight
        outputs.append(result)
    return outputs


class FL_KreaReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_KreaReference", display_name="FL Krea Reference", category="Fill Nodes/Conditioning",
            description="Define what to borrow from one Krea reference. Roles guide interpretation; they do not lock identity or geometry.",
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input("role", options=list(ROLES), default="style"),
                io.Float.Input("weight", default=1.0, min=0.0, max=1.0, step=0.05, tooltip="Reference strength: 0 is off, 0.05 applies 5%, and 1 applies the full contribution before overall influence. Multiple references add their contributions."),
                io.Combo.Input("resolution", options=list(RESOLUTIONS), default=512, tooltip="Longest-side limit before vision encoding."),
                io.Float.Input("start", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("end", default=1.0, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("fade", default=0.0, min=0.0, max=0.5, step=0.01, advanced=True, tooltip="Fraction of the window used to fade at each edge, interpolated in sigma space."),
                io.Combo.Input("reference_mode", options=["context", "full"], default="context", advanced=True,
                               tooltip="Context keeps image-informed text states to reduce copied layouts. Full also supplies visual tokens for stronger source resemblance."),
            ], outputs=[Reference.Output(display_name="reference")])

    @classmethod
    def execute(cls, image, enabled=True, role="style", weight=1.0, resolution=512, start=0.0, end=1.0, fade=0.0, reference_mode="context"):
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
            raise ValueError("Krea Reference needs one RGB image. Select one image from the batch first.")
        if role not in ROLES or resolution not in RESOLUTIONS or reference_mode not in ("context", "full"):
            raise ValueError("Krea Reference: choose a listed role, resolution, and reference mode.")
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Krea Reference weight must be between 0 and 1.")
        if not 0 <= start < end <= 1 or not 0 <= fade <= 0.5:
            raise ValueError("Krea Reference needs 0 <= start < end <= 1 and fade between 0 and 0.5.")
        return io.NodeOutput(dict(image=image, enabled=enabled, role=role, weight=weight,
                                  resolution=resolution, start=start, end=end, fade=fade, reference_mode=reference_mode))


class FL_KreaReferenceGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FL_KreaReferenceGuider", display_name="FL Krea Reference Guider", category="Fill Nodes/Conditioning",
            description="Blend independent Krea reference predictions. Use GUIDER with SamplerCustomAdvanced, or connect both MODEL and CONDITIONING to a KSampler (positive, CFG 1). Each active reference adds a model evaluation.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, default="A dog in the park."),
                io.Float.Input("influence", default=0.7, min=0.0, max=1.0, step=0.05, tooltip="Overall reference strength after blending. Zero uses only the scene prompt."),
                io.Autogrow.Input("references", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=Reference.Input("reference"), prefix="reference_", min=0, max=100)),
                io.Combo.Input("blend_mode", options=["add", "average"], default="add", optional=True,
                               tooltip="Add sums reference contributions. Average blends toward dividing them by the number of enabled references."),
                io.Float.Input("average_amount", default=1.0, min=0.0, max=1.0, step=0.05, optional=True,
                               display_mode=io.NumberDisplay.slider,
                               tooltip="In Average mode: 0 keeps the additive blend; 1 fully averages enabled references. Individual weights still apply. Fading or zero-weight references keep their share; disabled references are excluded."),
            ], outputs=[io.Guider.Output(), io.Model.Output(), io.Conditioning.Output()])

    @classmethod
    def execute(cls, model, clip, prompt, influence=0.7, references=None, blend_mode="add", average_amount=1.0):
        if not isinstance(model.model, comfy.model_base.Krea2) or not isinstance(clip.tokenizer, Krea2Tokenizer):
            raise ValueError("Use a Krea 2 model and CLIPLoader with type 'krea2'.")
        if not 0 <= influence <= 1:
            raise ValueError("Krea reference influence must be between 0 and 1.")
        if blend_mode not in ("add", "average"):
            raise ValueError("Krea Reference Guider: choose add or average.")
        if not 0 <= average_amount <= 1:
            raise ValueError("Krea reference average amount must be between 0 and 1.")
        if any(key in model.model_options for key in ("sampler_cfg_function", "sampler_pre_cfg_function", "sampler_post_cfg_function", "sampler_calc_cond_batch_function")):
            raise ValueError("Krea Reference Guider uses CFG 1. Remove custom CFG/guidance patches from its model input.")
        baseline = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        enabled_references = [reference for reference in (references or {}).values() if reference["enabled"]]
        if blend_mode == "average" and enabled_references:
            influence *= 1.0 - average_amount + average_amount / len(enabled_references)
        branches = []
        if influence > 0:
            for reference in enabled_references:
                if reference["weight"] > 0:
                    branches.append((reference, encode_reference(clip, prompt, reference)))
        guider = KreaReferenceGuider(model, baseline, branches, influence)
        conditioning = list(baseline)
        for (name, weight, bounds), (_, encoded) in zip(guider.references, branches):
            for tensor, metadata in encoded:
                conditioning.append([tensor, {**metadata, "fl_krea_reference": (name, influence * weight, bounds)}])
        sampler_model = model.clone()
        sampler_model.set_model_sampler_calc_cond_batch_function(reference_cond_batch)
        return io.NodeOutput(guider, sampler_model, conditioning)
