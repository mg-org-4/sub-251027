import comfy.model_base
from comfy.text_encoders.krea2 import Krea2Tokenizer

from ..conditioning.FL_KreaReference import FL_KreaReference, KreaReferenceGuider, encode_reference, reference_cond_batch
from .FL_KsamplerSEG_common import attach_conditioning, latent_bbox_from_image_bbox, unwrap_regions


class FL_KsamplerSEG_Krea:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "regions": ("SEG_REGIONS",),
            "source_image": ("IMAGE",),
            "prompt": ("STRING", {"multiline": True, "default": "Enhance natural detail while preserving the reference crop's content, composition, lighting and colors. Do not add objects or extend the scene."}),
            "reference_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "0 uses the prompt alone; 1 uses the matching source crop. Intermediate values blend both predictions and require an extra model evaluation."}),
            "reference_resolution": ([256, 512, 1024, 1280], {"default": 512}),
        }}

    RETURN_TYPES = ("MODEL", "SEG_REGIONS", "CONDITIONING")
    RETURN_NAMES = ("model", "regions", "conditioning")
    FUNCTION = "encode"
    CATEGORY = "Fill Nodes/Ksamplers"
    DESCRIPTION = "Encode each SEG sampler crop as its own Krea 2 visual reference. Connect the same resized source used by Regions and VAE Encode. Use all three outputs with FL Ksampler SEG at CFG 1."

    def encode(self, model, clip, regions, source_image, prompt, reference_strength=1.0, reference_resolution=512):
        regions = unwrap_regions(regions)
        if not isinstance(model.model, comfy.model_base.Krea2) or not isinstance(clip.tokenizer, Krea2Tokenizer):
            raise ValueError("Use a Krea 2 model and CLIPLoader with type 'krea2'.")
        if any(key in model.model_options for key in ("sampler_cfg_function", "sampler_pre_cfg_function", "sampler_post_cfg_function", "sampler_calc_cond_batch_function")):
            raise ValueError("Krea region references use CFG 1. Remove custom CFG/guidance patches from the model input.")
        if not 0 <= reference_strength <= 1:
            raise ValueError("Krea region reference strength must be between 0 and 1.")
        height, width = regions["image_size"]
        if source_image.ndim != 4 or source_image.shape[0] != 1 or source_image.shape[-1] != 3 or tuple(source_image.shape[1:3]) != (height, width):
            raise ValueError("Krea region references need one RGB source image at the same size as Regions. Connect the same resized image to both nodes and VAE Encode.")
        downscale = int(model.get_model_object("latent_format").spacial_downscale_ratio)
        if height % downscale or width % downscale:
            raise ValueError(f"Resize the source to multiples of {downscale} pixels so Krea references align with the VAE latent crops.")
        baseline = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        per_region = []
        for bbox in regions["padded_bboxes"]:
            conditioning = list(baseline)
            if reference_strength > 0:
                y0, x0, y1, x1 = latent_bbox_from_image_bbox(bbox, downscale, height // downscale, width // downscale)
                crop = source_image[:, y0 * downscale:min(height, y1 * downscale), x0 * downscale:min(width, x1 * downscale)]
                reference = FL_KreaReference.execute(
                    crop, role="custom",
                    weight=reference_strength, resolution=reference_resolution, reference_mode="full").result[0]
                encoded = encode_reference(clip, prompt, reference)
                guider = KreaReferenceGuider(model, baseline, [(reference, encoded)], 1.0)
                name, weight, bounds = guider.references[0]
                for tensor, metadata in encoded:
                    conditioning.append([tensor, {**metadata, "fl_krea_reference": (name, weight, bounds)}])
            per_region.append((conditioning, baseline))
        sampler_model = model.clone()
        sampler_model.set_model_sampler_calc_cond_batch_function(reference_cond_batch)
        return sampler_model, attach_conditioning(regions, per_region), baseline
