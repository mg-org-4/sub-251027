import os
import re
import json
import torch
import folder_paths

from .loader import load_flux_mod
from .sampler import common_ksampler
import comfy.samplers
import comfy.cli_args
from comfy import model_management
import node_helpers


def using_scaled_fp8(model_patcher):
    return (
        comfy.cli_args.args.fast
        and model_management.supports_fp8_compute(model_patcher.load_device)
    ) or model_patcher.model.model_config.scaled_fp8


class FluxModDiffusionLoader:
    NodeId = "FluxModDiffusionLoader"
    NodeName = "FluxMod Unified Model Loader"

    @classmethod
    def INPUT_TYPES(s):
        patches = ["None"] + folder_paths.get_filename_list("diffusion_models")
        checkpoint_paths = folder_paths.get_filename_list("diffusion_models")
        if "unet_gguf" in folder_paths.folder_names_and_paths:
            checkpoint_paths = checkpoint_paths + folder_paths.get_filename_list(
                "unet_gguf"
            )
        return {
            "required": {
                "unet_name": (checkpoint_paths, {"tooltip": "This loads both .safetensors and .GGUF models if ComfyUI-GGUF is installed."}),
                "guidance_name": (patches,),
                "quant_mode": (
                    ["bf16", "float8_e4m3fn (8 bit)", "float8_e5m2 (also 8 bit)"],
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_unet"
    CATEGORY = "advanced/FluxMod"

    def load_unet(
        self, *, unet_name, quant_mode, guidance_name=None, lite_patch_unet_name=None
    ):
        dtypes = {
            "bf16": torch.bfloat16,
            "float8_e4m3fn (8 bit)": torch.float8_e4m3fn,
            "float8_e5m2 (also 8 bit)": torch.float8_e5m2,
        }

        is_gguf = unet_name.lower().endswith(".gguf")
        unet_path = folder_paths.get_full_path(
            "unet_gguf" if is_gguf else "diffusion_models", unet_name
        )
        if guidance_name is not None:
            guidance_path = folder_paths.get_full_path(
                "diffusion_models", guidance_name
            )
        else:
            guidance_path = None
        if lite_patch_unet_name is not None:
            lite_patch_unet_name = folder_paths.get_full_path(
                "diffusion_models", lite_patch_unet_name
            )
        flux_mod = load_flux_mod(
            model_path=unet_path,
            timestep_guidance_path=guidance_path,
            linear_dtypes=dtypes[quant_mode],
            lite_patch_path=lite_patch_unet_name,
            is_gguf=is_gguf,
        )
        return (flux_mod,)


class FluxModDiffusionLoaderMini(FluxModDiffusionLoader):
    NodeId = "FluxModDiffusionLoaderMini"
    NodeName = "FluxMod Mini Unified Model Loader"

    @classmethod
    def INPUT_TYPES(s):
        patches = ["None"] + folder_paths.get_filename_list("diffusion_models")
        result = super().INPUT_TYPES()
        result["required"] |= {
            "lite_patch_unet_name": (patches,),
        }
        return result


class ChromaDiffusionLoader(FluxModDiffusionLoader):
    NodeId = "ChromaDiffusionLoader"
    NodeName = "Chroma Unified Model Loader"

    @classmethod
    def INPUT_TYPES(s):
        result = super().INPUT_TYPES()
        del result["required"]["guidance_name"]
        return result


class ChromaPromptTruncation:
    NodeId = "ChromaPaddingRemoval"
    NodeName = "Padding Removal"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "advanced/conditioning/flux"

    def append(self, conditioning):
        pruning_idx = conditioning[0][1]["attention_mask"].sum() + 1
        conditioning[0][0] = conditioning[0][0][:, :pruning_idx]
        del conditioning[0][1]["attention_mask"]
        conditioning[0][1]["pooled_output"] = torch.zeros(1, 768, dtype=torch.float32)
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": 0})
        return (c,)


class ChromaStyleModelApply:
    NodeId = "ChromaStyleModelApply"
    NodeName = "Chroma Style Model"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "style_model": ("STYLE_MODEL", ),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                             "truncate_percent": ("FLOAT", {"default": 1.00, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip": "Truncates clipvision conditioning to the first truncate_percent values when > 0. Truncates the last |truncate_percent| values when < 0."}),
                             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_stylemodel"

    CATEGORY = "conditioning/style_model"

    def apply_stylemodel(self, conditioning, style_model, clip_vision_output, strength, truncate_percent):
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)

        if truncate_percent < 1.0 and truncate_percent >= 0.0:
            cond_norm = cond.norm(2) # Take stylemodel cond norm before truncation
            cond = cond[:, :int(cond.shape[1] * truncate_percent), :] # Take first (729 * truncate_percent) blocks
            cond *= cond_norm / cond.norm(2) # Re-normalize
        elif truncate_percent < 0.0 and truncate_percent >= -1.0:
            cond_norm = cond.norm(2)
            cond = cond[:, -int(cond.shape[1] * abs(truncate_percent)):, :] # Take last (729 * truncate_percent) blocks
            cond *= cond_norm / cond.norm(2)

        cond *= strength # Normal strength

        c_out = []
        for t in conditioning:
            (txt, keys) = t
            keys = keys.copy()

            c_out.append([torch.cat((txt, cond), dim=1), keys])

        return (c_out,)


class ModelMover:
    NodeId = "ModelMover"
    NodeName = "???"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_unet"
    CATEGORY = "advanced/FluxMod"

    def load_unet(self, unet_name, guidance_name, quant_mode):
        dtypes = {
            "bf16": torch.bfloat16,
            "float8_e4m3fn (8 bit)": torch.float8_e4m3fn,
            "float8_e5m2 (also 8 bit)": torch.float8_e5m2,
        }

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        guidance_path = folder_paths.get_full_path("diffusion_models", guidance_name)
        flux_mod = load_flux_mod(
            model_path=unet_path,
            timestep_guidance_path=guidance_path,
            linear_dtypes=dtypes[quant_mode],
        )
        return (flux_mod,)


class SkipLayerForward:
    NodeId = "SkipLayerForward"
    NodeName = "FluxMod Prune Model Layers"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "skip_mmdit_layers": ("STRING", {"default": "10", "multiline": False}),
                "skip_dit_layers": ("STRING", {"default": "3, 4", "multiline": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "skip_layer"
    CATEGORY = "advanced/FluxMod"

    DESCRIPTION = "Prune model layers"

    def skip_layer(self, model, skip_mmdit_layers, skip_dit_layers):

        skip_mmdit_layers = re.split(r"\s*,\s*", skip_mmdit_layers)
        skip_mmdit_layers = [int(num) for num in skip_mmdit_layers]

        skip_dit_layers = re.split(r"\s*,\s*", skip_dit_layers)
        skip_dit_layers = [int(num) for num in skip_dit_layers]

        model.model.diffusion_model.skip_dit = skip_dit_layers
        model.model.diffusion_model.skip_mmdit = skip_mmdit_layers
        return (model,)


class KSamplerMod:
    NodeId = "KSamplerMod"
    NodeName = "FluxMod KSampler"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The model used for denoising the input latent."},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "The number of steps used in the denoising process.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.",
                    },
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."
                    },
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "tooltip": "The scheduler controls how noise is gradually removed to form the image."
                    },
                ),
                "positive": (
                    "CONDITIONING",
                    {
                        "tooltip": "The conditioning describing the attributes you want to include in the image."
                    },
                ),
                "negative": (
                    "CONDITIONING",
                    {
                        "tooltip": "The conditioning describing the attributes you want to exclude from the image."
                    },
                ),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.",
                    },
                ),
                "activation_casting": (
                    ["bf16", "fp16"],
                    {
                        "default": "bf16",
                        "tooltip": "cast model activation to bf16 or fp16, always use bf16 unless your card does not supports it",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"
    CATEGORY = "advanced/FluxMod"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
        activation_casting="bf16",
    ):
        if using_scaled_fp8(model):
            return common_ksampler(
                model,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
            )
        dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
        with torch.autocast(device_type="cuda", dtype=dtypes[activation_casting]):
            return common_ksampler(
                model,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
            )


class FluxModSamplerWrapperNode:
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"
    CATEGORY = "advanced/FluxMod"
    DESCRIPTION = "Enables FluxMod in float8 quant_mode to be used with advanced sampling nodes by wrapping another SAMPLER. If you are using multiple sampler wrappers, put this node closest to SamplerCustom/SamplerCustomAdvanced/etc."
    NodeId = "FluxModSamplerWrapperNode"
    NodeName = "FluxMod Sampler Wrapper"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "activation_casting": (
                    ("bf16", "fp16"),
                    {
                        "default": "bf16",
                        "tooltip": "Cast model activation to bf16 or fp16. Always use bf16 unless your card does not support it.",
                    },
                ),
            },
        }

    @classmethod
    def go(cls, *, sampler, activation_casting):
        dtype = torch.bfloat16 if activation_casting == "bf16" else torch.float16

        def wrapper(model, *args, **kwargs):
            if using_scaled_fp8(model.inner_model.model_patcher):
                return sampler.sampler_function(model, *args, **kwargs)
            with torch.autocast(
                device_type=model_management.get_torch_device().type, dtype=dtype
            ):
                return sampler.sampler_function(model, *args, **kwargs)

        sampler_wrapper = comfy.samplers.KSAMPLER(
            wrapper,
            extra_options=sampler.extra_options,
            inpaint_options=sampler.inpaint_options,
        )
        return (sampler_wrapper,)


node_list = [
    # loaders
    FluxModDiffusionLoader,
    FluxModDiffusionLoaderMini,
    ChromaDiffusionLoader,
    # Sampler
    KSamplerMod,
    # Modifiers
    FluxModSamplerWrapperNode,
    SkipLayerForward,
    ChromaPromptTruncation,
    ChromaStyleModelApply,
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for node in node_list:
    NODE_CLASS_MAPPINGS[node.NodeId] = node
    NODE_DISPLAY_NAME_MAPPINGS[node.NodeId] = node.NodeName
