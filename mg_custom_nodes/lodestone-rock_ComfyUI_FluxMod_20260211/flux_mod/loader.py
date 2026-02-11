import torch
import torch.nn as nn
from safetensors.torch import safe_open
from .model import FluxParams, FluxMod
from .layers import Approximator
import comfy
import comfy.model_patcher
from comfy import model_management
import comfy.supported_models_base
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
import comfy.conds
import comfy.ops


gguf = None

def ensure_gguf():
    global gguf
    if gguf:
        return
    import sys
    gguf = next(
        (mod for path, mod in sys.modules.items() if path.endswith("ComfyUI-GGUF")),
        None
    )
    if gguf is None:
        raise RuntimeError("Could not find ComfyUI-GGUF node: GGUF support requires ComfyUI-GGUF")

class ExternalFlux(comfy.supported_models_base.BASE):
    unet_config = {}
    unet_extra_config = {}
    latent_format = comfy.latent_formats.Flux
    memory_usage_factor = 2.8
      
    def __init__(self,):
        self.unet_config = {}
        self.latent_format = self.latent_format()
        self.unet_config["disable_unet_model_creation"] = True
    

class ExternalFluxModel(comfy.model_base.BaseModel):
    chroma_model_mode=False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        if self.chroma_model_mode:
            guidance = 0.0
        else:
            guidance = kwargs.get("guidance", 3.5)
            if guidance is None:
                guidance = 0.0
        out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor((guidance,)))
        return out

class ChromaFluxModel(ExternalFluxModel):
        chroma_model_mode=True

def load_selected_keys(filename, exclude_keywords=(), is_gguf=False):
    """Loads all keys from a safetensors file except those containing specified keywords.

    Args:
        filename: Path to the safetensors file.
        exclude_keywords: List of keywords to exclude.

    Returns:
        A dictionary containing the loaded tensors.
    """
    global gguf

    def is_excluded(key):
        return (
            not key.startswith("distilled_guidance_layer.") and
            any(keyword in key for keyword in exclude_keywords)
        )

    if is_gguf:
        # gguf_sd_loader will strip "model.diffusion_model." if it exists, so no need to handle it here.
        return {
            key: value
            for key, value in gguf.loader.gguf_sd_loader(filename).items()
            if not is_excluded(key)
        }
    tensors = {}
    if filename.endswith("pth"):
        state_dict = torch.load(filename, weights_only=True, map_location="cpu")
        for orig_key in state_dict.keys():
            if orig_key.startswith("model.diffusion_model."):
                key = orig_key[22:]
            else:
                key = orig_key
            if not is_excluded(key):
                tensors[key] = state_dict[orig_key]

    elif filename.endswith("safetensors"):
        with safe_open(filename, framework="pt") as f:
            for orig_key in f.keys():
                if orig_key.startswith("model.diffusion_model."):
                    key = orig_key[22:]
                else:
                    key = orig_key
                if not is_excluded(key):
                    tensors[key] = f.get_tensor(orig_key)
    else:
        raise NotImplementedError
    return tensors


def cast_layers(module, layer_type, dtype, exclude_keywords=()):
    """Casts layers in a module to the specified dtype, excluding layers with names containing keywords.

    Args:
        module (nn.Module): The module containing layers to cast.
        layer_type (type): The type of layers to cast (e.g., nn.Linear).
        dtype (torch.dtype): The target data type for casting.
        exclude_keywords (list, optional): A list of keywords to exclude from casting based on layer names. Defaults to [].

    Returns:
        None
    """

    for child_name, child in module.named_children():
        if isinstance(child, layer_type) and not any(keyword in child_name for keyword in exclude_keywords):
            # Cast the layer only if it's the desired type and not excluded
            # omit bias
            child.weight.data = child.weight.data.to(dtype=dtype)
        else:
            # Recursively apply to child modules
            cast_layers(child, layer_type, dtype, exclude_keywords)


def load_flux_mod(model_path, timestep_guidance_path=None, linear_dtypes=torch.bfloat16, lite_patch_path=None, is_gguf=False):

    if is_gguf:
        ensure_gguf()
    # just load safetensors here
    state_dict = load_selected_keys(model_path, {"mod", "time_in", "guidance_in", "vector_in"}, is_gguf=is_gguf)

    if timestep_guidance_path is None:
        # Chroma mode - we expect the timestep guidance to be bundled under
        # the key distilled_guidance_layer.
        if lite_patch_path is not None:
            raise ValueError("Internal error: lite patch specified in Chroma loader mode")
        timestep_state_dict = {
            k[25:]: v
            for k, v in state_dict.items()
            if k.startswith("distilled_guidance_layer.")
        }
        if not timestep_state_dict:
            raise RuntimeError("Could not find distilled guidance layer in Chroma model")
        n_layers = 0
        for key in timestep_state_dict:
            keysplit = key.split(".", 2)
            if (
                len(keysplit) == 3 and
                keysplit[0] == "norms" and
                keysplit[1].isnumeric() and
                keysplit[2] == "scale"
            ):
                n_layers = max(n_layers, int(keysplit[1]) + 1)
            del state_dict[f"distilled_guidance_layer.{key}"]
        if n_layers == 0:
            raise RuntimeError("Could not determine number of distilled guidance layers in Chroma model")
    else:
        timestep_state_dict = comfy.utils.load_torch_file(timestep_guidance_path)

        if "v3" in timestep_guidance_path:
            n_layers = 6
        elif "v2" in timestep_guidance_path:
            n_layers = 5
        else:
            n_layers = 4

    param_count = sum(x.numel() for x in state_dict.values())
    unet_dtype = torch.bfloat16

    load_device = model_management.get_torch_device()
    offload_device = model_management.unet_offload_device()


    params=FluxParams(
        in_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19 if lite_patch_path is None else 8,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        guidance_embed=False,
    )

    model_conf = ExternalFlux()
    model_class = ChromaFluxModel if timestep_guidance_path is None else ExternalFluxModel
    model = model_class(
        model_conf,
        model_type=comfy.model_base.ModelType.FLUX,
        device=model_management.get_torch_device()
    )
    unet_config = model_conf.unet_config
    if is_gguf:
        operations = model_conf.custom_operations = gguf.ops.GGMLOps()
        modelpatcher_class = gguf.nodes.GGUFModelPatcher
    else:
        modelpatcher_class = comfy.model_patcher.ModelPatcher
    if model_conf.custom_operations is None:
        fp8 = model_conf.optimizations.get("fp8", model_conf.scaled_fp8 is not None)
        operations = comfy.ops.pick_operations(
            unet_config.get("dtype"),
            model.manual_cast_dtype,
            fp8_optimizations=fp8,
            scaled_fp8=model_conf.scaled_fp8,
        )
    else:
        operations = model_conf.custom_operations
    
    model.diffusion_model = FluxMod(params=params, dtype=unet_dtype, operations=operations)

    model.diffusion_model.load_state_dict(state_dict)
    
    if lite_patch_path is not None:
        model.diffusion_model.lite = True
        # for _ in range(5,16):
        #     del model.diffusion_model.double_blocks[5]

        # model.diffusion_model.double_blocks[4].load_state_dict(comfy.utils.load_torch_file(lite_patch_path))

    model.diffusion_model.distilled_guidance_layer = Approximator(64 if lite_patch_path is None else 32, 3072, 5120, n_layers, operations=operations)
    model.diffusion_model.distilled_guidance_layer.load_state_dict(timestep_state_dict)
    model.diffusion_model.dtype = unet_dtype
    model.diffusion_model.eval()
    if not is_gguf:
        model.diffusion_model.to(unet_dtype)
        # we cast to fp8 for mixed matmul ops but omit the picky and sensitive layers
        cast_layers(model.diffusion_model, nn.Linear, dtype=linear_dtypes, exclude_keywords={"img_in", "final_layer", "scale"})
        if model_management.force_channels_last():
            model.diffusion_model.to(memory_format=torch.channels_last)

    model_patcher = modelpatcher_class(
        model,
        load_device = load_device,
        offload_device = offload_device,
    )
    return model_patcher
