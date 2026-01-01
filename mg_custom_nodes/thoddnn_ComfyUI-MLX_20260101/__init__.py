import numpy as np
import PIL.Image
import mlx.core as mx
from typing import Optional, Tuple
from PIL import Image
from .diffusionkit.mlx.tokenizer import Tokenizer, T5Tokenizer
from .diffusionkit.mlx.t5 import SD3T5Encoder
from .diffusionkit.mlx import load_t5_encoder, load_t5_tokenizer, load_tokenizer, load_text_encoder
from .diffusionkit.mlx.clip import CLIPTextModel
from .diffusionkit.mlx.model_io import load_flux
from .diffusionkit.mlx import FluxPipeline
import folder_paths
import torch
import os 
import gc
from pathlib import Path


def get_mlx_model_paths():
    """
    Scans for MLX-compatible image generation models in:
    1. HuggingFace cache (~/.cache/huggingface/hub/)
    2. ComfyUI diffusion_models folder
    
    Supports: Flux (schnell, dev, kontext), Stable Diffusion (1.5, 2.1, SDXL, SD3)
    Returns a list of valid model paths.
    """
    model_paths = []
    
    # MLX model patterns to search for
    mlx_patterns = [
        "models--*mlx-FLUX*",        # argmaxinc and other Flux models
        "models--*flux*.mlx*",        # mlx-community Flux models
        "models--mzbac--flux*",       # mzbac Flux variants (schnell, kontext)
        "models--mlx-community--*",   # mlx-community models
        "models--*stable-diffusion*.mlx*",  # MLX SD models
        "models--*sdxl*.mlx*",        # SDXL MLX models
        "models--*sd3*.mlx*",         # SD3 MLX models
    ]
    
    # Check HuggingFace cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        # Search for all MLX model patterns
        for pattern in mlx_patterns:
            for model_dir in hf_cache.glob(pattern):
                # Check if model has actual files (not just empty snapshots)
                snapshots_dir = model_dir / "snapshots"
                if snapshots_dir.exists():
                    # Get the latest snapshot (sorted by name, typically hash-based)
                    snapshots = sorted([d for d in snapshots_dir.iterdir() if d.is_dir()])
                    if snapshots:
                        latest_snapshot = snapshots[-1]
                        # Check if it has model files (.safetensors, .bin, or .npz for MLX)
                        if any(latest_snapshot.glob("*.safetensors")) or \
                           any(latest_snapshot.glob("*.bin")) or \
                           any(latest_snapshot.glob("*.npz")):
                            model_paths.append(str(latest_snapshot))
    
    # Check ComfyUI diffusion_models folder
    try:
        diffusion_folder = folder_paths.get_folder_paths("diffusion_models")
        for folder in diffusion_folder:
            folder_path = Path(folder)
            if folder_path.exists():
                # Look for MLX model directories (broader search)
                for model_dir in folder_path.glob("*mlx*"):
                    if model_dir.is_dir():
                        # Check if it has model files
                        if any(model_dir.glob("*.safetensors")) or \
                           any(model_dir.glob("*.bin")) or \
                           any(model_dir.glob("*.npz")):
                            model_paths.append(str(model_dir))
                # Also check for SD models
                for pattern in ["*flux*", "*stable-diffusion*", "*sdxl*", "*sd3*"]:
                    for model_dir in folder_path.glob(pattern):
                        if model_dir.is_dir():
                            if any(model_dir.glob("*.safetensors")) or \
                               any(model_dir.glob("*.bin")) or \
                               any(model_dir.glob("*.npz")):
                                if str(model_dir) not in model_paths:
                                    model_paths.append(str(model_dir))
    except:
        # folder_paths might not have diffusion_models registered yet
        # Try manual check
        comfy_models = Path(folder_paths.models_dir) / "diffusion_models"
        if comfy_models.exists():
            for model_dir in comfy_models.glob("*mlx*"):
                if model_dir.is_dir():
                    if any(model_dir.glob("*.safetensors")) or \
                       any(model_dir.glob("*.bin")) or \
                       any(model_dir.glob("*.npz")):
                        model_paths.append(str(model_dir))
    
    # If no models found, return a helpful placeholder
    if not model_paths:
        model_paths = ["No local MLX models found - download Flux/SD models first"]
    
    return model_paths

class MLXDecoder:
    """
    Decodes MLX latent representations into images.
    Converts MLX tensors to PyTorch format compatible with ComfyUI.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "latent_image": ("LATENT", ), "mlx_vae": ("mlx_vae", )}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    
    def decode(self, latent_image, mlx_vae):

        decoded = mlx_vae(latent_image)
        decoded = mx.clip(decoded / 2 + 0.5, 0, 1)

        mx.eval(decoded)

        # Convert MLX tensor to numpy array
        decoded_np = np.array(decoded.astype(mx.float16))

        # Convert numpy array to PyTorch tensor
        decoded_torch = torch.from_numpy(decoded_np).float()

        # Ensure the tensor is in the correct format (B, C, H, W)
        if decoded_torch.dim() == 3:
            decoded_torch = decoded_torch.unsqueeze(0)
        
        # Ensure the values are in the range [0, 1]
        decoded_torch = torch.clamp(decoded_torch, 0, 1)

        return (decoded_torch,)


class MLXSampler:
    """
    MLX-optimized sampler for generating images from text conditioning.
    Performs denoising diffusion with configurable steps, CFG, and seed.
    Supports optional negative conditioning for classifier-free guidance.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {"mlx_model": ("mlx_model",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
            "mlx_positive_conditioning": ("mlx_conditioning", ),
            "mlx_negative_conditioning": ("mlx_conditioning", ),
            "latent_image": ("LATENT", ),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate_image"

    def generate_image(self, mlx_model, seed, steps, cfg, mlx_positive_conditioning, latent_image, denoise, mlx_negative_conditioning=None):
        # Ensure seed is within valid range for MLX (32-bit)
        seed = seed & 0xffffffff 
        
        conditioning = mlx_positive_conditioning["conditioning"]
        pooled_conditioning = mlx_positive_conditioning["pooled_conditioning"]
        
        # Handle negative conditioning when CFG is enabled
        if mlx_negative_conditioning is not None and cfg > 0:
            negative_cond = mlx_negative_conditioning["conditioning"]
            negative_pooled = mlx_negative_conditioning["pooled_conditioning"]
            
            # Concatenate positive and negative conditioning
            # Format: [positive, negative] for CFG
            conditioning = mx.concatenate([conditioning, negative_cond], axis=0)
            pooled_conditioning = mx.concatenate([pooled_conditioning, negative_pooled], axis=0)
        
        num_steps = steps 
        cfg_weight = cfg
            
        batch, channels, height, width = latent_image["samples"].shape
        
        # Ensure latent dimensions are integers (ComfyUI sometimes passes floats)
        height = int(height)
        width = int(width)
        
        # Validate dimensions for Flux models
        # Flux requires dimensions divisible by 16 (patch size * vae scale)
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Latent dimensions must be divisible by 16. Got {height}x{width}. "
                f"For a {height*8}x{width*8} pixel image, use a {(height//16)*16}x{(width//16)*16} latent."
            )
        
        latent_size = (height, width)
        
        latents, iter_time  = mlx_model.denoise_latents(
            conditioning,
            pooled_conditioning,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            latent_size=latent_size,
            seed=seed,
            image_path=None,
            denoise=denoise,
        )

        mx.eval(latents)

        latents = latents.astype(mlx_model.activation_dtype)

        return (latents,)


class MLXLoadFlux:
    """
    Loads Flux models from HuggingFace Hub.
    Supports schnell, dev, and 4-bit quantized variants.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_version": ([
                        "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized",
                        "argmaxinc/mlx-FLUX.1-schnell",  
                        "argmaxinc/mlx-FLUX.1-dev"
                        ],)
        }}
    
    RETURN_TYPES = ("mlx_model", "mlx_vae", "mlx_conditioning")
    FUNCTION = "load_flux_model"

    def check_model_folder(self, filename):

        home_dir = os.path.expanduser("~")
        formatted_filename = filename.replace("/", "--")
        folder_path = os.path.join(home_dir, ".cache/huggingface/hub/models--" + formatted_filename)
        
        if os.path.exists(folder_path):
            print("Found existing model folder, verifying download...")
        else:
            print("Model folder not found, downloading from HuggingFace... 🤗")

    def load_flux_model(self, model_version):

        self.check_model_folder(model_version)

        try:
            print(f"Loading {model_version}...")
            model = FluxPipeline(model_version=model_version, low_memory_mode=False, w16=True, a16=True)
        except Exception as e:
            # Provide more helpful error message
            error_msg = str(e)
            if "does not have parameter" in error_msg or "mlp" in error_msg:
                raise RuntimeError(
                    f"Failed to load {model_version}. This may be due to:\n"
                    f"1. Incompatible model weights format\n"
                    f"2. Model not fully downloaded (try deleting cache and re-downloading)\n"
                    f"3. Model architecture mismatch\n\n"
                    f"Original error: {error_msg}\n\n"
                    f"Try: Delete ~/.cache/huggingface/hub/models--{model_version.replace('/', '--')} and restart ComfyUI"
                )
            raise

        clip = {
            "model_name": model_version,
            "clip_l_model": model.clip_l,
            "clip_l_tokenizer": model.tokenizer_l,
            "t5_model": model.t5_encoder,
            "t5_tokenizer": model.t5_tokenizer
        }
        
        print("Model successfully loaded.")
        

        return (model, model.decoder, clip)


class MLXLoadFluxLocal:
    """
    Loads MLX image generation models from local checkpoint files.
    Automatically detects Flux (schnell, dev, kontext), Stable Diffusion (1.5, 2.1, SDXL, SD3),
    and other MLX models from HuggingFace cache and ComfyUI folders.
    """
    @classmethod
    def INPUT_TYPES(s):
        # Get available local models
        available_models = get_mlx_model_paths()
        
        # Add option for custom path
        available_models.append("Custom path (enter below)")
        
        return {"required": {
            "model_version": ([
                        "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized",
                        "argmaxinc/mlx-FLUX.1-schnell",  
                        "argmaxinc/mlx-FLUX.1-dev"
                        ],),
            "local_path": (available_models,)
        },
        "optional": {
            "custom_path": ("STRING", {"default": "", "multiline": False})
        }}
    
    RETURN_TYPES = ("mlx_model", "mlx_vae", "mlx_conditioning")
    FUNCTION = "load_flux_model"

    def load_flux_model(self, model_version, local_path, custom_path=""):
        
        # Use custom path if specified
        if local_path == "Custom path (enter below)":
            if not custom_path or custom_path.strip() == "":
                raise ValueError("Please enter a custom path in the 'custom_path' field")
            local_path = custom_path.strip()
        
        # Check if it's the placeholder message
        if "No local" in local_path and "models found" in local_path:
            raise ValueError(
                "No MLX models found locally. Supported models:\n"
                "• Flux: schnell, dev, kontext (4-bit recommended, ~9GB)\n"
                "• Stable Diffusion: 1.5, 2.1, SDXL, SD3\n"
                "• Any MLX-community models from HuggingFace\n\n"
                "To add models:\n"
                "1. Download using MLXLoadFlux node (auto-cached), or\n"
                "2. Place model files in ComfyUI/models/diffusion_models/, or\n"
                "3. Models download to ~/.cache/huggingface/hub/\n"
                "4. Select 'Custom path (enter below)' to enter path manually"
            )
        
        if not os.path.exists(local_path):
            raise ValueError(f"Local model path does not exist: {local_path}")
        
        print(f"Loading model from local path: {local_path}")

        try:
            model = FluxPipeline(
                model_version=model_version, 
                low_memory_mode=False, 
                w16=True, 
                a16=True,
                local_ckpt=local_path
            )
        except Exception as e:
            # Provide more helpful error message
            error_msg = str(e)
            if "does not have parameter" in error_msg or "mlp" in error_msg:
                raise RuntimeError(
                    f"Failed to load model from {local_path}. This may be due to:\n"
                    f"1. Incompatible model weights format\n"
                    f"2. Corrupted model file\n"
                    f"3. Model architecture mismatch with {model_version}\n\n"
                    f"Original error: {error_msg}\n\n"
                    f"Ensure the local checkpoint matches the selected model version."
                )
            raise

        clip = {
            "model_name": model_version,
            "clip_l_model": model.clip_l,
            "clip_l_tokenizer": model.tokenizer_l,
            "t5_model": model.t5_encoder,
            "t5_tokenizer": model.t5_tokenizer
        }
        
        print("Local model successfully loaded.")
        
        return (model, model.decoder, clip)
    



class MLXClipTextEncoder: 
    """
    Encodes text prompts using CLIP and T5 encoders for MLX-based Flux models.
    Creates conditioning embeddings for image generation.
    For negative prompts with CFG: use two text encoder nodes (positive and negative)
    and connect them to the MLXSampler's positive and negative conditioning inputs.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}), 
                "mlx_conditioning": ("mlx_conditioning", {"forceInput":True})
            }
        }


    RETURN_TYPES = ("mlx_conditioning",)
    FUNCTION = "encode"


    def _tokenize(self, tokenizer, text: str, negative_text: Optional[str] = None):
        if negative_text is None:
            negative_text = ""
        if tokenizer.pad_with_eos:
            pad_token = tokenizer.eos_token
        else:
            pad_token = 0

        text = text.replace('’', '\'')

        # Tokenize the text
        tokens = [tokenizer.tokenize(text)]
        if tokenizer.pad_to_max_length:
            tokens[0].extend([pad_token] * (tokenizer.max_length - len(tokens[0])))
        if negative_text is not None:
            tokens += [tokenizer.tokenize(negative_text)]
        lengths = [len(t) for t in tokens]
        N = max(lengths)
        tokens = [t + [pad_token] * (N - len(t)) for t in tokens]
        tokens = mx.array(tokens)

        return tokens

    def encode(self, mlx_conditioning, text):

        T5_MAX_LENGTH = {
            "argmaxinc/mlx-stable-diffusion-3-medium": 512,
            "argmaxinc/mlx-FLUX.1-schnell": 256,
            "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized": 256,
            "argmaxinc/mlx-FLUX.1-dev": 512,
        }

        model_name = mlx_conditioning["model_name"]
        clip_l_encoder:CLIPTextModel = mlx_conditioning["clip_l_model"]
        clip_l_tokenizer:Tokenizer = mlx_conditioning["clip_l_tokenizer"]
        t5_encoder:SD3T5Encoder = mlx_conditioning["t5_model"]
        t5_tokenizer:T5Tokenizer = mlx_conditioning["t5_tokenizer"]
        
        # CLIP processing
        clip_tokens = self._tokenize(tokenizer=clip_l_tokenizer, text=text) 

        clip_l_embeddings = clip_l_encoder(clip_tokens[[0], :]) 

        clip_last_hidden_state = clip_l_embeddings.last_hidden_state
        clip_pooled_output = clip_l_embeddings.pooled_output
        
        # T5 processing
        t5_tokens = self._tokenize(tokenizer=t5_tokenizer, text=text) 

        padded_tokens_t5 = mx.zeros((1, T5_MAX_LENGTH[model_name])).astype(
            t5_tokens.dtype
        )

        padded_tokens_t5[:, : t5_tokens.shape[1]] = t5_tokens[
            [0], :
        ]  # Ignore negative text

        t5_embeddings = t5_encoder(padded_tokens_t5)

        # Use T5 embeddings as main conditioning
        conditioning = t5_embeddings
        
        output = {
            "conditioning": t5_embeddings,
            "pooled_conditioning": clip_pooled_output
        }



        return (output, ) 

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "MLXClipTextEncoder": MLXClipTextEncoder,
    "MLXLoadFlux": MLXLoadFlux,
    "MLXLoadFluxLocal": MLXLoadFluxLocal,
    "MLXSampler": MLXSampler,
    "MLXDecoder": MLXDecoder
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXClipTextEncoder": "MLX CLIP Text Encoder",
    "MLXLoadFlux": "MLX Load Flux Model from HF 🤗",
    "MLXLoadFluxLocal": "MLX Load Flux Model from Local Path 📁",
    "MLXSampler": "MLX Sampler",
    "MLXDecoder": "MLX Decoder"
}
