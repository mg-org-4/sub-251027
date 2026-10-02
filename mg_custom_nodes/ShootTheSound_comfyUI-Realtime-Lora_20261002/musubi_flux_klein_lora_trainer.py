"""
Musubi Tuner FLUX Klein LoRA Trainer Node for ComfyUI

Trains FLUX Klein LoRAs using kohya-ss/musubi-tuner.
Supports both Klein Base 4B and Klein Base 9B variants via dropdown selector.

Key differences from Z-Image:
- Uses flux_2_*.py training scripts (NOT zimage_*.py)
- Uses --timestep_sampling=flux2_shift (NOT "shift")
- Uses --fp8_text_encoder (NOT --fp8_llm)
- Requires --model_version flag (klein-base-4b or klein-base-9b)
- NO discrete_flow_shift parameter (automatic for FLUX Klein)
- NO LoRA conversion needed (outputs directly usable)
- Uses ae.safetensors VAE from FLUX.2-dev (NOT diffusers VAE)
"""

import os
import sys
import json
import hashlib
import tempfile
import shutil
import subprocess
from datetime import datetime
import numpy as np
from PIL import Image

import folder_paths

from .musubi_flux_klein_config_template import (
    generate_dataset_config,
    save_config,
    FLUX_KLEIN_VARIANTS,
    MUSUBI_FLUX_KLEIN_VRAM_PRESETS,
)


# Global config for Musubi FLUX Klein trainer
_musubi_flux_klein_config = {}
_musubi_flux_klein_config_file = os.path.join(os.path.dirname(__file__), ".musubi_flux_klein_config.json")

# Global cache for trained LoRAs
_musubi_flux_klein_lora_cache = {}
_musubi_flux_klein_cache_file = os.path.join(os.path.dirname(__file__), ".musubi_flux_klein_lora_cache.json")


def _load_musubi_config():
    """Load Musubi config from disk."""
    global _musubi_flux_klein_config
    if os.path.exists(_musubi_flux_klein_config_file):
        try:
            with open(_musubi_flux_klein_config_file, 'r', encoding='utf-8') as f:
                _musubi_flux_klein_config = json.load(f)
        except:
            _musubi_flux_klein_config = {}


def _save_musubi_config():
    """Save Musubi config to disk."""
    try:
        with open(_musubi_flux_klein_config_file, 'w', encoding='utf-8') as f:
            json.dump(_musubi_flux_klein_config, f, indent=2)
    except:
        pass


def _load_musubi_cache():
    """Load Musubi LoRA cache from disk."""
    global _musubi_flux_klein_lora_cache
    if os.path.exists(_musubi_flux_klein_cache_file):
        try:
            with open(_musubi_flux_klein_cache_file, 'r', encoding='utf-8') as f:
                _musubi_flux_klein_lora_cache = json.load(f)
        except:
            _musubi_flux_klein_lora_cache = {}


def _save_musubi_cache():
    """Save Musubi LoRA cache to disk."""
    try:
        with open(_musubi_flux_klein_cache_file, 'w', encoding='utf-8') as f:
            json.dump(_musubi_flux_klein_lora_cache, f)
    except:
        pass


def _compute_image_hash(images, captions, training_steps, learning_rate, lora_rank, vram_mode, output_name, model_variant, dit_model, vae_model, text_encoder, blocks_to_swap, use_folder_path=False):
    """Compute a hash of all images, captions, and training parameters."""
    hasher = hashlib.sha256()

    if use_folder_path:
        # For folder paths, hash the file paths and modification times
        for img_path in images:
            hasher.update(img_path.encode('utf-8'))
            if os.path.exists(img_path):
                hasher.update(str(os.path.getmtime(img_path)).encode('utf-8'))
    else:
        # For tensor inputs, hash the image data
        for img_tensor in images:
            img_np = (img_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            img_bytes = img_np.tobytes()
            hasher.update(img_bytes)

    # Include all captions and model paths in hash
    captions_str = "|".join(captions)
    params_str = f"musubi_flux_klein|{model_variant}|{captions_str}|{training_steps}|{learning_rate}|{lora_rank}|{vram_mode}|{output_name}|{len(images)}|{dit_model}|{vae_model}|{text_encoder}|{blocks_to_swap}"
    hasher.update(params_str.encode('utf-8'))

    return hasher.hexdigest()[:16]


def _get_venv_python_path(musubi_path):
    """Get the Python path for musubi-tuner venv based on platform.
    Checks both .venv (uv default) and venv (traditional) folders."""
    venv_folders = [".venv", "venv"]

    for venv_folder in venv_folders:
        if sys.platform == 'win32':
            python_path = os.path.join(musubi_path, venv_folder, "Scripts", "python.exe")
        else:
            python_path = os.path.join(musubi_path, venv_folder, "bin", "python")

        if os.path.exists(python_path):
            return python_path

    # Return traditional path for error messaging
    if sys.platform == 'win32':
        return os.path.join(musubi_path, "venv", "Scripts", "python.exe")
    else:
        return os.path.join(musubi_path, "venv", "bin", "python")


def _get_accelerate_path(musubi_path):
    """Get the accelerate path for musubi-tuner venv based on platform.
    Checks both .venv (uv default) and venv (traditional) folders."""
    venv_folders = [".venv", "venv"]

    for venv_folder in venv_folders:
        if sys.platform == 'win32':
            accel_path = os.path.join(musubi_path, venv_folder, "Scripts", "accelerate.exe")
        else:
            accel_path = os.path.join(musubi_path, venv_folder, "bin", "accelerate")

        if os.path.exists(accel_path):
            return accel_path

    # Return traditional path for error messaging
    if sys.platform == 'win32':
        return os.path.join(musubi_path, "venv", "Scripts", "accelerate.exe")
    else:
        return os.path.join(musubi_path, "venv", "bin", "accelerate")


def _get_model_path(name, folder_type):
    """Get full path to a model file from ComfyUI folders.
    Returns the name as-is if it's already an absolute path that exists."""
    if not name:
        return ""
    # If it's already an absolute path that exists, use it
    if os.path.isabs(name) and os.path.exists(name):
        return name
    # Try to get from ComfyUI folder
    try:
        return folder_paths.get_full_path(folder_type, name)
    except:
        return name


# Load config and cache on module import
_load_musubi_config()
_load_musubi_cache()


class MusubiFluxKleinLoraTrainer:
    """
    Trains a FLUX Klein LoRA from one or more images using Musubi Tuner.
    Supports both Klein Base 4B and Klein Base 9B variants.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Get saved settings or use defaults
        if sys.platform == 'win32':
            musubi_fallback = 'C:\\musubi-tuner'
        else:
            musubi_fallback = '~/musubi-tuner'

        saved = _musubi_flux_klein_config.get('trainer_settings', {})

        # Get available models from ComfyUI folders
        diffusion_models = folder_paths.get_filename_list("diffusion_models")
        vae_models = folder_paths.get_filename_list("vae")
        # Text encoders can be in clip or text_encoders folder
        try:
            text_encoders = folder_paths.get_filename_list("text_encoders")
        except:
            text_encoders = []
        try:
            clip_models = folder_paths.get_filename_list("clip")
        except:
            clip_models = []
        text_encoder_list = sorted(set(text_encoders + clip_models)) if (text_encoders or clip_models) else ["(no text encoders found)"]

        # Get saved model selections (for default)
        saved_dit = saved.get('dit_model', '')
        saved_vae = saved.get('vae_model', '')
        saved_te = saved.get('text_encoder', '')
        saved_variant = saved.get('model_variant', 'Klein Base 4B')

        # Build dropdown configs with saved defaults if available
        dit_config = {"tooltip": "FLUX Klein DiT model (flux-2-klein-base-4b.safetensors or flux-2-klein-base-9b.safetensors) from diffusion_models folder."}
        if saved_dit and saved_dit in diffusion_models:
            dit_config["default"] = saved_dit

        vae_config = {"tooltip": "FLUX 2 VAE model (ae.safetensors from black-forest-labs/FLUX.2-dev) from vae folder. NOT the diffusers format VAE."}
        if saved_vae and saved_vae in vae_models:
            vae_config["default"] = saved_vae

        te_config = {"tooltip": "Qwen3 text encoder (qwen_3_4b.safetensors for 4B, qwen_3_8b.safetensors for 9B) from text_encoders or clip folder."}
        if saved_te and saved_te in text_encoder_list:
            te_config["default"] = saved_te

        return {
            "required": {
                "model_variant": (["Klein Base 4B", "Klein Base 9B"], {
                    "default": saved_variant,
                    "tooltip": "FLUX Klein model variant. 4B uses Qwen3-4B text encoder, 9B uses Qwen3-8B."
                }),
                "inputcount": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Number of image inputs. Click 'Update inputs' button after changing."}),
                "images_path": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Path to folder containing training images. If provided, images from this folder are used instead of image inputs. Caption .txt files with matching names are used if present."
                }),
                "musubi_path": ("STRING", {
                    "default": _musubi_flux_klein_config.get('musubi_path', musubi_fallback),
                    "tooltip": "Path to musubi-tuner installation."
                }),
                "dit_model": (diffusion_models, dit_config),
                "vae_model": (vae_models, vae_config),
                "text_encoder": (text_encoder_list, te_config),
                "caption": ("STRING", {
                    "default": saved.get('caption', "photo of subject"),
                    "multiline": True,
                    "tooltip": "Default caption for all images. Per-image caption inputs override this."
                }),
                "training_steps": ("INT", {
                    "default": saved.get('training_steps', 400),
                    "min": 10,
                    "max": 5000,
                    "step": 10,
                    "tooltip": "Number of training steps. 400 is a good starting point."
                }),
                "learning_rate": ("FLOAT", {
                    "default": saved.get('learning_rate', 0.0001),
                    "min": 0.00001,
                    "max": 0.1,
                    "step": 0.00001,
                    "tooltip": "Learning rate. 0.0001 is recommended for FLUX Klein training."
                }),
                "lora_rank": ("INT", {
                    "default": saved.get('lora_rank', 32),
                    "min": 4,
                    "max": 128,
                    "step": 4,
                    "tooltip": "LoRA rank/dimension. 32 is recommended for FLUX Klein."
                }),
                "blocks_to_swap": ("INT", {
                    "default": saved.get('blocks_to_swap', 0),
                    "min": 0,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Number of transformer blocks to swap to CPU for VRAM savings. Max 13 for 4B, 16 for 9B. 0 = no swapping."
                }),
                "vram_mode": (["Max (1256px)", "Max (1256px) fp8", "Max (1256px) fp8 offload", "Medium (1024px)", "Medium (1024px) fp8", "Medium (1024px) fp8 offload", "Low (768px)", "Min (512px)"], {
                    "default": saved.get('vram_mode', "Low (768px)"),
                    "tooltip": "VRAM optimization preset. Controls resolution, gradient checkpointing, and fp8 settings."
                }),
                "keep_lora": ("BOOLEAN", {
                    "default": saved.get('keep_lora', True),
                    "tooltip": "If True, keeps the trained LoRA file."
                }),
                "output_name": ("STRING", {
                    "default": saved.get('output_name', "MyLora"),
                    "tooltip": "Custom name for the output LoRA. Timestamp will be appended."
                }),
                "custom_python_exe": ("STRING", {
                    "default": saved.get('custom_python_exe', ""),
                    "tooltip": "Advanced: Optionally enter the full path to a custom python.exe (e.g. C:\\my-venv\\Scripts\\python.exe). If empty, uses the venv inside musubi_path. The musubi_path field is still required for locating training scripts."
                }),
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Training image (not needed if images_path is set)."}),
                "caption_1": ("STRING", {"forceInput": True, "tooltip": "Caption for image_1. Overrides default caption."}),
                "image_2": ("IMAGE", {"tooltip": "Training image."}),
                "caption_2": ("STRING", {"forceInput": True, "tooltip": "Caption for image_2. Overrides default caption."}),
                "image_3": ("IMAGE", {"tooltip": "Training image."}),
                "caption_3": ("STRING", {"forceInput": True, "tooltip": "Caption for image_3. Overrides default caption."}),
                "image_4": ("IMAGE", {"tooltip": "Training image."}),
                "caption_4": ("STRING", {"forceInput": True, "tooltip": "Caption for image_4. Overrides default caption."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_path",)
    OUTPUT_TOOLTIPS = ("Path to the trained FLUX Klein LoRA file.",)
    FUNCTION = "train_flux_klein_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "Trains a FLUX Klein LoRA from images using Musubi Tuner. Supports both 4B and 9B variants."

    def train_flux_klein_lora(
        self,
        model_variant,
        inputcount,
        images_path,
        musubi_path,
        dit_model,
        vae_model,
        text_encoder,
        caption,
        training_steps,
        learning_rate,
        lora_rank,
        blocks_to_swap,
        vram_mode,
        keep_lora=True,
        output_name="MyLora",
        custom_python_exe="",
        image_1=None,
        **kwargs
    ):
        global _musubi_flux_klein_lora_cache

        # Get variant configuration
        variant_config = FLUX_KLEIN_VARIANTS.get(model_variant, FLUX_KLEIN_VARIANTS["Klein Base 4B"])
        model_version = variant_config["model_version"]
        max_blocks = variant_config["max_blocks_to_swap"]

        # Validate and cap blocks_to_swap based on variant
        if blocks_to_swap > max_blocks:
            print(f"[Musubi FLUX Klein] Warning: blocks_to_swap ({blocks_to_swap}) exceeds maximum ({max_blocks}) for {model_variant}. Capping to {max_blocks}.")
            blocks_to_swap = max_blocks

        # Expand paths
        musubi_path = os.path.expanduser(musubi_path.strip())

        # Get full paths from ComfyUI folders
        dit_path = _get_model_path(dit_model, "diffusion_models")
        vae_path = _get_model_path(vae_model, "vae")
        # Try text_encoders first, then clip
        text_encoder_path = _get_model_path(text_encoder, "text_encoders")
        if not text_encoder_path or not os.path.exists(text_encoder_path):
            text_encoder_path = _get_model_path(text_encoder, "clip")

        # Check if using folder path for images
        use_folder_path = False
        folder_images = []
        folder_captions = []

        if images_path and images_path.strip():
            images_path = os.path.expanduser(images_path.strip())
            if os.path.isdir(images_path):
                # Find all image files in the folder
                image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
                for filename in sorted(os.listdir(images_path)):
                    if filename.lower().endswith(image_extensions):
                        img_path = os.path.join(images_path, filename)
                        folder_images.append(img_path)

                        # Look for matching caption file
                        base_name = os.path.splitext(filename)[0]
                        caption_file = os.path.join(images_path, f"{base_name}.txt")
                        if os.path.exists(caption_file):
                            with open(caption_file, 'r', encoding='utf-8') as f:
                                folder_captions.append(f.read().strip())
                        else:
                            folder_captions.append(caption)  # Use default caption

                if folder_images:
                    use_folder_path = True
                    print(f"[Musubi FLUX Klein] Using {len(folder_images)} images from folder: {images_path}")
                else:
                    print(f"[Musubi FLUX Klein] No images found in folder: {images_path}, falling back to inputs")
            else:
                print(f"[Musubi FLUX Klein] Invalid folder path: {images_path}, falling back to inputs")

        if not use_folder_path:
            # Collect all images and captions from inputs
            all_images = []
            all_captions = []

            if image_1 is not None:
                all_images.append(image_1)
                cap_1 = kwargs.get("caption_1", "")
                all_captions.append(cap_1 if cap_1 else caption)

            for i in range(2, inputcount + 1):
                img = kwargs.get(f"image_{i}")
                if img is not None:
                    all_images.append(img)
                    cap = kwargs.get(f"caption_{i}", "")
                    all_captions.append(cap if cap else caption)

            if not all_images:
                raise ValueError("No images provided. Either set images_path to a folder containing images, or connect at least one image input.")

        num_images = len(folder_images) if use_folder_path else len(all_images)
        print(f"[Musubi FLUX Klein] Training {model_variant} with {num_images} image(s)")
        print(f"[Musubi FLUX Klein] DiT: {dit_model}")
        print(f"[Musubi FLUX Klein] VAE: {vae_model}")
        print(f"[Musubi FLUX Klein] Text Encoder: {text_encoder}")
        print(f"[Musubi FLUX Klein] Model version: {model_version}")

        # Get VRAM preset settings
        preset = MUSUBI_FLUX_KLEIN_VRAM_PRESETS.get(vram_mode, MUSUBI_FLUX_KLEIN_VRAM_PRESETS["Low (768px)"])
        print(f"[Musubi FLUX Klein] Using VRAM mode: {vram_mode}")

        # Validate paths
        accelerate_path = _get_accelerate_path(musubi_path)
        train_script = os.path.join(musubi_path, "src", "musubi_tuner", "flux_2_train_network.py")

        if not os.path.exists(accelerate_path):
            raise FileNotFoundError(f"Musubi Tuner accelerate not found at: {accelerate_path}")
        if not os.path.exists(train_script):
            raise FileNotFoundError(f"flux_2_train_network.py not found at: {train_script}")
        if not dit_path or not os.path.exists(dit_path):
            raise FileNotFoundError(f"DiT model not found at: {dit_path}")
        if not vae_path or not os.path.exists(vae_path):
            raise FileNotFoundError(f"VAE model not found at: {vae_path}")
        if not text_encoder_path or not os.path.exists(text_encoder_path):
            raise FileNotFoundError(f"Text encoder not found at: {text_encoder_path}")

        # Save settings
        global _musubi_flux_klein_config
        _musubi_flux_klein_config['musubi_path'] = musubi_path
        _musubi_flux_klein_config['trainer_settings'] = {
            'model_variant': model_variant,
            'dit_model': dit_model,
            'vae_model': vae_model,
            'text_encoder': text_encoder,
            'caption': caption,
            'training_steps': training_steps,
            'learning_rate': learning_rate,
            'lora_rank': lora_rank,
            'blocks_to_swap': blocks_to_swap,
            'vram_mode': vram_mode,
            'keep_lora': keep_lora,
            'output_name': output_name,
            'custom_python_exe': custom_python_exe,
        }
        _save_musubi_config()

        # Compute hash for caching
        if use_folder_path:
            image_hash = _compute_image_hash(folder_images, folder_captions, training_steps, learning_rate, lora_rank, vram_mode, output_name, model_variant, dit_model, vae_model, text_encoder, blocks_to_swap, use_folder_path=True)
        else:
            image_hash = _compute_image_hash(all_images, all_captions, training_steps, learning_rate, lora_rank, vram_mode, output_name, model_variant, dit_model, vae_model, text_encoder, blocks_to_swap, use_folder_path=False)

        # Check cache
        if keep_lora and image_hash in _musubi_flux_klein_lora_cache:
            cached_path = _musubi_flux_klein_lora_cache[image_hash]
            if os.path.exists(cached_path):
                print(f"[Musubi FLUX Klein] Cache hit! Reusing: {cached_path}")
                return (cached_path,)
            else:
                del _musubi_flux_klein_lora_cache[image_hash]
                _save_musubi_cache()

        # Generate run name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        variant_suffix = "4b" if model_variant == "Klein Base 4B" else "9b"
        run_name = f"{output_name}_{variant_suffix}_{timestamp}" if output_name else f"flux_klein_{variant_suffix}_lora_{image_hash}"

        # Output folder
        output_folder = os.path.join(musubi_path, "output")
        os.makedirs(output_folder, exist_ok=True)
        lora_output_path = os.path.join(output_folder, f"{run_name}.safetensors")

        # Auto-increment if file somehow still exists (same second)
        if os.path.exists(lora_output_path):
            counter = 1
            while os.path.exists(os.path.join(output_folder, f"{run_name}_{counter}.safetensors")):
                counter += 1
            run_name = f"{run_name}_{counter}"
            lora_output_path = os.path.join(output_folder, f"{run_name}.safetensors")
            print(f"[Musubi FLUX Klein] Name exists, using: {run_name}")

        # Create temp directory for images
        temp_dir = tempfile.mkdtemp(prefix="comfy_musubi_flux_klein_")
        image_folder = temp_dir  # Musubi uses image_directory directly

        try:
            # Save images with captions
            if use_folder_path:
                # Copy images from folder and create caption files
                for idx, (src_path, cap) in enumerate(zip(folder_images, folder_captions)):
                    ext = os.path.splitext(src_path)[1]
                    dest_path = os.path.join(image_folder, f"image_{idx+1:03d}{ext}")
                    shutil.copy2(src_path, dest_path)

                    caption_path = os.path.join(image_folder, f"image_{idx+1:03d}.txt")
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(cap)
            else:
                # Save tensor images
                for idx, img_tensor in enumerate(all_images):
                    img_data = img_tensor[0]
                    img_np = (img_data.cpu().numpy() * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np)

                    image_path = os.path.join(image_folder, f"image_{idx+1:03d}.png")
                    img_pil.save(image_path, "PNG")

                    caption_path = os.path.join(image_folder, f"image_{idx+1:03d}.txt")
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(all_captions[idx])

            print(f"[Musubi FLUX Klein] Saved {num_images} images to {image_folder}")

            # Generate dataset config
            config_content = generate_dataset_config(
                image_folder=image_folder,
                resolution=preset['resolution'],
                batch_size=preset['batch_size'],
                enable_bucket=True,
            )

            config_path = os.path.join(temp_dir, "dataset_config.toml")
            save_config(config_content, config_path)
            print(f"[Musubi FLUX Klein] Dataset config saved to {config_path}")

            # Set up subprocess environment
            startupinfo = None
            if sys.platform == 'win32':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            # Use custom python exe if provided, otherwise detect from musubi_path
            if custom_python_exe and custom_python_exe.strip():
                python_path = custom_python_exe.strip()
                if not os.path.exists(python_path):
                    raise FileNotFoundError(f"Custom python.exe not found at: {python_path}")
            else:
                python_path = _get_venv_python_path(musubi_path)

            # Pre-cache latents and text encoder outputs (REQUIRED for Musubi training)
            print(f"[Musubi FLUX Klein] Pre-caching latents and text encoder outputs...")

            # Cache latents
            cache_latents_script = os.path.join(musubi_path, "src", "musubi_tuner", "flux_2_cache_latents.py")
            if not os.path.exists(cache_latents_script):
                raise FileNotFoundError(f"flux_2_cache_latents.py not found at: {cache_latents_script}")

            print(f"[Musubi FLUX Klein] Caching VAE latents...")
            cache_latents_cmd = [
                python_path,
                cache_latents_script,
                f"--dataset_config={config_path}",
                f"--vae={vae_path}",
                f"--model_version={model_version}",  # Required for Klein-specific cache suffix
            ]

            cache_latents_process = subprocess.Popen(
                cache_latents_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=musubi_path,
                startupinfo=startupinfo,
                env=env,
            )

            for line in cache_latents_process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[musubi-tuner] {line}")

            cache_latents_process.wait()
            if cache_latents_process.returncode != 0:
                raise RuntimeError(f"Latent caching failed with code {cache_latents_process.returncode}")

            print(f"[Musubi FLUX Klein] VAE latents cached.")

            # Cache text encoder outputs
            cache_te_script = os.path.join(musubi_path, "src", "musubi_tuner", "flux_2_cache_text_encoder_outputs.py")
            if not os.path.exists(cache_te_script):
                raise FileNotFoundError(f"flux_2_cache_text_encoder_outputs.py not found at: {cache_te_script}")

            print(f"[Musubi FLUX Klein] Caching text encoder outputs...")
            cache_te_cmd = [
                python_path,
                cache_te_script,
                f"--dataset_config={config_path}",
                f"--text_encoder={text_encoder_path}",
                f"--model_version={model_version}",
                "--batch_size=1",
            ]

            # Use fp8 for text encoder caching if enabled (note: fp8_text_encoder for FLUX Klein)
            if preset.get('fp8_text_encoder', False):
                cache_te_cmd.append("--fp8_text_encoder")

            cache_te_process = subprocess.Popen(
                cache_te_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=musubi_path,
                startupinfo=startupinfo,
                env=env,
            )

            for line in cache_te_process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[musubi-tuner] {line}")

            cache_te_process.wait()
            if cache_te_process.returncode != 0:
                raise RuntimeError(f"Text encoder caching failed with code {cache_te_process.returncode}")

            print(f"[Musubi FLUX Klein] Text encoder outputs cached.")

            # Build training command
            # Note: FLUX Klein uses flux2_shift timestep sampling and NO discrete_flow_shift
            cmd = [
                accelerate_path,
                "launch",
                "--num_cpu_threads_per_process=1",
                f"--mixed_precision={preset['mixed_precision']}",
                train_script,
                f"--dit={dit_path}",
                f"--vae={vae_path}",
                f"--text_encoder={text_encoder_path}",
                f"--dataset_config={config_path}",
                f"--model_version={model_version}",
                "--sdpa",
                f"--mixed_precision={preset['mixed_precision']}",
                "--timestep_sampling=flux2_shift",
                "--weighting_scheme=none",
                # NO --discrete_flow_shift for FLUX Klein (automatic)
                f"--optimizer_type={preset['optimizer']}",
                f"--learning_rate={learning_rate}",
                f"--network_module=networks.lora_flux_2",
                f"--network_dim={lora_rank}",
                f"--network_alpha={lora_rank}",
                f"--max_train_steps={training_steps}",
                "--max_data_loader_n_workers=2",
                "--persistent_data_loader_workers",
                f"--output_dir={output_folder}",
                f"--output_name={run_name}",
                "--seed=42",
            ]

            # Add memory optimization flags
            if preset['gradient_checkpointing']:
                cmd.append("--gradient_checkpointing")

            if preset['fp8_scaled']:
                cmd.append("--fp8_base")
                cmd.append("--fp8_scaled")

            # Note: FLUX Klein uses --fp8_text_encoder (NOT --fp8_llm)
            if preset.get('fp8_text_encoder', False):
                cmd.append("--fp8_text_encoder")

            # Use user-specified blocks_to_swap if > 0, otherwise use preset
            effective_blocks = blocks_to_swap if blocks_to_swap > 0 else preset.get('blocks_to_swap', 0)
            # Cap to variant max
            if effective_blocks > max_blocks:
                effective_blocks = max_blocks
            if effective_blocks > 0:
                cmd.append(f"--blocks_to_swap={effective_blocks}")

            print(f"[Musubi FLUX Klein] Starting training: {run_name}")
            print(f"[Musubi FLUX Klein] Images: {num_images}, Steps: {training_steps}, LR: {learning_rate}, Rank: {lora_rank}")
            if effective_blocks > 0:
                print(f"[Musubi FLUX Klein] Blocks to swap: {effective_blocks}")

            # Run training
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=musubi_path,
                startupinfo=startupinfo,
                env=env,
            )

            # Stream output
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[musubi-tuner] {line}")

            process.wait()

            if process.returncode != 0:
                raise RuntimeError(f"Musubi Tuner training failed with code {process.returncode}")

            print(f"[Musubi FLUX Klein] Training completed!")

            # Find the trained LoRA (NO conversion needed for FLUX Klein)
            if not os.path.exists(lora_output_path):
                # Check for alternative naming
                possible_files = [f for f in os.listdir(output_folder) if f.startswith(run_name) and f.endswith('.safetensors')]
                if possible_files:
                    lora_output_path = os.path.join(output_folder, possible_files[-1])
                else:
                    raise FileNotFoundError(f"No LoRA file found in {output_folder}")

            print(f"[Musubi FLUX Klein] Trained LoRA: {lora_output_path}")

            # Handle caching
            if keep_lora:
                _musubi_flux_klein_lora_cache[image_hash] = lora_output_path
                _save_musubi_cache()
                print(f"[Musubi FLUX Klein] LoRA saved and cached at: {lora_output_path}")
            else:
                print(f"[Musubi FLUX Klein] LoRA available at: {lora_output_path}")

            return (lora_output_path,)

        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"[Musubi FLUX Klein] Warning: Could not clean up temp dir: {e}")
