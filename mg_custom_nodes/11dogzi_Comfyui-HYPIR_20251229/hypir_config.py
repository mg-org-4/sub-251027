import os

# Try to import folder_paths (ComfyUI specific)
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

from model_downloader import get_hypir_model_path

HYPIR_CONFIG = {
    "default_weight_path": "HYPIR_sd2.pth",
    "default_base_model_path": "stable-diffusion-2-1-base",
    "available_base_models": ["stable-diffusion-2-1-base"],
    "model_t": 200,
    "coeff_t": 100,
    "lora_rank": 256,
    "lora_modules": [
        "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
        "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
    ],
    "patch_size": 512,
    "vae_scale_factor": 8,
}

def get_default_weight_path():
    """Get the default weight path without downloading"""
    return HYPIR_CONFIG["default_weight_path"]

def get_base_model_path(base_model_name):
    """Get the base model path, only download if the base model folder is not in ComfyUI\models\HYPIR"""
    # Check if the HYPIR folder exists in ComfyUI models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(5):  # Up to 5 levels
        parent = os.path.dirname(current_dir)
        if os.path.exists(os.path.join(parent, "models")):
            models_dir = os.path.join(parent, "models")
            hypir_dir = os.path.join(models_dir, "HYPIR")
            
            # Check if HYPIR directory exists
            if os.path.exists(hypir_dir):
                # If HYPIR directory exists, check if there is a base model folder
                # For stable-diffusion-2-1-base, check if there is a corresponding folder
                base_model_folder = os.path.join(hypir_dir, base_model_name)
                if os.path.exists(base_model_folder):
                    print(f"Found local base model: {base_model_folder}")
                    return base_model_folder
                else:
                    print(f"Local base model not found, will automatically download: stabilityai/{base_model_name}")
                    return f"stabilityai/{base_model_name}"
            else:
                # If HYPIR directory does not exist, create it and return the original path (let diffusers download)
                os.makedirs(hypir_dir, exist_ok=True)
                print(f"Created HYPIR directory: {hypir_dir}")
                print(f"Will automatically download base model: stabilityai/{base_model_name}")
                return f"stabilityai/{base_model_name}"
            break
        current_dir = parent
    
    # If ComfyUI directory not found, return original path
    print(f"ComfyUI directory not found, using original path: stabilityai/{base_model_name}")
    return f"stabilityai/{base_model_name}" 