"""
Model Diff to LoRA Node for ComfyUI

Extracts a LoRA from the difference between two MODEL objects.
Captures the combined effect of multiple chained V2 analyzers/selective loaders
into a single distributable LoRA file.

Usage:
    Base Model -> model_before
    Base Model -> V2 Analyzer 1 -> V2 Analyzer 2 -> model_after
    model_before + model_after -> Model Diff to LoRA -> extracted_lora.safetensors
"""

import os
import re
import json
from datetime import datetime

import torch
import folder_paths
import comfy.lora
import comfy.utils
from safetensors.torch import save_file

# Path to store node config (last used save path)
_NODE_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_SAVE_PATH_CONFIG = os.path.join(_NODE_CONFIG_DIR, ".model_diff_save_path.json")


def _load_save_config() -> dict:
    """Load last used save path and filename from config file."""
    if os.path.exists(_SAVE_PATH_CONFIG):
        try:
            with open(_SAVE_PATH_CONFIG, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {'last_save_path': '', 'last_save_filename': 'extracted_lora'}


def _save_save_config(save_path: str, save_filename: str):
    """Save the last used path and filename to config file."""
    try:
        with open(_SAVE_PATH_CONFIG, 'w') as f:
            json.dump({
                'last_save_path': save_path,
                'last_save_filename': save_filename
            }, f, indent=2)
    except Exception as e:
        print(f"[Model Diff to LoRA] Warning: Could not save config: {e}")


def _extract_state_dict(model_patcher, apply_patches: bool = True) -> dict:
    """
    Extract weights from MODEL object (ModelPatcher).

    In ComfyUI, LoRAs are applied as patches stored in model_patcher.patches,
    not baked into the model weights. To get the effective weights after LoRA
    application, we need to apply these patches using comfy.lora.calculate_weight().

    Args:
        model_patcher: ComfyUI ModelPatcher object
        apply_patches: If True, apply any LoRA patches to get effective weights

    Returns:
        State dict with effective weights (after patch application), moved to CPU
    """
    model = model_patcher.model

    # Get the diffusion model (DiT/UNet) - works for all architectures
    if hasattr(model, 'diffusion_model'):
        diff_model = model.diffusion_model
    else:
        diff_model = model

    # Get base state dict
    base_state_dict = diff_model.state_dict()

    # Clone to CPU
    state_dict = {}
    for key, value in base_state_dict.items():
        state_dict[key] = value.detach().clone().cpu()

    # Apply patches if requested and if there are any
    if apply_patches and hasattr(model_patcher, 'patches') and model_patcher.patches:
        patches = model_patcher.patches
        patch_keys = list(patches.keys())
        file_keys = list(state_dict.keys())

        if patch_keys:
            # Detect prefix mapping between patch keys and model keys
            # Patches may have prefixes like 'diffusion_model.' that need to be stripped
            possible_prefixes = ['diffusion_model.', 'model.diffusion_model.', 'model.', '']
            detected_prefix = ''

            for prefix in possible_prefixes:
                test_key = patch_keys[0]
                if test_key.startswith(prefix):
                    stripped_key = test_key[len(prefix):]
                    if stripped_key in file_keys:
                        detected_prefix = prefix
                        break

            # Apply patches using ComfyUI's calculate_weight
            patched_count = 0
            for patch_key, patch_list in patches.items():
                # Strip prefix to match state dict keys
                file_key = patch_key[len(detected_prefix):] if patch_key.startswith(detected_prefix) else patch_key

                if file_key in state_dict:
                    original_weight = state_dict[file_key]
                    try:
                        # calculate_weight applies the LoRA patches to the base weight
                        modified_weight = comfy.lora.calculate_weight(
                            patch_list,
                            original_weight.clone(),
                            file_key,
                            intermediate_dtype=torch.float32
                        )
                        state_dict[file_key] = modified_weight.to(original_weight.dtype).cpu()
                        patched_count += 1
                    except Exception as e:
                        print(f"[Model Diff to LoRA] Warning: Failed to apply patch for {file_key}: {e}")

            if patched_count > 0:
                print(f"[Model Diff to LoRA] Applied {patched_count} patches to get effective weights")

    return state_dict


def _is_lora_trainable_key(key: str) -> bool:
    """
    Check if a key corresponds to a layer that LoRAs typically train.
    This filters out normalization layers, biases, embeddings, etc.
    """
    key_lower = key.lower()

    # Must be a weight (not bias, not norm)
    if not key.endswith('.weight'):
        return False

    # Skip normalization layers
    if any(x in key_lower for x in ['norm', 'ln_', 'layernorm', 'groupnorm', 'rmsnorm']):
        return False

    # Skip embedding layers (usually not trained)
    if 'embed' in key_lower and 'proj' not in key_lower:
        return False

    # Include attention layers (primary LoRA targets)
    if any(x in key_lower for x in ['to_q', 'to_k', 'to_v', 'to_out', 'query', 'key', 'value',
                                     'q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj',
                                     'qkv', 'attention', 'attn']):
        return True

    # Include MLP/FFN layers
    if any(x in key_lower for x in ['mlp', 'ffn', 'ff.', 'feed_forward', 'fc1', 'fc2',
                                     'proj_in', 'proj_out', 'linear1', 'linear2']):
        return True

    # Include projection layers
    if 'proj' in key_lower:
        return True

    # For transformer models, include layers with block indices
    if any(x in key_lower for x in ['layers.', 'blocks.', 'transformer_blocks']):
        return True

    # SDXL/SD15 specific
    if any(x in key_lower for x in ['input_blocks', 'output_blocks', 'middle_block']):
        return True

    return False


def _svd_decompose(weight_diff: torch.Tensor, rank: int, device: str = 'cpu') -> tuple:
    """
    Decompose a weight difference matrix into low-rank LoRA factors using SVD.

    Args:
        weight_diff: The difference tensor to decompose
        rank: Target rank for the decomposition
        device: Device to perform SVD on ('cuda' for GPU, 'cpu' for CPU)

    Returns:
        Tuple of (lora_up, lora_down) tensors on CPU
    """
    original_shape = weight_diff.shape

    # Handle different tensor dimensions
    # LoRA works on 2D matrices, so we need to reshape higher-dim tensors
    if len(original_shape) == 4:
        # Conv2d: (out_channels, in_channels, kH, kW) -> (out_channels, in_channels * kH * kW)
        out_channels = original_shape[0]
        weight_2d = weight_diff.view(out_channels, -1)
    elif len(original_shape) == 2:
        # Linear: already 2D
        weight_2d = weight_diff
    elif len(original_shape) == 1:
        # Bias or 1D - cannot decompose meaningfully
        return None, None
    else:
        # 3D or other - try to flatten last dims
        weight_2d = weight_diff.view(original_shape[0], -1)

    # Get actual dimensions
    m, n = weight_2d.shape

    # Clamp rank to valid range
    actual_rank = min(rank, min(m, n))
    if actual_rank < 1:
        return None, None

    try:
        # Move to target device and ensure float32 for SVD stability
        weight_2d = weight_2d.to(device=device, dtype=torch.float32)

        # Perform SVD on GPU if available (much faster)
        U, S, Vh = torch.linalg.svd(weight_2d, full_matrices=False)

        # Truncate to target rank
        U_r = U[:, :actual_rank]  # (m, r)
        S_r = S[:actual_rank]      # (r,)
        Vh_r = Vh[:actual_rank, :] # (r, n)

        # Distribute singular values: up gets sqrt(S), down gets sqrt(S)
        # This balances the magnitude between the two matrices
        sqrt_S = torch.sqrt(S_r)

        # lora_up: (m, r) - will be multiplied with input from lora_down
        # lora_down: (r, n) - first transformation
        # Use .contiguous() to ensure tensors can be saved by safetensors
        lora_up = (U_r * sqrt_S.unsqueeze(0)).to(dtype=torch.float16, device='cpu').contiguous()
        lora_down = (sqrt_S.unsqueeze(1) * Vh_r).to(dtype=torch.float16, device='cpu').contiguous()

        return lora_up, lora_down

    except Exception as e:
        print(f"[Model Diff to LoRA] SVD failed: {e}")
        return None, None


def _model_key_to_lora_key(model_key: str, architecture: str) -> str:
    """
    Transform a model weight key to LoRA naming convention.

    Model key: "layers.5.attention.to_q.weight"
    LoRA key base: "lora_unet_layers_5_attention_to_q"
    """
    # Remove .weight/.bias suffix
    base_key = model_key
    if base_key.endswith('.weight'):
        base_key = base_key[:-7]
    elif base_key.endswith('.bias'):
        base_key = base_key[:-5]

    # Replace dots with underscores for LoRA convention
    lora_base = base_key.replace('.', '_')

    # Add appropriate prefix based on architecture
    if architecture in ('ZIMAGE', 'FLUX', 'FLUX_KLEIN_4B', 'FLUX_KLEIN_9B', 'WAN', 'QWEN_IMAGE'):
        # These use diffusion_model prefix in the model
        if not lora_base.startswith('lora_'):
            lora_base = f"lora_unet_{lora_base}"
    elif architecture in ('SDXL', 'SD15'):
        if not lora_base.startswith('lora_'):
            lora_base = f"lora_unet_{lora_base}"

    return lora_base


def _detect_architecture_from_keys(keys: list) -> str:
    """
    Detect model architecture from state dict keys.
    Uses patterns from lora_analyzer_v2.py.
    """
    keys_lower = [k.lower() for k in keys]

    # Qwen-Image: transformer_blocks.N with img_mlp/txt_mlp/img_mod/txt_mod
    # Must check BEFORE FLUX since both have transformer_blocks
    if any('transformer_blocks' in k and any(x in k for x in ['img_mlp', 'txt_mlp', 'img_mod', 'txt_mod']) for k in keys_lower):
        return 'QWEN_IMAGE'

    # Z-Image: layers.N.attention (without FLUX-style blocks)
    if any('layers.' in k and ('attention' in k or 'adaln' in k) for k in keys_lower):
        # Check if it's FLUX-style (has transformer blocks) or Z-Image style
        if any('single_transformer_blocks' in k or 'double_blocks' in k for k in keys_lower):
            # Count blocks to distinguish FLUX variants
            double_blocks = set()
            single_blocks = set()
            for k in keys:
                match = re.search(r'double_blocks?[._]?(\d+)', k.lower())
                if match:
                    double_blocks.add(int(match.group(1)))
                match = re.search(r'single_transformer_blocks[._]?(\d+)', k.lower())
                if match:
                    single_blocks.add(int(match.group(1)))
                match = re.search(r'single_blocks[._]?(\d+)', k.lower())
                if match:
                    single_blocks.add(int(match.group(1)))

            num_double = len(double_blocks)
            num_single = len(single_blocks)

            # Klein 4B: 5 double, 20 single
            if 4 <= num_double <= 6 and 18 <= num_single <= 22:
                return 'FLUX_KLEIN_4B'
            # Klein 9B: 8 double, 24 single
            elif 7 <= num_double <= 9 and 22 <= num_single <= 26:
                return 'FLUX_KLEIN_9B'
            # Standard FLUX: 19 double, 38 single
            else:
                return 'FLUX'
        else:
            return 'ZIMAGE'

    # FLUX patterns: double_blocks/single_blocks
    if any('double_blocks' in k or 'single_blocks' in k for k in keys_lower):
        # Count blocks
        double_blocks = set()
        single_blocks = set()
        for k in keys:
            match = re.search(r'double_blocks?[._]?(\d+)', k.lower())
            if match:
                double_blocks.add(int(match.group(1)))
            match = re.search(r'single_transformer_blocks[._]?(\d+)', k.lower())
            if match:
                single_blocks.add(int(match.group(1)))
            match = re.search(r'(?<!transformer_)single_blocks[._]?(\d+)', k.lower())
            if match:
                single_blocks.add(int(match.group(1)))

        num_double = len(double_blocks)
        num_single = len(single_blocks)

        if 4 <= num_double <= 6 and 18 <= num_single <= 22:
            return 'FLUX_KLEIN_4B'
        elif 7 <= num_double <= 9 and 22 <= num_single <= 26:
            return 'FLUX_KLEIN_9B'
        else:
            return 'FLUX'

    # Wan: blocks.N with self_attn/cross_attn/ffn
    if any(('blocks.' in k or 'blocks_' in k) and any(x in k for x in ['self_attn', 'cross_attn', 'ffn']) for k in keys_lower):
        return 'WAN'

    # SDXL/SD15: check for UNet patterns
    if any('input_blocks' in k or 'output_blocks' in k or 'middle_block' in k for k in keys_lower):
        # SDXL has more blocks
        if any('input_blocks.7' in k or 'input_blocks.8' in k for k in keys_lower):
            return 'SDXL'
        return 'SD15'

    return 'UNKNOWN'


class ModelDiffToLoRA:
    """
    Extract a LoRA from the difference between two MODEL objects.

    Use this to capture the combined effect of multiple chained LoRA operations
    (V2 analyzers, selective loaders) into a single distributable LoRA file.

    Workflow:
        1. Connect your base model to model_before
        2. Chain the base through your LoRA processing nodes
        3. Connect the final output to model_after
        4. The node extracts the difference and saves it as a LoRA
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Load last used save path and filename
        config = _load_save_config()
        last_save_path = config.get('last_save_path', '')
        last_save_filename = config.get('last_save_filename', 'extracted_lora')

        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable extraction. Auto-disables after successful save."
                }),
                "model_before": ("MODEL", {
                    "tooltip": "Original base model (before LoRA chain)"
                }),
                "model_after": ("MODEL", {
                    "tooltip": "Modified model (after LoRA chain)"
                }),
                "output_rank": ("INT", {
                    "default": 64,
                    "min": 4,
                    "max": 256,
                    "step": 4,
                    "tooltip": "LoRA rank for SVD decomposition. Higher = more accurate but larger file."
                }),
                "output_path": ("STRING", {
                    "default": last_save_path,
                    "tooltip": "Save directory. Leave empty for ComfyUI/output/extracted_loras. Remembers last used path."
                }),
                "output_name": ("STRING", {
                    "default": last_save_filename,
                    "tooltip": "Filename prefix (timestamp will be appended). Remembers last used name."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_path", "info")
    OUTPUT_TOOLTIPS = (
        "Full path to the saved LoRA file (for chaining to loaders)",
        "Extraction summary with layer count, compression ratio, etc."
    )
    FUNCTION = "extract_lora"
    CATEGORY = "loaders/lora"
    OUTPUT_NODE = True
    DESCRIPTION = """Extracts a LoRA from the difference between two models.

Use this to capture the combined effect of multiple chained LoRA operations
into a single distributable LoRA file.

Connect your original base model to 'model_before' and the final
processed model (after V2 analyzers, selective loaders, etc.) to 'model_after'."""

    def extract_lora(self, enabled, model_before, model_after, output_rank, output_path, output_name):
        # Skip if disabled
        if not enabled:
            print("[Model Diff to LoRA] Extraction disabled, skipping")
            return {"ui": {"auto_disable": [False]}, "result": ("", "Extraction disabled")}

        # Minimum difference threshold - filters out unchanged layers (hardcoded)
        min_diff_threshold = 0.001

        print(f"[Model Diff to LoRA] Starting extraction with rank={output_rank}")

        # Extract state dicts (moves to CPU)
        # Both models get patches applied to capture effective weights
        # model_before: typically the base model (no patches, or some initial patches)
        # model_after: model after LoRA chain (has accumulated patches from V2 analyzers etc.)
        print("[Model Diff to LoRA] Extracting state dict from model_before...")
        state_before = _extract_state_dict(model_before, apply_patches=True)

        print("[Model Diff to LoRA] Extracting state dict from model_after...")
        state_after = _extract_state_dict(model_after, apply_patches=True)

        print(f"[Model Diff to LoRA] Before: {len(state_before)} keys, After: {len(state_after)} keys")

        # Detect architecture
        architecture = _detect_architecture_from_keys(list(state_before.keys()))
        print(f"[Model Diff to LoRA] Detected architecture: {architecture}")

        # Determine device for SVD (GPU is much faster)
        svd_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Model Diff to LoRA] Using {svd_device.upper()} for SVD decomposition")

        # Compute differences and decompose
        lora_dict = {}
        layers_extracted = 0
        layers_skipped_threshold = 0
        layers_skipped_decompose = 0
        total_diff_norm = 0.0

        # Get common keys and filter to LoRA-trainable layers only
        common_keys = set(state_before.keys()) & set(state_after.keys())
        trainable_keys = [k for k in common_keys if _is_lora_trainable_key(k)]
        print(f"[Model Diff to LoRA] Processing {len(trainable_keys)} trainable layers (filtered from {len(common_keys)} total)")

        total_keys = len(trainable_keys)
        pbar = comfy.utils.ProgressBar(total_keys)

        for i, key in enumerate(trainable_keys):
            weight_before = state_before[key]
            weight_after = state_after[key]

            # Skip if shapes don't match
            if weight_before.shape != weight_after.shape:
                print(f"[Model Diff to LoRA] [{i+1}/{total_keys}] {key} - SKIP (shape mismatch)")
                pbar.update(1)
                continue

            # Compute difference
            diff = weight_after.float() - weight_before.float()
            diff_norm = diff.norm().item()

            # Skip if difference is below threshold
            if diff_norm < min_diff_threshold:
                layers_skipped_threshold += 1
                print(f"[Model Diff to LoRA] [{i+1}/{total_keys}] {key} - SKIP (diff={diff_norm:.6f} < threshold)")
                pbar.update(1)
                continue

            total_diff_norm += diff_norm

            # SVD decompose (on GPU if available)
            lora_up, lora_down = _svd_decompose(diff, output_rank, device=svd_device)

            if lora_up is None or lora_down is None:
                layers_skipped_decompose += 1
                print(f"[Model Diff to LoRA] [{i+1}/{total_keys}] {key} - SKIP (SVD failed)")
                pbar.update(1)
                continue

            # Generate LoRA keys
            lora_base = _model_key_to_lora_key(key, architecture)

            # Store with lora_up/lora_down naming
            lora_dict[f"{lora_base}.lora_up.weight"] = lora_up
            lora_dict[f"{lora_base}.lora_down.weight"] = lora_down

            layers_extracted += 1
            print(f"[Model Diff to LoRA] [{i+1}/{total_keys}] {key} - EXTRACTED (diff={diff_norm:.4f}, rank={min(output_rank, min(lora_up.shape))})")

            # Clean up intermediate tensors
            del diff

            # Update progress bar
            pbar.update(1)

        # Clear GPU cache if used
        if svd_device == 'cuda':
            torch.cuda.empty_cache()

        # Clean up state dicts
        del state_before, state_after

        if layers_extracted == 0:
            info = "No layers extracted - models may be identical or differences below threshold"
            print(f"[Model Diff to LoRA] {info}")
            return {"ui": {"auto_disable": [False]}, "result": ("", info)}

        # Calculate compression ratio
        # Original size: sum of all diff matrices
        # Compressed size: sum of lora_up + lora_down for each layer
        original_params = 0
        compressed_params = 0
        for key, value in lora_dict.items():
            compressed_params += value.numel()
            if '.lora_up.weight' in key:
                # lora_up is (m, r), lora_down is (r, n)
                # Original matrix was (m, n)
                m, r = value.shape
                down_key = key.replace('.lora_up.weight', '.lora_down.weight')
                if down_key in lora_dict:
                    n = lora_dict[down_key].shape[1]
                    original_params += m * n

        compression_ratio = original_params / compressed_params if compressed_params > 0 else 0

        # Prepare output path
        if not output_path or not output_path.strip():
            # Default to ComfyUI output directory
            output_path = os.path.join(folder_paths.get_output_directory(), "extracted_loras")
        else:
            output_path = os.path.expanduser(output_path.strip())

        # Ensure directory exists
        os.makedirs(output_path, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_name and output_name.strip():
            base_name = output_name.strip()
            if base_name.lower().endswith('.safetensors'):
                base_name = base_name[:-12]
        else:
            base_name = "extracted_lora"
        filename = f"{base_name}_{timestamp}.safetensors"

        full_path = os.path.join(output_path, filename)

        # Prepare metadata
        metadata = {
            "ss_network_module": "lora",
            "ss_network_dim": str(output_rank),
            "ss_network_alpha": str(output_rank),
            "ss_base_model_version": architecture,
            "modelspec.title": f"Extracted LoRA - {output_name or 'unnamed'}",
            "modelspec.architecture": architecture.lower(),
            "extraction_method": "svd_diff",
            "extraction_rank": str(output_rank),
            "source_layers": str(layers_extracted),
            "extraction_date": datetime.now().isoformat(),
            "extracted_by": "comfyui-zimage-realtime-lora ModelDiffToLoRA",
            "extracted_url": "https://github.com/ShootTheSound/comfyUI-Realtime-Lora",
        }

        # Save LoRA
        print(f"[Model Diff to LoRA] Saving to {full_path}...")
        try:
            save_file(lora_dict, full_path, metadata=metadata)
            print(f"[Model Diff to LoRA] Saved successfully!")

            # Remember the save path and filename for next time
            _save_save_config(output_path, base_name)

        except Exception as e:
            info = f"Error saving LoRA: {e}"
            print(f"[Model Diff to LoRA] {info}")
            return {"ui": {"auto_disable": [False]}, "result": ("", info)}

        # Build info string
        file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
        info_lines = [
            f"Extracted LoRA saved: {filename}",
            f"Architecture: {architecture}",
            f"Layers extracted: {layers_extracted}",
            f"Layers skipped (below threshold): {layers_skipped_threshold}",
            f"Layers skipped (not trainable): {len(common_keys) - len(trainable_keys)}",
            f"Output rank: {output_rank}",
            f"Compression ratio: {compression_ratio:.1f}x",
            f"File size: {file_size_mb:.2f} MB",
            f"Total diff norm: {total_diff_norm:.4f}",
            f"SVD device: {svd_device.upper()}",
        ]
        info = "\n".join(info_lines)

        print(f"[Model Diff to LoRA] {info}")

        # Return with auto_disable=True to signal JS to disable the toggle
        return {"ui": {"auto_disable": [True]}, "result": (full_path, info)}


NODE_CLASS_MAPPINGS = {
    "ModelDiffToLoRA": ModelDiffToLoRA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelDiffToLoRA": "Model Diff to LoRA",
}
