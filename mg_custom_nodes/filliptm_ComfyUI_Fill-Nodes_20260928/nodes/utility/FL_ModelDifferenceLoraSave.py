import json
import logging
import os

import torch

import comfy.lora
import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths


def materialize_weight(model, key, device):
    weight, _, convert = comfy.model_patcher.get_key_weight(model.model, key)
    if key in model.backup:
        weight = model.backup[key].weight
    if key in model.hook_backup:
        weight = model.hook_backup[key][0]
    weight = comfy.model_management.cast_to_device(weight.detach(), device, torch.float32, copy=True)
    if convert is not None:
        weight = convert(weight, inplace=True)
    return comfy.lora.calculate_weight(model.patches.get(key, []), weight, key)


def factorize_difference(diff, rank):
    matrix = diff.flatten(1)
    rank = min(rank, *matrix.shape)
    if rank == min(matrix.shape):
        up, values, down = torch.linalg.svd(matrix, full_matrices=False)
    else:
        up, values, right = torch.svd_lowrank(matrix, q=min(rank + 16, *matrix.shape), niter=4)
        down = right.T
    up = up[:, :rank] * values[:rank]
    down = down[:rank]
    if diff.ndim == 4:
        up = up.reshape(diff.shape[0], rank, 1, 1)
        down = down.reshape(rank, *diff.shape[1:])
    return up, down


class FL_ModelDifferenceLoraSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "finetuned_model": ("MODEL", {"tooltip": "The fine-tune or merged model to extract."}),
            "base_model": ("MODEL", {"tooltip": "The original model. Difference = fine-tune minus base."}),
            "filename_prefix": ("STRING", {"default": "Krea2/Dirtyrealism_difference"}),
            "rank": ("INT", {"default": 128, "min": 1, "max": 4096}),
            "device": (["auto", "cpu"],),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_path",)
    FUNCTION = "save"
    CATEGORY = "🏵️Fill Nodes/Model"
    OUTPUT_NODE = True
    DESCRIPTION = "Extract fine-tune minus base as a LoRA in the configured LoRA folder. Higher rank preserves more detail. FP8 sources include quantization error."

    def save(self, finetuned_model, base_model, filename_prefix, rank, device):
        if not 1 <= rank <= 4096:
            raise ValueError("LoRA rank must be between 1 and 4096.")
        if device not in ("auto", "cpu"):
            raise ValueError("Choose auto or cpu for extraction.")
        root = folder_paths.get_folder_paths("loras")[0]
        if not filename_prefix.strip() or any(c in filename_prefix for c in ':*?"<>|\x00'):
            raise ValueError("Use a relative LoRA filename prefix, such as Krea2/Dirtyrealism_difference.")
        if not folder_paths.is_within_directory(root, os.path.join(root, filename_prefix)):
            raise ValueError("The LoRA filename must stay inside the LoRA folder.")
        directory, name, counter, _, _ = folder_paths.get_save_image_path(filename_prefix, root)
        path = os.path.join(directory, f"{name}_{counter:05}_.safetensors")
        if not folder_paths.is_within_directory(root, path):
            raise ValueError("The LoRA filename must stay inside the LoRA folder.")

        tuned = dict(finetuned_model.model.named_parameters())
        base = dict(base_model.model.named_parameters())
        keys = [k for k in tuned if k.startswith("diffusion_model.")]
        base_keys = {k for k in base if k.startswith("diffusion_model.")}
        if not keys or set(keys) != base_keys:
            raise ValueError("The fine-tune and base must have the same diffusion model parameters.")
        for key in keys:
            if tuned[key].shape != base[key].shape:
                raise ValueError(f"Model shape mismatch for {key}: {tuned[key].shape} vs {base[key].shape}.")

        compute_device = comfy.model_management.get_torch_device() if device == "auto" else torch.device("cpu")
        output = {}
        progress = comfy.utils.ProgressBar(len(keys))
        total_energy = 0.0
        captured_energy = 0.0
        for index, key in enumerate(keys):
            comfy.model_management.throw_exception_if_processing_interrupted()
            diff = materialize_weight(finetuned_model, key, compute_device)
            diff.sub_(materialize_weight(base_model, key, compute_device))
            if not torch.isfinite(diff).all():
                raise ValueError(f"Non-finite model difference in {key}.")
            energy = diff.square().sum().item()
            total_energy += energy
            if energy:
                lora_key = key[:-7] if key.endswith(".weight") else key
                if key.endswith(".weight") and diff.ndim in (2, 4):
                    up, down = factorize_difference(diff, rank)
                    captured_energy += up.square().sum().item()
                    output[lora_key + ".lora_up.weight"] = up.to(device="cpu", dtype=torch.float16, copy=True).contiguous()
                    output[lora_key + ".lora_down.weight"] = down.to(device="cpu", dtype=torch.float16, copy=True).contiguous()
                    del up, down
                else:
                    output[lora_key + ".diff"] = diff.to(device="cpu", dtype=torch.float16, copy=True).contiguous()
                    captured_energy += energy
            del diff
            progress.update(1)
            if (index + 1) % 16 == 0:
                logging.info("FL LoRA extraction: %s/%s parameters", index + 1, len(keys))

        if not output:
            raise ValueError("The models have no weight differences; no LoRA was saved.")
        if not all(torch.isfinite(t).all() for t in output.values()):
            raise ValueError("The extracted difference exceeds FP16 range; no LoRA was saved.")
        retained = captured_energy / total_energy
        metadata = {"format": "pt", "fl_extraction": json.dumps({
            "operation": "finetuned_model - base_model", "rank": rank,
            "retained_weight_energy": retained,
        })}
        comfy.utils.save_torch_file(output, path, metadata=metadata)
        logging.info("FL LoRA saved: %s (%.2f%% weight-difference energy retained)", path, 100 * retained)
        return (path,)
