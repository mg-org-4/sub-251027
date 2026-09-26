import json
from pathlib import Path
import uuid

from safetensors import safe_open

import comfy.lora
import comfy.model_patcher
import comfy.utils
import folder_paths
from comfy.ldm.krea2.model import SingleStreamDiT
from comfy.patcher_extension import PatcherInjection, WrappersMP

from .runtime import ACTIVE, SCOPE, FactorBank, ImageInjection, eject, inject, scope_forward


PROJECTIONS = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate",
               "mlp.gate", "mlp.up", "mlp.down")
ALLOWLIST = frozenset(
    f"diffusion_model.blocks.{block}.{projection}.lora_{factor}.weight"
    for block in range(28) for projection in PROJECTIONS for factor in ("A", "B")
)


def select_keys(keys):
    selected = set(keys) & ALLOWLIST
    prefixes = sorted(key.removesuffix(".lora_A.weight") for key in selected
                      if key.endswith(".lora_A.weight")
                      and key.removesuffix(".lora_A.weight") + ".lora_B.weight" in selected)
    return [prefix + suffix for prefix in prefixes for suffix in (".lora_A.weight", ".lora_B.weight")]


def read_selected(path):
    if Path(path).suffix.lower() in (".safetensors", ".sft"):
        with safe_open(path, framework="pt", device="cpu") as source:
            keys = source.keys()
            selected = {key: source.get_tensor(key) for key in select_keys(keys)}
            return selected, len(keys)
    source = comfy.utils.load_torch_file(path, safe_load=True)
    return {key: source[key] for key in select_keys(source)}, len(source)


class Krea2TCharacterLoraImageOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"), {
                "tooltip": "Character LoRA. Loads canonical A/B pairs for Krea2's shared DiT attention and MLP projections; other keys are skipped.",
            }),
            "strength_model": ("FLOAT", {
                "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                "tooltip": "Strength of this character adapter on image-token rows. Text-token rows receive no direct contribution from this adapter.",
            }),
            "enabled": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "load_lora"
    CATEGORY = "Krea2/loaders"
    DESCRIPTION = (
        "Applies a Krea2 character LoRA directly to image tokens and skips unsupported weights. "
        "Use in place of the regular loader for that character. Other LoRAs remain active. "
        "Text and image still interact through the model's normal attention."
    )

    @classmethod
    def IS_CHANGED(cls, model, lora_name, strength_model, enabled=True):
        if not enabled or strength_model == 0:
            return "bypassed"
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        stat = Path(path).stat()
        return path, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def load_lora(self, model, lora_name, strength_model=1.0, enabled=True):
        report = {"source": lora_name, "filter": "Krea2 shared DiT blocks 0–27; canonical A/B pairs",
                  "strength_model": strength_model, "scope": "image-token rows, including reference-image rows"}
        if not enabled or strength_model == 0:
            return model, json.dumps(dict(report, status="bypassed"), indent=2)

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        state, source_count = read_selected(path)
        report.update(source_tensor_keys=source_count, retained_tensor_keys=len(state),
                      skipped_tensor_keys=source_count - len(state))
        if not state:
            return model, json.dumps(dict(report, status="no matching complete A/B pairs"), indent=2)
        if not isinstance(model.get_model_object("diffusion_model"), SingleStreamDiT):
            raise TypeError("Image-only character LoRA requires the native Krea2 model.")

        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        adapters = comfy.lora.load_lora(state, key_map, log_missing=False)
        expected = {key.removesuffix(".lora_A.weight") + ".weight"
                    for key in state if key.endswith(".lora_A.weight")}
        if set(adapters) != expected:
            raise ValueError("The input Krea2 model does not expose all retained character LoRA targets.")

        bank = FactorBank(adapters, model.model_dtype())
        factors = comfy.model_patcher.ModelPatcher(bank, model.load_device, model.offload_device)
        patched = model.clone()
        owner = SCOPE + "." + uuid.uuid4().hex
        patched.set_additional_models(owner, [factors])
        patched.set_attachments(owner, ImageInjection(owner, tuple(adapters), strength_model))
        patched.set_injections(owner, [PatcherInjection(
            inject=lambda patcher: inject(patcher, owner),
            eject=lambda patcher: eject(patcher, owner),
        )])
        options = patched.model_options["transformer_options"]
        options[ACTIVE] = (*options.get(ACTIVE, ()), owner)
        patched.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, SCOPE)
        patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, SCOPE, scope_forward)
        report.update(status="active", adapted_matrices=len(adapters),
                      factor_bytes=factors.model_size(), existing_input_patches="preserved")
        return patched, json.dumps(report, indent=2)


NODE_CLASS_MAPPINGS = {"Krea2TCharacterLoraImageOnly": Krea2TCharacterLoraImageOnly}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2TCharacterLoraImageOnly": "Krea2T Character LoRA — Image Only"}
