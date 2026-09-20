import numpy as np
import torch
from PIL import Image, ImageOps

from comfy_api.latest import InputImpl
from comfy_extras.nodes_audio import load as load_audio

from .prompt_references import reference_path


class FL_Prompt_Reference_Library:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt_schedule": ("FL_PROMPT_SCHEDULE",)}}

    RETURN_TYPES = ("FL_PROMPT_REFERENCES",)
    RETURN_NAMES = ("reference_library",)
    FUNCTION = "load"
    CATEGORY = "🏵️Fill Nodes/Audio"

    def load(self, prompt_schedule):
        assets = prompt_schedule.get("reference_assets", {})
        selected = dict.fromkeys(asset_id for section in prompt_schedule["sections"]
                                 for asset_id in section.get("references", {}).get("asset_ids", []))
        resolved = {}
        for asset_id in selected:
            asset = assets[asset_id]
            path = reference_path(asset)
            kind = asset["kind"]
            if kind == "image":
                with Image.open(path) as image:
                    if image.width * image.height > 64_000_000:
                        raise ValueError("Reference image exceeds the 64-megapixel limit.")
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    value = torch.from_numpy(np.array(rgb).astype(np.float32) / 255.0).unsqueeze(0)
                resolved[asset_id] = {"kind": kind, "value": value}
            elif kind == "audio":
                waveform, rate = load_audio(str(path))
                resolved[asset_id] = {"kind": kind, "value": {"waveform": waveform.unsqueeze(0), "sample_rate": rate}}
            elif kind == "video":
                components = InputImpl.VideoFromFile(str(path)).get_components()
                resolved[asset_id] = {"kind": kind, "value": components.images, "audio": components.audio, "fps": float(components.frame_rate)}
            else:
                raise ValueError(f"Unsupported reference kind: {kind}")
        return ({"version": 1, "assets": resolved},)
