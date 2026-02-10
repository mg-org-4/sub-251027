import torch
import comfy.samplers
import comfy.utils
from comfy.samplers import SchedulerHandler


# Define the scheduler function (exactly as in original sigma_schedule)
def capitan_zit_scheduler(model, steps):
    device = comfy.model_management.get_torch_device()
    return torch.linspace(1.0, 0.0, steps + 1).to(device)


# Add to ComfyUI's scheduler handlers (corrected from SCHEDULER_DICT to SCHEDULER_HANDLERS)
comfy.samplers.SCHEDULER_HANDLERS["capitanZiT"] = SchedulerHandler(capitan_zit_scheduler, use_ms=True)
comfy.samplers.SCHEDULER_NAMES.append("capitanZiT")


class CapitanZiTLinearSigma:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {
                    "default": 9,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "generate"
    CATEGORY = "sampling/custom_sampling/sigmas"
    DESCRIPTION = "Generates linear sigma schedule (1.0 → 0.0) for CapitanZiT (exactly as is for Z-Image-Turbo flow-matching)"

    def generate(self, steps):
        device = comfy.model_management.get_torch_device()
        sigmas = torch.linspace(1.0, 0.0, steps + 1).to(device)
        return (sigmas,)


# These must be at module level for ComfyUI to discover the node
NODE_CLASS_MAPPINGS = {
    "CapitanZiTLinearSigma": CapitanZiTLinearSigma
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CapitanZiTLinearSigma": "CapitanZiT Linear Sigma (for Z-Image Turbo)"
}