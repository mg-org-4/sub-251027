#Modified/simplified version of the node from: https://github.com/pamparamm/sd-perturbed-attention
#If you want the one with more options see the above repo.

#My modified one here is more basic but has less chances of breaking with ComfyUI updates.

import comfy.model_patcher
import comfy.samplers
import torch

def build_patch(patchedBlocks, weight=1.0, sigma_start=0.0, sigma_end=1.0):
    def prompt_injection_patch(n, context_attn1: torch.Tensor, value_attn1, extra_options):
        (block, block_index) = extra_options.get('block', (None,None))
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9
        
        batch_prompt = n.shape[0] // len(extra_options["cond_or_uncond"])

        if sigma <= sigma_start and sigma >= sigma_end:
            if (block and f'{block}:{block_index}' in patchedBlocks and patchedBlocks[f'{block}:{block_index}']):
                if context_attn1.dim() == 3:
                    c = context_attn1[0].unsqueeze(0)
                else:
                    c = context_attn1[0][0].unsqueeze(0)
                b = patchedBlocks[f'{block}:{block_index}'][0][0].repeat(c.shape[0], 1, 1).to(context_attn1.device)
                out = torch.stack((c, b)).to(dtype=context_attn1.dtype) * weight
                out = out.repeat(1, batch_prompt, 1, 1) * weight

                return n, out, out 

        return n, context_attn1, value_attn1
    return prompt_injection_patch


class PromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "all":  ("CONDITIONING",),
                "input_4":  ("CONDITIONING",),
                "input_5":  ("CONDITIONING",),
                "input_7":  ("CONDITIONING",),
                "input_8":  ("CONDITIONING",),
                "middle_0": ("CONDITIONING",),
                "output_0": ("CONDITIONING",),
                "output_1": ("CONDITIONING",),
                "output_2": ("CONDITIONING",),
                "output_3": ("CONDITIONING",),
                "output_4": ("CONDITIONING",),
                "output_5": ("CONDITIONING",),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, all=None, input_4=None, input_5=None, input_7=None, input_8=None, middle_0=None, output_0=None, output_1=None, output_2=None, output_3=None, output_4=None, output_5=None, weight=1.0, start_at=0.0, end_at=1.0):
        if not any((all, input_4, input_5, input_7, input_8, middle_0, output_0, output_1, output_2, output_3, output_4, output_5)):
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        patchedBlocks = {}
        blocks = {'input': [4, 5, 7, 8], 'middle': [0], 'output': [0, 1, 2, 3, 4, 5]}

        for block in blocks:
            for index in blocks[block]:
                value = locals()[f"{block}_{index}"] if locals()[f"{block}_{index}"] is not None else all
                if value is not None:
                    patchedBlocks[f"{block}:{index}"] = value

        m.set_model_attn2_patch(build_patch(patchedBlocks, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)

class SimplePromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "block": (["input:4", "input:5", "input:7", "input:8", "middle:0", "output:0", "output:1", "output:2", "output:3", "output:4", "output:5"],),
                "conditioning": ("CONDITIONING",),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, block, conditioning=None, weight=1.0, start_at=0.0, end_at=1.0):
        if conditioning is None:
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        m.set_model_attn2_patch(build_patch({f"{block}": conditioning}, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)

class AdvancedPromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "locations": ("STRING", {"multiline": True, "default": "output:0,1.0\noutput:1,1.0"}),
                "conditioning": ("CONDITIONING",),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, locations: str, conditioning=None, start_at=0.0, end_at=1.0):
        if not conditioning:
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        for line in locations.splitlines():
            line = line.strip().strip('\n')
            weight = 1.0
            if ',' in line:
                line, weight = line.split(',')
                line = line.strip()
                weight = float(weight)
            if line:
                m.set_model_attn2_patch(build_patch({f"{line}": conditioning}, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)

class SD15PromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "all": ("CONDITIONING",),
                "input_3": ("CONDITIONING",),
                "input_4": ("CONDITIONING",),
                "input_5": ("CONDITIONING",),
                "input_6": ("CONDITIONING",),
                "input_7": ("CONDITIONING",),
                "input_8": ("CONDITIONING",),
                "input_9": ("CONDITIONING",),
                "input_10": ("CONDITIONING",),
                "input_11": ("CONDITIONING",),
                "middle_0": ("CONDITIONING",),
                "middle_1": ("CONDITIONING",),
                "middle_2": ("CONDITIONING",),
                "output_0": ("CONDITIONING",),
                "output_1": ("CONDITIONING",),
                "output_2": ("CONDITIONING",),
                "output_3": ("CONDITIONING",),
                "output_4": ("CONDITIONING",),
                "output_5": ("CONDITIONING",),
                "output_6": ("CONDITIONING",),
                "output_7": ("CONDITIONING",),
                "output_8": ("CONDITIONING",),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, all=None, input_3=None, input_4=None, input_5=None, input_6=None, input_7=None, input_8=None, input_9=None, input_10=None, input_11=None, middle_0=None, middle_1=None, middle_2=None, output_0=None, output_1=None, output_2=None, output_3=None, output_4=None, output_5=None, output_6=None, output_7=None, output_8=None, weight=1.0, start_at=0.0, end_at=1.0):
        if not any((all, input_3, input_4, input_5, input_6, input_7, input_8, input_9, input_10, input_11, middle_0, middle_1, middle_2, output_0, output_1, output_2, output_3, output_4, output_5, output_6, output_7, output_8)):
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        patchedBlocks = {}
        blocks = {
            'input': [3, 4, 5, 6, 7, 8, 9, 10, 11], 
            'middle': [0, 1, 2], 
            'output': [0, 1, 2, 3, 4, 5, 6, 7, 8]
        }

        for block in blocks:
            for index in blocks[block]:
                value = locals().get(f"{block}_{index}", None)
                if value is None:
                    value = all
                if value is not None:
                    patchedBlocks[f"{block}:{index}"] = value

        m.set_model_attn2_patch(build_patch(patchedBlocks, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)

class SimpleSD15PromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "block": (["input:3", "input:4", "input:5", "input:6", "input:7", "input:8", "input:9", "input:10", "input:11", "middle:0", "middle:1", "middle:2", "output:0", "output:1", "output:2", "output:3", "output:4", "output:5", "output:6", "output:7", "output:8"],),
                "conditioning": ("CONDITIONING",),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, block, conditioning=None, weight=1.0, start_at=0.0, end_at=1.0):
        if conditioning is None:
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        m.set_model_attn2_patch(build_patch({f"{block}": conditioning}, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)

class AdvancedSD15PromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "locations": ("STRING", {"multiline": True, "default": "input:7,1.0\nmiddle:0,1.0\noutput:0,1.0\noutput:1,1.0"}),
                "conditioning": ("CONDITIONING",),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, locations: str, conditioning=None, start_at=0.0, end_at=1.0):
        if not conditioning:
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        for line in locations.splitlines():
            line = line.strip().strip('\n')
            weight = 1.0
            if ',' in line:
                line, weight = line.split(',')
                line = line.strip()
                weight = float(weight)
            if line:
                m.set_model_attn2_patch(build_patch({f"{line}": conditioning}, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)


NODE_CLASS_MAPPINGS = {
    "PromptInjection": PromptInjection,
    "SimplePromptInjection": SimplePromptInjection,
    "AdvancedPromptInjection": AdvancedPromptInjection,
    "SD15PromptInjection": SD15PromptInjection,
    "SimpleSD15PromptInjection": SimpleSD15PromptInjection,
    "AdvancedSD15PromptInjection": AdvancedSD15PromptInjection
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptInjection": "Attn2 Prompt Injection",
    "SimplePromptInjection": "Attn2 Prompt Injection (simple)",
    "AdvancedPromptInjection": "Attn2 Prompt Injection (advanced)",
    "SD15PromptInjection": "Attn2 SD1.5 Prompt Injection",
    "SimpleSD15PromptInjection": "Attn2 SD1.5 Prompt Injection (simple)",
    "AdvancedSD15PromptInjection": "Attn2 SD1.5 Prompt Injection (advanced)"
}
