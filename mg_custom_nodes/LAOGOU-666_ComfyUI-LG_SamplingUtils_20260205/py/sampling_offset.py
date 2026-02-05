"""
ZImage Model Sampling 节点

用于调整 ZImage/Lumina2 模型的采样参数。
关键区别：ZImage 使用 multiplier=1.0，而 SD3 使用 multiplier=1000。
"""

import torch
import logging
import comfy.model_sampling


class ModelSamplingZImage:
    """
    ZImage/Lumina2 采样参数调整节点
    
    与 ModelSamplingSD3 的关键区别：
    - ZImage 使用 multiplier=1.0（timestep 范围 0-1）
    - SD3 使用 multiplier=1000（timestep 范围 0-1000）
    
    shift 参数控制噪声调度的偏移（通过 time_snr_shift 函数）：
    - shift=1.0: 无偏移，线性调度
    - shift>1.0: 向高噪声偏移，早期步骤更激进
    - ZImage 默认 shift=3.0
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "shift": ("FLOAT", {
                    "default": 3.0, 
                    "min": 0.0, 
                    "max": 100.0, 
                    "step": 0.01,
                    "tooltip": "噪声调度偏移量。ZImage 默认 3.0。shift=1.0 为线性，>1.0 向高噪声偏移"
                }),
                "multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.001,
                    "max": 10000.0,
                    "step": 0.001,
                    "tooltip": "Timestep 乘数。ZImage/AuraFlow=1.0，SD3/Flux=1000"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "🎈LAOGOU/Sampling Utils"
    DESCRIPTION = "调整 ZImage/Lumina2 模型的采样参数。可设置 shift 和 multiplier。"

    def patch(self, model, shift, multiplier):
        m = model.clone()

        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift, multiplier=multiplier)
        m.add_object_patch("model_sampling", model_sampling)
        
        # 调试输出
        logging.info(f"[ModelSamplingZImage] Applied: shift={shift}, multiplier={multiplier}")
        logging.info(f"[ModelSamplingZImage] sigma_min={model_sampling.sigma_min:.6f}, sigma_max={model_sampling.sigma_max:.6f}")
        logging.info(f"[ModelSamplingZImage] sigmas[0:5]={model_sampling.sigmas[:5].tolist()}")
        
        # 验证 patch 是否成功
        patched_ms = m.get_model_object("model_sampling")
        logging.info(f"[ModelSamplingZImage] Patch verification: patched shift={patched_ms.shift}, multiplier={patched_ms.multiplier}")
        logging.info(f"[ModelSamplingZImage] Patched sigmas[0:5]={patched_ms.sigmas[:5].tolist()}")
        
        return (m,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "ModelSamplingZImage": ModelSamplingZImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelSamplingZImage": "Model Sampling ZImage",
}
