import torch

import comfy.model_management as mm


def get_model_fn(model):
    #  sample, sample_null, cfg_scale
    def model_fn(z, sigma, positive, negative, cfg):
        model.dit.to(model.device)
        if hasattr(model.dit, "cublas_half_matmul") and model.dit.cublas_half_matmul:
            autocast_dtype = torch.float16
        else:
            autocast_dtype = torch.bfloat16
        
        with torch.autocast(mm.get_autocast_device(model.device), dtype=autocast_dtype):
            if cfg > 1.0:
                out_cond = model.dit(z, sigma, **positive)
                out_uncond = model.dit(z, sigma, **negative)
            else:
                out_cond = model.dit(z, sigma, **positive)
                return out_cond

        return out_uncond + cfg * (out_cond - out_uncond)
    
    return model_fn


def get_sample_args(model, cond_embeds, uncond_embeds):
    cond_args = {
        "y_mask": [cond_embeds["attention_mask"].to(model.device)],
        "y_feat": [cond_embeds["embeds"].to(model.device)]
    }

    uncond_args = {
        "y_mask": [uncond_embeds["attention_mask"].to(model.device)],
        "y_feat": [uncond_embeds["embeds"].to(model.device)]
    }
    return cond_args, uncond_args


def prepare_conds(positive, negative):
    #For compatibility with Comfy CLIPTextEncode
    if not isinstance(positive, dict):
        positive = {
            "embeds": positive[0][0],
            "attention_mask": positive[0][1]["attention_mask"].bool(),
            }
    if not isinstance(negative, dict):
        negative = {
            "embeds": negative[0][0],
            "attention_mask": negative[0][1]["attention_mask"].bool(),
            }
    return positive, negative


def generate_eta_values(steps, start_time, end_time, eta, eta_trend):
    end_time = min(end_time, steps)
    eta_values = [0] * steps
    
    if eta_trend == 'constant':
        for i in range(start_time, end_time):
            eta_values[i] = eta
    elif eta_trend == 'linear_increase':
        for i in range(start_time, end_time):
            progress = (i - start_time) / (end_time - start_time - 1)
            eta_values[i] = eta * progress
    elif eta_trend == 'linear_decrease':
        for i in range(start_time, end_time):
            progress = 1 - (i - start_time) / (end_time - start_time - 1)
            eta_values[i] = eta * progress
    
    return eta_values
