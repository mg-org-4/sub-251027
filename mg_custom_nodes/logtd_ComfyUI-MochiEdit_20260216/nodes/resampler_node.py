import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

import comfy.model_management as mm

from ..sampling.sampler import run_sampler
from ..sampling.sampling_functions import get_rf_reverse_sample_fn
from ..utils.sampling_utils import prepare_conds


class MochiWrapperResamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MOCHIMODEL",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0, "step": 0.01}),
                "eta": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 30.0, "step": 0.01}),
                "start_step": ("INT", {"default": 0, "min": 0}),
                "end_step": ("INT", {"default": 10, "min": 0}),
                "eta_trend": (['constant', 'linear_decrease', 'linear_increase'],),
                "sigmas": ("SIGMAS", {"tooltip": "Override sigma schedule and steps"}),
                "latents": ("LATENT", ),
                "original_latents": ("LATENT", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "MochiEdit/Wrapper"

    def process(self, model, positive, negative, cfg, eta, start_step, end_step, eta_trend, sigmas, latents, original_latents):
        mm.soft_empty_cache()

        sigmas = sigmas.tolist()
        if sigmas[-1] != 0.0:
            sigmas = [*sigmas, 0.0]
        latents = latents['samples']
        original_latents = original_latents['samples']
        positive, negative = prepare_conds(positive, negative)
            
        sampler_fn = get_rf_reverse_sample_fn(original_latents, eta, start_step, end_step, eta_trend)
        latents = run_sampler(model, latents, positive, negative, sigmas, cfg, sampler_fn, False, 0)
    
        mm.soft_empty_cache()

        return ({"samples": latents},)



