import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

import comfy.model_management as mm

from ..sampling.sampler import run_sampler
from ..sampling.sampling_functions import get_rf_forward_sample_fn
from ..utils.sampling_utils import prepare_conds


class MochiWrapperUnsamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MOCHIMODEL",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "gamma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 30.0, "step": 0.01}),
                "sigmas": ("SIGMAS", {"tooltip": "Override sigma schedule and steps"}),
                "latents": ("LATENT", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "MochiEdit"

    def process(self, model, positive, negative, seed, gamma, sigmas, latents):
        mm.soft_empty_cache()

        sigmas = sigmas.tolist()
        if sigmas[0] != 0.0:
            sigmas = [0.0, *sigmas]
        latents = latents['samples']
        positive, negative = prepare_conds(positive, negative)
            
        sampler_fn = get_rf_forward_sample_fn(gamma, seed)
        latents = run_sampler(model, latents, positive, negative, sigmas, 1.0, sampler_fn, False, 0)
    
        mm.soft_empty_cache()

        return ({"samples": latents},)



