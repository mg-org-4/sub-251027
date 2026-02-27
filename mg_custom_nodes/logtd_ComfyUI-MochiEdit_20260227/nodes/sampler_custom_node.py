import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

import comfy.model_management as mm

from ..sampling.sampler import run_sampler
from ..utils.sampling_utils import prepare_conds


class MochiWrapperSamplerCustomNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MOCHIMODEL",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sigmas": ("SIGMAS", {"tooltip": "Override sigma schedule and steps"}),
                "latents": ("LATENT", ),
                "sampler": ("SAMPLER", ),
                "add_noise": ("BOOLEAN", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "MochiEdit/Wrapper"

    def process(self, model, positive, negative, cfg, seed, sigmas, latents, sampler, add_noise):
        mm.soft_empty_cache()

        sigmas = sigmas.tolist()
        latents = latents['samples']
        positive, negative = prepare_conds(positive, negative)
            
        latents = run_sampler(model, latents, positive, negative, sigmas, cfg, sampler.sampler_function, add_noise, seed)
    
        mm.soft_empty_cache()

        return ({"samples": latents},)



