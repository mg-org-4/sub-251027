import logging

from comfy.samplers import KSAMPLER
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

from ..sampling.sampling_functions import get_rf_reverse_sample_fn


class MochiResamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "eta": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 30.0, "step": 0.01}),
                "start_step": ("INT", {"default": 0, "min": 0}),
                "end_step": ("INT", {"default": 10, "min": 0}),
                "eta_trend": (['constant', 'linear_decrease', 'linear_increase'],),
                "latents": ("LATENT", ),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "process"
    CATEGORY = "MochiEdit"

    def process(self, eta, start_step, end_step, eta_trend, latents):
        latent_image = latents['samples']

        sampler_fn = get_rf_reverse_sample_fn(latent_image, eta, start_step, end_step, eta_trend)
        sampler = KSAMPLER(sampler_fn)

        return (sampler,)



