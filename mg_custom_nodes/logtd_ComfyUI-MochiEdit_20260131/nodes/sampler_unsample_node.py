import logging

from comfy.samplers import KSAMPLER
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

from ..sampling.sampling_functions import get_rf_forward_sample_fn


class MochiUnsamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "gamma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "process"
    CATEGORY = "MochiEdit"

    def process(self, seed, gamma):
        sampler_fn = get_rf_forward_sample_fn(gamma, seed)
        sampler = KSAMPLER(sampler_fn)

        return (sampler,)



