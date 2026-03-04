import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

import torch

from ..utils.callback_utils import get_callback_fn
from ..utils.latent_utils import add_latent_noise
from ..utils.sampling_utils import get_model_fn, get_sample_args


def run_sampler(model, latents, positive, negative, sigmas, cfg, sampler_fn, add_noise=False, seed=0):
    # seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    generator = torch.Generator(device=model.device)
    generator.manual_seed(seed)
    
    # prepare latents
    latent_shape = latents.shape
    
    if add_noise:
        z = add_latent_noise(model, latent_shape, sigmas, latents, generator)
    else:
        z = latents.clone()

    # prepare model and args
    positive, negative = get_sample_args(model, positive, negative)
    model_fn = get_model_fn(model)
    
    # sampling
    callback_fn = get_callback_fn(model, len(sigmas)-1)
    extra_args = {
        "positive": positive,
        "negative": negative,
        "cfg": cfg
    }
    z = sampler_fn(model_fn, z, sigmas, callback=callback_fn, extra_args=extra_args)
    
    # cleanup
    model.dit.to(model.offload_device)

    return z
