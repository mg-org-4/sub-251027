import torch
import numpy as np
from comfy.model_patcher import ModelPatcher
from . import shared, rng_philox

class TorchHijack:
    """This is here to replace torch.randn_like of k-diffusion.

    k-diffusion has random_sampler argument for most samplers, but not for all, so
    this is needed to properly replace every use of torch.randn_like.

    We need to replace to make images generated in batches to be same as images generated individually."""

    def __init__(self, generator, randn_source, init=True):
        self.generator = generator
        self.randn_source = randn_source
        self.init = init

    def __getattr__(self, item):
        if item == 'randn_like':
            return self.randn_like

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def randn_like(self, x):
        return randn_without_seed(x, generator=self.generator, randn_source=self.randn_source)

def randn_without_seed(x, generator=None, randn_source="cpu"):
    """Generate a tensor with random numbers from a normal distribution using the previously initialized genrator.

    Use either randn() or manual_seed() to initialize the generator."""
    if randn_source == "nv":
        return torch.asarray(generator.randn(x.size()), device=x.device)
    else:
        return torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=generator.device, generator=generator).to(device=x.device)

def prepare_noise(latent_image, seed, noise_inds=None, device='cpu'):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    opts = None
    opts_found = False
    model = _find_outer_instance('model', ModelPatcher)
    if (model is not None and (opts:=model.model_options.get(shared.Options.KEY)) is None) or opts is None:
        import comfy.samplers
        guider = _find_outer_instance('guider', comfy.samplers.CFGGuider)
        model = getattr(guider, 'model_patcher', None)
    if (model is not None and (opts:=model.model_options.get(shared.Options.KEY)) is None) or opts is None:
        pass
    opts_found = opts is not None
    if not opts_found:
        opts = shared.opts_default
        device = torch.device("cpu")

    if opts.randn_source == 'gpu':
        import comfy.model_management
        device = comfy.model_management.get_torch_device()

    device_orig = device
    device = torch.device("cpu") if opts.randn_source == "cpu" else device_orig

    def get_generator(seed):
        nonlocal device, opts
        if opts.randn_source == 'nv':
            generator = rng_philox.Generator(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        return generator

    def get_generator_obj(seed):
        nonlocal opts
        generator = torch.manual_seed(seed)
        generator = generator_eta = get_generator(seed)
        if opts.eta_noise_seed_delta > 0:
            seed = min(int(seed + opts.eta_noise_seed_delta), int(0xffffffffffffffff))
            generator_eta = get_generator(seed)
        return (generator, generator_eta)

    generator, generator_eta = get_generator_obj(seed)
    randn_source = opts.randn_source

    # ========== hijack randn_like ===============
    import comfy.k_diffusion.sampling
    # if not hasattr(comfy.k_diffusion.sampling, 'torch_orig'):
    #     comfy.k_diffusion.sampling.torch_orig = comfy.k_diffusion.sampling.torch
    # comfy.k_diffusion.sampling.torch = TorchHijack(generator_eta, opts.randn_source)

    if not hasattr(comfy.k_diffusion.sampling, 'default_noise_sampler_orig'):
        comfy.k_diffusion.sampling.default_noise_sampler_orig = comfy.k_diffusion.sampling.default_noise_sampler
    if opts_found:
        th = TorchHijack(generator_eta, randn_source)
        def default_noise_sampler(x, seed=None, *args, **kwargs):
            nonlocal th
            return lambda sigma, sigma_next: th.randn_like(x)
        default_noise_sampler.init = True
        comfy.k_diffusion.sampling.default_noise_sampler = default_noise_sampler
    else:
        comfy.k_diffusion.sampling.default_noise_sampler = comfy.k_diffusion.sampling.default_noise_sampler_orig
    # =============================================

    if not opts_found:
        # No smZ Settings node in this graph, so smZNodes was not opted into.
        # Defer to the original instead of running a parallel implementation of
        # core's noise generation. This matches the other register_hooks()
        # patches, which already call orig_fn when Options.KEY is absent
        # (smZNodes.py get_area_and_mult / max_denoise / sampling_function).
        #
        # Seeds and existing outputs are unaffected: with the default
        # randn_source='cpu' the code below already reproduces
        # comfy.sample.prepare_noise bit-for-bit.
        #
        # It also keeps smZNodes transparent to latent types core adds later.
        # NestedTensor is the current example (LTX-AV, MiniMax-H3, TripoSplat):
        # its size() reports only tensors[0], so the code below would return
        # noise for the first stream alone and the sampler would fail packing
        # it against the full latent.
        from ..smZNodes import store
        return store.prepare_noise(latent_image, seed, noise_inds)

    # Handle nested tensors (e.g. video models like Wan) by delegating per inner tensor,
    # mirroring comfy.sample.prepare_noise behavior.
    if getattr(latent_image, 'is_nested', False):
        import comfy.nested_tensor
        tensors = latent_image.unbind()
        noises = []
        for t in tensors:
            noises.append(_generate_noise_tensor(t, generator, opts, device, device_orig, noise_inds))
        return comfy.nested_tensor.NestedTensor(noises)

    return _generate_noise_tensor(latent_image, generator, opts, device, device_orig, noise_inds)


def _generate_noise_tensor(latent_image, generator, opts, device, device_orig, noise_inds):
    if noise_inds is None:
        shape = latent_image.size()
        if opts.randn_source == 'nv':
            noise = torch.asarray(generator.randn(shape), dtype=latent_image.dtype, device=device)
        else:
            noise = torch.randn(shape, dtype=latent_image.dtype, layout=latent_image.layout, device=device, generator=generator)
        noise = noise.to(device=device_orig)
        return noise

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        shape = [1] + list(latent_image.size())[1:]
        if opts.randn_source == 'nv':
            noise = torch.asarray(generator.randn(shape), dtype=latent_image.dtype, device=device)
        else:
            noise = torch.randn(shape, dtype=latent_image.dtype, layout=latent_image.layout, device=device, generator=generator)
        noise = noise.to(device=device_orig)
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises

def _find_outer_instance(target:str, target_type=None, callback=None, max_len=10):
    import inspect
    frame = inspect.currentframe()
    i = 0
    while frame and i < max_len:
        if target in frame.f_locals:
            if callback is not None:
                return callback(frame)
            else:
                found = frame.f_locals[target]
                if isinstance(found, target_type):
                    return found
        frame = frame.f_back
        i += 1
    return None

