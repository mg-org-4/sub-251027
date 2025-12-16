import torch
from tqdm import tqdm, trange

from ..utils.sampling_utils import generate_eta_values


@torch.no_grad()
def mochi_sample(model, z, sigmas, callback=None):
    total_steps = len(sigmas)-1
    latent_shape = z.shape
    for i in tqdm(range(0, total_steps), desc="Processing Samples", total=total_steps):
        pred = model(z=z, sigma=torch.full([latent_shape[0]], sigmas[i], device=z.device))
        z = z + pred * (sigmas[i] - sigmas[i + 1])
        
        if callback is not None:
            callback(i, z)
    
    return z


def get_rf_forward_sample_fn(gamma, seed, correction=True):
    # Controlled Forward ODE (Algorithm 1)
    generator = torch.Generator()
    generator.manual_seed(seed)

    @torch.no_grad()
    def sample_forward(model, y0, sigmas, extra_args={}, callback=None, disable=None):
        Y = y0.clone()
        y1 = torch.randn(Y.shape, generator=generator).to(y0.device)
        N = len(sigmas)-1
        s_in = y0.new_ones([y0.shape[0]])
        for i in trange(N, disable=disable):
            # t_i = i/N 
            t_i = sigmas[i]

            # 6. Unconditional Vector field uti(Yti) = u(Yti, ti, Φ(“”); φ)
            unconditional_vector_field = -model(Y, sigmas[i]*s_in, **extra_args)
            
            if correction:
                # 7.Conditional Vector field  uti(Yti|y1) = (y1−Yti)/1−ti
                conditional_vector_field = (y1-Y)/(1-t_i)
                
                # 8. Controlled Vector field ti(Yti) = uti(Yti) + γ (uti(Yti|y1) − uti(Yti))
                controlled_vector_field = unconditional_vector_field + gamma * (conditional_vector_field - unconditional_vector_field)
            else:
                controlled_vector_field = unconditional_vector_field
            
            # 9. Next state Yti+1 = Yti + ˆuti(Yti) (σ(ti+1) − σ(ti))
            Y = Y + controlled_vector_field * (sigmas[i+1] - sigmas[i])

            if callback is not None:
                callback({'x': Y, 'denoised': Y, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i]})

        return Y

    return sample_forward


def get_rf_reverse_sample_fn(latent_image, eta, start_time, end_time, eta_trend):
    # Controlled Reverse ODE (Algorithm 2)
    @torch.no_grad()
    def sample_reverse(model, y1, sigmas, extra_args={}, callback=None, disable=None):
        latent_shape = y1.shape
        X = y1.clone()
        N = len(sigmas)-1
        y0 = latent_image.clone().to(y1.device)
        eta_values = generate_eta_values(N, start_time, end_time, eta, eta_trend)
        s_in = y0.new_ones([y0.shape[0]])
        for i in trange(N, disable=disable):
            # t_i = i/N 
            t_i = 1 - sigmas[i]

            # 5. Unconditional Vector field uti(Xti) = -u(Xti, 1-ti, Φ(“prompt”); φ)
            # torch.full([latent_shape[0]], sigmas[i], device=X.device)
            unconditional_vector_field = model(X, sigmas[i]*s_in, **extra_args)
            
            # 6.Conditional Vector field  uti(Xti|y0) = (y0−Xti)/(1−ti)
            conditional_vector_field = (y0-X)/(1-t_i)
            
            # 7. Controlled Vector field ti(Yti) = uti(Yti) + γ (uti(Yti|y1) − uti(Yti))
            controlled_vector_field = unconditional_vector_field + eta_values[i] * (conditional_vector_field - unconditional_vector_field)
            
            # 8. Next state Yti+1 = Yti + ˆuti(Yti) (σ(ti+1) − σ(ti))
            X = X + controlled_vector_field * (sigmas[i] - sigmas[i+1])
            # X = X + -unconditional_vector_field * (sigmas[i] - sigmas[i+1])

            if callback is not None:
                callback({'x': X, 'denoised': X, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i]})

        return X
    
    return sample_reverse
