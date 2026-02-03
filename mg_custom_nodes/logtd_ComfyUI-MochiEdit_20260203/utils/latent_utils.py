import torch


def get_latent_dimensions(num_frames, width, height):
    spatial_downsample = 8
    temporal_downsample = 6
    in_channels = 12
    B = 1
    C = in_channels
    T = (num_frames - 1) // temporal_downsample + 1
    H = height // spatial_downsample
    W = width // spatial_downsample

    return (B, C, T, H, W)


def add_latent_noise(model, latent_shape, sigma_schedule, samples, generator):
    z = torch.randn(
        latent_shape,
        device=model.device,
        generator=generator,
        dtype=torch.float32,
    )
    if samples is not None:
        z = z * sigma_schedule[0] + (1 -sigma_schedule[0]) * samples.to(model.device)
    return z
    