"""
File    : zsampler_turbo_core.py
Purpose : The core method for the "Z-Sampler Turbo" process.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Mar 16, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
         ComfyUI nodes designed specifically for the "Z-Image" model.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
import torch.nn.functional as F
import numpy as np
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.sampler_helpers
from typing         import Any
from .progress_bar  import ProgressPreview


def zsampler_turbo_core(latent_input             : dict[str, Any],
                        model                    : Any,
                        positive                 : list[ tuple[torch.Tensor,dict] ],
                        *,
                        seed                     : int,
                        steps                    : int,
                        initial_noise_bias_level : float              = 0.0,
                        initial_noise_overdose   : float              = 0.0,
                        noise_est_sample_size    : str | int | None   = None,
                        noise_est_sample_bias    : float              = 0.0,
                        noise_est_sample_scale   : float              = 1.0,
                        sigma_preset_name        : str | None         = None,
                        sigma_offsets            : list[float] | None = None,
                        sigma_limits             : tuple[float,float] | list[float] | None = None,
                        sigma_step_range         : tuple[int,int] | list[int] | None       = None,
                        start_with_noise         : bool                                    = True,
                        end_with_denoise         : bool                                    = True,
                        positive_stg2            : list[ tuple[torch.Tensor,dict] ] | None = None,
                        positive_stg3            : list[ tuple[torch.Tensor,dict] ] | None = None,
                        shuffle_seed             : int | None                              = None,
                        inject_noise_scales      : tuple[float,float,float] | None         = None,
                        inject_noise_freqs       : tuple[int  ,int  ,int  ] | None         = None,
                        progress_preview         : ProgressPreview
                        ) -> dict[str, Any]:
    """
    Perform the three-stage denoising process on a latent input.

    Args:
        latent_input             : A dict with the initial latent input to be denoised.
        model                    : The comfyui model to be used for denoising.
        positive                 : Positive conditioning for the denoising process.
        seed                     : The seed used for random number generation.
        steps                    : The total number of denoising steps. This value will
                                    be used internally to determine the sigmas values.
        initial_noise_bias_level : The level of adjustament from the calculated noise bias to
                                    apply before the first denoising step.
                                    0.0 = no noise bias adjustment;
                                    1.0 = using the 100% calculated noise bias
        initial_noise_overdose    : The level of over-amplitude in the initial noise scale.
                                    0.0 = default noise scale;
                                    positive values will increase the noise scale, e.g: 0.1 = 10% increment;
                                    negative values will reduce the noise scale, e.g: -0.1 = 10% decrement.
        noise_est_sample_size    : Size in pixels of the sample for initial noise estimation.
                                    A string can be used to specify the size in pixels, e.g: "512px".
                                    If `None`, the size of the latent input will be used.
        noise_est_sample_bias    : The bias of the pure noise used in the initial noise estimation,
        noise_est_sample_scale   : The scale of the pure noise used in the initial noise estimation.,
        sigma_offsets            : Optional list of offsets to be added to the calculated sigma values.
        sigma_limits             : Optional tuple with minimum and maximum limits for sigma values.
        sigma_step_range         : Optional tuple with the range of steps that will actually be performed.
        start_with_noise         : If `True` (default), begins the denoising process by injecting noise.
                                    Set to `False` to preserve noise from a previous process (chaining samplers).
        end_with_denoise         : If `True` (default), ends the denoising process by zeroing out residual noise.
                                    Set to `False` to preserve noise for a next process (chaining samplers).
        positive_stg2            : Optional positive conditioning to be used in the second stage.
                                    If `None` (default), the main positive conditioning will be used.
        positive_stg3            : Optional positive conditioning to be used in the third stage.
                                    If `None` (default), the main positive conditioning will be used.
        shuffle_seed             : Optional seed for shuffling the latent image between the stage1 and stage2.
                                    This results in a significant increase in the model's creativity,
                                    generating much more diverse images.
                                    If zero or None, no shuffling will be performed.
        inject_noise_scales      : Optional scales of the extra noise injected into the latent image in each stage.
                                    If `None` (default), no extra noise will be injected.
        inject_noise_freqs       : Optional frequencies at which additional noise is injected into the latent image
                                    during each stage. These frequencies determine the granularity of noise injection.
                                    For example, a value of 1024 means noise is injected into every pixel, while a
                                    value of 512 means noise is injected every second pixel, with intermediate pixels
                                    being interpolated. Lower frequency values result in smoother noise transitions
                                    across the image.
        progress_preview         : A `ProgressPreview` object for displaying progress during the denoising process.

    Returns:
        A dict with the denoised latent output.
    """

    # force inject_nise_scales to be a tuple of 3 elements or `None``
    if isinstance(inject_noise_scales, (tuple,list)):
        DEFAULTS = (0.0, 0.0, 0.0)
        inject_noise_scales = (*inject_noise_scales, *DEFAULTS[len(inject_noise_scales):])[:3]
    else:
        inject_noise_scales = None

    # force inject_noise_freqs to be a tuple of 3 elements or `None`
    if isinstance(inject_noise_freqs, (tuple,list)):
        DEFAULTS = (32,64,512)
        inject_noise_freqs = (*inject_noise_freqs, *DEFAULTS[len(inject_noise_freqs):])[:3]
    else:
        inject_noise_freqs = None

    # only "euler" sampler has been tested with this technique
    sampler_name = "euler"
    sampler = comfy.samplers.sampler_object(sampler_name)

    # z-image turbo is a cfg-distilled model requiring CFG=1.0, which discard
    # negative conditioning, here we set it to `positive` for simplicity
    negative = positive


    # validate sigma_limits & sigma_step_range
    if sigma_limits is not None:
        if not isinstance(sigma_limits, (tuple,list)) or len(sigma_limits)<2:
            raise ValueError("sigma_limits must be a tuple or list with at least two float values")
    if sigma_step_range is not None:
        if not isinstance(sigma_step_range, (tuple,list)) or len(sigma_step_range)<2:
            raise ValueError("sigma_step_range must be a tuple or list with at least two integer values")

    # check if the process should start from pure noise, this is usually the
    # case during image generation; other processes like inpainting and sampler
    # chaining usually require starting from an input image.
    start_from_pure_noise = True
    if sigma_limits and max(sigma_limits[0],sigma_limits[1]) < 1.0:
        start_from_pure_noise = False
    if sigma_step_range and sigma_step_range[0] > 0:
        start_from_pure_noise = False


    # Sigmas are divided into 3 stages:
    #
    #   Stages 1 and 2 combined function similarly to a standard denoising process,
    #   but with two key conditions:
    #     - stage 1 always has 2 steps and fixed sigmas, regardless of the total number of steps used.
    #     - there's a jump-back in sigmas between stage 1 and stage 2, there's no continuity. I'm not
    #       sure why it works, but after hundreds of tests where the final image consistently had better
    #       quality, I had to accept that this was a rule and it needed to be this way.
    #
    #   Once stages 1 and 2 are complete (acting as a standard denoising process), stage 3 begins.
    #   This is essentially a refining stage, where we go back with the sigmas, add the corresponding
    #   noise, and then do denoising from there.
    #
    #   The transition from Stage 2 to Stage 3 can also be thought of as applying an "Euler-ancestral"
    #   sampling method instead of the standard "Euler", but only for that single step between stages.
    #
    sigma_preset = SIGMA_PRESETS_BY_NAME.get(sigma_preset_name) if sigma_preset_name else None
    if not sigma_preset:
        sigma_preset = SIGMA_PRESETS_BY_NAME["alpha"]
    if len(sigma_preset) != 7:
        raise ValueError(f"Sigma presets must have 7 elements but the \"{sigma_preset_name}\" preset has {len(sigma_preset)} elements")
    index   = min(max( 3, steps), 9 ) - 3
    sigmas1 = list(sigma_preset[index][0])
    sigmas2 = list(sigma_preset[index][1])
    sigmas3 = list(sigma_preset[index][2])

    # when the number of steps is greater than 9, the same 9-step sigma
    # sequence is used, but the Stage 2 and Stage 3 are refined to match
    # the required number of steps
    if steps>9:
        additional_steps = (steps-9)
        n1 = int( 0.4 + 0.6 * additional_steps )
        n2 = additional_steps - n1
        sigmas2 = refine_sigma_sequence(sigmas2, n1)
        sigmas3 = refine_sigma_sequence(sigmas3, n2)


    # sigma0 is used only for estimating the initial noise bias (optional first step),
    # the denoising for that estimation step goes from sigma0 to sigmas1[0]
    sigma0 = 1.000

    # add the values of sigmas_offset to each sigma in the 3 lists
    if sigma_offsets:
        offset_iter = iter(sigma_offsets)
        sigma0 = min(1.000, sigma0 + next(offset_iter, 0.0))
        if isinstance(sigmas1, list):
            for i in range( len(sigmas1) ):
                sigmas1[i] += next(offset_iter, 0.0)
        if isinstance(sigmas2, list):
            for i in range(len(sigmas2) - 1):
                sigmas2[i] += next(offset_iter, 0.0)
        if isinstance(sigmas3, list):
            for i in range(len(sigmas3) - 1):
                sigmas3[i] += next(offset_iter, 0.0)


    # `sample_size` is noise_est_sample_size converted to integer (pixels)
    # or None if "image_size" option was selected
    if isinstance(noise_est_sample_size, str) and noise_est_sample_size.endswith("px"):
        sample_size = int(noise_est_sample_size[:-2])
    elif isinstance(noise_est_sample_size, (int,float)):
        sample_size = int(noise_est_sample_size)
    else:
        sample_size = None


    # estimate the initial noise features (bias & scale):
    # these parameters can be thought of as modifying the brightness and contrast
    # of the final image, though the actual impact is more nuanced; since the sigma
    # sequence starts with values below 1.0, using this modified initial noise
    # introduces low-frequency components that may be missing in the early stages
    # of the denoising process.
    # the initial noise bias represents a shift in the mean noise value, while
    # the initial noise scale controls the amplitude of the noise.
    # (they are only calculated if the generation starts from pure noise and
    #  if the user has specified a non-zero level for either parameter)
    initial_noise_bias  = 0.0
    initial_noise_scale = 1.0 + initial_noise_overdose
    if start_from_pure_noise and (initial_noise_bias_level != 0):
        bias, scale = estimate_initial_noise_features(
                                latent_input, model, seed, positive, positive,
                                sampler      = sampler,
                                sigmas       = [sigma0, sigmas1[0]],
                                sample_size  = sample_size,
                                sample_bias  = noise_est_sample_bias,
                                sample_scale = noise_est_sample_scale,
                                progress_preview = ProgressPreview( 100, parent=(progress_preview,0,100//steps) ),
                                )
        initial_noise_bias  = (bias / scale * initial_noise_bias_level)
        #-- OLD MTHOD ---------
        # initial_noise_bias  = (bias  / 100 * initial_noise_bias_level)
        # initial_noise_scale = (scale / 100 * initial_noise_scale_level) + (1-initial_noise_scale_level)
        # if initial_noise_overdose:
        #    initial_noise_scale *= 1+initial_noise_overdose if initial_noise_overdose>=0 else 1/(1-initial_noise_overdose)


    # execute the 3-stage denoising process
    latent_output = execute_3_stage_denoising(latent_input,
                                              model, seed, 1.0, positive, negative,
                                              positive_stg2        = positive_stg2,
                                              positive_stg3        = positive_stg3,
                                              sampler              = sampler,
                                              sigmas1              = sigmas1,
                                              sigmas2              = sigmas2,
                                              sigmas3              = sigmas3,
                                              sigma_limits         = sigma_limits,
                                              sigma_step_range     = sigma_step_range,
                                              initial_noise_bias   = initial_noise_bias,
                                              initial_noise_scale  = initial_noise_scale,
                                              start_with_noise     = start_with_noise,
                                              end_with_denoise     = end_with_denoise,
                                              shuffle_seed         = shuffle_seed,
                                              inject_noise_scales  = inject_noise_scales,
                                              inject_noise_freqs   = inject_noise_freqs,
                                              progress_preview = ProgressPreview( 100, parent=(progress_preview,100//steps,100) ),
                                              )
    return latent_output


    #----------------------------------------------------------------------------
    # Alternative 8/7-step sigma sequences discarded in the early stages of development
    # if steps==8:
    #     sigma0  = 1.000
    #     sigmas1 = [0.991, 0.980, 0.920]
    #     sigmas2 = [0.935, 0.8916, 0.7895, 0.000],
    #     sigmas3 = [0.6582, 0.3019, 0.0000]
    # if steps==7:
    #     sigma0  = 1.000
    #     sigmas1 = [0.991, 0.980, 0.920]
    #     sigmas2 = [0.935, 0.770, 0.000]
    #     sigmas3 = [0.6582, 0.3019, 0.0000]
    #----------------------------------------------------------------------------


def step_count(sigmas: torch.Tensor | None) -> int:
    return sigmas.shape[-1]-1 if sigmas is not None else 0

def execute_3_stage_denoising(latent_image,
                              model    : Any,
                              seed     : int,
                              cfg      : float,
                              positive : list,
                              negative : list,
                              *,
                              sampler             : comfy.samplers.KSAMPLER,
                              sigmas1             : torch.Tensor | list | None,
                              sigmas2             : torch.Tensor | list | None,
                              sigmas3             : torch.Tensor | list | None,
                              sigma_limits        : tuple[float,float] | list[float] | None = None,
                              sigma_step_range    : tuple[int,int]     | list[int]   | None = None,
                              initial_noise_bias  : torch.Tensor | float | int | None       = None,
                              initial_noise_scale : torch.Tensor | float | int | None       = 1.0,
                              start_with_noise    : bool                                    = True,
                              end_with_denoise    : bool                                    = True,
                              positive_stg2       : list | None                             = None,
                              positive_stg3       : list | None                             = None,
                              shuffle_seed        : int  | None                             = None,
                              inject_noise_scales : tuple[float,float,float] | None         = None,
                              inject_noise_freqs  : tuple[int  ,int  ,int  ] | None         = None,
                              progress_preview    : ProgressPreview,
                              ):
    """
    Executes a three-stage denoising process on the provided latent image.

    Args:
        latent_image       : Dictionary containing data about the initial latent image to be processed.
                              Includes keys like "samples" and optional "noise_mask", "batch_index".
        model              : ComfyUI object representing the model to use for denoising.
        seed               : The seed used to generate random noise.
        cfg (float)        : Classifier-free guidance scale that controls the influence of negative prompts.
                              A value of 1.0 means the negative prompt has no effect on generation.
        positive           : Positive prompts or conditioning applied to the model during denoising.
        negative           : Negative prompts or conditioning applied to the model during denoising.
        sampler            : ComfyUI object representing the sampler used for each denoising step.
        sigmas1            : Sigma values for the first stage of denoising. Can be a tensor, list, or None.
        sigmas2            : Sigma values for the second stage of denoising. Can be a tensor, list, or None.
        sigmas3            : Sigma values for the third stage of denoising. Can be a tensor, list, or None.
        sigma_limits       : The range of sigma values where the denoising process should be performed.
                             Steps with sigmas outside this range will be ignored.
                              Can be a tuple or list of two floats, or None if all steps should be processed.
        sigma_step_range   : The range of steps to be processed. Steps outside this range will be ignored.
                              Can be a tuple or list of two integers, or None if all steps should be processed.
        initial_noise_bias : An optional constant bias applied to the initial noise.
                              (where 0.0 = no bias; None = no bias)
                              Can be a tensor, scalar, or None.
                              If tensor, it must have shape [batch_size, channels, 1, 1].
        initial_noise_scale: An optional scale factor applied to the initial noise.
                              (where 0.0 = no-noise; 1.0 = standard deviation; None = standard deviation)
                              Can be a tensor, scalar, or None.
                              If tensor, it must have shape [batch_size, channels, 1, 1].
        start_with_noise   : If `True` (default), begins the denoising process by injecting noise.
                              Set to `False` to preserve noise from a previous process (chaining samplers).
        end_with_denoise   : If `True` (default), ends the denoising process by zeroing out residual noise.
                              Set to `False` to preserve noise for a next process (chaining samplers).
        shuffle_seed       : Optional seed for shuffling the latent image between the stage1 and stage2.
                              If zero or None, no shuffling will be performed.

    Returns:
        A dictionary with the updated latent image data after all three denoising stages.
    """
    # the positive conditioning for stage 2 and 3 are optional
    # if not provided, they will be the same as the main conditioning
    if positive_stg2 is None or (isinstance(positive_stg2,(list,tuple)) and len(positive_stg2) == 0):
        positive_stg2 = positive
    if positive_stg3 is None or (isinstance(positive_stg3,(list,tuple)) and len(positive_stg3) == 0):
        positive_stg3 = positive

    # if sigmas is a list then convert it to pytorch tensor
    if isinstance(sigmas1, list):
        sigmas1 = torch.tensor(sigmas1, device='cpu')
    if isinstance(sigmas2, list):
        sigmas2 = torch.tensor(sigmas2, device='cpu')
    if isinstance(sigmas3, list):
        sigmas3 = torch.tensor(sigmas3, device='cpu')

    # store the stage3 sigmas to later evaluate if it was truncated
    original_sigmas3 = sigmas3

    # truncate the sigmas to the specified range of steps. [start_step, end_step)
    if isinstance(sigma_step_range, (list,tuple)):
        i0 = 0
        i1 = i0 + step_count(sigmas1)
        i2 = i1 + step_count(sigmas2)
        sigmas1 = truncate_sigmas_by_step_range(sigmas1, sigma_step_range, first_sigma_step=i0)
        sigmas2 = truncate_sigmas_by_step_range(sigmas2, sigma_step_range, first_sigma_step=i1)
        sigmas3 = truncate_sigmas_by_step_range(sigmas3, sigma_step_range, first_sigma_step=i2)

    # truncate the sigmas to the specified range of values. [lower_value, upper_value]
    if isinstance(sigma_limits, (list,tuple)):
        sigmas1 = truncate_sigmas_by_value_range(sigmas1, sigma_limits)
        sigmas2 = truncate_sigmas_by_value_range(sigmas2, sigma_limits)
        sigmas3 = truncate_sigmas_by_value_range(sigmas3, sigma_limits)

    # always add noise at the beginning of stage3
    # unless its initial sigma has been truncated
    add_noise_in_stage3 = True
    if sigmas3 is not None and original_sigmas3 is not None:
        add_noise_in_stage3 = (sigmas3[0] == original_sigmas3[0])

    # calculate the progress level for each step
    prog0 = 0
    prog1 = prog0 + step_count(sigmas1)
    prog2 = prog1 + step_count(sigmas2)
    total = prog2 + step_count(sigmas3)


    #-- THREE-STAGE PROCESS ---------------------------
    if sigmas1 is not None:
        is_first_stage = True
        is_last_stage  = (sigmas2 is None and sigmas3 is None)
        add_noise     = (is_first_stage and start_with_noise)
        force_denoise = (is_last_stage  and end_with_denoise) or bool(shuffle_seed)

        latent_image = execute_sampler(latent_image,
                        model, seed, cfg, positive, negative,
                        sampler             = sampler,
                        sigmas              = sigmas1,
                        noise_bias          = initial_noise_bias  if add_noise else 0,
                        noise_scale         = initial_noise_scale if add_noise else 0,
                        force_final_denoise = force_denoise,
                        keep_masked_area    = True,
                        inject_noise_scale  = inject_noise_scales[0] if inject_noise_scales else 0,
                        inject_noise_freq   = inject_noise_freqs[0]  if inject_noise_freqs  else 0,
                        progress_preview    = ProgressPreview( prog1-prog0,
                                parent=(progress_preview, 100*prog0//total, 100*prog1//total)),
                        )

    if sigmas2 is not None:
        # normally this stage should NOT add noise (stage1 and stage2 must behave
        # as a single donoise process) but if no first stage was executed then
        # this stage is the first and therefore it must add noise
        is_first_stage = (sigmas1 is None)
        is_last_stage  = (sigmas3 is None)
        add_noise     = (is_first_stage and start_with_noise) or bool(shuffle_seed)
        force_denoise = (is_last_stage  and end_with_denoise)

        latent_image = execute_sampler(latent_image,
                        model, seed, cfg, positive_stg2, negative,
                        sampler             = sampler,
                        sigmas              = sigmas2,
                        noise_bias          = 0,
                        noise_scale         = 1.0 if add_noise else 0,
                        force_final_denoise = force_denoise,
                        keep_masked_area    = True,
                        shuffle_seed        = shuffle_seed,
                        inject_noise_scale  = inject_noise_scales[1] if inject_noise_scales else 0,
                        inject_noise_freq   = inject_noise_freqs[1]  if inject_noise_freqs  else 0,
                        progress_preview    = ProgressPreview( prog2-prog1,
                                parent=(progress_preview, 100*prog1//total, 100*prog2//total)),
                        )

    if sigmas3 is not None:
        is_first_stage = (sigmas1 is None and sigmas2 is None)
        is_last_stage  = True
        add_noise     = (is_first_stage and start_with_noise) or add_noise_in_stage3
        force_denoise = (is_last_stage  and end_with_denoise)

        latent_image = execute_sampler(latent_image,
                        model, 696969, cfg, positive_stg3, negative,
                        sampler             = sampler,
                        sigmas              = sigmas3,
                        noise_bias          = 0,
                        noise_scale         = 1.0 if add_noise else 0,
                        force_final_denoise = force_denoise,
                        keep_masked_area    = True,
                        inject_noise_scale  = inject_noise_scales[2] if inject_noise_scales else 0,
                        inject_noise_freq   = inject_noise_freqs[2]  if inject_noise_freqs  else 0,
                        progress_preview    = ProgressPreview( total-prog2,
                                parent=(progress_preview, 100*prog2//total, 100*total//total)),
                        )
    return latent_image


def execute_sampler(latent_image    : dict[str, Any],
                    model           : Any,
                    noise_seed      : int,
                    cfg             : float,
                    positive        : list,
                    negative        : list,
                    *,
                    sampler         : comfy.samplers.KSAMPLER,
                    sigmas          : list | torch.Tensor,
                    noise_bias      : torch.Tensor | float | int | None,
                    noise_scale     : torch.Tensor | float | int | None,
                    force_final_denoise : bool  = False,
                    keep_masked_area    : bool  = False,
                    fix_empty_latent    : bool  = True,
                    shuffle_seed        : int | None = 0,
                    inject_noise_scale  : float      = 0,
                    inject_noise_freq   : int        = 0,
                    progress_preview    : ProgressPreview | None = None,
                    ) -> dict[str, Any]:
    """
    Emulates comfyui's 'SamplerCustom' node but with extra functionality.

    Args:
        latent_image    : Dictionary containing data about the initial latent image to denoise.
                           (contains keys like "samples" and optional "noise_mask", "batch_index").
        model           : ComfyUI object representing the model to use for denoising.
        noise_seed      : The seed used to generate random noise.
        cfg             : Classifier-free guidance scale that controls the strength of negative prompts.
                           A value of 1.0 means the negative prompt has no effect on generation.
        positive        : Positive prompts or conditioning applied to the model during denoising.
        negative        : Negative prompts or conditioning applied to the model during denoising.
        sampler         : ComfyUI object representing the sampler used for each denoising step.
        sigmas          : Sigma values for each diffusion process step (can be list or torch.Tensor).
        noise_bias      : The constant bias applied to the initial noise.
                           (where 0.0 = no-bias; 1.0 = are you creazy??!; None = no-bias)
                           Can be a tensor, scalar, or None.
                           If tensor, it must have shape [batch_size, channels, 1, 1].
        noise_scale     : The scale factor applied to the initial noise.
                           (where 0.0 = no-noise; 1.0 = standard deviation; None = standard deviation)
                           Can be a tensor, scalar, or None.
                           If tensor, it must have shape [batch_size, channels, 1, 1].
        force_final_denoise: If `True`, forces the final denoising step to zero out residual noise,
                              use `False` (default) for chaining samplers to preserve noise for the next stage.
        keep_masked_area   : If True, the masked area of the latent image will be kept unchanged.
                              Comfyui tries to keep masked area unchanged when there's a mask active
                              but activating this flag we're sure that no change will happen at all.
        shuffle_seed       : Optional seed for shuffling the latent image.
                              If zero or None, no shuffling will be performed.
        progress_preview   : Optional callback for tracking progress. Defaults to None.

    Returns:
        A dictionary with the updated latent image data after denoising,
        including "samples" and any other relevant information.

    Notes:
        - If `sigmas` is provided as a list, it will be converted to a torch.Tensor.
        - The initial noise is scaled using `noise_scale`, and then the bias from `noise_bias` is applied.
    """

    # if sigmas is a list then convert it to tensor
    if isinstance(sigmas, list):
        sigmas = torch.tensor(sigmas, device='cpu')

    # if `noise_bias` is a torch.Tensor
    # then it should have shape [batch_size, channels, 1, 1]
    if isinstance(noise_bias, torch.Tensor):
        if noise_bias.ndim != 4 or noise_bias.shape[2:] != (1, 1):
            raise ValueError(f"Invalid `noise_bias` shape: expected [batch_size, channels, 1, 1], "
                                f"but got {list(noise_bias.shape)}")

    # if `noise_scale` is a torch.Tensor
    # then it should have shape [batch_size, channels, 1, 1]
    if isinstance(noise_scale, torch.Tensor):
        if noise_scale.ndim != 4 or noise_scale.shape[2:] != (1, 1):
            raise ValueError(f"Invalid `noise_scale` shape: expected [batch_size, channels, 1, 1], "
                                f"but got {list(noise_scale.shape)}")

    # if `noise_bias` or `noise_scale` is zero scalar then ignore it
    if isinstance(noise_bias, (float,int)) and noise_bias == 0:
        noise_bias = None
    if isinstance(noise_scale, (float,int)) and noise_scale == 1:
        noise_scale = None


    # extract all info that comes packaged in the `latent_image` dictionary
    latent     : dict                = latent_image.copy()
    samples    : torch.Tensor        = latent["samples"]
    noise_mask : torch.Tensor | None = latent.get("noise_mask")
    batch_index: int| None           = latent.get("batch_index")
    steps      : int                 = int( sigmas.shape[-1] )
    if fix_empty_latent:
        samples = comfy.sample.fix_empty_latent_channels(model, samples)

    # store original values in case the user is doing inpainting with a mask
    original_samples : torch.Tensor | None = samples
    original_mask    : torch.Tensor | None = noise_mask


    # shuffle the image if it was required
    if shuffle_seed:
        samples = shuffle_tensor(samples, shuffle_seed)

    # apply extra noise injection if it was required
    if inject_noise_scale and inject_noise_freq:
        samples = inject_low_freq_noise(samples,
                                        seed        = noise_seed,
                                        noise_scale = inject_noise_scale,
                                        noise_freq  = inject_noise_freq)

    # force a full denoising (with the last sigma to zero) if it was required
    if force_final_denoise:
        sigmas[-1] = 0


    # generate the noise needed by `comfy.sample.sample_custom(..)`;
    # if both `noise_scale` and `noise_bias` are 0, then no noise is generated
    if isinstance(noise_scale, (float,int)) and noise_scale == 0 and \
       isinstance(noise_bias , (float,int)) and noise_bias  == 0:
        comfy_noise = torch.zeros(samples.shape,
                                  dtype   = samples.dtype,
                                  layout  = samples.layout,
                                  device  = "cpu")
    else:
        comfy_noise = generate_noise(noise_seed, samples.shape,
                                     noise_bias     = noise_bias,
                                     noise_scale    = noise_scale,
                                     batch_subseeds = batch_index,
                                     dtype          = samples.dtype,
                                     layout         = samples.layout,
                                     device         = "cpu")

    # this wrapper increments by 1 the steps count reported by comfyui.
    # comfyui reports the end of the first denoising step as 0 (0% completion)
    # breaking internal ProgressPreview calculation, the wrapper solves that
    progress_wrapper = ProgressPreview( steps, parent=(progress_preview, 1, steps+1) )
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    # generates the denoised latent using a native function from comfyui
    samples = comfy.sample.sample_custom(model, comfy_noise, cfg, sampler, sigmas, positive, negative,
                                         samples, noise_mask=noise_mask, callback=progress_wrapper,
                                         disable_pbar=disable_pbar, seed=noise_seed)

    # when there's an inpainting mask, it seems like comfyui does not merge the
    # original image at the end of `sample_custom(..)`, so we manually merge it here
    if keep_masked_area and (original_mask is not None) and (original_samples is not None):
        original_mask = comfy.sampler_helpers.prepare_mask( original_mask, original_samples.shape, original_samples.device)
        if original_mask is not None:
            samples = samples * original_mask + ( 1.0 - original_mask ) * original_samples

    out = latent_image.copy()
    out["samples"] = samples
    return out


#============================= IMAGE SHUFFLING =============================#

def random_tensor_fragment(x        : torch.Tensor,
                           generator: torch.Generator,
                           size     : tuple[float, float] = (0.50, 0.75),
                           anchor   : str = 'left'
                           ) -> torch.Tensor:
    """
    Extracts a random rectangular fragment from a 4-dimensional tensor.

    This function generates a random fragment by selecting a random rectangle
    within the tensor and then resizing it to match the original tensor's dimensions.

    Args:
        x         : Input tensor of shape (B, C, H, W) representing an image or feature map.
        generator : Random number generator for reproducibility.
        size      : Minimum and maximum relative size of the fragment as a
                     ratio of the input tensor's width. Defaults to (0.50, 0.75)
        anchor    : Specifies the side to which the fragment is anchored.
                     Can be either 'left' or 'right'. Defaults to 'left'.
    Returns:
        A tensor of the same shape as the input tensor, containing a randomly selected fragment.
    """
    if x.dim() != 4:
        raise ValueError("Input tensor must be 4-D (B, C, H, W).")
    if anchor not in {"left", "right"}:
        raise ValueError("anchor must be either 'left' or 'right'.")

    B, C, H, W = x.shape

    # give the rectangle a random size within the provided limits.
    min_size, max_size = size
    ratio = torch.rand(1, generator=generator, device=x.device) * (max_size - min_size) + min_size
    frag_width  = int(W * ratio)
    frag_height = int(H * ratio)

    # place the rectangle in a random position inside the tensor
    top  = torch.randint(0, H - frag_height + 1, (1,), generator=generator, device="cpu").item()
    left = 0 if anchor == "left" else W - frag_width

    # fragment = (B, C, fh, fw)
    fragment = x[..., top:top + frag_height, left:left + frag_width]
    # return = (B, C, H, W)
    return F.interpolate(fragment, size=(H, W), mode='bilinear', align_corners=False)


def shuffle_tensor(x: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Shuffles the input latent image tensor.

    The function shuffles the tensor by dividing it into fragments, shuffling
    each fragment independently, and then combining them while preserving the
    overall standard deviation and mean of the original tensor.

    Args:
        x    : Input tensor of shape (B, C, H, W) representing the latent image.
        seed : Fixed seed for random number generation to ensure reproducibility.

    Returns:
        A new tensor with the same shape as `x`, where fragments of the
        original image are shuffled in a random manner.
    """
    if x.dim() != 4:
        raise ValueError("Input tensor must be (B, C, H, W).")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    x_scale = x.std (dim=(2,3), keepdim=True)
    x_bias  = x.mean(dim=(2,3), keepdim=True)

    generator = torch.Generator().manual_seed(seed)
    result =  random_tensor_fragment(x, generator, size=(0.50,0.75), anchor='left')
    result += random_tensor_fragment(x, generator, size=(0.50,0.75), anchor='right')

    result_scale = result.std (dim=(2,3), keepdim=True)
    result_bias  = result.mean(dim=(2,3), keepdim=True)

    # re-scale/shift to match original features
    scale_factor  = x_scale / result_scale.clamp(min=1e-6)
    combined_bias = x_bias - (result_bias * scale_factor)
    return result * scale_factor + combined_bias


#============================ NOISE PROCESSING =============================#

def generate_noise(seed           : int,
                   shape          : tuple[int, ...],
                   *,
                   noise_bias     : torch.Tensor | float | int | None = None,
                   noise_scale    : torch.Tensor | float | int | None = None,
                   batch_subseeds : list[int] | None                  = None,
                   dtype          : torch.dtype,
                   layout         : torch.layout,
                   device         : str | torch.device = "cpu"
                   ):
    """
    Generate batched noise with optional per-sample 'virtual' sub-seeds.
    """
    generator = torch.manual_seed(seed)
    return generate_noise_(generator, shape, dtype, layout, noise_bias, noise_scale, batch_subseeds, device)


def generate_noise_(generator      : torch.Generator,
                    shape          : tuple[int, ...],
                    dtype          : torch.dtype,
                    layout         : torch.layout,
                    noise_bias     : torch.Tensor | float | int | None = None,
                    noise_scale    : torch.Tensor | float | int | None = None,
                    batch_subseeds : list[int] | None                  = None,
                    device         : str | torch.device                = "cpu"
                    ):
    """
    Generate batched noise with optional per-sample 'virtual' sub-seeds.

    Args:
        generator      : A `torch.Generator` object for generating random numbers.
        shape          : Noise shape. The first dimension is the batch size.
        dtype          : The floating-point dtype of the generated tensor.
        layout         : torch.strided or torch.sparse_coo, etc.
        noise_bias     : Optional constant offset added to the noise.
                          if `None` or not provided, this is set to 0.0
        noise_scale    : Optional scale factor applied to the raw normal noise.
                          if `None` or not provided, this is set to 1.0
        batch_subseeds : Optional list of small integers (0, 1, 2, …) that act as virtual seeds
                          for every sample in the batch. Repetitions are allowed; repeated indices
                          yield identical noise for those samples. If `None` or empty, every sample
                          receives independent noise.
        device         : Target device for the generated tensor.
    Returns:
        A noise tensor of the requested shape, already biased and scaled.
    """
    if batch_subseeds:
        # batch subseeds were provided, indicating the virtual seed value for
        # generating noise for each batch element. These values should be small
        # integers like [0, 1, 2, 3] or [0, 1, 0, 1], or even [0, 0, 0, 0] if
        # identical noise is desired for all elements in the batch.
        unique_subseeds, inverse = np.unique(batch_subseeds, return_inverse=True)
        max_subseed              = unique_subseeds.max()
        subnoise_shape           = [1] + list(shape)[1:]
        subnoises : list[torch.Tensor] = []
        for subseed in range(max_subseed+1):
            subnoise = torch.randn(subnoise_shape, dtype=dtype, layout=layout, generator=generator, device=device)
            if subseed in unique_subseeds:
                subnoises.append(subnoise)
        noise = torch.cat( [subnoises[i] for i in inverse] )
    else:
        # if no batch subseeds are provided, generate a single noise tensor for
        # the entire batch with fully random values.
        noise = torch.randn(shape, dtype=dtype, layout=layout, generator=generator, device=device)

     # apply noise bias and scale if provided
    if noise_scale is not None: noise *= noise_scale
    if noise_bias  is not None: noise += noise_bias
    return noise


def estimate_initial_noise_features(latent_image,
                                    model    : Any,
                                    seed     : int,
                                    positive : list,
                                    negative : list,
                                    *,
                                    sampler      : comfy.samplers.KSAMPLER,
                                    sigmas       : list | torch.Tensor,
                                    sample_bias  : float = 0.0,
                                    sample_scale : float = 0.1,
                                    sample_size  : tuple[int, int] | int | None = None,
                                    progress_preview: ProgressPreview
                                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates bias and scale of the resultant latent image after denoising.

    The bias and scale are determined by denoising a pure noise sample with the
    provided sigma values, then calculating the mean and standard deviation
    across each channel of the resulting latent image.

    Args:
        latent_image : Dictionary containing information about the initial latent image,
                       only its width and height are used.
        model        : ComfyUI object representing the model to use for denoising.
        seed         : The seed used to generate random noise.
        positive     : Positive prompts or conditioning applied to the model during denoising.
        negative     : Negative prompts or conditioning applied to the model during denoising.
        sampler      : ComfyUI object representing the sampler used for each denoising step.
        sigmas       : Sigma values for each diffusion process step (can be list or torch.Tensor).
        sample_size  : The size in pixels of the sample. If `None`, the size of the latent image is used.
        sample_bias  : The bias of the pure noise sample before denoising.
        sample_scale : The scale of the pure noise sample before denoising.
        progress_preview: An object for reporting progress.

    Returns:
        A tuple containing two tensors:
        - The calculated noise bias, tensor of shape [batch_size, channels, 1, 1].
        - The calculated noise scale, tensor of shape [batch_size, channels, 1, 1].
    """
    #if method not in ['accurate', 'experimental']:
    #    raise ValueError(f'invalid bias calculation method: {method}')
    if isinstance(sigmas, list):
        sigmas = torch.tensor(sigmas, device='cpu')

    # if sample_size is an integer, it is assumed to be a square image
    if isinstance(sample_size, int):
        sample_size = (sample_size, sample_size)

    # calculate the number of diffusion steps (for the progress bar)
    steps = sigmas.shape[-1] - 1

    # if a sample_size is supplied, the 'latent_image' is replaced
    # by an empty one of the specified size
    if isinstance(sample_size, (tuple,list)) and len(sample_size)>=2:
        width, height = sample_size[0], sample_size[1]
        if width>=8 and height>=8:
            samples : torch.Tensor = latent_image["samples"]
            sample_shape = samples.shape[:-2] + ( int(height//8), int(width//8) )
            latent_image  = latent_image.copy()
            latent_image["samples"] =  torch.zeros(sample_shape, dtype=samples.dtype, layout=samples.layout, device="cpu")

    # run the sampler on pure noise and calculate the mean of the result
    latent_image = execute_sampler(latent_image,
                                   model, seed, 1.0, positive, negative,
                                   sampler             = sampler,
                                   sigmas              = sigmas,
                                   noise_bias          = sample_bias,
                                   noise_scale         = sample_scale,
                                   force_final_denoise = False,
                                   progress_preview = ProgressPreview( steps,
                                        parent=(progress_preview, 0, 100)),
                                   )

    samples = latent_image["samples"]
    bias  = samples.mean(dim=[2, 3], keepdim=True)
    scale = samples.std (dim=[2, 3], keepdim=True)

    # TODO: check clamp range
    #bias.clamp_(min=-4.5, max=4.5)
    return bias, scale


def inject_low_freq_noise(x           : torch.Tensor,
                          seed        : int,
                          *,
                          noise_scale : float = 1.0,
                          noise_freq  : int   = 1024,
                          ) -> torch.Tensor:
    """
    Injects low-frequency noise into the input tensor x.

    This function generates noise at a lower resolution based on the specified
    frequency, resulting in a smoother (low-frequency) noise effect when applied
    to the input tensor.

    Args:
        x           : Input tensor to which noise will be added.
        seed        : Seed for random noise generation to ensure reproducibility.
        noise_scale : Scale factor for the noise intensity. Default is 1.0.
                       A 0.0 value disables noise injection.
        noise_freq  : Frequency factor that determines the resolution at which
                       noise is generated. A lower value results in smoother noise.
                       For example, a noise_freq of 1024 means noise is injected
                       into every pixel, while a noise_freq of 512 means noise is
                       injected every second pixel, with intermediate pixels being
                       interpolated.
    Returns:
        The input tensor x with low-frequency noise injected.
    """
    h, w  = x.shape[-2:]
    if noise_scale <= 0.0  or  noise_freq < (1024/h)  or  noise_freq < (1024/w):
        return x

    low_res_shape = ( *x.shape[:-2], (h * noise_freq) // 1024, (w * noise_freq) // 1024 )

    noise = generate_noise(seed,
                           noise_scale = noise_scale,
                           shape       = low_res_shape,
                           dtype       = x.dtype,
                           layout      = x.layout,
                           device      = x.device,
                           )

    # inject the low frequency noise into the input tensor x and return
    return x + F.interpolate(noise,
                             size = (h, w),
                             mode = 'bilinear',
                             align_corners = False,
                             )


#============================ SIGMA OPERATIONS =============================#

def truncate_sigmas_by_step_range(sigmas    : torch.Tensor | None,
                                  step_range: tuple[int,int] | list[int] | None,
                                  *,
                                  first_sigma_step: int = 0,
                                  ) -> torch.Tensor | None:
    """
    Truncate sigmas tensor based on step range.

    Args:
        sigmas          : Tensor of sigma values.
        step_range      : Tuple or list of (start_step, end_step) inclusive-exclusive.
        first_sigma_step: The step number of the first sigma value in the tensor.

    Returns:
        Truncated sigmas tensor, or None if sigmas is invalid.
    """
    if sigmas is None or sigmas.numel() <= 1:
        return None
    if not isinstance(step_range, (list, tuple)) or len(step_range) < 2:
        return sigmas

    start_step = max(0, step_range[0] - first_sigma_step)  # first step (inclusive)
    end_step   = max(0, step_range[1] - first_sigma_step)  # last step (exclusive)
    sigmas = sigmas[start_step:end_step + 1]
    if sigmas.numel()<2:
        return None
    return sigmas


def truncate_sigmas_by_value_range(sigmas      : torch.Tensor | None,
                                   value_range : list[float] | tuple[float,float] | None,
                                   ) -> torch.Tensor | None:
    """
    Truncates a *descending* tensor of sigmas inside a restricted range.

    The returned tensor keeps the original order and contains only the values
    that fall between `limits[0]` and `limits[1]`; if the original tensor
    is clipped at either end, the corresponding limit value is appended
    so that the final result always includes both boundaries.
    If no value satisfies this condition the function returns `None`.

    Args:
        sigmas:
            1-D tensor with sigma values sorted in **descending** order.
        limits:
            Sequence with at least two elements.
            The first and second elements are used as the `(min, max)` range
            that defines the interval for filtering.

    Returns:
        A new tensor with the sigmas that lie inside `[min, max]`.
        The tensor preserves the original descending order.
        If `sigmas` is `None` the function returns `None`.
        If `sigmas` is totally outside the limits, the function returns `None`.
    """
    if sigmas is None or sigmas.numel()<=1:
        return None

    if value_range is None:
        return sigmas
    if not isinstance(value_range, (list, tuple)) or len(value_range) < 2:
        raise ValueError("`limits` must be a list or tuple with at least 2 elements")
    if not (sigmas[0] >= sigmas[-1]):
        raise ValueError("`sigmas` must be sorted in descending order")

    # extract the lower and upper limits
    lower, upper = value_range[0], value_range[1]
    if lower > upper:
        lower, upper = upper, lower

    # if the range of sigmas is totally outside the limits
    # then none of the sigmas should be returned
    if sigmas[-1] > upper or lower > sigmas[0]:
        return None

    # create the mask to filter out values outside the range
    valid_mask = (upper > sigmas) & (sigmas > lower)
    if valid_mask.all():
        return sigmas

    # filter the tensor using the mask
    truncated_sigmas = sigmas[valid_mask]

    # add the lower/upper limits to the tensor if it was actually truncated
    if not valid_mask[0]:
        upper_sigma = torch.tensor([upper], dtype=sigmas.dtype, device=sigmas.device)
        truncated_sigmas = torch.cat((upper_sigma, truncated_sigmas))
    if not valid_mask[-1]:
        lower_sigma = torch.tensor([lower], dtype=sigmas.dtype, device=sigmas.device)
        truncated_sigmas = torch.cat((truncated_sigmas, lower_sigma))

    return truncated_sigmas


def refine_sigma_sequence(sigmas: list[float] | None, insert_count: int) -> list[float]:
    """
    Refines a sequence of sigmas by inserting midpoints between neighbors.

    Args:
        sigmas      : List of sigmas (e.g., [0.948, 0.858, ..., 0.0])
        insert_count: Total number of new sigmas to insert into the list
    Returns:
        A new list containing the original points plus the added midpoints
    """
    if not sigmas or len(sigmas)<2:
        sigmas = [1.0, 0.0]

    # keep looping until there are no more sigmas to insert
    while insert_count > 0:
        new_sequence = [ sigmas[0] ]

        for i in range(len(sigmas) - 1):
            # insert a midpoint between current sigma and next sigma
            if insert_count > 0:
                new_sequence.append( (sigmas[i] + sigmas[i+1]) / 2 )
                insert_count -= 1
            # re-insert the original sigma
            new_sequence.append( sigmas[i+1] )

        sigmas = new_sequence

    return sigmas


#============================== SIGMA PRESETS ==============================#

ALPHA_SIGMA_PRESET = (
    (###3
        (0.991, 0.980, 0.920),                      #< +2 steps
        (0.942, 0.000),                             #< +1 step  | = 3 generation steps
        None,                                       #< (no refiner)
    ),(#4
        (0.991, 0.980, 0.920),                      #< +2 steps
        (0.942, 0.000),                             #< +1 step  | = 3 generation steps
        (0.790, 0.000),                             #< +1 step  | + 1 refiner step
    ),(#5
        (0.991, 0.980, 0.920),                      #< +2 steps
        (0.942, 0.780, 0.000),                      #< +2 steps | = 4 generation steps
        (0.620, 0.000),                             #< +1 step  | + 1 refiner step
    ),(#6
        (0.991, 0.980, 0.920),                      #< +2 steps
        (0.942, 0.780, 0.000),                      #< +2 steps | = 4 generation steps
        (0.658, 0.302, 0.000),                      #< +2 steps | + 2 refiner steps
    ),(#7
        (0.991, 0.980, 0.920),                      #< +2 steps
        (0.935, 0.892, 0.760, 0.000),               #< +3 steps | = 5 generation steps
        (0.658, 0.302, 0.000),                      #< +2 steps | + 2 refiner steps
    ),(#8
        (0.991, 0.980, 0.920),                      #< +2 steps
        (0.935, 0.900, 0.875, 0.750, 0.000),        #< +4 steps | = 6 generation steps
        (0.658, 0.302, 0.000),                      #< +2 steps | + 2 refiner steps
    ),(#9
        (0.991, 0.980, 0.920),                      #< +2 steps
        (0.935, 0.900, 0.875, 0.750, 0.000),        #< +4 steps | = 6 generation steps
        (0.658, 0.456, 0.200, 0.000),               #< +3 steps | + 3 refiner steps
    )
)
BRAVO_SIGMA_PRESET = (
    (###3
        (0.991, 0.920),                             #< +1 step
        (0.942, 0.000),                             #< +1 step  | = 2 generation steps
        (0.710, 0.000),                             #< +1 step  | + 1 refiner step
    ),(#4
        (0.991, 0.920),                             #< +1 step
        (0.935, 0.789, 0.000),                      #< +2 steps | = 3 generation steps
        (0.500, 0.000),                             #< +1 step  | + 1 refiner step
    ),(#5
        (0.991, 0.920),                             #< +1 step
        (0.935, 0.770, 0.690, 0.000),               #< +4 steps | = 4 generation steps
        (0.280, 0.000),                             #< +1 step  | + 1 refiner step
    ),(#6
        (0.991, 0.920),                             #< +1 step
        (0.935, 0.770, 0.690, 0.000),               #< +3 steps | = 4 generation steps
        (0.658, 0.302, 0.000),                      #< +2 steps | + 2 refiner steps
    ),(#7
        (0.991, 0.920),                             #< +1 step
        (0.935, 0.900, 0.875, 0.800, 0.000),        #< +4 steps | = 5 generation steps
        (0.658, 0.302, 0.000),                      #< +2 steps | + 2 refiner steps
    ),(#8
        (0.991, 0.920),                             #< +1 step
        (0.935, 0.900, 0.875, 0.820, 0.750, 0.000), #< +5 steps | = 6 generation steps
        (0.658, 0.302, 0.000),                      #< +2 steps | + 2 refiner steps
    ),(#9
        (0.991, 0.920),                             #< +1 step
        (0.935, 0.900, 0.875, 0.820, 0.750, 0.000), #< +5 steps | = 6 generation steps
        (0.658, 0.4556, 0.200, 0.000),              #< +3 steps | + 3 refiner steps
    )
)

SIGMA_PRESETS_BY_NAME = {
    "alpha"  : ALPHA_SIGMA_PRESET,
    "bravo"  : BRAVO_SIGMA_PRESET,
}

#=== DISCARDED SIGMA PRESETS ===
#
# C_SIGMA_PRESET = (
#     (   (0.992, 0.977, 0.917),                #< 2 steps
#         (0.948, 0.000),                       #< 1 step  (=3 generation steps)
#         None,                                 #< (no refiner)
#     ),(
#         (0.992, 0.977, 0.917),                #< 2 steps
#         (0.948, 0.000),                       #< 1 step  (=3 generation steps)
#         (0.700, 0.000),                       #< 1 step  (+1 refiner step)
#     ),(
#         (0.992, 0.977, 0.917),                #< 2 steps
#         (0.948, 0.740, 0.000),                #< 2 steps (=4 generation steps)
#         (0.700, 0.000),                       #< 1 step  (+1 refiner step)
#     ),(
#         (0.992, 0.977, 0.917),                #< 2 steps
#         (0.948, 0.828, 0.570, 0.000),         #< 3 steps (=5 generation steps)
#         (0.700, 0.000),                       #< 1 step  (+1 refiner step)
#     ),(
#         (0.992, 0.977, 0.917),                #< 2 steps
#         (0.948, 0.828, 0.570, 0.000),         #< 3 steps (=5 generation steps)
#         (0.700, 0.280, 0.000),                #< 2 steps (+2 refiner steps)
#     ),(
#         (0.992, 0.977, 0.917),                #< 2 steps
#         (0.948, 0.858, 0.725, 0.540, 0.000),  #< 4 steps (=6 generation steps)
#         (0.700, 0.280, 0.000),                #< 2 steps (+2 refiner steps)
#     ),(
#         (0.992, 0.977, 0.917),                #< 2 steps
#         (0.948, 0.858, 0.725, 0.540, 0.000),  #< 4 steps (=6 generation steps)
#         (0.708, 0.586, 0.270, 0.000),         #< 3 steps (+3 refiner steps)
#     )
#)
#
# D_SIGMA_PRESET = (
#     (
#         (0.990, 0.981, 0.911),                #< 2 steps
#         (0.943, 0.850, 0.775, 0.640, 0.000),  #< 4 steps (=6 generation steps)
#         (0.608, 0.486, 0.270, 0.000),         #< 3 steps (+3 refiner steps)
#     )
# )
# E_SIGMA_PRESET = (
#     (
#         (0.990, 0.980, 0.913),                #< 2 steps
#         (0.941, 0.858, 0.725, 0.540, 0.000),  #< 4 steps (=6 generation steps)
#         (0.708, 0.586, 0.270, 0.000),         #< 3 steps (+3 refiner steps)
#     )
# )
#
