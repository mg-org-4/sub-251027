"""
File    : zsampler_turbo_2_laboratory.py
Purpose : Development node for experimenting with the `zsampler_turbo_core(..)` method.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Mar 18, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
         ComfyUI nodes designed specifically for the "Z-Image" model.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

  ComfyUI V3 schema documentation can be found here:
  - https://docs.comfy.org/custom-nodes/v3_migration

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from typing            import Any
from comfy_api.latest  import io
from .lib.progress_bar        import ProgressPreview
from .lib.zsampler_turbo_core import zsampler_turbo_core
def io_Divider(id: str):
    return io.Custom("ZIPN_DIVIDER").Input(id = id)


class ZSamplerTurbo2Laboratory(io.ComfyNode):
    xTITLE         = "Z-Sampler Turbo ^2 (Laboratory)"
    xCATEGORY      = ""
    xCOMFY_NODE_ID = ""
    xDEPRECATED    = False

    #__ INPUT / OUTPUT ____________________________________
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            display_name  = cls.xTITLE,
            category      = cls.xCATEGORY,
            node_id       = cls.xCOMFY_NODE_ID,
            is_deprecated = cls.xDEPRECATED,
            description   = (
                'Efficiently denoises the latent image, specifically tuned for the "Z-Image Turbo" model. '
                'This node takes a Z-Image Turbo model, an initial latent image, and conditioning parameters, '
                'and produces a denoised output ready for further processing or decoding.'
            ),
            inputs=[
                io.Model.Input       ("model",
                                      tooltip="The model used for generating the latent images.",
                                     ),
                io.Conditioning.Input("positive",
                                      tooltip="The conditioning used to guide the generation process toward the desired content.",
                                     ),
                io.Latent.Input      ("latent_input",
                                      tooltip="The initial latent image to be modified; typically an 'Empty Latent' for text-to-image or an encoded image for img2img.",
                                     ),
                io.Int.Input         ("seed", default=1, min=1, max=0xffffffffffffffff, control_after_generate=True,
                                      tooltip="The seed used for the random noise generator, ensuring the same result is produced with the same value.",
                                     ),
                io.Int.Input         ("steps", default=8, min=3, max=20, step=1,
                                      tooltip="The number of iterations to be performed during the sampling process.",
                                     ),
                io.Float.Input       ("denoise", default=1.0, min=0.00, max=1.00, step=0.01,
                                      tooltip="The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.",
                                     ),

                io_Divider("divider1"),#=====================================

                io.Float.Input       ("initial_noise_bias_level", default=1.5, min=0.0, max=10.0, step=0.5,
                                      tooltip="The level of adjustament from the calculated noise bias "
                                              "to apply before the first denoising step. "
                                              "(0.0 means no noise bias adjustment; 1.0 means using the calculated noise bias).",
                                     ),
                io.Float.Input       ("initial_noise_overdose", default=0.2, min=-1.0, max=1.0, step=0.1,
                                      tooltip="The amount of over-amplitude in the initial noise generation. "
                                              "(negative values ​​will reduce any excessive amplitude)."
                                     ),
                io.Combo.Input       ("noise_est_sample_size", default="image_size", options=["image_size", "1024px", "512px", "256px"],
                                      tooltip="The size of the latent image used to calculate the initial noise. "
                                              "The smaller the image size, the faster the calculation of the first step. "
                                     ),
                io.Float.Input       ("noise_est_sample_bias", default=0.00, min=0.00, max=1.00, step=0.01,
                                      tooltip="The bias of the latent image used to calculate the initial noise. "
                                     ),
                io.Float.Input       ("noise_est_sample_scale", default=1.00, min=0.00, max=1.00, step=0.01,
                                      tooltip="The scale of the latent image used to calculate the initial noise. "
                                     ),

                io_Divider("divider2"),#=====================================

                io.Float.Input       ("inject_noise_st1", default=7.5, min=0.0, max=50.0, step=0.1,
                                      tooltip="The amount of noise to be injected into the latent image during the first stage. "
                                              "A value of 0 means that no noise will be injected. ",
                                     ),
                io.Int.Input         ("inject_freq_st1", default=32, min=0, max=1024,
                                      tooltip="?? ",
                                     ),
                io.Float.Input       ("inject_noise_st2", default=3.7, min=0.0, max=50.0, step=0.1,
                                      tooltip="The amount of noise to be injected into the latent image during the second stage. "
                                              "A value of 0 means that no noise will be injected. ",
                                     ),
                io.Int.Input         ("inject_freq_st2", default=64, min=0, max=1024,
                                      tooltip="?? ",
                                     ),
                io.Float.Input       ("inject_noise_st3", default=1.0, min=0.0, max=50.0, step=0.1,
                                      tooltip="The amount of noise to be injected into the latent image during the third stage. "
                                              "A value of 0 means that no noise will be injected. ",
                                     ),
                io.Int.Input         ("inject_freq_st3", default=512, min=0, max=1024,
                                      tooltip="?? ",
                                     ),

                io_Divider("divider3"),#=====================================

                io.Combo.Input       ("sigma_preset_name", default="bravo", options=["alpha", "bravo", "charlie"],
                                      tooltip="The set of predefined sigma values that are used during the denoise process. "
                                     ),
                io.Float.Input       ("sigma0_off", default=0.000, min=-1.000, max=1.000, step=0.001,
                                      #tooltip="",
                                     ),
                io.Float.Input       ("sigma1_off", default=0.000, min=-1.000, max=1.000, step=0.001,
                                      #tooltip="",
                                     ),
                io.Float.Input       ("sigma2_off", default=0.000, min=-1.000, max=1.000, step=0.001,
                                      #tooltip="",
                                     ),
                io.Float.Input       ("sigma3_off", default=0.000, min=-1.000, max=1.000, step=0.001,
                                      #tooltip="",
                                     ),
                io.Float.Input       ("sigma4_off", default=0.000, min=-1.000, max=1.000, step=0.001,
                                      #tooltip="",
                                     ),
                io.Float.Input       ("sigma5_off", default=0.000, min=-1.000, max=1.000, step=0.001,
                                      #tooltip="",
                                     ),
                io.Float.Input       ("sigma6_off", default=0.000, min=-1.000, max=1.000, step=0.001,
                                      #tooltip="",
                                     ),
                io.Float.Input       ("sigma7_off", default=0.000, min=-1.000, max=1.000, step=0.001,
                                      #tooltip="",
                                     ),
                io.Float.Input       ("sigma8_off", default=0.000, min=-1.000, max=1.000, step=0.001,
                                      #tooltip="",
                                     ),
                io.Float.Input       ("sigma9_off", default=0.000, min=-1.000, max=1.000, step=0.001,
                                      #tooltip="",
                                     ),
                io.Float.Input       ("sigma10_off", default=0.000, min=-1.000, max=1.000, step=0.001,
                                      #tooltip="",
                                     ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent_output", tooltip="The resulting denoised latent image, ready to be decoded by a VAE or passed to another sampler."),
            ]
        )

    #__ FUNCTION __________________________________________
    @classmethod
    def execute(cls,
                model,
                positive    : list,
                latent_input: dict[str, Any],
                seed        : int,
                steps       : int,
                denoise     : float,
                noise_est_sample_size    : str | int | None,
                noise_est_sample_bias    : float,
                noise_est_sample_scale   : float,
                initial_noise_bias_level : float,
                initial_noise_overdose   : float,
                inject_noise_st1         : float,
                inject_freq_st1          : int,
                inject_noise_st2         : float,
                inject_freq_st2          : int,
                inject_noise_st3         : float,
                inject_freq_st3          : int,
                sigma_preset_name        : str,
                sigma0_off               : float,
                sigma1_off               : float,
                sigma2_off               : float,
                sigma3_off               : float,
                sigma4_off               : float,
                sigma5_off               : float,
                sigma6_off               : float,
                sigma7_off               : float,
                sigma8_off               : float,
                sigma9_off               : float,
                sigma10_off              : float,
                **kwargs
                ) -> io.NodeOutput:

        # sets sigma limits when denoise is less than 1.0 (mostly when performing inpainting)
        sigma_limits = ( denoise**0.5 , 0 ) if denoise < 0.999 else None

        # creates a list of sigma offsets
        sigma_offsets = [sigma0_off, sigma1_off, sigma2_off, sigma3_off, sigma4_off, sigma5_off, sigma6_off, sigma7_off, sigma8_off, sigma9_off, sigma10_off]

        # creates a list of inject noise scales
        inject_noise_scales = (inject_noise_st1, inject_noise_st2, inject_noise_st3)
        inject_noise_freqs  = (inject_freq_st1 , inject_freq_st2 , inject_freq_st3 )

        # run the Z-Sampler Turbo core method on the latent image
        latent_output = zsampler_turbo_core(latent_input, model, positive,
                                            seed                      = seed,
                                            steps                     = steps,
                                            initial_noise_bias_level  = initial_noise_bias_level,
                                            initial_noise_overdose    = initial_noise_overdose,
                                            noise_est_sample_size     = noise_est_sample_size,
                                            noise_est_sample_bias     = noise_est_sample_bias,
                                            noise_est_sample_scale    = noise_est_sample_scale,
                                            sigma_preset_name         = sigma_preset_name,
                                            sigma_offsets             = sigma_offsets,
                                            sigma_limits              = sigma_limits,
                                            inject_noise_scales       = inject_noise_scales,
                                            inject_noise_freqs        = inject_noise_freqs,
                                            progress_preview = ProgressPreview.from_model( model ),
                                            )

        return io.NodeOutput(latent_output)
