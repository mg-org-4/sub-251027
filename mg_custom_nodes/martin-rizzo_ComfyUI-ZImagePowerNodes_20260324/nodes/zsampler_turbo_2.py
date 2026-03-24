"""
File    : zsampler_turbo_2.py
Purpose : Node for denoising latent images using "Z-Sampler Turbo" (second generation)
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Mar 20, 2026
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
def Divider(id: str):
    return io.Custom("ZIPN_DIVIDER").Input(id = id)



class ZSamplerTurbo2(io.ComfyNode):
    xTITLE         = "Z-Sampler Turbo :geN2"
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
                io.Int.Input         ("steps", default=9, min=3, max=20, step=1,
                                      tooltip="The number of iterations to be performed during the sampling process.",
                                     ),
                io.Float.Input       ("denoise", default=1.00, min=0.00, max=1.00, step=0.01,
                                      tooltip="The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.",
                                     ),

                Divider("divider"),#=========================================

                io.Float.Input       ("vibrance", default=0.0, min=-1.0, max=1.0, step=0.1,
                                      tooltip="The amount of over-amplitude in the initial noise to generate images with "
                                              "more pronounced contrasts and colors. 0.0 means no correction is applied. "
                                              "Negative values result in more washed-out images, while positive values "
                                              "enhance details and vibrancy. This parameter only affects the image when "
                                              "'denoise' is set to 1.00. "
                                     ),
                io.Boolean.Input     ("use_lowres_sample", default=False, label_on="yes", label_off="no",
                                      tooltip="When enabled, this option uses a smaller latent image to estimate initial "
                                              "noise features, accelerating the first step. If disabled, the full input "
                                              "image size is used. This parameter is only relevant when 'denoise' is set "
                                              "to 1.00."
                                     ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent_output",
                                 tooltip="The resulting denoised latent image, ready to be decoded by a VAE "
                                         "or passed to another node for further processing.",
                                ),
            ]
        )

    #__ FUNCTION __________________________________________
    @classmethod
    def execute(cls,
                model,
                positive     : list,
                latent_input : dict[str, Any],
                seed         : int,
                steps        : int,
                denoise      : float,
                vibrance     : float,
                use_lowres_sample: bool,
                **kwargs
                ) -> io.NodeOutput:

        # sets sigma limits when denoise is less than 1.0 (mostly when performing inpainting)
        sigma_limits = ( denoise**0.5 , 0 ) if denoise < 0.999 else None

        # hardcode the initial noise bias level to 1.5
        initial_noise_bias_level = 1.5

        # calculate the amount of noise overdose based on `vibrance`
        initial_noise_overdose = (0.2 * ((vibrance+1)**2) + 0.8) - 1

        # run the Z-Sampler Turbo core method on the latent image
        latent_output = zsampler_turbo_core(latent_input, model, positive,
                                            seed                      = seed,
                                            steps                     = steps,
                                            initial_noise_bias_level  = initial_noise_bias_level,
                                            initial_noise_overdose    = initial_noise_overdose,
                                            noise_est_sample_size     = 512 if use_lowres_sample else "image_size",
                                            sigma_preset_name         = "alpha",
                                            sigma_limits              = sigma_limits,
                                            progress_preview = ProgressPreview.from_model( model ),
                                            )

        return io.NodeOutput(latent_output)
