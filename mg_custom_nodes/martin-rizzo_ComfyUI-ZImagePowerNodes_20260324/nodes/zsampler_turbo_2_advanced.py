"""
File    : zsampler_turbo_2_advanced.py
Purpose : Node for denoising latent images using "Z-Sampler Turbo" (second generation)
          (this advanced version allows you to manually configure additional parameters)
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



class ZSamplerTurbo2Advanced(io.ComfyNode):
    xTITLE         = "Z-Sampler Turbo :geN2 (Advanced)"
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
                'Efficiently denoises latent images, specifically tuned for the "Z-Image Turbo" model. '
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
                io.Boolean.Input     ("add_noise", default=True, label_on="yes", label_off="no",
                                      tooltip="???",
                                     ),
                io.Int.Input         ("seed", default=1, min=1, max=0xffffffffffffffff, control_after_generate=True,
                                      tooltip="The seed used for the random noise generator, ensuring the same result is produced with the same value.",
                                     ),
                io.Int.Input         ("steps", default=8, min=3, max=20, step=1,
                                      tooltip="The number of iterations to be performed during the sampling process.",
                                     ),
                io.Int.Input         ("start_at_step", default=0, min=0, max=100, step=1,
                                      tooltip="The number of iterations to be performed during the sampling process.",
                                     ),
                io.Int.Input         ("end_at_step", default=100, min=0, max=100, step=1,
                                      tooltip="The number of iterations to be performed during the sampling process.",
                                     ),
                io.Boolean.Input     ("force_final_denoising", default=True, label_on="yes", label_off="no",
                                      tooltip="???",
                                     ),

                Divider("divider"),#=========================================

                io.Float.Input       ("vibrance", default=0.0, min=-1.0, max=1.0, step=0.1,
                                      tooltip="The amount of over-amplitude in the initial noise to generate images with "
                                              "more pronounced contrasts and colors. 0.0 means no correction is applied. "
                                              "Negative values result in more washed-out images, while positive values "
                                              "enhance details and vibrancy. "
                                              "This parameter only affects the image when 'start_at_step' is set to 0 "
                                              "and 'add_noise' is enabled. ",
                                     ),
                io.Float.Input       ("initial_bias_level", default=1.5, min=0.0, max=2.0, step=0.1,
                                      tooltip="The level of adjustament from the estimated noise bias to apply before "
                                              "the first denoising step. 0.0 means no noise bias adjustment; 1.0 means "
                                              "using the estimated noise bias. "
                                              "This parameter works only when 'start_at_step' is set to 0 "
                                              "and 'add_noise' is enabled. ",
                                     ),
                io.Combo.Input       ("initial_sample_size", default="image_size", options=["128px", "256px", "512px", "1024px", "image_size"],
                                      tooltip="The size of the latent image used to calculate the initial bias. "
                                              "The smaller the image size, the faster the calculation of the first step. "
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
                positive              : list,
                latent_input          : dict[str, Any],
                add_noise             : bool,
                seed                  : int,
                steps                 : int,
                start_at_step         : int,
                end_at_step           : int,
                force_final_denoising : bool,
                intensity_shift       : float,
                initial_bias_level    : float,
                initial_sample_size   : str,
                **kwargs
                ) -> io.NodeOutput:

        # calculate amount of noise overdose based on the `vibrance_adjustment`
        initial_noise_overdose = (0.2 * ((intensity_shift+1)**2) + 0.8) - 1

        # if the start/stop values restrict the number of steps,
        # apply that start/stop range using the `sigma_step_range` parameter
        sigma_step_range = None
        if start_at_step > 0 or end_at_step<steps:
            sigma_step_range = (start_at_step, end_at_step)

        # run the Z-Sampler Turbo core method on the latent image
        latent_output = zsampler_turbo_core(latent_input, model, positive,
                                            seed                      = seed,
                                            steps                     = steps,
                                            initial_noise_bias_level  = initial_bias_level,
                                            initial_noise_overdose    = initial_noise_overdose,
                                            noise_est_sample_size     = initial_sample_size,
                                            noise_est_sample_bias     = 0.0,
                                            noise_est_sample_scale    = 1.0,
                                            sigma_preset_name         = "alpha",
                                            sigma_step_range          = sigma_step_range,
                                            start_with_noise          = add_noise,
                                            end_with_denoise          = force_final_denoising,
                                            progress_preview = ProgressPreview.from_model( model ),
                                            )

        return io.NodeOutput(latent_output)

