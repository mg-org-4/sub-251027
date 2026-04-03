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
    xTITLE         = "Z-Sampler Turbo ^2 (Advanced)"
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
                'Efficiently denoises the latent image using a process specifically tuned for the "Z-Image Turbo". '
                'This node takes a Z-Image Turbo model, an initial latent image, and conditioning parameters, and '
                'produces a denoised latent output ready for decoding into the final image. This advanced version '
                'includes extra parameters for more precise control and chaining of samplers.'
            ),
            inputs=[
                io.Latent.Input      ("latent_input",
                                      tooltip="The initial latent image to be modified; typically an 'Empty Latent' "
                                              "for text-to-image or an encoded image for img2img. ",
                                     ),
                io.Model.Input       ("model",
                                      tooltip="The Z-Image Turbo model used for denoising the latent image. "
                                     ),
                io.Conditioning.Input("positive",
                                      tooltip="The main conditioning used to guide the generation process toward "
                                              "the desired content. ",
                                     ),
                io.Conditioning.Input("positive_stg2", optional=True,
                                      tooltip="This input is optional and can remain disconennect. It allows defining "
                                              "a different conditioning for the second stage of the denoising process. ",
                                     ),
                io.Conditioning.Input("positive_stg3", optional=True,
                                      tooltip="This input is optional and can remain disconennect. It allows defining "
                                              "a different conditioning for the third stage of the denoising process. ",
                                     ),
                io.Boolean.Input     ("add_noise", default=True, label_on="yes", label_off="no",
                                     tooltip="Determines whether to add initial noise to the latent image. Recommended "
                                             "for most cases. Disabling this is useful for sampler chaining when the "
                                             "input latent already contains residual noise from a previous process. ",
                                     ),
                io.Int.Input         ("seed", default=1, min=1, max=0xffffffffffffffff, control_after_generate=True,
                                      tooltip="The seed used for the random noise generator, ensuring the same result "
                                              "is produced with the same value.",
                                     ),
                io.Int.Input         ("steps", default=8, min=3, max=20, step=1,
                                      tooltip="The number of iterations to be performed during the denoising process.",
                                     ),
                io.Int.Input         ("start_at_step", default=0, min=0, max=100, step=1,
                                      tooltip="The step at which the sampling process should start, allowing for more "
                                              "precise control over the denoising process and enabling sampler chaining. ",
                                      ),
                io.Int.Input         ("end_at_step", default=100, min=0, max=100, step=1,
                                      tooltip="The step at which the sampling process should end. allowing for more "
                                              "precise control over the denoising process and enabling sampler chaining. ",
                                     ),
                io.Boolean.Input     ("force_final_denoising", default=True, label_on="yes", label_off="no",
                                      tooltip="Determines whether to force a full final denoising step, resulting in "
                                              "a output latent with no residual noise. Recommended for most cases. "
                                              "Disabling this is useful when residual noise is required for the next "
                                              "process in a sampler chain. ",
                                     ),

                Divider("divider"),#=========================================

                io.Float.Input       ("z_vibrance", default=0.0, min=-1.0, max=1.0, step=0.1,
                                      tooltip="The amount of over-amplitude in the initial noise to generate images with "
                                              "more pronounced contrasts and colors. 0.0 means no correction is applied. "
                                              "Negative values result in more washed-out images, while positive values "
                                              "enhance intensity and saturation. This parameter only affects the image "
                                              "when 'start_at_step' is set to 0 and 'add_noise' is enabled. ",
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
                io.Latent.Output(display_name="latent_output",
                                 tooltip="The resulting denoised latent image, ready to be decoded "
                                         "by a VAE or passed to another node for further processing. ",
                                )
            ]
        )

    #__ FUNCTION __________________________________________
    @classmethod
    def execute(cls,
                latent_input          : dict[str, Any],
                model                 : Any,
                positive              : list,
                add_noise             : bool,
                seed                  : int,
                steps                 : int,
                start_at_step         : int,
                end_at_step           : int,
                force_final_denoising : bool,
                z_vibrance            : float,
                initial_bias_level    : float,
                initial_sample_size   : str,
                positive_stg2         : list | None = None,
                positive_stg3         : list | None = None,
                **kwargs
                ) -> io.NodeOutput:

        # calculate the amount of noise overdose based on `z_vibrance`
        initial_noise_overdose = (0.2 * ((z_vibrance+1)**2) + 0.8) - 1

        # if the start/stop values restrict the number of steps,
        # apply that start/stop range using the `sigma_step_range` parameter
        sigma_step_range = None
        if start_at_step > 0 or end_at_step<steps:
            sigma_step_range = (start_at_step, end_at_step)

        # run the Z-Sampler Turbo core method on the latent image
        latent_output = zsampler_turbo_core(latent_input, model, positive,
                                            positive_stg2             = positive_stg2,
                                            positive_stg3             = positive_stg3,
                                            seed                      = seed,
                                            steps                     = steps,
                                            initial_noise_bias_level  = initial_bias_level,
                                            initial_noise_overdose    = initial_noise_overdose,
                                            noise_est_sample_size     = initial_sample_size,
                                            noise_est_sample_bias     = 0.0,
                                            noise_est_sample_scale    = 1.0,
                                            sigma_preset_name         = "bravo",
                                            sigma_step_range          = sigma_step_range,
                                            start_with_noise          = add_noise,
                                            end_with_denoise          = force_final_denoising,
                                            progress_preview = ProgressPreview.from_model( model ),
                                            )

        return io.NodeOutput(latent_output)

