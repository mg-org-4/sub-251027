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
    xTITLE         = "Z-Sampler Turbo ^2"
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
                'produces a denoised latent output ready for decoding into the final image.'
            ),
            inputs=[
                io.Latent.Input      ("latent_input",
                                      tooltip="The initial latent image to be modified; typically an 'Empty Latent' for "
                                              "text-to-image or an encoded image for img2img.",
                                     ),
                io.Model.Input       ("model",
                                      tooltip="The model used for generating the latent images.",
                                     ),
                io.Conditioning.Input("positive",
                                      tooltip="The conditioning used to guide the generation process toward the desired "
                                               "content.",
                                     ),
                io.Conditioning.Input("positive_stg2", optional=True,
                                      tooltip="This input is optional and can remain disconennect. It allows defining "
                                              "a different conditioning for the second stage of the denoising process.",
                                     ),
                io.Conditioning.Input("positive_stg3", optional=True,
                                      tooltip="This input is optional and can remain disconennect. It allows defining "
                                              "a different conditioning for the third stage of the denoising process.",
                                     ),
                io.Int.Input         ("seed", default=1, min=1, max=0xffffffffffffffff, control_after_generate=True,
                                      tooltip="The seed used for the random noise generator, ensuring the same result " 
                                              "is produced with the same value. ",
                                     ),
                io.Int.Input         ("steps", default=8, min=3, max=20, step=1,
                                      tooltip="The number of iterations to be performed during the sampling process.",
                                     ),
                io.Float.Input       ("denoise", default=1.00, min=0.00, max=1.00, step=0.01,
                                      tooltip="The amount of denoising applied, lower values will maintain the structure "
                                              "of the initial image allowing for image to image sampling.",
                                     ),

                Divider("divider"),#=========================================

                io.Float.Input       ("z_vibrance", default=0.0, min=-1.0, max=1.0, step=0.1,
                                      tooltip="The amount of over-amplitude in the initial noise to generate images with "
                                              "more pronounced contrasts and colors. 0.0 means no correction is applied. "
                                              "Negative values result in more washed-out images, while positive values "
                                              "enhance intensity and saturation. This parameter only affects the image "
                                              "when 'denoise' is set to 1.00. ",
                                     ),
                io.Combo.Input       ("mode", default="normal", options=["normal", "detailed", "variety"],
                                      tooltip="?? "
                                     ),
                io.Boolean.Input     ("lowres_sample", default=False, label_on="yes", label_off="no",
                                      tooltip="When enabled, this option uses a smaller latent image to estimate initial "
                                              "noise features, accelerating the first step. If disabled, the full input "
                                              "image size is used. This parameter is only relevant when 'denoise' is set "
                                              "to 1.00. ",
                                     ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent_output",
                                 tooltip="The resulting denoised latent image, ready to be decoded "
                                         "by a VAE or passed to another node for further processing. ",
                                ),
            ]
        )

    #__ FUNCTION __________________________________________
    @classmethod
    def execute(cls,
                latent_input  : dict[str, Any],
                model         : Any,
                positive      : list,
                seed          : int,
                steps         : int,
                denoise       : float,
                z_vibrance    : float,
                mode          : str,
                lowres_sample : bool,
                positive_stg2 : list | None = None,
                positive_stg3 : list | None = None,
                **kwargs
                ) -> io.NodeOutput:

        # sets sigma limits when denoise is less than 1.0 (mostly when performing inpainting)
        sigma_limits = ( denoise**0.5 , 0 ) if denoise < 0.999 else None

        # hardcode the initial noise bias level to 1.5
        initial_noise_bias_level = 1.5

        # calculate the amount of noise overdose based on `z_vibrance`
        initial_noise_overdose = (0.2 * ((z_vibrance+1)**2) + 0.8) - 1

        inject_noise_scales = None
        inject_noise_freqs  = None
        if mode == "detailed":
            inject_noise_scales = ( 7.5, 3.7, 3.0)
            inject_noise_freqs  = (32,64,768)
        elif mode == "variety":
            inject_noise_scales = (19.0, 1.0, 1.0)
            inject_noise_freqs  = (20,35,512)


        ## ( 7.5, 3.7, 1.0) / (32,64,512)
        ## (20.0, 1.0, 1.0) / (20,64,512)
        ## (18.0, 1.5, 1.0) / (20,40,512)

        # run the Z-Sampler Turbo core method on the latent image
        latent_output = zsampler_turbo_core(latent_input, model, positive,
                                            positive_stg2             = positive_stg2,
                                            positive_stg3             = positive_stg3,
                                            seed                      = seed,
                                            steps                     = steps,
                                            initial_noise_bias_level  = initial_noise_bias_level,
                                            initial_noise_overdose    = initial_noise_overdose,
                                            noise_est_sample_size     = 512 if lowres_sample else "image_size",
                                            sigma_preset_name         = "bravo",
                                            sigma_limits              = sigma_limits,
                                            inject_noise_scales       = inject_noise_scales,
                                            inject_noise_freqs        = inject_noise_freqs,
                                            progress_preview = ProgressPreview.from_model( model ),
                                            )

        return io.NodeOutput(latent_output)
