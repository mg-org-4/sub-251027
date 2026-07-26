"""
File    : vae_decoder_X21.py
Purpose : Experimental node to decode latent images back into pixels with filter support.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jun 24, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
     ComfyUI nodes designed to power the "Z-Image/Z-Image Turbo" models.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

 ComfyUI V3 Schema oficial documentation:
 - https://docs.comfy.org/custom-nodes/v3_migration

"""
import torch
import kornia
import comfy.sd
from typing           import Final
from comfy_api.latest import io


#class AdjustableVAEDecoderX21
class VAEDecoderX21(io.ComfyNode):
    xTITLE         = "Adjustable VAE Decoder ^G2.1"
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
                "An experimental alternative to the native ComfyUI VAEDecode node. It includes "
                "integrated post-processing tools to easily adjust color tone and contrast in a "
                "simple way without requiring external nodes."
            ),
            search_aliases=["decode", "decode latent", "latent to image", "render latent"],
            inputs=[
                io.Latent.Input      ("samples",
                                      tooltip="The latent representation to be decoded into an image."
                                     ),
                io.Vae.Input         ("vae",
                                      tooltip="The VAE model used to decode the latent input."
                                     ),
                io.Combo.Input       ("filter",
                                      options=["none", "bw", "color", "color_twist", "intensity_1", "intensity_2", ],
                                      tooltip="The color correction filter to apply to the resulting image."
                                     ),
                io.Float.Input       ("filter_shift", min=-0.5, max=0.5, default=0.0, step=0.1,
                                      tooltip="The calibration offset for the selected filter. "
                                              "This value has a range from -0.5 to 0.5 with 0.0 as the "
                                              "default ideal baseline balance."

                                      ),
                io.Boolean.Input     ("auto_contrast",
                                      tooltip="When enabled, automatically expands the dynamic range of the "
                                              "image to fix washed-out tones and optimize contrast boundaries."
                                     ),
            ],
            outputs=[
                io.Image.Output(tooltip="The final decoded image with post-processing adjustments applied."),
            ]
        )

    #__ FUNCTION __________________________________________
    @classmethod
    def execute(cls,
                vae          : comfy.sd.VAE,
                samples      : dict[str, torch.Tensor],
                filter       : str,
                filter_shift : float,
                auto_contrast: bool,
                ):

        # extract latents from samples
        latents: torch.Tensor = samples["samples"]
        if latents.is_nested:
            latents = latents.unbind()[0]

        # decode all latents to images
        # images = [B, H, W, C]
        images = vae.decode(latents)

        # convert video(?) tensor to standard image batch:
        # [B, T, H, W, C] -> [(B*T), H, W, C]
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        elif len(images.shape) == 3:
            images = images.unsqueeze(0)

        images = images.permute(0, 3, 1, 2)

        if filter == "bw":
            contrast_factor =  1 + (filter_shift*1.2 if filter_shift<0 else filter_shift*0.8)
            images = cls.adjust_hsv_components(images, saturation_target=0, contrast_scurve_factor=contrast_factor)

        elif filter == "color":
            contrast_factor =  1 + (filter_shift*1.0 if filter_shift<0 else filter_shift*1.0)
            images = cls.adjust_hsv_components(images, saturation_scurve_factor=contrast_factor, contrast_scurve_factor=(contrast_factor-1)*0.2 + 1)
            images = images

        elif filter == "color_twist":
            images = cls.adjust_hsv_components(images, hue_shift_factor=filter_shift*2)

        elif filter == "intensity_1":
            contrast_factor =  1 + (filter_shift*1.0 if filter_shift<0 else filter_shift*1.0)
            images = cls.adjust_hsv_components(images, saturation_target=0.35, contrast_scurve_factor=contrast_factor)

        elif filter == "intensity_2":
            contrast_factor =  1 + (filter_shift*1.0 if filter_shift<0 else filter_shift*1.0)
            images = cls.adjust_hsv_components(images, saturation_target=0.40, contrast_scurve_factor=contrast_factor)

        if auto_contrast:
            images = cls.stretch_histogram(images)

        images = images.permute(0, 2, 3, 1).contiguous()
        return (images, )


    #__ internal functions ________________________________

    @staticmethod
    def adjust_hsv_components(images: torch.Tensor,
                              *,
                              saturation_scurve_factor: float = 0.0,
                              contrast_scurve_factor  : float = 0.0,
                              saturation_target       : float = -1.0,
                              brightness_target       : float = -1.0,
                              hue_shift_factor        : float = 0.0
                              ) -> torch.Tensor:
        """
        Adjust HSV color space properties of RGB images using S-curve mapping and target normalization.

        Args:
            images                  : Input tensor of RGB images with shape [B, 3, H, W] and range [0, 1].
            saturation_scurve_factor: Power factor to apply an S-curve adjustment to saturation.
            contrast_scurve_factor  : Power factor to apply an S-curve adjustment to value/contrast.
            saturation_target       : Target mean saturation value (if < 0, the parameter is ignored.)
            brightness_target       : Target mean brightness value (if < 0, the parameter is ignored.)
            hue_shift_factor        : Factor to shift the hue based on the value component.

        Returns:
            A tensor of RGB images with the applied HSV adjustments, in the range.
        """

        # transform from RGB to HSV
        hsv = kornia.color.rgb_to_hsv(images)
        h = hsv[:, 0:1, :, :]
        s = hsv[:, 1:2, :, :]
        v = hsv[:, 2:3, :, :]

        # apply s-curve adjustment to saturation
        if saturation_scurve_factor > 0.0 and saturation_scurve_factor != 1.0:
            s = torch.sign(s) * torch.pow(torch.abs(s), 1.0 / saturation_scurve_factor)

        # apply s-curve adjustment to value (contrast)
        if contrast_scurve_factor > 0.0 and contrast_scurve_factor != 1.0:
            v = torch.sign(v) * torch.pow(torch.abs(v), 1.0 / contrast_scurve_factor)

        # normalize saturation to a target mean
        if saturation_target >= 0.0:
            s_mean = torch.mean(s, dim=(2, 3), keepdim=True) #< result: [B, 1, 1, 1]
            s_mean = torch.clamp(s_mean, min=1e-5)
            s = s * (saturation_target / s_mean)
            s = torch.clamp(s, 0.0, 1.0)

        # normalize brightness to a target mean
        if brightness_target >= 0.0:
            v_mean = torch.mean(s, dim=(2, 3), keepdim=True) #< result: [B, 1, 1, 1]
            v_mean = torch.clamp(v_mean, min=1e-5)
            v = v * (brightness_target / v_mean)
            v = torch.clamp(s, 0.0, 1.0)

        # apply hue shift
        if hue_shift_factor != 0.0:
            hue_shift = hue_shift_factor * (v - 0.5)
            h = torch.remainder(h + hue_shift, 2 * 3.14159)

        # reconstruct HSV and convert back to RGB
        hsv = torch.cat([h, s, v], dim=1)
        images = kornia.color.hsv_to_rgb(hsv)
        return images


    @staticmethod
    def stretch_histogram(images: torch.Tensor,
                          *,
                          q_lower: float = 0.001,
                          q_upper: float = 0.999
                          ) -> torch.Tensor:
        """
        Expands the dynamic range of a batch of images by stretching the histogram.

        Args:
            images : A tensor containing a batch of images. Shape [B, C, H, W]
                     with values normalized in the range [0.0, 1.0].
            q_lower: The lower quantile to ignore (e.g., 0.001 for bottom 0.1%).
            q_upper: The upper quantile to ignore (e.g., 0.999 for top 0.1%).

        Returns:
            A tensor containing the batch of images with the dynamic range expanded.
        """
        b, c, h, w = images.shape

        # flatten H, W, C dimensions per batch image -> [B, C*H*W]
        flattened = images.view(b, -1)

        # calculate quantiles for each image independently;
        # torch.quantile returns the value at which a specific percentage of data falls;
        # dim=1 ensures independent calculation per image; reshape to [B,1,1,1] allows broadcasting
        low_vals  = torch.quantile(flattened, q_lower, dim=1, keepdim=True).view (b, 1, 1, 1)
        high_vals = torch.quantile(flattened, q_upper, dim=1, keepdim=True).view (b, 1, 1, 1)

        # calculate the range of values for each image and stretch it to [0,1]
        dynamic_range = (high_vals - low_vals).clamp(min=1e-7)
        images = (images - low_vals) / dynamic_range

        # clamp as outliers outside the quantiles will exceed the [0,1] range
        return torch.clamp(images, 0.0, 1.0)


    @staticmethod
    def apply_dithering(images: torch.Tensor,
                        *,
                        amplitude_bits: float = 1.0
                        ) -> torch.Tensor:
        """
        Applies dithering to reduce banding by adding noise.

        Args:
            images        : A tensor containing a batch of images. Shape [B, C, H, W]
                            with values normalized in the range [0.0, 1.0].
            amplitude_bits: Amplitude of the noise relative to an 8-bit color step.
        Returns:

            A tensor containing the batch of images with dithering applied.
        """
        # the value of a single color step in an 8-bit space (1/255 ≈ 0.00392)
        STEP_SIZE: Final = 1.0 / 255.0

        # add uniform noise centered at zero with the desired amplitude
        noise = (torch.rand_like(images) - 0.5) * (amplitude_bits * STEP_SIZE)
        return torch.clamp(images + noise, 0.0, 1.0)
