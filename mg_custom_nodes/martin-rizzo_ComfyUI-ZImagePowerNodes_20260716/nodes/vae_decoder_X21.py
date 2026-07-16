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
from typing                 import Final
from comfy_api.latest       import io
from torchvision.transforms import v2
import comfy.sd
import torch


class VAEDecoderX21(io.ComfyNode):
    xTITLE         = "VAE Decoder ^G2.1"
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
                "Experimental node similar to native ComfyUI VAEDecode node but including a color filtering step. "
            ),
            search_aliases=["decode", "decode latent", "latent to image", "render latent"],
            inputs=[
                io.Latent.Input      ("samples",
                                      tooltip="The latent image to be decoded. "
                                     ),
                io.Vae.Input         ("vae",
                                      tooltip="The VAE model used for decoding the latent image. "
                                     ),
                io.Combo.Input       ("color_filter",
                                      options=["none", "intensity1", "intensity2"],
                                      tooltip="The color filter to be applied to the decoded image. "
                                     ),
            ],
            outputs=[
                io.Image.Output(tooltip="The decoded image. "),
            ]
        )

    #__ FUNCTION __________________________________________
    @classmethod
    def execute(cls, vae : comfy.sd.VAE, samples: dict[str, torch.Tensor], color_filter: str):

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

        if color_filter == "intensity1":
            images = cls.adjust_saturation(images, 1.50)

        if color_filter == "intensity2":
            images = cls.stretch_histogram(images)
            images = cls.adjust_saturation(images, 1.50)
            images = cls.apply_s_curve(images, contrast_factor=1.1)
            images = cls.apply_dithering(images, amplitude_bits=8)

        images = images.permute(0, 2, 3, 1).contiguous()
        return (images, )


    #__ internal functions ________________________________

    @staticmethod
    def apply_s_curve(images: torch.Tensor, contrast_factor: float = 1.5) -> torch.Tensor:
        """
        Applies a non-linear S-curve transformation to adjust the contrast of the input images.
        Args:
            images:          A tensor containing a batch of images. Shape [B, H, W, C]
                             with values normalized in the range [0.0, 1.0].
            contrast_factor: The intensity of the S-curve transformation.
                              *   1.0: Identity curve (no change to the image).
                              * > 1.0: Increases contrast by stretching the midtones.
                              * < 1.0: Reduces contrast by compressing the midtones.
        Returns:
            A tensor containing the batch of images with the transformation applied.
        """
        # temporarily map the range to [-1, 1]
        images = 2.0 * torch.clamp(images, 0.0, 1.0) - 1.0

        # apply exponential correction while preserving the original sign,
        # this depresses shadows (negatives) and elevates highlights (positives)
        # when contrast_factor is greater than 1
        corrected = torch.sign(images) * torch.pow(torch.abs(images), 1.0 / contrast_factor)

        # map back to the original range and return
        return (corrected + 1.0) / 2.0


    @staticmethod
    def stretch_histogram(images: torch.Tensor, q_lower: float = 0.001, q_upper: float = 0.999) -> torch.Tensor:
        """
        Expands the dynamic range of a batch of images by stretching the histogram.
        Args:
            images:   A tensor containing a batch of images. Shape [B, H, W, C]
                      with values normalized in the range [0.0, 1.0].
            q_lower:  The lower quantile to ignore (e.g., 0.001 for bottom 0.1%).
            q_upper:  The upper quantile to ignore (e.g., 0.999 for top 0.1%).
        Returns:
            A tensor containing the batch of images with the dynamic range expanded.
        """
        b, h, w, c = images.shape

        # flatten H, W, C dimensions per batch image -> [B, H*W*C]
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
    def adjust_saturation(images: torch.Tensor, saturation_factor: float) -> torch.Tensor:
        """
        Adjusts the saturation level of images.
        Args:
            images:            A tensor containing a batch of images. Shape [B, H, W, C]
                               with values normalized in the range [0.0, 1.0].
            saturation_factor: The multiplier for color saturation.
        Returns:
            A tensor containing the batch of images with adjusted saturation.
        """
        return v2.functional.adjust_saturation(images, saturation_factor)


    @staticmethod
    def apply_dithering(images: torch.Tensor, amplitude_bits: float = 1.0) -> torch.Tensor:
        """
        Applies dithering to reduce banding by adding noise.
        Args:
            images:         A tensor containing a batch of images. Shape [B, H, W, C]
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
