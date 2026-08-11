"""
File    : basic_image_filters.py
Purpose : Experimental node to apply simple filters to generated images.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jul 27, 2026
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
from enum                import Enum
from math                import pi
from typing              import Any
from functools           import cache
from comfy_api.latest    import io
from .                   import widgets as zp
from .core.helpers_image import adjust_hsv_components, stretch_histogram, apply_dithering, convert_to_rgb
class Filter(Enum):
    """Enum representing available image processing filters.
    Attributes:
        NONE            : Represents no filter applied.
        BLACK_AND_WHITE : Grayscale filter.
        SATURATION      :
        COLOR_POP       :
        COLOR           : Standard color level filter.
        COLOR_TWIST     : Hue component twist filter.
        CONTRAST        :
        INTENSITY_1     : Low color intensity filter.
        INTENSITY_2     : High color intensity filter.
    """
    NONE             = "none"
    BLACK_AND_WHITE  = "bw"
    COLOR            = "color"
    COLOR_POP        = "color_pop"
    COLOR_TWIST      = "color_twist"
    CONTRAST         = "contrast"
    INTENSITY_1      = "intensity_1"
    INTENSITY_2      = "intensity_2"
   #SATURATION_NOISE = "saturation_noise"


class BasicImageFilters(io.ComfyNode):
    xTITLE         = "Basic Image Filters"
    xDESCRIPTION   = (
        "Provides a set of image processing filters for post-generation color "
        "correction and tone adjustment. Supports up to three simultaneous "
        "filtering layers with individual intensity scaling, along with an "
        "optional automatic contrast normalization pass."
    )
    xCATEGORY      = ""
    xCOMFY_NODE_ID = ""
    xDEPRECATED    = False

    #__ INPUT / OUTPUT ____________________________________
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            display_name  = cls.xTITLE,
            description   = cls.xDESCRIPTION,
            category      = cls.xCATEGORY,
            node_id       = cls.xCOMFY_NODE_ID,
            is_deprecated = cls.xDEPRECATED,
            search_aliases=["decode", "decode latent", "latent to image", "render latent"],
            inputs=[
                io.Image.Input    ("images"),
                io.Boolean.Input  ("enable_auto_contrast",
                                   tooltip="When enabled, automatically adjusts the dynamic range of the "
                                           "image to fix washed-out tones and optimize contrast boundaries."
                                  ),
                io.Boolean.Input  ("enable_filters",
                                   tooltip="When enabled, applies the color correction filters configured below. "
                                           "If disabled, all filter selections and adjustments are bypassed."
                                  ),

                zp.Separator.Input("divider1", mode="divider"),#===================================

                io.Combo.Input    ("filter_1",
                                   options=cls.filters(),
                                   tooltip="The first color correction filter to apply to the resulting image."
                                  ),
                io.Float.Input    ("filter_1_adjust",
                                   default=0.0, min=-0.5, max=0.5, step=0.1, round=0.1,
                                   tooltip="The fine-tuning control for the filter effect. "
                                           "Range is -0.5 to 0.5, with 0.0 as the ideal baseline balance. "
                                  ),

                zp.Separator.Input("divider2", mode="spacer"),#====================================

                io.Combo.Input    ("filter_2",
                                   options=cls.filters(),
                                   tooltip="The second color correction filter to apply to the resulting image."
                                  ),
                io.Float.Input    ("filter_2_adjust",
                                   default=0.0, min=-0.5, max=0.5, step=0.1, round=0.1,
                                   tooltip="The fine-tuning control for the filter effect. "
                                           "Range is -0.5 to 0.5, with 0.0 as the ideal baseline balance. "
                                  ),

                zp.Separator.Input("divider3", mode="spacer"),#====================================

                io.Combo.Input    ("filter_3",
                                   options=cls.filters(),
                                   tooltip="The third color correction filter to apply to the resulting image."
                                  ),
                io.Float.Input    ("filter_3_adjust",
                                   default=0.0, min=-0.5, max=0.5, step=0.1, round=0.1,
                                   tooltip="The fine-tuning control for the filter effect. "
                                           "Range is -0.5 to 0.5, with 0.0 as the ideal baseline balance. "
                                  ),
            ],
            outputs=[
                io.Image.Output(tooltip="The final filtered image."),
            ],
        )

    #__ FUNCTION __________________________________________
    @classmethod
    def execute(cls,
                images: torch.Tensor,
                enable_auto_contrast: bool,
                enable_filters      : bool,
                filter_1            : str,
                filter_1_adjust     : float,
                filter_2            : str,
                filter_2_adjust     : float,
                filter_3            : str,
                filter_3_adjust     : float,
                divider1: Any = None,
                divider2: Any = None,
                divider3: Any = None,
                ):
        # round float values to 1 decimal place
        filter_1_adjust = round(filter_1_adjust, 1)
        filter_2_adjust = round(filter_2_adjust, 1)
        filter_3_adjust = round(filter_3_adjust, 1)

        # convert images to shape [B, C, H, W] compatible with torchvision & kornia
        images = images.permute(0, 3, 1, 2)
        color_space = 'rgb'

        # restrict values to [0,1] via contrast adjustment or direct clamping
        if enable_auto_contrast:
            images = stretch_histogram(images)
        else:
            images = torch.clamp(images, 0.0, 1.0)

        # apply filters, which may modify the color space of the images
        if enable_filters:
            images, color_space = cls.apply_filter_to_images(images, color_space, filter_1, filter_1_adjust)
            images, color_space = cls.apply_filter_to_images(images, color_space, filter_2, filter_2_adjust)
            images, color_space = cls.apply_filter_to_images(images, color_space, filter_3, filter_3_adjust)

        # convert back to RGB color space
        # and return to [B, H, W, C] shape, which is compatible with ComfyUI
        images = convert_to_rgb(images, color_space).permute(0, 2, 3, 1).contiguous()
        return (images, )


    #__ internal functions ________________________________

    @staticmethod
    def apply_filter_to_images(images     : torch.Tensor,
                               color_space: str,
                               filter     : Filter | str,
                               value      : float
                               ) -> tuple[torch.Tensor, str]:
        if isinstance(filter,str):
            filter = Filter(filter)

        if filter == Filter.NONE:
            return images, color_space

        elif filter == Filter.BLACK_AND_WHITE:
            if value>=0:
                images = adjust_hsv_components(images,
                                               saturation_factor      = 0,
                                               brightness_scurve_factor = 1-(value*1.33),
                                               input_color_space      = color_space)
            else:
                images = adjust_hsv_components(images,
                                               saturation_factor      = -value,
                                               input_color_space      = color_space)

            return images, 'hsv'

        elif filter == Filter.COLOR:
            images = adjust_hsv_components(images,
                                           saturation_factor        = 1+(value*1.25),
                                           input_color_space        = color_space)
            return images, 'hsv'

        elif filter == Filter.COLOR_POP:
            images = adjust_hsv_components(images,
                                           saturation_stretch = min(abs(value*3.33), 1.0),
                                           saturation_gamma         = 1+(value*1.75),
                                           #contrast_scurve_factor   = (value-1)*0.2 + 1,
                                           input_color_space        = color_space)
            return images, 'hsv'

        elif filter == Filter.COLOR_TWIST:
            images = adjust_hsv_components(images,
                                           hue_twist_factor  = value*(0.75*pi),
                                           input_color_space = color_space)
            return images, 'hsv'

        elif filter == Filter.CONTRAST:
            images = adjust_hsv_components(images,
                                           brightness_scurve_factor = 1+(value*0.66),
                                           input_color_space = color_space)
            return images, 'hsv'

        elif filter == Filter.INTENSITY_1:
            contrast_factor = 1 + (value*1.0 if value<0 else value*1.0)
            images = adjust_hsv_components(images,
                                           saturation_target      = 0.35,
                                           brightness_scurve_factor = contrast_factor,
                                           input_color_space      = color_space)
            return images, 'hsv'

        elif filter == Filter.INTENSITY_2:
            contrast_factor = 1 + (value*1.0 if value<0 else value*1.0)
            images = adjust_hsv_components(images,
                                           saturation_target      = 0.40,
                                           brightness_scurve_factor = contrast_factor,
                                           input_color_space      = color_space)
            return images, 'hsv'

        raise ValueError(f'Invalid filter: {filter.value}')


    @staticmethod
    @cache
    def filters() -> list[str]:
        return [f.value for f in Filter]
