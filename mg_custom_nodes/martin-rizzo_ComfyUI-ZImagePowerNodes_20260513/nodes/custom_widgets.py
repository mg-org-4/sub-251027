"""
File    : custom_widgets.py
Purpose : Custom ComfyUI widgets implemented specifically for this project.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : May 11, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
         ComfyUI nodes designed specifically for the "Z-Image" model.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

  The V3 schema documentation can be found here:
    - https://docs.comfy.org/custom-nodes/v3_migration

  The custom types section in the V3 schema documentation can be found here:
   - https://docs.comfy.org/custom-nodes/v3_migration#custom-types

"""
from comfy_api.latest import io
"""
A custom ComfyUI widget that can act as a spacer or divider.

Args:
    id: PEPE!
    mode (str): The visual style of the separator:<br/> 'spacer', 'divider', 'dotted', 'bold'.
                            - Segunda linea.
                            - Tercera linea.
    color: The color of the separator. Accepts a hexadecimal color string. Defaults to '#555555'.

"""



@io.comfytype(io_type="ZIPN_SEPARATOR")
class Separator:
    #Type = str

    class Input(io.Input):

        def __init__(self,
                     id       : str,
                     mode     : str | None = None,
                     color    : str | None = None,
                     height   : int | None = None,
                     thickness: int | None = None,
                     **kwargs
                     ):
            """
            <hr>A separator widget.

            Args:
                id (str):               A unique identifier for the input component.
                mode (str, optional):   The visual style of the separator `"spacer"`, `"divider"`, `"dotted"`, `"bold"`.
                                        Defaults to 'spacer'.
                color (str, optional):  The color of the separator. Accepts a hexadecimal color string.
                                        Defaults to '#555555'.
                height (int, optional): The height of the separator. Defaults to `20`.
                thickness (int, optional): The thickness of the separator. Defaults to `2`.
            """
            ALLOWED_MODES = ("spacer", "divider", "dotted", "bold")
            extra_dict = {}

            if mode is not None:
                if mode not in ALLOWED_MODES:
                    raise ValueError(f"Invalid mode: {mode}. Must be one of {ALLOWED_MODES}")
                extra_dict["mode"] = mode

            if color is not None:
                extra_dict["color"] = color

            if height is not None:
                extra_dict["height"] = height

            super().__init__(id, extra_dict=extra_dict, **kwargs)



    class Output(io.Output):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)


