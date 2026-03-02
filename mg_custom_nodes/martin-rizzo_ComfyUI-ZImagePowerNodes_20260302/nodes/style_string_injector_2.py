"""
File    : style_string_injector_2.py
Purpose : Inject a visual style into a prompt. (version 2)
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 22, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
         ComfyUI nodes designed specifically for the "Z-Image" model.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

 ComfyUI V3 Schema oficial documentation:
 - https://docs.comfy.org/custom-nodes/v3_migration

"""
from functools                   import cache
from comfy_api.latest            import io
from .lib.style_group            import StyleGroup
from .lib.style_helpers          import get_style_names, get_style_template
from ..styles.predefined_styles  import PREDEFINED_STYLE_GROUPS


class StyleStringInjector2(io.ComfyNode):
    xTITLE         = "Style String Injector"
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
                "Injects a style to your prompt. This node takes a text prompt and modifies it "
                "according to the selected style"
            ),
            inputs=[
                io.Combo.Input ("style", options=cls.style_names(),
                                tooltip="The visual style you want for your image.",
                               ),
                io.Custom("ZIPN_STYLE_GALLERY").Input("gallery",
                                                      tooltip="Open the style gallery to see all available styles."
                                                     ),
                io.String.Input("string", multiline=True, dynamic_prompts=True, force_input=True,
                                tooltip="The prompt to modify.",
                               ),
            ],
            outputs=[
                io.String.Output(tooltip="The prompt after applying the selected style."),
            ]
        )

    #__ FUNCTION __________________________________________
    @classmethod
    def execute(cls, style: str, string: str, **kwargs) -> io.NodeOutput:
        template = cls.predefined_style_template(style)

        # apply the style template to the prompt
        prompt = string
        if template:
            prompt = StyleGroup.apply_style_template(prompt, template, spicy_impact_booster=False)

        return io.NodeOutput( prompt )


    #__ VALIDATION ________________________________________
    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        return True



    #__ internal functions ________________________________

    @staticmethod
    @cache
    def style_names() -> list[str]:
        """Returns all available style names."""
        return get_style_names( PREDEFINED_STYLE_GROUPS, quoted = True )


    @staticmethod
    def predefined_style_template(name: str) -> str:
        """Returns the predefined template for the given style."""
        return get_style_template( PREDEFINED_STYLE_GROUPS, name, default="" )

