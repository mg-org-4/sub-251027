"""
File    : style_prompt_encoder_2.py
Purpose : Node to get conditioning embeddings from a given style + prompt (version 2)
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 16, 2026
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
from .core.system                import logger
from .core.style_group           import Style, StyleGroup
from .core.style_helpers         import get_style_names, get_style_template
from ..styles.predefined_styles  import PREDEFINED_STYLE_GROUPS


class StylePromptEncoder2(io.ComfyNode):
    xTITLE         = "Style & Prompt Encoder"
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
                "Transforms a text prompt into embeddings, automatically adapting the prompt to match "
                "the selected style. This node takes a prompt, adjusts its visual style according to "
                "the chosen option, and then encodes it using the provided text encoder (clip) to "
                "generate an embedding that will guide image generation."
            ),
            inputs=[
                io.Clip.Input  ("clip",
                                tooltip="The CLIP model used for encoding the text."
                               ),
                io.String.Input("customization", optional=True, multiline=True, force_input=True,
                                tooltip=(
                                  'An optional multi-line string to customize existing styles. '
                                  'Each style definition must start with ">>>" followed by the style name, and then include '
                                  'its description on the next lines. The description should incorporate "{$@}" where the '
                                  'main text prompt will be inserted.'),
                               ),
                io.Combo.Input ("style", options=cls.style_names(),
                                tooltip="The style you want for your image.",
                               ),
                io.Custom("ZIPN_STYLE_GALLERY").Input("gallery",
                                                      tooltip="Open the style gallery to see all available styles."
                                                     ),
                io.Custom("ZIPN_SPACER").Input("spacer"),
                io.String.Input("text", multiline=True, dynamic_prompts=True,
                                tooltip="The prompt to encode.",
                               ),
            ],
            outputs=[
                io.Conditioning.Output(tooltip="The encoded text used to guide the image generation."),
                io.String.Output(tooltip="The prompt after applying the selected visual style."),
            ]
        )

    #__ FUNCTION __________________________________________
    @classmethod
    def execute(cls,
                clip,
                style         : str,
                text          : str,
                customization : str = "",
                **kwargs
                ) -> io.NodeOutput:
        custom_styles   = StyleGroup.from_string(customization)
        custom_template = custom_styles.get_style_template(style) if Style.is_valid_name(style) else None

        # if user did not define a custom template for this style, then use the predefined one
        template = custom_template if custom_template else cls.predefined_style_template(style)

        # apply the style template to the prompt
        prompt = text
        if template:
            prompt = StyleGroup.apply_style_template(prompt, template, spicy_impact_booster=False)

        # encode the prompt using the provided text encoder (clip)
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        tokens = clip.tokenize(prompt)
        return io.NodeOutput( clip.encode_from_tokens_scheduled(tokens), prompt )

    #__ VALIDATION ________________________________________
    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        return True



    #__ internal functions ________________________________

    @staticmethod
    @cache
    def style_names() -> list[str]:
        """Returns all available style names."""
        names = get_style_names( PREDEFINED_STYLE_GROUPS, quoted = True )
        return names


    @staticmethod
    def predefined_style_template(name: str) -> str:
        """Returns the predefined template for the given style."""
        return get_style_template( PREDEFINED_STYLE_GROUPS, name, default="" )

