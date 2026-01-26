"""
File    : style_prompt_encoder.py
Purpose : Node to get conditioning embeddings from a given style + prompt.
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
from comfy_api.latest           import io
from .styles.style_group        import StyleGroup, apply_style_to_prompt
from .styles.predefined_styles  import PREDEFINED_STYLE_GROUPS


class StylePromptEncoder(io.ComfyNode):
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
                io.Combo.Input ("category", options=cls.category_names(), default=cls.default_category_name(),
                                tooltip="The category of styles you want to select from.",
                               ),
                io.Combo.Input ("style", options=cls.style_names(), default=cls.default_style_name(),
                                tooltip="The style you want for your image.",
                               ),
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
                category      : str,
                style: str,
                text          : str,
                customization : str = ""
                ) -> io.NodeOutput:
        style_to_apply = None
        prompt         = text
        custom_styles  = StyleGroup.from_string(customization)

        if isinstance(style, str) and style != "none":
            # first search inside the custom styles that the user has defined,
            # if not found, search inside the predefined styles
            style_to_apply = custom_styles.get_style(style)
            if not style_to_apply:
                style_to_apply = cls.get_predefined_style(style)

        # if the style was found, apply it to the prompt
        if style_to_apply:
            prompt = apply_style_to_prompt(prompt, style_to_apply, spicy_impact_booster=False)

        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        tokens = clip.tokenize(prompt)
        return io.NodeOutput( clip.encode_from_tokens_scheduled(tokens), prompt )

    #__ VALIDATION ________________________________________
    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        if kwargs["category"] not in cls.category_names():
            return f"The category name '{kwargs['category']}' is invalid. May be the node is from an older version."
        return True



    #__ internal functions ________________________________

    @classmethod
    def category_names(cls) -> list[str]:
        """Returns all available category names."""
        return [ group.category for group in PREDEFINED_STYLE_GROUPS ]


    @classmethod
    def style_names(cls) -> list[str]:
        """Returns all available style names."""
        names = ["none"]
        for style_group in PREDEFINED_STYLE_GROUPS:
            names.extend( style_group.get_names(quoted=True) )
        return names


    @classmethod
    def default_category_name(cls) -> str:
        return PREDEFINED_STYLE_GROUPS[0].category


    @classmethod
    def default_style_name(cls) -> str:
        return PREDEFINED_STYLE_GROUPS[0].get_names(quoted=True)[0]


    @classmethod
    def get_predefined_style(cls, style_name: str) -> str:
        """Returns a predefined style content by its name, searching inside all category groups."""
        for style_group in PREDEFINED_STYLE_GROUPS:
            style = style_group.get_style(style_name)
            if style:
                return style
        return ""

