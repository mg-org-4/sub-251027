"""
File    : my_top_10_styles_editor.py
Purpose : Node that lets users edit their personal top-10 visual styles.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 31, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
         ComfyUI nodes designed specifically for the "Z-Image" model.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

The V3 schema documentation can be found here:
 - https://docs.comfy.org/custom-nodes/v3_migration

"""
from functools                  import cache
from comfy_api.latest           import io
from .styles.predefined_styles  import PREDEFINED_STYLE_GROUPS
from .lib.style_helpers         import normalize_style_name



class MyTop10StylesEditor(io.ComfyNode):
    xTITLE         = "My Top-10 Style Editor"
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
                "Allows you to create a personalized list of your top 10 visual styles, "
                "selecting from all available options to build your ideal collection. "
                "This node is designed to work alongside 'My Top-10 Styles' to define "
                "and utilize your preferred styles."
            ),
            inputs=[
                io.Combo.Input( "style_1" , options=cls.all_style_names(), ),
                io.Combo.Input( "style_2" , options=cls.all_style_names(), ),
                io.Combo.Input( "style_3" , options=cls.all_style_names(), ),
                io.Combo.Input( "style_4" , options=cls.all_style_names(), ),
                io.Combo.Input( "style_5" , options=cls.all_style_names(), ),
                io.Combo.Input( "style_6" , options=cls.all_style_names(), ),
                io.Combo.Input( "style_7" , options=cls.all_style_names(), ),
                io.Combo.Input( "style_8" , options=cls.all_style_names(), ),
                io.Combo.Input( "style_9" , options=cls.all_style_names(), ),
                io.Combo.Input( "style_10", options=cls.all_style_names(), ),
            ],
            outputs=[
                io.Custom("TOP_STYLES").Output("TOP_STYLES",
                                               tooltip="Comma-separated string of 10 top styles.",
                                               ),
            ]
        )

    #__ FUNCTION __________________________________________
    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:

        # collect the top styles selected by the user
        top_styles = []
        for i in range(0, 100):
            input_id = f"style_{i+1}"
            if input_id not in kwargs:
                break
            top_styles.append( normalize_style_name(kwargs[input_id]) )

        # output the top styles as a array of strings
        return io.NodeOutput( top_styles )


    #__ VALIDATION ________________________________________
    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        #if kwargs["category"] not in cls.category_names():
        #    return f"The category name '{kwargs['style_type']}' is invalid. May be the node is from an older version."
        return True


    #__ internal functions ________________________________

    @staticmethod
    @cache
    def all_style_names() -> list[str]:
        """Returns all available style names."""
        names = ["none"]
        for style_group in PREDEFINED_STYLE_GROUPS:
            names.extend( style_group.get_names(quoted=True) )
        return names

