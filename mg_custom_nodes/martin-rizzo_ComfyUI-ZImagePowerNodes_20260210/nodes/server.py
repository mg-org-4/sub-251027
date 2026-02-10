"""
File    : server.py
Purpose : Provides server routes to be used by the nodes of the project.
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 22, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
         ComfyUI nodes designed specifically for the "Z-Image" model.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

 ComfyUI Server Routes oficial documentation:
 - https://docs.comfy.org/development/comfyui-server/comms_routes

"""
from functools                   import cache
from server                      import PromptServer
from aiohttp                     import web
from ..styles.predefined_styles  import PREDEFINED_STYLE_GROUPS
routes = PromptServer.instance.routes


@cache
def _get_last_version_styles() -> list[list[str]]:
    """
    Retrieves all styles from the last version of this project.
    Returns:
        list[list[str]]: A list containing lists of style data. Each inner list contains:
                         [0] name        (str): The name of the style.
                         [1] category    (str): The category to which the style belongs.
                         [2] description (str): Description of the style (currently empty).
                         [3] tags        (str): Tags associated with the style, comma-separated
                         [4] template    (str): Template associated with the style.
                         [5] thumbnail   (str): URL for the style's thumbnail (currently empty).
    """
    LAST_VERSION_STYLE_GROUPS = PREDEFINED_STYLE_GROUPS
    last_version_styles = []
    for style_group in LAST_VERSION_STYLE_GROUPS:
        category = style_group.category
        names    = style_group.get_names()
        for name in names:
            thumbnail   = ""
            description = ""
            tags        = ""
            template    = style_group.get_style_template(name)
            style_data : list[str] = [
                name,         # 0: name
                category,     # 1: category
                description,  # 2: description
                tags,         # 3: tags (comma-separated)
                template,     # 4: template
                thumbnail,    # 5: thumbnail url (front-end generated)
            ]
            last_version_styles.append(style_data)
    return last_version_styles


#============================== SERVER ROUTES ==============================#

@routes.get("/zi_power/styles/last_version")
async def get_last_version_styles(_):
    return web.json_response( _get_last_version_styles() )


