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
import os
import re
from aiohttp                     import web
from functools                   import cache
from server                      import PromptServer
from aiohttp                     import web
from ..styles.predefined_styles  import PREDEFINED_STYLE_GROUPS, STYLE_GROUPS_BY_VERSION
from .lib.helpers                import get_project_root
from .lib.style_group            import StyleGroup
routes = PromptServer.instance.routes



def _style_list(style_groups: list[StyleGroup]):
    """
    Generates a list of style data for all styles in the given style groups.
    Args:
        style_groups (list[StyleGroup]): A list of StyleGroup objects.
    Returns:
        list[list[str]]: A nested list where each inner list contains the style information:
            [0] name        (str): The name of the style.
            [1] category    (str): The category to which the style belongs.
            [2] description (str): Description of the style.
            [3] tags        (str): Tags associated with the style, comma-separated
            [4] thumbnail   (str): The filename of the thumbnail image (e.g. "casual_photo.jpg")
    """
    styles = []
    for style_group in style_groups:
        category = style_group.category
        for name in style_group.get_names():
            style = style_group.get_style(name)
            if not style: continue
            thumbnail   = f"{style.slug}.jpg"
            description = style.description
            tags        = style.comma_separated_tags
            style_data : list[str] = [
                name,         # 0: name
                category,     # 1: category
                description,  # 2: description
                tags,         # 3: tags (comma-separated)
                thumbnail,    # 4: thumbnail filename (e.g. "casual_photo.jpg")
            ]
            styles.append(style_data)
    return styles


@cache
def _cached_last_version_styles() -> list[list[str]]:
    """
    Returns a cached list of style data for the last version of predefined styles.

    This function caches the result so that it is only computed once,
    even if called multiple times.
    Returns:
        list[list[str]]: A list containing lists of style data. Each inner list contains:
            [0] name        (str): The name of the style.
            [1] category    (str): The category to which the style belongs.
            [2] description (str): Description of the style (currently empty).
            [3] tags        (str): Tags associated with the style, comma-separated
            [4] thumbnail   (str): The filename of the thumbnail image (e.g. "casual_photo.jpg")
    """
    return _style_list( PREDEFINED_STYLE_GROUPS )


def _sanitize_filename(filename: str) -> str:
    """
    Sanitizes a given filename to ensure it is valid and secure for most file systems.

    This function removes any characters from the filename that are not
    alphanumeric, underscores, or hyphens. This sanitization process helps
    in making filenames safe from potential security threats and reduces the
    risk of errors due to unsupported characters in different operating systems.

    Args:
        filename (str): The original filename to be sanitized.
    Returns:
        A sanitized version of the input filename,
        suitable for most file systems and secure against common threats.
    """
    name, ext = os.path.splitext( os.path.basename(filename) )
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', name) #< remove any non-alphanumeric characters from the name
    safe_ext  = re.sub(r'[^a-zA-Z0-9]'  , '', ext ) #< remove any non-alphanumeric characters from the extension
    return f"{safe_name}.{safe_ext}" if safe_ext else safe_name


#============================== SERVER ROUTES ==============================#

@routes.get("/zi_power/styles/last_version")
async def get_last_version_styles(_) -> web.StreamResponse:
    return web.json_response( _cached_last_version_styles() )


@routes.get("/zi_power/styles/by_version")
async def get_styles_by_version(request: web.Request) -> web.StreamResponse:
    version = request.query.get("v") or request.query.get("version")
    if not version or version not in STYLE_GROUPS_BY_VERSION:
        return web.Response(status=400)
    return web.json_response( _style_list( STYLE_GROUPS_BY_VERSION[version] ) )


@routes.get("/zi_power/styles/samples")
async def get_style_sample(request: web.Request) -> web.StreamResponse:
    #
    # To request a style sample, you should use:
    #    "/zi_power/styles/samples?file=my_sample_image.jpg&t=${T}"
    #    where T could be "Math.floor(Date.now() / 3600000)" for cache busting
    #
    file = request.query.get("file")
    file = _sanitize_filename(file) if file else None
    if file == "none.jpg":
        file = "00-no-style.jpg"

    fullpath = (get_project_root() / "styles" / "samples" / file) if file else None
    if not fullpath or not os.path.isfile(fullpath):
        fullpath = get_project_root() / "styles" / "samples" / "00-sample-not-available.jpg"
    return web.FileResponse(fullpath)
