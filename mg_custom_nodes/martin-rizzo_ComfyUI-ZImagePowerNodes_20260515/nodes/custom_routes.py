"""
File    : custom_routes.py
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
 - https://docs.comfy.org/development/comfyui-server/comms_routes#custom-routes

"""
import os
import re
from aiohttp                     import web
from functools                   import cache
from server                      import PromptServer
from aiohttp                     import web
from .core.style                 import StyleSet
from .core.predefined_styles     import PREDEFINED_STYLES
from .core.helpers               import get_project_root
routes = PromptServer.instance.routes



def _style_data(styles: StyleSet, add_none=False):
    """
    Generates a list of style data for all styles in the given set.

    Args:
        styles: The `StyleSet` object containing the styles to be processed.

    Returns:
        list[list[str]]: A nested list where each inner list contains the style information:
            [0] name        (str): The name of the style.
            [1] category    (str): The category to which the style belongs.
            [2] description (str): Description of the style.
            [3] tags        (str): Tags associated with the style, comma-separated
            [4] thumbnail   (str): The filename of the thumbnail image (e.g. "casual_photo.jpg")
    """
    result = []

    # add "none" option if requested
    if add_none:
        result.append(["none", "", "", "", "none.jpg"])

    for style in styles:
        style_data : list[str] = [
            style.name,                  # 0: name
            style.category,              # 1: category
            style.description,           # 2: description
            style.comma_separated_tags,  # 3: tags (comma-separated)
            f"{style.slug}.jpg",         # 4: thumbnail filename (e.g. "casual_photo.jpg")
        ]
        result.append(style_data)
    return result


@cache
def _last_version_style_data() -> list[list[str]]:
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
    return _style_data( PREDEFINED_STYLES.by_version("1.0.0"), add_none=True )


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
    return web.json_response( _last_version_style_data() )


@routes.get("/zi_power/styles/by_version")
async def get_styles_by_version(request: web.Request) -> web.StreamResponse:

    # try to get the version from query parameters
    version = request.query.get("v") or request.query.get("version")
    if not version:
        return web.Response(status=400)

    # try to get all styles from the version
    styles = PREDEFINED_STYLES.by_version(version)
    if not styles:
        return web.Response(status=400)

    return web.json_response( _style_data(styles) )



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
