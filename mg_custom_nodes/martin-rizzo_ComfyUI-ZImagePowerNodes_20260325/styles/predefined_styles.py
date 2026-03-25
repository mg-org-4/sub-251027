"""
File    : styles/styles/predefined_styles.py
Purpose : Last version of all predefined styles grouped by category.
          (this file acts as a redirection to the file containing the last version)
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Jan 25, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
         ComfyUI nodes designed specifically for the "Z-Image" model.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from ..nodes.lib.style_group import StyleGroup
from .predefined_styles_1_0 import PREDEFINED_STYLE_GROUPS_1_0
from .predefined_styles_0_9 import PREDEFINED_STYLE_GROUPS_0_9
from .predefined_styles_0_8 import PREDEFINED_STYLE_GROUPS_0_8

# Global variable that stores predefined style groups for the current version.
# This acts as an alias allowing for easier version updates.
PREDEFINED_STYLE_GROUPS : list[StyleGroup] = PREDEFINED_STYLE_GROUPS_1_0

# Global variable that stores all style groups by version.
STYLE_GROUPS_BY_VERSION = {
    "last": PREDEFINED_STYLE_GROUPS,
    "1.0" : PREDEFINED_STYLE_GROUPS_1_0,
    "0.9" : PREDEFINED_STYLE_GROUPS_0_9,
    "0.8" : PREDEFINED_STYLE_GROUPS_0_8
}


def number_of_predefined_styles() -> int:
    """
    Returns the total number of predefined styles, excluding custom styles.
    """
    count = 0
    for style_group in PREDEFINED_STYLE_GROUPS:
        if style_group.category not in ("custom",""):
            count += len( style_group.get_names() )
    return count
