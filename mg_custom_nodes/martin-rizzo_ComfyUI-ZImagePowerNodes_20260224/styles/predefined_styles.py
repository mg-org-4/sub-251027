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
from .predefined_styles_0_10beta import PREDEFINED_STYLE_GROUPS_0_10beta

# Global variable that stores predefined style groups.
# This acts as an alias allowing for easier version updates.
PREDEFINED_STYLE_GROUPS = PREDEFINED_STYLE_GROUPS_0_10beta



def number_of_predefined_styles() -> int:
    """
    Returns the total number of predefined styles, excluding custom styles.
    """
    count = 0
    for style_group in PREDEFINED_STYLE_GROUPS:
        if style_group.category not in ("custom",""):
            count += len( style_group.get_names() )
    return count
