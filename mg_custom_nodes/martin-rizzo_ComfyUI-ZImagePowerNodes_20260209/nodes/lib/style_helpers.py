"""
File    : style_helpers.py
Purpose : Functions to help with styles and style templates
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Feb 2, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
         ComfyUI nodes designed specifically for the "Z-Image" model.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
from typing       import Any
from .style_group import StyleGroup



def is_valid_style_name(name: str) -> bool:
    """
    Checks if a given style name is valid.
    Args:
        name (str): The potential style name to validate.
    Returns:
        bool: True if the name is valid, False otherwise.
    """
    # discard any name that is not a string
    if type(name) != str: return False

    # discard any empty string, dash or "none"
    normalized_name = name.strip().lower()
    return normalized_name not in ["", "-", "none"]



def normalize_style_name(name: str) -> str:
    """
    Normalizes a style name by removing quotes and validating it.
    Args:
        name (str): The potential style name to normalize.
    Returns:
        str: The normalized name if valid, otherwise an empty string.
    """
    if not is_valid_style_name(name):
        return ""

    # the name can be quoted with single or double quotes
    name = name.strip()
    if(  ( name.startswith("'") and name.endswith("'") )  or
         ( name.startswith('"') and name.endswith('"') )  ):
        name = name[1:-1]

    return name



def get_style_template(source: Any, name: str, default: str = "") -> str:
    """
    Retrieves the style template from a given source.

    This function operates similarly to `StyleGroup.get_style_template(..)`
    but allows different types of sources to be provided, including individual
    `StyleGroups` objects or collections of them (lists or tuples).

    Args:
        source  (Any): The source to search within. Can be an instance of `StyleGroup`,
                       or a collection like list/tuple containing StyleGroup instances.
        name    (str): The name of the style to retrieve.
        default (str): The default value if the style is not found. Defaults to an empty string.

    Returns:
        A string containing the style template or an empty string if the style is not found.
    """
    name = normalize_style_name(name)
    if not name:
        return default

    # if `source` is StyleGroup, get the template directly
    if type(source) == StyleGroup:
        return source.get_style_template(name)

    # if `source` is a list, search the template from all groups in it
    elif type(source) == list or type(source) == tuple:
        for style_group in source:
            template = get_style_template(style_group, name)
            if template:
                return template


    return default



def remove_style_from_text(text: str, name: str) -> str:
    """
    Removes a style definition from the given text.
    Args:
        text: The original text containing multiple style definitions.
        name: The name of the style to be removed.
    Returns:
        The modified text with the specified style removed.
    """
    name = normalize_style_name(name)
    if not name:
        return text

    lower_name = name.lower()
    lines      = text.splitlines()
    start, end = -1,-1

    # analyze line by line searching for the beginning and ending of
    # the template whose name matches with `name`
    for i, line in enumerate(lines):

        # search the beginning of style template definition
        if start<0:
            if not line.startswith(">>>"):
                continue
            style_name = line[3:].strip()
            if style_name.lower() == lower_name:
                start = i

        # search the end of style template definition
        # the markers ">::" and "{#" come from my previous project "Amazing Z-Image Workflow"
        else:
            if line.startswith((">>>", ">::", "{#")):
                end = i
                break

    # if the style is not found, return the original text
    if start<0:
        return text

    # if the end is not found, return everything until 'start'
    if end<0:
        return "\n".join( lines[:start] )

    # otherwise, remove the style from text ( `start`=inclusive | `end`=exclusive )
    return "\n".join( lines[:start] + lines[end:] )



def append_style_to_text(text: str, name: str, template: str) -> str:
    """
    Appends a style definition to the given text.
    Args:
        text    : The original text to which the style will be added.
        name    : The name of the new style.
        template: The content defining the new style.
    Returns:
        The modified text with the new style appended.
    """
    return f"{text}>>>{name}\n{template}\n\n\n"

