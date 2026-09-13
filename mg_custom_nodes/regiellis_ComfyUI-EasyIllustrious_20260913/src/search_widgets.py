"""Search-only widget implementations for Illustrious"""

from .quick_loader import get_search_placeholder


def create_search_input(data_type: str, field_name: str, tooltip: str = "") -> tuple:
    """Create a search input widget configuration"""
    placeholder = get_search_placeholder(data_type)

    return (
        "STRING",
        {
            "default": "",
            "tooltip": tooltip or f"Search {data_type} (start typing to find items)",
            "placeholder": placeholder,
            "multiline": False,
        },
    )


# Widget configurations for search inputs
SEARCH_WIDGETS = {
    "Character": create_search_input(
        "characters", "Character", "Search Danbooru characters"
    ),
    "Artist": create_search_input("artists", "Artist", "Search Danbooru artists"),
    "E621 Character": create_search_input(
        "e621_characters", "E621 Character", "Search E621 characters"
    ),
    "E621 Artist": create_search_input(
        "e621_artists", "E621 Artist", "Search E621 artists"
    ),
}
