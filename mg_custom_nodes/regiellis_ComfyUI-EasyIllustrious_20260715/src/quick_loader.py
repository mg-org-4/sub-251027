"""Search-only approach - no more dropdown lists"""


def get_search_placeholder(data_type: str) -> str:
    """Get placeholder text for search inputs"""
    if data_type == "characters":
        return "Search characters... (e.g. hatsune_miku, saber, nezuko)"
    elif data_type == "artists":
        return "Search artists... (e.g. kantoku, wlop, artgerm)"
    elif data_type == "e621_characters":
        return "Search E621 characters... (e.g. lucario, umbreon)"
    elif data_type == "e621_artists":
        return "Search E621 artists... (e.g. wolfy-nail, dimwitdog)"
    return "Search..."


def get_popular_suggestions(data_type: str) -> list:
    """Get popular suggestions to show as examples"""
    if data_type == "characters":
        return [
            "hatsune_miku",
            "saber",
            "nezuko_kamado",
            "yor_forger",
            "ganyu_(genshin_impact)",
        ]
    elif data_type == "artists":
        return ["kantoku", "wlop", "artgerm", "sakimichan", "ilya_kuvshinov"]
    elif data_type == "e621_characters":
        return ["lucario", "umbreon", "gardevoir", "lopunny", "braixen"]
    elif data_type == "e621_artists":
        return ["wolfy-nail", "dimwitdog", "reit", "braeburned", "tokifuji"]
    return []
