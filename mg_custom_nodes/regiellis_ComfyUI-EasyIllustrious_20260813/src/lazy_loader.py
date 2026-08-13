import functools
from typing import Dict, List, Optional, Any
from pathlib import Path


class LazyDataLoader:
    """Lazy loading and caching for large character/artist datasets"""

    def __init__(self):
        self._artists: Optional[Dict] = None
        self._characters: Optional[Dict] = None
        self._e621_artists: Optional[Dict] = None
        self._e621_characters: Optional[Dict] = None
        self._cache: Dict[str, List[str]] = {}

    @property
    def artists(self) -> Dict:
        if self._artists is None:
            from .artists import ARTISTS

            self._artists = ARTISTS
        return self._artists

    @property
    def characters(self) -> Dict:
        if self._characters is None:
            from .characters import CHARACTERS

            self._characters = CHARACTERS
        return self._characters

    @property
    def e621_artists(self) -> Dict:
        if self._e621_artists is None:
            from .e621_artists import E621_ARTISTS

            self._e621_artists = E621_ARTISTS
        return self._e621_artists

    @property
    def e621_characters(self) -> Dict:
        if self._e621_characters is None:
            from .e621_characters import E621_CHARACTERS

            self._e621_characters = E621_CHARACTERS
        return self._e621_characters

    @functools.lru_cache(maxsize=8)
    def get_list(self, data_type: str) -> List[str]:
        """Get cached list of keys for a data type"""
        if data_type == "artists":
            return ["-"] + list(self.artists.keys())
        elif data_type == "characters":
            return ["-"] + list(self.characters.keys())
        elif data_type == "e621_artists":
            return ["-"] + list(self.e621_artists.keys())
        elif data_type == "e621_characters":
            return ["-"] + list(self.e621_characters.keys())
        else:
            return ["-"]

    def get_paginated_list(
        self, data_type: str, page: int = 0, page_size: int = 100
    ) -> Dict[str, Any]:
        """Get paginated list for frontend loading"""
        full_list = self.get_list(data_type)
        start = page * page_size
        end = start + page_size

        return {
            "items": full_list[start:end],
            "total": len(full_list),
            "page": page,
            "page_size": page_size,
            "has_more": end < len(full_list),
        }

    def search(self, data_type: str, query: str, limit: int = 50) -> List[str]:
        """Search for matching items"""
        if not query:
            return self.get_list(data_type)[:limit]

        query_lower = query.lower()
        full_list = self.get_list(data_type)

        # Exact matches first
        exact_matches = [item for item in full_list if item.lower() == query_lower]

        # Prefix matches
        prefix_matches = [
            item
            for item in full_list
            if item.lower().startswith(query_lower) and item not in exact_matches
        ]

        # Contains matches
        contains_matches = [
            item
            for item in full_list
            if query_lower in item.lower()
            and item not in exact_matches
            and item not in prefix_matches
        ]

        results = exact_matches + prefix_matches + contains_matches
        return results[:limit]

    def get_data(self, data_type: str, key: str) -> Optional[Dict]:
        """Get full data for a specific key"""
        if data_type == "artists":
            return self.artists.get(key)
        elif data_type == "characters":
            return self.characters.get(key)
        elif data_type == "e621_artists":
            return self.e621_artists.get(key)
        elif data_type == "e621_characters":
            return self.e621_characters.get(key)

        # Return default value instead of None for better error handling
        return {}


# Global instance
data_loader = LazyDataLoader()
