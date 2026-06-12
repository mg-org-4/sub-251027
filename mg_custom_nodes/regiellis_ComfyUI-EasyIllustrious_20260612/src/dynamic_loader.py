"""Dynamic loader helper for dynamic character/artist lists.

Production behavior:
- Emits a UI event via PromptServer when a "load_more..." sentinel is selected
- Provides either a small, popular list for quick-pick or the full list
- Validates and auto-corrects selections with fuzzy search fallback
"""

from typing import Tuple, List
from .lazy_loader import data_loader
from .quick_loader import get_popular_suggestions

try:
    # ComfyUI server for UI events (available at runtime in ComfyUI)
    from server import PromptServer  # type: ignore
except Exception:  # pragma: no cover - allow offline imports
    PromptServer = None  # type: ignore


class DynamicListHandler:
    """Handles dynamic loading of full lists when needed"""

    @staticmethod
    def process_selection(value: str, data_type: str) -> str:
        """Process a selection and handle 'load_more...' by signaling the UI.

        Returns "-" after signaling, so UI can fetch and replace with the full list.
        """
        if value == "load_more...":
            try:
                if PromptServer and hasattr(PromptServer, "instance"):
                    # Notify UI listeners to open the full list modal for this type
                    PromptServer.instance.send_sync(
                        "illustrious.load_more",
                        {"data_type": data_type},
                    )
            except Exception:
                # Non-fatal if server signaling is unavailable
                pass
            return "-"
        return value

    @staticmethod
    def get_display_list(data_type: str, use_full: bool = False) -> List[str]:
        """Get a display list for dropdowns.

        - When use_full is True: returns the entire key list.
        - Otherwise: a compact, curated list with a load-more sentinel.
        """
        if use_full:
            return data_loader.get_list(data_type)

        quick = ["-"] + list(get_popular_suggestions(data_type))
        # Ensure the sentinel is present to allow full expansion from UI
        if "load_more..." not in quick:
            quick.append("load_more...")
        return quick

    @staticmethod
    def validate_selection(value: str, data_type: str) -> Tuple[bool, str]:
        """Validate if a selection exists in the full dataset"""
        if value in ["-", "load_more...", "custom_character", "custom_artist"]:
            return True, value

        # Check if value exists in the full dataset
        full_data = getattr(data_loader, data_type, None)
        if callable(full_data):  # guard: properties are attributes, not methods
            full_data = None
        if isinstance(full_data, dict) and value in full_data:
            return True, value

        # Try to find a close match
        search_results = data_loader.search(data_type, value, limit=1)
        if search_results:
            return True, search_results[0]

        return False, "-"
