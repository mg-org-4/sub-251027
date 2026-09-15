"""API handler for Illustrious list loading endpoints"""

import json
from typing import Dict, List, Any
from .lazy_loader import data_loader


class IllustriousAPI:
    """Handles API endpoints for dynamic list loading"""

    @staticmethod
    def get_paginated_list(
        data_type: str, search_query: str = "", page: int = 0, page_size: int = 200
    ) -> Dict[str, Any]:
        """Get paginated list with optional search filtering"""
        try:
            if search_query:
                # Use search functionality
                full_results = data_loader.search(
                    data_type, search_query, limit=10000
                )  # Get all search results
            else:
                # Get full list
                full_results = data_loader.get_list(data_type)

            # Calculate pagination
            total_items = len(full_results)
            start_idx = page * page_size
            end_idx = min(start_idx + page_size, total_items)

            # Get page data
            page_data = full_results[start_idx:end_idx]

            return {
                "success": True,
                "data": page_data,
                "total": total_items,
                "page": page,
                "page_size": page_size,
                "returned": len(page_data),
                "has_more": end_idx < total_items,
                "total_pages": (total_items + page_size - 1) // page_size,
                "data_type": data_type,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": ["-"],
                "total": 0,
                "page": 0,
                "page_size": page_size,
                "returned": 0,
                "has_more": False,
                "total_pages": 0,
                "data_type": data_type,
            }

    @staticmethod
    def search_lists(data_type: str, query: str, limit: int = 50) -> Dict[str, Any]:
        """Search within lists with smart matching"""
        try:
            results = data_loader.search(data_type, query, limit)

            return {
                "success": True,
                "data": results,
                "query": query,
                "returned": len(results),
                "data_type": data_type,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": ["-"],
                "query": query,
                "returned": 0,
                "data_type": data_type,
            }

    @staticmethod
    def get_popular_items(data_type: str, limit: int = 100) -> Dict[str, Any]:
        """Get most popular items from a data type"""
        try:
            full_list = data_loader.get_list(data_type)
            # Take first N items (they should be sorted by popularity)
            popular = full_list[:limit] if len(full_list) > limit else full_list

            return {
                "success": True,
                "data": popular,
                "total": len(full_list),
                "returned": len(popular),
                "data_type": data_type,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": ["-"],
                "total": 0,
                "returned": 0,
                "data_type": data_type,
            }
