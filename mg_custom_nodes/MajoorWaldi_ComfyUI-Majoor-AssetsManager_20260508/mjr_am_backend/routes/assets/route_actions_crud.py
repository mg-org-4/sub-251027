"""Compatibility facade for CRUD-oriented asset route actions."""

from __future__ import annotations

from .route_actions_batch_delete import handle_delete_assets
from .route_actions_delete import handle_delete_asset
from .route_actions_rename import handle_rename_asset

__all__ = [
    "handle_delete_asset",
    "handle_delete_assets",
    "handle_rename_asset",
]
