"""
Asset updater - handles rating and tag updates.
"""
from typing import Any

from ...adapters.core_assets import sync_user_metadata_by_asset_id
from ...adapters.db.sqlite import Sqlite
from ...data.repositories import TagsRepository
from ...shared import Result, get_logger

logger = get_logger(__name__)

MAX_TAG_LENGTH = 100


class AssetUpdater:
    """
    Handles asset metadata updates (ratings and tags).

    Provides methods to update asset ratings and normalized tags while
    maintaining data integrity.
    """

    def __init__(self, db: Sqlite, has_tags_text_column: bool = False):
        """
        Initialize asset updater.

        Args:
            db: Database adapter instance
            has_tags_text_column: Legacy compatibility parameter; ignored after v19.
        """
        self.db = db
        self._has_tags_text_column = False

    async def update_asset_rating(self, asset_id: int, rating: int) -> Result[dict[str, Any]]:
        """
        Update the rating for an asset.

        Args:
            asset_id: Asset ID
            rating: Rating value (0-5)

        Returns:
            Result with updated asset info
        """
        rating = max(0, min(5, int(rating or 0)))

        # Ensure asset exists
        check_result = await self.db.aquery(
            "SELECT id FROM assets WHERE id = ?",
            (asset_id,)
        )
        if not check_result.ok or not check_result.data or len(check_result.data) == 0:
            return Result.Err("NOT_FOUND", f"Asset not found: {asset_id}")

        # Update or insert asset_metadata (serialize per-asset to avoid race with enrichers)
        async with self.db.lock_for_asset(asset_id):
                result = await self.db.aexecute(
                    """
                    INSERT INTO asset_metadata (asset_id, rating)
                    SELECT ?, ?
                    WHERE EXISTS (SELECT 1 FROM assets WHERE id = ?)
                    ON CONFLICT(asset_id) DO UPDATE SET
                        rating = excluded.rating
                    WHERE EXISTS (SELECT 1 FROM assets WHERE id = excluded.asset_id)
                    """,
                    (asset_id, rating, asset_id)
                )

        if not result.ok:
            return Result.Err("UPDATE_FAILED", result.error or "Failed to update rating")

        await sync_user_metadata_by_asset_id(self.db, asset_id, rating=rating)
        return Result.Ok({"asset_id": asset_id, "rating": rating})

    async def update_asset_tags(self, asset_id: int, tags: list[str]) -> Result[dict[str, Any]]:
        """
        Update the tags for an asset.

        Args:
            asset_id: Asset ID
            tags: List of tag strings

        Returns:
            Result with updated asset info
        """
        # Ensure asset exists
        check_result = await self.db.aquery(
            "SELECT id FROM assets WHERE id = ?",
            (asset_id,)
        )
        if not check_result.ok or not check_result.data or len(check_result.data) == 0:
            return Result.Err("NOT_FOUND", f"Asset not found: {asset_id}")

        sanitized = self._sanitize_tags(tags)
        result = await self._write_asset_tags(asset_id, sanitized)

        if not result.ok:
            return Result.Err("UPDATE_FAILED", result.error or "Failed to update tags")

        await sync_user_metadata_by_asset_id(self.db, asset_id, tags=sanitized)
        return Result.Ok({"asset_id": asset_id, "tags": sanitized})

    def _sanitize_tags(self, tags: list[str]) -> list[str]:
        sanitized: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            if not isinstance(tag, str):
                continue
            cleaned = str(tag).strip()
            if not cleaned or len(cleaned) > MAX_TAG_LENGTH:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            sanitized.append(cleaned)
        return sanitized

    async def _write_asset_tags(self, asset_id: int, tags: list[str]) -> Result[Any]:
        async with self.db.lock_for_asset(asset_id):
            metadata_res = await self.db.aexecute(
                """
                INSERT INTO asset_metadata (asset_id, rating)
                SELECT ?, 0
                WHERE EXISTS (SELECT 1 FROM assets WHERE id = ?)
                ON CONFLICT(asset_id) DO NOTHING
                """,
                (asset_id, asset_id)
            )
            if not metadata_res.ok:
                return metadata_res
            return await TagsRepository(self.db).replace_all(asset_id, tags)

    async def get_all_tags(self) -> Result[list[str]]:
        """
        Get all unique tags from the database for autocomplete.

        Returns:
            Result with list of unique tags sorted alphabetically
        """
        result = await self.db.aquery(
            """
            SELECT name FROM tags ORDER BY name
            """
        )

        if not result.ok:
            return Result.Err("DB_ERROR", result.error or "Failed to read tags")

        return Result.Ok([str(row["name"]) for row in (result.data or [])])
