"""Migration v18 — rebuild ``asset_metadata_fts`` triggers to source the
``tags`` column from the normalized ``asset_tags`` + ``tags`` tables instead
of the legacy ``asset_metadata.tags`` JSON column.

After this migration the FTS index is **decoupled from the legacy JSON
column**, which is a prerequisite for eventually dropping
``asset_metadata.tags`` / ``asset_metadata.tags_text`` in a later
destructive migration.

Changes:

* ``asset_metadata_fts_insert`` / ``asset_metadata_fts_update`` are
  recreated so that the ``tags`` FTS column is filled with a space-joined
  ``group_concat`` of normalized tag names (computed via subquery on
  ``asset_tags`` JOIN ``tags``). The ``tags_text`` FTS column is filled
  with the same value for backward compatibility with existing MATCH
  queries that span all columns.
* Two new triggers ``asset_tags_fts_insert`` / ``asset_tags_fts_delete``
  keep the FTS row in sync when normalized tags change directly (e.g. via
  ``TagsRepository``) without an accompanying ``asset_metadata`` write.
* The full FTS index is rebuilt so existing rows reflect the new
  ``tags`` text source.

This migration is idempotent: triggers are dropped + recreated; the
reindex is a deterministic ``DELETE`` + ``INSERT … SELECT``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....shared import Result, get_logger
from .base import Migration

if TYPE_CHECKING:
    from ..sqlite_facade import Sqlite

logger = get_logger(__name__)


# Shared subquery: space-separated normalized tag names for one asset.
_TAGS_FROM_NORMALIZED = (
    "COALESCE("
    "(SELECT group_concat(t.name, ' ') "
    "FROM asset_tags at JOIN tags t ON t.id = at.tag_id "
    "WHERE at.asset_id = {asset_id_expr}), '')"
)


def _tags_expr(asset_id_expr: str) -> str:
    return _TAGS_FROM_NORMALIZED.format(asset_id_expr=asset_id_expr)


_DROP_LEGACY_TRIGGERS = """
DROP TRIGGER IF EXISTS asset_metadata_fts_insert;
DROP TRIGGER IF EXISTS asset_metadata_fts_update;
DROP TRIGGER IF EXISTS asset_tags_fts_insert;
DROP TRIGGER IF EXISTS asset_tags_fts_delete;
"""

# Note: the DELETE trigger on asset_metadata is unchanged from v17 and is
# left in place; we don't need to drop/recreate it.


def _create_triggers_sql() -> str:
    tags_new = _tags_expr("new.asset_id")
    tags_old = _tags_expr("old.asset_id")
    return f"""
    CREATE TRIGGER IF NOT EXISTS asset_metadata_fts_insert
    AFTER INSERT ON asset_metadata BEGIN
        INSERT INTO asset_metadata_fts(rowid, tags, tags_text, metadata_text)
        VALUES (
            new.asset_id,
            {tags_new},
            {tags_new},
            COALESCE(new.metadata_text, '')
        );
    END;

    CREATE TRIGGER IF NOT EXISTS asset_metadata_fts_update
    AFTER UPDATE ON asset_metadata BEGIN
        UPDATE asset_metadata_fts
        SET tags = {tags_new},
            tags_text = {tags_new},
            metadata_text = COALESCE(new.metadata_text, '')
        WHERE rowid = new.asset_id;
    END;

    CREATE TRIGGER IF NOT EXISTS asset_tags_fts_insert
    AFTER INSERT ON asset_tags
    WHEN EXISTS (SELECT 1 FROM asset_metadata_fts WHERE rowid = new.asset_id)
    BEGIN
        UPDATE asset_metadata_fts
        SET tags = {tags_new},
            tags_text = {tags_new}
        WHERE rowid = new.asset_id;
    END;

    CREATE TRIGGER IF NOT EXISTS asset_tags_fts_delete
    AFTER DELETE ON asset_tags
    WHEN EXISTS (SELECT 1 FROM asset_metadata_fts WHERE rowid = old.asset_id)
    BEGIN
        UPDATE asset_metadata_fts
        SET tags = {tags_old},
            tags_text = {tags_old}
        WHERE rowid = old.asset_id;
    END;
    """


def _reindex_sql() -> str:
    tags_m = _tags_expr("m.asset_id")
    return f"""
    DELETE FROM asset_metadata_fts;
    INSERT INTO asset_metadata_fts(rowid, tags, tags_text, metadata_text)
    SELECT m.asset_id,
           {tags_m},
           {tags_m},
           COALESCE(m.metadata_text, '')
    FROM asset_metadata m;
    """


class FtsFromNormalizedTagsMigration(Migration):
    """v18 — FTS ``tags`` column sourced from normalized tables."""

    version = 18
    name = "fts_from_normalized_tags"

    async def upgrade(self, db: Sqlite) -> Result[bool]:
        # Skip if the FTS table doesn't exist (legacy bootstrap order may
        # not have created it yet — repair_asset_metadata_fts will install
        # the same triggers via the updated schema_fts helpers).
        fts_exists = await db.aquery(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='asset_metadata_fts'"
        )
        if not fts_exists.ok:
            return Result.Err(
                "MIGRATION_QUERY_FAILED",
                f"v18 lookup failed: {fts_exists.error}",
            )
        if not fts_exists.data:
            logger.info("v18: asset_metadata_fts not present, skipping (will be installed lazily)")
            return Result.Ok(True)

        # 1. Drop legacy triggers.
        drop_result = await db.aexecutescript(_DROP_LEGACY_TRIGGERS)
        if not drop_result.ok:
            return Result.Err(
                "MIGRATION_DDL_FAILED",
                f"v18 drop triggers failed: {drop_result.error}",
            )

        # 2. Create new triggers.
        create_result = await db.aexecutescript(_create_triggers_sql())
        if not create_result.ok:
            return Result.Err(
                "MIGRATION_DDL_FAILED",
                f"v18 create triggers failed: {create_result.error}",
            )

        # 3. Rebuild FTS index from the normalized source so existing rows
        #    reflect the new ``tags`` derivation.
        reindex_result = await db.aexecutescript(_reindex_sql())
        if not reindex_result.ok:
            return Result.Err(
                "MIGRATION_REINDEX_FAILED",
                f"v18 reindex failed: {reindex_result.error}",
            )

        logger.info("v18: FTS triggers + index rebuilt from normalized tags")
        return Result.Ok(True)


MIGRATION = FtsFromNormalizedTagsMigration()
