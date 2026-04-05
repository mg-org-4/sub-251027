import sqlite3

import pytest


def _create_partial_schema(db_path, current_schema_version) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', ?)",
            (str(current_schema_version),),
        )
        conn.execute(
            """
            CREATE TABLE assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                ext TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE asset_metadata (
                asset_id INTEGER PRIMARY KEY,
                rating INTEGER DEFAULT 0
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


async def _assert_table_contains_columns(db, table: str, expected_columns: list[str]) -> None:
    cols = await db.aquery(f"PRAGMA table_info('{table}')")
    assert cols.ok, cols.error
    col_names = {row["name"] for row in (cols.data or [])}
    for name in expected_columns:
        assert name in col_names


async def _assert_schema_healed_queries(db) -> None:
    assert (
        await db.aquery("SELECT source, root_id, enhanced_caption, content_hash, phash, hash_state FROM assets LIMIT 0")
    ).ok
    assert (
        await db.aquery(
            """
            SELECT tags, tags_text, metadata_text, metadata_raw, has_workflow, has_generation_data, metadata_quality
            FROM asset_metadata
            LIMIT 0
            """
        )
    ).ok


@pytest.mark.asyncio
async def test_schema_self_heals_missing_columns(tmp_path):
    from mjr_am_backend.adapters.db.schema import CURRENT_SCHEMA_VERSION, migrate_schema
    from mjr_am_backend.adapters.db.sqlite import Sqlite

    db_path = tmp_path / "partial_schema.db"
    _create_partial_schema(db_path, CURRENT_SCHEMA_VERSION)

    db = Sqlite(str(db_path), attach={"vec": str(tmp_path / "vectors.sqlite")})
    try:
        result = await migrate_schema(db)
        assert result.ok, result.error

        await _assert_table_contains_columns(
            db,
            "assets",
            [
                "source",
                "root_id",
                "subfolder",
                "width",
                "height",
                "duration",
                "enhanced_caption",
                "indexed_at",
                "content_hash",
                "phash",
                "hash_state",
            ],
        )
        await _assert_table_contains_columns(
            db,
            "asset_metadata",
            [
                "tags",
                "tags_text",
                "metadata_text",
                "workflow_hash",
                "has_workflow",
                "has_generation_data",
                "metadata_quality",
                "metadata_raw",
            ],
        )
        await _assert_schema_healed_queries(db)
    finally:
        try:
            await db.aclose()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_sqlite_query_missing_column_triggers_self_heal(tmp_path):
    from mjr_am_backend.adapters.db.schema import ensure_tables_exist
    from mjr_am_backend.adapters.db.sqlite import Sqlite

    db_path = tmp_path / "missing_hash_columns.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE assets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                ext TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime INTEGER NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    db = Sqlite(str(db_path), attach={"vec": str(tmp_path / "vectors.sqlite")})
    try:
        # Ensure required companion tables exist, but intentionally do not run migrate_schema().
        assert (await ensure_tables_exist(db)).ok
        res = await db.aquery("SELECT a.content_hash FROM assets a LIMIT 0")
        assert res.ok, res.error
    finally:
        try:
            await db.aclose()
        except Exception:
            pass
