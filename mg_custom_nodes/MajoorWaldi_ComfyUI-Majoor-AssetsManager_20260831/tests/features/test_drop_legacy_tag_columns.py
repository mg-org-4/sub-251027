"""Regression tests for v19 dropping legacy tag columns."""

from __future__ import annotations

import json

import pytest
from mjr_am_backend.adapters.db.migrations.m017_normalize_tags import (
    NormalizeTagsMigration,
)
from mjr_am_backend.adapters.db.migrations.m018_fts_from_normalized_tags import (
    FtsFromNormalizedTagsMigration,
)
from mjr_am_backend.adapters.db.migrations.m019_drop_legacy_tag_columns import (
    DropLegacyTagColumnsMigration,
)
from mjr_am_backend.adapters.db.schema import migrate_schema
from mjr_am_backend.adapters.db.sqlite import Sqlite
from mjr_am_backend.data.repositories import TagsRepository

pytestmark = pytest.mark.asyncio


async def _make_legacy_db(tmp_path) -> Sqlite:
    db = Sqlite(str(tmp_path / "v19.db"), attach={"vec": str(tmp_path / "vec.db")})
    mig = await migrate_schema(db)
    if not mig.ok:
        pytest.fail(f"schema migration failed: {mig.error}")
    for ddl in (
        "ALTER TABLE asset_metadata ADD COLUMN tags TEXT DEFAULT ''",
        "ALTER TABLE asset_metadata ADD COLUMN tags_text TEXT DEFAULT ''",
    ):
        res = await db.aexecute(ddl)
        if not res.ok:
            pytest.fail(f"legacy column setup failed: {res.error}")
    v17 = await NormalizeTagsMigration().upgrade(db)
    if not v17.ok:
        pytest.fail(f"v17 setup failed: {v17.error}")
    return db


async def _seed_asset(db) -> int:
    ins = await db.aexecute(
        "INSERT INTO assets(filepath, filename, subfolder, source, root_id, kind, ext, size, mtime) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("C:/v19/a.png", "a.png", "", "output", "output", "image", ".png", 10, 123),
    )
    if not ins.ok:
        pytest.fail(f"asset insert failed: {ins.error}")
    row = await db.aquery("SELECT id FROM assets WHERE filepath = ?", ("C:/v19/a.png",))
    return int(row.data[0]["id"])


async def test_v19_drops_legacy_tag_columns_and_keeps_fts(tmp_path) -> None:
    db = await _make_legacy_db(tmp_path)
    asset_id = await _seed_asset(db)

    legacy = await db.aexecute(
        "INSERT INTO asset_metadata(asset_id, rating, tags, tags_text, metadata_text) "
        "VALUES (?, 3, ?, ?, ?)",
        (asset_id, json.dumps(["alpha"], ensure_ascii=False), "alpha", "meta"),
    )
    if not legacy.ok:
        pytest.fail(f"legacy metadata insert failed: {legacy.error}")

    sync = await TagsRepository(db).sync_from_legacy_row(asset_id)
    if not sync.ok:
        pytest.fail(f"legacy backfill setup failed: {sync.error}")

    for migration in (FtsFromNormalizedTagsMigration(), DropLegacyTagColumnsMigration()):
        result = await migration.upgrade(db)
        if not result.ok:
            pytest.fail(f"migration v{migration.version} failed: {result.error}")

    cols = await db.aquery("PRAGMA table_info(asset_metadata)")
    if not cols.ok:
        pytest.fail(f"column lookup failed: {cols.error}")
    names = {str(row["name"]) for row in cols.data or []}
    if "tags" in names or "tags_text" in names:
        pytest.fail("v19 left legacy tag columns behind")

    legacy_query = await db.aquery("SELECT tags FROM asset_metadata WHERE asset_id = ?", (asset_id,))
    if legacy_query.ok:
        pytest.fail("legacy tags column remained queryable after v19")

    normalized = await TagsRepository(db).list_for_asset(asset_id)
    if not normalized.ok:
        pytest.fail(f"normalized tag lookup failed: {normalized.error}")
    if [tag.name for tag in normalized.data or []] != ["alpha"]:
        pytest.fail("normalized tags were not preserved before legacy drop")

    fts = await db.aquery(
        "SELECT tags, tags_text, metadata_text FROM asset_metadata_fts WHERE rowid = ?",
        (asset_id,),
    )
    if not fts.ok or not fts.data:
        pytest.fail(f"FTS row missing after v19: {fts.error}")
    row = fts.data[0]
    if row.get("tags") != "alpha" or row.get("tags_text") != "alpha":
        pytest.fail(f"FTS tags not rebuilt from normalized tables: {row!r}")
    if row.get("metadata_text") != "meta":
        pytest.fail(f"FTS metadata_text not preserved: {row!r}")


async def test_v19_aborts_when_legacy_tags_were_not_backfilled(tmp_path) -> None:
    db = await _make_legacy_db(tmp_path)
    asset_id = await _seed_asset(db)
    legacy = await db.aexecute(
        "INSERT INTO asset_metadata(asset_id, rating, tags, tags_text) VALUES (?, 0, ?, ?)",
        (asset_id, json.dumps(["lost"], ensure_ascii=False), "lost"),
    )
    if not legacy.ok:
        pytest.fail(f"legacy metadata insert failed: {legacy.error}")

    result = await DropLegacyTagColumnsMigration().upgrade(db)
    if result.ok:
        pytest.fail("v19 should abort when legacy tags lack normalized rows")
    if result.code != "MIGRATION_INVARIANT_FAILED":
        pytest.fail(f"unexpected v19 failure code: {result.code}")
