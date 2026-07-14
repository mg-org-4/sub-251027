"""
Tests for importing ratings and tags logic, ensuring no data loss or incorrect overwrites.
"""
import json
from pathlib import Path

import pytest
from mjr_am_backend.adapters.db.migrations import MigrationRunner
from mjr_am_backend.adapters.db.migrations.registry import MIGRATIONS
from mjr_am_backend.adapters.db.schema import migrate_schema
from mjr_am_backend.adapters.db.sqlite import Sqlite
from mjr_am_backend.features.index.metadata_helpers import MetadataHelpers
from mjr_am_backend.shared import Result


async def _make_db(tmp_path: Path) -> Sqlite:
    db_path = tmp_path / "test.db"
    db = Sqlite(str(db_path), attach={"vec": str(tmp_path / "vectors.sqlite")})
    mig = await migrate_schema(db)
    assert mig.ok
    runner_res = await MigrationRunner(MIGRATIONS).run(db)
    assert runner_res.ok, runner_res.error
    return db


async def _normalized_tags(db: Sqlite, asset_id: int) -> list[str]:
    res = await db.aquery(
        "SELECT t.name AS name FROM asset_tags at "
        "JOIN tags t ON t.id = at.tag_id "
        "WHERE at.asset_id = ? ORDER BY t.name",
        (asset_id,),
    )
    assert res.ok
    return [str(row["name"]) for row in (res.data or [])]


@pytest.mark.asyncio
async def test_imports_rating_tags_when_db_empty(tmp_path: Path):
    db = await _make_db(tmp_path)
    try:
        ins = await db.aexecute(
            "INSERT INTO assets(filepath, filename, subfolder, source, root_id, kind, ext, size, mtime) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("C:\\x\\a.png", "a.png", "", "output", "output", "image", ".png", 123, 1700000000),
        )
        assert ins.ok
        asset_rows = (await db.aquery("SELECT id FROM assets")).data
        assert asset_rows
        asset_id = asset_rows[0]["id"]

        meta = Result.Ok({"rating": 4, "tags": ["foo", "bar"], "quality": "partial"})
        w = await MetadataHelpers.write_asset_metadata_row(db, asset_id, meta)
        assert w.ok

        rows = (await db.aquery("SELECT rating FROM asset_metadata WHERE asset_id = ?", (asset_id,))).data
        assert rows
        row = rows[0]
        assert row["rating"] == 4
        assert await _normalized_tags(db, asset_id) == ["bar", "foo"]
    finally:
        await db.aclose()


@pytest.mark.asyncio
async def test_does_not_override_existing_db_rating_tags(tmp_path: Path):
    db = await _make_db(tmp_path)
    try:
        ins = await db.aexecute(
            "INSERT INTO assets(filepath, filename, subfolder, source, root_id, kind, ext, size, mtime) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("C:\\x\\b.png", "b.png", "", "output", "output", "image", ".png", 123, 1700000000),
        )
        assert ins.ok
        asset_rows = (await db.aquery("SELECT id FROM assets")).data
        assert asset_rows
        asset_id = asset_rows[0]["id"]

        await db.aexecute(
            "INSERT INTO asset_metadata(asset_id, rating, has_workflow, has_generation_data, metadata_quality, metadata_raw) VALUES(?, ?, 0, 0, 'none', '{}')",
            (asset_id, 5),
        )
        from mjr_am_backend.data.repositories import TagsRepository

        seeded = await TagsRepository(db).replace_all(asset_id, ["keep"])
        assert seeded.ok

        meta = Result.Ok({"rating": 1, "tags": ["overwrite"], "quality": "partial"})
        w = await MetadataHelpers.write_asset_metadata_row(db, asset_id, meta)
        assert w.ok

        rows = (await db.aquery("SELECT rating FROM asset_metadata WHERE asset_id = ?", (asset_id,))).data
        assert rows
        row = rows[0]
        assert row["rating"] == 5
        assert await _normalized_tags(db, asset_id) == ["keep"]
    finally:
        await db.aclose()


@pytest.mark.asyncio
async def test_equal_quality_sparse_update_does_not_clobber_workflow_flags_or_raw(tmp_path: Path):
    db = await _make_db(tmp_path)
    try:
        ins = await db.aexecute(
            "INSERT INTO assets(filepath, filename, subfolder, source, root_id, kind, ext, size, mtime) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("C:\\x\\c.png", "c.png", "", "output", "output", "image", ".png", 123, 1700000000),
        )
        assert ins.ok
        asset_rows = (await db.aquery("SELECT id FROM assets")).data
        assert asset_rows
        asset_id = asset_rows[0]["id"]

        rich_partial = Result.Ok(
            {
                "quality": "partial",
                "workflow": {"nodes": [{"id": 1, "type": "KSampler"}]},
                "prompt": {"1": {"class_type": "KSampler", "inputs": {"steps": 20}}},
            }
        )
        first = await MetadataHelpers.write_asset_metadata_row(db, asset_id, rich_partial)
        assert first.ok

        sparse_partial = Result.Ok({"quality": "partial", "rating": 3, "tags": ["foo"]})
        second = await MetadataHelpers.write_asset_metadata_row(db, asset_id, sparse_partial)
        assert second.ok

        rows = (
            await db.aquery(
                "SELECT has_workflow, has_generation_data, metadata_quality, metadata_raw FROM asset_metadata WHERE asset_id = ?",
                (asset_id,),
            )
        ).data
        assert rows
        row = rows[0]
        assert row["has_workflow"] == 1
        assert row["has_generation_data"] == 1
        assert row["metadata_quality"] == "partial"
        raw = json.loads(str(row["metadata_raw"] or "{}"))
        assert "workflow" in raw
        assert "prompt" in raw
    finally:
        await db.aclose()
