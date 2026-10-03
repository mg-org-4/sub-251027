"""Tests covering dual-write of legacy ``asset_metadata.tags`` JSON
into the normalized ``tags`` / ``asset_tags`` tables (Phase 3.2 transition).

The writer sites under test are:

* :meth:`AssetUpdater.update_asset_tags` — direct user edit.
* :meth:`DuplicatesService.merge_tags_for_group` — duplicate merge.
* :meth:`TagsRepository.sync_from_legacy_row` — generic helper used by the
  enrichment UPSERT in ``metadata_helpers``.
"""

from __future__ import annotations

import itertools
import json

import pytest
from mjr_am_backend.adapters.db.migrations import MigrationRunner
from mjr_am_backend.adapters.db.migrations.registry import MIGRATIONS
from mjr_am_backend.data.repositories import TagsRepository
from mjr_am_backend.features.duplicates.service import DuplicatesService
from mjr_am_backend.features.index.updater import AssetUpdater

pytestmark = pytest.mark.asyncio

_seed_counter = itertools.count(10_000)


async def _setup(services):
    db = services["db"]
    result = await MigrationRunner(MIGRATIONS).run(db)
    if not result.ok:
        pytest.fail(f"Migration runner failed: {result.error}")
    return db


async def _seed_asset(db) -> int:
    cols = await db.aquery("PRAGMA table_info(assets)")
    notnull_cols = [
        c["name"]
        for c in (cols.data or [])
        if c["notnull"] and c["dflt_value"] is None and c["name"] != "id"
    ]
    seq = next(_seed_counter)
    values = tuple(f"unique_{c}_{seq}" for c in notnull_cols)
    if notnull_cols:
        placeholders = ", ".join(["?"] * len(notnull_cols))
        col_list = ", ".join(notnull_cols)
        ins = await db.aexecute(
            f"INSERT INTO assets ({col_list}) VALUES ({placeholders})", values
        )
    else:
        ins = await db.aexecute("INSERT INTO assets DEFAULT VALUES")
    if not ins.ok:
        pytest.fail(f"Seed asset failed: {ins.error}")
    row = await db.aquery("SELECT last_insert_rowid() AS id")
    return int((row.data or [{"id": 1}])[0]["id"])


async def _legacy_tags(db, asset_id: int) -> list[str]:
    row = await db.aquery(
        "SELECT tags FROM asset_metadata WHERE asset_id = ?", (asset_id,)
    )
    if not row.ok or not row.data:
        return []
    raw = row.data[0].get("tags")
    if not isinstance(raw, str) or not raw:
        return []
    parsed = json.loads(raw)
    return [t for t in parsed if isinstance(t, str)]


async def _legacy_tags_text(db, asset_id: int) -> str:
    row = await db.aquery(
        "SELECT tags_text FROM asset_metadata WHERE asset_id = ?", (asset_id,),
    )
    if not row.ok or not row.data:
        return ""
    return str(row.data[0].get("tags_text") or "")


async def _normalized_tags(db, asset_id: int) -> list[str]:
    res = await db.aquery(
        "SELECT t.name AS name FROM asset_tags at "
        "JOIN tags t ON t.id = at.tag_id WHERE at.asset_id = ? ORDER BY t.name",
        (asset_id,),
    )
    if not res.ok:
        return []
    return [str(r["name"]) for r in (res.data or [])]


# ── AssetUpdater.update_asset_tags ────────────────────────────────── #


async def test_asset_updater_mirrors_tags_to_normalized_tables(services) -> None:
    db = await _setup(services)
    asset_id = await _seed_asset(db)
    updater = AssetUpdater(db=db, has_tags_text_column=True)

    res = await updater.update_asset_tags(asset_id, ["alpha", "beta"])
    if not res.ok:
        pytest.fail(f"update_asset_tags failed: {res.error}")

    assert await _normalized_tags(db, asset_id) == ["alpha", "beta"]
    if await _legacy_tags(db, asset_id):
        pytest.fail("legacy JSON tags were written during Stop-Write")
    if await _legacy_tags_text(db, asset_id):
        pytest.fail("legacy tags_text was written during Stop-Write")


async def test_asset_updater_replace_overwrites_normalized(services) -> None:
    db = await _setup(services)
    asset_id = await _seed_asset(db)
    updater = AssetUpdater(db=db, has_tags_text_column=True)

    res = await updater.update_asset_tags(asset_id, ["a", "b", "c"])
    if not res.ok:
        pytest.fail(f"first update failed: {res.error}")
    res2 = await updater.update_asset_tags(asset_id, ["c", "d"])
    if not res2.ok:
        pytest.fail(f"second update failed: {res2.error}")

    assert await _normalized_tags(db, asset_id) == ["c", "d"]
    if await _legacy_tags(db, asset_id):
        pytest.fail("legacy JSON tags were written during Stop-Write")


async def test_asset_updater_empty_tags_clears_normalized(services) -> None:
    db = await _setup(services)
    asset_id = await _seed_asset(db)
    updater = AssetUpdater(db=db, has_tags_text_column=True)

    if not (await updater.update_asset_tags(asset_id, ["x", "y"])).ok:
        pytest.fail("seed update failed")
    if not (await updater.update_asset_tags(asset_id, [])).ok:
        pytest.fail("clear update failed")

    assert await _normalized_tags(db, asset_id) == []


# ── DuplicatesService.merge_tags_for_group ────────────────────────── #


async def _insert_metadata_with_tags(db, asset_id: int, tags: list[str]) -> None:
    for ddl in (
        "ALTER TABLE asset_metadata ADD COLUMN tags TEXT DEFAULT ''",
        "ALTER TABLE asset_metadata ADD COLUMN tags_text TEXT DEFAULT ''",
    ):
        await db.aexecute(ddl)
    tags_json = json.dumps(tags, ensure_ascii=False)
    ins = await db.aexecute(
        "INSERT INTO asset_metadata (asset_id, rating, tags, tags_text) "
        "VALUES (?, 0, ?, ?)",
        (asset_id, tags_json, " ".join(tags)),
    )
    if not ins.ok:
        pytest.fail(f"insert metadata failed: {ins.error}")


async def _insert_normalized_tags(db, asset_id: int, tags: list[str]) -> None:
    repo = TagsRepository(db)
    res = await repo.replace_all(asset_id, tags)
    if not res.ok:
        pytest.fail(f"insert normalized tags failed: {res.error}")


async def test_duplicates_merge_mirrors_merged_tags(services) -> None:
    db = await _setup(services)
    keep_id = await _seed_asset(db)
    other_id = await _seed_asset(db)
    await db.aexecute("INSERT INTO asset_metadata (asset_id, rating) VALUES (?, 0)", (keep_id,))
    await db.aexecute("INSERT INTO asset_metadata (asset_id, rating) VALUES (?, 0)", (other_id,))
    await _insert_normalized_tags(db, keep_id, ["keep1", "shared"])
    await _insert_normalized_tags(db, other_id, ["shared", "other"])

    svc = DuplicatesService(db)
    res = await svc.merge_tags_for_group(keep_id, [other_id])
    if not res.ok:
        pytest.fail(f"merge failed: {res.error}")

    normalized = sorted(await _normalized_tags(db, keep_id))
    assert set(normalized) == {"keep1", "shared", "other"}
    if await _legacy_tags(db, keep_id):
        pytest.fail("duplicates merge wrote legacy JSON tags during Stop-Write")


# ── TagsRepository.sync_from_legacy_row ──────────────────────────── #


async def test_sync_from_legacy_row_reads_back_committed_state(services) -> None:
    db = await _setup(services)
    asset_id = await _seed_asset(db)
    await _insert_metadata_with_tags(db, asset_id, ["foo", "bar"])

    repo = TagsRepository(db)
    res = await repo.sync_from_legacy_row(asset_id)
    if not res.ok:
        pytest.fail(f"sync failed: {res.error}")

    assert await _normalized_tags(db, asset_id) == ["bar", "foo"]


async def test_sync_from_legacy_row_handles_missing_row(services) -> None:
    db = await _setup(services)
    repo = TagsRepository(db)
    # asset_id with no asset_metadata entry
    res = await repo.sync_from_legacy_row(999_999)
    if not res.ok:
        pytest.fail(f"sync failed: {res.error}")
    assert await _normalized_tags(db, 999_999) == []


async def test_sync_from_legacy_row_handles_malformed_json(services) -> None:
    db = await _setup(services)
    asset_id = await _seed_asset(db)
    await db.aexecute("ALTER TABLE asset_metadata ADD COLUMN tags TEXT DEFAULT ''")
    ins = await db.aexecute(
        "INSERT INTO asset_metadata (asset_id, rating, tags) VALUES (?, 0, ?)",
        (asset_id, "{not valid json"),
    )
    if not ins.ok:
        pytest.fail(f"insert failed: {ins.error}")

    repo = TagsRepository(db)
    res = await repo.sync_from_legacy_row(asset_id)
    if not res.ok:
        pytest.fail(f"sync failed: {res.error}")
    assert await _normalized_tags(db, asset_id) == []


async def test_replace_all_dedups_case_insensitive(services) -> None:
    db = await _setup(services)
    asset_id = await _seed_asset(db)
    repo = TagsRepository(db)

    res = await repo.replace_all(asset_id, ["Cat", "cat", "Dog", "  ", "DOG"])
    if not res.ok:
        pytest.fail(f"replace_all failed: {res.error}")

    names = await _normalized_tags(db, asset_id)
    # Only one of Cat/cat and one of Dog/DOG retained; whitespace dropped.
    assert len(names) == 2
    assert {n.casefold() for n in names} == {"cat", "dog"}
