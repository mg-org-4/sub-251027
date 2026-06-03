"""Tests for migration v18 — FTS index sourced from normalized tag tables.

These tests prove that:

* The ``asset_metadata_fts`` index is updated **when normalized tags
  change** (``asset_tags`` INSERT / DELETE) via the new sync triggers.
* The ``tags`` FTS column reflects normalized tag names, **not** the
  legacy ``asset_metadata.tags`` JSON column.
* FTS ``MATCH`` queries find rows by their normalized tag text.
* Missing normalized tags do not influence FTS results.
"""

from __future__ import annotations

import itertools

import pytest

pytestmark = pytest.mark.asyncio

_seed_counter = itertools.count(80_000)


async def _seed_asset(db) -> int:
    cols = await db.aquery("PRAGMA table_info(assets)")
    notnull_cols = [
        c["name"]
        for c in (cols.data or [])
        if c["notnull"] and c["dflt_value"] is None and c["name"] != "id"
    ]
    seq = next(_seed_counter)
    values = tuple(f"fts_seed_{c}_{seq}" for c in notnull_cols)
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
    if "filepath" in notnull_cols:
        filepath = values[notnull_cols.index("filepath")]
        row = await db.aquery("SELECT id FROM assets WHERE filepath = ?", (filepath,))
    else:
        row = await db.aquery("SELECT id FROM assets ORDER BY id DESC LIMIT 1")
    if not row.ok or not row.data:
        pytest.fail(f"Seed asset id lookup failed: {row.error}")
    return int(row.data[0]["id"])


async def _insert_metadata(db, asset_id: int) -> None:
    res = await db.aexecute(
        "INSERT INTO asset_metadata (asset_id) VALUES (?)", (asset_id,)
    )
    if not res.ok:
        pytest.fail(f"insert asset_metadata failed: {res.error}")


async def _add_normalized_tag(db, asset_id: int, name: str) -> None:
    ins_tag = await db.aexecute(
        "INSERT OR IGNORE INTO tags(name) VALUES (?)", (name,)
    )
    if not ins_tag.ok:
        pytest.fail(f"tag insert failed: {ins_tag.error}")
    row = await db.aquery("SELECT id FROM tags WHERE name = ?", (name,))
    tag_id = int(row.data[0]["id"])
    link = await db.aexecute(
        "INSERT OR IGNORE INTO asset_tags(asset_id, tag_id) VALUES (?, ?)",
        (asset_id, tag_id),
    )
    if not link.ok:
        pytest.fail(f"link insert failed: {link.error}")


async def _remove_normalized_tag(db, asset_id: int, name: str) -> None:
    res = await db.aexecute(
        "DELETE FROM asset_tags WHERE asset_id = ? "
        "AND tag_id = (SELECT id FROM tags WHERE name = ?)",
        (asset_id, name),
    )
    if not res.ok:
        pytest.fail(f"delete asset_tag failed: {res.error}")


async def _fts_row(db, asset_id: int) -> dict | None:
    row = await db.aquery(
        "SELECT tags, tags_text, metadata_text FROM asset_metadata_fts WHERE rowid = ?",
        (asset_id,),
    )
    if not row.ok:
        pytest.fail(f"fts read failed: {row.error}")
    return row.data[0] if row.data else None


async def _fts_match_ids(db, query: str) -> set[int]:
    row = await db.aquery(
        "SELECT rowid AS r FROM asset_metadata_fts WHERE asset_metadata_fts MATCH ?",
        (query,),
    )
    if not row.ok:
        pytest.fail(f"fts MATCH failed: {row.error}")
    return {int(r["r"]) for r in (row.data or [])}


# ── Trigger behavior on asset_metadata writes ─────────────────────────────


async def test_metadata_insert_fills_fts_tags_from_normalized(services) -> None:
    db = services["db"]
    asset_id = await _seed_asset(db)
    await _add_normalized_tag(db, asset_id, "alpha")
    await _add_normalized_tag(db, asset_id, "beta")
    # asset_metadata insert AFTER normalized tags are set
    await _insert_metadata(db, asset_id)
    row = await _fts_row(db, asset_id)
    if row is None:
        pytest.fail("FTS row missing after asset_metadata insert")
    tag_text = (row["tags"] or "").split()
    if set(tag_text) != {"alpha", "beta"}:
        pytest.fail(f"unexpected FTS tags content: {row['tags']!r}")


async def test_metadata_insert_without_normalized_tags_keeps_fts_tags_empty(services) -> None:
    """FTS must reflect only normalized tables, empty when no links exist."""
    db = services["db"]
    asset_id = await _seed_asset(db)
    await _insert_metadata(db, asset_id)
    row = await _fts_row(db, asset_id)
    if row is None:
        pytest.fail("FTS row missing after insert")
    if (row["tags"] or "").strip():
        pytest.fail(
            f"FTS tags should be empty (no normalized tags) but got {row['tags']!r}"
        )
    hits = await _fts_match_ids(db, "ghost")
    if asset_id in hits:
        pytest.fail("FTS matched a tag absent from normalized tables")


# ── New triggers on asset_tags ────────────────────────────────────────────


async def test_asset_tags_insert_updates_fts(services) -> None:
    db = services["db"]
    asset_id = await _seed_asset(db)
    await _insert_metadata(db, asset_id)
    # Empty initially
    row = await _fts_row(db, asset_id)
    if row is None or (row["tags"] or "").strip():
        pytest.fail("expected empty initial tags in FTS")

    await _add_normalized_tag(db, asset_id, "freshtag")
    row = await _fts_row(db, asset_id)
    if row is None:
        pytest.fail("FTS row missing after asset_tags insert")
    if "freshtag" not in (row["tags"] or ""):
        pytest.fail(f"FTS tags did not reflect new asset_tags insert: {row['tags']!r}")

    hits = await _fts_match_ids(db, "freshtag")
    if asset_id not in hits:
        pytest.fail("FTS MATCH did not find the new tag")


async def test_asset_tags_delete_updates_fts(services) -> None:
    db = services["db"]
    asset_id = await _seed_asset(db)
    await _add_normalized_tag(db, asset_id, "keepme")
    await _add_normalized_tag(db, asset_id, "removeme")
    await _insert_metadata(db, asset_id)

    await _remove_normalized_tag(db, asset_id, "removeme")
    row = await _fts_row(db, asset_id)
    if row is None:
        pytest.fail("FTS row missing after asset_tags delete")
    remaining = set((row["tags"] or "").split())
    if remaining != {"keepme"}:
        pytest.fail(f"FTS tags after delete unexpected: {row['tags']!r}")

    hits = await _fts_match_ids(db, "removeme")
    if asset_id in hits:
        pytest.fail("FTS still matches deleted tag")


# ── Update / Reindex semantics ────────────────────────────────────────────


async def test_metadata_update_refreshes_fts_tags(services) -> None:
    """An UPDATE on asset_metadata must rebuild the tags column from the
    current normalized state, not from the (legacy) ``new.tags``."""
    db = services["db"]
    asset_id = await _seed_asset(db)
    await _insert_metadata(db, asset_id)
    await _add_normalized_tag(db, asset_id, "after_update")

    # Touch asset_metadata to fire UPDATE trigger
    res = await db.aexecute(
        "UPDATE asset_metadata SET metadata_text = ? WHERE asset_id = ?",
        ("some meta", asset_id),
    )
    if not res.ok:
        pytest.fail(f"metadata update failed: {res.error}")

    row = await _fts_row(db, asset_id)
    if row is None:
        pytest.fail("FTS row missing after metadata UPDATE")
    if "after_update" not in (row["tags"] or ""):
        pytest.fail(f"UPDATE trigger did not refresh tags: {row['tags']!r}")
    if "some meta" not in (row["metadata_text"] or ""):
        pytest.fail("UPDATE trigger lost metadata_text")


async def test_v18_migration_recorded(services) -> None:
    """M018 must appear in schema_migrations after bootstrap."""
    db = services["db"]
    row = await db.aquery(
        "SELECT name FROM schema_migrations WHERE version = 18"
    )
    if not row.ok or not row.data:
        pytest.fail("v18 migration missing from schema_migrations")
    if row.data[0]["name"] != "fts_from_normalized_tags":
        pytest.fail(f"unexpected v18 name: {row.data[0]['name']!r}")
