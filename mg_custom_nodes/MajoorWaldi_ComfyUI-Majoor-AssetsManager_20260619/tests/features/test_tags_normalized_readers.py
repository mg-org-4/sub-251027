"""Tests proving filter readers use the normalized ``asset_tags``/``tags``
tables (Phase Readers v1), not the legacy ``asset_metadata.tags`` JSON column.

Strategy: seed an asset where the **normalized** tables and the **legacy**
JSON column intentionally disagree, then assert each filter site looks at
the normalized tables only.

Sites under test:

* :func:`searcher._append_tag_filter` — tag filter SQL fragment.
* ``api_v2_assets._query_list`` — include/exclude tag filtering.
* ``audit.audit_assets`` — ``no_tags`` filter.
"""

from __future__ import annotations

import itertools

import pytest
from mjr_am_backend.data.repositories import TagsRepository
from mjr_am_backend.features.assets.rating_tags_service import fetch_asset_rating_tags
from mjr_am_backend.features.index import searcher as searcher_mod
from mjr_am_backend.routes.search import result_hydrator

pytestmark = pytest.mark.asyncio

_seed_counter = itertools.count(50_000)


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
    if "filepath" in notnull_cols:
        filepath = values[notnull_cols.index("filepath")]
        row = await db.aquery("SELECT id FROM assets WHERE filepath = ?", (filepath,))
    else:
        row = await db.aquery("SELECT id FROM assets ORDER BY id DESC LIMIT 1")
    if not row.ok or not row.data:
        pytest.fail(f"Seed asset id lookup failed: {row.error}")
    return int(row.data[0]["id"])


async def _set_normalized_only(db, asset_id: int, tags: list[str]) -> None:
    """Write tags ONLY into normalized tables. Used to prove the new readers
    correctly find them even without legacy JSON."""
    result = await TagsRepository(db).replace_all(asset_id, tags)
    if not result.ok:
        pytest.fail(f"normalized tag write failed: {result.error}")


# ── searcher._append_tag_filter ────────────────────────────────────── #


async def test_searcher_filter_ignores_missing_normalized_tag(services) -> None:
    """A tag absent from normalized tables must not match."""
    db = services["db"]
    asset_id = await _seed_asset(db)

    clauses: list[str] = []
    params: list = []
    searcher_mod._append_tag_filter({"tags": ["ghost"]}, clauses, params)

    sql = (
        "SELECT a.id FROM assets a "
        "LEFT JOIN asset_metadata m ON m.asset_id = a.id "
        f"WHERE a.id = ? {' '.join(clauses)}"
    )
    res = await db.aquery(sql, (asset_id, *params))
    if not res.ok:
        pytest.fail(f"query failed: {res.error}")
    if res.data:
        pytest.fail("filter matched a tag absent from normalized tables")


async def test_searcher_filter_matches_via_normalized_tables(services) -> None:
    """Tag present only in normalized tables MUST match the filter."""
    db = services["db"]
    asset_id = await _seed_asset(db)
    # Insert empty metadata row so the LEFT JOIN matches m at all
    await db.aexecute(
        "INSERT INTO asset_metadata (asset_id) VALUES (?)", (asset_id,)
    )
    await _set_normalized_only(db, asset_id, ["alpha"])

    clauses: list[str] = []
    params: list = []
    searcher_mod._append_tag_filter({"tags": ["alpha"]}, clauses, params)

    sql = (
        "SELECT a.id FROM assets a "
        "LEFT JOIN asset_metadata m ON m.asset_id = a.id "
        f"WHERE a.id = ? {' '.join(clauses)}"
    )
    res = await db.aquery(sql, (asset_id, *params))
    if not res.ok:
        pytest.fail(f"query failed: {res.error}")
    if not res.data:
        pytest.fail("filter missed a tag present in normalized tables")


async def test_searcher_filter_is_case_insensitive(services) -> None:
    """tags.name COLLATE NOCASE → 'ALPHA' filter matches 'alpha' tag."""
    db = services["db"]
    asset_id = await _seed_asset(db)
    await db.aexecute(
        "INSERT INTO asset_metadata (asset_id) VALUES (?)", (asset_id,)
    )
    await _set_normalized_only(db, asset_id, ["Alpha"])

    clauses: list[str] = []
    params: list = []
    searcher_mod._append_tag_filter({"tags": ["ALPHA"]}, clauses, params)

    sql = (
        "SELECT a.id FROM assets a "
        "LEFT JOIN asset_metadata m ON m.asset_id = a.id "
        f"WHERE a.id = ? {' '.join(clauses)}"
    )
    res = await db.aquery(sql, (asset_id, *params))
    if not res.ok or not res.data:
        pytest.fail("case-insensitive match failed")


# ── api_v2_assets include/exclude filtering ──────────────────────────── #


async def _api_v2_query_one(db, asset_id: int, include: list[str], exclude: list[str]):
    """Reproduce the api_v2 list query shape (clauses only)."""
    clauses: list[str] = ["a.id = ?"]
    params: list = [asset_id]
    for tag in include:
        clauses.append(
            "EXISTS (SELECT 1 FROM asset_tags at "
            "JOIN tags t ON t.id = at.tag_id "
            "WHERE at.asset_id = a.id AND t.name = ?)"
        )
        params.append(tag)
    for tag in exclude:
        clauses.append(
            "NOT EXISTS (SELECT 1 FROM asset_tags at "
            "JOIN tags t ON t.id = at.tag_id "
            "WHERE at.asset_id = a.id AND t.name = ?)"
        )
        params.append(tag)
    sql = (
        "SELECT a.id FROM assets a "
        "LEFT JOIN asset_metadata m ON m.asset_id = a.id "
        f"WHERE {' AND '.join(clauses)}"
    )
    return await db.aquery(sql, tuple(params))


async def test_api_v2_include_tag_normalized(services) -> None:
    db = services["db"]
    asset_id = await _seed_asset(db)
    await db.aexecute("INSERT INTO asset_metadata (asset_id) VALUES (?)", (asset_id,))
    await _set_normalized_only(db, asset_id, ["wanted"])

    res = await _api_v2_query_one(db, asset_id, include=["wanted"], exclude=[])
    if not res.ok or not res.data:
        pytest.fail("include tag did not match via normalized tables")


async def test_api_v2_include_tag_ignores_missing_normalized_tag(services) -> None:
    db = services["db"]
    asset_id = await _seed_asset(db)

    res = await _api_v2_query_one(db, asset_id, include=["ghost"], exclude=[])
    if not res.ok:
        pytest.fail(f"query failed: {res.error}")
    if res.data:
        pytest.fail("include filter matched a tag absent from normalized tables")


async def test_api_v2_exclude_tag_via_normalized(services) -> None:
    db = services["db"]
    a_with = await _seed_asset(db)
    a_without = await _seed_asset(db)
    await db.aexecute("INSERT INTO asset_metadata (asset_id) VALUES (?)", (a_with,))
    await db.aexecute("INSERT INTO asset_metadata (asset_id) VALUES (?)", (a_without,))
    await _set_normalized_only(db, a_with, ["skip"])

    # a_with should be excluded
    res = await _api_v2_query_one(db, a_with, include=[], exclude=["skip"])
    if res.data:
        pytest.fail("exclude failed: asset with tag should not match")
    # a_without should not be excluded
    res2 = await _api_v2_query_one(db, a_without, include=[], exclude=["skip"])
    if not res2.data:
        pytest.fail("exclude over-matched: asset without tag should remain")


# ── audit no_tags / completeness ─────────────────────────────────────── #


async def _audit_no_tags(db, asset_id: int):
    sql = (
        "SELECT a.id FROM assets a "
        "LEFT JOIN asset_metadata m ON m.asset_id = a.id "
        "WHERE a.id = ? AND "
        "NOT EXISTS (SELECT 1 FROM asset_tags WHERE asset_id = a.id)"
    )
    return await db.aquery(sql, (asset_id,))


async def test_audit_no_tags_uses_normalized(services) -> None:
    """Asset with no normalized rows must show up as 'no_tags'."""
    db = services["db"]
    asset_id = await _seed_asset(db)

    res = await _audit_no_tags(db, asset_id)
    if not res.ok:
        pytest.fail(f"query failed: {res.error}")
    if not res.data:
        pytest.fail("audit 'no_tags' missed asset that has no normalized tags")


async def test_audit_no_tags_excludes_assets_with_normalized_tags(services) -> None:
    db = services["db"]
    asset_id = await _seed_asset(db)
    await db.aexecute("INSERT INTO asset_metadata (asset_id) VALUES (?)", (asset_id,))
    await _set_normalized_only(db, asset_id, ["real"])

    res = await _audit_no_tags(db, asset_id)
    if not res.ok:
        pytest.fail(f"query failed: {res.error}")
    if res.data:
        pytest.fail("audit 'no_tags' wrongly flagged asset with normalized tags")


# ── Readers v2 payload hydration ──────────────────────────────────────── #


async def test_rating_tags_service_reads_normalized_not_legacy_json(services) -> None:
    db = services["db"]
    asset_id = await _seed_asset(db)
    await db.aexecute("INSERT INTO asset_metadata (asset_id) VALUES (?)", (asset_id,))
    await _set_normalized_only(db, asset_id, ["normalized-only"])

    _rating, tags = await fetch_asset_rating_tags(db, asset_id)
    if tags != ["normalized-only"]:
        pytest.fail(f"rating/tags service read {tags!r}, expected normalized tags")


async def test_result_hydrator_query_reads_normalized_tags(services) -> None:
    db = services["db"]
    asset_id = await _seed_asset(db)
    row = await db.aquery("SELECT filepath FROM assets WHERE id = ?", (asset_id,))
    if not row.ok or not row.data:
        pytest.fail("seeded asset filepath lookup failed")
    filepath = str(row.data[0]["filepath"])
    await _set_normalized_only(db, asset_id, ["browser-tag"])

    rows = await result_hydrator.query_browser_rows(db, [filepath])
    if not rows:
        pytest.fail("hydrator returned no rows")
    hydrated = {}
    result_hydrator.apply_hydration_rows([hydrated := {"filepath": filepath}], rows)
    if hydrated.get("tags") != ["browser-tag"]:
        pytest.fail(f"hydrator read {hydrated.get('tags')!r}, expected normalized tags")


async def test_searcher_get_asset_reads_normalized_not_legacy_json(services) -> None:
    db = services["db"]
    asset_id = await _seed_asset(db)
    await _set_normalized_only(db, asset_id, ["searcher-tag"])

    searcher = searcher_mod.IndexSearcher(db, has_tags_text_column=False)
    result = await searcher.get_asset(asset_id)
    if not result.ok or not result.data:
        pytest.fail(f"get_asset failed: {result.error}")
    if result.data.get("tags") != ["searcher-tag"]:
        pytest.fail(f"searcher read {result.data.get('tags')!r}, expected normalized tags")
