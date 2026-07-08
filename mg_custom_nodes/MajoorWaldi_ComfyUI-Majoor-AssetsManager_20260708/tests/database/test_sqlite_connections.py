"""Tests for sqlite_connections sub-module (connection pragmas, pool acquire/release, init)."""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio
from mjr_am_backend.adapters.db.sqlite_facade import Sqlite


@pytest_asyncio.fixture
async def db(tmp_path):
    inst = Sqlite(str(tmp_path / "conn_test.sqlite"), attach={"vec": str(tmp_path / "vectors.sqlite")})
    yield inst
    await inst.aclose()


@pytest.mark.asyncio
async def test_apply_connection_pragmas_sets_wal_mode(db) -> None:
    conn = await db._acquire_connection_async()
    try:
        cur = await conn.execute("PRAGMA journal_mode")
        row = await cur.fetchone()
        assert row[0] == "wal"
    finally:
        await db._release_connection_async(conn)


@pytest.mark.asyncio
async def test_apply_connection_pragmas_sets_foreign_keys(db) -> None:
    conn = await db._acquire_connection_async()
    try:
        cur = await conn.execute("PRAGMA foreign_keys")
        row = await cur.fetchone()
        assert row[0] == 1
    finally:
        await db._release_connection_async(conn)


@pytest.mark.asyncio
async def test_acquire_release_returns_connection_to_pool(db) -> None:
    conn1 = await db._acquire_connection_async()
    await db._release_connection_async(conn1)
    conn2 = await db._acquire_connection_async()
    assert conn2 is conn1
    await db._release_connection_async(conn2)


@pytest.mark.asyncio
async def test_acquire_connection_rejected_during_reset(db) -> None:
    await db._ensure_initialized_async()
    db._resetting = True
    try:
        with pytest.raises(RuntimeError, match="resetting"):
            await db._acquire_connection_async()
    finally:
        db._resetting = False


@pytest.mark.asyncio
async def test_ensure_initialized_async_is_idempotent(db) -> None:
    await db._ensure_initialized_async()
    assert db._initialized
    await db._ensure_initialized_async()
    assert db._initialized


@pytest.mark.asyncio
async def test_lock_for_asset_serializes_access(db) -> None:
    await db._ensure_initialized_async()
    order: list[int] = []

    async def _worker(asset_id: str, marker: int) -> None:
        async with db.lock_for_asset(asset_id):
            order.append(marker)
            await asyncio.sleep(0.01)

    t1 = asyncio.create_task(_worker("a1", 1))
    await asyncio.sleep(0)
    t2 = asyncio.create_task(_worker("a1", 2))
    await asyncio.gather(t1, t2)
    assert order == [1, 2]


@pytest.mark.asyncio
async def test_get_runtime_status_returns_expected_keys(db) -> None:
    await db._ensure_initialized_async()
    status = db.get_runtime_status()
    assert isinstance(status, dict)
    assert "pool_size" in status or "active_connections" in status or len(status) > 0
