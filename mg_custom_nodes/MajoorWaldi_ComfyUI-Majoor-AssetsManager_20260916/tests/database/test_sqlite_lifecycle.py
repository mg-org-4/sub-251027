"""Tests for sqlite_lifecycle sub-module (transactions, commit, rollback, retry)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from mjr_am_backend.adapters.db.sqlite_facade import Sqlite


@pytest_asyncio.fixture
async def db(tmp_path):
    inst = Sqlite(str(tmp_path / "life_test.sqlite"), attach={"vec": str(tmp_path / "vectors.sqlite")})
    yield inst
    await inst.aclose()


@pytest.mark.asyncio
async def test_begin_commit_transaction(db) -> None:
    await db.aexecute("CREATE TABLE IF NOT EXISTS t_tx (id INTEGER PRIMARY KEY, v TEXT)")
    async with db.atransaction(mode="immediate") as tx:
        assert tx.ok
        ins = await db.aexecute("INSERT INTO t_tx(v) VALUES (?)", ("committed",))
        assert ins.ok
    rows = await db.aquery("SELECT v FROM t_tx")
    assert rows.ok
    assert rows.data[0]["v"] == "committed"


@pytest.mark.asyncio
async def test_rollback_transaction_on_error(db) -> None:
    await db.aexecute("CREATE TABLE IF NOT EXISTS t_rb (id INTEGER PRIMARY KEY, v TEXT)")
    try:
        async with db.atransaction(mode="immediate"):
            await db.aexecute("INSERT INTO t_rb(v) VALUES (?)", ("rolled_back",))
            raise ValueError("force rollback")
    except ValueError:
        pass
    rows = await db.aquery("SELECT v FROM t_rb")
    assert rows.ok
    assert len(rows.data) == 0


@pytest.mark.asyncio
async def test_explicit_begin_commit_rollback(db) -> None:
    await db.aexecute("CREATE TABLE IF NOT EXISTS t_explicit (id INTEGER PRIMARY KEY, v TEXT)")
    tx_result = await db._begin_tx_async(mode="immediate")
    assert tx_result.ok
    token = tx_result.data
    await db.aexecute("INSERT INTO t_explicit(v) VALUES (?)", ("before_rollback",))
    rb_result = await db._rollback_tx_async(token)
    assert rb_result.ok
    rows = await db.aquery("SELECT v FROM t_explicit")
    assert rows.ok
    assert len(rows.data) == 0


@pytest.mark.asyncio
async def test_explicit_begin_then_commit(db) -> None:
    await db.aexecute("CREATE TABLE IF NOT EXISTS t_exp_c (id INTEGER PRIMARY KEY, v TEXT)")
    async with db.atransaction(mode="immediate") as tx:
        assert tx.ok
        await db.aexecute("INSERT INTO t_exp_c(v) VALUES (?)", ("persisted",))
    rows = await db.aquery("SELECT v FROM t_exp_c")
    assert rows.ok
    assert len(rows.data) == 1
    assert rows.data[0]["v"] == "persisted"


@pytest.mark.asyncio
async def test_commit_tx_missing_token_returns_err(db) -> None:
    await db._ensure_initialized_async()
    result = await db._commit_tx_async("tx_nonexistent")
    assert not result.ok
    assert "missing" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_rollback_tx_missing_token_returns_err(db) -> None:
    await db._ensure_initialized_async()
    result = await db._rollback_tx_async("tx_nonexistent")
    assert not result.ok
    assert "missing" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_nested_transactions_reuse_outer(db) -> None:
    await db.aexecute("CREATE TABLE IF NOT EXISTS t_nest (id INTEGER PRIMARY KEY, v TEXT)")
    async with db.atransaction(mode="immediate") as outer:
        assert outer.ok
        async with db.atransaction(mode="immediate") as inner:
            assert inner.ok
            await db.aexecute("INSERT INTO t_nest(v) VALUES (?)", ("nested",))
    rows = await db.aquery("SELECT v FROM t_nest")
    assert rows.ok
    assert len(rows.data) == 1
