"""Tests for sqlite_execution sub-module (query execution, timeout, retry, batch)."""

from __future__ import annotations

import pytest
import pytest_asyncio
from mjr_am_backend.adapters.db.sqlite_facade import Sqlite


@pytest_asyncio.fixture
async def db(tmp_path):
    inst = Sqlite(str(tmp_path / "exec_test.sqlite"), attach={"vec": str(tmp_path / "vectors.sqlite")})
    yield inst
    await inst.aclose()


@pytest.mark.asyncio
async def test_execute_async_rejects_unresolved_template(db) -> None:
    result = await db.aexecute(
        "SELECT * FROM assets WHERE id IN ({IN_CLAUSE})",
        (1,),
    )
    assert not result.ok
    assert "unresolved template" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_execute_async_insert_and_query(db) -> None:
    await db.aexecute("CREATE TABLE IF NOT EXISTS t_exec (id INTEGER PRIMARY KEY, v TEXT)")
    ins = await db.aexecute("INSERT INTO t_exec(v) VALUES (?)", ("hello",))
    assert ins.ok
    rows = await db.aquery("SELECT v FROM t_exec")
    assert rows.ok
    assert len(rows.data) == 1
    assert rows.data[0]["v"] == "hello"


@pytest.mark.asyncio
async def test_execute_returns_rowcount_on_write(db) -> None:
    await db.aexecute("CREATE TABLE IF NOT EXISTS t_rc (id INTEGER PRIMARY KEY, v TEXT)")
    await db.aexecute("INSERT INTO t_rc(v) VALUES ('a')")
    await db.aexecute("INSERT INTO t_rc(v) VALUES ('b')")
    upd = await db.aexecute("UPDATE t_rc SET v = 'x'")
    assert upd.ok
    assert upd.data in (2, True)


@pytest.mark.asyncio
async def test_executemany_async_batch_insert(db) -> None:
    await db.aexecute("CREATE TABLE IF NOT EXISTS t_batch (id INTEGER PRIMARY KEY, v TEXT)")
    batch = await db.aexecutemany(
        "INSERT INTO t_batch(v) VALUES (?)",
        [("a",), ("b",), ("c",)],
    )
    assert batch.ok
    rows = await db.aquery("SELECT v FROM t_batch ORDER BY v")
    assert rows.ok
    assert [r["v"] for r in rows.data] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_executescript_async_runs_multi_statement(db) -> None:
    script = (
        "CREATE TABLE IF NOT EXISTS t_script (id INTEGER PRIMARY KEY, v TEXT);\n"
        "INSERT INTO t_script(v) VALUES ('x');\n"
        "INSERT INTO t_script(v) VALUES ('y');\n"
    )
    result = await db.aexecutescript(script)
    assert result.ok
    rows = await db.aquery("SELECT v FROM t_script ORDER BY v")
    assert rows.ok
    assert len(rows.data) == 2


@pytest.mark.asyncio
async def test_execute_with_query_timeout_zero_does_not_timeout(db) -> None:
    """When query_timeout is 0 (default), queries run without timeout wrapper."""
    await db.aexecute("CREATE TABLE IF NOT EXISTS t_to (id INTEGER PRIMARY KEY)")
    result = await db.aexecute("INSERT INTO t_to DEFAULT VALUES")
    assert result.ok


@pytest.mark.asyncio
async def test_execute_integrity_error_returns_result_err(db) -> None:
    await db.aexecute("CREATE TABLE IF NOT EXISTS t_uniq (id INTEGER PRIMARY KEY, v TEXT UNIQUE)")
    await db.aexecute("INSERT INTO t_uniq(v) VALUES ('dup')")
    dup = await db.aexecute("INSERT INTO t_uniq(v) VALUES ('dup')")
    assert not dup.ok
    assert "integrity" in (dup.error or "").lower()
