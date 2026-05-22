"""Tests for the ``/api/v2/assets/*`` OpenAPI compat layer."""

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from mjr_am_backend.routes.handlers import api_v2_assets
from mjr_am_backend.shared import Result

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_app() -> web.Application:
    app = web.Application()
    routes = web.RouteTableDef()
    api_v2_assets.register_api_v2_asset_routes(routes)
    app.add_routes(routes)
    return app


class _FakeDB:
    """Tiny stand-in for the SQLite facade used by the compat layer."""

    def __init__(self, query_handler) -> None:
        self._handler = query_handler

    async def aquery(self, sql, params=None):
        return self._handler(sql, params)


def _row(**overrides):
    base = {
        "id": 1,
        "filename": "pic.png",
        "filepath": "/tmp/pic.png",
        "subfolder": "",
        "source": "output",
        "root_id": None,
        "kind": "image",
        "ext": ".png",
        "size": 1024,
        "mtime": 1_700_000_000,
        "width": 512,
        "height": 512,
        "duration": None,
        "created_at": "2026-05-01 12:00:00",
        "updated_at": "2026-05-02 12:00:00",
        "indexed_at": "2026-05-02 12:00:00",
        "content_hash": "blake3:" + "a" * 64,
        "phash": None,
        "hash_algo": "blake3",
        "enrichment_level": 2,
        "job_id": "job-xyz",
        "stack_id": None,
        "workflow_id": "wf-1",
        "source_node_id": "9",
        "source_node_type": "SaveImage",
        "tags": '["cat","cute"]',
        "metadata_raw": '{"foo":"bar"}',
        "workflow_type": "T2I",
        "metadata_quality": "full",
        "has_workflow": 1,
        "has_generation_data": 1,
        "generation_time_ms": 4200,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# _row_to_asset (unit, no app)
# --------------------------------------------------------------------------- #


def test_row_to_asset_shapes_fields_correctly():
    asset = api_v2_assets._row_to_asset(_row())
    assert asset["id"] == "1"
    assert asset["name"] == "pic.png"
    assert asset["size"] == 1024
    assert asset["width"] == 512
    assert asset["height"] == 512
    assert asset["mime_type"] == "image/png"
    assert asset["tags"] == ["cat", "cute"]
    assert asset["user_metadata"] == {"foo": "bar"}
    assert asset["hash"] == "blake3:" + "a" * 64
    assert asset["asset_hash"] == asset["hash"]
    assert asset["job_id"] == "job-xyz"
    assert asset["prompt_id"] == "job-xyz"  # deprecated alias
    assert asset["is_immutable"] is False
    assert asset["created_at"].startswith("2026-05-01")
    assert asset["metadata"]["workflow_id"] == "wf-1"
    assert asset["metadata"]["source_node_type"] == "SaveImage"


def test_row_to_asset_omits_hash_when_algo_not_blake3():
    asset = api_v2_assets._row_to_asset(
        _row(hash_algo="sha256", content_hash="deadbeef")
    )
    assert "hash" not in asset
    assert "asset_hash" not in asset


def test_resolve_asset_id_numeric_vs_blake3():
    where, params = api_v2_assets._resolve_asset_id("42")
    assert "a.id = ?" in where
    assert params == (42,)

    where, params = api_v2_assets._resolve_asset_id("blake3:" + "b" * 64)
    assert "content_hash" in where
    assert params == ("b" * 64, "blake3:" + "b" * 64)


# --------------------------------------------------------------------------- #
# Route-level
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_list_endpoint_returns_paginated_payload(monkeypatch):
    def _handle(sql, params):
        if "COUNT(*)" in sql:
            return Result.Ok([{"n": 3}])
        return Result.Ok([_row(id=1), _row(id=2, filename="other.jpg", ext=".jpg")])

    async def _services():
        return {"db": _FakeDB(_handle)}, None

    monkeypatch.setattr(api_v2_assets, "_require_services", _services)

    app = _build_app()
    req = make_mocked_request("GET", "/api/v2/assets?limit=2&offset=0", app=app)
    match = await app.router.resolve(req)
    resp = await match.handler(req)
    payload = json.loads(resp.text)
    assert resp.status == 200
    assert payload["total"] == 3
    assert payload["has_more"] is True
    assert len(payload["assets"]) == 2
    assert payload["assets"][0]["name"] == "pic.png"
    assert payload["assets"][1]["mime_type"] == "image/jpeg"


@pytest.mark.asyncio
async def test_get_endpoint_returns_404_when_missing(monkeypatch):
    def _handle(sql, params):
        return Result.Ok([])

    async def _services():
        return {"db": _FakeDB(_handle)}, None

    monkeypatch.setattr(api_v2_assets, "_require_services", _services)

    app = _build_app()
    req = make_mocked_request("GET", "/api/v2/assets/999", app=app)
    req.match_info["id"] = "999"
    resp = await api_v2_assets._get_asset(req)
    assert resp.status == 404


@pytest.mark.asyncio
async def test_get_endpoint_returns_asset(monkeypatch):
    def _handle(sql, params):
        return Result.Ok([_row()])

    async def _services():
        return {"db": _FakeDB(_handle)}, None

    monkeypatch.setattr(api_v2_assets, "_require_services", _services)
    req = make_mocked_request("GET", "/api/v2/assets/1")
    req.match_info["id"] = "1"
    resp = await api_v2_assets._get_asset(req)
    assert resp.status == 200
    body = json.loads(resp.text)
    assert body["id"] == "1"
    assert body["tags"] == ["cat", "cute"]


@pytest.mark.asyncio
async def test_hash_check_returns_200_on_match(monkeypatch):
    def _handle(sql, params):
        return Result.Ok([{"1": 1}])

    async def _services():
        return {"db": _FakeDB(_handle)}, None

    monkeypatch.setattr(api_v2_assets, "_require_services", _services)
    req = make_mocked_request("HEAD", f"/api/v2/assets/hash/{'a' * 64}")
    req.match_info["hash"] = "a" * 64
    resp = await api_v2_assets._check_asset_by_hash(req)
    assert resp.status == 200


@pytest.mark.asyncio
async def test_hash_check_returns_404_when_missing(monkeypatch):
    def _handle(sql, params):
        return Result.Ok([])

    async def _services():
        return {"db": _FakeDB(_handle)}, None

    monkeypatch.setattr(api_v2_assets, "_require_services", _services)
    req = make_mocked_request("HEAD", f"/api/v2/assets/hash/blake3:{'b' * 64}")
    req.match_info["hash"] = "blake3:" + "b" * 64
    resp = await api_v2_assets._check_asset_by_hash(req)
    assert resp.status == 404


@pytest.mark.asyncio
async def test_hash_check_rejects_malformed_input(monkeypatch):
    async def _services():
        return {"db": _FakeDB(lambda *_: Result.Ok([]))}, None

    monkeypatch.setattr(api_v2_assets, "_require_services", _services)
    req = make_mocked_request("HEAD", "/api/v2/assets/hash/notahash")
    req.match_info["hash"] = "notahash"
    resp = await api_v2_assets._check_asset_by_hash(req)
    assert resp.status == 404


@pytest.mark.asyncio
async def test_tags_endpoint_returns_tags(monkeypatch):
    def _handle(sql, params):
        return Result.Ok([_row()])

    async def _services():
        return {"db": _FakeDB(_handle)}, None

    monkeypatch.setattr(api_v2_assets, "_require_services", _services)
    req = make_mocked_request("GET", "/api/v2/assets/1/tags")
    req.match_info["id"] = "1"
    resp = await api_v2_assets._get_asset_tags(req)
    assert resp.status == 200
    body = json.loads(resp.text)
    assert body == {"tags": ["cat", "cute"]}


@pytest.mark.asyncio
async def test_content_endpoint_returns_404_when_path_disallowed(monkeypatch):
    def _handle(sql, params):
        return Result.Ok([_row(filepath="/nowhere/missing.png")])

    async def _services():
        return {"db": _FakeDB(_handle)}, None

    monkeypatch.setattr(api_v2_assets, "_require_services", _services)
    monkeypatch.setattr(api_v2_assets, "_is_path_allowed", lambda *a, **kw: False)

    req = make_mocked_request("GET", "/api/v2/assets/1/content")
    req.match_info["id"] = "1"
    resp = await api_v2_assets._get_asset_content(req)
    assert resp.status == 404
