"""Tests for the rate-limit state machine in security_rate_limit.py."""

from __future__ import annotations

import time
from typing import cast

import pytest
from aiohttp import web
from mjr_am_backend.routes.core.security_rate_limit import (
    _check_rate_limit,
    _reset_rate_limit_state_for_tests,
)


class _DummyTransport:
    def __init__(self, peername: tuple[str, int] = ("10.0.0.1", 9999)):
        self._peername = peername

    def get_extra_info(self, key: str):
        if key == "peername":
            return self._peername
        return None


class _FakeRequest:
    def __init__(self, *, ip: str = "10.0.0.1", port: int = 9999):
        self.headers: dict[str, str] = {}
        self.transport = _DummyTransport((ip, port))
        self.remote = ip
        self.app: dict = {}


def _make_req(ip: str = "10.0.0.1", port: int = 9999) -> web.Request:
    """Return a fake aiohttp Request cast to satisfy mypy."""
    return cast(web.Request, _FakeRequest(ip=ip, port=port))


@pytest.fixture(autouse=True)
def _clean_state():
    _reset_rate_limit_state_for_tests()
    yield
    _reset_rate_limit_state_for_tests()


class TestCheckRateLimit:
    """Unit tests for _check_rate_limit."""

    def test_allows_under_limit(self) -> None:
        req = _make_req()
        allowed, retry = _check_rate_limit(req, "search", max_requests=5, window_seconds=60)
        assert allowed is True
        assert retry is None

    def test_blocks_after_limit_exceeded(self) -> None:
        req = _make_req()
        for _ in range(5):
            allowed, _ = _check_rate_limit(req, "search", max_requests=5, window_seconds=60)
            assert allowed is True

        allowed, retry = _check_rate_limit(req, "search", max_requests=5, window_seconds=60)
        assert allowed is False
        assert isinstance(retry, int)
        assert retry >= 1

    def test_independent_endpoints(self) -> None:
        req = _make_req()
        for _ in range(3):
            _check_rate_limit(req, "ep_a", max_requests=3, window_seconds=60)

        allowed, _ = _check_rate_limit(req, "ep_a", max_requests=3, window_seconds=60)
        assert allowed is False

        allowed, _ = _check_rate_limit(req, "ep_b", max_requests=3, window_seconds=60)
        assert allowed is True

    def test_independent_clients(self) -> None:
        req_a = _make_req(ip="10.0.0.1")
        req_b = _make_req(ip="10.0.0.2")
        for _ in range(3):
            _check_rate_limit(req_a, "search", max_requests=3, window_seconds=60)

        allowed_a, _ = _check_rate_limit(req_a, "search", max_requests=3, window_seconds=60)
        assert allowed_a is False

        allowed_b, _ = _check_rate_limit(req_b, "search", max_requests=3, window_seconds=60)
        assert allowed_b is True

    def test_window_expiry_resets_counter(self) -> None:
        req = _make_req()
        for _ in range(3):
            _check_rate_limit(req, "search", max_requests=3, window_seconds=1)

        allowed, _ = _check_rate_limit(req, "search", max_requests=3, window_seconds=1)
        assert allowed is False

        time.sleep(1.1)

        allowed, _ = _check_rate_limit(req, "search", max_requests=3, window_seconds=1)
        assert allowed is True

    def test_client_eviction_on_overflow(self) -> None:
        for i in range(5):
            req = _make_req(ip=f"192.168.0.{i}")
            _check_rate_limit(req, "x", max_requests=100, window_seconds=60, _max_clients=3)

        req_first = _make_req(ip="192.168.0.0")
        allowed, _ = _check_rate_limit(
            req_first, "x", max_requests=1, window_seconds=60, _max_clients=3
        )
        assert allowed is True, "oldest client should have been evicted, counter reset"

    def test_download_rate_limit_response_or_none(self) -> None:
        from mjr_am_backend.features.assets.download_service import (
            download_rate_limit_response_or_none,
        )

        req = _make_req()

        def is_loopback(r: object) -> bool:
            return False

        check_rl = _check_rate_limit

        for _ in range(30):
            resp = download_rate_limit_response_or_none(
                req,
                preview=False,
                is_loopback_request=is_loopback,
                check_rate_limit=check_rl,
            )
            assert resp is None

        resp = download_rate_limit_response_or_none(
            req,
            preview=False,
            is_loopback_request=is_loopback,
            check_rate_limit=check_rl,
        )
        assert isinstance(resp, web.Response)
        assert resp.status == 429

    def test_loopback_bypasses_rate_limit(self) -> None:
        from mjr_am_backend.features.assets.download_service import (
            download_rate_limit_response_or_none,
        )

        req = _make_req()

        for _ in range(50):
            resp = download_rate_limit_response_or_none(
                req,
                preview=False,
                is_loopback_request=lambda r: True,
                check_rate_limit=_check_rate_limit,
            )
            assert resp is None
