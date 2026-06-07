import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestStartupLifecycleDiagnostics(unittest.TestCase):
    def setUp(self):
        from services.startup_lifecycle import reset_startup_lifecycle_for_tests

        reset_startup_lifecycle_for_tests()

    def tearDown(self):
        from services.startup_lifecycle import reset_startup_lifecycle_for_tests

        reset_startup_lifecycle_for_tests()

    def test_optional_warmup_timeout_degrades_without_blocking_ready(self):
        from services.startup_lifecycle import (
            get_startup_diagnostics,
            mark_startup_ready,
            start_optional_warmups,
        )

        release = threading.Event()

        def slow_warmup():
            release.wait(timeout=1.0)

        mark_startup_ready("routes")
        started_at = time.monotonic()
        start_optional_warmups([("slow_provider", slow_warmup, 0.01)])
        elapsed = time.monotonic() - started_at

        self.assertLess(elapsed, 0.05)

        deadline = time.monotonic() + 1.0
        diagnostics = get_startup_diagnostics()
        while time.monotonic() < deadline:
            diagnostics = get_startup_diagnostics()
            if diagnostics["warmups"]["slow_provider"]["state"] == "timed_out":
                break
            time.sleep(0.01)

        release.set()
        self.assertEqual(diagnostics["state"], "degraded-warmup")
        self.assertEqual(diagnostics["ready"], True)
        self.assertEqual(diagnostics["warmups"]["slow_provider"]["state"], "timed_out")

    def test_fatal_startup_state_is_distinct_from_warmup_degradation(self):
        from services.startup_lifecycle import (
            get_startup_diagnostics,
            mark_startup_fatal,
        )

        mark_startup_fatal("security_gate", RuntimeError("blocked"))
        diagnostics = get_startup_diagnostics()

        self.assertEqual(diagnostics["state"], "fatal-startup")
        self.assertFalse(diagnostics["ready"])
        self.assertEqual(diagnostics["fatal"]["phase"], "security_gate")
        self.assertIn("RuntimeError", diagnostics["fatal"]["error_type"])


class _DummyRoutes:
    def __init__(self):
        self.calls = []

    def _decorator(self, method, path):
        def _wrap(handler):
            self.calls.append((method, path, handler))
            return handler

        return _wrap

    def get(self, path):
        return self._decorator("GET", path)

    def post(self, path):
        return self._decorator("POST", path)

    def put(self, path):
        return self._decorator("PUT", path)

    def delete(self, path):
        return self._decorator("DELETE", path)


class _DummyRouter:
    def __init__(self):
        self.calls = []

    def add_route(self, method, path, handler):
        self.calls.append((method, path, handler))

    def add_post(self, path, handler):
        self.calls.append(("POST", path, handler))

    def add_get(self, path, handler):
        self.calls.append(("GET", path, handler))


class _DummyBridgeHandlers:
    def __init__(self, submit_service=None):
        self.submit_service = submit_service

    async def submit_handler(self, request=None):
        return request

    async def deliver_handler(self, request=None):
        return request

    async def health_handler(self, request=None):
        return request


class TestRouteBootstrapWarmupBoundary(unittest.TestCase):
    def setUp(self):
        from services.startup_lifecycle import reset_startup_lifecycle_for_tests

        reset_startup_lifecycle_for_tests()

    def tearDown(self):
        from services.startup_lifecycle import reset_startup_lifecycle_for_tests

        reset_startup_lifecycle_for_tests()

    def test_full_registration_marks_ready_before_optional_warmup_finishes(self):
        from services import route_bootstrap
        from services.startup_lifecycle import get_startup_diagnostics

        release = threading.Event()

        def slow_warmup():
            release.wait(timeout=1.0)

        app = SimpleNamespace(router=_DummyRouter())
        server = SimpleNamespace(routes=_DummyRoutes(), app=app)

        contract = {
            "register_routes": lambda server: setattr(server, "core_routes", True),
            "register_preset_routes": lambda app: setattr(app, "presets", True),
            "register_schedule_routes": lambda app, require_admin_token_fn=None: setattr(
                app, "schedules", True
            ),
            "BridgeHandlers": _DummyBridgeHandlers,
            "register_trigger_routes": lambda app, **kwargs: setattr(
                app, "triggers", True
            ),
            "register_approval_routes": lambda app, **kwargs: setattr(
                app, "approvals", True
            ),
        }

        with (
            patch(
                "services.route_bootstrap_contract.load_route_bootstrap_contract",
                return_value=contract,
            ),
            patch("services.scheduler.runner.get_scheduler_runner") as get_runner,
            patch("services.scheduler.runner.start_scheduler"),
            patch(
                "services.route_bootstrap._build_optional_startup_warmups",
                return_value=[("slow_provider", slow_warmup, 0.5)],
                create=True,
            ),
        ):
            get_runner.return_value = MagicMock()
            started_at = time.monotonic()
            route_bootstrap._do_full_registration(server)
            elapsed = time.monotonic() - started_at

        diagnostics = get_startup_diagnostics()
        release.set()

        self.assertLess(elapsed, 0.5)
        self.assertTrue(server.core_routes)
        self.assertTrue(app.triggers)
        self.assertTrue(app.approvals)
        self.assertTrue(diagnostics["ready"])
        self.assertIn(
            diagnostics["warmups"]["slow_provider"]["state"], {"running", "succeeded"}
        )

    def test_register_routes_once_marks_fatal_when_required_startup_fails(self):
        from services import route_bootstrap
        from services.startup_lifecycle import get_startup_diagnostics

        with (
            patch.object(route_bootstrap, "_routes_registered", False),
            patch.object(route_bootstrap, "_register_plugins_and_shutdown_hooks"),
            patch.object(
                route_bootstrap,
                "_initialize_registries_and_security_gate",
                side_effect=RuntimeError("security blocked"),
            ),
        ):
            with self.assertRaises(RuntimeError):
                route_bootstrap.register_routes_once()

        diagnostics = get_startup_diagnostics()
        self.assertEqual(diagnostics["state"], "fatal-startup")
        self.assertFalse(diagnostics["ready"])
        self.assertEqual(diagnostics["fatal"]["phase"], "required_startup")


if __name__ == "__main__":
    unittest.main()
