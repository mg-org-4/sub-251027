"""Contract-first tests for R220 API route hotspot decomposition."""

from __future__ import annotations

import importlib.util
import inspect
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _load_verifier():
    path = ROOT / "scripts" / "verify_api_route_contract.py"
    spec = importlib.util.spec_from_file_location("r220_route_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load R220 route contract verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestR220ApiRouteDecomposition(unittest.TestCase):
    def test_frozen_route_facade_and_openapi_contract_matches(self):
        verifier = _load_verifier()
        fixture_text = (ROOT / "tests" / "api_route_contract_r220.json").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            verifier._canonical_json(verifier.build_contract()), fixture_text
        )

    def test_owned_modules_are_real_one_way_boundaries(self):
        from api import route_handlers, route_orchestration

        handlers_source = inspect.getsource(route_handlers)
        orchestration_source = inspect.getsource(route_orchestration)
        for source in (handlers_source, orchestration_source):
            self.assertNotIn("import api.routes", source)
            self.assertNotIn("from api import routes", source)
            self.assertNotIn("from . import routes", source)

        for name in (
            "health_response",
            "logs_tail_response",
            "jobs_response",
            "trace_response",
        ):
            implementation = getattr(route_handlers, name)
            self.assertGreater(len(inspect.getsource(implementation).splitlines()), 12)

        for name in ("register_dual_route", "register_route_families"):
            implementation = getattr(route_orchestration, name)
            self.assertGreater(len(inspect.getsource(implementation).splitlines()), 12)

    def test_facade_exports_keep_exact_signatures_and_metadata(self):
        verifier = _load_verifier()
        expected = json.loads(
            (ROOT / "tests" / "api_route_contract_r220.json").read_text(
                encoding="utf-8"
            )
        )
        actual = verifier.build_contract()
        self.assertEqual(actual["facade_signatures"], expected["facade_signatures"])
        self.assertEqual(actual["facade_metadata"], expected["facade_metadata"])

    def test_owned_orchestrator_preserves_family_and_gate_order(self):
        from api import route_orchestration

        events = []

        def build(family):
            return lambda prefix, *_args: f"{family}:{prefix}"

        deps = route_orchestration.RouteRegistrationDependencies(
            build_core_route_specs=build("core"),
            build_assist_route_specs=build("assist"),
            build_connector_installation_route_specs=build("connector"),
            build_pack_route_specs=build("packs"),
            register_route_family=lambda _server, _register, specs: events.append(
                specs
            ),
            register_dual_route=lambda *_args: None,
            core_handlers={},
            assist=object(),
            connector_installation_handlers={},
            run_mae_startup_gate=lambda _server: events.append("mae"),
        )
        with (
            patch.object(
                route_orchestration,
                "_register_bridge",
                side_effect=lambda _server: events.append("bridge"),
            ),
            patch.object(
                route_orchestration,
                "_register_packs",
                side_effect=lambda *_args: events.append("packs"),
            ),
        ):
            route_orchestration.register_route_families(SimpleNamespace(), deps)

        self.assertEqual(
            events,
            [
                "core:/openclaw",
                "core:/moltbot",
                "assist:/openclaw",
                "assist:/moltbot",
                "connector:/openclaw",
                "connector:/moltbot",
                "bridge",
                "mae",
                "packs",
            ],
        )

    def test_optional_facade_families_remain_conditionally_absent(self):
        from api import route_orchestration

        events = []
        deps = route_orchestration.RouteRegistrationDependencies(
            build_core_route_specs=lambda prefix, *_args: f"core:{prefix}",
            build_assist_route_specs=lambda *_args: self.fail("assist must be absent"),
            build_connector_installation_route_specs=lambda *_args: self.fail(
                "connector must be absent"
            ),
            build_pack_route_specs=lambda *_args: (),
            register_route_family=lambda _server, _register, specs: events.append(
                specs
            ),
            register_dual_route=lambda *_args: None,
            core_handlers={},
            assist=None,
            connector_installation_handlers=None,
            run_mae_startup_gate=lambda _server: events.append("mae"),
        )
        with (
            patch.object(
                route_orchestration,
                "_register_bridge",
                side_effect=lambda _server: events.append("bridge"),
            ),
            patch.object(
                route_orchestration,
                "_register_packs",
                side_effect=lambda *_args: events.append("packs"),
            ),
        ):
            route_orchestration.register_route_families(SimpleNamespace(), deps)
        self.assertEqual(
            events,
            ["core:/openclaw", "core:/moltbot", "bridge", "mae", "packs"],
        )

    def test_dual_registration_preserves_every_supported_method_and_alias_order(self):
        from api.route_orchestration import register_dual_route

        standard = []
        direct = []

        class Routes:
            def __getattr__(self, method):
                return lambda path: lambda _handler: standard.append(
                    (method.upper(), path)
                )

        server = SimpleNamespace(
            routes=Routes(),
            app=SimpleNamespace(
                router=SimpleNamespace(
                    add_route=lambda method, path, _handler: direct.append(
                        (method, path)
                    )
                )
            ),
        )

        async def handler(_request):
            return None

        for method in ("GET", "POST", "PUT", "DELETE"):
            register_dual_route(server, method, f"/openclaw/{method.lower()}", handler)

        self.assertEqual(
            standard,
            [
                (method, f"/openclaw/{method.lower()}")
                for method in ("GET", "POST", "PUT", "DELETE")
            ],
        )
        self.assertEqual(
            direct,
            [
                (method, alias)
                for method in ("GET", "POST", "PUT", "DELETE")
                for alias in (
                    f"/openclaw/{method.lower()}",
                    f"/api/openclaw/{method.lower()}",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
