"""Contract-first tests for R222 connector router decomposition."""

from __future__ import annotations

import importlib.util
import inspect
import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _verifier():
    path = ROOT / "scripts" / "verify_connector_router_contract.py"
    spec = importlib.util.spec_from_file_location("r222_router_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load R222 verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestR222ConnectorRouterDecomposition(unittest.TestCase):
    def test_frozen_router_contract_matches_byte_for_byte(self):
        verifier = _verifier()
        fixture = (ROOT / "tests" / "connector_router_contract_r222.json").read_text(
            encoding="utf-8"
        )
        self.assertEqual(verifier._canonical_json(verifier.build_contract()), fixture)

    def test_owned_family_modules_are_substantive_and_one_way(self):
        from connector import (
            router_admin_handlers,
            router_chat_handlers,
            router_dispatch,
            router_execution_handlers,
        )

        required = {
            router_dispatch: (
                router_dispatch.RouterDispatchMixin,
                ("handle", "_check_command_authz"),
            ),
            router_execution_handlers: (
                router_execution_handlers.RouterExecutionMixin,
                ("_handle_run", "_handle_interrupt"),
            ),
            router_admin_handlers: (
                router_admin_handlers.RouterAdminMixin,
                ("_handle_jobs", "_handle_approvals_list"),
            ),
            router_chat_handlers: (
                router_chat_handlers.RouterChatMixin,
                ("_handle_chat", "_chat_general"),
            ),
        }
        for module, (owner, names) in required.items():
            source = inspect.getsource(module)
            self.assertNotIn("import connector.router", source)
            self.assertNotIn("from connector import router", source)
            self.assertNotIn("from . import router", source)
            for name in names:
                self.assertGreater(
                    len(inspect.getsource(getattr(owner, name)).splitlines()), 12
                )

    def test_command_aliases_are_unique_and_canonical_first(self):
        contract = _verifier().build_contract()
        seen = set()
        for entry in contract["command_table"]:
            self.assertTrue(entry["aliases"][0].startswith("/"))
            for alias in entry["aliases"]:
                self.assertNotIn(alias, seen)
                seen.add(alias)

    def test_dispatch_request_context_is_immutable(self):
        from connector.config import CommandClass
        from connector.contract import CommandRequest
        from connector.router_dispatch import RouterRequestContext

        request = CommandRequest(
            platform="telegram",
            channel_id="channel",
            sender_id="sender",
            username="user",
            message_id="message",
            text="/status",
            timestamp=0.0,
        )
        context = RouterRequestContext(
            request=request,
            parsed_command="/status",
            canonical_command="/status",
            args=(),
            command_class=CommandClass.PUBLIC,
        )
        self.assertIsInstance(context.args, tuple)
        with self.assertRaises(FrozenInstanceError):
            context.canonical_command = "/run"

    def test_upstream_route_and_config_contracts_are_unchanged(self):
        verifier = _verifier()
        expected = json.loads(
            (ROOT / "tests" / "connector_router_contract_r222.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            verifier.build_contract()["upstream_contract_digests"],
            expected["upstream_contract_digests"],
        )


if __name__ == "__main__":
    unittest.main()
