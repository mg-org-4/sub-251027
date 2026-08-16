"""Contract-first tests for R223 Slack/Feishu adapter decomposition."""

from __future__ import annotations

import importlib.util
import inspect
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _verifier():
    path = ROOT / "scripts" / "verify_platform_adapter_contract.py"
    spec = importlib.util.spec_from_file_location("r223_adapter_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load R223 verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestR223PlatformAdapterDecomposition(unittest.TestCase):
    def test_frozen_platform_contract_matches_byte_for_byte(self):
        verifier = _verifier()
        fixture = (ROOT / "tests" / "platform_adapter_contract_r223.json").read_text(
            encoding="utf-8"
        )
        self.assertEqual(verifier._canonical_json(verifier.build_contract()), fixture)

    def test_platform_owner_modules_are_substantive_and_one_way(self):
        from connector.platforms import (
            feishu_delivery_handlers,
            feishu_ingress_handlers,
            feishu_installation_handlers,
            slack_delivery_handlers,
            slack_ingress_handlers,
            slack_installation_handlers,
        )

        modules = (
            slack_installation_handlers,
            slack_ingress_handlers,
            slack_delivery_handlers,
            feishu_installation_handlers,
            feishu_ingress_handlers,
            feishu_delivery_handlers,
        )
        for module in modules:
            source = inspect.getsource(module)
            self.assertGreater(len(source.splitlines()), 40)
            self.assertNotIn("from . import slack_webhook", source)
            self.assertNotIn("from . import feishu_webhook", source)
            self.assertNotIn("import connector.platforms.slack_webhook", source)
            self.assertNotIn("import connector.platforms.feishu_webhook", source)

    def test_platforms_do_not_share_protocol_implementation_owner(self):
        expected = json.loads(
            (ROOT / "tests" / "platform_adapter_contract_r223.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotEqual(expected["slack"]["routes"], expected["feishu"]["routes"])

    def test_facade_patch_seams_remain_present(self):
        from connector.platforms import feishu_webhook, slack_webhook

        expected = json.loads(
            (ROOT / "tests" / "platform_adapter_contract_r223.json").read_text(
                encoding="utf-8"
            )
        )
        for seam in expected["slack"]["patch_seams"]:
            self.assertTrue(
                callable(getattr(slack_webhook, seam, None)) or seam == "logger"
            )
        for seam in expected["feishu"]["patch_seams"]:
            self.assertTrue(
                callable(getattr(feishu_webhook, seam, None)) or seam == "logger"
            )

    def test_r222_router_contract_digest_is_unchanged(self):
        verifier = _verifier()
        expected = json.loads(
            (ROOT / "tests" / "platform_adapter_contract_r223.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            verifier.build_contract()["router_contract_digest"],
            expected["router_contract_digest"],
        )


if __name__ == "__main__":
    unittest.main()
