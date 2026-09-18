"""Contract-first tests for R221 API config hotspot decomposition."""

from __future__ import annotations

import importlib.util
import inspect
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_verifier():
    path = ROOT / "scripts" / "verify_api_config_contract.py"
    spec = importlib.util.spec_from_file_location("r221_config_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load R221 config contract verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestR221ApiConfigDecomposition(unittest.TestCase):
    def test_frozen_config_facade_and_outer_contract_matches_byte_for_byte(self):
        verifier = _load_verifier()
        fixture = (ROOT / "tests" / "api_config_contract_r221.json").read_text(
            encoding="utf-8"
        )
        self.assertEqual(verifier._canonical_json(verifier.build_contract()), fixture)

    def test_all_established_patch_seams_remain_on_facade(self):
        from api import config

        verifier = _load_verifier()
        contract = verifier.build_contract()
        missing = [
            name for name in contract["patch_seams"] if not hasattr(config, name)
        ]
        self.assertEqual(missing, [])

    def test_owned_modules_are_substantive_one_way_boundaries(self):
        from api import (
            config_llm_handlers,
            config_model_handlers,
            config_projection_handlers,
        )

        expected = {
            config_projection_handlers: ("config_get_response", "config_put_response"),
            config_model_handlers: ("llm_models_response",),
            config_llm_handlers: ("llm_test_response", "llm_chat_response"),
        }
        for module, functions in expected.items():
            source = inspect.getsource(module)
            self.assertNotIn("import api.config", source)
            self.assertNotIn("from api import config", source)
            self.assertNotIn("from . import config", source)
            for name in functions:
                self.assertGreater(
                    len(inspect.getsource(getattr(module, name)).splitlines()), 20
                )

    def test_r220_route_and_openapi_digests_remain_frozen(self):
        verifier = _load_verifier()
        expected = json.loads(
            (ROOT / "tests" / "api_config_contract_r221.json").read_text(
                encoding="utf-8"
            )
        )
        actual = verifier.build_contract()
        self.assertEqual(
            actual["r220_route_contract_sha256"],
            expected["r220_route_contract_sha256"],
        )
        self.assertEqual(actual["openapi_sha256"], expected["openapi_sha256"])


if __name__ == "__main__":
    unittest.main()
