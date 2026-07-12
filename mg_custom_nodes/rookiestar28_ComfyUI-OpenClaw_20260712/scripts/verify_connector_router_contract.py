"""Verify the frozen R222 CommandRouter facade and command contract."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.contract_digest import stable_text_digest, write_text_lf  # noqa: E402

CONTRACT_PATH = ROOT / "tests" / "connector_router_contract_r222.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _command_table() -> list[dict[str, Any]]:
    from connector.router import CommandRouter

    tree = ast.parse(textwrap.dedent(inspect.getsource(CommandRouter.handle)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "handlers"
            for target in node.targets
        ):
            if not isinstance(node.value, ast.Dict):
                break
            entries = []
            for key, value in zip(node.value.keys, node.value.values, strict=True):
                if not isinstance(key, ast.Tuple) or not isinstance(value, ast.Tuple):
                    raise RuntimeError("invalid command table entry")
                aliases = [ast.literal_eval(item) for item in key.elts]
                handler_attr = value.elts[0]
                command_class = value.elts[1]
                if not isinstance(handler_attr, ast.Attribute) or not isinstance(
                    command_class, ast.Attribute
                ):
                    raise RuntimeError("invalid command table target")
                entries.append(
                    {
                        "aliases": aliases,
                        "handler": handler_attr.attr,
                        "class": command_class.attr,
                    }
                )
            return entries
    raise RuntimeError("CommandRouter.handle command table not found")


def build_contract() -> dict[str, Any]:
    from connector.router import CommandRouter

    method_names = sorted(
        {
            name
            for owner in CommandRouter.__mro__
            if owner is not object
            for name, value in owner.__dict__.items()
            if callable(value)
            and (name == "handle" or name.startswith("_"))
            and name != "_build_llm_client"
        }
    )
    digests = {}
    for filename in (
        "api_config_contract_r221.json",
        "api_route_contract_r220.json",
    ):
        path = ROOT / "tests" / filename
        digests[filename] = stable_text_digest(path)
    return {
        "schema_version": 1,
        "constructor_signature": str(inspect.signature(CommandRouter)),
        "facade_signatures": {
            name: str(inspect.signature(getattr(CommandRouter, name)))
            for name in method_names
        },
        "command_table": _command_table(),
        "instance_ownership": [
            "config",
            "client",
            "poller",
            "state",
            "_template_meta_cache",
            "_rate_limiter",
            "semantic_guard",
            "command_firewall",
        ],
        "response_matrix_owners": [
            "tests.connector.test_router_hotspot_r181",
            "tests.connector.test_r214_jobs_command",
            "tests.connector.test_router_admin",
            "tests.connector.test_router_command_authz_r80",
            "tests.connector.test_security",
            "tests.connector.test_chat",
            "tests.connector.test_chat_integration",
            "tests.connector.test_media_delivery",
            "tests.chat_connector.test_router_phase2",
            "tests.chat_connector.test_router_phase3",
        ],
        "upstream_contract_digests": digests,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    actual = build_contract()
    if args.write_baseline:
        write_text_lf(CONTRACT_PATH, _canonical_json(actual))
        print(f"CONNECTOR-ROUTER-CONTRACT-WRITTEN: {CONTRACT_PATH}")
        return 0
    expected = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if actual != expected:
        print("CONNECTOR-ROUTER-CONTRACT-FAIL")
        return 1
    print("CONNECTOR-ROUTER-CONTRACT-PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
