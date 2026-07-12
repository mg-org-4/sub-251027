"""Verify the frozen R223 Slack and Feishu adapter contracts."""

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

CONTRACT_PATH = ROOT / "tests" / "platform_adapter_contract_r223.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _method_signatures(owner: type) -> dict[str, str]:
    names = {
        name
        for cls in owner.__mro__
        if cls is not object
        for name, value in cls.__dict__.items()
        if callable(value)
        and (not name.startswith("__") or name == "__init__")
        and not name.startswith("_adapter_")
    }
    return {
        name: str(inspect.signature(getattr(owner, name))) for name in sorted(names)
    }


def _instance_ownership(owner: type) -> list[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(owner.__init__)))  # type: ignore[misc]
    names = set()
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            names.add(target.attr)
    return sorted(names)


def build_contract() -> dict[str, Any]:
    from connector.platforms.feishu_webhook import (
        FEISHU_DOMAIN_BASES,
        FEISHU_TOKEN_TTL_SEC,
        FEISHU_WEBHOOK_MAX_BODY_BYTES,
        FeishuDeliveryTarget,
        FeishuWebhookServer,
    )
    from connector.platforms.slack_webhook import (
        SLACK_SIGNING_VERSION,
        SLACK_TIMESTAMP_MAX_DRIFT_SEC,
        SlackWebhookServer,
    )

    router_contract = ROOT / "tests" / "connector_router_contract_r222.json"
    return {
        "schema_version": 1,
        "slack": {
            "class_constants": {
                "REPLAY_WINDOW_SEC": SlackWebhookServer.REPLAY_WINDOW_SEC,
                "NONCE_CACHE_SIZE": SlackWebhookServer.NONCE_CACHE_SIZE,
                "SLACK_SIGNING_VERSION": SLACK_SIGNING_VERSION,
                "SLACK_TIMESTAMP_MAX_DRIFT_SEC": SLACK_TIMESTAMP_MAX_DRIFT_SEC,
            },
            "method_signatures": _method_signatures(SlackWebhookServer),
            "instance_ownership": _instance_ownership(SlackWebhookServer),
            "routes": [
                ["POST", "slack_webhook_path", "handle_event"],
                ["POST", "slack_interactions_path", "handle_interaction"],
                ["GET", "slack_oauth_install_path", "handle_oauth_install"],
                ["GET", "slack_oauth_callback_path", "handle_oauth_callback"],
            ],
            "patch_seams": [
                "_import_aiohttp_web",
                "_make_response",
                "_make_json_response",
                "_make_redirect_response",
                "verify_slack_signature",
                "logger",
            ],
        },
        "feishu": {
            "class_constants": {
                "REPLAY_WINDOW_SEC": FeishuWebhookServer.REPLAY_WINDOW_SEC,
                "NONCE_CACHE_SIZE": FeishuWebhookServer.NONCE_CACHE_SIZE,
                "FEISHU_WEBHOOK_MAX_BODY_BYTES": FEISHU_WEBHOOK_MAX_BODY_BYTES,
                "FEISHU_TOKEN_TTL_SEC": FEISHU_TOKEN_TTL_SEC,
                "FEISHU_DOMAIN_BASES": FEISHU_DOMAIN_BASES,
            },
            "method_signatures": _method_signatures(FeishuWebhookServer),
            "delivery_target_signature": str(inspect.signature(FeishuDeliveryTarget)),
            "instance_ownership": _instance_ownership(FeishuWebhookServer),
            "routes": [
                ["POST", "feishu_webhook_path", "handle_event"],
                ["POST", "feishu_callback_path", "handle_callback"],
            ],
            "patch_seams": [
                "_import_aiohttp_web",
                "_make_response",
                "_make_json_response",
                "safe_request_json",
                "logger",
            ],
        },
        "response_matrix_owners": [
            "tests.test_r124_slack_ingress_contract",
            "tests.test_r125_slack_real_backend_lane",
            "tests.test_f57_slack_transport_parity",
            "tests.test_f58_slack_oauth_installations",
            "tests.test_f59_slack_interactions",
            "tests.test_f67_feishu_transport_parity",
            "tests.test_f68_feishu_installations",
            "tests.test_f69_feishu_callbacks",
            "tests.test_f74_reply_visibility_policy",
            "tests.security.test_s80_connector_ingress",
        ],
        "router_contract_digest": stable_text_digest(router_contract),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    actual = build_contract()
    if args.write_baseline:
        write_text_lf(CONTRACT_PATH, _canonical_json(actual))
        print(f"PLATFORM-ADAPTER-CONTRACT-WRITTEN: {CONTRACT_PATH}")
        return 0
    expected = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if actual != expected:
        print("PLATFORM-ADAPTER-CONTRACT-FAIL")
        return 1
    print("PLATFORM-ADAPTER-CONTRACT-PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
