"""Verify the frozen R221 API config facade and governance contract."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.contract_digest import stable_text_digest, write_text_lf  # noqa: E402

CONTRACT_PATH = ROOT / "tests" / "api_config_contract_r221.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _metadata(handler: Any) -> dict[str, Any]:
    from services.endpoint_manifest import get_metadata

    meta = get_metadata(handler)
    if meta is None:
        raise RuntimeError(f"missing endpoint metadata for {handler.__name__}")
    return {
        "auth": meta.auth_tier.value,
        "risk": meta.risk_tier.value,
        "plane": meta.route_plane.value if meta.route_plane else None,
        "summary": meta.summary,
        "description": meta.description,
        "audit": meta.audit_action,
        "scopes": list(meta.required_scopes),
    }


def build_contract() -> dict[str, Any]:
    from api import config

    handlers = (
        "config_get_handler",
        "llm_models_handler",
        "config_put_handler",
        "llm_test_handler",
        "llm_chat_handler",
    )
    patch_seams = (
        "web",
        "logger",
        "require_observability_access",
        "require_admin_token",
        "require_same_origin_if_no_token",
        "check_rate_limit",
        "build_rate_limit_response",
        "resolve_token_info",
        "emit_audit_event",
        "request_tenant_scope",
        "get_effective_config",
        "get_runtime_guardrails",
        "get_settings_schema",
        "update_config",
        "get_apply_semantics",
        "get_admin_token",
        "payload_contains_runtime_guardrails",
        "get_llm_egress_controls",
        "is_loopback_client",
        "get_client_ip",
        "resolve_model_list_target",
        "validate_model_list_target",
        "fetch_remote_model_list",
        "get_stale_cached_models",
        "_cache_get",
        "_format_llm_ssrf_error",
        "_llm_insecure_override_enabled",
        "LLMClient",
    )
    schema = config.get_settings_schema()
    route_contract = ROOT / "tests" / "api_route_contract_r220.json"
    openapi = ROOT / "docs" / "openapi.yaml"
    return {
        "schema_version": 1,
        "facade_signatures": {
            name: str(inspect.signature(getattr(config, name))) for name in handlers
        },
        "facade_metadata": {
            name: _metadata(getattr(config, name)) for name in handlers
        },
        "patch_seams": list(patch_seams),
        "provider_catalog": config.PROVIDER_CATALOG,
        "allowed_llm_keys": sorted(config.ALLOWED_LLM_KEYS),
        "model_cache": {
            "max_entries": config._MODEL_LIST_MAX_ENTRIES,
            "ttl_sec": config._MODEL_LIST_TTL_SEC,
            "exported_cache_type": type(config._MODEL_LIST_CACHE).__name__,
        },
        "settings_schema_sha256": _digest(schema),
        "apply_semantics": {
            "provider": config.get_apply_semantics(["provider"]),
            "model": config.get_apply_semantics(["model"]),
            "base_url": config.get_apply_semantics(["base_url"]),
        },
        "owned_response_matrices": {
            "config": [
                "tests.test_s66_api_config_guardrails",
                "tests.test_r53_apply_semantics",
                "tests.security.test_r99_sensitive_contract",
                "tests.test_r219_exception_boundary_phase2",
            ],
            "models": [
                "tests.test_api_model_list",
                "tests.test_r60_model_cache",
                "tests.test_r123_real_backend_model_list_lane",
                "tests.test_r155_exception_fidelity",
                "tests.test_llm_default_allowlist",
            ],
            "llm": [
                "tests.test_s28s29_chat_csrf_redaction",
                "tests.test_r219_exception_boundary_phase2",
            ],
        },
        "r220_route_contract_sha256": stable_text_digest(route_contract),
        "openapi_sha256": stable_text_digest(openapi),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    actual = build_contract()
    if args.write_baseline:
        write_text_lf(CONTRACT_PATH, _canonical_json(actual))
        print(f"API-CONFIG-CONTRACT-WRITTEN: {CONTRACT_PATH}")
        return 0
    expected = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if actual != expected:
        print("API-CONFIG-CONTRACT-FAIL: frozen config/facade contract drifted")
        return 1
    print("API-CONFIG-CONTRACT-PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
