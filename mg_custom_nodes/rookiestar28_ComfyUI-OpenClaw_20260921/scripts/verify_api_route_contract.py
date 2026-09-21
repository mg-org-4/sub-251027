"""Verify the frozen R220 API route/facade contract."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.contract_digest import stable_text_digest, write_text_lf  # noqa: E402

CONTRACT_PATH = ROOT / "tests" / "api_route_contract_r220.json"


class _NamedHandler:
    def __init__(self, name: str) -> None:
        self.__name__ = name


class _AttributeHandlers:
    def __getattr__(self, name: str) -> _NamedHandler:
        return _NamedHandler(name)


def _handler_map() -> defaultdict[str, _NamedHandler]:
    return defaultdict(lambda: _NamedHandler("unknown"))


def _normalize_specs(specs: Any) -> list[dict[str, str]]:
    return [
        {
            "method": spec.method,
            "path": spec.path,
            "handler": getattr(spec.handler, "__name__", type(spec.handler).__name__),
        }
        for spec in specs
    ]


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
    from api import routes
    from api.route_registrars import (
        build_assist_route_specs,
        build_connector_installation_route_specs,
        build_core_route_specs,
        build_pack_route_specs,
    )

    core_handlers = _handler_map()
    core_keys = inspect.getsource(build_core_route_specs)
    for key in {part.split('"', 1)[0] for part in core_keys.split('handlers["')[1:]}:
        core_handlers[key] = _NamedHandler(key)

    connector_handlers = _handler_map()
    connector_keys = inspect.getsource(build_connector_installation_route_specs)
    for key in {
        part.split('"', 1)[0] for part in connector_keys.split('handlers["')[1:]
    }:
        connector_handlers[key] = _NamedHandler(key)

    packs = _AttributeHandlers()
    assist = _AttributeHandlers()
    families: dict[str, list[dict[str, str]]] = {}
    for prefix in ("/openclaw", "/moltbot"):
        families[f"core:{prefix}"] = _normalize_specs(
            build_core_route_specs(prefix, core_handlers)
        )
        families[f"assist:{prefix}"] = _normalize_specs(
            build_assist_route_specs(prefix, assist)
        )
        families[f"connector_installations:{prefix}"] = _normalize_specs(
            build_connector_installation_route_specs(prefix, connector_handlers)
        )
        families[f"packs:{prefix}"] = _normalize_specs(
            build_pack_route_specs(prefix, packs)
        )

    facade_names = (
        "health_handler",
        "_ensure_observability_deps_ready",
        "logs_tail_handler",
        "jobs_handler",
        "_emit_jobs_list_audit",
        "trace_handler",
        "register_dual_route",
        "_resolve_mae_profile",
        "_run_mae_startup_gate",
        "register_routes",
    )
    facade = {
        name: str(inspect.signature(getattr(routes, name))) for name in facade_names
    }
    metadata = {
        name: _metadata(getattr(routes, name))
        for name in (
            "health_handler",
            "logs_tail_handler",
            "jobs_handler",
            "trace_handler",
        )
    }
    return {
        "schema_version": 1,
        "registration_order": [
            "startup_profile_gate",
            "core:/openclaw",
            "core:/moltbot",
            "assist:/openclaw",
            "assist:/moltbot",
            "connector_installations:/openclaw",
            "connector_installations:/moltbot",
            "bridge",
            "mae_posture_gate",
            "packs:/openclaw",
            "packs:/moltbot",
        ],
        "feature_conditions": {
            "assist": "assist is truthy",
            "connector_installations": "list handler is truthy",
            "bridge": "server.app exists and BRIDGE module is enabled",
            "packs": "optional pack imports succeed",
        },
        "direct_alias_rule": "each registered path also attempts path and /api+path",
        "legacy_rule": "moltbot handlers retain telemetry and deprecation headers",
        "families": families,
        "facade_signatures": facade,
        "facade_metadata": metadata,
        "openapi_sha256": stable_text_digest(ROOT / "docs" / "openapi.yaml"),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    actual = build_contract()
    if args.write_baseline:
        write_text_lf(CONTRACT_PATH, _canonical_json(actual))
        print(f"API-ROUTE-CONTRACT-WRITTEN: {CONTRACT_PATH}")
        return 0
    expected = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if actual != expected:
        print("API-ROUTE-CONTRACT-FAIL: frozen route/facade contract drifted")
        return 1
    print("API-ROUTE-CONTRACT-PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
