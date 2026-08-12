"""Owned PromptServer route registration and startup orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any


@dataclass(frozen=True)
class RouteRegistrationDependencies:
    build_core_route_specs: Callable[..., Any]
    build_assist_route_specs: Callable[..., Any]
    build_connector_installation_route_specs: Callable[..., Any]
    build_pack_route_specs: Callable[..., Any]
    register_route_family: Callable[..., None]
    register_dual_route: Callable[..., None]
    core_handlers: dict[str, Any]
    assist: Any
    connector_installation_handlers: dict[str, Any] | None
    run_mae_startup_gate: Callable[[Any], None]


def register_dual_route(
    server: Any,
    method: str,
    path: str,
    handler: Any,
    *,
    metrics: Any = None,
    legacy_headers_builder: Any = None,
) -> None:
    """Register PromptServer and direct aliases with one legacy wrapper."""

    if not callable(handler):
        print(
            f"[OpenClaw] Warning: Skipping route {method} {path} because handler is missing (None)."
        )
        return
    actual_handler = handler
    if path.startswith("/moltbot"):

        @wraps(handler)
        async def _deprecated_handler(request: Any) -> Any:
            try:
                if metrics:
                    metrics.inc("legacy_api_hits")
            except Exception:
                pass
            print(
                f"[OpenClaw] DEPRECATION WARNING: Legacy route accessed: {request.path}. Please migrate to /openclaw/* equivalents."
            )
            response = await handler(request)
            if legacy_headers_builder:
                headers = legacy_headers_builder(getattr(request, "path", path))
                response_headers = getattr(response, "headers", None)
                if (
                    headers
                    and response_headers is not None
                    and hasattr(response_headers, "update")
                ):
                    response_headers.update(headers)
            return response

        actual_handler = _deprecated_handler

    registrar = (
        getattr(server.routes, method.lower(), None)
        if method in {"GET", "POST", "PUT", "DELETE"}
        else None
    )
    if registrar is not None:
        registrar(path)(actual_handler)
    if hasattr(server, "app") and hasattr(server.app, "router"):
        for target in (path, "/api" + path):
            try:
                # IMPORTANT: direct aliases must retain the same legacy wrapper.
                server.app.router.add_route(method, target, actual_handler)
            except RuntimeError:
                pass
            except Exception as exc:
                print(
                    f"[OpenClaw] Warning: Failed to register fallback route {target}: {exc}"
                )


def run_mae_startup_gate(server: Any, resolve_profile: Callable[[], str]) -> None:
    """Validate the registered OpenClaw route posture for the active profile."""

    if not hasattr(server, "app"):
        return
    try:
        if __package__ and "." in __package__:
            from ..services.endpoint_manifest import (
                generate_manifest,
                validate_mae_posture,
            )
        else:
            from services.endpoint_manifest import (
                generate_manifest,
                validate_mae_posture,
            )
    except Exception as exc:
        print(f"[OpenClaw] Warning: S60 MAE gate unavailable: {exc}")
        return
    profile = resolve_profile()
    manifest = generate_manifest(server.app)
    scoped_manifest = [
        entry for entry in manifest if _is_openclaw_managed_path(entry.get("path", ""))
    ]
    ok, violations = validate_mae_posture(scoped_manifest, profile=profile)
    if ok:
        return
    message = "S60 MAE posture validation failed:\n" + "\n".join(
        f"- {item}" for item in violations
    )
    if profile in {"public", "hardened"}:
        raise RuntimeError(message)
    print(f"[OpenClaw] Warning: {message}")


def _is_openclaw_managed_path(path: str) -> bool:
    if not isinstance(path, str):
        return False
    return path.startswith(
        (
            "/openclaw",
            "/moltbot",
            "/api/openclaw",
            "/api/moltbot",
            "/bridge",
            "/api/bridge",
        )
    )


def _register_bridge(server: Any) -> None:
    try:
        try:
            from ..api.bridge import register_bridge_routes
            from ..services.modules import ModuleCapability, is_module_enabled
        except (ImportError, ValueError):
            from api.bridge import register_bridge_routes
            from services.modules import ModuleCapability, is_module_enabled
        if hasattr(server, "app") and is_module_enabled(ModuleCapability.BRIDGE):
            register_bridge_routes(server.app)
            print("[OpenClaw] Bridge routes registered")
        elif not is_module_enabled(ModuleCapability.BRIDGE):
            print("[OpenClaw] Bridge module disabled; skipping route registration")
    except ImportError:
        pass


def _register_packs(
    server: Any, prefixes: tuple[str, ...], deps: RouteRegistrationDependencies
) -> None:
    try:
        try:
            from ..api.packs import PacksHandlers
        except (ImportError, ValueError):
            from api.packs import PacksHandlers
        try:
            from ..config import DATA_DIR
        except (ImportError, ValueError):
            from config import DATA_DIR
        packs = PacksHandlers(DATA_DIR)
        for prefix in prefixes:
            deps.register_route_family(
                server,
                deps.register_dual_route,
                deps.build_pack_route_specs(prefix, packs),
            )
    except ImportError:
        pass


def register_route_families(server: Any, deps: RouteRegistrationDependencies) -> None:
    """Register all route families in the frozen R220 exposure order."""

    prefixes = ("/openclaw", "/moltbot")
    for prefix in prefixes:
        deps.register_route_family(
            server,
            deps.register_dual_route,
            deps.build_core_route_specs(prefix, deps.core_handlers),
        )
    if deps.assist:
        for prefix in prefixes:
            deps.register_route_family(
                server,
                deps.register_dual_route,
                deps.build_assist_route_specs(prefix, deps.assist),
            )
    if deps.connector_installation_handlers is not None:
        for prefix in prefixes:
            deps.register_route_family(
                server,
                deps.register_dual_route,
                deps.build_connector_installation_route_specs(
                    prefix, deps.connector_installation_handlers
                ),
            )
    _register_bridge(server)
    deps.run_mae_startup_gate(server)
    _register_packs(server, prefixes, deps)
