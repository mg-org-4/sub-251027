"""
Config API handlers (R21/S13/F20).
Provides GET/PUT /moltbot/config and optional /moltbot/llm/test.
"""

from __future__ import annotations

import json
import logging

if __package__ and "." in __package__:
    from ..services.import_fallback import import_attrs_dual, import_module_dual
else:
    from services.import_fallback import (  # type: ignore
        import_attrs_dual,
        import_module_dual,
    )

try:
    (PACK_VERSION,) = import_attrs_dual(
        __package__,
        "..config",
        "config",
        ("PACK_VERSION",),
    )
except ImportError:  # pragma: no cover
    PACK_VERSION = "0.1.0"

try:
    from aiohttp import web
except ImportError:  # pragma: no cover (optional for unit tests)
    # CRITICAL test/CI fallback:
    # Some CI/unit environments intentionally run without aiohttp installed.
    # Keep this module importable by providing a minimal `web` shim used by
    # handler tests (json_response/status/body), while production keeps real aiohttp.
    class _MockResponse:
        def __init__(
            self, payload: dict, status: int = 200, headers: dict | None = None
        ):
            self.status = status
            self.headers = headers or {}
            self.body = json.dumps(payload).encode("utf-8")

    class _MockWeb:
        _IS_MOCKWEB = True

        class Request:  # pragma: no cover - typing shim only
            pass

        class Response:  # pragma: no cover - typing shim only
            pass

        @staticmethod
        def json_response(
            payload: dict, status: int = 200, headers: dict | None = None
        ):
            return _MockResponse(payload, status=status, headers=headers)

    web = _MockWeb()  # type: ignore

# Import discipline:
# - In real ComfyUI runtimes, this pack is loaded as a package and must use package-relative imports.
# - In unit tests, modules may be imported as top-level (e.g., `api.*`), so we allow top-level fallbacks.
(
    is_loopback,
    require_admin_token,
    require_observability_access,
    resolve_token_info,
) = import_attrs_dual(
    __package__,
    "..services.access_control",
    "services.access_control",
    (
        "is_loopback",
        "require_admin_token",
        "require_observability_access",
        "resolve_token_info",
    ),
)
(emit_audit_event,) = import_attrs_dual(
    __package__,
    "..services.audit",
    "services.audit",
    ("emit_audit_event",),
)

try:
    (require_same_origin_if_no_token,) = import_attrs_dual(
        __package__,
        "..services.csrf_protection",
        "services.csrf_protection",
        ("require_same_origin_if_no_token",),
    )
except Exception:
    # CRITICAL test/CI fallback (DO NOT replace with a direct import):
    # Some unit-test environments import `api.config` without aiohttp installed.
    # `services.csrf_protection` imports aiohttp at module load, which can raise
    # ModuleNotFoundError and break unrelated tests (`test_r53`, `test_r60`).
    # Keep import-time behavior resilient by using a no-op guard in that case.
    def require_same_origin_if_no_token(*_args, **_kwargs):  # type: ignore
        return None


(LLMClient,) = import_attrs_dual(
    __package__,
    "..services.llm_client",
    "services.llm_client",
    ("LLMClient",),
)
(check_rate_limit, build_rate_limit_response) = import_attrs_dual(
    __package__,
    "..services.rate_limit",
    "services.rate_limit",
    ("check_rate_limit", "build_rate_limit_response"),
)
(get_client_ip,) = import_attrs_dual(
    __package__,
    "..services.request_ip",
    "services.request_ip",
    ("get_client_ip",),
)
(
    ALLOWED_LLM_KEYS,
    get_admin_token,
    get_apply_semantics,
    get_effective_config,
    get_llm_egress_controls,
    get_runtime_guardrails,
    get_settings_schema,
    is_loopback_client,
    update_config,
) = import_attrs_dual(
    __package__,
    "..services.runtime_config",
    "services.runtime_config",
    (
        "ALLOWED_LLM_KEYS",
        "get_admin_token",
        "get_apply_semantics",
        "get_effective_config",
        "get_llm_egress_controls",
        "get_runtime_guardrails",
        "get_settings_schema",
        "is_loopback_client",
        "update_config",
    ),
)
(
    TenantBoundaryError,
    request_tenant_scope,
) = import_attrs_dual(
    __package__,
    "..services.tenant_context",
    "services.tenant_context",
    ("TenantBoundaryError", "request_tenant_scope"),
)
(
    CODE_RUNTIME_ONLY_PERSIST_FORBIDDEN,
    payload_contains_runtime_guardrails,
) = import_attrs_dual(
    __package__,
    "..services.runtime_guardrails",
    "services.runtime_guardrails",
    ("CODE_RUNTIME_ONLY_PERSIST_FORBIDDEN", "payload_contains_runtime_guardrails"),
)

logger = logging.getLogger("ComfyUI-OpenClaw.api.config")

(
    ConfigHandlerDependencies,
    config_get_response,
    config_put_response,
) = import_attrs_dual(
    __package__,
    "..api.config_projection_handlers",
    "api.config_projection_handlers",
    ("ConfigHandlerDependencies", "config_get_response", "config_put_response"),
)
(llm_models_response,) = import_attrs_dual(
    __package__,
    "..api.config_model_handlers",
    "api.config_model_handlers",
    ("llm_models_response",),
)
(llm_chat_response, llm_test_response) = import_attrs_dual(
    __package__,
    "..api.config_llm_handlers",
    "api.config_llm_handlers",
    ("llm_chat_response", "llm_test_response"),
)

(
    _MODEL_LIST_CACHE,
    _MODEL_LIST_MAX_ENTRIES,
    _MODEL_LIST_TTL_SEC,
    _build_model_cache_key,
    _cache_get,
    _cache_put,
    _extract_models_from_payload,
    _format_llm_ssrf_error,
    _get_llm_allowed_hosts,
    _llm_insecure_override_enabled,
    fetch_remote_model_list,
    get_stale_cached_models,
    resolve_model_list_target,
    validate_model_list_target,
) = import_attrs_dual(
    __package__,
    "..services.llm_model_list",
    "services.llm_model_list",
    (
        "_MODEL_LIST_CACHE",
        "_MODEL_LIST_MAX_ENTRIES",
        "_MODEL_LIST_TTL_SEC",
        "build_model_cache_key",
        "cache_get",
        "cache_put",
        "extract_models_from_payload",
        "format_llm_ssrf_error",
        "get_llm_allowed_hosts",
        "llm_insecure_override_enabled",
        "fetch_remote_model_list",
        "get_stale_cached_models",
        "resolve_model_list_target",
        "validate_model_list_target",
    ),
)


# S14/R98 / R64: Import Endpoint Metadata
(
    AuthTier,
    RiskTier,
    RoutePlane,
    endpoint_metadata,
) = import_attrs_dual(
    __package__,
    "..services.endpoint_manifest",
    "services.endpoint_manifest",
    ("AuthTier", "RiskTier", "RoutePlane", "endpoint_metadata"),
)


# Provider catalog for UI dropdown (R16 dynamic)
PROVIDER_CATALOG = []

try:
    raw_catalog_module = import_module_dual(
        __package__,
        "..services.providers.catalog",
        "services.providers.catalog",
    )
    RAW_CATALOG = raw_catalog_module.PROVIDER_CATALOG

    for pid, info in RAW_CATALOG.items():
        PROVIDER_CATALOG.append(
            {
                "id": pid,
                "label": info.name,
                "requires_key": info.env_key_name is not None,
            }
        )
    # Ensure custom is present if not in catalog (though it is)
    if not any(p["id"] == "custom" for p in PROVIDER_CATALOG):
        PROVIDER_CATALOG.append(
            {"id": "custom", "label": "Custom OpenAI-compatible", "requires_key": True}
        )
except ImportError:
    # Fallback if catalog module missing
    PROVIDER_CATALOG = [
        {"id": "openai", "label": "OpenAI", "requires_key": True},
        {"id": "anthropic", "label": "Anthropic", "requires_key": True},
        {"id": "openrouter", "label": "OpenRouter", "requires_key": True},
        {"id": "gemini", "label": "Google Gemini", "requires_key": True},
        {"id": "groq", "label": "Groq", "requires_key": True},
        {"id": "deepseek", "label": "DeepSeek", "requires_key": True},
        {"id": "xai", "label": "xAI (Grok)", "requires_key": True},
        {"id": "ollama", "label": "Ollama (Local)", "requires_key": False},
        {"id": "lmstudio", "label": "LM Studio (Local)", "requires_key": False},
        {"id": "custom", "label": "Custom OpenAI-compatible", "requires_key": True},
    ]


def _handler_dependencies():
    """Capture established facade patch seams for owned config handlers."""

    return ConfigHandlerDependencies(
        web=web,
        logger=logger,
        provider_catalog=PROVIDER_CATALOG,
        pack_version=PACK_VERSION,
        require_observability_access=require_observability_access,
        require_admin_token=require_admin_token,
        require_same_origin_if_no_token=require_same_origin_if_no_token,
        resolve_token_info=resolve_token_info,
        emit_audit_event=emit_audit_event,
        check_rate_limit=check_rate_limit,
        build_rate_limit_response=build_rate_limit_response,
        get_client_ip=get_client_ip,
        is_loopback=is_loopback,
        get_admin_token=get_admin_token,
        get_apply_semantics=get_apply_semantics,
        get_effective_config=get_effective_config,
        get_llm_egress_controls=get_llm_egress_controls,
        get_runtime_guardrails=get_runtime_guardrails,
        get_settings_schema=get_settings_schema,
        is_loopback_client=is_loopback_client,
        update_config=update_config,
        tenant_boundary_error=TenantBoundaryError,
        request_tenant_scope=request_tenant_scope,
        runtime_only_code=CODE_RUNTIME_ONLY_PERSIST_FORBIDDEN,
        payload_contains_runtime_guardrails=payload_contains_runtime_guardrails,
        model_cache_get=_cache_get,
        format_llm_ssrf_error=_format_llm_ssrf_error,
        llm_insecure_override_enabled=_llm_insecure_override_enabled,
        fetch_remote_model_list=fetch_remote_model_list,
        get_stale_cached_models=get_stale_cached_models,
        resolve_model_list_target=resolve_model_list_target,
        validate_model_list_target=validate_model_list_target,
        llm_client=LLMClient,
    )


@endpoint_metadata(
    auth=AuthTier.OBSERVABILITY,
    risk=RiskTier.LOW,
    summary="Get configuration",
    description="Returns effective config, sources, and provider catalog.",
    audit="config.read",
    plane=RoutePlane.ADMIN,
)
async def config_get_handler(request: web.Request) -> web.Response:
    """
    GET /moltbot/config
    Returns effective config, sources, and provider catalog.
    Enforced by S14 Access Control.
    """
    # CRITICAL: owned implementation performs require_observability_access before reads.
    return await config_get_response(request, _handler_dependencies())


@endpoint_metadata(
    auth=AuthTier.ADMIN,
    risk=RiskTier.LOW,  # Read-only external fetch, but admin-gated
    summary="List remote models",
    description="Fetch a remote model list (best-effort) for OpenAI-compatible providers.",
    audit="llm.list_models",
    plane=RoutePlane.ADMIN,
)
async def llm_models_handler(request: web.Request) -> web.Response:
    """
    GET /openclaw/llm/models (legacy: /moltbot/llm/models)
    Fetch a remote model list (best-effort) for OpenAI-compatible providers.

    Security:
    - admin boundary
    - loopback-only unless OPENCLAW_ALLOW_REMOTE_ADMIN=1
    - SSRF policy enforced via LLM egress controls, including scoped private-network allowance
    """
    # CRITICAL: owned implementation performs require_admin_token( before network access.
    # CRITICAL S65: fetch_remote_model_list remains the safe_request_json egress owner.
    return await llm_models_response(request, _handler_dependencies())


@endpoint_metadata(
    auth=AuthTier.ADMIN,
    risk=RiskTier.HIGH,
    summary="Update configuration",
    description="Updates non-secret LLM config.",
    audit="config.update",
    plane=RoutePlane.ADMIN,
)
async def config_put_handler(request: web.Request) -> web.Response:
    """
    PUT /moltbot/config
    Updates non-secret LLM config. Protected by admin boundary (S13) + CSRF (S26+).
    """
    # CRITICAL: owned implementation performs require_admin_token( before mutation.
    return await config_put_response(request, _handler_dependencies())


@endpoint_metadata(
    auth=AuthTier.ADMIN,
    risk=RiskTier.MEDIUM,
    summary="Test LLM connection",
    description="Tests LLM connection using provided or stored credentials.",
    audit="llm.test_connection",
    plane=RoutePlane.ADMIN,
)
async def llm_test_handler(request: web.Request) -> web.Response:
    """
    POST /moltbot/llm/test
    Tests LLM connection. Protected by admin boundary (S13) + CSRF (S26+).
    """
    # CRITICAL: owned implementation performs require_admin_token( before provider access.
    return await llm_test_response(request, _handler_dependencies())


@endpoint_metadata(
    auth=AuthTier.ADMIN,
    risk=RiskTier.MEDIUM,  # Consumed by connector, costs money/tokens
    summary="Chat completion",
    description="Run a simple chat completion using server-side LLM config.",
    audit="llm.chat_completion",
    plane=RoutePlane.ADMIN,
)
async def llm_chat_handler(request: web.Request) -> web.Response:
    """
    POST /openclaw/llm/chat (legacy: /moltbot/llm/chat)
    Run a simple chat completion using server-side LLM config + keys.
    This endpoint is intended for the connector; no prompt content is logged.
    """
    # CRITICAL: owned implementation performs require_admin_token( before provider access.
    return await llm_chat_response(request, _handler_dependencies())
