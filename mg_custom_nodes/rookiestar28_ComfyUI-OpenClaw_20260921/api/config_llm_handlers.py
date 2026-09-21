"""Owned LLM connection-test and chat handler implementations."""

from __future__ import annotations

from typing import Any

from .config_projection_handlers import ConfigHandlerDependencies


async def llm_test_response(request: Any, deps: ConfigHandlerDependencies) -> Any:
    """Run the existing tenant-scoped, audited LLM connection test."""

    if deps.web is None:
        raise RuntimeError("aiohttp not available")
    try:
        from ..services.async_utils import run_in_thread
    except ImportError:
        from services.async_utils import run_in_thread
    admin_token_configured = bool(deps.get_admin_token())
    response = deps.require_same_origin_if_no_token(request, admin_token_configured)
    if response:
        return response
    if not deps.check_rate_limit(request, "admin"):
        return deps.build_rate_limit_response(
            request,
            "admin",
            web_module=deps.web,
            error="Rate limit exceeded",
            include_ok=True,
        )
    token_info = deps.resolve_token_info(request)
    allowed, error = deps.require_admin_token(request)
    if not allowed:
        deps.emit_audit_event(
            action="llm.test_connection",
            target="llm",
            outcome="deny",
            token_info=token_info,
            status_code=403,
            details={"reason": error or "unauthorized"},
            request=request,
        )
        return deps.web.json_response(
            {"ok": False, "error": error or "Unauthorized"}, status=403
        )
    try:
        with deps.request_tenant_scope(
            request=request, token_info=token_info, allow_default_when_missing=True
        ) as tenant:
            try:
                body = await request.json()
                if body is None:
                    body = {}
            except Exception:
                body = {}
            if body and not isinstance(body, dict):
                return deps.web.json_response(
                    {"ok": False, "error": "Expected JSON object body (or empty body)"},
                    status=400,
                )
            provider = (
                body.get("provider") if isinstance(body.get("provider"), str) else None
            )
            model = body.get("model") if isinstance(body.get("model"), str) else None
            base_url = (
                body.get("base_url") if isinstance(body.get("base_url"), str) else None
            )
            timeout_val = body.get("timeout_sec")
            timeout_sec = None
            if (
                isinstance(timeout_val, (int, float, str))
                and str(timeout_val).strip() != ""
            ):
                try:
                    timeout_sec = int(timeout_val)
                except (TypeError, ValueError, OverflowError):
                    return deps.web.json_response(
                        {"ok": False, "error": "timeout_sec must be an integer"},
                        status=400,
                    )
            retries_val = body.get("max_retries")
            max_retries = None
            if (
                isinstance(retries_val, (int, float, str))
                and str(retries_val).strip() != ""
            ):
                try:
                    max_retries = int(retries_val)
                except (TypeError, ValueError, OverflowError):
                    return deps.web.json_response(
                        {"ok": False, "error": "max_retries must be an integer"},
                        status=400,
                    )
            client = deps.llm_client(
                provider=provider,
                base_url=base_url,
                model=model,
                timeout=timeout_sec,
                max_retries=max_retries,
            )
            result = await run_in_thread(
                client.complete,
                system="You are a test assistant.",
                user_message="Respond with exactly: OK",
                max_tokens=10,
            )
            if result and "text" in result:
                deps.emit_audit_event(
                    action="llm.test_connection",
                    target=f"{client.provider}:{client.model}",
                    outcome="allow",
                    token_info=token_info,
                    status_code=200,
                    details={
                        "tenant_id": tenant.tenant_id,
                        "provider": client.provider,
                        "model": client.model,
                    },
                    request=request,
                )
                return deps.web.json_response(
                    {
                        "ok": True,
                        "tenant_id": tenant.tenant_id,
                        "message": "Connection successful",
                        "response": result["text"].strip(),
                        "provider": client.provider,
                        "model": client.model,
                    }
                )
            deps.emit_audit_event(
                action="llm.test_connection",
                target=f"{client.provider}:{client.model}",
                outcome="error",
                token_info=token_info,
                status_code=500,
                details={
                    "tenant_id": tenant.tenant_id,
                    "provider": client.provider,
                    "model": client.model,
                    "error": "Empty response",
                },
                request=request,
            )
            return deps.web.json_response(
                {"ok": False, "error": "Empty or invalid response from LLM"}
            )
    except deps.tenant_boundary_error as exc:
        deps.emit_audit_event(
            action="llm.test_connection",
            target="llm",
            outcome="deny",
            token_info=token_info,
            status_code=403,
            details={"reason": exc.code},
            request=request,
        )
        return deps.web.json_response(
            {"ok": False, "error": exc.code, "message": str(exc)}, status=403
        )
    except Exception as exc:
        deps.logger.error("LLM test failed (error_type=%s)", type(exc).__name__)
        deps.emit_audit_event(
            action="llm.test_connection",
            target="llm",
            outcome="error",
            token_info=token_info,
            status_code=500,
            details={"error": "llm_test_failed"},
            request=request,
        )
        return deps.web.json_response(
            {"ok": False, "error": "llm_test_failed"}, status=500
        )


async def llm_chat_response(request: Any, deps: ConfigHandlerDependencies) -> Any:
    """Run server-side tenant-scoped chat without logging prompt content."""

    if deps.web is None:
        raise RuntimeError("aiohttp not available")
    try:
        from ..services.async_utils import run_in_thread
    except ImportError:
        from services.async_utils import run_in_thread
    try:
        from ..services.provider_errors import ProviderHTTPError
    except ImportError:
        from services.provider_errors import ProviderHTTPError
    admin_token_configured = bool(deps.get_admin_token())
    response = deps.require_same_origin_if_no_token(request, admin_token_configured)
    if response:
        return response
    if not deps.check_rate_limit(request, "admin"):
        return deps.build_rate_limit_response(
            request,
            "admin",
            web_module=deps.web,
            error="Rate limit exceeded",
            include_ok=True,
        )
    token_info = deps.resolve_token_info(request)
    allowed, error = deps.require_admin_token(request)
    if not allowed:
        return deps.web.json_response(
            {"ok": False, "error": error or "Unauthorized"}, status=403
        )
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return deps.web.json_response(
            {"ok": False, "error": "Expected JSON object body"}, status=400
        )
    system = body.get("system") if isinstance(body.get("system"), str) else ""
    user_message = (
        body.get("user_message")
        if isinstance(body.get("user_message"), str)
        else body.get("message") if isinstance(body.get("message"), str) else ""
    )
    temperature = (
        body.get("temperature")
        if isinstance(body.get("temperature"), (int, float))
        else 0.7
    )
    max_tokens = (
        body.get("max_tokens") if isinstance(body.get("max_tokens"), int) else 1024
    )
    if not user_message:
        return deps.web.json_response(
            {"ok": False, "error": "missing_user_message"}, status=400
        )
    deps.logger.debug(
        "llm_chat: has_system=%s msg_len=%d temperature=%.2f max_tokens=%d",
        bool(system),
        len(user_message),
        temperature,
        max_tokens,
    )
    try:
        with deps.request_tenant_scope(
            request=request, token_info=token_info, allow_default_when_missing=True
        ) as tenant:
            client = deps.llm_client()

            def _run():
                return client.complete(
                    system=system,
                    user_message=user_message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            result = await run_in_thread(_run)
            text = result.get("text") or "" if isinstance(result, dict) else ""
            return deps.web.json_response(
                {"ok": True, "tenant_id": tenant.tenant_id, "text": text}
            )
    except deps.tenant_boundary_error as exc:
        return deps.web.json_response(
            {"ok": False, "error": exc.code, "message": str(exc)}, status=403
        )
    except ValueError as exc:
        return deps.web.json_response({"ok": False, "error": str(exc)}, status=400)
    except ProviderHTTPError as exc:
        payload = {
            "ok": False,
            "error": f"{exc.provider} HTTP {exc.status_code}: {exc.message}",
            "provider": exc.provider,
            "status_code": exc.status_code,
        }
        if getattr(exc, "retry_after", None):
            payload["retry_after"] = exc.retry_after
        return deps.web.json_response(payload, status=exc.status_code)
    except Exception as exc:
        deps.logger.warning(
            "LLM chat request failed: ***REDACTED*** (error_type=%s)",
            type(exc).__name__,
        )
        return deps.web.json_response(
            {"ok": False, "error": "llm_request_failed"}, status=500
        )
