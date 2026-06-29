"""
Operator-facing runtime config snapshot projection helpers.
"""

from __future__ import annotations

import os
from typing import Any, Callable


class RuntimeConfig:
    """
    Typed configuration snapshot.
    Aggregates effective settings from layered config sources.
    """

    def __init__(
        self,
        *,
        llm: dict[str, Any],
        runtime_guardrails: dict[str, Any],
        bridge_enabled: bool,
        allow_any_public_llm_host: bool,
        allow_insecure_base_url: bool,
        webhook_auth_mode: str,
        security_dangerous_bind_override: bool,
        admin_token_configured: bool,
    ):
        self.llm = llm
        self.runtime_guardrails = runtime_guardrails
        self.bridge_enabled = bridge_enabled
        self.allow_any_public_llm_host = allow_any_public_llm_host
        self.allow_insecure_base_url = allow_insecure_base_url
        self.webhook_auth_mode = webhook_auth_mode
        self.security_dangerous_bind_override = security_dangerous_bind_override
        self.admin_token_configured = admin_token_configured


def build_runtime_config_snapshot(
    *,
    get_effective_config: Callable[[], tuple[dict[str, Any], dict[str, str]]],
    get_runtime_guardrails_snapshot: Callable[[], dict[str, Any]],
    env_flag: Callable[[str, str, bool], bool],
    get_admin_token: Callable[[], str],
) -> RuntimeConfig:
    llm, _ = get_effective_config()
    return RuntimeConfig(
        llm=llm,
        runtime_guardrails=get_runtime_guardrails_snapshot(),
        bridge_enabled=env_flag(
            "OPENCLAW_BRIDGE_ENABLED",
            "MOLTBOT_BRIDGE_ENABLED",
            False,
        ),
        allow_any_public_llm_host=env_flag(
            "OPENCLAW_ALLOW_ANY_PUBLIC_LLM_HOST",
            "MOLTBOT_ALLOW_ANY_PUBLIC_LLM_HOST",
            False,
        ),
        allow_insecure_base_url=env_flag(
            "OPENCLAW_ALLOW_INSECURE_BASE_URL",
            "MOLTBOT_ALLOW_INSECURE_BASE_URL",
            False,
        ),
        webhook_auth_mode=os.environ.get("OPENCLAW_WEBHOOK_AUTH_MODE", ""),
        security_dangerous_bind_override=env_flag(
            "OPENCLAW_SECURITY_DANGEROUS_BIND_OVERRIDE",
            "MOLTBOT_SECURITY_DANGEROUS_BIND_OVERRIDE",
            False,
        ),
        admin_token_configured=bool(get_admin_token()),
    )
