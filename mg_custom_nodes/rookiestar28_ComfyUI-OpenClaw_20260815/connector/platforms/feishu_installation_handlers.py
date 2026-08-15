"""Owned Feishu installation, tenant-token, and bot-identity mixin."""

# ruff: noqa: UP006, UP035, UP045 -- preserve frozen facade annotations.

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from services.connector_installation_registry import InstallationResolution
from services.safe_io import STANDARD_OUTBOUND_POLICY, SafeIOHTTPError

from .feishu_installation_manager import FeishuBinding

# mypy: disable-error-code="attr-defined,no-any-return"


class FeishuInstallationMixin:
    def _resolve_inbound_binding(self, payload: Dict[str, Any]) -> FeishuBinding:
        header = payload.get("header") or {}
        verification_token = (
            str(payload.get("token", "") or "").strip()
            or str(header.get("token", "") or "").strip()
            or str(((payload.get("event") or {}).get("token")) or "").strip()
        )
        workspace_id = str(header.get("tenant_key", "") or "").strip()
        return self._installation_manager.resolve_inbound_binding(
            verification_token=verification_token,
            workspace_id=workspace_id,
            account_id=self._bound_account_id,
        )

    def _cache_key_for_binding(self, binding: FeishuBinding) -> str:
        return binding.installation_id or binding.account_id

    def _cached_bot_open_id(self, binding: FeishuBinding) -> str:
        return (
            self._bot_open_ids.get(self._cache_key_for_binding(binding), "")
            or self._bot_open_id
        )

    def _resolve_delivery_binding(
        self, *, workspace_id: str = "", account_id: str = ""
    ) -> Tuple[InstallationResolution, Optional[FeishuBinding], Dict[str, str]]:
        return self._installation_manager.resolve_binding(
            workspace_id=workspace_id,
            account_id=account_id or self._bound_account_id,
        )

    async def _get_tenant_access_token(
        self,
        *,
        binding: Optional[FeishuBinding] = None,
        workspace_id: str = "",
        account_id: str = "",
    ) -> str:
        resolution, effective_binding, secrets = self._resolve_delivery_binding(
            workspace_id=workspace_id,
            account_id=account_id or (binding.account_id if binding else ""),
        )
        if effective_binding is None or not resolution.ok:
            raise RuntimeError(
                f"feishu_binding_resolution_failed:{resolution.reject_reason or 'missing_binding'}"
            )
        cache_key = self._cache_key_for_binding(effective_binding)
        if self._tenant_access_tokens.get(
            cache_key
        ) and self._tenant_access_token_expires_at.get(cache_key, 0.0) > (
            time.time() + 30
        ):
            return self._tenant_access_tokens[cache_key]
        app_secret = str(
            secrets.get("app_secret", "") or effective_binding.app_secret
        ).strip()
        payload = {
            "app_id": effective_binding.app_id,
            "app_secret": app_secret,
        }
        url = f"{self._adapter_resolve_domain_base(effective_binding.domain)}/open-apis/auth/v3/tenant_access_token/internal"
        try:
            data = self._adapter_safe_request_json(
                method="POST",
                url=url,
                json_body=payload,
                headers={"Accept": "application/json"},
                content_type="application/json; charset=utf-8",
                timeout_sec=15,
                allow_hosts=self._adapter_allowed_api_hosts(effective_binding.domain),
                policy=STANDARD_OUTBOUND_POLICY,
            )
        except SafeIOHTTPError as exc:
            if resolution.installation is not None:
                self._installation_manager.mark_api_error(
                    resolution.installation.installation_id,
                    error_code=exc.reason,
                    status_code=exc.status_code,
                    details={"phase": "tenant_access_token"},
                )
            raise RuntimeError(
                f"feishu_token_fetch_failed:{exc.status_code}:{exc.reason}"
            ) from exc
        if data.get("code", 0) != 0:
            if resolution.installation is not None:
                self._installation_manager.mark_api_error(
                    resolution.installation.installation_id,
                    error_code=str(data.get("msg", "unknown") or "unknown"),
                    status_code=200,
                    details={"phase": "tenant_access_token"},
                )
            raise RuntimeError(
                f"feishu_token_fetch_failed:200:{data.get('msg', 'unknown')}"
            )
        token = str(data.get("tenant_access_token", "") or "").strip()
        if not token:
            raise RuntimeError("feishu_token_fetch_failed:missing_token")
        expire = int(
            data.get("expire", self._adapter_token_ttl_sec())
            or self._adapter_token_ttl_sec()
        )
        self._tenant_access_tokens[cache_key] = token
        self._tenant_access_token_expires_at[cache_key] = time.time() + max(60, expire)
        if resolution.installation is not None:
            self._installation_manager.mark_resolution_success(
                resolution.installation.installation_id,
                effective_binding.workspace_id,
            )
        return token

    async def _fetch_bot_open_id(
        self,
        *,
        binding: Optional[FeishuBinding] = None,
        workspace_id: str = "",
        account_id: str = "",
        allow_degrade: bool = False,
    ) -> str:
        resolution, effective_binding, _ = self._resolve_delivery_binding(
            workspace_id=workspace_id,
            account_id=account_id or (binding.account_id if binding else ""),
        )
        if effective_binding is None or not resolution.ok:
            return ""
        cache_key = self._cache_key_for_binding(effective_binding)
        if self._bot_open_ids.get(cache_key):
            return self._bot_open_ids[cache_key]
        token = await self._get_tenant_access_token(binding=effective_binding)
        url = f"{self._adapter_resolve_domain_base(effective_binding.domain)}/open-apis/bot/v3/info"
        try:
            data = self._adapter_safe_request_json(
                method="GET",
                url=url,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                timeout_sec=15,
                allow_hosts=self._adapter_allowed_api_hosts(effective_binding.domain),
                policy=STANDARD_OUTBOUND_POLICY,
            )
        except SafeIOHTTPError as exc:
            if resolution.installation is not None:
                self._installation_manager.mark_api_error(
                    resolution.installation.installation_id,
                    error_code=exc.reason,
                    status_code=exc.status_code,
                    details={"phase": "bot_info"},
                )
            if allow_degrade:
                return ""
            return ""
        if data.get("code", 0) != 0:
            if resolution.installation is not None:
                self._installation_manager.mark_api_error(
                    resolution.installation.installation_id,
                    error_code=str(data.get("msg", "unknown") or "unknown"),
                    status_code=200,
                    details={"phase": "bot_info"},
                )
            return ""
        bot_open_id = str(
            (((data.get("data") or {}).get("bot") or {}).get("open_id")) or ""
        ).strip()
        if bot_open_id:
            self._bot_open_ids[cache_key] = bot_open_id
            self._bot_open_id = bot_open_id
        return bot_open_id

    async def prime_bot_identity(self) -> None:
        try:
            await self._fetch_bot_open_id(
                account_id=self._bound_account_id
                or str(self.config.feishu_account_id or "").strip()
                or str(self.config.feishu_default_account_id or "").strip(),
                workspace_id=str(self.config.feishu_workspace_id or "").strip(),
                allow_degrade=True,
            )
        except Exception as exc:
            self._adapter_logger().debug("Feishu bot identity fetch failed: %s", exc)
