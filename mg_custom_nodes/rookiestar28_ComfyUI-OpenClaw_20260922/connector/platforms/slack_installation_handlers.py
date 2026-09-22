"""Owned Slack installation, OAuth, and workspace-identity mixin."""

# ruff: noqa: UP006, UP035, UP045 -- preserve frozen facade annotations.

from typing import Any, Dict, Optional, Tuple

# mypy: disable-error-code="attr-defined,has-type,no-any-return"


class SlackInstallationMixin:
    async def handle_oauth_install(self, request):
        _, web = self._adapter_import_aiohttp_web()
        if not self._installation_manager.can_handle_oauth():
            return self._adapter_make_response(
                web, status=503, text="Slack OAuth not configured"
            )
        state = self._installation_manager.issue_install_state()
        return self._adapter_make_redirect_response(
            web, self._installation_manager.build_install_url(state)
        )

    async def handle_oauth_callback(self, request):
        _, web = self._adapter_import_aiohttp_web()
        if not self._installation_manager.can_handle_oauth():
            return self._adapter_make_response(
                web, status=503, text="Slack OAuth not configured"
            )
        query = getattr(request, "query", {}) or {}
        if query.get("error"):
            return self._adapter_make_response(
                web,
                status=400,
                text=f"Slack OAuth rejected: {query.get('error')}",
            )
        state = str(query.get("state", "") or "").strip()
        code = str(query.get("code", "") or "").strip()
        if not state or not code:
            return self._adapter_make_response(
                web, status=400, text="Missing OAuth callback fields"
            )
        if not self._installation_manager.consume_install_state(state):
            return self._adapter_make_response(
                web, status=400, text="Invalid or replayed OAuth state"
            )
        try:
            payload = await self._installation_manager.exchange_code(code)
            installation = self._installation_manager.upsert_from_oauth_payload(payload)
            return self._adapter_make_response(
                web,
                status=200,
                text=(
                    "Slack installation complete for "
                    f"{installation.workspace_id} ({installation.installation_id})."
                ),
            )
        except Exception as exc:
            safe_text = self._adapter_safe_external_error_text(
                "Slack OAuth processing failed", exc
            )
            self._adapter_logger().warning("Slack OAuth callback failed: %s", safe_text)
            return self._adapter_make_response(
                web,
                status=502,
                text=safe_text,
            )

    def _get_bot_user_id(self, payload: Dict[str, Any], workspace_id: str) -> str:
        candidate = ""
        if workspace_id and workspace_id in self._bot_user_ids:
            return self._bot_user_ids[workspace_id]
        if self._bot_user_id:
            return self._bot_user_id
        authorizations = payload.get("authorizations", [])
        if authorizations and isinstance(authorizations, list):
            candidate = str((authorizations[0] or {}).get("user_id", "") or "").strip()
            if candidate:
                self._bot_user_id = candidate
                if workspace_id:
                    self._bot_user_ids[workspace_id] = candidate
                return candidate
        if workspace_id:
            workspace_resolution, _ = (
                self._installation_manager.resolve_workspace_tokens(workspace_id)
            )
            candidate = self._installation_manager.bot_user_id_for_installation(
                workspace_resolution.installation if workspace_resolution.ok else None
            )
            if candidate:
                self._bot_user_ids[workspace_id] = candidate
                if self._bot_user_id is None:
                    self._bot_user_id = candidate
        return candidate

    def _resolve_workspace_credentials(
        self, workspace_id: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        workspace_id = str(workspace_id or "").strip()
        if workspace_id:
            resolution, tokens = self._installation_manager.resolve_workspace_tokens(
                workspace_id
            )
            if resolution.ok and resolution.installation is not None:
                bot_token = tokens.get("bot_token")
                if bot_token:
                    self._installation_manager.mark_resolution_success(
                        resolution.installation.installation_id, workspace_id
                    )
                    return (
                        resolution.installation.installation_id,
                        bot_token,
                        workspace_id,
                    )
                self._adapter_logger().warning(
                    "Slack workspace %s resolved without bot token secret", workspace_id
                )
                return (
                    resolution.installation.installation_id,
                    None,
                    workspace_id,
                )
            if (
                not self._installation_manager.oauth_enabled
                and self.config.slack_bot_token
            ):
                return (None, self.config.slack_bot_token, workspace_id)
            self._adapter_logger().warning(
                "Slack workspace resolution failed for %s: %s (%s)",
                workspace_id,
                resolution.reject_reason,
                resolution.health_code,
            )
            return (None, None, workspace_id)
        if self.config.slack_bot_token:
            return (None, self.config.slack_bot_token, "")
        return (None, None, workspace_id)

    def _handle_lifecycle_event(self, workspace_id: str, event_type: str) -> None:
        installation_id = self._installation_manager.installation_id_for_workspace(
            workspace_id
        )
        try:
            if event_type == "app_uninstalled":
                self._installation_manager.mark_installation_health(
                    installation_id,
                    health_code="revoked",
                    reason="slack_app_uninstalled",
                    details={"workspace_id": workspace_id},
                )
                self._installation_manager.uninstall_installation(
                    installation_id, reason="slack_app_uninstalled"
                )
            elif event_type == "tokens_revoked":
                self._installation_manager.mark_installation_health(
                    installation_id,
                    health_code="invalid_token",
                    reason="slack_tokens_revoked",
                    details={"workspace_id": workspace_id},
                )
            elif event_type == "app_rate_limited":
                self._installation_manager.mark_installation_health(
                    installation_id,
                    health_code="degraded",
                    reason="slack_app_rate_limited",
                    details={"workspace_id": workspace_id},
                )
        except ValueError:
            self._adapter_logger().warning(
                "Slack lifecycle event for unbound workspace %s (%s)",
                workspace_id,
                event_type,
            )
