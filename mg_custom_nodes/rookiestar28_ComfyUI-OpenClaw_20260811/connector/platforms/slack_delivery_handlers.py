"""Owned Slack response and media-delivery mixin."""

# ruff: noqa: SIM117, UP006, UP035, UP045 -- preserve frozen behavior/signatures.

from typing import Any, Dict, Optional

from ..reply_visibility import decide_reply_visibility

# mypy: disable-error-code="attr-defined,no-any-return"


class SlackDeliveryMixin:
    async def _send_interactive_reply(
        self,
        *,
        channel_id: str,
        text: str,
        buttons: list[dict],
        thread_ts: str = "",
        delivery_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a Slack Block Kit message with bounded button actions."""
        try:
            import aiohttp as _aiohttp
        except ImportError:
            self._adapter_logger().warning(
                "aiohttp not available; cannot send Slack interactive reply"
            )
            return

        ctx = dict(delivery_context or {})
        if not thread_ts:
            thread_ts = str(ctx.get("thread_id", "") or "").strip()
        installation_id, bot_token, workspace_id = self._resolve_workspace_credentials(
            str(ctx.get("workspace_id", "") or "").strip()
        )
        if not bot_token:
            self._adapter_logger().warning(
                "Slack interactive reply dropped: no workspace token available (workspace=%s)",
                workspace_id or "legacy",
            )
            return

        elements: list[dict] = []
        for idx, button in enumerate(buttons[:5]):
            value = str(button.get("value", "") or "").strip()
            if not value:
                continue
            label = str(button.get("label", "") or "OpenClaw").strip()[:75]
            action_id = str(
                button.get("action_type")
                or button.get("action_id")
                or f"openclaw.{idx}"
            ).strip()[:255]
            element: Dict[str, Any] = {
                "type": "button",
                "text": {"type": "plain_text", "text": label or "OpenClaw"},
                "value": value[:2000],
                "action_id": action_id or f"openclaw.{idx}",
            }
            style = self._adapter_style_to_slack(str(button.get("style", "") or ""))
            if style:
                element["style"] = style
            elements.append(element)
        if not elements:
            if text:
                await self._send_reply(
                    channel_id=channel_id,
                    text=text,
                    thread_ts=thread_ts,
                    delivery_context=ctx,
                )
            return

        payload: Dict[str, Any] = {
            "channel": channel_id,
            "text": text or "OpenClaw",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (text or "OpenClaw")[:3000],
                    },
                },
                {"type": "actions", "elements": elements},
            ],
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts

        headers = {
            "Authorization": f"Bearer {bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        try:
            async with _aiohttp.ClientSession() as session:
                async with session.post(
                    "https://slack.com/api/chat.postMessage",
                    json=payload,
                    headers=headers,
                    timeout=_aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        if installation_id:
                            self._installation_manager.mark_api_error(
                                installation_id,
                                error_code=f"http_{resp.status}",
                                status_code=resp.status,
                                details={
                                    "workspace_id": workspace_id,
                                    "path": "chat.postMessage",
                                    "interactive": True,
                                },
                            )
                        return
                    data = await resp.json()
                    if not data.get("ok"):
                        if installation_id:
                            self._installation_manager.mark_api_error(
                                installation_id,
                                error_code=str(data.get("error", "unknown")),
                                details={
                                    "workspace_id": workspace_id,
                                    "path": "chat.postMessage",
                                    "interactive": True,
                                },
                            )
                    elif installation_id:
                        self._installation_manager.mark_installation_health(
                            installation_id,
                            health_code="ok",
                            reason="chat_post_message_interactive_ok",
                            details={"workspace_id": workspace_id},
                        )
        except Exception as e:
            self._adapter_logger().warning("Slack interactive reply failed: %s", e)

    async def _send_reply(
        self,
        channel_id: str,
        text: str,
        thread_ts: str = "",
        delivery_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a message via Slack Web API (chat.postMessage)."""
        ctx = dict(delivery_context or {})
        if not thread_ts:
            thread_ts = str(ctx.get("thread_id", "") or "").strip()
        decision = decide_reply_visibility(
            delivery_context=ctx,
            platform="slack",
            channel_kind=self._adapter_channel_kind(channel_id),
            in_thread=bool(thread_ts),
            text=text,
        )
        if decision.suppressed:
            self._adapter_logger().info(
                "Suppressed Slack reply channel=%s reason=%s",
                channel_id,
                decision.reason,
            )
            return
        try:
            import aiohttp as _aiohttp
        except ImportError:
            self._adapter_logger().warning(
                "aiohttp not available; cannot send Slack reply"
            )
            return

        installation_id, bot_token, workspace_id = self._resolve_workspace_credentials(
            str(ctx.get("workspace_id", "") or "").strip()
        )
        if not bot_token:
            self._adapter_logger().warning(
                "Slack reply dropped: no workspace token available (workspace=%s)",
                workspace_id or "legacy",
            )
            return

        url = "https://slack.com/api/chat.postMessage"
        headers = {
            "Authorization": f"Bearer {bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        payload: Dict[str, Any] = {
            "channel": channel_id,
            "text": text,
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts

        try:
            async with _aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=_aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        if installation_id:
                            self._installation_manager.mark_api_error(
                                installation_id,
                                error_code=f"http_{resp.status}",
                                status_code=resp.status,
                                details={
                                    "workspace_id": workspace_id,
                                    "path": "chat.postMessage",
                                },
                            )
                        self._adapter_logger().warning(
                            f"Slack API error: status={resp.status} body={body[:200]}"
                        )
                    else:
                        data = await resp.json()
                        if not data.get("ok"):
                            if installation_id:
                                self._installation_manager.mark_api_error(
                                    installation_id,
                                    error_code=str(data.get("error", "unknown")),
                                    details={
                                        "workspace_id": workspace_id,
                                        "path": "chat.postMessage",
                                    },
                                )
                            self._adapter_logger().warning(
                                f"Slack API error: {data.get('error', 'unknown')}"
                            )
                        elif installation_id:
                            self._installation_manager.mark_installation_health(
                                installation_id,
                                health_code="ok",
                                reason="chat_post_message_ok",
                                details={"workspace_id": workspace_id},
                            )
        except Exception as e:
            self._adapter_logger().warning(f"Slack reply failed: {e}")

    # ------------------------------------------------------------------
    # Platform contract: send_message / send_image
    # ------------------------------------------------------------------

    async def send_message(
        self,
        channel_id: str,
        text: str,
        delivery_context: Optional[Dict[str, Any]] = None,
    ):
        """Platform contract: send text message."""
        await self._send_reply(
            channel_id=channel_id,
            text=text,
            delivery_context=delivery_context,
        )

    async def send_image(
        self,
        channel_id: str,
        image_data: bytes,
        filename: str = "image.png",
        caption: Optional[str] = None,
        delivery_context: Optional[Dict[str, Any]] = None,
    ):
        """Platform contract: send image (Slack files.upload)."""
        try:
            import aiohttp as _aiohttp
        except ImportError:
            self._adapter_logger().warning(
                "aiohttp not available; cannot upload Slack image"
            )
            return

        ctx = dict(delivery_context or {})
        thread_ts = str(ctx.get("thread_id", "") or "").strip()
        installation_id, bot_token, workspace_id = self._resolve_workspace_credentials(
            str(ctx.get("workspace_id", "") or "").strip()
        )
        if not bot_token:
            self._adapter_logger().warning(
                "Slack image dropped: no workspace token available (workspace=%s)",
                workspace_id or "legacy",
            )
            return

        url = "https://slack.com/api/files.upload"
        headers = {
            "Authorization": f"Bearer {bot_token}",
        }
        data = _aiohttp.FormData()
        data.add_field("file", image_data, filename=filename, content_type="image/png")
        data.add_field("channels", channel_id)
        if caption:
            data.add_field("initial_comment", caption)
        if thread_ts:
            data.add_field("thread_ts", thread_ts)

        try:
            async with _aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=data,
                    headers=headers,
                    timeout=_aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        if installation_id:
                            self._installation_manager.mark_api_error(
                                installation_id,
                                error_code=f"http_{resp.status}",
                                status_code=resp.status,
                                details={
                                    "workspace_id": workspace_id,
                                    "path": "files.upload",
                                },
                            )
                        self._adapter_logger().warning(
                            f"Slack file upload error: status={resp.status}"
                        )
                    else:
                        resp_data = await resp.json()
                        if not resp_data.get("ok"):
                            if installation_id:
                                self._installation_manager.mark_api_error(
                                    installation_id,
                                    error_code=str(resp_data.get("error", "unknown")),
                                    details={
                                        "workspace_id": workspace_id,
                                        "path": "files.upload",
                                    },
                                )
                            self._adapter_logger().warning(
                                f"Slack file upload error: {resp_data.get('error')}"
                            )
                        elif installation_id:
                            self._installation_manager.mark_installation_health(
                                installation_id,
                                health_code="ok",
                                reason="files_upload_ok",
                                details={"workspace_id": workspace_id},
                            )
        except Exception as e:
            self._adapter_logger().warning(f"Slack image upload failed: {e}")
