"""Owned Feishu card, response, and media-delivery mixin."""

# ruff: noqa: UP006, UP035, UP045 -- preserve frozen facade annotations.

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import Any, Dict, Optional

from services.safe_io import STANDARD_OUTBOUND_POLICY, SafeIOHTTPError

from ..reply_visibility import decide_reply_visibility
from .feishu_installation_manager import FeishuBinding

# mypy: disable-error-code="attr-defined,no-any-return"


@dataclass
class FeishuDeliveryTarget:
    channel_id: str
    reply_to_message_id: str = ""
    workspace_id: str = ""
    account_id: str = ""


class FeishuDeliveryMixin:
    def _build_card_button_value(
        self,
        button: Dict[str, Any],
        *,
        target: FeishuDeliveryTarget,
        binding: FeishuBinding,
        signing_secret: str,
    ) -> Dict[str, Any]:
        contract = self._callback_contract_for_binding(
            binding=binding,
            signing_secret=signing_secret,
        )
        command_text = str(button.get("value", "") or "").strip()
        callback_payload = {
            "label": str(button.get("label", "") or "").strip(),
            "command": command_text,
            "approval_id": str(button.get("approval_id", "") or "").strip(),
            "workspace_id": target.workspace_id or binding.workspace_id,
            "account_id": target.account_id or binding.account_id,
            "channel_id": target.channel_id,
            "message_id": target.reply_to_message_id,
        }
        envelope = contract.build_envelope(
            request_id=secrets.token_hex(12),
            workspace_id=callback_payload["workspace_id"],
            action_type=self._adapter_infer_callback_action_type(command_text, button),
            payload=callback_payload,
        )
        return {
            "callback_envelope": dict(envelope.__dict__),
            "payload": callback_payload,
        }

    def _build_interactive_card(
        self,
        target: FeishuDeliveryTarget,
        text: str,
        buttons: list[dict],
        *,
        binding: FeishuBinding,
        secrets: Dict[str, str],
    ) -> Dict[str, Any]:
        signing_secret = str(
            secrets.get("app_secret", "") or binding.app_secret or ""
        ).strip()
        if not signing_secret:
            raise RuntimeError("feishu_callback_signing_secret_missing")
        actions = []
        for button in buttons[:6]:
            command_text = str(button.get("value", "") or "").strip()
            if not command_text:
                continue
            actions.append(
                {
                    "tag": "button",
                    "type": str(button.get("style", "") or "default"),
                    "text": {
                        "tag": "plain_text",
                        "content": str(button.get("label", "") or "OpenClaw"),
                    },
                    "value": self._build_card_button_value(
                        button,
                        target=target,
                        binding=binding,
                        signing_secret=signing_secret,
                    ),
                }
            )
        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": "blue",
                "title": {"tag": "plain_text", "content": "OpenClaw"},
            },
            "elements": [
                {"tag": "markdown", "content": text or "OpenClaw"},
                {"tag": "action", "actions": actions},
            ],
        }

    async def _send_interactive_reply(
        self,
        target: FeishuDeliveryTarget,
        text: str,
        buttons: list[dict],
    ) -> None:
        resolution, binding, secrets = self._resolve_delivery_binding(
            workspace_id=target.workspace_id,
            account_id=target.account_id,
        )
        if binding is None or not resolution.ok:
            self._adapter_logger().warning(
                "Feishu interactive reply dropped: no workspace binding available (%s / %s)",
                target.workspace_id or "no-workspace",
                target.account_id or "no-account",
            )
            return
        token = await self._get_tenant_access_token(
            binding=binding,
            workspace_id=target.workspace_id,
            account_id=target.account_id,
        )
        api_base = self._adapter_resolve_domain_base(binding.domain)
        card = self._build_interactive_card(
            target,
            text,
            buttons,
            binding=binding,
            secrets=secrets,
        )
        payload = {
            "content": json.dumps(card, ensure_ascii=False),
            "msg_type": "interactive",
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        if target.reply_to_message_id:
            url = (
                f"{api_base}/open-apis/im/v1/messages/"
                f"{target.reply_to_message_id}/reply"
            )
        else:
            url = f"{api_base}/open-apis/im/v1/messages?receive_id_type=chat_id"
            payload["receive_id"] = target.channel_id
        try:
            data = self._adapter_safe_request_json(
                method="POST",
                url=url,
                json_body=payload,
                headers=headers,
                content_type="application/json; charset=utf-8",
                timeout_sec=15,
                allow_hosts=self._adapter_allowed_api_hosts(binding.domain),
                policy=STANDARD_OUTBOUND_POLICY,
            )
        except SafeIOHTTPError as exc:
            if resolution.installation is not None:
                self._installation_manager.mark_api_error(
                    resolution.installation.installation_id,
                    error_code=exc.reason,
                    status_code=exc.status_code,
                    details={"phase": "interactive_reply"},
                )
            self._adapter_logger().warning(
                "Feishu interactive reply failed: status=%s", exc.status_code
            )
            return
        if data.get("code", 0) != 0:
            if resolution.installation is not None:
                self._installation_manager.mark_api_error(
                    resolution.installation.installation_id,
                    error_code=str(data.get("msg", "unknown") or "unknown"),
                    status_code=200,
                    details={"phase": "interactive_reply"},
                )
            self._adapter_logger().warning(
                "Feishu interactive reply failed: %s", data.get("msg", "unknown")
            )

    async def _send_reply(
        self,
        target: FeishuDeliveryTarget,
        text: str,
        *,
        delivery_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = dict(delivery_context or {})
        if target.workspace_id:
            ctx.setdefault("workspace_id", target.workspace_id)
        if target.account_id:
            ctx.setdefault("account_id", target.account_id)
        if target.reply_to_message_id:
            ctx.setdefault("thread_id", target.reply_to_message_id)
        decision = decide_reply_visibility(
            delivery_context=ctx,
            platform="feishu",
            channel_kind=str(ctx.get("chat_type", "") or ""),
            in_thread=bool(target.reply_to_message_id),
            text=text,
        )
        if decision.suppressed:
            self._adapter_logger().info(
                "Suppressed Feishu reply channel=%s reason=%s",
                target.channel_id,
                decision.reason,
            )
            return
        resolution, binding, _ = self._resolve_delivery_binding(
            workspace_id=target.workspace_id,
            account_id=target.account_id,
        )
        if binding is None or not resolution.ok:
            self._adapter_logger().warning(
                "Feishu reply dropped: no workspace binding available (%s / %s)",
                target.workspace_id or "no-workspace",
                target.account_id or "no-account",
            )
            return
        token = await self._get_tenant_access_token(
            binding=binding,
            workspace_id=target.workspace_id,
            account_id=target.account_id,
        )
        api_base = self._adapter_resolve_domain_base(binding.domain)
        payload = {
            "content": json.dumps({"text": text}, ensure_ascii=False),
            "msg_type": "text",
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        if target.reply_to_message_id:
            url = (
                f"{api_base}/open-apis/im/v1/messages/"
                f"{target.reply_to_message_id}/reply"
            )
        else:
            url = f"{api_base}/open-apis/im/v1/messages?receive_id_type=chat_id"
            payload["receive_id"] = target.channel_id
        try:
            data = self._adapter_safe_request_json(
                method="POST",
                url=url,
                json_body=payload,
                headers=headers,
                content_type="application/json; charset=utf-8",
                timeout_sec=15,
                allow_hosts=self._adapter_allowed_api_hosts(binding.domain),
                policy=STANDARD_OUTBOUND_POLICY,
            )
        except SafeIOHTTPError as exc:
            if resolution.installation is not None:
                self._installation_manager.mark_api_error(
                    resolution.installation.installation_id,
                    error_code=exc.reason,
                    status_code=exc.status_code,
                    details={"phase": "reply"},
                )
            self._adapter_logger().warning(
                "Feishu reply failed: status=%s", exc.status_code
            )
            return
        if data.get("code", 0) != 0:
            if resolution.installation is not None:
                self._installation_manager.mark_api_error(
                    resolution.installation.installation_id,
                    error_code=str(data.get("msg", "unknown") or "unknown"),
                    status_code=200,
                    details={"phase": "reply"},
                )
            self._adapter_logger().warning(
                "Feishu reply failed: %s",
                data.get("msg", "unknown"),
            )

    async def send_message(
        self,
        channel_id: str,
        text: str,
        delivery_context: Optional[Dict[str, Any]] = None,
    ):
        ctx = dict(delivery_context or {})
        await self._send_reply(
            self._adapter_delivery_target(
                channel_id=channel_id,
                reply_to_message_id=str(ctx.get("thread_id", "") or "").strip(),
                workspace_id=str(ctx.get("workspace_id", "") or "").strip(),
                account_id=str(ctx.get("account_id", "") or "").strip(),
            ),
            text,
            delivery_context=ctx,
        )

    async def send_image(
        self,
        channel_id: str,
        image_data: bytes,
        filename: str = "image.png",
        caption: Optional[str] = None,
        delivery_context: Optional[Dict[str, Any]] = None,
    ):
        ctx = dict(delivery_context or {})
        resolution, binding, _ = self._resolve_delivery_binding(
            workspace_id=str(ctx.get("workspace_id", "") or "").strip(),
            account_id=str(ctx.get("account_id", "") or "").strip(),
        )
        if binding is None or not resolution.ok:
            self._adapter_logger().warning(
                "Feishu image dropped: no workspace binding available (%s / %s)",
                str(ctx.get("workspace_id", "") or "").strip() or "no-workspace",
                str(ctx.get("account_id", "") or "").strip() or "no-account",
            )
            return
        token = await self._get_tenant_access_token(
            binding=binding,
            workspace_id=str(ctx.get("workspace_id", "") or "").strip(),
            account_id=str(ctx.get("account_id", "") or "").strip(),
        )
        api_base = self._adapter_resolve_domain_base(binding.domain)
        upload_headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }
        upload_body, upload_content_type = self._adapter_build_multipart_form(
            fields={"image_type": "message"},
            file_field="image",
            filename=filename,
            file_bytes=image_data,
            file_content_type="image/png",
        )
        try:
            upload_payload = self._adapter_safe_request_json(
                method="POST",
                url=f"{api_base}/open-apis/im/v1/images",
                raw_body=upload_body,
                headers=upload_headers,
                content_type=upload_content_type,
                timeout_sec=30,
                allow_hosts=self._adapter_allowed_api_hosts(binding.domain),
                policy=STANDARD_OUTBOUND_POLICY,
            )
        except SafeIOHTTPError as exc:
            if resolution.installation is not None:
                self._installation_manager.mark_api_error(
                    resolution.installation.installation_id,
                    error_code=exc.reason,
                    status_code=exc.status_code,
                    details={"phase": "image_upload"},
                )
            self._adapter_logger().warning(
                "Feishu image upload failed: status=%s", exc.status_code
            )
            return
        image_key = str(
            (upload_payload.get("data") or {}).get("image_key", "") or ""
        ).strip()
        if upload_payload.get("code", 0) != 0 or not image_key:
            if resolution.installation is not None:
                self._installation_manager.mark_api_error(
                    resolution.installation.installation_id,
                    error_code=str(upload_payload.get("msg", "unknown") or "unknown"),
                    status_code=200,
                    details={"phase": "image_upload"},
                )
            self._adapter_logger().warning(
                "Feishu image upload failed: %s",
                upload_payload.get("msg", "unknown"),
            )
            return
        message_payload = {
            "content": json.dumps({"image_key": image_key}, ensure_ascii=False),
            "msg_type": "image",
        }
        thread_id = str(ctx.get("thread_id", "") or "").strip()
        if thread_id:
            send_url = f"{api_base}/open-apis/im/v1/messages/{thread_id}/reply"
        else:
            send_url = f"{api_base}/open-apis/im/v1/messages?receive_id_type=chat_id"
            message_payload["receive_id"] = channel_id
        try:
            self._adapter_safe_request_json(
                method="POST",
                url=send_url,
                json_body=message_payload,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                content_type="application/json; charset=utf-8",
                timeout_sec=30,
                allow_hosts=self._adapter_allowed_api_hosts(binding.domain),
                policy=STANDARD_OUTBOUND_POLICY,
            )
        except SafeIOHTTPError as exc:
            if resolution.installation is not None:
                self._installation_manager.mark_api_error(
                    resolution.installation.installation_id,
                    error_code=exc.reason,
                    status_code=exc.status_code,
                    details={"phase": "image_send"},
                )
            self._adapter_logger().warning(
                "Feishu image send failed: status=%s", exc.status_code
            )
        if caption:
            await self.send_message(
                channel_id,
                caption,
                delivery_context=ctx,
            )
