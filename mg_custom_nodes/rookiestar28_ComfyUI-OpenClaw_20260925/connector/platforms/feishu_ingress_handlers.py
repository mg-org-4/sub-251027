"""Owned Feishu webhook ingress and callback transaction mixin."""

# ruff: noqa: UP006, UP035, UP045 -- preserve frozen facade annotations.

from __future__ import annotations

import json
import secrets
import time
from typing import Any, Dict, Optional, Tuple

from services.connector_callback_contract import (
    CallbackActorContext,
    CallbackDecisionCode,
    ConnectorCallbackContract,
)

from ..contract import CommandRequest
from .feishu_installation_manager import FeishuBinding

# mypy: disable-error-code="attr-defined,index,no-any-return"


class FeishuIngressMixin:
    async def handle_event(self, request):
        _, web = self._adapter_import_aiohttp_web()
        try:
            body = await request.read()
        except Exception:
            return self._adapter_make_response(web, status=400, text="Bad request")
        if len(body) > self._adapter_max_body_bytes():
            return self._adapter_make_response(
                web, status=413, text="Payload too large"
            )
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return self._adapter_make_response(web, status=400, text="Bad JSON")
        if self._is_challenge(payload):
            if not self._verify_request_token(payload):
                return self._adapter_make_response(
                    web, status=401, text="Invalid verification token"
                )
            return self._adapter_make_json_response(
                web, {"challenge": str(payload.get("challenge", "") or "")}
            )
        if not self._verify_request_token(payload):
            return self._adapter_make_response(
                web, status=401, text="Invalid verification token"
            )
        try:
            await self.process_event_payload(payload)
        except ValueError as exc:
            safe_code = self._adapter_safe_external_error_code("event_rejected", exc)
            self._adapter_logger().warning("Feishu event rejected: %s", safe_code)
            return self._adapter_make_response(
                web,
                status=400,
                text=safe_code,
            )
        return self._adapter_make_response(web, status=200, text="OK")

    async def handle_callback(self, request):
        _, web = self._adapter_import_aiohttp_web()
        try:
            body = await request.read()
        except Exception:
            return self._adapter_make_response(web, status=400, text="Bad request")
        if len(body) > self._adapter_max_body_bytes():
            return self._adapter_make_response(
                web, status=413, text="Payload too large"
            )
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return self._adapter_make_response(web, status=400, text="Bad JSON")
        try:
            response = await self.process_callback_payload(payload)
        except ValueError as exc:
            safe_code = self._adapter_safe_external_error_code("callback_rejected", exc)
            self._adapter_logger().warning("Feishu callback rejected: %s", safe_code)
            return self._adapter_make_json_response(
                web,
                {
                    "ok": False,
                    "error": safe_code,
                },
                status=403,
            )
        return self._adapter_make_json_response(web, response)

    def _is_challenge(self, payload: Dict[str, Any]) -> bool:
        return bool(
            payload.get("challenge")
            and str(payload.get("type", "") or "").strip().lower() == "url_verification"
        )

    def _verify_request_token(self, payload: Dict[str, Any]) -> bool:
        try:
            self._resolve_inbound_binding(payload)
            return True
        except ValueError:
            return False

    def _extract_callback_action(self, payload: Dict[str, Any]) -> Tuple[
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        Dict[str, Any],
        str,
        str,
    ]:
        header = payload.get("header") or {}
        event = payload.get("event") or {}
        action = payload.get("action") or event.get("action") or {}
        if not action and isinstance(event.get("actions"), list):
            first_action = event.get("actions")[0] if event.get("actions") else {}
            if isinstance(first_action, dict):
                action = first_action
        if not isinstance(action, dict):
            raise ValueError("invalid_callback_action")
        raw_value = action.get("value") or {}
        if isinstance(raw_value, str):
            raw_value = self._adapter_json_loads_safe(raw_value)
        if not isinstance(raw_value, dict):
            raise ValueError("invalid_callback_value")
        envelope = raw_value.get("callback_envelope") or {}
        callback_payload = raw_value.get("payload") or {}
        if not isinstance(envelope, dict) or not isinstance(callback_payload, dict):
            raise ValueError("invalid_callback_envelope")
        workspace_id = str(
            header.get("tenant_key")
            or event.get("tenant_key")
            or callback_payload.get("workspace_id")
            or ""
        ).strip()
        account_id = str(callback_payload.get("account_id", "") or "").strip()
        return header, event, envelope, callback_payload, workspace_id, account_id

    def _callback_contract_for_binding(
        self,
        *,
        binding: FeishuBinding,
        signing_secret: str,
    ) -> ConnectorCallbackContract:
        cache_key = self._cache_key_for_binding(binding)
        if (
            self._callback_contracts.get(cache_key) is not None
            and self._callback_contract_secrets.get(cache_key) == signing_secret
        ):
            return self._callback_contracts[cache_key]
        contract = ConnectorCallbackContract(
            signing_secret=signing_secret,
            installation_registry=self._installation_manager.registry,
            action_policy_map=self._adapter_callback_policy_map(),
        )
        self._callback_contracts[cache_key] = contract
        self._callback_contract_secrets[cache_key] = signing_secret
        return contract

    def _actor_context_for_callback(
        self,
        *,
        actor_id: str,
        actor_open_id: str,
        channel_id: str,
        message_id: str,
        workspace_id: str,
        account_id: str,
        command_text: str,
    ) -> Tuple[CallbackActorContext, CommandRequest]:
        request = CommandRequest(
            platform="feishu",
            sender_id=actor_id or actor_open_id,
            channel_id=channel_id or actor_id or actor_open_id,
            username=actor_id or actor_open_id,
            message_id=message_id or f"cb-{secrets.token_hex(4)}",
            text=command_text,
            timestamp=time.time(),
            workspace_id=workspace_id,
            thread_id=message_id,
            metadata={
                "account_id": account_id,
                "sender_open_id": actor_open_id,
                "interactive_callback": True,
            },
        )
        actor = CallbackActorContext(
            is_admin=self.router._is_admin(request.sender_id),
            is_trusted=self.router._is_trusted(request),
            user_id=request.sender_id,
            tenant_id=workspace_id or request.workspace_id or "",
        )
        return actor, request

    def _build_callback_response(
        self,
        *,
        ok: bool,
        text: str,
        response_type: str = "info",
        card: Optional[Dict[str, Any]] = None,
        duplicate: bool = False,
        decision_code: str = "",
    ) -> Dict[str, Any]:
        response = {
            "ok": ok,
            "duplicate": duplicate,
            "decision_code": decision_code,
            "toast": {
                "type": response_type,
                "content": text[:500] if text else "",
            },
        }
        if card is not None:
            response["card"] = card
        return response

    def _build_request(
        self,
        payload: Dict[str, Any],
        *,
        binding: FeishuBinding,
        bot_open_id: str,
    ) -> Optional[CommandRequest]:
        header = payload.get("header") or {}
        if (
            str(header.get("event_type", "") or "").strip()
            not in self._adapter_supported_event_types()
        ):
            return None
        event = payload.get("event") or {}
        message = event.get("message") or {}
        sender = event.get("sender") or {}
        sender_id = sender.get("sender_id") or {}
        mentions = self._adapter_normalize_mentions(message)
        sender_user_id = str(sender_id.get("user_id", "") or "").strip()
        sender_open_id = str(sender_id.get("open_id", "") or "").strip()
        chat_id = str(message.get("chat_id", "") or "").strip()
        chat_type = str(message.get("chat_type", "") or "").strip().lower()
        message_id = str(message.get("message_id", "") or "").strip()
        workspace_id = (
            str(header.get("tenant_key", "") or "").strip() or binding.workspace_id
        )
        if not sender_user_id and not sender_open_id:
            return None
        if not chat_id or not message_id:
            return None
        if sender_open_id and bot_open_id and sender_open_id == bot_open_id:
            return None
        raw_text = self._adapter_parse_message_text(message)
        if not raw_text:
            return None
        mentioned_bot = False
        if bot_open_id:
            for mention in mentions:
                open_id = str(((mention.get("id") or {}).get("open_id")) or "").strip()
                if open_id and open_id == bot_open_id:
                    mentioned_bot = True
                    break
        text = self._adapter_strip_bot_mention(raw_text, mentions, bot_open_id)
        if (
            chat_type == "group"
            and self.config.feishu_require_mention
            and not mentioned_bot
        ):
            return None
        effective_sender = sender_user_id or sender_open_id
        return CommandRequest(
            platform="feishu",
            sender_id=effective_sender,
            channel_id=chat_id,
            username=effective_sender,
            message_id=message_id,
            text=text,
            timestamp=time.time(),
            workspace_id=workspace_id,
            thread_id=(
                str(message.get("root_id", "") or "").strip()
                or (message_id if self.config.feishu_reply_in_thread else "")
            ),
            metadata={
                "account_id": binding.account_id,
                "chat_type": chat_type,
                "mentioned_bot": mentioned_bot,
                "message_type": str(message.get("message_type", "") or "").strip(),
                "sender_open_id": sender_open_id,
            },
        )

    async def process_event_payload(
        self,
        payload: Dict[str, Any],
        *,
        binding: Optional[FeishuBinding] = None,
    ) -> None:
        header = payload.get("header") or {}
        event_id = str(header.get("event_id", "") or "").strip()
        if not event_id:
            raise ValueError("Missing event_id")
        if not self._replay_guard.check_and_record(event_id):
            return
        effective_binding = binding or self._resolve_inbound_binding(payload)
        bot_open_id = self._cached_bot_open_id(effective_binding)
        message = (payload.get("event") or {}).get("message") or {}
        chat_type = str(message.get("chat_type", "") or "").strip().lower()
        if not bot_open_id and chat_type == "group":
            bot_open_id = await self._fetch_bot_open_id(
                binding=effective_binding, allow_degrade=True
            )
        request = self._build_request(
            payload,
            binding=effective_binding,
            bot_open_id=bot_open_id,
        )
        if request is None:
            return
        if self._user_allowlist.entries:
            user_result = self._user_allowlist.evaluate(str(request.sender_id))
            if user_result.decision == "deny":
                return
        if self._chat_allowlist.entries:
            chat_result = self._chat_allowlist.evaluate(str(request.channel_id))
            if chat_result.decision == "deny":
                return
        response = await self.router.handle(request)
        resp_text = str(getattr(response, "text", "") or "").strip()
        buttons = getattr(response, "buttons", []) or []
        target = self._adapter_delivery_target(
            channel_id=request.channel_id,
            reply_to_message_id=request.thread_id,
            workspace_id=request.workspace_id,
            account_id=str(request.metadata.get("account_id", "") or ""),
        )
        if buttons:
            await self._send_interactive_reply(target, resp_text, buttons)
        elif resp_text:
            await self._send_reply(
                target,
                resp_text,
                delivery_context={
                    "workspace_id": request.workspace_id,
                    "thread_id": request.thread_id,
                    "account_id": str(request.metadata.get("account_id", "") or ""),
                    "chat_type": str(request.metadata.get("chat_type", "") or ""),
                    "mentioned_bot": bool(request.metadata.get("mentioned_bot")),
                },
            )

    async def process_callback_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        _, _, envelope_dict, callback_payload, workspace_id, account_id = (
            self._extract_callback_action(payload)
        )
        resolution, binding, secrets = self._resolve_delivery_binding(
            workspace_id=workspace_id,
            account_id=account_id,
        )
        if binding is None or not resolution.ok:
            raise ValueError(resolution.reject_reason or "missing_binding")
        signing_secret = str(
            secrets.get("app_secret", "") or binding.app_secret or ""
        ).strip()
        if not signing_secret:
            raise ValueError("missing_callback_signing_secret")
        contract = self._callback_contract_for_binding(
            binding=binding,
            signing_secret=signing_secret,
        )
        event = payload.get("event") or {}
        operator = payload.get("operator") or event.get("operator") or {}
        operator_id = operator.get("operator_id") or operator.get("sender_id") or {}
        actor_id = str(
            operator.get("user_id")
            or operator_id.get("user_id")
            or callback_payload.get("actor_user_id")
            or ""
        ).strip()
        actor_open_id = str(
            operator.get("open_id")
            or operator_id.get("open_id")
            or callback_payload.get("actor_open_id")
            or ""
        ).strip()
        command_text = str(callback_payload.get("command", "") or "").strip()
        actor, request = self._actor_context_for_callback(
            actor_id=actor_id,
            actor_open_id=actor_open_id,
            channel_id=str(
                payload.get("open_chat_id")
                or event.get("open_chat_id")
                or callback_payload.get("channel_id")
                or ""
            ).strip(),
            message_id=str(
                payload.get("open_message_id")
                or event.get("open_message_id")
                or callback_payload.get("message_id")
                or ""
            ).strip(),
            workspace_id=workspace_id or binding.workspace_id,
            account_id=binding.account_id,
            command_text=command_text,
        )
        decision = contract.evaluate(
            platform="feishu",
            envelope_dict=envelope_dict,
            payload=callback_payload,
            actor=actor,
        )
        if decision.decision_code == CallbackDecisionCode.REJECT_REPLAY.value:
            return self._build_callback_response(
                ok=True,
                text="Action already processed.",
                response_type="info",
                duplicate=True,
                decision_code=decision.decision_code,
            )
        if not decision.ok and not decision.requires_approval:
            raise ValueError(decision.message or decision.decision_code)
        request.text = (
            self._adapter_force_approval_command(request.text)
            if decision.requires_approval
            else request.text
        )
        request_id = str(envelope_dict.get("request_id", "") or "")
        contract.acknowledge_request(request_id)
        try:
            response = await self.router.handle(request)
        except Exception:
            # IMPORTANT: failures before route completion remain retryable.
            # After router.handle returns, the action may already have side effects,
            # so completion failures must not release the claim for rerouting.
            contract.release_request_retryable(
                request_id, reason="feishu_callback_failed_before_commit"
            )
            raise
        contract.complete_request(request_id)
        response_text = str(getattr(response, "text", "") or "").strip() or (
            "Action processed."
        )
        response_buttons = getattr(response, "buttons", []) or []
        card = None
        if response_buttons:
            card = self._build_interactive_card(
                self._adapter_delivery_target(
                    channel_id=request.channel_id,
                    reply_to_message_id=request.thread_id,
                    workspace_id=request.workspace_id,
                    account_id=binding.account_id,
                ),
                response_text,
                response_buttons,
                binding=binding,
                secrets=secrets,
            )
        return self._build_callback_response(
            ok=True,
            text=response_text,
            response_type="success",
            card=card,
            decision_code=decision.decision_code,
        )
