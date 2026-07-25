"""Owned Slack signed-ingress and interaction transaction mixin."""

import json
import time
from typing import Any, Dict, Optional
from urllib.parse import parse_qs

from ..contract import CommandRequest

# ruff: noqa: SIM102, UP006, UP035, UP045 -- preserve frozen behavior/signatures.
# mypy: disable-error-code="attr-defined,no-any-return"


class SlackIngressMixin:
    async def handle_event(self, request):
        """POST handler for Slack Events API."""
        _, web = self._adapter_import_aiohttp_web()

        try:
            body_bytes = await request.read()
        except Exception:
            return self._adapter_make_response(web, status=400, text="Bad request")

        # -- Step 1: Signature verification (fail-closed) --
        timestamp = ""
        signature = ""
        if hasattr(request, "headers"):
            timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
            signature = request.headers.get("X-Slack-Signature", "")

        if not self._adapter_verify_slack_signature(
            signing_secret=self.config.slack_signing_secret or "",
            timestamp=timestamp,
            body=body_bytes,
            signature=signature,
        ):
            self._adapter_logger().warning(
                "Slack signature verification failed (rejected)"
            )
            return self._adapter_make_response(
                web, status=401, text="Invalid signature"
            )

        # -- Step 2: Parse payload --
        try:
            payload = json.loads(body_bytes)
        except json.JSONDecodeError:
            return self._adapter_make_response(web, status=400, text="Bad JSON")

        # -- Step 3: url_verification challenge (Webhook only) --
        if payload.get("type") == "url_verification":
            challenge = payload.get("challenge", "")
            return self._adapter_make_json_response(web, {"challenge": challenge})

        # -- Step 4: Process event --
        try:
            await self.process_event_payload(payload)
        except ValueError:
            return self._adapter_make_response(web, status=400, text="Bad Request")
        return self._adapter_make_response(web, status=200, text="OK")

    async def handle_interaction(self, request):
        """POST handler for Slack Block Kit interactivity callbacks."""
        _, web = self._adapter_import_aiohttp_web()

        try:
            body_bytes = await request.read()
        except Exception:
            return self._adapter_make_response(web, status=400, text="Bad request")

        timestamp = ""
        signature = ""
        if hasattr(request, "headers"):
            timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
            signature = request.headers.get("X-Slack-Signature", "")

        if not self._adapter_verify_slack_signature(
            signing_secret=self.config.slack_signing_secret or "",
            timestamp=timestamp,
            body=body_bytes,
            signature=signature,
        ):
            self._adapter_logger().warning(
                "Slack interaction signature verification failed (rejected)"
            )
            return self._adapter_make_response(
                web, status=401, text="Invalid signature"
            )

        parsed = parse_qs(body_bytes.decode("utf-8"), keep_blank_values=True)
        raw_payload = (parsed.get("payload") or [""])[0]
        if not raw_payload:
            return self._adapter_make_response(web, status=400, text="Missing payload")

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return self._adapter_make_response(web, status=400, text="Bad payload")
        if not isinstance(payload, dict):
            return self._adapter_make_response(web, status=400, text="Bad payload")

        try:
            routed = await self.process_interaction_payload(payload)
        except ValueError:
            return self._adapter_make_response(web, status=400, text="Bad Request")
        except Exception as exc:
            safe_text = self._adapter_safe_external_error_text(
                "Slack interaction failed", exc
            )
            self._adapter_logger().warning("Slack interaction failed: %s", safe_text)
            return self._adapter_make_response(web, status=500, text=safe_text)

        # Slack requires a fast acknowledgement for interactivity requests.
        # Keep the external response bounded; detailed action results are routed
        # through the existing reply/deferred-response surfaces.
        return self._adapter_make_json_response(
            web, {"ok": True, "routed": bool(routed)}
        )

    async def process_event_payload(self, payload: Dict[str, Any]) -> None:
        """
        Shared event processing path for both webhook and socket mode transports.
        """
        if payload.get("type") != "event_callback":
            return

        event = payload.get("event", {})
        event_id = payload.get("event_id", "")
        event_type = event.get("type", "")
        workspace_id = self._installation_manager.extract_workspace_id(payload)

        if event_type in ("app_uninstalled", "tokens_revoked", "app_rate_limited"):
            if workspace_id:
                self._handle_lifecycle_event(workspace_id, event_type)
            return

        # -- Step 5: Replay / dedupe guard --
        if not event_id:
            self._adapter_logger().warning("Slack event missing event_id (rejected)")
            raise ValueError("Missing event_id")

        if not self._replay_guard.check_and_record(event_id):
            self._adapter_logger().debug(
                f"Slack duplicate event_id={event_id} (accepted, no-op)"
            )
            return

        # -- Step 6: Bot-loop prevention --
        # Resolve bot user ID from authorizations or cache.
        bot_user_id = self._get_bot_user_id(payload, workspace_id)

        sender_id = event.get("user", "")
        if sender_id and bot_user_id and sender_id == bot_user_id:
            return

        if event.get("bot_id"):
            return

        subtype = event.get("subtype", "")
        if subtype and subtype not in ("", "file_share"):
            return

        # -- Step 7: Event normalization --
        text = event.get("text", "").strip()
        channel_id = event.get("channel", "")
        thread_ts = event.get("thread_ts", "")
        message_ts = event.get("ts", "")

        if event_type not in ("message", "app_mention"):
            return

        if not text or not sender_id:
            return

        # S67: Require mention in group channels.
        is_dm = channel_id.startswith("D")
        mentioned_bot = event_type == "app_mention" or (
            bool(bot_user_id) and f"<@{bot_user_id}>" in text
        )
        if not is_dm and self.config.slack_require_mention:
            if event_type != "app_mention":
                if bot_user_id and f"<@{bot_user_id}>" not in text:
                    return

        if bot_user_id:
            text = text.replace(f"<@{bot_user_id}>", "").strip()

        # -- Step 8: Allowlist checks (S67) --
        if self._user_allowlist.entries:
            user_result = self._user_allowlist.evaluate(sender_id)
            if user_result.decision == "deny":
                self._adapter_logger().warning(
                    f"Slack user {sender_id} denied by allowlist"
                )
                return

        if self._channel_allowlist.entries and channel_id:
            chan_result = self._channel_allowlist.evaluate(channel_id)
            if chan_result.decision == "deny":
                self._adapter_logger().warning(
                    f"Slack channel {channel_id} denied by allowlist"
                )
                return

        # -- Step 9: Build CommandRequest and route --
        req = CommandRequest(
            platform="slack",
            sender_id=sender_id,
            channel_id=channel_id,
            username=sender_id,
            message_id=event_id,
            text=text,
            timestamp=float(message_ts) if message_ts else time.time(),
            workspace_id=workspace_id,
            thread_id=thread_ts
            or (message_ts if self.config.slack_reply_in_thread else ""),
        )

        try:
            resp = await self.router.handle(req)
            resp_text = getattr(resp, "text", "")
            if not isinstance(resp_text, str):
                resp_text = str(resp_text) if resp_text is not None else ""

            buttons = getattr(resp, "buttons", []) or []
            if resp_text or buttons:
                if buttons:
                    await self._send_interactive_reply(
                        channel_id=channel_id,
                        text=resp_text or "OpenClaw",
                        buttons=buttons,
                        thread_ts=req.thread_id,
                        delivery_context={
                            "workspace_id": workspace_id,
                            "thread_id": req.thread_id,
                            "channel_kind": self._adapter_channel_kind(channel_id),
                            "mentioned": mentioned_bot,
                        },
                    )
                else:
                    await self._send_reply(
                        channel_id=channel_id,
                        text=resp_text,
                        thread_ts=req.thread_id,
                        delivery_context={
                            "workspace_id": workspace_id,
                            "thread_id": req.thread_id,
                            "channel_kind": self._adapter_channel_kind(channel_id),
                            "mentioned": mentioned_bot,
                        },
                    )
        except Exception as e:
            self._adapter_logger().error(
                "Slack event handling failed (error_type=%s)", type(e).__name__
            )

    async def process_interaction_payload(self, payload: Dict[str, Any]) -> bool:
        interaction_type = str(payload.get("type", "") or "").strip()
        if interaction_type not in self._adapter_interaction_types():
            return False

        request = self._build_interaction_request(payload)
        if request is None:
            return False

        replay_key = self._interaction_replay_key(payload, request)
        if self._interaction_lifecycle is None:  # pragma: no cover
            if not self._replay_guard.check_and_record(replay_key):
                self._adapter_logger().debug(
                    "Slack duplicate interaction %s (accepted, no-op)", replay_key
                )
                return False
            claim = None
        else:
            claim = self._interaction_lifecycle.claim(
                replay_key,
                metadata={
                    "platform": "slack",
                    "workspace_id": request.workspace_id,
                    "interaction_type": str(payload.get("type", "") or ""),
                },
            )
        if claim is not None and not claim.accepted:
            self._adapter_logger().debug(
                "Slack duplicate interaction %s state=%s code=%s (accepted, no-op)",
                replay_key,
                claim.record.state,
                claim.code,
            )
            return False

        # IMPORTANT: interactive run-like payloads must be routed through the same
        # approval semantics as text commands. Untrusted users get approval forced
        # before CommandRouter sees the request, avoiding a parallel bypass path.
        if request.text.startswith("/run") and not (
            self.router._is_admin(request) or self.router._is_trusted(request)
        ):
            request.text = self._adapter_force_approval_command(request.text)

        try:
            response = await self.router.handle(request)
        except Exception:
            # IMPORTANT: only failures before router completion are retryable.
            # Once router.handle returns, duplicate user actions must not reroute.
            if self._interaction_lifecycle is not None:
                self._interaction_lifecycle.release_retryable(
                    replay_key, reason="slack_interaction_failed_before_commit"
                )
            raise

        if self._interaction_lifecycle is not None:
            self._interaction_lifecycle.commit_success(replay_key, reason="routed")
        response_text = str(getattr(response, "text", "") or "").strip()
        response_buttons = getattr(response, "buttons", []) or []
        if response_text or response_buttons:
            if response_buttons:
                await self._send_interactive_reply(
                    channel_id=request.channel_id,
                    text=response_text or "Action processed.",
                    buttons=response_buttons,
                    thread_ts=request.thread_id,
                    delivery_context={
                        "workspace_id": request.workspace_id,
                        "thread_id": request.thread_id,
                    },
                )
            elif response_text:
                await self._send_reply(
                    channel_id=request.channel_id,
                    text=response_text,
                    thread_ts=request.thread_id,
                    delivery_context={
                        "workspace_id": request.workspace_id,
                        "thread_id": request.thread_id,
                    },
                )
        return True

    def _build_interaction_request(
        self, payload: Dict[str, Any]
    ) -> Optional[CommandRequest]:
        interaction_type = str(payload.get("type", "") or "").strip()
        command_text = self._extract_interaction_command(payload)
        if not command_text:
            return None

        team = payload.get("team") or {}
        user = payload.get("user") or {}
        container = payload.get("container") or {}
        channel = payload.get("channel") or {}
        view = payload.get("view") or {}
        message = payload.get("message") or {}
        action = self._first_action(payload)

        workspace_id = self._adapter_first_non_empty(
            team.get("id"),
            payload.get("team_id"),
            (
                payload.get("enterprise", {}).get("id")
                if isinstance(payload.get("enterprise"), dict)
                else ""
            ),
        )
        sender_id = self._adapter_first_non_empty(
            user.get("id"), payload.get("user_id")
        )
        channel_id = self._adapter_first_non_empty(
            channel.get("id"),
            container.get("channel_id"),
            payload.get("channel_id"),
        )
        message_id = self._adapter_first_non_empty(
            view.get("id"),
            action.get("action_ts"),
            container.get("message_ts"),
            payload.get("trigger_id"),
            f"slack-interaction-{int(time.time())}",
        )
        thread_id = self._adapter_first_non_empty(
            container.get("thread_ts"),
            message.get("thread_ts") if isinstance(message, dict) else "",
            container.get("message_ts"),
        )
        if not thread_id and self.config.slack_reply_in_thread:
            thread_id = self._adapter_first_non_empty(
                container.get("message_ts"), message.get("ts")
            )

        return CommandRequest(
            platform="slack",
            sender_id=sender_id,
            channel_id=channel_id or sender_id,
            username=self._adapter_first_non_empty(
                user.get("username"), user.get("name"), sender_id
            ),
            message_id=message_id,
            text=command_text,
            timestamp=time.time(),
            workspace_id=workspace_id,
            thread_id=thread_id,
            metadata={
                "interactive_callback": True,
                "interaction_type": interaction_type,
                "action_id": self._adapter_first_non_empty(
                    action.get("action_id"), view.get("callback_id")
                ),
                "response_url": str(payload.get("response_url", "") or ""),
            },
        )

    def _extract_interaction_command(self, payload: Dict[str, Any]) -> str:
        interaction_type = str(payload.get("type", "") or "").strip()
        if interaction_type == "block_actions":
            action = self._first_action(payload)
            selected = action.get("selected_option") or {}
            value = self._adapter_first_non_empty(
                action.get("value"),
                selected.get("value") if isinstance(selected, dict) else "",
                action.get("action_id"),
            )
            parsed = self._adapter_json_loads_safe(value)
            return self._adapter_first_non_empty(
                parsed.get("command"), parsed.get("value"), value
            )
        if interaction_type == "view_submission":
            view = payload.get("view") or {}
            private_meta = self._adapter_first_non_empty(view.get("private_metadata"))
            parsed = self._adapter_json_loads_safe(private_meta)
            if parsed:
                return self._adapter_first_non_empty(
                    parsed.get("command"), parsed.get("value")
                )
            if private_meta:
                return private_meta
            state = (view.get("state") or {}).get("values") or {}
            return self._extract_command_from_view_state(state)
        if interaction_type == "workflow_step_execute":
            workflow_step = payload.get("workflow_step") or {}
            inputs = workflow_step.get("inputs") or {}
            command = inputs.get("command") or {}
            if isinstance(command, dict):
                return self._adapter_first_non_empty(command.get("value"))
            return self._adapter_first_non_empty(workflow_step.get("callback_id"))
        return ""

    def _extract_command_from_view_state(self, state: Dict[str, Any]) -> str:
        if not isinstance(state, dict):
            return ""
        for block_value in state.values():
            if not isinstance(block_value, dict):
                continue
            for action_value in block_value.values():
                if not isinstance(action_value, dict):
                    continue
                candidate = self._adapter_first_non_empty(
                    action_value.get("value"),
                    (
                        (action_value.get("selected_option") or {}).get("value")
                        if isinstance(action_value.get("selected_option"), dict)
                        else ""
                    ),
                )
                parsed = self._adapter_json_loads_safe(candidate)
                command = self._adapter_first_non_empty(
                    parsed.get("command"), parsed.get("value"), candidate
                )
                if command:
                    return command
        return ""

    def _first_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        actions = payload.get("actions") or []
        if isinstance(actions, list) and actions and isinstance(actions[0], dict):
            return actions[0]
        return {}

    def _interaction_replay_key(
        self, payload: Dict[str, Any], request: CommandRequest
    ) -> str:
        action = self._first_action(payload)
        key_parts = [
            "interaction",
            str(payload.get("type", "") or ""),
            request.workspace_id,
            request.sender_id,
            request.channel_id,
            request.message_id,
            str(payload.get("trigger_id", "") or ""),
            str(action.get("action_id", "") or ""),
            str(action.get("action_ts", "") or ""),
            request.text,
        ]
        return ":".join(key_parts)

    # ------------------------------------------------------------------
    # Slack Web API reply
    # ------------------------------------------------------------------
