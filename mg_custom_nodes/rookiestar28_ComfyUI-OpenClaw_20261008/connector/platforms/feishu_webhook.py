"""
Feishu / Lark connector baseline adapter (F67).

Implements:
- webhook ingress with verification-token challenge response
- shared event normalization for webhook + long-connection transports
- DM / group mention gating into CommandRequest
- text / image delivery through Feishu Open API

Notes:
- Feishu "workspace" is represented by tenant_key for connector diagnostics.
- group traffic is gated by explicit bot mention unless disabled in config.
"""

from __future__ import annotations

import json
import logging
import secrets
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from ..config import ConnectorConfig
from ..router import CommandRouter
from ..security_profile import AllowlistPolicy, ReplayGuard
from .feishu_delivery_handlers import FeishuDeliveryMixin, FeishuDeliveryTarget
from .feishu_ingress_handlers import FeishuIngressMixin
from .feishu_installation_handlers import FeishuInstallationMixin
from .feishu_installation_manager import FeishuInstallationManager

try:
    from services.safe_io import safe_request_json
except ImportError:  # pragma: no cover
    from services.safe_io import safe_request_json  # type: ignore

try:
    from services.connector_callback_contract import ConnectorCallbackContract
except ImportError:  # pragma: no cover
    from services.connector_callback_contract import (  # type: ignore
        ConnectorCallbackContract,
    )

logger = logging.getLogger(__name__)

FEISHU_WEBHOOK_MAX_BODY_BYTES = 256 * 1024
FEISHU_TOKEN_TTL_SEC = 3600
FEISHU_DOMAIN_BASES = {
    "feishu": "https://open.feishu.cn",
    "lark": "https://open.larksuite.com",
}
_SUPPORTED_EVENT_TYPES = frozenset({"im.message.receive_v1"})
_PLACEHOLDER_TYPES = {
    "image": "<image>",
    "audio": "<audio>",
    "file": "<file>",
    "media": "<media>",
    "sticker": "<sticker>",
}
_FEISHU_CALLBACK_POLICY_MAP = {
    "approval.approve": "admin",
    "approval.reject": "admin",
    "command.status": "public",
    "command.run": "run",
}


def _import_aiohttp_web():
    # CRITICAL: do not replace with direct import; connector tests and minimal
    # CI envs intentionally exercise adapter startup without aiohttp installed.
    try:
        import aiohttp  # type: ignore
        from aiohttp import web  # type: ignore
    except ModuleNotFoundError:
        return None, None
    return aiohttp, web


class _CompatResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        text: str = "",
        content_type: str = "text/plain",
        body: Optional[bytes] = None,
    ):
        self.status = status
        self.text = text
        self.content_type = content_type
        self.body = body if body is not None else text.encode("utf-8")


def _make_response(web_mod, *, status: int = 200, text: str = "OK"):
    if web_mod is not None:
        return web_mod.Response(status=status, text=text)
    return _CompatResponse(status=status, text=text)


def _make_json_response(web_mod, data: Dict[str, Any], *, status: int = 200):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    if web_mod is not None:
        return web_mod.json_response(data, status=status)
    return _CompatResponse(
        status=status,
        text=body.decode("utf-8"),
        content_type="application/json",
        body=body,
    )


def _safe_external_error_code(default: str, _exc: Exception) -> str:
    # IMPORTANT: keep Feishu external failures constant. Returning exception-
    # derived codes/text here reopens the residual CodeQL stack-trace finding.
    return default


def _resolve_domain_base(domain: str) -> str:
    normalized = str(domain or "feishu").strip().lower()
    return FEISHU_DOMAIN_BASES.get(normalized, FEISHU_DOMAIN_BASES["feishu"])


def _allowed_api_hosts(domain: str) -> set[str]:
    host = urlparse(_resolve_domain_base(domain)).hostname or ""
    return {host} if host else set()


def _build_multipart_form(
    *,
    fields: Dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    file_content_type: str,
) -> Tuple[bytes, str]:
    boundary = f"----openclaw-feishu-{secrets.token_hex(8)}"
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (f'Content-Disposition: form-data; name="{key}"\r\n\r\n').encode(
                    "utf-8"
                ),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    parts.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {file_content_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _json_loads_safe(raw: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, ValueError):
        pass
    return {}


def _normalize_mentions(message: Dict[str, Any]) -> list[dict]:
    mentions = message.get("mentions") or []
    return mentions if isinstance(mentions, list) else []


def _strip_bot_mention(text: str, mentions: list[dict], bot_open_id: str) -> str:
    cleaned = text or ""
    for mention in mentions:
        key = str(mention.get("key", "") or "").strip()
        open_id = str(((mention.get("id") or {}).get("open_id")) or "").strip()
        if key and bot_open_id and open_id == bot_open_id:
            cleaned = cleaned.replace(key, " ")
    return " ".join(cleaned.split())


def _post_text_to_plain(parsed: Dict[str, Any]) -> str:
    pieces: list[str] = []
    title = str(parsed.get("title", "") or "").strip()
    if title:
        pieces.append(title)
    for row in parsed.get("content") or []:
        if not isinstance(row, list):
            continue
        row_pieces: list[str] = []
        for item in row:
            if not isinstance(item, dict):
                continue
            tag = str(item.get("tag", "") or "").strip().lower()
            if tag == "text":
                row_pieces.append(str(item.get("text", "") or ""))
            elif tag == "at":
                name = str(item.get("user_name", "") or "").strip()
                row_pieces.append(f"@{name}" if name else "@mentioned")
        line = "".join(row_pieces).strip()
        if line:
            pieces.append(line)
    return "\n".join(piece for piece in pieces if piece).strip()


def parse_feishu_message_text(message: Dict[str, Any]) -> str:
    msg_type = str(message.get("message_type", "") or "").strip().lower()
    raw_content = str(message.get("content", "") or "")
    parsed = _json_loads_safe(raw_content)
    if msg_type == "text":
        return str(parsed.get("text", "") or "").strip()
    if msg_type == "post":
        return _post_text_to_plain(parsed)
    if msg_type in _PLACEHOLDER_TYPES:
        return _PLACEHOLDER_TYPES[msg_type]
    return str(parsed.get("text", "") or "").strip()


def _infer_callback_action_type(command_text: str, button: Dict[str, Any]) -> str:
    explicit = str(button.get("action_type", "") or "").strip()
    if explicit:
        return explicit
    normalized = str(command_text or "").strip().lower()
    if normalized.startswith("/approve"):
        return "approval.approve"
    if normalized.startswith("/reject"):
        return "approval.reject"
    if normalized.startswith("/run"):
        return "command.run"
    if normalized.startswith("/status"):
        return "command.status"
    return "command.unknown"


def _force_approval_command(command_text: str) -> str:
    normalized = str(command_text or "").strip()
    if not normalized:
        return normalized
    if normalized.startswith("/run") and "--approval" not in normalized:
        return f"{normalized} --approval"
    return normalized


class FeishuWebhookServer(
    FeishuInstallationMixin,
    FeishuIngressMixin,
    FeishuDeliveryMixin,
):
    REPLAY_WINDOW_SEC = 300
    NONCE_CACHE_SIZE = 5000

    def __init__(
        self,
        config: ConnectorConfig,
        router: CommandRouter,
        *,
        installation_manager: Optional[FeishuInstallationManager] = None,
        bound_account_id: str = "",
    ):
        self.config = config
        self.router = router
        self._installation_manager = installation_manager or FeishuInstallationManager(
            config
        )
        self._bound_account_id = str(bound_account_id or "").strip()
        self.app = None
        self.runner = None
        self.site = None
        self._replay_guard = ReplayGuard(
            window_sec=self.REPLAY_WINDOW_SEC,
            max_entries=self.NONCE_CACHE_SIZE,
        )
        self._user_allowlist = AllowlistPolicy(
            config.feishu_allowed_users, strict=False
        )
        self._chat_allowlist = AllowlistPolicy(
            config.feishu_allowed_chats, strict=False
        )
        self._tenant_access_tokens: Dict[str, str] = {}
        self._tenant_access_token_expires_at: Dict[str, float] = {}
        self._bot_open_ids: Dict[str, str] = {}
        self._bot_open_id: str = ""
        self._callback_contracts: Dict[str, ConnectorCallbackContract] = {}
        self._callback_contract_secrets: Dict[str, str] = {}

    # IMPORTANT: keep facade patch seams live across extracted protocol owners.
    @staticmethod
    def _adapter_import_aiohttp_web():
        return _import_aiohttp_web()

    @staticmethod
    def _adapter_make_response(*args, **kwargs):
        return _make_response(*args, **kwargs)

    @staticmethod
    def _adapter_make_json_response(*args, **kwargs):
        return _make_json_response(*args, **kwargs)

    @staticmethod
    def _adapter_safe_external_error_code(*args, **kwargs):
        return _safe_external_error_code(*args, **kwargs)

    @staticmethod
    def _adapter_resolve_domain_base(*args, **kwargs):
        return _resolve_domain_base(*args, **kwargs)

    @staticmethod
    def _adapter_allowed_api_hosts(*args, **kwargs):
        return _allowed_api_hosts(*args, **kwargs)

    @staticmethod
    def _adapter_build_multipart_form(*args, **kwargs):
        return _build_multipart_form(*args, **kwargs)

    @staticmethod
    def _adapter_json_loads_safe(*args, **kwargs):
        return _json_loads_safe(*args, **kwargs)

    @staticmethod
    def _adapter_normalize_mentions(*args, **kwargs):
        return _normalize_mentions(*args, **kwargs)

    @staticmethod
    def _adapter_strip_bot_mention(*args, **kwargs):
        return _strip_bot_mention(*args, **kwargs)

    @staticmethod
    def _adapter_parse_message_text(*args, **kwargs):
        return parse_feishu_message_text(*args, **kwargs)

    @staticmethod
    def _adapter_infer_callback_action_type(*args, **kwargs):
        return _infer_callback_action_type(*args, **kwargs)

    @staticmethod
    def _adapter_force_approval_command(*args, **kwargs):
        return _force_approval_command(*args, **kwargs)

    @staticmethod
    def _adapter_safe_request_json(*args, **kwargs):
        return safe_request_json(*args, **kwargs)

    @staticmethod
    def _adapter_max_body_bytes():
        return FEISHU_WEBHOOK_MAX_BODY_BYTES

    @staticmethod
    def _adapter_token_ttl_sec():
        return FEISHU_TOKEN_TTL_SEC

    @staticmethod
    def _adapter_callback_policy_map():
        return _FEISHU_CALLBACK_POLICY_MAP

    @staticmethod
    def _adapter_supported_event_types():
        return _SUPPORTED_EVENT_TYPES

    @staticmethod
    def _adapter_delivery_target(*args, **kwargs):
        return FeishuDeliveryTarget(*args, **kwargs)

    @staticmethod
    def _adapter_logger():
        return logger

    async def start(self):
        aiohttp, web = _import_aiohttp_web()
        if aiohttp is None or web is None:
            logger.warning("aiohttp not installed. Skipping Feishu webhook adapter.")
            return
        if not self._installation_manager.has_bindings():
            logger.info(
                "Feishu adapter disabled "
                "(OPENCLAW_CONNECTOR_FEISHU_APP_ID / APP_SECRET missing)"
            )
            return
        has_event_ingress = any(
            binding.verification_token
            for binding in self._installation_manager.bindings()
        )
        has_callback_ingress = bool(str(self.config.feishu_callback_path or "").strip())
        if not has_event_ingress and not has_callback_ingress:
            logger.info(
                "Feishu webhook adapter disabled "
                "(verification token and callback path missing)"
            )
            return
        logger.info(
            "Starting Feishu webhook on %s:%s%s (%s)",
            self.config.feishu_bind_host,
            self.config.feishu_bind_port,
            self.config.feishu_webhook_path,
            self.config.feishu_domain,
        )
        self.app = web.Application(client_max_size=FEISHU_WEBHOOK_MAX_BODY_BYTES)
        if has_event_ingress:
            self.app.router.add_post(self.config.feishu_webhook_path, self.handle_event)
        if has_callback_ingress:
            self.app.router.add_post(
                self.config.feishu_callback_path,
                self.handle_callback,
            )
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(
            self.runner,
            self.config.feishu_bind_host,
            self.config.feishu_bind_port,
        )
        await self.site.start()

    async def stop(self):
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
