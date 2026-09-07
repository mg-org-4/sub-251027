"""
Slack Events API Webhook Adapter (F56).

Implements:
- Events API POST ingress with ``url_verification`` challenge response.
- Slack request authenticity via ``X-Slack-Signature`` + ``X-Slack-Request-Timestamp``
  (``v0:{ts}:{raw_body}`` HMAC-SHA256).
- Replay / duplicate guard (event_id + timestamp window).
- ``message`` / ``app_mention`` event normalization with de-duplication
  (avoid double-trigger when bot is mentioned in a regular message).
- CommandRequest conversion -> CommandRouter.
- Slack Web API thread or channel reply.

S67 Safety Profile:
- AllowlistPolicy for users and channels (fail-closed when configured).
- Bot-loop prevention (ignore messages from bot itself).
- Rate-limit delegation to CommandRouter (R80 authz + F32 rate limiter).
- Require-mention policy for group conversations.

Setup:
1. Create a Slack App at https://api.slack.com/apps.
2. Enable Events API; set Request URL to ``https://<host>/slack/events``.
3. Subscribe to ``message.channels``, ``message.groups``, ``message.im``,
   ``app_mention`` bot events.
4. Install app to workspace; copy Bot Token and Signing Secret.
5. Set env vars:
   - ``OPENCLAW_CONNECTOR_SLACK_BOT_TOKEN``
   - ``OPENCLAW_CONNECTOR_SLACK_SIGNING_SECRET``
"""

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Optional

from ..config import ConnectorConfig
from ..router import CommandRouter
from ..security_profile import AllowlistPolicy, ReplayGuard
from .slack_delivery_handlers import SlackDeliveryMixin
from .slack_ingress_handlers import SlackIngressMixin
from .slack_installation_handlers import SlackInstallationMixin
from .slack_installation_manager import SlackInstallationManager

try:
    from services.connector_replay_lifecycle import ConnectorReplayLifecycle
except ImportError:  # pragma: no cover
    ConnectorReplayLifecycle = None  # type: ignore

logger = logging.getLogger(__name__)

_SLACK_INTERACTION_TYPES = frozenset(
    {"block_actions", "view_submission", "workflow_step_execute"}
)


def _slack_channel_kind(channel_id: str) -> str:
    if str(channel_id or "").startswith("D"):
        return "dm"
    return "group"


# -- aiohttp compat layer (same pattern as kakao/whatsapp/wechat) -----------


def _import_aiohttp_web():
    try:
        import aiohttp
        from aiohttp import web
    except ModuleNotFoundError:
        return None, None
    return aiohttp, web


class _CompatResponse:
    """Minimal response shim for unit tests when aiohttp is unavailable."""

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


def _make_json_response(web_mod, data: dict, *, status: int = 200):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    if web_mod is not None:
        return web_mod.json_response(data, status=status)
    return _CompatResponse(
        status=status,
        text=body.decode("utf-8"),
        content_type="application/json",
        body=body,
    )


def _make_redirect_response(web_mod, url: str):
    if web_mod is not None:
        raise web_mod.HTTPFound(location=url)
    return _CompatResponse(status=302, text=url)


def _safe_external_error_text(default: str, _exc: Exception) -> str:
    # IMPORTANT: keep Slack external failures constant. Even "short safe-looking"
    # exception text remains scanner-tainted and can re-expose internal detail.
    return default


def _json_loads_safe(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _force_approval_command(command_text: str) -> str:
    normalized = str(command_text or "").strip()
    if normalized.startswith("/run") and "--approval" not in normalized:
        return f"{normalized} --approval"
    return normalized


def _style_to_slack(style: str) -> str:
    normalized = str(style or "").strip().lower()
    if normalized in {"primary", "danger"}:
        return normalized
    return "primary" if normalized in {"approve", "success"} else ""


# -- Slack signature verification -------------------------------------------

# Maximum acceptable clock skew for timestamp validation (5 minutes).
SLACK_TIMESTAMP_MAX_DRIFT_SEC = 300
SLACK_SIGNING_VERSION = "v0"


def verify_slack_signature(
    *,
    signing_secret: str,
    timestamp: str,
    body: bytes,
    signature: str,
) -> bool:
    """
    Verify Slack ``X-Slack-Signature`` using ``v0:{ts}:{body}`` HMAC-SHA256.

    Fail-closed: returns False on any missing/invalid input.
    """
    if not signing_secret or not timestamp or not signature:
        return False

    # Timestamp freshness check
    try:
        ts_int = int(timestamp)
    except (ValueError, TypeError):
        return False

    if abs(time.time() - ts_int) > SLACK_TIMESTAMP_MAX_DRIFT_SEC:
        return False

    # Compute expected signature
    sig_basestring = f"{SLACK_SIGNING_VERSION}:{timestamp}:{body.decode('utf-8')}"
    expected = (
        SLACK_SIGNING_VERSION
        + "="
        + hmac.new(
            signing_secret.encode("utf-8"),
            sig_basestring.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    )

    return hmac.compare_digest(expected, signature)


# -- Slack adapter ----------------------------------------------------------


class SlackWebhookServer(
    SlackInstallationMixin,
    SlackIngressMixin,
    SlackDeliveryMixin,
):
    """
    F56 -- Slack Events API adapter.

    Security invariants (S67 / R124):
    - CRITICAL: Reject unsigned or replay requests (fail-closed).
    - CRITICAL: Ignore bot's own messages (bot-loop prevention).
    - IMPORTANT: Deduplicate ``message`` + ``app_mention`` for the same event
      to prevent double command execution.
    - IMPORTANT: Respect ``require_mention`` policy for group channels.
    """

    REPLAY_WINDOW_SEC = 300
    NONCE_CACHE_SIZE = 5000

    def __init__(self, config: ConnectorConfig, router: CommandRouter):
        self.config = config
        self.router = router
        self.app = None
        self.runner = None
        self.site = None

        # S67: Replay / dedupe guard keyed by Slack event_id
        self._replay_guard = ReplayGuard(
            window_sec=self.REPLAY_WINDOW_SEC,
            max_entries=self.NONCE_CACHE_SIZE,
        )
        if ConnectorReplayLifecycle is None:  # pragma: no cover
            self._interaction_lifecycle = None
        else:
            self._interaction_lifecycle = ConnectorReplayLifecycle(
                ttl_sec=self.REPLAY_WINDOW_SEC,
                max_entries=self.NONCE_CACHE_SIZE,
            )

        # S67: Allowlists (fail-closed when configured)
        self._user_allowlist = AllowlistPolicy(config.slack_allowed_users, strict=False)
        self._channel_allowlist = AllowlistPolicy(
            config.slack_allowed_channels, strict=False
        )
        self._installation_manager = SlackInstallationManager(config)

        # Bot user ID (resolved on first event or set from config)
        self._bot_user_id: Optional[str] = None  # type: ignore[assignment]
        self._bot_user_ids: Dict[str, str] = {}

    # IMPORTANT: resolve facade globals at call time; integration suites and
    # minimal-host shims patch these security/protocol seams directly.
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
    def _adapter_make_redirect_response(*args, **kwargs):
        return _make_redirect_response(*args, **kwargs)

    @staticmethod
    def _adapter_safe_external_error_text(*args, **kwargs):
        return _safe_external_error_text(*args, **kwargs)

    @staticmethod
    def _adapter_verify_slack_signature(*args, **kwargs):
        return verify_slack_signature(*args, **kwargs)

    @staticmethod
    def _adapter_json_loads_safe(*args, **kwargs):
        return _json_loads_safe(*args, **kwargs)

    @staticmethod
    def _adapter_first_non_empty(*args, **kwargs):
        return _first_non_empty(*args, **kwargs)

    @staticmethod
    def _adapter_force_approval_command(*args, **kwargs):
        return _force_approval_command(*args, **kwargs)

    @staticmethod
    def _adapter_style_to_slack(*args, **kwargs):
        return _style_to_slack(*args, **kwargs)

    @staticmethod
    def _adapter_channel_kind(*args, **kwargs):
        return _slack_channel_kind(*args, **kwargs)

    @staticmethod
    def _adapter_interaction_types():
        return _SLACK_INTERACTION_TYPES

    @staticmethod
    def _adapter_logger():
        return logger

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        aiohttp, web = _import_aiohttp_web()
        if aiohttp is None or web is None:
            logger.warning("aiohttp not installed. Skipping Slack adapter.")
            return

        if not self.config.slack_signing_secret:
            logger.info(
                "Slack adapter disabled "
                "(OPENCLAW_CONNECTOR_SLACK_SIGNING_SECRET missing)"
            )
            return
        if (
            not self.config.slack_bot_token
            and not self._installation_manager.can_handle_oauth()
        ):
            logger.info(
                "Slack adapter disabled "
                "(legacy bot token missing and Slack OAuth flow not configured)"
            )
            return

        logger.info(
            f"Starting Slack Webhook on "
            f"{self.config.slack_bind_host}:{self.config.slack_bind_port}"
            f"{self.config.slack_webhook_path}"
        )

        self.app = web.Application()
        self.app.router.add_post(self.config.slack_webhook_path, self.handle_event)
        self.app.router.add_post(
            self.config.slack_interactions_path, self.handle_interaction
        )
        if self._installation_manager.can_handle_oauth():
            self.app.router.add_get(
                self.config.slack_oauth_install_path, self.handle_oauth_install
            )
            self.app.router.add_get(
                self.config.slack_oauth_callback_path, self.handle_oauth_callback
            )

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(
            self.runner, self.config.slack_bind_host, self.config.slack_bind_port
        )
        await self.site.start()

    async def stop(self):
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------
