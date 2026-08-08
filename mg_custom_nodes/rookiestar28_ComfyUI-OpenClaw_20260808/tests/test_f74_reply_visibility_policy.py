import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from connector.config import ConnectorConfig
from connector.contract import CommandResponse, Platform
from connector.platforms.feishu_webhook import FeishuWebhookServer
from connector.platforms.slack_webhook import SlackWebhookServer
from connector.platforms.telegram_polling import TelegramPolling
from connector.reply_visibility import decide_reply_visibility
from connector.results_poller import ResultsPoller


class _FakeResponse:
    status = 200

    async def text(self):
        return "OK"

    async def json(self):
        return {"ok": True}


class _FakePostContext:
    async def __aenter__(self):
        return _FakeResponse()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self):
        self.posts = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return _FakePostContext()


class _MockPlatform(Platform):
    async def send_image(
        self,
        channel_id,
        image_data,
        filename="image.png",
        caption=None,
        delivery_context=None,
    ):
        pass

    async def send_message(self, channel_id, text, delivery_context=None):
        pass


class TestF74ReplyVisibilityPolicy(unittest.IsolatedAsyncioTestCase):
    def test_shared_policy_matrix(self):
        cases = [
            (
                "dm",
                decide_reply_visibility(
                    delivery_context={"chat_type": "p2p"}, platform="feishu", text="ok"
                ),
                True,
                "visible",
            ),
            (
                "group mention",
                decide_reply_visibility(
                    delivery_context={"chat_type": "group", "mentioned_bot": True},
                    platform="feishu",
                    text="ok",
                ),
                True,
                "visible",
            ),
            (
                "group no mention",
                decide_reply_visibility(
                    delivery_context={"chat_type": "group", "mentioned_bot": False},
                    platform="feishu",
                    text="ok",
                ),
                False,
                "group_no_mention",
            ),
            (
                "thread",
                decide_reply_visibility(
                    delivery_context={
                        "chat_type": "group",
                        "mentioned_bot": False,
                        "thread_id": "t-1",
                    },
                    platform="slack",
                    text="ok",
                ),
                True,
                "visible",
            ),
            (
                "internal",
                decide_reply_visibility(
                    delivery_context={"internal_delivery": True},
                    platform="telegram",
                    text="ok",
                ),
                False,
                "internal_delivery",
            ),
            (
                "tool only",
                decide_reply_visibility(
                    delivery_context={"reply_visibility": "tool_only"},
                    platform="telegram",
                    text="ok",
                ),
                False,
                "text_reply_suppressed",
            ),
            (
                "interactive",
                decide_reply_visibility(
                    delivery_context={"reply_visibility": "tool_only"},
                    platform="slack",
                    text="approval",
                    has_buttons=True,
                ),
                True,
                "interactive_action_required",
            ),
        ]
        for label, decision, visible, reason in cases:
            with self.subTest(label=label):
                self.assertEqual(decision.visible, visible)
                self.assertEqual(decision.reason, reason)

    async def test_result_poller_suppressed_text_is_successful_noop(self):
        config = ConnectorConfig()
        client = MagicMock()
        platform = _MockPlatform()
        platform.send_message = AsyncMock(side_effect=AssertionError("must not send"))
        platform.send_image = AsyncMock()
        poller = ResultsPoller(config, client, {"test": platform})

        await poller._deliver_results(
            "p-quiet",
            {"outputs": {}},
            "test",
            "c-1",
            delivery_context={"reply_visibility": "tool_only"},
        )

        platform.send_message.assert_not_called()
        platform.send_image.assert_not_called()


class TestF74ReplyVisibilityAdapters(unittest.IsolatedAsyncioTestCase):
    async def test_telegram_suppressed_response_returns_success_without_http_send(self):
        config = ConnectorConfig()
        config.telegram_bot_token = "telegram-token"
        server = TelegramPolling(config, MagicMock())
        server.session = _FakeSession()

        delivered = await server._send_response(
            -100123,
            CommandResponse(text="quiet"),
            delivery_context={"reply_visibility": "tool_only"},
        )

        self.assertTrue(delivered)
        self.assertEqual(server.session.posts, [])

    async def test_slack_suppressed_reply_skips_chat_post(self):
        config = ConnectorConfig()
        config.slack_bot_token = "slack-token"
        server = SlackWebhookServer(config, MagicMock())

        fake_aiohttp = MagicMock()
        with patch.dict(sys.modules, {"aiohttp": fake_aiohttp}):
            await server._send_reply(
                channel_id="C_F74",
                text="quiet",
                delivery_context={"reply_visibility": "tool_only"},
            )

        fake_aiohttp.ClientSession.assert_not_called()

    async def test_feishu_suppressed_reply_skips_open_api_send(self):
        config = ConnectorConfig()
        server = FeishuWebhookServer(config, MagicMock())

        with patch(
            "connector.platforms.feishu_webhook.safe_request_json"
        ) as mock_safe_request:
            await server.send_message(
                "oc_group_1",
                "quiet",
                delivery_context={"reply_visibility": "tool_only"},
            )

        mock_safe_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
