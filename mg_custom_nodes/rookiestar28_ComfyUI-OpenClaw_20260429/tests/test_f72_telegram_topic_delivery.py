import unittest
from unittest.mock import AsyncMock

from connector.config import ConnectorConfig
from connector.contract import CommandResponse
from connector.platforms.telegram_polling import TelegramPolling


class _FakeResponse:
    status = 200

    async def text(self):
        return "ok"


class _FakePostContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self):
        self.posts = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return _FakePostContext(_FakeResponse())


class _FakeRouter:
    def __init__(self):
        self.requests = []

    async def handle(self, req):
        self.requests.append(req)
        return CommandResponse(text="topic reply")


def _form_field_value(form_data, name):
    for headers, _extra, value in form_data._fields:
        if headers.get("name") == name:
            return value
    return None


class TestTelegramTopicDelivery(unittest.IsolatedAsyncioTestCase):
    def _server(self):
        cfg = ConnectorConfig()
        cfg.telegram_bot_token = "token"
        cfg.telegram_allowed_chats = [-100123]
        router = _FakeRouter()
        server = TelegramPolling(cfg, router)
        server.session = _FakeSession()
        return server, router

    async def test_inbound_topic_update_preserves_thread_and_replies_to_topic(self):
        server, router = self._server()

        await server._process_update(
            {
                "update_id": 1,
                "message": {
                    "message_id": 77,
                    "message_thread_id": 456,
                    "chat": {"id": -100123, "type": "supergroup"},
                    "from": {"id": 42, "username": "alice"},
                    "text": "/status",
                },
            }
        )

        self.assertEqual(router.requests[0].thread_id, "456")
        _url, kwargs = server.session.posts[-1]
        self.assertEqual(kwargs["json"]["message_thread_id"], 456)

    async def test_send_message_includes_valid_thread_id(self):
        server, _router = self._server()

        await server.send_message(
            "-100123",
            "done",
            delivery_context={"thread_id": "456"},
        )

        _url, kwargs = server.session.posts[-1]
        self.assertEqual(kwargs["json"]["message_thread_id"], 456)

    async def test_send_image_includes_valid_thread_id(self):
        server, _router = self._server()

        await server.send_image(
            "-100123",
            b"image",
            filename="out.png",
            delivery_context={"thread_id": "456"},
        )

        _url, kwargs = server.session.posts[-1]
        self.assertEqual(_form_field_value(kwargs["data"], "message_thread_id"), "456")

    async def test_malformed_thread_id_is_diagnostic_not_telegram_parameter(self):
        server, _router = self._server()
        server._send_thread_diagnostic = AsyncMock()

        await server.send_message(
            "-100123",
            "done",
            delivery_context={"thread_id": "abc123"},
        )

        _url, kwargs = server.session.posts[-1]
        self.assertNotIn("message_thread_id", kwargs["json"])
        server._send_thread_diagnostic.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
