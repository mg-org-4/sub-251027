import unittest
from unittest.mock import AsyncMock, MagicMock

from connector.config import ConnectorConfig
from connector.contract import CommandRequest
from connector.router import CommandRouter


class TestR181RouterHotspot(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = ConnectorConfig()
        self.client = MagicMock()
        self.client.get_health = AsyncMock(
            return_value={"ok": True, "data": {"stats": {}}}
        )
        self.client.get_prompt_queue = AsyncMock(
            return_value={"ok": True, "data": {"exec_info": {"queue_remaining": 2}}}
        )
        self.router = CommandRouter(self.config, self.client)

    def _req(self, text: str) -> CommandRequest:
        return CommandRequest(
            platform="telegram",
            channel_id="chat-1",
            sender_id="user-1",
            username="user",
            message_id="m-1",
            text=text,
            timestamp=0,
        )

    async def test_telegram_username_suffix_is_normalized_before_dispatch(self):
        response = await self.router.handle(self._req("/help@openclaw_bot"))

        self.assertIn("/run <template>", response.text)

    async def test_bot_mention_prefix_promotes_following_slash_command(self):
        response = await self.router.handle(self._req("@openclaw_bot /status"))

        self.assertIn("System Status", response.text)


if __name__ == "__main__":
    unittest.main()
