import unittest
from asyncio import run
from unittest.mock import MagicMock

# from aiohttp import web # Removed for CI compatibility (no extra deps)
from api.routes import register_dual_route


class TestRouteRegistration(unittest.TestCase):
    def test_dual_registration(self):
        # Mock Server
        server = MagicMock()
        server.routes = MagicMock()
        server.app = MagicMock()  # Mock the whole app
        server.app.router = MagicMock()
        server.app.router.add_route = MagicMock()

        # Test Handler (dummy)
        async def handler(req):
            return "response"

        # Call function
        register_dual_route(server, "GET", "/moltbot/test", handler)

        # 1. Verify standard PromptServer.routes usage
        server.routes.get.assert_called_with("/moltbot/test")

        # 2. Verify fallback app.router usage (legacy + shim)
        # Note: add_route calls might happen in any order
        calls = server.app.router.add_route.call_args_list
        # Expect ("GET", "/moltbot/test", handler) and ("GET", "/api/moltbot/test", handler)

        paths_registered = [c.args[1] for c in calls]
        self.assertIn("/moltbot/test", paths_registered)
        self.assertIn("/api/moltbot/test", paths_registered)

    def test_direct_legacy_fallback_uses_deprecation_wrapper(self):
        server = MagicMock()
        server.routes = MagicMock()
        server.app = MagicMock()
        server.app.router = MagicMock()
        server.app.router.add_route = MagicMock()

        class Response:
            def __init__(self):
                self.headers = {}

        class Request:
            path = "/api/moltbot/test"

        async def handler(req):
            return Response()

        register_dual_route(server, "GET", "/moltbot/test", handler)

        fallback_handler = None
        for call in server.app.router.add_route.call_args_list:
            if call.args[1] == "/api/moltbot/test":
                fallback_handler = call.args[2]
                break

        self.assertIsNotNone(fallback_handler)
        self.assertIsNot(fallback_handler, handler)

        response = run(fallback_handler(Request()))
        self.assertEqual(response.headers["Deprecation"], "true")
        self.assertEqual(
            response.headers["X-OpenClaw-Canonical-Path"],
            "/api/openclaw/test",
        )


if __name__ == "__main__":
    unittest.main()
