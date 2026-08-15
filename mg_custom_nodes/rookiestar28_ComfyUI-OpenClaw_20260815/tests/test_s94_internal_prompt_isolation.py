import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from connector.config import ConnectorConfig
from connector.contract import CommandRequest
from connector.router import CommandRouter
from services.audit_events import build_audit_event
from services.job_events import JobEvent

try:
    import api.routes

    AIOHTTP_AVAILABLE = True
except Exception:
    AIOHTTP_AVAILABLE = False


class TestS94InternalPromptIsolation(unittest.TestCase):
    def test_operator_sanitizer_drops_marked_internal_content_only(self):
        from services.internal_content import mark_internal_content
        from services.reasoning_redaction import sanitize_operator_payload

        payload = {
            "messages": [
                {"role": "user", "content": "user asked about internal lighting"},
                mark_internal_content(
                    "flush compaction helper prompt",
                    kind="maintenance_prompt",
                    metadata={"phase": "compaction"},
                ),
            ]
        }

        sanitized = sanitize_operator_payload(payload)

        self.assertEqual(
            sanitized["messages"],
            [{"role": "user", "content": "user asked about internal lighting"}],
        )
        self.assertNotIn("flush compaction helper prompt", str(sanitized))

    def test_job_event_json_and_sse_drop_internal_content(self):
        from services.internal_content import mark_internal_content

        evt = JobEvent(
            seq=1,
            event_type="completed",
            prompt_id="p1",
            trace_id="t1",
            data={
                "visible": "final status",
                "messages": [mark_internal_content("repair prompt body")],
            },
        )

        self.assertEqual(evt.to_dict()["data"]["visible"], "final status")
        self.assertEqual(evt.to_dict()["data"]["messages"], [])
        self.assertNotIn("repair prompt body", evt.to_sse())

    def test_audit_event_payload_and_meta_drop_internal_content(self):
        from services.internal_content import mark_internal_content

        event = build_audit_event(
            "maintenance.test",
            payload={
                "status": "ok",
                "prompt": mark_internal_content("diagnostic repair prompt"),
            },
            meta={"note": mark_internal_content("transcript flush prompt")},
        )

        self.assertEqual(event["payload"], {"status": "ok"})
        self.assertEqual(event["meta"], {})
        self.assertNotIn("diagnostic repair prompt", json.dumps(event))
        self.assertNotIn("transcript flush prompt", json.dumps(event))


@unittest.skipUnless(AIOHTTP_AVAILABLE, "aiohttp not available")
class TestS94TraceHandler(unittest.IsolatedAsyncioTestCase):
    async def test_trace_handler_drops_internal_content(self):
        from services.internal_content import mark_internal_content

        mock_request = MagicMock()
        mock_request.match_info = {"prompt_id": "p1"}
        mock_request.headers = {}
        mock_request.query = {}

        mock_web = MagicMock()
        mock_record = MagicMock()
        mock_record.to_dict.return_value = {
            "prompt_id": "p1",
            "events": [
                {
                    "event": "maintenance",
                    "meta": {
                        "visible": "kept",
                        "internal": mark_internal_content("hidden maintenance text"),
                    },
                }
            ],
        }

        with (
            patch.object(api.routes, "web", mock_web),
            patch.object(
                api.routes,
                "_ensure_observability_deps_ready",
                return_value=(True, None),
            ),
            patch.object(api.routes, "require_admin_token", return_value=(True, None)),
            patch.object(api.routes, "trace_store") as mock_trace_store,
            patch(
                "services.reasoning_redaction.get_client_ip", return_value="127.0.0.1"
            ),
        ):
            mock_trace_store.get.return_value = mock_record
            await api.routes.trace_handler(mock_request)

        args, _ = mock_web.json_response.call_args
        trace = args[0]["trace"]
        self.assertEqual(trace["events"][0]["meta"], {"visible": "kept"})
        self.assertNotIn("hidden maintenance text", str(trace))


class TestS94ConnectorTrace(unittest.IsolatedAsyncioTestCase):
    async def test_connector_trace_command_drops_internal_content(self):
        from services.internal_content import mark_internal_content

        client = MagicMock()
        client.get_trace = AsyncMock(
            return_value={
                "ok": True,
                "data": {
                    "events": [
                        {
                            "meta": {
                                "status": "ok",
                                "internal": mark_internal_content(
                                    "connector-visible repair prompt"
                                ),
                            }
                        }
                    ]
                },
            }
        )
        config = ConnectorConfig()
        config.admin_users = ["admin1"]
        config.admin_token = "token"
        router = CommandRouter(config, client)

        req = CommandRequest(
            platform="telegram",
            channel_id="c1",
            sender_id="admin1",
            username="tester",
            message_id="m1",
            text="/trace p1",
            timestamp=1.0,
        )

        resp = await router.handle(req)

        self.assertIn("status", resp.text)
        self.assertNotIn("connector-visible repair prompt", resp.text)


if __name__ == "__main__":
    unittest.main()
