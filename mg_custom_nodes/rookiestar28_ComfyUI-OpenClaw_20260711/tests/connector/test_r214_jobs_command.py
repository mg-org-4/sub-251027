"""Connector jobs v1 parsing, summary, fallback, and authorization contracts."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from connector.config import ConnectorConfig
from connector.contract import CommandRequest
from connector.router import CommandRouter


def _request(text: str = "/jobs", *, sender: str = "admin-user") -> CommandRequest:
    return CommandRequest(
        platform="test",
        sender_id=sender,
        channel_id="channel-1",
        username="operator",
        message_id="message-1",
        text=text,
        timestamp=0,
    )


def _job(job_id: str, status: str, **hostile) -> dict:
    return {"id": job_id, "status": status, **hostile}


def _success(jobs: list[dict], *, total: int | None = None, version=1) -> dict:
    body = {
        "ok": True,
        "contract_version": version,
        "jobs": jobs,
        "pagination": {
            "offset": 0,
            "limit": 50,
            "total": len(jobs) if total is None else total,
            "has_more": (len(jobs) if total is None else total) > len(jobs),
            "warnings": [],
        },
        "source": {"adapter": "comfy_execution.jobs", "authority": "in_process"},
        "scan": {
            "window": 10000,
            "examined": len(jobs),
            "excluded": 0,
            "malformed": 0,
            "truncated": False,
        },
    }
    return {"ok": True, "status": 200, "data": body}


class TestConnectorJobsCommand(unittest.TestCase):
    def setUp(self):
        config = ConnectorConfig()
        config.admin_users = ["admin-user"]
        config.admin_token = "configured-admin-token"
        self.client = MagicMock()
        self.client.get_prompt_queue = AsyncMock(
            return_value={
                "ok": True,
                "status": 200,
                "data": {"exec_info": {"queue_remaining": 7}},
            }
        )
        self.client.get_jobs = AsyncMock(return_value=_success([]))
        self.router = CommandRouter(config, self.client)

    def _run(self, text: str = "/jobs", *, sender: str = "admin-user"):
        return asyncio.run(self.router.handle(_request(text, sender=sender)))

    def test_active_terminal_mix_is_bounded_and_never_dumps_raw_payload(self):
        jobs = [
            _job(
                "job-pending",
                "pending",
                workflow_id="secret-workflow",
                preview_output={"filename": "secret.png"},
                error="secret-error",
                tenant_id="secret-tenant",
                trace_id="secret-trace",
                prompt="secret-prompt",
                reasoning="secret-reasoning",
                internal="secret-internal",
            ),
            _job("job-running", "in_progress"),
            _job("job-completed", "completed"),
            _job("job-failed", "failed"),
            _job("job-cancelled", "cancelled"),
            _job("job-sixth", "completed"),
        ]
        self.client.get_jobs.return_value = _success(jobs, total=12)

        response = self._run()

        self.assertIn("[Jobs] Authoritative snapshot", response.text)
        self.assertIn("Snapshot total: 12; returned page: 6", response.text)
        self.assertIn("Active 2 (pending 1, in progress 1)", response.text)
        self.assertIn("Terminal 4 (completed 2, failed 1, cancelled 1)", response.text)
        self.assertIn("- job-pending — pending", response.text)
        self.assertIn("- job-running — in progress", response.text)
        self.assertIn("Showing 5 of 6 returned jobs", response.text)
        self.assertNotIn("job-sixth", response.text)
        for secret in (
            "secret-workflow",
            "secret.png",
            "secret-error",
            "secret-tenant",
            "secret-trace",
            "secret-prompt",
            "secret-reasoning",
            "secret-internal",
            "workflow_id",
            "preview_output",
        ):
            self.assertNotIn(secret, response.text)
        self.assertNotIn(str(jobs), response.text)
        self.assertNotIn("{'", response.text)
        self.assertLessEqual(len(response.text), 1000)
        self.client.get_prompt_queue.assert_not_called()

    def test_authoritative_empty_is_distinct_and_never_falls_back(self):
        response = self._run()

        self.assertEqual(response.text, "[Jobs] No jobs in the authoritative snapshot.")
        self.client.get_prompt_queue.assert_not_called()

    def test_exact_unsupported_and_degraded_errors_use_labeled_coarse_fallback(self):
        cases = (
            (501, "jobs_host_contract_unsupported"),
            (503, "jobs_backend_unavailable"),
        )
        for status, error in cases:
            with self.subTest(status=status):
                self.client.get_jobs.return_value = {
                    "ok": False,
                    "status": status,
                    "error": error,
                    "data": {"ok": False, "error": error},
                }
                self.client.get_prompt_queue.reset_mock()

                response = self._run()

                self.assertEqual(
                    response.text,
                    "[Jobs fallback] Queue remaining: 7 "
                    "(coarse count; not an authoritative jobs snapshot).",
                )
                self.client.get_prompt_queue.assert_awaited_once()

    def test_access_denial_never_uses_queue_fallback_or_raw_error(self):
        self.client.get_jobs.return_value = {
            "ok": False,
            "status": 403,
            "error": "secret-auth-detail",
            "data": {"error": "secret-auth-detail"},
        }

        response = self._run()

        self.assertEqual(
            response.text,
            "[Jobs] Access denied. Check connector Admin authorization and token posture.",
        )
        self.assertNotIn("secret-auth-detail", response.text)
        self.client.get_prompt_queue.assert_not_called()

    def test_transport_and_arbitrary_errors_never_fallback(self):
        cases = (
            {"ok": False, "error": "secret-network-failure"},
            {"ok": False, "status": 500, "error": "secret-server-failure"},
            {
                "ok": False,
                "status": 501,
                "error": "some-other-unsupported-secret",
            },
        )
        for payload in cases:
            with self.subTest(payload=payload.get("status")):
                self.client.get_jobs.return_value = payload
                self.client.get_prompt_queue.reset_mock()
                response = self._run()
                self.assertEqual(
                    response.text,
                    "[Jobs] Could not fetch the authoritative jobs snapshot.",
                )
                self.assertNotIn("secret", response.text)
                self.client.get_prompt_queue.assert_not_called()

    def test_unknown_or_malformed_success_is_content_free_and_does_not_fallback(self):
        oversized = _success([])
        oversized["data"]["jobs"] = [
            _job(f"job-{index}", "pending") for index in range(201)
        ]
        bad_total = _success([])
        bad_total["data"]["pagination"]["total"] = True
        inconsistent = _success([_job("job", "pending")])
        inconsistent["data"]["pagination"]["has_more"] = True
        malformed = (
            _success([], version=2),
            {"ok": True, "status": 200, "data": {"contract_version": 1}},
            {
                "ok": True,
                "status": 200,
                "data": {
                    "ok": True,
                    "contract_version": 1,
                    "jobs": "secret-not-a-list",
                    "pagination": {},
                },
            },
            _success([_job("job", "unknown-secret-status")]),
            _success([_job("job\nsecret-control", "pending")]),
            _success([_job("<b>secret-markup</b>", "pending")]),
            oversized,
            bad_total,
            inconsistent,
        )
        for payload in malformed:
            with self.subTest(payload=json.dumps(payload, default=str)[:40]):
                self.client.get_jobs.return_value = payload
                self.client.get_prompt_queue.reset_mock()
                response = self._run()
                self.assertEqual(
                    response.text,
                    "[Jobs] Malformed or unsupported jobs response.",
                )
                self.assertNotIn("secret", response.text)
                self.client.get_prompt_queue.assert_not_called()

    def test_malformed_queue_count_cannot_escape_fallback_boundary(self):
        self.client.get_jobs.return_value = {
            "ok": False,
            "status": 503,
            "error": "jobs_backend_unavailable",
        }
        for value in (True, -1, 1000001, "secret-unbounded"):
            with self.subTest(value=value):
                self.client.get_prompt_queue.return_value = {
                    "ok": True,
                    "data": {"exec_info": {"queue_remaining": value}},
                }
                response = self._run()
                self.assertEqual(
                    response.text,
                    "[Jobs fallback] Coarse queue count is unavailable.",
                )
                self.assertNotIn("secret", response.text)

    def test_non_admin_aliases_are_denied_before_backend_call(self):
        for command in ("/jobs", "jobs", "queue"):
            with self.subTest(command=command):
                self.client.get_jobs.reset_mock()
                response = self._run(command, sender="ordinary-user")
                self.assertIn("Access Denied", response.text)
                self.client.get_jobs.assert_not_called()

    def test_admin_aliases_share_the_v1_contract(self):
        for command in ("/jobs", "jobs", "queue"):
            with self.subTest(command=command):
                self.client.get_jobs.reset_mock()
                response = self._run(command)
                self.assertEqual(
                    response.text, "[Jobs] No jobs in the authoritative snapshot."
                )
                self.client.get_jobs.assert_awaited_once()

    def test_missing_backend_admin_token_fails_before_jobs_request(self):
        self.router.config.admin_token = None

        response = self._run()

        self.assertIn("Admin token not configured", response.text)
        self.client.get_jobs.assert_not_called()

    def test_long_safe_job_id_is_display_capped(self):
        long_id = "job-" + "a" * 80
        self.client.get_jobs.return_value = _success([_job(long_id, "completed")])

        response = self._run()

        rendered_id = response.text.split("- ", 1)[1].split(" —", 1)[0]
        self.assertEqual(len(rendered_id), 24)
        self.assertTrue(rendered_id.endswith("..."))
        self.assertNotIn(long_id, response.text)

    def test_help_places_jobs_in_admin_section(self):
        response = self._run("/help", sender="ordinary-user")
        public, admin = response.text.split("Admin Only:", 1)
        self.assertNotIn("/jobs", public)
        self.assertIn("/jobs - Authoritative jobs summary", admin)

    def test_public_chat_status_does_not_fetch_or_forward_admin_jobs(self):
        self.client.get_health = AsyncMock(return_value={"ok": True, "data": {}})
        llm = MagicMock()
        llm.chat = AsyncMock(return_value="bounded status")

        response = asyncio.run(self.router._chat_status(llm))

        self.assertEqual(response.text, "bounded status")
        self.client.get_jobs.assert_not_called()
        user_prompt = llm.chat.await_args.args[1]
        self.assertIn("admin-only; use /jobs as an authorized operator", user_prompt)
        self.assertNotIn("contract_version", user_prompt)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
