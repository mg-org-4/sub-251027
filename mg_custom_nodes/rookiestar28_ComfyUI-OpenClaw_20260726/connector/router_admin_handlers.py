"""Owned status, approval, schedule, and introspection command mixin."""

# ruff: noqa: UP006, UP035, UP045 -- preserve the frozen public annotations.
# mypy: disable-error-code="attr-defined"

from collections.abc import Mapping
from typing import List, Optional

from .contract import CommandRequest, CommandResponse
from .jobs_summary import JobsContractError, format_jobs_summary, format_queue_fallback

try:
    from services.reasoning_redaction import sanitize_operator_payload
except Exception:  # pragma: no cover - connector tests may stub import graph

    def sanitize_operator_payload(value, **_):  # type: ignore
        return value


class RouterAdminMixin:
    async def _handle_status(
        self, req: CommandRequest, args: List[str]
    ) -> CommandResponse:
        health = await self.client.get_health()
        queue = await self.client.get_prompt_queue()

        # New standardized response handling
        health_ok = health.get("ok")

        status_icon = "Online" if health_ok else "Offline"
        details = []

        if health_ok:
            data = health.get("data", {})
            stats = data.get("stats", {})
            details.append(f"Logs: {stats.get('logs_processed', 0)}")
            details.append(f"Errors: {stats.get('errors_captured', 0)}")

            q_res = queue.get("data", {})
            q_rem = q_res.get("exec_info", {}).get("queue_remaining", 0)
            details.append(f"Queue: {q_rem}")
        else:
            details.append(f"Error: {health.get('error')}")

        return CommandResponse(
            text=f"[{status_icon}] System Status\n"
            + "\n".join(f"- {d}" for d in details)
        )

    def _require_admin_token_configured(self) -> Optional[CommandResponse]:
        """
        F32 WP3: Check if admin token is configured before running admin commands.
        Fail-fast with clear error message instead of 403/500 later.

        IMPORTANT (recurring CI failure mode):
        - Admin-only commands are gated by BOTH:
          (1) sender is an admin user, AND
          (2) the connector admin token is configured (OPENCLAW_CONNECTOR_ADMIN_TOKEN).
        - Unit tests that exercise admin command handlers MUST set `config.admin_token`,
          otherwise they will correctly receive the config error response.
        """
        if not self.config.admin_token:
            return CommandResponse(
                text="[Error] Admin token not configured. Set OPENCLAW_CONNECTOR_ADMIN_TOKEN and restart connector."
            )
        return None

    async def _handle_approvals_list(
        self, req: CommandRequest, args: List[str]
    ) -> CommandResponse:
        # F32 WP3: Guard
        if err := self._require_admin_token_configured():
            return err

        res = await self.client.get_approvals()
        if not res.get("ok"):
            return CommandResponse(
                text=f"[Error] Failed to list approvals: {res.get('error')}"
            )

        items = res.get("items", [])
        if not items:
            return CommandResponse(text="No pending approvals.")

        pending_count = res.get("pending_count")
        lines = []
        buttons = []
        for i in items:
            # IMPORTANT (stability): the backend approval schema uses:
            # `approval_id`, `template_id`, `status`, `requested_by`, `source`.
            # Do not “simplify” these keys to `id/description/requester` unless you also
            # update the backend API + all tests. This mismatch previously caused silent
            # bad output and brittle regressions.
            approval_id = i.get("approval_id") or i.get("id") or "unknown"
            template_id = i.get("template_id") or "unknown"
            status = i.get("status") or "unknown"
            requested_by = i.get("requested_by") or "unknown"
            source = i.get("source") or "unknown"

            lines.append(
                f"- {approval_id} [{status}] template={template_id} by={requested_by} source={source}"
            )
        for i in items[:3]:
            approval_id = i.get("approval_id") or i.get("id") or "unknown"
            short_id = str(approval_id)[:8]
            buttons.append(
                {
                    "label": f"Approve {short_id}",
                    "value": f"/approve {approval_id}",
                    "action_type": "approval.approve",
                    "approval_id": approval_id,
                    "style": "primary",
                }
            )
            buttons.append(
                {
                    "label": f"Reject {short_id}",
                    "value": f"/reject {approval_id}",
                    "action_type": "approval.reject",
                    "approval_id": approval_id,
                    "style": "danger",
                }
            )

        header = "Pending Approvals"
        if isinstance(pending_count, int):
            header += f" ({pending_count})"
        return CommandResponse(
            text=header + ":\n" + "\n".join(lines),
            buttons=buttons,
        )

    async def _handle_approve(
        self, req: CommandRequest, args: List[str]
    ) -> CommandResponse:
        if not args:
            return CommandResponse(text="Usage: /approve <id>")

        # F32 WP3: Guard
        if err := self._require_admin_token_configured():
            return err

        # Assuming auto_execute=True by default for chat logic
        res = await self.client.approve_request(args[0], auto_execute=True)
        if not res.get("ok"):
            return CommandResponse(text=f"[Failed] {res.get('error')}")

        data = res.get("data", {})
        msg = f"[Approved] {args[0]}"

        # Phase 4: Show execution result
        if "prompt_id" in data:
            pid = data["prompt_id"]
            msg += f"\nExecuted: {pid}"
            if self.poller:
                # Approval request might have come from different flow, but usually user invoking /approve
                # wants the result. Using current req context is safest assumption for "ChatOps".
                self.poller.track_job(
                    pid,
                    req.platform,
                    req.channel_id,
                    req.sender_id,
                    delivery_context=self._delivery_context(req),
                )
        elif data.get("executed") is False:
            msg += "\n(Not Executed)"
            if err := data.get("execution_error"):
                msg += f"\nError: {err}"

        return CommandResponse(text=msg)

    async def _handle_reject(
        self, req: CommandRequest, args: List[str]
    ) -> CommandResponse:
        if not args:
            return CommandResponse(text="Usage: /reject <id> [reason]")

        # F32 WP3: Guard
        if err := self._require_admin_token_configured():
            return err

        reason = " ".join(args[1:]) if len(args) > 1 else "Rejected via chat"
        res = await self.client.reject_request(args[0], reason)
        if not res.get("ok"):
            return CommandResponse(text=f"[Failed] {res.get('error')}")

        return CommandResponse(text=f"[Rejected] {args[0]}")

    async def _handle_schedules_list(
        self, req: CommandRequest, args: List[str]
    ) -> CommandResponse:
        # F32 WP3: Guard
        if err := self._require_admin_token_configured():
            return err

        res = await self.client.get_schedules()
        if not res.get("ok"):
            return CommandResponse(text=f"[Error] {res.get('error')}")

        scheds = res.get("schedules", [])
        if not scheds:
            return CommandResponse(text="No schedules found.")

        lines = []
        for s in scheds:
            status = "+" if s.get("enabled") else "-"
            lines.append(
                f"[{status}] {s.get('id')}: {s.get('cron')} - {s.get('template_id')}"
            )

        return CommandResponse(text="Schedules:\n" + "\n".join(lines))

    async def _handle_schedule_subcommand(
        self, req: CommandRequest, args: List[str]
    ) -> CommandResponse:
        if len(args) < 2:
            return CommandResponse(text="Usage: /schedule <run|toggle> <id>")

        # F32 WP3: Guard
        if err := self._require_admin_token_configured():
            return err

        sub = args[0].lower()
        sid = args[1]

        if sub == "run":
            res = await self.client.run_schedule(sid)
            if not res.get("ok"):
                return CommandResponse(text=f"[Error] {res.get('error')}")
            return CommandResponse(text=f"[Success] Schedule {sid} triggered manually.")
        else:
            return CommandResponse(text="Not implemented yet.")

    async def _handle_help(
        self, req: CommandRequest, args: List[str]
    ) -> CommandResponse:
        return CommandResponse(
            text=(
                "OpenClaw Connector\n"
                "/status - Check system health and queue\n"
                "/run <template> [prompt] [k=v] - Run a generation (trusted users auto-exec; others require approval)\n"
                "/stop [job_id ...] - Cancel jobs by id; no args sends Global Interrupt (Admin)\n"
                "/history <id> - Job details\n"
                "Admin Only:\n"
                "/jobs - Authoritative jobs summary\n"
                "/approvals - List pending approvals\n"
                "/approve <id>, /reject <id>\n"
                "/schedules, /schedule run <id>\n"
                "/trace <id> - Execution trace"
            )
        )

    async def _handle_history(
        self, req: CommandRequest, args: List[str]
    ) -> CommandResponse:
        if not args:
            return CommandResponse(text="Usage: /history <prompt_id>")
        res = await self.client.get_history(args[0])
        if not res.get("ok"):
            return CommandResponse(text=f"[Error] {res.get('error')}")

        # Simple format
        data = res.get("data", {})
        status = data.get("status", {}).get("status_str", "unknown")
        # Assuming backend returns a structure we can summarise
        return CommandResponse(
            text=f"Job {args[0]}: {status}\nFull details: not implemented in connector view yet."
        )

    async def _handle_trace(
        self, req: CommandRequest, args: List[str]
    ) -> CommandResponse:
        if not args:
            return CommandResponse(text="Usage: /trace <prompt_id>")

        # F32 WP3: Guard
        if err := self._require_admin_token_configured():
            return err

        res = await self.client.get_trace(args[0])
        if not res.get("ok"):
            return CommandResponse(text=f"[Error] {res.get('error')}")

        # Dump trace
        sanitized = sanitize_operator_payload(res.get("data"))
        return CommandResponse(text=f"Trace {args[0]}: {str(sanitized)[:1000]}...")

    async def _handle_jobs(
        self, req: CommandRequest, args: List[str]
    ) -> CommandResponse:
        if err := self._require_admin_token_configured():
            return err

        res = await self.client.get_jobs()
        if not isinstance(res, Mapping):
            return CommandResponse(
                text="[Jobs] Could not fetch the authoritative jobs snapshot."
            )
        if res.get("ok") is True:
            try:
                return CommandResponse(text=format_jobs_summary(res.get("data")))
            except JobsContractError:
                return CommandResponse(
                    text="[Jobs] Malformed or unsupported jobs response."
                )

        status = res.get("status")
        error = res.get("error")
        access_denied = (
            isinstance(status, int)
            and not isinstance(status, bool)
            and status in {401, 403}
        )
        if access_denied:
            return CommandResponse(
                text="[Jobs] Access denied. Check connector Admin authorization and token posture."
            )
        fallback_allowed = isinstance(error, str) and (
            (status == 501 and error == "jobs_host_contract_unsupported")
            or (status == 503 and error == "jobs_backend_unavailable")
        )
        if fallback_allowed:
            return CommandResponse(
                text=format_queue_fallback(await self.client.get_prompt_queue())
            )
        return CommandResponse(
            text="[Jobs] Could not fetch the authoritative jobs snapshot."
        )

    # -------------------------------------------------------------------------
    # F30: Chat LLM Assistant
    # -------------------------------------------------------------------------
