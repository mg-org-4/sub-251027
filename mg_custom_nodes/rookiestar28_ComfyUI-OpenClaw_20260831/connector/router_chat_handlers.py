"""Owned chat and semantic-guard command-family mixin."""

# ruff: noqa: UP006, UP035 -- preserve the frozen public annotations.
# mypy: disable-error-code="attr-defined"

from typing import Any, Dict, List

from .contract import CommandRequest, CommandResponse
from .llm_client import LLMClient
from .prompts import CHAT_STATUS_PROMPT, CHAT_SYSTEM_PROMPT
from .semantic_guard import GuardAction


class RouterChatMixin:
    async def _handle_chat(
        self, req: CommandRequest, args: List[str]
    ) -> CommandResponse:
        """
        /chat [subcommand] <message>
        Subcommands: run, template, status
        Default: general chat

        Security: Never auto-executes commands. Only suggests command text.
        """
        llm = self._build_llm_client()

        if not await llm.is_configured():
            return CommandResponse(
                text="[Chat Error] LLM not configured. Configure in OpenClaw Settings."
            )

        # Parse subcommand
        if not args:
            return CommandResponse(
                text="Usage: /chat <message> or /chat run|template|status <request>"
            )

        subcommand = args[0].lower()
        message = " ".join(args[1:]) if len(args) > 1 else ""

        trust_level = "TRUSTED" if self._is_trusted(req) else "UNTRUSTED"

        if subcommand == "run":
            return await self._chat_run(llm, message, trust_level)
        elif subcommand == "template":
            return await self._chat_template(llm, message)
        elif subcommand == "status":
            return await self._chat_status(llm)
        else:
            # General chat: first word is part of message
            full_message = " ".join(args)
            return await self._chat_general(llm, full_message, trust_level)

    async def _chat_general(
        self, llm: LLMClient, message: str, trust_level: str
    ) -> CommandResponse:
        """General chat with assistant."""
        # S44: Semantic Guard Evaluation
        decision = self.semantic_guard.evaluate_request(message, {"trust": trust_level})

        if decision.action == GuardAction.DENY:
            return CommandResponse(
                text=(
                    "[Blocked] Request denied by semantic policy "
                    f"({decision.reason}). {self._policy_kv(decision.to_contract())}"
                )
            )

        system_prompt = CHAT_SYSTEM_PROMPT.format(trust_level=trust_level)
        response = await llm.chat(system_prompt, message)

        # S44: Output Validation + SAFE_REPLY sanitization.
        try:
            response = self.semantic_guard.validate_output(
                response, "general", decision.action
            )
        except ValueError as e:
            return CommandResponse(
                text=(
                    "[Validation Error] Assistant output invalid: "
                    f"{e}. {self._policy_kv({'code': 'semantic_output_invalid', 'severity': 'medium', 'action': 'deny', 'reason': str(e)})}"
                )
            )

        if decision.action == GuardAction.SAFE_REPLY:
            safe_response = (
                response
                or "I can help with general guidance, but commands are restricted for this request."
            )
            return CommandResponse(
                text=(
                    f"[Safe Mode] {safe_response}\n\n"
                    f"(Policy: {self._policy_kv(decision.to_contract())})"
                )
            )

        return CommandResponse(text=response)

    async def _chat_run(
        self, llm: LLMClient, request: str, trust_level: str
    ) -> CommandResponse:
        """Suggest a /run command based on user request."""
        if not request:
            return CommandResponse(
                text="Usage: /chat run <description of what you want>"
            )

        # S44: Semantic Guard Evaluation
        decision = self.semantic_guard.evaluate_request(request, {"trust": trust_level})

        if decision.action == GuardAction.DENY:
            return CommandResponse(
                text=(
                    "[Blocked] Request denied by semantic policy "
                    f"({decision.reason}). {self._policy_kv(decision.to_contract())}"
                )
            )

        # Force Approval Override based on Risk
        force_approval_policy = decision.action == GuardAction.FORCE_APPROVAL

        # Get available templates (simplified - could fetch from API)
        templates = "txt2img, img2img, upscale (examples)"

        system_prompt = CHAT_SYSTEM_PROMPT.format(trust_level=trust_level)
        user_prompt = f"""User wants to run a generation. Suggest a `/run` command.

Request: {request}
Available templates: {templates}
Trust level: {trust_level}

Remember: {"add --approval flag" if trust_level == "UNTRUSTED" else "no --approval needed"}.
Output only the command in a code block."""

        response = await llm.chat(system_prompt, user_prompt)

        # S44: Output Structure Validation
        try:
            response = self.semantic_guard.validate_output(
                response, "run", decision.action
            )
        except ValueError as e:
            return CommandResponse(
                text=(
                    "[Validation Error] Assistant output invalid: "
                    f"{e}. {self._policy_kv({'code': 'semantic_output_invalid', 'severity': 'high', 'action': 'deny', 'reason': str(e)})}"
                )
            )

        # R97: Command Firewall - Extract and Validate
        import re

        cmd_match = re.search(r"```(?:bash)?\s*(.*?)\s*```", response, re.DOTALL)
        raw_cmd = cmd_match.group(1).strip() if cmd_match else response.strip()

        # Validate through Firewall
        normalized = self.command_firewall.validate_suggestion(raw_cmd)

        if not normalized.is_safe:
            return CommandResponse(
                text=(
                    "[Safety Block] Assistant suggested unsafe command: "
                    f"{normalized.safety_reason}. {self._policy_kv(normalized.to_contract())}"
                )
            )

        # R97: Strict /run enforcement (Remediation for Medium Severity)
        # CRITICAL: keep this check. /chat run must never emit non-/run commands.
        if normalized.command != "/run":
            return CommandResponse(
                text=(
                    "[Policy Block] Only /run commands are allowed in this mode. "
                    f"Got: {normalized.command}. "
                    f"{self._policy_kv({'code': 'firewall_non_run_command', 'severity': 'high', 'action': 'deny', 'reason': 'non_run_command_in_run_mode'})}"
                )
            )

        # R97/S44: Apply Policy Overrides
        # If risk was elevated, ensure --approval is present
        if (
            force_approval_policy
            and "--approval" not in normalized.args
            and "approval" not in normalized.flags
        ):
            normalized.args.append("--approval")

        final_cmd = normalized.to_string()

        # Return as code block for easy copy-paste (or auto-execution UI cues)
        if force_approval_policy:
            return CommandResponse(
                text=(
                    f"```\n{final_cmd}\n```\n"
                    f"(Policy: {self._policy_kv(decision.to_contract())})"
                )
            )
        return CommandResponse(text=f"```\n{final_cmd}\n```")

    @staticmethod
    def _policy_kv(contract: Dict[str, Any]) -> str:
        ordered = ("code", "severity", "action", "reason")
        parts = []
        for key in ordered:
            value = contract.get(key)
            if value is not None:
                parts.append(f"{key}={value}")
        return "[" + ", ".join(parts) + "]"

    async def _chat_template(self, llm: LLMClient, request: str) -> CommandResponse:
        """Generate a template JSON suggestion."""
        if not request:
            return CommandResponse(text="Usage: /chat template <description>")

        system_prompt = CHAT_SYSTEM_PROMPT.format(trust_level="N/A")
        user_prompt = f"""Generate a workflow template JSON for this request:

Request: {request}

Output:
1. Suggested filename
2. Template JSON in a code block

Keep it minimal."""

        response = await llm.chat(system_prompt, user_prompt)
        return CommandResponse(text=response)

    async def _chat_status(self, llm: LLMClient) -> CommandResponse:
        """Summarize system status using LLM."""
        # Fetch status data
        health = await self.client.get_health()
        queue = await self.client.get_prompt_queue()

        status_data = {
            "health": health.get("data", {}) if health.get("ok") else "unavailable",
            "jobs": "admin-only; use /jobs as an authorized operator",
            "queue": queue.get("data", {}) if queue.get("ok") else "unavailable",
        }

        system_prompt = CHAT_SYSTEM_PROMPT.format(trust_level="N/A")
        user_prompt = CHAT_STATUS_PROMPT.format(status_data=status_data)

        response = await llm.chat(system_prompt, user_prompt)
        return CommandResponse(text=response)
