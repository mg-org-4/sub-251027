"""
S41 Hardened Enforcement Gate.

Enforces mandatory security controls when running in HARDENED profile.
Fails startup if critical controls are missing or misconfigured.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Tuple

from .runtime_profile import get_runtime_profile, is_hardened_mode

if TYPE_CHECKING:
    from .effective_security_posture import EffectiveSecurityPosture

try:
    from .connector_allowlist_posture import evaluate_connector_allowlist_posture
except Exception:
    from services.connector_allowlist_posture import (  # type: ignore
        evaluate_connector_allowlist_posture,
    )

logger = logging.getLogger(__name__)


class SecurityGate:
    """
    Startup gate that strictly enforces security controls.
    """

    @staticmethod
    def _check_network_exposure() -> bool:
        """
        Check if the server is binding to a public/non-loopback interface.
        Inspects sys.argv for '--listen' or '0.0.0.0'.

        Returns:
            bool: True if potentially exposed to network, False if loopback only.
        """
        import sys

        args = sys.argv
        # Check for --listen flag (which defaults to 0.0.0.0 in ComfyUI)
        if "--listen" in args:
            return True

        # Check for explicit host bind
        # This is a heuristic; robust arg parsing is hard without importing main.
        # But for security gate, false positive is better than false negative.
        # If any arg looks like an IP that isn't loopback...
        # For now, rely on --listen as the primary signal.
        return False

    @staticmethod
    def verify_mandatory_controls(
        posture: EffectiveSecurityPosture | None = None,
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Check if all mandatory controls for the current profile are active.
        Returns: (passed: bool, warnings: List[str], fatal_errors: List[str])
        """
        warnings = []
        fatal_errors = []
        hardened = (
            posture.runtime_profile == "hardened"
            if posture is not None
            else is_hardened_mode()
        )
        runtime_profile_value = (
            posture.runtime_profile
            if posture is not None
            else get_runtime_profile().value
        )

        def _emit_startup_audit(action: str, outcome: str, details: dict) -> None:
            try:
                from .audit import emit_audit_event
            except Exception:
                try:
                    from services.audit import emit_audit_event  # type: ignore
                except Exception:
                    return
            emit_audit_event(
                action=action,
                target="startup",
                outcome=outcome,
                status_code=0,
                details=details,
            )

        # IMPORTANT: keep this as startup visibility only (not gate-fatal).
        # OPENCLAW_LOCALHOST_ALLOW_NO_ORIGIN is an explicit operator override for
        # localhost tooling; surfacing it early avoids silent CSRF-boundary drift.
        allow_no_origin = (
            posture.localhost_no_origin_override
            if posture is not None
            else (
                os.environ.get("OPENCLAW_LOCALHOST_ALLOW_NO_ORIGIN", "").strip().lower()
                == "true"
            )
        )
        if allow_no_origin:
            logger.warning(
                "S68: OPENCLAW_LOCALHOST_ALLOW_NO_ORIGIN=true active; "
                "requests without Origin/Sec-Fetch-Site will be allowed in localhost convenience mode."
            )
            _emit_startup_audit(
                action="startup.csrf_no_origin_override",
                outcome="warn",
                details={
                    "env": "OPENCLAW_LOCALHOST_ALLOW_NO_ORIGIN",
                    "value": "true",
                    "profile": runtime_profile_value,
                },
            )

        # 1. Access Control (S45 Update)
        try:
            from .access_control import is_any_token_configured, is_auth_configured

            is_exposed = (
                posture.network_exposed
                if posture is not None
                else SecurityGate._check_network_exposure()
            )
            # S45 Policy: If exposed, ANY token is sufficient to say "we are not wide open".
            # (Though Admin token is preferred for full protection, basic auth presence satisfies "not accidentally open")
            auth_ready = (
                (
                    posture.admin_token_configured
                    or posture.observability_token_configured
                )
                if posture is not None
                else is_any_token_configured()
            )

            if is_exposed and not auth_ready:
                # Check for explicit override
                dangerous_bind_override = (
                    posture.dangerous_bind_override if posture is not None else None
                )
                if dangerous_bind_override is None:
                    from .runtime_config import get_config

                    dangerous_bind_override = (
                        get_config().security_dangerous_bind_override
                    )

                if dangerous_bind_override:
                    warnings.append(
                        "WARNING: Server is exposed (--listen) without Authentication, but override is active.\n"
                        "  This is a DANGEROUS configuration. Remote Code Execution is possible if port is accessible."
                    )
                    _emit_startup_audit(
                        action="startup.dangerous_override",
                        outcome="allow",
                        details={
                            "reason": "exposed_without_auth",
                            "override": True,
                            "profile": runtime_profile_value,
                        },
                    )
                    # Do NOT block startup (S45 Override Contract)
                else:
                    # S45: Exposed + No Auth = FATAL (Always, regardless of profile)
                    fatal_errors.append(
                        "CRITICAL SECURITY RISK: Server is exposed (--listen) without Authentication!\n"
                        "  Action Required: Set OPENCLAW_ADMIN_TOKEN (or OPENCLAW_OBSERVABILITY_TOKEN).\n"
                        "  Startup is BLOCKED to prevent RCE.\n"
                        "  (To bypass: set OPENCLAW_SECURITY_DANGEROUS_BIND_OVERRIDE=1)"
                    )
            elif not auth_ready:
                # Loopback + No Auth
                # Use strict is_auth_configured (Admin) for Hardened profile loopback check?
                # "HARDENED profile requires Authentication even on loopback."
                admin_ready = (
                    posture.admin_token_configured
                    if posture is not None
                    else is_auth_configured()
                )
                if hardened and not admin_ready:
                    warnings.append(
                        "HARDENED profile requires Admin Authentication even on loopback."
                    )
        except ImportError:
            warnings.append("Could not import access_control service")

        # 2. Egress Policy (SSRF)
        if posture is None:
            from .runtime_config import get_config

            config = get_config()
            allow_any_public_llm_host = config.allow_any_public_llm_host
            allow_insecure_base_url = config.allow_insecure_base_url
            webhook_auth_mode = config.webhook_auth_mode
        else:
            allow_any_public_llm_host = posture.allow_any_public_llm_host
            allow_insecure_base_url = posture.allow_insecure_base_url
            webhook_auth_mode = posture.webhook_auth_mode

        if allow_any_public_llm_host:
            warnings.append(
                "OPENCLAW_ALLOW_ANY_PUBLIC_LLM_HOST is enabled (Egress check bypassed)"
            )

        if allow_insecure_base_url:
            warnings.append(
                "OPENCLAW_ALLOW_INSECURE_BASE_URL is enabled (SSRF check bypassed)"
            )

        # 3. Webhook Security (if Webhook module enabled)
        from .modules import ModuleCapability, is_module_enabled

        if is_module_enabled(ModuleCapability.WEBHOOK):
            if webhook_auth_mode == "unset":
                warnings.append(
                    "Webhook module enabled but OPENCLAW_WEBHOOK_AUTH_MODE not set"
                )

        # 3.5 Tool Sandbox Posture (S47)
        try:
            from .tool_runner import evaluate_tool_sandbox_posture, is_tools_enabled

            if is_tools_enabled():
                sandbox_ok, sandbox_issues = evaluate_tool_sandbox_posture()
                if not sandbox_ok:
                    for issue in sandbox_issues:
                        warnings.append(f"Tool Sandbox FAILED: {issue}")
        except ImportError:
            warnings.append("Tool sandbox posture checker failed to import")

        # 4. Redaction
        try:
            from .redaction import redact_text

            if not callable(redact_text):
                warnings.append("Redaction service is not callable")
        except ImportError:
            warnings.append("Redaction service failed to import")

        # 5. Permission Posture (S42)
        try:
            from .permission_posture import evaluate_startup_permissions

            perm_allowed, perm_results = evaluate_startup_permissions()
            if not perm_allowed:
                for res in perm_results:
                    if res.severity == "fail":
                        warnings.append(f"Permission Check FAILED: {res.message}")
        except ImportError:
            warnings.append("Permission posture service failed to import")

        # 6. Control-Plane Split Enforcement (S62)
        try:
            from .control_plane import enforce_control_plane_startup

            cp_result = enforce_control_plane_startup(posture=posture)
            if not cp_result.get("startup_passed", True):
                for err in cp_result.get("errors", []):
                    fatal_errors.append(f"S62 Control-Plane: {err}")
            for w in cp_result.get("warnings", []):
                warnings.append(f"S62 Control-Plane: {w}")
        except ImportError:
            warnings.append("S62 control_plane module failed to import")

        # 7. Connector allowlist fail-closed posture (S71)
        if posture is not None:
            connector_unguarded = list(posture.connector_unguarded_platforms)
            connector_allowlist_vars = list(
                posture.connector_recommended_allowlist_vars
            )
        else:
            connector_posture = evaluate_connector_allowlist_posture(os.environ)
            connector_unguarded = [
                str(item) for item in connector_posture["unguarded_platforms"]
            ]
            connector_allowlist_vars = [
                str(item) for item in connector_posture["recommended_allowlist_vars"]
            ]
        if connector_unguarded:
            deployment_profile = (
                posture.deployment_profile
                if posture is not None
                else os.environ.get("OPENCLAW_DEPLOYMENT_PROFILE", "").strip().lower()
            )
            platforms = ", ".join(connector_unguarded)
            allowlist_vars = ", ".join(connector_allowlist_vars)
            msg = (
                "Connector allowlist coverage missing for active platform(s): "
                f"{platforms}. Configure allowlists ({allowlist_vars}) before enabling ingress."
            )

            # CRITICAL: hardened/public must fail closed for unallowlisted connector ingress.
            if hardened:
                warnings.append(f"S71 (hardened fail-closed): {msg}")
                _emit_startup_audit(
                    action="startup.connector_allowlist_posture",
                    outcome="error",
                    details={
                        "mode": "hardened",
                        "unguarded_platforms": connector_unguarded,
                        "deployment_profile": deployment_profile or "unset",
                    },
                )
            elif deployment_profile == "public":
                fatal_errors.append(f"S71 (public fail-closed): {msg}")
                _emit_startup_audit(
                    action="startup.connector_allowlist_posture",
                    outcome="error",
                    details={
                        "mode": "public",
                        "unguarded_platforms": connector_unguarded,
                    },
                )
            else:
                warnings.append(f"S71 (warn-only): {msg}")
                _emit_startup_audit(
                    action="startup.connector_allowlist_posture",
                    outcome="warn",
                    details={
                        "mode": "warn_only",
                        "unguarded_platforms": connector_unguarded,
                        "deployment_profile": deployment_profile or "unset",
                    },
                )

        # In HARDENED mode, treat all warnings as FATAL
        if hardened and warnings:
            fatal_errors.extend(warnings)
            warnings = []

        passed = len(fatal_errors) == 0
        return passed, warnings, fatal_errors


def enforce_startup_gate(
    posture: EffectiveSecurityPosture | None = None,
) -> None:
    """
    Run the security gate.
    If in HARDENED mode and checks fail -> Raise SystemExit.
    If in MINIMAL mode and checks fail -> Log warnings.
    """
    is_hardened = (
        posture.runtime_profile == "hardened"
        if posture is not None
        else is_hardened_mode()
    )
    mode_str = "HARDENED" if is_hardened else "MINIMAL"

    logger.info(f"Running S41 Security Gate ({mode_str} profile)...")

    passed, warnings, fatal_errors = SecurityGate.verify_mandatory_controls(
        posture=posture
    )

    # Log warnings first (non-blocking unless hardened)
    if warnings:
        warn_msg = f"Security Gate WARNINGS ({len(warnings)} issues):\n" + "\n".join(
            [f"- {i}" for i in warnings]
        )
        if is_hardened:
            # In Hardened mode, warnings become fatal.
            logger.critical(warn_msg)
            fatal_errors.append("HARDENED profile requires 0 warnings.")
        else:
            logger.warning(warn_msg)

    if passed and not fatal_errors:
        logger.info("Security Gate: PASS")
        return

    # Handle fatal errors (S45 Fail-Closed for Critical/Hardened failures)
    error_msg = (
        f"Security Gate FAILED ({len(fatal_errors)} fatal errors):\n"
        + "\n".join([f"- {i}" for i in fatal_errors])
    )

    logger.critical(error_msg)
    logger.critical("FATAL: Security controls failed. Startup aborted.")

    # CRITICAL: S41/S45 fatal gate failures must remain fail-closed in all profiles.
    raise RuntimeError(error_msg)
