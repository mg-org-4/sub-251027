"""
S62 Control-Plane Split Enforcement.

Defines the control-plane mode contract and enforces trust boundaries
for public deployments.

Modes:
- EMBEDDED: All control surfaces run in-process (default for local/lan).
- SPLIT:    High-risk control surfaces are delegated to an external
            control plane. UX-plane only remains in-process.

Enforcement rule:
- profile=public -> control_plane_mode defaults to SPLIT.
- profile=public + mode=SPLIT -> block in-process high-risk surfaces.
- profile=public + mode=EMBEDDED -> requires explicit override + warning.
"""

from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Tuple

if TYPE_CHECKING:
    from .effective_security_posture import EffectiveSecurityPosture

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode contract
# ---------------------------------------------------------------------------


class ControlPlaneMode(enum.Enum):
    """Control-plane execution mode."""

    EMBEDDED = "embedded"
    SPLIT = "split"


ENV_CONTROL_PLANE_MODE = "OPENCLAW_CONTROL_PLANE_MODE"
ENV_CONTROL_PLANE_URL = "OPENCLAW_CONTROL_PLANE_URL"
ENV_CONTROL_PLANE_TOKEN = "OPENCLAW_CONTROL_PLANE_TOKEN"
ENV_SPLIT_COMPAT_OVERRIDE = "OPENCLAW_SPLIT_COMPAT_OVERRIDE"


# ---------------------------------------------------------------------------
# High-risk surface registry
# ---------------------------------------------------------------------------

# Each entry: (surface_id, human description)
HIGH_RISK_SURFACES: FrozenSet[Tuple[str, str]] = frozenset(
    {
        ("webhook_execute", "Webhook execute ingress"),
        ("callback_egress", "Callback egress dispatch"),
        ("secrets_write", "Secrets write/update endpoints"),
        ("tool_execution", "Tool execution paths"),
        ("registry_sync", "Registry sync activation paths"),
        ("transforms_exec", "Transforms execution paths"),
    }
)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def _effective_posture(
    posture: EffectiveSecurityPosture | None = None,
) -> EffectiveSecurityPosture | None:
    if posture is not None:
        return posture
    try:
        from .effective_security_posture import get_effective_security_posture

        return get_effective_security_posture(required=False)
    except ImportError:
        return None


def resolve_control_plane_mode(
    deployment_profile: str = "",
    *,
    posture: EffectiveSecurityPosture | None = None,
) -> ControlPlaneMode:
    """
    Determine the active control-plane mode.

    Rules:
    1. Explicit env override takes precedence.
    2. profile=public defaults to SPLIT.
    3. Everything else defaults to EMBEDDED.
    """
    effective = _effective_posture(posture)
    if effective is not None:
        return ControlPlaneMode(effective.control_plane_mode)

    explicit = os.environ.get(ENV_CONTROL_PLANE_MODE, "").lower().strip()
    if explicit == "split":
        return ControlPlaneMode.SPLIT
    if explicit == "embedded":
        return ControlPlaneMode.EMBEDDED

    # Default: public -> split, else embedded
    if deployment_profile == "public":
        return ControlPlaneMode.SPLIT

    return ControlPlaneMode.EMBEDDED


def is_split_mode(*, posture: EffectiveSecurityPosture | None = None) -> bool:
    """Convenience check for split mode."""

    from .deployment_profile import evaluate_deployment_profile

    effective = _effective_posture(posture)
    profile = (
        effective.deployment_profile
        if effective is not None
        else os.environ.get("OPENCLAW_DEPLOYMENT_PROFILE", "local")
    )
    return (
        resolve_control_plane_mode(profile, posture=effective) == ControlPlaneMode.SPLIT
    )


# ---------------------------------------------------------------------------
# Surface blocking
# ---------------------------------------------------------------------------


def get_blocked_surfaces(
    deployment_profile: str,
    mode: Optional[ControlPlaneMode] = None,
    *,
    posture: EffectiveSecurityPosture | None = None,
) -> List[Tuple[str, str]]:
    """
    Return list of (surface_id, reason) blocked in current configuration.

    In public + split: all HIGH_RISK_SURFACES are blocked.
    In embedded or non-public: nothing blocked.
    """
    effective = _effective_posture(posture)
    if effective is not None:
        deployment_profile = effective.deployment_profile
        mode = ControlPlaneMode(effective.control_plane_mode)
    elif mode is None:
        mode = resolve_control_plane_mode(deployment_profile)

    if deployment_profile == "public" and mode == ControlPlaneMode.SPLIT:
        return [(sid, desc) for sid, desc in sorted(HIGH_RISK_SURFACES)]

    return []


def is_surface_blocked(
    surface_id: str,
    *,
    posture: EffectiveSecurityPosture | None = None,
) -> bool:
    """Check if a specific surface is blocked in current config."""
    effective = _effective_posture(posture)
    profile = (
        effective.deployment_profile
        if effective is not None
        else os.environ.get("OPENCLAW_DEPLOYMENT_PROFILE", "local")
    )
    blocked = get_blocked_surfaces(profile, posture=effective)
    return any(sid == surface_id for sid, _ in blocked)


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------


@dataclass
class SplitPrereqReport:
    """Result of split-mode prerequisite validation."""

    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_split_prerequisites(
    *,
    posture: EffectiveSecurityPosture | None = None,
) -> SplitPrereqReport:
    """
    Validate that all prerequisites for split mode are met.

    Required:
    - OPENCLAW_CONTROL_PLANE_URL must be set and non-empty.
    - OPENCLAW_CONTROL_PLANE_TOKEN must be set and non-empty.

    Returns a report. If report.passed is False, startup should fail closed.
    """
    report = SplitPrereqReport()

    effective = _effective_posture(posture)
    url_configured = (
        effective.control_plane_url_configured
        if effective is not None
        else bool(os.environ.get(ENV_CONTROL_PLANE_URL, "").strip())
    )
    token_configured = (
        effective.control_plane_token_configured
        if effective is not None
        else bool(os.environ.get(ENV_CONTROL_PLANE_TOKEN, "").strip())
    )

    if not url_configured:
        report.passed = False
        report.errors.append(
            f"S62: Split mode requires {ENV_CONTROL_PLANE_URL} but it is not set."
        )

    if not token_configured:
        report.passed = False
        report.errors.append(
            f"S62: Split mode requires {ENV_CONTROL_PLANE_TOKEN} but it is not set."
        )

    # Check for compat override (dev-only, auditable)
    compat_override = (
        effective.control_plane_compat_override
        if effective is not None
        else os.environ.get(ENV_SPLIT_COMPAT_OVERRIDE, "").lower().strip()
        in ("1", "true", "yes")
    )
    if compat_override:
        report.warnings.append(
            "S62: OPENCLAW_SPLIT_COMPAT_OVERRIDE is active. "
            "This bypasses split enforcement and is for dev-only use."
        )

    return report


def enforce_control_plane_startup(
    *,
    posture: EffectiveSecurityPosture | None = None,
) -> Dict:
    """
    Run control-plane startup validation.

    Called during application startup. Behavior:
    - public + split + missing prereqs -> fail closed (raise SystemExit)
    - public + embedded + no override  -> fail closed
    - public + embedded + override     -> warn (dev-only)
    - local/lan + any                  -> pass with info

    Returns diagnostic dict for startup report.
    """
    effective = _effective_posture(posture)
    profile = (
        effective.deployment_profile
        if effective is not None
        else os.environ.get("OPENCLAW_DEPLOYMENT_PROFILE", "local")
    )
    mode = resolve_control_plane_mode(profile, posture=effective)
    compat_override = (
        effective.control_plane_compat_override
        if effective is not None
        else os.environ.get(ENV_SPLIT_COMPAT_OVERRIDE, "").lower().strip()
        in ("1", "true", "yes")
    )

    result: Dict = {
        "deployment_profile": profile,
        "control_plane_mode": mode.value,
        "blocked_surfaces": [
            {"id": sid, "reason": desc}
            for sid, desc in get_blocked_surfaces(profile, mode, posture=effective)
        ],
        "startup_passed": True,
        "errors": [],
        "warnings": [],
    }

    if profile == "public" and mode == ControlPlaneMode.SPLIT:
        prereq = validate_split_prerequisites(posture=effective)
        if not prereq.passed:
            result["startup_passed"] = False
            result["errors"] = prereq.errors
            result["warnings"] = prereq.warnings
            logger.error(
                "S62: Split mode startup validation FAILED. " f"Errors: {prereq.errors}"
            )
        else:
            result["warnings"] = prereq.warnings
            logger.info(
                f"S62: Split mode active. "
                f"{len(result['blocked_surfaces'])} high-risk surfaces blocked."
            )

    elif profile == "public" and mode == ControlPlaneMode.EMBEDDED:
        if not compat_override:
            result["startup_passed"] = False
            result["errors"].append(
                "S62: public profile requires split mode. "
                "Set OPENCLAW_CONTROL_PLANE_MODE=split and configure "
                "external control plane, or set "
                "OPENCLAW_SPLIT_COMPAT_OVERRIDE=1 for dev-only bypass."
            )
            logger.error("S62: public + embedded without override. Startup blocked.")
        else:
            result["warnings"].append(
                "S62: public + embedded with compat override. "
                "HIGH-RISK: all control surfaces are in-process. "
                "This configuration is for development only."
            )
            logger.warning(
                "S62: Running public+embedded with compat override (DEV ONLY)."
            )
            # R102 Hook
            try:
                from .security_telemetry import get_security_telemetry

                get_security_telemetry().record_dangerous_override(
                    "SPLIT_COMPAT_OVERRIDE", "system_env"
                )
            except ImportError:
                pass

    else:
        # local/lan: always pass
        logger.info(
            f"S62: Control-plane mode={mode.value} for profile={profile}. "
            "No enforcement required."
        )

    return result
