"""Compatibility alias for the bootstrap lifecycle implementation module."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from .bootstrap import lifecycle as _implementation

if TYPE_CHECKING:
    # Static-only exports keep legacy imports typed without a second runtime owner.
    from .bootstrap.lifecycle import MAX_DIAGNOSTIC_MS as MAX_DIAGNOSTIC_MS
    from .bootstrap.lifecycle import MAX_WARMUPS as MAX_WARMUPS
    from .bootstrap.lifecycle import SCHEMA_VERSION as SCHEMA_VERSION
    from .bootstrap.lifecycle import STARTUP_DEGRADED_WARMUP as STARTUP_DEGRADED_WARMUP
    from .bootstrap.lifecycle import STARTUP_DIAGNOSTIC_KEYS as STARTUP_DIAGNOSTIC_KEYS
    from .bootstrap.lifecycle import STARTUP_FATAL as STARTUP_FATAL
    from .bootstrap.lifecycle import STARTUP_READY as STARTUP_READY
    from .bootstrap.lifecycle import STARTUP_STARTING as STARTUP_STARTING
    from .bootstrap.lifecycle import WARMUP_FAILED as WARMUP_FAILED
    from .bootstrap.lifecycle import WARMUP_PENDING as WARMUP_PENDING
    from .bootstrap.lifecycle import WARMUP_RUNNING as WARMUP_RUNNING
    from .bootstrap.lifecycle import WARMUP_SUCCEEDED as WARMUP_SUCCEEDED
    from .bootstrap.lifecycle import WARMUP_TIMED_OUT as WARMUP_TIMED_OUT
    from .bootstrap.lifecycle import StartupLifecycle as StartupLifecycle
    from .bootstrap.lifecycle import StartupOutcome as StartupOutcome
    from .bootstrap.lifecycle import StartupPhase as StartupPhase
    from .bootstrap.lifecycle import StartupReason as StartupReason
    from .bootstrap.lifecycle import StartupState as StartupState
    from .bootstrap.lifecycle import StartupTransitionError as StartupTransitionError
    from .bootstrap.lifecycle import WarmupOutcome as WarmupOutcome
    from .bootstrap.lifecycle import WarmupSpec as WarmupSpec
    from .bootstrap.lifecycle import WarmupState as WarmupState
    from .bootstrap.lifecycle import get_startup_diagnostics as get_startup_diagnostics
    from .bootstrap.lifecycle import get_startup_outcome as get_startup_outcome
    from .bootstrap.lifecycle import (
        mark_bootstrap_import_failed as mark_bootstrap_import_failed,
    )
    from .bootstrap.lifecycle import mark_host_waiting as mark_host_waiting
    from .bootstrap.lifecycle import (
        mark_required_initialization_started as mark_required_initialization_started,
    )
    from .bootstrap.lifecycle import mark_retry_exhausted as mark_retry_exhausted
    from .bootstrap.lifecycle import (
        mark_route_registration_started as mark_route_registration_started,
    )
    from .bootstrap.lifecycle import mark_startup_fatal as mark_startup_fatal
    from .bootstrap.lifecycle import mark_startup_ready as mark_startup_ready
    from .bootstrap.lifecycle import (
        reset_startup_lifecycle_for_tests as reset_startup_lifecycle_for_tests,
    )
    from .bootstrap.lifecycle import start_optional_warmups as start_optional_warmups

# IMPORTANT: alias the module object; copied re-exports split singleton and patch state.
sys.modules[__name__] = _implementation
