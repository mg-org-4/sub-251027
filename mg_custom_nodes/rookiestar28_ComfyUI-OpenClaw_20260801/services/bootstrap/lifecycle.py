"""Typed startup lifecycle outcomes and redacted public diagnostics.

Required startup work still fails closed in callers. This module owns only the
phase/result state machine and optional post-ready warmup observations; it does
not own ComfyUI's application lifecycle.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("ComfyUI-OpenClaw.services.startup_lifecycle")

SCHEMA_VERSION = 1
MAX_DIAGNOSTIC_MS = 86_400_000
MAX_WARMUPS = 16
STARTUP_DIAGNOSTIC_KEYS = (
    "schema_version",
    "phase",
    "state",
    "reason_code",
    "ready",
    "degraded",
    "fatal",
    "attempt",
    "max_attempts",
    "elapsed_ms",
    "phase_elapsed_ms",
    "ready_elapsed_ms",
    "warmups",
)


class StartupPhase(str, Enum):
    PACKAGE_IMPORT = "package_import"
    REQUIRED_INITIALIZATION = "required_initialization"
    HOST_WAIT = "host_wait"
    ROUTE_REGISTRATION = "route_registration"
    COMPLETE = "complete"
    OPTIONAL_WARMUP = "optional_warmup"


class StartupState(str, Enum):
    STARTING = "starting"
    INITIALIZING = "initializing"
    WAITING_FOR_HOST = "waiting_for_host"
    REGISTERING_ROUTES = "registering_routes"
    READY = "ready"
    DEGRADED = "degraded"
    FATAL = "fatal"


class StartupReason(str, Enum):
    BOOTSTRAP_STARTED = "bootstrap_started"
    BOOTSTRAP_IMPORT_FAILED = "bootstrap_import_failed"
    REQUIRED_INITIALIZATION_STARTED = "required_initialization_started"
    REQUIRED_INITIALIZATION_FAILED = "required_initialization_failed"
    HOST_NOT_READY = "host_not_ready"
    ROUTE_REGISTRATION_STARTED = "route_registration_started"
    ROUTE_REGISTRATION_SUCCEEDED = "route_registration_succeeded"
    ROUTE_REGISTRATION_FAILED = "route_registration_failed"
    RETRY_EXHAUSTED = "retry_exhausted"
    WARMUP_STARTED = "warmup_started"
    WARMUP_SUCCEEDED = "warmup_succeeded"
    WARMUP_FAILED = "warmup_failed"
    WARMUP_TIMED_OUT = "warmup_timed_out"


class WarmupState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


# Compatibility constants retained for existing internal imports.
STARTUP_STARTING = StartupState.STARTING.value
STARTUP_READY = StartupState.READY.value
STARTUP_DEGRADED_WARMUP = StartupState.DEGRADED.value
STARTUP_FATAL = StartupState.FATAL.value
WARMUP_PENDING = WarmupState.PENDING.value
WARMUP_RUNNING = WarmupState.RUNNING.value
WARMUP_SUCCEEDED = WarmupState.SUCCEEDED.value
WARMUP_FAILED = WarmupState.FAILED.value
WARMUP_TIMED_OUT = WarmupState.TIMED_OUT.value

WarmupSpec = tuple[str, Callable[[], Any], float]


class StartupTransitionError(RuntimeError):
    """Stable transition failure that never embeds caller/source content."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class WarmupOutcome:
    name: str
    state: WarmupState
    reason_code: StartupReason
    timeout_ms: int
    duration_ms: int

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "reason_code": self.reason_code.value,
            "timeout_ms": self.timeout_ms,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class StartupOutcome:
    schema_version: int
    phase: StartupPhase
    state: StartupState
    reason_code: StartupReason
    ready: bool
    degraded: bool
    fatal: bool
    attempt: int
    max_attempts: int
    elapsed_ms: int
    phase_elapsed_ms: int
    ready_elapsed_ms: int | None
    warmups: tuple[WarmupOutcome, ...]

    def to_diagnostics(self) -> dict[str, Any]:
        """Return a fresh, ordered, JSON-safe public projection."""

        return {
            "schema_version": self.schema_version,
            "phase": self.phase.value,
            "state": self.state.value,
            "reason_code": self.reason_code.value,
            "ready": self.ready,
            "degraded": self.degraded,
            "fatal": self.fatal,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "elapsed_ms": self.elapsed_ms,
            "phase_elapsed_ms": self.phase_elapsed_ms,
            "ready_elapsed_ms": self.ready_elapsed_ms,
            "warmups": [warmup.to_diagnostics() for warmup in self.warmups],
        }


@dataclass
class _WarmupRecord:
    name: str
    state: WarmupState
    reason_code: StartupReason
    timeout_sec: float
    started_at: float | None = None
    completed_at: float | None = None


def _bounded_ms(seconds: float) -> int:
    if not math.isfinite(seconds) or seconds <= 0:
        return 0
    return min(round(seconds * 1000.0), MAX_DIAGNOSTIC_MS)


_WARMUP_NAME_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _safe_warmup_name(value: Any) -> str:
    normalized = _WARMUP_NAME_RE.sub("_", str(value or "warmup")).strip("._-")
    return (normalized or "warmup")[:64]


class StartupLifecycle:
    """Single lock-protected owner of startup phase and warmup outcomes."""

    def __init__(self, *, monotonic_fn: Callable[[], float] = time.monotonic):
        self._clock = monotonic_fn
        self._lock = threading.RLock()
        self._generation = 0
        self._reset_locked()

    def _reset_locked(self) -> None:
        now = self._clock()
        self._started_at = now
        self._phase_started_at = now
        self._ready_at: float | None = None
        self._phase = StartupPhase.PACKAGE_IMPORT
        self._state = StartupState.STARTING
        self._reason_code = StartupReason.BOOTSTRAP_STARTED
        self._attempt = 0
        self._max_attempts = 0
        self._warmups: dict[str, _WarmupRecord] = {}

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._reset_locked()

    def _require_nonterminal(self) -> None:
        if self._state is StartupState.FATAL:
            raise StartupTransitionError("TERMINAL_STATE")

    def _set(
        self,
        *,
        phase: StartupPhase,
        state: StartupState,
        reason_code: StartupReason,
        attempt: int | None = None,
        max_attempts: int | None = None,
    ) -> None:
        now = self._clock()
        if phase is not self._phase:
            self._phase_started_at = now
        self._phase = phase
        self._state = state
        self._reason_code = reason_code
        if attempt is not None:
            self._attempt = attempt
        if max_attempts is not None:
            self._max_attempts = max_attempts
        if state is StartupState.READY and self._ready_at is None:
            self._ready_at = now

    def mark_required_initialization_started(self) -> None:
        with self._lock:
            self._require_nonterminal()
            if self._state is not StartupState.STARTING:
                raise StartupTransitionError("INVALID_TRANSITION")
            self._set(
                phase=StartupPhase.REQUIRED_INITIALIZATION,
                state=StartupState.INITIALIZING,
                reason_code=StartupReason.REQUIRED_INITIALIZATION_STARTED,
            )

    def mark_host_waiting(self, *, attempt: int, max_attempts: int) -> None:
        with self._lock:
            self._require_nonterminal()
            if max_attempts <= 0 or attempt < 0 or attempt > max_attempts:
                raise StartupTransitionError("ATTEMPT_OUT_OF_RANGE")
            if self._state is StartupState.INITIALIZING:
                if attempt != 0:
                    raise StartupTransitionError("ATTEMPT_OUT_OF_RANGE")
            elif self._state is StartupState.WAITING_FOR_HOST:
                if attempt <= self._attempt:
                    raise StartupTransitionError("ATTEMPT_NOT_INCREASING")
                if max_attempts != self._max_attempts:
                    raise StartupTransitionError("ATTEMPT_BOUND_CHANGED")
            else:
                raise StartupTransitionError("INVALID_TRANSITION")
            self._set(
                phase=StartupPhase.HOST_WAIT,
                state=StartupState.WAITING_FOR_HOST,
                reason_code=StartupReason.HOST_NOT_READY,
                attempt=attempt,
                max_attempts=max_attempts,
            )

    def mark_route_registration_started(
        self, *, attempt: int = 0, max_attempts: int = 0
    ) -> None:
        with self._lock:
            self._require_nonterminal()
            if self._state not in {
                StartupState.INITIALIZING,
                StartupState.WAITING_FOR_HOST,
            }:
                raise StartupTransitionError("INVALID_TRANSITION")
            if attempt < 0 or max_attempts < 0 or attempt > max_attempts:
                raise StartupTransitionError("ATTEMPT_OUT_OF_RANGE")
            if self._state is StartupState.INITIALIZING:
                if attempt != 0 or max_attempts != 0:
                    raise StartupTransitionError("ATTEMPT_OUT_OF_RANGE")
            else:
                if max_attempts != self._max_attempts:
                    raise StartupTransitionError("ATTEMPT_BOUND_CHANGED")
                if attempt <= self._attempt:
                    raise StartupTransitionError("ATTEMPT_NOT_INCREASING")
            self._set(
                phase=StartupPhase.ROUTE_REGISTRATION,
                state=StartupState.REGISTERING_ROUTES,
                reason_code=StartupReason.ROUTE_REGISTRATION_STARTED,
                attempt=attempt,
                max_attempts=max_attempts,
            )

    def mark_ready(self) -> None:
        with self._lock:
            self._require_nonterminal()
            if self._state is not StartupState.REGISTERING_ROUTES:
                raise StartupTransitionError("INVALID_TRANSITION")
            self._set(
                phase=StartupPhase.COMPLETE,
                state=StartupState.READY,
                reason_code=StartupReason.ROUTE_REGISTRATION_SUCCEEDED,
            )

    def mark_fatal(
        self,
        *,
        phase: StartupPhase,
        reason_code: StartupReason,
    ) -> None:
        with self._lock:
            self._require_nonterminal()
            allowed = {
                StartupReason.BOOTSTRAP_IMPORT_FAILED,
                StartupReason.REQUIRED_INITIALIZATION_FAILED,
                StartupReason.ROUTE_REGISTRATION_FAILED,
                StartupReason.RETRY_EXHAUSTED,
            }
            if reason_code not in allowed:
                raise StartupTransitionError("INVALID_FATAL_REASON")
            expected_phase = {
                StartupReason.BOOTSTRAP_IMPORT_FAILED: StartupPhase.PACKAGE_IMPORT,
                StartupReason.REQUIRED_INITIALIZATION_FAILED: (
                    StartupPhase.REQUIRED_INITIALIZATION
                ),
                StartupReason.ROUTE_REGISTRATION_FAILED: (
                    StartupPhase.ROUTE_REGISTRATION
                ),
                StartupReason.RETRY_EXHAUSTED: StartupPhase.HOST_WAIT,
            }[reason_code]
            if phase is not expected_phase:
                raise StartupTransitionError("FATAL_PHASE_MISMATCH")
            allowed_states = {
                StartupReason.BOOTSTRAP_IMPORT_FAILED: {
                    StartupState.STARTING,
                },
                StartupReason.REQUIRED_INITIALIZATION_FAILED: {
                    StartupState.INITIALIZING,
                },
                StartupReason.ROUTE_REGISTRATION_FAILED: {
                    StartupState.INITIALIZING,
                    StartupState.WAITING_FOR_HOST,
                    StartupState.REGISTERING_ROUTES,
                },
                StartupReason.RETRY_EXHAUSTED: {
                    StartupState.WAITING_FOR_HOST,
                },
            }[reason_code]
            if self._state not in allowed_states:
                raise StartupTransitionError("INVALID_FATAL_TRANSITION")
            self._set(
                phase=phase,
                state=StartupState.FATAL,
                reason_code=reason_code,
            )

    def mark_retry_exhausted(self) -> None:
        with self._lock:
            self._require_nonterminal()
            if self._state is not StartupState.WAITING_FOR_HOST:
                raise StartupTransitionError("INVALID_TRANSITION")
            self._set(
                phase=StartupPhase.HOST_WAIT,
                state=StartupState.FATAL,
                reason_code=StartupReason.RETRY_EXHAUSTED,
            )

    def begin_warmup(self, name: str, timeout_sec: float) -> tuple[bool, int, str]:
        safe_name = _safe_warmup_name(name)
        timeout_sec = max(0.01, min(float(timeout_sec or 5.0), 60.0))
        with self._lock:
            self._require_nonterminal()
            if self._state not in {StartupState.READY, StartupState.DEGRADED}:
                raise StartupTransitionError("WARMUP_BEFORE_READY")
            existing = self._warmups.get(safe_name)
            if existing is not None:
                return False, self._generation, safe_name
            if len(self._warmups) >= MAX_WARMUPS:
                raise StartupTransitionError("WARMUP_LIMIT_EXCEEDED")
            self._warmups[safe_name] = _WarmupRecord(
                name=safe_name,
                state=WarmupState.PENDING,
                reason_code=StartupReason.WARMUP_STARTED,
                timeout_sec=timeout_sec,
            )
            return True, self._generation, safe_name

    def mark_warmup_running(self, name: str, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            record = self._warmups.get(name)
            if record is None:
                return
            record.state = WarmupState.RUNNING
            record.reason_code = StartupReason.WARMUP_STARTED
            record.started_at = self._clock()
            if self._state is StartupState.READY:
                self._set(
                    phase=StartupPhase.OPTIONAL_WARMUP,
                    state=StartupState.READY,
                    reason_code=StartupReason.WARMUP_STARTED,
                )

    def finish_warmup(
        self,
        name: str,
        generation: int,
        *,
        state: WarmupState,
    ) -> None:
        reason_by_state = {
            WarmupState.SUCCEEDED: StartupReason.WARMUP_SUCCEEDED,
            WarmupState.FAILED: StartupReason.WARMUP_FAILED,
            WarmupState.TIMED_OUT: StartupReason.WARMUP_TIMED_OUT,
        }
        reason = reason_by_state.get(state)
        if reason is None:
            raise StartupTransitionError("INVALID_WARMUP_RESULT")
        with self._lock:
            if generation != self._generation:
                return
            record = self._warmups.get(name)
            if record is None:
                return
            if record.state in {
                WarmupState.SUCCEEDED,
                WarmupState.FAILED,
                WarmupState.TIMED_OUT,
            }:
                return
            record.state = state
            record.reason_code = reason
            record.completed_at = self._clock()
            if state in {WarmupState.FAILED, WarmupState.TIMED_OUT}:
                self._set(
                    phase=StartupPhase.OPTIONAL_WARMUP,
                    state=StartupState.DEGRADED,
                    reason_code=reason,
                )
            elif self._state is StartupState.READY:
                self._set(
                    phase=StartupPhase.OPTIONAL_WARMUP,
                    state=StartupState.READY,
                    reason_code=reason,
                )

    def snapshot(self) -> StartupOutcome:
        with self._lock:
            now = self._clock()
            warmups = []
            for name in sorted(self._warmups):
                record = self._warmups[name]
                started_at = record.started_at
                completed_at = record.completed_at
                if started_at is None:
                    duration = 0.0
                else:
                    duration = (
                        completed_at if completed_at is not None else now
                    ) - started_at
                warmups.append(
                    WarmupOutcome(
                        name=record.name,
                        state=record.state,
                        reason_code=record.reason_code,
                        timeout_ms=_bounded_ms(record.timeout_sec),
                        duration_ms=_bounded_ms(duration),
                    )
                )
            ready = self._state in {StartupState.READY, StartupState.DEGRADED}
            return StartupOutcome(
                schema_version=SCHEMA_VERSION,
                phase=self._phase,
                state=self._state,
                reason_code=self._reason_code,
                ready=ready,
                degraded=self._state is StartupState.DEGRADED,
                fatal=self._state is StartupState.FATAL,
                attempt=self._attempt,
                max_attempts=self._max_attempts,
                elapsed_ms=_bounded_ms(now - self._started_at),
                phase_elapsed_ms=_bounded_ms(now - self._phase_started_at),
                ready_elapsed_ms=(
                    _bounded_ms(now - self._ready_at)
                    if self._ready_at is not None
                    else None
                ),
                warmups=tuple(warmups),
            )


_LIFECYCLE = StartupLifecycle()


def get_startup_outcome() -> StartupOutcome:
    return _LIFECYCLE.snapshot()


def get_startup_diagnostics() -> dict[str, Any]:
    return get_startup_outcome().to_diagnostics()


def mark_required_initialization_started() -> None:
    _LIFECYCLE.mark_required_initialization_started()


def mark_host_waiting(*, attempt: int, max_attempts: int) -> None:
    _LIFECYCLE.mark_host_waiting(attempt=attempt, max_attempts=max_attempts)


def mark_route_registration_started(*, attempt: int = 0, max_attempts: int = 0) -> None:
    _LIFECYCLE.mark_route_registration_started(
        attempt=attempt,
        max_attempts=max_attempts,
    )


def mark_startup_ready(phase: str = "routes") -> None:
    """Compatibility facade that reaches the required ready transition."""

    _ = phase
    outcome = _LIFECYCLE.snapshot()
    if outcome.state is StartupState.STARTING:
        _LIFECYCLE.mark_required_initialization_started()
        _LIFECYCLE.mark_route_registration_started()
        _LIFECYCLE.mark_ready()
        return
    if outcome.state is StartupState.REGISTERING_ROUTES:
        _LIFECYCLE.mark_ready()
        return
    if outcome.state in {StartupState.READY, StartupState.DEGRADED}:
        return
    raise StartupTransitionError("INVALID_TRANSITION")


def mark_startup_fatal(
    phase: str,
    exc: BaseException | None = None,
    *,
    reason_code: StartupReason | str | None = None,
) -> None:
    """Record a stable fatal classification without retaining ``exc``."""

    _ = exc
    phase_map = {
        "package_import": StartupPhase.PACKAGE_IMPORT,
        "required_startup": StartupPhase.REQUIRED_INITIALIZATION,
        "required_initialization": StartupPhase.REQUIRED_INITIALIZATION,
        "route_registration": StartupPhase.ROUTE_REGISTRATION,
        "route_registration_retry": StartupPhase.HOST_WAIT,
        "host_wait": StartupPhase.HOST_WAIT,
    }
    resolved_phase = phase_map.get(str(phase), StartupPhase.REQUIRED_INITIALIZATION)
    if reason_code is None:
        default_reasons = {
            StartupPhase.PACKAGE_IMPORT: StartupReason.BOOTSTRAP_IMPORT_FAILED,
            StartupPhase.REQUIRED_INITIALIZATION: (
                StartupReason.REQUIRED_INITIALIZATION_FAILED
            ),
            StartupPhase.ROUTE_REGISTRATION: (StartupReason.ROUTE_REGISTRATION_FAILED),
            StartupPhase.HOST_WAIT: StartupReason.RETRY_EXHAUSTED,
        }
        resolved_reason = default_reasons[resolved_phase]
    else:
        resolved_reason = (
            reason_code
            if isinstance(reason_code, StartupReason)
            else StartupReason(str(reason_code))
        )
    _LIFECYCLE.mark_fatal(
        phase=resolved_phase,
        reason_code=resolved_reason,
    )


def mark_bootstrap_import_failed(exc: BaseException) -> None:
    mark_startup_fatal(
        "package_import",
        exc,
        reason_code=StartupReason.BOOTSTRAP_IMPORT_FAILED,
    )


def mark_retry_exhausted() -> None:
    _LIFECYCLE.mark_retry_exhausted()


def start_optional_warmups(specs: Iterable[WarmupSpec]) -> None:
    for name, fn, timeout_sec in tuple(specs or ()):
        _start_optional_warmup(str(name), fn, float(timeout_sec))


def reset_startup_lifecycle_for_tests() -> None:
    _LIFECYCLE.reset()


def _start_optional_warmup(
    name: str, fn: Callable[[], Any], timeout_sec: float
) -> None:
    should_start, generation, safe_name = _LIFECYCLE.begin_warmup(name, timeout_sec)
    if not should_start:
        return
    monitor = threading.Thread(
        target=_warmup_monitor,
        args=(safe_name, generation, fn, max(0.01, min(timeout_sec, 60.0))),
        name=f"openclaw-warmup-monitor-{safe_name}",
        daemon=True,
    )
    try:
        monitor.start()
    except Exception:
        _LIFECYCLE.finish_warmup(
            safe_name,
            generation,
            state=WarmupState.FAILED,
        )
        raise


def _warmup_monitor(
    name: str,
    generation: int,
    fn: Callable[[], Any],
    timeout_sec: float,
) -> None:
    done = threading.Event()
    result: dict[str, bool] = {}

    def _worker() -> None:
        try:
            fn()
            result["ok"] = True
        except Exception:
            # SECURITY: never retain or log arbitrary exception content.
            result["ok"] = False
        finally:
            done.set()

    _LIFECYCLE.mark_warmup_running(name, generation)
    worker = threading.Thread(
        target=_worker,
        name=f"openclaw-warmup-{name}",
        daemon=True,
    )
    try:
        worker.start()
    except Exception as exc:
        _LIFECYCLE.finish_warmup(
            name,
            generation,
            state=WarmupState.FAILED,
        )
        logger.warning(
            "Optional startup warmup worker could not start "
            "(component=%s, error_type=%s)",
            name,
            type(exc).__name__,
        )
        return

    if not done.wait(timeout=timeout_sec):
        _LIFECYCLE.finish_warmup(
            name,
            generation,
            state=WarmupState.TIMED_OUT,
        )
        logger.warning(
            "Optional startup warmup timed out (component=%s, reason_code=%s)",
            name,
            StartupReason.WARMUP_TIMED_OUT.value,
        )
        return

    if result.get("ok"):
        _LIFECYCLE.finish_warmup(
            name,
            generation,
            state=WarmupState.SUCCEEDED,
        )
        logger.info("Optional startup warmup completed (component=%s)", name)
        return

    _LIFECYCLE.finish_warmup(
        name,
        generation,
        state=WarmupState.FAILED,
    )
    logger.warning(
        "Optional startup warmup failed (component=%s, reason_code=%s)",
        name,
        StartupReason.WARMUP_FAILED.value,
    )
