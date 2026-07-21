"""
Startup lifecycle diagnostics and optional warmup boundaries.

Required startup work still fails closed in callers. This module only tracks
readiness and runs optional warmups without delaying route availability.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Iterable, Optional

logger = logging.getLogger("ComfyUI-OpenClaw.services.startup_lifecycle")

STARTUP_STARTING = "starting"
STARTUP_READY = "ready"
STARTUP_DEGRADED_WARMUP = "degraded-warmup"
STARTUP_FATAL = "fatal-startup"

WARMUP_PENDING = "pending"
WARMUP_RUNNING = "running"
WARMUP_SUCCEEDED = "succeeded"
WARMUP_FAILED = "failed"
WARMUP_TIMED_OUT = "timed_out"

WarmupSpec = tuple[str, Callable[[], Any], float]

_LOCK = threading.RLock()
_STARTED_AT = time.time()
_READY = False
_READY_PHASE: Optional[str] = None
_READY_AT: Optional[float] = None
_FATAL: Optional[Dict[str, Any]] = None
_WARMUPS: Dict[str, Dict[str, Any]] = {}


def mark_startup_ready(phase: str = "routes") -> None:
    """Mark required startup work as ready."""
    global _READY, _READY_AT, _READY_PHASE
    with _LOCK:
        if _FATAL is not None:
            return
        _READY = True
        _READY_PHASE = str(phase or "routes")
        _READY_AT = time.time()


def mark_startup_fatal(phase: str, exc: BaseException) -> None:
    """Record a fatal required-startup failure."""
    global _FATAL, _READY
    with _LOCK:
        _READY = False
        _FATAL = {
            "phase": str(phase or "startup"),
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "ts": time.time(),
        }


def start_optional_warmups(specs: Iterable[WarmupSpec]) -> None:
    """Start optional warmups in background monitor threads."""
    for name, fn, timeout_sec in list(specs or []):
        _start_optional_warmup(str(name), fn, float(timeout_sec))


def get_startup_diagnostics() -> Dict[str, Any]:
    """Return a bounded diagnostic snapshot for health/operator views."""
    with _LOCK:
        warmups = {name: dict(record) for name, record in _WARMUPS.items()}
        fatal = dict(_FATAL) if _FATAL else None
        ready = bool(_READY and fatal is None)
        degraded = any(
            record.get("state") in {WARMUP_FAILED, WARMUP_TIMED_OUT}
            for record in warmups.values()
        )
        if fatal:
            state = STARTUP_FATAL
        elif ready and degraded:
            state = STARTUP_DEGRADED_WARMUP
        elif ready:
            state = STARTUP_READY
        else:
            state = STARTUP_STARTING
        return {
            "state": state,
            "ready": ready,
            "ready_phase": _READY_PHASE,
            "started_at": _STARTED_AT,
            "ready_at": _READY_AT,
            "fatal": fatal,
            "warmups": warmups,
        }


def reset_startup_lifecycle_for_tests() -> None:
    """Reset in-memory lifecycle state for tests."""
    global _READY, _READY_AT, _READY_PHASE, _FATAL, _STARTED_AT
    with _LOCK:
        _STARTED_AT = time.time()
        _READY = False
        _READY_PHASE = None
        _READY_AT = None
        _FATAL = None
        _WARMUPS.clear()


def _start_optional_warmup(
    name: str, fn: Callable[[], Any], timeout_sec: float
) -> None:
    timeout_sec = max(0.01, min(float(timeout_sec or 5.0), 60.0))
    with _LOCK:
        existing = _WARMUPS.get(name)
        if existing and existing.get("state") in {WARMUP_RUNNING, WARMUP_SUCCEEDED}:
            return
        _WARMUPS[name] = {
            "state": WARMUP_PENDING,
            "timeout_sec": timeout_sec,
            "started_at": None,
            "completed_at": None,
            "duration_sec": None,
            "error_type": None,
            "error": None,
        }

    monitor = threading.Thread(
        target=_warmup_monitor,
        args=(name, fn, timeout_sec),
        name=f"openclaw-warmup-monitor-{name}",
        daemon=True,
    )
    monitor.start()


def _warmup_monitor(name: str, fn: Callable[[], Any], timeout_sec: float) -> None:
    started_at = time.time()
    done = threading.Event()
    result: Dict[str, Any] = {}

    def _worker() -> None:
        try:
            result["value"] = fn()
            result["ok"] = True
        except Exception as exc:  # pragma: no cover - defensive outer guard
            result["ok"] = False
            result["exc"] = exc
        finally:
            done.set()

    with _LOCK:
        if name in _WARMUPS:
            _WARMUPS[name]["state"] = WARMUP_RUNNING
            _WARMUPS[name]["started_at"] = started_at

    worker = threading.Thread(
        target=_worker,
        name=f"openclaw-warmup-{name}",
        daemon=True,
    )
    worker.start()

    if not done.wait(timeout=timeout_sec):
        _finish_warmup(
            name,
            WARMUP_TIMED_OUT,
            started_at,
            error_type="TimeoutError",
            error=f"optional warmup exceeded {timeout_sec:.2f}s",
        )
        logger.warning(
            "R188: optional startup warmup timed out: %s (%.2fs)",
            name,
            timeout_sec,
        )
        return

    exc = result.get("exc")
    if result.get("ok"):
        _finish_warmup(name, WARMUP_SUCCEEDED, started_at)
        logger.info("R188: optional startup warmup completed: %s", name)
        return

    _finish_warmup(
        name,
        WARMUP_FAILED,
        started_at,
        error_type=type(exc).__name__ if exc else "Exception",
        error=str(exc)[:500] if exc else "unknown warmup failure",
    )
    logger.warning("R188: optional startup warmup failed: %s: %s", name, exc)


def _finish_warmup(
    name: str,
    state: str,
    started_at: float,
    *,
    error_type: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    completed_at = time.time()
    with _LOCK:
        record = _WARMUPS.setdefault(name, {})
        record.update(
            {
                "state": state,
                "completed_at": completed_at,
                "duration_sec": max(0.0, completed_at - started_at),
                "error_type": error_type,
                "error": error,
            }
        )
