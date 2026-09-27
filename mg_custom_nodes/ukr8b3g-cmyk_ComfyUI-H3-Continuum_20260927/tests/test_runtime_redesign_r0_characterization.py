from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import ComfyUI_H3_Continuum_Join.run_storage as storage_runtime
from ComfyUI_H3_Continuum_Join.run_storage import RunStorageController, RunStorageError
from ComfyUI_H3_Continuum_Join.v2 import sequence
from ComfyUI_H3_Continuum_Join.v2.session import entry_to_state
from ComfyUI_H3_Continuum_Join.v3.review_control import (
    GENERATION_MODE_REVIEW,
    REVIEW_ACTION_CONTINUE,
    REVISION_STATUS_REVIEW_READY,
)

from . import test_v38_review_execution_cap as cap
from . import test_v38_review_run_storage as storage_cases


def _run_with_sources(
    monkeypatch,
    runtime,
    *,
    chunks: int,
    session=None,
    initial_state=None,
    reroll_from_chunk: int = 0,
    continuity: str | None = None,
    limit: int | None = None,
):
    """Exercise the unchanged public run_sequence contract with extra sources."""

    original = cap.sequence.run_sequence

    def forwarded(**kwargs):
        kwargs["initial_state"] = initial_state
        kwargs["reroll_from_chunk"] = reroll_from_chunk
        if continuity is not None:
            kwargs["continuity"] = continuity
        return original(**kwargs)

    with monkeypatch.context() as local:
        local.setattr(cap.sequence, "run_sequence", forwarded)
        return cap._run_sequence(
            runtime,
            chunks=chunks,
            session=session,
            limit=limit,
        )


def _state_from_session(session: dict, *, clip_index: int = 9) -> dict:
    state = dict(entry_to_state(session["chunks"][-1]))
    state["clip_index"] = clip_index
    return state


def test_r0_storage_off_plain_full_run_uses_no_review_policy_or_filesystem(
    monkeypatch, tmp_path
):
    runtime = cap._install_fake_runtime(monkeypatch)

    def unexpected(*args, **kwargs):
        pytest.fail("Storage OFF Full Run must not resolve Run Storage Review policy")

    monkeypatch.setattr(storage_runtime, "resolve_review_execution", unexpected)
    entries, last_state, session, _ = cap._run_sequence(runtime, chunks=3)

    assert len(entries) == len(session["chunks"]) == len(runtime.samples) == 3
    assert last_state["clip_index"] == 3
    assert list(tmp_path.rglob("*")) == []


def test_r0_storage_off_compatible_session_restores_prefix(monkeypatch):
    runtime = cap._install_fake_runtime(monkeypatch)
    _, _, saved, _ = cap._run_sequence(runtime, chunks=1)
    runtime.samples.clear()

    entries, last_state, session, _ = cap._run_sequence(
        runtime,
        chunks=2,
        session=saved,
    )

    assert len(entries) == len(session["chunks"]) == 2
    assert entries[0]["reused"] is True
    assert len(runtime.samples) == 1
    assert last_state["clip_index"] == 2


def test_r0_storage_off_session_regenerate_from_preserves_only_prior_prefix(monkeypatch):
    runtime = cap._install_fake_runtime(monkeypatch)
    _, _, saved, _ = cap._run_sequence(runtime, chunks=3)
    runtime.samples.clear()

    entries, last_state, session, _ = _run_with_sources(
        monkeypatch,
        runtime,
        chunks=3,
        session=saved,
        reroll_from_chunk=2,
    )

    assert len(entries) == len(session["chunks"]) == 3
    assert [entry["reused"] for entry in entries] == [True, False, False]
    assert len(runtime.samples) == 2
    assert last_state["clip_index"] == 3


def test_r0_storage_off_incompatible_session_is_advisory_and_starts_fresh(monkeypatch):
    runtime = cap._install_fake_runtime(monkeypatch)
    _, _, saved, _ = cap._run_sequence(runtime, chunks=1)
    incompatible = dict(saved)
    incompatible["width"] = 128
    runtime.samples.clear()

    entries, last_state, session, report = cap._run_sequence(
        runtime,
        chunks=1,
        session=incompatible,
    )

    assert len(entries) == len(session["chunks"]) == len(runtime.samples) == 1
    assert entries[0]["reused"] is False
    assert last_state["clip_index"] == 1
    assert "saved session was ignored; generated a fresh run" in report


def test_r0_storage_off_invalid_session_suppresses_valid_state_without_fallback(
    monkeypatch,
):
    runtime = cap._install_fake_runtime(monkeypatch)
    _, _, source, _ = cap._run_sequence(runtime, chunks=1)
    state = _state_from_session(source)
    runtime.samples.clear()

    entries, last_state, session, report = _run_with_sources(
        monkeypatch,
        runtime,
        chunks=1,
        session={"magic": "not-a-session"},
        initial_state=state,
    )

    # Current V3.8 behavior is advisory, not an exception: the invalid Session
    # is rejected, the simultaneously supplied State remains suppressed, and a
    # fresh Chunk 1 is generated. R0 records this fact without changing it.
    assert len(entries) == len(session["chunks"]) == len(runtime.samples) == 1
    assert last_state["clip_index"] == 1
    assert entries[0]["plan"]["continuation"] is False
    assert "saved session was ignored; generated a fresh run" in report


def test_r0_storage_off_compatible_session_suppresses_initial_state(monkeypatch, caplog):
    runtime = cap._install_fake_runtime(monkeypatch)
    _, _, saved, _ = cap._run_sequence(runtime, chunks=1)
    state = _state_from_session(saved)
    runtime.samples.clear()

    entries, last_state, _, _ = _run_with_sources(
        monkeypatch,
        runtime,
        chunks=2,
        session=saved,
        initial_state=state,
    )

    assert len(entries) == 2
    assert entries[0]["reused"] is True
    assert len(runtime.samples) == 1
    assert last_state["clip_index"] == 2
    assert "using the session and ignoring initial_state" in caplog.text


def test_r0_storage_off_initial_state_continues_without_session(monkeypatch):
    runtime = cap._install_fake_runtime(monkeypatch)
    _, _, source, _ = cap._run_sequence(runtime, chunks=1)
    state = _state_from_session(source)
    runtime.samples.clear()

    entries, last_state, session, _ = _run_with_sources(
        monkeypatch,
        runtime,
        chunks=1,
        initial_state=state,
    )

    assert len(entries) == len(session["chunks"]) == len(runtime.samples) == 1
    assert entries[0]["plan"]["continuation"] is True
    assert entries[0]["plan"]["clip_index"] == 10
    assert last_state["clip_index"] == 10


def test_r0_storage_off_initial_state_rejects_invalid_regenerate_from(monkeypatch):
    runtime = cap._install_fake_runtime(monkeypatch)
    _, _, source, _ = cap._run_sequence(runtime, chunks=1)
    state = _state_from_session(source)

    with pytest.raises(
        sequence.SequenceRuntimeError,
        match="with initial_state, reroll_from_chunk can only be 0 or 1",
    ):
        _run_with_sources(
            monkeypatch,
            runtime,
            chunks=3,
            initial_state=state,
            reroll_from_chunk=2,
        )


def test_r0_run_storage_on_rejects_explicit_session_before_filesystem_io(
    monkeypatch, tmp_path
):
    runtime = cap._install_fake_runtime(monkeypatch)
    _, _, saved, _ = cap._run_sequence(runtime, chunks=1)
    monkeypatch.setattr(storage_runtime, "_output_root", lambda: tmp_path)
    controller = RunStorageController("r0-explicit-session")
    monkeypatch.setattr(storage_runtime, "get_active_run_storage", lambda: controller)

    with pytest.raises(
        RunStorageError,
        match="Run Storage cannot be combined with an explicit Session",
    ):
        cap._run_sequence(runtime, chunks=2, session=saved)

    assert list(tmp_path.rglob("*")) == []


def test_r0_run_storage_saved_prefix_precedes_initial_state(monkeypatch):
    runtime = cap._install_fake_runtime(monkeypatch)
    _, _, saved, _ = cap._run_sequence(runtime, chunks=1)
    state = _state_from_session(saved)
    runtime.samples.clear()
    storage = cap._CommitRecorder(runtime.timeline)
    storage.prepare = lambda **kwargs: saved
    monkeypatch.setattr(storage_runtime, "get_active_run_storage", lambda: storage)

    entries, last_state, _, _ = _run_with_sources(
        monkeypatch,
        runtime,
        chunks=2,
        initial_state=state,
    )

    assert len(entries) == 2
    assert entries[0]["reused"] is True
    assert len(runtime.samples) == 1
    assert last_state["clip_index"] == 2


def test_r0_run_storage_without_prefix_uses_initial_state(monkeypatch):
    runtime = cap._install_fake_runtime(monkeypatch)
    _, _, source, _ = cap._run_sequence(runtime, chunks=1)
    state = _state_from_session(source)
    runtime.samples.clear()
    storage = cap._CommitRecorder(runtime.timeline)
    monkeypatch.setattr(storage_runtime, "get_active_run_storage", lambda: storage)

    entries, last_state, _, _ = _run_with_sources(
        monkeypatch,
        runtime,
        chunks=1,
        initial_state=state,
    )

    assert len(entries) == len(runtime.samples) == 1
    assert entries[0]["plan"]["continuation"] is True
    assert last_state["clip_index"] == 10


@pytest.mark.parametrize(
    ("continuity", "expected_context"),
    (("Balanced — 22 frames", 22), ("Strong — 39 frames (Experimental)", 39)),
)
def test_r0_continuity_modes_keep_their_context_contract(
    monkeypatch, continuity, expected_context
):
    runtime = cap._install_fake_runtime(monkeypatch)
    entries, _, _, _ = _run_with_sources(
        monkeypatch,
        runtime,
        chunks=2,
        continuity=continuity,
    )

    assert entries[0]["plan"]["context_frames"] == 5
    assert entries[1]["plan"]["context_frames"] == expected_context
    assert entries[1]["plan"]["state_capacity_frames"] == 39


def _failure_controller(tmp_path, contract: dict) -> RunStorageController:
    controller = storage_cases._controller(tmp_path, contract)
    controller.lock = storage_runtime._RunLock(controller.run_root / ".lock")
    controller.revision_root.mkdir(parents=True, exist_ok=True)
    controller._write_manifest()
    return controller


@pytest.mark.parametrize(
    ("stage", "committed"),
    (
        ("before sampling", 0),
        ("after sampling before chunk commit", 0),
        ("after manifest update", 1),
        ("before output projection", 1),
        ("during output projection", 1),
        ("before storage finalize", 1),
    ),
)
def test_r0_failure_baseline_marks_revision_interrupted(
    tmp_path, stage, committed
):
    contract = storage_cases._contract(chunks=1)
    controller = _failure_controller(tmp_path, contract)

    with pytest.raises(RuntimeError, match=stage):
        with controller:
            if committed:
                controller.commit_chunk(storage_cases._entry(0, contract), position=0)
            raise RuntimeError(stage)

    manifest = json.loads(controller._manifest_path().read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert len(manifest["chunks"]) == committed
    assert stage in manifest["last_error"]


def test_r5_terminal_entry_cannot_expose_an_interrupted_half_group(
    tmp_path,
):
    contract = storage_cases._contract(chunks=2, terminal=True)
    controller = _failure_controller(tmp_path, contract)

    with pytest.raises(RunStorageError, match="physical group is incomplete"):
        with controller:
            controller.commit_chunk(storage_cases._entry(0, contract), position=0)

    manifest = json.loads(controller._manifest_path().read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["chunks"] == []
    project = json.loads((controller.run_root / "project.json").read_text(encoding="utf-8"))
    assert project["canonical_storage_revision_id"] is None


def test_r0_failure_before_manifest_update_is_recovered_by_exception_handler(
    monkeypatch, tmp_path
):
    contract = storage_cases._contract(chunks=1)
    controller = _failure_controller(tmp_path, contract)
    original = controller._write_manifest
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("before manifest update")
        return original()

    monkeypatch.setattr(controller, "_write_manifest", fail_once)
    with pytest.raises(RuntimeError, match="before manifest update"):
        with controller:
            controller.commit_chunk(storage_cases._entry(0, contract), position=0)

    manifest = json.loads(controller._manifest_path().read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert len(manifest["chunks"]) == 1
    assert calls == 2


def test_r0_failure_before_canonical_update_keeps_current_recovery_semantics(
    monkeypatch, tmp_path
):
    contract = storage_cases._contract(chunks=1)
    controller = _failure_controller(tmp_path, contract)
    entry = storage_cases._entry(0, contract)
    original = controller._write_project
    calls = 0

    def fail_once(*, set_canonical=False):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("before canonical pointer update")
        return original(set_canonical=set_canonical)

    monkeypatch.setattr(controller, "_write_project", fail_once)
    with pytest.raises(RuntimeError, match="before canonical pointer update"):
        with controller:
            controller.commit_chunk(entry, position=0)
            controller.finalize(
                session={"session_id": "r0", "chunks": [entry]},
                report="r0 failure baseline",
            )

    manifest = json.loads(controller._manifest_path().read_text(encoding="utf-8"))
    project = json.loads((controller.run_root / "project.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert calls == 2
    assert project["canonical_storage_revision_id"] is None
