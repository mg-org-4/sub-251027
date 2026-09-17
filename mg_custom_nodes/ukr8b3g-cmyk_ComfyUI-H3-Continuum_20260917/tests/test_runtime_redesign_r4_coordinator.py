from __future__ import annotations

import ast
import inspect

import pytest

from ComfyUI_H3_Continuum_Join.run_storage import RunStorageError
from ComfyUI_H3_Continuum_Join.v2 import sequence
from ComfyUI_H3_Continuum_Join.v3 import runtime_coordinator as coordinator_module
from ComfyUI_H3_Continuum_Join.v3.runtime_coordinator import (
    InternalRuntimeCoordinator,
)
from ComfyUI_H3_Continuum_Join.v3.sampling_engine import sample_physical_group


EXPECTED_RUN_SEQUENCE_PARAMETERS = (
    "model", "clip", "video_vae", "audio_vae", "sampler", "sigmas",
    "first_frame", "last_frame", "prompt_plan", "width", "height",
    "continuity", "base_seed", "audio_continuity", "exact_total_duration",
    "diagnostics_mode", "reroll_from_chunk", "reroll_nonce",
    "strict_compatibility", "debug", "seam_correction", "enable_preview",
    "session", "initial_state", "latent_only", "reference_assets",
    "reference_audio_source", "reference_audio_vae",
    "driving_audio_source", "driving_audio_vae", "reference_video_source",
    "timeline_video_source", "guide_source", "capture_refine_context",
    "memory_attribution", "prompt_conditioning_cache",
    "reference_encode_cache", "continuation_transport",
    "max_new_physical_groups", "_memory_attribution_collector",
    "_diagnostic_continuation_policy",
)


def _coordinator(*, storage=False, session=None, state=None):
    return InternalRuntimeCoordinator(
        storage_controller=(object() if storage else None),
        input_session=session,
        initial_state=state,
    )


def test_r4_run_sequence_keeps_the_exact_explicit_41_keyword_signature():
    public = inspect.signature(sequence.run_sequence)
    internal = inspect.signature(sequence._run_runtime_internal)

    assert tuple(public.parameters) == EXPECTED_RUN_SEQUENCE_PARAMETERS
    assert tuple(internal.parameters) == EXPECTED_RUN_SEQUENCE_PARAMETERS
    assert public == internal
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in public.parameters.values()
    )


def test_r4_storage_plus_explicit_session_keeps_existing_error():
    coordinator = _coordinator(storage=True, session={"session": True})

    with pytest.raises(
        RunStorageError,
        match="Run Storage cannot be combined with an explicit Session",
    ):
        coordinator.assert_storage_session_compatible()


def test_r4_explicit_session_suppresses_state_before_session_validation(caplog):
    state = {"would": "otherwise be considered"}
    coordinator = _coordinator(
        session={"invalid": "session payload"},
        state=state,
    )

    session, effective_state = coordinator.prepare_inputs()

    assert session == {"invalid": "session payload"}
    assert effective_state is None
    assert coordinator.initial_state is None
    assert "using the session and ignoring initial_state" in caplog.text


def test_r4_usable_storage_prefix_precedes_initial_state(monkeypatch):
    state = {"state": True}
    coordinator = _coordinator(storage=True, state=state)
    monkeypatch.setattr(
        coordinator_module,
        "entry_to_state",
        lambda entry: {"width": 64, "height": 64, "source": entry["source"]},
    )

    selected = coordinator.select_continuation_source(
        preserved=[{"source": "storage"}],
        initial_state=state,
        chunks=2,
        width=64,
        height=64,
        reroll_from_chunk=0,
        effective_reroll_from_chunk=0,
        terminal_merge_enabled=False,
    )

    assert selected.continuation_source_facts.kind == "run_storage"
    assert selected.continuation_source_facts.accepted_chunks == 1
    assert selected.previous_state["source"] == "storage"


def test_r4_initial_state_is_considered_only_without_usable_prefix(monkeypatch):
    state = {"state": True}
    coordinator = _coordinator(storage=True, state=state)
    monkeypatch.setattr(
        coordinator_module,
        "validate_state",
        lambda value: {"width": 64, "height": 64, "source": value["state"]},
    )

    selected = coordinator.select_continuation_source(
        preserved=[],
        initial_state=state,
        chunks=2,
        width=64,
        height=64,
        reroll_from_chunk=0,
        effective_reroll_from_chunk=0,
        terminal_merge_enabled=False,
    )

    assert selected.continuation_source_facts.kind == "initial_state"
    assert selected.continuation_source_facts.accepted_chunks == 0
    assert selected.previous_state["source"] is True


def test_r4_sampling_engine_owns_one_sample_and_preserves_observer_order():
    events = []

    class Sigmas:
        shape = (5,)

    class LayoutProfiler:
        def begin_sampling_group(self, **kwargs):
            events.append(("layout_begin", kwargs))
            return "layout-token"

        def finish_sampling_group(self, token):
            events.append(("layout_finish", token))

    class DiagnosticPolicy:
        def before_sampling(self, **kwargs):
            events.append(("diagnostic_before", kwargs))
            return "diagnostic-token"

    def planner_event(_planner, event, **kwargs):
        events.append((f"planner_{event}", kwargs))

    def sample_callable(**kwargs):
        events.append(("sample", kwargs))
        return "sampled"

    sampled, diagnostic_token = sample_physical_group(
        sample_callable=sample_callable,
        model="model",
        conditioning="conditioning",
        latent="latent",
        sampler="sampler",
        sigmas=Sigmas(),
        seed=42,
        enable_preview=True,
        physical_group=2,
        logical_chunks=(2,),
        terminal_atomic=False,
        layout_validation_profiler=LayoutProfiler(),
        packed_row_planner="planner",
        planner_event=planner_event,
        diagnostic_policy=DiagnosticPolicy(),
        context_frames=22,
    )

    assert sampled == "sampled"
    assert diagnostic_token == "diagnostic-token"
    assert [name for name, _ in events] == [
        "layout_begin",
        "diagnostic_before",
        "planner_begin_group",
        "sample",
        "planner_finish_group",
        "layout_finish",
    ]


def test_r4_runtime_uses_sampling_engine_and_engine_has_no_storage_import():
    runtime_tree = ast.parse(inspect.getsource(sequence._run_runtime_internal))
    calls = [
        node.func.id
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    engine_source = inspect.getsource(inspect.getmodule(sample_physical_group))

    assert calls.count("sample_physical_group") == 2
    assert calls.count("sample_chunk") == 0
    assert "run_storage" not in engine_source.lower()


def test_r4_continuation_priority_branches_have_one_owner():
    runtime_source = inspect.getsource(coordinator_module)
    sequence_source = inspect.getsource(sequence._run_runtime_internal)

    assert runtime_source.count("class InternalRuntimeCoordinator") == 1
    assert runtime_source.count("def project_selected_continuation_source") == 1
    assert "selected_kind" in runtime_source
    assert "selected_source_kind" not in sequence_source
    assert "validate_state(initial_state)" not in sequence_source
