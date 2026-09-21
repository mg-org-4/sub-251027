from __future__ import annotations

import ast
import inspect
from pathlib import Path

import ComfyUI_H3_Continuum_Join.run_storage as storage_runtime
from ComfyUI_H3_Continuum_Join.run_storage import EmptyValidatedPrefix
from ComfyUI_H3_Continuum_Join.v3.runtime_coordinator import (
    project_selected_continuation_source,
)
from ComfyUI_H3_Continuum_Join.v3.execution_planner import (
    PROJECTION_CURRENT_REVIEW_UNIT,
    PROJECTION_FULL_ACCEPTED_PREFIX,
    build_execution_plan,
)
from ComfyUI_H3_Continuum_Join.v3.planning_types import (
    ContinuationSourceFacts,
    PrefixFacts,
)
from ComfyUI_H3_Continuum_Join.v3.review_control import (
    GENERATION_MODE_REVIEW,
    REVIEW_ACTION_CONTINUE,
    REVISION_STATUS_COMPLETE,
    REVISION_STATUS_REVIEW_READY,
    RUN_STORAGE_SAVE_AUTO_RESUME,
    ReviewUnit,
    resolve_review_execution,
)

from . import test_v38_review_execution_cap as cap


def _source(
    kind: str,
    *,
    accepted: int,
    reroll: int,
    boundaries: tuple[tuple[int, int], ...],
) -> ContinuationSourceFacts:
    return project_selected_continuation_source(
        kind=kind,
        selected_source={
            "accepted_chunks": accepted,
            "reroll_from_chunk": reroll,
            "physical_group_boundaries": boundaries,
        },
    )


def _review_continue(*, chunks: int, prefix: int, terminal: bool):
    return resolve_review_execution(
        generation_mode=GENERATION_MODE_REVIEW,
        review_action=REVIEW_ACTION_CONTINUE,
        configured_chunks=chunks,
        validated_prefix_count=prefix,
        terminal_merge_enabled=terminal,
        terminal_pair_start=(chunks - 1 if terminal else None),
        manual_regenerate_from=0,
        run_storage_mode=RUN_STORAGE_SAVE_AUTO_RESUME,
        latest_review_unit=(
            None
            if prefix == 0
            else ReviewUnit(start=prefix, end=prefix, physical_group=prefix)
        ),
        latest_revision_status=(
            None if prefix == 0 else REVISION_STATUS_REVIEW_READY
        ),
        latest_effective_nonce=0,
    )


def test_r3_projects_each_already_selected_continuation_source_kind():
    run_storage = _source(
        "run_storage",
        accepted=2,
        reroll=0,
        boundaries=((1, 1), (2, 2)),
    )
    explicit_session = _source(
        "explicit_session",
        accepted=1,
        reroll=2,
        boundaries=((1, 1),),
    )
    initial_state = _source(
        "initial_state",
        accepted=0,
        reroll=1,
        boundaries=(),
    )
    none = project_selected_continuation_source(
        kind="none",
        selected_source=None,
    )

    assert run_storage == ContinuationSourceFacts(
        "run_storage", 2, 0, ((1, 1), (2, 2))
    )
    assert explicit_session == ContinuationSourceFacts(
        "explicit_session", 1, 2, ((1, 1),)
    )
    assert initial_state == ContinuationSourceFacts(
        "initial_state", 0, 1, ()
    )
    assert none == ContinuationSourceFacts("none", 0, 0, ())


def test_r3_projection_boundary_has_no_selection_policy_calls():
    tree = ast.parse(inspect.getsource(project_selected_continuation_source))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called_names.isdisjoint(
        {
            "find_latest_review_head",
            "resolve_review_execution",
            "resolve_take_execution",
            "validate_session",
            "validate_state",
        }
    )


def test_r3_full_plan_owns_all_pending_physical_groups():
    plan = build_execution_plan(
        continuation_source_facts=ContinuationSourceFacts("none", 0, 0, ()),
        configured_chunks=3,
        terminal_merge_enabled=False,
        max_new_physical_groups=None,
        review_execution=None,
    )

    assert [group.logical_chunks for group in plan.groups_to_generate] == [
        (1,),
        (2,),
        (3,),
    ]
    assert plan.expected_status == REVISION_STATUS_COMPLETE
    assert plan.expected_review_unit is None
    assert plan.projection.kind == PROJECTION_FULL_ACCEPTED_PREFIX


def test_r3_review_plan_generates_one_group_and_projects_only_that_unit():
    execution = _review_continue(chunks=3, prefix=1, terminal=False)
    plan = build_execution_plan(
        continuation_source_facts=ContinuationSourceFacts(
            "run_storage", 1, 0, ((1, 1),)
        ),
        configured_chunks=3,
        terminal_merge_enabled=False,
        max_new_physical_groups=execution.max_new_physical_groups,
        review_execution=execution,
        retained_review_unit=ReviewUnit(1, 1, 1),
    )

    assert [group.logical_chunks for group in plan.groups_to_generate] == [(2,)]
    assert plan.expected_status == REVISION_STATUS_REVIEW_READY
    assert plan.expected_review_unit == ReviewUnit(2, 2, 2)
    assert plan.projection.kind == PROJECTION_CURRENT_REVIEW_UNIT
    assert (plan.projection.start_chunk, plan.projection.end_chunk) == (2, 2)


def test_r3_terminal_group_is_atomic_and_completion_projects_full_prefix():
    execution = _review_continue(chunks=3, prefix=1, terminal=True)
    plan = build_execution_plan(
        continuation_source_facts=ContinuationSourceFacts(
            "run_storage", 1, 0, ((1, 1),)
        ),
        configured_chunks=3,
        terminal_merge_enabled=True,
        max_new_physical_groups=execution.max_new_physical_groups,
        review_execution=execution,
        retained_review_unit=ReviewUnit(1, 1, 1),
    )

    assert len(plan.groups_to_generate) == 1
    assert plan.groups_to_generate[0].logical_chunks == (2, 3)
    assert plan.groups_to_generate[0].terminal_atomic is True
    assert plan.expected_status == REVISION_STATUS_COMPLETE
    assert plan.expected_review_unit == ReviewUnit(2, 3, 2)
    assert plan.projection.kind == PROJECTION_FULL_ACCEPTED_PREFIX


def test_r3_planner_api_accepts_only_one_selected_source():
    parameters = inspect.signature(build_execution_plan).parameters
    assert "continuation_source_facts" in parameters
    assert "storage_prefix_facts" not in parameters
    assert "explicit_session_facts" not in parameters
    assert "initial_state_facts" not in parameters


def test_r3_planner_module_imports_no_torch_or_comfyui():
    module_path = Path(inspect.getsourcefile(build_execution_plan))
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
    assert not any("comfy" in name.lower() for name in imported)


def test_r3_validated_prefix_projection_contract_remains_prefix_facts():
    facts = EmptyValidatedPrefix(configured_chunks=3).to_planning_facts()
    assert isinstance(facts, PrefixFacts)
    assert not isinstance(facts, ContinuationSourceFacts)


def test_r3_storage_off_runtime_consumes_the_pure_group_plan(monkeypatch):
    runtime = cap._install_fake_runtime(monkeypatch)
    entries, _, session, _ = cap._run_sequence(runtime, chunks=3)

    assert len(entries) == len(session["chunks"]) == 3
    assert len(runtime.samples) == 3
    assert storage_runtime.get_active_run_storage() is None
