from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import ComfyUI_H3_Continuum_Join.run_storage as storage_runtime
from ComfyUI_H3_Continuum_Join.run_storage import (
    EmptyValidatedPrefix,
    RunStorageController,
    ValidatedPrefix,
)
from ComfyUI_H3_Continuum_Join.v3.planning_types import PrefixFacts
from ComfyUI_H3_Continuum_Join.v3.review_control import (
    GENERATION_MODE_REVIEW,
    REVIEW_ACTION_CONTINUE,
)

from . import test_v38_review_run_storage as cases


def _review_controller(tmp_path, contract: dict) -> RunStorageController:
    controller = RunStorageController("phase-d-review")
    controller.run_root = tmp_path / "run"
    controller.revisions_root = controller.run_root / "revisions"
    controller.prompts = [
        f"prompt {index + 1}" for index in range(contract["chunk_count"])
    ]
    controller.configure_review(
        generation_mode=GENERATION_MODE_REVIEW,
        review_action=REVIEW_ACTION_CONTINUE,
    )
    return controller


def test_r2_raw_prefix_is_loaded_before_the_single_review_decision(
    monkeypatch, tmp_path
):
    contract = cases._contract(chunks=3)
    cases._persist(
        tmp_path,
        contract,
        prefix=1,
        review_unit=(1, 1),
        updated_utc="2026-09-14T00:00:00+00:00",
    )
    controller = _review_controller(tmp_path, contract)
    events = []
    original_load = controller._load_entry
    original_resolve = storage_runtime.resolve_review_execution

    def load(*args, **kwargs):
        events.append("raw_loaded")
        return original_load(*args, **kwargs)

    def resolve(**kwargs):
        events.append("review_decision")
        return original_resolve(**kwargs)

    monkeypatch.setattr(controller, "_load_entry", load)
    monkeypatch.setattr(storage_runtime, "resolve_review_execution", resolve)

    prefix = controller._load_validated_review_prefix(contract, resume_safe=True)
    assert isinstance(prefix, ValidatedPrefix)
    assert events == ["raw_loaded"]
    controller._resolve_review_contract(
        contract,
        requested_nonce=0,
        resume_safe=True,
        validated_prefix=prefix,
    )

    assert events == ["raw_loaded", "review_decision"]
    assert controller.review_runtime_metrics == {
        "review_decision_created": 1,
        "review_reconcile": 0,
    }
    assert not hasattr(controller, "_reconcile_review_execution_with_reused_prefix")


def test_r2_validated_prefix_is_structurally_immutable_and_defensive(tmp_path):
    contract = cases._contract(chunks=3)
    cases._persist(
        tmp_path,
        contract,
        prefix=1,
        review_unit=(1, 1),
        updated_utc="2026-09-14T00:00:00+00:00",
    )
    controller = _review_controller(tmp_path, contract)
    candidate = controller.find_latest_review_head(contract)
    prefix = controller._freeze_validated_prefix(candidate, contract=contract)

    assert isinstance(prefix.entries, tuple)
    assert isinstance(prefix.records, tuple)
    assert prefix.physical_group_boundaries == ((1, 1),)
    with pytest.raises(FrozenInstanceError):
        prefix.status = "changed"
    with pytest.raises(TypeError):
        prefix.manifest["status"] = "changed"
    candidate["manifest"]["status"] = "changed-after-freeze"
    assert prefix.manifest["status"] != "changed-after-freeze"

    facts = prefix.to_planning_facts()
    assert isinstance(facts, PrefixFacts)
    assert facts.accepted_chunks == 1
    assert facts.configured_chunks == 3
    assert facts.review_unit.start == facts.review_unit.end == 1


def test_r2_empty_validated_prefix_only_means_no_run_storage_prefix():
    prefix = EmptyValidatedPrefix(configured_chunks=3)
    facts = prefix.to_planning_facts()

    assert prefix.entries == prefix.records == ()
    assert facts == PrefixFacts(
        accepted_chunks=0,
        configured_chunks=3,
        revision_id=None,
        status=None,
        review_unit=None,
        effective_nonce=0,
        branch_regenerate_from=0,
        physical_group_boundaries=(),
    )


def test_r2_repository_validation_and_finalize_do_not_create_review_decisions(
    monkeypatch, tmp_path
):
    contract = cases._contract(chunks=3)

    def unexpected(**kwargs):
        pytest.fail("metadata validation must not create a Review Decision")

    monkeypatch.setattr(storage_runtime, "resolve_review_execution", unexpected)
    cases._persist(
        tmp_path,
        contract,
        prefix=1,
        review_unit=(1, 1),
        updated_utc="2026-09-14T00:00:00+00:00",
    )
    controller = _review_controller(tmp_path, contract)
    prefix = controller._load_validated_review_prefix(contract, resume_safe=True)

    assert isinstance(prefix, ValidatedPrefix)
    assert controller.review_runtime_metrics["review_decision_created"] == 0


def test_r2_a_second_review_decision_is_rejected(tmp_path):
    contract = cases._contract(chunks=3)
    controller = _review_controller(tmp_path, contract)
    prefix = controller._load_validated_review_prefix(contract, resume_safe=True)
    controller._resolve_review_contract(
        contract,
        requested_nonce=0,
        resume_safe=True,
        validated_prefix=prefix,
    )

    with pytest.raises(storage_runtime.RunStorageError, match="already created"):
        controller._resolve_review_contract(
            contract,
            requested_nonce=0,
            resume_safe=True,
            validated_prefix=prefix,
        )


def test_r2_fixed_prefix_cannot_be_replaced_before_decision(tmp_path):
    contract = cases._contract(chunks=3)
    controller = _review_controller(tmp_path, contract)
    fixed = controller._load_validated_review_prefix(contract, resume_safe=True)
    replacement = EmptyValidatedPrefix(configured_chunks=3)

    assert replacement is not fixed
    with pytest.raises(storage_runtime.RunStorageError, match="cannot replace"):
        controller._resolve_review_contract(
            contract,
            requested_nonce=0,
            resume_safe=True,
            validated_prefix=replacement,
        )
    assert controller.review_runtime_metrics == {
        "review_decision_created": 0,
        "review_reconcile": 0,
    }


def test_r2_take_policy_also_creates_exactly_one_decision(tmp_path):
    contract = cases._contract(chunks=3, boundary=2, nonce=1)
    saved = cases._persist(
        tmp_path,
        contract,
        prefix=2,
        review_unit=(2, 2),
        updated_utc="2026-09-14T00:00:01+00:00",
    )
    revision = cases._take(cases._project(saved), group=2, nonce=1)

    controller, _, _, _ = cases._resolve(
        tmp_path,
        cases._contract(chunks=3),
        take_action=storage_runtime.TAKE_ACTION_CONTINUE,
        take_group=2,
        take_revision_id=revision["revision_id"],
    )

    assert isinstance(controller.validated_prefix, ValidatedPrefix)
    assert controller.review_runtime_metrics == {
        "review_decision_created": 1,
        "review_reconcile": 0,
    }
