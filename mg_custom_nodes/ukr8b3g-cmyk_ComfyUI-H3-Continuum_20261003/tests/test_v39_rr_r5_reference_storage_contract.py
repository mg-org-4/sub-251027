"""RR-R5 effective group identity and explicit Session evidence."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
import torch

from ComfyUI_H3_Continuum_Join.reference import REFERENCE_SIZE_MATCH_OUTPUT
from ComfyUI_H3_Continuum_Join.v3.reference_routing import (
    REFERENCE_SLOT_IDS,
    compile_reference_routing_schedule,
)
from ComfyUI_H3_Continuum_Join.v3.reference_runtime import ReferenceInputSet, ReferenceRoutingRuntime
from ComfyUI_H3_Continuum_Join.v3.reference_plan_inspector import project_reference_plan
from ComfyUI_H3_Continuum_Join.v3 import reference_storage_contract as storage_contract
from ComfyUI_H3_Continuum_Join.v3.reference_storage_contract import (
    ReferenceStorageContractError,
    ReferenceStoragePlan,
    build_reference_storage_plan,
    compatible_session_prefix,
    routed_session_identity,
    session_routing_settings,
    validate_reference_group_contracts,
)


def _image(value):
    return torch.full((1, 64, 64, 3), value, dtype=torch.float32)


def _runtime(*, routes, images=None, chunks=3, terminal=False):
    selectors = {slot: "off" for slot in REFERENCE_SLOT_IDS}
    selectors.update(routes)
    slots = [None] * 9
    for slot, image in (images or {}).items():
        slots[int(slot[1:]) - 1] = image
    return ReferenceRoutingRuntime(
        schedule=compile_reference_routing_schedule(
            total_chunks=chunks,
            terminal_merge_enabled=terminal,
            mode="Custom",
            selectors_by_slot=selectors,
        ),
        inputs=ReferenceInputSet(tuple(slots), 64, 64, REFERENCE_SIZE_MATCH_OUTPUT),
    )


def _plan(runtime, *, prompts=None, terminal_prompt=None, policy=None, through=None, last="none"):
    return build_reference_storage_plan(
        runtime=runtime,
        prompts=prompts or ["person walks", "person turns", "person speaks"],
        first_frame_hash="none",
        last_frame_hash=last,
        terminal_prompt=terminal_prompt,
        terminal_prompt_policy=policy,
        through_chunk=through,
    )


def test_reused_group_keeps_planned_disconnected_and_prompt_warnings():
    kwargs = {"routes": {"R1": "1"}, "chunks": 1}
    generated = _runtime(**kwargs)
    generated.storage_plan = _plan(generated, prompts=["@R1 runs"])
    generated.accepted_chunks = 1
    generated.generated_groups = (1,)
    expected = project_reference_plan(generated)["groups"][0]

    reused = _runtime(**kwargs)
    reused.storage_plan = _plan(reused, prompts=["@R1 runs"])
    reused.accepted_chunks = 1
    actual = project_reference_plan(reused)["groups"][0]
    assert expected["contract_sha256"] == actual["contract_sha256"]
    assert expected["warnings"] == actual["warnings"]
    assert expected["warnings"] == [
        "R1 is routed to physical group 1 but no image is connected",
        "R1 is not active in this physical group; R1's @R tag was left unchanged",
    ]


def test_plan_warning_publication_is_atomic_and_observed_warnings_win():
    runtime = _runtime(routes={"R1": "1"}, chunks=3, terminal=True)
    runtime.group_warnings[1] = ("observed warning",)
    runtime.observed_group_contracts[1] = {"descriptor": {}, "sha256": "a" * 64}
    with pytest.raises(ReferenceStorageContractError, match="terminal physical prompt"):
        _plan(runtime, prompts=["@R1", "second", "third"], policy="last")
    assert runtime.group_warnings == {1: ("observed warning",)}

    _plan(runtime, prompts=["@R1", "second", "third"],
          terminal_prompt="terminal", policy="last")
    assert runtime.group_warnings[1] == ("observed warning",)


def test_group_lookup_decodes_one_record_and_returns_independent_values(monkeypatch):
    payloads = tuple(json.dumps({"physical_group": group, "references": []})
                     for group in (1, 2, 3))
    plan = ReferenceStoragePlan(payloads, "a" * 64, ((), (), ()))
    original_loads = storage_contract.json.loads
    calls = []

    def counting_loads(payload, *args, **kwargs):
        calls.append(payload)
        return original_loads(payload, *args, **kwargs)

    monkeypatch.setattr(storage_contract.json, "loads", counting_loads)
    expected = {
        "descriptor": json.loads(payloads[2]),
        "sha256": hashlib.sha256(payloads[2].encode("utf-8")).hexdigest(),
    }
    calls.clear()
    first = plan.group_for(3)
    assert calls == [payloads[2]]
    assert first == expected
    first["descriptor"]["references"].append("changed")
    first["sha256"] = "0" * 64
    assert plan.group_for(3) == expected
    calls.clear()
    assert len(plan.group_contracts) == 3
    assert calls == list(payloads)
    with pytest.raises(ReferenceStorageContractError, match="physical group 4"):
        plan.group_for(4)


def test_only_effective_groups_change_and_unselected_image_is_not_prepared():
    images = {"R1": _image(0.1), "R4": _image(0.4), "R9": object()}
    before = _plan(_runtime(routes={"R1": "all"}, images=images))
    after = _plan(_runtime(routes={"R1": "1-2", "R4": "3"}, images=images))
    assert [item["sha256"] for item in before.group_contracts[:2]] == [
        item["sha256"] for item in after.group_contracts[:2]
    ]
    assert before.group_contracts[2]["sha256"] != after.group_contracts[2]["sha256"]
    assert after.group_for(3)["descriptor"]["references"][0]["source_slot_id"] == "R4"
    assert after.group_for(3)["descriptor"]["picture_order"] == ["R4"]
    validate_reference_group_contracts(after.group_contracts, chunk_count=3, terminal_merge_enabled=False)


def test_raw_selector_spelling_and_disconnected_request_do_not_change_group_hash():
    images = {"R1": _image(0.1)}
    a = _plan(_runtime(routes={"R1": "1,2"}, images=images))
    b = _plan(_runtime(routes={"R1": "1-2", "R9": "3"}, images=images))
    assert [record["sha256"] for record in a.group_contracts] == [
        record["sha256"] for record in b.group_contracts
    ]


def test_session_evidence_preserves_only_complete_compatible_group_prefix():
    images = {"R1": _image(0.1), "R4": _image(0.4)}
    old = _plan(_runtime(routes={"R1": "all"}, images=images))
    new = _plan(_runtime(routes={"R1": "1-2", "R4": "3"}, images=images))
    evidence = session_routing_settings(old, 3)
    assert compatible_session_prefix(evidence, new, saved_chunks=3) == 2
    assert compatible_session_prefix(session_routing_settings(old, 2), new, saved_chunks=2) == 2
    assert routed_session_identity("a" * 64) != "a" * 64
    assert routed_session_identity("a" * 64) == routed_session_identity("a" * 64)
    bad = copy.deepcopy(evidence)
    bad["accepted_groups"][1]["logical_chunks"] = [3]
    with pytest.raises(ReferenceStorageContractError):
        compatible_session_prefix(bad, new, saved_chunks=3)


def test_terminal_pair_is_one_identity_and_prompt_policy_is_group_local():
    images = {"R1": _image(0.1)}
    runtime = _runtime(routes={"R1": "all"}, images=images, chunks=4, terminal=True)
    prompts = ["one", "two", "three", "four"]
    shared = _plan(runtime, prompts=prompts, terminal_prompt="one shared", policy="shared")
    timeline = _plan(runtime, prompts=prompts, terminal_prompt="timeline pair", policy="timeline")
    assert len(shared.group_contracts) == 3
    assert [r["sha256"] for r in shared.group_contracts[:2]] == [
        r["sha256"] for r in timeline.group_contracts[:2]
    ]
    assert shared.group_contracts[2]["sha256"] != timeline.group_contracts[2]["sha256"]
    with pytest.raises(ReferenceStorageContractError):
        session_routing_settings(shared, 3)
    assert session_routing_settings(shared, 4)["accepted_groups"][-1]["logical_chunks"] == [3, 4]
    validate_reference_group_contracts(shared.group_contracts, chunk_count=4, terminal_merge_enabled=True)


def test_last_image_is_only_in_groups_where_qwen_presents_it():
    images = {"R1": _image(0.1)}
    no_references = _runtime(routes={}, images=images)
    before = _plan(no_references, last="a" * 64)
    after = _plan(no_references, last="b" * 64)
    assert [r["sha256"] for r in before.group_contracts[:2]] == [
        r["sha256"] for r in after.group_contracts[:2]
    ]
    assert before.group_contracts[2]["sha256"] != after.group_contracts[2]["sha256"]
    with_references = _runtime(routes={"R1": "all"}, images=images)
    first = _plan(with_references, last="a" * 64)
    changed = _plan(with_references, last="b" * 64)
    assert first.group_contracts[0]["sha256"] != changed.group_contracts[0]["sha256"]


def test_tampered_group_digest_or_boundary_is_rejected():
    plan = _plan(_runtime(routes={"R1": "all"}, images={"R1": _image(0.1)}))
    records = copy.deepcopy(plan.group_contracts)
    records[1]["descriptor"]["references"][0]["picture_number"] = 99
    with pytest.raises(ReferenceStorageContractError):
        validate_reference_group_contracts(records, chunk_count=3, terminal_merge_enabled=False)
    records = copy.deepcopy(plan.group_contracts)
    records[1]["descriptor"]["logical_chunks"] = [2, 3]
    with pytest.raises(ReferenceStorageContractError):
        validate_reference_group_contracts(records, chunk_count=3, terminal_merge_enabled=False)
