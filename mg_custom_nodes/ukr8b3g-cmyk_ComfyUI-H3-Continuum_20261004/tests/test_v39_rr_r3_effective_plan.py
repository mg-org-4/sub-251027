"""RR-R3 CPU contracts for Reference slot and Core Picture correspondence."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest
import torch

from ComfyUI_H3_Continuum_Join.reference import (
    REFERENCE_SIZE_MATCH_OUTPUT,
    build_hybrid_presentation_items,
    prepare_reference_assets,
)
from ComfyUI_H3_Continuum_Join.v3.reference_effective_plan import (
    compile_effective_reference_plan,
)
from ComfyUI_H3_Continuum_Join.v3.reference_routing import (
    REFERENCE_SLOT_IDS,
    compile_reference_routing_schedule,
)


def _selectors(**overrides: str) -> dict[str, str]:
    result = {slot_id: "off" for slot_id in REFERENCE_SLOT_IDS}
    result.update(overrides)
    return result


def _schedule(*, chunks: int = 4, terminal: bool = False, mode=None, selectors=None):
    return compile_reference_routing_schedule(
        total_chunks=chunks,
        terminal_merge_enabled=terminal,
        mode=mode,
        selectors_by_slot=selectors,
    )


def _hash(slot_id: str) -> str:
    return hashlib.sha256(slot_id.encode("ascii")).hexdigest()


def _plan(schedule, slots=("R1", "R4", "R9"), *, first=False, last=False):
    return compile_effective_reference_plan(
        routing_schedule=schedule,
        source_slot_ids=slots,
        reference_image_hashes=tuple(_hash(slot_id) for slot_id in slots),
        reference_asset_count=len(slots),
        has_first_image=first,
        has_last_image=last,
    )


def test_unset_routing_preserves_sparse_legacy_reference_order():
    schedule = _schedule(chunks=3)
    plan = _plan(schedule)

    assert plan.valid and plan.legacy_fallback
    assert plan.connected_source_slot_ids == ("R1", "R4", "R9")
    assert tuple(route.logical_chunk for route in plan.logical_routes) == (1, 2, 3)
    for route in plan.logical_routes:
        assert route.requested_source_slot_ids == REFERENCE_SLOT_IDS
        assert [item.source_slot_id for item in route.picture_map.references] == ["R1", "R4", "R9"]
        assert [item.asset_index for item in route.picture_map.references] == [0, 1, 2]
        assert [item.image_sha256 for item in route.picture_map.references] == [
            _hash(slot_id) for slot_id in ("R1", "R4", "R9")
        ]
        assert [item.picture_number for item in route.picture_map.references] == [1, 2, 3]
    assert all(group.status == "active" for group in plan.physical_routes)


@pytest.mark.parametrize(
    ("has_first", "has_last", "first_number", "last_number", "ref_numbers"),
    [
        (False, False, None, None, (1, 2)),
        (True, False, 1, None, (2, 3)),
        (False, True, None, 1, (2, 3)),
        (True, True, 1, 2, (3, 4)),
    ],
)
def test_anchor_offsets_match_existing_core_presentation_order(
    has_first, has_last, first_number, last_number, ref_numbers
):
    plan = _plan(_schedule(chunks=1), slots=("R4", "R9"), first=has_first, last=has_last)
    picture_map = plan.logical_routes[0].picture_map
    first = object() if has_first else None
    last = object() if has_last else None
    images = (object(), object())
    presentation = build_hybrid_presentation_items(
        [{"type": "image", "data": image} for image in images],
        first_image=first,
        last_image=last,
    )

    assert plan.valid
    assert picture_map.first_picture_number == first_number
    assert picture_map.last_picture_number == last_number
    assert tuple(item.picture_number for item in picture_map.references) == ref_numbers
    assert tuple(item.asset_index for item in picture_map.references) == (0, 1)
    for item in picture_map.references:
        assert presentation[item.picture_number - 1]["data"] is images[item.asset_index]
        assert item.picture_tag == f"<Picture {item.picture_number}>"
    if first_number is not None:
        assert presentation[first_number - 1]["data"] is first
    if last_number is not None:
        assert presentation[last_number - 1]["data"] is last


def test_custom_routes_recompact_pictures_without_losing_fixed_slot_identity():
    schedule = _schedule(
        mode="Custom",
        selectors=_selectors(R4="1,3", R9="2-3"),
    )
    plan = _plan(schedule, first=True)

    assert plan.valid
    assert tuple(route.logical_chunk for route in plan.logical_routes) == (1, 2, 3, 4)
    assert [
        tuple((item.source_slot_id, item.asset_index, item.picture_number)
              for item in route.picture_map.references)
        for route in plan.logical_routes
    ] == [
        (("R4", 1, 2),),
        (("R9", 2, 2),),
        (("R4", 1, 2), ("R9", 2, 3)),
        (),
    ]
    assert plan.physical_routes[-1].status == "empty"
    assert plan.physical_routes[-1].picture_map.first_picture_number == 1


def test_requested_but_disconnected_slot_stays_visible_as_an_empty_effective_route():
    schedule = _schedule(chunks=2, mode="Custom", selectors=_selectors(R5="1"))
    plan = _plan(schedule, slots=("R4", "R9"))

    assert plan.valid
    assert plan.logical_routes[0].requested_source_slot_ids == ("R5",)
    assert plan.logical_routes[0].picture_map.references == ()
    assert plan.physical_routes[0].routing_status == "consistent"
    assert plan.physical_routes[0].status == "empty"
    assert plan.physical_routes[0].requested_source_slot_ids == ("R5",)


def test_terminal_conflict_has_no_group_picture_map_but_retains_both_logical_maps():
    schedule = _schedule(
        chunks=3,
        terminal=True,
        mode="Custom",
        selectors=_selectors(R4="2", R9="3"),
    )
    plan = _plan(schedule, slots=("R4", "R9"), first=True, last=True)

    assert plan.valid and plan.has_route_conflicts
    assert [item.source_slot_id for item in plan.logical_routes[1].picture_map.references] == ["R4"]
    assert [item.source_slot_id for item in plan.logical_routes[2].picture_map.references] == ["R9"]
    terminal = plan.physical_routes[-1]
    assert terminal.logical_chunks == (2, 3)
    assert terminal.terminal_atomic
    assert terminal.status == terminal.routing_status == "conflict"
    assert terminal.requested_source_slot_ids is None
    assert terminal.picture_map is None


def test_consistent_terminal_route_has_one_shared_group_picture_map():
    schedule = _schedule(
        chunks=3, terminal=True, mode="Custom", selectors=_selectors(R4="2-3")
    )
    plan = _plan(schedule, slots=("R4", "R9"), first=True, last=True)

    assert plan.valid and not plan.has_route_conflicts
    terminal = plan.physical_routes[-1]
    assert terminal.logical_chunks == (2, 3)
    assert terminal.status == "active"
    assert terminal.picture_map.first_picture_number == 1
    assert terminal.picture_map.last_picture_number == 2
    assert tuple((item.source_slot_id, item.asset_index, item.picture_number)
                 for item in terminal.picture_map.references) == (("R4", 0, 3),)


def test_explicit_custom_all_off_keeps_first_last_without_reference_pictures():
    schedule = _schedule(chunks=3, terminal=True, mode="Custom", selectors=_selectors())
    plan = _plan(schedule, slots=("R1",), first=True, last=True)

    assert plan.valid and not plan.legacy_fallback
    assert all(route.picture_map.references == () for route in plan.logical_routes)
    assert all(group.status == "empty" for group in plan.physical_routes)
    assert plan.physical_routes[-1].picture_map.first_picture_number == 1
    assert plan.physical_routes[-1].picture_map.last_picture_number == 2


def test_same_image_content_in_two_slots_retains_two_picture_identities():
    image = torch.full((1, 32, 32, 3), 0.5)
    assets = prepare_reference_assets(
        reference_image_1=image,
        reference_image_2=None,
        reference_image_4=image,
        output_width=32,
        output_height=32,
        size_mode=REFERENCE_SIZE_MATCH_OUTPUT,
    )
    assert assets is not None
    assert assets.image_hashes[0] == assets.image_hashes[1]
    plan = compile_effective_reference_plan(
        routing_schedule=_schedule(chunks=1),
        source_slot_ids=assets.source_slot_ids,
        reference_image_hashes=assets.image_hashes,
        reference_asset_count=assets.count,
        has_first_image=False,
        has_last_image=False,
    )
    assert plan.valid
    assert tuple((item.source_slot_id, item.asset_index, item.picture_number)
                 for item in plan.logical_routes[0].picture_map.references) == (
        ("R1", 0, 1), ("R4", 1, 2)
    )
    assert tuple(item.image_sha256 for item in plan.logical_routes[0].picture_map.references) == assets.image_hashes


@pytest.mark.parametrize(
    ("slots", "count", "expected_issue"),
    [
        ((), 1, "missing_source_slots"),
        (("R1",), 2, "asset_count_mismatch"),
        (("R1", "R1"), 2, "duplicate_source_slot"),
        (("R9", "R4"), 2, "source_slot_order"),
        (("R10",), 1, "unknown_source_slot"),
    ],
)
def test_broken_source_metadata_is_classified_without_a_partial_plan(slots, count, expected_issue):
    plan = compile_effective_reference_plan(
        routing_schedule=_schedule(chunks=2),
        source_slot_ids=slots,
        reference_image_hashes=tuple(_hash(slot_id) for slot_id in slots),
        reference_asset_count=count,
        has_first_image=False,
        has_last_image=False,
    )
    assert not plan.valid
    assert plan.logical_routes == plan.physical_routes == ()
    assert expected_issue in {issue.code for issue in plan.issues}


def test_invalid_r2_schedule_and_malformed_logical_route_are_classified():
    invalid = _schedule(chunks=2, mode="Custom", selectors=_selectors(R1="3"))
    result = _plan(invalid)
    assert not result.valid
    assert any(issue.code == "routing_chunk_out_of_range" for issue in result.issues)

    good = _schedule(chunks=2)
    bad_logical = replace(good.logical_routes[0], reference_slot_ids=("R10",))
    bad = replace(good, logical_routes=(bad_logical, *good.logical_routes[1:]))
    malformed = _plan(bad)
    assert not malformed.valid
    assert any(issue.code == "invalid_routing_schedule" for issue in malformed.issues)


def test_anchor_flags_require_explicit_boolean_presence():
    result = compile_effective_reference_plan(
        routing_schedule=_schedule(chunks=1),
        source_slot_ids=(),
        reference_image_hashes=(),
        reference_asset_count=0,
        has_first_image=1,
        has_last_image=False,
    )
    assert not result.valid
    assert result.issues[0].code == "invalid_anchor_flags"


@pytest.mark.parametrize(
    ("hashes", "issue_code"),
    [
        ((), "hash_count_mismatch"),
        (("not-a-sha",), "invalid_image_hash"),
    ],
)
def test_invalid_reference_content_identity_does_not_create_a_partial_plan(hashes, issue_code):
    result = compile_effective_reference_plan(
        routing_schedule=_schedule(chunks=1),
        source_slot_ids=("R4",),
        reference_image_hashes=hashes,
        reference_asset_count=1,
        has_first_image=False,
        has_last_image=False,
    )
    assert not result.valid
    assert result.logical_routes == result.physical_routes == ()
    assert issue_code in {issue.code for issue in result.issues}
