"""RR-R2 CPU contracts for selectors and logical/physical routing schedules."""

from __future__ import annotations

import pytest

from ComfyUI_H3_Continuum_Join.branch_provenance import physical_groups
from ComfyUI_H3_Continuum_Join.v3.reference_routing import (
    REFERENCE_SLOT_IDS,
    compile_reference_routing_schedule,
    parse_chunk_selector,
)


def _selectors(**overrides: str) -> dict[str, str]:
    result = {slot_id: "off" for slot_id in REFERENCE_SLOT_IDS}
    result.update(overrides)
    return result


def test_absent_mode_preserves_legacy_all_for_every_logical_chunk():
    schedule = compile_reference_routing_schedule(
        total_chunks=4,
        terminal_merge_enabled=True,
    )

    assert schedule.valid
    assert schedule.mode == "all"
    assert schedule.legacy_fallback is True
    assert tuple(route.logical_chunk for route in schedule.logical_routes) == (1, 2, 3, 4)
    assert all(route.reference_slot_ids == REFERENCE_SLOT_IDS for route in schedule.logical_routes)
    assert tuple(route.logical_chunks for route in schedule.physical_routes) == tuple(
        tuple(range(group.start, group.end + 1))
        for group in physical_groups(chunks=4, terminal_merge_enabled=True)
    )


def test_explicit_all_matches_legacy_routes_but_is_not_legacy_fallback():
    legacy = compile_reference_routing_schedule(
        total_chunks=3,
        terminal_merge_enabled=False,
    )
    explicit = compile_reference_routing_schedule(
        total_chunks=3,
        terminal_merge_enabled=False,
        mode="All",
        selectors_by_slot={"not-used": object()},
    )

    assert explicit.valid
    assert explicit.legacy_fallback is False
    assert explicit.logical_routes == legacy.logical_routes


def test_custom_selector_forms_compile_in_fixed_slot_order():
    schedule = compile_reference_routing_schedule(
        total_chunks=6,
        terminal_merge_enabled=False,
        mode="Custom",
        selectors_by_slot=_selectors(
            R1="all",
            R2="off",
            R3="1-3",
            R4="2, 4, 6",
            R5="1-2, 4-5",
            R9="6",
        ),
    )

    assert schedule.valid
    by_chunk = {route.logical_chunk: route.reference_slot_ids for route in schedule.logical_routes}
    assert by_chunk == {
        1: ("R1", "R3", "R5"),
        2: ("R1", "R3", "R4", "R5"),
        3: ("R1", "R3"),
        4: ("R1", "R4", "R5"),
        5: ("R1", "R5"),
        6: ("R1", "R4", "R9"),
    }


def test_explicit_custom_all_off_is_valid_and_distinct_from_legacy_all():
    schedule = compile_reference_routing_schedule(
        total_chunks=3,
        terminal_merge_enabled=True,
        mode="Custom",
        selectors_by_slot=_selectors(),
    )

    assert schedule.valid
    assert schedule.legacy_fallback is False
    assert all(route.reference_slot_ids == () for route in schedule.logical_routes)
    assert all(route.status == "empty" for route in schedule.physical_routes)
    assert all(route.reference_slot_ids == () for route in schedule.physical_routes)


@pytest.mark.parametrize(
    ("selector", "expected_code"),
    [
        ("", "selector_empty"),
        ("0", "chunk_out_of_range"),
        ("6", "chunk_out_of_range"),
        ("3-2", "reversed_range"),
        ("1,,2", "empty_selector_item"),
        ("1-3,3", "duplicate_chunk"),
        ("all,2", "invalid_selector_item"),
        ("-1", "invalid_selector_item"),
        (None, "selector_not_string"),
    ],
)
def test_invalid_selector_is_classified_without_partial_selection(selector, expected_code):
    parsed = parse_chunk_selector(selector, total_chunks=5)

    assert not parsed.valid
    assert parsed.kind == "invalid"
    assert parsed.logical_chunks == ()
    assert expected_code in {issue.code for issue in parsed.issues}


def test_selector_parser_preserves_all_off_and_expands_inclusive_ranges():
    assert parse_chunk_selector("ALL", total_chunks=4).logical_chunks == (1, 2, 3, 4)
    assert parse_chunk_selector(" off ", total_chunks=4).kind == "off"
    assert parse_chunk_selector(" 1 - 2, 4 ", total_chunks=4).logical_chunks == (1, 2, 4)


def test_resume_does_not_rebase_project_wide_logical_chunk_numbers():
    schedule = compile_reference_routing_schedule(
        total_chunks=5,
        terminal_merge_enabled=False,
        mode="Custom",
        selectors_by_slot=_selectors(R4="4-5"),
    )

    assert schedule.valid
    assert tuple(route.logical_chunk for route in schedule.logical_routes) == (1, 2, 3, 4, 5)
    assert tuple(route.logical_chunk for route in schedule.logical_routes if route.reference_slot_ids) == (4, 5)
    assert tuple(route.reference_slot_ids for route in schedule.logical_routes) == (
        (), (), (), ("R4",), ("R4",)
    )


def test_terminal_atomic_group_classifies_route_conflict_without_auto_correction():
    schedule = compile_reference_routing_schedule(
        total_chunks=4,
        terminal_merge_enabled=True,
        mode="Custom",
        selectors_by_slot=_selectors(R1="3", R2="4"),
    )

    terminal = schedule.physical_routes[-1]
    assert schedule.valid
    assert terminal.logical_chunks == (3, 4)
    assert terminal.terminal_atomic is True
    assert terminal.status == "conflict"
    assert terminal.logical_reference_slot_ids == (("R1",), ("R2",))
    assert terminal.reference_slot_ids is None
    assert schedule.logical_routes[2].reference_slot_ids == ("R1",)
    assert schedule.logical_routes[3].reference_slot_ids == ("R2",)


def test_terminal_atomic_group_accepts_identical_routes_and_empty_route():
    consistent = compile_reference_routing_schedule(
        total_chunks=4,
        terminal_merge_enabled=True,
        mode="Custom",
        selectors_by_slot=_selectors(R7="3-4"),
    )
    empty = compile_reference_routing_schedule(
        total_chunks=4,
        terminal_merge_enabled=True,
        mode="Custom",
        selectors_by_slot=_selectors(),
    )

    assert consistent.physical_routes[-1].status == "consistent"
    assert consistent.physical_routes[-1].reference_slot_ids == ("R7",)
    assert empty.physical_routes[-1].status == "empty"
    assert empty.physical_routes[-1].reference_slot_ids == ()


def test_route_changes_between_separate_physical_groups_are_not_conflicts():
    schedule = compile_reference_routing_schedule(
        total_chunks=3,
        terminal_merge_enabled=False,
        mode="Custom",
        selectors_by_slot=_selectors(R1="1", R2="2-3"),
    )

    assert tuple(route.status for route in schedule.physical_routes) == (
        "consistent",
        "consistent",
        "consistent",
    )
    assert tuple(route.reference_slot_ids for route in schedule.physical_routes) == (
        ("R1",), ("R2",), ("R2",)
    )


def test_custom_missing_selectors_unknown_slots_and_invalid_modes_are_classified():
    missing = compile_reference_routing_schedule(
        total_chunks=2,
        terminal_merge_enabled=False,
        mode="Custom",
        selectors_by_slot={"R1": "off"},
    )
    unknown = compile_reference_routing_schedule(
        total_chunks=2,
        terminal_merge_enabled=False,
        mode="Custom",
        selectors_by_slot=_selectors(R9="off", R10="all"),
    )
    invalid_mode = compile_reference_routing_schedule(
        total_chunks=2,
        terminal_merge_enabled=False,
        mode="Automatic",
    )

    assert not missing.valid
    assert sum(issue.code == "missing_selector" for issue in missing.issues) == 8
    assert not unknown.valid
    assert any(issue.code == "unknown_reference_slot" for issue in unknown.issues)
    assert not invalid_mode.valid
    assert invalid_mode.issues[0].code == "invalid_mode"


def test_invalid_terminal_merge_topology_is_classified():
    schedule = compile_reference_routing_schedule(
        total_chunks=1,
        terminal_merge_enabled=True,
        mode="All",
    )

    assert not schedule.valid
    assert schedule.logical_routes == ()
    assert schedule.physical_routes == ()
    assert schedule.issues[0].code == "invalid_physical_groups"
