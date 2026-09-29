from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

import ComfyUI_H3_Continuum_Join.run_storage as storage_runtime
from ComfyUI_H3_Continuum_Join.run_storage import RunStorageError
from ComfyUI_H3_Continuum_Join.v2.session import SessionValidationError

from . import test_v38_review_run_storage as storage_cases


def _ready_controller(tmp_path, *, chunks: int = 2, terminal: bool = False):
    contract = storage_cases._contract(chunks=chunks, terminal=terminal)
    controller = storage_cases._controller(tmp_path, contract)
    controller.lock = storage_runtime._RunLock(controller.run_root / ".lock")
    controller.revision_root.mkdir(parents=True, exist_ok=True)
    controller._write_manifest()
    return controller, contract


def _raw_path(controller, record: dict) -> Path:
    return controller.revision_root / "chunks" / record["filename"]


def test_r5_retry_uses_a_new_raw_without_overwriting_the_accepted_raw(tmp_path):
    controller, contract = _ready_controller(tmp_path, chunks=1)
    controller.commit_group(
        (storage_cases._entry(0, contract),), positions=(0,)
    )
    first = dict(controller.manifest["chunks"][0])
    first_path = _raw_path(controller, first)
    first_bytes = first_path.read_bytes()

    replacement = storage_cases._entry(0, contract)
    replacement["video"] = replacement["video"] + 1
    controller.commit_group((replacement,), positions=(0,))
    second = dict(controller.manifest["chunks"][0])

    assert second["filename"] != first["filename"]
    assert first_path.read_bytes() == first_bytes
    assert hashlib.sha256(first_bytes).hexdigest() == first["file_sha256"]
    assert _raw_path(controller, second).is_file()


def test_r5_terminal_group_uses_one_transaction_and_one_manifest_switch(
    monkeypatch, tmp_path
):
    controller, contract = _ready_controller(tmp_path, terminal=True)
    writes = 0
    original = controller._write_manifest

    def counted_write():
        nonlocal writes
        writes += 1
        return original()

    monkeypatch.setattr(controller, "_write_manifest", counted_write)
    controller.commit_group(
        (
            storage_cases._entry(0, contract),
            storage_cases._entry(1, contract),
        ),
        positions=(0, 1),
    )

    records = controller.manifest["chunks"]
    assert writes == 1
    assert [record["sequence_index"] for record in records] == [0, 1]
    transaction_parts = [
        record["filename"].split("-txn-", 1)[1].split("-chunk-", 1)[0]
        for record in records
    ]
    assert len(set(transaction_parts)) == 1


def test_r5_terminal_half_is_rejected_before_any_raw_write(tmp_path):
    controller, contract = _ready_controller(tmp_path, terminal=True)

    with pytest.raises(RunStorageError, match="physical group is incomplete"):
        controller.commit_chunk(storage_cases._entry(0, contract), position=0)

    assert controller.manifest["chunks"] == []
    assert not list((controller.revision_root / "chunks").glob("*.safetensors"))


def test_r5_validates_every_entry_before_writing_group_raw(tmp_path):
    controller, contract = _ready_controller(tmp_path, terminal=True)
    invalid = dict(storage_cases._entry(1, contract))
    invalid.pop("video")

    with pytest.raises(SessionValidationError):
        controller.commit_group(
            (storage_cases._entry(0, contract), invalid), positions=(0, 1)
        )

    assert controller.manifest["chunks"] == []
    assert not list((controller.revision_root / "chunks").glob("*.safetensors"))


def test_r5_second_raw_failure_leaves_only_unreferenced_orphan(
    monkeypatch, tmp_path
):
    controller, contract = _ready_controller(tmp_path, terminal=True)
    original = storage_runtime.save_file
    calls = 0

    def fail_second(tensors, filename, *, metadata):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic second raw failure")
        return original(tensors, filename, metadata=metadata)

    monkeypatch.setattr(storage_runtime, "save_file", fail_second)
    with pytest.raises(RuntimeError, match="second raw failure"):
        controller.commit_group(
            (
                storage_cases._entry(0, contract),
                storage_cases._entry(1, contract),
            ),
            positions=(0, 1),
        )

    persisted = json.loads(controller._manifest_path().read_text(encoding="utf-8"))
    assert persisted["chunks"] == []
    assert controller.manifest["chunks"] == []
    assert len(list((controller.revision_root / "chunks").glob("*.safetensors"))) == 1


def test_r5_catchable_manifest_failure_recovers_only_the_complete_group(
    monkeypatch, tmp_path
):
    controller, contract = _ready_controller(tmp_path, terminal=True)
    original = controller._write_manifest
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic manifest switch failure")
        return original()

    monkeypatch.setattr(controller, "_write_manifest", fail_once)
    with pytest.raises(RuntimeError, match="manifest switch failure"):
        with controller:
            controller.commit_group(
                (
                    storage_cases._entry(0, contract),
                    storage_cases._entry(1, contract),
                ),
                positions=(0, 1),
            )

    persisted = json.loads(controller._manifest_path().read_text(encoding="utf-8"))
    assert calls == 2
    assert persisted["status"] == "interrupted"
    assert [record["sequence_index"] for record in persisted["chunks"]] == [0, 1]


def test_r5_commit_group_does_not_write_the_canonical_project(monkeypatch, tmp_path):
    controller, contract = _ready_controller(tmp_path, chunks=1)
    project_writes = 0

    def unexpected_project_write(*, set_canonical=False):
        nonlocal project_writes
        project_writes += 1

    monkeypatch.setattr(controller, "_write_project", unexpected_project_write)
    controller.commit_group(
        (storage_cases._entry(0, contract),), positions=(0,)
    )

    assert project_writes == 0
    assert not (controller.run_root / "project.json").exists()


def test_r5_reader_keeps_legacy_chunk_filename_compatibility(tmp_path):
    controller, contract = _ready_controller(tmp_path, chunks=1)
    controller.commit_group(
        (storage_cases._entry(0, contract),), positions=(0,)
    )
    record = controller.manifest["chunks"][0]
    current_path = _raw_path(controller, record)
    legacy_path = current_path.with_name("chunk_0001.safetensors")
    current_path.replace(legacy_path)
    record["filename"] = legacy_path.name
    record["file_size"] = legacy_path.stat().st_size
    record["file_sha256"] = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
    controller._write_manifest()

    entries, records = controller._valid_prefix(
        controller.manifest,
        list(contract["chunk_contract_hashes"]),
        current_contract=contract,
    )

    assert len(entries) == len(records) == 1
    assert records[0]["filename"] == "chunk_0001.safetensors"


@pytest.mark.parametrize(
    ("stage", "expected_records", "expected_raws"),
    (
        ("raw_before", 0, 0),
        ("raw_mid", 0, 0),
        ("first_raw_done", 0, 1),
        ("before_manifest", 0, 2),
        ("manifest_temp_mid", 0, 2),
        ("before_replace", 0, 2),
        ("after_replace", 2, 2),
        ("after_directory_durability", 2, 2),
    ),
)
def test_r5_fresh_process_kill_never_exposes_a_partial_terminal_group(
    tmp_path, stage, expected_records, expected_raws
):
    worker = (
        Path(__file__).parent / "helpers" / "r5_process_crash_worker.py"
    )
    scenario_root = tmp_path / stage
    process = subprocess.Popen(
        [sys.executable, str(worker), str(scenario_root), stage],
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    signal = scenario_root / "reached.json"
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline and not signal.exists():
            if process.poll() is not None:
                raise AssertionError(
                    f"crash worker exited before {stage}: {process.returncode}"
                )
            time.sleep(0.05)
        assert signal.exists(), f"crash worker did not reach {stage}"
        process.terminate()
        process.wait(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=15)

    manifest_paths = list((scenario_root / "run" / "revisions").glob("*/manifest.json"))
    assert len(manifest_paths) == 1
    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    assert len(manifest["chunks"]) == expected_records
    raw_paths = list((manifest_paths[0].parent / "chunks").glob("*.safetensors"))
    assert len(raw_paths) == expected_raws
    for record in manifest["chunks"]:
        raw = manifest_paths[0].parent / "chunks" / record["filename"]
        assert raw.is_file()
        assert raw.stat().st_size == record["file_size"]
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == record["file_sha256"]


@pytest.mark.parametrize(
    ("stage", "project_exists"),
    (
        ("finalize_manifest_1", False),
        ("finalize_manifest_2", False),
        ("project_before_replace", False),
        ("project_after_replace", True),
    ),
)
def test_r5_finalize_and_canonical_kill_windows_preserve_fresh_reader_selection(
    tmp_path, stage, project_exists
):
    worker = Path(__file__).parent / "helpers" / "r5_process_crash_worker.py"
    scenario_root = tmp_path / stage
    process = subprocess.Popen(
        [sys.executable, str(worker), str(scenario_root), stage],
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    signal = scenario_root / "reached.json"
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline and not signal.exists():
            if process.poll() is not None:
                raise AssertionError(
                    f"crash worker exited before {stage}: {process.returncode}"
                )
            time.sleep(0.05)
        assert signal.exists(), f"crash worker did not reach {stage}"
        process.terminate()
        process.wait(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=15)

    run_root = scenario_root / "run"
    project_path = run_root / "project.json"
    assert project_path.exists() is project_exists
    manifests = list((run_root / "revisions").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert len(manifest["chunks"]) == 2

    fresh = storage_runtime.RunStorageController("phase-d-review")
    fresh.run_root = run_root
    fresh.revisions_root = run_root / "revisions"
    _, catalog, chains = fresh._provenance_catalog()
    selected_revision, selected_chain = fresh._canonical_project_selection(
        catalog=catalog,
        chains=chains,
    )
    # The R0 fallback selects a complete legacy-style manifest even before its
    # provenance and project pointer are published. R5 preserves that exact
    # fresh-reader behavior; only commit_group raw publication changes.
    assert selected_revision == manifest["revision_id"]
    assert selected_chain
