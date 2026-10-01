from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "ComfyUI_H3_Continuum_Join"
if PACKAGE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = package

import ComfyUI_H3_Continuum_Join.run_storage as storage_runtime


def _load_storage_cases():
    path = ROOT / "tests" / "test_v38_review_run_storage.py"
    spec = importlib.util.spec_from_file_location("r5_worker_storage_cases", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    scenario_root = Path(sys.argv[1]).resolve()
    stage = sys.argv[2]
    scenario_root.mkdir(parents=True, exist_ok=True)
    signal = scenario_root / "reached.json"

    def reached() -> None:
        signal.write_text(
            json.dumps({"stage": stage, "pid": os.getpid()}),
            encoding="utf-8",
        )
        while True:
            time.sleep(1)

    cases = _load_storage_cases()
    contract = cases._contract(chunks=2, terminal=True)
    controller = cases._controller(scenario_root, contract)
    controller.revision_root.mkdir(parents=True, exist_ok=True)
    controller._write_manifest()
    entries = (cases._entry(0, contract), cases._entry(1, contract))

    if stage == "raw_before":
        reached()

    if stage == "raw_mid":
        def partial_save(tensors, filename, *, metadata):
            with open(filename, "wb") as handle:
                handle.write(b"partial-r5-raw")
                handle.flush()
                os.fsync(handle.fileno())
                reached()

        storage_runtime.save_file = partial_save

    if stage == "first_raw_done":
        original_hash = storage_runtime._file_sha256
        hash_calls = 0

        def observed_hash(path):
            nonlocal hash_calls
            value = original_hash(path)
            hash_calls += 1
            if hash_calls == 2:
                reached()
            return value

        storage_runtime._file_sha256 = observed_hash

    if stage == "before_manifest":
        original_write_manifest = controller._write_manifest

        def before_manifest():
            reached()
            return original_write_manifest()

        controller._write_manifest = before_manifest

    if stage == "manifest_temp_mid":
        def partial_json_dump(payload, handle, **kwargs):
            handle.write('{"partial":')
            handle.flush()
            os.fsync(handle.fileno())
            reached()

        storage_runtime.json.dump = partial_json_dump

    if stage in {"before_replace", "after_replace"}:
        original_replace = storage_runtime._replace_json_atomically

        def observed_replace(temporary, destination):
            if Path(destination).name != "manifest.json":
                return original_replace(temporary, destination)
            if stage == "before_replace":
                reached()
            result = original_replace(temporary, destination)
            reached()
            return result

        storage_runtime._replace_json_atomically = observed_replace

    if stage == "after_directory_durability":
        original_fsync_dir = storage_runtime._fsync_dir

        def observed_fsync_dir(path):
            result = original_fsync_dir(path)
            if Path(path).resolve() == controller.revision_root.resolve():
                reached()
            return result

        storage_runtime._fsync_dir = observed_fsync_dir

    controller.commit_group(entries, positions=(0, 1))

    if stage in {"finalize_manifest_1", "finalize_manifest_2"}:
        original_finalize_write = controller._write_manifest
        finalize_writes = 0

        def observed_finalize_write():
            nonlocal finalize_writes
            result = original_finalize_write()
            finalize_writes += 1
            expected = 1 if stage == "finalize_manifest_1" else 2
            if finalize_writes == expected:
                reached()
            return result

        controller._write_manifest = observed_finalize_write

    if stage in {"project_before_replace", "project_after_replace"}:
        original_project_replace = storage_runtime._replace_json_atomically

        def observed_project_replace(temporary, destination):
            if Path(destination).name != "project.json":
                return original_project_replace(temporary, destination)
            if stage == "project_before_replace":
                reached()
            result = original_project_replace(temporary, destination)
            reached()
            return result

        storage_runtime._replace_json_atomically = observed_project_replace

    controller.finalize(
        session={"session_id": "r5-crash-worker", "chunks": list(entries)},
        report="r5 process crash fixture",
    )
    raise RuntimeError(f"stage did not block: {stage}")


if __name__ == "__main__":
    main()
