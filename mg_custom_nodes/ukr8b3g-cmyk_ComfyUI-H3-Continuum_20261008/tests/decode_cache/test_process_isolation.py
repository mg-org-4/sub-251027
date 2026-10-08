"""Process termination tests on the host OS, not a Windows/power-loss claim."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

WORKER = Path(__file__).parent / "helpers" / "cache_process_worker.py"


@pytest.mark.parametrize("point", ["during_raw", "before_publish", "after_publish"])
def test_process_kill_leaves_only_disposable_private_files(tmp_path, point):
    parent = tmp_path / "private-cache-parent"
    parent.mkdir()
    marker = tmp_path / "ready.txt"
    user_file = tmp_path / "untouched-run-storage.txt"
    user_file.write_text("existing accepted data")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
    p = subprocess.Popen([sys.executable, str(WORKER), str(parent), str(marker), point],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    try:
        deadline = time.monotonic() + 20
        while not marker.exists() and time.monotonic() < deadline:
            if p.poll() is not None:
                raise AssertionError(p.stderr.read().decode())
            time.sleep(0.03)
        assert marker.exists(), "worker never reached the kill boundary"
        p.kill()
        p.wait(timeout=5)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=5)
        if p.stderr:
            p.stderr.close()
    raw = list(parent.rglob("*.npy"))
    temporary = list(parent.rglob("*.tmp"))
    if point == "after_publish":
        assert len(raw) == 1 and not temporary
        assert np.array_equal(np.load(raw[0], allow_pickle=False), np.ones((64, 64, 3), dtype=np.float32))
    else:
        assert not raw and len(temporary) == 1
    reader = subprocess.run([sys.executable, str(WORKER), str(parent), str(marker), "reader"],
                            capture_output=True, text=True, env=env, check=True, timeout=20)
    assert json.loads(reader.stdout.splitlines()[-1]) == {"hit": False, "root_created": False}
    assert user_file.read_text() == "existing accepted data"
