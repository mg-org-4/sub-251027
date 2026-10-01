"""Isolated disposable-cache fault worker; never opens Run Storage or a ComfyUI backend."""
from pathlib import Path
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from decode_cache import store as module
from decode_cache.store import DecodeStore, MiB

parent, marker, point = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
store = DecodeStore(temp_parent=parent, memory_probe=lambda: 10**12, headroom_bytes=0)
store.configure("Auto", MiB, 8*MiB, 0)
if point == "reader":
    print(json.dumps({"hit": store.get("same-key") is not None, "root_created": store.root is not None}))
    sys.exit(0)


def pause():
    marker.write_text(point, encoding="utf-8")
    while True:
        time.sleep(0.05)


if point == "during_raw":
    original_blocks = module.tensor_blocks
    def blocks(tensor):
        iterator = iter(original_blocks(tensor))
        first = next(iterator)
        yield first[:16]
        pause()
    module.tensor_blocks = blocks
elif point == "before_publish":
    replace = os.replace
    def before_replace(src, dst):
        pause()
        return replace(src, dst)
    os.replace = before_replace

result = store.put("same-key", torch.ones(64, 64, 3), "video", {}, (), store.epoch)
if point == "after_publish":
    pause()
raise RuntimeError("fault point was not reached")
