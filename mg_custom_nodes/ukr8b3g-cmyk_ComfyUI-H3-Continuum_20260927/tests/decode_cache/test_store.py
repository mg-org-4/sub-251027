import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

from decode_cache.identity import CacheBypass
from decode_cache.store import DecodeStore, MiB


@pytest.fixture
def store(tmp_path):
    s = DecodeStore(temp_parent=tmp_path, memory_probe=lambda: 10**12,
                    disk_probe=lambda p: SimpleNamespace(free=10**12), headroom_bytes=0)
    s.configure("Auto", 1024**2, 1024**2, 0)
    yield s
    s.close()


def put(store, key, x, stream="video"):
    return store.put(key, x, stream, {}, (), store.epoch)


@pytest.mark.parametrize("mode", ["Auto", "RAM"])
def test_hit_bit_exact_mutation_isolation(store, mode):
    store.configure(mode, MiB, MiB, 0)
    x = torch.arange(120).float().reshape(5, 3, 4, 2)
    replacement = put(store, "a", x)
    first = replacement[0] if replacement else x
    first.add_(7)
    hit, _ = store.get("a")
    assert torch.equal(hit, torch.arange(120).float().reshape(5, 3, 4, 2))
    hit.zero_()
    again, _ = store.get("a")
    assert again.sum().item() > 0


@pytest.mark.parametrize("mode", ["Auto", "RAM"])
def test_view_mutation_does_not_poison(store, mode):
    store.configure(mode, MiB, MiB, 0)
    put(store, "a", torch.ones(4, 3, 2, 1))
    hit, _ = store.get("a")
    hit[1:3].mul_(0)
    other, _ = store.get("a")
    assert other.sum() == 24


def test_noncontiguous_serialization(store):
    x = torch.arange(120).float().reshape(5, 3, 4, 2).transpose(0, 2)
    put(store, "a", x)
    y, _ = store.get("a")
    assert y.dtype == x.dtype and torch.equal(x, y)


def test_dtype_unchanged(store):
    for dtype in (torch.float16, torch.float32, torch.float64):
        x = torch.arange(12).to(dtype)
        put(store, str(dtype), x)
        y, _ = store.get(str(dtype))
        assert y.dtype == dtype and torch.equal(x, y)


def test_ram_lru_and_budget(store):
    store.configure("RAM", 32, MiB, 0)
    put(store, "a", torch.ones(4))
    put(store, "b", torch.ones(4)*2)
    assert store.get("a") is not None
    put(store, "c", torch.ones(4)*3)
    assert store.get("b") is None
    assert store.ram_bytes == 32


def test_ram_pressure_evicts(store):
    store.configure("RAM", MiB, MiB, 0)
    put(store, "a", torch.ones(4))
    store.headroom_bytes = 1024
    store.memory_probe = lambda: 0
    store.configure("RAM", MiB, MiB, 0)
    assert store.get("a") is None
    assert store.ram_bytes == 0


def test_auto_under_ram_pressure_uses_disk(store):
    store.memory_probe = lambda: 0
    put(store, "a", torch.ones(24), "audio")
    assert store.entries["a"].path is not None


def test_disk_full_skips_cache(store):
    store.disk_probe = lambda p: SimpleNamespace(free=0)
    assert put(store, "a", torch.ones(4)) is None
    assert store.get("a") is None


def test_oversized_skips(store):
    store.configure("Auto", 0, 8, 0)
    assert put(store, "a", torch.ones(4)) is None
    assert store.root is None


def test_living_mmap_survives_clear_and_accounts_bytes(store):
    hit = put(store, "a", torch.ones(4))[0]
    path = store.entries["a"].path
    store.clear()
    assert path.exists()
    assert store.disk_bytes > 0
    hit.add_(2)
    assert torch.equal(hit, torch.ones(4)*3)
    del hit
    gc.collect()
    store.stats()
    assert not path.exists()
    assert store.disk_bytes == 0


def test_no_cross_process_store_reuse(store, tmp_path):
    put(store, "a", torch.ones(4))
    other = DecodeStore(temp_parent=tmp_path)
    assert other.get("a") is None
    other.close()


@pytest.mark.parametrize("damage", ["truncate", "header", "data", "delete"])
def test_corrupt_disk_invalidates(store, damage):
    result = put(store, "a", torch.ones(20))
    del result
    gc.collect()
    path = store.entries["a"].path
    if damage == "delete":
        path.unlink()
    elif damage == "truncate":
        path.write_bytes(b"bad")
    else:
        with open(path, "r+b") as f:
            f.seek(-1 if damage == "data" else 0, 2 if damage == "data" else 0)
            b = f.read(1)
            f.seek(-1, 1)
            f.write(bytes([b[0] ^ 1]))
    with pytest.raises((CacheBypass, OSError, ValueError)):
        store.get("a")
    assert store.get("a") is None


def test_corrupt_ram_invalidates(store):
    store.configure("RAM", MiB, MiB, 0)
    put(store, "a", torch.ones(4))
    store.entries["a"].tensor.zero_()
    with pytest.raises(CacheBypass):
        store.get("a")
    assert store.get("a") is None


def test_dead_vae_eviction(store):
    class Owner:
        pass
    owner = Owner()
    store.put("a", torch.ones(4), "audio", {}, (weakref.ref(owner),), store.epoch)
    del owner
    gc.collect()
    assert store.get("a") is None


def test_write_fault_no_published_entry(store, monkeypatch):
    def fail(*args):
        raise OSError("read-only test cache")
    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        put(store, "a", torch.ones(4))
    assert not store.entries
    assert not list(store.root.iterdir())


def test_unknown_temp_is_never_loaded_or_removed(store):
    root = store._directory()
    stale = root / "foreign.tmp"
    stale.write_text("not a cache entry")
    store.close()
    assert stale.exists()


def test_reset_and_disabled_no_storage(store):
    put(store, "a", torch.ones(4))
    store.configure("Off", MiB, MiB, 1)
    assert store.get("a") is None
    assert put(store, "b", torch.ones(4)) is None


@pytest.mark.parametrize("mode", ["Auto", "RAM"])
def test_token_change_clears_once_then_same_token_keeps_cache(store, mode):
    store.configure(mode, MiB, MiB, 0)
    put(store, "before", torch.ones(4))
    epoch = store.epoch
    store.configure(mode, MiB, MiB, 0)
    assert store.get("before") is not None
    assert store.epoch == epoch
    # The UI increments only the submitted INT; configure acts on next Queue.
    store.configure(mode, MiB, MiB, 1)
    assert store.get("before") is None
    assert store.epoch > epoch
    put(store, "after", torch.ones(4))
    epoch = store.epoch
    store.configure(mode, MiB, MiB, 1)
    assert store.get("after") is not None
    assert store.epoch == epoch


def test_epoch_prevents_inflight_reinsertion(store):
    epoch = store.epoch
    store.clear()
    assert store.put("a", torch.ones(4), "video", {}, (), epoch) is None
    assert store.get("a") is None


def test_pending_disk_files_are_in_quota(store):
    store.configure("Auto", 0, 6000, 0)
    x = torch.ones(1000)  # 4000 bytes + header; next insert must not delete a live mmap
    first = put(store, "a", x)
    assert first is None  # conservative header reservation is > 6000
    store.configure("Auto", 0, 10000, 0)
    first = put(store, "a", x)
    assert first is not None
    store.clear()
    # 4128 + conservative 8096 > quota; pinned pending file counts
    assert put(store, "b", x) is None
    del first
    gc.collect()
    assert put(store, "b", x) is not None


@pytest.mark.parametrize("mode", ["Auto", "RAM"])
def test_output_metadata_is_defensively_copied(store, mode):
    store.configure(mode, MiB, MiB, 0)
    extra = {"sample_rate": 32000, "future_metadata": {"labels": ["original"]}}
    store.put("a", torch.ones(12), "video", extra, (), store.epoch)
    extra["future_metadata"]["labels"][0] = "caller mutation"
    _, first = store.get("a")
    assert first["future_metadata"]["labels"] == ["original"]
    first["future_metadata"]["labels"][0] = "downstream mutation"
    _, second = store.get("a")
    assert second["future_metadata"]["labels"] == ["original"]


def test_delete_failure_is_accounted_and_retried(store, monkeypatch):
    put(store, "a", torch.ones(12))
    path = store.entries["a"].path
    original_unlink = Path.unlink
    def denied(self, *args, **kwargs):
        if self == path:
            raise PermissionError("simulated Windows sharing violation")
        return original_unlink(self, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", denied)
        store.clear()
        assert store.disk_bytes > 0 and len(store.retired) == 1
    store.stats()
    assert not path.exists() and store.disk_bytes == 0


def test_symlink_cache_file_is_not_opened(store, tmp_path):
    put(store, "a", torch.ones(12))
    path = store.entries["a"].path
    outside = tmp_path / "outside-user-file.bin"
    outside.write_bytes(b"DO NOT TOUCH")
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links not available")
    with pytest.raises(CacheBypass):
        store.get("a")
    assert outside.read_bytes() == b"DO NOT TOUCH"


def test_store_garbage_collection_cleans_unleased_private_files(tmp_path):
    s = DecodeStore(temp_parent=tmp_path, memory_probe=lambda: 10**12, headroom_bytes=0)
    put(s, "a", torch.ones(12))
    root = s.root
    assert root.exists()
    ref = weakref.ref(s)
    del s
    gc.collect()
    assert ref() is None and not root.exists()


def test_explicit_private_cache_parent_uses_only_child_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("H3_DECODE_CACHE_TEMP_DIR", str(tmp_path))
    s = DecodeStore(memory_probe=lambda: 10**12, headroom_bytes=0)
    put(s, "a", torch.ones(12))
    assert s.root.parent == tmp_path and s.root.name.startswith("h3-decode-cache-")
    s.close()
    assert tmp_path.exists()
