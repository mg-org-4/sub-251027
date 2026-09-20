"""Private, disposable decoded tensors. No Run Storage or global GPU lifecycle access."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import threading
import uuid
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch

from .identity import CacheBypass, tensor_blocks, tensor_digest

MiB = 1024**2
GiB = 1024**3


def available_ram():
    try:
        import psutil  # already provided by ComfyUI; optional for cache operation
        return int(psutil.virtual_memory().available)
    except (ImportError, OSError, AttributeError):
        return 0  # unknown headroom must not trigger RAM retention


@dataclass
class Entry:
    shape: tuple
    dtype: str
    digest: str
    payload_bytes: int
    guards: tuple
    extra: dict
    tensor: torch.Tensor | None = None
    path: Path | None = None
    file_bytes: int = 0
    leases: list = field(default_factory=list)

    def live(self):
        self.leases[:] = [ref for ref in self.leases if ref() is not None]
        return bool(self.leases)


def _cleanup_private_files(state):
    """Best effort at object/process exit; never close arrays still owned downstream."""
    for path, entry in list(state["entries"].items()):
        if not entry.live():
            try:
                path.unlink(missing_ok=True)
                state["entries"].pop(path, None)
            except OSError:
                pass
    if state["root"] is not None:
        try:
            state["root"].rmdir()
        except OSError:
            pass


class DecodeStore:
    """One store per Helper instance. Auto uses disk for video and small RAM for audio.

    RAM hits return a clone. Disk hits return a new copy-on-write mmap, never
    a writable shared backing file. Already-returned results outlive eviction.
    """

    def __init__(self, temp_parent=None, memory_probe=available_ram, disk_probe=shutil.disk_usage,
                 headroom_bytes=2*GiB, interrupt=None):
        self._lock = threading.RLock()
        self.entries = OrderedDict()
        self.retired = []
        self.root = None
        self.temp_parent = temp_parent if temp_parent is not None else (os.environ.get("H3_DECODE_CACHE_TEMP_DIR") or None)
        self.memory_probe = memory_probe
        self.disk_probe = disk_probe
        self.headroom_bytes = headroom_bytes
        self.interrupt = interrupt or (lambda: None)
        self.ram_limit = 512*MiB
        self.disk_limit = 16*GiB
        self.mode = "Auto"
        self.reset_token = 0
        self.epoch = 0
        self.max_entries = 256
        # Finalizer state has only paths, disk-entry metadata and weak leases;
        # no decoded RAM tensor, VAE, store, node, or GPU model is retained.
        self._cleanup_state = {"root": None, "entries": {}}
        weakref.finalize(self, _cleanup_private_files, self._cleanup_state)

    def configure(self, mode, ram_limit, disk_limit, reset_token):
        with self._lock:
            settings = (mode, int(ram_limit), int(disk_limit), int(reset_token))
            if settings != (self.mode, self.ram_limit, self.disk_limit, self.reset_token):
                self.clear()
                self.mode, self.ram_limit, self.disk_limit, self.reset_token = settings
            self._reap()
            if self.memory_probe() < self.headroom_bytes:
                for key, entry in list(self.entries.items()):
                    if entry.tensor is not None:
                        self._drop(key)

    @property
    def ram_bytes(self):
        return sum(e.payload_bytes for e in self.entries.values() if e.tensor is not None)

    @property
    def disk_bytes(self):
        return sum(e.file_bytes for e in self.entries.values() if e.path is not None) + sum(e.file_bytes for e in self.retired)

    def _delete(self, entry):
        if entry.path is None:
            return True
        if entry.live():
            return False
        try:
            entry.path.unlink(missing_ok=True)
            self._cleanup_state["entries"].pop(entry.path, None)
            return True
        except OSError:
            return False

    def _drop(self, key):
        entry = self.entries.pop(key)
        if entry.path is not None and not self._delete(entry):
            self.retired.append(entry)

    def _reap(self):
        self.retired[:] = [e for e in self.retired if not self._delete(e)]
        for key, entry in list(self.entries.items()):
            if not all(ref() is not None for ref in entry.guards):
                self._drop(key)

    def clear(self):
        with self._lock:
            self.epoch += 1
            for key in list(self.entries):
                self._drop(key)
            self._reap()

    def close(self):
        self.clear()
        if self.root is not None:
            try:
                self.root.rmdir()  # only an empty helper-owned directory
            except OSError:
                pass

    def _directory(self):
        if self.root is None:
            self.root = Path(tempfile.mkdtemp(prefix=f"h3-decode-cache-{os.getpid()}-", dir=self.temp_parent))
            self._cleanup_state["root"] = self.root
        return self.root

    def _ram_room(self, size):
        while self.ram_bytes + size > self.ram_limit:
            victim = next((k for k, e in self.entries.items() if e.tensor is not None), None)
            if victim is None:
                return False
            self._drop(victim)
        # Budget is additional retention only, not a claim about whole-process RSS.
        return self.memory_probe() >= self.headroom_bytes + 2*size

    def _disk_room(self, size):
        self._reap()
        while self.disk_bytes + size > self.disk_limit:
            victim = next((k for k, e in self.entries.items() if e.path is not None and not e.live()), None)
            if victim is None:
                return False
            self._drop(victim)
        root = self._directory()
        return self.disk_probe(root).free >= size + 256*MiB

    def _read_disk(self, entry):
        path = entry.path
        if path is None or path.is_symlink() or path.stat().st_size != entry.file_bytes:
            raise CacheBypass("disk cache file missing/changed")
        # Verify the exact serialized payload before exposing a mapping.
        # Header/size are also checked; pickle is never allowed.
        array = np.load(path, mmap_mode="c", allow_pickle=False, max_header_size=4096)
        if tuple(array.shape) != entry.shape or array.dtype.str != entry.dtype or not array.flags.c_contiguous:
            raise CacheBypass("disk cache header mismatch")
        if array.nbytes != entry.payload_bytes:
            raise CacheBypass("disk cache size mismatch")
        raw = memoryview(array).cast("B")
        h = hashlib.sha256()
        for start in range(0, len(raw), 8*MiB):
            self.interrupt()
            h.update(raw[start:start+8*MiB])
        if h.hexdigest() != entry.digest:
            raise CacheBypass("disk cache checksum mismatch")
        entry.leases.append(weakref.ref(array))
        # mode='c' makes writable, private pages. mode='r' is NOT used.
        return torch.from_numpy(array)

    def get(self, key):
        with self._lock:
            self._reap()
            entry = self.entries.get(key)
            if entry is None:
                return None
            if not all(ref() is not None for ref in entry.guards):
                self._drop(key)
                return None
            self.entries.move_to_end(key)
            if entry.tensor is not None:
                if self.memory_probe() < self.headroom_bytes + entry.payload_bytes:
                    self._drop(key)
                    return None
                if (tuple(entry.tensor.shape) != entry.shape
                        or entry.tensor.numpy().dtype.str != entry.dtype
                        or tensor_digest(entry.tensor, self.interrupt) != entry.digest):
                    self._drop(key)
                    raise CacheBypass("RAM cache checksum/header mismatch")
                # Independent storage protects the cache from in-place seam/color edits.
                return entry.tensor.clone(), deepcopy(entry.extra)
            try:
                tensor = self._read_disk(entry)
            except Exception:
                self._drop(key)
                raise
            return tensor, deepcopy(entry.extra)

    def put(self, key, tensor, stream, extra, guards, expected_epoch):
        with self._lock:
            if expected_epoch != self.epoch or self.mode == "Off":
                return None
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                    or tensor.layout != torch.strided or tensor.is_nested or tensor.numel() == 0):
                return None
            if tensor.dtype not in (torch.float16, torch.float32, torch.float64):
                return None  # never silently downcast decoded output
            size = tensor.numel() * tensor.element_size()
            if not all(ref() is not None for ref in guards):
                return None
            if key in self.entries:
                self._drop(key)
            while len(self.entries) >= self.max_entries:
                self._drop(next(iter(self.entries)))
            use_ram = self.mode == "RAM" or (stream == "audio" and size <= 16*MiB)
            if use_ram and self._ram_room(size):
                stored = tensor.detach().clone(memory_format=torch.contiguous_format)
                self.entries[key] = Entry(tuple(stored.shape), stored.numpy().dtype.str,
                    tensor_digest(stored, self.interrupt), size, guards, deepcopy(extra), tensor=stored)
                return None  # original MISS output is independently owned
            if self.mode != "Auto" or self.disk_limit == 0 or size+4096 > self.disk_limit:
                return None
            if not self._disk_room(size+4096):
                return None
            root = self._directory()
            filename = uuid.uuid4().hex
            temp = root / (filename + ".tmp")
            final = root / (filename + ".npy")
            dtype = {torch.float16: np.dtype("float16"), torch.float32: np.dtype("float32"),
                     torch.float64: np.dtype("float64")}[tensor.dtype]
            h = hashlib.sha256()
            try:
                with open(temp, "xb") as f:
                    np.lib.format.write_array_header_1_0(f, {
                        "descr": dtype.str, "fortran_order": False, "shape": tuple(tensor.shape)})
                    for block in tensor_blocks(tensor):
                        self.interrupt()
                        h.update(block)
                        if f.write(block) != len(block):
                            raise OSError("short cache write")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp, final)
                entry = Entry(tuple(tensor.shape), dtype.str, h.hexdigest(), size, guards,
                              deepcopy(extra), path=final, file_bytes=final.stat().st_size)
                self.entries[key] = entry
                self._cleanup_state["entries"][final] = entry
                # Returning a COW mapping releases the large anonymous MISS output
                # after this call rather than keeping an extra private RAM copy.
                return self.get(key)
            except BaseException:
                if key in self.entries:
                    self._drop(key)
                for path in (temp, final):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                raise

    def contains(self, key):
        with self._lock:
            return key in self.entries

    def stats(self):
        with self._lock:
            self._reap()
            return {"entries": len(self.entries), "ram_bytes": self.ram_bytes,
                    "disk_bytes": self.disk_bytes, "pending_delete_files": len(self.retired)}
