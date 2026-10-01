import asyncio
import json
import os
from pathlib import Path
import re
import threading
import uuid

from safetensors import safe_open
from safetensors.torch import save_file

import folder_paths
from comfy_execution.cache_provider import CacheProvider, CacheValue


class PosterLayerCache(CacheProvider):
    """Exact RGBA recovery for poster assets evicted by ComfyUI's RAM cache."""

    def __init__(self, directory=None, maximum_bytes=1024**3):
        self.directory = Path(directory) if directory is not None else Path(folder_paths.get_user_directory()) / "__cache" / "fl_poster_layers_v1"
        self.maximum_bytes = maximum_bytes
        self.lock = threading.Lock()

    def should_cache(self, context, value=None):
        return context.class_type == "FL_PosterLayerAsset" and re.fullmatch(r"[0-9a-f]{64}", context.cache_key_hash) is not None

    async def on_lookup(self, context):
        if not self.should_cache(context):
            return None
        return await asyncio.to_thread(self._lookup, context.cache_key_hash)

    def _lookup(self, key):
        path = self.directory / (key + ".safetensors")
        with self.lock:
            if not path.is_file():
                return None
            with safe_open(path, framework="pt", device="cpu") as data:
                asset = json.loads(data.metadata()["asset"])
                # Release the file mapping so Windows can evict old cache files.
                asset["image"] = data.get_tensor("image").clone()
            os.utime(path, None)
        return CacheValue(outputs=[[asset]])

    async def on_store(self, context, value):
        if not self.should_cache(context):
            return
        # ComfyUI caches a list of mapped values for each output socket.
        asset = value.outputs[0][0]
        await asyncio.to_thread(self._store, context.cache_key_hash, asset)

    def _store(self, key, asset):
        with self.lock:
            self._write(key, asset)

    def _write(self, key, asset):
        self.directory.mkdir(parents=True, exist_ok=True)
        image = asset["image"].detach().cpu().contiguous()
        if image.numel() * image.element_size() > self.maximum_bytes:
            return
        metadata = {k: v for k, v in asset.items() if k != "image"}
        target = self.directory / (key + ".safetensors")
        temporary = self.directory / (key + "." + uuid.uuid4().hex + ".tmp")
        save_file({"image": image}, str(temporary), metadata={"asset": json.dumps(metadata)})
        os.replace(temporary, target)
        files = sorted((p for p in self.directory.iterdir() if re.fullmatch(r"[0-9a-f]{64}\.safetensors", p.name)), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        for path in files:
            if total <= self.maximum_bytes:
                break
            if path == target:
                continue
            size = path.stat().st_size
            path.unlink()
            total -= size
