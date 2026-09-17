"""Small, content-addressed disk cache for completed Director packets.

The cache deliberately stores only JSON/text outputs.  It never retains model,
processor, tensor, or CUDA objects, so a hit can be served before the
DiffusionGemma runtime is loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DIRECTOR_CACHE_SCHEMA = "dg-director-cache/2"
DIRECTOR_CACHE_KEY_SCHEMA = "dg-director-cache-key/2"
_SMALL_FILE_HASH_LIMIT = 64 * 1024 * 1024
_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def _json_safe(value: Any) -> Any:
    """Convert cache-key inputs to deterministic, JSON-safe values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_safe(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    raise TypeError(
        "Director cache identity contains an unsupported value of type "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    stat = path.stat()
    identity = (str(path.resolve(strict=False)), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _FILE_HASH_CACHE.get(identity)
    if cached:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[identity] = value
    return value


def _file_identity(path: Path, *, content_hash: bool) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as exc:
        return {"name": path.name, "missing": True, "error": str(exc)}
    result: dict[str, Any] = {
        "name": path.name,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_hash and stat.st_size <= _SMALL_FILE_HASH_LIMIT:
        try:
            result["sha256"] = _sha256_file(path)
        except OSError as exc:
            result["hash_error"] = str(exc)
    return result


def checkpoint_identity(model_path: str, status: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a cheap checkpoint identity without hashing model-sized shards."""

    path = Path(str(model_path or "")).expanduser()
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = path.absolute()
    result: dict[str, Any] = {
        "resolved_path": str(resolved),
        "revision": str((status or {}).get("_commit_hash", "") or ""),
    }
    if not resolved.exists():
        result["missing"] = True
        return result
    if resolved.is_file():
        result["kind"] = "file"
        # Hash ordinary config/checkpoint sidecars, but identify multi-gigabyte
        # GGUF/model files by resolved path plus high-resolution file metadata.
        result["file"] = _file_identity(
            resolved,
            content_hash=resolved.stat().st_size <= _SMALL_FILE_HASH_LIMIT,
        )
        return result

    result["kind"] = "directory"
    identity_names = (
        "config.json",
        "chat_template.jinja",
        "generation_config.json",
        "hf_quant_config.json",
        "added_tokens.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "model.safetensors.index.json",
    )
    result["identity_files"] = [
        _file_identity(resolved / name, content_hash=True)
        for name in identity_names
        if (resolved / name).is_file()
    ]

    shard_names: set[str] = set()
    index_path = resolved / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if isinstance(weight_map, dict):
                shard_names.update(
                    str(name)
                    for name in weight_map.values()
                    if isinstance(name, str) and name
                )
        except (OSError, ValueError):
            pass
    if not shard_names:
        shard_names.update(path.name for path in resolved.glob("*.safetensors"))
    result["weight_files"] = [
        _file_identity(resolved / name, content_hash=False)
        for name in sorted(shard_names)
    ]
    return result


def implementation_identity(root: Path, relative_paths: Sequence[str]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in relative_paths:
        path = root / relative
        record = {"path": relative, **_file_identity(path, content_hash=True)}
        files.append(record)
    return {"files": files, "sha256": canonical_sha256(files)}


def ordered_tensor_content_hashes(images: Any) -> list[dict[str, Any]]:
    """Hash the exact ordered tensor samples delivered to the processor."""

    if images is None:
        return []
    whole_value_is_one_sample = False
    try:
        image_shape = tuple(int(value) for value in images.shape)
        whole_value_is_one_sample = len(image_shape) == 3
        sample_count = 1 if whole_value_is_one_sample else int(image_shape[0])
    except Exception:
        try:
            sample_count = len(images)
        except Exception:
            sample_count = 1

    hashes: list[dict[str, Any]] = []
    for index in range(max(0, sample_count)):
        digest = hashlib.sha256()
        try:
            sample = images if whole_value_is_one_sample else images[index]
            if hasattr(sample, "detach"):
                sample = sample.detach().cpu().contiguous()
            shape = [int(value) for value in getattr(sample, "shape", ())]
            dtype = str(getattr(sample, "dtype", type(sample).__name__))
            if hasattr(sample, "numpy"):
                array = sample.numpy()
                digest.update(memoryview(array).cast("B"))
            else:
                digest.update(repr(sample).encode("utf-8"))
            hashes.append(
                {
                    "position": index,
                    "shape": shape,
                    "dtype": dtype,
                    "sha256": digest.hexdigest(),
                }
            )
        except Exception as exc:
            digest.update(f"hash-error:{type(exc).__name__}:{exc}".encode("utf-8"))
            hashes.append(
                {
                    "position": index,
                    "hash_error": str(exc),
                    "sha256": digest.hexdigest(),
                }
            )
    return hashes


def cache_root() -> Path:
    override = os.environ.get("DG_DIRECTOR_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve(strict=False)
    try:
        import folder_paths

        return (
            Path(folder_paths.get_system_user_directory("cache")).resolve(strict=False)
            / "diffusiongemma_prompt_builder"
            / "director_cache"
        )
    except Exception:
        return (
            Path(__file__).resolve().parents[2]
            / "user"
            / "__cache"
            / "diffusiongemma_prompt_builder"
            / "director_cache"
        )


def cache_path(cache_key: str) -> Path:
    normalized = str(cache_key or "").strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("Director cache key must be a 64-character SHA-256 digest.")
    return cache_root() / normalized[:2] / f"{normalized}.json"


def load_cache_entry(cache_key: str) -> tuple[dict[str, Any] | None, str]:
    path = cache_path(cache_key)
    if not path.is_file():
        return None, "not_found"
    try:
        maximum_bytes = max(
            1024,
            int(os.environ.get("DG_DIRECTOR_CACHE_MAX_BYTES", str(16 * 1024 * 1024))),
        )
        if path.stat().st_size > maximum_bytes:
            return None, "entry_too_large"
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"read_error:{type(exc).__name__}:{exc}"
    if not isinstance(payload, dict):
        return None, "invalid_root"
    if payload.get("schema") != DIRECTOR_CACHE_SCHEMA:
        return None, "schema_mismatch"
    if payload.get("cache_key") != cache_key:
        return None, "key_mismatch"
    if not isinstance(payload.get("outputs"), dict):
        return None, "outputs_missing"
    expected_checksum = str(payload.get("outputs_sha256", "") or "")
    if len(expected_checksum) != 64 or expected_checksum != canonical_sha256(payload["outputs"]):
        return None, "outputs_checksum_mismatch"
    return payload, "hit"


def save_cache_entry(cache_key: str, payload: Mapping[str, Any]) -> Path:
    path = cache_path(cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(payload)
    entry["schema"] = DIRECTOR_CACHE_SCHEMA
    entry["cache_key"] = cache_key
    entry["outputs_sha256"] = canonical_sha256(entry.get("outputs", {}))
    encoded = json.dumps(entry, indent=2, sort_keys=True, ensure_ascii=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{cache_key}.",
            suffix=".partial",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            try:
                partial = Path(temporary_name)
                if partial.exists():
                    partial.unlink()
            except OSError:
                pass
    return path


__all__ = [
    "DIRECTOR_CACHE_KEY_SCHEMA",
    "DIRECTOR_CACHE_SCHEMA",
    "cache_path",
    "cache_root",
    "canonical_sha256",
    "checkpoint_identity",
    "implementation_identity",
    "load_cache_entry",
    "ordered_tensor_content_hashes",
    "save_cache_entry",
]
