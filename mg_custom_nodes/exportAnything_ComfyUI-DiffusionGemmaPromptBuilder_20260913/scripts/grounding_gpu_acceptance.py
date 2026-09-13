#!/usr/bin/env python3
# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

"""Offline GPU acceptance tiers for the DiffusionGemma Grounding Guard.

This runner is intentionally opt-in.  It never downloads a model, never
modifies Transformers or ModelOpt files, and prints one machine-readable JSON
document to stdout.  Tier failures affect the process exit code only when the
tier is named with ``--require``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterator, Sequence


SCHEMA_ID = "dg-grounding-gpu-acceptance/1"
TIER_NAMES = {
    1: "processor_transport_and_counterfactual_shapes",
    2: "nvfp4_four_step_telemetry_smoke",
    3: "telemetry_equivalence_and_overhead",
}
FIXED_PAIR_SEEDS = (0, 1, 42, 1337, 20260807)
MAX_TELEMETRY_OVERHEAD_PERCENT = 10.0
MAX_EXTRA_PEAK_VRAM_BYTES = 512 * 1024 * 1024

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMFY_APP_ROOT = REPOSITORY_ROOT.parents[1]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local DiffusionGemma Grounding Guard GPU acceptance tiers. "
            "The default tier is processor-only and report-only."
        )
    )
    parser.add_argument(
        "model_path",
        help="Local DiffusionGemma Hugging Face checkpoint directory; URLs are not accepted.",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        type=int,
        choices=tuple(TIER_NAMES),
        default=[1],
        help="Tiers to execute (default: 1).",
    )
    parser.add_argument(
        "--require",
        nargs="*",
        type=int,
        choices=tuple(TIER_NAMES),
        default=[],
        help="Tiers whose failure should return a nonzero exit code.",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=20.0,
        help="Maximum CUDA memory supplied to the existing NVFP4 bridge (default: 20.0).",
    )
    return parser.parse_args(argv)


def _normalize_tiers(
    selected: Sequence[int], required: Sequence[int]
) -> tuple[list[int], list[int]]:
    required_values = sorted({int(value) for value in required})
    selected_values = sorted({int(value) for value in selected} | set(required_values))
    invalid = [value for value in selected_values + required_values if value not in TIER_NAMES]
    if invalid:
        raise ValueError(f"unknown acceptance tier(s): {sorted(set(invalid))}")
    return selected_values, required_values


def _load_nodes_module() -> Any:
    # Direct-file import keeps this script usable from the repository root and
    # from ComfyUI's virtual environment without importing ComfyUI's server.
    for path in (REPOSITORY_ROOT, COMFY_APP_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module_name = "_diffusiongemma_gpu_acceptance_nodes"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, REPOSITORY_ROOT / "nodes.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the repository's nodes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


@contextmanager
def _offline_environment() -> Iterator[None]:
    updates = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DG_PROGRESS": "0",
        "TOKENIZERS_PARALLELISM": "false",
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _four_step_sampling_kwargs(_profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Harness-local version of the native profile with exactly four steps."""

    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        EntropyBoundSamplerConfig,
    )

    settings = {
        "profile": "gpu_acceptance_4_step",
        "max_denoising_steps": 4,
        "temperature_start": 0.8,
        "temperature_end": 0.4,
        "entropy_bound": 0.1,
        "stability_threshold": None,
        "confidence_threshold": None,
        "adaptive_stopping": False,
    }
    kwargs = {
        "max_denoising_steps": 4,
        "t_min": 0.4,
        "t_max": 0.8,
        "sampler_config": EntropyBoundSamplerConfig(entropy_bound=0.1),
        "stability_threshold": None,
        "confidence_threshold": None,
    }
    return kwargs, settings


@contextmanager
def _patched_nodes_runtime(dg_nodes: Any, processor: Any, model: Any) -> Iterator[None]:
    """Reuse one loaded runtime and scope the four-step override to this process."""

    original_loader = dg_nodes._load_transformers_model
    original_sampling = dg_nodes._guarded_sampling_kwargs
    dg_nodes._load_transformers_model = lambda _config: (processor, model)
    dg_nodes._guarded_sampling_kwargs = _four_step_sampling_kwargs
    try:
        yield
    finally:
        dg_nodes._guarded_sampling_kwargs = original_sampling
        dg_nodes._load_transformers_model = original_loader


def _model_path_metadata(dg_nodes: Any, model_path: Path) -> dict[str, Any]:
    try:
        raw = dg_nodes._model_path_info(str(model_path))
    except Exception as exc:
        return {"path_kind": "unknown", "error": str(exc), "is_nvfp4": False}
    info = dict(raw) if isinstance(raw, dict) else {}
    info["is_nvfp4"] = bool(
        str(info.get("path_kind", "")).lower() == "nvfp4_hf_repo"
        or str(info.get("quant_algo", "")).upper() == "NVFP4"
    )
    # Keep the acceptance report compact and avoid exposing unrelated metadata.
    allowed = (
        "path_kind",
        "model_type",
        "architecture",
        "quant_algo",
        "quant_method",
        "quant_producer",
        "quant_producer_version",
        "transformers_version_required",
        "has_generation_config",
        "has_modelopt_state",
        "runtime_hint",
        "is_nvfp4",
        "error",
    )
    return {key: info[key] for key in allowed if key in info}


def _environment_info() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "network_policy": "offline_local_files_only",
    }
    try:
        import torch

        result.update(
            {
                "torch": str(torch.__version__),
                "cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda": str(torch.version.cuda or ""),
                "cuda_device_count": int(torch.cuda.device_count()),
            }
        )
        if torch.cuda.is_available():
            result["cuda_device"] = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception as exc:
        result.update({"cuda_available": False, "torch_error": str(exc)})
    try:
        import transformers

        result["transformers"] = str(transformers.__version__)
    except Exception as exc:
        result["transformers_error"] = str(exc)
    return result


def _runtime_config(dg_nodes: Any, model_path: Path, max_memory_gb: float) -> Any:
    return dg_nodes.RuntimeConfig(
        model_path=str(model_path),
        backend="transformers_inprocess",
        dtype="auto",
        quantization="modelopt_nvfp4",
        local_files_only=True,
        unload_policy="keep_loaded",
        max_memory_gb=float(max_memory_gb),
        status={
            "ready": True,
            "supports_pixels": True,
            "acceptance_harness": True,
            "network_policy": "local_files_only",
        },
    )


def _synthetic_images(torch_module: Any, frame_count: int, variant: str) -> Any:
    """Return deterministic ComfyUI IMAGE tensors without reading or writing media."""

    torch = torch_module
    count = max(1, int(frame_count))
    axis = torch.linspace(0.0, 1.0, 64, dtype=torch.float32)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    frames = []
    for index in range(count):
        phase = float(index) / max(1, count - 1)
        if variant == "counterfactual":
            red = (1.0 - x) * (0.55 + 0.45 * phase)
            green = torch.remainder(y + 0.31 + phase * 0.23, 1.0)
            blue = (x < (0.25 + phase * 0.5)).to(dtype=torch.float32)
        else:
            red = torch.remainder(x + phase * 0.17, 1.0)
            green = y * (0.55 + 0.45 * phase)
            blue = torch.clamp((x + y + phase) / 2.5, 0.0, 1.0)
        frames.append(torch.stack((red, green, blue), dim=-1))
    return torch.stack(frames, dim=0).contiguous()


def _media_metadata(kind: str, frame_count: int) -> dict[str, Any]:
    if kind == "image":
        return {
            "source": "image",
            "reference_image_count": int(frame_count),
            "image_role": "synthetic_transport_probe",
            "width": 64,
            "height": 64,
        }
    return {
        "source": "video",
        "duration_seconds": float(max(1, frame_count - 1)),
        "sample_fps": 1.0,
        "source_fps": 1.0,
        "sampled_frame_count": int(frame_count),
        "video_sampled_frame_count": int(frame_count),
        "sampled_indices": list(range(frame_count)),
        "sampled_timecodes_seconds": [float(index) for index in range(frame_count)],
        "transformers_video_transport": "video_tokens",
        "width": 64,
        "height": 64,
    }


def _tensor_summary_and_hashes(inputs: Any) -> tuple[dict[str, Any], dict[str, str]]:
    shapes: dict[str, Any] = {}
    pixel_hashes: dict[str, str] = {}
    keys = list(inputs.keys()) if hasattr(inputs, "keys") else []
    for raw_key in keys:
        key = str(raw_key)
        try:
            value = inputs[raw_key]
        except Exception:
            continue
        shape = getattr(value, "shape", None)
        if shape is None:
            continue
        shapes[key] = {
            "shape": [int(item) for item in shape],
            "dtype": str(getattr(value, "dtype", "")),
        }
        if "pixel" not in key.lower():
            continue
        try:
            cpu_value = value.detach().to(device="cpu").contiguous()
            byte_view = cpu_value.view(dtype=__import__("torch").uint8)
            digest = hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()
            pixel_hashes[key] = digest
            del byte_view, cpu_value
        except Exception as exc:
            pixel_hashes[key] = f"hash_error:{exc}"
    return shapes, pixel_hashes


def _process_synthetic_case(
    dg_nodes: Any,
    processor: Any,
    model_or_stub: Any,
    *,
    kind: str,
    images: Any,
) -> dict[str, Any]:
    frame_count = int(images.shape[0])
    metadata = _media_metadata(kind, frame_count)
    context = dg_nodes.MediaContext(images=images, source=kind, metadata=metadata)
    messages, processor_kwargs = dg_nodes._transformers_messages_and_processor_kwargs(
        "Report the directly visible synthetic colors and geometry.",
        context,
        visual_first=True,
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        processor_kwargs=processor_kwargs,
        enable_thinking=False,
    )
    registry = dg_nodes.build_asset_registry(metadata, frame_count)
    proof = dg_nodes._processor_transport_proof(
        inputs, model_or_stub, processor, messages, registry
    )
    tensor_summary, pixel_hashes = _tensor_summary_and_hashes(inputs)
    result = {
        "input_keys": sorted(str(key) for key in inputs.keys()),
        "tensor_summary": tensor_summary,
        "pixel_hashes": pixel_hashes,
        "transport_proof": proof,
    }
    del inputs
    return result


def _compare_same_shape_cases(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    first_hashes = first.get("pixel_hashes", {})
    second_hashes = second.get("pixel_hashes", {})
    common_hash_keys = sorted(set(first_hashes) & set(second_hashes))
    content_differs = bool(common_hash_keys) and any(
        first_hashes[key] != second_hashes[key] for key in common_hash_keys
    )
    input_keys_identical = first.get("input_keys") == second.get("input_keys")
    tensor_shapes_identical = first.get("tensor_summary") == second.get("tensor_summary")
    transport_confirmed = bool(
        first.get("transport_proof", {}).get("pixel_transport_confirmed")
        and second.get("transport_proof", {}).get("pixel_transport_confirmed")
    )
    return {
        "passed": bool(
            input_keys_identical
            and tensor_shapes_identical
            and content_differs
            and transport_confirmed
        ),
        "input_keys_identical": input_keys_identical,
        "tensor_shapes_identical": tensor_shapes_identical,
        "pixel_content_hashes_differ": content_differs,
        "transport_confirmed_for_both": transport_confirmed,
        "common_pixel_keys": common_hash_keys,
    }


def _run_tier1(
    dg_nodes: Any,
    model_path: Path,
    *,
    processor: Any | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "name": TIER_NAMES[1],
        "status": "fail",
        "passed": False,
        "processor_source": "loaded_gpu_runtime" if processor is not None else "processor_only",
    }
    try:
        import torch

        if processor is None:
            from transformers import AutoConfig, AutoProcessor

            processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
            config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
            model_or_stub = SimpleNamespace(config=config)
        else:
            model_or_stub = model

        image_primary = _process_synthetic_case(
            dg_nodes,
            processor,
            model_or_stub,
            kind="image",
            images=_synthetic_images(torch, 1, "primary"),
        )
        image_counterfactual = _process_synthetic_case(
            dg_nodes,
            processor,
            model_or_stub,
            kind="image",
            images=_synthetic_images(torch, 1, "counterfactual"),
        )
        video_primary = _process_synthetic_case(
            dg_nodes,
            processor,
            model_or_stub,
            kind="video",
            images=_synthetic_images(torch, 3, "primary"),
        )
        video_counterfactual = _process_synthetic_case(
            dg_nodes,
            processor,
            model_or_stub,
            kind="video",
            images=_synthetic_images(torch, 3, "counterfactual"),
        )
        video_shorter = _process_synthetic_case(
            dg_nodes,
            processor,
            model_or_stub,
            kind="video",
            images=_synthetic_images(torch, 2, "primary"),
        )

        image_comparison = _compare_same_shape_cases(image_primary, image_counterfactual)
        video_comparison = _compare_same_shape_cases(video_primary, video_counterfactual)
        frame_count_shape_changed = bool(
            video_primary.get("input_keys") == video_shorter.get("input_keys")
            and video_primary.get("tensor_summary") != video_shorter.get("tensor_summary")
            and video_shorter.get("transport_proof", {}).get("pixel_transport_confirmed")
            and video_shorter.get("transport_proof", {}).get("processor_observed_sample_count") == 2
        )
        passed = bool(
            image_comparison.get("passed")
            and video_comparison.get("passed")
            and frame_count_shape_changed
        )
        result.update(
            {
                "status": "pass" if passed else "fail",
                "passed": passed,
                "image_same_shape_counterfactual": image_comparison,
                "video_same_shape_counterfactual": video_comparison,
                "video_frame_count_counterfactual": {
                    "passed": frame_count_shape_changed,
                    "three_frame_pixel_tensors": video_primary.get("transport_proof", {}).get(
                        "pixel_tensors", {}
                    ),
                    "two_frame_pixel_tensors": video_shorter.get("transport_proof", {}).get(
                        "pixel_tensors", {}
                    ),
                },
                "image_transport_proof": image_primary.get("transport_proof", {}),
                "video_transport_proof": video_primary.get("transport_proof", {}),
                "counterfactual_pixel_hashes": {
                    "image_primary": image_primary.get("pixel_hashes", {}),
                    "image_counterfactual": image_counterfactual.get("pixel_hashes", {}),
                    "video_primary": video_primary.get("pixel_hashes", {}),
                    "video_counterfactual": video_counterfactual.get("pixel_hashes", {}),
                },
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    return result


def _cuda_device(torch_module: Any, model: Any) -> Any:
    torch = torch_module
    model_device = getattr(model, "device", None)
    try:
        candidate = torch.device(model_device)
        if candidate.type == "cuda":
            return candidate
    except Exception:
        pass
    return torch.device("cuda", torch.cuda.current_device())


@contextmanager
def _capture_generated_token_ids(model: Any, holder: dict[str, Any]) -> Iterator[None]:
    """Capture plain generated IDs without retaining the output tensor."""

    original_generate = model.generate
    instance_dict = getattr(model, "__dict__", {})
    had_instance_attribute = isinstance(instance_dict, dict) and "generate" in instance_dict

    def wrapped_generate(*args: Any, **kwargs: Any) -> Any:
        generated = original_generate(*args, **kwargs)
        try:
            sequences = getattr(generated, "sequences", generated)
            input_ids = kwargs.get("input_ids")
            input_length = int(input_ids.shape[-1]) if input_ids is not None else 0
            sequence_length = int(sequences.shape[-1])
            includes_prompt_prefix = False
            if input_ids is not None and input_length > 0 and sequence_length >= input_length:
                try:
                    import torch

                    includes_prompt_prefix = bool(
                        torch.equal(sequences[:, :input_length], input_ids)
                    )
                except Exception:
                    includes_prompt_prefix = False
            generated_slice = (
                sequences[:, input_length:]
                if includes_prompt_prefix
                else sequences
            )
            holder["token_ids"] = generated_slice.detach().to(device="cpu").tolist()
            del generated_slice
        except Exception as exc:
            holder["capture_error"] = f"{type(exc).__name__}: {exc}"
        return generated

    setattr(model, "generate", wrapped_generate)
    try:
        yield
    finally:
        if had_instance_attribute:
            setattr(model, "generate", original_generate)
        else:
            try:
                delattr(model, "generate")
            except Exception:
                setattr(model, "generate", original_generate)


def _token_signature(token_ids: Any) -> dict[str, Any]:
    batches = token_ids if isinstance(token_ids, list) else []
    normalized: list[list[int]] = []
    for batch in batches:
        if isinstance(batch, list):
            normalized.append([int(value) for value in batch])
    encoded = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    flattened = [value for batch in normalized for value in batch]
    return {
        "batch_count": len(normalized),
        "token_count": len(flattened),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "first_token_ids": flattened[:8],
        "last_token_ids": flattened[-8:] if flattened else [],
    }


def _text_signature(text: str) -> dict[str, Any]:
    encoded = str(text).encode("utf-8")
    return {
        "character_count": len(str(text)),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _compact_telemetry(value: dict[str, Any]) -> dict[str, Any]:
    telemetry = value if isinstance(value, dict) else {}
    final_output = telemetry.get("final_output", {})
    return {
        "schema_version": telemetry.get("schema_version", ""),
        "max_denoising_steps": telemetry.get("max_denoising_steps"),
        "selected_steps": telemetry.get("selected_steps", []),
        "forward_count": int(telemetry.get("forward_count", 0) or 0),
        "canvas_forward_counts": telemetry.get("canvas_forward_counts", {}),
        "final_token_count": final_output.get("token_count", []),
        "refusal": telemetry.get("refusal", {}),
        "errors": telemetry.get("errors", []),
        "truncated": bool(telemetry.get("truncated", False)),
    }


def _public_run_record(record: dict[str, Any], *, include_transport: bool = False) -> dict[str, Any]:
    transport = record.get("transport", {})
    result = {
        "telemetry_enabled": bool(record.get("telemetry_enabled")),
        "seed": int(record.get("seed", 0)),
        "wall_seconds": record.get("wall_seconds"),
        "generation_seconds": record.get("generation_seconds"),
        "peak_vram_allocated_bytes": int(record.get("peak_vram_allocated_bytes", 0) or 0),
        "peak_vram_reserved_bytes": int(record.get("peak_vram_reserved_bytes", 0) or 0),
        "token_signature": _token_signature(record.get("_token_ids")),
        "text_signature": _text_signature(str(record.get("_decoded_text", ""))),
        "capture_error": str(record.get("capture_error", "")),
        "pixel_transport_confirmed": bool(transport.get("pixel_transport_confirmed")),
        "effective_sampling": record.get("effective_sampling", {}),
        "telemetry": _compact_telemetry(record.get("telemetry", {})),
    }
    if include_transport:
        result["transport"] = transport
    return result


def _run_generation_once(
    dg_nodes: Any,
    config: Any,
    processor: Any,
    model: Any,
    *,
    seed: int,
    telemetry_enabled: bool,
) -> dict[str, Any]:
    import torch

    images = _synthetic_images(torch, 1, "primary")
    metadata = _media_metadata("image", 1)
    context = dg_nodes.MediaContext(images=images, source="image", metadata=metadata)
    registry = dg_nodes.build_asset_registry(metadata, 1)
    options = dg_nodes.BackendRunOptions(
        sampling_profile="full_48_diagnostic",
        thinking_mode="off",
        seed=int(seed),
        enable_telemetry=bool(telemetry_enabled),
        require_transport=True,
        fail_closed_telemetry=bool(telemetry_enabled),
        visual_first=True,
        asset_registry=registry,
        stage="gpu_acceptance",
    )
    canvas_length = int(getattr(getattr(model, "config", None), "canvas_length", 256) or 256)
    device = _cuda_device(torch, model)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    capture: dict[str, Any] = {}
    started = time.perf_counter()
    with _capture_generated_token_ids(model, capture):
        backend_result = dg_nodes._run_transformers_detailed(
            config,
            (
                "Inspect the synthetic image and return a concise factual description. "
                "Do not invent objects that are not visible."
            ),
            context,
            canvas_length,
            options,
        )
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "seed": int(seed),
        "telemetry_enabled": bool(telemetry_enabled),
        "wall_seconds": round(wall_seconds, 6),
        "generation_seconds": float(backend_result.timing.get("generation_seconds", 0.0) or 0.0),
        "peak_vram_allocated_bytes": peak_allocated,
        "peak_vram_reserved_bytes": peak_reserved,
        "_token_ids": capture.get("token_ids", []),
        "_decoded_text": backend_result.decoded_text,
        "capture_error": capture.get("capture_error", ""),
        "transport": backend_result.transport,
        "telemetry": backend_result.telemetry,
        "effective_sampling": backend_result.effective_sampling,
        "forward_counts": backend_result.forward_counts,
        "canvas_length": canvas_length,
    }


def _telemetry_run_valid(record: dict[str, Any]) -> bool:
    telemetry = record.get("telemetry", {})
    canvas_counts = telemetry.get("canvas_forward_counts", {}) if isinstance(telemetry, dict) else {}
    return bool(
        record.get("telemetry_enabled")
        and record.get("transport", {}).get("pixel_transport_confirmed")
        and not record.get("capture_error")
        and _token_signature(record.get("_token_ids")).get("token_count", 0) > 0
        and record.get("effective_sampling", {}).get("max_denoising_steps") == 4
        and record.get("effective_sampling", {}).get("adaptive_stopping") is False
        and int(telemetry.get("forward_count", 0) or 0) == 4
        and len(canvas_counts) == 1
        and list(canvas_counts.values()) == [4]
        and not telemetry.get("errors")
    )


def _run_tier2(
    dg_nodes: Any, config: Any, processor: Any, model: Any, model_info: dict[str, Any]
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "name": TIER_NAMES[2],
        "status": "fail",
        "passed": False,
        "model_is_nvfp4": bool(model_info.get("is_nvfp4")),
    }
    try:
        run = _run_generation_once(
            dg_nodes,
            config,
            processor,
            model,
            seed=FIXED_PAIR_SEEDS[0],
            telemetry_enabled=True,
        )
        passed = bool(model_info.get("is_nvfp4") and _telemetry_run_valid(run))
        result.update(
            {
                "status": "pass" if passed else "fail",
                "passed": passed,
                "four_step_one_canvas_confirmed": _telemetry_run_valid(run),
                "run": _public_run_record(run, include_transport=True),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    return result


def _evaluate_pair_records(pair_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    overhead_values: list[float] = []
    extra_vram_values: list[int] = []
    all_identical = len(pair_records) == len(FIXED_PAIR_SEEDS)
    all_telemetry_valid = len(pair_records) == len(FIXED_PAIR_SEEDS)
    pair_summaries: list[dict[str, Any]] = []

    for pair in pair_records:
        disabled = pair.get("disabled", {})
        enabled = pair.get("enabled", {})
        # Compare the complete guarded call, including telemetry setup,
        # summary, transport validation, and decode—not just model.generate.
        disabled_seconds = float(disabled.get("wall_seconds", 0.0) or 0.0)
        enabled_seconds = float(enabled.get("wall_seconds", 0.0) or 0.0)
        if disabled_seconds > 0.0:
            overhead = ((enabled_seconds - disabled_seconds) / disabled_seconds) * 100.0
            overhead_values.append(overhead)
        else:
            overhead = None
            all_identical = False

        signed_vram_delta = int(enabled.get("peak_vram_allocated_bytes", 0) or 0) - int(
            disabled.get("peak_vram_allocated_bytes", 0) or 0
        )
        extra_vram = max(0, signed_vram_delta)
        extra_vram_values.append(extra_vram)
        token_ids_identical = disabled.get("_token_ids") == enabled.get("_token_ids")
        decoded_text_identical = disabled.get("_decoded_text") == enabled.get("_decoded_text")
        telemetry_valid = _telemetry_run_valid(enabled)
        all_identical = all_identical and token_ids_identical and decoded_text_identical
        all_telemetry_valid = all_telemetry_valid and telemetry_valid
        pair_summaries.append(
            {
                "seed": int(pair.get("seed", 0)),
                "execution_order": pair.get("execution_order", []),
                "token_ids_identical": token_ids_identical,
                "decoded_text_identical": decoded_text_identical,
                "telemetry_run_valid": telemetry_valid,
                "telemetry_overhead_percent": round(overhead, 6) if overhead is not None else None,
                "signed_peak_vram_delta_bytes": signed_vram_delta,
                "extra_peak_vram_bytes": extra_vram,
                "disabled": _public_run_record(disabled),
                "enabled": _public_run_record(enabled),
            }
        )

    median_overhead = statistics.median(overhead_values) if len(overhead_values) == len(FIXED_PAIR_SEEDS) else None
    max_extra_vram = max(extra_vram_values) if len(extra_vram_values) == len(FIXED_PAIR_SEEDS) else None
    overhead_passed = bool(
        median_overhead is not None and median_overhead <= MAX_TELEMETRY_OVERHEAD_PERCENT
    )
    vram_passed = bool(
        max_extra_vram is not None and max_extra_vram <= MAX_EXTRA_PEAK_VRAM_BYTES
    )
    passed = bool(all_identical and all_telemetry_valid and overhead_passed and vram_passed)
    return {
        "passed": passed,
        "pair_count": len(pair_records),
        "fixed_seeds": list(FIXED_PAIR_SEEDS),
        "all_token_ids_and_text_identical": all_identical,
        "all_telemetry_runs_valid": all_telemetry_valid,
        "median_telemetry_overhead_percent": (
            round(median_overhead, 6) if median_overhead is not None else None
        ),
        "maximum_extra_peak_vram_bytes": max_extra_vram,
        "thresholds": {
            "median_telemetry_overhead_percent_max": MAX_TELEMETRY_OVERHEAD_PERCENT,
            "extra_peak_vram_bytes_max": MAX_EXTRA_PEAK_VRAM_BYTES,
        },
        "overhead_passed": overhead_passed,
        "vram_passed": vram_passed,
        "pairs": pair_summaries,
    }


def _run_tier3(dg_nodes: Any, config: Any, processor: Any, model: Any) -> dict[str, Any]:
    started = time.perf_counter()
    pair_records: list[dict[str, Any]] = []
    error = ""
    for index, seed in enumerate(FIXED_PAIR_SEEDS):
        order = ["disabled", "enabled"] if index % 2 == 0 else ["enabled", "disabled"]
        runs: dict[str, dict[str, Any]] = {}
        try:
            for label in order:
                runs[label] = _run_generation_once(
                    dg_nodes,
                    config,
                    processor,
                    model,
                    seed=seed,
                    telemetry_enabled=label == "enabled",
                )
                gc.collect()
            pair_records.append(
                {
                    "seed": seed,
                    "execution_order": order,
                    "disabled": runs["disabled"],
                    "enabled": runs["enabled"],
                }
            )
        except Exception as exc:
            error = f"seed={seed}:{type(exc).__name__}: {exc}"
            break

    evaluation = _evaluate_pair_records(pair_records)
    result = {
        "name": TIER_NAMES[3],
        "status": "pass" if evaluation.get("passed") else "fail",
        "passed": bool(evaluation.get("passed")),
        **evaluation,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    if error:
        result["error"] = error
    return result


def _not_run_tier(tier: int) -> dict[str, Any]:
    return {
        "name": TIER_NAMES[tier],
        "status": "not_run",
        "passed": False,
    }


def _failed_tier(tier: int, message: str) -> dict[str, Any]:
    return {
        "name": TIER_NAMES[tier],
        "status": "fail",
        "passed": False,
        "error": str(message),
    }


def _finalize_report(report: dict[str, Any], required: Sequence[int]) -> None:
    required_values = [int(value) for value in required]
    failed_required = [
        tier
        for tier in required_values
        if not report.get("tiers", {}).get(str(tier), {}).get("passed", False)
    ]
    selected = [int(value) for value in report.get("requested_tiers", [])]
    failed_selected = [
        tier
        for tier in selected
        if not report.get("tiers", {}).get(str(tier), {}).get("passed", False)
    ]
    report["overall"] = {
        "report_only": not bool(required_values),
        "required_passed": not bool(failed_required),
        "failed_required_tiers": failed_required,
        "selected_passed": not bool(failed_selected),
        "failed_selected_tiers": failed_selected,
    }


def _exit_code(report: dict[str, Any]) -> int:
    return 0 if report.get("overall", {}).get("required_passed", False) else 1


def run_acceptance(args: argparse.Namespace, *, dg_nodes: Any | None = None) -> dict[str, Any]:
    selected, required = _normalize_tiers(args.tiers, args.require)
    model_path = Path(str(args.model_path)).expanduser().resolve()
    report: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "model_path": str(model_path),
        "requested_tiers": selected,
        "required_tiers": required,
        "thresholds": {
            "median_telemetry_overhead_percent_max": MAX_TELEMETRY_OVERHEAD_PERCENT,
            "extra_peak_vram_bytes_max": MAX_EXTRA_PEAK_VRAM_BYTES,
            "denoising_steps": 4,
            "canvas_count": 1,
            "pair_count": len(FIXED_PAIR_SEEDS),
        },
        "environment": _environment_info(),
        "tiers": {str(tier): _not_run_tier(tier) for tier in TIER_NAMES},
        "runtime_lifecycle": {
            "model_load_attempts": 0,
            "model_load_succeeded": False,
            "release_attempts": 0,
            "release_succeeded": False,
        },
        "warnings": [],
    }
    started = time.perf_counter()
    needs_gpu = bool({2, 3} & set(selected))
    processor = None
    model = None
    config = None
    load_error = ""

    if dg_nodes is None:
        dg_nodes = _load_nodes_module()
    model_info = _model_path_metadata(dg_nodes, model_path)
    report["model"] = model_info

    try:
        if needs_gpu:
            if not model_path.is_dir():
                load_error = "model_path must be an existing local checkpoint directory"
            elif not model_info.get("is_nvfp4"):
                load_error = "GPU acceptance tiers require an NVFP4 DiffusionGemma checkpoint"
            else:
                config = _runtime_config(dg_nodes, model_path, float(args.max_memory_gb))
                report["runtime_lifecycle"]["model_load_attempts"] += 1
                try:
                    processor, model = dg_nodes._load_transformers_model(config)
                    report["runtime_lifecycle"]["model_load_succeeded"] = True
                except Exception as exc:
                    load_error = f"{type(exc).__name__}: {exc}"

        if 1 in selected:
            report["tiers"]["1"] = _run_tier1(
                dg_nodes,
                model_path,
                processor=processor,
                model=model,
            )

        if needs_gpu and (processor is None or model is None or config is None):
            for tier in sorted({2, 3} & set(selected)):
                report["tiers"][str(tier)] = _failed_tier(tier, load_error or "runtime load failed")
        elif needs_gpu:
            with _patched_nodes_runtime(dg_nodes, processor, model):
                if 2 in selected:
                    report["tiers"]["2"] = _run_tier2(
                        dg_nodes, config, processor, model, model_info
                    )
                if 3 in selected:
                    report["tiers"]["3"] = _run_tier3(
                        dg_nodes, config, processor, model
                    )
    finally:
        if needs_gpu:
            report["runtime_lifecycle"]["release_attempts"] += 1
            try:
                dg_nodes._release_transformers_runtime()
                report["runtime_lifecycle"]["release_succeeded"] = True
            except Exception as exc:
                report["warnings"].append(f"runtime_release_failed:{type(exc).__name__}:{exc}")

    report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    _finalize_report(report, required)
    return report


def _emergency_report(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    selected, required = _normalize_tiers(args.tiers, args.require)
    report = {
        "schema": SCHEMA_ID,
        "model_path": str(Path(str(args.model_path)).expanduser()),
        "requested_tiers": selected,
        "required_tiers": required,
        "tiers": {
            str(tier): (
                _failed_tier(tier, f"harness_error:{type(exc).__name__}: {exc}")
                if tier in selected
                else _not_run_tier(tier)
            )
            for tier in TIER_NAMES
        },
        "warnings": [],
        "harness_error": f"{type(exc).__name__}: {exc}",
    }
    _finalize_report(report, required)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    # Third-party loaders occasionally print progress to stdout. Route that
    # incidental output to stderr so stdout remains one parseable JSON value.
    with _offline_environment(), redirect_stdout(sys.stderr):
        try:
            report = run_acceptance(args)
        except Exception as exc:  # Always preserve a JSON result after parsing succeeds.
            report = _emergency_report(args, exc)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False))
    return _exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
