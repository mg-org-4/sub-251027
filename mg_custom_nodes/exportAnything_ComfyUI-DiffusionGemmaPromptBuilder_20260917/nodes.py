# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import gc
import hashlib
import json
import logging
import math
import os
import re
import shlex
import subprocess
import threading
import time
import unicodedata
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ModuleNotFoundError:  # Standalone repository tests do not add ComfyUI to sys.path.
    class ExecutionBlocker:  # type: ignore[no-redef]
        def __init__(self, message: str | None) -> None:
            self.message = message

try:
    from .director_cache import (
        DIRECTOR_CACHE_KEY_SCHEMA,
        DIRECTOR_CACHE_SCHEMA,
        cache_path as _director_disk_cache_path,
        canonical_sha256 as _canonical_sha256,
        checkpoint_identity as _checkpoint_identity,
        implementation_identity as _implementation_identity,
        load_cache_entry as _load_director_cache_entry,
        ordered_tensor_content_hashes as _ordered_tensor_content_hashes,
        save_cache_entry as _save_director_cache_entry,
    )
    from .grounding_guard import (
        EVIDENCE_TOKEN_BUDGETS,
        GROUNDING_LEDGER_SCHEMA_ID,
        GROUNDING_REPORT_SCHEMA_ID,
        GroundingGuardConfig,
        GroundingGuardDecision,
        GroundingLedgerError,
        build_asset_registry,
        build_grounding_report,
        compact_grounding_report,
        decide_grounding_guard,
        extract_grounding_ledger,
        normalize_grounding_guard_config,
        persist_grounding_trace,
        validate_grounding_ledger,
    )
    from .grounding_telemetry import build_diffusiongemma_telemetry
    from .ltx25_contract import (
        LTX25_CAMERA_CAPABILITIES,
        LTX25_CONTRACT_SCHEMA,
        LTX25_GENERATION_MODES,
        LTX25_LONG_HORIZON_MODES,
        ltx25_compiler_contract,
        ltx25_complexity_budget,
        ltx25_long_horizon_plan,
        normalize_ltx25_generation_mode,
        normalize_ltx25_camera_capability,
        normalize_ltx25_long_horizon_mode,
        resolve_ltx25_generation_mode,
        validate_ltx25_prompt,
    )
except ImportError:  # Direct-file imports used by the repository smoke tests.
    from director_cache import (
        DIRECTOR_CACHE_KEY_SCHEMA,
        DIRECTOR_CACHE_SCHEMA,
        cache_path as _director_disk_cache_path,
        canonical_sha256 as _canonical_sha256,
        checkpoint_identity as _checkpoint_identity,
        implementation_identity as _implementation_identity,
        load_cache_entry as _load_director_cache_entry,
        ordered_tensor_content_hashes as _ordered_tensor_content_hashes,
        save_cache_entry as _save_director_cache_entry,
    )
    from grounding_guard import (
        EVIDENCE_TOKEN_BUDGETS,
        GROUNDING_LEDGER_SCHEMA_ID,
        GROUNDING_REPORT_SCHEMA_ID,
        GroundingGuardConfig,
        GroundingGuardDecision,
        GroundingLedgerError,
        build_asset_registry,
        build_grounding_report,
        compact_grounding_report,
        decide_grounding_guard,
        extract_grounding_ledger,
        normalize_grounding_guard_config,
        persist_grounding_trace,
        validate_grounding_ledger,
    )
    from grounding_telemetry import build_diffusiongemma_telemetry
    from ltx25_contract import (
        LTX25_CAMERA_CAPABILITIES,
        LTX25_CONTRACT_SCHEMA,
        LTX25_GENERATION_MODES,
        LTX25_LONG_HORIZON_MODES,
        ltx25_compiler_contract,
        ltx25_complexity_budget,
        ltx25_long_horizon_plan,
        normalize_ltx25_generation_mode,
        normalize_ltx25_camera_capability,
        normalize_ltx25_long_horizon_mode,
        resolve_ltx25_generation_mode,
        validate_ltx25_prompt,
    )


CATEGORY = "prompt/diffusiongemma"
COMPATIBILITY_CATEGORY = f"{CATEGORY}/compatibility"
OPTIONAL_CATEGORY = f"{CATEGORY}/optional"
EXPERIMENTAL_MOTION_CATEGORY = f"{CATEGORY}/experimental-motion-planning"
MODEL_TYPE = "DG_MODEL_CONFIG"
MEDIA_TYPE = "DG_MEDIA_CONTEXT"
CONTEXT_TYPE = "DG_CONTEXT"
TARGET_PROFILE_TYPE = "DG_TARGET_PROFILE_CONFIG"
H3_REFERENCE_POLICY_TYPE = "DG_H3_REFERENCE_POLICY_CONFIG"
GROUNDING_GUARD_TYPE = "DG_GROUNDING_GUARD_CONFIG"
GROUNDING_TELEMETRY_SCHEMA_VERSION = "dg-denoising-telemetry/1"
DIRECTOR_RUNTIME_SCHEMA_VERSION = "dg-director-runtime/1"
DIRECTOR_CACHE_MODE_CHOICES = ["reuse", "refresh", "off"]
DIRECTOR_EXECUTION_CONTRACT_REVISION = "h3-reference-policy-v18"

DEFAULT_GGUF_MODEL_PATH = r"Z:\models\LLM\diffusiongemma-26B-A4B-it-GGUF\diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
DEFAULT_GGUF_CLI_PATH = r"C:\tools\llama.cpp-diffusiongemma\build-clang-cuda\bin\llama-diffusion-cli.exe"
DEFAULT_GGUF_EXTRA_ARGS = "-ngl all -cmoe -sm none -mg 0 --diffusion-eb auto --diffusion-kv-cache auto --diffusion-gpu-sampling auto --diffusion-eb-max-steps 24 --diffusion-steps 24 --temp 0.45 -no-cnv --log-verbosity 1"
DEFAULT_CUDA_RUNTIME_BIN = r"C:\tools\cuda-13.3-conda\Library\bin"
DEFAULT_NVFP4_MODEL_DIR = str(Path.cwd() / "models" / "LLM" / "diffusiongemma-26B-A4B-it-NVFP4")
DEFAULT_TRANSFORMERS_MODEL_DIR = str(Path.cwd() / "models" / "LLM" / "diffusiongemma-26B-A4B-it")
DEFAULT_ADVANCED_MODEL_PATH = DEFAULT_NVFP4_MODEL_DIR if Path(DEFAULT_NVFP4_MODEL_DIR).exists() else DEFAULT_TRANSFORMERS_MODEL_DIR
DEFAULT_MODEL_PATH = DEFAULT_GGUF_MODEL_PATH if Path(DEFAULT_GGUF_MODEL_PATH).exists() else DEFAULT_TRANSFORMERS_MODEL_DIR
FINAL_JSON_OPEN = "<|final_json|>"
FINAL_JSON_CLOSE = "<|end_final_json|>"
FINAL_JSON_OPEN_MARKERS = (FINAL_JSON_OPEN, "<|answer|>", "<|final|>")
FINAL_JSON_CLOSE_MARKERS = (FINAL_JSON_CLOSE, "<|/final_json|>", "<|end_final|>", "<|end|>")
MIN_MODELOPT_NVFP4_BRIDGE_MEMORY_GB = 18.0
MEMORY_PRESET_SAFE_20 = "20.0 GB - safe practical"
MEMORY_PRESET_MIN_18 = "18.0 GB - minimum supported"
MEMORY_PRESET_FULL_GPU = "0 GB - full GPU"
MEMORY_PRESET_CHOICES = [MEMORY_PRESET_SAFE_20, MEMORY_PRESET_MIN_18, MEMORY_PRESET_FULL_GPU]
RESOLUTION_SELECTOR_ASPECT_LABELS = {
    "1:1": "1:1 (Square)",
    "2:3": "2:3 (Portrait Photo)",
    "3:2": "3:2 (Photo)",
    "3:4": "3:4 (Portrait Standard)",
    "4:3": "4:3 (Standard)",
    "9:16": "9:16 (Portrait Widescreen)",
    "16:9": "16:9 (Widescreen)",
    "21:9": "21:9 (Ultrawide)",
}
RESOLUTION_SELECTOR_AUTO = "Auto (target/source)"
RESOLUTION_SELECTOR_ASPECT_CHOICES = [
    RESOLUTION_SELECTOR_AUTO,
    *RESOLUTION_SELECTOR_ASPECT_LABELS.values(),
]
RESOLUTION_SELECTOR_RATIOS = {
    "1:1": (1, 1),
    "2:3": (2, 3),
    "3:2": (3, 2),
    "3:4": (3, 4),
    "4:3": (4, 3),
    "9:16": (9, 16),
    "16:9": (16, 9),
    "21:9": (21, 9),
}
REFERENCE_PREP_ASPECT_CHOICES = ["source", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"]
MEDIA_SYNTHESIS_MODE_CHOICES = ["video_recreation", "image_identity_video_control"]
DEFAULT_MASTER_PROMPT = """You are a local ComfyUI prompt architect. Treat the user's text as a content brief: it only needs to say what the video or image should be about. Do not require the user to ask for prompt-writing boilerplate such as detailed subject, background, lighting, camera movement, pacing, composition, or style; infer useful generator-ready detail when it is missing, while preserving every explicit user constraint. Model-specific target rules below override this general expansion rule. Preserve an explicitly requested or evidenced static hold. A conditioned still frame fixes its camera geometry at that instant; it does not by itself prove that the camera remains locked afterward.

Use the user's text, any backend-consumable pixels, and any supplied visual_description to produce final, practical prompts for generation models. If the active backend is metadata-only for image/video, do not pretend you saw pixels; use only the user's text, the visual_description, and factual media metadata. Preserve requested subjects, exact text, composition, motion, style, camera behavior, visual capture language, audio mode, audio guidance, and constraints. In image_identity_video_control workflows, treat a request to replace the video subject with the image subject as a strict identity-and-appearance transfer: face, hair, body type, wardrobe, accessories, styling, and distinguishing traits come from the reference image, while the video contributes pose, action, blocking, depth/canny/edge structure, camera choreography, timing, composition, and scene geometry. Do not copy the control-video subject's wardrobe, hair, face, body type, accessories, or styling into the final prompt unless the user explicitly asks for those video-subject traits. For LTX source-video work, explicitly observe camera choreography before writing the final prompt: first-frame and ending framing, shot size, camera height/angle, camera support or stability, camera path, zoom/dolly/truck/boom/pan/tilt/roll/orbit/rotation, subject-camera relationship, focus behavior, depth/parallax, and visible lens/capture changes. Transcribe observed camera work precisely instead of reducing it to generic phrases like camera movement, zoom out, or pan left. Visual capture language includes exposure level, overexposed or blown highlights, underexposure, flash/direct light, bloom, halation, lens blur, bokeh, depth of field, focal length impression, perspective compression, lens distortion, motion blur, grain/noise, white balance, contrast, dynamic range, and camera angle. Do not describe invisible intent. Faithful mode preserves explicitly requested camera energy. For LTX targets only, that preservation is bounded by the LTX Target Profile's Camera Capability: Stable / base model simplifies unsupported risky camera movement, while Advanced / controlled camera preserves the prior fast or compound choreography policy. Other targets do not inherit this LTX-only capability gate and must follow their own model-specific camera contract. Every prompt field must contain the finished prompt text or finished prompt JSON, never instructions about how to write a prompt. Return only valid JSON with ltx_prompt, ideogram_prompt, minimax_h3_prompt, negative_prompt, scene_segments, and metadata."""

MINIMAX_H3_MASTER_PROMPT = """You are a local ComfyUI MiniMax H3 prompt architect. Treat the user's text as a content brief: it only needs to say what the video should be about. Infer useful generator-ready subject, environment, lighting, camera, motion, pacing, composition, style, and sound detail when it is missing while preserving every explicit user constraint. Preserve an explicitly requested or evidenced static hold. A conditioned still frame fixes its camera geometry at that instant; it does not by itself prove that the camera remains locked afterward.

Use the user's text, any backend-consumable pixels, and any supplied visual_description to produce one final, practical MiniMax H3 prompt. If the active backend is metadata-only, do not pretend you saw pixels; use only the user's text, the visual_description, and factual media metadata. Preserve requested subjects, exact text, composition, motion, style, camera behavior, visual capture language, audio guidance, and constraints. Visual capture language includes exposure level, blown highlights, underexposure, flash or direct light, bloom, halation, lens blur, bokeh, depth of field, focal length impression, perspective compression, lens distortion, motion blur, grain or noise, white balance, contrast, dynamic range, and camera angle. Do not describe invisible intent. The finished prompt must never contain instructions about how to write a prompt. Return only the JSON object requested by the model-facing contract below."""


@dataclass
class RuntimeConfig:
    model_path: str
    backend: str
    dtype: str
    quantization: str
    local_files_only: bool
    unload_policy: str
    max_memory_gb: float
    cli_path: str = ""
    extra_args: str = ""
    temperature: float = 0.45
    fallback_backend: str = "template"
    status: dict[str, Any] = field(default_factory=dict)


@dataclass
class MediaContext:
    images: Any = None
    source: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GemmaContext:
    user_prompt: str = ""
    images: Any = None
    source: str = "none"
    media_metadata: dict[str, Any] = field(default_factory=dict)
    visual_description: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class TargetProfileConfig:
    target_profile: str = "ltx"
    audio_mode: str = "auto_scene_audio"
    audio_guidance: str = ""
    target_duration_seconds: float = 0.0
    ltx_style: str = ""
    ideogram_aspect_ratio: str = "1:1"
    ideogram_render_style: str = ""
    ideogram_exact_text: str = ""
    ideogram_json_output: bool = True
    negative_prompt_mode: str = "auto"
    negative_prompt_guidance: str = ""
    minimax_h3_mode: str = "t2va"
    minimax_h3_shot_count: str = "auto"
    minimax_h3_dialogue_mode: str = "auto"
    minimax_h3_dialogue_line_count: int = 2
    minimax_h3_dialogue_guidance: str = ""
    ltx_generation_mode: str = "auto"
    ltx_long_horizon_mode: str = "off"
    ltx_camera_capability: str = "stable"


@dataclass
class H3ReferencePolicyConfig:
    """Host-owned rules for turning connected Ref2VA media into a manifest."""

    layout: str = "auto"
    custom_manifest: str = ""
    expected_subject_count: int = 0


@dataclass
class BackendRunOptions:
    """Per-call controls used only by the Grounding Guard path.

    A ``None`` options value is deliberately meaningful: it selects the
    historical backend call without changing its processor or RNG behavior.
    """

    sampling_profile: str = "checkpoint_defaults"
    thinking_mode: str = "auto"
    seed: int = 0
    enable_telemetry: bool = True
    require_transport: bool = False
    fail_closed_telemetry: bool = False
    visual_first: bool = True
    asset_registry: dict[str, Any] = field(default_factory=dict)
    call_budget: "BackendCallBudget | None" = None
    stage: str = "backend"


@dataclass
class BackendRunResult:
    decoded_text: str
    native_decoded_text: str = ""
    stage: str = "backend"
    transport: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    effective_sampling: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    forward_counts: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    generated_tokens: int = 0
    generated_characters: int = 0


class VisualTransportError(RuntimeError):
    def __init__(self, message: str, transport: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.transport = copy.deepcopy(transport or {})


@dataclass
class BackendCallBudget:
    max_calls: int = 4
    attempted_calls: int = 0

    def claim(self) -> int:
        if self.attempted_calls >= self.max_calls:
            raise RuntimeError(f"Grounding Guard exceeded its hard {self.max_calls}-call ceiling.")
        self.attempted_calls += 1
        return self.attempted_calls


_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_NVFP4_SMOKE_CACHE: dict[str, dict[str, Any]] = {}
_CURRENT_DIRECTOR_RUNTIME: ContextVar[dict[str, Any] | None] = ContextVar(
    "diffusiongemma_director_runtime",
    default=None,
)


def _new_director_runtime_metrics(cache_mode: str) -> dict[str, Any]:
    return {
        "schema": DIRECTOR_RUNTIME_SCHEMA_VERSION,
        "model_load_seconds": 0.0,
        "nvfp4_bridge_seconds": 0.0,
        "media_preprocess_seconds": 0.0,
        # DiffusionGemma performs multimodal encoding inside generate(); the
        # current Transformers surface does not expose a safe boundary for it.
        "multimodal_prefill_seconds": 0.0,
        "multimodal_prefill_is_separately_measured": False,
        "generation_seconds": 0.0,
        "grounding_validation_seconds": 0.0,
        "grounding_attempt_count": 0,
        "generated_token_count": 0,
        "generated_character_count": 0,
        "model_unload_seconds": 0.0,
        "total_director_seconds": 0.0,
        "peak_vram_mb": 0,
        "model_call_count": 0,
        "target_repair_attempt_count": 0,
        "dialogue_patch_attempt_count": 0,
        "retry_reason_count": 0,
        "grounding_first_pass_accepted": False,
        "cache_mode": str(cache_mode or "reuse"),
        "cache_hit": False,
        "cache_lookup_seconds": 0.0,
        "cache_write_seconds": 0.0,
        "runtime_model_cache_hit_count": 0,
        "runtime_model_cold_load_count": 0,
        "calls": [],
        "notes": {
            "nvfp4_bridge_seconds": "ModelOpt expert-bank packing/loading; this is a subset of model_load_seconds.",
            "media_preprocess_seconds": "Tensor-to-PIL conversion, chat-template processing, and input transfer inside the Director; upstream Media Sampler decode is not included.",
            "multimodal_prefill_seconds": "Included in generation_seconds until Transformers exposes a safe separate boundary.",
            "peak_vram_mb": "Process-wide CUDA peak allocated memory measured only for cache misses.",
        },
    }


def _director_metric_add(name: str, value: float | int) -> None:
    metrics = _CURRENT_DIRECTOR_RUNTIME.get()
    if not isinstance(metrics, dict):
        return
    current = metrics.get(name, 0)
    if isinstance(current, int) and isinstance(value, int):
        metrics[name] = current + value
    else:
        metrics[name] = _safe_float(current, 0.0) + _safe_float(value, 0.0)


def _director_metric_set(name: str, value: Any) -> None:
    metrics = _CURRENT_DIRECTOR_RUNTIME.get()
    if isinstance(metrics, dict):
        metrics[name] = value


def _director_metric_append_call(value: dict[str, Any]) -> None:
    metrics = _CURRENT_DIRECTOR_RUNTIME.get()
    if not isinstance(metrics, dict):
        return
    calls = metrics.setdefault("calls", [])
    if isinstance(calls, list):
        calls.append(copy.deepcopy(value))


def _reset_director_peak_vram() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.cuda.reset_peak_memory_stats()
        return True
    except Exception:
        return False


def _director_peak_vram_mb() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(round(float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)))
    except Exception:
        return 0


def _rounded_director_runtime(metrics: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(metrics)
    for key in (
        "model_load_seconds",
        "nvfp4_bridge_seconds",
        "media_preprocess_seconds",
        "multimodal_prefill_seconds",
        "generation_seconds",
        "grounding_validation_seconds",
        "model_unload_seconds",
        "total_director_seconds",
        "cache_lookup_seconds",
        "cache_write_seconds",
    ):
        result[key] = round(max(0.0, _safe_float(result.get(key), 0.0)), 6)
    for key in (
        "grounding_attempt_count",
        "generated_token_count",
        "generated_character_count",
        "peak_vram_mb",
        "model_call_count",
        "target_repair_attempt_count",
        "dialogue_patch_attempt_count",
        "retry_reason_count",
        "runtime_model_cache_hit_count",
        "runtime_model_cold_load_count",
    ):
        result[key] = int(max(0.0, _safe_float(result.get(key), 0.0)))
    return result


def _dg_log(message: str, *args: Any) -> None:
    logging.info("[DiffusionGemma] " + message, *args)


def _progress_enabled() -> bool:
    return os.environ.get("DG_PROGRESS", "1").strip().lower() not in {"0", "false", "off", "no"}


def _release_transformers_runtime() -> None:
    _MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True)


def _prompt_json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _ideogram_json_dumps(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _memory_preset_to_gb(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    lowered = text.lower()
    if "full" in lowered:
        return 0.0
    if "minimum" in lowered:
        return MIN_MODELOPT_NVFP4_BRIDGE_MEMORY_GB
    if "safe" in lowered:
        return 20.0
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match:
        return _safe_float(match.group(0), 20.0)
    return 20.0


def _run_blocking_with_progress(
    label: str,
    fn: Any,
    *,
    node_id: str | None = None,
    estimated_seconds: float | None = None,
) -> Any:
    if not _progress_enabled():
        return fn()

    estimate = max(5.0, _safe_float(os.environ.get("DG_PROGRESS_ESTIMATE_SECONDS", ""), estimated_seconds or 120.0))
    estimate_units = int(math.ceil(estimate))
    interval = max(0.25, _safe_float(os.environ.get("DG_PROGRESS_INTERVAL_SECONDS", ""), 1.0))
    result_box: dict[str, Any] = {}

    def worker() -> None:
        try:
            result_box["value"] = fn()
        except BaseException as exc:
            result_box["error"] = exc

    progress_bar = None
    tqdm_bar = None
    try:
        from comfy.utils import ProgressBar

        progress_bar = ProgressBar(estimate_units, node_id=node_id)
        progress_bar.update_absolute(0, estimate_units)
    except Exception:
        progress_bar = None
    try:
        from tqdm.auto import tqdm

        tqdm_bar = tqdm(
            total=estimate_units,
            desc=label,
            unit="s",
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
    except Exception:
        tqdm_bar = None

    thread = threading.Thread(target=worker, name=f"{label} worker", daemon=True)
    start = time.perf_counter()
    last_reported = 0
    displayed_total = estimate_units
    thread.start()
    try:
        while thread.is_alive():
            elapsed = time.perf_counter() - start
            displayed_total = max(displayed_total, int(math.ceil(elapsed + interval)))
            current = min(int(elapsed), displayed_total)
            if progress_bar is not None:
                progress_bar.update_absolute(current, displayed_total)
            if tqdm_bar is not None:
                if displayed_total != tqdm_bar.total:
                    tqdm_bar.total = displayed_total
                    tqdm_bar.refresh()
                delta = max(0, current - last_reported)
                if delta:
                    tqdm_bar.update(delta)
                    last_reported += delta
            time.sleep(interval)
        thread.join()
        elapsed = time.perf_counter() - start
        total = max(1, last_reported, int(math.ceil(elapsed)))
        if progress_bar is not None:
            progress_bar.update_absolute(total, total)
        if tqdm_bar is not None:
            if total != tqdm_bar.total:
                tqdm_bar.total = total
            remaining = max(0.0, total - tqdm_bar.n)
            if remaining:
                tqdm_bar.update(remaining)
    finally:
        if tqdm_bar is not None:
            tqdm_bar.close()

    if "error" in result_box:
        raise result_box["error"]
    return result_box.get("value")


def _normalize_aspect_ratio_key(aspect_ratio: Any) -> str:
    text = str(aspect_ratio or "").strip()
    match = re.search(r"\b(\d+)\s*:\s*(\d+)\b", text)
    if not match:
        return "1:1"
    key = f"{int(match.group(1))}:{int(match.group(2))}"
    return key if key in RESOLUTION_SELECTOR_RATIOS else "1:1"


def _resolution_selector_preset(aspect_ratio: Any) -> str:
    return RESOLUTION_SELECTOR_ASPECT_LABELS[_normalize_aspect_ratio_key(aspect_ratio)]


def _known_aspect_ratio_label_from_dimensions(width: int, height: int) -> str:
    src_w = max(1, int(width))
    src_h = max(1, int(height))
    source_ratio = src_w / float(src_h)
    best_key = ""
    best_delta = float("inf")
    for key, (w_ratio, h_ratio) in RESOLUTION_SELECTOR_RATIOS.items():
        delta = abs(source_ratio - (w_ratio / float(h_ratio)))
        if delta < best_delta:
            best_delta = delta
            best_key = key
    if best_key and best_delta <= 0.02:
        return RESOLUTION_SELECTOR_ASPECT_LABELS[best_key]
    return f"Custom / source ({src_w}x{src_h})"


def _resolution_selector_dimensions(aspect_ratio: Any, megapixels: Any = 2.0, multiple: Any = 8) -> tuple[int, int]:
    key = _normalize_aspect_ratio_key(aspect_ratio)
    w_ratio, h_ratio = RESOLUTION_SELECTOR_RATIOS[key]
    mp = max(0.1, min(16.0, _safe_float(megapixels, 2.0)))
    mult = int(max(8, min(128, _safe_float(multiple, 8))))
    mult = max(1, int(round(mult / 4)) * 4)
    total_pixels = mp * 1024 * 1024
    scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
    width = max(mult, round(w_ratio * scale / mult) * mult)
    height = max(mult, round(h_ratio * scale / mult) * mult)
    return int(width), int(height)


def _nearest_resolution_aspect_ratio_key(width: Any, height: Any) -> str:
    src_w = max(1, int(_safe_float(width, 1)))
    src_h = max(1, int(_safe_float(height, 1)))
    source_ratio = src_w / float(src_h)
    return min(
        RESOLUTION_SELECTOR_RATIOS,
        key=lambda key: abs(
            source_ratio
            - (
                RESOLUTION_SELECTOR_RATIOS[key][0]
                / float(RESOLUTION_SELECTOR_RATIOS[key][1])
            )
        ),
    )


def _splitter_resolution_aspect_ratio(
    requested: Any,
    context: "GemmaContext",
    target: dict[str, Any],
    packet_aspect_ratio: Any,
) -> tuple[str, str]:
    selection = str(requested or RESOLUTION_SELECTOR_AUTO).strip()
    profile = _normalize_target_profile(str(target.get("target_profile", "ltx")))
    if profile == "ideogram4":
        return _normalize_aspect_ratio_key(packet_aspect_ratio), "ideogram_target_profile"

    auto_aliases = {
        "",
        "auto",
        "auto (recommended)",
        RESOLUTION_SELECTOR_AUTO.casefold(),
    }
    if selection.casefold() not in auto_aliases:
        match = re.search(r"\b(\d+)\s*:\s*(\d+)\b", selection)
        if match:
            requested_key = f"{int(match.group(1))}:{int(match.group(2))}"
            if requested_key in RESOLUTION_SELECTOR_RATIOS:
                return requested_key, "splitter_widget"

    if profile == "ltx":
        metadata = context.media_metadata if isinstance(context, GemmaContext) else {}
        for width_key, height_key, source in (
            ("ltx_first_frame_width", "ltx_first_frame_height", "ltx_first_frame"),
            ("width", "height", "media_dimensions"),
        ):
            width = int(_safe_float(metadata.get(width_key), 0))
            height = int(_safe_float(metadata.get(height_key), 0))
            if width > 0 and height > 0:
                return _nearest_resolution_aspect_ratio_key(width, height), source
        return "16:9", "ltx_default"
    return _normalize_aspect_ratio_key(packet_aspect_ratio), "legacy_target_profile"


def _round_dimension_to_multiple(value: float, multiple: int) -> int:
    mult = int(max(1, multiple))
    return max(mult, int(round(float(value) / mult) * mult))


def _dimensions_for_long_edge(width: int, height: int, long_edge: int, multiple: int = 8) -> tuple[int, int]:
    src_w = max(1, int(width))
    src_h = max(1, int(height))
    edge = max(64, int(long_edge))
    scale = edge / float(max(src_w, src_h))
    return (
        _round_dimension_to_multiple(src_w * scale, multiple),
        _round_dimension_to_multiple(src_h * scale, multiple),
    )


def _aspect_dimensions_for_long_edge(aspect_ratio: Any, long_edge: int, multiple: int = 8) -> tuple[int, int]:
    key = _normalize_aspect_ratio_key(aspect_ratio)
    w_ratio, h_ratio = RESOLUTION_SELECTOR_RATIOS[key]
    edge = max(64, int(long_edge))
    if w_ratio >= h_ratio:
        width = edge
        height = max(1, int(round(edge * (h_ratio / float(w_ratio)))))
    else:
        height = edge
        width = max(1, int(round(edge * (w_ratio / float(h_ratio)))))
    return (_round_dimension_to_multiple(width, multiple), _round_dimension_to_multiple(height, multiple))


def _center_crop_dimensions_for_aspect(width: int, height: int, aspect_ratio: Any) -> tuple[int, int]:
    key = _normalize_aspect_ratio_key(aspect_ratio)
    target_w, target_h = RESOLUTION_SELECTOR_RATIOS[key]
    src_w = max(1, int(width))
    src_h = max(1, int(height))
    src_ratio = src_w / float(src_h)
    dst_ratio = target_w / float(target_h)
    if abs(src_ratio - dst_ratio) < 1e-6:
        return (src_w, src_h)
    if src_ratio > dst_ratio:
        crop_w = max(1, int(round(src_h * dst_ratio)))
        return (min(crop_w, src_w), src_h)
    crop_h = max(1, int(round(src_w / dst_ratio)))
    return (src_w, min(crop_h, src_h))


def _center_crop_image_batch(image: Any, crop_width: int, crop_height: int) -> Any:
    try:
        import torch

        tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        _, src_h, src_w, _ = tensor.shape
        crop_w = max(1, min(int(crop_width), int(src_w)))
        crop_h = max(1, min(int(crop_height), int(src_h)))
        x0 = max(0, (int(src_w) - crop_w) // 2)
        y0 = max(0, (int(src_h) - crop_h) // 2)
        return tensor[:, y0 : y0 + crop_h, x0 : x0 + crop_w, :]
    except Exception:
        return image


def _resize_image_batch(image: Any, width: int, height: int) -> Any:
    import torch

    tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    original_dtype = tensor.dtype
    original_device = tensor.device
    work = tensor
    if not torch.is_floating_point(work):
        work = work.float()
    nchw = work.permute(0, 3, 1, 2).contiguous()
    try:
        resized = torch.nn.functional.interpolate(
            nchw,
            size=(max(1, int(height)), max(1, int(width))),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    except TypeError:
        resized = torch.nn.functional.interpolate(
            nchw,
            size=(max(1, int(height)), max(1, int(width))),
            mode="bilinear",
            align_corners=False,
        )
    result = resized.permute(0, 2, 3, 1).contiguous()
    if torch.is_floating_point(tensor) and result.dtype != original_dtype:
        result = result.to(dtype=original_dtype)
    return result.to(device=original_device)


def _h3_reference_dimensions_for_pixel_budget(
    width: int,
    height: int,
    pixel_budget: float,
    multiple: int = 32,
) -> tuple[int, int]:
    """Return downscale-only, aspect-preserving dimensions within a pixel budget."""

    src_w = max(1, int(width))
    src_h = max(1, int(height))
    mult = max(1, int(multiple))
    minimum_pixels = mult * mult
    budget = float(pixel_budget)
    if budget < minimum_pixels:
        raise ValueError(
            f"MiniMax H3 reference pixel budget must allow at least {mult}x{mult} pixels; got {budget:.0f}."
        )
    scale = min(1.0, math.sqrt(budget / float(src_w * src_h)))
    target_w = max(mult, int(math.floor((src_w * scale) / mult)) * mult)
    target_h = max(mult, int(math.floor((src_h * scale) / mult)) * mult)
    target_w = min(target_w, max(mult, (src_w // mult) * mult))
    target_h = min(target_h, max(mult, (src_h // mult) * mult))
    if target_w > src_w or target_h > src_h:
        raise ValueError(
            f"MiniMax H3 references must be at least {mult} pixels on each axis; got {src_w}x{src_h}."
        )
    while target_w * target_h > budget and (target_w > mult or target_h > mult):
        width_scale = target_w / float(src_w)
        height_scale = target_h / float(src_h)
        if target_w > mult and (target_h == mult or width_scale >= height_scale):
            target_w -= mult
        else:
            target_h -= mult
    if target_w * target_h > budget:
        raise ValueError(
            f"MiniMax H3 reference pixel budget {budget:.0f} cannot fit a {mult}x{mult} aligned image."
        )
    return (int(target_w), int(target_h))


def _h3_native_match_reference_dimensions(
    width: int,
    height: int,
    generation_width: int,
    generation_height: int,
    multiple: int = 32,
) -> tuple[int, int]:
    """Mirror MiniMaxH3ReferenceToVideo's current ``match`` sizing for telemetry."""

    src_w = max(1, int(width))
    src_h = max(1, int(height))
    canvas_pixels = max(1, int(generation_width)) * max(1, int(generation_height))
    scale = min(1.0, math.sqrt(canvas_pixels / float(src_w * src_h)))
    mult = max(1, int(multiple))
    return (
        max(mult, int(round((src_w * scale) / mult)) * mult),
        max(mult, int(round((src_h * scale) / mult)) * mult),
    )


def _h3_shared_reference_pixel_budgets(
    source_pixels_1: int,
    source_pixels_2: int,
    total_pixel_budget: float,
    reference_1_share: float,
) -> tuple[float, float]:
    """Split one budget between two refs and reassign unused source capacity."""

    total = max(0.0, float(total_pixel_budget))
    share = max(0.0, min(1.0, float(reference_1_share)))
    budget_1 = total * share
    budget_2 = total - budget_1
    if source_pixels_1 < budget_1:
        budget_2 += budget_1 - float(source_pixels_1)
        budget_1 = float(source_pixels_1)
    if source_pixels_2 < budget_2:
        budget_1 += budget_2 - float(source_pixels_2)
        budget_2 = float(source_pixels_2)
    return (budget_1, budget_2)


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _pad_nvfp4_block_scale_rows(block_scale: Any, output_rows: int) -> Any:
    import torch

    rounded_rows = ((int(output_rows) + 127) // 128) * 128
    if int(block_scale.shape[0]) >= rounded_rows:
        return block_scale
    pad_shape = (rounded_rows - int(block_scale.shape[0]), int(block_scale.shape[1]))
    padding = torch.zeros(pad_shape, dtype=block_scale.dtype, device=block_scale.device)
    return torch.cat([block_scale, padding], dim=0)


def _modelopt_to_comfy_fp4_packing(qdata: Any) -> Any:
    return ((qdata & 0x0F) << 4) | ((qdata & 0xF0) >> 4)


def _modelopt_to_comfy_block_scale(block_scale: Any) -> Any:
    from comfy_kitchen.float_utils import to_blocked

    return to_blocked(block_scale, flatten=False)


def _safetensors_index(model_path: str) -> dict[str, str]:
    index_path = Path(model_path) / "model.safetensors.index.json"
    index_data = _read_json_file(index_path)
    weight_map = index_data.get("weight_map")
    return weight_map if isinstance(weight_map, dict) else {}


def _nvfp4_bridge_smoke(model_path: str) -> dict[str, Any]:
    cache_key = str(Path(model_path))
    cached = _NVFP4_SMOKE_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()
    result: dict[str, Any] = {"passed": False, "error": "", "projection": ""}
    try:
        import torch
        from safetensors import safe_open
        from comfy.quant_ops import QuantizedTensor, get_layout_class

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        root = Path(model_path)
        weight_map = _safetensors_index(model_path)
        prefix = "model.decoder.layers.0.experts.0.gate_proj"
        required = [f"{prefix}.{name}" for name in ("weight", "weight_scale", "weight_scale_2", "input_scale")]
        if not all(key in weight_map for key in required):
            raise RuntimeError("Expected layer-0 expert gate_proj NVFP4 tensors were not found.")

        handles: dict[str, Any] = {}

        def get_tensor(key: str) -> Any:
            shard = weight_map[key]
            handle = handles.get(shard)
            if handle is None:
                handle = safe_open(str(root / shard), framework="pt", device="cpu")
                handles[shard] = handle
            return handle.get_tensor(key)

        try:
            qdata = _modelopt_to_comfy_fp4_packing(get_tensor(f"{prefix}.weight")).to("cuda", non_blocking=True)
            block_scale = _modelopt_to_comfy_block_scale(get_tensor(f"{prefix}.weight_scale")).to("cuda", non_blocking=True)
            tensor_scale = get_tensor(f"{prefix}.weight_scale_2").to("cuda", non_blocking=True)
            input_scale = get_tensor(f"{prefix}.input_scale").to("cuda", non_blocking=True)
        finally:
            handles.clear()

        out_features = int(qdata.shape[0])
        in_features = int(qdata.shape[1]) * 2
        layout_cls = get_layout_class("TensorCoreNVFP4Layout")
        weight_params = layout_cls.Params(
            scale=tensor_scale,
            block_scale=block_scale,
            orig_dtype=torch.bfloat16,
            orig_shape=(out_features, in_features),
        )
        weight = QuantizedTensor(qdata.to(torch.uint8), "TensorCoreNVFP4Layout", weight_params)
        x = torch.zeros((2, in_features), dtype=torch.bfloat16, device="cuda")
        qx = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout", scale=input_scale)
        y = torch.nn.functional.linear(qx, weight, None)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(y.float()).all()):
            raise RuntimeError("NVFP4 bridge smoke produced non-finite output.")
        result.update(
            {
                "passed": True,
                "projection": prefix,
                "output_shape": list(y.shape),
                "output_dtype": str(y.dtype).replace("torch.", ""),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    _NVFP4_SMOKE_CACHE[cache_key] = result.copy()
    return result


def _model_path_info(model_path: str) -> dict[str, Any]:
    path = Path(model_path or "")
    info: dict[str, Any] = {
        "path_kind": "missing",
        "is_gguf_file": False,
        "is_hf_repo": False,
        "model_type": "",
        "architecture": "",
        "quant_algo": "",
        "quant_method": "",
        "quant_producer": "",
        "quant_producer_version": "",
        "transformers_version_required": "",
        "has_generation_config": False,
        "has_modelopt_state": False,
        "runtime_hint": "",
    }
    if not path.exists():
        return info
    if path.is_file():
        info["path_kind"] = "gguf_file" if path.suffix.lower() == ".gguf" else "file"
        info["is_gguf_file"] = path.suffix.lower() == ".gguf"
        if info["is_gguf_file"]:
            info["runtime_hint"] = "llama-diffusion-cli GGUF"
        return info
    if not path.is_dir():
        info["path_kind"] = "unknown"
        return info

    config = _read_json_file(path / "config.json")
    hf_quant_config = _read_json_file(path / "hf_quant_config.json")
    quant_config = config.get("quantization_config") if isinstance(config.get("quantization_config"), dict) else {}
    if not quant_config:
        quant_config = hf_quant_config.get("quantization") if isinstance(hf_quant_config.get("quantization"), dict) else {}
    producer = quant_config.get("producer") if isinstance(quant_config.get("producer"), dict) else {}
    if not producer:
        producer = hf_quant_config.get("producer") if isinstance(hf_quant_config.get("producer"), dict) else {}
    architectures = config.get("architectures") if isinstance(config.get("architectures"), list) else []
    info.update(
        {
            "is_hf_repo": bool(config),
            "model_type": str(config.get("model_type", "")),
            "architecture": str(architectures[0]) if architectures else "",
            "quant_algo": str(quant_config.get("quant_algo", "")),
            "quant_method": str(quant_config.get("quant_method", config.get("quantization_config", {}).get("quant_method", "")) if isinstance(config.get("quantization_config"), dict) else ""),
            "quant_producer": str(producer.get("name", "")),
            "quant_producer_version": str(producer.get("version", "")),
            "transformers_version_required": str(config.get("transformers_version", "")),
            "has_generation_config": (path / "generation_config.json").exists(),
            "has_modelopt_state": (path / "modelopt_state.pth").exists(),
        }
    )
    if info["quant_algo"].upper() == "NVFP4":
        info["path_kind"] = "nvfp4_hf_repo"
        info["runtime_hint"] = "NVIDIA ModelOpt NVFP4 checkpoint"
    elif config:
        info["path_kind"] = "hf_repo"
        info["runtime_hint"] = "Transformers whole-repo model"
    else:
        info["path_kind"] = "directory"
    return info


def _runtime_status(config: RuntimeConfig) -> dict[str, Any]:
    model_info = _model_path_info(config.model_path)
    status: dict[str, Any] = {
        "backend": config.backend,
        "model_path": config.model_path,
        "model_path_exists": Path(config.model_path).exists(),
        **model_info,
        "local_files_only": config.local_files_only,
        "max_memory_gb": config.max_memory_gb,
        "temperature": config.temperature,
        "temperature_semantics": (
            "GGUF compatibility scalar; in-process DiffusionGemma Grounding Guard calls use the selected native "
            "0.8-to-0.4 denoising sampling profile."
        ),
        "transformers_importable": False,
        "diffusion_gemma_module": False,
        "diffusion_gemma_class": False,
        "auto_processor": False,
        "vllm_importable": False,
        "modelopt_importable": False,
        "comfy_kitchen_importable": False,
        "fp_quant_importable": False,
        "qutlass_importable": False,
        "tensorrt_llm_importable": False,
        "torch_cuda_available": False,
        "torch_cuda_capability": "",
        "transformers_version": "",
        "processor_class": "",
        "modelopt_nvfp4_supported": False,
        "comfy_nvfp4_bridge_supported": False,
        "nvfp4_bridge_min_memory_gb": MIN_MODELOPT_NVFP4_BRIDGE_MEMORY_GB,
        "nvfp4_bridge_memory_guard_passed": True,
        "nvfp4_bridge_smoke": {},
        "ready": False,
        "supports_pixels": False,
        "supports_video": False,
        "supports_thinking": False,
        "media_policy": "unknown",
        "notes": [],
    }
    try:
        import importlib.util

        status["transformers_importable"] = importlib.util.find_spec("transformers") is not None
        status["diffusion_gemma_module"] = importlib.util.find_spec("transformers.models.diffusion_gemma") is not None
        status["vllm_importable"] = importlib.util.find_spec("vllm") is not None
        status["modelopt_importable"] = bool(
            importlib.util.find_spec("modelopt") or importlib.util.find_spec("nvidia_modelopt")
        )
        status["comfy_kitchen_importable"] = importlib.util.find_spec("comfy_kitchen") is not None
        status["fp_quant_importable"] = importlib.util.find_spec("fp_quant") is not None
        status["qutlass_importable"] = importlib.util.find_spec("qutlass") is not None
        status["tensorrt_llm_importable"] = importlib.util.find_spec("tensorrt_llm") is not None
        try:
            import transformers

            status["transformers_version"] = str(getattr(transformers, "__version__", ""))
        except Exception:
            pass
        try:
            import torch

            status["torch_cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                status["torch_cuda_capability"] = ".".join(str(part) for part in torch.cuda.get_device_capability())
        except Exception as exc:
            if config.backend == "transformers_inprocess":
                status["notes"].append(f"Torch CUDA probe failed: {exc}")
        try:
            from transformers import AutoProcessor  # noqa: F401

            status["auto_processor"] = True
            if config.backend == "transformers_inprocess" and status["model_path_exists"] and status["is_hf_repo"]:
                try:
                    processor = AutoProcessor.from_pretrained(config.model_path, local_files_only=True)
                    status["processor_class"] = type(processor).__name__
                except Exception as exc:
                    status["notes"].append(f"AutoProcessor load failed: {exc}")
        except Exception as exc:
            if config.backend == "transformers_inprocess":
                status["notes"].append(f"AutoProcessor import failed: {exc}")
        try:
            from transformers import DiffusionGemmaForBlockDiffusion  # noqa: F401

            status["diffusion_gemma_class"] = True
        except Exception as exc:
            if config.backend == "transformers_inprocess":
                status["notes"].append(f"DiffusionGemmaForBlockDiffusion import failed: {exc}")
    except Exception as exc:
        if config.backend == "transformers_inprocess":
            status["notes"].append(f"Transformers probe failed: {exc}")

    if config.backend == "transformers_inprocess":
        modelopt_nvfp4 = status["path_kind"] == "nvfp4_hf_repo" and status["quant_method"] == "modelopt"
        low_bridge_memory = bool(
            modelopt_nvfp4
            and config.quantization == "modelopt_nvfp4"
            and 0 < float(config.max_memory_gb) < MIN_MODELOPT_NVFP4_BRIDGE_MEMORY_GB
        )
        status["nvfp4_bridge_memory_guard_passed"] = not low_bridge_memory
        if modelopt_nvfp4 and config.quantization == "modelopt_nvfp4":
            smoke = _nvfp4_bridge_smoke(config.model_path)
            status["nvfp4_bridge_smoke"] = smoke
            status["comfy_nvfp4_bridge_supported"] = bool(
                smoke.get("passed")
                and status["comfy_kitchen_importable"]
                and status["torch_cuda_available"]
                and status["diffusion_gemma_class"]
                and status["auto_processor"]
            )
        status["modelopt_nvfp4_supported"] = bool(
            modelopt_nvfp4
            and (
                status["comfy_nvfp4_bridge_supported"]
                or (
                    status["modelopt_importable"]
                    and status["tensorrt_llm_importable"]
                    and status["has_modelopt_state"]
                )
            )
        )
        status["ready"] = bool(
            status["model_path_exists"]
            and status["auto_processor"]
            and status["diffusion_gemma_class"]
            and not (modelopt_nvfp4 and not status["modelopt_nvfp4_supported"])
            and not low_bridge_memory
        )
        status["supports_pixels"] = bool(status["ready"])
        status["supports_video"] = bool(status["ready"])
        status["supports_thinking"] = bool(status["ready"])
        status["media_policy"] = "pixels_when_ready"
        if modelopt_nvfp4 and status["comfy_nvfp4_bridge_supported"]:
            status["notes"].append(
                "This NVFP4 repo uses quant_method=modelopt, which vanilla Transformers does not deserialize. "
                "The node will use the local Comfy NVFP4 bridge: Transformers supplies DiffusionGemma structure "
                "and Comfy/comfy_kitchen supplies packed NVFP4 expert matmul."
            )
            if low_bridge_memory:
                status["notes"].append(
                    f"The local NVFP4 bridge does not offload by max_memory_gb yet. Set max_memory_gb to "
                    f"{MIN_MODELOPT_NVFP4_BRIDGE_MEMORY_GB:g} or higher, or 0 for an explicit full-GPU load. "
                    "Lower values are blocked to avoid a partial CUDA load that can require a ComfyUI restart."
                )
        elif modelopt_nvfp4 and not status["modelopt_nvfp4_supported"]:
            status["notes"].append(
                "This NVFP4 repo uses quant_method=modelopt. Transformers 5.12.1 exposes DiffusionGemma, "
                "but it does not register modelopt as an in-process quantizer. The local ModelOpt package "
                "also lacks a modelopt_state.pth restore file for this repo, TensorRT-LLM is not available, "
                "and the Comfy NVFP4 bridge smoke did not pass. "
                "Blocking before full weight load to avoid a CPU-memory hang."
            )
    elif config.backend == "gguf_subprocess":
        status["cli_path_exists"] = bool(config.cli_path and Path(config.cli_path).exists())
        status["cuda_runtime_bin_exists"] = Path(DEFAULT_CUDA_RUNTIME_BIN).exists()
        status["ready"] = bool(status["model_path_exists"] and status["cli_path_exists"] and status["is_gguf_file"])
        status["supports_pixels"] = False
        status["supports_video"] = False
        status["supports_thinking"] = bool(status["ready"])
        status["media_policy"] = "metadata_only"
        if not status["cli_path_exists"]:
            status["notes"].append("GGUF subprocess backend needs a foreground llama-diffusion-cli path.")
        if status["model_path_exists"] and not status["is_gguf_file"]:
            if status["path_kind"] == "nvfp4_hf_repo":
                status["notes"].append(
                    "This path is an NVFP4 Hugging Face/vLLM repo, not a .gguf file. "
                    "The clean Model Loader runs llama-diffusion-cli and needs the GGUF model path."
                )
            elif status["is_hf_repo"]:
                status["notes"].append(
                    "This path is a Hugging Face repo folder, not a .gguf file. "
                    "Use Model Loader (Advanced) with transformers_inprocess only after proof gates pass."
                )
            else:
                status["notes"].append("GGUF subprocess backend needs model_path to point directly to a .gguf file.")
    elif config.backend in {"qwen_vl_placeholder", "minicpm_v_placeholder"}:
        status["ready"] = False
        status["notes"].append(
            "Fallback backend lane is reserved for comparative benchmarking; no Qwen/MiniCPM runner is loaded by this pack yet."
        )
    else:
        status["ready"] = config.backend == "template"
        status["supports_thinking"] = False
        status["media_policy"] = "template"

    if config.backend == "transformers_inprocess" and not status["ready"]:
        if not status["diffusion_gemma_class"]:
            status["notes"].append(
                "This ComfyUI venv can keep the node pack loaded, but it cannot run DiffusionGemma until "
                "a compatible Transformers build exposes DiffusionGemmaForBlockDiffusion."
            )
        if status["path_kind"] == "nvfp4_hf_repo" and not status["comfy_nvfp4_bridge_supported"]:
            status["notes"].append(
                "NVFP4 was detected. Use this path only after a supported in-process ModelOpt NVFP4 loader is "
                "available; the current local foreground fallback remains the GGUF llama-diffusion-cli route."
            )
    return status


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```(?:json|text|markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_delimited_final_payload(text: str) -> tuple[str, bool]:
    value = str(text or "")
    if not value:
        return "", False
    lowered = value.lower()
    starts: list[tuple[int, str]] = []
    for marker in FINAL_JSON_OPEN_MARKERS:
        index = lowered.find(marker.lower())
        if index >= 0:
            starts.append((index, marker))
    if not starts:
        return value, False
    start_index, open_marker = min(starts, key=lambda item: item[0])
    payload_start = start_index + len(open_marker)
    payload_end = len(value)
    for marker in FINAL_JSON_CLOSE_MARKERS:
        index = lowered.find(marker.lower(), payload_start)
        if index >= 0:
            payload_end = min(payload_end, index)
    return value[payload_start:payload_end].strip(), True


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\|thought\|>.*?<\|end_thought\|>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\|thought\|>.*?(?=<\|(?:final_json|answer|final)\|>)", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\|channel\|>thought.*?(?=<\|channel\|>|$)", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\|?channel\|?>\s*thought\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|?channel\|?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    return text.strip()


def _strip_cli_runtime_lines(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        if re.match(r"\s*(total time:|throughput:|diffusion step:)", line, flags=re.IGNORECASE):
            continue
        if re.match(r"\s*\d+\.\d+\.\d+\.\d+\s+[IWE]\s+", line):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _strip_generation_artifacts(text: str) -> str:
    value = str(text or "")
    if not value:
        return ""
    replacements = {
        "\ufeff": "",
        "\ufffd": "",
        "ï»¿": "",
        "ç´‹": " ",
    }
    for artifact, replacement in replacements.items():
        value = value.replace(artifact, replacement)
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", value)
    value = re.sub(r"(?:(?<=\s)|(?<=[,.;:!?]))(?:[ÃÂâ][^\sA-Za-z0-9]{1,8}|[ÃÂ][A-Za-z0-9]{1,4})(?=\s|[,.;:!?]|$)", " ", value)
    value = re.sub(r"\s*,\s+to\s+(him|her|them)\b", r" near \1", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+([,.!?;:])", r"\1", value)
    value = re.sub(r"([,.!?;:])(?=[^\s,.!?;:])", r"\1 ", value)
    value = re.sub(
        r'([,.!?;:])\s+(["\'”’»])(?=\s|$|[,.;:!?)\]}])',
        r"\1\2",
        value,
    )
    return re.sub(r"\s+", " ", value).strip()


def _split_extra_args(extra_args: str) -> list[str]:
    if not extra_args.strip():
        return []
    if os.name != "nt":
        return shlex.split(extra_args)
    return [part.strip("\"'") for part in shlex.split(extra_args, posix=False)]


def _format_temperature(value: Any, default: float = 0.2) -> str:
    temp = max(0.0, min(2.0, _safe_float(value, default)))
    return f"{temp:.3f}".rstrip("0").rstrip(".")


def _extra_args_with_temperature(extra_args: str, temperature: float) -> str:
    value = (extra_args or "").strip()
    temp = _format_temperature(temperature)
    if not value:
        return f"--temp {temp}"
    pattern = r"(?<!\S)(?:--temp|-temp|--temperature|-temperature)\s+\S+"
    if re.search(pattern, value):
        return re.sub(pattern, f"--temp {temp}", value, count=1)
    return f"{value} --temp {temp}".strip()


def _subprocess_env_for_cli(cli_path: str) -> dict[str, str]:
    env = os.environ.copy()
    path_key = "Path" if "Path" in env else "PATH"
    existing_path = env.get(path_key, "")
    candidates = [
        str(Path(cli_path).parent) if cli_path else "",
        DEFAULT_CUDA_RUNTIME_BIN,
    ]
    prefix = [path for path in candidates if path and Path(path).exists()]
    env[path_key] = os.pathsep.join([*prefix, existing_path]) if existing_path else os.pathsep.join(prefix)
    cuda_root = Path(DEFAULT_CUDA_RUNTIME_BIN).parent
    if cuda_root.exists():
        env.setdefault("CUDA_PATH", str(cuda_root))
    return env


def _sanitize_prompt_text(
    text: str,
    fallback: str = "",
    max_chars: int = 4000,
    strip_thinking: bool = True,
    strip_markdown: bool = True,
) -> str:
    value = (text or "").strip()
    if strip_thinking:
        value = _strip_thinking(value)
    if strip_markdown:
        value = _strip_markdown(value)
    value = _strip_generation_artifacts(value)
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        value = (fallback or "").strip()
    if max_chars > 0 and len(value) > max_chars:
        value = value[:max_chars].rsplit(" ", 1)[0].strip() or value[:max_chars].strip()
    return value


def _sanitize_structured_prompt_text(
    text: str,
    fallback: str = "",
    max_chars: int = 8000,
    strip_thinking: bool = True,
    strip_markdown: bool = True,
) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if strip_thinking:
        value = _strip_thinking(value)
    if strip_markdown:
        value = _strip_markdown(value)
    replacements = {
        "\ufeff": "",
        "\ufffd": "",
        "ï»¿": "",
    }
    for artifact, replacement in replacements.items():
        value = value.replace(artifact, replacement)
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", value)
    lines = []
    for line in value.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    value = "\n".join(lines)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    if not value:
        value = str(fallback or "").strip()
    if max_chars > 0 and len(value) > max_chars:
        value = value[:max_chars].rstrip()
    return value


_MINIMAX_H3_REF_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)


def _minimax_h3_ref_sections(prompt: str) -> dict[str, str] | None:
    pattern = r"(?ms)^" + r"\s*\n+\s*".join(
        rf"{re.escape(field_name)}\s*:\s*(.*?)" for field_name in _MINIMAX_H3_REF_FIELDS[:-1]
    ) + rf"\s*\n+\s*{re.escape(_MINIMAX_H3_REF_FIELDS[-1])}\s*:\s*(.*)$"
    match = re.fullmatch(pattern, prompt)
    if not match:
        return None
    return {
        field_name: match.group(index + 1).strip()
        for index, field_name in enumerate(_MINIMAX_H3_REF_FIELDS)
    }


def _rebuild_minimax_h3_ref_prompt(sections: dict[str, str]) -> str:
    return "\n\n".join(f"{field_name}:\n{sections.get(field_name, '').strip()}" for field_name in _MINIMAX_H3_REF_FIELDS)


def _normalize_minimax_h3_prompt(text: str, minimax_h3_mode: str = "t2va") -> str:
    prompt = _sanitize_structured_prompt_text(text, "", 0)
    if not prompt:
        return ""
    # These are unambiguous transport defects, not creative edits. Host-only
    # grounding annotations must never reach H3, and DiffusionGemma sometimes
    # drops the leading N from the exact N/A absence sentinel.
    prompt = _strip_grounding_role_annotations(prompt)
    prompt = re.sub(
        r"(?im)^(?P<prefix>[ \t]*(?:overall_soundscape|non_diegetic_music)[ \t]*:[ \t]*(?:\n[ \t]*)?)/A(?=[ \t]*(?:$|\n))",
        lambda match: f"{match.group('prefix')}N/A",
        prompt,
    )

    if _normalize_minimax_h3_mode(minimax_h3_mode) == "ref2va":
        # DiffusionGemma occasionally drops the leading "n" from this one
        # required heading. The line-anchored alias is unambiguous and only
        # changes the field label; its generated music guidance is preserved.
        music_alias_pattern = re.compile(r"(?im)^\s*on[ _-]*diegetic[ _-]*music\s*:")
        canonical_music_pattern = re.compile(
            r"(?im)^\s*non[ _-]*diegetic[ _-]*music\s*:"
        )
        music_alias_matches = list(music_alias_pattern.finditer(prompt))
        canonical_music_matches = list(canonical_music_pattern.finditer(prompt))
        preceding_heading_positions: list[int] = []
        preceding_headings_are_canonical = True
        for field_name in _MINIMAX_H3_REF_FIELDS[:-1]:
            field_matches = list(re.finditer(rf"(?im)^\s*{re.escape(field_name)}\s*:", prompt))
            if len(field_matches) != 1:
                preceding_headings_are_canonical = False
                break
            preceding_heading_positions.append(field_matches[0].start())
        preceding_structure_is_canonical = bool(
            preceding_headings_are_canonical
            and preceding_heading_positions == sorted(preceding_heading_positions)
        )
        # A common constrained-decoding miss is a duplicated, empty final music
        # heading. The first section is complete, so dropping only the empty EOF
        # duplicate is lossless and avoids sending the same packet through two
        # expensive model repair attempts.
        if (
            preceding_structure_is_canonical
            and len(canonical_music_matches) == 2
            and canonical_music_matches[0].start() > preceding_heading_positions[-1]
            and canonical_music_matches[1].start() > canonical_music_matches[0].start()
            and prompt[
                canonical_music_matches[0].end() : canonical_music_matches[1].start()
            ].strip()
            and not prompt[canonical_music_matches[1].end() :].strip()
        ):
            prompt = prompt[: canonical_music_matches[1].start()].rstrip()
            canonical_music_matches = list(canonical_music_pattern.finditer(prompt))

        alias_has_nonempty_body = bool(
            len(music_alias_matches) == 1
            and (
                prompt[music_alias_matches[0].end() :].strip()
                if not canonical_music_matches
                else music_alias_matches[0].start() < canonical_music_matches[0].start()
                and prompt[
                    music_alias_matches[0].end() : canonical_music_matches[0].start()
                ].strip()
            )
        )
        empty_canonical_footer_after_alias = bool(
            len(music_alias_matches) == 1
            and len(canonical_music_matches) == 1
            and music_alias_matches[0].start() < canonical_music_matches[0].start()
            and not prompt[canonical_music_matches[0].end() :].strip()
        )
        if (
            preceding_structure_is_canonical
            and len(music_alias_matches) == 1
            and music_alias_matches[0].start() > preceding_heading_positions[-1]
            and alias_has_nonempty_body
            and (not canonical_music_matches or empty_canonical_footer_after_alias)
        ):
            if empty_canonical_footer_after_alias:
                prompt = prompt[: canonical_music_matches[0].start()].rstrip()
            prompt = music_alias_pattern.sub("non_diegetic_music:", prompt, count=1)
        prompt = re.sub(
            r"<\s*(subject|picture|image|video|audio)\s+(\d+)\s*>",
            lambda match: (
                f"<Subject {int(match.group(2))}>"
                if match.group(1).casefold() == "subject"
                else f"<{'Picture' if match.group(1).casefold() in {'picture', 'image'} else match.group(1).title()} {int(match.group(2))}>"
            ),
            prompt,
            flags=re.IGNORECASE,
        )
        for field_name in _MINIMAX_H3_REF_FIELDS:
            prompt = re.sub(
                rf"(?im)^\s*{re.escape(field_name)}\s*:",
                f"{field_name}:",
                prompt,
            )
        sections = _minimax_h3_ref_sections(prompt)
        return _rebuild_minimax_h3_ref_prompt(sections) if sections else prompt

    fields = (
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
    )
    for field_name in fields:
        prompt = re.sub(
            rf"(?im)^\s*{re.escape(field_name)}\s*:",
            f"{field_name}:",
            prompt,
        )

    field_matches = {
        field_name: re.search(rf"(?m)^{re.escape(field_name)}\s*:", prompt)
        for field_name in fields
    }
    if not any(field_matches.values()) and re.match(r"^\[Shot 1\](?=\s|$)", prompt):
        blocks = [block.strip() for block in re.split(r"\n\s*\n", prompt) if block.strip()]
        if len(blocks) >= 3:
            integrated = "\n\n".join(blocks[:-2])
            prompt = (
                f"integrated_multimodal_description: {integrated}\n\n"
                f"overall_soundscape: {blocks[-2]}\n\n"
                f"non_diegetic_music: {blocks[-1]}"
            )
    elif (
        not field_matches["integrated_multimodal_description"]
        and field_matches["overall_soundscape"]
        and field_matches["non_diegetic_music"]
        and re.match(r"^\[Shot 1\](?=\s|$)", prompt)
    ):
        prompt = f"integrated_multimodal_description: {prompt}"

    structured_match = re.fullmatch(
        r"(?ms)integrated_multimodal_description\s*:\s*(.*?)\s*\n+\s*overall_soundscape\s*:\s*(.*?)\s*\n+\s*non_diegetic_music\s*:\s*(.*)",
        prompt,
    )
    if structured_match:
        integrated = structured_match.group(1).strip()
        shot_labels = re.findall(r"\[Shot (\d+)\]", integrated, flags=re.IGNORECASE)
        if shot_labels == ["1"]:
            next_shot = 1

            def label_unmarked_cut(match: re.Match[str]) -> str:
                nonlocal next_shot
                next_shot += 1
                return f"[Shot {next_shot}] {match.group(0)}"

            integrated = re.sub(
                r"\bAt\s+\d{2,}:\d{2}\.\d{3},?\s+(?=the\s+(?:camera|shot|scene)\s+(?:(?:hard\s+)?cuts|transitions|changes|switches|cross-dissolves|fades|wipes)\s+(?:to|into)\b)",
                label_unmarked_cut,
                integrated,
                flags=re.IGNORECASE,
            )
        prompt = (
            f"integrated_multimodal_description: {integrated}\n\n"
            f"overall_soundscape: {structured_match.group(2).strip()}\n\n"
            f"non_diegetic_music: {structured_match.group(3).strip()}"
        )

    return re.sub(
        r"(\[Shot (?:[2-9]|[1-9]\d+)\]\s+At\s+\d{2,}:\d{2}\.\d{3})(?!,)(?=\s+the\s+(?:camera|shot|scene)\b)",
        r"\1,",
        prompt,
        flags=re.IGNORECASE,
    )


def _minimax_h3_ref_definition_items(text: str) -> list[tuple[str, str]]:
    return [
        (f"<{kind.title()} {int(ordinal)}>", description.strip())
        for kind, ordinal, description in re.findall(
            r"(?im)^[ \t]*<[ \t]*(Subject|Picture|Video|Audio)[ \t]+(\d+)[ \t]*>"
            r"[ \t]*(?::|=|[-–—]|\bis\b)[ \t]*(\S.*)$",
            text or "",
        )
    ]


def _minimax_h3_ref2va_subject_count(prompt_text: Any) -> int:
    sections = _minimax_h3_ref_sections(
        _sanitize_structured_prompt_text(str(prompt_text or ""), "", 0)
    )
    if sections is None:
        return 0
    return len(
        {
            tag
            for tag, _description in _minimax_h3_ref_definition_items(
                sections["subject_definitions"]
            )
            if tag.startswith("<Subject ")
        }
    )


def _minimax_h3_ref_summary_task_types(user_prompt: str, reference_manifest: str) -> list[str]:
    manifest = _normalize_minimax_h3_reference_manifest(reference_manifest)
    definitions = _minimax_h3_reference_definitions(manifest)
    tags = [tag for tag, _description in definitions]
    combined = f"{user_prompt} {manifest}"
    task_types: list[str] = []

    if _has_unnegated_prompt_phrase(
        combined,
        r"\b(?:keyframes?|first[-\s]+frames?|last[-\s]+frames?|first\s+and\s+last\s+frames?|interpolat(?:e|ion|ing))\b",
    ):
        task_types.append("keyframe completion")
    if any(tag.startswith(("<Picture ", "<Video ")) for tag in tags):
        task_types.append("reference generation")
    if any(tag.startswith("<Video ") for tag in tags):
        if re.search(r"\b(?:continu(?:e|ation)|extend(?:ing)?|append|carry\s+on)\b", combined, flags=re.IGNORECASE):
            task_types.append("video continuation")
        elif re.search(r"\b(?:edit|replace|remove|modify|change|transform|restyl(?:e|ing)|rework)\b", combined, flags=re.IGNORECASE):
            task_types.append("video editing")
    if any(tag.startswith("<Audio ") for tag in tags):
        if re.search(r"\b(?:copy|reuse|retain|preserve|keep)\b[^.!?\n]{0,80}\b(?:audio|soundtrack|voice|music)\b", combined, flags=re.IGNORECASE):
            task_types.append("audio reuse")
        else:
            task_types.append("audio reference")
    return list(dict.fromkeys(task_types or ["reference generation"]))


def _repair_minimax_h3_ref_cut_timestamps(
    detailed: str,
    duration_seconds: float = 0.0,
) -> str:
    """Fill only unusable later-shot cut instants on a bounded local clock.

    Valid authored cut instants are retained.  When authored instants conflict,
    the maximum-size increasing subset is kept and only the unusable gaps are
    interpolated.  Shot labels, shot bodies, and shot count are never changed.
    """

    duration_milliseconds = int(
        round(max(0.0, _safe_float(duration_seconds, 0.0)) * 1000.0)
    )
    shot_pattern = re.compile(r"\[Shot (?P<shot>\d+)\]")
    shots = list(shot_pattern.finditer(detailed or ""))
    shot_numbers = [int(match.group("shot")) for match in shots]
    shot_count = len(shots)
    if (
        duration_milliseconds <= shot_count
        or shot_count <= 1
        or shot_numbers != list(range(1, shot_count + 1))
    ):
        return detailed

    authored_prefix_pattern = re.compile(
        r"\s+At\s+(?P<minutes>\d{2,}):(?P<seconds>\d{2})\."
        r"(?P<milliseconds>\d{3})(?P<comma>,?)",
        flags=re.IGNORECASE,
    )
    validator_prefix_pattern = re.compile(
        r"\s+At\s+(?P<minutes>\d{2,}):(?P<seconds>\d{2})\."
        r"(?P<milliseconds>\d{3}),"
    )
    candidates: list[tuple[int, int]] = []
    prefix_matches: dict[int, re.Match[str]] = {}
    canonical_prefixes: dict[int, bool] = {}
    for shot_index, marker in enumerate(shots[1:], start=2):
        authored = authored_prefix_pattern.match(detailed, marker.end())
        if authored is None:
            continue
        prefix_matches[shot_index] = authored
        canonical_prefixes[shot_index] = bool(
            validator_prefix_pattern.match(detailed, marker.end())
        )
        seconds = int(authored.group("seconds"))
        if seconds >= 60:
            continue
        authored_milliseconds = (
            int(authored.group("minutes")) * 60_000
            + seconds * 1000
            + int(authored.group("milliseconds"))
        )
        # An explicit authored cut at or beyond the clip boundary is a
        # semantic scheduling error, not a missing/local-format defect.  Keep
        # it intact so the strict validator reports the distinct
        # ``minimax_h3_cut_timestamp_out_of_range`` reason and the established
        # refinement path can re-plan it.
        if authored_milliseconds >= duration_milliseconds:
            return detailed
        # Leave one millisecond for every missing cut and for the terminal
        # video boundary.  H3 durations are much wider than this guard, but it
        # keeps interpolation exact after integer rounding.
        if (
            authored_milliseconds < shot_index - 1
            or duration_milliseconds - authored_milliseconds
            < shot_count - shot_index + 1
        ):
            continue
        candidates.append((shot_index, authored_milliseconds))

    # Longest increasing feasible subsequence: retain as many authored cut
    # instants as possible.  Equal-length ties prefer earlier shot anchors.
    paths: list[list[tuple[int, int]]] = []
    for shot_index, authored_milliseconds in candidates:
        best_path: list[tuple[int, int]] = []
        for previous_path in paths:
            previous_shot, previous_milliseconds = previous_path[-1]
            if (
                previous_shot < shot_index
                and authored_milliseconds - previous_milliseconds
                >= shot_index - previous_shot
                and len(previous_path) > len(best_path)
            ):
                best_path = previous_path
        paths.append([*best_path, (shot_index, authored_milliseconds)])
    selected_path: list[tuple[int, int]] = []
    if paths:
        maximum_length = max(len(path) for path in paths)
        selected_path = min(
            (path for path in paths if len(path) == maximum_length),
            key=lambda path: tuple(shot for shot, _milliseconds in path),
        )

    anchors = [(1, 0), *selected_path, (shot_count + 1, duration_milliseconds)]
    repaired_times: dict[int, int] = {
        shot_index: authored_milliseconds
        for shot_index, authored_milliseconds in selected_path
    }
    for (start_shot, start_milliseconds), (end_shot, end_milliseconds) in zip(
        anchors,
        anchors[1:],
    ):
        shot_span = end_shot - start_shot
        time_span = end_milliseconds - start_milliseconds
        for shot_index in range(start_shot + 1, end_shot):
            offset = shot_index - start_shot
            repaired_times[shot_index] = start_milliseconds + (
                time_span * offset + shot_span // 2
            ) // shot_span

    replacements: list[tuple[int, int, str]] = []
    selected_shots = {shot_index for shot_index, _milliseconds in selected_path}
    for shot_index, marker in enumerate(shots[1:], start=2):
        cut_milliseconds = repaired_times[shot_index]
        minutes, remainder = divmod(cut_milliseconds, 60_000)
        seconds, milliseconds = divmod(remainder, 1000)
        replacement = f" At {minutes:02d}:{seconds:02d}.{milliseconds:03d},"
        authored = prefix_matches.get(shot_index)
        if (
            shot_index in selected_shots
            and canonical_prefixes.get(shot_index, False)
        ):
            continue
        if authored is None:
            replacements.append((marker.end(), marker.end(), replacement))
        else:
            replacements.append((authored.start(), authored.end(), replacement))

    repaired = detailed
    for start, end, replacement in sorted(replacements, reverse=True):
        repaired = repaired[:start] + replacement + repaired[end:]
    return repaired


def _repair_minimax_h3_timed_cut_openers(timeline: str) -> str:
    """Canonicalize only timestamp-adjacent H3 cut transport syntax.

    The model-authored shot body remains byte-for-byte intact. When a later
    shot starts directly with action or camera motion, a neutral cut prefix is
    inserted so native H3 receives the delimiter its parser requires.
    """

    repaired = str(timeline or "")
    # The live T2VA failure omitted the comma only before an otherwise
    # unmistakable smash-cut opener (``At 00:06.000 the smash cuts to``).
    # Restore punctuation only for that exact transition family; ordinary
    # timestamp-adjacent prose remains untouched and will still fail closed.
    repaired = re.sub(
        r"(\[Shot (?:[2-9]|[1-9]\d+)\]\s+At\s+\d{2,}:\d{2}\.\d{3})(?!,)"
        r"(?=\s+(?:the\s+)?(?:a\s+)?smash(?:[- ]+cut|[- ]?cuts?)\s+to\b)",
        r"\1,",
        repaired,
        flags=re.IGNORECASE,
    )
    timestamp_prefix = (
        r"(\[Shot (?:[2-9]|[1-9]\d+)\]\s+At\s+\d{2,}:\d{2}\.\d{3},\s*)"
    )
    repaired = re.sub(
        timestamp_prefix
        + r"the\s+camera\s+smash(?:[- ]+cut|[- ]?cuts?)\s+to\s*:?\s*",
        r"\1the camera hard cuts to ",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(
        timestamp_prefix
        + r"(?:the\s+)?(?:a\s+)?smash(?:[- ]+cut|[- ]?cuts?)\s+to\s*:?\s*",
        r"\1the camera hard cuts to ",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(
        timestamp_prefix
        + r"the\s+camera\s+(?:quick|clean)\s+cuts?\s+to\s+",
        r"\1the camera cuts to ",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(
        timestamp_prefix + r"(?:a\s+)?hard\s+cut\s+to\s*:?\s*",
        r"\1the camera hard cuts to ",
        repaired,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(
        timestamp_prefix + r"(?:a\s+)?(?:quick|clean)?\s*cut\s+to\s*:?\s*",
        r"\1the camera cuts to ",
        repaired,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"(\[Shot (?:[2-9]|[1-9]\d+)\]\s+At\s+\d{2,}:\d{2}\.\d{3},)"
        r"(?!\s*the\s+(?:camera|shot|scene)\s+(?:(?:hard\s+)?cuts|transitions|changes|switches|cross-dissolves|fades|wipes)\s+(?:to|into)\b)"
        r"(\s*)",
        r"\1\2the shot cuts to: ",
        repaired,
        flags=re.IGNORECASE,
    )


def _repair_minimax_h3_ref_detailed_timeline(detailed: str, duration_seconds: float = 0.0) -> str:
    range_pattern = re.compile(
        r"\[Shot (?P<shot>\d+)\]\s*\(?\s*"
        r"(?P<start_major>\d{1,2}):(?P<start_minor>\d{2})(?:\.(?P<start_fraction>\d{1,3}))?\s*[-–—]\s*"
        r"(?P<end_major>\d{1,2}):(?P<end_minor>\d{2})(?:\.(?P<end_fraction>\d{1,3}))?\s*\)?\s*:?\s*",
        flags=re.IGNORECASE,
    )
    range_matches = list(range_pattern.finditer(detailed))
    duration = max(0.0, _safe_float(duration_seconds, 0.0))

    def range_time(match: re.Match[str], prefix: str, mode: str) -> float:
        major = int(match.group(f"{prefix}_major"))
        minor = int(match.group(f"{prefix}_minor"))
        fraction_text = match.group(f"{prefix}_fraction") or ""
        if mode == "minutes_seconds":
            if minor >= 60:
                return -1.0
            fraction = int(fraction_text.ljust(3, "0")) / 1000.0 if fraction_text else 0.0
            return major * 60.0 + minor + fraction
        if fraction_text or minor >= 100:
            return -1.0
        return major + minor / 100.0

    def valid_range_mode(mode: str) -> bool:
        previous_end: float | None = None
        for match in range_matches:
            start = range_time(match, "start", mode)
            end = range_time(match, "end", mode)
            if start < 0 or end <= start:
                return False
            if int(match.group("shot")) == 1 and start != 0:
                return False
            if previous_end is not None and abs(start - previous_end) > 0.01:
                return False
            if duration > 0 and (start >= duration or end > duration + 0.05):
                return False
            previous_end = end
        return bool(range_matches)

    range_mode = ""
    if valid_range_mode("minutes_seconds"):
        range_mode = "minutes_seconds"
    elif duration > 0 and valid_range_mode("seconds_hundredths"):
        range_mode = "seconds_hundredths"
    unsafe_explicit_range = bool(range_matches) and not range_mode
    if range_mode:
        def replace_range(match: re.Match[str]) -> str:
            shot = int(match.group("shot"))
            start = range_time(match, "start", range_mode)
            if shot == 1:
                return "[Shot 1] "
            minutes = int(start // 60)
            seconds = int(start % 60)
            milliseconds = int(round((start - int(start)) * 1000))
            if milliseconds == 1000:
                seconds += 1
                milliseconds = 0
                if seconds == 60:
                    minutes += 1
                    seconds = 0
            following = match.string[match.end() :]
            article = "a " if re.match(
                r"(?:close[- ]up|medium(?:[- ]wide)? shot|wide shot|long shot|high[- ]angle|low[- ]angle|bird['’]s[- ]eye)",
                following,
                flags=re.IGNORECASE,
            ) else ""
            return f"[Shot {shot}] At {minutes:02d}:{seconds:02d}.{milliseconds:03d}, the camera cuts to {article}"

        detailed = range_pattern.sub(replace_range, detailed)
        detailed = re.sub(
            r"(\[Shot (?:[2-9]|[1-9]\d+)\]\s+At\s+\d{2,}:\d{2}\.\d{3},\s*the camera cuts to (?:a )?)([A-Z])",
            lambda match: f"{match.group(1)}{match.group(2).casefold()}",
            detailed,
        )

    seconds_range_pattern = re.compile(
        r"\[Shot (?P<shot>\d+)\]\s*\(?\s*"
        r"(?P<start>\d+(?:\.\d{1,3})?)\s*s?\s*[-–—]\s*"
        r"(?P<end>\d+(?:\.\d{1,3})?)\s*s\s*\)?\s*:?\s*",
        flags=re.IGNORECASE,
    )
    seconds_range_matches = list(seconds_range_pattern.finditer(detailed))

    def valid_seconds_ranges() -> bool:
        previous_end: float | None = None
        for match in seconds_range_matches:
            start = float(match.group("start"))
            end = float(match.group("end"))
            if end <= start:
                return False
            if int(match.group("shot")) == 1 and abs(start) > 0.001:
                return False
            if previous_end is not None and abs(start - previous_end) > 0.01:
                return False
            if duration > 0 and (start >= duration or end > duration + 0.05):
                return False
            previous_end = end
        return bool(seconds_range_matches)

    seconds_ranges_valid = valid_seconds_ranges()
    unsafe_explicit_range = unsafe_explicit_range or (
        bool(seconds_range_matches) and not seconds_ranges_valid
    )
    if seconds_ranges_valid:
        def replace_seconds_range(match: re.Match[str]) -> str:
            shot = int(match.group("shot"))
            start = float(match.group("start"))
            if shot == 1:
                return "[Shot 1] "
            total_milliseconds = int(round(start * 1000.0))
            minutes, remainder = divmod(total_milliseconds, 60_000)
            seconds, milliseconds = divmod(remainder, 1000)
            following = match.string[match.end() :]
            article = "a " if re.match(
                r"(?:close[- ]up|medium(?:[- ]wide)? shot|wide shot|long shot|high[- ]angle|low[- ]angle|bird['’]s[- ]eye)",
                following,
                flags=re.IGNORECASE,
            ) else ""
            return f"[Shot {shot}] At {minutes:02d}:{seconds:02d}.{milliseconds:03d}, the camera cuts to {article}"

        detailed = seconds_range_pattern.sub(replace_seconds_range, detailed)
        detailed = re.sub(
            r"(\[Shot (?:[2-9]|[1-9]\d+)\]\s+At\s+\d{2,}:\d{2}\.\d{3},\s*the camera cuts to (?:a )?)([A-Z])",
            lambda match: f"{match.group(1)}{match.group(2).casefold()}",
            detailed,
        )
    detailed = re.sub(
        r"(\[Shot 1\])\s+At\s+00:00\.000,?\s*",
        r"\1 ",
        detailed,
        flags=re.IGNORECASE,
    )
    if not unsafe_explicit_range:
        detailed = _repair_minimax_h3_ref_cut_timestamps(detailed, duration)
    detailed = re.sub(
        r"(\[Shot (?:[2-9]|[1-9]\d+)\]\s+At\s+\d{2,}:\d{2}\.\d{3})(?!,)(?=\s+\S)",
        r"\1,",
        detailed,
        flags=re.IGNORECASE,
    )
    return _repair_minimax_h3_timed_cut_openers(detailed)


def _repair_minimax_h3_ref_subject_timeline_labels(sections: dict[str, str]) -> None:
    detailed = sections["detailed_description"]
    shot1_match = re.search(r"\[Shot 1\]", detailed, flags=re.IGNORECASE)
    if not shot1_match:
        return

    retention = sections["retention_analysis"]
    timeline = detailed[shot1_match.start() :]
    missing_subjects = [
        (tag, description)
        for tag, description in _minimax_h3_ref_definition_items(
            sections["subject_definitions"]
        )
        if tag.startswith("<Subject ") and tag not in timeline
    ]
    repairs: list[tuple[str, str]] = []
    implicit_environment_tag = ""
    for tag, description in missing_subjects:
        retention_match = re.search(
            rf"(?m)^\s*{re.escape(tag)}(?:\s*\([^\n)]*\))?\s*:\s*(?P<body>\S.*)$",
            retention,
        )
        if retention_match is None:
            return
        subject_semantics = f"{description} {retention_match.group('body')}"
        if not re.search(
            r"\b(?:environment|landscape|setting|world|location|background|scenery|production design|visual style|palette|"
            r"lighting|town|city|street|square|field|forest|room|interior|exterior)\b|"
            r"^\s*(?:the\s+|an?\s+)?(?:void\s+of\s+space|(?:outer|deep)\s+space|"
            r"(?:curved\s+)?horizon\s+of\s+(?:the\s+)?earth|earth(?:['’]s)?\s+(?:curved\s+)?(?:horizon|curve)|"
            r"(?:orbital|planetary)\s+(?:space|vista|scene)|starfield|starscape|night\s+sky)\b",
            subject_semantics,
            flags=re.IGNORECASE,
        ) or re.search(
            r"\b(?:person|character|courier|protagonist|antagonist|creature|performer|actor|man|woman|boy|girl|animal|monster)\b",
            description,
            flags=re.IGNORECASE,
        ):
            return
        source_tags = _minimax_h3_reference_tags(description)
        if len(source_tags) == 1 and source_tags[0] in timeline:
            repairs.append((tag, source_tags[0]))
            continue
        if (
            len(missing_subjects) == 1
            and not source_tags
            and re.search(
                r"\b(?:environment|landscape|setting|world|location|background|scenery|town|city|street|square|field|"
                r"forest|room|interior|exterior)\b|"
                r"^\s*(?:the\s+|an?\s+)?(?:void\s+of\s+space|(?:outer|deep)\s+space|"
                r"(?:curved\s+)?horizon\s+of\s+(?:the\s+)?earth|earth(?:['’]s)?\s+(?:curved\s+)?(?:horizon|curve)|"
                r"(?:orbital|planetary)\s+(?:space|vista|scene)|starfield|starscape|night\s+sky)\b",
                subject_semantics,
                flags=re.IGNORECASE,
            )
        ):
            implicit_environment_tag = tag
            continue
        return

    if repairs and len({source_tag for _tag, source_tag in repairs}) != len(repairs):
        return
    for tag, source_tag in repairs:
        timeline = timeline.replace(source_tag, f"{tag} (derived from {source_tag})", 1)
    if implicit_environment_tag:
        relative_end = shot1_match.end() - shot1_match.start()
        suffix = timeline[relative_end:]
        separator = "" if suffix.startswith((" ", "\n")) else " "
        timeline = (
            f"{timeline[:relative_end]} The setting is {implicit_environment_tag} as defined above."
            f"{separator}{suffix}"
        )
    if not repairs and not implicit_environment_tag:
        return
    sections["detailed_description"] = f"{detailed[: shot1_match.start()]}{timeline}"


def _repair_minimax_h3_ref_retention_rows(
    sections: dict[str, str],
    definition_items: list[tuple[str, str]] | None = None,
    reference_manifest: str = "",
) -> None:
    """Canonicalize only an incomplete or invalid Ref2VA retention table."""

    items = list(
        definition_items
        if definition_items is not None
        else _minimax_h3_ref_definition_items(sections.get("subject_definitions", ""))
    )
    if not items:
        return
    retention = sections.get("retention_analysis", "")
    manifest_definitions = dict(
        _minimax_h3_reference_definitions(
            _normalize_minimax_h3_reference_manifest(reference_manifest)
        )
    )

    def allowed_markers(kind: str) -> set[str]:
        return (
            _MINIMAX_H3_REF_AUDIO_RETENTION
            if kind == "Audio"
            else _MINIMAX_H3_REF_VISUAL_RETENTION
        )

    def default_marker(
        kind: str,
        tag: str,
        definition: str,
        authored_marker: str = "",
    ) -> str:
        if kind == "Subject":
            return "fully_preserved"
        if kind != "Audio":
            return "attribute_transfer"
        if authored_marker == "fully_preserved":
            return "fully_copy"
        if authored_marker == "partially_preserved":
            return "partially_copy"
        declared_semantics = (
            f"{definition} {manifest_definitions.get(tag, '')}"
        )
        if re.search(
            r"\b(?:exact|locked|copy|copied|reuse|reused|retain|retained|preserve|preserved)\b",
            declared_semantics,
            flags=re.IGNORECASE,
        ):
            return "fully_copy"
        return "reference"

    all_rows_valid = True
    for tag, _definition in items:
        kind_match = re.fullmatch(r"<(Subject|Picture|Video|Audio) \d+>", tag)
        if kind_match is None:
            all_rows_valid = False
            break
        marker_pattern = "|".join(
            re.escape(marker) for marker in sorted(allowed_markers(kind_match.group(1)))
        )
        if not re.search(
            rf"(?m)^\s*{re.escape(tag)}(?:\s*\([^\n)]*\))?\s*:\s*"
            rf"(?:{marker_pattern})\s+-\s*\S",
            retention,
        ):
            all_rows_valid = False
            break
    if all_rows_valid:
        return

    retention_lines: list[str] = []
    all_marker_pattern = (
        r"fully_preserved|partially_preserved|attribute_transfer|weak_reference|"
        r"fully_copy|partially_copy|reference"
    )
    for tag, definition in items:
        tag_match = re.fullmatch(r"<(Subject|Picture|Video|Audio) (\d+)>", tag)
        if tag_match is None:
            continue
        kind, ordinal = tag_match.groups()
        existing_match = re.search(
            rf"(?is)(?:<|\[)\s*{kind}\s+{ordinal}\s*(?:>|\])"
            rf"(?:\s*(?:\([^\n)]*\)|\[[^\n\]]*\]))*\s*:\s*"
            r"(.*?)(?=(?:<|\[)\s*(?:Subject|Picture|Video|Audio)\s+\d+\s*(?:>|\])|$)",
            retention,
        )
        existing = (
            existing_match.group(1).strip().strip("[]").strip()
            if existing_match
            else ""
        )
        marker_match = re.match(
            rf"^({all_marker_pattern})\s*-\s*(.*)$",
            existing,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if marker_match:
            marker = marker_match.group(1).casefold()
            retention_detail = marker_match.group(2).strip().strip("[]").strip()
        else:
            marker = default_marker(kind, tag, definition)
            retention_detail = re.sub(
                r"^[A-Za-z][A-Za-z_]{1,31}\s*-\s*",
                "",
                re.sub(r"^[^:]{0,48}:\s*", "", existing),
            ).strip().strip("[]").strip()
        if marker not in allowed_markers(kind):
            marker = default_marker(kind, tag, definition, marker)
        retention_lines.append(
            f"{tag}: {marker} - {retention_detail or definition}"
        )
    if retention_lines:
        sections["retention_analysis"] = "\n".join(retention_lines)


def _repair_minimax_h3_ref_manifest_definitions(
    sections: dict[str, str],
    reference_manifest: str = "",
) -> bool:
    """Restore only host-declared reference assets omitted by the model.

    The reference manifest is a validated workflow contract, so copying one of
    its missing Picture/Video/Audio declarations is deterministic contract
    completion rather than inferred creative content. Existing authored
    definitions remain byte-stable, and unknown prompt tags still fail strict
    validation.
    """

    subject_definitions = str(sections.get("subject_definitions", "") or "").strip()
    declared_tags = set(_minimax_h3_reference_tags(subject_definitions))
    missing_lines: list[str] = []
    for tag, description in _minimax_h3_reference_definitions(reference_manifest):
        if tag in declared_tags:
            continue
        clean_description = _strip_grounding_role_annotations(description).strip(" []")
        if not clean_description:
            continue
        missing_lines.append(f"{tag}: {clean_description}")
        declared_tags.add(tag)
    if not missing_lines:
        return False
    sections["subject_definitions"] = "\n".join(
        item for item in (subject_definitions, *missing_lines) if item
    )
    return True


def _repair_minimax_h3_single_subject_timeline_label(
    sections: dict[str, str],
) -> None:
    """Restore one unambiguous semantic Subject label without guessing roles."""

    subject_items = [
        (tag, description)
        for tag, description in _minimax_h3_ref_definition_items(
            sections.get("subject_definitions", "")
        )
        if tag.startswith("<Subject ")
    ]
    if len(subject_items) != 1:
        return
    subject_tag, description = subject_items[0]
    detailed = sections.get("detailed_description", "")
    shot1_match = re.search(r"\[Shot 1\]", detailed, flags=re.IGNORECASE)
    if shot1_match is None:
        return
    timeline = detailed[shot1_match.start() :]
    if subject_tag in timeline:
        return

    visual_sources = [
        tag
        for tag in _minimax_h3_reference_tags(description)
        if tag.startswith(("<Picture ", "<Video "))
    ]
    if len(visual_sources) == 1 and visual_sources[0] in timeline:
        timeline = timeline.replace(
            visual_sources[0],
            f"{subject_tag} (derived from {visual_sources[0]})",
            1,
        )
    else:
        relative_end = shot1_match.end() - shot1_match.start()
        timeline = (
            f"{timeline[:relative_end]} The shot establishes {subject_tag} as defined above;"
            f"{timeline[relative_end:]}"
        )
    sections["detailed_description"] = f"{detailed[: shot1_match.start()]}{timeline}"


def _repair_minimax_h3_ref_misplaced_detailed_description(
    sections: dict[str, str],
) -> bool:
    """Recover a generated timeline placed after retention rows.

    DiffusionGemma can omit only the ``detailed_description:`` heading while
    still authoring the style opening and every [Shot N] block.  The generic
    section parser then assigns that tail to ``retention_analysis``; rebuilding
    the retention table used to discard the complete timeline.  Move only an
    unambiguous post-table tail and otherwise leave the prompt fail-closed.
    """

    if str(sections.get("detailed_description", "") or "").strip():
        return False
    retention = str(sections.get("retention_analysis", "") or "")
    shot1_match = re.search(r"\[Shot 1\]", retention, flags=re.IGNORECASE)
    if shot1_match is None:
        return False
    marker_pattern = "|".join(
        re.escape(marker)
        for marker in sorted(
            _MINIMAX_H3_REF_VISUAL_RETENTION | _MINIMAX_H3_REF_AUDIO_RETENTION,
            key=len,
            reverse=True,
        )
    )
    row_pattern = re.compile(
        rf"(?im)^[ \t]*<\s*(?:Subject|Picture|Video|Audio)\s+[1-9]\d*\s*>"
        rf"(?:[ \t]*(?:\([^\n)]*\)|\[[^\n\]]*\]))?[ \t]*:[ \t]*"
        rf"(?:{marker_pattern})[ \t]+-[ \t]+\S[^\r\n]*$"
    )
    rows = [match for match in row_pattern.finditer(retention) if match.end() <= shot1_match.start()]
    if not rows:
        return False
    last_row_end = max(match.end() for match in rows)
    detailed_tail = retention[last_row_end:].strip()
    if not detailed_tail or re.search(r"\[Shot 1\]", detailed_tail, flags=re.IGNORECASE) is None:
        return False
    sections["retention_analysis"] = retention[:last_row_end].strip()
    sections["detailed_description"] = detailed_tail
    return True


def _repair_minimax_h3_ref_single_picture_subject_definition(
    sections: dict[str, str],
) -> bool:
    """Promote one authored Picture definition to one used semantic Subject.

    This is intentionally narrow: there must be no parsed Subject definition,
    exactly one Subject tag used by summary/timeline, and exactly one Picture
    definition with a substantive, identity-bearing authored description.
    Environment/style plates are never promoted into performers. Multi-subject
    or otherwise ambiguous prompts remain blocked for model repair.
    """

    definition_items = _minimax_h3_ref_definition_items(
        sections.get("subject_definitions", "")
    )
    if any(tag.startswith("<Subject ") for tag, _description in definition_items):
        return False
    used_text = "\n".join(
        (
            str(sections.get("summary", "") or ""),
            str(sections.get("detailed_description", "") or ""),
        )
    )
    used_subjects = {
        f"<Subject {int(ordinal)}>"
        for ordinal in re.findall(
            r"<\s*Subject\s+([1-9]\d*)\s*>",
            used_text,
            flags=re.IGNORECASE,
        )
    }
    picture_items = [
        (tag, description)
        for tag, description in definition_items
        if tag.startswith("<Picture ")
    ]
    if len(used_subjects) != 1 or len(picture_items) != 1:
        return False
    picture_tag, picture_description = picture_items[0]
    if len(re.findall(r"[A-Za-z0-9]{2,}", picture_description)) < 4:
        return False
    if not _has_unnegated_prompt_phrase(
        picture_description,
        r"\b(?:person|people|human|character|courier|protagonist|antagonist|creature|"
        r"performer|actor|actress|dancer|singer|musician|man|woman|male|female|lady|gentleman|"
        r"boy|girl|child|adult|humanoid|robot|android|animal|monster|hero|heroine)\b",
    ):
        return False
    subject_tag = next(iter(used_subjects))
    promoted_description = (
        f"{picture_description.rstrip().rstrip('.')} Reference source: {picture_tag}."
    )
    repaired_items: list[tuple[str, str]] = [(subject_tag, promoted_description)]
    repaired_items.extend(
        (tag, description)
        for tag, description in definition_items
        if tag != picture_tag
    )
    sections["subject_definitions"] = "\n".join(
        f"{tag}: {description}" for tag, description in repaired_items
    )
    return True


def _repair_minimax_h3_ref2va_contract_delimiters(
    prompt_text: Any,
) -> tuple[str, list[str]]:
    """Insert only omitted colons in otherwise exact Ref2VA contract lines."""

    prompt = str(prompt_text or "")
    repairs: list[str] = []

    def repair_definition_body(body: str) -> str:
        definition_pattern = re.compile(
            r"(?m)^(?P<tag>[ \t]*<\s*(?:Subject|Picture|Video|Audio)\s+[1-9]\d*\s*>)"
            r"(?P<gap>[ \t]+)(?P<description>\S.*)$",
            flags=re.IGNORECASE,
        )

        def insert_definition_colon(match: re.Match[str]) -> str:
            description = match.group("description")
            if re.match(r"(?:[:=]|[-–—](?:\s|$)|is\b)", description, flags=re.IGNORECASE):
                return match.group(0)
            if len(re.findall(r"[A-Za-z0-9]{2,}", description)) < 4:
                return match.group(0)
            canonical_tag = re.sub(r"\s+", " ", match.group("tag").strip())
            repairs.append(f"inserted_ref_definition_colon:{canonical_tag}")
            return f"{match.group('tag')}:{match.group('gap')}{description}"

        return definition_pattern.sub(insert_definition_colon, body)

    all_retention_markers = sorted(
        _MINIMAX_H3_REF_VISUAL_RETENTION | _MINIMAX_H3_REF_AUDIO_RETENTION,
        key=len,
        reverse=True,
    )
    marker_pattern = "|".join(re.escape(marker) for marker in all_retention_markers)

    def repair_retention_body(body: str) -> str:
        retention_pattern = re.compile(
            rf"(?m)^(?P<tag>[ \t]*<\s*(?P<kind>Subject|Picture|Video|Audio)\s+[1-9]\d*\s*>"
            rf"(?:[ \t]*(?:\([^\n)]*\)|\[[^\n\]]*\]))?)(?P<gap>[ \t]+)(?P<marker>{marker_pattern})"
            r"(?P<suffix>[ \t]+-[ \t]+\S.*)$",
            flags=re.IGNORECASE,
        )

        def insert_retention_colon(match: re.Match[str]) -> str:
            marker = match.group("marker").casefold()
            allowed = (
                _MINIMAX_H3_REF_AUDIO_RETENTION
                if match.group("kind").casefold() == "audio"
                else _MINIMAX_H3_REF_VISUAL_RETENTION
            )
            if marker not in allowed:
                return match.group(0)
            canonical_tag_match = re.search(
                r"<\s*(Subject|Picture|Video|Audio)\s+([1-9]\d*)\s*>",
                match.group("tag"),
                flags=re.IGNORECASE,
            )
            canonical_tag = (
                f"<{canonical_tag_match.group(1).title()} {int(canonical_tag_match.group(2))}>"
                if canonical_tag_match
                else match.group("tag").strip()
            )
            repairs.append(f"inserted_ref_retention_colon:{canonical_tag}")
            return (
                f"{match.group('tag')}:{match.group('gap')}{match.group('marker')}"
                f"{match.group('suffix')}"
            )

        return retention_pattern.sub(insert_retention_colon, body)

    section_specs = (
        ("subject_definitions", "summary", repair_definition_body),
        ("retention_analysis", "detailed_description", repair_retention_body),
    )
    replacements: list[tuple[int, int, str]] = []
    for field_name, next_field, repair_body in section_specs:
        match = re.search(
            rf"(?ms)(?:\A|\n\n){re.escape(field_name)}\s*:\s*\n"
            rf"(?P<body>.*?)\n\n{re.escape(next_field)}\s*:",
            prompt,
        )
        if match is None:
            continue
        body = match.group("body")
        repaired_body = repair_body(body)
        if repaired_body != body:
            replacements.append((match.start("body"), match.end("body"), repaired_body))
    for start, end, replacement in sorted(replacements, reverse=True):
        prompt = prompt[:start] + replacement + prompt[end:]
    return prompt, list(dict.fromkeys(repairs))


def _repair_minimax_h3_ref2va_structure(
    prompt_text: str,
    user_prompt: str = "",
    reference_manifest: str = "",
    duration_seconds: float = 0.0,
) -> str:
    prompt = _normalize_minimax_h3_prompt(prompt_text, "ref2va")
    prompt, _contract_delimiter_repairs = _repair_minimax_h3_ref2va_contract_delimiters(
        prompt
    )
    if not prompt:
        return ""
    timeline_duration = max(
        0.0,
        _safe_float(duration_seconds, 0.0) or _explicit_prompt_duration_seconds(user_prompt),
    )

    canonical_sections = _minimax_h3_ref_sections(prompt)
    if canonical_sections is not None:
        summary = canonical_sections["summary"].strip()
        summary_match = re.match(r"^\[([^\]\n]+)\]\s*(.*)$", summary, flags=re.DOTALL)
        summary_body = summary_match.group(2).strip() if summary_match else summary
        actual_task_types = [item.strip() for item in summary_match.group(1).split("+")] if summary_match else []
        expected_task_types = _minimax_h3_ref_summary_task_types(user_prompt, reference_manifest)
        if actual_task_types != expected_task_types:
            canonical_sections["summary"] = f"[{' + '.join(expected_task_types)}] {summary_body}".rstrip()
        _repair_minimax_h3_ref_single_picture_subject_definition(canonical_sections)
        _repair_minimax_h3_ref_manifest_definitions(
            canonical_sections,
            reference_manifest,
        )
        _repair_minimax_h3_ref_retention_rows(
            canonical_sections,
            reference_manifest=reference_manifest,
        )
        _repair_minimax_h3_ref_subject_timeline_labels(canonical_sections)
        _repair_minimax_h3_single_subject_timeline_label(canonical_sections)
        canonical_sections["detailed_description"] = _repair_minimax_h3_ref_detailed_timeline(
            canonical_sections["detailed_description"],
            timeline_duration,
        )
        repaired_prompt = _rebuild_minimax_h3_ref_prompt(canonical_sections)
        return repaired_prompt if repaired_prompt != prompt else prompt

    heading_pattern = re.compile(
        r"(?im)^[ \t]*(subject[ _-]*definitions|summary(?:[ \t]*\[[^\]\n]*\])?|retention[ _-]*analysis|"
        r"detail(?:ed)?[ _-]*description|overall[ _-]*soundscape|non[ _-]*diegetic[ _-]*music)"
        r"[ \t]*(?P<delimiter>:[ \t]*|(?=\r?$))"
    )
    matches = list(heading_pattern.finditer(prompt))
    if not matches:
        return prompt

    heading_names = [
        re.sub(
            r"[\s-]+",
            "_",
            re.sub(r"\s*\[[^\]\n]*\]\s*$", "", match.group(1)).strip().casefold(),
        )
        for match in matches
    ]
    heading_names = ["detailed_description" if name == "detail_description" else name for name in heading_names]
    prefix = prompt[: matches[0].start()].strip()
    has_colonless_heading = any(not match.group("delimiter").startswith(":") for match in matches)
    if has_colonless_heading and not (
        (not prefix and heading_names == list(_MINIMAX_H3_REF_FIELDS))
        or (
            prefix
            and heading_names
            == ["retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"]
        )
    ):
        return prompt
    if prefix and "subject_definitions" not in heading_names and "summary" not in heading_names:
        if heading_names not in (
            ["retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"],
            ["overall_soundscape", "non_diegetic_music"],
        ):
            return prompt
    if (
        not any(name.startswith("summary") for name in heading_names)
        and "subject_definitions" not in heading_names
        and not _minimax_h3_ref_definition_items(prefix)
    ):
        return prompt

    sections = {field_name: "" for field_name in _MINIMAX_H3_REF_FIELDS}
    for index, match in enumerate(matches):
        raw_name = re.sub(r"\s*\[[^\]\n]*\]\s*$", "", match.group(1))
        field_name = re.sub(r"[\s-]+", "_", raw_name.strip().casefold())
        if field_name == "detail_description":
            field_name = "detailed_description"
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        value = prompt[match.end() : value_end].strip()
        if field_name in sections:
            sections[field_name] = "\n".join(item for item in (sections[field_name], value) if item).strip()
    if has_colonless_heading and any(not sections[field_name] for field_name in heading_names):
        return prompt

    prefix_recovered = False
    if prefix and heading_names in (
        ["retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"],
        ["overall_soundscape", "non_diegetic_music"],
    ):
        prefix_blocks = [block.strip() for block in re.split(r"\n\s*\n", prefix) if block.strip()]
        retention_markers = _MINIMAX_H3_REF_VISUAL_RETENTION | _MINIMAX_H3_REF_AUDIO_RETENTION
        retention_pattern = re.compile(
            r"(?im)^\s*<\s*(?:Subject|Picture|Video|Audio)\s+\d+\s*>"
            rf"(?:\s*(?:\([^\n)]*\)|\[[^\n\]]*\]))*\s*:\s*(?:{'|'.join(sorted(retention_markers))})\s*-\s*\S"
        )

        def is_retention_block(block: str) -> bool:
            tagged_items = _minimax_h3_ref_definition_items(block)
            lines = [line for line in block.splitlines() if line.strip()]
            return bool(tagged_items) and len(retention_pattern.findall(block)) == len(tagged_items) == len(lines)

        def is_definition_block(block: str) -> bool:
            lines = [line for line in block.splitlines() if line.strip()]
            return bool(lines) and len(_minimax_h3_ref_definition_items(block)) == len(lines)

    if prefix and heading_names == ["retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"]:
        summary_index = next(
            (
                index
                for index, block in enumerate(prefix_blocks)
                if re.match(r"^\[(?!Shot\b)[^\]\n]+\]\s+\S", block, flags=re.IGNORECASE | re.DOTALL)
            ),
            None,
        )
        definition_blocks = prefix_blocks[:summary_index] if summary_index is not None else []
        if (
            summary_index
            and summary_index == len(prefix_blocks) - 1
            and all(is_definition_block(block) and not retention_pattern.search(block) for block in definition_blocks)
            and is_retention_block(sections["retention_analysis"])
            and re.search(r"\[Shot 1\]", sections["detailed_description"], flags=re.IGNORECASE)
            and sections["overall_soundscape"]
            and sections["non_diegetic_music"]
        ):
            sections["subject_definitions"] = "\n".join(definition_blocks)
            sections["summary"] = prefix_blocks[summary_index]
            prefix_recovered = True
        else:
            return prompt

    if prefix and heading_names == ["overall_soundscape", "non_diegetic_music"]:
        summary_index = next(
            (
                index
                for index, block in enumerate(prefix_blocks)
                if re.match(r"^\[(?!Shot\b)[^\]\n]+\]\s+\S", block, flags=re.IGNORECASE | re.DOTALL)
            ),
            None,
        )
        definition_blocks = prefix_blocks[:summary_index] if summary_index is not None else []
        retention_start = summary_index + 1 if summary_index is not None else 0
        retention_end = retention_start
        while retention_end < len(prefix_blocks) and is_retention_block(prefix_blocks[retention_end]):
            retention_end += 1
        detailed_blocks = prefix_blocks[retention_end:]
        detailed_shot_index = next(
            (
                index
                for index, block in enumerate(detailed_blocks)
                if re.search(r"\[Shot 1\]", block, flags=re.IGNORECASE)
            ),
            None,
        )
        stray_tagged_detail = bool(
            detailed_shot_index is not None
            and any(_minimax_h3_ref_definition_items(block) for block in detailed_blocks[:detailed_shot_index])
        )
        if (
            summary_index
            and all(is_definition_block(block) and not retention_pattern.search(block) for block in definition_blocks)
            and retention_end > retention_start
            and detailed_blocks
            and not stray_tagged_detail
            and re.search(r"\[Shot 1\]", "\n\n".join(detailed_blocks), flags=re.IGNORECASE)
        ):
            sections["subject_definitions"] = "\n".join(definition_blocks)
            sections["summary"] = prefix_blocks[summary_index]
            sections["retention_analysis"] = "\n".join(prefix_blocks[retention_start:retention_end])
            sections["detailed_description"] = "\n\n".join(detailed_blocks)
            prefix_recovered = True
        else:
            return prompt

    if prefix and not prefix_recovered:
        if not sections["subject_definitions"]:
            sections["subject_definitions"] = prefix
        elif _minimax_h3_ref_definition_items(prefix):
            sections["subject_definitions"] = f"{prefix}\n{sections['subject_definitions']}".strip()

    _repair_minimax_h3_ref_misplaced_detailed_description(sections)
    _repair_minimax_h3_ref_single_picture_subject_definition(sections)
    _repair_minimax_h3_ref_manifest_definitions(sections, reference_manifest)

    definition_items: list[tuple[str, str]] = []
    for tag, description in _minimax_h3_ref_definition_items(sections["subject_definitions"]):
        if tag not in {item_tag for item_tag, _item_description in definition_items}:
            definition_items.append((tag, description.strip(" []")))

    definition_text = "\n".join(f"{tag}: {description}" for tag, description in definition_items)
    if definition_items:
        sections["subject_definitions"] = definition_text

    summary = sections["summary"].strip()
    summary_match = re.match(r"^\[([^\]\n]+)\]\s*(.*)$", summary, flags=re.DOTALL)
    if summary_match:
        summary = summary_match.group(2).strip()
    task_prefix = " + ".join(_minimax_h3_ref_summary_task_types(user_prompt, reference_manifest))
    sections["summary"] = f"[{task_prefix}] {summary}".rstrip()

    _repair_minimax_h3_ref_retention_rows(
        sections,
        definition_items=definition_items,
        reference_manifest=reference_manifest,
    )

    _repair_minimax_h3_ref_subject_timeline_labels(sections)
    _repair_minimax_h3_single_subject_timeline_label(sections)
    sections["detailed_description"] = _repair_minimax_h3_ref_detailed_timeline(
        sections["detailed_description"],
        timeline_duration,
    )
    return _rebuild_minimax_h3_ref_prompt(sections)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    payload, _had_boundary = _extract_delimited_final_payload(text)
    cleaned = _strip_markdown(_strip_thinking(payload))
    candidates = [cleaned]
    candidates.append(re.sub(r"^\s*\{\s*(?=\{)", "", cleaned).strip())
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first >= 0 and last > first:
        candidates.append(cleaned[first : last + 1])
        candidates.append(re.sub(r"^\s*\{\s*(?=\{)", "", cleaned[first : last + 1]).strip())
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


class _DuplicatePromptPacketKey(ValueError):
    pass


def _reject_duplicate_prompt_packet_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicatePromptPacketKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_prompt_packet_json(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_prompt_packet_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {value}")
            ),
        )
    except _DuplicatePromptPacketKey:
        raise
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _remove_json_trailing_commas(text: str) -> tuple[str, int]:
    output: list[str] = []
    in_string = False
    escaped = False
    removed = 0
    index = 0
    while index < len(text):
        character = text[index]
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            output.append(character)
            index += 1
            continue
        if character == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                removed += 1
                index += 1
                continue
        output.append(character)
        index += 1
    return "".join(output), removed


def _append_missing_root_object_closer(text: str) -> str | None:
    stack: list[str] = []
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            stack.append(character)
        elif character in "]}":
            if not stack:
                return None
            opening = stack.pop()
            if (opening, character) not in {("[", "]"), ("{", "}")}:
                return None
    if in_string or escaped or stack != ["{"]:
        return None
    return text + "}"


def _deterministically_repaired_prompt_packet_json(
    raw_output: str,
) -> tuple[dict[str, Any] | None, list[str], bool]:
    """Parse a compiler packet with a narrow, value-preserving repair set.

    Grounding ledgers never use this parser. Duplicate keys are always rejected,
    and no claim, prompt section, provenance list, tag ordinal, or role is
    synthesized here.
    """

    original = str(raw_output or "")
    text = original.strip()
    repairs: list[str] = []
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
        repairs.append("removed_utf8_bom")

    try:
        exact = _strict_prompt_packet_json(text)
    except _DuplicatePromptPacketKey:
        return None, [], False
    if exact is not None:
        return exact, repairs, not repairs

    # Native thinking/final markers and one outer JSON fence are transport
    # wrappers; removing them cannot change values inside the JSON object.
    stripped_thinking = text
    thinking_patterns = (
        re.compile(r"\A<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL),
        re.compile(
            r"\A<\|thought\|>.*?<\|end_thought\|>\s*",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    )
    for pattern in thinking_patterns:
        while True:
            match = pattern.match(stripped_thinking)
            if not match:
                break
            stripped_thinking = stripped_thinking[match.end() :].lstrip()
            repairs.append("removed_native_thinking_wrapper")
    text = stripped_thinking

    delimited, had_boundary = _extract_delimited_final_payload(text)
    if had_boundary:
        text = delimited.strip()
        repairs.append("removed_final_answer_wrapper")

    fence_match = re.fullmatch(
        r"\s*```(?:json)?\s*(.*?)\s*```\s*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence_match:
        text = fence_match.group(1).strip()
        repairs.append("removed_outer_json_fence")

    prefix_match = re.fullmatch(
        r"\s*(?:here\s+is\s+(?:the\s+)?(?:final\s+)?json|final\s+json)\s*:\s*(\{.*\})\s*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if prefix_match:
        text = prefix_match.group(1).strip()
        repairs.append("removed_known_json_prefix")

    candidates: list[tuple[str, list[str]]] = [(text, list(repairs))]
    if text.startswith("{{"):
        candidates.append((text[1:].lstrip(), [*repairs, "removed_extra_outer_open_brace"]))
    if text.endswith(","):
        candidates.append((text[:-1].rstrip(), [*repairs, "removed_root_trailing_comma"]))

    expanded: list[tuple[str, list[str]]] = []
    for candidate, candidate_repairs in candidates:
        expanded.append((candidate, candidate_repairs))
        without_commas, removed_count = _remove_json_trailing_commas(candidate)
        if removed_count:
            expanded.append(
                (
                    without_commas,
                    [*candidate_repairs, f"removed_trailing_commas:{removed_count}"],
                )
            )
        closed = _append_missing_root_object_closer(candidate)
        if closed is not None:
            expanded.append((closed, [*candidate_repairs, "appended_missing_root_closer"]))
            closed_without_commas, closed_removed_count = _remove_json_trailing_commas(
                closed
            )
            if closed_removed_count:
                expanded.append(
                    (
                        closed_without_commas,
                        [
                            *candidate_repairs,
                            "appended_missing_root_closer",
                            f"removed_trailing_commas:{closed_removed_count}",
                        ],
                    )
                )
        if removed_count:
            closed_without_commas = _append_missing_root_object_closer(without_commas)
            if closed_without_commas is not None:
                expanded.append(
                    (
                        closed_without_commas,
                        [
                            *candidate_repairs,
                            f"removed_trailing_commas:{removed_count}",
                            "appended_missing_root_closer",
                        ],
                    )
                )

    seen: set[str] = set()
    for candidate, candidate_repairs in expanded:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = _strict_prompt_packet_json(candidate)
        except _DuplicatePromptPacketKey:
            return None, [], False
        if parsed is not None:
            return parsed, list(dict.fromkeys(candidate_repairs)), False
    return None, [], False


def _repair_single_h3_dialogue_closer_json_escape(
    raw_output: str,
    minimax_h3_mode: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Remove only an invalid JSON backslash before a native ``</d>`` tag.

    The recovery is limited to the model-facing one-key H3 envelope.  The
    resulting text must parse as strict JSON with no duplicate or extra packet
    keys, and its value must already contain a complete canonical H3 prompt.
    No prompt prose or dialogue delimiter is synthesized.
    """

    text = str(raw_output or "").strip()
    if not text:
        return None, []
    payload, had_boundary = _extract_delimited_final_payload(text)
    if had_boundary:
        text = payload.strip()
    if not re.match(
        r'\A\s*\{\s*"minimax_h3_prompt"\s*:',
        text,
        flags=re.DOTALL,
    ):
        return None, []
    packet_key_pattern = re.compile(
        r'"(?:ltx_prompt|ideogram_prompt|minimax_h3_prompt|negative_prompt|scene_segments|metadata|grounding_ledger)"\s*:',
        flags=re.IGNORECASE,
    )
    packet_keys = packet_key_pattern.findall(text)
    if (
        len(packet_keys) != 1
        or not re.fullmatch(
            r'"minimax_h3_prompt"\s*:',
            packet_keys[0],
            flags=re.IGNORECASE,
        )
    ):
        return None, []
    repaired_text, repair_count = re.subn(
        r"\\(?=</d>)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    if repair_count <= 0:
        return None, []
    try:
        parsed = _strict_prompt_packet_json(repaired_text)
    except _DuplicatePromptPacketKey:
        return None, []
    if (
        parsed is None
        or set(parsed) != {"minimax_h3_prompt"}
        or not isinstance(parsed.get("minimax_h3_prompt"), str)
        or not _extract_complete_minimax_h3_plain_prompt(
            parsed["minimax_h3_prompt"],
            minimax_h3_mode,
        )
    ):
        return None, []
    repairs = []
    if had_boundary:
        repairs.append("removed_final_answer_wrapper")
    repairs.append(f"removed_invalid_h3_dialogue_closer_json_escape:{repair_count}")
    return parsed, repairs


def _single_paragraph(text: str) -> str:
    return _strip_generation_artifacts(re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip())


def _ensure_terminal_period(text: str) -> str:
    value = _single_paragraph(text)
    if not value:
        return ""
    if value[-1] in ".!?":
        return value
    return f"{value}."


def _sentence_case(text: str) -> str:
    value = _single_paragraph(text)
    if value and value[0].islower():
        return value[0].upper() + value[1:]
    return value


def _clean_prompt_request_text(text: str) -> str:
    value = _single_paragraph(text)
    if _is_media_reference_only_request(value):
        return ""
    patterns = [
        r"^(?:create|write|generate|make)\s+(?:an?\s+)?(?:ltx(?:[- ]?2(?:\.(?:3|5))?)?\s+)?(?:script|prompt)\s+for\s+(?:image|video)\s+generation\s+(?:featuring|with|of|about)\s+",
        r"^(?:create|write|generate|make)\s+(?:an?\s+)?(?:ltx(?:[- ]?2(?:\.(?:3|5))?)?\s+)?(?:script|prompt)\s+(?:featuring|with|of|about)\s+",
        r"^(?:create|write|generate|make)\s+(?:an?\s+)?(?:image|video)\s+(?:generation\s+)?(?:script|prompt)\s+(?:featuring|with|of|about)\s+",
    ]
    cleaned = value
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned).strip()
    return _normalize_hex_text(_sentence_case(cleaned or value))


def _clean_visual_description(text: str, max_chars: int = 3000) -> str:
    value = _strip_generation_artifacts(_strip_markdown(_strip_thinking(str(text or ""))))
    value = re.sub(r"\s+", " ", value).strip()
    if max_chars > 0 and len(value) > max_chars:
        value = value[:max_chars].rsplit(" ", 1)[0].strip() or value[:max_chars].strip()
    return value


def _visual_description_from_metadata(media_metadata: dict[str, Any] | None) -> str:
    if not isinstance(media_metadata, dict):
        return ""
    return _clean_visual_description(str(media_metadata.get("visual_description", "")))


def _normalize_media_synthesis_mode(value: Any) -> str:
    text = str(value or "video_recreation").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"image_identity_video_control", "identity_image_video_control", "image_identity", "identity_reference"}:
        return "image_identity_video_control"
    return "video_recreation"


def _is_image_identity_video_control(media_metadata: dict[str, Any] | None) -> bool:
    return _normalize_media_synthesis_mode(
        media_metadata.get("media_synthesis_mode") if isinstance(media_metadata, dict) else ""
    ) == "image_identity_video_control"


def _is_minimax_h3_reference_context(media_metadata: dict[str, Any] | None) -> bool:
    if not isinstance(media_metadata, dict):
        return False
    value = str(media_metadata.get("minimax_h3_mode", "")).strip().lower().replace("-", "_")
    return value in {"ref2va", "ref2v", "r2v", "reference", "reference_to_video", "reference_to_audio_video"}


def _attached_reference_image_count(media_metadata: dict[str, Any] | None) -> int:
    """Return only reference images that are present in the processor batch.

    Older image+video contexts predate the explicit attachment flag, so their
    declared count remains authoritative. Newer contexts fail closed when the
    sampler records that a declared still could not be attached.
    """

    if not isinstance(media_metadata, dict):
        return 0
    source = str(media_metadata.get("source", "none") or "none").strip().lower()
    if _is_minimax_h3_reference_context(media_metadata):
        declared = int(
            max(
                0.0,
                _safe_float(
                    media_metadata.get(
                        "minimax_h3_reference_image_batch_count",
                        media_metadata.get("reference_image_count", 0),
                    ),
                    0.0,
                ),
            )
        )
    else:
        declared = int(max(0.0, _safe_float(media_metadata.get("reference_image_count"), 0.0)))

    if "reference_image_backend_attached" in media_metadata:
        if not bool(media_metadata.get("reference_image_backend_attached")):
            return 0
    elif source not in {"image", "image+video"}:
        return 0

    if source == "image+video" and declared == 0 and not _is_minimax_h3_reference_context(media_metadata):
        # Compatibility for legacy ordinary image+video contexts.
        return 1
    return declared


def _synthesis_contract_text(media_metadata: dict[str, Any] | None) -> str:
    if _is_image_identity_video_control(media_metadata):
        return (
            "Use the reference image as the identity and appearance source for the output subject, including face, hair, body type, wardrobe, accessories, styling, and distinguishing visual traits. "
            "Use the video frames only as control structure for pose, depth, canny/edge layout, composition, action, timing, camera motion, blocking, and scene geometry. "
            "Do not copy the control-video subject's clothing, hair, face, body type, accessories, or style unless the user explicitly asks for those video-subject traits."
        )
    return "Recreate the source video or media evidence directly unless the user prompt asks for a different transformation."


def _copy_synthesis_metadata(target: dict[str, Any], media_metadata: dict[str, Any]) -> None:
    for key in (
        "media_synthesis_mode",
        "image_role",
        "video_role",
        "reference_image_count",
        "reference_image_backend_attached",
        "reference_image_resized_for_video_control",
        "reference_image_resize_mode",
        "reference_image_original_width",
        "reference_image_original_height",
        "reference_image_backend_width",
        "reference_image_backend_height",
        "video_sampled_frame_count",
        "synthesis_contract",
        "ltx25_context_schema",
        "ltx_generation_mode_hint",
        "ltx_first_frame_attached",
        "ltx_last_frame_attached",
        "ltx_frame_pair_attached",
        "ltx_frame_roles",
        "image_roles",
        "ltx_first_frame_width",
        "ltx_first_frame_height",
        "ltx_last_frame_width",
        "ltx_last_frame_height",
    ):
        if key in media_metadata:
            target[key] = media_metadata[key]


def _media_source_has_pixels(media_metadata: dict[str, Any] | None) -> bool:
    if not isinstance(media_metadata, dict):
        return False
    source = str(media_metadata.get("source", "none")).lower()
    return source in {"image", "video", "image+video"} or bool(media_metadata.get("sampled_frame_count"))


def _media_pixels_sent_to_backend(media_metadata: dict[str, Any] | None) -> bool:
    return bool(isinstance(media_metadata, dict) and media_metadata.get("pixels_sent_to_backend"))


def _media_grounding_mode(media_metadata: dict[str, Any] | None) -> str:
    if isinstance(media_metadata, dict) and isinstance(media_metadata.get("verified_grounding_ledger"), dict):
        return "verified_ledger"
    if not _media_source_has_pixels(media_metadata):
        return "none"
    if _media_pixels_sent_to_backend(media_metadata):
        return "pixels"
    if _visual_description_from_metadata(media_metadata):
        return "visual_description"
    return "metadata_only"


def _video_transport_mode() -> str:
    value = os.environ.get("DG_VIDEO_TRANSPORT", "sampled_frame_images").strip().lower()
    value = value.replace("-", "_").replace(" ", "_")
    if value in {"video", "video_token", "video_tokens", "native_video"}:
        return "video_tokens"
    return "sampled_frame_images"


def _video_frame_timecodes(media_metadata: dict[str, Any] | None, frame_count: int) -> list[float]:
    if frame_count <= 0:
        return []
    if not isinstance(media_metadata, dict):
        return [float(i) for i in range(frame_count)]
    indices = media_metadata.get("sampled_indices")
    source_fps = _safe_float(media_metadata.get("source_fps"), 0.0)
    if isinstance(indices, list) and source_fps > 0:
        values: list[float] = []
        for index in indices[:frame_count]:
            try:
                values.append(float(index) / source_fps)
            except Exception:
                values.append(float(len(values)))
        if len(values) == frame_count:
            return values
    duration = _safe_float(media_metadata.get("duration_seconds"), 0.0)
    if duration > 0 and frame_count > 1:
        step = duration / max(1, frame_count - 1)
        return [idx * step for idx in range(frame_count)]
    sample_fps = max(0.001, _safe_float(media_metadata.get("sample_fps"), 1.0))
    return [idx / sample_fps for idx in range(frame_count)]


def _display_video_frame_count(media_metadata: dict[str, Any] | None, frame_count: int) -> int:
    if not isinstance(media_metadata, dict):
        return frame_count
    video_count = int(max(0, _safe_float(media_metadata.get("video_sampled_frame_count"), 0.0)))
    if video_count > 0:
        return min(frame_count, video_count)
    return frame_count


def _sampled_video_frame_sequence_intro(media_metadata: dict[str, Any] | None, frame_count: int) -> str:
    reference_count = min(frame_count, _attached_reference_image_count(media_metadata))
    video_frame_count = _display_video_frame_count(media_metadata, max(0, frame_count - reference_count))
    timecodes = _video_frame_timecodes(media_metadata, video_frame_count)
    shown = ", ".join(f"{idx + 1}={seconds:.2f}s" for idx, seconds in enumerate(timecodes[:24]))
    if video_frame_count > 24:
        shown = f"{shown}, ..."
    source = str(media_metadata.get("source", "video")) if isinstance(media_metadata, dict) else "video"
    if _is_minimax_h3_reference_context(media_metadata):
        picture_intro = (
            f"The first {reference_count} attached image(s) are ordered MiniMax H3 reference pictures <Picture 1> through <Picture {reference_count}>. "
            if reference_count
            else "No reference pictures are attached for pixel analysis. "
        )
        return (
            f"{picture_intro}The remaining images are ordered sampled frames from <Video 1> at "
            f"{shown or 'unknown'}; they are analysis evidence, not additional Picture labels. Describe each reference only for the role "
            "declared in the H3 reference manifest, and do not cross-wire identity, style, motion, camera, composition, or temporal roles."
        )
    if source == "image+video" and _is_image_identity_video_control(media_metadata):
        if reference_count == 1:
            lead = (
                "The first attached image is the identity reference for the output subject. "
                "The remaining attached images are ordered sampled video control frames."
            )
        elif reference_count > 1:
            lead = (
                f"The first {reference_count} attached images are identity references for the output subject. "
                "The remaining attached images are ordered sampled video control frames."
            )
        else:
            lead = (
                "No identity reference image is attached; every attached image is an ordered sampled video control frame."
            )
        return (
            f"{lead} Video frame times: {shown or 'unknown'}. "
            "Describe the reference-image subject performing the video action/control structure. "
            "Use the identity image for face, hair, body type, wardrobe, accessories, styling, and distinguishing visual traits. "
            "Use the video frames for pose, depth, canny/edge layout, blocking, action, timing, composition, camera motion, and scene geometry. "
            "Do not copy the control-video subject's clothing, hair, face, body type, accessories, or styling into the prompt unless the user explicitly asks for those video-subject traits. "
            "Do not let the identity or appearance of the person in the control video override the reference-image subject. "
            "Preserve the opening framing, any visible camera movement or stable hold during the clip, and the closing framing."
        )
    if source == "image+video":
        lead = "The attached images are an optional still reference followed by ordered sampled frames from one video."
    else:
        lead = "The attached images are ordered sampled frames from one video."
    return (
        f"{lead} Frame times: {shown or 'unknown'}. Treat the images as one temporal sequence. "
        "Ground the answer only in visible subjects, clothing, setting, action, camera behavior, and motion that appear in these frames. "
        "Preserve the opening framing, any visible camera movement or stable hold during the clip, and the closing framing. "
        "If the camera is mostly static, say that rather than inventing a zoom, pan, reveal, exit, or wider pull-back. "
        "Do not invent water, animals, vehicles, dialogue, locations, or extra characters unless they are visibly present."
    )


def _ltx25_conditioning_frame_intro(media_metadata: dict[str, Any] | None) -> str:
    metadata = media_metadata if isinstance(media_metadata, dict) else {}
    first_attached = metadata.get("ltx_first_frame_attached") is True
    last_attached = metadata.get("ltx_last_frame_attached") is True
    if first_attached and last_attached:
        return (
            "The first attached image is the exact LTX-2.5 FIRST FRAME anchor and the second attached image is the exact LAST FRAME anchor. "
            "Inspect and preserve each frame's subjects and count, identity and appearance, environment, time of day, lighting, color, spatial composition, shot scale, and camera geometry at that endpoint. "
            "A still endpoint does not by itself prove a static camera between anchors. Describe one physically continuous transition from the first anchor to the last; never swap their order or treat them as generic style references."
        )
    if first_attached:
        return (
            "The attached image is the exact LTX-2.5 FIRST FRAME anchor. Inspect and preserve its subjects and count, identity and appearance, environment, time of day, lighting, color, spatial composition, shot scale, and opening camera geometry before describing plausible motion from it. "
            "The still fixes the first instant, not camera immobility afterward."
        )
    if last_attached:
        return (
            "The attached image is an LTX-2.5 LAST FRAME anchor, but no first-frame anchor is attached. Record its visible facts without pretending that a valid first-and-last-frame conditioning pair exists."
        )
    return ""


def _media_grounding_instruction(media_metadata: dict[str, Any] | None) -> str:
    mode = _media_grounding_mode(media_metadata)
    visual_description = _visual_description_from_metadata(media_metadata)
    capture_facts = (
        "Also preserve visual capture facts such as exposure level, blown or clipped highlights, flash/direct light, "
        "bloom, halation, lens blur, bokeh, depth of field, focal length impression, perspective compression, lens distortion, "
        "motion blur, grain/noise, white balance, contrast, dynamic range, and camera angle."
    )
    if mode == "verified_ledger":
        ledger = media_metadata.get("verified_grounding_ledger", {}) if isinstance(media_metadata, dict) else {}
        return (
            "The host-validated grounding ledger below is the sole visual evidence for this compiler pass. Use only "
            "observed_facts by their fact_id when describing source content. Keep inferred_facts, creative_additions, "
            "and user-requested target details distinct; never promote them into source observations. No pixels are "
            f"attached to this pass. Verified ledger: {_json_dumps(ledger)}"
        )
    if mode == "pixels":
        ltx_frame_intro = _ltx25_conditioning_frame_intro(media_metadata)
        if ltx_frame_intro:
            return f"{ltx_frame_intro} {capture_facts}"
        source = str(media_metadata.get("source", "none")).lower() if isinstance(media_metadata, dict) else "none"
        transport = str(media_metadata.get("transformers_video_transport", _video_transport_mode())) if isinstance(media_metadata, dict) else _video_transport_mode()
        if source in {"video", "image+video"} and transport == "sampled_frame_images":
            if _is_minimax_h3_reference_context(media_metadata):
                picture_count = int(max(0.0, _safe_float(media_metadata.get("minimax_h3_reference_image_batch_count"), 0.0)))
                return (
                    f"MiniMax H3 full-reference evidence is attached in declared order: the first {picture_count} image(s) map to "
                    "<Picture N>, followed by sampled analysis frames from <Video 1>. Use each asset only for its manifest role; do not "
                    "cross-wire subject identity, wardrobe, environment, style, action, camera, cut rhythm, composition, or temporal structure. "
                    "The sampled video frames are not additional Picture labels. "
                    f"{capture_facts}"
                )
            if _is_image_identity_video_control(media_metadata):
                return (
                    "Image+video synthesis mode is active. The first attached image is the identity reference for the output subject; "
                    "the ordered video frames are the control source for pose, depth, canny/edge layout, blocking, action, timing, composition, camera motion, and scene geometry. "
                    "Describe the reference-image subject performing the video action/control structure. The reference image controls face, hair, body type, wardrobe, accessories, styling, and distinguishing visual traits. "
                    "Do not copy the control-video subject's clothing, hair, face, body type, accessories, or styling unless explicitly requested, and do not let the video subject's identity dominate the output. "
                    "Preserve source-video chronology, opening framing, visible camera movement or stable hold, and closing framing while transferring the identity from the reference image. "
                    f"{capture_facts}"
                )
            return (
                "Ordered sampled video frames are attached through the image pathway. Use them as a time sequence, "
                "and base the prompt only on visible facts that appear in those frames. Treat time of day, ambient brightness, "
                "visible light sources, shadows, contrast, and exposure as source-identity facts to preserve. "
                "Preserve the source clip chronologically: opening framing, any visible camera movement or stable hold through the middle, and the closing framing. "
                "If the camera is mostly static, say that. Do not invent zooms, pans, reveals, exits, wider blocking, or lens shifts that are not visibly present. "
                f"{capture_facts}"
            )
        return (
            "Media pixels are attached to the backend request. Use visible facts from those pixels, and do not invent details beyond them. "
            "Treat time of day, ambient brightness, visible light sources, shadows, contrast, and exposure as source-identity facts to preserve. "
            f"{capture_facts}"
        )
    if mode == "visual_description":
        return (
            "Media pixels are not visible to the active backend. Treat this visual_description as the only visual evidence, including any stated time of day, ambient brightness, light sources, shadows, contrast, and exposure: "
            f"{visual_description}. {capture_facts}"
        )
    if mode == "metadata_only":
        return (
            "Image/video tensors were supplied, but the active backend cannot inspect pixels and no visual_description was supplied. "
            "Do not describe the image/video as if seen. Do not use generic phrases such as 'provided scene' or 'high-fidelity production design'. "
            "Use only concrete visual facts present in the user's text; if the user only asks to describe the media, report the limitation in metadata."
        )
    return "No image/video media is attached."


def _is_media_reference_only_request(text: str) -> bool:
    value = _single_paragraph(text).lower()
    if not value:
        return True
    media_words = ("image", "photo", "picture", "video", "clip", "frame", "still", "scene", "media")
    ask_words = ("describe", "analyze", "caption", "interrogate", "what is", "what's", "turn this", "use this", "from this")
    if not any(word in value for word in media_words) or not any(word in value for word in ask_words):
        return False
    descriptive_words = re.findall(r"[a-z0-9][a-z0-9'-]{2,}", value)
    filler = {
        "describe", "analyze", "caption", "interrogate", "what", "this", "that", "image", "photo",
        "picture", "video", "clip", "frame", "still", "scene", "media", "turn", "into", "from",
        "use", "prompt", "ltx", "ideogram", "the", "please", "make", "create", "generate",
    }
    specific = [word for word in descriptive_words if word not in filler]
    return len(specific) <= 2


def _looks_like_unseen_media_bluff(text: str) -> bool:
    value = _single_paragraph(text).lower()
    if not value:
        return False
    phrases = [
        "provided scene",
        "essence of the provided",
        "high-fidelity production design",
        "soft, natural light filters through the atmosphere",
        "subtle, realistic movements",
        "intricate details of their wardrobe",
        "professional film-production aesthetic",
        "balanced color grading",
        "realistic atmospheric depth",
    ]
    hits = sum(1 for phrase in phrases if phrase in value)
    return hits >= 2


def _truthy_metadata(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    value = _single_paragraph(text).lower()
    return any(phrase in value for phrase in phrases)


def _has_visual_claim_not_in_user(prompt: str, user_prompt: str, phrases: tuple[str, ...]) -> bool:
    prompt_value = _single_paragraph(prompt).lower()
    user_value = _single_paragraph(user_prompt).lower()
    return any(phrase in prompt_value and phrase not in user_value for phrase in phrases)


def _contains_daylight_claim(text: str) -> bool:
    return _contains_any_phrase(
        text,
        (
            "daylight",
            "broad daylight",
            "sunlit",
            "sun-lit",
            "sunny",
            "blue sky",
            "clear sky",
            "morning light",
            "afternoon light",
            "golden hour",
        ),
    )


def _contains_night_evidence(text: str) -> bool:
    return _contains_any_phrase(
        text,
        (
            "night",
            "nighttime",
            "after dark",
            "dark street",
            "low light",
            "moonlit",
            "neon",
            "streetlight",
        ),
    )


def _contains_static_camera_evidence(text: str) -> bool:
    return _contains_any_phrase(
        text,
        (
            "static camera",
            "locked off",
            "locked-off",
            "fixed camera",
            "stationary camera",
            "tripod",
            "stable hold",
            "no camera movement",
            "no pan",
            "no zoom",
        ),
    )


def _contains_camera_motion_claim(text: str) -> bool:
    return _contains_any_phrase(
        text,
        (
            "camera pans",
            "camera pan",
            "panning",
            "slow pan",
            "camera tilts",
            "tilting",
            "zoom",
            "zooms",
            "push in",
            "pushes in",
            "pull back",
            "pulls back",
            "pull-back",
            "dolly",
            "tracking shot",
            "camera tracks",
            "crane shot",
            "reveal",
        ),
    )


def _verified_prompt_claims(
    *,
    prompt_text: str,
    context: GemmaContext,
    media_metadata: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    prompt_text = _single_paragraph(_sanitize_prompt_text(prompt_text, "", 8000))
    if not prompt_text:
        reasons.append("empty_generation_prompt")
        return False, reasons

    grounding_mode = _media_grounding_mode(media_metadata)
    user_prompt = _single_paragraph(_sanitize_prompt_text(context.user_prompt, "", 4000))
    visual_description = _clean_visual_description(
        context.visual_description or str(media_metadata.get("visual_description", ""))
    )
    if grounding_mode == "metadata_only":
        claim_phrases = (
            "daylight",
            "sunlit",
            "sunny",
            "blue sky",
            "night",
            "neon",
            "streetlight",
            "person",
            "character",
            "camera",
            "lens",
            "close-up",
            "wide shot",
            "foreground",
            "background",
        )
        if _is_media_reference_only_request(user_prompt):
            reasons.append("metadata_only_media_reference_request")
        elif _looks_like_unseen_media_bluff(prompt_text):
            reasons.append("metadata_only_unseen_media_bluff")
        elif _has_visual_claim_not_in_user(prompt_text, user_prompt, claim_phrases):
            reasons.append("metadata_only_unverified_visual_claim")

    if visual_description:
        if _contains_night_evidence(visual_description) and _contains_daylight_claim(prompt_text):
            reasons.append("visual_description_contradiction_daylight_claim")
        if _contains_static_camera_evidence(visual_description) and _contains_camera_motion_claim(prompt_text):
            reasons.append("visual_description_contradiction_camera_motion")

    return not reasons, reasons


def _has_unnegated_prompt_phrase(text: str, pattern: str) -> bool:
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        clause_start = max(
            text.rfind(".", 0, match.start()),
            text.rfind(";", 0, match.start()),
            text.rfind("!", 0, match.start()),
            text.rfind("?", 0, match.start()),
            text.rfind("\n", 0, match.start()),
        )
        prefix = text[clause_start + 1 : match.start()]
        redirects = list(
            re.finditer(
                r"\b(?:but|instead|rather)\b|,\s*(?:use|make|keep|choose|render)\b",
                prefix,
                flags=re.IGNORECASE,
            )
        )
        if redirects:
            prefix = prefix[redirects[-1].end() :]
        if re.search(
            r"\b(?:no|not|never|without|avoid(?:ing)?|exclude(?:d|s|ing)?|do\s+not|don't|must\s+not|mustn't|cannot|can't)\b",
            prefix,
            flags=re.IGNORECASE,
        ):
            continue
        return True
    return False


def _has_negated_prompt_phrase(text: str, pattern: str) -> bool:
    for negation in re.finditer(
        r"\b(?:no|not|never|without|avoid(?:ing)?|exclude(?:d|s|ing)?|do\s+not|don't|must\s+not|mustn't|cannot|can't)\b",
        text,
        flags=re.IGNORECASE,
    ):
        if re.match(r"\s+only\b", text[negation.end() :], flags=re.IGNORECASE):
            continue
        sentence_end = re.search(r"[.!?\n]", text[negation.end() :])
        span_end = negation.end() + (sentence_end.start() if sentence_end else 220)
        if re.search(pattern, text[negation.end() : min(len(text), span_end, negation.end() + 220)], flags=re.IGNORECASE):
            return True
    return False


def _minimax_h3_battle_intent_requested(text: str) -> bool:
    """Return true for combat intent, not compound character/style labels."""

    source = str(text or "")
    scrubbed = list(source)
    for match in re.finditer(r"\b(?:battle|fight|duel|combat|clash)\b", source, flags=re.IGNORECASE):
        before = source[match.start() - 1] if match.start() else ""
        after = source[match.end()] if match.end() < len(source) else ""
        descriptor_role = bool(
            re.match(
                r"\s+(?:cyborg|robot|android|automaton|armor|armour|suit|gear|form|mode|"
                r"costume|outfit|design|machine|vehicle|bear)\b",
                source[match.end() :],
                flags=re.IGNORECASE,
            )
        )
        if before in "-_/" or after in "-_/" or descriptor_role:
            scrubbed[match.start() : match.end()] = " " * (match.end() - match.start())
    return _has_unnegated_prompt_phrase(
        "".join(scrubbed),
        r"\b(?:battle|fight|duel|combat|clash)\b",
    )


_MINIMAX_H3_LIVE_ACTION_PATTERN = r"\b(?:live[- ]action|practical\s+(?:film|photography)|photoreal(?:istic|ism)?|realistic\s+film\s+photography)\b"
_MINIMAX_H3_STOP_MOTION_PATTERN = r"\b(?:stop[- ]motion|claymation|clay[- ]animation|puppet[- ]animation)\b"
_MINIMAX_H3_3D_ANIMATION_PATTERN = r"\b(?:3[ -]?d(?:\s+(?:cg|cgi|animation|animated|render(?:ed|ing)?))?|cgi?(?:[- ](?:animated|animation|render(?:ed|ing)?))?|computer[- ](?:animated|generated)|cg[- ]animation)\b"
_MINIMAX_H3_2D_ANIMATION_PATTERN = r"\b(?:2[ -]?d(?:\s+(?:animation|animated|art))?|hand[- ]drawn|cel[- ](?:animation|animated|shaded)|traditional[- ]animation|anime|cartoon)\b"
_MINIMAX_H3_ANIMATION_PATTERN = r"\b(?:animated|animation)\b"


def _minimax_h3_requested_visual_medium(user_prompt: str) -> str:
    text = _single_paragraph(_sanitize_prompt_text(user_prompt, "", 8000))
    if not text:
        return ""
    if _has_unnegated_prompt_phrase(text, _MINIMAX_H3_3D_ANIMATION_PATTERN):
        return "3d_animation"
    if _has_unnegated_prompt_phrase(text, _MINIMAX_H3_STOP_MOTION_PATTERN):
        return "stop_motion"
    if _has_unnegated_prompt_phrase(text, _MINIMAX_H3_LIVE_ACTION_PATTERN):
        return "live_action"
    if _has_unnegated_prompt_phrase(text, _MINIMAX_H3_2D_ANIMATION_PATTERN):
        return "2d_animation"
    if _has_unnegated_prompt_phrase(text, _MINIMAX_H3_ANIMATION_PATTERN):
        if re.search(r"\bpok[eé]mon(?:[- ]inspired)?\b", text, flags=re.IGNORECASE):
            return "2d_animation"
        return "animation"
    return ""


def _minimax_h3_requested_style_locks(user_prompt: str) -> list[str]:
    text = _single_paragraph(_sanitize_prompt_text(user_prompt, "", 8000))
    locks: list[str] = []
    for match in re.finditer(r"\b([A-Za-z0-9][A-Za-z0-9'’]*)[- ]inspired\b", text, flags=re.IGNORECASE):
        locks.append(f"{match.group(1)}-inspired")
    for match in re.finditer(
        r"\b(?:the\s+)?style\s+(?:(?:should|must|will)\s+be|is|:)\s*"
        r"(?:reimagined\s+)?(?:into|as|in)?\s*([^.!?\n]+)",
        text,
        flags=re.IGNORECASE,
    ):
        lock = re.sub(r"^(?:a|an|the)\s+", "", match.group(1).strip(" ,:;"), flags=re.IGNORECASE)
        if lock:
            locks.append(lock)
    deduped: list[str] = []
    for lock in locks:
        if lock.casefold() not in {item.casefold() for item in deduped}:
            deduped.append(lock)
    return deduped


def _minimax_h3_style_lock_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    normalized = "".join(character for character in unicodedata.normalize("NFKD", text) if not unicodedata.combining(character))
    for token in re.findall(r"[a-z0-9]+", normalized.casefold().replace("’", "'")):
        if len(token) <= 1 or token in {"the", "and", "with", "style", "visual", "look"}:
            continue
        if re.fullmatch(r"\d{3,4}s", token):
            token = token[:-1]
        tokens.add(token)
    return tokens


def _minimax_h3_style_locks_present(prompt_text: str, style_locks: list[str]) -> bool:
    prompt_tokens = _minimax_h3_style_lock_tokens(prompt_text)
    return all(_minimax_h3_style_lock_tokens(lock).issubset(prompt_tokens) for lock in style_locks)


def _apply_minimax_h3_visual_medium_lock(
    prompt_text: str,
    user_prompt: str,
    minimax_h3_mode: str = "t2va",
) -> str:
    mode = _normalize_minimax_h3_mode(minimax_h3_mode)
    prompt = _normalize_minimax_h3_prompt(prompt_text, mode)
    medium = _minimax_h3_requested_visual_medium(user_prompt)
    if mode == "ref2va":
        sections = _minimax_h3_ref_sections(prompt)
        if not sections or not medium:
            return prompt
        detailed = sections["detailed_description"]
        shot_match = re.search(r"\[Shot 1\]", detailed)
        if not shot_match:
            return prompt
        opening = detailed[: shot_match.start()]
        anchor_patterns = {
            "2d_animation": r"(?:\b2[ -]?d\s+hand[- ]drawn\s+cel[- ]animation\b|\b2[ -]?d\b[^.!?\n]{0,80}\b(?:hand[- ]drawn|cel[- ]animation|cartoon|anime)\b|\b(?:hand[- ]drawn|cel[- ]animation|cartoon|anime)\b[^.!?\n]{0,80}\b2[ -]?d\b)",
            "3d_animation": _MINIMAX_H3_3D_ANIMATION_PATTERN,
            "stop_motion": _MINIMAX_H3_STOP_MOTION_PATTERN,
            "live_action": _MINIMAX_H3_LIVE_ACTION_PATTERN,
            "animation": rf"(?:{_MINIMAX_H3_ANIMATION_PATTERN}|{_MINIMAX_H3_2D_ANIMATION_PATTERN}|{_MINIMAX_H3_3D_ANIMATION_PATTERN}|{_MINIMAX_H3_STOP_MOTION_PATTERN})",
        }
        anchors = {
            "2d_animation": "The target video uses immutable 2D hand-drawn cel animation, with no live action, photorealism, or 3D CGI. ",
            "3d_animation": "The target video uses immutable stylized 3D computer animation. ",
            "stop_motion": "The target video uses immutable practical stop-motion animation. ",
            "live_action": "The target video uses immutable realistic live-action practical film photography. ",
            "animation": "The target video uses immutable stylized animation, with no live action or photorealism. ",
        }
        anchor_match = re.search(anchor_patterns[medium], opening[:400], flags=re.IGNORECASE)
        if not anchor_match:
            sections["detailed_description"] = f"{anchors[medium]}{detailed}"
            return _rebuild_minimax_h3_ref_prompt(sections)
        additions: list[str] = []
        if medium == "2d_animation":
            if not _has_negated_prompt_phrase(detailed, _MINIMAX_H3_LIVE_ACTION_PATTERN):
                additions.append("no live action or photorealism")
            if not _has_negated_prompt_phrase(detailed, _MINIMAX_H3_3D_ANIMATION_PATTERN):
                additions.append("no 3D CGI")
        elif medium == "animation" and not _has_negated_prompt_phrase(detailed, _MINIMAX_H3_LIVE_ACTION_PATTERN):
            additions.append("no live action or photorealism")
        if additions:
            insert_at = anchor_match.end()
            opening = f"{opening[:insert_at]}, with {' and '.join(additions)}{opening[insert_at:]}"
            sections["detailed_description"] = f"{opening}{detailed[shot_match.start():]}"
            return _rebuild_minimax_h3_ref_prompt(sections)
        return prompt
    opening_match = re.match(r"^integrated_multimodal_description\s*:\s*\[Shot 1\]\s*", prompt)
    if not prompt or not medium or not opening_match:
        return prompt
    opening = prompt[opening_match.end() : opening_match.end() + 320]
    anchor_patterns = {
        "2d_animation": r"(?:\b2[ -]?d\s+hand[- ]drawn\s+cel[- ]animation\b|\b2[ -]?d\b[^.!?\n]{0,80}\b(?:hand[- ]drawn|cel[- ]animation|cartoon|anime)\b|"
        r"\b(?:hand[- ]drawn|cel[- ]animation|cartoon|anime)\b[^.!?\n]{0,80}\b2[ -]?d\b)",
        "3d_animation": _MINIMAX_H3_3D_ANIMATION_PATTERN,
        "stop_motion": _MINIMAX_H3_STOP_MOTION_PATTERN,
        "live_action": _MINIMAX_H3_LIVE_ACTION_PATTERN,
        "animation": rf"(?:{_MINIMAX_H3_ANIMATION_PATTERN}|{_MINIMAX_H3_2D_ANIMATION_PATTERN}|{_MINIMAX_H3_3D_ANIMATION_PATTERN}|{_MINIMAX_H3_STOP_MOTION_PATTERN})",
    }
    anchor_match = re.search(anchor_patterns[medium], opening, flags=re.IGNORECASE)
    if not anchor_match:
        anchors = {
            "2d_animation": "2D hand-drawn cel animation, with no live action, photorealism, or 3D CGI. ",
            "3d_animation": "Stylized 3D computer animation. ",
            "stop_motion": "Practical stop-motion animation. ",
            "live_action": "Realistic live-action practical film photography. ",
            "animation": "Stylized animation, with no live action or photorealism. ",
        }
        return f"{prompt[:opening_match.end()]}{anchors[medium]}{prompt[opening_match.end():]}"
    anchor_end = opening_match.end() + anchor_match.end()
    integrated = prompt.split("\n\noverall_soundscape:", 1)[0]
    if medium == "2d_animation":
        live_action_excluded = _has_negated_prompt_phrase(integrated, _MINIMAX_H3_LIVE_ACTION_PATTERN)
        three_d_excluded = _has_negated_prompt_phrase(integrated, _MINIMAX_H3_3D_ANIMATION_PATTERN)
        if not live_action_excluded or not three_d_excluded:
            separator = "" if prompt[anchor_end : anchor_end + 1] == "," else ","
            return f"{prompt[:anchor_end]}{separator} with no live action, photorealism, or 3D CGI{prompt[anchor_end:]}"
    elif medium == "animation" and not _has_negated_prompt_phrase(integrated, _MINIMAX_H3_LIVE_ACTION_PATTERN):
        separator = "" if prompt[anchor_end : anchor_end + 1] == "," else ","
        return f"{prompt[:anchor_end]}{separator} with no live action or photorealism{prompt[anchor_end:]}"
    return prompt


def _apply_minimax_h3_style_locks(
    prompt_text: str,
    user_prompt: str,
    minimax_h3_mode: str = "t2va",
) -> str:
    mode = _normalize_minimax_h3_mode(minimax_h3_mode)
    prompt = _normalize_minimax_h3_prompt(prompt_text, mode)
    style_locks = _minimax_h3_requested_style_locks(user_prompt)
    if not prompt or not style_locks:
        return prompt
    if mode == "ref2va":
        sections = _minimax_h3_ref_sections(prompt)
        if not sections:
            return prompt
        detailed = sections["detailed_description"]
        shot_match = re.search(r"\[Shot 1\]", detailed)
        if not shot_match:
            return prompt
        opening = detailed[: shot_match.start()].strip()
        missing = [lock for lock in style_locks if not _minimax_h3_style_locks_present(opening, [lock])]
        if not missing:
            return prompt
        style_sentence = f"The target video faithfully uses {' and '.join(missing)} visual styling."
        sections["detailed_description"] = (
            f"{opening} {style_sentence} {detailed[shot_match.start():]}".strip()
            if opening
            else f"{style_sentence} {detailed}"
        )
        return _rebuild_minimax_h3_ref_prompt(sections)
    opening_match = re.match(r"^integrated_multimodal_description\s*:\s*\[Shot 1\]\s*", prompt)
    if not opening_match:
        return prompt
    opening = prompt[opening_match.end() : opening_match.end() + 320]
    bracketed_style = re.search(r"\[[^\]\n]{1,240}\]", opening)
    if bracketed_style and _minimax_h3_style_locks_present(bracketed_style.group(0), style_locks):
        canonical_style = f"faithfully using {' and '.join(style_locks)} visual styling"
        start = opening_match.end() + bracketed_style.start()
        end = opening_match.end() + bracketed_style.end()
        prompt = f"{prompt[:start]}{canonical_style}{prompt[end:]}"
        opening = prompt[opening_match.end() : opening_match.end() + 320]
    missing = [lock for lock in style_locks if not _minimax_h3_style_locks_present(prompt[:900], [lock])]
    if not missing:
        return prompt
    medium_anchor = re.search(
        r"\b(?:2[ -]?d\s+hand[- ]drawn\s+cel[- ]animation|stylized\s+3[ -]?d\s+computer\s+animation|"
        r"practical\s+stop[- ]motion\s+animation|realistic\s+live[- ]action\s+practical\s+film\s+photography|stylized\s+animation)\b",
        opening,
        flags=re.IGNORECASE,
    )
    leading_separator = ""
    if medium_anchor:
        split_at = opening_match.end() + medium_anchor.end()
        if prompt[split_at : split_at + 1] == ",":
            split_at += 1
        else:
            leading_separator = ","
    else:
        insertion_point = prompt.find(",", opening_match.end(), opening_match.end() + 220)
        insert_after_delimiter = insertion_point >= 0
        if insertion_point < 0:
            first_sentence_end = prompt.find(".", opening_match.end(), opening_match.end() + 220)
            insertion_point = first_sentence_end if first_sentence_end >= 0 else opening_match.end()
            insert_after_delimiter = first_sentence_end >= 0
        split_at = insertion_point + 1 if insert_after_delimiter else insertion_point
    style_text = " and ".join(missing)
    suffix = prompt[split_at:]
    separator = "" if suffix.startswith((" ", "\n")) else " "
    return f"{prompt[:split_at]}{leading_separator} faithfully using {style_text} visual styling,{separator}{suffix}"


_MINIMAX_H3_SOUND_EVENT_FIDELITY = (
    (
        "fire_blast",
        r"\b(?:(?:fire|flame)[- ](?:blast|burst)|(?:crackle|roar|whoosh)\s+of\s+(?:a\s+)?(?:fire|flames?))\b",
        r"\b(?:fire|flames?|fiery|burns?|burning|ignites?|ignition|blaze)\b",
    ),
    (
        "gunfire",
        r"\b(?:gunshots?|gunfire|rifle\s+shots?|pistol\s+shots?)\b",
        r"\b(?:gun|pistol|rifle|revolver|bullet|gunfire|shoots?|fires?\s+(?:a\s+)?(?:round|bullet))\b",
    ),
    (
        "explosion",
        r"\b(?:explosion|detonation|explosive\s+boom)\b",
        r"\b(?:explodes?|explosion|detonates?|detonation|blasts?\s+apart)\b",
    ),
)


def _minimax_h3_orphaned_sound_events(integrated: str, soundscape: str) -> list[str]:
    return [
        label
        for label, sound_pattern, visual_pattern in _MINIMAX_H3_SOUND_EVENT_FIDELITY
        if re.search(sound_pattern, soundscape, flags=re.IGNORECASE)
        and not re.search(visual_pattern, integrated, flags=re.IGNORECASE)
    ]


def _apply_minimax_h3_sound_fidelity(
    prompt_text: str,
    user_prompt: str,
    minimax_h3_mode: str = "t2va",
) -> str:
    mode = _normalize_minimax_h3_mode(minimax_h3_mode)
    prompt = _normalize_minimax_h3_prompt(prompt_text, mode)
    ref_sections = _minimax_h3_ref_sections(prompt) if mode == "ref2va" else None
    if ref_sections:
        integrated = ref_sections["detailed_description"]
        soundscape = ref_sections["overall_soundscape"]
        music = ref_sections["non_diegetic_music"]
    else:
        integrated = ""
        soundscape = ""
        music = ""
    structured_match = re.fullmatch(
        r"(?ms)integrated_multimodal_description\s*:\s*(.*?)\s*\n+\s*overall_soundscape\s*:\s*(.*?)\s*\n+\s*non_diegetic_music\s*:\s*(.*)",
        prompt,
    )
    if not ref_sections and not structured_match:
        return prompt
    if structured_match:
        integrated = structured_match.group(1).strip()
        soundscape = structured_match.group(2).strip()
        music = structured_match.group(3).strip()
    orphaned = set(_minimax_h3_orphaned_sound_events(integrated, soundscape))
    removable_patterns = [
        sound_pattern
        for label, sound_pattern, _visual_pattern in _MINIMAX_H3_SOUND_EVENT_FIDELITY
        if label in orphaned and not _has_unnegated_prompt_phrase(user_prompt, sound_pattern)
    ]
    if not removable_patterns:
        return prompt
    parts = [part.strip() for part in re.split(r",\s*", soundscape) if part.strip()]
    kept = [
        re.sub(r"^and\s+", "", part, flags=re.IGNORECASE).strip()
        for part in parts
        if not any(re.search(pattern, part, flags=re.IGNORECASE) for pattern in removable_patterns)
    ]
    if not kept:
        repaired_soundscape = "N/A"
    elif len(kept) == 1:
        repaired_soundscape = kept[0].rstrip(".") + "."
    elif len(kept) == 2:
        repaired_soundscape = f"{kept[0].rstrip(' .')} and {kept[1].rstrip(' .')}."
    else:
        repaired_soundscape = f"{', '.join(item.rstrip(' .') for item in kept[:-1])}, and {kept[-1].rstrip(' .')}."
    if ref_sections:
        ref_sections["overall_soundscape"] = repaired_soundscape
        return _rebuild_minimax_h3_ref_prompt(ref_sections)
    return (
        f"integrated_multimodal_description: {integrated}\n\n"
        f"overall_soundscape: {repaired_soundscape}\n\n"
        f"non_diegetic_music: {music}"
    )


def _minimax_h3_visual_medium_instruction(medium: str) -> str:
    if medium == "2d_animation":
        return (
            'The requested rendering medium is immutable 2D animation. Begin the first sentence immediately after [Shot 1] with '
            '"2D hand-drawn cel animation," preserve drawn linework, cel shading, stylized proportions, and design continuity across every cut, '
            'and explicitly state "no live action, photorealism, or 3D CGI." Cinematic, period, and genre direction changes staging, lighting, '
            "camera language, and editing inside that medium; it never changes the medium."
        )
    if medium == "3d_animation":
        return (
            'The requested rendering medium is immutable stylized 3D computer animation. Begin the first sentence immediately after [Shot 1] '
            'with "Stylized 3D computer animation," and preserve the same rendering language and subject designs across every cut.'
        )
    if medium == "stop_motion":
        return (
            'The requested rendering medium is immutable stop-motion animation. Begin the first sentence immediately after [Shot 1] with '
            '"Practical stop-motion animation," name the requested material or puppet technique, and preserve it across every cut.'
        )
    if medium == "live_action":
        return (
            'The requested rendering medium is immutable live action. Begin the first sentence immediately after [Shot 1] with '
            '"Realistic live-action practical film photography," and preserve the photographic texture across every cut; cinematic language '
            "must not silently convert it into cartoon or fully animated rendering."
        )
    if medium == "animation":
        return (
            'The requested rendering medium is immutable animation. Choose the single concrete animation technique best supported by the user\'s '
            "style cues, name that technique in the first words immediately after [Shot 1], preserve it across every cut, and explicitly exclude "
            "live action and photorealism."
        )
    return ""


_MINIMAX_H3_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_MINIMAX_H3_MAX_CUSTOM_SHOT_COUNT = 99
_MINIMAX_H3_SHOT_COUNT_CHOICES = ("auto", *(str(value) for value in range(1, 13)), "custom")
_MINIMAX_H3_DIALOGUE_MODES = ("auto", "required", "off")
_MINIMAX_H3_MAX_DIALOGUE_LINE_COUNT = 12
_MINIMAX_H3_MAX_EXPECTED_SUBJECT_COUNT = 99


def _normalize_minimax_h3_expected_subject_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(_MINIMAX_H3_MAX_EXPECTED_SUBJECT_COUNT, count))


def _minimax_h3_subject_count_instruction(value: Any) -> str:
    count = _normalize_minimax_h3_expected_subject_count(value)
    if count <= 0:
        return (
            "Create the fewest semantic <Subject N> labels needed for visible content that must be tracked independently. "
            "Do not split ordinary wardrobe, props, environment, lighting, composition, style, actions, or effects into separate "
            "Subject labels when they can remain attributes of a principal Subject or the shot timeline."
        )
    noun = "label" if count == 1 else "labels"
    range_text = "<Subject 1>" if count == 1 else f"<Subject 1> through <Subject {count}>"
    return (
        f"Create exactly {count} consecutive semantic Subject {noun}, {range_text}, and no other <Subject N> labels. "
        "This explicit H3 Reference Context setting is authoritative and counts independently tracked semantic content units, "
        "not attached <Picture N> assets or only human characters. Allocate Subject labels first to independently acting or "
        "identity-bearing people, creatures, and hero objects required by the brief. Fold ordinary wardrobe, props, environment, "
        "lighting, palette, composition, style, actions, and effects into the relevant Subject definition or shot timeline unless "
        "they truly must be tracked independently and the selected count has capacity."
    )


def _normalize_minimax_h3_shot_count(value: Any) -> str:
    normalized = str(value if value is not None else "auto").strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized.isdigit() and 1 <= int(normalized) <= _MINIMAX_H3_MAX_CUSTOM_SHOT_COUNT:
        return str(int(normalized))
    return "auto"


def _resolve_minimax_h3_shot_count(selector: Any, custom_shot_count: Any = 12) -> str:
    selected = str(selector if selector is not None else "auto").strip().lower()
    if selected == "custom":
        return _normalize_minimax_h3_shot_count(custom_shot_count)
    return _normalize_minimax_h3_shot_count(selected)


def _minimax_h3_requested_shot_count(user_prompt: str) -> int:
    text = _single_paragraph(_sanitize_prompt_text(user_prompt, "", 8000))
    labels = sorted({int(value) for value in re.findall(r"\bShot\s+(\d{1,2})\b", text, flags=re.IGNORECASE)})
    if len(labels) >= 2 and labels == list(range(1, labels[-1] + 1)):
        return labels[-1]
    count_match = re.search(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2})\s+"
        r"(?:(?:separate|distinct|individual|rapid[- ]cut|hard[- ]cut)\s+){0,2}shots?\b",
        text,
        flags=re.IGNORECASE,
    )
    if not count_match:
        return 0
    token = count_match.group(1).lower()
    return _MINIMAX_H3_NUMBER_WORDS.get(token, int(token) if token.isdigit() else 0)


def _minimax_h3_effective_shot_count(user_prompt: str, shot_count: Any = "auto") -> int:
    normalized = _normalize_minimax_h3_shot_count(shot_count)
    if normalized != "auto":
        return int(normalized)
    return _minimax_h3_requested_shot_count(user_prompt)


def _minimax_h3_shot_count_instruction(shot_count: Any) -> str:
    normalized = _normalize_minimax_h3_shot_count(shot_count)
    if normalized == "auto":
        return "Use the fewest shots that fully express the request."
    noun = "shot" if normalized == "1" else "shots"
    return (
        f"Use exactly {normalized} consecutively numbered {noun}. This explicit Target Profile shot-count setting is authoritative "
        "and overrides any conflicting shot-count wording in the user brief."
    )


def _minimax_h3_timestamp_label(seconds: Any) -> str:
    try:
        total_milliseconds = max(0, int(round(float(seconds) * 1000.0)))
    except (TypeError, ValueError, OverflowError):
        total_milliseconds = 0
    minutes, remainder = divmod(total_milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _minimax_h3_timeline_bound_instruction(
    shot_count: Any,
    duration_seconds: Any,
) -> str:
    try:
        count = max(0, int(shot_count))
    except (TypeError, ValueError, OverflowError):
        count = 0
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError, OverflowError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        return ""

    end_label = _minimax_h3_timestamp_label(duration)
    endpoint_rule = (
        f"The duration endpoint {end_label} is exclusive: it is the end of the video, never a valid shot-start timestamp. "
    )
    if count <= 1:
        return endpoint_rule + "Shot 1 has no timestamp and occupies the available duration."

    cut_symbols = " < ".join(f"t{shot}" for shot in range(2, count + 1))
    return (
        f"Before writing, plan exactly {count - 1} later cut-start timestamps for the {count}-shot timeline and enforce "
        f"0 < {cut_symbols} < {end_label}. {endpoint_rule}"
        f"Shot {count} must begin early enough to leave meaningful time for its complete action, sound, and any dialogue; do not leave it only a fractional-frame tail."
    )


def _normalize_minimax_h3_dialogue_mode(value: Any) -> str:
    normalized = str(value if value is not None else "auto").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "none": "off",
        "no_dialogue": "off",
        "brief_only": "auto",
        "generate": "required",
        "required_generate": "required",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in _MINIMAX_H3_DIALOGUE_MODES else "auto"


def _normalize_minimax_h3_dialogue_line_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return 2
    return max(1, min(_MINIMAX_H3_MAX_DIALOGUE_LINE_COUNT, count))


def _minimax_h3_dialogue_guidance_text(value: Any) -> str:
    return _sanitize_prompt_text(
        str(value or ""),
        "",
        1200,
        strip_thinking=True,
        strip_markdown=True,
    ).strip()


def _minimax_h3_dialogue_block_count(prompt_text: Any) -> int:
    prompt = str(prompt_text or "")
    return len(re.findall(r"<d>\[[^\]\n]+\]\s*\S.*?</d>", prompt, flags=re.DOTALL))


_MINIMAX_H3_DIALOGUE_TAGLIKE_PATTERN = re.compile(
    r"(?:<\s*/?\s*d(?:\s*/)?\s*>|&lt;\s*/?\s*d(?:\s*/)?\s*&gt;)",
    flags=re.IGNORECASE,
)

_MINIMAX_H3_DIALOGUE_LANG_ATTRIBUTE_OPEN_PATTERN = re.compile(
    r"<\s*d\s+lang\s*=\s*(?P<quote>['\"])(?P<language>[A-Za-z][A-Za-z -]{0,31})(?P=quote)\s*>",
    flags=re.IGNORECASE,
)

_MINIMAX_H3_DIALOGUE_DUPLICATE_PREFIX_CLOSE_PATTERN = re.compile(
    r"<\s*/\s*<\s*/\s*d\s*>",
    flags=re.IGNORECASE,
)


def _minimax_h3_timeline_bounds(
    prompt_text: Any,
    minimax_h3_mode: Any,
) -> tuple[int, int] | None:
    prompt = str(prompt_text or "")
    field_name = (
        "detailed_description"
        if _normalize_minimax_h3_mode(minimax_h3_mode) == "ref2va"
        else "integrated_multimodal_description"
    )
    opening = re.search(rf"(?m)^{re.escape(field_name)}\s*:\s*", prompt)
    if not opening:
        return None
    closing = re.search(r"(?m)^overall_soundscape\s*:", prompt[opening.end() :])
    if not closing:
        return None
    return opening.end(), opening.end() + closing.start()


def _repair_minimax_h3_t2va_contract_transport(
    prompt_text: Any,
    minimax_h3_mode: Any,
) -> tuple[str, list[str]]:
    """Repair deterministic T2VA syntax without rewriting creative content.

    Repairs are deliberately limited to the integrated shot timeline: native
    cut delimiters, the common ``<d>English>`` bracket loss, and unambiguous
    speaker placeholders such as ``(Sx: Girl)``. Ambiguous speaker mappings
    are left untouched for the validator (or a permissive generation gate).
    """

    prompt = str(prompt_text or "")
    if _normalize_minimax_h3_mode(minimax_h3_mode) != "t2va":
        return prompt, []
    bounds = _minimax_h3_timeline_bounds(prompt, "t2va")
    if bounds is None:
        return prompt, []
    timeline_start, timeline_end = bounds
    original_timeline = prompt[timeline_start:timeline_end]
    timeline = re.sub(
        r"(\[Shot (?:[2-9]|[1-9]\d+)\]\s+At\s+)([0-5]?\d):(\d{3})(?=\s*,)",
        lambda match: f"{match.group(1)}00:{int(match.group(2)):02d}.{match.group(3)}",
        original_timeline,
        flags=re.IGNORECASE,
    )
    repairs: list[str] = []
    if timeline != original_timeline:
        repairs.append("canonicalized_timed_cut_timestamps")
    timeline_before_cut_openers = timeline
    timeline = _repair_minimax_h3_timed_cut_openers(timeline)
    if timeline != timeline_before_cut_openers:
        repairs.append("canonicalized_timed_cut_openers")

    # DiffusionGemma occasionally drops only the square brackets around the
    # language label. Repair all such blocks transactionally; nested,
    # unbalanced, empty, or cross-shot markup is not guessed at.
    literal_tags = list(re.finditer(r"</?d>", timeline, flags=re.IGNORECASE))
    language_replacements: list[tuple[int, int, str]] = []
    language_structure_valid = True
    open_tag: re.Match[str] | None = None
    for tag in literal_tags:
        closing = bool(re.fullmatch(r"</d>", tag.group(0), flags=re.IGNORECASE))
        if not closing:
            if open_tag is not None:
                language_structure_valid = False
                break
            open_tag = tag
            continue
        if open_tag is None:
            language_structure_valid = False
            break
        between = timeline[open_tag.end() : tag.start()]
        if "[Shot " in between or not between.strip():
            language_structure_valid = False
            break
        if not re.match(r"[ \t]*\[[^\]\n]+\]\s*\S", between):
            malformed = re.match(
                r"(?P<leading>[ \t]*)(?P<language>[A-Za-z][A-Za-z -]{0,31})>(?=\s*\S)",
                between,
            )
            if malformed is None:
                language_structure_valid = False
                break
            language_replacements.append(
                (
                    open_tag.end(),
                    open_tag.end() + malformed.end(),
                    f"{malformed.group('leading')}[{malformed.group('language')}]",
                )
            )
        open_tag = None
    if open_tag is not None:
        language_structure_valid = False
    if language_structure_valid and language_replacements:
        for start, end, replacement in sorted(language_replacements, reverse=True):
            timeline = timeline[:start] + replacement + timeline[end:]
        repairs.append("restored_dialogue_language_brackets")

    # Resolve only cues that are actually associated with a canonical dialogue
    # block. Stable labels receive stable numeric IDs; existing IDs are
    # reserved. A transaction with more than one valid interpretation is left
    # wholly unchanged.
    dialogue_blocks = list(
        re.finditer(
            r"<d>\[[^\]\n]+\]\s*\S(?:(?!</?d>|\[Shot \d+\]).)*?</d>",
            timeline,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    speaker_pattern = re.compile(
        r"\(\s*S(?P<speaker>[1-9]\d*|x)"
        r"(?:\s*[-:]\s*(?P<label>[^()\r\n]{1,64}?))?\s*\)",
        flags=re.IGNORECASE,
    )
    selected_cues: dict[tuple[int, int], re.Match[str]] = {}
    for dialogue in dialogue_blocks:
        shot_markers = list(
            re.finditer(r"\[Shot \d+\]", timeline[: dialogue.start()], flags=re.IGNORECASE)
        )
        shot_start = shot_markers[-1].end() if shot_markers else 0
        window_start = max(shot_start, dialogue.start() - 500)
        cues = [
            cue
            for cue in speaker_pattern.finditer(timeline, window_start, dialogue.start())
            if not any(
                block.start() <= cue.start() < block.end()
                for block in dialogue_blocks
            )
        ]
        if cues:
            cue = cues[-1]
            selected_cues[(cue.start(), cue.end())] = cue

    numeric_ids: set[int] = set()
    unlabeled_numeric_ids: set[int] = set()
    label_to_id: dict[str, int] = {}
    id_to_label: dict[int, str] = {}
    placeholder_labels: list[str] = []
    bare_placeholders: list[re.Match[str]] = []
    speaker_mapping_valid = True
    for cue in selected_cues.values():
        speaker = cue.group("speaker").casefold()
        label = " ".join(str(cue.group("label") or "").split())
        label_key = label.casefold()
        if speaker != "x":
            speaker_id = int(speaker)
            numeric_ids.add(speaker_id)
            if not label_key:
                unlabeled_numeric_ids.add(speaker_id)
                continue
            if (
                (label_key in label_to_id and label_to_id[label_key] != speaker_id)
                or (speaker_id in id_to_label and id_to_label[speaker_id] != label_key)
            ):
                speaker_mapping_valid = False
                break
            label_to_id[label_key] = speaker_id
            id_to_label[speaker_id] = label_key
        elif label_key:
            placeholder_labels.append(label_key)
        else:
            bare_placeholders.append(cue)

    unresolved_labels = [
        label for label in dict.fromkeys(placeholder_labels) if label not in label_to_id
    ]
    if unresolved_labels and unlabeled_numeric_ids:
        speaker_mapping_valid = False
    if speaker_mapping_valid:
        next_id = 1
        for label in unresolved_labels:
            while next_id in numeric_ids:
                next_id += 1
            label_to_id[label] = next_id
            id_to_label[next_id] = label
            numeric_ids.add(next_id)
            next_id += 1
        possible_ids = set(numeric_ids)
        if bare_placeholders and not possible_ids:
            if len(bare_placeholders) == 1:
                possible_ids = {1}
                numeric_ids.add(1)
            else:
                speaker_mapping_valid = False
        if bare_placeholders and len(possible_ids) != 1:
            speaker_mapping_valid = False

    cue_replacements: list[tuple[int, int, str]] = []
    if speaker_mapping_valid:
        sole_id = next(iter(numeric_ids)) if len(numeric_ids) == 1 else None
        for cue in selected_cues.values():
            speaker = cue.group("speaker").casefold()
            label = " ".join(str(cue.group("label") or "").split()).casefold()
            if speaker == "x":
                speaker_id = label_to_id.get(label) if label else sole_id
            else:
                speaker_id = int(speaker)
            if speaker_id is not None and cue.group(0) != f"(S{speaker_id})":
                cue_replacements.append((cue.start(), cue.end(), f"(S{speaker_id})"))
        if cue_replacements:
            for start, end, replacement in sorted(cue_replacements, reverse=True):
                timeline = timeline[:start] + replacement + timeline[end:]
            repairs.append("canonicalized_dialogue_speaker_ids")

    repaired = prompt[:timeline_start] + timeline + prompt[timeline_end:]
    if repaired == prompt:
        return prompt, []
    return repaired, list(dict.fromkeys(repairs))


def _repair_minimax_h3_dialogue_transport(
    prompt_text: Any,
    minimax_h3_mode: Any,
) -> tuple[str, list[str]]:
    """Canonicalize only unambiguous H3 dialogue delimiters in the timeline.

    Spoken text, speakers, shots, timestamps, and every non-dialogue byte remain
    model-authored. Ambiguous or unbalanced markup is returned unchanged so the
    validator can fail closed.
    """

    prompt = str(prompt_text or "")
    bounds = _minimax_h3_timeline_bounds(prompt, minimax_h3_mode)
    if bounds is None:
        return prompt, []
    timeline_start, timeline_end = bounds
    timeline = prompt[timeline_start:timeline_end]
    repairs: list[str] = []

    # A recurring DiffusionGemma transport error emits HTML-style language
    # attributes and an extra ``</`` before the closer. Accept only the exact,
    # single-attribute shape and a conservative language token. The existing
    # transactional tag parser below still rejects orphaned, nested, or
    # cross-shot blocks, so these substitutions never rescue ambiguous markup.
    attribute_openers = list(
        _MINIMAX_H3_DIALOGUE_LANG_ATTRIBUTE_OPEN_PATTERN.finditer(timeline)
    )
    if any(
        re.match(r"[ \t]*\[[^\]\n]+\]", timeline[opener.end() :])
        for opener in attribute_openers
    ):
        return prompt, []
    if attribute_openers:
        timeline = _MINIMAX_H3_DIALOGUE_LANG_ATTRIBUTE_OPEN_PATTERN.sub(
            lambda match: f"<d>[{' '.join(match.group('language').split())}]",
            timeline,
        )
        repairs.append("canonicalized_dialogue_language_attribute")

    if _MINIMAX_H3_DIALOGUE_DUPLICATE_PREFIX_CLOSE_PATTERN.search(timeline):
        timeline = _MINIMAX_H3_DIALOGUE_DUPLICATE_PREFIX_CLOSE_PATTERN.sub(
            "</d>",
            timeline,
        )
        repairs.append("canonicalized_dialogue_duplicate_prefix_close")

    tokens = list(_MINIMAX_H3_DIALOGUE_TAGLIKE_PATTERN.finditer(timeline))
    if not tokens:
        return prompt, []

    output: list[str] = []
    cursor = 0
    block_open = False
    for token in tokens:
        raw_token = token.group(0)
        lowered = raw_token.casefold()
        is_entity = lowered.startswith("&lt;")
        token_inner = (
            re.sub(r"\A&lt;|&gt;\Z", "", raw_token, flags=re.IGNORECASE)
            if is_entity
            else raw_token[1:-1]
        )
        compact_inner = re.sub(r"\s+", "", token_inner).casefold()
        explicitly_closing = compact_inner.startswith("/")
        self_closing = compact_inner.endswith("/") and not explicitly_closing
        following = timeline[token.end() :]
        language_follows = bool(
            re.match(r"\s*\[[A-Za-z][^\]\n]{0,31}\]\s*\S", following)
        )

        output.append(timeline[cursor:token.start()])
        if not block_open:
            if explicitly_closing or self_closing or not language_follows:
                return prompt, []
            output.append("<d>")
            block_open = True
            cursor = token.end()
            whitespace = re.match(r"[ \t\n]+(?=\[[A-Za-z][^\]\n]{0,31}\])", timeline[cursor:])
            if whitespace:
                cursor += whitespace.end()
                repairs.append("removed_dialogue_open_language_whitespace")
            if raw_token != "<d>":
                repairs.append("canonicalized_dialogue_open_tag")
            if is_entity:
                repairs.append("decoded_dialogue_tag_entity")
            continue

        if explicitly_closing or self_closing or (not language_follows):
            output.append("</d>")
            block_open = False
            cursor = token.end()
            if self_closing:
                repairs.append("canonicalized_dialogue_self_closer")
            elif not explicitly_closing:
                repairs.append("canonicalized_repeated_dialogue_open_as_close")
            elif raw_token != "</d>":
                repairs.append("canonicalized_dialogue_close_tag")
            if is_entity:
                repairs.append("decoded_dialogue_tag_entity")
            continue

        # A second opener followed by a language label is a genuinely nested or
        # duplicated utterance boundary. Do not guess which block it belongs to.
        return prompt, []

    if block_open:
        return prompt, []
    output.append(timeline[cursor:])
    repaired_timeline = "".join(output)
    if repairs:
        open_count = repaired_timeline.count("<d>")
        close_count = repaired_timeline.count("</d>")
        valid_blocks = len(
            re.findall(
                r"<d>\[[^\]\n]+\]\s*\S(?:(?!</?d>|\[Shot \d+\]).)*?</d>",
                repaired_timeline,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        if not open_count or open_count != close_count or valid_blocks != open_count:
            return prompt, []
    repaired = prompt[:timeline_start] + repaired_timeline + prompt[timeline_end:]
    if repaired == prompt:
        return prompt, []
    return repaired, list(dict.fromkeys(repairs))


def _minimax_h3_dialogue_diagnostics(
    prompt_text: Any,
    minimax_h3_mode: Any,
) -> dict[str, Any]:
    prompt = str(prompt_text or "")
    bounds = _minimax_h3_timeline_bounds(prompt, minimax_h3_mode)
    timeline = prompt[bounds[0] : bounds[1]] if bounds is not None else ""
    taglike = _MINIMAX_H3_DIALOGUE_TAGLIKE_PATTERN.findall(timeline)
    return {
        "timeline_found": bounds is not None,
        "open_count": timeline.count("<d>"),
        "close_count": timeline.count("</d>"),
        "valid_block_count": _minimax_h3_dialogue_block_count(timeline),
        "taglike_count": len(taglike),
        "escaped_tag_count": len(
            re.findall(r"&lt;\s*/?\s*d(?:\s*/)?\s*&gt;", timeline, flags=re.IGNORECASE)
        ),
    }


def _minimax_h3_dialogue_instruction(
    dialogue_mode: Any,
    dialogue_line_count: Any = 2,
    dialogue_guidance: Any = "",
    minimax_h3_mode: Any = "t2va",
) -> str:
    mode = _normalize_minimax_h3_dialogue_mode(dialogue_mode)
    count = _normalize_minimax_h3_dialogue_line_count(dialogue_line_count)
    guidance = _minimax_h3_dialogue_guidance_text(dialogue_guidance)
    ref2va = _normalize_minimax_h3_mode(minimax_h3_mode) == "ref2va"
    if mode == "required":
        noun = "utterance" if count == 1 else "utterances"
        guidance_instruction = (
            f" Dialogue direction: {guidance} Preserve any quoted wording in that direction verbatim."
            if guidance
            else " Author concise dialogue that advances the requested action or character intent."
        )
        speaker_pattern = (
            "A locked medium shot holds on <Subject 1> (S1) as they look up and say with quiet awe: <d>[English] The clouds are moving.</d> Their lips close. "
            "For Ref2VA, repeat an adjacent '<Subject N> (S1)' pair (using S2, S3, and so on for additional speakers) in the same shot before every spoken block, even when that Subject was named earlier. "
            "Keep a stable one-to-one mapping: the same Subject always uses the same S ID and an S ID never belongs to another Subject. "
            "Never output the literal placeholder (Sx), and never use a bare numbered cue after multiple Subject tags."
            if ref2va
            else (
                "A locked medium shot holds (S1) as they look up and say with quiet awe: "
                "<d>[English] The clouds are moving.</d> Their lips close."
            )
        )
        return (
            f"Dialogue is mandatory and explicitly authorized by the Target Profile. Write exactly {count} complete spoken {noun}, "
            f"each inside an existing camera-defined shot. Follow this complete pattern: {speaker_pattern} "
            f"The speaker ID stays outside the tags; only [Language] and the spoken words belong inside. Use literal numbered cues such as (S1) and (S2), never the placeholder (Sx) or a role/name suffix inside the cue. Never write <d>[S1], ordinary quoted speech, "
            f"or delivery directions inside <d>. The final prompt must contain exactly {count} literal <d> openers and {count} literal </d> closers.{guidance_instruction} "
            "Plan the spoken timeline before camera choreography; dialogue completeness takes priority over decorative camera coverage or a descriptive "
            "word-count target. Reserve enough uninterrupted time for every line, prefer a locked or gently moving "
            "dialogue shot when useful, and do not add cuts merely to place dialogue. For an on-screen speaker, describe mouth and jaw motion naturally "
            "synchronized to the words, then close the lips after the final line. Do not repeat dialogue in overall_soundscape or non_diegetic_music."
        )
    if mode == "off":
        return (
            "Spoken dialogue is disabled by the Target Profile. Do not include <d> tags, intelligible spoken conversation, narration, or voiceover. "
            "Singing and visible vocal performance are governed separately by the music-performance contract; do not infer, schedule, force, or suppress them from dialogue_mode. "
            "Non-verbal ambience, Foley, and music still follow the selected audio mode."
        )
    guidance_instruction = (
        f" If the brief requests speech, apply this dialogue direction: {guidance}."
        if guidance
        else ""
    )
    speaker_rule = (
        "For Ref2VA, preserve exact requested speech with the speaking <Subject N> and its stable numbered cue such as (S1) repeated together in the same shot before each "
        "<d>[Language] spoken words</d> block; never output literal (Sx) or use a bare speaker cue after multiple Subject tags."
        if ref2va
        else (
            "Preserve exact requested speech with a stable numbered cue such as (S1) in the same shot before each "
            "<d>[Language] spoken words</d> block."
        )
    )
    return (
        f"Dialogue is brief-driven only. {speaker_rule} Do not invent dialogue when the brief does not request it."
        f"{guidance_instruction}"
    )


_MINIMAX_H3_DIALOGUE_PATCH_SCHEMA = "dg-h3-dialogue-patch/1"
_MINIMAX_H3_DIALOGUE_DELIVERIES = (
    "neutral",
    "quiet awe",
    "quiet resolve",
    "calm confidence",
    "firm resolve",
    "gentle warmth",
    "urgent warning",
    "soft hesitation",
    "joyful excitement",
    "somber restraint",
    "playful confidence",
    "breathless urgency",
    "whispered secrecy",
    "clear determination",
)
_MINIMAX_H3_DIALOGUE_LOCAL_REASONS = {
    "minimax_h3_dialogue_outside_timeline",
    "minimax_h3_dialogue_tags_invalid",
    "minimax_h3_dialogue_speaker_invalid",
    "minimax_h3_dialogue_count_mismatch",
    "minimax_h3_dialogue_forbidden",
}


def _build_minimax_h3_dialogue_patch_prompt(
    user_prompt: str,
    candidate_prompt: str,
    missing_line_count: int,
    minimax_h3_mode: str,
    dialogue_guidance: str,
    evidence_context: str = "",
) -> str:
    missing = max(1, min(_MINIMAX_H3_MAX_DIALOGUE_LINE_COUNT, int(missing_line_count)))
    mode = _normalize_minimax_h3_mode(minimax_h3_mode)
    prompt = str(candidate_prompt or "")
    bounds = _minimax_h3_timeline_bounds(prompt, mode)
    timeline = prompt[bounds[0] : bounds[1]].strip() if bounds is not None else ""
    shot_matches = list(re.finditer(r"\[Shot (\d+)\]", timeline))
    shots = [int(match.group(1)) for match in shot_matches]
    shot_bodies = {
        int(match.group(1)): timeline[
            match.end() : shot_matches[index + 1].start() if index + 1 < len(shot_matches) else len(timeline)
        ]
        for index, match in enumerate(shot_matches)
    }
    speakers = sorted(
        {speaker.upper() for speaker in re.findall(r"\((S\d+)\)", timeline, flags=re.IGNORECASE)}
    )
    subjects: list[str] = []
    if mode == "ref2va":
        sections = _minimax_h3_ref_sections(prompt)
        if sections is not None:
            subjects = [
                tag
                for tag, _description in _minimax_h3_ref_definition_items(
                    sections["subject_definitions"]
                )
                if tag.startswith("<Subject ")
            ]
    guidance = _minimax_h3_dialogue_guidance_text(dialogue_guidance)
    evidence_rule = (
        "\nVerified evidence context (do not add visual facts beyond it):\n" + str(evidence_context).strip()
        if str(evidence_context or "").strip()
        else ""
    )
    subject_rule = (
        f"subject_tag must be one of {json.dumps(subjects)} and must already appear in the selected shot."
        if mode == "ref2va"
        else 'subject_tag must be the empty string "".'
    )
    speaker_rule = (
        f"Reuse an existing speaker_id from the selected subject/shot when present ({speakers}); if an existing Ref2VA subject has no speaker ID yet, "
        "use the matching ordinal (for example <Subject 1> uses S1 and <Subject 2> uses S2)."
        if mode == "ref2va"
        else f"speaker_id must already appear in the selected shot; available IDs are {speakers}."
    )
    example_records: list[dict[str, Any]] = []
    example_deliveries = ("quiet awe", "firm resolve", "calm confidence")
    for index in range(missing):
        shot_index = (
            min(len(shots) - 1, round(index * (len(shots) - 1) / max(1, missing - 1)))
            if shots
            else 0
        )
        example_shot = shots[shot_index] if shots else 1
        example_body = shot_bodies.get(example_shot, "")
        example_subject = next(
            (subject for subject in subjects if subject in example_body),
            subjects[0] if mode == "ref2va" and subjects else "",
        )
        subject_ordinal = re.search(r"\d+", example_subject)
        body_speakers = [
            speaker.upper()
            for speaker in re.findall(r"\((S\d+)\)", example_body, flags=re.IGNORECASE)
        ]
        example_speaker = (
            f"S{subject_ordinal.group()}"
            if subject_ordinal is not None
            else body_speakers[0]
            if body_speakers
            else speakers[min(index, len(speakers) - 1)]
            if speakers
            else "S1"
        )
        example_records.append(
            {
                "shot": example_shot,
                "subject_tag": example_subject,
                "speaker_id": example_speaker,
                "language": "English",
                "delivery": example_deliveries[index % len(example_deliveries)],
                "text": f"Concise line {index + 1}.",
            }
        )
    example_payload = {
        "schema": _MINIMAX_H3_DIALOGUE_PATCH_SCHEMA,
        "dialogue_patch": example_records,
    }
    return (
        "You are a dialogue patch compiler. The storyboard below is immutable. Do not rewrite, summarize, or return it. "
        f"Author exactly {missing} missing spoken line{'s' if missing != 1 else ''} as one strict JSON object and nothing else. "
        f"Use only existing shot numbers {shots}. {subject_rule} {speaker_rule} "
        f"language is a short language name such as English. delivery must be exactly one of {json.dumps(list(_MINIMAX_H3_DIALOGUE_DELIVERIES))}. "
        "text contains only the exact words spoken, "
        "with no quotation marks, tags, speaker labels, stage directions, or line breaks. Keep each line concise enough for its existing shot. "
        "Do not add cuts or change camera, timing, action, references, audio sections, or provenance.\n"
        f"Dialogue direction: {guidance or 'Author concise speech that advances the requested action or character intent.'}\n"
        f"Creative brief: {str(user_prompt or '').strip()}\n"
        "Return exactly this shape with no Markdown or extra keys:\n"
        f"{_json_dumps(example_payload)}\n"
        f"IMMUTABLE STORYBOARD:\n{prompt}"
        f"{evidence_rule}"
    )


def _strict_minimax_h3_dialogue_patch_payload(raw_output: Any) -> dict[str, Any] | None:
    text = str(raw_output or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return None
    anchored_thinking = (
        re.compile(r"\A<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL),
        re.compile(r"\A<\|thought\|>.*?<\|end_thought\|>\s*", flags=re.IGNORECASE | re.DOTALL),
    )
    consumed = True
    while consumed:
        consumed = False
        for pattern in anchored_thinking:
            match = pattern.match(text)
            if match:
                text = text[match.end() :].lstrip()
                consumed = True
                break
    lowered = text.casefold()
    matching_open = next(
        (marker for marker in FINAL_JSON_OPEN_MARKERS if lowered.startswith(marker.casefold())),
        None,
    )
    if matching_open is not None:
        matching_close = next(
            (marker for marker in FINAL_JSON_CLOSE_MARKERS if lowered.endswith(marker.casefold())),
            None,
        )
        if matching_close is None:
            return None
        text = text[len(matching_open) : len(text) - len(matching_close)].strip()
    if "```" in text or re.search(r"</?think\b|<\|[^>\n]+\|>", text, flags=re.IGNORECASE):
        return None
    try:
        return _strict_prompt_packet_json(text)
    except _DuplicatePromptPacketKey:
        return None


def _validated_minimax_h3_dialogue_patch_records(
    raw_output: Any,
    candidate_prompt: str,
    minimax_h3_mode: str,
    expected_count: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    payload = _strict_minimax_h3_dialogue_patch_payload(raw_output)
    if payload is None:
        return [], ["dialogue_patch_not_strict_json"]
    if set(payload) != {"schema", "dialogue_patch"} or payload.get("schema") != _MINIMAX_H3_DIALOGUE_PATCH_SCHEMA:
        return [], ["dialogue_patch_schema_invalid"]
    records = payload.get("dialogue_patch")
    if not isinstance(records, list) or len(records) != int(expected_count):
        return [], ["dialogue_patch_count_invalid"]

    prompt = str(candidate_prompt or "")
    mode = _normalize_minimax_h3_mode(minimax_h3_mode)
    bounds = _minimax_h3_timeline_bounds(prompt, mode)
    if bounds is None:
        return [], ["dialogue_patch_timeline_missing"]
    timeline = prompt[bounds[0] : bounds[1]]
    shot_matches = list(re.finditer(r"\[Shot (\d+)\]", timeline))
    shot_bodies = {
        int(match.group(1)): timeline[
            match.end() : shot_matches[index + 1].start() if index + 1 < len(shot_matches) else len(timeline)
        ]
        for index, match in enumerate(shot_matches)
    }
    subject_tags: set[str] = set()
    subject_speakers: dict[str, str] = {}
    speaker_subjects: dict[str, str] = {}
    if mode == "ref2va":
        sections = _minimax_h3_ref_sections(prompt)
        if sections is None:
            return [], ["dialogue_patch_reference_sections_invalid"]
        subject_tags = {
            tag
            for tag, _description in _minimax_h3_ref_definition_items(
                sections["subject_definitions"]
            )
            if tag.startswith("<Subject ")
        }
        for tag, speaker in re.findall(
            r"(<Subject \d+>)\s*\((S\d+)\)",
            timeline,
            flags=re.IGNORECASE,
        ):
            tag = re.sub(
                r"(?i)<subject\s+(\d+)>",
                lambda match: f"<Subject {int(match.group(1))}>",
                tag,
            )
            speaker = speaker.upper()
            if tag in subject_speakers and subject_speakers[tag] != speaker:
                return [], ["dialogue_patch_existing_speaker_mapping_ambiguous"]
            if speaker in speaker_subjects and speaker_subjects[speaker] != tag:
                return [], ["dialogue_patch_existing_speaker_mapping_ambiguous"]
            subject_speakers[tag] = speaker
            speaker_subjects[speaker] = tag

    safe_text_pattern = re.compile(
        r"(?:\[Shot\b|subject_definitions\s*:|summary\s*:|retention_analysis\s*:|"
        r"detailed_description\s*:|integrated_multimodal_description\s*:|overall_soundscape\s*:|"
        r"non_diegetic_music\s*:|GROUNDING_EVIDENCE_REPORT_ID|USED_GROUNDING_FACT_IDS|<\||\|>)",
        flags=re.IGNORECASE,
    )
    validated: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for record in records:
        expected_keys = {"shot", "subject_tag", "speaker_id", "language", "delivery", "text"}
        if not isinstance(record, dict) or set(record) != expected_keys:
            return [], ["dialogue_patch_record_shape_invalid"]
        shot = record.get("shot")
        if isinstance(shot, bool) or not isinstance(shot, int) or shot not in shot_bodies:
            return [], ["dialogue_patch_shot_invalid"]
        subject_tag = record.get("subject_tag")
        speaker_id = record.get("speaker_id")
        language = record.get("language")
        delivery = record.get("delivery")
        spoken_text = record.get("text")
        if not all(isinstance(value, str) for value in (subject_tag, speaker_id, language, delivery, spoken_text)):
            return [], ["dialogue_patch_record_type_invalid"]
        if not re.fullmatch(r"S[1-9]\d*", speaker_id, flags=re.IGNORECASE):
            return [], ["dialogue_patch_speaker_invalid"]
        speaker_id = speaker_id.upper()
        if not re.fullmatch(r"[A-Za-z][A-Za-z -]{0,31}", language):
            return [], ["dialogue_patch_language_invalid"]
        normalized_delivery = delivery.strip().casefold()
        if normalized_delivery not in _MINIMAX_H3_DIALOGUE_DELIVERIES:
            return [], ["dialogue_patch_delivery_invalid"]
        if (
            not spoken_text.strip()
            or spoken_text != spoken_text.strip()
            or not any(character.isalnum() for character in spoken_text)
            or len(spoken_text) > 200
            or len(re.findall(r"\b[\w'’\-]+\b", spoken_text)) > 30
            or any(character in spoken_text for character in "<>[]\r\n")
            or any(unicodedata.category(character).startswith("C") for character in spoken_text)
            or safe_text_pattern.search(spoken_text)
        ):
            return [], ["dialogue_patch_text_invalid"]
        body = shot_bodies[shot]
        if mode == "ref2va":
            if subject_tag not in subject_tags or subject_tag not in body:
                return [], ["dialogue_patch_subject_invalid"]
            subject_ordinal_match = re.fullmatch(r"<Subject ([1-9]\d*)>", subject_tag)
            expected_unmapped_speaker = (
                f"S{subject_ordinal_match.group(1)}"
                if subject_ordinal_match is not None
                else ""
            )
            if subject_tag in subject_speakers and subject_speakers[subject_tag] != speaker_id:
                return [], ["dialogue_patch_speaker_subject_mismatch"]
            if subject_tag not in subject_speakers and speaker_id != expected_unmapped_speaker:
                return [], ["dialogue_patch_speaker_subject_mismatch"]
            if speaker_id in speaker_subjects and speaker_subjects[speaker_id] != subject_tag:
                return [], ["dialogue_patch_speaker_subject_mismatch"]
            body_speakers = {
                speaker.upper()
                for speaker in re.findall(r"\((S\d+)\)", body, flags=re.IGNORECASE)
            }
            if body_speakers and speaker_id not in body_speakers:
                return [], ["dialogue_patch_speaker_invalid"]
        elif subject_tag != "" or not re.search(
            rf"\({re.escape(speaker_id)}\)",
            body,
            flags=re.IGNORECASE,
        ):
            return [], ["dialogue_patch_speaker_invalid"]
        identity = (shot, speaker_id, spoken_text.casefold())
        if identity in seen:
            return [], ["dialogue_patch_duplicate_record"]
        seen.add(identity)
        validated.append(
            {
                "shot": shot,
                "subject_tag": subject_tag,
                "speaker_id": speaker_id,
                "language": language,
                "delivery": normalized_delivery,
                "text": spoken_text,
            }
        )
    return validated, []


def _apply_minimax_h3_dialogue_patch_records(
    candidate_prompt: str,
    minimax_h3_mode: str,
    records: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    prompt = str(candidate_prompt or "")
    mode = _normalize_minimax_h3_mode(minimax_h3_mode)
    bounds = _minimax_h3_timeline_bounds(prompt, mode)
    if bounds is None:
        return prompt, [], ["dialogue_patch_timeline_missing"]
    timeline_start, timeline_end = bounds
    timeline = prompt[timeline_start:timeline_end]
    shot_matches = list(re.finditer(r"\[Shot (\d+)\]", timeline))
    shot_ranges = {
        int(match.group(1)): (
            match.end(),
            shot_matches[index + 1].start() if index + 1 < len(shot_matches) else len(timeline),
        )
        for index, match in enumerate(shot_matches)
    }
    by_shot: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_shot.setdefault(int(record["shot"]), []).append(record)
    insertions: list[tuple[int, str, int]] = []
    operation_report: list[dict[str, Any]] = []
    for shot, shot_records in by_shot.items():
        if shot not in shot_ranges:
            return prompt, [], ["dialogue_patch_shot_invalid"]
        body_start, body_end = shot_ranges[shot]
        body = timeline[body_start:body_end]
        final_hold = re.search(r"\bThe final frame holds on\b", body, flags=re.IGNORECASE)
        local_position = final_hold.start() if final_hold else len(body.rstrip())
        absolute_position = timeline_start + body_start + local_position
        sentences: list[str] = []
        for record in shot_records:
            speaker_prefix = (
                f"{record['subject_tag']} ({record['speaker_id']})"
                if mode == "ref2va"
                else f"({record['speaker_id']})"
            )
            sentence = (
                f"{speaker_prefix} speaks on-screen with {record['delivery']}, mouth and jaw synchronized to the words: "
                f"<d>[{record['language']}] {record['text']}</d> Their lips close after the line."
            )
            sentences.append(sentence)
            operation_report.append(
                {
                    "shot": shot,
                    "speaker_id": record["speaker_id"],
                    "language": record["language"],
                    "text_sha256": hashlib.sha256(record["text"].encode("utf-8")).hexdigest(),
                }
            )
        insertion = " ".join(sentences)
        prefix = "" if absolute_position > 0 and prompt[absolute_position - 1].isspace() else " "
        suffix = "" if absolute_position < len(prompt) and prompt[absolute_position].isspace() else " "
        insertions.append((absolute_position, f"{prefix}{insertion}{suffix}", shot))

    patched = prompt
    applied: list[tuple[int, str]] = []
    for position, insertion, _shot in sorted(insertions, key=lambda item: item[0], reverse=True):
        patched = patched[:position] + insertion + patched[position:]
        applied.append((position, insertion))
    reconstructed = patched
    for position, insertion in sorted(applied, key=lambda item: item[0]):
        if reconstructed[position : position + len(insertion)] != insertion:
            return prompt, [], ["dialogue_patch_immutability_check_failed"]
        reconstructed = reconstructed[:position] + reconstructed[position + len(insertion) :]
    if reconstructed != prompt:
        return prompt, [], ["dialogue_patch_immutability_check_failed"]
    return patched, operation_report, []


_MINIMAX_H3_REF2VA_CONTRACT_PATCH_SCHEMA = "dg-h3-ref2va-contract-patch/1"
_MINIMAX_H3_REF2VA_CONTRACT_LOCAL_REASONS = {
    "minimax_h3_ref_subject_definition_invalid",
    "minimax_h3_dialogue_speaker_invalid",
}


def _minimax_h3_ref2va_subject_definition_bounds(
    prompt_text: Any,
) -> tuple[int, int] | None:
    match = re.match(
        r"(?ms)\Asubject_definitions\s*:\s*\n(?P<body>.*?)\n\nsummary\s*:",
        str(prompt_text or ""),
    )
    if match is None:
        return None
    return match.start("body"), match.end("body")


def _minimax_h3_ref2va_subject_tags(prompt_text: Any) -> list[str]:
    ordinals = {
        int(ordinal)
        for ordinal in re.findall(
            r"<\s*Subject\s+([1-9]\d*)\s*>",
            str(prompt_text or ""),
            flags=re.IGNORECASE,
        )
    }
    return [f"<Subject {ordinal}>" for ordinal in sorted(ordinals)]


def _minimax_h3_ref2va_dialogue_anchors(prompt_text: Any) -> list[dict[str, Any]]:
    prompt = str(prompt_text or "")
    bounds = _minimax_h3_timeline_bounds(prompt, "ref2va")
    if bounds is None:
        return []
    timeline_start, timeline_end = bounds
    timeline = prompt[timeline_start:timeline_end]
    shots = list(re.finditer(r"\[Shot ([1-9]\d*)\]", timeline))
    dialogue_pattern = re.compile(
        r"<d>\[[^\]\n]+\]\s*\S(?:(?!</?d>|\[Shot \d+\]).)*?</d>",
        flags=re.DOTALL,
    )
    anchors: list[dict[str, Any]] = []
    for ordinal, dialogue in enumerate(dialogue_pattern.finditer(timeline), start=1):
        shot_index = max(
            (index for index, shot in enumerate(shots) if shot.start() < dialogue.start()),
            default=-1,
        )
        if shot_index < 0:
            continue
        shot = shots[shot_index]
        shot_start = shot.end()
        shot_end = shots[shot_index + 1].start() if shot_index + 1 < len(shots) else len(timeline)
        preceding = timeline[shot_start:dialogue.start()]
        subject_tags = [
            f"<Subject {int(value)}>"
            for value in re.findall(
                r"<\s*Subject\s+([1-9]\d*)\s*>",
                preceding,
                flags=re.IGNORECASE,
            )
        ]
        subject_tags = list(dict.fromkeys(subject_tags))
        speaker_pairs = [
            (f"<Subject {int(subject_ordinal)}>", speaker.upper())
            for subject_ordinal, speaker in re.findall(
                r"<\s*Subject\s+([1-9]\d*)\s*>\s*\((S[1-9]\d*)\)",
                preceding,
                flags=re.IGNORECASE,
            )
        ]
        recognized_speaker = re.search(
            r"\((?P<speaker_ids>S[1-9]\d*(?:\s*,\s*S[1-9]\d*)*)\)[^<]{0,500}$",
            preceding[-500:],
            flags=re.IGNORECASE,
        )
        recognized_speaker_ids = (
            [
                speaker_id.upper()
                for speaker_id in re.findall(
                    r"S[1-9]\d*",
                    recognized_speaker.group("speaker_ids"),
                    flags=re.IGNORECASE,
                )
            ]
            if recognized_speaker is not None
            else []
        )
        exact_block = dialogue.group(0)
        anchors.append(
            {
                "ordinal": ordinal,
                "shot": int(shot.group(1)),
                "dialogue_sha256": hashlib.sha256(exact_block.encode("utf-8")).hexdigest(),
                "dialogue_text": exact_block,
                "start": timeline_start + dialogue.start(),
                "end": timeline_start + dialogue.end(),
                "shot_end": timeline_start + shot_end,
                "candidate_subject_tags": subject_tags,
                "speaker_pairs": speaker_pairs,
                "recognized_speaker_ids": recognized_speaker_ids,
                "speaker_missing": recognized_speaker is None,
            }
        )
    return anchors


def _minimax_h3_ref2va_global_speaker_mappings(
    prompt_text: Any,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    sections = _minimax_h3_ref_sections(str(prompt_text or ""))
    detailed = sections["detailed_description"] if sections is not None else ""
    subject_speakers: dict[str, str] = {}
    speaker_subjects: dict[str, str] = {}
    reasons: list[str] = []

    def register_mapping(subject_tag: str, speaker_id: str) -> None:
        speaker_id = speaker_id.upper()
        if (
            subject_tag in subject_speakers
            and subject_speakers[subject_tag] != speaker_id
        ) or (
            speaker_id in speaker_subjects
            and speaker_subjects[speaker_id] != subject_tag
        ):
            reasons.append(
                "ref2va_contract_patch_existing_speaker_mapping_ambiguous"
            )
            return
        subject_speakers[subject_tag] = speaker_id
        speaker_subjects[speaker_id] = subject_tag

    for subject_ordinal, speaker_id in re.findall(
        r"<\s*Subject\s+([1-9]\d*)\s*>\s*\((S[1-9]\d*)\)",
        detailed,
        flags=re.IGNORECASE,
    ):
        subject_tag = f"<Subject {int(subject_ordinal)}>"
        register_mapping(subject_tag, speaker_id)
        if reasons:
            break
    if not reasons:
        for anchor in _minimax_h3_ref2va_dialogue_anchors(prompt_text):
            if anchor["speaker_missing"]:
                continue
            subject_tags = list(dict.fromkeys(anchor["candidate_subject_tags"]))
            speaker_ids = list(dict.fromkeys(anchor["recognized_speaker_ids"]))
            if not subject_tags or not speaker_ids:
                reasons.append(
                    "ref2va_contract_patch_existing_speaker_mapping_ambiguous"
                )
                break
            for speaker_id in speaker_ids:
                bound_subject = speaker_subjects.get(speaker_id)
                if bound_subject is not None:
                    if bound_subject not in subject_tags:
                        reasons.append(
                            "ref2va_contract_patch_existing_speaker_mapping_ambiguous"
                        )
                        break
                    continue
                if len(speaker_ids) != 1 or len(subject_tags) != 1:
                    reasons.append(
                        "ref2va_contract_patch_existing_speaker_mapping_ambiguous"
                    )
                    break
                register_mapping(subject_tags[0], speaker_id)
                if reasons:
                    break
            if reasons:
                break
    return subject_speakers, speaker_subjects, list(dict.fromkeys(reasons))


def _minimax_h3_ref2va_grounding_facts(
    verified_ledger: Any,
) -> dict[str, dict[str, Any]]:
    ledger = verified_ledger if isinstance(verified_ledger, dict) else {}
    facts: dict[str, dict[str, Any]] = {}
    for item in ledger.get("observed_facts", []):
        if not isinstance(item, dict) or str(item.get("confidence", "")).casefold() not in {
            "high",
            "medium",
        }:
            continue
        fact_id = str(item.get("fact_id", "")).strip()
        claim = str(item.get("claim", "")).strip()
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}", fact_id)
            or not claim
            or "\n" in claim
            or "\r" in claim
            or any(unicodedata.category(character).startswith("C") for character in claim)
            or re.search(
                r"<\s*(?:Subject|Picture|Video|Audio)\s+\d+\s*>|"
                r"(?:subject_definitions|summary|retention_analysis|detailed_description|"
                r"overall_soundscape|non_diegetic_music)\s*:|"
                r"\[\s*Shot\b|</?\s*d\s*>|&lt;/?\s*d\s*&gt;|"
                r"\(\s*S[1-9]\d*(?:\s*,\s*S[1-9]\d*)*\s*\)|"
                r"\b(?:GROUNDING_EVIDENCE_REPORT_ID|USED_GROUNDING_FACT_IDS)\s*:|"
                r"<\s*/?\s*think\b|<\|[^\n>]{0,64}\|>",
                claim,
                flags=re.IGNORECASE,
            )
        ):
            continue
        asset_ids = {
            str(evidence.get("asset_id", "")).strip().casefold()
            for evidence in item.get("evidence", [])
            if isinstance(evidence, dict) and str(evidence.get("asset_id", "")).strip()
        }
        if asset_ids:
            facts[fact_id] = {"claim": claim, "asset_ids": asset_ids}
    return facts


def _minimax_h3_reference_tag_asset_ids(tag: str) -> set[str]:
    match = re.fullmatch(r"<(Picture|Video|Audio) ([1-9]\d*)>", str(tag or ""))
    if match is None:
        return set()
    kind, ordinal = match.groups()
    normalized = kind.casefold()
    if normalized == "picture":
        return {f"picture:{ordinal}", f"image:{ordinal}"}
    return {f"{normalized}:{ordinal}"}


def _minimax_h3_ref2va_contract_patch_feasibility_reasons(
    candidate_prompt: str,
    failure_reasons: list[str],
    reference_manifest: str,
    verified_ledger: dict[str, Any],
) -> list[str]:
    failure_set = set(failure_reasons)
    if not failure_set or not failure_set.issubset(
        _MINIMAX_H3_REF2VA_CONTRACT_LOCAL_REASONS
    ):
        return ["ref2va_contract_patch_trigger_invalid"]
    subject_tags = _minimax_h3_ref2va_subject_tags(candidate_prompt)
    if not subject_tags or [int(re.search(r"\d+", tag).group()) for tag in subject_tags] != list(
        range(1, len(subject_tags) + 1)
    ):
        return ["ref2va_contract_patch_subject_set_invalid"]

    reasons: list[str] = []
    if "minimax_h3_ref_subject_definition_invalid" in failure_set:
        if _minimax_h3_ref2va_subject_definition_bounds(candidate_prompt) is None:
            reasons.append("ref2va_contract_patch_definition_bounds_invalid")
        declared_sources = set(_minimax_h3_reference_tags(reference_manifest))
        facts = _minimax_h3_ref2va_grounding_facts(verified_ledger)
        if not declared_sources or any(tag.startswith("<Audio ") for tag in declared_sources):
            reasons.append("ref2va_contract_patch_source_scope_unsupported")
        elif not facts or any(
            not any(
                _minimax_h3_reference_tag_asset_ids(source_tag) & fact["asset_ids"]
                for fact in facts.values()
            )
            for source_tag in declared_sources
        ):
            reasons.append("ref2va_contract_patch_grounding_scope_unavailable")

    if "minimax_h3_dialogue_speaker_invalid" in failure_set:
        anchors = _minimax_h3_ref2va_dialogue_anchors(candidate_prompt)
        missing = [anchor for anchor in anchors if anchor["speaker_missing"]]
        identities = {
            (anchor["shot"], anchor["dialogue_sha256"])
            for anchor in missing
        }
        if (
            not missing
            or len(identities) != len(missing)
            or any(len(anchor["candidate_subject_tags"]) != 1 for anchor in missing)
        ):
            reasons.append("ref2va_contract_patch_dialogue_anchor_ambiguous")
        (
            subject_speakers,
            speaker_subjects,
            mapping_reasons,
        ) = _minimax_h3_ref2va_global_speaker_mappings(candidate_prompt)
        reasons.extend(mapping_reasons)
        for anchor in missing:
            if len(anchor["candidate_subject_tags"]) != 1:
                continue
            subject_tag = anchor["candidate_subject_tags"][0]
            subject_ordinal = int(re.search(r"\d+", subject_tag).group())
            speaker_id = subject_speakers.get(subject_tag, f"S{subject_ordinal}")
            if (
                speaker_id in speaker_subjects
                and speaker_subjects[speaker_id] != subject_tag
            ):
                reasons.append("ref2va_contract_patch_speaker_mapping_conflict")
                continue
            subject_speakers[subject_tag] = speaker_id
            speaker_subjects[speaker_id] = subject_tag
    return list(dict.fromkeys(reasons))


def _build_minimax_h3_ref2va_contract_patch_prompt(
    user_prompt: str,
    candidate_prompt: str,
    reference_manifest: str,
    evidence_report_id: str,
    verified_ledger: dict[str, Any],
    failure_reasons: list[str],
) -> str:
    failure_set = set(failure_reasons)
    definition_patch_required = (
        "minimax_h3_ref_subject_definition_invalid" in failure_set
    )
    speaker_patch_required = "minimax_h3_dialogue_speaker_invalid" in failure_set
    subjects = _minimax_h3_ref2va_subject_tags(candidate_prompt)
    declared_sources = _minimax_h3_reference_tags(reference_manifest)
    facts = _minimax_h3_ref2va_grounding_facts(verified_ledger)
    missing_anchors = [
        {
            "shot": anchor["shot"],
            "dialogue_sha256": anchor["dialogue_sha256"],
            "dialogue_text": anchor["dialogue_text"],
            "candidate_subject_tags": anchor["candidate_subject_tags"],
        }
        for anchor in _minimax_h3_ref2va_dialogue_anchors(candidate_prompt)
        if anchor["speaker_missing"]
    ]
    example_fact_ids = list(facts)[:1]
    example_sources = declared_sources[:1]
    example_subjects = [
        {
            "subject_tag": subject,
            "source_tags": example_sources,
            "grounding_fact_ids": example_fact_ids,
        }
        for subject in subjects
    ] if definition_patch_required else []
    example_speakers = [
        {
            "shot": anchor["shot"],
            "dialogue_sha256": anchor["dialogue_sha256"],
            "subject_tag": (
                anchor["candidate_subject_tags"][-1]
                if anchor["candidate_subject_tags"]
                else subjects[0]
                if subjects
                else "<Subject 1>"
            ),
        }
        for anchor in missing_anchors
    ] if speaker_patch_required else []
    example = {
        "schema": _MINIMAX_H3_REF2VA_CONTRACT_PATCH_SCHEMA,
        "evidence_report_id": evidence_report_id,
        "subject_definitions": example_subjects,
        "dialogue_speakers": example_speakers,
    }
    return (
        "You are a surgical MiniMax H3 Ref2VA contract patch compiler. The storyboard is immutable. Return one strict JSON object and nothing else. "
        "Do not rewrite any shot, timestamp, camera/action text, dialogue words or tags, soundscape, music, or provenance. "
        "Only when minimax_h3_ref_subject_definition_invalid is listed, provide one definition-selection record for every used Subject tag using verified fact IDs and declared source tags; the host copies the exact verified claims. Otherwise subject_definitions must be an empty array and the host preserves that section byte-for-byte. "
        "Only when minimax_h3_dialogue_speaker_invalid is listed, provide one record for every listed dialogue block missing a speaker cue. The selected Subject must be the sole Subject tag before that block in the same shot; the host derives the stable speaker ID and inserts only its canonical numbered cue such as (S1). Otherwise dialogue_speakers must be an empty array. "
        "Never invent, omit, or renumber a Subject, asset, fact, shot, or dialogue block. Use every root and record key exactly once and add no extra keys.\n"
        f"Failures to repair: {_json_dumps(failure_reasons)}\n"
        f"Creative brief: {str(user_prompt or '').strip()}\n"
        f"Reference manifest: {reference_manifest}\n"
        f"Evidence report ID: {evidence_report_id}\n"
        f"Verified facts: {_json_dumps(verified_ledger)}\n"
        f"Used Subject tags: {_json_dumps(subjects)}\n"
        f"Dialogue blocks missing cues: {_json_dumps(missing_anchors)}\n"
        "Return exactly this shape, replacing only the example selections:\n"
        f"{_json_dumps(example)}\n"
        f"IMMUTABLE CANDIDATE:\n{candidate_prompt}"
    )


def _validated_minimax_h3_ref2va_contract_patch(
    raw_output: Any,
    candidate_prompt: str,
    reference_manifest: str,
    evidence_report_id: str,
    verified_ledger: dict[str, Any],
    failure_reasons: list[str],
) -> tuple[dict[str, Any], list[str]]:
    payload = _strict_minimax_h3_dialogue_patch_payload(raw_output)
    if payload is None:
        return {}, ["ref2va_contract_patch_not_strict_json"]
    if set(payload) != {
        "schema",
        "evidence_report_id",
        "subject_definitions",
        "dialogue_speakers",
    } or payload.get("schema") != _MINIMAX_H3_REF2VA_CONTRACT_PATCH_SCHEMA:
        return {}, ["ref2va_contract_patch_schema_invalid"]
    if payload.get("evidence_report_id") != evidence_report_id:
        return {}, ["ref2va_contract_patch_evidence_report_mismatch"]

    failure_set = set(failure_reasons)
    definition_patch_required = (
        "minimax_h3_ref_subject_definition_invalid" in failure_set
    )
    speaker_patch_required = "minimax_h3_dialogue_speaker_invalid" in failure_set
    if not (definition_patch_required or speaker_patch_required):
        return {}, ["ref2va_contract_patch_trigger_invalid"]

    subject_tags = _minimax_h3_ref2va_subject_tags(candidate_prompt)
    if not subject_tags or [int(re.search(r"\d+", tag).group()) for tag in subject_tags] != list(
        range(1, len(subject_tags) + 1)
    ):
        return {}, ["ref2va_contract_patch_subject_set_invalid"]
    declared_sources = set(_minimax_h3_reference_tags(reference_manifest))
    if definition_patch_required and (
        not declared_sources or any(tag.startswith("<Audio ") for tag in declared_sources)
    ):
        return {}, ["ref2va_contract_patch_source_scope_unsupported"]
    facts = _minimax_h3_ref2va_grounding_facts(verified_ledger)
    definitions = payload.get("subject_definitions")
    if not isinstance(definitions, list) or (
        definition_patch_required and len(definitions) != len(subject_tags)
    ) or (not definition_patch_required and definitions):
        return {}, ["ref2va_contract_patch_definition_count_invalid"]
    rendered_definitions: list[str] = []
    validated_definitions: list[dict[str, Any]] = []
    seen_subjects: set[str] = set()
    covered_sources: set[str] = set()
    for record in definitions:
        if not isinstance(record, dict) or set(record) != {
            "subject_tag",
            "source_tags",
            "grounding_fact_ids",
        }:
            return {}, ["ref2va_contract_patch_definition_shape_invalid"]
        subject_tag = record.get("subject_tag")
        source_tags = record.get("source_tags")
        fact_ids = record.get("grounding_fact_ids")
        if subject_tag not in subject_tags or subject_tag in seen_subjects:
            return {}, ["ref2va_contract_patch_subject_invalid"]
        if (
            not isinstance(source_tags, list)
            or not source_tags
            or any(not isinstance(tag, str) for tag in source_tags)
            or len(source_tags) != len(set(source_tags))
            or not set(source_tags).issubset(declared_sources)
        ):
            return {}, ["ref2va_contract_patch_source_invalid"]
        if (
            not isinstance(fact_ids, list)
            or not fact_ids
            or any(not isinstance(fact_id, str) for fact_id in fact_ids)
            or len(fact_ids) != len(set(fact_ids))
            or any(fact_id not in facts for fact_id in fact_ids)
        ):
            return {}, ["ref2va_contract_patch_fact_invalid"]
        selected_asset_ids = set().union(*(facts[fact_id]["asset_ids"] for fact_id in fact_ids))
        record_asset_ids = set().union(
            *(_minimax_h3_reference_tag_asset_ids(source_tag) for source_tag in source_tags)
        )
        if any(
            not (_minimax_h3_reference_tag_asset_ids(source_tag) & selected_asset_ids)
            for source_tag in source_tags
        ) or any(
            not (facts[fact_id]["asset_ids"] & record_asset_ids)
            for fact_id in fact_ids
        ):
            return {}, ["ref2va_contract_patch_fact_source_mismatch"]
        claims = [facts[fact_id]["claim"] for fact_id in fact_ids]
        description = " ".join(dict.fromkeys(claims)).strip()
        if len(re.findall(r"[A-Za-z0-9]{2,}", description)) < 4:
            return {}, ["ref2va_contract_patch_definition_claim_short"]
        source_text = ", ".join(source_tags)
        rendered_definitions.append(
            f"{subject_tag}: {description} Reference source: {source_text}."
        )
        validated_definitions.append(
            {
                "subject_tag": subject_tag,
                "source_tags": list(source_tags),
                "grounding_fact_ids": list(fact_ids),
            }
        )
        seen_subjects.add(subject_tag)
        covered_sources.update(source_tags)
    if definition_patch_required and (
        seen_subjects != set(subject_tags) or covered_sources != declared_sources
    ):
        return {}, ["ref2va_contract_patch_coverage_invalid"]

    anchors = _minimax_h3_ref2va_dialogue_anchors(candidate_prompt)
    missing = [anchor for anchor in anchors if anchor["speaker_missing"]]
    speaker_records = payload.get("dialogue_speakers")
    if not isinstance(speaker_records, list) or (
        speaker_patch_required and len(speaker_records) != len(missing)
    ) or (not speaker_patch_required and speaker_records):
        return {}, ["ref2va_contract_patch_speaker_count_invalid"]
    anchor_index = {
        (anchor["shot"], anchor["dialogue_sha256"]): anchor
        for anchor in missing
    }
    if len(anchor_index) != len(missing):
        return {}, ["ref2va_contract_patch_dialogue_anchor_ambiguous"]
    validated_speakers: list[dict[str, Any]] = []
    seen_anchors: set[tuple[int, str]] = set()
    (
        existing_subject_speakers,
        existing_speaker_subjects,
        mapping_reasons,
    ) = _minimax_h3_ref2va_global_speaker_mappings(candidate_prompt)
    if mapping_reasons:
        return {}, list(mapping_reasons)
    for record in speaker_records:
        if not isinstance(record, dict) or set(record) != {
            "shot",
            "dialogue_sha256",
            "subject_tag",
        }:
            return {}, ["ref2va_contract_patch_speaker_shape_invalid"]
        shot = record.get("shot")
        dialogue_sha256 = record.get("dialogue_sha256")
        subject_tag = record.get("subject_tag")
        identity = (shot, dialogue_sha256)
        anchor = anchor_index.get(identity)
        if (
            isinstance(shot, bool)
            or not isinstance(shot, int)
            or not isinstance(dialogue_sha256, str)
            or subject_tag not in subject_tags
            or anchor is None
            or identity in seen_anchors
            or subject_tag not in anchor["candidate_subject_tags"]
            or len(anchor["candidate_subject_tags"]) != 1
        ):
            return {}, ["ref2va_contract_patch_speaker_anchor_invalid"]
        subject_ordinal = int(re.search(r"\d+", subject_tag).group())
        speaker_id = existing_subject_speakers.get(subject_tag, f"S{subject_ordinal}")
        if speaker_id in existing_speaker_subjects and existing_speaker_subjects[speaker_id] != subject_tag:
            return {}, ["ref2va_contract_patch_speaker_mapping_conflict"]
        existing_subject_speakers[subject_tag] = speaker_id
        existing_speaker_subjects[speaker_id] = subject_tag
        validated_speakers.append(
            {
                "shot": shot,
                "dialogue_sha256": dialogue_sha256,
                "subject_tag": subject_tag,
                "speaker_id": speaker_id,
            }
        )
        seen_anchors.add(identity)
    if seen_anchors != set(anchor_index):
        return {}, ["ref2va_contract_patch_speaker_coverage_invalid"]
    return {
        "subject_definitions": "\n".join(rendered_definitions),
        "replace_subject_definitions": definition_patch_required,
        "definition_records": validated_definitions,
        "dialogue_speakers": validated_speakers,
        "grounding_fact_ids": sorted(
            {
                fact_id
                for record in validated_definitions
                for fact_id in record["grounding_fact_ids"]
            }
        ),
    }, []


def _apply_minimax_h3_ref2va_contract_patch(
    candidate_prompt: str,
    patch_plan: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    prompt = str(candidate_prompt or "")
    replace_definitions = bool(patch_plan.get("replace_subject_definitions"))
    original_bounds = (
        _minimax_h3_ref2va_subject_definition_bounds(prompt)
        if replace_definitions
        else None
    )
    replacement = str(patch_plan.get("subject_definitions", "") or "")
    if replace_definitions and (original_bounds is None or not replacement):
        return prompt, [], ["ref2va_contract_patch_definition_bounds_invalid"]
    if replace_definitions:
        original_start, original_end = original_bounds
        definition_patched = prompt[:original_start] + replacement + prompt[original_end:]
    else:
        original_start = original_end = -1
        definition_patched = prompt
    anchors = {
        (anchor["shot"], anchor["dialogue_sha256"]): anchor
        for anchor in _minimax_h3_ref2va_dialogue_anchors(definition_patched)
    }
    insertions: list[tuple[int, str, dict[str, Any]]] = []
    operations: list[dict[str, Any]] = []
    if replace_definitions:
        operations.append({
            "operation": "replace_subject_definitions",
            "definition_count": len(patch_plan.get("definition_records", [])),
            "before_sha256": hashlib.sha256(
                prompt[original_start:original_end].encode("utf-8")
            ).hexdigest(),
            "after_sha256": hashlib.sha256(replacement.encode("utf-8")).hexdigest(),
        })
    for record in patch_plan.get("dialogue_speakers", []):
        identity = (record["shot"], record["dialogue_sha256"])
        anchor = anchors.get(identity)
        if anchor is None or not anchor["speaker_missing"]:
            return prompt, [], ["ref2va_contract_patch_dialogue_anchor_changed"]
        insertion = f"({record['speaker_id']}) "
        insertions.append((anchor["start"], insertion, record))
    patched = definition_patched
    applied: list[tuple[int, str]] = []
    for position, insertion, record in sorted(insertions, key=lambda item: item[0], reverse=True):
        patched = patched[:position] + insertion + patched[position:]
        applied.append((position, insertion))
        operations.append(
            {
                "operation": "insert_dialogue_speaker",
                "shot": record["shot"],
                "dialogue_sha256": record["dialogue_sha256"],
                "subject_tag": record["subject_tag"],
                "speaker_id": record["speaker_id"],
            }
        )
    reconstructed = patched
    for position, insertion in sorted(applied, key=lambda item: item[0]):
        if reconstructed[position : position + len(insertion)] != insertion:
            return prompt, [], ["ref2va_contract_patch_immutability_check_failed"]
        reconstructed = reconstructed[:position] + reconstructed[position + len(insertion) :]
    if reconstructed != definition_patched:
        return prompt, [], ["ref2va_contract_patch_immutability_check_failed"]
    if replace_definitions:
        patched_bounds = _minimax_h3_ref2va_subject_definition_bounds(reconstructed)
        if patched_bounds is None:
            return prompt, [], ["ref2va_contract_patch_immutability_check_failed"]
        patched_start, patched_end = patched_bounds
        restored = reconstructed[:patched_start] + prompt[original_start:original_end] + reconstructed[patched_end:]
        if restored != prompt:
            return prompt, [], ["ref2va_contract_patch_immutability_check_failed"]
    elif reconstructed != prompt:
        return prompt, [], ["ref2va_contract_patch_immutability_check_failed"]
    return patched, operations, []


def _minimax_h3_shot_has_camera_spec(shot_body: str) -> bool:
    return bool(
        re.search(
            r"\b(?:extreme\s+close[- ]up|close[- ]up|medium(?:[- ](?:close|wide))?\s+shot|wide\s+shot|long\s+shot|full[- ]body\s+shot|"
            r"two[- ]shot|over[- ]the[- ]shoulder|pov|point[- ]of[- ]view|bird'?s[- ]eye|overhead|high[- ]angle|low[- ]angle|dutch[- ]angle|"
            r"profile(?:\s+view)?|locked[- ]off|static(?:\s+hold)?|same\s+framing\s+holds|tracking|tracks?|push(?:es)?\s+in|pull(?:s)?\s+back|"
            r"zooms?|pans?|tilts?|trucks?|doll(?:y|ies)|pedestal|arcs?|orbits?|cranes?|handheld|steadicam|gimbal|"
            r"whip[- ]?pans?|pronounced(?:\s+or\s+layered)?\s+parallax|layered\s+parallax|camera-induced\s+parallax|"
            r"camera\s+(?:holds?|shakes?|follows?|moves?|rises?|descends?|rolls?|rotates?|spins?|swirls?|sweeps?|circles?|whips?|travels?)|"
            r"viewpoint\s+(?:rolls?|rotates?|spins?|swirls?|sweeps?|circles?)|"
            r"(?:rolling|rotating|spinning|swirling|sweeping|circling)\s+camera)\b",
            shot_body,
            flags=re.IGNORECASE,
        )
    )


_MINIMAX_H3_SAFE_CAMERA_HOLD = "The camera remains locked-off in a stable view."


def _repair_minimax_h3_unspecified_shot_cameras(
    prompt_text: Any,
    minimax_h3_mode: Any = "t2va",
) -> tuple[str, list[int]]:
    """Add a deterministic static camera clause to otherwise unspecified shots.

    Camera motion is creative content, so the host must not invent a pan, orbit,
    zoom, or other path merely to satisfy the H3 contract. A locked-off hold is
    the conservative fallback: it makes the camera behavior explicit while
    leaving every model-authored byte and every already-specified shot intact.
    Malformed prompts remain unchanged for the normal fail-closed validators.
    """

    prompt = str(prompt_text or "")
    bounds = _minimax_h3_timeline_bounds(prompt, minimax_h3_mode)
    if bounds is None:
        return prompt, []

    timeline_start, timeline_end = bounds
    timeline = prompt[timeline_start:timeline_end]
    shots = list(
        re.finditer(r"\[Shot ([1-9]\d*)\]", timeline, flags=re.IGNORECASE)
    )
    if not shots:
        return prompt, []

    insertions: list[tuple[int, str]] = []
    repaired_shots: list[int] = []
    for index, shot in enumerate(shots):
        body_start = shot.end()
        body_end = shots[index + 1].start() if index + 1 < len(shots) else len(timeline)
        body = timeline[body_start:body_end]
        # A camera-only sentence must never turn a structurally empty shot into
        # apparently valid visual content. Leave empty bodies untouched so the
        # existing fail-closed visual-timeline validator can reject them.
        if not body.strip():
            continue
        if _minimax_h3_shot_has_camera_spec(body):
            continue
        insertion_position = timeline_start + body_start + len(body.rstrip())
        insertions.append((insertion_position, f" {_MINIMAX_H3_SAFE_CAMERA_HOLD}"))
        repaired_shots.append(int(shot.group(1)))

    repaired = prompt
    for position, insertion in reversed(insertions):
        repaired = repaired[:position] + insertion + repaired[position:]
    return repaired, repaired_shots


_MINIMAX_H3_REF_TASK_TYPES = {
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
}
_MINIMAX_H3_REF_VISUAL_RETENTION = {
    "fully_preserved",
    "partially_preserved",
    "attribute_transfer",
    "weak_reference",
}
_MINIMAX_H3_REF_AUDIO_RETENTION = {
    "fully_copy",
    "partially_copy",
    "reference",
    "weak_reference",
}


def _minimax_h3_ref_detail_word_budget(
    duration_seconds: float,
    shot_count: int,
) -> tuple[int, int, int]:
    """Return a duration/shot-aware (minimum, target low, target high) budget.

    A fixed 120-word floor made a fully specified five-second, one-shot Ref2VA
    prompt fail even after two model repairs. That amount of prose is useful
    for a dense or long timeline, but it encourages action cramming in a short
    continuous shot. Keep the established 120-word safety floor for dense or
    long work while allowing concise short shots that still pass the existing
    subject, style, camera, timing, and final-state checks.
    """

    duration = max(0.0, _safe_float(duration_seconds, 0.0))
    shots = max(1, int(_safe_float(shot_count, 1.0)))
    minimum = min(
        120,
        max(
            60,
            int(math.ceil(duration * 6.0)),
            shots * 24,
        ),
    )
    target_low = min(500, max(minimum, shots * 20 + 20))
    target_high = min(500, max(target_low + 40, shots * 35 + 30))
    return minimum, target_low, target_high


def _minimax_h3_ref2va_validation_reasons(
    prompt_text: str,
    duration_seconds: float,
    user_prompt: str = "",
    audio_mode: str = "auto_scene_audio",
    max_prompt_chars: int = 0,
    reference_manifest: str = "",
    minimax_h3_shot_count: str = "auto",
    minimax_h3_dialogue_mode: str = "auto",
    minimax_h3_dialogue_line_count: int = 2,
    minimax_h3_dialogue_guidance: str = "",
    minimax_h3_expected_subject_count: int = 0,
) -> list[str]:
    prompt = _sanitize_structured_prompt_text(prompt_text, "", 0)
    manifest = _normalize_minimax_h3_reference_manifest(reference_manifest)
    manifest_definitions = _minimax_h3_reference_definitions(manifest)
    manifest_tags = [tag for tag, _description in manifest_definitions]
    reasons = _minimax_h3_reference_manifest_validation_reasons(manifest)
    if re.search(
        r"(?:\[\s*)?dg\s*:[^\]\r\n]*(?:\]|$)",
        prompt,
        flags=re.IGNORECASE | re.MULTILINE,
    ):
        reasons.append("minimax_h3_prompt_contains_grounding_role_annotation")
    if max_prompt_chars > 0 and len(prompt) > max_prompt_chars:
        reasons.append("minimax_h3_prompt_truncated")

    positions: list[int] = []
    for field_name in _MINIMAX_H3_REF_FIELDS:
        matches = list(re.finditer(rf"(?m)^{re.escape(field_name)}\s*:", prompt))
        if not matches:
            reasons.append(f"minimax_h3_ref_missing_{field_name}")
            positions.append(-1)
        else:
            positions.append(matches[0].start())
            if len(matches) != 1:
                reasons.append("minimax_h3_ref_sections_invalid")
    if all(position >= 0 for position in positions) and positions != sorted(positions):
        reasons.append("minimax_h3_ref_field_order_invalid")
    if prompt and not re.match(r"^subject_definitions\s*:", prompt):
        reasons.append("minimax_h3_ref_sections_invalid")
    if any(not re.search(rf"\n\n{re.escape(field_name)}\s*:", prompt) for field_name in _MINIMAX_H3_REF_FIELDS[1:]):
        reasons.append("minimax_h3_ref_field_spacing_invalid")
    if re.search(
        r"(?m)^(?!subject_definitions\s*:|summary\s*:|retention_analysis\s*:|detailed_description\s*:|overall_soundscape\s*:|non_diegetic_music\s*:)[A-Za-z][A-Za-z0-9 _-]{0,48}\s*:",
        prompt,
    ):
        reasons.append("minimax_h3_extra_top_level_field")

    sections = _minimax_h3_ref_sections(prompt)
    if sections is None:
        reasons.append("minimax_h3_ref_sections_invalid")
        return list(dict.fromkeys(reasons))
    for field_name in _MINIMAX_H3_REF_FIELDS:
        if not sections[field_name]:
            reasons.append(f"minimax_h3_ref_missing_{field_name}")

    subject_definitions = sections["subject_definitions"]
    summary = sections["summary"]
    retention = sections["retention_analysis"]
    detailed = sections["detailed_description"]
    soundscape = sections["overall_soundscape"]
    music = sections["non_diegetic_music"]
    if re.fullmatch(r"/\s*A", music, flags=re.IGNORECASE):
        reasons.append("minimax_h3_non_diegetic_music_invalid")

    if len(_MINIMAX_H3_DIALOGUE_TAGLIKE_PATTERN.findall(prompt)) != len(
        _MINIMAX_H3_DIALOGUE_TAGLIKE_PATTERN.findall(detailed)
    ):
        reasons.append("minimax_h3_dialogue_outside_timeline")
    if _normalize_minimax_h3_dialogue_mode(minimax_h3_dialogue_mode) == "off" and (
        _MINIMAX_H3_DIALOGUE_TAGLIKE_PATTERN.search(prompt)
    ):
        reasons.append("minimax_h3_dialogue_forbidden")
    _subject_speakers, _speaker_subjects, speaker_mapping_reasons = (
        _minimax_h3_ref2va_global_speaker_mappings(prompt)
    )
    if speaker_mapping_reasons:
        reasons.append("minimax_h3_dialogue_speaker_invalid")

    prompt_asset_tags = set(_minimax_h3_reference_tags(prompt))
    declared_asset_tags = set(manifest_tags)
    if declared_asset_tags - set(_minimax_h3_reference_tags(subject_definitions)):
        reasons.append("minimax_h3_ref_missing_asset_tag")
    if prompt_asset_tags - declared_asset_tags:
        reasons.append("minimax_h3_ref_undefined_tag")

    definition_items = _minimax_h3_ref_definition_items(subject_definitions)
    defined_item_tags = [tag for tag, _description in definition_items]
    defined_subjects = [tag for tag in defined_item_tags if tag.startswith("<Subject ")]
    definition_map = {tag: description for tag, description in definition_items}
    manifest_role_map = {tag: description for tag, description in manifest_definitions}
    picture_2_role = manifest_role_map.get("<Picture 2>", "")
    multi_entity_contact_sheet = bool(
        re.search(r"\bmulti[- ]entity contact sheet\b", picture_2_role, flags=re.IGNORECASE)
    )
    same_subject_contact_sheet = bool(
        re.search(r"\bsame[- ]subject contact sheet\b", picture_2_role, flags=re.IGNORECASE)
    )
    if multi_entity_contact_sheet:
        primary_definition = definition_map.get("<Subject 1>", "")
        secondary_definitions = [
            description
            for tag, description in definition_items
            if tag.startswith("<Subject ") and tag != "<Subject 1>"
        ]
        if (
            "<Picture 1>" not in primary_definition
            or "<Picture 2>" in primary_definition
            or not secondary_definitions
            or any(
                "<Picture 2>" not in description or "<Picture 1>" in description
                for description in secondary_definitions
            )
        ):
            reasons.append("minimax_h3_ref_contact_sheet_subject_source_invalid")
    elif same_subject_contact_sheet:
        primary_definition = definition_map.get("<Subject 1>", "")
        if not all(tag in primary_definition for tag in ("<Picture 1>", "<Picture 2>")):
            reasons.append("minimax_h3_ref_contact_sheet_subject_source_invalid")
    if (multi_entity_contact_sheet or same_subject_contact_sheet) and re.search(
        r"\b(?:contact[- ]sheet|reference[- ]sheet|collage|thumbnail\s+cells?|"
        r"(?:sheet|image)\s+grid|grid\s+(?:layout|cells?)|panel\s+seams?|sheet\s+borders?)\b",
        detailed,
        flags=re.IGNORECASE,
    ):
        reasons.append("minimax_h3_ref_contact_sheet_layout_transfer_invalid")
    expected_subject_count = _normalize_minimax_h3_expected_subject_count(
        minimax_h3_expected_subject_count
    )
    all_subjects = {
        f"<Subject {int(ordinal)}>"
        for ordinal in re.findall(r"<\s*Subject\s+(\d+)\s*>", prompt, flags=re.IGNORECASE)
    }
    if len(defined_item_tags) != len(set(defined_item_tags)) or any(
        len(re.findall(r"[A-Za-z0-9]{2,}", description)) < 4 for _tag, description in definition_items
    ):
        reasons.append("minimax_h3_ref_subject_definition_invalid")
    if all_subjects - set(defined_subjects):
        reasons.append("minimax_h3_ref_subject_definition_invalid")
    subject_ordinals = sorted(int(re.search(r"\d+", tag).group()) for tag in set(defined_subjects))
    if subject_ordinals and subject_ordinals != list(range(1, max(subject_ordinals) + 1)):
        reasons.append("minimax_h3_ref_subject_definition_invalid")
    if expected_subject_count and len(set(defined_subjects)) != expected_subject_count:
        reasons.append("minimax_h3_ref_subject_count_mismatch")
    if any(tag not in retention or tag not in detailed for tag in defined_subjects):
        reasons.append("minimax_h3_ref_subject_usage_invalid")

    summary_match = re.match(r"^\[([^\]\n]+)\]\s+\S", summary)
    task_types: list[str] = []
    if summary_match:
        task_types = [item.strip() for item in summary_match.group(1).split("+")]
    if (
        not summary_match
        or not task_types
        or len(task_types) != len(set(task_types))
        or any(task_type not in _MINIMAX_H3_REF_TASK_TYPES for task_type in task_types)
    ):
        reasons.append("minimax_h3_ref_summary_invalid")

    for tag in defined_item_tags:
        kind = re.match(r"<(Subject|Picture|Video|Audio)\s", tag).group(1)
        markers = _MINIMAX_H3_REF_AUDIO_RETENTION if kind == "Audio" else _MINIMAX_H3_REF_VISUAL_RETENTION
        marker_pattern = "|".join(re.escape(marker) for marker in sorted(markers))
        if not re.search(
            rf"(?m)^\s*{re.escape(tag)}(?:\s*\([^\n)]*\))?\s*:\s*(?:{marker_pattern})\s+-\s*\S",
            retention,
        ):
            reasons.append("minimax_h3_ref_retention_invalid")
            break

    shot1_match = re.search(r"\[Shot 1\]", detailed)
    style_opening = detailed[: shot1_match.start()].strip() if shot1_match else ""
    if not shot1_match or len(re.findall(r"[A-Za-z0-9]{2,}", style_opening)) < 5:
        reasons.append("minimax_h3_ref_style_opening_missing")
    requested_shot_count = _minimax_h3_effective_shot_count(
        user_prompt,
        minimax_h3_shot_count,
    )
    parsed_shot_count = len(re.findall(r"\[Shot \d+\]", detailed))
    minimum_detail_words, _target_low, _target_high = (
        _minimax_h3_ref_detail_word_budget(
            duration_seconds,
            requested_shot_count or parsed_shot_count or 1,
        )
    )
    if (
        task_types
        and "video editing" not in task_types
        and len(re.findall(r"\b[\w'’-]+\b", detailed)) < minimum_detail_words
    ):
        reasons.append("minimax_h3_ref_detailed_description_short")

    if shot1_match:
        timeline = detailed[shot1_match.end() :].lstrip()
        integrated = f"[Shot 1] {style_opening} {timeline}".strip()
        integrated = re.sub(r"<\s*(?:Subject|Picture|Video|Audio)\s+\d+\s*>", "the referenced element", integrated)
        t2va_prompt = (
            f"integrated_multimodal_description: {integrated}\n\n"
            f"overall_soundscape: {soundscape}\n\n"
            f"non_diegetic_music: {music}"
        )
        t2va_prompt = re.sub(
            r"<\s*(?:Subject|Picture|Video|Audio)\s+\d+\s*>",
            "the referenced element",
            t2va_prompt,
            flags=re.IGNORECASE,
        )
        reasons.extend(
            _minimax_h3_prompt_validation_reasons(
                t2va_prompt,
                duration_seconds,
                user_prompt,
                audio_mode,
                0,
                "t2va",
                "",
                minimax_h3_shot_count,
                minimax_h3_dialogue_mode,
                minimax_h3_dialogue_line_count,
                minimax_h3_dialogue_guidance,
            )
        )

    return list(dict.fromkeys(reasons))


def _minimax_h3_prompt_validation_reasons(
    prompt_text: str,
    duration_seconds: float,
    user_prompt: str = "",
    audio_mode: str = "auto_scene_audio",
    max_prompt_chars: int = 0,
    minimax_h3_mode: str = "t2va",
    reference_manifest: str = "",
    minimax_h3_shot_count: str = "auto",
    minimax_h3_dialogue_mode: str = "auto",
    minimax_h3_dialogue_line_count: int = 2,
    minimax_h3_dialogue_guidance: str = "",
    minimax_h3_expected_subject_count: int = 0,
) -> list[str]:
    if _normalize_minimax_h3_mode(minimax_h3_mode) == "ref2va":
        return _minimax_h3_ref2va_validation_reasons(
            prompt_text,
            duration_seconds,
            user_prompt,
            audio_mode,
            max_prompt_chars,
            reference_manifest,
            minimax_h3_shot_count,
            minimax_h3_dialogue_mode,
            minimax_h3_dialogue_line_count,
            minimax_h3_dialogue_guidance,
            minimax_h3_expected_subject_count,
        )
    prompt = _sanitize_structured_prompt_text(prompt_text, "", 0)
    reasons: list[str] = []
    if max_prompt_chars > 0 and len(prompt) > max_prompt_chars:
        reasons.append("minimax_h3_prompt_truncated")
    if re.search(r"<\s*(?:Subject|Picture|Video|Audio)\s+\d+\s*>", prompt, flags=re.IGNORECASE):
        reasons.append("minimax_h3_t2va_reference_tag")
    fields = (
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
    )
    positions: list[int] = []
    for field_name in fields:
        matches = list(re.finditer(rf"(?m)^{re.escape(field_name)}\s*:", prompt))
        if not matches:
            reasons.append(f"minimax_h3_missing_{field_name}")
            positions.append(-1)
        else:
            positions.append(matches[0].start())
            if len(matches) != 1:
                reasons.append(f"minimax_h3_duplicate_{field_name}")
    if all(position >= 0 for position in positions) and positions != sorted(positions):
        reasons.append("minimax_h3_field_order_invalid")
    if prompt and not re.match(r"^integrated_multimodal_description\s*:\s*\[Shot 1\]", prompt):
        reasons.append("minimax_h3_opening_invalid")
    if not re.search(r"\n\noverall_soundscape\s*:", prompt) or not re.search(r"\n\nnon_diegetic_music\s*:", prompt):
        reasons.append("minimax_h3_field_spacing_invalid")
    soundscape_match = re.search(
        r"(?ms)^overall_soundscape\s*:\s*(.+?)\n\nnon_diegetic_music\s*:",
        prompt,
    )
    music_match = re.search(r"(?ms)^non_diegetic_music\s*:\s*(.+?)\s*$", prompt)
    soundscape = soundscape_match.group(1).strip() if soundscape_match else ""
    music = music_match.group(1).strip() if music_match else ""
    if not soundscape:
        reasons.append("minimax_h3_empty_overall_soundscape")
    if not music:
        reasons.append("minimax_h3_empty_non_diegetic_music")
    elif re.fullmatch(r"/\s*A", music, flags=re.IGNORECASE):
        reasons.append("minimax_h3_non_diegetic_music_invalid")

    integrated = prompt
    soundscape_index = positions[1] if len(positions) > 1 else -1
    if soundscape_index >= 0:
        integrated = prompt[:soundscape_index].rstrip()
    if _minimax_h3_orphaned_sound_events(integrated, soundscape):
        reasons.append("minimax_h3_orphaned_sound_event")
    if re.search(
        r"(?m)^(?!integrated_multimodal_description\s*:|overall_soundscape\s*:|non_diegetic_music\s*:)[A-Za-z][A-Za-z0-9 _-]{0,48}\s*:",
        prompt,
    ):
        reasons.append("minimax_h3_extra_top_level_field")
    if re.search(r"\[\s*\d+(?:\.\d+)?s\s*[-–—]\s*\d+(?:\.\d+)?s\s*\]", integrated, flags=re.IGNORECASE) or re.search(
        r"\[Shot \d+\]\s*\(?\s*\d+(?:\.\d+)?\s*s?\s*[-–—]\s*\d+(?:\.\d+)?\s*s\b",
        integrated,
        flags=re.IGNORECASE,
    ):
        reasons.append("minimax_h3_noncanonical_time_range")

    shot_pattern = re.compile(r"\[Shot (\d+)\](?:\s+At\s+(\d{2,}):(\d{2})\.(\d{3}),\s*)?")
    shot_markers = list(re.finditer(r"\[\s*shot\b[^\]\n]*\]", integrated, flags=re.IGNORECASE))
    shots = list(shot_pattern.finditer(integrated))
    if len(shot_markers) != len(shots) or any(marker.start() != shot.start() for marker, shot in zip(shot_markers, shots)):
        reasons.append("minimax_h3_shot_syntax_invalid")
    timed_cut_markers = re.findall(
        r"\bAt\s+\d{2,}:\d{2}\.\d{3},?\s+the\s+(?:camera|shot|scene)\s+"
        r"(?:(?:hard\s+)?cuts|transitions|changes|switches|cross-dissolves|fades|wipes)\s+(?:to|into)\b",
        integrated,
        flags=re.IGNORECASE,
    )
    if len(timed_cut_markers) > max(0, len(shots) - 1):
        reasons.append("minimax_h3_unlabeled_cut")
    shot_numbers = [int(match.group(1)) for match in shots]
    if not shot_numbers or shot_numbers != list(range(1, len(shot_numbers) + 1)):
        reasons.append("minimax_h3_shot_numbering_invalid")
    requested_shot_count = _minimax_h3_effective_shot_count(user_prompt, minimax_h3_shot_count)
    if requested_shot_count and len(shots) != requested_shot_count:
        reasons.append("minimax_h3_requested_shot_count_mismatch")
    shot_bodies = [
        integrated[match.end() : shots[index + 1].start() if index + 1 < len(shots) else len(integrated)].strip()
        for index, match in enumerate(shots)
    ]
    if shots:
        shot1_body = shot_bodies[0]
        if shots[0].group(2) is not None or re.match(r"^At\b", shot1_body, flags=re.IGNORECASE):
            reasons.append("minimax_h3_shot1_timestamp_invalid")
        if not shot1_body or not re.search(r"[A-Za-z0-9]", shot1_body):
            reasons.append("minimax_h3_empty_visual_timeline")

        requested_medium = _minimax_h3_requested_visual_medium(user_prompt)
        opening = shot1_body[:320]
        requested_style_locks = _minimax_h3_requested_style_locks(user_prompt)
        if requested_style_locks and not _minimax_h3_style_locks_present(shot1_body[:600], requested_style_locks):
            reasons.append("minimax_h3_requested_style_missing")
        medium_present = True
        if requested_medium == "2d_animation":
            medium_present = bool(
                re.search(r"\b2[ -]?d\b", opening, flags=re.IGNORECASE)
                and re.search(r"\b(?:hand[- ]drawn|cel[- ]animation|cartoon|anime)\b", opening, flags=re.IGNORECASE)
            )
        elif requested_medium == "3d_animation":
            medium_present = bool(re.search(_MINIMAX_H3_3D_ANIMATION_PATTERN, opening, flags=re.IGNORECASE))
        elif requested_medium == "stop_motion":
            medium_present = bool(re.search(_MINIMAX_H3_STOP_MOTION_PATTERN, opening, flags=re.IGNORECASE))
        elif requested_medium == "live_action":
            medium_present = bool(re.search(_MINIMAX_H3_LIVE_ACTION_PATTERN, opening, flags=re.IGNORECASE))
        elif requested_medium == "animation":
            medium_present = bool(
                re.search(
                    rf"(?:{_MINIMAX_H3_ANIMATION_PATTERN}|{_MINIMAX_H3_2D_ANIMATION_PATTERN}|{_MINIMAX_H3_3D_ANIMATION_PATTERN}|{_MINIMAX_H3_STOP_MOTION_PATTERN})",
                    opening,
                    flags=re.IGNORECASE,
                )
            )
        if not medium_present:
            reasons.append("minimax_h3_requested_visual_medium_missing")
        if requested_medium in {"2d_animation", "animation"}:
            live_action_excluded = _has_negated_prompt_phrase(integrated, _MINIMAX_H3_LIVE_ACTION_PATTERN)
            three_d_excluded = _has_negated_prompt_phrase(integrated, _MINIMAX_H3_3D_ANIMATION_PATTERN)
            if not live_action_excluded or (requested_medium == "2d_animation" and not three_d_excluded):
                reasons.append("minimax_h3_requested_visual_exclusion_missing")
            if _has_unnegated_prompt_phrase(integrated, _MINIMAX_H3_LIVE_ACTION_PATTERN) or (
                requested_medium == "2d_animation"
                and _has_unnegated_prompt_phrase(integrated, _MINIMAX_H3_3D_ANIMATION_PATTERN)
            ):
                reasons.append("minimax_h3_requested_visual_medium_conflict")

        if any(not _minimax_h3_shot_has_camera_spec(body) for body in shot_bodies):
            reasons.append("minimax_h3_shot_camera_unspecified")
        final_hold_requested = _has_unnegated_prompt_phrase(
            user_prompt,
            r"\b(?:freeze(?:[- ]frame)?|final\s+(?:frame|shot)\s+holds?|hold(?:ing|s)?\s+(?:the\s+)?(?:final|last)\s+(?:frame|shot))\b",
        )
        if final_hold_requested and not re.search(
            r"\b(?:freeze(?:s|[- ]frame)?|static\s+hold|held\s+frame|(?:final|last)\s+frame\s+holds?)\b",
            shot_bodies[-1],
            flags=re.IGNORECASE,
        ):
            reasons.append("minimax_h3_requested_final_hold_missing")
        final_frame_match = re.search(r"\bThe final frame holds on\b(.+)$", shot_bodies[-1], flags=re.IGNORECASE | re.DOTALL)
        if final_frame_match and not final_hold_requested and re.search(
            r"\b(?:mid[- ]air|mid[- ]clash|mid[- ]fight|mid[- ]attack|airborne|about to|prepares? to|begins? to|starts? to|charges? (?:an? |the )?attack|"
            r"runs? toward|sprints? toward|disappears?|vanishes?)\b",
            final_frame_match.group(1),
            flags=re.IGNORECASE,
        ):
            reasons.append("minimax_h3_unresolved_final_state")
        if _minimax_h3_battle_intent_requested(user_prompt) and not re.search(
            r"\b(?:knock(?:s|ed|ing)?(?:\s+back(?:ward)?)?|throw(?:s|n|ing)?(?:\s+back(?:ward)?)?|recoil(?:s|ed|ing)?|"
            r"stagger(?:s|ed|ing)?|stumbl(?:es|ed|ing)|skids?|skidd(?:ing|ed)|slides? back|slid back|falls?|fell|collapses?|"
            r"blocks?|blocked|dodges?|dodged|evades?|evaded|deflects?|deflected|absorbs?|absorbed|counters?|countered|"
            r"driven back|forced back|sent (?:backward|sprawling|to the ground)|pushed back|regains? (?:its|his|her|their) footing)\b",
            " ".join(shot_bodies[1:]),
            flags=re.IGNORECASE,
        ):
            reasons.append("minimax_h3_battle_result_missing")

    previous_cut = 0.0
    transition_effect_used = False
    for match in shots[1:]:
        if match.group(2) is None:
            reasons.append("minimax_h3_cut_timestamp_invalid")
            continue
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        milliseconds = int(match.group(4))
        if seconds >= 60:
            reasons.append("minimax_h3_cut_timestamp_invalid")
            continue
        cut_time = minutes * 60.0 + seconds + milliseconds / 1000.0
        if cut_time <= previous_cut:
            reasons.append("minimax_h3_cut_timestamp_invalid")
        if duration_seconds > 0 and cut_time >= duration_seconds:
            reasons.append("minimax_h3_cut_timestamp_out_of_range")
        transition_text = integrated[match.end() :].lstrip()
        if re.match(
            r"the (?:camera|shot|scene) (?:cross-dissolves|fades|wipes) (?:to|into)\b",
            transition_text,
            flags=re.IGNORECASE,
        ):
            transition_effect_used = True
        if not re.match(
            r"the (?:camera|shot|scene) (?:(?:hard )?cuts|transitions|changes|switches|cross-dissolves|fades|wipes) (?:to|into)\b",
            transition_text,
            flags=re.IGNORECASE,
        ):
            reasons.append("minimax_h3_cut_transition_invalid")
        previous_cut = cut_time

    if transition_effect_used and not re.search(
        r"\b(?:cross[- ]dissolves?|fades?|wipes?)\b",
        user_prompt,
        flags=re.IGNORECASE,
    ):
        reasons.append("minimax_h3_unrequested_transition_effect")

    dialogue_mode = _normalize_minimax_h3_dialogue_mode(minimax_h3_dialogue_mode)
    dialogue_line_count = _normalize_minimax_h3_dialogue_line_count(minimax_h3_dialogue_line_count)
    dialogue_block_count = _minimax_h3_dialogue_block_count(integrated)
    prompt_taglike = _MINIMAX_H3_DIALOGUE_TAGLIKE_PATTERN.findall(prompt)
    integrated_taglike = _MINIMAX_H3_DIALOGUE_TAGLIKE_PATTERN.findall(integrated)
    dialogue_structure_reasons: list[str] = []
    if len(prompt_taglike) != len(integrated_taglike):
        dialogue_structure_reasons.append("minimax_h3_dialogue_outside_timeline")
    canonical_tag_count = integrated.count("<d>") + integrated.count("</d>")
    if (
        len(integrated_taglike) != canonical_tag_count
        or integrated.count("<d>") != integrated.count("</d>")
        or (
            "<d>" in integrated
            and len(
                re.findall(
                    r"<d>\[[^\]\n]+\]\s*\S(?:(?!</?d>|\[Shot \d+\]).)*?</d>",
                    integrated,
                    flags=re.DOTALL,
                )
            )
            != integrated.count("<d>")
        )
    ):
        dialogue_structure_reasons.append("minimax_h3_dialogue_tags_invalid")
    if "minimax_h3_dialogue_tags_invalid" not in dialogue_structure_reasons:
        shot_starts = list(re.finditer(r"\[Shot \d+\]", integrated))
        for dialogue in re.finditer(r"<d>", integrated):
            shot_start = max(
                (match.end() for match in shot_starts if match.start() < dialogue.start()),
                default=0,
            )
            preceding = integrated[max(shot_start, dialogue.start() - 500) : dialogue.start()]
            if not re.search(
                r"\(S[1-9]\d*(?:\s*,\s*S[1-9]\d*)*\)[^<]{0,500}$",
                preceding,
                flags=re.IGNORECASE,
            ):
                dialogue_structure_reasons.append("minimax_h3_dialogue_speaker_invalid")
                break
    reasons.extend(dialogue_structure_reasons)

    if (
        dialogue_mode == "required"
        and not dialogue_structure_reasons
        and dialogue_block_count != dialogue_line_count
    ):
        reasons.append("minimax_h3_dialogue_count_mismatch")
    elif dialogue_mode == "off" and bool(prompt_taglike):
        reasons.append("minimax_h3_dialogue_forbidden")
    if dialogue_mode == "required" and _normalize_audio_mode(audio_mode) == "visual_only":
        reasons.append("minimax_h3_dialogue_audio_mode_conflict")

    if _normalize_audio_mode(audio_mode) == "visual_only":
        if soundscape.casefold() != "n/a" or music.casefold() != "n/a":
            reasons.append("minimax_h3_visual_only_audio_fields_invalid")
        if re.search(
            r"<d>|</d>|\b(?:we hear|audible|audio|soundscape|foley|ambience|music|score|soundtrack|voiceover|says?|asks?|replies?|shouts?|whispers?|sings?)\b",
            integrated,
            flags=re.IGNORECASE,
        ):
            reasons.append("minimax_h3_visual_only_audio_cue")

    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped


def _reason_strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    reasons: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        reason = item.strip()
        if reason and reason not in reasons:
            reasons.append(reason)
    return reasons


def _strict_grounding_guard_block(
    metadata: Any,
) -> tuple[bool, list[str], dict[str, Any]]:
    if not isinstance(metadata, dict):
        return False, [], {}
    report = (
        metadata.get("grounding_guard")
        if isinstance(metadata.get("grounding_guard"), dict)
        else {}
    )
    decision = str(report.get("decision", "")).strip().lower()
    config = report.get("config") if isinstance(report.get("config"), dict) else {}
    strict_would_block = bool(
        report.get("schema") == GROUNDING_REPORT_SCHEMA_ID
        and str(config.get("mode", "")).strip().lower() == "strict"
        and (
            report.get("would_block") is True
            or report.get("grounding_guard_would_block") is True
        )
    )
    if decision != "block" and not strict_would_block:
        return False, [], report
    reasons = _reason_strings(report.get("blocked_reasons"))
    return True, reasons or ["visual_grounding_unverified"], report


def _packet_generation_readiness(
    packet: dict[str, Any],
    context: GemmaContext,
    target_config: TargetProfileConfig | dict[str, Any],
    prompt_text: str,
    media_metadata: dict[str, Any],
    max_prompt_chars: int = 8000,
) -> tuple[bool, list[str]]:
    metadata = packet.get("metadata") if isinstance(packet.get("metadata"), dict) else {}
    guard_blocked, guard_reasons, _report = _strict_grounding_guard_block(metadata)
    if guard_blocked:
        return False, guard_reasons
    reasons: list[str] = []
    json_parse_known = "json_parse_valid" in metadata
    if json_parse_known and not _truthy_metadata(metadata.get("json_parse_valid")):
        reasons.append("json_parse_invalid")
    if _truthy_metadata(metadata.get("plain_text_salvage")) or metadata.get("salvage_warning") or metadata.get("json_parse_warning"):
        reasons.append("salvaged_output")
    if _truthy_metadata(metadata.get("used_template_fallback")) or metadata.get("fallback_reason"):
        reasons.append("template_fallback")
    if _truthy_metadata(metadata.get("template_used_for_missing_fields")):
        reasons.append("template_filled_missing_fields")
    if metadata.get("visual_grounding_warning"):
        reasons.append("metadata_only_visual_grounding_warning")
    if _is_image_identity_video_control(media_metadata):
        if _safe_float(media_metadata.get("reference_image_count"), 0.0) <= 0:
            reasons.append("synthesis_missing_identity_image")
        elif media_metadata.get("reference_image_backend_attached") is False:
            reasons.append("synthesis_identity_image_not_attached")
        if _safe_float(media_metadata.get("video_sampled_frame_count"), 0.0) <= 0:
            reasons.append("synthesis_missing_control_video")

    target = _target_profile_config_to_dict(target_config)
    if target["target_profile"] == "minimax_h3":
        raw_h3_prompt = str(packet.get("minimax_h3_prompt", "") or "")
        h3_mode = target["minimax_h3_mode"]
        reference_manifest = _normalize_minimax_h3_reference_manifest(
            str(media_metadata.get("minimax_h3_reference_manifest", ""))
        )
        reasons.extend(
            _minimax_h3_prompt_validation_reasons(
                raw_h3_prompt,
                _minimax_h3_duration_seconds(media_metadata, target["target_duration_seconds"], context.user_prompt),
                context.user_prompt,
                target["audio_mode"],
                int(max_prompt_chars),
                h3_mode,
                reference_manifest,
                target["minimax_h3_shot_count"],
                target["minimax_h3_dialogue_mode"],
                target["minimax_h3_dialogue_line_count"],
                target["minimax_h3_dialogue_guidance"],
                _normalize_minimax_h3_expected_subject_count(
                    media_metadata.get("minimax_h3_expected_subject_count", 0)
                ),
            )
        )
    elif target["target_profile"] == "ltx":
        ltx25_diagnostics = _ltx25_contract_diagnostics(
            prompt_text,
            context,
            target,
            media_metadata,
        )
        metadata["ltx25_contract"] = ltx25_diagnostics
        metadata["ltx_generation_mode_requested"] = ltx25_diagnostics["requested_mode"]
        metadata["ltx_generation_mode_effective"] = ltx25_diagnostics["effective_mode"]
        metadata["ltx_generation_mode_inference"] = ltx25_diagnostics["mode_inference"]
        reasons.extend(ltx25_diagnostics.get("reasons", []))
        reasons.extend(
            _ltx_exact_voiceover_validation_reasons(
                prompt_text,
                target["audio_mode"],
                target["audio_guidance"],
                context.user_prompt,
            )
        )

    claims_ok, claim_reasons = _verified_prompt_claims(
        prompt_text=prompt_text,
        context=context,
        media_metadata=media_metadata,
    )
    if not claims_ok:
        reasons.extend(claim_reasons)

    deduped: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        clean = str(reason or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return not deduped, deduped


def _split_segments(prompt: str, duration_seconds: float, max_segments: int = 6) -> list[dict[str, Any]]:
    protected = re.sub(r"\b(EXT|INT)\.", r"\1<DOT>", prompt or "", flags=re.IGNORECASE)
    sentences = [s.strip().replace("<DOT>", ".") for s in re.split(r"(?<=[.!?])\s+", protected) if s.strip()]
    if len(sentences) > 1 and sentences[0].lower().startswith("style:"):
        sentences = [f"{sentences[0]} {sentences[1]}".strip(), *sentences[2:]]
    if not sentences:
        sentences = [prompt.strip()] if prompt.strip() else ["A clear scene with a visible subject and motion."]
    max_segments = max(1, min(max_segments, len(sentences)))
    chunks: list[str] = []
    for idx in range(max_segments):
        chunks.append(sentences[idx])
    if len(sentences) > max_segments:
        chunks[-1] = " ".join([chunks[-1], *sentences[max_segments:]]).strip()
    segment_duration = duration_seconds / len(chunks) if duration_seconds > 0 else 0.0
    return [
        {
            "index": idx,
            "duration_seconds": round(segment_duration, 3) if segment_duration else 0.0,
            "prompt": chunk,
        }
        for idx, chunk in enumerate(chunks)
    ]


def _segments_to_director_strings(segments: list[dict[str, Any]]) -> tuple[str, str]:
    prompts = [str(seg.get("prompt", "")).strip() for seg in segments if str(seg.get("prompt", "")).strip()]
    lengths = []
    for seg in segments[: len(prompts)]:
        duration = _safe_float(seg.get("duration_seconds"), 0.0)
        lengths.append(str(max(1, int(round(duration)))) if duration > 0 else "1")
    return " | ".join(prompts), ", ".join(lengths)


def _load_scene_segments(scene_segments_json: str) -> list[dict[str, Any]] | None:
    parsed = _extract_json_object(scene_segments_json)
    if parsed and isinstance(parsed.get("scene_segments"), list):
        return parsed["scene_segments"]
    if scene_segments_json.strip().startswith("["):
        try:
            loaded = json.loads(scene_segments_json)
            if isinstance(loaded, list):
                return loaded
        except Exception:
            return None
    return None


def _normalize_segments(
    segments: list[dict[str, Any]] | None,
    fallback_prompt: str,
    duration_seconds: float,
    max_segments: int,
) -> list[dict[str, Any]]:
    limit = max(1, min(int(max_segments), 24))
    cleaned: list[dict[str, Any]] = []
    for segment in segments or []:
        if isinstance(segment, dict):
            prompt = _sanitize_prompt_text(str(segment.get("prompt", "")), "", 3000)
            duration = _safe_float(segment.get("duration_seconds"), 0.0)
        else:
            prompt = _sanitize_prompt_text(str(segment), "", 3000)
            duration = 0.0
        if prompt:
            cleaned.append({"duration_seconds": max(0.0, duration), "prompt": prompt})

    if not cleaned:
        return _split_segments(fallback_prompt, duration_seconds, limit)

    if len(cleaned) > limit:
        head = cleaned[: limit - 1]
        tail = cleaned[limit - 1 :]
        merged_prompt = _single_paragraph(" ".join(str(item.get("prompt", "")) for item in tail))
        merged_duration = sum(_safe_float(item.get("duration_seconds"), 0.0) for item in tail)
        cleaned = [*head, {"duration_seconds": merged_duration, "prompt": merged_prompt}]

    if duration_seconds > 0:
        segment_duration = duration_seconds / len(cleaned)
        for item in cleaned:
            item["duration_seconds"] = round(segment_duration, 3)

    return [
        {
            "index": idx,
            "duration_seconds": round(_safe_float(item.get("duration_seconds"), 0.0), 3),
            "prompt": _single_paragraph(str(item.get("prompt", ""))),
        }
        for idx, item in enumerate(cleaned)
    ]


def _normalize_audio_mode(audio_mode: str) -> str:
    value = (audio_mode or "").strip().lower()
    if value in {"auto_scene_audio", "explicit_sound_design", "visual_only"}:
        return value
    return "auto_scene_audio"


def _audio_guidance_text(audio_guidance: str) -> str:
    return _sanitize_prompt_text(audio_guidance, "", 700, strip_thinking=True, strip_markdown=True).rstrip(".")


_LTX_QUOTED_TEXT_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ('"', '"', re.compile(r'"([^"\r\n]{1,500})"')),
    ("“", "”", re.compile(r"“([^”\r\n]{1,500})”")),
    (
        "'",
        "'",
        re.compile(r"(?<!\w)'((?:[^'\r\n]|(?<=\w)'(?=\w)){1,500})'(?!\w)"),
    ),
    (
        "‘",
        "’",
        re.compile(r"(?<!\w)‘((?:[^’\r\n]|(?<=\w)’(?=\w)){1,500})’(?!\w)"),
    ),
)
_LTX_SPEECH_CUE_PATTERN = re.compile(
    r"\b(?:voice[- ]?over|narrat(?:or|ion)|off[- ]screen\s+voice)\b",
    flags=re.IGNORECASE,
)
_LTX_EXACT_VOICEOVER_PATTERN = re.compile(
    r"\b(?:exact(?:ly)?(?:\s+words?)?|verbatim)\b",
    flags=re.IGNORECASE,
)
_LTX_SINGLE_VOICEOVER_PATTERN = re.compile(
    r"\b(?:one|single|sole)\b|\bno\s+other\s+(?:speech|dialogue|narration|voiceover|words?)\b",
    flags=re.IGNORECASE,
)
_LTX_SPEECH_DELIVERY_PATTERN = re.compile(
    r"\b(?:say(?:s|ing)?|speak(?:s|ing)?|whisper(?:s|ing)?|intone(?:s|d|ing)?|"
    r"narrate(?:s|d|ing)?|utter(?:s|ed|ing)?|murmur(?:s|ed|ing)?|declare(?:s|d|ing)?|"
    r"ask(?:s|ed|ing)?|repl(?:y|ies|ied|ying)|announce(?:s|d|ing)?|recite(?:s|d|ing)?|"
    r"deliver(?:s|ed|ing)?)\b",
    flags=re.IGNORECASE,
)
_LTX_VISIBLE_TEXT_QUOTE_PATTERN = re.compile(
    r"\b(?:sign|title\s+card|caption|subtitle|screen|written\s+text|lettering|poster|billboard)\b"
    r"[^.!?\"'”’]{0,80}\b(?:read(?:s|ing)?|say(?:s|ing)?|display(?:s|ed|ing)?|show(?:s|ed|ing)?)\b",
    flags=re.IGNORECASE,
)


def _ltx_quoted_text_spans(text: str) -> list[dict[str, Any]]:
    value = str(text or "")
    spans: list[dict[str, Any]] = []
    for opening, closing, pattern in _LTX_QUOTED_TEXT_PATTERNS:
        for match in pattern.finditer(value):
            spans.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "opening": opening,
                    "closing": closing,
                    "text": match.group(1),
                }
            )
    return sorted(spans, key=lambda item: (int(item["start"]), int(item["end"])))


def _ltx_spoken_text_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _ltx_exact_single_voiceover_request(
    audio_guidance: str,
    request_text: str = "",
) -> dict[str, Any] | None:
    guidance = _audio_guidance_text(audio_guidance)
    if guidance and _LTX_SPEECH_CUE_PATTERN.search(guidance):
        spans = _ltx_quoted_text_spans(guidance)
        if len(spans) == 1:
            span = spans[0]
            exact_context = guidance[max(0, int(span["start"]) - 220) : int(span["start"])]
            if (
                _LTX_EXACT_VOICEOVER_PATTERN.search(exact_context)
                and _LTX_SINGLE_VOICEOVER_PATTERN.search(guidance)
            ):
                spoken_text = re.sub(r"\s+", " ", str(span["text"])).strip()
                if spoken_text and len(spoken_text) <= 240 and len(spoken_text.split()) <= 40:
                    return {
                        "guidance": guidance,
                        "source": "target.audio_guidance",
                        "spoken_text": spoken_text,
                        "spoken_text_key": _ltx_spoken_text_key(spoken_text),
                    }

    # A quoted off-screen line in the user's own request is authoritative even
    # when the optional Audio guidance widget is blank. This keeps safety from
    # depending on one particular workflow wiring choice.
    request = _sanitize_prompt_text(request_text, "", 0)
    request_spans = [
        span
        for span in _ltx_quoted_text_spans(request)
        if _ltx_quote_has_speech_cue(request, span)
    ]
    if len(request_spans) != 1:
        return None
    span = request_spans[0]
    spoken_text = re.sub(r"\s+", " ", str(span["text"])).strip()
    if not spoken_text or len(spoken_text) > 240 or len(spoken_text.split()) > 40:
        return None
    return {
        "guidance": request,
        "source": "context.user_prompt",
        "spoken_text": spoken_text,
        "spoken_text_key": _ltx_spoken_text_key(spoken_text),
    }


def _ltx_quote_has_any_speech_cue(prompt: str, span: dict[str, Any]) -> bool:
    start = max(0, int(span["start"]) - 240)
    preceding = prompt[start : int(span["start"])]
    sentence_parts = re.split(r"[.!?][\"'”’]?\s+", preceding)
    attribution = sentence_parts[-1] if sentence_parts else preceding
    if _LTX_VISIBLE_TEXT_QUOTE_PATTERN.search(attribution):
        return False
    return bool(_LTX_SPEECH_DELIVERY_PATTERN.search(attribution))


def _ltx_quote_has_speech_cue(prompt: str, span: dict[str, Any]) -> bool:
    start = max(0, int(span["start"]) - 240)
    preceding = prompt[start : int(span["start"])]
    sentence_parts = re.split(r"[.!?][\"'”’]?\s+", preceding)
    attribution = sentence_parts[-1] if sentence_parts else preceding
    cue_matches = list(_LTX_SPEECH_CUE_PATTERN.finditer(attribution))
    if not cue_matches:
        return False
    governed_text = attribution[cue_matches[-1].start() :]
    if _LTX_VISIBLE_TEXT_QUOTE_PATTERN.search(governed_text):
        return False
    return bool(_ltx_quote_has_any_speech_cue(prompt, span) or governed_text.rstrip().endswith(":"))


def _ltx_voiceover_clause_start(prompt: str, quote_start: int) -> int | None:
    before_quote = prompt[: max(0, int(quote_start))]
    sentence_start = 0
    for match in re.finditer(r"[.!?][\"'”’]?\s+", before_quote):
        sentence_start = match.end()
    attribution = prompt[sentence_start:quote_start]
    cue_matches = list(_LTX_SPEECH_CUE_PATTERN.finditer(attribution))
    if not cue_matches:
        return None
    cue_position = cue_matches[-1].start()
    clause_start = 0
    for match in re.finditer(
        r"(?:,\s*)?\b(?:while|then|and|as)\s+(?=(?:an?|the)\b)",
        attribution,
        flags=re.IGNORECASE,
    ):
        if match.end() <= cue_position:
            clause_start = match.start()
    if clause_start == 0:
        for match in re.finditer(
            r"[,;]\s+(?=(?:an?|the)\s+[^,;.!?]{0,160}\b(?:voice[- ]?over|narrator)\b)",
            attribution,
            flags=re.IGNORECASE,
        ):
            if match.end() <= cue_position:
                clause_start = match.start()
    return sentence_start + clause_start


def _ltx_terminal_voiceover_cue(prompt: str, clause_start: int, quote_start: int) -> str:
    cue = prompt[clause_start:quote_start].strip()
    cue = re.sub(r"^[,;:\s]+", "", cue)
    cue = re.sub(r"^(?:while|then|and|as)\s+", "", cue, flags=re.IGNORECASE)
    cue = cue.rstrip(" ,;:")
    if (
        not cue
        or len(cue) > 220
        or not _LTX_SPEECH_CUE_PATTERN.search(cue)
    ):
        cue = "an off-screen narrator speaks"
    elif cue[:1].isupper():
        cue = cue[:1].lower() + cue[1:]
    if not re.search(r"\b(?:exactly once|one time)\b", cue, flags=re.IGNORECASE):
        cue += " exactly once"
    return f"Over the final visual beat, {cue}"


def _repair_ltx_exact_voiceover_boundary(
    prompt: str,
    audio_guidance: str,
    max_chars: int = 8000,
    audio_mode: str = "explicit_sound_design",
    request_text: str = "",
) -> tuple[str, dict[str, Any]]:
    value = _sanitize_prompt_text(prompt, "", 0)
    request = _ltx_exact_single_voiceover_request(audio_guidance, request_text)
    report: dict[str, Any] = {
        "eligible": bool(request),
        "applied": False,
        "strategy": "terminal_exact_offscreen_voiceover",
        "source": str(request.get("source", "none")) if request else "none",
    }
    if _normalize_audio_mode(audio_mode) == "visual_only":
        report["eligible"] = False
        report["status"] = "disabled_for_visual_only"
        return value, report
    if request is None:
        report["status"] = "not_applicable"
        return value, report

    spoken_text = str(request["spoken_text"])
    spoken_key = str(request["spoken_text_key"])
    report["spoken_text_sha256"] = hashlib.sha256(spoken_text.encode("utf-8")).hexdigest()[:16]
    spans = _ltx_quoted_text_spans(value)
    all_speech_spans = [span for span in spans if _ltx_quote_has_any_speech_cue(value, span)]
    cue_spans = [span for span in spans if _ltx_quote_has_speech_cue(value, span)]
    matching = [span for span in cue_spans if _ltx_spoken_text_key(span["text"]) == spoken_key]
    report["all_speech_quote_count"] = len(all_speech_spans)
    report["speech_cue_quote_count"] = len(cue_spans)
    report["matching_quote_count"] = len(matching)
    if len(all_speech_spans) != 1:
        report["status"] = "additional_or_ambiguous_speech"
        return value, report
    if len(cue_spans) != 1 or len(matching) != 1:
        report["status"] = "ambiguous_or_missing_model_voiceover"
        return value, report

    span = matching[0]
    expected_quote = f'"{spoken_text}"'
    if (
        int(span["end"]) == len(value)
        and str(span["opening"]) == '"'
        and str(span["closing"]) == '"'
        and value.endswith(expected_quote)
    ):
        report["status"] = "already_terminal"
        return value, report

    clause_start = _ltx_voiceover_clause_start(value, int(span["start"]))
    if clause_start is None:
        report["status"] = "voiceover_clause_not_found"
        return value, report
    voiceover_sentence_start = 0
    for sentence_boundary in re.finditer(
        r"[.!?][\"'”’]?\s+",
        value[: int(span["start"])],
    ):
        voiceover_sentence_start = sentence_boundary.end()
    if re.search(
        r"\b(?:at\s+the\s+(?:beginning|start)|at\s+first|initially|the\s+scene\s+opens|"
        r"before\s+(?:the|any)|early\s+in)\b",
        value[voiceover_sentence_start : int(span["start"])],
        flags=re.IGNORECASE,
    ):
        report["status"] = "explicit_nonterminal_timing"
        return value, report
    terminal_cue = _ltx_terminal_voiceover_cue(value, clause_start, int(span["start"]))

    before = value[:clause_start].rstrip(" ,;:")
    after = value[int(span["end"]) :].lstrip()
    if after.startswith(".") and spoken_text.endswith((".", "!", "?")):
        after = after[1:].lstrip()
    after = re.sub(r"^[,;:]\s*", "", after)
    after = re.sub(r"^(?:and|then)\s+", "", after, flags=re.IGNORECASE)
    after = re.sub(r"^while\s+", "Meanwhile, ", after, flags=re.IGNORECASE)
    if after and after[:1].islower():
        after = after[:1].upper() + after[1:]
    if before and not re.search(r"[.!?][\"'”’]?$", before):
        before += "."
    visual_prompt = _single_paragraph(" ".join(part for part in (before, after) if part))
    repaired = _single_paragraph(f"{visual_prompt} {terminal_cue}: {expected_quote}")
    if max_chars > 0 and len(repaired) > int(max_chars):
        report["status"] = "repair_would_exceed_prompt_limit"
        return value, report
    repaired_spans = _ltx_quoted_text_spans(repaired)
    repaired_matching_cues = [
        candidate
        for candidate in repaired_spans
        if _ltx_quote_has_speech_cue(repaired, candidate)
        and _ltx_spoken_text_key(candidate["text"]) == spoken_key
    ]
    if (
        not repaired.endswith(expected_quote)
        or len(repaired_matching_cues) != 1
        or int(repaired_matching_cues[0]["end"]) != len(repaired)
    ):
        report["status"] = "repair_postcondition_failed"
        return value, report
    report["applied"] = True
    report["status"] = "repaired"
    return repaired, report


def _ltx_exact_voiceover_validation_reasons(
    prompt: str,
    audio_mode: str,
    audio_guidance: str,
    request_text: str = "",
) -> list[str]:
    value = _sanitize_prompt_text(prompt, "", 0)
    spans = _ltx_quoted_text_spans(value)
    all_speech_spans = [span for span in spans if _ltx_quote_has_any_speech_cue(value, span)]
    if _normalize_audio_mode(audio_mode) == "visual_only":
        return ["ltx_visual_only_contains_speech"] if all_speech_spans else []
    request = _ltx_exact_single_voiceover_request(audio_guidance, request_text)
    if request is None:
        return []
    spoken_text = str(request["spoken_text"])
    spoken_key = str(request["spoken_text_key"])
    cue_spans = [span for span in spans if _ltx_quote_has_speech_cue(value, span)]
    matching = [span for span in cue_spans if _ltx_spoken_text_key(span["text"]) == spoken_key]
    reasons: list[str] = []
    if len(all_speech_spans) != 1:
        reasons.append("ltx_exact_voiceover_additional_or_ambiguous_speech")
    if len(cue_spans) != 1 or len(matching) != 1:
        reasons.append("ltx_exact_voiceover_missing_or_ambiguous")
        return reasons
    span = matching[0]
    if str(span["opening"]) != '"' or str(span["closing"]) != '"':
        reasons.append("ltx_exact_voiceover_quote_noncanonical")
    if not spoken_text.endswith((".", "!", "?")):
        reasons.append("ltx_exact_voiceover_terminal_punctuation_missing")
    if int(span["end"]) != len(value) or not value.endswith(f'"{spoken_text}"'):
        reasons.append("ltx_exact_voiceover_not_terminal")
    return reasons


def _ltx25_contract_diagnostics(
    prompt: str,
    context: GemmaContext,
    target: TargetProfileConfig | dict[str, Any],
    media_metadata: dict[str, Any],
) -> dict[str, Any]:
    target_dict = _target_profile_config_to_dict(target)
    mode_report = _ltx25_mode_report(target_dict["ltx_generation_mode"], media_metadata)
    planning_duration = _ltx25_planning_duration_seconds(
        media_metadata,
        target_dict["target_duration_seconds"],
        context.user_prompt,
    )
    validation = validate_ltx25_prompt(
        prompt,
        mode_report,
        media_metadata,
        planning_duration,
        target_dict["ltx_long_horizon_mode"],
        target_dict["ltx_camera_capability"],
    )
    validation["requested_mode"] = mode_report["configured_mode"]
    validation["effective_mode"] = mode_report["resolved_mode"]
    validation["mode_inference"] = mode_report["resolution_source"]
    validation["planning_duration_seconds"] = round(planning_duration, 3)
    long_horizon = validation.get("complexity", {}).get("long_horizon", {})
    if isinstance(long_horizon, dict):
        audio_required = bool(
            long_horizon.get("active")
            and target_dict["audio_mode"] != "visual_only"
        )
        audio_present = _contains_audio_terms(prompt)
        audio_continuity_present = bool(
            audio_present
            and re.search(
                r"\b(?:continues?|continuous(?:ly)?|throughout|through most|steady|seamless(?:ly)?|"
                r"persists?|carries? through|remains?)\b",
                prompt or "",
                flags=re.IGNORECASE,
            )
        )
        long_horizon["audio"] = {
            "required": audio_required,
            "audio_cue_present": audio_present,
            "continuity_cue_present": audio_continuity_present,
        }
        if audio_required and not audio_continuity_present:
            warning = "ltx_long_horizon_continuous_audio_cue_missing"
            long_horizon.setdefault("warnings", [])
            if warning not in long_horizon["warnings"]:
                long_horizon["warnings"].append(warning)
            validation.setdefault("warnings", [])
            if warning not in validation["warnings"]:
                validation["warnings"].append(warning)
        validation["ltx_long_horizon_mode_requested"] = long_horizon.get(
            "requested_mode",
            target_dict["ltx_long_horizon_mode"],
        )
        validation["ltx_long_horizon_active"] = bool(long_horizon.get("active"))
        validation["ltx_long_horizon_activation_reason"] = long_horizon.get(
            "activation_reason",
            "disabled",
        )
    return validation


def _contains_audio_terms(prompt: str) -> bool:
    return bool(
        re.search(
            r"\b(audio|sound|sounds|sonic|music|voice|voices|dialogue|ambience|ambient|foley|silence|silent|"
            r"croak|croaks|chirp|chirps|rustle|rustles|rustling|footstep|footsteps|whisper|whispers|hum|hums|humming|"
            r"buzz|buzzes|buzzing|ring|rings|ringing|rain|rains|raining|patter|patters|thunder|waves|traffic|engine|engines)\b",
            prompt or "",
            flags=re.IGNORECASE,
        )
    )


def _audio_instruction(audio_mode: str, audio_guidance: str, target_profile: str = "ltx") -> str:
    mode = _normalize_audio_mode(audio_mode)
    guidance = _audio_guidance_text(audio_guidance)
    if _normalize_target_profile(target_profile) == "minimax_h3":
        if mode == "visual_only":
            return (
                "Set overall_soundscape and non_diegetic_music to N/A. Do not invent dialogue, singing, ambience, Foley, or music."
            )
        if guidance:
            return (
                "Place synchronized dialogue, diegetic music, and event-specific sounds at the exact visible beat inside "
                "integrated_multimodal_description; summarize only ambience, Foley, and non-verbal sounds in overall_soundscape; "
                f"describe audience-only score in non_diegetic_music. Apply this requested sound design: {guidance}."
            )
        if mode == "explicit_sound_design":
            return (
                "Build concrete native audio with the video: synchronize dialogue and event-specific sounds inside the shot timeline, "
                "write 1-4 sentences of ambience/Foley in overall_soundscape, and write 1-3 sentences of audience-only score in "
                "non_diegetic_music with instrumentation, tempo/rhythm, and dynamic development. Avoid abstract mood or explanations "
                "of the score's emotional purpose. Do not duplicate dialogue or music across fields."
            )
        return (
            "Infer restrained, scene-specific native audio from visible actions and setting. Put synchronized events in the shot timeline, "
            "ambient/Foley sound in overall_soundscape, and use non_diegetic_music: N/A unless score is requested or clearly implied. "
            "If the user explicitly requests complete silence, set both audio fields to N/A."
        )
    if mode == "visual_only":
        return "Do not include sound, music, dialogue, ambience, or other audio cues in the LTX prompt."
    if guidance:
        return (
            "Weave this sound design into the LTX prompt as part of the scene, not as a generic appended sentence: "
            f"{guidance}."
        )
    if mode == "explicit_sound_design":
        return (
            "Invent concrete diegetic sound design from the visible action and setting, and weave it into the LTX prompt. "
            "Mention specific sources such as movement, environment, voices, machinery, weather, room tone, or silence only when they fit."
        )
    return (
        "When useful for video, weave one concise, scene-specific audio cue into the LTX prompt itself. "
        "Avoid generic lines like natural scene audio supports the motion."
    )


def _clean_control_text(text: str, max_chars: int = 500) -> str:
    return _sanitize_prompt_text(text, "", max_chars, strip_thinking=True, strip_markdown=True).rstrip(".")


def _effective_duration_seconds(media_metadata: dict[str, Any], target_duration_seconds: float) -> float:
    requested = _safe_float(target_duration_seconds, 0.0)
    if requested > 0:
        return requested
    return _safe_float(media_metadata.get("duration_seconds"), 0.0)


def _ltx25_planning_duration_seconds(
    media_metadata: dict[str, Any],
    target_duration_seconds: float,
    user_prompt: str = "",
) -> float:
    return (
        _effective_duration_seconds(media_metadata, target_duration_seconds)
        or _explicit_prompt_duration_seconds(user_prompt)
        or 5.0
    )


def _ltx25_mode_report(
    configured_mode: Any,
    media_metadata: dict[str, Any],
) -> dict[str, Any]:
    image_count = int(max(0.0, _safe_float(media_metadata.get("sampled_frame_count"), 0.0)))
    return resolve_ltx25_generation_mode(configured_mode, media_metadata, image_count)


def _duration_unit_seconds(value: str, unit: str) -> float:
    multiplier = 3600.0 if unit.lower().startswith(("h", "hr")) else 60.0 if unit.lower().startswith(("m", "min")) else 1.0
    return _safe_float(value, 0.0) * multiplier


def _explicit_prompt_duration_seconds(user_prompt: str) -> float:
    text = _single_paragraph(_sanitize_prompt_text(user_prompt, "", 4000)).lower()
    if not text:
        return 0.0
    combined = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\s*(?:and\s*)?(\d+(?:\.\d+)?)\s*(?:seconds?|secs?)\b",
        text,
    )
    if combined:
        return _safe_float(combined.group(1), 0.0) * 60.0 + _safe_float(combined.group(2), 0.0)
    timecode = re.search(r"\b(?:duration|runtime|length)\s*(?:is|of|:)?\s*(\d{1,3}):(\d{2}(?:\.\d+)?)\b", text)
    if timecode and _safe_float(timecode.group(2), 60.0) < 60.0:
        return _safe_float(timecode.group(1), 0.0) * 60.0 + _safe_float(timecode.group(2), 0.0)
    unit = r"hours?|hrs?|hr|h|minutes?|mins?|min|m|seconds?|secs?|sec|s"
    patterns = (
        rf"\b(?:duration|runtime|length)\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)\s*({unit})\b",
        rf"\b(?:lasting|lasts?|runs?|running|for)\s+(\d+(?:\.\d+)?)\s*({unit})\b",
        rf"\b(?:make|create|generate|render|produce)\b.{{0,48}}?\b(\d+(?:\.\d+)?)\s*[- ]?\s*({unit})\b",
        rf"\b(\d+(?:\.\d+)?)\s*-\s*({unit})\b",
        rf"\b(\d+(?:\.\d+)?)\s*[- ]?\s*({unit})\s+(?:video|clip|sequence|animation|film)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _duration_unit_seconds(match.group(1), match.group(2))
    return 0.0


def _minimax_h3_duration_seconds(
    media_metadata: dict[str, Any],
    target_duration_seconds: float,
    user_prompt: str = "",
) -> float:
    effective = _effective_duration_seconds(media_metadata, target_duration_seconds)
    if effective > 0:
        return effective
    return _explicit_prompt_duration_seconds(user_prompt) or 5.0


def _minimax_h3_duration_advisory(duration_seconds: float) -> str:
    if 4.0 <= duration_seconds <= 15.0:
        return ""
    return "Outside MiniMax's published 4-15 second range; preserved without clamping for local generation."


def _target_context(
    media_metadata: dict[str, Any],
    target_duration_seconds: float,
    ltx_style: str,
    ideogram_aspect_ratio: str,
    ideogram_render_style: str,
    ideogram_exact_text: str,
    ideogram_json_output: bool,
    creativity_mode: str = "editorial",
    creative_strength: float = 0.6,
    negative_prompt_mode: str = "auto",
    negative_prompt_guidance: str = "",
    ltx_generation_mode: str = "auto",
    ltx_long_horizon_mode: str = "off",
    user_prompt: str = "",
    ltx_camera_capability: str = "advanced",
) -> dict[str, Any]:
    effective_duration = round(_effective_duration_seconds(media_metadata, target_duration_seconds), 3)
    ltx_mode_report = _ltx25_mode_report(ltx_generation_mode, media_metadata)
    ltx_planning_duration = _ltx25_planning_duration_seconds(
        media_metadata,
        target_duration_seconds,
        user_prompt,
    )
    ltx_long_horizon = ltx25_long_horizon_plan(
        ltx_long_horizon_mode,
        ltx_planning_duration,
    )
    return {
        "target_duration_seconds": effective_duration,
        "ltx_duration_seconds": effective_duration,
        "ltx_style": _clean_control_text(ltx_style, 240),
        "ltx25_contract_schema": LTX25_CONTRACT_SCHEMA,
        "ltx_generation_mode_requested": ltx_mode_report["configured_mode"],
        "ltx_generation_mode_effective": ltx_mode_report["resolved_mode"],
        "ltx_generation_mode_inference": ltx_mode_report["resolution_source"],
        "ltx_planning_duration_seconds": round(ltx_planning_duration, 3),
        "ltx_long_horizon_mode": ltx_long_horizon["requested_mode"],
        "ltx_long_horizon_active": bool(ltx_long_horizon["active"]),
        "ltx_long_horizon_activation_reason": ltx_long_horizon["activation_reason"],
        "ltx_long_horizon_experimental": bool(ltx_long_horizon["experimental"]),
        "ltx_long_horizon_plan": ltx_long_horizon,
        "ltx_camera_capability": normalize_ltx25_camera_capability(
            ltx_camera_capability
        ),
        "ltx_complexity_limits": ltx25_complexity_budget(
            ltx_mode_report["resolved_mode"],
            ltx_planning_duration,
            ltx_long_horizon["requested_mode"],
        ),
        "ideogram_aspect_ratio": ideogram_aspect_ratio,
        "ideogram_render_style": _clean_control_text(ideogram_render_style, 240),
        "ideogram_exact_text": _sanitize_prompt_text(
            ideogram_exact_text,
            "",
            1000,
            strip_thinking=True,
            strip_markdown=True,
        ),
        "ideogram_json_output": bool(ideogram_json_output),
        "creativity_mode": _normalize_creativity_mode(creativity_mode),
        "creative_strength": _normalize_creative_strength(creative_strength),
        "negative_prompt_mode": _normalize_negative_prompt_mode(negative_prompt_mode),
        "negative_prompt_guidance": _negative_prompt_guidance_text(negative_prompt_guidance),
    }


def _model_visible_target_controls(
    controls: dict[str, Any],
    target_profile: str,
) -> dict[str, Any]:
    """Project shared host controls down to the selected model target."""

    profile = _normalize_target_profile(target_profile)
    if profile == "minimax_h3":
        common_keys = {
            "target_duration_seconds",
            "creativity_mode",
            "creative_strength",
        }
        projected = {
            key: copy.deepcopy(value)
            for key, value in controls.items()
            if key in common_keys or key.startswith("minimax_h3_")
        }
        projected["exclusion_mode"] = copy.deepcopy(
            controls.get("negative_prompt_mode", "auto")
        )
        projected["exclusion_guidance"] = copy.deepcopy(
            controls.get("negative_prompt_guidance", "")
        )
        return projected
    if profile == "ideogram4":
        return {
            key: copy.deepcopy(value)
            for key, value in controls.items()
            if not key.startswith(("ltx_", "ltx25_", "minimax_h3_"))
        }
    return {
        key: copy.deepcopy(value)
        for key, value in controls.items()
        if not key.startswith(("ideogram_", "minimax_h3_"))
    }


def _model_visible_media_metadata(
    media_metadata: dict[str, Any],
    target_profile: str,
) -> dict[str, Any]:
    """Keep inactive-target implementation data out of the model prompt."""

    profile = _normalize_target_profile(target_profile)
    hidden_prefixes: tuple[str, ...]
    if profile == "minimax_h3":
        hidden_prefixes = ("ltx_", "ltx25_", "ideogram_")
    elif profile == "ideogram4":
        hidden_prefixes = ("ltx_", "ltx25_", "minimax_h3_")
    else:
        hidden_prefixes = ("ideogram_", "minimax_h3_")
    return {
        key: copy.deepcopy(value)
        for key, value in media_metadata.items()
        if not key.startswith(hidden_prefixes)
    }


def _model_visible_master_prompt(master_prompt: str, target_profile: str) -> str:
    """Replace the bundled all-target preamble for an H3-only request."""

    value = str(master_prompt or "").strip()
    if _normalize_target_profile(target_profile) != "minimax_h3":
        return value
    default = DEFAULT_MASTER_PROMPT.strip()
    if value.startswith(default):
        suffix = value[len(default) :]
        return f"{MINIMAX_H3_MASTER_PROMPT.strip()}{suffix}"
    return value


def _normalize_creativity_mode(creativity_mode: str) -> str:
    value = (creativity_mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"faithful", "editorial", "cinematic", "concept_art", "wild"}:
        return value
    return "editorial"


def _normalize_creative_strength(creative_strength: Any) -> float:
    return round(max(0.0, min(1.5, _safe_float(creative_strength, 0.6))), 3)


def _creativity_instruction(creativity_mode: str, creative_strength: float) -> str:
    mode = _normalize_creativity_mode(creativity_mode)
    strength = _normalize_creative_strength(creative_strength)
    if strength <= 0.05 or mode == "faithful":
        return (
            "Faithful mode: preserve the user prompt and supplied media tightly. Add only necessary concrete details for clarity; "
            "do not add new props, eras, wardrobe, materials, worldbuilding, or stylistic surprises unless requested."
        )
    if mode == "editorial":
        return (
            f"Editorial creativity strength {strength}: preserve the core request while adding tasteful art direction: "
            "specific materials, lighting, palette, spatial layout, styling, typography, and product or character details."
        )
    if mode == "cinematic":
        return (
            f"Cinematic creativity strength {strength}: preserve the core request while adding filmic production design: "
            "era cues, set design, lens/camera language, blocking, lighting contrast, atmosphere, props, wardrobe, and sound when relevant."
        )
    if mode == "concept_art":
        return (
            f"Concept-art creativity strength {strength}: preserve the subject and exact requested text while adding distinctive worldbuilding, "
            "genre, environment, props, silhouettes, materials, color motifs, and memorable visual hooks."
        )
    return (
        f"Wild creativity strength {strength}: preserve the required subject, exact text, and safety constraints, but push bolder art direction, "
        "surprising genre fusion, unusual set details, dramatic composition, and memorable specifics. Keep the output coherent and usable."
    )


def _creative_strength_band(creative_strength: Any) -> str:
    strength = _normalize_creative_strength(creative_strength)
    if strength <= 0.05:
        return "faithful"
    if strength <= 0.35:
        return "subtle"
    if strength <= 0.8:
        return "moderate"
    if strength <= 1.15:
        return "strong"
    return "maximal"


def _ltx_motion_ambition_instruction(creative_strength: Any) -> str:
    """Translate creative strength into camera-path ambition for LTX video."""

    band = _creative_strength_band(creative_strength)
    if band == "subtle":
        return (
            "Use deliberate camera motion with visible but restrained displacement or parallax and a concrete end frame."
        )
    if band == "moderate":
        return (
            "Use clearly perceptible spatial displacement, compatible layered camera-motion phases, layered parallax, and an intentional ending composition."
        )
    if band == "strong":
        return (
            "Use assertive multi-phase spatial travel with purposeful acceleration or deceleration, layered parallax, and a decisive ending composition."
        )
    if band == "maximal":
        return (
            "Use an ambitious but physically executable compound camera path with pronounced spatial travel, speed variation, foreground occlusion or reveal, layered parallax, and a decisive visual payoff."
        )
    return "Add no optional camera move."


def _ltx_creativity_instruction(
    creativity_mode: str,
    creative_strength: float,
    generation_mode: str,
    camera_capability: str = "advanced",
) -> str:
    """Return LTX-specific creativity policy instead of image-prompt residue.

    Creativity changes *what kind* of compatible detail the Director may add;
    it is deliberately separate from sampling temperature and seed.  Conditioned
    modes keep their frame anchors authoritative while still giving every mode a
    useful temporal-motion and sound vocabulary.
    """

    mode = _normalize_creativity_mode(creativity_mode)
    strength = _normalize_creative_strength(creative_strength)
    band = _creative_strength_band(strength)
    resolved_mode = normalize_ltx25_generation_mode(generation_mode)
    conditioned = resolved_mode in {"image_to_video", "first_last_frame"}
    if normalize_ltx25_camera_capability(camera_capability) == "stable":
        if mode == "faithful" or band == "faithful":
            treatment = (
                "Faithful LTX direction: preserve the requested subject, action, sound, style, and supplied-media facts tightly without optional story beats or camera ambition."
            )
        elif mode == "editorial":
            treatment = (
                f"Editorial LTX direction at {band} intensity (strength {strength}): improve clarity through controlled blocking, readable action timing, material response, restrained atmosphere, clean causal sound, and polished composition."
            )
        elif mode == "cinematic":
            treatment = (
                f"Cinematic LTX direction at {band} intensity (strength {strength}): emphasize filmic performance, motivated lighting, lens and focus character, atmosphere, synchronized sound, and a legible terminal composition."
            )
        elif mode == "concept_art":
            treatment = (
                f"Concept-art LTX direction at {band} intensity (strength {strength}): emphasize distinctive materials, color, silhouette, environmental response, imaginative motion by existing visible elements, and coherent depth."
            )
        else:
            treatment = (
                f"Wild LTX direction at {band} intensity (strength {strength}): push bold art direction, performance, choreography, environmental motion, lighting response, and sound among the requested or already visible elements while keeping spatial geometry coherent."
            )
        camera_policy = (
            "Camera capability is Stable / base model and overrides creativity intensity. Prefer a locked, static, or stabilized camera. "
            "If camera movement materially improves legibility, use only one restrained single-axis move: a short gentle push or pull, a small pan or tilt, or a short lateral track, then settle into stable framing. "
            "Do not author an orbit, circular arc, circle-around, roll, rotation, spin, camera swirl, sweeping or whip movement, dolly zoom, camera shake, handheld pursuit, compound or multi-axis path, direction reversal, pronounced/layered/dizzying parallax, or sustained camera travel. "
            "A user request that genuinely requires one of those moves must use Advanced / controlled camera; do not silently turn Stable mode into an advanced path."
        )
        if conditioned:
            anchor_policy = (
                "Frame anchors, explicit style guidance, user facts, exact speech, and continuity take precedence over creativity. "
                "Preserve anchored identity, subject count, pose or endpoint state, framing, composition, camera geometry, location, and time of day. "
                "When future camera behavior is unspecified, hold the supplied viewpoint; fill time with subject, prop, lighting, atmospheric, environmental, and sound evolution rather than camera travel. "
                "Do not add or replace a subject or actor, cut, location, or other anchored visual fact."
            )
        else:
            anchor_policy = (
                "The user brief, explicit style guidance, exact speech, duration, shot budget, and continuity contract take precedence over creativity. "
                "Keep every addition visible or audible within the available time."
            )
        return f"{treatment} {camera_policy} {anchor_policy}"

    if mode == "faithful" or band == "faithful":
        treatment = (
            "Faithful LTX direction: execute only motion, camera behavior, sound, and visible changes required by the brief or supported by the supplied media. "
            "Preserve requested camera speed, direction, and every compatible movement phase; do not downgrade fast, energetic, or compound choreography to a slow or static move. "
            "Add no optional subject, prop, location, story beat, cut, camera move, dialogue, or stylistic premise."
        )
    elif mode == "editorial":
        treatment = (
            f"Editorial LTX direction at {band} intensity (strength {strength}): improve clarity and polish through controlled blocking, readable action timing, "
            "coherent material response, restrained atmospheric detail, clean causal sound, and precise subject-responsive tracking or reframing. "
            "Choose a clean continuous lateral, depth, or vertical path when camera motion improves legibility. Do not introduce typography unless the brief requests visible text."
        )
    elif mode == "cinematic":
        treatment = (
            f"Cinematic LTX direction at {band} intensity (strength {strength}): emphasize filmic pacing, purposeful blocking, lens/depth and focus behavior, "
            "motivated lighting response, atmosphere, synchronized sound, and a designed spatial camera path. "
            "Choose a coherent subset of dolly, truck, boom, arc, orbit, controlled handheld pursuit, foreground passes, and focus transitions rather than defaulting to a generic pan or slow zoom."
        )
    elif mode == "concept_art":
        treatment = (
            f"Concept-art LTX direction at {band} intensity (strength {strength}): emphasize distinctive but coherent material, color, silhouette, environmental response, "
            "imaginative motion by existing visible elements, expressive perspective change, and bold but continuous parallax. "
            "Let lateral, vertical, orbital, or depth travel reveal the existing design without turning visual treatment into extra subjects, locations, cuts, or unrelated story beats."
        )
    else:
        treatment = (
            f"Wild LTX direction at {band} intensity (strength {strength}): push one bold, coherent motion-and-sound premise using the requested or already visible subjects, props, and setting. "
            "Choose one memorable compound camera-subject interaction—such as accelerating pursuit, a sweeping arc or orbit, a boom or roll, or an occlusion reveal—over multiplying actors, cuts, locations, or contradictory visual facts."
        )

    if mode != "faithful" and band != "faithful":
        treatment = (
            f"{treatment} {_ltx_motion_ambition_instruction(strength)} "
            "Do not default to a locked shot, a slow zoom, or a generic pan solely because the input is a still image. "
            "Honor any explicit locked-off, static-camera, motion-path, or speed instruction from the user or verified media."
        )

    if conditioned:
        if resolved_mode == "first_last_frame":
            if mode == "faithful" or band == "faithful":
                precedence = (
                    "Frame anchors, explicit style guidance, user facts, exact speech, and the continuity contract take precedence over creativity. "
                    "The supplied frames fix both endpoint poses, framing, compositions, and camera geometries, but they do not fix the camera path between them. Preserve every explicitly requested camera speed, direction, and compatible phase exactly. When the bridge is unspecified, choose the least-invasive physically suitable continuous camera state or path that connects the exact endpoints. "
                    "Always honor an explicit locked-off or static-camera instruction; it wins over the creativity mode. "
                    "Do not add or replace a subject or actor, cut, location, time of day, endpoint camera geometry, or other anchored visual fact."
                )
            else:
                precedence = (
                    "Frame anchors, explicit style guidance, user facts, exact speech, and the continuity contract take precedence over creativity. "
                    "The supplied frames fix both endpoint poses, framing, compositions, and camera geometries, but they do not fix the camera path between them or require an inert bridge. An otherwise unspecified bridge may use the selected mode's dynamic, physically continuous camera choreography between those exact endpoints. "
                    "Always honor an explicit locked-off or static-camera instruction; it wins over the creativity mode. "
                    "Do not add or replace a subject or actor, cut, location, time of day, endpoint camera geometry, or other anchored visual fact."
                )
        else:
            if mode == "faithful" or band == "faithful":
                precedence = (
                    "Frame anchors, explicit style guidance, user facts, exact speech, and the continuity contract take precedence over creativity. "
                    "The supplied first frame fixes the opening pose, framing, composition, and camera geometry only at the first instant; it does not fix the future camera trajectory or imply a locked camera afterward. Preserve every explicitly requested camera speed, direction, and compatible phase exactly. When the future path is unspecified, choose the least-invasive physically suitable continuous camera state or path from that exact opening. "
                    "Always honor an explicit locked-off or static-camera instruction; it wins over the creativity mode. "
                    "Do not add or replace a subject or actor, cut, location, time of day, opening camera geometry, or other anchored visual fact."
                )
            else:
                precedence = (
                    "Frame anchors, explicit style guidance, user facts, exact speech, and the continuity contract take precedence over creativity. "
                    "The supplied first frame fixes the opening pose, framing, composition, and camera geometry only at the first instant; it does not fix the future camera trajectory or imply a locked camera afterward. An otherwise unspecified future camera path may use the selected mode's dynamic, physically continuous camera choreography from that exact opening. "
                    "Always honor an explicit locked-off or static-camera instruction; it wins over the creativity mode. "
                    "Do not add or replace a subject or actor, cut, location, time of day, opening camera geometry, or other anchored visual fact."
                )
    else:
        precedence = (
            "The user brief, explicit style guidance, exact speech, duration, shot budget, and continuity contract take precedence over creativity. "
            "Keep every addition visible or audible within the available time."
        )
    return f"{treatment} {precedence}"


def _target_structure_anchor(
    target_profile: str,
    ideogram_json_output: bool,
    minimax_h3_mode: str = "t2va",
    ltx_generation_mode: str = "text_to_video",
    ltx_long_horizon_active: bool = False,
    ltx_camera_capability: str = "advanced",
) -> str:
    profile = _normalize_target_profile(target_profile)
    if profile == "minimax_h3":
        if _normalize_minimax_h3_mode(minimax_h3_mode) == "ref2va":
            return (
                "MiniMax H3 Ref2VA prompt: exactly six English sections in this order: subject_definitions, summary, "
                "retention_analysis, detailed_description, overall_soundscape, non_diegetic_music. Define semantic <Subject N> "
                "labels from the declared <Picture N>, <Video N>, and <Audio N> assets; begin summary with the official "
                "square-bracketed task types; put one or two style sentences before [Shot 1] in detailed_description; and use "
                "the exact declared reference labels wherever their roles apply. Preserve these field names, order, and blank lines."
            )
        return (
            "MiniMax H3 T2VA prompt: integrated_multimodal_description begins directly with [Shot 1] and no timestamp; "
            "later cuts use [Shot N] At MM:SS.mmm, the camera cuts to... with strictly increasing cut times; then one blank line, "
            "overall_soundscape; then one blank line, non_diegetic_music. Preserve these exact field names, order, and line breaks."
        )
    if profile == "ideogram4":
        if ideogram_json_output:
            return (
                "Ideogram JSON: source time-of-day, exposure, lens/depth-of-field, and lighting -> background -> primary subject/elements -> spatial lighting/style -> exact quoted text content. "
                "Map these into the caption JSON schema, not prose instructions."
            )
        return (
            "Ideogram prompt: source time-of-day, exposure, lens/depth-of-field, and lighting -> background -> primary subject -> spatial lighting/style -> exact quoted text content. "
            "Keep text placement and typography concrete."
        )
    ltx_mode = normalize_ltx25_generation_mode(ltx_generation_mode)
    stable_camera = normalize_ltx25_camera_capability(
        ltx_camera_capability
    ) == "stable"
    if ltx_long_horizon_active and ltx_mode == "image_to_video":
        if stable_camera:
            return (
                "Experimental long-horizon LTX-2.5 I2V prompt: minimal critical anchor set from the supplied first frame -> "
                "establish a locked/stabilized camera state -> sustain one dominant subject-and-environment motion through most of the same take -> "
                "settle/hold on a concrete terminal composition -> continuous synchronized audio when enabled. These are four pacing "
                "phases inside one shot, not chapters or cuts. Fill time without sustained camera travel; emit only one compact chronological paragraph."
            )
        return (
            "Experimental long-horizon LTX-2.5 I2V prompt: minimal critical anchor set from the supplied first frame -> "
            "establish -> commit to one dominant physically continuous path -> sustain/reveal through most of the same take -> "
            "settle/hold on a concrete terminal composition -> continuous synchronized audio when enabled. These are four pacing "
            "phases inside one shot, not chapters or cuts. Silently conserve every anchor not explicitly requested to transform; "
            "emit only one compact chronological paragraph."
        )
    if ltx_long_horizon_active and ltx_mode == "first_last_frame":
        if stable_camera:
            return (
                "Experimental long-horizon LTX-2.5 FLF prompt: minimal critical endpoint correspondences -> establish -> "
                "sustain one causal subject-and-environment bridge with a locked or single-axis stabilized camera -> settle/hold on the exact supplied last frame -> continuous synchronized audio when enabled. "
                "These are four pacing phases inside one shot, not chapters or cuts. Emit one compact chronological paragraph."
            )
        return (
            "Experimental long-horizon LTX-2.5 FLF prompt: minimal critical endpoint correspondences -> establish -> commit -> "
            "sustain one causal bridge -> settle/hold on the exact supplied last frame -> continuous synchronized audio when enabled. "
            "These are four pacing phases inside one shot, not chapters or cuts. Emit one compact chronological paragraph."
        )
    if ltx_long_horizon_active:
        if stable_camera:
            return (
                "Experimental long-horizon LTX-2.5 prompt: establish a stable camera state -> sustain one dominant subject-and-environment motion -> "
                "settle/hold on a concrete terminal composition -> continuous synchronized audio when enabled. Fill time without sustained "
                "camera travel. These are four pacing phases, not chapters, shots, or cuts. Emit one compact chronological paragraph."
            )
        return (
            "Experimental long-horizon LTX-2.5 prompt: establish -> commit to one dominant continuous camera-and-subject path -> "
            "sustain/reveal through most of the take -> settle/hold on a concrete terminal composition -> continuous synchronized "
            "audio when enabled. These are four pacing phases, not chapters, shots, or cuts. Emit one compact chronological paragraph."
        )
    if ltx_mode == "image_to_video":
        if stable_camera:
            return (
                "LTX-2.5 I2V prompt: exact supplied first-frame facts -> one continuous plausible subject action with a locked/stabilized camera or one restrained single-axis move -> concrete ending composition -> synchronized sound when enabled. Write one chronological paragraph with no cuts."
            )
        return (
            "LTX-2.5 I2V prompt: exact supplied first-frame facts -> one continuous plausible subject action and physically continuous camera path -> concrete ending composition -> synchronized sound when enabled. Compatible simultaneous or sequential camera-motion phases may form that path. Write one chronological paragraph with no cuts."
        )
    if ltx_mode == "first_last_frame":
        return (
            "LTX-2.5 FLF prompt: exact supplied first-frame facts -> one continuous causal transition -> exact supplied last-frame state -> synchronized sound when enabled. Write one chronological paragraph with no cuts."
        )
    if ltx_mode == "legacy_video":
        return (
            "LTX-2.5 source-video prompt: source identity -> opening framing -> chronological action and observed camera choreography -> lens/focus/exposure/blocking -> closing framing -> sound when enabled. Write one continuous ready-to-generate paragraph."
        )
    return (
        "LTX-2.5 T2V prompt: established subjects and setting -> opening shot scale, camera state, and viewpoint -> chronological action within the duration budget -> ending composition -> synchronized sound when enabled. Write one continuous ready-to-generate paragraph."
    )


def _normalize_negative_prompt_mode(mode: str) -> str:
    value = (mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"auto", "empty", "custom"}:
        return value
    return "auto"


def _negative_prompt_guidance_text(text: str) -> str:
    return _sanitize_prompt_text(text, "", 1200, strip_thinking=True, strip_markdown=True).rstrip(".")


def _negative_prompt_instruction(mode: str, guidance: str) -> str:
    mode = _normalize_negative_prompt_mode(mode)
    guidance = _negative_prompt_guidance_text(guidance)
    if mode == "empty":
        return "Return an empty negative_prompt string."
    if mode == "custom":
        if guidance:
            return f"Use this exact negative_prompt string and do not add to it: {guidance}"
        return "Return an empty negative_prompt string because custom mode has no supplied text."
    return (
        "Write a short comma-separated negative_prompt only when it will help the target generator avoid obvious artifacts, "
        "unwanted text, extra subjects, bad anatomy, or layout failures. Return an empty string when no negative prompt is needed."
    )


def _apply_negative_prompt_policy(value: Any, mode: str, guidance: str) -> str:
    mode = _normalize_negative_prompt_mode(mode)
    guidance = _negative_prompt_guidance_text(guidance)
    if mode == "empty":
        return ""
    if mode == "custom":
        return guidance
    return _sanitize_prompt_text(str(value or ""), "", 1200)


def _creative_art_direction_phrase(
    creativity_mode: str,
    creative_strength: float,
    target: str = "image",
    generation_mode: str = "text_to_video",
    camera_capability: str = "advanced",
) -> str:
    mode = _normalize_creativity_mode(creativity_mode)
    strength = _normalize_creative_strength(creative_strength)
    if strength <= 0.05 or mode == "faithful":
        return ""
    target_value = str(target or "image").strip().lower().replace("-", "_").replace(".", "_").replace(" ", "_")
    if target_value in {"ltx", "ltx2", "ltx2_3", "ltx_2_3", "ltx2_5", "ltx_2_5"}:
        band = _creative_strength_band(strength)
        conditioned = normalize_ltx25_generation_mode(generation_mode) in {
            "image_to_video",
            "first_last_frame",
        }
        if normalize_ltx25_camera_capability(camera_capability) == "stable":
            if mode == "editorial":
                phrase = f"{band.capitalize()} editorial treatment adds polished blocking, material response, restrained atmosphere, and causal sound."
            elif mode == "cinematic":
                phrase = f"{band.capitalize()} cinematic treatment adds filmic performance, motivated lighting and focus, atmosphere, and synchronized sound."
            elif mode == "concept_art":
                phrase = f"{band.capitalize()} concept-art treatment gives existing forms and materials coherent motion, color response, and atmospheric detail."
            else:
                phrase = f"{band.capitalize()} wild treatment drives bold performance, environmental motion, lighting response, and sound among the requested or visible elements."
            camera_clause = (
                "The camera remains locked or stabilized, or makes only one short gentle single-axis push, pull, pan, tilt, or lateral track before settling."
            )
            if conditioned:
                return (
                    f"{phrase} {camera_clause} The supplied frame anchors remain exact at their anchored instants; no new actor, location, cut, or unrelated action is introduced."
                )
            return f"{phrase} {camera_clause} No extra cut, actor, or unrelated action is introduced."
        if mode == "editorial":
            phrase = (
                f"{band.capitalize()} editorial treatment keeps the action legible through polished blocking, precise subject-responsive tracking or reframing, material response, restrained atmosphere, and causal sound."
            )
        elif mode == "cinematic":
            phrase = (
                f"{band.capitalize()} cinematic treatment uses a designed continuous dolly, truck, boom, arc, orbit, or controlled handheld path with filmic pacing, layered parallax, motivated focus and depth behavior, atmospheric response, and synchronized sound."
            )
        elif mode == "concept_art":
            phrase = (
                f"{band.capitalize()} concept-art treatment gives existing forms and materials imaginative but coherent motion, expressive continuous perspective travel, bold parallax, color response, and atmospheric detail."
            )
        else:
            phrase = (
                f"{band.capitalize()} wild treatment drives one bold, physically continuous compound camera-subject interaction and sound payoff among the requested or already visible subjects and props."
            )
        if conditioned:
            return (
                f"{phrase} The supplied frame anchors remain exact at their anchored instants; no new actor, location, cut, or unrelated action is introduced. A still anchor does not require the camera to remain locked between anchored instants."
            )
        return f"{phrase} No extra cut, actor, or unrelated action is introduced."
    if mode == "editorial":
        return "Refined editorial art direction adds premium materials, deliberate styling, polished lighting, and a clear visual hierarchy."
    if mode == "cinematic":
        return "Cinematic art direction adds film-set texture, purposeful blocking, lens-aware composition, motivated lighting, and atmospheric detail."
    if mode == "concept_art":
        return "Concept-art direction adds distinctive worldbuilding, era cues, unusual props, bold silhouettes, and a memorable visual hook."
    return "Bold surreal art direction adds unexpected genre fusion, striking set details, dramatic composition, and vivid memorable specifics while keeping the core subject recognizable."


def _normalize_target_profile(target_profile: str) -> str:
    value = (target_profile or "").strip().lower().replace("-", "_").replace(".", "_").replace(" ", "_")
    if value in {"ltx2", "ltx2_3", "ltx_2_3", "ltx2_5", "ltx_2_5", "ltx"}:
        return "ltx"
    if value in {"ideogram", "ideogram4", "ideogram4_json", "ideogram_4"}:
        return "ideogram4"
    if value in {"minimax", "minimaxh3", "minimax_h3", "h3", "hailuo_h3", "hailuo_03"}:
        return "minimax_h3"
    if value == "both":
        return "ltx"
    return "ltx"


def _normalize_minimax_h3_mode(mode: str) -> str:
    value = (mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"ref2va", "ref2v", "r2v", "reference", "reference_to_video", "reference_to_audio_video"}:
        return "ref2va"
    return "t2va"


_MINIMAX_H3_REFERENCE_POLICY_SCHEMA = "dg-h3-reference-policy/1"
_MINIMAX_H3_REFERENCE_POLICY_CHOICES = (
    "Auto (recommended)",
    "One picture - all visual attributes",
    "Primary subject + environment/style",
    "Primary subject + same-subject contact sheet",
    "Primary subject + supporting subjects/objects contact sheet",
    "Ordered independent subjects/objects",
    "Custom manifest",
)

_MINIMAX_H3_ONE_PICTURE_MANIFEST = (
    "<Picture 1>: [dg:identity,appearance,object,count,color,environment,lighting,composition] "
    "sole visual reference for all reusable visible content and scene attributes; picture count does not determine semantic Subject count."
)
_MINIMAX_H3_SUBJECT_ENVIRONMENT_MANIFEST = (
    "<Picture 1>: [dg:identity,appearance,color] protagonist identity, face, body, silhouette, colors, wardrobe, "
    "accessories, and distinguishing traits; preserve exactly across every shot; this is a full-sequence reference, "
    "not a first or last frame.\n"
    "<Picture 2>: [dg:environment,lighting,color,composition] environment, production design, palette, lighting, "
    "composition, and visual style; do not copy any pictured person or creature identity; this is a full-sequence "
    "reference, not a keyframe."
)
_MINIMAX_H3_SAME_SUBJECT_CONTACT_SHEET_MANIFEST = (
    "<Picture 1>: [dg:identity,appearance,object,count,color] sole authority for the primary <Subject 1> identity, "
    "body, wardrobe, accessories, colors, and distinguishing design; preserve it consistently across the sequence.\n"
    "<Picture 2>: [dg:identity,appearance,object,color] one same-subject contact sheet containing alternate views or "
    "details of that same primary <Subject 1>; use every panel only as corroborating identity and appearance evidence. "
    "Multiple views remain one semantic Subject. Never reproduce the sheet grid, cells, panel seams, borders, labels, "
    "backgrounds, repeated poses, or layout in the generated scene."
)
_MINIMAX_H3_MULTI_ENTITY_CONTACT_SHEET_MANIFEST = (
    "<Picture 1>: [dg:identity,appearance,object,count,color] sole authority for the primary <Subject 1> identity, "
    "body, wardrobe, accessories, colors, and distinguishing design; preserve it consistently across the sequence.\n"
    "<Picture 2>: [dg:identity,appearance,object,count,color] one multi-entity contact sheet containing independently "
    "selectable supporting people, creatures, or hero objects named in the brief. Create one semantic <Subject N> for "
    "each distinct referenced entity actually requested in the video and cite <Picture 2> in every such definition. "
    "Multiple views of one entity remain one Subject. Never transfer the sheet grid, cells, panel seams, borders, labels, "
    "backgrounds, repeated poses, layout, or blended traits between entities."
)


_MINIMAX_H3_REFERENCE_MANIFEST_PRESETS = {
    "custom": "",
    "1 image - all visual attributes": _MINIMAX_H3_ONE_PICTURE_MANIFEST,
    "2 images - subject + environment/style": _MINIMAX_H3_SUBJECT_ENVIRONMENT_MANIFEST,
    "2 images - main subject + same-subject contact sheet": _MINIMAX_H3_SAME_SUBJECT_CONTACT_SHEET_MANIFEST,
    "2 images - main subject + multi-entity contact sheet": _MINIMAX_H3_MULTI_ENTITY_CONTACT_SHEET_MANIFEST,
}


def _normalize_minimax_h3_reference_policy_layout(value: Any) -> str:
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value if value is not None else "auto").strip().casefold(),
    ).strip("_")
    aliases = {
        "": "auto",
        "auto": "auto",
        "auto_recommended": "auto",
        "one_picture_all_visual_attributes": "one_picture_all_visual_attributes",
        "1_image_all_visual_attributes": "one_picture_all_visual_attributes",
        "primary_subject_environment_style": "primary_subject_environment_style",
        "2_images_subject_environment_style": "primary_subject_environment_style",
        "primary_subject_same_subject_contact_sheet": "primary_subject_same_subject_contact_sheet",
        "2_images_main_subject_same_subject_contact_sheet": "primary_subject_same_subject_contact_sheet",
        "primary_subject_supporting_subjects_objects_contact_sheet": "primary_subject_multi_entity_contact_sheet",
        "primary_subject_multi_entity_contact_sheet": "primary_subject_multi_entity_contact_sheet",
        "2_images_main_subject_multi_entity_contact_sheet": "primary_subject_multi_entity_contact_sheet",
        "ordered_independent_subjects_objects": "ordered_independent_subjects_objects",
        "custom": "custom_manifest",
        "custom_manifest": "custom_manifest",
    }
    return aliases.get(normalized, "auto")


def _h3_reference_policy_config_to_dict(
    config: H3ReferencePolicyConfig | dict[str, Any] | str | None,
) -> dict[str, Any]:
    if isinstance(config, H3ReferencePolicyConfig):
        data = asdict(config)
    elif isinstance(config, dict):
        data = config.copy()
    elif isinstance(config, str):
        data = {"layout": config}
    else:
        data = {}
    return {
        "schema": _MINIMAX_H3_REFERENCE_POLICY_SCHEMA,
        "layout": _normalize_minimax_h3_reference_policy_layout(data.get("layout", "auto")),
        "custom_manifest": _normalize_minimax_h3_reference_manifest(
            str(data.get("custom_manifest", ""))
        ),
        "expected_subject_count": _normalize_minimax_h3_expected_subject_count(
            data.get("expected_subject_count", 0)
        ),
    }


def _make_h3_reference_policy_config(
    layout: Any = "auto",
    custom_manifest: str = "",
    expected_subject_count: int = 0,
) -> H3ReferencePolicyConfig:
    data = _h3_reference_policy_config_to_dict(
        {
            "layout": layout,
            "custom_manifest": custom_manifest,
            "expected_subject_count": expected_subject_count,
        }
    )
    data.pop("schema", None)
    return H3ReferencePolicyConfig(**data)


def _minimax_h3_reference_policy_manifest(
    config: H3ReferencePolicyConfig | dict[str, Any] | str | None,
    picture_count: int,
    video_count: int = 0,
) -> str:
    """Compile a host-owned Ref2VA manifest from an explicit layout and inventory."""

    policy = _h3_reference_policy_config_to_dict(config)
    layout = policy["layout"]
    picture_count = max(0, int(picture_count))
    video_count = max(0, int(video_count))
    if picture_count > 9 or video_count > 3:
        raise ValueError(
            "MiniMax H3 Ref2VA supports at most 9 Picture references and 3 Video references."
        )
    if layout == "custom_manifest":
        manifest = policy["custom_manifest"]
        if not manifest:
            raise ValueError(
                "MiniMax H3 Ref2VA Custom manifest is selected, but the policy node's custom_manifest box is empty."
            )
        return manifest
    if picture_count == 0 and video_count == 0:
        raise ValueError(
            "MiniMax H3 Ref2VA has no visual reference in DG_CONTEXT. Connect the same ordered image(s) or video "
            "to the Context Hub for DiffusionGemma analysis and to the matching native H3 reference sockets."
        )

    required_picture_counts = {
        "one_picture_all_visual_attributes": 1,
        "primary_subject_environment_style": 2,
        "primary_subject_same_subject_contact_sheet": 2,
        "primary_subject_multi_entity_contact_sheet": 2,
    }
    required_count = required_picture_counts.get(layout)
    if required_count is not None and picture_count != required_count:
        contact_note = (
            " A contact sheet is one Picture asset, so connect the primary image and the contact sheet as two separate images."
            if "contact_sheet" in layout
            else ""
        )
        raise ValueError(
            f"MiniMax H3 Ref2VA reference policy '{layout}' requires exactly {required_count} attached Picture "
            f"reference(s), but DG_CONTEXT contains {picture_count}.{contact_note}"
        )

    lines: list[str] = []
    if layout == "one_picture_all_visual_attributes":
        lines.append(_MINIMAX_H3_ONE_PICTURE_MANIFEST)
    elif layout == "primary_subject_environment_style":
        lines.extend(_MINIMAX_H3_SUBJECT_ENVIRONMENT_MANIFEST.splitlines())
    elif layout == "primary_subject_same_subject_contact_sheet":
        lines.extend(_MINIMAX_H3_SAME_SUBJECT_CONTACT_SHEET_MANIFEST.splitlines())
    elif layout == "primary_subject_multi_entity_contact_sheet":
        lines.extend(_MINIMAX_H3_MULTI_ENTITY_CONTACT_SHEET_MANIFEST.splitlines())
    elif layout == "ordered_independent_subjects_objects":
        for ordinal in range(1, picture_count + 1):
            lines.append(
                f"<Picture {ordinal}>: [dg:identity,appearance,object,count,color] sole visual authority for ordered "
                f"independent semantic subject or hero object {ordinal}; preserve its visible identity, geometry, materials, "
                "colors, and distinguishing traits without blending attributes from another Picture."
            )
    elif picture_count == 1:
        lines.append(_MINIMAX_H3_ONE_PICTURE_MANIFEST)
    elif picture_count >= 2:
        lines.append(
            "<Picture 1>: [dg:identity,appearance,object,count,color] primary subject or hero object identity, body, "
            "wardrobe, accessories, geometry, colors, and distinguishing traits; preserve consistently across the sequence."
        )
        for ordinal in range(2, picture_count + 1):
            lines.append(
                f"<Picture {ordinal}>: [dg:identity,appearance,object,count,color,environment,lighting,composition] "
                "ordered supporting visual reference. Use the user brief to decide whether it supplies a supporting subject, "
                "hero object, environment, wardrobe, prop, palette, lighting, or composition; transfer only requested content. "
                "If it is a board or contact sheet, extract requested entities rather than reproducing its grid, panels, labels, "
                "background, or layout, and never blend traits between distinct items."
            )

    for ordinal in range(1, video_count + 1):
        lines.append(
            f"<Video {ordinal}>: [dg:action,motion,camera,temporal,composition] ordered temporal reference for visible "
            "action, motion, blocking, camera choreography, composition changes, timing, and continuity; do not override "
            "Picture-sourced identity or appearance unless the brief explicitly asks for that transfer."
        )
    return _normalize_minimax_h3_reference_manifest("\n".join(lines))


def _normalize_minimax_h3_reference_manifest(manifest: str) -> str:
    value = _sanitize_structured_prompt_text(manifest, "", 6000)
    return re.sub(
        r"<\s*(picture|image|video|audio)\s+(\d+)\s*>",
        lambda match: f"<{'Picture' if match.group(1).casefold() in {'picture', 'image'} else match.group(1).title()} {int(match.group(2))}>",
        value,
        flags=re.IGNORECASE,
    )


def _resolve_minimax_h3_reference_manifest(manifest: str, preset: Any = "custom") -> str:
    selected = str(preset if preset is not None else "custom").strip().lower()
    preset_manifest = _MINIMAX_H3_REFERENCE_MANIFEST_PRESETS.get(selected, "")
    return _normalize_minimax_h3_reference_manifest(preset_manifest or manifest)


def _strip_grounding_role_annotations(text: str) -> str:
    return re.sub(
        # Accept the canonical closed form and a line-dangling form produced
        # when generation truncates before the closing bracket. Do not consume
        # neighboring lines, since each manifest role is line scoped.
        r"[ \t]*(?:\[\s*)?dg\s*:[^\]\r\n]*(?:\]|(?=\r?$))[ \t]*",
        " ",
        str(text or ""),
        flags=re.IGNORECASE | re.MULTILINE,
    ).strip()


def _minimax_h3_reference_tags(text: str) -> list[str]:
    tags: list[str] = []
    for kind, ordinal in re.findall(r"<\s*(Picture|Video|Audio)\s+(\d+)\s*>", text or "", flags=re.IGNORECASE):
        tag = f"<{kind.title()} {int(ordinal)}>"
        if tag not in tags:
            tags.append(tag)
    return tags


def _minimax_h3_reference_definitions(manifest: str) -> list[tuple[str, str]]:
    definitions: list[tuple[str, str]] = []
    for match in re.finditer(
        r"(?m)^\s*<\s*(Picture|Video|Audio)\s+(\d+)\s*>\s*[:=\-]\s*(\S.*)$",
        _normalize_minimax_h3_reference_manifest(manifest),
        flags=re.IGNORECASE,
    ):
        definitions.append((f"<{match.group(1).title()} {int(match.group(2))}>", match.group(3).strip()))
    return definitions


def _minimax_h3_reference_manifest_validation_reasons(manifest: str) -> list[str]:
    normalized = _normalize_minimax_h3_reference_manifest(manifest)
    definitions = _minimax_h3_reference_definitions(normalized)
    reasons: list[str] = []
    if not definitions:
        return ["minimax_h3_ref_manifest_missing"]

    tags = [tag for tag, _description in definitions]
    if len(tags) != len(set(tags)):
        reasons.append("minimax_h3_ref_manifest_duplicate_tag")
    limits = {"Picture": 9, "Video": 3, "Audio": 6}
    for kind, limit in limits.items():
        ordinals = [
            int(match.group(1))
            for tag in tags
            if (match := re.fullmatch(rf"<{kind} (\d+)>", tag))
        ]
        if ordinals and sorted(set(ordinals)) != list(range(1, max(ordinals) + 1)):
            reasons.append("minimax_h3_ref_manifest_nonconsecutive_tags")
        if ordinals and max(ordinals) > limit:
            reasons.append("minimax_h3_ref_manifest_limit_exceeded")
    if not any(tag.startswith(("<Picture ", "<Video ")) for tag in tags):
        reasons.append("minimax_h3_ref_manifest_audio_only")
    if any(len(re.findall(r"[A-Za-z0-9]{2,}", description)) < 3 for _tag, description in definitions):
        reasons.append("minimax_h3_ref_manifest_role_missing")

    defined = set(tags)
    mentioned = set(_minimax_h3_reference_tags(normalized))
    if mentioned - defined:
        reasons.append("minimax_h3_ref_manifest_undefined_tag")
    return list(dict.fromkeys(reasons))


def _normalize_hex_palette(*texts: str, max_colors: int = 16) -> list[str]:
    colors: list[str] = []
    for text in texts:
        for match in re.findall(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b", text or ""):
            raw = match.upper()
            if len(raw) == 4:
                raw = "#" + "".join(ch * 2 for ch in raw[1:])
            if raw not in colors:
                colors.append(raw)
            if len(colors) >= max_colors:
                return colors
    return colors


def _normalize_hex_text(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        raw = match.group(0).upper()
        if len(raw) == 4:
            return "#" + "".join(ch * 2 for ch in raw[1:])
        return raw

    return re.sub(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b", repl, text or "")


def _is_photo_style(base_prompt: str, render_style: str, style_description: dict[str, Any] | None = None) -> bool:
    text = f"{base_prompt} {render_style}"
    if re.search(r"\b(photo|photograph|photoreal|photorealistic|camera|lens|cinematic|portrait|macro|product photography)\b", text, re.IGNORECASE):
        return True
    if re.search(r"\b(illustration|poster|typography|logo|vector|painting|anime|comic|3d render|graphic design|isometric)\b", text, re.IGNORECASE):
        return False
    if isinstance(style_description, dict):
        if "photo" in style_description or str(style_description.get("medium", "")).lower() == "photograph":
            return True
        if "art_style" in style_description:
            return False
    return False


def _coerce_string(value: Any, fallback: str = "") -> str:
    text = _single_paragraph(str(value or ""))
    return _normalize_hex_text(text or fallback)


def _ideogram_style_description(render_style: str, exact_text: str) -> dict[str, Any]:
    style = _clean_control_text(render_style, 240)
    is_photo = _is_photo_style("", style)
    aesthetics = style or ("clean, legible, balanced" if exact_text.strip() else "natural, coherent")
    palette = _normalize_hex_palette(style, exact_text, max_colors=16)
    if is_photo:
        result = {
            "aesthetics": aesthetics,
            "lighting": "clear, controlled lighting that supports the requested subject",
            "photo": "realistic photographic detail with natural perspective",
            "medium": "photograph",
        }
        if palette:
            result["color_palette"] = palette
        return result
    result = {
        "aesthetics": aesthetics,
        "lighting": "clear, balanced lighting",
        "medium": "graphic_design" if exact_text.strip() else "illustration",
        "art_style": style or "clean, coherent visual design",
    }
    if palette:
        result["color_palette"] = palette
    return result


def _ideogram_prompt_object(
    base_prompt: str,
    aspect_ratio: str,
    render_style: str,
    exact_text: str,
    creativity_mode: str = "editorial",
    creative_strength: float = 0.6,
) -> dict[str, Any]:
    base = _normalize_hex_text(_clean_prompt_request_text(base_prompt) or "A clean, coherent image matching the user's request.")
    art_direction = _creative_art_direction_phrase(creativity_mode, creative_strength, "image")
    element_desc = f"{base} Composition is planned for a {aspect_ratio} frame."
    if art_direction:
        element_desc = f"{element_desc} {art_direction}"
    background = "supporting background that does not compete with the main subject"
    if art_direction:
        background = f"Background supports the requested subject with cohesive environment design. {art_direction}"
    elements: list[dict[str, Any]] = [
        {
            "type": "obj",
            "desc": element_desc,
        }
    ]
    if exact_text.strip():
        elements.append(
            {
                "type": "text",
                "text": exact_text.strip(),
                "desc": "Exact readable typography integrated into the requested layout.",
            }
        )
    return {
        "high_level_description": base,
        "style_description": _ideogram_style_description(render_style, exact_text),
        "compositional_deconstruction": {
            "background": background,
            "elements": elements,
        },
    }


def _normalize_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    bbox: list[int] = []
    for item in value:
        try:
            bbox.append(max(0, min(1000, int(round(float(item))))))
        except Exception:
            return None
    return bbox


def _normalize_ideogram_element(element: Any, exact_text: str = "") -> dict[str, Any] | None:
    if not isinstance(element, dict):
        desc = _single_paragraph(str(element or ""))
        return {"type": "obj", "desc": desc} if desc else None

    element_type = str(element.get("type", "")).strip().lower()
    text_value = _single_paragraph(str(element.get("text", "") or ""))
    if element_type not in {"obj", "text"}:
        element_type = "text" if text_value else "obj"

    desc = _coerce_string(
        element.get("desc")
        or element.get("description")
        or element.get("placement")
        or element.get("layout")
        or element.get("prompt"),
        "Main requested visual element.",
    )
    bbox = _normalize_bbox(element.get("bbox"))
    palette = _normalize_hex_palette(
        " ".join(str(item) for item in element.get("color_palette", []) if isinstance(element.get("color_palette"), list)),
        desc,
        max_colors=5,
    )

    result: dict[str, Any] = {"type": element_type}
    if bbox:
        result["bbox"] = bbox
    if element_type == "text":
        result["text"] = text_value or exact_text.strip()
    result["desc"] = desc
    if palette:
        result["color_palette"] = palette
    if element_type == "text" and not result.get("text"):
        return None
    return result


def _repair_ideogram_caption_object(
    value: Any,
    fallback_prompt: str,
    aspect_ratio: str,
    render_style: str,
    exact_text: str,
    creativity_mode: str = "editorial",
    creative_strength: float = 0.6,
) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    fallback = _ideogram_prompt_object(fallback_prompt, aspect_ratio, render_style, exact_text, creativity_mode, creative_strength)
    high_level = _coerce_string(source.get("high_level_description"), fallback["high_level_description"])

    source_style = source.get("style_description") if isinstance(source.get("style_description"), dict) else {}
    is_photo = _is_photo_style(high_level, render_style, source_style)
    palette = _normalize_hex_palette(
        " ".join(str(item) for item in source_style.get("color_palette", []) if isinstance(source_style.get("color_palette"), list)),
        render_style,
        high_level,
        exact_text,
        max_colors=16,
    )
    aesthetics = _coerce_string(source_style.get("aesthetics"), _ideogram_style_description(render_style, exact_text).get("aesthetics", "clean, coherent"))
    lighting = _coerce_string(source_style.get("lighting"), _ideogram_style_description(render_style, exact_text).get("lighting", "clear, balanced lighting"))
    if is_photo:
        style_description = {
            "aesthetics": aesthetics,
            "lighting": lighting,
            "photo": _coerce_string(source_style.get("photo") or render_style, "realistic photographic detail with natural perspective"),
            "medium": "photograph",
        }
    else:
        style_description = {
            "aesthetics": aesthetics,
            "lighting": lighting,
            "medium": _coerce_string(source_style.get("medium"), "graphic_design" if exact_text.strip() else "illustration"),
            "art_style": _coerce_string(source_style.get("art_style") or render_style, "clean, coherent visual design"),
        }
    if palette:
        style_description["color_palette"] = palette

    source_comp = source.get("compositional_deconstruction") if isinstance(source.get("compositional_deconstruction"), dict) else {}
    background = _coerce_string(
        source_comp.get("background"),
        fallback["compositional_deconstruction"]["background"],
    )
    elements: list[dict[str, Any]] = []
    for element in source_comp.get("elements", []) if isinstance(source_comp.get("elements"), list) else []:
        normalized = _normalize_ideogram_element(element, exact_text)
        if normalized:
            elements.append(normalized)
    if not elements:
        elements = fallback["compositional_deconstruction"]["elements"]
    if exact_text.strip() and not any(item.get("type") == "text" and item.get("text") == exact_text.strip() for item in elements):
        elements.append({"type": "text", "text": exact_text.strip(), "desc": "Exact readable typography integrated into the requested layout."})

    return {
        "high_level_description": high_level,
        "style_description": style_description,
        "compositional_deconstruction": {
            "background": background,
            "elements": elements,
        },
    }


def _ideogram_prompt_value_to_text(
    value: Any,
    fallback_prompt: str,
    aspect_ratio: str,
    render_style: str,
    exact_text: str,
    json_output: bool,
    max_chars: int,
    creativity_mode: str = "editorial",
    creative_strength: float = 0.6,
) -> str:
    if json_output:
        candidate = value
        if isinstance(candidate, str):
            parsed = _extract_json_object(candidate)
            candidate = parsed if parsed is not None else {}
        repaired = _repair_ideogram_caption_object(
            candidate,
            fallback_prompt,
            aspect_ratio,
            render_style,
            exact_text,
            creativity_mode,
            creative_strength,
        )
        text = _ideogram_json_dumps(repaired)
        return _strip_thinking(text).strip()
    return _prompt_field_to_text(
        value if not isinstance(value, dict) else _ideogram_prompt_text(fallback_prompt, aspect_ratio, render_style, exact_text),
        _ideogram_prompt_text(fallback_prompt, aspect_ratio, render_style, exact_text),
        max_chars,
    )


def _ideogram_prompt_text(
    base_prompt: str,
    aspect_ratio: str,
    render_style: str,
    exact_text: str,
) -> str:
    clauses = [_single_paragraph(base_prompt) or "A clean, coherent image matching the user's request."]
    clauses.append(f"Aspect ratio {aspect_ratio}.")
    style = _clean_control_text(render_style, 240)
    if style:
        clauses.append(f"Style: {style}.")
    if exact_text.strip():
        clauses.append(f"Render the exact readable text: {exact_text.strip()}.")
    clauses.append("Use deliberate layout, clear hierarchy, readable typography, and clean negative space.")
    return _single_paragraph(" ".join(clauses))


def _prompt_field_to_text(value: Any, fallback: str, max_chars: int) -> str:
    if isinstance(value, (dict, list)):
        text = _prompt_json_dumps(value)
    else:
        text = str(value or "")
    return _sanitize_prompt_text(text, fallback, max_chars)


def _apply_ltx2_controls(
    prompt: str,
    duration_seconds: float,
    style: str,
    include_audio: bool,
    audio_guidance: str = "",
) -> str:
    value = _single_paragraph(prompt)
    if not value:
        return ""
    clauses: list[str] = []
    style_value = _sanitize_prompt_text(style, "", 300, strip_thinking=True, strip_markdown=True).rstrip(".")
    if style_value:
        clauses.append(f"Style: {style_value}.")
    clauses.append(value)
    combined = _single_paragraph(" ".join(clauses))
    if include_audio:
        guidance = _audio_guidance_text(audio_guidance)
        if guidance and not _contains_audio_terms(combined):
            combined = f"{combined} Audio: {guidance}."
    return _single_paragraph(combined)


def _metadata_from_media(media_context: MediaContext | None) -> dict[str, Any]:
    if not isinstance(media_context, MediaContext):
        return {"source": "none"}
    return media_context.metadata.copy()


def _media_context_from_gemma_context(context: GemmaContext | None) -> MediaContext | None:
    if not isinstance(context, GemmaContext):
        return None
    if context.images is None:
        return None
    return MediaContext(images=context.images, source=context.source, metadata=context.media_metadata.copy())


def _gemma_context_from_media(
    user_prompt: str,
    media_context: MediaContext | None = None,
    visual_description: str = "",
) -> GemmaContext:
    metadata = _metadata_from_media(media_context)
    cleaned_visual_description = _clean_visual_description(visual_description or str(metadata.get("visual_description", "")))
    if cleaned_visual_description:
        metadata["visual_description"] = cleaned_visual_description
        metadata.setdefault("visual_description_source", "upstream_text")
    warnings = list(metadata.get("warnings", [])) if isinstance(metadata.get("warnings"), list) else []
    return GemmaContext(
        user_prompt=str(user_prompt or ""),
        images=media_context.images if isinstance(media_context, MediaContext) else None,
        source=str(metadata.get("source", "none")),
        media_metadata=metadata,
        visual_description=cleaned_visual_description,
        warnings=warnings,
    )


def _gemma_context_to_jsonable(context: GemmaContext | None) -> dict[str, Any]:
    if not isinstance(context, GemmaContext):
        return {"user_prompt": "", "source": "none", "media": {"source": "none"}, "warnings": []}
    metadata = context.media_metadata.copy()
    metadata["pixel_tensor_present"] = bool(context.images is not None)
    return {
        "user_prompt": _sanitize_prompt_text(context.user_prompt, "", 2000),
        "source": context.source,
        "visual_description": _clean_visual_description(context.visual_description),
        "media": metadata,
        "warnings": list(context.warnings),
    }


def _gemma_context_preview(context: GemmaContext | None) -> str:
    data = _gemma_context_to_jsonable(context)
    media = data.get("media", {}) if isinstance(data.get("media"), dict) else {}
    bits = [
        f"Prompt: {_sanitize_prompt_text(str(data.get('user_prompt', '')), '', 300)}",
        f"Media source: {media.get('source', data.get('source', 'none'))}",
    ]
    if media.get("duration_seconds"):
        bits.append(f"Duration: {float(media.get('duration_seconds')):.2f}s")
    if media.get("sampled_frame_count"):
        bits.append(f"Sampled frames: {media.get('sampled_frame_count')}")
    visual_description = _clean_visual_description(str(data.get("visual_description", "")))
    if visual_description:
        bits.append(f"Visual description: {_sanitize_prompt_text(visual_description, '', 500)}")
    warnings = data.get("warnings", [])
    if warnings:
        bits.append("Warnings: " + "; ".join(str(item) for item in warnings[:4]))
    return "\n".join(bits)


def _target_profile_config_to_dict(config: TargetProfileConfig | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(config, TargetProfileConfig):
        data = asdict(config)
    elif isinstance(config, dict):
        data = config.copy()
    else:
        data = {}
    return {
        "target_profile": _normalize_target_profile(str(data.get("target_profile", "ltx"))),
        "audio_mode": _normalize_audio_mode(str(data.get("audio_mode", "auto_scene_audio"))),
        "audio_guidance": _audio_guidance_text(str(data.get("audio_guidance", ""))),
        "target_duration_seconds": max(0.0, _safe_float(data.get("target_duration_seconds"), 0.0)),
        "ltx_style": _clean_control_text(str(data.get("ltx_style", "")), 240),
        "ltx_generation_mode": normalize_ltx25_generation_mode(data.get("ltx_generation_mode", "auto")),
        "ltx_long_horizon_mode": normalize_ltx25_long_horizon_mode(
            data.get("ltx_long_horizon_mode", "off")
        ),
        "ltx_camera_capability": normalize_ltx25_camera_capability(
            data.get("ltx_camera_capability", "stable")
        ),
        "ideogram_aspect_ratio": str(data.get("ideogram_aspect_ratio", "1:1") or "1:1"),
        "ideogram_render_style": _clean_control_text(str(data.get("ideogram_render_style", "")), 240),
        "ideogram_exact_text": _sanitize_prompt_text(
            str(data.get("ideogram_exact_text", "")),
            "",
            1000,
            strip_thinking=True,
            strip_markdown=True,
        ),
        "ideogram_json_output": bool(data.get("ideogram_json_output", True)),
        "negative_prompt_mode": _normalize_negative_prompt_mode(str(data.get("negative_prompt_mode", "auto"))),
        "negative_prompt_guidance": _negative_prompt_guidance_text(str(data.get("negative_prompt_guidance", ""))),
        "minimax_h3_mode": _normalize_minimax_h3_mode(str(data.get("minimax_h3_mode", "t2va"))),
        "minimax_h3_shot_count": _normalize_minimax_h3_shot_count(data.get("minimax_h3_shot_count", "auto")),
        "minimax_h3_dialogue_mode": _normalize_minimax_h3_dialogue_mode(
            data.get("minimax_h3_dialogue_mode", "auto")
        ),
        "minimax_h3_dialogue_line_count": _normalize_minimax_h3_dialogue_line_count(
            data.get("minimax_h3_dialogue_line_count", 2)
        ),
        "minimax_h3_dialogue_guidance": _minimax_h3_dialogue_guidance_text(
            data.get("minimax_h3_dialogue_guidance", "")
        ),
    }


def _make_target_profile_config(
    target_profile: str = "ltx",
    audio_mode: str = "auto_scene_audio",
    audio_guidance: str = "",
    target_duration_seconds: float = 0.0,
    ltx_style: str = "",
    ideogram_aspect_ratio: str = "1:1",
    ideogram_render_style: str = "",
    ideogram_exact_text: str = "",
    ideogram_json_output: bool = True,
    negative_prompt_mode: str = "auto",
    negative_prompt_guidance: str = "",
    minimax_h3_mode: str = "t2va",
    minimax_h3_shot_count: str = "auto",
    minimax_h3_dialogue_mode: str = "auto",
    minimax_h3_dialogue_line_count: int = 2,
    minimax_h3_dialogue_guidance: str = "",
    ltx_generation_mode: str = "auto",
    ltx_long_horizon_mode: str = "off",
    ltx_camera_capability: str = "stable",
) -> TargetProfileConfig:
    data = _target_profile_config_to_dict(
        {
            "target_profile": target_profile,
            "audio_mode": audio_mode,
            "audio_guidance": audio_guidance,
            "target_duration_seconds": target_duration_seconds,
            "ltx_style": ltx_style,
            "ideogram_aspect_ratio": ideogram_aspect_ratio,
            "ideogram_render_style": ideogram_render_style,
            "ideogram_exact_text": ideogram_exact_text,
            "ideogram_json_output": ideogram_json_output,
            "negative_prompt_mode": negative_prompt_mode,
            "negative_prompt_guidance": negative_prompt_guidance,
            "minimax_h3_mode": minimax_h3_mode,
            "minimax_h3_shot_count": minimax_h3_shot_count,
            "minimax_h3_dialogue_mode": minimax_h3_dialogue_mode,
            "minimax_h3_dialogue_line_count": minimax_h3_dialogue_line_count,
            "minimax_h3_dialogue_guidance": minimax_h3_dialogue_guidance,
            "ltx_generation_mode": ltx_generation_mode,
            "ltx_long_horizon_mode": ltx_long_horizon_mode,
            "ltx_camera_capability": ltx_camera_capability,
        }
    )
    return TargetProfileConfig(**data)


def _normalize_thinking_mode(thinking_mode: str) -> str:
    value = (thinking_mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"auto", "on", "off"}:
        return value
    return "auto"


def _effective_target_thinking_mode(thinking_mode: str, target_profile: str) -> str:
    """Use answer-only checkpoint mode for H3 unless thinking was explicit."""

    mode = _normalize_thinking_mode(thinking_mode)
    if _normalize_target_profile(target_profile) == "minimax_h3" and mode == "auto":
        return "off"
    return mode


def _thinking_instruction(thinking_mode: str) -> str:
    mode = _normalize_thinking_mode(thinking_mode)
    if mode == "off":
        return "Do not include chain-of-thought or hidden reasoning; return only the final JSON."
    if mode == "on":
        return "You may write concise reasoning in a <think>...</think> block before the final JSON. The final JSON must still be complete and directly usable."
    return "Use concise private reasoning only if it improves the prompt plan; put any reasoning in <think>...</think> before the final JSON."


def _split_reasoning_and_answer(raw_output: str) -> tuple[str, str]:
    text = str(raw_output or "")
    delimited_answer, has_final_boundary = _extract_delimited_final_payload(text)
    reasoning_parts: list[str] = []
    for match in re.finditer(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL):
        part = _sanitize_prompt_text(match.group(1), "", 4000, strip_thinking=False, strip_markdown=True)
        if part:
            reasoning_parts.append(part)
    for match in re.finditer(r"<\|thought\|>(.*?)(?=<\|(?:end_thought|final_json|answer|final)\|>)", text, flags=re.IGNORECASE | re.DOTALL):
        part = _sanitize_prompt_text(match.group(1), "", 4000, strip_thinking=False, strip_markdown=True)
        if part:
            reasoning_parts.append(part)
    for match in re.finditer(r"<\|channel\|>thought(.*?)(?=<\|channel\|>|$)", text, flags=re.IGNORECASE | re.DOTALL):
        part = _sanitize_prompt_text(match.group(1), "", 4000, strip_thinking=False, strip_markdown=True)
        if part:
            reasoning_parts.append(part)
    if has_final_boundary:
        return (_single_paragraph(" ".join(reasoning_parts)), delimited_answer.strip())

    # Some chat templates decode the channel marker itself as the bare first
    # line ``thought``/``analysis`` after special tokens are removed.  Never
    # promote that unfinished planning trace to a generator prompt.  A plain
    # final/answer heading or the first JSON object is still a safe boundary.
    plain_thought = re.match(
        r"\A\s*(?:thought|analysis)\s*(?:\r?\n|:)\s*",
        text,
        flags=re.IGNORECASE,
    )
    if plain_thought:
        remainder = text[plain_thought.end() :]
        plain_final = re.search(
            r"(?im)^\s*(?:final(?:_json)?|answer)\s*:?\s*$",
            remainder,
        )
        first_json = remainder.find("{")
        if plain_final and (first_json < 0 or plain_final.start() < first_json):
            reasoning_source = remainder[: plain_final.start()]
            answer = remainder[plain_final.end() :].strip()
        elif first_json >= 0:
            reasoning_source = remainder[:first_json]
            answer = remainder[first_json:].strip()
        else:
            reasoning_source = remainder
            answer = ""
        reasoning = _sanitize_prompt_text(
            reasoning_source,
            "",
            4000,
            strip_thinking=False,
            strip_markdown=True,
        )
        if reasoning:
            reasoning_parts.append(reasoning)
        return (_single_paragraph(" ".join(reasoning_parts)), answer)

    if (
        re.match(r"\A\s*<think(?:>|\s)", text, flags=re.IGNORECASE)
        and not re.search(r"</think>", text, flags=re.IGNORECASE)
    ) or (
        re.match(r"\A\s*<\|thought\|>", text, flags=re.IGNORECASE)
        and not re.search(
            r"<\|(?:end_thought|final_json|answer|final)\|>",
            text,
            flags=re.IGNORECASE,
        )
    ):
        reasoning = _sanitize_prompt_text(
            re.sub(r"\A\s*(?:<think[^>]*>|<\|thought\|>)", "", text, flags=re.IGNORECASE),
            "",
            4000,
            strip_thinking=False,
            strip_markdown=True,
        )
        return (_single_paragraph(" ".join([*reasoning_parts, reasoning])), "")

    answer = _strip_thinking(text)
    first_json = answer.find("{")
    if first_json > 0:
        preface = _sanitize_prompt_text(answer[:first_json], "", 2000, strip_thinking=False, strip_markdown=True)
        if preface:
            reasoning_parts.append(preface)
        answer = answer[first_json:]
    return (_single_paragraph(" ".join(reasoning_parts)), answer.strip())


def _runtime_supports_pixels(config: RuntimeConfig) -> bool:
    return bool(config.status.get("supports_pixels")) if isinstance(config, RuntimeConfig) else False


def _media_policy_notes(config: RuntimeConfig, context: GemmaContext | None) -> list[str]:
    if not isinstance(context, GemmaContext) or context.images is None:
        return []
    if _runtime_supports_pixels(config):
        return ["Media pixels will be passed to the active backend."]
    return ["Active backend is metadata-only for image/video; sampled pixels were not sent to DiffusionGemma runtime."]


def _image_batch_to_pil(images: Any) -> list[Any]:
    if images is None:
        return []
    try:
        import numpy as np
        from PIL import Image
        import torch

        tensor = images.detach().cpu() if isinstance(images, torch.Tensor) else images
        if hasattr(tensor, "numpy"):
            arr = tensor.numpy()
        else:
            arr = np.asarray(tensor)
        if arr.ndim == 3:
            arr = arr[None, ...]
        pil_images = []
        for frame in arr:
            frame = np.clip(frame[..., :3] * 255.0, 0, 255).astype("uint8")
            pil_images.append(Image.fromarray(frame))
        return pil_images
    except Exception:
        return []


def _image_batch_hwc_shape(images: Any) -> tuple[int, int, int] | None:
    shape = getattr(images, "shape", None)
    if shape is None:
        return None
    try:
        dims = tuple(int(item) for item in shape)
    except Exception:
        return None
    if len(dims) >= 4:
        return dims[-3], dims[-2], dims[-1]
    if len(dims) == 3:
        return dims[0], dims[1], dims[2]
    return None


def _fit_reference_image_to_video_frames(reference_image: Any, video_frames: Any) -> tuple[Any, dict[str, Any]]:
    report: dict[str, Any] = {"reference_image_backend_attached": False}
    try:
        first_reference = reference_image[:1] if len(getattr(reference_image, "shape", ())) >= 4 else reference_image
    except Exception:
        first_reference = reference_image

    reference_shape = _image_batch_hwc_shape(first_reference)
    video_shape = _image_batch_hwc_shape(video_frames)
    if reference_shape:
        report["reference_image_original_height"] = reference_shape[0]
        report["reference_image_original_width"] = reference_shape[1]
    if video_shape:
        report["reference_image_backend_height"] = video_shape[0]
        report["reference_image_backend_width"] = video_shape[1]
    if not reference_shape or not video_shape:
        return first_reference, report
    if reference_shape[:2] == video_shape[:2] and reference_shape[2] == video_shape[2]:
        try:
            import torch

            if torch.is_tensor(first_reference) and torch.is_tensor(video_frames):
                aligned = first_reference
                if aligned.ndim == 3 and video_frames.ndim == 4:
                    aligned = aligned.unsqueeze(0)
                first_reference = aligned.to(device=video_frames.device, dtype=video_frames.dtype)
        except Exception:
            pass
        report["reference_image_resized_for_video_control"] = False
        return first_reference, report

    try:
        import torch
        import torch.nn.functional as F

        if not torch.is_tensor(first_reference) or not torch.is_tensor(video_frames):
            report["reference_image_resize_error"] = "reference image or video frames are not torch tensors"
            return first_reference, report

        target_h, target_w, target_c = video_shape
        ref = first_reference
        if ref.ndim == 3:
            ref = ref.unsqueeze(0)
        ref = ref[:1].to(device=video_frames.device, dtype=video_frames.dtype)
        if int(ref.shape[-1]) > target_c:
            ref = ref[..., :target_c]
        elif int(ref.shape[-1]) < target_c:
            channel_pad = target_c - int(ref.shape[-1])
            ref = F.pad(ref, (0, channel_pad, 0, 0, 0, 0), mode="constant", value=0.0)

        nchw = ref.permute(0, 3, 1, 2).float()
        src_h = max(1, int(nchw.shape[-2]))
        src_w = max(1, int(nchw.shape[-1]))
        scale = min(float(target_h) / float(src_h), float(target_w) / float(src_w))
        resized_h = max(1, min(int(target_h), int(round(src_h * scale))))
        resized_w = max(1, min(int(target_w), int(round(src_w * scale))))
        resized = F.interpolate(nchw, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
        pad_top = max(0, (int(target_h) - resized_h) // 2)
        pad_bottom = max(0, int(target_h) - resized_h - pad_top)
        pad_left = max(0, (int(target_w) - resized_w) // 2)
        pad_right = max(0, int(target_w) - resized_w - pad_left)
        fitted = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
        fitted = fitted[..., : int(target_h), : int(target_w)]
        prepared = fitted.permute(0, 2, 3, 1).to(dtype=video_frames.dtype)
        report["reference_image_resized_for_video_control"] = True
        report["reference_image_resize_mode"] = "aspect_fit_replicate_pad"
        return prepared, report
    except Exception as exc:
        report["reference_image_resize_error"] = str(exc)
        return first_reference, report


_MINIMAX_H3_CAMERA_SENTENCE_CONTRACT = (
    "For every shot, immediately after its [Shot N] marker and, for Shot 2 onward, the required cut phrase, write "
    "one explicit camera-behavior sentence. "
    "Honor explicit user and reference camera choreography exactly. When direction is absent, Faithful mode invents no "
    "new choreography, while Editorial, Cinematic, Concept-art, and Wild modes may author shot-appropriate movement "
    "proportional to creative strength that supports performance, rhythm, spatial legibility, and the available shot time; "
    "do not default every unspecified shot to a locked hold. H3 may use a hold, push or pull, pan or tilt, track or crane, orbit or arc, roll or "
    "rotation, spin or swirl, sweeping or whip movement, controlled shake, pronounced or layered parallax, or a coherent "
    "compound/multi-axis path. These are permitted tools, not compulsory ingredients. For a moving shot, state the motion "
    "type, support or path, direction, meaningful amplitude and speed, subject relationship, and intended ending framing, "
    "and keep the motion physically continuous and achievable before the next cut. Tie pronounced parallax to named depth "
    "layers and actual lateral, depth, or orbital camera travel. For a compound route, combine compatible simultaneous axes "
    "or order its phases explicitly; a purposeful reversal is allowed when its timing and endpoint are clear. Never claim "
    "that the camera is locked and traveling at the same instant, teleport the camera, or sacrifice subject and reference "
    "readability for the entire shot. An incoming cut, shot size, angle, subject framing, or the word handheld "
    "alone is not that shot's camera behavior. This declaration is structural, not decorative: never omit it to save words "
    "in a dense multi-shot timeline. Use the exact fallback sentence 'The camera remains locked-off in a stable view.' only "
    "when a static composition genuinely serves the shot and no requested, reference-required, or dynamically useful path applies."
)


_MINIMAX_H3_REFINABLE_REASONS = {
    "minimax_h3_prompt_truncated",
    "minimax_h3_extra_top_level_field",
    "minimax_h3_non_diegetic_music_invalid",
    "minimax_h3_prompt_contains_grounding_role_annotation",
    "minimax_h3_noncanonical_time_range",
    "minimax_h3_shot_syntax_invalid",
    "minimax_h3_unlabeled_cut",
    "minimax_h3_shot_numbering_invalid",
    "minimax_h3_shot1_timestamp_invalid",
    "minimax_h3_empty_visual_timeline",
    "minimax_h3_cut_timestamp_invalid",
    "minimax_h3_cut_timestamp_out_of_range",
    "minimax_h3_requested_visual_medium_missing",
    "minimax_h3_requested_visual_exclusion_missing",
    "minimax_h3_requested_visual_medium_conflict",
    "minimax_h3_requested_style_missing",
    "minimax_h3_requested_shot_count_mismatch",
    "minimax_h3_shot_camera_unspecified",
    "minimax_h3_requested_final_hold_missing",
    "minimax_h3_unresolved_final_state",
    "minimax_h3_battle_result_missing",
    "minimax_h3_cut_transition_invalid",
    "minimax_h3_orphaned_sound_event",
    "minimax_h3_dialogue_outside_timeline",
    "minimax_h3_dialogue_tags_invalid",
    "minimax_h3_dialogue_speaker_invalid",
    "minimax_h3_dialogue_count_mismatch",
    "minimax_h3_dialogue_forbidden",
    "minimax_h3_ref_sections_invalid",
    "minimax_h3_ref_field_order_invalid",
    "minimax_h3_ref_field_spacing_invalid",
    "minimax_h3_ref_missing_subject_definitions",
    "minimax_h3_ref_missing_summary",
    "minimax_h3_ref_missing_retention_analysis",
    "minimax_h3_ref_missing_detailed_description",
    "minimax_h3_ref_missing_overall_soundscape",
    "minimax_h3_ref_missing_non_diegetic_music",
    "minimax_h3_ref_missing_asset_tag",
    "minimax_h3_ref_undefined_tag",
    "minimax_h3_ref_subject_definition_invalid",
    "minimax_h3_ref_subject_count_mismatch",
    "minimax_h3_ref_subject_usage_invalid",
    "minimax_h3_ref_summary_invalid",
    "minimax_h3_ref_retention_invalid",
    "minimax_h3_ref_style_opening_missing",
    "minimax_h3_ref_detailed_description_short",
}

_MINIMAX_H3_CATASTROPHIC_STRUCTURE_REASONS = {
    "minimax_h3_prompt_truncated",
    "minimax_h3_extra_top_level_field",
    "minimax_h3_opening_invalid",
    "minimax_h3_field_order_invalid",
    "minimax_h3_field_spacing_invalid",
    "minimax_h3_missing_integrated_multimodal_description",
    "minimax_h3_missing_overall_soundscape",
    "minimax_h3_missing_non_diegetic_music",
    "minimax_h3_empty_overall_soundscape",
    "minimax_h3_empty_non_diegetic_music",
    "minimax_h3_ref_sections_invalid",
    "minimax_h3_ref_field_order_invalid",
    "minimax_h3_ref_field_spacing_invalid",
    "minimax_h3_ref_missing_subject_definitions",
    "minimax_h3_ref_missing_summary",
    "minimax_h3_ref_missing_retention_analysis",
    "minimax_h3_ref_missing_detailed_description",
    "minimax_h3_ref_missing_overall_soundscape",
    "minimax_h3_ref_missing_non_diegetic_music",
}


def _minimax_h3_refinement_candidate_rank(
    candidate_prompt: str,
    reasons: list[str],
    parse_valid: bool,
    salvage_warning: str = "",
) -> tuple[int, int, int, int]:
    normalized_reasons = list(dict.fromkeys(str(reason) for reason in reasons if str(reason)))
    return (
        int(not parse_valid or bool(str(salvage_warning or "").strip())),
        int(not str(candidate_prompt or "").strip()),
        sum(reason in _MINIMAX_H3_CATASTROPHIC_STRUCTURE_REASONS for reason in normalized_reasons),
        len(normalized_reasons),
    )


def _build_minimax_h3_refinement_prompt(
    user_prompt: str,
    candidate_prompt: str,
    duration_seconds: float,
    audio_mode: str,
    reasons: list[str],
    minimax_h3_mode: str = "t2va",
    reference_manifest: str = "",
    strict_retry: bool = False,
    minimax_h3_shot_count: str = "auto",
    native_prompt_only: bool = False,
    minimax_h3_dialogue_mode: str = "auto",
    minimax_h3_dialogue_line_count: int = 2,
    minimax_h3_dialogue_guidance: str = "",
    minimax_h3_expected_subject_count: int = 0,
) -> str:
    minimax_h3_mode = _normalize_minimax_h3_mode(minimax_h3_mode)
    reference_manifest = _strip_grounding_role_annotations(
        _normalize_minimax_h3_reference_manifest(reference_manifest)
    )
    requested_medium = _minimax_h3_requested_visual_medium(user_prompt)
    style_locks = _minimax_h3_requested_style_locks(user_prompt)
    configured_shot_count = _normalize_minimax_h3_shot_count(minimax_h3_shot_count)
    requested_shot_count = _minimax_h3_effective_shot_count(user_prompt, configured_shot_count)
    parsed_candidate_shot_count = len(
        re.findall(r"\[Shot \d+\]", candidate_prompt, flags=re.IGNORECASE)
    )
    (
        minimum_detail_words,
        target_detail_words_low,
        target_detail_words_high,
    ) = _minimax_h3_ref_detail_word_budget(
        duration_seconds,
        requested_shot_count or parsed_candidate_shot_count or 1,
    )
    shot_count_source = (
        "target_profile"
        if configured_shot_count != "auto"
        else "user_prompt"
        if requested_shot_count
        else "model_auto"
    )
    dialogue_mode = _normalize_minimax_h3_dialogue_mode(minimax_h3_dialogue_mode)
    dialogue_line_count = _normalize_minimax_h3_dialogue_line_count(minimax_h3_dialogue_line_count)
    dialogue_guidance = _minimax_h3_dialogue_guidance_text(minimax_h3_dialogue_guidance)
    dialogue_instruction = _minimax_h3_dialogue_instruction(
        dialogue_mode,
        dialogue_line_count,
        dialogue_guidance,
        minimax_h3_mode,
    )
    expected_subject_count = _normalize_minimax_h3_expected_subject_count(
        minimax_h3_expected_subject_count
    )
    subject_count_instruction = _minimax_h3_subject_count_instruction(
        expected_subject_count
    )
    timeline_bound_instruction = _minimax_h3_timeline_bound_instruction(
        requested_shot_count,
        duration_seconds,
    )
    corrections: list[str] = []
    if requested_medium and minimax_h3_mode == "ref2va":
        corrections.append(
            f"State the immutable {requested_medium.replace('_', ' ')} medium in the one-or-two-sentence style opening before [Shot 1], "
            "and keep that medium across every cut."
        )
    elif requested_medium:
        corrections.append(_minimax_h3_visual_medium_instruction(requested_medium))
    if style_locks:
        style_location = "in the style opening before [Shot 1]" if minimax_h3_mode == "ref2va" else "near the start of Shot 1"
        corrections.append(f"Copy these exact style phrases {style_location}: {'; '.join(style_locks)}.")
    if configured_shot_count != "auto":
        corrections.append(_minimax_h3_shot_count_instruction(configured_shot_count))
    elif requested_shot_count:
        corrections.append(f"Use exactly {requested_shot_count} consecutively numbered shots.")
    if timeline_bound_instruction:
        corrections.append(timeline_bound_instruction)
    corrections.append(dialogue_instruction)
    corrections.append(_MINIMAX_H3_CAMERA_SENTENCE_CONTRACT)
    if minimax_h3_mode == "ref2va":
        corrections.append(subject_count_instruction)
        if any(
            reason in reasons
            for reason in (
                "minimax_h3_ref_missing_detailed_description",
                "minimax_h3_ref_style_opening_missing",
                "minimax_h3_ref_detailed_description_short",
            )
        ):
            required_shots = (
                f"exactly {requested_shot_count} consecutive [Shot N] blocks"
                if requested_shot_count
                else "a complete consecutively numbered [Shot N] timeline"
            )
            corrections.append(
                "Rebuild detailed_description rather than copying a blank or malformed section. Start with one or two concrete medium/style "
                "sentences containing at least five words before [Shot 1], then write "
                f"{required_shots}. The complete detailed_description must contain at least {minimum_detail_words} words; target about "
                f"{target_detail_words_low} to {target_detail_words_high} words for this duration and shot count. Every shot must contain visible "
                "action plus its own explicit camera-behavior sentence, and no placeholder or omitted-shot shorthand is allowed."
                + (
                    " For a dense timeline, keep each shot concise—about 20 to 35 words—so every required shot and the resolved final state fit in the response."
                    if (requested_shot_count or parsed_candidate_shot_count) >= 4
                    else " Keep the short shot physically plausible; do not pad it with extra cuts, actions, or camera moves merely to reach the word budget."
                )
            )
        if "minimax_h3_ref_subject_definition_invalid" in reasons:
            corrections.append(
                "Rebuild subject_definitions. Every used <Subject N> must have exactly one consecutive, colon-delimited definition containing "
                "at least four descriptive words and citing its declared <Picture N> or <Video N> source where applicable; do not leave a bare "
                "Subject tag, and do not use any Subject tag that lacks a definition."
            )
    if "minimax_h3_dialogue_count_mismatch" in reasons:
        corrections.append(
            f"The rewritten shot timeline must contain exactly {dialogue_line_count} complete <d>[Language] ...</d> blocks. "
            + (
                "Place them within the existing shot count and repeat the speaking <Subject N> with its stable numbered (S1), (S2), ... cue in the same shot before every block; never output literal (Sx)."
                if minimax_h3_mode == "ref2va"
                else "Place them within the existing shot count and give every block a stable numbered speaker cue such as (S1) or (S2) in the same shot before it; never output literal (Sx) or a role/name suffix inside the parentheses."
            )
        )
    if "minimax_h3_dialogue_forbidden" in reasons:
        corrections.append("Remove all dialogue tags and intelligible spoken, narrated, voiceover, conversational, or sung words.")
    if any(
        reason in reasons
        for reason in (
            "minimax_h3_dialogue_outside_timeline",
            "minimax_h3_dialogue_tags_invalid",
            "minimax_h3_dialogue_speaker_invalid",
        )
    ):
        corrections.append(
            (
                "Keep every complete dialogue block inside the shot timeline. In Ref2VA, write the speaking semantic label and stable numbered ID together as "
                "'<Subject N> (S1)' in the same shot before every <d>[Language] spoken words</d> block; repeat that pair for every line, keep the mapping "
                "one-to-one, and never output literal (Sx) or use a bare numbered cue after multiple Subject tags. Never place dialogue in analysis, soundscape, or music sections."
                if minimax_h3_mode == "ref2va"
                else "Keep every complete dialogue block inside the shot timeline, with a stable numbered cue such as (S1) in the same shot before "
                "<d>[Language] spoken words</d>; never output literal (Sx), put a role/name suffix inside the parentheses, or place dialogue in analysis, soundscape, or music sections."
            )
        )
    if "minimax_h3_shot_camera_unspecified" in reasons:
        corrections.append(
            "Repair every action-only or framing-only shot by inserting its explicit camera-behavior sentence. Preserve requested "
            "or reference-authored choreography, and infer a physically coherent H3 camera path when it supports the performance, "
            "rhythm, or spatial reveal. Orbiting, swirling, sweeping or whip movement, pronounced parallax, and coherent compound "
            "paths are valid H3 choices. Use the exact locked-off fallback only when a static composition genuinely serves the shot."
        )
    if "minimax_h3_requested_final_hold_missing" in reasons:
        corrections.append("Put the requested freeze or hold in the final shot and name the held composition.")
    if "minimax_h3_requested_visual_exclusion_missing" in reasons:
        corrections.append(
            "State the required medium exclusions explicitly in the style opening: for 2D animation write exactly "
            "'no live action, photorealism, or 3D CGI'; for other animation write 'no live action or photorealism'."
        )
    if any(
        reason in reasons
        for reason in (
            "minimax_h3_noncanonical_time_range",
            "minimax_h3_shot1_timestamp_invalid",
            "minimax_h3_cut_timestamp_invalid",
            "minimax_h3_cut_timestamp_out_of_range",
            "minimax_h3_cut_transition_invalid",
        )
    ):
        corrections.append(
            "Shot 1 must begin exactly with '[Shot 1]' and no timestamp. Never write start-end ranges. Begin every later shot exactly "
            "with one start instant: '[Shot N] At MM:SS.mmm, the camera cuts to ...'; put camera motion after that cut phrase."
        )
    if "minimax_h3_cut_timestamp_out_of_range" in reasons:
        end_label = _minimax_h3_timestamp_label(duration_seconds)
        corrections.append(
            f"For this exact runtime, every later cut start must be strictly earlier than {end_label}. {end_label} is the exclusive video end, "
            "not a valid start for the last shot. Replan the cut starts so the final shot has meaningful dwell time for its complete action, sound, and dialogue."
        )
    if "minimax_h3_battle_result_missing" in reasons:
        corrections.append(
            "Repair the battle causality: establish both opponents and their screen positions; every attack must name attacker, target, "
            "contact or counter, the target's visible physical reaction, and the resulting positions carried into the next shot. Use an "
            "unambiguous reaction such as 'is knocked backward,' 'blocks and slides two steps,' 'dodges and lands,' 'staggers,' or 'falls'; "
            "a glow, charge, lunge, shockwave, impact sound, or mid-clash pose without the opponent's reaction is not a result."
        )
    if "minimax_h3_unresolved_final_state" in reasons:
        corrections.append(
            "Replace the unresolved ending with both relevant subjects in an observable post-action state; do not end mid-air, charging, "
            "approaching, beginning an attack, mid-clash, or disappearing unless the brief explicitly requests that cliffhanger."
        )
    if "minimax_h3_orphaned_sound_event" in reasons:
        corrections.append(
            "Rebuild overall_soundscape from the final visible timeline after rewriting it. Delete any fire blast, gunshot, explosion, or "
            "other event sound whose visible cause no longer exists; keep requested ambience and synchronized action sounds only."
        )
    if "minimax_h3_non_diegetic_music_invalid" in reasons:
        corrections.append(
            "Replace the malformed '/A' music sentinel with exactly 'N/A' when no audience-only score is present, or with complete "
            "instrumentation, tempo/rhythm, and dynamic-development prose when music is requested."
        )
    if "minimax_h3_prompt_contains_grounding_role_annotation" in reasons:
        corrections.append(
            "Remove every complete or dangling [dg:...] host grounding-role annotation from the rewritten native prompt while preserving "
            "the referenced asset labels and their natural-language roles."
        )
    if "minimax_h3_ref_missing_asset_tag" in reasons:
        corrections.append(
            "Keep every declared <Picture N>, <Video N>, and <Audio N> tag in subject_definitions. For identity-, appearance-, environment-, "
            "or style-only assets, cite the asset inside the matching <Subject N> definition instead of creating a redundant standalone asset role."
        )
    if "minimax_h3_ref_subject_usage_invalid" in reasons:
        corrections.append(
            "Every defined <Subject N> must have its own retention_analysis line and must appear by that exact Subject tag in the shot timeline. "
            "Never substitute its source <Picture N> or <Video N> tag for the Subject tag; at first appearance, write '<Subject N> (derived from "
            "<Picture N>)' when provenance must also be explicit."
        )
    if "minimax_h3_ref_contact_sheet_subject_source_invalid" in reasons:
        corrections.append(
            "Repair contact-sheet provenance. For the multi-entity policy, <Subject 1> must cite only <Picture 1>; create one consecutive "
            "secondary <Subject N> per distinct requested supporting entity and cite <Picture 2> in every secondary definition. Multiple "
            "views of one entity remain one Subject. For the same-subject policy, <Subject 1> must cite both <Picture 1> and <Picture 2>. "
            "A standalone <Picture 2> row does not satisfy these bindings."
        )
    if "minimax_h3_ref_contact_sheet_layout_transfer_invalid" in reasons:
        corrections.append(
            "Remove all contact-sheet, collage, grid, cell, panel-seam, border, label, thumbnail, and sheet-layout content from the generated "
            "shot timeline. Extract only the requested subjects or objects into the intended scene."
        )
    # Keep the retry target-isolated too. Showing inactive LTX/Ideogram fields
    # here made a clean H3-only first pass fall back into the old all-target
    # planning behavior during structural refinement.
    output_shape = {
        "minimax_h3_prompt": "REWRITTEN THREE-FIELD MINIMAX H3 PROMPT",
    }
    if minimax_h3_mode == "ref2va":
        output_shape["minimax_h3_prompt"] = "REWRITTEN SIX-SECTION MINIMAX H3 REF2VA PROMPT"
        retry_instruction = ""
        if strict_retry:
            retry_instruction = (
                "This is the one bounded retry after an incomplete rewrite. Return no reasoning. Preserve sections that already validate, but "
                "replace every blank or invalid listed section instead of copying its defect. Emit each exact heading once and end every heading line with its literal colon: subject_definitions:, "
                "summary:, retention_analysis:, detailed_description:, overall_soundscape:, and non_diegetic_music:. Never write "
                "detail_description, leave a section blank, "
                "drop a shot, stop mid-sentence, or place any content after the final JSON boundary.\n\n"
            )
        output_instruction = (
            "Return only the complete rewritten native MiniMax H3 prompt. Begin directly with subject_definitions: "
            "and complete the non_diegetic_music section before any separately specified provenance footer. Do not return JSON, quotes, Markdown, commentary, or final-answer markers."
            if native_prompt_only
            else f"Return only {FINAL_JSON_OPEN}{_json_dumps(output_shape)}{FINAL_JSON_CLOSE} with the placeholder replaced by the complete rewritten prompt."
        )
        return (
            "You are the final quality-control rewrite pass for a MiniMax H3 Ref2VA full-reference prompt. Rewrite the candidate, preserving "
            "its good concrete details, while fixing every listed failure. Do not explain or critique. Treat both the user brief and reference "
            "manifest as binding; never invent, omit, renumber, merge, or cross-wire an asset label.\n\n"
            f"User brief: {user_prompt.strip()}\n"
            f"Exact duration: {duration_seconds:.3f} seconds\n"
            f"Audio mode: {_normalize_audio_mode(audio_mode)}\n"
            f"Reference manifest:\n{reference_manifest}\n"
            f"Requested visual medium: {requested_medium or 'unspecified'}\n"
            f"Requested style locks: {'; '.join(style_locks) if style_locks else 'none'}\n"
            f"Shot-count selector: {configured_shot_count}; effective required count: {requested_shot_count or 'automatic'}; source: {shot_count_source}\n"
            f"Dialogue mode: {dialogue_mode}; required line count: {dialogue_line_count if dialogue_mode == 'required' else 'not enforced'}; guidance: {dialogue_guidance or 'none'}\n"
            f"Expected semantic Subject count: {expected_subject_count or 'Auto'}\n"
            f"Failures to repair: {_json_dumps(reasons)}\n"
            f"Required corrections: {' '.join(item for item in corrections if item)}\n\n"
            "The rewritten minimax_h3_prompt must contain exactly these literal heading lines in order, each ending with its colon: "
            "subject_definitions:, summary:, retention_analysis:, detailed_description:, overall_soundscape:, and non_diegetic_music:. "
            "Put one blank line between sections. In subject_definitions, map the "
            "manifest assets to consecutive semantic <Subject N> labels or independently defined whole-asset roles. Write every definition as "
            "'<Subject N>: description' (a colon immediately follows the tag). Begin summary with the exact "
            "official bracketed task types. Give every independently defined visible label one fixed visual retention marker and every independently "
            "defined Audio label one fixed audio marker. Once an element is assigned <Subject N>, use that exact Subject tag in the shot timeline and "
            "use its <Picture N> or <Video N> source only as provenance, never as a replacement label. Put one or two concrete medium/style sentences before [Shot 1], then a detailed shot timeline "
            "with stable reference labels, consecutive shots, increasing cut instants below the duration, the required explicit camera-behavior "
            "sentence in every shot, causal continuity, "
            "synchronized sound, and a resolved final state. Do not use integrated_multimodal_description or a negative section.\n\n"
            f"Semantic Subject-count contract: {subject_count_instruction}\n\n"
            f"{retry_instruction}"
            f"Candidate prompt: {_json_dumps(candidate_prompt)}\n\n"
            f"{output_instruction}"
        )
    output_instruction = (
        "Return only the complete rewritten native MiniMax H3 prompt. Begin directly with "
        "integrated_multimodal_description: [Shot 1] and complete the non_diegetic_music field before any separately specified provenance footer. Do not return JSON, "
        "quotes, Markdown, commentary, or final-answer markers."
        if native_prompt_only
        else f"Return only {FINAL_JSON_OPEN}{_json_dumps(output_shape)}{FINAL_JSON_CLOSE} with the placeholder replaced by the complete rewritten prompt."
    )
    return (
        "You are the final quality-control rewrite pass for a MiniMax H3 T2VA prompt. Rewrite the candidate, preserving its good concrete "
        "details, while fixing every listed failure. Do not explain or critique. Follow the Target Profile dialogue contract exactly; do not add "
        "visible text or subjects that the brief does not request. Add only the cuts required by the brief or Target Profile shot-count control. "
        "Treat both as binding.\n\n"
        f"User brief: {user_prompt.strip()}\n"
        f"Exact duration: {duration_seconds:.3f} seconds\n"
        f"Audio mode: {_normalize_audio_mode(audio_mode)}\n"
        f"Requested visual medium: {requested_medium or 'unspecified'}\n"
        f"Requested style locks: {'; '.join(style_locks) if style_locks else 'none'}\n"
        f"Shot-count selector: {configured_shot_count}; effective required count: {requested_shot_count or 'automatic'}; source: {shot_count_source}\n"
        f"Dialogue mode: {dialogue_mode}; required line count: {dialogue_line_count if dialogue_mode == 'required' else 'not enforced'}; guidance: {dialogue_guidance or 'none'}\n"
        f"Failures to repair: {_json_dumps(reasons)}\n"
        f"Required corrections: {' '.join(item for item in corrections if item)}\n\n"
        "The rewritten minimax_h3_prompt must contain only integrated_multimodal_description, overall_soundscape, and non_diegetic_music in "
        "that order with one blank line between fields. Begin directly with integrated_multimodal_description: [Shot 1]. Shot 1 has no "
        "timestamp; every later cut begins [Shot N] At MM:SS.mmm, the camera cuts to... with consecutive labels and increasing times below "
        "the exact duration. Keep stable subject labels, designs, props, and screen geography. Put event sounds beside their visible causes. "
        "End with 'The final frame holds on ...' followed by a resolved, observable subject state and camera composition.\n\n"
        f"Candidate prompt: {_json_dumps(candidate_prompt)}\n\n"
        f"{output_instruction}"
    )


def _build_model_prompt(
    user_prompt: str,
    master_prompt: str,
    target_profile: str,
    media_metadata: dict[str, Any],
    audio_mode: str = "auto_scene_audio",
    audio_guidance: str = "",
    target_duration_seconds: float = 0.0,
    ltx_style: str = "",
    ideogram_aspect_ratio: str = "1:1",
    ideogram_render_style: str = "",
    ideogram_exact_text: str = "",
    ideogram_json_output: bool = True,
    creativity_mode: str = "editorial",
    creative_strength: float = 0.6,
    negative_prompt_mode: str = "auto",
    negative_prompt_guidance: str = "",
    thinking_mode: str = "off",
    minimax_h3_mode: str = "",
    minimax_h3_reference_manifest: str = "",
    minimax_h3_shot_count: str = "auto",
    native_h3_output: bool = False,
    minimax_h3_dialogue_mode: str = "auto",
    minimax_h3_dialogue_line_count: int = 2,
    minimax_h3_dialogue_guidance: str = "",
    ltx_generation_mode: str = "auto",
    ltx_long_horizon_mode: str = "off",
    ltx_camera_capability: str = "advanced",
) -> str:
    prompt_media_metadata = copy.deepcopy(media_metadata)
    if "minimax_h3_reference_manifest" in prompt_media_metadata:
        prompt_media_metadata["minimax_h3_reference_manifest"] = (
            _strip_grounding_role_annotations(
                str(prompt_media_metadata["minimax_h3_reference_manifest"])
            )
        )
    if isinstance(prompt_media_metadata.get("image_roles"), list):
        prompt_media_metadata["image_roles"] = [
            _strip_grounding_role_annotations(str(role))
            for role in prompt_media_metadata["image_roles"]
        ]
    target_profile = _normalize_target_profile(target_profile)
    master_prompt = _model_visible_master_prompt(master_prompt, target_profile)
    minimax_h3_mode = _normalize_minimax_h3_mode(
        minimax_h3_mode or str(media_metadata.get("minimax_h3_mode", "t2va"))
    )
    minimax_h3_reference_manifest = _strip_grounding_role_annotations(
        _normalize_minimax_h3_reference_manifest(
            minimax_h3_reference_manifest
            or str(media_metadata.get("minimax_h3_reference_manifest", ""))
        )
    )
    minimax_h3_reference_definitions = _minimax_h3_reference_definitions(minimax_h3_reference_manifest)
    minimax_h3_reference_tags = [tag for tag, _description in minimax_h3_reference_definitions]
    minimax_h3_expected_subject_count = _normalize_minimax_h3_expected_subject_count(
        media_metadata.get("minimax_h3_expected_subject_count", 0)
    )
    minimax_h3_subject_count_instruction = _minimax_h3_subject_count_instruction(
        minimax_h3_expected_subject_count
    )
    minimax_h3_shot_count = _normalize_minimax_h3_shot_count(minimax_h3_shot_count)
    minimax_h3_effective_shot_count = _minimax_h3_effective_shot_count(user_prompt, minimax_h3_shot_count)
    minimax_h3_duration = _minimax_h3_duration_seconds(
        media_metadata,
        target_duration_seconds,
        user_prompt,
    )
    (
        minimax_h3_detail_minimum_words,
        minimax_h3_detail_target_low,
        minimax_h3_detail_target_high,
    ) = _minimax_h3_ref_detail_word_budget(
        minimax_h3_duration,
        minimax_h3_effective_shot_count or 1,
    )
    if minimax_h3_shot_count != "auto":
        minimax_h3_generation_shot_instruction = _minimax_h3_shot_count_instruction(minimax_h3_shot_count)
    elif minimax_h3_effective_shot_count:
        minimax_h3_generation_shot_instruction = (
            f"Use exactly {minimax_h3_effective_shot_count} consecutively numbered shots as requested in the user brief."
        )
    else:
        minimax_h3_generation_shot_instruction = _minimax_h3_shot_count_instruction("auto")
    minimax_h3_dialogue_mode = _normalize_minimax_h3_dialogue_mode(minimax_h3_dialogue_mode)
    minimax_h3_dialogue_line_count = _normalize_minimax_h3_dialogue_line_count(
        minimax_h3_dialogue_line_count
    )
    minimax_h3_dialogue_guidance = _minimax_h3_dialogue_guidance_text(
        minimax_h3_dialogue_guidance
    )
    minimax_h3_generation_dialogue_instruction = _minimax_h3_dialogue_instruction(
        minimax_h3_dialogue_mode,
        minimax_h3_dialogue_line_count,
        minimax_h3_dialogue_guidance,
        minimax_h3_mode,
    )
    audio_mode = _normalize_audio_mode(audio_mode)
    audio_guidance = _audio_guidance_text(audio_guidance)
    audio_instruction = _audio_instruction(audio_mode, audio_guidance, target_profile)
    measured_audio_report = prompt_media_metadata.get("ltx_measured_audio_report")
    if (
        target_profile == "ltx"
        and audio_mode != "visual_only"
        and isinstance(measured_audio_report, dict)
    ):
        measured_audio_instruction = (
            "An exact rendered song excerpt has already been selected for LTX audio-to-video conditioning. "
            "Treat ltx_measured_audio_report as measured evidence and its excerpt timing as authoritative; "
            "requested BPM, key, energy, or meter are not facts unless the report confirms them. Choreograph "
            "body motion, dancing, reframing, and emphasis around the actual pulse and event density. Vocal-activity "
            "proxies do not by themselves authorize visible singing or mouth articulation; a downstream LTX "
            "performance-mode control decides whether the performer lip-syncs or dances without singing. Keep "
            "identity-preserving movement calmer through dense transients or unstable passages and "
            "place larger readable moves in the report's stable or lower-density spans. Do not print hashes, "
            "scores, thresholds, paths, or diagnostic language in ltx_prompt, and do not invent quoted lyrics "
            "that are absent from the user brief or audio guidance."
        )
    else:
        measured_audio_instruction = "No decoded song measurement is connected."
    negative_prompt_mode = _normalize_negative_prompt_mode(negative_prompt_mode)
    negative_prompt_guidance = _negative_prompt_guidance_text(negative_prompt_guidance)
    negative_instruction = _negative_prompt_instruction(negative_prompt_mode, negative_prompt_guidance)
    if target_profile == "minimax_h3":
        h3_description_field = "detailed_description" if minimax_h3_mode == "ref2va" else "integrated_multimodal_description"
        if negative_prompt_mode == "empty":
            negative_instruction = (
                "Do not add a separate exclusions section. Preserve only exclusions stated explicitly "
                f"in the user's creative brief, inside {h3_description_field}."
            )
        else:
            negative_instruction = (
                "Fold only relevant requested exclusions into concise natural prose inside "
                f"{h3_description_field}; do not create a negative section or Stable Diffusion-style defect list."
            )
    thinking_mode = _effective_target_thinking_mode(thinking_mode, target_profile)
    thinking_instruction = _thinking_instruction(thinking_mode)
    controls = _target_context(
        media_metadata,
        target_duration_seconds,
        ltx_style,
        ideogram_aspect_ratio,
        ideogram_render_style,
        ideogram_exact_text,
        ideogram_json_output,
        creativity_mode,
        creative_strength,
        negative_prompt_mode,
        negative_prompt_guidance,
        ltx_generation_mode,
        ltx_long_horizon_mode,
        user_prompt,
        ltx_camera_capability,
    )
    minimax_h3_timeline_bound_instruction = _minimax_h3_timeline_bound_instruction(
        minimax_h3_effective_shot_count,
        minimax_h3_duration,
    )
    minimax_h3_requested_medium = _minimax_h3_requested_visual_medium(user_prompt)
    minimax_h3_medium_instruction = _minimax_h3_visual_medium_instruction(minimax_h3_requested_medium)
    minimax_h3_style_locks = _minimax_h3_requested_style_locks(user_prompt)
    minimax_h3_style_instruction = ""
    if minimax_h3_style_locks:
        style_location = (
            "inside the one-or-two-sentence style opening before [Shot 1]"
            if minimax_h3_mode == "ref2va"
            else "within the first 600 characters after [Shot 1]"
        )
        minimax_h3_style_instruction = (
            f"Requested style locks: {'; '.join(minimax_h3_style_locks)}. Copy each phrase, allowing only spelling normalization, "
            f"{style_location}; do not leave a requested aesthetic, era, or genre merely implied by props."
        )
    if target_profile == "minimax_h3":
        controls["minimax_h3_mode"] = minimax_h3_mode
        controls["minimax_h3_shot_count"] = minimax_h3_shot_count
        controls["minimax_h3_effective_shot_count"] = minimax_h3_effective_shot_count
        controls["minimax_h3_shot_count_source"] = (
            "target_profile"
            if minimax_h3_shot_count != "auto"
            else "user_prompt"
            if minimax_h3_effective_shot_count
            else "model_auto"
        )
        controls["minimax_h3_dialogue_mode"] = minimax_h3_dialogue_mode
        controls["minimax_h3_dialogue_line_count"] = minimax_h3_dialogue_line_count
        controls["minimax_h3_dialogue_guidance"] = minimax_h3_dialogue_guidance
        controls["minimax_h3_planning_duration_seconds"] = round(minimax_h3_duration, 3)
        controls["minimax_h3_duration_advisory"] = _minimax_h3_duration_advisory(minimax_h3_duration)
        if minimax_h3_mode == "ref2va":
            controls["minimax_h3_reference_manifest"] = minimax_h3_reference_manifest
            controls["minimax_h3_reference_tags"] = minimax_h3_reference_tags
            controls["minimax_h3_expected_subject_count"] = minimax_h3_expected_subject_count
            controls["minimax_h3_expected_subject_count_source"] = (
                "h3_reference_context" if minimax_h3_expected_subject_count else "auto"
            )
            controls["minimax_h3_reference_manifest_reasons"] = _minimax_h3_reference_manifest_validation_reasons(
                minimax_h3_reference_manifest
            )
        if minimax_h3_requested_medium:
            controls["minimax_h3_requested_visual_medium"] = minimax_h3_requested_medium
            controls["minimax_h3_visual_medium_lock"] = minimax_h3_medium_instruction
        if minimax_h3_style_locks:
            controls["minimax_h3_requested_style_locks"] = minimax_h3_style_locks
    model_visible_controls = _model_visible_target_controls(controls, target_profile)
    model_visible_media_metadata = _model_visible_media_metadata(
        prompt_media_metadata,
        target_profile,
    )
    creativity_instruction = _creativity_instruction(creativity_mode, creative_strength)
    if target_profile == "ltx":
        effective_ltx_mode = str(controls.get("ltx_generation_mode_effective", "text_to_video"))
        creativity_instruction = _ltx_creativity_instruction(
            creativity_mode,
            creative_strength,
            effective_ltx_mode,
            str(controls.get("ltx_camera_capability", "advanced")),
        )
    if target_profile == "minimax_h3":
        creativity_instruction = (
            f"{creativity_instruction} For MiniMax H3, every added detail must be visibly or audibly realizable inside the selected duration. "
            "Follow the Target Profile dialogue contract; do not add singing, visible text, logos, or extra subjects unless requested. "
            "Add only the cuts required by the brief or Target Profile shot-count control. "
            f"{minimax_h3_generation_dialogue_instruction} {minimax_h3_medium_instruction} {minimax_h3_style_instruction}"
        )
    structure_anchor = _target_structure_anchor(
        target_profile,
        ideogram_json_output,
        minimax_h3_mode,
        str(controls.get("ltx_generation_mode_effective", "text_to_video")),
        bool(controls.get("ltx_long_horizon_active")),
        str(controls.get("ltx_camera_capability", "advanced")),
    )
    media_grounding_mode = _media_grounding_mode(media_metadata)
    media_grounding_instruction = _media_grounding_instruction(media_metadata)
    ideogram_shape: Any
    if ideogram_json_output:
        ideogram_shape = {
            "high_level_description": "One or two sentence summary of the whole image.",
            "style_description": {
                "aesthetics": "Aesthetic keywords.",
                "lighting": "Lighting description including source time of day, ambient brightness, light sources, shadows, contrast, exposure level, blown highlights, bloom, halation, and dynamic range.",
                "photo": "Use this key only for photographic outputs; include lens/camera language such as focal length impression, depth of field, bokeh, lens blur, motion blur, grain/noise, perspective compression, distortion, and camera angle; omit art_style when photo is present.",
                "medium": "photograph",
                "color_palette": ["#RRGGBB optional uppercase hex values"],
            },
            "compositional_deconstruction": {
                "background": "Required background/environment description including source time of day, lighting state, exposure behavior, lens blur/bokeh, and background sharpness.",
                "elements": [
                    {
                        "type": "obj",
                        "bbox": [0, 0, 1000, 1000],
                        "desc": "Object/subject description, including focus, subject sharpness, overexposed highlights, flash/direct light, shadow detail, lens blur, or motion blur when visible. Omit bbox if uncertain.",
                    },
                    {
                        "type": "text",
                        "bbox": [0, 0, 1000, 1000],
                        "text": "Literal text to render exactly.",
                        "desc": "Typography and placement description. Omit bbox if uncertain.",
                    }
                ],
            },
        }
    else:
        ideogram_shape = (
            "One detailed Ideogram4 prompt sentence or paragraph with aspect ratio, style, exact readable text, "
            "layout, typography, colors, and negative constraints. If the user brief is short, expand it into a complete image prompt with subject, background environment, composition, lighting, style, color palette, and text placement when relevant. Preserve source time of day, lighting state, exposure, lens blur, bokeh, depth of field, focal length impression, motion blur, grain/noise, contrast, and camera angle exactly when image/video evidence is supplied."
        )
    if target_profile == "ltx":
        ideogram_shape = "Return an empty string for LTX target profile."
    elif target_profile == "minimax_h3":
        ideogram_shape = "Return an empty string for MiniMax H3 target profile."
    ltx_mode_effective = str(controls.get("ltx_generation_mode_effective", "text_to_video"))
    ltx_mode_contract = ltx25_compiler_contract(
        ltx_mode_effective,
        controls.get("ltx_planning_duration_seconds", 5.0),
        controls.get("ltx_complexity_limits"),
        controls.get("ltx_long_horizon_mode", "off"),
        audio_enabled=audio_mode != "visual_only",
        camera_capability=controls.get("ltx_camera_capability", "advanced"),
    )
    ltx_detail_scope = (
        "visible detail"
        if audio_mode == "visual_only"
        else "visible and audible detail"
    )
    ltx_speech_audio_shape = (
        "Do not include sound, music, speech, narration, voiceover, singing, ambience, Foley, room tone, or other audio cues."
        if audio_mode == "visual_only"
        else (
            "If speech, narration, voiceover, or singing is requested, identify the speaker or off-screen narrator and delivery, including language or accent when relevant, and preserve any user-supplied wording verbatim. Put only the exact intelligible words intended to be heard inside balanced straight double quotation marks, escape those quotes correctly in JSON, and keep terminal punctuation immediately before the closing quote with no intervening whitespace. Keep acting, camera, action, and sound direction outside the quotes so LTX cannot interpret that prose as additional speech. For off-screen voiceover, explicitly say that visible subjects do not speak. For an exact single off-screen voiceover, describe every requested closing visual beat before the final voiceover attribution, then make the exact quoted words the final non-whitespace characters of the entire prompt. Never append a speech-stop sentence, camera or action direction, sound-design note, or any other prose after that final closing quotation mark. Otherwise do not invent dialogue. Weave concrete sound into the timeline beside the action."
        )
    )
    stable_ltx_camera = normalize_ltx25_camera_capability(
        controls.get("ltx_camera_capability", "advanced")
    ) == "stable"
    if stable_ltx_camera:
        ltx_video_camera_shape = (
            "When video evidence is supplied, observe its opening and closing framing and whether the camera is stable. Under Stable / base-model capability, do not transcribe or invent an orbit, arc, roll, rotation, spin, swirl, sweeping or whip move, dolly zoom, shake, handheld pursuit, compound path, reversal, pronounced parallax, or sustained camera travel; use a stable hold or one restrained single-axis equivalent while preserving subject action and blocking. "
        )
        ltx_conditioned_camera_shape = (
            "For LTX I2V or FLF, creativity may change art direction, performance, lighting, environmental motion, and sound, but it may not raise camera ambition. When temporal camera evidence is absent, default to a locked/stabilized view or one short gentle push, pull, pan, tilt, or lateral track, then settle. Risky camera choreography requires Advanced / controlled camera. "
        )
    else:
        ltx_video_camera_shape = (
            "When video evidence is supplied, extract camera choreography before finalizing: first-frame shot size and angle, camera height and support, subject-camera relationship, movement type and direction, zoom/dolly/truck/boom/pan/tilt/roll/orbit/rotation, focus behavior, lens/depth changes, foreground-background parallax, and final frame. Preserve the opening framing, any visible camera movement or stable hold over time, and the closing framing inside the same paragraph. If the camera performs a compound move such as zooming out while rotating 180 degrees and following the subject across a room, describe it as one continuous camera-subject relationship with start and end framing. If the camera is mostly static, say that rather than inventing a zoom, pan, reveal, exit, or wider pull-back. "
        )
        ltx_conditioned_camera_shape = (
            "If input is vague, add concrete production detail without adding unrequested characters, cuts, locations, or action beats. For LTX I2V or FLF, a non-faithful creativity mode may author mode-appropriate camera choreography even when a still frame contains no temporal camera evidence, unless the user or verified evidence explicitly requires a locked/static camera. Start from every anchored camera geometry exactly, keep the path physically continuous, and let compatible simultaneous or sequential camera-motion phases form one start-to-end path rather than treating them as separate shots or cuts. "
        )
    ltx_shape: Any = (
        "Final LTX-2.5 positive prompt only. "
        f"{ltx_mode_contract} "
        "If image_identity_video_control synthesis is active, describe the reference-image subject performing the video action/control structure; the image controls identity and appearance, including face, hair, body type, wardrobe, accessories, styling, and distinguishing visual traits, while the video controls pose, depth, canny/edge layout, blocking, motion, timing, composition, camera choreography, and scene geometry. "
        "When this synthesis mode is active, do not copy the control-video subject's clothing, hair, face, body type, accessories, or styling unless explicitly requested, and do not let the person in the control video override the identity or appearance from the reference image. "
        f"If the user brief is short, complete only the {ltx_detail_scope} needed by the resolved generation mode and its duration budget. "
        "When still-image evidence is supplied, preserve source time of day, ambient brightness, visible light sources, shadows, contrast, exposure, overexposed or blown highlights, flash/direct light, bloom, halation, lens blur, bokeh, depth of field, focal length impression, perspective compression, lens distortion, motion blur, grain/noise, white balance, dynamic range, and camera angle exactly at each anchored instant. A still anchor fixes that instant, not camera immobility afterward. "
        f"{ltx_video_camera_shape}"
        f"{ltx_conditioned_camera_shape}"
        f"{ltx_speech_audio_shape} "
        "No headings, timestamps, JSON, Markdown, shot labels unless requested, non-visual senses, or phrases like 'The scene opens with'."
    )
    if target_profile == "ideogram4":
        ltx_shape = "Return an empty string for Ideogram4 target profile."
    elif target_profile == "minimax_h3":
        ltx_shape = "Return an empty string for MiniMax H3 target profile."
    minimax_h3_shape: Any = "Return an empty string unless the target profile is minimax_h3."
    if target_profile == "minimax_h3" and minimax_h3_mode == "ref2va":
        minimax_h3_shape = (
            f"Final MiniMax H3 Ref2VA full-reference prompt only, planned for exactly {minimax_h3_duration:.3f} seconds. The JSON string must "
            "contain exactly six English sections, in this exact order with one blank line between sections: subject_definitions, summary, "
            "retention_analysis, detailed_description, overall_soundscape, non_diegetic_music. Every section heading must be its own literal "
            "colon-terminated line: subject_definitions:, summary:, retention_analysis:, detailed_description:, overall_soundscape:, and "
            "non_diegetic_music:. Do not use the T2VA "
            "integrated_multimodal_description field. The declared reference manifest is authoritative: use every declared asset tag with exactly "
            "the listed spelling and number, never invent an undeclared asset tag, never omit an asset, and never silently exchange the jobs of two assets. "
            "When a declared Picture is a contact sheet, the entire sheet is one Picture asset: never promote cells or panels into extra Picture labels. "
            "Treat repeated views of one entity as one semantic Subject; for a declared multi-entity contact sheet, create one semantic Subject only for "
            "each distinct sheet entity actually requested by the brief, cite that contact-sheet Picture in each such definition, and never copy the "
            "sheet grid, cells, borders, labels, backgrounds, repeated poses, or layout into the target scene. "
            "In subject_definitions, put each separately tracked item on its own line. Create consecutive semantic <Subject N> labels for visible "
            "people, creatures, objects, environments, costumes, styles, actions, or effects that will recur in the target. A <Subject N> denotes "
            "reusable visible content, not an attachment slot; cite the <Picture N> or <Video N> that supplies each part of its identity, appearance, "
            "motion, environment, or style. Once visible content has a <Subject N> label, use that exact Subject label in every shot where it appears; "
            "the source asset tag may accompany its first appearance as provenance but must never replace the Subject label. Define a standalone <Picture N> only when the image itself is a concrete composition or storyboard anchor. "
            f"Subject-count contract: {minimax_h3_subject_count_instruction} "
            "Define a standalone <Video N> for whole-video editing, continuation, camera, cut, rhythm, or temporal-structure roles. Define every "
            "<Audio N> that is copied or referenced, and bind a voice reference to its <Subject N> with a numbered cue such as (S1) when it belongs to a visible speaker. "
            "For this local Ref2VA path, a Picture used only for identity, wardrobe, environment, or style is a semantic reference, not a guaranteed "
            "first-frame or last-frame pixel lock; claim keyframe completion only when the manifest explicitly assigns a concrete frame-anchor role. "
            "In summary, write one short paragraph beginning with one square-bracketed combination of only these exact task types, joined by ' + ': "
            "keyframe completion, reference generation, video editing, video continuation, audio reuse, audio reference. Select task types from actual "
            "roles rather than asset presence, then name the main <Subject N> and asset relationships without adding new labels. "
            "In retention_analysis, write one line for every independently defined label. For <Subject N>, <Picture N>, and <Video N>, use exactly one "
            "of fully_preserved, partially_preserved, attribute_transfer, weak_reference, followed by ' - ' and a concrete explanation of what remains "
            "and where it appears. For <Audio N>, use exactly one of fully_copy, partially_copy, reference, weak_reference. Do not put speaker IDs here. "
            "In detailed_description, first establish the immutable rendering medium and requested visual style in one or two natural-English sentences "
            f"before [Shot 1]. Then write a shot-by-shot playback description targeting about {minimax_h3_detail_target_low} to "
            f"{minimax_h3_detail_target_high} words for this duration and shot count, and never fewer than {minimax_h3_detail_minimum_words} words "
            "for generation tasks; scale video edits to source "
            "complexity. When many shots are requested, keep each shot concise—about 20 to 35 words—and prioritize emitting every required heading and "
            "shot over decorative prose; never omit the detailed_description heading or collapse shots into placeholders. Shot 1 has no timestamp. Every later cut begins [Shot N] At MM:SS.mmm, followed by a natural-English cut phrase, with "
            "consecutive labels and strictly increasing cut instants below the selected duration; never use time ranges. "
            f"{minimax_h3_generation_shot_instruction} {minimax_h3_timeline_bound_instruction} Insert each reference label at "
            "its first clear appearance and wherever its declared role takes effect. At first appearance, concretely restate each important referenced "
            "subject's visible identity, position, action, lighting, and interaction rather than relying on the analysis sections as a plot summary. "
            "Silently track stable subject identity, design, wardrobe, props, screen geography, causal state changes, and final state across cuts. Give every "
            f"shot its required camera behavior. {_MINIMAX_H3_CAMERA_SENTENCE_CONTRACT} "
            f"{minimax_h3_generation_dialogue_instruction} Cite <Audio N> only where its copy/reference relationship becomes audible. "
            "Put synchronized actions and sounds beside their visible causes. "
            "overall_soundscape summarizes ambience and physical sounds without repeating dialogue or audience-only music; non_diegetic_music specifies "
            "audience-only instrumentation, tempo, and dynamic development. Cite an <Audio N> in the matching audio section when that layer is copied or "
            "referenced. Use N/A for an absent layer. End on a resolved, observable final subject state and composition. Do not emit Markdown, extra headings, "
            "a negative section, schema explanations, legacy Director commands, or any reference label that the manifest does not support."
        )
    elif target_profile == "minimax_h3":
        minimax_h3_shape = (
            f"Final MiniMax H3 T2VA prompt only, planned for exactly {minimax_h3_duration:.3f} seconds. The JSON string must literally begin "
            "integrated_multimodal_description: [Shot 1] and contain exactly this three-field skeleton with one blank line between fields: "
            "integrated_multimodal_description: [Shot 1] ...; overall_soundscape: ...; non_diegetic_music: ... . "
            "Shot 1 has no timestamp. Later shots must be consecutive and begin "
            "[Shot N] At MM:SS.mmm, followed by 'the camera cuts to' or another official natural-English cut phrase; default to hard cuts, and use a "
            "cross-dissolve, fade, or wipe only when the user explicitly requests that effect. Use strictly increasing cut instants below "
            "the selected duration, never time ranges. Convert user-supplied ranges into cut instants and resolve overlaps chronologically. "
            f"{minimax_h3_generation_shot_instruction} {minimax_h3_timeline_bound_instruction} Scale action/dialogue density to the available time, including "
            "user-selected durations longer than 15 seconds without shortening or clamping them. Treat the user brief as a binding directing specification, "
            "not a loose theme list. Silently make a continuity ledger before writing: immutable rendering medium; exact subject names or stable labels; each "
            "subject's type, silhouette, face, colors, markings, wardrobe, and props; initial spatial relationship; location state; causal state after every beat; "
            "and the requested final state. Do not output the ledger. Preserve every requested aesthetic, era, and genre phrase; an 'X-inspired' request is an "
            "aesthetic lock, not permission to replace invented or fantasy creatures with generic real animals. Every recurring subject keeps the same label and "
            "defining traits after every cut. Each shot must begin from the prior shot's resulting position and state, then show a chronological cause, preparation, "
            "path of action, impact or counteraction, physical reaction, and observable result as appropriate. For a battle, establish both opponents and their "
            "screen geography before the first attack, show who targets whom, and carry every attack's consequence into the next beat. "
            f"{_MINIMAX_H3_CAMERA_SENTENCE_CONTRACT} Use a cut only when it "
            "reveals genuinely new subject, space, state, viewpoint, or time. "
            f"{minimax_h3_generation_dialogue_instruction} For "
            "voiceover, use 'says in an off-screen voiceover' and state that the corresponding on-screen lips remain closed. Use <scenetrans> only when speech "
            "continues across a cut and <cutoff> only for deliberate truncation at the end. "
            "Put synchronized dialogue, diegetic music, and every event-specific sound immediately beside its visible cause inside the integrated timeline; "
            "overall_soundscape may summarize ambience and Foley but cannot be the only place an action sound appears. Write 1-4 sentences of "
            "ambience, Foley, and non-verbal sounds in overall_soundscape, without repeating dialogue or music. Write 1-3 sentences in non_diegetic_music "
            "that specify audience-only instrumentation, tempo/rhythm, and dynamic changes tied to the requested cut or action beats, or N/A when there is no "
            "score; avoid abstract mood and explanations of the score's emotional purpose. End the last shot with a sentence beginning exactly 'The final frame "
            "holds on ...' that names the observable resulting subject state and camera composition. Resolve the final action instead of starting a new action, "
            "vaguely disappearing, or ending on 'about to,' 'prepares to,' or 'begins to' unless the user explicitly requests an unresolved cliffhanger. "
            "Preserve intentional visible text verbatim in double quotes. Fold only relevant exclusions "
            "into concise natural prose inside integrated_multimodal_description. Do not emit extra headings, Markdown, reference tags, a negative section, "
            "plot-summary filler, overlapping ranges, legacy Hailuo Director bracket commands, or explanations. Before returning, self-check that every requested "
            "subject, action, camera beat, sound beat, exclusion, dialogue line, and visible text item is represented once in the correct timeline position."
        )
    if target_profile == "minimax_h3" and native_h3_output:
        verified_ledger = (
            prompt_media_metadata.get("verified_grounding_ledger")
            if isinstance(prompt_media_metadata.get("verified_grounding_ledger"), dict)
            else {}
        )
        structural_metadata = {
            key: value
            for key, value in model_visible_media_metadata.items()
            if key
            not in {
                "verified_grounding_ledger",
                "warnings",
                "grounding_report",
                "grounding_guard",
            }
        }
        content_policy = re.sub(
            r"\s*Return only valid JSON with ltx_prompt, ideogram_prompt, minimax_h3_prompt, negative_prompt, scene_segments, and metadata\.\s*$",
            "",
            master_prompt.strip(),
            flags=re.IGNORECASE,
        ).strip()
        native_contract = str(minimax_h3_shape).replace(
            "The JSON string", "The native prompt"
        )
        opening_anchor = (
            "subject_definitions:"
            if minimax_h3_mode == "ref2va"
            else "integrated_multimodal_description: [Shot 1]"
        )
        evidence_report_id = str(
            prompt_media_metadata.get("grounding_evidence_report_id", "")
        )
        available_fact_ids = list(
            _eligible_grounding_fact_assets(verified_ledger)
        )
        timeline_final_block = (
            "TIMELINE — FINAL PREWRITE CHECK:\n"
            f"- {minimax_h3_timeline_bound_instruction}\n"
            "- Write and verify the complete cut-start schedule before drafting camera, action, sound, or dialogue prose.\n\n"
            if minimax_h3_timeline_bound_instruction
            else ""
        )
        required_dialogue_final_block = ""
        if minimax_h3_dialogue_mode == "required":
            dialogue_example = (
                "A locked medium shot holds on <Subject 1> (S1) as they look up and say with quiet awe: <d>[English] The clouds are moving.</d> Their lips close."
                if minimax_h3_mode == "ref2va"
                else "A locked medium shot holds (S1) as they look up and say with quiet awe: <d>[English] The clouds are moving.</d> Their lips close."
            )
            ref2va_speaker_check = (
                "- Ref2VA rule: repeat the speaking '<Subject N> (S1)' pair (using S2, S3, and so on for additional speakers) in the same shot before every line, including later lines by the same speaker. "
                "Keep Subject-to-Speaker IDs one-to-one; never output literal (Sx) or use a bare numbered cue after multiple Subject tags.\n"
                if minimax_h3_mode == "ref2va"
                else ""
            )
            required_dialogue_final_block = (
                "REQUIRED DIALOGUE — FINAL PREWRITE CHECK:\n"
                f"- Plan exactly {minimax_h3_dialogue_line_count} on-screen spoken line"
                f"{'s' if minimax_h3_dialogue_line_count != 1 else ''} before spending words on decorative camera coverage.\n"
                f"- Dialogue direction: {minimax_h3_dialogue_guidance or 'concise speech that advances the requested action or character intent.'}\n"
                f"- Use this complete pattern inside existing chronological shots: {dialogue_example} The camera phrase, action, delivery, and words may change, but the camera phrase may not be omitted and the tag bytes stay literal.\n"
                f"{ref2va_speaker_check}"
                "- (S1) stays outside <d>; [English] begins each block; </d> ends it. Never use <d>[S1], quoted speech without tags, "
                "or stage direction text inside a dialogue block.\n"
                f"- Before returning, count literal <d>={minimax_h3_dialogue_line_count}, literal </d>={minimax_h3_dialogue_line_count}, "
                f"and complete <d>[Language] words</d> blocks={minimax_h3_dialogue_line_count}. Reserve enough uninterrupted shot time for every line; "
                "do not create extra cuts merely to place speech. A dialogue line supplements its shot; it never replaces that shot's explicit camera path or locked/static hold.\n\n"
            )
        return (
            "You are the strict MiniMax H3 Director compiler. Convert a host-verified visual evidence ledger plus the "
            "user's creative brief into one generator-ready native MiniMax H3 prompt. The evidence ledger is the sole "
            "authority for what the supplied references visibly contain. Requested future action, staging, sound, and "
            "creative additions may come from the user brief, but never present an invented source attribute as an "
            "observed reference fact.\n\n"
            "OUTPUT CONTRACT (overrides every JSON or packet-format instruction elsewhere in the supplied policy): Return "
            "the native MiniMax H3 prompt followed by the two exact provenance lines specified below. Do not return JSON, "
            f"a quoted string, Markdown, reasoning, commentary, or final-answer markers. The first non-whitespace text must be exactly {opening_anchor}.\n\n"
            "CONTENT POLICY (its content rules are binding; any output-format language inside it is not):\n"
            f"{content_policy}\n\n"
            f"NATIVE H3 CONTRACT:\n{native_contract}\n\n"
            "GROUNDING RULES:\n"
            "- Use only high- or medium-confidence observed_facts from the verified ledger as claims about reference pixels.\n"
            "- Preserve the declared asset ordinals and reference roles exactly.\n"
            "- Do not copy host-only [dg:...] annotations into the prompt.\n"
            "- Do not mention fact IDs or the evidence report inside the native H3 prompt body.\n"
            "- If the ledger cannot support a requested reference-specific description, omit that unsupported source claim.\n\n"
            f"Verified evidence ledger: {_json_dumps(verified_ledger)}\n"
            f"Reference manifest: {minimax_h3_reference_manifest if minimax_h3_mode == 'ref2va' else 'not used'}\n"
            f"Structural media context: {_json_dumps(structural_metadata)}\n"
            f"Target controls: {_json_dumps(model_visible_controls)}\n"
            f"Creative direction: {creativity_instruction}\n"
            f"Audio mode: {audio_mode}\n"
            f"Audio instruction: {audio_instruction}\n"
            f"Audio guidance: {audio_guidance or 'none'}\n"
            f"Negative prompt instruction: {negative_instruction}\n"
            f"User creative brief: {user_prompt.strip()}\n\n"
            f"{timeline_final_block}"
            f"{required_dialogue_final_block}"
            "After completing non_diegetic_music, write one blank line and then exactly these two final machine lines. "
            "On the second line, list only the high/medium fact IDs whose reference-derived content the native prompt actually uses; "
            "the list must be non-empty and must cover every required reference asset:\n"
            f"GROUNDING_EVIDENCE_REPORT_ID: {evidence_report_id}\n"
            f"USED_GROUNDING_FACT_IDS: {','.join(available_fact_ids)}\n\n"
            f"Begin now with {opening_anchor}"
        )

    if target_profile == "minimax_h3":
        # The host expands this minimal model response back into the canonical
        # packet after parsing.  Inactive target fields are deliberately absent:
        # showing them to Director caused it to plan and sometimes populate
        # unrelated outputs before reaching the H3 prompt.
        contract = {"minimax_h3_prompt": minimax_h3_shape}
    else:
        contract = {
            "ltx_prompt": ltx_shape if target_profile == "ltx" else "",
            "ideogram_prompt": ideogram_shape if target_profile == "ideogram4" else "",
            "minimax_h3_prompt": "",
            "negative_prompt": "Follow the requested negative prompt mode.",
            # Segments and runtime/target metadata are host-owned and are rebuilt
            # after parsing.  Asking the model to echo them made the combined LTX
            # audit packet exceed a 1,024-token output budget before its JSON closed.
            "scene_segments": [],
            "metadata": {},
        }
    if media_grounding_mode == "verified_ledger":
        verified_ledger = (
            media_metadata.get("verified_grounding_ledger")
            if isinstance(media_metadata.get("verified_grounding_ledger"), dict)
            else {}
        )
        fact_assets_by_id = _eligible_grounding_fact_assets(verified_ledger)
        available_fact_ids = list(fact_assets_by_id)
        required_asset_ids = [
            str(asset_id)
            for asset_id in media_metadata.get("grounding_required_asset_ids", [])
            if isinstance(asset_id, str) and asset_id
        ]
        seeded_fact_ids: list[str] = []
        for asset_id in required_asset_ids:
            matching_fact_id = next(
                (
                    fact_id
                    for fact_id, fact_asset_ids in fact_assets_by_id.items()
                    if asset_id in fact_asset_ids
                ),
                None,
            )
            if matching_fact_id and matching_fact_id not in seeded_fact_ids:
                seeded_fact_ids.append(matching_fact_id)
        contract.setdefault("metadata", {}).update(
            {
                "grounding_evidence_report_id": str(
                    media_metadata.get("grounding_evidence_report_id", "")
                ),
                "used_grounding_fact_ids": (
                    seeded_fact_ids or available_fact_ids[:1] or ["fact-1"]
                ),
            }
        )
    if GROUNDING_LEDGER_SCHEMA_ID in master_prompt:
        contract["grounding_ledger"] = {
            "schema": GROUNDING_LEDGER_SCHEMA_ID,
            "analysis_status": "grounded | uncertain | refused | transport_error",
            "observed_facts": [
                {
                    "fact_id": "fact-1",
                    "claim": "One directly visible factual claim.",
                    "confidence": "high | medium | low",
                    "categories": ["object"],
                    "evidence": [
                        {
                            "asset_id": "image:1",
                            "sample_ordinal": 1,
                            "source_frame_index": 0,
                            "timecode_seconds": 0.0,
                        }
                    ],
                    "typed_claims": [],
                }
            ],
            "inferred_facts": [],
            "creative_additions": [],
            "uncertainties": [],
            "grounding_failure_reasons": [],
        }
    if target_profile == "ideogram4":
        profile_rules = (
            "- Ideogram JSON must follow its caption schema: top-level high_level_description, style_description, compositional_deconstruction; compositional_deconstruction contains background then elements; elements use type=obj/text and desc/text keys; no unknown layout/aspect_ratio keys inside the caption.\n"
            "- Ideogram color_palette values must be uppercase #RRGGBB hex strings if included; omit color_palette when no hex palette is known.\n"
            "- Use Ideogram aspect ratio as layout guidance only; do not include aspect_ratio as a JSON caption key.\n"
            "- Preserve exact quoted or user-specified text verbatim for Ideogram text elements.\n"
            "- Respect the selected negative prompt mode.\n"
        )
    elif target_profile == "minimax_h3" and minimax_h3_mode == "ref2va":
        profile_rules = (
            "- Use duration as pacing guidance. Never clamp the selected or explicitly written duration to 15 seconds; the locally exposed frame count is user-controlled, while the model's published trained range remains an advisory. Every cut instant must fit the actual planning duration.\n"
            "- Treat the reference manifest as an exact asset-role contract. Asset ordinals are one-based per type. Never invent, drop, renumber, merge, or cross-wire <Picture N>, <Video N>, or <Audio N> labels.\n"
            "- Use <Subject N> for semantic visible content and use asset labels for provenance or whole-asset roles. An image that only supplies identity or style belongs inside a Subject definition; do not falsely call it a first/last-frame lock.\n"
            f"- Semantic Subject-count contract: {minimax_h3_subject_count_instruction}\n"
            "- Every defined <Subject N> must appear by that exact tag in retention_analysis and in the shot timeline. A source asset tag may accompany the Subject at first appearance, but it never replaces the Subject tag.\n"
            "- Return exactly subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, and non_diegetic_music in that order, separated by one blank line. End every literal heading line with its colon: subject_definitions:, summary:, retention_analysis:, detailed_description:, overall_soundscape:, non_diegetic_music:.\n"
            "- Use only official summary task types and fixed retention markers. Explain exact preservation, transfer, copy, or reference behavior for each independently defined item.\n"
            "- Put one or two medium/style sentences before [Shot 1]. Then keep consecutive shot labels, a separate explicit camera-behavior sentence inside every shot, strictly increasing cut instants, stable subjects, causal continuity, and a resolved final state. Preserve requested or referenced camera choreography and freely use physically coherent H3 orbit, swirl, sweep/whip, parallax, or compound paths when they serve the shot; use a locked hold only when static framing genuinely serves it.\n"
            "- Fold only relevant user-requested exclusions into detailed_description without adding a negative section.\n"
            "- Preserve exact dialogue and visible text verbatim. Dialogue belongs only inside <d>[Language] ...</d> with a stable numbered speaker ID such as (S1); never output literal (Sx) or a role/name suffix inside the cue; soundscape must not repeat dialogue or music.\n"
            "- For visual_only audio mode, set both overall_soundscape and non_diegetic_music to N/A and do not use reference audio in the target timeline.\n"
        )
    elif target_profile == "minimax_h3":
        profile_rules = (
            "- Use duration as pacing guidance. Never clamp the selected or explicitly written duration to 15 seconds; 4-15 seconds is advisory only, and every cut instant must fit the actual planning duration.\n"
            "- Treat the requested rendering medium, aesthetic, era, and genre as immutable. State the medium in the first words after [Shot 1]; cinematic direction controls staging and camera work inside that medium and never overrides it.\n"
            "- Plan a silent continuity ledger for subject designs, props, spatial relationships, causal state changes, and the final state. Reuse stable subject labels and carry each shot's observable result into the next shot.\n"
            "- Fold only relevant requested exclusions into integrated_multimodal_description without adding a negative section.\n"
            "- Preserve exactly one integrated_multimodal_description field, one overall_soundscape field, and one non_diegetic_music field in that order, separated by one blank line.\n"
            "- Shot 1 has no timestamp. Every later shot uses a strictly increasing MM:SS.mmm cut instant inside the selected duration; do not emit start-end ranges or overlapping ranges. Default to hard cuts; use a cross-dissolve, fade, or wipe only when explicitly requested.\n"
            "- Give every shot its own explicit camera-behavior sentence; framing, angle, handheld, and 'the camera cuts' do not substitute for that declaration. Preserve requested camera choreography and choose expressive, physically coherent H3 movement—including orbit, swirl, sweep/whip, pronounced parallax, or compound travel—when it serves the shot; use a locked hold only when static framing genuinely serves it. End the last shot with a concrete final-frame composition and resolved visible state.\n"
            "- Preserve exact dialogue and visible text verbatim. Dialogue belongs only inside <d>[Language] ...</d> with a stable numbered speaker ID such as (S1); never output literal (Sx) or a role/name suffix inside the cue; soundscape must not repeat dialogue or music.\n"
            "- Place each event-specific sound beside the visible event that causes it inside integrated_multimodal_description; overall_soundscape is only the scene-wide ambience and Foley summary.\n"
            "- non_diegetic_music specifies instrumentation, tempo/rhythm, and dynamic development, not abstract mood or the score's emotional purpose; use N/A when no score is present.\n"
            "- For visual_only audio mode, set both overall_soundscape and non_diegetic_music to N/A and do not add dialogue, singing, ambience, Foley, or music.\n"
        )
    else:
        effective_ltx_mode = str(controls.get("ltx_generation_mode_effective", "text_to_video"))
        ltx_long_horizon_active = bool(controls.get("ltx_long_horizon_active"))
        if stable_ltx_camera:
            stable_camera_rule = (
                "- Camera capability is Stable / base model and overrides creativity-driven camera ambition. Prefer a locked/static/stabilized camera; otherwise use only one short gentle single-axis push, pull, pan, tilt, or lateral track, then settle. Never author orbit, arc, roll, rotation, spin, swirl, sweeping/whip movement, dolly zoom, shake, handheld pursuit, compound/multi-axis travel, reversal, pronounced parallax, or sustained camera travel.\n"
            )
            if effective_ltx_mode == "legacy_video":
                profile_rules = (
                    "- Respect the selected negative prompt mode.\n"
                    + stable_camera_rule
                    + "- Preserve source-video subject action, pose, depth/canny layout, blocking, timing, composition, and opening/closing framing, but translate unsupported camera choreography to a stable hold or one safe single-axis equivalent. Select Advanced / controlled camera to preserve an observed risky camera path.\n"
                    "- In image_identity_video_control, the reference image controls subject identity and appearance; do not copy the control-video subject's wardrobe, hair, face, body type, accessories, or styling unless requested.\n"
                )
            elif effective_ltx_mode == "image_to_video" and ltx_long_horizon_active:
                profile_rules = (
                    "- Respect the selected negative prompt mode.\n"
                    + stable_camera_rule
                    + "- Experimental Long Horizon is active. Derive a minimal critical anchor set from the connected first frame, then use establish, commit, sustain/reveal, and settle/hold as elastic phases inside one shot.\n"
                    "- Preserve the exact opening geometry and fill the long middle with readable subject, prop, lighting, atmospheric, environmental, and sound evolution rather than camera travel or extra plot beats. End on a concrete stable composition and hold.\n"
                )
            elif effective_ltx_mode == "image_to_video":
                profile_rules = (
                    "- Respect the selected negative prompt mode.\n"
                    + stable_camera_rule
                    + "- Sentence one must accurately ground the connected first-frame image: subject identity and count, appearance, environment, time of day, lighting, composition, shot scale, viewpoint, and opening camera geometry.\n"
                    "- Continue from that exact frame in one plausible take with one achievable subject action and a concrete ending state. Carry creativity through art direction, performance, environmental motion, lighting, and sound without inventing camera travel.\n"
                )
            elif effective_ltx_mode == "first_last_frame" and ltx_long_horizon_active:
                profile_rules = (
                    "- Respect the selected negative prompt mode.\n"
                    + stable_camera_rule
                    + "- Experimental Long Horizon is active. Retain critical endpoint correspondences and use one causal subject-and-environment bridge inside a stable shot; fill time without sustained camera travel.\n"
                    "- Settle into the supplied last-frame composition only near the end, then hold. If the endpoints require a compound camera bridge, require Advanced / controlled camera instead of inventing it under Stable.\n"
                )
            elif effective_ltx_mode == "first_last_frame":
                profile_rules = (
                    "- Respect the selected negative prompt mode.\n"
                    + stable_camera_rule
                    + "- Ground the connected first frame, preserve stable subject correspondences, and describe one causal subject-and-environment bridge ending in the exact supplied last frame. Use the least-invasive stable camera state compatible with both endpoints.\n"
                    "- If endpoint camera geometries cannot be joined by a locked or single-axis stabilized bridge, require Advanced / controlled camera rather than authoring compound or reversing travel.\n"
                )
            else:
                profile_rules = (
                    "- Respect the selected negative prompt mode.\n"
                    + stable_camera_rule
                    + "- This is unconditioned LTX-2.5 text-to-video. Establish subjects, setting, lighting, composition, shot scale, viewpoint, and a stable camera state directly from the brief. Use only duration-budgeted cuts and action beats.\n"
                )
        elif effective_ltx_mode == "legacy_video":
            profile_rules = (
                "- Respect the selected negative prompt mode.\n"
                "- This is the backward-compatible LTX source-video path. If media_synthesis_mode is image_identity_video_control, the reference image controls subject identity and appearance: face, hair, body type, wardrobe, accessories, styling, and distinguishing visual traits. The video controls pose, depth, canny/edge layout, composition, motion, timing, camera choreography, blocking, and scene geometry.\n"
                "- In image_identity_video_control, do not copy the control-video subject's wardrobe, hair, face, body type, accessories, or styling unless the user explicitly asks for those video-subject traits. If the image subject and video subject differ, replace the video subject appearance with the reference-image appearance.\n"
                "- Perform camera-choreography extraction before finalizing: first-frame framing, shot size, camera height/angle, camera support or stability, camera-subject relationship, movement type and direction, zoom/dolly/truck/boom/pan/tilt/roll/orbit/rotation, focus behavior, lens/depth changes, parallax, and final frame.\n"
                "- Preserve complex visible camera moves as continuous relationships, not generic labels. For example, if the source zooms out while rotating 180 degrees and tracking the actress to the other side of the room, say plainly where the camera starts, how it moves relative to her, and where it settles.\n"
                "- Preserve the clip chronologically by keeping the opening framing, any visible camera motion or stable hold, and the closing framing. If the source does not visibly zoom, pan, reveal, pull back, widen, rotate, orbit, or stage a new action beat, do not invent one.\n"
            )
        elif effective_ltx_mode == "image_to_video" and ltx_long_horizon_active:
            profile_rules = (
                "- Respect the selected negative prompt mode.\n"
                "- Experimental Long Horizon is active. Silently derive a minimal critical anchor set from the connected first frame, but do not dump a full static inventory into the emitted caption. Open with only identity, count, key props, lighting, and spatial relationships that must be conserved; then describe what changes next.\n"
                "- Continue from the exact frame in one physically plausible take. Treat establish, commit, sustain/reveal, and settle/hold as four elastic pacing phases inside that same shot, never as chapters or cuts.\n"
                "- Use one dominant camera intention, no unrequested reversal, and at most four independent causal actions. Fill the long middle with sustained travel, parallax, micro-motion, ambience, and synchronized transient sounds rather than extra plot beats.\n"
                "- Preserve every anchor that the brief does not ask to change. Do not freeze, omit, or weaken an identity, object, lighting, environment, or state transformation that the user explicitly requests.\n"
                "- End in a concrete observable composition, decelerate into it only near the end, and hold. Keep one continuous audio identity when audio is enabled.\n"
            )
        elif effective_ltx_mode == "image_to_video":
            profile_rules = (
                "- Respect the selected negative prompt mode.\n"
                "- Sentence one must accurately ground the connected first-frame image: subject identity and count, appearance, environment, time of day, lighting, composition, shot scale, viewpoint, and opening camera geometry. A still frame does not establish a temporal static or moving camera state.\n"
                "- Continue from that exact frame in one physically plausible take. Do not use a cut, montage, teleport, abrupt reframe, contradictory location or lighting, or an unseen extra actor required to execute the main action.\n"
                "- Use the LTX-2.5 duration limits in Target controls; prefer one achievable subject action and one physically continuous camera path with a concrete end state. That path may contain compatible simultaneous or sequential camera-motion phases and may be assertive, fast, or compound when the selected creativity mode and strength call for it. Do not collapse non-faithful modes into a locked shot, slow zoom, or generic pan merely because the input is a still image.\n"
            )
        elif effective_ltx_mode == "first_last_frame" and ltx_long_horizon_active:
            profile_rules = (
                "- Respect the selected negative prompt mode.\n"
                "- Experimental Long Horizon is active. Silently retain only the critical endpoint correspondences and causal state needed to join the supplied frames; do not exhaustively recaption either still image.\n"
                "- Use establish, commit, sustain/reveal, and settle/hold as four elastic pacing phases inside one continuous causal bridge, never as chapters or cuts.\n"
                "- Use one dominant camera intention and at most four independent causal actions. Preserve every stable anchor while still carrying out every endpoint or transformation explicitly requested by the brief.\n"
                "- Decelerate into the supplied last-frame composition only near the end, then hold it. Keep one continuous audio identity when audio is enabled.\n"
            )
        elif effective_ltx_mode == "first_last_frame":
            profile_rules = (
                "- Respect the selected negative prompt mode.\n"
                "- Ground the connected first frame first, preserve stable subject correspondences, then describe one continuous causal bridge that ends in the connected last frame's exact subjects, environment, lighting, composition, shot scale, and viewpoint.\n"
                "- Do not use a cut, montage, teleport, identity swap, discontinuous lighting change, or camera move that cannot physically join the two anchors.\n"
                "- Use the LTX-2.5 duration limits in Target controls and remove incompatible action beats until the transition is achievable. The bridge may still use assertive, fast, or compound continuous camera choreography when the selected creativity mode and strength call for it; the endpoints constrain the path but do not require an inert transition.\n"
            )
        else:
            profile_rules = (
                "- Respect the selected negative prompt mode.\n"
                "- This is unconditioned LTX-2.5 text-to-video. Establish concrete visible subjects, setting, lighting, composition, shot scale, viewpoint, and camera state from the brief without claiming to preserve a nonexistent first frame.\n"
                "- Use only the cuts and action beats allowed by the duration limits in Target controls. Every shot must restate its shot scale, camera motion or static hold, and viewpoint.\n"
                "- Prefer a single coherent shot when the brief does not require a meaningful cut.\n"
            )
    model_facing_controls = _model_visible_target_controls(controls, target_profile)
    if target_profile == "ltx" and isinstance(controls.get("ltx_complexity_limits"), dict):
        # Keep the duration-scaled motion-phase value for host diagnostics and
        # compatibility, but do not show that advisory integer to Director.
        # The language model repeatedly interpreted it as a hard one-move cap.
        model_facing_limits = dict(controls["ltx_complexity_limits"])
        model_facing_limits.pop("recommended_camera_motion_phases", None)
        model_facing_controls["ltx_complexity_limits"] = model_facing_limits
        long_plan = controls.get("ltx_long_horizon_plan")
        if isinstance(long_plan, dict):
            # The full schedule is already expressed as elastic prose in the
            # compiler contract. Keep only status here so the raw phase array
            # cannot be mistaken for extra shots or requested JSON output.
            model_facing_controls["ltx_long_horizon_plan"] = {
                "requested_mode": long_plan.get("requested_mode", "off"),
                "active": bool(long_plan.get("active")),
                "activation_reason": long_plan.get("activation_reason", "disabled"),
                "experimental": bool(long_plan.get("experimental")),
            }
    dialogue_output_rule = (
        f"- MiniMax H3 dialogue contract: {minimax_h3_generation_dialogue_instruction}\n"
        if target_profile == "minimax_h3"
        else "- Do not invent dialogue unless the user asked for speech, talking, singing, or conversation.\n"
    )
    if target_profile == "minimax_h3":
        packet_scope_rules = (
            "- The JSON object must contain minimax_h3_prompt"
            + (" and grounding_ledger" if "grounding_ledger" in contract else "")
            + ", and no other top-level keys.\n"
            "- Write only the finished MiniMax H3 prompt inside minimax_h3_prompt. Do not emit planning notes, a checklist, self-correction, schema commentary, or an alternate-target prompt.\n"
        )
        exclusion_labels = (
            f"Exclusion mode: {negative_prompt_mode}\n"
            f"Exclusion instruction: {negative_instruction}\n"
            f"Exclusion guidance: {negative_prompt_guidance or 'none'}\n"
        )
    else:
        packet_scope_rules = (
            "- Prompt fields must be ready to paste directly into the selected generator.\n"
            "- Only write the selected target profile; leave inactive prompt fields empty and scene_segments empty when the selected target does not use them.\n"
        )
        exclusion_labels = (
            f"Negative prompt mode: {negative_prompt_mode}\n"
            f"Negative prompt instruction: {negative_instruction}\n"
            f"Negative prompt guidance: {negative_prompt_guidance or 'none'}\n"
        )
    return (
        f"{master_prompt.strip()}\n\n"
        "Return the final answer as JSON matching this shape:\n"
        f"{_json_dumps(contract)}\n\n"
        "Critical output rules:\n"
        f"{packet_scope_rules}"
        "- Interpret User raw prompt as the creative brief for the target output, even when it is only a subject, moment, style, or rough idea. Supply the missing generator-prompt details yourself.\n"
        "- Do not put writing instructions, schema explanations, or adapter instructions inside any prompt field.\n"
        f"- Final response boundary: start the final answer with {FINAL_JSON_OPEN}, write one valid JSON object, close with {FINAL_JSON_CLOSE}, and write nothing after the closing marker.\n"
        "- Apply creative direction by adding specific visual details, not by adding meta commentary about creativity.\n"
        f"{dialogue_output_rule}"
        f"{profile_rules}\n"
        "- Preserve source lighting and time of day as non-negotiable visual identity when media or visual_description is supplied. If the source is night, the final prompt must repeatedly and concretely say nighttime/after dark, visible artificial light sources, deep shadows, and no daylight/sunlit/blue-sky conditions unless those are visibly present.\n"
        "- Preserve source camera/capture language as non-negotiable visual identity when media or visual_description is supplied. Use concrete industry terms when visible or strongly implied: overexposed highlights, clipped whites, underexposed shadows, direct flash, bloom, halation, shallow depth of field, soft focus, lens blur, bokeh, motion blur, focal length impression, wide-angle distortion, telephoto compression, high ISO grain/noise, white balance, high contrast, low dynamic range, and camera angle.\n"
        "- If Media grounding mode is metadata_only, do not claim to see or describe the supplied image/video pixels. Dimensions, frame counts, and sample indices are not visual evidence.\n"
        "- If Media grounding mode is visual_description, treat the visual_description as the only visual evidence and make the final prompt specific to it.\n"
        "- If Media grounding mode is pixels, ground observations in the attached pixels.\n\n"
        "- If Media grounding mode is verified_ledger, use only host-validated observed_facts as visual source evidence, cite the fact IDs used in metadata.used_grounding_fact_ids, and record metadata.grounding_evidence_report_id. No pixels are attached in this compiler mode.\n\n"
        f"Target profile: {target_profile}\n"
        f"Target controls: {_json_dumps(model_facing_controls)}\n"
        f"MiniMax H3 reference manifest: {minimax_h3_reference_manifest if minimax_h3_mode == 'ref2va' else 'not used'}\n"
        f"Media grounding mode: {media_grounding_mode}\n"
        f"Media grounding instruction: {media_grounding_instruction}\n"
        f"Creative direction: {creativity_instruction}\n"
        f"Thinking mode: {thinking_mode}\n"
        f"Thinking instruction: {thinking_instruction}\n"
        f"Audio mode: {audio_mode}\n"
        f"Audio instruction: {audio_instruction}\n"
        f"Audio guidance: {audio_guidance or 'none'}\n"
        f"Measured audio instruction: {measured_audio_instruction}\n"
        f"{exclusion_labels}"
        f"Media metadata: {_json_dumps(model_visible_media_metadata)}\n"
        f"User raw prompt: {user_prompt.strip()}\n"
        f"Final structure anchor: {structure_anchor}\n"
        f"Final response boundary: {FINAL_JSON_OPEN} JSON_OBJECT {FINAL_JSON_CLOSE}"
    )


def _template_packet(
    user_prompt: str,
    target_profile: str,
    media_metadata: dict[str, Any],
    duration_seconds: float,
    audio_mode: str = "auto_scene_audio",
    audio_guidance: str = "",
    target_duration_seconds: float = 0.0,
    ltx_style: str = "",
    ideogram_aspect_ratio: str = "1:1",
    ideogram_render_style: str = "",
    ideogram_exact_text: str = "",
    ideogram_json_output: bool = True,
    creativity_mode: str = "editorial",
    creative_strength: float = 0.6,
    negative_prompt_mode: str = "auto",
    negative_prompt_guidance: str = "",
    ltx_generation_mode: str = "auto",
    ltx_long_horizon_mode: str = "off",
    ltx_camera_capability: str = "advanced",
) -> dict[str, Any]:
    target_profile = _normalize_target_profile(target_profile)
    negative_prompt_mode = _normalize_negative_prompt_mode(negative_prompt_mode)
    negative_prompt_guidance = _negative_prompt_guidance_text(negative_prompt_guidance)
    prompt_base = _clean_prompt_request_text(user_prompt)
    visual_description = _visual_description_from_metadata(media_metadata)
    media_grounding_mode = _media_grounding_mode(media_metadata)
    limitation_prompt = False
    if visual_description:
        if prompt_base and not _is_media_reference_only_request(user_prompt):
            base = _single_paragraph(f"{visual_description}. User direction: {prompt_base}")
        else:
            base = visual_description
    elif media_grounding_mode == "pixels" and _is_media_reference_only_request(user_prompt):
        limitation_prompt = True
        base = (
            "DiffusionGemma received media pixels, but it did not return usable grounded prompt text. "
            "Re-run with runtime_required enabled to surface backend errors, or add a visual_description caption node."
        )
    elif media_grounding_mode == "metadata_only" and _is_media_reference_only_request(user_prompt):
        limitation_prompt = True
        base = (
            "Media pixels were supplied, but the active DiffusionGemma backend cannot inspect them. "
            "Add a visual_description from an image or video caption node, or use a pixel-capable backend."
        )
    else:
        base = prompt_base or "A clear cinematic scene with a distinct subject, setting, style, and motion."
    controls = _target_context(
        media_metadata,
        target_duration_seconds,
        ltx_style,
        ideogram_aspect_ratio,
        ideogram_render_style,
        ideogram_exact_text,
        ideogram_json_output,
        creativity_mode,
        creative_strength,
        negative_prompt_mode,
        negative_prompt_guidance,
        ltx_generation_mode,
        ltx_long_horizon_mode,
        user_prompt,
        ltx_camera_capability,
    )
    art_direction = _creative_art_direction_phrase(
        creativity_mode,
        creative_strength,
        target_profile,
        str(controls.get("ltx_generation_mode_effective", "text_to_video")),
        str(controls.get("ltx_camera_capability", "advanced")),
    )
    effective_duration = _effective_duration_seconds(media_metadata, target_duration_seconds) or duration_seconds
    if target_profile == "minimax_h3":
        h3_duration = _minimax_h3_duration_seconds(media_metadata, target_duration_seconds, user_prompt)
        controls["minimax_h3_planning_duration_seconds"] = round(h3_duration, 3)
        controls["minimax_h3_duration_advisory"] = _minimax_h3_duration_advisory(h3_duration)
    media_hint = ""
    if media_grounding_mode == "visual_description":
        media_hint = "The prompt is grounded in the supplied visual description rather than direct pixel analysis."
    elif media_grounding_mode == "pixels" and _is_image_identity_video_control(media_metadata):
        count = media_metadata.get("video_sampled_frame_count", media_metadata.get("sampled_frame_count", 0))
        media_hint = (
            f"The output subject identity comes from the reference image, while the action/control structure stays continuous across {count} sampled video frames. "
            "Use the reference image for face, hair, body type, wardrobe, accessories, styling, and distinguishing visual traits. "
            "Preserve the video pose, depth/canny-style layout, blocking, camera behavior, opening framing, visible movement or stable hold, closing framing, source exposure, lens/depth-of-field character, blur, grain, contrast, and lighting state. "
            "Do not copy the control-video subject's wardrobe, hair, face, body type, accessories, or styling, and do not let the person in the control video override the reference-image identity or appearance."
        )
    elif media_grounding_mode == "pixels" and media_metadata.get("source") == "video":
        count = media_metadata.get("sampled_frame_count", 0)
        media_hint = (
            f"The action stays continuous across the sampled video evidence from {count} frames, preserving subject motion, camera behavior, "
            "opening framing, any visible camera movement or stable hold, closing framing, source exposure, lens/depth-of-field character, blur, grain, contrast, and lighting state. "
            "Do not invent zooms, pans, reveals, exits, or wider pull-backs unless they are visibly present."
        )
    elif media_grounding_mode == "pixels" and media_metadata.get("source") in {"image", "image+video"}:
        media_hint = (
            "The composition preserves the supplied image evidence, visual identity, layout, style, exposure, lens/depth-of-field character, "
            "blur, grain, contrast, and lighting state."
        )

    audio_mode = _normalize_audio_mode(audio_mode)
    audio_guidance = _audio_guidance_text(audio_guidance)
    audio_hint = ""
    if audio_guidance and audio_mode != "visual_only":
        audio_hint = f"Scene-specific sound design features {audio_guidance}."
    elif audio_mode == "explicit_sound_design":
        audio_hint = "Concrete diegetic sound follows the visible action and setting."

    ltx_clauses: list[str] = []
    if controls["ltx_style"] and not limitation_prompt:
        ltx_clauses.append(f"Style: {controls['ltx_style']}.")
    ltx_clauses.append(_ensure_terminal_period(base))
    if media_hint and not limitation_prompt:
        ltx_clauses.append(media_hint)
    if art_direction and not limitation_prompt:
        ltx_clauses.append(art_direction)
    if audio_hint and not limitation_prompt:
        ltx_clauses.append(audio_hint)
    ltx = _single_paragraph(" ".join(ltx_clauses))
    h3_clauses = [_ensure_terminal_period(base)]
    if media_hint and not limitation_prompt:
        h3_clauses.append(media_hint)
    if art_direction and not limitation_prompt:
        h3_clauses.append(art_direction)
    if negative_prompt_guidance and negative_prompt_mode != "empty" and not limitation_prompt:
        h3_clauses.append(f"Exclude these unwanted elements or behaviors: {negative_prompt_guidance}.")
    h3_integrated = _single_paragraph(" ".join(h3_clauses))
    if audio_mode == "visual_only":
        h3_soundscape = "N/A"
        h3_music = "N/A"
    else:
        music_terms = r"\b(music|score|soundtrack|orchestra|orchestral|percussion|piano|guitar|strings|synth)\b"
        guidance_is_musical = bool(re.search(music_terms, audio_guidance, flags=re.IGNORECASE))
        h3_soundscape = (
            "Natural ambience and physical sounds follow the visible setting and actions."
            if guidance_is_musical
            else audio_guidance or "Natural ambience and physical sounds follow the visible setting and actions."
        )
        score_requested = bool(
            re.search(
                music_terms,
                f"{user_prompt} {audio_guidance}",
                flags=re.IGNORECASE,
            )
        )
        if guidance_is_musical:
            h3_music = audio_guidance
        elif score_requested:
            h3_music = "A restrained instrumental score uses measured percussion at a moderate tempo and builds dynamically with the action."
        else:
            h3_music = "N/A"
    minimax_h3 = _sanitize_structured_prompt_text(
        "\n\n".join(
            (
                f"integrated_multimodal_description: [Shot 1] {h3_integrated}",
                f"overall_soundscape: {h3_soundscape}",
                f"non_diegetic_music: {h3_music}",
            )
        ),
        "",
        8000,
    )
    synthesis_metadata: dict[str, Any] = {}
    _copy_synthesis_metadata(synthesis_metadata, media_metadata)

    if ideogram_json_output:
        ideogram: Any = _ideogram_prompt_object(
            base,
            ideogram_aspect_ratio,
            ideogram_render_style,
            controls["ideogram_exact_text"],
            creativity_mode,
            creative_strength,
        )
    else:
        ideogram = _ideogram_prompt_text(
            base,
            ideogram_aspect_ratio,
            ideogram_render_style,
            controls["ideogram_exact_text"],
        )
    if target_profile == "ltx":
        ideogram = ""
        minimax_h3 = ""
    elif target_profile == "ideogram4":
        ltx = ""
        minimax_h3 = ""
    else:
        ltx = ""
        ideogram = ""
    segments = _split_segments(ltx, effective_duration) if ltx and target_profile == "ltx" else []
    return {
        "ltx_prompt": ltx,
        "ideogram_prompt": ideogram,
        "minimax_h3_prompt": minimax_h3,
        "negative_prompt": ""
        if target_profile == "minimax_h3"
        else _apply_negative_prompt_policy("", negative_prompt_mode, negative_prompt_guidance),
        "scene_segments": segments,
        "metadata": {
            "backend": "template",
            "fallback_reason": "No compatible DiffusionGemma runtime was used.",
            "target_profile": target_profile,
            "visual_grounding_mode": media_grounding_mode,
            "media": media_metadata,
            "audio_mode": audio_mode,
            "audio_guidance": audio_guidance,
            "negative_prompt_mode": negative_prompt_mode,
            "negative_prompt_guidance": negative_prompt_guidance,
            "controls": controls,
            "ready_for_generation": False,
            "blocked_reasons": ["template_fallback"],
            **synthesis_metadata,
        },
    }


def _extract_labeled_prompt(text: str, label: str, structured: bool = False) -> str:
    labels = r"(?:ltx_prompt|ideogram_prompt|minimax_h3_prompt|negative_prompt|scene_segments|metadata)"
    pattern = rf"['\"]?\b{re.escape(label)}\b['\"]?\s*[:=]\s*(.*?)(?=['\"]?\b{labels}\b['\"]?\s*[:=]|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    value = match.group(1).strip().rstrip(",")
    if value.startswith('"') and value.endswith('"'):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, str):
                value = decoded
        except Exception:
            value = value.strip('"')
    if structured:
        return _sanitize_structured_prompt_text(value, "", 8000)
    return _sanitize_prompt_text(value, "", 8000)


def _has_salvageable_plain_text(text: str) -> bool:
    cleaned = _strip_markdown(_strip_thinking(text or "")).strip()
    if not cleaned:
        return False
    if cleaned in {"{", "}", "[", "]", "{}", "[]"}:
        return False
    letters = re.findall(r"[A-Za-z]", cleaned)
    words = re.findall(r"[A-Za-z]{2,}", cleaned)
    return len(letters) >= 20 and len(words) >= 4


def _extract_complete_minimax_h3_plain_prompt(
    text: str,
    minimax_h3_mode: str,
) -> str:
    """Extract only a complete canonical H3 block from non-JSON output."""

    value = _sanitize_structured_prompt_text(text, "", 0)
    mode = _normalize_minimax_h3_mode(minimax_h3_mode)
    opening = (
        r"(?im)^subject_definitions\s*:"
        if mode == "ref2va"
        else r"(?im)^integrated_multimodal_description\s*:\s*\[Shot\s+1\]"
    )
    match = re.search(opening, value)
    if not match:
        return ""
    candidate = _normalize_minimax_h3_prompt(value[match.start() :], mode)
    if mode == "ref2va":
        sections = _minimax_h3_ref_sections(candidate)
        if not sections or any(not sections.get(field, "").strip() for field in _MINIMAX_H3_REF_FIELDS):
            return ""
        return candidate
    structured = re.fullmatch(
        r"(?ms)integrated_multimodal_description\s*:\s*(.*?)\s*\n+\s*"
        r"overall_soundscape\s*:\s*(.*?)\s*\n+\s*"
        r"non_diegetic_music\s*:\s*(.*)",
        candidate,
    )
    if not structured or any(not structured.group(index).strip() for index in (1, 2, 3)):
        return ""
    return candidate


def _packet_from_plain_text_output(
    raw_output: str,
    user_prompt: str,
    target_profile: str,
    media_metadata: dict[str, Any],
    duration_seconds: float,
    audio_mode: str,
    audio_guidance: str,
    target_duration_seconds: float,
    ltx_style: str,
    ideogram_aspect_ratio: str,
    ideogram_render_style: str,
    ideogram_exact_text: str,
    ideogram_json_output: bool,
    creativity_mode: str,
    creative_strength: float,
    negative_prompt_mode: str,
    negative_prompt_guidance: str,
    max_output_chars: int,
    ltx_generation_mode: str = "auto",
    ltx_long_horizon_mode: str = "off",
    ltx_camera_capability: str = "advanced",
) -> dict[str, Any] | None:
    profile = _normalize_target_profile(target_profile)
    text = (
        _sanitize_structured_prompt_text(raw_output, "", int(max_output_chars))
        if profile == "minimax_h3"
        else _sanitize_prompt_text(raw_output, "", int(max_output_chars))
    )
    if not text:
        return None

    ltx_labeled = _extract_labeled_prompt(text, "ltx_prompt")
    ideogram_labeled = _extract_labeled_prompt(text, "ideogram_prompt")
    minimax_h3_labeled = _extract_labeled_prompt(text, "minimax_h3_prompt", structured=True)
    negative_labeled = _extract_labeled_prompt(text, "negative_prompt")
    minimax_h3_plain = ""
    if profile == "minimax_h3":
        minimax_h3_plain = _extract_complete_minimax_h3_plain_prompt(
            minimax_h3_labeled or text,
            str(media_metadata.get("minimax_h3_mode", "t2va")),
        )
        if not minimax_h3_plain:
            return None
    elif not (ltx_labeled or ideogram_labeled or _has_salvageable_plain_text(text)):
        return None
    template = _template_packet(
        user_prompt,
        target_profile,
        media_metadata,
        duration_seconds,
        audio_mode,
        audio_guidance,
        target_duration_seconds,
        ltx_style,
        ideogram_aspect_ratio,
        ideogram_render_style,
        ideogram_exact_text,
        ideogram_json_output,
        creativity_mode,
        creative_strength,
        negative_prompt_mode,
        negative_prompt_guidance,
        ltx_generation_mode,
        ltx_long_horizon_mode,
        ltx_camera_capability,
    )
    plain = _clean_prompt_request_text(ltx_labeled or ideogram_labeled or text)
    if profile == "ideogram4":
        template["ideogram_prompt"] = ideogram_labeled or plain
    elif profile == "minimax_h3":
        template["minimax_h3_prompt"] = _sanitize_structured_prompt_text(
            minimax_h3_plain,
            "",
            int(max_output_chars),
        )
        template["negative_prompt"] = ""
        template["scene_segments"] = []
    else:
        template["ltx_prompt"] = ltx_labeled or plain
        if ideogram_labeled:
            template["ideogram_prompt"] = ideogram_labeled
    if profile != "minimax_h3":
        template["negative_prompt"] = _apply_negative_prompt_policy(
            negative_labeled or template.get("negative_prompt", ""),
            negative_prompt_mode,
            negative_prompt_guidance,
        )
        template["scene_segments"] = _split_segments(
            str(template.get("ltx_prompt", "")),
            _effective_duration_seconds(media_metadata, target_duration_seconds) or duration_seconds,
        )
    metadata = template.setdefault("metadata", {})
    metadata.pop("fallback_reason", None)
    metadata.update(
        {
            "plain_text_salvage": True,
            "json_parse_warning": "Model output did not contain valid JSON; plain text output was used.",
            "template_used_for_missing_fields": True,
        }
    )
    return template


def _strict_native_minimax_h3_prompt(
    raw_output: str,
    minimax_h3_mode: str,
) -> tuple[str, str, list[str], list[str]]:
    """Extract only a native H3 prompt, without inventing packet structure.

    Native model output may carry the runtime's final-answer markers, which are
    transport metadata. Everything inside the payload remains model-authored and
    must begin with the target's first native heading. JSON, Markdown, quoted
    strings, conversational prefixes, and trailing final markers are rejected.
    """

    text = str(raw_output or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    repairs: list[str] = []
    if not text:
        return "", "", [], repairs

    anchored_thinking_patterns = (
        re.compile(r"\A<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL),
        re.compile(
            r"\A<\|thought\|>.*?<\|end_thought\|>\s*",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    )
    consumed = True
    while consumed:
        consumed = False
        for pattern in anchored_thinking_patterns:
            match = pattern.match(text)
            if match:
                text = text[match.end() :].lstrip()
                repairs.append("removed_anchored_native_thinking_wrapper")
                consumed = True
                break

    lowered = text.lower()
    matching_open = next(
        (marker for marker in FINAL_JSON_OPEN_MARKERS if lowered.startswith(marker.lower())),
        None,
    )
    if matching_open is not None:
        matching_close = next(
            (marker for marker in FINAL_JSON_CLOSE_MARKERS if lowered.endswith(marker.lower())),
            None,
        )
        if matching_close is None:
            return "", "", [], repairs
        text = text[len(matching_open) : len(text) - len(matching_close)].strip()
        repairs.append("removed_native_final_answer_wrapper")

    if (
        not text
        or "```" in text
        or re.search(r"</?think\b|<\|[^>\n]+\|>", text, flags=re.IGNORECASE)
    ):
        return "", "", [], repairs
    if text[0] in {'{', '[', '"', "'", '`'}:
        return "", "", [], repairs

    lines = text.splitlines()
    if len(lines) < 4 or lines[-3].strip():
        return "", "", [], repairs
    report_prefix = "GROUNDING_EVIDENCE_REPORT_ID:"
    facts_prefix = "USED_GROUNDING_FACT_IDS:"
    if not lines[-2].startswith(report_prefix) or not lines[-1].startswith(facts_prefix):
        return "", "", [], repairs
    evidence_report_id = lines[-2][len(report_prefix) :].strip()
    fact_text = lines[-1][len(facts_prefix) :].strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", evidence_report_id):
        return "", "", [], repairs
    used_fact_ids = [item.strip() for item in fact_text.split(",")]
    if (
        not used_fact_ids
        or any(
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}", item)
            for item in used_fact_ids
        )
        or len(set(used_fact_ids)) != len(used_fact_ids)
    ):
        return "", "", [], repairs
    text = "\n".join(lines[:-3]).rstrip()

    mode = _normalize_minimax_h3_mode(minimax_h3_mode)
    opening_pattern = (
        r"\Asubject[_ ]definitions\s*:?\s*(?:\n|$)"
        if mode == "ref2va"
        else r"\Aintegrated_multimodal_description\s*:\s*\[Shot\s+1\]"
    )
    if not re.match(opening_pattern, text, flags=re.IGNORECASE):
        return "", "", [], repairs
    return (
        _sanitize_structured_prompt_text(
            text,
            "",
            0,
            strip_thinking=False,
            strip_markdown=False,
        ),
        evidence_report_id,
        used_fact_ids,
        repairs,
    )


def repair_or_salvage_prompt_packet(
    raw_output: str,
    fallback_context: GemmaContext | str | None,
    target_config: TargetProfileConfig | dict[str, Any] | None,
    max_output_chars: int = 8000,
    creativity_mode: str = "editorial",
    creative_strength: float = 0.6,
    deterministic_transport_repair: bool = False,
    native_h3_output: bool = False,
) -> tuple[dict[str, Any], bool, str]:
    context = fallback_context if isinstance(fallback_context, GemmaContext) else _gemma_context_from_media(str(fallback_context or ""))
    target = _target_profile_config_to_dict(target_config)
    media_metadata = context.media_metadata.copy()
    duration = _safe_float(media_metadata.get("duration_seconds"), 0.0)
    transport_repairs: list[str] = []
    model_output_json_parse_valid = False
    if native_h3_output and target["target_profile"] == "minimax_h3":
        (
            native_h3_prompt,
            declared_evidence_report_id,
            declared_used_fact_ids,
            transport_repairs,
        ) = _strict_native_minimax_h3_prompt(raw_output, target["minimax_h3_mode"])
        parsed = None
        if native_h3_prompt:
            parsed = _template_packet(
                context.user_prompt,
                target["target_profile"],
                media_metadata,
                duration,
                target["audio_mode"],
                target["audio_guidance"],
                target["target_duration_seconds"],
                target["ltx_style"],
                target["ideogram_aspect_ratio"],
                target["ideogram_render_style"],
                target["ideogram_exact_text"],
                target["ideogram_json_output"],
                creativity_mode,
                creative_strength,
                target["negative_prompt_mode"],
                target["negative_prompt_guidance"],
                target["ltx_generation_mode"],
                target["ltx_long_horizon_mode"],
                target["ltx_camera_capability"],
            )
            parsed["minimax_h3_prompt"] = native_h3_prompt
            packet_metadata = (
                parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
            )
            for key in (
                "fallback_reason",
                "ready_for_generation",
                "blocked_reasons",
            ):
                packet_metadata.pop(key, None)
            verified_ledger = (
                media_metadata.get("verified_grounding_ledger")
                if isinstance(media_metadata.get("verified_grounding_ledger"), dict)
                else {}
            )
            grounding_basis_fact_ids = list(
                _eligible_grounding_fact_assets(verified_ledger)
            )
            packet_metadata.update(
                {
                    "compiler_output_contract": "native_minimax_h3_prompt_with_provenance/1",
                    "compiler_output_contract_parse_valid": True,
                    "compiler_packet_structure_source": "host",
                    "compiler_provenance_source": "model_declared_native_footer",
                    "grounding_evidence_report_id": declared_evidence_report_id,
                    "used_grounding_fact_ids": declared_used_fact_ids,
                    "compiler_grounding_basis_fact_ids": grounding_basis_fact_ids,
                }
            )
            parsed["metadata"] = packet_metadata
    elif deterministic_transport_repair:
        parsed, transport_repairs, model_output_json_parse_valid = (
            _deterministically_repaired_prompt_packet_json(raw_output)
        )
    else:
        parsed = _extract_json_object(raw_output)
        model_output_json_parse_valid = parsed is not None
    if parsed is None and target["target_profile"] == "minimax_h3":
        parsed, h3_transport_repairs = (
            _repair_single_h3_dialogue_closer_json_escape(
                raw_output,
                target["minimax_h3_mode"],
            )
        )
        if parsed is not None:
            transport_repairs = list(
                dict.fromkeys([*transport_repairs, *h3_transport_repairs])
            )
    if parsed is not None:
        host_structural_repairs: list[str] = []
        if target["target_profile"] == "minimax_h3":
            for field_name, default_value in (
                ("ltx_prompt", ""),
                ("ideogram_prompt", ""),
                ("negative_prompt", ""),
                ("scene_segments", []),
            ):
                if field_name not in parsed:
                    parsed[field_name] = copy.deepcopy(default_value)
                    host_structural_repairs.append(
                        f"completed_inactive_packet_field:{field_name}"
                    )
            h3_mode = target["minimax_h3_mode"]
            reference_manifest = _normalize_minimax_h3_reference_manifest(
                str(media_metadata.get("minimax_h3_reference_manifest", ""))
            )
            normalized_h3_prompt = _normalize_minimax_h3_prompt(
                str(parsed.get("minimax_h3_prompt", "") or ""),
                h3_mode,
            )
            ref2va_contract_delimiter_repairs: list[str] = []
            if h3_mode == "ref2va":
                (
                    normalized_h3_prompt,
                    ref2va_contract_delimiter_repairs,
                ) = _repair_minimax_h3_ref2va_contract_delimiters(
                    normalized_h3_prompt
                )
                normalized_h3_prompt = _repair_minimax_h3_ref2va_structure(
                    normalized_h3_prompt,
                    context.user_prompt,
                    reference_manifest,
                    _minimax_h3_duration_seconds(
                        media_metadata,
                        target["target_duration_seconds"],
                        context.user_prompt,
                    ),
                )
            normalized_h3_prompt, t2va_contract_transport_repairs = (
                _repair_minimax_h3_t2va_contract_transport(
                    normalized_h3_prompt,
                    h3_mode,
                )
            )
            normalized_h3_prompt, dialogue_transport_repairs = (
                _repair_minimax_h3_dialogue_transport(
                    normalized_h3_prompt,
                    h3_mode,
                )
            )
            medium_locked_h3_prompt = _apply_minimax_h3_visual_medium_lock(
                normalized_h3_prompt,
                context.user_prompt,
                h3_mode,
            )
            style_locked_h3_prompt = _apply_minimax_h3_style_locks(
                medium_locked_h3_prompt,
                context.user_prompt,
                h3_mode,
            )
            sound_repaired_h3_prompt = _apply_minimax_h3_sound_fidelity(
                style_locked_h3_prompt,
                context.user_prompt,
                h3_mode,
            )
            (
                parsed["minimax_h3_prompt"],
                camera_hold_repaired_shots,
            ) = _repair_minimax_h3_unspecified_shot_cameras(
                sound_repaired_h3_prompt,
                h3_mode,
            )
            if not isinstance(parsed.get("metadata"), dict):
                parsed["metadata"] = {}
            parsed["metadata"]["minimax_h3_mode"] = h3_mode
            if h3_mode == "ref2va":
                parsed["metadata"]["minimax_h3_reference_tags"] = _minimax_h3_reference_tags(reference_manifest)
                if ref2va_contract_delimiter_repairs:
                    parsed["metadata"][
                        "minimax_h3_ref2va_contract_delimiter_repairs"
                    ] = list(ref2va_contract_delimiter_repairs)
            if medium_locked_h3_prompt != normalized_h3_prompt:
                if not isinstance(parsed.get("metadata"), dict):
                    parsed["metadata"] = {}
                parsed["metadata"]["minimax_h3_visual_medium_lock_injected"] = True
            if style_locked_h3_prompt != medium_locked_h3_prompt:
                if not isinstance(parsed.get("metadata"), dict):
                    parsed["metadata"] = {}
                parsed["metadata"]["minimax_h3_style_locks_injected"] = True
            if sound_repaired_h3_prompt != style_locked_h3_prompt:
                if not isinstance(parsed.get("metadata"), dict):
                    parsed["metadata"] = {}
                parsed["metadata"]["minimax_h3_sound_fidelity_repaired"] = True
            if camera_hold_repaired_shots:
                parsed["metadata"]["minimax_h3_camera_contract_repair"] = {
                    "strategy": "locked_off_stable_view",
                    "shots": camera_hold_repaired_shots,
                    "repair_count": len(camera_hold_repaired_shots),
                }
            if dialogue_transport_repairs:
                parsed["metadata"]["minimax_h3_dialogue_transport_repairs"] = list(
                    dialogue_transport_repairs
                )
            if t2va_contract_transport_repairs:
                parsed["metadata"][
                    "minimax_h3_t2va_contract_transport_repairs"
                ] = list(t2va_contract_transport_repairs)
        if deterministic_transport_repair or transport_repairs:
            if not isinstance(parsed.get("metadata"), dict):
                parsed["metadata"] = {}
            parsed["metadata"]["model_output_json_parse_valid"] = bool(
                model_output_json_parse_valid
            )
            parsed["metadata"]["deterministic_transport_repair"] = bool(
                transport_repairs
            )
            parsed["metadata"]["deterministic_transport_repairs"] = list(
                transport_repairs
            )
            if host_structural_repairs:
                parsed["metadata"]["host_structural_repairs"] = host_structural_repairs
        if native_h3_output:
            parsed["metadata"]["native_prompt_transport_repairs"] = list(
                transport_repairs
            )
        return parsed, True, ""
    salvaged = _packet_from_plain_text_output(
        raw_output,
        context.user_prompt,
        target["target_profile"],
        media_metadata,
        duration,
        target["audio_mode"],
        target["audio_guidance"],
        target["target_duration_seconds"],
        target["ltx_style"],
        target["ideogram_aspect_ratio"],
        target["ideogram_render_style"],
        target["ideogram_exact_text"],
        target["ideogram_json_output"],
        creativity_mode,
        creative_strength,
        target["negative_prompt_mode"],
        target["negative_prompt_guidance"],
        int(max_output_chars),
        target["ltx_generation_mode"],
        target["ltx_long_horizon_mode"],
        target["ltx_camera_capability"],
    )
    if salvaged is not None:
        return salvaged, False, "Model output was not valid JSON; labeled/plain text salvage was used."
    fallback = _template_packet(
        context.user_prompt,
        target["target_profile"],
        media_metadata,
        duration,
        target["audio_mode"],
        target["audio_guidance"],
        target["target_duration_seconds"],
        target["ltx_style"],
        target["ideogram_aspect_ratio"],
        target["ideogram_render_style"],
        target["ideogram_exact_text"],
        target["ideogram_json_output"],
        creativity_mode,
        creative_strength,
        target["negative_prompt_mode"],
        target["negative_prompt_guidance"],
        target["ltx_generation_mode"],
        target["ltx_long_horizon_mode"],
        target["ltx_camera_capability"],
    )
    return fallback, False, "Model output was empty or unusable; template fallback packet was used."


_MINIMAX_H3_RESOLVED_REFERENCE_METADATA_KEYS = (
    "minimax_h3_reference_manifest_source",
    "minimax_h3_reference_manifest_preset",
    "minimax_h3_reference_policy_schema",
    "minimax_h3_reference_policy_layout",
    "minimax_h3_reference_policy",
    "minimax_h3_expected_subject_count",
)


def _restore_resolved_minimax_h3_reference_metadata(
    media_metadata: dict[str, Any],
    packet_media: dict[str, Any],
    context: GemmaContext,
    target: dict[str, Any],
) -> None:
    """Carry a host-resolved Ref2VA policy through the downstream splitter.

    The Director receives a policy after Context Hub has already emitted its
    DG_CONTEXT value.  Its final packet therefore contains the authoritative
    synthesized manifest, while the splitter's separately connected upstream
    context can still contain the original blank field.  Restore only the
    policy/auto sources produced by the host, and only when their declared
    Picture/Video inventory exactly matches the tensors in that context.
    """

    if not (
        target.get("target_profile") == "minimax_h3"
        and target.get("minimax_h3_mode") == "ref2va"
    ):
        return
    manifest_source = str(
        packet_media.get("minimax_h3_reference_manifest_source", "")
    ).strip()
    if manifest_source not in {"policy_node", "director_auto"}:
        return
    manifest = _normalize_minimax_h3_reference_manifest(
        str(packet_media.get("minimax_h3_reference_manifest", ""))
    )
    if _minimax_h3_reference_manifest_validation_reasons(manifest):
        return

    picture_count, video_count = _minimax_h3_reference_inventory(context)
    definitions = _minimax_h3_reference_definitions(manifest)
    picture_ordinals = sorted(
        int(match.group(1))
        for tag, _description in definitions
        if (match := re.fullmatch(r"<Picture (\d+)>", tag))
    )
    video_ordinals = sorted(
        int(match.group(1))
        for tag, _description in definitions
        if (match := re.fullmatch(r"<Video (\d+)>", tag))
    )
    if picture_ordinals != list(range(1, picture_count + 1)):
        return
    if video_ordinals != list(range(1, video_count + 1)):
        return

    media_metadata["minimax_h3_reference_manifest"] = manifest
    for key in _MINIMAX_H3_RESOLVED_REFERENCE_METADATA_KEYS:
        if key in packet_media:
            media_metadata[key] = copy.deepcopy(packet_media[key])
    media_metadata.update(
        {
            "minimax_h3_reference_tags": [tag for tag, _description in definitions],
            "minimax_h3_reference_manifest_reasons": [],
            "minimax_h3_reference_image_batch_count": picture_count,
            "minimax_h3_reference_video_attached_for_analysis": bool(video_count),
            "reference_image_count": picture_count,
            "reference_image_backend_attached": bool(
                picture_count and context.images is not None
            ),
        }
    )
    stale_warning_prefixes = (
        "H3 reference manifest needs correction:",
        "DiffusionGemma received ",
    )
    restored_warnings = [
        warning
        for warning in (
            list(media_metadata.get("warnings", []))
            if isinstance(media_metadata.get("warnings"), list)
            else []
        )
        if not str(warning).startswith(stale_warning_prefixes)
    ]
    for warning in (
        packet_media.get("warnings", [])
        if isinstance(packet_media.get("warnings"), list)
        else []
    ):
        if (
            not str(warning).startswith(stale_warning_prefixes)
            and warning not in restored_warnings
        ):
            restored_warnings.append(warning)
    media_metadata["warnings"] = restored_warnings


def _packet_to_prompt_outputs(
    packet: dict[str, Any],
    context: GemmaContext,
    target_config: TargetProfileConfig | dict[str, Any],
    max_output_chars: int,
    creativity_mode: str = "editorial",
    creative_strength: float = 0.6,
) -> tuple[str, str, str, str, str, str, str, str, str]:
    target = _target_profile_config_to_dict(target_config)
    profile = target["target_profile"]
    prompt_fallback = _clean_prompt_request_text(context.user_prompt)
    packet_metadata = packet.get("metadata") if isinstance(packet.get("metadata"), dict) else {}
    packet["metadata"] = packet_metadata
    packet_media = packet_metadata.get("media") if isinstance(packet_metadata.get("media"), dict) else {}
    media_metadata = context.media_metadata.copy()
    if packet_media:
        packet_metadata["model_reported_media"] = packet_media.copy()
        _restore_resolved_minimax_h3_reference_metadata(
            media_metadata,
            packet_media,
            context,
            target,
        )
        if (
            isinstance(context, GemmaContext)
            and context.images is not None
            and any(key in packet_metadata for key in ("media_policy", "runtime_ready", "raw_output_available"))
        ):
            for key in (
                "pixel_tensor_present",
                "pixels_sent_to_backend",
                "visual_grounding_mode",
                "transformers_video_transport",
            ):
                if key in packet_media:
                    media_metadata[key] = packet_media[key]
            if isinstance(packet_media.get("warnings"), list):
                existing = list(media_metadata.get("warnings", [])) if isinstance(media_metadata.get("warnings"), list) else []
                for warning in packet_media["warnings"]:
                    if warning not in existing:
                        existing.append(warning)
                media_metadata["warnings"] = existing
    media_metadata["visual_grounding_mode"] = _media_grounding_mode(media_metadata)
    packet_metadata["media"] = media_metadata.copy()
    _copy_synthesis_metadata(packet_metadata, media_metadata)
    duration = _safe_float(media_metadata.get("duration_seconds"), 0.0)
    effective_duration = _effective_duration_seconds(media_metadata, target["target_duration_seconds"])
    ltx_prompt = ""
    if profile == "ltx":
        ltx_prompt = _prompt_field_to_text(packet.get("ltx_prompt", ""), prompt_fallback, int(max_output_chars))
        if _media_grounding_mode(media_metadata) == "metadata_only" and _looks_like_unseen_media_bluff(ltx_prompt):
            replacement = _template_packet(
                context.user_prompt,
                "ltx",
                media_metadata,
                duration,
                target["audio_mode"],
                target["audio_guidance"],
                target["target_duration_seconds"],
                target["ltx_style"],
                target["ideogram_aspect_ratio"],
                target["ideogram_render_style"],
                target["ideogram_exact_text"],
                target["ideogram_json_output"],
                creativity_mode,
                creative_strength,
                target["negative_prompt_mode"],
                target["negative_prompt_guidance"],
                target["ltx_generation_mode"],
                target["ltx_long_horizon_mode"],
                target["ltx_camera_capability"],
            )
            ltx_prompt = _prompt_field_to_text(replacement.get("ltx_prompt", ""), prompt_fallback, int(max_output_chars))
            packet.setdefault("metadata", {})["visual_grounding_warning"] = (
                "Generic media-description output was replaced because the active backend cannot inspect pixels."
            )
        ltx_prompt, voiceover_boundary_report = _repair_ltx_exact_voiceover_boundary(
            ltx_prompt,
            target["audio_guidance"],
            int(max_output_chars),
            audio_mode=target["audio_mode"],
            request_text=context.user_prompt,
        )
        packet["ltx_prompt"] = ltx_prompt
        upstream_voiceover_report = packet_metadata.get("ltx_exact_voiceover_boundary")
        if (
            isinstance(upstream_voiceover_report, dict)
            and upstream_voiceover_report.get("applied") is True
            and voiceover_boundary_report.get("status") == "already_terminal"
        ):
            voiceover_boundary_report = copy.deepcopy(upstream_voiceover_report)
            voiceover_boundary_report["verified_at_splitter"] = True
        packet_metadata["ltx_exact_voiceover_boundary"] = voiceover_boundary_report
        if voiceover_boundary_report.get("applied"):
            # Model-authored segments still contain the unsafe pre-repair order.
            # Rebuild them below from the repaired prompt instead of leaking it
            # through the Director-string compatibility outputs.
            packet["scene_segments"] = []
    ideogram_value = packet.get("ideogram_prompt", "")
    if not ideogram_value and {"high_level_description", "compositional_deconstruction"}.issubset(packet.keys()):
        ideogram_value = packet
    ideogram_prompt = ""
    if profile == "ideogram4":
        ideogram_prompt = _ideogram_prompt_value_to_text(
            ideogram_value,
            prompt_fallback,
            target["ideogram_aspect_ratio"],
            target["ideogram_render_style"],
            target["ideogram_exact_text"],
            target["ideogram_json_output"],
            int(max_output_chars),
            creativity_mode,
            creative_strength,
        )
        if _media_grounding_mode(media_metadata) == "metadata_only" and _looks_like_unseen_media_bluff(ideogram_prompt):
            replacement = _template_packet(
                context.user_prompt,
                "ideogram4",
                media_metadata,
                duration,
                target["audio_mode"],
                target["audio_guidance"],
                target["target_duration_seconds"],
                target["ltx_style"],
                target["ideogram_aspect_ratio"],
                target["ideogram_render_style"],
                target["ideogram_exact_text"],
                target["ideogram_json_output"],
                creativity_mode,
                creative_strength,
                target["negative_prompt_mode"],
                target["negative_prompt_guidance"],
                target["ltx_generation_mode"],
                target["ltx_long_horizon_mode"],
                target["ltx_camera_capability"],
            )
            ideogram_prompt = _ideogram_prompt_value_to_text(
                replacement.get("ideogram_prompt", ""),
                prompt_fallback,
                target["ideogram_aspect_ratio"],
                target["ideogram_render_style"],
                target["ideogram_exact_text"],
                target["ideogram_json_output"],
                int(max_output_chars),
                creativity_mode,
                creative_strength,
            )
            packet.setdefault("metadata", {})["visual_grounding_warning"] = (
                "Generic media-description output was replaced because the active backend cannot inspect pixels."
            )
    minimax_h3_prompt = ""
    if profile == "minimax_h3":
        h3_mode = target["minimax_h3_mode"]
        reference_manifest = _normalize_minimax_h3_reference_manifest(
            str(media_metadata.get("minimax_h3_reference_manifest", ""))
        )
        packet_metadata["minimax_h3_mode"] = h3_mode
        expected_subject_count = _normalize_minimax_h3_expected_subject_count(
            media_metadata.get("minimax_h3_expected_subject_count", 0)
        )
        if h3_mode == "ref2va":
            packet_metadata["minimax_h3_expected_subject_count"] = expected_subject_count
            packet_metadata["minimax_h3_subject_count_contract_scope"] = (
                "semantic_subject_definition_labels"
            )
        configured_shot_count = target["minimax_h3_shot_count"]
        effective_shot_count = _minimax_h3_effective_shot_count(context.user_prompt, configured_shot_count)
        packet_metadata["minimax_h3_shot_count"] = configured_shot_count
        packet_metadata["minimax_h3_effective_shot_count"] = effective_shot_count
        packet_metadata["minimax_h3_shot_count_source"] = (
            "target_profile"
            if configured_shot_count != "auto"
            else "user_prompt"
            if effective_shot_count
            else "model_auto"
        )
        dialogue_mode = target["minimax_h3_dialogue_mode"]
        dialogue_line_count = target["minimax_h3_dialogue_line_count"]
        packet_metadata["minimax_h3_dialogue_mode"] = dialogue_mode
        packet_metadata["minimax_h3_dialogue_line_count"] = dialogue_line_count
        packet_metadata["minimax_h3_dialogue_contract_scope"] = (
            "structural_presence_count_markup"
        )
        if target["minimax_h3_dialogue_guidance"]:
            packet_metadata["minimax_h3_dialogue_guidance"] = target["minimax_h3_dialogue_guidance"]
        if h3_mode == "ref2va":
            packet_metadata["minimax_h3_reference_tags"] = _minimax_h3_reference_tags(reference_manifest)
        requested_medium = _minimax_h3_requested_visual_medium(context.user_prompt)
        if requested_medium:
            packet_metadata["minimax_h3_requested_visual_medium"] = requested_medium
        requested_style_locks = _minimax_h3_requested_style_locks(context.user_prompt)
        if requested_style_locks:
            packet_metadata["minimax_h3_requested_style_locks"] = requested_style_locks
        normalized_h3_prompt = _normalize_minimax_h3_prompt(
            str(packet.get("minimax_h3_prompt", "") or ""),
            h3_mode,
        )
        ref2va_contract_delimiter_repairs: list[str] = []
        if h3_mode == "ref2va":
            (
                normalized_h3_prompt,
                ref2va_contract_delimiter_repairs,
            ) = _repair_minimax_h3_ref2va_contract_delimiters(
                normalized_h3_prompt
            )
            normalized_h3_prompt = _repair_minimax_h3_ref2va_structure(
                normalized_h3_prompt,
                context.user_prompt,
                reference_manifest,
                _minimax_h3_duration_seconds(
                    media_metadata,
                    target["target_duration_seconds"],
                    context.user_prompt,
                ),
            )
        normalized_h3_prompt, t2va_contract_transport_repairs = (
            _repair_minimax_h3_t2va_contract_transport(
                normalized_h3_prompt,
                h3_mode,
            )
        )
        normalized_h3_prompt, dialogue_transport_repairs = (
            _repair_minimax_h3_dialogue_transport(
                normalized_h3_prompt,
                h3_mode,
            )
        )
        if dialogue_transport_repairs:
            existing_repairs = packet_metadata.get("minimax_h3_dialogue_transport_repairs")
            merged_repairs = list(existing_repairs) if isinstance(existing_repairs, list) else []
            for repair in dialogue_transport_repairs:
                if repair not in merged_repairs:
                    merged_repairs.append(repair)
            packet_metadata["minimax_h3_dialogue_transport_repairs"] = merged_repairs
        if t2va_contract_transport_repairs:
            existing_repairs = packet_metadata.get(
                "minimax_h3_t2va_contract_transport_repairs"
            )
            merged_repairs = (
                list(existing_repairs) if isinstance(existing_repairs, list) else []
            )
            for repair in t2va_contract_transport_repairs:
                if repair not in merged_repairs:
                    merged_repairs.append(repair)
            packet_metadata[
                "minimax_h3_t2va_contract_transport_repairs"
            ] = merged_repairs
        if ref2va_contract_delimiter_repairs:
            packet_metadata[
                "minimax_h3_ref2va_contract_delimiter_repairs"
            ] = list(ref2va_contract_delimiter_repairs)
        medium_locked_h3_prompt = _apply_minimax_h3_visual_medium_lock(
            normalized_h3_prompt,
            context.user_prompt,
            h3_mode,
        )
        style_locked_h3_prompt = _apply_minimax_h3_style_locks(
            medium_locked_h3_prompt,
            context.user_prompt,
            h3_mode,
        )
        sound_repaired_h3_prompt = _apply_minimax_h3_sound_fidelity(
            style_locked_h3_prompt,
            context.user_prompt,
            h3_mode,
        )
        (
            packet["minimax_h3_prompt"],
            camera_hold_repaired_shots,
        ) = _repair_minimax_h3_unspecified_shot_cameras(
            sound_repaired_h3_prompt,
            h3_mode,
        )
        if medium_locked_h3_prompt != normalized_h3_prompt:
            packet_metadata["minimax_h3_visual_medium_lock_injected"] = True
        if style_locked_h3_prompt != medium_locked_h3_prompt:
            packet_metadata["minimax_h3_style_locks_injected"] = True
        if sound_repaired_h3_prompt != style_locked_h3_prompt:
            packet_metadata["minimax_h3_sound_fidelity_repaired"] = True
        if camera_hold_repaired_shots:
            packet_metadata["minimax_h3_camera_contract_repair"] = {
                "strategy": "locked_off_stable_view",
                "shots": camera_hold_repaired_shots,
                "repair_count": len(camera_hold_repaired_shots),
            }
        dialogue_reasons = _minimax_h3_prompt_validation_reasons(
            packet["minimax_h3_prompt"],
            _minimax_h3_duration_seconds(
                media_metadata,
                target["target_duration_seconds"],
                context.user_prompt,
            ),
            context.user_prompt,
            target["audio_mode"],
            int(max_output_chars),
            h3_mode,
            reference_manifest,
            target["minimax_h3_shot_count"],
            dialogue_mode,
            dialogue_line_count,
            target["minimax_h3_dialogue_guidance"],
            expected_subject_count,
        )
        actual_dialogue_line_count = _minimax_h3_dialogue_block_count(
            packet["minimax_h3_prompt"]
        )
        packet_metadata["minimax_h3_dialogue_actual_line_count"] = actual_dialogue_line_count
        packet_metadata["minimax_h3_dialogue_contract_satisfied"] = not any(
            reason in _MINIMAX_H3_DIALOGUE_LOCAL_REASONS
            or reason == "minimax_h3_dialogue_audio_mode_conflict"
            for reason in dialogue_reasons
        )
        packet_metadata["minimax_h3_dialogue_diagnostics"] = (
            _minimax_h3_dialogue_diagnostics(packet["minimax_h3_prompt"], h3_mode)
        )
        if h3_mode == "ref2va":
            actual_subject_count = _minimax_h3_ref2va_subject_count(
                packet["minimax_h3_prompt"]
            )
            packet_metadata["minimax_h3_actual_subject_count"] = actual_subject_count
            packet_metadata["minimax_h3_subject_count_contract_satisfied"] = bool(
                expected_subject_count <= 0
                or actual_subject_count == expected_subject_count
            )
        minimax_h3_prompt = _sanitize_structured_prompt_text(
            packet["minimax_h3_prompt"],
            "",
            int(max_output_chars),
        )
    negative_prompt = ""
    if profile != "minimax_h3":
        negative_prompt = _apply_negative_prompt_policy(
            packet.get("negative_prompt", ""),
            target["negative_prompt_mode"],
            target["negative_prompt_guidance"],
        )
    scene_segments = packet.get("scene_segments")
    if profile in {"ideogram4", "minimax_h3"}:
        scene_segments = []
    elif not isinstance(scene_segments, list) or not scene_segments:
        scene_segments = _split_segments(ltx_prompt, effective_duration or duration)
    local_prompts, segment_lengths = _segments_to_director_strings(scene_segments)
    if profile == "ideogram4":
        prompt_to_verify = ideogram_prompt
    elif profile == "minimax_h3":
        prompt_to_verify = minimax_h3_prompt
    else:
        prompt_to_verify = ltx_prompt
    ready_for_generation, blocked_reasons = _packet_generation_readiness(
        packet,
        context,
        target,
        prompt_to_verify,
        media_metadata,
        int(max_output_chars),
    )
    packet_metadata["ready_for_generation"] = bool(ready_for_generation)
    packet_metadata["blocked_reasons"] = blocked_reasons
    guard_blocked, _guard_reasons, _guard_report = _strict_grounding_guard_block(
        packet_metadata
    )
    packet_metadata["claim_verification_passed"] = bool(
        not guard_blocked
        and not any(
            reason.startswith("metadata_only_")
            or reason.startswith("visual_description_contradiction")
            or reason.startswith("minimax_h3_")
            or reason.startswith("ltx_")
            for reason in blocked_reasons
        )
    )
    return (
        ltx_prompt,
        ideogram_prompt,
        negative_prompt,
        _json_dumps(scene_segments),
        local_prompts,
        segment_lengths,
        target["ideogram_aspect_ratio"],
        _json_dumps(packet.get("metadata") if isinstance(packet.get("metadata"), dict) else {}),
        minimax_h3_prompt,
    )


class _ComfyModelOptNVFP4Experts:
    @staticmethod
    def build(text_config: Any, ops: Any, device: Any):
        import torch

        class ComfyModelOptNVFP4Experts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = int(text_config.num_experts)
                self.hidden_dim = int(text_config.hidden_size)
                self.intermediate_dim = int(text_config.moe_intermediate_size)
                self.gate_proj = ops.MoEExperts(self.num_experts, self.hidden_dim, self.intermediate_dim, bias=False, device=device)
                self.up_proj = ops.MoEExperts(self.num_experts, self.hidden_dim, self.intermediate_dim, bias=False, device=device)
                self.down_proj = ops.MoEExperts(self.num_experts, self.intermediate_dim, self.hidden_dim, bias=False, device=device)
                from transformers.activations import ACT2FN

                self.act_fn = ACT2FN[text_config.hidden_activation]

            @staticmethod
            def _expert_linear(bank: Any, input_tensor: Any, expert_idx: int) -> Any:
                from comfy.quant_ops import QuantizedTensor

                weight = getattr(bank, "weight", None)
                if isinstance(weight, QuantizedTensor):
                    expert_weight = bank._expert_qt_from(weight, expert_idx)
                    input_scale = getattr(bank, "input_scale", None)
                    scale = None
                    if input_scale is not None:
                        scale = input_scale[expert_idx] if getattr(input_scale, "dim", lambda: 0)() else input_scale
                        scale = scale.to(device=input_tensor.device)
                    quantized_input = QuantizedTensor.from_float(input_tensor, bank.layout_type, scale=scale)
                    return torch.nn.functional.linear(quantized_input, expert_weight, None)
                return bank.expert_linear(input_tensor, expert_idx)

            def forward(self, hidden_states: Any, top_k_index: Any, top_k_weights: Any) -> Any:
                final_hidden_states = torch.zeros_like(hidden_states)
                with torch.no_grad():
                    expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
                    expert_mask = expert_mask.permute(2, 1, 0)
                    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

                for expert_idx_tensor in expert_hit:
                    expert_idx = int(expert_idx_tensor[0].item())
                    if expert_idx == self.num_experts:
                        continue
                    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                    current_state = hidden_states[token_idx]
                    gate = self._expert_linear(self.gate_proj, current_state, expert_idx)
                    up = self._expert_linear(self.up_proj, current_state, expert_idx)
                    current_hidden_states = self.act_fn(gate) * up
                    current_hidden_states = self._expert_linear(self.down_proj, current_hidden_states, expert_idx)
                    current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                    final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

                return final_hidden_states

        return ComfyModelOptNVFP4Experts()


def _resolve_model_leaf(model: Any, name: str) -> tuple[Any, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _model_has_leaf(model: Any, name: str) -> bool:
    try:
        parent, leaf = _resolve_model_leaf(model, name)
    except Exception:
        return False
    return leaf in getattr(parent, "_parameters", {}) or leaf in getattr(parent, "_buffers", {}) or hasattr(parent, leaf)


def _set_model_leaf(model: Any, name: str, value: Any) -> None:
    import torch

    parent, leaf = _resolve_model_leaf(model, name)
    if isinstance(value, torch.nn.Parameter):
        parent._parameters[leaf] = value
    elif leaf in getattr(parent, "_parameters", {}):
        parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
    elif leaf in getattr(parent, "_buffers", {}):
        parent._buffers[leaf] = value
    else:
        setattr(parent, leaf, value)


def _decoder_to_encoder_language_key(key: str) -> str:
    if key == "model.decoder.embed_tokens.weight":
        return "model.encoder.language_model.embed_tokens.weight"
    if key == "model.decoder.norm.weight":
        return "model.encoder.language_model.norm.weight"
    if key.startswith("model.decoder.layers."):
        remainder = key[len("model.decoder.layers.") :]
        if ".experts." in remainder or remainder.endswith(".layer_scalar") or remainder == "layer_scalar":
            return ""
        return f"model.encoder.language_model.layers.{remainder}"
    return ""


def _is_modelopt_expert_tensor_key(key: str) -> bool:
    return bool(re.search(r"^model\.decoder\.layers\.\d+\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.", key))


def _load_tensor_from_weight_map(model_path: str, weight_map: dict[str, str], key: str, handles: dict[str, Any]) -> Any:
    from safetensors import safe_open

    shard = weight_map[key]
    handle = handles.get(shard)
    if handle is None:
        handle = safe_open(str(Path(model_path) / shard), framework="pt", device="cpu")
        handles[shard] = handle
    return handle.get_tensor(key)


def _install_nvfp4_bank(
    bank: Any,
    model_path: str,
    weight_map: dict[str, str],
    layer_idx: int,
    projection: str,
    in_features: int,
    out_features: int,
    device: Any,
) -> None:
    import torch
    from comfy.quant_ops import QuantizedTensor, get_layout_class

    handles: dict[str, Any] = {}

    def tensor_for(expert_idx: int, suffix: str) -> Any:
        key = f"model.decoder.layers.{layer_idx}.experts.{expert_idx}.{projection}.{suffix}"
        return _load_tensor_from_weight_map(model_path, weight_map, key, handles)

    try:
        qdata = _modelopt_to_comfy_fp4_packing(
            torch.stack([tensor_for(i, "weight") for i in range(bank.num_experts)], dim=0)
        ).to(device=device)
        block_scale = torch.stack(
            [_modelopt_to_comfy_block_scale(tensor_for(i, "weight_scale")) for i in range(bank.num_experts)],
            dim=0,
        ).to(device=device)
        tensor_scale = torch.stack([tensor_for(i, "weight_scale_2") for i in range(bank.num_experts)], dim=0).to(
            device=device
        )
        input_scale = torch.stack([tensor_for(i, "input_scale") for i in range(bank.num_experts)], dim=0).to(device=device)
    finally:
        handles.clear()

    layout_cls = get_layout_class("TensorCoreNVFP4Layout")
    params = layout_cls.Params(
        scale=tensor_scale,
        block_scale=block_scale,
        orig_dtype=torch.bfloat16,
        orig_shape=(bank.num_experts, int(out_features), int(in_features)),
    )
    bank.quant_format = "nvfp4"
    bank.layout_type = "TensorCoreNVFP4Layout"
    bank.weight = torch.nn.Parameter(
        QuantizedTensor(qdata.to(torch.uint8), bank.layout_type, params),
        requires_grad=False,
    )
    bank.input_scale = torch.nn.Parameter(input_scale, requires_grad=False)


def _replace_and_load_nvfp4_experts(model: Any, config: Any, model_path: str, weight_map: dict[str, str], device: Any) -> None:
    import torch
    from comfy.ops import mixed_precision_ops

    ops = mixed_precision_ops({"format": "nvfp4"}, compute_dtype=torch.bfloat16)
    text_config = config.text_config
    total_layers = int(text_config.num_hidden_layers)
    _dg_log("Loading NVFP4 expert banks: %s layers.", total_layers)
    layer_iter = range(total_layers)
    tqdm_bar = None
    if _progress_enabled():
        try:
            from tqdm.auto import tqdm

            tqdm_bar = tqdm(layer_iter, total=total_layers, desc="DiffusionGemma NVFP4 experts", unit="layer", leave=True)
            layer_iter = tqdm_bar
        except Exception:
            tqdm_bar = None
    for layer_idx in layer_iter:
        experts = _ComfyModelOptNVFP4Experts.build(text_config, ops, device)
        _install_nvfp4_bank(
            experts.gate_proj,
            model_path,
            weight_map,
            layer_idx,
            "gate_proj",
            int(text_config.hidden_size),
            int(text_config.moe_intermediate_size),
            device,
        )
        _install_nvfp4_bank(
            experts.up_proj,
            model_path,
            weight_map,
            layer_idx,
            "up_proj",
            int(text_config.hidden_size),
            int(text_config.moe_intermediate_size),
            device,
        )
        _install_nvfp4_bank(
            experts.down_proj,
            model_path,
            weight_map,
            layer_idx,
            "down_proj",
            int(text_config.moe_intermediate_size),
            int(text_config.hidden_size),
            device,
        )
        model.model.decoder.layers[layer_idx].experts = experts
        model.model.encoder.language_model.layers[layer_idx].experts = experts
    if tqdm_bar is not None:
        tqdm_bar.close()
    _dg_log("Loaded NVFP4 expert banks.")


def _load_modelopt_nvfp4_bridge(config: RuntimeConfig) -> tuple[Any, Any]:
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoProcessor, DiffusionGemmaForBlockDiffusion

    load_start = time.perf_counter()
    _dg_log(
        "Preparing ModelOpt NVFP4 bridge. model_path=%s max_memory_gb=%s",
        config.model_path,
        config.max_memory_gb,
    )
    if 0 < float(config.max_memory_gb) < MIN_MODELOPT_NVFP4_BRIDGE_MEMORY_GB:
        raise RuntimeError(
            "The local ModelOpt NVFP4 bridge currently keeps DiffusionGemma weights on cuda:0 and does not "
            f"support a {config.max_memory_gb:g} GiB memory cap. Use max_memory_gb >= "
            f"{MIN_MODELOPT_NVFP4_BRIDGE_MEMORY_GB:g} for this bridge, or 0 for an explicit full-GPU load."
        )

    smoke = _nvfp4_bridge_smoke(config.model_path)
    if not smoke.get("passed"):
        raise RuntimeError(f"Comfy NVFP4 bridge smoke failed: {smoke.get('error', 'unknown error')}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("ModelOpt NVFP4 bridge requires CUDA.")

    weight_map = _safetensors_index(config.model_path)
    if not weight_map:
        raise RuntimeError("No model.safetensors.index.json weight_map found for NVFP4 checkpoint.")

    _dg_log("Loading DiffusionGemma config and processor.")
    hf_config = AutoConfig.from_pretrained(config.model_path, local_files_only=config.local_files_only)
    processor = AutoProcessor.from_pretrained(config.model_path, local_files_only=config.local_files_only)
    with init_empty_weights():
        model = DiffusionGemmaForBlockDiffusion(hf_config)

    bridge_start = time.perf_counter()
    try:
        _replace_and_load_nvfp4_experts(model, hf_config, config.model_path, weight_map, device)
    finally:
        _director_metric_add("nvfp4_bridge_seconds", time.perf_counter() - bridge_start)

    handles: dict[str, Any] = {}
    try:
        dense_keys = [key for key in sorted(weight_map) if not _is_modelopt_expert_tensor_key(key)]
        _dg_log("Loading non-expert tensors: %s tensors.", len(dense_keys))
        key_iter = dense_keys
        tqdm_bar = None
        if _progress_enabled():
            try:
                from tqdm.auto import tqdm

                tqdm_bar = tqdm(dense_keys, total=len(dense_keys), desc="DiffusionGemma tensors", unit="tensor", leave=True)
                key_iter = tqdm_bar
            except Exception:
                tqdm_bar = None
        for key in key_iter:
            if _is_modelopt_expert_tensor_key(key):
                continue
            tensor = _load_tensor_from_weight_map(config.model_path, weight_map, key, handles).to(device=device)
            target_keys = [key]
            tied_key = _decoder_to_encoder_language_key(key)
            if tied_key and _model_has_leaf(model, tied_key):
                target_keys.append(tied_key)
            parent, leaf = _resolve_model_leaf(model, target_keys[0])
            if leaf in getattr(parent, "_parameters", {}):
                shared_value = torch.nn.Parameter(tensor, requires_grad=False)
            else:
                shared_value = tensor
            for target_key in target_keys:
                if _model_has_leaf(model, target_key):
                    _set_model_leaf(model, target_key, shared_value)
    finally:
        try:
            if "tqdm_bar" in locals() and tqdm_bar is not None:
                tqdm_bar.close()
        except Exception:
            pass
        handles.clear()

    if hasattr(model, "lm_head") and _model_has_leaf(model, "model.decoder.embed_tokens.weight"):
        _set_model_leaf(model, "lm_head.weight", model.model.decoder.embed_tokens.weight)
    if (Path(config.model_path) / "generation_config.json").exists():
        model.generation_config = model.generation_config_class.from_pretrained(
            config.model_path,
            local_files_only=config.local_files_only,
        )
    model.eval()
    model.to(device)

    meta_names = [
        name
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]
        if getattr(tensor, "device", None) is not None and tensor.device.type == "meta"
    ]
    if meta_names:
        raise RuntimeError(f"NVFP4 bridge left {len(meta_names)} tensors on meta device, first: {meta_names[:8]}")
    _dg_log("ModelOpt NVFP4 bridge ready in %.2fs.", time.perf_counter() - load_start)
    return processor, model


def _load_transformers_model(config: RuntimeConfig) -> tuple[Any, Any]:
    key = json.dumps(
        {
            "model_path": config.model_path,
            "dtype": config.dtype,
            "quantization": config.quantization,
            "local_files_only": config.local_files_only,
            "max_memory_gb": config.max_memory_gb,
        },
        sort_keys=True,
    )
    if key in _MODEL_CACHE:
        _director_metric_add("runtime_model_cache_hit_count", 1)
        _dg_log("Using cached Transformers runtime. backend=%s quantization=%s max_memory_gb=%s", config.backend, config.quantization, config.max_memory_gb)
        return _MODEL_CACHE[key]

    cold_load_start = time.perf_counter()
    _director_metric_add("runtime_model_cold_load_count", 1)
    # Only one in-process DiffusionGemma profile can be resident safely on this
    # workstation. Evict old max_memory/temperature/model profiles before a new
    # load so changing loader settings does not double-allocate VRAM.
    _release_transformers_runtime()

    model_info = _model_path_info(config.model_path)
    try:
        _dg_log(
            "Loading Transformers runtime. path_kind=%s quantization=%s max_memory_gb=%s",
            model_info.get("path_kind", "unknown"),
            config.quantization,
            config.max_memory_gb,
        )
        if model_info["path_kind"] == "nvfp4_hf_repo" and model_info["quant_method"] == "modelopt":
            if config.quantization == "modelopt_nvfp4":
                processor, model = _load_modelopt_nvfp4_bridge(config)
                _MODEL_CACHE[key] = (processor, model)
                _dg_log("Transformers runtime cached.")
                return processor, model
            allow_unsupported = os.environ.get("DG_ALLOW_UNSUPPORTED_MODELOPT_NVFP4_LOAD", "").strip() == "1"
            if not allow_unsupported:
                raise RuntimeError(
                    "The selected NVFP4 checkpoint uses quant_method=modelopt. This ComfyUI runtime has "
                    "DiffusionGemmaForBlockDiffusion, but Transformers does not support modelopt NVFP4 "
                    "deserialization here. Choose quantization=modelopt_nvfp4 to use the Comfy NVFP4 bridge, "
                    "or set DG_ALLOW_UNSUPPORTED_MODELOPT_NVFP4_LOAD=1 only for future runtime experiments."
                )

        from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

        kwargs: dict[str, Any] = {
            "dtype": "auto" if config.dtype == "auto" else config.dtype,
            "device_map": "auto",
            "local_files_only": config.local_files_only,
        }
        if config.max_memory_gb > 0:
            kwargs["max_memory"] = {0: f"{config.max_memory_gb}GiB", "cpu": "64GiB"}

        processor = AutoProcessor.from_pretrained(config.model_path, local_files_only=config.local_files_only)
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(config.model_path, **kwargs)
        _MODEL_CACHE[key] = (processor, model)
        _dg_log("Transformers runtime cached.")
        return processor, model
    except Exception:
        _release_transformers_runtime()
        raise
    finally:
        _director_metric_add("model_load_seconds", time.perf_counter() - cold_load_start)


def _transformers_messages_and_processor_kwargs(
    prompt: str,
    media_context: MediaContext | None,
    visual_first: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    content: list[dict[str, Any]] = []
    pil_images = _image_batch_to_pil(media_context.images if isinstance(media_context, MediaContext) else None)
    media_metadata = media_context.metadata if isinstance(media_context, MediaContext) else {}
    source = media_metadata.get("source", "none") if isinstance(media_metadata, dict) else "none"
    processor_kwargs: dict[str, Any] = {}
    if visual_first and pil_images:
        # Guarded calls keep every visual item ahead of the final instructions.
        # The host-generated registry embedded in ``prompt`` remains the source
        # of truth for asset ordinals and roles.
        if _is_minimax_h3_reference_context(media_metadata):
            picture_count = int(
                max(0.0, _safe_float(media_metadata.get("minimax_h3_reference_image_batch_count"), 0.0))
            )
            picture_count = min(picture_count, len(pil_images))
            for image in pil_images:
                content.append({"type": "image", "image": image})
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"The first {picture_count} image(s) are ordered H3 <Picture N> evidence and the remaining "
                        "images are ordered sampled <Video 1> frames. "
                        + _sampled_video_frame_sequence_intro(media_metadata, len(pil_images))
                    ),
                }
            )
        elif _ltx25_conditioning_frame_intro(media_metadata):
            for image in pil_images:
                content.append({"type": "image", "image": image})
            content.append(
                {"type": "text", "text": _ltx25_conditioning_frame_intro(media_metadata)}
            )
        elif source in {"video", "image+video"}:
            transport = str(media_metadata.get("transformers_video_transport", _video_transport_mode()))
            if _is_image_identity_video_control(media_metadata):
                transport = "sampled_frame_images"
            if transport == "video_tokens":
                content.append({"type": "video", "video": pil_images})
                duration = (
                    _safe_float(media_metadata.get("duration_seconds"), 0.0)
                    if isinstance(media_metadata, dict)
                    else 0.0
                )
                effective_fps = (
                    len(pil_images) / duration
                    if duration > 0
                    else _safe_float(media_metadata.get("sample_fps"), 1.0)
                )
                processor_kwargs["videos_kwargs"] = {
                    "num_frames": max(1, len(pil_images)),
                    "video_metadata": {
                        "total_num_frames": len(pil_images),
                        "fps": max(0.001, effective_fps),
                        "width": int(media_metadata.get("width", 0) or 0) or None,
                        "height": int(media_metadata.get("height", 0) or 0) or None,
                        "duration": duration or None,
                    },
                }
            else:
                for image in pil_images:
                    content.append({"type": "image", "image": image})
                content.append(
                    {
                        "type": "text",
                        "text": _sampled_video_frame_sequence_intro(media_metadata, len(pil_images)),
                    }
                )
        else:
            for image in pil_images:
                content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}], processor_kwargs
    if pil_images and _is_minimax_h3_reference_context(media_metadata):
        picture_count = int(max(0.0, _safe_float(media_metadata.get("minimax_h3_reference_image_batch_count"), 0.0)))
        picture_count = min(picture_count, len(pil_images))
        for index, image in enumerate(pil_images[:picture_count], start=1):
            content.append(
                {
                    "type": "text",
                    "text": f"<Picture {index}> pixel evidence. Apply only the role assigned to <Picture {index}> in the H3 reference manifest.",
                }
            )
            content.append({"type": "image", "image": image})
        video_images = pil_images[picture_count:]
        if video_images:
            content.append({"type": "text", "text": _sampled_video_frame_sequence_intro(media_metadata, len(pil_images))})
            for image in video_images:
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": "End of sampled <Video 1> analysis frames."})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}], processor_kwargs
    if pil_images and _ltx25_conditioning_frame_intro(media_metadata):
        intro = _ltx25_conditioning_frame_intro(media_metadata)
        roles = list(media_metadata.get("ltx_frame_roles", [])) if isinstance(media_metadata, dict) else []
        for index, image in enumerate(pil_images):
            label = roles[index].replace("_", " ").upper() if index < len(roles) else f"FRAME {index + 1}"
            content.append({"type": "text", "text": f"LTX-2.5 {label} pixel evidence."})
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": intro})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}], processor_kwargs
    if pil_images and source in {"video", "image+video"}:
        frame_count = len(pil_images)
        transport = str(media_metadata.get("transformers_video_transport", _video_transport_mode()))
        if _is_image_identity_video_control(media_metadata):
            transport = "sampled_frame_images"
        if transport == "video_tokens":
            content.append({"type": "video", "video": pil_images})
            duration = _safe_float(media_metadata.get("duration_seconds"), 0.0) if isinstance(media_metadata, dict) else 0.0
            effective_fps = frame_count / duration if duration > 0 else _safe_float(media_metadata.get("sample_fps"), 1.0)
            video_metadata = {
                "total_num_frames": frame_count,
                "fps": max(0.001, effective_fps),
                "width": int(media_metadata.get("width", 0) or 0) or None,
                "height": int(media_metadata.get("height", 0) or 0) or None,
                "duration": duration or None,
            }
            processor_kwargs["videos_kwargs"] = {
                "num_frames": max(1, frame_count),
                "video_metadata": video_metadata,
            }
        else:
            content.append({"type": "text", "text": _sampled_video_frame_sequence_intro(media_metadata, frame_count)})
            if _is_image_identity_video_control(media_metadata):
                reference_count = min(_attached_reference_image_count(media_metadata), frame_count)
                reference_images = pil_images[:reference_count]
                video_images = pil_images[reference_count:]
                if reference_images:
                    content.append(
                        {
                            "type": "text",
                            "text": (
                                "IDENTITY REFERENCE IMAGE(S): use only these image(s) for the output subject's face, hair, "
                                "body type, wardrobe, accessories, styling, and distinguishing visual traits. These appearance "
                                "traits override the control-video subject."
                            ),
                        }
                    )
                    for image in reference_images:
                        content.append({"type": "image", "image": image})
                else:
                    content.append(
                        {
                            "type": "text",
                            "text": (
                                "IDENTITY REFERENCE MISSING: no identity-reference pixels were attached. Do not treat any "
                                "video frame as identity evidence or invent identity and appearance details."
                            ),
                        }
                    )
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "VIDEO CONTROL FRAMES: use the following frames only for pose, action, blocking, depth/canny/edge "
                            "layout, camera motion, timing, composition, lighting/capture, and scene geometry. Do not copy the "
                            "control-video subject's clothing, hair, face, body type, accessories, or styling."
                        ),
                    }
                )
                for image in video_images:
                    content.append({"type": "image", "image": image})
                end_text = (
                    "End video control frames. The identity reference is not part of the video timeline. The final prompt "
                    "must describe the reference-image subject, including reference wardrobe and accessories, performing "
                    "the video motion/control structure."
                    if reference_images
                    else "End video control frames. Identity and appearance remain unverified because no identity-reference pixels were attached."
                )
                content.append({"type": "text", "text": end_text})
            else:
                for image in pil_images:
                    content.append({"type": "image", "image": image})
    else:
        for image in pil_images:
            content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}], processor_kwargs


def _processor_inputs_have_pixels(inputs: dict[str, Any]) -> bool:
    return any(
        value is not None
        for key, value in inputs.items()
        if "pixel" in str(key).lower() or "video" in str(key).lower()
    )


class _MultimodalPrefillTimer:
    """Measure the first encoder call that actually receives pixel tensors."""

    def __init__(self, model: Any, enabled: bool) -> None:
        self.seconds = 0.0
        self.measured = False
        self._started = False
        self._finished = False
        self._wall_start = 0.0
        self._wall_end = 0.0
        self._start_event = None
        self._end_event = None
        self._handles: list[Any] = []
        if not enabled:
            return
        encoder = getattr(getattr(model, "model", None), "encoder", None)
        if encoder is None:
            return
        try:
            import torch

            if torch.cuda.is_available():
                self._start_event = torch.cuda.Event(enable_timing=True)
                self._end_event = torch.cuda.Event(enable_timing=True)

            def has_pixels(kwargs: dict[str, Any]) -> bool:
                return any(
                    value is not None
                    for key, value in kwargs.items()
                    if "pixel" in str(key).lower() or "video" in str(key).lower()
                )

            def before(_module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                if self._started or not has_pixels(kwargs):
                    return
                self._started = True
                self._wall_start = time.perf_counter()
                if self._start_event is not None:
                    self._start_event.record()

            def after(
                _module: Any,
                _args: tuple[Any, ...],
                kwargs: dict[str, Any],
                _output: Any,
            ) -> None:
                if self._finished or not self._started or not has_pixels(kwargs):
                    return
                if self._end_event is not None:
                    self._end_event.record()
                self._wall_end = time.perf_counter()
                self._finished = True

            installed_handles: list[Any] = []
            try:
                installed_handles.append(
                    encoder.register_forward_pre_hook(before, with_kwargs=True)
                )
                installed_handles.append(
                    encoder.register_forward_hook(after, with_kwargs=True)
                )
            except Exception:
                for handle in installed_handles:
                    try:
                        handle.remove()
                    except Exception:
                        pass
                raise
            self._handles = installed_handles
        except Exception:
            self._handles = []

    def finish(self) -> tuple[float, bool]:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []
        if not self._finished:
            return 0.0, False
        if self._start_event is not None and self._end_event is not None:
            try:
                self._end_event.synchronize()
                self.seconds = max(0.0, float(self._start_event.elapsed_time(self._end_event)) / 1000.0)
                self.measured = True
                return self.seconds, self.measured
            except Exception:
                pass
        self.seconds = max(0.0, self._wall_end - self._wall_start)
        self.measured = self.seconds > 0.0
        return self.seconds, self.measured


def _generated_token_count(sequences: Any, input_length: int, processor: Any) -> int:
    try:
        generated = sequences[:, max(0, int(input_length)) :]
        payload = generated.detach().cpu().tolist() if hasattr(generated, "detach") else generated.tolist()
    except Exception:
        return 0
    if payload and not isinstance(payload[0], (list, tuple)):
        payload = [payload]
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    count = 0
    for row in payload or []:
        count += sum(1 for token in row if pad_token_id is None or int(token) != int(pad_token_id))
    return int(count)


def _record_generated_output(
    sequences: Any,
    input_length: int,
    processor: Any,
    decoded_text: str,
) -> tuple[int, int]:
    token_count = _generated_token_count(sequences, input_length, processor)
    character_count = len(str(decoded_text or ""))
    _director_metric_add("generated_token_count", token_count)
    _director_metric_add("generated_character_count", character_count)
    return token_count, character_count


def _run_transformers(
    config: RuntimeConfig,
    prompt: str,
    media_context: MediaContext | None,
    max_new_tokens: int,
    node_id: str | None = None,
) -> str:
    run_start = time.perf_counter()
    processor, model = _load_transformers_model(config)
    media_message_start = time.perf_counter()
    messages, processor_kwargs = _transformers_messages_and_processor_kwargs(prompt, media_context)
    media_message_seconds = time.perf_counter() - media_message_start
    content = messages[0].get("content", []) if messages else []
    image_count = sum(1 for item in content if isinstance(item, dict) and item.get("type") == "image")
    video_count = sum(1 for item in content if isinstance(item, dict) and item.get("type") == "video")
    _dg_log(
        "Preparing processor inputs. images=%s videos=%s max_new_tokens=%s temperature=%.3g",
        image_count,
        video_count,
        max_new_tokens,
        config.temperature,
    )

    processor_start = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        processor_kwargs=processor_kwargs,
    )
    processor_seconds = time.perf_counter() - processor_start
    input_tokens = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
    _dg_log("Processor inputs ready in %.2fs. input_tokens=%s", processor_seconds, input_tokens)
    transfer_start = time.perf_counter()
    try:
        import torch

        inputs = {
            key: value.to(model.device) if isinstance(value, torch.Tensor) and hasattr(model, "device") else value
            for key, value in inputs.items()
        }
    except Exception:
        pass
    input_transfer_seconds = time.perf_counter() - transfer_start
    _director_metric_add(
        "media_preprocess_seconds",
        media_message_seconds + processor_seconds + input_transfer_seconds,
    )

    generate_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    denoise_steps = os.environ.get("DG_MAX_DENOISING_STEPS", "").strip()
    if denoise_steps:
        try:
            generate_kwargs["max_denoising_steps"] = max(1, int(denoise_steps))
        except Exception:
            pass
    if config.temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = config.temperature
    _dg_log("Generation started. max_new_tokens=%s", max_new_tokens)

    def do_generate() -> Any:
        import torch

        with torch.inference_mode():
            return model.generate(**inputs, **generate_kwargs)

    prefill_timer = _MultimodalPrefillTimer(model, _processor_inputs_have_pixels(inputs))
    generate_start = time.perf_counter()
    try:
        generated = _run_blocking_with_progress(
            "DiffusionGemma generate",
            do_generate,
            node_id=node_id,
            estimated_seconds=max(30.0, min(300.0, 45.0 + (float(max_new_tokens) * 0.12))),
        )
    finally:
        generation_seconds = time.perf_counter() - generate_start
        prefill_seconds, prefill_measured = prefill_timer.finish()
        _director_metric_add("generation_seconds", generation_seconds)
        if prefill_measured:
            _director_metric_add("multimodal_prefill_seconds", prefill_seconds)
            _director_metric_set("multimodal_prefill_is_separately_measured", True)
    _dg_log("Generation finished in %.2fs.", generation_seconds)
    sequences = getattr(generated, "sequences", generated)
    decode_start = time.perf_counter()
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    if hasattr(processor, "batch_decode"):
        try:
            text = processor.batch_decode(sequences[:, input_len:], skip_special_tokens=True)[0]
        except Exception:
            text = processor.batch_decode(sequences, skip_special_tokens=True)[0]
    elif hasattr(processor, "decode"):
        text = processor.decode(sequences[0], skip_special_tokens=True)
    else:
        text = str(sequences)
    decode_seconds = time.perf_counter() - decode_start
    generated_tokens, generated_characters = _record_generated_output(
        sequences,
        input_len,
        processor,
        text,
    )
    _director_metric_append_call(
        {
            "stage": "director",
            "max_new_tokens": int(max_new_tokens),
            "input_token_count": int(input_tokens),
            "generated_token_count": generated_tokens,
            "generated_character_count": generated_characters,
            "media_message_seconds": round(media_message_seconds, 6),
            "processor_seconds": round(processor_seconds, 6),
            "input_transfer_seconds": round(input_transfer_seconds, 6),
            "multimodal_prefill_seconds": round(prefill_seconds, 6),
            "multimodal_prefill_measured": bool(prefill_measured),
            "generation_seconds": round(generation_seconds, 6),
            "decode_seconds": round(decode_seconds, 6),
            "total_seconds": round(time.perf_counter() - run_start, 6),
        }
    )
    _dg_log("Decode finished in %.2fs. output_chars=%s total_runtime=%.2fs", decode_seconds, len(text), time.perf_counter() - run_start)
    return text


def _shape_list(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return []
    try:
        return [int(item) for item in shape]
    except Exception:
        return []


def _tensor_count_equal(value: Any, token_id: Any) -> int:
    if value is None or token_id is None:
        return 0
    try:
        return int((value == int(token_id)).sum().item())
    except Exception:
        return 0


def _processor_transport_proof(
    inputs: Any,
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    asset_registry: dict[str, Any] | None,
) -> dict[str, Any]:
    """Summarize what the processor actually produced without retaining tensors."""

    registry = asset_registry if isinstance(asset_registry, dict) else {}
    expected_samples = int(max(0, _safe_float(registry.get("expected_image_count"), 0.0)))
    content = messages[0].get("content", []) if messages else []
    content_image_count = sum(
        1 for item in content if isinstance(item, dict) and item.get("type") == "image"
    )
    content_video_count = sum(
        1 for item in content if isinstance(item, dict) and item.get("type") == "video"
    )
    keys = sorted(str(key) for key in inputs.keys()) if hasattr(inputs, "keys") else []
    tensor_facts: dict[str, dict[str, Any]] = {}
    media_position_facts: dict[str, dict[str, Any]] = {}
    observed_samples = 0
    observed_position_samples = 0
    processor_pixel_patch_count = 0
    for key in keys:
        try:
            value = inputs[key]
        except Exception:
            continue
        shape = _shape_list(value)
        if not shape:
            continue
        lower = key.lower()
        if ("image_position" in lower or "video_position" in lower) and len(shape) >= 1:
            media_position_facts[key] = {
                "shape": shape,
                "dtype": str(getattr(value, "dtype", "")),
            }
            observed_position_samples += (
                int(shape[1]) if "video" in lower and len(shape) >= 2 else int(shape[0])
            )
        is_pixel_tensor = "pixel" in lower and len(shape) >= 3
        if not is_pixel_tensor:
            continue
        dtype = str(getattr(value, "dtype", ""))
        tensor_facts[key] = {"shape": shape, "dtype": dtype}
        if "video" in lower and len(shape) >= 4:
            observed_samples += int(shape[1])
            if len(shape) >= 3:
                processor_pixel_patch_count += int(shape[1]) * int(shape[2])
        else:
            observed_samples += int(shape[0])
            if len(shape) >= 2:
                processor_pixel_patch_count += int(shape[0]) * int(shape[1])

    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    model_config = getattr(model, "config", None)
    image_token_id = getattr(model_config, "image_token_id", None) or getattr(processor, "image_token_id", None)
    video_token_id = getattr(model_config, "video_token_id", None) or getattr(processor, "video_token_id", None)
    image_media_tokens = _tensor_count_equal(input_ids, image_token_id)
    video_media_tokens = _tensor_count_equal(input_ids, video_token_id)
    effective_visual_token_budget = image_media_tokens + video_media_tokens
    media_present = expected_samples > 0
    media_tokens_present = (image_media_tokens + video_media_tokens) > 0
    media_positions_present = bool(media_position_facts)
    relevant_token_id_known = bool(
        (content_video_count and video_token_id is not None)
        or (content_image_count and image_token_id is not None)
    )
    media_token_coverage = bool(
        (content_video_count and video_media_tokens >= content_video_count)
        or (content_image_count and image_media_tokens >= content_image_count)
    )
    position_counts_match = bool(
        not media_positions_present or observed_position_samples == expected_samples
    )
    token_counts_match = bool(not relevant_token_id_known or media_token_coverage)
    counts_match = bool(
        not media_present
        or (
            observed_samples == expected_samples
            and (content_image_count == expected_samples or content_video_count == 1)
            and (media_tokens_present or media_positions_present)
            and token_counts_match
            and position_counts_match
        )
    )
    confirmed = bool(media_present and tensor_facts and counts_match)
    mismatch_reasons: list[str] = []
    if media_present and not tensor_facts:
        mismatch_reasons.append("processor_returned_no_pixel_tensor")
    if media_present and observed_samples != expected_samples:
        mismatch_reasons.append(
            f"processor_sample_count_mismatch:expected={expected_samples}:observed={observed_samples}"
        )
    if media_present and content_image_count != expected_samples and content_video_count != 1:
        mismatch_reasons.append(
            "chat_template_media_count_mismatch:"
            f"expected_samples={expected_samples}:images={content_image_count}:videos={content_video_count}"
        )
    if media_present and not media_tokens_present and not media_positions_present:
        mismatch_reasons.append("processor_returned_no_media_tokens_or_position_ids")
    if media_present and relevant_token_id_known and not media_token_coverage:
        mismatch_reasons.append(
            "processor_media_token_coverage_mismatch:"
            f"images={image_media_tokens}/{content_image_count}:"
            f"videos={video_media_tokens}/{content_video_count}"
        )
    if media_present and media_positions_present and not position_counts_match:
        mismatch_reasons.append(
            "processor_position_sample_count_mismatch:"
            f"expected={expected_samples}:observed={observed_position_samples}"
        )
    return {
        "model_input_keys": keys,
        "pixel_tensors": tensor_facts,
        "media_position_tensors": media_position_facts,
        "expected_sample_count": expected_samples,
        "processor_observed_sample_count": observed_samples,
        "processor_observed_position_sample_count": observed_position_samples,
        "chat_image_placeholder_count": content_image_count,
        "chat_video_placeholder_count": content_video_count,
        "image_media_token_count": image_media_tokens,
        "video_media_token_count": video_media_tokens,
        "media_token_coverage_confirmed": media_token_coverage,
        "position_sample_coverage_confirmed": position_counts_match,
        "effective_visual_token_budget": effective_visual_token_budget,
        "processor_pixel_patch_count": processor_pixel_patch_count,
        "pixel_transport_confirmed": confirmed,
        "not_applicable": not media_present,
        "mismatch_reasons": mismatch_reasons,
    }


def _guarded_sampling_kwargs(profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = "full_48_diagnostic" if profile == "full_48_diagnostic" else "checkpoint_defaults"
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import EntropyBoundSamplerConfig

    adaptive_stopping = normalized == "checkpoint_defaults"
    settings = {
        "profile": normalized,
        "max_denoising_steps": 48,
        "temperature_start": 0.8,
        "temperature_end": 0.4,
        "entropy_bound": 0.1,
        "stability_threshold": 1,
        "confidence_threshold": 0.005,
        "adaptive_stopping": adaptive_stopping,
    }
    kwargs: dict[str, Any] = {
        "max_denoising_steps": 48,
        "t_min": 0.4,
        "t_max": 0.8,
        "sampler_config": EntropyBoundSamplerConfig(entropy_bound=0.1),
        "stability_threshold": 1 if adaptive_stopping else None,
        "confidence_threshold": 0.005 if adaptive_stopping else None,
    }
    return kwargs, settings


def _telemetry_integrity_errors(telemetry: Any) -> list[str]:
    """Detect silent callback/API drift in a supposedly enabled telemetry run."""

    if not isinstance(telemetry, dict):
        return ["telemetry_summary_missing"]
    errors: list[str] = []
    if telemetry.get("schema_version") != GROUNDING_TELEMETRY_SCHEMA_VERSION:
        errors.append("telemetry_schema_missing_or_unsupported")
    if int(telemetry.get("forward_count", 0) or 0) <= 0:
        errors.append("logits_processor_callback_not_observed")
    snapshots = telemetry.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        errors.append("draft_streamer_snapshot_not_observed")
    if telemetry.get("stream_ended") is not True:
        errors.append("draft_streamer_end_not_observed")
    if not isinstance(telemetry.get("final_output"), dict):
        errors.append("draft_streamer_final_output_missing")
    return errors


def _run_transformers_detailed(
    config: RuntimeConfig,
    prompt: str,
    media_context: MediaContext | None,
    max_new_tokens: int,
    options: BackendRunOptions,
    node_id: str | None = None,
) -> BackendRunResult:
    """Run DiffusionGemma with transport proof and passive per-call telemetry."""

    run_start = time.perf_counter()
    processor, model = _load_transformers_model(config)
    media_message_start = time.perf_counter()
    messages, processor_kwargs = _transformers_messages_and_processor_kwargs(
        prompt,
        media_context,
        visual_first=bool(options.visual_first),
    )
    media_message_seconds = time.perf_counter() - media_message_start
    processor_start = time.perf_counter()
    template_kwargs: dict[str, Any] = {}
    if options.thinking_mode == "on":
        template_kwargs["enable_thinking"] = True
    elif options.thinking_mode == "off":
        template_kwargs["enable_thinking"] = False
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        processor_kwargs=processor_kwargs,
        **template_kwargs,
    )
    processor_seconds = time.perf_counter() - processor_start
    _director_metric_add(
        "media_preprocess_seconds",
        media_message_seconds + processor_seconds,
    )
    input_tokens = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
    transport = _processor_transport_proof(inputs, model, processor, messages, options.asset_registry)
    transport["processor_seconds"] = round(processor_seconds, 6)
    if options.require_transport and not transport.get("not_applicable") and not transport.get(
        "pixel_transport_confirmed"
    ):
        details = ",".join(transport.get("mismatch_reasons", [])) or "unconfirmed"
        raise VisualTransportError(f"visual_transport_error:{details}", transport)

    import torch

    input_transfer_start = time.perf_counter()
    inputs = {
        key: value.to(model.device) if isinstance(value, torch.Tensor) and hasattr(model, "device") else value
        for key, value in inputs.items()
    }
    input_transfer_seconds = time.perf_counter() - input_transfer_start
    _director_metric_add("media_preprocess_seconds", input_transfer_seconds)
    sampling_kwargs, effective_sampling = _guarded_sampling_kwargs(options.sampling_profile)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        **sampling_kwargs,
    }
    collector = None
    telemetry_setup_error = ""
    if options.enable_telemetry:
        try:
            collector, logits_observer, draft_streamer = build_diffusiongemma_telemetry(
                initial_input_length=input_tokens,
                max_denoising_steps=int(effective_sampling["max_denoising_steps"]),
                canvas_size=int(getattr(getattr(model, "config", None), "canvas_length", 256) or 256),
                tokenizer=getattr(processor, "tokenizer", processor),
            )
            from transformers import LogitsProcessorList

            generate_kwargs["logits_processor"] = LogitsProcessorList([logits_observer])
            generate_kwargs["streamer"] = draft_streamer
        except Exception as exc:
            telemetry_setup_error = str(exc)
            if options.fail_closed_telemetry:
                raise RuntimeError(f"visual_grounding_telemetry_error:{exc}") from exc

    seed = int(options.seed) & ((1 << 64) - 1)

    def do_generate() -> Any:
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                return model.generate(**inputs, **generate_kwargs)

    prefill_timer = _MultimodalPrefillTimer(model, _processor_inputs_have_pixels(inputs))
    generate_start = time.perf_counter()
    try:
        generated = _run_blocking_with_progress(
            "DiffusionGemma guarded generate",
            do_generate,
            node_id=node_id,
            estimated_seconds=max(30.0, min(300.0, 45.0 + (float(max_new_tokens) * 0.12))),
        )
    finally:
        generate_seconds = time.perf_counter() - generate_start
        prefill_seconds, prefill_measured = prefill_timer.finish()
        _director_metric_add("generation_seconds", generate_seconds)
        if prefill_measured:
            _director_metric_add("multimodal_prefill_seconds", prefill_seconds)
            _director_metric_set("multimodal_prefill_is_separately_measured", True)
    sequences = getattr(generated, "sequences", generated)
    decode_start = time.perf_counter()
    native_decoded_text = ""
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    if hasattr(processor, "batch_decode"):
        decode_sequences = sequences[:, input_len:]
        try:
            decoded_text = processor.batch_decode(decode_sequences, skip_special_tokens=True)[0]
        except Exception:
            decode_sequences = sequences
            decoded_text = processor.batch_decode(decode_sequences, skip_special_tokens=True)[0]
        try:
            native_decoded_text = processor.batch_decode(
                decode_sequences,
                skip_special_tokens=False,
            )[0]
        except Exception:
            native_decoded_text = ""
    elif hasattr(processor, "decode"):
        decoded_text = processor.decode(sequences[0], skip_special_tokens=True)
        try:
            native_decoded_text = processor.decode(sequences[0], skip_special_tokens=False)
        except Exception:
            native_decoded_text = ""
    else:
        decoded_text = str(sequences)
        native_decoded_text = decoded_text
    decode_seconds = time.perf_counter() - decode_start
    generated_tokens, generated_characters = _record_generated_output(
        sequences,
        input_len,
        processor,
        str(decoded_text),
    )

    telemetry: dict[str, Any] = {}
    if collector is not None:
        try:
            telemetry = collector.summary()
        except Exception as exc:
            if options.fail_closed_telemetry:
                raise RuntimeError(f"visual_grounding_telemetry_error:{exc}") from exc
            telemetry = {"errors": [{"stage": "summary", "message": str(exc)}]}
    if telemetry_setup_error:
        telemetry.setdefault("errors", []).append(
            {"stage": "setup", "message": telemetry_setup_error}
        )
    if options.enable_telemetry:
        existing_integrity_messages = {
            str(item.get("message", ""))
            for item in telemetry.get("errors", [])
            if isinstance(item, dict)
        }
        for message in _telemetry_integrity_errors(telemetry):
            if message not in existing_integrity_messages:
                telemetry.setdefault("errors", []).append(
                    {"stage": "integrity", "message": message}
                )
    if options.fail_closed_telemetry and telemetry.get("errors"):
        raise RuntimeError(
            "visual_grounding_telemetry_error:" + _json_dumps(telemetry.get("errors"))
        )

    tokens_per_forward = getattr(generated, "tokens_per_forward", None)
    try:
        tokens_per_forward_value = tokens_per_forward.detach().cpu().tolist()
    except Exception:
        tokens_per_forward_value = []
    total_seconds = time.perf_counter() - run_start
    run_result = BackendRunResult(
        decoded_text=str(decoded_text),
        native_decoded_text=str(native_decoded_text),
        stage=str(options.stage or "backend"),
        transport=transport,
        telemetry=telemetry,
        effective_sampling=effective_sampling,
        timing={
            "media_message_seconds": round(media_message_seconds, 6),
            "processor_seconds": round(processor_seconds, 6),
            "input_transfer_seconds": round(input_transfer_seconds, 6),
            "multimodal_prefill_seconds": round(prefill_seconds, 6),
            "multimodal_prefill_measured": bool(prefill_measured),
            "generation_seconds": round(generate_seconds, 6),
            "decode_seconds": round(decode_seconds, 6),
            "total_seconds": round(total_seconds, 6),
        },
        forward_counts={
            "decoder_forward_count": int(telemetry.get("forward_count", 0) or 0),
            "tokens_per_forward": tokens_per_forward_value,
        },
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        generated_characters=generated_characters,
    )
    _director_metric_append_call(
        {
            "stage": run_result.stage,
            "max_new_tokens": int(max_new_tokens),
            "input_token_count": run_result.input_tokens,
            "generated_token_count": run_result.generated_tokens,
            "generated_character_count": run_result.generated_characters,
            **copy.deepcopy(run_result.timing),
        }
    )
    return run_result


def _run_gguf_subprocess(config: RuntimeConfig, prompt: str, max_new_tokens: int, node_id: str | None = None) -> str:
    if not config.cli_path or not Path(config.cli_path).exists():
        raise RuntimeError("GGUF backend requires a valid llama-diffusion-cli path.")
    if not Path(config.model_path).exists():
        raise RuntimeError("GGUF backend requires model_path to point at a .gguf file.")
    cmd = [config.cli_path, "-m", config.model_path, "-p", prompt, "-n", str(max_new_tokens)]
    cmd.extend(_split_extra_args(config.extra_args))
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    _dg_log("GGUF subprocess started. max_new_tokens=%s", max_new_tokens)

    def run_subprocess() -> Any:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=_subprocess_env_for_cli(config.cli_path),
            creationflags=creationflags,
            timeout=900,
        )

    subprocess_start = time.perf_counter()
    result = _run_blocking_with_progress(
        "DiffusionGemma GGUF",
        run_subprocess,
        node_id=node_id,
        estimated_seconds=max(30.0, min(300.0, 45.0 + (float(max_new_tokens) * 0.12))),
    )
    if result.returncode != 0:
        raise RuntimeError(f"GGUF subprocess failed with code {result.returncode}: {result.stderr.strip()}")
    generation_seconds = time.perf_counter() - subprocess_start
    _director_metric_add("generation_seconds", generation_seconds)
    text = _strip_cli_runtime_lines(result.stdout)
    _director_metric_add("generated_character_count", len(text))
    _dg_log("GGUF subprocess finished in %.2fs.", generation_seconds)
    return text


def _run_backend(
    config: RuntimeConfig,
    prompt: str,
    media_context: MediaContext | None,
    max_new_tokens: int,
    node_id: str | None = None,
) -> str:
    if config.backend == "transformers_inprocess":
        return _run_transformers(config, prompt, media_context, max_new_tokens, node_id=node_id)
    if config.backend == "gguf_subprocess":
        return _run_gguf_subprocess(config, prompt, max_new_tokens, node_id=node_id)
    if config.backend in {"qwen_vl_placeholder", "minicpm_v_placeholder"}:
        raise RuntimeError(
            f"{config.backend} is a reserved benchmark lane. Install or wire a concrete local VLM backend before enabling runtime_required."
        )
    raise RuntimeError(f"Backend {config.backend!r} does not provide model inference.")


def _run_backend_detailed(
    config: RuntimeConfig,
    prompt: str,
    media_context: MediaContext | None,
    max_new_tokens: int,
    options: BackendRunOptions,
    node_id: str | None = None,
) -> BackendRunResult:
    if config.backend == "transformers_inprocess":
        return _run_transformers_detailed(
            config,
            prompt,
            media_context,
            max_new_tokens,
            options,
            node_id=node_id,
        )
    started = time.perf_counter()
    text = _run_backend(config, prompt, media_context, max_new_tokens, node_id=node_id)
    expected = int(max(0, _safe_float(options.asset_registry.get("expected_image_count"), 0.0)))
    transport = {
        "model_input_keys": [],
        "pixel_tensors": {},
        "expected_sample_count": expected,
        "processor_observed_sample_count": 0,
        "pixel_transport_confirmed": False,
        "not_applicable": expected == 0,
        "mismatch_reasons": ["backend_has_no_processor_pixel_transport"] if expected else [],
    }
    if options.require_transport and expected:
        raise VisualTransportError(
            "visual_transport_error:backend_has_no_processor_pixel_transport",
            transport,
        )
    run_result = BackendRunResult(
        decoded_text=text,
        stage=str(options.stage or "backend"),
        transport=transport,
        telemetry={
            "errors": [
                {
                    "stage": "backend",
                    "message": "Passive denoising telemetry is available only for in-process DiffusionGemma.",
                }
            ]
        },
        effective_sampling={
            "profile": "gguf_compatibility_scalar",
            "temperature": config.temperature,
        },
        timing={"total_seconds": round(time.perf_counter() - started, 6)},
        generated_characters=len(text),
    )
    _director_metric_append_call(
        {
            "stage": run_result.stage,
            "max_new_tokens": int(max_new_tokens),
            "input_token_count": 0,
            "generated_token_count": 0,
            "generated_character_count": run_result.generated_characters,
            **copy.deepcopy(run_result.timing),
        }
    )
    return run_result


def _maybe_unload(config: RuntimeConfig) -> None:
    if config.unload_policy != "unload_after_run":
        return
    unload_start = time.perf_counter()
    try:
        _release_transformers_runtime()
    finally:
        _director_metric_add("model_unload_seconds", time.perf_counter() - unload_start)


def _runtime_config_with_temperature(config: RuntimeConfig, temperature: float) -> RuntimeConfig:
    temp = max(0.0, min(2.0, _safe_float(temperature, config.temperature)))
    updated = RuntimeConfig(
        model_path=config.model_path,
        backend=config.backend,
        dtype=config.dtype,
        quantization=config.quantization,
        local_files_only=config.local_files_only,
        unload_policy=config.unload_policy,
        max_memory_gb=config.max_memory_gb,
        cli_path=config.cli_path,
        extra_args=_extra_args_with_temperature(config.extra_args, temp),
        temperature=temp,
        fallback_backend=config.fallback_backend,
        status=config.status.copy(),
    )
    updated.status["temperature"] = temp
    return updated


def _run_backend_for_packet(
    model_config: RuntimeConfig,
    prompt: str,
    media_context: MediaContext | None,
    max_new_tokens: int,
    node_id: str | None,
    backend_options: BackendRunOptions | None,
    backend_results: list[BackendRunResult] | None,
) -> str:
    if backend_options is None:
        return _run_backend(
            model_config,
            prompt,
            media_context,
            int(max_new_tokens),
            node_id=node_id,
        )
    call_options = copy.copy(backend_options)
    call_ordinal = (
        backend_options.call_budget.claim()
        if isinstance(backend_options.call_budget, BackendCallBudget)
        else ((len(backend_results) if isinstance(backend_results, list) else 0) + 1)
    )
    call_options.seed = (int(call_options.seed) + call_ordinal - 1) & ((1 << 64) - 1)
    base_stage = str(call_options.stage or "backend")
    existing_results = backend_results if isinstance(backend_results, list) else []
    if any(result.stage == base_stage for result in existing_results):
        repair_count = 1 + sum(
            1 for result in existing_results if result.stage.startswith("target_repair_")
        )
        call_options.stage = f"target_repair_{repair_count}"
    result = _run_backend_detailed(
        model_config,
        prompt,
        media_context,
        int(max_new_tokens),
        call_options,
        node_id=node_id,
    )
    result.stage = str(call_options.stage or "backend")
    if isinstance(backend_results, list):
        backend_results.append(result)
    return result.decoded_text


def _run_generation_packet_legacy(
    model_config: RuntimeConfig,
    gemma_context: GemmaContext,
    target_config: TargetProfileConfig | dict[str, Any],
    master_prompt: str = DEFAULT_MASTER_PROMPT,
    runtime_required: bool = False,
    max_new_tokens: int = 768,
    max_output_chars: int = 8000,
    creativity_mode: str = "editorial",
    creative_strength: float = 0.6,
    thinking_mode: str = "off",
    node_id: str | None = None,
    backend_options: BackendRunOptions | None = None,
    backend_results: list[BackendRunResult] | None = None,
    manage_unload: bool = True,
    max_refinement_attempts_override: int | None = None,
    refinement_evidence_context: str = "",
    deterministic_transport_repair: bool = False,
    native_h3_output: bool = False,
) -> tuple[dict[str, Any], str, str, str, bool, str, dict[str, Any]]:
    total_start = time.perf_counter()
    if not isinstance(model_config, RuntimeConfig):
        raise TypeError("model_config must come from DiffusionGemma Model Loader.")
    context = gemma_context if isinstance(gemma_context, GemmaContext) else _gemma_context_from_media("")
    target = _target_profile_config_to_dict(target_config)
    context = _context_with_minimax_h3_reference_policy(context, target)
    if (
        target["target_profile"] == "minimax_h3"
        and target["audio_mode"] == "visual_only"
        and target["minimax_h3_dialogue_mode"] == "required"
    ):
        raise ValueError(
            "minimax_h3_dialogue_audio_mode_conflict: dialogue_mode=required cannot be used with audio_mode=visual_only"
        )
    creativity_mode = _normalize_creativity_mode(creativity_mode)
    creative_strength = _normalize_creative_strength(creative_strength)
    thinking_mode = _effective_target_thinking_mode(
        thinking_mode,
        target["target_profile"],
    )
    media_notes = _media_policy_notes(model_config, context)
    media_metadata = context.media_metadata.copy()
    if target["target_profile"] == "minimax_h3":
        media_metadata["minimax_h3_mode"] = target["minimax_h3_mode"]
        if target["minimax_h3_mode"] == "ref2va":
            media_metadata["minimax_h3_reference_manifest"] = _normalize_minimax_h3_reference_manifest(
                str(media_metadata.get("minimax_h3_reference_manifest", ""))
            )
            media_metadata["minimax_h3_expected_subject_count"] = (
                _normalize_minimax_h3_expected_subject_count(
                    media_metadata.get("minimax_h3_expected_subject_count", 0)
                )
            )
    minimax_h3_expected_subject_count = _normalize_minimax_h3_expected_subject_count(
        media_metadata.get("minimax_h3_expected_subject_count", 0)
    )
    pixels_present = bool(context.images is not None)
    pixels_sent = bool(pixels_present and _runtime_supports_pixels(model_config))
    visual_description = _clean_visual_description(context.visual_description or str(media_metadata.get("visual_description", "")))
    media_metadata["pixel_tensor_present"] = pixels_present
    media_metadata["pixels_sent_to_backend"] = pixels_sent
    if pixels_sent and str(media_metadata.get("source", "none")).lower() in {"video", "image+video"}:
        media_metadata["transformers_video_transport"] = (
            "sampled_frame_images"
            if _is_image_identity_video_control(media_metadata) or _is_minimax_h3_reference_context(media_metadata)
            else _video_transport_mode()
        )
    if visual_description:
        media_metadata["visual_description"] = visual_description
        media_metadata.setdefault("visual_description_source", "upstream_text")
    media_metadata["visual_grounding_mode"] = _media_grounding_mode(media_metadata)
    warnings = list(media_metadata.get("warnings", [])) if isinstance(media_metadata.get("warnings"), list) else []
    for note in media_notes:
        if note not in warnings:
            warnings.append(note)
    if pixels_present and not pixels_sent and not visual_description:
        note = "No visual_description was supplied; the active backend cannot inspect image/video pixels and will use only text plus metadata."
        if note not in warnings:
            warnings.append(note)
    media_metadata["warnings"] = warnings
    working_context = GemmaContext(
        user_prompt=context.user_prompt,
        images=context.images,
        source=context.source,
        media_metadata=media_metadata,
        visual_description=visual_description,
        warnings=warnings,
    )
    model_prompt = _build_model_prompt(
        working_context.user_prompt,
        master_prompt,
        target["target_profile"],
        media_metadata,
        target["audio_mode"],
        target["audio_guidance"],
        target["target_duration_seconds"],
        target["ltx_style"],
        target["ideogram_aspect_ratio"],
        target["ideogram_render_style"],
        target["ideogram_exact_text"],
        target["ideogram_json_output"],
        creativity_mode,
        creative_strength,
        target["negative_prompt_mode"],
        target["negative_prompt_guidance"],
        thinking_mode,
        target["minimax_h3_mode"],
        str(media_metadata.get("minimax_h3_reference_manifest", "")),
        target["minimax_h3_shot_count"],
        native_h3_output,
        target["minimax_h3_dialogue_mode"],
        target["minimax_h3_dialogue_line_count"],
        target["minimax_h3_dialogue_guidance"],
        target["ltx_generation_mode"],
        target["ltx_long_horizon_mode"],
        target["ltx_camera_capability"],
    )
    raw_output = ""
    answer_text = ""
    reasoning_text = ""
    fallback_reason = ""
    packet: dict[str, Any]
    parse_valid = False
    salvage_warning = ""
    backend_seconds = 0.0
    postprocess_seconds = 0.0
    refinement_backend_seconds = 0.0
    refinement_report: dict[str, Any] = {}
    try:
        if target["target_profile"] == "minimax_h3" and target["minimax_h3_mode"] == "ref2va":
            manifest_reasons = _minimax_h3_reference_manifest_validation_reasons(
                str(media_metadata.get("minimax_h3_reference_manifest", ""))
            )
            if manifest_reasons:
                raise ValueError("MiniMax H3 Ref2VA manifest is invalid: " + ", ".join(manifest_reasons))
        if model_config.backend == "template":
            raise RuntimeError("Template backend selected.")
        backend_media_context = _media_context_from_gemma_context(working_context) if _runtime_supports_pixels(model_config) else None
        backend_start = time.perf_counter()
        _dg_log("Backend dispatch. backend=%s pixels_sent=%s", model_config.backend, bool(backend_media_context))
        raw_output = _run_backend_for_packet(
            model_config,
            model_prompt,
            backend_media_context,
            int(max_new_tokens),
            node_id,
            backend_options,
            backend_results,
        )
        backend_seconds = time.perf_counter() - backend_start
        _dg_log("Backend returned in %.2fs. raw_output_chars=%s", backend_seconds, len(raw_output))
        postprocess_start = time.perf_counter()
        if native_h3_output:
            answer_text = raw_output
            reasoning_text = ""
        else:
            reasoning_text, answer_text = _split_reasoning_and_answer(raw_output)
        packet, parse_valid, salvage_warning = repair_or_salvage_prompt_packet(
            answer_text,
            working_context,
            target,
            int(max_output_chars),
            creativity_mode,
            creative_strength,
            deterministic_transport_repair,
            native_h3_output,
        )
        if target["target_profile"] == "minimax_h3" and parse_valid:
            h3_duration = _minimax_h3_duration_seconds(
                media_metadata,
                target["target_duration_seconds"],
                working_context.user_prompt,
            )
            initial_h3_prompt = str(packet.get("minimax_h3_prompt", "") or "")
            initial_h3_reasons = _minimax_h3_prompt_validation_reasons(
                initial_h3_prompt,
                h3_duration,
                working_context.user_prompt,
                target["audio_mode"],
                int(max_output_chars),
                target["minimax_h3_mode"],
                str(media_metadata.get("minimax_h3_reference_manifest", "")),
                target["minimax_h3_shot_count"],
                target["minimax_h3_dialogue_mode"],
                target["minimax_h3_dialogue_line_count"],
                target["minimax_h3_dialogue_guidance"],
                minimax_h3_expected_subject_count,
            )
            default_refinement_attempts = 2 if target["minimax_h3_mode"] == "ref2va" else 1
            max_refinement_attempts = (
                default_refinement_attempts
                if max_refinement_attempts_override is None
                else max(0, min(2, int(max_refinement_attempts_override)))
            )

            def attempt_deterministic_ref2va_speaker_patch(
                candidate_prompt: str,
                candidate_reasons: list[str],
            ) -> tuple[str, dict[str, Any], bool]:
                patch_report: dict[str, Any] = {
                    "attempted": True,
                    "accepted": False,
                    "source": "deterministic_unambiguous_subject_anchor",
                    "trigger_reasons": list(candidate_reasons),
                }
                if set(candidate_reasons) != {
                    "minimax_h3_dialogue_speaker_invalid"
                }:
                    patch_report["last_reasons"] = [
                        "ref2va_deterministic_speaker_patch_trigger_invalid"
                    ]
                    return candidate_prompt, patch_report, False
                verified_ledger = (
                    media_metadata.get("verified_grounding_ledger")
                    if isinstance(media_metadata.get("verified_grounding_ledger"), dict)
                    else {}
                )
                evidence_report_id = str(
                    media_metadata.get("grounding_evidence_report_id", "")
                )
                reference_manifest = str(
                    media_metadata.get("minimax_h3_reference_manifest", "")
                )
                preflight_reasons = (
                    _minimax_h3_ref2va_contract_patch_feasibility_reasons(
                        candidate_prompt,
                        candidate_reasons,
                        reference_manifest,
                        verified_ledger,
                    )
                )
                if not evidence_report_id:
                    preflight_reasons = [
                        *preflight_reasons,
                        "ref2va_contract_patch_evidence_report_missing",
                    ]
                if preflight_reasons:
                    patch_report["last_reasons"] = list(
                        dict.fromkeys(preflight_reasons)
                    )
                    return candidate_prompt, patch_report, False
                missing_anchors = [
                    anchor
                    for anchor in _minimax_h3_ref2va_dialogue_anchors(
                        candidate_prompt
                    )
                    if anchor["speaker_missing"]
                ]
                payload = {
                    "schema": _MINIMAX_H3_REF2VA_CONTRACT_PATCH_SCHEMA,
                    "evidence_report_id": evidence_report_id,
                    "subject_definitions": [],
                    "dialogue_speakers": [
                        {
                            "shot": anchor["shot"],
                            "dialogue_sha256": anchor["dialogue_sha256"],
                            "subject_tag": anchor["candidate_subject_tags"][0],
                        }
                        for anchor in missing_anchors
                    ],
                }
                patch_plan, validation_reasons = (
                    _validated_minimax_h3_ref2va_contract_patch(
                        _json_dumps(payload),
                        candidate_prompt,
                        reference_manifest,
                        evidence_report_id,
                        verified_ledger,
                        candidate_reasons,
                    )
                )
                if validation_reasons:
                    patch_report["last_reasons"] = list(validation_reasons)
                    return candidate_prompt, patch_report, False
                patched_prompt, operations, apply_reasons = (
                    _apply_minimax_h3_ref2va_contract_patch(
                        candidate_prompt,
                        patch_plan,
                    )
                )
                patched_reasons = _minimax_h3_prompt_validation_reasons(
                    patched_prompt,
                    h3_duration,
                    working_context.user_prompt,
                    target["audio_mode"],
                    int(max_output_chars),
                    target["minimax_h3_mode"],
                    reference_manifest,
                    target["minimax_h3_shot_count"],
                    target["minimax_h3_dialogue_mode"],
                    target["minimax_h3_dialogue_line_count"],
                    target["minimax_h3_dialogue_guidance"],
                    minimax_h3_expected_subject_count,
                )
                final_reasons = list(
                    dict.fromkeys([*apply_reasons, *patched_reasons])
                )
                if final_reasons:
                    patch_report["last_reasons"] = final_reasons
                    return candidate_prompt, patch_report, False
                patch_report.update(
                    {
                        "accepted": True,
                        "operations": operations,
                        "grounding_fact_ids": [],
                        "compiled_prompt_sha256": hashlib.sha256(
                            patched_prompt.encode("utf-8")
                        ).hexdigest(),
                    }
                )
                return patched_prompt, patch_report, True

            def attempt_ref2va_contract_patch(
                candidate_prompt: str,
                candidate_reasons: list[str],
                attempt_limit: int,
            ) -> tuple[str, dict[str, Any], bool]:
                nonlocal backend_seconds, refinement_backend_seconds
                patch_report: dict[str, Any] = {
                    "attempted": True,
                    "accepted": False,
                    "trigger_reasons": list(candidate_reasons),
                    "attempts": [],
                }
                verified_ledger = (
                    media_metadata.get("verified_grounding_ledger")
                    if isinstance(media_metadata.get("verified_grounding_ledger"), dict)
                    else {}
                )
                evidence_report_id = str(
                    media_metadata.get("grounding_evidence_report_id", "")
                )
                reference_manifest = str(
                    media_metadata.get("minimax_h3_reference_manifest", "")
                )
                base_patch_prompt = _build_minimax_h3_ref2va_contract_patch_prompt(
                    working_context.user_prompt,
                    candidate_prompt,
                    reference_manifest,
                    evidence_report_id,
                    verified_ledger,
                    candidate_reasons,
                )
                retry_reasons: list[str] = []
                for patch_index in range(max(0, min(2, int(attempt_limit)))):
                    patch_prompt = base_patch_prompt
                    if retry_reasons:
                        patch_prompt = (
                            f"{base_patch_prompt}\n\nRETRY CORRECTION: The previous sidecar was rejected for "
                            f"{_json_dumps(retry_reasons)}. Return a fresh complete sidecar fixing only those strict schema or evidence-selection defects."
                        )
                    attempt_report: dict[str, Any] = {
                        "attempt": patch_index + 1,
                        "stage": f"ref2va_contract_patch_{patch_index + 1}",
                        "prompt_chars": len(patch_prompt),
                    }
                    patch_report["attempts"].append(attempt_report)
                    patch_options = (
                        copy.copy(backend_options)
                        if backend_options is not None
                        else BackendRunOptions()
                    )
                    patch_options.stage = f"ref2va_contract_patch_{patch_index + 1}"
                    patch_start = time.perf_counter()
                    try:
                        patch_raw = _run_backend_for_packet(
                            model_config,
                            patch_prompt,
                            None,
                            min(int(max_new_tokens), 768),
                            node_id,
                            patch_options,
                            backend_results,
                        )
                        attempt_report["output_sha256"] = hashlib.sha256(
                            patch_raw.encode("utf-8")
                        ).hexdigest()
                        patch_plan, validation_reasons = (
                            _validated_minimax_h3_ref2va_contract_patch(
                                patch_raw,
                                candidate_prompt,
                                reference_manifest,
                                evidence_report_id,
                                verified_ledger,
                                candidate_reasons,
                            )
                        )
                        retry_reasons = list(validation_reasons)
                        attempt_report["patch_validation_reasons"] = list(
                            validation_reasons
                        )
                        if not validation_reasons:
                            patched_prompt, operations, apply_reasons = (
                                _apply_minimax_h3_ref2va_contract_patch(
                                    candidate_prompt,
                                    patch_plan,
                                )
                            )
                            patched_reasons = _minimax_h3_prompt_validation_reasons(
                                patched_prompt,
                                h3_duration,
                                working_context.user_prompt,
                                target["audio_mode"],
                                int(max_output_chars),
                                target["minimax_h3_mode"],
                                reference_manifest,
                                target["minimax_h3_shot_count"],
                                target["minimax_h3_dialogue_mode"],
                                target["minimax_h3_dialogue_line_count"],
                                target["minimax_h3_dialogue_guidance"],
                                minimax_h3_expected_subject_count,
                            )
                            retry_reasons = list(
                                dict.fromkeys([*apply_reasons, *patched_reasons])
                            )
                            attempt_report["patch_apply_reasons"] = list(
                                apply_reasons
                            )
                            attempt_report["candidate_reasons"] = list(
                                patched_reasons
                            )
                            if not apply_reasons and not patched_reasons:
                                attempt_report["accepted"] = True
                                patch_report.update(
                                    {
                                        "accepted": True,
                                        "accepted_attempt": patch_index + 1,
                                        "operations": operations,
                                        "grounding_fact_ids": list(
                                            patch_plan.get("grounding_fact_ids", [])
                                        ),
                                        "output_sha256": attempt_report[
                                            "output_sha256"
                                        ],
                                        "compiled_prompt_sha256": hashlib.sha256(
                                            patched_prompt.encode("utf-8")
                                        ).hexdigest(),
                                    }
                                )
                                return patched_prompt, patch_report, True
                    except Exception as patch_exc:
                        if (
                            backend_options is not None
                            and backend_options.fail_closed_telemetry
                            and "visual_grounding_telemetry_error"
                            in str(patch_exc).lower()
                        ):
                            raise
                        retry_reasons = [str(patch_exc)]
                        attempt_report["error"] = str(patch_exc)
                        patch_report["error"] = str(patch_exc)
                    finally:
                        elapsed = time.perf_counter() - patch_start
                        backend_seconds += elapsed
                        refinement_backend_seconds += elapsed
                        attempt_report["backend_seconds"] = round(elapsed, 3)
                patch_report["last_reasons"] = list(retry_reasons)
                patch_report["backend_seconds"] = round(
                    sum(
                        _safe_float(attempt.get("backend_seconds"), 0.0)
                        for attempt in patch_report["attempts"]
                    ),
                    3,
                )
                return candidate_prompt, patch_report, False

            initial_reason_set = set(initial_h3_reasons)
            initial_contract_patch_preflight_reasons = (
                _minimax_h3_ref2va_contract_patch_feasibility_reasons(
                    initial_h3_prompt,
                    initial_h3_reasons,
                    str(media_metadata.get("minimax_h3_reference_manifest", "")),
                    media_metadata.get("verified_grounding_ledger")
                    if isinstance(
                        media_metadata.get("verified_grounding_ledger"), dict
                    )
                    else {},
                )
                if target["minimax_h3_mode"] == "ref2va"
                and initial_reason_set
                and initial_reason_set.issubset(
                    _MINIMAX_H3_REF2VA_CONTRACT_LOCAL_REASONS
                )
                else []
            )
            if (
                target["minimax_h3_mode"] == "ref2va"
                and initial_reason_set == {"minimax_h3_dialogue_speaker_invalid"}
                and not initial_contract_patch_preflight_reasons
            ):
                (
                    deterministic_patched_prompt,
                    deterministic_patch_report,
                    deterministic_patch_accepted,
                ) = attempt_deterministic_ref2va_speaker_patch(
                    initial_h3_prompt,
                    initial_h3_reasons,
                )
                refinement_report = {
                    "attempted": True,
                    "accepted": deterministic_patch_accepted,
                    "repair_mode": "deterministic_unique_subject_speaker_patch",
                    "trigger_reasons": list(initial_h3_reasons),
                    "initial_reasons": list(initial_h3_reasons),
                    "deterministic_speaker_patch": deterministic_patch_report,
                    "backend_seconds": round(refinement_backend_seconds, 3),
                }
                if deterministic_patch_accepted:
                    packet["minimax_h3_prompt"] = deterministic_patched_prompt
                    packet_metadata = (
                        packet.get("metadata")
                        if isinstance(packet.get("metadata"), dict)
                        else {}
                    )
                    packet_metadata.update(
                        {
                            "minimax_h3_ref2va_contract_patch_source": "deterministic_unique_subject_speaker_patch",
                            "minimax_h3_ref2va_contract_patch_operations": deterministic_patch_report.get(
                                "operations", []
                            ),
                            "minimax_h3_ref2va_contract_compiled_prompt_sha256": deterministic_patch_report.get(
                                "compiled_prompt_sha256", ""
                            ),
                        }
                    )
                    packet["metadata"] = packet_metadata
                    initial_h3_prompt = deterministic_patched_prompt
                    initial_h3_reasons = []
                    initial_reason_set = set()
                else:
                    initial_contract_patch_preflight_reasons = list(
                        deterministic_patch_report.get(
                            "last_reasons",
                            ["ref2va_deterministic_speaker_patch_failed"],
                        )
                    )
            dialogue_only_failure = bool(initial_reason_set) and initial_reason_set.issubset(
                _MINIMAX_H3_DIALOGUE_LOCAL_REASONS
            )
            if (
                target["minimax_h3_mode"] == "ref2va"
                and initial_reason_set == {"minimax_h3_dialogue_speaker_invalid"}
                and initial_contract_patch_preflight_reasons
            ):
                dialogue_only_failure = False
            ref2va_contract_patch_attempted = False
            if (
                target["minimax_h3_mode"] == "ref2va"
                and initial_reason_set
                and initial_reason_set.issubset(
                    _MINIMAX_H3_REF2VA_CONTRACT_LOCAL_REASONS
                )
                and "minimax_h3_ref_subject_definition_invalid"
                in initial_reason_set
                and not initial_contract_patch_preflight_reasons
                and max_refinement_attempts > 0
            ):
                (
                    contract_patched_prompt,
                    contract_patch_report,
                    contract_patch_accepted,
                ) = attempt_ref2va_contract_patch(
                    initial_h3_prompt,
                    initial_h3_reasons,
                    max_refinement_attempts,
                )
                ref2va_contract_patch_attempted = True
                refinement_report = {
                    "attempted": True,
                    "accepted": contract_patch_accepted,
                    "repair_mode": "ref2va_contract_patch",
                    "trigger_reasons": list(initial_h3_reasons),
                    "initial_reasons": list(initial_h3_reasons),
                    "ref2va_contract_patch": contract_patch_report,
                    "backend_seconds": round(refinement_backend_seconds, 3),
                }
                if contract_patch_accepted:
                    packet["minimax_h3_prompt"] = contract_patched_prompt
                    packet_metadata = (
                        packet.get("metadata")
                        if isinstance(packet.get("metadata"), dict)
                        else {}
                    )
                    packet_metadata.update(
                        {
                            "minimax_h3_ref2va_contract_patch_source": "verified_ledger_host_compiled",
                            "minimax_h3_ref2va_contract_patch_operations": contract_patch_report.get(
                                "operations", []
                            ),
                            "minimax_h3_ref2va_contract_patch_output_sha256": contract_patch_report.get(
                                "output_sha256", ""
                            ),
                            "minimax_h3_ref2va_contract_compiled_prompt_sha256": contract_patch_report.get(
                                "compiled_prompt_sha256", ""
                            ),
                        }
                    )
                    packet_metadata["used_grounding_fact_ids"] = list(
                        dict.fromkeys(
                            [
                                *(
                                    packet_metadata.get("used_grounding_fact_ids", [])
                                    if isinstance(
                                        packet_metadata.get("used_grounding_fact_ids"),
                                        list,
                                    )
                                    else []
                                ),
                                *contract_patch_report.get("grounding_fact_ids", []),
                            ]
                        )
                    )
                    packet["metadata"] = packet_metadata
                    initial_h3_prompt = contract_patched_prompt
                    initial_h3_reasons = []
                    initial_reason_set = set()
                    dialogue_only_failure = False
            if (
                target["minimax_h3_dialogue_mode"] == "required"
                and initial_reason_set == {"minimax_h3_dialogue_count_mismatch"}
                and max_refinement_attempts > 0
            ):
                existing_dialogue_count = _minimax_h3_dialogue_block_count(
                    initial_h3_prompt
                )
                missing_dialogue_count = (
                    target["minimax_h3_dialogue_line_count"]
                    - existing_dialogue_count
                )
                dialogue_patch_report: dict[str, Any] = {
                    "attempted": True,
                    "accepted": False,
                    "missing_line_count": missing_dialogue_count,
                    "initial_reasons": list(initial_h3_reasons),
                    "attempts": [],
                    "transport_diagnostics": _minimax_h3_dialogue_diagnostics(
                        initial_h3_prompt,
                        target["minimax_h3_mode"],
                    ),
                }
                if missing_dialogue_count > 0:
                    base_dialogue_patch_prompt = _build_minimax_h3_dialogue_patch_prompt(
                        working_context.user_prompt,
                        initial_h3_prompt,
                        missing_dialogue_count,
                        target["minimax_h3_mode"],
                        target["minimax_h3_dialogue_guidance"],
                        refinement_evidence_context,
                    )
                    patch_token_budget = min(
                        int(max_new_tokens),
                        max(192, min(768, 128 + 96 * missing_dialogue_count)),
                    )
                    retry_reasons: list[str] = []
                    for patch_index in range(max_refinement_attempts):
                        dialogue_patch_prompt = base_dialogue_patch_prompt
                        if retry_reasons:
                            dialogue_patch_prompt = (
                                f"{base_dialogue_patch_prompt}\n\nRETRY CORRECTION: The previous sidecar was rejected for "
                                f"{_json_dumps(retry_reasons)}. Return a fresh complete sidecar that fixes only those machine-format defects."
                            )
                        attempt_report: dict[str, Any] = {
                            "attempt": patch_index + 1,
                            "stage": f"dialogue_patch_{patch_index + 1}",
                            "prompt_chars": len(dialogue_patch_prompt),
                        }
                        dialogue_patch_report["attempts"].append(attempt_report)
                        patch_options = (
                            copy.copy(backend_options)
                            if backend_options is not None
                            else BackendRunOptions()
                        )
                        patch_options.stage = f"dialogue_patch_{patch_index + 1}"
                        patch_start = time.perf_counter()
                        try:
                            dialogue_patch_raw = _run_backend_for_packet(
                                model_config,
                                dialogue_patch_prompt,
                                None,
                                patch_token_budget,
                                node_id,
                                patch_options,
                                backend_results,
                            )
                            attempt_report["output_sha256"] = hashlib.sha256(
                                dialogue_patch_raw.encode("utf-8")
                            ).hexdigest()
                            patch_records, patch_validation_reasons = (
                                _validated_minimax_h3_dialogue_patch_records(
                                    dialogue_patch_raw,
                                    initial_h3_prompt,
                                    target["minimax_h3_mode"],
                                    missing_dialogue_count,
                                )
                            )
                            retry_reasons = list(patch_validation_reasons)
                            attempt_report["patch_validation_reasons"] = list(
                                patch_validation_reasons
                            )
                            dialogue_patch_report["patch_validation_reasons"] = list(
                                patch_validation_reasons
                            )
                            if not patch_validation_reasons:
                                (
                                    patched_h3_prompt,
                                    patch_operations,
                                    patch_apply_reasons,
                                ) = _apply_minimax_h3_dialogue_patch_records(
                                    initial_h3_prompt,
                                    target["minimax_h3_mode"],
                                    patch_records,
                                )
                                patched_reasons = _minimax_h3_prompt_validation_reasons(
                                    patched_h3_prompt,
                                    h3_duration,
                                    working_context.user_prompt,
                                    target["audio_mode"],
                                    int(max_output_chars),
                                    target["minimax_h3_mode"],
                                    str(
                                        media_metadata.get(
                                            "minimax_h3_reference_manifest",
                                            "",
                                        )
                                    ),
                                    target["minimax_h3_shot_count"],
                                    target["minimax_h3_dialogue_mode"],
                                    target["minimax_h3_dialogue_line_count"],
                                    target["minimax_h3_dialogue_guidance"],
                                    minimax_h3_expected_subject_count,
                                )
                                new_non_dialogue_reasons = [
                                    reason
                                    for reason in patched_reasons
                                    if reason not in initial_reason_set
                                    and reason not in _MINIMAX_H3_DIALOGUE_LOCAL_REASONS
                                ]
                                retry_reasons = list(
                                    dict.fromkeys(
                                        [
                                            *patch_apply_reasons,
                                            *patched_reasons,
                                            *new_non_dialogue_reasons,
                                        ]
                                    )
                                )
                                attempt_report["patch_apply_reasons"] = list(
                                    patch_apply_reasons
                                )
                                attempt_report["candidate_reasons"] = list(
                                    patched_reasons
                                )
                                attempt_report["new_non_dialogue_reasons"] = list(
                                    new_non_dialogue_reasons
                                )
                                dialogue_patch_report["patch_apply_reasons"] = list(
                                    patch_apply_reasons
                                )
                                dialogue_patch_report["candidate_reasons"] = list(
                                    patched_reasons
                                )
                                dialogue_patch_report["new_non_dialogue_reasons"] = list(
                                    new_non_dialogue_reasons
                                )
                                if (
                                    not patch_apply_reasons
                                    and not patched_reasons
                                    and not new_non_dialogue_reasons
                                ):
                                    packet["minimax_h3_prompt"] = patched_h3_prompt
                                    packet_metadata = (
                                        packet.get("metadata")
                                        if isinstance(packet.get("metadata"), dict)
                                        else {}
                                    )
                                    packet_metadata.update(
                                        {
                                            "minimax_h3_dialogue_patch_source": "model_authored_host_compiled",
                                            "minimax_h3_dialogue_patch_inserted_count": len(
                                                patch_operations
                                            ),
                                            "minimax_h3_dialogue_patch_shots": [
                                                operation["shot"]
                                                for operation in patch_operations
                                            ],
                                            "minimax_h3_dialogue_patch_operations": patch_operations,
                                            "minimax_h3_dialogue_patch_output_sha256": attempt_report[
                                                "output_sha256"
                                            ],
                                            "minimax_h3_dialogue_compiled_prompt_sha256": hashlib.sha256(
                                                patched_h3_prompt.encode("utf-8")
                                            ).hexdigest(),
                                        }
                                    )
                                    packet["metadata"] = packet_metadata
                                    initial_h3_prompt = patched_h3_prompt
                                    initial_h3_reasons = []
                                    initial_reason_set = set()
                                    dialogue_only_failure = False
                                    attempt_report["accepted"] = True
                                    dialogue_patch_report["accepted"] = True
                                    dialogue_patch_report["accepted_attempt"] = (
                                        patch_index + 1
                                    )
                                    dialogue_patch_report["inserted_count"] = len(
                                        patch_operations
                                    )
                                    dialogue_patch_report["shots"] = [
                                        operation["shot"]
                                        for operation in patch_operations
                                    ]
                                    break
                        except Exception as dialogue_patch_exc:
                            if (
                                backend_options is not None
                                and backend_options.fail_closed_telemetry
                                and "visual_grounding_telemetry_error"
                                in str(dialogue_patch_exc).lower()
                            ):
                                raise
                            retry_reasons = [str(dialogue_patch_exc)]
                            attempt_report["error"] = str(dialogue_patch_exc)
                            dialogue_patch_report["error"] = str(dialogue_patch_exc)
                        finally:
                            patch_backend_seconds = time.perf_counter() - patch_start
                            refinement_backend_seconds += patch_backend_seconds
                            backend_seconds += patch_backend_seconds
                            attempt_report["backend_seconds"] = round(
                                patch_backend_seconds,
                                3,
                            )
                    dialogue_patch_report["backend_seconds"] = round(
                        sum(
                            _safe_float(attempt.get("backend_seconds"), 0.0)
                            for attempt in dialogue_patch_report["attempts"]
                        ),
                        3,
                    )
                    refinement_report = {
                        "attempted": True,
                        "accepted": bool(dialogue_patch_report.get("accepted")),
                        "repair_mode": "dialogue_patch",
                        "trigger_reasons": ["minimax_h3_dialogue_count_mismatch"],
                        "initial_reasons": ["minimax_h3_dialogue_count_mismatch"],
                        "dialogue_patch": dialogue_patch_report,
                        "backend_seconds": round(refinement_backend_seconds, 3),
                    }
            refinable_reasons = [
                reason
                for reason in initial_h3_reasons
                if reason in _MINIMAX_H3_REFINABLE_REASONS
                and not (
                    dialogue_only_failure
                    and reason in _MINIMAX_H3_DIALOGUE_LOCAL_REASONS
                )
                and not (
                    ref2va_contract_patch_attempted
                    and reason in _MINIMAX_H3_REF2VA_CONTRACT_LOCAL_REASONS
                )
            ]
            if refinable_reasons and max_refinement_attempts > 0:
                refinement_report = {
                    "attempted": True,
                    "accepted": False,
                    "trigger_reasons": refinable_reasons,
                    "initial_reasons": initial_h3_reasons,
                    "attempts": [],
                }
                refinement_candidate_prompt = initial_h3_prompt
                refinement_candidate_reasons = initial_h3_reasons
                refinement_base = "initial"
                best_candidate_prompt = initial_h3_prompt
                best_candidate_reasons = initial_h3_reasons
                best_candidate_rank = _minimax_h3_refinement_candidate_rank(
                    initial_h3_prompt,
                    initial_h3_reasons,
                    parse_valid,
                    salvage_warning,
                )
                best_candidate_attempt = 0
                for refinement_index in range(max_refinement_attempts):
                    refinement_prompt = _build_minimax_h3_refinement_prompt(
                        working_context.user_prompt,
                        refinement_candidate_prompt,
                        h3_duration,
                        target["audio_mode"],
                        refinement_candidate_reasons,
                        target["minimax_h3_mode"],
                        str(media_metadata.get("minimax_h3_reference_manifest", "")),
                        strict_retry=refinement_index > 0,
                        minimax_h3_shot_count=target["minimax_h3_shot_count"],
                        native_prompt_only=native_h3_output,
                        minimax_h3_dialogue_mode=target["minimax_h3_dialogue_mode"],
                        minimax_h3_dialogue_line_count=target["minimax_h3_dialogue_line_count"],
                        minimax_h3_dialogue_guidance=target["minimax_h3_dialogue_guidance"],
                        minimax_h3_expected_subject_count=minimax_h3_expected_subject_count,
                    )
                    if refinement_evidence_context:
                        refinement_prompt = (
                            f"{refinement_prompt}\n\nVERIFIED VISUAL EVIDENCE LEDGER (the sole visual evidence for this repair):\n"
                            f"{refinement_evidence_context}"
                        )
                    if native_h3_output:
                        refinement_ledger = (
                            media_metadata.get("verified_grounding_ledger")
                            if isinstance(
                                media_metadata.get("verified_grounding_ledger"), dict
                            )
                            else {}
                        )
                        refinement_fact_ids = list(
                            _eligible_grounding_fact_assets(refinement_ledger)
                        )
                        refinement_prompt = (
                            f"{refinement_prompt}\n\nAfter the rewritten non_diegetic_music section, write one blank line and then exactly these two final machine lines. "
                            "List only the available fact IDs whose reference-derived content the rewrite actually uses; the list must cover every required reference asset:\n"
                            f"GROUNDING_EVIDENCE_REPORT_ID: {media_metadata.get('grounding_evidence_report_id', '')}\n"
                            f"USED_GROUNDING_FACT_IDS: {','.join(refinement_fact_ids)}"
                        )
                    if refinement_index == 0:
                        refinement_report["prompt_chars"] = len(refinement_prompt)
                    attempt_report = {
                        "attempt": refinement_index + 1,
                        "base": refinement_base,
                        "prompt_chars": len(refinement_prompt),
                        "trigger_reasons": refinement_candidate_reasons,
                    }
                    refinement_report["attempts"].append(attempt_report)
                    try:
                        refinement_start = time.perf_counter()
                        try:
                            refined_raw_output = _run_backend_for_packet(
                                model_config,
                                refinement_prompt,
                                backend_media_context,
                                int(max_new_tokens),
                                node_id,
                                backend_options,
                                backend_results,
                            )
                        finally:
                            attempt_backend_seconds = time.perf_counter() - refinement_start
                            refinement_backend_seconds += attempt_backend_seconds
                            backend_seconds += attempt_backend_seconds
                            attempt_report["backend_seconds"] = round(attempt_backend_seconds, 3)
                        if native_h3_output:
                            refined_answer = refined_raw_output
                            refined_reasoning = ""
                        else:
                            refined_reasoning, refined_answer = (
                                _split_reasoning_and_answer(refined_raw_output)
                            )
                        refined_packet, refined_parse_valid, refined_salvage_warning = repair_or_salvage_prompt_packet(
                            refined_answer or refined_raw_output,
                            working_context,
                            target,
                            int(max_output_chars),
                            creativity_mode,
                            creative_strength,
                            deterministic_transport_repair,
                            native_h3_output,
                        )
                        refined_h3_prompt = str(refined_packet.get("minimax_h3_prompt", "") or "")
                        refined_h3_reasons = _minimax_h3_prompt_validation_reasons(
                            refined_h3_prompt,
                            h3_duration,
                            working_context.user_prompt,
                            target["audio_mode"],
                            int(max_output_chars),
                            target["minimax_h3_mode"],
                            str(media_metadata.get("minimax_h3_reference_manifest", "")),
                            target["minimax_h3_shot_count"],
                            target["minimax_h3_dialogue_mode"],
                            target["minimax_h3_dialogue_line_count"],
                            target["minimax_h3_dialogue_guidance"],
                            minimax_h3_expected_subject_count,
                        )
                        contract_patch_failed = False
                        remaining_refinement_attempts = max_refinement_attempts - (
                            refinement_index + 1
                        )
                        refined_reason_set = set(refined_h3_reasons)
                        refined_contract_patch_preflight_reasons = (
                            _minimax_h3_ref2va_contract_patch_feasibility_reasons(
                                refined_h3_prompt,
                                refined_h3_reasons,
                                str(
                                    media_metadata.get(
                                        "minimax_h3_reference_manifest", ""
                                    )
                                ),
                                media_metadata.get("verified_grounding_ledger")
                                if isinstance(
                                    media_metadata.get("verified_grounding_ledger"),
                                    dict,
                                )
                                else {},
                            )
                            if target["minimax_h3_mode"] == "ref2va"
                            and refined_reason_set
                            and refined_reason_set.issubset(
                                _MINIMAX_H3_REF2VA_CONTRACT_LOCAL_REASONS
                            )
                            else []
                        )
                        if (
                            target["minimax_h3_mode"] == "ref2va"
                            and refined_reason_set
                            == {"minimax_h3_dialogue_speaker_invalid"}
                            and not refined_contract_patch_preflight_reasons
                        ):
                            (
                                deterministic_patched_prompt,
                                deterministic_patch_report,
                                deterministic_patch_accepted,
                            ) = attempt_deterministic_ref2va_speaker_patch(
                                refined_h3_prompt,
                                refined_h3_reasons,
                            )
                            attempt_report[
                                "deterministic_speaker_patch"
                            ] = deterministic_patch_report
                            refinement_report[
                                "deterministic_speaker_patch"
                            ] = deterministic_patch_report
                            if deterministic_patch_accepted:
                                refined_h3_prompt = deterministic_patched_prompt
                                refined_packet["minimax_h3_prompt"] = (
                                    deterministic_patched_prompt
                                )
                                refined_h3_reasons = []
                                refined_reason_set = set()
                                refined_metadata = (
                                    refined_packet.get("metadata")
                                    if isinstance(
                                        refined_packet.get("metadata"), dict
                                    )
                                    else {}
                                )
                                refined_metadata.update(
                                    {
                                        "minimax_h3_ref2va_contract_patch_source": "deterministic_unique_subject_speaker_patch",
                                        "minimax_h3_ref2va_contract_patch_operations": deterministic_patch_report.get(
                                            "operations", []
                                        ),
                                        "minimax_h3_ref2va_contract_compiled_prompt_sha256": deterministic_patch_report.get(
                                            "compiled_prompt_sha256", ""
                                        ),
                                    }
                                )
                                refined_packet["metadata"] = refined_metadata
                        if (
                            target["minimax_h3_mode"] == "ref2va"
                            and refined_reason_set
                            and refined_reason_set.issubset(
                                _MINIMAX_H3_REF2VA_CONTRACT_LOCAL_REASONS
                            )
                            and "minimax_h3_ref_subject_definition_invalid"
                            in refined_reason_set
                            and remaining_refinement_attempts > 0
                            and not refined_contract_patch_preflight_reasons
                        ):
                            (
                                contract_patched_prompt,
                                contract_patch_report,
                                contract_patch_accepted,
                            ) = attempt_ref2va_contract_patch(
                                refined_h3_prompt,
                                refined_h3_reasons,
                                remaining_refinement_attempts,
                            )
                            attempt_report[
                                "ref2va_contract_patch"
                            ] = contract_patch_report
                            refinement_report[
                                "ref2va_contract_patch"
                            ] = contract_patch_report
                            if contract_patch_accepted:
                                refined_h3_prompt = contract_patched_prompt
                                refined_packet["minimax_h3_prompt"] = (
                                    contract_patched_prompt
                                )
                                refined_h3_reasons = []
                                refined_metadata = (
                                    refined_packet.get("metadata")
                                    if isinstance(
                                        refined_packet.get("metadata"), dict
                                    )
                                    else {}
                                )
                                refined_metadata.update(
                                    {
                                        "minimax_h3_ref2va_contract_patch_source": "verified_ledger_host_compiled",
                                        "minimax_h3_ref2va_contract_patch_operations": contract_patch_report.get(
                                            "operations", []
                                        ),
                                        "minimax_h3_ref2va_contract_patch_output_sha256": contract_patch_report.get(
                                            "output_sha256", ""
                                        ),
                                        "minimax_h3_ref2va_contract_compiled_prompt_sha256": contract_patch_report.get(
                                            "compiled_prompt_sha256", ""
                                        ),
                                    }
                                )
                                refined_metadata["used_grounding_fact_ids"] = list(
                                    dict.fromkeys(
                                        [
                                            *(
                                                refined_metadata.get(
                                                    "used_grounding_fact_ids", []
                                                )
                                                if isinstance(
                                                    refined_metadata.get(
                                                        "used_grounding_fact_ids"
                                                    ),
                                                    list,
                                                )
                                                else []
                                            ),
                                            *contract_patch_report.get(
                                                "grounding_fact_ids", []
                                            ),
                                        ]
                                    )
                                )
                                refined_packet["metadata"] = refined_metadata
                            else:
                                contract_patch_failed = True
                        attempt_report["parse_valid"] = bool(refined_parse_valid)
                        if refined_salvage_warning:
                            attempt_report["salvage_warning"] = refined_salvage_warning
                        attempt_report["candidate_reasons"] = refined_h3_reasons
                        candidate_rank = _minimax_h3_refinement_candidate_rank(
                            refined_h3_prompt,
                            refined_h3_reasons,
                            refined_parse_valid,
                            refined_salvage_warning,
                        )
                        attempt_report["candidate_rank"] = list(candidate_rank)
                        refinement_report["candidate_reasons"] = refined_h3_reasons
                        refinement_report["last_candidate_reasons"] = refined_h3_reasons
                        if refined_parse_valid and not refined_salvage_warning and not refined_h3_reasons:
                            initial_metadata = packet.get("metadata") if isinstance(packet.get("metadata"), dict) else {}
                            refined_metadata = refined_packet.get("metadata") if isinstance(refined_packet.get("metadata"), dict) else {}
                            for preserved_key in (
                                "observations",
                                "grounding_evidence_report_id",
                                "used_grounding_fact_ids",
                            ):
                                if preserved_key in initial_metadata and preserved_key not in refined_metadata:
                                    refined_metadata[preserved_key] = initial_metadata[preserved_key]
                            refined_packet["metadata"] = refined_metadata
                            packet = refined_packet
                            raw_output = refined_raw_output
                            reasoning_text = refined_reasoning
                            answer_text = refined_answer
                            parse_valid = refined_parse_valid
                            salvage_warning = refined_salvage_warning
                            attempt_report["accepted"] = True
                            refinement_report["accepted"] = True
                            refinement_report["accepted_attempt"] = refinement_index + 1
                            break

                        rejected_prompt = _sanitize_structured_prompt_text(refined_h3_prompt, "", 6000)
                        attempt_report["accepted"] = False
                        attempt_report["rejected_candidate_prompt"] = rejected_prompt
                        refinement_report["rejected_candidate_prompt"] = rejected_prompt
                        if candidate_rank < best_candidate_rank:
                            best_candidate_prompt = refined_h3_prompt
                            best_candidate_reasons = refined_h3_reasons
                            best_candidate_rank = candidate_rank
                            best_candidate_attempt = refinement_index + 1
                            attempt_report["selected_as_retry_base"] = True
                        if contract_patch_failed:
                            break
                        if refinement_index + 1 >= max_refinement_attempts:
                            break
                        refinement_candidate_prompt = best_candidate_prompt
                        refinement_candidate_reasons = best_candidate_reasons
                        refinement_base = "initial" if best_candidate_attempt == 0 else f"attempt_{best_candidate_attempt}"
                        refinement_report["retry_attempted"] = True
                        refinement_report["retry_base"] = refinement_base
                    except Exception as refinement_exc:
                        if (
                            backend_options is not None
                            and backend_options.fail_closed_telemetry
                            and "visual_grounding_telemetry_error"
                            in str(refinement_exc).lower()
                        ):
                            raise
                        attempt_report["error"] = str(refinement_exc)
                        refinement_report["error"] = str(refinement_exc)
                        _dg_log(
                            "MiniMax H3 refinement pass %s failed; keeping the first candidate. error=%s",
                            refinement_index + 1,
                            refinement_exc,
                        )
                        break
                if not refinement_report["accepted"]:
                    refinement_report["candidate_reasons"] = best_candidate_reasons
                    refinement_report["best_candidate_attempt"] = best_candidate_attempt
                    refinement_report["best_candidate_rank"] = list(best_candidate_rank)
                    if (
                        best_candidate_attempt > 0
                        and best_candidate_rank[0] == 0
                        and parse_valid
                        and not salvage_warning
                    ):
                        # Keep strict fail-closed readiness, but do not throw away a
                        # parse-valid refinement that is strictly closer to the
                        # target contract than the initial candidate.  This makes
                        # the next bounded retry start from actual progress instead
                        # of repeatedly presenting the stale original failures.
                        packet["minimax_h3_prompt"] = best_candidate_prompt
                        refinement_report["applied_best_rejected_candidate"] = True
                        refinement_report["applied_best_rejected_candidate_attempt"] = (
                            best_candidate_attempt
                        )
                        refinement_report["applied_best_rejected_candidate_reasons"] = list(
                            best_candidate_reasons
                        )
                refinement_report["backend_seconds"] = round(refinement_backend_seconds, 3)
        if refinement_report:
            if not isinstance(packet.get("metadata"), dict):
                packet["metadata"] = {}
            packet["metadata"]["minimax_h3_refinement"] = refinement_report
        postprocess_seconds = max(0.0, time.perf_counter() - postprocess_start - refinement_backend_seconds)
    except Exception as exc:
        fallback_reason = str(exc)
        if (
            manage_unload
            and model_config.backend == "transformers_inprocess"
            and model_config.unload_policy != "unload_after_run"
        ):
            forced_unload_start = time.perf_counter()
            try:
                _release_transformers_runtime()
            finally:
                _director_metric_add(
                    "model_unload_seconds",
                    time.perf_counter() - forced_unload_start,
                )
        if runtime_required:
            raise RuntimeError(f"DiffusionGemma runtime failed and runtime_required is true: {exc}") from exc
        postprocess_start = time.perf_counter()
        packet = _template_packet(
            working_context.user_prompt,
            target["target_profile"],
            media_metadata,
            _safe_float(media_metadata.get("duration_seconds"), 0.0),
            target["audio_mode"],
            target["audio_guidance"],
            target["target_duration_seconds"],
            target["ltx_style"],
            target["ideogram_aspect_ratio"],
            target["ideogram_render_style"],
            target["ideogram_exact_text"],
            target["ideogram_json_output"],
            creativity_mode,
            creative_strength,
            target["negative_prompt_mode"],
            target["negative_prompt_guidance"],
            target["ltx_generation_mode"],
            target["ltx_long_horizon_mode"],
            target["ltx_camera_capability"],
        )
        packet.setdefault("metadata", {})["fallback_reason"] = fallback_reason
        postprocess_seconds = time.perf_counter() - postprocess_start
    finally:
        if manage_unload:
            _maybe_unload(model_config)

    if target["target_profile"] == "ltx":
        current_ltx_prompt = _prompt_field_to_text(
            packet.get("ltx_prompt", ""),
            _clean_prompt_request_text(working_context.user_prompt),
            int(max_output_chars),
        )
        repaired_ltx_prompt, voiceover_boundary_report = _repair_ltx_exact_voiceover_boundary(
            current_ltx_prompt,
            target["audio_guidance"],
            int(max_output_chars),
            audio_mode=target["audio_mode"],
            request_text=working_context.user_prompt,
        )
        packet["ltx_prompt"] = repaired_ltx_prompt
        if not isinstance(packet.get("metadata"), dict):
            packet["metadata"] = {}
        packet["metadata"]["ltx_exact_voiceover_boundary"] = voiceover_boundary_report
        if voiceover_boundary_report.get("applied"):
            packet["scene_segments"] = []

    total_seconds = time.perf_counter() - total_start
    final_json_preview = _json_dumps(packet)
    metadata = packet.get("metadata") if isinstance(packet.get("metadata"), dict) else {}
    metadata.update(
        {
            "backend": model_config.backend,
            "runtime_ready": bool(model_config.status.get("ready")),
            "raw_output_available": bool(raw_output),
            "used_template_fallback": bool(fallback_reason),
            "json_parse_valid": bool(parse_valid),
            "salvage_warning": salvage_warning,
            "media_policy": model_config.status.get("media_policy", "unknown"),
            "media": media_metadata,
            "audio_mode": target["audio_mode"],
            "audio_guidance": target["audio_guidance"],
            "target_duration_seconds": (
                _minimax_h3_duration_seconds(media_metadata, target["target_duration_seconds"], working_context.user_prompt)
                if target["target_profile"] == "minimax_h3"
                else _effective_duration_seconds(media_metadata, target["target_duration_seconds"])
            ),
            "ltx_style": target["ltx_style"],
            "ltx_generation_mode_requested": target["ltx_generation_mode"],
            "ltx_long_horizon_mode_requested": target["ltx_long_horizon_mode"],
            "ltx_camera_capability": target["ltx_camera_capability"],
            "ideogram_aspect_ratio": target["ideogram_aspect_ratio"],
            "ideogram_render_style": target["ideogram_render_style"],
            "ideogram_exact_text": target["ideogram_exact_text"],
            "ideogram_json_output": bool(target["ideogram_json_output"]),
            "creativity_mode": creativity_mode,
            "creative_strength": creative_strength,
            "thinking_mode": thinking_mode,
            "reasoning_available": bool(reasoning_text),
            "negative_prompt_mode": target["negative_prompt_mode"],
            "negative_prompt_guidance": target["negative_prompt_guidance"],
            "timing": {
                "backend_generation_seconds": round(backend_seconds, 3),
                "postprocess_seconds": round(postprocess_seconds, 3),
                "total_node_seconds": round(total_seconds, 3),
            },
            "lengths": {
                "model_prompt_chars": len(model_prompt),
                "raw_output_chars": len(raw_output),
                "answer_text_chars": len(answer_text),
                "reasoning_text_chars": len(reasoning_text),
                "final_json_chars": len(final_json_preview),
                "max_new_tokens": int(max_new_tokens),
            },
        }
    )
    if target["target_profile"] == "minimax_h3":
        h3_duration = _minimax_h3_duration_seconds(
            media_metadata,
            target["target_duration_seconds"],
            working_context.user_prompt,
        )
        metadata["minimax_h3_planning_duration_seconds"] = round(h3_duration, 3)
        metadata["minimax_h3_duration_advisory"] = _minimax_h3_duration_advisory(h3_duration)
        metadata["minimax_h3_mode"] = target["minimax_h3_mode"]
        configured_shot_count = target["minimax_h3_shot_count"]
        effective_shot_count = _minimax_h3_effective_shot_count(
            working_context.user_prompt,
            configured_shot_count,
        )
        metadata["minimax_h3_shot_count"] = configured_shot_count
        metadata["minimax_h3_effective_shot_count"] = effective_shot_count
        metadata["minimax_h3_shot_count_source"] = (
            "target_profile"
            if configured_shot_count != "auto"
            else "user_prompt"
            if effective_shot_count
            else "model_auto"
        )
        dialogue_mode = target["minimax_h3_dialogue_mode"]
        dialogue_line_count = target["minimax_h3_dialogue_line_count"]
        actual_dialogue_line_count = _minimax_h3_dialogue_block_count(
            str(packet.get("minimax_h3_prompt", "") or "")
        )
        metadata["minimax_h3_dialogue_mode"] = dialogue_mode
        metadata["minimax_h3_dialogue_line_count"] = dialogue_line_count
        metadata["minimax_h3_dialogue_contract_scope"] = (
            "structural_presence_count_markup"
        )
        metadata["minimax_h3_dialogue_actual_line_count"] = actual_dialogue_line_count
        dialogue_validation_reasons = _minimax_h3_prompt_validation_reasons(
            str(packet.get("minimax_h3_prompt", "") or ""),
            h3_duration,
            working_context.user_prompt,
            target["audio_mode"],
            int(max_output_chars),
            target["minimax_h3_mode"],
            str(media_metadata.get("minimax_h3_reference_manifest", "")),
            target["minimax_h3_shot_count"],
            dialogue_mode,
            dialogue_line_count,
            target["minimax_h3_dialogue_guidance"],
            minimax_h3_expected_subject_count,
        )
        metadata["minimax_h3_dialogue_contract_satisfied"] = not any(
            reason in _MINIMAX_H3_DIALOGUE_LOCAL_REASONS
            or reason == "minimax_h3_dialogue_audio_mode_conflict"
            for reason in dialogue_validation_reasons
        )
        metadata["minimax_h3_dialogue_diagnostics"] = _minimax_h3_dialogue_diagnostics(
            str(packet.get("minimax_h3_prompt", "") or ""),
            target["minimax_h3_mode"],
        )
        if target["minimax_h3_dialogue_guidance"]:
            metadata["minimax_h3_dialogue_guidance"] = target["minimax_h3_dialogue_guidance"]
        if target["minimax_h3_mode"] == "ref2va":
            metadata["minimax_h3_reference_tags"] = _minimax_h3_reference_tags(
                str(media_metadata.get("minimax_h3_reference_manifest", ""))
            )
            actual_subject_count = _minimax_h3_ref2va_subject_count(
                str(packet.get("minimax_h3_prompt", "") or "")
            )
            metadata["minimax_h3_expected_subject_count"] = (
                minimax_h3_expected_subject_count
            )
            metadata["minimax_h3_actual_subject_count"] = actual_subject_count
            metadata["minimax_h3_subject_count_contract_scope"] = (
                "semantic_subject_definition_labels"
            )
            metadata["minimax_h3_subject_count_contract_satisfied"] = bool(
                minimax_h3_expected_subject_count <= 0
                or actual_subject_count == minimax_h3_expected_subject_count
            )
        requested_medium = _minimax_h3_requested_visual_medium(working_context.user_prompt)
        if requested_medium:
            metadata["minimax_h3_requested_visual_medium"] = requested_medium
        requested_style_locks = _minimax_h3_requested_style_locks(working_context.user_prompt)
        if requested_style_locks:
            metadata["minimax_h3_requested_style_locks"] = requested_style_locks
    _copy_synthesis_metadata(metadata, media_metadata)
    readiness_prompt = ""
    if target["target_profile"] == "ideogram4":
        readiness_prompt = _ideogram_prompt_value_to_text(
            packet.get("ideogram_prompt", ""),
            _clean_prompt_request_text(working_context.user_prompt),
            target["ideogram_aspect_ratio"],
            target["ideogram_render_style"],
            target["ideogram_exact_text"],
            target["ideogram_json_output"],
            int(max_output_chars),
            creativity_mode,
            creative_strength,
        )
    elif target["target_profile"] == "minimax_h3":
        readiness_prompt = _sanitize_structured_prompt_text(
            str(packet.get("minimax_h3_prompt", "") or ""),
            "",
            int(max_output_chars),
        )
    else:
        readiness_prompt = _prompt_field_to_text(
            packet.get("ltx_prompt", ""),
            _clean_prompt_request_text(working_context.user_prompt),
            int(max_output_chars),
        )
    ready_for_generation, blocked_reasons = _packet_generation_readiness(
        packet,
        working_context,
        target,
        readiness_prompt,
        media_metadata,
        int(max_output_chars),
    )
    metadata["ready_for_generation"] = bool(ready_for_generation)
    metadata["blocked_reasons"] = blocked_reasons
    guard_blocked, _guard_reasons, _guard_report = _strict_grounding_guard_block(
        metadata
    )
    metadata["claim_verification_passed"] = bool(
        not guard_blocked
        and not any(
            reason.startswith("metadata_only_")
            or reason.startswith("visual_description_contradiction")
            or reason.startswith("minimax_h3_")
            or reason.startswith("ltx_")
            for reason in blocked_reasons
        )
    )
    packet["metadata"] = metadata
    return packet, raw_output, reasoning_text, fallback_reason, parse_valid, salvage_warning, metadata


def _media_image_count(images: Any) -> int:
    if images is None:
        return 0
    shape = getattr(images, "shape", None)
    if shape is not None:
        try:
            return int(shape[0]) if len(shape) >= 4 else 1
        except Exception:
            pass
    try:
        return len(images)
    except Exception:
        return 1


def _minimax_h3_reference_inventory(context: GemmaContext) -> tuple[int, int]:
    """Count Ref2VA Pictures separately from sampled Video frames."""

    metadata = context.media_metadata if isinstance(context.media_metadata, dict) else {}
    source = str(metadata.get("source", context.source or "none") or "none").strip().casefold()
    actual_batch_count = _media_image_count(context.images)
    explicit_picture_count = metadata.get("minimax_h3_reference_image_batch_count")
    if explicit_picture_count is not None:
        picture_count = int(max(0.0, _safe_float(explicit_picture_count, 0.0)))
    elif source == "image":
        picture_count = actual_batch_count
    elif source == "image+video":
        picture_count = int(
            max(0.0, _safe_float(metadata.get("reference_image_count"), 0.0))
        )
    else:
        picture_count = 0
    video_count = 1 if (
        source in {"video", "image+video"}
        or bool(metadata.get("minimax_h3_reference_video_attached_for_analysis"))
    ) else 0
    return picture_count, video_count


def _context_with_minimax_h3_reference_policy(
    context: GemmaContext,
    target_config: TargetProfileConfig | dict[str, Any] | None,
    reference_policy_config: H3ReferencePolicyConfig | dict[str, Any] | str | None = None,
) -> GemmaContext:
    """Resolve Ref2VA roles before cache lookup, grounding, and backend dispatch.

    A connected typed policy intentionally overrides stale inline Context-Hub
    presets. Without one, an existing nonempty manifest remains authoritative;
    only a genuinely missing manifest is synthesized from attached visual media.
    """

    if not isinstance(context, GemmaContext):
        raise TypeError("gemma_context must come from DiffusionGemma Context Hub.")
    target = _target_profile_config_to_dict(target_config)
    if not (
        target["target_profile"] == "minimax_h3"
        and target["minimax_h3_mode"] == "ref2va"
    ):
        return context

    metadata = copy.deepcopy(context.media_metadata)
    picture_count, video_count = _minimax_h3_reference_inventory(context)
    existing_manifest = _normalize_minimax_h3_reference_manifest(
        str(metadata.get("minimax_h3_reference_manifest", ""))
    )
    connected_policy = reference_policy_config is not None
    if connected_policy:
        policy = _h3_reference_policy_config_to_dict(reference_policy_config)
        manifest = _minimax_h3_reference_policy_manifest(
            policy,
            picture_count,
            video_count,
        )
        manifest_source = "policy_node"
        manifest_preset = "policy_node"
        expected_subject_count = policy["expected_subject_count"]
        resolved_layout = policy["layout"]
    elif existing_manifest:
        manifest = existing_manifest
        manifest_source = str(
            metadata.get("minimax_h3_reference_manifest_source", "context_manifest")
        )
        manifest_preset = str(
            metadata.get("minimax_h3_reference_manifest_preset", "custom")
        )
        expected_subject_count = _normalize_minimax_h3_expected_subject_count(
            metadata.get("minimax_h3_expected_subject_count", 0)
        )
        resolved_layout = str(
            metadata.get("minimax_h3_reference_policy_layout", "explicit_context_manifest")
        )
        policy = {
            "schema": _MINIMAX_H3_REFERENCE_POLICY_SCHEMA,
            "layout": resolved_layout,
            "custom_manifest": "",
            "expected_subject_count": expected_subject_count,
        }
    else:
        policy = _h3_reference_policy_config_to_dict(None)
        manifest = _minimax_h3_reference_policy_manifest(
            policy,
            picture_count,
            video_count,
        )
        manifest_source = "director_auto"
        manifest_preset = "director_auto"
        expected_subject_count = policy["expected_subject_count"]
        resolved_layout = policy["layout"]

    definitions = _minimax_h3_reference_definitions(manifest)
    manifest_reasons = _minimax_h3_reference_manifest_validation_reasons(manifest)
    policy_metadata = {
        "schema": _MINIMAX_H3_REFERENCE_POLICY_SCHEMA,
        "layout": resolved_layout,
        "expected_subject_count": expected_subject_count,
    }
    warnings = list(context.warnings)
    if connected_policy:
        stale_prefixes = (
            "H3 reference manifest needs correction:",
            "DiffusionGemma received ",
        )
        warnings = [
            warning
            for warning in warnings
            if not str(warning).startswith(stale_prefixes)
        ]
    metadata.update(
        {
            "minimax_h3_mode": "ref2va",
            "minimax_h3_reference_manifest": manifest,
            "minimax_h3_reference_manifest_source": manifest_source,
            "minimax_h3_reference_manifest_preset": manifest_preset,
            "minimax_h3_reference_policy_schema": _MINIMAX_H3_REFERENCE_POLICY_SCHEMA,
            "minimax_h3_reference_policy_layout": resolved_layout,
            "minimax_h3_reference_policy": policy_metadata,
            "minimax_h3_reference_tags": [tag for tag, _description in definitions],
            "minimax_h3_reference_manifest_reasons": manifest_reasons,
            "minimax_h3_reference_image_batch_count": picture_count,
            "minimax_h3_reference_video_attached_for_analysis": bool(video_count),
            "minimax_h3_expected_subject_count": expected_subject_count,
            "reference_image_count": picture_count,
            "reference_image_backend_attached": bool(picture_count and context.images is not None),
            "warnings": warnings,
        }
    )
    return GemmaContext(
        user_prompt=context.user_prompt,
        images=context.images,
        source=context.source,
        media_metadata=metadata,
        visual_description=context.visual_description,
        warnings=warnings,
    )


def _grounding_asset_registry(context: GemmaContext) -> dict[str, Any]:
    return build_asset_registry(context.media_metadata, _media_image_count(context.images))


def _audit_grounding_master_prompt(asset_registry: dict[str, Any]) -> str:
    fact_limit = _strict_evidence_fact_limit(asset_registry)
    return (
        f"{DEFAULT_MASTER_PROMPT}\n\n"
        f"GROUNDING GUARD AUDIT CONTRACT ({GROUNDING_LEDGER_SCHEMA_ID}): In the same final JSON object, add a "
        "top-level grounding_ledger. Before writing any generator prompt, inspect every host-listed asset and separate "
        "direct observations from inference and creative additions. The ledger must contain exactly schema, "
        "analysis_status, observed_facts, inferred_facts, creative_additions, uncertainties, and "
        "grounding_failure_reasons. Each observed fact contains exactly fact_id, claim, confidence, categories, "
        "evidence, and optional typed_claims. Cite only the host asset/sample/frame/time values below. If visual "
        "inspection fails or a refusal occurs, say so with analysis_status refused/transport_error/uncertain and do "
        f"not fabricate observations. Limit observed facts to the {fact_limit} most useful facts. The host, not you, determines "
        "the final status and overwrites supplied asset/status metadata. For every H3 Picture/Video fact, include at "
        "least one declared category from that asset's host grounding_role_categories; never copy [dg:...] annotations "
        "into a generator prompt. Typed-claim numeric values must be signed 64-bit integers; encode exact decimal "
        "measurements as strings.\n"
        f"Host asset registry: {_json_dumps(asset_registry)}"
    )


def _audit_grounding_retry_master_prompt(asset_registry: dict[str, Any]) -> str:
    return (
        f"{_audit_grounding_master_prompt(asset_registry)}\n\n"
        "AUDIT RETRY: The previous response did not close a strict final JSON object or its grounding ledger was malformed. "
        "Return the requested object directly. Do not emit reasoning, planning notes, a checklist, self-correction, Markdown, or alternate drafts."
    )


def _validate_audit_grounding_response(
    result: BackendRunResult | None,
    fallback_raw: str,
    registry: dict[str, Any],
    external_evidence_json: str,
    final_text: str,
) -> tuple[Any, str]:
    """Extract and validate the ledger selected for an audit attempt."""

    source_raw = str(
        getattr(result, "decoded_text", "")
        if result is not None
        else fallback_raw
    )
    warning = ""
    try:
        _audit_reasoning, answer = _split_reasoning_and_answer(source_raw)
        ledger = extract_grounding_ledger(answer or source_raw)
    except GroundingLedgerError as exc:
        ledger = {}
        warning = str(exc)
    validation = validate_grounding_ledger(
        ledger,
        registry,
        external_evidence_json,
        final_text=final_text or source_raw,
    )
    return validation, warning


def _strict_evidence_fact_limit(asset_registry: dict[str, Any]) -> int:
    registry_assets = asset_registry.get("assets", [])
    visual_assets = [
        asset
        for asset in registry_assets
        if isinstance(asset, dict) and asset.get("kind") in {"image", "video"}
    ] if isinstance(registry_assets, list) else []
    if len(visual_assets) == 1:
        only_asset = visual_assets[0]
        samples = only_asset.get("samples", [])
        if only_asset.get("kind") == "image" and isinstance(samples, list) and len(samples) <= 1:
            return 4
    video_count = sum(asset.get("kind") == "video" for asset in visual_assets)
    return max(4, min(8, len(visual_assets) * 2 + video_count * 2))


def _strict_evidence_max_new_tokens(
    max_new_tokens: int,
    retry: bool = False,
    evidence_token_budget: str = "auto",
) -> int:
    budget = str(evidence_token_budget or "auto").strip().lower()
    if budget in EVIDENCE_TOKEN_BUDGETS and budget != "auto":
        return int(budget)
    return max(384, min(int(max_new_tokens), 1024 if retry else 768))


def _strict_evidence_prompt(asset_registry: dict[str, Any], retry: bool = False) -> str:
    fact_limit = _strict_evidence_fact_limit(asset_registry)
    retry_instruction = (
        "This is the one permitted focused retry. The previous ledger did not pass validation. Ignore all creative "
        "intent and return the smallest sufficient asset-by-asset factual checklist. Keep every JSON object and array "
        f"closed, and use no more than {fact_limit} observed facts. State only what is directly visible in the reduced "
        "opening/middle/closing frame evidence."
        if retry
        else "Perform a factual evidence pass before any creative compilation."
    )
    registry_assets = asset_registry.get("assets", [])
    first_asset = registry_assets[0] if isinstance(registry_assets, list) and registry_assets else {}
    example_reference: dict[str, Any] = {
        "asset_id": str(first_asset.get("asset_id", "image:1"))
    }
    first_samples = first_asset.get("samples", []) if isinstance(first_asset, dict) else []
    if isinstance(first_samples, list) and first_samples:
        first_sample = first_samples[0]
        if isinstance(first_sample, dict):
            for key in ("sample_ordinal", "source_frame_index", "timecode_seconds"):
                if key in first_sample:
                    example_reference[key] = first_sample[key]
    role_categories = (
        first_asset.get("grounding_role_categories", [])
        if isinstance(first_asset, dict)
        else []
    )
    example_category = (
        str(role_categories[0])
        if isinstance(role_categories, list) and role_categories
        else "object"
    )
    example_ledger = {
        "schema": GROUNDING_LEDGER_SCHEMA_ID,
        "analysis_status": "grounded",
        "observed_facts": [
            {
                "fact_id": "fact-1",
                "claim": "directly visible fact",
                "confidence": "high",
                "categories": [example_category],
                "evidence": [example_reference],
                "typed_claims": [],
            }
        ],
        "inferred_facts": [],
        "creative_additions": [],
        "uncertainties": [],
        "grounding_failure_reasons": [],
    }
    return (
        f"{retry_instruction}\n"
        "Return one strict JSON object only: no Markdown fence, final-answer marker, prose prefix, reasoning, or trailing "
        "text. Use exactly this root shape:\n"
        f"{_json_dumps(example_ledger)}\n"
        "Allowed confidence is high, medium, or low. Allowed categories are identity, appearance, object, count, "
        "color, text, spatial, environment, lighting, action, motion, camera, composition, temporal, or other. Use "
        "at least one high/medium observation for every required still. For a multi-frame video, cite distinct temporal "
        "regions. For identity-plus-control, identity/appearance comes only from the still; action/motion or "
        "camera/composition evidence comes only from the video. For every H3 Picture/Video fact, include at least one "
        "declared category from that asset's authoritative grounding_role_categories. The [dg:...] role annotations are "
        "host-only metadata and must never appear in observations or generator prompts. Never infer unseen content. If you cannot inspect an asset, "
        "return uncertain or refused with empty/factually limited observations and explain why; this is safer than "
        "guessing. Typed-claim numeric values must be signed 64-bit integers; encode exact decimal measurements as "
        f"strings. Use no more than {fact_limit} observed facts total. Combine related visible attributes into one "
        "claim with multiple categories, keep each claim at 24 words or fewer, cite one sufficient evidence reference "
        "per still-image fact, and leave typed_claims empty unless a precise visible count or exact text requires one. "
        "Completing and closing the strict JSON object takes priority over adding another fact.\n"
        f"HOST ASSET REGISTRY (authoritative): {_json_dumps(asset_registry)}"
    )


def _strict_evidence_json_payload(
    raw_output: str,
    *,
    repair_log: list[str] | None = None,
) -> str:
    """Accept strict evidence JSON with one bounded transport repair.

    Native thinking wrappers are transport metadata. A duplicated root opening
    brace is also safe to remove when the resulting value independently passes
    the strict duplicate-key ledger parser. No ledger field, claim, citation,
    category, status, or closing structure is synthesized here.
    """

    text = str(raw_output or "").strip()
    if not text:
        raise GroundingLedgerError("grounding evidence output is empty")

    thinking_patterns = (
        re.compile(r"\A<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL),
        re.compile(
            r"\A<\|thought\|>.*?<\|end_thought\|>\s*",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    )
    consumed = True
    while consumed:
        consumed = False
        for pattern in thinking_patterns:
            match = pattern.match(text)
            if match:
                text = text[match.end() :].lstrip()
                consumed = True
                break

    if not text.startswith("{"):
        raise GroundingLedgerError(
            "grounding evidence must be exact JSON without a conversational prefix"
        )
    try:
        extract_grounding_ledger(text, allow_combined=False)
        return text
    except GroundingLedgerError as exact_error:
        if not text.startswith("{{"):
            raise
        candidate = text[1:].lstrip()
        try:
            extract_grounding_ledger(candidate, allow_combined=False)
        except GroundingLedgerError:
            raise exact_error
        if repair_log is not None:
            repair_log.append("removed_extra_outer_open_brace")
        return candidate


def _slice_media_images(images: Any, positions: list[int]) -> Any:
    if images is None:
        return None
    try:
        import torch

        if isinstance(images, torch.Tensor):
            index = torch.tensor(positions, dtype=torch.long, device=images.device)
            return images.index_select(0, index)
    except Exception:
        pass
    try:
        return images[positions]
    except Exception:
        return [images[index] for index in positions]


def _focused_retry_context(context: GemmaContext) -> GemmaContext:
    count = _media_image_count(context.images)
    metadata = context.media_metadata.copy()
    reference_count = _attached_reference_image_count(metadata)
    reference_count = min(reference_count, count)
    video_count = max(0, count - reference_count)
    if video_count <= 3:
        return GemmaContext(
            user_prompt=context.user_prompt,
            images=context.images,
            source=context.source,
            media_metadata=metadata,
            visual_description=context.visual_description,
            warnings=list(context.warnings),
        )
    relative_video_positions = list(dict.fromkeys([0, video_count // 2, video_count - 1]))
    absolute_positions = list(range(reference_count)) + [
        reference_count + position for position in relative_video_positions
    ]
    sampled_indices = metadata.get("sampled_indices")
    if isinstance(sampled_indices, list):
        metadata["sampled_indices"] = [
            sampled_indices[position]
            for position in relative_video_positions
            if 0 <= position < len(sampled_indices)
        ]
    timecodes = metadata.get("sampled_timecodes_seconds")
    if isinstance(timecodes, list):
        metadata["sampled_timecodes_seconds"] = [
            timecodes[position]
            for position in relative_video_positions
            if 0 <= position < len(timecodes)
        ]
    metadata["video_sampled_frame_count"] = len(relative_video_positions)
    metadata["sampled_frame_count"] = len(absolute_positions)
    metadata["grounding_retry_frame_reduction"] = {
        "strategy": "opening_middle_closing",
        "original_video_sample_count": video_count,
        "selected_video_sample_ordinals": [position + 1 for position in relative_video_positions],
    }
    return GemmaContext(
        user_prompt=context.user_prompt,
        images=_slice_media_images(context.images, absolute_positions),
        source=context.source,
        media_metadata=metadata,
        visual_description=context.visual_description,
        warnings=list(context.warnings),
    )


def _telemetry_error_text(result: BackendRunResult | None) -> str:
    if result is None or not isinstance(result.telemetry, dict):
        return ""
    errors = result.telemetry.get("errors")
    errors = list(errors) if isinstance(errors, list) else []
    existing_integrity_messages = {
        str(item.get("message", ""))
        for item in errors
        if isinstance(item, dict)
    }
    for message in _telemetry_integrity_errors(result.telemetry):
        if message not in existing_integrity_messages:
            errors.append({"stage": "integrity", "message": message})
    if not errors:
        return ""
    rendered: list[str] = []
    for error in errors[:8]:
        if isinstance(error, dict):
            rendered.append(f"{error.get('stage', 'telemetry')}:{error.get('message', '')}")
        else:
            rendered.append(str(error))
    return "; ".join(rendered)


def _telemetry_refusal(result: BackendRunResult | None) -> bool:
    if result is None or not isinstance(result.telemetry, dict):
        return False
    refusal = result.telemetry.get("refusal")
    return bool(isinstance(refusal, dict) and refusal.get("detected"))


def _trajectory_summary(results: list[BackendRunResult]) -> dict[str, Any]:
    snapshots: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for call_index, result in enumerate(results, start=1):
        telemetry = result.telemetry if isinstance(result.telemetry, dict) else {}
        call_snapshots = telemetry.get("snapshots") if isinstance(telemetry.get("snapshots"), list) else []
        selected = call_snapshots if len(call_snapshots) <= 2 else [call_snapshots[0], call_snapshots[-1]]
        for snapshot in selected:
            if isinstance(snapshot, dict):
                snapshots.append({"call": call_index, **copy.deepcopy(snapshot)})
        calls.append(
            {
                "call": call_index,
                "stage": result.stage,
                "forward_count": int(telemetry.get("forward_count", 0) or 0),
                "refusal": copy.deepcopy(telemetry.get("refusal", {})),
                "errors": copy.deepcopy(telemetry.get("errors", [])),
                "truncated": bool(telemetry.get("truncated")),
            }
        )
    return {"calls": calls, "snapshots": snapshots, "truncated_to_first_final_per_call": True}


def _grounding_report_id(ledger: dict[str, Any] | None, registry: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"ledger": ledger, "asset_registry": registry},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "dggr-" + hashlib.sha256(canonical).hexdigest()[:20]


_STRICT_COMPILER_STRUCTURAL_METADATA_KEYS = (
    "source",
    "duration_seconds",
    "source_fps",
    "sample_fps",
    "frame_count",
    "sampled_frame_count",
    "video_sampled_frame_count",
    "sampled_indices",
    "sampled_timecodes_seconds",
    "width",
    "height",
    "media_synthesis_mode",
    "image_role",
    "video_role",
    "reference_image_count",
    "reference_image_backend_attached",
    "transformers_video_transport",
    "minimax_h3_mode",
    "minimax_h3_reference_manifest",
    "minimax_h3_reference_tags",
    "minimax_h3_expected_subject_count",
    "minimax_h3_reference_image_batch_count",
    "minimax_h3_reference_video_attached_for_analysis",
    "minimax_h3_reference_manifest_source",
    "minimax_h3_reference_policy_schema",
    "minimax_h3_reference_policy_layout",
    "minimax_h3_reference_policy",
)


def _strict_compiler_metadata(
    context: GemmaContext,
    verified_ledger: dict[str, Any],
    evidence_report_id: str,
    transport: dict[str, Any],
    asset_registry: dict[str, Any],
) -> dict[str, Any]:
    """Build a ledger-only compiler context from host-owned structural fields."""

    original = context.media_metadata if isinstance(context.media_metadata, dict) else {}
    metadata = {
        key: copy.deepcopy(original[key])
        for key in _STRICT_COMPILER_STRUCTURAL_METADATA_KEYS
        if key in original
    }
    metadata["source"] = str(original.get("source", context.source or "none"))
    if "minimax_h3_reference_manifest" in metadata:
        metadata["minimax_h3_reference_manifest"] = _strip_grounding_role_annotations(
            str(metadata["minimax_h3_reference_manifest"])
        )
    if "media_synthesis_mode" in metadata:
        metadata["media_synthesis_mode"] = _normalize_media_synthesis_mode(
            metadata["media_synthesis_mode"]
        )
        metadata["synthesis_contract"] = _synthesis_contract_text(metadata)
    metadata.update(
        {
            "verified_grounding_ledger": copy.deepcopy(verified_ledger),
            "grounding_evidence_report_id": str(evidence_report_id),
            "pixel_tensor_present": False,
            "pixels_sent_to_backend": False,
            "pixel_transport_confirmed": bool(transport.get("pixel_transport_confirmed")),
            "visual_grounding_mode": "verified_ledger",
            "grounding_required_asset_ids": [
                str(asset.get("asset_id"))
                for asset in asset_registry.get("assets", [])
                if isinstance(asset, dict)
                and asset.get("asset_id")
                and asset.get("required_for_grounding", True)
            ],
        }
    )
    return metadata


def _eligible_grounding_fact_assets(
    ledger: dict[str, Any],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for fact in ledger.get("observed_facts", []):
        if not isinstance(fact, dict) or fact.get("confidence") not in {"high", "medium"}:
            continue
        fact_id = fact.get("fact_id")
        evidence = fact.get("evidence")
        if not isinstance(fact_id, str) or not isinstance(evidence, list):
            continue
        asset_ids = {
            str(reference.get("asset_id"))
            for reference in evidence
            if isinstance(reference, dict) and reference.get("asset_id")
        }
        if asset_ids:
            result[fact_id] = asset_ids
    return result


def _validated_compiler_provenance(
    packet_metadata: Any,
    evidence_report_id: str,
    available_fact_ids: list[str],
    fact_assets_by_id: dict[str, set[str]] | None = None,
    required_asset_ids: list[str] | None = None,
) -> list[str]:
    if not isinstance(packet_metadata, dict):
        raise GroundingLedgerError(
            "visual_grounding_schema_invalid:compiler_metadata_missing"
        )
    if packet_metadata.get("grounding_evidence_report_id") != evidence_report_id:
        raise GroundingLedgerError(
            "visual_grounding_schema_invalid:compiler_evidence_report_id_missing_or_mismatch"
        )
    declared = packet_metadata.get("used_grounding_fact_ids")
    if (
        not isinstance(declared, list)
        or not declared
        or any(not isinstance(item, str) or not item for item in declared)
        or len(set(declared)) != len(declared)
    ):
        raise GroundingLedgerError(
            "visual_grounding_schema_invalid:compiler_used_fact_ids_invalid"
        )
    unknown = sorted(set(declared) - set(available_fact_ids))
    if unknown:
        raise GroundingLedgerError(
            "visual_grounding_schema_invalid:compiler_used_fact_ids_unknown:"
            + ",".join(unknown)
        )
    if fact_assets_by_id is not None and required_asset_ids is not None:
        used_assets = {
            asset_id
            for fact_id in declared
            for asset_id in fact_assets_by_id.get(fact_id, set())
        }
        missing_assets = sorted(set(required_asset_ids) - used_assets)
        if missing_assets:
            raise GroundingLedgerError(
                "visual_grounding_schema_invalid:compiler_fact_coverage_missing_assets:"
                + ",".join(missing_assets)
            )
    return list(declared)


def _grounding_forward_count(results: list[BackendRunResult]) -> int:
    return sum(
        int(result.telemetry.get("forward_count", 0) or 0)
        for result in results
        if isinstance(result.telemetry, dict)
    )


def _grounding_timings(results: list[BackendRunResult]) -> dict[str, Any]:
    return {
        "calls": [
            {
                "stage": result.stage,
                "input_token_count": int(result.input_tokens),
                "generated_token_count": int(result.generated_tokens),
                "generated_character_count": int(result.generated_characters),
                **copy.deepcopy(result.timing),
            }
            for result in results
        ],
        "total_backend_seconds": round(
            sum(_safe_float(result.timing.get("total_seconds"), 0.0) for result in results),
            6,
        ),
        "total_generated_token_count": sum(int(result.generated_tokens) for result in results),
        "total_generated_character_count": sum(int(result.generated_characters) for result in results),
    }


def _grounding_effective_sampling(
    results: list[BackendRunResult],
    guard: GroundingGuardConfig,
    model_config: RuntimeConfig,
    *,
    guarded_attempted: bool,
) -> dict[str, Any]:
    calls = [
        {"stage": result.stage, **copy.deepcopy(result.effective_sampling)}
        for result in results
        if isinstance(result.effective_sampling, dict) and result.effective_sampling
    ]
    if calls:
        effective = copy.deepcopy(calls[0])
        effective["sampling_started"] = True
        effective["call_count"] = len(calls)
        effective["calls"] = calls
        return effective
    if guarded_attempted:
        effective = guard.sampling_settings
        effective["profile"] = guard.sampling_profile
        effective["sampling_started"] = False
        return effective
    denoising_steps = os.environ.get("DG_MAX_DENOISING_STEPS", "").strip()
    return {
        "profile": "legacy_scalar_temperature",
        "backend": model_config.backend,
        "temperature": model_config.temperature,
        "max_denoising_steps": int(denoising_steps) if denoising_steps.isdigit() else None,
        "sampling_started": True,
    }


def _blocked_grounding_packet(
    target_config: TargetProfileConfig | dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    target = _target_profile_config_to_dict(target_config)
    return {
        "ltx_prompt": "",
        "ideogram_prompt": "",
        "minimax_h3_prompt": "",
        "negative_prompt": "",
        "scene_segments": [],
        "metadata": {
            "target_profile": target["target_profile"],
            "ready_for_generation": False,
            "blocked_reasons": list(report.get("blocked_reasons", ["visual_grounding_unverified"])),
            "grounding_guard": report,
        },
    }


def _attach_grounding_report(
    packet: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    metadata = packet.get("metadata") if isinstance(packet.get("metadata"), dict) else {}
    metadata["grounding_guard"] = report
    metadata["grounding_status"] = str(report.get("analysis_status", "uncertain"))
    media = metadata.get("media") if isinstance(metadata.get("media"), dict) else None
    transport = report.get("transport") if isinstance(report.get("transport"), dict) else {}
    if (
        media is not None
        and not transport.get("not_applicable")
        and "pixel_transport_confirmed" in transport
    ):
        confirmed = bool(transport.get("pixel_transport_confirmed"))
        media["pixel_transport_confirmed"] = confirmed
        media["pixels_sent_to_backend"] = confirmed
    packet["metadata"] = metadata
    return metadata


def _trace_output_root() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory())
    except Exception:
        return Path(__file__).resolve().parents[2] / "output"


def _trace_asset_hashes(context: GemmaContext, registry: dict[str, Any]) -> dict[str, str]:
    if context.images is None:
        return {}
    hashes: dict[str, str] = {}
    for asset in registry.get("assets", []):
        if not isinstance(asset, dict):
            continue
        digest = hashlib.sha256()
        for position in asset.get("tensor_batch_positions", []):
            try:
                sample = context.images[int(position)]
                if hasattr(sample, "detach"):
                    sample = sample.detach().cpu().contiguous()
                if hasattr(sample, "numpy"):
                    digest.update(sample.numpy().tobytes())
                else:
                    digest.update(repr(sample).encode("utf-8"))
            except Exception as exc:
                digest.update(f"hash-error:{exc}".encode("utf-8"))
        hashes[str(asset.get("asset_id", "unknown"))] = digest.hexdigest()
    return hashes


def _maybe_save_grounding_trace(
    config: GroundingGuardConfig,
    report: dict[str, Any],
    context: GemmaContext,
    registry: dict[str, Any],
    evidence_responses: list[str],
    compiler_response: str,
    backend_results: list[BackendRunResult],
    model_config: RuntimeConfig,
) -> tuple[dict[str, Any], str]:
    if not config.save_detailed_trace:
        return report, ""
    try:
        trace = {
            "schema": "dg-grounding-trace/1",
            "report": report,
            "asset_hashes_sha256": _trace_asset_hashes(context, registry),
            "raw_probe_responses": list(evidence_responses),
            "raw_compiler_response": compiler_response,
            "raw_stage_responses": [
                {
                    "stage": item.stage,
                    "decoded_text": item.decoded_text,
                }
                for item in backend_results
            ],
            "detailed_trajectory": [copy.deepcopy(item.telemetry) for item in backend_results],
            "transport": [copy.deepcopy(item.transport) for item in backend_results],
            "environment": {
                "backend": model_config.backend,
                "model_path": model_config.model_path,
                "model_revision": model_config.status.get("_commit_hash", ""),
                "transformers_version": model_config.status.get("transformers_version", ""),
                "quantization": model_config.quantization,
            },
            "timings": _grounding_timings(backend_results),
        }
        path = persist_grounding_trace(
            trace,
            _trace_output_root(),
            config.trace_subfolder,
        )
        updated = copy.deepcopy(report)
        updated["saved_trace_path"] = str(path)
        return compact_grounding_report(updated), ""
    except Exception as exc:
        updated = copy.deepcopy(report)
        warnings = updated.get("warnings") if isinstance(updated.get("warnings"), list) else []
        warnings.append(f"Grounding trace was not saved: {exc}")
        updated["warnings"] = list(dict.fromkeys(warnings))
        return compact_grounding_report(updated), str(exc)


def _run_generation_packet(
    model_config: RuntimeConfig,
    gemma_context: GemmaContext,
    target_config: TargetProfileConfig | dict[str, Any],
    master_prompt: str = DEFAULT_MASTER_PROMPT,
    runtime_required: bool = False,
    max_new_tokens: int = 768,
    max_output_chars: int = 8000,
    creativity_mode: str = "editorial",
    creative_strength: float = 0.6,
    thinking_mode: str = "off",
    node_id: str | None = None,
    grounding_guard_config: GroundingGuardConfig | dict[str, Any] | str | None = None,
) -> tuple[dict[str, Any], str, str, str, bool, str, dict[str, Any]]:
    """Grounding-aware wrapper around the frozen legacy packet generator."""

    guard = normalize_grounding_guard_config(grounding_guard_config)
    context = gemma_context if isinstance(gemma_context, GemmaContext) else _gemma_context_from_media("")
    target_summary = _target_profile_config_to_dict(target_config)
    context = _context_with_minimax_h3_reference_policy(context, target_summary)
    thinking_mode = _effective_target_thinking_mode(
        thinking_mode,
        target_summary["target_profile"],
    )
    registry = _grounding_asset_registry(context)
    media_present = bool(registry.get("expected_image_count"))

    if guard.mode == "off":
        result = _run_generation_packet_legacy(
            model_config,
            context,
            target_config,
            master_prompt,
            runtime_required,
            max_new_tokens,
            max_output_chars,
            creativity_mode,
            creative_strength,
            thinking_mode,
            node_id=node_id,
        )
        packet, raw_output, reasoning_text, fallback_reason, parse_valid, salvage_warning, _metadata = result
        decision = decide_grounding_guard(
            guard,
            None,
            media_present=media_present,
            transport_confirmed=False,
        )
        report = build_grounding_report(
            guard,
            registry,
            decision,
            effective_sampling=_grounding_effective_sampling(
                [], guard, model_config, guarded_attempted=False
            ),
        )
        metadata = _attach_grounding_report(packet, report)
        return packet, raw_output, reasoning_text, fallback_reason, parse_valid, salvage_warning, metadata

    backend_results: list[BackendRunResult] = []
    call_budget = BackendCallBudget(max_calls=4)
    evidence_responses: list[str] = []
    compiler_response = ""
    declared_visual_asset_ids = [
        str(asset_id)
        for asset_id in registry.get("declared_visual_asset_ids", [])
        if isinstance(asset_id, str) and asset_id
    ]
    unattached_declared_asset_ids = [
        str(asset_id)
        for asset_id in registry.get("unattached_declared_visual_asset_ids", [])
        if isinstance(asset_id, str) and asset_id
    ]
    unexpected_attached_asset_ids = [
        str(asset_id)
        for asset_id in registry.get("unexpected_attached_visual_asset_ids", [])
        if isinstance(asset_id, str) and asset_id
    ]
    declared_attachment_mismatch = (
        str(registry.get("minimax_h3_mode", "")) == "ref2va"
        and bool(unattached_declared_asset_ids or unexpected_attached_asset_ids)
    )
    declared_transport_failure = {
        "not_applicable": False,
        "pixel_transport_confirmed": False,
        "expected_asset_count": len(declared_visual_asset_ids),
        "processor_observed_asset_count": len(registry.get("assets", [])),
        "mismatch_reasons": [
            *(
                [
                    "declared_visual_assets_not_attached:"
                    + ",".join(unattached_declared_asset_ids)
                ]
                if unattached_declared_asset_ids
                else []
            ),
            *(
                [
                    "attached_visual_assets_not_declared:"
                    + ",".join(unexpected_attached_asset_ids)
                ]
                if unexpected_attached_asset_ids
                else []
            ),
        ],
    }
    try:
        if guard.mode == "strict" and declared_attachment_mismatch:
            decision = decide_grounding_guard(
                guard,
                None,
                media_present=True,
                transport_confirmed=False,
            )
            report = build_grounding_report(
                guard,
                registry,
                decision,
                transport=declared_transport_failure,
                attempt_count=0,
                retry_reasons=("declared_visual_asset_attachment_mismatch",),
                warnings=registry.get("warnings", []),
                effective_sampling=_grounding_effective_sampling(
                    [], guard, model_config, guarded_attempted=True
                ),
            )
            report["model_call_count"] = 0
            report = compact_grounding_report(report)
            packet = _blocked_grounding_packet(target_config, report)
            return (
                packet,
                "",
                "",
                "visual_transport_error:declared_visual_asset_attachment_mismatch",
                False,
                "",
                packet["metadata"],
            )

        if not media_present:
            packet, raw_output, reasoning_text, fallback_reason, parse_valid, salvage_warning, _metadata = (
                _run_generation_packet_legacy(
                    model_config,
                    context,
                    target_config,
                    master_prompt,
                    runtime_required,
                    max_new_tokens,
                    max_output_chars,
                    creativity_mode,
                    creative_strength,
                    thinking_mode,
                    node_id=node_id,
                    manage_unload=False,
                )
            )
            decision = decide_grounding_guard(
                guard,
                None,
                media_present=bool(declared_visual_asset_ids),
                transport_confirmed=False,
            )
            report = build_grounding_report(
                guard,
                registry,
                decision,
                transport=(
                    declared_transport_failure
                    if declared_visual_asset_ids
                    else {"not_applicable": True}
                ),
                warnings=registry.get("warnings", []),
                effective_sampling=_grounding_effective_sampling(
                    [], guard, model_config, guarded_attempted=False
                ),
            )
            metadata = _attach_grounding_report(packet, report)
            return packet, raw_output, reasoning_text, fallback_reason, parse_valid, salvage_warning, metadata

        if guard.mode == "audit":
            audit_retry_reasons: list[str] = []
            audit_attempt_count = 1
            audit_result_index = 0
            audit_options = BackendRunOptions(
                sampling_profile=guard.sampling_profile,
                thinking_mode=_normalize_thinking_mode(thinking_mode),
                seed=guard.seed,
                enable_telemetry=True,
                require_transport=False,
                fail_closed_telemetry=False,
                visual_first=True,
                asset_registry=registry,
                call_budget=call_budget,
                stage="audit_director",
            )
            packet, raw_output, reasoning_text, fallback_reason, parse_valid, salvage_warning, _metadata = (
                _run_generation_packet_legacy(
                    model_config,
                    context,
                    target_config,
                    _audit_grounding_master_prompt(registry),
                    runtime_required,
                    max_new_tokens,
                    max_output_chars,
                    creativity_mode,
                    creative_strength,
                    thinking_mode,
                    node_id=node_id,
                    backend_options=audit_options,
                    backend_results=backend_results,
                    manage_unload=False,
                    deterministic_transport_repair=True,
                )
            )
            audit_target = _target_profile_config_to_dict(target_config)
            audit_evidence_result = backend_results[0] if backend_results else None
            validation, validation_warning = _validate_audit_grounding_response(
                audit_evidence_result,
                raw_output,
                registry,
                guard.external_evidence_json,
                raw_output,
            )
            retry_reason = ""
            if not parse_valid or bool(salvage_warning):
                retry_reason = "audit_retry:prompt_packet_was_not_strict_json"
            elif not validation.schema_valid:
                retry_reason = "audit_retry:grounding_ledger_schema_invalid"
            if (
                audit_target["target_profile"] in {"ltx", "minimax_h3"}
                and guard.retry_on_uncertain
                and bool(retry_reason)
                and call_budget.attempted_calls < call_budget.max_calls
            ):
                audit_retry_reasons.append(retry_reason)
                audit_retry_options = copy.copy(audit_options)
                audit_retry_options.stage = "audit_director_retry"
                audit_retry_options.thinking_mode = "off"
                audit_result_index = len(backend_results)
                audit_attempt_count += 1
                (
                    packet,
                    raw_output,
                    reasoning_text,
                    fallback_reason,
                    parse_valid,
                    salvage_warning,
                    _metadata,
                ) = _run_generation_packet_legacy(
                    model_config,
                    context,
                    target_config,
                    _audit_grounding_retry_master_prompt(registry),
                    runtime_required,
                    max_new_tokens,
                    max_output_chars,
                    creativity_mode,
                    creative_strength,
                    "off",
                    node_id=node_id,
                    backend_options=audit_retry_options,
                    backend_results=backend_results,
                    manage_unload=False,
                    deterministic_transport_repair=True,
                )
            audit_evidence_result = (
                backend_results[audit_result_index]
                if len(backend_results) > audit_result_index
                else (backend_results[-1] if backend_results else None)
            )
            grounding_validation_start = time.perf_counter()
            validation, validation_warning = _validate_audit_grounding_response(
                audit_evidence_result,
                raw_output,
                registry,
                guard.external_evidence_json,
                raw_output,
            )
            transport = audit_evidence_result.transport if audit_evidence_result is not None else {}
            telemetry_error = "; ".join(
                item
                for item in (_telemetry_error_text(result) for result in backend_results)
                if item
            )
            telemetry_refusal = any(_telemetry_refusal(result) for result in backend_results)
            decision = decide_grounding_guard(
                guard,
                validation,
                media_present=True,
                transport_confirmed=bool(transport.get("pixel_transport_confirmed")),
                telemetry_error=telemetry_error,
                telemetry_refusal=telemetry_refusal,
            )
            _director_metric_add(
                "grounding_validation_seconds",
                time.perf_counter() - grounding_validation_start,
            )
            report = build_grounding_report(
                guard,
                registry,
                decision,
                validation,
                transport=transport,
                trajectory_summary=_trajectory_summary(backend_results),
                attempt_count=audit_attempt_count,
                retry_reasons=audit_retry_reasons,
                timings=_grounding_timings(backend_results),
                forward_call_count=_grounding_forward_count(backend_results),
                warnings=([validation_warning] if validation_warning else []),
                effective_sampling=_grounding_effective_sampling(
                    backend_results, guard, model_config, guarded_attempted=True
                ),
                verification_level=(
                    "transport+structured_self_report+external_evidence"
                    if validation.accepted_external_claim_count > 0
                    else "transport+structured_self_report"
                ),
            )
            report["evidence_report_id"] = _grounding_report_id(validation.ledger, registry)
            # The budget is authoritative in production.  Keeping the observed
            # detailed-result count as a lower bound also makes reporting robust
            # to injected/custom backends that append results without claiming
            # the shared budget themselves.
            report["model_call_count"] = max(
                call_budget.attempted_calls,
                len(backend_results),
            )
            report = compact_grounding_report(report)
            packet.pop("grounding_ledger", None)
            report, _trace_error = _maybe_save_grounding_trace(
                guard,
                report,
                context,
                registry,
                [
                    result.decoded_text
                    for result in backend_results
                    if result.stage in {"audit_director", "audit_director_retry"}
                ],
                raw_output,
                backend_results,
                model_config,
            )
            metadata = _attach_grounding_report(packet, report)
            return packet, raw_output, reasoning_text, fallback_reason, parse_valid, salvage_warning, metadata

        # Strict mode: evidence from pixels first, then a pixel-free compiler.
        retry_reasons: list[str] = []
        validation = None
        decision = None
        transport: dict[str, Any] = {}
        telemetry_error = ""
        final_registry = registry
        final_evidence_context = context
        attempts_used = 0
        evidence_transport_repairs: list[dict[str, Any]] = []
        max_attempts = 2 if guard.retry_on_uncertain else 1
        for attempt_index in range(max_attempts):
            attempt_context = context if attempt_index == 0 else _focused_retry_context(context)
            attempt_registry = _grounding_asset_registry(attempt_context)
            final_registry = attempt_registry
            final_evidence_context = attempt_context
            attempt_options = BackendRunOptions(
                sampling_profile=guard.sampling_profile,
                thinking_mode=_normalize_thinking_mode(thinking_mode),
                seed=(guard.seed + attempt_index) & ((1 << 64) - 1),
                enable_telemetry=True,
                require_transport=True,
                fail_closed_telemetry=True,
                visual_first=True,
                asset_registry=attempt_registry,
                call_budget=call_budget,
                stage=f"evidence_attempt_{attempt_index + 1}",
            )
            attempts_used += 1
            try:
                call_budget.claim()
                result = _run_backend_detailed(
                    model_config,
                    _strict_evidence_prompt(attempt_registry, retry=attempt_index > 0),
                    _media_context_from_gemma_context(attempt_context),
                    _strict_evidence_max_new_tokens(
                        int(max_new_tokens),
                        retry=attempt_index > 0,
                        evidence_token_budget=guard.evidence_token_budget,
                    ),
                    attempt_options,
                    node_id=node_id,
                )
                result.stage = attempt_options.stage
                backend_results.append(result)
                evidence_responses.append(result.decoded_text)
                transport = result.transport
                telemetry_error = _telemetry_error_text(result)
                grounding_validation_start = time.perf_counter()
                try:
                    attempt_transport_repairs: list[str] = []
                    evidence_answer = _strict_evidence_json_payload(
                        result.decoded_text,
                        repair_log=attempt_transport_repairs,
                    )
                    if attempt_transport_repairs:
                        evidence_transport_repairs.append(
                            {
                                "attempt": attempt_index + 1,
                                "repairs": list(attempt_transport_repairs),
                            }
                        )
                    ledger = extract_grounding_ledger(
                        evidence_answer,
                        allow_combined=False,
                    )
                    extraction_error = ""
                except GroundingLedgerError as exc:
                    ledger = {}
                    extraction_error = str(exc)
                validation = validate_grounding_ledger(
                    ledger,
                    attempt_registry,
                    guard.external_evidence_json,
                    final_text=result.decoded_text,
                )
                decision = decide_grounding_guard(
                    guard,
                    validation,
                    media_present=True,
                    transport_confirmed=bool(transport.get("pixel_transport_confirmed")),
                    telemetry_error=telemetry_error,
                    telemetry_refusal=_telemetry_refusal(result),
                )
                _director_metric_add(
                    "grounding_validation_seconds",
                    time.perf_counter() - grounding_validation_start,
                )
                if not decision.would_block:
                    break
                specific = decision.blocked_reasons[-1] if decision.blocked_reasons else "visual_grounding_unverified"
                if extraction_error:
                    retry_reasons.append(f"{specific}:{extraction_error}")
                else:
                    retry_reasons.append(specific)
                if specific in {"visual_transport_error", "visual_grounding_telemetry_error"}:
                    break
            except VisualTransportError as exc:
                transport = exc.transport
                retry_reasons.append("visual_transport_error")
                decision = decide_grounding_guard(
                    guard,
                    None,
                    media_present=True,
                    transport_confirmed=False,
                )
                break
            except Exception as exc:
                message = str(exc)
                if "visual_grounding_telemetry_error" in message:
                    telemetry_error = message
                    retry_reasons.append("visual_grounding_telemetry_error")
                    decision = decide_grounding_guard(
                        guard,
                        validation,
                        media_present=True,
                        transport_confirmed=bool(transport.get("pixel_transport_confirmed")),
                        telemetry_error=telemetry_error,
                    )
                    break
                else:
                    retry_reasons.append(f"visual_grounding_uncertain:{message}")
                    decision = GroundingGuardDecision(
                        mode="strict",
                        analysis_status="uncertain",
                        decision="block",
                        would_block=True,
                        blocked_reasons=(
                            "visual_grounding_unverified",
                            "visual_grounding_uncertain",
                        ),
                        warnings=(f"evidence_backend_failure:{message}",),
                    )

        if decision is None:
            decision = decide_grounding_guard(
                guard,
                validation,
                media_present=True,
                transport_confirmed=bool(transport.get("pixel_transport_confirmed")),
                telemetry_error=telemetry_error,
            )

        evidence_report_id = _grounding_report_id(
            validation.ledger if validation is not None else None,
            final_registry,
        )
        report = build_grounding_report(
            guard,
            final_registry,
            decision,
            validation,
            transport=transport,
            trajectory_summary=_trajectory_summary(backend_results),
            attempt_count=attempts_used,
            retry_reasons=retry_reasons,
            timings=_grounding_timings(backend_results),
            forward_call_count=_grounding_forward_count(backend_results),
            effective_sampling=_grounding_effective_sampling(
                backend_results, guard, model_config, guarded_attempted=True
            ),
            verification_level=(
                "transport+structured_self_report+external_evidence"
                if validation is not None and validation.accepted_external_claim_count > 0
                else "transport+structured_self_report"
            ),
        )
        report["evidence_report_id"] = evidence_report_id
        report["model_call_count"] = call_budget.attempted_calls
        if evidence_transport_repairs:
            report["evidence_transport_repairs"] = evidence_transport_repairs
        if attempts_used > 1:
            report["initial_asset_registry"] = registry
        report = compact_grounding_report(report)

        if decision.would_block or validation is None or not validation.grounded:
            report, _trace_error = _maybe_save_grounding_trace(
                guard,
                report,
                final_evidence_context,
                final_registry,
                evidence_responses,
                "",
                backend_results,
                model_config,
            )
            packet = _blocked_grounding_packet(target_config, report)
            metadata = packet["metadata"]
            raw_output = evidence_responses[-1] if evidence_responses else ""
            return packet, raw_output, "", "", False, "", metadata

        verified_ledger = copy.deepcopy(validation.ledger or {})
        fact_assets_by_id = _eligible_grounding_fact_assets(verified_ledger)
        fact_ids = list(fact_assets_by_id)
        required_asset_ids = [
            str(asset.get("asset_id"))
            for asset in final_registry.get("assets", [])
            if isinstance(asset, dict)
            and asset.get("asset_id")
            and asset.get("required_for_grounding", True)
        ]
        compiler_metadata = _strict_compiler_metadata(
            context,
            verified_ledger,
            evidence_report_id,
            transport,
            final_registry,
        )
        compiler_context = GemmaContext(
            user_prompt=context.user_prompt,
            images=None,
            source=context.source,
            media_metadata=compiler_metadata,
            visual_description="",
            warnings=[],
        )
        used_fact_ids: list[str] = []
        compiler_response = ""
        native_h3_compiler = (
            _target_profile_config_to_dict(target_config)["target_profile"]
            == "minimax_h3"
        )
        try:
            compiler_attempt_index = 0
            while True:
                compiler_options = BackendRunOptions(
                    sampling_profile=guard.sampling_profile,
                    thinking_mode=_normalize_thinking_mode(thinking_mode),
                    # The shared call ordinal is added by _run_backend_for_packet;
                    # keep the base stable so a retry receives the next adjacent seed.
                    seed=(guard.seed + 1000) & ((1 << 64) - 1),
                    enable_telemetry=True,
                    require_transport=False,
                    fail_closed_telemetry=True,
                    visual_first=True,
                    asset_registry=build_asset_registry({}, 0),
                    call_budget=call_budget,
                    stage="compiler" if compiler_attempt_index == 0 else "compiler_retry",
                )
                remaining_repairs = max(
                    0, call_budget.max_calls - call_budget.attempted_calls - 1
                )
                packet, compiler_response, reasoning_text, fallback_reason, parse_valid, salvage_warning, _metadata = (
                    _run_generation_packet_legacy(
                        model_config,
                        compiler_context,
                        target_config,
                        master_prompt,
                        True,
                        max_new_tokens,
                        max_output_chars,
                        creativity_mode,
                        creative_strength,
                        thinking_mode,
                        node_id=node_id,
                        backend_options=compiler_options,
                        backend_results=backend_results,
                        manage_unload=False,
                        max_refinement_attempts_override=remaining_repairs,
                        refinement_evidence_context=_json_dumps(verified_ledger),
                        deterministic_transport_repair=True,
                        native_h3_output=native_h3_compiler,
                    )
                )
                if parse_valid and not salvage_warning:
                    break
                if compiler_attempt_index >= 1 or call_budget.attempted_calls >= call_budget.max_calls:
                    raise GroundingLedgerError(
                        "visual_grounding_schema_invalid:compiler_native_h3_output_invalid"
                        if native_h3_compiler
                        else "visual_grounding_schema_invalid:compiler_packet_was_not_strict_json"
                    )
                retry_reasons.append(
                    "compiler_retry:compiler_native_h3_output_invalid"
                    if native_h3_compiler
                    else "compiler_retry:compiler_packet_was_not_strict_json"
                )
                compiler_attempt_index += 1
            compiler_packet_metadata = (
                packet.get("metadata") if isinstance(packet.get("metadata"), dict) else {}
            )
            provenance_validation_start = time.perf_counter()
            used_fact_ids = _validated_compiler_provenance(
                compiler_packet_metadata,
                evidence_report_id,
                fact_ids,
                fact_assets_by_id,
                required_asset_ids,
            )
            _director_metric_add(
                "grounding_validation_seconds",
                time.perf_counter() - provenance_validation_start,
            )
            if not compiler_packet_metadata.get("ready_for_generation"):
                compiler_blocked_reasons: list[str] = []
                refinement = compiler_packet_metadata.get("minimax_h3_refinement")
                if isinstance(refinement, dict):
                    compiler_blocked_reasons.extend(
                        reason
                        for reason in _reason_strings(refinement.get("candidate_reasons"))
                        if reason not in compiler_blocked_reasons
                    )
                compiler_blocked_reasons.extend(
                    reason
                    for reason in _reason_strings(compiler_packet_metadata.get("blocked_reasons"))
                    if reason not in compiler_blocked_reasons
                )
                compiler_blocked_reasons = [
                    reason
                    for reason in compiler_blocked_reasons
                    if re.fullmatch(r"[a-z0-9][a-z0-9_.:-]{0,159}", reason)
                ][:8]
                compiler_reason_detail = ",".join(compiler_blocked_reasons)
                failure_reason = "visual_grounding_uncertain:compiler_packet_not_ready"
                if compiler_reason_detail:
                    failure_reason = f"{failure_reason}:{compiler_reason_detail}"
                raise GroundingLedgerError(
                    failure_reason
                )
        except Exception as exc:
            retry_reasons.append(f"compiler_failure:{exc}")
            compiler_failure_text = str(exc).lower()
            if "telemetry" in compiler_failure_text:
                compiler_failure_reason = "visual_grounding_telemetry_error"
            elif "visual_grounding_schema_invalid" in compiler_failure_text:
                compiler_failure_reason = "visual_grounding_schema_invalid"
            else:
                compiler_failure_reason = "visual_grounding_uncertain"
            failed_decision = GroundingGuardDecision(
                mode="strict",
                analysis_status="uncertain",
                decision="block",
                would_block=True,
                blocked_reasons=("visual_grounding_unverified", compiler_failure_reason),
                warnings=(f"compiler_failure:{exc}",),
            )
            report = build_grounding_report(
                guard,
                final_registry,
                failed_decision,
                validation,
                transport=transport,
                trajectory_summary=_trajectory_summary(backend_results),
                attempt_count=attempts_used,
                retry_reasons=retry_reasons,
                timings=_grounding_timings(backend_results),
                forward_call_count=_grounding_forward_count(backend_results),
                effective_sampling=_grounding_effective_sampling(
                    backend_results, guard, model_config, guarded_attempted=True
                ),
                verification_level=(
                    "transport+structured_self_report+external_evidence"
                    if validation.accepted_external_claim_count > 0
                    else "transport+structured_self_report"
                ),
            )
            report["evidence_report_id"] = evidence_report_id
            report["model_call_count"] = call_budget.attempted_calls
            if attempts_used > 1:
                report["initial_asset_registry"] = registry
            report = compact_grounding_report(report)
            report, _trace_error = _maybe_save_grounding_trace(
                guard,
                report,
                final_evidence_context,
                final_registry,
                evidence_responses,
                compiler_response,
                backend_results,
                model_config,
            )
            packet = _blocked_grounding_packet(target_config, report)
            metadata = packet["metadata"]
            return packet, "", "", str(exc), False, "", metadata

        if call_budget.attempted_calls > 4:
            raise RuntimeError("Grounding Guard exceeded its hard four-call ceiling.")
        packet.pop("grounding_ledger", None)
        packet_metadata = packet.get("metadata") if isinstance(packet.get("metadata"), dict) else {}
        packet_metadata["grounding_evidence_report_id"] = evidence_report_id
        packet_metadata["used_grounding_fact_ids"] = used_fact_ids
        packet_metadata["available_grounding_fact_ids"] = fact_ids
        packet_metadata["grounding_compiler_pixels_sent_to_backend"] = False
        report = build_grounding_report(
            guard,
            final_registry,
            decision,
            validation,
            transport=transport,
            trajectory_summary=_trajectory_summary(backend_results),
            attempt_count=attempts_used,
            retry_reasons=retry_reasons,
            timings=_grounding_timings(backend_results),
            forward_call_count=_grounding_forward_count(backend_results),
            effective_sampling=_grounding_effective_sampling(
                backend_results, guard, model_config, guarded_attempted=True
            ),
            verification_level=(
                "transport+structured_self_report+external_evidence"
                if validation.accepted_external_claim_count > 0
                else "transport+structured_self_report"
            ),
        )
        report["evidence_report_id"] = evidence_report_id
        report["used_fact_ids"] = used_fact_ids
        report["available_fact_ids"] = fact_ids
        report["model_call_count"] = call_budget.attempted_calls
        compiler_transport_repairs = packet_metadata.get(
            "deterministic_transport_repairs"
        )
        if isinstance(compiler_transport_repairs, list) and compiler_transport_repairs:
            report["compiler_deterministic_transport_repairs"] = list(
                compiler_transport_repairs
            )
        host_structural_repairs = packet_metadata.get("host_structural_repairs")
        if isinstance(host_structural_repairs, list) and host_structural_repairs:
            report["compiler_host_structural_repairs"] = list(
                host_structural_repairs
            )
        report["compiler_transport"] = {
            "pixels_sent_to_backend": False,
            "evidence_source": "validated_ledger_only",
        }
        if attempts_used > 1:
            report["initial_asset_registry"] = registry
        report = compact_grounding_report(report)
        report, _trace_error = _maybe_save_grounding_trace(
            guard,
            report,
            final_evidence_context,
            final_registry,
            evidence_responses,
            compiler_response,
            backend_results,
            model_config,
        )
        packet["metadata"] = packet_metadata
        metadata = _attach_grounding_report(packet, report)
        return packet, compiler_response, reasoning_text, fallback_reason, parse_valid, salvage_warning, metadata
    finally:
        _maybe_unload(model_config)


class DiffusionGemmaModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {
                        "default": DEFAULT_ADVANCED_MODEL_PATH,
                        "multiline": False,
                        "tooltip": "Whole Hugging Face repo folder for the in-process Transformers/NVFP4 runtime.",
                    },
                ),
                "backend": (
                    [
                        "transformers_inprocess",
                    ],
                    {"default": "transformers_inprocess"},
                ),
                "dtype": (["auto", "bfloat16", "float16", "float32"], {"default": "auto"}),
                "quantization": (
                    ["modelopt_nvfp4", "none", "bitsandbytes_4bit", "quanto", "nvfp4_metadata_only"],
                    {"default": "modelopt_nvfp4"},
                ),
                "local_files_only": ("BOOLEAN", {"default": True}),
                "unload_policy": (["keep_loaded", "unload_after_run"], {"default": "keep_loaded"}),
                "max_memory_gb": (
                    MEMORY_PRESET_CHOICES,
                    {
                        "default": MEMORY_PRESET_SAFE_20,
                        "tooltip": "NVFP4 bridge memory profile. 20 GB is the safest practical default; 18 GB is the supported floor; 0 requests full GPU load.",
                    },
                ),
            },
            "optional": {
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "GGUF compatibility temperature. In-process DiffusionGemma uses the native denoising sampling profile selected by Grounding Guard instead.",
                    },
                ),
            },
        }

    RETURN_TYPES = (MODEL_TYPE, "STRING")
    RETURN_NAMES = ("model_config", "runtime_status_json")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(
        self,
        model_path: str,
        backend: str = "transformers_inprocess",
        dtype: str = "auto",
        quantization: str = "modelopt_nvfp4",
        local_files_only: bool = True,
        unload_policy: str = "keep_loaded",
        max_memory_gb: Any = MEMORY_PRESET_SAFE_20,
        cli_path: str = "",
        extra_args: str = "",
        temperature: float = 0.45,
    ):
        valid_backends = {"transformers_inprocess", "gguf_subprocess", "qwen_vl_placeholder", "minicpm_v_placeholder", "template"}
        valid_dtypes = {"auto", "bfloat16", "float16", "float32"}
        valid_quantization = {"none", "modelopt_nvfp4", "bitsandbytes_4bit", "quanto", "nvfp4_metadata_only", "gguf_q4_q5"}
        valid_unload = {"keep_loaded", "unload_after_run"}
        backend = backend if backend in valid_backends else ("gguf_subprocess" if Path(DEFAULT_GGUF_MODEL_PATH).exists() else "transformers_inprocess")
        dtype = dtype if dtype in valid_dtypes else "auto"
        quantization = quantization if quantization in valid_quantization else "none"
        unload_policy = unload_policy if unload_policy in valid_unload else "keep_loaded"
        cli_path = (cli_path or "").strip()
        extra_args = (extra_args or "").strip()
        if cli_path.startswith("-") or (cli_path and not Path(cli_path).exists() and Path(DEFAULT_GGUF_CLI_PATH).exists()):
            if not extra_args or extra_args == cli_path:
                extra_args = cli_path if cli_path.startswith("-") else extra_args
            cli_path = DEFAULT_GGUF_CLI_PATH
        if not cli_path and backend == "gguf_subprocess":
            cli_path = DEFAULT_GGUF_CLI_PATH
        if not extra_args and backend == "gguf_subprocess":
            extra_args = DEFAULT_GGUF_EXTRA_ARGS
        temp_value = max(0.0, min(2.0, _safe_float(temperature, 0.45)))
        memory_gb = _memory_preset_to_gb(max_memory_gb)
        effective_extra_args = _extra_args_with_temperature(extra_args, temp_value)
        config = RuntimeConfig(
            model_path=model_path.strip(),
            backend=backend,
            dtype=dtype,
            quantization=quantization,
            local_files_only=bool(local_files_only),
            unload_policy=unload_policy,
            max_memory_gb=memory_gb,
            cli_path=cli_path,
            extra_args=effective_extra_args,
            temperature=temp_value,
        )
        config.status = _runtime_status(config)
        return (config, _json_dumps(config.status))


class DiffusionGemmaSimpleModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {
                        "default": DEFAULT_MODEL_PATH,
                        "multiline": False,
                        "tooltip": "Path to the local DiffusionGemma GGUF file.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Generation randomness. 0.2 is literal; 0.45 balanced; 0.65+ is more creative.",
                    },
                ),
            },
        }

    RETURN_TYPES = (MODEL_TYPE, "STRING")
    RETURN_NAMES = ("model_config", "runtime_status_json")
    FUNCTION = "load"
    CATEGORY = COMPATIBILITY_CATEGORY

    def load(self, model_path: str, temperature: float = 0.45):
        return DiffusionGemmaModelLoader().load(
            model_path,
            "gguf_subprocess",
            "auto",
            "none",
            True,
            "keep_loaded",
            0.0,
            DEFAULT_GGUF_CLI_PATH,
            DEFAULT_GGUF_EXTRA_ARGS,
            temperature,
        )


class DiffusionGemmaModelPathLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {
                        "default": DEFAULT_MODEL_PATH,
                        "multiline": False,
                        "tooltip": "Path to the local DiffusionGemma GGUF file. Sampling temperature is set on DiffusionGemma CoT Generator.",
                    },
                ),
            },
        }

    RETURN_TYPES = (MODEL_TYPE, "STRING")
    RETURN_NAMES = ("model_config", "runtime_status_json")
    FUNCTION = "load"
    CATEGORY = COMPATIBILITY_CATEGORY

    def load(self, model_path: str):
        return DiffusionGemmaModelLoader().load(
            model_path,
            "gguf_subprocess",
            "auto",
            "none",
            True,
            "keep_loaded",
            0.0,
            DEFAULT_GGUF_CLI_PATH,
            DEFAULT_GGUF_EXTRA_ARGS,
            0.45,
        )


class DiffusionGemmaMediaSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sample_fps": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 24.0, "step": 0.1}),
                "max_duration_seconds": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 600.0, "step": 1.0}),
                "max_frames": ("INT", {"default": 60, "min": 1, "max": 512}),
                "overlong_policy": (["trim", "reject"], {"default": "trim"}),
                "include_image_with_video": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "visual_description": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                "media_synthesis_mode": (MEDIA_SYNTHESIS_MODE_CHOICES, {"default": "video_recreation"}),
            },
        }

    RETURN_TYPES = (MEDIA_TYPE, "STRING")
    RETURN_NAMES = ("media_context", "media_metadata_json")
    FUNCTION = "sample"
    CATEGORY = OPTIONAL_CATEGORY

    def sample(
        self,
        sample_fps: float,
        max_duration_seconds: float,
        max_frames: int,
        overlong_policy: str,
        include_image_with_video: bool,
        image=None,
        video=None,
        visual_description: str = "",
        media_synthesis_mode: str = "video_recreation",
    ):
        synthesis_mode = _normalize_media_synthesis_mode(media_synthesis_mode)
        metadata: dict[str, Any] = {
            "source": "none",
            "sample_fps": float(sample_fps),
            "max_duration_seconds": float(max_duration_seconds),
            "max_frames": int(max_frames),
            "media_synthesis_mode": synthesis_mode,
            "image_role": "none",
            "video_role": "none",
            "reference_image_count": 0,
            "reference_image_backend_attached": False,
            "video_sampled_frame_count": 0,
            "synthesis_contract": _synthesis_contract_text({"media_synthesis_mode": synthesis_mode}),
            "warnings": [],
        }
        sampled = None

        if video is not None:
            duration = _safe_float(video.get_duration(), 0.0)
            metadata["source"] = "video"
            metadata["duration_seconds"] = duration
            try:
                metadata["frame_count"] = int(video.get_frame_count())
            except Exception as exc:
                metadata["warnings"].append(f"Could not read frame count without materializing video: {exc}")
            try:
                width, height = video.get_dimensions()
                metadata["width"] = int(width)
                metadata["height"] = int(height)
            except Exception as exc:
                metadata["warnings"].append(f"Could not read video dimensions: {exc}")

            source_video = video
            if duration > max_duration_seconds:
                if overlong_policy == "reject":
                    raise ValueError(
                        f"Video is {duration:.2f}s, above the {max_duration_seconds:.2f}s DiffusionGemma analysis cap."
                    )
                source_video = video.as_trimmed(0.0, float(max_duration_seconds), strict_duration=False)
                metadata["trimmed_to_seconds"] = float(max_duration_seconds)
                metadata["warnings"].append("Video was trimmed for DiffusionGemma analysis.")
                if source_video is None:
                    raise ValueError("Video trim failed.")

            components = source_video.get_components()
            frames = components.images
            frame_count = int(frames.shape[0])
            fps = float(components.frame_rate)
            stride = max(1, int(round(fps / max(sample_fps, 0.1))))
            indices = list(range(0, frame_count, stride))
            if len(indices) > max_frames:
                step = max(1, math.ceil(len(indices) / max_frames))
                indices = indices[::step][:max_frames]
            try:
                import torch

                index_tensor = torch.tensor(indices, dtype=torch.long, device=frames.device)
                sampled = frames.index_select(0, index_tensor)
            except Exception:
                sampled = frames[indices]
            metadata["source_fps"] = fps
            metadata["sampled_indices"] = indices
            metadata["sampled_frame_count"] = len(indices)
            metadata["video_sampled_frame_count"] = len(indices)
            metadata["video_role"] = (
                "control_structure_pose_depth_canny_composition_motion_camera"
                if synthesis_mode == "image_identity_video_control"
                else "primary_video_recreation_source"
            )

        if image is not None:
            metadata["reference_image_count"] = _media_image_count(image)
            metadata["image_role"] = (
                "identity_reference"
                if synthesis_mode == "image_identity_video_control"
                else ("optional_still_reference" if sampled is not None else "primary_image_source")
            )
            should_include_image = bool(include_image_with_video or (synthesis_mode == "image_identity_video_control" and sampled is not None))
            if sampled is not None and should_include_image:
                try:
                    import torch

                    reference_for_backend, reference_report = _fit_reference_image_to_video_frames(image, sampled)
                    metadata.update(reference_report)
                    sampled = torch.cat([reference_for_backend, sampled], dim=0)
                    metadata["source"] = "image+video"
                    metadata["sampled_frame_count"] = int(sampled.shape[0])
                    metadata["reference_image_backend_attached"] = True
                    if metadata.get("reference_image_resized_for_video_control"):
                        metadata["warnings"].append(
                            "Reference image was aspect-fitted and padded to match sampled video frame dimensions for synthesis evidence."
                        )
                except Exception as exc:
                    metadata["warnings"].append(f"Could not prepend image to sampled video frames: {exc}")
            elif sampled is None:
                sampled = image
                metadata["source"] = "image"
                metadata["sampled_frame_count"] = int(image.shape[0]) if hasattr(image, "shape") else 1
                metadata["reference_image_backend_attached"] = True
        elif synthesis_mode == "image_identity_video_control":
            metadata["image_role"] = "missing_identity_reference"

        if video is None and synthesis_mode == "image_identity_video_control":
            metadata["video_role"] = "missing_control_video"

        cleaned_visual_description = _clean_visual_description(visual_description)
        if cleaned_visual_description:
            metadata["visual_description"] = cleaned_visual_description
            metadata["visual_description_source"] = "upstream_text"

        context = MediaContext(images=sampled, source=metadata["source"], metadata=metadata)
        return (context, _json_dumps(metadata))


class DiffusionGemmaContextHub:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True, "dynamicPrompts": True},
                ),
                "sample_fps": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 24.0, "step": 0.1}),
                "max_duration_seconds": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 600.0, "step": 1.0}),
                "max_frames": ("INT", {"default": 60, "min": 1, "max": 512}),
                "overlong_policy": (["trim", "reject"], {"default": "trim"}),
                "include_image_with_video": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "For LTX-2.5, this is the first-frame anchor. Leave it empty for text-to-video. Connect the same image to the native LTX first-frame conditioning node.",
                    },
                ),
                "video": ("VIDEO",),
                "visual_description": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                "media_synthesis_mode": (MEDIA_SYNTHESIS_MODE_CHOICES, {"default": "video_recreation"}),
                "last_frame_image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional LTX-2.5 last-frame anchor. Auto mode selects first+last-frame video only when this dedicated socket and the first-frame image socket are both connected.",
                    },
                ),
            },
        }

    RETURN_TYPES = (CONTEXT_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("gemma_context", "context_json", "context_preview")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(
        self,
        user_prompt: str,
        sample_fps: float = 1.0,
        max_duration_seconds: float = 60.0,
        max_frames: int = 60,
        overlong_policy: str = "trim",
        include_image_with_video: bool = True,
        image=None,
        video=None,
        visual_description: str = "",
        media_synthesis_mode: str = "video_recreation",
        last_frame_image=None,
    ):
        media_context, _metadata_json = DiffusionGemmaMediaSampler().sample(
            sample_fps,
            max_duration_seconds,
            max_frames,
            overlong_policy,
            include_image_with_video,
            image,
            video,
            visual_description,
            media_synthesis_mode,
        )
        ordinary_ltx_frames = video is None and _normalize_media_synthesis_mode(media_synthesis_mode) != "image_identity_video_control"
        first_attached = bool(ordinary_ltx_frames and image is not None)
        last_attached = bool(ordinary_ltx_frames and last_frame_image is not None)
        if ordinary_ltx_frames and last_frame_image is not None:
            try:
                import torch

                last = last_frame_image[:1] if len(getattr(last_frame_image, "shape", ())) >= 4 else last_frame_image
                if image is not None:
                    first = image[:1] if len(getattr(image, "shape", ())) >= 4 else image
                    fitted_last, fit_report = _fit_reference_image_to_video_frames(last, first)
                    media_context.images = torch.cat([first, fitted_last], dim=0)
                    if fit_report.get("reference_image_resized_for_video_control"):
                        media_context.metadata["ltx_last_frame_resized_for_analysis"] = True
                        media_context.metadata["ltx_last_frame_analysis_resize_mode"] = fit_report.get(
                            "reference_image_resize_mode", "aspect_fit_replicate_pad"
                        )
                else:
                    media_context.images = last
                media_context.source = "image"
                media_context.metadata["source"] = "image"
            except Exception as exc:
                media_context.metadata.setdefault("warnings", []).append(
                    f"Could not attach the LTX last-frame image for Director analysis: {exc}"
                )
        if ordinary_ltx_frames:
            roles: list[str] = []
            image_roles: list[str] = []
            if first_attached:
                roles.append("first_frame")
                image_roles.append(
                    "LTX first-frame anchor: identity, appearance, objects, count, environment, lighting, color, spatial composition, camera, and framing "
                    "[dg:identity,appearance,object,count,environment,lighting,color,spatial,composition,camera]"
                )
            if last_attached:
                roles.append("last_frame")
                image_roles.append(
                    "LTX last-frame anchor: identity, appearance, objects, count, environment, lighting, color, spatial composition, camera, and framing "
                    "[dg:identity,appearance,object,count,environment,lighting,color,spatial,composition,camera]"
                )
            mode_hint = (
                "first_last_frame"
                if last_attached
                else "image_to_video"
                if first_attached
                else "text_to_video"
            )
            first_shape = _image_batch_hwc_shape(image) if first_attached else None
            last_shape = _image_batch_hwc_shape(last_frame_image) if last_attached else None
            media_context.metadata.update(
                {
                    "ltx25_context_schema": LTX25_CONTRACT_SCHEMA,
                    "ltx_generation_mode_hint": mode_hint,
                    "ltx_first_frame_attached": first_attached,
                    "ltx_last_frame_attached": last_attached,
                    "ltx_frame_pair_attached": bool(first_attached and last_attached),
                    "ltx_frame_roles": roles,
                    "image_roles": image_roles,
                    "reference_image_count": _media_image_count(media_context.images),
                    "reference_image_backend_attached": bool(roles and media_context.images is not None),
                    "sampled_frame_count": _media_image_count(media_context.images),
                }
            )
            if first_shape:
                media_context.metadata["ltx_first_frame_height"] = first_shape[0]
                media_context.metadata["ltx_first_frame_width"] = first_shape[1]
            if last_shape:
                media_context.metadata["ltx_last_frame_height"] = last_shape[0]
                media_context.metadata["ltx_last_frame_width"] = last_shape[1]
        context = _gemma_context_from_media(user_prompt, media_context, visual_description)
        if context.images is not None:
            note = "Media pixels are sampled and carried in DG_CONTEXT; the generator decides whether the active backend can consume them."
            if note not in context.warnings:
                context.warnings.append(note)
            context.media_metadata["warnings"] = context.warnings
        return (context, _json_dumps(_gemma_context_to_jsonable(context)), _gemma_context_preview(context))


class DiffusionGemmaH3ReferencePolicy:
    """Reusable dropdown rules for Ref2VA reference-role assignment."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout": (
                    list(_MINIMAX_H3_REFERENCE_POLICY_CHOICES),
                    {
                        "default": "Auto (recommended)",
                        "tooltip": "Auto derives a safe manifest from the ordered visual references carried by DG_CONTEXT. Contact-sheet choices treat the entire sheet as one Picture and prevent its grid or labels from becoming scene content.",
                    },
                ),
                "expected_subject_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": _MINIMAX_H3_MAX_EXPECTED_SUBJECT_COUNT,
                        "step": 1,
                        "tooltip": "0 is Auto. For a primary subject plus K selected secondary contact-sheet entities, enter 1 + K. Repeated views of one entity still count once.",
                    },
                ),
                "custom_manifest": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Used only by Custom manifest. Write one exact <Picture N>, <Video N>, or <Audio N> role per line. Named dropdown layouts ignore this box.",
                    },
                ),
            }
        }

    RETURN_TYPES = (H3_REFERENCE_POLICY_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("reference_policy_config", "policy_json", "policy_preview")
    FUNCTION = "build"
    CATEGORY = OPTIONAL_CATEGORY
    DESCRIPTION = (
        "Assigns reusable MiniMax H3 Ref2VA reference roles without hand-writing the Context Hub manifest. "
        "Connect it to the Director's H3 reference policy socket."
    )

    def build(
        self,
        layout: str = "Auto (recommended)",
        expected_subject_count: int = 0,
        custom_manifest: str = "",
    ):
        config = _make_h3_reference_policy_config(
            layout,
            custom_manifest,
            expected_subject_count,
        )
        payload = _h3_reference_policy_config_to_dict(config)
        resolved_layout = payload["layout"]
        if resolved_layout == "custom_manifest":
            preview = payload["custom_manifest"] or "Custom manifest is selected; enter at least one visual asset role."
        elif resolved_layout == "one_picture_all_visual_attributes":
            preview = _minimax_h3_reference_policy_manifest(config, 1)
        elif resolved_layout in {
            "primary_subject_environment_style",
            "primary_subject_same_subject_contact_sheet",
            "primary_subject_multi_entity_contact_sheet",
        }:
            preview = _minimax_h3_reference_policy_manifest(config, 2)
        elif resolved_layout == "ordered_independent_subjects_objects":
            preview = (
                "At queue time, each ordered Picture becomes the authority for one independent subject or hero object. "
                "Picture count is read from DG_CONTEXT."
            )
        else:
            preview = (
                "At queue time, Director reads the actual DG_CONTEXT inventory: one Picture becomes the complete visual "
                "source; multiple Pictures become a primary reference plus ordered supporting references; attached Video "
                "evidence receives motion/camera/temporal duties."
            )
        preview = (
            f"Layout: {resolved_layout}\nExpected semantic Subjects: "
            f"{payload['expected_subject_count'] or 'Auto'}\n{preview}"
        )
        return (config, _json_dumps(payload), preview)


class DiffusionGemmaH3ReferenceContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True, "dynamicPrompts": True},
                ),
                "reference_manifest": (
                    "STRING",
                    {
                        "default": "<Picture 1>: [dg:identity,appearance] protagonist identity, face, hair, body, wardrobe, and distinguishing traits; preserve fully.",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "One role per connected Ref2VA asset. Use exact one-based tags and describe what must or must not transfer. Strict grounding requires every Picture and Video role to include a host category annotation such as [dg:identity,appearance], [dg:action,motion,camera,temporal], or [dg:object,color,text]. These annotations are stripped before compilation. Paired video soundtracks consume Audio labels before standalone audio references.",
                    },
                ),
                "sample_fps": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 24.0, "step": 0.1}),
                "max_duration_seconds": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 600.0, "step": 1.0}),
                "max_frames": ("INT", {"default": 60, "min": 1, "max": 512}),
                "overlong_policy": (["trim", "reject"], {"default": "trim"}),
            },
            "optional": {
                "reference_images": (
                    "IMAGE",
                    {
                        "tooltip": "Optional ordered image batch for DiffusionGemma to inspect. Batch item 1 maps to <Picture 1>, and so on. Connect the same images separately to MiniMaxH3ReferenceToVideo.",
                    },
                ),
                "reference_video": (
                    "VIDEO",
                    {
                        "tooltip": "Optional reference video for DiffusionGemma to inspect as <Video 1>. The native H3 node still needs its 24-fps IMAGE frame batch connected separately.",
                    },
                ),
                "visual_description": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                "reference_manifest_preset": (
                    list(_MINIMAX_H3_REFERENCE_MANIFEST_PRESETS),
                    {
                        "default": "custom",
                        "tooltip": "Custom uses the editable reference_manifest above. The one-image preset treats one picture as the complete visual source without implying one semantic Subject. The two-image preset assigns Picture 1 to subject identity and Picture 2 to environment/style, matching the standard Ref2VA socket order.",
                    },
                ),
                "expected_subject_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": _MINIMAX_H3_MAX_EXPECTED_SUBJECT_COUNT,
                        "step": 1,
                        "tooltip": "0 is Auto. A positive value requires exactly that many semantic <Subject N> labels in Ref2VA. This counts independently tracked people, creatures, objects, environments, styles, actions, or effects—not Picture inputs and not only human characters.",
                    },
                ),
            },
        }

    RETURN_TYPES = (CONTEXT_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("gemma_context", "context_json", "context_preview")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(
        self,
        user_prompt: str,
        reference_manifest: str,
        sample_fps: float = 1.0,
        max_duration_seconds: float = 60.0,
        max_frames: int = 60,
        overlong_policy: str = "trim",
        reference_images=None,
        reference_video=None,
        visual_description: str = "",
        reference_manifest_preset: str = "custom",
        expected_subject_count: int = 0,
    ):
        expected_subject_count = _normalize_minimax_h3_expected_subject_count(
            expected_subject_count
        )
        normalized_manifest = _resolve_minimax_h3_reference_manifest(
            reference_manifest,
            reference_manifest_preset,
        )
        definitions = _minimax_h3_reference_definitions(normalized_manifest)
        manifest_reasons = _minimax_h3_reference_manifest_validation_reasons(normalized_manifest)
        auto_manifest_pending = bool(
            not normalized_manifest
            and manifest_reasons == ["minimax_h3_ref_manifest_missing"]
        )
        if auto_manifest_pending:
            # A blank inline manifest is now an intentional Auto request.  The
            # Director resolves it from the real DG_CONTEXT inventory before
            # cache lookup, grounding, or model dispatch.
            manifest_reasons = []
        reference_image_count = (
            int(reference_images.shape[0])
            if getattr(reference_images, "shape", None) is not None and len(reference_images.shape) >= 4
            else (1 if reference_images is not None else 0)
        )
        media_context, _metadata_json = DiffusionGemmaMediaSampler().sample(
            sample_fps,
            max_duration_seconds,
            max_frames,
            overlong_policy,
            True,
            reference_images if reference_video is None else None,
            reference_video,
            visual_description,
            "video_recreation",
        )
        if reference_images is not None and reference_video is not None and media_context.images is not None:
            try:
                import torch

                prepared_references = []
                resized = False
                for index in range(reference_image_count):
                    prepared, report = _fit_reference_image_to_video_frames(
                        reference_images[index : index + 1],
                        media_context.images,
                    )
                    prepared_references.append(prepared)
                    resized = resized or bool(report.get("reference_image_resized_for_video_control"))
                media_context.images = torch.cat([*prepared_references, media_context.images], dim=0)
                media_context.source = "image+video"
                media_context.metadata.update(
                    {
                        "source": "image+video",
                        "reference_image_count": reference_image_count,
                        "reference_image_backend_attached": True,
                        "sampled_frame_count": int(media_context.images.shape[0]),
                        "minimax_h3_reference_images_resized_for_video_analysis": resized,
                    }
                )
            except Exception as exc:
                media_context.metadata.setdefault("warnings", []).append(
                    f"Could not attach all H3 reference pictures beside the sampled reference video: {exc}"
                )
        else:
            media_context.metadata["reference_image_count"] = reference_image_count
        context = _gemma_context_from_media(user_prompt, media_context, visual_description)
        context.media_metadata.update(
            {
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_manifest": normalized_manifest,
                "minimax_h3_reference_manifest_source": (
                    "pending_auto" if auto_manifest_pending else "context_manifest"
                ),
                "minimax_h3_reference_manifest_preset": (
                    reference_manifest_preset
                    if str(reference_manifest_preset).strip().lower() in _MINIMAX_H3_REFERENCE_MANIFEST_PRESETS
                    else "custom"
                ),
                "minimax_h3_reference_tags": [tag for tag, _description in definitions],
                "minimax_h3_reference_manifest_reasons": manifest_reasons,
                "minimax_h3_reference_image_batch_count": reference_image_count,
                "minimax_h3_reference_video_attached_for_analysis": bool(reference_video is not None),
                "minimax_h3_expected_subject_count": expected_subject_count,
            }
        )
        if reference_video is not None:
            context.media_metadata["transformers_video_transport"] = "sampled_frame_images"
        context.source = "minimax_h3_ref2va"
        if manifest_reasons:
            warning = "H3 reference manifest needs correction: " + ", ".join(manifest_reasons)
            if warning not in context.warnings:
                context.warnings.append(warning)
        manifest_picture_count = sum(tag.startswith("<Picture ") for tag, _description in definitions)
        manifest_video_count = sum(tag.startswith("<Video ") for tag, _description in definitions)
        manifest_audio_count = sum(tag.startswith("<Audio ") for tag, _description in definitions)
        if (
            normalized_manifest
            and reference_image_count
            and reference_image_count != manifest_picture_count
        ):
            context.warnings.append(
                f"DiffusionGemma received {reference_image_count} reference picture(s) for analysis, while the manifest declares {manifest_picture_count}; the native H3 connections remain authoritative."
            )
        if normalized_manifest and reference_video is not None and manifest_video_count != 1:
            context.warnings.append(
                f"The analysis socket inspects one reference video as <Video 1>, while the manifest declares {manifest_video_count}; describe additional videos in the manifest or visual_description and connect them directly to native H3."
            )
        elif normalized_manifest and manifest_video_count > 1:
            context.warnings.append(
                "DiffusionGemma can inspect only <Video 1> through this context node; additional declared videos still connect directly to native H3 and rely on their manifest roles."
            )
        if manifest_audio_count:
            context.warnings.append(
                "Reference audio is not inspected by DiffusionGemma; its declared role is compiled from the manifest, while the audio signal connects directly to native H3."
            )
        context.media_metadata["warnings"] = context.warnings
        preview = _gemma_context_preview(context)
        preview += (
            "\nExpected semantic H3 Subjects: "
            + (str(expected_subject_count) if expected_subject_count else "Auto")
        )
        if normalized_manifest:
            preview += "\nH3 reference manifest:\n" + normalized_manifest
        elif auto_manifest_pending:
            preview += "\nH3 reference roles: Auto — Director will derive them from the attached visual references at queue time."
        return (context, _json_dumps(_gemma_context_to_jsonable(context)), preview)


class DiffusionGemmaReferencePrep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_aspect_ratio": (REFERENCE_PREP_ASPECT_CHOICES, {"default": "source"}),
                "aspect_policy": (
                    ["preserve_source", "center_crop_to_target"],
                    {
                        "default": "preserve_source",
                        "tooltip": "preserve_source keeps the original framing and only scales by long edge. center_crop_to_target crops to the selected aspect ratio before resizing.",
                    },
                ),
                "long_edge": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 256,
                        "max": 4096,
                        "step": 8,
                        "tooltip": "Recommended practical range for DiffusionGemma image ingestion is about 1024-1536 on the long edge.",
                    },
                ),
                "multiple": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "tooltip": "Rounds output width and height to this multiple.",
                    },
                ),
                "upscale_smaller_images": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Off is the efficient default: images below the requested long edge are never enlarged. Enable only when you intentionally want a fixed larger Director-analysis size.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "prep_metadata_json", "width", "height", "resolution_selector_preset")
    FUNCTION = "prepare"
    CATEGORY = OPTIONAL_CATEGORY

    def prepare(
        self,
        image: Any,
        target_aspect_ratio: str = "source",
        aspect_policy: str = "preserve_source",
        long_edge: int = 1280,
        multiple: int = 8,
        upscale_smaller_images: bool = False,
    ):
        try:
            import torch

            tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
        except Exception:
            tensor = image

        if not hasattr(tensor, "shape") or len(tensor.shape) < 4:
            raise ValueError("DiffusionGemma Reference Prep requires an IMAGE tensor with shape [batch, height, width, channels].")

        source_height = int(tensor.shape[1])
        source_width = int(tensor.shape[2])
        chosen_policy = str(aspect_policy or "preserve_source")
        chosen_ratio = str(target_aspect_ratio or "source")

        cropped = tensor
        crop_applied = False
        effective_ratio = chosen_ratio

        if chosen_ratio != "source" and chosen_policy == "center_crop_to_target":
            crop_width, crop_height = _center_crop_dimensions_for_aspect(source_width, source_height, chosen_ratio)
            if crop_width != source_width or crop_height != source_height:
                cropped = _center_crop_image_batch(tensor, crop_width, crop_height)
                crop_applied = True
            working_width = crop_width
            working_height = crop_height
            target_width, target_height = _aspect_dimensions_for_long_edge(chosen_ratio, long_edge, multiple)
        else:
            working_width = source_width
            working_height = source_height
            effective_ratio = chosen_ratio if chosen_ratio != "source" else _normalize_aspect_ratio_key(f"{source_width}:{source_height}")
            target_width, target_height = _dimensions_for_long_edge(source_width, source_height, long_edge, multiple)

        if not bool(upscale_smaller_images):
            mult = max(1, int(multiple))
            scale = min(1.0, max(64, int(long_edge)) / float(max(working_width, working_height)))
            target_width = max(mult, int(math.floor((working_width * scale) / mult)) * mult)
            target_height = max(mult, int(math.floor((working_height * scale) / mult)) * mult)

        prepared = (
            cropped
            if (target_width, target_height) == (int(working_width), int(working_height))
            else _resize_image_batch(cropped, target_width, target_height)
        )

        metadata = {
            "source_width": source_width,
            "source_height": source_height,
            "target_aspect_ratio": chosen_ratio,
            "aspect_policy": chosen_policy,
            "crop_applied": crop_applied,
            "working_width": int(working_width),
            "working_height": int(working_height),
            "output_width": int(target_width),
            "output_height": int(target_height),
            "long_edge": int(long_edge),
            "multiple": int(multiple),
            "upscale_smaller_images": bool(upscale_smaller_images),
            "resolution_selector_preset": _resolution_selector_preset(chosen_ratio)
            if chosen_ratio != "source"
            else _known_aspect_ratio_label_from_dimensions(target_width, target_height),
            "note": "Pre-resizing for DiffusionGemma consistency/efficiency. Preserve-source with upscaling disabled is safest; center-crop can standardize framing when you intentionally want one aspect ratio.",
        }
        return (
            prepared,
            _json_dumps(metadata),
            int(target_width),
            int(target_height),
            str(metadata["resolution_selector_preset"]),
        )


class DiffusionGemmaH3ReferencePairPrep:
    """Cap two H3 image references to one shared, downscale-only pixel budget."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image_1": (
                    "IMAGE",
                    {
                        "tooltip": "The image connected to MiniMax H3 ref_image_0 / <Picture 1>. Keep this socket order aligned with the Director manifest.",
                    },
                ),
                "reference_image_2": (
                    "IMAGE",
                    {
                        "tooltip": "The image connected to MiniMax H3 ref_image_1 / <Picture 2>. The two outputs remain separate and keep their own aspect ratios.",
                    },
                ),
                "generation_width": (
                    "INT",
                    {
                        "default": 1376,
                        "min": 256,
                        "max": 8192,
                        "step": 32,
                        "tooltip": "Connect the same width used by MiniMax H3. It defines the shared reference budget, not a crop or output-video resize.",
                    },
                ),
                "generation_height": (
                    "INT",
                    {
                        "default": 768,
                        "min": 256,
                        "max": 8192,
                        "step": 32,
                        "tooltip": "Connect the same height used by MiniMax H3.",
                    },
                ),
                "combined_reference_area_ratio": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.25,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Total pixel budget for both images divided by one generation frame. 1.0 shares one frame's pixel area across both references; 0.5 is faster but may reduce identity/style fidelity.",
                    },
                ),
                "reference_1_share": (
                    "FLOAT",
                    {
                        "default": 0.60,
                        "min": 0.10,
                        "max": 0.90,
                        "step": 0.05,
                        "tooltip": "Preferred share of the combined budget for Picture 1. Any allocation that Picture 1 cannot use is automatically reassigned to Picture 2, and vice versa.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "INT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = (
        "reference_image_1",
        "reference_image_2",
        "prep_metadata_json",
        "reference_1_width",
        "reference_1_height",
        "reference_2_width",
        "reference_2_height",
        "combined_reference_megapixels",
        "estimated_packed_reference_rows",
    )
    FUNCTION = "prepare"
    CATEGORY = OPTIONAL_CATEGORY
    DESCRIPTION = (
        "Downscales two MiniMax H3 image references under one shared pixel budget while preserving each aspect ratio. "
        "It never batches, crops, stretches, pads, or upscales the references, and it does not change the final video resolution, duration, or sampler steps."
    )

    @staticmethod
    def _single_image_tensor(image: Any, label: str) -> Any:
        import torch

        tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
        if tensor.ndim != 4:
            raise ValueError(f"{label} must be an IMAGE tensor shaped [1, height, width, channels].")
        if int(tensor.shape[0]) != 1:
            raise ValueError(
                f"{label} must contain exactly one image; got batch size {int(tensor.shape[0])}. "
                "Keep Picture 1 and Picture 2 on separate sockets because native MiniMax H3 uses only the first item from each reference socket."
            )
        if int(tensor.shape[3]) != 3:
            raise ValueError(f"{label} must be an RGB IMAGE with 3 channels; got {int(tensor.shape[3])}.")
        if int(tensor.shape[1]) < 32 or int(tensor.shape[2]) < 32:
            raise ValueError(f"{label} must be at least 32x32 pixels for MiniMax H3.")
        return tensor

    def prepare(
        self,
        reference_image_1: Any,
        reference_image_2: Any,
        generation_width: int = 1376,
        generation_height: int = 768,
        combined_reference_area_ratio: float = 1.0,
        reference_1_share: float = 0.60,
    ):
        image_1 = self._single_image_tensor(reference_image_1, "reference_image_1")
        image_2 = self._single_image_tensor(reference_image_2, "reference_image_2")

        generation_width = max(256, int(generation_width))
        generation_height = max(256, int(generation_height))
        area_ratio = max(0.25, min(2.0, float(combined_reference_area_ratio)))
        share_1 = max(0.10, min(0.90, float(reference_1_share)))
        total_pixel_budget = float(generation_width * generation_height) * area_ratio

        source_1_height, source_1_width = int(image_1.shape[1]), int(image_1.shape[2])
        source_2_height, source_2_width = int(image_2.shape[1]), int(image_2.shape[2])
        source_1_pixels = source_1_width * source_1_height
        source_2_pixels = source_2_width * source_2_height
        budget_1, budget_2 = _h3_shared_reference_pixel_budgets(
            source_1_pixels,
            source_2_pixels,
            total_pixel_budget,
            share_1,
        )

        target_1_width, target_1_height = _h3_reference_dimensions_for_pixel_budget(
            source_1_width,
            source_1_height,
            budget_1,
            32,
        )
        target_2_width, target_2_height = _h3_reference_dimensions_for_pixel_budget(
            source_2_width,
            source_2_height,
            budget_2,
            32,
        )

        prepared_1 = (
            image_1
            if (target_1_width, target_1_height) == (source_1_width, source_1_height)
            else _resize_image_batch(image_1, target_1_width, target_1_height)
        )
        prepared_2 = (
            image_2
            if (target_2_width, target_2_height) == (source_2_width, source_2_height)
            else _resize_image_batch(image_2, target_2_width, target_2_height)
        )

        target_1_pixels = target_1_width * target_1_height
        target_2_pixels = target_2_width * target_2_height
        combined_pixels = target_1_pixels + target_2_pixels
        packed_rows_1 = (target_1_width // 32) * (target_1_height // 32)
        packed_rows_2 = (target_2_width // 32) * (target_2_height // 32)
        combined_packed_rows = packed_rows_1 + packed_rows_2

        native_1_width, native_1_height = _h3_native_match_reference_dimensions(
            source_1_width,
            source_1_height,
            generation_width,
            generation_height,
            32,
        )
        native_2_width, native_2_height = _h3_native_match_reference_dimensions(
            source_2_width,
            source_2_height,
            generation_width,
            generation_height,
            32,
        )
        native_match_rows = (
            (native_1_width // 32) * (native_1_height // 32)
            + (native_2_width // 32) * (native_2_height // 32)
        )
        packed_row_reduction = (
            max(0.0, 100.0 * (1.0 - (combined_packed_rows / float(native_match_rows))))
            if native_match_rows
            else 0.0
        )

        def reference_metadata(
            ordinal: int,
            source_width: int,
            source_height: int,
            allocated_budget: float,
            output_width: int,
            output_height: int,
            packed_rows: int,
        ) -> dict[str, Any]:
            return {
                "picture_tag": f"<Picture {ordinal}>",
                "source_width": source_width,
                "source_height": source_height,
                "source_pixels": source_width * source_height,
                "allocated_pixel_budget": int(math.floor(allocated_budget)),
                "output_width": output_width,
                "output_height": output_height,
                "output_pixels": output_width * output_height,
                "output_megapixels": (output_width * output_height) / float(1024 * 1024),
                "scale": min(output_width / float(source_width), output_height / float(source_height)),
                "resized": (output_width, output_height) != (source_width, source_height),
                "estimated_packed_reference_rows": packed_rows,
            }

        metadata = {
            "schema": "dg-h3-reference-pair-prep/1",
            "generation_width": generation_width,
            "generation_height": generation_height,
            "combined_reference_area_ratio": area_ratio,
            "reference_1_share": share_1,
            "multiple": 32,
            "combined_pixel_budget": int(math.floor(total_pixel_budget)),
            "combined_output_pixels": combined_pixels,
            "combined_output_megapixels": combined_pixels / float(1024 * 1024),
            "estimated_packed_reference_rows": combined_packed_rows,
            "native_match_estimated_packed_reference_rows": native_match_rows,
            "packed_reference_row_reduction_percent": packed_row_reduction,
            "references": [
                reference_metadata(
                    1,
                    source_1_width,
                    source_1_height,
                    budget_1,
                    target_1_width,
                    target_1_height,
                    packed_rows_1,
                ),
                reference_metadata(
                    2,
                    source_2_width,
                    source_2_height,
                    budget_2,
                    target_2_width,
                    target_2_height,
                    packed_rows_2,
                ),
            ],
            "note": (
                "This caps only the two image-reference conditioning streams. It does not resize the generated video or reduce its frame count or sampler steps. "
                "Connect the two IMAGE outputs to separate native H3 reference sockets in the same order and leave ref_image_size set to match."
            ),
        }
        return (
            prepared_1,
            prepared_2,
            _json_dumps(metadata),
            target_1_width,
            target_1_height,
            target_2_width,
            target_2_height,
            combined_pixels / float(1024 * 1024),
            combined_packed_rows,
        )


def _target_profile_node_result(config: TargetProfileConfig) -> tuple[TargetProfileConfig, str]:
    if (
        config.target_profile == "minimax_h3"
        and config.audio_mode == "visual_only"
        and config.minimax_h3_dialogue_mode == "required"
    ):
        raise ValueError(
            "MiniMax H3 dialogue_mode=required conflicts with audio_mode=visual_only. Enable scene audio or set dialogue_mode to auto/off."
        )
    return (config, _json_dumps(_target_profile_config_to_dict(config)))


class DiffusionGemmaLTX25TargetProfile:
    """Model-specific LTX-2.5 target controls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generation_mode": (
                    ["Auto (recommended)", "Text to video", "Image to video", "First + last frame"],
                    {
                        "default": "Auto (recommended)",
                        "tooltip": "Auto selects text-to-video when Context Hub has no frame, image-to-video from its first-frame image socket, and first+last-frame only when both dedicated frame sockets are connected. Override only for unusual graph wiring.",
                    },
                ),
                "target_duration_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "step": 0.1,
                        "tooltip": "At 0, use media duration, then an explicit duration in the brief, then a 5-second planning default. This sets LTX-2.5 shot, action, sound, and speech budgets; camera-motion phase density is advisory, not a hard limit.",
                    },
                ),
                "style_guidance": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional LTX visual treatment. Keep this concrete and compatible with any conditioning frames.",
                    },
                ),
                "audio_mode": (
                    ["auto_scene_audio", "explicit_sound_design", "visual_only"],
                    {"default": "auto_scene_audio"},
                ),
                "audio_guidance": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional ambience, effects, music, speaker, delivery, or exact quoted wording for LTX's joint audio-video prompt.",
                    },
                ),
                "negative_prompt_mode": (
                    ["auto", "empty", "custom"],
                    {
                        "default": "auto",
                        "tooltip": "Auto lets Director author a useful LTX negative prompt, empty forces no negative prompt, and custom returns the supplied guidance.",
                    },
                ),
                "negative_prompt_guidance": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional custom LTX negative prompt text.",
                    },
                ),
                "long_horizon_mode": (
                    ["Off", "Auto (>20 seconds)", "On"],
                    {
                        "default": "Off",
                        "tooltip": "Experimental. Off preserves the existing compiler. Auto activates only above 20 seconds. On forces a silent establish → commit → sustain/reveal → settle/hold subject/environment continuity plan, a compact ≤200-word caption, and advisory diagnostics. Stable camera capability fills the long middle without sustained camera travel.",
                    },
                ),
            },
            "optional": {
                "camera_capability": (
                    ["Stable / base model", "Advanced / controlled camera"],
                    {
                        "default": "Stable / base model",
                        "tooltip": "Stable is the safe default for unassisted LTX I2V: locked/stabilized framing or one restrained push, pull, pan, tilt, or lateral track. Advanced preserves ambitious orbit, roll, sweeping-parallax, compound, or controlled-camera choreography and should be used only with suitable motion control or a camera LoRA.",
                    },
                ),
            },
        }

    RETURN_TYPES = (TARGET_PROFILE_TYPE, "STRING")
    RETURN_NAMES = ("target_profile_config", "target_profile_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "LTX-2.5-only target profile with generation mode, duration, style, audio, negative-prompt, and experimental long-horizon planning controls."

    def build(
        self,
        generation_mode: str,
        target_duration_seconds: float,
        style_guidance: str,
        audio_mode: str,
        audio_guidance: str,
        negative_prompt_mode: str,
        negative_prompt_guidance: str,
        long_horizon_mode: str = "Off",
        camera_capability: str = "Stable / base model",
    ):
        config = _make_target_profile_config(
            target_profile="ltx",
            audio_mode=audio_mode,
            audio_guidance=audio_guidance,
            target_duration_seconds=target_duration_seconds,
            ltx_style=style_guidance,
            negative_prompt_mode=negative_prompt_mode,
            negative_prompt_guidance=negative_prompt_guidance,
            ltx_generation_mode=generation_mode,
            ltx_long_horizon_mode=long_horizon_mode,
            ltx_camera_capability=camera_capability,
        )
        return _target_profile_node_result(config)


class DiffusionGemmaMiniMaxH3TargetProfile:
    """Model-specific MiniMax-H3 target controls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generation_mode": (
                    ["t2va", "ref2va"],
                    {
                        "default": "t2va",
                        "tooltip": "t2va writes the three-field text-to-video prompt. ref2va writes MiniMax's six-section full-reference prompt for MiniMaxH3ReferenceToVideo.",
                    },
                ),
                "target_duration_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "step": 0.1,
                        "tooltip": "At 0, use media duration, then an explicit duration in the brief, then a 5-second planning default. This sets H3 timeline and complexity budgets.",
                    },
                ),
                "audio_mode": (
                    ["auto_scene_audio", "explicit_sound_design", "visual_only"],
                    {"default": "auto_scene_audio"},
                ),
                "audio_guidance": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional ambience, effects, music, speaker, language, delivery, or exact quoted wording.",
                    },
                ),
                "shot_count": (
                    list(_MINIMAX_H3_SHOT_COUNT_CHOICES),
                    {
                        "default": "auto",
                        "tooltip": "Auto honors an exact shot count written in the brief. Select 1 to 12, or choose custom and type a count below.",
                    },
                ),
                "custom_shot_count": (
                    "INT",
                    {
                        "default": 12,
                        "min": 1,
                        "max": _MINIMAX_H3_MAX_CUSTOM_SHOT_COUNT,
                        "step": 1,
                        "tooltip": "Used only when shot_count is custom.",
                    },
                ),
                "dialogue_mode": (
                    list(_MINIMAX_H3_DIALOGUE_MODES),
                    {
                        "default": "auto",
                        "tooltip": "Auto preserves requested speech without inventing it. Required authorizes dialogue and enforces the exact line count. Off prohibits dialogue.",
                    },
                ),
                "dialogue_line_count": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": _MINIMAX_H3_MAX_DIALOGUE_LINE_COUNT,
                        "step": 1,
                        "tooltip": "Used only when dialogue_mode is required. One line is one complete <d>[Language] ...</d> utterance with a numbered cue such as (S1).",
                    },
                ),
                "dialogue_guidance": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional speaker, language, delivery, story purpose, or exact quoted wording.",
                    },
                ),
                "negative_prompt_mode": (
                    ["auto", "empty", "custom"],
                    {
                        "default": "auto",
                        "tooltip": "H3 has no separate negative-conditioning channel. Auto/custom fold relevant exclusions into the integrated H3 description; empty ignores this guidance while preserving exclusions written in the creative brief.",
                    },
                ),
                "negative_prompt_guidance": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Unwanted elements or behaviors to express as concise natural-language exclusions inside the H3 prompt, not as a separate negative prompt.",
                    },
                ),
            },
            "optional": {
                "shot_count_override": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": _MINIMAX_H3_MAX_CUSTOM_SHOT_COUNT,
                        "step": 1,
                        "tooltip": "0 keeps the visible native [Shot N] selector. A positive value overrides only Director's native storyboard/cut count; never connect a Project Master generation-lane count here.",
                    },
                ),
            },
        }

    RETURN_TYPES = (TARGET_PROFILE_TYPE, "STRING")
    RETURN_NAMES = ("target_profile_config", "target_profile_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "MiniMax-H3-only target profile with T2VA/Ref2VA, duration, audio, shot, dialogue, and negative-prompt controls."

    def build(
        self,
        generation_mode: str,
        target_duration_seconds: float,
        audio_mode: str,
        audio_guidance: str,
        shot_count: str,
        custom_shot_count: int,
        dialogue_mode: str,
        dialogue_line_count: int,
        dialogue_guidance: str,
        negative_prompt_mode: str,
        negative_prompt_guidance: str,
        shot_count_override: int = 0,
    ):
        resolved_shot_count = _resolve_minimax_h3_shot_count(shot_count, custom_shot_count)
        try:
            override = int(shot_count_override)
        except (TypeError, ValueError, OverflowError):
            override = 0
        if 1 <= override <= _MINIMAX_H3_MAX_CUSTOM_SHOT_COUNT:
            resolved_shot_count = str(override)
        config = _make_target_profile_config(
            target_profile="minimax_h3",
            audio_mode=audio_mode,
            audio_guidance=audio_guidance,
            target_duration_seconds=target_duration_seconds,
            negative_prompt_mode=negative_prompt_mode,
            negative_prompt_guidance=negative_prompt_guidance,
            minimax_h3_mode=generation_mode,
            minimax_h3_shot_count=resolved_shot_count,
            minimax_h3_dialogue_mode=dialogue_mode,
            minimax_h3_dialogue_line_count=dialogue_line_count,
            minimax_h3_dialogue_guidance=dialogue_guidance,
        )
        return _target_profile_node_result(config)


class DiffusionGemmaIdeogram4TargetProfile:
    """Model-specific Ideogram 4 target controls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (
                    ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"],
                    {"default": "1:1"},
                ),
                "render_style": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional visual medium, finish, or art direction for the generated image.",
                    },
                ),
                "exact_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Exact visible wording Ideogram should render. Leave blank when no text is requested.",
                    },
                ),
                "json_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Choose the ideogram_prompt representation: on returns caption-schema JSON; off returns ready-to-use prose. This does not add another output socket.",
                    },
                ),
                "negative_prompt_mode": (["auto", "empty", "custom"], {"default": "auto"}),
                "negative_prompt_guidance": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = (TARGET_PROFILE_TYPE, "STRING")
    RETURN_NAMES = ("target_profile_config", "target_profile_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "Ideogram 4-only target profile with image composition, exact-text, structured-output, and negative-prompt controls; no video or audio settings."

    def build(
        self,
        aspect_ratio: str,
        render_style: str,
        exact_text: str,
        json_output: bool,
        negative_prompt_mode: str,
        negative_prompt_guidance: str,
    ):
        config = _make_target_profile_config(
            target_profile="ideogram4",
            audio_mode="visual_only",
            ideogram_aspect_ratio=aspect_ratio,
            ideogram_render_style=render_style,
            ideogram_exact_text=exact_text,
            ideogram_json_output=json_output,
            negative_prompt_mode=negative_prompt_mode,
            negative_prompt_guidance=negative_prompt_guidance,
        )
        return _target_profile_node_result(config)


class DiffusionGemmaTargetProfile:
    """Compatibility surface for workflows saved before model-specific targets."""

    DEPRECATED = True
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_profile": (["ltx", "ideogram4", "minimax_h3"], {"default": "ltx"}),
                "audio_mode": (
                    ["auto_scene_audio", "explicit_sound_design", "visual_only"],
                    {"default": "auto_scene_audio"},
                ),
                "audio_guidance": ("STRING", {"default": "", "multiline": True}),
                "target_duration_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "step": 0.1,
                        "tooltip": "User-selected target duration. At 0, LTX-2.5 and H3 use media duration, then an explicit duration in the brief, then a 5-second planning default. Director uses this value to cap shots, action beats, sound layers, and spoken words; LTX camera-motion phase density is advisory rather than a hard cap.",
                    },
                ),
                "ltx_style": ("STRING", {"default": "", "multiline": False}),
                "ideogram_aspect_ratio": (
                    ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"],
                    {"default": "1:1"},
                ),
                "ideogram_render_style": ("STRING", {"default": "", "multiline": False}),
                "ideogram_exact_text": ("STRING", {"default": "", "multiline": True}),
                "ideogram_json_output": ("BOOLEAN", {"default": True}),
                "negative_prompt_mode": (["auto", "empty", "custom"], {"default": "auto"}),
                "negative_prompt_guidance": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "minimax_h3_mode": (
                    ["t2va", "ref2va"],
                    {
                        "default": "t2va",
                        "tooltip": "t2va writes the three-field text-to-video prompt. ref2va writes MiniMax's six-section full-reference prompt for MiniMaxH3ReferenceToVideo.",
                    },
                ),
                "minimax_h3_shot_count": (
                    list(_MINIMAX_H3_SHOT_COUNT_CHOICES),
                    {
                        "default": "auto",
                        "tooltip": "Auto honors an exact shot count written in the brief. Select 1 to 12, or choose custom and type a count below. An explicit value requires exactly that many consecutive H3 shots and overrides conflicting wording in the brief.",
                    },
                ),
                "minimax_h3_custom_shot_count": (
                    "INT",
                    {
                        "default": 12,
                        "min": 1,
                        "max": _MINIMAX_H3_MAX_CUSTOM_SHOT_COUNT,
                        "step": 1,
                        "tooltip": "Used only when shot_count is custom. Type any count from 1 to 99; counts above 12 usually need more duration and Director output-token budget.",
                    },
                ),
                "minimax_h3_dialogue_mode": (
                    list(_MINIMAX_H3_DIALOGUE_MODES),
                    {
                        "default": "auto",
                        "tooltip": "Auto preserves speech requested in the brief but does not invent it. Required authorizes Director to create dialogue and enforces the exact line count below. Off prohibits dialogue while leaving other enabled audio available.",
                    },
                ),
                "minimax_h3_dialogue_line_count": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": _MINIMAX_H3_MAX_DIALOGUE_LINE_COUNT,
                        "step": 1,
                        "tooltip": "Used only when dialogue_mode is required. One line is one complete <d>[Language] ...</d> utterance with a numbered cue such as (S1).",
                    },
                ),
                "minimax_h3_dialogue_guidance": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional speaker, language, delivery, story-purpose, or exact quoted wording. In required mode, blank guidance lets Director author concise lines.",
                    },
                ),
                "ltx_generation_mode": (
                    ["Auto (recommended)", "Text to video", "Image to video", "First + last frame"],
                    {
                        "default": "Auto (recommended)",
                        "tooltip": "Auto selects text-to-video when Context Hub has no frame, image-to-video from its first-frame image socket, and first+last-frame only when both dedicated frame sockets are connected. Override only for unusual graph wiring.",
                    },
                ),
                "ltx_long_horizon_mode": (
                    ["Off", "Auto (>20 seconds)", "On"],
                    {
                        "default": "Off",
                        "tooltip": "Compatibility control for LTX only. Auto activates the experimental long-horizon compiler above 20 seconds; Off preserves prior behavior.",
                    },
                ),
                "ltx_camera_capability": (
                    ["Stable / base model", "Advanced / controlled camera"],
                    {
                        "default": "Stable / base model",
                        "tooltip": "Compatibility control for LTX only. Stable avoids camera paths that normally require dedicated motion control or a camera LoRA; Advanced preserves the prior ambitious camera policy.",
                    },
                ),
            },
        }

    RETURN_TYPES = (TARGET_PROFILE_TYPE, "STRING")
    RETURN_NAMES = ("target_profile_config", "target_profile_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(
        self,
        target_profile: str,
        audio_mode: str,
        audio_guidance: str,
        target_duration_seconds: float,
        ltx_style: str,
        ideogram_aspect_ratio: str,
        ideogram_render_style: str,
        ideogram_exact_text: str,
        ideogram_json_output: bool,
        negative_prompt_mode: str,
        negative_prompt_guidance: str,
        minimax_h3_mode: str = "t2va",
        minimax_h3_shot_count: str = "auto",
        minimax_h3_custom_shot_count: int = 12,
        minimax_h3_dialogue_mode: str = "auto",
        minimax_h3_dialogue_line_count: int = 2,
        minimax_h3_dialogue_guidance: str = "",
        ltx_generation_mode: str = "Auto (recommended)",
        ltx_long_horizon_mode: str = "Off",
        ltx_camera_capability: str = "Stable / base model",
    ):
        config = _make_target_profile_config(
            target_profile,
            audio_mode,
            audio_guidance,
            target_duration_seconds,
            ltx_style,
            ideogram_aspect_ratio,
            ideogram_render_style,
            ideogram_exact_text,
            ideogram_json_output,
            negative_prompt_mode,
            negative_prompt_guidance,
            minimax_h3_mode,
            _resolve_minimax_h3_shot_count(
                minimax_h3_shot_count,
                minimax_h3_custom_shot_count,
            ),
            minimax_h3_dialogue_mode,
            minimax_h3_dialogue_line_count,
            minimax_h3_dialogue_guidance,
            ltx_generation_mode=ltx_generation_mode,
            ltx_long_horizon_mode=ltx_long_horizon_mode,
            ltx_camera_capability=ltx_camera_capability,
        )
        return _target_profile_node_result(config)


class DiffusionGemmaGroundingGuardSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["off", "audit", "strict"], {"default": "audit"}),
                "retry_on_uncertain": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Strict mode may make one additional factual evidence attempt. Audit mode may make one fresh combined retry when an LTX or H3 packet is malformed, salvaged, or contains an invalid grounding ledger.",
                    },
                ),
                "sampling_profile": (
                    ["checkpoint_defaults", "full_48_diagnostic"],
                    {
                        "default": "checkpoint_defaults",
                        "tooltip": "Native 48-step 0.8-to-0.4 DiffusionGemma schedule; diagnostic disables adaptive stopping.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Scoped unsigned 64-bit seed; global CPU/CUDA RNG state is restored after each call. In audit/strict mode this also selects the Director variant, so refresh with a fixed seed intentionally reproduces the same result. Use randomize/increment when you want a new prompt variant.",
                    },
                ),
                "save_detailed_trace": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Opt in to JSON-only evidence traces beneath ComfyUI/output; media and logits are never copied.",
                    },
                ),
                "trace_subfolder": (
                    "STRING",
                    {
                        "default": "diffusiongemma_grounding",
                        "multiline": False,
                        "tooltip": "Relative subfolder beneath ComfyUI/output.",
                    },
                ),
            },
            "optional": {
                "external_evidence_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Optional dg-external-evidence/1 typed claims from local OCR, detectors, pose/tracking/motion, SigLIP, or manual review.",
                    },
                ),
                "evidence_token_budget": (
                    list(EVIDENCE_TOKEN_BUDGETS),
                    {
                        "default": "auto",
                        "advanced": True,
                        "tooltip": "Strict evidence-only output budget. Auto uses 768 tokens initially and 1024 on retry; an explicit value applies independently to every evidence attempt. Higher values take longer but do not weaken validation. Ignored in off/audit modes.",
                    },
                ),
            },
        }

    RETURN_TYPES = (GROUNDING_GUARD_TYPE, "STRING")
    RETURN_NAMES = ("grounding_guard_config", "config_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, save_detailed_trace: bool = False, **_kwargs: Any):
        # A requested trace is an output side effect and must be produced for
        # every queue. Ordinary guard configurations remain normally cacheable.
        # Comfy passes None here when the widget has been converted to a linked
        # input, because IS_CHANGED receives literal values only. Conservatively
        # rerun so a linked true value cannot lose trace side effects.
        if save_detailed_trace is None or bool(save_detailed_trace):
            return float("nan")
        return "grounding-guard-config-v1"

    def build(
        self,
        mode: str = "audit",
        retry_on_uncertain: bool = True,
        sampling_profile: str = "checkpoint_defaults",
        seed: int = 0,
        save_detailed_trace: bool = False,
        trace_subfolder: str = "diffusiongemma_grounding",
        external_evidence_json: str = "",
        evidence_token_budget: str = "auto",
    ):
        config = normalize_grounding_guard_config(
            {
                "mode": mode,
                "retry_on_uncertain": retry_on_uncertain,
                "sampling_profile": sampling_profile,
                "seed": seed,
                "save_detailed_trace": save_detailed_trace,
                "trace_subfolder": trace_subfolder,
                "external_evidence_json": external_evidence_json,
                "evidence_token_budget": evidence_token_budget,
            }
        )
        return (config, _json_dumps(config.to_dict()))


_DIRECTOR_IMPLEMENTATION_FILES = (
    "nodes.py",
    "ltx25_contract.py",
    "director_cache.py",
    "grounding_guard.py",
    "grounding_telemetry.py",
    "schemas/prompt_packet.schema.json",
    "schemas/grounding_evidence.schema.json",
    "schemas/external_evidence.schema.json",
)


def _normalize_director_cache_mode(value: Any) -> str:
    mode = str(value or "reuse").strip().lower()
    return mode if mode in DIRECTOR_CACHE_MODE_CHOICES else "reuse"


def _director_cache_key(
    model_config: RuntimeConfig,
    context: GemmaContext,
    target_config: TargetProfileConfig | dict[str, Any],
    guard_config: GroundingGuardConfig | dict[str, Any] | str | None,
    *,
    temperature: float,
    creativity_mode: str,
    creative_strength: float,
    thinking_mode: str,
    max_new_tokens: int,
) -> str:
    root = Path(__file__).resolve().parent
    guard = normalize_grounding_guard_config(guard_config)
    dependency_versions: dict[str, str] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version

        for package_name in (
            "torch",
            "transformers",
            "accelerate",
            "comfy-kitchen",
            "safetensors",
        ):
            try:
                dependency_versions[package_name] = version(package_name)
            except PackageNotFoundError:
                dependency_versions[package_name] = "not-installed"
    except Exception:
        dependency_versions = {
            "transformers": str(model_config.status.get("transformers_version", "")),
        }
    runtime_identity: dict[str, Any] = {
        "backend": model_config.backend,
        "dtype": model_config.dtype,
        "quantization": model_config.quantization,
        "local_files_only": bool(model_config.local_files_only),
        "max_memory_gb": float(model_config.max_memory_gb),
        "temperature": float(model_config.temperature),
        "fallback_backend": model_config.fallback_backend,
        "extra_args": model_config.extra_args,
        "checkpoint": _checkpoint_identity(model_config.model_path, model_config.status),
    }
    if model_config.cli_path:
        runtime_identity["cli"] = _checkpoint_identity(model_config.cli_path, {})
    identity = {
        "schema": DIRECTOR_CACHE_KEY_SCHEMA,
        "compiler_protocol": "native_minimax_h3_prompt_with_provenance/1",
        "implementation": _implementation_identity(root, _DIRECTOR_IMPLEMENTATION_FILES),
        "dependency_versions": dependency_versions,
        "system_prompt_sha256": _canonical_sha256(DEFAULT_MASTER_PROMPT),
        "runtime": runtime_identity,
        "context": {
            "user_prompt": str(context.user_prompt or ""),
            "source": str(context.source or "none"),
            "visual_description": str(context.visual_description or ""),
            "warnings": list(context.warnings),
            "media_metadata": copy.deepcopy(context.media_metadata),
            "ordered_processor_tensor_hashes": _ordered_tensor_content_hashes(context.images),
        },
        "target_profile": _target_profile_config_to_dict(target_config),
        "grounding_guard": guard.to_dict(),
        "generator": {
            "temperature": float(temperature),
            "creativity_mode": _normalize_creativity_mode(creativity_mode),
            "creative_strength": _normalize_creative_strength(creative_strength),
            "thinking_mode": _normalize_thinking_mode(thinking_mode),
            "max_new_tokens": int(max_new_tokens),
            "max_output_chars": 8000,
        },
        "result_affecting_environment": {
            "DG_VIDEO_TRANSPORT": os.environ.get("DG_VIDEO_TRANSPORT", ""),
            "DG_MAX_DENOISING_STEPS": os.environ.get("DG_MAX_DENOISING_STEPS", ""),
        },
    }
    return _canonical_sha256(identity)


def _validated_cached_director_outputs(
    entry: dict[str, Any],
    guard: GroundingGuardConfig,
) -> tuple[dict[str, Any] | None, str]:
    outputs = entry.get("outputs")
    if not isinstance(outputs, dict):
        return None, "outputs_missing"
    required_string_fields = (
        "final_json",
        "reasoning_text",
        "raw_response",
        "metadata_json",
        "grounding_status",
        "grounding_report_json",
        "minimax_h3_prompt",
    )
    if any(not isinstance(outputs.get(name), str) for name in required_string_fields):
        return None, "output_type_invalid"
    if not isinstance(outputs.get("ready_for_generation"), bool):
        return None, "readiness_type_invalid"
    try:
        packet = json.loads(outputs["final_json"])
        metadata = json.loads(outputs["metadata_json"])
        report = json.loads(outputs["grounding_report_json"])
    except Exception as exc:
        return None, f"output_json_invalid:{type(exc).__name__}"
    if not isinstance(packet, dict) or not isinstance(metadata, dict) or not isinstance(report, dict):
        return None, "output_json_root_invalid"
    packet_metadata = packet.get("metadata")
    if not isinstance(packet_metadata, dict) or packet_metadata != metadata:
        return None, "metadata_mismatch"
    ready = bool(metadata.get("ready_for_generation"))
    if ready is not outputs["ready_for_generation"]:
        return None, "readiness_mismatch"
    if str(packet.get("minimax_h3_prompt", "") or "") != outputs["minimax_h3_prompt"]:
        return None, "minimax_h3_prompt_mismatch"
    packet_report = metadata.get("grounding_guard")
    if not isinstance(packet_report, dict) or packet_report != report:
        return None, "grounding_report_mismatch"
    if str(report.get("analysis_status", "not_run")) != outputs["grounding_status"]:
        return None, "grounding_status_mismatch"
    decision = str(report.get("decision", ""))
    if bool(report.get("grounding_guard_would_block")) or decision == "block":
        return None, "cached_guard_blocked"
    if guard.mode == "strict" and decision not in {"pass", "not_applicable"}:
        return None, "strict_guard_decision_invalid"
    if (
        guard.mode == "strict"
        and str(metadata.get("target_profile", "")) == "minimax_h3"
        and (
            metadata.get("compiler_output_contract")
            != "native_minimax_h3_prompt_with_provenance/1"
            or metadata.get("compiler_output_contract_parse_valid") is not True
        )
    ):
        return None, "strict_h3_compiler_contract_invalid"
    if not ready:
        return None, "cached_packet_not_ready"
    return {
        **outputs,
        "packet": packet,
        "metadata": metadata,
        "grounding_report": report,
    }, "hit"


def _director_cache_eligibility(
    packet: dict[str, Any],
    metadata: dict[str, Any],
    report: dict[str, Any],
    target_config: TargetProfileConfig | dict[str, Any],
    guard: GroundingGuardConfig,
) -> tuple[bool, str]:
    if not bool(metadata.get("ready_for_generation")):
        return False, "packet_not_ready"
    if bool(metadata.get("used_template_fallback")):
        return False, "template_fallback"
    if not bool(metadata.get("json_parse_valid")):
        return False, "model_packet_not_parseable"
    if metadata.get("salvage_warning") or metadata.get("plain_text_salvage"):
        return False, "salvaged_packet"
    decision = str(report.get("decision", ""))
    if bool(report.get("grounding_guard_would_block")) or decision == "block":
        return False, "grounding_guard_blocked"
    if decision not in {"pass", "not_applicable", "disabled"}:
        return False, "grounding_not_verified_for_cache"
    target = _target_profile_config_to_dict(target_config)
    if target["target_profile"] == "minimax_h3" and not str(packet.get("minimax_h3_prompt", "") or "").strip():
        return False, "minimax_h3_prompt_empty"
    if (
        guard.mode == "strict"
        and target["target_profile"] == "minimax_h3"
        and (
            metadata.get("compiler_output_contract")
            != "native_minimax_h3_prompt_with_provenance/1"
            or metadata.get("compiler_output_contract_parse_valid") is not True
        )
    ):
        return False, "strict_h3_compiler_contract_invalid"
    if guard.mode == "strict" and decision not in {"pass", "not_applicable"}:
        return False, "strict_grounding_not_verified"
    return True, "eligible"


def _director_runtime_diagnostics(
    metrics: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    retry_reasons = report.get("retry_reasons") if isinstance(report.get("retry_reasons"), list) else []
    attempt_count = int(max(0.0, _safe_float(report.get("attempt_count"), 0.0)))
    model_call_count = int(max(0.0, _safe_float(report.get("model_call_count"), len(metrics.get("calls", [])))))
    calls = metrics.get("calls") if isinstance(metrics.get("calls"), list) else []
    metrics["grounding_attempt_count"] = attempt_count
    metrics["model_call_count"] = max(model_call_count, len(calls))
    metrics["target_repair_attempt_count"] = sum(
        1
        for call in calls
        if str(call.get("stage", "")).startswith("target_repair_")
        or str(call.get("stage", "")) == "compiler_retry"
    )
    metrics["dialogue_patch_attempt_count"] = sum(
        1
        for call in calls
        if str(call.get("stage", "")).startswith("dialogue_patch_")
    )
    metrics["retry_reason_count"] = len(retry_reasons)
    metrics["grounding_first_pass_accepted"] = bool(
        str(report.get("decision", "")) == "pass" and attempt_count == 1
    )
    return _rounded_director_runtime(metrics)


def _director_cache_payload(
    cache_key: str,
    packet: dict[str, Any],
    reasoning_text: str,
    raw_response: str,
    metadata: dict[str, Any],
    grounding_status: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": DIRECTOR_CACHE_SCHEMA,
        "cache_key": cache_key,
        "created_at_epoch_seconds": int(time.time()),
        "outputs": {
            "final_json": _json_dumps(packet),
            "reasoning_text": str(reasoning_text or ""),
            "raw_response": str(raw_response or ""),
            "metadata_json": _json_dumps(metadata),
            "grounding_status": str(grounding_status or "not_run"),
            "grounding_report_json": _json_dumps(report),
            "minimax_h3_prompt": str(packet.get("minimax_h3_prompt", "") or ""),
            "ready_for_generation": bool(metadata.get("ready_for_generation")),
        },
    }


def _normalize_measured_audio_report_json(value: Any) -> dict[str, Any] | None:
    """Validate one bounded machine report for LTX audio-aware direction."""

    if value is None:
        return None
    if isinstance(value, dict):
        parsed: Any = copy.deepcopy(value)
    else:
        text = str(value or "").strip()
        if not text:
            return None
        if len(text) > 100_000:
            raise ValueError("Measured audio report JSON is too large.")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Measured audio report must be one valid JSON object.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Measured audio report must be one valid JSON object.")

    def bounded(item: Any, depth: int = 0) -> Any:
        if depth > 5:
            return "[depth limited]"
        if item is None or isinstance(item, (bool, int)):
            return item
        if isinstance(item, float):
            return round(item, 6) if math.isfinite(item) else None
        if isinstance(item, str):
            return item[:800]
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for key, child in list(item.items())[:64]:
                result[str(key)[:96]] = bounded(child, depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            return [bounded(child, depth + 1) for child in list(item)[:32]]
        return str(item)[:800]

    normalized = bounded(parsed)
    assert isinstance(normalized, dict)
    if len(_json_dumps(normalized)) > 20_000:
        raise ValueError(
            "Measured audio report is too detailed for Director. Connect the compact QC report output."
        )
    return normalized


def _context_with_measured_audio_report(
    context: GemmaContext,
    measured_audio_report_json: Any,
) -> GemmaContext:
    if not isinstance(context, GemmaContext):
        raise TypeError("gemma_context must come from DiffusionGemma Context Hub.")
    report = _normalize_measured_audio_report_json(measured_audio_report_json)
    if report is None:
        return context
    metadata = copy.deepcopy(context.media_metadata)
    metadata["ltx_measured_audio_report"] = report
    return GemmaContext(
        user_prompt=context.user_prompt,
        images=context.images,
        source=context.source,
        media_metadata=metadata,
        visual_description=context.visual_description,
        warnings=list(context.warnings),
    )


class DiffusionGemmaCoTGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_config": (MODEL_TYPE,),
                "gemma_context": (CONTEXT_TYPE,),
                "target_profile_config": (TARGET_PROFILE_TYPE,),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "GGUF compatibility temperature. Rewrites GGUF --temp; guarded in-process DiffusionGemma uses its native 0.8-to-0.4 sampling profile instead.",
                    },
                ),
                "creativity_mode": (
                    ["faithful", "editorial", "cinematic", "concept_art", "wild"],
                    {
                        "default": "editorial",
                        "tooltip": "All five modes apply to MiniMax H3 and LTX I2V/FLF. An explicit camera path or static/locked request always wins. For H3, non-faithful modes may author expressive, physically coherent per-shot choreography. For conditioned LTX, a frame anchor fixes only its anchored instant and the LTX Camera Capability still governs future-path ambition. This does not change sampling randomness, temperature, or seed.",
                    },
                ),
                "creative_strength": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.5,
                        "step": 0.05,
                        "tooltip": "Intensity inside the selected creativity mode: 0-0.05 faithful, up to 0.35 subtle, up to 0.8 moderate, up to 1.15 strong, then maximal. MiniMax H3 scales shot-appropriate camera ambition without inheriting LTX's Camera Capability. For LTX, camera ambition scales only with Advanced / controlled camera; Stable / base model overrides it. This does not randomize output or authorize extra cuts, actors, or unrelated actions.",
                    },
                ),
                "thinking_mode": (["auto", "on", "off"], {"default": "auto"}),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 128,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Output budget. Detailed MiniMax H3 storyboards and Ideogram JSON benefit from 1024 or more tokens.",
                    },
                ),
                "director_cache_mode": (
                    DIRECTOR_CACHE_MODE_CHOICES,
                    {
                        "default": "reuse",
                        "tooltip": "reuse keeps an unchanged graph branch cached and can serve a verified disk result after restart; refresh recomputes and replaces an eligible disk result; off recomputes every queue without disk caching. Refresh does not randomize: identical inputs plus a fixed Director seed are intentionally reproducible. Select refresh for one retry only, queue once, then immediately return this control to reuse; leaving refresh selected reruns Director on every queue.",
                    },
                ),
            },
            "optional": {
                "grounding_guard_mode": (
                    ["inherit", "off", "audit", "strict"],
                    {
                        "default": "inherit",
                        "tooltip": "Direct per-Director override. Inherit uses a connected Grounding Guard Settings node, or audit when that socket is disconnected. Choose off for fast iteration; audit/strict add visual evidence validation and may add model calls.",
                    },
                ),
                "grounding_guard_config": (GROUNDING_GUARD_TYPE,),
                "measured_audio_report_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Connect the compact report from Music Audition + QC after ACE-Step renders. Director will choreograph the LTX shot around the selected waveform's measured pulse, transient density, tonal stability, and excerpt timing instead of guessing from requested metadata.",
                    },
                ),
                "h3_reference_policy": (
                    H3_REFERENCE_POLICY_TYPE,
                    {
                        "tooltip": "Optional H3 Reference Policy dropdown. A connected named policy overrides stale Context Hub manifest text. Leave disconnected for automatic role assignment from attached Ref2VA media.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "final_json",
        "reasoning_text",
        "raw_response",
        "metadata_json",
        "grounding_status",
        "grounding_report_json",
    )
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, director_cache_mode: str = "reuse", **_kwargs: Any):
        # In reuse mode, let Comfy preserve an unchanged branch in memory. This
        # is essential for multi-model graphs: editing an LTX-only control must
        # not invalidate an independent H3 Director and sampler chain. The
        # contract revision invalidates that native entry after implementation
        # changes, while input signatures still cover all graph dependencies.
        # Refresh and off are explicit every-queue execution modes.
        if director_cache_mode is None:
            return float("nan")
        if _normalize_director_cache_mode(director_cache_mode) == "reuse":
            return f"director-reuse:{DIRECTOR_EXECUTION_CONTRACT_REVISION}"
        return float("nan")

    def generate(
        self,
        model_config: RuntimeConfig,
        gemma_context: GemmaContext,
        target_profile_config: TargetProfileConfig,
        temperature: float,
        creativity_mode: str,
        creative_strength: float,
        thinking_mode: str,
        max_new_tokens: int = 1024,
        director_cache_mode: str = "reuse",
        grounding_guard_config: GroundingGuardConfig | dict[str, Any] | str | None = None,
        grounding_guard_mode: str = "inherit",
        measured_audio_report_json: str = "",
        unique_id: str | None = None,
        h3_reference_policy: H3ReferencePolicyConfig | dict[str, Any] | str | None = None,
    ):
        director_start = time.perf_counter()
        cache_mode = _normalize_director_cache_mode(director_cache_mode)
        runtime_config = _runtime_config_with_temperature(model_config, temperature)
        guard = normalize_grounding_guard_config(grounding_guard_config)
        guard_mode_override = str(grounding_guard_mode or "inherit").strip().lower()
        if guard_mode_override in {"off", "audit", "strict"}:
            guard = normalize_grounding_guard_config(
                guard,
                mode=guard_mode_override,
            )
        director_context = _context_with_measured_audio_report(
            gemma_context,
            measured_audio_report_json,
        )
        director_context = _context_with_minimax_h3_reference_policy(
            director_context,
            target_profile_config,
            h3_reference_policy,
        )
        metrics = _new_director_runtime_metrics(cache_mode)
        metric_token = _CURRENT_DIRECTOR_RUNTIME.set(metrics)
        cache_key = ""
        cache_info: dict[str, Any] = {
            "schema": DIRECTOR_CACHE_SCHEMA,
            "mode": cache_mode,
            "hit": False,
            "status": "disabled" if cache_mode == "off" else "pending",
        }
        try:
            cached_outputs: dict[str, Any] | None = None
            cache_allowed = cache_mode != "off" and not guard.save_detailed_trace
            if guard.save_detailed_trace:
                cache_info["status"] = "bypassed_for_detailed_trace"
            if cache_allowed:
                lookup_start = time.perf_counter()
                try:
                    cache_key = _director_cache_key(
                        runtime_config,
                        director_context,
                        target_profile_config,
                        guard,
                        temperature=temperature,
                        creativity_mode=creativity_mode,
                        creative_strength=creative_strength,
                        thinking_mode=thinking_mode,
                        max_new_tokens=int(max_new_tokens),
                    )
                    cache_info["key"] = cache_key
                    if cache_mode == "reuse":
                        entry, load_status = _load_director_cache_entry(cache_key)
                        cache_info["status"] = load_status
                        if entry is not None:
                            cached_outputs, validation_status = _validated_cached_director_outputs(
                                entry,
                                guard,
                            )
                            cache_info["status"] = validation_status
                    else:
                        cache_info["status"] = "refresh_requested"
                except Exception as exc:
                    cache_info["status"] = f"cache_key_or_read_error:{type(exc).__name__}"
                    cache_info["error"] = str(exc)[:500]
                    cached_outputs = None
                finally:
                    metrics["cache_lookup_seconds"] = time.perf_counter() - lookup_start

            if cached_outputs is not None:
                metrics["cache_hit"] = True
                cache_info["hit"] = True
                cache_info["status"] = "hit"
                # Preserve unload-before-H3 even if another node left a
                # compatible runtime resident before this disk-cache hit.
                _maybe_unload(runtime_config)
                packet = cached_outputs["packet"]
                metadata = cached_outputs["metadata"]
                report = cached_outputs["grounding_report"]
                metrics["total_director_seconds"] = time.perf_counter() - director_start
                runtime_diagnostics = _director_runtime_diagnostics(metrics, report)
                cache_info["lookup_seconds"] = runtime_diagnostics["cache_lookup_seconds"]
                metadata["director_runtime"] = runtime_diagnostics
                metadata["director_cache"] = copy.deepcopy(cache_info)
                packet["metadata"] = metadata
                return (
                    _json_dumps(packet),
                    cached_outputs["reasoning_text"],
                    cached_outputs["raw_response"],
                    _json_dumps(metadata),
                    cached_outputs["grounding_status"],
                    _json_dumps(report),
                )

            peak_tracking = _reset_director_peak_vram()
            packet, raw_output, reasoning_text, _fallback_reason, _parse_valid, _salvage_warning, metadata = _run_generation_packet(
                runtime_config,
                director_context,
                target_profile_config,
                DEFAULT_MASTER_PROMPT,
                True,
                int(max_new_tokens),
                8000,
                creativity_mode,
                creative_strength,
                thinking_mode,
                node_id=str(unique_id) if unique_id is not None else None,
                grounding_guard_config=guard,
            )
            if peak_tracking:
                metrics["peak_vram_mb"] = _director_peak_vram_mb()
            report = metadata.get("grounding_guard") if isinstance(metadata.get("grounding_guard"), dict) else {}
            grounding_status = str(report.get("analysis_status", "not_run"))
            if cache_allowed and cache_key:
                eligible, eligibility_reason = _director_cache_eligibility(
                    packet,
                    metadata,
                    report,
                    target_profile_config,
                    guard,
                )
                cache_info["eligible"] = bool(eligible)
                cache_info["eligibility_reason"] = eligibility_reason
            else:
                eligible = False
                cache_info["eligible"] = False
                cache_info.setdefault("eligibility_reason", cache_info["status"])

            metrics["total_director_seconds"] = time.perf_counter() - director_start
            runtime_diagnostics = _director_runtime_diagnostics(metrics, report)
            cache_info["lookup_seconds"] = runtime_diagnostics["cache_lookup_seconds"]
            metadata["director_runtime"] = runtime_diagnostics
            metadata["director_cache"] = copy.deepcopy(cache_info)
            packet["metadata"] = metadata

            if eligible:
                write_start = time.perf_counter()
                try:
                    payload = _director_cache_payload(
                        cache_key,
                        packet,
                        reasoning_text,
                        raw_output,
                        metadata,
                        grounding_status,
                        report,
                    )
                    _save_director_cache_entry(cache_key, payload)
                    cache_info["stored"] = True
                    cache_info["status"] = "stored"
                except Exception as exc:
                    cache_info["stored"] = False
                    cache_info["status"] = f"write_error:{type(exc).__name__}"
                    cache_info["error"] = str(exc)[:500]
                finally:
                    metrics["cache_write_seconds"] = time.perf_counter() - write_start

            metrics["total_director_seconds"] = time.perf_counter() - director_start
            runtime_diagnostics = _director_runtime_diagnostics(metrics, report)
            cache_info["lookup_seconds"] = runtime_diagnostics["cache_lookup_seconds"]
            cache_info["write_seconds"] = runtime_diagnostics["cache_write_seconds"]
            metadata["director_runtime"] = runtime_diagnostics
            metadata["director_cache"] = copy.deepcopy(cache_info)
            packet["metadata"] = metadata
            return (
                _json_dumps(packet),
                reasoning_text,
                raw_output,
                _json_dumps(metadata),
                grounding_status,
                _json_dumps(report),
            )
        finally:
            _CURRENT_DIRECTOR_RUNTIME.reset(metric_token)


def _splatstage_blueprint_schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "splatstage_show_blueprint.schema.json"


_SPLATSTAGE_PLATE_FOREGROUND_TOKEN_PATTERN = re.compile(
    r"\b(?:person|people|human|man|men|woman|women|girl|boy|child|children|"
    r"face|body|silhouette|crowd|performer|presenter|dancer|singer)\b",
    re.IGNORECASE,
)


def _splatstage_blueprint_errors(value: Any, schema: dict[str, Any]) -> list[str]:
    try:
        from jsonschema import Draft202012Validator
    except Exception as exc:
        raise RuntimeError("DiffusionGemma SplatStage Planner requires jsonschema.") from exc

    def json_path(error: Any) -> str:
        parts = [str(part) for part in error.absolute_path]
        return "$" + "".join(f"[{part}]" if part.isdigit() else f".{part}" for part in parts)

    errors = [
        f"{json_path(error)}: {error.message}"
        for error in sorted(
            Draft202012Validator(schema).iter_errors(value),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    ]
    if not isinstance(value, dict):
        return errors
    music = value.get("music")
    if isinstance(music, dict):
        mode = str(music.get("vocal_mode", ""))
        lyrics = str(music.get("lyrics", "")).strip()
        if mode == "instrumental" and lyrics.casefold() not in {"[instrumental]", "instrumental"}:
            errors.append("$.music.lyrics: instrumental blueprints must use [Instrumental]")
        if mode == "vocal_hook" and lyrics.casefold() in {"[instrumental]", "instrumental"}:
            errors.append("$.music.lyrics: vocal_hook requires singable lyrics")
        if len([line for line in lyrics.splitlines() if line.strip()]) > 48:
            errors.append("$.music.lyrics: at most 48 non-empty lyric lines are allowed")
    performer = value.get("performer")
    if isinstance(performer, dict):
        segmentation = str(performer.get("segmentation_prompt", "")).strip()
        categories = [
            part.strip()
            for part in re.split(r"[,;]", segmentation)
            if part.strip()
        ]
        if len(categories) > 6:
            errors.append(
                "$.performer.segmentation_prompt: use at most 6 comma-separated SAM3 object categories"
            )
        if len(categories) == 1 and re.search(
            r"(?i)\b(?:and|holding|carrying|playing|wearing|wielding)\b",
            segmentation,
        ):
            errors.append(
                "$.performer.segmentation_prompt: separate the performer, worn silhouette items, "
                "instruments, and props with commas instead of describing them as one prose subject"
            )
    technical = re.compile(
        r"(?i)(?:\b(?:node(?:\s+id)?|sampler|checkpoint|model[_ -]?path|filename[_ -]?prefix|"
        r"workflow|output[_ -]?directory|denoise|cuda|comfyui)\b|"
        r"\b\d+\s+(?:sampling\s+)?steps\b|"
        r"[A-Za-z]:[\\/]|(?:^|[\s\"'])/(?:home|mnt|tmp|var)/)"
    )
    positive_negation = re.compile(
        r"(?i)\b(?:no|not|without|avoid|exclude|excluding|free\s+of|absent)\b|"
        r"\b(?:non[- ]?text(?:ual)?|non[- ]?symbolic|text[- ]free|letter[- ]free|"
        r"word[- ]free|logo[- ]free|watermark[- ]free|people[- ]free|person[- ]free)\b"
    )
    for section in ("performer", "background", "fx"):
        parent = value.get(section)
        positive = parent.get("positive_prompt") if isinstance(parent, dict) else None
        if isinstance(positive, str) and positive_negation.search(positive):
            errors.append(
                f"$.{section}.positive_prompt: use affirmative visual language only; "
                f"put exclusions in $.{section}.negative_prompt"
            )
        if isinstance(positive, str) and section in {"background", "fx"}:
            foreground_match = _SPLATSTAGE_PLATE_FOREGROUND_TOKEN_PATTERN.search(
                positive
            )
            if foreground_match is not None:
                errors.append(
                    f"$.{section}.positive_prompt: people-free plate required; remove foreground "
                    f"person token {foreground_match.group(0)!r} and describe only the environment "
                    "or abstract effect"
                )
    for section, field_name in (
        ("", "creative_summary"),
        ("music", "ace_tags"),
        ("performer", "positive_prompt"),
        ("performer", "negative_prompt"),
        ("background", "positive_prompt"),
        ("background", "negative_prompt"),
        ("fx", "positive_prompt"),
        ("fx", "negative_prompt"),
    ):
        parent = value if not section else value.get(section)
        text = parent.get(field_name) if isinstance(parent, dict) else None
        if isinstance(text, str) and technical.search(text):
            dotted = f"{section}.{field_name}".strip(".")
            errors.append(f"$.{dotted}: technical workflow instructions are not allowed")
    return errors


def _normalize_splatstage_lane_cardinality(
    value: Any,
    *,
    selection_index: int,
) -> tuple[Any, dict[str, Any]]:
    """Canonicalize an accidental one-or-two-option lane list to one object.

    The node workflow generates two clips from one creative lane contract. Some
    model responses mirror that physical clip count as arrays despite the JSON
    schema. Preserve fail-closed validation for malformed data, but safely
    select the same seed-addressed option across coherent lane alternatives.
    """

    if not isinstance(value, dict):
        return value, {}
    normalized = json.loads(json.dumps(value, ensure_ascii=False))
    report: dict[str, Any] = {}
    for lane in ("performer", "background", "fx"):
        options = normalized.get(lane)
        if not (
            isinstance(options, list)
            and 1 <= len(options) <= 2
            and all(isinstance(option, dict) for option in options)
        ):
            continue
        selected = int(selection_index) % len(options)
        normalized[lane] = options[selected]
        report[lane] = {
            "input_count": len(options),
            "selected_index": selected,
            "policy": "seed_addressed_single_lane_selection",
        }
    return normalized, report


def _normalize_splatstage_blueprint_identity(
    value: Any,
    schema: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Canonicalize only the schema document ID alias to its payload identity.

    The JSON Schema document is identified by ``splatstage.show_blueprint@1``,
    while blueprint payloads deliberately carry separate ``schema`` and
    ``version`` fields. DiffusionGemma can reasonably copy the document ID into
    the payload. Accept that one exact, version-matched alias without relaxing
    validation for any other schema name or version.
    """

    if not isinstance(value, dict):
        return value, {}
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return value, {}
    schema_contract = properties.get("schema")
    version_contract = properties.get("version")
    if not isinstance(schema_contract, dict) or not isinstance(version_contract, dict):
        return value, {}
    expected_schema = schema_contract.get("const")
    expected_version = version_contract.get("const")
    document_id = schema.get("$id")
    if not (
        isinstance(expected_schema, str)
        and isinstance(expected_version, int)
        and isinstance(document_id, str)
        and document_id == f"{expected_schema}@{expected_version}"
    ):
        return value, {}
    if value.get("schema") != document_id:
        return value, {}
    if value.get("version") != expected_version:
        return value, {}

    normalized = json.loads(json.dumps(value, ensure_ascii=False))
    normalized["schema"] = expected_schema
    return normalized, {
        "input_schema": document_id,
        "canonical_schema": expected_schema,
        "version": expected_version,
        "policy": "exact_version_matched_document_id_alias",
    }


_SPLATSTAGE_ASPECT_PATTERN = re.compile(
    r"^\s*(?P<width>(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?P<separator>:|/|x|×|by)\s*"
    r"(?P<height>(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?:\([^()\r\n]{1,80}\))?\s*$",
    re.IGNORECASE,
)


def _normalize_splatstage_duration_seconds(value: Any) -> float:
    """Return one positive finite planning duration without imposing a preset cap."""

    if isinstance(value, bool):
        raise ValueError("SplatStage duration_seconds must be a positive finite number.")
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SplatStage duration_seconds must be a positive finite number."
        ) from exc
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("SplatStage duration_seconds must be a positive finite number.")
    return duration


def _normalize_splatstage_aspect_ratio(value: Any) -> str:
    """Canonicalize a positive W:H ratio or WxH pixel-dimension expression."""

    text = str(value or "").strip()
    match = _SPLATSTAGE_ASPECT_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(
            "SplatStage aspect_ratio must be a positive W:H ratio or WxH resolution "
            "such as 9:16 or 1920x1080."
        )
    try:
        width = Decimal(match.group("width"))
        height = Decimal(match.group("height"))
    except InvalidOperation as exc:
        raise ValueError(
            "SplatStage aspect_ratio must be a positive W:H ratio or WxH resolution "
            "such as 9:16 or 1920x1080."
        ) from exc
    if not width.is_finite() or not height.is_finite() or width <= 0 or height <= 0:
        raise ValueError(
            "SplatStage aspect_ratio must use two positive finite dimensions."
        )

    separator = match.group("separator").casefold()
    if separator in {"x", "×", "by"}:
        width_integral = width == width.to_integral_value()
        height_integral = height == height.to_integral_value()
        if width_integral and height_integral:
            width_int = int(width)
            height_int = int(height)
            common = math.gcd(width_int, height_int)
            return f"{width_int // common}:{height_int // common}"

    def component(number: Decimal) -> str:
        normalized = format(number.normalize(), "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        return normalized

    return f"{component(width)}:{component(height)}"


_SPLATSTAGE_PRODUCTION_MODE_CHOICES = (
    "Source passthrough",
    "Joint video-safe plan",
    "Audition and select",
)


_SPLATSTAGE_EXPLICIT_GENRE_AUTHORITY_INSTRUCTION = (
    "Music-genre authority: If the creative brief explicitly requests a soundtrack genre or "
    "subgenre, treat that request as an immutable, highest-priority music constraint. Visual "
    "subject matter, setting, lighting, wardrobe, motion, energy, and words such as neon, club, "
    "nightclub, dance, or party may shape the visuals but must not change, broaden, hybridize, or "
    "override the requested genre. Juxtaposing a genre against a visual setting means audiovisual "
    "contrast, not musical fusion. Start ace_tags with the requested genre or subgenre, followed by "
    "genre-native instrumentation, groove, vocal character, arrangement, and production texture. "
    "When a different genre is explicit, do not introduce EDM, dance-pop, electronic-club, "
    "four-on-the-floor, synth-bass, riser, build, or drop signifiers unless the brief explicitly "
    "requests those signifiers or asks for a fusion, hybrid, crossover, or remix. These are planning "
    "rules; do not copy prohibition wording into ace_tags or lyrics. Lyrics may reference the visual "
    "setting as imagery, but must not introduce an incompatible drop, dance break, build, or other "
    "music-production directive unless the brief requests it."
)


def _normalize_splatstage_production_mode(value: Any) -> str:
    """Return the stable internal name for the song-production strategy."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    aliases = {
        "": "source_passthrough",
        "default": "source_passthrough",
        "source": "source_passthrough",
        "source_passthrough": "source_passthrough",
        "joint": "joint_video_safe_plan",
        "joint_video_safe": "joint_video_safe_plan",
        "joint_video_safe_plan": "joint_video_safe_plan",
        "video_safe": "joint_video_safe_plan",
        "audition": "audition_and_select",
        "audition_select": "audition_and_select",
        "audition_and_select": "audition_and_select",
    }
    if normalized not in aliases:
        raise ValueError(
            "SplatStage production_mode must be Source passthrough, "
            "Joint video-safe plan, or Audition and select."
        )
    return aliases[normalized]


def _splatstage_music_planning_instruction(production_mode: str) -> str:
    """Compile an ACE-native song brief without post-generation rewrites."""

    mode = _normalize_splatstage_production_mode(production_mode)
    shared = (
        "Choose bpm, key, and time_signature as dedicated metadata, then write ace_tags and lyrics "
        "as one naturally correlated ACE-Step plan. Do not repeat a numeric BPM, key name, meter, "
        "chord symbols, Roman-numeral progression, or music-theory lecture in ace_tags; the dedicated "
        "fields are the only authority for those values. Keep ace_tags in ordinary caption language "
        "covering genre, instrumentation, vocal character, arrangement, mood, and production texture. "
        "Write complete singable phrases, normally 6 to 10 syllables per sung line, and preserve room "
        "for instrumental breathing."
    )
    if mode == "source_passthrough":
        return (
            f"{shared} Source passthrough mode is active: author one internally coherent song bundle "
            "and expect every music field to reach ACE-Step unchanged. Do not rely on a downstream "
            "router to repair tempo, harmony, caption, or lyrics."
        )
    if mode == "joint_video_safe_plan":
        return (
            f"{shared} Joint video-safe plan mode is active: design the bundle this way from the outset, "
            "rather than rewriting it later. Use one stable perceived pulse with no double-time drum "
            "illusion, restrained and even transient density, a stable tonal center without abrupt "
            "major/minor or remote-key changes, and complete vocal phrases separated by instrumental "
            "space. Create section lift through melody, register, harmony, texture, or dynamics instead "
            "of denser percussion. These are creative planning constraints, not literal text to copy "
            "into ace_tags."
        )
    return (
        f"{shared} Audition and select mode is active: author one coherent source contract that can be "
        "rendered with several independent song seeds. Preserve enough natural variation for candidate "
        "auditioning, while keeping one stable perceived pulse, moderate and even transient density, a "
        "stable tonal family without abrupt modulation, complete vocal phrases, and instrumental breathing "
        "space. Avoid contradictory metadata and overcrowded lyrics. A downstream waveform "
        "quality gate will select and lock the safest actual excerpt; never claim that requested metadata "
        "guarantees the rendered tempo or tonality."
    )


def _splatstage_blueprint_prompt(
    user_prompt: str,
    root_seed: int,
    schema: dict[str, Any],
    *,
    duration_seconds: float,
    aspect_ratio: str,
    production_mode: str = "Source passthrough",
    visual_context: str = "",
) -> str:
    duration = _normalize_splatstage_duration_seconds(duration_seconds)
    normalized_aspect_ratio = _normalize_splatstage_aspect_ratio(aspect_ratio)
    normalized_production_mode = _normalize_splatstage_production_mode(production_mode)
    music_planning_instruction = _splatstage_music_planning_instruction(
        normalized_production_mode
    )
    compact_schema = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    duration_label = f"{duration:.12g}"
    excerpt_label = f"{duration_label}-second"
    cleaned_visual_context = _clean_visual_description(visual_context)
    visual_context_block = (
        f"\nAvailable image-derived visual context:\n{cleaned_visual_context}\n"
        if cleaned_visual_context
        else ""
    )
    return f"""You are the creative director for a fully automatic SplatStage music-video run.
Return exactly one JSON object and nothing else. It must validate against the JSON Schema below.
The two payload identity fields are exact and separate: use "schema":"splatstage.show_blueprint" and
"version":1. Never put the schema document identifier "splatstage.show_blueprint@1" in the schema field.

Creative brief:
{user_prompt}
{visual_context_block}
When Context Hub image pixels are attached, inspect them and use their visible subject, wardrobe, palette, environment,
and mood as creative evidence for the production concept, lyrical imagery, performer, background, and effects. Do not
contradict the visible image in the visual lanes merely to satisfy a generic genre convention. Image grounding informs
the coherent production concept; it does not require every visible object to be restated in lyrics.

{_SPLATSTAGE_EXPLICIT_GENRE_AUTHORITY_INSTRUCTION}

Production facts for this run are context only: duration {duration_label} seconds, aspect ratio {normalized_aspect_ratio},
two separately generated performer clips, two people-free background clips, and two people-free FX plates. The JSON still
contains exactly one performer object, one background object, and one FX object—never arrays or clip-specific lists. The
workflow derives both clip variants from each single creative lane definition. Do not put technical production
settings, model names, workflow instructions, file paths, node IDs, samplers, steps, or output prefixes in the JSON.
The root seed is {int(root_seed)}; use it only as creative entropy and do not include it in the JSON.

Every positive_prompt must use affirmative visual language only. Never put exclusions such as "no words",
"without people", "text-free", or "non-symbolic" in a positive_prompt. Put every unwanted element as a plain
comma-separated concept in that lane's negative_prompt instead; for example, positive_prompt "abstract neon
particles on black" and negative_prompt "text, words, letters, logos".

Interpret a generic pop-music request as a compact English 90-second vocal song with original lyrics. An explicit
instrumental request overrides that default and must use vocal_mode "instrumental" with lyrics "[Instrumental]".
Music production strategy: {normalized_production_mode}. {music_planning_instruction}
An explicit language request overrides English. Make the performer prompt describe one visible primary person with
useful full-body or medium-full motion. The performer segmentation_prompt is a SAM3 object-category bundle, not a
sentence: write 1 to 6 short comma-separated noun phrases. Put the primary performer first, followed by every visibly
present silhouette-extending costume item, worn headwear, handheld instrument, and attached prop described in the
performer prompt. For example: "cyberpunk mariachi musician, black charro suit, wide-brimmed mariachi hat, glowing
electric guitar". Omit categories that are not visibly present. Background and FX positive prompts must contain none
of these foreground-person tokens: person, people, human, man, men, woman, women, girl, boy, child, children, face,
body, silhouette, crowd, performer, presenter, dancer, or singer. The FX positive prompt must affirmatively describe isolated abstract light or
particle effects on a pure black backing. Put people, faces, logos, typography, letters, words, and symbols in the FX
negative prompt. The final video uses a {excerpt_label} excerpt, but ACE-Step
generates a complete 90-second source song first. For vocal music, target 10 to 14 sung lines for that full source
song; bracketed section labels do not count. Use one short singable clause and normally 6 to 10 syllables per line,
about two bars per sung line. Use at most four sung lines in a Verse or Chorus and at most two in a Pre-Chorus,
Bridge, or Outro. Use ACE bracketed section labels such as [Verse] and [Chorus], never parenthesized headings.
Repeat the Chorus wording verbatim. Reserve explicit [Instrumental] intro, interlude, or outro
space so the singer does not have to rush continuously through the whole 90 seconds.

JSON Schema:
{compact_schema}"""


class DiffusionGemmaSplatStagePlanner:
    """Strict creative planner for SplatStage Auto Director."""

    DESCRIPTION = (
        "Builds a strict SplatStage creative blueprint. Duration and aspect ratio are "
        "planning context only; downstream generators remain responsible for their own "
        "supported clocks and raster dimensions. The production strategy can preserve one "
        "source plan, author a joint video-safe plan, or prepare a multi-candidate audition."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_config": (MODEL_TYPE,),
                "gemma_context": (CONTEXT_TYPE,),
                "root_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9223372036854775807,
                        "control_after_generate": False,
                    },
                ),
                "duration_seconds": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 0.01,
                        "step": 0.01,
                        "tooltip": "Any positive finite planning duration. This node does not impose an upper preset; the downstream audio/video generator may still have its own limit.",
                    },
                ),
                "aspect_ratio": (
                    "STRING",
                    {
                        "default": "16:9",
                        "multiline": False,
                        "tooltip": "Creative framing context only. Enter any positive W:H ratio or WxH resolution, for example 9:16, 1:1, or 1920x1080. Downstream width and height remain authoritative.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 2048, "min": 768, "max": 3072, "step": 128},
                ),
                "production_mode": (
                    list(_SPLATSTAGE_PRODUCTION_MODE_CHOICES),
                    {
                        "default": "Source passthrough",
                        "tooltip": "Source passthrough sends one coherent song plan unchanged. Joint video-safe plan reduces transient, tempo, tonal, and lyric pressure before ACE-Step composes. Audition and select prepares one contract for several song seeds and a decoded-waveform quality gate.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("blueprint_json", "is_valid", "validation_errors_json", "raw_response")
    FUNCTION = "plan"
    OUTPUT_NODE = True
    CATEGORY = EXPERIMENTAL_MOTION_CATEGORY

    def plan(
        self,
        model_config: RuntimeConfig,
        gemma_context: GemmaContext,
        root_seed: int,
        duration_seconds: float,
        aspect_ratio: str,
        temperature: float,
        max_new_tokens: int,
        production_mode: str = "Source passthrough",
        unique_id: str | None = None,
    ):
        if not isinstance(model_config, RuntimeConfig):
            raise TypeError("model_config must come from DiffusionGemma Model Loader.")
        if not isinstance(gemma_context, GemmaContext):
            raise TypeError("gemma_context must come from DiffusionGemma Context Hub.")
        normalized_duration = _normalize_splatstage_duration_seconds(duration_seconds)
        normalized_aspect_ratio = _normalize_splatstage_aspect_ratio(aspect_ratio)
        normalized_production_mode = _normalize_splatstage_production_mode(
            production_mode
        )
        user_prompt = " ".join(str(gemma_context.user_prompt or "").split())
        if not user_prompt:
            raise ValueError("SplatStage Planner requires a non-empty creative prompt.")
        if len(user_prompt) > 4000:
            raise ValueError("SplatStage Planner prompt must be at most 4000 characters.")

        schema_path = _splatstage_blueprint_schema_path()
        if not schema_path.is_file():
            raise RuntimeError(f"SplatStage blueprint schema is missing: {schema_path}")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        runtime_config = _runtime_config_with_temperature(model_config, float(temperature))
        node_id = str(unique_id) if unique_id is not None else None
        backend_media_context = (
            _media_context_from_gemma_context(gemma_context)
            if _runtime_supports_pixels(runtime_config)
            else None
        )
        visual_context = _clean_visual_description(
            gemma_context.visual_description
            or str(gemma_context.media_metadata.get("visual_description", ""))
        )

        try:
            try:
                import torch

                torch.manual_seed(int(root_seed) & 0x7FFF_FFFF_FFFF_FFFF)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(root_seed) & 0x7FFF_FFFF_FFFF_FFFF)
            except Exception:
                pass
            initial_raw = _run_backend(
                runtime_config,
                _splatstage_blueprint_prompt(
                    user_prompt,
                    int(root_seed),
                    schema,
                    duration_seconds=normalized_duration,
                    aspect_ratio=normalized_aspect_ratio,
                    production_mode=normalized_production_mode,
                    visual_context=visual_context,
                ),
                backend_media_context,
                int(max_new_tokens),
                node_id=node_id,
            )
            parsed = _extract_json_object(initial_raw)
            parsed, initial_identity_normalization = (
                _normalize_splatstage_blueprint_identity(parsed, schema)
            )
            parsed, initial_normalization = _normalize_splatstage_lane_cardinality(
                parsed,
                selection_index=int(root_seed),
            )
            errors = _splatstage_blueprint_errors(parsed, schema)
            raw_record: dict[str, Any] = {
                "production_context": {
                    "duration_seconds": normalized_duration,
                    "aspect_ratio": normalized_aspect_ratio,
                    "production_mode": normalized_production_mode,
                    "pixels_sent_to_backend": bool(backend_media_context),
                    "visual_context_present": bool(visual_context),
                },
                "initial": initial_raw,
            }
            if initial_identity_normalization:
                raw_record["initial_identity_normalization"] = (
                    initial_identity_normalization
                )
            if initial_normalization:
                raw_record["initial_lane_normalization"] = initial_normalization
            if errors:
                if parsed is None:
                    raw_record["repair_strategy"] = (
                        "full_task_retry_after_unparseable_response"
                    )
                    repair_prompt = (
                        "The previous answer had no parseable JSON object. Start over and complete "
                        "the full task below; return exactly one schema-valid JSON object and nothing "
                        "else.\n\n"
                        + _splatstage_blueprint_prompt(
                            user_prompt,
                            int(root_seed),
                            schema,
                            duration_seconds=normalized_duration,
                            aspect_ratio=normalized_aspect_ratio,
                            production_mode=normalized_production_mode,
                            visual_context=visual_context,
                        )
                    )
                else:
                    repair_prompt = (
                        "Repair the following candidate so it validates. Return only the corrected JSON object. "
                        "Do not add fields and do not explain. performer, background, and fx must each be "
                        "one JSON object, never arrays; choose one coherent option if the candidate contains alternatives.\n\n"
                        "Repair JSON quoting, escapes, and delimiters without paraphrasing, respelling, truncating, "
                        "or otherwise changing any lyric line unless a validation error explicitly names music.lyrics. "
                        "Preserve every readable lyric character and word exactly.\n\n"
                        "Set the identity fields exactly to schema=\"splatstage.show_blueprint\" and version=1; "
                        "do not use the schema document ID splatstage.show_blueprint@1 as the schema value.\n\n"
                        "Validation errors:\n- "
                        + "\n- ".join(errors)
                        + "\n\nCandidate:\n"
                        + json.dumps(parsed, ensure_ascii=False)
                    )
                repair_seed = (int(root_seed) + 1) & 0x7FFF_FFFF_FFFF_FFFF
                try:
                    import torch

                    torch.manual_seed(repair_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(repair_seed)
                except Exception:
                    pass
                repaired_raw = _run_backend(
                    runtime_config,
                    repair_prompt,
                    backend_media_context,
                    int(max_new_tokens),
                    node_id=node_id,
                )
                raw_record["repair"] = repaired_raw
                parsed = _extract_json_object(repaired_raw)
                parsed, repair_identity_normalization = (
                    _normalize_splatstage_blueprint_identity(parsed, schema)
                )
                if repair_identity_normalization:
                    raw_record["repair_identity_normalization"] = (
                        repair_identity_normalization
                    )
                parsed, repair_normalization = _normalize_splatstage_lane_cardinality(
                    parsed,
                    selection_index=int(root_seed),
                )
                if repair_normalization:
                    raw_record["repair_lane_normalization"] = repair_normalization
                errors = _splatstage_blueprint_errors(parsed, schema)
            if errors or parsed is None:
                raise RuntimeError(
                    "DiffusionGemma could not produce a valid SplatStage blueprint after one repair: "
                    + "; ".join(errors or ["response did not contain a JSON object"])
                )
            blueprint_json = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            raw_response = json.dumps(raw_record, ensure_ascii=False)
            result = (blueprint_json, True, "[]", raw_response)
            return {
                "ui": {
                    "splatstage_show_blueprint": [blueprint_json],
                    "splatstage_raw_response": [raw_response],
                },
                "result": result,
            }
        finally:
            _maybe_unload(runtime_config)


def _splatstage_efficient_blueprint_schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "splatstage_efficient_show_blueprint.schema.json"


def _splatstage_efficient_blueprint_errors(
    value: Any,
    schema: dict[str, Any],
) -> list[str]:
    try:
        from jsonschema import Draft202012Validator
    except Exception as exc:
        raise RuntimeError(
            "DiffusionGemma Efficient SplatStage Planner requires jsonschema."
        ) from exc

    def json_path(error: Any) -> str:
        parts = [str(part) for part in error.absolute_path]
        return "$" + "".join(
            f"[{part}]" if part.isdigit() else f".{part}" for part in parts
        )

    errors = [
        f"{json_path(error)}: {error.message}"
        for error in sorted(
            Draft202012Validator(schema).iter_errors(value),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    ]
    if not isinstance(value, dict):
        return errors

    positive_negation = re.compile(
        r"(?i)\b(?:no|not|without|avoid|exclude|excluding|free\s+of|absent)\b|"
        r"\b(?:non[- ]?text(?:ual)?|non[- ]?symbolic|text[- ]free|letter[- ]free|"
        r"word[- ]free|logo[- ]free|watermark[- ]free|people[- ]free|person[- ]free)\b"
    )
    technical = re.compile(
        r"(?i)(?:\b(?:node(?:\s+id)?|sampler|checkpoint|model[_ -]?path|filename[_ -]?prefix|"
        r"workflow|output[_ -]?directory|denoise|cuda|comfyui)\b|"
        r"\b\d+\s+(?:sampling\s+)?steps\b|[A-Za-z]:[\\/])"
    )
    positive_fields: list[tuple[str, Any]] = []
    for lane in ("performer", "background", "fx"):
        section = value.get(lane)
        if not isinstance(section, dict):
            continue
        positive_fields.append((f"$.{lane}.positive_prompt", section.get("positive_prompt")))
        if lane in {"performer", "background"}:
            variations = section.get("variations")
            if isinstance(variations, list):
                positive_fields.extend(
                    (f"$.{lane}.variations[{index}]", variation)
                    for index, variation in enumerate(variations)
                )
    for path, text in positive_fields:
        if isinstance(text, str) and positive_negation.search(text):
            errors.append(
                f"{path}: use affirmative visual language only; put exclusions in the lane negative_prompt"
            )
        if isinstance(text, str) and technical.search(text):
            errors.append(f"{path}: technical workflow instructions are not allowed")

    fx = value.get("fx")
    if isinstance(fx, dict):
        fx_positive = str(fx.get("positive_prompt", ""))
        if re.search(
            r"(?i)\b(?:touchdesigner|software|application|app|editor|viewport|"
            r"interface|node\s+graph|control\s+panel|menu|screenshot)\b",
            fx_positive,
        ):
            errors.append(
                "$.fx.positive_prompt: describe only the show-specific artwork; "
                "SplatStage supplies the locked TouchDesigner-inspired finished-artwork style"
            )

    performer = value.get("performer")
    if isinstance(performer, dict):
        segmentation = str(performer.get("segmentation_prompt", "")).strip()
        categories = [
            part.strip() for part in re.split(r"[,;]", segmentation) if part.strip()
        ]
        if not 1 <= len(categories) <= 6:
            errors.append(
                "$.performer.segmentation_prompt: use 1 to 6 comma-separated object categories"
            )
        if len(categories) == 1 and re.search(
            r"(?i)\b(?:and|holding|carrying|playing|wearing|wielding)\b",
            segmentation,
        ):
            errors.append(
                "$.performer.segmentation_prompt: separate the performer, headwear, costume, instrument, and props with commas"
            )
        generic_component = re.compile(
            r"(?i)^(?:face and head|head and face|hair and worn headwear|"
            r"costume and worn accessories|held instrument and attached props|"
            r"full body person|torso|arms|hands|legs)$"
        )
        if any(generic_component.fullmatch(category) for category in categories[1:]):
            errors.append(
                "$.performer.segmentation_prompt: list only identity-specific items actually present; "
                "generic anatomy and placeholder accessory categories are added deterministically"
            )

    music = value.get("music")
    if isinstance(music, dict):
        mode = str(music.get("vocal_mode", ""))
        lyrics = str(music.get("lyrics", "")).strip()
        if mode == "instrumental" and lyrics.casefold() not in {
            "[instrumental]",
            "instrumental",
        }:
            errors.append(
                "$.music.lyrics: instrumental blueprints must use [Instrumental]"
            )
        if mode == "vocal_hook" and lyrics.casefold() in {
            "[instrumental]",
            "instrumental",
        }:
            errors.append("$.music.lyrics: vocal_hook requires singable lyrics")
        lyric_lines = [line for line in lyrics.splitlines() if line.strip()]
        if len(lyric_lines) > 48:
            errors.append("$.music.lyrics: at most 48 non-empty lyric lines are allowed")
    return errors


def _normalize_splatstage_efficient_segmentation(value: Any) -> Any:
    """Make the canonical identity authoritative for the SAM3 prompt bundle.

    Generic anatomy queries caused SAM3 to union unrelated people from crowded
    source plates.  The efficient workflow now asks for one connected character
    identity and adds completeness language downstream.  This normalizer keeps
    only concrete, actually authored costume/headwear/instrument/prop hints.
    """

    if not isinstance(value, dict):
        return value
    normalized = copy.deepcopy(value)
    performer = normalized.get("performer")
    if not isinstance(performer, dict):
        return normalized
    identity = " ".join(str(performer.get("identity_description", "")).strip().split())
    identity = re.sub(r"(?i)^(?:a|an|the)\s+", "", identity).strip(" ,.;")
    authored = [
        " ".join(part.strip().split()).strip(" ,.;")
        for part in re.split(r"[,;]", str(performer.get("segmentation_prompt", "")))
        if part.strip()
    ]
    if not identity and authored:
        identity = authored[0]
    if not identity:
        return normalized

    generic = re.compile(
        r"(?i)^(?:face and head|head and face|hair and worn headwear|"
        r"costume and worn accessories|held instrument and attached props|"
        r"full body person|torso|arms|hands|legs)$"
    )
    specific = re.compile(
        r"(?i)\b(?:hat|helmet|headdress|crown|cap|hood|turban|goggles|glasses|"
        r"jacket|coat|robe|dress|suit|armor|charro|backpack|cape|wings?|"
        r"guitar|bass|violin|cello|trumpet|trombone|saxophone|drum|keyboard|"
        r"piano|microphone|sword|staff|cane|shield|instrument|prop)\w*\b"
    )
    candidates = authored[1:] if authored else []
    candidates.extend(
        " ".join(part.strip().split()).strip(" ,.;")
        for part in re.split(r"[,;]", str(performer.get("positive_prompt", "")))
        if part.strip()
    )
    selected = [identity]
    for category in candidates:
        if generic.fullmatch(category) or not specific.search(category):
            continue
        if category.casefold() in {item.casefold() for item in selected}:
            continue
        selected.append(category)
        if len(selected) >= 6:
            break
    performer["segmentation_prompt"] = ", ".join(selected)
    return normalized


def _splatstage_efficient_blueprint_prompt(
    user_prompt: str,
    root_seed: int,
    schema: dict[str, Any],
) -> str:
    compact_schema = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return f"""You are the creative director for an efficient, fully automatic SplatStage music video.
Return exactly one JSON object and nothing else. It must validate against the JSON Schema below.
Use the exact payload identity \"schema\":\"splatstage.efficient_show_blueprint\" and \"version\":1.

Creative brief:
{user_prompt}

{_SPLATSTAGE_EXPLICIT_GENRE_AUTHORITY_INSTRUCTION}

The production creates one complete 90-second song, three separate five-second performer videos,
two separate five-second people-free background videos, and one still-image abstract FX source.
Describe one canonical performer identity, then provide exactly three affirmative shot/action/camera
variations that preserve that identity. Describe one background art direction, then provide exactly
two affirmative environment/camera variations. The segmentation_prompt is a comma-separated SAM3
identity hint bundle with at most six entries. Restate the canonical performer identity first, then list
only concrete silhouette-extending items actually present, such as a specific hat, costume, held
instrument, or attached prop. Do not invent generic placeholders for absent items. Example:
\"cyberpunk mariachi musician, wide-brimmed mariachi hat, black charro suit, glowing electric guitar\".

Every positive_prompt and every variation must use affirmative visual language only. Never write
phrases such as \"no words\", \"without people\", \"text-free\", or \"non-symbolic\" in positive
fields. Put exclusions only in negative_prompt. The background positive prompt must describe scenery
without introducing a person, crowd, performer, dancer, singer, face, body, or human silhouette. The
FX positive_prompt is the show-specific creative slot for one richly art-directed luminous motif.
Coordinate it with the same theme, music, performer, and background, and explicitly specify: the
abstract motif or shape language; a limited color palette; material and texture; rhythmic energy or
particle behavior; and a clear spatial composition with controlled negative space. Use affirmative
visual language and prefer organic light phenomena or sculptural forms over recognizable icons.
Do not name TouchDesigner or any software, application, interface, editor, node graph, control panel,
viewport, or screenshot in the positive prompt. SplatStage adds a locked TouchDesigner-inspired
finished-artwork style spine after planning. Put people, faces, typography, letters, words, logos,
watermarks, symbols, UI, HUD, labels, and signage in the FX negative prompt.

Interpret a generic pop request as an English vocal song with an original hook and 10 to 14 sung lyric
lines for the full 90-second song. Bracketed section labels do not count. Use one short clause and normally
6 to 10 syllables per line. Use ACE bracketed section labels such as [Verse] and [Chorus], never parenthesized
headings; repeat the Chorus verbatim and reserve explicit [Instrumental] breathing space.
Choose bpm, key, and time_signature first; make ace_tags and lyrics consistent with them, use one stable
tonal center and simple diatonic four-bar phrase loops, and avoid arbitrary modulation. Explicit instrumental
or language requests override that default. Do not include model names, node IDs, dimensions, frame
counts, samplers, steps, paths, output prefixes, or other technical workflow instructions. The root
seed {int(root_seed)} is creative entropy only and must not appear in the JSON.

JSON Schema:
{compact_schema}"""


_SPLATSTAGE_EFFICIENT_PLANNER_PROMPT_VERSION = 3


def _splatstage_efficient_cache_path(user_prompt: str, root_seed: int) -> Path:
    import hashlib
    import folder_paths

    identity = hashlib.sha256(
        json.dumps(
            {
                "schema": "splatstage.efficient_planner_cache_input",
                "version": 1,
                "user_prompt": " ".join(str(user_prompt).split()),
                "root_seed": int(root_seed),
                "planner_prompt_version": _SPLATSTAGE_EFFICIENT_PLANNER_PROMPT_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    root = (
        Path(folder_paths.get_output_directory()).resolve()
        / "4d_splat"
        / "splatstage_efficient_node_workflow"
        / "planner"
        / identity
    )
    return root / "blueprint.json"


class DiffusionGemmaSplatStageEfficientPlanner:
    """One-call creative planner for the efficient full-song node workflow."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_config": (MODEL_TYPE,),
                "gemma_context": (CONTEXT_TYPE,),
                "root_seed": (
                    "INT",
                    {
                        "default": 26073001,
                        "min": 0,
                        "max": 9223372036854775807,
                        "control_after_generate": False,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 3072, "min": 1024, "max": 4096, "step": 128},
                ),
                "reuse_cached_blueprint": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = (
        "blueprint_json",
        "is_valid",
        "validation_errors_json",
        "raw_response",
    )
    FUNCTION = "plan"
    OUTPUT_NODE = True
    CATEGORY = EXPERIMENTAL_MOTION_CATEGORY

    def plan(
        self,
        model_config: RuntimeConfig,
        gemma_context: GemmaContext,
        root_seed: int,
        temperature: float,
        max_new_tokens: int,
        reuse_cached_blueprint: bool,
        unique_id: str | None = None,
    ):
        if not isinstance(model_config, RuntimeConfig):
            raise TypeError("model_config must come from DiffusionGemma Model Loader.")
        if not isinstance(gemma_context, GemmaContext):
            raise TypeError("gemma_context must come from DiffusionGemma Context Hub.")
        user_prompt = " ".join(str(gemma_context.user_prompt or "").split())
        if not user_prompt:
            raise ValueError("Efficient SplatStage Planner requires a non-empty prompt.")
        if len(user_prompt) > 4000:
            raise ValueError("Efficient SplatStage Planner prompt must be at most 4000 characters.")

        schema_path = _splatstage_efficient_blueprint_schema_path()
        if not schema_path.is_file():
            raise RuntimeError(f"Efficient SplatStage blueprint schema is missing: {schema_path}")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        cache_path = _splatstage_efficient_cache_path(user_prompt, int(root_seed))
        if bool(reuse_cached_blueprint) and cache_path.is_file():
            try:
                cached = _normalize_splatstage_efficient_segmentation(
                    json.loads(cache_path.read_text(encoding="utf-8"))
                )
                errors = _splatstage_efficient_blueprint_errors(cached, schema)
                if not errors:
                    atomic_cache = cache_path.with_name(
                        f".{cache_path.name}.{os.getpid()}.partial"
                    )
                    atomic_cache.write_text(
                        json.dumps(cached, ensure_ascii=False, sort_keys=True, indent=2),
                        encoding="utf-8",
                    )
                    os.replace(atomic_cache, cache_path)
                    blueprint_json = json.dumps(
                        cached, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    raw_response = json.dumps(
                        {"cache": "reused", "path": str(cache_path)}, ensure_ascii=False
                    )
                    return {
                        "ui": {
                            "splatstage_efficient_show_blueprint": [blueprint_json],
                            "splatstage_raw_response": [raw_response],
                        },
                        "result": (blueprint_json, True, "[]", raw_response),
                    }
            except (OSError, json.JSONDecodeError):
                pass

        runtime_config = _runtime_config_with_temperature(model_config, float(temperature))
        node_id = str(unique_id) if unique_id is not None else None
        try:
            initial_raw = _run_backend(
                runtime_config,
                _splatstage_efficient_blueprint_prompt(
                    user_prompt, int(root_seed), schema
                ),
                None,
                int(max_new_tokens),
                node_id=node_id,
            )
            parsed = _normalize_splatstage_efficient_segmentation(
                _extract_json_object(initial_raw)
            )
            errors = _splatstage_efficient_blueprint_errors(parsed, schema)
            raw_record: dict[str, Any] = {"initial": initial_raw}
            if errors:
                repair_prompt = (
                    "Repair this candidate to satisfy the validation errors. Return only the corrected JSON object, "
                    "with no explanation and no extra fields. Keep every positive prompt and variation affirmative; "
                    "move exclusions into negative_prompt.\n\nValidation errors:\n- "
                    + "\n- ".join(errors)
                    + "\n\nCandidate:\n"
                    + (
                        json.dumps(parsed, ensure_ascii=False)
                        if parsed is not None
                        else initial_raw
                    )
                )
                repaired_raw = _run_backend(
                    runtime_config,
                    repair_prompt,
                    None,
                    int(max_new_tokens),
                    node_id=node_id,
                )
                raw_record["repair"] = repaired_raw
                parsed = _normalize_splatstage_efficient_segmentation(
                    _extract_json_object(repaired_raw)
                )
                errors = _splatstage_efficient_blueprint_errors(parsed, schema)
            if errors or parsed is None:
                raise RuntimeError(
                    "DiffusionGemma could not produce a valid efficient SplatStage blueprint after one repair: "
                    + "; ".join(errors or ["response did not contain a JSON object"])
                )

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            partial = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.partial")
            partial.write_text(
                json.dumps(parsed, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            os.replace(partial, cache_path)
            diagnostic = cache_path.parent / "raw_response.json"
            diagnostic.write_text(
                json.dumps(raw_record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            blueprint_json = json.dumps(
                parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            raw_response = json.dumps(raw_record, ensure_ascii=False)
            return {
                "ui": {
                    "splatstage_efficient_show_blueprint": [blueprint_json],
                    "splatstage_raw_response": [raw_response],
                },
                "result": (blueprint_json, True, "[]", raw_response),
            }
        finally:
            _maybe_unload(runtime_config)


class DiffusionGemmaJSONSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_json": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            },
            "optional": {
                "gemma_context": (CONTEXT_TYPE,),
                "target_profile_config": (TARGET_PROFILE_TYPE,),
                "fallback_prompt": ("STRING", {"default": "", "multiline": True}),
                "resolution_megapixels": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": "Used only for the splitter's resolution_width/resolution_height outputs. Matches Resolution Selector megapixels.",
                    },
                ),
                "resolution_multiple": (
                    "INT",
                    {
                        "default": 8,
                        "min": 8,
                        "max": 128,
                        "step": 4,
                        "tooltip": "Used only for the splitter's resolution_width/resolution_height outputs. Matches Resolution Selector multiple.",
                    },
                ),
                "resolution_aspect_ratio": (
                    list(RESOLUTION_SELECTOR_ASPECT_CHOICES),
                    {
                        "default": RESOLUTION_SELECTOR_AUTO,
                        "tooltip": "Aspect ratio used by resolution_width/resolution_height. Auto matches an attached LTX first frame, uses the Ideogram target ratio, or defaults unconditioned LTX to 16:9. Select a ratio here to override it explicitly.",
                    },
                ),
                "resolution_aspect_ratio_override": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Optional canonical W:H ratio supplied by a project-level contract. When connected and non-empty, it overrides the local aspect widget without shifting legacy widget values.",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "BOOLEAN",
        "STRING",
        "INT",
        "INT",
        "BOOLEAN",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "ltx_prompt",
        "ideogram_prompt",
        "negative_prompt",
        "aspect_ratio",
        "metadata_json",
        "scene_segments_json",
        "local_prompts",
        "segment_lengths",
        "is_valid",
        "resolution_selector_preset",
        "resolution_width",
        "resolution_height",
        "ready_for_generation",
        "minimax_h3_prompt",
        "candidate_ltx_prompt",
        "candidate_negative_prompt",
        "candidate_minimax_h3_prompt",
    )
    FUNCTION = "split"
    CATEGORY = CATEGORY

    def split(
        self,
        final_json: str,
        gemma_context: GemmaContext | None = None,
        target_profile_config: TargetProfileConfig | None = None,
        fallback_prompt: str = "",
        resolution_megapixels: float = 2.0,
        resolution_multiple: int = 8,
        resolution_aspect_ratio: str = RESOLUTION_SELECTOR_AUTO,
        resolution_aspect_ratio_override: str = "",
    ):
        split_start = time.perf_counter()
        context = gemma_context if isinstance(gemma_context, GemmaContext) else _gemma_context_from_media(fallback_prompt)
        target = target_profile_config if isinstance(target_profile_config, TargetProfileConfig) else _make_target_profile_config()
        cleaned = _strip_markdown(_strip_thinking(str(final_json or "")))
        strict_packet: dict[str, Any] | None = None
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                strict_packet = parsed
        except Exception:
            strict_packet = None
        if strict_packet is None:
            packet, is_valid, warning = repair_or_salvage_prompt_packet(final_json, context, target, 8000)
        else:
            packet, is_valid, warning = strict_packet, True, ""
        packet_metadata_for_split = packet.setdefault("metadata", {}) if isinstance(packet, dict) else {}
        if isinstance(packet_metadata_for_split, dict):
            if "model_output_json_parse_valid" not in packet_metadata_for_split:
                packet_metadata_for_split["model_output_json_parse_valid"] = bool(
                    packet_metadata_for_split.get(
                        "json_parse_valid",
                        strict_packet is not None,
                    )
                )
            packet_metadata_for_split["splitter_input_json_valid"] = bool(
                strict_packet is not None
            )
            if warning:
                packet_metadata_for_split["salvage_warning"] = warning
        ltx_prompt, ideogram_prompt, negative_prompt, scene_segments_json, local_prompts, segment_lengths, packet_aspect_ratio, packet_metadata, minimax_h3_prompt = _packet_to_prompt_outputs(
            packet,
            context,
            target,
            8000,
        )
        # Preserve diagnostic copies before the fail-closed routing outputs are
        # blanked. These appended sockets are diagnostic or explicit-policy
        # inputs: native generation must continue through a Generation Gate.
        candidate_ltx_prompt = ltx_prompt
        candidate_negative_prompt = negative_prompt
        candidate_minimax_h3_prompt = minimax_h3_prompt
        target_dict = _target_profile_config_to_dict(target)
        project_aspect_override = str(resolution_aspect_ratio_override or "").strip()
        requested_resolution_aspect_ratio = (
            project_aspect_override or resolution_aspect_ratio
        )
        aspect_ratio, resolution_aspect_ratio_source = _splitter_resolution_aspect_ratio(
            requested_resolution_aspect_ratio,
            context,
            target_dict,
            packet_aspect_ratio,
        )
        if project_aspect_override:
            match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", project_aspect_override)
            normalized_override = (
                f"{int(match.group(1))}:{int(match.group(2))}" if match else ""
            )
            if normalized_override not in RESOLUTION_SELECTOR_RATIOS:
                raise ValueError(
                    "resolution_aspect_ratio_override must be one of the supported canonical W:H ratios: "
                    + ", ".join(RESOLUTION_SELECTOR_RATIOS)
                    + "."
                )
            aspect_ratio = normalized_override
            resolution_aspect_ratio_source = "project_master_contract"
        resolution_selector_preset = _resolution_selector_preset(aspect_ratio)
        resolution_width, resolution_height = _resolution_selector_dimensions(
            aspect_ratio,
            resolution_megapixels,
            resolution_multiple,
        )
        metadata = _extract_json_object(packet_metadata) or {}
        ready_for_generation = bool(metadata.get("ready_for_generation"))
        blocked_reasons = metadata.get("blocked_reasons") if isinstance(metadata.get("blocked_reasons"), list) else []
        if not ready_for_generation:
            ltx_prompt = ""
            ideogram_prompt = ""
            negative_prompt = ""
            scene_segments_json = "[]"
            local_prompts = ""
            segment_lengths = ""
            minimax_h3_prompt = ""
        metadata.update(
            {
                "splitter_json_valid": bool(is_valid and ready_for_generation),
                "splitter_input_json_valid": bool(strict_packet is not None),
                "splitter_warning": warning,
                "target_profile": target_dict["target_profile"],
                "aspect_ratio": aspect_ratio,
                "ready_for_generation": ready_for_generation,
                "blocked_reasons": blocked_reasons,
                "resolution_selector_preset": resolution_selector_preset,
                "resolution_width": resolution_width,
                "resolution_height": resolution_height,
                "resolution_megapixels": max(0.1, min(16.0, _safe_float(resolution_megapixels, 2.0))),
                "resolution_multiple": int(max(8, min(128, _safe_float(resolution_multiple, 8)))),
                "resolution_aspect_ratio_requested": str(
                    requested_resolution_aspect_ratio or RESOLUTION_SELECTOR_AUTO
                ),
                "resolution_aspect_ratio_source": resolution_aspect_ratio_source,
                "candidate_prompt_diagnostic_only": True,
                "candidate_ltx_prompt_sha256": (
                    hashlib.sha256(candidate_ltx_prompt.encode("utf-8")).hexdigest()
                    if candidate_ltx_prompt
                    else ""
                ),
                "candidate_negative_prompt_sha256": (
                    hashlib.sha256(candidate_negative_prompt.encode("utf-8")).hexdigest()
                    if candidate_negative_prompt
                    else ""
                ),
                "candidate_minimax_h3_prompt_sha256": (
                    hashlib.sha256(candidate_minimax_h3_prompt.encode("utf-8")).hexdigest()
                    if candidate_minimax_h3_prompt
                    else ""
                ),
                "splitter_timing": {
                    "split_seconds": round(time.perf_counter() - split_start, 3),
                    "input_chars": len(str(final_json or "")),
                },
            }
        )
        return (
            ltx_prompt,
            ideogram_prompt,
            negative_prompt,
            aspect_ratio,
            _json_dumps(metadata),
            scene_segments_json,
            local_prompts,
            segment_lengths,
            bool(is_valid and ready_for_generation),
            resolution_selector_preset,
            resolution_width,
            resolution_height,
            ready_for_generation,
            minimax_h3_prompt,
            candidate_ltx_prompt,
            candidate_negative_prompt,
            candidate_minimax_h3_prompt,
        )


_LTX25_REASON_MESSAGES = {
    "ltx_h3_reference_context_incompatible": "LTX-2.5 is connected to an H3 Reference Context. Use DiffusionGemma Context Hub; leave both frame sockets empty for T2V, connect image for I2V, or connect image plus last_frame_image for FLF.",
    "ltx_t2v_has_conditioning_frame": "Text-to-video mode has a conditioning frame attached. Disconnect the frame or select Image to video / First + last frame.",
    "ltx_i2v_first_frame_missing": "Image-to-video needs a first-frame image in Context Hub's image socket.",
    "ltx_flf_first_frame_missing": "First+last-frame mode needs a first-frame image in Context Hub's image socket.",
    "ltx_flf_last_frame_missing": "First+last-frame mode needs an image in Context Hub's last_frame_image socket.",
    "ltx_first_frame_grounding_incomplete": "The first-frame analysis did not verify enough subject/setting/composition facts. Re-run Grounding Guard or provide a factual visual description.",
    "ltx_last_frame_grounding_incomplete": "The last-frame analysis did not verify enough subject/setting/composition facts. Re-run Grounding Guard or provide a factual visual description.",
    "ltx_first_frame_prompt_conflict": "The generated opening contradicts the verified first frame (for example day/night, interior/exterior, or close-up/wide framing). Revise the brief or remove the conditioning frame.",
    "ltx_conditioned_mode_requires_single_continuous_take": "I2V and first+last-frame prompts must use one continuous take; remove cuts or switch to T2V.",
    "ltx_unspecified_rapid_cuts": "Replace vague rapid/quick cuts with a small number of explicit shot changes, or use one continuous shot.",
    "ltx_shot_scale_missing": "Each LTX-2.5 shot needs an opening shot type: extreme wide shot, wide shot, medium shot, medium close-up, close-up, or extreme close-up. An ending scale does not replace the opening type.",
    "ltx_camera_state_missing": "Each LTX-2.5 shot needs an explicit camera state: either a static/locked hold or a physically continuous camera path.",
    "ltx_viewpoint_missing": "Each LTX-2.5 shot needs a viewpoint relative to the subject, such as eye-level front view, low-angle side view, or overhead.",
    "ltx_shot_count_exceeds_duration_budget": "The prompt asks for more shots than fit the selected duration. Remove cuts or increase duration.",
    "ltx_camera_complexity_exceeds_duration_budget": "An older cached LTX validation packet applied the retired numeric camera-move limit. Refresh the Director prompt; compatible motion phases inside one continuous path are now advisory.",
    "ltx_stable_camera_orbit_or_rotation": "Stable / base-model camera mode rejects orbit, circular-arc, roll, rotation, spin, and camera-swirl language. Use a locked/stabilized view or one restrained single-axis move, or explicitly select Advanced / controlled camera when the workflow has suitable motion control or a camera LoRA.",
    "ltx_stable_camera_sweeping_or_whip_motion": "Stable / base-model camera mode rejects sweeping and whip camera movement. Replace it with a locked/stabilized view or a short gentle push, pull, pan, tilt, or lateral track.",
    "ltx_stable_camera_dolly_zoom": "Stable / base-model camera mode rejects dolly-zoom/vertigo movement. Use a simple push or pull, or select Advanced / controlled camera with suitable support.",
    "ltx_stable_camera_shake_or_handheld_pursuit": "Stable / base-model camera mode rejects camera shake and handheld pursuit. Use stable framing or a restrained stabilized track.",
    "ltx_stable_camera_compound_or_reversal": "Stable / base-model camera mode allows at most one restrained single-axis camera move and rejects compound, multi-axis, or reversing paths.",
    "ltx_stable_camera_pronounced_parallax": "Stable / base-model camera mode rejects pronounced, layered, or dizzying parallax. Preserve readable geometry with stable framing and only minimal natural depth change.",
    "ltx_stable_camera_sustained_travel": "Stable long-horizon mode must fill time with subject, prop, lighting, atmospheric, environmental, and sound evolution—not sustained camera travel.",
    "ltx_action_density_exceeds_duration_budget": "The prompt contains too many major action beats for the selected duration.",
    "ltx_audio_density_exceeds_duration_budget": "The prompt contains too many independent sound layers for the selected duration.",
    "ltx_speech_exceeds_duration_budget": "The quoted speech is too long to be spoken clearly within the selected duration.",
}

_MINIMAX_H3_REASON_MESSAGES = {
    "minimax_h3_ref_subject_count_mismatch": (
        "MiniMax H3 Ref2VA returned the wrong number of semantic <Subject N> labels. Set H3 Reference Context's "
        "expected_subject_count to 0 for Auto, or keep the exact count and queue again after Director repair."
    ),
    "minimax_h3_non_diegetic_music_invalid": (
        "MiniMax H3 returned the malformed music sentinel '/A'. Use exactly 'N/A' when no score is present, or provide a complete music description."
    ),
    "minimax_h3_prompt_contains_grounding_role_annotation": (
        "MiniMax H3 copied a host-only dg grounding-role annotation into the native prompt. Remove the complete or dangling dg annotation and queue again."
    ),
}


def _friendly_generation_block_reasons(reasons: list[str], metadata: dict[str, Any]) -> list[str]:
    diagnostics = metadata.get("ltx25_contract") if isinstance(metadata.get("ltx25_contract"), dict) else {}
    complexity = diagnostics.get("complexity") if isinstance(diagnostics.get("complexity"), dict) else {}
    observed = complexity.get("observed") if isinstance(complexity.get("observed"), dict) else {}
    budget = complexity.get("budget") if isinstance(complexity.get("budget"), dict) else {}
    messages: list[str] = []
    for reason in reasons:
        message = _LTX25_REASON_MESSAGES.get(
            reason,
            _MINIMAX_H3_REASON_MESSAGES.get(reason, reason),
        )
        if reason == "ltx_shot_count_exceeds_duration_budget" and observed and budget:
            message += f" Detected {observed.get('estimated_shots', '?')}; limit {budget.get('max_shots', '?')}."
        elif reason == "ltx_camera_complexity_exceeds_duration_budget" and observed and budget:
            matches = observed.get("camera_motion_matches")
            phrases = [
                str(item.get("text", "")).strip()
                for item in matches
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ] if isinstance(matches, list) else []
            if phrases:
                message += f" Retired matcher phrases: {', '.join(phrases)}."
        elif reason == "ltx_action_density_exceeds_duration_budget" and observed and budget:
            message += f" Detected {observed.get('distinct_major_actions', '?')}; recommended limit {budget.get('max_major_actions', '?')}."
        elif reason == "minimax_h3_ref_subject_count_mismatch":
            expected = _normalize_minimax_h3_expected_subject_count(
                metadata.get("minimax_h3_expected_subject_count", 0)
            )
            actual = int(
                max(
                    0,
                    _safe_float(metadata.get("minimax_h3_actual_subject_count"), 0.0),
                )
            )
            if expected:
                message += f" Expected {expected}; generated {actual}."
        messages.append(message)
    return messages


class DiffusionGemmaGenerationGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "ready_for_generation": ("BOOLEAN", {"default": False, "forceInput": True}),
            },
            "optional": {
                "metadata_json": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "validation_policy": (
                    ["strict", "warn_and_continue"],
                    {
                        "default": "strict",
                    "tooltip": "Strict blocks every validator failure. Warn and continue permits a non-empty MiniMax-H3 candidate when every remaining failure comes from the MiniMax-H3 validator; Grounding Guard and non-H3 failures always remain blocking.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "gate"
    CATEGORY = CATEGORY

    def gate(
        self,
        prompt: str,
        ready_for_generation: bool,
        metadata_json: str = "",
        validation_policy: str = "strict",
    ):
        metadata = _extract_json_object(metadata_json) or {}
        guard_blocked, guard_reasons, guard_report = _strict_grounding_guard_block(
            metadata
        )
        if guard_blocked:
            details = ", ".join(guard_reasons)
            message = "DiffusionGemma Grounding Guard blocked generation."
            if details:
                message += f" Guard reasons: {details}."
            validation = (
                guard_report.get("validation")
                if isinstance(guard_report.get("validation"), dict)
                else {}
            )
            diagnostics = _reason_strings(validation.get("errors"))
            diagnostics.extend(
                reason
                for reason in _reason_strings(guard_report.get("retry_reasons"))
                if reason not in diagnostics
            )
            if diagnostics:
                displayed = diagnostics[:8]
                diagnostic_text = ", ".join(reason[:320] for reason in displayed)
                if len(diagnostics) > len(displayed):
                    diagnostic_text += f", plus {len(diagnostics) - len(displayed)} more"
                message += f" Guard details: {diagnostic_text}."
            message += " Fix the grounding issue and queue again."
            raise ValueError(message)
        candidate_prompt = str(prompt or "")
        blocked_reasons = _reason_strings(metadata.get("blocked_reasons"))
        policy = (
            "warn_and_continue"
            if str(validation_policy or "").strip().casefold() == "warn_and_continue"
            else "strict"
        )
        h3_warning_override = bool(
            policy == "warn_and_continue"
            and candidate_prompt.strip()
            and blocked_reasons
            and all(reason.startswith("minimax_h3_") for reason in blocked_reasons)
        )
        if not bool(ready_for_generation) and not h3_warning_override:
            details = " ".join(_friendly_generation_block_reasons(blocked_reasons, metadata))
            message = "DiffusionGemma blocked generation because the prompt did not pass validation."
            if details:
                message += f" Fix these issues and queue again: {details}"
            raise ValueError(message)
        if not candidate_prompt.strip():
            raise ValueError("DiffusionGemma blocked generation because the validated prompt is empty.")
        if h3_warning_override:
            _dg_log(
                "MiniMax-H3 validation warning override passed a non-empty candidate. reasons=%s",
                ",".join(blocked_reasons),
            )
        return (candidate_prompt,)


class DiffusionGemmaBranchGenerationGate:
    """Fail closed for one branch without aborting independent Comfy outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return copy.deepcopy(DiffusionGemmaGenerationGate.INPUT_TYPES())

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("prompt", "status", "ready")
    FUNCTION = "gate"
    CATEGORY = CATEGORY

    def gate(
        self,
        prompt: str,
        ready_for_generation: bool,
        metadata_json: str = "",
        validation_policy: str = "strict",
    ):
        try:
            validated_prompt = DiffusionGemmaGenerationGate().gate(
                prompt,
                ready_for_generation,
                metadata_json,
                validation_policy,
            )[0]
        except ValueError as exc:
            # A message-less ExecutionBlocker prevents only consumers of the
            # prompt output from running.  The status output remains available,
            # and unrelated comparison branches continue normally.
            status = str(exc)
            status += (
                " This branch produced no media output even if ComfyUI reports the "
                "overall queue as completed."
                " If this unchanged branch should retry, select refresh once on "
                "its CoT Generator, queue it, then return the setting to reuse."
            )
            return (ExecutionBlocker(None), status, False)
        if (
            not bool(ready_for_generation)
            and str(validation_policy or "").strip().casefold() == "warn_and_continue"
        ):
            status = (
                "DiffusionGemma branch continued with MiniMax-H3 validation warnings "
                "under the selected policy."
            )
        else:
            status = "DiffusionGemma branch is ready for generation."
        return (validated_prompt, status, True)


class DiffusionGemmaPromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_config": (MODEL_TYPE,),
                "user_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                "master_prompt": ("STRING", {"default": DEFAULT_MASTER_PROMPT, "multiline": True}),
                "target_profile": (["ltx", "ideogram4"], {"default": "ltx"}),
                "runtime_required": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If true, fail instead of using deterministic template output when DiffusionGemma is unavailable.",
                    },
                ),
                "max_new_tokens": ("INT", {"default": 768, "min": 64, "max": 4096}),
                "max_output_chars": ("INT", {"default": 4000, "min": 512, "max": 20000}),
                "audio_mode": (
                    ["auto_scene_audio", "explicit_sound_design", "visual_only"],
                    {
                        "default": "auto_scene_audio",
                        "tooltip": "Controls how DiffusionGemma writes audio into the LTX prompt before generation.",
                    },
                ),
                "audio_guidance": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional sound design instruction for DiffusionGemma to weave into ltx_prompt.",
                    },
                ),
                "target_duration_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 600.0,
                        "step": 0.1,
                        "tooltip": "Optional target duration for the direct LTX prompt. Video media duration wins only when this is 0.",
                    },
                ),
                "ltx_style": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional LTX style phrase for the Prompt Builder to weave into ltx_prompt.",
                    },
                ),
                "ideogram_aspect_ratio": (
                    ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
                    {"default": "1:1"},
                ),
                "ideogram_render_style": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional Ideogram4 visual style or medium.",
                    },
                ),
                "ideogram_exact_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional exact text that Ideogram4 should render verbatim.",
                    },
                ),
                "ideogram_json_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If true, ideogram_prompt is returned as structured Ideogram4 JSON text.",
                    },
                ),
                "creativity_mode": (
                    ["faithful", "editorial", "cinematic", "concept_art", "wild"],
                    {
                        "default": "editorial",
                        "tooltip": "Controls how boldly DiffusionGemma adds art direction while preserving the requested subject and exact text.",
                    },
                ),
                "creative_strength": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.5,
                        "step": 0.05,
                        "tooltip": "Amount of added detail. 0 is faithful; 0.6 editorial; 1.0+ more concept-art/wild.",
                    },
                ),
                "negative_prompt_mode": (
                    ["auto", "empty", "custom"],
                    {
                        "default": "auto",
                        "tooltip": "auto lets the model emit useful negatives; empty forces none; custom returns your supplied negative prompt exactly.",
                    },
                ),
                "negative_prompt_guidance": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Used only when negative_prompt_mode is custom, or as guidance for auto negatives.",
                    },
                ),
            },
            "optional": {
                "media_context": (MEDIA_TYPE,),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "ltx_prompt",
        "ideogram_prompt",
        "negative_prompt",
        "metadata_json",
        "scene_segments_json",
        "local_prompts",
        "segment_lengths",
    )
    FUNCTION = "build"
    CATEGORY = COMPATIBILITY_CATEGORY

    def build(
        self,
        model_config: RuntimeConfig,
        user_prompt: str,
        master_prompt: str,
        target_profile: str,
        runtime_required: bool,
        max_new_tokens: int,
        max_output_chars: int,
        audio_mode: str = "auto_scene_audio",
        audio_guidance: str = "",
        target_duration_seconds: float = 0.0,
        ltx_style: str = "",
        ideogram_aspect_ratio: str = "1:1",
        ideogram_render_style: str = "",
        ideogram_exact_text: str = "",
        ideogram_json_output: bool = True,
        creativity_mode: str = "editorial",
        creative_strength: float = 0.6,
        negative_prompt_mode: str = "auto",
        negative_prompt_guidance: str = "",
        media_context: MediaContext | None = None,
    ):
        if not isinstance(model_config, RuntimeConfig):
            raise TypeError("model_config must come from DiffusionGemma Model Loader.")

        context = _gemma_context_from_media(user_prompt, media_context)
        target_config_obj = _make_target_profile_config(
            target_profile,
            audio_mode,
            audio_guidance,
            target_duration_seconds,
            ltx_style,
            ideogram_aspect_ratio,
            ideogram_render_style,
            ideogram_exact_text,
            ideogram_json_output,
            negative_prompt_mode,
            negative_prompt_guidance,
        )
        packet, raw_output, reasoning_text, fallback_reason, parse_valid, salvage_warning, metadata = _run_generation_packet(
            model_config,
            context,
            target_config_obj,
            master_prompt,
            bool(runtime_required),
            int(max_new_tokens),
            int(max_output_chars),
            creativity_mode,
            creative_strength,
            "off",
        )
        (
            ltx_prompt,
            ideogram_prompt,
            negative_prompt,
            scene_segments_json,
            local_prompts,
            segment_lengths,
            _aspect_ratio,
            _packet_metadata,
            _minimax_h3_prompt,
        ) = _packet_to_prompt_outputs(
            packet,
            context,
            target_config_obj,
            int(max_output_chars),
            _normalize_creativity_mode(creativity_mode),
            _normalize_creative_strength(creative_strength),
        )
        metadata.update(
            {
                "raw_output_available": bool(raw_output),
                "reasoning_available": bool(reasoning_text),
                "used_template_fallback": bool(fallback_reason),
                "json_parse_valid": bool(parse_valid),
                "salvage_warning": salvage_warning,
            }
        )
        if not bool(metadata.get("ready_for_generation")):
            ltx_prompt = ""
            ideogram_prompt = ""
            negative_prompt = ""
            scene_segments_json = "[]"
            local_prompts = ""
            segment_lengths = ""
        return (
            ltx_prompt,
            ideogram_prompt,
            negative_prompt,
            _json_dumps(metadata),
            scene_segments_json,
            local_prompts,
            segment_lengths,
        )

        target_profile = _normalize_target_profile(target_profile)
        media_metadata = _metadata_from_media(media_context)
        duration = _safe_float(media_metadata.get("duration_seconds"), 0.0)
        audio_mode = _normalize_audio_mode(audio_mode)
        audio_guidance = _audio_guidance_text(audio_guidance)
        creativity_mode = _normalize_creativity_mode(creativity_mode)
        creative_strength = _normalize_creative_strength(creative_strength)
        negative_prompt_mode = _normalize_negative_prompt_mode(negative_prompt_mode)
        negative_prompt_guidance = _negative_prompt_guidance_text(negative_prompt_guidance)
        effective_duration = _effective_duration_seconds(media_metadata, target_duration_seconds)
        model_prompt = _build_model_prompt(
            user_prompt,
            master_prompt,
            target_profile,
            media_metadata,
            audio_mode,
            audio_guidance,
            target_duration_seconds,
            ltx_style,
            ideogram_aspect_ratio,
            ideogram_render_style,
            ideogram_exact_text,
            ideogram_json_output,
            creativity_mode,
            creative_strength,
            negative_prompt_mode,
            negative_prompt_guidance,
        )
        raw_output = ""
        fallback_reason = ""
        packet: dict[str, Any]

        try:
            if model_config.backend == "template":
                raise RuntimeError("Template backend selected.")
            raw_output = _run_backend(model_config, model_prompt, media_context, int(max_new_tokens))
            parsed = _extract_json_object(raw_output)
            if parsed is None:
                salvaged = _packet_from_plain_text_output(
                    raw_output,
                    user_prompt,
                    target_profile,
                    media_metadata,
                    duration,
                    audio_mode,
                    audio_guidance,
                    target_duration_seconds,
                    ltx_style,
                    ideogram_aspect_ratio,
                    ideogram_render_style,
                    ideogram_exact_text,
                    ideogram_json_output,
                    creativity_mode,
                    creative_strength,
                    negative_prompt_mode,
                    negative_prompt_guidance,
                    int(max_output_chars),
                )
                if salvaged is None:
                    raise RuntimeError("Model output did not contain valid JSON or usable plain text.")
                packet = salvaged
            else:
                packet = parsed
        except Exception as exc:
            fallback_reason = str(exc)
            if model_config.backend == "transformers_inprocess":
                _release_transformers_runtime()
            if runtime_required:
                raise RuntimeError(f"DiffusionGemma runtime failed and runtime_required is true: {exc}") from exc
            packet = _template_packet(
                user_prompt,
                target_profile,
                media_metadata,
                duration,
                audio_mode,
                audio_guidance,
                target_duration_seconds,
                ltx_style,
                ideogram_aspect_ratio,
                ideogram_render_style,
                ideogram_exact_text,
                ideogram_json_output,
                creativity_mode,
                creative_strength,
                negative_prompt_mode,
                negative_prompt_guidance,
            )
            packet.setdefault("metadata", {})["fallback_reason"] = fallback_reason
        finally:
            _maybe_unload(model_config)

        prompt_fallback = _clean_prompt_request_text(user_prompt)
        ltx_prompt = _prompt_field_to_text(packet.get("ltx_prompt", ""), prompt_fallback, int(max_output_chars))
        ideogram_value = packet.get("ideogram_prompt", "")
        if not ideogram_value and {"high_level_description", "compositional_deconstruction"}.issubset(packet.keys()):
            ideogram_value = packet
        ideogram_prompt = _ideogram_prompt_value_to_text(
            ideogram_value,
            prompt_fallback,
            ideogram_aspect_ratio,
            ideogram_render_style,
            ideogram_exact_text,
            bool(ideogram_json_output),
            int(max_output_chars),
            creativity_mode,
            creative_strength,
        )
        negative_prompt = _apply_negative_prompt_policy(
            packet.get("negative_prompt", ""),
            negative_prompt_mode,
            negative_prompt_guidance,
        )
        scene_segments = packet.get("scene_segments")
        if not isinstance(scene_segments, list) or not scene_segments:
            scene_segments = _split_segments(ltx_prompt, effective_duration or duration)
        local_prompts, segment_lengths = _segments_to_director_strings(scene_segments)
        metadata = packet.get("metadata") if isinstance(packet.get("metadata"), dict) else {}
        metadata.update(
            {
                "backend": model_config.backend,
                "runtime_ready": bool(model_config.status.get("ready")),
                "raw_output_available": bool(raw_output),
                "used_template_fallback": bool(fallback_reason),
                "media": media_metadata,
                "audio_mode": audio_mode,
                "audio_guidance": audio_guidance,
                "target_duration_seconds": effective_duration,
                "ltx_style": _clean_control_text(ltx_style, 240),
                "ideogram_aspect_ratio": ideogram_aspect_ratio,
                "ideogram_render_style": _clean_control_text(ideogram_render_style, 240),
                "ideogram_exact_text": _sanitize_prompt_text(
                    ideogram_exact_text,
                    "",
                    1000,
                    strip_thinking=True,
                    strip_markdown=True,
                ),
                "ideogram_json_output": bool(ideogram_json_output),
                "creativity_mode": creativity_mode,
                "creative_strength": creative_strength,
                "negative_prompt_mode": negative_prompt_mode,
                "negative_prompt_guidance": negative_prompt_guidance,
            }
        )
        return (
            ltx_prompt,
            ideogram_prompt,
            negative_prompt,
            _json_dumps(metadata),
            _json_dumps(scene_segments),
            local_prompts,
            segment_lengths,
        )


class DiffusionGemmaSimplePromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_config": (MODEL_TYPE,),
                "user_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                "target_profile": (["ltx", "ideogram4"], {"default": "ltx"}),
                "audio_mode": (
                    ["auto_scene_audio", "explicit_sound_design", "visual_only"],
                    {
                        "default": "auto_scene_audio",
                        "tooltip": "Controls whether LTX prompts include woven-in sound design.",
                    },
                ),
                "audio_guidance": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional sound design instruction for DiffusionGemma to weave into ltx_prompt.",
                    },
                ),
                "target_duration_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 600.0,
                        "step": 0.1,
                        "tooltip": "Optional target duration for the direct LTX prompt. Video media duration wins only when this is 0.",
                    },
                ),
                "ltx_style": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional LTX style phrase for the Prompt Builder to weave into ltx_prompt.",
                    },
                ),
                "ideogram_aspect_ratio": (
                    ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
                    {"default": "1:1"},
                ),
                "ideogram_render_style": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional Ideogram4 visual style or medium.",
                    },
                ),
                "ideogram_exact_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional exact text that Ideogram4 should render verbatim.",
                    },
                ),
                "ideogram_json_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If true, ideogram_prompt is returned as structured Ideogram4 JSON text.",
                    },
                ),
                "creativity_mode": (
                    ["faithful", "editorial", "cinematic", "concept_art", "wild"],
                    {
                        "default": "editorial",
                        "tooltip": "Controls how boldly DiffusionGemma adds art direction while preserving the requested subject and exact text.",
                    },
                ),
                "creative_strength": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.5,
                        "step": 0.05,
                        "tooltip": "Amount of added detail. 0 is faithful; 0.6 editorial; 1.0+ more concept-art/wild.",
                    },
                ),
                "negative_prompt_mode": (
                    ["auto", "empty", "custom"],
                    {
                        "default": "auto",
                        "tooltip": "auto lets the model emit useful negatives; empty forces none; custom returns your supplied negative prompt exactly.",
                    },
                ),
                "negative_prompt_guidance": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Used only when negative_prompt_mode is custom, or as guidance for auto negatives.",
                    },
                ),
            },
            "optional": {
                "media_context": (MEDIA_TYPE,),
            },
        }

    RETURN_TYPES = DiffusionGemmaPromptBuilder.RETURN_TYPES
    RETURN_NAMES = DiffusionGemmaPromptBuilder.RETURN_NAMES
    FUNCTION = "build"
    CATEGORY = COMPATIBILITY_CATEGORY

    def build(
        self,
        model_config: RuntimeConfig,
        user_prompt: str,
        target_profile: str,
        audio_mode: str = "auto_scene_audio",
        audio_guidance: str = "",
        target_duration_seconds: float = 0.0,
        ltx_style: str = "",
        ideogram_aspect_ratio: str = "1:1",
        ideogram_render_style: str = "",
        ideogram_exact_text: str = "",
        ideogram_json_output: bool = True,
        creativity_mode: str = "editorial",
        creative_strength: float = 0.6,
        negative_prompt_mode: str = "auto",
        negative_prompt_guidance: str = "",
        media_context: MediaContext | None = None,
    ):
        return DiffusionGemmaPromptBuilder().build(
            model_config,
            user_prompt,
            DEFAULT_MASTER_PROMPT,
            target_profile,
            False,
            768,
            8000,
            audio_mode,
            audio_guidance,
            target_duration_seconds,
            ltx_style,
            ideogram_aspect_ratio,
            ideogram_render_style,
            ideogram_exact_text,
            ideogram_json_output,
            creativity_mode,
            creative_strength,
            negative_prompt_mode,
            negative_prompt_guidance,
            media_context,
        )


class LTX2PromptAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                "duration_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1}),
                "style": ("STRING", {"default": "", "multiline": False}),
                "include_audio": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If true, the adapter may add a final scene-specific audio cue when one is missing. If false, it leaves audio untouched; use Prompt Builder audio_mode=visual_only to request no audio upstream.",
                    },
                ),
                "max_segments": ("INT", {"default": 6, "min": 1, "max": 24}),
                "audio_guidance": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional final LTX sound design cue. For best results, set audio on DiffusionGemma Prompt Builder so the model writes it upstream.",
                    },
                ),
            },
            "optional": {
                "metadata_json": ("STRING", {"default": "", "multiline": True}),
                "scene_segments_json": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("ltx_prompt", "local_prompts", "segment_lengths", "metadata_json")
    FUNCTION = "adapt"
    CATEGORY = OPTIONAL_CATEGORY

    def adapt(
        self,
        base_prompt: str,
        duration_seconds: float,
        style: str,
        include_audio: bool,
        max_segments: int,
        audio_guidance: str = "",
        metadata_json: str = "",
        scene_segments_json: str = "",
    ):
        metadata = _extract_json_object(metadata_json) or {}
        media = metadata.get("media") if isinstance(metadata.get("media"), dict) else {}
        effective_duration = float(duration_seconds) if float(duration_seconds) > 0 else _safe_float(media.get("duration_seconds"), 0.0)
        audio_guidance = _audio_guidance_text(audio_guidance)
        source_prompt = _single_paragraph(base_prompt)
        source_segments = _load_scene_segments(scene_segments_json)
        if not source_prompt and source_segments:
            source_prompt = _single_paragraph(
                " ".join(
                    str(segment.get("prompt", "")) if isinstance(segment, dict) else str(segment)
                    for segment in source_segments
                )
            )
        prompt = _apply_ltx2_controls(source_prompt, effective_duration, style, bool(include_audio), audio_guidance)
        if not prompt:
            metadata.update(
                {
                    "adapter": "ltx",
                    "duration_seconds": effective_duration,
                    "include_audio": bool(include_audio),
                    "audio_guidance": audio_guidance,
                    "max_segments": int(max_segments),
                    "segment_count": 0,
                    "style": _sanitize_prompt_text(style, "", 300, strip_thinking=True, strip_markdown=True).rstrip("."),
                    "warning": "LTX adapter received no base_prompt or scene_segments_json, so it returned an empty prompt instead of inventing one.",
                }
            )
            return ("", "", "", _json_dumps(metadata))

        segments = _normalize_segments(source_segments, source_prompt, effective_duration, int(max_segments))
        segment_style = _sanitize_prompt_text(style, "", 300, strip_thinking=True, strip_markdown=True).rstrip(".")
        if segment_style:
            style_prefix = f"Style: {segment_style}."
            for segment in segments:
                segment_prompt = str(segment.get("prompt", ""))
                if not segment_prompt.lower().startswith(style_prefix.lower()):
                    segment["prompt"] = _single_paragraph(f"{style_prefix} {segment_prompt}")
        local_prompts, segment_lengths = _segments_to_director_strings(segments)
        metadata.update(
            {
                "adapter": "ltx",
                "duration_seconds": effective_duration,
                "include_audio": bool(include_audio),
                "audio_guidance": audio_guidance,
                "max_segments": int(max_segments),
                "segment_count": len(segments),
                "style": segment_style,
            }
        )
        return (prompt, local_prompts, segment_lengths, _json_dumps(metadata))


class Ideogram4PromptAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                "aspect_ratio": (
                    ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
                    {"default": "1:1"},
                ),
                "render_style": ("STRING", {"default": "", "multiline": False}),
                "exact_text": ("STRING", {"default": "", "multiline": True}),
                "json_style_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "metadata_json": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("ideogram_prompt", "metadata_json")
    FUNCTION = "adapt"
    CATEGORY = OPTIONAL_CATEGORY

    def adapt(
        self,
        base_prompt: str,
        aspect_ratio: str,
        render_style: str,
        exact_text: str,
        json_style_output: bool,
        metadata_json: str = "",
    ):
        prompt = _single_paragraph(base_prompt)
        metadata = _extract_json_object(metadata_json) or {}
        metadata["adapter"] = "ideogram4"
        metadata["aspect_ratio"] = aspect_ratio
        metadata["exact_text"] = exact_text.strip()
        metadata["json_style_output"] = bool(json_style_output)
        return (
            _ideogram_prompt_value_to_text(
                prompt,
                prompt,
                aspect_ratio,
                render_style,
                exact_text.strip(),
                bool(json_style_output),
                20000,
            ),
            _json_dumps(metadata),
        )


class PromptSanitizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "fallback_prompt": ("STRING", {"default": "", "multiline": True}),
                "mode": (["plain_prompt", "json", "ltx", "ideogram4"], {"default": "plain_prompt"}),
                "max_chars": ("INT", {"default": 4000, "min": 1, "max": 20000}),
                "strip_thinking": ("BOOLEAN", {"default": True}),
                "strip_markdown": ("BOOLEAN", {"default": True}),
                "require_json": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("sanitized_prompt", "is_valid", "metadata_json")
    FUNCTION = "sanitize"
    CATEGORY = OPTIONAL_CATEGORY

    def sanitize(
        self,
        prompt: str,
        fallback_prompt: str,
        mode: str,
        max_chars: int,
        strip_thinking: bool,
        strip_markdown: bool,
        require_json: bool,
    ):
        value = _sanitize_prompt_text(prompt, fallback_prompt, int(max_chars), bool(strip_thinking), bool(strip_markdown))
        valid = True
        metadata: dict[str, Any] = {"mode": mode, "used_fallback": False}
        if require_json or mode == "json":
            parsed = _extract_json_object(value)
            if parsed is None:
                value = _sanitize_prompt_text(fallback_prompt, "", int(max_chars), bool(strip_thinking), bool(strip_markdown))
                valid = False
                metadata["used_fallback"] = True
                metadata["error"] = "Prompt did not contain valid JSON."
            else:
                value = _json_dumps(parsed)
        if mode in {"ltx", "ltx2", "ltx2_3"}:
            value = _single_paragraph(value)
        if mode == "ideogram4":
            value = _single_paragraph(value)
        return (value, valid, _json_dumps(metadata))


class TextOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "optional": {
                "label": ("STRING", {"default": "DiffusionGemma output", "multiline": False}),
                "save_to_file": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "diffusiongemma_prompt", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "saved_path")
    FUNCTION = "show"
    OUTPUT_NODE = True
    CATEGORY = OPTIONAL_CATEGORY

    def show(
        self,
        text: str,
        label: str = "DiffusionGemma output",
        save_to_file: bool = True,
        filename_prefix: str = "diffusiongemma_prompt",
    ):
        title = _sanitize_prompt_text(label, "DiffusionGemma output", 160, strip_thinking=False, strip_markdown=True)
        value = str(text or "")
        saved_path = ""
        if save_to_file:
            try:
                import folder_paths

                output_dir = Path(folder_paths.get_output_directory()) / "diffusiongemma"
                output_dir.mkdir(parents=True, exist_ok=True)
                safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename_prefix.strip() or "diffusiongemma_prompt")
                counter = 1
                while True:
                    candidate = output_dir / f"{safe_prefix}_{counter:05}.txt"
                    if not candidate.exists():
                        candidate.write_text(value, encoding="utf-8")
                        saved_path = str(candidate)
                        break
                    counter += 1
            except Exception as exc:
                saved_path = f"Save failed: {exc}"
        ui_text = f"{title}\n\n{value}".strip()
        if saved_path:
            ui_text = f"{ui_text}\n\nSaved: {saved_path}"
        return {"ui": {"text": [ui_text]}, "result": (value, saved_path)}


NODE_CLASS_MAPPINGS = {
    "DiffusionGemmaModelLoader": DiffusionGemmaModelLoader,
    "DiffusionGemmaContextHub": DiffusionGemmaContextHub,
    "DiffusionGemmaH3ReferencePolicy": DiffusionGemmaH3ReferencePolicy,
    "DiffusionGemmaH3ReferenceContext": DiffusionGemmaH3ReferenceContext,
    "DiffusionGemmaReferencePrep": DiffusionGemmaReferencePrep,
    "DiffusionGemmaH3ReferencePairPrep": DiffusionGemmaH3ReferencePairPrep,
    "DiffusionGemmaTargetProfile": DiffusionGemmaTargetProfile,
    "DiffusionGemmaLTX25TargetProfile": DiffusionGemmaLTX25TargetProfile,
    "DiffusionGemmaMiniMaxH3TargetProfile": DiffusionGemmaMiniMaxH3TargetProfile,
    "DiffusionGemmaIdeogram4TargetProfile": DiffusionGemmaIdeogram4TargetProfile,
    "DiffusionGemmaGroundingGuardSettings": DiffusionGemmaGroundingGuardSettings,
    "DiffusionGemmaCoTGenerator": DiffusionGemmaCoTGenerator,
    "DiffusionGemmaSplatStagePlanner": DiffusionGemmaSplatStagePlanner,
    "DiffusionGemmaSplatStageEfficientPlanner": DiffusionGemmaSplatStageEfficientPlanner,
    "DiffusionGemmaJSONSplitter": DiffusionGemmaJSONSplitter,
    "DiffusionGemmaGenerationGate": DiffusionGemmaGenerationGate,
    "DiffusionGemmaBranchGenerationGate": DiffusionGemmaBranchGenerationGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionGemmaModelLoader": "DiffusionGemma Model Loader (Advanced)",
    "DiffusionGemmaContextHub": "DiffusionGemma Context Hub",
    "DiffusionGemmaH3ReferencePolicy": "DiffusionGemma H3 Reference Policy",
    "DiffusionGemmaH3ReferenceContext": "DiffusionGemma H3 Reference Context",
    "DiffusionGemmaReferencePrep": "DiffusionGemma Reference Prep",
    "DiffusionGemmaH3ReferencePairPrep": "DiffusionGemma H3 Reference Pair Prep",
    "DiffusionGemmaTargetProfile": "DiffusionGemma Target Profile (Legacy — all models)",
    "DiffusionGemmaLTX25TargetProfile": "DiffusionGemma LTX-2.5 Target Profile",
    "DiffusionGemmaMiniMaxH3TargetProfile": "DiffusionGemma MiniMax-H3 Target Profile",
    "DiffusionGemmaIdeogram4TargetProfile": "DiffusionGemma Ideogram 4 Target Profile",
    "DiffusionGemmaGroundingGuardSettings": "DiffusionGemma Grounding Guard Settings",
    "DiffusionGemmaCoTGenerator": "DiffusionGemma CoT Generator",
    "DiffusionGemmaSplatStagePlanner": "DiffusionGemma SplatStage Planner (Experimental)",
    "DiffusionGemmaSplatStageEfficientPlanner": "DiffusionGemma SplatStage Efficient Planner (Experimental)",
    "DiffusionGemmaJSONSplitter": "DiffusionGemma JSON Splitter",
    "DiffusionGemmaGenerationGate": "DiffusionGemma Generation Gate",
    "DiffusionGemmaBranchGenerationGate": "DiffusionGemma Branch Generation Gate",
}
