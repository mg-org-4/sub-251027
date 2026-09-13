# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fail-safe vocal activity, transcription, and authored-lyric alignment.

This node deliberately treats timing as optional evidence.  A failed separator,
missing transcript, weak lyric match, or model exception returns a structured
``natural_fallback`` report; it never blocks the downstream video workflow.
Whisper is only evaluated for the explicit Lyrics + lip sync mode.
"""

from __future__ import annotations

from collections import OrderedDict
from difflib import SequenceMatcher
import hashlib
import inspect
import json
import math
import re
import struct
import threading
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as torch_functional


CATEGORY = "prompt/diffusiongemma/audio-production"
TIMED_LYRICS_REPORT_SCHEMA = "diffusiongemma.timed_lyrics_report"
TIMED_LYRICS_REPORT_VERSION = 1
PERFORMANCE_MODES = (
    "Natural / audio-led sync",
    "Dance / music sync",
    "Lyrics + lip sync",
)
LANGUAGES = (
    "en",
    "auto",
    "fr",
    "es",
    "de",
    "it",
    "pt",
    "nl",
    "ru",
    "zh",
    "ja",
    "ko",
)
ACTIVITY_WINDOW_SECONDS = 0.5
MAX_WHISPER_CHUNK_SECONDS = 15.0
WHISPER_SAMPLE_RATE = 16_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TIMESTAMP_TOKEN_RE = re.compile(r"^<\|([0-9]+(?:\.[0-9]+)?)\|>$")
_SECTION_HEADER_RE = re.compile(r"^\s*\[[^\]]{1,80}\]\s*$")
_PLAIN_SECTION_RE = re.compile(
    r"^\s*(?:intro|outro|verse|pre[ -]?chorus|chorus|refrain|hook|bridge|"
    r"interlude|break|breakdown|instrumental|solo|drop|post[ -]?chorus)"
    r"(?:\s+[0-9ivx]+)?\s*:?\s*$",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)?", re.UNICODE)

_CACHE_MAX_ENTRIES = 8
_ANALYSIS_CACHE: OrderedDict[tuple[Any, ...], tuple[str, str, bool, str]] = OrderedDict()
_CACHE_LOCK = threading.RLock()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _lyrics_sha256(lyrics: Any) -> str:
    return hashlib.sha256(str(lyrics or "").encode("utf-8")).hexdigest()


def _audio_parts(audio: Any, label: str) -> tuple[torch.Tensor, int]:
    if not isinstance(audio, Mapping):
        raise ValueError(f"{label} must be a ComfyUI AUDIO mapping.")
    waveform = audio.get("waveform")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError(f"{label} waveform must have shape [batch, channels, samples].")
    if any(int(size) <= 0 for size in waveform.shape):
        raise ValueError(f"{label} waveform dimensions must be non-empty.")
    try:
        sample_rate = int(audio.get("sample_rate", 0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} sample rate must be a positive integer.") from exc
    if sample_rate <= 0:
        raise ValueError(f"{label} sample rate must be a positive integer.")
    return waveform, sample_rate


def _waveform_sha256(audio: Any) -> str:
    """Match the exact decoded-waveform identity used by audio production."""

    waveform, sample_rate = _audio_parts(audio, "final_audio")
    value = waveform.detach()
    if value.device.type != "cpu":
        value = value.to("cpu")
    value = value.contiguous()
    payload = memoryview(value.view(torch.uint8).numpy()).cast("B")
    digest = hashlib.sha256()
    digest.update(b"diffusiongemma-audio-waveform@1\0")
    digest.update(struct.pack("<Q", sample_rate))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(struct.pack("<I", value.ndim))
    for size in value.shape:
        digest.update(struct.pack("<Q", int(size)))
    for offset in range(0, len(payload), 1024 * 1024):
        digest.update(payload[offset : offset + 1024 * 1024])
    return digest.hexdigest()


def _positive_finite(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(number) or number < 0.0 or (not allow_zero and number <= 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a finite {qualifier} number.")
    return number


def _normalize_mode(value: Any) -> str:
    key = re.sub(r"\s+", " ", str(value or "").strip()).casefold()
    aliases = {
        "natural": "Natural / audio-led sync",
        "natural / audio led sync": "Natural / audio-led sync",
        "natural / audio-led sync": "Natural / audio-led sync",
        "audio led": "Natural / audio-led sync",
        "audio-led": "Natural / audio-led sync",
        "dance": "Dance / music sync",
        "dance / music sync": "Dance / music sync",
        "lyrics": "Lyrics + lip sync",
        "lyrics + lip sync": "Lyrics + lip sync",
    }
    if key not in aliases:
        raise ValueError("performance_mode must be Natural, Dance, or Lyrics + lip sync.")
    return aliases[key]


def _model_from_pipeline(pipeline: Any) -> Any:
    if isinstance(pipeline, Mapping):
        return pipeline.get("model")
    return getattr(pipeline, "model", None)


def _processor_from_pipeline(pipeline: Any) -> Any:
    if isinstance(pipeline, Mapping):
        return pipeline.get("processor")
    return getattr(pipeline, "processor", None)


def _model_id(pipeline: Any) -> str:
    if isinstance(pipeline, Mapping):
        explicit = pipeline.get("model_id") or pipeline.get("name")
        if explicit:
            return str(explicit)
    explicit = getattr(pipeline, "model_id", None)
    if explicit:
        return str(explicit)
    model = _model_from_pipeline(pipeline)
    config = getattr(model, "config", None)
    for value in (
        getattr(config, "_name_or_path", None),
        getattr(config, "name_or_path", None),
        getattr(model, "name_or_path", None),
    ):
        if value:
            return str(value)
    if model is not None:
        return f"{type(model).__module__}.{type(model).__qualname__}"
    return "unknown-whisper-pipeline"


def _mono_cpu(audio: Any, label: str) -> tuple[torch.Tensor, int, float]:
    waveform, sample_rate = _audio_parts(audio, label)
    value = waveform.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains NaN or infinite samples.")
    mono = value.mean(dim=(0, 1)).contiguous()
    return mono, sample_rate, mono.numel() / float(sample_rate)


def _resample_1d(value: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return value.contiguous()
    target_samples = max(1, int(round(value.numel() * target_rate / float(source_rate))))
    return torch_functional.interpolate(
        value.reshape(1, 1, -1),
        size=target_samples,
        mode="linear",
        align_corners=False,
    ).reshape(-1).contiguous()


def _stem_aligned_to_mix(
    vocal_stem: Any,
    final_audio: Any,
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]:
    mix, mix_rate, mix_duration = _mono_cpu(final_audio, "final_audio")
    stem, stem_rate, stem_duration = _mono_cpu(vocal_stem, "vocal_stem")
    duration_error = abs(stem_duration - mix_duration)
    tolerance = max(0.05, 1.0 / mix_rate)
    if duration_error > tolerance:
        raise ValueError(
            "vocal_stem duration is not aligned to final_audio "
            f"({stem_duration:.3f}s versus {mix_duration:.3f}s)."
        )
    stem = _resample_1d(stem, stem_rate, mix_rate)
    if abs(stem.numel() - mix.numel()) > max(1, int(round(tolerance * mix_rate))):
        raise ValueError("vocal_stem sample count cannot be aligned to final_audio.")
    if stem.numel() < mix.numel():
        stem = torch_functional.pad(stem, (0, mix.numel() - stem.numel()))
    else:
        stem = stem[: mix.numel()]
    return stem.contiguous(), mix, mix_rate, {
        "mix_sample_rate": mix_rate,
        "stem_source_sample_rate": stem_rate,
        "mix_duration_seconds": mix_duration,
        "stem_duration_seconds": stem_duration,
        "duration_error_seconds": duration_error,
    }


def _stem_full_mix_similarity(stem: torch.Tensor, mix: torch.Tensor) -> dict[str, Any]:
    # Bound the diagnostic calculation for long songs without changing its
    # deterministic coverage of the waveform.
    maximum_points = 1_000_000
    stride = max(1, int(math.ceil(max(stem.numel(), mix.numel()) / maximum_points)))
    left = stem[::stride].to(torch.float64)
    right = mix[::stride].to(torch.float64)
    mix_energy = float(torch.mean(right.square()).item())
    stem_energy = float(torch.mean(left.square()).item())
    if mix_energy <= 1.0e-14 and stem_energy <= 1.0e-14:
        return {
            "correlation": 1.0,
            "optimal_mix_gain": 1.0,
            "normalized_residual_rms": 0.0,
            "near_identical_full_mix": True,
        }
    denominator = float(torch.dot(right, right).item())
    gain = float(torch.dot(left, right).item()) / max(denominator, 1.0e-14)
    residual = left - gain * right
    residual_ratio = math.sqrt(float(torch.mean(residual.square()).item())) / max(
        math.sqrt(stem_energy), 1.0e-12
    )
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    corr_denominator = float(
        torch.linalg.vector_norm(centered_left) * torch.linalg.vector_norm(centered_right)
    )
    correlation = (
        float(torch.dot(centered_left, centered_right).item()) / corr_denominator
        if corr_denominator > 1.0e-14
        else 0.0
    )
    near_identical = correlation >= 0.999 and residual_ratio <= 0.025
    return {
        "correlation": round(correlation, 8),
        "optimal_mix_gain": round(gain, 8),
        "normalized_residual_rms": round(residual_ratio, 8),
        "near_identical_full_mix": near_identical,
    }


def _interval_complement(
    intervals: Sequence[Mapping[str, float]], duration: float
) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    cursor = 0.0
    for interval in intervals:
        start = max(cursor, float(interval["start_seconds"]))
        end = min(duration, float(interval["end_seconds"]))
        if start > cursor + 1.0e-6:
            result.append(
                {"start_seconds": round(cursor, 6), "end_seconds": round(start, 6)}
            )
        cursor = max(cursor, end)
    if cursor < duration - 1.0e-6:
        result.append(
            {"start_seconds": round(cursor, 6), "end_seconds": round(duration, 6)}
        )
    return result


def _vocal_activity(
    stem: torch.Tensor,
    mix: torch.Tensor,
    sample_rate: int,
    excerpt_start: float,
    excerpt_duration: float,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, Any]]:
    start_sample = int(round(excerpt_start * sample_rate))
    end_sample = int(round((excerpt_start + excerpt_duration) * sample_rate))
    excerpt_stem = stem[start_sample:end_sample]
    excerpt_mix = mix[start_sample:end_sample]
    expected_samples = int(round(excerpt_duration * sample_rate))
    if excerpt_stem.numel() + 1 < expected_samples or excerpt_mix.numel() + 1 < expected_samples:
        raise ValueError("The selected excerpt extends beyond final_audio or vocal_stem.")
    usable = min(excerpt_stem.numel(), excerpt_mix.numel())
    excerpt_stem = excerpt_stem[:usable]
    excerpt_mix = excerpt_mix[:usable]
    window_samples = max(1, int(round(ACTIVITY_WINDOW_SECONDS * sample_rate)))
    records: list[dict[str, float | bool]] = []
    for offset in range(0, usable, window_samples):
        stop = min(usable, offset + window_samples)
        stem_window = excerpt_stem[offset:stop]
        mix_window = excerpt_mix[offset:stop]
        stem_rms = math.sqrt(float(torch.mean(stem_window.square()).item()))
        mix_rms = math.sqrt(float(torch.mean(mix_window.square()).item()))
        records.append(
            {
                "start": offset / float(sample_rate),
                "end": stop / float(sample_rate),
                "stem_rms": stem_rms,
                "mix_rms": mix_rms,
                "ratio": stem_rms / max(mix_rms, 1.0e-8),
                "active": False,
            }
        )
    positive_rms = torch.tensor(
        [float(item["stem_rms"]) for item in records], dtype=torch.float64
    )
    p90 = float(torch.quantile(positive_rms, 0.90).item()) if records else 0.0
    absolute_threshold = max(3.0e-4, min(0.02, 0.12 * p90))
    ratio_threshold = 0.06
    for item in records:
        item["active"] = bool(
            float(item["stem_rms"]) >= absolute_threshold
            and float(item["ratio"]) >= ratio_threshold
        )

    # Smooth one-window separator dropouts, then discard isolated low-energy
    # blips.  A genuinely strong half-second phrase remains eligible.
    for index in range(1, len(records) - 1):
        if (
            not bool(records[index]["active"])
            and bool(records[index - 1]["active"])
            and bool(records[index + 1]["active"])
        ):
            records[index]["active"] = True
    for index, item in enumerate(records):
        if not bool(item["active"]):
            continue
        left = index > 0 and bool(records[index - 1]["active"])
        right = index + 1 < len(records) and bool(records[index + 1]["active"])
        if not left and not right and float(item["stem_rms"]) < max(0.002, 2.5 * absolute_threshold):
            item["active"] = False

    intervals: list[dict[str, float]] = []
    current_start: float | None = None
    current_end = 0.0
    for item in records:
        if bool(item["active"]):
            if current_start is None:
                current_start = float(item["start"])
            current_end = float(item["end"])
        elif current_start is not None:
            intervals.append(
                {
                    "start_seconds": round(current_start, 6),
                    "end_seconds": round(min(excerpt_duration, current_end), 6),
                }
            )
            current_start = None
    if current_start is not None:
        intervals.append(
            {
                "start_seconds": round(current_start, 6),
                "end_seconds": round(min(excerpt_duration, current_end), 6),
            }
        )
    active_seconds = sum(
        max(0.0, item["end_seconds"] - item["start_seconds"]) for item in intervals
    )
    diagnostics = {
        "window_seconds": ACTIVITY_WINDOW_SECONDS,
        "window_count": len(records),
        "absolute_stem_rms_threshold": absolute_threshold,
        "stem_to_mix_rms_ratio_threshold": ratio_threshold,
        "active_window_count": sum(bool(item["active"]) for item in records),
        "active_seconds": active_seconds,
        "active_coverage": active_seconds / max(excerpt_duration, 1.0e-9),
        "stem_rms_p90": p90,
    }
    return intervals, _interval_complement(intervals, excerpt_duration), diagnostics


def _authored_lines(lyrics: Any) -> list[str]:
    result: list[str] = []
    for raw_line in str(lyrics or "").splitlines():
        line = raw_line.strip()
        if not line or _SECTION_HEADER_RE.fullmatch(line) or _PLAIN_SECTION_RE.fullmatch(line):
            continue
        result.append(line)
    return result


def _normalized_words(text: Any) -> list[str]:
    return [
        match.group(0).replace("\u2019", "'").casefold()
        for match in _WORD_RE.finditer(str(text or ""))
    ]


def _transcript_similarity(left: Any, right: Any) -> float:
    left_words = _normalized_words(left)
    right_words = _normalized_words(right)
    if not left_words or not right_words:
        return 0.0
    left_text = " ".join(left_words)
    right_text = " ".join(right_words)
    if left_text == right_text:
        return 1.0
    sequence_score = SequenceMatcher(None, left_text, right_text).ratio()
    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for word in left_words:
        left_counts[word] = left_counts.get(word, 0) + 1
    for word in right_words:
        right_counts[word] = right_counts.get(word, 0) + 1
    overlap = sum(min(count, right_counts.get(word, 0)) for word, count in left_counts.items())
    precision = overlap / len(left_words)
    recall = overlap / len(right_words)
    word_f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    containment = min(len(left_words), len(right_words)) / max(len(left_words), len(right_words))
    if left_text in right_text or right_text in left_text:
        sequence_score = max(sequence_score, containment)
    return max(0.0, min(1.0, 0.55 * sequence_score + 0.45 * word_f1))


def _align_phrases(
    phrases: Sequence[Mapping[str, Any]],
    lines: Sequence[str],
    excerpt_start: float,
    song_duration: float,
    minimum_confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    minimum_line_index = 0
    candidate_evaluations = 0
    weakest = 1.0
    for phrase in phrases:
        transcript = str(phrase.get("text", "")).strip()
        if not _normalized_words(transcript):
            continue
        midpoint = excerpt_start + 0.5 * (
            float(phrase["start_seconds"]) + float(phrase["end_seconds"])
        )
        expected_position = max(0.0, min(1.0, midpoint / max(song_duration, 1.0e-9)))
        expected_index = expected_position * max(0, len(lines) - 1)
        best: tuple[float, float, int, int, list[str]] | None = None
        for start in range(minimum_line_index, len(lines)):
            for count in range(1, min(3, len(lines) - start) + 1):
                selected = list(lines[start : start + count])
                similarity = _transcript_similarity(transcript, " ".join(selected))
                candidate_center = start + 0.5 * (count - 1)
                position_score = 1.0 - abs(candidate_center - expected_index) / max(len(lines), 1)
                # Song position is deliberately weak: it resolves repeated choruses
                # but cannot rescue an uncertain lexical match.
                rank_score = similarity + 0.03 * max(0.0, position_score)
                candidate_evaluations += 1
                candidate = (rank_score, similarity, start, count, selected)
                if best is None or candidate[:4] > best[:4]:
                    best = candidate
        if best is None or best[1] < minimum_confidence:
            observed = 0.0 if best is None else best[1]
            raise ValueError(
                "alignment_confidence_below_threshold: "
                f"{observed:.3f} < {minimum_confidence:.3f} for {transcript!r}"
            )
        _, confidence, start, count, selected = best
        weakest = min(weakest, confidence)
        events.append(
            {
                "start_seconds": round(float(phrase["start_seconds"]), 6),
                "end_seconds": round(float(phrase["end_seconds"]), 6),
                "text": transcript,
                "transcript": transcript,
                "authored_lines": selected,
                "confidence": round(confidence, 6),
            }
        )
        # Whisper may split one authored line into several timestamp phrases.
        # Keep the lower bound non-decreasing (rather than strictly advancing)
        # so adjacent fragments can safely resolve to that same authored line.
        minimum_line_index = start
    if not events:
        raise ValueError("whisper_returned_no_timestamped_vocal_phrases")
    return events, {
        "event_count": len(events),
        "authored_line_count": len(lines),
        "candidate_evaluations": candidate_evaluations,
        "minimum_event_confidence": weakest,
        "monotonic_authored_line_alignment": True,
        "song_position_tie_break_weight": 0.03,
    }


def _timestamp_value(token: Any) -> float | None:
    match = _TIMESTAMP_TOKEN_RE.fullmatch(str(token or ""))
    if not match:
        return None
    try:
        value = float(match.group(1))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _tokens_to_text(tokens: Sequence[Any], tokenizer: Any = None) -> str:
    values = [str(token) for token in tokens if _timestamp_value(token) is None]
    values = [
        token
        for token in values
        if not (token.startswith("<|") and token.endswith("|>"))
    ]
    if tokenizer is not None:
        converter = getattr(tokenizer, "convert_tokens_to_string", None)
        if callable(converter):
            try:
                return str(converter(values)).strip()
            except (TypeError, ValueError, RuntimeError):
                pass
    text = "".join(values).replace("\u0120", " ").replace("\u2581", " ")
    if not text.strip():
        text = " ".join(values)
    return re.sub(r"\s+", " ", text).strip()


def _response_phrases(
    response: Any,
    chunk_duration: float,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    if isinstance(response, tuple):
        mappings = [item for item in response if isinstance(item, Mapping)]
        response = mappings[-1] if mappings else (response[0] if response else None)
    if not isinstance(response, Mapping):
        return []
    chunks = response.get("chunks")
    result: list[dict[str, Any]] = []
    if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes, bytearray)):
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                continue
            stamp = chunk.get("timestamp", chunk.get("timestamps"))
            if not isinstance(stamp, Sequence) or isinstance(stamp, (str, bytes)) or len(stamp) != 2:
                continue
            try:
                start, end = float(stamp[0]), float(stamp[1])
            except (TypeError, ValueError, OverflowError):
                continue
            text = str(chunk.get("text", "")).strip()
            if (
                text
                and math.isfinite(start)
                and math.isfinite(end)
                and start >= -0.05
                and end > start
                and start < chunk_duration + 0.05
            ):
                result.append(
                    {
                        "start_seconds": max(0.0, start),
                        "end_seconds": min(chunk_duration, end),
                        "text": text,
                    }
                )
        if result:
            return result
    tokens = response.get("tokens")
    if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes, bytearray)):
        return []
    markers = [
        (index, value)
        for index, token in enumerate(tokens)
        if (value := _timestamp_value(token)) is not None
    ]
    for marker_index, (position, start) in enumerate(markers):
        next_position = len(tokens)
        end = chunk_duration
        if marker_index + 1 < len(markers):
            next_position, end = markers[marker_index + 1]
        text = _tokens_to_text(tokens[position + 1 : next_position], tokenizer)
        if (
            text
            and start >= -0.05
            and end > start
            and start < chunk_duration + 0.05
        ):
            result.append(
                {
                    "start_seconds": max(0.0, start),
                    "end_seconds": min(chunk_duration, end),
                    "text": text,
                }
            )
    return result


def _call_with_supported_arguments(function: Callable[..., Any], **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(kwargs["audio"], kwargs.get("language", "en"), True)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    supported = (
        kwargs
        if accepts_kwargs
        else {name: value for name, value in kwargs.items() if name in signature.parameters}
    )
    return function(**supported)


def _custom_transcriber(pipeline: Any) -> Callable[..., Any] | None:
    if isinstance(pipeline, Mapping) and callable(pipeline.get("transcribe")):
        return pipeline["transcribe"]
    transcribe = getattr(pipeline, "transcribe", None)
    if callable(transcribe):
        return transcribe
    if callable(pipeline) and not isinstance(pipeline, Mapping):
        return pipeline
    return None


def _runtime_device(model: Any) -> Any:
    try:
        from comfy.model_management import get_torch_device

        return get_torch_device()
    except (ImportError, AttributeError, RuntimeError):
        return getattr(model, "device", torch.device("cpu"))


def _prepare_model(pipeline: Any) -> None:
    if _custom_transcriber(pipeline) is not None:
        return
    model = _model_from_pipeline(pipeline)
    processor = _processor_from_pipeline(pipeline)
    if model is None or processor is None:
        raise ValueError("whisper_pipeline must contain MTB processor and model values.")
    mover = getattr(model, "to", None)
    if callable(mover):
        mover(_runtime_device(model))
    evaluator = getattr(model, "eval", None)
    if callable(evaluator):
        evaluator()


def _offload_model(pipeline: Any) -> None:
    model = _model_from_pipeline(pipeline)
    mover = getattr(model, "to", None)
    if callable(mover):
        try:
            mover("cpu")
        except (RuntimeError, TypeError, ValueError):
            pass
    try:
        torch.cuda.empty_cache()
    except (RuntimeError, AttributeError):
        pass


def _direct_mtb_transcription(
    pipeline: Any,
    mono: torch.Tensor,
    sample_rate: int,
    language: str,
) -> Mapping[str, Any]:
    processor = _processor_from_pipeline(pipeline)
    model = _model_from_pipeline(pipeline)
    if processor is None or model is None:
        raise ValueError("whisper_pipeline must contain MTB processor and model values.")
    audio = _resample_1d(mono, sample_rate, WHISPER_SAMPLE_RATE)
    processed = processor(audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
    input_features = (
        processed.get("input_features")
        if isinstance(processed, Mapping)
        else getattr(processed, "input_features", None)
    )
    if not isinstance(input_features, torch.Tensor):
        raise ValueError("Whisper processor did not return input_features.")
    device = getattr(model, "device", torch.device("cpu"))
    model_dtype = getattr(model, "dtype", None)
    if not isinstance(model_dtype, torch.dtype):
        try:
            model_dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration, TypeError):
            model_dtype = None
    if (
        isinstance(model_dtype, torch.dtype)
        and bool(torch.is_floating_point(input_features))
        and (model_dtype.is_floating_point or model_dtype.is_complex)
    ):
        input_features = input_features.to(device=device, dtype=model_dtype)
    else:
        input_features = input_features.to(device)
    generate_kwargs: dict[str, Any] = {
        "task": "transcribe",
        "return_timestamps": True,
        "no_repeat_ngram_size": 3,
        "num_beams": 5,
        "length_penalty": 1.0,
    }
    if language != "auto":
        generate_kwargs["language"] = language
    attention_mask = (
        processed.get("attention_mask")
        if isinstance(processed, Mapping)
        else getattr(processed, "attention_mask", None)
    )
    if isinstance(attention_mask, torch.Tensor):
        generate_kwargs["attention_mask"] = attention_mask.to(device)
    max_length = getattr(getattr(model, "config", None), "max_length", None)
    if isinstance(max_length, int) and max_length > 0:
        generate_kwargs["max_length"] = max_length
    with torch.inference_mode():
        predicted_ids = model.generate(input_features, **generate_kwargs)
    row = predicted_ids[0]
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Whisper processor has no tokenizer.")
    tokens = tokenizer.convert_ids_to_tokens(row)
    decoded = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return {
        "text": str(decoded[0] if decoded else ""),
        "tokens": list(tokens),
        "language": language,
    }


def _transcribe_active_intervals(
    pipeline: Any,
    stem: torch.Tensor,
    sample_rate: int,
    excerpt_start: float,
    intervals: Sequence[Mapping[str, float]],
    language: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phrases: list[dict[str, Any]] = []
    chunk_count = 0
    transcriber = _custom_transcriber(pipeline)
    tokenizer = getattr(_processor_from_pipeline(pipeline), "tokenizer", None)
    for interval in intervals:
        interval_start = float(interval["start_seconds"])
        interval_end = float(interval["end_seconds"])
        chunk_start = interval_start
        while chunk_start < interval_end - 1.0e-6:
            chunk_end = min(interval_end, chunk_start + MAX_WHISPER_CHUNK_SECONDS)
            absolute_song_start = excerpt_start + chunk_start
            start_sample = int(round(absolute_song_start * sample_rate))
            end_sample = int(round((excerpt_start + chunk_end) * sample_rate))
            mono = stem[start_sample:end_sample].contiguous()
            duration = mono.numel() / float(sample_rate)
            if duration <= 0.0 or duration > MAX_WHISPER_CHUNK_SECONDS + 0.01:
                raise ValueError("invalid_whisper_chunk_duration")
            chunk_audio = {"waveform": mono.reshape(1, 1, -1), "sample_rate": sample_rate}
            if transcriber is not None:
                response = _call_with_supported_arguments(
                    transcriber,
                    audio=chunk_audio,
                    language=language,
                    return_timestamps=True,
                    chunk_offset_seconds=chunk_start,
                    absolute_song_start_seconds=absolute_song_start,
                )
            else:
                response = _direct_mtb_transcription(pipeline, mono, sample_rate, language)
            local_phrases = _response_phrases(response, duration, tokenizer)
            for phrase in local_phrases:
                start = chunk_start + float(phrase["start_seconds"])
                end = chunk_start + float(phrase["end_seconds"])
                if end > start and start < interval_end + 0.05:
                    phrases.append(
                        {
                            "start_seconds": max(interval_start, start),
                            "end_seconds": min(interval_end, end),
                            "text": str(phrase["text"]).strip(),
                        }
                    )
            chunk_count += 1
            chunk_start = chunk_end
    phrases.sort(key=lambda item: (float(item["start_seconds"]), float(item["end_seconds"])))
    previous_end = -math.inf
    for phrase in phrases:
        start = float(phrase["start_seconds"])
        end = float(phrase["end_seconds"])
        if start + 0.05 < previous_end or end <= start:
            raise ValueError("Whisper returned invalid or non-monotonic timestamp offsets.")
        previous_end = max(previous_end, end)
    return phrases, {
        "chunk_count": chunk_count,
        "maximum_chunk_seconds": MAX_WHISPER_CHUNK_SECONDS,
        "timestamped_phrase_count": len(phrases),
        "mtb_thirty_second_offset_logic_used": False,
        "chunk_offsets_added_explicitly": True,
    }


def _base_report(
    *,
    performance_mode: str,
    lyrics_hash: str,
    master_audio_sha256: str,
    song_duration: float,
    excerpt_start: float,
    excerpt_duration: float,
    language: str,
    minimum_confidence: float,
    model_id: str | None,
    analysis_status: str,
    timing_ready: bool,
) -> dict[str, Any]:
    return {
        "schema": TIMED_LYRICS_REPORT_SCHEMA,
        "version": TIMED_LYRICS_REPORT_VERSION,
        "analysis_status": analysis_status,
        "timing_ready": timing_ready,
        "performance_mode": performance_mode,
        "master_audio_sha256": str(master_audio_sha256 or "").strip().lower(),
        "lyrics_sha256": lyrics_hash,
        "model_id": model_id,
        "language": language,
        "minimum_alignment_confidence": minimum_confidence,
        "song_duration_seconds": song_duration,
        "excerpt": {
            "start_seconds": excerpt_start,
            "duration_seconds": excerpt_duration,
            "end_seconds": excerpt_start + excerpt_duration,
        },
        "vocal_intervals": [],
        "instrumental_intervals": [],
        "events": [],
        "diagnostics": {"cache_hit": False},
        "warnings": [],
    }


def _preview(report: Mapping[str, Any]) -> str:
    events = report.get("events")
    if not isinstance(events, list) or not events:
        warnings = report.get("warnings")
        reason = str(warnings[0]) if isinstance(warnings, list) and warnings else str(
            report.get("analysis_status", "not_required")
        )
        return f"No forced lyric timing — {reason}."
    lines = ["Verified authored-lyric timing (excerpt-relative):"]
    for event in events:
        authored = " / ".join(str(line) for line in event.get("authored_lines", []))
        lines.append(
            f"{float(event['start_seconds']):06.2f}-{float(event['end_seconds']):06.2f}  "
            f"[{float(event['confidence']):.2f}]  {authored}  <-  {event['transcript']}"
        )
    return "\n".join(lines)


def _outputs(report: dict[str, Any], status: str) -> tuple[str, str, bool, str]:
    return (_json(report), status, True, _preview(report))


def _fallback(
    report: dict[str, Any],
    warning: str,
    *,
    diagnostic: Any = None,
) -> tuple[str, str, bool, str]:
    report["analysis_status"] = "natural_fallback"
    report["timing_ready"] = False
    report["events"] = []
    report.setdefault("warnings", []).append(str(warning))
    if diagnostic is not None:
        report.setdefault("diagnostics", {})["fallback_detail"] = str(diagnostic)[:500]
    status = (
        "Natural audio-led fallback is safe: lyric timing was not trusted "
        f"({warning}). No forced singing instructions should be emitted."
    )
    return _outputs(report, status)


def _cache_key(
    master_hash: str,
    excerpt_start: float,
    excerpt_duration: float,
    lyrics_hash: str,
    model_id: str,
    language: str,
    minimum_confidence: float,
) -> tuple[Any, ...]:
    return (
        master_hash,
        round(excerpt_start, 6),
        round(excerpt_duration, 6),
        lyrics_hash,
        model_id,
        language,
        round(minimum_confidence, 6),
    )


def _cache_get(key: tuple[Any, ...]) -> tuple[str, str, bool, str] | None:
    with _CACHE_LOCK:
        value = _ANALYSIS_CACHE.get(key)
        if value is None:
            return None
        _ANALYSIS_CACHE.move_to_end(key)
    report = json.loads(value[0])
    report.setdefault("diagnostics", {})["cache_hit"] = True
    report["diagnostics"]["cache_key_sha256"] = hashlib.sha256(
        repr(key).encode("utf-8")
    ).hexdigest()
    status = "Timed lyric alignment restored from the deterministic analysis cache."
    return _outputs(report, status)


def _cache_put(key: tuple[Any, ...], outputs: tuple[str, str, bool, str]) -> None:
    with _CACHE_LOCK:
        _ANALYSIS_CACHE[key] = outputs
        _ANALYSIS_CACHE.move_to_end(key)
        while len(_ANALYSIS_CACHE) > _CACHE_MAX_ENTRIES:
            _ANALYSIS_CACHE.popitem(last=False)


class DiffusionGemmaTimedLyricsAnalyzer:
    """Produce optional, evidence-backed lyric timing without blocking video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_audio": ("AUDIO",),
                "performance_mode": ("STRING", {"forceInput": True}),
                "lyrics": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                ),
                "master_audio_sha256": ("STRING", {"forceInput": True}),
                "song_duration_seconds": (
                    "FLOAT",
                    {"default": 90.0, "min": 0.1, "max": 3600.0, "step": 0.1},
                ),
                "excerpt_start_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1},
                ),
                "excerpt_duration_seconds": (
                    "FLOAT",
                    {"default": 25.0, "min": 0.1, "max": 60.0, "step": 0.1},
                ),
                "language": (list(LANGUAGES), {"default": "en"}),
                "minimum_alignment_confidence": (
                    "FLOAT",
                    {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "vocal_stem": ("AUDIO", {"lazy": True}),
                "whisper_pipeline": ("WHISPER_PIPELINE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = (
        "timed_lyrics_report_json",
        "status",
        "safe_ready",
        "aligned_preview",
    )
    FUNCTION = "analyze"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Lazily measures a vocal stem and aligns <=15-second Whisper phrases to authored lyrics. "
        "Any uncertainty returns a safe Natural/audio-led fallback instead of blocking generation."
    )

    @classmethod
    def check_lazy_status(cls, **kwargs: Any) -> list[str]:
        try:
            mode = _normalize_mode(kwargs.get("performance_mode", ""))
        except ValueError:
            return []
        if mode != "Lyrics + lip sync":
            return []
        return [
            name
            for name in ("vocal_stem", "whisper_pipeline")
            if name in kwargs and kwargs[name] is None
        ]

    def analyze(
        self,
        final_audio: Any,
        performance_mode: Any,
        lyrics: Any,
        master_audio_sha256: Any,
        song_duration_seconds: Any,
        excerpt_start_seconds: Any,
        excerpt_duration_seconds: Any,
        language: Any,
        minimum_alignment_confidence: Any,
        vocal_stem: Any = None,
        whisper_pipeline: Any = None,
    ) -> tuple[str, str, bool, str]:
        lyrics_hash = _lyrics_sha256(lyrics)
        try:
            mode = _normalize_mode(performance_mode)
        except ValueError:
            mode = "Natural / audio-led sync"
        try:
            song_duration = _positive_finite(song_duration_seconds, "song_duration_seconds")
            excerpt_start = _positive_finite(
                excerpt_start_seconds, "excerpt_start_seconds", allow_zero=True
            )
            excerpt_duration = _positive_finite(
                excerpt_duration_seconds, "excerpt_duration_seconds"
            )
            minimum_confidence = _positive_finite(
                minimum_alignment_confidence,
                "minimum_alignment_confidence",
                allow_zero=True,
            )
            if minimum_confidence > 1.0:
                raise ValueError("minimum_alignment_confidence must be at most 1.")
        except ValueError:
            # Widget validation normally makes this unreachable.  Preserve the
            # fail-safe contract even for API-authored workflows.
            song_duration = max(0.0, float(song_duration_seconds or 0.0))
            excerpt_start = max(0.0, float(excerpt_start_seconds or 0.0))
            excerpt_duration = max(0.0, float(excerpt_duration_seconds or 0.0))
            minimum_confidence = 0.55
        selected_language = str(language or "en").strip().casefold()
        if selected_language not in LANGUAGES:
            selected_language = "en"
        report = _base_report(
            performance_mode=mode,
            lyrics_hash=lyrics_hash,
            master_audio_sha256=str(master_audio_sha256 or ""),
            song_duration=song_duration,
            excerpt_start=excerpt_start,
            excerpt_duration=excerpt_duration,
            language=selected_language,
            minimum_confidence=minimum_confidence,
            model_id=None,
            analysis_status="not_required" if mode != "Lyrics + lip sync" else "natural_fallback",
            timing_ready=False,
        )
        if mode != "Lyrics + lip sync":
            report["warnings"] = ["timed_lyrics_analysis_not_required_for_selected_mode"]
            status = (
                f"{mode}: timed lyric analysis is not required; lazy vocal separation and "
                "Whisper inputs were not evaluated."
            )
            return _outputs(report, status)

        if vocal_stem is None:
            # A connected Whisper loader may already have placed its model on
            # the GPU even when the separator socket is absent.  Do not retain
            # that allocation on this safe early-exit path.
            if whisper_pipeline is not None:
                _offload_model(whisper_pipeline)
            return _fallback(report, "vocal_stem_missing")
        if whisper_pipeline is None:
            return _fallback(report, "whisper_pipeline_missing")
        model_identifier = _model_id(whisper_pipeline)
        report["model_id"] = model_identifier
        try:
            master_hash = str(master_audio_sha256 or "").strip().lower()
            if not _SHA256_RE.fullmatch(master_hash):
                return _fallback(report, "master_audio_sha256_invalid")
            if excerpt_duration <= 0.0 or excerpt_start + excerpt_duration > song_duration + 0.05:
                return _fallback(report, "excerpt_outside_song_duration")
            final_hash = _waveform_sha256(final_audio)
            report["diagnostics"]["final_audio_sha256"] = final_hash
            if final_hash != master_hash:
                return _fallback(report, "master_audio_hash_mismatch")
            authored = _authored_lines(lyrics)
            report["diagnostics"]["authored_non_section_line_count"] = len(authored)
            if not authored:
                return _fallback(report, "authored_lyrics_empty")
            stem, mix, sample_rate, alignment_diagnostics = _stem_aligned_to_mix(
                vocal_stem, final_audio
            )
            report["diagnostics"]["audio_alignment"] = alignment_diagnostics
            if mix.numel() / float(sample_rate) + 0.05 < song_duration:
                return _fallback(report, "final_audio_shorter_than_song_duration")
            if excerpt_start + excerpt_duration > mix.numel() / float(sample_rate) + 0.05:
                return _fallback(report, "excerpt_outside_audio")
            similarity = _stem_full_mix_similarity(stem, mix)
            report["diagnostics"]["stem_full_mix_comparison"] = similarity
            if bool(similarity["near_identical_full_mix"]):
                return _fallback(
                    report,
                    "vocal_stem_matches_full_mix_separator_fallback",
                )
            vocal_intervals, instrumental_intervals, activity_diagnostics = _vocal_activity(
                stem,
                mix,
                sample_rate,
                excerpt_start,
                excerpt_duration,
            )
            report["vocal_intervals"] = vocal_intervals
            report["instrumental_intervals"] = instrumental_intervals
            report["diagnostics"]["vocal_activity"] = activity_diagnostics
            if not vocal_intervals:
                return _fallback(report, "no_vocal_activity_detected")
            key = _cache_key(
                master_hash,
                excerpt_start,
                excerpt_duration,
                lyrics_hash,
                model_identifier,
                selected_language,
                minimum_confidence,
            )
            cached = _cache_get(key)
            if cached is not None:
                return cached
            report["diagnostics"]["cache_key_sha256"] = hashlib.sha256(
                repr(key).encode("utf-8")
            ).hexdigest()
            _prepare_model(whisper_pipeline)
            phrases, transcription_diagnostics = _transcribe_active_intervals(
                whisper_pipeline,
                stem,
                sample_rate,
                excerpt_start,
                vocal_intervals,
                selected_language,
            )
            report["diagnostics"]["transcription"] = transcription_diagnostics
            events, alignment_report = _align_phrases(
                phrases,
                authored,
                excerpt_start,
                song_duration,
                minimum_confidence,
            )
            report["events"] = events
            report["diagnostics"]["alignment"] = alignment_report
            report["analysis_status"] = "timing_ready"
            report["timing_ready"] = True
            report["warnings"] = []
            status = (
                f"Timed lyrics ready: {len(events)} excerpt-relative event(s) aligned to "
                f"authored lyrics at >= {minimum_confidence:.2f} confidence."
            )
            outputs = _outputs(report, status)
            _cache_put(key, outputs)
            return outputs
        except Exception as exc:  # Timing evidence is never a generation blocker.
            warning = "timed_lyrics_analysis_uncertain"
            message = str(exc)
            if message.startswith("alignment_confidence_below_threshold"):
                warning = "alignment_confidence_below_threshold"
            elif "no_timestamped" in message:
                warning = "whisper_returned_no_timestamped_phrases"
            elif "Whisper" in message or "whisper" in message:
                warning = "whisper_transcription_failed"
            elif "vocal_stem" in message:
                warning = "vocal_stem_invalid_or_misaligned"
            report["diagnostics"]["exception_type"] = type(exc).__name__
            return _fallback(report, warning, diagnostic=message)
        finally:
            _offload_model(whisper_pipeline)


NODE_CLASS_MAPPINGS = {
    "DiffusionGemmaTimedLyricsAnalyzer": DiffusionGemmaTimedLyricsAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionGemmaTimedLyricsAnalyzer": "DiffusionGemma Timed Lyrics Analyzer",
}


__all__ = [
    "DiffusionGemmaTimedLyricsAnalyzer",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
