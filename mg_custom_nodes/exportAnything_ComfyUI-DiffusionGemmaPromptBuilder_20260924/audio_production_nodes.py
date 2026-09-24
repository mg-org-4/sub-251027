# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lightweight decoded-audio production controls for ACE-Step and LTX.

The analysis in this module is deliberately CPU-only and uses no learned model.
It measures the waveform that will actually condition LTX instead of trusting
soft text metadata such as a requested BPM or key.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
import re
import struct
from typing import Any, Mapping

import torch
import torch.nn.functional as torch_functional

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ModuleNotFoundError:  # Standalone repository tests do not add ComfyUI.
    class ExecutionBlocker:  # type: ignore[no-redef]
        def __init__(self, message: str | None) -> None:
            self.message = message


CATEGORY = "prompt/diffusiongemma/audio-production"
ANALYSIS_SCHEMA = "diffusiongemma.decoded_audio_analysis"
SELECTION_REPORT_SCHEMA = "diffusiongemma.audio_candidate_selection"
GUIDE_REPORT_SCHEMA = "diffusiongemma.ltx_audio_guide"
PERFORMANCE_REPORT_SCHEMA = "diffusiongemma.ltx_performance_prompt"
ANALYSIS_VERSION = 1
SELECTION_POLICY_REVISION = 4
ANALYSIS_SAMPLE_RATE = 6000
ANALYSIS_N_FFT = 512
ANALYSIS_HOP = 128
TONAL_WINDOW_SECONDS = 4.0
MAX_CANONICAL_HALF_TIME_RELATIVE_ERROR = 0.05
MAX_CANONICAL_DOUBLE_TIME_RELATIVE_ERROR = 0.05
MAX_SAFE_DOUBLE_TIME_EXCERPT_ONSETS_PER_SECOND = 4.0
MIN_SAFE_DOUBLE_TIME_VISUAL_RECOVERY_COVERAGE = 0.70
MIN_SAFE_DOUBLE_TIME_EXCERPT_SCORE = 0.90
MIN_SAFE_DOUBLE_TIME_TONAL_FAMILY_CONSISTENCY = 0.80
MIN_SAFE_DOUBLE_TIME_TONAL_TRANSITION_STABILITY = 0.80
MIN_SAFE_DOUBLE_TIME_SUPPORTING_SIGNAL_COUNT = 2
SELECTION_MODES = (
    "auto_select",
    "lock_candidate_1",
    "lock_candidate_2",
    "lock_candidate_3",
    "lock_candidate_4",
    "lock_by_hash",
)
GUIDE_MODES = ("full_mix", "sync_safe", "vocal_only")
LTX_PERFORMANCE_MODES = (
    "Dance / music sync",
    "Lyrics + lip sync",
    "Natural / audio-led sync",
)
MAX_LTX_PERFORMANCE_LYRIC_LINES = 8
MAX_LTX_PERFORMANCE_LYRIC_CHARS = 1600
PRODUCTION_CONCEPTS = (
    "Source passthrough",
    "Joint video-safe plan",
    "Audition and select",
)
ACE_REFERENCE_MODES = ("Compose new", "Cover reference")
SOUNDTRACK_SOURCES = ("Generate with ACE-Step", "Upload song")
AUDIO_SOURCE_POLICIES = ("ace_step", "uploaded_song")
MAX_UPLOAD_DURATION_SECONDS = 600.0
MAX_UPLOAD_CHANNELS = 8
MAX_UPLOAD_SAMPLE_RATE = 192_000
MAX_UPLOAD_SCALAR_SAMPLES = 80_000_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KEY_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _db(value: float, floor: float = -120.0) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return floor
    return max(floor, 20.0 * math.log10(value))


def _audio_parts(audio: Any, label: str = "audio") -> tuple[torch.Tensor, int]:
    if not isinstance(audio, Mapping):
        raise ValueError(f"{label} must be a ComfyUI AUDIO mapping.")
    waveform = audio.get("waveform")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError(f"{label} waveform must have shape [batch, channels, samples].")
    if any(int(size) <= 0 for size in waveform.shape):
        raise ValueError(f"{label} waveform dimensions must be non-empty.")
    try:
        sample_rate = int(audio.get("sample_rate", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} sample rate must be a positive integer.") from exc
    if sample_rate <= 0:
        raise ValueError(f"{label} sample rate must be a positive integer.")
    return waveform, sample_rate


def waveform_sha256(audio: Any) -> str:
    """Hash the exact waveform representation plus its timing metadata."""

    waveform, sample_rate = _audio_parts(audio)
    value = waveform.detach()
    if value.device.type != "cpu":
        value = value.to("cpu")
    value = value.contiguous()
    byte_tensor = value.view(torch.uint8)
    array = byte_tensor.numpy()
    payload = memoryview(array).cast("B")
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


def _analysis_mono(audio: Any) -> tuple[torch.Tensor, int, dict[str, Any], dict[str, float]]:
    waveform, sample_rate = _audio_parts(audio)
    value = waveform.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("audio waveform contains NaN or infinite samples.")
    source_peak = float(value.abs().max().item())
    source_rms = float(torch.sqrt(torch.mean(value.square())).item())
    source_clipped_fraction = float((value.abs() >= 0.999).to(torch.float32).mean().item())
    mono = value.mean(dim=(0, 1)).contiguous()
    source_samples = int(mono.numel())
    duration = source_samples / float(sample_rate)
    analysis_samples = max(1, int(round(duration * ANALYSIS_SAMPLE_RATE)))
    if sample_rate != ANALYSIS_SAMPLE_RATE or analysis_samples != source_samples:
        mono = torch_functional.interpolate(
            mono.reshape(1, 1, -1),
            size=analysis_samples,
            mode="linear",
            align_corners=False,
        ).reshape(-1)
    return (
        mono,
        ANALYSIS_SAMPLE_RATE,
        {
            "sample_rate": sample_rate,
            "shape": [int(size) for size in waveform.shape],
            "dtype": str(waveform.dtype),
            "duration_seconds": duration,
        },
        {
            "peak": source_peak,
            "rms": source_rms,
            "clipped_fraction": source_clipped_fraction,
        },
    )


def _normalized_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.to(dtype=torch.float64)
    right = right.to(dtype=torch.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    if denominator <= 1.0e-12:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def _key_estimate(chroma: torch.Tensor) -> tuple[str, float]:
    major = torch.tensor(_MAJOR_PROFILE, dtype=torch.float64)
    minor = torch.tensor(_MINOR_PROFILE, dtype=torch.float64)
    scores: list[tuple[float, str]] = []
    for root in range(12):
        scores.append((_normalized_correlation(chroma, torch.roll(major, root)), f"{_KEY_NAMES[root]} major"))
        scores.append((_normalized_correlation(chroma, torch.roll(minor, root)), f"{_KEY_NAMES[root]} minor"))
    scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
    margin = scores[0][0] - scores[1][0]
    return scores[0][1], _clamp((margin + 0.05) / 0.75)


def _key_family(key_name: str) -> str:
    root_name, quality = key_name.split(" ", 1)
    root = _KEY_NAMES.index(root_name)
    major_root = root if quality == "major" else (root + 3) % 12
    minor_root = (major_root + 9) % 12
    return f"{_KEY_NAMES[major_root]} major / {_KEY_NAMES[minor_root]} minor"


def _key_transition_metrics(keys: list[str]) -> dict[str, float]:
    if len(keys) < 2:
        return {
            "key_transition_rate": 0.0,
            "tonal_family_transition_rate": 0.0,
            "remote_tonal_jump_rate": 0.0,
            "parallel_major_minor_flip_rate": 0.0,
            "tonal_transition_stability": 1.0,
        }
    circle = (0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5)
    circle_position = {pitch: index for index, pitch in enumerate(circle)}
    transitions = 0
    family_transitions = 0
    remote = 0
    parallel_flips = 0
    for left, right in zip(keys, keys[1:]):
        left_root_name, left_quality = left.split(" ", 1)
        right_root_name, right_quality = right.split(" ", 1)
        left_root = _KEY_NAMES.index(left_root_name)
        right_root = _KEY_NAMES.index(right_root_name)
        if left != right:
            transitions += 1
        left_family_root = left_root if left_quality == "major" else (left_root + 3) % 12
        right_family_root = right_root if right_quality == "major" else (right_root + 3) % 12
        if left_family_root != right_family_root:
            family_transitions += 1
            distance = abs(circle_position[left_family_root] - circle_position[right_family_root])
            distance = min(distance, 12 - distance)
            if distance >= 3:
                remote += 1
        if left_root == right_root and left_quality != right_quality:
            parallel_flips += 1
    denominator = len(keys) - 1
    transition_rate = transitions / denominator
    family_rate = family_transitions / denominator
    remote_rate = remote / denominator
    parallel_rate = parallel_flips / denominator
    stability = _clamp(
        1.0
        - 0.25 * transition_rate
        - 0.25 * family_rate
        - 0.35 * remote_rate
        - 0.15 * parallel_rate
    )
    return {
        "key_transition_rate": transition_rate,
        "tonal_family_transition_rate": family_rate,
        "remote_tonal_jump_rate": remote_rate,
        "parallel_major_minor_flip_rate": parallel_rate,
        "tonal_transition_stability": stability,
    }


def _tempo_metrics(
    onset_envelope: torch.Tensor,
    frames_per_second: float,
    expected_bpm: float,
) -> dict[str, Any]:
    centered = onset_envelope.to(dtype=torch.float64)
    centered = centered - centered.mean()
    minimum_lag = max(1, int(round(frames_per_second * 60.0 / 240.0)))
    maximum_lag = min(
        max(minimum_lag, int(round(frames_per_second * 60.0 / 40.0))),
        max(minimum_lag, int(centered.numel()) // 2),
    )
    correlations: list[tuple[float, float, int]] = []
    for lag in range(minimum_lag, maximum_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
        correlation = float(torch.dot(left, right) / denominator) if denominator > 1.0e-12 else 0.0
        bpm = 60.0 * frames_per_second / lag
        correlations.append((correlation, bpm, lag))
    if not correlations:
        return {
            "raw_bpm": 0.0,
            "canonical_bpm": 0.0,
            "tempo_relation": "undetected",
            "tempo_confidence": 0.0,
            "tempo_equivalents": [],
        }
    target = float(expected_bpm)
    target_equivalents = [
        value
        for value in (target / 2.0, target, target * 2.0)
        if target > 0.0 and 40.0 <= value <= 240.0
    ]

    def periodicity_rank(item: tuple[float, float, int]) -> tuple[float, float]:
        correlation, bpm, _lag = item
        if target_equivalents:
            distance = min(abs(math.log2(bpm / value)) for value in target_equivalents)
            # Reject unrelated third/subharmonic autocorrelation peaks while still
            # allowing exact half- and double-time interpretations.
            correlation -= 0.25 * distance
        return (round(correlation, 6), bpm)

    # Prefer the faster periodic interpretation when correlations are effectively tied;
    # the canonicalization below records whether that is a double-time pulse.
    best = max(correlations, key=periodicity_rank)
    if target_equivalents:
        direct = min(correlations, key=lambda item: abs(math.log2(item[1] / target)))
        if target * 2.0 <= 240.0:
            doubled = min(correlations, key=lambda item: abs(math.log2(item[1] / (target * 2.0))))
            if doubled[0] >= max(0.30, 0.75 * direct[0]):
                best = doubled
        if best is direct and target / 2.0 >= 40.0:
            halved = min(correlations, key=lambda item: abs(math.log2(item[1] / (target / 2.0))))
            if halved[0] >= max(0.30, direct[0] + 0.10):
                best = halved
    ordered = sorted(item[0] for item in correlations)
    baseline = ordered[len(ordered) // 2]
    confidence = _clamp((best[0] - baseline) / max(1.0e-6, 1.0 - baseline))
    raw_bpm = float(best[1])
    equivalents = sorted(
        {
            round(value, 6)
            for value in (raw_bpm / 2.0, raw_bpm, raw_bpm * 2.0)
            if 40.0 <= value <= 240.0
        }
    )
    if math.isfinite(target) and target > 0.0:
        canonical = min(equivalents, key=lambda value: (abs(math.log2(value / target)), abs(value - target)))
    else:
        canonical = raw_bpm
        while canonical > 140.0 and canonical / 2.0 >= 40.0:
            canonical /= 2.0
        while canonical < 70.0 and canonical * 2.0 <= 240.0:
            canonical *= 2.0
    ratio = raw_bpm / canonical if canonical > 0.0 else 1.0
    if ratio > 1.75:
        relation = "double_time_detected"
    elif ratio < 0.625:
        relation = "half_time_detected"
    else:
        relation = "direct"
    return {
        "raw_bpm": raw_bpm,
        "canonical_bpm": canonical,
        "tempo_relation": relation,
        "tempo_confidence": confidence,
        "tempo_equivalents": equivalents,
    }


def _tonal_metrics(power: torch.Tensor, sample_rate: int) -> tuple[dict[str, Any], torch.Tensor]:
    frequencies = torch.fft.rfftfreq(ANALYSIS_N_FFT, d=1.0 / sample_rate)
    valid = (frequencies >= 55.0) & (frequencies <= min(3500.0, sample_rate / 2.0))
    selected_frequencies = frequencies[valid]
    selected_power = power[valid]
    midi = torch.round(69.0 + 12.0 * torch.log2(selected_frequencies / 440.0)).to(torch.long)
    pitch_classes = torch.remainder(midi, 12)
    chroma = torch.zeros((12, power.shape[1]), dtype=torch.float64)
    chroma.index_add_(0, pitch_classes, selected_power.to(torch.float64))
    frames_per_window = max(1, int(round(TONAL_WINDOW_SECONDS * sample_rate / ANALYSIS_HOP)))
    step = max(1, frames_per_window // 2)
    windows: list[torch.Tensor] = []
    for start in range(0, chroma.shape[1], step):
        end = min(chroma.shape[1], start + frames_per_window)
        if end - start < max(1, frames_per_window // 2) and windows:
            break
        profile = chroma[:, start:end].sum(dim=1)
        total = float(profile.sum())
        if total > 1.0e-12:
            windows.append(profile / total)
    if not windows:
        windows = [torch.full((12,), 1.0 / 12.0, dtype=torch.float64)]
    stack = torch.stack(windows)
    centroid = stack.mean(dim=0)
    centroid = centroid / max(float(centroid.sum()), 1.0e-12)
    centroid_norm = max(float(torch.linalg.vector_norm(centroid)), 1.0e-12)
    similarities = [
        float(torch.dot(window, centroid) / max(float(torch.linalg.vector_norm(window)) * centroid_norm, 1.0e-12))
        for window in windows
    ]
    keys = [_key_estimate(window)[0] for window in windows]
    dominant_key, key_count = Counter(keys).most_common(1)[0]
    key_agreement = key_count / len(keys)
    entropy = -float(torch.sum(centroid * torch.log(centroid.clamp_min(1.0e-12))))
    clarity = _clamp(1.0 - entropy / math.log(12.0))
    key_name, key_confidence = _key_estimate(centroid)
    stability = _clamp(
        0.45 * (sum(similarities) / len(similarities))
        + 0.30 * key_agreement
        + 0.25 * clarity
    )
    transitions = _key_transition_metrics(keys)
    return (
        {
            "tonal_window_stability": stability,
            "tonal_centroid_similarity": sum(similarities) / len(similarities),
            "tonal_center_agreement": key_agreement,
            "tonal_clarity": clarity,
            "dominant_key_estimate": key_name,
            "dominant_key_confidence": key_confidence,
            "tonal_window_count": len(windows),
            "tonal_window_keys": keys,
            **transitions,
        },
        chroma,
    )


def _spectral_flatness_metrics(power: torch.Tensor, sample_rate: int) -> dict[str, float]:
    """Return conservative broadband-noise proxies over active audible frames."""

    frequencies = torch.fft.rfftfreq(ANALYSIS_N_FFT, d=1.0 / sample_rate)
    audible = (frequencies >= 55.0) & (frequencies <= min(3500.0, sample_rate / 2.0))
    band = power[audible].to(torch.float64)
    if band.numel() == 0 or band.shape[1] == 0:
        values = torch.ones((1,), dtype=torch.float64)
    else:
        arithmetic = band.mean(dim=0)
        peak_frame_energy = float(arithmetic.max())
        active = arithmetic > max(1.0e-14, peak_frame_energy * 1.0e-6)
        if not bool(active.any()):
            values = torch.ones((1,), dtype=torch.float64)
        else:
            active_band = band[:, active].clamp_min(1.0e-18)
            geometric = torch.exp(torch.log(active_band).mean(dim=0))
            values = (geometric / active_band.mean(dim=0).clamp_min(1.0e-18)).clamp(0.0, 1.0)
    return {
        "spectral_flatness_mean": float(values.mean()),
        "spectral_flatness_median": float(values.median()),
        "spectral_flatness_p90": float(torch.quantile(values, 0.90)),
    }


def _range_score(value: float, ideal_low: float, ideal_high: float, outer_low: float, outer_high: float) -> float:
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        return _clamp((value - outer_low) / max(1.0e-9, ideal_low - outer_low))
    return _clamp((outer_high - value) / max(1.0e-9, outer_high - ideal_high))


def _excerpt_metrics(
    mono: torch.Tensor,
    power: torch.Tensor,
    chroma: torch.Tensor,
    onset_peaks: torch.Tensor,
    sample_rate: int,
    excerpt_duration_seconds: float,
    forced_start_seconds: float | None = None,
) -> dict[str, Any]:
    duration = mono.numel() / float(sample_rate)
    excerpt_duration = min(float(excerpt_duration_seconds), duration)
    sample_count = max(1, int(round(excerpt_duration * sample_rate)))
    frame_count = max(1, int(round(excerpt_duration * sample_rate / ANALYSIS_HOP)))
    step_samples = max(1, int(round(min(1.0, excerpt_duration / 8.0) * sample_rate)))
    frequency = torch.fft.rfftfreq(ANALYSIS_N_FFT, d=1.0 / sample_rate)
    mid_band = (frequency >= 150.0) & (frequency <= min(3500.0, sample_rate / 2.0))
    audible_band = (frequency >= 55.0) & (frequency <= min(3500.0, sample_rate / 2.0))
    global_chroma = chroma.sum(dim=1)
    global_chroma_norm = max(float(torch.linalg.vector_norm(global_chroma)), 1.0e-12)
    candidates: list[dict[str, Any]] = []
    last_start = max(0, mono.numel() - sample_count)
    if forced_start_seconds is not None:
        forced = float(forced_start_seconds)
        if not math.isfinite(forced) or forced < 0.0:
            raise ValueError("locked_start_seconds must be a non-negative finite value.")
        forced_sample = int(round(forced * sample_rate))
        if forced_sample > last_start:
            raise ValueError("locked excerpt start does not leave enough audio for the requested duration.")
        starts = [forced_sample]
    else:
        starts = list(range(0, last_start + 1, step_samples))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
    for start_sample in starts:
        end_sample = min(mono.numel(), start_sample + sample_count)
        segment = mono[start_sample:end_sample]
        rms = float(torch.sqrt(torch.mean(segment.square())).item())
        peak = float(segment.abs().max().item())
        rms_dbfs = _db(rms)
        crest_db = _db(peak / max(rms, 1.0e-12), floor=0.0)
        start_frame = min(power.shape[1] - 1, max(0, start_sample // ANALYSIS_HOP))
        end_frame = min(power.shape[1], start_frame + frame_count)
        local_power = power[:, start_frame:end_frame]
        audible_energy = float(local_power[audible_band].sum())
        mid_ratio = float(local_power[mid_band].sum()) / max(audible_energy, 1.0e-12)
        local_chroma = chroma[:, start_frame:end_frame].sum(dim=1)
        tonal_similarity = float(
            torch.dot(local_chroma, global_chroma)
            / max(float(torch.linalg.vector_norm(local_chroma)) * global_chroma_norm, 1.0e-12)
        )
        local_onsets = int(onset_peaks[start_frame:end_frame].sum().item())
        onset_density = local_onsets / max(excerpt_duration, 1.0e-9)
        local_key_window_frames = max(1, int(round(2.0 * sample_rate / ANALYSIS_HOP)))
        local_keys: list[str] = []
        local_key_details: list[dict[str, Any]] = []
        for local_start in range(start_frame, end_frame, local_key_window_frames):
            local_end = min(end_frame, local_start + local_key_window_frames)
            if local_end <= local_start:
                continue
            key_profile = chroma[:, local_start:local_end].sum(dim=1)
            if float(key_profile.sum()) > 1.0e-12:
                key_name, key_confidence = _key_estimate(key_profile)
                local_key_details.append(
                    {"key": key_name, "confidence": key_confidence}
                )
                if key_confidence >= 0.25:
                    local_keys.append(key_name)
        local_families = [_key_family(key) for key in local_keys]
        if local_families:
            tonal_family, family_count = Counter(local_families).most_common(1)[0]
            tonal_family_consistency = family_count / len(local_families)
        else:
            tonal_family = "undetected"
            # No confident transition evidence is advisory rather than proof of
            # instability. Global chroma clarity remains part of the score.
            tonal_family_consistency = 1.0
        transition_metrics = _key_transition_metrics(local_keys)
        activity_score = _range_score(rms_dbfs, -28.0, -8.0, -48.0, -2.0)
        vocal_band_score = _clamp((mid_ratio - 0.30) / 0.55)
        vocal_proxy = _clamp(0.60 * vocal_band_score + 0.40 * activity_score)
        onset_score = _range_score(onset_density, 0.35, 3.75, 0.0, 7.0)
        crest_score = _range_score(crest_db, 5.0, 16.0, 1.0, 26.0)
        score = _clamp(
            0.20 * _clamp(tonal_similarity)
            + 0.12 * tonal_family_consistency
            + 0.13 * transition_metrics["tonal_transition_stability"]
            + 0.20 * onset_score
            + 0.18 * activity_score
            + 0.12 * vocal_proxy
            + 0.05 * crest_score
        )
        candidates.append(
            {
                "start_seconds": start_sample / float(sample_rate),
                "duration_seconds": excerpt_duration,
                "score": score,
                "rms_dbfs": rms_dbfs,
                "crest_factor_db": crest_db,
                "onset_density_per_second": onset_density,
                "tonal_similarity": tonal_similarity,
                "tonal_family": tonal_family,
                "tonal_family_consistency": tonal_family_consistency,
                "local_key_windows": local_keys,
                "local_key_window_details": local_key_details,
                "confident_key_transition_window_count": len(local_keys),
                **transition_metrics,
                "mid_band_energy_ratio": mid_ratio,
                "vocal_presence_proxy": vocal_proxy,
                "likely_vocal_active_proxy": bool(vocal_proxy >= 0.45 and rms_dbfs >= -40.0),
                "start_was_locked": forced_start_seconds is not None,
            }
        )
    # A likely vocal-active window always outranks dead air. Lower start time is
    # the deterministic tie-breaker.
    chosen = max(
        candidates,
        key=lambda item: (
            bool(item["likely_vocal_active_proxy"]),
            round(float(item["score"]), 9),
            -float(item["start_seconds"]),
        ),
    )
    selected_start_sample = int(round(float(chosen["start_seconds"]) * sample_rate))
    selected_end_sample = min(mono.numel(), selected_start_sample + sample_count)
    timeline_window_seconds = min(2.0, excerpt_duration)
    timeline_window_samples = max(1, int(round(timeline_window_seconds * sample_rate)))
    timeline_entries: list[dict[str, Any]] = []
    previous_profile: torch.Tensor | None = None
    for relative_start_sample in range(0, max(1, selected_end_sample - selected_start_sample), timeline_window_samples):
        absolute_start = selected_start_sample + relative_start_sample
        absolute_end = min(selected_end_sample, absolute_start + timeline_window_samples)
        if absolute_end <= absolute_start:
            continue
        segment = mono[absolute_start:absolute_end]
        local_duration = (absolute_end - absolute_start) / float(sample_rate)
        local_rms_dbfs = _db(float(torch.sqrt(torch.mean(segment.square())).item()))
        start_frame = min(power.shape[1] - 1, max(0, absolute_start // ANALYSIS_HOP))
        end_frame = min(
            power.shape[1],
            max(start_frame + 1, int(math.ceil(absolute_end / ANALYSIS_HOP))),
        )
        local_power = power[:, start_frame:end_frame]
        local_audible = float(local_power[audible_band].sum())
        local_mid_ratio = float(local_power[mid_band].sum()) / max(local_audible, 1.0e-12)
        local_profile = chroma[:, start_frame:end_frame].sum(dim=1)
        local_key = _key_estimate(local_profile)[0] if float(local_profile.sum()) > 1.0e-12 else "undetected"
        if previous_profile is None:
            tonal_change = 0.0
        else:
            tonal_change = 1.0 - float(
                torch.dot(local_profile, previous_profile)
                / max(
                    float(torch.linalg.vector_norm(local_profile))
                    * float(torch.linalg.vector_norm(previous_profile)),
                    1.0e-12,
                )
            )
        previous_profile = local_profile
        onset_density = float(onset_peaks[start_frame:end_frame].sum().item()) / max(local_duration, 1.0e-9)
        activity = _range_score(local_rms_dbfs, -28.0, -8.0, -48.0, -2.0)
        vocal_proxy = _clamp(0.60 * _clamp((local_mid_ratio - 0.30) / 0.55) + 0.40 * activity)
        visual_recovery = bool(
            local_rms_dbfs >= -45.0
            and onset_density <= 4.7
            and tonal_change <= 0.45
        )
        vocal_active_stable = bool(visual_recovery and vocal_proxy >= 0.45)
        timeline_entries.append(
            {
                "relative_start_seconds": relative_start_sample / float(sample_rate),
                "relative_end_seconds": (absolute_end - selected_start_sample) / float(sample_rate),
                "onset_density_per_second": onset_density,
                "rms_dbfs": local_rms_dbfs,
                "mid_band_vocal_presence_proxy": vocal_proxy,
                "key_proxy": local_key,
                "tonal_change_proxy": _clamp(tonal_change),
                "low_density_visual_recovery_proxy": visual_recovery,
                "vocal_active_stable_proxy": vocal_active_stable,
            }
        )

    def merged_intervals(flag: str) -> list[dict[str, Any]]:
        intervals: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for entry in timeline_entries:
            if not entry[flag]:
                if current is not None:
                    intervals.append(current)
                    current = None
                continue
            if current is None:
                current = {
                    "relative_start_seconds": entry["relative_start_seconds"],
                    "relative_end_seconds": entry["relative_end_seconds"],
                    "window_count": 1,
                    "mean_onset_density_per_second": entry["onset_density_per_second"],
                    "maximum_tonal_change_proxy": entry["tonal_change_proxy"],
                }
            else:
                count = int(current["window_count"])
                current["relative_end_seconds"] = entry["relative_end_seconds"]
                current["window_count"] = count + 1
                current["mean_onset_density_per_second"] = (
                    float(current["mean_onset_density_per_second"]) * count
                    + float(entry["onset_density_per_second"])
                ) / (count + 1)
                current["maximum_tonal_change_proxy"] = max(
                    float(current["maximum_tonal_change_proxy"]),
                    float(entry["tonal_change_proxy"]),
                )
        if current is not None:
            intervals.append(current)
        if len(intervals) > 8:
            intervals = sorted(
                intervals,
                key=lambda item: (
                    float(item["relative_end_seconds"]) - float(item["relative_start_seconds"]),
                    -float(item["mean_onset_density_per_second"]),
                ),
                reverse=True,
            )[:8]
            intervals.sort(key=lambda item: float(item["relative_start_seconds"]))
        return intervals

    visual_recovery_intervals = merged_intervals("low_density_visual_recovery_proxy")
    vocal_active_intervals = merged_intervals("vocal_active_stable_proxy")
    timeline_duration = max(float(chosen["duration_seconds"]), 1.0e-9)

    def coverage_ratio(flag: str) -> float:
        covered_seconds = sum(
            max(
                0.0,
                float(entry["relative_end_seconds"])
                - float(entry["relative_start_seconds"]),
            )
            for entry in timeline_entries
            if bool(entry[flag])
        )
        return _clamp(covered_seconds / timeline_duration)

    # Calculate coverage from the complete timeline before bounding the copy
    # embedded in reports. These ratios let the selector distinguish a benign
    # eighth-note/octave alias from sustained high-pressure double-time motion.
    chosen["low_density_visual_recovery_coverage"] = coverage_ratio(
        "low_density_visual_recovery_proxy"
    )
    chosen["vocal_active_stable_coverage"] = coverage_ratio(
        "vocal_active_stable_proxy"
    )
    total_timeline_entries = len(timeline_entries)
    if total_timeline_entries > 24:
        selected_indices = sorted(
            {
                round(index * (total_timeline_entries - 1) / 23)
                for index in range(24)
            }
        )
        timeline_entries = [timeline_entries[index] for index in selected_indices]
    chosen["selected_excerpt_timeline"] = {
        "window_seconds": timeline_window_seconds,
        "entry_count_total": total_timeline_entries,
        "entries_returned": len(timeline_entries),
        "bounded_max_entries": 24,
        "truncated": total_timeline_entries > len(timeline_entries),
        "entries": timeline_entries,
    }
    chosen["low_density_visual_recovery_intervals"] = visual_recovery_intervals
    chosen["vocal_active_stable_intervals"] = vocal_active_intervals
    chosen["timeline_note"] = (
        "All entries are lightweight signal proxies relative to the selected excerpt. "
        "Visual-recovery intervals do not require vocals; vocal-active intervals do. "
        "These proxies are not vocal transcription or beat-grid ground truth."
    )
    return chosen


def analyze_decoded_audio(
    audio: Any,
    *,
    expected_bpm: float = 0.0,
    excerpt_duration_seconds: float = 20.0,
    locked_start_seconds: float | None = None,
) -> dict[str, Any]:
    """Measure a decoded ComfyUI AUDIO mapping without using a learned model."""

    expected = float(expected_bpm)
    excerpt_duration = float(excerpt_duration_seconds)
    if not math.isfinite(expected) or expected < 0.0 or expected > 300.0:
        raise ValueError("expected_bpm must be 0 (auto) or a finite value from 1 to 300.")
    if not math.isfinite(excerpt_duration) or excerpt_duration <= 0.0:
        raise ValueError("excerpt_duration_seconds must be a positive finite number.")
    mono, sample_rate, identity, source_metrics = _analysis_mono(audio)
    duration = float(identity["duration_seconds"])
    peak = float(source_metrics["peak"])
    rms = float(source_metrics["rms"])
    crest = peak / max(rms, 1.0e-12)
    clipped_fraction = float(source_metrics["clipped_fraction"])
    analysis_mono_peak = float(mono.abs().max().item())
    analysis_mono_rms = float(torch.sqrt(torch.mean(mono.square())).item())
    padded = mono
    if padded.numel() < ANALYSIS_N_FFT:
        padded = torch_functional.pad(padded, (0, ANALYSIS_N_FFT - padded.numel()))
    window = torch.hann_window(ANALYSIS_N_FFT, dtype=torch.float32)
    spectrum = torch.stft(
        padded,
        n_fft=ANALYSIS_N_FFT,
        hop_length=ANALYSIS_HOP,
        win_length=ANALYSIS_N_FFT,
        window=window,
        center=False,
        return_complex=True,
    )
    magnitude = spectrum.abs()
    power = magnitude.square()
    log_magnitude = torch.log1p(magnitude)
    onset = torch.zeros(log_magnitude.shape[1], dtype=torch.float32)
    if log_magnitude.shape[1] > 1:
        onset[1:] = torch.relu(log_magnitude[:, 1:] - log_magnitude[:, :-1]).mean(dim=0)
    onset = torch_functional.avg_pool1d(
        torch_functional.pad(onset.reshape(1, 1, -1), (1, 1), mode="replicate"),
        kernel_size=3,
        stride=1,
    ).reshape(-1)
    threshold = float(onset.median() + 0.75 * onset.std(unbiased=False))
    local_max = torch_functional.max_pool1d(
        torch_functional.pad(onset.reshape(1, 1, -1), (3, 3), mode="replicate"),
        kernel_size=7,
        stride=1,
    ).reshape(-1)
    onset_peaks = (onset >= local_max) & (onset > max(threshold, 1.0e-7))
    onset_count = int(onset_peaks.sum().item())
    onset_density = onset_count / max(duration, 1.0e-9)
    frames_per_second = sample_rate / float(ANALYSIS_HOP)
    tempo = _tempo_metrics(onset, frames_per_second, expected)
    tonal, chroma = _tonal_metrics(power, sample_rate)
    spectral_flatness = _spectral_flatness_metrics(power, sample_rate)
    excerpt = _excerpt_metrics(
        mono,
        power,
        chroma,
        onset_peaks,
        sample_rate,
        excerpt_duration,
        locked_start_seconds,
    )
    rms_dbfs = _db(rms)
    crest_db = _db(crest, floor=0.0)
    canonical_tempo_relative_error = None
    if expected > 0.0 and float(tempo["canonical_bpm"]) > 0.0:
        canonical_tempo_relative_error = abs(
            float(tempo["canonical_bpm"]) - expected
        ) / expected
    tempo_alignment = 1.0
    if expected > 0.0 and tempo["canonical_bpm"] > 0.0:
        tempo_alignment = math.exp(-4.0 * abs(math.log2(float(tempo["canonical_bpm"]) / expected)))
    tempo_score = _clamp(0.65 * float(tempo["tempo_confidence"]) + 0.35 * tempo_alignment)
    aligned_half_time = (
        tempo["tempo_relation"] == "half_time_detected"
        and canonical_tempo_relative_error is not None
        and canonical_tempo_relative_error <= MAX_CANONICAL_HALF_TIME_RELATIVE_ERROR
    )
    # With no requested BPM, half/double labels are only a signal-derived
    # interpretation of autocorrelation peaks. Do not punish an otherwise
    # usable song for that arbitrary canonicalization.
    if expected > 0.0 and tempo["tempo_relation"] != "direct" and not aligned_half_time:
        tempo_score *= 0.20
    onset_score = _range_score(onset_density, 0.35, 4.25, 0.0, 8.0)
    rms_score = _range_score(rms_dbfs, -28.0, -8.0, -50.0, -1.0)
    crest_score = _range_score(crest_db, 5.0, 18.0, 1.0, 30.0)
    clipping_score = _clamp(1.0 - clipped_fraction / 0.02)
    score = _clamp(
        0.32 * float(tonal["tonal_window_stability"])
        + 0.18 * tempo_score
        + 0.17 * onset_score
        + 0.12 * rms_score
        + 0.08 * crest_score
        + 0.05 * clipping_score
        + 0.08 * float(excerpt["score"])
    )
    return {
        "schema": ANALYSIS_SCHEMA,
        "version": ANALYSIS_VERSION,
        **identity,
        "analysis_sample_rate": sample_rate,
        "expected_bpm": expected,
        "waveform_sha256": waveform_sha256(audio),
        "peak_dbfs": _db(peak),
        "rms_dbfs": rms_dbfs,
        "crest_factor_db": crest_db,
        "clipped_sample_fraction": clipped_fraction,
        "analysis_mono_peak_dbfs": _db(analysis_mono_peak),
        "analysis_mono_rms_dbfs": _db(analysis_mono_rms),
        "onset_count": onset_count,
        "onset_density_per_second": onset_density,
        "detected_bpm_raw": float(tempo["raw_bpm"]),
        "detected_bpm": float(tempo["canonical_bpm"]),
        "tempo_relation": tempo["tempo_relation"],
        "tempo_confidence": float(tempo["tempo_confidence"]),
        "tempo_equivalents": [float(value) for value in tempo["tempo_equivalents"]],
        "canonical_tempo_relative_error": canonical_tempo_relative_error,
        **tonal,
        **spectral_flatness,
        "suggested_excerpt": excerpt,
        "production_score": score,
        "analysis_notes": [
            "Key and vocal activity are signal-derived proxies, not transcription or note-level proof.",
            "Spectral flatness is a conservative broadband-noise/artifact proxy, not a genre judgment.",
            "Half/double tempo equivalents are compared before scoring against expected BPM.",
            "A half-time accent pattern is advisory and unpenalized when its canonical tempo matches the requested BPM within 5%.",
            "A canonically aligned double-time reading is advisory when the selected excerpt proves low transient pressure and vocal activity plus at least two supporting recovery, excerpt-score, tonal-family, or transition-stability signals; ordinary hard QC gates still apply, and its tempo subscore remains penalized.",
            "QC limits are empirical provisional thresholds for this music-video workflow, not universal music-quality claims.",
        ],
    }


def _is_canonically_aligned_low_pressure_double_time(
    analysis: Mapping[str, Any],
) -> bool:
    if str(analysis.get("tempo_relation", "")) != "double_time_detected":
        return False
    relative_error = analysis.get("canonical_tempo_relative_error")
    if not isinstance(relative_error, (int, float)) or not math.isfinite(
        float(relative_error)
    ):
        return False
    excerpt = analysis.get("suggested_excerpt")
    if not isinstance(excerpt, Mapping):
        return False
    supporting_signals = (
        float(excerpt.get("low_density_visual_recovery_coverage", 0.0))
        >= MIN_SAFE_DOUBLE_TIME_VISUAL_RECOVERY_COVERAGE,
        float(excerpt.get("score", 0.0)) >= MIN_SAFE_DOUBLE_TIME_EXCERPT_SCORE,
        float(excerpt.get("tonal_family_consistency", 0.0))
        >= MIN_SAFE_DOUBLE_TIME_TONAL_FAMILY_CONSISTENCY,
        float(excerpt.get("tonal_transition_stability", 0.0))
        >= MIN_SAFE_DOUBLE_TIME_TONAL_TRANSITION_STABILITY,
    )
    # An octave-related autocorrelation peak is a tempo interpretation, not an
    # independent quality failure. Canonical alignment, measured onset pressure,
    # and vocal activity are mandatory. The four partly overlapping quality
    # proxies form a deterministic evidence bundle: two are sufficient, so one
    # conservative proxy cannot veto several stronger independent measurements.
    # Clipping, artifacts, silence, duration, tonal failure, and the user-selected
    # production-score floor remain independently enforced in _candidate_failures().
    return bool(
        float(relative_error) <= MAX_CANONICAL_DOUBLE_TIME_RELATIVE_ERROR
        and float(excerpt.get("onset_density_per_second", math.inf))
        <= MAX_SAFE_DOUBLE_TIME_EXCERPT_ONSETS_PER_SECOND
        and bool(excerpt.get("likely_vocal_active_proxy"))
        and sum(bool(value) for value in supporting_signals)
        >= MIN_SAFE_DOUBLE_TIME_SUPPORTING_SIGNAL_COUNT
    )


def _normalize_audio_source_policy(value: Any) -> str:
    normalized = str(value or "ace_step").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "ace": "ace_step",
        "ace_step": "ace_step",
        "generated": "ace_step",
        "generated_candidates": "ace_step",
        "upload": "uploaded_song",
        "uploaded": "uploaded_song",
        "upload_song": "uploaded_song",
        "uploaded_song": "uploaded_song",
    }
    if normalized not in aliases:
        raise ValueError(
            f"source_policy must be one of {', '.join(AUDIO_SOURCE_POLICIES)}."
        )
    return aliases[normalized]


def _candidate_advisories(
    analysis: Mapping[str, Any],
    *,
    source_policy: str = "ace_step",
    minimum_score: float = 0.52,
) -> list[str]:
    policy = _normalize_audio_source_policy(source_policy)
    relation = str(analysis.get("tempo_relation", ""))
    relative_error = analysis.get("canonical_tempo_relative_error")
    advisories: list[str] = []
    if (
        relation == "half_time_detected"
        and isinstance(relative_error, (int, float))
        and math.isfinite(float(relative_error))
        and float(relative_error) <= MAX_CANONICAL_HALF_TIME_RELATIVE_ERROR
    ):
        advisories.append("canonically_aligned_half_time_pulse")
    elif _is_canonically_aligned_low_pressure_double_time(analysis):
        advisories.append("canonically_aligned_low_pressure_double_time_pulse")
    if policy != "uploaded_song":
        return advisories

    advisories.insert(0, "source_locked_uploaded_song")
    excerpt = analysis["suggested_excerpt"]
    if float(analysis["tonal_window_stability"]) < 0.25:
        advisories.append("source_locked_advisory_unstable_tonal_windows")
    if float(analysis["onset_density_per_second"]) > 8.0:
        advisories.append("source_locked_advisory_excessive_onset_density")
    if float(excerpt["onset_density_per_second"]) > 4.7:
        advisories.append("source_locked_advisory_dense_selected_excerpt")
    expected_bpm = float(analysis.get("expected_bpm", 0.0))
    if relation != "direct" and not any(
        value.startswith("canonically_aligned_") for value in advisories
    ):
        advisories.append(
            "source_locked_advisory_signal_only_tempo_interpretation"
            if expected_bpm <= 0.0
            else "source_locked_advisory_requested_tempo_mismatch"
        )
    transition_evidence = int(excerpt.get("confident_key_transition_window_count", 0))
    if transition_evidence >= 3 and float(excerpt.get("tonal_family_consistency", 0.0)) < 0.60:
        advisories.append("source_locked_advisory_tonal_family_variation")
    if transition_evidence >= 4 and float(excerpt.get("remote_tonal_jump_rate", 0.0)) > 0.34:
        advisories.append("source_locked_advisory_remote_tonal_jumps")
    if transition_evidence >= 4 and float(excerpt.get("parallel_major_minor_flip_rate", 0.0)) > 0.34:
        advisories.append("source_locked_advisory_parallel_mode_variation")
    if transition_evidence >= 4 and float(excerpt.get("tonal_transition_stability", 0.0)) < 0.50:
        advisories.append("source_locked_advisory_tonal_transition_variation")
    global_tonal_windows = int(analysis.get("tonal_window_count", 0))
    if global_tonal_windows >= 4 and float(analysis.get("remote_tonal_jump_rate", 0.0)) > 0.34:
        advisories.append("source_locked_advisory_global_remote_tonal_jumps")
    if global_tonal_windows >= 4 and float(analysis.get("tonal_transition_stability", 0.0)) < 0.50:
        advisories.append("source_locked_advisory_global_tonal_variation")
    if not bool(excerpt["likely_vocal_active_proxy"]):
        advisories.append("source_locked_advisory_no_vocal_proxy")
    if float(analysis["production_score"]) < float(minimum_score):
        advisories.append("source_locked_advisory_below_generated_candidate_score")
    return advisories


def _candidate_failures(
    analysis: Mapping[str, Any],
    *,
    excerpt_duration_seconds: float,
    minimum_score: float,
    source_policy: str = "ace_step",
) -> list[str]:
    policy = _normalize_audio_source_policy(source_policy)
    failures: list[str] = []
    # ACE-Step quantizes duration to a 25 Hz latent clock (40 ms per step).
    # Treat one rounded latent step as equivalent to the requested wall clock;
    # larger shortages still fail closed.
    ace_duration_tolerance_seconds = 0.05
    if (
        float(analysis["duration_seconds"]) + ace_duration_tolerance_seconds
        < excerpt_duration_seconds
    ):
        failures.append("audio_shorter_than_requested_excerpt")
    if float(analysis["rms_dbfs"]) < -48.0:
        failures.append("silent_or_near_silent")
    if float(analysis["clipped_sample_fraction"]) > 0.02:
        failures.append("excessive_clipping")
    if (
        float(analysis.get("spectral_flatness_mean", 1.0)) > 0.35
        and float(analysis.get("tonal_clarity", 0.0)) < 0.04
    ):
        failures.append("broadband_noise_or_artifact_like_spectrum")
    # An uploaded song is user-selected source material, not an ACE audition
    # candidate. Preserve only technical integrity gates; musical style,
    # tempo, vocal presence, onset pressure, tonal movement, and the generated-
    # candidate score remain visible advisories but cannot substitute or reject
    # the user's soundtrack.
    if policy == "uploaded_song":
        return failures
    if float(analysis["tonal_window_stability"]) < 0.25:
        failures.append("unstable_tonal_windows")
    if float(analysis["onset_density_per_second"]) > 8.0:
        failures.append("excessive_onset_density")
    excerpt = analysis["suggested_excerpt"]
    if float(excerpt["onset_density_per_second"]) > 4.7:
        failures.append("selected_excerpt_onset_density_above_empirical_limit")
    # Tempo autocorrelation can label ordinary subdivisions as an octave alias.
    # Aligned half-time is harmless. Aligned double-time remains riskier, but is
    # admitted when the actual selected excerpt independently proves low onset
    # pressure and vocal activity plus a two-of-four supporting evidence bundle.
    # Recovery coverage ranks accepted candidates, while the ordinary tonal,
    # artifact, and score checks below remain hard gates. Unlike aligned half-time,
    # accepted double-time keeps its tempo-score penalty.
    tempo_relation = str(analysis.get("tempo_relation"))
    tempo_advisories = _candidate_advisories(analysis)
    if tempo_relation != "direct" and not tempo_advisories:
        if tempo_relation == "double_time_detected":
            failures.append("double_time_outside_low_pressure_contract")
        elif tempo_relation == "half_time_detected":
            failures.append("half_time_outside_canonical_alignment_contract")
        else:
            failures.append("tempo_undetected_or_unrelated")
    transition_evidence = int(excerpt.get("confident_key_transition_window_count", 0))
    if transition_evidence >= 3 and float(excerpt.get("tonal_family_consistency", 0.0)) < 0.60:
        failures.append("selected_excerpt_tonal_family_inconsistent")
    if transition_evidence >= 4 and float(excerpt.get("remote_tonal_jump_rate", 0.0)) > 0.34:
        failures.append("selected_excerpt_remote_tonal_jumps")
    if transition_evidence >= 4 and float(excerpt.get("parallel_major_minor_flip_rate", 0.0)) > 0.34:
        failures.append("selected_excerpt_parallel_mode_instability")
    if transition_evidence >= 4 and float(excerpt.get("tonal_transition_stability", 0.0)) < 0.50:
        failures.append("selected_excerpt_tonal_transition_instability")
    global_tonal_windows = int(analysis.get("tonal_window_count", 0))
    if global_tonal_windows >= 4 and float(analysis.get("remote_tonal_jump_rate", 0.0)) > 0.34:
        failures.append("global_remote_tonal_jumps")
    if global_tonal_windows >= 4 and float(analysis.get("tonal_transition_stability", 0.0)) < 0.50:
        failures.append("global_tonal_transition_instability")
    if not bool(excerpt["likely_vocal_active_proxy"]):
        failures.append("no_non_silent_vocal_active_proxy_window")
    if float(analysis["production_score"]) < minimum_score:
        failures.append("production_score_below_threshold")
    return failures


def _selection_failure(
    report: dict[str, Any],
    message: str,
) -> tuple[Any, float, Any, Any, str, str, bool]:
    report["ready"] = False
    report["status"] = message
    return (
        ExecutionBlocker(message),
        0.0,
        # The Project Master Contract consumes this output as its immutable
        # audio identity.  Returning an empty STRING let that independent
        # branch execute and replace the useful QC diagnosis with a misleading
        # "invalid SHA-256" exception.  Propagate the same blocker so every
        # song-dependent branch stops at the actual selection failure.
        ExecutionBlocker(message),
        # The compact report is a hard scheduling dependency of Director in
        # the production workflow. Block it as well as AUDIO so a failed
        # audition cannot spend minutes running Director before LTX stops.
        ExecutionBlocker(message),
        _json(report),
        message,
        False,
    )


def _candidate_summary(entry: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "index": entry.get("index"),
        "present": bool(entry.get("present")),
        "passed": bool(entry.get("passed")),
        "failures": list(entry.get("failures", []))[:12],
        "advisories": list(entry.get("advisories", []))[:12],
    }
    analysis = entry.get("analysis")
    if isinstance(analysis, Mapping):
        summary.update(
            {
                "waveform_sha256": analysis.get("waveform_sha256", ""),
                "production_score": analysis.get("production_score"),
                "expected_bpm": analysis.get("expected_bpm"),
                "detected_bpm_raw": analysis.get("detected_bpm_raw"),
                "detected_bpm": analysis.get("detected_bpm"),
                "tempo_relation": analysis.get("tempo_relation"),
                "canonical_tempo_relative_error": analysis.get(
                    "canonical_tempo_relative_error"
                ),
                "onset_density_per_second": analysis.get("onset_density_per_second"),
                "tonal_window_stability": analysis.get("tonal_window_stability"),
                "tonal_clarity": analysis.get("tonal_clarity"),
                "spectral_flatness_mean": analysis.get("spectral_flatness_mean"),
                "suggested_excerpt_score": analysis.get("suggested_excerpt", {}).get(
                    "score"
                ),
                "suggested_excerpt_start_seconds": analysis.get(
                    "suggested_excerpt", {}
                ).get("start_seconds"),
                "suggested_excerpt_visual_recovery_coverage": analysis.get(
                    "suggested_excerpt", {}
                ).get("low_density_visual_recovery_coverage"),
                "suggested_excerpt_vocal_active_coverage": analysis.get(
                    "suggested_excerpt", {}
                ).get("vocal_active_stable_coverage"),
            }
        )
    if entry.get("error"):
        summary["error"] = str(entry["error"])[:400]
    return summary


def _compact_director_report(
    audit_report: Mapping[str, Any],
    selected_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = audit_report.get("settings", {})
    compact: dict[str, Any] = {
        "schema": "diffusiongemma.music_audition_report",
        "version": ANALYSIS_VERSION,
        "selection_policy_revision": audit_report.get(
            "selection_policy_revision", SELECTION_POLICY_REVISION
        ),
        "status": audit_report.get("status", ""),
        "ready": bool(audit_report.get("ready")),
        "selected_candidate_index": audit_report.get("selected_candidate_index"),
        "selected_audio_sha256": audit_report.get("selected_waveform_sha256", ""),
        "selection_lock_token": audit_report.get("selection_lock_token", ""),
        "lock": {
            key: audit_report.get("lock", {}).get(key)
            for key in (
                "active",
                "satisfied",
                "excerpt_start_locked",
                "resolved_candidate_index",
                "resolved_waveform_sha256",
                "resolved_start_seconds",
                "failure_reason",
            )
            if key in audit_report.get("lock", {})
        },
        "settings": {
            "candidate_count": settings.get("candidate_count"),
            "selection_mode": settings.get("selection_mode"),
            "expected_bpm": settings.get("expected_bpm"),
            "excerpt_duration_seconds": settings.get("excerpt_duration_seconds"),
            "minimum_score": settings.get("minimum_score"),
            "empirical_provisional_thresholds": settings.get("empirical_provisional_thresholds", {}),
        },
        "candidate_summaries": [
            _candidate_summary(entry)
            for entry in list(audit_report.get("candidates", []))[:4]
            if isinstance(entry, Mapping)
        ],
    }
    if isinstance(selected_analysis, Mapping):
        excerpt = selected_analysis.get("suggested_excerpt", {})
        timeline = excerpt.get("selected_excerpt_timeline", {}) if isinstance(excerpt, Mapping) else {}
        compact["qc"] = {
            key: selected_analysis.get(key)
            for key in (
                "production_score",
                "duration_seconds",
                "expected_bpm",
                "detected_bpm_raw",
                "detected_bpm",
                "tempo_relation",
                "tempo_confidence",
                "onset_density_per_second",
                "rms_dbfs",
                "crest_factor_db",
                "tonal_window_stability",
                "tonal_transition_stability",
                "tonal_clarity",
                "spectral_flatness_mean",
                "dominant_key_estimate",
                "dominant_key_confidence",
            )
        }
        compact["selected_excerpt"] = {
            "start_seconds": excerpt.get("start_seconds"),
            "duration_seconds": excerpt.get("duration_seconds"),
            "score": excerpt.get("score"),
            "onset_rate_hz": excerpt.get("onset_density_per_second"),
            "rms_dbfs": excerpt.get("rms_dbfs"),
            "tonal_family_stability": excerpt.get("tonal_family_consistency"),
            "tonal_transition_stability": excerpt.get("tonal_transition_stability"),
            "vocal_presence_proxy": excerpt.get("vocal_presence_proxy"),
            "likely_vocal_active_proxy": excerpt.get("likely_vocal_active_proxy"),
            "low_density_visual_recovery_coverage": excerpt.get(
                "low_density_visual_recovery_coverage"
            ),
            "vocal_active_stable_coverage": excerpt.get(
                "vocal_active_stable_coverage"
            ),
            "timeline": {
                "window_seconds": timeline.get("window_seconds"),
                "entry_count_total": timeline.get("entry_count_total"),
                "truncated": timeline.get("truncated"),
                "entries": list(timeline.get("entries", []))[:24],
            },
            "low_density_visual_recovery_intervals": list(
                excerpt.get("low_density_visual_recovery_intervals", [])
            )[:8],
            "vocal_active_stable_intervals": list(
                excerpt.get("vocal_active_stable_intervals", [])
            )[:8],
            "proxy_note": excerpt.get("timeline_note", ""),
        }
    encoded = _json(compact)
    if len(encoded) > 12_000 and "selected_excerpt" in compact:
        compact["selected_excerpt"]["timeline"]["entries"] = compact["selected_excerpt"]["timeline"]["entries"][:12]
        compact["selected_excerpt"]["timeline"]["truncated"] = True
        compact["selected_excerpt"]["low_density_visual_recovery_intervals"] = compact["selected_excerpt"]["low_density_visual_recovery_intervals"][:4]
        compact["selected_excerpt"]["vocal_active_stable_intervals"] = compact["selected_excerpt"]["vocal_active_stable_intervals"][:4]
        encoded = _json(compact)
    if len(encoded) > 12_000:
        compact["candidate_summaries"] = compact["candidate_summaries"][:1]
        compact["report_compacted_to_fit_director"] = True
    return compact


def _normalize_selection_mode(value: Any) -> str:
    mode = str(value or "").strip().casefold().replace(" ", "_")
    if mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {', '.join(SELECTION_MODES)}.")
    return mode


class DiffusionGemmaAudioCandidateSelector:
    """Measure decoded song candidates and fail closed unless one passes QC."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "candidate_count": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                "selection_mode": (list(SELECTION_MODES), {"default": "auto_select"}),
                "expected_bpm": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 300.0,
                        "step": 0.1,
                        "tooltip": "Requested tempo used to interpret equivalent pulse rates. Canonically aligned half-time is advisory; an aligned double-time subdivision is advisory only with <=5% canonical error, <=4.0 excerpt onsets/s, vocal activity, and at least two supporting recovery/score/tonal signals; other tempo aliases block. Use 0 for signal-only analysis.",
                    },
                ),
                "excerpt_duration_seconds": (
                    "FLOAT",
                    {"default": 20.0, "min": 0.5, "max": 600.0, "step": 0.1},
                ),
                "minimum_score": (
                    "FLOAT",
                    {
                        "default": 0.52,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Numeric production-score floor only. Independent duration, clipping, tempo, onset, tonal, vocal-activity, and artifact failures still block selection.",
                    },
                ),
            },
            "optional": {
                "locked_waveform_sha256": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Required by lock_by_hash; any mismatch fails closed.",
                    },
                ),
                "locked_start_seconds": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 2000.0,
                        "step": 0.1,
                        "tooltip": "-1 uses deterministic auto-start; non-negative values lock the exact excerpt start.",
                    },
                ),
                "candidate_1": ("AUDIO", {"lazy": True}),
                "candidate_2": ("AUDIO", {"lazy": True}),
                "candidate_3": ("AUDIO", {"lazy": True}),
                "candidate_4": ("AUDIO", {"lazy": True}),
                "source_policy": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "ace_step applies generated-candidate audition gates. uploaded_song "
                            "source-locks the user's track: technical integrity remains blocking "
                            "while tempo, vocals, style, and production-score heuristics are advisory."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = (
        "selected_audio",
        "suggested_start_seconds",
        "waveform_sha256",
        "director_report_json",
        "audit_report_json",
        "status",
        "ready",
    )
    FUNCTION = "select"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "QC-checks one decoded song for Source/Joint production, or auditions two to four songs. "
        "Auto-select favors direct/aligned-half tempo, then evidence-qualified double-time recovery, actual excerpt quality, and whole-song score. "
        "An uploaded song uses source-locked technical-integrity gates while musical heuristics remain advisory. "
        "A locked candidate or hash never silently falls back to another song."
    )

    @classmethod
    def check_lazy_status(cls, **kwargs):
        count = max(1, min(4, int(kwargs.get("candidate_count", 2))))
        mode = _normalize_selection_mode(kwargs.get("selection_mode", "auto_select"))
        if mode == "lock_by_hash":
            lock_hash = str(kwargs.get("locked_waveform_sha256", "") or "").strip().casefold()
            # Let select() emit the explicit fail-closed report before resolving
            # any expensive ACE lane when the lock itself cannot possibly match.
            if not _SHA256_RE.fullmatch(lock_hash):
                return []
        locked_match = re.fullmatch(r"lock_candidate_([1-4])", mode)
        if locked_match is not None:
            index = int(locked_match.group(1))
            names = [f"candidate_{index}"] if index <= count else []
        else:
            names = [f"candidate_{index}" for index in range(1, count + 1)]
        # Comfy omits an unconnected optional input entirely. Request only a
        # connected-but-unresolved lazy socket; requesting an absent socket
        # raises NodeInputError before our friendly fail-closed result can run.
        return [name for name in names if name in kwargs and kwargs[name] is None]

    def select(
        self,
        candidate_count,
        selection_mode,
        expected_bpm,
        excerpt_duration_seconds,
        minimum_score,
        locked_waveform_sha256="",
        locked_start_seconds=-1.0,
        candidate_1=None,
        candidate_2=None,
        candidate_3=None,
        candidate_4=None,
        source_policy="ace_step",
    ):
        count = int(candidate_count)
        if count < 1 or count > 4:
            raise ValueError("candidate_count must be from 1 to 4.")
        mode = _normalize_selection_mode(selection_mode)
        policy = _normalize_audio_source_policy(source_policy)
        expected = float(expected_bpm)
        excerpt_duration = float(excerpt_duration_seconds)
        threshold = float(minimum_score)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_score must be a finite value from 0 to 1.")
        candidates = [candidate_1, candidate_2, candidate_3, candidate_4]
        lock_hash = str(locked_waveform_sha256 or "").strip().casefold()
        locked_start = float(locked_start_seconds)
        if not math.isfinite(locked_start) or locked_start < -1.0:
            raise ValueError("locked_start_seconds must be -1 (auto) or a non-negative finite value.")
        lock_match = re.fullmatch(r"lock_candidate_([1-4])", mode)
        indices = [int(lock_match.group(1))] if lock_match is not None else list(range(1, count + 1))
        report: dict[str, Any] = {
            "schema": SELECTION_REPORT_SCHEMA,
            "version": ANALYSIS_VERSION,
            "selection_policy_revision": SELECTION_POLICY_REVISION,
            "settings": {
                "candidate_count": count,
                "selection_mode": mode,
                "expected_bpm": expected,
                "excerpt_duration_seconds": excerpt_duration,
                "minimum_score": threshold,
                "source_policy": policy,
                "locked_start_seconds": locked_start,
                "empirical_provisional_thresholds": {
                    "maximum_selected_excerpt_onsets_per_second": 4.7,
                    "minimum_selected_excerpt_tonal_family_consistency": 0.60,
                    "maximum_noise_like_spectral_flatness_mean": 0.35,
                    "minimum_global_tonal_transition_stability": 0.50,
                    "maximum_global_remote_tonal_jump_rate": 0.34,
                    "maximum_audio_duration_shortfall_seconds": 0.05,
                    "aligned_half_time_policy": "advisory_without_tempo_score_penalty",
                    "maximum_aligned_half_time_relative_error": MAX_CANONICAL_HALF_TIME_RELATIVE_ERROR,
                    "double_time_policy": "mandatory_alignment_pressure_vocal_plus_two_of_four_quality_signals",
                    "maximum_aligned_double_time_relative_error": MAX_CANONICAL_DOUBLE_TIME_RELATIVE_ERROR,
                    "maximum_safe_double_time_excerpt_onsets_per_second": MAX_SAFE_DOUBLE_TIME_EXCERPT_ONSETS_PER_SECOND,
                    "minimum_safe_double_time_visual_recovery_coverage": MIN_SAFE_DOUBLE_TIME_VISUAL_RECOVERY_COVERAGE,
                    "minimum_safe_double_time_excerpt_score": MIN_SAFE_DOUBLE_TIME_EXCERPT_SCORE,
                    "minimum_safe_double_time_tonal_family_consistency": MIN_SAFE_DOUBLE_TIME_TONAL_FAMILY_CONSISTENCY,
                    "minimum_safe_double_time_tonal_transition_stability": MIN_SAFE_DOUBLE_TIME_TONAL_TRANSITION_STABILITY,
                    "minimum_safe_double_time_supporting_signal_count": MIN_SAFE_DOUBLE_TIME_SUPPORTING_SIGNAL_COUNT,
                    "double_time_supporting_signal_policy": "at_least_two_of_recovery_excerpt_score_tonal_family_transition_stability",
                    "accepted_double_time_tempo_score_policy": "retain_non_direct_penalty",
                    "misaligned_or_undetected_tempo_policy": "hard_failure",
                    "scope": (
                        "ACE-Step candidate audition heuristic"
                        if policy == "ace_step"
                        else "source-locked uploaded song; technical integrity gates only"
                    ),
                },
                "auto_selection_order": [
                    "tempo_safety_class_descending",
                    "double_time_visual_recovery_coverage_descending",
                    "selected_excerpt_score_descending",
                    "full_song_production_score_descending",
                    "candidate_index_ascending",
                ],
            },
            "lock": {
                "active": mode != "auto_select",
                "requested_waveform_sha256": lock_hash,
                "requested_start_seconds": locked_start,
                "excerpt_start_locked": mode != "auto_select" and locked_start >= 0.0,
                "satisfied": False,
                "failure_reason": "",
            },
            "candidates": [],
            "selected_candidate_index": None,
            "selected_waveform_sha256": "",
        }
        if lock_match is not None and indices[0] > count:
            message = f"Audio selection is locked to candidate {indices[0]}, outside candidate_count={count}."
            report["lock"]["failure_reason"] = "locked_candidate_outside_candidate_count"
            return _selection_failure(report, message)
        if mode == "lock_by_hash" and not _SHA256_RE.fullmatch(lock_hash):
            message = "Audio selection is locked by hash, but no valid lowercase/uppercase SHA-256 was supplied."
            report["lock"]["failure_reason"] = "invalid_locked_waveform_sha256"
            return _selection_failure(report, message)

        measured: dict[int, dict[str, Any]] = {}
        missing: list[int] = []
        for index in indices:
            audio = candidates[index - 1]
            if audio is None:
                missing.append(index)
                report["candidates"].append(
                    {"index": index, "present": False, "passed": False, "failures": ["candidate_missing"]}
                )
                continue
            try:
                analysis = analyze_decoded_audio(
                    audio,
                    expected_bpm=expected,
                    excerpt_duration_seconds=excerpt_duration,
                    locked_start_seconds=(
                        locked_start
                        if mode != "auto_select" and locked_start >= 0.0
                        else None
                    ),
                )
                failures = _candidate_failures(
                    analysis,
                    excerpt_duration_seconds=excerpt_duration,
                    minimum_score=threshold,
                    source_policy=policy,
                )
                entry = {
                    "index": index,
                    "present": True,
                    "passed": not failures,
                    "failures": failures,
                    "advisories": _candidate_advisories(
                        analysis,
                        source_policy=policy,
                        minimum_score=threshold,
                    ),
                    "analysis": analysis,
                }
            except (RuntimeError, TypeError, ValueError) as exc:
                entry = {
                    "index": index,
                    "present": True,
                    "passed": False,
                    "failures": ["analysis_error"],
                    "error": str(exc),
                }
            report["candidates"].append(entry)
            measured[index] = entry

        if mode in {"auto_select", "lock_by_hash"} and missing:
            message = f"Audio audition expected {count} candidates; missing lanes: {', '.join(map(str, missing))}."
            if mode == "lock_by_hash":
                report["lock"]["failure_reason"] = "candidate_lane_missing"
            return _selection_failure(report, message)

        selected_index: int | None = None
        if lock_match is not None:
            selected_index = indices[0]
            entry = measured.get(selected_index)
            if not entry or not bool(entry.get("passed")):
                report["lock"]["failure_reason"] = "locked_candidate_failed_qc"
                failure_codes = ", ".join(
                    str(value) for value in (entry or {}).get("failures", [])
                ) or "unspecified_qc_failure"
                return _selection_failure(
                    report,
                    f"Audio selection is locked to candidate {selected_index}, but that candidate did not pass decoded-audio QC: {failure_codes}.",
                )
        elif mode == "lock_by_hash":
            hash_matches = [
                index
                for index, entry in measured.items()
                if entry.get("analysis", {}).get("waveform_sha256") == lock_hash
            ]
            if not hash_matches:
                report["lock"]["failure_reason"] = "locked_hash_not_present"
                return _selection_failure(report, "The locked waveform hash is not present among the audition candidates.")
            selected_index = min(hash_matches)
            if not bool(measured[selected_index].get("passed")):
                report["lock"]["failure_reason"] = "locked_hash_failed_qc"
                failure_codes = ", ".join(
                    str(value)
                    for value in measured[selected_index].get("failures", [])
                ) or "unspecified_qc_failure"
                return _selection_failure(
                    report,
                    "The locked waveform is present but did not pass decoded-audio QC: "
                    f"{failure_codes}.",
                )
        else:
            passing = [entry for entry in measured.values() if bool(entry.get("passed"))]
            if passing:
                selected_entry = max(
                    passing,
                    key=lambda entry: (
                        0
                        if "canonically_aligned_low_pressure_double_time_pulse"
                        in entry.get("advisories", [])
                        else 1,
                        round(
                            float(
                                entry["analysis"]["suggested_excerpt"].get(
                                    "low_density_visual_recovery_coverage", 0.0
                                )
                            ),
                            9,
                        )
                        if "canonically_aligned_low_pressure_double_time_pulse"
                        in entry.get("advisories", [])
                        else 1.0,
                        round(
                            float(
                                entry["analysis"]["suggested_excerpt"].get(
                                    "score", 0.0
                                )
                            ),
                            9,
                        ),
                        round(float(entry["analysis"]["production_score"]), 9),
                        -int(entry["index"]),
                    ),
                )
                selected_index = int(selected_entry["index"])

        if selected_index is None:
            hard_failure_details = "; ".join(
                f"candidate {int(entry.get('index', 0))}: "
                + ", ".join(str(value) for value in entry.get("failures", [])[:4])
                for entry in report["candidates"]
                if entry.get("failures")
            )
            message = (
                "No decoded song candidate passed decoded-audio QC. minimum_score is only "
                "the numeric score floor; independent hard QC gates still apply."
            )
            if hard_failure_details:
                message += f" Hard QC — {hard_failure_details}."
            return _selection_failure(report, message)
        selected_entry = measured[selected_index]
        selected_analysis = selected_entry["analysis"]
        selected_hash = str(selected_analysis["waveform_sha256"])
        selected_start = float(selected_analysis["suggested_excerpt"]["start_seconds"])
        lock_token_payload = (
            f"diffusiongemma-audio-selection-lock@1|{selected_hash}|"
            f"{selected_start:.9f}|{excerpt_duration:.9f}"
        )
        selection_lock_token = hashlib.sha256(lock_token_payload.encode("utf-8")).hexdigest()
        report["selected_candidate_index"] = selected_index
        report["selected_waveform_sha256"] = selected_hash
        report["selected_suggested_excerpt"] = selected_analysis["suggested_excerpt"]
        report["selection_lock_token"] = selection_lock_token
        report["ready"] = True
        report["lock"].update(
            {
                "satisfied": mode != "auto_select",
                "resolved_candidate_index": selected_index,
                "resolved_waveform_sha256": selected_hash,
                "resolved_start_seconds": selected_start,
                "selection_lock_token": selection_lock_token,
            }
        )
        selected_excerpt_score = float(
            selected_analysis["suggested_excerpt"].get("score", 0.0)
        )
        advisory_values = list(selected_entry.get("advisories", []))
        advisory_text = ""
        if "canonically_aligned_half_time_pulse" in advisory_values:
            advisory_text = (
                "; accepted half-time pulse "
                f"{float(selected_analysis['detected_bpm_raw']):.1f} BPM as canonical "
                f"{float(selected_analysis['detected_bpm']):.1f} BPM versus requested "
                f"{float(selected_analysis['expected_bpm']):.1f} BPM (advisory)"
            )
        elif "canonically_aligned_low_pressure_double_time_pulse" in advisory_values:
            selected_excerpt = selected_analysis["suggested_excerpt"]
            advisory_text = (
                "; accepted low-pressure double-time subdivision "
                f"{float(selected_analysis['detected_bpm_raw']):.1f} BPM as canonical "
                f"{float(selected_analysis['detected_bpm']):.1f} BPM versus requested "
                f"{float(selected_analysis['expected_bpm']):.1f} BPM; excerpt onset density "
                f"{float(selected_excerpt['onset_density_per_second']):.2f}/s, recovery "
                f"{100.0 * float(selected_excerpt['low_density_visual_recovery_coverage']):.0f}% "
                "(advisory; tempo-score penalty retained)"
            )
        elif advisory_values:
            advisory_text = "; advisory " + ", ".join(
                str(value) for value in advisory_values
            )
        status_prefix = (
            "Source-locked upload candidate"
            if policy == "uploaded_song"
            else "Decoded-audio QC selected candidate"
        )
        status = (
            f"{status_prefix} {selected_index} at excerpt score "
            f"{selected_excerpt_score:.3f} (full-song score "
            f"{float(selected_analysis['production_score']):.3f}){advisory_text}; "
            f"waveform lock {selected_hash[:12]}…."
        )
        if policy == "uploaded_song":
            status = status.replace(
                " at excerpt score",
                " passed technical integrity; measured excerpt score",
                1,
            )
        report["status"] = status
        director_report = _compact_director_report(report, selected_analysis)
        selected_audio = candidates[selected_index - 1]
        # Return the exact upstream waveform object. The SHA-256 output is its
        # immutable identity; no normalization, resampling, or gain is applied.
        return (
            selected_audio,
            selected_start,
            selected_hash,
            _json(director_report),
            _json(report),
            status,
            True,
        )


def _fit_stem_to_reference(stem: Any, final_audio: Any) -> torch.Tensor:
    stem_waveform, stem_rate = _audio_parts(stem, "vocal_stem")
    final_waveform, final_rate = _audio_parts(final_audio, "final_audio")
    value = stem_waveform.detach().to(device="cpu", dtype=torch.float32)
    final_shape = tuple(int(size) for size in final_waveform.shape)
    if stem_rate != final_rate:
        target_length = max(1, int(round(value.shape[-1] * final_rate / float(stem_rate))))
        value = torch_functional.interpolate(value, size=target_length, mode="linear", align_corners=False)
    if value.shape[0] == 1 and final_shape[0] > 1:
        value = value.repeat(final_shape[0], 1, 1)
    if value.shape[0] != final_shape[0]:
        raise ValueError("vocal_stem batch count must equal final_audio batch count or be one.")
    if value.shape[1] == 1 and final_shape[1] > 1:
        value = value.repeat(1, final_shape[1], 1)
    elif final_shape[1] == 1 and value.shape[1] > 1:
        value = value.mean(dim=1, keepdim=True)
    elif value.shape[1] != final_shape[1]:
        raise ValueError("vocal_stem channels cannot be aligned with final_audio channels.")
    if value.shape[-1] < final_shape[-1]:
        value = torch_functional.pad(value, (0, final_shape[-1] - value.shape[-1]))
    else:
        value = value[..., : final_shape[-1]]
    if not bool(torch.isfinite(value).all()):
        raise ValueError("vocal_stem contains NaN or infinite samples.")
    return value.contiguous()


def _sync_safe_guide(final_audio: Any, vocal_stem: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
    final_waveform, sample_rate = _audio_parts(final_audio, "final_audio")
    source = final_waveform.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(source).all()):
        raise ValueError("final_audio contains NaN or infinite samples.")
    if source.shape[1] > 1:
        center = source.mean(dim=1, keepdim=True)
        source_centered = center + 0.35 * (source - center)
    else:
        source_centered = source
    used_stem = vocal_stem is not None
    if used_stem:
        vocals = _fit_stem_to_reference(vocal_stem, final_audio)
        guide = vocals + 0.25 * (source_centered - vocals)
    else:
        guide = source_centered
    kernel = max(3, int(round(sample_rate * 0.00075)))
    if kernel % 2 == 0:
        kernel += 1
    kernel = min(kernel, max(3, int(guide.shape[-1]) // 2 * 2 - 1))
    if kernel >= 3 and guide.shape[-1] > kernel:
        padding = kernel // 2
        smoothed = torch_functional.avg_pool1d(
            torch_functional.pad(guide, (padding, padding), mode="reflect"),
            kernel_size=kernel,
            stride=1,
        )
        guide = smoothed + 0.45 * (guide - smoothed)
    source_rms = torch.sqrt(torch.mean(source.square(), dim=-1, keepdim=True)).clamp_min(1.0e-8)
    guide_rms = torch.sqrt(torch.mean(guide.square(), dim=-1, keepdim=True)).clamp_min(1.0e-8)
    gain = (source_rms / guide_rms).clamp(0.5, 1.35)
    guide = guide * gain
    peak = guide.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
    guide = guide * torch.minimum(torch.ones_like(peak), 0.98 / peak)
    output = {"waveform": guide.contiguous(), "sample_rate": sample_rate}
    return output, {
        "used_vocal_stem": used_stem,
        "center_channel_emphasis": bool(source.shape[1] > 1),
        "accompaniment_gain_with_stem": 0.25 if used_stem else None,
        "high_frequency_residual_gain": 0.45,
        "peak_limit": 0.98,
    }


class DiffusionGemmaLTXAudioGuide:
    """Build a low-risk LTX guide while preserving the untouched final master."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_audio": ("AUDIO",),
                "mode": (list(GUIDE_MODES), {"default": "sync_safe"}),
            },
            "optional": {"vocal_stem": ("AUDIO", {"lazy": True})},
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = (
        "conditioning_audio",
        "final_audio",
        "conditioning_sha256",
        "guide_report_json",
        "status",
        "ready",
    )
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Separates the audio LTX hears from the untouched soundtrack that is muxed. "
        "sync_safe uses center emphasis and lightweight transient softening only."
    )

    @classmethod
    def check_lazy_status(cls, **kwargs):
        normalized = str(kwargs.get("mode", "") or "").strip().casefold()
        if normalized == "vocal_only" and "vocal_stem" in kwargs and kwargs["vocal_stem"] is None:
            return ["vocal_stem"]
        return []

    def build(self, final_audio, mode, vocal_stem=None):
        normalized = str(mode or "").strip().casefold()
        report: dict[str, Any] = {
            "schema": GUIDE_REPORT_SCHEMA,
            "version": ANALYSIS_VERSION,
            "mode": normalized,
            "final_audio_preserved": True,
        }
        try:
            final_hash = waveform_sha256(final_audio)
            if normalized not in GUIDE_MODES:
                raise ValueError(f"mode must be one of {', '.join(GUIDE_MODES)}.")
            if normalized == "full_mix":
                conditioning = final_audio
                processing = {"operation": "identity"}
            elif normalized == "vocal_only":
                if vocal_stem is None:
                    message = "vocal_only LTX guidance requires a connected vocal stem."
                    report.update(
                        {
                            "ready": False,
                            "status": message,
                            "final_waveform_sha256": final_hash,
                            "failure_reason": "vocal_stem_missing",
                        }
                    )
                    return (ExecutionBlocker(None), final_audio, "", _json(report), message, False)
                waveform = _fit_stem_to_reference(vocal_stem, final_audio)
                _final_waveform, sample_rate = _audio_parts(final_audio, "final_audio")
                conditioning = {"waveform": waveform, "sample_rate": sample_rate}
                processing = {"operation": "aligned_vocal_stem", "used_vocal_stem": True}
            else:
                # sync_safe is intentionally a lightweight full-mix transform.
                # A connected stem separator must remain unevaluated unless the
                # user explicitly selects vocal_only.
                conditioning, processing = _sync_safe_guide(final_audio, None)
            conditioning_hash = waveform_sha256(conditioning)
        except (RuntimeError, TypeError, ValueError) as exc:
            message = f"LTX audio guide is blocked: {exc}"
            report.update({"ready": False, "status": message, "failure_reason": "invalid_audio"})
            return (ExecutionBlocker(None), final_audio, "", _json(report), message, False)
        status = f"LTX audio guide ready in {normalized} mode; final master remains untouched."
        report.update(
            {
                "ready": True,
                "status": status,
                "processing": processing,
                "final_waveform_sha256": final_hash,
                "conditioning_waveform_sha256": conditioning_hash,
            }
        )
        return (conditioning, final_audio, conditioning_hash, _json(report), status, True)


def _normalize_ltx_performance_mode(value: Any) -> str:
    normalized = " ".join(str(value or "").strip().casefold().replace("_", " ").split())
    aliases = {
        "dance": "Dance / music sync",
        "music sync": "Dance / music sync",
        "dance/music sync": "Dance / music sync",
        "dance / music sync": "Dance / music sync",
        "lyrics": "Lyrics + lip sync",
        "lip sync": "Lyrics + lip sync",
        "lyrics+lip sync": "Lyrics + lip sync",
        "lyrics + lip sync": "Lyrics + lip sync",
        "natural": "Natural / audio-led sync",
        "audio led": "Natural / audio-led sync",
        "audio-led": "Natural / audio-led sync",
        "audio led sync": "Natural / audio-led sync",
        "audio-led sync": "Natural / audio-led sync",
        "natural/audio-led sync": "Natural / audio-led sync",
        "natural / audio led sync": "Natural / audio-led sync",
        "natural / audio-led sync": "Natural / audio-led sync",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "performance_mode must be Dance / music sync, Lyrics + lip sync, "
            "or Natural / audio-led sync."
        ) from exc


def _authored_sung_lyric_lines(value: Any) -> tuple[str, list[str]]:
    """Return normalized source text and exact complete sung lines.

    Only line-ending and outer-whitespace normalization is applied. Standalone
    bracketed section directives are intentionally excluded: LTX should see the
    words a performer may articulate, not ``[Verse]`` or ``[Instrumental]``.
    """

    source = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    sung_lines: list[str] = []
    for raw_line in source.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"\[[^\]\n]{1,120}\]", line):
            continue
        sung_lines.append(line)
    return source, sung_lines


def _lyric_word_count(value: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", value, flags=re.UNICODE))


def _bounded_proportional_lyric_window(
    lyrics: Any,
    *,
    song_duration_seconds: Any,
    excerpt_start_seconds: Any,
    excerpt_duration_seconds: Any,
) -> tuple[str, dict[str, Any]]:
    """Choose a deterministic, complete-line lyric window for an audio excerpt.

    ACE/QC currently provides no transcript or word timestamps. The selection is
    therefore a transparent planning heuristic: sung lines are treated as evenly
    distributed over the source-song clock, then a one-line context margin is
    added and bounded by the LTX speech-density, line-count, and character caps.
    """

    source, sung_lines = _authored_sung_lyric_lines(lyrics)
    if not sung_lines:
        raise ValueError(
            "Lyrics + lip sync requires at least one complete sung lyric line; "
            "section labels and [Instrumental] do not count."
        )
    try:
        song_duration = float(song_duration_seconds)
        excerpt_start = float(excerpt_start_seconds)
        excerpt_duration = float(excerpt_duration_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("Song and excerpt timing values must be finite numbers.") from exc
    if not math.isfinite(song_duration) or song_duration <= 0.0:
        raise ValueError("song_duration_seconds must be a positive finite number.")
    if not math.isfinite(excerpt_start) or excerpt_start < 0.0:
        raise ValueError("excerpt_start_seconds must be a non-negative finite number.")
    if not math.isfinite(excerpt_duration) or excerpt_duration <= 0.0:
        raise ValueError("excerpt_duration_seconds must be a positive finite number.")

    line_count = len(sung_lines)
    clamped_start = min(excerpt_start, song_duration)
    clamped_end = min(song_duration, clamped_start + excerpt_duration)
    if clamped_end <= clamped_start:
        # An excerpt that starts exactly at/past the source end maps to its last
        # authored line rather than wrapping around to the beginning.
        raw_first = line_count - 1
        raw_last = line_count - 1
    else:
        raw_first = min(
            line_count - 1,
            int(math.floor((clamped_start / song_duration) * line_count)),
        )
        raw_last = min(
            line_count - 1,
            max(raw_first, int(math.ceil((clamped_end / song_duration) * line_count)) - 1),
        )

    first = max(0, raw_first - 1)
    last = min(line_count - 1, raw_last + 1)
    target_center = (raw_first + raw_last) / 2.0

    while last - first + 1 > MAX_LTX_PERFORMANCE_LYRIC_LINES:
        if target_center - first > last - target_center:
            first += 1
        else:
            last -= 1

    # Match the established LTX spoken-word planning ceiling. Complete lines
    # are removed from the farther edge; individual lyric lines are never cut.
    word_budget = max(1, int(math.floor(excerpt_duration * 2.4)))

    def selected_text() -> str:
        return "\n".join(sung_lines[first : last + 1])

    while first < last:
        candidate = selected_text()
        if (
            _lyric_word_count(candidate) <= word_budget
            and len(candidate) <= MAX_LTX_PERFORMANCE_LYRIC_CHARS
        ):
            break
        if target_center - first > last - target_center:
            first += 1
        else:
            last -= 1

    selected = selected_text()
    selected_words = _lyric_word_count(selected)
    if selected_words > word_budget:
        raise ValueError(
            "No complete authored lyric line fits the selected excerpt's "
            f"{word_budget}-word LTX speech budget. Increase the video duration "
            "or use Natural / audio-led sync or Dance / music sync."
        )
    if len(selected) > MAX_LTX_PERFORMANCE_LYRIC_CHARS:
        raise ValueError(
            "No complete authored lyric line fits the LTX lyric prompt character cap."
        )

    metadata = {
        "mapping_method": "uniform_sung_line_proportion_with_one_line_context",
        "alignment": "deterministic_planning_heuristic_not_asr_or_word_alignment",
        "source_lyrics_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_sung_line_count": line_count,
        "raw_proportional_line_range_1_based": [raw_first + 1, raw_last + 1],
        "selected_line_range_1_based": [first + 1, last + 1],
        "selected_line_count": last - first + 1,
        "selected_word_count": selected_words,
        "speech_word_budget": word_budget,
        "selected_character_count": len(selected),
        "selected_lyrics_sha256": hashlib.sha256(selected.encode("utf-8")).hexdigest(),
        "selection_role": "lexical_and_pronunciation_candidates_only",
        "is_timing_schedule": False,
        "requires_every_selected_line_performed": False,
        "section_directives_excluded": True,
        "lyrics_preserved_verbatim_by_complete_line": True,
        "song_duration_seconds": song_duration,
        "requested_excerpt_start_seconds": excerpt_start,
        "requested_excerpt_duration_seconds": excerpt_duration,
        "mapped_excerpt_start_seconds": clamped_start,
        "mapped_excerpt_end_seconds": clamped_end,
    }
    return selected, metadata


class DiffusionGemmaLTXPerformancePrompt:
    """Apply a dance, lyric-reference, or natural audio-led LTX prompt suffix."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ltx_prompt": ("STRING", {"forceInput": True, "multiline": True}),
                "performance_mode": (
                    list(LTX_PERFORMANCE_MODES),
                    {
                        "default": "Natural / audio-led sync",
                        "tooltip": "Dance suppresses visible vocal articulation. Lyrics supplies bounded lexical/pronunciation candidates but never timing. Natural lets connected audio alone govern articulation.",
                    },
                ),
                "song_duration_seconds": (
                    "FLOAT",
                    {"default": 90.0, "min": 0.01, "max": 86400.0, "step": 0.01},
                ),
                "excerpt_start_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.01},
                ),
                "excerpt_duration_seconds": (
                    "FLOAT",
                    {"default": 20.0, "min": 0.01, "max": 86400.0, "step": 0.01},
                ),
            },
            "optional": {
                "lyrics": ("STRING", {"forceInput": True, "multiline": True}),
                "performance_mode_override": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Optional typed upstream source of truth. When connected, this overrides the legacy local dropdown without removing it from older workflows.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "ltx_prompt",
        "selected_lyrics",
        "status",
        "performance_report_json",
        "performance_mode",
    )
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Post-processes a validated LTX prompt without touching the frozen audio guide or final master. "
        "Dance suppresses visible vocal articulation; Lyrics supplies bounded lexical/pronunciation "
        "candidates for verified timing; Natural leaves articulation entirely audio-led."
    )

    def apply(
        self,
        ltx_prompt,
        performance_mode,
        song_duration_seconds,
        excerpt_start_seconds,
        excerpt_duration_seconds,
        lyrics="",
        performance_mode_override="",
    ):
        override = str(performance_mode_override or "").strip()
        mode = _normalize_ltx_performance_mode(override or performance_mode)
        incoming_prompt = str(ltx_prompt or "")
        report: dict[str, Any] = {
            "schema": PERFORMANCE_REPORT_SCHEMA,
            "version": ANALYSIS_VERSION,
            "mode": mode,
            "audio_conditioning_changed": False,
            "final_soundtrack_changed": False,
            "selected_lyrics_are_timing_schedule": False,
            "selected_lyrics_require_every_line_performed": False,
        }
        if not incoming_prompt.strip():
            status = (
                "LTX performance prompt blocked: the upstream generation gate returned an empty prompt."
            )
            report.update(
                {
                    "ready": False,
                    "status": status,
                    "failure_reason": "empty_upstream_prompt",
                    "lyrics_included": False,
                    "selected_lyrics_role": "none",
                    "articulation_timing_authority": "none_generation_blocked",
                }
            )
            return ("", "", status, _json(report), mode)

        if mode == "Dance / music sync":
            selected_lyrics = ""
            directive = (
                "Performance mode — Dance / music sync (authoritative): preserve the connected "
                "soundtrack exactly, but no visible person sings, speaks, mouths words, or lip-syncs. "
                "Any visible performer keeps their lips gently closed, jaw relaxed, and mouth free of "
                "phoneme-shaped articulation throughout. Synchronize full-body choreography, footwork, "
                "weight shifts, gestures, expression, hair and fabric motion, scene changes, and camera "
                "accents to the music's beat, phrasing, dynamics, lulls, and transitions. This final mode "
                "contract overrides any earlier vocal-performance wording."
            )
            status = (
                "Dance / music sync ready: lyric text is excluded and the soundtrack guides body, "
                "scene, and camera timing only."
            )
            report.update(
                {
                    "ready": True,
                    "status": status,
                    "lyrics_included": False,
                    "visible_vocal_articulation": "suppressed",
                    "selected_lyrics_role": "none",
                    "articulation_timing_authority": "dance_mode_suppression",
                    "unverified_timing_fallback": "not_applicable",
                    "non_vocal_or_ambiguous_gap_policy": "visible_vocal_articulation_suppressed",
                }
            )
        elif mode == "Natural / audio-led sync":
            selected_lyrics = ""
            directive = (
                "Performance mode — Natural / audio-led sync (authoritative): preserve the connected "
                "soundtrack exactly. Let the connected audio alone govern whether and when any visible "
                "articulation naturally occurs. Do not pre-script mouth shapes, facial articulation, or "
                "vocal performance. During non-vocal or ambiguous gaps, keep facial behavior natural "
                "and never invent or pantomime words. Synchronize body motion, scene changes, and camera "
                "accents naturally to the audible performance. This final mode contract overrides any "
                "earlier instruction that schedules or suppresses vocal articulation."
            )
            status = (
                "Natural / audio-led sync ready: lyric text is excluded and connected audio alone "
                "governs whether and when natural articulation occurs."
            )
            report.update(
                {
                    "ready": True,
                    "status": status,
                    "lyrics_included": False,
                    "visible_vocal_articulation": "audio_led_not_prescheduled",
                    "selected_lyrics_role": "none",
                    "articulation_timing_authority": "connected_audio_alone",
                    "unverified_timing_fallback": "Natural / audio-led sync",
                    "non_vocal_or_ambiguous_gap_policy": "no_invented_or_pantomimed_words",
                }
            )
        else:
            selected_lyrics, lyric_report = _bounded_proportional_lyric_window(
                lyrics,
                song_duration_seconds=song_duration_seconds,
                excerpt_start_seconds=excerpt_start_seconds,
                excerpt_duration_seconds=excerpt_duration_seconds,
            )
            quoted_lyrics = "\n".join(
                json.dumps(line, ensure_ascii=False)
                for line in selected_lyrics.splitlines()
            )
            directive = (
                "Performance mode — Lyrics + lip sync (authoritative): the connected audio and "
                "downstream verified vocal timing are the only authorities for whether and when visible "
                "articulation occurs. The exact authored lines below are lexical and pronunciation "
                "candidates only: they are not timing evidence, do not schedule articulation, and do "
                "not require every supplied line to be performed. Candidate text never overrides the "
                "audio. Use a candidate word only where it is actually audible in a verified vocal "
                "interval; never infer vocal action from this proportional text selection. Wherever "
                "verified timing is absent, fall back to Natural / audio-led sync and let connected "
                "audio alone govern natural articulation. During instrumental, non-vocal, or ambiguous "
                "gaps, never invent or pantomime words. Never paraphrase, reorder, or vocalize section "
                "labels. Each candidate line is supplied in ordinary straight quotes.\n\nCandidate "
                "authored lyric lines (lexical/pronunciation reference only):\n"
                + quoted_lyrics
            )
            status = (
                "Lyrics + lip sync ready with "
                f"{lyric_report['selected_line_count']} candidate authored lines "
                f"({lyric_report['selected_word_count']} words). They are lexical/pronunciation "
                "reference only, never a schedule or performance checklist. Downstream verified vocal "
                "timing is authoritative; intervals without it fall back to Natural / audio-led sync."
            )
            report.update(
                {
                    "ready": True,
                    "status": status,
                    "lyrics_included": True,
                    "visible_vocal_articulation": "permitted_only_by_verified_timing_and_audio",
                    "selected_lyrics_role": "lexical_and_pronunciation_candidates_only",
                    "articulation_timing_authority": "downstream_verified_vocal_timing_then_connected_audio",
                    "unverified_timing_fallback": "Natural / audio-led sync",
                    "non_vocal_or_ambiguous_gap_policy": "no_invented_or_pantomimed_words",
                    "lyric_window": lyric_report,
                    "lyric_lines_supplied_in_straight_quotes": True,
                }
            )

        # Preserve the validated incoming prompt byte-for-byte as the prefix.
        final_prompt = incoming_prompt + "\n\n" + directive
        report["incoming_prompt_preserved_verbatim"] = final_prompt.startswith(incoming_prompt)
        report["incoming_prompt_character_count"] = len(incoming_prompt)
        report["final_prompt_character_count"] = len(final_prompt)
        return (final_prompt, selected_lyrics, status, _json(report), mode)


class DiffusionGemmaMusicVideoPerformanceMode:
    """Compile one pre-Director performance choice into H3 audio guidance."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_audio_guidance": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                ),
                "performance_mode": (
                    list(LTX_PERFORMANCE_MODES),
                    {"default": "Natural / audio-led sync"},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("performance_mode", "target_audio_guidance", "status")
    FUNCTION = "compile"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Makes Dance/Lyrics/Natural a single pre-Director source of truth. It changes only prompt "
        "guidance, never the selected song, conditioning waveform, or final soundtrack."
    )

    def compile(self, base_audio_guidance, performance_mode):
        mode = _normalize_ltx_performance_mode(performance_mode)
        base = str(base_audio_guidance or "").strip()
        if mode == "Dance / music sync":
            directive = (
                "Performance mode is Dance / music sync. Stage readable full-body choreography and "
                "musical scene accents, but no visible person sings, speaks, mouths words, or "
                "lip-syncs; keep facial motion naturally non-vocal. Do not author H3 <d> spoken-"
                "dialogue blocks."
            )
        elif mode == "Natural / audio-led sync":
            directive = (
                "Performance mode is Natural / audio-led sync. The connected audio alone governs "
                "whether and when natural visible articulation occurs. Do not pre-script mouth or "
                "facial articulation. Non-vocal or ambiguous gaps contain no invented or pantomimed "
                "words. Do not author H3 <d> spoken-dialogue blocks."
            )
        else:
            directive = (
                "Performance mode is Lyrics + lip sync. Do not reserve or command a face to perform "
                "lyrics, and do not author shot-level or timestamped vocal actions. Downstream verified "
                "vocal timing is the sole authority for scheduling visible articulation; selected lyric "
                "text supplied later is lexical/pronunciation reference only. Wherever verified timing "
                "is absent, fall back to Natural / audio-led sync: connected audio alone governs natural "
                "articulation, with no invented or pantomimed words during non-vocal or ambiguous gaps. "
                "Do not author H3 <d> spoken-dialogue blocks."
            )
        guidance = (base + "\n\n" + directive).strip()
        return (
            mode,
            guidance,
            f"{mode} is the shared pre-Director performance contract; audio is unchanged.",
        )


def _reference_mode_token(value: Any) -> str:
    normalized = str(value or "").strip().casefold().replace("_", " ")
    if normalized in {"compose", "compose new"}:
        return "compose_new"
    if normalized in {"cover", "cover reference"}:
        return "cover_reference"
    raise ValueError("ACE reference mode must be Compose new or Cover reference.")


class DiffusionGemmaACEReferenceMode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mode": (list(ACE_REFERENCE_MODES), {"default": "Compose new"})}}

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("generate_audio_codes", "mode_token")
    FUNCTION = "route"
    CATEGORY = CATEGORY
    DESCRIPTION = "Compose generates ACE semantic codes; Cover uses a lazy reference-audio latent."

    def route(self, mode):
        token = _reference_mode_token(mode)
        return (token == "compose_new", token)


class DiffusionGemmaACECoverConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "mode_token": ("STRING", {"forceInput": True}),
            },
            "optional": {"reference_latent": ("LATENT", {"lazy": True})},
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("conditioning", "status", "ready")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "Passes Compose conditioning through, or fail-closed attaches a lazy Cover timbre latent."

    @classmethod
    def check_lazy_status(cls, **kwargs):
        try:
            mode = _reference_mode_token(kwargs.get("mode_token", ""))
        except ValueError:
            return []
        if (
            mode == "cover_reference"
            and "reference_latent" in kwargs
            and kwargs["reference_latent"] is None
        ):
            return ["reference_latent"]
        return []

    def apply(self, conditioning, mode_token, reference_latent=None):
        try:
            mode = _reference_mode_token(mode_token)
        except ValueError as exc:
            return (ExecutionBlocker(None), str(exc), False)
        if mode == "compose_new":
            return (conditioning, "ACE Compose mode: semantic audio-code generation remains enabled.", True)
        if not isinstance(reference_latent, Mapping) or not isinstance(reference_latent.get("samples"), torch.Tensor):
            return (
                ExecutionBlocker(None),
                "ACE Cover mode is blocked because no valid reference audio latent is connected.",
                False,
            )
        try:
            import node_helpers

            routed = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_audio_timbre_latents": [reference_latent["samples"]]},
                append=True,
            )
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            return (ExecutionBlocker(None), f"ACE Cover conditioning is blocked: {exc}", False)
        return (routed, "ACE Cover mode: reference timbre latent attached; semantic code generation should be disabled.", True)


def _resolve_uploaded_song_path(audio_file: Any) -> str:
    """Resolve an uploaded song without allowing reads outside ComfyUI input."""

    try:
        import folder_paths
    except ModuleNotFoundError as exc:  # Standalone tests do not add ComfyUI.
        raise ValueError("ComfyUI's input directory is unavailable.") from exc
    relative_name, annotated_base = folder_paths.annotated_filepath(str(audio_file or ""))
    input_dir = os.path.realpath(folder_paths.get_input_directory())
    if annotated_base is not None and os.path.realpath(annotated_base) != input_dir:
        raise ValueError("Uploaded songs must be selected from ComfyUI's input directory.")
    audio_path = os.path.realpath(os.path.join(input_dir, relative_name))
    try:
        inside_input = os.path.commonpath((input_dir, audio_path)) == input_dir
    except ValueError:
        inside_input = False
    if not inside_input:
        raise ValueError("Uploaded songs must be selected from ComfyUI's input directory.")
    if not str(relative_name).strip() or not os.path.isfile(audio_path):
        raise ValueError("Choose or upload a song before selecting Upload song.")
    return audio_path


def _validate_uploaded_decode_shape(channels: int, samples: int, sample_rate: int) -> None:
    channel_count = int(channels)
    sample_count = int(samples)
    rate = int(sample_rate)
    if channel_count < 1 or channel_count > MAX_UPLOAD_CHANNELS:
        raise ValueError(
            f"Uploaded songs must have from 1 to {MAX_UPLOAD_CHANNELS} audio channels."
        )
    if rate < 1 or rate > MAX_UPLOAD_SAMPLE_RATE:
        raise ValueError(
            f"Uploaded song sample rate must be from 1 to {MAX_UPLOAD_SAMPLE_RATE} Hz."
        )
    if sample_count < 1:
        raise ValueError("Uploaded song contains no decoded samples.")
    duration = sample_count / float(rate)
    if duration > MAX_UPLOAD_DURATION_SECONDS:
        raise ValueError(
            f"Uploaded song exceeds the {MAX_UPLOAD_DURATION_SECONDS:.0f}-second decode limit."
        )
    if channel_count * sample_count > MAX_UPLOAD_SCALAR_SAMPLES:
        raise ValueError("Uploaded song exceeds the bounded decoded-PCM sample limit.")


def _load_uploaded_audio_bounded(audio_path: str) -> tuple[torch.Tensor, int]:
    """Decode with hard duration/channel/sample bounds instead of unbounded accumulation."""

    try:
        import av
        from comfy_extras.nodes_audio import f32_pcm
    except ImportError as exc:
        raise ValueError("ComfyUI's audio decoder is unavailable.") from exc
    try:
        with av.open(audio_path) as audio_file:
            if not audio_file.streams.audio:
                raise ValueError("No audio stream was found in the uploaded file.")
            stream = audio_file.streams.audio[0]
            sample_rate = int(stream.codec_context.sample_rate or 0)
            channels = int(stream.channels or 0)
            _validate_uploaded_decode_shape(channels, 1, sample_rate)
            frames: list[torch.Tensor] = []
            decoded_samples = 0
            for frame in audio_file.decode(streams=stream.index):
                buffer = torch.from_numpy(frame.to_ndarray())
                if buffer.ndim != 2:
                    raise ValueError("Decoded audio frame does not have two dimensions.")
                if int(buffer.shape[0]) != channels:
                    if int(buffer.numel()) % channels:
                        raise ValueError("Decoded audio frame cannot be aligned to its channel count.")
                    buffer = buffer.reshape(-1, channels).t()
                prospective_samples = decoded_samples + int(buffer.shape[1])
                _validate_uploaded_decode_shape(channels, prospective_samples, sample_rate)
                frames.append(buffer)
                decoded_samples = prospective_samples
            if not frames:
                raise ValueError("No audio frames were decoded from the uploaded file.")
            waveform = f32_pcm(torch.cat(frames, dim=1))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    _validate_uploaded_decode_shape(
        int(waveform.shape[0]), int(waveform.shape[1]), sample_rate
    )
    return waveform, sample_rate


class DiffusionGemmaUploadSong:
    """Lazy-safe ComfyUI-input loader for an optional custom soundtrack branch."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths

            input_dir = folder_paths.get_input_directory()
            os.makedirs(input_dir, exist_ok=True)
            files = folder_paths.filter_files_content_types(
                os.listdir(input_dir),
                ["audio", "video"],
            )
        except ModuleNotFoundError:  # Standalone repository tests.
            files = []
        return {
            "required": {
                "audio_file": (
                    [""] + sorted(files),
                    {
                        "audio_upload": True,
                        "tooltip": (
                            "Upload or choose a song from ComfyUI input. The file is decoded only "
                            "when Soundtrack source is set to Upload song."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = (
        "audio",
        "duration_seconds",
        "waveform_sha256",
        "status",
        "ready",
    )
    FUNCTION = "load_audio"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Uploads a custom soundtrack without waking the branch while ACE-Step is selected. "
        "The waveform is returned unchanged after decoding and is measured again by the shared audition/QC node."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, audio_file):
        # Comfy validates dormant lazy branches. Defer file validation until
        # this branch is actually selected so a blank uploader cannot poison
        # an otherwise valid ACE-Step queue.
        return True

    @classmethod
    def IS_CHANGED(cls, audio_file):
        try:
            path = _resolve_uploaded_song_path(audio_file)
            stat = os.stat(path)
            # Comfy includes lazy ancestors in cache signatures even when they
            # do not execute. Avoid SHA-reading a dormant upload on every ACE
            # queue while still invalidating when the selected file changes.
            return (
                f"stat@1:{str(audio_file)}:{int(stat.st_size)}:"
                f"{int(stat.st_mtime_ns)}"
            )
        except (OSError, ValueError):
            return f"unselected:{str(audio_file or '').strip()}"

    def load_audio(self, audio_file):
        audio_path = _resolve_uploaded_song_path(audio_file)
        try:
            waveform, sample_rate = _load_uploaded_audio_bounded(audio_path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"Uploaded song could not be decoded: {exc}") from exc
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}
        decoded, decoded_rate = _audio_parts(audio, "uploaded_audio")
        duration = decoded.shape[-1] / float(decoded_rate)
        digest = waveform_sha256(audio)
        status = (
            f"Uploaded song decoded: {duration:.3f}s at {decoded_rate} Hz; "
            f"waveform {digest[:12]}…."
        )
        return (audio, duration, digest, status, True)


def _normalize_soundtrack_source(value: Any) -> str:
    normalized = str(value or "").strip().casefold().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    aliases = {
        "generate with ace step": "Generate with ACE-Step",
        "ace step": "Generate with ACE-Step",
        "generated candidates": "Generate with ACE-Step",
        "upload song": "Upload song",
        "uploaded song": "Upload song",
    }
    if normalized not in aliases:
        raise ValueError(f"source_mode must be one of {', '.join(SOUNDTRACK_SOURCES)}.")
    return aliases[normalized]


class DiffusionGemmaSongSourceRouter:
    """Lazily choose ACE candidates or one uploaded song before shared QC."""

    @classmethod
    def INPUT_TYPES(cls):
        lazy_force = {"lazy": True, "forceInput": True}
        return {
            "required": {
                "source_mode": (
                    list(SOUNDTRACK_SOURCES),
                    {
                        "default": "Generate with ACE-Step",
                        "tooltip": (
                            "Generate keeps the existing ACE audition. Upload bypasses all ACE audio "
                            "lanes and sends one custom song through the same QC, hash lock, excerpt, "
                            "H3 conditioning, and final mux."
                        ),
                    },
                ),
                "uploaded_expected_bpm": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 300.0,
                        "step": 0.1,
                        "tooltip": "Optional known BPM for the uploaded song. Leave 0 for signal-only tempo analysis.",
                    },
                ),
                "uploaded_lyrics": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Optional exact lyrics for an uploaded vocal song. Leave blank for Dance "
                            "or Natural mode; Lyrics + lip sync requires real sung lyrics."
                        ),
                    },
                ),
            },
            "optional": {
                "generated_candidate_count": ("INT", dict(lazy_force)),
                "generated_expected_bpm": ("FLOAT", dict(lazy_force)),
                "generated_lyrics": (
                    "STRING",
                    {**lazy_force, "multiline": True},
                ),
                "generated_duration_seconds": ("FLOAT", dict(lazy_force)),
                "generated_candidate_1": ("AUDIO", {"lazy": True}),
                "generated_candidate_2": ("AUDIO", {"lazy": True}),
                "generated_candidate_3": ("AUDIO", {"lazy": True}),
                "generated_candidate_4": ("AUDIO", {"lazy": True}),
                "uploaded_audio": ("AUDIO", {"lazy": True}),
                "uploaded_duration_seconds": ("FLOAT", dict(lazy_force)),
                "uploaded_waveform_sha256": ("STRING", dict(lazy_force)),
                "uploaded_status": ("STRING", dict(lazy_force)),
                "uploaded_ready": ("BOOLEAN", dict(lazy_force)),
            },
        }

    RETURN_TYPES = (
        "AUDIO",
        "AUDIO",
        "AUDIO",
        "AUDIO",
        "INT",
        "FLOAT",
        "STRING",
        "FLOAT",
        "STRING",
        "STRING",
        "BOOLEAN",
    )
    RETURN_NAMES = (
        "candidate_1",
        "candidate_2",
        "candidate_3",
        "candidate_4",
        "effective_candidate_count",
        "effective_expected_bpm",
        "selected_lyrics",
        "selected_duration_seconds",
        "source_token",
        "status",
        "ready",
    )
    FUNCTION = "route"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "A lazy soundtrack-origin gate. Upload mode never evaluates ACE candidates; ACE mode never "
        "loads the uploaded file. Both routes converge before the existing decoded-audio QC and hash lock."
    )

    @classmethod
    def check_lazy_status(cls, **kwargs):
        try:
            mode = _normalize_soundtrack_source(kwargs.get("source_mode", "Generate with ACE-Step"))
        except ValueError:
            return []
        if mode == "Upload song":
            names = (
                "uploaded_audio",
                "uploaded_duration_seconds",
                "uploaded_waveform_sha256",
                "uploaded_status",
                "uploaded_ready",
            )
            return [name for name in names if name in kwargs and kwargs[name] is None]

        metadata_names = (
            "generated_candidate_count",
            "generated_expected_bpm",
            "generated_lyrics",
            "generated_duration_seconds",
        )
        pending = [name for name in metadata_names if name in kwargs and kwargs[name] is None]
        if pending:
            return pending
        try:
            count = int(kwargs.get("generated_candidate_count"))
        except (TypeError, ValueError):
            return []
        if count < 1 or count > 4:
            return []
        names = [f"generated_candidate_{index}" for index in range(1, count + 1)]
        return [name for name in names if name in kwargs and kwargs[name] is None]

    def route(
        self,
        source_mode,
        uploaded_expected_bpm,
        uploaded_lyrics,
        generated_candidate_count=None,
        generated_expected_bpm=None,
        generated_lyrics=None,
        generated_duration_seconds=None,
        generated_candidate_1=None,
        generated_candidate_2=None,
        generated_candidate_3=None,
        generated_candidate_4=None,
        uploaded_audio=None,
        uploaded_duration_seconds=None,
        uploaded_waveform_sha256=None,
        uploaded_status=None,
        uploaded_ready=None,
    ):
        mode = _normalize_soundtrack_source(source_mode)
        if mode == "Upload song":
            if uploaded_audio is None:
                raise ValueError("Upload song is selected, but no song is connected. Upload or choose a song and queue again.")
            waveform, sample_rate = _audio_parts(uploaded_audio, "uploaded_audio")
            duration = waveform.shape[-1] / float(sample_rate)
            if not math.isfinite(duration) or duration <= 0.0:
                raise ValueError("The uploaded song must have a positive finite duration.")
            if uploaded_duration_seconds is not None:
                claimed_duration = float(uploaded_duration_seconds)
                if not math.isfinite(claimed_duration) or abs(claimed_duration - duration) > 1.0e-6:
                    raise ValueError("The uploaded song duration metadata does not match its decoded waveform.")
            expected_bpm = float(uploaded_expected_bpm)
            if not math.isfinite(expected_bpm) or not 0.0 <= expected_bpm <= 300.0:
                raise ValueError("uploaded_expected_bpm must be a finite value from 0 to 300.")
            if uploaded_ready is False:
                raise ValueError(str(uploaded_status or "The uploaded song loader is not ready."))
            supplied_hash = str(uploaded_waveform_sha256 or "").strip().casefold()
            if supplied_hash and not _SHA256_RE.fullmatch(supplied_hash):
                raise ValueError("The uploaded waveform SHA-256 metadata is invalid.")
            if supplied_hash:
                actual_hash = waveform_sha256(uploaded_audio)
                if supplied_hash != actual_hash:
                    raise ValueError(
                        "The uploaded waveform SHA-256 metadata does not match its decoded audio."
                    )
            lyrics = str(uploaded_lyrics or "").strip()
            candidates = (uploaded_audio,) * 4
            bpm_status = f"known BPM {expected_bpm:.1f}" if expected_bpm > 0.0 else "signal-only BPM analysis"
            status = (
                f"Upload song selected: {duration:.3f}s, {bpm_status}; ACE audio generation is dormant. "
                "The uploaded waveform now enters shared decoded-audio QC and hash locking."
            )
            return (*candidates, 1, expected_bpm, lyrics, duration, "uploaded_song", status, True)

        if generated_candidate_count is None:
            raise ValueError("ACE-Step source is selected, but generated_candidate_count is not connected.")
        count = int(generated_candidate_count)
        if count < 1 or count > 4:
            raise ValueError("generated_candidate_count must be from 1 to 4.")
        if generated_expected_bpm is None:
            raise ValueError("ACE-Step source is selected, but generated_expected_bpm is not connected.")
        expected_bpm = float(generated_expected_bpm)
        if not math.isfinite(expected_bpm) or not 0.0 <= expected_bpm <= 300.0:
            raise ValueError("generated_expected_bpm must be a finite value from 0 to 300.")
        if generated_duration_seconds is None:
            raise ValueError("ACE-Step source is selected, but generated_duration_seconds is not connected.")
        duration = float(generated_duration_seconds)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("generated_duration_seconds must be positive and finite.")
        generated = [
            generated_candidate_1,
            generated_candidate_2,
            generated_candidate_3,
            generated_candidate_4,
        ]
        missing = [index for index in range(1, count + 1) if generated[index - 1] is None]
        if missing:
            raise ValueError(
                "ACE-Step source expected generated candidate lanes: "
                + ", ".join(map(str, missing))
                + "."
            )
        first = generated[0]
        candidates = tuple(value if value is not None else first for value in generated)
        lyrics = str(generated_lyrics or "").strip()
        status = (
            f"ACE-Step source selected: {count} decoded candidate{'s' if count != 1 else ''}; "
            "the upload loader is dormant and shared decoded-audio QC remains authoritative."
        )
        return (*candidates, count, expected_bpm, lyrics, duration, "ace_step", status, True)


def _splatstage_seed(root_seed: int, kind: str) -> int:
    message = f"splatstage-autodirect@1|{int(root_seed)}|{kind}|0"
    return int.from_bytes(
        hashlib.blake2b(message.encode("utf-8"), digest_size=8).digest(),
        "big",
    ) & 0x7FFF_FFFF_FFFF_FFFF


def _audition_seed(root_seed: int, index: int) -> int:
    return _splatstage_seed(root_seed, f"music:audition:{int(index)}")


class DiffusionGemmaSongSeedFanout:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root_seed": (
                    "INT",
                    {"default": 26073001, "min": 0, "max": 0x7FFF_FFFF_FFFF_FFFF},
                ),
                "production_concept": (
                    list(PRODUCTION_CONCEPTS),
                    {"default": "Source passthrough"},
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "candidate_1_seed",
        "candidate_2_seed",
        "candidate_3_seed",
        "candidate_4_seed",
        "seed_report_json",
    )
    FUNCTION = "fanout"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Derives ACE song seeds that exactly match the SplatStage Router: one music seed for "
        "Source/Joint production, or four zero-based audition seeds."
    )

    def fanout(self, root_seed, production_concept):
        root = int(root_seed)
        concept = str(production_concept)
        if concept not in PRODUCTION_CONCEPTS:
            raise ValueError(f"production_concept must be one of {', '.join(PRODUCTION_CONCEPTS)}.")
        audition_seeds = [_audition_seed(root, index) for index in range(4)]
        if concept == "Audition and select":
            seeds = audition_seeds
            effective_seeds = seeds
            derivation = "splatstage-autodirect@1|{root}|music:audition:{zero_based_index}|0"
            audition_indices: list[int] = [0, 1, 2, 3]
        else:
            music_seed = _splatstage_seed(root, "music")
            # Only lane 1 is requested when candidate_count=1. Repeating the
            # one production seed on dormant outputs avoids implying that
            # Source/Joint mode secretly authored additional candidates.
            seeds = [music_seed] * 4
            effective_seeds = [music_seed]
            derivation = "splatstage-autodirect@1|{root}|music|0"
            audition_indices = []
        report = {
            "schema": "diffusiongemma.song_seed_fanout",
            "version": 1,
            "root_seed": root,
            "production_concept": concept,
            "derivation": derivation,
            "candidate_seeds": effective_seeds,
            "output_lane_seeds": seeds,
            "audition_indices": audition_indices,
            "independent_from_ltx_seed": True,
            "matches_splatstage_router": True,
        }
        return (*seeds, _json(report))


class DiffusionGemmaMusicProductionConcept:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "production_concept": (
                    list(PRODUCTION_CONCEPTS),
                    {"default": "Source passthrough"},
                ),
                "audition_candidate_count": (
                    "INT",
                    {"default": 2, "min": 2, "max": 4, "step": 1},
                ),
            }
        }

    # Legacy list-valued combo inputs (Planner) require the concrete option
    # list, while V3 COMBO inputs (Router) explicitly accept list outputs.
    RETURN_TYPES = (list(PRODUCTION_CONCEPTS), list(PRODUCTION_CONCEPTS), "INT")
    RETURN_NAMES = ("planner_mode", "router_mode", "candidate_count")
    FUNCTION = "route"
    CATEGORY = CATEGORY
    DESCRIPTION = "One shared production choice for Planner and Router; this node does not mutate music."

    def route(self, production_concept, audition_candidate_count):
        value = str(production_concept)
        if value not in PRODUCTION_CONCEPTS:
            raise ValueError(f"production_concept must be one of {', '.join(PRODUCTION_CONCEPTS)}.")
        audition_count = int(audition_candidate_count)
        if audition_count < 2 or audition_count > 4:
            raise ValueError("audition_candidate_count must be from 2 to 4.")
        effective_count = audition_count if value == "Audition and select" else 1
        return (value, value, effective_count)


NODE_CLASS_MAPPINGS = {
    "DiffusionGemmaAudioCandidateSelector": DiffusionGemmaAudioCandidateSelector,
    "DiffusionGemmaUploadSong": DiffusionGemmaUploadSong,
    "DiffusionGemmaSongSourceRouter": DiffusionGemmaSongSourceRouter,
    "DiffusionGemmaLTXAudioGuide": DiffusionGemmaLTXAudioGuide,
    "DiffusionGemmaLTXPerformancePrompt": DiffusionGemmaLTXPerformancePrompt,
    "DiffusionGemmaMusicVideoPerformanceMode": DiffusionGemmaMusicVideoPerformanceMode,
    "DiffusionGemmaACEReferenceMode": DiffusionGemmaACEReferenceMode,
    "DiffusionGemmaACECoverConditioning": DiffusionGemmaACECoverConditioning,
    "DiffusionGemmaSongSeedFanout": DiffusionGemmaSongSeedFanout,
    "DiffusionGemmaMusicProductionConcept": DiffusionGemmaMusicProductionConcept,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionGemmaAudioCandidateSelector": "DiffusionGemma Decoded Song Audition & Lock",
    "DiffusionGemmaUploadSong": "DiffusionGemma Upload Song",
    "DiffusionGemmaSongSourceRouter": "DiffusionGemma Soundtrack Source",
    "DiffusionGemmaLTXAudioGuide": "DiffusionGemma LTX Sync-Safe Audio Guide",
    "DiffusionGemmaLTXPerformancePrompt": "DiffusionGemma LTX Performance Prompt",
    "DiffusionGemmaMusicVideoPerformanceMode": "DiffusionGemma Music-Video Performance Mode",
    "DiffusionGemmaACEReferenceMode": "DiffusionGemma ACE Compose / Cover Mode",
    "DiffusionGemmaACECoverConditioning": "DiffusionGemma ACE Cover Conditioning Gate",
    "DiffusionGemmaSongSeedFanout": "DiffusionGemma Song Audition Seed Fanout",
    "DiffusionGemmaMusicProductionConcept": "DiffusionGemma Music Production Concept",
}
