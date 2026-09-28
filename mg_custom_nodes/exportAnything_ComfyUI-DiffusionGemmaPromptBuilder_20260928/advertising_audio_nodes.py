# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

"""Advertisement-specific decoded-audio selection and deterministic mixing.

The existing music-video selector intentionally remains untouched.  This
module reuses its CPU waveform analysis and all ordinary quality gates, while
making the one previously implicit vocal-presence gate explicit in a typed
advertisement soundtrack contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

import torch
import torch.nn.functional as torch_functional

try:  # Loaded as a ComfyUI package.
    from . import audio_production_nodes as legacy_audio
    from .advertising_contract_nodes import (
        SOUNDTRACK_SCHEMA,
        VERSION,
        _strict_json_object,
        _validate_schema,
    )
except ImportError:  # Standalone repository tests.
    import audio_production_nodes as legacy_audio
    from advertising_contract_nodes import (
        SOUNDTRACK_SCHEMA,
        VERSION,
        _strict_json_object,
        _validate_schema,
    )


CATEGORY = "prompt/diffusiongemma/advertising"
SELECTION_SCHEMA = "diffusiongemma.advertisement_audio_selection"
MIX_SCHEMA = "diffusiongemma.advertisement_audio_mix"
SELECTION_VERSION = 1
MIX_VERSION = 1
CLIPPING_POLICIES = ("Attenuate mix to ceiling", "Fail on clipping")
SOUNDTRACK_SOURCE_MODES = ("MiniMax Music 3", "Upload song", "Legacy ACE-Step")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _parse_soundtrack_contract(value: Any) -> dict[str, Any]:
    payload = _strict_json_object(value, "soundtrack_contract_json")
    _validate_schema(payload, "advertisement_soundtrack.schema.json", "soundtrack contract")
    if (
        payload.get("schema") != SOUNDTRACK_SCHEMA
        or payload.get("version") != VERSION
        or not payload.get("ready")
    ):
        raise ValueError("soundtrack_contract_json is not a ready supported advertisement soundtrack contract.")
    return payload


def _content_mode(contract: Mapping[str, Any]) -> str:
    mode = str(contract.get("effective_content_mode", ""))
    if mode not in {"instrumental", "vocal"}:
        raise ValueError("soundtrack contract effective content mode is invalid.")
    return mode


def _normalize_ad_source_policy(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().casefold()).strip("_")
    aliases = {
        "": "minimax_music3",
        "minimax_music3": "minimax_music3",
        "music3": "minimax_music3",
        "minimax_3": "minimax_music3",
        "ace": "ace_step",
        "ace_step": "ace_step",
        "legacy_ace": "ace_step",
        "upload": "uploaded_song",
        "upload_song": "uploaded_song",
        "uploaded_song": "uploaded_song",
    }
    if normalized not in aliases:
        raise ValueError("source_policy must be minimax_music3, uploaded_song, or ace_step.")
    return aliases[normalized]


def _legacy_qc_policy(source_policy: str) -> str:
    return "uploaded_song" if source_policy == "uploaded_song" else "ace_step"


def _safe_nonvocal_double_time(analysis: Mapping[str, Any]) -> bool:
    if str(analysis.get("tempo_relation", "")) != "double_time_detected":
        return False
    relative_error = analysis.get("canonical_tempo_relative_error")
    excerpt = analysis.get("suggested_excerpt")
    if not isinstance(relative_error, (int, float)) or not isinstance(excerpt, Mapping):
        return False
    if not math.isfinite(float(relative_error)):
        return False
    supporting = (
        float(excerpt.get("low_density_visual_recovery_coverage", 0.0))
        >= legacy_audio.MIN_SAFE_DOUBLE_TIME_VISUAL_RECOVERY_COVERAGE,
        float(excerpt.get("score", 0.0))
        >= legacy_audio.MIN_SAFE_DOUBLE_TIME_EXCERPT_SCORE,
        float(excerpt.get("tonal_family_consistency", 0.0))
        >= legacy_audio.MIN_SAFE_DOUBLE_TIME_TONAL_FAMILY_CONSISTENCY,
        float(excerpt.get("tonal_transition_stability", 0.0))
        >= legacy_audio.MIN_SAFE_DOUBLE_TIME_TONAL_TRANSITION_STABILITY,
    )
    return bool(
        float(relative_error) <= legacy_audio.MAX_CANONICAL_DOUBLE_TIME_RELATIVE_ERROR
        and float(excerpt.get("onset_density_per_second", math.inf))
        <= legacy_audio.MAX_SAFE_DOUBLE_TIME_EXCERPT_ONSETS_PER_SECOND
        and sum(bool(value) for value in supporting)
        >= legacy_audio.MIN_SAFE_DOUBLE_TIME_SUPPORTING_SIGNAL_COUNT
    )


def _requested_tempo_mismatch(analysis: Mapping[str, Any]) -> bool:
    expected = analysis.get("expected_bpm")
    relative_error = analysis.get("canonical_tempo_relative_error")
    return bool(
        isinstance(expected, (int, float))
        and math.isfinite(float(expected))
        and float(expected) > 0.0
        and isinstance(relative_error, (int, float))
        and math.isfinite(float(relative_error))
        and float(relative_error) > 0.10
    )


def _content_aware_failures(
    analysis: Mapping[str, Any],
    *,
    excerpt_duration_seconds: float,
    minimum_score: float,
    source_policy: str,
    content_mode: str,
) -> list[str]:
    failures = list(
        legacy_audio._candidate_failures(
            analysis,
            excerpt_duration_seconds=excerpt_duration_seconds,
            minimum_score=minimum_score,
            source_policy=_legacy_qc_policy(source_policy),
        )
    )
    excerpt = analysis.get("suggested_excerpt")
    vocal_proxy = bool(
        isinstance(excerpt, Mapping) and excerpt.get("likely_vocal_active_proxy")
    )
    if content_mode == "instrumental":
        failures = [
            code
            for code in failures
            if code != "no_non_silent_vocal_active_proxy_window"
        ]
        if (
            "double_time_outside_low_pressure_contract" in failures
            and _safe_nonvocal_double_time(analysis)
        ):
            failures.remove("double_time_outside_low_pressure_contract")
    elif content_mode == "vocal" and not vocal_proxy:
        if "no_non_silent_vocal_active_proxy_window" not in failures:
            failures.append("no_non_silent_vocal_active_proxy_window")
    else:  # pragma: no cover - contract validation constrains this.
        raise ValueError("Unsupported advertisement soundtrack content mode.")
    return failures


def _content_advisories(
    analysis: Mapping[str, Any],
    *,
    source_policy: str,
    minimum_score: float,
    content_mode: str,
) -> list[str]:
    advisories = list(
        legacy_audio._candidate_advisories(
            analysis,
            source_policy=_legacy_qc_policy(source_policy),
            minimum_score=minimum_score,
        )
    )
    excerpt = analysis.get("suggested_excerpt")
    vocal_proxy = bool(
        isinstance(excerpt, Mapping) and excerpt.get("likely_vocal_active_proxy")
    )
    if content_mode == "instrumental":
        advisories.append("instrumental_contract_vocal_proxy_not_required")
        if vocal_proxy:
            advisories.append("instrumental_contract_possible_midband_vocal_proxy_not_proof")
        if _safe_nonvocal_double_time(analysis):
            advisories.append("instrumental_low_pressure_double_time_without_vocal_requirement")
    else:
        advisories.append("vocal_contract_vocal_proxy_required")
    if source_policy == "minimax_music3":
        advisories.append("generated_source_minimax_music3")
    elif source_policy == "ace_step":
        advisories.append("generated_source_legacy_ace_step")
    if _requested_tempo_mismatch(analysis):
        advisories.append("requested_tempo_mismatch_signal_estimate")
    return list(dict.fromkeys(advisories))


def _selection_failure(report: dict[str, Any], message: str):
    report["ready"] = False
    report["status"] = message
    blocker = legacy_audio.ExecutionBlocker(message)
    encoded = _json(report)
    return (blocker, 0.0, blocker, blocker, encoded, message, False)


def _candidate_summary(entry: Mapping[str, Any]) -> dict[str, Any]:
    analysis = entry.get("analysis")
    summary: dict[str, Any] = {
        "index": int(entry.get("index", 0)),
        "present": bool(entry.get("present")),
        "passed": bool(entry.get("passed")),
        "failures": list(entry.get("failures", [])),
        "advisories": list(entry.get("advisories", [])),
    }
    if isinstance(analysis, Mapping):
        excerpt = analysis.get("suggested_excerpt")
        summary["analysis"] = {
            "waveform_sha256": str(analysis.get("waveform_sha256", "")),
            "duration_seconds": float(analysis.get("duration_seconds", 0.0)),
            "production_score": float(analysis.get("production_score", 0.0)),
            "tempo_relation": str(analysis.get("tempo_relation", "")),
            "expected_bpm": float(analysis.get("expected_bpm", 0.0)),
            "detected_bpm": float(analysis.get("detected_bpm", 0.0)),
            "canonical_tempo_relative_error": (
                float(analysis["canonical_tempo_relative_error"])
                if isinstance(analysis.get("canonical_tempo_relative_error"), (int, float))
                and math.isfinite(float(analysis["canonical_tempo_relative_error"]))
                else None
            ),
            "requested_tempo_match_within_10_percent": (
                not _requested_tempo_mismatch(analysis)
                if float(analysis.get("expected_bpm", 0.0)) > 0.0
                and isinstance(analysis.get("canonical_tempo_relative_error"), (int, float))
                else None
            ),
            "likely_vocal_active_proxy": bool(
                isinstance(excerpt, Mapping)
                and excerpt.get("likely_vocal_active_proxy")
            ),
            "suggested_excerpt_start_seconds": float(
                excerpt.get("start_seconds", 0.0)
                if isinstance(excerpt, Mapping)
                else 0.0
            ),
            "suggested_excerpt_score": float(
                excerpt.get("score", 0.0)
                if isinstance(excerpt, Mapping)
                else 0.0
            ),
        }
    if entry.get("error"):
        summary["error"] = str(entry["error"])[:500]
    return summary


class DiffusionGemmaAdvertisementSoundtrackSourceRouter:
    """Lazily select MiniMax Music 3, upload, or legacy ACE audio."""

    @classmethod
    def INPUT_TYPES(cls):
        lazy = {"lazy": True, "forceInput": True}
        return {
            "required": {
                "source_mode": (
                    list(SOUNDTRACK_SOURCE_MODES),
                    {
                        "default": "MiniMax Music 3",
                        "tooltip": "Only the selected source branch is requested. All routes converge before advertisement content-aware QC.",
                    },
                )
            },
            "optional": {
                "music3_candidate_count": ("INT", dict(lazy)),
                "music3_expected_bpm": ("FLOAT", dict(lazy)),
                "music3_lyrics": ("STRING", {**lazy, "multiline": True}),
                "music3_duration_seconds": ("FLOAT", dict(lazy)),
                "music3_candidate_1": ("AUDIO", {"lazy": True}),
                "music3_candidate_2": ("AUDIO", {"lazy": True}),
                "music3_candidate_3": ("AUDIO", {"lazy": True}),
                "music3_candidate_4": ("AUDIO", {"lazy": True}),
                "uploaded_audio": ("AUDIO", {"lazy": True}),
                "uploaded_duration_seconds": ("FLOAT", dict(lazy)),
                "uploaded_expected_bpm": ("FLOAT", dict(lazy)),
                "uploaded_lyrics": ("STRING", {**lazy, "multiline": True}),
                "uploaded_waveform_sha256": ("STRING", dict(lazy)),
                "uploaded_status": ("STRING", dict(lazy)),
                "uploaded_ready": ("BOOLEAN", dict(lazy)),
                "ace_candidate_count": ("INT", dict(lazy)),
                "ace_expected_bpm": ("FLOAT", dict(lazy)),
                "ace_lyrics": ("STRING", {**lazy, "multiline": True}),
                "ace_duration_seconds": ("FLOAT", dict(lazy)),
                "ace_candidate_1": ("AUDIO", {"lazy": True}),
                "ace_candidate_2": ("AUDIO", {"lazy": True}),
                "ace_candidate_3": ("AUDIO", {"lazy": True}),
                "ace_candidate_4": ("AUDIO", {"lazy": True}),
            },
        }

    RETURN_TYPES = (
        "AUDIO", "AUDIO", "AUDIO", "AUDIO", "INT", "FLOAT", "STRING",
        "FLOAT", "STRING", "STRING", "BOOLEAN",
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
        "source_policy",
        "status",
        "ready",
    )
    FUNCTION = "route"
    CATEGORY = CATEGORY
    DESCRIPTION = "A lazy three-way ad soundtrack router. MiniMax Music 3 is the default generated source, upload is source-locked, and legacy ACE remains available without waking dormant branches."

    @classmethod
    def check_lazy_status(cls, **kwargs):
        mode = str(kwargs.get("source_mode", "MiniMax Music 3"))
        if mode == "Upload song":
            names = (
                "uploaded_audio",
                "uploaded_duration_seconds",
                "uploaded_expected_bpm",
                "uploaded_lyrics",
                "uploaded_waveform_sha256",
                "uploaded_status",
                "uploaded_ready",
            )
            return [name for name in names if name in kwargs and kwargs[name] is None]
        prefix = "ace" if mode == "Legacy ACE-Step" else "music3"
        metadata = (
            f"{prefix}_candidate_count",
            f"{prefix}_expected_bpm",
            f"{prefix}_lyrics",
            f"{prefix}_duration_seconds",
        )
        pending = [name for name in metadata if name in kwargs and kwargs[name] is None]
        if pending:
            return pending
        try:
            count = int(kwargs.get(f"{prefix}_candidate_count"))
        except (TypeError, ValueError):
            return []
        if not 1 <= count <= 4:
            return []
        names = [f"{prefix}_candidate_{index}" for index in range(1, count + 1)]
        return [name for name in names if name in kwargs and kwargs[name] is None]

    @staticmethod
    def _generated(prefix: str, policy: str, kwargs: Mapping[str, Any]):
        count_value = kwargs.get(f"{prefix}_candidate_count")
        bpm_value = kwargs.get(f"{prefix}_expected_bpm")
        duration_value = kwargs.get(f"{prefix}_duration_seconds")
        if count_value is None or bpm_value is None or duration_value is None:
            raise ValueError(f"The selected {prefix} source is missing candidate count, BPM, or duration metadata.")
        count = int(count_value)
        if not 1 <= count <= 4:
            raise ValueError(f"{prefix}_candidate_count must be from 1 to 4.")
        bpm = float(bpm_value)
        duration = float(duration_value)
        if not math.isfinite(bpm) or not 0.0 <= bpm <= 300.0:
            raise ValueError(f"{prefix}_expected_bpm must be finite from 0 to 300.")
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"{prefix}_duration_seconds must be positive and finite.")
        generated = [kwargs.get(f"{prefix}_candidate_{index}") for index in range(1, 5)]
        missing = [index for index in range(1, count + 1) if generated[index - 1] is None]
        if missing:
            raise ValueError(
                f"The selected {prefix} source is missing candidate lane(s): {', '.join(map(str, missing))}."
            )
        first = generated[0]
        candidates = tuple(value if value is not None else first for value in generated)
        lyrics = str(kwargs.get(f"{prefix}_lyrics") or "").strip()
        label = "MiniMax Music 3" if policy == "minimax_music3" else "legacy ACE-Step"
        status = (
            f"{label} selected with {count} candidate{'s' if count != 1 else ''}; "
            "upload and the other generated soundtrack branch remain dormant."
        )
        return (*candidates, count, bpm, lyrics, duration, policy, status, True)

    def route(self, source_mode, **kwargs):
        mode = str(source_mode)
        if mode not in SOUNDTRACK_SOURCE_MODES:
            raise ValueError("source_mode is unsupported.")
        if mode == "MiniMax Music 3":
            return self._generated("music3", "minimax_music3", kwargs)
        if mode == "Legacy ACE-Step":
            return self._generated("ace", "ace_step", kwargs)

        audio = kwargs.get("uploaded_audio")
        if audio is None:
            raise ValueError("Upload song is selected, but uploaded_audio is not connected.")
        waveform, sample_rate = _audio_parts(audio, "uploaded_audio")
        duration = int(waveform.shape[-1]) / float(sample_rate)
        claimed_duration = kwargs.get("uploaded_duration_seconds")
        if claimed_duration is not None and abs(float(claimed_duration) - duration) > 0.05:
            raise ValueError("uploaded_duration_seconds does not match the decoded waveform.")
        bpm = float(kwargs.get("uploaded_expected_bpm") or 0.0)
        if not math.isfinite(bpm) or not 0.0 <= bpm <= 300.0:
            raise ValueError("uploaded_expected_bpm must be finite from 0 to 300.")
        supplied_hash = str(kwargs.get("uploaded_waveform_sha256") or "").strip().casefold()
        actual_hash = legacy_audio.waveform_sha256(audio)
        if supplied_hash and (not _SHA256_RE.fullmatch(supplied_hash) or supplied_hash != actual_hash):
            raise ValueError("uploaded_waveform_sha256 does not match the decoded waveform.")
        if kwargs.get("uploaded_ready") is False:
            raise ValueError(str(kwargs.get("uploaded_status") or "The uploaded soundtrack is not ready."))
        lyrics = str(kwargs.get("uploaded_lyrics") or "").strip()
        status = (
            f"Uploaded soundtrack selected at {duration:.3f}s; all generated soundtrack branches remain dormant, "
            f"and waveform {actual_hash[:12]}… enters content-aware QC."
        )
        return (audio, audio, audio, audio, 1, bpm, lyrics, duration, "uploaded_song", status, True)


class DiffusionGemmaAdvertisementAudioCandidateSelector:
    """Select a locked soundtrack with content-aware vocal QC."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "soundtrack_contract_json": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                ),
                "candidate_count": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "selection_mode": (list(legacy_audio.SELECTION_MODES), {"default": "auto_select"}),
                "expected_bpm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
                "excerpt_duration_seconds": ("FLOAT", {"default": 30.0, "min": 0.5, "max": 600.0, "step": 0.1}),
                "minimum_score": ("FLOAT", {"default": 0.52, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "locked_waveform_sha256": ("STRING", {"default": "", "multiline": False}),
                "locked_start_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2000.0, "step": 0.1}),
                "candidate_1": ("AUDIO", {"lazy": True}),
                "candidate_2": ("AUDIO", {"lazy": True}),
                "candidate_3": ("AUDIO", {"lazy": True}),
                "candidate_4": ("AUDIO", {"lazy": True}),
                "source_policy": (
                    "STRING",
                    {
                        "default": "minimax_music3",
                        "forceInput": True,
                        "tooltip": "ace_step retains generated-candidate musical QC; uploaded_song retains technical integrity gates plus the explicit vocal-content contract.",
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
    DESCRIPTION = "Run existing decoded-audio QC with an explicit advertisement content contract: instrumental music does not require a vocal proxy, while vocal music does."

    @classmethod
    def check_lazy_status(cls, **kwargs):
        try:
            count = max(1, min(4, int(kwargs.get("candidate_count", 1))))
            mode = legacy_audio._normalize_selection_mode(kwargs.get("selection_mode", "auto_select"))
        except (TypeError, ValueError):
            return []
        lock_match = re.fullmatch(r"lock_candidate_([1-4])", mode)
        names = (
            [f"candidate_{int(lock_match.group(1))}"]
            if lock_match is not None and int(lock_match.group(1)) <= count
            else [f"candidate_{index}" for index in range(1, count + 1)]
        )
        return [name for name in names if name in kwargs and kwargs[name] is None]

    def select(
        self,
        soundtrack_contract_json,
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
        source_policy="minimax_music3",
    ):
        contract = _parse_soundtrack_contract(soundtrack_contract_json)
        content_mode = _content_mode(contract)
        count = int(candidate_count)
        if not 1 <= count <= 4:
            raise ValueError("candidate_count must be from 1 to 4.")
        mode = legacy_audio._normalize_selection_mode(selection_mode)
        policy = _normalize_ad_source_policy(source_policy)
        expected = float(expected_bpm)
        duration = float(excerpt_duration_seconds)
        threshold = float(minimum_score)
        locked_start = float(locked_start_seconds)
        if not math.isfinite(expected) or not 0.0 <= expected <= 300.0:
            raise ValueError("expected_bpm must be finite from 0 to 300.")
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("excerpt_duration_seconds must be positive and finite.")
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_score must be finite from 0 to 1.")
        if not math.isfinite(locked_start) or locked_start < -1.0:
            raise ValueError("locked_start_seconds must be -1 or non-negative.")
        lock_hash = str(locked_waveform_sha256 or "").strip().casefold()
        lock_match = re.fullmatch(r"lock_candidate_([1-4])", mode)
        indices = [int(lock_match.group(1))] if lock_match is not None else list(range(1, count + 1))
        report: dict[str, Any] = {
            "schema": SELECTION_SCHEMA,
            "version": SELECTION_VERSION,
            "settings": {
                "candidate_count": count,
                "selection_mode": mode,
                "expected_bpm": expected,
                "excerpt_duration_seconds": duration,
                "minimum_score": threshold,
                "source_policy": policy,
                "soundtrack_content_mode": content_mode,
                "content_evidence_policy": contract["content_evidence_policy"],
            },
            "lock": {
                "active": mode != "auto_select",
                "requested_waveform_sha256": lock_hash,
                "requested_start_seconds": locked_start,
                "satisfied": False,
                "failure_reason": "",
            },
            "candidates": [],
            "selected_candidate_index": None,
            "selected_waveform_sha256": "",
            "ready": False,
        }
        if lock_match is not None and indices[0] > count:
            report["lock"]["failure_reason"] = "locked_candidate_outside_candidate_count"
            return _selection_failure(report, f"Audio selection is locked to candidate {indices[0]}, outside candidate_count={count}.")
        if mode == "lock_by_hash" and not _SHA256_RE.fullmatch(lock_hash):
            report["lock"]["failure_reason"] = "invalid_locked_waveform_sha256"
            return _selection_failure(report, "Audio selection is locked by hash, but no valid SHA-256 was supplied.")

        candidates = [candidate_1, candidate_2, candidate_3, candidate_4]
        measured: dict[int, dict[str, Any]] = {}
        missing: list[int] = []
        for index in indices:
            audio = candidates[index - 1]
            if audio is None:
                entry = {
                    "index": index,
                    "present": False,
                    "passed": False,
                    "failures": ["candidate_missing"],
                    "advisories": [],
                }
                missing.append(index)
            else:
                try:
                    analysis = legacy_audio.analyze_decoded_audio(
                        audio,
                        expected_bpm=expected,
                        excerpt_duration_seconds=duration,
                        locked_start_seconds=(
                            locked_start
                            if mode != "auto_select" and locked_start >= 0.0
                            else None
                        ),
                    )
                    failures = _content_aware_failures(
                        analysis,
                        excerpt_duration_seconds=duration,
                        minimum_score=threshold,
                        source_policy=policy,
                        content_mode=content_mode,
                    )
                    entry = {
                        "index": index,
                        "present": True,
                        "passed": not failures,
                        "failures": failures,
                        "advisories": _content_advisories(
                            analysis,
                            source_policy=policy,
                            minimum_score=threshold,
                            content_mode=content_mode,
                        ),
                        "analysis": analysis,
                    }
                except (RuntimeError, TypeError, ValueError) as exc:
                    entry = {
                        "index": index,
                        "present": True,
                        "passed": False,
                        "failures": ["analysis_error"],
                        "advisories": [],
                        "error": str(exc),
                    }
            measured[index] = entry
            report["candidates"].append(_candidate_summary(entry))

        if mode in {"auto_select", "lock_by_hash"} and missing:
            report["lock"]["failure_reason"] = "candidate_lane_missing" if mode == "lock_by_hash" else ""
            return _selection_failure(report, f"Advertisement audio audition is missing candidate lane(s): {', '.join(map(str, missing))}.")

        selected_index: int | None = None
        if lock_match is not None:
            selected_index = indices[0]
            entry = measured[selected_index]
            if not entry.get("passed"):
                report["lock"]["failure_reason"] = "locked_candidate_failed_qc"
                return _selection_failure(
                    report,
                    f"Locked candidate {selected_index} failed advertisement audio QC: "
                    + ", ".join(entry.get("failures", []))
                    + ".",
                )
        elif mode == "lock_by_hash":
            matches = [
                index
                for index, entry in measured.items()
                if entry.get("analysis", {}).get("waveform_sha256") == lock_hash
            ]
            if not matches:
                report["lock"]["failure_reason"] = "locked_hash_not_present"
                return _selection_failure(report, "The locked waveform hash is not present among advertisement audio candidates.")
            selected_index = min(matches)
            if not measured[selected_index].get("passed"):
                report["lock"]["failure_reason"] = "locked_hash_failed_qc"
                return _selection_failure(report, "The locked waveform is present but failed advertisement audio QC.")
        else:
            passing = [entry for entry in measured.values() if entry.get("passed")]
            if passing:
                selected = max(
                    passing,
                    key=lambda entry: (
                        round(float(entry["analysis"]["suggested_excerpt"].get("score", 0.0)), 9),
                        round(float(entry["analysis"].get("production_score", 0.0)), 9),
                        -int(entry["index"]),
                    ),
                )
                selected_index = int(selected["index"])

        if selected_index is None:
            details = "; ".join(
                f"candidate {entry['index']}: {', '.join(entry.get('failures', []))}"
                for entry in measured.values()
            )
            return _selection_failure(
                report,
                "No advertisement soundtrack candidate passed content-aware decoded-audio QC. " + details,
            )

        entry = measured[selected_index]
        analysis = entry["analysis"]
        selected_hash = str(analysis["waveform_sha256"])
        start = float(analysis["suggested_excerpt"]["start_seconds"])
        lock_token = _sha_text(
            f"diffusiongemma-ad-audio-lock@1|{selected_hash}|{start:.9f}|{duration:.9f}|{content_mode}"
        )
        report["selected_candidate_index"] = selected_index
        report["selected_waveform_sha256"] = selected_hash
        report["selection_lock_token"] = lock_token
        report["ready"] = True
        report["lock"].update(
            {
                "satisfied": mode != "auto_select",
                "resolved_candidate_index": selected_index,
                "resolved_waveform_sha256": selected_hash,
                "resolved_start_seconds": start,
            }
        )
        status = (
            f"Advertisement audio selected candidate {selected_index} under the explicit {content_mode} contract; "
            f"waveform lock {selected_hash[:12]}…. Vocal presence remains a signal proxy, not transcription proof."
        )
        report["status"] = status
        _validate_schema(report, "advertisement_audio_selection.schema.json", "advertisement audio selection")
        audit_json = _json(report)
        director_report = {
            "schema": "diffusiongemma.advertisement_audio_selection_compact",
            "version": SELECTION_VERSION,
            "ready": True,
            "soundtrack_content_mode": content_mode,
            "selected_candidate_index": selected_index,
            "selected_audio_sha256": selected_hash,
            "selection_lock_token": lock_token,
            "qc": _candidate_summary(entry),
        }
        return (
            candidates[selected_index - 1],
            start,
            selected_hash,
            _json(director_report),
            audit_json,
            status,
            True,
        )


def _audio_parts(audio: Any, label: str) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = legacy_audio._audio_parts(audio, label)
    value = waveform.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains NaN or infinite samples.")
    return value, int(sample_rate)


def _dbfs(value: float) -> float:
    if value <= 0.0 or not math.isfinite(value):
        return -120.0
    return max(-120.0, 20.0 * math.log10(value))


def _fit_voice_over(
    voice_over: Any,
    *,
    sample_rate: int,
    batch_count: int,
    channel_count: int,
    target_samples: int,
    start_samples: int,
) -> torch.Tensor:
    value, source_rate = _audio_parts(voice_over, "voice_over")
    if source_rate != sample_rate:
        resampled_samples = max(1, int(round(value.shape[-1] * sample_rate / float(source_rate))))
        value = torch_functional.interpolate(
            value,
            size=resampled_samples,
            mode="linear",
            align_corners=False,
        )
    if value.shape[0] == 1 and batch_count > 1:
        value = value.repeat(batch_count, 1, 1)
    if int(value.shape[0]) != batch_count:
        raise ValueError("voice_over batch count must equal the music batch count or be one.")
    if value.shape[1] == 1 and channel_count > 1:
        value = value.repeat(1, channel_count, 1)
    elif channel_count == 1 and value.shape[1] > 1:
        value = value.mean(dim=1, keepdim=True)
    elif int(value.shape[1]) != channel_count:
        raise ValueError("voice_over channels cannot be aligned with the music channels.")
    output = torch.zeros((batch_count, channel_count, target_samples), dtype=torch.float32)
    usable = max(0, min(int(value.shape[-1]), target_samples - start_samples))
    if usable <= 0:
        raise ValueError("voice_over_start_seconds leaves no voice-over samples inside the master duration.")
    output[..., start_samples : start_samples + usable] = value[..., :usable]
    return output


def _same_length_pool(value: torch.Tensor, kernel: int, *, maximum: bool = False) -> torch.Tensor:
    kernel = max(1, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    if kernel == 1:
        return value
    padding = kernel // 2
    padded = torch_functional.pad(value, (padding, padding), mode="replicate")
    return (
        torch_functional.max_pool1d(padded, kernel_size=kernel, stride=1)
        if maximum
        else torch_functional.avg_pool1d(padded, kernel_size=kernel, stride=1)
    )


class DiffusionGemmaAdvertisementAudioMixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "music_audio": ("AUDIO",),
                "soundtrack_contract_json": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "target_duration_seconds": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 60.0, "step": 0.01}),
                "music_gain_db": ("FLOAT", {"default": 0.0, "min": -30.0, "max": 12.0, "step": 0.5}),
                "voice_over_gain_db": ("FLOAT", {"default": 0.0, "min": -30.0, "max": 12.0, "step": 0.5}),
                "ducking_db": ("FLOAT", {"default": -9.0, "min": -30.0, "max": 0.0, "step": 0.5}),
                "duck_attack_ms": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 1000.0, "step": 5.0}),
                "duck_release_ms": ("FLOAT", {"default": 250.0, "min": 0.0, "max": 3000.0, "step": 5.0}),
                "peak_ceiling_dbfs": ("FLOAT", {"default": -1.0, "min": -12.0, "max": 0.0, "step": 0.1}),
                "clipping_policy": (list(CLIPPING_POLICIES), {"default": CLIPPING_POLICIES[0]}),
            },
            "optional": {
                "voice_over": ("AUDIO", {"lazy": True}),
                "voice_over_start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = (
        "motion_guide_audio",
        "final_mix_audio",
        "motion_guide_sha256",
        "final_mix_sha256",
        "mix_report_json",
        "status",
        "ready",
    )
    FUNCTION = "mix"
    CATEGORY = CATEGORY
    DESCRIPTION = "Keep advertisement motion conditioning music-only while creating a separate exact-duration final mix with optional non-diegetic VO and deterministic ducking."

    def mix(
        self,
        music_audio,
        soundtrack_contract_json,
        target_duration_seconds,
        music_gain_db,
        voice_over_gain_db,
        ducking_db,
        duck_attack_ms,
        duck_release_ms,
        peak_ceiling_dbfs,
        clipping_policy,
        voice_over=None,
        voice_over_start_seconds=0.0,
    ):
        contract = _parse_soundtrack_contract(soundtrack_contract_json)
        vo_policy = str(contract["voice_over"]["policy"])
        if vo_policy == "separate_non_diegetic" and voice_over is None:
            raise ValueError("The soundtrack contract requires separate non-diegetic VO, but voice_over is not connected.")
        if vo_policy == "none" and voice_over is not None:
            raise ValueError("voice_over is connected, but the soundtrack contract explicitly sets VO to None.")
        if str(clipping_policy) not in CLIPPING_POLICIES:
            raise ValueError("clipping_policy is unsupported.")
        duration = float(target_duration_seconds)
        if not math.isfinite(duration) or not 0.0 < duration <= 60.0:
            raise ValueError("target_duration_seconds must be finite from greater than 0 to 60.")
        start_seconds = float(voice_over_start_seconds)
        if not math.isfinite(start_seconds) or start_seconds < 0.0 or start_seconds >= duration:
            raise ValueError("voice_over_start_seconds must fall inside the master duration.")
        numeric_values = {
            "music_gain_db": (music_gain_db, -30.0, 12.0),
            "voice_over_gain_db": (voice_over_gain_db, -30.0, 12.0),
            "ducking_db": (ducking_db, -30.0, 0.0),
            "duck_attack_ms": (duck_attack_ms, 0.0, 1000.0),
            "duck_release_ms": (duck_release_ms, 0.0, 3000.0),
            "peak_ceiling_dbfs": (peak_ceiling_dbfs, -12.0, 0.0),
        }
        resolved: dict[str, float] = {}
        for name, (value, minimum, maximum) in numeric_values.items():
            number = float(value)
            if not math.isfinite(number) or not minimum <= number <= maximum:
                raise ValueError(f"{name} must be finite from {minimum:g} to {maximum:g}.")
            resolved[name] = number

        music, sample_rate = _audio_parts(music_audio, "music_audio")
        target_samples = int(round(duration * sample_rate))
        source_samples = int(music.shape[-1])
        deficit = max(0, target_samples - source_samples)
        rounding_tolerance_samples = max(1, int(round(sample_rate * 0.001)))
        if deficit > rounding_tolerance_samples:
            raise ValueError(
                f"music_audio is {music.shape[-1] / float(sample_rate):.3f}s, shorter than the {duration:.3f}s master."
            )
        if deficit:
            music = torch_functional.pad(music, (0, deficit))
        music = music[..., :target_samples]
        music_gain = 10.0 ** (resolved["music_gain_db"] / 20.0)
        motion_waveform = (music * music_gain).contiguous()
        motion_guide = {"waveform": motion_waveform, "sample_rate": sample_rate}

        voice_canvas = torch.zeros_like(motion_waveform)
        voice_present = voice_over is not None
        if voice_present:
            voice_canvas = _fit_voice_over(
                voice_over,
                sample_rate=sample_rate,
                batch_count=int(motion_waveform.shape[0]),
                channel_count=int(motion_waveform.shape[1]),
                target_samples=target_samples,
                start_samples=int(round(start_seconds * sample_rate)),
            )
            voice_canvas = voice_canvas * (10.0 ** (resolved["voice_over_gain_db"] / 20.0))

        if voice_present:
            mono_abs = voice_canvas.abs().mean(dim=(0, 1), keepdim=True)
            detector_kernel = max(1, int(round(sample_rate * 0.020)))
            detected = _same_length_pool(mono_abs, detector_kernel)
            threshold = 10.0 ** (-45.0 / 20.0)
            activity = (detected >= threshold).to(dtype=torch.float32)
            attack_kernel = max(1, int(round(sample_rate * resolved["duck_attack_ms"] / 1000.0)))
            release_kernel = max(1, int(round(sample_rate * resolved["duck_release_ms"] / 1000.0)))
            activity = _same_length_pool(activity, attack_kernel, maximum=True)
            activity = _same_length_pool(activity, release_kernel)
            activity = activity.clamp(0.0, 1.0)
            duck_linear = 10.0 ** (resolved["ducking_db"] / 20.0)
            duck_gain = 1.0 - activity * (1.0 - duck_linear)
            ducked_music = motion_waveform * duck_gain
        else:
            activity = torch.zeros((1, 1, target_samples), dtype=torch.float32)
            duck_gain = torch.ones_like(activity)
            ducked_music = motion_waveform
        pre_ceiling_mix = ducked_music + voice_canvas
        pre_peak = float(pre_ceiling_mix.abs().max().item())
        ceiling_linear = 10.0 ** (resolved["peak_ceiling_dbfs"] / 20.0)
        attenuation = 1.0
        if pre_peak > ceiling_linear + 1.0e-12:
            if str(clipping_policy) == "Fail on clipping":
                raise ValueError(
                    f"Final advertisement mix sample peak {_dbfs(pre_peak):.2f} dBFS exceeds the configured {resolved['peak_ceiling_dbfs']:.2f} dBFS ceiling."
                )
            attenuation = ceiling_linear / pre_peak
        final_waveform = (pre_ceiling_mix * attenuation).contiguous()
        final_mix = {"waveform": final_waveform, "sample_rate": sample_rate}
        motion_hash = legacy_audio.waveform_sha256(motion_guide)
        final_hash = legacy_audio.waveform_sha256(final_mix)
        final_peak = float(final_waveform.abs().max().item())
        report = {
            "schema": MIX_SCHEMA,
            "version": MIX_VERSION,
            "soundtrack_contract_sha256": _sha_text(_json(contract)),
            "duration_seconds": round(duration, 9),
            "sample_rate": sample_rate,
            "sample_count": target_samples,
            "duration_alignment": {
                "source_sample_count": source_samples,
                "target_sample_count": target_samples,
                "right_padding_samples": deficit,
                "rounding_tolerance_samples": rounding_tolerance_samples,
                "policy": "right_pad_at_most_one_millisecond_then_exact_crop",
            },
            "channels": int(final_waveform.shape[1]),
            "batch_count": int(final_waveform.shape[0]),
            "voice_over": {
                "policy": vo_policy,
                "connected": voice_present,
                "start_seconds": round(start_seconds, 9) if voice_present else 0.0,
                "included_in_motion_guide": False,
                "included_in_final_mix": voice_present,
                "activity_fraction_signal_proxy": float(activity.mean().item()),
            },
            "ducking": {
                "ducking_db": resolved["ducking_db"],
                "attack_ms": resolved["duck_attack_ms"],
                "release_ms": resolved["duck_release_ms"],
                "minimum_applied_gain": float(duck_gain.min().item()),
                "algorithm": "20ms_absolute_level_proxy_with_bounded_attack_release_smoothing",
            },
            "levels": {
                "motion_guide_sample_peak_dbfs": _dbfs(float(motion_waveform.abs().max().item())),
                "pre_ceiling_final_mix_sample_peak_dbfs": _dbfs(pre_peak),
                "final_mix_sample_peak_dbfs": _dbfs(final_peak),
                "peak_ceiling_dbfs": resolved["peak_ceiling_dbfs"],
                "global_attenuation_db": _dbfs(attenuation),
                "final_clipped_sample_fraction": float(
                    (final_waveform.abs() >= 1.0).to(torch.float32).mean().item()
                ),
                "lufs": "not_measured",
                "true_peak": "not_measured",
            },
            "motion_guide_sha256": motion_hash,
            "final_mix_sha256": final_hash,
            "motion_guide_policy": "music_only_never_non_diegetic_voice_over",
            "final_mix_policy": "music_plus_optional_separate_non_diegetic_voice_over",
            "ready": True,
        }
        _validate_schema(report, "advertisement_audio_mix.schema.json", "advertisement audio mix")
        status = (
            f"Advertisement audio mixed to exactly {duration:.3f}s at {sample_rate} Hz. "
            f"Motion guide {motion_hash[:12]}… remains music-only; final mix {final_hash[:12]}… "
            + ("contains separately mixed VO." if voice_present else "contains no VO.")
        )
        return (motion_guide, final_mix, motion_hash, final_hash, _json(report), status, True)


NODE_CLASS_MAPPINGS = {
    "DiffusionGemmaAdvertisementSoundtrackSourceRouter": DiffusionGemmaAdvertisementSoundtrackSourceRouter,
    "DiffusionGemmaAdvertisementAudioCandidateSelector": DiffusionGemmaAdvertisementAudioCandidateSelector,
    "DiffusionGemmaAdvertisementAudioMixer": DiffusionGemmaAdvertisementAudioMixer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusionGemmaAdvertisementSoundtrackSourceRouter": "DiffusionGemma Advertisement Soundtrack Source",
    "DiffusionGemmaAdvertisementAudioCandidateSelector": "DiffusionGemma Advertisement Audio Audition & Lock",
    "DiffusionGemmaAdvertisementAudioMixer": "DiffusionGemma Advertisement Motion Guide / Final Mix",
}


__all__ = [
    "SELECTION_SCHEMA",
    "MIX_SCHEMA",
    "DiffusionGemmaAdvertisementAudioCandidateSelector",
    "DiffusionGemmaAdvertisementSoundtrackSourceRouter",
    "DiffusionGemmaAdvertisementAudioMixer",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
