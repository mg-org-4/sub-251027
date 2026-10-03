# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

"""Deterministic, offline LTX-2.5 A/B benchmark plus a legacy diagnostic arm.

The primary A/B compares the raw brief through the native LTX-2.5 prompt
enhancer against the mode-aware DiffusionGemma prompt with native enhancement
disabled. ``current`` is a legacy diagnostic arm. The offline preflight loads
no model or network service; its emitted render plan fixes every downstream
control and seed while recording the intentional prompt-producer difference.

Structural scoring is intentionally literal.  Benchmark authors declare the
observable concepts, source/last-frame anchors, contradictions, and action or
camera vocabulary for each case.  The scorer does not pretend that regexes can
predict aesthetic video quality; optional human/runtime ratings can be merged
after controlled renders have been produced elsewhere.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


MANIFEST_SCHEMA_VERSION = "dg-ltx25-prompt-benchmark/1"
REPORT_SCHEMA_VERSION = "dg-ltx25-prompt-benchmark-report/1"
GENERATION_PLAN_SCHEMA_VERSION = "dg-ltx25-generation-plan/1"
RUNTIME_RESULTS_SCHEMA_VERSION = "dg-ltx25-runtime-results/1"

CANDIDATE_IDS = ("raw", "current", "revised")
PROMPT_PRODUCERS = {
    "raw": "native_ltx25_prompt_enhancer",
    "current": "diffusiongemma_legacy_ltx_contract",
    "revised": "diffusiongemma_ltx25_mode_contract",
}
PRIMARY_AB_CANDIDATES = ("raw", "revised")
MODES = {"t2v", "i2v", "flf"}
CONDITIONING_KINDS = {
    "t2v": "text_only",
    "i2v": "first_frame",
    "flf": "first_last_frames",
}
CONCEPT_SCOPES = {"general", "first_frame", "last_frame"}
MATCH_POLICIES = {"any", "all"}
RUNTIME_RATING_IDS = (
    "prompt_adherence",
    "motion_coherence",
    "visual_continuity",
    "audio_alignment",
)

_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")
_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)")
_CUT_RE = re.compile(
    r"\b(?:hard\s+cut|smash\s+cut|jump\s+cut|match\s+cut|cut|cuts|cutting|dissolve|wipe)\b",
    re.IGNORECASE,
)
_REFERENCE_DEPENDENCY_RE = re.compile(
    r"\b(?:reference|provided|input|source|existing)\s+(?:image|frame|still|photo)\b",
    re.IGNORECASE,
)
_START_CUE_RE = re.compile(
    r"\b(?:begins?|starts?|opens?|first\s+frame|opening\s+frame|initially)\b",
    re.IGNORECASE,
)
_END_CUE_RE = re.compile(
    r"\b(?:ends?|finishes?|final\s+frame|last\s+frame|by\s+the\s+end|settles?)\b",
    re.IGNORECASE,
)

_GENERATION_CONTROL_KEYS = {
    "model",
    "text_encoder",
    "precision",
    "width",
    "height",
    "fps",
    "frames",
    "duration_seconds",
    "sampler",
    "scheduler",
    "stage_1_steps",
    "stage_2_steps",
    "video_cfg",
    "audio_cfg",
    "negative_prompt",
    "prompt_enhancer",
    "conditioning",
}
_CONDITIONING_KEYS = {"kind", "asset_ids", "strength"}
_RUBRIC_KEYS = {
    "word_range",
    "sentence_range",
    "max_cuts",
    "max_camera_operations",
    "max_action_beats",
    "camera_operation_terms",
    "action_terms",
    "required_concepts",
    "prohibited_concepts",
}

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "ltx25_prompt_contract_v1"
    / "manifest.json"
)


class ManifestError(ValueError):
    """The benchmark manifest is malformed or internally inconsistent."""


class RuntimeResultsError(ValueError):
    """Optional generated-result annotations do not match the benchmark plan."""


def _require_mapping(value: Any, label: str, error_type: type[ValueError]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise error_type(f"{label} must be a JSON object.")
    return value


def _require_string(value: Any, label: str, error_type: type[ValueError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{label} must be a non-empty string.")
    return value.strip()


def _string_list(
    value: Any,
    label: str,
    error_type: type[ValueError],
    *,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise error_type(f"{label} must be a list of non-empty strings.")
    normalized = [item.strip() for item in value]
    if not allow_empty and not normalized:
        raise error_type(f"{label} must not be empty.")
    if len(normalized) != len(set(normalized)):
        raise error_type(f"{label} must not contain duplicates.")
    return normalized


def _bounded_int(
    value: Any,
    label: str,
    error_type: type[ValueError],
    *,
    minimum: int = 0,
    maximum: int = 2**31 - 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise error_type(f"{label} must be an integer in {minimum}..{maximum}.")
    return int(value)


def _bounded_number(
    value: Any,
    label: str,
    error_type: type[ValueError],
    *,
    minimum: float = 0.0,
    maximum: float = 1_000_000.0,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not minimum <= float(value) <= maximum
    ):
        raise error_type(f"{label} must be a number in {minimum}..{maximum}.")
    return float(value)


def _validate_exact_keys(
    value: dict[str, Any],
    required: set[str],
    label: str,
    error_type: type[ValueError],
) -> None:
    unknown = set(value) - required
    missing = required - set(value)
    if unknown:
        raise error_type(f"{label} has unknown keys: {sorted(unknown)}")
    if missing:
        raise error_type(f"{label} is missing keys: {sorted(missing)}")


def _validate_range(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ManifestError(f"{label} must be a two-item integer range.")
    lower = _bounded_int(value[0], f"{label}[0]", ManifestError, maximum=10_000)
    upper = _bounded_int(value[1], f"{label}[1]", ManifestError, maximum=10_000)
    if lower > upper:
        raise ManifestError(f"{label} lower bound must not exceed its upper bound.")
    return [lower, upper]


def _validate_conditioning(value: Any, mode: str, label: str) -> dict[str, Any]:
    conditioning = _require_mapping(value, label, ManifestError)
    _validate_exact_keys(conditioning, _CONDITIONING_KEYS, label, ManifestError)
    kind = _require_string(conditioning.get("kind"), f"{label}.kind", ManifestError)
    expected_kind = CONDITIONING_KINDS[mode]
    if kind != expected_kind:
        raise ManifestError(
            f"{label}.kind must be {expected_kind!r} for mode {mode!r}, got {kind!r}."
        )
    asset_ids = _string_list(
        conditioning.get("asset_ids"),
        f"{label}.asset_ids",
        ManifestError,
        allow_empty=True,
    )
    expected_assets = {"t2v": 0, "i2v": 1, "flf": 2}[mode]
    if len(asset_ids) != expected_assets:
        raise ManifestError(
            f"{label}.asset_ids must contain {expected_assets} item(s) for mode {mode!r}."
        )
    strength = _bounded_number(
        conditioning.get("strength"),
        f"{label}.strength",
        ManifestError,
        maximum=1.0,
    )
    if mode == "t2v" and strength != 0.0:
        raise ManifestError(f"{label}.strength must be 0.0 for text-only generation.")
    if mode != "t2v" and strength <= 0.0:
        raise ManifestError(f"{label}.strength must be greater than zero for {mode!r}.")
    return {"kind": kind, "asset_ids": asset_ids, "strength": strength}


def _validate_generation_control(value: Any, mode: str, label: str) -> dict[str, Any]:
    control = _require_mapping(value, label, ManifestError)
    _validate_exact_keys(control, _GENERATION_CONTROL_KEYS, label, ManifestError)
    normalized: dict[str, Any] = {}
    for key in ("model", "text_encoder", "precision", "sampler", "scheduler"):
        normalized[key] = _require_string(control.get(key), f"{label}.{key}", ManifestError)
    for key in ("width", "height", "fps", "frames", "stage_1_steps", "stage_2_steps"):
        normalized[key] = _bounded_int(
            control.get(key), f"{label}.{key}", ManifestError, minimum=1, maximum=100_000
        )
    for key in ("duration_seconds", "video_cfg", "audio_cfg"):
        normalized[key] = _bounded_number(
            control.get(key), f"{label}.{key}", ManifestError, maximum=1_000.0
        )
    negative_prompt = control.get("negative_prompt")
    if not isinstance(negative_prompt, str):
        raise ManifestError(f"{label}.negative_prompt must be a string.")
    normalized["negative_prompt"] = negative_prompt
    prompt_enhancer = control.get("prompt_enhancer")
    if not isinstance(prompt_enhancer, bool):
        raise ManifestError(f"{label}.prompt_enhancer must be a boolean.")
    normalized["prompt_enhancer"] = prompt_enhancer
    normalized["conditioning"] = _validate_conditioning(
        control.get("conditioning"), mode, f"{label}.conditioning"
    )

    if normalized["frames"] % 8 != 1:
        raise ManifestError(f"{label}.frames must equal 1 plus a multiple of 8.")
    expected_duration = (normalized["frames"] - 1) / normalized["fps"]
    if abs(normalized["duration_seconds"] - expected_duration) > 1e-6:
        raise ManifestError(
            f"{label}.duration_seconds must equal (frames - 1) / fps ({expected_duration:g})."
        )
    return normalized


def _validate_concept(value: Any, label: str, *, prohibited: bool) -> dict[str, Any]:
    concept = _require_mapping(value, label, ManifestError)
    required = {"id", "weight", "patterns"} if prohibited else {
        "id",
        "scope",
        "weight",
        "match",
        "patterns",
    }
    _validate_exact_keys(concept, required, label, ManifestError)
    concept_id = _require_string(concept.get("id"), f"{label}.id", ManifestError)
    if not _CASE_ID_RE.fullmatch(concept_id):
        raise ManifestError(f"{label}.id has an invalid format.")
    normalized = {
        "id": concept_id,
        "weight": _bounded_int(
            concept.get("weight"), f"{label}.weight", ManifestError, minimum=1, maximum=100
        ),
        "patterns": _string_list(concept.get("patterns"), f"{label}.patterns", ManifestError),
    }
    if prohibited:
        return normalized
    scope = _require_string(concept.get("scope"), f"{label}.scope", ManifestError)
    match = _require_string(concept.get("match"), f"{label}.match", ManifestError)
    if scope not in CONCEPT_SCOPES:
        raise ManifestError(f"{label}.scope must be one of {sorted(CONCEPT_SCOPES)}.")
    if match not in MATCH_POLICIES:
        raise ManifestError(f"{label}.match must be one of {sorted(MATCH_POLICIES)}.")
    normalized.update({"scope": scope, "match": match})
    return normalized


def _validate_rubric(value: Any, mode: str, label: str) -> dict[str, Any]:
    rubric = _require_mapping(value, label, ManifestError)
    _validate_exact_keys(rubric, _RUBRIC_KEYS, label, ManifestError)
    required_raw = rubric.get("required_concepts")
    prohibited_raw = rubric.get("prohibited_concepts")
    if not isinstance(required_raw, list) or not required_raw:
        raise ManifestError(f"{label}.required_concepts must be a non-empty list.")
    if not isinstance(prohibited_raw, list):
        raise ManifestError(f"{label}.prohibited_concepts must be a list.")
    required = [
        _validate_concept(item, f"{label}.required_concepts[{index}]", prohibited=False)
        for index, item in enumerate(required_raw)
    ]
    prohibited = [
        _validate_concept(item, f"{label}.prohibited_concepts[{index}]", prohibited=True)
        for index, item in enumerate(prohibited_raw)
    ]
    ids = [item["id"] for item in required + prohibited]
    if len(ids) != len(set(ids)):
        raise ManifestError(f"{label} concept IDs must be unique across required and prohibited rules.")
    scopes = Counter(item["scope"] for item in required)
    if mode == "t2v" and (scopes["first_frame"] or scopes["last_frame"]):
        raise ManifestError(f"{label} T2V concepts cannot use first_frame or last_frame scopes.")
    if mode == "i2v" and (not scopes["first_frame"] or scopes["last_frame"]):
        raise ManifestError(
            f"{label} I2V requires first_frame concepts and cannot use last_frame concepts."
        )
    if mode == "flf" and (not scopes["first_frame"] or not scopes["last_frame"]):
        raise ManifestError(f"{label} FLF requires both first_frame and last_frame concepts.")
    return {
        "word_range": _validate_range(rubric.get("word_range"), f"{label}.word_range"),
        "sentence_range": _validate_range(
            rubric.get("sentence_range"), f"{label}.sentence_range"
        ),
        "max_cuts": _bounded_int(
            rubric.get("max_cuts"), f"{label}.max_cuts", ManifestError, maximum=20
        ),
        "max_camera_operations": _bounded_int(
            rubric.get("max_camera_operations"),
            f"{label}.max_camera_operations",
            ManifestError,
            maximum=20,
        ),
        "max_action_beats": _bounded_int(
            rubric.get("max_action_beats"),
            f"{label}.max_action_beats",
            ManifestError,
            maximum=50,
        ),
        "camera_operation_terms": _string_list(
            rubric.get("camera_operation_terms"),
            f"{label}.camera_operation_terms",
            ManifestError,
        ),
        "action_terms": _string_list(
            rubric.get("action_terms"), f"{label}.action_terms", ManifestError
        ),
        "required_concepts": required,
        "prohibited_concepts": prohibited,
    }


def validate_manifest(data: Any) -> dict[str, Any]:
    """Validate and normalize a ``dg-ltx25-prompt-benchmark/1`` manifest."""

    manifest = _require_mapping(data, "manifest", ManifestError)
    _validate_exact_keys(
        manifest,
        {"schema_version", "name", "description", "seeds", "cases"},
        "manifest",
        ManifestError,
    )
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(f"schema_version must be {MANIFEST_SCHEMA_VERSION!r}.")
    name = _require_string(manifest.get("name"), "manifest.name", ManifestError)
    description = _require_string(
        manifest.get("description"), "manifest.description", ManifestError
    )
    seeds_raw = manifest.get("seeds")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ManifestError("manifest.seeds must be a non-empty list of integers.")
    seeds = [
        _bounded_int(seed, f"manifest.seeds[{index}]", ManifestError, maximum=2**63 - 1)
        for index, seed in enumerate(seeds_raw)
    ]
    if len(seeds) != len(set(seeds)):
        raise ManifestError("manifest.seeds must not contain duplicates.")
    cases_raw = manifest.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ManifestError("manifest.cases must be a non-empty list.")

    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for index, value in enumerate(cases_raw):
        label = f"manifest.cases[{index}]"
        case = _require_mapping(value, label, ManifestError)
        _validate_exact_keys(
            case,
            {
                "id",
                "mode",
                "brief",
                "generation_control",
                "candidates",
                "rubric",
                "notes",
            },
            label,
            ManifestError,
        )
        case_id = _require_string(case.get("id"), f"{label}.id", ManifestError)
        if not _CASE_ID_RE.fullmatch(case_id):
            raise ManifestError(f"{label}.id has an invalid format.")
        if case_id in case_ids:
            raise ManifestError(f"Duplicate case ID {case_id!r}.")
        case_ids.add(case_id)
        mode = _require_string(case.get("mode"), f"{label}.mode", ManifestError)
        if mode not in MODES:
            raise ManifestError(f"{label}.mode must be one of {sorted(MODES)}.")
        candidates = _require_mapping(case.get("candidates"), f"{label}.candidates", ManifestError)
        _validate_exact_keys(candidates, set(CANDIDATE_IDS), f"{label}.candidates", ManifestError)
        normalized_candidates = {
            candidate_id: _require_string(
                candidates.get(candidate_id),
                f"{label}.candidates.{candidate_id}",
                ManifestError,
            )
            for candidate_id in CANDIDATE_IDS
        }
        if len(set(normalized_candidates.values())) != len(CANDIDATE_IDS):
            raise ManifestError(f"{label}.candidates must contain three distinct prompts.")
        notes = case.get("notes")
        if not isinstance(notes, str):
            raise ManifestError(f"{label}.notes must be a string.")
        cases.append(
            {
                "id": case_id,
                "mode": mode,
                "brief": _require_string(case.get("brief"), f"{label}.brief", ManifestError),
                "generation_control": _validate_generation_control(
                    case.get("generation_control"), mode, f"{label}.generation_control"
                ),
                "candidates": normalized_candidates,
                "rubric": _validate_rubric(case.get("rubric"), mode, f"{label}.rubric"),
                "notes": notes.strip(),
            }
        )
    covered_modes = {case["mode"] for case in cases}
    if covered_modes != MODES:
        raise ManifestError(
            f"manifest.cases must cover all modes {sorted(MODES)}; got {sorted(covered_modes)}."
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": name,
        "description": description,
        "seeds": seeds,
        "cases": cases,
    }


def load_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        data = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read manifest {path}: {exc}") from exc
    return validate_manifest(data), payload


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contains(text: str, pattern: str) -> bool:
    return pattern.casefold() in text.casefold()


def _matched_patterns(text: str, patterns: Iterable[str]) -> list[str]:
    return [pattern for pattern in patterns if _contains(text, pattern)]


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _sentence_count(text: str) -> int:
    return sum(1 for match in _SENTENCE_RE.finditer(text) if match.group(0).strip())


def _term_count(text: str, terms: Iterable[str]) -> tuple[int, list[str]]:
    matched = _matched_patterns(text, terms)
    return len(matched), matched


def _rule(
    rule_id: str,
    passed: bool,
    weight: int,
    detail: str,
    *,
    observed: Any = None,
) -> dict[str, Any]:
    result = {
        "id": rule_id,
        "passed": bool(passed),
        "weight": int(weight),
        "earned": int(weight) if passed else 0,
        "detail": detail,
    }
    if observed is not None:
        result["observed"] = observed
    return result


def _mode_rule(
    mode: str,
    text: str,
    cut_count: int,
    required_rules: list[dict[str, Any]],
) -> dict[str, Any]:
    if mode == "t2v":
        dependency = _REFERENCE_DEPENDENCY_RE.search(text)
        return _rule(
            "mode.t2v_self_contained",
            dependency is None,
            4,
            "T2V must be self-contained and must not depend on a source frame.",
            observed=dependency.group(0) if dependency else "none",
        )

    required_by_scope: dict[str, list[dict[str, Any]]] = {
        "first_frame": [
            rule for rule in required_rules if rule.get("scope") == "first_frame"
        ],
        "last_frame": [
            rule for rule in required_rules if rule.get("scope") == "last_frame"
        ],
    }
    first_frame_ok = bool(required_by_scope["first_frame"]) and all(
        rule["passed"] for rule in required_by_scope["first_frame"]
    )
    if mode == "i2v":
        passed = first_frame_ok and cut_count == 0
        return _rule(
            "mode.i2v_first_frame_feasible",
            passed,
            4,
            "I2V must retain every declared first-frame anchor and remain a continuous take.",
            observed={"all_first_frame_anchors": first_frame_ok, "cuts": cut_count},
        )

    start = _START_CUE_RE.search(text)
    end = _END_CUE_RE.search(text)
    last_frame_ok = bool(required_by_scope["last_frame"]) and all(
        rule["passed"] for rule in required_by_scope["last_frame"]
    )
    ordered = bool(start and end and start.start() < end.start())
    passed = first_frame_ok and last_frame_ok and ordered and cut_count == 0
    return _rule(
        "mode.flf_endpoint_transition",
        passed,
        4,
        "FLF must name both endpoint anchors in order and describe one continuous transition.",
        observed={
            "all_first_frame_anchors": first_frame_ok,
            "all_last_frame_anchors": last_frame_ok,
            "ordered_start_and_end_cues": ordered,
            "cuts": cut_count,
        },
    )


def score_prompt(case: dict[str, Any], candidate_id: str, prompt: str) -> dict[str, Any]:
    """Score one prompt using the case's literal, deterministic rubric."""

    rubric = case["rubric"]
    words = _word_count(prompt)
    sentences = _sentence_count(prompt)
    cut_count = len(_CUT_RE.findall(prompt))
    camera_count, camera_matches = _term_count(prompt, rubric["camera_operation_terms"])
    action_count, action_matches = _term_count(prompt, rubric["action_terms"])
    rules: list[dict[str, Any]] = [
        _rule(
            "budget.words",
            rubric["word_range"][0] <= words <= rubric["word_range"][1],
            2,
            f"Word count must be within {rubric['word_range'][0]}..{rubric['word_range'][1]}.",
            observed=words,
        ),
        _rule(
            "budget.sentences",
            rubric["sentence_range"][0] <= sentences <= rubric["sentence_range"][1],
            1,
            "Sentence count must fit the case's duration-aware range.",
            observed=sentences,
        ),
        _rule(
            "budget.cuts",
            cut_count <= rubric["max_cuts"],
            3,
            f"Explicit cuts must not exceed {rubric['max_cuts']}.",
            observed=cut_count,
        ),
        _rule(
            "budget.camera_operations",
            camera_count <= rubric["max_camera_operations"],
            2,
            f"Distinct declared camera operations must not exceed {rubric['max_camera_operations']}.",
            observed={"count": camera_count, "matches": camera_matches},
        ),
        _rule(
            "budget.action_beats",
            action_count <= rubric["max_action_beats"],
            3,
            f"Distinct declared action beats must not exceed {rubric['max_action_beats']}.",
            observed={"count": action_count, "matches": action_matches},
        ),
    ]

    required_rules: list[dict[str, Any]] = []
    for concept in rubric["required_concepts"]:
        matches = _matched_patterns(prompt, concept["patterns"])
        passed = bool(matches) if concept["match"] == "any" else len(matches) == len(
            concept["patterns"]
        )
        result = _rule(
            f"required.{concept['id']}",
            passed,
            concept["weight"],
            f"Required {concept['scope']} concept ({concept['match']} pattern match).",
            observed=matches,
        )
        result["scope"] = concept["scope"]
        required_rules.append(result)
    rules.extend(required_rules)

    for concept in rubric["prohibited_concepts"]:
        matches = _matched_patterns(prompt, concept["patterns"])
        rules.append(
            _rule(
                f"prohibited.{concept['id']}",
                not matches,
                concept["weight"],
                "Prompt must not contain this contradiction or non-observable control language.",
                observed=matches,
            )
        )

    rules.append(_mode_rule(case["mode"], prompt, cut_count, required_rules))
    earned = sum(rule["earned"] for rule in rules)
    possible = sum(rule["weight"] for rule in rules)
    score = round(100.0 * earned / possible, 2) if possible else 0.0
    failed = [rule["id"] for rule in rules if not rule["passed"]]
    return {
        "candidate": candidate_id,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "score": score,
        "earned_weight": earned,
        "possible_weight": possible,
        "metrics": {
            "word_count": words,
            "sentence_count": sentences,
            "cut_count": cut_count,
            "camera_operation_count": camera_count,
            "action_beat_count": action_count,
        },
        "failed_rule_ids": failed,
        "rules": rules,
    }


def build_generation_plan(
    manifest: dict[str, Any], manifest_sha256: str
) -> dict[str, Any]:
    """Create a deterministic render matrix; this function never runs a model."""

    runs: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        downstream_control = {
            key: value
            for key, value in case["generation_control"].items()
            if key != "prompt_enhancer"
        }
        downstream_control_hash = _canonical_hash(downstream_control)
        for candidate_id in CANDIDATE_IDS:
            native_enhancer = candidate_id == "raw"
            prompt_input = case["brief"] if native_enhancer else case["candidates"][candidate_id]
            run_control = dict(case["generation_control"])
            run_control["prompt_enhancer"] = native_enhancer
            for seed in manifest["seeds"]:
                runs.append(
                    {
                        "run_id": f"{case['id']}:{candidate_id}:{seed}",
                        "case_id": case["id"],
                        "mode": case["mode"],
                        "candidate": candidate_id,
                        "benchmark_role": (
                            "A_native_enhancer"
                            if candidate_id == "raw"
                            else "B_diffusiongemma_ltx25"
                            if candidate_id == "revised"
                            else "legacy_diagnostic"
                        ),
                        "prompt_producer": PROMPT_PRODUCERS[candidate_id],
                        "seed": seed,
                        "prompt": prompt_input,
                        "prompt_sha256": hashlib.sha256(
                            prompt_input.encode("utf-8")
                        ).hexdigest(),
                        "structural_reference_prompt": case["candidates"][candidate_id],
                        "generation_control": run_control,
                        "generation_control_sha256": _canonical_hash(run_control),
                        "downstream_control_sha256": downstream_control_hash,
                    }
                )
    return {
        "schema_version": GENERATION_PLAN_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "candidate_order": list(CANDIDATE_IDS),
        "primary_ab_candidates": list(PRIMARY_AB_CANDIDATES),
        "prompt_producers": PROMPT_PRODUCERS,
        "seeds": manifest["seeds"],
        "runs": runs,
        "execution": {
            "performed": False,
            "note": "This plan is data only. A sends the raw brief through the native LTX-2.5 enhancer; B sends the DiffusionGemma LTX-2.5 prompt with native enhancement disabled. Downstream controls and paired seeds are identical.",
        },
    }


def _load_runtime_results(
    path: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeResultsError(f"Could not read runtime results {path}: {exc}") from exc
    result = _require_mapping(data, "runtime results", RuntimeResultsError)
    _validate_exact_keys(
        result,
        {"schema_version", "manifest_sha256", "runs"},
        "runtime results",
        RuntimeResultsError,
    )
    if result.get("schema_version") != RUNTIME_RESULTS_SCHEMA_VERSION:
        raise RuntimeResultsError(
            f"runtime results schema_version must be {RUNTIME_RESULTS_SCHEMA_VERSION!r}."
        )
    if result.get("manifest_sha256") != manifest_sha256:
        raise RuntimeResultsError("runtime results manifest_sha256 does not match this manifest.")
    raw_runs = result.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise RuntimeResultsError("runtime results runs must be a non-empty list.")
    cases = {case["id"]: case for case in manifest["cases"]}
    valid_seeds = set(manifest["seeds"])
    seen: set[tuple[str, str, int]] = set()
    runs: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_runs):
        label = f"runtime results runs[{index}]"
        run = _require_mapping(raw, label, RuntimeResultsError)
        _validate_exact_keys(
            run,
            {"case_id", "candidate", "seed", "output", "ratings", "notes"},
            label,
            RuntimeResultsError,
        )
        case_id = _require_string(run.get("case_id"), f"{label}.case_id", RuntimeResultsError)
        candidate = _require_string(
            run.get("candidate"), f"{label}.candidate", RuntimeResultsError
        )
        seed = _bounded_int(
            run.get("seed"), f"{label}.seed", RuntimeResultsError, maximum=2**63 - 1
        )
        if case_id not in cases:
            raise RuntimeResultsError(f"{label}.case_id is not declared by the manifest.")
        if candidate not in CANDIDATE_IDS:
            raise RuntimeResultsError(f"{label}.candidate must be one of {list(CANDIDATE_IDS)}.")
        if seed not in valid_seeds:
            raise RuntimeResultsError(f"{label}.seed is not declared by the manifest.")
        key = (case_id, candidate, seed)
        if key in seen:
            raise RuntimeResultsError(f"{label} duplicates run {case_id}:{candidate}:{seed}.")
        seen.add(key)
        output = _require_string(run.get("output"), f"{label}.output", RuntimeResultsError)
        ratings = _require_mapping(run.get("ratings"), f"{label}.ratings", RuntimeResultsError)
        _validate_exact_keys(
            ratings,
            set(RUNTIME_RATING_IDS),
            f"{label}.ratings",
            RuntimeResultsError,
        )
        normalized_ratings = {
            rating_id: _bounded_number(
                ratings.get(rating_id),
                f"{label}.ratings.{rating_id}",
                RuntimeResultsError,
                maximum=5.0,
            )
            for rating_id in RUNTIME_RATING_IDS
        }
        notes = run.get("notes")
        if not isinstance(notes, str):
            raise RuntimeResultsError(f"{label}.notes must be a string.")
        runs.append(
            {
                "case_id": case_id,
                "candidate": candidate,
                "seed": seed,
                "output": output,
                "ratings": normalized_ratings,
                "mean_rating": round(
                    sum(normalized_ratings.values()) / len(normalized_ratings), 3
                ),
                "notes": notes.strip(),
            }
        )
    expected_count = len(cases) * len(CANDIDATE_IDS) * len(valid_seeds)
    by_candidate: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATE_IDS:
        candidate_runs = [run for run in runs if run["candidate"] == candidate]
        by_candidate[candidate] = {
            "run_count": len(candidate_runs),
            "mean_rating": round(
                sum(run["mean_rating"] for run in candidate_runs) / len(candidate_runs), 3
            )
            if candidate_runs
            else None,
            "mean_by_dimension": {
                rating_id: round(
                    sum(run["ratings"][rating_id] for run in candidate_runs)
                    / len(candidate_runs),
                    3,
                )
                if candidate_runs
                else None
                for rating_id in RUNTIME_RATING_IDS
            },
        }
    return {
        "status": "complete" if len(runs) == expected_count else "partial",
        "run_count": len(runs),
        "expected_run_count": expected_count,
        "coverage": round(len(runs) / expected_count, 4),
        "candidate_metrics": by_candidate,
        "runs": runs,
    }


def evaluate_manifest(
    manifest_path: Path,
    *,
    runtime_results_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate all prompt candidates without requiring a model or network."""

    manifest_path = manifest_path.resolve()
    manifest, manifest_bytes = load_manifest(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    case_reports: list[dict[str, Any]] = []
    scores_by_candidate: dict[str, list[float]] = {
        candidate: [] for candidate in CANDIDATE_IDS
    }
    wins = Counter({candidate: 0 for candidate in CANDIDATE_IDS})
    for case in manifest["cases"]:
        candidate_reports = {
            candidate: score_prompt(case, candidate, case["candidates"][candidate])
            for candidate in CANDIDATE_IDS
        }
        ranking = sorted(
            CANDIDATE_IDS,
            key=lambda candidate: (-candidate_reports[candidate]["score"], CANDIDATE_IDS.index(candidate)),
        )
        top_score = candidate_reports[ranking[0]]["score"]
        for candidate in CANDIDATE_IDS:
            scores_by_candidate[candidate].append(candidate_reports[candidate]["score"])
            if candidate_reports[candidate]["score"] == top_score:
                wins[candidate] += 1
        downstream_control = {
            key: value
            for key, value in case["generation_control"].items()
            if key != "prompt_enhancer"
        }
        control_hash = _canonical_hash(downstream_control)
        case_reports.append(
            {
                "case_id": case["id"],
                "mode": case["mode"],
                "brief": case["brief"],
                "generation_control": case["generation_control"],
                "downstream_control_sha256": control_hash,
                "downstream_control_is_fixed_across_candidates": True,
                "prompt_producers": PROMPT_PRODUCERS,
                "candidates": candidate_reports,
                "ranking": ranking,
                "winner_score": top_score,
                "revised_vs_current_delta": round(
                    candidate_reports["revised"]["score"]
                    - candidate_reports["current"]["score"],
                    2,
                ),
                "primary_ab_structural_delta_b_minus_a": round(
                    candidate_reports["revised"]["score"]
                    - candidate_reports["raw"]["score"],
                    2,
                ),
            }
        )
    candidate_summary = {
        candidate: {
            "mean_score": round(
                sum(scores_by_candidate[candidate]) / len(scores_by_candidate[candidate]), 2
            ),
            "min_score": min(scores_by_candidate[candidate]),
            "max_score": max(scores_by_candidate[candidate]),
            "case_win_count": wins[candidate],
        }
        for candidate in CANDIDATE_IDS
    }
    runtime = (
        _load_runtime_results(runtime_results_path, manifest, manifest_sha256)
        if runtime_results_path is not None
        else {
            "status": "not_supplied",
            "run_count": 0,
            "note": "No model was loaded. Pass --runtime-results to merge controlled render ratings.",
        }
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "manifest": {
            "schema_version": manifest["schema_version"],
            "name": manifest["name"],
            "sha256": manifest_sha256,
            "case_count": len(manifest["cases"]),
            "seeds": manifest["seeds"],
        },
        "execution": {
            "offline": True,
            "model_loaded": False,
            "network_used": False,
        },
        "candidate_summary": candidate_summary,
        "primary_ab": {
            "A": {"candidate": "raw", "prompt_producer": PROMPT_PRODUCERS["raw"]},
            "B": {"candidate": "revised", "prompt_producer": PROMPT_PRODUCERS["revised"]},
            "structural_mean_delta_b_minus_a": round(
                candidate_summary["revised"]["mean_score"]
                - candidate_summary["raw"]["mean_score"],
                2,
            ),
            "note": "Structural scoring is only a preflight. Controlled paired renders and blinded ratings decide video quality.",
        },
        "revised_vs_current_mean_delta": round(
            candidate_summary["revised"]["mean_score"]
            - candidate_summary["current"]["mean_score"],
            2,
        ),
        "runtime_results": runtime,
        "cases": case_reports,
        "scoring_notes": [
            "Structural scores are deterministic contract checks, not predictions of aesthetic quality.",
            "A and B intentionally differ only in prompt production: native LTX-2.5 enhancer on raw brief versus mode-aware DiffusionGemma prompt with native enhancement disabled.",
            "Model, precision, conditioning assets/strength, resolution, frames, fps, sampler, scheduler, steps, CFG, negative prompt, and seed are paired and identical.",
            "Action and camera counts use declared literal vocabulary from each case fixture.",
            "I2V and FLF feasibility requires all declared endpoint anchors, not just generic image wording.",
            "Optional runtime ratings are external annotations on a fixed seed/control generation plan.",
        ],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare LTX-2.5 raw/current/revised prompts offline under fixed controls."
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Benchmark manifest (default: {DEFAULT_MANIFEST}).",
    )
    parser.add_argument("--output", type=Path, help="Optional structural report JSON path.")
    parser.add_argument(
        "--emit-generation-plan",
        type=Path,
        help="Write the fixed prompt/seed/control render matrix without running a model.",
    )
    parser.add_argument(
        "--runtime-results",
        type=Path,
        help="Optional externally rated controlled-render results to merge into the report.",
    )
    args = parser.parse_args(argv)
    try:
        report = evaluate_manifest(
            args.manifest,
            runtime_results_path=args.runtime_results,
        )
        if args.emit_generation_plan is not None:
            manifest, manifest_bytes = load_manifest(args.manifest.resolve())
            plan = build_generation_plan(
                manifest, hashlib.sha256(manifest_bytes).hexdigest()
            )
            _write_json(args.emit_generation_plan, plan)
        if args.output is not None:
            _write_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        ManifestError,
        RuntimeResultsError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
