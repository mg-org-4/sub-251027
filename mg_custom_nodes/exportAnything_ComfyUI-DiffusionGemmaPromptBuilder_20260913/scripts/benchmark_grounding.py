# Copyright (c) 2026 exportAnything. All rights reserved.
# SPDX-License-Identifier: MIT

"""Offline scoring scaffold for versioned DiffusionGemma grounding benchmarks.

The scorer deliberately uses stable fact IDs supplied by a benchmark author. It
does not use a language model, network service, or fuzzy text matching. Input
fixtures are read only to prove that the evaluated case is present; all scores
come from pre-recorded result JSON.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
import re
from typing import Any


MANIFEST_SCHEMA_VERSION = "dg-grounding-benchmark/1"
RESULT_SCHEMA_VERSION = "dg-grounding-benchmark-result/2"
REPORT_SCHEMA_VERSION = "dg-grounding-benchmark-report/2"
CATEGORIES = {
    "synthetic_image",
    "synthetic_video",
    "private_clean",
    "private_known_failure",
}
GROUNDING_MODES = {"off", "audit", "strict"}
PROMPTING_CONDITIONS = {
    "evidence_only",
    "current_director",
    "guard_audit_combined",
    "guard_strict",
}
SAMPLING_PROFILES = {"checkpoint_defaults", "full_48_diagnostic"}
MEDIA_VARIANTS = {
    "case_default",
    "real",
    "neutral",
    "unrelated",
    "original",
    "reversed",
    "shuffled",
    "frozen_first_frame",
    "gaussian_distorted",
}
PRECISION_MODES = {"nvfp4", "bf16", "fp16", "fp32"}
TARGET_PROFILES = {"ltx", "h3_t2va", "h3_ref2va", "ideogram"}
FACT_METRIC_TAGS = {
    "supported_fact",
    "unsupported_fact",
    "temporal_order",
    "object",
    "count",
    "color",
    "text",
    "spatial",
    "motion",
    "camera",
}
RELEASE_CATEGORY_COUNTS = {
    "synthetic_image": 10,
    "synthetic_video": 8,
    "private_clean": 6,
    "private_known_failure": 6,
}
RELEASE_HOLDOUT_COUNTS = {
    "synthetic_image": 2,
    "synthetic_video": 2,
    "private_clean": 1,
    "private_known_failure": 1,
}
RELEASE_THRESHOLDS = {
    "supported_fact_recall": 0.80,
    "clean_strict_pass_rate": 0.85,
    "clean_false_block_rate": 0.15,
    "counterfactual_discrimination": 0.90,
    "temporal_order_accuracy": 0.80,
    "known_failure_holdout_silent_pass_count": 0,
}
_CONDITION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
COUNTERFACTUAL_VARIANTS = {
    "real",
    "neutral",
    "unrelated",
    "original",
    "reversed",
    "shuffled",
    "frozen_first_frame",
}
BASELINE_VARIANTS = {"real", "original"}
GUARD_DECISIONS = {"disabled", "not_applicable", "pass", "warn", "block"}
ANALYSIS_STATUSES = {
    "not_run",
    "not_applicable",
    "grounded",
    "uncertain",
    "refused",
    "transport_error",
}


class ManifestError(ValueError):
    pass


class ResultError(ValueError):
    pass


def _require_mapping(value: Any, label: str, error_type: type[ValueError]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise error_type(f"{label} must be a JSON object.")
    return value


def _require_string(value: Any, label: str, error_type: type[ValueError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{label} must be a non-empty string.")
    return value.strip()


def _string_list(value: Any, label: str, error_type: type[ValueError]) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise error_type(f"{label} must be a list of non-empty strings.")
    normalized = [item.strip() for item in value]
    if len(normalized) != len(set(normalized)):
        raise error_type(f"{label} must not contain duplicates.")
    return normalized


def _seed_list(value: Any, label: str, error_type: type[ValueError]) -> list[int]:
    if not isinstance(value, list) or not value:
        raise error_type(f"{label} must be a non-empty list of 64-bit integers.")
    lower = -(2**63)
    upper = 2**63 - 1
    if any(isinstance(item, bool) or not isinstance(item, int) or not lower <= item <= upper for item in value):
        raise error_type(f"{label} must contain only signed 64-bit integers.")
    if len(value) != len(set(value)):
        raise error_type(f"{label} must not contain duplicate seeds.")
    return list(value)


def _relative_manifest_path(value: Any, label: str) -> str:
    text = _require_string(value, label, ManifestError).replace("\\", "/")
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or text.startswith("//")
        or ":" in text
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ManifestError(f"{label} must be a confined relative path without traversal.")
    return pure.as_posix()


def _resolve_below(base: Path, relative: str, label: str) -> Path:
    root = base.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"{label} escapes the manifest directory: {relative!r}.") from exc
    return resolved


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


def _validate_condition(value: Any, label: str) -> dict[str, Any]:
    condition = _require_mapping(value, label, ResultError)
    required = {
        "condition_id",
        "prompting",
        "sampling_profile",
        "media_variant",
        "precision",
        "telemetry_enabled",
        "frame_budget",
        "visual_token_budget",
        "model_revision",
        "release_candidate",
    }
    unknown = set(condition) - required
    missing = required - set(condition)
    if unknown:
        raise ResultError(f"{label} has unknown keys: {sorted(unknown)}")
    if missing:
        raise ResultError(f"{label} is missing keys: {sorted(missing)}")
    condition_id = _require_string(condition.get("condition_id"), f"{label}.condition_id", ResultError)
    if not _CONDITION_ID_RE.fullmatch(condition_id):
        raise ResultError(f"{label}.condition_id has an invalid format.")
    prompting = _require_string(condition.get("prompting"), f"{label}.prompting", ResultError)
    if prompting not in PROMPTING_CONDITIONS:
        raise ResultError(f"{label}.prompting must be one of {sorted(PROMPTING_CONDITIONS)}.")
    sampling = _require_string(
        condition.get("sampling_profile"), f"{label}.sampling_profile", ResultError
    )
    if sampling not in SAMPLING_PROFILES:
        raise ResultError(f"{label}.sampling_profile must be one of {sorted(SAMPLING_PROFILES)}.")
    media_variant = _require_string(
        condition.get("media_variant"), f"{label}.media_variant", ResultError
    )
    if media_variant not in MEDIA_VARIANTS:
        raise ResultError(f"{label}.media_variant must be one of {sorted(MEDIA_VARIANTS)}.")
    precision = _require_string(condition.get("precision"), f"{label}.precision", ResultError)
    if precision not in PRECISION_MODES:
        raise ResultError(f"{label}.precision must be one of {sorted(PRECISION_MODES)}.")
    telemetry_enabled = condition.get("telemetry_enabled")
    release_candidate = condition.get("release_candidate")
    if not isinstance(telemetry_enabled, bool) or not isinstance(release_candidate, bool):
        raise ResultError(f"{label}.telemetry_enabled and release_candidate must be booleans.")
    model_revision = _require_string(
        condition.get("model_revision"), f"{label}.model_revision", ResultError
    )
    if len(model_revision) > 256:
        raise ResultError(f"{label}.model_revision must be at most 256 characters.")
    return {
        "condition_id": condition_id,
        "prompting": prompting,
        "sampling_profile": sampling,
        "media_variant": media_variant,
        "precision": precision,
        "telemetry_enabled": telemetry_enabled,
        "frame_budget": _bounded_int(
            condition.get("frame_budget"), f"{label}.frame_budget", ResultError
        ),
        "visual_token_budget": _bounded_int(
            condition.get("visual_token_budget"), f"{label}.visual_token_budget", ResultError
        ),
        "model_revision": model_revision,
        "release_candidate": release_candidate,
    }


def _condition_key(condition: dict[str, Any]) -> str:
    return json.dumps(condition, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _counterfactual_condition_key(condition: dict[str, Any]) -> str:
    comparable = {
        key: value for key, value in condition.items() if key != "media_variant"
    }
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fact_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ManifestError(f"{label} must be a list.")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_fact in enumerate(value):
        fact = _require_mapping(raw_fact, f"{label}[{index}]", ManifestError)
        unknown = set(fact) - {
            "id",
            "claim",
            "aliases",
            "frame_indices",
            "timecodes",
            "metric_tags",
        }
        if unknown:
            raise ManifestError(f"{label}[{index}] has unknown keys: {sorted(unknown)}")
        fact_id = _require_string(fact.get("id"), f"{label}[{index}].id", ManifestError)
        if fact_id in seen:
            raise ManifestError(f"{label} contains duplicate fact ID {fact_id!r}.")
        seen.add(fact_id)
        aliases = _string_list(fact.get("aliases", []), f"{label}[{index}].aliases", ManifestError)
        frame_indices = fact.get("frame_indices", [])
        if (
            not isinstance(frame_indices, list)
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in frame_indices)
        ):
            raise ManifestError(f"{label}[{index}].frame_indices must contain non-negative integers.")
        timecodes = fact.get("timecodes", [])
        if (
            not isinstance(timecodes, list)
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) or item < 0 for item in timecodes)
        ):
            raise ManifestError(f"{label}[{index}].timecodes must contain non-negative numbers.")
        claim = fact.get("claim", "")
        if claim and not isinstance(claim, str):
            raise ManifestError(f"{label}[{index}].claim must be a string when supplied.")
        metric_tags = _string_list(
            fact.get("metric_tags"), f"{label}[{index}].metric_tags", ManifestError
        )
        if not metric_tags or any(tag not in FACT_METRIC_TAGS for tag in metric_tags):
            raise ManifestError(
                f"{label}[{index}].metric_tags must be a non-empty subset of {sorted(FACT_METRIC_TAGS)}."
            )
        normalized.append(
            {
                "id": fact_id,
                "claim": claim.strip() if isinstance(claim, str) else "",
                "aliases": aliases,
                "frame_indices": list(frame_indices),
                "timecodes": [float(item) for item in timecodes],
                "metric_tags": metric_tags,
            }
        )
    return normalized


def validate_manifest(data: Any) -> dict[str, Any]:
    """Validate and normalize a dg-grounding-benchmark/1 manifest."""
    manifest = _require_mapping(data, "manifest", ManifestError)
    unknown = set(manifest) - {"schema_version", "name", "description", "seeds", "cases"}
    if unknown:
        raise ManifestError(f"Manifest has unknown keys: {sorted(unknown)}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(f"schema_version must be {MANIFEST_SCHEMA_VERSION!r}.")
    name = _require_string(manifest.get("name"), "manifest.name", ManifestError)
    description = manifest.get("description", "")
    if description and not isinstance(description, str):
        raise ManifestError("manifest.description must be a string when supplied.")
    default_seeds = _seed_list(manifest.get("seeds", [0]), "manifest.seeds", ManifestError)
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ManifestError("manifest.cases must be a non-empty list.")

    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    group_variants: dict[str, set[str]] = {}
    for index, raw_case in enumerate(raw_cases):
        label = f"manifest.cases[{index}]"
        case = _require_mapping(raw_case, label, ManifestError)
        unknown = set(case) - {
            "id",
            "category",
            "fixture",
            "fixture_sha256",
            "result",
            "private",
            "holdout",
            "target_profile",
            "seeds",
            "expected_facts",
            "prohibited_facts",
            "ambiguity_labels",
            "counterfactual_group",
            "counterfactual_variant",
            "notes",
        }
        if unknown:
            raise ManifestError(f"{label} has unknown keys: {sorted(unknown)}")
        case_id = _require_string(case.get("id"), f"{label}.id", ManifestError)
        if case_id in case_ids:
            raise ManifestError(f"Duplicate case ID {case_id!r}.")
        case_ids.add(case_id)
        category = _require_string(case.get("category"), f"{label}.category", ManifestError)
        if category not in CATEGORIES:
            raise ManifestError(f"{label}.category must be one of {sorted(CATEGORIES)}.")
        fixture = _relative_manifest_path(case.get("fixture"), f"{label}.fixture")
        fixture_sha256 = _require_string(
            case.get("fixture_sha256"), f"{label}.fixture_sha256", ManifestError
        ).lower()
        if len(fixture_sha256) != 64 or any(character not in "0123456789abcdef" for character in fixture_sha256):
            raise ManifestError(f"{label}.fixture_sha256 must be a 64-character hexadecimal SHA-256.")
        result_path = _relative_manifest_path(
            case.get("result", f"results/{case_id}.json"), f"{label}.result"
        )
        private = case.get("private", category.startswith("private_"))
        holdout = case.get("holdout", False)
        if not isinstance(private, bool) or not isinstance(holdout, bool):
            raise ManifestError(f"{label}.private and holdout must be booleans.")
        expected_private = category.startswith("private_")
        if private != expected_private:
            raise ManifestError(
                f"{label}.private must be {str(expected_private).lower()} for category {category!r}."
            )
        target_profile = case.get("target_profile", "")
        notes = case.get("notes", "")
        if not isinstance(target_profile, str) or not isinstance(notes, str):
            raise ManifestError(f"{label}.target_profile and notes must be strings.")
        target_profile = target_profile.strip()
        if target_profile not in TARGET_PROFILES:
            raise ManifestError(f"{label}.target_profile must be one of {sorted(TARGET_PROFILES)}.")
        expected = _fact_list(case.get("expected_facts", []), f"{label}.expected_facts")
        prohibited = _fact_list(case.get("prohibited_facts", []), f"{label}.prohibited_facts")
        if any("supported_fact" not in fact["metric_tags"] for fact in expected):
            raise ManifestError(f"{label}.expected_facts must all include metric tag 'supported_fact'.")
        if any("unsupported_fact" not in fact["metric_tags"] for fact in prohibited):
            raise ManifestError(f"{label}.prohibited_facts must all include metric tag 'unsupported_fact'.")
        overlap = {fact["id"] for fact in expected} & {fact["id"] for fact in prohibited}
        if overlap:
            raise ManifestError(f"{label} marks fact IDs as both expected and prohibited: {sorted(overlap)}")
        ambiguity_labels = _string_list(
            case.get("ambiguity_labels", []), f"{label}.ambiguity_labels", ManifestError
        )
        group = case.get("counterfactual_group", "")
        variant = case.get("counterfactual_variant", "")
        if group or variant:
            group = _require_string(group, f"{label}.counterfactual_group", ManifestError)
            variant = _require_string(variant, f"{label}.counterfactual_variant", ManifestError)
            if variant not in COUNTERFACTUAL_VARIANTS:
                raise ManifestError(
                    f"{label}.counterfactual_variant must be one of {sorted(COUNTERFACTUAL_VARIANTS)}."
                )
            variants = group_variants.setdefault(group, set())
            if variant in variants:
                raise ManifestError(f"Counterfactual group {group!r} repeats variant {variant!r}.")
            variants.add(variant)
        cases.append(
            {
                "id": case_id,
                "category": category,
                "fixture": fixture,
                "fixture_sha256": fixture_sha256,
                "result": result_path,
                "private": private,
                "holdout": holdout,
                "target_profile": target_profile,
                "seeds": _seed_list(case.get("seeds", default_seeds), f"{label}.seeds", ManifestError),
                "expected_facts": expected,
                "prohibited_facts": prohibited,
                "ambiguity_labels": ambiguity_labels,
                "counterfactual_group": group,
                "counterfactual_variant": variant,
                "notes": notes.strip(),
            }
        )

    for group, variants in group_variants.items():
        baselines = variants & BASELINE_VARIANTS
        if len(baselines) != 1 or len(variants) < 2:
            raise ManifestError(
                f"Counterfactual group {group!r} must have exactly one real/original baseline and at least one variant."
            )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "name": name,
        "description": description.strip() if isinstance(description, str) else "",
        "seeds": default_seeds,
        "cases": cases,
    }


def _load_json(path: Path, error_type: type[ValueError], label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise error_type(f"Could not read {label} {path}: {exc}") from exc


def load_manifest(path: Path) -> dict[str, Any]:
    return validate_manifest(_load_json(path, ManifestError, "manifest"))


def _validate_result(data: Any, case: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    result = _require_mapping(data, f"result {path}", ResultError)
    unknown = set(result) - {"schema_version", "case_id", "runs"}
    if unknown:
        raise ResultError(f"Result {path} has unknown keys: {sorted(unknown)}")
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ResultError(f"Result {path} schema_version must be {RESULT_SCHEMA_VERSION!r}.")
    if result.get("case_id") != case["id"]:
        raise ResultError(f"Result {path} case_id does not match {case['id']!r}.")
    raw_runs = result.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ResultError(f"Result {path} runs must be a non-empty list.")

    expected_seeds = set(case["seeds"])
    seen_run_keys: set[tuple[int, str]] = set()
    condition_seeds: dict[str, set[int]] = {}
    conditions: dict[str, dict[str, Any]] = {}
    condition_modes: dict[str, str] = {}
    condition_ids: dict[str, str] = {}
    runs: list[dict[str, Any]] = []
    for index, raw_run in enumerate(raw_runs):
        label = f"result {path} runs[{index}]"
        run = _require_mapping(raw_run, label, ResultError)
        unknown = set(run) - {
            "seed",
            "mode",
            "condition",
            "guard_decision",
            "analysis_status",
            "reported_fact_ids",
            "unsupported_fact_ids",
            "metadata",
        }
        if unknown:
            raise ResultError(f"{label} has unknown keys: {sorted(unknown)}")
        seeds = _seed_list([run.get("seed")], f"{label}.seed", ResultError)
        seed = seeds[0]
        if seed not in expected_seeds:
            raise ResultError(f"{label}.seed {seed} is not declared by the manifest case.")
        mode = _require_string(run.get("mode"), f"{label}.mode", ResultError)
        if mode not in GROUNDING_MODES:
            raise ResultError(f"{label}.mode must be one of {sorted(GROUNDING_MODES)}.")
        condition = _validate_condition(run.get("condition"), f"{label}.condition")
        condition_key = _condition_key(condition)
        run_key = (seed, condition_key)
        if run_key in seen_run_keys:
            raise ResultError(
                f"{label} duplicates seed {seed} for condition {condition['condition_id']!r}."
            )
        seen_run_keys.add(run_key)
        prior_key = condition_ids.get(condition["condition_id"])
        if prior_key is not None and prior_key != condition_key:
            raise ResultError(
                f"{label}.condition_id {condition['condition_id']!r} maps to conflicting condition values."
            )
        condition_ids[condition["condition_id"]] = condition_key
        prior_mode = condition_modes.get(condition_key)
        if prior_mode is not None and prior_mode != mode:
            raise ResultError(
                f"Condition {condition['condition_id']!r} cannot mix modes {prior_mode!r} and {mode!r}."
            )
        condition_modes[condition_key] = mode
        condition_seeds.setdefault(condition_key, set()).add(seed)
        conditions[condition_key] = condition
        if condition["release_candidate"] and mode != "strict":
            raise ResultError(f"{label}: release_candidate conditions must run in strict mode.")
        decision = _require_string(run.get("guard_decision"), f"{label}.guard_decision", ResultError)
        if decision not in GUARD_DECISIONS:
            raise ResultError(f"{label}.guard_decision must be one of {sorted(GUARD_DECISIONS)}.")
        allowed_decisions = {
            "off": {"disabled"},
            "audit": {"pass", "warn"},
            "strict": {"pass", "block"},
        }[mode]
        if decision not in allowed_decisions:
            raise ResultError(
                f"{label}.guard_decision {decision!r} is invalid for mode {mode!r}."
            )
        status = _require_string(run.get("analysis_status"), f"{label}.analysis_status", ResultError)
        if status not in ANALYSIS_STATUSES:
            raise ResultError(f"{label}.analysis_status must be one of {sorted(ANALYSIS_STATUSES)}.")
        reported = _string_list(run.get("reported_fact_ids", []), f"{label}.reported_fact_ids", ResultError)
        unsupported = _string_list(
            run.get("unsupported_fact_ids", []), f"{label}.unsupported_fact_ids", ResultError
        )
        if not set(unsupported) <= set(reported):
            raise ResultError(f"{label}.unsupported_fact_ids must be a subset of reported_fact_ids.")
        metadata = run.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ResultError(f"{label}.metadata must be an object when supplied.")
        runs.append(
            {
                "seed": seed,
                "mode": mode,
                "condition": condition,
                "condition_key": condition_key,
                "guard_decision": decision,
                "analysis_status": status,
                "reported_fact_ids": reported,
                "unsupported_fact_ids": unsupported,
            }
        )
    for condition_key, present_seeds in condition_seeds.items():
        missing = sorted(expected_seeds - present_seeds)
        if missing:
            condition = conditions[condition_key]
            raise ResultError(
                f"Result {path} condition {condition['condition_id']!r} is incomplete; "
                f"missing declared seeds {missing}."
            )
    release_conditions = [
        condition for condition in conditions.values() if condition["release_candidate"]
    ]
    if len(release_conditions) > 1:
        raise ResultError(f"Result {path} declares more than one release_candidate condition.")
    return runs


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score_run(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    expected = {fact["id"] for fact in case["expected_facts"]}
    temporal_expected = {
        fact["id"]
        for fact in case["expected_facts"]
        if "temporal_order" in fact.get("metric_tags", [])
    }
    prohibited = {fact["id"] for fact in case["prohibited_facts"]}
    reported = set(run["reported_fact_ids"])
    explicit_unsupported = set(run["unsupported_fact_ids"])
    supported = (reported & expected) - explicit_unsupported
    prohibited_hits = reported & prohibited
    unsupported = (reported - expected) | explicit_unsupported
    return {
        "case_id": case["id"],
        "category": case["category"],
        "private": case["private"],
        "holdout": case["holdout"],
        "seed": run["seed"],
        "mode": run["mode"],
        "condition": dict(run["condition"]),
        "condition_key": run["condition_key"],
        "guard_decision": run["guard_decision"],
        "analysis_status": run["analysis_status"],
        "expected_fact_count": len(expected),
        "supported_fact_ids": sorted(supported),
        "missing_expected_fact_ids": sorted(expected - supported),
        "temporal_order_expected_fact_ids": sorted(temporal_expected),
        "temporal_order_supported_fact_ids": sorted(supported & temporal_expected),
        "reported_fact_count": len(reported),
        "unsupported_fact_ids": sorted(unsupported),
        "prohibited_fact_hits": sorted(prohibited_hits),
        "counterfactual_group": case["counterfactual_group"],
        "counterfactual_variant": case["counterfactual_variant"],
    }


def _counterfactual_metrics(scored_runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for run in scored_runs:
        group = run["counterfactual_group"]
        if group:
            comparison_key = _counterfactual_condition_key(run["condition"])
            grouped.setdefault((group, comparison_key, run["seed"]), []).append(run)

    pairs: list[dict[str, Any]] = []
    for (group, condition_key, seed), runs in sorted(grouped.items()):
        baseline = next((run for run in runs if run["counterfactual_variant"] in BASELINE_VARIANTS), None)
        if baseline is None:
            continue
        baseline_signature = (
            baseline["guard_decision"],
            tuple(baseline["supported_fact_ids"]),
            tuple(baseline["unsupported_fact_ids"]),
        )
        for alternative in runs:
            if alternative is baseline:
                continue
            alternative_signature = (
                alternative["guard_decision"],
                tuple(alternative["supported_fact_ids"]),
                tuple(alternative["unsupported_fact_ids"]),
            )
            pairs.append(
                {
                    "group": group,
                    "seed": seed,
                    "condition_id": baseline["condition"]["condition_id"],
                    "condition_key": condition_key,
                    "baseline_case_id": baseline["case_id"],
                    "alternative_case_id": alternative["case_id"],
                    "alternative_variant": alternative["counterfactual_variant"],
                    "distinguished": baseline_signature != alternative_signature,
                }
            )
    distinguished = sum(1 for pair in pairs if pair["distinguished"])
    return {
        "pair_count": len(pairs),
        "distinguished_pair_count": distinguished,
        "discrimination_rate": _ratio(distinguished, len(pairs)),
        "pairs": pairs,
    }


def _release_slice_metrics(scored_runs: list[dict[str, Any]]) -> dict[str, Any]:
    expected_total = sum(run["expected_fact_count"] for run in scored_runs)
    supported_total = sum(len(run["supported_fact_ids"]) for run in scored_runs)
    temporal_expected = sum(len(run["temporal_order_expected_fact_ids"]) for run in scored_runs)
    temporal_supported = sum(len(run["temporal_order_supported_fact_ids"]) for run in scored_runs)
    clean_runs = [run for run in scored_runs if run["category"] == "private_clean"]
    clean_passes = sum(run["guard_decision"] == "pass" for run in clean_runs)
    clean_blocks = sum(run["guard_decision"] == "block" for run in clean_runs)
    known_failure_runs = [
        run for run in scored_runs if run["category"] == "private_known_failure"
    ]
    known_failure_silent_passes = sum(
        run["guard_decision"] != "block" and bool(run["unsupported_fact_ids"])
        for run in known_failure_runs
    )
    return {
        "case_count": len({run["case_id"] for run in scored_runs}),
        "run_count": len(scored_runs),
        "supported_fact_recall": _ratio(supported_total, expected_total),
        "temporal_order_accuracy": _ratio(temporal_supported, temporal_expected),
        "temporal_order_expected_fact_count": temporal_expected,
        "clean_control_run_count": len(clean_runs),
        "clean_strict_pass_rate": _ratio(clean_passes, len(clean_runs)),
        "clean_false_block_rate": _ratio(clean_blocks, len(clean_runs)),
        "known_failure_run_count": len(known_failure_runs),
        "known_failure_silent_pass_count": known_failure_silent_passes,
    }


def _release_manifest_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    category_counts = Counter(case["category"] for case in manifest["cases"])
    holdout_counts = Counter(
        case["category"] for case in manifest["cases"] if case["holdout"]
    )
    reasons: list[str] = []
    if dict(category_counts) != RELEASE_CATEGORY_COUNTS:
        reasons.append("release_category_counts_mismatch")
    if dict(holdout_counts) != RELEASE_HOLDOUT_COUNTS:
        reasons.append("release_holdout_counts_mismatch")
    if len(manifest["seeds"]) != 3:
        reasons.append("release_requires_exactly_three_manifest_seeds")
    if any(case["seeds"] != manifest["seeds"] for case in manifest["cases"]):
        reasons.append("release_cases_must_use_manifest_seed_set")
    if not any(
        "temporal_order" in fact.get("metric_tags", [])
        for case in manifest["cases"]
        for fact in case["expected_facts"]
    ):
        reasons.append("release_manifest_has_no_temporal_order_facts")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "category_counts": dict(sorted(category_counts.items())),
        "required_category_counts": dict(RELEASE_CATEGORY_COUNTS),
        "holdout_counts": dict(sorted(holdout_counts.items())),
        "required_holdout_counts": dict(RELEASE_HOLDOUT_COUNTS),
        "seed_count": len(manifest["seeds"]),
    }


def _release_completeness(
    manifest: dict[str, Any],
    release_runs: list[dict[str, Any]],
    fixture_case_ids: set[str],
    result_case_ids: set[str],
) -> dict[str, Any]:
    expected_ids = {case["id"] for case in manifest["cases"]}
    cases_by_id = {case["id"]: case for case in manifest["cases"]}
    release_case_ids = {run["case_id"] for run in release_runs}
    expected_run_count = sum(len(case["seeds"]) for case in manifest["cases"])
    stable_signatures = {
        (
            run["condition"]["condition_id"],
            run["condition"]["prompting"],
            run["condition"]["sampling_profile"],
            run["condition"]["precision"],
            run["condition"]["telemetry_enabled"],
            run["condition"]["model_revision"],
        )
        for run in release_runs
    }
    reasons: list[str] = []
    missing_fixtures = sorted(expected_ids - fixture_case_ids)
    missing_results = sorted(expected_ids - result_case_ids)
    missing_release_conditions = sorted(expected_ids - release_case_ids)
    if missing_fixtures:
        reasons.append("fixtures_missing")
    if missing_results:
        reasons.append("results_missing")
    if missing_release_conditions:
        reasons.append("release_candidate_conditions_missing")
    if len(release_runs) != expected_run_count:
        reasons.append("release_candidate_run_count_mismatch")
    if len(stable_signatures) != 1:
        reasons.append("release_candidate_core_condition_not_uniform")
    media_variant_mismatches = sorted(
        {
            run["case_id"]
            for run in release_runs
            if run["condition"]["media_variant"]
            != (
                cases_by_id[run["case_id"]]["counterfactual_variant"]
                or "case_default"
            )
        }
    )
    if media_variant_mismatches:
        reasons.append("release_candidate_media_variant_mismatch")
    return {
        "complete": not reasons,
        "reasons": reasons,
        "expected_case_count": len(expected_ids),
        "release_case_count": len(release_case_ids),
        "expected_run_count": expected_run_count,
        "release_run_count": len(release_runs),
        "missing_fixture_case_ids": missing_fixtures,
        "missing_result_case_ids": missing_results,
        "missing_release_condition_case_ids": missing_release_conditions,
        "core_condition_signature_count": len(stable_signatures),
        "media_variant_mismatch_case_ids": media_variant_mismatches,
    }


def _threshold_check(actual: float | int | None, operator: str, target: float | int) -> dict[str, Any]:
    if actual is None:
        passed = False
    elif operator == ">=":
        passed = actual >= target
    elif operator == "<=":
        passed = actual <= target
    elif operator == "==":
        passed = actual == target
    else:  # pragma: no cover - all callers use constants above
        raise ValueError(f"Unsupported threshold operator {operator!r}.")
    return {"actual": actual, "operator": operator, "target": target, "passed": passed}


def _release_candidate_report(
    manifest: dict[str, Any],
    scored_runs: list[dict[str, Any]],
    fixture_case_ids: set[str],
    result_case_ids: set[str],
) -> dict[str, Any]:
    release_runs = [
        run
        for run in scored_runs
        if run["mode"] == "strict" and run["condition"]["release_candidate"]
    ]
    development = [run for run in release_runs if not run["holdout"]]
    holdout = [run for run in release_runs if run["holdout"]]
    overall_metrics = _release_slice_metrics(release_runs)
    development_metrics = _release_slice_metrics(development)
    holdout_metrics = _release_slice_metrics(holdout)
    counterfactual = _counterfactual_metrics(release_runs)
    known_holdout_silent = (
        holdout_metrics["known_failure_silent_pass_count"]
        if holdout_metrics["known_failure_run_count"]
        else None
    )
    checks = {
        "supported_fact_recall": _threshold_check(
            overall_metrics["supported_fact_recall"], ">=", RELEASE_THRESHOLDS["supported_fact_recall"]
        ),
        "clean_strict_pass_rate": _threshold_check(
            overall_metrics["clean_strict_pass_rate"], ">=", RELEASE_THRESHOLDS["clean_strict_pass_rate"]
        ),
        "clean_false_block_rate": _threshold_check(
            overall_metrics["clean_false_block_rate"], "<=", RELEASE_THRESHOLDS["clean_false_block_rate"]
        ),
        "counterfactual_discrimination": _threshold_check(
            counterfactual["discrimination_rate"],
            ">=",
            RELEASE_THRESHOLDS["counterfactual_discrimination"],
        ),
        "temporal_order_accuracy": _threshold_check(
            overall_metrics["temporal_order_accuracy"], ">=", RELEASE_THRESHOLDS["temporal_order_accuracy"]
        ),
        "known_failure_holdout_silent_pass_count": _threshold_check(
            known_holdout_silent,
            "==",
            RELEASE_THRESHOLDS["known_failure_holdout_silent_pass_count"],
        ),
    }
    manifest_contract = _release_manifest_contract(manifest)
    completeness = _release_completeness(
        manifest, release_runs, fixture_case_ids, result_case_ids
    )
    eligible = manifest_contract["valid"] and completeness["complete"]
    thresholds_pass = all(check["passed"] for check in checks.values())
    passed = eligible and thresholds_pass
    status = "pass" if passed else ("fail" if eligible else "incomplete")
    return {
        "status": status,
        "passed": passed,
        "eligible_for_verdict": eligible,
        "manifest_contract": manifest_contract,
        "completeness": completeness,
        "slices": {
            "overall": overall_metrics,
            "development": development_metrics,
            "holdout": holdout_metrics,
        },
        "counterfactual": counterfactual,
        "thresholds": checks,
    }


def evaluate_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = validate_manifest(json.loads(manifest_bytes.decode("utf-8")))
    base = manifest_path.parent
    skipped_cases: list[dict[str, str]] = []
    skipped_runs: list[dict[str, Any]] = []
    scored_runs: list[dict[str, Any]] = []
    fixture_case_ids: set[str] = set()
    result_case_ids: set[str] = set()

    for case in manifest["cases"]:
        fixture_path = _resolve_below(base, case["fixture"], f"Fixture path for {case['id']!r}")
        if not fixture_path.is_file():
            if case["private"]:
                skipped_cases.append({"case_id": case["id"], "reason": "private_fixture_missing"})
                continue
            raise ManifestError(f"Public fixture for case {case['id']!r} is missing: {fixture_path}")
        fixture_case_ids.add(case["id"])
        actual_fixture_sha256 = _sha256_file(fixture_path)
        if actual_fixture_sha256 != case["fixture_sha256"]:
            raise ManifestError(
                f"Fixture hash mismatch for case {case['id']!r}: expected "
                f"{case['fixture_sha256']}, got {actual_fixture_sha256}."
            )
        result_path = _resolve_below(base, case["result"], f"Result path for {case['id']!r}")
        if not result_path.is_file():
            skipped_cases.append({"case_id": case["id"], "reason": "result_missing"})
            continue
        runs = _validate_result(_load_json(result_path, ResultError, "result"), case, result_path)
        result_case_ids.add(case["id"])
        scored_runs.extend(_score_run(case, run) for run in runs)

    expected_total = sum(run["expected_fact_count"] for run in scored_runs)
    supported_total = sum(len(run["supported_fact_ids"]) for run in scored_runs)
    reported_total = sum(run["reported_fact_count"] for run in scored_runs)
    unsupported_total = sum(len(run["unsupported_fact_ids"]) for run in scored_runs)
    prohibited_total = sum(
        len(next(case for case in manifest["cases"] if case["id"] == run["case_id"])["prohibited_facts"])
        for run in scored_runs
    )
    prohibited_hits = sum(len(run["prohibited_fact_hits"]) for run in scored_runs)
    silent_hallucinations = sum(
        1
        for run in scored_runs
        if run["guard_decision"] != "block" and bool(run["unsupported_fact_ids"])
    )
    known_failure_runs = [run for run in scored_runs if run["category"] == "private_known_failure"]
    contained_known_failures = sum(
        1
        for run in known_failure_runs
        if run["guard_decision"] == "block" or not run["unsupported_fact_ids"]
    )

    release_candidate = _release_candidate_report(
        manifest, scored_runs, fixture_case_ids, result_case_ids
    )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "manifest": {
            "schema_version": manifest["schema_version"],
            "name": manifest["name"],
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "case_count": len(manifest["cases"]),
        },
        "metrics": {
            "scored_case_count": len({run["case_id"] for run in scored_runs}),
            "scored_run_count": len(scored_runs),
            "skipped_case_count": len(skipped_cases),
            "skipped_run_count": len(skipped_runs),
            "supported_fact_recall": _ratio(supported_total, expected_total),
            "supported_fact_precision": _ratio(supported_total, reported_total),
            "unsupported_fact_rate": _ratio(unsupported_total, reported_total),
            "prohibited_fact_hit_rate": _ratio(prohibited_hits, prohibited_total),
            "silent_hallucination_run_count": silent_hallucinations,
            "known_failure_containment_rate": _ratio(contained_known_failures, len(known_failure_runs)),
        },
        "counterfactual": _counterfactual_metrics(scored_runs),
        "release_candidate": release_candidate,
        "skipped_cases": skipped_cases,
        "skipped_runs": skipped_runs,
        "runs": scored_runs,
        "scoring_notes": [
            "Fact scores use exact manifest fact IDs; aliases are descriptive and are not fuzzy-matched.",
            "Missing private fixtures are skips. Missing public fixtures are manifest errors.",
            "Every recorded experiment condition must contain every seed declared by its case.",
            "Release thresholds use only strict runs whose typed condition declares release_candidate=true.",
            "A release verdict is impossible while any fixture, result, release condition, or declared seed is missing.",
            "Counterfactual pairs are distinguished when guard decision or scored fact sets differ.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a versioned grounding benchmark entirely offline.")
    parser.add_argument("manifest", type=Path, help="Path to a dg-grounding-benchmark/1 JSON manifest.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional report path. Without this option the script writes no files.",
    )
    args = parser.parse_args(argv)
    try:
        report = evaluate_manifest(args.manifest)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0
    except (ManifestError, ResultError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
