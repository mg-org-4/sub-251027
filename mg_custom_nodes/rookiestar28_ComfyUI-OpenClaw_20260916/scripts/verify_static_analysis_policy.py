"""Incremental Ruff/Mypy debt-ratchet verifier.

The policy owns source paths and normalized diagnostic counts. Line numbers are
intentionally excluded so harmless edits do not churn the baseline; count changes
still fail and require an explicit reviewed baseline refresh.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, order=True)
class Diagnostic:
    tool: str
    path: str
    code: str
    message: str


class _ProcessResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


class ToolExecutionError(RuntimeError):
    """Raised with content-free context when a quality tool cannot run."""


def _normalize_message(message: Any, repo_root: Path | None = None) -> str:
    normalized = " ".join(str(message or "").split())
    if repo_root is not None:
        variants = {
            str(repo_root.resolve()),
            str(repo_root.resolve()).replace("\\", "/"),
        }
        for variant in sorted(variants, key=len, reverse=True):
            if variant:
                normalized = normalized.replace(variant, "<repo>")
    return normalized


def _is_safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    candidate = Path(value)
    return not candidate.is_absolute() and ".." not in candidate.parts


def _repo_relative_path(path_value: Any, repo_root: Path) -> str:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = repo_root / path
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"diagnostic path is outside repository: {path_value}"
        ) from exc
    return relative.as_posix()


def _path_within(path: str, root: str) -> bool:
    path_obj = Path(path)
    root_obj = Path(root)
    return path_obj == root_obj or root_obj in path_obj.parents


def _excluded_path_values(policy: Mapping[str, Any]) -> tuple[str, ...]:
    entries = policy.get("excluded_paths", [])
    if not isinstance(entries, list):
        return ()
    return tuple(
        str(entry.get("path", ""))
        for entry in entries
        if isinstance(entry, dict) and entry.get("path")
    )


def _tracked_python_files(repo_root: Path) -> frozenset[str] | None:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--", "*.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )
    if result.returncode != 0:
        return None
    # IMPORTANT: governance ownership must not include ignored maintainer-local files.
    return frozenset(
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines()
        if line.strip()
    )


def discover_owned_python_files(
    repo_root: Path, policy: Mapping[str, Any]
) -> tuple[str, ...]:
    excluded = _excluded_path_values(policy)
    tracked_files = _tracked_python_files(repo_root)
    discovered: set[str] = set()
    for root_value in policy.get("production_roots", []):
        root_path = repo_root / str(root_value)
        candidates: Iterable[Path]
        if root_path.is_file():
            candidates = (root_path,) if root_path.suffix == ".py" else ()
        elif root_path.is_dir():
            candidates = root_path.rglob("*.py")
        else:
            continue
        for candidate in candidates:
            relative = _repo_relative_path(candidate, repo_root)
            if tracked_files is not None and relative not in tracked_files:
                continue
            if any(_path_within(relative, excluded_path) for excluded_path in excluded):
                continue
            discovered.add(relative)
    return tuple(sorted(discovered))


def _baseline_counter(policy: Mapping[str, Any]) -> Counter[Diagnostic]:
    baseline: Counter[Diagnostic] = Counter()
    entries = policy.get("baseline", [])
    if not isinstance(entries, list):
        return baseline
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        diagnostic = Diagnostic(
            tool=str(entry.get("tool", "")),
            path=str(entry.get("path", "")),
            code=str(entry.get("code", "")),
            message=_normalize_message(entry.get("message", "")),
        )
        count = entry.get("count", 0)
        if isinstance(count, int) and count > 0:
            baseline[diagnostic] += count
    return baseline


def validate_policy(repo_root: Path, policy: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if policy.get("schema_version") != 1:
        failures.append("schema_version must be 1")

    review = policy.get("review")
    if not isinstance(review, dict):
        failures.append("review must be an object")
    else:
        owner = review.get("owner")
        if not isinstance(owner, str) or not owner.strip():
            failures.append("review.owner must be a non-empty string")
        parsed_dates: dict[str, date] = {}
        for key in ("reviewed_at", "next_review_by"):
            try:
                parsed_dates[key] = date.fromisoformat(str(review.get(key, "")))
            except ValueError:
                failures.append(f"review.{key} must be an ISO date")
        if (
            len(parsed_dates) == 2
            and parsed_dates["next_review_by"] < parsed_dates["reviewed_at"]
        ):
            failures.append("review.next_review_by must not precede reviewed_at")

    tools = policy.get("tools")
    if not isinstance(tools, dict) or set(tools) != {"ruff", "mypy"}:
        failures.append("tools must contain exactly ruff and mypy")
        tools = {}
    for tool_name in ("ruff", "mypy"):
        config = tools.get(tool_name)
        if (
            not isinstance(config, dict)
            or not isinstance(config.get("version"), str)
            or not config["version"].strip()
        ):
            failures.append(f"tools.{tool_name}.version must be a non-empty string")

    roots = policy.get("production_roots")
    if not isinstance(roots, list) or not roots:
        failures.append("production_roots must be a non-empty list")
        roots = []
    seen_roots: set[str] = set()
    valid_roots: list[str] = []
    for index, value in enumerate(roots):
        if not _is_safe_relative_path(value):
            failures.append(f"production_roots[{index}] is unsafe: {value!r}")
            continue
        value = str(value)
        if value in seen_roots:
            failures.append(f"duplicate production root: {value}")
            continue
        seen_roots.add(value)
        valid_roots.append(value)
        if not (repo_root / value).exists():
            failures.append(f"production root is missing: {value}")

    excluded_entries = policy.get("excluded_paths")
    if not isinstance(excluded_entries, list):
        failures.append("excluded_paths must be a list")
        excluded_entries = []
    seen_excluded: set[str] = set()
    for index, entry in enumerate(excluded_entries):
        if not isinstance(entry, dict):
            failures.append(f"excluded_paths[{index}] must be an object")
            continue
        value = entry.get("path")
        reason = entry.get("reason")
        if not _is_safe_relative_path(value):
            failures.append(f"excluded_paths[{index}] is unsafe: {value!r}")
            continue
        value = str(value)
        if value in seen_excluded:
            failures.append(f"duplicate excluded path: {value}")
        seen_excluded.add(value)
        if not isinstance(reason, str) or not reason.strip():
            failures.append(f"excluded path {value} is missing a reason")
        if not any(_path_within(value, root) for root in valid_roots):
            failures.append(f"excluded path is outside owned roots: {value}")
        if not (repo_root / value).exists():
            failures.append(f"excluded path is missing: {value}")

    owned_files = set(discover_owned_python_files(repo_root, policy))
    strict_paths = policy.get("strict_paths")
    if not isinstance(strict_paths, list):
        failures.append("strict_paths must be a list")
        strict_paths = []
    seen_strict: set[str] = set()
    for index, value in enumerate(strict_paths):
        if not _is_safe_relative_path(value):
            failures.append(f"strict_paths[{index}] is unsafe: {value!r}")
            continue
        value = str(value)
        if value in seen_strict:
            failures.append(f"duplicate strict path: {value}")
        seen_strict.add(value)
        if not any(_path_within(path, value) for path in owned_files):
            failures.append(f"strict path has no owned Python files: {value}")

    baseline_entries = policy.get("baseline")
    if not isinstance(baseline_entries, list):
        failures.append("baseline must be a list")
        baseline_entries = []
    seen_diagnostics: set[Diagnostic] = set()
    for index, entry in enumerate(baseline_entries):
        if not isinstance(entry, dict):
            failures.append(f"baseline[{index}] must be an object")
            continue
        diagnostic = Diagnostic(
            tool=str(entry.get("tool", "")),
            path=str(entry.get("path", "")),
            code=str(entry.get("code", "")),
            message=_normalize_message(entry.get("message", "")),
        )
        if diagnostic.tool not in {"ruff", "mypy"}:
            failures.append(f"baseline[{index}] has unknown tool {diagnostic.tool!r}")
        if diagnostic.path not in owned_files:
            failures.append(
                f"baseline[{index}] path is not an owned Python file: {diagnostic.path}"
            )
        if not diagnostic.code or not diagnostic.message:
            failures.append(f"baseline[{index}] code/message must be non-empty")
        count = entry.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            failures.append(f"baseline[{index}] count must be a positive integer")
        if diagnostic in seen_diagnostics:
            failures.append(f"duplicate baseline diagnostic: {diagnostic}")
        seen_diagnostics.add(diagnostic)

    if baseline_entries != serialize_baseline(_baseline_counter(policy)):
        failures.append("baseline must use canonical sorted serialization")

    return failures


def validate_tool_versions(
    policy: Mapping[str, Any], detected_versions: Mapping[str, str]
) -> list[str]:
    failures: list[str] = []
    tools = policy.get("tools", {})
    for tool_name in ("ruff", "mypy"):
        config = tools.get(tool_name, {}) if isinstance(tools, dict) else {}
        expected = config.get("version") if isinstance(config, dict) else None
        found = detected_versions.get(tool_name, "missing")
        if expected != found:
            failures.append(
                f"{tool_name} version drift: expected {expected}, found {found}"
            )
    return failures


def validate_requirement_pins(
    policy: Mapping[str, Any], requirement_lines: Iterable[str]
) -> list[str]:
    requirements: dict[str, str] = {}
    for raw_line in requirement_lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(r"^(ruff|mypy)(?:\[.*\])?(.*)$", line, re.IGNORECASE)
        if match:
            requirements[match.group(1).lower()] = (
                match.group(1).lower() + match.group(2).strip()
            )

    failures: list[str] = []
    tools = policy.get("tools", {})
    for tool_name in sorted(("ruff", "mypy")):
        config = tools.get(tool_name, {}) if isinstance(tools, dict) else {}
        version = config.get("version") if isinstance(config, dict) else None
        expected = f"{tool_name}=={version}"
        found = requirements.get(tool_name, "missing")
        if found != expected:
            failures.append(
                f"{tool_name} requirement drift: expected {expected}, found {found}"
            )
    return failures


def _run_command(
    runner: Callable[..., _ProcessResult], command: list[str], repo_root: Path
) -> _ProcessResult:
    return runner(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )


def _parse_tool_version(tool_name: str, output: str) -> str:
    match = re.search(rf"\b{re.escape(tool_name)}\s+([0-9]+(?:\.[0-9]+)+)", output)
    if not match:
        raise ToolExecutionError(f"{tool_name} version output was not recognized")
    return match.group(1)


def run_static_analysis(
    repo_root: Path,
    policy: Mapping[str, Any],
    *,
    runner: Callable[..., _ProcessResult] = subprocess.run,
) -> tuple[dict[str, str], Counter[Diagnostic]]:
    owned_files = list(discover_owned_python_files(repo_root, policy))
    versions: dict[str, str] = {}

    for tool_name in ("ruff", "mypy"):
        command = [sys.executable, "-m", tool_name, "--version"]
        result = _run_command(runner, command, repo_root)
        if result.returncode != 0:
            raise ToolExecutionError(
                f"{tool_name} version check failed with exit code {result.returncode}"
            )
        versions[tool_name] = _parse_tool_version(tool_name, result.stdout)

    commands = (
        (
            "ruff",
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--output-format",
                "json",
                "--no-cache",
                *owned_files,
            ],
        ),
        (
            "mypy",
            [
                sys.executable,
                "-m",
                "mypy",
                "--output",
                "json",
                "--no-incremental",
                "--explicit-package-bases",
                "--no-warn-unused-configs",
                "--no-error-summary",
                "--no-site-packages",
                "--ignore-missing-imports",
                *owned_files,
            ],
        ),
    )

    diagnostics: Counter[Diagnostic] = Counter()
    for tool_name, command in commands:
        result = _run_command(runner, command, repo_root)
        if result.returncode not in {0, 1}:
            # SECURITY: stderr may contain private host paths or source content.
            # Keep the public/loggable failure content-free and deterministic.
            raise ToolExecutionError(
                f"{tool_name} execution failed with exit code {result.returncode}"
            )
        if tool_name == "ruff":
            diagnostics.update(parse_ruff_output(result.stdout, repo_root))
        else:
            diagnostics.update(parse_mypy_output(result.stdout, repo_root))
    return versions, diagnostics


def compare_diagnostics(
    policy: Mapping[str, Any], current: Counter[Diagnostic]
) -> list[str]:
    failures: list[str] = []
    baseline = _baseline_counter(policy)
    strict_paths = tuple(str(value) for value in policy.get("strict_paths", []))

    for diagnostic, count in sorted(current.items()):
        if any(_path_within(diagnostic.path, path) for path in strict_paths):
            failures.append(
                "strict path diagnostic: "
                f"{diagnostic.tool}:{diagnostic.path}:{diagnostic.code} x{count}"
            )

    for diagnostic in sorted(set(baseline) | set(current)):
        expected = baseline.get(diagnostic, 0)
        found = current.get(diagnostic, 0)
        label = (
            f"{diagnostic.tool}:{diagnostic.path}:{diagnostic.code}:"
            f"{diagnostic.message}"
        )
        if found > expected:
            failures.append(f"new debt: {label} expected {expected}, found {found}")
        elif found < expected:
            failures.append(
                f"stale baseline: {label} expected {expected}, found {found}"
            )
    return failures


def _diagnostic_from_payload(
    *, tool: str, payload: Mapping[str, Any], path_key: str, repo_root: Path
) -> Diagnostic:
    return Diagnostic(
        tool=tool,
        path=_repo_relative_path(payload.get(path_key, ""), repo_root),
        code=str(payload.get("code") or "unknown"),
        message=_normalize_message(payload.get("message", ""), repo_root),
    )


def parse_ruff_output(raw: str, repo_root: Path) -> Counter[Diagnostic]:
    payload = json.loads(raw or "[]")
    if not isinstance(payload, list):
        raise ValueError("Ruff JSON output must be a list")
    diagnostics: Counter[Diagnostic] = Counter()
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("Ruff JSON diagnostic must be an object")
        diagnostics[
            _diagnostic_from_payload(
                tool="ruff", payload=entry, path_key="filename", repo_root=repo_root
            )
        ] += 1
    return diagnostics


def parse_mypy_output(raw: str, repo_root: Path) -> Counter[Diagnostic]:
    stripped = raw.strip()
    if not stripped:
        return Counter()
    if stripped.startswith("["):
        payloads = json.loads(stripped)
    else:
        payloads = [json.loads(line) for line in stripped.splitlines() if line.strip()]
    if not isinstance(payloads, list):
        raise ValueError("Mypy JSON output must be a list or JSON lines")
    diagnostics: Counter[Diagnostic] = Counter()
    for entry in payloads:
        if not isinstance(entry, dict):
            raise ValueError("Mypy JSON diagnostic must be an object")
        if entry.get("severity", "error") != "error":
            continue
        diagnostics[
            _diagnostic_from_payload(
                tool="mypy", payload=entry, path_key="file", repo_root=repo_root
            )
        ] += 1
    return diagnostics


def serialize_baseline(diagnostics: Counter[Diagnostic]) -> list[dict[str, Any]]:
    return [
        {
            "tool": diagnostic.tool,
            "path": diagnostic.path,
            "code": diagnostic.code,
            "message": diagnostic.message,
            "count": count,
        }
        for diagnostic, count in sorted(diagnostics.items())
        if count > 0
    ]


def with_updated_baseline(
    policy: Mapping[str, Any], diagnostics: Counter[Diagnostic]
) -> dict[str, Any]:
    updated = deepcopy(dict(policy))
    updated["baseline"] = serialize_baseline(diagnostics)
    return updated


def evaluate_policy(
    repo_root: Path,
    policy: Mapping[str, Any],
    *,
    requirement_lines: Iterable[str],
    runner: Callable[..., _ProcessResult] = subprocess.run,
) -> tuple[list[str], Counter[Diagnostic]]:
    failures = validate_policy(repo_root, policy)
    if failures:
        return failures, Counter()
    failures.extend(validate_requirement_pins(policy, requirement_lines))

    versions, diagnostics = run_static_analysis(repo_root, policy, runner=runner)
    failures.extend(validate_tool_versions(policy, versions))
    failures.extend(compare_diagnostics(policy, diagnostics))
    return failures, diagnostics


def _write_policy(path: Path, policy: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the incremental Ruff/Mypy static-analysis debt policy."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--policy", default="tests/static_analysis_policy.json")
    parser.add_argument("--requirements", default="requirements-quality.txt")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Explicitly replace the accepted diagnostic baseline after review.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    policy_path = repo_root / args.policy
    requirements_path = repo_root / args.requirements
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        requirement_lines = requirements_path.read_text(encoding="utf-8").splitlines()
        structural_failures = validate_policy(repo_root, policy)
        structural_failures.extend(validate_requirement_pins(policy, requirement_lines))
        if structural_failures:
            for failure in structural_failures:
                print(f"STATIC-ANALYSIS-FAIL: {failure}")
            return 1

        versions, diagnostics = run_static_analysis(repo_root, policy)
        failures = validate_tool_versions(policy, versions)
        if args.write_baseline:
            strict_policy = with_updated_baseline(policy, diagnostics)
            strict_failures = [
                failure
                for failure in compare_diagnostics(strict_policy, diagnostics)
                if failure.startswith("strict path diagnostic:")
            ]
            if strict_failures:
                for failure in strict_failures:
                    print(f"STATIC-ANALYSIS-FAIL: {failure}")
                return 1
            if failures:
                for failure in failures:
                    print(f"STATIC-ANALYSIS-FAIL: {failure}")
                return 1
            _write_policy(policy_path, strict_policy)
            print(
                "STATIC-ANALYSIS-BASELINE-WRITTEN: "
                f"{len(diagnostics)} fingerprints, {sum(diagnostics.values())} findings"
            )
            return 0

        failures.extend(compare_diagnostics(policy, diagnostics))
        if failures:
            for failure in failures:
                print(f"STATIC-ANALYSIS-FAIL: {failure}")
            return 1
        print(
            "STATIC-ANALYSIS-PASS: "
            f"{len(diagnostics)} fingerprints, {sum(diagnostics.values())} governed findings"
        )
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"STATIC-ANALYSIS-FAIL: invalid policy or tool output ({type(exc).__name__})"
        )
        return 1
    except ToolExecutionError as exc:
        print(f"STATIC-ANALYSIS-FAIL: {exc}")
        print(
            "STATIC-ANALYSIS-REMEDIATION: use the project-local Python to install "
            "requirements-quality.txt"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
