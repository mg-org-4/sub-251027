"""Run the local changed-file quality gate used for pre-push and manual checks."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_CC = "25"
XENON_TARGET_RANK = "D"


def _git_output(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()


def _changed_files(*, suffixes: tuple[str, ...]) -> list[str]:
    diffs: list[str] = []
    base_ref = os.environ.get("GITHUB_BASE_REF", "").strip()
    try:
        if base_ref:
            diffs.append(_git_output(["git", "diff", "--diff-filter=AM", "--name-only", f"origin/{base_ref}...HEAD"]))
        else:
            try:
                diffs.append(_git_output(["git", "diff", "--diff-filter=AM", "--name-only", "@{upstream}...HEAD"]))
            except Exception:
                diffs.append(_git_output(["git", "diff", "--diff-filter=AM", "--name-only", "HEAD~1...HEAD"]))
    except Exception:
        pass

    for cmd in (
        ["git", "diff", "--diff-filter=AM", "--name-only"],
        ["git", "diff", "--cached", "--diff-filter=AM", "--name-only"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ):
        try:
            diffs.append(_git_output(cmd))
        except Exception:
            continue

    changed: list[str] = []
    for diff in diffs:
        for raw in diff.splitlines():
            candidate = raw.strip()
            if not candidate or not candidate.endswith(suffixes):
                continue
            path = ROOT / candidate
            if path.exists():
                changed.append(candidate)
    return sorted(dict.fromkeys(changed))


def _run(cmd: list[str], *, label: str, env: dict[str, str] | None = None) -> None:
    print(f"\n==> {label}")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def _python_cmd(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _npm_cmd(*args: str) -> list[str]:
    candidates = ["npm.cmd", "npm"] if os.name == "nt" else ["npm"]
    for candidate in candidates:
        if shutil.which(candidate):
            return [candidate, *args]
    raise SystemExit("npm is required for Vitest checks but was not found in PATH.")


def _npm_available() -> bool:
    return any(shutil.which(candidate) for candidate in (("npm.cmd", "npm") if os.name == "nt" else ("npm",)))


def main() -> int:
    changed_python = _changed_files(suffixes=(".py",))
    changed_vitest = _changed_files(suffixes=(".vitest.mjs", ".test.mjs"))

    if changed_python:
        _run(
            _python_cmd("ruff", "check", *changed_python),
            label="Ruff lint (changed Python files)",
        )
        _run(
            _python_cmd("xenon", "-b", XENON_TARGET_RANK, "-m", XENON_TARGET_RANK, "-a", "B", *changed_python),
            label="Xenon changed-file complexity gate",
        )
        _run(
            [sys.executable, "scripts/check_cc_changed.py"],
            label="Radon changed-file complexity gate",
            env={**os.environ, "MJR_COMPLEXITY_MAX_CC": MAX_CC},
        )
    else:
        print("\n==> Changed-file Python gate")
        print("No changed Python files detected; skipping incremental Python checks.")

    if changed_python:
        _run(
            _python_cmd("mypy", "--config-file", "mypy.ini", "--follow-imports=silent", *changed_python),
            label="Mypy (changed Python files)",
        )
    else:
        print("\n==> Mypy (changed Python files)")
        print("No changed Python files detected; skipping incremental mypy checks.")

    if changed_vitest:
        if not _npm_available():
            raise SystemExit("npm is required for Vitest checks but was not found in PATH.")
        _run(_npm_cmd("exec", "vitest", "run", *changed_vitest), label="Vitest (changed test files)")
    else:
        print("\n==> Vitest (changed test files)")
        print("No changed Vitest files detected; skipping incremental frontend tests.")

    print("\nChanged-file quality gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
