"""Install the repository's local Git hooks via pre-commit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pre_commit", *args], cwd=ROOT, check=True)


def main() -> int:
    _run("install", "--hook-type", "pre-commit", "--hook-type", "pre-push")
    _run("install-hooks")
    print("Local Git hooks installed for pre-commit and pre-push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
