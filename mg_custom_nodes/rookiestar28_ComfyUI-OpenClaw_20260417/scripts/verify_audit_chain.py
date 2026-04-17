"""Verify the retained OpenClaw audit hash chain across active and rotated files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the retained OpenClaw audit hash chain."
    )
    parser.add_argument(
        "--path",
        default="",
        help="Explicit audit log path. Defaults to the configured OpenClaw audit path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a human summary.",
    )
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(_repo_root()))
    from services.audit import verify_audit_chain

    result = verify_audit_chain(args.path or None)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        status = "PASS" if result.ok else "FAIL"
        print(f"Audit Chain Verification: {status}")
        print(f"Files checked: {len(result.files_checked)}")
        print(f"Entries checked: {result.entries_checked}")
        print(f"Window start prev_hash: {result.window_start_prev_hash}")
        print(f"Terminal hash: {result.terminal_hash}")
        print(f"Window truncated: {result.window_truncated}")
        if result.issues:
            print("Issues:")
            for issue in result.issues:
                print(
                    f"- {issue.code} at {issue.file_path}:{issue.line_number} -> {issue.message}"
                )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
