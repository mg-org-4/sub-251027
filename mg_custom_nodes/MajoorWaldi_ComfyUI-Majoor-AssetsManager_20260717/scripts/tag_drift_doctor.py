"""Compare legacy JSON tags with normalized asset_tags rows.

Usage:
    python scripts/tag_drift_doctor.py path/to/index.sqlite

Post-v19 databases no longer have ``asset_metadata.tags``; in that case the
doctor reports OK and exits successfully.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _legacy_tags(raw: object) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        if not isinstance(item, str):
            continue
        tag = item.strip()
        key = tag.casefold()
        if not tag or key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return sorted(out, key=str.casefold)


def _normalized_tags(conn: sqlite3.Connection, asset_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT t.name FROM asset_tags at "
        "JOIN tags t ON t.id = at.tag_id "
        "WHERE at.asset_id = ? ORDER BY t.name COLLATE NOCASE",
        (asset_id,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def check_tag_drift(db_path: Path) -> tuple[int, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        columns = _table_columns(conn, "asset_metadata")
        if "tags" not in columns:
            print("OK: asset_metadata.tags is absent (post-v19 database).")
            return 0, 0

        rows = conn.execute(
            "SELECT asset_id, tags FROM asset_metadata "
            "WHERE COALESCE(tags, '') NOT IN ('', '[]')"
        ).fetchall()
        drift = 0
        for asset_id_raw, raw_tags in rows:
            asset_id = int(asset_id_raw)
            legacy = _legacy_tags(raw_tags)
            normalized = _normalized_tags(conn, asset_id)
            if [tag.casefold() for tag in legacy] != [tag.casefold() for tag in normalized]:
                drift += 1
                print(
                    f"DRIFT asset_id={asset_id}: "
                    f"legacy={legacy!r} normalized={normalized!r}"
                )
        checked = len(rows)
        if drift:
            print(f"FAIL: {drift} drifted asset(s) out of {checked} checked.")
        else:
            print(f"OK: no tag drift across {checked} legacy-tagged asset(s).")
        return checked, drift
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path, help="Path to the Majoor SQLite index database")
    args = parser.parse_args(argv)
    if not args.db.exists():
        print(f"Database not found: {args.db}", file=sys.stderr)
        return 2
    _checked, drift = check_tag_drift(args.db)
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
