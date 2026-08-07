"""Portable digest and write helpers for governed text contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def normalize_text_newlines(payload: bytes) -> bytes:
    """Return text bytes with CRLF and lone CR represented as LF."""
    # IMPORTANT: normalize text newlines; raw hashing breaks frozen contracts after Windows checkout.
    return payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def stable_text_digest(path: Path) -> str:
    """Hash governed text independently of checkout newline representation."""
    return hashlib.sha256(normalize_text_newlines(path.read_bytes())).hexdigest()


def write_text_lf(path: Path, text: str) -> None:
    """Write UTF-8 contract text with explicit LF newlines on every platform."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
