"""Safe response helpers for connector-served local media."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

DANGEROUS_CONTENT_TYPES = {
    "text/html",
    "text/html-sandboxed",
    "application/xhtml+xml",
    "text/javascript",
    "application/javascript",
    "application/x-javascript",
    "application/ecmascript",
    "text/css",
    "image/svg+xml",
    "application/xml",
    "text/xml",
    "message/rfc822",
}


def is_dangerous_content_type(content_type: str | None) -> bool:
    """Return True for browser-renderable active content types."""
    if not content_type:
        return False
    normalized = content_type.split(";", 1)[0].strip().lower()
    if normalized in DANGEROUS_CONTENT_TYPES:
        return True
    return normalized.endswith("+xml") or normalized.endswith("/xml")


def _content_disposition_filename(name: str) -> str:
    safe_name = name.replace("\r", "").replace("\n", "")
    safe_name = safe_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'filename="{safe_name}"'


def build_connector_media_response(web: Any, path: Path):
    """Build a hardened FileResponse for signed connector media files."""
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    disposition = _content_disposition_filename(path.name)

    # IMPORTANT: connector media is user-controlled. Dangerous active content
    # must download instead of rendering inline in the local media origin.
    if is_dangerous_content_type(content_type):
        content_type = "application/octet-stream"
        disposition = f"attachment; {_content_disposition_filename(path.name)}"

    return web.FileResponse(
        path,
        headers={
            "Content-Disposition": disposition,
            "Content-Type": content_type,
            "X-Content-Type-Options": "nosniff",
        },
    )


__all__ = [
    "DANGEROUS_CONTENT_TYPES",
    "build_connector_media_response",
    "is_dangerous_content_type",
]
