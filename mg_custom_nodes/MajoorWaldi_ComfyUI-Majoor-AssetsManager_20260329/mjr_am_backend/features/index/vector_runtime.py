"""
Shared lazy vector-service initialization.

Centralizes runtime creation of VectorService/VectorSearcher so route handlers
and indexing flows do not each build their own instances with divergent state.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ...config import is_vector_search_enabled


def _get_vector_runtime_lock(services: dict[str, Any]) -> asyncio.Lock:
    lock = services.get("_vector_runtime_lock")
    if isinstance(lock, asyncio.Lock):
        return lock
    lock = asyncio.Lock()
    services["_vector_runtime_lock"] = lock
    return lock


def _build_vector_runtime(db: Any) -> tuple[Any, Any]:
    from .vector_searcher import VectorSearcher
    from .vector_service import VectorService

    vector_service = VectorService()
    vector_searcher = VectorSearcher(db, vector_service)
    return vector_service, vector_searcher


async def ensure_vector_runtime(
    services: dict[str, Any] | None,
    *,
    logger: Any = None,
    reason: str = "runtime",
) -> tuple[Any | None, Any | None]:
    if not is_vector_search_enabled() or not isinstance(services, dict):
        return None, None

    vector_service = services.get("vector_service")
    vector_searcher = services.get("vector_searcher")
    if vector_service is not None and vector_searcher is not None:
        return vector_service, vector_searcher

    db = services.get("db")
    if db is None:
        return None, None

    lock = _get_vector_runtime_lock(services)
    async with lock:
        vector_service = services.get("vector_service")
        vector_searcher = services.get("vector_searcher")
        if vector_service is not None and vector_searcher is not None:
            return vector_service, vector_searcher
        vector_service, vector_searcher = _build_vector_runtime(db)
        services["vector_service"] = vector_service
        services["vector_searcher"] = vector_searcher
        if logger is not None:
            try:
                logger.info("Vector services initialized lazily (%s)", str(reason or "runtime"))
            except Exception:
                pass
        return vector_service, vector_searcher
