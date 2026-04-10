"""Custom-scope listing helper extracted from ``listing_scopes``."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from aiohttp import web

from mjr_am_backend.shared import Result


async def handle_custom_scope(
    *,
    request: web.Request,
    query: str,
    limit: int,
    offset: int,
    sort_key: str,
    filters: dict[str, Any],
    subfolder: str,
    root_id: str,
    require_services: Callable[[], Any],
    touch_enrichment_pause: Callable[..., Any],
    resolve_custom_root_fn: Callable[[str], Any],
    is_loopback_request: Callable[[web.Request], bool],
    browser_mode_needs_post_filters: Callable[[dict[str, Any] | None], bool],
    hydrate_browser_assets_from_db: Callable[[Any, list[dict[str, Any]]], Any],
    postfilter_browser_assets: Callable[..., tuple[list[dict[str, Any]], int]],
    kickoff_background_scan: Callable[..., Any],
    list_filesystem_assets: Callable[..., Any],
    dedupe_result_assets_payload: Callable[[dict[str, Any]], dict[str, Any]],
    json_response: Callable[[Any], web.Response],
    logger: Any,
) -> web.Response:
    from mjr_am_backend.features.browser import (
        list_filesystem_browser_entries,
        list_visible_subfolders,
    )

    browser_mode = str(request.query.get("browser_mode", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not str(root_id or "").strip():
        if not browser_mode:
            return json_response(Result.Err("INVALID_INPUT", "Missing custom_root_id"))
        if not is_loopback_request(request):
            return json_response(Result.Err("FORBIDDEN", "Custom browser mode is loopback-only"))

        svc, _ = await require_services()
        needs_post_filters = browser_mode_needs_post_filters(filters or None)
        browser_limit = 0 if needs_post_filters else limit
        browser_offset = 0 if needs_post_filters else offset
        browser_result = await asyncio.to_thread(
            list_filesystem_browser_entries,
            subfolder,
            query,
            browser_limit,
            browser_offset,
            kind_filter=str(filters.get("kind") or "") if filters else "",
            min_size_bytes=int((filters or {}).get("min_size_bytes") or 0),
            max_size_bytes=int((filters or {}).get("max_size_bytes") or 0),
            min_width=int((filters or {}).get("min_width") or 0),
            min_height=int((filters or {}).get("min_height") or 0),
            max_width=int((filters or {}).get("max_width") or 0),
            max_height=int((filters or {}).get("max_height") or 0),
        )
        try:
            if browser_result.ok and isinstance(browser_result.data, dict):
                assets = browser_result.data.get("assets") or []
                hydrated_assets = await hydrate_browser_assets_from_db(svc, assets)
                if needs_post_filters:
                    paged_assets, total = postfilter_browser_assets(
                        hydrated_assets,
                        query=query,
                        filters=filters or None,
                        sort_key=sort_key,
                        limit=limit,
                        offset=offset,
                    )
                    browser_result.data["assets"] = paged_assets
                    browser_result.data["total"] = total
                    browser_result.data["limit"] = limit
                    browser_result.data["offset"] = offset
                    browser_result.data["query"] = query
                else:
                    browser_result.data["assets"] = hydrated_assets
                browser_result.data = dedupe_result_assets_payload(browser_result.data)
        except Exception as exc:
            logger.debug("Failed to hydrate browser assets from DB: %s", exc)
            if browser_result.ok and isinstance(browser_result.data, dict):
                raw_assets = browser_result.data.get("assets") or []
                paged_assets, total = postfilter_browser_assets(
                    raw_assets,
                    query=query,
                    filters=filters or None,
                    sort_key=sort_key,
                    limit=limit,
                    offset=offset,
                )
                browser_result.data["assets"] = paged_assets
                browser_result.data["total"] = total
                browser_result.data["limit"] = limit
                browser_result.data["offset"] = offset
        return json_response(browser_result)

    root_result = resolve_custom_root_fn(str(root_id or ""))
    if not root_result.ok:
        return json_response(root_result)
    root_dir = root_result.data
    svc, _ = await require_services()
    touch_enrichment_pause(svc, seconds=1.5)

    if query == "*" and offset == 0 and not filters:
        await kickoff_background_scan(
            str(root_dir),
            source="custom",
            root_id=str(root_id),
            recursive=False,
            incremental=True,
        )
    result = await list_filesystem_assets(
        root_dir,
        subfolder,
        query,
        limit,
        offset,
        asset_type="custom",
        root_id=str(root_id),
        filters=filters or None,
        index_service=(svc or {}).get("index") if isinstance(svc, dict) else None,
        sort=sort_key,
    )
    if result.ok and isinstance(result.data, dict):
        result.data = dedupe_result_assets_payload(result.data)
        folder_result = list_visible_subfolders(root_dir, subfolder, str(root_id or ""))
        if not folder_result.ok:
            return json_response(folder_result)
        folder_assets = folder_result.data or []
        if query not in ("", "*"):
            query_lower = query.lower()
            folder_assets = [
                item
                for item in folder_assets
                if query_lower in str(item.get("filename") or "").lower()
            ]
        if filters and filters.get("kind"):
            folder_assets = []
        file_assets = result.data.get("assets") or []
        hybrid_assets = folder_assets + file_assets if offset == 0 else file_assets
        result.data["assets"] = hybrid_assets
        try:
            base_total = int(result.data.get("total") or 0)
        except Exception:
            base_total = len(file_assets)
        result.data["total"] = base_total + (len(folder_assets) if offset == 0 else 0)
        result.data["scope"] = "custom"
    return json_response(result)
