"""All-scope listing helper extracted from ``listing_scopes``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from aiohttp import web

from mjr_am_backend.shared import Result


async def handle_all_scope(
    *,
    query: str,
    limit: int,
    offset: int,
    sort_key: str,
    filters: dict[str, Any],
    include_total: bool,
    require_services: Callable[[], Any],
    touch_enrichment_pause: Callable[..., Any],
    runtime_output_root: Callable[[Any], Any],
    get_input_directory: Callable[[], str],
    kickoff_background_scan: Callable[..., Any],
    list_filesystem_assets: Callable[..., Any],
    dedupe_result_assets_payload: Callable[[dict[str, Any]], dict[str, Any]],
    is_under_root: Callable[[str, str], bool],
    exclude_assets_under_root: Callable[[list[dict[str, Any]], str], list[dict[str, Any]]],
    json_response: Callable[[Any], web.Response],
    search_chunk_min: int,
    search_chunk_max: int,
    search_merge_trim_start: int,
    search_merge_trim_ratio: int,
) -> web.Response:
    svc, error_result = await require_services()
    if error_result:
        return json_response(error_result)
    touch_enrichment_pause(svc, seconds=1.5)

    output_root = await runtime_output_root(svc)
    input_root = str(Path(get_input_directory()).resolve(strict=False))

    input_indexed = False
    try:
        has_input = await svc["index"].has_assets_under_root(input_root)
        if has_input.ok:
            input_indexed = bool(has_input.data)
    except Exception:
        input_indexed = False

    if not input_indexed and query == "*" and offset == 0 and not filters:
        await kickoff_background_scan(
            str(Path(input_root)),
            source="input",
            recursive=False,
            incremental=True,
        )

    if input_indexed:
        scoped = await svc["index"].search_scoped(
            query,
            roots=[output_root, input_root],
            limit=limit,
            offset=offset,
            filters=filters or None,
            include_total=include_total,
            sort=sort_key,
        )
        if not scoped.ok:
            return json_response(scoped)
        data = scoped.data or {}
        assets = data.get("assets") or []

        for asset in assets:
            source = str((asset or {}).get("source") or "").strip().lower()
            if source in ("input", "output", "custom"):
                asset["type"] = source
            else:
                filepath = str((asset or {}).get("filepath") or "")
                asset["type"] = "input" if filepath and is_under_root(input_root, filepath) else "output"

        payload = dedupe_result_assets_payload(
            {
                "assets": assets,
                "total": int(data.get("total") or 0),
                "limit": limit,
                "offset": offset,
                "query": query,
                "scope": "all",
                "sort": sort_key,
            }
        )
        return json_response(Result.Ok(payload))

    if not limit:
        out_count = await svc["index"].search_scoped(
            query,
            roots=[output_root],
            limit=0,
            offset=0,
            filters={**(filters or {}), "source": "output"},
            include_total=True,
            sort=sort_key,
        )
        if not out_count.ok:
            return json_response(out_count)
        in_count = await list_filesystem_assets(
            Path(input_root),
            "",
            query,
            0,
            0,
            asset_type="input",
            filters=filters or None,
            sort=sort_key,
        )
        if not in_count.ok:
            return json_response(in_count)
        total = int((out_count.data or {}).get("total") or 0) + int((in_count.data or {}).get("total") or 0)
        return json_response(
            Result.Ok(
                {
                    "assets": [],
                    "total": total,
                    "limit": 0,
                    "offset": offset,
                    "query": query,
                    "scope": "all",
                    "sort": sort_key,
                }
            )
        )

    chunk = max(search_chunk_min, min(search_chunk_max, int(limit) * 2))
    out_offset = 0
    in_offset = 0
    out_buf: list[dict[str, Any]] = []
    in_buf: list[dict[str, Any]] = []
    out_i = 0
    in_i = 0
    out_total: int | None = None
    in_total: int | None = None

    def _mtime(item: dict[str, Any]) -> int:
        try:
            return int((item or {}).get("mtime") or 0)
        except Exception:
            return 0

    def _name(item: dict[str, Any]) -> str:
        try:
            return str((item or {}).get("filename") or "").lower()
        except Exception:
            return ""

    def _path(item: dict[str, Any]) -> str:
        try:
            return str((item or {}).get("filepath") or "").lower()
        except Exception:
            return ""

    def _asset_key(item: dict[str, Any]) -> tuple[Any, str]:
        if sort_key == "name_asc":
            return (_name(item), _path(item))
        if sort_key == "name_desc":
            return (_name(item), _path(item))
        if sort_key == "mtime_asc":
            return (_mtime(item), _path(item))
        return (_mtime(item), _path(item))

    def _pick_output_item(out_item: dict[str, Any], in_item: dict[str, Any]) -> bool:
        out_key = _asset_key(out_item)
        in_key = _asset_key(in_item)
        if sort_key == "name_desc":
            return out_key >= in_key
        if sort_key == "mtime_desc":
            return out_key >= in_key
        return out_key <= in_key

    async def _fill_out() -> Any:
        nonlocal out_total, out_offset
        if out_total is not None and out_offset >= out_total:
            return None
        res = await svc["index"].search_scoped(
            query,
            roots=[output_root],
            limit=chunk,
            offset=out_offset,
            filters={**(filters or {}), "source": "output"},
            include_total=True,
            sort=sort_key,
        )
        if not res.ok:
            return res
        data = res.data or {}
        incoming_total = data.get("total")
        if incoming_total is not None:
            out_total = int(incoming_total or 0)
        items = data.get("assets") or []
        for asset in items:
            asset["type"] = "output"
        out_offset += len(items)
        out_buf.extend(items)
        return None

    async def _fill_in() -> Any:
        nonlocal in_total, in_offset
        if in_total is not None and in_offset >= in_total:
            return None
        res = await list_filesystem_assets(
            Path(input_root),
            "",
            query,
            chunk,
            in_offset,
            asset_type="input",
            filters=filters or None,
            sort=sort_key,
        )
        if not res.ok:
            return res
        data = res.data or {}
        in_total = int(data.get("total") or 0)
        items = data.get("assets") or []
        in_offset += len(items)
        in_buf.extend(items)
        return None

    err = await _fill_out()
    if err:
        return json_response(err)
    err = await _fill_in()
    if err:
        return json_response(err)

    total = int(out_total or 0) + int(in_total or 0)
    target = offset + limit
    produced = 0
    page: list[dict[str, Any]] = []

    while produced < target:
        if out_i >= len(out_buf) and (out_total is None or out_offset < out_total):
            err = await _fill_out()
            if err:
                return json_response(err)
        if in_i >= len(in_buf) and (in_total is None or in_offset < in_total):
            err = await _fill_in()
            if err:
                return json_response(err)

        if out_i >= len(out_buf) and in_i >= len(in_buf):
            break

        pick_out = False
        out_has = out_i < len(out_buf)
        in_has = in_i < len(in_buf)
        if out_has and in_has:
            pick_out = _pick_output_item(out_buf[out_i], in_buf[in_i])
        elif out_has:
            pick_out = True

        if pick_out:
            item = out_buf[out_i]
            out_i += 1
        else:
            item = in_buf[in_i]
            in_i += 1

        if out_i > search_merge_trim_start and out_i >= len(out_buf) // search_merge_trim_ratio:
            out_buf = out_buf[out_i:]
            out_i = 0
        if in_i > search_merge_trim_start and in_i >= len(in_buf) // search_merge_trim_ratio:
            in_buf = in_buf[in_i:]
            in_i = 0

        if produced >= offset:
            page.append(item)
            if len(page) >= limit:
                break
        produced += 1

    payload = dedupe_result_assets_payload(
        {
            "assets": page,
            "total": total,
            "limit": limit,
            "offset": offset,
            "query": query,
            "scope": "all",
            "sort": sort_key,
        }
    )
    return json_response(Result.Ok(payload))
