"""All-scope listing helper extracted from ``listing_scopes``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from aiohttp import web
from mjr_am_backend.shared import Result


def _propagate_all_scope_error(result: Result[Any], default_code: str, default_error: str) -> Result[dict[str, Any]]:
    return Result.Err(result.code or default_code, str(result.error or default_error))


def _build_all_scope_payload(
    *,
    assets: list[dict[str, Any]],
    total: int,
    limit: int,
    offset: int,
    query: str,
    sort_key: str,
    dedupe_result_assets_payload: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    return dedupe_result_assets_payload(
        {
            "assets": assets,
            "total": total,
            "limit": limit,
            "offset": offset,
            "query": query,
            "scope": "all",
            "sort": sort_key,
        }
    )


async def _is_input_indexed(svc: dict[str, Any], input_root: str) -> bool:
    try:
        has_input = await svc["index"].has_assets_under_root(input_root)
    except Exception:
        return False
    if not has_input.ok:
        return False
    return bool(has_input.data)


async def _maybe_kickoff_input_scan(
    *,
    input_indexed: bool,
    query: str,
    offset: int,
    filters: dict[str, Any],
    input_root: str,
    kickoff_background_scan: Callable[..., Any],
) -> None:
    if input_indexed or query != "*" or offset != 0 or filters:
        return
    await kickoff_background_scan(
        str(Path(input_root)),
        source="input",
        recursive=False,
        incremental=True,
    )


def _annotate_all_scope_assets(
    assets: list[dict[str, Any]],
    *,
    input_root: str,
    is_under_root: Callable[[str, str], bool],
) -> list[dict[str, Any]]:
    for asset in assets:
        source = str((asset or {}).get("source") or "").strip().lower()
        if source in ("input", "output", "custom"):
            asset["type"] = source
            continue
        filepath = str((asset or {}).get("filepath") or "")
        asset["type"] = "input" if filepath and is_under_root(input_root, filepath) else "output"
    return assets


async def _handle_all_scope_indexed(
    *,
    svc: dict[str, Any],
    query: str,
    limit: int,
    offset: int,
    sort_key: str,
    filters: dict[str, Any],
    include_total: bool,
    output_root: str,
    input_root: str,
    is_under_root: Callable[[str, str], bool],
    dedupe_result_assets_payload: Callable[[dict[str, Any]], dict[str, Any]],
) -> Result[dict[str, Any]]:
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
        return scoped
    data = scoped.data or {}
    assets = _annotate_all_scope_assets(data.get("assets") or [], input_root=input_root, is_under_root=is_under_root)
    return Result.Ok(
        _build_all_scope_payload(
            assets=assets,
            total=int(data.get("total") or 0),
            limit=limit,
            offset=offset,
            query=query,
            sort_key=sort_key,
            dedupe_result_assets_payload=dedupe_result_assets_payload,
        )
    )


async def _handle_all_scope_count_only(
    *,
    svc: dict[str, Any],
    query: str,
    offset: int,
    sort_key: str,
    filters: dict[str, Any],
    output_root: str,
    input_root: str,
    list_filesystem_assets: Callable[..., Any],
    dedupe_result_assets_payload: Callable[[dict[str, Any]], dict[str, Any]],
) -> Result[dict[str, Any]]:
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
        return out_count
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
        return in_count
    total = int((out_count.data or {}).get("total") or 0) + int((in_count.data or {}).get("total") or 0)
    return Result.Ok(
        _build_all_scope_payload(
            assets=[],
            total=total,
            limit=0,
            offset=offset,
            query=query,
            sort_key=sort_key,
            dedupe_result_assets_payload=dedupe_result_assets_payload,
        )
    )


def _asset_mtime(item: dict[str, Any]) -> int:
    try:
        return int((item or {}).get("mtime") or 0)
    except Exception:
        return 0


def _asset_name(item: dict[str, Any]) -> str:
    try:
        return str((item or {}).get("filename") or "").lower()
    except Exception:
        return ""


def _asset_path(item: dict[str, Any]) -> str:
    try:
        return str((item or {}).get("filepath") or "").lower()
    except Exception:
        return ""


def _asset_sort_key(item: dict[str, Any], sort_key: str) -> tuple[Any, str]:
    if sort_key in {"name_asc", "name_desc"}:
        return (_asset_name(item), _asset_path(item))
    return (_asset_mtime(item), _asset_path(item))


def _pick_output_item(out_item: dict[str, Any], in_item: dict[str, Any], sort_key: str) -> bool:
    out_key = _asset_sort_key(out_item, sort_key)
    in_key = _asset_sort_key(in_item, sort_key)
    if sort_key in {"name_desc", "mtime_desc"}:
        return out_key >= in_key
    return out_key <= in_key


async def _fill_output_buffer(
    state: dict[str, Any],
    *,
    svc: dict[str, Any],
    query: str,
    sort_key: str,
    filters: dict[str, Any],
    output_root: str,
) -> Result[None] | None:
    if state["out_total"] is not None and state["out_offset"] >= state["out_total"]:
        return None
    res = await svc["index"].search_scoped(
        query,
        roots=[output_root],
        limit=state["chunk"],
        offset=state["out_offset"],
        filters={**(filters or {}), "source": "output"},
        include_total=True,
        sort=sort_key,
    )
    if not res.ok:
        return res
    data = res.data or {}
    incoming_total = data.get("total")
    if incoming_total is not None:
        state["out_total"] = int(incoming_total or 0)
    items = data.get("assets") or []
    for asset in items:
        asset["type"] = "output"
    state["out_offset"] += len(items)
    state["out_buf"].extend(items)
    return None


async def _fill_input_buffer(
    state: dict[str, Any],
    *,
    input_root: str,
    query: str,
    sort_key: str,
    filters: dict[str, Any],
    list_filesystem_assets: Callable[..., Any],
) -> Result[None] | None:
    if state["in_total"] is not None and state["in_offset"] >= state["in_total"]:
        return None
    res = await list_filesystem_assets(
        Path(input_root),
        "",
        query,
        state["chunk"],
        state["in_offset"],
        asset_type="input",
        filters=filters or None,
        sort=sort_key,
    )
    if not res.ok:
        return res
    data = res.data or {}
    state["in_total"] = int(data.get("total") or 0)
    items = data.get("assets") or []
    state["in_offset"] += len(items)
    state["in_buf"].extend(items)
    return None


def _trim_merge_buffer(state: dict[str, Any], *, prefix: str, trim_start: int, trim_ratio: int) -> None:
    index_key = f"{prefix}_i"
    buffer_key = f"{prefix}_buf"
    current_index = state[index_key]
    current_buffer = state[buffer_key]
    if current_index <= trim_start:
        return
    if current_index < len(current_buffer) // trim_ratio:
        return
    state[buffer_key] = current_buffer[current_index:]
    state[index_key] = 0


async def _merge_all_scope_page(
    *,
    svc: dict[str, Any],
    query: str,
    limit: int,
    offset: int,
    sort_key: str,
    filters: dict[str, Any],
    output_root: str,
    input_root: str,
    list_filesystem_assets: Callable[..., Any],
    search_chunk_min: int,
    search_chunk_max: int,
    search_merge_trim_start: int,
    search_merge_trim_ratio: int,
) -> Result[dict[str, Any]]:
    state: dict[str, Any] = {
        "chunk": max(search_chunk_min, min(search_chunk_max, int(limit) * 2)),
        "out_offset": 0,
        "in_offset": 0,
        "out_buf": [],
        "in_buf": [],
        "out_i": 0,
        "in_i": 0,
        "out_total": None,
        "in_total": None,
    }

    err = await _fill_output_buffer(state, svc=svc, query=query, sort_key=sort_key, filters=filters, output_root=output_root)
    if err:
        return _propagate_all_scope_error(err, "SEARCH_FAILED", "Failed to fill output buffer")
    err = await _fill_input_buffer(
        state,
        input_root=input_root,
        query=query,
        sort_key=sort_key,
        filters=filters,
        list_filesystem_assets=list_filesystem_assets,
    )
    if err:
        return _propagate_all_scope_error(err, "SEARCH_FAILED", "Failed to fill input buffer")

    total = int(state["out_total"] or 0) + int(state["in_total"] or 0)
    target = offset + limit
    produced = 0
    page: list[dict[str, Any]] = []

    while produced < target:
        if state["out_i"] >= len(state["out_buf"]) and (state["out_total"] is None or state["out_offset"] < state["out_total"]):
            err = await _fill_output_buffer(
                state,
                svc=svc,
                query=query,
                sort_key=sort_key,
                filters=filters,
                output_root=output_root,
            )
            if err:
                return _propagate_all_scope_error(err, "SEARCH_FAILED", "Failed to refill output buffer")
        if state["in_i"] >= len(state["in_buf"]) and (state["in_total"] is None or state["in_offset"] < state["in_total"]):
            err = await _fill_input_buffer(
                state,
                input_root=input_root,
                query=query,
                sort_key=sort_key,
                filters=filters,
                list_filesystem_assets=list_filesystem_assets,
            )
            if err:
                return _propagate_all_scope_error(err, "SEARCH_FAILED", "Failed to refill input buffer")

        out_has = state["out_i"] < len(state["out_buf"])
        in_has = state["in_i"] < len(state["in_buf"])
        if not out_has and not in_has:
            break

        pick_out = out_has and (not in_has or _pick_output_item(state["out_buf"][state["out_i"]], state["in_buf"][state["in_i"]], sort_key))
        if pick_out:
            item = state["out_buf"][state["out_i"]]
            state["out_i"] += 1
        else:
            item = state["in_buf"][state["in_i"]]
            state["in_i"] += 1

        _trim_merge_buffer(state, prefix="out", trim_start=search_merge_trim_start, trim_ratio=search_merge_trim_ratio)
        _trim_merge_buffer(state, prefix="in", trim_start=search_merge_trim_start, trim_ratio=search_merge_trim_ratio)

        if produced >= offset:
            page.append(item)
            if len(page) >= limit:
                break
        produced += 1

    return Result.Ok({"assets": page, "total": total})


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

    input_indexed = await _is_input_indexed(svc, input_root)
    await _maybe_kickoff_input_scan(
        input_indexed=input_indexed,
        query=query,
        offset=offset,
        filters=filters,
        input_root=input_root,
        kickoff_background_scan=kickoff_background_scan,
    )

    if input_indexed:
        scoped_res = await _handle_all_scope_indexed(
            svc=svc,
            query=query,
            limit=limit,
            offset=offset,
            sort_key=sort_key,
            filters=filters,
            include_total=include_total,
            output_root=output_root,
            input_root=input_root,
            is_under_root=is_under_root,
            dedupe_result_assets_payload=dedupe_result_assets_payload,
        )
        return json_response(scoped_res)

    if not limit:
        count_res = await _handle_all_scope_count_only(
            svc=svc,
            query=query,
            offset=offset,
            sort_key=sort_key,
            filters=filters,
            output_root=output_root,
            input_root=input_root,
            list_filesystem_assets=list_filesystem_assets,
            dedupe_result_assets_payload=dedupe_result_assets_payload,
        )
        return json_response(count_res)

    merge_res = await _merge_all_scope_page(
        svc=svc,
        query=query,
        limit=limit,
        offset=offset,
        sort_key=sort_key,
        filters=filters,
        output_root=output_root,
        input_root=input_root,
        list_filesystem_assets=list_filesystem_assets,
        search_chunk_min=search_chunk_min,
        search_chunk_max=search_chunk_max,
        search_merge_trim_start=search_merge_trim_start,
        search_merge_trim_ratio=search_merge_trim_ratio,
    )
    if not merge_res.ok:
        return json_response(merge_res)
    merge_data = merge_res.data or {}
    payload = _build_all_scope_payload(
        assets=merge_data.get("assets") or [],
        total=int(merge_data.get("total") or 0),
        limit=limit,
        offset=offset,
        query=query,
        sort_key=sort_key,
        dedupe_result_assets_payload=dedupe_result_assets_payload,
    )
    return json_response(Result.Ok(payload))
