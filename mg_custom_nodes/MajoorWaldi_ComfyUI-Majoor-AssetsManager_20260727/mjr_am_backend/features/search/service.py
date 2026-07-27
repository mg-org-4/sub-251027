"""Normalize query params shared by list/search handlers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ...shared import Result


@dataclass(slots=True)
class SearchRequestContext:
    query: str
    raw_query: str
    inline_filters: dict[str, Any]
    limit: int
    offset: int
    sort_key: str
    filters: dict[str, Any]
    include_total: bool


def _parse_limit_offset(
    query_params: Mapping[str, str],
    *,
    default_limit: int,
    default_offset: int,
    max_limit: int,
    max_offset: int,
    clamp_limit: bool,
) -> Result[tuple[int, int]]:
    try:
        limit = int(query_params.get("limit", str(default_limit)))
        offset = int(query_params.get("offset", str(default_offset)))
    except ValueError:
        return Result.Err("INVALID_INPUT", "Invalid limit or offset")

    if clamp_limit:
        limit = max(0, min(max_limit, limit))
    elif limit < 0 or limit > max_limit:
        return Result.Err("INVALID_INPUT", f"limit must be between 0 and {max_limit}")

    if offset < 0 or offset > max_offset:
        return Result.Err("INVALID_INPUT", f"offset must be between 0 and {max_offset}")

    return Result.Ok((limit, offset))


def _parse_include_total(query_params: Mapping[str, str]) -> bool:
    return str(query_params.get("include_total", "1") or "").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _parse_search_query_text(
    query_params: Mapping[str, str],
    parse_inline_query_filters: Callable[[str], tuple[str, dict[str, Any]]],
) -> tuple[str, dict[str, Any]]:
    raw_query = str(query_params.get("q") or "").strip()
    parsed_query, inline_filters = parse_inline_query_filters(raw_query)
    return (parsed_query if parsed_query else "*"), inline_filters


def _parse_search_pagination(
    query_params: Mapping[str, str],
    *,
    default_limit: int,
    default_offset: int,
    max_limit: int,
    max_offset: int,
    clamp_limit: bool,
) -> Result[tuple[int, int]]:
    return _parse_limit_offset(
        query_params,
        default_limit=default_limit,
        default_offset=default_offset,
        max_limit=max_limit,
        max_offset=max_offset,
        clamp_limit=clamp_limit,
    )


def _parse_search_filters(
    query_params: Mapping[str, str],
    inline_filters: dict[str, Any],
    parse_request_filters: Callable[[Mapping[str, str], dict[str, Any]], Result[dict[str, Any]]],
) -> Result[dict[str, Any]]:
    return parse_request_filters(query_params, inline_filters)


def _build_search_request_context(
    query_params: Mapping[str, str],
    *,
    default_limit: int,
    default_offset: int,
    max_limit: int,
    max_offset: int,
    parse_inline_query_filters: Callable[[str], tuple[str, dict[str, Any]]],
    parse_request_filters: Callable[[Mapping[str, str], dict[str, Any]], Result[dict[str, Any]]],
    normalize_sort_key: Callable[[Any], str],
    clamp_limit: bool,
) -> Result[SearchRequestContext]:
    raw_query = str(query_params.get("q") or "").strip()
    query, inline_filters = _parse_search_query_text(query_params, parse_inline_query_filters)

    pagination_res = _parse_search_pagination(
        query_params,
        default_limit=default_limit,
        default_offset=default_offset,
        max_limit=max_limit,
        max_offset=max_offset,
        clamp_limit=clamp_limit,
    )
    if not pagination_res.ok:
        return Result.Err(
            pagination_res.code or "INVALID_INPUT",
            pagination_res.error or "Invalid pagination",
        )
    limit, offset = pagination_res.data  # type: ignore[misc]

    filters_res = _parse_search_filters(query_params, inline_filters, parse_request_filters)
    if not filters_res.ok:
        return Result.Err(filters_res.code or "INVALID_INPUT", filters_res.error or "Invalid filters")

    filters = filters_res.data or {}
    include_total = _parse_include_total(query_params)
    sort_key = normalize_sort_key(query_params.get("sort"))
    return Result.Ok(
        SearchRequestContext(
            query=query,
            raw_query=raw_query,
            inline_filters=inline_filters or {},
            limit=limit,
            offset=offset,
            sort_key=sort_key,
            filters=filters,
            include_total=include_total,
        )
    )


def parse_search_request(
    query_params: Mapping[str, str],
    *,
    default_limit: int,
    default_offset: int,
    max_limit: int,
    max_offset: int,
    parse_inline_query_filters: Callable[[str], tuple[str, dict[str, Any]]],
    parse_request_filters: Callable[[Mapping[str, str], dict[str, Any]], Result[dict[str, Any]]],
    normalize_sort_key: Callable[[Any], str],
    clamp_limit: bool,
) -> Result[SearchRequestContext]:
    return _build_search_request_context(
        query_params,
        default_limit=default_limit,
        default_offset=default_offset,
        max_limit=max_limit,
        max_offset=max_offset,
        parse_inline_query_filters=parse_inline_query_filters,
        parse_request_filters=parse_request_filters,
        normalize_sort_key=normalize_sort_key,
        clamp_limit=clamp_limit,
    )
