"""Batch-delete-oriented asset route action extracted from ``route_actions_crud``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from aiohttp import web

from mjr_am_backend.shared import Result
from mjr_am_backend.shared import sanitize_error_message as _safe_error_message


async def handle_delete_assets(
    request: web.Request,
    *,
    prepare_asset_ids_context: Callable[[web.Request], Awaitable[Result[Any]]],
    normalize_path: Callable[[str], Path | None],
    is_path_allowed: Callable[[Path], bool],
    is_path_allowed_custom: Callable[[Path], bool],
    is_resolved_path_allowed: Callable[[Path], bool],
    delete_file_safe: Callable[[Path], Result[Any]],
    audit_asset_write: Callable[..., Awaitable[None]],
    json_response: Callable[[Any], web.Response],
    require_services: Callable[[], Awaitable[tuple[dict[str, Any], Result[Any] | None]]],
    resolve_security_prefs: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    require_operation_enabled: Callable[..., Result[Any]],
    require_write_access: Callable[[web.Request], Result[Any]],
    check_rate_limit: Callable[..., tuple[bool, int | None]],
    read_json: Callable[[web.Request], Awaitable[Any]],
    csrf_error: Callable[[web.Request], str | None],
    safe_error_message: Callable[[Exception, str], str] = _safe_error_message,
    logger: Any = None,
) -> web.Response:
    ids_ctx_res = await prepare_asset_ids_context(
        request,
        operation="assets_delete",
        rate_limit_endpoint="assets_delete",
        max_requests=10,
        window_seconds=60,
        require_services=require_services,
        resolve_security_prefs=resolve_security_prefs,
        require_operation_enabled=require_operation_enabled,
        require_write_access=require_write_access,
        check_rate_limit=check_rate_limit,
        read_json=read_json,
        csrf_error=csrf_error,
    )
    if not ids_ctx_res.ok:
        return json_response(ids_ctx_res)
    ids_ctx = ids_ctx_res.data
    svc = ids_ctx.services if ids_ctx else {}
    validated_ids = ids_ctx.asset_ids if ids_ctx else []

    try:
        res = await svc["db"].aquery_in(
            "SELECT id, filepath FROM assets WHERE {IN_CLAUSE}",
            "id",
            validated_ids,
        )
        if not res.ok:
            return json_response(res)

        found_assets = res.data or []
        found_ids = {row["id"] for row in found_assets if isinstance(row, dict) and "id" in row}
        missing_ids = set(validated_ids) - found_ids
        if missing_ids:
            return json_response(Result.Err("NOT_FOUND", f"Assets not found: {sorted(missing_ids)}"))
    except Exception as exc:
        return json_response(Result.Err("DB_ERROR", safe_error_message(exc, "Failed to validate assets")))

    validated_assets = []
    for asset_row in found_assets:
        asset_id = asset_row.get("id")
        raw_path = asset_row.get("filepath")
        if asset_id is None:
            return json_response(Result.Err("DB_ERROR", "Missing asset id in DB row"))
        if not raw_path or not isinstance(raw_path, str):
            return json_response(Result.Err("INVALID_INPUT", f"Invalid path for asset ID {asset_id}"))

        candidate = normalize_path(raw_path)
        if not candidate:
            return json_response(Result.Err("INVALID_INPUT", f"Invalid asset path for ID {asset_id}"))

        allowed_path = is_path_allowed(candidate) or is_path_allowed_custom(candidate)
        if not allowed_path:
            return json_response(Result.Err("FORBIDDEN", f"Path for asset ID {asset_id} is not within allowed roots"))

        try:
            resolved = candidate.resolve(strict=True)
        except Exception:
            resolved = None
        if resolved is not None and not is_resolved_path_allowed(resolved):
            return json_response(Result.Err("FORBIDDEN", f"Path for asset ID {asset_id} is not within allowed roots"))

        validated_assets.append({"id": int(asset_id), "filepath": str(raw_path), "resolved": resolved})

    validated_assets.sort(key=lambda item: item["id"])
    file_deletion_errors = []
    assets_ready_for_db_delete = []

    for asset_info in validated_assets:
        resolved = asset_info["resolved"]
        if resolved and resolved.exists() and resolved.is_file():
            try:
                del_res = delete_file_safe(resolved)
                if not del_res.ok:
                    file_deletion_errors.append(
                        {"asset_id": asset_info["id"], "error": str(del_res.error or "delete failed")}
                    )
                    continue
            except Exception as exc:
                file_deletion_errors.append(
                    {
                        "asset_id": asset_info["id"],
                        "error": safe_error_message(exc, "File deletion failed"),
                    }
                )
                continue
        assets_ready_for_db_delete.append(asset_info)

    async def _db_delete_many(selected_assets: list[dict[str, Any]]) -> Result:
        if not selected_assets:
            return Result.Ok({"deleted_ids": [], "db_errors": [], "deleted": 0})

        deleted_ids: list[int] = []
        db_errors: list[dict[str, Any]] = []
        unique_ids = sorted({int(asset_info["id"]) for asset_info in selected_assets})
        unique_paths = sorted(
            {str(asset_info["filepath"]) for asset_info in selected_assets if asset_info.get("filepath")}
        )

        def _chunks(values: list[Any], size: int) -> list[list[Any]]:
            return [values[i:i + size] for i in range(0, len(values), size)] if size > 0 else [values]

        async def _delete_where_in(column: str, table: str, values: list[Any]) -> Result:
            allowed = {
                "assets": frozenset({"id", "filepath"}),
                "scan_journal": frozenset({"id", "filepath"}),
                "metadata_cache": frozenset({"id", "filepath"}),
            }
            if table not in allowed or column not in allowed[table]:
                return Result.Err("INVALID_INPUT", f"Disallowed delete target: {table!r}.{column!r}")
            if not values:
                return Result.Ok(True)

            for chunk in _chunks(values, 900):
                placeholders = ",".join(["?"] * len(chunk))
                query = f"DELETE FROM {table} WHERE {column} IN ({placeholders})"
                delete_res = await svc["db"].aexecute(query, tuple(chunk))
                if not delete_res.ok:
                    return Result.Err("DB_ERROR", str(delete_res.error or f"Failed deleting from {table}"))
            return Result.Ok(True)

        try:
            async with svc["db"].atransaction(mode="immediate"):
                assets_del = await _delete_where_in("id", "assets", unique_ids)
                if not assets_del.ok:
                    raise RuntimeError(str(assets_del.error or "assets batch delete failed"))
                sj_del = await _delete_where_in("filepath", "scan_journal", unique_paths)
                if not sj_del.ok:
                    raise RuntimeError(str(sj_del.error or "scan_journal batch delete failed"))
                mc_del = await _delete_where_in("filepath", "metadata_cache", unique_paths)
                if not mc_del.ok:
                    raise RuntimeError(str(mc_del.error or "metadata_cache batch delete failed"))
            deleted_ids.extend(unique_ids)
            return Result.Ok({"deleted_ids": deleted_ids, "db_errors": db_errors, "deleted": len(deleted_ids)})
        except Exception as exc:
            if logger:
                logger.warning("Batch delete failed, falling back to per-asset deletion: %s", exc)

        for asset_info in selected_assets:
            asset_id = int(asset_info["id"])
            filepath = str(asset_info["filepath"])
            try:
                del_res = await svc["db"].aexecute("DELETE FROM assets WHERE id = ?", (asset_id,))
                if not del_res.ok:
                    db_errors.append({"asset_id": asset_id, "error": str(del_res.error or "DB delete failed")})
                    continue
                await svc["db"].aexecute("DELETE FROM scan_journal WHERE filepath = ?", (filepath,))
                await svc["db"].aexecute("DELETE FROM metadata_cache WHERE filepath = ?", (filepath,))
                deleted_ids.append(asset_id)
            except Exception as exc:
                db_errors.append({"asset_id": asset_id, "error": safe_error_message(exc, "DB delete failed")})

        return Result.Ok({"deleted_ids": deleted_ids, "db_errors": db_errors, "deleted": len(deleted_ids)})

    try:
        db_res = await _db_delete_many(assets_ready_for_db_delete)
        if not db_res.ok:
            return json_response(db_res)

        payload = db_res.data or {}
        deleted_ids = [int(item) for item in (payload.get("deleted_ids") or [])]
        db_errors = payload.get("db_errors") or []
        failed_ids = [
            int(item.get("asset_id"))
            for item in file_deletion_errors
            if isinstance(item, dict) and item.get("asset_id") is not None
        ]
        failed_ids += [
            int(item.get("asset_id"))
            for item in db_errors
            if isinstance(item, dict) and item.get("asset_id") is not None
        ]

        if file_deletion_errors or db_errors:
            result = Result.Ok(
                {
                    "deleted": len(deleted_ids),
                    "deleted_ids": deleted_ids,
                    "failed_ids": sorted(set(failed_ids)),
                },
                partial=True,
                errors=[*file_deletion_errors, *db_errors],
                attempted=len(validated_assets),
                deleted=len(deleted_ids),
                failed=len(set(failed_ids)),
            )
            await audit_asset_write(
                request,
                svc,
                "assets_delete",
                f"assets:{','.join(str(value) for value in validated_ids[:25])}",
                result,
                attempted=len(validated_assets),
                deleted=len(deleted_ids),
                failed=len(set(failed_ids)),
            )
            return json_response(result)

        result = Result.Ok({"deleted": len(deleted_ids), "deleted_ids": deleted_ids})
        await audit_asset_write(
            request,
            svc,
            "assets_delete",
            f"assets:{','.join(str(value) for value in validated_ids[:25])}",
            result,
            attempted=len(validated_assets),
            deleted=len(deleted_ids),
            failed=0,
        )
        return json_response(result)
    except Exception as exc:
        if logger:
            logger.error("Database deletion failed: %s", exc)
        result = Result.Err("DB_ERROR", safe_error_message(exc, "Failed to delete asset records"))
        await audit_asset_write(
            request,
            svc,
            "assets_delete",
            f"assets:{','.join(str(value) for value in validated_ids[:25])}",
            result,
            attempted=len(validated_assets),
        )
        return json_response(result)
