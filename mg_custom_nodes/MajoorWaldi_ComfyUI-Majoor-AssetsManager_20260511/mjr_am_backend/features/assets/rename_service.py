"""Rename-oriented asset business logic."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ...shared import Result
from .models import AssetRenameTarget


def _propagate_rename_error(result: Result[Any], default_code: str, default_error: str) -> Result[dict[str, Any]]:
    return Result.Err(result.code or default_code, str(result.error or default_error))


def _compute_rename_path(current_resolved: Path, new_name: str) -> Path:
    return current_resolved.parent / new_name


def _check_rename_conflict(current_resolved: Path, new_path: Path, new_name: str) -> Result[None]:
    if not new_path.exists():
        return Result.Ok(None)

    same_file = False
    try:
        same_file = bool(new_path.samefile(current_resolved))
    except Exception:
        same_file = False
    same_path_ignoring_case = str(new_path).lower() == str(current_resolved).lower()
    if same_file and same_path_ignoring_case:
        return Result.Ok(None)
    return Result.Err("CONFLICT", f"File '{new_name}' already exists")


def _rename_file(current_resolved: Path, new_path: Path, *, safe_error_message: Callable[[Exception, str], str]) -> Result[None]:
    try:
        current_resolved.rename(new_path)
    except Exception as exc:
        return Result.Err("RENAME_FAILED", safe_error_message(exc, "Failed to rename file"))
    return Result.Ok(None)


async def _rollback_physical_rename(
    current_resolved: Path,
    new_path: Path,
    *,
    matched_asset_id: int | None,
    logger: Any = None,
) -> None:
    try:
        if new_path.exists() and not current_resolved.exists():
            new_path.rename(current_resolved)
    except Exception as rollback_exc:
        if logger:
            logger.error("Failed to rollback rename for asset %s: %s", matched_asset_id, rollback_exc)


async def _stat_renamed_file(
    current_resolved: Path,
    new_path: Path,
    *,
    matched_asset_id: int | None,
    safe_error_message: Callable[[Exception, str], str],
    logger: Any = None,
) -> Result[int]:
    try:
        return Result.Ok(int(new_path.stat().st_mtime))
    except FileNotFoundError:
        await _rollback_physical_rename(current_resolved, new_path, matched_asset_id=matched_asset_id, logger=logger)
        return Result.Err("NOT_FOUND", "Renamed file does not exist")
    except Exception as exc:
        await _rollback_physical_rename(current_resolved, new_path, matched_asset_id=matched_asset_id, logger=logger)
        return Result.Err("FS_ERROR", safe_error_message(exc, "Failed to stat renamed file"))


async def _update_rename_records(
    services: dict[str, Any],
    *,
    matched_asset_id: int | None,
    current_filepath_where: str,
    current_filepath_params: tuple[Any, ...],
    new_name: str,
    new_path: Path,
    mtime: int,
) -> Result[None]:
    try:
        async with services["db"].atransaction(mode="immediate"):
            defer_fk = await services["db"].aexecute("PRAGMA defer_foreign_keys = ON")
            if not defer_fk.ok:
                raise RuntimeError(defer_fk.error or "Failed to defer foreign key checks")

            if matched_asset_id is not None:
                update_res = await services["db"].aexecute(
                    "UPDATE assets SET filename = ?, filepath = ?, mtime = ? WHERE id = ?",
                    (new_name, str(new_path), mtime, matched_asset_id),
                )
                if not update_res.ok:
                    raise RuntimeError(update_res.error or "Failed to update assets filepath")
                try:
                    updated_rows = int(update_res.data or 0)
                except Exception:
                    updated_rows = 0
                if updated_rows <= 0:
                    raise RuntimeError("Asset row not found for rename")
            else:
                up2 = await services["db"].aexecute(
                    f"UPDATE assets SET filename = ?, filepath = ?, mtime = ? WHERE {current_filepath_where}",
                    (new_name, str(new_path), mtime, *current_filepath_params),
                )
                if not up2.ok:
                    raise RuntimeError(up2.error or "Failed to update assets filepath")

            sj_res = await services["db"].aexecute(
                f"UPDATE scan_journal SET filepath = ? WHERE {current_filepath_where}",
                (str(new_path), *current_filepath_params),
            )
            if not sj_res.ok:
                raise RuntimeError(sj_res.error or "Failed to update scan_journal filepath")

            mc_res = await services["db"].aexecute(
                f"UPDATE metadata_cache SET filepath = ? WHERE {current_filepath_where}",
                (str(new_path), *current_filepath_params),
            )
            if not mc_res.ok:
                raise RuntimeError(mc_res.error or "Failed to update metadata_cache filepath")
    except Exception as exc:
        return Result.Err("DB_ERROR", str(exc))
    return Result.Ok(None)


async def _reindex_renamed_asset(
    services: dict[str, Any],
    *,
    current_source: str,
    current_root_id: str,
    new_path: Path,
    infer_source_and_root_id_from_path: Callable[..., Awaitable[tuple[str, str]]],
    is_within_root: Callable[[Path, Path], bool],
    get_runtime_output_root: Callable[[], str],
    get_input_directory: Callable[[], str],
    list_custom_roots: Callable[[], Any],
    to_thread_timeout_s: int | float,
    logger: Any = None,
) -> None:
    try:
        source = current_source
        root_id = current_root_id or None
        if not source:
            inferred_source, inferred_root_id = await infer_source_and_root_id_from_path(
                new_path,
                get_runtime_output_root(),
                is_within_root=is_within_root,
                get_runtime_output_root_fn=get_runtime_output_root,
                get_input_directory=get_input_directory,
                list_custom_roots_fn=list_custom_roots,
                logger=logger,
            )
            source = inferred_source
            root_id = inferred_root_id
        index_svc = services.get("index") if isinstance(services, dict) else None
        if index_svc and hasattr(index_svc, "index_paths"):
            await asyncio.wait_for(
                index_svc.index_paths([new_path], str(new_path.parent), False, source or "output", root_id),
                timeout=to_thread_timeout_s,
            )
    except Exception as exc:
        if logger:
            logger.debug("Post-rename targeted reindex skipped: %s", exc)


async def _load_fresh_asset(services: dict[str, Any], matched_asset_id: int | None) -> dict[str, Any] | None:
    if matched_asset_id is None:
        return None
    try:
        fr = await services["db"].aquery(
            """
            SELECT a.id, a.filename, a.subfolder, a.filepath, a.source, a.root_id, a.kind, a.ext,
                   a.size, a.mtime, a.width, a.height, a.duration,
                   COALESCE(m.rating, 0) AS rating, COALESCE(m.tags, '[]') AS tags
            FROM assets a
            LEFT JOIN asset_metadata m ON m.asset_id = a.id
            WHERE a.id = ?
            LIMIT 1
            """,
            (matched_asset_id,),
        )
    except Exception:
        return None
    if fr.ok and fr.data:
        return fr.data[0]
    return None


async def rename_asset_and_sync(
    *,
    services: dict[str, Any],
    target: AssetRenameTarget,
    infer_source_and_root_id_from_path: Callable[..., Awaitable[tuple[str, str]]],
    is_within_root: Callable[[Path, Path], bool],
    get_runtime_output_root: Callable[[], str],
    get_input_directory: Callable[[], str],
    list_custom_roots: Callable[[], Any],
    to_thread_timeout_s: int | float,
    safe_error_message: Callable[[Exception, str], str],
    logger: Any = None,
) -> Result[dict[str, Any]]:
    matched_asset_id = target.matched_asset_id
    current_resolved = target.current_resolved
    current_filename = target.current_filename
    current_source = target.current_source
    current_root_id = target.current_root_id
    current_filepath_where = target.filepath_where
    current_filepath_params = target.filepath_params
    new_name = target.new_name

    new_path = _compute_rename_path(current_resolved, new_name)
    conflict_res = _check_rename_conflict(current_resolved, new_path, new_name)
    if not conflict_res.ok:
        return _propagate_rename_error(conflict_res, "CONFLICT", "File rename conflict")

    rename_res = _rename_file(current_resolved, new_path, safe_error_message=safe_error_message)
    if not rename_res.ok:
        return _propagate_rename_error(rename_res, "RENAME_FAILED", "Failed to rename file")

    stat_res = await _stat_renamed_file(
        current_resolved,
        new_path,
        matched_asset_id=matched_asset_id,
        safe_error_message=safe_error_message,
        logger=logger,
    )
    if not stat_res.ok:
        return _propagate_rename_error(stat_res, "FS_ERROR", "Failed to stat renamed file")
    mtime = int(stat_res.data or 0)

    update_res = await _update_rename_records(
        services,
        matched_asset_id=matched_asset_id,
        current_filepath_where=current_filepath_where,
        current_filepath_params=current_filepath_params,
        new_name=new_name,
        new_path=new_path,
        mtime=mtime,
    )
    if not update_res.ok:
        await _rollback_physical_rename(current_resolved, new_path, matched_asset_id=matched_asset_id, logger=logger)
        return Result.Err("DB_ERROR", safe_error_message(Exception(str(update_res.error)), "Failed to update asset record"))

    await _reindex_renamed_asset(
        services,
        current_source=current_source,
        current_root_id=current_root_id,
        new_path=new_path,
        infer_source_and_root_id_from_path=infer_source_and_root_id_from_path,
        is_within_root=is_within_root,
        get_runtime_output_root=get_runtime_output_root,
        get_input_directory=get_input_directory,
        list_custom_roots=list_custom_roots,
        to_thread_timeout_s=to_thread_timeout_s,
        logger=logger,
    )

    fresh_asset = await _load_fresh_asset(services, matched_asset_id)

    return Result.Ok(
        {
            "renamed": 1,
            "old_name": current_filename,
            "new_name": new_name,
            "old_path": str(current_resolved),
            "new_path": str(new_path),
            "asset": fresh_asset,
        }
    )


__all__ = ["rename_asset_and_sync"]
