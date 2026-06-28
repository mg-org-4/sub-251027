# ADR 0009 - Tags Normalization

**Status**: Accepted
**Date**: 2026-05-22
**Deciders**: Backend team

## Context

Tags were historically stored as JSON text in `asset_metadata.tags`, with
`asset_metadata.tags_text` as a denormalized search companion. That made tag
filters, faceting, autocomplete, deduplication, and case-insensitive matching
depend on JSON string parsing or ad hoc SQL expressions.

The project needed a storage model that supports case-insensitive tag
uniqueness, fast asset/tag filtering, reliable FTS indexing, and a staged
migration path for existing databases.

## Decision

User tags are stored in normalized relational tables:

- `tags(id, name COLLATE NOCASE)`
- `asset_tags(asset_id, tag_id)`

The normalized tables are the source of truth. `asset_metadata.tags` and
`asset_metadata.tags_text` are removed by migration v19.

## Migration Timeline

- v17 `normalize_tags`: create `tags` and `asset_tags`, then backfill from the
  legacy JSON column when it exists.
- v18 `fts_from_normalized_tags`: rebuild `asset_metadata_fts` triggers so FTS
  tag text comes from `asset_tags` joined to `tags`.
- v19 `drop_legacy_tag_columns`: verify legacy JSON rows have normalized links,
  rebuild `asset_metadata` without `tags` and `tags_text`, then reindex FTS.

## Write Strategy

During the transition, writers dual-wrote legacy JSON and normalized rows.
After Readers v2 completed, Stop-Write changed application writers to update
only `asset_tags` / `tags`.

Legacy sync helpers remain only for migration and recovery scenarios where an
old database may still contain JSON tags before v19 runs.

## FTS Guarantees

From v18 onward, `asset_metadata_fts.tags` and
`asset_metadata_fts.tags_text` are populated from normalized tags. The
`tags_text` FTS column is kept as an FTS compatibility column, not as an
`asset_metadata` column.

## Consequences

- Tag filters and autocomplete operate on indexed relational data.
- Case-insensitive deduplication is enforced by storage.
- Dropping legacy columns removes ambiguity about the source of truth.
- Tests that simulate pre-v19 databases must explicitly add legacy columns.

## Verification

- `tests/features/test_drop_legacy_tag_columns.py`
- `tests/features/test_tags_normalized_readers.py`
- `tests/features/test_tags_dual_write.py`
- `tests/features/test_fts_from_normalized.py`
