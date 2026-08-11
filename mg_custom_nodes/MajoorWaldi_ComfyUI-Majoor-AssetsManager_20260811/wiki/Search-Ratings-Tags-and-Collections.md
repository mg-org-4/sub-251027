# Search, Ratings, Tags, and Collections

## Search

The extension supports indexed search across filenames and metadata, with ranking and filtering designed for large asset libraries.

Search workflows include:
- filename lookup
- metadata lookup
- filter combinations for type, date, size, rating, workflow, and resolution

Useful search behaviors highlighted by the docs:

- SQLite FTS5 and BM25 ranking are used for fast result ordering
- quoted phrases can improve precision
- inline attribute tokens such as `rating:5` or `ext:webp` can apply filters directly from the search box
- workflow-aware and metadata-aware search is a major part of the system, not an afterthought

## Filtering Strategy

The best results usually come from combining a broad text query with one or two structural filters:

1. start with a concept or model term
2. narrow by file kind or extension
3. apply rating or date filters
4. sort by relevance, date, or rating depending on the task

## Ratings

Ratings let you quickly separate keepers from drafts. The project also supports synchronization paths for environments that can write ratings metadata back to files.

When metadata writeback is available, ratings are more portable and can survive file movement better than purely in-app annotations.

## Tags

Tags provide a flexible organization layer on top of raw folders. They are useful for style families, subject matter, projects, and review states.

The docs also support hierarchical styles such as `subject:portrait` or `style:cinematic`, which is useful once a library gets too large for flat keywords.

## Collections

Collections let you save curated groups of assets across scopes. This is useful when assembling references, comparing outputs, or building reusable review sets.

Collections are one of the most practical ways to move from ad-hoc browsing to repeatable review workflows.

## Recommended Organization Pattern

For most users, a strong pattern is:

- ratings for fast quality triage
- tags for semantic grouping
- collections for explicit working sets and review batches

Those three layers solve different problems and work better together than separately.

## Canonical Docs

- [Search and Filtering](https://github.com/MajoorWaldi/ComfyUI-Majoor-AssetsManager/blob/main/docs/SEARCH_FILTERING.md)
- [Ratings, Tags, and Collections](https://github.com/MajoorWaldi/ComfyUI-Majoor-AssetsManager/blob/main/docs/RATINGS_TAGS_COLLECTIONS.md)
