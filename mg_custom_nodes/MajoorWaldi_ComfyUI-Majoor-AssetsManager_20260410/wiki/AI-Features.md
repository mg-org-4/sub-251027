# AI Features

## Overview

Majoor Assets Manager includes optional AI-assisted discovery and enrichment workflows.

Examples include:
- semantic search from natural language queries
- image similarity search
- automatic tag suggestion
- caption enrichment
- prompt alignment and quality scoring

These features depend on optional models and packages and are not required for the base extension.

## Privacy Model For AI Features

The current documentation clarifies an important point: AI inference is intended to run locally on the user's machine.

In practice, that means:
- prompts and images are not sent to a hosted cloud inference service for semantic search, similarity, captions, or prompt alignment
- internet access is mainly needed for the initial model download from HuggingFace
- once models are cached locally, AI features can work offline

This is the main reason the project now has a dedicated privacy and offline guide.

## When To Use AI Features

AI features are most useful when:
- the library is too large for manual browsing alone
- filenames are inconsistent or unhelpful
- you want thematic grouping across many unrelated folders
- you want better retrieval from prompt or visual intent

## Token Clarification

Two token concepts appear in the docs and should not be confused:

- HuggingFace token: optional, used for model download reliability and rate limits
- Majoor API token: used to secure remote write access to the local backend

Neither token is described as a hosted inference key for uploading user prompts or assets.

## Canonical Docs

- [AI Features Guide](https://github.com/MajoorWaldi/ComfyUI-Majoor-AssetsManager/blob/main/docs/AI_FEATURES.md)
- [Privacy and Offline Guide](https://github.com/MajoorWaldi/ComfyUI-Majoor-AssetsManager/blob/main/docs/PRIVACY_OFFLINE.md)
- [Security Environment Variables](https://github.com/MajoorWaldi/ComfyUI-Majoor-AssetsManager/blob/main/docs/SECURITY_ENV_VARS.md)
