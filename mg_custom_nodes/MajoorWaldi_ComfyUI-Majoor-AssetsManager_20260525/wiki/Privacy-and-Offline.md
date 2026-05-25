# Privacy and Offline

## Short Answer

The current project documentation states that AI-assisted features are intended to run locally on the user's machine.

That means:
- prompts and images are not uploaded to a hosted cloud inference service for semantic search, similarity, captions, or prompt alignment
- internet access is mainly required for first-time model download
- once models are cached locally, AI features can work offline

## What Uses The Network

The main network-facing part of the AI workflow is model download from HuggingFace on first use.

After the models are cached locally, the normal AI workflows can operate offline.

## Token Clarification

The docs now clearly separate two token concepts:

- HuggingFace token: optional, helps with model download access and rate limits
- Majoor API token: protects remote write access to the local Majoor backend

These are different concerns. The Majoor API token is not described as a hosted AI inference token.

## Why This Page Matters

This distinction is easy to miss if a user only reads the README or sees the word token in settings. The dedicated privacy/offline guide is one of the most useful additions to the documentation set and deserves a first-class wiki page.

## Canonical Docs

- [Privacy and Offline Guide](https://github.com/MajoorWaldi/ComfyUI-Majoor-AssetsManager/blob/main/docs/PRIVACY_OFFLINE.md)
- [AI Features Guide](https://github.com/MajoorWaldi/ComfyUI-Majoor-AssetsManager/blob/main/docs/AI_FEATURES.md)
- [Security Environment Variables](https://github.com/MajoorWaldi/ComfyUI-Majoor-AssetsManager/blob/main/docs/SECURITY_ENV_VARS.md)
- [Settings Configuration](https://github.com/MajoorWaldi/ComfyUI-Majoor-AssetsManager/blob/main/docs/SETTINGS_CONFIGURATION.md)
