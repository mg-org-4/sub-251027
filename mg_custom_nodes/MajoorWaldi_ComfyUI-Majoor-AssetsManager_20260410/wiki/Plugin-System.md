# Plugin System

## Overview

The repository now documents a plugin system for custom metadata extractors. This is one of the most significant architecture topics added beyond the original end-user feature set.

The goal is to let third parties add extractor logic without modifying the core codebase.

## What The Plugin System Adds

- plugin discovery from configured directories
- validation before loading
- priority-based extractor selection
- runtime registry and lifecycle management
- example plugins for custom node ecosystems

The documented architecture also separates roles cleanly between loader, registry, manager, and validation logic. That makes the plugin track easier to reason about than a single monolithic plugin hook.

## Why It Matters

Without plugins, metadata extraction has to stay hardcoded in the core project. With the plugin track, new formats and custom-node metadata can be supported with less coupling and better isolation.

This is especially relevant for ComfyUI ecosystems where custom nodes often emit metadata that core extractors cannot interpret cleanly.

## Security Model

The docs describe validation and sandbox-oriented constraints for plugin code. Examples of blocked categories include:

- dynamic code execution
- unrestricted subprocess usage
- direct network access
- unsafe deserialization

This is important because extractor plugins run close to user assets and metadata.

## Authoring And Examples

The repository already includes:
- plugin design documentation
- implementation summary
- quick reference for authors
- example plugins under the `plugins/` folder

Documented examples include custom extractor patterns for ecosystem-specific metadata such as WanVideo and rgthree-style workflows.

## Typical Lifecycle

The documented lifecycle is:

1. discovery of plugin files
2. validation against security rules
3. loading and registration
4. extraction at runtime
5. cleanup and reload support

That is a strong sign that the plugin track is intended as a real extension mechanism, not just a loose future idea.

## Canonical Docs

- [Plugin System Design](https://github.com/MajoorWaldi/ComfyUI-Majoor-AssetsManager/blob/main/docs/PLUGIN_SYSTEM_DESIGN.md)
- [Plugin Implementation Summary](https://github.com/MajoorWaldi/ComfyUI-Majoor-AssetsManager/blob/main/docs/PLUGIN_SYSTEM_IMPLEMENTATION_SUMMARY.md)
- [Plugin Quick Reference](https://github.com/MajoorWaldi/ComfyUI-Majoor-AssetsManager/blob/main/docs/PLUGIN_QUICK_REFERENCE.md)
- [Plugin Examples README](https://github.com/MajoorWaldi/ComfyUI-Majoor-AssetsManager/blob/main/plugins/README.md)
