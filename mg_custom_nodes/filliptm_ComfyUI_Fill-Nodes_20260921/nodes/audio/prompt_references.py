import json
from pathlib import Path

import folder_paths


def reference_document(value, section_count):
    if not value:
        return {"version": 1, "assets": {}, "sections": []}
    document = json.loads(value) if isinstance(value, str) else value
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError("Unsupported prompt reference document version.")
    assets = document.get("assets", {})
    sections = document.get("sections", [])
    if not isinstance(assets, dict) or len(assets) > 2048:
        raise ValueError("Reference library must contain at most 2048 assets.")
    if not isinstance(sections, list) or len(sections) != section_count:
        raise ValueError("Reference sections no longer match the timeline. Reopen the sequencer.")
    seen = set()
    for section in sections:
        if not isinstance(section, dict):
            raise ValueError("Invalid reference section.")
        section_id = section.get("id")
        if not isinstance(section_id, str) or not section_id or section_id in seen:
            raise ValueError("Reference sections require unique, stable IDs.")
        seen.add(section_id)
        mode = section.get("mode", "defaults")
        ids = section.get("asset_ids", [])
        if mode not in {"defaults", "custom", "none"} or not isinstance(ids, list):
            raise ValueError("Invalid section reference selection.")
        if any(not isinstance(asset_id, str) for asset_id in ids) or len(ids) != len(set(ids)) or any(asset_id not in assets for asset_id in ids):
            raise ValueError("Section references contain missing or duplicate assets.")
        if mode != "custom" and ids:
            raise ValueError("Only custom reference sections may select assets.")
    for asset_id, asset in assets.items():
        if not isinstance(asset_id, str) or not isinstance(asset, dict):
            raise ValueError("Invalid reference asset.")
        if asset.get("kind") not in {"image", "video", "audio"}:
            raise ValueError(f"Reference {asset_id} has an unsupported media type.")
        reference_path(asset, must_exist=False)
    return document


def reference_file_fingerprint(value):
    if not value:
        return ()
    document = json.loads(value) if isinstance(value, str) else value
    document = reference_document(document, len(document.get("sections", [])))
    selected = dict.fromkeys(asset_id for section in document["sections"] for asset_id in section.get("asset_ids", []))
    files = []
    for asset_id in selected:
        path = reference_path(document["assets"][asset_id])
        stat = path.stat()
        files.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(files)


def reference_path(asset, must_exist=True):
    roots = {"input": folder_paths.get_input_directory(), "output": folder_paths.get_output_directory()}
    storage = asset.get("type", "input")
    if storage not in roots:
        raise ValueError("References must be stored in ComfyUI input or output.")
    filename = asset.get("filename", "")
    subfolder = asset.get("subfolder", "")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename or any(character in filename for character in "/\\:"):
        raise ValueError("Invalid reference filename.")
    if not isinstance(subfolder, str):
        raise ValueError("Invalid reference subfolder.")
    root = Path(roots[storage]).resolve()
    path = (root / subfolder / filename).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Reference path must stay inside its ComfyUI media directory.")
    if must_exist and not path.is_file():
        raise ValueError(f"Missing reference file: {filename}. Copy the workflow's reference media into ComfyUI.")
    return path


def apply_reference_sections(sections, document):
    for section, reference in zip(sections, document["sections"]):
        section["section_id"] = reference["id"]
        section["references"] = {"mode": reference.get("mode", "defaults"), "asset_ids": list(reference.get("asset_ids", []))}
    return sections
