STORYBOARD_SCHEMA = {
    "type": "array",
    "description": "Image generation requests, queued automatically with ComfyUI credits when the Writer finishes. Empty unless the user requests images. Prefer 2x2 for sections up to six seconds; 3x3 for longer sections. Describe chronological panel beats. Across all requests repeat the same precise cast identity, clothing, palette, drawing technique and world design, honoring selected moodboard roles. Change only action, framing and explicitly requested story changes; do not redesign characters between sections.",
    "items": {"type": "object", "properties": {
        "index": {"type": "integer", "minimum": 0},
        "grid": {"type": "integer", "enum": [2, 3]},
        "prompt": {"type": "string", "minLength": 1},
    }, "required": ["index", "grid", "prompt"], "additionalProperties": False},
}

REFERENCE_ASSIGNMENT_SCHEMA = {
    "type": "array", "maxItems": 8,
    "description": "Apply these reference assignments directly to scoped sections. Use only asset IDs supplied in the reference library context.",
    "items": {"type": "object", "properties": {
        "index": {"type": "integer", "minimum": 0},
        "mode": {"type": "string", "enum": ["defaults", "custom", "none"]},
        "asset_ids": {"type": "array", "items": {"type": "string"}},
    }, "required": ["index", "mode", "asset_ids"], "additionalProperties": False},
}


def normalize_asset_ids(values):
    if not isinstance(values, list) or len(values) > 2048 or any(not isinstance(value, str) or not 1 <= len(value) <= 128 for value in values):
        raise ValueError("Invalid reference library IDs.")
    return list(dict.fromkeys(values))


def normalize_reference_assignments(values, allowed_indices, asset_ids):
    if not isinstance(values, list) or len(values) > 8:
        raise ValueError("Request at most eight reference assignments.")
    result = []
    seen = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Invalid reference assignment.")
        index, mode = value.get("index"), value.get("mode")
        ids = normalize_asset_ids(value.get("asset_ids", []))
        if type(index) is not int or index not in allowed_indices or index in seen:
            raise ValueError("Reference assignment targets an unavailable or duplicate section.")
        if mode not in {"defaults", "custom", "none"} or (mode != "custom" and ids) or any(asset_id not in asset_ids for asset_id in ids):
            raise ValueError("Reference assignment uses unavailable assets or an invalid mode.")
        seen.add(index)
        result.append({"index": index, "mode": mode, "asset_ids": ids})
    return result


def normalize_storyboard_actions(values, allowed_indices):
    if not isinstance(values, list):
        raise ValueError("Storyboard requests must be a list.")
    result = []
    seen = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Invalid storyboard request.")
        index, grid, prompt = value.get("index"), value.get("grid"), value.get("prompt")
        if type(index) is not int or index not in allowed_indices or index in seen:
            raise ValueError("Storyboard request targets an unavailable or duplicate section.")
        if type(grid) is not int or grid not in (2, 3) or not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 64000:
            raise ValueError("Storyboard requests need a 2x2 or 3x3 grid and a bounded prompt.")
        seen.add(index)
        result.append({"index": index, "grid": grid, "prompt": prompt.strip()})
    return result
