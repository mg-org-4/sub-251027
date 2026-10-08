"""Export a completed ComfyUI prompt as an immutable advertisement API fixture.

The exported prompt graph remains directly submit-able to ComfyUI.  A sibling
manifest records its runtime provenance and canonical SHA-256 without adding
non-node keys to the graph itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import urllib.request
from typing import Any


KNOWN_GOOD_V6_SHA256 = (
    "594f6aef94531f87b74659e1e8004cccdff00716d0cfe263cee6bf2149545c83"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument(
        "--known-good-ui",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1]
        / "examples"
        / "15_minimax_h3_ref2va_music_video_v6.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ui_bytes = args.known_good_ui.read_bytes()
    ui_hash = _sha256_bytes(ui_bytes)
    if ui_hash != KNOWN_GOOD_V6_SHA256:
        raise SystemExit(
            "Known-good V6 hash mismatch; refusing to derive an advertisement fixture. "
            f"Expected {KNOWN_GOOD_V6_SHA256}, received {ui_hash}."
        )

    output = args.output.resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if not args.force and (output.exists() or manifest_path.exists()):
        raise SystemExit("Output already exists; pass --force only for an intentional refresh.")

    history = _read_json(
        f"{args.base_url.rstrip('/')}/history/{args.prompt_id}"
    )
    entry = history.get(args.prompt_id)
    if not isinstance(entry, dict):
        raise SystemExit(f"Prompt {args.prompt_id} was not found in ComfyUI history.")
    prompt_record = entry.get("prompt")
    if not isinstance(prompt_record, list) or len(prompt_record) < 3:
        raise SystemExit("History entry does not contain an API prompt graph.")
    graph = prompt_record[2]
    if not isinstance(graph, dict) or not graph:
        raise SystemExit("History API prompt graph is empty or malformed.")

    status = entry.get("status", {})
    if not isinstance(status, dict) or not bool(status.get("completed")):
        raise SystemExit("Only a completed prompt may become a runtime fixture.")

    graph_hash = _sha256_bytes(_canonical_json(graph).encode("utf-8"))
    class_counts: dict[str, int] = {}
    for node in graph.values():
        if not isinstance(node, dict):
            raise SystemExit("History prompt contains a malformed node record.")
        class_type = str(node.get("class_type", ""))
        if not class_type:
            raise SystemExit("History prompt contains a node without class_type.")
        class_counts[class_type] = class_counts.get(class_type, 0) + 1

    manifest = {
        "schema": "diffusiongemma.advertisement_runtime_fixture",
        "version": 1,
        "prompt_id": args.prompt_id,
        "completed": True,
        "api_graph_sha256": graph_hash,
        "api_node_count": len(graph),
        "class_counts": dict(sorted(class_counts.items())),
        "known_good_ui_path": args.known_good_ui.resolve().as_posix(),
        "known_good_ui_sha256": ui_hash,
        "source": "local_comfyui_history",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
