from __future__ import annotations

import json
import importlib.util
import os
import sys
from pathlib import Path
from uuid import UUID
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "examples" / "workflows" / "MiniMax_H3_Continuum_V39.json"
PACKAGE_NAME = "ComfyUI_H3_Continuum_Join"
if PACKAGE_NAME not in sys.modules:
    package_root = Path(os.environ.get("H3_CONTINUUM_PACKAGE_ROOT", ROOT))
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME, package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(package_root)]
    sys.modules[PACKAGE_NAME] = module

def _graph():
    graph = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    assert len(nodes) == len(graph["nodes"])
    assert len(links) == len(graph["links"])
    for link in links.values():
        link_id, source_id, source_slot, target_id, target_slot, link_type = link
        assert link_id in nodes[source_id]["outputs"][source_slot]["links"]
        assert nodes[target_id]["inputs"][target_slot]["link"] == link_id
        assert nodes[source_id]["outputs"][source_slot]["type"] in ("*", link_type)
        assert nodes[target_id]["inputs"][target_slot]["type"] in ("*", link_type)
    return graph, nodes, links


def test_official_v39_is_a_separate_complete_workflow():
    graph, nodes, links = _graph()
    assert UUID(graph["id"]).version == 4
    assert len(nodes) == 32
    assert len(links) == 39
    assert graph["last_node_id"] >= max(nodes)
    assert graph["last_link_id"] >= max(links)
    assert nodes[312]["type"] == "H3ContinuumSamplerV39"
    assert nodes[348]["type"] == "H3ContinuumReferenceImagesV39"
    assert nodes[344]["type"] == "H3DecodeCacheHelper"
    assert nodes[306]["type"] == "H3ContinuumAssembleSeamV35"
    assert nodes[228]["type"] == "CreateVideo"
    assert nodes[191]["type"] == "SaveVideo"
    assert nodes[191]["widgets_values"][0] == "video/MiniMax_H3_Continuum_V39"
    assert nodes[150]["widgets_values"][0] is False  # Spectrum OFF
    assert nodes[348]["widgets_values"][:10] == ["Per chunk", *(["off"] * 9)]
    assert [item["name"] for item in nodes[348]["inputs"]] == [
        f"reference_image_{slot}" for slot in range(1, 10)
    ]
    assert all(item["link"] is not None for item in nodes[348]["inputs"][:6])
    assert all(item["link"] is None for item in nodes[348]["inputs"][6:])
    for slot in range(1, 7):
        link = links[nodes[348]["inputs"][slot - 1]["link"]]
        assert nodes[link[1]]["type"] == "H3EasyLoadImage"
        assert nodes[link[1]]["mode"] == 4  # No unknown source image is silently enabled.
    assert any(link[1] == 348 and link[3] == 312 for link in links.values())
    assert all(not item["name"].startswith("reference_image_") for item in nodes[312]["inputs"])


def test_official_v39_initial_settings_and_prompt_are_consistent():
    graph, nodes, links = _graph()
    sampler = nodes[312]
    named = sampler["widgets_values_named"]
    assert sampler["widgets_values"] == list(named.values())
    assert UUID(named["project_id"]).version == 4
    assert named["prompt_mode"] == "Timeline"
    assert named["chunks"] == 2
    assert named["chunk_seconds"] == 10
    assert named["size_source"] == "First Image"
    assert named["preset"] == "Draft — 0.30 MP"
    assert (named["width"], named["height"]) == (480, 640)
    assert named["video_reference_mode"] == "Repeat Reference"
    assert named["generation_mode"] == "Full Run"
    assert nodes[317]["mode"] == 0  # The user-selected First Image loader is enabled.
    assert nodes[188]["type"] == "PrimitiveStringMultiline"
    assert nodes[188]["widgets_values"] == [""]
    assert nodes[188]["widgets_values_named"]["value"] == ""
    prompt_input = next(item for item in sampler["inputs"] if item["name"] == "sequence_prompt")
    assert links[prompt_input["link"]][1] == 188


def test_official_v39_zip_contains_exact_json_bytes():
    payload = WORKFLOW.read_bytes()
    with ZipFile(WORKFLOW.with_suffix(".zip")) as archive:
        assert archive.namelist() == [WORKFLOW.name]
        assert archive.read(WORKFLOW.name) == payload


def test_template_gallery_uses_official_workflows_only():
    gallery = ROOT / "examples"
    assert {path.name for path in gallery.glob("*.json")} == {
        "MiniMax_H3_Continuum_V39.json",
        "MiniMax_H3_Continuum_V38X2.json",
    }
    assert (gallery / WORKFLOW.name).read_bytes() == WORKFLOW.read_bytes()
    legacy = "MiniMax_H3_Continuum_V38X2.json"
    assert (gallery / legacy).read_bytes() == (WORKFLOW.parent / legacy).read_bytes()
