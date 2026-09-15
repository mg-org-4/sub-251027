from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT.parent
    / "ComfyUI-LTXVideo"
    / "example_workflows"
    / "2.3"
    / "LTX-2.3_ICLoRA_Outpaint_Two_Stage_Distilled.json"
)
OUTPUT = ROOT / "examples" / "08_ltx23_reframe_iclora_outpaint_two_stage.json"
INSTALLED_OUTPUT = (
    ROOT.parents[1]
    / "user"
    / "default"
    / "workflows"
    / "LTX 2.3 Reframe IC-LoRA Outpaint - Verified.json"
)

REMOVED_NODE_IDS = {5168, 5363, 5371, 5393}
REMOVED_LINK_IDS = {14383, 14395, 14445, 14446}


def node_by_id(workflow: dict[str, Any], node_id: int) -> dict[str, Any]:
    return next(node for node in workflow["nodes"] if node["id"] == node_id)


def build_reframe_node() -> dict[str, Any]:
    return {
        "id": 5368,
        "type": "LTXReframeLayout",
        "pos": [-4225.854240441995, 3547.938320763212],
        "size": [600, 760],
        "flags": {},
        "order": 12,
        "mode": 0,
        "inputs": [],
        "outputs": [
            {"name": "padded_frames", "type": "IMAGE", "links": [14364]},
            {"name": "outpaint_mask", "type": "MASK", "links": [14369]},
            {"name": "source_mask", "type": "MASK", "links": []},
            {"name": "source_frames", "type": "IMAGE", "links": []},
            {"name": "audio", "type": "AUDIO", "links": [14422]},
            {"name": "fps", "type": "FLOAT", "links": [13889, 13968]},
            {"name": "source_video", "type": "VIDEO", "links": []},
            {"name": "target_width", "type": "INT", "links": []},
            {"name": "target_height", "type": "INT", "links": []},
            {"name": "placement_json", "type": "STRING", "links": []},
        ],
        "title": "1. LTX Reframe Layout · Source & Framing",
        "properties": {"Node name for S&R": "LTXReframeLayout"},
        "widgets_values": [
            "LTX-2_00180_.mp4",
            "9:16",
            576,
            1024,
            0.5,
            0.5,
            0.7,
            32,
        ],
        "color": "#1f4d52",
        "bgcolor": "#16383c",
    }


def build_mask_batch_node(
    node_id: int,
    *,
    pos: list[float],
    mask_link: int,
    size_link: int,
    output_link: int,
    title: str,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "MaskExpandBatch+",
        "pos": pos,
        "size": [270, 106],
        "flags": {},
        "order": 63 if node_id == 5400 else 64,
        "mode": 0,
        "inputs": [
            {"name": "mask", "type": "MASK", "link": mask_link},
            {
                "name": "size",
                "type": "INT",
                "widget": {"name": "size"},
                "link": size_link,
            },
        ],
        "outputs": [{"name": "MASK", "type": "MASK", "links": [output_link]}],
        "title": title,
        "properties": {
            "cnr_id": "comfyui_essentials",
            "Node name for S&R": "MaskExpandBatch+",
        },
        "widgets_values": [121, "expand"],
        "color": "#5b4636",
        "bgcolor": "#3c3027",
    }


def clean_graph_references(workflow: dict[str, Any]) -> None:
    valid_node_ids = {node["id"] for node in workflow["nodes"]}
    workflow["links"] = [
        link
        for link in workflow["links"]
        if link[0] not in REMOVED_LINK_IDS
        and link[1] in valid_node_ids
        and link[3] in valid_node_ids
    ]
    valid_link_ids = {link[0] for link in workflow["links"]}

    for node in workflow["nodes"]:
        for item in node.get("inputs", []):
            if item.get("link") not in valid_link_ids:
                item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = [
                link_id for link_id in item.get("links") or [] if link_id in valid_link_ids
            ]

    extra = workflow.get("extra", {})
    reroutes = extra.get("reroutes", [])
    for reroute in reroutes:
        reroute["linkIds"] = [
            link_id for link_id in reroute.get("linkIds", []) if link_id in valid_link_ids
        ]
    extra["reroutes"] = [reroute for reroute in reroutes if reroute.get("linkIds")]
    extra["linkExtensions"] = [
        extension
        for extension in extra.get("linkExtensions", [])
        if extension.get("id") in valid_link_ids
    ]

    for group in workflow.get("groups", []):
        group["nodes"] = [node_id for node_id in group.get("nodes", []) if node_id in valid_node_ids]

    workflow["last_node_id"] = max(valid_node_ids)
    workflow["last_link_id"] = max(valid_link_ids)


def validate_graph(workflow: dict[str, Any]) -> None:
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    if len(nodes) != len(workflow["nodes"]):
        raise ValueError("Duplicate node IDs remain in the generated workflow.")
    if len(links) != len(workflow["links"]):
        raise ValueError("Duplicate link IDs remain in the generated workflow.")

    for link_id, link in links.items():
        if link[1] not in nodes or link[3] not in nodes:
            raise ValueError(f"Link {link_id} has a missing endpoint: {link}")
        source_output = nodes[link[1]]["outputs"][link[2]]
        destination_input = nodes[link[3]]["inputs"][link[4]]
        if link_id not in (source_output.get("links") or []):
            raise ValueError(f"Link {link_id} is absent from its source output socket.")
        if destination_input.get("link") != link_id:
            raise ValueError(f"Link {link_id} is absent from its destination input socket.")

    reframe = nodes.get(5368)
    if reframe is None or reframe["type"] != "LTXReframeLayout":
        raise ValueError("The generated workflow is missing LTXReframeLayout node 5368.")

    expected = {
        14364: (5368, 0, 5360, 0, "IMAGE"),
        14369: (5368, 1, 5365, 0, "MASK"),
        13889: (5368, 5, 1241, 2, "FLOAT"),
        13968: (5368, 5, 5227, 2, "FLOAT"),
        14422: (5368, 4, 5382, 0, "AUDIO"),
        14376: (5365, 0, 5400, 0, "MASK"),
        14377: (5364, 0, 5401, 0, "MASK"),
        14447: (5054, 2, 5400, 1, "INT"),
        14448: (5054, 2, 5401, 1, "INT"),
        14449: (5400, 0, 5226, 2, "MASK"),
        14450: (5401, 0, 5266, 2, "MASK"),
    }
    for link_id, contract in expected.items():
        actual = tuple(links[link_id][1:])
        if actual != contract:
            raise ValueError(f"Link {link_id} is {actual}, expected {contract}.")


def build() -> dict[str, Any]:
    workflow = json.loads(SOURCE.read_text(encoding="utf-8"))
    workflow["id"] = str(
        uuid.uuid5(uuid.NAMESPACE_URL, "ltx-2.3-reframe-iclora-outpaint-two-stage")
    )
    workflow["nodes"] = [
        node for node in workflow["nodes"] if node["id"] not in REMOVED_NODE_IDS | {5368}
    ]
    workflow["nodes"].append(build_reframe_node())
    workflow["nodes"].extend(
        [
            build_mask_batch_node(
                5400,
                pos=[1050.0, 3261.0],
                mask_link=14376,
                size_link=14447,
                output_link=14449,
                title="Full-Res Mask · Match Video Frame Count",
            ),
            build_mask_batch_node(
                5401,
                pos=[-660.0, 3432.0],
                mask_link=14377,
                size_link=14448,
                output_link=14450,
                title="Half-Res Mask · Match Video Frame Count",
            ),
        ]
    )

    links = {link[0]: link for link in workflow["links"]}
    links[14364] = [14364, 5368, 0, 5360, 0, "IMAGE"]
    links[14369] = [14369, 5368, 1, 5365, 0, "MASK"]
    links[13889] = [13889, 5368, 5, 1241, 2, "FLOAT"]
    links[13968] = [13968, 5368, 5, 5227, 2, "FLOAT"]
    links[14422] = [14422, 5368, 4, 5382, 0, "AUDIO"]
    links[14376] = [14376, 5365, 0, 5400, 0, "MASK"]
    links[14377] = [14377, 5364, 0, 5401, 0, "MASK"]
    links[14447] = [14447, 5054, 2, 5400, 1, "INT"]
    links[14448] = [14448, 5054, 2, 5401, 1, "INT"]
    links[14449] = [14449, 5400, 0, 5226, 2, "MASK"]
    links[14450] = [14450, 5401, 0, 5266, 2, "MASK"]
    workflow["links"] = list(links.values())

    node_by_id(workflow, 5226)["inputs"][2]["link"] = 14449
    node_by_id(workflow, 5266)["inputs"][2]["link"] = 14450
    node_by_id(workflow, 5054)["outputs"][2]["links"].extend([14447, 14448])

    node_by_id(workflow, 5023)["widgets_values"][0] = (
        "gemma_3_12B_it_fp4_mixed.safetensors"
    )
    node_by_id(workflow, 4922)["widgets_values"][0] = (
        "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    )
    node_by_id(workflow, 5011)["widgets_values"][0] = (
        "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"
    )
    node_by_id(workflow, 2483)["widgets_values"][0] = (
        "A cinematic live-action scene continuing naturally beyond the source frame, "
        "with a coherent environment, realistic lighting, consistent subjects and camera "
        "motion, seamless edges, and stable fine detail."
    )
    node_by_id(workflow, 5228)["widgets_values"][0] = "video/LTX_Reframe_Outpaint"
    node_by_id(workflow, 5356)["widgets_values"][0] = (
        "### Reframe Setup\n\nUse **LTX Reframe Layout** to choose the final canvas and place "
        "the source. `source_scale < 1` zooms out for outpainting; `source_scale > 1` "
        "zooms in and crops. The example starts at **576 × 1024**, 121 frames, with "
        "audio, so it is suitable for a first end-to-end test."
    )

    for group in workflow.get("groups", []):
        if group["id"] == 164:
            group["title"] = "1. Reframe Source Video"
            group["bounding"] = [-4235.854240441995, 3507.938320763212, 620, 800]
        elif group["id"] == 168:
            group["title"] = "2. Prepare Two-Stage Outpaint Conditions"
        elif group["id"] == 178:
            group["title"] = "Stage 2 Full-Resolution Mask"
        elif group["id"] == 179:
            group["title"] = "Stage 1 Half-Resolution Mask"

    workflow.setdefault("extra", {})["ltx_reframe_workflow"] = {
        "schema": "ltx_reframe_iclora_outpaint/v1",
        "based_on": SOURCE.name,
        "source_node": "LTXReframeLayout",
        "default_source": "LTX-2_00180_.mp4",
        "default_target": [576, 1024],
        "required_lora": "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
        "mask_batch_safety": "MaskExpandBatch+ before both Laplacian blends",
    }

    clean_graph_references(workflow)
    validate_graph(workflow)
    return workflow


def main() -> None:
    workflow = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
    INSTALLED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT, INSTALLED_OUTPUT)
    print(f"Wrote {OUTPUT}")
    print(f"Installed {INSTALLED_OUTPUT}")
    print(f"Nodes: {len(workflow['nodes'])}; links: {len(workflow['links'])}")


if __name__ == "__main__":
    main()
