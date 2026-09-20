from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ComfyUI_H3_Continuum_Join import nodes as continuum_nodes


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "examples/workflows/MiniMax_H3_Continuum_V38X2.json"


def test_builtin_helper_preserves_public_node_contract():
    helper = continuum_nodes.NODE_CLASS_MAPPINGS["H3DecodeCacheHelper"]
    assert continuum_nodes.NODE_DISPLAY_NAME_MAPPINGS["H3DecodeCacheHelper"] == "Decode Cache Helper"
    assert helper.RETURN_TYPES == ("IMAGE", "AUDIO", "STRING")
    assert helper.RETURN_NAMES == ("images", "audio", "report")
    assert helper.INPUT_IS_LIST is True
    assert helper.OUTPUT_IS_LIST == (True, True, False)
    assert helper.CATEGORY == "MiniMax H3/Continuum/Helpers"
    assert list(helper.INPUT_TYPES()["required"]) == [
        "cache_mode", "ram_budget_mb", "disk_budget_gb", "reset_token"
    ]
    required = helper.INPUT_TYPES()["required"]
    assert required["cache_mode"][1]["default"] == "Auto"
    assert required["ram_budget_mb"][1]["default"] == 256
    assert required["disk_budget_gb"][1]["default"] == 8
    assert required["reset_token"][1]["default"] == 0
    assert required["reset_token"][0] == "INT"


def test_supplied_workflow_is_preserved_and_routes_through_builtin_helper():
    payload = WORKFLOW.read_bytes()
    assert hashlib.sha256(payload).hexdigest().upper() == (
        "6C445D62979632BC1A801B1C28DBD639C23B51CC156EBAE18837BAAA757E592B"
    )
    workflow = json.loads(payload)
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert nodes[312]["type"] == "H3ContinuumSamplerV38"
    assert nodes[344]["type"] == "H3DecodeCacheHelper"
    assert nodes[306]["type"] == "H3ContinuumAssembleSeamV35"
    assert nodes[344]["widgets_values"] == ["Auto", 256, 8, 0]
    links = {(link[1], link[2], link[3], link[4], link[5]) for link in workflow["links"]}
    assert (312, 0, 344, 0, "LATENT") in links
    assert (312, 1, 344, 2, "LATENT") in links
    assert (344, 0, 306, 0, "IMAGE") in links
    assert (344, 1, 306, 1, "AUDIO") in links
    assert (312, 2, 306, 2, "H3_CONTINUUM_ASSEMBLY_PLAN") in links
    assert not any(node["type"] in {"VAEDecode", "VAEDecodeAudio"} for node in workflow["nodes"])


def test_sampler_and_finalize_remain_separate_and_core_direct_workflow_remains():
    assert "H3ContinuumSamplerV38" in continuum_nodes.NODE_CLASS_MAPPINGS
    assert "H3ContinuumAssembleSeamV35" in continuum_nodes.NODE_CLASS_MAPPINGS
    core = json.loads(
        (ROOT / "examples/workflows/MiniMax_H3_Continuum_V38x.json").read_text(encoding="utf-8")
    )
    types = {node["type"] for node in core["nodes"]}
    assert {"VAEDecode", "VAEDecodeAudio", "H3ContinuumAssembleSeamV35"} <= types
