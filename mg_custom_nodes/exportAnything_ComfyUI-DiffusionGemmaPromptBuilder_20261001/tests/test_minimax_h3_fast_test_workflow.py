from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATH = REPO_ROOT / "tools" / "migrate_minimax_h3_fast_test.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_fast_test", MIGRATION_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load migration module: {MIGRATION_PATH}")
MIGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIGRATION)

LIVE_CAPTURE = Path(
    r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_LIVE_UNSAVED_CAPTURE_20260822_0057.json"
)


def _input(name: str, type_name: str) -> dict:
    return {"name": name, "type": type_name, "link": None}


def _output(name: str, type_name: str) -> dict:
    return {"name": name, "type": type_name, "links": []}


def _node(
    node_id: int,
    node_type: str,
    *,
    inputs: list[tuple[str, str]] = (),
    outputs: list[tuple[str, str]] = (),
    widgets: list | None = None,
    pos: tuple[float, float] = (0.0, 0.0),
    title: str | None = None,
) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "pos": list(pos),
        "size": [300.0, 100.0],
        "flags": {},
        "order": node_id,
        "mode": 0,
        "inputs": [_input(name, type_name) for name, type_name in inputs],
        "outputs": [_output(name, type_name) for name, type_name in outputs],
        "title": title,
        "properties": {"Node name for S&R": node_type},
        "widgets_values": list(widgets or []),
    }


def _main_connect(
    workflow: dict,
    link_id: int,
    origin: dict,
    output_name: str,
    target: dict,
    input_name: str,
    link_type: str,
) -> None:
    output_slot = next(
        index for index, item in enumerate(origin["outputs"]) if item["name"] == output_name
    )
    input_slot = next(
        index for index, item in enumerate(target["inputs"]) if item["name"] == input_name
    )
    workflow["links"].append(
        [link_id, origin["id"], output_slot, target["id"], input_slot, link_type]
    )
    origin["outputs"][output_slot]["links"].append(link_id)
    target["inputs"][input_slot]["link"] = link_id


def _sub_connect(
    subgraph: dict,
    link_id: int,
    origin: dict,
    output_name: str,
    target: dict,
    input_name: str,
    link_type: str,
) -> None:
    output_slot = next(
        index for index, item in enumerate(origin["outputs"]) if item["name"] == output_name
    )
    input_slot = next(
        index for index, item in enumerate(target["inputs"]) if item["name"] == input_name
    )
    subgraph["links"].append(
        {
            "id": link_id,
            "origin_id": origin["id"],
            "origin_slot": output_slot,
            "target_id": target["id"],
            "target_slot": input_slot,
            "type": link_type,
        }
    )
    origin["outputs"][output_slot]["links"].append(link_id)
    target["inputs"][input_slot]["link"] = link_id


def _fixture() -> dict:
    duration = _node(
        178,
        "PrimitiveFloat",
        outputs=[("FLOAT", "FLOAT")],
        widgets=[15.0],
        title="Master duration",
    )
    splitter = _node(
        187,
        "DiffusionGemmaJSONSplitter",
        widgets=["", 1.0, 32, "9:16 (Portrait Widescreen)"],
        title="Resolution",
    )
    loader = _node(
        661,
        "UNETLoader",
        outputs=[("MODEL", "MODEL")],
        widgets=["minimax_h3_ref2va_int8_convrot.safetensors", "default"],
    )
    lora = _node(
        713,
        "LoraLoaderModelOnly",
        inputs=[("model", "MODEL")],
        outputs=[("MODEL", "MODEL")],
        widgets=["minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors", 1.0],
    )
    sage = _node(
        715,
        "PathchSageAttentionKJ",
        inputs=[("model", "MODEL")],
        outputs=[("MODEL", "MODEL")],
        widgets=["auto", False],
    )
    cache = _node(
        712,
        "EasyCache",
        inputs=[("model", "MODEL")],
        outputs=[("MODEL", "MODEL")],
        widgets=[0.2, 0.15, 0.95, True],
    )
    compile_node = _node(
        714,
        "TorchCompileModel",
        inputs=[("model", "MODEL")],
        outputs=[("MODEL", "MODEL")],
        widgets=["inductor"],
    )
    outer = _node(
        710,
        "h3-fast-test-subgraph",
        inputs=[("model", "MODEL")],
        outputs=[("IMAGE", "IMAGE")],
    )
    project = _node(
        676,
        "DiffusionGemmaProjectMasterContract",
        inputs=[("production_duration_seconds", "FLOAT")],
        outputs=[("project_manifest_json", "STRING")],
        widgets=[15.0, "9:16", 0.0, 15.0, "MiniMax H3 Ref2VA", "{}", 15.0],
    )
    target = _node(
        673,
        "DiffusionGemmaMiniMaxH3TargetProfile",
        widgets=["ref2va", 0, "auto_scene_audio", "", "auto", 12, "off", 2, "", "auto", "", 0],
    )
    planner = _node(
        677,
        "DiffusionGemmaAudioAwareMultiShotPlanner",
        inputs=[("project_manifest_json", "STRING")],
        outputs=[("plan_json", "STRING")],
        widgets=[0.0, 15.0, 12.0, 5.0, 15.0],
    )
    nodes = [
        duration,
        splitter,
        loader,
        lora,
        sage,
        cache,
        compile_node,
        outer,
        project,
        target,
        planner,
    ]
    workflow = {
        "id": "fixture-workflow",
        "revision": 12,
        "last_node_id": 715,
        "last_link_id": 1405,
        "version": 0.4,
        "nodes": nodes,
        "links": [],
        "groups": [],
        "definitions": {"subgraphs": []},
        "extra": {"fixture": {"native_shot_blocks": 4, "active_lanes": 1}},
    }
    _main_connect(workflow, 1398, loader, "MODEL", lora, "model", "MODEL")
    _main_connect(workflow, 1404, lora, "MODEL", sage, "model", "MODEL")
    _main_connect(workflow, 1405, sage, "MODEL", cache, "model", "MODEL")
    _main_connect(workflow, 1402, cache, "MODEL", compile_node, "model", "MODEL")
    _main_connect(workflow, 1403, compile_node, "MODEL", outer, "model", "MODEL")
    _main_connect(
        workflow, 1224, duration, "FLOAT", project, "production_duration_seconds", "FLOAT"
    )
    _main_connect(
        workflow,
        1244,
        project,
        "project_manifest_json",
        planner,
        "project_manifest_json",
        "STRING",
    )

    sampler_select = _node(
        657,
        "KSamplerSelect",
        outputs=[("SAMPLER", "SAMPLER")],
        widgets=["res_multistep"],
    )
    scheduler = _node(
        658,
        "BasicScheduler",
        inputs=[("model", "MODEL")],
        outputs=[("SIGMAS", "SIGMAS")],
        widgets=["simple", 8, 1.0],
    )
    h3_ids = [667, 684, 692, 700]
    guider_ids = [660, 686, 694, 702]
    sampler_ids = [659, 687, 695, 703]
    sub_nodes = [sampler_select, scheduler]
    h3_nodes = []
    guiders = []
    samplers = []
    for lane, (h3_id, guider_id, sampler_id) in enumerate(
        zip(h3_ids, guider_ids, sampler_ids)
    ):
        y = float(lane * 970)
        h3 = _node(
            h3_id,
            "MiniMaxH3ReferenceToVideo",
            outputs=[("positive", "CONDITIONING"), ("LATENT", "LATENT")],
            pos=(3748.0, y + 204.0),
        )
        guider = _node(
            guider_id,
            "BasicGuider",
            inputs=[("model", "MODEL"), ("conditioning", "CONDITIONING")],
            outputs=[("GUIDER", "GUIDER")],
            pos=(4198.0, y + 354.0),
        )
        sampler = _node(
            sampler_id,
            "SamplerCustomAdvanced",
            inputs=[
                ("noise", "NOISE"),
                ("guider", "GUIDER"),
                ("sampler", "SAMPLER"),
                ("sigmas", "SIGMAS"),
                ("latent_image", "LATENT"),
            ],
            outputs=[("output", "LATENT")],
            pos=(4618.0, y + 274.0),
        )
        h3_nodes.append(h3)
        guiders.append(guider)
        samplers.append(sampler)
        sub_nodes.extend([h3, guider, sampler])
    subgraph = {
        "id": "h3-fast-test-subgraph",
        "name": "New Subgraph",
        "version": 1,
        "revision": 12,
        "state": {"lastNodeId": 715, "lastLinkId": 1405},
        "inputNode": {"id": -10},
        "outputNode": {"id": -20},
        "inputs": [],
        "outputs": [],
        "nodes": sub_nodes,
        "links": [],
        "groups": [],
        "widgets": [],
        "config": {},
        "extra": {},
    }
    link_id = 1259
    for h3, guider, sampler in zip(h3_nodes, guiders, samplers):
        _sub_connect(
            subgraph, link_id, h3, "positive", guider, "conditioning", "CONDITIONING"
        )
        link_id += 1
        _sub_connect(
            subgraph, link_id, h3, "LATENT", sampler, "latent_image", "LATENT"
        )
        link_id += 1
    workflow["definitions"]["subgraphs"].append(subgraph)
    return workflow


def _node_by_id(workflow: dict, node_id: int) -> dict:
    for node in workflow["nodes"]:
        if int(node["id"]) == int(node_id):
            return node
    for subgraph in workflow["definitions"]["subgraphs"]:
        for node in subgraph["nodes"]:
            if int(node["id"]) == int(node_id):
                return node
    raise KeyError(node_id)


class FastTestMigrationUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _fixture()
        self.source_before = copy.deepcopy(self.source)
        self.migrated = MIGRATION.migrate_workflow(
            self.source, source_file_sha256="a" * 64
        )

    def test_source_is_unchanged_and_migration_is_idempotent(self) -> None:
        self.assertEqual(self.source, self.source_before)
        MIGRATION.validate_workflow(
            self.migrated, require_benchmark_defaults=True
        )
        self.assertEqual(
            MIGRATION.migrate_workflow(self.migrated), self.migrated
        )

    def test_official_fast_asset_and_sampling_contract_is_active(self) -> None:
        marker = self.migrated["extra"][MIGRATION.MIGRATION_SCHEMA]
        loader = _node_by_id(self.migrated, marker["nodes"]["loader"])
        lora = _node_by_id(self.migrated, marker["nodes"]["lora"])
        sage = _node_by_id(self.migrated, marker["nodes"]["sage"])
        sigma = _node_by_id(self.migrated, marker["nodes"]["sigma_shift"])
        self.assertEqual(loader["widgets_values"][0], MIGRATION.FAST_MODEL)
        self.assertEqual(lora["widgets_values"], [MIGRATION.FAST_LORA, 1.0])
        self.assertEqual(sage["widgets_values"], ["auto", False])
        self.assertEqual(
            sigma["widgets_values"],
            [MIGRATION.FAST_VIDEO_SHIFT, MIGRATION.FAST_AUDIO_SHIFT],
        )
        self.assertEqual(
            marker["assets"]["turbo_lora"]["expected_sha256"],
            MIGRATION.FAST_LORA_SHA256,
        )
        subgraph = self.migrated["definitions"]["subgraphs"][0]
        sampler = next(
            node for node in subgraph["nodes"] if node["type"] == "KSamplerSelect"
        )
        scheduler = next(
            node for node in subgraph["nodes"] if node["type"] == "BasicScheduler"
        )
        self.assertEqual(sampler["widgets_values"], ["euler"])
        self.assertEqual(scheduler["widgets_values"], ["simple", 4, 1.0])

    def test_easycache_and_compile_are_disconnected_bypasses(self) -> None:
        marker = self.migrated["extra"][MIGRATION.MIGRATION_SCHEMA]
        disabled_ids = {
            marker["nodes"]["disabled_easycache"],
            marker["nodes"]["disabled_torch_compile"],
        }
        self.assertTrue(
            all(_node_by_id(self.migrated, node_id)["mode"] == 4 for node_id in disabled_ids)
        )
        self.assertFalse(
            any(
                int(link[1]) in disabled_ids or int(link[3]) in disabled_ids
                for link in self.migrated["links"]
            )
        )

    def test_each_lane_cleans_conditioning_but_not_latent(self) -> None:
        marker = self.migrated["extra"][MIGRATION.MIGRATION_SCHEMA]
        self.assertEqual(len(marker["nodes"]["conditioning_cleanup"]), 4)
        subgraph = self.migrated["definitions"]["subgraphs"][0]
        links = {int(link["id"]): link for link in subgraph["links"]}
        h3_nodes = sorted(
            [node for node in subgraph["nodes"] if node["type"] == "MiniMaxH3ReferenceToVideo"],
            key=lambda node: node["pos"][1],
        )
        guiders = sorted(
            [node for node in subgraph["nodes"] if node["type"] == "BasicGuider"],
            key=lambda node: node["pos"][1],
        )
        samplers = sorted(
            [node for node in subgraph["nodes"] if node["type"] == "SamplerCustomAdvanced"],
            key=lambda node: node["pos"][1],
        )
        for h3, cleanup_id, guider, sampler in zip(
            h3_nodes,
            marker["nodes"]["conditioning_cleanup"],
            guiders,
            samplers,
        ):
            cleanup = _node_by_id(self.migrated, cleanup_id)
            self.assertEqual(cleanup["type"], MIGRATION.CLEANUP_NODE_TYPE)
            self.assertEqual(cleanup["inputs"][0]["name"], MIGRATION.CLEANUP_INPUT)
            self.assertEqual(cleanup["outputs"][0]["name"], MIGRATION.CLEANUP_OUTPUT)
            to_cleanup = links[int(cleanup["inputs"][0]["link"])]
            to_guider = links[int(guider["inputs"][1]["link"])]
            latent = links[int(sampler["inputs"][4]["link"])]
            self.assertEqual(int(to_cleanup["origin_id"]), int(h3["id"]))
            self.assertEqual(int(to_guider["origin_id"]), int(cleanup["id"]))
            self.assertEqual(int(latent["origin_id"]), int(h3["id"]))

    def test_fast_defaults_remain_configurable_after_creation(self) -> None:
        self.assertEqual(_node_by_id(self.migrated, 178)["widgets_values"], [5.0])
        self.assertEqual(_node_by_id(self.migrated, 187)["widgets_values"][1], 0.4)
        edited = copy.deepcopy(self.migrated)
        _node_by_id(edited, 178)["widgets_values"][0] = 12.0
        _node_by_id(edited, 187)["widgets_values"][1] = 0.8
        MIGRATION.validate_workflow(edited)
        self.assertEqual(MIGRATION.migrate_workflow(edited), edited)

    def test_project_native_shot_and_lane_contract_topology_is_preserved(self) -> None:
        before_links = {
            tuple(link)
            for link in self.source["links"]
            if int(link[1]) in {178, 676, 673, 677}
            or int(link[3]) in {178, 676, 673, 677}
        }
        after_links = {
            tuple(link)
            for link in self.migrated["links"]
            if int(link[1]) in {178, 676, 673, 677}
            or int(link[3]) in {178, 676, 673, 677}
        }
        self.assertEqual(before_links, after_links)
        self.assertEqual(
            self.migrated["extra"]["fixture"],
            {"native_shot_blocks": 4, "active_lanes": 1},
        )
        semantics = self.migrated["extra"][MIGRATION.MIGRATION_SCHEMA][
            "generation_semantics"
        ]
        self.assertEqual(semantics["lazy_generation_lane_capacity"], 4)
        self.assertTrue(semantics["project_and_native_shot_contract_preserved"])
        self.assertEqual(
            semantics["cleanup_node_type"], MIGRATION.CLEANUP_NODE_TYPE
        )
        self.assertFalse(semantics["cleanup_backend_output_node"])

    def test_changed_source_chain_fails_closed(self) -> None:
        broken = _fixture()
        outer = _node_by_id(broken, 710)
        link_id = outer["inputs"][0]["link"]
        link = next(link for link in broken["links"] if int(link[0]) == int(link_id))
        link[1] = 715
        with self.assertRaises(MIGRATION.WorkflowError):
            MIGRATION.migrate_workflow(broken, source_file_sha256="b" * 64)

    def test_cleanup_or_latent_contract_break_is_rejected(self) -> None:
        broken = copy.deepcopy(self.migrated)
        marker = broken["extra"][MIGRATION.MIGRATION_SCHEMA]
        cleanup = _node_by_id(broken, marker["nodes"]["conditioning_cleanup"][0])
        cleanup["outputs"][0]["links"] = []
        with self.assertRaises(MIGRATION.WorkflowError):
            MIGRATION.validate_workflow(broken)

    def test_file_migration_creates_only_a_new_sibling_and_is_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            output = root / "source_FAST.json"
            source_bytes = json.dumps(self.source, indent=2).encode("utf-8") + b"\n"
            source.write_bytes(source_bytes)
            source_hash = MIGRATION._file_hash(source_bytes)
            migrated, observed_hash, written = MIGRATION.migrate_file(
                source,
                output_path=output,
                write=True,
                expected_source_sha256=source_hash,
            )
            self.assertEqual(observed_hash, source_hash)
            self.assertEqual(written, output.resolve())
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), migrated)
            _same, _hash, second_written = MIGRATION.migrate_file(
                source,
                output_path=output,
                write=True,
                expected_source_sha256=source_hash,
            )
            self.assertIsNone(second_written)
            self.assertEqual(source.read_bytes(), source_bytes)

    def test_output_may_never_equal_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.json"
            source.write_text(json.dumps(self.source), encoding="utf-8")
            with self.assertRaises(MIGRATION.WorkflowError):
                MIGRATION.migrate_file(
                    source,
                    output_path=source,
                    write=True,
                    expected_source_sha256=MIGRATION._file_hash(source.read_bytes()),
                )


@unittest.skipUnless(LIVE_CAPTURE.is_file(), "exact live capture is absent")
class ExactLiveCaptureIntegrationTests(unittest.TestCase):
    def test_exact_captured_unsaved_canvas_migrates_without_writing(self) -> None:
        migrated, source_hash, written = MIGRATION.migrate_file(
            LIVE_CAPTURE,
            output_path=MIGRATION.DEFAULT_OUTPUT,
            write=False,
            expected_source_sha256=MIGRATION.SOURCE_CAPTURE_SHA256,
        )
        self.assertEqual(source_hash, MIGRATION.SOURCE_CAPTURE_SHA256)
        self.assertIsNone(written)
        MIGRATION.validate_workflow(migrated, require_benchmark_defaults=True)
        marker = migrated["extra"][MIGRATION.MIGRATION_SCHEMA]
        self.assertEqual(marker["source"]["file_sha256"], source_hash)
        self.assertEqual(
            marker["generation_semantics"]["lazy_generation_lane_capacity"], 4
        )


class InstalledCleanupBackendContractTests(unittest.TestCase):
    def test_live_backend_reports_cleanup_passthrough_is_not_an_output_node(self) -> None:
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8188/object_info/FL_UnloadAllModels",
                timeout=2.0,
            ) as response:
                payload = json.load(response)
        except (OSError, urllib.error.URLError) as exc:
            self.skipTest(f"local ComfyUI object_info is unavailable: {exc}")
        contract = payload["FL_UnloadAllModels"]
        self.assertFalse(contract["output_node"])
        self.assertEqual(contract["input_order"]["required"], ["value"])
        self.assertEqual(contract["output_name"], ["*"])
        self.assertEqual(contract["output"], ["*"])


if __name__ == "__main__":
    unittest.main()
