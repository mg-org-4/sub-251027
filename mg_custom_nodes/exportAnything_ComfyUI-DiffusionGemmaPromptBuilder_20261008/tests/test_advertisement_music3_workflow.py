from __future__ import annotations

import copy
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "tools" / "build_advertisement_music3_workflow.py"
SPEC = importlib.util.spec_from_file_location("advertisement_music3_workflow_builder", BUILDER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError("Could not load Advertisement Music3 workflow builder.")
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _scalar_definition(value: object) -> list[object]:
    if isinstance(value, bool):
        return ["BOOLEAN", {"default": value}]
    if isinstance(value, int):
        return ["INT", {"default": value}]
    if isinstance(value, float):
        return ["FLOAT", {"default": value}]
    return ["STRING", {"default": value, "multiline": isinstance(value, str) and "\n" in value}]


def synthetic_object_info(graph: dict[str, object]) -> dict[str, object]:
    """Build the smallest /object_info envelope needed by this exact graph."""

    class_inputs: dict[str, dict[str, list[object]]] = {}
    class_order: dict[str, list[str]] = {}
    maximum_output_slot: dict[str, int] = {}
    class_titles: dict[str, str] = {}
    for node in graph.values():
        class_type = str(node["class_type"])
        class_inputs.setdefault(class_type, {})
        class_order.setdefault(class_type, [])
        class_titles.setdefault(class_type, class_type)
        for name, value in node.get("inputs", {}).items():
            if name not in class_order[class_type]:
                class_order[class_type].append(name)
            if not builder._is_connection(value):
                class_inputs[class_type][name] = _scalar_definition(value)
            else:
                class_inputs[class_type].setdefault(name, ["*", {}])
    for node in graph.values():
        for value in node.get("inputs", {}).values():
            if not builder._is_connection(value):
                continue
            origin_id, origin_slot = value
            origin_type = str(graph[origin_id]["class_type"])
            maximum_output_slot[origin_type] = max(
                maximum_output_slot.get(origin_type, -1), int(origin_slot)
            )
    result: dict[str, object] = {}
    for class_type, inputs in class_inputs.items():
        output_count = maximum_output_slot.get(class_type, -1) + 1
        result[class_type] = {
            "input": {"required": inputs},
            "input_order": {"required": class_order[class_type]},
            "output": ["*"] * output_count,
            "output_name": [f"output_{index}" for index in range(output_count)],
            "display_name": class_titles[class_type],
            "python_module": (
                "custom_nodes.ComfyUI-DiffusionGemmaPromptBuilder"
                if class_type.startswith("DiffusionGemma")
                else "test.hermetic_object_info"
            ),
        }
    return result


class _ObjectInfoServer:
    def __init__(self, payload: dict[str, object], port: int = 0):
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib handler API
                if self.path.rstrip("/") != "/object_info":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, _format, *args):
                del args

        self.server = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _node(workflow: dict[str, object], class_type: str) -> dict[str, object]:
    matches = [node for node in workflow["nodes"] if node.get("type") == class_type]
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one {class_type}, found {len(matches)}.")
    return matches[0]


def _link(workflow: dict[str, object], target: dict[str, object], input_name: str) -> list[object]:
    slot = next(item for item in target["inputs"] if item.get("name") == input_name)
    return next(item for item in workflow["links"] if item[0] == slot["link"])


class AdvertisementMusic3WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field_test = builder._load_guarded(
            builder.FIELD_TEST_API,
            canonical_hash=builder.FIELD_TEST_API_CANONICAL_SHA256,
        )
        cls.music3 = builder._load_guarded(
            builder.MUSIC3_API,
            canonical_hash=builder.MUSIC3_API_CANONICAL_SHA256,
        )
        cls.graph = builder._with_lazy_fallbacks(cls.music3, cls.field_test)
        cls.object_info = synthetic_object_info(cls.graph)
        cls.workflow = builder.build_workflow(cls.graph, cls.object_info)

    def test_v6_byte_hash_is_guarded_and_output_records_lineage(self):
        source = builder._load_guarded(
            builder.KNOWN_GOOD_UI,
            byte_hash=builder.KNOWN_GOOD_UI_SHA256,
        )
        self.assertEqual(
            builder._sha256_bytes(builder.KNOWN_GOOD_UI.read_bytes()),
            builder.KNOWN_GOOD_UI_SHA256,
        )
        marker = self.workflow["extra"][builder.SCHEMA]
        self.assertEqual(marker["known_good_v6_file_sha256"], builder.KNOWN_GOOD_UI_SHA256)
        self.assertTrue(marker["known_good_v6_is_never_overwritten"])
        with tempfile.TemporaryDirectory() as temporary:
            tampered = Path(temporary) / "v6.json"
            tampered.write_bytes(builder.KNOWN_GOOD_UI.read_bytes() + b"\n")
            with self.assertRaisesRegex(builder.WorkflowError, "Immutable source changed"):
                builder._load_guarded(tampered, byte_hash=builder.KNOWN_GOOD_UI_SHA256)
        self.assertNotEqual(self.workflow["id"], source.get("id"))

    def test_workflow_uuid_is_fresh_valid_and_flattened(self):
        parsed = uuid.UUID(str(self.workflow["id"]))
        self.assertEqual(str(parsed), builder.WORKFLOW_UUID)
        self.assertEqual(self.workflow["revision"], 1)
        marker = self.workflow["extra"][builder.SCHEMA]
        self.assertTrue(marker["workflow_uuid_is_fresh"])
        self.assertEqual(marker["graph_form"], "flattened_independent_no_shared_subgraph_uuid")

    def test_music3_default_has_lazy_upload_and_ace_fallback_links(self):
        router = _node(self.workflow, "DiffusionGemmaAdvertisementSoundtrackSourceRouter")
        self.assertEqual(router["widgets_values"][0], "MiniMax Music 3")
        music_link = _link(self.workflow, router, "music3_candidate_1")
        upload_link = _link(self.workflow, router, "uploaded_audio")
        ace_links = [_link(self.workflow, router, f"ace_candidate_{index}") for index in range(1, 5)]
        by_id = {node["id"]: node for node in self.workflow["nodes"]}
        self.assertEqual(by_id[music_link[1]]["type"], "VAEDecodeAudioTiled")
        self.assertEqual(by_id[upload_link[1]]["type"], "DiffusionGemmaUploadSong")
        self.assertTrue(all(by_id[link[1]]["type"] == "VAEDecodeAudio" for link in ace_links))
        self.assertEqual(self.workflow["extra"][builder.SCHEMA]["audio_source_default"], "MiniMax Music 3")
        self.assertEqual(
            self.workflow["extra"][builder.SCHEMA]["fallbacks"],
            ["Upload song", "Legacy ACE-Step"],
        )
        selectors = [
            node
            for node in self.workflow["nodes"]
            if node.get("type") == "DiffusionGemmaAdvertisementAudioCandidateSelector"
        ]
        verifier = next(
            node for node in selectors if "Verify exact" in str(node.get("title", ""))
        )
        policy_link = _link(self.workflow, verifier, "source_policy")
        self.assertEqual(by_id[policy_link[1]]["type"], "DiffusionGemmaAdvertisementSoundtrackSourceRouter")
        self.assertEqual(policy_link[2], 8)
        bpm_link = _link(self.workflow, verifier, "expected_bpm")
        self.assertEqual(by_id[bpm_link[1]]["type"], "DiffusionGemmaAdvertisementSoundtrackSourceRouter")
        self.assertEqual(bpm_link[2], 5)

    def test_h3_topology_keeps_p1_p2_p3_and_persistent_p4_only_after_lane_one(self):
        h3_nodes = sorted(
            [node for node in self.workflow["nodes"] if node.get("type") == "MiniMaxH3ReferenceToVideo"],
            key=lambda node: float(node["pos"][1]),
        )
        self.assertEqual(len(h3_nodes), 4)
        by_id = {node["id"]: node for node in self.workflow["nodes"]}
        for lane, node in enumerate(h3_nodes, start=1):
            for name, expected_slot in (
                ("ref_images.ref_image_0", 0),
                ("ref_images.ref_image_1", 1),
                ("ref_images.ref_image_2", 2),
            ):
                link = _link(self.workflow, node, name)
                self.assertEqual(by_id[link[1]]["type"], "DiffusionGemmaAdvertisementReferenceAssetPrep")
                self.assertEqual(link[2], expected_slot)
            p4 = [item for item in node["inputs"] if item.get("name") == "ref_images.ref_image_3"]
            if lane == 1:
                self.assertEqual(p4, [])
            else:
                self.assertEqual(len(p4), 1)
                link = next(item for item in self.workflow["links"] if item[0] == p4[0]["link"])
                self.assertEqual(by_id[link[1]]["type"], "DiffusionGemmaAdvertisementRelayGate")
        self.assertEqual(
            sum(node.get("type") == "DiffusionGemmaAdvertisementRelayArtifact" for node in self.workflow["nodes"]),
            3,
        )
        self.assertEqual(
            sum(node.get("type") == "DiffusionGemmaAdvertisementRelayGate" for node in self.workflow["nodes"]),
            3,
        )

    def test_baseline_rejects_acceleration_and_inherited_convenience_runtime_nodes(self):
        types = [str(node.get("type", "")) for node in self.workflow["nodes"]]
        self.assertFalse(set(types).intersection(builder.FORBIDDEN_RUNTIME_CLASSES))
        self.assertEqual(types.count("DiffusionGemmaAdvertisementMemoryBarrier"), 5)
        self.assertNotIn("DiffusionGemmaSplatStagePlanner", types)
        self.assertNotIn("SplatStageBlueprintRouter", types)
        self.assertNotIn("DiffusionGemmaContextHub", types)
        self.assertEqual(types.count("SaveVideo"), 3)
        self.assertEqual(types.count("PreviewAny"), 4)
        self.assertEqual(types.count("PreviewAudio"), 1)
        self.assertEqual(types.count("SaveAudio"), 1)

    def test_saved_workflow_is_live_metadata_built_and_has_editable_performance_guidance(self):
        saved = json.loads(builder.OUTPUT.read_text(encoding="utf-8"))
        builder.validate_workflow(saved)
        self.assertEqual(len(saved["nodes"]), 135)
        self.assertEqual(len(saved["links"]), 356)
        controls = _node(saved, "DiffusionGemmaAdvertisementWorkflowControls")
        self.assertEqual(controls["widgets_values"], ["30 seconds", 8, "Natural / audio-led sync"])
        combo_outputs = {
            "performance_mode",
            "master_aspect_ratio",
            "h3_generation_mode",
            "h3_shot_count_mode",
            "h3_audio_mode",
            "h3_dialogue_mode",
        }
        by_id = {node["id"]: node for node in saved["nodes"]}
        by_link = {link[0]: link for link in saved["links"]}
        for output in controls["outputs"]:
            if output["name"] not in combo_outputs:
                continue
            self.assertEqual(output["type"], "COMBO")
            for link_id in output.get("links") or []:
                link = by_link[link_id]
                self.assertEqual(link[5], "COMBO")
                target = by_id[link[3]]
                self.assertEqual(target["inputs"][link[4]]["type"], "COMBO")
        performance = _node(saved, "DiffusionGemmaMusicVideoPerformanceMode")
        guidance_link = _link(saved, performance, "base_audio_guidance")
        self.assertEqual(by_id[guidance_link[1]]["type"], "PrimitiveStringMultiline")
        note = _node(saved, "MarkdownNote")
        self.assertEqual(note["properties"], {})
        load_images = [node for node in saved["nodes"] if node.get("type") == "LoadImage"]
        self.assertEqual(len(load_images), 3)
        self.assertTrue(
            all(
                len(node["widgets_values"]) == 2
                and node["widgets_values"][1] == "image"
                for node in load_images
            )
        )
        expected_inputs = [
            {
                "localized_name": "image",
                "name": "image",
                "type": "COMBO",
                "widget": {"name": "image"},
                "link": None,
            },
            {
                "localized_name": "choose file to upload",
                "name": "upload",
                "type": "IMAGEUPLOAD",
                "widget": {"name": "upload"},
                "link": None,
            },
        ]
        self.assertTrue(all(node["inputs"] == expected_inputs for node in load_images))

    def test_check_and_generation_run_against_hermetic_object_info(self):
        with _ObjectInfoServer(self.object_info) as server, tempfile.TemporaryDirectory() as temporary:
            check_stdout = io.StringIO()
            with redirect_stdout(check_stdout):
                self.assertEqual(
                    builder.main(["--object-info-url", server.url, "--check"]),
                    0,
                )
            checked = json.loads(check_stdout.getvalue())
            self.assertEqual(checked["workflow_id"], builder.WORKFLOW_UUID)
            output = Path(temporary) / "advertisement.json"
            generate_stdout = io.StringIO()
            with redirect_stdout(generate_stdout):
                self.assertEqual(
                    builder.main(
                        ["--object-info-url", server.url, "--output", str(output)]
                    ),
                    0,
                )
            generated = json.loads(output.read_text(encoding="utf-8"))
            summary = json.loads(generate_stdout.getvalue())
            self.assertEqual(summary["action"], "created")
            self.assertEqual(summary["sha256"], builder._sha256_bytes(output.read_bytes()))
            self.assertEqual(generated, self.workflow)

    def test_validator_rejects_nonmatching_saved_socket_types(self):
        saved = json.loads(builder.OUTPUT.read_text(encoding="utf-8"))
        tampered = copy.deepcopy(saved)
        campaign = _node(tampered, "DiffusionGemmaAdvertisementCampaignContract")
        duration_input = next(
            item for item in campaign["inputs"] if item["name"] == "production_duration_seconds"
        )
        duration_input["type"] = "INT"
        with self.assertRaisesRegex(builder.WorkflowError, "link type mismatch"):
            builder.validate_workflow(tampered)

    def test_live_builder_rejects_an_unknown_api_input_name(self):
        graph = copy.deepcopy(self.graph)
        graph["710:667"]["inputs"]["definitely_unknown"] = ["716", 0]
        with self.assertRaisesRegex(
            builder.WorkflowError,
            r"MiniMaxH3ReferenceToVideo has no connected input definitely_unknown \(710:667\)",
        ):
            builder.build_workflow(graph, self.object_info)


if __name__ == "__main__":
    unittest.main()
