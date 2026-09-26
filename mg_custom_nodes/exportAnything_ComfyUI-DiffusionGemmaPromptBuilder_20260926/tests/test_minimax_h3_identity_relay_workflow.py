from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATH = REPO_ROOT / "tools" / "migrate_minimax_h3_identity_relay.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_identity_relay", MIGRATION_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot load migration module: {MIGRATION_PATH}")
MIGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIGRATION)

CAPTURE = Path(
    r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_EXECUTED_423S_CAPTURE_b43059cc_20260822.json"
)
SIBLING = Path(
    r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_DUAL_IDENTITY_P3_RELAY_GUARDED_V4_20260822.json"
)
SUPERSEDED_SIBLINGS = (
    Path(
        r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_DUAL_IDENTITY_P3_RELAY_GUARDED_20260822.json"
    ),
    Path(
        r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_DUAL_IDENTITY_P3_RELAY_GUARDED_V2_20260822.json"
    ),
    Path(
        r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_DUAL_IDENTITY_P3_RELAY_GUARDED_V3_20260822.json"
    ),
)
CAPTURE_SHA256 = "207ae087b7c8021ff396628480a4f382a21f993f1ddd27337bb051a0cbca3a17"
SIBLING_SHA256 = "fe4261c2dd4fb16df6acf05d8735beb274a8e2d80821936dcda0271d2978d6fa"
SUPERSEDED_SIBLING_SHA256 = (
    "c03c26244239f878195fccd521a9017a32f105f7209af06e0189afa1879c1281",
    "265f27188a689d517b6f55bda28b693b6881ce0068e712ea95fe5c4865a45f3f",
    "a1d203aebe00235a8743f4811b629295b927b4c080e89ab5eb9f9a182bfd355c",
)
LEGACY_SAVED_COPIES = (
    Path(
        r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_STABILIZED_H3.json"
    ),
    Path(
        r"C:\ComfyUI\app\user\default\workflows\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_STABILIZED_H3.json"
    ),
)
LEGACY_SAVED_SHA256 = "4431ef6269782be1cafbef350800ab01b376b094db7c62d7b8fe335004aac7d8"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _node(nodes: list[dict], node_id: int) -> dict:
    matches = [item for item in nodes if int(item.get("id", -1)) == int(node_id)]
    if len(matches) != 1:
        raise AssertionError(f"Expected one node {node_id}; found {len(matches)}")
    return matches[0]


def _one(nodes: list[dict], node_type: str) -> dict:
    matches = [item for item in nodes if item.get("type") == node_type]
    if len(matches) != 1:
        raise AssertionError(f"Expected one {node_type}; found {len(matches)}")
    return matches[0]


@unittest.skipUnless(CAPTURE.is_file(), "exact 423-second executed capture is absent")
class IdentityRelayMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.capture_bytes = CAPTURE.read_bytes()
        cls.source = json.loads(cls.capture_bytes.decode("utf-8"))
        cls.migrated = MIGRATION.migrate_workflow(
            cls.source,
            source_file_sha256=CAPTURE_SHA256,
            source_history_prompt_id=MIGRATION.DEFAULT_HISTORY_PROMPT_ID,
        )

    def test_capture_is_the_exact_successful_history_graph(self) -> None:
        self.assertEqual(_sha256(CAPTURE), CAPTURE_SHA256)
        marker = self.source["extra"][MIGRATION.SOURCE_EXPANSION_SCHEMA]
        self.assertEqual(marker["runtime_lane_node_ids"]["h3"], [667, 684, 692, 700])
        self.assertEqual(
            marker["runtime_lane_node_ids"]["tails"], [682, 690, 698]
        )

    def test_source_object_is_not_mutated(self) -> None:
        before = copy.deepcopy(self.source)
        MIGRATION.migrate_workflow(
            self.source,
            source_file_sha256=CAPTURE_SHA256,
            source_history_prompt_id=MIGRATION.DEFAULT_HISTORY_PROMPT_ID,
        )
        self.assertEqual(self.source, before)
        self.assertEqual(CAPTURE.read_bytes(), self.capture_bytes)

    def test_dual_identity_and_shared_manifest_are_guarded(self) -> None:
        nodes = self.migrated["nodes"]
        planner = _one(nodes, MIGRATION.PLANNER_TYPE)
        context = _one(nodes, MIGRATION.CONTEXT_TYPE)
        pair = _one(nodes, MIGRATION.PAIR_PREP_TYPE)
        batch = _one(nodes, "BatchImagesNode")
        manifest = next(
            item
            for item in nodes
            if item.get("type") == "PrimitiveStringMultiline"
            and item.get("title", "").startswith("H3 REFERENCE MANIFEST")
        )
        self.assertEqual(planner["widgets_values"][-2:], [2, "Off"])
        self.assertEqual(context["widgets_values"][-1], 1)
        self.assertEqual(pair["widgets_values"][-2:], [1.0, 0.40])
        self.assertEqual(manifest["widgets_values"], [MIGRATION.DUAL_IDENTITY_MANIFEST])
        self.assertIn("do not transfer its layout, grid, seams", MIGRATION.DUAL_IDENTITY_MANIFEST)
        self.assertIn("panels are not separate people", MIGRATION.DUAL_IDENTITY_MANIFEST)
        context_origin, _ = MIGRATION._origin_main(
            self.migrated, context, "reference_manifest"
        )
        planner_origin, _ = MIGRATION._origin_main(
            self.migrated, planner, "reference_manifest"
        )
        image_origin, _ = MIGRATION._origin_main(
            self.migrated, context, "reference_images"
        )
        self.assertEqual(context_origin["id"], manifest["id"])
        self.assertEqual(planner_origin["id"], manifest["id"])
        self.assertEqual(image_origin["id"], batch["id"])

    def test_picture_2_uses_the_hash_guarded_generated_identity_sheet(self) -> None:
        extension = self.migrated["extra"][MIGRATION.SOURCE_EXPANSION_SCHEMA][
            "identity_relay_extension"
        ]
        secondary = _node(
            self.migrated["nodes"], extension["secondary_identity_loader"]
        )
        self.assertEqual(
            secondary["widgets_values"][0],
            MIGRATION.SECONDARY_IDENTITY_RELATIVE_PATH,
        )
        self.assertEqual(
            Path(MIGRATION.SECONDARY_IDENTITY_RELATIVE_PATH).parent, Path(".")
        )
        self.assertNotIn("/", MIGRATION.SECONDARY_IDENTITY_RELATIVE_PATH)
        self.assertNotIn("\\", MIGRATION.SECONDARY_IDENTITY_RELATIVE_PATH)
        self.assertIn("generated multi-view identity sheet", secondary["title"])
        self.assertEqual(
            extension["secondary_identity_asset"]["sha256"],
            MIGRATION.SECONDARY_IDENTITY_SHA256,
        )
        asset = MIGRATION.DEFAULT_COMFY_INPUT_ROOT / Path(
            MIGRATION.SECONDARY_IDENTITY_RELATIVE_PATH
        )
        self.assertEqual(_sha256(asset), MIGRATION.SECONDARY_IDENTITY_SHA256)
        nested_source = (
            MIGRATION.DEFAULT_COMFY_INPUT_ROOT
            / "identity_refs"
            / MIGRATION.SECONDARY_IDENTITY_RELATIVE_PATH
        )
        self.assertTrue(nested_source.is_file())
        self.assertEqual(_sha256(nested_source), MIGRATION.SECONDARY_IDENTITY_SHA256)
        self.assertEqual(asset.read_bytes(), nested_source.read_bytes())

    def test_explanatory_text_is_truthful_about_preserved_b430_settings(self) -> None:
        start = _node(self.migrated["nodes"], 116)["widgets_values"][0]
        outer = _node(self.migrated["nodes"], 710)
        subgraph = next(
            item
            for item in self.migrated["definitions"]["subgraphs"]
            if item["id"] == MIGRATION.FAST_SUBGRAPH_ID
        )
        sampler = _node(subgraph["nodes"], 657)
        scheduler = _node(subgraph["nodes"], 658)
        lora = _node(self.migrated["nodes"], 713)
        self.assertNotIn("4-step", start)
        self.assertNotIn("Euler/simple", start)
        self.assertNotIn("Turbo 4-step", outer["title"])
        self.assertIn("res_multistep", sampler["title"])
        self.assertIn("simple / 8", scheduler["title"])
        self.assertIn("0.60", lora["title"])
        self.assertIn("res_multistep", subgraph["name"])
        self.assertEqual(sampler["widgets_values"], ["res_multistep"])
        self.assertEqual(scheduler["widgets_values"], ["simple", 8, 1])
        self.assertEqual(lora["widgets_values"][1], 0.60)
        extension = self.migrated["extra"][MIGRATION.SOURCE_EXPANSION_SCHEMA][
            "identity_relay_extension"
        ]
        self.assertEqual(
            extension["observed_executed_fast_controls"],
            {
                "sampler_node": 657,
                "sampler": "res_multistep",
                "scheduler_node": 658,
                "scheduler": "simple",
                "steps": 8,
                "denoise": 1.0,
                "lora_node": 713,
                "lora_name": lora["widgets_values"][0],
                "lora_strength": 0.60,
            },
        )

    def test_every_lane_has_two_clean_identity_inputs_and_relay_moves_to_p3(self) -> None:
        extension = self.migrated["extra"][MIGRATION.SOURCE_EXPANSION_SCHEMA][
            "identity_relay_extension"
        ]
        subgraph = next(
            item
            for item in self.migrated["definitions"]["subgraphs"]
            if item["id"] == MIGRATION.FAST_SUBGRAPH_ID
        )
        runtime = self.migrated["extra"][MIGRATION.SOURCE_EXPANSION_SCHEMA][
            "runtime_lane_node_ids"
        ]
        h3_nodes = [_node(subgraph["nodes"], item) for item in runtime["h3"]]
        gates = [
            _node(subgraph["nodes"], item) for item in extension["relay_gate_nodes"]
        ]
        scalers = [
            _node(subgraph["nodes"], item) for item in extension["relay_scale_nodes"]
        ]
        p2_slot = next(
            index
            for index, item in enumerate(subgraph["inputs"])
            if item["name"] == "identity_picture_2"
        )
        for lane, h3 in enumerate(h3_nodes, start=1):
            p2_origin, p2_origin_slot, _ = MIGRATION._origin_subgraph(
                subgraph, h3, "ref_images.ref_image_1"
            )
            self.assertEqual((p2_origin, p2_origin_slot), (-10, p2_slot))
            if lane > 1:
                p3_origin, _p3_slot, _ = MIGRATION._origin_subgraph(
                    subgraph, h3, "ref_images.ref_image_2"
                )
                self.assertEqual(p3_origin, gates[lane - 2]["id"])
        for scaler in scalers:
            self.assertEqual(scaler["type"], "ImageScaleToTotalPixels")
            self.assertEqual(
                scaler["widgets_values"],
                [
                    "area",
                    MIGRATION.RELAY_REFERENCE_MEBIPIXELS,
                    MIGRATION.RELAY_REFERENCE_RESOLUTION_STEPS,
                ],
            )

    def test_423_second_fast_path_controls_are_identical(self) -> None:
        source_fast = self.source["extra"]["diffusiongemma.minimax_h3_fast_test"]
        migrated_fast = self.migrated["extra"]["diffusiongemma.minimax_h3_fast_test"]
        self.assertEqual(migrated_fast, source_fast)
        source_subgraph = next(
            item
            for item in self.source["definitions"]["subgraphs"]
            if item["id"] == MIGRATION.FAST_SUBGRAPH_ID
        )
        migrated_subgraph = next(
            item
            for item in self.migrated["definitions"]["subgraphs"]
            if item["id"] == MIGRATION.FAST_SUBGRAPH_ID
        )
        self.assertEqual(_node(self.migrated["nodes"], 713)["widgets_values"][1], 0.60)
        self.assertEqual(
            _node(migrated_subgraph["nodes"], 657)["widgets_values"],
            ["res_multistep"],
        )
        self.assertEqual(
            _node(migrated_subgraph["nodes"], 658)["widgets_values"],
            ["simple", 8, 1.0],
        )
        for node_id in (
            656,
            657,
            658,
            659,
            660,
            663,
            685,
            686,
            687,
            688,
            693,
            694,
            695,
            696,
            701,
            702,
            703,
            704,
            717,
            718,
            719,
            720,
        ):
            migrated_node = copy.deepcopy(_node(migrated_subgraph["nodes"], node_id))
            source_node = copy.deepcopy(_node(source_subgraph["nodes"], node_id))
            if node_id in {657, 658}:
                migrated_node.pop("title", None)
                source_node.pop("title", None)
            self.assertEqual(migrated_node, source_node)

    def test_validator_fails_closed_on_relay_or_fast_path_corruption(self) -> None:
        broken_relay = copy.deepcopy(self.migrated)
        extension = broken_relay["extra"][MIGRATION.SOURCE_EXPANSION_SCHEMA][
            "identity_relay_extension"
        ]
        subgraph = next(
            item
            for item in broken_relay["definitions"]["subgraphs"]
            if item["id"] == MIGRATION.FAST_SUBGRAPH_ID
        )
        _node(subgraph["nodes"], extension["relay_scale_nodes"][0])["widgets_values"][
            1
        ] = 1.0
        with self.assertRaises(MIGRATION.WorkflowError):
            MIGRATION.validate_workflow(broken_relay, source=self.source)

        broken_fast = copy.deepcopy(self.migrated)
        _node(broken_fast["nodes"], 713)["widgets_values"][1] = 0.5
        with self.assertRaises(MIGRATION.WorkflowError):
            MIGRATION.validate_workflow(broken_fast, source=self.source)

    def test_exclusive_writer_never_overwrites_a_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "sibling.json"
            first_hash = MIGRATION._exclusive_json_write(output, self.migrated)
            before = output.read_bytes()
            self.assertEqual(first_hash, hashlib.sha256(before).hexdigest())
            with self.assertRaises(FileExistsError):
                MIGRATION._exclusive_json_write(output, self.migrated)
            self.assertEqual(output.read_bytes(), before)


@unittest.skipUnless(SIBLING.is_file(), "guarded dual-identity sibling is absent")
class WrittenSiblingTests(unittest.TestCase):
    def test_written_sibling_hash_and_provenance(self) -> None:
        self.assertEqual(_sha256(SIBLING), SIBLING_SHA256)
        workflow = json.loads(SIBLING.read_text(encoding="utf-8"))
        marker = workflow["extra"][MIGRATION.IDENTITY_RELAY_SCHEMA]
        self.assertEqual(marker["source_workflow_file_sha256"], CAPTURE_SHA256)
        self.assertEqual(
            marker["source_history_prompt_id"], MIGRATION.DEFAULT_HISTORY_PROMPT_ID
        )
        MIGRATION.validate_workflow(workflow)

    def test_original_saved_workflows_remain_byte_identical(self) -> None:
        for path in LEGACY_SAVED_COPIES:
            if path.is_file():
                self.assertEqual(_sha256(path), LEGACY_SAVED_SHA256, str(path))

    def test_superseded_sibling_was_not_overwritten(self) -> None:
        for path, expected in zip(
            SUPERSEDED_SIBLINGS, SUPERSEDED_SIBLING_SHA256
        ):
            if path.is_file():
                self.assertEqual(_sha256(path), expected, str(path))

    def test_root_picture_2_is_enumerated_by_live_loadimage_inventory(self) -> None:
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8188/object_info/LoadImage", timeout=2.0
            ) as response:
                contract = json.load(response)["LoadImage"]
        except (OSError, KeyError, urllib.error.URLError) as exc:
            self.skipTest(f"local ComfyUI LoadImage inventory unavailable: {exc}")
        choices = contract["input"]["required"]["image"][0]
        self.assertIn(MIGRATION.SECONDARY_IDENTITY_RELATIVE_PATH, choices)
        self.assertNotIn(
            f"identity_refs/{MIGRATION.SECONDARY_IDENTITY_RELATIVE_PATH}", choices
        )


if __name__ == "__main__":
    unittest.main()
