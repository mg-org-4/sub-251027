from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "tools" / "build_advertisement_music3_glowbloom_api.py"
SPEC = importlib.util.spec_from_file_location("advertisement_music3_api_builder", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


class AdvertisementMusic3APIFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.known_good_bytes = builder.KNOWN_GOOD_UI.read_bytes()
        cls.source = builder._load_guarded_json(
            builder.SOURCE_API,
            canonical_hash=builder.SOURCE_API_CANONICAL_SHA256,
        )
        cls.graph = builder.build_graph(cls.source)

    def test_known_good_v6_is_hash_guarded_and_unchanged(self):
        self.assertEqual(
            hashlib.sha256(self.known_good_bytes).hexdigest(),
            builder.KNOWN_GOOD_UI_SHA256,
        )

    def test_graph_is_acyclic_and_reference_prep_has_explicit_format_size(self):
        builder.validate_graph(self.graph)
        self.assertEqual(self.graph["803"]["inputs"]["generation_width"], 480)
        self.assertEqual(self.graph["803"]["inputs"]["generation_height"], 864)
        self.assertFalse(
            any(
                isinstance(value, list) and value[0] == "187"
                for value in self.graph["803"]["inputs"].values()
            )
        )

    def test_graph_rejects_a_dependency_cycle(self):
        graph = copy.deepcopy(self.graph)
        graph["803"]["inputs"]["generation_width"] = ["187", 10]
        with self.assertRaisesRegex(builder.FixtureError, "dependency cycle"):
            builder.validate_graph(graph)

    def test_runtime_baseline_is_self_contained(self):
        classes = {node["class_type"] for node in self.graph.values()}
        self.assertIn("DiffusionGemmaAdvertisementMemoryBarrier", classes)
        self.assertIn("DiffusionGemmaAdvertisementDirectorPacketRepair", classes)
        self.assertNotIn("PathchSageAttentionKJ", classes)
        self.assertNotIn("easy cleanGpuUsed", classes)
        self.assertNotIn("FL_UnloadAllModels", classes)
        self.assertNotIn("EasyCache", classes)
        self.assertNotIn("TorchCompileModel", classes)
        self.assertEqual(self.graph["716"]["inputs"]["model"], ["713", 0])
        self.assertEqual(self.graph["182"]["inputs"]["model_path"], builder.DIRECTOR_MODEL_PATH)

    def test_visible_h3_labels_match_the_actual_reference_and_step_contracts(self):
        self.assertEqual(
            self.graph["710:658"]["_meta"]["title"],
            "Shared H3 schedule — simple / 7 sampling steps",
        )
        self.assertEqual(self.graph["710:658"]["inputs"]["steps"], 7)
        expected = (
            "H3 GENERATION LANE 1 — P1+P2 performer + P3 product + Audio 1",
            "H3 GENERATION LANE 2 — P1+P2 performer + P3 product + P4 relay + Audio 1",
            "H3 GENERATION LANE 3 — P1+P2 performer + P3 product + P4 relay + Audio 1",
            "H3 GENERATION LANE 4 — P1+P2 performer + P3 product + P4 relay + Audio 1",
        )
        for node_id, title in zip(("710:667", "710:684", "710:692", "710:700"), expected):
            self.assertEqual(self.graph[node_id]["_meta"]["title"], title)

    def test_manifest_describes_the_exact_api_graph_and_uses_real_hashes(self):
        manifest = builder._manifest(self.graph)
        self.assertEqual(manifest["api_audio_sources"], ["MiniMax Music 3"])
        self.assertEqual(
            manifest["companion_ui_only_audio_sources"],
            ["Upload song", "Legacy ACE-Step"],
        )
        self.assertEqual(manifest["models"], builder.MODEL_INVENTORY)
        self.assertEqual(set(manifest["models"]), {"director", "minimax_h3", "minimax_music3"})
        self.assertEqual(len(manifest["models"]["minimax_h3"]), 5)
        for asset in manifest["assets"].values():
            self.assertRegex(asset["sha256"], re.compile(r"^[0-9a-f]{64}$"))

    def test_director_transport_repair_is_narrow_and_observable(self):
        self.assertEqual(self.graph["185"]["inputs"]["mode"], "off")
        self.assertEqual(
            self.graph["185"]["_meta"]["title"],
            "5b. H3 Grounding Guard — OFF baseline (contract validation stays active)",
        )
        self.assertEqual(
            self.graph["820"]["class_type"],
            "DiffusionGemmaAdvertisementDirectorPacketRepair",
        )
        self.assertEqual(self.graph["820"]["inputs"]["director_final_json"], ["186", 0])
        self.assertEqual(self.graph["820"]["inputs"]["director_raw_response"], ["186", 2])
        self.assertEqual(self.graph["820"]["inputs"]["director_metadata_json"], ["186", 3])
        self.assertEqual(self.graph["820"]["inputs"]["grounding_status"], ["186", 4])
        self.assertEqual(self.graph["820"]["inputs"]["grounding_report_json"], ["186", 5])
        self.assertEqual(self.graph["820"]["inputs"]["reference_contract_json"], ["802", 0])
        self.assertEqual(self.graph["187"]["inputs"]["final_json"], ["820", 0])
        self.assertEqual(self.graph["902"]["inputs"]["source"], ["186", 2])

    def test_music3_and_reference_topology_matches_advertisement_contract(self):
        self.assertEqual(
            self.graph["806"]["inputs"]["clip_name"],
            builder.MUSIC3_MODELS["text_encoder"],
        )
        self.assertEqual(
            self.graph["807"]["inputs"]["unet_name"],
            builder.MUSIC3_MODELS["diffusion_model"],
        )
        self.assertEqual(
            self.graph["808"]["inputs"]["vae_name"],
            builder.MUSIC3_MODELS["vae"],
        )
        for lane, node_id in enumerate(("710:667", "710:684", "710:692", "710:700"), start=1):
            inputs = self.graph[node_id]["inputs"]
            self.assertNotIn("model", inputs)
            self.assertEqual(inputs["ref_images.ref_image_0"], ["803", 0])
            self.assertEqual(inputs["ref_images.ref_image_1"], ["803", 1])
            self.assertEqual(inputs["ref_images.ref_image_2"], ["803", 2])
            self.assertEqual("ref_images.ref_image_3" in inputs, lane > 1)
        for node_id in ("710:658", "710:660", "710:686", "710:694", "710:702"):
            self.assertEqual(self.graph[node_id]["inputs"]["model"], ["716", 0])

    def test_fixture_has_master_two_real_cutdowns_and_honest_qa(self):
        save_nodes = [
            node for node in self.graph.values() if node["class_type"] == "SaveVideo"
        ]
        self.assertEqual(len(save_nodes), 3)
        self.assertTrue(all("REVIEW DRAFT" in node["_meta"]["title"] for node in save_nodes))
        self.assertTrue(
            all("/review_drafts/" in node["inputs"]["filename_prefix"] for node in save_nodes)
        )
        checks = json.loads(self.graph["836"]["inputs"]["checks_json"])
        self.assertEqual(checks["technical_integrity"]["status"], "pass")
        self.assertEqual(checks["product_identity"], "not_measured")
        self.assertEqual(checks["performer_identity"], "not_measured")
        self.assertEqual(checks["copy_legibility"], "not_measured")
        self.assertEqual(checks["audio_sync"], "not_measured")
        self.assertEqual(self.graph["900"]["class_type"], "PreviewAny")
        self.assertEqual(self.graph["900"]["inputs"]["source"], ["815", 4])
        self.assertEqual(self.graph["901"]["class_type"], "PreviewAny")
        self.assertEqual(self.graph["901"]["inputs"]["source"], ["187", 4])
        self.assertEqual(self.graph["902"]["class_type"], "PreviewAny")
        self.assertEqual(self.graph["902"]["inputs"]["source"], ["186", 2])

    def test_one_settings_node_controls_every_duplicate_prone_consumer(self):
        self.assertEqual(
            self.graph["849"]["class_type"],
            "DiffusionGemmaAdvertisementWorkflowControls",
        )
        for node_id, input_name, slot in (
            ("801", "production_duration_seconds", 0),
            ("804", "target_duration_seconds", 0),
            ("183", "target_duration_seconds", 0),
            ("673", "target_duration_seconds", 0),
            ("676", "production_duration_seconds", 0),
            ("815", "excerpt_duration_seconds", 0),
            ("816", "duration", 0),
            ("817", "target_duration_seconds", 0),
            ("818", "excerpt_duration_seconds", 0),
            ("673", "custom_shot_count", 1),
            ("821", "native_shot_count", 1),
            ("675", "performance_mode", 2),
            ("823", "performance_mode", 2),
            ("835", "cutdown_15_start_seconds", 3),
            ("835", "cutdown_6_start_seconds", 4),
            ("801", "deliverables_json", 5),
            ("676", "deliverables_json", 5),
            ("801", "aspect_ratio", 6),
            ("676", "aspect_ratio", 6),
            ("673", "generation_mode", 7),
            ("673", "shot_count", 8),
            ("673", "shot_count_override", 9),
            ("673", "audio_mode", 10),
            ("673", "dialogue_mode", 11),
            ("676", "excerpt_start_seconds", 12),
            ("676", "generation_model", 13),
            ("676", "max_h3_shot_seconds", 14),
            ("187", "resolution_aspect_ratio_override", 15),
        ):
            self.assertEqual(self.graph[node_id]["inputs"][input_name], ["849", slot])
        self.assertEqual(self.graph["818"]["inputs"]["expected_bpm"], ["814", 5])


if __name__ == "__main__":
    unittest.main()
