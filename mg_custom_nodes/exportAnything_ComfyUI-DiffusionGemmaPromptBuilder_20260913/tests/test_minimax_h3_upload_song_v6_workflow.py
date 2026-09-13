from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "migrate_minimax_h3_upload_song_v6.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_upload_song_v6", TOOL_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot load migration tool: {TOOL_PATH}")
MIGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIGRATION)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(workflow: dict) -> str:
    payload = json.dumps(
        workflow,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _node(workflow: dict, node_id: int) -> dict:
    matches = [node for node in workflow["nodes"] if int(node["id"]) == int(node_id)]
    if len(matches) != 1:
        raise AssertionError(f"Expected one node {node_id}; found {len(matches)}")
    return matches[0]


def _origin(workflow: dict, target: dict, input_name: str) -> tuple[dict, str]:
    return MIGRATION._origin(workflow, target, input_name)


class MiniMaxH3UploadSongV6RepositoryArtifactTests(unittest.TestCase):
    def test_checked_in_operational_workflow_is_exact_and_valid(self) -> None:
        self.assertTrue(MIGRATION.REPOSITORY_EXAMPLE.is_file())
        workflow = json.loads(
            MIGRATION.REPOSITORY_EXAMPLE.read_text(encoding="utf-8")
        )
        self.assertEqual(
            _canonical_sha256(workflow),
            MIGRATION.REPOSITORY_EXAMPLE_CANONICAL_SHA256,
        )
        MIGRATION.validate_workflow(workflow)
        marker = workflow["extra"][MIGRATION.MIGRATION_SCHEMA]
        self.assertEqual(marker["version"], MIGRATION.MIGRATION_VERSION)
        self.assertEqual(
            marker["source_workflow_file_sha256"],
            MIGRATION.SOURCE_V5_SHA256,
        )


@unittest.skipUnless(MIGRATION.SOURCE_V5.is_file(), "attached timed-lyrics V5 is absent")
class MiniMaxH3UploadSongV6WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_bytes = MIGRATION.SOURCE_V5.read_bytes()
        cls.source = json.loads(cls.source_bytes.decode("utf-8"))
        cls.migrated = MIGRATION.migrate_workflow(
            cls.source, source_file_sha256=MIGRATION.SOURCE_V5_SHA256
        )
        cls.marker = cls.migrated["extra"][MIGRATION.MIGRATION_SCHEMA]
        cls.loader = _node(cls.migrated, cls.marker["nodes"]["upload_loader"])
        cls.router = _node(cls.migrated, cls.marker["nodes"]["source_router"])

    def test_exact_attached_v5_is_immutable_source_evidence(self) -> None:
        self.assertEqual(_sha256(MIGRATION.SOURCE_V5), MIGRATION.SOURCE_V5_SHA256)
        before = copy.deepcopy(self.source)
        MIGRATION.migrate_workflow(
            self.source, source_file_sha256=MIGRATION.SOURCE_V5_SHA256
        )
        self.assertEqual(self.source, before)
        self.assertEqual(MIGRATION.SOURCE_V5.read_bytes(), self.source_bytes)
        self.assertTrue(self.marker["source_is_never_overwritten"])
        self.assertEqual(
            self.marker["source_workflow_file_sha256"],
            MIGRATION.SOURCE_V5_SHA256,
        )

    def test_revision_identity_subgraphs_and_user_widgets_are_preserved(self) -> None:
        self.assertEqual(self.migrated["id"], self.source["id"])
        self.assertEqual(self.migrated["revision"], self.source["revision"] + 1)
        self.assertEqual(self.migrated["definitions"], self.source["definitions"])
        for node_id in (
            MIGRATION.PRODUCTION_CONCEPT_NODE_ID,
            MIGRATION.ACE_REFERENCE_MODE_NODE_ID,
            MIGRATION.AUDIO_SELECTOR_NODE_ID,
            MIGRATION.TIMED.PERFORMANCE_MODE_NODE_ID,
            MIGRATION.PROJECT_CONTRACT_NODE_ID,
        ):
            self.assertEqual(
                _node(self.migrated, node_id).get("widgets_values"),
                _node(self.source, node_id).get("widgets_values"),
            )
        self.assertEqual(
            _node(self.migrated, MIGRATION.PRODUCTION_CONCEPT_NODE_ID)["widgets_values"],
            ["Source passthrough", 2],
        )
        self.assertEqual(
            _node(self.migrated, MIGRATION.ACE_REFERENCE_MODE_NODE_ID)["widgets_values"],
            ["Compose new"],
        )
        self.assertEqual(
            _node(self.migrated, MIGRATION.PROJECT_CONTRACT_NODE_ID)["widgets_values"][-1],
            15,
        )

    def test_loader_and_router_have_explicit_safe_defaults(self) -> None:
        self.assertEqual(self.loader["type"], MIGRATION.UPLOAD_LOADER_TYPE)
        self.assertEqual(self.loader["widgets_values"], [""])
        self.assertEqual(
            [item["name"] for item in self.loader["inputs"]], ["audio_file"]
        )
        self.assertEqual(
            [item["name"] for item in self.loader["outputs"]],
            ["audio", "duration_seconds", "waveform_sha256", "status", "ready"],
        )
        self.assertEqual(self.router["type"], MIGRATION.SOURCE_ROUTER_TYPE)
        self.assertEqual(
            self.router["widgets_values"], [MIGRATION.GENERATE_MODE, 0.0, ""]
        )
        self.assertIn("Upload song", self.router["title"])
        self.assertEqual(
            [item["name"] for item in self.router["outputs"]],
            [
                "candidate_1",
                "candidate_2",
                "candidate_3",
                "candidate_4",
                "effective_candidate_count",
                "effective_expected_bpm",
                "selected_lyrics",
                "selected_duration_seconds",
                "source_token",
                "status",
                "ready",
            ],
        )
        for item in self.router["inputs"][3:]:
            self.assertEqual(item.get("shape"), 7, item["name"])

    def test_generated_and_uploaded_sources_are_exactly_wired_to_router(self) -> None:
        generated = {
            "generated_candidate_count": (
                MIGRATION.PRODUCTION_CONCEPT_NODE_ID,
                "candidate_count",
            ),
            "generated_expected_bpm": (MIGRATION.EXPECTED_BPM_NODE_ID, "FLOAT"),
            "generated_lyrics": (MIGRATION.BLUEPRINT_ROUTER_NODE_ID, "lyrics"),
            "generated_duration_seconds": (
                MIGRATION.GENERATED_DURATION_NODE_ID,
                "FLOAT",
            ),
            "generated_candidate_1": (MIGRATION.ACE_CANDIDATES_NODE_ID, "AUDIO"),
            "generated_candidate_2": (MIGRATION.ACE_CANDIDATES_NODE_ID, "AUDIO_1"),
            "generated_candidate_3": (MIGRATION.ACE_CANDIDATES_NODE_ID, "AUDIO_2"),
            "generated_candidate_4": (MIGRATION.ACE_CANDIDATES_NODE_ID, "AUDIO_3"),
        }
        for input_name, expected in generated.items():
            origin, output = _origin(self.migrated, self.router, input_name)
            self.assertEqual((origin["id"], output), expected)
        uploaded = {
            "uploaded_audio": "audio",
            "uploaded_duration_seconds": "duration_seconds",
            "uploaded_waveform_sha256": "waveform_sha256",
            "uploaded_status": "status",
            "uploaded_ready": "ready",
        }
        for input_name, output_name in uploaded.items():
            origin, output = _origin(self.migrated, self.router, input_name)
            self.assertEqual((origin["id"], output), (self.loader["id"], output_name))

    def test_existing_selector_remains_single_qc_and_hash_authority(self) -> None:
        selector = _node(self.migrated, MIGRATION.AUDIO_SELECTOR_NODE_ID)
        expected = {
            "candidate_1": "candidate_1",
            "candidate_2": "candidate_2",
            "candidate_3": "candidate_3",
            "candidate_4": "candidate_4",
            "candidate_count": "effective_candidate_count",
            "expected_bpm": "effective_expected_bpm",
            "source_policy": "source_token",
        }
        for input_name, output_name in expected.items():
            origin, output = _origin(self.migrated, selector, input_name)
            self.assertEqual((origin["id"], output), (self.router["id"], output_name))
        # Existing output ordering is contract-sensitive and must not be rewritten.
        self.assertEqual(
            [item["name"] for item in selector["outputs"]],
            [
                "selected_audio",
                "suggested_start_seconds",
                "waveform_sha256",
                "director_report_json",
                "audit_report_json",
                "status",
                "ready",
            ],
        )
        project = _node(self.migrated, MIGRATION.PROJECT_CONTRACT_NODE_ID)
        origin, output = _origin(self.migrated, project, "master_audio_sha256")
        self.assertEqual(
            (origin["id"], output),
            (MIGRATION.AUDIO_SELECTOR_NODE_ID, "waveform_sha256"),
        )

    def test_selected_lyrics_and_exact_duration_replace_generated_metadata_only_at_seam(self) -> None:
        for target_id in (
            MIGRATION.LYRIC_WINDOW_NODE_ID,
            MIGRATION.PROJECT_CONTRACT_NODE_ID,
            MIGRATION.TIMED_ANALYZER_NODE_ID,
        ):
            origin, output = _origin(
                self.migrated, _node(self.migrated, target_id), "lyrics"
            )
            self.assertEqual(
                (origin["id"], output), (self.router["id"], "selected_lyrics")
            )
        for target_id in (
            MIGRATION.LYRIC_WINDOW_NODE_ID,
            MIGRATION.TIMED_ANALYZER_NODE_ID,
        ):
            origin, output = _origin(
                self.migrated,
                _node(self.migrated, target_id),
                "song_duration_seconds",
            )
            self.assertEqual(
                (origin["id"], output),
                (self.router["id"], "selected_duration_seconds"),
            )
        planner = _node(self.migrated, MIGRATION.H3_PLANNER_NODE_ID)
        origin, output = _origin(self.migrated, planner, "lyrics")
        self.assertEqual(
            (origin["id"], output),
            (MIGRATION.LYRIC_WINDOW_NODE_ID, "selected_lyrics"),
        )

    def test_help_and_inherited_policy_markers_are_normalized_additively(self) -> None:
        self.assertEqual(self.marker["policy"], MIGRATION.UPLOAD_SOURCE_POLICY_MARKER)
        self.assertIn(
            "not waveform passthrough",
            self.marker["policy"]["source_passthrough_semantics"],
        )
        self.assertEqual(
            self.marker["policy"]["upload_semantics"]["selector_policy"],
            "source_locked_uploaded_song",
        )
        self.assertIn(
            "tempo interpretation",
            self.marker["policy"]["upload_semantics"]["advisory_qc"],
        )
        extra = self.migrated["extra"]
        music = extra[MIGRATION.PRODUCTION.MIGRATION_SCHEMA]
        self.assertEqual(music["version"], MIGRATION.PRODUCTION.MIGRATION_VERSION)
        self.assertEqual(
            music["selector_policy"], MIGRATION.PRODUCTION.SELECTOR_POLICY_MARKER
        )
        expansion = extra[MIGRATION.EXPANSION.MIGRATION_SCHEMA]
        self.assertEqual(
            expansion["generation_lane_boundary_policy"],
            MIGRATION.EXPANSION.BOUNDARY_POLICY_MARKER,
        )
        timed = extra[MIGRATION.TIMED.MIGRATION_SCHEMA]
        self.assertEqual(
            timed["audio_selector_policy"],
            MIGRATION.TIMED.AUDIO_SELECTOR_POLICY_MARKER,
        )
        self.assertEqual(
            timed["generation_lane_boundary_policy"],
            MIGRATION.TIMED.BOUNDARY_POLICY_MARKER,
        )
        start = _node(self.migrated, MIGRATION.START_NODE_ID)["widgets_values"][0]
        self.assertIn(MIGRATION.UPLOAD_START_COPY, start)
        self.assertIn(MIGRATION.TIMED.CURRENT_AUDIO_QC_COPY, start)
        self.assertIn(MIGRATION.TIMED.CURRENT_BOUNDARY_POLICY_COPY, start)
        self.assertIn("Source passthrough", start)
        self.assertIn(MIGRATION.H3_DANCE_HELP, start)
        self.assertNotIn(MIGRATION.LEGACY_DANCE_HELP, start)
        guidance = _node(
            self.migrated, MIGRATION.AUDIO_GUIDANCE_NODE_ID
        )["widgets_values"][0]
        self.assertEqual(guidance, MIGRATION.H3_AUDIO_GUIDANCE)
        self.assertIn("expressive, physically coherent H3 camera choreography", guidance)
        self.assertIn("orbiting", guidance)
        self.assertIn("pronounced parallax", guidance)
        self.assertNotIn("restrained camera accents", guidance)
        self.assertEqual(
            self.marker["h3_camera_policy"],
            {
                "revision": 1,
                "ltx_camera_capability_applies": False,
                "dynamic_paths_allowed": [
                    "orbit",
                    "swirl",
                    "sweep_or_whip",
                    "pronounced_parallax",
                    "coherent_compound_path",
                ],
                "creative_energy_source": "director_creativity_mode_and_strength",
            },
        )

    def test_added_nodes_do_not_overlap_existing_nodes_and_group_contains_them(self) -> None:
        def rectangle(node: dict) -> tuple[float, float, float, float]:
            x, y = map(float, node["pos"])
            width, height = map(float, node["size"])
            return x, y, x + width, y + height

        def overlaps(left: dict, right: dict) -> bool:
            a = rectangle(left)
            b = rectangle(right)
            return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

        self.assertFalse(overlaps(self.loader, self.router))
        for added in (self.loader, self.router):
            for existing in self.source["nodes"]:
                self.assertFalse(
                    overlaps(added, existing),
                    f"added node {added['id']} overlaps source node {existing['id']}",
                )
        groups = [group for group in self.migrated["groups"] if int(group["id"]) == 35]
        self.assertEqual(len(groups), 1)
        gx, gy, gw, gh = map(float, groups[0]["bounding"])
        for added in (self.loader, self.router):
            left, top, right, bottom = rectangle(added)
            self.assertGreaterEqual(left, gx)
            self.assertGreaterEqual(top, gy)
            self.assertLessEqual(right, gx + gw)
            self.assertLessEqual(bottom, gy + gh)

    def test_migration_is_idempotent_without_duplicate_nodes_or_revision(self) -> None:
        repeated = MIGRATION.migrate_workflow(
            self.migrated, source_file_sha256="already-owned-v6"
        )
        self.assertEqual(repeated, self.migrated)
        self.assertEqual(
            len(
                [
                    node
                    for node in repeated["nodes"]
                    if node["type"] == MIGRATION.UPLOAD_LOADER_TYPE
                ]
            ),
            1,
        )
        self.assertEqual(
            len(
                [
                    node
                    for node in repeated["nodes"]
                    if node["type"] == MIGRATION.SOURCE_ROUTER_TYPE
                ]
            ),
            1,
        )

    def test_validator_fails_closed_on_route_policy_or_user_control_drift(self) -> None:
        broken = copy.deepcopy(self.migrated)
        router = _node(broken, self.router["id"])
        router["widgets_values"][0] = "Unsupported source"
        with self.assertRaises(MIGRATION.WorkflowError):
            MIGRATION.validate_workflow(broken, source=self.source)

        # A user's saved Upload choice remains a valid, idempotent V6 artifact.
        upload_saved = copy.deepcopy(self.migrated)
        router = _node(upload_saved, self.router["id"])
        router["widgets_values"] = [MIGRATION.UPLOAD_MODE, 123.0, "Exact sung lyrics"]
        MIGRATION.validate_workflow(upload_saved)
        self.assertEqual(
            MIGRATION.migrate_workflow(
                upload_saved, source_file_sha256="already-owned-v6"
            ),
            upload_saved,
        )

        broken = copy.deepcopy(self.migrated)
        marker = broken["extra"][MIGRATION.MIGRATION_SCHEMA]
        marker["policy"]["revision"] = 999
        with self.assertRaises(MIGRATION.WorkflowError):
            MIGRATION.validate_workflow(broken, source=self.source)

        broken = copy.deepcopy(self.migrated)
        selector = _node(broken, MIGRATION.AUDIO_SELECTOR_NODE_ID)
        selector["widgets_values"][4] = 0.1
        with self.assertRaises(MIGRATION.WorkflowError):
            MIGRATION.validate_workflow(broken, source=self.source)

    def test_writer_is_deterministic_and_replaces_only_owned_v6(self) -> None:
        _workflow, payload, _source_hash = MIGRATION.build_artifact()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / MIGRATION.V6_FILENAME
            self.assertEqual(MIGRATION._write_new_or_verify(target, payload), "created")
            self.assertEqual(MIGRATION._write_new_or_verify(target, payload), "verified")
            owned = json.loads(payload.decode("utf-8"))
            owned["extra"]["workflow_note"] += " changed"
            target.write_text(json.dumps(owned), encoding="utf-8")
            with self.assertRaises(MIGRATION.WorkflowError):
                MIGRATION._write_new_or_verify(target, payload)
            self.assertEqual(
                MIGRATION._write_new_or_verify(target, payload, replace_owned=True),
                "replaced",
            )
            target.write_text("unrecognized", encoding="utf-8")
            with self.assertRaises(MIGRATION.WorkflowError):
                MIGRATION._write_new_or_verify(target, payload, replace_owned=True)

    def test_wrong_source_hash_is_rejected(self) -> None:
        with self.assertRaises(MIGRATION.WorkflowError):
            MIGRATION.migrate_workflow(self.source, source_file_sha256="0" * 64)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
