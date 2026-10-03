from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "migrate_minimax_h3_timed_lyrics_v5.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_timed_lyrics_v5", TOOL_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot load migration tool: {TOOL_PATH}")
MIGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIGRATION)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _node(workflow: dict, node_id: int) -> dict:
    matches = [item for item in workflow["nodes"] if int(item["id"]) == int(node_id)]
    if len(matches) != 1:
        raise AssertionError(f"Expected one node {node_id}; found {len(matches)}")
    return matches[0]


def _origin(workflow: dict, target: dict, input_name: str) -> tuple[dict, str]:
    return MIGRATION._origin(workflow, target, input_name)


@unittest.skipUnless(MIGRATION.SOURCE_V4.is_file(), "verified V4 workflow is absent")
class MiniMaxH3TimedLyricsV5WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_bytes = MIGRATION.SOURCE_V4.read_bytes()
        cls.source = json.loads(cls.source_bytes.decode("utf-8"))
        cls.migrated = MIGRATION.migrate_workflow(
            cls.source,
            source_file_sha256=MIGRATION.SOURCE_V4_SHA256,
        )
        cls.marker = cls.migrated["extra"][MIGRATION.MIGRATION_SCHEMA]

    def test_exact_v4_is_immutable_source_evidence(self) -> None:
        self.assertEqual(_sha256(MIGRATION.SOURCE_V4), MIGRATION.SOURCE_V4_SHA256)
        before = copy.deepcopy(self.source)
        MIGRATION.migrate_workflow(
            self.source,
            source_file_sha256=MIGRATION.SOURCE_V4_SHA256,
        )
        self.assertEqual(self.source, before)
        self.assertEqual(MIGRATION.SOURCE_V4.read_bytes(), self.source_bytes)
        self.assertTrue(self.marker["source_is_never_overwritten"])

    def test_revision_id_and_h3_subgraph_are_preserved(self) -> None:
        self.assertEqual(self.migrated["id"], self.source["id"])
        self.assertEqual(self.migrated["revision"], self.source["revision"] + 1)
        self.assertEqual(self.migrated["definitions"], self.source["definitions"])
        self.assertEqual(
            self.marker["source_workflow_file_sha256"],
            MIGRATION.SOURCE_V4_SHA256,
        )

    def test_one_authoritative_control_defaults_to_natural(self) -> None:
        performance = _node(self.migrated, MIGRATION.PERFORMANCE_MODE_NODE_ID)
        legacy = _node(self.migrated, MIGRATION.LEGACY_LYRIC_WINDOW_NODE_ID)
        director = _node(self.migrated, MIGRATION.DIRECTOR_NODE_ID)
        self.assertEqual(performance["widgets_values"][0], MIGRATION.NATURAL_MODE)
        self.assertIn("one authoritative control", performance["title"])
        self.assertEqual(legacy["widgets_values"][0], MIGRATION.NATURAL_MODE)
        self.assertTrue(legacy["flags"]["collapsed"])
        self.assertIn("mode comes only from H3 PERFORMANCE MODE", legacy["title"])
        start_text = _node(self.migrated, MIGRATION.START_NODE_ID)["widgets_values"][0]
        self.assertIn("Natural / audio-led sync", start_text)
        self.assertIn("Only verified, hash-locked timing events", start_text)
        self.assertIn("safely becomes Natural/audio-led behavior", start_text)
        self.assertIn("expressive and physically coherent H3 camera work", start_text)
        self.assertNotIn("camera-safe scene", start_text)
        self.assertEqual(director["widgets_values"][-1], "reuse")
        self.assertEqual(
            self.marker["director_cache_mode"],
            {
                "node": MIGRATION.DIRECTOR_NODE_ID,
                "default": "reuse",
                "refresh_is_one_run_only": True,
            },
        )
        self.assertEqual(
            self.marker["audio_selector_policy"],
            MIGRATION.AUDIO_SELECTOR_POLICY_MARKER,
        )
        self.assertEqual(
            self.migrated["extra"]["diffusiongemma.music_video_production"]
            ["selector_policy"],
            MIGRATION.AUDIO_SELECTOR_POLICY_MARKER,
        )
        self.assertEqual(
            self.marker["generation_lane_boundary_policy"],
            MIGRATION.BOUNDARY_POLICY_MARKER,
        )
        self.assertEqual(
            self.migrated["extra"]["diffusiongemma.minimax_music_video_expansion"]
            ["generation_lane_boundary_policy"],
            MIGRATION.BOUNDARY_POLICY_MARKER,
        )
        self.assertIn(MIGRATION.CURRENT_AUDIO_QC_COPY, start_text)
        self.assertNotIn(MIGRATION.LEGACY_AUDIO_QC_COPY, start_text)
        self.assertIn(MIGRATION.CURRENT_BOUNDARY_POLICY_COPY, start_text)

    def test_lazy_separator_whisper_and_analyzer_are_exactly_wired(self) -> None:
        nodes = self.marker["nodes"]
        separator = _node(self.migrated, nodes["separator"])
        whisper = _node(self.migrated, nodes["whisper_loader"])
        analyzer = _node(self.migrated, nodes["analyzer"])
        status_preview = _node(self.migrated, nodes["status_preview"])
        alignment_preview = _node(self.migrated, nodes["alignment_preview"])
        planner = _node(self.migrated, MIGRATION.PLANNER_NODE_ID)

        self.assertEqual(separator["type"], MIGRATION.SEPARATOR_TYPE)
        self.assertEqual(separator["widgets_values"], [10.0, 0.5, "half_sine"])
        self.assertEqual(whisper["type"], MIGRATION.WHISPER_LOADER_TYPE)
        self.assertEqual(whisper["widgets_values"], ["large-v3-turbo", False])
        self.assertEqual(analyzer["type"], MIGRATION.ANALYZER_TYPE)
        self.assertEqual(
            analyzer["widgets_values"], [90.0, 0.0, 25.0, "en", 0.55]
        )
        self.assertEqual(
            [item["name"] for item in analyzer["inputs"]],
            [
                "final_audio",
                "performance_mode",
                "lyrics",
                "master_audio_sha256",
                "song_duration_seconds",
                "excerpt_start_seconds",
                "excerpt_duration_seconds",
                "vocal_stem",
                "whisper_pipeline",
            ],
        )
        self.assertEqual(analyzer["inputs"][-2]["shape"], 7)
        self.assertEqual(analyzer["inputs"][-1]["shape"], 7)
        self.assertEqual(planner["inputs"][-1]["name"], "timed_lyrics_report_json")
        self.assertEqual(planner["inputs"][-1]["shape"], 7)
        self.assertEqual(
            planner["title"],
            "AUDIO-AWARE H3 MULTI-LANE PLAN — evidence-preferred / deterministic fallback",
        )

        def rectangle(node: dict) -> tuple[float, float, float, float]:
            left, top = map(float, node["pos"])
            width, height = map(float, node["size"])
            return left, top, left + width, top + height

        def overlaps(left: dict, right: dict) -> bool:
            left_rect = rectangle(left)
            right_rect = rectangle(right)
            return not (
                left_rect[2] <= right_rect[0]
                or right_rect[2] <= left_rect[0]
                or left_rect[3] <= right_rect[1]
                or right_rect[3] <= left_rect[1]
            )

        added_nodes = [
            separator,
            whisper,
            analyzer,
            status_preview,
            alignment_preview,
        ]
        for left_index, left in enumerate(added_nodes):
            for right in added_nodes[left_index + 1 :]:
                self.assertFalse(
                    overlaps(left, right),
                    f"added node {left['id']} overlaps added node {right['id']}",
                )
            for existing in self.source["nodes"]:
                self.assertFalse(
                    overlaps(left, existing),
                    f"added node {left['id']} overlaps preexisting node {existing['id']}",
                )

        planning_groups = [
            group
            for group in self.migrated.get("groups", [])
            if str(group.get("title", "")).startswith(
                "3. Verified H3 project + lazy timed lyrics"
            )
        ]
        self.assertEqual(len(planning_groups), 1)
        group_left, group_top, group_width, group_height = map(
            float, planning_groups[0]["bounding"]
        )
        group_right = group_left + group_width
        group_bottom = group_top + group_height
        for node in added_nodes:
            left, top, right, bottom = rectangle(node)
            self.assertGreaterEqual(left, group_left)
            self.assertGreaterEqual(top, group_top)
            self.assertLessEqual(right, group_right)
            self.assertLessEqual(bottom, group_bottom)

        expected = {
            "final_audio": (MIGRATION.AUDIO_GUIDE_NODE_ID, "final_audio"),
            "performance_mode": (
                MIGRATION.PERFORMANCE_MODE_NODE_ID,
                "performance_mode",
            ),
            "lyrics": (MIGRATION.LYRICS_NODE_ID, "lyrics"),
            "master_audio_sha256": (
                MIGRATION.AUDIO_SELECTOR_NODE_ID,
                "waveform_sha256",
            ),
            "song_duration_seconds": (MIGRATION.SONG_DURATION_NODE_ID, "FLOAT"),
            "excerpt_start_seconds": (
                MIGRATION.AUDIO_SELECTOR_NODE_ID,
                "suggested_start_seconds",
            ),
            "excerpt_duration_seconds": (MIGRATION.VIDEO_DURATION_NODE_ID, "FLOAT"),
            "vocal_stem": (separator["id"], "vocals"),
            "whisper_pipeline": (whisper["id"], "pipeline"),
        }
        for input_name, (origin_id, output_name) in expected.items():
            origin, output = _origin(self.migrated, analyzer, input_name)
            self.assertEqual((origin["id"], output), (origin_id, output_name))
        separator_origin, separator_output = _origin(
            self.migrated, separator, "audio"
        )
        self.assertEqual(
            (separator_origin["id"], separator_output),
            (MIGRATION.AUDIO_GUIDE_NODE_ID, "final_audio"),
        )
        timing_origin, timing_output = _origin(
            self.migrated, planner, "timed_lyrics_report_json"
        )
        self.assertEqual(
            (timing_origin["id"], timing_output),
            (analyzer["id"], "timed_lyrics_report_json"),
        )

    def test_marker_states_fail_safe_contract_without_claiming_forced_timing(self) -> None:
        self.assertEqual(
            self.marker["performance_modes"], list(MIGRATION.PERFORMANCE_MODES)
        )
        self.assertEqual(self.marker["default_performance_mode"], MIGRATION.NATURAL_MODE)
        self.assertTrue(self.marker["analysis"]["lazy_for_non_lyrics_modes"])
        self.assertEqual(
            self.marker["timing_report"]["schema"],
            "diffusiongemma.timed_lyrics_report",
        )
        self.assertTrue(self.marker["timing_report"]["lyrics_are_not_a_timing_schedule"])
        fallback = self.marker["fallback_policy"]
        self.assertEqual(fallback["mode"], MIGRATION.NATURAL_MODE)
        self.assertFalse(fallback["generation_blocked_by_analysis_failure"])
        self.assertEqual(fallback["instrumental_intervals"], "explicit non-vocal behavior")

    def test_validator_fails_closed_on_timing_or_model_drift(self) -> None:
        broken = copy.deepcopy(self.migrated)
        whisper = _node(broken, self.marker["nodes"]["whisper_loader"])
        whisper["widgets_values"][0] = "tiny"
        with self.assertRaises(MIGRATION.WorkflowError):
            MIGRATION.validate_workflow(broken, source=self.source)

        broken = copy.deepcopy(self.migrated)
        planner = _node(broken, MIGRATION.PLANNER_NODE_ID)
        planner["inputs"][-1]["shape"] = 0
        with self.assertRaises(MIGRATION.WorkflowError):
            MIGRATION.validate_workflow(broken, source=self.source)

        broken = copy.deepcopy(self.migrated)
        director = _node(broken, MIGRATION.DIRECTOR_NODE_ID)
        director["widgets_values"][-1] = "refresh"
        with self.assertRaises(MIGRATION.WorkflowError):
            MIGRATION.validate_workflow(broken, source=self.source)

    def test_writer_creates_new_artifact_and_refuses_different_overwrite(self) -> None:
        _workflow, payload, _source_hash = MIGRATION.build_artifact()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / MIGRATION.V5_FILENAME
            self.assertEqual(MIGRATION._write_new_or_verify(target, payload), "created")
            self.assertEqual(MIGRATION._write_new_or_verify(target, payload), "verified")
            target.write_text("different", encoding="utf-8")
            with self.assertRaises(MIGRATION.WorkflowError):
                MIGRATION._write_new_or_verify(target, payload)
            with self.assertRaises(MIGRATION.WorkflowError):
                MIGRATION._write_new_or_verify(target, payload, replace_owned=True)

    def test_installed_models_exist_at_the_exact_runtime_paths(self) -> None:
        whisper = (
            Path(r"C:\ComfyUI\app\models\whisper") / "whisper-large-v3-turbo"
        )
        demucs = Path(
            r"C:\Users\danrh\.cache\torch\hub\torchaudio\models\hdemucs_high_trained.pt"
        )
        self.assertTrue((whisper / "model.safetensors").is_file())
        self.assertEqual(
            (whisper / "model.safetensors").stat().st_size,
            MIGRATION.WHISPER_MODEL_BYTES,
        )
        self.assertEqual(
            _sha256(whisper / "model.safetensors"),
            MIGRATION.WHISPER_MODEL_SHA256,
        )
        self.assertTrue(demucs.is_file())
        self.assertEqual(demucs.stat().st_size, MIGRATION.DEMUCS_CHECKPOINT_BYTES)
        self.assertEqual(_sha256(demucs), MIGRATION.DEMUCS_CHECKPOINT_SHA256)
        self.assertEqual(
            self.marker["analysis"]["whisper_model_file"]["sha256"],
            MIGRATION.WHISPER_MODEL_SHA256,
        )
        self.assertEqual(
            self.marker["analysis"]["demucs_checkpoint"]["sha256"],
            MIGRATION.DEMUCS_CHECKPOINT_SHA256,
        )


_VERIFY_DEPLOYED_V5_ARTIFACTS = os.environ.get(
    "DG_VERIFY_DEPLOYED_V5_ARTIFACTS", ""
).strip().lower() in {"1", "true", "yes", "on"}


@unittest.skipUnless(
    _VERIFY_DEPLOYED_V5_ARTIFACTS
    and MIGRATION.DEFAULT_DESKTOP_OUTPUT.is_file()
    and MIGRATION.DEFAULT_USER_OUTPUT.is_file(),
    "set DG_VERIFY_DEPLOYED_V5_ARTIFACTS=1 to audit mutable external V5 copies",
)
class MiniMaxH3TimedLyricsV5ArtifactTests(unittest.TestCase):
    def test_desktop_and_comfyui_copies_are_exact_and_valid(self) -> None:
        desktop = MIGRATION.DEFAULT_DESKTOP_OUTPUT.read_bytes()
        user = MIGRATION.DEFAULT_USER_OUTPUT.read_bytes()
        self.assertEqual(desktop, user)
        workflow = json.loads(desktop.decode("utf-8"))
        source = json.loads(MIGRATION.SOURCE_V4.read_text(encoding="utf-8"))
        MIGRATION.validate_workflow(workflow, source=source)


if __name__ == "__main__":
    unittest.main()
