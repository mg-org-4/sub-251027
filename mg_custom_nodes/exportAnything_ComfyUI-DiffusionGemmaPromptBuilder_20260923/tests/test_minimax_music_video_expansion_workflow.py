from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATH = REPO_ROOT / "tools" / "migrate_minimax_music_video_expansion.py"
SPEC = importlib.util.spec_from_file_location("minimax_music_video_migration", MIGRATION_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot load migration module: {MIGRATION_PATH}")
MIGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIGRATION)

WORKFLOW = Path(
    r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized.json"
)
BACKUP = WORKFLOW.with_suffix(WORKFLOW.suffix + MIGRATION.BACKUP_SUFFIX)
EXPECTED_ORIGINAL_SHA256 = (
    "4cb0d937140822ee7dd82fcda41385d645a90de34101b88c55769d7ddb7009fe"
)
LIVE_STABILITY_CAPTURE = Path(
    r"C:\Users\danrh\Desktop\14_minimax_ref2va+AUDIO_IN_PROCESS_SYNC_organized_BASELINE_FAST_H3_LIVE_UNSAVED_PRE_REPAIR_20260822_0442.json"
)
LIVE_STABILITY_CAPTURE_SHA256 = (
    "3d647ff4d1d147e64006902121b5f8ca0f03bcc2fc318a348cc3f9c56d2cc9a1"
)


def _fixture_path() -> Path:
    if BACKUP.is_file():
        return BACKUP
    return WORKFLOW


def _node(workflow: dict, node_id: int) -> dict:
    matches = [node for node in workflow["nodes"] if int(node["id"]) == node_id]
    if len(matches) != 1:
        raise AssertionError(f"Expected node {node_id}; found {len(matches)}")
    return matches[0]


def _one(workflow: dict, node_type: str) -> dict:
    matches = [node for node in workflow["nodes"] if node.get("type") == node_type]
    if len(matches) != 1:
        raise AssertionError(f"Expected one {node_type}; found {len(matches)}")
    return matches[0]


def _output(node: dict, name: str) -> dict:
    return next(item for item in node["outputs"] if item.get("name") == name)


def _input(node: dict, name: str) -> dict:
    return next(item for item in node["inputs"] if item.get("name") == name)


def _origin(workflow: dict, node: dict, input_name: str) -> tuple[dict, str]:
    link_id = _input(node, input_name).get("link")
    if link_id is None:
        raise AssertionError(f"{node['id']}.{input_name} is unconnected")
    link = next(link for link in workflow["links"] if int(link[0]) == int(link_id))
    origin = _node(workflow, int(link[1]))
    return origin, origin["outputs"][int(link[2])]["name"]


def _as_v1(workflow: dict) -> dict:
    legacy = copy.deepcopy(workflow)
    marker = legacy["extra"][MIGRATION.MIGRATION_SCHEMA]
    marker["version"] = MIGRATION.LEGACY_MIGRATION_VERSION
    marker.pop("max_h3_generation_lanes", None)
    marker.pop("max_h3_generation_lane_seconds", None)
    marker.pop("count_semantics", None)
    for key in (
        "generation_lane_planner",
        "generation_lane_seed_fanout",
        "generation_lane_assembler",
    ):
        marker["nodes"].pop(key, None)
    return legacy


def _add_legacy_lane_count_reroute(
    workflow: dict,
    *,
    connected: bool,
    title: str | None = None,
    mode: int = 4,
) -> tuple[int, int, int | None]:
    graph = MIGRATION.MainGraph(workflow, MIGRATION.IdAllocator(workflow))
    project = graph.one(MIGRATION.PROJECT_NODE_TYPE)
    target = graph.node(673)
    reroute_id = graph.ids.node()
    reroute = {
        "id": reroute_id,
        "type": "Reroute",
        "pos": [0.0, 0.0],
        "size": [75.0, 26.0],
        "flags": {},
        "order": 0,
        "mode": mode,
        "inputs": [
            {
                "name": "",
                "type": "*",
                "widget": {"name": "value"},
                "link": None,
            }
        ],
        "outputs": [{"name": "", "type": "INT", "links": []}],
        "properties": {"horizontal": False, "showOutputText": False},
        "widgets_values": None,
    }
    if title is not None:
        reroute["title"] = title
    graph.add(reroute)
    input_link = graph.connect(
        project,
        "recommended_h3_shot_count",
        reroute,
        "",
        "INT",
    )
    output_link = None
    if connected:
        output_link = graph.connect(
            reroute,
            "",
            target,
            "shot_count_override",
            "INT",
        )
    workflow["last_node_id"] = graph.ids.last_node
    workflow["last_link_id"] = graph.ids.last_link
    return reroute_id, input_link, output_link


@unittest.skipUnless(WORKFLOW.is_file() or BACKUP.is_file(), "attached MiniMax workflow is absent")
class MiniMaxMusicVideoExpansionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_path = _fixture_path()
        cls.source_bytes = cls.fixture_path.read_bytes()
        cls.source = json.loads(cls.source_bytes.decode("utf-8"))
        # Always run through the public entry point: a v1 expansion is upgraded
        # in memory, while a v2 expansion remains byte-for-byte idempotent.
        cls.migrated = MIGRATION.migrate_workflow(
            cls.source,
            source_file_sha256=hashlib.sha256(cls.source_bytes).hexdigest(),
        )

    def test_exact_attached_source_hash_is_known_before_first_write(self) -> None:
        if MIGRATION.MIGRATION_SCHEMA not in self.source.get("extra", {}):
            self.assertEqual(
                hashlib.sha256(self.source_bytes).hexdigest(),
                EXPECTED_ORIGINAL_SHA256,
            )

    def test_migration_validates_and_is_idempotent(self) -> None:
        MIGRATION.validate_workflow(self.migrated)
        self.assertEqual(
            MIGRATION.migrate_workflow(self.migrated),
            self.migrated,
        )

    def test_natural_mode_is_appended_default_without_reindexing_saved_modes(self) -> None:
        self.assertEqual(
            MIGRATION.PERFORMANCE_MODES,
            (
                "Dance / music sync",
                "Lyrics + lip sync",
                "Natural / audio-led sync",
            ),
        )
        self.assertEqual(
            MIGRATION.DEFAULT_PERFORMANCE_MODE, "Natural / audio-led sync"
        )
        marker = self.migrated["extra"][MIGRATION.MIGRATION_SCHEMA][
            "performance_mode_contract"
        ]
        self.assertEqual(marker, MIGRATION.PERFORMANCE_MODE_MARKER)
        note = _node(self.migrated, 116)["widgets_values"][0]
        self.assertIn("defaults to **Natural / audio-led sync**", note)
        self.assertIn("lyrics are not a timing schedule", note)

    def test_boundary_policy_is_advisory_and_truthfully_documented(self) -> None:
        marker = self.migrated["extra"][MIGRATION.MIGRATION_SCHEMA]
        self.assertEqual(
            marker["generation_lane_boundary_policy"],
            MIGRATION.BOUNDARY_POLICY_MARKER,
        )
        note = _node(self.migrated, 116)["widgets_values"][0]
        self.assertIn("deterministic balanced seams", note)
        self.assertIn("Missing ideal cut evidence is advisory", note)
        planner = _one(self.migrated, MIGRATION.PLANNER_NODE_TYPE)
        self.assertEqual(
            planner["title"],
            "AUDIO-AWARE H3 MULTI-LANE PLAN — evidence-preferred / deterministic fallback",
        )
        self.assertIn(
            "deterministic balanced fallback",
            self.migrated["extra"]["workflow_note"],
        )

    def test_invalid_saved_mode_falls_back_to_natural_without_changing_valid_choices(self) -> None:
        edited = copy.deepcopy(self.migrated)
        control = _one(edited, MIGRATION.PERFORMANCE_MODE_NODE_TYPE)
        control["widgets_values"] = ["unsupported legacy value"]
        repaired = MIGRATION.migrate_workflow(edited)
        repaired_control = _one(repaired, MIGRATION.PERFORMANCE_MODE_NODE_TYPE)
        self.assertEqual(
            repaired_control["widgets_values"], ["Natural / audio-led sync"]
        )
        self.assertEqual(_node(repaired, 621)["widgets_values"][0], "Lyrics + lip sync")

    def test_user_controls_ace_subgraph_and_legacy_marker_are_preserved(self) -> None:
        migrated = self.migrated
        source_marker = self.source.get("extra", {}).get(MIGRATION.MIGRATION_SCHEMA)
        if isinstance(source_marker, dict) and int(source_marker.get("version", 0)) >= 2:
            self.skipTest("pre-v2 fixture is unavailable")
        self.assertEqual(
            migrated.get("definitions"),
            self.source.get("definitions"),
        )
        self.assertEqual(
            migrated["extra"]["diffusiongemma.music_video_production"],
            self.source["extra"]["diffusiongemma.music_video_production"],
        )
        removed = {511, 549, 655, 669, 672}
        changed_widget_contract = {116, 624, 665, 673}
        migrated_ids = {int(node["id"]) for node in migrated["nodes"]}
        for source_node in self.source["nodes"]:
            node_id = int(source_node["id"])
            if node_id in removed or node_id in changed_widget_contract:
                continue
            self.assertIn(node_id, migrated_ids)
            self.assertEqual(
                _node(migrated, node_id).get("widgets_values"),
                source_node.get("widgets_values"),
                f"widgets changed on original node {node_id}",
            )
        self.assertEqual(
            _node(migrated, 177)["widgets_values"],
            _node(self.source, 177)["widgets_values"],
        )
        self.assertEqual(_node(migrated, 178)["widgets_values"], [15])
        self.assertEqual(_node(migrated, 621)["widgets_values"][0], "Lyrics + lip sync")

    def test_song_context_and_h3_reference_context_are_separate(self) -> None:
        migrated = self.migrated
        song_context = _node(migrated, 183)
        h3_context = _one(migrated, MIGRATION.H3_CONTEXT_NODE_TYPE)
        self.assertEqual(song_context["type"], "DiffusionGemmaContextHub")
        origin, output = _origin(migrated, _node(migrated, 578), "gemma_context")
        self.assertEqual((origin["id"], output), (183, "gemma_context"))
        for target_id in (186, 187):
            origin, output = _origin(migrated, _node(migrated, target_id), "gemma_context")
            self.assertEqual((origin["id"], output), (h3_context["id"], "gemma_context"))
        manifest = h3_context["widgets_values"][0]
        self.assertIn("<Picture 1>:", manifest)
        self.assertIn("<Audio 1>:", manifest)
        self.assertEqual(h3_context["widgets_values"][-1], 0)

    def test_project_controls_aspect_and_generation_lanes_without_overriding_native_shots(self) -> None:
        migrated = self.migrated
        project = _one(migrated, MIGRATION.PROJECT_NODE_TYPE)
        splitter = _node(migrated, 187)
        target = _node(migrated, 673)
        aspect_origin, aspect_output = _origin(
            migrated, splitter, "resolution_aspect_ratio_override"
        )
        self.assertEqual((aspect_origin["id"], aspect_output), (project["id"], "master_aspect_ratio"))
        self.assertIsNone(_input(target, "shot_count_override").get("link"))
        self.assertEqual(
            _input(target, "shot_count_override").get("widget"),
            {"name": "shot_count_override"},
        )
        self.assertEqual(target["widgets_values"][-1], 0)
        marker = migrated["extra"][MIGRATION.MIGRATION_SCHEMA]
        self.assertEqual(marker["version"], 2)
        self.assertEqual(marker["master_video_duration_seconds"], 15.0)
        self.assertEqual(marker["master_aspect_ratio"], "9:16")
        self.assertEqual(marker["max_h3_generation_lanes"], 4)
        self.assertEqual(marker["max_h3_generation_lane_seconds"], 15.0)
        self.assertFalse(
            marker["count_semantics"][
                "project_master_to_target_shot_override_connected"
            ]
        )
        self.assertIn("native_[Shot_N]_blocks", marker["count_semantics"]["target_profile_shot_count"])

    def test_v1_upgrade_removes_only_the_conflated_link_and_preserves_user_state(self) -> None:
        legacy = copy.deepcopy(self.migrated)
        marker = legacy["extra"][MIGRATION.MIGRATION_SCHEMA]
        marker["version"] = MIGRATION.LEGACY_MIGRATION_VERSION
        marker.pop("max_h3_generation_lanes", None)
        marker.pop("max_h3_generation_lane_seconds", None)
        marker.pop("count_semantics", None)
        for key in (
            "generation_lane_planner",
            "generation_lane_seed_fanout",
            "generation_lane_assembler",
        ):
            marker["nodes"].pop(key, None)

        graph = MIGRATION.MainGraph(legacy, MIGRATION.IdAllocator(legacy))
        project = graph.one(MIGRATION.PROJECT_NODE_TYPE)
        target = graph.node(673)
        target["widgets_values"][4:6] = ["custom", 7]
        target["title"] = "USER CUSTOM H3 TARGET TITLE"
        note = graph.node(116)
        note["title"] = "USER CUSTOM START NOTE"
        note["widgets_values"] = ["Keep my production instructions unchanged."]
        unrelated = graph.node(177)
        unrelated["title"] = "USER CUSTOM CREATIVE BRIEF"
        unrelated["pos"] = [123.0, 456.0]
        override_link = graph.connect(
            project,
            "recommended_h3_shot_count",
            target,
            "shot_count_override",
            "INT",
        )
        legacy["last_link_id"] = graph.ids.last_link
        preserved_links = [
            copy.deepcopy(link)
            for link in legacy["links"]
            if int(link[0]) != int(override_link)
        ]
        preserved_definitions = copy.deepcopy(legacy.get("definitions"))
        preserved_node_ids = [int(node["id"]) for node in legacy["nodes"]]

        upgraded = MIGRATION.migrate_workflow(legacy)
        MIGRATION.validate_workflow(upgraded)
        upgraded_target = _node(upgraded, 673)
        self.assertIsNone(_input(upgraded_target, "shot_count_override").get("link"))
        self.assertEqual(upgraded_target["widgets_values"][4:6], ["custom", 7])
        self.assertEqual(upgraded_target["widgets_values"][-1], 0)
        self.assertEqual(upgraded_target["title"], "USER CUSTOM H3 TARGET TITLE")
        self.assertEqual(_node(upgraded, 116)["title"], "USER CUSTOM START NOTE")
        self.assertEqual(
            _node(upgraded, 116)["widgets_values"],
            ["Keep my production instructions unchanged."],
        )
        self.assertEqual(_node(upgraded, 177)["title"], "USER CUSTOM CREATIVE BRIEF")
        self.assertEqual(_node(upgraded, 177)["pos"], [123.0, 456.0])
        self.assertEqual(upgraded.get("definitions"), preserved_definitions)
        self.assertEqual([int(node["id"]) for node in upgraded["nodes"]], preserved_node_ids)
        self.assertEqual(upgraded["links"], preserved_links)
        self.assertEqual(
            upgraded["extra"][MIGRATION.MIGRATION_SCHEMA]["version"],
            MIGRATION.MIGRATION_VERSION,
        )
        self.assertEqual(
            upgraded["extra"][MIGRATION.MIGRATION_SCHEMA][
                "generation_lane_boundary_policy"
            ],
            MIGRATION.BOUNDARY_POLICY_MARKER,
        )

    def test_v1_upgrade_removes_exact_sole_legacy_reroute_chain(self) -> None:
        legacy = _as_v1(self.migrated)
        reroute_id, input_link, output_link = _add_legacy_lane_count_reroute(
            legacy, connected=True
        )
        self.assertIsNotNone(output_link)

        upgraded = MIGRATION.migrate_workflow(legacy)
        MIGRATION.validate_workflow(upgraded)
        self.assertNotIn(reroute_id, {int(node["id"]) for node in upgraded["nodes"]})
        self.assertFalse(
            {input_link, int(output_link)}
            & {int(link[0]) for link in upgraded["links"]}
        )
        self.assertIsNone(_input(_node(upgraded, 673), "shot_count_override").get("link"))

    def test_v1_upgrade_removes_exact_disconnected_legacy_reroute(self) -> None:
        legacy = _as_v1(self.migrated)
        reroute_id, input_link, output_link = _add_legacy_lane_count_reroute(
            legacy, connected=False
        )
        self.assertIsNone(output_link)

        upgraded = MIGRATION.migrate_workflow(legacy)
        MIGRATION.validate_workflow(upgraded)
        self.assertNotIn(reroute_id, {int(node["id"]) for node in upgraded["nodes"]})
        self.assertNotIn(input_link, {int(link[0]) for link in upgraded["links"]})
        self.assertIsNone(_input(_node(upgraded, 673), "shot_count_override").get("link"))

    def test_v1_upgrade_refuses_shared_legacy_reroute(self) -> None:
        legacy = _as_v1(self.migrated)
        reroute_id, _input_link, _output_link = _add_legacy_lane_count_reroute(
            legacy, connected=True
        )
        graph = MIGRATION.MainGraph(legacy, MIGRATION.IdAllocator(legacy))
        reroute = graph.node(reroute_id)
        sink_id = graph.ids.node()
        sink = {
            "id": sink_id,
            "type": "Reroute",
            "pos": [100.0, 0.0],
            "size": [75.0, 26.0],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [{"name": "", "type": "*", "link": None}],
            "outputs": [{"name": "", "type": "INT", "links": []}],
            "properties": {},
            "widgets_values": None,
        }
        graph.add(sink)
        graph.connect(reroute, "", sink, "", "INT")
        legacy["last_node_id"] = graph.ids.last_node
        legacy["last_link_id"] = graph.ids.last_link

        with self.assertRaisesRegex(MIGRATION.WorkflowError, "shared/user-authored"):
            MIGRATION.migrate_workflow(legacy)

    def test_v1_upgrade_refuses_customized_user_reroute(self) -> None:
        legacy = _as_v1(self.migrated)
        _add_legacy_lane_count_reroute(
            legacy,
            connected=True,
            title="USER SHOT OVERRIDE ROUTE",
        )
        with self.assertRaisesRegex(MIGRATION.WorkflowError, "customized/user-authored"):
            MIGRATION.migrate_workflow(legacy)

    def test_v2_start_note_warns_that_refresh_is_one_run_only(self) -> None:
        note = _node(self.migrated, 116)["widgets_values"][0]
        self.assertIn("Leave it on **reuse**", note)
        self.assertIn("**refresh** is a one-run diagnostic", note)

    def test_new_node_socket_order_matches_the_landed_contracts(self) -> None:
        project = _one(self.migrated, MIGRATION.PROJECT_NODE_TYPE)
        planner = _one(self.migrated, MIGRATION.PLANNER_NODE_TYPE)
        assembler = _one(self.migrated, MIGRATION.ASSEMBLER_NODE_TYPE)
        self.assertEqual(
            [item["name"] for item in project["inputs"]],
            [
                "creative_brief",
                "production_duration_seconds",
                "master_audio_sha256",
                "lyrics",
                "excerpt_start_seconds",
                "excerpt_duration_seconds",
            ],
        )
        self.assertEqual(
            [item["name"] for item in planner["inputs"]],
            [
                "measured_audio_report_json",
                "base_h3_prompt",
                "performance_mode",
                "excerpt_start_seconds",
                "excerpt_duration_seconds",
                "max_shot_seconds",
                "project_manifest_json",
                "lyrics",
            ],
        )
        self.assertEqual(
            [item["name"] for item in assembler["inputs"]],
            [
                "final_audio",
                "plan_json",
                "target_duration_seconds",
                "shot_1_images",
                "shot_2_images",
                "shot_3_images",
                "shot_4_images",
            ],
        )

    def test_performance_toggle_has_one_typed_mode_source_and_no_ltx_prompt_path(self) -> None:
        migrated = self.migrated
        guidance = _node(migrated, 624)["widgets_values"][0]
        self.assertIn("expressive, physically coherent H3 camera choreography", guidance)
        self.assertIn("orbiting", guidance)
        self.assertIn("pronounced parallax", guidance)
        self.assertNotIn("restrained camera accents", guidance)
        performance = _node(migrated, 621)
        performance_control = _one(migrated, MIGRATION.PERFORMANCE_MODE_NODE_TYPE)
        planner = _one(migrated, MIGRATION.PLANNER_NODE_TYPE)
        self.assertEqual(
            [item["name"] for item in performance["outputs"]],
            ["ltx_prompt", "selected_lyrics", "status", "performance_report_json", "performance_mode"],
        )
        self.assertEqual(_output(performance, "ltx_prompt").get("links"), [])
        mode_origin, mode_output = _origin(migrated, planner, "performance_mode")
        lyric_origin, lyric_output = _origin(migrated, planner, "lyrics")
        self.assertEqual(
            (mode_origin["id"], mode_output),
            (performance_control["id"], "performance_mode"),
        )
        self.assertEqual((lyric_origin["id"], lyric_output), (621, "selected_lyrics"))
        lyric_mode_origin, lyric_mode_output = _origin(
            migrated, performance, "performance_mode_override"
        )
        self.assertEqual(
            (lyric_mode_origin["id"], lyric_mode_output),
            (performance_control["id"], "performance_mode"),
        )
        guidance_origin, guidance_output = _origin(
            migrated, _node(migrated, 673), "audio_guidance"
        )
        self.assertEqual(
            (guidance_origin["id"], guidance_output),
            (performance_control["id"], "target_audio_guidance"),
        )
        self.assertEqual(performance_control["widgets_values"], ["Lyrics + lip sync"])
        self.assertEqual(_node(migrated, 673)["widgets_values"][6], "off")
        maximum_origin, maximum_output = _origin(
            migrated, planner, "max_shot_seconds"
        )
        project = _one(migrated, MIGRATION.PROJECT_NODE_TYPE)
        self.assertEqual(
            (maximum_origin["id"], maximum_output),
            (project["id"], "max_h3_shot_seconds"),
        )

    def test_four_lazy_h3_lanes_use_audio_1_and_retained_cut_continuity(self) -> None:
        migrated = self.migrated
        marker = migrated["extra"][MIGRATION.MIGRATION_SCHEMA]
        runtime = marker["runtime_lane_node_ids"]
        self.assertEqual(len(runtime["h3"]), 4)
        self.assertEqual(len(runtime["trims"]), 4)
        self.assertEqual(len(runtime["tail_math"]), 3)
        self.assertEqual(len(runtime["tails"]), 3)
        h3_nodes = [_node(migrated, node_id) for node_id in runtime["h3"]]
        for index, h3 in enumerate(h3_nodes, start=1):
            names = [item["name"] for item in h3["inputs"]]
            self.assertIn("ref_audios.ref_audio_0", names)
            self.assertNotIn("ref_audios.ref_audio_1", names)
            prompt_origin, prompt_output = _origin(migrated, h3, "prompt")
            self.assertEqual(prompt_origin["type"], MIGRATION.PLANNER_NODE_TYPE)
            self.assertEqual(prompt_output, f"shot_{index}_prompt")
        for math_id in runtime["tail_math"]:
            self.assertEqual(
                _node(migrated, math_id)["widgets_values"],
                ["max(0, round((a + b) * 24) - round(a * 24) - 1)"],
            )

    def test_final_mux_uses_assembler_and_pristine_ace_audio(self) -> None:
        migrated = self.migrated
        assembler = _one(migrated, MIGRATION.ASSEMBLER_NODE_TYPE)
        create_video = _node(migrated, 664)
        audio_origin, audio_output = _origin(migrated, assembler, "final_audio")
        self.assertEqual((audio_origin["type"], audio_output), ("DiffusionGemmaLTXAudioGuide", "final_audio"))
        images_origin, images_output = _origin(migrated, create_video, "images")
        mux_audio_origin, mux_audio_output = _origin(migrated, create_video, "audio")
        self.assertEqual((images_origin["id"], images_output), (assembler["id"], "assembled_images"))
        self.assertEqual((mux_audio_origin["id"], mux_audio_output), (assembler["id"], "final_audio"))
        self.assertFalse(any(node["type"] == "VAEDecodeAudio" for node in migrated["nodes"]))

    def test_link_and_security_corruption_fail_closed(self) -> None:
        broken = copy.deepcopy(self.migrated)
        assembler = _one(broken, MIGRATION.ASSEMBLER_NODE_TYPE)
        _input(assembler, "final_audio")["link"] = None
        with self.assertRaises(MIGRATION.WorkflowError):
            MIGRATION.validate_workflow(broken)

        wrong_type = copy.deepcopy(self.migrated)
        wrong_type["links"][0][5] = "DELIBERATELY_WRONG_TYPE"
        with self.assertRaises(MIGRATION.WorkflowError):
            MIGRATION.validate_workflow(wrong_type)

        secret = copy.deepcopy(self.source)
        secret.setdefault("extra", {})["api_key"] = "do-not-write-this"
        with self.assertRaises(MIGRATION.WorkflowError):
            MIGRATION.migrate_workflow(secret)

    def test_atomic_write_makes_one_verified_recovery_backup(self) -> None:
        source_marker = self.source.get("extra", {}).get(MIGRATION.MIGRATION_SCHEMA)
        if isinstance(source_marker, dict) and int(source_marker.get("version", 0)) >= 2:
            self.skipTest("pre-v2 fixture is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / WORKFLOW.name
            shutil.copy2(self.fixture_path, target)
            source_bytes = target.read_bytes()
            migrated, source_hash, backup = MIGRATION.migrate_file(
                target,
                write=True,
                expected_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            )
            self.assertIsNotNone(backup)
            assert backup is not None
            self.assertEqual(backup.read_bytes(), source_bytes)
            self.assertEqual(source_hash, hashlib.sha256(source_bytes).hexdigest())
            MIGRATION.validate_workflow(migrated)
            migrated_bytes = target.read_bytes()
            unchanged, _second_hash, second_backup = MIGRATION.migrate_file(
                target,
                write=True,
            )
            self.assertIsNone(second_backup)
            self.assertEqual(target.read_bytes(), migrated_bytes)
            self.assertEqual(backup.read_bytes(), source_bytes)
            self.assertEqual(unchanged, migrated)

    def test_mismatched_existing_backup_is_never_reused_or_overwritten(self) -> None:
        source_marker = self.source.get("extra", {}).get(MIGRATION.MIGRATION_SCHEMA)
        if isinstance(source_marker, dict) and int(source_marker.get("version", 0)) >= 2:
            self.skipTest("pre-v2 fixture is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / WORKFLOW.name
            shutil.copy2(self.fixture_path, target)
            source_bytes = target.read_bytes()
            backup = target.with_suffix(target.suffix + MIGRATION.BACKUP_SUFFIX)
            backup.write_bytes(b"unrelated recovery copy")
            with self.assertRaises(MIGRATION.WorkflowError):
                MIGRATION.migrate_file(
                    target,
                    write=True,
                    expected_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                )
            self.assertEqual(target.read_bytes(), source_bytes)
            self.assertEqual(backup.read_bytes(), b"unrelated recovery copy")

    def test_source_digest_guard_refuses_the_wrong_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / WORKFLOW.name
            shutil.copy2(self.fixture_path, target)
            with self.assertRaises(MIGRATION.WorkflowError):
                MIGRATION.migrate_file(
                    target,
                    write=True,
                    expected_source_sha256="0" * 64,
                )
            self.assertFalse(
                target.with_suffix(target.suffix + MIGRATION.BACKUP_SUFFIX).exists()
            )


@unittest.skipUnless(
    LIVE_STABILITY_CAPTURE.is_file(), "exact FAST H3 live stability capture is absent"
)
class ExactFastH3LiveStabilityCaptureTests(unittest.TestCase):
    def test_exact_capture_v1_to_v2_dry_run_removes_only_legacy_reroute_chain(self) -> None:
        source_bytes = LIVE_STABILITY_CAPTURE.read_bytes()
        self.assertEqual(
            hashlib.sha256(source_bytes).hexdigest(),
            LIVE_STABILITY_CAPTURE_SHA256,
        )
        source = json.loads(source_bytes.decode("utf-8"))
        migrated = MIGRATION.migrate_workflow(
            source,
            source_file_sha256=LIVE_STABILITY_CAPTURE_SHA256,
        )
        MIGRATION.validate_workflow(migrated)
        self.assertEqual(
            migrated["extra"][MIGRATION.MIGRATION_SCHEMA]["version"],
            MIGRATION.MIGRATION_VERSION,
        )
        self.assertNotIn(711, {int(node["id"]) for node in migrated["nodes"]})
        self.assertFalse(
            {1396, 1416} & {int(link[0]) for link in migrated["links"]}
        )
        target = _node(migrated, 673)
        self.assertIsNone(_input(target, "shot_count_override").get("link"))
        self.assertEqual(target["widgets_values"][4:6], ["custom", 15])
        note = _node(migrated, 116)["widgets_values"][0]
        self.assertIn("Leave it on **reuse**", note)
        self.assertEqual(MIGRATION.migrate_workflow(migrated), migrated)


if __name__ == "__main__":
    unittest.main()
