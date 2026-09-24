from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATH = REPO_ROOT / "tools" / "migrate_music_video_production.py"
SPEC = importlib.util.spec_from_file_location("music_video_workflow_migration", MIGRATION_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError(f"Cannot load migration module: {MIGRATION_PATH}")
MIGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIGRATION)


WORKFLOWS = (
    Path(r"C:\ComfyUI\app\user\default\workflows\14_ltx25_i2v+AUDIO_IN_PROCESS_SYNC.json"),
    Path(r"C:\Users\danrh\Desktop\14_ltx25_i2v+AUDIO_IN_PROCESS_SYNC.json"),
)
APP_WORKFLOW = WORKFLOWS[0]
ORGANIZED_WORKFLOW = Path(
    r"C:\Users\danrh\Desktop\14_ltx25_i2v+AUDIO_IN_PROCESS_SYNC_organized.json"
)


def _node(workflow: dict, node_id: int) -> dict:
    return next(node for node in workflow["nodes"] if int(node["id"]) == node_id)


def _first_widget(node: dict):
    value = node.get("widgets_values")
    return value[0] if isinstance(value, list) else value


def _all_nodes(workflow: dict):
    yield ("main", None), workflow["nodes"]
    for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
        yield ("subgraph", str(subgraph["id"])), subgraph.get("nodes", [])


def _candidate_encoders(workflow: dict) -> list[dict]:
    return [
        node
        for _scope, nodes in _all_nodes(workflow)
        for node in nodes
        if node.get("type") == "TextEncodeAceStepAudio1.5"
    ]


def _topology_signature(workflow: dict) -> dict:
    return {
        "main_nodes": [
            {
                "id": node["id"],
                "type": node["type"],
                "inputs": [
                    (item.get("name"), item.get("type"), item.get("link"))
                    for item in node.get("inputs", [])
                ],
                "outputs": [
                    (item.get("name"), item.get("type"), item.get("links"))
                    for item in node.get("outputs", [])
                ],
            }
            for node in workflow["nodes"]
        ],
        "main_links": workflow["links"],
        "subgraphs": [
            {
                "id": subgraph["id"],
                "inputs": subgraph.get("inputs", []),
                "outputs": subgraph.get("outputs", []),
                "nodes": [
                    {
                        "id": node["id"],
                        "type": node["type"],
                        "inputs": [
                            (item.get("name"), item.get("type"), item.get("link"))
                            for item in node.get("inputs", [])
                        ],
                        "outputs": [
                            (item.get("name"), item.get("type"), item.get("links"))
                            for item in node.get("outputs", [])
                        ],
                    }
                    for node in subgraph.get("nodes", [])
                ],
                "links": subgraph.get("links", []),
            }
            for subgraph in workflow.get("definitions", {}).get("subgraphs", [])
        ],
    }


def _non_model_widget_signature(workflow: dict) -> dict:
    signature = {}
    for scope, nodes in _all_nodes(workflow):
        for node in nodes:
            if int(node["id"]) in {116, 510, 559}:
                continue
            if node.get("type") == MIGRATION.PERFORMANCE_NODE_TYPE:
                continue
            if str(node.get("title", "")) == MIGRATION.PERFORMANCE_PREVIEW_TITLE:
                continue
            signature[(scope, int(node["id"]))] = copy.deepcopy(
                node.get("widgets_values")
            )
    return signature


@unittest.skipUnless(all(path.is_file() for path in WORKFLOWS), "local synchronized workflow copies are absent")
class MusicVideoProductionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflows = [json.loads(path.read_text(encoding="utf-8")) for path in WORKFLOWS]

    def test_both_workflow_sources_migrate_and_pass_graph_and_production_invariants(self) -> None:
        for path, workflow in zip(WORKFLOWS, self.workflows):
            with self.subTest(path=str(path)):
                migrated = MIGRATION.migrate_workflow(workflow)
                MIGRATION.validate_workflow(migrated)
                marker = migrated["extra"][MIGRATION.MIGRATION_SCHEMA]
                self.assertEqual(marker["version"], MIGRATION.MIGRATION_VERSION)
                self.assertEqual(marker["production_concept"], "Audition and select")
                self.assertEqual(marker["audition_candidate_count"], 2)
                self.assertEqual(marker["selector_policy"]["revision"], 4)
                self.assertEqual(
                    marker["selector_policy"]["aligned_double_time"],
                    "mandatory_alignment_pressure_vocal_plus_two_of_four_quality_signals",
                )
                self.assertEqual(
                    marker["selector_policy"]["double_time"],
                    "hard_failure_outside_evidence_bundle",
                )
                self.assertEqual(
                    marker["selector_policy"]["supporting_signals_required"], 2
                )

    def test_performance_modes_append_natural_without_reindexing_legacy_choices(self) -> None:
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

    def test_user_values_remain_independent_between_the_two_copies(self) -> None:
        app, desktop = self.workflows
        self.assertEqual(_first_widget(_node(app, 178)), 20)
        self.assertEqual(_first_widget(_node(desktop, 178)), 25)
        self.assertEqual(_node(app, 569)["widgets_values"][0], 44)
        self.assertEqual(_node(desktop, 569)["widgets_values"][0], 30)
        self.assertEqual(_node(app, 418)["widgets_values"][0], 347267244880141)
        self.assertEqual(_node(desktop, 418)["widgets_values"][0], 347267244880141)
        self.assertEqual(
            app["extra"][MIGRATION.MIGRATION_SCHEMA]["legacy_router_mode"],
            "stabilized",
        )
        self.assertEqual(
            desktop["extra"][MIGRATION.MIGRATION_SCHEMA]["legacy_router_mode"],
            "default",
        )

    def test_migration_is_idempotent(self) -> None:
        for path, workflow in zip(WORKFLOWS, self.workflows):
            with self.subTest(path=str(path)):
                migrated = MIGRATION.migrate_workflow(workflow)
                self.assertEqual(MIGRATION.migrate_workflow(migrated), migrated)

    def test_v7_to_current_upgrade_preserves_controls_and_both_ace_topologies(self) -> None:
        outer_encoder_counts = []
        for path, current in zip(WORKFLOWS, self.workflows):
            with self.subTest(path=str(path)):
                legacy = copy.deepcopy(current)
                marker = legacy["extra"][MIGRATION.MIGRATION_SCHEMA]
                marker["version"] = 7
                canonical = marker["canonical_ace"]
                canonical["second_language_model"] = "qwen_1.7b_ace15.safetensors"
                canonical["ace_4b_upgrade"] = (
                    "not installed; requires qwen_4b_ace15.safetensors"
                )
                canonical.pop("second_language_model_sha256", None)
                canonical.pop("second_language_model_parameter_class", None)
                canonical.pop("second_language_model_role", None)
                canonical.pop("second_language_model_url", None)
                loader = next(
                    node for node in legacy["nodes"] if node["type"] == "DualCLIPLoader"
                )
                loader["widgets_values"][1] = "qwen_1.7b_ace15.safetensors"
                loader["properties"]["models"][1].update(
                    {
                        "name": "qwen_1.7b_ace15.safetensors",
                        "url": (
                            "https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/"
                            "resolve/main/split_files/text_encoders/"
                            "qwen_1.7b_ace15.safetensors"
                        ),
                        "directory": "text_encoders",
                    }
                )
                for encode in _candidate_encoders(legacy):
                    encode["title"] = str(encode["title"]).replace(
                        "ACE 4B LM", "ACE 1.7B LM"
                    )

                subgraph_topology_before = _topology_signature(legacy)["subgraphs"]
                widgets_before = _non_model_widget_signature(legacy)
                migrated = MIGRATION.migrate_workflow(legacy)
                MIGRATION.validate_workflow(migrated)
                self.assertEqual(
                    _topology_signature(migrated)["subgraphs"],
                    subgraph_topology_before,
                )
                self.assertEqual(
                    _non_model_widget_signature(migrated),
                    widgets_before,
                )
                outer_encoder_counts.append(
                    sum(
                        node.get("type") == "TextEncodeAceStepAudio1.5"
                        for node in migrated["nodes"]
                    )
                )
        self.assertEqual(outer_encoder_counts, [0, 4])

    def test_official_ace_4b_loader_metadata_and_labels_are_persisted(self) -> None:
        for path, workflow in zip(WORKFLOWS, self.workflows):
            with self.subTest(path=str(path)):
                loader = next(
                    node for node in workflow["nodes"] if node["type"] == "DualCLIPLoader"
                )
                self.assertEqual(
                    loader["widgets_values"][1], MIGRATION.ACE_4B_LANGUAGE_MODEL
                )
                second_asset = loader["properties"]["models"][1]
                self.assertEqual(second_asset["name"], MIGRATION.ACE_4B_LANGUAGE_MODEL)
                self.assertEqual(second_asset["url"], MIGRATION.ACE_4B_LANGUAGE_MODEL_URL)
                self.assertEqual(second_asset["directory"], "text_encoders")
                encoders = _candidate_encoders(workflow)
                self.assertEqual(len(encoders), 4)
                self.assertTrue(
                    all("ACE 4B LM" in str(node.get("title", "")) for node in encoders)
                )
                canonical = workflow["extra"][MIGRATION.MIGRATION_SCHEMA]["canonical_ace"]
                for key, expected in MIGRATION.ACE_4B_METADATA_MARKER.items():
                    self.assertEqual(canonical[key], expected)
                self.assertNotIn("ace_4b_upgrade", canonical)
                note_copy = " ".join(
                    str(value) for value in _node(workflow, 116)["widgets_values"]
                )
                self.assertIn("official ACE-specific **4B language model**", note_copy)
                self.assertIn(MIGRATION.ACE_4B_LANGUAGE_MODEL, note_copy)
                self.assertIn("juxtaposition is not musical fusion", note_copy)

    @unittest.skipUnless(ORGANIZED_WORKFLOW.is_file(), "organized delivery is absent")
    def test_attached_organized_delivery_is_already_persisted_at_current_revision(self) -> None:
        organized = json.loads(ORGANIZED_WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual(MIGRATION.migrate_workflow(organized), organized)
        MIGRATION.validate_workflow(organized)

        performance = next(
            node
            for node in organized["nodes"]
            if node.get("type") == MIGRATION.PERFORMANCE_NODE_TYPE
        )
        self.assertEqual(
            [item.get("name") for item in performance.get("inputs", [])],
            [
                "ltx_prompt",
                "lyrics",
                "song_duration_seconds",
                "excerpt_start_seconds",
                "excerpt_duration_seconds",
            ],
        )
        self.assertNotIn(
            "performance_mode",
            {item.get("name") for item in performance.get("inputs", [])},
        )

    def test_v9_to_v10_adds_only_the_stable_camera_widget(self) -> None:
        for path, workflow in zip(WORKFLOWS, self.workflows):
            with self.subTest(path=str(path)):
                legacy = copy.deepcopy(MIGRATION.migrate_workflow(workflow))
                marker = legacy["extra"][MIGRATION.MIGRATION_SCHEMA]
                marker["version"] = 9
                marker.pop("ltx_camera_capability", None)
                target_before = copy.deepcopy(_node(legacy, 510))
                if (
                    target_before.get("widgets_values")
                    and target_before["widgets_values"][-1]
                    in MIGRATION.CAMERA_CAPABILITIES
                ):
                    legacy_target = _node(legacy, 510)
                    legacy_target["widgets_values"] = legacy_target["widgets_values"][:-1]
                    target_before = copy.deepcopy(legacy_target)

                topology_before = _topology_signature(legacy)
                migrated = MIGRATION.migrate_workflow(legacy)
                MIGRATION.validate_workflow(migrated)
                target_after = _node(migrated, 510)

                self.assertEqual(_topology_signature(migrated), topology_before)
                self.assertEqual(target_after["pos"], target_before["pos"])
                self.assertEqual(target_after["inputs"], target_before["inputs"])
                self.assertEqual(
                    target_after["widgets_values"],
                    target_before["widgets_values"]
                    + [MIGRATION.DEFAULT_CAMERA_CAPABILITY],
                )
                self.assertNotIn(
                    MIGRATION.CAMERA_CAPABILITY_INPUT,
                    {item.get("name") for item in target_after.get("inputs", [])},
                )
                self.assertEqual(
                    migrated["extra"][MIGRATION.MIGRATION_SCHEMA][
                        "ltx_camera_capability"
                    ],
                    MIGRATION.CAMERA_CAPABILITY_MARKER,
                )
                self.assertEqual(MIGRATION.migrate_workflow(migrated), migrated)

    def test_current_migration_preserves_advanced_camera_selection(self) -> None:
        for path, workflow in zip(WORKFLOWS, self.workflows):
            with self.subTest(path=str(path)):
                edited = MIGRATION.migrate_workflow(workflow)
                _node(edited, 510)["widgets_values"][-1] = (
                    "Advanced / controlled camera"
                )
                migrated = MIGRATION.migrate_workflow(edited)
                self.assertEqual(
                    _node(migrated, 510)["widgets_values"][-1],
                    "Advanced / controlled camera",
                )

    def test_planner_uses_the_fixed_ace_song_duration_in_both_copies(self) -> None:
        for path, workflow in zip(WORKFLOWS, self.workflows):
            with self.subTest(path=str(path)):
                planner = _node(workflow, 578)
                duration_slot = next(
                    index
                    for index, item in enumerate(planner["inputs"])
                    if item["name"] == "duration_seconds"
                )
                link_id = planner["inputs"][duration_slot]["link"]
                link = next(link for link in workflow["links"] if int(link[0]) == int(link_id))
                self.assertEqual(int(link[1]), 562)
                self.assertEqual(int(link[2]), 0)
                self.assertEqual(int(link[3]), 578)
                self.assertEqual(int(link[4]), duration_slot)

    def test_ace_auraflow_shift_is_persisted_as_a_widget_list(self) -> None:
        for path, workflow in zip(WORKFLOWS, self.workflows):
            with self.subTest(path=str(path)):
                self.assertEqual(_node(workflow, 563)["widgets_values"], [3.0])

    def test_visible_qc_help_matches_the_selector_policy(self) -> None:
        for path, workflow in zip(WORKFLOWS, self.workflows):
            with self.subTest(path=str(path)):
                note = _node(workflow, 116)
                note_copy = " ".join(str(value) for value in note["widgets_values"])
                self.assertIn(
                    "at least two of visual-recovery coverage",
                    note_copy,
                )
                audit = next(
                    node
                    for node in workflow["nodes"]
                    if str(node.get("title", "")).startswith("FULL MUSIC QC AUDIT")
                )
                self.assertIn("advisories", audit["title"])

    def test_disabled_missing_node_experiment_is_absent(self) -> None:
        for path, workflow in zip(WORKFLOWS, self.workflows):
            with self.subTest(path=str(path)):
                workflow = MIGRATION.migrate_workflow(workflow)
                subgraph = next(
                    item
                    for item in workflow["definitions"]["subgraphs"]
                    if item["id"] == MIGRATION.LTX_SUBGRAPH_ID
                )
                self.assertNotIn("Film Grain", {node["type"] for node in subgraph["nodes"]})
                self.assertNotIn(
                    "lut test (disabled)",
                    {str(group.get("title", "")).strip().casefold() for group in subgraph.get("groups", [])},
                )

    def test_a2v_is_neutral_by_default_and_final_audio_is_distinct(self) -> None:
        for workflow in self.workflows:
            outer = next(
                node for node in workflow["nodes"] if node["type"] == MIGRATION.LTX_SUBGRAPH_ID
            )
            self.assertEqual(outer["widgets_values"][-1], 1.0)
            self.assertNotEqual(
                next(item for item in outer["inputs"] if item["name"] == "conditioning_audio")["link"],
                next(item for item in outer["inputs"] if item["name"] == "final_soundtrack")["link"],
            )
            subgraph = next(
                item
                for item in workflow["definitions"]["subgraphs"]
                if item["id"] == MIGRATION.LTX_SUBGRAPH_ID
            )
            self.assertEqual(subgraph["inputs"][15]["linkIds"], [985])
            self.assertEqual(subgraph["inputs"][16]["linkIds"], [967])
            self.assertEqual(len(subgraph["inputs"][17]["linkIds"]), 2)

    def test_performance_toggle_is_post_gate_and_does_not_touch_audio_paths(self) -> None:
        for path, workflow in zip(WORKFLOWS, self.workflows):
            with self.subTest(path=str(path)):
                workflow = MIGRATION.migrate_workflow(workflow)
                controls = [
                    node
                    for node in workflow["nodes"]
                    if node.get("type") == MIGRATION.PERFORMANCE_NODE_TYPE
                ]
                self.assertEqual(len(controls), 1)
                control = controls[0]
                self.assertIn(
                    control["widgets_values"][0], MIGRATION.PERFORMANCE_MODES
                )
                self.assertEqual(
                    control["widgets_values"][1:],
                    MIGRATION.PERFORMANCE_WIDGET_DEFAULTS[1:],
                )
                marker = workflow["extra"][MIGRATION.MIGRATION_SCHEMA][
                    "ltx_performance_mode"
                ]
                self.assertEqual(marker["modes"], list(MIGRATION.PERFORMANCE_MODES))
                self.assertEqual(marker["default"], MIGRATION.DEFAULT_PERFORMANCE_MODE)
                self.assertIn("lyrics are not a timing schedule", marker["lyrics_policy"])

                nodes = {int(node["id"]): node for node in workflow["nodes"]}
                links = {int(link[0]): link for link in workflow["links"]}

                def origin(input_name: str, node: dict = control):
                    item = next(
                        value for value in node["inputs"] if value["name"] == input_name
                    )
                    link = links[int(item["link"])]
                    return nodes[int(link[1])], int(link[2])

                selector = next(
                    node
                    for node in workflow["nodes"]
                    if node.get("type") == "DiffusionGemmaAudioCandidateSelector"
                )
                self.assertEqual(origin("ltx_prompt"), (_node(workflow, 188), 0))
                self.assertEqual(origin("lyrics"), (_node(workflow, 556), 4))
                self.assertEqual(origin("song_duration_seconds"), (_node(workflow, 562), 0))
                self.assertEqual(origin("excerpt_start_seconds"), (selector, 1))
                self.assertEqual(origin("excerpt_duration_seconds"), (_node(workflow, 178), 0))

                outer = next(
                    node
                    for node in workflow["nodes"]
                    if node.get("type") == MIGRATION.LTX_SUBGRAPH_ID
                )
                self.assertEqual(origin("value", outer), (control, 0))
                self.assertFalse(
                    any(
                        int(link[1]) == 188 and int(link[3]) == int(outer["id"])
                        for link in workflow["links"]
                    )
                )
                guide = next(
                    node
                    for node in workflow["nodes"]
                    if node.get("type") == "DiffusionGemmaLTXAudioGuide"
                )
                self.assertEqual(origin("conditioning_audio", outer), (guide, 0))
                self.assertEqual(origin("final_soundtrack", outer), (guide, 1))

                previews = [
                    node
                    for node in workflow["nodes"]
                    if str(node.get("title", ""))
                    == MIGRATION.PERFORMANCE_PREVIEW_TITLE
                ]
                self.assertEqual(len(previews), 1)
                self.assertEqual(origin("source", previews[0]), (control, 1))

                target = _node(workflow, 510)
                self.assertEqual(
                    target["widgets_values"][4],
                    MIGRATION.NEUTRAL_PERFORMANCE_GUIDANCE,
                )

    def test_current_migration_preserves_a_user_selected_lip_sync_mode(self) -> None:
        for path, workflow in zip(WORKFLOWS, self.workflows):
            with self.subTest(path=str(path)):
                edited = copy.deepcopy(workflow)
                control = next(
                    node
                    for node in edited["nodes"]
                    if node.get("type") == MIGRATION.PERFORMANCE_NODE_TYPE
                )
                control["widgets_values"][0] = "Lyrics + lip sync"
                migrated = MIGRATION.migrate_workflow(edited)
                selected = next(
                    node
                    for node in migrated["nodes"]
                    if node.get("type") == MIGRATION.PERFORMANCE_NODE_TYPE
                )
                self.assertEqual(selected["widgets_values"][0], "Lyrics + lip sync")
                self.assertEqual(
                    selected["widgets_values"][1:],
                    MIGRATION.PERFORMANCE_WIDGET_DEFAULTS[1:],
                )

    def test_v8_to_v9_adds_only_the_performance_seam_and_neutralizes_known_guidance(self) -> None:
        for path, workflow in zip(WORKFLOWS, self.workflows):
            with self.subTest(path=str(path)):
                legacy = copy.deepcopy(MIGRATION.migrate_workflow(workflow))
                control = next(
                    node
                    for node in legacy["nodes"]
                    if node.get("type") == MIGRATION.PERFORMANCE_NODE_TYPE
                )
                preview = next(
                    node
                    for node in legacy["nodes"]
                    if str(node.get("title", ""))
                    == MIGRATION.PERFORMANCE_PREVIEW_TITLE
                )
                removed_node_ids = {int(control["id"]), int(preview["id"])}
                removed_link_ids = {
                    int(link[0])
                    for link in legacy["links"]
                    if int(link[1]) in removed_node_ids or int(link[3]) in removed_node_ids
                }
                legacy["nodes"] = [
                    node
                    for node in legacy["nodes"]
                    if int(node["id"]) not in removed_node_ids
                ]
                legacy["links"] = [
                    link
                    for link in legacy["links"]
                    if int(link[0]) not in removed_link_ids
                ]
                for node in legacy["nodes"]:
                    for item in node.get("inputs", []):
                        if item.get("link") in removed_link_ids:
                            item["link"] = None
                    for output in node.get("outputs", []):
                        if isinstance(output.get("links"), list):
                            output["links"] = [
                                link_id
                                for link_id in output["links"]
                                if int(link_id) not in removed_link_ids
                            ]

                gate = _node(legacy, 188)
                outer = next(
                    node
                    for node in legacy["nodes"]
                    if node.get("type") == MIGRATION.LTX_SUBGRAPH_ID
                )
                direct_link_id = int(legacy["last_link_id"]) + 1
                gate_slot = next(
                    index
                    for index, item in enumerate(gate["outputs"])
                    if item["name"] == "prompt"
                )
                outer_slot = next(
                    index
                    for index, item in enumerate(outer["inputs"])
                    if item["name"] == "value"
                )
                legacy["links"].append(
                    [
                        direct_link_id,
                        int(gate["id"]),
                        gate_slot,
                        int(outer["id"]),
                        outer_slot,
                        "STRING",
                    ]
                )
                gate["outputs"][gate_slot].setdefault("links", []).append(
                    direct_link_id
                )
                outer["inputs"][outer_slot]["link"] = direct_link_id
                legacy["last_link_id"] = direct_link_id
                marker = legacy["extra"][MIGRATION.MIGRATION_SCHEMA]
                marker["version"] = 8
                marker.pop("ltx_performance_mode", None)
                _node(legacy, 510)["widgets_values"][4] = (
                    MIGRATION.LEGACY_FORCED_LIP_SYNC_GUIDANCE
                )

                widgets_before = _non_model_widget_signature(legacy)
                migrated = MIGRATION.migrate_workflow(legacy)
                MIGRATION.validate_workflow(migrated)
                self.assertEqual(
                    _non_model_widget_signature(migrated), widgets_before
                )
                self.assertEqual(
                    _node(migrated, 510)["widgets_values"][4],
                    MIGRATION.NEUTRAL_PERFORMANCE_GUIDANCE,
                )
                inserted = next(
                    node
                    for node in migrated["nodes"]
                    if node.get("type") == MIGRATION.PERFORMANCE_NODE_TYPE
                )
                self.assertEqual(
                    inserted["widgets_values"][0], "Natural / audio-led sync"
                )

@unittest.skipUnless(APP_WORKFLOW.is_file(), "app workflow is absent")
class AppCameraCapabilityWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = json.loads(APP_WORKFLOW.read_text(encoding="utf-8"))

    def test_app_delivery_is_current_secure_and_idempotent(self) -> None:
        migrated = MIGRATION.migrate_workflow(self.workflow)
        MIGRATION.validate_workflow(migrated)
        self.assertEqual(MIGRATION.migrate_workflow(migrated), migrated)
        self.assertEqual(MIGRATION._sensitive_key_paths(migrated), [])
        marker = migrated["extra"][MIGRATION.MIGRATION_SCHEMA]["ltx_performance_mode"]
        self.assertEqual(marker["default"], "Natural / audio-led sync")
        self.assertEqual(marker["modes"], list(MIGRATION.PERFORMANCE_MODES))
        target = _node(migrated, 510)
        self.assertEqual(len(target["widgets_values"]), 9)
        self.assertEqual(
            target["widgets_values"][-1], MIGRATION.DEFAULT_CAMERA_CAPABILITY
        )

    def test_v9_app_serialization_migrates_without_state_or_topology_changes(self) -> None:
        legacy = copy.deepcopy(MIGRATION.migrate_workflow(self.workflow))
        marker = legacy["extra"][MIGRATION.MIGRATION_SCHEMA]
        marker["version"] = 9
        marker.pop("ltx_camera_capability", None)
        target = _node(legacy, 510)
        if target["widgets_values"][-1] in MIGRATION.CAMERA_CAPABILITIES:
            target["widgets_values"] = target["widgets_values"][:-1]

        performance_before = copy.deepcopy(
            next(
                node
                for node in legacy["nodes"]
                if node.get("type") == MIGRATION.PERFORMANCE_NODE_TYPE
            )
        )
        self.assertEqual(
            [item.get("name") for item in performance_before.get("inputs", [])],
            [
                "ltx_prompt",
                "lyrics",
                "performance_mode",
                "song_duration_seconds",
                "excerpt_start_seconds",
                "excerpt_duration_seconds",
            ],
        )
        topology_before = _topology_signature(legacy)
        groups_before = copy.deepcopy(legacy.get("groups", []))
        positions_before = {
            (scope, int(node["id"])): copy.deepcopy(node.get("pos"))
            for scope, nodes in _all_nodes(legacy)
            for node in nodes
        }
        widgets_before = {
            (scope, int(node["id"])): copy.deepcopy(node.get("widgets_values"))
            for scope, nodes in _all_nodes(legacy)
            for node in nodes
            if int(node["id"]) != 510
        }
        target_values_before = copy.deepcopy(target["widgets_values"])

        migrated = MIGRATION.migrate_workflow(legacy)
        MIGRATION.validate_workflow(migrated)
        self.assertEqual(_topology_signature(migrated), topology_before)
        self.assertEqual(migrated.get("groups", []), groups_before)
        self.assertEqual(
            {
                (scope, int(node["id"])): node.get("pos")
                for scope, nodes in _all_nodes(migrated)
                for node in nodes
            },
            positions_before,
        )
        self.assertEqual(
            {
                (scope, int(node["id"])): node.get("widgets_values")
                for scope, nodes in _all_nodes(migrated)
                for node in nodes
                if int(node["id"]) != 510
            },
            widgets_before,
        )
        self.assertEqual(
            next(
                node
                for node in migrated["nodes"]
                if node.get("type") == MIGRATION.PERFORMANCE_NODE_TYPE
            ),
            performance_before,
        )
        migrated_target = _node(migrated, 510)
        self.assertEqual(
            migrated_target["widgets_values"],
            target_values_before + [MIGRATION.DEFAULT_CAMERA_CAPABILITY],
        )
        self.assertEqual(MIGRATION.migrate_workflow(migrated), migrated)


@unittest.skipUnless(ORGANIZED_WORKFLOW.is_file(), "organized delivery is absent")
class OrganizedCameraCapabilityWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = json.loads(ORGANIZED_WORKFLOW.read_text(encoding="utf-8"))
        cls.migrated = MIGRATION.migrate_workflow(cls.workflow)

    def test_delivery_is_current_secure_and_idempotent(self) -> None:
        MIGRATION.validate_workflow(self.migrated)
        self.assertEqual(MIGRATION.migrate_workflow(self.migrated), self.migrated)
        self.assertEqual(MIGRATION._sensitive_key_paths(self.migrated), [])

        target = _node(self.migrated, 510)
        self.assertEqual(
            target["widgets_values"][-1], MIGRATION.DEFAULT_CAMERA_CAPABILITY
        )
        self.assertEqual(len(target["widgets_values"]), 9)
        self.assertNotIn(
            MIGRATION.CAMERA_CAPABILITY_INPUT,
            {item.get("name") for item in target.get("inputs", [])},
        )
        marker = self.migrated["extra"][MIGRATION.MIGRATION_SCHEMA]
        self.assertEqual(marker["version"], MIGRATION.MIGRATION_VERSION)
        self.assertEqual(
            marker["ltx_camera_capability"], MIGRATION.CAMERA_CAPABILITY_MARKER
        )

    def test_ui_resaved_performance_shape_is_preserved(self) -> None:
        performance = next(
            node
            for node in self.migrated["nodes"]
            if node.get("type") == MIGRATION.PERFORMANCE_NODE_TYPE
        )
        self.assertEqual(
            [item.get("name") for item in performance.get("inputs", [])],
            [
                "ltx_prompt",
                "lyrics",
                "song_duration_seconds",
                "excerpt_start_seconds",
                "excerpt_duration_seconds",
            ],
        )
        self.assertIn(performance["widgets_values"][0], MIGRATION.PERFORMANCE_MODES)
        self.assertEqual(
            performance["widgets_values"][1:],
            MIGRATION.PERFORMANCE_WIDGET_DEFAULTS[1:],
        )

    def test_v9_upgrade_changes_no_topology_or_protected_ltx_internals(self) -> None:
        legacy = copy.deepcopy(self.migrated)
        legacy_marker = legacy["extra"][MIGRATION.MIGRATION_SCHEMA]
        legacy_marker["version"] = 9
        legacy_marker.pop("ltx_camera_capability", None)
        target = _node(legacy, 510)
        target["widgets_values"] = target["widgets_values"][:-1]

        topology_before = _topology_signature(legacy)
        performance_before = copy.deepcopy(
            next(
                node
                for node in legacy["nodes"]
                if node.get("type") == MIGRATION.PERFORMANCE_NODE_TYPE
            )
        )
        ltx_before = next(
            item
            for item in legacy["definitions"]["subgraphs"]
            if item["id"] == MIGRATION.LTX_SUBGRAPH_ID
        )
        protected_nodes_before = copy.deepcopy(
            [node for node in ltx_before["nodes"] if int(node["id"]) in {625, 626}]
        )
        protected_links_before = copy.deepcopy(
            [
                link
                for link in ltx_before["links"]
                if int(link["id"]) in {1138, 1139, 1140, 1141, 1142}
            ]
        )

        migrated = MIGRATION.migrate_workflow(legacy)
        MIGRATION.validate_workflow(migrated)
        self.assertEqual(_topology_signature(migrated), topology_before)
        self.assertEqual(
            next(
                node
                for node in migrated["nodes"]
                if node.get("type") == MIGRATION.PERFORMANCE_NODE_TYPE
            ),
            performance_before,
        )
        ltx_after = next(
            item
            for item in migrated["definitions"]["subgraphs"]
            if item["id"] == MIGRATION.LTX_SUBGRAPH_ID
        )
        self.assertEqual(
            [node for node in ltx_after["nodes"] if int(node["id"]) in {625, 626}],
            protected_nodes_before,
        )
        self.assertEqual(
            [
                link
                for link in ltx_after["links"]
                if int(link["id"]) in {1138, 1139, 1140, 1141, 1142}
            ],
            protected_links_before,
        )

    def test_advanced_camera_selection_survives_repeat_migration(self) -> None:
        edited = copy.deepcopy(self.migrated)
        _node(edited, 510)["widgets_values"][-1] = (
            "Advanced / controlled camera"
        )
        migrated = MIGRATION.migrate_workflow(edited)
        self.assertEqual(
            _node(migrated, 510)["widgets_values"][-1],
            "Advanced / controlled camera",
        )


if __name__ == "__main__":
    unittest.main()
