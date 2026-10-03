from __future__ import annotations

import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import advertising_contract_nodes as creative
import advertising_planning_nodes as advertising
import nodes as director_core
import production_planning_nodes as legacy


EIGHT_SHOT_PROMPT = """subject_definitions:
<Subject 1>: the same performer jointly defined by <Picture 1> and <Picture 2>.
<Subject 2>: the same beverage package defined by <Picture 3>.

summary:
A colorful beverage advertisement follows one performer and one governed product package.

retention_analysis:
<Subject 1>: fully_preserved - face, hair, body, and wardrobe remain the same.
<Subject 2>: fully_preserved - package geometry, material, colors, and label remain the same.
<Audio 1>: reference - exact campaign soundtrack and timing authority.

detailed_description:
Bright live-action commercial photography with readable stable framing.
[Shot 1] A wide opening introduces the performer carrying the beverage.
[Shot 2] At 00:03.000, a clean cut shows a medium walking view.
[Shot 3] At 00:06.000, a close product presentation fills the foreground.
[Shot 4] At 00:09.000, a stable group composition responds to the beat.
[Shot 5] At 00:15.000, a clean cut opens the second campaign movement.
[Shot 6] At 00:18.000, the performer opens and enjoys the beverage.
[Shot 7] At 00:21.000, a product beauty shot holds package geometry clearly.
[Shot 8] At 00:24.000, the campaign resolves toward the finishing card.

overall_soundscape:
The supplied <Audio 1> remains the exact soundtrack authority.

non_diegetic_music:
Use the supplied <Audio 1> without inventing a replacement song."""


def build_upstream(audio_hash: str = "a" * 64) -> tuple[str, str, str, str]:
    campaign = creative.DiffusionGemmaAdvertisementCampaignContract().build(
        "GLOWBLOOM",
        "Sparkling Botanical Water",
        "Yuzu Mint",
        "Trend-aware adults in their twenties",
        "Drive trial and brand recognition",
        "Open something brighter",
        "Find your flavor",
        "[]",
        30.0,
        "9:16",
        3.0,
        '{"primary":"9:16","adaptations":["1:1","16:9"],"cutdowns_seconds":[15,6]}',
        campaign_name="GLOWBLOOM Brighter",
    )[0]
    references = creative.DiffusionGemmaAdvertisementReferenceContract().build(
        creative.REFERENCE_SLOT_POLICIES[0],
        "One adult performer in a cobalt jacket with consistent face, hair, body, and wardrobe.",
        "One slim GLOWBLOOM Yuzu Mint can with fixed geometry, green-yellow palette, and printed label.",
        "geometry, proportions, material, label placement, green-yellow color system",
        "product contact sheet",
    )[0]
    soundtrack = creative.DiffusionGemmaAdvertisementSoundtrackContract().build(
        "Instrumental",
        "bright indie electronic pop with a tactile human groove",
        "optimistic, stylish, lightly playful",
        "dry drums, bass guitar, hand percussion, glassy synth accents",
        30.0,
        108.0,
        "4",
        "unknown",
        "",
        "None",
    )[0]
    project = legacy.DiffusionGemmaProjectMasterContract().build(
        "Create the contracted GLOWBLOOM advertisement.",
        30.0,
        "9:16",
        audio_hash,
        "",
        0.0,
        30.0,
        "MiniMax H3 Ref2VA",
        '{"primary":"9:16","adaptations":["1:1","16:9"],"cutdowns_seconds":[15,6]}',
        15.0,
    )[0]
    return campaign, references, soundtrack, project


def build_contract(audio_hash: str = "a" * 64, boundary: str = "relay_continuity") -> tuple:
    campaign, references, soundtrack, project = build_upstream(audio_hash)
    return advertising.AdvertisementMasterContract().build(
        campaign,
        references,
        soundtrack,
        project,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        audio_hash,
        8,
        boundary,
    )


def build_plan(boundary: str = "relay_continuity", audio_hash: str = "a" * 64) -> tuple[str, tuple]:
    contract = build_contract(audio_hash=audio_hash, boundary=boundary)[0]
    defaults = advertising.AdvertisementPlanningDefaults().build(
        contract,
        "The performer carries, opens, presents, and drinks the can while its package remains recognizable.",
        "Campaign default",
    )
    result = advertising.AdvertisementMultiShotPlanner().plan(
        contract,
        EIGHT_SHOT_PROMPT,
        defaults[0],
        defaults[1],
        "Natural / audio-led sync",
    )
    return contract, result


class AdvertisementPlanningTests(unittest.TestCase):
    @staticmethod
    def _director_host_inputs(*, parse_valid: bool, blocked: bool = False):
        report = {
            "schema": "dg-grounding-report/1",
            "analysis_status": "not_run",
            "decision": "block" if blocked else "disabled",
            "would_block": blocked,
            "grounding_guard_would_block": blocked,
        }
        metadata = {
            "json_parse_valid": parse_valid,
            "grounding_guard": report,
            "director_runtime": {"schema": "dg-director-runtime/1", "sentinel": "preserve"},
            "director_cache": {"schema": "dg-director-cache/2", "sentinel": "preserve"},
        }
        if not parse_valid:
            metadata.update(
                {
                    "plain_text_salvage": True,
                    "salvage_warning": "invalid JSON was salvaged",
                    "json_parse_warning": "model output was invalid JSON",
                    "template_used_for_missing_fields": True,
                    "used_template_fallback": False,
                }
            )
        packet = {
            "ltx_prompt": "",
            "ideogram_prompt": "",
            "metadata": metadata,
            "minimax_h3_prompt": "HOST_VALID" if parse_valid else "HOST_FALLBACK",
            "scene_segments": [],
        }
        return (
            json.dumps(packet),
            json.dumps(metadata),
            "not_run",
            json.dumps(report),
        )

    def test_director_transport_repair_preserves_host_governance_and_rejects_duplicate_keys(self):
        raw = r'''<|final_json|>
{"ltx_prompt":"","ideogram_prompt":"","metadata":{},"minimax_h3_prompt":"subject_definitions:\n<Subject 1> An adult performer with consistent face hair wardrobe and build.\n<Subject 2> A slim orange beverage can with fixed geometry color label and material.\n\nsummary:\n[keyframe completion] A governed advertisement.\n\nretention_analysis:\n<Subject 1>: fully_preserved - performer.\n<Subject 2>: fully_preserved - package.\n\ndetail_description:\nBright commercial photography.\n[Shot 1] The performer presents the package.\n\noverall_soundscape:\nA can opens.\n\nnon_diegetic_music:\An upbeat score.","scene_segments":[]}
<|end_final_json|>'''
        final_json, metadata_json, status, report_json = self._director_host_inputs(parse_valid=False)
        references = build_upstream()[1]
        result = advertising.AdvertisementDirectorPacketRepair().repair(
            final_json,
            raw,
            metadata_json,
            status,
            report_json,
            references,
        )
        payload = json.loads(result[0])
        report = json.loads(result[1])
        self.assertIn(r"non_diegetic_music:\An upbeat score.", payload["minimax_h3_prompt"])
        self.assertIn("Reference sources: <Picture 1>, <Picture 2>.", payload["minimax_h3_prompt"])
        self.assertIn("Reference sources: <Picture 3>.", payload["minimax_h3_prompt"])
        self.assertEqual(report["invalid_escape_repair_count"], 1)
        self.assertFalse(report["creative_content_synthesized"])
        self.assertEqual(report["route"], "raw_transport_recovery")
        self.assertEqual(payload["metadata"]["director_runtime"]["sentinel"], "preserve")
        self.assertEqual(payload["metadata"]["director_cache"]["sentinel"], "preserve")
        self.assertEqual(payload["metadata"]["grounding_guard"], json.loads(report_json))
        self.assertTrue(payload["metadata"]["json_parse_valid"])
        self.assertNotIn("plain_text_salvage", payload["metadata"])
        self.assertTrue(result[-1])
        manifest = json.loads(references)["reference_manifest"]
        normalized = director_core._normalize_minimax_h3_prompt(
            payload["minimax_h3_prompt"],
            "ref2va",
        )
        normalized, _repairs = director_core._repair_minimax_h3_ref2va_contract_delimiters(
            normalized
        )
        normalized = director_core._repair_minimax_h3_ref2va_structure(
            normalized,
            "Create the governed advertisement.",
            manifest,
            30.0,
        )
        sections = director_core._minimax_h3_ref_sections(normalized)
        self.assertIsNotNone(sections)
        self.assertEqual(director_core._minimax_h3_ref2va_subject_count(normalized), 2)
        self.assertNotIn("fully_preserved - (", sections["retention_analysis"])
        self.assertIn("fully_preserved - performer.", sections["retention_analysis"])
        self.assertIn("fully_preserved - package.", sections["retention_analysis"])

        duplicate = r'<|final_json|>{"minimax_h3_prompt":"one\A","minimax_h3_prompt":"two"}<|end_final_json|>'
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            advertising.AdvertisementDirectorPacketRepair().repair(
                final_json,
                duplicate,
                metadata_json,
                status,
                report_json,
                references,
            )
        for nonfinite in ("1e999", "-1e999", "NaN", "Infinity"):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                advertising._strict_director_packet_json(f'{{"value":{nonfinite}}}')

        stale_references = json.loads(references)
        stale_references["picture_slots"][0]["subject_tag"] = "<Subject 2>"
        with self.assertRaisesRegex(ValueError, "hashed manifest authority"):
            advertising.AdvertisementDirectorPacketRepair().repair(
                final_json,
                raw,
                metadata_json,
                status,
                report_json,
                json.dumps(stale_references),
            )

    def test_director_packet_adapter_uses_host_valid_packet_and_preserves_grounding_block(self):
        references = build_upstream()[1]
        valid_final, valid_metadata, status, valid_report = self._director_host_inputs(parse_valid=True)
        passthrough = advertising.AdvertisementDirectorPacketRepair().repair(
            valid_final,
            "malicious raw text that must not replace the host packet",
            valid_metadata,
            status,
            valid_report,
            references,
        )
        passthrough_packet = json.loads(passthrough[0])
        self.assertEqual(passthrough_packet["minimax_h3_prompt"], "HOST_VALID")
        self.assertEqual(json.loads(passthrough[1])["route"], "host_final_json_passthrough")

        blocked_final, blocked_metadata, blocked_status, blocked_report = self._director_host_inputs(
            parse_valid=False,
            blocked=True,
        )
        blocked = advertising.AdvertisementDirectorPacketRepair().repair(
            blocked_final,
            r'<|final_json|>{"minimax_h3_prompt":"apparently valid\A"}<|end_final_json|>',
            blocked_metadata,
            blocked_status,
            blocked_report,
            references,
        )
        blocked_packet = json.loads(blocked[0])
        self.assertEqual(blocked_packet["minimax_h3_prompt"], "HOST_FALLBACK")
        self.assertFalse(blocked[-1])
        self.assertFalse(blocked_packet["metadata"]["json_parse_valid"])
        self.assertTrue(blocked_packet["metadata"]["grounding_guard"]["would_block"])
        self.assertEqual(json.loads(blocked[1])["route"], "host_grounding_block_preserved")

        tampered_metadata = json.loads(valid_metadata)
        tampered_metadata["director_runtime"]["sentinel"] = "changed"
        with self.assertRaisesRegex(ValueError, "does not match"):
            advertising.AdvertisementDirectorPacketRepair().repair(
                valid_final,
                "unused",
                json.dumps(tampered_metadata),
                status,
                valid_report,
                references,
            )

        for parse_valid, ambiguous_value in (
            (True, None),
            (True, 1),
            (False, None),
            (False, 0),
        ):
            base_final, base_metadata, base_status, base_report = self._director_host_inputs(
                parse_valid=parse_valid
            )
            packet = json.loads(base_final)
            metadata = json.loads(base_metadata)
            metadata["model_output_json_parse_valid"] = ambiguous_value
            packet["metadata"] = metadata
            with self.assertRaisesRegex(ValueError, "JSON boolean"):
                advertising.AdvertisementDirectorPacketRepair().repair(
                    json.dumps(packet),
                    "unused",
                    json.dumps(metadata),
                    base_status,
                    base_report,
                    references,
                )

        for flag_name in ("plain_text_salvage", "used_template_fallback"):
            base_final, base_metadata, base_status, base_report = self._director_host_inputs(
                parse_valid=False
            )
            packet = json.loads(base_final)
            metadata = json.loads(base_metadata)
            metadata[flag_name] = 1
            packet["metadata"] = metadata
            with self.assertRaisesRegex(ValueError, "JSON boolean"):
                advertising.AdvertisementDirectorPacketRepair().repair(
                    json.dumps(packet),
                    "unused",
                    json.dumps(metadata),
                    base_status,
                    base_report,
                    references,
                )

        strict_final, strict_metadata, strict_status, strict_report = self._director_host_inputs(
            parse_valid=False,
            blocked=True,
        )
        numeric_metadata = json.loads(strict_metadata)
        numeric_report = json.loads(strict_report)
        numeric_metadata["grounding_guard"]["would_block"] = 1
        numeric_metadata["grounding_guard"]["grounding_guard_would_block"] = 1
        numeric_report["would_block"] = 1
        numeric_report["grounding_guard_would_block"] = 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            advertising.AdvertisementDirectorPacketRepair().repair(
                strict_final,
                "unused",
                json.dumps(numeric_metadata),
                strict_status,
                json.dumps(numeric_report),
                references,
            )
        numeric_packet = json.loads(strict_final)
        numeric_packet["metadata"] = numeric_metadata
        with self.assertRaisesRegex(ValueError, "JSON booleans"):
            advertising.AdvertisementDirectorPacketRepair().repair(
                json.dumps(numeric_packet),
                "unused",
                json.dumps(numeric_metadata),
                strict_status,
                json.dumps(numeric_report),
                references,
            )

    def test_saved_ui_exposes_only_relay_seams_supported_by_the_shipped_graph(self):
        master_options = advertising.AdvertisementMasterContract.INPUT_TYPES()["required"]["boundary_mode"][0]
        seam_options = advertising.AdvertisementPlanningDefaults.INPUT_TYPES()["required"]["seam_style"][0]
        self.assertEqual(master_options, ["relay_continuity"])
        self.assertEqual(seam_options, ["Campaign default"])
        self.assertEqual(
            advertising.AdvertisementMultiShotPlanner.INPUT_TYPES()["required"]["performance_mode"][0],
            ["Dance / music sync", "Lyrics + lip sync", "Natural / audio-led sync"],
        )

    def test_contract_and_plan_embedded_hashes_are_enforced(self):
        contract_json, result = build_plan()
        contract = json.loads(contract_json)
        contract["message"] = "tampered"
        with self.assertRaisesRegex(ValueError, "contract_sha256"):
            advertising._contract(json.dumps(contract))
        plan = json.loads(result[0])
        plan["source_native_shot_count"] = 7
        with self.assertRaisesRegex(ValueError, "plan_sha256"):
            advertising._plan(json.dumps(plan))

    def test_tensor_hash_preserves_digest_without_materializing_multi_gigabyte_bytes(self):
        value = torch.arange(48, dtype=torch.float32).reshape(2, 2, 4, 3)
        header = advertising._json(
            {"shape": list(value.shape), "dtype": "float32"}
        ).encode("utf-8")
        expected = hashlib.sha256(header + value.numpy().tobytes(order="C")).hexdigest()
        self.assertEqual(advertising._tensor_sha256(value), expected)
        self.assertNotIn(".tobytes", inspect.getsource(advertising._tensor_sha256))

    def test_30_seconds_uses_two_lanes_and_preserves_eight_native_shots_and_four_roles(self):
        contract_json, result = build_plan()
        contract = json.loads(contract_json)
        plan = json.loads(result[0])
        self.assertEqual(contract["h3_planning"]["generation_lane_count"], 2)
        self.assertEqual(result[3], 2)
        self.assertEqual(plan["source_native_shot_count"], 8)
        self.assertEqual(
            sorted({value for lane in plan["lanes"] for value in lane["source_native_shot_indices"]}),
            list(range(1, 9)),
        )
        self.assertEqual([lane["duration_seconds"] for lane in plan["lanes"]], [15.0, 15.0])
        self.assertEqual(plan["lanes"][0]["reference_picture_tags"], ["<Picture 1>", "<Picture 2>", "<Picture 3>"])
        self.assertEqual(plan["lanes"][1]["reference_picture_tags"], ["<Picture 1>", "<Picture 2>", "<Picture 3>", "<Picture 4>"])
        self.assertIn("<Subject 2>", result[4])
        self.assertIn("<Picture 3>", result[9])
        self.assertTrue(all(item["product_action"] for lane in plan["lanes"] for item in lane["product_presence"]))

    def test_plain_picture_citations_in_governed_subject_rows_are_canonicalized(self):
        prompt = EIGHT_SHOT_PROMPT.replace(
            "<Subject 1>: the same performer jointly defined by <Picture 1> and <Picture 2>.",
            "<Subject 1>: (Picture 1, Picture 2): the same performer.",
        ).replace(
            "<Subject 2>: the same beverage package defined by <Picture 3>.",
            "<Subject 2>: (Picture 3): the same beverage package.",
        )
        contract = build_contract()[0]
        defaults = advertising.AdvertisementPlanningDefaults().build(
            contract,
            "Keep the can visible.",
            "Campaign default",
        )
        result = advertising.AdvertisementMultiShotPlanner().plan(
            contract,
            prompt,
            defaults[0],
            defaults[1],
            "Natural / audio-led sync",
        )
        plan = json.loads(result[0])
        self.assertIn("(<Picture 1>, <Picture 2>)", plan["lanes"][0]["prompt"])
        self.assertIn("(<Picture 3>)", plan["lanes"][0]["prompt"])

    def test_defaults_guarantee_hero_product_and_valid_typed_seam_state(self):
        contract = build_contract()[0]
        defaults = advertising.AdvertisementPlanningDefaults().build(contract, "Keep the can visible.", "Campaign default")
        shots = json.loads(defaults[0])
        states = json.loads(defaults[1])
        self.assertTrue(any(item["product_presence"] == "hero" for item in shots))
        self.assertTrue(all(item["product_action"] == "Keep the can visible." for item in shots))
        self.assertEqual(states[0]["exit_state"], states[1]["entry_state"])
        self.assertEqual(states[1]["boundary_mode"], "relay_continuity")
        self.assertTrue(defaults[4])

    def test_seam_mismatch_fails_closed(self):
        contract = build_contract()[0]
        defaults = advertising.AdvertisementPlanningDefaults().build(contract, "Keep the can visible.", "Campaign default")
        states = json.loads(defaults[1])
        states[1]["entry_state"]["composition"] = "wrong composition"
        with self.assertRaisesRegex(ValueError, "seam mismatch"):
            advertising.AdvertisementMultiShotPlanner().plan(
                contract,
                EIGHT_SHOT_PROMPT,
                defaults[0],
                json.dumps(states),
                "Natural / audio-led sync",
            )

    def test_relay_persists_exact_retained_tail_not_h3_padding(self):
        contract_json, result = build_plan()
        plan = json.loads(result[0])
        retained = plan["lanes"][0]["retained_master_frames"]
        images = torch.zeros((retained + 2, 8, 8, 3), dtype=torch.float32)
        images[retained - 1] = 0.375
        images[-1] = 0.875
        with tempfile.TemporaryDirectory() as temporary:
            persisted = advertising.AdvertisementRelayArtifact().persist(
                images,
                result[0],
                plan["campaign_id"],
                1,
                2,
                artifact_root=temporary,
            )
            manifest = json.loads(persisted[0])
            self.assertEqual(manifest["source_retained_frame_index"], retained - 1)
            loaded = advertising.AdvertisementRelayGate().load(
                persisted[0],
                result[0],
                plan["campaign_id"],
                2,
                artifact_root=temporary,
            )
            self.assertTrue(torch.equal(loaded[0], images[retained - 1 : retained]))
            self.assertTrue(Path(temporary, manifest["filename"]).is_file())
            with self.assertRaisesRegex(ValueError, "campaign identity"):
                advertising.AdvertisementRelayGate().load(
                    persisted[0], result[0], "f" * 64, 2, artifact_root=temporary
                )

    def test_outputs_validate_against_advertisement_schemas(self):
        contract_json, result = build_plan()
        for filename, payload in (
            ("advertisement_master_contract.schema.json", json.loads(contract_json)),
            ("advertisement_h3_plan.schema.json", json.loads(result[0])),
        ):
            schema = json.loads((ROOT / "schemas" / filename).read_text(encoding="utf-8"))
            Draft202012Validator(schema).validate(payload)


if __name__ == "__main__":
    unittest.main()
