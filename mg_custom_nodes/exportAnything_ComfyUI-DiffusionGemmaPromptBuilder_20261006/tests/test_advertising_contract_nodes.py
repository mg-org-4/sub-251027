from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import torch

import advertising_contract_nodes as contracts


ROOT = Path(__file__).resolve().parents[1]


def build_campaign(**overrides):
    values = {
        "brand_name": "AFTERS",
        "product_name": "Sparkling Tea",
        "variant_name": "Yuzu + White Peach",
        "audience": "Trend-conscious adults in their twenties",
        "campaign_objective": "Build awareness for a sophisticated zero-proof social drink.",
        "headline": "STAY FOR THE AFTER.",
        "call_to_action": "FIND YOUR AFTER.",
        "claims_json": json.dumps(
            [
                {
                    "text": "Zero proof",
                    "status": "approved",
                    "evidence": "Approved formulation specification 2026-08",
                },
                {"text": "Improves focus", "status": "prohibited", "evidence": ""},
            ]
        ),
        "production_duration_seconds": 30.0,
        "aspect_ratio": "9:16",
        "end_card_duration_seconds": 3.0,
        "deliverables_json": '{"primary":"9:16","adaptations":["1:1","16:9"],"cutdowns_seconds":[15,6]}',
        "campaign_name": "Stay for the After",
        "legal_line": "Product shown is a fictional concept.",
        "offer_text": "",
        "price_text": "",
    }
    values.update(overrides)
    return contracts.DiffusionGemmaAdvertisementCampaignContract().build(**values)


def build_soundtrack(**overrides):
    values = {
        "content_mode": "Instrumental",
        "genre_style": "percussive alternative pop, not EDM",
        "mood": "bright, spontaneous, tactile",
        "instrumentation": "rubbery bass, dry drums, handclaps, muted guitar",
        "target_duration_seconds": 30.0,
        "bpm": 105.0,
        "time_signature": "4",
        "language": "unknown",
        "lyrics": "",
        "voice_over_policy": "Separate non-diegetic VO",
        "arrangement_notes": "Clean edit points near 6, 15, and 27 seconds.",
        "do_not_sound_like": "four-on-the-floor nightclub EDM",
    }
    values.update(overrides)
    return contracts.DiffusionGemmaAdvertisementSoundtrackContract().build(**values)


class AdvertisementCampaignContractTests(unittest.TestCase):
    def test_workflow_controls_are_one_safe_authority_and_derive_tail_cutdowns(self):
        node = contracts.DiffusionGemmaAdvertisementWorkflowControls()
        result = node.resolve("30 seconds", 8, "Natural / audio-led sync")
        self.assertEqual(result[:5], (30.0, 8, "Natural / audio-led sync", 15.0, 24.0))
        self.assertEqual(json.loads(result[5])["cutdowns_seconds"], [15, 6])
        self.assertEqual(result[6], "9:16")
        self.assertEqual(
            result[7:15],
            (
                "ref2va",
                "custom",
                0,
                "auto_scene_audio",
                "off",
                0.0,
                "MiniMax H3 Ref2VA",
                15.0,
            ),
        )
        self.assertEqual(result[15], "9:16")
        self.assertTrue(result[-1])
        self.assertEqual(
            contracts.DiffusionGemmaAdvertisementWorkflowControls.INPUT_TYPES()["required"]["master_duration"][0],
            ["30 seconds"],
        )
        self.assertEqual(
            contracts.DiffusionGemmaAdvertisementWorkflowControls.INPUT_TYPES()["required"]["performance_mode"][0],
            ["Natural / audio-led sync", "Dance / music sync"],
        )
        self.assertEqual(
            contracts.DiffusionGemmaAdvertisementWorkflowControls.RETURN_TYPES[2],
            ["Dance / music sync", "Lyrics + lip sync", "Natural / audio-led sync"],
        )
        self.assertEqual(
            contracts.DiffusionGemmaAdvertisementWorkflowControls.RETURN_TYPES[7:15],
            (
                ["t2va", "ref2va"],
                ["auto", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "custom"],
                "INT",
                ["auto_scene_audio", "explicit_sound_design", "visual_only"],
                ["auto", "required", "off"],
                "FLOAT",
                "STRING",
                "FLOAT",
            ),
        )
        self.assertEqual(
            contracts.DiffusionGemmaAdvertisementWorkflowControls.RETURN_TYPES[15],
            "STRING",
        )

    def test_exact_copy_claim_governance_timing_and_delivery_are_locked(self):
        result = build_campaign()
        payload = json.loads(result[0])
        exact_copy = json.loads(result[2])
        self.assertTrue(result[-1])
        self.assertEqual(payload["schema"], contracts.CAMPAIGN_SCHEMA)
        self.assertEqual(payload["timeline"]["end_card"]["start_seconds"], 27.0)
        self.assertEqual(payload["deliverables"]["render_status"], "requested_not_rendered")
        self.assertEqual(exact_copy["approved_claims"], ["Zero proof"])
        self.assertNotIn("Improves focus", result[1])
        self.assertEqual(
            payload["exact_copy_sha256"],
            hashlib.sha256(result[2].encode("utf-8")).hexdigest(),
        )

    def test_exact_copy_is_normalized_once_without_changing_brand_case(self):
        result = build_campaign(
            brand_name="eBay   Labs",
            headline="Stay   curious",
            call_to_action="Try   it",
        )
        payload = json.loads(result[0])
        exact = json.loads(result[2])
        self.assertEqual(payload["brand"]["brand_name"], "eBay Labs")
        self.assertEqual(exact["headline"], "Stay curious")
        self.assertEqual(exact["call_to_action"], "Try it")

    def test_approved_claim_without_evidence_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "no substantiation"):
            build_campaign(
                claims_json='[{"text":"Clinically proven energy","status":"approved","evidence":""}]'
            )

    def test_end_card_and_cutdowns_must_fit_inside_master(self):
        with self.assertRaisesRegex(ValueError, "24 fps"):
            build_campaign(production_duration_seconds=30.1)
        with self.assertRaisesRegex(ValueError, "must be from"):
            build_campaign(end_card_duration_seconds=0.0)
        with self.assertRaisesRegex(ValueError, "shorter than the production"):
            build_campaign(
                production_duration_seconds=10.0,
                end_card_duration_seconds=10.0,
                deliverables_json='{"primary":"9:16","adaptations":[],"cutdowns_seconds":[6]}',
            )
        with self.assertRaisesRegex(ValueError, "cutdown must be shorter"):
            build_campaign(
                deliverables_json='{"primary":"9:16","adaptations":[],"cutdowns_seconds":[30]}'
            )


class AdvertisementReferenceContractTests(unittest.TestCase):
    def test_saved_ui_exposes_only_the_reference_topology_the_shipped_graph_supports(self):
        options = contracts.DiffusionGemmaAdvertisementReferenceContract.INPUT_TYPES()["required"]["slot_policy"][0]
        self.assertEqual(options, ["2 performer + 1 product + relay"])

    def test_default_uses_two_performer_assets_product_three_and_relay_four(self):
        result = contracts.DiffusionGemmaAdvertisementReferenceContract().build(
            slot_policy=contracts.REFERENCE_SLOT_POLICIES[0],
            performer_description="A clearly adult performer in a cobalt jacket.",
            product_description="A slim warm-bone can with a citron stripe.",
            product_retention_attributes="slim geometry, warm-bone body, citron stripe, cobalt wordmark layout",
            product_reference_kind="product contact sheet",
            package_copy_json='["AFTERS","YUZU + WHITE PEACH"]',
        )
        payload = json.loads(result[0])
        self.assertEqual(result[2:6], (2, 2, "<Picture 3>", "<Picture 4>"))
        self.assertEqual(
            [slot["role"] for slot in payload["picture_slots"]],
            ["performer_primary", "performer_supplemental", "product_package"],
        )
        self.assertEqual(payload["subjects"][1]["subject_tag"], "<Subject 2>")
        self.assertTrue(payload["relay"]["enabled"])
        self.assertIn("continuity_relay", payload["relay"]["manifest_reservation"])
        self.assertIn("never performer identity or product", payload["relay"]["manifest_reservation"])
        self.assertIn("panel multiplicity", payload["picture_slots"][2]["non_transfer"])

    def test_semantic_parser_uses_annotations_not_wording_passwords(self):
        first = (
            "<Picture 1>: [dg:identity,appearance] "
            "[ad:role=performer_primary;subject=subject_1;retention=fully_preserved] same adult woman only"
        )
        second = (
            "<Picture 1>: [dg:identity,appearance] "
            "[ad:role=performer_primary;subject=subject_1;retention=fully_preserved] hero portrait with completely different prose"
        )
        keys = ("tag", "role", "subject", "retention")
        parsed_first = contracts.parse_advertisement_reference_manifest(first)[0]
        parsed_second = contracts.parse_advertisement_reference_manifest(second)[0]
        self.assertEqual(
            {key: parsed_first[key] for key in keys},
            {key: parsed_second[key] for key in keys},
        )
        with self.assertRaisesRegex(ValueError, "exactly one \[ad"):
            contracts.parse_advertisement_reference_manifest(
                "<Picture 1>: same subject identity with no typed annotation"
            )

    def test_reference_asset_prep_keeps_three_roles_separate_and_hash_locked(self):
        image = torch.linspace(0.0, 1.0, steps=96 * 64 * 3).reshape(1, 96, 64, 3)
        performer_sheet = torch.flip(image, dims=(2,))
        product_sheet = 1.0 - image
        result = contracts.DiffusionGemmaAdvertisementReferenceAssetPrep().prepare(
            image,
            performer_sheet,
            product_sheet,
            256,
            256,
            0.5,
            0.6,
            0.25,
        )
        report = json.loads(result[7])
        self.assertEqual(result[6].shape, (3, 256, 256, 3))
        self.assertEqual(len({result[3], result[4], result[5]}), 3)
        self.assertEqual(
            [asset["role"] for asset in report["assets"]],
            ["performer_hero", "performer_sheet", "product_sheet"],
        )
        self.assertTrue(all(asset["prepared_role_sha256"] for asset in report["assets"]))
        self.assertTrue(report["product_budget"]["independent_from_performer_budget"])
        self.assertFalse(report["identity_product_combined_sheet"])
        repeat = contracts.DiffusionGemmaAdvertisementReferenceAssetPrep().prepare(
            image, performer_sheet, product_sheet, 256, 256, 0.5, 0.6, 0.25
        )
        self.assertEqual(result[3:6], repeat[3:6])

        with self.assertRaisesRegex(ValueError, "three distinct source images"):
            contracts.DiffusionGemmaAdvertisementReferenceAssetPrep().prepare(
                image, image, image, 256, 256, 0.5, 0.6, 0.25
            )


class AdvertisementSoundtrackContractTests(unittest.TestCase):
    def test_saved_ui_hides_voice_over_until_the_graph_has_a_vo_input(self):
        options = contracts.DiffusionGemmaAdvertisementSoundtrackContract.INPUT_TYPES()["required"]["voice_over_policy"][0]
        self.assertEqual(options, ["None"])

    def test_instrumental_music3_caption_is_structured_and_has_headroom(self):
        result = build_soundtrack()
        payload = json.loads(result[0])
        self.assertEqual(
            result[2],
            "[Intro]\n\n[Instrumental]\n\n[Bridge]\n\n[Instrumental]\n\n[Outro]",
        )
        self.assertEqual(result[3], "instrumental")
        self.assertEqual(result[5], 35.0)
        self.assertIn("Global Metadata:", result[1])
        self.assertIn("Vocal Details:", result[1])
        self.assertIn("Arrangement:", result[1])
        self.assertIn("Selectable advertisement master: 30 seconds", result[1])
        self.assertEqual(payload["music3"]["generation_duration_seconds"], 35.0)
        self.assertFalse(payload["voice_over"]["included_in_motion_guide"])

    def test_vocal_and_auto_modes_emit_tagged_lyrics(self):
        vocal = build_soundtrack(
            content_mode="Vocal",
            lyrics="Plans changed, perfect.",
            language="en",
            voice_over_policy="None",
        )
        self.assertTrue(vocal[2].startswith("[Verse]\n"))
        self.assertEqual(vocal[3], "vocal")
        auto = build_soundtrack(
            content_mode="Auto",
            lyrics="[Chorus]\nStay for the after.",
            language="en",
            voice_over_policy="None",
        )
        self.assertEqual(auto[3], "vocal")
        with self.assertRaisesRegex(ValueError, "cannot contain sung lyrics"):
            build_soundtrack(content_mode="Instrumental", lyrics="Sing this line")

    def test_music3_adapter_preserves_governed_prompt_and_duration(self):
        soundtrack = build_soundtrack()
        adapted = contracts.DiffusionGemmaAdvertisementMusic3PromptAdapter().adapt(
            soundtrack[0]
        )
        self.assertEqual(adapted[:5], (soundtrack[1], soundtrack[2], "instrumental", 105.0, 35.0))
        self.assertEqual(adapted[5], 105)
        self.assertEqual(contracts.DiffusionGemmaAdvertisementMusic3PromptAdapter.RETURN_TYPES[5], "INT")
        unspecified = contracts.DiffusionGemmaAdvertisementMusic3PromptAdapter().adapt(
            build_soundtrack(bpm=0.0)[0]
        )
        self.assertEqual(unspecified[3], 0.0)
        self.assertEqual(unspecified[5], 120)


class LegacyArtifactImmutabilityTests(unittest.TestCase):
    EXPECTED = {
        "audio_production_nodes.py": "3f0d1f5b3f5e002d6944a6eb914c6a46c0fe4679828f3546fd672a8b1a700e21",
        "production_planning_nodes.py": "7ae98e808062aab16d8a470c5c8143c9881333354167b79a4d88a3110ab6e1e8",
        "nodes.py": "fb474fd2848bbda5df82b2e5d1568125bee04efa7faaf30dfdb41db2be023452",
        "examples/15_minimax_h3_ref2va_music_video_v6.json": "594f6aef94531f87b74659e1e8004cccdff00716d0cfe263cee6bf2149545c83",
    }

    def test_advertisement_extension_did_not_mutate_known_good_legacy_files(self):
        for relative, expected in self.EXPECTED.items():
            with self.subTest(relative=relative):
                actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
