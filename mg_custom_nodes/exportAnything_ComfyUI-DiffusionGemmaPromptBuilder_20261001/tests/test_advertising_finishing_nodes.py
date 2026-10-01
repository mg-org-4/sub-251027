from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import torch
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for path in (ROOT, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import advertising_finishing_nodes as finishing
from audio_production_nodes import waveform_sha256
from test_advertising_planning_nodes import build_plan


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def revised_contract(contract_json: str, adaptations: list[str]) -> str:
    contract = json.loads(contract_json)
    contract["requested_deliverables"]["adaptations"] = adaptations
    contract.pop("campaign_id", None)
    contract.pop("contract_sha256", None)
    campaign_id = hashlib.sha256(canonical(contract).encode("utf-8")).hexdigest()
    contract["campaign_id"] = campaign_id
    contract["contract_sha256"] = hashlib.sha256(canonical(contract).encode("utf-8")).hexdigest()
    return canonical(contract)


def revised_exact_copy(contract_json: str, **changes: object) -> str:
    contract = json.loads(contract_json)
    contract["mandatory_copy"].update(changes)
    if "call_to_action" in changes:
        contract["call_to_action"] = str(changes["call_to_action"])
    contract.pop("campaign_id", None)
    contract.pop("contract_sha256", None)
    campaign_id = hashlib.sha256(canonical(contract).encode("utf-8")).hexdigest()
    contract["campaign_id"] = campaign_id
    contract["contract_sha256"] = hashlib.sha256(canonical(contract).encode("utf-8")).hexdigest()
    return canonical(contract)


def revised_cutdowns(contract_json: str, cutdowns: list[int]) -> str:
    contract = json.loads(contract_json)
    contract["requested_deliverables"]["cutdowns_seconds"] = cutdowns
    contract.pop("campaign_id", None)
    contract.pop("contract_sha256", None)
    campaign_id = hashlib.sha256(canonical(contract).encode("utf-8")).hexdigest()
    contract["campaign_id"] = campaign_id
    contract["contract_sha256"] = hashlib.sha256(canonical(contract).encode("utf-8")).hexdigest()
    return canonical(contract)


class AdvertisementFinishingTests(unittest.TestCase):
    sample_rate = 240

    def setUp(self):
        self.audio = {
            "waveform": torch.linspace(-0.5, 0.5, 30 * self.sample_rate, dtype=torch.float32)
            .view(1, 1, -1)
            .repeat(1, 2, 1),
            "sample_rate": self.sample_rate,
        }
        audio_hash = waveform_sha256(self.audio)
        self.contract_json, plan_result = build_plan(audio_hash=audio_hash)
        self.plan_json = plan_result[0]
        self.plan = json.loads(self.plan_json)
        lane_images = []
        for lane_index, lane in enumerate(self.plan["lanes"], start=1):
            generated = int(lane["generated_h3_frames"])
            lane_images.append(torch.full((generated, 112, 64, 3), lane_index / 4.0))
        assembled = finishing.AdvertisementMasterAssembler().assemble(
            self.contract_json,
            self.plan_json,
            self.audio,
            lane_1_images=lane_images[0],
            lane_2_images=lane_images[1],
        )
        self.assembled = assembled[0]
        self.assembly_report = assembled[2]

    def _finish(self):
        end_card = finishing.EndCardRenderer().render(
            self.contract_json,
            64,
            112,
            "#10141f",
            "#ffffff",
            "#76f7c5",
        )
        finished = finishing.AdvertisementMasterFinisher().finish(
            self.assembled,
            self.audio,
            end_card[0],
            end_card[1],
            self.contract_json,
            self.assembly_report,
        )
        return end_card, finished

    def test_master_assembly_verifies_identity_audio_and_exact_frames(self):
        report = json.loads(self.assembly_report)
        self.assertEqual(self.assembled.shape[0], 30 * 24)
        self.assertEqual(report["exact_master_frames"], 720)
        self.assertEqual([row["retained_frames"] for row in report["lanes"]], [360, 360])
        self.assertTrue(all(row["discarded_padding_frames"] >= 0 for row in report["lanes"]))
        self.assertTrue(report["pixel_aspect_evidence"]["matched"])
        mismatched_plan = json.loads(self.plan_json)
        mismatched_plan["campaign_id"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "plan_sha256"):
            finishing.AdvertisementMasterAssembler().assemble(
                self.contract_json,
                json.dumps(mismatched_plan),
                self.audio,
                lane_1_images=torch.zeros((362, 64, 64, 3)),
                lane_2_images=torch.zeros((362, 64, 64, 3)),
            )

    def test_master_assembly_rejects_pixels_that_disagree_with_contract_aspect(self):
        lanes = []
        for lane in self.plan["lanes"]:
            lanes.append(torch.zeros((int(lane["generated_h3_frames"]), 64, 64, 3)))
        with self.assertRaisesRegex(ValueError, "does not match the contracted 9:16"):
            finishing.AdvertisementMasterAssembler().assemble(
                self.contract_json,
                self.plan_json,
                self.audio,
                lane_1_images=lanes[0],
                lane_2_images=lanes[1],
            )

    def test_end_card_is_exactly_72_frames_and_finisher_replaces_tail(self):
        end_card, finished = self._finish()
        report = json.loads(end_card[1])
        finish_report = json.loads(finished[2])
        self.assertEqual(end_card[0].shape[0], 72)
        self.assertEqual(report["frame_count"], 72)
        self.assertEqual(report["rendered_copy"]["headline"], "Open something brighter")
        self.assertEqual(report["rendered_copy"]["variant_name"], "Yuzu Mint")
        self.assertEqual(report["rendered_copy"]["call_to_action"], "Find your flavor")
        self.assertEqual(finished[0].shape[0], 720)
        self.assertEqual(finish_report["end_card_replaced_tail_frames"], 72)
        self.assertFalse(finish_report["duration_changed"])
        self.assertTrue(torch.equal(finished[0][-72:], end_card[0]))

    def test_finisher_rejects_stale_reports_and_mismatched_pixels(self):
        end_card = finishing.EndCardRenderer().render(
            self.contract_json, 64, 112, "#10141f", "#ffffff", "#76f7c5"
        )
        stale = json.loads(self.assembly_report)
        stale["exact_master_frames"] -= 1
        with self.assertRaisesRegex(ValueError, "report_sha256"):
            finishing.AdvertisementMasterFinisher().finish(
                self.assembled,
                self.audio,
                end_card[0],
                end_card[1],
                self.contract_json,
                json.dumps(stale),
            )
        changed = self.assembled.clone()
        changed[0, 0, 0, 0] = 1.0 - changed[0, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "pixels do not match"):
            finishing.AdvertisementMasterFinisher().finish(
                changed,
                self.audio,
                end_card[0],
                end_card[1],
                self.contract_json,
                self.assembly_report,
            )

    def test_end_card_renders_all_nonempty_exact_copy_in_a_recorded_hierarchy(self):
        contract = revised_exact_copy(
            self.contract_json,
            offer_text="Two cans for the first tasting",
            price_text="$5",
            legal_line="Limited availability. See site for details.",
            approved_claims=["Contains zero grams of sugar."],
        )
        node = finishing.EndCardRenderer()
        first = node.render(contract, 96, 160, "#10141f", "#ffffff", "#76f7c5")
        second = node.render(contract, 96, 160, "#10141f", "#ffffff", "#76f7c5")
        report = json.loads(first[1])
        rendered = report["rendered_copy"]
        self.assertEqual(first[0].shape, (72, 160, 96, 3))
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertEqual(first[1], second[1])
        self.assertEqual(rendered["headline"], "Open something brighter")
        self.assertEqual(rendered["offer_text"], "Two cans for the first tasting")
        self.assertEqual(rendered["price_text"], "$5")
        self.assertEqual(rendered["legal_line"], "Limited availability. See site for details.")
        self.assertEqual(rendered["approved_claims"], ["Contains zero grams of sugar."])
        hierarchy_fields = [row["field"] for row in report["copy_hierarchy"]]
        self.assertEqual(
            hierarchy_fields,
            [
                "brand_name",
                "product_name+variant_name",
                "headline",
                "offer_text",
                "price_text",
                "call_to_action",
                "approved_claims",
                "legal_line",
            ],
        )
        self.assertEqual(report["copy_fit_policy"], "deterministic_scale_and_pixel_wrap_fail_closed_no_clipping")

    def test_end_card_preserves_authored_brand_case_in_pixels(self):
        mixed = revised_exact_copy(self.contract_json)
        mixed_payload = json.loads(mixed)
        mixed_payload["brand_name"] = "eBay"
        mixed_payload["mandatory_copy"]["brand_name"] = "eBay"
        mixed_payload.pop("campaign_id", None)
        mixed_payload.pop("contract_sha256", None)
        mixed_payload["campaign_id"] = hashlib.sha256(canonical(mixed_payload).encode("utf-8")).hexdigest()
        mixed_payload["contract_sha256"] = hashlib.sha256(canonical(mixed_payload).encode("utf-8")).hexdigest()
        upper_payload = json.loads(canonical(mixed_payload))
        upper_payload["brand_name"] = "EBAY"
        upper_payload["mandatory_copy"]["brand_name"] = "EBAY"
        upper_payload.pop("campaign_id", None)
        upper_payload.pop("contract_sha256", None)
        upper_payload["campaign_id"] = hashlib.sha256(canonical(upper_payload).encode("utf-8")).hexdigest()
        upper_payload["contract_sha256"] = hashlib.sha256(canonical(upper_payload).encode("utf-8")).hexdigest()
        node = finishing.EndCardRenderer()
        mixed_images = node.render(canonical(mixed_payload), 96, 160, "#10141f", "#ffffff", "#76f7c5")[0]
        upper_images = node.render(canonical(upper_payload), 96, 160, "#10141f", "#ffffff", "#76f7c5")[0]
        self.assertFalse(torch.equal(mixed_images, upper_images))

    def test_end_card_copy_overflow_fails_closed_without_changing_timing(self):
        contract = revised_exact_copy(self.contract_json, legal_line="LEGAL" * 300)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            finishing.EndCardRenderer().render(
                contract, 64, 64, "#10141f", "#ffffff", "#76f7c5"
            )

    def test_end_card_rejects_oversized_tensor_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "2 GiB allocation ceiling"):
            finishing.EndCardRenderer().render(
                self.contract_json,
                4096,
                4096,
                "#10141f",
                "#ffffff",
                "#76f7c5",
            )

    def test_real_15_and_6_second_cutdowns_slice_exact_frames_and_audio(self):
        _end_card, finished = self._finish()
        result = finishing.AdvertisementCutdownRenderer().render(
            finished[0],
            finished[1],
            self.contract_json,
            0.0,
            24.0,
        )
        manifest = json.loads(result[4])
        self.assertEqual(result[0].shape[0], 15 * 24)
        self.assertEqual(result[1]["waveform"].shape[-1], 15 * self.sample_rate)
        self.assertEqual(result[2].shape[0], 6 * 24)
        self.assertEqual(result[3]["waveform"].shape[-1], 6 * self.sample_rate)
        self.assertEqual([row["status"] for row in manifest["cutdowns"]], ["rendered", "rendered"])
        self.assertTrue(torch.equal(result[0], finished[0][: 15 * 24]))
        self.assertTrue(torch.equal(result[2], finished[0][24 * 24 : 30 * 24]))
        self.assertTrue(torch.equal(result[3]["waveform"], self.audio["waveform"][..., 24 * self.sample_rate :]))
        schema = json.loads((ROOT / "schemas" / "advertisement_delivery_manifest.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(manifest)

        with self.assertRaisesRegex(ValueError, "requested cutdowns"):
            finishing.AdvertisementCutdownRenderer().render(
                finished[0],
                finished[1],
                revised_cutdowns(self.contract_json, [10]),
                15.0,
                24.0,
            )

    def test_impossible_and_unrendered_adaptations_are_never_claimed_rendered(self):
        _end_card, finished = self._finish()
        contract = revised_contract(self.contract_json, ["9:16", "1:1", "cinema-circle"])
        adaptation_result = finishing.AdvertisementAdaptationStatus().evaluate(contract)
        self.assertFalse(adaptation_result[2])
        result = finishing.AdvertisementCutdownRenderer().render(
            finished[0], finished[1], contract, 0.0, 24.0, adaptation_result[0]
        )
        statuses = {row["aspect_ratio"]: row["status"] for row in json.loads(result[4])["adaptations"]}
        self.assertEqual(statuses["9:16"], "not_applicable_same_as_master")
        self.assertEqual(statuses["1:1"], "planned_not_rendered")
        self.assertEqual(statuses["cinema-circle"], "impossible_invalid_aspect")
        self.assertNotIn("rendered", statuses.values())
        schema = json.loads((ROOT / "schemas" / "advertisement_adaptation_status.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(json.loads(adaptation_result[0]))

    def test_media_qa_gate_distinguishes_pass_fail_and_not_measured(self):
        _end_card, finished = self._finish()
        delivery = finishing.AdvertisementCutdownRenderer().render(
            finished[0], finished[1], self.contract_json, 0.0, 24.0
        )[4]
        gate = finishing.MediaQAGate()
        required = '["product_identity","audio_sync"]'
        unknown = gate.evaluate(self.contract_json, delivery, '{"product_identity":"pass"}', required)
        self.assertEqual(unknown[1], "not_measured")
        self.assertFalse(unknown[3])
        failed = gate.evaluate(
            self.contract_json,
            delivery,
            '{"product_identity":"pass","audio_sync":{"status":"fail","evidence":"visible drift"}}',
            required,
        )
        self.assertEqual(failed[1], "fail")
        self.assertFalse(failed[3])
        passed = gate.evaluate(
            self.contract_json,
            delivery,
            json.dumps(
                {
                    name: {"status": "pass", "evidence": f"measured {name}"}
                    for name in finishing.MANDATORY_MEDIA_QA_CHECKS
                }
            ),
            required,
        )
        report = json.loads(passed[0])
        self.assertEqual(passed[1], "pass")
        self.assertTrue(passed[3])
        self.assertEqual(report["host_required_checks"], list(finishing.MANDATORY_MEDIA_QA_CHECKS))
        schema = json.loads((ROOT / "schemas" / "advertisement_media_qa.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(report)

        omitted = gate.evaluate(
            self.contract_json,
            delivery,
            '{"technical_integrity":{"status":"pass","evidence":"exact assembly"}}',
            '["technical_integrity"]',
        )
        self.assertEqual(omitted[1], "not_measured")
        self.assertFalse(omitted[3])
        self.assertEqual(
            {row["check"] for row in json.loads(omitted[0])["checks"]},
            set(finishing.MANDATORY_MEDIA_QA_CHECKS),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            gate.evaluate(
                self.contract_json,
                delivery,
                "{}",
                '["technical_integrity","technical_integrity"]',
            )

        bare_pass = gate.evaluate(
            self.contract_json,
            delivery,
            '{"product_identity":"pass","audio_sync":{"status":"pass","evidence":"measured"}}',
            required,
        )
        bare_report = json.loads(bare_pass[0])
        self.assertEqual(bare_pass[1], "not_measured")
        self.assertEqual(bare_report["checks"][0]["status"], "not_measured")
        self.assertIn("without evidence", bare_report["checks"][0]["evidence"])


if __name__ == "__main__":
    unittest.main()
