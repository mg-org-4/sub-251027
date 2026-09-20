from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import grounding_guard as guard


def valid_ledger(*facts: dict, status: str = "grounded") -> dict:
    return {
        "schema": guard.GROUNDING_LEDGER_SCHEMA_ID,
        "analysis_status": status,
        "observed_facts": list(facts),
        "inferred_facts": [],
        "creative_additions": [],
        "uncertainties": [],
        "grounding_failure_reasons": [],
    }


def fact(
    fact_id: str,
    claim: str,
    categories: list[str],
    evidence: list[dict],
    *,
    confidence: str = "high",
    typed_claims: list[dict] | None = None,
) -> dict:
    result = {
        "fact_id": fact_id,
        "claim": claim,
        "confidence": confidence,
        "categories": categories,
        "evidence": evidence,
    }
    if typed_claims is not None:
        result["typed_claims"] = typed_claims
    return result


class GroundingGuardConfigTests(unittest.TestCase):
    def test_disconnected_defaults_to_audit_and_native_checkpoint_profile(self) -> None:
        config = guard.normalize_grounding_guard_config()
        self.assertEqual(config.mode, "audit")
        self.assertTrue(config.retry_on_uncertain)
        self.assertEqual(config.evidence_token_budget, "auto")
        self.assertEqual(config.seed, 0)
        self.assertTrue(config.sampling_settings["adaptive_stopping"])
        self.assertEqual(config.sampling_settings["temperature_start"], 0.8)
        self.assertEqual(config.sampling_settings["temperature_end"], 0.4)

    def test_normalization_fails_invalid_values_toward_safe_defaults(self) -> None:
        config = guard.normalize_grounding_guard_config(
            {
                "mode": "disabled-by-typo",
                "retry_on_uncertain": "no",
                "sampling_profile": "mystery",
                "evidence_token_budget": "4096",
                "seed": -99,
                "trace_subfolder": "../escape",
                "external_evidence_json": {"schema": "example"},
            }
        )
        self.assertEqual(config.mode, "audit")
        self.assertFalse(config.retry_on_uncertain)
        self.assertEqual(config.sampling_profile, "checkpoint_defaults")
        self.assertEqual(config.evidence_token_budget, "auto")
        self.assertEqual(config.seed, 0)
        self.assertEqual(config.trace_subfolder, "diffusiongemma_grounding")
        self.assertEqual(json.loads(config.external_evidence_json)["schema"], "example")

    def test_evidence_token_budget_normalizes_to_bounded_canvas_aligned_choices(self) -> None:
        for supplied, expected in (
            ("768", "768"),
            (1024, "1024"),
            ("1280", "1280"),
            (True, "auto"),
            ("unbounded", "auto"),
        ):
            with self.subTest(supplied=supplied):
                config = guard.normalize_grounding_guard_config(
                    {"evidence_token_budget": supplied}
                )
                self.assertEqual(config.evidence_token_budget, expected)
                self.assertEqual(config.to_dict()["evidence_token_budget"], expected)

    def test_full_diagnostic_profile_only_disables_adaptive_stopping(self) -> None:
        defaults = guard.normalize_grounding_guard_config().sampling_settings
        diagnostic = guard.normalize_grounding_guard_config(
            {"sampling_profile": "full_48_diagnostic"}
        ).sampling_settings
        self.assertFalse(diagnostic["adaptive_stopping"])
        self.assertEqual(
            {key: value for key, value in defaults.items() if key != "adaptive_stopping"},
            {key: value for key, value in diagnostic.items() if key != "adaptive_stopping"},
        )


class AssetRegistryTests(unittest.TestCase):
    def test_identity_image_and_video_frames_receive_stable_roles_and_times(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "image+video",
                "media_synthesis_mode": "image_identity_video_control",
                "reference_image_count": 1,
                "image_role": "identity_reference",
                "video_role": "control_structure_pose_depth_canny_composition_motion_camera",
                "sampled_indices": [0, 24, 48],
                "source_fps": 24.0,
            },
            image_count=4,
        )
        self.assertEqual([asset["asset_id"] for asset in registry["assets"]], ["image:1", "video:1"])
        self.assertEqual(registry["assigned_image_count"], 4)
        video = registry["assets"][1]
        self.assertEqual(video["tensor_batch_positions"], [1, 2, 3])
        self.assertEqual(
            [sample["source_frame_index"] for sample in video["samples"]],
            [0, 24, 48],
        )
        self.assertEqual(
            [sample["timecode_seconds"] for sample in video["samples"]],
            [0.0, 1.0, 2.0],
        )
        self.assertEqual(
            [sample["temporal_region"] for sample in video["samples"]],
            ["opening", "middle", "closing"],
        )

    def test_h3_registry_uses_picture_ids_and_manifest_roles(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "image+video",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 2,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: protagonist identity\n"
                    "<Picture 2>: wardrobe style\n"
                    "<Video 1>: action and camera timing"
                ),
                "sampled_indices": [10, 20],
                "source_fps": 10.0,
            },
            image_count=4,
        )
        self.assertEqual(
            [asset["asset_id"] for asset in registry["assets"]],
            ["picture:1", "picture:2", "video:1"],
        )
        self.assertEqual(registry["assets"][0]["role"], "protagonist identity")
        self.assertEqual(registry["assets"][1]["tensor_batch_positions"], [1])
        self.assertEqual(registry["assets"][2]["tensor_batch_positions"], [2, 3])

    def test_empty_media_has_empty_registry(self) -> None:
        registry = guard.build_asset_registry({"source": "none"}, 0)
        self.assertEqual(registry["assets"], [])
        self.assertEqual(registry["assigned_image_count"], 0)

    def test_unattached_optional_still_does_not_shift_video_samples(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "video",
                "reference_image_count": 1,
                "reference_image_backend_attached": False,
                "sampled_indices": [4, 8, 12],
                "source_fps": 4.0,
            },
            3,
        )
        self.assertEqual([asset["asset_id"] for asset in registry["assets"]], ["video:1"])
        self.assertEqual(registry["assets"][0]["tensor_batch_positions"], [0, 1, 2])
        self.assertEqual(
            [sample["source_frame_index"] for sample in registry["assets"][0]["samples"]],
            [4, 8, 12],
        )
        self.assertIn(
            "unattached_reference_image_excluded_from_asset_registry",
            registry["warnings"],
        )

    def test_failed_h3_picture_attachment_does_not_relabel_video_frames(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "video",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 2,
                "reference_image_backend_attached": False,
                "sampled_indices": [5, 15, 25],
                "source_fps": 5.0,
            },
            3,
        )
        self.assertEqual([asset["asset_id"] for asset in registry["assets"]], ["video:1"])
        self.assertEqual(registry["assets"][0]["tensor_batch_positions"], [0, 1, 2])
        self.assertIn(
            "unattached_reference_image_excluded_from_asset_registry",
            registry["warnings"],
        )


class LedgerExtractionTests(unittest.TestCase):
    def test_bundled_schema_has_the_runtime_contract_identity(self) -> None:
        schema = guard.load_grounding_evidence_schema()
        self.assertEqual(
            schema["properties"]["schema"]["const"],
            guard.GROUNDING_LEDGER_SCHEMA_ID,
        )
        self.assertFalse(schema["additionalProperties"])
        external_schema = guard.load_external_evidence_schema()
        self.assertEqual(
            external_schema["properties"]["schema"]["const"],
            guard.EXTERNAL_EVIDENCE_SCHEMA_ID,
        )
        self.assertFalse(external_schema["additionalProperties"])
        evidence_reference = schema["$defs"]["evidenceReference"]
        self.assertEqual(
            evidence_reference["dependentRequired"]["timecode_seconds"],
            ["sample_ordinal"],
        )
        self.assertEqual(schema["$defs"]["observedFact"]["properties"]["claim"]["pattern"], r"\S")

    def test_extracts_standalone_or_combined_ledger(self) -> None:
        ledger = valid_ledger(status="uncertain")
        self.assertEqual(guard.extract_grounding_ledger(json.dumps(ledger)), ledger)
        self.assertEqual(guard.extract_grounding_ledger({"grounding_ledger": ledger}), ledger)
        with self.assertRaisesRegex(guard.GroundingLedgerError, "root must be the ledger"):
            guard.extract_grounding_ledger(
                {"grounding_ledger": ledger},
                allow_combined=False,
            )

    def test_rejects_markdown_salvage_and_trailing_content(self) -> None:
        ledger_text = json.dumps(valid_ledger(status="uncertain"))
        with self.assertRaises(guard.GroundingLedgerError):
            guard.extract_grounding_ledger(f"```json\n{ledger_text}\n```")
        with self.assertRaises(guard.GroundingLedgerError):
            guard.extract_grounding_ledger(ledger_text + " trailing")

    def test_rejects_duplicate_keys_and_excessive_json_nesting(self) -> None:
        duplicate = (
            '{"schema":"dg-grounding-ledger/1","analysis_status":"refused",'
            '"analysis_status":"grounded"}'
        )
        with self.assertRaisesRegex(guard.GroundingLedgerError, "duplicate JSON object key"):
            guard.extract_grounding_ledger(duplicate)
        with self.assertRaisesRegex(guard.GroundingLedgerError, "nesting limit"):
            guard.extract_grounding_ledger("[" * 1100 + "0" + "]" * 1100)

    def test_extreme_json_numbers_fail_closed_without_parser_exceptions(self) -> None:
        for numeric_token in ("9" * 5000, "1e1000000000000000000"):
            with self.subTest(numeric_token=numeric_token[:24]):
                with self.assertRaisesRegex(guard.GroundingLedgerError, "JSON number"):
                    guard.extract_grounding_ledger(
                        '{"schema":"dg-grounding-ledger/1","analysis_status":'
                        f'{numeric_token}}}'
                    )
                claims, warnings = guard.parse_external_evidence(
                    '{"schema":"dg-external-evidence/1","provider":"manual",'
                    '"claims":[{"asset_id":"image:1","claim_type":"object.count",'
                    f'"value":{numeric_token}}}]}}'
                )
                self.assertEqual(claims, [])
                self.assertIn(
                    "external_evidence_json_number_invalid_or_out_of_range",
                    warnings,
                )


class LedgerValidationTests(unittest.TestCase):
    def identity_registry(self) -> dict:
        return guard.build_asset_registry(
            {
                "source": "image+video",
                "media_synthesis_mode": "image_identity_video_control",
                "reference_image_count": 1,
                "image_role": "identity_reference",
                "video_role": "control_structure_pose_depth_canny_composition_motion_camera",
                "sampled_indices": [0, 12, 24],
                "source_fps": 12.0,
            },
            4,
        )

    def grounded_identity_ledger(self) -> dict:
        return valid_ledger(
            fact(
                "identity",
                "The identity still shows a person with a red jacket.",
                ["identity", "appearance", "color"],
                [{"asset_id": "image:1"}],
                typed_claims=[
                    {
                        "asset_id": "image:1",
                        "claim_type": "wardrobe.jacket_color",
                        "value": "red",
                    }
                ],
            ),
            fact(
                "opening_action",
                "The person begins standing still in the opening sample.",
                ["action", "motion", "temporal"],
                [
                    {
                        "asset_id": "video:1",
                        "sample_ordinal": 1,
                        "source_frame_index": 0,
                        "timecode_seconds": 0.0,
                    }
                ],
            ),
            fact(
                "closing_camera",
                "The closing sample is a wider camera composition after forward motion.",
                ["camera", "composition", "motion", "temporal"],
                [
                    {
                        "asset_id": "video:1",
                        "sample_ordinal": 3,
                        "source_frame_index": 24,
                        "timecode_seconds": 2.0,
                    }
                ],
            ),
        )

    def test_valid_grounded_identity_control_ledger(self) -> None:
        result = guard.validate_grounding_ledger(
            self.grounded_identity_ledger(),
            self.identity_registry(),
        )
        self.assertTrue(result.schema_valid)
        self.assertTrue(result.grounded)
        self.assertEqual(result.analysis_status, "grounded")
        self.assertEqual(result.errors, ())
        self.assertEqual(set(result.covered_asset_ids), {"image:1", "video:1"})
        self.assertEqual(set(result.temporal_regions["video:1"]), {"opening", "closing"})

    def test_unknown_assets_and_frame_mismatch_cannot_pass(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][1]["evidence"][0]["source_frame_index"] = 999
        ledger["observed_facts"][2]["evidence"][0]["asset_id"] = "video:9"
        result = guard.validate_grounding_ledger(ledger, self.identity_registry())
        self.assertFalse(result.grounded)
        self.assertIn("invalid_frame_reference", result.blocking_reasons)
        self.assertIn("unknown_asset", result.blocking_reasons)

    def test_unknown_video_reference_without_sample_fields_is_schema_invalid(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][1]["evidence"] = [{"asset_id": "video:99"}]
        result = guard.validate_grounding_ledger(ledger, self.identity_registry())
        self.assertFalse(result.schema_valid)
        self.assertIn("unknown_asset", result.blocking_reasons)
        self.assertIn("invalid_frame_reference", result.blocking_reasons)
        self.assertTrue(any("video_reference_missing" in error for error in result.errors))

    def test_video_reference_missing_all_sample_fields_is_schema_invalid(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][1]["evidence"] = [{"asset_id": "video:1"}]
        result = guard.validate_grounding_ledger(ledger, self.identity_registry())
        self.assertFalse(result.schema_valid)
        self.assertIn("invalid_frame_reference", result.blocking_reasons)

    def test_extreme_sample_ordinal_is_rejected_without_integer_conversion(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][1]["evidence"][0]["sample_ordinal"] = 1 << 10000
        result = guard.validate_grounding_ledger(ledger, self.identity_registry())
        self.assertFalse(result.schema_valid)
        self.assertTrue(any("sample_ordinal:invalid" in error for error in result.errors))

    def test_still_frame_or_time_requires_sample_ordinal(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][0]["evidence"] = [
            {
                "asset_id": "image:1",
                "source_frame_index": 999,
                "timecode_seconds": 999.0,
            }
        ]
        result = guard.validate_grounding_ledger(ledger, self.identity_registry())
        self.assertFalse(result.schema_valid)
        self.assertFalse(result.grounded)
        self.assertIn("invalid_frame_reference", result.blocking_reasons)
        self.assertTrue(
            any("frame_or_time_requires_sample_ordinal" in error for error in result.errors)
        )

    def test_low_confidence_only_and_one_temporal_region_are_insufficient(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][0]["confidence"] = "low"
        ledger["observed_facts"][2]["confidence"] = "low"
        result = guard.validate_grounding_ledger(ledger, self.identity_registry())
        self.assertFalse(result.grounded)
        self.assertIn("insufficient_coverage", result.blocking_reasons)
        self.assertTrue(any("missing_high_or_medium_coverage" in error for error in result.errors))
        self.assertTrue(any("insufficient_temporal_regions" in error for error in result.errors))

    def test_unhashable_confidence_values_are_schema_errors_not_exceptions(self) -> None:
        for malformed in ({}, []):
            with self.subTest(malformed=type(malformed).__name__):
                ledger = self.grounded_identity_ledger()
                ledger["observed_facts"][0]["confidence"] = malformed
                result = guard.validate_grounding_ledger(ledger, self.identity_registry())
                self.assertFalse(result.schema_valid)
                self.assertFalse(result.grounded)
                self.assertTrue(any("confidence:invalid" in error for error in result.errors))

    def test_non_array_failure_reasons_are_schema_errors_not_exceptions(self) -> None:
        for malformed in (None, False, 1, {}):
            with self.subTest(malformed=type(malformed).__name__):
                ledger = self.grounded_identity_ledger()
                ledger["grounding_failure_reasons"] = malformed
                result = guard.validate_grounding_ledger(ledger, self.identity_registry())
                self.assertFalse(result.schema_valid)
                self.assertFalse(result.grounded)
                self.assertTrue(
                    any(
                        "grounding_failure_reasons:must_be_array" in error
                        for error in result.errors
                    )
                )

    def test_identity_and_control_roles_cannot_be_cross_wired(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"].append(
            fact(
                "video_identity",
                "The video actor supplies the output identity.",
                ["identity"],
                [
                    {
                        "asset_id": "video:1",
                        "sample_ordinal": 2,
                        "source_frame_index": 12,
                        "timecode_seconds": 1.0,
                    }
                ],
            )
        )
        result = guard.validate_grounding_ledger(ledger, self.identity_registry())
        self.assertFalse(result.grounded)
        self.assertIn("role_cross_wired", result.blocking_reasons)

    def test_h3_reference_roles_cannot_be_cross_wired(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "image",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 2,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:identity,appearance] protagonist identity, face, body, wardrobe, and accessories\n"
                    "<Picture 2>: [dg:environment,lighting,color,composition] town-square environment, palette, lighting, and 2D animation style; "
                    "do not copy any pictured person identity"
                ),
            },
            2,
        )
        correct = valid_ledger(
            fact(
                "picture_one_identity",
                "Picture one shows the protagonist's identity and appearance.",
                ["identity", "appearance"],
                [{"asset_id": "picture:1"}],
            ),
            fact(
                "picture_two_environment",
                "Picture two shows the environment and lighting palette.",
                ["environment", "lighting", "color"],
                [{"asset_id": "picture:2"}],
            ),
        )
        self.assertTrue(guard.validate_grounding_ledger(correct, registry).grounded)

        swapped = valid_ledger(
            fact(
                "wrong_environment",
                "Picture one supplies the town-square environment.",
                ["environment", "lighting"],
                [{"asset_id": "picture:1"}],
            ),
            fact(
                "wrong_identity",
                "Picture two supplies the protagonist identity.",
                ["identity", "appearance"],
                [{"asset_id": "picture:2"}],
            ),
        )
        result = guard.validate_grounding_ledger(swapped, registry)
        self.assertFalse(result.grounded)
        self.assertIn("role_cross_wired", result.blocking_reasons)

        relabeled = valid_ledger(
            fact(
                "hidden_environment_swap",
                "Picture one supplies the town-square environment and lighting.",
                ["other"],
                [{"asset_id": "picture:1"}],
            ),
            fact(
                "hidden_identity_swap",
                "Picture two supplies the actor identity and facial appearance.",
                ["other"],
                [{"asset_id": "picture:2"}],
            ),
        )
        relabeled_result = guard.validate_grounding_ledger(relabeled, registry)
        self.assertFalse(relabeled_result.grounded)
        self.assertIn("role_cross_wired", relabeled_result.blocking_reasons)

    def test_h3_wardrobe_category_alias_is_canonicalized_without_losing_coverage(
        self,
    ) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "image",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 2,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:identity,appearance,color] protagonist identity, "
                    "face, body, silhouette, colors, wardrobe, and accessories\n"
                    "<Picture 2>: [dg:environment,lighting,color,composition] "
                    "environment, palette, lighting, and style"
                ),
            },
            2,
        )
        ledger = valid_ledger(
            fact(
                "picture_one_appearance",
                "The protagonist has long hair and wears a green garment.",
                ["appearance", "color", "wardrobe"],
                [{"asset_id": "picture:1"}],
            ),
            fact(
                "picture_two_environment",
                "The environment has red curtains and dramatic red lighting.",
                ["environment", "color", "lighting"],
                [{"asset_id": "picture:2"}],
            ),
        )

        result = guard.validate_grounding_ledger(ledger, registry)

        self.assertTrue(result.schema_valid)
        self.assertTrue(result.grounded)
        self.assertEqual(
            ledger["observed_facts"][0]["categories"],
            ["appearance", "color", "wardrobe"],
        )
        self.assertEqual(
            result.ledger["observed_facts"][0]["categories"],
            ["appearance", "color"],
        )
        self.assertIn(
            "ledger.observed_facts[0].categories:canonicalized:wardrobe->appearance",
            result.warnings,
        )
        self.assertFalse(
            any("role_specific_category_missing" in error for error in result.errors)
        )

    def test_wardrobe_alias_accepts_read_only_fact_mapping(self) -> None:
        class ReadOnlyFact(Mapping):
            def __init__(self, value: dict) -> None:
                self._value = value

            def __getitem__(self, key):
                return self._value[key]

            def __iter__(self):
                return iter(self._value)

            def __len__(self) -> int:
                return len(self._value)

        registry = guard.build_asset_registry(
            {
                "source": "image",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 1,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:identity,appearance,color] protagonist identity, "
                    "appearance, wardrobe, and colors"
                ),
            },
            1,
        )
        ledger = valid_ledger(
            ReadOnlyFact(
                fact(
                    "read_only_wardrobe",
                    "The protagonist wears a green garment.",
                    ["appearance", "color", "wardrobe"],
                    [{"asset_id": "picture:1"}],
                )
            )
        )

        result = guard.validate_grounding_ledger(ledger, registry)

        self.assertTrue(result.schema_valid)
        self.assertTrue(result.grounded)
        self.assertEqual(
            result.ledger["observed_facts"][0]["categories"],
            ["appearance", "color"],
        )

    def test_unrecognized_fact_category_remains_schema_blocking(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][0]["categories"].append("wardrobe_style")

        result = guard.validate_grounding_ledger(ledger, self.identity_registry())

        self.assertFalse(result.schema_valid)
        self.assertFalse(result.grounded)
        self.assertIn("schema_invalid", result.blocking_reasons)
        self.assertIn(
            "ledger.observed_facts[0].categories:invalid",
            result.errors,
        )

    def test_raw_duplicate_fact_category_remains_schema_blocking(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][0]["categories"].append("appearance")

        result = guard.validate_grounding_ledger(ledger, self.identity_registry())

        self.assertFalse(result.schema_valid)
        self.assertIn(
            "ledger.observed_facts[0].categories:invalid",
            result.errors,
        )

    def test_wardrobe_alias_does_not_bypass_reference_role_validation(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "image",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 1,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:environment,lighting,color,composition] "
                    "environment and lighting only"
                ),
            },
            1,
        )
        ledger = valid_ledger(
            fact(
                "cross_wired_wardrobe",
                "Picture one supplies the protagonist's wardrobe and appearance.",
                ["wardrobe"],
                [{"asset_id": "picture:1"}],
            )
        )

        result = guard.validate_grounding_ledger(ledger, registry)

        self.assertTrue(result.schema_valid)
        self.assertFalse(result.grounded)
        self.assertIn("role_cross_wired", result.blocking_reasons)

    def test_h3_single_image_static_standing_description_is_not_motion_cross_wiring(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "image",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 1,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:identity,appearance,color,environment,lighting,composition] "
                    "sole visual reference for visible subject and scene attributes."
                ),
            },
            1,
        )
        ledger = valid_ledger(
            fact(
                "single_picture_visible_attributes",
                (
                    "A character with yellow hair, blue eyes, and a pink dress stands "
                    "on a green hill under a blue sky."
                ),
                ["appearance", "color", "environment"],
                [{"asset_id": "picture:1"}],
            )
        )

        result = guard.validate_grounding_ledger(ledger, registry)

        self.assertTrue(result.grounded, result.errors)
        self.assertNotIn("role_cross_wired", result.blocking_reasons)

    def test_h3_environment_accepts_incidental_apparel_terms(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "image",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 2,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:identity,appearance,color] protagonist identity and appearance\n"
                    "<Picture 2>: [dg:environment,lighting,color,composition] environment palette and lighting"
                ),
            },
            2,
        )
        safe_claims = (
            (
                "The environment palette includes a bright blue sky, a green hill, and a pink dress.",
                ["color"],
            ),
            ("A pink dress appears against the green hill.", ["environment"]),
            ("A scarlet coat hangs beside the stone arches.", ["environment"]),
        )
        for claim, categories in safe_claims:
            with self.subTest(claim=claim):
                ledger = valid_ledger(
                    fact(
                        "picture_one_identity",
                        "Picture one shows the protagonist's identity and appearance.",
                        ["identity", "appearance"],
                        [{"asset_id": "picture:1"}],
                    ),
                    fact(
                        "picture_two_environment",
                        claim,
                        categories,
                        [{"asset_id": "picture:2"}],
                    ),
                )

                result = guard.validate_grounding_ledger(ledger, registry)

                self.assertTrue(result.grounded)
                self.assertNotIn("role_cross_wired", result.blocking_reasons)

    def test_h3_apparel_requires_identity_context_before_it_crosses_roles(self) -> None:
        safe_claims = (
            "The pink dress flutters as the actor turns.",
            "The camera uses a slow dolly as a pink dress hangs in the shop window.",
            "The actor takes two steps while the pink dress flutters.",
            "The dress defines the room's pink-and-gold palette.",
            "A well-worn red coat hangs beside the stone arches.",
            "The coat shows visible wear.",
            "Don stands beside a pink dress.",
            "The actor in the room sees a pink dress hanging by the door.",
            "A pink dress hangs in lighting provided by candles.",
            "The image supplies colors for the dress.",
            "The camera uses the framing around a dress.",
            "The actor takes the route by a dress.",
            "A sporting goods store displays a costume.",
            "The actor shows visible wear on a red coat.",
            "The picture provides a coat-of-arms emblem for the town hall.",
            "The image provides a dress-shaped sculpture for the plaza.",
            "The actor in a dress rehearsal turns toward camera.",
        )
        for claim in safe_claims:
            with self.subTest(claim=claim):
                self.assertFalse(
                    guard._claim_role_categories(claim)
                    & {"identity", "appearance"}
                )

        unsafe_claims = (
            "The actor wears a pink dress.",
            "A woman in a pink dress stands beside the doorway.",
            "The video supplies the dress.",
            "The protagonist’s pink dress remains visible.",
            "Alice's pink dress remains visible.",
            "The actors’ costumes remain visible.",
            "The dress on the actor remains visible.",
            "The protagonist in a richly embroidered floor-length pink silk dress remains visible.",
            "A woman in an ornate, floor-length pink dress stands beside the doorway.",
            "The actor proudly wears a pink dress.",
            "The video shows a dress, copied from Picture 2.",
            "The protagonist's elegant, floor-length dress remains visible.",
            "The performer wears a pink dress.",
            "The performer's pink dress remains visible.",
            "The performer in a pink dress turns left.",
            "She wears a pink dress.",
            "The courier wears a red coat.",
            "Alice wears a pink dress.",
            "The actor, in a pink dress, turns toward camera.",
            "A pink dress is worn by Alice.",
            "The red-coated woman crosses the street.",
            "A coat-wearing protagonist sprints across the roof.",
            "The performer still wears a pink dress.",
            "At closing, she still wears the same red coat.",
            "The performer continues to wear a pink dress.",
        )
        for claim in unsafe_claims:
            with self.subTest(claim=claim):
                self.assertTrue(
                    guard._claim_role_categories(claim)
                    & {"identity", "appearance"}
                )

    def test_h3_palette_clothing_exception_rejects_identity_or_transfer_language(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "image",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 2,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:identity,appearance,color] protagonist identity and appearance\n"
                    "<Picture 2>: [dg:environment,lighting,color,composition] environment palette and lighting"
                ),
            },
            2,
        )
        unsafe_palette_claims = (
            "The environment palette includes a pink dress that defines the protagonist identity.",
            "The environment palette includes a pink dress that must be copied to the protagonist.",
            "The environment palette includes a pink dress and the same person's facial appearance.",
            "The environment palette includes the protagonist’s pink dress.",
            "The environment shows an actor wearing a pink dress.",
        )
        for claim in unsafe_palette_claims:
            with self.subTest(claim=claim):
                ledger = valid_ledger(
                    fact(
                        "picture_one_identity",
                        "Picture one shows the protagonist's identity and appearance.",
                        ["identity", "appearance"],
                        [{"asset_id": "picture:1"}],
                    ),
                    fact(
                        "picture_two_palette",
                        claim,
                        ["color"],
                        [{"asset_id": "picture:2"}],
                    ),
                )

                result = guard.validate_grounding_ledger(ledger, registry)

                self.assertFalse(result.grounded)
                self.assertIn("role_cross_wired", result.blocking_reasons)

    def test_h3_declared_pictures_must_all_be_attached(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "image",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 2,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:identity,appearance] protagonist\n"
                    "<Picture 2>: [dg:environment,lighting] location"
                ),
            },
            1,
        )
        result = guard.validate_grounding_ledger(
            valid_ledger(
                fact(
                    "picture_one",
                    "Picture one supplies the protagonist identity.",
                    ["identity", "appearance"],
                    [{"asset_id": "picture:1"}],
                )
            ),
            registry,
        )
        self.assertFalse(result.grounded)
        self.assertIn("insufficient_coverage", result.blocking_reasons)
        self.assertTrue(
            any("h3_declared_attached_asset_mismatch" in error for error in result.errors)
        )

    def test_h3_declared_videos_must_all_be_attached(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "video",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 0,
                "minimax_h3_reference_manifest": (
                    "<Video 1>: [dg:motion,temporal] primary motion\n"
                    "<Video 2>: [dg:motion,temporal] secondary motion"
                ),
                "sampled_indices": [0, 10],
                "source_fps": 10.0,
            },
            2,
        )
        result = guard.validate_grounding_ledger(
            valid_ledger(
                fact(
                    "opening_motion",
                    "Motion begins in the opening sample.",
                    ["motion", "temporal"],
                    [
                        {
                            "asset_id": "video:1",
                            "sample_ordinal": 1,
                            "source_frame_index": 0,
                            "timecode_seconds": 0.0,
                        }
                    ],
                ),
                fact(
                    "closing_motion",
                    "Motion continues in the closing sample.",
                    ["motion", "temporal"],
                    [
                        {
                            "asset_id": "video:1",
                            "sample_ordinal": 2,
                            "source_frame_index": 10,
                            "timecode_seconds": 1.0,
                        }
                    ],
                ),
            ),
            registry,
        )
        self.assertFalse(result.grounded)
        self.assertIn("insufficient_coverage", result.blocking_reasons)
        self.assertTrue(
            any("h3_declared_attached_asset_mismatch" in error for error in result.errors)
        )

    def test_h3_video_role_rejects_actor_identity_even_with_temporal_coverage(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "image+video",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 1,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:identity,appearance] protagonist identity and appearance\n"
                    "<Video 1>: [dg:action,motion,temporal,camera] action, motion, timing, and camera; do not copy actor identity"
                ),
                "sampled_indices": [0, 10, 20],
                "source_fps": 10.0,
            },
            4,
        )
        ledger = valid_ledger(
            fact(
                "picture_identity",
                "Picture one supplies the protagonist identity and appearance.",
                ["identity", "appearance"],
                [{"asset_id": "picture:1"}],
            ),
            fact(
                "opening_video_identity",
                "The opening video actor supplies a different facial identity.",
                ["identity"],
                [
                    {
                        "asset_id": "video:1",
                        "sample_ordinal": 1,
                        "source_frame_index": 0,
                        "timecode_seconds": 0.0,
                    }
                ],
            ),
            fact(
                "closing_video_identity",
                "The closing video preserves that actor's appearance.",
                ["appearance"],
                [
                    {
                        "asset_id": "video:1",
                        "sample_ordinal": 3,
                        "source_frame_index": 20,
                        "timecode_seconds": 2.0,
                    }
                ],
            ),
        )
        result = guard.validate_grounding_ledger(ledger, registry)
        self.assertFalse(result.grounded)
        self.assertIn("role_cross_wired", result.blocking_reasons)

    def test_h3_action_role_accepts_actor_as_subject_not_identity_transfer(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "image+video",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 1,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:identity,appearance] protagonist identity and appearance\n"
                    "<Video 1>: [dg:action,motion,temporal,camera] action, motion, timing, and camera; do not copy actor identity"
                ),
                "sampled_indices": [0, 10],
                "source_fps": 10.0,
            },
            3,
        )
        ledger = valid_ledger(
            fact(
                "picture_identity",
                "Picture one supplies the protagonist identity and facial appearance.",
                ["identity", "appearance"],
                [{"asset_id": "picture:1"}],
            ),
            fact(
                "opening_action",
                "The actor walks left as the camera pans right in the opening sample.",
                ["action", "motion", "camera", "temporal"],
                [
                    {
                        "asset_id": "video:1",
                        "sample_ordinal": 1,
                        "source_frame_index": 0,
                        "timecode_seconds": 0.0,
                    }
                ],
            ),
            fact(
                "closing_action",
                "The actor keeps walking as the camera pans in the closing sample.",
                ["action", "motion", "camera", "temporal"],
                [
                    {
                        "asset_id": "video:1",
                        "sample_ordinal": 2,
                        "source_frame_index": 10,
                        "timecode_seconds": 1.0,
                    }
                ],
            ),
        )
        self.assertTrue(guard.validate_grounding_ledger(ledger, registry).grounded)

    def test_h3_free_form_roles_and_semantic_cross_wiring_fail_closed(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "image",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 2,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:identity,appearance] hero likeness and outfit details\n"
                    "<Picture 2>: [dg:environment,lighting] environment and lighting"
                ),
            },
            2,
        )
        ledger = valid_ledger(
            fact(
                "disguised_environment",
                "A cobblestone plaza with stone arches fills the view.",
                ["identity"],
                [{"asset_id": "picture:1"}],
            ),
            fact(
                "disguised_identity",
                "A red-bearded individual wearing a scarlet coat fills the view.",
                ["environment"],
                [{"asset_id": "picture:2"}],
            ),
        )
        result = guard.validate_grounding_ledger(ledger, registry)
        self.assertFalse(result.grounded)
        self.assertIn("role_cross_wired", result.blocking_reasons)

        unknown_registry = guard.build_asset_registry(
            {
                "source": "image",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 1,
                "minimax_h3_reference_manifest": "<Picture 1>: source alpha",
            },
            1,
        )
        unknown_result = guard.validate_grounding_ledger(
            valid_ledger(
                fact(
                    "unknown_role",
                    "A red square is visible.",
                    ["object", "color"],
                    [{"asset_id": "picture:1"}],
                )
            ),
            unknown_registry,
        )
        self.assertFalse(unknown_result.grounded)
        self.assertIn("insufficient_coverage", unknown_result.blocking_reasons)
        self.assertTrue(
            any("h3_role_contract_not_explicit" in error for error in unknown_result.errors)
        )

        unclassified_result = guard.validate_grounding_ledger(
            valid_ledger(
                fact(
                    "opaque_claim",
                    "Tesserae dominate.",
                    ["identity"],
                    [{"asset_id": "picture:1"}],
                ),
                fact(
                    "valid_environment",
                    "Picture two supplies the environment and lighting.",
                    ["environment", "lighting"],
                    [{"asset_id": "picture:2"}],
                ),
            ),
            registry,
        )
        self.assertTrue(unclassified_result.grounded)

    def test_h3_shared_fact_roles_are_recognized_as_explicit_contracts(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "image",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 3,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:text,object] exact title text and logo\n"
                    "<Picture 2>: [dg:object,color] product color reference\n"
                    "<Picture 3>: [dg:object,text,spatial] left-right spatial layout"
                ),
            },
            3,
        )
        ledger = valid_ledger(
            fact(
                "title_text",
                "The title text reads ALPHA beside the logo.",
                ["text", "object"],
                [{"asset_id": "picture:1"}],
            ),
            fact(
                "product_color",
                "The product color is scarlet red.",
                ["object", "color"],
                [{"asset_id": "picture:2"}],
            ),
            fact(
                "spatial_layout",
                "The logo is left of the title in the spatial layout.",
                ["object", "text", "spatial"],
                [{"asset_id": "picture:3"}],
            ),
        )
        self.assertTrue(guard.validate_grounding_ledger(ledger, registry).grounded)

    def test_h3_explicit_host_categories_accept_natural_claims(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "image+video",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 2,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:identity,appearance] red sports car; preserve shape and decals\n"
                    "<Picture 2>: [dg:environment,lighting] beach location\n"
                    "<Video 1>: [dg:action,motion,temporal] performance; identity must not transfer"
                ),
                "sampled_indices": [0, 10],
                "source_fps": 10.0,
            },
            4,
        )
        self.assertEqual(
            registry["assets"][0]["grounding_role_contract_source"],
            "explicit",
        )
        ledger = valid_ledger(
            fact(
                "identity_detail",
                "The figure has green eyes and a triangular birthmark.",
                ["identity", "appearance"],
                [{"asset_id": "picture:1"}],
            ),
            fact(
                "environment_detail",
                "A sandy shore meets turquoise water.",
                ["environment", "lighting"],
                [{"asset_id": "picture:2"}],
            ),
            fact(
                "opening_action",
                "The subject nods and lifts a cup.",
                ["action", "motion", "temporal"],
                [
                    {
                        "asset_id": "video:1",
                        "sample_ordinal": 1,
                        "source_frame_index": 0,
                        "timecode_seconds": 0.0,
                    }
                ],
            ),
            fact(
                "closing_action",
                "The subject waves and sips.",
                ["action", "motion", "temporal"],
                [
                    {
                        "asset_id": "video:1",
                        "sample_ordinal": 2,
                        "source_frame_index": 10,
                        "timecode_seconds": 1.0,
                    }
                ],
            ),
        )
        self.assertTrue(guard.validate_grounding_ledger(ledger, registry).grounded)

    def test_h3_negative_identity_and_editing_role_vocabulary(self) -> None:
        for role in (
            "action and camera reference; identity must not transfer",
            "action reference; exclude identity",
            "action reference; without identity transfer",
            "action reference; ignore identity",
            "action reference; identity is not a source",
            "whole-video editing and continuation",
            "cut rhythm reference",
        ):
            allowed, specific = guard._h3_role_category_contract(role)
            self.assertTrue(specific, role)
            self.assertFalse({"identity", "appearance"} & specific, role)
            self.assertTrue(specific.issubset(allowed), role)
        for preservation_role in (
            "no identity changes; preserve face, hair, and wardrobe",
            "identity is not limited to the face",
            "do not alter identity; preserve appearance",
            "identity must not drift",
        ):
            _allowed, specific = guard._h3_role_category_contract(preservation_role)
            self.assertTrue({"identity", "appearance"} & specific, preservation_role)

    def test_h3_video_identity_prose_cannot_hide_under_motion_categories(self) -> None:
        registry = guard.build_asset_registry(
            {
                "source": "image+video",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 1,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:identity,appearance] protagonist identity and appearance\n"
                    "<Video 1>: [dg:action,motion,temporal] action, motion, and timing; do not copy actor identity"
                ),
                "sampled_indices": [0, 10],
                "source_fps": 10.0,
            },
            3,
        )
        ledger = valid_ledger(
            fact(
                "picture_identity",
                "Picture one supplies the protagonist identity and facial appearance.",
                ["identity", "appearance"],
                [{"asset_id": "picture:1"}],
            ),
            fact(
                "opening_disguised_identity",
                "A bearded individual in a scarlet coat fills the opening frame.",
                ["action", "motion", "temporal"],
                [
                    {
                        "asset_id": "video:1",
                        "sample_ordinal": 1,
                        "source_frame_index": 0,
                        "timecode_seconds": 0.0,
                    }
                ],
            ),
            fact(
                "closing_disguised_identity",
                "The bearded individual in the scarlet coat fills the closing frame.",
                ["action", "motion", "temporal"],
                [
                    {
                        "asset_id": "video:1",
                        "sample_ordinal": 2,
                        "source_frame_index": 10,
                        "timecode_seconds": 1.0,
                    }
                ],
            ),
        )
        result = guard.validate_grounding_ledger(ledger, registry)
        self.assertFalse(result.grounded)
        self.assertIn("role_cross_wired", result.blocking_reasons)

    def test_exact_typed_external_conflict_blocks_but_unknown_provider_is_advisory(self) -> None:
        conflicting = {
            "schema": guard.EXTERNAL_EVIDENCE_SCHEMA_ID,
            "provider": "ocr",
            "claims": [
                {
                    "asset_id": "image:1",
                    "claim_type": "wardrobe.jacket_color",
                    "value": "blue",
                }
            ],
        }
        result = guard.validate_grounding_ledger(
            self.grounded_identity_ledger(),
            self.identity_registry(),
            conflicting,
        )
        self.assertFalse(result.grounded)
        self.assertIn("external_typed_conflict", result.blocking_reasons)
        self.assertEqual(len(result.external_conflicts), 1)
        self.assertEqual(result.accepted_external_claim_count, 1)

        advisory = dict(conflicting, provider="custom-vlm")
        advisory_result = guard.validate_grounding_ledger(
            self.grounded_identity_ledger(),
            self.identity_registry(),
            advisory,
        )
        self.assertTrue(advisory_result.grounded)
        self.assertIn("external_evidence_unknown_provider", advisory_result.warnings)
        self.assertEqual(advisory_result.accepted_external_claim_count, 0)

    def test_invalid_external_envelope_is_informational_and_never_credited(self) -> None:
        invalid = {
            "schema": guard.EXTERNAL_EVIDENCE_SCHEMA_ID,
            "provider": "ocr",
            "claims": [
                {
                    "asset_id": "image:1",
                    "claim_type": "wardrobe.jacket_color",
                    "value": "blue",
                }
            ],
            "spoofed_metadata": {"verification_level": "independent"},
        }
        result = guard.validate_grounding_ledger(
            self.grounded_identity_ledger(),
            self.identity_registry(),
            invalid,
        )
        self.assertTrue(result.grounded)
        self.assertEqual(result.accepted_external_claim_count, 0)
        self.assertIn("external_evidence_envelope_invalid", result.warnings)

    def test_external_envelope_is_exact_and_all_or_nothing(self) -> None:
        claims = [
            {
                "asset_id": "image:1",
                "claim_type": "wardrobe.jacket_color",
                "value": "blue",
                "extra": "hide-conflict",
            },
            {
                "asset_id": "image:1",
                "claim_type": "wardrobe.jacket_color",
                "value": "red",
            },
        ]
        invalid_claim_result = guard.validate_grounding_ledger(
            self.grounded_identity_ledger(),
            self.identity_registry(),
            {
                "schema": guard.EXTERNAL_EVIDENCE_SCHEMA_ID,
                "provider": "ocr",
                "claims": claims,
            },
        )
        self.assertTrue(invalid_claim_result.grounded)
        self.assertEqual(invalid_claim_result.accepted_external_claim_count, 0)
        self.assertIn("external_evidence_claim_0_has_unknown_fields", invalid_claim_result.warnings)

        spaced_provider = guard.validate_grounding_ledger(
            self.grounded_identity_ledger(),
            self.identity_registry(),
            {
                "schema": guard.EXTERNAL_EVIDENCE_SCHEMA_ID,
                "provider": " OCR ",
                "claims": [claims[1]],
            },
        )
        self.assertEqual(spaced_provider.accepted_external_claim_count, 0)
        self.assertIn("external_evidence_unknown_provider", spaced_provider.warnings)

        duplicate_json = (
            '{"schema":"dg-external-evidence/1","provider":"custom-vlm",'
            '"provider":"ocr","claims":[]}'
        )
        parsed, warnings = guard.parse_external_evidence(duplicate_json)
        self.assertEqual(parsed, [])
        self.assertIn("external_evidence_duplicate_json_key", warnings)
        parsed, warnings = guard.parse_external_evidence("[" * 1100 + "0" + "]" * 1100)
        self.assertEqual(parsed, [])
        self.assertIn("external_evidence_json_nesting_limit", warnings)

    def test_internally_contradictory_external_claims_block(self) -> None:
        ledger = self.grounded_identity_ledger()
        external = {
            "schema": guard.EXTERNAL_EVIDENCE_SCHEMA_ID,
            "provider": "manual",
            "claims": [
                {
                    "asset_id": "image:1",
                    "claim_type": "wardrobe.jacket_color",
                    "value": "red",
                },
                {
                    "asset_id": "image:1",
                    "claim_type": "wardrobe.jacket_color",
                    "value": "blue",
                },
            ],
        }
        result = guard.validate_grounding_ledger(
            ledger,
            self.identity_registry(),
            external,
        )
        self.assertFalse(result.grounded)
        self.assertIn("external_typed_conflict", result.blocking_reasons)
        self.assertEqual(
            result.external_conflicts,
            ("image:1:wardrobe.jacket_color:asset",),
        )

    def test_external_frame_bounds_are_informational_and_not_accepted_as_proof(self) -> None:
        external = {
            "schema": guard.EXTERNAL_EVIDENCE_SCHEMA_ID,
            "provider": "motion",
            "claims": [
                {
                    "asset_id": "video:1",
                    "sample_ordinal": 99,
                    "source_frame_index": 999,
                    "timecode_seconds": 999.0,
                    "claim_type": "motion_direction",
                    "value": "left_to_right",
                    "confidence": "high",
                }
            ],
        }
        result = guard.validate_grounding_ledger(
            self.grounded_identity_ledger(),
            self.identity_registry(),
            external,
        )
        self.assertTrue(result.grounded)
        self.assertEqual(result.accepted_external_claim_count, 0)
        self.assertIn("external_evidence_claim_0_sample_out_of_bounds", result.warnings)

    def test_identity_control_accepts_action_or_camera_video_evidence(self) -> None:
        action_only = self.grounded_identity_ledger()
        action_only["observed_facts"][2]["categories"] = ["action", "motion", "temporal"]
        self.assertTrue(
            guard.validate_grounding_ledger(action_only, self.identity_registry()).grounded
        )

        camera_only = self.grounded_identity_ledger()
        camera_only["observed_facts"][1]["categories"] = ["camera", "composition"]
        camera_only["observed_facts"][2]["categories"] = ["camera", "composition"]
        self.assertTrue(
            guard.validate_grounding_ledger(camera_only, self.identity_registry()).grounded
        )

    def test_model_typed_claim_sample_bounds_are_blocking(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][1]["typed_claims"] = [
            {
                "asset_id": "video:1",
                "sample_ordinal": 99,
                "claim_type": "motion_direction",
                "value": "left_to_right",
            }
        ]
        result = guard.validate_grounding_ledger(ledger, self.identity_registry())
        self.assertFalse(result.grounded)
        self.assertIn("invalid_frame_reference", result.blocking_reasons)

    def test_video_typed_claim_requires_and_conflicts_at_exact_sample(self) -> None:
        omitted = self.grounded_identity_ledger()
        omitted["observed_facts"][1]["typed_claims"] = [
            {
                "asset_id": "video:1",
                "claim_type": "motion.direction",
                "value": "left",
            }
        ]
        omitted_result = guard.validate_grounding_ledger(omitted, self.identity_registry())
        self.assertFalse(omitted_result.schema_valid)
        self.assertTrue(
            any("required_for_video_claim" in error for error in omitted_result.errors)
        )

        sample_specific = self.grounded_identity_ledger()
        sample_specific["observed_facts"][1]["typed_claims"] = [
            {
                "asset_id": "video:1",
                "sample_ordinal": 1,
                "claim_type": "motion.direction",
                "value": "left",
            }
        ]
        conflict = guard.validate_grounding_ledger(
            sample_specific,
            self.identity_registry(),
            {
                "schema": guard.EXTERNAL_EVIDENCE_SCHEMA_ID,
                "provider": "motion",
                "claims": [
                    {
                        "asset_id": "video:1",
                        "sample_ordinal": 1,
                        "claim_type": "motion.direction",
                        "value": "right",
                    }
                ],
            },
        )
        self.assertFalse(conflict.grounded)
        self.assertIn("external_typed_conflict", conflict.blocking_reasons)

    def test_model_typed_claim_missing_value_is_schema_invalid(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][0]["typed_claims"] = [
            {
                "asset_id": "image:1",
                "claim_type": "wardrobe.jacket_color",
                "sample_ordinal": 1,
            }
        ]
        result = guard.validate_grounding_ledger(ledger, self.identity_registry())
        self.assertFalse(result.schema_valid)
        self.assertFalse(result.grounded)
        self.assertIn("schema_invalid", result.blocking_reasons)

    def test_model_typed_claim_must_be_bound_to_enclosing_fact_evidence(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][0]["typed_claims"] = [
            {
                "asset_id": "video:1",
                "sample_ordinal": 1,
                "claim_type": "wardrobe.jacket_color",
                "value": "red",
            }
        ]
        result = guard.validate_grounding_ledger(ledger, self.identity_registry())
        self.assertFalse(result.schema_valid)
        self.assertFalse(result.grounded)
        self.assertIn("schema_invalid", result.blocking_reasons)
        self.assertTrue(any("not_bound_to_fact_evidence" in error for error in result.errors))

    def test_model_typed_claim_internal_conflict_is_schema_invalid(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][0]["typed_claims"] = [
            {
                "asset_id": "image:1",
                "claim_type": "wardrobe.jacket_color",
                "value": "red",
            },
            {
                "asset_id": "image:1",
                "claim_type": "wardrobe.jacket_color",
                "value": "blue",
            },
        ]
        result = guard.validate_grounding_ledger(ledger, self.identity_registry())
        self.assertFalse(result.schema_valid)
        self.assertFalse(result.grounded)
        self.assertIn("schema_invalid", result.blocking_reasons)
        self.assertTrue(any("internal_conflict" in error for error in result.errors))

    def test_single_sample_typed_claim_aliases_bind_and_conflict_canonically(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][0]["typed_claims"] = [
            {
                "asset_id": "image:1",
                "sample_ordinal": 1,
                "claim_type": "wardrobe.jacket_color",
                "value": "red",
            }
        ]
        self.assertTrue(guard.validate_grounding_ledger(ledger, self.identity_registry()).grounded)

        ledger["observed_facts"][0]["typed_claims"].append(
            {
                "asset_id": "image:1",
                "claim_type": "wardrobe.jacket_color",
                "value": "blue",
            }
        )
        conflict = guard.validate_grounding_ledger(ledger, self.identity_registry())
        self.assertFalse(conflict.schema_valid)
        self.assertTrue(any("internal_conflict" in error for error in conflict.errors))

        external_conflict = guard.validate_grounding_ledger(
            self.grounded_identity_ledger(),
            self.identity_registry(),
            {
                "schema": guard.EXTERNAL_EVIDENCE_SCHEMA_ID,
                "provider": "manual",
                "claims": [
                    {
                        "asset_id": "image:1",
                        "sample_ordinal": 1,
                        "claim_type": "wardrobe.jacket_color",
                        "value": "blue",
                    }
                ],
            },
        )
        self.assertFalse(external_conflict.grounded)
        self.assertIn("external_typed_conflict", external_conflict.blocking_reasons)

    def test_typed_numbers_use_json_numeric_equality_without_precision_loss(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][0]["typed_claims"] = [
            {
                "asset_id": "image:1",
                "claim_type": "object.count",
                "value": 1,
            },
            {
                "asset_id": "image:1",
                "claim_type": "object.count",
                "value": 1.0,
            },
        ]
        self.assertTrue(guard.validate_grounding_ledger(ledger, self.identity_registry()).grounded)

        ledger["observed_facts"][0]["typed_claims"] = [
            {
                "asset_id": "image:1",
                "claim_type": "object.serial",
                "value": 123456789012345678901234567890,
            },
            {
                "asset_id": "image:1",
                "claim_type": "object.serial",
                "value": 123456789012345678901234567891,
            },
        ]
        precise_conflict = guard.validate_grounding_ledger(ledger, self.identity_registry())
        self.assertFalse(precise_conflict.schema_valid)
        self.assertTrue(
            any(
                "numeric_value_must_be_signed_64_bit_integer" in error
                for error in precise_conflict.errors
            )
        )

        model_text = json.dumps(self.grounded_identity_ledger()).replace(
            '"value": "red"',
            '"value": 9007199254740993.0',
            1,
        )
        precise_external = (
            '{"schema":"dg-external-evidence/1","provider":"manual","claims":['
            '{"asset_id":"image:1","claim_type":"wardrobe.jacket_color",'
            '"value":9007199254740992.0}]}'
        )
        external_result = guard.validate_grounding_ledger(
            guard.extract_grounding_ledger(model_text),
            self.identity_registry(),
            precise_external,
        )
        self.assertFalse(external_result.grounded)
        self.assertIn("external_typed_conflict", external_result.blocking_reasons)

        fractional = self.grounded_identity_ledger()
        fractional["observed_facts"][0]["typed_claims"] = [
            {
                "asset_id": "image:1",
                "claim_type": "object.measurement",
                "value": "__FIRST_DECIMAL__",
            },
            {
                "asset_id": "image:1",
                "claim_type": "object.measurement",
                "value": "__SECOND_DECIMAL__",
            },
        ]
        fractional_text = (
            json.dumps(fractional)
            .replace('"__FIRST_DECIMAL__"', "0.123456789012345678901")
            .replace('"__SECOND_DECIMAL__"', "0.123456789012345678902")
        )
        fractional_result = guard.validate_grounding_ledger(
            guard.extract_grounding_ledger(fractional_text),
            self.identity_registry(),
        )
        self.assertFalse(fractional_result.schema_valid)
        self.assertTrue(
            any(
                "numeric_value_must_be_signed_64_bit_integer" in error
                for error in fractional_result.errors
            )
        )

    def test_explicit_null_optional_fields_are_schema_invalid(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][0]["evidence"][0]["sample_ordinal"] = None
        ledger["observed_facts"][0]["typed_claims"][0]["sample_ordinal"] = None
        result = guard.validate_grounding_ledger(ledger, self.identity_registry())
        self.assertFalse(result.schema_valid)
        self.assertIn("schema_invalid", result.blocking_reasons)

        parsed, warnings = guard.parse_external_evidence(
            {
                "schema": guard.EXTERNAL_EVIDENCE_SCHEMA_ID,
                "provider": "manual",
                "claims": [
                    {
                        "asset_id": "image:1",
                        "claim_type": "object.count",
                        "value": 1,
                        "sample_ordinal": None,
                    }
                ],
            }
        )
        self.assertEqual(parsed, [])
        self.assertIn("external_evidence_claim_0_invalid_sample", warnings)

    def test_json_schema_integral_numbers_and_minimums_match_runtime(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][1]["evidence"][0]["sample_ordinal"] = 1.0
        ledger["observed_facts"][1]["evidence"][0]["source_frame_index"] = 0.0
        ledger["observed_facts"][0]["typed_claims"][0]["sample_ordinal"] = 1.0
        self.assertTrue(guard.validate_grounding_ledger(ledger, self.identity_registry()).grounded)

        invalid = self.grounded_identity_ledger()
        invalid["observed_facts"][1]["evidence"][0]["source_frame_index"] = -1.0
        invalid["observed_facts"][1]["evidence"][0]["timecode_seconds"] = -1.0
        result = guard.validate_grounding_ledger(invalid, self.identity_registry())
        self.assertFalse(result.schema_valid)
        self.assertIn("schema_invalid", result.blocking_reasons)

        parsed, warnings = guard.parse_external_evidence(
            {
                "schema": guard.EXTERNAL_EVIDENCE_SCHEMA_ID,
                "provider": "manual",
                "claims": [
                    {
                        "asset_id": "image:1",
                        "claim_type": "object.count",
                        "value": 1,
                        "sample_ordinal": 1.0,
                        "source_frame_index": 0.0,
                        "timecode_seconds": 0.0,
                    }
                ],
            }
        )
        self.assertEqual(warnings, [])
        self.assertEqual(parsed[0]["sample_ordinal"], 1)
        self.assertEqual(parsed[0]["source_frame_index"], 0)

    def test_matching_high_precision_timecode_is_replaced_by_host_value(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["observed_facts"][1]["evidence"][0]["timecode_seconds"] = (
            "__PRECISE_TIMECODE__"
        )
        serialized = json.dumps(ledger).replace(
            '"__PRECISE_TIMECODE__"',
            "0.0000000000000000000000000000000000000001",
        )
        result = guard.validate_grounding_ledger(
            guard.extract_grounding_ledger(serialized),
            self.identity_registry(),
        )
        self.assertTrue(result.grounded)
        self.assertEqual(
            result.ledger["observed_facts"][1]["evidence"][0]["timecode_seconds"],
            0.0,
        )

    def test_additional_properties_and_spoofed_host_fields_are_rejected(self) -> None:
        ledger = self.grounded_identity_ledger()
        ledger["ready_for_generation"] = True
        ledger["observed_facts"][0]["host_status"] = "verified"
        result = guard.validate_grounding_ledger(ledger, self.identity_registry())
        self.assertFalse(result.schema_valid)
        self.assertFalse(result.grounded)
        self.assertIn("schema_invalid", result.blocking_reasons)

    def test_refusal_clause_overrides_a_claimed_grounded_status(self) -> None:
        result = guard.validate_grounding_ledger(
            self.grounded_identity_ledger(),
            self.identity_registry(),
            final_text="I am sorry, but I cannot analyze this image.",
        )
        self.assertFalse(result.grounded)
        self.assertEqual(result.analysis_status, "refused")
        self.assertEqual(result.model_analysis_status, "grounded")
        self.assertEqual(result.ledger["analysis_status"], "refused")
        self.assertIn("refusal_detected", result.blocking_reasons)


class RefusalDetectionTests(unittest.TestCase):
    def test_only_complete_clauses_match(self) -> None:
        self.assertEqual(guard.detect_refusal_phrases("harmful guidelines apology"), [])
        self.assertEqual(guard.detect_refusal_phrases("I cannot see a red object."), [])
        self.assertIn(
            "cannot_assist",
            guard.detect_refusal_phrases("I cannot assist with that request."),
        )
        self.assertEqual(
            guard.detect_refusal_phrases("I'm sorry, but I cannot analyze this image."),
            ["apology_refusal"],
        )
        self.assertEqual(
            guard.detect_refusal_phrases("I must decline this request."),
            ["must_decline"],
        )
        self.assertEqual(
            guard.detect_refusal_phrases("I cannot view this image."),
            ["cannot_assist"],
        )
        self.assertEqual(
            guard.detect_refusal_phrases("I'm unable to review the video."),
            ["unable_to_assist"],
        )


class GuardDecisionTests(unittest.TestCase):
    def simple_validation(self) -> guard.GroundingLedgerValidation:
        registry = guard.build_asset_registry(
            {"source": "image", "image_role": "primary_image_source"}, 1
        )
        ledger = valid_ledger(
            fact("object", "A green cube is visible.", ["object", "color"], [{"asset_id": "image:1"}])
        )
        return guard.validate_grounding_ledger(ledger, registry)

    def test_off_is_disabled_and_no_media_is_not_applicable(self) -> None:
        disabled = guard.decide_grounding_guard(
            {"mode": "off"},
            None,
            media_present=True,
            transport_confirmed=False,
        )
        self.assertEqual(disabled.decision, "disabled")
        self.assertFalse(disabled.would_block)

        no_media = guard.decide_grounding_guard(
            {"mode": "strict"},
            None,
            media_present=False,
            transport_confirmed=False,
        )
        self.assertEqual(no_media.decision, "not_applicable")

    def test_audit_warns_where_strict_blocks(self) -> None:
        audit = guard.decide_grounding_guard(
            {"mode": "audit"},
            None,
            media_present=True,
            transport_confirmed=False,
        )
        strict = guard.decide_grounding_guard(
            {"mode": "strict"},
            None,
            media_present=True,
            transport_confirmed=False,
        )
        self.assertEqual(audit.decision, "warn")
        self.assertEqual(strict.decision, "block")
        self.assertTrue(audit.would_block)
        self.assertEqual(
            strict.blocked_reasons,
            ("visual_grounding_unverified", "visual_transport_error"),
        )

    def test_verified_grounding_passes_and_telemetry_error_blocks_strict(self) -> None:
        validation = self.simple_validation()
        passed = guard.decide_grounding_guard(
            {"mode": "strict"},
            validation,
            media_present=True,
            transport_confirmed=True,
        )
        self.assertEqual(passed.decision, "pass")
        failed = guard.decide_grounding_guard(
            {"mode": "strict"},
            validation,
            media_present=True,
            transport_confirmed=True,
            telemetry_error="processor callback failed",
        )
        self.assertEqual(failed.decision, "block")
        self.assertIn("visual_grounding_telemetry_error", failed.blocked_reasons)

    def test_telemetry_error_is_not_mislabeled_when_no_result_can_return_transport(self) -> None:
        failure = guard.decide_grounding_guard(
            {"mode": "strict"},
            None,
            media_present=True,
            transport_confirmed=False,
            telemetry_error="collector summary failed",
        )
        self.assertEqual(failure.analysis_status, "uncertain")
        self.assertEqual(
            failure.blocked_reasons,
            ("visual_grounding_unverified", "visual_grounding_telemetry_error"),
        )


class ReportAndTraceTests(unittest.TestCase):
    def test_compaction_keeps_first_and_final_snapshots(self) -> None:
        report = {
            "schema": guard.GROUNDING_REPORT_SCHEMA_ID,
            "analysis_status": "grounded",
            "decision": "pass",
            "trajectory_summary": {
                "snapshots": [
                    {"step": index, "draft": "x" * 5000} for index in range(8)
                ]
            },
        }
        compact = guard.compact_grounding_report(report, max_bytes=12_000)
        snapshots = compact["trajectory_summary"]["snapshots"]
        self.assertEqual([item["step"] for item in snapshots], [0, 7])
        self.assertTrue(compact["report_truncated"])
        self.assertLessEqual(
            len(guard.grounding_report_json(compact).encode("utf-8")),
            12_000,
        )

    def test_public_pretty_report_never_exceeds_256_kibibytes(self) -> None:
        report = {
            "schema": guard.GROUNDING_REPORT_SCHEMA_ID,
            "analysis_status": "uncertain",
            "decision": "warn",
            "trajectory_summary": {
                "snapshots": [
                    {"step": index, "draft": "é" * 12_000}
                    for index in range(16)
                ]
            },
        }
        compact = guard.compact_grounding_report(report)
        self.assertLessEqual(
            len(guard.grounding_report_json(compact).encode("utf-8")),
            guard.MAX_COMPACT_REPORT_BYTES,
        )

    def test_report_contains_host_decision_and_sampling_contract(self) -> None:
        registry = guard.build_asset_registry({"source": "none"}, 0)
        decision = guard.decide_grounding_guard(
            {"mode": "audit"},
            None,
            media_present=False,
            transport_confirmed=False,
        )
        report = guard.build_grounding_report(
            {"mode": "audit"},
            registry,
            decision,
            attempt_count=0,
        )
        self.assertEqual(report["decision"], "not_applicable")
        self.assertEqual(report["effective_sampling"]["max_denoising_steps"], 48)

    def test_transport_tensor_descriptors_are_not_mistaken_for_raw_pixels(self) -> None:
        report = guard.compact_grounding_report(
            {
                "schema": guard.GROUNDING_REPORT_SCHEMA_ID,
                "transport": {
                    "pixel_tensors": {
                        "pixel_values": {
                            "shape": [1, 2520, 768],
                            "dtype": "torch.float32",
                        },
                        "pixel_values_videos": {
                            "shape": [1, 3, 2520, 768],
                            "dtype": "torch.float32",
                        },
                    }
                },
            }
        )
        self.assertEqual(
            report["transport"]["pixel_tensors"]["pixel_values"]["shape"],
            [1, 2520, 768],
        )

    def test_oversized_tensor_shape_descriptor_cannot_smuggle_payload(self) -> None:
        with self.assertRaises(guard.GroundingGuardError):
            guard.compact_grounding_report(
                {
                    "transport": {
                        "pixel_tensors": {
                            "pixel_values": {
                                "shape": [index % 256 for index in range(4096)],
                                "dtype": "torch.uint8",
                            }
                        }
                    }
                }
            )

    def test_trace_is_json_only_and_confined_below_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "output"
            path = guard.persist_grounding_trace(
                {"schema": guard.GROUNDING_REPORT_SCHEMA_ID, "asset_hashes": ["abc"]},
                output_root,
                "diffusiongemma_grounding/session",
                filename="trace-one",
            )
            self.assertEqual(path.suffix, ".json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["asset_hashes"], ["abc"])
            self.assertEqual(path.resolve().relative_to(output_root.resolve()).parts[0], "diffusiongemma_grounding")

            with self.assertRaises(guard.GroundingGuardError):
                guard.persist_grounding_trace({}, output_root, "../escape")
            with self.assertRaises(guard.GroundingGuardError):
                guard.persist_grounding_trace({"raw_logits": [1, 2, 3]}, output_root)
            with self.assertRaises(guard.GroundingGuardError):
                guard.persist_grounding_trace({"video_frames": [[0, 1]]}, output_root)
            with self.assertRaises(guard.GroundingGuardError):
                guard.persist_grounding_trace({"pixel_values": [[0, 1]]}, output_root)
            with self.assertRaises(guard.GroundingGuardError):
                guard.persist_grounding_trace({"pixel_values_videos": [[0, 1]]}, output_root)


if __name__ == "__main__":
    unittest.main()
