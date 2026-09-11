from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]


def load_nodes_module():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    package_name = f"diffusiongemma_grounding_integration_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        package_name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load DiffusionGemma Prompt Builder")
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    return sys.modules[f"{package_name}.nodes"]


class GroundingGuardIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()

    def runtime(self):
        return self.nodes.RuntimeConfig(
            model_path="unused-local-checkpoint",
            backend="transformers_inprocess",
            dtype="auto",
            quantization="none",
            local_files_only=True,
            unload_policy="unload_after_run",
            max_memory_gb=20.0,
            status={"ready": True, "supports_pixels": True},
        )

    def target(self):
        return self.nodes.TargetProfileConfig(target_profile="ltx")

    def image_context(self):
        return self.nodes.GemmaContext(
            user_prompt="Describe the reference faithfully.",
            images=torch.zeros((1, 8, 8, 3), dtype=torch.float32),
            source="image",
            media_metadata={
                "source": "image",
                "reference_image_count": 1,
                "image_role": "primary_image_source",
            },
        )

    def video_context(self, frame_count: int = 5):
        return self.nodes.GemmaContext(
            user_prompt="Describe the video faithfully.",
            images=torch.zeros((frame_count, 8, 8, 3), dtype=torch.float32),
            source="video",
            media_metadata={
                "source": "video",
                "sampled_indices": [index * 10 for index in range(frame_count)],
                "sampled_frame_count": frame_count,
                "video_sampled_frame_count": frame_count,
                "source_fps": 10.0,
                "sample_fps": 1.0,
                "video_role": "primary_video_recreation_source",
            },
        )

    @staticmethod
    def uncertain_ledger(status: str = "uncertain") -> dict:
        return {
            "schema": "dg-grounding-ledger/1",
            "analysis_status": status,
            "observed_facts": [],
            "inferred_facts": [],
            "creative_additions": [],
            "uncertainties": ["Visual evidence could not be verified."],
            "grounding_failure_reasons": ["insufficient visual certainty"],
        }

    @staticmethod
    def grounded_ledger(registry: dict) -> dict:
        facts: list[dict] = []
        fact_index = 1
        for asset in registry.get("assets", []):
            samples = asset.get("samples", [])
            chosen_samples = samples
            if asset.get("kind") == "video" and len(samples) > 1:
                chosen_samples = [samples[0], samples[-1]]
            for sample in chosen_samples:
                evidence = {
                    "asset_id": asset["asset_id"],
                    "sample_ordinal": sample["sample_ordinal"],
                    "source_frame_index": sample["source_frame_index"],
                    "timecode_seconds": sample["timecode_seconds"],
                }
                category = "action" if asset.get("kind") == "video" else "object"
                facts.append(
                    {
                        "fact_id": f"fact-{fact_index}",
                        "claim": f"A directly visible fact from {asset['asset_id']}.",
                        "confidence": "high",
                        "categories": [category],
                        "evidence": [evidence],
                        "typed_claims": [],
                    }
                )
                fact_index += 1
        return {
            "schema": "dg-grounding-ledger/1",
            "analysis_status": "grounded",
            "observed_facts": facts,
            "inferred_facts": [],
            "creative_additions": [],
            "uncertainties": [],
            "grounding_failure_reasons": [],
        }

    @staticmethod
    def packet(prompt: str = "A verified generator prompt.", ledger: dict | None = None) -> dict:
        packet = {
            "ltx_prompt": prompt,
            "ideogram_prompt": "",
            "minimax_h3_prompt": "",
            "negative_prompt": "",
            "scene_segments": [],
            "metadata": {
                "target_profile": "ltx",
                "ready_for_generation": True,
                "blocked_reasons": [],
            },
        }
        if ledger is not None:
            packet["grounding_ledger"] = copy.deepcopy(ledger)
        return packet

    def backend_result(
        self,
        ledger: dict,
        *,
        transport_confirmed: bool = True,
        telemetry_refusal: bool = False,
    ):
        return self.nodes.BackendRunResult(
            decoded_text=json.dumps(ledger),
            transport={
                "pixel_transport_confirmed": transport_confirmed,
                "not_applicable": False,
                "mismatch_reasons": [],
            },
            telemetry={
                "schema_version": "dg-denoising-telemetry/1",
                "forward_count": 2,
                "snapshots": [{"canvas": 1, "step": 1}],
                "stream_ended": True,
                "final_output": {"text": [], "token_count": [1]},
                "errors": [],
                "refusal": {"detected": telemetry_refusal},
            },
            effective_sampling={"profile": "checkpoint_defaults"},
            timing={"total_seconds": 0.01},
        )

    def truncated_evidence_result(self, registry: dict, complete_fact_count: int = 5):
        ledger = self.grounded_ledger(registry)
        template = ledger["observed_facts"][0]
        facts: list[dict] = []
        for fact_index in range(1, complete_fact_count + 2):
            fact = copy.deepcopy(template)
            fact["fact_id"] = f"fact-{fact_index}"
            fact["claim"] = f"Visible detail {fact_index} deliberately described."
            facts.append(fact)
        ledger["observed_facts"] = facts
        decoded_text = json.dumps(ledger)
        partial_claim = f"Visible detail {complete_fact_count + 1} deliberately"
        decoded_text = decoded_text[: decoded_text.index(partial_claim) + len(partial_claim)]
        result = self.backend_result(ledger)
        result.decoded_text = decoded_text
        return result

    @staticmethod
    def legacy_result(packet: dict):
        return packet, "compiler raw", "", "", True, "", packet["metadata"]

    def test_settings_and_cot_public_contract_preserve_existing_outputs(self) -> None:
        settings_inputs = self.nodes.DiffusionGemmaGroundingGuardSettings.INPUT_TYPES()
        required = settings_inputs["required"]
        optional = settings_inputs["optional"]
        self.assertEqual(required["mode"][0], ["off", "audit", "strict"])
        self.assertEqual(required["mode"][1]["default"], "audit")
        self.assertTrue(required["retry_on_uncertain"][1]["default"])
        self.assertEqual(required["sampling_profile"][1]["default"], "checkpoint_defaults")
        self.assertEqual(required["seed"][1]["max"], (1 << 64) - 1)
        self.assertTrue(optional["external_evidence_json"][1]["forceInput"])
        self.assertEqual(
            optional["evidence_token_budget"][0],
            ["auto", "768", "1024", "1280"],
        )
        self.assertEqual(optional["evidence_token_budget"][1]["default"], "auto")
        self.assertTrue(optional["evidence_token_budget"][1]["advanced"])
        self.assertEqual(
            self.nodes.DiffusionGemmaGroundingGuardSettings.RETURN_TYPES,
            ("DG_GROUNDING_GUARD_CONFIG", "STRING"),
        )

        cot_inputs = self.nodes.DiffusionGemmaCoTGenerator.INPUT_TYPES()
        self.assertIn("grounding_guard_config", cot_inputs["optional"])
        self.assertEqual(
            self.nodes.DiffusionGemmaCoTGenerator.RETURN_NAMES[:4],
            ("final_json", "reasoning_text", "raw_response", "metadata_json"),
        )
        self.assertEqual(
            self.nodes.DiffusionGemmaCoTGenerator.RETURN_NAMES[4:],
            ("grounding_status", "grounding_report_json"),
        )

        config, config_json = self.nodes.DiffusionGemmaGroundingGuardSettings().build()
        self.assertEqual(config.mode, "audit")
        self.assertEqual(config.evidence_token_budget, "auto")
        self.assertEqual(json.loads(config_json)["sampling_settings"]["max_denoising_steps"], 48)
        self.assertEqual(json.loads(config_json)["evidence_token_budget"], "auto")

        legacy_config, _ = self.nodes.DiffusionGemmaGroundingGuardSettings().build(
            "strict",
            False,
            "checkpoint_defaults",
            7,
            False,
            "legacy_trace",
            '{"schema":"legacy-external"}',
        )
        self.assertEqual(legacy_config.evidence_token_budget, "auto")
        self.assertEqual(
            json.loads(legacy_config.external_evidence_json)["schema"],
            "legacy-external",
        )

        selected_config, selected_json = self.nodes.DiffusionGemmaGroundingGuardSettings().build(
            evidence_token_budget="1280"
        )
        self.assertEqual(selected_config.evidence_token_budget, "1280")
        self.assertEqual(json.loads(selected_json)["evidence_token_budget"], "1280")

    def test_strict_guard_block_survives_splitter_and_generation_gate(self) -> None:
        guard_reasons = [
            "visual_grounding_unverified",
            "visual_grounding_insufficient_coverage",
        ]
        guard_report = {
            "schema": "dg-grounding-report/1",
            "analysis_status": "uncertain",
            "decision": "block",
            "would_block": True,
            "grounding_guard_would_block": True,
            "blocked_reasons": guard_reasons,
            "retry_reasons": ["role coverage remained incomplete after retry"],
            "validation": {
                "errors": ["ledger:role_specific_category_missing:picture:2"]
            },
        }
        context = self.image_context()
        splitter = self.nodes.DiffusionGemmaJSONSplitter()
        gate = self.nodes.DiffusionGemmaGenerationGate()
        prompt_index = {"ltx": 0, "ideogram4": 1, "minimax_h3": 13}

        for profile in prompt_index:
            with self.subTest(profile=profile):
                target = self.nodes.TargetProfileConfig(
                    target_profile=profile,
                    minimax_h3_mode="t2va",
                )
                blocked_packet = self.nodes._blocked_grounding_packet(
                    target,
                    copy.deepcopy(guard_report),
                )

                split = splitter.split(json.dumps(blocked_packet), context, target)
                metadata = json.loads(split[4])

                self.assertFalse(split[12])
                self.assertEqual(split[0], "")
                self.assertEqual(split[1], "")
                self.assertEqual(split[13], "")
                self.assertEqual(metadata["blocked_reasons"], guard_reasons)
                self.assertFalse(metadata["claim_verification_passed"])

                with self.assertRaises(ValueError) as raised:
                    gate.gate(split[prompt_index[profile]], split[12], split[4])
                message = str(raised.exception)
                self.assertIn("Grounding Guard blocked generation", message)
                self.assertIn("visual_grounding_insufficient_coverage", message)
                self.assertIn(
                    "ledger:role_specific_category_missing:picture:2", message
                )
                self.assertNotIn("minimax_h3_ref_missing", message)
                self.assertNotIn("empty_generation_prompt", message)

        stale_metadata = {
            "blocked_reasons": ["empty_generation_prompt"],
            "grounding_guard": guard_report,
        }
        with self.assertRaisesRegex(
            ValueError, "visual_grounding_insufficient_coverage"
        ):
            gate.gate(
                "A stale splitter incorrectly supplied this fallback prompt.",
                True,
                json.dumps(stale_metadata),
            )

        for broken_decision in (None, "blok"):
            with self.subTest(broken_decision=broken_decision):
                malformed_report = copy.deepcopy(guard_report)
                malformed_report["config"] = {"mode": "strict"}
                if broken_decision is None:
                    malformed_report.pop("decision")
                else:
                    malformed_report["decision"] = broken_decision
                target = self.nodes.TargetProfileConfig(target_profile="ltx")
                blocked_packet = self.nodes._blocked_grounding_packet(
                    target,
                    malformed_report,
                )
                split = splitter.split(json.dumps(blocked_packet), context, target)
                self.assertFalse(split[12])
                self.assertEqual(split[0], "")
                self.assertEqual(
                    json.loads(split[4])["blocked_reasons"], guard_reasons
                )
                with self.assertRaisesRegex(
                    ValueError, "visual_grounding_insufficient_coverage"
                ):
                    gate.gate(split[0], split[12], split[4])

        audit_metadata = {
            "grounding_guard": {
                **guard_report,
                "decision": "warn",
                "config": {"mode": "audit"},
            }
        }
        self.assertEqual(
            gate.gate("Audit mode remains non-blocking.", True, json.dumps(audit_metadata)),
            ("Audit mode remains non-blocking.",),
        )

    def test_branch_generation_gate_blocks_only_its_prompt_output(self) -> None:
        gate = self.nodes.DiffusionGemmaBranchGenerationGate()
        metadata = {
            "blocked_reasons": [
                "json_parse_invalid",
                "minimax_h3_ref_missing_summary",
            ]
        }

        blocked_prompt, status, ready = gate.gate(
            "An invalid H3 prompt that must not reach the sampler.",
            False,
            json.dumps(metadata),
        )

        self.assertIsInstance(blocked_prompt, self.nodes.ExecutionBlocker)
        self.assertIsNone(blocked_prompt.message)
        self.assertFalse(ready)
        self.assertIn("did not pass validation", status)
        self.assertIn("json_parse_invalid", status)
        self.assertIn("minimax_h3_ref_missing_summary", status)
        self.assertIn("produced no media output", status)
        self.assertIn("select refresh once", status)

        valid_prompt = "A validated prompt for an independent comparison branch."
        self.assertEqual(
            gate.gate(valid_prompt, True, json.dumps({"blocked_reasons": []})),
            (
                valid_prompt,
                "DiffusionGemma branch is ready for generation.",
                True,
            ),
        )

    def test_declared_h3_asset_without_tensor_blocks_strict_before_generation(self) -> None:
        runtime = self.runtime()
        context = self.nodes.GemmaContext(
            user_prompt="Use the declared character reference.",
            images=None,
            source="image",
            media_metadata={
                "source": "image",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 1,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:identity,appearance] protagonist identity"
                ),
            },
        )
        target = self.nodes.TargetProfileConfig(
            target_profile="minimax_h3",
            minimax_h3_mode="ref2va",
        )
        with patch.object(
            self.nodes, "_run_generation_packet_legacy"
        ) as legacy, patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                target,
                grounding_guard_config={"mode": "strict"},
            )
        legacy.assert_not_called()
        unload.assert_called_once_with(runtime)
        self.assertEqual(result[0]["minimax_h3_prompt"], "")
        self.assertFalse(result[-1]["ready_for_generation"])
        self.assertEqual(
            result[-1]["blocked_reasons"][:2],
            ["visual_grounding_unverified", "visual_transport_error"],
        )
        self.assertEqual(result[-1]["grounding_guard"]["model_call_count"], 0)
        self.assertIn(
            "picture:1",
            result[-1]["grounding_guard"]["asset_registry"][
                "unattached_declared_visual_asset_ids"
            ],
        )

    def test_declared_h3_asset_without_tensor_only_warns_in_audit(self) -> None:
        runtime = self.runtime()
        context = self.nodes.GemmaContext(
            user_prompt="Use the declared character reference.",
            images=None,
            source="image",
            media_metadata={
                "source": "image",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 1,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:identity,appearance] protagonist identity"
                ),
            },
        )
        target = self.nodes.TargetProfileConfig(
            target_profile="minimax_h3",
            minimax_h3_mode="ref2va",
        )
        packet = self.packet("")
        packet["ltx_prompt"] = ""
        packet["minimax_h3_prompt"] = "legacy H3 prompt retained in audit"
        packet["metadata"]["target_profile"] = "minimax_h3"
        with patch.object(
            self.nodes,
            "_run_generation_packet_legacy",
            return_value=self.legacy_result(packet),
        ) as legacy, patch.object(self.nodes, "_maybe_unload"):
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                target,
                grounding_guard_config={"mode": "audit"},
            )
        legacy.assert_called_once()
        self.assertEqual(
            result[0]["minimax_h3_prompt"],
            "legacy H3 prompt retained in audit",
        )
        self.assertTrue(result[-1]["ready_for_generation"])
        self.assertEqual(result[-1]["grounding_guard"]["decision"], "warn")
        self.assertTrue(
            result[-1]["grounding_guard"]["grounding_guard_would_block"]
        )
        self.assertEqual(
            result[-1]["grounding_guard"]["blocked_reasons"][:2],
            ["visual_grounding_unverified", "visual_transport_error"],
        )

    def test_strict_evidence_rejects_prose_prefix_but_accepts_native_thinking(self) -> None:
        ledger = self.grounded_ledger(
            self.nodes._grounding_asset_registry(self.image_context())
        )
        exact = json.dumps(ledger)
        self.assertEqual(
            self.nodes._strict_evidence_json_payload(f"<think>check citations</think>\n{exact}"),
            exact,
        )
        with self.assertRaisesRegex(
            self.nodes.GroundingLedgerError,
            "without a conversational prefix",
        ):
            self.nodes._strict_evidence_json_payload(
                f"<|thought|>check citations<|end_thought|><|final_json|>{exact}<|end_final_json|>"
            )
        with self.assertRaisesRegex(
            self.nodes.GroundingLedgerError,
            "without a conversational prefix",
        ):
            self.nodes._strict_evidence_json_payload(f"Here is the evidence:\n{exact}")
        with self.assertRaisesRegex(
            self.nodes.GroundingLedgerError,
            "root must be the ledger",
        ):
            self.nodes._strict_evidence_json_payload(
                json.dumps({"grounding_ledger": ledger, "ltx_prompt": "must not pass"})
            )

    def test_strict_evidence_example_uses_the_authoritative_h3_asset_id(self) -> None:
        registry = self.nodes.build_asset_registry(
            {
                "source": "image",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 1,
                "minimax_h3_reference_manifest": (
                    "<Picture 1>: [dg:identity,appearance] protagonist identity"
                ),
            },
            1,
        )
        prompt = self.nodes._strict_evidence_prompt(registry)
        self.assertIn('"asset_id": "picture:1"', prompt)
        self.assertNotIn('"asset_id": "image:1"', prompt)
        self.assertIn("Use no more than 4 observed facts total", prompt)
        self.assertIn("Completing and closing the strict JSON object takes priority", prompt)
        retry_prompt = self.nodes._strict_evidence_prompt(registry, retry=True)
        self.assertIn("previous ledger did not pass validation", retry_prompt)
        self.assertIn("Keep every JSON object and array closed", retry_prompt)
        self.assertEqual(self.nodes._strict_evidence_max_new_tokens(2048), 768)
        self.assertEqual(self.nodes._strict_evidence_max_new_tokens(2048, retry=True), 1024)
        for budget, expected in (("768", 768), ("1024", 1024), ("1280", 1280)):
            with self.subTest(evidence_token_budget=budget):
                self.assertEqual(
                    self.nodes._strict_evidence_max_new_tokens(
                        512,
                        evidence_token_budget=budget,
                    ),
                    expected,
                )
                self.assertEqual(
                    self.nodes._strict_evidence_max_new_tokens(
                        512,
                        retry=True,
                        evidence_token_budget=budget,
                    ),
                    expected,
                )

    def test_h3_grounding_annotations_are_host_only(self) -> None:
        manifest = (
            "<Picture 1>: [dg:identity,appearance] protagonist identity\n"
            "<Video 1>: [dg:action,motion,camera,temporal] movement reference"
        )
        context = self.nodes.GemmaContext(
            user_prompt="Use the references faithfully.",
            images=torch.zeros((3, 8, 8, 3), dtype=torch.float32),
            source="image+video",
            media_metadata={
                "source": "image+video",
                "minimax_h3_mode": "ref2va",
                "minimax_h3_reference_image_batch_count": 1,
                "minimax_h3_reference_manifest": manifest,
                "sampled_indices": [0, 10],
                "source_fps": 10.0,
            },
        )
        registry = self.nodes._grounding_asset_registry(context)
        compiler_metadata = self.nodes._strict_compiler_metadata(
            context,
            self.grounded_ledger(registry),
            "dggr-test",
            {"pixel_transport_confirmed": True},
            registry,
        )
        self.assertNotIn("[dg:", json.dumps(compiler_metadata).lower())

        model_prompt = self.nodes._build_model_prompt(
            context.user_prompt,
            self.nodes.DEFAULT_MASTER_PROMPT,
            "minimax_h3",
            context.media_metadata,
            minimax_h3_mode="ref2va",
            minimax_h3_reference_manifest=manifest,
        )
        self.assertNotIn("[dg:", model_prompt.lower())
        self.assertIn("<Picture 1>: protagonist identity", model_prompt)
        self.assertIn(
            "minimax_h3_prompt_contains_grounding_role_annotation",
            self.nodes._minimax_h3_ref2va_validation_reasons(
                "subject_definitions: [dg:identity] leaked annotation",
                5.0,
                reference_manifest=manifest,
            ),
        )
        refinement_prompt = self.nodes._build_minimax_h3_refinement_prompt(
            context.user_prompt,
            "candidate prompt",
            5.0,
            "auto_scene_audio",
            ["minimax_h3_ref_missing_asset_tag"],
            "ref2va",
            manifest,
        )
        self.assertNotIn("[dg:", refinement_prompt.lower())
        self.assertIn("<Picture 1>: protagonist identity", refinement_prompt)

    def test_strict_prose_prefixed_ledger_blocks_without_compiler(self) -> None:
        runtime = self.runtime()
        ledger = self.grounded_ledger(
            self.nodes._grounding_asset_registry(self.image_context())
        )
        result_with_prefix = self.backend_result(ledger)
        result_with_prefix.decoded_text = "Here is the evidence:\n" + json.dumps(ledger)
        with patch.object(
            self.nodes,
            "_run_backend_detailed",
            return_value=result_with_prefix,
        ), patch.object(
            self.nodes, "_run_generation_packet_legacy"
        ) as compiler, patch.object(self.nodes, "_maybe_unload"):
            result = self.nodes._run_generation_packet(
                runtime,
                self.image_context(),
                self.target(),
                grounding_guard_config={"mode": "strict", "retry_on_uncertain": False},
            )
        compiler.assert_not_called()
        self.assertEqual(result[0]["ltx_prompt"], "")
        self.assertEqual(
            result[-1]["blocked_reasons"][:2],
            ["visual_grounding_unverified", "visual_grounding_schema_invalid"],
        )

    def test_unattached_identity_still_never_consumes_the_first_video_frame(self) -> None:
        frame_markers = [object() for _ in range(6)]
        context = self.nodes.GemmaContext(
            user_prompt="Use the still identity and video action.",
            images=torch.zeros((6, 8, 8, 3), dtype=torch.float32),
            source="video",
            media_metadata={
                "source": "video",
                "media_synthesis_mode": "image_identity_video_control",
                "reference_image_count": 1,
                "reference_image_backend_attached": False,
                "sampled_indices": [0, 10, 20, 30, 40, 50],
                "video_sampled_frame_count": 6,
                "source_fps": 10.0,
            },
        )

        with patch.object(self.nodes, "_image_batch_to_pil", return_value=frame_markers):
            messages, _kwargs = self.nodes._transformers_messages_and_processor_kwargs(
                "Describe only verified evidence.",
                self.nodes.MediaContext(
                    images=context.images,
                    source=context.source,
                    metadata=context.media_metadata,
                ),
            )
        content = messages[0]["content"]
        rendered_text = "\n".join(
            item["text"] for item in content if item.get("type") == "text"
        )
        transported_images = [item["image"] for item in content if item.get("type") == "image"]
        self.assertIn("IDENTITY REFERENCE MISSING", rendered_text)
        self.assertNotIn("IDENTITY REFERENCE IMAGE(S)", rendered_text)
        self.assertEqual(transported_images, frame_markers)

        retry = self.nodes._focused_retry_context(context)
        self.assertEqual(int(retry.images.shape[0]), 3)
        self.assertEqual(retry.media_metadata["sampled_indices"], [0, 30, 50])
        self.assertEqual(
            retry.media_metadata["grounding_retry_frame_reduction"]["original_video_sample_count"],
            6,
        )

    def test_rank_four_native_video_transport_uses_processor_token_fallback(self) -> None:
        registry = self.nodes.build_asset_registry(
            {
                "source": "video",
                "sampled_indices": [0, 10, 20],
                "source_fps": 10.0,
                "video_role": "primary_video_source",
            },
            3,
        )
        inputs = {
            "input_ids": torch.tensor([[1, 222, 2]], dtype=torch.long),
            "pixel_values_videos": torch.zeros((1, 3, 2, 4)),
            "video_position_ids": torch.zeros((1, 3, 2, 2), dtype=torch.long),
        }
        model = SimpleNamespace(
            config=SimpleNamespace(image_token_id=111, video_token_id=None)
        )
        processor = SimpleNamespace(image_token_id=111, video_token_id=222)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": [object(), object(), object()]},
                    {"type": "text", "text": "Describe."},
                ],
            }
        ]

        proof = self.nodes._processor_transport_proof(
            inputs, model, processor, messages, registry
        )
        self.assertTrue(proof["pixel_transport_confirmed"])
        self.assertEqual(proof["processor_observed_sample_count"], 3)
        self.assertEqual(proof["video_media_token_count"], 1)
        self.assertEqual(proof["processor_observed_position_sample_count"], 3)
        self.assertEqual(proof["effective_visual_token_budget"], 1)
        self.assertEqual(proof["processor_pixel_patch_count"], 6)

    def test_transport_rejects_incomplete_tokens_and_position_sample_counts(self) -> None:
        registry = self.nodes.build_asset_registry(
            {"source": "video", "sampled_indices": [0, 1, 2]},
            3,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": object()},
                    {"type": "image", "image": object()},
                    {"type": "image", "image": object()},
                    {"type": "text", "text": "Describe."},
                ],
            }
        ]
        processor = SimpleNamespace(image_token_id=111, video_token_id=222)
        model = SimpleNamespace(config=SimpleNamespace(image_token_id=111, video_token_id=222))
        inputs = {
            "input_ids": torch.tensor([[1, 111, 2]], dtype=torch.long),
            "pixel_values": torch.zeros((3, 2, 4)),
            "image_position_ids": torch.zeros((2, 2, 2), dtype=torch.long),
        }
        proof = self.nodes._processor_transport_proof(
            inputs, model, processor, messages, registry
        )
        self.assertFalse(proof["pixel_transport_confirmed"])
        self.assertTrue(
            any("media_token_coverage_mismatch" in reason for reason in proof["mismatch_reasons"])
        )
        self.assertTrue(
            any("position_sample_count_mismatch" in reason for reason in proof["mismatch_reasons"])
        )

    def test_missing_telemetry_callbacks_are_detected_as_integrity_errors(self) -> None:
        result = self.nodes.BackendRunResult(
            decoded_text="{}",
            telemetry={
                "schema_version": "dg-denoising-telemetry/1",
                "forward_count": 0,
                "snapshots": [],
                "stream_ended": False,
                "final_output": {},
                "errors": [],
            },
        )
        error = self.nodes._telemetry_error_text(result)
        self.assertIn("logits_processor_callback_not_observed", error)
        self.assertIn("draft_streamer_snapshot_not_observed", error)
        self.assertIn("draft_streamer_end_not_observed", error)

    def test_off_delegates_to_frozen_legacy_path(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        expected_packet = self.packet("legacy prompt")
        expected_packet["metadata"]["media"] = {
            "pixels_sent_to_backend": True,
            "pixel_transport_confirmed": True,
        }
        expected_result = self.legacy_result(expected_packet)
        with patch.object(
            self.nodes, "_run_generation_packet_legacy", return_value=expected_result
        ) as legacy, patch.object(self.nodes, "_run_backend_detailed") as detailed:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={"mode": "off"},
            )

        legacy.assert_called_once()
        detailed.assert_not_called()
        self.assertEqual(result[0]["ltx_prompt"], "legacy prompt")
        self.assertEqual(result[0]["metadata"]["grounding_guard"]["decision"], "disabled")
        self.assertTrue(result[0]["metadata"]["media"]["pixels_sent_to_backend"])
        self.assertTrue(result[0]["metadata"]["media"]["pixel_transport_confirmed"])
        self.assertEqual(
            result[0]["metadata"]["grounding_guard"]["effective_sampling"]["profile"],
            "legacy_scalar_temperature",
        )
        self.assertNotIn("backend_options", legacy.call_args.kwargs)

    def test_audit_warning_does_not_change_readiness_or_prompt(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        ledger = self.uncertain_ledger()
        initial_packet = self.packet("keep this prompt", ledger)

        captured_kwargs: dict = {}

        def audit_legacy(*_args, **kwargs):
            captured_kwargs.update(kwargs)
            kwargs["backend_results"].append(
                self.backend_result(ledger, transport_confirmed=True)
            )
            return self.legacy_result(initial_packet)

        with patch.object(
            self.nodes, "_run_generation_packet_legacy", side_effect=audit_legacy
        ), patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={"mode": "audit", "seed": 9},
            )

        packet, metadata = result[0], result[-1]
        self.assertEqual(packet["ltx_prompt"], "keep this prompt")
        self.assertTrue(metadata["ready_for_generation"])
        self.assertEqual(metadata["blocked_reasons"], [])
        self.assertEqual(metadata["grounding_guard"]["decision"], "warn")
        self.assertTrue(metadata["grounding_guard"]["grounding_guard_would_block"])
        self.assertNotIn("grounding_ledger", packet)
        self.assertNotIn("max_refinement_attempts_override", captured_kwargs)
        unload.assert_called_once_with(runtime)

    def test_audit_h3_parse_invalid_retries_once_and_uses_retry_result(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        registry = self.nodes._grounding_asset_registry(context)
        ledger = self.grounded_ledger(registry)
        target = self.nodes.TargetProfileConfig(
            target_profile="minimax_h3",
            minimax_h3_mode="ref2va",
        )

        invalid_packet = self.packet("")
        invalid_packet["ltx_prompt"] = ""
        invalid_packet["minimax_h3_prompt"] = "salvaged first attempt"
        invalid_packet["metadata"]["target_profile"] = "minimax_h3"
        invalid_packet["metadata"]["ready_for_generation"] = False

        valid_packet = copy.deepcopy(invalid_packet)
        valid_packet["minimax_h3_prompt"] = "strict JSON retry result"
        valid_packet["metadata"]["ready_for_generation"] = True
        valid_packet["metadata"]["blocked_reasons"] = []

        observed_stages: list[str] = []

        def audit_legacy(*_args, **kwargs):
            backend_options = kwargs["backend_options"]
            backend_options.call_budget.claim()
            stage = backend_options.stage
            observed_stages.append(stage)
            result = self.backend_result(ledger, transport_confirmed=True)
            result.stage = stage
            kwargs["backend_results"].append(result)
            if stage == "audit_director":
                return (
                    copy.deepcopy(invalid_packet),
                    "not strict JSON",
                    "",
                    "",
                    False,
                    "salvaged malformed output",
                    copy.deepcopy(invalid_packet["metadata"]),
                )
            return self.legacy_result(copy.deepcopy(valid_packet))

        with patch.object(
            self.nodes, "_run_generation_packet_legacy", side_effect=audit_legacy
        ) as legacy, patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                target,
                grounding_guard_config={
                    "mode": "audit",
                    "retry_on_uncertain": True,
                },
            )

        packet, metadata = result[0], result[-1]
        report = metadata["grounding_guard"]
        self.assertEqual(legacy.call_count, 2)
        self.assertEqual(observed_stages, ["audit_director", "audit_director_retry"])
        self.assertEqual(packet["minimax_h3_prompt"], "strict JSON retry result")
        self.assertTrue(metadata["ready_for_generation"])
        self.assertEqual(report["attempt_count"], 2)
        self.assertEqual(report["model_call_count"], 2)
        self.assertEqual(
            report["retry_reasons"],
            ["audit_retry:prompt_packet_was_not_strict_json"],
        )
        self.assertEqual(
            [call["stage"] for call in report["trajectory_summary"]["calls"]],
            ["audit_director", "audit_director_retry"],
        )
        unload.assert_called_once_with(runtime)

    def test_audit_h3_parse_invalid_does_not_retry_when_retry_is_disabled(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        ledger = self.grounded_ledger(
            self.nodes._grounding_asset_registry(context)
        )
        target = self.nodes.TargetProfileConfig(
            target_profile="minimax_h3",
            minimax_h3_mode="ref2va",
        )
        invalid_packet = self.packet("")
        invalid_packet["ltx_prompt"] = ""
        invalid_packet["minimax_h3_prompt"] = "salvaged invalid H3 output"
        invalid_packet["metadata"].update(
            {
                "target_profile": "minimax_h3",
                "ready_for_generation": False,
                "blocked_reasons": ["json_parse_invalid", "salvaged_output"],
            }
        )
        observed_stages: list[str] = []

        def audit_legacy(*_args, **kwargs):
            backend_options = kwargs["backend_options"]
            backend_options.call_budget.claim()
            observed_stages.append(backend_options.stage)
            result = self.backend_result(ledger, transport_confirmed=True)
            result.stage = backend_options.stage
            kwargs["backend_results"].append(result)
            return (
                copy.deepcopy(invalid_packet),
                "not strict JSON",
                "",
                "",
                False,
                "salvaged malformed output",
                copy.deepcopy(invalid_packet["metadata"]),
            )

        with patch.object(
            self.nodes, "_run_generation_packet_legacy", side_effect=audit_legacy
        ) as legacy, patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                target,
                grounding_guard_config={
                    "mode": "audit",
                    "retry_on_uncertain": False,
                },
            )

        metadata = result[-1]
        report = metadata["grounding_guard"]
        self.assertEqual(legacy.call_count, 1)
        self.assertEqual(observed_stages, ["audit_director"])
        self.assertFalse(result[4])
        self.assertTrue(result[5])
        self.assertFalse(metadata["ready_for_generation"])
        self.assertEqual(report["attempt_count"], 1)
        self.assertEqual(report["model_call_count"], 1)
        self.assertEqual(report["retry_reasons"], [])
        unload.assert_called_once_with(runtime)

    def test_audit_ltx_parse_invalid_retries_once_and_uses_retry_result(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        ledger = self.grounded_ledger(
            self.nodes._grounding_asset_registry(context)
        )
        invalid_packet = self.packet("salvaged invalid LTX output")
        invalid_packet["metadata"].update(
            {
                "ready_for_generation": False,
                "blocked_reasons": ["json_parse_invalid", "salvaged_output"],
            }
        )
        valid_packet = self.packet("strict JSON LTX retry result")
        observed_stages: list[str] = []

        def audit_legacy(*_args, **kwargs):
            backend_options = kwargs["backend_options"]
            backend_options.call_budget.claim()
            observed_stages.append(backend_options.stage)
            result = self.backend_result(ledger, transport_confirmed=True)
            result.stage = backend_options.stage
            kwargs["backend_results"].append(result)
            if backend_options.stage == "audit_director_retry":
                return self.legacy_result(copy.deepcopy(valid_packet))
            return (
                copy.deepcopy(invalid_packet),
                "not strict JSON",
                "",
                "",
                False,
                "salvaged malformed output",
                copy.deepcopy(invalid_packet["metadata"]),
            )

        with patch.object(
            self.nodes, "_run_generation_packet_legacy", side_effect=audit_legacy
        ) as legacy, patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={
                    "mode": "audit",
                    "retry_on_uncertain": True,
                },
            )

        packet, metadata = result[0], result[-1]
        report = metadata["grounding_guard"]
        self.assertEqual(legacy.call_count, 2)
        self.assertEqual(observed_stages, ["audit_director", "audit_director_retry"])
        self.assertEqual(packet["ltx_prompt"], "strict JSON LTX retry result")
        self.assertTrue(result[4])
        self.assertFalse(result[5])
        self.assertTrue(metadata["ready_for_generation"])
        self.assertEqual(report["attempt_count"], 2)
        self.assertEqual(report["model_call_count"], 2)
        self.assertEqual(
            report["retry_reasons"],
            ["audit_retry:prompt_packet_was_not_strict_json"],
        )
        self.assertEqual(
            [call["stage"] for call in report["trajectory_summary"]["calls"]],
            ["audit_director", "audit_director_retry"],
        )
        unload.assert_called_once_with(runtime)

    def test_audit_ltx_schema_invalid_ledger_retries_once(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        registry = self.nodes._grounding_asset_registry(context)
        valid_ledger = self.grounded_ledger(registry)
        schema_invalid_ledger = {
            "schema": "dg-grounding-ledger/1",
            "analysis_status": "grounded",
        }
        packet = self.packet("valid LTX packet on both audit attempts")
        observed_stages: list[str] = []

        def audit_legacy(*_args, **kwargs):
            backend_options = kwargs["backend_options"]
            backend_options.call_budget.claim()
            stage = backend_options.stage
            observed_stages.append(stage)
            ledger = (
                valid_ledger
                if stage == "audit_director_retry"
                else schema_invalid_ledger
            )
            result = self.backend_result(ledger, transport_confirmed=True)
            result.stage = stage
            kwargs["backend_results"].append(result)
            return self.legacy_result(copy.deepcopy(packet))

        with patch.object(
            self.nodes, "_run_generation_packet_legacy", side_effect=audit_legacy
        ) as legacy, patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={
                    "mode": "audit",
                    "retry_on_uncertain": True,
                },
            )

        metadata = result[-1]
        report = metadata["grounding_guard"]
        self.assertEqual(legacy.call_count, 2)
        self.assertEqual(observed_stages, ["audit_director", "audit_director_retry"])
        self.assertTrue(metadata["ready_for_generation"])
        self.assertEqual(report["attempt_count"], 2)
        self.assertEqual(report["model_call_count"], 2)
        self.assertEqual(
            report["retry_reasons"],
            ["audit_retry:grounding_ledger_schema_invalid"],
        )
        self.assertTrue(report["validation"]["schema_valid"])
        self.assertEqual(report["validation"]["blocking_reasons"], [])
        self.assertEqual(
            [call["stage"] for call in report["trajectory_summary"]["calls"]],
            ["audit_director", "audit_director_retry"],
        )
        unload.assert_called_once_with(runtime)

    def test_audit_ltx_parse_invalid_retry_exhaustion_remains_fail_closed(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        ledger = self.grounded_ledger(
            self.nodes._grounding_asset_registry(context)
        )
        observed_stages: list[str] = []

        def audit_legacy(*_args, **kwargs):
            backend_options = kwargs["backend_options"]
            backend_options.call_budget.claim()
            stage = backend_options.stage
            observed_stages.append(stage)
            backend_result = self.backend_result(ledger, transport_confirmed=True)
            backend_result.stage = stage
            kwargs["backend_results"].append(backend_result)
            invalid_packet = self.packet(
                f"salvaged invalid LTX output from {stage}"
            )
            invalid_packet["metadata"].update(
                {
                    "ready_for_generation": False,
                    "blocked_reasons": ["json_parse_invalid", "salvaged_output"],
                }
            )
            return (
                invalid_packet,
                "not strict JSON",
                "",
                "",
                False,
                "salvaged malformed output",
                invalid_packet["metadata"],
            )

        with patch.object(
            self.nodes, "_run_generation_packet_legacy", side_effect=audit_legacy
        ) as legacy, patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={
                    "mode": "audit",
                    "retry_on_uncertain": True,
                },
            )

        packet, metadata = result[0], result[-1]
        report = metadata["grounding_guard"]
        self.assertEqual(legacy.call_count, 2)
        self.assertEqual(observed_stages, ["audit_director", "audit_director_retry"])
        self.assertFalse(result[4])
        self.assertTrue(result[5])
        self.assertFalse(metadata["ready_for_generation"])
        self.assertIn("json_parse_invalid", metadata["blocked_reasons"])
        self.assertEqual(report["attempt_count"], 2)
        self.assertEqual(report["model_call_count"], 2)
        self.assertEqual(
            report["retry_reasons"],
            ["audit_retry:prompt_packet_was_not_strict_json"],
        )
        with self.assertRaisesRegex(ValueError, "did not pass validation"):
            self.nodes.DiffusionGemmaGenerationGate().gate(
                packet["ltx_prompt"],
                metadata["ready_for_generation"],
                json.dumps(metadata),
            )
        unload.assert_called_once_with(runtime)

    def test_audit_h3_parse_invalid_retry_exhaustion_remains_fail_closed(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        ledger = self.grounded_ledger(
            self.nodes._grounding_asset_registry(context)
        )
        target = self.nodes.TargetProfileConfig(
            target_profile="minimax_h3",
            minimax_h3_mode="ref2va",
        )
        observed_stages: list[str] = []

        def audit_legacy(*_args, **kwargs):
            backend_options = kwargs["backend_options"]
            backend_options.call_budget.claim()
            observed_stages.append(backend_options.stage)
            result = self.backend_result(ledger, transport_confirmed=True)
            result.stage = backend_options.stage
            kwargs["backend_results"].append(result)
            packet = self.packet("")
            packet["ltx_prompt"] = ""
            packet["minimax_h3_prompt"] = (
                f"salvaged invalid H3 output from {backend_options.stage}"
            )
            packet["metadata"].update(
                {
                    "target_profile": "minimax_h3",
                    "ready_for_generation": False,
                    "blocked_reasons": [
                        "json_parse_invalid",
                        "salvaged_output",
                        "minimax_h3_ref_missing_summary",
                    ],
                }
            )
            return (
                packet,
                "not strict JSON",
                "",
                "",
                False,
                "salvaged malformed output",
                packet["metadata"],
            )

        with patch.object(
            self.nodes, "_run_generation_packet_legacy", side_effect=audit_legacy
        ) as legacy, patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                target,
                grounding_guard_config={
                    "mode": "audit",
                    "retry_on_uncertain": True,
                },
            )

        packet, metadata = result[0], result[-1]
        report = metadata["grounding_guard"]
        self.assertEqual(legacy.call_count, 2)
        self.assertEqual(observed_stages, ["audit_director", "audit_director_retry"])
        self.assertFalse(result[4])
        self.assertTrue(result[5])
        self.assertFalse(metadata["ready_for_generation"])
        self.assertIn("json_parse_invalid", metadata["blocked_reasons"])
        self.assertEqual(report["attempt_count"], 2)
        self.assertEqual(report["model_call_count"], 2)
        self.assertEqual(
            report["retry_reasons"],
            ["audit_retry:prompt_packet_was_not_strict_json"],
        )
        self.assertEqual(
            [call["stage"] for call in report["trajectory_summary"]["calls"]],
            ["audit_director", "audit_director_retry"],
        )
        with self.assertRaisesRegex(ValueError, "did not pass validation"):
            self.nodes.DiffusionGemmaGenerationGate().gate(
                packet["minimax_h3_prompt"],
                metadata["ready_for_generation"],
                json.dumps(metadata),
            )
        unload.assert_called_once_with(runtime)

    def test_audit_preserves_initial_ledger_and_aggregates_repair_telemetry(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        context.visual_description = "unverified purple dragon description"
        context.media_metadata["visual_description"] = context.visual_description
        context.media_metadata["observations"] = ["unverified purple dragon observation"]
        registry = self.nodes._grounding_asset_registry(context)
        ledger = self.grounded_ledger(registry)
        combined_packet = self.packet("initial prompt", ledger)
        initial = self.backend_result(ledger)
        initial.decoded_text = json.dumps(combined_packet)
        initial.stage = "audit_director"
        repair = self.backend_result(ledger, telemetry_refusal=True)
        repair.decoded_text = "I cannot help with this request."
        repair.stage = "target_repair_1"
        repaired_packet = self.packet("accepted repaired prompt")

        def audit_legacy(*_args, **kwargs):
            kwargs["backend_results"].extend([initial, repair])
            return self.legacy_result(repaired_packet)

        with patch.object(
            self.nodes, "_run_generation_packet_legacy", side_effect=audit_legacy
        ), patch.object(self.nodes, "_maybe_unload"):
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={"mode": "audit"},
            )

        packet, report = result[0], result[-1]["grounding_guard"]
        self.assertEqual(packet["ltx_prompt"], "accepted repaired prompt")
        self.assertTrue(packet["metadata"]["ready_for_generation"])
        self.assertEqual(report["validated_ledger"]["analysis_status"], "grounded")
        self.assertEqual(report["decision"], "warn")
        self.assertEqual(report["analysis_status"], "refused")
        self.assertEqual(report["model_call_count"], 2)
        self.assertEqual(
            [call["stage"] for call in report["trajectory_summary"]["calls"]],
            ["audit_director", "target_repair_1"],
        )

    def test_strict_refusal_blocks_without_compiler_and_unloads_once(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        refused = self.uncertain_ledger("refused")
        refused["uncertainties"] = []
        refused["grounding_failure_reasons"] = ["I cannot analyze this image."]
        with patch.object(
            self.nodes,
            "_run_backend_detailed",
            return_value=self.backend_result(refused, telemetry_refusal=True),
        ) as evidence, patch.object(
            self.nodes, "_run_generation_packet_legacy"
        ) as compiler, patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={
                    "mode": "strict",
                    "retry_on_uncertain": False,
                },
            )

        packet, metadata = result[0], result[-1]
        evidence.assert_called_once()
        compiler.assert_not_called()
        self.assertEqual(packet["ltx_prompt"], "")
        self.assertFalse(metadata["ready_for_generation"])
        self.assertEqual(
            metadata["blocked_reasons"][:2],
            ["visual_grounding_unverified", "visual_grounding_refused"],
        )
        unload.assert_called_once_with(runtime)

    def test_strict_grounded_pass_compiles_from_ledger_without_pixels(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        registry = self.nodes._grounding_asset_registry(context)
        ledger = self.grounded_ledger(registry)
        captured: dict = {}

        def compiler(*args, **kwargs):
            compiler_context = args[1]
            captured["context"] = compiler_context
            captured["options"] = kwargs["backend_options"]
            captured["max_repairs"] = kwargs["max_refinement_attempts_override"]
            packet = self.packet("ledger-only compiler prompt")
            packet["metadata"]["grounding_evidence_report_id"] = compiler_context.media_metadata[
                "grounding_evidence_report_id"
            ]
            packet["metadata"]["used_grounding_fact_ids"] = [
                fact["fact_id"]
                for fact in compiler_context.media_metadata["verified_grounding_ledger"][
                    "observed_facts"
                ]
            ]
            packet["grounding_ledger"] = {"spoofed": True}
            return self.legacy_result(packet)

        with patch.object(
            self.nodes,
            "_run_backend_detailed",
            return_value=self.backend_result(ledger),
        ), patch.object(
            self.nodes, "_run_generation_packet_legacy", side_effect=compiler
        ) as compiler_mock, patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={"mode": "strict", "seed": 17},
            )

        compiler_mock.assert_called_once()
        compiler_context = captured["context"]
        self.assertIsNone(compiler_context.images)
        self.assertEqual(
            compiler_context.media_metadata["verified_grounding_ledger"], ledger
        )
        self.assertFalse(compiler_context.media_metadata["pixels_sent_to_backend"])
        self.assertNotIn("visual_description", compiler_context.media_metadata)
        self.assertNotIn("observations", compiler_context.media_metadata)
        self.assertNotIn("purple dragon", json.dumps(compiler_context.media_metadata))
        self.assertEqual(captured["options"].asset_registry["expected_image_count"], 0)
        self.assertEqual(captured["max_repairs"], 2)
        packet, metadata = result[0], result[-1]
        self.assertEqual(packet["ltx_prompt"], "ledger-only compiler prompt")
        self.assertNotIn("grounding_ledger", packet)
        self.assertEqual(metadata["grounding_guard"]["decision"], "pass")
        self.assertEqual(
            metadata["grounding_evidence_report_id"],
            metadata["grounding_guard"]["evidence_report_id"],
        )
        self.assertEqual(
            metadata["used_grounding_fact_ids"],
            [fact["fact_id"] for fact in ledger["observed_facts"]],
        )
        self.assertEqual(
            metadata["available_grounding_fact_ids"],
            [fact["fact_id"] for fact in ledger["observed_facts"]],
        )
        self.assertFalse(metadata["grounding_compiler_pixels_sent_to_backend"])
        self.assertEqual(
            metadata["grounding_guard"]["compiler_transport"],
            {
                "pixels_sent_to_backend": False,
                "evidence_source": "validated_ledger_only",
            },
        )
        unload.assert_called_once_with(runtime)

    def test_strict_compiler_unknown_fact_provenance_blocks(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        ledger = self.grounded_ledger(self.nodes._grounding_asset_registry(context))

        def compiler(*args, **_kwargs):
            compiler_context = args[1]
            packet = self.packet("untrusted compiler prompt")
            packet["metadata"]["grounding_evidence_report_id"] = compiler_context.media_metadata[
                "grounding_evidence_report_id"
            ]
            packet["metadata"]["used_grounding_fact_ids"] = ["invented-fact-id"]
            return self.legacy_result(packet)

        with patch.object(
            self.nodes, "_run_backend_detailed", return_value=self.backend_result(ledger)
        ), patch.object(
            self.nodes, "_run_generation_packet_legacy", side_effect=compiler
        ), patch.object(self.nodes, "_maybe_unload"):
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={"mode": "strict", "retry_on_uncertain": False},
            )

        self.assertEqual(result[0]["ltx_prompt"], "")
        self.assertFalse(result[-1]["ready_for_generation"])
        self.assertEqual(
            result[-1]["blocked_reasons"][:2],
            ["visual_grounding_unverified", "visual_grounding_schema_invalid"],
        )

    def test_strict_compiler_readiness_failure_preserves_validator_reasons(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        ledger = self.grounded_ledger(self.nodes._grounding_asset_registry(context))

        def compiler(*args, **_kwargs):
            compiler_context = args[1]
            packet = self.packet("")
            packet["metadata"]["ready_for_generation"] = False
            packet["metadata"]["blocked_reasons"] = [
                "minimax_h3_cut_timestamp_invalid"
            ]
            packet["metadata"]["grounding_evidence_report_id"] = (
                compiler_context.media_metadata["grounding_evidence_report_id"]
            )
            packet["metadata"]["used_grounding_fact_ids"] = [
                fact["fact_id"]
                for fact in compiler_context.media_metadata["verified_grounding_ledger"][
                    "observed_facts"
                ]
            ]
            packet["metadata"]["minimax_h3_refinement"] = {
                "candidate_reasons": ["minimax_h3_noncanonical_time_range"]
            }
            return self.legacy_result(packet)

        with patch.object(
            self.nodes, "_run_backend_detailed", return_value=self.backend_result(ledger)
        ), patch.object(
            self.nodes, "_run_generation_packet_legacy", side_effect=compiler
        ), patch.object(self.nodes, "_maybe_unload"):
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={"mode": "strict", "retry_on_uncertain": False},
            )

        packet, metadata = result[0], result[-1]
        self.assertEqual(packet["ltx_prompt"], "")
        self.assertFalse(metadata["ready_for_generation"])
        retry_detail = metadata["grounding_guard"]["retry_reasons"][0]
        self.assertIn("minimax_h3_cut_timestamp_invalid", retry_detail)
        self.assertIn("minimax_h3_noncanonical_time_range", retry_detail)

        gate = self.nodes.DiffusionGemmaGenerationGate()
        with self.assertRaises(ValueError) as raised:
            gate.gate("", False, json.dumps(metadata))
        self.assertIn("minimax_h3_cut_timestamp_invalid", str(raised.exception))
        self.assertIn("minimax_h3_noncanonical_time_range", str(raised.exception))

    def test_strict_compiler_retries_malformed_json_once_within_call_budget(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        ledger = self.grounded_ledger(self.nodes._grounding_asset_registry(context))
        calls: list[dict] = []

        def compiler(*args, **kwargs):
            compiler_context = args[1]
            options = kwargs["backend_options"]
            options.call_budget.claim()
            calls.append(
                {
                    "stage": options.stage,
                    "seed": options.seed,
                    "max_repairs": kwargs["max_refinement_attempts_override"],
                }
            )
            packet = self.packet("ledger-only compiler prompt")
            if len(calls) == 1:
                return packet, "malformed compiler output", "", "", False, "", packet["metadata"]
            packet["metadata"]["grounding_evidence_report_id"] = (
                compiler_context.media_metadata["grounding_evidence_report_id"]
            )
            packet["metadata"]["used_grounding_fact_ids"] = [
                fact["fact_id"]
                for fact in compiler_context.media_metadata["verified_grounding_ledger"][
                    "observed_facts"
                ]
            ]
            return self.legacy_result(packet)

        with patch.object(
            self.nodes, "_run_backend_detailed", return_value=self.backend_result(ledger)
        ), patch.object(
            self.nodes, "_run_generation_packet_legacy", side_effect=compiler
        ), patch.object(self.nodes, "_maybe_unload"):
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={"mode": "strict", "seed": 9},
            )

        self.assertEqual(
            calls,
            [
                {"stage": "compiler", "seed": 1009, "max_repairs": 2},
                {"stage": "compiler_retry", "seed": 1009, "max_repairs": 1},
            ],
        )
        packet, metadata = result[0], result[-1]
        self.assertEqual(packet["ltx_prompt"], "ledger-only compiler prompt")
        self.assertTrue(metadata["ready_for_generation"])
        self.assertEqual(metadata["grounding_guard"]["model_call_count"], 3)
        self.assertIn(
            "compiler_retry:compiler_packet_was_not_strict_json",
            metadata["grounding_guard"]["retry_reasons"],
        )

    def test_compiler_provenance_must_cover_every_required_asset(self) -> None:
        metadata = {
            "grounding_evidence_report_id": "dggr-test",
            "used_grounding_fact_ids": ["fact-image"],
        }
        fact_assets = {
            "fact-image": {"image:1"},
            "fact-video": {"video:1"},
        }
        with self.assertRaisesRegex(
            self.nodes.GroundingLedgerError,
            "compiler_fact_coverage_missing_assets:video:1",
        ):
            self.nodes._validated_compiler_provenance(
                metadata,
                "dggr-test",
                ["fact-image", "fact-video"],
                fact_assets,
                ["image:1", "video:1"],
            )

        metadata["used_grounding_fact_ids"] = ["fact-image", "fact-video"]
        self.assertEqual(
            self.nodes._validated_compiler_provenance(
                metadata,
                "dggr-test",
                ["fact-image", "fact-video"],
                fact_assets,
                ["image:1", "video:1"],
            ),
            ["fact-image", "fact-video"],
        )

    def test_h3_target_repair_uses_only_the_same_ledger_context(self) -> None:
        runtime = self.runtime()
        ledger = self.grounded_ledger(
            self.nodes._grounding_asset_registry(self.image_context())
        )
        compiler_context = self.nodes.GemmaContext(
            user_prompt="Create a concise H3 shot.",
            images=None,
            source="image",
            media_metadata={
                "source": "image",
                "verified_grounding_ledger": ledger,
                "pixels_sent_to_backend": False,
                "visual_grounding_mode": "verified_ledger",
            },
        )
        target = self.nodes.TargetProfileConfig(
            target_profile="minimax_h3",
            minimax_h3_mode="t2va",
        )
        invalid_packet = self.packet("")
        invalid_packet["minimax_h3_prompt"] = "invalid timeline"
        invalid_packet["metadata"]["target_profile"] = "minimax_h3"
        invalid_packet["metadata"]["grounding_evidence_report_id"] = "dggr-test"
        invalid_packet["metadata"]["used_grounding_fact_ids"] = [
            ledger["observed_facts"][0]["fact_id"]
        ]
        repaired_packet = copy.deepcopy(invalid_packet)
        repaired_packet["minimax_h3_prompt"] = "verified repaired timeline"
        calls: list[tuple] = []

        def detailed(_config, prompt, media_context, _max_tokens, options, **_kwargs):
            calls.append((prompt, media_context, copy.copy(options)))
            return self.nodes.BackendRunResult(decoded_text=f"response-{len(calls)}")

        def parse_packet(raw, *_args, **_kwargs):
            packet = invalid_packet if raw == "response-1" else repaired_packet
            return copy.deepcopy(packet), True, ""

        def h3_reasons(prompt, *_args, **_kwargs):
            return [] if "verified repaired" in prompt else ["minimax_h3_empty_visual_timeline"]

        budget = self.nodes.BackendCallBudget(max_calls=4)
        options = self.nodes.BackendRunOptions(
            sampling_profile="checkpoint_defaults",
            seed=31,
            enable_telemetry=True,
            asset_registry=self.nodes.build_asset_registry({}, 0),
            call_budget=budget,
            stage="compiler",
        )
        backend_results: list = []
        with patch.object(
            self.nodes, "_run_backend_detailed", side_effect=detailed
        ), patch.object(
            self.nodes, "repair_or_salvage_prompt_packet", side_effect=parse_packet
        ), patch.object(
            self.nodes, "_minimax_h3_prompt_validation_reasons", side_effect=h3_reasons
        ):
            result = self.nodes._run_generation_packet_legacy(
                runtime,
                compiler_context,
                target,
                runtime_required=True,
                backend_options=options,
                backend_results=backend_results,
                manage_unload=False,
                max_refinement_attempts_override=2,
                refinement_evidence_context=json.dumps(ledger, sort_keys=True),
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(budget.attempted_calls, 2)
        self.assertEqual([item.stage for item in backend_results], ["compiler", "target_repair_1"])
        self.assertTrue(
            all(call[1] is None or call[1].images is None for call in calls)
        )
        self.assertNotIn("VERIFIED VISUAL EVIDENCE LEDGER", calls[0][0])
        self.assertIn("VERIFIED VISUAL EVIDENCE LEDGER", calls[1][0])
        self.assertIn(ledger["observed_facts"][0]["fact_id"], calls[1][0])
        self.assertEqual(result[0]["minimax_h3_prompt"], "verified repaired timeline")
        self.assertEqual(result[0]["metadata"]["grounding_evidence_report_id"], "dggr-test")
        self.assertEqual(
            result[0]["metadata"]["used_grounding_fact_ids"],
            [ledger["observed_facts"][0]["fact_id"]],
        )

    def test_legacy_runtime_exception_unloads_once(self) -> None:
        runtime = self.runtime()
        with patch.object(
            self.nodes,
            "_run_backend_for_packet",
            side_effect=RuntimeError("backend exploded"),
        ), patch.object(self.nodes, "_release_transformers_runtime") as release:
            with self.assertRaisesRegex(RuntimeError, "runtime_required is true"):
                self.nodes._run_generation_packet_legacy(
                    runtime,
                    self.image_context(),
                    self.target(),
                    runtime_required=True,
                )
        release.assert_called_once()

    def test_strict_compiler_failure_does_not_relabel_valid_evidence_as_bad_schema(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        registry = self.nodes._grounding_asset_registry(context)
        ledger = self.grounded_ledger(registry)

        with patch.object(
            self.nodes,
            "_run_backend_detailed",
            return_value=self.backend_result(ledger),
        ), patch.object(
            self.nodes,
            "_run_generation_packet_legacy",
            side_effect=RuntimeError("compiler exploded"),
        ), patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={"mode": "strict", "retry_on_uncertain": False},
            )

        packet, metadata = result[0], result[-1]
        self.assertFalse(metadata["ready_for_generation"])
        self.assertEqual(packet["ltx_prompt"], "")
        self.assertEqual(
            metadata["blocked_reasons"][:2],
            ["visual_grounding_unverified", "visual_grounding_uncertain"],
        )
        self.assertNotIn(
            "visual_grounding_schema_invalid",
            metadata["grounding_guard"]["blocked_reasons"],
        )
        self.assertIsNotNone(metadata["grounding_guard"]["validated_ledger"])
        unload.assert_called_once_with(runtime)

    def test_truncated_strict_evidence_retries_concisely_and_compiles(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        evidence_calls: list[tuple] = []
        retry_ledgers: list[dict] = []

        def evidence(_config, prompt, media_context, max_tokens, options, **_kwargs):
            evidence_calls.append((prompt, media_context, options, max_tokens))
            if len(evidence_calls) == 1:
                return self.truncated_evidence_result(options.asset_registry)
            ledger = self.grounded_ledger(options.asset_registry)
            retry_ledgers.append(ledger)
            return self.backend_result(ledger)

        def compiler_after_retry(*args, **_kwargs):
            compiler_context = args[1]
            verified_ledger = compiler_context.media_metadata["verified_grounding_ledger"]
            packet = self.packet("compiled after concise evidence retry")
            packet["metadata"]["grounding_evidence_report_id"] = (
                compiler_context.media_metadata["grounding_evidence_report_id"]
            )
            packet["metadata"]["used_grounding_fact_ids"] = [
                verified_ledger["observed_facts"][0]["fact_id"]
            ]
            return self.legacy_result(packet)

        with patch.object(
            self.nodes, "_run_backend_detailed", side_effect=evidence
        ), patch.object(
            self.nodes,
            "_run_generation_packet_legacy",
            side_effect=compiler_after_retry,
        ) as compiler, patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={"mode": "strict", "retry_on_uncertain": True},
                max_new_tokens=2048,
            )

        packet, metadata = result[0], result[-1]
        self.assertEqual(len(evidence_calls), 2)
        self.assertEqual([call[3] for call in evidence_calls], [768, 1024])
        retry_prompt = evidence_calls[1][0]
        self.assertIn("Use no more than 4 observed facts total", retry_prompt)
        self.assertIn("keep each claim at 24 words or fewer", retry_prompt)
        self.assertIn("Completing and closing the strict JSON object takes priority", retry_prompt)
        self.assertEqual(int(evidence_calls[1][1].images.shape[0]), 1)
        self.assertEqual(len(retry_ledgers), 1)
        self.assertLessEqual(len(retry_ledgers[0]["observed_facts"]), 4)
        compiler.assert_called_once()
        self.assertEqual(packet["ltx_prompt"], "compiled after concise evidence retry")
        self.assertTrue(metadata["ready_for_generation"])
        self.assertEqual(metadata["grounding_guard"]["attempt_count"], 2)
        self.assertEqual(metadata["grounding_guard"]["decision"], "pass")
        self.assertTrue(
            any(
                reason.startswith("visual_grounding_schema_invalid:")
                for reason in metadata["grounding_guard"]["retry_reasons"]
            )
        )
        unload.assert_called_once_with(runtime)

    def test_repeated_truncated_strict_evidence_fails_closed_without_partial_facts(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        evidence_calls: list[tuple] = []

        def evidence(_config, prompt, media_context, max_tokens, options, **_kwargs):
            evidence_calls.append((prompt, media_context, options, max_tokens))
            return self.truncated_evidence_result(options.asset_registry)

        with patch.object(
            self.nodes, "_run_backend_detailed", side_effect=evidence
        ) as backend, patch.object(
            self.nodes, "_run_generation_packet_legacy"
        ) as compiler, patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={"mode": "strict", "retry_on_uncertain": True},
                max_new_tokens=2048,
            )

        packet, metadata = result[0], result[-1]
        report = metadata["grounding_guard"]
        self.assertEqual(backend.call_count, 2)
        self.assertEqual([call[3] for call in evidence_calls], [768, 1024])
        compiler.assert_not_called()
        self.assertEqual(packet["ltx_prompt"], "")
        self.assertFalse(metadata["ready_for_generation"])
        self.assertEqual(
            metadata["blocked_reasons"][:2],
            ["visual_grounding_unverified", "visual_grounding_schema_invalid"],
        )
        self.assertEqual(report["attempt_count"], 2)
        self.assertEqual(report["model_call_count"], 2)
        self.assertEqual(len(report["retry_reasons"]), 2)
        self.assertTrue(
            all(
                reason.startswith("visual_grounding_schema_invalid:")
                for reason in report["retry_reasons"]
            )
        )
        self.assertEqual(report["validated_ledger"].get("observed_facts", []), [])
        self.assertEqual(report["validation"]["covered_asset_ids"], [])
        self.assertFalse(report["validation"]["schema_valid"])
        unload.assert_called_once_with(runtime)

    def test_uncertain_evidence_retries_with_reduced_original_frames(self) -> None:
        runtime = self.runtime()
        context = self.video_context(5)
        evidence_calls: list[tuple] = []

        def evidence(_config, prompt, media_context, max_tokens, options, **_kwargs):
            evidence_calls.append((prompt, media_context, options, max_tokens))
            if len(evidence_calls) == 1:
                return self.backend_result(self.uncertain_ledger())
            return self.backend_result(self.grounded_ledger(options.asset_registry))

        def retry_compiler(*args, **_kwargs):
            compiler_context = args[1]
            packet = self.packet("retry compiler prompt")
            packet["metadata"]["grounding_evidence_report_id"] = compiler_context.media_metadata[
                "grounding_evidence_report_id"
            ]
            packet["metadata"]["used_grounding_fact_ids"] = [
                compiler_context.media_metadata["verified_grounding_ledger"]["observed_facts"][0][
                    "fact_id"
                ]
            ]
            return self.legacy_result(packet)

        with patch.object(
            self.nodes, "_run_backend_detailed", side_effect=evidence
        ), patch.object(
            self.nodes,
            "_run_generation_packet_legacy",
            side_effect=retry_compiler,
        ) as compiler, patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={
                    "mode": "strict",
                    "retry_on_uncertain": True,
                    "seed": 23,
                    "evidence_token_budget": "1280",
                },
                max_new_tokens=512,
            )

        self.assertEqual(len(evidence_calls), 2)
        retry_prompt, retry_media, retry_options, retry_max_tokens = evidence_calls[1]
        self.assertIn("one permitted focused retry", retry_prompt)
        self.assertEqual(evidence_calls[0][3], 1280)
        self.assertEqual(retry_max_tokens, 1280)
        self.assertEqual(int(retry_media.images.shape[0]), 3)
        self.assertEqual(retry_options.asset_registry["expected_image_count"], 3)
        self.assertEqual(
            retry_media.metadata["grounding_retry_frame_reduction"][
                "selected_video_sample_ordinals"
            ],
            [1, 3, 5],
        )
        compiler.assert_called_once()
        self.assertEqual(result[-1]["grounding_guard"]["attempt_count"], 2)
        self.assertEqual(result[-1]["grounding_guard"]["decision"], "pass")
        self.assertEqual(
            result[-1]["grounding_guard"]["config"]["evidence_token_budget"],
            "1280",
        )
        self.assertEqual(
            compiler.call_args.kwargs["max_refinement_attempts_override"], 1
        )
        unload.assert_called_once_with(runtime)

    def test_retry_exhaustion_blocks_without_compilation(self) -> None:
        runtime = self.runtime()
        context = self.video_context(5)
        with patch.object(
            self.nodes,
            "_run_backend_detailed",
            side_effect=[
                self.backend_result(self.uncertain_ledger()),
                self.backend_result(self.uncertain_ledger()),
            ],
        ) as evidence, patch.object(
            self.nodes, "_run_generation_packet_legacy"
        ) as compiler, patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                self.target(),
                grounding_guard_config={"mode": "strict", "retry_on_uncertain": True},
            )

        packet, metadata = result[0], result[-1]
        self.assertEqual(evidence.call_count, 2)
        compiler.assert_not_called()
        self.assertEqual(packet["ltx_prompt"], "")
        self.assertFalse(metadata["ready_for_generation"])
        self.assertEqual(metadata["grounding_guard"]["attempt_count"], 2)
        self.assertEqual(metadata["grounding_guard"]["model_call_count"], 2)
        self.assertEqual(
            metadata["blocked_reasons"][:2],
            ["visual_grounding_unverified", "visual_grounding_uncertain"],
        )
        unload.assert_called_once_with(runtime)

    def test_backend_exceptions_exhaust_retry_as_uncertain_not_transport(self) -> None:
        runtime = self.runtime()
        with patch.object(
            self.nodes,
            "_run_backend_detailed",
            side_effect=[RuntimeError("model forward failed"), RuntimeError("model forward failed")],
        ) as evidence, patch.object(
            self.nodes, "_run_generation_packet_legacy"
        ) as compiler, patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                self.image_context(),
                self.target(),
                grounding_guard_config={"mode": "strict", "retry_on_uncertain": True},
            )

        packet, metadata = result[0], result[-1]
        self.assertEqual(evidence.call_count, 2)
        compiler.assert_not_called()
        self.assertEqual(packet["ltx_prompt"], "")
        self.assertEqual(
            metadata["blocked_reasons"][:2],
            ["visual_grounding_unverified", "visual_grounding_uncertain"],
        )
        self.assertNotIn("visual_transport_error", metadata["blocked_reasons"])
        unload.assert_called_once_with(runtime)

    def test_strict_telemetry_exception_uses_the_telemetry_block_reason(self) -> None:
        runtime = self.runtime()
        with patch.object(
            self.nodes,
            "_run_backend_detailed",
            side_effect=RuntimeError("visual_grounding_telemetry_error:summary failed"),
        ) as evidence, patch.object(
            self.nodes, "_run_generation_packet_legacy"
        ) as compiler, patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                self.image_context(),
                self.target(),
                grounding_guard_config={"mode": "strict", "retry_on_uncertain": True},
            )

        packet, metadata = result[0], result[-1]
        evidence.assert_called_once()
        compiler.assert_not_called()
        self.assertEqual(packet["ltx_prompt"], "")
        self.assertEqual(
            metadata["blocked_reasons"][:2],
            ["visual_grounding_unverified", "visual_grounding_telemetry_error"],
        )
        unload.assert_called_once_with(runtime)

    def test_strict_target_repair_telemetry_error_preserves_failure_reason(self) -> None:
        runtime = self.runtime()
        context = self.image_context()
        registry = self.nodes._grounding_asset_registry(context)
        ledger = self.grounded_ledger(registry)
        target = self.nodes.TargetProfileConfig(
            target_profile="minimax_h3",
            minimax_h3_mode="t2va",
        )
        compiler_packet = self.packet("")
        compiler_packet["ltx_prompt"] = ""
        compiler_packet["minimax_h3_prompt"] = (
            "[Shot 1] invalid timeline. [Shot 2] missing a required timestamp."
        )
        compiler_packet["metadata"]["target_profile"] = "minimax_h3"
        evidence_report_id = self.nodes._grounding_report_id(ledger, registry)
        stages: list[str] = []

        def backend_side_effect(_config, _prompt, _media, _tokens, options, **_kwargs):
            stages.append(options.stage)
            if options.stage == "evidence_attempt_1":
                return self.backend_result(ledger)
            if options.stage == "compiler":
                result = self.backend_result(compiler_packet)
                result.decoded_text = (
                    "integrated_multimodal_description: [Shot 1] invalid timeline. "
                    "[Shot 2] missing a required timestamp.\n\n"
                    "overall_soundscape: N/A\n\n"
                    "non_diegetic_music: N/A\n\n"
                    f"GROUNDING_EVIDENCE_REPORT_ID: {evidence_report_id}\n"
                    "USED_GROUNDING_FACT_IDS: fact-1"
                )
                return result
            if options.stage == "target_repair_1":
                raise RuntimeError("visual_grounding_telemetry_error:repair summary failed")
            raise AssertionError(f"unexpected backend stage: {options.stage}")

        with patch.object(
            self.nodes,
            "_run_backend_detailed",
            side_effect=backend_side_effect,
        ), patch.object(self.nodes, "_maybe_unload") as unload:
            result = self.nodes._run_generation_packet(
                runtime,
                context,
                target,
                grounding_guard_config={"mode": "strict", "retry_on_uncertain": False},
            )

        self.assertEqual(stages, ["evidence_attempt_1", "compiler", "target_repair_1"])
        self.assertEqual(result[0]["minimax_h3_prompt"], "")
        self.assertFalse(result[-1]["ready_for_generation"])
        self.assertEqual(
            result[-1]["blocked_reasons"][:2],
            ["visual_grounding_unverified", "visual_grounding_telemetry_error"],
        )
        self.assertEqual(result[-1]["grounding_guard"]["model_call_count"], 3)
        unload.assert_called_once_with(runtime)

    def test_call_budget_counts_failed_backend_attempts(self) -> None:
        budget = self.nodes.BackendCallBudget(max_calls=4)
        options = self.nodes.BackendRunOptions(
            enable_telemetry=False,
            call_budget=budget,
        )
        with patch.object(
            self.nodes, "_run_backend_detailed", side_effect=RuntimeError("boom")
        ) as backend:
            for _ in range(4):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    self.nodes._run_backend_for_packet(
                        self.runtime(),
                        "prompt",
                        None,
                        128,
                        None,
                        options,
                        [],
                    )
            with self.assertRaisesRegex(RuntimeError, r"4-call ceiling"):
                self.nodes._run_backend_for_packet(
                    self.runtime(),
                    "prompt",
                    None,
                    128,
                    None,
                    options,
                    [],
                )

        self.assertEqual(budget.attempted_calls, 4)
        self.assertEqual(backend.call_count, 4)

    def test_compiler_retry_uses_the_next_adjacent_effective_seed(self) -> None:
        budget = self.nodes.BackendCallBudget(max_calls=4)
        budget.claim()  # Evidence attempt 1 already consumed the first call.
        effective: list[tuple[str, int]] = []

        def backend(_config, _prompt, _media, _tokens, options, **_kwargs):
            effective.append((options.stage, options.seed))
            return self.nodes.BackendRunResult(decoded_text="{}")

        results: list = []
        with patch.object(self.nodes, "_run_backend_detailed", side_effect=backend):
            for stage in ("compiler", "compiler_retry"):
                options = self.nodes.BackendRunOptions(
                    seed=1009,
                    stage=stage,
                    enable_telemetry=False,
                    call_budget=budget,
                )
                self.nodes._run_backend_for_packet(
                    self.runtime(),
                    "prompt",
                    None,
                    128,
                    None,
                    options,
                    results,
                )

        self.assertEqual(effective, [("compiler", 1010), ("compiler_retry", 1011)])
        self.assertEqual(budget.attempted_calls, 3)

    def test_guarded_transformers_call_restores_cpu_rng_state(self) -> None:
        nodes = self.nodes

        class FakeProcessor:
            tokenizer = None
            image_token_id = 111
            video_token_id = 222

            def apply_chat_template(self, *_args, **_kwargs):
                return {
                    "input_ids": torch.tensor([[7, 8]], dtype=torch.long),
                    "attention_mask": torch.ones((1, 2), dtype=torch.long),
                }

            def batch_decode(self, _sequences, **_kwargs):
                return ["deterministic output"]

        class FakeModel:
            device = torch.device("cpu")
            config = SimpleNamespace(
                image_token_id=111,
                video_token_id=None,
                canvas_length=256,
            )

            def __init__(self):
                self.random_draw = None

            def generate(self, **kwargs):
                self.random_draw = int(torch.randint(0, 10000, (1,)).item())
                generated = torch.tensor([[self.random_draw % 100]], dtype=torch.long)
                return SimpleNamespace(
                    sequences=torch.cat([kwargs["input_ids"], generated], dim=-1),
                    tokens_per_forward=torch.tensor([1]),
                )

        model = FakeModel()
        options = nodes.BackendRunOptions(
            sampling_profile="checkpoint_defaults",
            thinking_mode="off",
            seed=987654321,
            enable_telemetry=False,
            asset_registry=nodes.build_asset_registry({}, 0),
        )
        torch.manual_seed(12345)
        state_before = torch.random.get_rng_state().clone()
        with patch.object(
            nodes, "_load_transformers_model", return_value=(FakeProcessor(), model)
        ), patch.object(
            nodes,
            "_run_blocking_with_progress",
            side_effect=lambda _label, function, **_kwargs: function(),
        ), patch.object(torch.cuda, "is_available", return_value=False), patch.object(
            torch.cuda, "device_count", return_value=0
        ):
            result = nodes._run_transformers_detailed(
                self.runtime(),
                "prompt",
                None,
                128,
                options,
            )

        self.assertEqual(result.decoded_text, "deterministic output")
        self.assertIsNotNone(model.random_draw)
        self.assertTrue(torch.equal(state_before, torch.random.get_rng_state()))

    def test_passive_telemetry_does_not_change_tokens_for_the_same_seed(self) -> None:
        nodes = self.nodes

        class FakeProcessor:
            tokenizer = None
            image_token_id = 111
            video_token_id = 222

            def apply_chat_template(self, *_args, **_kwargs):
                return {
                    "input_ids": torch.tensor([[7, 8]], dtype=torch.long),
                    "attention_mask": torch.ones((1, 2), dtype=torch.long),
                }

            def batch_decode(self, sequences, **_kwargs):
                return [" ".join(str(int(token)) for token in sequences[0].tolist())]

        class FakeModel:
            device = torch.device("cpu")
            config = SimpleNamespace(
                image_token_id=111,
                video_token_id=None,
                canvas_length=256,
            )

            def __init__(self):
                self.generated_token = None

            def generate(self, **kwargs):
                scores = torch.rand((1, 256, 17), dtype=torch.float32)
                processors = kwargs.get("logits_processor")
                if processors:
                    observed = processors[0](kwargs["input_ids"], scores, cur_step=1)
                    if observed is not scores:
                        raise AssertionError("telemetry replaced the scores tensor")
                token = scores[:, 0, :].argmax(dim=-1, keepdim=True)
                self.generated_token = int(token[0, 0])
                streamer = kwargs.get("streamer")
                if streamer is not None:
                    streamer.put(kwargs["input_ids"])
                    streamer.put_draft(token)
                    streamer.put(token)
                    streamer.end()
                return SimpleNamespace(
                    sequences=torch.cat([kwargs["input_ids"], token], dim=-1),
                    tokens_per_forward=torch.tensor([1]),
                )

        def run(enable_telemetry: bool):
            model = FakeModel()
            options = nodes.BackendRunOptions(
                sampling_profile="checkpoint_defaults",
                thinking_mode="off",
                seed=424242,
                enable_telemetry=enable_telemetry,
                fail_closed_telemetry=enable_telemetry,
                asset_registry=nodes.build_asset_registry({}, 0),
            )
            with patch.object(
                nodes, "_load_transformers_model", return_value=(FakeProcessor(), model)
            ), patch.object(
                nodes,
                "_run_blocking_with_progress",
                side_effect=lambda _label, function, **_kwargs: function(),
            ), patch.object(torch.cuda, "is_available", return_value=False), patch.object(
                torch.cuda, "device_count", return_value=0
            ):
                result = nodes._run_transformers_detailed(
                    self.runtime(), "prompt", None, 128, options
                )
            return model.generated_token, result

        torch.manual_seed(9988)
        state_before = torch.random.get_rng_state().clone()
        token_without, result_without = run(False)
        token_with, result_with = run(True)
        self.assertEqual(token_without, token_with)
        self.assertEqual(result_without.decoded_text, result_with.decoded_text)
        self.assertFalse(result_without.telemetry)
        self.assertEqual(result_with.telemetry["forward_count"], 1)
        self.assertTrue(torch.equal(state_before, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
