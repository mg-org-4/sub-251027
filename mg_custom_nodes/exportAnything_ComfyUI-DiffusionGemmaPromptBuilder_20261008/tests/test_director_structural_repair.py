from __future__ import annotations

import importlib.util
import json
import sys
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]


def load_nodes_module():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    package_name = f"diffusiongemma_structural_repair_{uuid.uuid4().hex}"
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


class DirectorStructuralRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()

    def context(self):
        return self.nodes.GemmaContext(
            user_prompt="Create a concise scene.",
            source="none",
            media_metadata={"source": "none"},
        )

    def h3_target(self):
        return self.nodes.TargetProfileConfig(
            target_profile="minimax_h3",
            minimax_h3_mode="t2va",
        )

    def test_exact_json_is_valid_without_transport_repair(self) -> None:
        raw = json.dumps(
            {
                "ltx_prompt": "",
                "ideogram_prompt": "",
                "minimax_h3_prompt": "Keep this exact semantic value,} intact.",
                "negative_prompt": "",
                "scene_segments": [],
                "metadata": {},
            }
        )
        packet, parse_valid, warning = self.nodes.repair_or_salvage_prompt_packet(
            raw,
            self.context(),
            self.h3_target(),
            deterministic_transport_repair=True,
        )
        self.assertTrue(parse_valid)
        self.assertEqual(warning, "")
        self.assertTrue(packet["metadata"]["model_output_json_parse_valid"])
        self.assertFalse(packet["metadata"]["deterministic_transport_repair"])
        self.assertIn("semantic value,} intact", packet["minimax_h3_prompt"])

    def test_outer_fence_and_trailing_comma_are_repaired_without_semantic_fill(self) -> None:
        raw = """```json
{
  "minimax_h3_prompt": "An intentionally opaque H3 prompt.",
  "metadata": {},
}
```"""
        packet, parse_valid, warning = self.nodes.repair_or_salvage_prompt_packet(
            raw,
            self.context(),
            self.h3_target(),
            deterministic_transport_repair=True,
        )
        self.assertTrue(parse_valid)
        self.assertEqual(warning, "")
        self.assertFalse(packet["metadata"]["model_output_json_parse_valid"])
        self.assertTrue(packet["metadata"]["deterministic_transport_repair"])
        repairs = packet["metadata"]["deterministic_transport_repairs"]
        self.assertIn("removed_outer_json_fence", repairs)
        self.assertIn("removed_trailing_commas:1", repairs)
        self.assertEqual(packet["ltx_prompt"], "")
        self.assertEqual(packet["ideogram_prompt"], "")
        self.assertEqual(packet["negative_prompt"], "")
        self.assertEqual(packet["scene_segments"], [])

    def test_one_missing_root_closer_and_trailing_comma_are_repaired(self) -> None:
        parsed, repairs, original_valid = (
            self.nodes._deterministically_repaired_prompt_packet_json(
                '{"minimax_h3_prompt":"unchanged","metadata":{},'
            )
        )
        self.assertIsInstance(parsed, dict)
        self.assertFalse(original_valid)
        self.assertIn("appended_missing_root_closer", repairs)
        self.assertIn("removed_trailing_commas:1", repairs)
        self.assertEqual(parsed["minimax_h3_prompt"], "unchanged")

    def test_live_t2va_invalid_dialogue_closer_escape_is_recovered(self) -> None:
        raw = r'''<|final_json|>
{
  "minimax_h3_prompt": "integrated_multimodal_description: [Shot 1] 2D hand-drawn cel animation, preserve drawn linework, cel shading, stylized proportions, and design continuity across every cut, no live action, photorealism, or 3D CGI. A young girl with wide, determined eyes clutches a tattered teddy bear to her chest, surrounded by a circle of shadowy, menacing enemies in a dim alley. The girl looks up with a defiant smirk and says, (S1) <d lang=\"English\">maybe you haven't met my friend yet...\</d>. The camera performs a slow zoom-in on her face to capture her growing resolve.\n\n[Shot 2] At 00:03.500 the camera cuts to the enemies, a group of figures with distorted features leaning back and pointing, erupt in mocking laughter. The camera is a static low-angle shot to emphasize their perceived dominance.\n\n[Shot 3] At 00:06.000 the smash cuts to a wide shot of the girl in the center of the frame, holding the teddy bear high. She screams with intense energy, (S2) <d lang=\"English\">TEDDY BEAR ACTIVATE</</d> as she hurls the bear into the night sky. The camera tracks the bear upward with a rapid tilt movement.\n\n[Shot 4] At 00:08.200 the camera cuts to the bear mid-air, which rapidly morphs with shifting mechanical plates and glowing blue circuitry, revealing a massive teddy bear-battle cyborg. The cyborg descends rapidly and lands with a heavy, metallic thud on its feet directly behind the girl, sending up a cloud of dust. The final frame holds on the girl standing confidently in front of the towering cyborg in a wide shot composition.\n\noverall_soundscape: Distant echoing laughter, the whistle of the bear being thrown, mechanical whirring during the transformation, and a heavy resonant metallic impact as the cyborg hits the ground.\n\nnon_diegetic_music: High-tension orchestral swell that builds during the throw and a sudden, powerful brass hit when the cyborg lands."
}
<|end_final_json|>'''
        brief = (
            "2d anime of a young girl holding a teddy bear surrounded by enemies. "
            "She states maybe you haven't met my friend yet. She screams TEDDY BEAR "
            "ACTIVATE as the bear transforms into a teddy bear-battle cyborg. "
            "No subtitles or words."
        )
        packet, parse_valid, warning = self.nodes.repair_or_salvage_prompt_packet(
            raw,
            self.nodes.GemmaContext(
                user_prompt=brief,
                source="none",
                media_metadata={"source": "none", "minimax_h3_mode": "t2va"},
            ),
            self.h3_target(),
        )
        self.assertTrue(parse_valid)
        self.assertEqual(warning, "")
        self.assertEqual(packet["ltx_prompt"], "")
        self.assertEqual(packet["ideogram_prompt"], "")
        self.assertEqual(packet["negative_prompt"], "")
        self.assertEqual(packet["scene_segments"], [])
        self.assertIn("maybe you haven't met my friend yet", packet["minimax_h3_prompt"])
        self.assertNotIn(r"\</d>", packet["minimax_h3_prompt"])
        self.assertNotIn("lang=", packet["minimax_h3_prompt"])
        self.assertNotIn("</</d>", packet["minimax_h3_prompt"])
        self.assertEqual(packet["minimax_h3_prompt"].count("<d>"), 2)
        self.assertEqual(packet["minimax_h3_prompt"].count("</d>"), 2)
        validation_reasons = self.nodes._minimax_h3_prompt_validation_reasons(
            packet["minimax_h3_prompt"],
            10.0,
            "2d anime girl; no subtitles or words",
        )
        self.assertNotIn("minimax_h3_dialogue_tags_invalid", validation_reasons)
        self.assertNotIn("minimax_h3_timeline_invalid", validation_reasons)
        metadata = packet["metadata"]
        self.assertFalse(metadata["model_output_json_parse_valid"])
        self.assertTrue(metadata["deterministic_transport_repair"])
        self.assertIn(
            "removed_invalid_h3_dialogue_closer_json_escape:1",
            metadata["deterministic_transport_repairs"],
        )
        self.assertEqual(
            self.nodes._minimax_h3_prompt_validation_reasons(
                packet["minimax_h3_prompt"],
                10.0,
                brief,
            ),
            [],
        )

    def test_h3_dialogue_closer_escape_recovery_rejects_non_single_key_packets(self) -> None:
        canonical = (
            "integrated_multimodal_description: [Shot 1] A locked camera holds as (S1) "
            r"<d>[English] hello\</d>. The final frame holds on the centered subject.\n\n"
            r"overall_soundscape: N/A\n\nnon_diegetic_music: N/A"
        )
        for raw in (
            '{"minimax_h3_prompt":"' + canonical + '","metadata":{}}',
            '{"minimax_h3_prompt":"' + canonical + '","minimax_h3_prompt":"duplicate"}',
            '{"minimax_h3_prompt":"' + canonical.replace("hello", 'hello "ambiguous"') + '"}',
        ):
            with self.subTest(raw=raw[-80:]):
                packet, parse_valid, warning = self.nodes.repair_or_salvage_prompt_packet(
                    raw,
                    self.context(),
                    self.h3_target(),
                )
                self.assertFalse(parse_valid)
                self.assertTrue(warning)
                self.assertFalse(
                    packet.get("metadata", {}).get("deterministic_transport_repair", False)
                )

    def test_duplicate_keys_are_never_deterministically_repaired(self) -> None:
        raw = (
            '{"minimax_h3_prompt":"first","minimax_h3_prompt":"second",'
            '"metadata":{}}'
        )
        parsed, repairs, original_valid = (
            self.nodes._deterministically_repaired_prompt_packet_json(raw)
        )
        self.assertIsNone(parsed)
        self.assertEqual(repairs, [])
        self.assertFalse(original_valid)
        packet, parse_valid, warning = self.nodes.repair_or_salvage_prompt_packet(
            raw,
            self.context(),
            self.h3_target(),
            deterministic_transport_repair=True,
        )
        self.assertFalse(parse_valid)
        self.assertTrue(warning)
        self.assertNotEqual(packet.get("minimax_h3_prompt"), "second")

    def test_grounding_evidence_parser_remains_strict(self) -> None:
        malformed = (
            '{"schema":"dg-grounding-ledger/1","analysis_status":"uncertain",'
            '"observed_facts":[],"inferred_facts":[],"creative_additions":[],'
            '"uncertainties":[],"grounding_failure_reasons":[],}'
        )
        with self.assertRaises(self.nodes.GroundingLedgerError):
            self.nodes._strict_evidence_json_payload(malformed)

    def test_grounding_evidence_repairs_only_one_extra_root_open_brace(self) -> None:
        ledger = {
            "schema": "dg-grounding-ledger/1",
            "analysis_status": "grounded",
            "observed_facts": [],
            "inferred_facts": [],
            "creative_additions": [],
            "uncertainties": [],
            "grounding_failure_reasons": [],
        }
        repairs: list[str] = []
        payload = self.nodes._strict_evidence_json_payload(
            "{" + json.dumps(ledger),
            repair_log=repairs,
        )
        self.assertEqual(json.loads(payload), ledger)
        self.assertEqual(repairs, ["removed_extra_outer_open_brace"])

        duplicate_key = (
            '{{"schema":"dg-grounding-ledger/1","schema":"dg-grounding-ledger/1",'
            '"analysis_status":"grounded","observed_facts":[],"inferred_facts":[],'
            '"creative_additions":[],"uncertainties":[],"grounding_failure_reasons":[]}'
        )
        with self.assertRaises(self.nodes.GroundingLedgerError):
            self.nodes._strict_evidence_json_payload(duplicate_key)

    def test_splitter_preserves_original_parse_failure_and_fails_closed(self) -> None:
        packet = {
            "ltx_prompt": "This prompt must not pass after an origin parse failure.",
            "ideogram_prompt": "",
            "minimax_h3_prompt": "",
            "negative_prompt": "",
            "scene_segments": [],
            "metadata": {
                "json_parse_valid": False,
                "ready_for_generation": True,
                "blocked_reasons": [],
            },
        }
        result = self.nodes.DiffusionGemmaJSONSplitter().split(
            json.dumps(packet),
            gemma_context=self.context(),
            target_profile_config=self.nodes.TargetProfileConfig(target_profile="ltx"),
        )
        metadata = json.loads(result[4])
        self.assertFalse(metadata["json_parse_valid"])
        self.assertFalse(metadata["model_output_json_parse_valid"])
        self.assertTrue(metadata["splitter_input_json_valid"])
        self.assertFalse(result[12])
        self.assertEqual(result[0], "")
        self.assertIn("json_parse_invalid", metadata["blocked_reasons"])

    def test_native_h3_compiler_output_is_host_packetized_without_json_salvage(self) -> None:
        context = self.context()
        context.media_metadata.update(
            {
                "grounding_evidence_report_id": "dggr-test",
                "verified_grounding_ledger": {
                    "observed_facts": [
                        {
                            "fact_id": "fact-1",
                            "confidence": "high",
                            "evidence": [{"asset_id": "image:1"}],
                        }
                    ]
                },
            }
        )
        raw = (
            "integrated_multimodal_description: [Shot 1] A locked camera holds on the subject. "
            "The final frame holds on the subject centered in the unchanged composition.\n\n"
            "overall_soundscape: N/A\n\n"
            "non_diegetic_music: N/A\n\n"
            "GROUNDING_EVIDENCE_REPORT_ID: dggr-test\n"
            "USED_GROUNDING_FACT_IDS: fact-1"
        )
        packet, parse_valid, warning = self.nodes.repair_or_salvage_prompt_packet(
            raw,
            context,
            self.h3_target(),
            deterministic_transport_repair=True,
            native_h3_output=True,
        )
        self.assertTrue(parse_valid)
        self.assertEqual(warning, "")
        self.assertTrue(packet["minimax_h3_prompt"].startswith(
            "integrated_multimodal_description: [Shot 1]"
        ))
        metadata = packet["metadata"]
        self.assertEqual(
            metadata["compiler_output_contract"],
            "native_minimax_h3_prompt_with_provenance/1",
        )
        self.assertTrue(metadata["compiler_output_contract_parse_valid"])
        self.assertFalse(metadata["model_output_json_parse_valid"])
        self.assertEqual(metadata["compiler_packet_structure_source"], "host")
        self.assertEqual(
            metadata["compiler_provenance_source"],
            "model_declared_native_footer",
        )
        self.assertEqual(metadata["used_grounding_fact_ids"], ["fact-1"])
        self.assertIsInstance(json.loads(json.dumps(packet)), dict)

    def test_native_h3_compiler_rejects_old_json_and_conversational_wrappers(self) -> None:
        native = (
            "integrated_multimodal_description: [Shot 1] A static hold.\n\n"
            "overall_soundscape: N/A\n\n"
            "non_diegetic_music: N/A\n\n"
            "GROUNDING_EVIDENCE_REPORT_ID: dggr-test\n"
            "USED_GROUNDING_FACT_IDS: fact-1"
        )
        for raw in (
            json.dumps({"minimax_h3_prompt": native, "metadata": {}}),
            "Here is the prompt:\n" + native,
            "```text\n" + native + "\n```",
        ):
            with self.subTest(raw=raw[:32]):
                _packet, parse_valid, warning = (
                    self.nodes.repair_or_salvage_prompt_packet(
                        raw,
                        self.context(),
                        self.h3_target(),
                        deterministic_transport_repair=True,
                        native_h3_output=True,
                    )
                )
                self.assertFalse(parse_valid)
                self.assertTrue(warning)

    def test_native_h3_compiler_rejects_residual_control_tokens(self) -> None:
        body = (
            "integrated_multimodal_description: [Shot 1] A static camera holds.\n\n"
            "overall_soundscape: N/A\n\n"
            "non_diegetic_music: N/A"
        )
        footer = (
            "\n\nGROUNDING_EVIDENCE_REPORT_ID: dggr-test\n"
            "USED_GROUNDING_FACT_IDS: fact-1"
        )
        for raw in (
            body + " <|garbage|>" + footer,
            body + " <think>hidden</think>" + footer,
            body + " <think>unterminated" + footer,
            body + footer + "\n<|garbage|>",
        ):
            with self.subTest(raw=raw[-48:]):
                prompt, report_id, fact_ids, repairs = (
                    self.nodes._strict_native_minimax_h3_prompt(raw, "t2va")
                )
                self.assertEqual(prompt, "")
                self.assertEqual(report_id, "")
                self.assertEqual(fact_ids, [])
                self.assertIsInstance(repairs, list)

    def test_native_h3_model_prompt_does_not_request_a_model_generated_packet(self) -> None:
        model_prompt = self.nodes._build_model_prompt(
            "Keep the character still.",
            self.nodes.DEFAULT_MASTER_PROMPT,
            "minimax_h3",
            {
                "source": "none",
                "minimax_h3_mode": "t2va",
                "grounding_evidence_report_id": "dggr-test",
                "verified_grounding_ledger": {
                    "observed_facts": [
                        {
                            "fact_id": "fact-1",
                            "confidence": "high",
                            "evidence": [{"asset_id": "image:1"}],
                        }
                    ]
                },
            },
            minimax_h3_mode="t2va",
            native_h3_output=True,
        )
        self.assertIn(
            "Return the native MiniMax H3 prompt followed by the two exact provenance lines",
            model_prompt,
        )
        self.assertIn(
            "Begin now with integrated_multimodal_description: [Shot 1]",
            model_prompt,
        )
        self.assertNotIn("Return the final answer as JSON matching this shape", model_prompt)
        self.assertNotIn("<|final_json|>", model_prompt)
        self.assertIn("GROUNDING_EVIDENCE_REPORT_ID: dggr-test", model_prompt)
        self.assertIn("USED_GROUNDING_FACT_IDS: fact-1", model_prompt)


if __name__ == "__main__":
    unittest.main()
