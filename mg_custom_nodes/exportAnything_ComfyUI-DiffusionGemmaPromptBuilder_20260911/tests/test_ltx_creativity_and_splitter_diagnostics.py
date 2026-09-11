from __future__ import annotations

import importlib.util
import json
import re
import sys
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]


def load_nodes_module():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    package_name = f"diffusiongemma_ltx_creativity_{uuid.uuid4().hex}"
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


class LtxCreativityAndSplitterDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()

    @staticmethod
    def _semantic_contract(text: str) -> str:
        """Remove labels/scalars so distinctness must come from actual policy."""

        normalized = text.casefold()
        normalized = re.sub(
            r"\b(?:faithful|editorial|cinematic|concept[- ]art|wild|"
            r"subtle|moderate|strong|maximal|intensity|strength)\b",
            " ",
            normalized,
        )
        normalized = re.sub(r"\b\d+(?:\.\d+)?\b", " ", normalized)
        return " ".join(normalized.split())

    @staticmethod
    def _prompt_metadata(generation_mode: str) -> dict:
        if generation_mode == "image_to_video":
            return {
                "source": "image",
                "reference_image_count": 1,
                "ltx_first_frame_attached": True,
                "ltx_last_frame_attached": False,
            }
        return {"source": "none"}

    def test_ltx_modes_have_distinct_semantic_contracts_in_t2v_and_i2v(self) -> None:
        modes = ("faithful", "editorial", "cinematic", "concept_art", "wild")

        for generation_mode in ("text_to_video", "image_to_video"):
            contracts: dict[str, str] = {}
            semantic_contracts: dict[str, str] = {}
            for mode in modes:
                contract = self.nodes._ltx_creativity_instruction(
                    mode,
                    0.6,
                    generation_mode,
                )
                contracts[mode] = contract
                semantic_contracts[mode] = self._semantic_contract(contract)

                model_prompt = self.nodes._build_model_prompt(
                    "A woman takes one measured step while the camera holds steady.",
                    self.nodes.DEFAULT_MASTER_PROMPT,
                    "ltx",
                    self._prompt_metadata(generation_mode),
                    creativity_mode=mode,
                    creative_strength=0.6,
                    ltx_generation_mode=generation_mode,
                )
                self.assertIn(
                    f"Creative direction: {contract}",
                    model_prompt,
                    (generation_mode, mode),
                )

            with self.subTest(generation_mode=generation_mode):
                self.assertEqual(len(set(contracts.values())), len(modes))
                self.assertEqual(
                    len(set(semantic_contracts.values())),
                    len(modes),
                    "Mode contracts must differ by policy, not only their mode label.",
                )

    def test_i2v_director_exposes_every_creativity_mode(self) -> None:
        mode_spec = self.nodes.DiffusionGemmaCoTGenerator.INPUT_TYPES()["required"][
            "creativity_mode"
        ]
        self.assertEqual(
            tuple(mode_spec[0]),
            ("faithful", "editorial", "cinematic", "concept_art", "wild"),
        )

        for mode in mode_spec[0]:
            with self.subTest(mode=mode):
                self.assertEqual(self.nodes._normalize_creativity_mode(mode), mode)
                model_prompt = self.nodes._build_model_prompt(
                    (
                        "From the supplied first frame, the cyclist accelerates through the "
                        "alley while the camera rapidly tracks beside her, swings into an orbit, "
                        "and cranes upward in one unbroken take."
                    ),
                    self.nodes.DEFAULT_MASTER_PROMPT,
                    "ltx",
                    self._prompt_metadata("image_to_video"),
                    creativity_mode=mode,
                    creative_strength=1.0,
                    ltx_generation_mode="image_to_video",
                )
                self.assertIn(f'"creativity_mode": "{mode}"', model_prompt)

    def test_creative_i2v_modes_authorize_dynamic_continuous_camera_design(self) -> None:
        """Conditioning fixes frame facts, not every possible future trajectory."""

        for mode in ("editorial", "cinematic", "concept_art", "wild"):
            contract = self.nodes._ltx_creativity_instruction(
                mode,
                1.0,
                "image_to_video",
            ).casefold()
            with self.subTest(mode=mode):
                self.assertIn("camera", contract)
                self.assertTrue(
                    any(
                        cue in contract
                        for cue in (
                            "brisk",
                            "energetic",
                            "dynamic",
                            "expressive",
                            "compound",
                            "multi-phase",
                        )
                    ),
                    f"{mode} never gives the Director a dynamic camera vocabulary: {contract}",
                )
                self.assertIn(
                    "continuous",
                    contract,
                    f"{mode} must frame compatible camera phases as one continuous take.",
                )
                self.assertIn(
                    "unspecified",
                    contract,
                    f"{mode} must distinguish an unspecified future path from an explicit hold.",
                )
                self.assertTrue(
                    bool(
                        re.search(
                            r"\b(?:may|can|allow(?:s|ed)?|authoriz(?:e|es|ed))\b",
                            contract,
                        )
                    ),
                    f"{mode} describes motion but never authorizes a compatible camera path.",
                )
                self.assertNotRegex(
                    contract,
                    r"do not (?:add|invent|replace)[^.]{0,100}camera (?:move|path|motion)",
                    f"{mode} still carries a blanket conditioned-camera prohibition.",
                )

                # More expressive camera work must not weaken I2V's hard continuity locks.
                self.assertRegex(contract, r"(?:do not|never|no new)[^.]{0,100}\bcut\b")
                self.assertRegex(
                    contract,
                    r"(?:do not|never|no new)[^.]{0,100}\b(?:actor|subject)\b",
                )
                self.assertRegex(
                    contract,
                    r"(?:do not|never|no new)[^.]{0,140}\blocation\b",
                )

    def test_first_frame_anchors_opening_pose_not_future_camera_trajectory(self) -> None:
        for mode in ("faithful", "editorial", "cinematic", "concept_art", "wild"):
            contract = self.nodes._ltx_creativity_instruction(
                mode,
                1.0,
                "image_to_video",
            ).casefold()
            with self.subTest(mode=mode):
                self.assertTrue(
                    "opening" in contract and any(word in contract for word in ("pose", "framing")),
                    "The I2V contract must say what the still actually anchors.",
                )
                self.assertTrue(
                    any(word in contract for word in ("future", "subsequent"))
                    and any(word in contract for word in ("trajectory", "camera path"))
                    and any(word in contract for word in ("does not fix", "doesn't fix", "not fix")),
                    "A first frame must not be treated as evidence for a static future trajectory.",
                )

    def test_explicit_static_i2v_direction_wins_in_every_mode(self) -> None:
        static_brief = (
            "Preserve the supplied first-frame composition. The camera remains completely "
            "static and locked off while the woman turns toward the window."
        )

        for mode in ("faithful", "editorial", "cinematic", "concept_art", "wild"):
            contract = self.nodes._ltx_creativity_instruction(
                mode,
                1.0,
                "image_to_video",
            ).casefold()
            packet = self.nodes._template_packet(
                static_brief,
                "ltx",
                self._prompt_metadata("image_to_video"),
                10.0,
                target_duration_seconds=10.0,
                creativity_mode=mode,
                creative_strength=1.0,
            )
            fallback_prompt = packet["ltx_prompt"].casefold()

            with self.subTest(mode=mode):
                self.assertIn("camera remains completely static and locked off", fallback_prompt)
                static_contract_sentences = [
                    sentence.strip()
                    for sentence in re.split(r"(?<=[.!?])\s+", contract)
                    if sentence.strip()
                ]
                self.assertTrue(
                    any(
                        "explicit" in sentence
                        and any(state in sentence for state in ("static", "locked"))
                        and any(
                            priority in sentence
                            for priority in ("preserve", "honor", "wins", "remain", "keep")
                        )
                        for sentence in static_contract_sentences
                    ),
                    "Creative modes must not override an explicit static/locked camera request.",
                )

    def test_faithful_requested_camera_energy_is_not_downgraded(self) -> None:
        dynamic_brief = (
            "From the supplied first frame, the runner bursts forward while the camera rapidly "
            "tracks beside her, whips through a half-orbit, then cranes upward without a cut."
        )
        contract = self.nodes._ltx_creativity_instruction(
            "faithful",
            1.0,
            "image_to_video",
        ).casefold()
        packet = self.nodes._template_packet(
            dynamic_brief,
            "ltx",
            self._prompt_metadata("image_to_video"),
            10.0,
            target_duration_seconds=10.0,
            creativity_mode="faithful",
            creative_strength=1.0,
        )
        fallback_prompt = packet["ltx_prompt"].casefold()

        self.assertIn("rapidly tracks", fallback_prompt)
        self.assertIn("whips through a half-orbit", fallback_prompt)
        self.assertNotRegex(
            fallback_prompt,
            r"\b(?:slow|slowly|static|locked[- ]off)\b",
            "Template fallback must not silently flatten an explicitly dynamic brief.",
        )
        for required_semantic in ("speed", "direction", "phase"):
            self.assertIn(required_semantic, contract)
        self.assertTrue(
            any(
                phrase in contract
                for phrase in (
                    "do not default to",
                    "do not downgrade",
                    "do not reduce",
                    "never replace",
                    "must preserve requested camera",
                    "preserve requested camera speed",
                    "honor requested camera speed",
                )
            ),
            "Faithful I2V needs an explicit anti-flattening rule; merely copying the raw brief "
            "has repeatedly produced slow/static interpretations.",
        )
        self.assertIn("slow", contract)
        self.assertIn("static", contract)

    def test_model_facing_i2v_contract_has_no_numeric_camera_cap_or_forced_slow_move(self) -> None:
        model_prompt = self.nodes._build_model_prompt(
            "The subject starts from the supplied still, then runs through the alley.",
            self.nodes.DEFAULT_MASTER_PROMPT,
            "ltx",
            self._prompt_metadata("image_to_video"),
            creativity_mode="cinematic",
            creative_strength=1.0,
            ltx_generation_mode="image_to_video",
        ).casefold()

        self.assertNotIn('"recommended_camera_motion_phases"', model_prompt)
        self.assertNotRegex(
            model_prompt,
            r"(?:at most|maximum|max(?:imum)? of|limit(?:ed)? to)\s+\d+\s+camera",
        )
        for forced_slow_phrase in (
            "prefer slow",
            "use a slow camera",
            "keep the camera slow",
            "camera should move slowly",
            "choose slow camera",
        ):
            self.assertNotIn(forced_slow_phrase, model_prompt)

    def test_ltx_strengths_use_qualitative_bands_not_only_numbers(self) -> None:
        expected_bands = {
            0.1: "subtle",
            0.6: "moderate",
            1.0: "strong",
            1.5: "maximal",
        }

        for generation_mode in ("text_to_video", "image_to_video"):
            contracts: list[str] = []
            for strength, expected_band in expected_bands.items():
                contract = self.nodes._ltx_creativity_instruction(
                    "cinematic",
                    strength,
                    generation_mode,
                )
                contracts.append(contract)
                with self.subTest(
                    generation_mode=generation_mode,
                    strength=strength,
                ):
                    self.assertIn(expected_band, contract.casefold())

            without_numbers = {
                re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", contract)
                for contract in contracts
            }
            with self.subTest(generation_mode=generation_mode):
                self.assertEqual(
                    len(without_numbers),
                    len(expected_bands),
                    "Strength levels must remain distinct after numeric scalars are removed.",
                )

    def test_conditioned_ltx_modes_share_anchor_precedence_and_forbid_additions(self) -> None:
        precedence = (
            "Frame anchors, explicit style guidance, user facts, exact speech, and "
            "the continuity contract take precedence"
        )
        suffixes: dict[str, str] = {}

        for mode in ("faithful", "editorial", "cinematic", "concept_art", "wild"):
            contract = self.nodes._ltx_creativity_instruction(
                mode,
                1.5,
                "image_to_video",
            )
            with self.subTest(mode=mode):
                self.assertIn(precedence, contract)
            suffixes[mode] = contract[contract.index(precedence) :].casefold()

        prohibition_cues = (
            "do not",
            "must not",
            "may not",
            "never",
            "forbid",
            "add no",
            "no extra",
            "no new",
            "no additional",
            "no unrequested",
        )

        for mode, suffix in suffixes.items():
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", suffix)
                if sentence.strip()
            ]
            with self.subTest(mode=mode, contract="opening_anchor"):
                self.assertIn("opening pose", suffix)
                self.assertIn("opening camera geometry", suffix)
                self.assertIn("does not fix the future camera trajectory", suffix)
                self.assertIn("does not", suffix)
                self.assertIn("locked camera afterward", suffix)
                self.assertRegex(
                    suffix,
                    r"explicit locked-off or static-camera instruction[^.]*wins over the creativity mode",
                )

            for family in (
                ("cut", "cuts"),
                ("actor", "actors", "subject", "subjects"),
                ("location", "locations"),
                ("time of day",),
            ):
                relevant = [
                    sentence
                    for sentence in sentences
                    if any(re.search(rf"\b{re.escape(noun)}\b", sentence) for noun in family)
                ]
                with self.subTest(mode=mode, protected_family=family[0]):
                    self.assertTrue(
                        relevant,
                        f"Conditioned contract never mentions {family[0]} additions.",
                    )
                    self.assertTrue(
                        any(any(cue in sentence for cue in prohibition_cues) for sentence in relevant),
                        f"Conditioned contract mentions {family[0]} but does not prohibit adding one.",
                    )

        faithful_suffix = suffixes["faithful"]
        self.assertIn("least-invasive", faithful_suffix)
        self.assertNotIn("selected mode's dynamic", faithful_suffix)
        for mode in ("editorial", "cinematic", "concept_art", "wild"):
            with self.subTest(mode=mode, contract="unspecified_path_policy"):
                self.assertIn("otherwise unspecified future camera path", suffixes[mode])
                self.assertIn("selected mode's dynamic", suffixes[mode])
                self.assertIn("physically continuous camera choreography", suffixes[mode])

    def test_blocked_splitter_routes_blanks_but_exposes_candidate_diagnostics(self) -> None:
        candidate_ltx = (
            "A static medium shot at eye level holds on a woman in a quiet studio. "
            "She takes one measured step while the locked camera remains steady."
        )
        candidate_negative = "extra people, duplicate subject, subtitles"
        candidate_h3 = (
            "integrated_multimodal_description: [Shot 1] A locked view holds.\n\n"
            "overall_soundscape: N/A\n\nnon_diegetic_music: N/A"
        )
        grounding_reasons = ["visual_grounding_unverified"]
        packet = {
            "ltx_prompt": candidate_ltx,
            "ideogram_prompt": "",
            "minimax_h3_prompt": candidate_h3,
            "negative_prompt": candidate_negative,
            "scene_segments": [],
            "metadata": {
                "target_profile": "ltx",
                "ready_for_generation": False,
                "blocked_reasons": grounding_reasons,
                "grounding_guard": {
                    "schema": "dg-grounding-report/1",
                    "config": {"mode": "strict"},
                    "analysis_status": "uncertain",
                    "decision": "block",
                    "would_block": True,
                    "grounding_guard_would_block": True,
                    "blocked_reasons": grounding_reasons,
                },
            },
        }
        context = self.nodes.GemmaContext(
            user_prompt="A woman takes one measured step in a quiet studio.",
            source="none",
            media_metadata={"source": "none"},
        )
        target = self.nodes.TargetProfileConfig(
            target_profile="ltx",
            ltx_generation_mode="text_to_video",
        )

        split = self.nodes.DiffusionGemmaJSONSplitter().split(
            json.dumps(packet),
            context,
            target,
        )

        self.assertEqual(
            self.nodes.DiffusionGemmaJSONSplitter.RETURN_NAMES[:14],
            (
                "ltx_prompt",
                "ideogram_prompt",
                "negative_prompt",
                "aspect_ratio",
                "metadata_json",
                "scene_segments_json",
                "local_prompts",
                "segment_lengths",
                "is_valid",
                "resolution_selector_preset",
                "resolution_width",
                "resolution_height",
                "ready_for_generation",
                "minimax_h3_prompt",
            ),
        )
        self.assertEqual(
            self.nodes.DiffusionGemmaJSONSplitter.RETURN_NAMES[14:17],
            (
                "candidate_ltx_prompt",
                "candidate_negative_prompt",
                "candidate_minimax_h3_prompt",
            ),
        )
        self.assertEqual(
            self.nodes.DiffusionGemmaJSONSplitter.RETURN_TYPES[14:17],
            ("STRING", "STRING", "STRING"),
        )
        self.assertEqual(len(split), 17)
        self.assertEqual(split[0], "")
        self.assertEqual(split[2], "")
        self.assertFalse(split[12])
        self.assertEqual(split[14], candidate_ltx)
        self.assertEqual(split[15], candidate_negative)
        self.assertEqual(split[16], "")

        h3_candidate = "integrated_multimodal_description: [Shot 1] incomplete"
        h3_packet = {
            "ltx_prompt": "",
            "ideogram_prompt": "",
            "minimax_h3_prompt": h3_candidate,
            "negative_prompt": "",
            "scene_segments": [],
            "metadata": {
                "target_profile": "minimax_h3",
                "ready_for_generation": False,
                "blocked_reasons": ["minimax_h3_missing_overall_soundscape"],
            },
        }
        h3_target = self.nodes.TargetProfileConfig(
            target_profile="minimax_h3",
            minimax_h3_mode="t2va",
        )
        h3_split = self.nodes.DiffusionGemmaJSONSplitter().split(
            json.dumps(h3_packet),
            context,
            h3_target,
        )
        self.assertEqual(h3_split[13], "")
        self.assertEqual(h3_split[16], h3_candidate)

    def test_generation_gate_warn_policy_only_bypasses_h3_validation(self) -> None:
        gate = self.nodes.DiffusionGemmaGenerationGate()
        candidate = "A non-empty native MiniMax-H3 prompt candidate."
        h3_metadata = json.dumps(
            {
                "target_profile": "minimax_h3",
                "blocked_reasons": [
                    "minimax_h3_cut_transition_invalid",
                    "minimax_h3_dialogue_speaker_invalid",
                ],
            }
        )

        with self.assertRaises(ValueError):
            gate.gate(candidate, False, h3_metadata, "strict")
        self.assertEqual(
            gate.gate(candidate, False, h3_metadata, "warn_and_continue"),
            (candidate,),
        )

        with self.assertRaisesRegex(ValueError, "json_parse_invalid"):
            gate.gate(
                candidate,
                False,
                json.dumps(
                    {
                        "blocked_reasons": [
                            "minimax_h3_cut_transition_invalid",
                            "json_parse_invalid",
                        ]
                    }
                ),
                "warn_and_continue",
            )
        with self.assertRaisesRegex(ValueError, "did not pass validation"):
            gate.gate(
                "",
                False,
                h3_metadata,
                "warn_and_continue",
            )

    def test_generation_gate_warn_policy_never_bypasses_strict_grounding(self) -> None:
        gate = self.nodes.DiffusionGemmaGenerationGate()
        metadata = {
            "blocked_reasons": ["minimax_h3_cut_transition_invalid"],
            "grounding_guard": {
                "schema": "dg-grounding-report/1",
                "config": {"mode": "strict"},
                "analysis_status": "uncertain",
                "decision": "block",
                "would_block": True,
                "grounding_guard_would_block": True,
                "blocked_reasons": ["visual_grounding_unverified"],
            },
        }
        with self.assertRaisesRegex(ValueError, "Grounding Guard blocked generation"):
            gate.gate(
                "A non-empty candidate.",
                False,
                json.dumps(metadata),
                "warn_and_continue",
            )


if __name__ == "__main__":
    unittest.main()
