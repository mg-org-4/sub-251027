from __future__ import annotations

import importlib.util
import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
SCHEMA = ROOT / "schemas" / "splatstage_show_blueprint.schema.json"

DIRECTOR_NODE_IDS = {
    "DiffusionGemmaModelLoader",
    "DiffusionGemmaContextHub",
    "DiffusionGemmaH3ReferenceContext",
    "DiffusionGemmaTargetProfile",
    "DiffusionGemmaLTX25TargetProfile",
    "DiffusionGemmaMiniMaxH3TargetProfile",
    "DiffusionGemmaIdeogram4TargetProfile",
    "DiffusionGemmaGroundingGuardSettings",
    "DiffusionGemmaCoTGenerator",
    "DiffusionGemmaJSONSplitter",
    "DiffusionGemmaGenerationGate",
    "DiffusionGemmaBranchGenerationGate",
}
EXPERIMENTAL_NODE_IDS = {
    "DiffusionGemmaSplatStagePlanner",
    "DiffusionGemmaSplatStageEfficientPlanner",
}
OPTIONAL_NODE_IDS = {
    "DiffusionGemmaH3ReferencePolicy",
    "DiffusionGemmaReferencePrep",
    "DiffusionGemmaH3ReferencePairPrep",
}


def load_nodes_module():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    package_name = f"diffusiongemma_splatstage_fixture_{uuid.uuid4().hex}"
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


class SplatStagePlannerMusicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    def test_strict_schema_is_bundled_with_the_package(self) -> None:
        self.assertEqual(self.nodes._splatstage_blueprint_schema_path(), SCHEMA)

    def test_prompt_requests_a_full_song_before_the_video_excerpt(self) -> None:
        prompt = self.nodes._splatstage_blueprint_prompt(
            "make a pop-music video",
            42,
            self.schema,
            duration_seconds=20.0,
            aspect_ratio="16:9",
        )
        self.assertIn("complete 90-second source song", prompt)
        self.assertIn("final video uses a 20-second excerpt", prompt)
        self.assertIn("two separately generated performer clips", prompt)
        self.assertIn("one performer object, one background object, and one FX object", prompt)
        self.assertIn("never arrays or clip-specific lists", prompt)
        self.assertIn("target 10 to 14 sung lines", prompt)
        self.assertIn("6 to 10 syllables per line", prompt)
        self.assertIn("about two bars per sung line", prompt)
        self.assertIn("Repeat the Chorus wording verbatim", prompt)
        self.assertIn("[Instrumental] intro, interlude, or outro", prompt)
        self.assertIn("dedicated metadata", prompt)
        self.assertIn("ordinary caption language", prompt)
        self.assertIn("Source passthrough mode is active", prompt)
        self.assertIn("reach ACE-Step unchanged", prompt)
        self.assertIn("When Context Hub image pixels are attached", prompt)
        self.assertNotIn("Roman-numeral progression", prompt.split("Music production strategy:", 1)[0])
        self.assertIn("SAM3 object-category bundle", prompt)
        self.assertIn("wide-brimmed mariachi hat", prompt)
        self.assertIn("electric guitar", prompt)
        self.assertIn("positive_prompt must use affirmative visual language only", prompt)
        self.assertIn("person, people, human, man, men, woman, women", prompt)
        self.assertIn("silhouette, crowd, performer, presenter, dancer, or singer", prompt)
        self.assertIn('"schema":"splatstage.show_blueprint"', prompt)
        self.assertIn('"version":1', prompt)
        self.assertIn(
            'Never put the schema document identifier "splatstage.show_blueprint@1"',
            prompt,
        )

    def test_planner_contract_preserves_input_order_and_allows_open_duration_and_aspect(self) -> None:
        required = self.nodes.DiffusionGemmaSplatStagePlanner.INPUT_TYPES()["required"]
        self.assertEqual(
            list(required),
            [
                "model_config",
                "gemma_context",
                "root_seed",
                "duration_seconds",
                "aspect_ratio",
                "temperature",
                "max_new_tokens",
                "production_mode",
            ],
        )
        duration_type, duration_spec = required["duration_seconds"]
        self.assertEqual(duration_type, "FLOAT")
        self.assertEqual(duration_spec["default"], 20.0)
        self.assertGreater(duration_spec["min"], 0.0)
        self.assertNotIn("max", duration_spec)
        aspect_type, aspect_spec = required["aspect_ratio"]
        self.assertEqual(aspect_type, "STRING")
        self.assertEqual(aspect_spec["default"], "16:9")
        production_choices, production_spec = required["production_mode"]
        self.assertEqual(
            production_choices,
            ["Source passthrough", "Joint video-safe plan", "Audition and select"],
        )
        self.assertEqual(production_spec["default"], "Source passthrough")

    def test_production_modes_are_distinct_upstream_planning_contracts(self) -> None:
        source = self.nodes._splatstage_blueprint_prompt(
            "make a pop-music video",
            42,
            self.schema,
            duration_seconds=20.0,
            aspect_ratio="16:9",
            production_mode="Source passthrough",
        )
        safe = self.nodes._splatstage_blueprint_prompt(
            "make a pop-music video",
            42,
            self.schema,
            duration_seconds=20.0,
            aspect_ratio="16:9",
            production_mode="Joint video-safe plan",
        )
        audition = self.nodes._splatstage_blueprint_prompt(
            "make a pop-music video",
            42,
            self.schema,
            duration_seconds=20.0,
            aspect_ratio="16:9",
            production_mode="Audition and select",
        )
        self.assertIn("reach ACE-Step unchanged", source)
        self.assertIn("no double-time drum illusion", safe)
        self.assertIn("stable tonal center", safe)
        self.assertIn("several independent song seeds", audition)
        self.assertIn("moderate and even transient density", audition)
        self.assertIn("stable tonal family", audition)
        for prompt in (source, safe, audition):
            self.assertIn("Do not repeat a numeric BPM", prompt)
            self.assertIn("6 to 10 syllables", prompt)

    def test_explicit_music_genre_overrides_visual_setting_in_both_planners(self) -> None:
        creative_brief = (
            "Create a strict country soundtrack juxtaposed against a neon nightclub."
        )
        standard = self.nodes._splatstage_blueprint_prompt(
            creative_brief,
            42,
            self.schema,
            duration_seconds=20.0,
            aspect_ratio="16:9",
            visual_context="A singer dances beneath neon club lighting.",
        )
        efficient_schema = json.loads(
            (
                ROOT
                / "schemas"
                / "splatstage_efficient_show_blueprint.schema.json"
            ).read_text(encoding="utf-8")
        )
        efficient = self.nodes._splatstage_efficient_blueprint_prompt(
            creative_brief,
            42,
            efficient_schema,
        )
        shared_rule = self.nodes._SPLATSTAGE_EXPLICIT_GENRE_AUTHORITY_INSTRUCTION

        for prompt in (standard, efficient):
            self.assertIn(shared_rule, prompt)
            self.assertIn("immutable, highest-priority music constraint", prompt)
            self.assertIn("audiovisual contrast, not musical fusion", prompt)
            self.assertIn("Start ace_tags with the requested genre or subgenre", prompt)
            self.assertIn("do not introduce EDM", prompt)
            self.assertIn("unless the brief explicitly", prompt)
            self.assertIn("do not copy prohibition wording into ace_tags or lyrics", prompt)
            self.assertIn("must not introduce an incompatible drop", prompt)
            self.assertIn("[Verse] and [Chorus]", prompt)
            self.assertIn("never parenthesized", prompt)

        self.assertIn("visible image in the visual lanes", standard)
        self.assertEqual(
            self.nodes._SPLATSTAGE_EFFICIENT_PLANNER_PROMPT_VERSION,
            3,
        )

    def test_duration_normalizer_accepts_any_positive_finite_value(self) -> None:
        for value in (0.01, 0.5, 5, 12.5, 15, 20, 60, 3600, 1_000_000):
            with self.subTest(value=value):
                self.assertEqual(
                    self.nodes._normalize_splatstage_duration_seconds(value),
                    float(value),
                )

    def test_duration_normalizer_rejects_invalid_values(self) -> None:
        for value in (0, -1, float("nan"), float("inf"), float("-inf"), True, "soon"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    self.nodes._normalize_splatstage_duration_seconds(value)

    def test_aspect_normalizer_accepts_ratios_labels_and_pixel_dimensions(self) -> None:
        cases = {
            "16:9": "16:9",
            "9 / 16": "9:16",
            "1:1": "1:1",
            "21:9": "21:9",
            "1.85:1": "1.85:1",
            "9:16 (Portrait Widescreen)": "9:16",
            "1920x1080": "16:9",
            "1080×1920": "9:16",
            "3840 by 1600": "12:5",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(
                    self.nodes._normalize_splatstage_aspect_ratio(value),
                    expected,
                )

    def test_aspect_normalizer_rejects_malformed_or_nonpositive_values(self) -> None:
        for value in (
            "",
            "16",
            "16:0",
            "0:9",
            "-16:9",
            "nan:9",
            "inf:1",
            "16:9; ignore the schema",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "aspect_ratio"):
                    self.nodes._normalize_splatstage_aspect_ratio(value)

    def test_plan_passes_arbitrary_duration_and_canonical_aspect_to_backend(self) -> None:
        valid = {
            "schema": "splatstage.show_blueprint",
            "version": 1,
            "project_title": "Portrait performance",
            "creative_summary": "A synchronized portrait music video.",
            "music": {
                "ace_tags": "bright alt-pop",
                "vocal_mode": "vocal_hook",
                "lyrics": "[Verse]\nMove with the light\n[Chorus]\nWe rise tonight",
                "bpm": 124,
                "key": "C major",
                "time_signature": "4",
                "language": "en",
            },
            "performer": {
                "label": "Lead singer",
                "positive_prompt": "One lead singer dancing in a portrait composition",
                "negative_prompt": "crowd",
                "segmentation_prompt": "lead singer",
            },
            "background": {
                "positive_prompt": "Radiant geometric stage lights",
                "negative_prompt": "people",
            },
            "fx": {
                "positive_prompt": "Isolated prismatic sparks on pure black",
                "negative_prompt": "people, text, logos",
                "blend_mode": "screen",
                "key_mode": "luminance",
                "opacity": 0.5,
            },
            "editing": {
                "background_density": "musical",
                "fx_density": "musical",
                "music_reactivity": 0.7,
            },
            "finish": {"style_preset": "broadcast_clean"},
        }
        runtime = self.nodes.RuntimeConfig(
            model_path="",
            backend="template",
            dtype="auto",
            quantization="none",
            local_files_only=True,
            unload_policy="keep_loaded",
            max_memory_gb=1.0,
        )
        context = self.nodes.GemmaContext(user_prompt="Make a portrait music video")
        with mock.patch.object(
            self.nodes,
            "_run_backend",
            return_value=json.dumps(valid),
        ) as backend:
            result = self.nodes.DiffusionGemmaSplatStagePlanner().plan(
                runtime,
                context,
                42,
                60.0,
                "1080x1920",
                0.35,
                2048,
            )
        model_prompt = backend.call_args.args[1]
        self.assertIn("duration 60 seconds, aspect ratio 9:16", model_prompt)
        self.assertEqual(result["result"][1], True)
        raw_record = json.loads(result["result"][3])
        self.assertEqual(
            raw_record["production_context"],
            {
                "duration_seconds": 60.0,
                "aspect_ratio": "9:16",
                "production_mode": "source_passthrough",
                "pixels_sent_to_backend": False,
                "visual_context_present": False,
            },
        )

    def test_unparseable_initial_response_retries_the_full_schema_task(self) -> None:
        valid = {
            "schema": "splatstage.show_blueprint",
            "version": 1,
            "project_title": "Country contrast",
            "creative_summary": "A country performance against neon club visuals.",
            "music": {
                "ace_tags": "traditional country, acoustic guitar, pedal steel, fiddle",
                "vocal_mode": "vocal_hook",
                "lyrics": "[Verse]\nDust turns gold tonight\n[Chorus]\nTake the long road home",
                "bpm": 92,
                "key": "G major",
                "time_signature": "4",
                "language": "en",
            },
            "performer": {
                "label": "Country singer",
                "positive_prompt": "One country singer performing beneath magenta light",
                "negative_prompt": "crowd, duplicate person, text",
                "segmentation_prompt": "country singer, acoustic guitar",
            },
            "background": {
                "positive_prompt": "Empty neon nightclub interior with cobalt haze",
                "negative_prompt": "people, performers, crowd, text",
            },
            "fx": {
                "positive_prompt": "Isolated amber dust ribbons on pure black",
                "negative_prompt": "people, faces, text, logos",
                "blend_mode": "screen",
                "key_mode": "luminance",
                "opacity": 0.5,
            },
            "editing": {
                "background_density": "musical",
                "fx_density": "calm",
                "music_reactivity": 0.5,
            },
            "finish": {"style_preset": "broadcast_clean"},
        }
        runtime = self.nodes.RuntimeConfig(
            model_path="",
            backend="template",
            dtype="auto",
            quantization="none",
            local_files_only=True,
            unload_policy="keep_loaded",
            max_memory_gb=1.0,
        )
        context = self.nodes.GemmaContext(
            user_prompt=(
                "Create a strict country soundtrack juxtaposed against a neon nightclub."
            )
        )
        with mock.patch.object(
            self.nodes,
            "_run_backend",
            side_effect=["I cannot provide that blueprint.", json.dumps(valid)],
        ) as backend:
            result = self.nodes.DiffusionGemmaSplatStagePlanner().plan(
                runtime,
                context,
                42,
                25.0,
                "16:9",
                0.35,
                2048,
            )

        self.assertEqual(backend.call_count, 2)
        retry_prompt = backend.call_args_list[1].args[1]
        self.assertIn("previous answer had no parseable JSON object", retry_prompt)
        self.assertIn("Creative brief:\n" + context.user_prompt, retry_prompt)
        self.assertIn("JSON Schema:", retry_prompt)
        self.assertIn(
            self.nodes._SPLATSTAGE_EXPLICIT_GENRE_AUTHORITY_INSTRUCTION,
            retry_prompt,
        )
        self.assertNotIn("Validation errors:", retry_prompt)
        self.assertEqual(result["result"][1], True)
        raw_record = json.loads(result["result"][3])
        self.assertEqual(
            raw_record["repair_strategy"],
            "full_task_retry_after_unparseable_response",
        )

    def test_people_in_background_or_fx_positive_prompt_are_rejected_before_router(self) -> None:
        valid = {
            "schema": "splatstage.show_blueprint",
            "version": 1,
            "project_title": "Retro party",
            "creative_summary": "A synchronized retro performance.",
            "music": {
                "ace_tags": "retro alt-pop",
                "vocal_mode": "vocal_hook",
                "lyrics": "[Verse]\nTurn up the light\n[Chorus]\nWe own the night",
                "bpm": 120,
                "key": "C major",
                "time_signature": "4",
                "language": "en",
            },
            "performer": {
                "label": "Lead singer",
                "positive_prompt": "One lead singer dancing beneath pink light",
                "negative_prompt": "crowd",
                "segmentation_prompt": "lead singer",
            },
            "background": {
                "positive_prompt": "Dim party room with pink and blue ambient lighting",
                "negative_prompt": "people, faces",
            },
            "fx": {
                "positive_prompt": "Isolated prismatic sparks on pure black",
                "negative_prompt": "people, text, logos",
                "blend_mode": "screen",
                "key_mode": "luminance",
                "opacity": 0.5,
            },
            "editing": {
                "background_density": "musical",
                "fx_density": "musical",
                "music_reactivity": 0.7,
            },
            "finish": {"style_preset": "broadcast_clean"},
        }
        leaked_background = json.loads(json.dumps(valid))
        leaked_background["background"]["positive_prompt"] = (
            "dimly lit indoor party room, pink and blue ambient lighting, "
            "blurred silhouettes of people in distance, 90s decor, hazy atmosphere"
        )
        errors = self.nodes._splatstage_blueprint_errors(
            leaked_background,
            self.schema,
        )
        self.assertTrue(
            any(
                error.startswith("$.background.positive_prompt: people-free plate required")
                and "'people'" in error
                for error in errors
            )
        )

        leaked_fx = json.loads(json.dumps(valid))
        leaked_fx["fx"]["positive_prompt"] = "A singer-shaped light figure on pure black"
        self.assertTrue(
            any(
                error.startswith("$.fx.positive_prompt: people-free plate required")
                and "'singer'" in error
                for error in self.nodes._splatstage_blueprint_errors(
                    leaked_fx,
                    self.schema,
                )
            )
        )

    def test_planner_repairs_people_leak_before_emitting_blueprint(self) -> None:
        invalid = {
            "schema": "splatstage.show_blueprint",
            "version": 1,
            "project_title": "Retro party",
            "creative_summary": "A synchronized retro performance.",
            "music": {
                "ace_tags": "retro alt-pop",
                "vocal_mode": "vocal_hook",
                "lyrics": "[Verse]\nTurn up the light\n[Chorus]\nWe own the night",
                "bpm": 120,
                "key": "C major",
                "time_signature": "4",
                "language": "en",
            },
            "performer": {
                "label": "Lead singer",
                "positive_prompt": "One lead singer dancing beneath pink light",
                "negative_prompt": "crowd",
                "segmentation_prompt": "lead singer",
            },
            "background": {
                "positive_prompt": "Blurred silhouettes of people in a hazy party room",
                "negative_prompt": "faces",
            },
            "fx": {
                "positive_prompt": "Isolated prismatic sparks on pure black",
                "negative_prompt": "people, text, logos",
                "blend_mode": "screen",
                "key_mode": "luminance",
                "opacity": 0.5,
            },
            "editing": {
                "background_density": "musical",
                "fx_density": "musical",
                "music_reactivity": 0.7,
            },
            "finish": {"style_preset": "broadcast_clean"},
        }
        repaired = json.loads(json.dumps(invalid))
        repaired["background"]["positive_prompt"] = (
            "Dim party room with pink and blue ambient lighting, 90s decor, and haze"
        )
        repaired["background"]["negative_prompt"] = "people, faces"
        runtime = self.nodes.RuntimeConfig(
            model_path="",
            backend="template",
            dtype="auto",
            quantization="none",
            local_files_only=True,
            unload_policy="keep_loaded",
            max_memory_gb=1.0,
        )
        context = self.nodes.GemmaContext(user_prompt="Make a retro party music video")
        with mock.patch.object(
            self.nodes,
            "_run_backend",
            side_effect=[json.dumps(invalid), json.dumps(repaired)],
        ) as backend:
            result = self.nodes.DiffusionGemmaSplatStagePlanner().plan(
                runtime,
                context,
                42,
                20.0,
                "9:16",
                0.35,
                2048,
            )
        self.assertEqual(backend.call_count, 2)
        repair_prompt = backend.call_args_list[1].args[1]
        self.assertIn("$.background.positive_prompt: people-free plate required", repair_prompt)
        emitted = json.loads(result["result"][0])
        self.assertEqual(
            emitted["background"]["positive_prompt"],
            repaired["background"]["positive_prompt"],
        )
        self.assertEqual(
            self.nodes._splatstage_blueprint_errors(emitted, self.schema),
            [],
        )

    def test_planner_sends_context_hub_pixels_and_visual_context_to_backend(self) -> None:
        runtime = self.nodes.RuntimeConfig(
            model_path="unused",
            backend="transformers_inprocess",
            dtype="auto",
            quantization="none",
            local_files_only=True,
            unload_policy="keep_loaded",
            max_memory_gb=1.0,
            status={"supports_pixels": True},
        )
        pixels = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
        context = self.nodes.GemmaContext(
            user_prompt="Write a song and video for this image",
            images=pixels,
            source="image",
            media_metadata={"visual_description": "A singer in a cobalt dress beneath amber lights."},
            visual_description="A singer in a cobalt dress beneath amber lights.",
        )
        with mock.patch.object(self.nodes, "_run_backend", return_value="{}") as backend:
            with self.assertRaisesRegex(RuntimeError, "could not produce a valid"):
                self.nodes.DiffusionGemmaSplatStagePlanner().plan(
                    runtime,
                    context,
                    42,
                    90.0,
                    "9:16",
                    0.35,
                    2048,
                )
        first_call = backend.call_args_list[0]
        self.assertIn("A singer in a cobalt dress beneath amber lights.", first_call.args[1])
        media_context = first_call.args[2]
        self.assertIsNotNone(media_context)
        self.assertIs(media_context.images, pixels)
        self.assertIs(backend.call_args_list[1].args[2], media_context)

    def test_full_song_lyric_budget_accepts_48_lines_but_not_49(self) -> None:
        def blueprint(line_count: int) -> dict:
            return {
                "schema": "splatstage.show_blueprint",
                "version": 1,
                "project_title": "Song",
                "creative_summary": "A compact performance video.",
                "music": {
                    "ace_tags": "bright pop",
                    "vocal_mode": "vocal_hook",
                    "lyrics": "\n".join(
                        f"Original lyric line {index}" for index in range(line_count)
                    ),
                    "bpm": 120,
                    "key": "C major",
                    "time_signature": "4",
                    "language": "en",
                },
                "performer": {
                    "label": "Singer",
                    "positive_prompt": "One singer performing",
                    "negative_prompt": "crowd",
                    "segmentation_prompt": "the singer",
                },
                "background": {
                    "positive_prompt": "Abstract lights",
                    "negative_prompt": "people",
                },
                "fx": {
                    "positive_prompt": "Prism sparks",
                    "negative_prompt": "people, text",
                    "blend_mode": "screen",
                    "key_mode": "luminance",
                    "opacity": 0.5,
                },
                "editing": {
                    "background_density": "musical",
                    "fx_density": "musical",
                    "music_reactivity": 0.5,
                },
                "finish": {"style_preset": "broadcast_clean"},
            }

        self.assertEqual(
            self.nodes._splatstage_blueprint_errors(
                blueprint(48), self.schema
            ),
            [],
        )
        self.assertTrue(
            any(
                "at most 48" in error
                for error in self.nodes._splatstage_blueprint_errors(
                    blueprint(49), self.schema
                )
            )
        )

    def test_segmentation_prompt_uses_separate_sam3_object_categories(self) -> None:
        valid = {
            "schema": "splatstage.show_blueprint",
            "version": 1,
            "project_title": "Mariachi",
            "creative_summary": "A cyberpunk mariachi performance.",
            "music": {
                "ace_tags": "cyberpunk mariachi",
                "vocal_mode": "instrumental",
                "lyrics": "[Instrumental]",
                "bpm": 128,
                "key": "G minor",
                "time_signature": "4",
                "language": "en",
            },
            "performer": {
                "label": "Lead mariachi",
                "positive_prompt": "One cyberpunk mariachi playing a glowing guitar",
                "negative_prompt": "crowd",
                "segmentation_prompt": (
                    "cyberpunk mariachi musician, black charro suit, "
                    "wide-brimmed mariachi hat, glowing electric guitar"
                ),
            },
            "background": {
                "positive_prompt": "Martian landscape",
                "negative_prompt": "people",
            },
            "fx": {
                "positive_prompt": "isolated sparks",
                "negative_prompt": "people, text",
                "blend_mode": "screen",
                "key_mode": "luminance",
                "opacity": 0.5,
            },
            "editing": {
                "background_density": "musical",
                "fx_density": "musical",
                "music_reactivity": 0.5,
            },
            "finish": {"style_preset": "broadcast_clean"},
        }
        self.assertEqual(
            self.nodes._splatstage_blueprint_errors(valid, self.schema),
            [],
        )
        leaked = json.loads(json.dumps(valid))
        leaked["fx"]["positive_prompt"] = "Neon sparks, no words, text-free overlay"
        polarity_errors = self.nodes._splatstage_blueprint_errors(leaked, self.schema)
        self.assertTrue(any("affirmative visual language" in error for error in polarity_errors))
        invalid = json.loads(json.dumps(valid))
        invalid["performer"]["segmentation_prompt"] = (
            "the musician in the charro suit and guitar"
        )
        self.assertTrue(
            any(
                "separate the performer" in error
                for error in self.nodes._splatstage_blueprint_errors(
                    invalid, self.schema
                )
            )
        )

    def test_two_option_lane_lists_are_seed_addressed_to_one_coherent_contract(self) -> None:
        candidate = {
            "performer": [{"label": "firefighter"}, {"label": "country singer"}],
            "background": [{"name": "fire station"}, {"name": "burning skyline"}],
            "fx": [{"name": "embers"}, {"name": "smoke"}],
        }
        normalized, report = self.nodes._normalize_splatstage_lane_cardinality(
            candidate,
            selection_index=5,
        )
        self.assertEqual(normalized["performer"]["label"], "country singer")
        self.assertEqual(normalized["background"]["name"], "burning skyline")
        self.assertEqual(normalized["fx"]["name"], "smoke")
        self.assertEqual(report["performer"]["selected_index"], 1)
        self.assertIsInstance(candidate["performer"], list)

    def test_malformed_or_oversized_lane_lists_remain_invalid(self) -> None:
        candidate = {
            "performer": [{"label": "a"}, {"label": "b"}, {"label": "c"}],
            "background": [],
            "fx": ["not an object"],
        }
        normalized, report = self.nodes._normalize_splatstage_lane_cardinality(
            candidate,
            selection_index=0,
        )
        self.assertEqual(normalized, candidate)
        self.assertEqual(report, {})

    def test_schema_document_id_alias_is_canonicalized_only_for_matching_version(self) -> None:
        candidate = {
            "schema": "splatstage.show_blueprint@1",
            "version": 1,
            "project_title": "Alias candidate",
        }
        normalized, report = self.nodes._normalize_splatstage_blueprint_identity(
            candidate,
            self.schema,
        )
        self.assertEqual(normalized["schema"], "splatstage.show_blueprint")
        self.assertEqual(report["input_schema"], "splatstage.show_blueprint@1")
        self.assertEqual(candidate["schema"], "splatstage.show_blueprint@1")

        wrong_version = dict(candidate, version=2)
        unchanged, report = self.nodes._normalize_splatstage_blueprint_identity(
            wrong_version,
            self.schema,
        )
        self.assertEqual(unchanged, wrong_version)
        self.assertEqual(report, {})

        unknown_alias = dict(candidate, schema="splatstage.show_blueprint@999")
        unchanged, report = self.nodes._normalize_splatstage_blueprint_identity(
            unknown_alias,
            self.schema,
        )
        self.assertEqual(unchanged, unknown_alias)
        self.assertEqual(report, {})


class EfficientSplatStagePlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()
        cls.schema = json.loads(
            (ROOT / "schemas" / "splatstage_efficient_show_blueprint.schema.json").read_text(
                encoding="utf-8"
            )
        )

    @staticmethod
    def valid_blueprint() -> dict:
        return {
            "schema": "splatstage.efficient_show_blueprint",
            "version": 1,
            "project_title": "Cyber Mariachi",
            "creative_summary": "A cyberpunk mariachi performance on Mars.",
            "music": {
                "ace_tags": "cyber mariachi pop, vocal hook",
                "vocal_mode": "vocal_hook",
                "lyrics": "[Verse]\nRed dust in the light\n[Chorus]\nPlay it to the stars",
                "bpm": 124,
                "key": "C major",
                "time_signature": "4",
                "language": "en",
            },
            "performer": {
                "identity_description": "One cyberpunk mariachi guitarist in a black charro suit",
                "positive_prompt": "Full-body performance beneath magenta concert light",
                "negative_prompt": "crowd, duplicate person, text, logos",
                "segmentation_prompt": "mariachi guitarist, mariachi hat, charro suit, electric guitar",
                "variations": [
                    "Low-angle strumming with a slow dolly",
                    "Energetic side view with rhythmic footwork",
                    "Centered hero framing with sweeping arm motion",
                ],
            },
            "background": {
                "positive_prompt": "Martian neon concert architecture and red dust",
                "negative_prompt": "people, performers, crowd, text, logos",
                "variations": [
                    "Wide canyon stage with drifting haze",
                    "Graphic orbital city vista with slow parallax",
                ],
            },
            "fx": {
                "positive_prompt": "Isolated luminous prismatic guitar-wave geometry on a pure black background",
                "negative_prompt": "people, faces, text, words, letters, logos, symbols",
                "blend_mode": "screen",
                "key_mode": "luminance",
                "opacity": 0.55,
            },
            "editing": {"density": "musical", "music_reactivity": "medium"},
            "finish": {"style_preset": "broadcast_clean"},
        }

    def test_schema_requires_exact_candidate_counts_and_affirmative_prompts(self) -> None:
        value = self.valid_blueprint()
        self.assertEqual(
            self.nodes._splatstage_efficient_blueprint_errors(value, self.schema), []
        )
        value["performer"]["variations"] = value["performer"]["variations"][:2]
        value["fx"]["positive_prompt"] = "Glowing particles, no words, text-free"
        errors = self.nodes._splatstage_efficient_blueprint_errors(value, self.schema)
        self.assertTrue(any("variations" in error for error in errors))
        self.assertTrue(any("affirmative" in error for error in errors))

    def test_prompt_locks_full_song_short_assets_and_semantic_union(self) -> None:
        prompt = self.nodes._splatstage_efficient_blueprint_prompt(
            "make a pop-music video", 42, self.schema
        )
        self.assertIn("complete 90-second song", prompt)
        self.assertIn("three separate five-second performer videos", prompt)
        self.assertIn("canonical performer identity first", prompt)
        self.assertIn("only concrete silhouette-extending items actually present", prompt)
        self.assertIn("richly art-directed luminous motif", prompt)
        self.assertIn("limited color palette", prompt)
        self.assertIn("rhythmic energy or", prompt)
        self.assertIn("clear spatial composition", prompt)
        self.assertIn("Do not name TouchDesigner", prompt)
        self.assertIn("locked TouchDesigner-inspired", prompt)
        self.assertIn("affirmative visual language", prompt)
        self.assertIn("10 to 14 sung lyric", prompt)
        self.assertIn("6 to 10 syllables per line", prompt)
        self.assertIn("repeat the Chorus verbatim", prompt)
        self.assertIn("one stable", prompt)
        self.assertIn("diatonic four-bar phrase loops", prompt)

    def test_fx_creative_slot_rejects_software_or_interface_language(self) -> None:
        value = self.valid_blueprint()
        value["fx"]["positive_prompt"] = (
            "TouchDesigner application screenshot with a cyan particle vortex"
        )
        errors = self.nodes._splatstage_efficient_blueprint_errors(value, self.schema)
        self.assertTrue(
            any("locked TouchDesigner-inspired" in error for error in errors)
        )

    def test_efficient_segmentation_rejects_free_standing_generic_categories(self) -> None:
        value = self.valid_blueprint()
        value["performer"]["segmentation_prompt"] = (
            "mariachi guitarist, face and head, hair and worn headwear, "
            "costume and worn accessories, held instrument and attached props"
        )
        errors = self.nodes._splatstage_efficient_blueprint_errors(value, self.schema)
        self.assertTrue(any("identity-specific items" in error for error in errors))

    def test_efficient_segmentation_uses_identity_and_specific_items_only(self) -> None:
        value = self.valid_blueprint()
        value["performer"]["segmentation_prompt"] = (
            "cyberpunk mariachi guitarist, face and head, hair and worn headwear, "
            "black charro suit, electric guitar"
        )
        normalized = self.nodes._normalize_splatstage_efficient_segmentation(value)
        segmentation = normalized["performer"]["segmentation_prompt"]
        self.assertTrue(
            segmentation.startswith("One cyberpunk mariachi guitarist in a black charro suit")
        )
        self.assertNotIn("face and head", segmentation)
        self.assertNotIn("hair and worn headwear", segmentation)
        self.assertIn("black charro suit", segmentation)
        self.assertIn("electric guitar", segmentation)
        self.assertLessEqual(len(segmentation.split(",")), 6)
        self.assertEqual(
            self.nodes._splatstage_efficient_blueprint_errors(normalized, self.schema),
            [],
        )
        self.assertEqual(
            normalized["performer"]["positive_prompt"],
            value["performer"]["positive_prompt"],
        )

    def test_efficient_planner_is_registered_with_safe_widget_contract(self) -> None:
        self.assertIs(
            self.nodes.NODE_CLASS_MAPPINGS["DiffusionGemmaSplatStageEfficientPlanner"],
            self.nodes.DiffusionGemmaSplatStageEfficientPlanner,
        )
        required = self.nodes.DiffusionGemmaSplatStageEfficientPlanner.INPUT_TYPES()[
            "required"
        ]
        self.assertEqual(required["max_new_tokens"][1]["default"], 3072)
        self.assertGreaterEqual(required["max_new_tokens"][1]["min"], 1024)
        self.assertNotIn("duration_seconds", required)


class ExportedNodeSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()

    def test_export_map_has_model_specific_targets_plus_optional_and_experimental_nodes(self) -> None:
        expected = DIRECTOR_NODE_IDS | OPTIONAL_NODE_IDS | EXPERIMENTAL_NODE_IDS
        self.assertEqual(set(self.nodes.NODE_CLASS_MAPPINGS), expected)
        self.assertEqual(set(self.nodes.NODE_DISPLAY_NAME_MAPPINGS), expected)

    def test_experimental_planners_are_visibly_separate(self) -> None:
        for node_id in DIRECTOR_NODE_IDS:
            self.assertEqual(
                self.nodes.NODE_CLASS_MAPPINGS[node_id].CATEGORY,
                self.nodes.CATEGORY,
            )
        for node_id in EXPERIMENTAL_NODE_IDS:
            self.assertEqual(
                self.nodes.NODE_CLASS_MAPPINGS[node_id].CATEGORY,
                self.nodes.EXPERIMENTAL_MOTION_CATEGORY,
            )
            self.assertTrue(
                self.nodes.NODE_DISPLAY_NAME_MAPPINGS[node_id].endswith("(Experimental)")
            )
        for node_id in OPTIONAL_NODE_IDS:
            self.assertEqual(
                self.nodes.NODE_CLASS_MAPPINGS[node_id].CATEGORY,
                self.nodes.OPTIONAL_CATEGORY,
            )


if __name__ == "__main__":
    unittest.main()
