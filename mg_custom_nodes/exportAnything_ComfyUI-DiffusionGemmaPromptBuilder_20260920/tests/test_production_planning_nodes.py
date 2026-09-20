from __future__ import annotations

import json
import hashlib
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import production_planning_nodes as planning
import timed_lyrics_nodes as timed_lyrics


REF_PROMPT = """subject_definitions:
<Picture 1>: [reference] performer identity, wardrobe, monochrome style, and setting.
<Audio 1>: [reference] exact soundtrack rhythm, vocal phrasing, and dynamics.

summary:
A monochrome performance follows the locked song in two connected segments.

retention_analysis:
Retain <Picture 1> identity and <Audio 1> timing in every segment.

detailed_description:
Black-and-white live-action music-video photography with stable readable framing.
[Shot 1] A medium shot follows the performer through the opening phrase.
[Shot 2] At 00:12.500, a hard cut reveals a wider stage performance through the ending phrase.

overall_soundscape:
The supplied <Audio 1> reference remains the timing authority with no invented replacement song.

non_diegetic_music:
Use the supplied <Audio 1> composition as reference; final delivery preserves the locked master."""

ONE_SHOT_REF_PROMPT = REF_PROMPT.replace(
    "\n[Shot 2] At 00:12.500, a hard cut reveals a wider stage performance through the ending phrase.",
    "",
)

FOUR_SHOT_REF_PROMPT = ONE_SHOT_REF_PROMPT.replace(
    "[Shot 1] A medium shot follows the performer through the opening phrase.",
    """[Shot 1] A medium shot follows the performer through the opening phrase.
[Shot 2] At 00:03.500, the camera cuts to a stable side view as the dance phrase changes.
[Shot 3] At 00:07.250, the camera cuts to a locked wide view of the full stage.
[Shot 4] At 00:11.000, the camera cuts to a steady close view for the ending accent.""",
)

FOUR_SHOT_LONG_REF_PROMPT = REF_PROMPT.replace(
    "[Shot 1] A medium shot follows the performer through the opening phrase.\n[Shot 2] At 00:12.500, a hard cut reveals a wider stage performance through the ending phrase.",
    """[Shot 1] A medium shot follows the performer through the opening phrase.
[Shot 2] At 00:06.000, the camera cuts to a stable side view as the dance phrase changes.
[Shot 3] At 00:12.500, a hard cut reveals a wider stage performance through the next phrase.
[Shot 4] At 00:19.000, the camera cuts to a steady close view for the ending accent.""",
)

FIFTEEN_SHOT_LONG_REF_PROMPT = REF_PROMPT.replace(
    "[Shot 1] A medium shot follows the performer through the opening phrase.\n[Shot 2] At 00:12.500, a hard cut reveals a wider stage performance through the ending phrase.",
    "\n".join(
        ["[Shot 1] A medium shot follows the performer through the opening phrase."]
        + [
            f"[Shot {index}] At 00:{timestamp:06.3f}, a stable editorial angle continues the performance."
            for index, timestamp in enumerate(
                (1.75, 3.5, 5.25, 7.0, 8.75, 10.5, 12.5, 14.25, 16.0, 17.75, 19.5, 21.25, 23.0, 24.0),
                start=2,
            )
        ]
    ),
)

FIFTEEN_SHOT_60_REF_PROMPT = REF_PROMPT.replace(
    "[Shot 1] A medium shot follows the performer through the opening phrase.\n[Shot 2] At 00:12.500, a hard cut reveals a wider stage performance through the ending phrase.",
    "\n".join(
        ["[Shot 1] A medium shot follows the performer through the opening phrase."]
        + [
            f"[Shot {index}] At 00:{timestamp:06.3f}, a stable editorial angle continues the performance."
            for index, timestamp in enumerate(range(4, 60, 4), start=2)
        ]
    ),
)

DUAL_IDENTITY_REF_PROMPT = REF_PROMPT.replace(
    "<Picture 1>: [reference] performer identity, wardrobe, monochrome style, and setting.",
    "<Subject 1>: the same woman jointly defined by <Picture 1> and <Picture 2>, including her face, feminine appearance, hair, build, and wardrobe.\n"
    "<Picture 1>: primary identity, body, wardrobe, and target-composition authority for <Subject 1>.\n"
    "<Picture 2>: multi-panel identity evidence for the same <Subject 1> only; its layout, grid, seams, backgrounds, pose sequence, and panels as separate people do not transfer.",
).replace(
    "Retain <Picture 1> identity and <Audio 1> timing in every segment.",
    "<Subject 1>: fully_preserved - her face, feminine appearance, hair, build, and wardrobe remain unchanged in every segment.\n"
    "<Picture 1>: attribute_transfer - its primary identity evidence defines <Subject 1>.\n"
    "<Picture 2>: attribute_transfer - its complementary identity evidence defines the same <Subject 1>.\n"
    "<Audio 1>: reference - its timing guides every segment.",
)

DUAL_IDENTITY_60_REF_PROMPT = DUAL_IDENTITY_REF_PROMPT.replace(
    "[Shot 1] A medium shot follows the performer through the opening phrase.\n[Shot 2] At 00:12.500, a hard cut reveals a wider stage performance through the ending phrase.",
    "\n".join(
        ["[Shot 1] A medium shot follows the performer through the opening phrase."]
        + [
            f"[Shot {index}] At 00:{timestamp:06.3f}, a stable editorial angle continues the performance."
            for index, timestamp in enumerate(range(4, 60, 4), start=2)
        ]
    ),
)

DUAL_IDENTITY_MANIFEST = (
    "<Picture 1>: [dg:identity,appearance,object,color,composition] primary identity, body, wardrobe, and target-composition authority for the same woman.\n"
    "<Picture 2>: [dg:identity,appearance] multi-panel identity evidence for the same woman only; do not transfer its layout, grid, seams, backgrounds, or pose sequence, and its panels are not separate people.\n"
    "<Audio 1>: [dg:audio,rhythm] locked soundtrack timing reference."
)

DEFAULT_LYRICS = (
    "First exact line\nSecond exact line\nThird exact line\nFourth exact line"
)


def measured_report(
    duration: float,
    intervals: list[tuple[float, float]],
    *,
    start: float = 10.0,
    waveform_sha256: str = "a" * 64,
) -> str:
    return json.dumps(
        {
            "schema": "diffusiongemma.music_audition_report",
            "version": 1,
            "ready": True,
            "selected_audio_sha256": waveform_sha256,
            "settings": {"excerpt_duration_seconds": duration},
            "selected_excerpt": {
                "start_seconds": start,
                "duration_seconds": duration,
                "low_density_visual_recovery_intervals": [
                    {
                        "relative_start_seconds": start,
                        "relative_end_seconds": end,
                    }
                    for start, end in intervals
                ],
                "timeline": {"entries": []},
            },
        }
    )


def timed_lyrics_report(
    duration: float,
    events: list[dict[str, object]],
    *,
    start: float = 0.0,
    lyrics: str = DEFAULT_LYRICS,
    waveform_sha256: str = "a" * 64,
    vocal_intervals: list[tuple[float, float]] | None = None,
    instrumental_intervals: list[tuple[float, float]] | None = None,
    timing_ready: bool = True,
    analysis_status: str | None = None,
    minimum_alignment_confidence: float = 0.70,
) -> str:
    return json.dumps(
        {
            "schema": "diffusiongemma.timed_lyrics_report",
            "version": 1,
            "analysis_status": analysis_status
            or ("timing_ready" if timing_ready else "natural_fallback"),
            "timing_ready": timing_ready,
            "master_audio_sha256": waveform_sha256,
            "lyrics_sha256": hashlib.sha256(lyrics.encode("utf-8")).hexdigest(),
            "minimum_alignment_confidence": minimum_alignment_confidence,
            "excerpt": {
                "start_seconds": start,
                "duration_seconds": duration,
                "end_seconds": start + duration,
            },
            "events": events,
            "vocal_intervals": [
                {"start_seconds": item_start, "end_seconds": item_end}
                for item_start, item_end in (vocal_intervals or [])
            ],
            "instrumental_intervals": [
                {"start_seconds": item_start, "end_seconds": item_end}
                for item_start, item_end in (instrumental_intervals or [])
            ],
            "warnings": [],
        }
    )


class ProductionPlanningNodeTests(unittest.TestCase):
    def project(self, duration: float = 25.0, start: float = 10.0):
        return planning.DiffusionGemmaProjectMasterContract().build(
            "Create a black-and-white rap performance.",
            duration,
            "9:16",
            "a" * 64,
            DEFAULT_LYRICS,
            start,
            duration,
            "MiniMax H3 Ref2VA",
            '{"primary":"9:16","adaptations":["16:9","1:1"]}',
            15.0,
        )

    def test_project_contract_is_deterministic_and_recommends_h3_count(self):
        first = self.project(25.0)
        second = self.project(25.0)
        manifest = json.loads(first[0])
        self.assertEqual(first, second)
        self.assertEqual(first[2], "9:16")
        self.assertEqual(first[3], 2)
        self.assertEqual(manifest["project_id"], first[1])
        self.assertEqual(manifest["audio_lock"]["waveform_sha256"], "a" * 64)
        self.assertTrue(first[7])
        self.assertEqual(first[8], 15.0)

    def test_project_rejects_mismatched_excerpt_clock(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            planning.DiffusionGemmaProjectMasterContract().build(
                "brief", 25.0, "9:16", "a" * 64, "lyrics", 0, 20, "H3", "{}", 15
            )

    def test_project_empty_audio_lock_points_to_music_qc(self):
        with self.assertRaisesRegex(ValueError, "Check MUSIC QC STATUS"):
            planning.DiffusionGemmaProjectMasterContract().build(
                "brief", 25.0, "9:16", "", "lyrics", 0, 25, "H3", "{}", 15
            )

    def test_one_shot_plan_preserves_native_sections(self):
        project = self.project(15.0, 20.0)[0]
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(15.0, [(0.0, 15.0)], start=20.0),
            ONE_SHOT_REF_PROMPT,
            "Lyrics + lip sync",
            20.0,
            15.0,
            12.0,
            5.0,
            15.0,
            project,
            "First exact line\nSecond exact line",
        )
        self.assertEqual(result[3], 1)
        self.assertEqual(result[7], 362)
        self.assertTrue(result[8])
        self.assertEqual(result[4].count("subject_definitions:"), 1)
        self.assertEqual(result[4].count("[Shot 1]"), 1)
        self.assertNotIn("[Shot 2]", result[4])

    def test_one_generation_lane_preserves_four_native_h3_shots(self):
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(15.0, [], start=0.0),
            FOUR_SHOT_REF_PROMPT,
            "Dance / music sync",
            0.0,
            15.0,
            12.0,
            5.0,
            15.0,
            self.project(15.0, 0.0)[0],
        )
        plan = json.loads(result[0])
        self.assertEqual(result[3], 1)
        self.assertEqual(result[4].count("[Shot "), 4)
        self.assertIn("[Shot 2] At 00:03.500", result[4])
        self.assertIn("[Shot 3] At 00:07.250", result[4])
        self.assertIn("[Shot 4] At 00:11.000", result[4])
        self.assertEqual(plan["effective_generation_lane_count"], 1)
        self.assertEqual(plan["source_native_shot_count"], 4)
        self.assertEqual(plan["shots"][0]["native_shot_count"], 4)

    def test_multi_shot_timed_lyrics_are_audio_gated_segment_wide_hints(self):
        timing = timed_lyrics_report(
            15.0,
            [
                {
                    "start_seconds": 1.0,
                    "end_seconds": 3.0,
                    "authored_lines": ["First exact line"],
                    "transcript": "First exact line",
                    "confidence": 0.96,
                },
                {
                    "start_seconds": 5.0,
                    "end_seconds": 7.0,
                    "authored_lines": ["Second exact line"],
                    "transcript": "Second exact line",
                    "confidence": 0.94,
                },
            ],
            vocal_intervals=[(1.0, 7.0)],
        )
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(15.0, [], start=0.0),
            FOUR_SHOT_REF_PROMPT,
            "Lyrics + lip sync",
            0.0,
            15.0,
            12.0,
            5.0,
            15.0,
            self.project(15.0, 0.0)[0],
            DEFAULT_LYRICS,
            timed_lyrics_report_json=timing,
        )
        prompt = result[4]
        scope = "Across all 4 native shots in this generated segment"
        self.assertIn(scope, prompt)
        self.assertLess(prompt.index(scope), prompt.index("[Shot 1]"))
        self.assertEqual(prompt.count('"First exact line"'), 1)
        self.assertEqual(prompt.count('"Second exact line"'), 1)
        self.assertIn("audio-gated lexical hints only", prompt)
        self.assertIn("not a timing schedule", prompt)
        self.assertNotIn("sing these exact lines once", prompt)

    def test_multiple_native_shots_are_partitioned_across_duration_lanes(self):
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(25.0, [(12.0, 13.0)], start=0.0),
            FOUR_SHOT_LONG_REF_PROMPT,
            "Dance / music sync",
            0.0,
            25.0,
            12.0,
            5.0,
            15.0,
            self.project(25.0, 0.0)[0],
        )
        plan = json.loads(result[0])
        self.assertEqual(result[3], 2)
        self.assertEqual(result[4].count("[Shot "), 2)
        self.assertIn("[Shot 2] At 00:06.000", result[4])
        self.assertEqual(result[9].count("[Shot "), 2)
        self.assertIn("[Shot 2] At 00:06.500", result[9])
        self.assertNotIn("[Shot 3]", result[9])
        self.assertEqual(
            [lane["native_shot_count"] for lane in plan["shots"]],
            [2, 2],
        )

    def test_fifteen_native_shots_over_25_seconds_use_two_generation_lanes(self):
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(25.0, [(12.0, 13.0)], start=0.0),
            FIFTEEN_SHOT_LONG_REF_PROMPT,
            "Dance / music sync",
            0.0,
            25.0,
            12.0,
            5.0,
            15.0,
            self.project(25.0, 0.0)[0],
        )
        plan = json.loads(result[0])
        self.assertEqual(result[3], 2)
        self.assertEqual(plan["effective_generation_lane_count"], 2)
        self.assertEqual(plan["source_native_shot_count"], 15)
        self.assertEqual(
            sum(lane["native_shot_count"] for lane in plan["shots"]),
            15,
        )
        self.assertEqual(
            [lane["native_shot_count"] for lane in plan["shots"]],
            [7, 8],
        )
        self.assertTrue(
            all(lane["duration_seconds"] <= 15.0 for lane in plan["shots"])
        )
        self.assertTrue(result[8])
        self.assertTrue(result[13])
        self.assertFalse(result[18])
        self.assertFalse(result[23])

    def test_stale_preferred_lane_hint_is_limited_by_project_master_ceiling(self):
        project = planning.DiffusionGemmaProjectMasterContract().build(
            "Create a black-and-white rap performance.",
            25.0,
            "9:16",
            "a" * 64,
            "",
            0.0,
            25.0,
            "MiniMax H3 Ref2VA",
            "{}",
            7.0,
        )[0]
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(
                25.0,
                [(6.9, 7.1), (12.4, 12.6), (19.4, 19.6)],
                start=0.0,
            ),
            FIFTEEN_SHOT_LONG_REF_PROMPT,
            "Dance / music sync",
            0.0,
            25.0,
            12.0,
            5.0,
            7.0,
            project,
        )
        plan = json.loads(result[0])
        policy = plan["generation_lane_duration_policy"]
        self.assertTrue(result[2])
        self.assertEqual(result[3], 4)
        self.assertEqual(policy["requested_preferred_seconds"], 12.0)
        self.assertEqual(policy["effective_preferred_seconds"], 7.0)
        self.assertTrue(policy["preferred_was_normalized"])
        self.assertEqual(policy["maximum_source"], "project_master_contract")
        self.assertIn("limited from 12s to 7s", result[1])

    def test_in_range_preferred_lane_hint_is_recorded_unchanged(self):
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(15.0, [], start=0.0),
            ONE_SHOT_REF_PROMPT,
            "Dance / music sync",
            0.0,
            15.0,
            12.0,
            5.0,
            15.0,
            self.project(15.0, 0.0)[0],
        )
        policy = json.loads(result[0])["generation_lane_duration_policy"]
        self.assertEqual(policy["requested_preferred_seconds"], 12.0)
        self.assertEqual(policy["effective_preferred_seconds"], 12.0)
        self.assertFalse(policy["preferred_was_normalized"])
        self.assertFalse(policy["preferred_realized"])
        self.assertEqual(policy["realized_lane_seconds"], [15.0])
        self.assertIn("cannot be realized", result[1])

    def test_hard_lane_constraints_remain_fail_closed(self):
        project = self.project(15.0, 0.0)[0]
        common = (
            measured_report(15.0, [], start=0.0),
            ONE_SHOT_REF_PROMPT,
            "Dance / music sync",
            0.0,
            15.0,
            12.0,
        )
        with self.assertRaisesRegex(ValueError, "min_shot_seconds <= max_shot_seconds"):
            planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
                *common, 16.0, 15.0, project
            )
        with self.assertRaisesRegex(ValueError, "Project Master Contract"):
            planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
                *common, 5.0, 14.0, project
            )
        nonfinite_project = json.loads(project)
        nonfinite_project["h3_segmentation"]["maximum_shot_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite positive number"):
            planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
                *common, 5.0, 15.0, json.dumps(nonfinite_project)
            )
        malformed_project = json.loads(project)
        malformed_project["h3_segmentation"] = []
        with self.assertRaisesRegex(ValueError, "h3_segmentation must be a JSON object"):
            planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
                *common, 5.0, 15.0, json.dumps(malformed_project)
            )

    def test_two_shot_plan_uses_recovery_cut_and_measured_lyric_events(self):
        project = self.project(25.0, 30.0)[0]
        timing = timed_lyrics_report(
            25.0,
            [
                {
                    "start_seconds": 1.0,
                    "end_seconds": 4.0,
                    "authored_lines": ["First exact line", "Second exact line"],
                    "transcript": "First exact line Second exact line",
                    "confidence": 0.95,
                },
                {
                    "start_seconds": 14.0,
                    "end_seconds": 18.0,
                    "authored_lines": ["Third exact line", "Fourth exact line"],
                    "transcript": "Third exact line Fourth exact line",
                    "confidence": 0.93,
                },
            ],
            start=30.0,
            vocal_intervals=[(1.0, 4.0), (14.0, 18.0)],
        )
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(25.0, [(10.0, 15.0)], start=30.0),
            REF_PROMPT,
            "Lyrics + lip sync",
            30.0,
            25.0,
            12.0,
            5.0,
            15.0,
            project,
            DEFAULT_LYRICS,
            timed_lyrics_report_json=timing,
        )
        plan = json.loads(result[0])
        self.assertEqual(result[3], 2)
        self.assertAlmostEqual(result[5], 0.0)
        self.assertAlmostEqual(result[10], 12.5)
        self.assertAlmostEqual(result[6] + result[11], 25.0)
        self.assertAlmostEqual(plan["shots"][1]["absolute_song_start_seconds"], 42.5)
        self.assertIn("First exact line", result[4])
        self.assertNotIn("Third exact line", result[4])
        self.assertIn("Third exact line", result[9])
        self.assertNotIn("First exact line", result[9])
        self.assertIn("<Picture 2>", result[9])
        self.assertIn("<Picture 1>", result[9])
        self.assertEqual(result[7] % 17, 5)
        self.assertEqual(result[12] % 17, 5)
        self.assertNotRegex(result[9].lower(), r"\b(?:hard\s+cut|camera\s+cuts?\s+to)\b")
        self.assertIn("The shot opens on a wider stage", result[9])

    def test_dual_identity_policy_moves_later_lane_relay_to_picture_3(self):
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(25.0, [(10.0, 15.0)], start=0.0),
            DUAL_IDENTITY_REF_PROMPT,
            "Dance / music sync",
            0.0,
            25.0,
            12.0,
            5.0,
            15.0,
            self.project(25.0, 0.0)[0],
            "",
            2,
            "Previous lane tail",
            DUAL_IDENTITY_MANIFEST,
        )
        plan = json.loads(result[0])
        policy = plan["reference_policy"]
        self.assertEqual(policy["schema"], planning.IDENTITY_RELAY_POLICY_SCHEMA)
        self.assertEqual(policy["identity_picture_tags"], ["<Picture 1>", "<Picture 2>"])
        self.assertEqual(policy["identity_subject_tag"], "<Subject 1>")
        self.assertEqual(policy["subject_binding_validation"], "strict")
        self.assertEqual(policy["contract_repairs"], [])
        self.assertEqual(policy["continuity_picture_tag"], "<Picture 3>")
        self.assertEqual(policy["manifest_validation"], "strict")
        self.assertNotIn("<Picture 3>", result[4])
        self.assertIn("<Picture 3>", result[9])
        self.assertIn("multi-panel identity evidence", result[4])
        self.assertIn("Do not reproduce its panel layout", result[4])
        self.assertIn("not an identity authority", result[9])
        self.assertIn("body, wardrobe, and target-composition authority", result[9])
        self.assertEqual(
            policy["identity_picture_roles"]["<Picture 2>"],
            "same_subject_multi_panel_identity_evidence_only",
        )
        self.assertEqual(
            plan["shots"][0]["reference_picture_tags"],
            ["<Picture 1>", "<Picture 2>"],
        )
        self.assertEqual(
            plan["shots"][1]["reference_picture_tags"],
            ["<Picture 1>", "<Picture 2>", "<Picture 3>"],
        )
        self.assertFalse(plan["shots"][0]["continuity_reference_required"])
        self.assertTrue(plan["shots"][1]["continuity_reference_required"])

    def test_dual_identity_policy_repairs_director_binding_and_retention_drift(self):
        split_binding = DUAL_IDENTITY_REF_PROMPT.replace(
            "<Subject 1>: the same woman jointly defined by <Picture 1> and <Picture 2>, including her face, feminine appearance, hair, build, and wardrobe.",
            "<Subject 1>: the same woman defined by <Picture 1>, including her face, feminine appearance, hair, build, and wardrobe.",
        ).replace(
            "<Subject 1>: fully_preserved - her face, feminine appearance, hair, build, and wardrobe remain unchanged in every segment.",
            "<Subject 1>: attribute_transfer - her face, hair, body type, and wardrobe are transferred from",
        ).replace(
            "<Picture 2>: multi-panel identity evidence for the same <Subject 1> only; its layout, grid, seams, backgrounds, pose sequence, and panels as separate people do not transfer.",
            "<Picture 2>: multi-panel identity evidence for the same woman only; do not transfer its layout, grid, seams, backgrounds, or pose sequence, and its panels are not separate people.",
        )
        wrong_retention_only = DUAL_IDENTITY_REF_PROMPT.replace(
            "<Subject 1>: fully_preserved - her face, feminine appearance, hair, build, and wardrobe remain unchanged in every segment.",
            "<Subject 1>: attribute_transfer - her face, hair, body type, and wardrobe are transferred from",
        )
        is_delimited_binding = split_binding.replace(
            "<Picture 2>:",
            "<Picture 2> is",
        )

        for prompt, expected_repairs in (
            (
                split_binding,
                [
                    "joint_identity_subject_binding",
                    "fully_preserved_identity_retention",
                ],
            ),
            (wrong_retention_only, ["fully_preserved_identity_retention"]),
            (
                is_delimited_binding,
                [
                    "joint_identity_subject_binding",
                    "fully_preserved_identity_retention",
                ],
            ),
        ):
            with self.subTest(expected_repairs=expected_repairs):
                result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
                    measured_report(25.0, [(10.0, 15.0)], start=0.0),
                    prompt,
                    "Dance / music sync",
                    0.0,
                    25.0,
                    12.0,
                    5.0,
                    15.0,
                    self.project(25.0, 0.0)[0],
                    "",
                    2,
                    "Off",
                    DUAL_IDENTITY_MANIFEST,
                )
                plan = json.loads(result[0])
                policy = plan["reference_policy"]
                self.assertEqual(policy["identity_subject_tag"], "<Subject 1>")
                self.assertEqual(policy["subject_binding_validation"], "host_normalized")
                self.assertEqual(policy["contract_repairs"], expected_repairs)
                if "joint_identity_subject_binding" in expected_repairs:
                    self.assertIn(
                        "The same performer is jointly identified by <Picture 1> and <Picture 2>",
                        result[4],
                    )
                self.assertIn(
                    "<Subject 1>: fully_preserved - Preserve the same identity, face, gender presentation",
                    result[4],
                )

    def test_relay_off_omits_continuity_tag_and_does_not_request_lazy_tail(self):
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(25.0, [(10.0, 15.0)], start=0.0),
            DUAL_IDENTITY_REF_PROMPT,
            "Dance / music sync",
            0.0,
            25.0,
            12.0,
            5.0,
            15.0,
            self.project(25.0, 0.0)[0],
            "",
            2,
            "Off",
            DUAL_IDENTITY_MANIFEST,
        )
        plan = json.loads(result[0])
        self.assertEqual(plan["reference_policy"]["continuity_relay_mode"], "off")
        self.assertEqual(plan["reference_policy"]["continuity_picture_tag"], "")
        self.assertNotIn("<Picture 3>", result[4])
        self.assertNotIn("<Picture 3>", result[9])
        self.assertFalse(plan["shots"][1]["continuity_reference_required"])

        gate = planning.DiffusionGemmaH3RelayReferenceGate()
        self.assertEqual(gate.check_lazy_status(result[0], 2, None), [])
        routed = gate.route(result[0], 2, None)
        self.assertIsNone(routed[0])
        self.assertEqual(routed[1], "")
        self.assertFalse(routed[2])
        self.assertIn("independently", routed[3])

    def test_relay_gate_lazily_routes_one_rgb_tail_using_policy_ordinal(self):
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(25.0, [(10.0, 15.0)], start=0.0),
            DUAL_IDENTITY_REF_PROMPT,
            "Dance / music sync",
            0.0,
            25.0,
            12.0,
            5.0,
            15.0,
            self.project(25.0, 0.0)[0],
            "",
            2,
            "Previous lane tail",
            DUAL_IDENTITY_MANIFEST,
        )
        gate = planning.DiffusionGemmaH3RelayReferenceGate()
        self.assertEqual(gate.check_lazy_status(result[0], 2, None), ["relay_image"])
        tail = torch.rand((1, 768, 1376, 3), dtype=torch.float32)
        routed = gate.route(result[0], 2, tail)
        self.assertIs(routed[0], tail)
        self.assertEqual(routed[1], "<Picture 3>")
        self.assertTrue(routed[2])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            gate.route(result[0], 2, torch.rand((2, 64, 64, 3)))

    def test_dual_identity_manifest_and_subject_binding_fail_closed(self):
        common = (
            measured_report(25.0, [(10.0, 15.0)], start=0.0),
            DUAL_IDENTITY_REF_PROMPT,
            "Dance / music sync",
            0.0,
            25.0,
            12.0,
            5.0,
            15.0,
            self.project(25.0, 0.0)[0],
            "",
            2,
            "Previous lane tail",
        )
        planner = planning.DiffusionGemmaAudioAwareMultiShotPlanner()
        with self.assertRaisesRegex(ValueError, "require reference_manifest"):
            planner.plan(*common, "")
        bad_role = DUAL_IDENTITY_MANIFEST.replace(
            "[dg:identity,appearance] multi-panel",
            "[dg:environment,style] multi-panel",
        )
        with self.assertRaisesRegex(ValueError, "Picture 2.*identity,appearance"):
            planner.plan(*common, bad_role)
        split_subject = DUAL_IDENTITY_REF_PROMPT.replace(
            "<Subject 1>: the same woman jointly defined by <Picture 1> and <Picture 2>, including her face, feminine appearance, hair, build, and wardrobe.",
            "<Subject 1>: woman A defined only by <Picture 1>.\n<Subject 2>: woman B defined only by <Picture 2>.",
        ).replace(
            "<Picture 2>: multi-panel identity evidence for the same <Subject 1> only",
            "<Picture 2>: multi-panel identity evidence for <Subject 2> only",
        )
        with self.assertRaisesRegex(ValueError, "conflicting or ambiguous"):
            planner.plan(
                common[0],
                split_subject,
                *common[2:],
                DUAL_IDENTITY_MANIFEST,
            )
        different_identity = DUAL_IDENTITY_REF_PROMPT.replace(
            "multi-panel identity evidence for the same <Subject 1> only",
            "multi-panel identity evidence for a different woman",
        )
        with self.assertRaisesRegex(ValueError, "different identity"):
            planner.plan(
                common[0],
                different_identity,
                *common[2:],
                DUAL_IDENTITY_MANIFEST,
            )
        picture_2_only = DUAL_IDENTITY_REF_PROMPT.replace(
            "<Subject 1>: the same woman jointly defined by <Picture 1> and <Picture 2>, including her face, feminine appearance, hair, build, and wardrobe.",
            "<Subject 1>: the woman defined only by <Picture 2>.",
        ).replace(
            "<Picture 1>: primary identity, body, wardrobe, and target-composition authority for <Subject 1>.",
            "<Picture 1>: primary identity, body, wardrobe, and target-composition authority for the same woman.",
        ).replace(
            "the same <Subject 1> only",
            "the same woman only",
        )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            planner.plan(
                common[0],
                picture_2_only,
                *common[2:],
                DUAL_IDENTITY_MANIFEST,
            )
        ambiguous_picture_2 = DUAL_IDENTITY_REF_PROMPT.replace(
            "<Subject 1>: the same woman jointly defined by <Picture 1> and <Picture 2>, including her face, feminine appearance, hair, build, and wardrobe.",
            "<Subject 1>: the woman defined only by <Picture 1>.",
        ).replace(
            "<Picture 2>: multi-panel identity evidence for the same <Subject 1> only; its layout, grid, seams, backgrounds, pose sequence, and panels as separate people do not transfer.",
            "<Picture 2>: multi-panel portrait reference with no declared identity relationship.",
        )
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            planner.plan(
                common[0],
                ambiguous_picture_2,
                *common[2:],
                DUAL_IDENTITY_MANIFEST,
            )
        weak_retention = DUAL_IDENTITY_REF_PROMPT.replace(
            "<Subject 1>: fully_preserved -",
            "<Subject 1>: weak_reference -",
        )
        with self.assertRaisesRegex(ValueError, "must be fully_preserved"):
            planner.plan(
                common[0],
                weak_retention,
                *common[2:],
                DUAL_IDENTITY_MANIFEST,
            )
        partial_retention = DUAL_IDENTITY_REF_PROMPT.replace(
            "<Subject 1>: fully_preserved -",
            "<Subject 1>: partially_preserved -",
        )
        with self.assertRaisesRegex(ValueError, "must be fully_preserved"):
            planner.plan(
                common[0],
                partial_retention,
                *common[2:],
                DUAL_IDENTITY_MANIFEST,
            )
        missing_retention = DUAL_IDENTITY_REF_PROMPT.replace(
            "<Subject 1>: fully_preserved - her face, feminine appearance, hair, build, and wardrobe remain unchanged in every segment.\n",
            "",
        )
        with self.assertRaisesRegex(ValueError, "exactly one retention row"):
            planner.plan(
                common[0],
                missing_retention,
                *common[2:],
                DUAL_IDENTITY_MANIFEST,
            )
        duplicate_retention = DUAL_IDENTITY_REF_PROMPT.replace(
            "<Subject 1>: fully_preserved - her face, feminine appearance, hair, build, and wardrobe remain unchanged in every segment.",
            "<Subject 1>: fully_preserved - her face, feminine appearance, hair, build, and wardrobe remain unchanged in every segment.\n<Subject 1>: attribute_transfer - duplicate conflicting row.",
        )
        with self.assertRaisesRegex(ValueError, "exactly one retention row"):
            planner.plan(
                common[0],
                duplicate_retention,
                *common[2:],
                DUAL_IDENTITY_MANIFEST,
            )
        malformed_duplicate_retention = DUAL_IDENTITY_REF_PROMPT.replace(
            "<Subject 1>: fully_preserved - her face, feminine appearance, hair, build, and wardrobe remain unchanged in every segment.",
            "<Subject 1>: fully_preserved - her face, feminine appearance, hair, build, and wardrobe remain unchanged in every segment.\n<Subject 1>: weak_reference",
        )
        with self.assertRaisesRegex(ValueError, "exactly one retention row"):
            planner.plan(
                common[0],
                malformed_duplicate_retention,
                *common[2:],
                DUAL_IDENTITY_MANIFEST,
            )
        duplicate_picture_2 = DUAL_IDENTITY_REF_PROMPT.replace(
            "<Audio 1>: [reference] exact soundtrack rhythm, vocal phrasing, and dynamics.",
            "<Picture 2>: duplicate identity row for the same <Subject 1>.\n<Audio 1>: [reference] exact soundtrack rhythm, vocal phrasing, and dynamics.",
        )
        with self.assertRaisesRegex(ValueError, "Duplicate identity Picture"):
            planner.plan(
                common[0],
                duplicate_picture_2,
                *common[2:],
                DUAL_IDENTITY_MANIFEST,
            )
        malformed_duplicate_picture_2 = DUAL_IDENTITY_REF_PROMPT.replace(
            "<Audio 1>: [reference] exact soundtrack rhythm, vocal phrasing, and dynamics.",
            "<Picture 2> malformed duplicate row.\n<Audio 1>: [reference] exact soundtrack rhythm, vocal phrasing, and dynamics.",
        )
        with self.assertRaisesRegex(ValueError, "Duplicate identity Picture"):
            planner.plan(
                common[0],
                malformed_duplicate_picture_2,
                *common[2:],
                DUAL_IDENTITY_MANIFEST,
            )
        negated_distinct = DUAL_IDENTITY_REF_PROMPT.replace(
            "same <Subject 1> only;",
            "same <Subject 1> only and not a different person;",
        )
        negated_result = planner.plan(
            common[0],
            negated_distinct,
            *common[2:],
            DUAL_IDENTITY_MANIFEST,
        )
        self.assertTrue(negated_result[2])
        cannot_be_distinct = DUAL_IDENTITY_REF_PROMPT.replace(
            "same <Subject 1> only;",
            "same <Subject 1> only and cannot be interpreted as a different person;",
        )
        cannot_result = planner.plan(
            common[0],
            cannot_be_distinct,
            *common[2:],
            DUAL_IDENTITY_MANIFEST,
        )
        self.assertTrue(cannot_result[2])
        additive_identity = DUAL_IDENTITY_REF_PROMPT.replace(
            "same <Subject 1> only;",
            "same <Subject 1> plus an additional woman;",
        )
        with self.assertRaisesRegex(ValueError, "different identity"):
            planner.plan(
                common[0],
                additive_identity,
                *common[2:],
                DUAL_IDENTITY_MANIFEST,
            )
        conflicting_picture = DUAL_IDENTITY_REF_PROMPT.replace(
            "<Audio 1>: [reference] exact soundtrack rhythm, vocal phrasing, and dynamics.",
            "<Picture 3>: unrelated environment plate.\n<Audio 1>: [reference] exact soundtrack rhythm, vocal phrasing, and dynamics.",
        )
        with self.assertRaisesRegex(ValueError, "outside the identity-anchor contract"):
            planner.plan(
                common[0],
                conflicting_picture,
                *common[2:],
                DUAL_IDENTITY_MANIFEST,
            )

    def test_lyrics_mode_marks_a_confirmed_instrumental_lane_as_non_vocal(self):
        timing = timed_lyrics_report(
            25.0,
            [
                {
                    "start_seconds": 1.0,
                    "end_seconds": 4.0,
                    "authored_lines": ["First exact line"],
                    "transcript": "First exact line",
                    "confidence": 0.97,
                }
            ],
            vocal_intervals=[(1.0, 4.0)],
            instrumental_intervals=[(12.5, 25.0)],
        )
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(25.0, [(10.0, 15.0)], start=0.0),
            REF_PROMPT,
            "Lyrics + lip sync",
            0,
            25,
            12,
            5,
            15,
            self.project(25.0, 0.0)[0],
            DEFAULT_LYRICS,
            timed_lyrics_report_json=timing,
        )
        plan = json.loads(result[0])
        empty_lanes = [shot for shot in plan["shots"] if not shot["lyrics"]]
        self.assertEqual(len(empty_lanes), 1)
        self.assertEqual(empty_lanes[0]["lyrics_timing_state"], "confirmed_instrumental")
        self.assertIn("entire generated segment is instrumental", empty_lanes[0]["prompt"])
        self.assertIn("No visible person sings", empty_lanes[0]["prompt"])

    def test_multishot_plan_uses_native_fallback_without_recovery_evidence(self):
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(25.0, [], start=0.0),
            REF_PROMPT,
            "Dance / music sync",
            0,
            25,
            12,
            5,
            15,
            self.project(25.0, 0.0)[0],
        )
        plan = json.loads(result[0])
        self.assertTrue(result[2])
        self.assertEqual(
            plan["generation_lane_boundary_policy"]["strategy"],
            "native_only_fallback",
        )
        self.assertEqual([item["seconds"] for item in plan["inter_lane_cuts"]], [12.5])
        self.assertTrue(plan["inter_lane_cuts"][0]["native_shot_boundary"])
        self.assertFalse(
            plan["inter_lane_cuts"][0]["measured_low_density_recovery"]
        )
        self.assertNotIn("every inter-lane cut is both", result[1])

    def test_mixed_boundary_fallback_carries_the_active_native_shot(self):
        prompt = ONE_SHOT_REF_PROMPT.replace(
            "[Shot 1] A medium shot follows the performer through the opening phrase.",
            "\n".join(
                (
                    "[Shot 1] A medium shot follows the performer through the opening phrase.",
                    "[Shot 2] At 00:12.000, a stable side view continues the performance.",
                    "[Shot 3] At 00:28.000, a locked wide view carries the ending phrase.",
                )
            ),
        )
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(40.0, [(24.9, 25.1)], start=0.0),
            prompt,
            "Natural / audio-led sync",
            0,
            40,
            12,
            5,
            15,
            self.project(40.0, 0.0)[0],
        )
        plan = json.loads(result[0])
        self.assertEqual(
            plan["generation_lane_boundary_policy"]["strategy"],
            "mixed_native_or_recovery_fallback",
        )
        self.assertEqual([item["seconds"] for item in plan["inter_lane_cuts"]], [12.0, 25.1])
        self.assertEqual(
            [item["selection_origin"] for item in plan["inter_lane_cuts"]],
            ["native", "measured_recovery"],
        )
        self.assertEqual(
            [shot["source_shot_continuation_at_lane_start"] for shot in plan["shots"]],
            [False, False, True],
        )
        self.assertIn("[Shot 2] At 00:02.900", plan["shots"][2]["prompt"])

    def test_exact_60_second_capacity_grid_uses_truthful_balanced_fallback(self):
        project = planning.DiffusionGemmaProjectMasterContract().build(
            "Create a black-and-white rap performance.",
            60.0,
            "9:16",
            "a" * 64,
            "",
            2.0,
            60.0,
            "MiniMax H3 Ref2VA",
            "{}",
            15.0,
        )[0]
        arguments = (
            measured_report(
                60.0,
                [(0.0, 14.0), (20.0, 48.0), (50.0, 58.0)],
                start=2.0,
            ),
            DUAL_IDENTITY_60_REF_PROMPT,
            "Natural / audio-led sync",
            2.0,
            60.0,
            7.0,
            5.0,
            15.0,
            project,
            "",
            2,
            "Off",
            DUAL_IDENTITY_MANIFEST,
        )
        first = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(*arguments)
        second = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(*arguments)
        self.assertEqual(first, second)
        plan = json.loads(first[0])
        policy = plan["generation_lane_boundary_policy"]
        duration_policy = plan["generation_lane_duration_policy"]
        self.assertTrue(first[2])
        self.assertEqual(first[3], 4)
        self.assertEqual(plan["boundary_selection_revision"], 2)
        self.assertEqual(policy["strategy"], "duration_balanced_fallback")
        self.assertTrue(policy["fallback_used"])
        self.assertTrue(policy["capacity_saturated"])
        self.assertEqual(
            [item["seconds"] for item in plan["inter_lane_cuts"]],
            [15.0, 30.0, 45.0],
        )
        self.assertEqual(
            [item["selection_origin"] for item in plan["inter_lane_cuts"]],
            ["duration_balanced"] * 3,
        )
        self.assertEqual(
            [item["native_shot_boundary"] for item in plan["inter_lane_cuts"]],
            [False, False, False],
        )
        self.assertEqual(
            [item["measured_low_density_recovery"] for item in plan["inter_lane_cuts"]],
            [False, True, True],
        )
        self.assertEqual([shot["duration_seconds"] for shot in plan["shots"]], [15.0] * 4)
        self.assertEqual([shot["generated_h3_frames"] for shot in plan["shots"]], [362] * 4)
        self.assertEqual([shot["retained_master_frames"] for shot in plan["shots"]], [360] * 4)
        self.assertEqual(sum(shot["retained_master_frames"] for shot in plan["shots"]), 1440)
        self.assertEqual(
            [shot["absolute_song_start_seconds"] for shot in plan["shots"]],
            [2.0, 17.0, 32.0, 47.0],
        )
        self.assertEqual(plan["source_native_shot_count"], 15)
        self.assertEqual([shot["native_shot_count"] for shot in plan["shots"]], [4, 4, 4, 3])
        self.assertEqual(
            [shot["rendered_source_fragment_count"] for shot in plan["shots"]],
            [4, 5, 5, 4],
        )
        self.assertEqual(
            [shot["source_shot_continuation_at_lane_start"] for shot in plan["shots"]],
            [False, True, True, True],
        )
        self.assertIn("[Shot 2] At 00:01.000", plan["shots"][1]["prompt"])
        self.assertIn("[Shot 2] At 00:02.000", plan["shots"][2]["prompt"])
        self.assertIn("[Shot 2] At 00:03.000", plan["shots"][3]["prompt"])
        self.assertTrue(
            all(
                shot["reference_picture_tags"] == ["<Picture 1>", "<Picture 2>"]
                for shot in plan["shots"]
            )
        )
        self.assertTrue(all("<Picture 3>" not in shot["prompt"] for shot in plan["shots"]))
        self.assertEqual(duration_policy["realized_lane_seconds"], [15.0] * 4)
        self.assertEqual(duration_policy["realized_mean_seconds"], 15.0)
        self.assertFalse(duration_policy["preferred_realized"])
        self.assertIn("0/3 native and 2/3 measured-recovery", first[1])
        self.assertIn("7s preference cannot be realized", first[1])
        self.assertNotIn("every inter-lane cut is both", first[1])

    def test_balanced_carry_slices_embedded_cues_without_replaying_or_losing_them(self):
        prompt = ONE_SHOT_REF_PROMPT.replace(
            "[Shot 1] A medium shot follows the performer through the opening phrase.",
            "\n".join(
                (
                    "[Shot 1] A medium shot follows the opening.",
                    "[Shot 2] At 00:10.000, a stable side view continues. "
                    "At 00:12.000, she raises one hand. At 00:20.000, she turns.",
                    "[Shot 3] At 00:25.000, a locked wide view ends the phrase.",
                )
            ),
        )
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(30.0, [], start=0.0),
            prompt,
            "Natural / audio-led sync",
            0,
            30,
            12,
            5,
            15,
            self.project(30.0, 0.0)[0],
        )
        plan = json.loads(result[0])
        self.assertTrue(result[2])
        self.assertEqual([item["seconds"] for item in plan["inter_lane_cuts"]], [15.0])
        self.assertEqual(
            plan["generation_lane_boundary_policy"]["strategy"],
            "duration_balanced_fallback",
        )
        self.assertTrue(plan["shots"][1]["source_shot_continuation_at_lane_start"])
        self.assertEqual(plan["shots"][1]["carry_in_source_native_shot_index"], 2)
        self.assertIn("Opening-state history from before this generation lane", result[9])
        self.assertIn("she raises one hand", result[9])
        self.assertIn("At 00:05.000, she turns", result[9])
        self.assertIn("[Shot 2] At 00:10.000", result[9])
        self.assertNotIn("At 00:20.000", result[4])
        self.assertNotIn("she turns", result[4])

    def test_native_boundary_identity_is_exact_while_recovery_tolerance_remains_measured(self):
        prompt = ONE_SHOT_REF_PROMPT.replace(
            "[Shot 1] A medium shot follows the performer through the opening phrase.",
            "[Shot 1] An opening action continues.\n"
            "[Shot 2] At 00:15.050, the second view begins.",
        )
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(30.0, [(14.9, 15.0)], start=0.0),
            prompt,
            "Natural / audio-led sync",
            0,
            30,
            12,
            5,
            15,
            self.project(30.0, 0.0)[0],
        )
        plan = json.loads(result[0])
        cut = plan["inter_lane_cuts"][0]
        self.assertEqual(cut["seconds"], 15.0)
        self.assertEqual(cut["selection_origin"], "measured_recovery")
        self.assertFalse(cut["native_shot_boundary"])
        self.assertTrue(cut["measured_low_density_recovery"])
        self.assertTrue(plan["shots"][1]["source_shot_continuation_at_lane_start"])
        self.assertEqual(plan["shots"][1]["carry_in_source_native_shot_index"], 1)
        self.assertIn("[Shot 2] At 00:00.050", result[9])

    def test_no_carry_metadata_is_neutral_for_native_multi_block_lanes(self):
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(25.0, [(12.0, 13.0)], start=0.0),
            FIFTEEN_SHOT_LONG_REF_PROMPT,
            "Natural / audio-led sync",
            0,
            25,
            12,
            5,
            15,
            self.project(25.0, 0.0)[0],
        )
        plan = json.loads(result[0])
        self.assertEqual(
            [shot["source_shot_continuation_at_lane_start"] for shot in plan["shots"]],
            [False, False],
        )
        self.assertEqual(
            [shot["carry_in_source_native_shot_index"] for shot in plan["shots"]],
            [0, 0],
        )
        self.assertEqual(
            [shot["carry_in_source_native_shot_start_seconds"] for shot in plan["shots"]],
            [0.0, 0.0],
        )

    def test_planner_rejects_duplicate_numbered_and_empty_native_shots(self):
        duplicate = ONE_SHOT_REF_PROMPT.replace(
            "[Shot 1] A medium shot follows the performer through the opening phrase.",
            "[Shot 1] An opening action.\n"
            "[Shot 1] At 00:10.000, a duplicate-number view.",
        )
        with self.assertRaisesRegex(ValueError, "consecutively numbered"):
            planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
                measured_report(15.0, [], start=0.0),
                duplicate,
                "Natural / audio-led sync",
                0,
                15,
                12,
                5,
                15,
                self.project(15.0, 0.0)[0],
            )
        empty = ONE_SHOT_REF_PROMPT.replace(
            "[Shot 1] A medium shot follows the performer through the opening phrase.",
            "[Shot 1]   ",
        )
        with self.assertRaisesRegex(ValueError, "no renderable description"):
            planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
                measured_report(15.0, [], start=0.0),
                empty,
                "Natural / audio-led sync",
                0,
                15,
                12,
                5,
                15,
                self.project(15.0, 0.0)[0],
            )

    def test_assembler_is_lazy_trims_each_shot_and_preserves_audio_identity(self):
        project = self.project(25.0, 30.0)[0]
        plan_result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(25.0, [(10.0, 15.0)], start=30.0),
            REF_PROMPT,
            "Dance / music sync",
            30,
            25,
            12,
            5,
            15,
            project,
            "should be ignored in dance mode",
        )
        assembler = planning.DiffusionGemmaH3ShotAssembler()
        audio = {"waveform": torch.zeros((1, 2, 25 * 4_000)), "sample_rate": 4_000}
        self.assertEqual(
            assembler.check_lazy_status(audio, plan_result[0], 25, 24, shot_1_images=None, shot_2_images=None),
            ["shot_1_images", "shot_2_images"],
        )
        shot_1 = torch.zeros((plan_result[7], 2, 3, 3))
        shot_2 = torch.ones((plan_result[12], 2, 3, 3))
        assembled = assembler.assemble(
            audio,
            plan_result[0],
            25,
            24,
            shot_1_images=shot_1,
            shot_2_images=shot_2,
        )
        report = json.loads(assembled[2])
        self.assertTrue(assembled[4])
        self.assertIs(assembled[1], audio)
        self.assertEqual(assembled[0].shape[0], 600)
        self.assertEqual(report["shots"][0]["retained_frames"], 300)
        self.assertEqual(report["shots"][1]["retained_frames"], 300)
        self.assertTrue(torch.all(assembled[0][:300] == 0))
        self.assertTrue(torch.all(assembled[0][300:] == 1))

    def test_dance_mode_ignores_lyrics_and_suppresses_visible_vocal_articulation(self):
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(15.0, [(0.0, 15.0)], start=0.0),
            ONE_SHOT_REF_PROMPT,
            "Dance / music sync",
            0,
            15,
            12,
            5,
            15,
            self.project(15.0, 0.0)[0],
            "These secret lyrics must not enter the H3 prompt",
        )
        self.assertNotIn("These secret lyrics", result[4])
        self.assertIn("No visible person sings", result[4])
        self.assertEqual(json.loads(result[0])["performance_mode"], "Dance / music sync")

    def test_natural_mode_uses_audio_only_without_lyrics_or_forced_mouth_state(self):
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(15.0, [(0.0, 15.0)], start=0.0),
            ONE_SHOT_REF_PROMPT,
            "Natural / audio-led sync",
            0,
            15,
            12,
            5,
            15,
            self.project(15.0, 0.0)[0],
            "These written words must never become a schedule",
            timed_lyrics_report_json="{not valid and deliberately ignored",
        )
        plan = json.loads(result[0])
        self.assertEqual(plan["performance_mode"], "Natural / audio-led sync")
        self.assertEqual(plan["effective_performance_mode"], "Natural / audio-led sync")
        self.assertEqual(plan["lyrics_timing_state"], "natural_audio_led")
        self.assertEqual(plan["shots"][0]["lyrics_timing_state"], "natural_audio_led")
        self.assertEqual(plan["shots"][0]["timed_lyrics_events"], [])
        self.assertIn("<Audio 1> alone decides whether and when", result[4])
        self.assertIn("do not force singing or blanket closed-mouth behavior", result[4])
        self.assertNotIn("These written words", result[4])
        self.assertNotIn("No visible person sings", result[4])

    def test_crossing_timed_event_is_clipped_and_rebased_into_both_lanes(self):
        timing = timed_lyrics_report(
            25.0,
            [
                {
                    "start_seconds": 11.5,
                    "end_seconds": 13.5,
                    "authored_lines": ["Second exact line"],
                    "transcript": "Second exact line",
                    "confidence": 0.98,
                }
            ],
            vocal_intervals=[(11.5, 13.5)],
        )
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(25.0, [(12.0, 13.0)], start=0.0),
            REF_PROMPT,
            "Lyrics + lip sync",
            0,
            25,
            12,
            5,
            15,
            self.project(25.0, 0.0)[0],
            DEFAULT_LYRICS,
            timed_lyrics_report_json=timing,
        )
        plan = json.loads(result[0])
        first = plan["shots"][0]["timed_lyrics_events"][0]
        second = plan["shots"][1]["timed_lyrics_events"][0]
        self.assertEqual((first["start_seconds"], first["end_seconds"]), (11.5, 12.5))
        self.assertFalse(first["clipped_at_lane_start"])
        self.assertTrue(first["clipped_at_lane_end"])
        self.assertEqual((second["start_seconds"], second["end_seconds"]), (0.0, 1.0))
        self.assertTrue(second["clipped_at_lane_start"])
        self.assertFalse(second["clipped_at_lane_end"])
        self.assertIn('00:11.500-00:12.500 "Second exact line"', result[4])
        self.assertIn('00:00.000-00:01.000 "Second exact line"', result[9])
        self.assertTrue(plan["lyrics_not_a_timing_schedule"])
        self.assertTrue(all(shot["lyrics_not_a_timing_schedule"] for shot in plan["shots"]))

    def test_analyzer_report_flows_directly_into_planner_lexical_hints(self):
        sample_rate = 1_000
        duration = 12.0
        sample_count = int(sample_rate * duration)
        clock = torch.arange(sample_count, dtype=torch.float32) / sample_rate
        accompaniment = 0.06 * torch.sin(2.0 * torch.pi * 73.0 * clock)
        vocal_mask = (clock >= 2.0) & (clock < 5.0)
        voice = torch.where(
            vocal_mask,
            0.15 * torch.sin(2.0 * torch.pi * 181.0 * clock),
            torch.zeros_like(clock),
        )
        final_audio = {
            "waveform": (accompaniment + voice).reshape(1, 1, -1),
            "sample_rate": sample_rate,
        }
        vocal_stem = {
            "waveform": voice.reshape(1, 1, -1),
            "sample_rate": sample_rate,
        }

        def transcribe(
            audio,
            language,
            return_timestamps,
            chunk_offset_seconds,
            absolute_song_start_seconds,
        ):
            del audio, language, return_timestamps, chunk_offset_seconds, absolute_song_start_seconds
            return {
                "chunks": [
                    {
                        "text": "we light the night",
                        "timestamp": [0.25, 1.75],
                    }
                ]
            }

        lyrics = "[Verse]\nWe light the night\nAnd hold the sky"
        audio_hash = timed_lyrics._waveform_sha256(final_audio)
        analyzer_result = timed_lyrics.DiffusionGemmaTimedLyricsAnalyzer().analyze(
            final_audio,
            "Lyrics + lip sync",
            lyrics,
            audio_hash,
            duration,
            0.0,
            duration,
            "en",
            0.55,
            vocal_stem,
            {"model_id": "fake/whisper-tiny", "transcribe": transcribe},
        )
        produced_report = json.loads(analyzer_result[0])
        self.assertTrue(produced_report["timing_ready"])
        project = planning.DiffusionGemmaProjectMasterContract().build(
            "Create an audio-led monochrome performance.",
            duration,
            "9:16",
            audio_hash,
            lyrics,
            0.0,
            duration,
            "MiniMax H3 Ref2VA",
            "{}",
            15.0,
        )[0]
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(
                duration,
                [(0.0, duration)],
                start=0.0,
                waveform_sha256=audio_hash,
            ),
            ONE_SHOT_REF_PROMPT,
            "Lyrics + lip sync",
            0.0,
            duration,
            12.0,
            5.0,
            15.0,
            project,
            lyrics,
            timed_lyrics_report_json=analyzer_result[0],
        )
        plan = json.loads(result[0])
        self.assertEqual(plan["effective_performance_mode"], "Lyrics + lip sync")
        self.assertEqual(plan["lyrics_timing_state"], "timing_ready")
        self.assertEqual(plan["shots"][0]["lyrics_timing_state"], "timed_lyrics")
        self.assertEqual(
            plan["shots"][0]["timed_lyrics_events"][0]["authored_lines"],
            ["We light the night"],
        )
        self.assertIn('"We light the night"', result[4])
        self.assertNotIn('"we light the night"', result[4])

    def test_missing_stale_weak_or_invalid_timing_safely_falls_back_to_natural(self):
        valid_event = {
            "start_seconds": 1.0,
            "end_seconds": 2.0,
            "authored_lines": ["First exact line"],
            "transcript": "First exact line",
            "confidence": 0.95,
        }
        cases = {
            "missing": "",
            "invalid_json": "{broken",
            "stale_audio": timed_lyrics_report(
                15.0, [valid_event], waveform_sha256="b" * 64
            ),
            "stale_lyrics": timed_lyrics_report(
                15.0, [valid_event], lyrics="different authored lyrics"
            ),
            "stale_excerpt": timed_lyrics_report(
                15.0, [valid_event], start=3.0
            ),
            "not_ready": timed_lyrics_report(
                15.0, [], timing_ready=False
            ),
            "weak": timed_lyrics_report(
                15.0,
                [{**valid_event, "confidence": 0.2}],
                minimum_alignment_confidence=0.7,
            ),
        }
        for label, timing in cases.items():
            with self.subTest(label=label):
                result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
                    measured_report(15.0, [(0.0, 15.0)], start=0.0),
                    ONE_SHOT_REF_PROMPT,
                    "Lyrics + lip sync",
                    0,
                    15,
                    12,
                    5,
                    15,
                    self.project(15.0, 0.0)[0],
                    DEFAULT_LYRICS,
                    timed_lyrics_report_json=timing,
                )
                plan = json.loads(result[0])
                self.assertEqual(plan["performance_mode"], "Lyrics + lip sync")
                self.assertEqual(
                    plan["effective_performance_mode"],
                    "Natural / audio-led sync",
                )
                self.assertEqual(plan["lyrics_timing_state"], "natural_fallback")
                self.assertFalse(plan["timed_lyrics_report"]["timing_ready"])
                self.assertTrue(plan["timed_lyrics_report"]["warnings"])
                self.assertEqual(plan["shots"][0]["timed_lyrics_events"], [])
                self.assertIn("alone decides whether and when", result[4])
                self.assertNotIn("First exact line", result[4])
                self.assertNotIn("sing these exact lines once", result[4])

    def test_optional_timing_input_appends_without_shifting_legacy_outputs(self):
        planner = planning.DiffusionGemmaAudioAwareMultiShotPlanner()
        optional_names = list(planner.INPUT_TYPES()["optional"])
        self.assertEqual(optional_names[-1], "timed_lyrics_report_json")
        self.assertEqual(planning.PERFORMANCE_MODES[:2], (
            "Dance / music sync",
            "Lyrics + lip sync",
        ))
        self.assertEqual(planning.PERFORMANCE_MODES[-1], "Natural / audio-led sync")
        common = (
            measured_report(15.0, [(0.0, 15.0)], start=0.0),
            ONE_SHOT_REF_PROMPT,
            "Dance / music sync",
            0,
            15,
            12,
            5,
            15,
            self.project(15.0, 0.0)[0],
        )
        legacy = planner.plan(*common)
        appended = planner.plan(*common, timed_lyrics_report_json="")
        self.assertEqual(legacy, appended)
        self.assertEqual(len(legacy), 24)
        self.assertEqual(planner.RETURN_NAMES[:4], (
            "plan_json",
            "status",
            "ready",
            "effective_shot_count",
        ))
        self.assertEqual(planner.RETURN_NAMES[4:9], (
            "shot_1_prompt",
            "shot_1_start",
            "shot_1_duration",
            "shot_1_frames",
            "shot_1_ready",
        ))
        self.assertTrue(legacy[8])
        self.assertFalse(legacy[13])

    def test_planner_rejects_unknown_performance_mode(self):
        with self.assertRaisesRegex(ValueError, "performance_mode"):
            planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
                measured_report(15.0, [(0.0, 15.0)]),
                REF_PROMPT,
                "karaoke",
                0,
                15,
                12,
                5,
                15,
                self.project(15.0, 0.0)[0],
            )

    def test_planner_rejects_conflicting_h3_spoken_dialogue_blocks(self):
        conflicting = ONE_SHOT_REF_PROMPT.replace(
            "A medium shot follows the performer through the opening phrase.",
            "A medium shot follows the performer. <d>[English] We leave now.</d>",
        )
        with self.assertRaisesRegex(ValueError, "spoken-dialogue"):
            planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
                measured_report(15.0, [(0.0, 15.0)], start=0.0),
                conflicting,
                "Dance / music sync",
                0,
                15,
                12,
                5,
                15,
                self.project(15.0, 0.0)[0],
            )

    def test_dance_mode_rejects_active_visible_vocal_cues_but_allows_prohibitions(self):
        active = ONE_SHOT_REF_PROMPT.replace(
            "A medium shot follows the performer through the opening phrase.",
            "A medium shot follows the performer as she lip-syncs the chorus.",
        )
        with self.assertRaisesRegex(ValueError, "active visible"):
            planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
                measured_report(15.0, [(0.0, 15.0)], start=0.0),
                active,
                "Dance / music sync",
                0,
                15,
                12,
                5,
                15,
                self.project(15.0, 0.0)[0],
            )
        prohibited = ONE_SHOT_REF_PROMPT.replace(
            "A medium shot follows the performer through the opening phrase.",
            "A medium shot follows the performer; she never sings or mouths words.",
        )
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(15.0, [(0.0, 15.0)], start=0.0),
            prohibited,
            "Dance / music sync",
            0,
            15,
            12,
            5,
            15,
            self.project(15.0, 0.0)[0],
        )
        self.assertTrue(result[2])

        for prohibited_wording in (
            "No visible person sings, speaks, mouths words, or lip-syncs.",
            "The performer does not sing, speak, mouth words, or lip-sync.",
            "The performer never sings, speaks, mouths words, or lip-syncs.",
            "The performer refrains from singing, speaking, mouthing words, or lip-syncing.",
            "The performer avoids singing or speaking.",
            "The performer does not sing and never mouths words.",
        ):
            prohibited_prompt = ONE_SHOT_REF_PROMPT.replace(
                "A medium shot follows the performer through the opening phrase.",
                prohibited_wording,
            )
            result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
                measured_report(15.0, [(0.0, 15.0)], start=0.0),
                prohibited_prompt,
                "Dance / music sync",
                0,
                15,
                12,
                5,
                15,
                self.project(15.0, 0.0)[0],
            )
            self.assertTrue(result[2], prohibited_wording)

        for active_wording in (
            "Without changing framing, the visible performer sings the chorus.",
            "The performer avoids camera shake and sings the chorus.",
            "The performer never turns away and sings the chorus.",
            "The performer does not move, then sings the chorus.",
            "The performer sings while the soundtrack swells.",
            "The lead singer sings in time with <Audio 1>.",
            "The woman mouths every word over the non-diegetic song.",
            "The performer sings while an off-screen crowd cheers.",
            "An off-screen drummer plays while the performer sings.",
            "The audio-only backing track plays as the woman mouths every word.",
            "The performer does not dance then sings.",
            "The performer never stops and sings.",
            "The performer does not move and sings.",
            "The performer never pauses before singing.",
        ):
            active_prompt = ONE_SHOT_REF_PROMPT.replace(
                "A medium shot follows the performer through the opening phrase.",
                active_wording,
            )
            with self.assertRaisesRegex(ValueError, "active visible"):
                planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
                    measured_report(15.0, [(0.0, 15.0)], start=0.0),
                    active_prompt,
                    "Dance / music sync",
                    0,
                    15,
                    12,
                    5,
                    15,
                    self.project(15.0, 0.0)[0],
                )

        vocal_soundtrack = prohibited.replace(
            "Use the supplied <Audio 1> composition as reference; final delivery preserves the locked master.",
            "The non-diegetic soundtrack has a female singer singing the chorus over live drums.",
        )
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(15.0, [(0.0, 15.0)], start=0.0),
            vocal_soundtrack,
            "Dance / music sync",
            0,
            15,
            12,
            5,
            15,
            self.project(15.0, 0.0)[0],
        )
        self.assertTrue(result[2])

    def test_planner_rejects_stale_or_wrong_song_measurement(self):
        stale = json.loads(measured_report(15.0, [(0.0, 15.0)], start=0.0))
        stale["schema"] = "wrong.schema"
        with self.assertRaisesRegex(ValueError, "supported music audition report"):
            planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
                json.dumps(stale),
                ONE_SHOT_REF_PROMPT,
                "Dance / music sync",
                0,
                15,
                12,
                5,
                15,
                self.project(15.0, 0.0)[0],
            )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
                measured_report(15.0, [(0.0, 15.0)], start=0.0, waveform_sha256="b" * 64),
                ONE_SHOT_REF_PROMPT,
                "Dance / music sync",
                0,
                15,
                12,
                5,
                15,
                self.project(15.0, 0.0)[0],
            )

    def test_cut_search_finds_complete_non_greedy_sequence(self):
        boundaries = planning._choose_boundaries(
            30.0,
            3,
            8.0,
            12.0,
            [(9.0, "measured"), (12.0, "measured"), (22.0, "measured")],
        )
        self.assertEqual([item[0] for item in boundaries], [0.0, 12.0, 22.0, 30.0])

    def test_embedded_source_timestamp_is_rebased_to_local_shot_clock(self):
        prompt = REF_PROMPT.replace(
            "through the ending phrase.",
            "through the ending phrase. At 00:20.000, the performer raises one hand.",
        )
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(25.0, [(12.0, 13.0)], start=0.0),
            prompt,
            "Dance / music sync",
            0,
            25,
            12,
            5,
            15,
            self.project(25.0, 0.0)[0],
        )
        self.assertIn("At 00:07.500", result[9])
        self.assertNotIn("At 00:20.000", result[9])

    def test_untimed_dense_lyrics_fall_back_without_proportional_scheduling(self):
        dense_line = " ".join(f"word{index}" for index in range(45))
        result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(25.0, [(12.4, 12.6)], start=0.0),
            REF_PROMPT,
            "Lyrics + lip sync",
            0,
            25,
            12,
            5,
            15,
            self.project(25.0, 0.0)[0],
            dense_line + "\ntwo words",
        )
        plan = json.loads(result[0])
        self.assertEqual(plan["effective_performance_mode"], "Natural / audio-led sync")
        self.assertEqual(plan["lyrics_timing_state"], "natural_fallback")
        self.assertNotIn("word1", result[4])
        self.assertNotIn("2.4 words per second", result[1])

    def test_fractional_cut_frame_allocation_sums_to_exact_master(self):
        duration = 25.03
        project = self.project(duration, 0.0)[0]
        plan_result = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(duration, [(12.4, 12.6)], start=0.0),
            REF_PROMPT,
            "Dance / music sync",
            0,
            duration,
            12,
            5,
            15,
            project,
        )
        plan = json.loads(plan_result[0])
        self.assertEqual(
            sum(shot["retained_master_frames"] for shot in plan["shots"]),
            round(duration * 24),
        )
        assembler = planning.DiffusionGemmaH3ShotAssembler()
        audio = {
            "waveform": torch.zeros((1, 2, round(duration * 4_000))),
            "sample_rate": 4_000,
        }
        assembled = assembler.assemble(
            audio,
            plan_result[0],
            duration,
            24,
            shot_1_images=torch.zeros((plan_result[7], 2, 3, 3)),
            shot_2_images=torch.ones((plan_result[12], 2, 3, 3)),
        )
        self.assertTrue(assembled[4], assembled[3])
        self.assertEqual(assembled[0].shape[0], round(duration * 24))

    def test_assembler_rejects_target_clock_or_audio_duration_mismatch(self):
        plan = planning.DiffusionGemmaAudioAwareMultiShotPlanner().plan(
            measured_report(15.0, [(0.0, 15.0)], start=0.0),
            ONE_SHOT_REF_PROMPT,
            "Dance / music sync",
            0,
            15,
            12,
            5,
            15,
            self.project(15.0, 0.0)[0],
        )
        images = torch.zeros((plan[7], 2, 3, 3))
        assembler = planning.DiffusionGemmaH3ShotAssembler()
        wrong_target = assembler.assemble(
            {"waveform": torch.zeros((1, 2, 15_000)), "sample_rate": 1_000},
            plan[0],
            10,
            24,
            shot_1_images=images,
        )
        self.assertFalse(wrong_target[4])
        self.assertIn("target duration", wrong_target[3])
        wrong_audio = assembler.assemble(
            {"waveform": torch.zeros((1, 2, 14_000)), "sample_rate": 1_000},
            plan[0],
            15,
            24,
            shot_1_images=images,
        )
        self.assertFalse(wrong_audio[4])
        self.assertIn("audio duration", wrong_audio[3])

    def test_delivery_manifest_is_honest_about_plan_only_variants(self):
        result = planning.DiffusionGemmaMultiFormatDeliveryPlanner().plan(
            self.project(25.0)[0],
            "Master + social adaptations",
            "Plan 15s + 6s cutdowns",
            10.0,
        )
        manifest = json.loads(result[0])
        self.assertTrue(manifest["plan_only"])
        self.assertTrue(result[3])
        planned = [item for item in manifest["deliverables"] if item["id"] != "primary_master"]
        self.assertTrue(planned)
        self.assertTrue(all(item["status"] == "planned_only_not_rendered" for item in planned))

    def test_delivery_manifest_honors_requested_adaptation_aspects(self):
        project = planning.DiffusionGemmaProjectMasterContract().build(
            "brief",
            15,
            "9:16",
            "a" * 64,
            "lyrics",
            0,
            15,
            "MiniMax H3 Ref2VA",
            '{"primary":"9:16 master","adaptations":["4:3"]}',
            15,
        )[0]
        result = planning.DiffusionGemmaMultiFormatDeliveryPlanner().plan(
            project, "Master + social adaptations", "None", 10
        )
        manifest = json.loads(result[0])
        self.assertEqual(
            [item["aspect_ratio"] for item in manifest["deliverables"]],
            ["9:16", "4:3"],
        )
        self.assertEqual(manifest["requested_deliverables"]["adaptations"], ["4:3"])

    def test_seed_fanout_is_project_stable(self):
        contract = self.project(25.0)[0]
        node = planning.DiffusionGemmaH3ShotSeedFanout()
        first = node.fanout(42, contract)
        second = node.fanout(42, contract)
        third = node.fanout(43, contract)
        self.assertEqual(first, second)
        self.assertNotEqual(first[:4], third[:4])
        self.assertEqual(len(set(first[:4])), 4)

    def test_all_nodes_are_registered(self):
        self.assertEqual(
            set(planning.NODE_CLASS_MAPPINGS),
            {
                "DiffusionGemmaProjectMasterContract",
                "DiffusionGemmaAudioAwareMultiShotPlanner",
                "DiffusionGemmaH3RelayReferenceGate",
                "DiffusionGemmaMultiFormatDeliveryPlanner",
                "DiffusionGemmaH3ShotSeedFanout",
                "DiffusionGemmaH3ShotAssembler",
            },
        )


if __name__ == "__main__":
    unittest.main()
