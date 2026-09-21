from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from scripts import benchmark_grounding
from scripts import generate_grounding_fixtures


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPOSITORY_ROOT / "benchmarks" / "grounding_v1"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png_chunks(path: Path) -> list[tuple[bytes, bytes]]:
    payload = path.read_bytes()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError(f"{path} is not a PNG file")
    chunks: list[tuple[bytes, bytes]] = []
    offset = 8
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        chunks.append((kind, payload[data_start : data_start + length]))
        offset = data_start + length + 4
    return chunks


class GroundingBenchmarkFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.manifest = benchmark_grounding.validate_manifest(cls.raw_manifest)

    def test_manifest_has_exact_versioned_category_counts(self):
        self.assertEqual(self.manifest["schema_version"], "dg-grounding-benchmark/1")
        self.assertEqual(len(self.manifest["cases"]), 30)
        self.assertEqual(
            Counter(case["category"] for case in self.manifest["cases"]),
            Counter(
                {
                    "synthetic_image": 10,
                    "synthetic_video": 8,
                    "private_clean": 6,
                    "private_known_failure": 6,
                }
            ),
        )
        self.assertEqual(self.manifest["seeds"], [17, 101, 809])
        for case in self.manifest["cases"]:
            self.assertEqual(case["seeds"], [17, 101, 809], case["id"])

    def test_holdout_allocation_is_nearest_whole_case_to_twenty_percent(self):
        holdout_counts = Counter(
            case["category"] for case in self.manifest["cases"] if case["holdout"]
        )
        self.assertEqual(
            holdout_counts,
            Counter(
                {
                    "synthetic_image": 2,
                    "synthetic_video": 2,
                    "private_clean": 1,
                    "private_known_failure": 1,
                }
            ),
        )
        expected_ratios = {
            "synthetic_image": 2 / 10,
            "synthetic_video": 2 / 8,
            "private_clean": 1 / 6,
            "private_known_failure": 1 / 6,
        }
        totals = Counter(case["category"] for case in self.manifest["cases"])
        for category, ratio in expected_ratios.items():
            self.assertEqual(holdout_counts[category] / totals[category], ratio)

    def test_public_fixture_hashes_are_exact_and_private_hashes_are_sentinels(self):
        public_count = 0
        private_count = 0
        for case in self.manifest["cases"]:
            fixture_path = BENCHMARK_ROOT / case["fixture"]
            if case["private"]:
                private_count += 1
                self.assertFalse(fixture_path.exists(), case["id"])
                self.assertEqual(case["fixture_sha256"], "0" * 64, case["id"])
            else:
                public_count += 1
                self.assertTrue(fixture_path.is_file(), case["id"])
                self.assertEqual(_sha256(fixture_path), case["fixture_sha256"], case["id"])
        self.assertEqual(public_count, 18)
        self.assertEqual(private_count, 12)

    def test_every_case_has_scoring_annotations_and_all_target_profiles_are_covered(self):
        profiles = set()
        for case in self.manifest["cases"]:
            profiles.add(case["target_profile"])
            self.assertTrue(case["expected_facts"], case["id"])
            self.assertTrue(case["prohibited_facts"], case["id"])
            self.assertTrue(case["ambiguity_labels"], case["id"])
            for fact in case["expected_facts"] + case["prohibited_facts"]:
                self.assertTrue(fact["claim"], f"{case['id']}:{fact['id']}")
                self.assertTrue(fact["aliases"], f"{case['id']}:{fact['id']}")
                self.assertTrue(fact["metric_tags"], f"{case['id']}:{fact['id']}")
            for fact in case["expected_facts"]:
                self.assertIn("supported_fact", fact["metric_tags"], case["id"])
            for fact in case["prohibited_facts"]:
                self.assertIn("unsupported_fact", fact["metric_tags"], case["id"])
        self.assertEqual(profiles, {"ltx", "h3_t2va", "h3_ref2va", "ideogram"})

    def test_video_facts_reference_distinct_frames_and_exact_timecodes(self):
        videos = [case for case in self.manifest["cases"] if case["category"] == "synthetic_video"]
        self.assertEqual(len(videos), 8)
        for case in videos:
            self.assertTrue(
                any("temporal_order" in fact["metric_tags"] for fact in case["expected_facts"]),
                case["id"],
            )
            referenced_frames = {
                frame
                for fact in case["expected_facts"]
                for frame in fact["frame_indices"]
            }
            referenced_times = {
                timecode
                for fact in case["expected_facts"]
                for timecode in fact["timecodes"]
            }
            self.assertGreaterEqual(len(referenced_frames), 2, case["id"])
            self.assertIn(0, referenced_frames, case["id"])
            self.assertIn(5, referenced_frames, case["id"])
            self.assertIn(0.0, referenced_times, case["id"])
            self.assertIn(1.25, referenced_times, case["id"])

    def test_counterfactual_groups_cover_all_required_variants(self):
        groups: dict[str, set[str]] = {}
        for case in self.manifest["cases"]:
            if case["counterfactual_group"]:
                groups.setdefault(case["counterfactual_group"], set()).add(
                    case["counterfactual_variant"]
                )
        self.assertEqual(groups["image_shape_swap"], {"real", "neutral", "unrelated"})
        self.assertEqual(
            groups["video_motion_order"],
            {"original", "reversed", "shuffled", "frozen_first_frame"},
        )

    def test_video_files_are_six_frame_animated_pngs(self):
        for case in self.manifest["cases"]:
            if case["category"] != "synthetic_video":
                continue
            chunks = _png_chunks(BENCHMARK_ROOT / case["fixture"])
            animation_control = [data for kind, data in chunks if kind == b"acTL"]
            frame_controls = [data for kind, data in chunks if kind == b"fcTL"]
            self.assertEqual(len(animation_control), 1, case["id"])
            frame_count, play_count = struct.unpack(">II", animation_control[0])
            self.assertEqual(frame_count, 6, case["id"])
            self.assertEqual(play_count, 0, case["id"])
            self.assertEqual(len(frame_controls), 6, case["id"])

    def test_generator_reproduces_all_checked_in_bytes(self):
        self.assertEqual(generate_grounding_fixtures.check_benchmark(BENCHMARK_ROOT), [])

    def test_scorer_validates_manifest_without_fake_performance_results(self):
        report = benchmark_grounding.evaluate_manifest(MANIFEST_PATH)
        self.assertEqual(report["manifest"]["case_count"], 30)
        self.assertEqual(report["metrics"]["scored_case_count"], 0)
        self.assertEqual(report["metrics"]["scored_run_count"], 0)
        self.assertEqual(report["metrics"]["skipped_case_count"], 30)
        self.assertEqual(report["release_candidate"]["status"], "incomplete")
        self.assertFalse(report["release_candidate"]["eligible_for_verdict"])
        self.assertFalse(report["release_candidate"]["passed"])
        self.assertFalse((BENCHMARK_ROOT / "results").exists())

    def test_complete_strict_release_candidate_can_receive_an_explicit_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = json.loads(json.dumps(self.raw_manifest))
            for case in manifest["cases"]:
                fixture_path = root / case["fixture"]
                fixture_path.parent.mkdir(parents=True, exist_ok=True)
                fixture_payload = f"unit fixture for {case['id']}".encode("utf-8")
                fixture_path.write_bytes(fixture_payload)
                case["fixture_sha256"] = hashlib.sha256(fixture_payload).hexdigest()

                media_variant = case["counterfactual_variant"] or "case_default"
                condition = {
                    "condition_id": "release-v1",
                    "prompting": "guard_strict",
                    "sampling_profile": "checkpoint_defaults",
                    "media_variant": media_variant,
                    "precision": "nvfp4",
                    "telemetry_enabled": True,
                    "frame_budget": 6 if case["category"] == "synthetic_video" else 1,
                    "visual_token_budget": 1024,
                    "model_revision": "unit-release-revision",
                    "release_candidate": True,
                }
                expected_ids = [fact["id"] for fact in case["expected_facts"]]
                result = {
                    "schema_version": benchmark_grounding.RESULT_SCHEMA_VERSION,
                    "case_id": case["id"],
                    "runs": [
                        {
                            "seed": seed,
                            "mode": "strict",
                            "condition": condition,
                            "guard_decision": "pass",
                            "analysis_status": "grounded",
                            "reported_fact_ids": expected_ids,
                            "unsupported_fact_ids": [],
                            "metadata": {},
                        }
                        for seed in case["seeds"]
                    ],
                }
                result_path = root / case["result"]
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(result), encoding="utf-8")

            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = benchmark_grounding.evaluate_manifest(manifest_path)

        release = report["release_candidate"]
        self.assertEqual(release["status"], "pass")
        self.assertTrue(release["eligible_for_verdict"])
        self.assertTrue(release["passed"])
        self.assertEqual(release["completeness"]["release_run_count"], 90)
        self.assertEqual(release["slices"]["development"]["run_count"], 72)
        self.assertEqual(release["slices"]["holdout"]["run_count"], 18)
        self.assertEqual(release["slices"]["overall"]["temporal_order_accuracy"], 1.0)
        self.assertEqual(release["counterfactual"]["discrimination_rate"], 1.0)
        self.assertTrue(all(item["passed"] for item in release["thresholds"].values()))


if __name__ == "__main__":
    unittest.main()
