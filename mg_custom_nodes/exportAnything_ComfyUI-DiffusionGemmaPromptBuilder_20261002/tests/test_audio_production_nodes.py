from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
BAD_REAL_OUTPUT = COMFY_ROOT / "output" / "video" / "ltx2.5_diffusiongemma_i2v_00086_.mp4"
GOOD_REAL_OUTPUT = COMFY_ROOT / "output" / "video" / "ltx2.5_diffusiongemma_i2v_00087_.mp4"


def load_audio_module():
    module_name = f"diffusiongemma_audio_production_fixture_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "audio_production_nodes.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load audio production nodes")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_package_modules():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    package_name = f"diffusiongemma_audio_package_fixture_{uuid.uuid4().hex}"
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
    return package, sys.modules[f"{package_name}.nodes"]


def synthetic_song(
    *,
    bpm: float = 90.0,
    duration_seconds: float = 12.0,
    sample_rate: int = 8_000,
    channels: int = 1,
    gain: float = 1.0,
    pitch_hz: float = 220.0,
):
    samples = int(round(duration_seconds * sample_rate))
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    tone = (
        0.18 * torch.sin(2.0 * torch.pi * pitch_hz * time)
        + 0.10 * torch.sin(2.0 * torch.pi * pitch_hz * 1.5 * time)
        + 0.07 * torch.sin(2.0 * torch.pi * pitch_hz * 2.0 * time)
    )
    beat_period = 60.0 / bpm
    beat_phase = torch.remainder(time, beat_period)
    pulse = 0.20 * torch.exp(-beat_phase * 35.0)
    waveform = ((tone + pulse) * gain).clamp(-0.98, 0.98)
    if channels == 1:
        routed = waveform.reshape(1, 1, -1)
    else:
        routed = torch.stack((waveform, waveform * 0.82), dim=0).unsqueeze(0)
    return {"waveform": routed.contiguous(), "sample_rate": sample_rate}


def silent_audio(*, duration_seconds: float = 12.0, sample_rate: int = 8_000):
    return {
        "waveform": torch.zeros((1, 1, int(duration_seconds * sample_rate)), dtype=torch.float32),
        "sample_rate": sample_rate,
    }


def decode_mp4_audio(path: Path):
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    interleaved = np.frombuffer(raw, dtype="<f4")
    if interleaved.size == 0 or interleaved.size % 2:
        raise RuntimeError(f"no valid stereo audio decoded from {path}")
    waveform = torch.from_numpy(interleaved.copy()).reshape(-1, 2).T.unsqueeze(0)
    return {"waveform": waveform, "sample_rate": 48_000}


class AudioAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audio = load_audio_module()

    def test_waveform_hash_is_exact_and_deterministic(self) -> None:
        source = synthetic_song()
        first = self.audio.waveform_sha256(source)
        second = self.audio.waveform_sha256(source)
        self.assertEqual(first, second)
        changed = {"waveform": source["waveform"].clone(), "sample_rate": source["sample_rate"]}
        changed["waveform"][..., 100] += 1.0e-4
        self.assertNotEqual(first, self.audio.waveform_sha256(changed))
        retimed = {"waveform": source["waveform"], "sample_rate": source["sample_rate"] + 1}
        self.assertNotEqual(first, self.audio.waveform_sha256(retimed))

    def test_analysis_is_deterministic_and_reports_bounded_timeline_proxies(self) -> None:
        source = synthetic_song(duration_seconds=52.0)
        first = self.audio.analyze_decoded_audio(
            source,
            expected_bpm=90.0,
            excerpt_duration_seconds=50.0,
        )
        second = self.audio.analyze_decoded_audio(
            source,
            expected_bpm=90.0,
            excerpt_duration_seconds=50.0,
        )
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(first["tempo_relation"], "direct")
        excerpt = first["suggested_excerpt"]
        timeline = excerpt["selected_excerpt_timeline"]
        self.assertGreater(timeline["entry_count_total"], 24)
        self.assertEqual(timeline["entries_returned"], 24)
        self.assertTrue(timeline["truncated"])
        self.assertLessEqual(len(excerpt["low_density_visual_recovery_intervals"]), 8)
        self.assertLessEqual(len(excerpt["vocal_active_stable_intervals"]), 8)
        self.assertIn("not vocal transcription", excerpt["timeline_note"])
        for entry in timeline["entries"]:
            self.assertIn("low_density_visual_recovery_proxy", entry)
            self.assertIn("vocal_active_stable_proxy", entry)

    def test_canonically_aligned_low_pressure_double_time_is_advisory(self) -> None:
        double_time = self.audio.analyze_decoded_audio(
            synthetic_song(bpm=180.0),
            expected_bpm=90.0,
            excerpt_duration_seconds=4.0,
        )
        failures = self.audio._candidate_failures(
            double_time,
            excerpt_duration_seconds=4.0,
            minimum_score=0.52,
        )
        self.assertEqual(double_time["tempo_relation"], "double_time_detected")
        self.assertEqual(failures, [])
        self.assertEqual(
            self.audio._candidate_advisories(double_time),
            ["canonically_aligned_low_pressure_double_time_pulse"],
        )
        excerpt = double_time["suggested_excerpt"]
        self.assertLessEqual(
            excerpt["onset_density_per_second"],
            self.audio.MAX_SAFE_DOUBLE_TIME_EXCERPT_ONSETS_PER_SECOND,
        )
        self.assertGreaterEqual(
            excerpt["low_density_visual_recovery_coverage"],
            self.audio.MIN_SAFE_DOUBLE_TIME_VISUAL_RECOVERY_COVERAGE,
        )

    def test_double_time_safety_contract_boundaries_fail_closed(self) -> None:
        safe = self.audio.analyze_decoded_audio(
            synthetic_song(bpm=180.0),
            expected_bpm=90.0,
            excerpt_duration_seconds=4.0,
        )
        unsafe_mutations = {
            "canonical_error": (
                "canonical_tempo_relative_error",
                self.audio.MAX_CANONICAL_DOUBLE_TIME_RELATIVE_ERROR + 0.001,
            ),
            "excerpt_onsets": (
                "suggested_excerpt.onset_density_per_second",
                self.audio.MAX_SAFE_DOUBLE_TIME_EXCERPT_ONSETS_PER_SECOND + 0.001,
            ),
            "vocal_activity": (
                "suggested_excerpt.likely_vocal_active_proxy",
                False,
            ),
        }
        for label, (path, value) in unsafe_mutations.items():
            with self.subTest(label=label):
                analysis = copy.deepcopy(safe)
                if path.startswith("suggested_excerpt."):
                    analysis["suggested_excerpt"][path.split(".", 1)[1]] = value
                else:
                    analysis[path] = value
                failures = self.audio._candidate_failures(
                    analysis,
                    excerpt_duration_seconds=4.0,
                    minimum_score=0.0,
                )
                self.assertIn("double_time_outside_low_pressure_contract", failures)

        supporting_signals = (
            (
                "low_density_visual_recovery_coverage",
                self.audio.MIN_SAFE_DOUBLE_TIME_VISUAL_RECOVERY_COVERAGE,
            ),
            ("score", self.audio.MIN_SAFE_DOUBLE_TIME_EXCERPT_SCORE),
            (
                "tonal_family_consistency",
                self.audio.MIN_SAFE_DOUBLE_TIME_TONAL_FAMILY_CONSISTENCY,
            ),
            (
                "tonal_transition_stability",
                self.audio.MIN_SAFE_DOUBLE_TIME_TONAL_TRANSITION_STABILITY,
            ),
        )
        weak = copy.deepcopy(safe)
        weak["suggested_excerpt"].update(
            {
                "low_density_visual_recovery_coverage": 0.0,
                "score": 0.0,
                # Stay above the independent general tonal hard gates so this
                # test isolates the double-time evidence bundle.
                "tonal_family_consistency": 0.61,
                "tonal_transition_stability": 0.51,
            }
        )
        for index, (key, passing_value) in enumerate(supporting_signals):
            with self.subTest(one_support_signal=key):
                one_signal = copy.deepcopy(weak)
                one_signal["suggested_excerpt"][key] = passing_value
                failures = self.audio._candidate_failures(
                    one_signal,
                    excerpt_duration_seconds=4.0,
                    minimum_score=0.0,
                )
                self.assertIn(
                    "double_time_outside_low_pressure_contract", failures
                )

            with self.subTest(two_support_signals=key):
                two_signals = copy.deepcopy(weak)
                second_key, second_value = supporting_signals[(index + 1) % len(supporting_signals)]
                two_signals["suggested_excerpt"][key] = passing_value
                two_signals["suggested_excerpt"][second_key] = second_value
                failures = self.audio._candidate_failures(
                    two_signals,
                    excerpt_duration_seconds=4.0,
                    minimum_score=0.0,
                )
                self.assertNotIn(
                    "double_time_outside_low_pressure_contract", failures
                )

    def test_latest_canonical_double_time_candidates_are_not_false_rejected(self) -> None:
        baseline = self.audio.analyze_decoded_audio(
            synthetic_song(bpm=180.0),
            expected_bpm=90.0,
            excerpt_duration_seconds=4.0,
        )
        # Exact decisive measurements from the two 92 BPM workflow candidates
        # that policy revision 3 rejected solely as double-time. Each has low
        # measured onset pressure and complementary strong excerpt evidence.
        candidates = (
            {
                "production_score": 0.733087918364718,
                "score": 0.949877043973501,
                "onsets": 2.96,
                "recovery": 0.52,
                "tonal_family": 1.0,
                "tonal_transition": 1.0,
            },
            {
                "production_score": 0.743437958264606,
                "score": 0.8823016780651074,
                "onsets": 3.52,
                "recovery": 1.0,
                "tonal_family": 0.75,
                "tonal_transition": 0.8333333333333333,
            },
        )
        for index, values in enumerate(candidates, start=1):
            with self.subTest(candidate=index):
                analysis = copy.deepcopy(baseline)
                analysis.update(
                    {
                        "expected_bpm": 92.0,
                        "detected_bpm_raw": 187.5,
                        "detected_bpm": 93.75,
                        "tempo_relation": "double_time_detected",
                        "canonical_tempo_relative_error": 0.0190217391304348,
                        "production_score": values["production_score"],
                    }
                )
                analysis["suggested_excerpt"].update(
                    {
                        "score": values["score"],
                        "onset_density_per_second": values["onsets"],
                        "low_density_visual_recovery_coverage": values["recovery"],
                        "tonal_family_consistency": values["tonal_family"],
                        "tonal_transition_stability": values["tonal_transition"],
                        "likely_vocal_active_proxy": True,
                    }
                )
                failures = self.audio._candidate_failures(
                    analysis,
                    excerpt_duration_seconds=4.0,
                    minimum_score=0.52,
                )
                self.assertEqual(failures, [])
                self.assertEqual(
                    self.audio._candidate_advisories(analysis),
                    ["canonically_aligned_low_pressure_double_time_pulse"],
                )

    def test_canonically_aligned_half_time_is_advisory_but_misaligned_is_hard(self) -> None:
        aligned = self.audio.analyze_decoded_audio(
            synthetic_song(bpm=62.5),
            expected_bpm=124.0,
            excerpt_duration_seconds=4.0,
        )
        aligned_failures = self.audio._candidate_failures(
            aligned,
            excerpt_duration_seconds=4.0,
            minimum_score=0.52,
        )
        self.assertEqual(aligned["tempo_relation"], "half_time_detected")
        self.assertAlmostEqual(aligned["detected_bpm"], 125.0, places=6)
        self.assertLess(aligned["canonical_tempo_relative_error"], 0.01)
        self.assertNotIn(
            "half_time_outside_canonical_alignment_contract",
            aligned_failures,
        )
        self.assertEqual(
            self.audio._candidate_advisories(aligned),
            ["canonically_aligned_half_time_pulse"],
        )
        self.assertNotIn("production_score_below_threshold", aligned_failures)

        misaligned = self.audio.analyze_decoded_audio(
            synthetic_song(bpm=62.5),
            expected_bpm=100.0,
            excerpt_duration_seconds=4.0,
        )
        misaligned_failures = self.audio._candidate_failures(
            misaligned,
            excerpt_duration_seconds=4.0,
            minimum_score=0.0,
        )
        self.assertEqual(misaligned["tempo_relation"], "half_time_detected")
        self.assertGreater(
            misaligned["canonical_tempo_relative_error"],
            self.audio.MAX_CANONICAL_HALF_TIME_RELATIVE_ERROR,
        )
        self.assertIn(
            "half_time_outside_canonical_alignment_contract",
            misaligned_failures,
        )

    def test_broadband_noise_cannot_masquerade_as_a_good_song(self) -> None:
        generator = torch.Generator().manual_seed(20260820)
        noise = {
            "waveform": 0.2 * torch.randn((1, 2, 96_000), generator=generator),
            "sample_rate": 8_000,
        }
        analysis = self.audio.analyze_decoded_audio(
            noise,
            expected_bpm=90.0,
            excerpt_duration_seconds=4.0,
        )
        failures = self.audio._candidate_failures(
            analysis,
            excerpt_duration_seconds=4.0,
            minimum_score=0.52,
        )
        self.assertGreater(analysis["spectral_flatness_mean"], 0.35)
        self.assertIn("broadband_noise_or_artifact_like_spectrum", failures)

        # A short noise burst can acquire a spurious key-confidence score.  It
        # must still fail on its broadband, non-tonal spectrum.
        short_generator = torch.Generator().manual_seed(1)
        short_noise = {
            "waveform": 0.2 * torch.randn((1, 2, 16_000), generator=short_generator),
            "sample_rate": 8_000,
        }
        short_analysis = self.audio.analyze_decoded_audio(
            short_noise,
            expected_bpm=0.0,
            excerpt_duration_seconds=1.0,
        )
        short_failures = self.audio._candidate_failures(
            short_analysis,
            excerpt_duration_seconds=1.0,
            minimum_score=0.52,
        )
        self.assertIn("broadband_noise_or_artifact_like_spectrum", short_failures)

    def test_global_remote_tonal_churn_cannot_hide_outside_selected_excerpt(self) -> None:
        sample_rate = 8_000
        segment_samples = sample_rate * 2
        time = torch.arange(segment_samples, dtype=torch.float32) / sample_rate
        frequencies = (220.0, 349.23, 554.37, 261.63, 415.30, 659.25)
        waveform = torch.cat(
            [0.25 * torch.sin(2.0 * torch.pi * frequency * time) for frequency in frequencies]
        ).reshape(1, 1, -1)
        analysis = self.audio.analyze_decoded_audio(
            {"waveform": waveform, "sample_rate": sample_rate},
            expected_bpm=0.0,
            excerpt_duration_seconds=4.0,
        )
        failures = self.audio._candidate_failures(
            analysis,
            excerpt_duration_seconds=4.0,
            minimum_score=0.0,
        )
        self.assertGreaterEqual(analysis["tonal_window_count"], 4)
        self.assertTrue(
            "global_remote_tonal_jumps" in failures
            or "global_tonal_transition_instability" in failures,
            failures,
        )

    def test_source_channel_clipping_is_measured_before_mono_downmix(self) -> None:
        clipped = synthetic_song(channels=2)
        clipped["waveform"] = clipped["waveform"].clone()
        clipped["waveform"][:, 0, :12_000] = 1.0
        analysis = self.audio.analyze_decoded_audio(
            clipped,
            expected_bpm=90.0,
            excerpt_duration_seconds=4.0,
        )
        failures = self.audio._candidate_failures(
            analysis,
            excerpt_duration_seconds=4.0,
            minimum_score=0.52,
        )
        self.assertGreater(analysis["clipped_sample_fraction"], 0.02)
        self.assertAlmostEqual(analysis["peak_dbfs"], 0.0, places=5)
        self.assertIn("excessive_clipping", failures)

    def test_ace_duration_quantization_allows_one_latent_step_only(self) -> None:
        analysis = self.audio.analyze_decoded_audio(
            synthetic_song(duration_seconds=25.12),
            expected_bpm=90.0,
            excerpt_duration_seconds=25.12,
        )
        rounded_failures = self.audio._candidate_failures(
            analysis,
            excerpt_duration_seconds=25.13,
            minimum_score=0.0,
        )
        truly_short_failures = self.audio._candidate_failures(
            analysis,
            excerpt_duration_seconds=25.20,
            minimum_score=0.0,
        )
        self.assertNotIn("audio_shorter_than_requested_excerpt", rounded_failures)
        self.assertIn("audio_shorter_than_requested_excerpt", truly_short_failures)


class AudioSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audio = load_audio_module()
        cls.selector = cls.audio.DiffusionGemmaAudioCandidateSelector()

    def select(self, **overrides):
        values = {
            "candidate_count": 2,
            "selection_mode": "auto_select",
            "expected_bpm": 90.0,
            "excerpt_duration_seconds": 4.0,
            "minimum_score": 0.52,
            "candidate_1": silent_audio(),
            "candidate_2": synthetic_song(),
        }
        values.update(overrides)
        return self.selector.select(**values)

    def test_auto_selection_returns_the_exact_pristine_audio_and_two_reports(self) -> None:
        good = synthetic_song(channels=2)
        result = self.select(candidate_2=good)
        self.assertTrue(result[6], result[5])
        self.assertIs(result[0], good)
        self.assertEqual(result[0]["waveform"].data_ptr(), good["waveform"].data_ptr())
        self.assertEqual(result[2], self.audio.waveform_sha256(good))
        director = json.loads(result[3])
        audit = json.loads(result[4])
        self.assertEqual(director["selected_candidate_index"], 2)
        self.assertEqual(audit["selected_candidate_index"], 2)
        self.assertLessEqual(len(result[3]), 12_000)
        self.assertIn("timeline", director["selected_excerpt"])
        self.assertIn("low_density_visual_recovery_intervals", director["selected_excerpt"])
        self.assertIn("vocal_active_stable_intervals", director["selected_excerpt"])

    def test_no_passing_candidate_fails_closed(self) -> None:
        result = self.select(
            candidate_count=1,
            candidate_1=silent_audio(),
            candidate_2=None,
        )
        self.assertFalse(result[6])
        self.assertIn("No decoded song candidate passed", result[5])
        self.assertIsInstance(result[0], self.audio.ExecutionBlocker)
        self.assertIsInstance(result[2], self.audio.ExecutionBlocker)
        self.assertIsInstance(result[3], self.audio.ExecutionBlocker)
        self.assertIn("minimum_score is only", result[0].message)
        self.assertEqual(result[2].message, result[0].message)
        self.assertIn("silent_or_near_silent", result[3].message)
        audit = json.loads(result[4])
        self.assertFalse(audit["ready"])
        self.assertIn("No decoded song candidate passed", audit["status"])
        self.assertIn("candidate 1:", result[5])
        self.assertIn("silent_or_near_silent", result[5])

    def test_aligned_half_time_passes_at_default_threshold_with_visible_advisory(self) -> None:
        aligned = synthetic_song(bpm=62.5)
        result = self.select(
            candidate_count=1,
            expected_bpm=124.0,
            candidate_1=aligned,
            candidate_2=None,
        )
        self.assertTrue(result[6], result[5])
        self.assertIs(result[0], aligned)
        self.assertIn("accepted half-time pulse 62.5 BPM", result[5])
        self.assertIn("canonical 125.0 BPM", result[5])
        audit = json.loads(result[4])
        self.assertEqual(
            audit["candidates"][0]["advisories"],
            ["canonically_aligned_half_time_pulse"],
        )
        self.assertEqual(
            audit["settings"]["empirical_provisional_thresholds"][
                "aligned_half_time_policy"
            ],
            "advisory_without_tempo_score_penalty",
        )
        self.assertEqual(
            audit["selection_policy_revision"],
            self.audio.SELECTION_POLICY_REVISION,
        )

    def test_low_pressure_double_time_passes_with_visible_advisory(self) -> None:
        aligned = synthetic_song(bpm=180.0)
        result = self.select(
            candidate_count=1,
            expected_bpm=90.0,
            candidate_1=aligned,
            candidate_2=None,
        )
        self.assertTrue(result[6], result[5])
        self.assertIs(result[0], aligned)
        self.assertIn("accepted low-pressure double-time subdivision", result[5])
        self.assertIn("tempo-score penalty retained", result[5])
        audit = json.loads(result[4])
        self.assertEqual(
            audit["candidates"][0]["advisories"],
            ["canonically_aligned_low_pressure_double_time_pulse"],
        )
        thresholds = audit["settings"]["empirical_provisional_thresholds"]
        self.assertEqual(
            thresholds["double_time_policy"],
            "mandatory_alignment_pressure_vocal_plus_two_of_four_quality_signals",
        )
        self.assertEqual(
            thresholds["minimum_safe_double_time_supporting_signal_count"],
            2,
        )
        self.assertEqual(
            thresholds["accepted_double_time_tempo_score_policy"],
            "retain_non_direct_penalty",
        )

    def test_live_two_candidate_double_time_audition_selects_instead_of_blocking(self) -> None:
        first_audio = synthetic_song(bpm=180.0, pitch_hz=220.0)
        second_audio = synthetic_song(bpm=180.0, pitch_hz=246.94)

        def live_like(audio, *, production, score, onsets, recovery, family, transition):
            analysis = self.audio.analyze_decoded_audio(
                audio, expected_bpm=92.0, excerpt_duration_seconds=4.0
            )
            analysis.update(
                {
                    "expected_bpm": 92.0,
                    "detected_bpm_raw": 187.5,
                    "detected_bpm": 93.75,
                    "tempo_relation": "double_time_detected",
                    "canonical_tempo_relative_error": 0.0190217391304348,
                    "production_score": production,
                }
            )
            analysis["suggested_excerpt"].update(
                {
                    "score": score,
                    "onset_density_per_second": onsets,
                    "low_density_visual_recovery_coverage": recovery,
                    "tonal_family_consistency": family,
                    "tonal_transition_stability": transition,
                    "likely_vocal_active_proxy": True,
                }
            )
            return analysis

        first = live_like(
            first_audio,
            production=0.733087918364718,
            score=0.949877043973501,
            onsets=2.96,
            recovery=0.52,
            family=1.0,
            transition=1.0,
        )
        second = live_like(
            second_audio,
            production=0.743437958264606,
            score=0.8823016780651074,
            onsets=3.52,
            recovery=1.0,
            family=0.75,
            transition=0.8333333333333333,
        )
        with mock.patch.object(
            self.audio, "analyze_decoded_audio", side_effect=[first, second]
        ):
            result = self.select(
                candidate_1=first_audio,
                candidate_2=second_audio,
                expected_bpm=92.0,
                excerpt_duration_seconds=4.0,
            )

        self.assertTrue(result[6], result[5])
        self.assertIs(result[0], second_audio)
        audit = json.loads(result[4])
        self.assertEqual(audit["selection_policy_revision"], 4)
        self.assertEqual(audit["selected_candidate_index"], 2)
        self.assertTrue(all(entry["passed"] for entry in audit["candidates"]))
        self.assertTrue(
            all(
                entry["advisories"]
                == ["canonically_aligned_low_pressure_double_time_pulse"]
                for entry in audit["candidates"]
            )
        )

    def test_auto_selection_prioritizes_the_ltx_excerpt_score(self) -> None:
        first_audio = synthetic_song(pitch_hz=220.0)
        second_audio = synthetic_song(pitch_hz=246.94)
        first = self.audio.analyze_decoded_audio(
            first_audio,
            expected_bpm=90.0,
            excerpt_duration_seconds=4.0,
        )
        second = self.audio.analyze_decoded_audio(
            second_audio,
            expected_bpm=90.0,
            excerpt_duration_seconds=4.0,
        )
        first["production_score"] = 0.95
        first["suggested_excerpt"]["score"] = 0.70
        second["production_score"] = 0.80
        second["suggested_excerpt"]["score"] = 0.96
        with mock.patch.object(
            self.audio,
            "analyze_decoded_audio",
            side_effect=[first, second],
        ):
            result = self.select(
                candidate_1=first_audio,
                candidate_2=second_audio,
                minimum_score=0.0,
            )
        self.assertTrue(result[6], result[5])
        self.assertIs(result[0], second_audio)
        audit = json.loads(result[4])
        self.assertEqual(audit["selected_candidate_index"], 2)
        self.assertIn("excerpt score 0.960", result[5])
        summaries = json.loads(result[3])["candidate_summaries"]
        self.assertEqual(summaries[0]["suggested_excerpt_score"], 0.70)
        self.assertEqual(summaries[1]["suggested_excerpt_score"], 0.96)

    def test_double_time_candidates_prioritize_visual_recovery_before_excerpt_score(self) -> None:
        first_audio = synthetic_song(bpm=180.0, pitch_hz=220.0)
        second_audio = synthetic_song(bpm=180.0, pitch_hz=246.94)
        first = self.audio.analyze_decoded_audio(
            first_audio, expected_bpm=90.0, excerpt_duration_seconds=4.0
        )
        second = self.audio.analyze_decoded_audio(
            second_audio, expected_bpm=90.0, excerpt_duration_seconds=4.0
        )
        first["suggested_excerpt"]["score"] = 0.97
        first["suggested_excerpt"]["low_density_visual_recovery_coverage"] = 0.90
        second["suggested_excerpt"]["score"] = 0.94
        second["suggested_excerpt"]["low_density_visual_recovery_coverage"] = 1.00
        with mock.patch.object(
            self.audio, "analyze_decoded_audio", side_effect=[first, second]
        ):
            result = self.select(
                candidate_1=first_audio,
                candidate_2=second_audio,
                expected_bpm=90.0,
                excerpt_duration_seconds=4.0,
                minimum_score=0.0,
            )
        self.assertTrue(result[6], result[5])
        self.assertIs(result[0], second_audio)
        audit = json.loads(result[4])
        self.assertEqual(audit["selected_candidate_index"], 2)
        self.assertEqual(
            audit["settings"]["auto_selection_order"][:2],
            [
                "tempo_safety_class_descending",
                "double_time_visual_recovery_coverage_descending",
            ],
        )
        summaries = json.loads(result[3])["candidate_summaries"]
        self.assertEqual(
            summaries[0]["suggested_excerpt_visual_recovery_coverage"], 0.90
        )
        self.assertEqual(
            summaries[1]["suggested_excerpt_visual_recovery_coverage"], 1.00
        )

    def test_excerpt_score_ties_use_song_score_then_candidate_index(self) -> None:
        first_audio = synthetic_song(pitch_hz=220.0)
        second_audio = synthetic_song(pitch_hz=246.94)
        first = self.audio.analyze_decoded_audio(
            first_audio, expected_bpm=90.0, excerpt_duration_seconds=4.0
        )
        second = self.audio.analyze_decoded_audio(
            second_audio, expected_bpm=90.0, excerpt_duration_seconds=4.0
        )
        first["suggested_excerpt"]["score"] = 0.90
        second["suggested_excerpt"]["score"] = 0.90
        first["production_score"] = 0.80
        second["production_score"] = 0.85
        with mock.patch.object(
            self.audio, "analyze_decoded_audio", side_effect=[first, second]
        ):
            result = self.select(
                candidate_1=first_audio,
                candidate_2=second_audio,
                minimum_score=0.0,
            )
        self.assertIs(result[0], second_audio)

        second["production_score"] = 0.80
        with mock.patch.object(
            self.audio, "analyze_decoded_audio", side_effect=[first, second]
        ):
            tied = self.select(
                candidate_1=first_audio,
                candidate_2=second_audio,
                minimum_score=0.0,
            )
        self.assertIs(tied[0], first_audio)

    def test_minimum_score_tooltip_explains_independent_hard_failures(self) -> None:
        spec = self.audio.DiffusionGemmaAudioCandidateSelector.INPUT_TYPES()["required"]
        tooltip = spec["minimum_score"][1]["tooltip"]
        self.assertIn("score floor only", tooltip)
        self.assertIn("still block selection", tooltip)
        tempo_tooltip = spec["expected_bpm"][1]["tooltip"]
        self.assertIn("aligned half-time", tempo_tooltip)
        self.assertIn("aligned double-time subdivision", tempo_tooltip)
        self.assertIn("other tempo aliases block", tempo_tooltip)

    def test_candidate_and_hash_locks_never_fall_back(self) -> None:
        first = synthetic_song(bpm=80.0)
        second = synthetic_song(bpm=90.0, pitch_hz=246.94)
        locked = self.select(
            selection_mode="lock_candidate_1",
            candidate_1=first,
            candidate_2=second,
            locked_start_seconds=2.0,
        )
        self.assertTrue(locked[6], locked[5])
        self.assertIs(locked[0], first)
        self.assertAlmostEqual(locked[1], 2.0, places=6)
        audit = json.loads(locked[4])
        self.assertTrue(audit["lock"]["satisfied"])
        self.assertEqual(len(audit["selection_lock_token"]), 64)
        repeated = self.select(
            selection_mode="lock_candidate_1",
            candidate_1=first,
            candidate_2=second,
            locked_start_seconds=2.0,
        )
        self.assertEqual(
            json.loads(repeated[4])["selection_lock_token"],
            audit["selection_lock_token"],
        )
        moved = self.select(
            selection_mode="lock_candidate_1",
            candidate_1=first,
            candidate_2=second,
            locked_start_seconds=3.0,
        )
        self.assertNotEqual(
            json.loads(moved[4])["selection_lock_token"],
            audit["selection_lock_token"],
        )

        by_hash = self.select(
            selection_mode="lock_by_hash",
            candidate_1=first,
            candidate_2=second,
            locked_waveform_sha256=self.audio.waveform_sha256(second),
        )
        self.assertTrue(by_hash[6], by_hash[5])
        self.assertIs(by_hash[0], second)
        missing_hash = self.select(
            selection_mode="lock_by_hash",
            candidate_1=first,
            candidate_2=second,
            locked_waveform_sha256="0" * 64,
        )
        self.assertFalse(missing_hash[6])
        self.assertIn("not present", missing_hash[5])

        failed_lock = self.select(
            candidate_count=1,
            selection_mode="lock_candidate_1",
            candidate_1=silent_audio(),
            candidate_2=None,
        )
        self.assertFalse(failed_lock[6])
        self.assertIn("silent_or_near_silent", failed_lock[5])
        self.assertIn("silent_or_near_silent", failed_lock[0].message)

    def test_lazy_candidate_surface_requests_exactly_the_effective_lanes(self) -> None:
        check = self.audio.DiffusionGemmaAudioCandidateSelector.check_lazy_status
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                connected_unresolved = {
                    f"candidate_{index}": None for index in range(1, count + 1)
                }
                self.assertEqual(
                    check(
                        candidate_count=count,
                        selection_mode="auto_select",
                        **connected_unresolved,
                    ),
                    [f"candidate_{index}" for index in range(1, count + 1)],
                )
        self.assertEqual(
            check(candidate_count=4, selection_mode="auto_select"),
            [],
        )
        self.assertEqual(
            check(
                candidate_count=4,
                selection_mode="lock_candidate_3",
                candidate_3=None,
            ),
            ["candidate_3"],
        )
        self.assertEqual(
            check(
                candidate_count=3,
                selection_mode="auto_select",
                candidate_1=object(),
                candidate_2=object(),
            ),
            [],
        )
        self.assertEqual(
            check(
                candidate_count=3,
                selection_mode="auto_select",
                candidate_1=object(),
                candidate_2=object(),
                candidate_3=None,
            ),
            ["candidate_3"],
        )
        self.assertEqual(
            check(
                candidate_count=4,
                selection_mode="lock_by_hash",
                locked_waveform_sha256="",
                candidate_1=None,
                candidate_2=None,
                candidate_3=None,
                candidate_4=None,
            ),
            [],
        )
        self.assertEqual(
            check(
                candidate_count=2,
                selection_mode="lock_by_hash",
                locked_waveform_sha256="a" * 64,
                candidate_1=None,
                candidate_2=None,
            ),
            ["candidate_1", "candidate_2"],
        )
        out_of_range = self.select(
            candidate_count=1,
            selection_mode="lock_candidate_2",
            candidate_1=synthetic_song(),
            candidate_2=None,
        )
        self.assertFalse(out_of_range[6])
        self.assertIn("outside candidate_count=1", out_of_range[5])

    def test_four_candidate_compact_report_is_director_safe(self) -> None:
        package, director_nodes = load_package_modules()
        source = synthetic_song(duration_seconds=24.0)
        result = self.selector.select(
            candidate_count=4,
            selection_mode="auto_select",
            expected_bpm=90.0,
            excerpt_duration_seconds=20.0,
            minimum_score=0.52,
            candidate_1=source,
            candidate_2=source,
            candidate_3=source,
            candidate_4=source,
        )
        self.assertTrue(result[6], result[5])
        self.assertLessEqual(len(result[3]), 12_000)
        self.assertGreater(len(result[4]), len(result[3]))
        normalized = director_nodes._normalize_measured_audio_report_json(result[3])
        self.assertEqual(normalized["selected_candidate_index"], 1)
        self.assertIn("selected_excerpt", normalized)
        del package


class OrchestrationAndGuideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audio = load_audio_module()

    def test_soundtrack_source_router_contract_and_true_lazy_boundaries(self) -> None:
        node = self.audio.DiffusionGemmaSongSourceRouter()
        inputs = node.INPUT_TYPES()
        source_contract = inputs["required"]["source_mode"]
        self.assertEqual(source_contract[0], ["Generate with ACE-Step", "Upload song"])
        self.assertEqual(source_contract[1]["default"], "Generate with ACE-Step")
        self.assertEqual(inputs["required"]["uploaded_expected_bpm"][1]["default"], 0.0)
        self.assertTrue(inputs["required"]["uploaded_lyrics"][1]["multiline"])
        for name in (
            "generated_candidate_count",
            "generated_expected_bpm",
            "generated_lyrics",
            "generated_duration_seconds",
            "generated_candidate_1",
            "generated_candidate_2",
            "generated_candidate_3",
            "generated_candidate_4",
            "uploaded_audio",
            "uploaded_duration_seconds",
            "uploaded_waveform_sha256",
            "uploaded_status",
            "uploaded_ready",
        ):
            self.assertTrue(inputs["optional"][name][1]["lazy"], name)

        upload_pending = node.check_lazy_status(
            source_mode="Upload song",
            generated_candidate_count=None,
            generated_candidate_1=None,
            uploaded_audio=None,
            uploaded_duration_seconds=None,
            uploaded_waveform_sha256=None,
            uploaded_status=None,
            uploaded_ready=None,
        )
        self.assertEqual(
            upload_pending,
            [
                "uploaded_audio",
                "uploaded_duration_seconds",
                "uploaded_waveform_sha256",
                "uploaded_status",
                "uploaded_ready",
            ],
        )
        generated_metadata_pending = node.check_lazy_status(
            source_mode="Generate with ACE-Step",
            generated_candidate_count=None,
            generated_expected_bpm=None,
            generated_lyrics=None,
            generated_duration_seconds=None,
            uploaded_audio=None,
        )
        self.assertEqual(
            generated_metadata_pending,
            [
                "generated_candidate_count",
                "generated_expected_bpm",
                "generated_lyrics",
                "generated_duration_seconds",
            ],
        )
        generated_candidates_pending = node.check_lazy_status(
            source_mode="Generate with ACE-Step",
            generated_candidate_count=2,
            generated_expected_bpm=92.0,
            generated_lyrics="line",
            generated_duration_seconds=90.0,
            generated_candidate_1=None,
            generated_candidate_2=None,
            generated_candidate_3=None,
            generated_candidate_4=None,
            uploaded_audio=None,
        )
        self.assertEqual(
            generated_candidates_pending,
            ["generated_candidate_1", "generated_candidate_2"],
        )

    def test_soundtrack_source_router_upload_preserves_exact_waveform_and_metadata(self) -> None:
        node = self.audio.DiffusionGemmaSongSourceRouter()
        uploaded = synthetic_song(duration_seconds=7.25, sample_rate=8_000)
        generated = synthetic_song(duration_seconds=9.0, pitch_hz=330.0)
        digest = self.audio.waveform_sha256(uploaded)
        result = node.route(
            "Upload song",
            0.0,
            "  First real line\nSecond real line  ",
            generated_candidate_count=4,
            generated_expected_bpm=180.0,
            generated_lyrics="must not leak",
            generated_duration_seconds=90.0,
            generated_candidate_1=generated,
            uploaded_audio=uploaded,
            uploaded_duration_seconds=7.25,
            uploaded_waveform_sha256=digest,
            uploaded_status="decoded",
            uploaded_ready=True,
        )
        for index in range(4):
            self.assertIs(result[index], uploaded)
        self.assertEqual(result[4], 1)
        self.assertEqual(result[5], 0.0)
        self.assertEqual(result[6], "First real line\nSecond real line")
        self.assertAlmostEqual(result[7], 7.25)
        self.assertEqual(result[8], "uploaded_song")
        self.assertIn("ACE audio generation is dormant", result[9])
        self.assertTrue(result[10])
        self.assertNotIn("must not leak", result)

    def test_soundtrack_source_router_ace_preserves_requested_candidate_lanes(self) -> None:
        node = self.audio.DiffusionGemmaSongSourceRouter()
        first = synthetic_song(pitch_hz=220.0)
        second = synthetic_song(pitch_hz=246.94)
        ignored_upload = synthetic_song(pitch_hz=392.0)
        result = node.route(
            "Generate with ACE-Step",
            0.0,
            "uploaded text must not leak",
            generated_candidate_count=2,
            generated_expected_bpm=94.0,
            generated_lyrics="ACE lyric",
            generated_duration_seconds=90.0,
            generated_candidate_1=first,
            generated_candidate_2=second,
            uploaded_audio=ignored_upload,
        )
        self.assertIs(result[0], first)
        self.assertIs(result[1], second)
        self.assertIs(result[2], first)
        self.assertIs(result[3], first)
        self.assertEqual(result[4:9], (2, 94.0, "ACE lyric", 90.0, "ace_step"))
        self.assertIn("upload loader is dormant", result[9])
        self.assertTrue(result[10])

    def test_soundtrack_source_router_fails_clearly_for_bad_selected_source(self) -> None:
        node = self.audio.DiffusionGemmaSongSourceRouter()
        with self.assertRaisesRegex(ValueError, "no song is connected"):
            node.route("Upload song", 0.0, "")
        uploaded = synthetic_song(duration_seconds=5.0)
        with self.assertRaisesRegex(ValueError, "duration metadata"):
            node.route(
                "Upload song",
                0.0,
                "",
                uploaded_audio=uploaded,
                uploaded_duration_seconds=7.0,
            )
        with self.assertRaisesRegex(ValueError, "candidate lanes: 2"):
            node.route(
                "Generate with ACE-Step",
                0.0,
                "",
                generated_candidate_count=2,
                generated_expected_bpm=90.0,
                generated_lyrics="",
                generated_duration_seconds=30.0,
                generated_candidate_1=uploaded,
            )
        with self.assertRaisesRegex(ValueError, "does not match its decoded audio"):
            node.route(
                "Upload song",
                0.0,
                "",
                uploaded_audio=uploaded,
                uploaded_duration_seconds=5.0,
                uploaded_waveform_sha256="0" * 64,
                uploaded_ready=True,
            )

    def test_uploaded_song_is_source_locked_while_technical_integrity_stays_hard(self) -> None:
        selector = self.audio.DiffusionGemmaAudioCandidateSelector()
        source = synthetic_song(duration_seconds=30.0)
        analysis = self.audio.analyze_decoded_audio(
            source,
            expected_bpm=0.0,
            excerpt_duration_seconds=25.0,
        )
        analysis.update(
            {
                "tonal_window_stability": 0.10,
                "onset_density_per_second": 9.0,
                "tempo_relation": "half_time_detected",
                "canonical_tempo_relative_error": None,
                "tonal_window_count": 5,
                "remote_tonal_jump_rate": 0.75,
                "tonal_transition_stability": 0.20,
                "spectral_flatness_mean": 0.10,
                "tonal_clarity": 0.30,
                "production_score": 0.10,
            }
        )
        analysis["suggested_excerpt"].update(
            {
                "onset_density_per_second": 5.5,
                "likely_vocal_active_proxy": False,
                "confident_key_transition_window_count": 5,
                "tonal_family_consistency": 0.20,
                "remote_tonal_jump_rate": 0.75,
                "parallel_major_minor_flip_rate": 0.75,
                "tonal_transition_stability": 0.20,
            }
        )
        with mock.patch.object(
            self.audio,
            "analyze_decoded_audio",
            return_value=analysis,
        ):
            uploaded = selector.select(
                1,
                "auto_select",
                0.0,
                25.0,
                0.52,
                candidate_1=source,
                source_policy="uploaded_song",
            )
            generated = selector.select(
                1,
                "auto_select",
                0.0,
                25.0,
                0.52,
                candidate_1=source,
                source_policy="ace_step",
            )
        self.assertTrue(uploaded[6], uploaded[5])
        self.assertIs(uploaded[0], source)
        report = json.loads(uploaded[4])
        self.assertEqual(report["settings"]["source_policy"], "uploaded_song")
        advisories = report["candidates"][0]["advisories"]
        self.assertIn("source_locked_uploaded_song", advisories)
        self.assertIn("source_locked_advisory_no_vocal_proxy", advisories)
        self.assertIn("source_locked_advisory_signal_only_tempo_interpretation", advisories)
        self.assertIn("source_locked_advisory_below_generated_candidate_score", advisories)
        self.assertIn("passed technical integrity", uploaded[5])
        self.assertFalse(generated[6])

        clipped = copy.deepcopy(analysis)
        clipped["clipped_sample_fraction"] = 0.50
        failures = self.audio._candidate_failures(
            clipped,
            excerpt_duration_seconds=25.0,
            minimum_score=0.52,
            source_policy="uploaded_song",
        )
        self.assertEqual(failures, ["excessive_clipping"])

    def test_signal_only_bpm_does_not_penalize_canonical_pulse_choice(self) -> None:
        source = synthetic_song(bpm=180.0, duration_seconds=30.0)
        analysis = self.audio.analyze_decoded_audio(
            source,
            expected_bpm=0.0,
            excerpt_duration_seconds=25.0,
        )
        self.assertGreaterEqual(analysis["production_score"], 0.0)
        failures = self.audio._candidate_failures(
            analysis,
            excerpt_duration_seconds=25.0,
            minimum_score=0.0,
            source_policy="uploaded_song",
        )
        self.assertNotIn("tempo_undetected_or_unrelated", failures)
        self.assertNotIn("double_time_outside_low_pressure_contract", failures)
        self.assertNotIn("half_time_outside_canonical_alignment_contract", failures)

    def test_upload_song_loader_is_blank_safe_and_returns_exact_decoded_audio(self) -> None:
        loader = self.audio.DiffusionGemmaUploadSong()
        contract = loader.INPUT_TYPES()["required"]["audio_file"]
        self.assertEqual(contract[0][0], "")
        self.assertTrue(contract[1]["audio_upload"])
        self.assertTrue(loader.VALIDATE_INPUTS(""))

        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory)
            song_path = input_dir / "song.wav"
            song_path.write_bytes(b"fixture audio bytes")
            folder_paths = types.ModuleType("folder_paths")
            folder_paths.annotated_filepath = lambda value: (str(value), None)
            folder_paths.get_input_directory = lambda: str(input_dir)
            waveform = torch.linspace(-0.5, 0.5, 16_000, dtype=torch.float32).reshape(1, -1)
            with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}), mock.patch.object(
                self.audio,
                "_load_uploaded_audio_bounded",
                return_value=(waveform, 8_000),
            ):
                fingerprint = loader.IS_CHANGED("song.wav")
                self.assertTrue(fingerprint.startswith("stat@1:song.wav:"), fingerprint)
                self.assertIn(f":{song_path.stat().st_size}:", fingerprint)
                result = loader.load_audio("song.wav")
            self.assertEqual(result[0]["waveform"].shape, (1, 1, 16_000))
            self.assertEqual(result[0]["sample_rate"], 8_000)
            self.assertEqual(result[1], 2.0)
            self.assertEqual(result[2], self.audio.waveform_sha256(result[0]))
            self.assertIn("2.000s", result[3])
            self.assertTrue(result[4])

    def test_upload_song_path_and_decode_limits_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (output_dir / "outside.wav").write_bytes(b"outside")
            folder_paths = types.ModuleType("folder_paths")

            def annotated(value):
                text = str(value)
                if text.endswith("[output]"):
                    return text[:-8].strip(), str(output_dir)
                return text, None

            folder_paths.annotated_filepath = annotated
            folder_paths.get_input_directory = lambda: str(input_dir)
            with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
                with self.assertRaisesRegex(ValueError, "input directory"):
                    self.audio._resolve_uploaded_song_path("outside.wav [output]")
                with self.assertRaisesRegex(ValueError, "input directory"):
                    self.audio._resolve_uploaded_song_path("../output/outside.wav")
                with self.assertRaisesRegex(ValueError, "Choose or upload"):
                    self.audio._resolve_uploaded_song_path("missing.wav")

        with self.assertRaisesRegex(ValueError, "600-second"):
            self.audio._validate_uploaded_decode_shape(2, 48_000 * 601, 48_000)
        with self.assertRaisesRegex(ValueError, "1 to 8"):
            self.audio._validate_uploaded_decode_shape(9, 48_000, 48_000)
        with self.assertRaisesRegex(ValueError, "decoded-PCM"):
            self.audio._validate_uploaded_decode_shape(8, 10_000_001, 48_000)

    def test_shared_production_concept_controls_effective_candidate_count(self) -> None:
        node = self.audio.DiffusionGemmaMusicProductionConcept()
        planner_combo = list(self.audio.PRODUCTION_CONCEPTS)
        output_types = node.RETURN_TYPES
        # Concrete list-valued outputs remain compatible with the Planner's
        # legacy combo and Comfy's V3 COMBO input without importing mutable
        # global Comfy test stubs into this isolated unit suite.
        self.assertEqual(output_types[0], planner_combo)
        self.assertEqual(output_types[1], planner_combo)
        for concept in ("Source passthrough", "Joint video-safe plan"):
            self.assertEqual(node.route(concept, 4), (concept, concept, 1))
        self.assertEqual(
            node.route("Audition and select", 3),
            ("Audition and select", "Audition and select", 3),
        )

    def test_song_seed_fanout_matches_fixed_splatstage_vectors(self) -> None:
        fanout = self.audio.DiffusionGemmaSongSeedFanout()
        source = fanout.fanout(26_073_001, "Source passthrough")
        self.assertEqual(source[:4], (6_862_475_389_844_981_512,) * 4)
        source_report = json.loads(source[4])
        self.assertEqual(source_report["candidate_seeds"], [6_862_475_389_844_981_512])
        audition = fanout.fanout(26_073_001, "Audition and select")
        expected = (
            1_347_126_533_689_113_756,
            1_268_862_306_067_732_500,
            7_666_022_862_567_662_684,
            52_859_412_772_563_515,
        )
        self.assertEqual(audition[:4], expected)
        report = json.loads(audition[4])
        self.assertEqual(report["candidate_seeds"], list(expected))
        for index, actual in enumerate(expected):
            payload = f"splatstage-autodirect@1|26073001|music:audition:{index}|0"
            derived = int.from_bytes(
                hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(),
                "big",
            ) & 0x7FFF_FFFF_FFFF_FFFF
            self.assertEqual(actual, derived)

    def test_ltx_guide_preserves_master_and_stem_branch_is_truly_lazy(self) -> None:
        node = self.audio.DiffusionGemmaLTXAudioGuide()
        final = synthetic_song(channels=2)
        full = node.build(final, "full_mix")
        self.assertTrue(full[5], full[4])
        self.assertIs(full[0], final)
        self.assertIs(full[1], final)
        self.assertEqual(
            node.check_lazy_status(final_audio=final, mode="full_mix"),
            [],
        )
        self.assertEqual(
            node.check_lazy_status(final_audio=final, mode="sync_safe"),
            [],
        )
        self.assertEqual(
            node.check_lazy_status(final_audio=final, mode="vocal_only"),
            [],
        )
        self.assertEqual(
            node.check_lazy_status(final_audio=final, mode="vocal_only", vocal_stem=None),
            ["vocal_stem"],
        )

        sync = node.build(final, "sync_safe", vocal_stem=None)
        self.assertTrue(sync[5], sync[4])
        self.assertIs(sync[1], final)
        self.assertEqual(sync[0]["waveform"].shape, final["waveform"].shape)
        self.assertEqual(sync[0]["sample_rate"], final["sample_rate"])
        self.assertFalse(json.loads(sync[3])["processing"]["used_vocal_stem"])

        blocked = node.build(final, "vocal_only")
        self.assertFalse(blocked[5])
        self.assertIn("requires a connected vocal stem", blocked[4])
        short_stem = synthetic_song(
            duration_seconds=6.0,
            sample_rate=4_000,
            channels=1,
            pitch_hz=246.94,
        )
        vocal = node.build(final, "vocal_only", vocal_stem=short_stem)
        self.assertTrue(vocal[5], vocal[4])
        self.assertIs(vocal[1], final)
        self.assertEqual(vocal[0]["waveform"].shape, final["waveform"].shape)
        self.assertEqual(vocal[0]["sample_rate"], final["sample_rate"])

    def test_ltx_performance_prompt_contract_and_dance_mode(self) -> None:
        node = self.audio.DiffusionGemmaLTXPerformancePrompt()
        inputs = node.INPUT_TYPES()
        self.assertEqual(
            inputs["required"]["performance_mode"][0],
            [
                "Dance / music sync",
                "Lyrics + lip sync",
                "Natural / audio-led sync",
            ],
        )
        self.assertEqual(
            inputs["required"]["performance_mode"][1]["default"],
            "Natural / audio-led sync",
        )
        self.assertEqual(
            inputs["optional"]["performance_mode_override"][0],
            "STRING",
        )
        self.assertTrue(
            inputs["optional"]["performance_mode_override"][1]["forceInput"]
        )
        self.assertEqual(
            node.RETURN_NAMES,
            (
                "ltx_prompt",
                "selected_lyrics",
                "status",
                "performance_report_json",
                "performance_mode",
            ),
        )

        incoming = "Validated Director caption with trailing whitespace.  "
        secret_lyrics = "[Verse]\nThese words must never enter dance mode"
        result = node.apply(
            incoming,
            "Dance / music sync",
            90.0,
            24.0,
            20.0,
            secret_lyrics,
        )
        self.assertTrue(result[0].startswith(incoming))
        self.assertEqual(result[1], "")
        self.assertNotIn("These words must never enter dance mode", result[0])
        for phrase in ("no visible person sings", "speaks", "mouths words", "lip-syncs"):
            self.assertIn(phrase, result[0])
        self.assertIn("full-body choreography", result[0])
        report = json.loads(result[3])
        self.assertTrue(report["ready"])
        self.assertFalse(report["lyrics_included"])
        self.assertTrue(report["incoming_prompt_preserved_verbatim"])
        self.assertFalse(report["audio_conditioning_changed"])
        self.assertFalse(report["final_soundtrack_changed"])
        self.assertEqual(report["selected_lyrics_role"], "none")
        self.assertFalse(report["selected_lyrics_are_timing_schedule"])
        self.assertEqual(report["articulation_timing_authority"], "dance_mode_suppression")
        self.assertEqual(result[4], "Dance / music sync")

        overridden = node.apply(
            incoming,
            "Dance / music sync",
            90.0,
            24.0,
            20.0,
            "[Verse]\nSing this exact line",
            "Lyrics + lip sync",
        )
        self.assertEqual(overridden[4], "Lyrics + lip sync")
        self.assertIn("Sing this exact line", overridden[1])

        for alias in (
            "natural",
            "audio_led_sync",
            "audio-led sync",
            "Natural / audio-led sync",
        ):
            self.assertEqual(
                self.audio._normalize_ltx_performance_mode(alias),
                "Natural / audio-led sync",
            )

    def test_ltx_performance_prompt_natural_mode_is_neutral_and_audio_led(self) -> None:
        node = self.audio.DiffusionGemmaLTXPerformancePrompt()
        incoming = "Validated Director caption."
        secret_lyrics = "Words that must not become a performance schedule"
        result = node.apply(
            incoming,
            "Natural / audio-led sync",
            90.0,
            24.0,
            20.0,
            secret_lyrics,
        )

        self.assertEqual(result[1], "")
        self.assertEqual(result[4], "Natural / audio-led sync")
        self.assertTrue(result[0].startswith(incoming))
        self.assertNotIn(secret_lyrics, result[0])
        self.assertIn("connected audio alone govern", result[0])
        self.assertIn("never invent or pantomime words", result[0])
        self.assertNotIn("lips gently closed", result[0])
        self.assertNotIn("Candidate authored lyric lines", result[0])
        report = json.loads(result[3])
        self.assertTrue(report["ready"])
        self.assertFalse(report["lyrics_included"])
        self.assertEqual(report["selected_lyrics_role"], "none")
        self.assertFalse(report["selected_lyrics_are_timing_schedule"])
        self.assertFalse(report["selected_lyrics_require_every_line_performed"])
        self.assertEqual(report["articulation_timing_authority"], "connected_audio_alone")
        self.assertEqual(report["unverified_timing_fallback"], "Natural / audio-led sync")
        self.assertEqual(
            report["non_vocal_or_ambiguous_gap_policy"],
            "no_invented_or_pantomimed_words",
        )

    def test_music_video_performance_mode_is_pre_director_and_audio_neutral(self) -> None:
        node = self.audio.DiffusionGemmaMusicVideoPerformanceMode()
        mode_contract = node.INPUT_TYPES()["required"]["performance_mode"]
        self.assertEqual(mode_contract[0], list(self.audio.LTX_PERFORMANCE_MODES))
        self.assertEqual(mode_contract[1]["default"], "Natural / audio-led sync")
        dance = node.compile("Preserve the locked song.", "Dance / music sync")
        self.assertEqual(dance[0], "Dance / music sync")
        self.assertIn("Preserve the locked song.", dance[1])
        self.assertIn("no visible person sings", dance[1])
        self.assertIn("Do not author H3 <d>", dance[1])

        natural = node.compile("Preserve the locked song.", "Natural / audio-led sync")
        self.assertEqual(natural[0], "Natural / audio-led sync")
        self.assertIn("connected audio alone governs", natural[1])
        self.assertIn("no invented or pantomimed words", natural[1])
        self.assertNotIn("lips gently closed", natural[1])
        self.assertNotIn("Reserve one", natural[1])

        lyrics = node.compile("Preserve the locked song.", "Lyrics + lip sync")
        self.assertEqual(lyrics[0], "Lyrics + lip sync")
        self.assertNotIn("Reserve one", lyrics[1])
        self.assertNotIn("unobscured, readable face", lyrics[1])
        self.assertIn("do not author shot-level or timestamped vocal actions", lyrics[1])
        self.assertIn("Downstream verified vocal timing is the sole authority", lyrics[1])
        self.assertIn("lexical/pronunciation reference only", lyrics[1])
        self.assertIn("fall back to Natural / audio-led sync", lyrics[1])
        self.assertIn("no invented or pantomimed words", lyrics[1])
        self.assertIn("audio is unchanged", lyrics[2])

    def test_ltx_performance_prompt_selects_exact_bounded_lyric_window(self) -> None:
        node = self.audio.DiffusionGemmaLTXPerformancePrompt()
        lyrics = (
            "[Verse]\n"
            "First copper sunrise\n"
            "Second highway turning\n"
            "Third old radio calling\n"
            "Fourth white line burning\n"
            "[Chorus]\n"
            "Fifth come home tonight\n"
            "Sixth hold on tight\n"
            "Seventh stars are falling\n"
            "Eighth hear me calling"
        )
        result = node.apply(
            "A single lead performer remains clearly visible.",
            "Lyrics + lip sync",
            80.0,
            20.0,
            20.0,
            lyrics,
        )
        expected = (
            "Second highway turning\n"
            "Third old radio calling\n"
            "Fourth white line burning\n"
            "Fifth come home tonight"
        )
        self.assertEqual(result[1], expected)
        quoted_expected = "\n".join(
            json.dumps(line) for line in expected.splitlines()
        )
        self.assertIn(
            "Candidate authored lyric lines (lexical/pronunciation reference only):\n"
            + quoted_expected,
            result[0],
        )
        self.assertNotIn("[Verse]", result[0])
        self.assertNotIn("[Chorus]", result[0])
        self.assertNotIn("First copper sunrise", result[0])
        self.assertNotIn("Sixth hold on tight", result[0])
        self.assertIn("Candidate text never overrides the audio", result[0])
        self.assertIn("downstream verified vocal timing", result[0])
        self.assertIn("do not require every supplied line to be performed", result[0])
        self.assertIn("instrumental, non-vocal, or ambiguous gaps", result[0])
        self.assertIn("never invent or pantomime words", result[0])
        report = json.loads(result[3])
        window = report["lyric_window"]
        self.assertEqual(window["selected_line_range_1_based"], [2, 5])
        self.assertEqual(window["selected_line_count"], 4)
        self.assertEqual(
            window["alignment"],
            "deterministic_planning_heuristic_not_asr_or_word_alignment",
        )
        self.assertTrue(window["section_directives_excluded"])
        self.assertEqual(
            window["selection_role"],
            "lexical_and_pronunciation_candidates_only",
        )
        self.assertFalse(window["is_timing_schedule"])
        self.assertFalse(window["requires_every_selected_line_performed"])
        self.assertTrue(report["lyric_lines_supplied_in_straight_quotes"])
        self.assertEqual(
            report["selected_lyrics_role"],
            "lexical_and_pronunciation_candidates_only",
        )
        self.assertFalse(report["selected_lyrics_are_timing_schedule"])
        self.assertFalse(report["selected_lyrics_require_every_line_performed"])
        self.assertEqual(
            report["articulation_timing_authority"],
            "downstream_verified_vocal_timing_then_connected_audio",
        )
        self.assertEqual(report["unverified_timing_fallback"], "Natural / audio-led sync")
        self.assertEqual(
            report["non_vocal_or_ambiguous_gap_policy"],
            "no_invented_or_pantomimed_words",
        )
        self.assertIn("never a schedule or performance checklist", result[2])
        self.assertEqual(result[4], "Lyrics + lip sync")

    def test_ltx_performance_prompt_fails_clearly_without_sung_lines(self) -> None:
        node = self.audio.DiffusionGemmaLTXPerformancePrompt()
        with self.assertRaisesRegex(ValueError, "requires at least one complete sung lyric line"):
            node.apply(
                "Validated prompt.",
                "Lyrics + lip sync",
                90.0,
                0.0,
                20.0,
                "[Instrumental]\n[Outro]",
            )
        with self.assertRaisesRegex(ValueError, "speech budget"):
            node.apply(
                "Validated prompt.",
                "Lyrics + lip sync",
                90.0,
                0.0,
                1.0,
                "This complete lyric line cannot fit",
            )

    def test_ltx_performance_prompt_keeps_empty_generation_gate_closed(self) -> None:
        node = self.audio.DiffusionGemmaLTXPerformancePrompt()
        result = node.apply(
            " \n",
            "Lyrics + lip sync",
            90.0,
            0.0,
            20.0,
            "",
        )
        self.assertEqual(result[0], "")
        self.assertEqual(result[1], "")
        report = json.loads(result[3])
        self.assertFalse(report["ready"])
        self.assertEqual(report["failure_reason"], "empty_upstream_prompt")

    def test_ace_cover_conditioning_is_lazy_and_fails_closed(self) -> None:
        mode = self.audio.DiffusionGemmaACEReferenceMode()
        self.assertEqual(mode.route("Compose new"), (True, "compose_new"))
        self.assertEqual(mode.route("Cover reference"), (False, "cover_reference"))
        gate = self.audio.DiffusionGemmaACECoverConditioning()
        conditioning = [["embedding", {}]]
        self.assertEqual(
            gate.check_lazy_status(conditioning=conditioning, mode_token="compose_new"),
            [],
        )
        self.assertEqual(
            gate.check_lazy_status(conditioning=conditioning, mode_token="cover_reference"),
            [],
        )
        self.assertEqual(
            gate.check_lazy_status(
                conditioning=conditioning,
                mode_token="cover_reference",
                reference_latent=None,
            ),
            ["reference_latent"],
        )
        compose = gate.apply(conditioning, "compose_new")
        self.assertTrue(compose[2])
        self.assertIs(compose[0], conditioning)
        missing = gate.apply(conditioning, "cover_reference")
        self.assertFalse(missing[2])

        calls = []

        def conditioning_set_values(value, additions, append=False):
            calls.append((value, additions, append))
            return ["routed", additions]

        helper = types.SimpleNamespace(conditioning_set_values=conditioning_set_values)
        latent = {"samples": torch.ones((1, 8, 4), dtype=torch.float32)}
        with mock.patch.dict(sys.modules, {"node_helpers": helper}):
            covered = gate.apply(conditioning, "cover_reference", reference_latent=latent)
        self.assertTrue(covered[2], covered[1])
        self.assertEqual(calls[0][1]["reference_audio_timbre_latents"][0].data_ptr(), latent["samples"].data_ptr())
        self.assertTrue(calls[0][2])

    def test_audio_nodes_are_registered_through_package_surface(self) -> None:
        package, _nodes = load_package_modules()
        expected = {
            "DiffusionGemmaAudioCandidateSelector",
            "DiffusionGemmaUploadSong",
            "DiffusionGemmaSongSourceRouter",
            "DiffusionGemmaLTXAudioGuide",
            "DiffusionGemmaLTXPerformancePrompt",
            "DiffusionGemmaMusicVideoPerformanceMode",
            "DiffusionGemmaACEReferenceMode",
            "DiffusionGemmaACECoverConditioning",
            "DiffusionGemmaSongSeedFanout",
            "DiffusionGemmaMusicProductionConcept",
        }
        self.assertTrue(expected.issubset(package.NODE_CLASS_MAPPINGS))
        self.assertTrue(expected.issubset(package.NODE_DISPLAY_NAME_MAPPINGS))


@unittest.skipUnless(
    shutil.which("ffmpeg") and BAD_REAL_OUTPUT.exists() and GOOD_REAL_OUTPUT.exists(),
    "known local 00086/00087 output pair is unavailable",
)
class RealOutputRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audio = load_audio_module()

    def test_known_bad_00086_fails_while_known_good_00087_passes(self) -> None:
        bad = self.audio.analyze_decoded_audio(
            decode_mp4_audio(BAD_REAL_OUTPUT),
            expected_bpm=85.0,
            excerpt_duration_seconds=20.0,
        )
        good = self.audio.analyze_decoded_audio(
            decode_mp4_audio(GOOD_REAL_OUTPUT),
            expected_bpm=128.0,
            excerpt_duration_seconds=20.0,
        )
        bad_failures = self.audio._candidate_failures(
            bad,
            excerpt_duration_seconds=20.0,
            minimum_score=0.52,
        )
        good_failures = self.audio._candidate_failures(
            good,
            excerpt_duration_seconds=20.0,
            minimum_score=0.52,
        )
        self.assertIn("double_time_outside_low_pressure_contract", bad_failures)
        self.assertIn("selected_excerpt_tonal_family_inconsistent", bad_failures)
        self.assertEqual(good["tempo_relation"], "direct")
        self.assertEqual(good_failures, [])
        self.assertLess(bad["production_score"], good["production_score"])


if __name__ == "__main__":
    unittest.main()
