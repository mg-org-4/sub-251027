from __future__ import annotations

import json
import math
import unittest
from unittest.mock import patch

import torch

import advertising_audio_nodes as audio
import advertising_contract_nodes as contracts


def soundtrack_contract(content_mode="Instrumental", voice_over_policy="None"):
    lyrics = "[Verse]\nA real sung line." if content_mode == "Vocal" else ""
    language = "en" if content_mode == "Vocal" else "unknown"
    return contracts.DiffusionGemmaAdvertisementSoundtrackContract().build(
        content_mode=content_mode,
        genre_style="percussive alternative pop",
        mood="bright",
        instrumentation="bass, drums, handclaps",
        target_duration_seconds=2.0,
        bpm=105.0,
        time_signature="4",
        language=language,
        lyrics=lyrics,
        voice_over_policy=voice_over_policy,
        arrangement_notes="One complete two-second test arc.",
    )[0]


def fake_audio(value=0.1, seconds=3.0, sample_rate=1000):
    waveform = torch.full((1, 1, int(seconds * sample_rate)), float(value))
    return {"waveform": waveform, "sample_rate": sample_rate}


def fake_analysis(candidate, vocal_proxy=False):
    return {
        "waveform_sha256": audio.legacy_audio.waveform_sha256(candidate),
        "duration_seconds": 3.0,
        "production_score": 0.9,
        "tempo_relation": "direct",
        "expected_bpm": 105.0,
        "detected_bpm": 105.0,
        "canonical_tempo_relative_error": 0.0,
        "suggested_excerpt": {
            "start_seconds": 0.0,
            "score": 0.9,
            "likely_vocal_active_proxy": bool(vocal_proxy),
            "onset_density_per_second": 1.0,
            "low_density_visual_recovery_coverage": 1.0,
            "tonal_family_consistency": 1.0,
            "tonal_transition_stability": 1.0,
        },
    }


class AdvertisementAudioSelectorTests(unittest.TestCase):
    def select(self, contract, candidate, **overrides):
        values = {
            "soundtrack_contract_json": contract,
            "candidate_count": 1,
            "selection_mode": "auto_select",
            "expected_bpm": 105.0,
            "excerpt_duration_seconds": 2.0,
            "minimum_score": 0.5,
            "candidate_1": candidate,
            "source_policy": "minimax_music3",
        }
        values.update(overrides)
        return audio.DiffusionGemmaAdvertisementAudioCandidateSelector().select(**values)

    def test_generated_instrumental_does_not_require_vocal_proxy(self):
        candidate = fake_audio()
        analysis = fake_analysis(candidate, vocal_proxy=False)
        with patch.object(audio.legacy_audio, "analyze_decoded_audio", return_value=analysis), patch.object(
            audio.legacy_audio,
            "_candidate_failures",
            return_value=["no_non_silent_vocal_active_proxy_window"],
        ), patch.object(audio.legacy_audio, "_candidate_advisories", return_value=[]):
            result = self.select(soundtrack_contract("Instrumental"), candidate)
        self.assertTrue(result[-1])
        report = json.loads(result[4])
        self.assertEqual(report["settings"]["source_policy"], "minimax_music3")
        self.assertIn(
            "instrumental_contract_vocal_proxy_not_required",
            report["candidates"][0]["advisories"],
        )

    def test_requested_tempo_mismatch_is_reported_even_when_octave_relation_is_direct(self):
        candidate = fake_audio()
        analysis = fake_analysis(candidate, vocal_proxy=False)
        analysis.update(
            {
                "expected_bpm": 122.0,
                "detected_bpm": 87.890625,
                "canonical_tempo_relative_error": abs(87.890625 - 122.0) / 122.0,
            }
        )
        with patch.object(audio.legacy_audio, "analyze_decoded_audio", return_value=analysis), patch.object(
            audio.legacy_audio,
            "_candidate_failures",
            return_value=[],
        ), patch.object(audio.legacy_audio, "_candidate_advisories", return_value=[]):
            result = self.select(soundtrack_contract("Instrumental"), candidate, expected_bpm=122.0)
        self.assertTrue(result[-1])
        candidate_report = json.loads(result[4])["candidates"][0]
        self.assertIn("requested_tempo_mismatch_signal_estimate", candidate_report["advisories"])
        self.assertFalse(candidate_report["analysis"]["requested_tempo_match_within_10_percent"])
        self.assertAlmostEqual(candidate_report["analysis"]["expected_bpm"], 122.0)

    def test_vocal_mode_requires_proxy_and_instrumental_keeps_other_failures(self):
        candidate = fake_audio()
        analysis = fake_analysis(candidate, vocal_proxy=False)
        with patch.object(audio.legacy_audio, "analyze_decoded_audio", return_value=analysis), patch.object(
            audio.legacy_audio,
            "_candidate_failures",
            return_value=["no_non_silent_vocal_active_proxy_window"],
        ), patch.object(audio.legacy_audio, "_candidate_advisories", return_value=[]):
            vocal = self.select(soundtrack_contract("Vocal"), candidate)
        self.assertFalse(vocal[-1])
        self.assertIn("no_non_silent_vocal", vocal[-2])

        with patch.object(audio.legacy_audio, "analyze_decoded_audio", return_value=analysis), patch.object(
            audio.legacy_audio,
            "_candidate_failures",
            return_value=["no_non_silent_vocal_active_proxy_window", "excessive_clipping"],
        ), patch.object(audio.legacy_audio, "_candidate_advisories", return_value=[]):
            instrumental = self.select(soundtrack_contract("Instrumental"), candidate)
        self.assertFalse(instrumental[-1])
        self.assertIn("excessive_clipping", instrumental[-2])

    def test_hash_lock_selects_exact_candidate_without_fallback(self):
        first = fake_audio(0.05)
        second = fake_audio(0.1)

        def analyze(candidate, **_kwargs):
            return fake_analysis(candidate, vocal_proxy=False)

        lock_hash = audio.legacy_audio.waveform_sha256(second)
        with patch.object(audio.legacy_audio, "analyze_decoded_audio", side_effect=analyze), patch.object(
            audio.legacy_audio,
            "_candidate_failures",
            return_value=["no_non_silent_vocal_active_proxy_window"],
        ), patch.object(audio.legacy_audio, "_candidate_advisories", return_value=[]):
            result = self.select(
                soundtrack_contract("Instrumental"),
                first,
                candidate_count=2,
                candidate_2=second,
                selection_mode="lock_by_hash",
                locked_waveform_sha256=lock_hash,
            )
        self.assertTrue(result[-1])
        self.assertIs(result[0], second)
        self.assertEqual(result[2], lock_hash)
        self.assertEqual(json.loads(result[4])["selected_candidate_index"], 2)


class AdvertisementSoundtrackRouterTests(unittest.TestCase):
    def test_lazy_status_requests_only_selected_source(self):
        pending = audio.DiffusionGemmaAdvertisementSoundtrackSourceRouter.check_lazy_status(
            source_mode="MiniMax Music 3",
            music3_candidate_count=1,
            music3_expected_bpm=105.0,
            music3_lyrics="[Instrumental]",
            music3_duration_seconds=35.0,
            music3_candidate_1=None,
            uploaded_audio=None,
            ace_candidate_1=None,
        )
        self.assertEqual(pending, ["music3_candidate_1"])

    def test_music3_default_and_upload_emit_distinct_provenance(self):
        candidate = fake_audio(seconds=3.0)
        router = audio.DiffusionGemmaAdvertisementSoundtrackSourceRouter()
        generated = router.route(
            "MiniMax Music 3",
            music3_candidate_count=1,
            music3_expected_bpm=105.0,
            music3_lyrics="[Instrumental]",
            music3_duration_seconds=3.0,
            music3_candidate_1=candidate,
        )
        self.assertEqual(generated[4], 1)
        self.assertEqual(generated[8], "minimax_music3")
        uploaded = router.route(
            "Upload song",
            uploaded_audio=candidate,
            uploaded_duration_seconds=3.0,
            uploaded_expected_bpm=0.0,
            uploaded_lyrics="",
            uploaded_waveform_sha256=audio.legacy_audio.waveform_sha256(candidate),
            uploaded_ready=True,
        )
        self.assertEqual(uploaded[8], "uploaded_song")
        self.assertIs(uploaded[0], candidate)


class AdvertisementAudioMixerTests(unittest.TestCase):
    def test_voice_over_never_enters_motion_guide_but_enters_final_mix(self):
        sample_rate = 1000
        clock = torch.arange(2000, dtype=torch.float32) / sample_rate
        music_wave = (0.08 * torch.sin(2.0 * math.pi * 30.0 * clock)).reshape(1, 1, -1)
        voice_wave = torch.full((1, 1, 500), 0.08)
        music = {"waveform": music_wave, "sample_rate": sample_rate}
        voice = {"waveform": voice_wave, "sample_rate": sample_rate}
        result = audio.DiffusionGemmaAdvertisementAudioMixer().mix(
            music_audio=music,
            soundtrack_contract_json=soundtrack_contract(
                "Instrumental", "Separate non-diegetic VO"
            ),
            target_duration_seconds=2.0,
            music_gain_db=0.0,
            voice_over_gain_db=0.0,
            ducking_db=-9.0,
            duck_attack_ms=20.0,
            duck_release_ms=50.0,
            peak_ceiling_dbfs=-1.0,
            clipping_policy="Attenuate mix to ceiling",
            voice_over=voice,
            voice_over_start_seconds=0.5,
        )
        motion, final = result[0], result[1]
        report = json.loads(result[4])
        self.assertTrue(torch.equal(motion["waveform"], music_wave))
        self.assertFalse(torch.equal(final["waveform"], motion["waveform"]))
        self.assertNotEqual(result[2], result[3])
        self.assertFalse(report["voice_over"]["included_in_motion_guide"])
        self.assertTrue(report["voice_over"]["included_in_final_mix"])
        self.assertEqual(report["levels"]["lufs"], "not_measured")
        self.assertEqual(final["waveform"].shape[-1], 2000)
        self.assertLess(report["ducking"]["minimum_applied_gain"], 1.0)

    def test_no_voice_over_keeps_motion_and_final_hashes_equal(self):
        music = fake_audio(0.05, seconds=2.0)
        result = audio.DiffusionGemmaAdvertisementAudioMixer().mix(
            music,
            soundtrack_contract("Instrumental", "None"),
            2.0,
            0.0,
            0.0,
            -9.0,
            20.0,
            50.0,
            -1.0,
            "Attenuate mix to ceiling",
        )
        self.assertEqual(result[2], result[3])
        self.assertTrue(torch.equal(result[0]["waveform"], result[1]["waveform"]))

    def test_one_sample_rounding_deficit_is_padded_but_material_shortfall_fails(self):
        sample_rate = 44100
        short_by_one = {
            "waveform": torch.full((1, 2, sample_rate - 1), 0.05),
            "sample_rate": sample_rate,
        }
        result = audio.DiffusionGemmaAdvertisementAudioMixer().mix(
            short_by_one,
            soundtrack_contract("Instrumental", "None"),
            1.0,
            0.0,
            0.0,
            -9.0,
            20.0,
            50.0,
            -1.0,
            "Attenuate mix to ceiling",
        )
        report = json.loads(result[4])
        self.assertEqual(result[0]["waveform"].shape[-1], sample_rate)
        self.assertEqual(report["duration_alignment"]["right_padding_samples"], 1)
        self.assertEqual(float(result[0]["waveform"][..., -1].abs().max()), 0.0)

        materially_short = {
            "waveform": torch.full((1, 2, sample_rate - 100), 0.05),
            "sample_rate": sample_rate,
        }
        with self.assertRaisesRegex(ValueError, "shorter than"):
            audio.DiffusionGemmaAdvertisementAudioMixer().mix(
                materially_short,
                soundtrack_contract("Instrumental", "None"),
                1.0,
                0.0,
                0.0,
                -9.0,
                20.0,
                50.0,
                -1.0,
                "Attenuate mix to ceiling",
            )

    def test_clipping_fail_policy_is_explicit(self):
        music = fake_audio(1.0, seconds=2.0)
        with self.assertRaisesRegex(ValueError, "exceeds the configured"):
            audio.DiffusionGemmaAdvertisementAudioMixer().mix(
                music,
                soundtrack_contract("Instrumental", "None"),
                2.0,
                6.0,
                0.0,
                -9.0,
                20.0,
                50.0,
                -1.0,
                "Fail on clipping",
            )


if __name__ == "__main__":
    unittest.main()
