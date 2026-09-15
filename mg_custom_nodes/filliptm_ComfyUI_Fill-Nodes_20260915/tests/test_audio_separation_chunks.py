import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch


spec = importlib.util.spec_from_file_location("separation_test", Path(__file__).parents[1] / "nodes/audio/FL_Audio_Separation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class IdentitySeparator(torch.nn.Module):
    sources = ("drums", "bass", "other", "vocals")

    def forward(self, mix):
        return mix.unsqueeze(1).expand(-1, 4, -1, -1)


class SeparationChunkTests(unittest.TestCase):
    def test_loading_failure_does_not_return_fake_stems(self):
        with patch("torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS.get_model", side_effect=RuntimeError("missing weights")):
            with self.assertRaisesRegex(RuntimeError, "FL Audio Separation failed: missing weights"):
                module.FL_Audio_Separation().separate_audio({"waveform": torch.ones(1, 2, 100), "sample_rate": 44100})

    def test_overlaps_and_tail_reconstruct_without_gaps(self):
        node = module.FL_Audio_Separation()
        for length in (3, 20, 21, 57, 101):
            for overlap in (0, .2, .7):
                for fade in ("linear", "half_sine", "logarithmic", "exponential"):
                    with self.subTest(length=length, overlap=overlap, fade=fade):
                        mix = torch.rand(1, 2, length)
                        result = node._separate_sources(IdentitySeparator(), mix, 10,
                            segment=2, overlap=overlap, chunk_fade_shape=fade)
                        torch.testing.assert_close(result, mix.unsqueeze(1).expand(-1, 4, -1, -1))

    def test_invalid_overlap(self):
        with self.assertRaisesRegex(ValueError, "shorter"):
            module.FL_Audio_Separation()._separate_sources(IdentitySeparator(), torch.ones(1, 2, 30),
                10, segment=1, overlap=1)


if __name__ == "__main__":
    unittest.main()
