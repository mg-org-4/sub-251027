import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import torch


spec = importlib.util.spec_from_file_location("drum_bands_test", Path(__file__).parents[1] / "nodes/audio/FL_Audio_Drum_Detector.py")
drums = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drums)


class DrumBandTests(unittest.TestCase):
    def test_independent_bands_detect_transients_and_keep_duration(self):
        rate = 24000
        signal = np.zeros(rate * 3, np.float32)
        time = np.arange(2400) / rate
        for start, frequency in ((0.4, 80), (1.1, 1200), (1.9, 8000)):
            burst = np.sin(2 * np.pi * frequency * time) * np.hanning(len(time))
            offset = round(start * rate)
            signal[offset:offset+len(burst)] += burst.astype(np.float32)
        audio = {"waveform": torch.from_numpy(signal).reshape(1, 1, -1), "sample_rate": rate}
        result = json.loads(drums.FL_Audio_Drum_Detector().detect_drums(audio, detection_mode="independent_bands")[0])
        self.assertEqual(result['duration'], 3)
        for key, expected in (("kick_times", .4), ("snare_times", 1.1), ("hihat_times", 1.9)):
            self.assertTrue(any(abs(value - expected) < .15 for value in result[key]), (key, result[key]))
            self.assertTrue(all(0 <= value < 3 for value in result[key]))

    def test_silence_produces_no_fake_hits(self):
        audio = {"waveform": torch.zeros(1, 1, 24000), "sample_rate": 24000}
        result = json.loads(drums.FL_Audio_Drum_Detector().detect_drums(audio, detection_mode="independent_bands")[0])
        for key in ('kick_times', 'snare_times', 'hihat_times'):
            self.assertEqual(result[key], [])

    def test_existing_workflows_keep_original_default(self):
        config = drums.FL_Audio_Drum_Detector.INPUT_TYPES()
        self.assertEqual(config['optional']['detection_mode'][1]['default'], 'classified_onsets')
        self.assertEqual(list(config['optional'])[:3], ['kick_sensitivity', 'snare_sensitivity', 'hihat_sensitivity'])


if __name__ == '__main__':
    unittest.main()
