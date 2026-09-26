import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
import numpy as np


root = Path(__file__).parents[1] / "nodes"
for name, path in (("scan_edit_test", root), ("scan_edit_test.vfx", root / "vfx"), ("scan_edit_test.audio", root / "audio")):
    package = ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
spec = importlib.util.spec_from_file_location("scan_edit_test.vfx.FL_ScanAudioEdit", root / "vfx/FL_ScanAudioEdit.py")
edit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edit)


def envelope(values, fps=24):
    return {"type": "fl_audio_envelope", "version": 1, "values": values, "fps": fps,
            "duration": len(values) / fps, "total_frames": len(values)}


class ScanAudioEditTests(unittest.TestCase):
    def test_random_order_changes_only_at_snare_onsets(self):
        images = torch.rand(24,32,32,3)
        quiet = envelope([0]*24)
        snare = envelope([float(i in (3,4,12,13)) for i in range(24)])
        observed = []
        original_order = edit.order_windows
        def capture(events,mode,ranks):
            observed.append(tuple(ranks))
            return original_order(events,mode,ranks)
        with patch.object(edit,"order_windows",side_effect=capture):
            edit.FL_ScanAudioEdit().render_mapped(images,images,images,quiet,snare,quiet,
                "24",24,7,2,8,1.7,5,1,1,0,"audio_locked",window_order="random_on_snare")
        self.assertEqual(len(observed),24)
        for start,end in ((0,3),(3,12),(12,24)):
            self.assertEqual(len(set(observed[start:end])),1)
        self.assertNotEqual(observed[2],observed[3])
        self.assertNotEqual(observed[11],observed[12])

    def test_window_order_is_stable_and_priority_is_on_top(self):
        events = [{"cursor":i,"effect":i} for i in range(3)]
        self.assertEqual(edit.order_windows(events,"newest_on_top",[]),events)
        for mode,effect in (("voxel_on_top",0),("edge_on_top",1),("depth_on_top",2)):
            self.assertEqual(edit.order_windows(events,mode,[])[-1]["effect"],effect)
        ranks = [2,0,1]
        self.assertEqual([e["cursor"] for e in edit.order_windows(events,"random_on_snare",ranks)],[1,2,0])
        self.assertEqual([e["cursor"] for e in edit.order_windows(events[1:],"random_on_snare",ranks)],[1,2])

    def test_reveal_blends_and_fades(self):
        for mode,value in (("normal",.5),("screen",.52),("add",.6)):
            region = np.full((2,2,3),.2,np.float32)
            edit.blend_reveal(region,np.full_like(region,.8),.5,mode)
            np.testing.assert_allclose(region,value,atol=1e-7)
        event = {"start":0,"end":25}
        self.assertEqual(edit.reveal_fade(0,event,24,0,0),1)
        self.assertEqual(edit.reveal_fade(2,event,24,.2,.2),0)
        self.assertEqual(edit.reveal_fade(12,event,24,.2,.2),1)
        self.assertEqual(edit.reveal_fade(24,event,24,.2,.2),0)

    def test_cursor_events_follow_audio_not_cuts(self):
        pulses = [float(i in (3, 21, 39)) for i in range(60)]
        quiet = [0] * 60
        events = edit.cursor_plan((pulses, quiet, quiet), 2, 99, 24)
        self.assertEqual(events, edit.cursor_plan((pulses, quiet, quiet), 2, 99, 24))
        self.assertEqual([e['start'] for e in events], [3, 21, 39])
        self.assertEqual(edit.cursor_plan((quiet, quiet, quiet), 2, 99, 24), [])
        self.assertNotEqual(events, edit.cursor_plan((pulses, quiet, quiet), 2, 100, 24))
        for event in events:
            self.assertTrue(all(.05 <= x <= .95 for x in event['target']))
        self.assertEqual(edit.cursor_plan((pulses, pulses, pulses), 0, 99, 24), [])

    def test_audio_locked_preserves_every_frame(self):
        images = torch.rand(24, 32, 32, 3)
        beat = envelope([float(i % 7 == 0) for i in range(24)])
        _, before, _, report = edit.FL_ScanAudioEdit().render(images, images, images, beat, beat, beat,
            "12,12", 24, 1, 2, 8, 1.7, 2, 1, 1, .8, timing_mode="audio_locked")
        torch.testing.assert_close(before, images)
        report = json.loads(report)
        self.assertEqual(report['source_indices'], list(range(24)))
        self.assertTrue(any(s['start_frame'] == 12 for s in report['segments']))
        self.assertTrue(any(s['edit_type'] == 'camera_cut' for s in report['segments']))

    def test_plan_is_repeatable_in_bounds_and_onset_driven(self):
        kick = [float(frame in (5, 12, 20, 28)) for frame in range(36)]
        args = ([12, 12, 12, 12], kick, [0] * 36, 4, 10, 1.7, 73)
        indices, segments = edit.edit_plan(*args)
        self.assertEqual((indices, segments), edit.edit_plan(*args))
        self.assertEqual(len(indices), 36)
        self.assertTrue(all(0 <= index < 48 for index in indices))
        self.assertEqual({s['shot'] for s in segments[:4]}, {1, 2, 3, 4})
        self.assertEqual(segments[1]['start_frame'], 5)
        self.assertEqual(segments[1]['trigger'], 'audio_onset')
        self.assertEqual(segments[4]['edit_type'], 'jump_cut')
        self.assertEqual(segments[4]['shot'], segments[3]['shot'])
        for segment in segments:
            bank = (segment['shot'] - 1) * 12
            self.assertTrue(all(bank <= i < bank + 12 for i in indices[segment['start_frame']:segment['end_frame']]))

    def test_invalid_cut_bounds(self):
        with self.assertRaisesRegex(ValueError, "maximum cut length"):
            edit.edit_plan([12], [0] * 12, [0] * 12, 10, 5, 1, 0)

    def test_aligned_outputs_masks_and_no_mutation(self):
        source = torch.arange(24).reshape(24, 1, 1, 1).expand(24, 64, 64, 3).float() / 24
        scan = torch.ones_like(source) * 0.2
        normal = torch.zeros_like(source)
        normal[:, :, :, 1] = 1
        original = source.clone()
        beats = envelope([float(i % 7 == 0) for i in range(24)])
        result, before, mask, report = edit.FL_ScanAudioEdit().render(source, scan, normal, beats, beats, beats,
            "12,12", 24, 73, 6, 12, 1.7, 2, 1.2, 1, 0.8)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(tuple(mask.shape), (24, 64, 64))
        self.assertTrue((mask >= 0).all() and (mask <= 1).all())
        self.assertGreater(float(mask.sum()), 0)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue((result >= 0).all() and (result <= 1).all())
        indices = json.loads(report)['source_indices']
        torch.testing.assert_close(before, source[indices])
        torch.testing.assert_close(source, original)
        self.assertTrue((normal[:, :, :, 1] == 1).all())

    def test_wrong_envelope_fps_and_bank_length_fail(self):
        images = torch.zeros(4, 32, 32, 3)
        beat = envelope([0] * 4)
        args = [images, images, images, beat, beat, envelope([0] * 4, 30), "4", 24, 1, 2, 4, 1, 1, 1, 1, 1]
        with self.assertRaisesRegex(ValueError, "match the edit FPS"):
            edit.FL_ScanAudioEdit().render(*args)
        args[5], args[6] = beat, "5"
        with self.assertRaisesRegex(ValueError, "add up"):
            edit.FL_ScanAudioEdit().render(*args)

    def test_zero_reveal_strength_outputs_empty_mask(self):
        images = torch.zeros(8, 64, 64, 3)
        beat = envelope([0] * 8)
        _, _, mask, _ = edit.FL_ScanAudioEdit().render(images, images, images, beat, beat, beat,
            "8", 24, 1, 2, 8, 1, 2, 1, 0, 0)
        self.assertEqual(float(mask.sum()), 0)


if __name__ == "__main__":
    unittest.main()
