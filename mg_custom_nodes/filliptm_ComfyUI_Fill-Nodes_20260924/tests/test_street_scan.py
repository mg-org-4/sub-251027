import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import cv2
import math
import torch


spec = importlib.util.spec_from_file_location("street_scan_test", Path(__file__).parents[1] / "nodes/vfx/FL_StreetScan.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


class StreetScanTests(unittest.TestCase):
    def test_default_stack_is_pixel_identical_to_original(self):
        matte = np.zeros((96,128),np.uint8)
        matte[16:75,20:98] = 255
        layers = np.random.default_rng(5).random((96,128,6),dtype=np.float32)
        for phase in (0,.3,1):
            filled,hull = scan.fill_projected_gaps(layers.copy(),matte.copy())
            expected = np.zeros((96,128,3),np.float32)
            step = max(2,round(128*(.009+.004)))
            for level in (4,3,2,1):
                shift = np.float32([[1,0,level*step],[0,1,level*step*(.65+.2*math.sin(phase*math.tau))]])
                plate = cv2.warpAffine(hull,shift,(128,96),flags=cv2.INTER_NEAREST)
                expected[plate>0] = (.035,.02,.9) if level%2==0 else (.04,.04,.055)
                contours,_ = cv2.findContours(plate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(expected,contours,-1,(.85,.9,1),1,cv2.LINE_8)
            expected[hull>0] = filled[:,:,:3][hull>0]
            contours,_ = cv2.findContours(hull,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(expected,contours,-1,(1,1,1),1,cv2.LINE_8)
            actual = scan.digital_layers(layers.copy(),matte.copy(),phase,1)[2]
            np.testing.assert_array_equal(actual,expected)

    def test_stack_controls_preserve_foreground_and_disable_cleanly(self):
        matte = np.zeros((96,128),np.uint8)
        matte[24:64,36:80] = 255
        layers = np.full((96,128,6),.3,np.float32)
        base = scan.digital_layers(layers.copy(),matte.copy(),.2,1)[2]
        off = scan.digital_layers(layers.copy(),matte.copy(),.2,1,count=0)[2]
        transparent = scan.digital_layers(layers.copy(),matte.copy(),.2,1,opacity=0)[2]
        np.testing.assert_array_equal(off,transparent)
        for options in ({"count":8},{"spacing":3},{"x":-1},{"y":-1},{"rotation":8},{"opacity":.4},{"palette":"cyan"}):
            result = scan.digital_layers(layers.copy(),matte.copy(),.2,1,**options)[2]
            self.assertFalse(np.array_equal(result,base),options)
            np.testing.assert_array_equal(result[26:62,38:78],base[26:62,38:78])
            self.assertTrue(np.isfinite(result).all())

    def test_digital_mode_does_not_use_eroded_mask(self):
        images = torch.rand(2, 32, 32, 3)
        depth = torch.ones_like(images) * .5
        masks = torch.zeros(2, 32, 32)
        tracks = {"width": 32, "height": 32, "frames": [[], []]}
        args = (images, depth, depth, masks, tracks, 24, 41, 5, 1, .74)
        first = scan.FL_StreetScanComposite().render(*args, 0, 0, 0, 0, edge_style='digital_layers')
        second = scan.FL_StreetScanComposite().render(*args, 1, 0, 0, 1, edge_style='digital_layers')
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)

    def test_digital_layers_fill_projection_gaps_and_keep_normals_aligned(self):
        layers = np.zeros((32, 32, 6), np.float32)
        layers[4:25, 4:25] = [.2, .3, .4, .6, .7, .8]
        matte = np.zeros((32, 32), np.uint8)
        matte[4:25, 4:25] = 255
        matte[10:12, 10:12] = 0
        layers[10:12, 10:12] = 0
        result, filled, composite = scan.digital_layers(layers, matte, .2, 1)
        np.testing.assert_allclose(result[10, 10], [.2, .3, .4, .6, .7, .8])
        self.assertTrue((filled[4:25, 4:25] == 255).all())
        self.assertTrue(np.isfinite(composite).all())
        self.assertTrue((composite[filled == 0] > 0).any())

    def test_identity_projection_preserves_pixels(self):
        rgb = np.random.default_rng(10).random((32, 32, 3), dtype=np.float32)
        depth = np.ones((32, 32), np.float32) * 0.5
        px, py, z = scan.project_depth(depth, 0, 0, 1, 1)
        result, matte = scan.splat(rgb, np.ones((32, 32), bool), px, py, z, (0, 0, 0))
        np.testing.assert_array_equal(result, rgb)
        self.assertTrue((matte == 255).all())

    def test_z_buffer_keeps_nearest_surface(self):
        rgb = np.array([[[1, 0, 0], [0, 1, 0]]], np.float32)
        result, _ = scan.splat(rgb, np.ones((1, 2), bool), np.zeros((1, 2)), np.zeros((1, 2)), np.array([[2, 1]]), (0, 0, 0))
        np.testing.assert_array_equal(result[0, 0], [0, 1, 0])

    def test_fragment_is_deterministic_and_preserves_subject(self):
        depth = np.ones((64, 64), np.float32) * 0.5
        subject = np.zeros_like(depth)
        subject[:5, :5] = 1
        mask = scan.scene_fragment(depth, subject, 41, 0.5)
        np.testing.assert_array_equal(mask, scan.scene_fragment(depth, subject, 41, 0.5))
        self.assertTrue(mask[:5, :5].all())
        self.assertFalse(mask.all())

    def test_detector_produces_real_masks_and_stable_ids(self):
        segment = SimpleNamespace(bbox=(4, 5, 20, 25), crop_region=(4, 5, 20, 25),
            cropped_mask=np.ones((20, 16), np.float32), confidence=0.9)
        detector = SimpleNamespace(detect=lambda *args, **kwargs: ((32, 32), [segment]))
        masks, tracks = scan.FL_ScanVideoDetections().detect(torch.zeros(2, 32, 32, 3), detector, 0.35)
        self.assertEqual(tuple(masks.shape), (2, 32, 32))
        self.assertEqual(tracks["frames"][0][0]["id"], tracks["frames"][1][0]["id"])
        self.assertEqual(float(masks[0].sum()), 320)

    def test_composite_is_bounded_repeatable_and_does_not_modify_inputs(self):
        images = torch.rand(3, 64, 64, 3)
        original = images.clone()
        depth = torch.ones_like(images) * 0.5
        masks = torch.zeros(3, 64, 64)
        tracks = {"width": 64, "height": 64, "frames": [[], [], []]}
        args = (images, depth, depth, masks, tracks, 24, 41, 8, 1.4, 0.88, 0.4, 0.8, 0.8, 0.4)
        first, matte, normals = scan.FL_StreetScanComposite().render(*args)
        second, _, _ = scan.FL_StreetScanComposite().render(*args)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(images, original)
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue((first >= 0).all() and (first <= 1).all())
        self.assertEqual(tuple(matte.shape), (3, 64, 64))
        self.assertEqual(normals.shape, images.shape)
        self.assertTrue(torch.isfinite(normals).all())

    def test_pose_overlay_uses_canvas_coordinates_and_rejects_wrong_batch(self):
        images = torch.zeros(1, 64, 64, 3)
        depth = torch.ones_like(images) * 0.5
        masks = torch.zeros(1, 64, 64)
        tracks = {"width": 64, "height": 64, "frames": [[]]}
        args = (images, depth, depth, masks, tracks, 24, 20, 0, 1, 1, 0.4, 0, 0, 0)
        points = [[16 + index * 4, 64, 1] for index in range(18)]
        pose = [{"canvas_width": 128, "canvas_height": 128,
                 "people": [{"pose_keypoints_2d": [v for point in points for v in point]}]}]
        plain, _, _ = scan.FL_StreetScanComposite().render(*args)
        overlay, _, _ = scan.FL_StreetScanComposite().render(*args, pose_keypoints=pose, pose_opacity=1)
        self.assertGreater(float((overlay[0, 30:35] - plain[0, 30:35]).abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, "one DWPose keypoint frame"):
            scan.FL_StreetScanComposite().render(*args, pose_keypoints=[])


if __name__ == "__main__":
    unittest.main()
