import importlib.util
from pathlib import Path
import unittest

import torch


spec = importlib.util.spec_from_file_location('voxel_test', Path(__file__).parents[1] / 'nodes/vfx/FL_VoxelNormalRelief.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class VoxelNormalTests(unittest.TestCase):
    def test_output_is_aligned_bounded_deterministic_and_input_unchanged(self):
        normals = torch.rand(3, 47, 61, 3)
        depth = torch.rand_like(normals)
        original = normals.clone()
        args = (normals, depth, 8, .65, .2, .7, 24, 41)
        a = module.FL_VoxelNormalRelief().render(*args)[0]
        b = module.FL_VoxelNormalRelief().render(*args)[0]
        self.assertEqual(a.shape, normals.shape)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(normals, original)
        self.assertTrue(torch.isfinite(a).all() and (a >= 0).all() and (a <= 1).all())

    def test_animation_and_depth_change_relief(self):
        normals = torch.ones(12, 48, 48, 3) * .7
        depth = torch.ones_like(normals) * .5
        node = module.FL_VoxelNormalRelief()
        moving = node.render(normals, depth, 12, 1, .4, 1, 24, 41)[0]
        still = node.render(normals, depth, 12, 1, 0, 1, 24, 41)[0]
        torch.testing.assert_close(still[0], still[-1])
        self.assertFalse(torch.equal(moving[0], moving[-1]))
        flat = node.render(normals, torch.zeros_like(depth), 12, 1, 0, 1, 24, 41)[0]
        self.assertFalse(torch.equal(flat, still))

    def test_mismatched_frames_fail(self):
        with self.assertRaisesRegex(ValueError, 'aligned'):
            module.FL_VoxelNormalRelief().render(torch.zeros(2, 32, 32, 3), torch.zeros(1, 32, 32, 3),
                8, 1, .2, 1, 24, 0)

    def test_flat_surface_keeps_visible_cube_sides(self):
        normals = torch.ones(1, 48, 48, 3) * .7
        depth = torch.ones_like(normals) * .5
        image = module.FL_VoxelNormalRelief().render(normals, depth, 12, .65, 0, .7, 24, 41)[0]
        for shade in (.48, .68):
            self.assertTrue(torch.isclose(image[..., 0], torch.tensor(.7 * shade)).any())

    def test_speed_zero_holds_accumulated_phase(self):
        normals=torch.ones(12,48,48,3)*.7
        depth=torch.ones_like(normals)*.5
        values={"relief":[1]*12,"animation":[.8]*12,"speed":[1]*4+[0]*8}
        output=module.FL_VoxelNormalRelief().render_animated(normals,depth,12,1,.8,1,24,41,values)[0]
        torch.testing.assert_close(output[4],output[-1])
        self.assertFalse(torch.equal(output[0],output[4]))


if __name__ == '__main__':
    unittest.main()
