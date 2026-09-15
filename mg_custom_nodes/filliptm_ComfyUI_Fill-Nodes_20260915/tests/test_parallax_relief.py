import unittest

import torch

from test_layered_parallax import m, plate, render


class ReliefTests(unittest.TestCase):
    def test_zero_strength_matches_original(self):
        p = plate([1, 0, 0], 3, 1)
        a = render({'layer_0': p}, background=torch.zeros(1,32,64,3), frames=5)
        b = render({'layer_0': p}, background=torch.zeros(1,32,64,3), frames=5, relief_scope='background', relief_strength=0)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_artwork_relief_changes_motion_not_locked(self):
        p = plate([1, 0, 0], 3, 1)
        bg = torch.zeros(1, 32, 64, 3)
        depth = torch.ones(2, 32, 64, 3)
        a = render({'layer_0': p}, background=bg, frames=5)
        b = render({'layer_0': p}, background=bg, frames=5, relief_scope='background + artwork', depth_maps=depth, relief_strength=.4)
        self.assertGreater(float((a[0]-b[0]).abs().max()), .05)
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)

    def test_text_stays_flat(self):
        p = dict(plate([1, 0, 0], 3, 1), kind='text')
        bg = torch.zeros(1, 32, 64, 3)
        a = render({'layer_0': p}, background=bg, frames=5)[0]
        b = render({'layer_0': p}, background=bg, frames=5, relief_scope='background + artwork', depth_maps=torch.ones(2, 32, 64, 3))[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_neutral_depth_matches_flat_projection(self):
        p = plate([1, 0, 0], 3, 1)
        bg = torch.zeros(1, 32, 64, 3)
        a = render({'layer_0': p}, background=bg, frames=5)[0]
        b = render({'layer_0': p}, background=bg, frames=5, relief_scope='background + artwork', depth_maps=torch.full((2,32,64,3),.5), depth_smoothing=0)[0]
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_invalid_batch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'one background'):
            render({}, background=torch.zeros(1,32,64,3), relief_scope='background', depth_maps=torch.zeros(3,32,64,3))

    def test_source_batch_dynamic_and_alpha_neutralized(self):
        stack = dict(background=torch.zeros(1,32,64,3), layers=[plate([1,0,0],3,1),plate([0,1,0],2,1)])
        images=m.FL_ParallaxDepthSources.execute(stack,126).result[0]
        self.assertEqual(images.shape[0],3)
        self.assertEqual(images.shape[-1],3)
        torch.testing.assert_close(images[1,0,0],torch.full((3,),.5))

    def test_invert_and_smoothing(self):
        d=torch.zeros(1,8,8,3);d[:,4:,4:]=1
        a=m.prepare_relief(d,False,2,torch.device('cpu'))
        b=m.prepare_relief(d,True,2,torch.device('cpu'))
        torch.testing.assert_close(a+b,torch.ones_like(a))
        self.assertTrue(torch.any((a>0)&(a<1)))


if __name__ == '__main__':
    unittest.main()
