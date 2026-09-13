import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nodes.nodes_inpaint import (
    _extract_bounds,
    _apply_mask_blur_and_threshold,
    _color_correct,
    DaSiWa_InpaintCropPrep,
    DaSiWa_InpaintComposite,
)


def _make_mask(h, w, y1, y2, x1, x2):
    """Create a binary mask with a filled rectangle."""
    m = torch.zeros(h, w)
    m[y1:y2, x1:x2] = 1.0
    return m


class TestExtractBounds:
    def test_simple_rectangle(self):
        m = _make_mask(100, 100, 20, 80, 30, 70)
        x, y, bw, bh = _extract_bounds(m, grow=0)
        assert x == 30
        assert y == 20
        assert bw == 40
        assert bh == 60

    def test_grow_padding(self):
        m = _make_mask(100, 100, 20, 80, 30, 70)
        x, y, bw, bh = _extract_bounds(m, grow=10)
        assert x == 20
        assert y == 10
        assert bw == 60
        assert bh == 80

    def test_clamp_to_image_bounds(self):
        m = _make_mask(50, 50, 2, 48, 2, 48)
        x, y, bw, bh = _extract_bounds(m, grow=10)
        assert x == 0
        assert y == 0
        assert bw == 50
        assert bh == 50

    def test_empty_mask_raises(self):
        m = torch.zeros(100, 100)
        with pytest.raises(ValueError, match="mask is empty"):
            _extract_bounds(m, grow=0)


class TestMaskBlurAndThreshold:
    def test_no_blur_preserves_values(self):
        m = torch.ones(64, 64)
        result = _apply_mask_blur_and_threshold(m, blur_radius=0, min_val=0, max_val=1)
        assert result.shape == (64, 64)
        assert result.min() >= 0.99

    def test_clamping(self):
        m = torch.ones(64, 64) * 0.5
        result = _apply_mask_blur_and_threshold(m, blur_radius=0, min_val=0.01, max_val=0.8)
        assert result.max() <= 0.8
        assert result.min() >= 0.01


class TestColorCorrect:
    def test_none_returns_unchanged(self):
        img = torch.rand(1, 32, 32, 3)
        ref = torch.rand(1, 32, 32, 3)
        result = _color_correct(img, ref, "None")
        assert torch.allclose(result, img)

    def test_match_channels_adjusts_statistics(self):
        src = torch.ones(1, 32, 32, 3) * 0.5
        dst = torch.ones(1, 32, 32, 3) * 0.8
        result = _color_correct(src, dst, "Match Channels")
        assert result.mean() > src.mean()


class TestInpaintCropPrep:
    def test_crops_and_scales(self):
        node = DaSiWa_InpaintCropPrep()
        img = torch.rand(1, 512, 512, 3)
        mask = _make_mask(512, 512, 100, 400, 150, 350)
        cropped, c_mask, x, y, w, h = node.execute(
            image=img, mask=mask,
            target_width=1024, target_height=1024,
            mask_blur=0, mask_min=0, mask_max=1,
            grow_px=0, can_shrink=True,
        )
        assert cropped.shape == (1, 1024, 1024, 3)
        assert c_mask.shape == (1024, 1024)
        assert x == 150
        assert y == 100

    def test_grow_expands_crop(self):
        node = DaSiWa_InpaintCropPrep()
        img = torch.rand(1, 512, 512, 3)
        mask = _make_mask(512, 512, 200, 300, 200, 300)
        _, _, x, y, w, h = node.execute(
            image=img, mask=mask,
            target_width=512, target_height=512,
            mask_blur=0, mask_min=0, mask_max=1,
            grow_px=50, can_shrink=True,
        )
        assert w == 200
        assert h == 200


class TestInpaintComposite:
    def test_composites_at_position(self):
        node = DaSiWa_InpaintComposite()
        dest = torch.zeros(1, 512, 512, 3)
        src = torch.ones(1, 100, 100, 3)
        mask = torch.ones(100, 100)
        result = node.execute(destination=dest, source=src, mask=mask, x=200, y=200, w=100, h=100, correction_method="None")[0]
        region = result[0, 200:300, 200:300, :].mean()
        assert region > 0.99

    def test_transparent_mask_preserves_destination(self):
        node = DaSiWa_InpaintComposite()
        dest = torch.ones(1, 512, 512, 3) * 0.5
        src = torch.zeros(1, 100, 100, 3)
        mask = torch.zeros(100, 100)
        result = node.execute(destination=dest, source=src, mask=mask, x=200, y=200, w=100, h=100, correction_method="None")[0]
        region = result[0, 200:300, 200:300, :].mean()
        assert abs(region.item() - 0.5) < 0.01

    def test_scales_source_and_mask_to_crop_size(self):
        """Source/mask come at target res (1024), must shrink to crop (200x200)."""
        node = DaSiWa_InpaintComposite()
        dest = torch.zeros(1, 512, 512, 3)
        src = torch.ones(1, 1024, 1024, 3)
        mask = torch.ones(1024, 1024)
        result = node.execute(destination=dest, source=src, mask=mask, x=156, y=156, w=200, h=200, correction_method="None")[0]
        region = result[0, 156:356, 156:356, :].mean()
        assert region > 0.99
