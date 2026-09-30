import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ComfyUI"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nodes"))
import nodes_rtx_upscaler_refiner as rtx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RTX effect requires CUDA")
def test_regular_node_keeps_single_image_batch_with_default_chunking(monkeypatch):
    class FakeEffect:
        def run(self, frame):
            return type("Result", (), {"image": frame.clone()})()

    monkeypatch.setattr(rtx, "_import_vfx", lambda: (object(), object()))
    monkeypatch.setattr(rtx, "_maybe_vfx_effect", lambda *args, **kwargs: __import__("contextlib").nullcontext(FakeEffect() if args[1] else None))
    monkeypatch.setattr(rtx, "_cleanup_cuda", lambda **kwargs: None)
    monkeypatch.setattr(rtx, "force_gc_and_cleanup", lambda *args: None)
    options = rtx.DaSiWa_RTX_UpscalerRefiner.INPUT_TYPES()["optional"]
    assert options["chunking"][1]["default"] is True
    assert options["lossless_fp16"][1]["default"] is True
    images = torch.full((5, 8, 8, 3), 0.1, dtype=torch.float32)
    kwargs = dict(images=images, denoise=True, denoise_quality="High", deblur=False,
                  deblur_quality="High", upscale="Off", upscale_quality="High",
                  resize_type="Same Size", scale=1.0, megapixels=1.0,
                  width=8, height=8, divisible_by="8", ratio_preset="1:1",
                  resize_method="Center Crop (Fill)", device_id=0, chunk_frames=2)
    chunked, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**kwargs)
    unchunked, = rtx.DaSiWa_RTX_UpscalerRefiner().execute(**kwargs, chunking=False)
    assert chunked.shape == images.shape
    assert chunked.dtype == torch.float32
    assert torch.equal(chunked, unchunked)


def test_regular_batch_lossless_fp16_only_when_exact():
    exact = torch.full((5, 8, 8, 3), 0.25, dtype=torch.float32)
    result = rtx._lossless_fp16_image_batch(exact, 2)
    assert result.dtype == torch.float16 and torch.equal(result.float(), exact)
    exact[4, 0, 0, 0] = 0.1
    result = rtx._lossless_fp16_image_batch(exact, 2)
    assert result is exact and result.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RTX effect requires CUDA")
def test_direct_sdk_copy_keeps_pixels_after_sdk_reuses_buffer():
    class ReusingEffect:
        def __init__(self):
            self.buffer = torch.empty((3, 8, 8), device="cuda")

        def run(self, frame):
            self.buffer.copy_(frame)
            return type("Result", (), {"image": self.buffer})()

    effect = ReusingEffect()
    row = torch.empty((8, 8, 3), dtype=torch.float32)
    frame = torch.full((3, 8, 8), 0.25, device="cuda")
    rtx._run_vfx_effect_into(effect, frame, row, torch.device("cuda:0"))
    effect.run(torch.ones_like(frame))
    assert torch.equal(row, torch.full_like(row, 0.25))
