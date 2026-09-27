from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from ComfyUI_H3_Continuum_Join import nodes as root_nodes
from ComfyUI_H3_Continuum_Join.v3.easy_video_nodes import ExecutionBlocker, _resample_video_frames
from ComfyUI_H3_Continuum_Join.v3.video_adapter_nodes import H3ContinuumVideoAdapter


def _outputs(output):
    return output.args if hasattr(output, "args") else output.result


def test_video_adapter_is_registered_without_file_upload_or_enable():
    assert root_nodes.NODE_CLASS_MAPPINGS["H3ContinuumVideoAdapter"] is H3ContinuumVideoAdapter
    schema = H3ContinuumVideoAdapter.define_schema()
    assert [x.id for x in schema.inputs] == ["video", "force_rate"]
    assert schema.inputs[1].default == 24.0
    assert len(schema.outputs) == 2


@pytest.mark.parametrize("source_rate,count", [(Fraction(24),240), (Fraction(30),300), (Fraction(30000,1001),300)])
def test_adapter_keeps_existing_resampling_and_exact_audio(source_rate, count):
    images = torch.arange(count, dtype=torch.float32).view(count, 1, 1, 1).expand(-1, 2, 2, 3).clone()
    audio = {"waveform": torch.arange(64, dtype=torch.float32).reshape(1, 1, 64), "sample_rate":32000}
    calls = []
    class Video:
        def get_components(self):
            calls.append(True)
            return SimpleNamespace(images=images, audio=audio, frame_rate=source_rate)
    output_images, output_audio = _outputs(H3ContinuumVideoAdapter.execute(Video(), 24.0))
    assert len(calls) == 1
    assert torch.equal(output_images, _resample_video_frames(images, source_rate, 24.0))
    assert output_images.shape[0] == 240
    assert output_audio is audio
    if source_rate == 24:
        assert output_images is images


def test_adapter_source_rate_and_silent_video_contract():
    images = torch.zeros(30, 2, 2, 3)
    video = SimpleNamespace(get_components=lambda: SimpleNamespace(images=images, audio=None, frame_rate=Fraction(30)))
    out_images, audio = _outputs(H3ContinuumVideoAdapter.execute(video, 0))
    assert out_images is images
    assert isinstance(audio, ExecutionBlocker)
