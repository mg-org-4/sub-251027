from pathlib import Path
import sys
import uuid

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.modules["decode_cache_test_fixtures"] = sys.modules[__name__]


class NativeVideoModel(torch.nn.Module):
    __module__ = "comfy.ldm.minimax.vae"

    def __init__(self):
        super().__init__()
        self.register_parameter("weight", torch.nn.Parameter(torch.ones(1), requires_grad=False))
        self.register_buffer("offset", torch.zeros(1))
        self.tile_size = 256
        self.eval()

    def decode(self, tensor):
        return tensor * self.weight + self.offset


class NativeAudioModel(NativeVideoModel):
    __module__ = "comfy.ldm.minimax.audio_vae"


class Patcher:
    def __init__(self):
        self.patches_uuid = uuid.uuid4()
        self.object_patches = {}


class FakeVAE:
    """Controlled delegate fixture, NOT a real MiniMax model/GPU execution."""
    __module__ = "comfy.sd"

    def __init__(self, stream="video"):
        self.first_stage_model = NativeVideoModel() if stream == "video" else NativeAudioModel()
        self.patcher = Patcher()
        self.vae_dtype = torch.float32
        self.output_device = torch.device("cpu")
        self.device = torch.device("cpu")
        self.audio_sample_rate = 32000
        self.calls = 0

    def decode(self, tensor):
        self.calls += 1
        return self.first_stage_model.decode(tensor)

    def process_output(self, x):
        return x


class NestedFixture:
    is_nested = True

    def __init__(self, video, audio):
        self.parts = video, audio

    def unbind(self):
        return self.parts


@pytest.fixture
def vae():
    return FakeVAE()


@pytest.fixture
def samples():
    return {"samples": torch.arange(120, dtype=torch.float32).reshape(5, 3, 4, 2) / 120}
