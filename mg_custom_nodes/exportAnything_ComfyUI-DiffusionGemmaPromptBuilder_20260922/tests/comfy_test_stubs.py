from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import wave

import torch
import torch.nn.functional as torch_functional


def install_comfy_stubs() -> None:
    """Install the narrow ComfyUI seams exercised by the reframe unit tests."""

    folder_paths = ModuleType("folder_paths")
    folder_paths.get_input_directory = lambda: str(Path.cwd())
    folder_paths.annotated_filepath = lambda name: (name, None)
    folder_paths.filter_files_content_types = lambda files, _types: list(files)

    def load_audio(path: str):
        with wave.open(str(path), "rb") as handle:
            if handle.getsampwidth() != 2:
                raise ValueError("test audio stub supports 16-bit PCM only")
            channels = handle.getnchannels()
            frame_count = handle.getnframes()
            sample_rate = handle.getframerate()
            payload = bytearray(handle.readframes(frame_count))
        samples = torch.frombuffer(payload, dtype=torch.int16).reshape(-1, channels)
        waveform = samples.transpose(0, 1).to(torch.float32).div_(32768.0).contiguous()
        return waveform, sample_rate

    comfy_extras = ModuleType("comfy_extras")
    comfy_extras.__path__ = []
    comfy_audio = ModuleType("comfy_extras.nodes_audio")
    comfy_audio.load = load_audio
    comfy_extras.nodes_audio = comfy_audio

    comfy = ModuleType("comfy")
    comfy.__path__ = []
    comfy_utils = ModuleType("comfy.utils")
    comfy_utils.common_upscale = lambda images, width, height, _method, _crop: (
        torch_functional.interpolate(
            images,
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
        )
    )
    comfy.utils = comfy_utils

    class UnsupportedVideo:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("video decoding is outside this unit-test seam")

    comfy_latest = ModuleType("comfy_api.latest")
    comfy_latest.InputImpl = SimpleNamespace(
        VideoFromFile=UnsupportedVideo,
        VideoFromComponents=UnsupportedVideo,
    )
    comfy_latest.Types = SimpleNamespace(
        VideoComponents=lambda **values: SimpleNamespace(**values)
    )
    comfy_api = ModuleType("comfy_api")
    comfy_api.__path__ = []
    comfy_api.latest = comfy_latest

    aiohttp_web = ModuleType("aiohttp.web")
    aiohttp_web.json_response = lambda data, status=200: SimpleNamespace(
        data=data,
        status=status,
    )
    aiohttp = ModuleType("aiohttp")
    aiohttp.__path__ = []
    aiohttp.web = aiohttp_web

    sys.modules.update(
        {
            "aiohttp": aiohttp,
            "aiohttp.web": aiohttp_web,
            "comfy": comfy,
            "comfy.utils": comfy_utils,
            "comfy_api": comfy_api,
            "comfy_api.latest": comfy_latest,
            "comfy_extras": comfy_extras,
            "comfy_extras.nodes_audio": comfy_audio,
            "folder_paths": folder_paths,
        }
    )
