"""CPU tests: real reference/grid helpers, deterministic recording VAE doubles.

These tests do not establish real GPU visual parity, memory peaks or frontend
compatibility. They can also run in the full repository's normal pytest suite.
"""
from __future__ import annotations

import copy
import dataclasses
import sys
import types

import pytest
import torch

from ComfyUI_H3_Continuum_Join import reference_video as reference
from ComfyUI_H3_Continuum_Join import temporal
from ComfyUI_H3_Continuum_Join import video_reference_modes as modes


class RecordingVAE:
    def __init__(self):
        self.inputs = []

    def encode(self, frames):
        self.inputs.append(frames.clone())
        t = temporal.video_latent_t(int(frames.shape[0]))
        # A CPU stand-in sufficient to verify the actual encoder's payload path.
        return torch.full((1, 24, t, 2, 2), float(frames[0, 0, 0, 0]))


def frames(count=720, channels=3):
    return torch.arange(count, dtype=torch.float32)[:, None, None, None].expand(count, 32, 32, channels).clone() / 1000


def source(count=720, seconds=5.0):
    return modes.prepare_follow_video_source(
        frames(count), chunk_seconds=seconds, output_width=32, output_height=32,
        size_mode=reference.REFERENCE_VIDEO_SIZE_MATCH_OUTPUT,
    )


def test_mode_is_optional_legacy_default_and_has_help():
    options, definition = modes.mode_input_definition()
    assert options == (modes.FOLLOW_TIMELINE, modes.REPEAT_REFERENCE)
    assert definition["default"] == modes.REPEAT_REFERENCE
    assert "Experimental" in definition["tooltip"]


@pytest.mark.parametrize("seconds", [5.0, 10.0, 15.0])
def test_follow_uses_natural_visible_frame_plan_not_queue_counter(seconds):
    value = source(2000, seconds)
    first = temporal.align_frame_count_up(round(seconds * 24))
    retained = first
    a = modes.select_follow_window(value, start_frame=0, visible_frames=first)
    shape = temporal.make_extension_shape(22, (round(2 * seconds * 24) - retained) / 24)
    b = modes.select_follow_window(value, start_frame=retained, visible_frames=shape.net_new_frames)
    assert b.source_start == a.source_stop == retained
    assert b.source_start != 0
    assert b.available_frames == shape.total_frames - 22
    # A future third group does not retroactively crop this source interval.
    again = modes.select_follow_window(value, start_frame=retained, visible_frames=shape.net_new_frames)
    assert again == b


@pytest.mark.parametrize("start,count", [(0, 243), (124, 238), (243, 119)])
def test_physical_terminal_window_is_not_split_or_context_shifted(start, count):
    window = modes.select_follow_window(source(), start_frame=start, visible_frames=count)
    assert window.source_start == start
    assert window.source_stop == start + count
    assert window.available_frames == count


@pytest.mark.parametrize("start,count", [(0, 124), (124, 119), (243, 238)])
def test_qwen_and_vae_receive_exactly_the_same_selected_interval(start, count):
    value = source()
    vae = RecordingVAE()
    assets, window = modes.encode_follow_video_group(vae, value, start_frame=start, visible_frames=count)
    sent = vae.inputs[0]
    assert torch.equal(sent[:count], value.frames[start:start + count])
    assert torch.equal(assets.item["data"], sent[::12])
    assert assets.item["timestamps"] == [i / 2 for i in range(len(sent[::12]))]
    assert assets.block["ref_audio_t"] == 0
    assert assets.block["audio_latent"] is None
    assert assets.block["latent"].device.type == "cpu"
    assert sent.shape[0] == window.encoded_frames


@pytest.mark.parametrize("available", [5, 6, 21, 22, 23, 119, 120, 124, 237, 240, 243])
def test_only_short_native_grid_padding_never_looping(available):
    value = source(available)
    vae = RecordingVAE()
    assets, window = modes.encode_follow_video_group(vae, value, start_frame=0, visible_frames=500)
    sent = vae.inputs[0]
    assert 0 <= sent.shape[0] - available <= 16
    assert sent.shape[0] % 17 == 5
    assert torch.equal(sent[:available], value.frames)
    if sent.shape[0] > available:
        assert torch.equal(sent[available:], value.frames[-1:].expand(sent.shape[0] - available, -1, -1, -1))
    assert window.status == "partial_source"


@pytest.mark.parametrize("remaining", [0, 1, 2, 3, 4])
def test_exhausted_or_too_short_reference_continues_without_video(remaining):
    value = source(120 + remaining)
    vae = RecordingVAE()
    calls = []
    def builder(**kwargs):
        calls.append(kwargs)
        return {("same prompt", False): "text-only"}
    cache, window = modes.build_follow_group_conditioning(
        value, vae, start_frame=120, visible_frames=124,
        conditioning_builder=builder, conditioning_kwargs={"prompts": ["same prompt"]},
    )
    assert calls[0]["timeline_video_assets"] is None
    assert not vae.inputs
    assert cache[("same prompt", False)] == "text-only"
    assert window.status == ("source_exhausted" if remaining == 0 else "short_remainder_skipped")


def test_repeated_prompt_uses_separate_cache_and_payload_for_each_group():
    value = source()
    vae = RecordingVAE()
    kwargs = {"prompts": ["the same text"]}
    def builder(**request):
        return {"the same text": request["timeline_video_assets"].item["data"]}
    a, wa = modes.build_follow_group_conditioning(
        value, vae, start_frame=0, visible_frames=124,
        conditioning_builder=builder, conditioning_kwargs=kwargs,
    )
    b, wb = modes.build_follow_group_conditioning(
        value, vae, start_frame=124, visible_frames=119,
        conditioning_builder=builder, conditioning_kwargs=kwargs,
    )
    assert a is not b
    assert wa.identity != wb.identity
    assert a["the same text"][0, 0, 0, 0] != b["the same text"][0, 0, 0, 0]
    assert kwargs == {"prompts": ["the same text"]}  # no caller mutation


def test_reference_cache_key_contains_slice_and_respects_vae_identity(monkeypatch):
    # Exercise the REAL encode_reference_video_cached function using a small
    # cache-double, not a claim about the production cache's resource behavior.
    class Cache:
        def __init__(self):
            self.items = {}
            self.keys = []
        def lookup(self, vae, key, event_sink=None):
            self.keys.append((vae, key))
            result = self.items.get((vae, key))
            if event_sink: event_sink("hit" if result is not None else "miss")
            return result
        def supports_vae(self, vae):
            return True
        def store(self, vae, key, value, event_sink=None):
            self.items[(vae, key)] = value
    cache = Cache()
    module = types.ModuleType("ComfyUI_H3_Continuum_Join.v3.ref_encode_cache")
    module.get_ref_encode_cache = lambda: cache
    module.make_ref_encode_cache_key = lambda *args: args
    monkeypatch.setitem(sys.modules, module.__name__, module)
    a, b = RecordingVAE(), RecordingVAE()
    value = source()
    for vae, start in ((a, 0), (a, 124), (a, 0), (b, 0)):
        modes.encode_follow_video_group(vae, value, start_frame=start, visible_frames=124, cache_enabled=True)
    assert len(a.inputs) == 2
    assert len(b.inputs) == 1
    assert cache.keys[0][1] == cache.keys[2][1]
    assert cache.keys[0][1] != cache.keys[1][1]
    assert cache.keys[0][0] is not cache.keys[3][0]


@pytest.mark.parametrize("n", [120, 124, 243])
def test_first_matching_interval_uses_legacy_encoder_with_exact_input_parity(n):
    raw = frames(n)
    old_source = reference.prepare_reference_video_source(raw, target_frames=temporal.align_frame_count_up(n), output_width=32, output_height=32, size_mode=reference.REFERENCE_VIDEO_SIZE_MATCH_OUTPUT)
    old_vae, new_vae = RecordingVAE(), RecordingVAE()
    old = reference.encode_reference_video(old_vae, old_source)
    new, _ = modes.encode_follow_video_group(new_vae, source(n), start_frame=0, visible_frames=n)
    assert torch.equal(old_vae.inputs[0], new_vae.inputs[0])
    assert torch.equal(old.item["data"], new.item["data"])
    assert old.item["timestamps"] == new.item["timestamps"]
    assert torch.equal(old.block["latent"], new.block["latent"])


def test_contract_stable_for_extension_distinct_from_repeat():
    value = source()
    again = source()
    assert value.contract == again.contract
    assert "chunks" not in value.contract and "total_duration" not in value.contract
    old = reference.prepare_reference_video_source(frames(), target_frames=124, output_width=32, output_height=32)
    assert value.combined_hash != old.combined_hash
    assert reference.combine_reference_video_identity("base", value) != reference.combine_reference_video_identity("base", old)
    assert reference.combine_reference_video_identity("base", None) == "base"


def test_source_is_owned_cpu_snapshot_and_future_pixels_change_identity():
    raw = frames()
    value = modes.prepare_follow_video_source(raw, chunk_seconds=5, output_width=32, output_height=32)
    saved = value.frames.clone()
    raw[-1].fill_(0.9)
    changed = modes.prepare_follow_video_source(raw, chunk_seconds=5, output_width=32, output_height=32)
    assert torch.equal(value.frames, saved)
    assert value.combined_hash != changed.combined_hash
    assert value.frames.device.type == "cpu" and value.frames.dtype == torch.float32


def test_alpha_is_not_video_reference_content():
    rgba = frames(22, 4)
    rgba[..., 3] = 0.9
    rgb = modes.prepare_follow_video_source(rgba[..., :3], chunk_seconds=5, output_width=32, output_height=32)
    alpha = modes.prepare_follow_video_source(rgba, chunk_seconds=5, output_width=32, output_height=32)
    assert alpha.combined_hash == rgb.combined_hash
    assert torch.equal(alpha.frames, rgb.frames)


@pytest.mark.parametrize("start,count", [(-1, 20), (0, 0), (True, 2), (0, False), (0.2, 10), (0, 10.5)])
def test_invalid_internal_range_fails_without_sampling(start, count):
    with pytest.raises(ValueError):
        modes.select_follow_window(source(22), start_frame=start, visible_frames=count)


def test_invalid_frame_payload_detected():
    with pytest.raises(reference.ReferenceVideoError):
        modes.prepare_follow_video_source(torch.zeros(5, 2), chunk_seconds=5, output_width=32, output_height=32)
    raw = frames(22)
    raw[0, 0, 0, 0] = float("nan")
    with pytest.raises(reference.ReferenceVideoError):
        modes.prepare_follow_video_source(raw, chunk_seconds=5, output_width=32, output_height=32)


def test_metadata_is_plain_json_and_does_not_contain_tensors():
    import json
    value = source()
    window = modes.select_follow_window(value, start_frame=124, visible_frames=119)
    assert json.loads(json.dumps(window.contract)) == window.contract
    assert json.loads(json.dumps(value.contract)) == value.contract
    assert modes.FOLLOW_PLAN_KEY.startswith("_h3_continuum_")


def test_tests_do_not_initialize_cuda():
    assert not torch.cuda.is_initialized()
