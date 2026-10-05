"""R0/Fixed-only cache parity; no Sampling or CUDA is used by these tests."""
from types import SimpleNamespace

import pytest
import torch

from ComfyUI_H3_Continuum_Join.v2 import h3_builder, sequence
from ComfyUI_H3_Continuum_Join.v3 import reference_runtime
from ComfyUI_H3_Continuum_Join.v3.reference_storage_contract import (
    ReferenceStorageContractError, build_reference_storage_plan,
)
from .test_v39_rr_r4_reference_runtime import _runtime, _image, _Clip, _VAE


def _clip():
    clip = _Clip()
    clip.patcher = SimpleNamespace(patches_uuid="patch-a")
    return clip


def _prepare(runtime, clip, *, group=1, prompt="running", first=None, last=None, **kwargs):
    return runtime.prepare_group(
        physical_group=group, prompt=prompt, clip=clip, video_vae=_VAE(),
        first_image=first, last_image=last, prompt_conditioning_cache=True,
        **kwargs,
    )


@pytest.mark.parametrize("first", [None, _image(0.1)])
def test_same_fixed_payload_hits_and_preserves_group_specific_contract(first):
    router = _runtime(slots={}, selectors={}, chunks=2)
    clip, events = _clip(), []
    one = _prepare(router, clip, first=first, prompt_cache_event=events.append)
    two = _prepare(router, clip, group=2, first=first, prompt_cache_event=events.append)
    assert len(clip.calls) == 1 and events == ["miss", "hit"]
    assert one.conditioning is not two.conditioning
    assert torch.equal(one.conditioning[0][0], two.conditioning[0][0])
    assert one.conditioning[0][1] == two.conditioning[0][1]
    assert one.physical_group == 1 and two.physical_group == 2
    assert router.observed_group_contracts[1] != router.observed_group_contracts[2]


def test_cache_is_identical_to_v38_and_shares_existing_clip_cache():
    first = _image(0.1)
    clip, events = _clip(), []
    assets = h3_builder.IdentityAssets(first, None, None, None, h3_builder._tensor_fingerprint(first))
    legacy = sequence._conditioning_cache(
        clip=clip, prompts=["running"], assets=assets, final_has_last_frame=False,
        prompt_conditioning_cache=True, prompt_cache_event=events.append,
    )[("running", False)]
    result = _prepare(_runtime(slots={}, selectors={}), clip, first=first,
                      prompt_cache_event=events.append)
    assert len(clip.calls) == 1 and events == ["miss", "hit"]
    assert torch.equal(legacy[0][0], result.conditioning[0][0])
    assert legacy[0][1] == result.conditioning[0][1]


def test_returned_metadata_cannot_poison_cache_or_next_group():
    router, clip = _runtime(slots={}, selectors={}), _clip()
    one = _prepare(router, clip)
    one.conditioning[0][1]["minimax_keyframes"] = [{"bad": True}]
    two = _prepare(router, clip, group=2)
    assert "minimax_keyframes" not in two.conditioning[0][1]
    assert len(clip.calls) == 1


@pytest.mark.parametrize("change", ["prompt", "first", "last", "patch", "layer", "clip"])
def test_changed_payload_or_clip_never_reuses_old_conditioning(change):
    router, clip = _runtime(slots={}, selectors={}, chunks=2), _clip()
    first, last = _image(0.1), _image(0.9)
    _prepare(router, clip, first=first, last=last)
    prompt, target = "running", clip
    if change == "prompt": prompt = "walking"
    if change == "first": first.fill_(0.2)
    if change == "last": last.fill_(0.8)
    if change == "patch": clip.patcher.patches_uuid = "patch-b"
    if change == "layer": clip.layer_idx = -2
    if change == "clip": target = _clip()
    events = []
    _prepare(router, target, group=2, prompt=prompt, first=first, last=last,
             prompt_cache_event=events.append)
    assert events == ["miss"]
    assert len(clip.calls) + (len(target.calls) if target is not clip else 0) == 2


def test_last_image_is_cached_by_actual_presentation_not_just_presence():
    router, clip = _runtime(slots={}, selectors={}, chunks=3), _clip()
    first, last = _image(0.1), _image(0.9)
    events = []
    for group in (1, 2, 3):
        _prepare(router, clip, group=group, first=first, last=last,
                 include_last_image=group == 3, prompt_cache_event=events.append)
    assert events == ["miss", "hit", "miss"]
    assert len(clip.calls[0][1]["images"]) == 1
    assert len(clip.calls[1][1]["images"]) == 2


def test_any_selected_reference_in_schedule_keeps_existing_direct_path():
    router = _runtime(slots={"R1": _image(0.2)}, selectors={"R1": "2"}, chunks=3)
    clip = _clip()
    for group in (1, 2, 3):
        _prepare(router, clip, group=group)
    assert len(clip.calls) == 3
    assert not hasattr(clip, h3_builder._PROMPT_CACHE_ATTR)


@pytest.mark.parametrize("asset", ["reference_audio_assets", "timeline_video_assets"])
def test_multimodal_inputs_bypass_without_changing_direct_encode_kwargs(monkeypatch, asset):
    calls = []
    marker = object()
    def direct(clip, prompt, **kwargs):
        calls.append(kwargs)
        return [[torch.zeros(1), {"prompt": prompt}]]
    monkeypatch.setattr(reference_runtime, "encode_prompt_conditioning", direct)
    router, clip = _runtime(slots={}, selectors={}), _clip()
    for group in (1, 2):
        _prepare(router, clip, group=group, **{asset: marker})
    assert len(calls) == 2 and all(call[asset] is marker for call in calls)
    assert not hasattr(clip, h3_builder._PROMPT_CACHE_ATTR)


def test_special_clip_modes_fall_back_without_new_error_or_stop():
    router, clip = _runtime(slots={}, selectors={}), _clip()
    clip.use_clip_schedule = True
    events = []
    for group in (1, 2):
        _prepare(router, clip, group=group, prompt_cache_event=events.append)
    assert len(clip.calls) == 2 and events == ["bypass_special_clip"] * 2


def test_cuda_conditioning_is_not_retained_by_routed_cache(monkeypatch):
    monkeypatch.setattr(h3_builder, "_contains_cuda_tensor", lambda _: True)
    router, clip = _runtime(slots={}, selectors={}), _clip()
    events = []
    for group in (1, 2):
        _prepare(router, clip, group=group, prompt_cache_event=events.append)
    assert len(clip.calls) == 2 and events == ["bypass_cuda_output"] * 2


def test_saved_contract_preflight_is_not_skipped_by_cache_hit():
    router, clip = _runtime(slots={}, selectors={}, chunks=2), _clip()
    plan = build_reference_storage_plan(runtime=router, prompts=["running"] * 2,
                                       first_frame_hash="none", last_frame_hash="none")
    _prepare(router, clip, expected_group_contract=plan.group_for(1))
    with pytest.raises(ReferenceStorageContractError, match="changed after"):
        _prepare(router, clip, group=2, prompt="changed",
                 expected_group_contract=plan.group_for(2))
    assert len(clip.calls) == 1


def test_default_disabled_retains_existing_two_encodes():
    router, clip = _runtime(slots={}, selectors={}, chunks=2), _clip()
    for group in (1, 2):
        router.prepare_group(physical_group=group, prompt="running", clip=clip,
                             video_vae=_VAE(), first_image=None, last_image=None)
    assert len(clip.calls) == 2
