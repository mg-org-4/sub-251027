"""RR-R4 selected-Reference preparation and Qwen/DiT conditioning contracts."""

from __future__ import annotations

import pytest
import torch

from ComfyUI_H3_Continuum_Join.reference import REFERENCE_SIZE_MATCH_OUTPUT
from ComfyUI_H3_Continuum_Join.v3.reference_routing import (
    REFERENCE_SLOT_IDS,
    compile_reference_routing_schedule,
)
from ComfyUI_H3_Continuum_Join.v3.reference_runtime import (
    ReferenceInputSet,
    ReferenceRoutingRuntime,
    ReferenceRuntimeContractError,
    rewrite_reference_tags,
)
from ComfyUI_H3_Continuum_Join.v3.reference_effective_plan import compile_group_picture_map
from ComfyUI_H3_Continuum_Join.v3.reference_storage_contract import (
    ReferenceStorageContractError, build_reference_storage_plan,
)


def _image(value: float):
    return torch.full((1, 64, 64, 3), value, dtype=torch.float32)


class _VAE:
    def __init__(self):
        self.values = []

    def encode(self, image):
        value = float(image.mean())
        self.values.append(value)
        return torch.full((1, 24, 1, 4, 4), value)


class _Clip:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros((1, 1, 2)), {"prompt": tokens}]]


def _selectors(**overrides):
    selectors = {slot_id: "off" for slot_id in REFERENCE_SLOT_IDS}
    selectors.update(overrides)
    return selectors


def _runtime(*, slots, selectors, chunks=3, terminal=False):
    images = [None] * 9
    for slot_id, image in slots.items():
        images[int(slot_id[1:]) - 1] = image
    inputs = ReferenceInputSet(
        tuple(images), 64, 64, REFERENCE_SIZE_MATCH_OUTPUT,
    )
    schedule = compile_reference_routing_schedule(
        total_chunks=chunks,
        terminal_merge_enabled=terminal,
        mode="Custom",
        selectors_by_slot=_selectors(**selectors),
    )
    return ReferenceRoutingRuntime(schedule=schedule, inputs=inputs)


def test_each_group_reaches_only_its_selected_image_in_vae_qwen_and_dit():
    a, b = _image(0.25), _image(0.75)
    # R9 is deliberately not a Tensor. An unselected image must not be validated.
    runtime = _runtime(
        slots={"R1": a, "R4": b, "R9": object()},
        selectors={"R1": "1", "R4": "2"},
    )
    vae, clip = _VAE(), _Clip()
    first = _image(0.1)

    first_group = runtime.prepare_group(
        physical_group=1, prompt="@R1 runs; <Subject 1> watches",
        clip=clip, video_vae=vae, first_image=first, last_image=None,
    )
    assert first_group.selected_assets.source_slot_ids == ("R1",)
    assert first_group.effective_prompt == "<Picture 2> runs; <Subject 1> watches"
    assert first_group.picture_map.references[0].source_slot_id == "R1"
    assert first_group.picture_map.references[0].picture_number == 2
    assert len(vae.values) == 1 and vae.values[0] == pytest.approx(0.25)
    first_items = [item["data"] for item in clip.calls[0][1]["minimax_ref_items"]]
    assert first_items[0] is first and torch.equal(first_items[1], a)
    assert len(first_group.conditioning[0][1]["minimax_refs"]) == 1
    assert float(first_group.conditioning[0][1]["minimax_refs"][0]["latent"].mean()) == pytest.approx(0.25)

    second_group = runtime.prepare_group(
        physical_group=2, prompt="@R4 runs; <Subject 1> watches",
        clip=clip, video_vae=vae, first_image=first, last_image=None,
    )
    assert second_group.selected_assets.source_slot_ids == ("R4",)
    assert second_group.effective_prompt == "<Picture 2> runs; <Subject 1> watches"
    assert vae.values == pytest.approx([0.25, 0.75])
    second_items = [item["data"] for item in clip.calls[1][1]["minimax_ref_items"]]
    assert second_items[0] is first and torch.equal(second_items[1], b)
    assert float(second_group.conditioning[0][1]["minimax_refs"][0]["latent"].mean()) == pytest.approx(0.75)


def test_empty_custom_group_never_falls_back_to_all_references():
    runtime = _runtime(slots={"R1": _image(0.25)}, selectors={"R1": "1"})
    vae, clip = _VAE(), _Clip()
    result = runtime.prepare_group(
        physical_group=2, prompt="no reference here", clip=clip, video_vae=vae,
        first_image=None, last_image=None,
    )
    assert result.selected_assets is None
    assert result.picture_map.references == ()
    assert result.conditioning[0][1].get("minimax_refs") is None
    assert clip.calls == [("no reference here", {"images": []})]
    assert vae.values == []


def test_empty_nonfinal_group_keeps_legacy_last_image_timing():
    runtime = _runtime(slots={}, selectors={}, chunks=2)
    first, last = _image(0.1), _image(0.9)
    clip = _Clip()
    initial = runtime.prepare_group(
        physical_group=1, prompt="running", clip=clip, video_vae=_VAE(),
        first_image=first, last_image=last, include_last_image=False,
    )
    final = runtime.prepare_group(
        physical_group=2, prompt="running", clip=clip, video_vae=_VAE(),
        first_image=first, last_image=last, include_last_image=True,
    )
    assert initial.picture_map.last_picture_number is None
    assert final.picture_map.last_picture_number == 2
    assert clip.calls[0][1]["images"] == [first]
    assert clip.calls[1][1]["images"] == [first, last]


def test_same_prompt_different_group_reference_is_encoded_twice():
    runtime = _runtime(
        slots={"R1": _image(0.25), "R4": _image(0.75)},
        selectors={"R1": "1", "R4": "2"},
    )
    vae, clip = _VAE(), _Clip()
    results = [
        runtime.prepare_group(
            physical_group=group, prompt="running", clip=clip, video_vae=vae,
            first_image=None, last_image=None,
        )
        for group in (1, 2)
    ]
    assert len(clip.calls) == 2
    assert results[0].picture_map.references[0].image_sha256 != results[1].picture_map.references[0].image_sha256
    assert results[0].conditioning is not results[1].conditioning


def test_disconnected_selected_slot_warns_without_changing_prompt_or_sampling_contract():
    runtime = _runtime(slots={}, selectors={"R7": "1"}, chunks=1)
    result = runtime.prepare_group(
        physical_group=1, prompt="@R7 moves", clip=_Clip(), video_vae=_VAE(),
        first_image=None, last_image=None,
    )
    assert result.effective_prompt == "@R7 moves"
    assert result.selected_assets is None
    assert any("R7 is routed" in warning for warning in result.warnings)
    assert any("@R tag was left unchanged" in warning for warning in result.warnings)


def test_duplicate_content_keeps_two_fixed_slot_identities_and_picture_numbers():
    image = _image(0.4)
    runtime = _runtime(
        slots={"R1": image, "R4": image},
        selectors={"R1": "all", "R4": "all"}, chunks=1,
    )
    result = runtime.prepare_group(
        physical_group=1, prompt="@R1 and @R4", clip=_Clip(), video_vae=_VAE(),
        first_image=None, last_image=None,
    )
    refs = result.picture_map.references
    assert [(item.source_slot_id, item.picture_number) for item in refs] == [
        ("R1", 1), ("R4", 2)
    ]
    assert refs[0].image_sha256 == refs[1].image_sha256
    assert result.effective_prompt == "<Picture 1> and <Picture 2>"


def test_terminal_conflict_is_not_silently_merged_or_split():
    runtime = _runtime(
        slots={"R1": _image(0.25), "R4": _image(0.75)},
        selectors={"R1": "2", "R4": "3"}, terminal=True,
    )
    with pytest.raises(ReferenceRuntimeContractError, match="conflicting terminal"):
        runtime.check_physical_contract(chunks=3, terminal_merge_enabled=True)
    with pytest.raises(ReferenceRuntimeContractError, match="no single Reference route"):
        runtime.prepare_group(
            physical_group=2, prompt="running", clip=_Clip(), video_vae=_VAE(),
            first_image=None, last_image=None,
        )


def test_raw_identity_excludes_never_selected_image_and_changes_with_route():
    bad_unselected = object()
    left = _runtime(
        slots={"R1": _image(0.25), "R9": bad_unselected},
        selectors={"R1": "1"}, chunks=1,
    )
    right = _runtime(
        slots={"R1": _image(0.25), "R9": bad_unselected},
        selectors={"R9": "1"}, chunks=1,
    )
    # Neither selected nor unselected raw image content is touched merely to
    # construct an RR-R4 queue identity; RR-R5 owns persistent reuse identity.
    assert len(left.global_identity("base")) == 64
    right._run_nonce = left._run_nonce
    assert left.global_identity("base") != right.global_identity("base")


def test_reference_rewrite_is_exact_and_never_rewrites_subject_or_partial_tokens():
    picture_map = compile_group_picture_map(
        source_slot_ids=("R4",),
        reference_image_hashes=("a" * 64,),
        has_first_image=True,
        has_last_image=True,
    )
    effective, warnings = rewrite_reference_tags(
        "@R4 <Subject 4> @R5 @R10 X@R4 @R4suffix", picture_map
    )
    assert effective == "<Picture 3> <Subject 4> @R5 @R10 X@R4 @R4suffix"
    assert len(warnings) == 1 and "R5" in warnings[0]


def test_rr_r5_changed_selected_image_is_rejected_before_vae_or_prompt_encode():
    image = _image(0.25)
    runtime = _runtime(slots={"R1": image}, selectors={"R1": "1"}, chunks=1)
    plan = build_reference_storage_plan(
        runtime=runtime, prompts=["@R1 runs"],
        first_frame_hash="none", last_frame_hash="none",
    )
    image.fill_(0.75)
    vae, clip = _VAE(), _Clip()
    with pytest.raises(ReferenceStorageContractError, match="changed after"):
        runtime.prepare_group(
            physical_group=1, prompt="@R1 runs", clip=clip, video_vae=vae,
            first_image=None, last_image=None,
            expected_group_contract=plan.group_for(1),
        )
    assert vae.values == []
    assert clip.calls == []
