"""RR-R1: retain fixed Reference slot identity without changing V3.8 inputs."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from ComfyUI_H3_Continuum_Join.reference import (
    REFERENCE_SIZE_MATCH_OUTPUT,
    ReferenceAssets,
    ReferenceConditioningError,
    ReferenceImageBundle,
    encode_reference_latents,
    encode_reference_latents_cached,
    encode_reference_prompt,
    prepare_reference_assets,
    resolve_reference_image_inputs,
)
from ComfyUI_H3_Continuum_Join.v3.ref_encode_cache import clear_ref_encode_cache


def _image(value: float, *, size: int = 32) -> torch.Tensor:
    return torch.full((1, size, size, 3), value, dtype=torch.float32)


def _prepare(
    slots: dict[int, torch.Tensor],
    *,
    direct_legacy: dict[int, torch.Tensor] | None = None,
    output_size: int = 32,
) -> ReferenceAssets | None:
    direct_legacy = direct_legacy or {}
    bundle = ReferenceImageBundle(
        tuple(slots.get(index) for index in range(4, 10))
    ) if any(index >= 4 for index in slots) else None
    return prepare_reference_assets(
        reference_image_1=slots.get(1),
        reference_image_2=slots.get(2),
        reference_image_3=slots.get(3),
        reference_image_4=direct_legacy.get(4),
        reference_image_5=direct_legacy.get(5),
        image_references=bundle,
        output_width=output_size,
        output_height=output_size,
        size_mode=REFERENCE_SIZE_MATCH_OUTPUT,
    )


class RecordingVAE:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.full((1, 24, 1, 2, 2), float(image.mean().item()))


class RecordingClip:
    def __init__(self) -> None:
        self.items = None

    def tokenize(self, prompt, *, minimax_ref_items):
        self.items = minimax_ref_items
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros((1, 1)), {}]]


def test_all_empty_has_no_active_assets_but_nine_raw_positions():
    assert resolve_reference_image_inputs() == (None,) * 9
    assert _prepare({}) is None


@pytest.mark.parametrize(
    ("positions", "expected_ids"),
    [
        ((1,), ("R1",)),
        ((3,), ("R3",)),
        ((1, 3), ("R1", "R3")),
        ((4, 6, 9), ("R4", "R6", "R9")),
        ((2, 5, 9), ("R2", "R5", "R9")),
        ((9,), ("R9",)),
    ],
)
def test_sparse_slots_keep_source_ids_after_compaction(positions, expected_ids):
    slots = {position: _image(index / 10) for index, position in enumerate(positions, 1)}
    assets = _prepare(slots)

    assert assets is not None
    assert assets.source_slot_ids == expected_ids
    assert assets.count == len(positions)
    assert len(assets.source_slot_ids) == len(assets.images) == len(assets.image_hashes)
    assert all(latent is None for latent in assets.latents)
    for index, position in enumerate(positions):
        assert torch.equal(assets.images[index], slots[position])


def test_duplicate_image_keeps_two_slot_identities_and_two_presentations():
    image = _image(0.5)
    assets = _prepare({1: image, 4: image})

    assert assets is not None
    assert assets.source_slot_ids == ("R1", "R4")
    assert assets.count == 2
    assert assets.image_hashes[0] == assets.image_hashes[1]
    assert torch.equal(assets.images[0], assets.images[1])
    clear_ref_encode_cache()
    try:
        vae = RecordingVAE()
        encoded = encode_reference_latents_cached(vae, assets)
        assert encoded.source_slot_ids == ("R1", "R4")
        assert len(encoded.latents) == 2
        assert vae.calls == 1
    finally:
        clear_ref_encode_cache()


def test_legacy_direct_four_and_five_keep_ids_and_existing_collision_error():
    four, five = _image(0.25), _image(0.75)
    direct = _prepare({}, direct_legacy={4: four, 5: five})
    mixed = _prepare({5: five, 9: _image(1)}, direct_legacy={4: four})

    assert direct is not None and direct.source_slot_ids == ("R4", "R5")
    assert mixed is not None and mixed.source_slot_ids == ("R4", "R5", "R9")
    assert mixed.contract["reference_image_3"]["reference_position"] == 3
    assert mixed.contract["reference_image_3"]["sha256"] == mixed.image_hashes[2]
    assert "source_slot_ids" not in mixed.contract
    with pytest.raises(ReferenceConditioningError, match="connected through both"):
        _prepare({4: four}, direct_legacy={4: four})


def test_sparse_legacy_hash_and_contract_are_golden_compatible():
    assets = _prepare({1: _image(0), 3: _image(1)})

    assert assets is not None
    assert assets.source_slot_ids == ("R1", "R3")
    assert assets.image_hashes == (
        "f3c5e40f8a9e1c00c656265dd2ef1c2a58923cbbe538326ab3ac389e4aa0f66d",
        "fc2c027881d2ae8406cf551703cf5cdb281ce72913faa3c6702cfa2e9b8f3be1",
    )
    assert assets.combined_hash == (
        "7d0d92b68c0904cce938bb37232e3bd23a11b9265d902c6ea348e888729ad5e6"
    )
    assert assets.contract == {
        "reference_contract_version": 1,
        "count": 2,
        "size_mode": REFERENCE_SIZE_MATCH_OUTPUT,
        "image_hashes": list(assets.image_hashes),
        "combined_hash": assets.combined_hash,
    }


def test_original_slot_survives_resize_and_processed_content_hash(monkeypatch):
    comfy = ModuleType("comfy")
    utils = ModuleType("comfy.utils")
    calls = []

    def common_upscale(image, width, height, method, crop):
        calls.append((width, height, method, crop))
        return torch.nn.functional.interpolate(image, size=(height, width))

    utils.common_upscale = common_upscale
    comfy.utils = utils
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)

    assets = _prepare({5: _image(0.25, size=64), 9: _image(0.75, size=64)})

    assert assets is not None
    assert assets.source_slot_ids == ("R5", "R9")
    assert calls == [(32, 32, "lanczos", "disabled")] * 2
    assert [tuple(image.shape) for image in assets.images] == [(1, 32, 32, 3)] * 2
    assert [float(image.mean()) for image in assets.images] == [0.25, 0.75]
    assert assets.image_hashes[0] != assets.image_hashes[1]
    assert assets.contract["image_hashes"] == list(assets.image_hashes)


def test_slot_identity_survives_plain_and_cached_vae_encode():
    assets = _prepare({2: _image(0.25), 9: _image(0.75)})
    assert assets is not None
    plain_vae = RecordingVAE()
    plain = encode_reference_latents(plain_vae, assets)
    assert plain.source_slot_ids == ("R2", "R9")
    assert [float(latent.mean()) for latent in plain.latents] == [0.25, 0.75]
    assert plain_vae.calls == 2

    clear_ref_encode_cache()
    try:
        cached_vae = RecordingVAE()
        cold = encode_reference_latents_cached(cached_vae, assets)
        warm = encode_reference_latents_cached(cached_vae, assets)
        assert cold.source_slot_ids == warm.source_slot_ids == ("R2", "R9")
        assert [float(latent.mean()) for latent in warm.latents] == [0.25, 0.75]
        assert cached_vae.calls == 2
    finally:
        clear_ref_encode_cache()


def test_existing_qwen_and_dit_order_remains_compact_with_hybrid_anchors():
    assets = _prepare({3: _image(0.25), 9: _image(0.75)})
    assert assets is not None
    encoded = encode_reference_latents(RecordingVAE(), assets)
    clip = RecordingClip()
    first, last = _image(0), _image(1)

    conditioning = encode_reference_prompt(
        clip, "Use the references.", encoded, first_image=first, last_image=last
    )

    assert encoded.source_slot_ids == ("R3", "R9")
    assert all(
        item["data"] is expected
        for item, expected in zip(
            clip.items, (first, last, *encoded.images), strict=True
        )
    )
    refs = conditioning[0][1]["minimax_refs"]
    assert [float(item["latent"].mean()) for item in refs] == [0.25, 0.75]


def test_legacy_manual_assets_constructor_remains_valid_without_slot_metadata():
    assets = ReferenceAssets(
        images=(_image(0.5),),
        latents=(None,),
        image_hashes=("legacy",),
        combined_hash="legacy",
        size_mode=REFERENCE_SIZE_MATCH_OUTPUT,
    )
    assert assets.source_slot_ids == ()
