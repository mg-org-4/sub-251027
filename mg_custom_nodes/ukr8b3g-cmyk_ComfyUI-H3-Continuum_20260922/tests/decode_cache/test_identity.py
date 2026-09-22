import gc
import uuid
import weakref

import pytest
import torch

from decode_cache.identity import CacheBypass, ObjectTokens, make_key, selected_latent, tensor_digest, vae_signature
from decode_cache_test_fixtures import FakeVAE, NestedFixture


def delegate():
    pass


def test_same_content_different_object_same_key(samples):
    other = {"samples": samples["samples"].clone()}
    assert make_key(samples, "video", "s") == make_key(other, "video", "s")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.int16])
def test_tensor_bytes_all_dtypes(dtype):
    x = torch.arange(60).to(dtype).reshape(3, 4, 5)
    assert tensor_digest(x) == tensor_digest(x.clone())
    x[0, 0, 0] += 1
    assert tensor_digest(x) != tensor_digest(torch.arange(60).to(dtype).reshape(3, 4, 5))


@pytest.mark.parametrize("shape", [(), (0,), (1,), (3, 4), (1, 2, 8), (2, 1, 3, 4)])
def test_block_hash_matches_raw(shape):
    import hashlib
    from decode_cache.identity import tensor_blocks
    x = torch.ones(shape)
    actual = hashlib.sha256(b"".join(bytes(b) for b in tensor_blocks(x, 9))).hexdigest()
    expected = hashlib.sha256(x.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    assert actual == expected


def test_noncontiguous_order():
    x = torch.arange(120).reshape(4, 5, 6).transpose(0, 2)
    assert tensor_digest(x) == tensor_digest(x.contiguous())


@pytest.mark.parametrize("change", ["content", "dtype", "shape", "vae", "stream", "reset", "sample_rate", "extra"])
def test_key_invalidations(samples, change):
    kwargs = {"samples": samples, "stream": "video", "signature": "s", "reset_token": 0}
    before = make_key(**kwargs)
    if change == "content":
        samples["samples"][0, 0, 0, 0] += 1
    elif change == "dtype":
        samples["samples"] = samples["samples"].double()
    elif change == "shape":
        samples["samples"] = samples["samples"].reshape(6, 2, 5, 2)
    elif change == "vae":
        kwargs["signature"] = "other"
    elif change == "stream":
        kwargs["stream"] = "audio"
    elif change == "reset":
        kwargs["reset_token"] = 1
    elif change == "sample_rate":
        samples["sample_rate"] = 22050
    else:
        samples["noise_mask"] = torch.zeros(1)
    assert make_key(**kwargs) != before


def test_audio_rate_override_key():
    x = torch.zeros(1, 2, 8)
    assert make_key({"samples": x, "sample_rate": 16000}, "audio", "s") != make_key({"samples": x, "sample_rate": 32000}, "audio", "s")


def test_nested_only_selected_stream():
    x, y = torch.ones(1, 24, 2, 2, 2), torch.ones(1, 32, 2, 4)
    s = {"samples": NestedFixture(x, y)}
    v = make_key(s, "video", "s")
    a = make_key(s, "audio", "s")
    y.add_(3)
    assert make_key(s, "video", "s") == v
    assert make_key(s, "audio", "s") != a
    assert selected_latent(s, "audio") is y


def test_opaque_metadata_bypass(samples):
    samples["opaque"] = object()
    with pytest.raises(CacheBypass):
        make_key(samples, "video", "s")


def test_inference_tensor_hash_no_version_required():
    with torch.inference_mode():
        x = torch.ones(4)
        assert tensor_digest(x) == tensor_digest(x.clone())


def test_identity_does_not_keep_vae_alive():
    vae = FakeVAE()  # no pytest fixture owner remains alive
    signature, refs = vae_signature(vae, "video", delegate)
    ref = weakref.ref(vae)
    del vae
    gc.collect()
    assert ref() is None
    assert refs[0]() is None


def test_object_tokens_not_address_only():
    tokens = ObjectTokens()
    a, b = FakeVAE(), FakeVAE()
    assert tokens.token(a) == tokens.token(a)
    assert tokens.token(a) != tokens.token(b)


@pytest.mark.parametrize("change", ["patch_uuid", "weight_copy", "buffer_copy", "tiling", "dtype", "new_vae"])
def test_vae_signature_changes(vae, change):
    a = vae_signature(vae, "video", delegate)[0]
    if change == "patch_uuid":
        vae.patcher.patches_uuid = uuid.uuid4()
    elif change == "weight_copy":
        vae.first_stage_model.weight.copy_(torch.ones(1)*2)
    elif change == "buffer_copy":
        vae.first_stage_model.offset.add_(1)
    elif change == "tiling":
        vae.first_stage_model.tile_size = 512
    elif change == "dtype":
        vae.vae_dtype = torch.float16
    else:
        vae = FakeVAE()
    assert vae_signature(vae, "video", delegate)[0] != a


def test_unknown_vae_and_training_bypass(vae):
    with pytest.raises(CacheBypass):
        vae_signature(object(), "video", delegate)
    vae.first_stage_model.train()
    with pytest.raises(CacheBypass):
        vae_signature(vae, "video", delegate)


def test_relocation_buffer_identity_not_used(vae):
    a = vae_signature(vae, "video", delegate)[0]
    vae.first_stage_model.offset = vae.first_stage_model.offset.clone()
    assert vae_signature(vae, "video", delegate)[0] == a


def test_empty_native_model_options_allowed_but_hooks_bypass(vae):
    vae.patcher.model_options = {"transformer_options": {}}
    vae_signature(vae, "video", delegate)
    vae.patcher.model_options["transformer_options"]["wrapper"] = object()
    with pytest.raises(CacheBypass):
        vae_signature(vae, "video", delegate)


def test_forward_hook_disables_cache_only(vae):
    handle = vae.first_stage_model.register_forward_hook(lambda *args: None)
    try:
        with pytest.raises(CacheBypass):
            vae_signature(vae, "video", delegate)
    finally:
        handle.remove()


def test_inference_context_has_separate_identity(vae):
    before = vae_signature(vae, "video", delegate)[0]
    with torch.inference_mode():
        assert vae_signature(vae, "video", delegate)[0] != before
