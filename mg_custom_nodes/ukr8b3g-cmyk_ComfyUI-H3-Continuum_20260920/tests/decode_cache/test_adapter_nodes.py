import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from decode_cache.service import DecodeService, native_delegate
from decode_cache.identity import selected_latent, vae_signature
from decode_cache.nodes import H3DecodeCacheHelper, NODE_CLASS_MAPPINGS
from decode_cache.store import DecodeStore, MiB
from decode_cache_test_fixtures import FakeVAE, NestedFixture


def fake_video(vae, samples):
    return vae.decode(selected_latent(samples, "video"))


def fake_audio(vae, samples):
    # A controlled Core-helper surrogate returns sample-rate metadata as Core does.
    return {"waveform": vae.decode(selected_latent(samples, "audio")),
            "sample_rate": samples.get("sample_rate", vae.audio_sample_rate)}


def resolver(stream):
    f = fake_video if stream == "video" else fake_audio
    return f, f


@pytest.fixture
def service(tmp_path):
    store = DecodeStore(temp_parent=tmp_path, memory_probe=lambda: 10**12,
                        disk_probe=lambda p: SimpleNamespace(free=10**12), headroom_bytes=0)
    store.configure("Auto", MiB, 8*MiB, 0)
    service = DecodeService(store=store, delegate_resolver=resolver, interrupt=lambda: None)
    yield service
    service.store.close()


def test_delegate_called_once_warm(service, vae, samples):
    a, e = service.decode(samples, vae, "video")
    b, f = service.decode(samples, vae, "video")
    assert e["status"] == "miss" and f["status"] == "hit"
    assert vae.calls == 1 and torch.equal(a, b)


def test_audio_metadata_override_and_mutation(service):
    vae = FakeVAE("audio")
    samples = {"samples": torch.ones(1, 2, 40), "sample_rate": 22050}
    a, _ = service.decode(samples, vae, "audio")
    a["waveform"].zero_()
    a["sample_rate"] = 5
    b, e = service.decode(samples, vae, "audio")
    assert e["status"] == "hit"
    assert b["sample_rate"] == 22050 and b["waveform"].sum() == 80
    samples["sample_rate"] = 32000
    _, e = service.decode(samples, vae, "audio")
    assert e["status"] == "miss"


def test_core_error_not_retried(service, vae, samples):
    calls = []
    def broken(vae, samples):
        calls.append(1)
        raise RuntimeError("native decode failed")
    service.delegate_resolver = lambda stream: (broken, broken)
    with pytest.raises(RuntimeError, match="native decode failed"):
        service.decode(samples, vae, "video")
    assert len(calls) == 1


@pytest.mark.parametrize("point", ["signature", "get", "put"])
def test_cache_fail_open(service, vae, samples, monkeypatch, point):
    def broken(*args, **kwargs):
        raise OSError("fixture-only cache failure")
    if point == "signature":
        service.signature_fn = broken
    else:
        monkeypatch.setattr(service.store, point, broken)
    result, event = service.decode(samples, vae, "video")
    assert torch.equal(result, samples["samples"])
    assert vae.calls == 1


def test_interrupt_not_swallowed(service, vae, samples, monkeypatch):
    class InterruptProcessingException(Exception):
        pass
    def interrupted(*args):
        raise InterruptProcessingException()
    monkeypatch.setattr(service.store, "get", interrupted)
    with pytest.raises(InterruptProcessingException):
        service.decode(samples, vae, "video")
    assert vae.calls == 0


def test_unsupported_vae_still_decodes(service, samples):
    class Unknown:
        def decode(self, tensor):
            return tensor.clone()
    result, event = service.decode(samples, Unknown(), "video")
    assert event["status"] == "bypass" and torch.equal(result, samples["samples"])


def test_vae_change_during_decode_skips_store(service, vae, samples):
    def changing(vae, samples):
        vae.first_stage_model.tile_size += 1
        return vae.decode(selected_latent(samples, "video"))
    service.delegate_resolver = lambda stream: (changing, changing)
    _, event = service.decode(samples, vae, "video")
    assert "store_skipped" in event and not service.store.entries


def test_q1_q2_q3_q4_physical_lists(service):
    vae = FakeVAE()
    groups = [{"samples": torch.ones(124 if i == 0 else 141, 2, 2, 3)*i} for i in range(3)]
    statuses = []
    for sequence in ([groups[0]], [groups[1]], groups, groups):
        statuses.append([service.decode(s, vae, "video")[1]["status"] for s in sequence])
    assert statuses == [["miss"], ["miss"], ["hit", "hit", "miss"], ["hit", "hit", "hit"]]
    assert vae.calls == 3


def test_terminal_entry_never_sliced(service):
    vae = FakeVAE()
    x = torch.rand(260, 2, 2, 3)
    for _ in range(2):
        out, _ = service.decode({"samples": x}, vae, "video")
        assert torch.equal(out, x)
    assert vae.calls == 1


def test_retry_changed_group_only_misses(service):
    vae = FakeVAE()
    a = {"samples": torch.ones(8, 2, 2, 3)}
    b = {"samples": torch.ones(8, 2, 2, 3)*2}
    service.decode(a, vae, "video")
    service.decode(b, vae, "video")
    b["samples"].add_(1)
    assert service.decode(a, vae, "video")[1]["status"] == "hit"
    assert service.decode(b, vae, "video")[1]["status"] == "miss"


def test_native_functions_are_delegated(monkeypatch, vae, samples):
    calls = []
    mod = ModuleType("nodes")
    class VAEDecode:
        def decode(self, vae, samples):
            calls.append(("video", vae, samples))
            return (samples["samples"],)
    mod.VAEDecode = VAEDecode
    monkeypatch.setitem(sys.modules, "nodes", mod)
    audio = ModuleType("comfy_extras.nodes_audio")
    def audio_helper(vae, samples):
        calls.append(("audio", vae, samples))
        return {"waveform": samples["samples"], "sample_rate": 12345}
    audio.vae_decode_audio = audio_helper
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_audio", audio)
    assert native_delegate("video")[1](vae, samples) is samples["samples"]
    assert native_delegate("audio")[1](vae, samples)["sample_rate"] == 12345
    assert len(calls) == 2


def test_one_public_node_and_nan():
    assert list(NODE_CLASS_MAPPINGS) == ["H3DecodeCacheHelper"]
    assert H3DecodeCacheHelper.INPUT_IS_LIST
    assert H3DecodeCacheHelper.OUTPUT_IS_LIST == (True, True, False)
    assert math.isnan(H3DecodeCacheHelper.IS_CHANGED())


@pytest.mark.parametrize("streams", ["video", "audio", "both", "none"])
def test_node_optional_independent_lists(service, streams):
    node = H3DecodeCacheHelper()
    node.service = service
    kwargs = {}
    if streams in ("video", "both"):
        kwargs.update(video_samples=[{"samples": torch.ones(4, 2, 2, 3)}]*2, video_vae=[FakeVAE()])
    if streams in ("audio", "both"):
        kwargs.update(audio_samples=[{"samples": torch.ones(1, 2, 16)}], audio_vae=[FakeVAE("audio")])
    images, audio, report = node.decode(["Auto"], [1], [1], [0], **kwargs)
    assert len(images) == (2 if streams in ("video", "both") else 0)
    assert len(audio) == (1 if streams in ("audio", "both") else 0)
    assert json.loads(report)["mode"] == "Auto"


def test_none_position_no_broadcast(service):
    node = H3DecodeCacheHelper()
    node.service = service
    images, _, _ = node.decode(["Auto"], [1], [1], [0],
                             video_samples=[None, {"samples": torch.ones(4, 2, 2, 3)}], video_vae=[FakeVAE()])
    assert images[0] is None and images[1].shape == (4, 2, 2, 3)


def test_off_disables_content_cache(service, vae, samples):
    node = H3DecodeCacheHelper()
    node.service = service
    for _ in range(2):
        node.decode(["Off"], [1], [0], [0], video_samples=[samples], video_vae=[vae])
    assert vae.calls == 2 and not service.store.entries


def test_cache_configuration_fault_is_nonfatal(service, vae, samples, monkeypatch):
    node = H3DecodeCacheHelper()
    node.service = service
    def bad(*a):
        raise OSError("memory probe unavailable")
    monkeypatch.setattr(service.store, "configure", bad)
    result = node.decode(["Auto"], [1], [1], [0], video_samples=[samples], video_vae=[vae])
    assert vae.calls == 1
    assert "cache_disabled_reason" in json.loads(result[2])


def test_builtin_helper_is_publicly_registered():
    from ComfyUI_H3_Continuum_Join import nodes as continuum_nodes

    assert "H3DecodeCacheHelper" in continuum_nodes.NODE_CLASS_MAPPINGS
    assert continuum_nodes.NODE_DISPLAY_NAME_MAPPINGS["H3DecodeCacheHelper"] == "Decode Cache Helper"


def test_forbidden_runtime_imports():
    import ast
    root = Path(__file__).resolve().parents[2]/"decode_cache"
    forbidden = {"run_storage", "runtime_coordinator", "execution_planner", "sampling_engine"}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not forbidden.intersection((node.module or "").split("."))


def test_cache_memory_error_returns_already_decoded_result(service, vae, samples, monkeypatch):
    def no_memory(*args, **kwargs):
        raise MemoryError("cache allocation failed")
    monkeypatch.setattr(service.store, "put", no_memory)
    out, event = service.decode(samples, vae, "video")
    assert torch.equal(out, samples["samples"])
    assert vae.calls == 1 and "cache_write_error" in event


def test_native_oom_propagates_exactly_once(service, vae, samples):
    calls = []
    def oom(*args, **kwargs):
        calls.append(1)
        raise torch.OutOfMemoryError("native fixture OOM")
    service.delegate_resolver = lambda stream: (oom, oom)
    with pytest.raises(torch.OutOfMemoryError):
        service.decode(samples, vae, "video")
    assert calls == [1]


def test_cancel_during_cache_store_is_not_a_native_retry(service, vae, samples, monkeypatch):
    class InterruptProcessingException(Exception):
        pass
    def canceled(*args, **kwargs):
        raise InterruptProcessingException("cancel fixture")
    monkeypatch.setattr(service.store, "put", canceled)
    with pytest.raises(InterruptProcessingException):
        service.decode(samples, vae, "video")
    assert vae.calls == 1


def test_two_simultaneous_node_calls_serialize_one_miss(service, vae, samples):
    from concurrent.futures import ThreadPoolExecutor
    node = H3DecodeCacheHelper()
    node.service = service
    def run():
        return node.decode(["Auto"], [1], [1], [0], video_samples=[samples], video_vae=[vae])
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(lambda _: run(), range(2)))
    assert vae.calls == 1
    assert torch.equal(result[0][0][0], result[1][0][0])
