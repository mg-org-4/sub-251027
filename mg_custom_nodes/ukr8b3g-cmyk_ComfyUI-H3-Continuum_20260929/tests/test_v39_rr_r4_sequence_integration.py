"""RR-R4 hidden Sequence integration; public V3.8 remains on its old path."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from ComfyUI_H3_Continuum_Join.constants import (
    DIAGNOSTICS_BASIC,
    DIAGNOSTICS_OFF,
    PROMPT_MODE_FIXED,
    V2_CONTINUITY_OPTIONS,
)
from ComfyUI_H3_Continuum_Join.temporal import audio_latent_t, video_latent_t
from ComfyUI_H3_Continuum_Join.v2 import sequence
from ComfyUI_H3_Continuum_Join.v2.h3_builder import IdentityAssets, _tensor_fingerprint
from ComfyUI_H3_Continuum_Join.v2.prompts import make_prompt_plan
from ComfyUI_H3_Continuum_Join.v3.reference_routing import (
    REFERENCE_SLOT_IDS,
    compile_reference_routing_schedule,
)
from ComfyUI_H3_Continuum_Join.v3.reference_runtime import (
    ReferenceInputSet,
    ReferenceRoutingRuntime,
)


class _Nested:
    def __init__(self, parts):
        self.parts = tuple(parts)

    def unbind(self):
        return self.parts


class _Clip:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros((1, 1, 2)), {"prompt": tokens}]]


class _VAE:
    def __init__(self):
        self.values = []

    def encode(self, image):
        value = float(image.mean())
        self.values.append(value)
        return torch.full((1, 24, 1, 4, 4), value)


def _image(value):
    return torch.full((1, 64, 64, 3), value)


def _runtime(monkeypatch, *, first=None, last=None):
    first_latent = torch.full((1, 24, 1, 4, 4), 0.1) if first is not None else None
    last_latent = torch.full((1, 24, 1, 4, 4), 0.9) if last is not None else None
    identity = IdentityAssets(first, first_latent, last, last_latent, _tensor_fingerprint(first))
    model = SimpleNamespace(
        model=SimpleNamespace(diffusion_model=SimpleNamespace()),
        model_options={}, wrappers={},
        model_dtype=lambda: torch.bfloat16,
        model_size=lambda: 0,
    )
    samples = []

    def empty_latent(width, height, frames):
        return {"samples": _Nested((
            torch.zeros((1, 24, video_latent_t(frames), height // 16, width // 16)),
            torch.zeros((1, 32, 2, audio_latent_t(frames))),
        ))}

    def sample_chunk(**kwargs):
        metadata = kwargs["conditioning"][0][1]
        refs = metadata.get("minimax_refs") or []
        samples.append({
            "prompt": metadata["prompt"],
            "refs": [float(item["latent"].mean()) for item in refs if item.get("kind") == "image"],
            "physical_group": len(samples) + 1,
        })
        video, audio = kwargs["latent"]["samples"].unbind()
        return {"samples": _Nested((video.clone(), audio.clone()))}

    monkeypatch.setattr(sequence, "check_comfy_h3_runtime", lambda: [])
    monkeypatch.setattr(sequence, "prepare_identity_assets", lambda *a, **k: identity)
    monkeypatch.setattr(sequence, "encode_identity_latents", lambda *a, **k: identity)
    monkeypatch.setattr(sequence, "empty_h3_latent", empty_latent)
    monkeypatch.setattr(sequence, "accelerator_summary", lambda _model: "accelerators")
    monkeypatch.setattr(sequence, "clone_model_for_chunk", lambda model, **_kwargs: model)
    monkeypatch.setattr(sequence, "latent_from_cpu", lambda video, audio: {"samples": _Nested((video, audio))})
    monkeypatch.setattr(sequence, "sample_chunk", sample_chunk)
    return SimpleNamespace(model=model, clip=_Clip(), vae=_VAE(), samples=samples)


def _router(*, slots, selected_by_slot, chunks=3, terminal=False):
    images = [None] * 9
    for slot_id, image in slots.items():
        images[int(slot_id[1:]) - 1] = image
    selectors = {slot_id: "off" for slot_id in REFERENCE_SLOT_IDS}
    selectors.update(selected_by_slot)
    schedule = compile_reference_routing_schedule(
        total_chunks=chunks, terminal_merge_enabled=terminal,
        mode="Custom", selectors_by_slot=selectors,
    )
    inputs = ReferenceInputSet(tuple(images), 64, 64, "Match Output")
    return ReferenceRoutingRuntime(schedule=schedule, inputs=inputs)


def _run(runtime, router, *, chunks=3, limit=None, first=None, last=None,
         session=None, script="running", diagnostics=DIAGNOSTICS_BASIC):
    prompt_plan = make_prompt_plan(
        mode=PROMPT_MODE_FIXED, script=script, chunks=chunks,
        chunk_seconds=5.0,
    )
    return sequence.run_sequence_with_reference_routing(
        reference_routing_runtime=router,
        model=runtime.model, clip=runtime.clip, video_vae=runtime.vae,
        audio_vae=None, sampler=object(), sigmas=torch.tensor([1.0, 0.0]),
        first_frame=first, last_frame=last, prompt_plan=prompt_plan,
        width=64, height=64, continuity=V2_CONTINUITY_OPTIONS[1],
        base_seed=42, audio_continuity=True, exact_total_duration=False,
        diagnostics_mode=diagnostics, reroll_from_chunk=0,
        reroll_nonce=0, strict_compatibility=False, debug=False,
        session=session, latent_only=True, max_new_physical_groups=limit,
    )


def test_review_limited_sequence_prepares_only_groups_to_generate(monkeypatch):
    runtime = _runtime(monkeypatch)
    router = _router(
        slots={"R1": _image(0.2), "R4": _image(0.8), "R9": object()},
        selected_by_slot={"R1": "1", "R4": "2", "R9": "3"},
    )
    entries, _last_state, _session, _report = _run(runtime, router, limit=2)
    assert len(entries) == 2
    assert runtime.vae.values == pytest.approx([0.2, 0.8])
    assert all(len(sample["refs"]) == 1 for sample in runtime.samples)
    assert [sample["refs"][0] for sample in runtime.samples] == pytest.approx([0.2, 0.8])
    assert len(runtime.clip.calls) == 2
    assert [call[1]["minimax_ref_items"][0]["data"].mean().item() for call in runtime.clip.calls] == pytest.approx([0.2, 0.8])


def test_custom_all_off_sequence_samples_without_any_reference(monkeypatch):
    runtime = _runtime(monkeypatch)
    router = _router(slots={"R1": object()}, selected_by_slot={}, chunks=1)
    entries, _last_state, _session, _report = _run(runtime, router, chunks=1)
    assert len(entries) == 1 and len(runtime.samples) == 1
    assert runtime.samples[0]["refs"] == []
    assert runtime.vae.values == []
    assert runtime.clip.calls[0][1] == {"images": []}


def test_terminal_physical_pair_uses_one_shared_reference_set(monkeypatch):
    first, last = _image(0.1), _image(0.9)
    runtime = _runtime(monkeypatch, first=first, last=last)
    router = _router(
        slots={"R1": _image(0.2), "R4": _image(0.8)},
        selected_by_slot={"R1": "1", "R4": "2-3"}, terminal=True,
    )
    entries, _last_state, _session, _report = _run(
        runtime, router, first=first, last=last,
    )
    assert len(entries) == 3
    assert len(runtime.samples) == 2
    assert runtime.vae.values == pytest.approx([0.2, 0.8])
    assert all(len(sample["refs"]) == 1 for sample in runtime.samples)
    assert [sample["refs"][0] for sample in runtime.samples] == pytest.approx([0.2, 0.8])
    assert [len(call[1]["minimax_ref_items"]) for call in runtime.clip.calls] == [3, 3]


def test_routed_session_without_rr_r5_evidence_is_ignored(monkeypatch):
    runtime = _runtime(monkeypatch)
    router = _router(slots={}, selected_by_slot={}, chunks=1)
    _, _, session, _ = _run(runtime, router, chunks=1)
    legacy_session = copy.deepcopy(session)
    legacy_session["settings"].pop("reference_routing_v1")
    runtime.samples.clear()
    entries, _, _, report = _run(runtime, router, chunks=1, session=legacy_session)
    assert len(entries) == 1 and len(runtime.samples) == 1
    assert "saved routed Session was ignored" in report
    assert sequence._REFERENCE_ROUTING_RUNTIME.get() is None


def test_inactive_reference_tag_is_warning_only_even_when_diagnostics_off(monkeypatch):
    runtime = _runtime(monkeypatch)
    router = _router(slots={}, selected_by_slot={"R7": "1"}, chunks=1)
    entries, _state, _session, report = _run(
        runtime, router, chunks=1, script="@R7 runs",
        diagnostics=DIAGNOSTICS_OFF,
    )
    assert len(entries) == 1
    assert runtime.samples[0]["prompt"] == "@R7 runs"
    assert "RR-R4 Warning: group 1" in report
    assert "R7 is not active" in report
