from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from ComfyUI_H3_Continuum_Join.v3 import driving_nodes
from ComfyUI_H3_Continuum_Join.v3.node_facade import (
    RuntimeOutput,
    execute_v38_runtime_request,
    make_runtime_request,
)
from ComfyUI_H3_Continuum_Join.v3.resolution import (
    H3_SIZE_SOURCE_LEGACY,
    H3_SIZE_SOURCE_MANUAL,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _Resolution:
    width: int = 704
    height: int = 1024
    actual_mp: float = 0.720896
    aspect_source: str = "Manual Width / Height"
    warnings: tuple[str, ...] = ()


def _request(*, mode="Off", size_source=H3_SIZE_SOURCE_LEGACY, inputs=None):
    return make_runtime_request(
        runtime_kwargs={"marker": object()},
        fixed_runtime_kwargs={"width": 704, "height": 1024},
        resolution=_Resolution(),
        size_source=size_source,
        diagnostics_mode=mode,
        diagnostics_inputs={"chunk_seconds": 5.0, **(inputs or {})},
    )


def _append(status, diagnostics, *, mode):
    return f"{status}|{mode}|{diagnostics}"


def test_r6_request_freezes_keyword_containers_without_copying_runtime_objects():
    marker = object()
    request = make_runtime_request(
        runtime_kwargs={"marker": marker},
        fixed_runtime_kwargs={"width": 704},
        resolution=_Resolution(),
        size_source=H3_SIZE_SOURCE_LEGACY,
        diagnostics_mode="Off",
        diagnostics_inputs={"chunk_seconds": 5.0, "reference": marker},
    )

    assert request.runtime_kwargs["marker"] is marker
    assert request.diagnostics_inputs["reference"] is marker
    with pytest.raises(TypeError):
        request.runtime_kwargs["other"] = object()
    with pytest.raises(TypeError):
        request.diagnostics_inputs["other"] = object()


def test_r6_request_rejects_duplicate_runtime_keywords():
    with pytest.raises(TypeError, match="multiple values"):
        make_runtime_request(
            runtime_kwargs={"width": 1},
            fixed_runtime_kwargs={"width": 704},
            resolution=_Resolution(),
            size_source=H3_SIZE_SOURCE_LEGACY,
            diagnostics_mode="Off",
            diagnostics_inputs={},
        )


def test_r6_runtime_output_preserves_nonstandard_values_and_tuple_identity():
    marker = object()
    assert RuntimeOutput(marker).as_public_tuple() is marker

    output = (object(), object(), object(), "status", object())
    request = _request()
    result = execute_v38_runtime_request(
        request,
        runtime_adapter=lambda **kwargs: output,
        legacy_size_source=H3_SIZE_SOURCE_LEGACY,
        diagnostics_off="Off",
        diagnostics_full="Detailed Report",
        reference_video_size_default="Efficient",
        build_diagnostics=lambda **kwargs: pytest.fail("must not run"),
        append_status=_append,
    )
    assert result is output


def test_r6_facade_decorates_status_only_after_existing_runtime_returns():
    video, audio, plan, tail = object(), object(), object(), object()
    output = (video, audio, plan, "base", tail)
    seen = {}
    request = _request(
        mode="Basic",
        size_source=H3_SIZE_SOURCE_MANUAL,
        inputs={
            "reference_image_1": "image-1",
            "reference_size": "Match Output",
            "reference_video_1": "guide",
            "guide": "still",
        },
    )

    def build(**kwargs):
        seen.update(kwargs)
        return "diagnostics"

    result = execute_v38_runtime_request(
        request,
        runtime_adapter=lambda **kwargs: output,
        legacy_size_source=H3_SIZE_SOURCE_LEGACY,
        diagnostics_off="Off",
        diagnostics_full="Detailed Report",
        reference_video_size_default="Efficient",
        build_diagnostics=build,
        append_status=_append,
    )

    assert result[:3] == output[:3]
    assert result[4] is tail
    assert "Resolution: 704 x 1024 (0.72 MP)" in result[3]
    assert result[3].endswith("|Basic|diagnostics")
    assert seen["video_latents"] is video
    assert seen["audio_latents"] is audio
    assert seen["assembly_plan"] is plan
    assert seen["reference_images"] == ("image-1", None, None)
    assert seen["still_guide_active"] is True


def test_r6_diagnostic_failure_keeps_runtime_payloads_and_returns_status_warning():
    payload = (object(), object(), object(), "base", object())
    result = execute_v38_runtime_request(
        _request(mode="Detailed Report"),
        runtime_adapter=lambda **kwargs: payload,
        legacy_size_source=H3_SIZE_SOURCE_LEGACY,
        diagnostics_off="Off",
        diagnostics_full="Detailed Report",
        reference_video_size_default="Efficient",
        build_diagnostics=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad")),
        append_status=_append,
    )

    assert result[:3] == payload[:3]
    assert result[4] is payload[4]
    assert "generation result was preserved" in result[3]
    assert "RuntimeError: bad" in result[3]


def test_r6_v38_node_builds_one_request_then_delegates_to_existing_runtime_adapter(monkeypatch):
    captured = {}

    def facade(request, **kwargs):
        captured["request"] = request
        captured["adapter"] = kwargs["runtime_adapter"]
        return "facade-result"

    monkeypatch.setattr(driving_nodes, "execute_v38_runtime_request", facade)
    result = driving_nodes.H3ContinuumSamplerV38().run(
        chunk_seconds=5.0,
        size_source=H3_SIZE_SOURCE_MANUAL,
        width=704,
        height=1024,
        project_id="stable-project",
    )

    request = captured["request"]
    assert result == "facade-result"
    assert request.runtime_kwargs["width"] == 704
    assert request.runtime_kwargs["height"] == 1024
    assert request.runtime_kwargs["project_id"] == "stable-project"
    assert request.runtime_kwargs["reference_encode_cache"] is True
    assert "size_source" not in request.runtime_kwargs
    assert callable(captured["adapter"])


def test_r6_facade_module_has_no_runtime_owner_imports():
    tree = ast.parse((ROOT / "v3" / "node_facade.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {
        "torch",
        "run_storage",
        "runtime_coordinator",
        "execution_planner",
        "sampling_engine",
    }
    assert not any(
        name == forbidden_name or name.endswith(f".{forbidden_name}")
        for name in imported
        for forbidden_name in forbidden
    )
