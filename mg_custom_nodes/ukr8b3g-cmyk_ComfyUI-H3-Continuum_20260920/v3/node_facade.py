"""Thin V3.8 public-node to runtime boundary.

This module deliberately owns no continuation, review, storage, planning, or
sampling decision.  It only carries already-resolved public-node values into
the existing hardening/runtime call and preserves its public return value.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..reference import resolve_reference_image_inputs


@dataclass(frozen=True)
class RuntimeRequest:
    """Resolved V3.8 node inputs for one existing runtime invocation.

    Tensor and model objects deliberately keep their identity.  Only the two
    keyword containers are copied and frozen so the facade cannot mutate the
    caller's widget/input mapping after the request boundary.
    """

    runtime_kwargs: Mapping[str, Any]
    resolution: Any
    size_source: str
    diagnostics_mode: str
    diagnostics_inputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "runtime_kwargs",
            MappingProxyType(dict(self.runtime_kwargs)),
        )
        object.__setattr__(
            self,
            "diagnostics_inputs",
            MappingProxyType(dict(self.diagnostics_inputs)),
        )


@dataclass(frozen=True)
class RuntimeOutput:
    """Existing runtime result with status-only facade decoration helpers."""

    value: Any

    @property
    def has_status(self) -> bool:
        return isinstance(self.value, tuple) and len(self.value) >= 4

    @property
    def status(self) -> str:
        if not self.has_status:
            raise TypeError("runtime output has no public status field")
        return str(self.value[3])

    def with_status(self, status: str) -> "RuntimeOutput":
        if not self.has_status:
            return self
        return RuntimeOutput((*self.value[:3], str(status), *self.value[4:]))

    def as_public_tuple(self) -> Any:
        """Return the unchanged nonstandard test double or public tuple."""

        return self.value


def make_runtime_request(
    *,
    runtime_kwargs: Mapping[str, Any],
    fixed_runtime_kwargs: Mapping[str, Any],
    resolution: Any,
    size_source: str,
    diagnostics_mode: str,
    diagnostics_inputs: Mapping[str, Any],
) -> RuntimeRequest:
    """Build the one R6 request without silently accepting duplicate inputs."""

    duplicate = set(runtime_kwargs).intersection(fixed_runtime_kwargs)
    if duplicate:
        name = sorted(duplicate)[0]
        raise TypeError(
            "H3 Continuum Sampler V3.8 received multiple values for "
            f"runtime keyword {name!r}"
        )
    merged = dict(runtime_kwargs)
    merged.update(fixed_runtime_kwargs)
    return RuntimeRequest(
        runtime_kwargs=merged,
        resolution=resolution,
        size_source=str(size_source),
        diagnostics_mode=str(diagnostics_mode),
        diagnostics_inputs=diagnostics_inputs,
    )


def execute_v38_runtime_request(
    request: RuntimeRequest,
    *,
    runtime_adapter: Callable[..., Any],
    legacy_size_source: str,
    diagnostics_off: str,
    diagnostics_full: str,
    reference_video_size_default: str,
    build_diagnostics: Callable[..., Any],
    append_status: Callable[..., str],
) -> Any:
    """Call the existing runtime once and add V3.8 display-only diagnostics."""

    output = RuntimeOutput(runtime_adapter(**request.runtime_kwargs))
    if not output.has_status:
        return output.as_public_tuple()

    if request.size_source != legacy_size_source:
        resolution = request.resolution
        lines = [
            (
                f"Resolution: {resolution.width} x {resolution.height} "
                f"({resolution.actual_mp:.2f} MP); source={resolution.aspect_source}."
            )
        ]
        lines.extend(resolution.warnings)
        output = output.with_status(output.status.rstrip() + "\n" + "\n".join(lines))

    if request.diagnostics_mode == diagnostics_off:
        return output.as_public_tuple()

    try:
        inputs = request.diagnostics_inputs
        diagnostics = build_diagnostics(
            video_latents=output.value[0],
            audio_latents=output.value[1],
            assembly_plan=output.value[2],
            output_width=request.resolution.width,
            output_height=request.resolution.height,
            chunk_seconds=float(inputs["chunk_seconds"]),
            reference_images=resolve_reference_image_inputs(
                inputs.get("reference_image_1"),
                inputs.get("reference_image_2"),
                inputs.get("reference_image_3"),
                inputs.get("reference_image_4"),
                inputs.get("reference_image_5"),
                inputs.get("image_references"),
            ),
            reference_size=str(inputs.get("reference_size", "Match Output")),
            video_guide=inputs.get("reference_video_1"),
            video_guide_size=str(
                inputs.get("video_reference_size", reference_video_size_default)
            ),
            still_guide_active=inputs.get("guide") is not None,
        )
        status = append_status(
            output.status,
            diagnostics,
            mode=request.diagnostics_mode,
        )
    except Exception as exc:
        message = (
            "V3.8 Reliability\n"
            "Reliability diagnostics unavailable; generation result was preserved."
        )
        if request.diagnostics_mode == diagnostics_full:
            message += f" ({type(exc).__name__}: {exc})"
        status = output.status.rstrip() + "\n" + message
    return output.with_status(status).as_public_tuple()
