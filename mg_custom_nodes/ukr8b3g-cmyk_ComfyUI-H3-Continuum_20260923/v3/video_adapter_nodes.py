"""Core VIDEO to H3 frame batches, with no file/upload/Enable frontend."""
from __future__ import annotations

from fractions import Fraction

from .easy_video_nodes import ExecutionBlocker, _resample_video_frames, io


class H3ContinuumVideoAdapter(io.ComfyNode):
    """Reuse the established frame-rate conversion; audio is passed unchanged."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ContinuumVideoAdapter",
            display_name="H3 Continuum Video Adapter",
            category="MiniMax H3/Continuum/Input",
            description=(
                "Converts Core Load Video's VIDEO to IMAGE frames and AUDIO. "
                "Use Force Rate 24 for H3 Video Guide. No file selector or Enable. "
                "Disable the whole video-input group, or bypass this adapter, "
                "to disconnect its optional Sampler inputs."
            ),
            inputs=[
                io.Video.Input("video"),
                io.Float.Input(
                    "force_rate", default=24.0, min=0.0, max=120.0, step=0.01,
                    display_name="Force Rate",
                    tooltip=(
                        "24 resamples to H3's expected FPS without changing nominal "
                        "duration or audio. 0 keeps source frames (not normally for H3)."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(cls, video, force_rate: float = 24.0):
        components = video.get_components()
        images = _resample_video_frames(
            components.images, Fraction(components.frame_rate), float(force_rate)
        )
        audio = components.audio
        if audio is None:
            audio = ExecutionBlocker(None)
        return io.NodeOutput(images, audio)


NODE_CLASS_MAPPINGS = {"H3ContinuumVideoAdapter": H3ContinuumVideoAdapter}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumVideoAdapter": "H3 Continuum Video Adapter",
}
