"""IDLoraOneStageSampler and IDLoraTwoStageSampler nodes — core generation: denoise + decode video + decode audio."""

from __future__ import annotations

from fractions import Fraction

import torch
from comfy_api.latest import io, Input, InputImpl, Types

from .pipeline_wrapper import (
    IDLoraOneStagePipeline,
    IDLoraTwoStagePipeline,
    IDLoraConditioningType,
    compute_resolution_match_aspect,
)


class IDLoraOneStageSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IDLoraOneStageSampler",
            display_name="ID-LoRA One-Stage Sampler",
            category="ID-LoRA",
            description="Generate audio+video with speaker identity transfer using the ID-LoRA one-stage pipeline.",
            inputs=[
                io.Custom("ID_LORA_PIPELINE").Input("pipeline", tooltip="Loaded ID-LoRA pipeline."),
                io.Custom("ID_LORA_CONDITIONING").Input("conditioning", tooltip="Encoded prompt conditioning."),
                io.Int.Input("seed", default=42, min=0, max=2**31 - 1),
                io.Int.Input("height", default=512, min=64, max=2048, step=32),
                io.Int.Input("width", default=512, min=64, max=2048, step=32),
                io.Int.Input("num_frames", default=121, min=1, max=1000, step=1),
                io.Int.Input("num_inference_steps", default=30, min=1, max=200, step=1),
                io.Float.Input("frame_rate", default=25.0, min=1.0, max=120.0, step=0.1),
                io.Float.Input("video_guidance_scale", default=3.0, min=0.0, max=30.0, step=0.1),
                io.Float.Input("audio_guidance_scale", default=7.0, min=0.0, max=30.0, step=0.1),
                io.Boolean.Input("auto_resolution", default=True,
                                 tooltip="Auto-detect resolution from first-frame aspect ratio."),
                io.Int.Input("max_resolution", default=512, min=64, max=2048, step=32,
                             tooltip="Maximum long-side resolution for auto-resolution (ignored when auto_resolution is off)."),
                io.Image.Input("first_frame", optional=True, tooltip="Optional first-frame image for face conditioning."),
                io.Audio.Input("reference_audio", optional=True, tooltip="Optional reference audio for speaker identity transfer."),
            ],
            outputs=[
                io.Video.Output(display_name="Video", tooltip="Generated video with audio."),
            ],
        )

    @classmethod
    def execute(
        cls,
        pipeline: IDLoraOneStagePipeline,
        conditioning: dict,
        first_frame: Input.Image | None,
        reference_audio: Input.Audio | None,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        frame_rate: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        auto_resolution: bool,
        max_resolution: int,
    ) -> io.NodeOutput:
        # Load heavy models now (after prompt encoding freed the text encoder)
        if pipeline._transformer is None:
            pipeline.load_models()

        # Convert ComfyUI IMAGE [B,H,W,C] -> pipeline [C,H,W]
        condition_image = None
        if first_frame is not None:
            condition_image = first_frame[0].permute(2, 0, 1)  # [H,W,C] -> [C,H,W]

            if auto_resolution:
                src_h, src_w = first_frame.shape[1], first_frame.shape[2]
                height, width = compute_resolution_match_aspect(src_h, src_w, max_long=max_resolution)
                print(f"[ID-LoRA] Auto-resolution: {src_w}x{src_h} -> {width}x{height}")

        # Convert ComfyUI AUDIO {"waveform":[B,C,S],"sample_rate":int} -> pipeline [C,S]
        ref_audio = None
        ref_sr = 16000
        if reference_audio is not None:
            ref_audio = reference_audio["waveform"][0]  # [C, S]
            ref_sr = reference_audio["sample_rate"]

        v_context_p = conditioning["v_context_p"]
        a_context_p = conditioning["a_context_p"]
        v_context_n = conditioning["v_context_n"]
        a_context_n = conditioning["a_context_n"]

        video_tensor, audio_output = pipeline(
            v_context_p=v_context_p,
            a_context_p=a_context_p,
            v_context_n=v_context_n,
            a_context_n=a_context_n,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guidance_scale=video_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            reference_audio=ref_audio,
            reference_audio_sample_rate=ref_sr,
            condition_image=condition_image,
        )

        # Convert pipeline video [C,F,H,W] -> ComfyUI images [F,H,W,C]
        images = video_tensor.permute(1, 2, 3, 0)  # [F,H,W,C]

        # Convert pipeline audio [C,S] -> ComfyUI audio dict
        vocoder_sr = getattr(pipeline, "_vocoder_sr", 24000)
        audio_dict = {
            "waveform": audio_output.unsqueeze(0),  # [1,C,S]
            "sample_rate": vocoder_sr,
        }

        # Bundle into ComfyUI Video type
        video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=images,
                audio=audio_dict,
                frame_rate=Fraction(frame_rate),
            )
        )

        return io.NodeOutput(video)


class IDLoraTwoStageSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IDLoraTwoStageSampler",
            display_name="ID-LoRA Two-Stage Sampler",
            category="ID-LoRA",
            description=(
                "Generate audio+video with speaker identity transfer using the ID-LoRA two-stage pipeline. "
                "Stage 1 generates at the specified resolution, stage 2 refines at 2x resolution."
            ),
            inputs=[
                io.Custom("ID_LORA_PIPELINE").Input("pipeline", tooltip="Loaded ID-LoRA two-stage pipeline."),
                io.Custom("ID_LORA_CONDITIONING").Input("conditioning", tooltip="Encoded prompt conditioning."),
                io.Int.Input("seed", default=42, min=0, max=2**31 - 1),
                io.Int.Input("height", default=512, min=64, max=2048, step=32,
                             tooltip="Stage 1 height. Output will be 2x this value."),
                io.Int.Input("width", default=512, min=64, max=2048, step=32,
                             tooltip="Stage 1 width. Output will be 2x this value."),
                io.Int.Input("num_frames", default=121, min=1, max=1000, step=1),
                io.Int.Input("num_inference_steps", default=30, min=1, max=200, step=1,
                             tooltip="Number of denoising steps for stage 1. Stage 2 uses 3 fixed steps."),
                io.Float.Input("frame_rate", default=25.0, min=1.0, max=120.0, step=0.1),
                io.Float.Input("video_guidance_scale", default=3.0, min=0.0, max=30.0, step=0.1),
                io.Float.Input("audio_guidance_scale", default=7.0, min=0.0, max=30.0, step=0.1),
                io.Boolean.Input("auto_resolution", default=True,
                                 tooltip="Auto-detect resolution from first-frame aspect ratio."),
                io.Int.Input("max_resolution", default=512, min=64, max=2048, step=32,
                             tooltip="Maximum long-side resolution for auto-resolution (ignored when auto_resolution is off)."),
                io.Boolean.Input("hq_mode", default=True,
                                 tooltip="Use res2s second-order sampler for higher quality (slower)."),
                io.Image.Input("first_frame", optional=True, tooltip="Optional first-frame image for face conditioning."),
                io.Audio.Input("reference_audio", optional=True, tooltip="Optional reference audio for speaker identity transfer."),
            ],
            outputs=[
                io.Video.Output(display_name="Video", tooltip="Generated video with audio (2x input resolution)."),
            ],
        )

    @classmethod
    def execute(
        cls,
        pipeline: IDLoraTwoStagePipeline,
        conditioning: dict,
        first_frame: Input.Image | None,
        reference_audio: Input.Audio | None,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        frame_rate: float,
        video_guidance_scale: float,
        audio_guidance_scale: float,
        auto_resolution: bool,
        max_resolution: int,
        hq_mode: bool,
    ) -> io.NodeOutput:
        # Load stage-1 models now (after prompt encoding freed the text encoder)
        if getattr(pipeline, "_stage_1_transformer", None) is None:
            pipeline.load_stage_1_models()

        # Convert ComfyUI IMAGE [B,H,W,C] -> pipeline [C,H,W]
        condition_image = None
        if first_frame is not None:
            condition_image = first_frame[0].permute(2, 0, 1)  # [H,W,C] -> [C,H,W]

            if auto_resolution:
                src_h, src_w = first_frame.shape[1], first_frame.shape[2]
                height, width = compute_resolution_match_aspect(src_h, src_w, max_long=max_resolution)
                print(f"[ID-LoRA] Auto-resolution: {src_w}x{src_h} -> {width}x{height} (stage 1), output {width*2}x{height*2}")

        # Convert ComfyUI AUDIO {"waveform":[B,C,S],"sample_rate":int} -> pipeline [C,S]
        ref_audio = None
        ref_sr = 16000
        if reference_audio is not None:
            ref_audio = reference_audio["waveform"][0]  # [C, S]
            ref_sr = reference_audio["sample_rate"]

        v_context_p = conditioning["v_context_p"]
        a_context_p = conditioning["a_context_p"]
        v_context_n = conditioning["v_context_n"]
        a_context_n = conditioning["a_context_n"]

        video_tensor, audio_output = pipeline(
            v_context_p=v_context_p,
            a_context_p=a_context_p,
            v_context_n=v_context_n,
            a_context_n=a_context_n,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guidance_scale=video_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            hq_mode=hq_mode,
            reference_audio=ref_audio,
            reference_audio_sample_rate=ref_sr,
            condition_image=condition_image,
        )

        # Convert pipeline video [C,F,H,W] -> ComfyUI images [F,H,W,C]
        images = video_tensor.permute(1, 2, 3, 0)  # [F,H,W,C]

        # Convert pipeline audio [C,S] -> ComfyUI audio dict
        vocoder_sr = getattr(pipeline, "_vocoder_sr", 24000)
        audio_dict = {
            "waveform": audio_output.unsqueeze(0),  # [1,C,S]
            "sample_rate": vocoder_sr,
        }

        # Bundle into ComfyUI Video type
        video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=images,
                audio=audio_dict,
                frame_rate=Fraction(frame_rate),
            )
        )

        return io.NodeOutput(video)
