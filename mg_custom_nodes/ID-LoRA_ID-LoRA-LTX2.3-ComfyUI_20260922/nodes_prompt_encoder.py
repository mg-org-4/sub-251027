"""IDLoraPromptEncoder node — encode text prompts into video/audio conditioning."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ltx_pipelines.utils import encode_prompts, cleanup_memory
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT

from .pipeline_wrapper import IDLoraOneStagePipeline, IDLoraConditioningType


class IDLoraPromptEncoder(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IDLoraPromptEncoder",
            display_name="ID-LoRA Prompt Encoder",
            category="ID-LoRA",
            description="Encode positive and negative prompts into conditioning tensors for ID-LoRA generation.",
            inputs=[
                io.Custom("ID_LORA_PIPELINE").Input("pipeline", tooltip="Loaded ID-LoRA pipeline."),
                io.String.Input("prompt", multiline=True, default="",
                                tooltip="Positive text prompt describing the desired generation."),
                io.String.Input("negative_prompt", multiline=True,
                                default=DEFAULT_NEGATIVE_PROMPT,
                                tooltip="Negative text prompt."),
            ],
            outputs=[
                io.Custom("ID_LORA_CONDITIONING").Output(
                    display_name="Conditioning",
                    tooltip="Encoded video/audio conditioning tensors.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        pipeline: IDLoraOneStagePipeline,
        prompt: str,
        negative_prompt: str,
    ) -> io.NodeOutput:
        device = pipeline.device

        results = encode_prompts(
            prompts=[prompt, negative_prompt],
            model_ledger=pipeline.model_ledger,
        )
        ctx_p, ctx_n = results

        conditioning = {
            "v_context_p": ctx_p.video_encoding.to(device),
            "a_context_p": ctx_p.audio_encoding.to(device),
            "v_context_n": ctx_n.video_encoding.to(device),
            "a_context_n": ctx_n.audio_encoding.to(device),
        }

        cleanup_memory()
        return io.NodeOutput(conditioning)
