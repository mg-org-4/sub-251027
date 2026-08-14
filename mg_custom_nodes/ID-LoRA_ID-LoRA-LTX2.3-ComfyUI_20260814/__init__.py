"""ComfyUI custom node package for ID-LoRA-2.3 inference (one-stage and two-stage)."""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .nodes_model_loader import IDLoraModelLoader, IDLoraTwoStageModelLoader
from .nodes_prompt_encoder import IDLoraPromptEncoder
from .nodes_sampler import IDLoraOneStageSampler, IDLoraTwoStageSampler


class IDLoraExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            IDLoraModelLoader,
            IDLoraTwoStageModelLoader,
            IDLoraPromptEncoder,
            IDLoraOneStageSampler,
            IDLoraTwoStageSampler,
        ]


async def comfy_entrypoint() -> IDLoraExtension:
    return IDLoraExtension()
