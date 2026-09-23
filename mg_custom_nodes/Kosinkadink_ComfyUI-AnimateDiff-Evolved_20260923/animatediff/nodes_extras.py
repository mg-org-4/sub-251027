from typing import Union

import torch
from torch import Tensor

from comfy_api.latest import io

import folder_paths
import nodes as comfy_nodes
from comfy.model_patcher import ModelPatcher
import comfy.model_patcher
import comfy.samplers
from comfy.sd import load_checkpoint_guess_config

from .logger import logger
from .utils_model import BetaSchedules
from .utils_motion import extend_to_batch_size, prepare_mask_batch
from .model_injection import get_vanilla_model_patcher
from .cfg_extras import perturbed_attention_guidance_patch, rescale_cfg_patch


class AnimateDiffUnload(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_AnimateDiffUnload',
            display_name='AnimateDiff Unload 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/extras',
            inputs=[io.Model.Input('model')],
            outputs=[io.Model.Output('MODEL')]
        )
    @classmethod
    def execute(cls, model: ModelPatcher):
        # return model clone with ejected params
        #model = eject_params_from_model(model)
        model = get_vanilla_model_patcher(model)
        return io.NodeOutput(model.clone())


class CheckpointLoaderSimpleWithNoiseSelect(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='CheckpointLoaderSimpleWithNoiseSelect',
            display_name='Load Checkpoint w/ Noise Select 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/extras',
            inputs=[io.Combo.Input('ckpt_name', options=folder_paths.get_filename_list("checkpoints")), io.Combo.Input('beta_schedule', options=['autoselect', 'use existing', 'sqrt_linear (AnimateDiff)', 'linear (AnimateDiff-SDXL)', 'linear (HotshotXL/default)', 'avg(sqrt_linear,linear)', 'lcm avg(sqrt_linear,linear)', 'lcm', 'lcm[100_ots]', 'lcm >> sqrt_linear', 'sqrt', 'cosine', 'squaredcos_cap_v2'] , default='use existing'), io.Boolean.Input('use_custom_scale_factor', default=False, optional=True), io.Float.Input('scale_factor', default=0.18215, max=1.0, min=0.0, step=1e-05, optional=True)],
            outputs=[io.Model.Output('MODEL'), io.Clip.Output('CLIP'), io.Vae.Output('VAE')]
        )
    @classmethod
    def execute(cls, ckpt_name, beta_schedule, output_vae=True, output_clip=True, use_custom_scale_factor=False, scale_factor=0.18215):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        # register chosen beta schedule on model - convert to beta_schedule name recognized by ComfyUI
        new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, out[0])
        if new_model_sampling is not None:
            out[0].model.model_sampling = new_model_sampling
        if use_custom_scale_factor:
            out[0].model.latent_format.scale_factor = scale_factor
        return io.NodeOutput(*out)


class EmptyLatentImageLarge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_EmptyLatentImageLarge',
            display_name='Empty Latent Image (Big Batch) 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/extras',
            inputs=[io.Int.Input('width', default=512, max=16384, min=64, step=8), io.Int.Input('height', default=512, max=16384, min=64, step=8), io.Int.Input('batch_size', default=1, max=262144, min=1)],
            outputs=[io.Latent.Output('LATENT')]
        )
    @classmethod
    def execute(cls, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return io.NodeOutput({"samples":latent})


class PerturbedAttentionGuidanceMultival(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_PerturbedAttentionGuidanceMultival',
            display_name='PerturbedAttnGuide [Multival] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/extras',
            inputs=[io.Model.Input('model'), io.Custom('MULTIVAL').Input('scale_multival')],
            outputs=[io.Model.Output('MODEL')]
        )
    @classmethod
    def execute(cls, model: ModelPatcher, scale_multival: Union[float, Tensor]):
        m = model.clone()
        m.set_model_sampler_post_cfg_function(perturbed_attention_guidance_patch(scale_multival))

        return io.NodeOutput(m)


class RescaleCFGMultival(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_RescaleCFGMultival',
            display_name='RescaleCFG [Multival] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/extras',
            inputs=[io.Model.Input('model'), io.Custom('MULTIVAL').Input('mult_multival')],
            outputs=[io.Model.Output('MODEL')]
        )
    @classmethod
    def execute(cls, model: ModelPatcher, mult_multival: Union[float, Tensor]):
        m = model.clone()
        m.set_model_sampler_cfg_function(rescale_cfg_patch(mult_multival))
        return io.NodeOutput(m)
