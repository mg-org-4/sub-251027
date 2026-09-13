from comfy_api.latest import io
from typing import Union
import torch

from nodes import VAEEncode
import comfy.utils
from comfy.sd import VAE

from .ad_settings import AnimateDiffSettings
from .logger import logger
from .utils_model import ScaleMethods, CropMethods, get_available_motion_models, vae_encode_raw_batched
from .utils_motion import ADKeyframeGroup
from .motion_lora import MotionLoraList
from .model_injection import (MotionModelGroup, MotionModelPatcher, get_mm_attachment, create_fresh_encoder_only_model,
                              load_motion_module_gen2, inject_img_encoder_into_model)
from .motion_module_ad import AnimateDiffFormat
from .nodes_gen2 import ApplyAnimateDiffModelNode


class ApplyAnimateLCMI2VModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_ApplyAnimateLCMI2VModel',
            display_name='Apply AnimateLCM-I2V Model 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/AnimateLCM-I2V',
            inputs=[
                io.Custom("MOTION_MODEL_ADE").Input('motion_model'),
                io.Latent.Input('ref_latent'),
                io.Float.Input('ref_drift', default=0.0, max=10.0, min=0.0, step=0.001),
                io.Boolean.Input('apply_ref_when_disabled', default=False),
                io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001),
                io.Float.Input('end_percent', default=1.0, max=1.0, min=0.0, step=0.001),
                io.Custom("MOTION_LORA").Input('motion_lora', optional=True),
                io.Custom("MULTIVAL").Input('scale_multival', optional=True),
                io.Custom("MULTIVAL").Input('effect_multival', optional=True),
                io.Custom("AD_KEYFRAMES").Input('ad_keyframes', optional=True),
                io.Custom("M_MODELS").Input('prev_m_models', optional=True),
                io.Custom("PER_BLOCK").Input('per_block', optional=True),
            ],
            outputs=[
                io.Custom("M_MODELS").Output('M_MODELS'),
            ],
        )
    

    @classmethod
    def execute(cls, motion_model: MotionModelPatcher, ref_latent: dict, ref_drift: float=0.0, apply_ref_when_disabled=False, start_percent: float=0.0, end_percent: float=1.0,
                           motion_lora: MotionLoraList=None, ad_keyframes: ADKeyframeGroup=None,
                           scale_multival=None, effect_multival=None, per_block=None,
                           prev_m_models: MotionModelGroup=None,):
        new_m_models = ApplyAnimateDiffModelNode.execute( motion_model, start_percent=start_percent, end_percent=end_percent,
                                                                    motion_lora=motion_lora, ad_keyframes=ad_keyframes,
                                                                    scale_multival=scale_multival, effect_multival=effect_multival, per_block=per_block,
                                                                    prev_m_models=prev_m_models).args
        # most recent added model will always be first in list;
        curr_model = new_m_models[0].models[0]
        # confirm that model contains img_encoder
        if curr_model.model.img_encoder is None:
            raise Exception(f"Motion model '{curr_model.model.mm_info.mm_name}' does not contain an img_encoder; cannot be used with Apply AnimateLCM-I2V Model node.")
        attachment = get_mm_attachment(curr_model)
        attachment.orig_img_latents = ref_latent["samples"]
        attachment.orig_ref_drift = ref_drift
        attachment.orig_apply_ref_when_disabled = apply_ref_when_disabled
        return io.NodeOutput(*new_m_models)


class LoadAnimateLCMI2VModelNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_LoadAnimateLCMI2VModel',
            display_name='Load AnimateLCM-I2V Model 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/AnimateLCM-I2V',
            inputs=[
                io.Combo.Input('model_name', options=get_available_motion_models()),
                io.Custom("AD_SETTINGS").Input('ad_settings', optional=True),
            ],
            outputs=[
                io.Custom("MOTION_MODEL_ADE").Output('MOTION_MODEL'),
                io.Custom("MOTION_MODEL_ADE").Output('encoder_only'),
            ],
        )
    

    @classmethod
    def execute(cls, model_name: str, ad_settings: AnimateDiffSettings=None):
        # load motion module and motion settings, if included
        motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=ad_settings)
        # make sure model is an AnimateLCM-I2V model
        if motion_model.model.mm_info.mm_format != AnimateDiffFormat.ANIMATELCM:
            raise Exception(f"Motion model '{motion_model.model.mm_info.mm_name}' is not an AnimateLCM-I2V model; selected model is not AnimateLCM, and does not contain an img_encoder.")
        if motion_model.model.img_encoder is None:
            raise Exception(f"Motion model '{motion_model.model.mm_info.mm_name}' is not an AnimateLCM-I2V model; selected model IS AnimateLCM, but does NOT contain an img_encoder.")
        # create encoder-only motion model
        encoder_only_motion_model = create_fresh_encoder_only_model(motion_model=motion_model)
        return io.NodeOutput(motion_model, encoder_only_motion_model)


class LoadAnimateDiffAndInjectI2VNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_InjectI2VIntoAnimateDiffModel',
            display_name='🧪Inject I2V into AnimateDiff Model 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/AnimateLCM-I2V/🧪experimental',
            inputs=[
                io.Combo.Input('model_name', options=get_available_motion_models()),
                io.Custom("MOTION_MODEL_ADE").Input('motion_model'),
                io.Custom("AD_SETTINGS").Input('ad_settings', optional=True),
            ],
            outputs=[
                io.Custom("MOTION_MODEL_ADE").Output('MOTION_MODEL'),
            ],
            is_experimental=True,
        )
    


    @classmethod
    def execute(cls, model_name: str, motion_model: MotionModelPatcher, ad_settings: AnimateDiffSettings=None):
        # make sure model w/ encoder actually has encoder
        if motion_model.model.img_encoder is None:
            raise Exception("Passed-in motion model was expected to have an img_encoder, but did not.")
        # load motion module and motion settings, if included
        loaded_motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=ad_settings)
        inject_img_encoder_into_model(motion_model=loaded_motion_model, w_encoder=motion_model)
        return io.NodeOutput(loaded_motion_model,)


class UpscaleAndVaeEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_UpscaleAndVAEEncode',
            display_name='Scale Ref Image and VAE Encode 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/AnimateLCM-I2V',
            inputs=[
                io.Image.Input('image'),
                io.Vae.Input('vae'),
                io.Latent.Input('latent_size'),
                io.Combo.Input('scale_method', options=['nearest-exact', 'bilinear', 'area', 'bicubic', 'lanczos']),
                io.Combo.Input('crop', options=['disabled', 'center'], default='center'),
            ],
            outputs=[
                io.Latent.Output('LATENT'),
            ],
        )
    


    @classmethod
    def execute(cls, image: torch.Tensor, vae: VAE, latent_size: torch.Tensor, scale_method: str, crop: str):
        b, c, h, w = latent_size["samples"].size()
        image = image.movedim(-1,1)
        image = comfy.utils.common_upscale(samples=image, width=w*8, height=h*8, upscale_method=scale_method, crop=crop)
        image = image.movedim(1,-1)
        # now that images are the expected size, VAEEncode them
        return io.NodeOutput({"samples": vae_encode_raw_batched(vae, image)},)
