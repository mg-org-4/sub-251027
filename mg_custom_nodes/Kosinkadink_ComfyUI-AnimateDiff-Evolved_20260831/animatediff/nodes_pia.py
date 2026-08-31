from comfy_api.latest import io
from typing import Union
import torch
from torch import Tensor
import math

from comfy.sd import VAE

from .ad_settings import AnimateDiffSettings
from .logger import logger
from .utils_model import BIGMIN, BIGMAX, get_available_motion_models
from .utils_motion import ADKeyframeGroup, InputPIA, InputPIA_Multival, extend_list_to_batch_size, extend_to_batch_size, prepare_mask_batch
from .motion_lora import MotionLoraList
from .model_injection import MotionModelGroup, MotionModelPatcher, get_mm_attachment, load_motion_module_gen2, inject_pia_conv_in_into_model
from .motion_module_ad import AnimateDiffFormat
from .nodes_gen2 import ApplyAnimateDiffModelNode, ADKeyframeNode


# Preset values ported over from PIA repository:
# https://github.com/open-mmlab/PIA/blob/main/animatediff/utils/util.py
class PIA_RANGES:
    ANIMATION_SMALL = "Animation (Small Motion)"
    ANIMATION_MEDIUM = "Animation (Medium Motion)"
    ANIMATION_LARGE = "Animation (Large Motion)"
    LOOP_SMALL = "Loop (Small Motion)"
    LOOP_MEDIUM = "Loop (Medium Motion)"
    LOOP_LARGE = "Loop (Large Motion)"
    STYLE_TRANSFER_SMALL = "Style Transfer (Small Motion)"
    STYLE_TRANSFER_MEDIUM = "Style Transfer (Medium Motion)"
    STYLE_TRANSFER_LARGE = "Style Transfer (Large Motion)"

    _LOOPED = [LOOP_SMALL, LOOP_MEDIUM, LOOP_LARGE]
    _LIST_ALL = [ANIMATION_SMALL, ANIMATION_MEDIUM, ANIMATION_LARGE,
                 LOOP_SMALL, LOOP_MEDIUM, LOOP_LARGE,
                 STYLE_TRANSFER_SMALL, STYLE_TRANSFER_MEDIUM, STYLE_TRANSFER_LARGE]

    _MAPPING = {
        ANIMATION_SMALL: [1.0, 0.9, 0.85, 0.85, 0.85, 0.8],
        ANIMATION_MEDIUM: [1.0, 0.8, 0.8, 0.8, 0.79, 0.78, 0.75],
        ANIMATION_LARGE: [1.0, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5, 0.5],
        LOOP_SMALL: [1.0, 0.9, 0.85, 0.85, 0.85, 0.8],
        LOOP_MEDIUM: [1.0, 0.8, 0.8, 0.8, 0.79, 0.78, 0.75],
        LOOP_LARGE: [1.0, 0.8, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5],
        STYLE_TRANSFER_SMALL: [0.5, 0.4, 0.4, 0.4, 0.35, 0.3],
        STYLE_TRANSFER_MEDIUM: [0.5, 0.4, 0.4, 0.4, 0.35, 0.35, 0.3, 0.25, 0.2],
        STYLE_TRANSFER_LARGE: [0.5, 0.2],
    }

    @classmethod
    def get_preset(cls, preset: str) -> list[float]:
        if preset in cls._MAPPING:
            return cls._MAPPING[preset]
        raise Exception(f"PIA Preset '{preset}' is not recognized.")
    
    @classmethod
    def is_looped(cls, preset: str) -> bool:
        return preset in cls._LOOPED


class InputPIA_PaperPresets(InputPIA):
    def __init__(self, preset: str, index: int, mult_multival: Union[float, Tensor]=None, effect_multival: Union[float, Tensor]=None):
        super().__init__(effect_multival=effect_multival)
        self.preset = preset
        self.index = index
        self.mult_multival = mult_multival if mult_multival is not None else 1.0
    
    def get_mask(self, x: Tensor):
        b, c, h, w = x.shape
        values = PIA_RANGES.get_preset(self.preset)
        # if preset is looped, make values loop
        if PIA_RANGES.is_looped(self.preset):
            # even length
            if b % 2 == 0:
                # extend to half length to get half of the loop
                values = extend_list_to_batch_size(values, b // 2)
                # apply second half of loop (just reverse it)
                values += list(reversed(values))
            # odd length
            else:
                inter_values = extend_list_to_batch_size(values, b // 2)
                middle_vals = [values[min(len(inter_values), len(values)-1)]]
                # make middle vals long enough to fill in gaps (or none if not needed)
                middle_vals = middle_vals * (max(0, b-2*len(inter_values)))
                values = inter_values + middle_vals + list(reversed(inter_values))
        # otherwise, just extend values to desired length
        else:
            values = extend_list_to_batch_size(values, b)
        assert len(values) == b

        index = self.index
        # handle negative index
        if index < 0:
            index = b + index
        # constrain index between 0 and b-1
        index = max(0, min(b-1, index))
        # center values around targer index
        order = [abs(i - index) for i in range(b)]
        real_values = [values[order[i]] for i in range(b)]
        # using real values, generate masks
        tensor_values = torch.tensor(real_values).unsqueeze(-1).unsqueeze(-1)
        mask = torch.ones(size=(b, h, w)) * tensor_values
        # apply multi_multival to mask
        if type(self.mult_multival) == Tensor or not math.isclose(self.mult_multival, 1.0):
            real_mult = self.mult_multival
            if type(real_mult) == Tensor:
                real_mult = extend_to_batch_size(prepare_mask_batch(real_mult, x.shape), b).squeeze(1)
            mask = mask * real_mult
        return mask


class ApplyAnimateDiffPIAModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_ApplyAnimateDiffModelWithPIA',
            display_name='Apply AnimateDiff-PIA Model 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA',
            inputs=[
                io.Custom("MOTION_MODEL_ADE").Input('motion_model'),
                io.Image.Input('image'),
                io.Vae.Input('vae'),
                io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001),
                io.Float.Input('end_percent', default=1.0, max=1.0, min=0.0, step=0.001),
                io.Custom("PIA_INPUT").Input('pia_input', optional=True),
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
    def execute(cls, motion_model: MotionModelPatcher, image: Tensor, vae: VAE,
                           start_percent: float=0.0, end_percent: float=1.0, pia_input: InputPIA=None,
                           motion_lora: MotionLoraList=None, ad_keyframes: ADKeyframeGroup=None,
                           scale_multival=None, effect_multival=None, ref_multival=None, per_block=None,
                           prev_m_models: MotionModelGroup=None,):
        new_m_models = ApplyAnimateDiffModelNode.execute( motion_model, start_percent=start_percent, end_percent=end_percent,
                                                                    motion_lora=motion_lora, ad_keyframes=ad_keyframes,
                                                                    scale_multival=scale_multival, effect_multival=effect_multival, per_block=per_block,
                                                                    prev_m_models=prev_m_models).args
        # most recent added model will always be first in list;
        curr_model = new_m_models[0].models[0]
        # confirm that model is PIA
        if curr_model.model.mm_info.mm_format != AnimateDiffFormat.PIA:
            raise Exception(f"Motion model '{curr_model.model.mm_info.mm_name}' is not a PIA model; cannot be used with Apply AnimateDiff-PIA Model node.")
        attachment = get_mm_attachment(curr_model)
        attachment.orig_pia_images = image
        attachment.pia_vae = vae
        if pia_input is None:
            pia_input = InputPIA_Multival(1.0)
        attachment.pia_input = pia_input
        #curr_model.pia_multival = ref_multival
        return io.NodeOutput(*new_m_models)


class LoadAnimateDiffAndInjectPIANode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_InjectPIAIntoAnimateDiffModel',
            display_name='🧪Inject PIA into AnimateDiff Model 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA/🧪experimental',
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
        # make sure model actually has PIA conv_in
        if motion_model.model.conv_in is None:
            raise Exception("Passed-in motion model was expected to be PIA (contain conv_in), but did not.")
        # load motion module and motion settings, if included
        loaded_motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=ad_settings)
        inject_pia_conv_in_into_model(motion_model=loaded_motion_model, w_pia=motion_model)
        return io.NodeOutput(loaded_motion_model,)


class PIA_ADKeyframeNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_PIA_AnimateDiffKeyframe',
            display_name='AnimateDiff-PIA Keyframe 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA',
            inputs=[
                io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001),
                io.Custom("AD_KEYFRAMES").Input('prev_ad_keyframes', optional=True),
                io.Custom("MULTIVAL").Input('scale_multival', optional=True),
                io.Custom("MULTIVAL").Input('effect_multival', optional=True),
                io.Custom("PIA_INPUT").Input('pia_input', optional=True),
                io.Boolean.Input('inherit_missing', optional=True, default=True),
                io.Int.Input('guarantee_steps', optional=True, default=1, max=9007199254740991, min=0),
            ],
            outputs=[
                io.Custom("AD_KEYFRAMES").Output('AD_KEYFRAMES'),
            ],
        )
    


    @classmethod
    def execute(cls,
                      start_percent: float, prev_ad_keyframes=None,
                      scale_multival: Union[float, torch.Tensor]=None, effect_multival: Union[float, torch.Tensor]=None,
                      pia_input: InputPIA=None,
                      inherit_missing: bool=True, guarantee_steps: int=1):
        return io.NodeOutput(*ADKeyframeNode.execute(
                    start_percent=start_percent, prev_ad_keyframes=prev_ad_keyframes,
                    scale_multival=scale_multival, effect_multival=effect_multival, pia_input=pia_input,
                    inherit_missing=inherit_missing, guarantee_steps=guarantee_steps
                ).args)


class InputPIA_MultivalNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_InputPIA_Multival',
            display_name='PIA Input [Multival] 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA',
            inputs=[
                io.Custom("MULTIVAL").Input('multival'),
            ],
            outputs=[
                io.Custom("PIA_INPUT").Output('PIA_INPUT'),
            ],
        )
    

    @classmethod
    def execute(cls, multival: Union[float, Tensor], effect_multival: Union[float, Tensor]=None):
        return io.NodeOutput(InputPIA_Multival(multival, effect_multival),)


class InputPIA_PaperPresetsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_InputPIA_PaperPresets',
            display_name='PIA Input [Paper Presets] 🎭🅐🅓②',
            category='Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA',
            inputs=[
                io.Combo.Input('preset', options=['Animation (Small Motion)', 'Animation (Medium Motion)', 'Animation (Large Motion)', 'Loop (Small Motion)', 'Loop (Medium Motion)', 'Loop (Large Motion)', 'Style Transfer (Small Motion)', 'Style Transfer (Medium Motion)', 'Style Transfer (Large Motion)']),
                io.Int.Input('batch_index', default=0, max=9007199254740991, min=-9007199254740991, step=1),
                io.Custom("MULTIVAL").Input('mult_multival', optional=True),
                io.Boolean.Input('print_values', optional=True, default=False),
            ],
            outputs=[
                io.Custom("PIA_INPUT").Output('PIA_INPUT'),
            ],
        )


    @classmethod
    def execute(cls, preset: str, batch_index: int, mult_multival: Union[float, Tensor]=None, print_values: bool=False, effect_multival: Union[float, Tensor]=None):
        # verify preset exists - function will throw error if does not
        values = PIA_RANGES.get_preset(preset)
        if print_values:
            logger.info(f"PIA Preset '{preset}': {values}")
        return io.NodeOutput(InputPIA_PaperPresets(preset=preset, index=batch_index, mult_multival=mult_multival, effect_multival=effect_multival),)
