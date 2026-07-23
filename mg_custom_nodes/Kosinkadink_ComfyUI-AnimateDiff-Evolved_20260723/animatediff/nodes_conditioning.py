from comfy_api.latest import io
import uuid
import folder_paths
from typing import Union
from torch import Tensor
from collections.abc import Iterable
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
import comfy.sd
from comfy.hooks import HookGroup, HookKeyframeGroup, HookKeyframe
import comfy_extras.nodes_hooks
import comfy.hooks
import comfy.utils
from .utils_model import BIGMAX, InterpolationMethod
from .logger import logger

class COND_CONST:
    COND_AREA_DEFAULT = 'default'
    COND_AREA_MASK_BOUNDS = 'mask bounds'
    _LIST_COND_AREA = [COND_AREA_DEFAULT, COND_AREA_MASK_BOUNDS]

class CreateLoraHookKeyframeInterpolationDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_LoraHookKeyframeInterpolation', display_name='LoRA Hook Keyframes Interp. 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning/schedule lora hooks', inputs=[io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001), io.Float.Input('end_percent', default=1.0, max=1.0, min=0.0, step=0.001), io.Float.Input('strength_start', default=1.0, max=10.0, min=0.0, step=0.001), io.Float.Input('strength_end', default=1.0, max=10.0, min=0.0, step=0.001), io.Combo.Input('interpolation', options=['linear', 'ease_in', 'ease_out', 'ease_in_out']), io.Int.Input('intervals', default=5, max=100, min=2, step=1), io.Boolean.Input('print_keyframes', default=False), io.Custom('HOOK_KEYFRAMES').Input('prev_hook_kf', optional=True)], outputs=[io.Custom('HOOK_KEYFRAMES').Output('HOOK_KF')], is_deprecated=True)

    @classmethod
    def execute(cls, start_percent: float, end_percent: float, strength_start: float, strength_end: float, interpolation: str, intervals: int, prev_hook_kf: HookKeyframeGroup=None, print_keyframes=False):
        if prev_hook_kf:
            prev_hook_kf = prev_hook_kf.clone()
        else:
            prev_hook_kf = HookKeyframeGroup()
        percents = InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=intervals, method=InterpolationMethod.LINEAR)
        strengths = InterpolationMethod.get_weights(num_from=strength_start, num_to=strength_end, length=intervals, method=interpolation)
        is_first = True
        for percent, strength in zip(percents, strengths):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_hook_kf.add(HookKeyframe(strength=strength, start_percent=percent, guarantee_steps=guarantee_steps))
            if print_keyframes:
                logger.info(f'HookKeyframe - start_percent:{percent} = {strength}')
        return io.NodeOutput(prev_hook_kf)

class PairedConditioningSetMaskHookedDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_PairedConditioningSetMask', display_name='Set Props on Conds 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning', inputs=[io.Conditioning.Input('positive_ADD'), io.Conditioning.Input('negative_ADD'), io.Float.Input('strength', default=1.0, max=10.0, min=0.0, step=0.01), io.Combo.Input('set_cond_area', options=['default', 'mask bounds']), io.Mask.Input('opt_mask', optional=True), io.Custom('HOOKS').Input('opt_lora_hook', optional=True), io.Custom('TIMESTEPS_RANGE').Input('opt_timesteps', optional=True)], outputs=[io.Conditioning.Output('positive'), io.Conditioning.Output('negative')], is_deprecated=True)

    @classmethod
    def execute(cls, positive_ADD, negative_ADD, strength: float, set_cond_area: str, opt_mask: Tensor=None, opt_lora_hook: HookGroup=None, opt_timesteps: tuple=None):
        final_positive, final_negative = comfy.hooks.set_conds_props(conds=[positive_ADD, negative_ADD], strength=strength, set_cond_area=set_cond_area, mask=opt_mask, hooks=opt_lora_hook, timesteps_range=opt_timesteps)
        return io.NodeOutput(final_positive, final_negative)

class ConditioningSetMaskHookedDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ConditioningSetMask', display_name='Set Props on Cond 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning/single cond ops', inputs=[io.Conditioning.Input('cond_ADD'), io.Float.Input('strength', default=1.0, max=10.0, min=0.0, step=0.01), io.Combo.Input('set_cond_area', options=['default', 'mask bounds']), io.Mask.Input('opt_mask', optional=True), io.Custom('HOOKS').Input('opt_lora_hook', optional=True), io.Custom('TIMESTEPS_RANGE').Input('opt_timesteps', optional=True)], outputs=[io.Conditioning.Output('CONDITIONING')], is_deprecated=True)

    @classmethod
    def execute(cls, cond_ADD, strength: float, set_cond_area: str, opt_mask: Tensor=None, opt_lora_hook: HookGroup=None, opt_timesteps: tuple=None):
        final_conditioning, = comfy.hooks.set_conds_props(conds=[cond_ADD], strength=strength, set_cond_area=set_cond_area, mask=opt_mask, hooks=opt_lora_hook, timesteps_range=opt_timesteps)
        return io.NodeOutput(final_conditioning)

class PairedConditioningSetMaskAndCombineHookedDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_PairedConditioningSetMaskAndCombine', display_name='Set Props and Combine Conds 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning', inputs=[io.Conditioning.Input('positive'), io.Conditioning.Input('negative'), io.Conditioning.Input('positive_ADD'), io.Conditioning.Input('negative_ADD'), io.Float.Input('strength', default=1.0, max=10.0, min=0.0, step=0.01), io.Combo.Input('set_cond_area', options=['default', 'mask bounds']), io.Mask.Input('opt_mask', optional=True), io.Custom('HOOKS').Input('opt_lora_hook', optional=True), io.Custom('TIMESTEPS_RANGE').Input('opt_timesteps', optional=True)], outputs=[io.Conditioning.Output('positive'), io.Conditioning.Output('negative')], is_deprecated=True)

    @classmethod
    def execute(cls, positive, negative, positive_ADD, negative_ADD, strength: float, set_cond_area: str, opt_mask: Tensor=None, opt_lora_hook: HookGroup=None, opt_timesteps: tuple=None):
        final_positive, final_negative = comfy.hooks.set_conds_props_and_combine(conds=[positive, negative], new_conds=[positive_ADD, negative_ADD], strength=strength, set_cond_area=set_cond_area, mask=opt_mask, hooks=opt_lora_hook, timesteps_range=opt_timesteps)
        return io.NodeOutput(final_positive, final_negative)

class ConditioningSetMaskAndCombineHookedDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ConditioningSetMaskAndCombine', display_name='Set Props and Combine Cond 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning/single cond ops', inputs=[io.Conditioning.Input('cond'), io.Conditioning.Input('cond_ADD'), io.Float.Input('strength', default=1.0, max=10.0, min=0.0, step=0.01), io.Combo.Input('set_cond_area', options=['default', 'mask bounds']), io.Mask.Input('opt_mask', optional=True), io.Custom('HOOKS').Input('opt_lora_hook', optional=True), io.Custom('TIMESTEPS_RANGE').Input('opt_timesteps', optional=True)], outputs=[io.Conditioning.Output('CONDITIONING')], is_deprecated=True)

    @classmethod
    def execute(cls, cond, cond_ADD, strength: float, set_cond_area: str, opt_mask: Tensor=None, opt_lora_hook: HookGroup=None, opt_timesteps: tuple=None):
        final_conditioning, = comfy.hooks.set_conds_props_and_combine(conds=[cond], new_conds=[cond_ADD], strength=strength, set_cond_area=set_cond_area, mask=opt_mask, hooks=opt_lora_hook, timesteps_range=opt_timesteps)
        return io.NodeOutput(final_conditioning)

class PairedConditioningSetUnmaskedAndCombineHookedDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_PairedConditioningSetUnmaskedAndCombine', display_name='Set Unmasked Conds 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning', inputs=[io.Conditioning.Input('positive'), io.Conditioning.Input('negative'), io.Conditioning.Input('positive_DEFAULT'), io.Conditioning.Input('negative_DEFAULT'), io.Custom('HOOKS').Input('opt_lora_hook', optional=True)], outputs=[io.Conditioning.Output('positive'), io.Conditioning.Output('negative')], is_deprecated=True)

    @classmethod
    def execute(cls, positive, negative, positive_DEFAULT, negative_DEFAULT, opt_lora_hook: HookGroup=None):
        final_positive, final_negative = comfy.hooks.set_default_conds_and_combine(conds=[positive, negative], new_conds=[positive_DEFAULT, negative_DEFAULT], hooks=opt_lora_hook)
        return io.NodeOutput(final_positive, final_negative)

class ConditioningSetUnmaskedAndCombineHookedDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ConditioningSetUnmaskedAndCombine', display_name='Set Unmasked Cond 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning/single cond ops', inputs=[io.Conditioning.Input('cond'), io.Conditioning.Input('cond_DEFAULT'), io.Custom('HOOKS').Input('opt_lora_hook', optional=True)], outputs=[io.Conditioning.Output('CONDITIONING')], is_deprecated=True)

    @classmethod
    def execute(cls, cond, cond_DEFAULT, opt_lora_hook: HookGroup=None):
        final_conditioning, = comfy.hooks.set_default_conds_and_combine(conds=[cond], new_conds=[cond_DEFAULT], hooks=opt_lora_hook)
        return io.NodeOutput(final_conditioning)

class PairedConditioningCombineDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_PairedConditioningCombine', display_name='Manual Combine Conds 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning', inputs=[io.Conditioning.Input('positive_A'), io.Conditioning.Input('negative_A'), io.Conditioning.Input('positive_B'), io.Conditioning.Input('negative_B')], outputs=[io.Conditioning.Output('positive'), io.Conditioning.Output('negative')], is_deprecated=True)

    @classmethod
    def execute(cls, positive_A, negative_A, positive_B, negative_B):
        final_positive, final_negative = comfy.hooks.set_conds_props_and_combine(conds=[positive_A, negative_A], new_conds=[positive_B, negative_B])
        return io.NodeOutput(final_positive, final_negative)

class ConditioningCombineDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ConditioningCombine', display_name='Manual Combine Cond 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning/single cond ops', inputs=[io.Conditioning.Input('cond_A'), io.Conditioning.Input('cond_B')], outputs=[io.Conditioning.Output('CONDITIONING')], is_deprecated=True)

    @classmethod
    def execute(cls, cond_A, cond_B):
        final_conditioning, = comfy.hooks.set_conds_props_and_combine(conds=[cond_A], new_conds=[cond_B])
        return io.NodeOutput(final_conditioning)

class ConditioningTimestepsNodeDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_TimestepsConditioning', display_name='Timesteps Conditioning 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning', inputs=[io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001), io.Float.Input('end_percent', default=1.0, max=1.0, min=0.0, step=0.001)], outputs=[io.Custom('TIMESTEPS_RANGE').Output('TIMESTEPS_RANGE')], is_deprecated=True)

    @classmethod
    def execute(cls, start_percent: float, end_percent: float):
        return io.NodeOutput((start_percent, end_percent))

class SetLoraHookKeyframesDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_SetLoraHookKeyframe', display_name='Set LoRA Hook Keyframes 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning', inputs=[io.Custom('HOOKS').Input('lora_hook'), io.Custom('HOOK_KEYFRAMES').Input('hook_kf')], outputs=[io.Custom('HOOKS').Output('HOOKS')], is_deprecated=True)

    @classmethod
    def execute(cls, lora_hook: HookGroup, hook_kf: HookKeyframeGroup):
        new_lora_hook = lora_hook.clone()
        new_lora_hook.set_keyframes_on_hooks(hook_kf=hook_kf)
        return io.NodeOutput(new_lora_hook)

class CreateLoraHookKeyframeDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_LoraHookKeyframe', display_name='LoRA Hook Keyframe 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning/schedule lora hooks', inputs=[io.Float.Input('strength_model', default=1.0, max=20.0, min=-20.0, step=0.01), io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001), io.Int.Input('guarantee_steps', default=1, max=9007199254740991, min=0), io.Custom('HOOK_KEYFRAMES').Input('prev_hook_kf', optional=True)], outputs=[io.Custom('HOOK_KEYFRAMES').Output('HOOK_KF')], is_deprecated=True)

    @classmethod
    def execute(cls, strength_model: float, start_percent: float, guarantee_steps: float, prev_hook_kf: HookKeyframeGroup=None):
        if prev_hook_kf:
            prev_hook_kf = prev_hook_kf.clone()
        else:
            prev_hook_kf = HookKeyframeGroup()
        keyframe = HookKeyframe(strength=strength_model, start_percent=start_percent, guarantee_steps=guarantee_steps)
        prev_hook_kf.add(keyframe)
        return io.NodeOutput(prev_hook_kf)

class CreateLoraHookKeyframeFromStrengthListDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_LoraHookKeyframeFromStrengthList', display_name='LoRA Hook Keyframes From List 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning/schedule lora hooks', inputs=[io.Float.Input('strengths_float', default=-1, force_input=True, min=-1, step=0.001), io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001), io.Float.Input('end_percent', default=1.0, max=1.0, min=0.0, step=0.001), io.Boolean.Input('print_keyframes', default=False), io.Custom('HOOK_KEYFRAMES').Input('prev_hook_kf', optional=True)], outputs=[io.Custom('HOOK_KEYFRAMES').Output('HOOK_KF')], is_deprecated=True)

    @classmethod
    def execute(cls, strengths_float: Union[float, list[float]], start_percent: float, end_percent: float, prev_hook_kf: HookKeyframeGroup=None, print_keyframes=False):
        if prev_hook_kf:
            prev_hook_kf = prev_hook_kf.clone()
        else:
            prev_hook_kf = HookKeyframeGroup()
        if type(strengths_float) in (float, int):
            strengths_float = [float(strengths_float)]
        elif isinstance(strengths_float, Iterable):
            pass
        else:
            raise Exception(f'strengths_float must be either an interable input or a float, but was {type(strengths_float).__repr__}.')
        percents = InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=len(strengths_float), method=InterpolationMethod.LINEAR)
        is_first = True
        for percent, strength in zip(percents, strengths_float):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_hook_kf.add(HookKeyframe(strength=strength, start_percent=percent, guarantee_steps=guarantee_steps))
            if print_keyframes:
                logger.info(f'HookKeyframe - start_percent:{percent} = {strength}')
        return io.NodeOutput(prev_hook_kf)

class MaskableLoraLoaderDEPR(io.ComfyNode):
    loaded_lora = None

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_RegisterLoraHook', display_name='Register LoRA Hook 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning/register lora hooks', inputs=[io.Model.Input('model'), io.Clip.Input('clip'), io.Combo.Input('lora_name', options=folder_paths.get_filename_list('loras')), io.Float.Input('strength_model', default=1.0, max=20.0, min=-20.0, step=0.01), io.Float.Input('strength_clip', default=1.0, max=20.0, min=-20.0, step=0.01)], outputs=[io.Model.Output('MODEL'), io.Clip.Output('CLIP'), io.Custom('HOOKS').Output('HOOKS')], is_deprecated=True)


    @classmethod
    def execute(cls, model: Union[ModelPatcher], clip: CLIP, lora_name: str, strength_model: float, strength_clip: float):
        if strength_model == 0 and strength_clip == 0:
            return io.NodeOutput(model, clip, None)
        lora_path = folder_paths.get_full_path('loras', lora_name)
        lora = None
        if cls.loaded_lora is not None:
            if cls.loaded_lora[0] == lora_path:
                lora = cls.loaded_lora[1]
            else:
                temp = cls.loaded_lora
                cls.loaded_lora = None
                del temp
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            cls.loaded_lora = (lora_path, lora)
        model_lora, clip_lora, hooks = comfy.hooks.load_hook_lora_for_models(model=model, clip=clip, lora=lora, strength_model=strength_model, strength_clip=strength_clip)
        return io.NodeOutput(model_lora, clip_lora, hooks)

class MaskableLoraLoaderModelOnlyDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_RegisterLoraHookModelOnly', display_name='Register LoRA Hook (Model Only) 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning/register lora hooks', inputs=[io.Model.Input('model'), io.Combo.Input('lora_name', options=folder_paths.get_filename_list('loras')), io.Float.Input('strength_model', default=1.0, max=20.0, min=-20.0, step=0.01)], outputs=[io.Model.Output('MODEL'), io.Custom('HOOKS').Output('HOOKS')], is_deprecated=True)

    @classmethod
    def execute(cls, model: ModelPatcher, lora_name: str, strength_model: float):
        model_lora, _, hooks = MaskableLoraLoaderDEPR.execute(model=model, clip=None, lora_name=lora_name, strength_model=strength_model, strength_clip=0).args
        return io.NodeOutput(model_lora, hooks)

class MaskableSDModelLoaderDEPR(io.ComfyNode, comfy_extras.nodes_hooks.CreateHookModelAsLora):
    loaded_weights = None

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_RegisterModelAsLoraHook', display_name='Register Model as LoRA Hook 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning/register lora hooks', inputs=[io.Model.Input('model'), io.Clip.Input('clip'), io.Combo.Input('ckpt_name', options=folder_paths.get_filename_list('checkpoints')), io.Float.Input('strength_model', default=1.0, max=20.0, min=-20.0, step=0.01), io.Float.Input('strength_clip', default=1.0, max=20.0, min=-20.0, step=0.01)], outputs=[io.Model.Output('MODEL'), io.Clip.Output('CLIP'), io.Custom('HOOKS').Output('HOOKS')], is_deprecated=True, is_experimental=True)

    @classmethod
    def execute(cls, model: ModelPatcher, clip: CLIP, ckpt_name: str, strength_model: float, strength_clip: float):
        returned = comfy_extras.nodes_hooks.CreateHookModelAsLora.create_hook(
            cls, ckpt_name=ckpt_name, strength_model=strength_model, strength_clip=strength_clip
        )
        return io.NodeOutput(model.clone(), clip.clone(), returned[0])

class MaskableSDModelLoaderModelOnlyDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_RegisterModelAsLoraHookModelOnly', display_name='Register Model as LoRA Hook (MO) 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning/register lora hooks', inputs=[io.Model.Input('model'), io.Combo.Input('ckpt_name', options=folder_paths.get_filename_list('checkpoints')), io.Float.Input('strength_model', default=1.0, max=20.0, min=-20.0, step=0.01)], outputs=[io.Model.Output('MODEL'), io.Custom('HOOKS').Output('HOOKS')], is_deprecated=True, is_experimental=True)

    @classmethod
    def execute(cls, model: ModelPatcher, ckpt_name: str, strength_model: float):
        model_lora, _, hooks = MaskableSDModelLoaderDEPR.execute(model=model, clip=None, ckpt_name=ckpt_name, strength_model=strength_model, strength_clip=0).args
        return io.NodeOutput(model_lora, hooks)

class SetModelLoraHookDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_AttachLoraHookToConditioning', display_name='Set Model LoRA Hook 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning/single cond ops', inputs=[io.Conditioning.Input('conditioning'), io.Custom('HOOKS').Input('lora_hook')], outputs=[io.Conditioning.Output('CONDITIONING')], is_deprecated=True)

    @classmethod
    def execute(cls, conditioning, lora_hook: HookGroup):
        return io.NodeOutput(comfy.hooks.set_hooks_for_conditioning(conditioning, lora_hook))

class SetClipLoraHookDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_AttachLoraHookToCLIP', display_name='Set CLIP LoRA Hook 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning', inputs=[io.Clip.Input('clip'), io.Custom('HOOKS').Input('lora_hook')], outputs=[io.Clip.Output('hook_CLIP')], is_deprecated=True)

    @classmethod
    def execute(cls, clip: CLIP, lora_hook: HookGroup):
        return io.NodeOutput(*comfy_extras.nodes_hooks.SetClipHooks.apply_hooks(cls, clip, False, lora_hook))

class CombineLoraHooksDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_CombineLoraHooks', display_name='Combine LoRA Hooks [2] 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning/combine lora hooks', inputs=[io.Custom('HOOKS').Input('lora_hook_A', optional=True), io.Custom('HOOKS').Input('lora_hook_B', optional=True)], outputs=[io.Custom('HOOKS').Output('HOOKS')], is_deprecated=True)

    @classmethod
    def execute(cls, lora_hook_A: HookGroup=None, lora_hook_B: HookGroup=None):
        candidates = [lora_hook_A, lora_hook_B]
        return io.NodeOutput(HookGroup.combine_all_hooks(candidates))

class CombineLoraHookFourOptionalDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_CombineLoraHooksFour', display_name='Combine LoRA Hooks [4] 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning/combine lora hooks', inputs=[io.Custom('HOOKS').Input('lora_hook_A', optional=True), io.Custom('HOOKS').Input('lora_hook_B', optional=True), io.Custom('HOOKS').Input('lora_hook_C', optional=True), io.Custom('HOOKS').Input('lora_hook_D', optional=True)], outputs=[io.Custom('HOOKS').Output('HOOKS')], is_deprecated=True)

    @classmethod
    def execute(cls, lora_hook_A: HookGroup=None, lora_hook_B: HookGroup=None, lora_hook_C: HookGroup=None, lora_hook_D: HookGroup=None):
        candidates = [lora_hook_A, lora_hook_B, lora_hook_C, lora_hook_D]
        return io.NodeOutput(HookGroup.combine_all_hooks(candidates))

class CombineLoraHookEightOptionalDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_CombineLoraHooksEight', display_name='Combine LoRA Hooks [8] 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/conditioning/combine lora hooks', inputs=[io.Custom('HOOKS').Input('lora_hook_A', optional=True), io.Custom('HOOKS').Input('lora_hook_B', optional=True), io.Custom('HOOKS').Input('lora_hook_C', optional=True), io.Custom('HOOKS').Input('lora_hook_D', optional=True), io.Custom('HOOKS').Input('lora_hook_E', optional=True), io.Custom('HOOKS').Input('lora_hook_F', optional=True), io.Custom('HOOKS').Input('lora_hook_G', optional=True), io.Custom('HOOKS').Input('lora_hook_H', optional=True)], outputs=[io.Custom('HOOKS').Output('HOOKS')], is_deprecated=True)

    @classmethod
    def execute(cls, lora_hook_A: HookGroup=None, lora_hook_B: HookGroup=None, lora_hook_C: HookGroup=None, lora_hook_D: HookGroup=None, lora_hook_E: HookGroup=None, lora_hook_F: HookGroup=None, lora_hook_G: HookGroup=None, lora_hook_H: HookGroup=None):
        candidates = [lora_hook_A, lora_hook_B, lora_hook_C, lora_hook_D, lora_hook_E, lora_hook_F, lora_hook_G, lora_hook_H]
        return io.NodeOutput(HookGroup.combine_all_hooks(candidates))
