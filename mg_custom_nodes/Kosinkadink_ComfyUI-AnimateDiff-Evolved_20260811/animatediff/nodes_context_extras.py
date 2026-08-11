from comfy_api.latest import io
from torch import Tensor
from typing import Union
from collections.abc import Iterable
from .context import ContextOptionsGroup
from .context_extras import ContextExtrasGroup, ContextRef, ContextRefTune, ContextRefMode, ContextRefKeyframeGroup, ContextRefKeyframe, NaiveReuse, NaiveReuseKeyframe, NaiveReuseKeyframeGroup
from .utils_model import BIGMAX, InterpolationMethod
from .utils_scheduling import convert_str_to_indexes
from .logger import logger

class SetContextExtrasOnContextOptions(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ContextExtras_Set', display_name='Set Context Extras 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/context extras', inputs=[io.Custom('CONTEXT_OPTIONS').Input('context_opts'), io.Custom('CONTEXT_EXTRAS').Input('context_extras', optional=True)], outputs=[io.Custom('CONTEXT_OPTIONS').Output('CONTEXT_OPTS')])

    @classmethod
    def execute(cls, context_opts: ContextOptionsGroup, context_extras: ContextExtrasGroup=None):
        context_opts = context_opts.clone()
        if context_extras is not None:
            context_opts.extras = context_extras.clone()
        return io.NodeOutput(context_opts)

class ContextExtras_NaiveReuse(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ContextExtras_NaiveReuse', display_name='Context Extras◆NaiveReuse 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/context extras', inputs=[io.Custom('CONTEXT_EXTRAS').Input('prev_extras', optional=True), io.Custom('MULTIVAL').Input('strength_multival', optional=True), io.Custom('NAIVEREUSE_KEYFRAME').Input('naivereuse_kf', optional=True), io.Float.Input('start_percent', optional=True, default=0.0, max=1.0, min=0.0, step=0.001), io.Float.Input('end_percent', optional=True, default=0.15, max=1.0, min=0.0, step=0.001), io.Float.Input('weighted_mean', optional=True, default=0.95, max=1.0, min=0.0, step=0.001)], outputs=[io.Custom('CONTEXT_EXTRAS').Output('CONTEXT_EXTRAS')])

    @classmethod
    def execute(cls, start_percent=0.0, end_percent=0.1, weighted_mean=0.95, strength_multival: Union[float, Tensor]=None, naivereuse_kf: NaiveReuseKeyframeGroup=None, prev_extras: ContextExtrasGroup=None):
        if prev_extras is None:
            prev_extras = prev_extras = ContextExtrasGroup()
        prev_extras = prev_extras.clone()
        naive_reuse = NaiveReuse(start_percent=start_percent, end_percent=end_percent, weighted_mean=weighted_mean, multival_opt=strength_multival, naivereuse_kf=naivereuse_kf)
        prev_extras.add(naive_reuse)
        return io.NodeOutput(prev_extras)

class NaiveReuse_KeyframeMultivalNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ContextExtras_NaiveReuse_Keyframe', display_name='NaiveReuse Keyframe 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/context extras/naivereuse', inputs=[io.Custom('NAIVEREUSE_KEYFRAME').Input('prev_kf', optional=True), io.Custom('MULTIVAL').Input('mult_multival', optional=True), io.Float.Input('mult', optional=True, default=1.0, max=1.0, min=0.0, step=0.001), io.Float.Input('start_percent', optional=True, default=0.0, max=1.0, min=0.0, step=0.001), io.Int.Input('guarantee_steps', optional=True, default=1, max=9007199254740991, min=0), io.Boolean.Input('inherit_missing', optional=True, default=True)], outputs=[io.Custom('NAIVEREUSE_KEYFRAME').Output('NAIVEREUSE_KF')])

    @classmethod
    def execute(cls, prev_kf=None, mult=1.0, mult_multival=None, start_percent=0.0, guarantee_steps=1, inherit_missing=True):
        if prev_kf is None:
            prev_kf = NaiveReuseKeyframeGroup()
        prev_kf = prev_kf.clone()
        kf = NaiveReuseKeyframe(mult=mult, mult_multival=mult_multival, start_percent=start_percent, guarantee_steps=guarantee_steps, inherit_missing=inherit_missing)
        prev_kf.add(kf)
        return io.NodeOutput(prev_kf)

class NaiveReuse_KeyframeInterpolationNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ContextExtras_NaiveReuse_KeyframeInterpolation', display_name='NaiveReuse Keyframes Interp. 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/context extras/naivereuse', inputs=[io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001), io.Float.Input('end_percent', default=1.0, max=1.0, min=0.0, step=0.001), io.Float.Input('mult_start', default=1.0, max=1.0, min=0.0, step=0.001), io.Float.Input('mult_end', default=1.0, max=1.0, min=0.0, step=0.001), io.Combo.Input('interpolation', options=['linear', 'ease_in', 'ease_out', 'ease_in_out']), io.Int.Input('intervals', default=50, max=100, min=2, step=1), io.Boolean.Input('inherit_missing', default=True), io.Boolean.Input('print_keyframes', default=False), io.Custom('NAIVEREUSE_KEYFRAME').Input('prev_kf', optional=True), io.Custom('MULTIVAL').Input('mult_multival', optional=True)], outputs=[io.Custom('NAIVEREUSE_KEYFRAME').Output('NAIVEREUSE_KF')])

    @classmethod
    def execute(cls, start_percent: float, end_percent: float, mult_start: float, mult_end: float, interpolation: str, intervals: int, inherit_missing=True, prev_kf: NaiveReuseKeyframeGroup=None, mult_multival=None, print_keyframes=False):
        if prev_kf is None:
            prev_kf = NaiveReuseKeyframeGroup()
        prev_kf = prev_kf.clone()
        prev_kf = prev_kf.clone()
        percents = InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=intervals, method=InterpolationMethod.LINEAR)
        mults = InterpolationMethod.get_weights(num_from=mult_start, num_to=mult_end, length=intervals, method=interpolation)
        is_first = True
        for percent, mult in zip(percents, mults):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_kf.add(NaiveReuseKeyframe(mult=mult, mult_multival=mult_multival, start_percent=percent, guarantee_steps=guarantee_steps, inherit_missing=inherit_missing))
            if print_keyframes:
                logger.info(f'NaiveReuseKeyframe - start_percent:{percent} = {mult}')
        return io.NodeOutput(prev_kf)

class NaiveReuse_KeyframeFromListNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ContextExtras_NaiveReuse_KeyframeFromList', display_name='NaiveReuse Keyframes From List 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/context extras/naivereuse', inputs=[io.Float.Input('mults_float', default=-1, force_input=True, min=-1, step=0.001), io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001), io.Float.Input('end_percent', default=1.0, max=1.0, min=0.0, step=0.001), io.Boolean.Input('inherit_missing', default=True), io.Boolean.Input('print_keyframes', default=False), io.Custom('NAIVEREUSE_KEYFRAME').Input('prev_kf', optional=True), io.Custom('MULTIVAL').Input('mult_multival', optional=True)], outputs=[io.Custom('NAIVEREUSE_KEYFRAME').Output('NAIVEREUSE_KF')])

    @classmethod
    def execute(cls, mults_float: Union[float, list[float]], start_percent: float, end_percent: float, inherit_missing=True, prev_kf: NaiveReuseKeyframeGroup=None, mult_multival=None, print_keyframes=False):
        if prev_kf is None:
            prev_kf = NaiveReuseKeyframeGroup()
        prev_kf = prev_kf.clone()
        if type(mults_float) in (float, int):
            mults_float = [float(mults_float)]
        elif isinstance(mults_float, Iterable):
            pass
        else:
            raise Exception(f'strengths_float must be either an interable input or a float, but was {type(mults_float).__repr__}.')
        percents = InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=len(mults_float), method=InterpolationMethod.LINEAR)
        is_first = True
        for percent, mult in zip(percents, mults_float):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_kf.add(NaiveReuseKeyframe(mult=mult, mult_multival=mult_multival, start_percent=percent, guarantee_steps=guarantee_steps, inherit_missing=inherit_missing))
            if print_keyframes:
                logger.info(f'NaiveReuseKeyframe - start_percent:{percent} = {mult}')
        return io.NodeOutput(prev_kf)

class ContextExtras_ContextRef(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ContextExtras_ContextRef', display_name='Context Extras◆ContextRef 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/context extras', inputs=[io.Custom('CONTEXT_EXTRAS').Input('prev_extras', optional=True), io.Custom('MULTIVAL').Input('strength_multival', optional=True), io.Custom('CONTEXTREF_MODE').Input('contextref_mode', optional=True), io.Custom('CONTEXTREF_TUNE').Input('contextref_tune', optional=True), io.Custom('CONTEXTREF_KEYFRAME').Input('contextref_kf', optional=True), io.Float.Input('start_percent', optional=True, default=0.0, max=1.0, min=0.0, step=0.001), io.Float.Input('end_percent', optional=True, default=0.25, max=1.0, min=0.0, step=0.001)], outputs=[io.Custom('CONTEXT_EXTRAS').Output('CONTEXT_EXTRAS')])

    @classmethod
    def execute(cls, start_percent=0.0, end_percent=0.1, strength_multival: Union[float, Tensor]=None, contextref_mode: ContextRefMode=None, contextref_tune: ContextRefTune=None, contextref_kf: ContextRefKeyframeGroup=None, prev_extras: ContextExtrasGroup=None):
        if prev_extras is None:
            prev_extras = prev_extras = ContextExtrasGroup()
        prev_extras = prev_extras.clone()
        if contextref_tune is None:
            contextref_tune = ContextRefTune(attn_style_fidelity=1.0, attn_ref_weight=1.0, attn_strength=1.0)
        if contextref_mode is None:
            contextref_mode = ContextRefMode.init_first()
        context_ref = ContextRef(start_percent=start_percent, end_percent=end_percent, strength_multival=strength_multival, tune=contextref_tune, mode=contextref_mode, keyframe=contextref_kf)
        prev_extras.add(context_ref)
        return io.NodeOutput(prev_extras)

class ContextRef_KeyframeMultivalNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ContextExtras_ContextRef_Keyframe', display_name='ContextRef Keyframe 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/context extras/contextref', inputs=[io.Custom('CONTEXTREF_KEYFRAME').Input('prev_kf', optional=True), io.Custom('MULTIVAL').Input('mult_multival', optional=True), io.Custom('CONTEXTREF_MODE').Input('mode_replace', optional=True), io.Custom('CONTEXTREF_TUNE').Input('tune_replace', optional=True), io.Float.Input('mult', optional=True, default=1.0, max=1.0, min=0.0, step=0.001), io.Float.Input('start_percent', optional=True, default=0.0, max=1.0, min=0.0, step=0.001), io.Int.Input('guarantee_steps', optional=True, default=1, max=9007199254740991, min=0), io.Boolean.Input('inherit_missing', optional=True, default=True)], outputs=[io.Custom('CONTEXTREF_KEYFRAME').Output('CONTEXTREF_KF')])

    @classmethod
    def execute(cls, prev_kf: ContextRefKeyframeGroup=None, mult=1.0, mult_multival=None, mode_replace=None, tune_replace=None, start_percent=1.0, guarantee_steps=1, inherit_missing=True):
        if prev_kf is None:
            prev_kf = ContextRefKeyframeGroup()
        prev_kf = prev_kf.clone()
        kf = ContextRefKeyframe(mult=mult, mult_multival=mult_multival, tune_replace=tune_replace, mode_replace=mode_replace, start_percent=start_percent, guarantee_steps=guarantee_steps, inherit_missing=inherit_missing)
        prev_kf.add(kf)
        return io.NodeOutput(prev_kf)

class ContextRef_KeyframeInterpolationNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ContextExtras_ContextRef_KeyframeInterpolation', display_name='ContextRef Keyframes Interp. 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/context extras/contextref', inputs=[io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001), io.Float.Input('end_percent', default=1.0, max=1.0, min=0.0, step=0.001), io.Float.Input('mult_start', default=1.0, max=1.0, min=0.0, step=0.001), io.Float.Input('mult_end', default=1.0, max=1.0, min=0.0, step=0.001), io.Combo.Input('interpolation', options=['linear', 'ease_in', 'ease_out', 'ease_in_out']), io.Int.Input('intervals', default=50, max=100, min=2, step=1), io.Boolean.Input('inherit_missing', default=True), io.Boolean.Input('print_keyframes', default=False), io.Custom('CONTEXTREF_KEYFRAME').Input('prev_kf', optional=True), io.Custom('MULTIVAL').Input('mult_multival', optional=True), io.Custom('CONTEXTREF_MODE').Input('mode_replace', optional=True), io.Custom('CONTEXTREF_TUNE').Input('tune_replace', optional=True)], outputs=[io.Custom('CONTEXTREF_KEYFRAME').Output('CONTEXTREF_KF')])

    @classmethod
    def execute(cls, start_percent: float, end_percent: float, mult_start: float, mult_end: float, interpolation: str, intervals: int, inherit_missing=True, prev_kf: ContextRefKeyframeGroup=None, mult_multival=None, mode_replace=None, tune_replace=None, print_keyframes=False):
        if prev_kf is None:
            prev_kf = ContextRefKeyframeGroup()
        prev_kf = prev_kf.clone()
        percents = InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=intervals, method=InterpolationMethod.LINEAR)
        mults = InterpolationMethod.get_weights(num_from=mult_start, num_to=mult_end, length=intervals, method=interpolation)
        is_first = True
        for percent, mult in zip(percents, mults):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_kf.add(ContextRefKeyframe(mult=mult, mult_multival=mult_multival, tune_replace=tune_replace, mode_replace=mode_replace, start_percent=percent, guarantee_steps=guarantee_steps, inherit_missing=inherit_missing))
            if print_keyframes:
                logger.info(f'ContextRefKeyframe - start_percent:{percent} = {mult}')
        return io.NodeOutput(prev_kf)

class ContextRef_KeyframeFromListNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ContextExtras_ContextRef_KeyframeFromList', display_name='ContextRef Keyframes From List 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/context extras/contextref', inputs=[io.Float.Input('mults_float', default=-1, force_input=True, min=-1, step=0.001), io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001), io.Float.Input('end_percent', default=1.0, max=1.0, min=0.0, step=0.001), io.Boolean.Input('inherit_missing', default=True), io.Boolean.Input('print_keyframes', default=False), io.Custom('CONTEXTREF_KEYFRAME').Input('prev_kf', optional=True), io.Custom('MULTIVAL').Input('mult_multival', optional=True), io.Custom('CONTEXTREF_MODE').Input('mode_replace', optional=True), io.Custom('CONTEXTREF_TUNE').Input('tune_replace', optional=True)], outputs=[io.Custom('CONTEXTREF_KEYFRAME').Output('CONTEXTREF_KF')])

    @classmethod
    def execute(cls, mults_float: Union[float, list[float]], start_percent: float, end_percent: float, inherit_missing=True, prev_kf: ContextRefKeyframeGroup=None, mult_multival=None, mode_replace=None, tune_replace=None, print_keyframes=False):
        if prev_kf is None:
            prev_kf = ContextRefKeyframeGroup()
        prev_kf = prev_kf.clone()
        if type(mults_float) in (float, int):
            mults_float = [float(mults_float)]
        elif isinstance(mults_float, Iterable):
            pass
        else:
            raise Exception(f'strengths_float must be either an interable input or a float, but was {type(mults_float).__repr__}.')
        percents = InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=len(mults_float), method=InterpolationMethod.LINEAR)
        is_first = True
        for percent, mult in zip(percents, mults_float):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_kf.add(ContextRefKeyframe(mult=mult, mult_multival=mult_multival, tune_replace=tune_replace, mode_replace=mode_replace, start_percent=percent, guarantee_steps=guarantee_steps, inherit_missing=inherit_missing))
            if print_keyframes:
                logger.info(f'ContextRefKeyframe - start_percent:{percent} = {mult}')
        return io.NodeOutput(prev_kf)

class ContextRef_ModeFirst(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ContextExtras_ContextRef_ModeFirst', display_name='ContextRef Mode◆First 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/context extras/contextref', inputs=[], outputs=[io.Custom('CONTEXTREF_MODE').Output('CONTEXTREF_MODE')])

    @classmethod
    def execute(cls):
        mode = ContextRefMode.init_first()
        return io.NodeOutput(mode)

class ContextRef_ModeSliding(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ContextExtras_ContextRef_ModeSliding', display_name='ContextRef Mode◆Sliding 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/context extras/contextref', inputs=[io.Int.Input('sliding_width', optional=True, default=2, max=9007199254740991, min=2, step=1)], outputs=[io.Custom('CONTEXTREF_MODE').Output('CONTEXTREF_MODE')])

    @classmethod
    def execute(cls, sliding_width):
        mode = ContextRefMode.init_sliding(sliding_width=sliding_width)
        return io.NodeOutput(mode)

class ContextRef_ModeIndexes(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ContextExtras_ContextRef_ModeIndexes', display_name='ContextRef Mode◆Indexes 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/context extras/contextref', inputs=[io.String.Input('switch_on_idxs', optional=True, default=''), io.Boolean.Input('always_include_0', optional=True, default=True)], outputs=[io.Custom('CONTEXTREF_MODE').Output('CONTEXTREF_MODE')])

    @classmethod
    def execute(cls, switch_on_idxs: str, always_include_0: bool):
        idxs = set(convert_str_to_indexes(indexes_str=switch_on_idxs, length=0, allow_range=False))
        if always_include_0 and 0 not in idxs:
            idxs.add(0)
        mode = ContextRefMode.init_indexes(indexes=idxs)
        return io.NodeOutput(mode)

class ContextRef_TuneAttnAdain(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ContextExtras_ContextRef_TuneAttnAdain', display_name='ContextRef Tune◆Attn+Adain 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/context extras/contextref', inputs=[io.Float.Input('attn_style_fidelity', optional=True, default=1.0, max=1.0, min=0.0, step=0.01), io.Float.Input('attn_ref_weight', optional=True, default=1.0, max=1.0, min=0.0, step=0.01), io.Float.Input('attn_strength', optional=True, default=1.0, max=1.0, min=0.0, step=0.01), io.Float.Input('adain_style_fidelity', optional=True, default=1.0, max=1.0, min=0.0, step=0.01), io.Float.Input('adain_ref_weight', optional=True, default=1.0, max=1.0, min=0.0, step=0.01), io.Float.Input('adain_strength', optional=True, default=1.0, max=1.0, min=0.0, step=0.01)], outputs=[io.Custom('CONTEXTREF_TUNE').Output('CONTEXTREF_TUNE')])

    @classmethod
    def execute(cls, attn_style_fidelity=1.0, attn_ref_weight=1.0, attn_strength=1.0, adain_style_fidelity=1.0, adain_ref_weight=1.0, adain_strength=1.0):
        params = ContextRefTune(attn_style_fidelity=attn_style_fidelity, adain_style_fidelity=adain_style_fidelity, attn_ref_weight=attn_ref_weight, adain_ref_weight=adain_ref_weight, attn_strength=attn_strength, adain_strength=adain_strength)
        return io.NodeOutput(params)

class ContextRef_TuneAttn(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ContextExtras_ContextRef_TuneAttn', display_name='ContextRef Tune◆Attn 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/context extras/contextref', inputs=[io.Float.Input('attn_style_fidelity', optional=True, default=1.0, max=1.0, min=0.0, step=0.01), io.Float.Input('attn_ref_weight', optional=True, default=1.0, max=1.0, min=0.0, step=0.01), io.Float.Input('attn_strength', optional=True, default=1.0, max=1.0, min=0.0, step=0.01)], outputs=[io.Custom('CONTEXTREF_TUNE').Output('CONTEXTREF_TUNE')])

    @classmethod
    def execute(cls, attn_style_fidelity=1.0, attn_ref_weight=1.0, attn_strength=1.0):
        output = ContextRef_TuneAttnAdain.execute(attn_style_fidelity=attn_style_fidelity, attn_ref_weight=attn_ref_weight, attn_strength=attn_strength, adain_ref_weight=0.0, adain_style_fidelity=0.0, adain_strength=0.0)
        return io.NodeOutput(*output.args)
