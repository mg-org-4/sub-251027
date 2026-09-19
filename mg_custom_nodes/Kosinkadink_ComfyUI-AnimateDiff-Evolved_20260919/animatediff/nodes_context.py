from comfy_api.latest import io
from torch import Tensor
from typing import Union
import comfy.samplers
from comfy.model_patcher import ModelPatcher
from .context import ContextFuseMethod, ContextOptions, ContextOptionsGroup, ContextSchedules, generate_context_visualization
from .utils_model import BIGMAX, MAX_RESOLUTION
LENGTH_MAX = 128
STRIDE_MAX = 32
OVERLAP_MAX = 128

class LoopedUniformContextOptionsNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_LoopedUniformContextOptions', display_name='Context Options◆Looped Uniform 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts', inputs=[io.Int.Input('context_length', default=16, max=128, min=1), io.Int.Input('context_stride', default=1, max=32, min=1), io.Int.Input('context_overlap', default=4, max=128, min=0), io.Boolean.Input('closed_loop', default=False), io.Combo.Input('fuse_method', options=['pyramid', 'flat', 'overlap-linear', '🔬delayed reverse sawtooth', '🔬pyramid-sigma', '🔬pyramid-sigma inverse', '🔬gauss-sigma', '🔬gauss-sigma inverse', '🔬random'], optional=True), io.Boolean.Input('use_on_equal_length', optional=True, default=False), io.Float.Input('start_percent', optional=True, default=0.0, max=1.0, min=0.0, step=0.001), io.Int.Input('guarantee_steps', optional=True, default=1, max=9007199254740991, min=0), io.Custom('CONTEXT_OPTIONS').Input('prev_context', optional=True), io.Custom('VIEW_OPTS').Input('view_opts', optional=True)], outputs=[io.Custom('CONTEXT_OPTIONS').Output('CONTEXT_OPTS')])

    @classmethod
    def execute(cls, context_length: int, context_stride: int, context_overlap: int, closed_loop: bool, fuse_method: str=ContextFuseMethod.FLAT, use_on_equal_length=False, start_percent: float=0.0, guarantee_steps: int=1, view_opts: ContextOptions=None, prev_context: ContextOptionsGroup=None):
        if prev_context is None:
            prev_context = ContextOptionsGroup()
        prev_context = prev_context.clone()
        context_options = ContextOptions(context_length=context_length, context_stride=context_stride, context_overlap=context_overlap, context_schedule=ContextSchedules.UNIFORM_LOOPED, closed_loop=closed_loop, fuse_method=fuse_method, use_on_equal_length=use_on_equal_length, start_percent=start_percent, guarantee_steps=guarantee_steps, view_options=view_opts)
        prev_context.add(context_options)
        return io.NodeOutput(prev_context)

class LegacyLoopedUniformContextOptionsNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_AnimateDiffUniformContextOptions', display_name='Context Options◆Looped Uniform 🎭🅐🅓', category='', inputs=[io.Int.Input('context_length', default=16, max=128, min=1), io.Int.Input('context_stride', default=1, max=32, min=1), io.Int.Input('context_overlap', default=4, max=128, min=0), io.Combo.Input('context_schedule', options=['uniform']), io.Boolean.Input('closed_loop', default=False), io.Combo.Input('fuse_method', options=['pyramid', 'flat', 'overlap-linear', '🔬delayed reverse sawtooth', '🔬pyramid-sigma', '🔬pyramid-sigma inverse', '🔬gauss-sigma', '🔬gauss-sigma inverse', '🔬random'], optional=True, default='flat'), io.Boolean.Input('use_on_equal_length', optional=True, default=False), io.Float.Input('start_percent', optional=True, default=0.0, max=1.0, min=0.0, step=0.001), io.Int.Input('guarantee_steps', optional=True, default=1, max=9007199254740991, min=0), io.Custom('CONTEXT_OPTIONS').Input('prev_context', optional=True), io.Custom('VIEW_OPTS').Input('view_opts', optional=True)], outputs=[io.Custom('CONTEXT_OPTIONS').Output('CONTEXT_OPTS')], is_deprecated=True)

    @classmethod
    def execute(cls, fuse_method: str=ContextFuseMethod.FLAT, context_schedule: str=None, **kwargs):
        return LoopedUniformContextOptionsNode.execute(fuse_method=fuse_method, **kwargs)

class StandardUniformContextOptionsNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_StandardUniformContextOptions', display_name='Context Options◆Standard Uniform 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts', inputs=[io.Int.Input('context_length', default=16, max=128, min=1), io.Int.Input('context_stride', default=1, max=32, min=1), io.Int.Input('context_overlap', default=4, max=128, min=0), io.Combo.Input('fuse_method', options=['pyramid', 'flat', 'overlap-linear', '🔬delayed reverse sawtooth', '🔬pyramid-sigma', '🔬pyramid-sigma inverse', '🔬gauss-sigma', '🔬gauss-sigma inverse', '🔬random'], optional=True), io.Boolean.Input('use_on_equal_length', optional=True, default=False), io.Float.Input('start_percent', optional=True, default=0.0, max=1.0, min=0.0, step=0.001), io.Int.Input('guarantee_steps', optional=True, default=1, max=9007199254740991, min=0), io.Custom('CONTEXT_OPTIONS').Input('prev_context', optional=True), io.Custom('VIEW_OPTS').Input('view_opts', optional=True)], outputs=[io.Custom('CONTEXT_OPTIONS').Output('CONTEXT_OPTS')])

    @classmethod
    def execute(cls, context_length: int, context_stride: int, context_overlap: int, fuse_method: str=ContextFuseMethod.PYRAMID, use_on_equal_length=False, start_percent: float=0.0, guarantee_steps: int=1, view_opts: ContextOptions=None, prev_context: ContextOptionsGroup=None):
        if prev_context is None:
            prev_context = ContextOptionsGroup()
        prev_context = prev_context.clone()
        context_options = ContextOptions(context_length=context_length, context_stride=context_stride, context_overlap=context_overlap, context_schedule=ContextSchedules.UNIFORM_STANDARD, closed_loop=False, fuse_method=fuse_method, use_on_equal_length=use_on_equal_length, start_percent=start_percent, guarantee_steps=guarantee_steps, view_options=view_opts)
        prev_context.add(context_options)
        return io.NodeOutput(prev_context)

class StandardStaticContextOptionsNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_StandardStaticContextOptions', display_name='Context Options◆Standard Static 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts', inputs=[io.Int.Input('context_length', default=16, max=128, min=1), io.Int.Input('context_overlap', default=4, max=128, min=0), io.Combo.Input('fuse_method', options=['pyramid', 'relative', 'flat', 'overlap-linear', '🔬delayed reverse sawtooth', '🔬pyramid-sigma', '🔬pyramid-sigma inverse', '🔬gauss-sigma', '🔬gauss-sigma inverse', '🔬random'], optional=True), io.Boolean.Input('use_on_equal_length', optional=True, default=False), io.Float.Input('start_percent', optional=True, default=0.0, max=1.0, min=0.0, step=0.001), io.Int.Input('guarantee_steps', optional=True, default=1, max=9007199254740991, min=0), io.Custom('CONTEXT_OPTIONS').Input('prev_context', optional=True), io.Custom('VIEW_OPTS').Input('view_opts', optional=True)], outputs=[io.Custom('CONTEXT_OPTIONS').Output('CONTEXT_OPTS')])

    @classmethod
    def execute(cls, context_length: int, context_overlap: int, fuse_method: str=ContextFuseMethod.PYRAMID, use_on_equal_length=False, start_percent: float=0.0, guarantee_steps: int=1, view_opts: ContextOptions=None, prev_context: ContextOptionsGroup=None):
        if prev_context is None:
            prev_context = ContextOptionsGroup()
        prev_context = prev_context.clone()
        context_options = ContextOptions(context_length=context_length, context_stride=None, context_overlap=context_overlap, context_schedule=ContextSchedules.STATIC_STANDARD, fuse_method=fuse_method, use_on_equal_length=use_on_equal_length, start_percent=start_percent, guarantee_steps=guarantee_steps, view_options=view_opts)
        prev_context.add(context_options)
        return io.NodeOutput(prev_context)

class BatchedContextOptionsNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_BatchedContextOptions', display_name='Context Options◆Batched [Non-AD] 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts', inputs=[io.Int.Input('context_length', default=16, max=128, min=1), io.Float.Input('start_percent', optional=True, default=0.0, max=1.0, min=0.0, step=0.001), io.Int.Input('guarantee_steps', optional=True, default=1, max=9007199254740991, min=0), io.Custom('CONTEXT_OPTIONS').Input('prev_context', optional=True)], outputs=[io.Custom('CONTEXT_OPTIONS').Output('CONTEXT_OPTS')])

    @classmethod
    def execute(cls, context_length: int, start_percent: float=0.0, guarantee_steps: int=1, prev_context: ContextOptionsGroup=None):
        if prev_context is None:
            prev_context = ContextOptionsGroup()
        prev_context = prev_context.clone()
        context_options = ContextOptions(context_length=context_length, context_overlap=0, context_schedule=ContextSchedules.BATCHED, start_percent=start_percent, guarantee_steps=guarantee_steps)
        prev_context.add(context_options)
        return io.NodeOutput(prev_context)

class ViewAsContextOptionsNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_ViewsOnlyContextOptions', display_name='Context Options◆Views Only [VRAM⇈] 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts', inputs=[io.Custom('VIEW_OPTS').Input('view_opts_req'), io.Float.Input('start_percent', optional=True, default=0.0, max=1.0, min=0.0, step=0.001), io.Int.Input('guarantee_steps', optional=True, default=1, max=9007199254740991, min=0), io.Custom('CONTEXT_OPTIONS').Input('prev_context', optional=True)], outputs=[io.Custom('CONTEXT_OPTIONS').Output('CONTEXT_OPTS')])

    @classmethod
    def execute(cls, view_opts_req: ContextOptions, start_percent: float=0.0, guarantee_steps: int=1, prev_context: ContextOptionsGroup=None):
        if prev_context is None:
            prev_context = ContextOptionsGroup()
        prev_context = prev_context.clone()
        context_options = ContextOptions(context_schedule=ContextSchedules.VIEW_AS_CONTEXT, start_percent=start_percent, guarantee_steps=guarantee_steps, view_options=view_opts_req, use_on_equal_length=True)
        prev_context.add(context_options)
        return io.NodeOutput(prev_context)

class StandardStaticViewOptionsNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_StandardStaticViewOptions', display_name='View Options◆Standard Static 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/view opts', inputs=[io.Int.Input('view_length', default=16, max=128, min=1), io.Int.Input('view_overlap', default=4, max=128, min=0), io.Combo.Input('fuse_method', options=['pyramid', 'flat', 'overlap-linear', '🔬delayed reverse sawtooth', '🔬pyramid-sigma', '🔬pyramid-sigma inverse', '🔬gauss-sigma', '🔬gauss-sigma inverse', '🔬random'], optional=True)], outputs=[io.Custom('VIEW_OPTS').Output('VIEW_OPTS')])

    @classmethod
    def execute(cls, view_length: int, view_overlap: int, fuse_method: str=ContextFuseMethod.FLAT):
        view_options = ContextOptions(context_length=view_length, context_stride=None, context_overlap=view_overlap, context_schedule=ContextSchedules.STATIC_STANDARD, fuse_method=fuse_method)
        return io.NodeOutput(view_options)

class StandardUniformViewOptionsNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_StandardUniformViewOptions', display_name='View Options◆Standard Uniform 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/view opts', inputs=[io.Int.Input('view_length', default=16, max=128, min=1), io.Int.Input('view_stride', default=1, max=32, min=1), io.Int.Input('view_overlap', default=4, max=128, min=0), io.Combo.Input('fuse_method', options=['pyramid', 'flat', 'overlap-linear', '🔬delayed reverse sawtooth', '🔬pyramid-sigma', '🔬pyramid-sigma inverse', '🔬gauss-sigma', '🔬gauss-sigma inverse', '🔬random'], optional=True)], outputs=[io.Custom('VIEW_OPTS').Output('VIEW_OPTS')])

    @classmethod
    def execute(cls, view_length: int, view_overlap: int, view_stride: int, fuse_method: str=ContextFuseMethod.PYRAMID):
        view_options = ContextOptions(context_length=view_length, context_stride=view_stride, context_overlap=view_overlap, context_schedule=ContextSchedules.UNIFORM_STANDARD, fuse_method=fuse_method)
        return io.NodeOutput(view_options)

class LoopedUniformViewOptionsNode(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_LoopedUniformViewOptions', display_name='View Options◆Looped Uniform 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/view opts', inputs=[io.Int.Input('view_length', default=16, max=128, min=1), io.Int.Input('view_stride', default=1, max=32, min=1), io.Int.Input('view_overlap', default=4, max=128, min=0), io.Boolean.Input('closed_loop', default=False), io.Combo.Input('fuse_method', options=['pyramid', 'flat', 'overlap-linear', '🔬delayed reverse sawtooth', '🔬pyramid-sigma', '🔬pyramid-sigma inverse', '🔬gauss-sigma', '🔬gauss-sigma inverse', '🔬random'], optional=True), io.Boolean.Input('use_on_equal_length', optional=True, default=False)], outputs=[io.Custom('VIEW_OPTS').Output('VIEW_OPTS')])

    @classmethod
    def execute(cls, view_length: int, view_overlap: int, view_stride: int, closed_loop: bool, fuse_method: str=ContextFuseMethod.PYRAMID, use_on_equal_length=False):
        view_options = ContextOptions(context_length=view_length, context_stride=view_stride, context_overlap=view_overlap, context_schedule=ContextSchedules.UNIFORM_LOOPED, closed_loop=closed_loop, fuse_method=fuse_method, use_on_equal_length=use_on_equal_length)
        return io.NodeOutput(view_options)

class VisualizeContextOptionsKAdv(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_VisualizeContextOptionsKAdv', display_name='Visualize Context Options (K.Adv.) 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/visualize', inputs=[io.Model.Input('model'), io.Combo.Input('sampler_name', options=comfy.samplers.KSampler.SAMPLERS), io.Combo.Input('scheduler', options=comfy.samplers.KSampler.SCHEDULERS), io.Custom('CONTEXT_OPTIONS').Input('context_opts', optional=True), io.Int.Input('visual_width', optional=True, default=1440, max=16384, min=32), io.Int.Input('latents_length', optional=True, default=32, max=9007199254740991, min=1), io.Int.Input('steps', optional=True, default=20, max=9007199254740991, min=0), io.Int.Input('start_step', optional=True, default=0, max=9007199254740991, min=0), io.Int.Input('end_step', optional=True, default=20, max=9007199254740991, min=1)], outputs=[io.Image.Output('IMAGE')])

    @classmethod
    def execute(cls, model: ModelPatcher, sampler_name: str, scheduler: str, context_opts: ContextOptionsGroup=None, visual_width=1440, latents_length=32, steps=20, start_step=0, end_step=20):
        images = generate_context_visualization(model=model, context_opts=context_opts, width=visual_width, video_length=latents_length, sampler_name=sampler_name, scheduler=scheduler, steps=steps, start_step=start_step, end_step=end_step)
        return io.NodeOutput(images)

class VisualizeContextOptionsK(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_VisualizeContextOptionsK', display_name='Visualize Context Options (K.) 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/visualize', inputs=[io.Model.Input('model'), io.Combo.Input('sampler_name', options=comfy.samplers.KSampler.SAMPLERS), io.Combo.Input('scheduler', options=comfy.samplers.KSampler.SCHEDULERS), io.Custom('CONTEXT_OPTIONS').Input('context_opts', optional=True), io.Int.Input('visual_width', optional=True, default=1440, max=16384, min=32), io.Int.Input('latents_length', optional=True, default=32, max=9007199254740991, min=1), io.Int.Input('steps', optional=True, default=20, max=9007199254740991, min=0), io.Float.Input('denoise', optional=True, default=1.0, max=1.0, min=0.0, step=0.01)], outputs=[io.Image.Output('IMAGE')])

    @classmethod
    def execute(cls, model: ModelPatcher, sampler_name: str, scheduler: str, context_opts: ContextOptionsGroup=None, visual_width=1440, latents_length=32, steps=20, denoise=1.0):
        images = generate_context_visualization(model=model, context_opts=context_opts, width=visual_width, video_length=latents_length, sampler_name=sampler_name, scheduler=scheduler, steps=steps, denoise=denoise)
        return io.NodeOutput(images)

class VisualizeContextOptionsSCustom(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_VisualizeContextOptionsSCustom', display_name='Visualize Context Options (S.Cus.) 🎭🅐🅓', category='Animate Diff 🎭🅐🅓/context opts/visualize', inputs=[io.Model.Input('model'), io.Sigmas.Input('sigmas'), io.Custom('CONTEXT_OPTIONS').Input('context_opts', optional=True), io.Int.Input('visual_width', optional=True, default=1440, max=16384, min=32), io.Int.Input('latents_length', optional=True, default=32, max=9007199254740991, min=1)], outputs=[io.Image.Output('IMAGE')])

    @classmethod
    def execute(cls, model: ModelPatcher, sigmas, context_opts: ContextOptionsGroup=None, visual_width=1440, latents_length=32):
        images = generate_context_visualization(model=model, context_opts=context_opts, width=visual_width, video_length=latents_length, sigmas=sigmas)
        return io.NodeOutput(images)
