from comfy_api.latest import io
from typing import Union
from torch import Tensor
from collections.abc import Iterable

from comfy.sd import VAE

from .freeinit import FreeInitFilter
from .sample_settings import (FreeInitOptions, IterationOptions, AncestralOptions,
                              NoiseLayerAdd, NoiseLayerAddWeighted, NoiseLayerNormalizedSum, NoiseLayerGroup, NoiseLayerReplace, NoiseLayerType,
                              SeedNoiseGeneration, SampleSettings, NoiseCalibration, NoiseDeterminism,
                              CustomCFGKeyframeGroup, CustomCFGKeyframe, CFGExtrasGroup, CFGExtras,
                              NoisedImageToInjectGroup, NoisedImageToInject, NoisedImageInjectOptions)
from .utils_model import BIGMIN, BIGMAX, MAX_RESOLUTION, SigmaSchedule, InterpolationMethod
from .cfg_extras import perturbed_attention_guidance_patch, rescale_cfg_patch, set_model_options_sampler_cfg_function, set_model_options_post_cfg_function
from .logger import logger


class SampleSettingsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_AnimateDiffSamplingSettings',
            display_name='Sample Settings 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓',
            inputs=[
                io.Int.Input('batch_offset', default=0, max=9007199254740991, min=0),
                io.Combo.Input('noise_type', options=NoiseLayerType.LIST),
                io.Combo.Input('seed_gen', options=SeedNoiseGeneration.LIST),
                io.Int.Input('seed_offset', default=0, max=9007199254740991, min=-9007199254740991),
                io.Custom("NOISE_LAYERS").Input('noise_layers', optional=True),
                io.Custom("ITERATION_OPTS").Input('iteration_opts', optional=True),
                io.Int.Input('seed_override', optional=True, default=0, force_input=True, max=18446744073709551615, min=0),
                io.Boolean.Input('adapt_denoise_steps', optional=True, default=False),
                io.Custom("CUSTOM_CFG").Input('custom_cfg', optional=True),
                io.Custom("SIGMA_SCHEDULE").Input('sigma_schedule', optional=True),
                io.Custom("IMAGE_INJECT").Input('image_inject', optional=True),
                io.Custom("ANCESTRAL_OPTS").Input('ancestral_opts', optional=True),
            ],
            outputs=[
                io.Custom("SAMPLE_SETTINGS").Output('settings'),
            ],
        )


    @classmethod
    def execute(cls, batch_offset: int, noise_type: str, seed_gen: str, seed_offset: int, noise_layers: NoiseLayerGroup=None,
                        iteration_opts: IterationOptions=None, seed_override: int=None, adapt_denoise_steps=False,
                        custom_cfg: CustomCFGKeyframeGroup=None, sigma_schedule: SigmaSchedule=None, image_inject: NoisedImageToInjectGroup=None,
                        noise_calib: NoiseCalibration=None, ancestral_opts=None) -> io.NodeOutput:
        sampling_settings = SampleSettings(batch_offset=batch_offset, noise_type=noise_type, seed_gen=seed_gen, seed_offset=seed_offset, noise_layers=noise_layers,
                                           iteration_opts=iteration_opts, seed_override=seed_override, adapt_denoise_steps=adapt_denoise_steps,
                                           custom_cfg=custom_cfg, sigma_schedule=sigma_schedule, image_injection=image_inject, noise_calibration=noise_calib,
                                           ancestral_opts=ancestral_opts)
        return io.NodeOutput(sampling_settings,)


class AncestralOptionsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_AncestralOptions',
            display_name='Ancestral Options 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings',
            inputs=[
                io.Combo.Input('noise_type', options=NoiseLayerType.LIST_ANCESTRAL),
                io.Int.Input('seed_offset', default=0, max=9007199254740991, min=-9007199254740991),
                io.Int.Input('seed_override', optional=True, default=0, force_input=True, max=18446744073709551615, min=0),
            ],
            outputs=[
                io.Custom("ANCESTRAL_OPTS").Output('ANCESTRAL_OPTS'),
            ],
        )


    @classmethod
    def execute(cls, noise_type: str, seed_offset: int, determinism: str=NoiseDeterminism.DEFAULT, seed_override: int=None) -> io.NodeOutput:
        if isinstance(seed_override, Iterable):
            raise Exception("Passing in a list of seeds for Ancestral Options is not supported at this time.")
        return io.NodeOutput(AncestralOptions(noise_type=noise_type, determinism=determinism, seed_offset=seed_offset, seed_override=seed_override),)


class NoiseLayerReplaceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_NoiseLayerReplace',
            display_name='Noise Layer [Replace] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/noise layers',
            inputs=[
                io.Int.Input('batch_offset', default=0, max=9007199254740991, min=0),
                io.Combo.Input('noise_type', options=NoiseLayerType.LIST),
                io.Combo.Input('seed_gen_override', options=SeedNoiseGeneration.LIST_WITH_OVERRIDE),
                io.Int.Input('seed_offset', default=0, max=9007199254740991, min=-9007199254740991),
                io.Custom("NOISE_LAYERS").Input('prev_noise_layers', optional=True),
                io.Mask.Input('mask_optional', optional=True),
                io.Int.Input('seed_override', optional=True, default=0, force_input=True, max=18446744073709551615, min=0),
            ],
            outputs=[
                io.Custom("NOISE_LAYERS").Output('NOISE_LAYERS'),
            ],
        )


    @classmethod
    def execute(cls, batch_offset: int, noise_type: str, seed_gen_override: str, seed_offset: int,
                      prev_noise_layers: NoiseLayerGroup=None, mask_optional: Tensor=None, seed_override: int=None,) -> io.NodeOutput:
        # prepare prev_noise_layers
        if prev_noise_layers is None:
            prev_noise_layers = NoiseLayerGroup()
        prev_noise_layers = prev_noise_layers.clone()
        # create layer
        layer = NoiseLayerReplace(noise_type=noise_type, batch_offset=batch_offset, seed_gen_override=seed_gen_override, seed_offset=seed_offset,
                                  seed_override=seed_override, mask=mask_optional)
        prev_noise_layers.add_to_start(layer)
        return io.NodeOutput(prev_noise_layers,)


class NoiseLayerAddNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_NoiseLayerAdd',
            display_name='Noise Layer [Add] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/noise layers',
            inputs=[
                io.Int.Input('batch_offset', default=0, max=9007199254740991, min=0),
                io.Combo.Input('noise_type', options=NoiseLayerType.LIST),
                io.Combo.Input('seed_gen_override', options=SeedNoiseGeneration.LIST_WITH_OVERRIDE),
                io.Int.Input('seed_offset', default=0, max=9007199254740991, min=-9007199254740991),
                io.Float.Input('noise_weight', default=0.5, max=10.0, min=0.0, step=0.001),
                io.Custom("NOISE_LAYERS").Input('prev_noise_layers', optional=True),
                io.Mask.Input('mask_optional', optional=True),
                io.Int.Input('seed_override', optional=True, default=0, force_input=True, max=18446744073709551615, min=0),
            ],
            outputs=[
                io.Custom("NOISE_LAYERS").Output('NOISE_LAYERS'),
            ],
        )


    @classmethod
    def execute(cls, batch_offset: int, noise_type: str, seed_gen_override: str, seed_offset: int,
                      noise_weight: float,
                      prev_noise_layers: NoiseLayerGroup=None, mask_optional: Tensor=None, seed_override: int=None,) -> io.NodeOutput:
        # prepare prev_noise_layers
        if prev_noise_layers is None:
            prev_noise_layers = NoiseLayerGroup()
        prev_noise_layers = prev_noise_layers.clone()
        # create layer
        layer = NoiseLayerAdd(noise_type=noise_type, batch_offset=batch_offset, seed_gen_override=seed_gen_override, seed_offset=seed_offset,
                              seed_override=seed_override, mask=mask_optional,
                              noise_weight=noise_weight)
        prev_noise_layers.add_to_start(layer)
        return io.NodeOutput(prev_noise_layers,)


class NoiseLayerAddWeightedNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_NoiseLayerAddWeighted',
            display_name='Noise Layer [Add Weighted] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/noise layers',
            inputs=[
                io.Int.Input('batch_offset', default=0, max=9007199254740991, min=0),
                io.Combo.Input('noise_type', options=NoiseLayerType.LIST),
                io.Combo.Input('seed_gen_override', options=SeedNoiseGeneration.LIST_WITH_OVERRIDE),
                io.Int.Input('seed_offset', default=0, max=9007199254740991, min=-9007199254740991),
                io.Float.Input('noise_weight', default=0.5, max=10.0, min=0.0, step=0.001),
                io.Float.Input('balance_multiplier', default=1.0, min=0.0, step=0.001),
                io.Custom("NOISE_LAYERS").Input('prev_noise_layers', optional=True),
                io.Mask.Input('mask_optional', optional=True),
                io.Int.Input('seed_override', optional=True, default=0, force_input=True, max=18446744073709551615, min=0),
            ],
            outputs=[
                io.Custom("NOISE_LAYERS").Output('NOISE_LAYERS'),
            ],
        )


    @classmethod
    def execute(cls, batch_offset: int, noise_type: str, seed_gen_override: str, seed_offset: int,
                      noise_weight: float, balance_multiplier: float,
                      prev_noise_layers: NoiseLayerGroup=None, mask_optional: Tensor=None, seed_override: int=None,) -> io.NodeOutput:
        # prepare prev_noise_layers
        if prev_noise_layers is None:
            prev_noise_layers = NoiseLayerGroup()
        prev_noise_layers = prev_noise_layers.clone()
        # create layer
        layer = NoiseLayerAddWeighted(noise_type=noise_type, batch_offset=batch_offset, seed_gen_override=seed_gen_override, seed_offset=seed_offset,
                              seed_override=seed_override, mask=mask_optional,
                              noise_weight=noise_weight, balance_multiplier=balance_multiplier)
        prev_noise_layers.add_to_start(layer)
        return io.NodeOutput(prev_noise_layers,)


class NoiseLayerNormalizedSumNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_NoiseLayerNormalizedSum',
            display_name='Noise Layer [Normalized Sum] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/noise layers',
            inputs=[
                io.Int.Input('batch_offset', default=0, max=9007199254740991, min=0),
                io.Combo.Input('noise_type', options=NoiseLayerType.LIST),
                io.Combo.Input('seed_gen_override', options=SeedNoiseGeneration.LIST_WITH_OVERRIDE),
                io.Int.Input('seed_offset', default=0, max=9007199254740991, min=-9007199254740991),
                io.Float.Input('noise_weight', default=0.5, max=1.0, min=0.0, step=0.001),
                io.Custom("NOISE_LAYERS").Input('prev_noise_layers', optional=True),
                io.Mask.Input('mask_optional', optional=True),
                io.Int.Input('seed_override', optional=True, default=0, force_input=True, max=18446744073709551615, min=0),
            ],
            outputs=[
                io.Custom("NOISE_LAYERS").Output('NOISE_LAYERS'),
            ],
        )


    @classmethod
    def execute(cls, batch_offset: int, noise_type: str, seed_gen_override: str, seed_offset: int,
                      noise_weight: float,
                      prev_noise_layers: NoiseLayerGroup=None, mask_optional: Tensor=None, seed_override: int=None,) -> io.NodeOutput:
        # prepare prev_noise_layers
        if prev_noise_layers is None:
            prev_noise_layers = NoiseLayerGroup()
        prev_noise_layers = prev_noise_layers.clone()
        # create layer
        layer = NoiseLayerNormalizedSum(noise_type=noise_type, batch_offset=batch_offset, seed_gen_override=seed_gen_override, seed_offset=seed_offset,
                              seed_override=seed_override, mask=mask_optional,
                              noise_weight=noise_weight)
        prev_noise_layers.add_to_start(layer)
        return io.NodeOutput(prev_noise_layers,)


class IterationOptionsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_IterationOptsDefault',
            display_name='Default Iteration Options 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/iteration opts',
            inputs=[
                io.Int.Input('iterations', default=1, min=1),
                io.Int.Input('iter_batch_offset', optional=True, default=0, max=9007199254740991, min=0),
                io.Int.Input('iter_seed_offset', optional=True, default=0, max=9007199254740991, min=-9007199254740991),
            ],
            outputs=[
                io.Custom("ITERATION_OPTS").Output('ITERATION_OPTS'),
            ],
        )


    @classmethod
    def execute(cls, iterations: int, iter_batch_offset: int=0, iter_seed_offset: int=0) -> io.NodeOutput:
        iter_opts = IterationOptions(iterations=iterations, iter_batch_offset=iter_batch_offset, iter_seed_offset=iter_seed_offset)
        return io.NodeOutput(iter_opts,)


class FreeInitOptionsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_IterationOptsFreeInit',
            display_name='FreeInit Iteration Options 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/iteration opts',
            inputs=[
                io.Int.Input('iterations', default=2, min=1),
                io.Combo.Input('filter', options=FreeInitFilter.LIST),
                io.Float.Input('d_s', default=0.25, max=1.0, min=0.0, step=0.001),
                io.Float.Input('d_t', default=0.25, max=1.0, min=0.0, step=0.001),
                io.Int.Input('n_butterworth', default=4, max=100, min=1),
                io.Int.Input('sigma_step', default=999, max=999, min=1),
                io.Boolean.Input('apply_to_1st_iter', default=False),
                io.Combo.Input('init_type', options=FreeInitOptions.LIST),
                io.Int.Input('iter_batch_offset', optional=True, default=0, max=9007199254740991, min=0),
                io.Int.Input('iter_seed_offset', optional=True, default=1, max=9007199254740991, min=-9007199254740991),
            ],
            outputs=[
                io.Custom("ITERATION_OPTS").Output('ITERATION_OPTS'),
            ],
        )


    @classmethod
    def execute(cls, iterations: int, filter: str, d_s: float, d_t: float, n_butterworth: int,
                         sigma_step: int, apply_to_1st_iter: bool, init_type: str,
                         iter_batch_offset: int=0, iter_seed_offset: int=1) -> io.NodeOutput:
        # init_type does nothing for now, not until I add more methods of applying low+high freq noise
        iter_opts = FreeInitOptions(iterations=iterations, step=sigma_step, apply_to_1st_iter=apply_to_1st_iter,
                                    filter=filter, d_s=d_s, d_t=d_t, n=n_butterworth, init_type=init_type,
                                    iter_batch_offset=iter_batch_offset, iter_seed_offset=iter_seed_offset)
        return io.NodeOutput(iter_opts,)


class NoiseCalibrationNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "calib_iterations": ("INT", {"default": 1, "min": 1, "step": 1}),
                "thresh_freq": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("NOISE_CALIBRATION",)
    RETURN_NAMES = ("NOISE_CALIB",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings"
    FUNCTION = "create_noisecalibration"

    def create_noisecalibration(self, calib_iterations: int, thresh_freq: float):
        noise_calib = NoiseCalibration(scale=thresh_freq, calib_iterations=calib_iterations)
        return (noise_calib,)


class CustomCFGNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_CustomCFG',
            display_name='Custom CFG [Multival] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/custom cfg',
            inputs=[
                io.Custom("MULTIVAL").Input('cfg_multival'),
                io.Custom("CFG_EXTRAS").Input('cfg_extras', optional=True),
            ],
            outputs=[
                io.Custom("CUSTOM_CFG").Output('CUSTOM_CFG'),
            ],
        )


    @classmethod
    def execute(cls, cfg_multival: Union[float, Tensor], cfg_extras: CFGExtrasGroup=None) -> io.NodeOutput:
        keyframe = CustomCFGKeyframe(cfg_multival=cfg_multival, cfg_extras=cfg_extras)
        cfg_custom = CustomCFGKeyframeGroup()
        cfg_custom.add(keyframe)
        return io.NodeOutput(cfg_custom,)


class CustomCFGSimpleNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_CustomCFGSimple',
            display_name='Custom CFG 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/custom cfg',
            inputs=[
                io.Float.Input('cfg', default=8.0, max=100.0, min=0.0, step=0.1),
                io.Custom("CFG_EXTRAS").Input('cfg_extras', optional=True),
            ],
            outputs=[
                io.Custom("CUSTOM_CFG").Output('CUSTOM_CFG'),
            ],
        )


    @classmethod
    def execute(cls, cfg: float, cfg_extras: CFGExtrasGroup=None) -> io.NodeOutput:
        return CustomCFGNode.execute( cfg_multival=cfg, cfg_extras=cfg_extras)


class CustomCFGKeyframeNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_CustomCFGKeyframe',
            display_name='Custom CFG Keyframe [Multival] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/custom cfg',
            inputs=[
                io.Custom("MULTIVAL").Input('cfg_multival'),
                io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001),
                io.Int.Input('guarantee_steps', default=1, max=9007199254740991, min=0),
                io.Custom("CUSTOM_CFG").Input('prev_custom_cfg', optional=True),
                io.Custom("CFG_EXTRAS").Input('cfg_extras', optional=True),
            ],
            outputs=[
                io.Custom("CUSTOM_CFG").Output('CUSTOM_CFG'),
            ],
        )


    @classmethod
    def execute(cls, cfg_multival: Union[float, Tensor], start_percent: float=0.0, guarantee_steps: int=1,
                          prev_custom_cfg: CustomCFGKeyframeGroup=None, cfg_extras: CFGExtrasGroup=None) -> io.NodeOutput:
        if not prev_custom_cfg:
            prev_custom_cfg = CustomCFGKeyframeGroup()
        prev_custom_cfg = prev_custom_cfg.clone()
        keyframe = CustomCFGKeyframe(cfg_multival=cfg_multival, start_percent=start_percent, guarantee_steps=guarantee_steps, cfg_extras=cfg_extras)
        prev_custom_cfg.add(keyframe)
        return io.NodeOutput(prev_custom_cfg,)


class CustomCFGKeyframeSimpleNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_CustomCFGKeyframeSimple',
            display_name='Custom CFG Keyframe 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/custom cfg',
            inputs=[
                io.Float.Input('cfg', default=8.0, max=100.0, min=0.0, step=0.1),
                io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001),
                io.Int.Input('guarantee_steps', default=1, max=9007199254740991, min=0),
                io.Custom("CUSTOM_CFG").Input('prev_custom_cfg', optional=True),
                io.Custom("CFG_EXTRAS").Input('cfg_extras', optional=True),
            ],
            outputs=[
                io.Custom("CUSTOM_CFG").Output('CUSTOM_CFG'),
            ],
        )


    @classmethod
    def execute(cls, cfg: float, start_percent: float=0.0, guarantee_steps: int=1,
                          prev_custom_cfg: CustomCFGKeyframeGroup=None, cfg_extras: CFGExtrasGroup=None) -> io.NodeOutput:
        return CustomCFGKeyframeNode.execute( cfg_multival=cfg, start_percent=start_percent,
                                                       guarantee_steps=guarantee_steps, prev_custom_cfg=prev_custom_cfg, cfg_extras=cfg_extras)


class CustomCFGKeyframeInterpolationNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_CustomCFGKeyframeInterpolation',
            display_name='Custom CFG Keyframes Interp. 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/custom cfg',
            inputs=[
                io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001),
                io.Float.Input('end_percent', default=1.0, max=1.0, min=0.0, step=0.001),
                io.Float.Input('cfg_start', default=8.0, max=100.0, min=0.0, step=0.1),
                io.Float.Input('cfg_end', default=8.0, max=100.0, min=0.0, step=0.1),
                io.Combo.Input('interpolation', options=InterpolationMethod._LIST),
                io.Int.Input('intervals', default=50, max=100, min=2, step=1),
                io.Boolean.Input('print_keyframes', default=False),
                io.Custom("CUSTOM_CFG").Input('prev_custom_cfg', optional=True),
                io.Custom("CFG_EXTRAS").Input('cfg_extras', optional=True),
            ],
            outputs=[
                io.Custom("CUSTOM_CFG").Output('CUSTOM_CFG'),
            ],
        )


    @classmethod
    def execute(cls,
                          start_percent: float, end_percent: float,
                          cfg_start: float, cfg_end: float, interpolation: str, intervals: int,
                          prev_custom_cfg: CustomCFGKeyframeGroup=None, cfg_extras: CFGExtrasGroup=None,
                          print_keyframes=False) -> io.NodeOutput:
        if not prev_custom_cfg:
            prev_custom_cfg = CustomCFGKeyframeGroup()
        prev_custom_cfg = prev_custom_cfg.clone()
        percents = InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=intervals, method=InterpolationMethod.LINEAR)
        cfgs = InterpolationMethod.get_weights(num_from=cfg_start, num_to=cfg_end, length=intervals, method=interpolation)

        is_first = True
        for percent, cfg in zip(percents, cfgs):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_custom_cfg.add(CustomCFGKeyframe(cfg_multival=float(cfg), start_percent=percent, guarantee_steps=guarantee_steps, cfg_extras=cfg_extras))
            if print_keyframes:
                logger.info(f"CustomCFGKeyframe - start_percent:{percent} = {cfg}")
        return io.NodeOutput(prev_custom_cfg,)


class CustomCFGKeyframeFromListNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_CustomCFGKeyframeFromList',
            display_name='Custom CFG Keyframes From List 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/custom cfg',
            inputs=[
                io.Float.Input('cfgs_float', default=-1, force_input=True, min=-1, step=0.001),
                io.Float.Input('start_percent', default=0.0, max=1.0, min=0.0, step=0.001),
                io.Float.Input('end_percent', default=1.0, max=1.0, min=0.0, step=0.001),
                io.Boolean.Input('print_keyframes', default=False),
                io.Custom("CUSTOM_CFG").Input('prev_custom_cfg', optional=True),
                io.Custom("CFG_EXTRAS").Input('cfg_extras', optional=True),
            ],
            outputs=[
                io.Custom("CUSTOM_CFG").Output('CUSTOM_CFG'),
            ],
        )


    @classmethod
    def execute(cls, cfgs_float: Union[float, list[float]],
                              start_percent: float, end_percent: float,
                              prev_custom_cfg: CustomCFGKeyframeGroup=None, cfg_extras: CFGExtrasGroup=None,
                              print_keyframes=False) -> io.NodeOutput:
        if not prev_custom_cfg:
            prev_custom_cfg = CustomCFGKeyframeGroup()
        prev_custom_cfg = prev_custom_cfg.clone()
        if type(cfgs_float) in (float, int):
            cfgs_float = [float(cfgs_float)]
        elif isinstance(cfgs_float, Iterable):
            pass
        else:
            raise Exception(f"strengths_float must be either an interable input or a float, but was {type(cfgs_float).__repr__}.")
        percents = InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=len(cfgs_float), method=InterpolationMethod.LINEAR)

        is_first = True
        for percent, cfg in zip(percents, cfgs_float):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_custom_cfg.add(CustomCFGKeyframe(cfg_multival=float(cfg), start_percent=percent, guarantee_steps=guarantee_steps, cfg_extras=cfg_extras))
            if print_keyframes:
                logger.info(f"CustomCFGKeyframe - start_percent:{percent} = {cfg}")
        return io.NodeOutput(prev_custom_cfg,)


class CFGExtrasPAGNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_CFGExtrasPAG',
            display_name='CFG Extras◆PAG [Multival] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/cfg extras',
            inputs=[
                io.Custom("MULTIVAL").Input('scale_multival'),
                io.Custom("CFG_EXTRAS").Input('prev_extras', optional=True),
            ],
            outputs=[
                io.Custom("CFG_EXTRAS").Output('CFG_EXTRAS'),
            ],
        )



    @classmethod
    def execute(cls, scale_multival: Union[float, Tensor], prev_extras: CFGExtrasGroup=None) -> io.NodeOutput:
        if prev_extras is None:
            prev_extras = CFGExtrasGroup()
        prev_extras = prev_extras.clone()

        patch = perturbed_attention_guidance_patch(scale_multival)
        def call_extras(model_options: dict[str]):
            return set_model_options_post_cfg_function(model_options.copy(), patch)

        extra = CFGExtras(call_extras)
        prev_extras.add(extra)
        return io.NodeOutput(prev_extras,)


class CFGExtrasPAGSimpleNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_CFGExtrasPAGSimple',
            display_name='CFG Extras◆PAG 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/cfg extras',
            inputs=[
                io.Float.Input('scale', default=3.0, max=100.0, min=0.0, round=0.01, step=0.1),
                io.Custom("CFG_EXTRAS").Input('prev_extras', optional=True),
            ],
            outputs=[
                io.Custom("CFG_EXTRAS").Output('CFG_EXTRAS'),
            ],
        )



    @classmethod
    def execute(cls, scale: float, prev_extras: CFGExtrasGroup=None) -> io.NodeOutput:
        return CFGExtrasPAGNode.execute(scale_multival=scale, prev_extras=prev_extras)


class CFGExtrasRescaleCFGNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_CFGExtrasRescaleCFG',
            display_name='CFG Extras◆RescaleCFG [Multival] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/cfg extras',
            inputs=[
                io.Custom("MULTIVAL").Input('mult_multival'),
                io.Custom("CFG_EXTRAS").Input('prev_extras', optional=True),
            ],
            outputs=[
                io.Custom("CFG_EXTRAS").Output('CFG_EXTRAS'),
            ],
        )



    @classmethod
    def execute(cls, mult_multival: Union[float, Tensor], prev_extras: CFGExtrasGroup=None) -> io.NodeOutput:
        if prev_extras is None:
            prev_extras = CFGExtrasGroup()
        prev_extras = prev_extras.clone()

        patch = rescale_cfg_patch(mult_multival)
        def call_extras(model_options: dict[str]):
            return set_model_options_sampler_cfg_function(model_options.copy(), patch)

        extra = CFGExtras(call_extras)
        prev_extras.add(extra)
        return io.NodeOutput(prev_extras,)


class CFGExtrasRescaleCFGSimpleNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_CFGExtrasRescaleCFGSimple',
            display_name='CFG Extras◆RescaleCFG 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/cfg extras',
            inputs=[
                io.Float.Input('multiplier', default=0.7, max=1.0, min=0.0, step=0.01),
                io.Custom("CFG_EXTRAS").Input('prev_extras', optional=True),
            ],
            outputs=[
                io.Custom("CFG_EXTRAS").Output('CFG_EXTRAS'),
            ],
        )



    @classmethod
    def execute(cls, multiplier: float, prev_extras: CFGExtrasGroup=None) -> io.NodeOutput:
        return CFGExtrasRescaleCFGNode.execute(mult_multival=multiplier, prev_extras=prev_extras)


class NoisedImageInjectionNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_NoisedImageInjection',
            display_name='Image Injection 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/image inject',
            inputs=[
                io.Image.Input('image'),
                io.Vae.Input('vae'),
                io.Mask.Input('mask_opt', optional=True),
                io.Boolean.Input('invert_mask', optional=True, default=False),
                io.Boolean.Input('resize_image', optional=True, default=True),
                io.Float.Input('start_percent', optional=True, default=0.0, max=1.0, min=0.0, step=0.001),
                io.Int.Input('guarantee_steps', optional=True, default=1, max=9007199254740991, min=1),
                io.Custom("IMAGE_INJECT_OPTIONS").Input('img_inject_opts', optional=True),
                io.Custom("MULTIVAL").Input('strength_multival', optional=True),
                io.Custom("IMAGE_INJECT").Input('prev_image_inject', optional=True),
            ],
            outputs=[
                io.Custom("IMAGE_INJECT").Output('IMAGE_INJECT'),
            ],
        )


    @classmethod
    def execute(cls, image: Tensor, vae: VAE, invert_mask: bool, resize_image: bool, start_percent: float,
                            mask_opt: Tensor=None, strength_multival: Union[float, Tensor]=None, prev_image_inject: NoisedImageToInjectGroup=None, guarantee_steps=1,
                            img_inject_opts=None) -> io.NodeOutput:
        if not prev_image_inject:
            prev_image_inject = NoisedImageToInjectGroup()
        prev_image_inject = prev_image_inject.clone()
        to_inject = NoisedImageToInject(image=image, mask=mask_opt, vae=vae, invert_mask=invert_mask, resize_image=resize_image, strength_multival=strength_multival,
                                        start_percent=start_percent, guarantee_steps=guarantee_steps,
                                        img_inject_opts=img_inject_opts)
        prev_image_inject.add(to_inject)
        return io.NodeOutput(prev_image_inject,)


class NoisedImageInjectOptionsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_NoisedImageInjectOptions',
            display_name='Image Injection Options 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/image inject',
            inputs=[
                io.Int.Input('composite_x', optional=True, default=0, max=16384, min=0, step=1),
                io.Int.Input('composite_y', optional=True, default=0, max=16384, min=0, step=1),
            ],
            outputs=[
                io.Custom("IMAGE_INJECT_OPTIONS").Output('IMG_INJECT_OPTS'),
            ],
        )


    @classmethod
    def execute(cls, composite_x=0, composite_y=0) -> io.NodeOutput:
        return io.NodeOutput(NoisedImageInjectOptions(x=composite_x, y=composite_y),)
