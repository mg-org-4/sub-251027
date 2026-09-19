from comfy_api.latest import io
import json
import os
import shutil
import subprocess
from typing import Dict, List
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
from comfy.model_patcher import ModelPatcher
from .ad_settings import AnimateDiffSettings, AdjustGroup, AdjustPE, AdjustWeight
from .context import ContextOptionsGroup, ContextOptions, ContextSchedules
from .logger import logger
from .utils_model import Folders, BetaSchedules, get_available_motion_models
from .utils_motion import ADKeyframeGroup
from .motion_lora import MotionLoraList
from .model_injection import ModelPatcherHelper, InjectionParams, MotionModelGroup, get_mm_attachment, load_motion_module_gen1
from .sampling import outer_sample_wrapper, sliding_calc_cond_batch
from .sample_settings import SampleSettings

class AnimateDiffLoaderDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='AnimateDiffLoaderV1', display_name='🚫AnimateDiff Loader [DEPRECATED] 🎭🅐🅓', category='', inputs=[io.Model.Input('model'), io.Custom('LATENT').Input('latents'), io.Combo.Input('model_name', options=get_available_motion_models()), io.Boolean.Input('unlimited_area_hack', default=False), io.Combo.Input('beta_schedule', options=['sqrt_linear (AnimateDiff)', 'use existing', 'autoselect', 'linear (AnimateDiff-SDXL)', 'linear (HotshotXL/default)', 'avg(sqrt_linear,linear)', 'lcm avg(sqrt_linear,linear)', 'lcm', 'lcm[100_ots]', 'lcm >> sqrt_linear', 'sqrt', 'cosine', 'squaredcos_cap_v2'])], outputs=[io.Model.Output('MODEL'), io.Custom('LATENT').Output('LATENT')], is_deprecated=True)

    @classmethod
    def execute(cls, model: ModelPatcher, latents: Dict[str, torch.Tensor], model_name: str, unlimited_area_hack: bool, beta_schedule: str):
        motion_model = load_motion_module_gen1(model_name, model)
        init_frames_len = len(latents['samples'])
        params = InjectionParams(unlimited_area_hack=unlimited_area_hack, apply_v2_properly=False)
        model = model.clone()
        helper = ModelPatcherHelper(model)
        helper.set_all_properties(outer_sampler_wrapper=outer_sample_wrapper, calc_cond_batch_wrapper=sliding_calc_cond_batch, params=params, motion_models=MotionModelGroup(motion_model))
        if beta_schedule == BetaSchedules.AUTOSELECT and (not model.motion_models.is_empty()):
            beta_schedule = model.motion_models[0].model.get_best_beta_schedule(log=True)
        new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, model)
        if new_model_sampling is not None:
            model.add_object_patch('model_sampling', new_model_sampling)
        del motion_model
        return io.NodeOutput(model, latents)

class AnimateDiffLoaderAdvancedDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_AnimateDiffLoaderV1Advanced', display_name='🚫AnimateDiff Loader (Advanced) [DEPRECATED] 🎭🅐🅓', category='', inputs=[io.Model.Input('model'), io.Custom('LATENT').Input('latents'), io.Combo.Input('model_name', options=get_available_motion_models()), io.Boolean.Input('unlimited_area_hack', default=False), io.Int.Input('context_length', default=16, max=1000, min=0), io.Int.Input('context_stride', default=1, max=1000, min=1), io.Int.Input('context_overlap', default=4, max=1000, min=0), io.Combo.Input('context_schedule', options=['uniform']), io.Boolean.Input('closed_loop', default=False), io.Combo.Input('beta_schedule', options=['sqrt_linear (AnimateDiff)', 'use existing', 'autoselect', 'linear (AnimateDiff-SDXL)', 'linear (HotshotXL/default)', 'avg(sqrt_linear,linear)', 'lcm avg(sqrt_linear,linear)', 'lcm', 'lcm[100_ots]', 'lcm >> sqrt_linear', 'sqrt', 'cosine', 'squaredcos_cap_v2'])], outputs=[io.Model.Output('MODEL'), io.Custom('LATENT').Output('LATENT')], is_deprecated=True)

    @classmethod
    def execute(cls, model: ModelPatcher, latents: Dict[str, torch.Tensor], model_name: str, unlimited_area_hack: bool, context_length: int, context_stride: int, context_overlap: int, context_schedule: str, closed_loop: bool, beta_schedule: str):
        motion_model = load_motion_module_gen1(model_name, model)
        init_frames_len = len(latents['samples'])
        params = InjectionParams(unlimited_area_hack=unlimited_area_hack, apply_v2_properly=False)
        context_group = ContextOptionsGroup()
        context_group.add(ContextOptions(context_length=context_length, context_stride=context_stride, context_overlap=context_overlap, context_schedule=context_schedule, closed_loop=closed_loop))
        params.set_context(context_options=context_group)
        model = model.clone()
        helper = ModelPatcherHelper(model)
        helper.set_all_properties(outer_sampler_wrapper=outer_sample_wrapper, calc_cond_batch_wrapper=sliding_calc_cond_batch, params=params, motion_models=MotionModelGroup(motion_model))
        if beta_schedule == BetaSchedules.AUTOSELECT and (not model.motion_models.is_empty()):
            beta_schedule = model.motion_models[0].model.get_best_beta_schedule(log=True)
        new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, model)
        if new_model_sampling is not None:
            model.add_object_patch('model_sampling', new_model_sampling)
        del motion_model
        return io.NodeOutput(model, latents)

class LegacyAnimateDiffLoaderWithContextDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_AnimateDiffLoaderWithContext', display_name='AnimateDiff Loader [Legacy] 🎭🅐🅓①', category='Animate Diff 🎭🅐🅓/① Gen1 nodes ①', inputs=[io.Model.Input('model'), io.Combo.Input('model_name', options=get_available_motion_models()), io.Combo.Input('beta_schedule', options=['autoselect', 'use existing', 'sqrt_linear (AnimateDiff)', 'linear (AnimateDiff-SDXL)', 'linear (HotshotXL/default)', 'avg(sqrt_linear,linear)', 'lcm avg(sqrt_linear,linear)', 'lcm', 'lcm[100_ots]', 'lcm >> sqrt_linear', 'sqrt', 'cosine', 'squaredcos_cap_v2'], default='autoselect'), io.Custom('CONTEXT_OPTIONS').Input('context_options', optional=True), io.Custom('MOTION_LORA').Input('motion_lora', optional=True), io.Custom('AD_SETTINGS').Input('ad_settings', optional=True), io.Custom('SAMPLE_SETTINGS').Input('sample_settings', optional=True), io.Float.Input('motion_scale', optional=True, default=1.0, min=0.0, step=0.001), io.Boolean.Input('apply_v2_models_properly', optional=True, default=True), io.Custom('AD_KEYFRAMES').Input('ad_keyframes', optional=True)], outputs=[io.Model.Output('MODEL')], is_deprecated=True)

    @classmethod
    def execute(cls, model: ModelPatcher, model_name: str, beta_schedule: str, context_options: ContextOptionsGroup=None, motion_lora: MotionLoraList=None, ad_settings: AnimateDiffSettings=None, motion_model_settings: AnimateDiffSettings=None, sample_settings: SampleSettings=None, motion_scale: float=1.0, apply_v2_models_properly: bool=False, ad_keyframes: ADKeyframeGroup=None):
        if ad_settings is not None:
            motion_model_settings = ad_settings
        motion_model = load_motion_module_gen1(model_name, model, motion_lora=motion_lora, motion_model_settings=motion_model_settings)
        params = InjectionParams(unlimited_area_hack=False, apply_v2_properly=apply_v2_models_properly)
        if context_options:
            params.set_context(context_options)
        if not motion_model_settings:
            motion_model_settings = AnimateDiffSettings()
        motion_model_settings.attn_scale = motion_scale
        params.set_motion_model_settings(motion_model_settings)
        attachment = get_mm_attachment(motion_model)
        if params.motion_model_settings.mask_attn_scale is not None:
            attachment.scale_multival = params.motion_model_settings.mask_attn_scale * params.motion_model_settings.attn_scale
        else:
            attachment.scale_multival = params.motion_model_settings.attn_scale
        attachment.keyframes = ad_keyframes.clone() if ad_keyframes else ADKeyframeGroup()
        model = model.clone()
        helper = ModelPatcherHelper(model)
        helper.set_all_properties(outer_sampler_wrapper=outer_sample_wrapper, calc_cond_batch_wrapper=sliding_calc_cond_batch, params=params, sample_settings=sample_settings, motion_models=MotionModelGroup(motion_model))
        sample_settings = helper.get_sample_settings()
        if sample_settings.custom_cfg is not None:
            logger.info('[Sample Settings] custom_cfg is set; will override any KSampler cfg values or patches.')
        if sample_settings.sigma_schedule is not None:
            logger.info('[Sample Settings] sigma_schedule is set; will override beta_schedule.')
            model.add_object_patch('model_sampling', sample_settings.sigma_schedule.clone().model_sampling)
        else:
            if beta_schedule == BetaSchedules.AUTOSELECT and helper.get_motion_models():
                beta_schedule = helper.get_motion_models()[0].model.get_best_beta_schedule(log=True)
            new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, model)
            if new_model_sampling is not None:
                model.add_object_patch('model_sampling', new_model_sampling)
        del motion_model
        return io.NodeOutput(model)

class AnimateDiffCombineDEPR(io.ComfyNode):

    @classmethod
    def get_formats(cls):
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path is not None:
            return ['image/gif', 'image/webp'] + ['video/' + x[:-5] for x in folder_paths.get_filename_list(Folders.VIDEO_FORMATS)]
        cls.ffmpeg_warning_already_shown = True
        return ['image/gif', 'image/webp']

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_AnimateDiffCombine', display_name='🚫AnimateDiff Combine [DEPRECATED, Use Video Combine (VHS) Instead!] 🎭🅐🅓', category='', inputs=[io.Image.Input('images'), io.Int.Input('frame_rate', default=8, max=24, min=1, step=1), io.Int.Input('loop_count', default=0, max=100, min=0, step=1), io.String.Input('filename_prefix', default='AnimateDiff'), io.Combo.Input('format', options=cls.get_formats()), io.Boolean.Input('pingpong', default=False), io.Boolean.Input('save_image', default=True)], outputs=[io.Custom('GIF').Output('GIF')], is_deprecated=True, is_output_node=True, hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo])
    ffmpeg_warning_already_shown = False

    @classmethod
    def execute(cls, images, frame_rate: int, loop_count: int, filename_prefix='AnimateDiff', format='image/gif', pingpong=False, save_image=True, prompt=None, extra_pnginfo=None):
        prompt = cls.hidden.prompt
        extra_pnginfo = cls.hidden.extra_pnginfo
        logger.warning('Do not use AnimateDiff Combine node, it is deprecated. Use Video Combine node from ComfyUI-VideoHelperSuite instead. Video nodes from VideoHelperSuite are actively maintained, more feature-rich, and also automatically attempts to get ffmpeg.')
        frames: List[Image.Image] = []
        for image in images:
            img = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            frames.append(img)
        output_dir = folder_paths.get_output_directory() if save_image else folder_paths.get_temp_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir)
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text('prompt', json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
        file = f'{filename}_{counter:05}_.png'
        file_path = os.path.join(full_output_folder, file)
        frames[0].save(file_path, pnginfo=metadata, compress_level=4)
        if pingpong:
            frames = frames + frames[-2:0:-1]
        format_type, format_ext = format.split('/')
        file = f'{filename}_{counter:05}_.{format_ext}'
        file_path = os.path.join(full_output_folder, file)
        if format_type == 'image':
            frames[0].save(file_path, format=format_ext.upper(), save_all=True, append_images=frames[1:], duration=round(1000 / frame_rate), loop=loop_count, compress_level=4)
        else:
            ffmpeg_path = shutil.which('ffmpeg')
            if ffmpeg_path is None:
                raise ProcessLookupError('Could not find ffmpeg')
            video_format_path = folder_paths.get_full_path('video_formats', format_ext + '.json')
            with open(video_format_path, 'r') as stream:
                video_format = json.load(stream)
            file = f"{filename}_{counter:05}_.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            dimensions = f'{frames[0].width}x{frames[0].height}'
            args = [ffmpeg_path, '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', dimensions, '-r', str(frame_rate), '-i', '-'] + video_format['main_pass'] + [file_path]
            env = os.environ.copy()
            if 'environment' in video_format:
                env.update(video_format['environment'])
            with subprocess.Popen(args, stdin=subprocess.PIPE, env=env) as proc:
                for frame in frames:
                    proc.stdin.write(frame.tobytes())
        previews = [{'filename': file, 'subfolder': subfolder, 'type': 'output' if save_image else 'temp', 'format': format}]
        return io.NodeOutput.from_dict({'ui': {'images': previews, 'animated': (True,)}})

class AnimateDiffModelSettingsDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_AnimateDiffModelSettings_Release', display_name='🚫[DEPR] Motion Model Settings 🎭🅐🅓①', category='', inputs=[io.Float.Input('min_motion_scale', default=1.0, min=0.0, step=0.001), io.Float.Input('max_motion_scale', default=1.0, min=0.0, step=0.001), io.Mask.Input('mask_motion_scale', optional=True)], outputs=[io.Custom('AD_SETTINGS').Output('AD_SETTINGS')], is_deprecated=True)

    @classmethod
    def execute(cls, mask_motion_scale: torch.Tensor=None, min_motion_scale: float=1.0, max_motion_scale: float=1.0):
        motion_model_settings = AnimateDiffSettings(mask_attn_scale=mask_motion_scale, mask_attn_scale_min=min_motion_scale, mask_attn_scale_max=max_motion_scale)
        return io.NodeOutput(motion_model_settings)

class AnimateDiffModelSettingsSimpleDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_AnimateDiffModelSettingsSimple', display_name='🚫[DEPR] Motion Model Settings (Simple) 🎭🅐🅓①', category='', inputs=[io.Int.Input('motion_pe_stretch', default=0, min=0, step=1), io.Mask.Input('mask_motion_scale', optional=True), io.Float.Input('min_motion_scale', optional=True, default=1.0, min=0.0, step=0.001), io.Float.Input('max_motion_scale', optional=True, default=1.0, min=0.0, step=0.001)], outputs=[io.Custom('AD_SETTINGS').Output('AD_SETTINGS')], is_deprecated=True)

    @classmethod
    def execute(cls, motion_pe_stretch: int, mask_motion_scale: torch.Tensor=None, min_motion_scale: float=1.0, max_motion_scale: float=1.0):
        adjust_pe = AdjustGroup(AdjustPE(motion_pe_stretch=motion_pe_stretch))
        motion_model_settings = AnimateDiffSettings(adjust_pe=adjust_pe, mask_attn_scale=mask_motion_scale, mask_attn_scale_min=min_motion_scale, mask_attn_scale_max=max_motion_scale)
        return io.NodeOutput(motion_model_settings)

class AnimateDiffModelSettingsAdvancedDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_AnimateDiffModelSettings', display_name='🚫[DEPR] Motion Model Settings (Advanced) 🎭🅐🅓①', category='', inputs=[io.Float.Input('pe_strength', default=1.0, max=10.0, min=0.0, step=0.0001), io.Float.Input('attn_strength', default=1.0, max=10.0, min=0.0, step=0.0001), io.Float.Input('other_strength', default=1.0, max=10.0, min=0.0, step=0.0001), io.Int.Input('motion_pe_stretch', default=0, min=0, step=1), io.Int.Input('cap_initial_pe_length', default=0, min=0, step=1), io.Int.Input('interpolate_pe_to_length', default=0, min=0, step=1), io.Int.Input('initial_pe_idx_offset', default=0, min=0, step=1), io.Int.Input('final_pe_idx_offset', default=0, min=0, step=1), io.Mask.Input('mask_motion_scale', optional=True), io.Float.Input('min_motion_scale', optional=True, default=1.0, min=0.0, step=0.001), io.Float.Input('max_motion_scale', optional=True, default=1.0, min=0.0, step=0.001)], outputs=[io.Custom('AD_SETTINGS').Output('AD_SETTINGS')], is_deprecated=True)

    @classmethod
    def execute(cls, pe_strength: float, attn_strength: float, other_strength: float, motion_pe_stretch: int, cap_initial_pe_length: int, interpolate_pe_to_length: int, initial_pe_idx_offset: int, final_pe_idx_offset: int, mask_motion_scale: torch.Tensor=None, min_motion_scale: float=1.0, max_motion_scale: float=1.0):
        adjust_pe = AdjustGroup(AdjustPE(motion_pe_stretch=motion_pe_stretch, cap_initial_pe_length=cap_initial_pe_length, interpolate_pe_to_length=interpolate_pe_to_length, initial_pe_idx_offset=initial_pe_idx_offset, final_pe_idx_offset=final_pe_idx_offset))
        adjust_weight = AdjustGroup(AdjustWeight(pe_MULT=pe_strength, attn_MULT=attn_strength, other_MULT=other_strength))
        motion_model_settings = AnimateDiffSettings(adjust_pe=adjust_pe, adjust_weight=adjust_weight, mask_attn_scale=mask_motion_scale, mask_attn_scale_min=min_motion_scale, mask_attn_scale_max=max_motion_scale)
        return io.NodeOutput(motion_model_settings)

class AnimateDiffModelSettingsAdvancedAttnStrengthsDEPR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='ADE_AnimateDiffModelSettingsAdvancedAttnStrengths', display_name='🚫[DEPR] Motion Model Settings (Adv. Attn) 🎭🅐🅓①', category='', inputs=[io.Float.Input('pe_strength', default=1.0, max=10.0, min=0.0, step=0.0001), io.Float.Input('attn_strength', default=1.0, max=10.0, min=0.0, step=0.0001), io.Float.Input('attn_q_strength', default=1.0, max=10.0, min=0.0, step=0.0001), io.Float.Input('attn_k_strength', default=1.0, max=10.0, min=0.0, step=0.0001), io.Float.Input('attn_v_strength', default=1.0, max=10.0, min=0.0, step=0.0001), io.Float.Input('attn_out_weight_strength', default=1.0, max=10.0, min=0.0, step=0.0001), io.Float.Input('attn_out_bias_strength', default=1.0, max=10.0, min=0.0, step=0.0001), io.Float.Input('other_strength', default=1.0, max=10.0, min=0.0, step=0.0001), io.Int.Input('motion_pe_stretch', default=0, min=0, step=1), io.Int.Input('cap_initial_pe_length', default=0, min=0, step=1), io.Int.Input('interpolate_pe_to_length', default=0, min=0, step=1), io.Int.Input('initial_pe_idx_offset', default=0, min=0, step=1), io.Int.Input('final_pe_idx_offset', default=0, min=0, step=1), io.Mask.Input('mask_motion_scale', optional=True), io.Float.Input('min_motion_scale', optional=True, default=1.0, min=0.0, step=0.001), io.Float.Input('max_motion_scale', optional=True, default=1.0, min=0.0, step=0.001)], outputs=[io.Custom('AD_SETTINGS').Output('AD_SETTINGS')], is_deprecated=True)

    @classmethod
    def execute(cls, pe_strength: float, attn_strength: float, attn_q_strength: float, attn_k_strength: float, attn_v_strength: float, attn_out_weight_strength: float, attn_out_bias_strength: float, other_strength: float, motion_pe_stretch: int, cap_initial_pe_length: int, interpolate_pe_to_length: int, initial_pe_idx_offset: int, final_pe_idx_offset: int, mask_motion_scale: torch.Tensor=None, min_motion_scale: float=1.0, max_motion_scale: float=1.0):
        adjust_pe = AdjustGroup(AdjustPE(motion_pe_stretch=motion_pe_stretch, cap_initial_pe_length=cap_initial_pe_length, interpolate_pe_to_length=interpolate_pe_to_length, initial_pe_idx_offset=initial_pe_idx_offset, final_pe_idx_offset=final_pe_idx_offset))
        adjust_weight = AdjustGroup(AdjustWeight(pe_MULT=pe_strength, attn_MULT=attn_strength, attn_q_MULT=attn_q_strength, attn_k_MULT=attn_k_strength, attn_v_MULT=attn_v_strength, attn_out_weight_MULT=attn_out_weight_strength, attn_out_bias_MULT=attn_out_bias_strength, other_MULT=other_strength))
        motion_model_settings = AnimateDiffSettings(adjust_pe=adjust_pe, adjust_weight=adjust_weight, mask_attn_scale=mask_motion_scale, mask_attn_scale_min=min_motion_scale, mask_attn_scale_max=max_motion_scale)
        return io.NodeOutput(motion_model_settings)
