from comfy_api.latest import io
import torch

import comfy.samplers

from .utils_model import BetaSchedules, SigmaSchedule, ModelSamplingType, ModelSamplingConfig, InterpolationMethod


def validate_sigma_schedule_compatibility(schedule_A: SigmaSchedule, schedule_B: SigmaSchedule,
                                          name_a: str="sigma_schedule_A", name_b: str="sigma_schedule_B"):
    if schedule_A.total_sigmas() != schedule_B.total_sigmas():
            raise Exception(f"Weighted Average cannot be taken of Sigma Schedules that do not have the same amount of sigmas; " +
                            f"{name_a} has {schedule_A.total_sigmas()} sigmas (lcm={schedule_A.is_lcm()}), " +
                            f"{name_b} has {schedule_B.total_sigmas()} sigmas (lcm={schedule_B.is_lcm()}).")


class SigmaScheduleNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_SigmaSchedule',
            display_name='Create Sigma Schedule 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/sigma schedule',
            inputs=[
                io.Combo.Input('beta_schedule', options=BetaSchedules.ALIAS_ACTIVE_LIST),
            ],
            outputs=[
                io.Custom("SIGMA_SCHEDULE").Output('SIGMA_SCHEDULE'),
            ],
        )


    @classmethod
    def execute(cls, beta_schedule: str) -> io.NodeOutput:
        model_type = ModelSamplingType.from_alias(ModelSamplingType.EPS)
        new_model_sampling = BetaSchedules._to_model_sampling(alias=beta_schedule,
                                                              model_type=model_type)
        return io.NodeOutput(SigmaSchedule(model_sampling=new_model_sampling, model_type=model_type))


class RawSigmaScheduleNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_RawSigmaSchedule',
            display_name='Create Raw Sigma Schedule 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/sigma schedule',
            inputs=[
                io.Combo.Input('raw_beta_schedule', options=BetaSchedules.RAW_BETA_SCHEDULE_LIST),
                io.Float.Input('linear_start', default=0.00085, max=1.0, min=0.0, step=1e-06),
                io.Float.Input('linear_end', default=0.012, max=1.0, min=0.0, step=1e-06),
                io.Combo.Input('sampling', options=ModelSamplingType._FULL_LIST),
                io.Int.Input('lcm_original_timesteps', default=50, max=1000, min=1),
                io.Boolean.Input('zsnr', default=False),
            ],
            outputs=[
                io.Custom("SIGMA_SCHEDULE").Output('SIGMA_SCHEDULE'),
            ],
        )


    @classmethod
    def execute(cls, raw_beta_schedule: str, linear_start: float, linear_end: float,# cosine_s: float,
                           sampling: str, lcm_original_timesteps: int, zsnr: bool, lcm_zsnr: bool=None) -> io.NodeOutput:
        if lcm_zsnr is not None:
            zsnr = lcm_zsnr
        # from pathlib import Path
        # log_name = 'enforce_zero_terminal_snr_betas'
        # betas_file = Path(__file__).parent.parent / rf"{log_name}.pt"
        # given_betas = torch.load(betas_file, weights_only=True)
        # given_betas[-1] = 0.0
        new_config = ModelSamplingConfig(beta_schedule=raw_beta_schedule, linear_start=linear_start, linear_end=linear_end)#, given_betas=given_betas)
        if sampling != ModelSamplingType.LCM:
            lcm_original_timesteps=None
        model_type = ModelSamplingType.from_alias(sampling)
        new_model_sampling = BetaSchedules._to_model_sampling(alias=BetaSchedules.AUTOSELECT, model_type=model_type, config_override=new_config, original_timesteps=lcm_original_timesteps)
        if zsnr:
            SigmaSchedule.apply_zsnr(new_model_sampling=new_model_sampling)
        return io.NodeOutput(SigmaSchedule(model_sampling=new_model_sampling, model_type=model_type))


class WeightedAverageSigmaScheduleNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_SigmaScheduleWeightedAverage',
            display_name='Sigma Schedule Weighted Mean 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/sigma schedule',
            inputs=[
                io.Custom("SIGMA_SCHEDULE").Input('schedule_A'),
                io.Custom("SIGMA_SCHEDULE").Input('schedule_B'),
                io.Float.Input('weight_A', default=0.5, max=1.0, min=0.0, step=0.001),
            ],
            outputs=[
                io.Custom("SIGMA_SCHEDULE").Output('SIGMA_SCHEDULE'),
            ],
        )


    @classmethod
    def execute(cls, schedule_A: SigmaSchedule, schedule_B: SigmaSchedule, weight_A: float) -> io.NodeOutput:
        validate_sigma_schedule_compatibility(schedule_A, schedule_B)
        new_sigmas = schedule_A.model_sampling.sigmas * weight_A + schedule_B.model_sampling.sigmas * (1-weight_A)
        combo_schedule = schedule_A.clone()
        combo_schedule.model_sampling.set_sigmas(new_sigmas)
        return io.NodeOutput(combo_schedule)


class InterpolatedWeightedAverageSigmaScheduleNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_SigmaScheduleWeightedAverageInterp',
            display_name='Sigma Schedule Interp. Mean 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/sigma schedule',
            inputs=[
                io.Custom("SIGMA_SCHEDULE").Input('schedule_A'),
                io.Custom("SIGMA_SCHEDULE").Input('schedule_B'),
                io.Float.Input('weight_A_Start', default=0.5, max=1.0, min=0.0, step=0.001),
                io.Float.Input('weight_A_End', default=0.5, max=1.0, min=0.0, step=0.001),
                io.Combo.Input('interpolation', options=InterpolationMethod._LIST),
            ],
            outputs=[
                io.Custom("SIGMA_SCHEDULE").Output('SIGMA_SCHEDULE'),
            ],
        )


    @classmethod
    def execute(cls, schedule_A: SigmaSchedule, schedule_B: SigmaSchedule,
                           weight_A_Start: float, weight_A_End: float, interpolation: str) -> io.NodeOutput:
        validate_sigma_schedule_compatibility(schedule_A, schedule_B)
        # get reverse weights, since sigmas are currently reversed
        weights = InterpolationMethod.get_weights(num_from=weight_A_Start, num_to=weight_A_End,
                                                  length=schedule_A.total_sigmas(), method=interpolation, reverse=True)
        weights = weights.to(schedule_A.model_sampling.sigmas.dtype).to(schedule_A.model_sampling.sigmas.device)
        new_sigmas = schedule_A.model_sampling.sigmas * weights + schedule_B.model_sampling.sigmas * (1.0-weights)
        combo_schedule = schedule_A.clone()
        combo_schedule.model_sampling.set_sigmas(new_sigmas)
        return io.NodeOutput(combo_schedule)


class SplitAndCombineSigmaScheduleNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_SigmaScheduleSplitAndCombine',
            display_name='Sigma Schedule Split Combine 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/sigma schedule',
            inputs=[
                io.Custom("SIGMA_SCHEDULE").Input('schedule_Start'),
                io.Custom("SIGMA_SCHEDULE").Input('schedule_End'),
                io.Float.Input('idx_split_percent', default=0.5, max=1.0, min=0.0, step=0.001),
            ],
            outputs=[
                io.Custom("SIGMA_SCHEDULE").Output('SIGMA_SCHEDULE'),
            ],
        )


    @classmethod
    def execute(cls, schedule_Start: SigmaSchedule, schedule_End: SigmaSchedule, idx_split_percent: float) -> io.NodeOutput:
        validate_sigma_schedule_compatibility(schedule_Start, schedule_End)
        # first, calculate index to act as split; get diff from 1.0 since sigmas are flipped at this stage
        idx = int((1.0-idx_split_percent) * schedule_Start.total_sigmas())
        new_sigmas = torch.cat([schedule_End.model_sampling.sigmas[:idx], schedule_Start.model_sampling.sigmas[idx:]], dim=0)
        new_schedule = schedule_Start.clone()
        new_schedule.model_sampling.set_sigmas(new_sigmas)
        return io.NodeOutput(new_schedule)


class SigmaScheduleToSigmasNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_SigmaScheduleToSigmas',
            display_name='Sigma Schedule To Sigmas 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/sample settings/sigma schedule',
            inputs=[
                io.Custom("SIGMA_SCHEDULE").Input('sigma_schedule'),
                io.Combo.Input('scheduler', options=comfy.samplers.SCHEDULER_NAMES),
                io.Int.Input('steps', default=20, max=10000, min=1),
                io.Float.Input('denoise', default=1.0, max=1.0, min=0.0, step=0.01),
            ],
            outputs=[
                io.Sigmas.Output('SIGMAS'),
            ],
        )


    @classmethod
    def execute(cls, sigma_schedule: SigmaSchedule, scheduler: str, steps: int, denoise: float) -> io.NodeOutput:
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return io.NodeOutput(torch.FloatTensor([]))
            total_steps = int(steps/denoise)

        sigmas = comfy.samplers.calculate_sigmas(sigma_schedule, scheduler, total_steps).cpu()
        sigmas = sigmas[-(steps + 1):]
        return io.NodeOutput(sigmas)
