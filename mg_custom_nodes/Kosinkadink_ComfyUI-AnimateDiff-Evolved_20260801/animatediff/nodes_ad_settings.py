from comfy_api.latest import io
from .ad_settings import AdjustPE, AdjustWeight, AdjustGroup, AnimateDiffSettings
from .utils_model import BIGMAX


class AnimateDiffSettingsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_AnimateDiffSettings',
            display_name='AnimateDiff Settings 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/ad settings',
            inputs=[
                io.Custom("PE_ADJUST").Input('pe_adjust', optional=True),
                io.Custom("WEIGHT_ADJUST").Input('weight_adjust', optional=True),
            ],
            outputs=[
                io.Custom("AD_SETTINGS").Output('AD_SETTINGS'),
            ],
        )


    @classmethod
    def execute(cls, pe_adjust: AdjustGroup=None, weight_adjust: AdjustGroup=None) -> io.NodeOutput:
        return io.NodeOutput(AnimateDiffSettings(adjust_pe=pe_adjust, adjust_weight=weight_adjust))


class ManualAdjustPENode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_AdjustPEManual',
            display_name='Adjust PE [Manual] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/ad settings/pe adjust',
            inputs=[
                io.Int.Input('cap_initial_pe_length', default=0, min=0, step=1),
                io.Int.Input('interpolate_pe_to_length', default=0, min=0, step=1),
                io.Int.Input('initial_pe_idx_offset', default=0, min=0, step=1),
                io.Int.Input('final_pe_idx_offset', default=0, min=0, step=1),
                io.Boolean.Input('print_adjustment', default=False),
                io.Custom("PE_ADJUST").Input('prev_pe_adjust', optional=True),
            ],
            outputs=[
                io.Custom("PE_ADJUST").Output('PE_ADJUST'),
            ],
        )


    @classmethod
    def execute(cls, cap_initial_pe_length: int, interpolate_pe_to_length: int,
                      initial_pe_idx_offset: int, final_pe_idx_offset: int, print_adjustment: bool,
                      prev_pe_adjust: AdjustGroup=None) -> io.NodeOutput:
        if prev_pe_adjust is None:
            prev_pe_adjust = AdjustGroup()
        prev_pe_adjust = prev_pe_adjust.clone()
        adjust = AdjustPE(cap_initial_pe_length=cap_initial_pe_length, interpolate_pe_to_length=interpolate_pe_to_length,
                          initial_pe_idx_offset=initial_pe_idx_offset, final_pe_idx_offset=final_pe_idx_offset,
                          print_adjustment=print_adjustment)
        prev_pe_adjust.add(adjust)
        return io.NodeOutput(prev_pe_adjust)


class SweetspotStretchPENode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_AdjustPESweetspotStretch',
            display_name='Adjust PE [Sweetspot] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/ad settings/pe adjust',
            inputs=[
                io.Int.Input('sweetspot', default=16, max=9007199254740991, min=0),
                io.Int.Input('new_sweetspot', default=16, max=9007199254740991, min=0),
                io.Boolean.Input('print_adjustment', default=False),
                io.Custom("PE_ADJUST").Input('prev_pe_adjust', optional=True),
            ],
            outputs=[
                io.Custom("PE_ADJUST").Output('PE_ADJUST'),
            ],
        )


    @classmethod
    def execute(cls, sweetspot: int, new_sweetspot: int, print_adjustment: bool, prev_pe_adjust: AdjustGroup=None) -> io.NodeOutput:
        if prev_pe_adjust is None:
            prev_pe_adjust = AdjustGroup()
        prev_pe_adjust = prev_pe_adjust.clone()
        adjust = AdjustPE(cap_initial_pe_length=sweetspot, interpolate_pe_to_length=new_sweetspot,
                          print_adjustment=print_adjustment)
        prev_pe_adjust.add(adjust)
        return io.NodeOutput(prev_pe_adjust)


class FullStretchPENode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_AdjustPEFullStretch',
            display_name='Adjust PE [Full Stretch] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/ad settings/pe adjust',
            inputs=[
                io.Int.Input('pe_stretch', default=0, max=9007199254740991, min=0),
                io.Boolean.Input('print_adjustment', default=False),
                io.Custom("PE_ADJUST").Input('prev_pe_adjust', optional=True),
            ],
            outputs=[
                io.Custom("PE_ADJUST").Output('PE_ADJUST'),
            ],
        )


    @classmethod
    def execute(cls, pe_stretch: int, print_adjustment: bool, prev_pe_adjust: AdjustGroup=None) -> io.NodeOutput:
        if prev_pe_adjust is None:
            prev_pe_adjust = AdjustGroup()
        prev_pe_adjust = prev_pe_adjust.clone()
        adjust = AdjustPE(motion_pe_stretch=pe_stretch,
                          print_adjustment=print_adjustment)
        prev_pe_adjust.add(adjust)
        return io.NodeOutput(prev_pe_adjust)


class WeightAdjustAllAddNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_AdjustWeightAllAdd',
            display_name='Adjust Weight [All◆Add] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/ad settings/weight adjust',
            inputs=[
                io.Float.Input('all_ADD', default=0.0, max=2.0, min=-2.0, step=1e-06),
                io.Boolean.Input('print_adjustment', default=False),
                io.Custom("WEIGHT_ADJUST").Input('prev_weight_adjust', optional=True),
            ],
            outputs=[
                io.Custom("WEIGHT_ADJUST").Output('WEIGHT_ADJUST'),
            ],
        )


    @classmethod
    def execute(cls, all_ADD: float, print_adjustment: bool, prev_weight_adjust: AdjustGroup=None) -> io.NodeOutput:
        if prev_weight_adjust is None:
            prev_weight_adjust = AdjustGroup()
        prev_weight_adjust = prev_weight_adjust.clone()
        adjust = AdjustWeight(
            all_ADD=all_ADD,
            print_adjustment=print_adjustment
        )
        prev_weight_adjust.add(adjust)
        return io.NodeOutput(prev_weight_adjust)


class WeightAdjustAllMultNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_AdjustWeightAllMult',
            display_name='Adjust Weight [All◆Mult] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/ad settings/weight adjust',
            inputs=[
                io.Float.Input('all_MULT', default=1.0, max=2.0, min=0.0, step=1e-06),
                io.Boolean.Input('print_adjustment', default=False),
                io.Custom("WEIGHT_ADJUST").Input('prev_weight_adjust', optional=True),
            ],
            outputs=[
                io.Custom("WEIGHT_ADJUST").Output('WEIGHT_ADJUST'),
            ],
        )


    @classmethod
    def execute(cls, all_MULT: float, print_adjustment: bool, prev_weight_adjust: AdjustGroup=None) -> io.NodeOutput:
        if prev_weight_adjust is None:
            prev_weight_adjust = AdjustGroup()
        prev_weight_adjust = prev_weight_adjust.clone()
        adjust = AdjustWeight(
            all_MULT=all_MULT,
            print_adjustment=print_adjustment
        )
        prev_weight_adjust.add(adjust)
        return io.NodeOutput(prev_weight_adjust)


class WeightAdjustIndivAddNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_AdjustWeightIndivAdd',
            display_name='Adjust Weight [Indiv◆Add] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/ad settings/weight adjust',
            inputs=[
                io.Float.Input('pe_ADD', default=0.0, max=2.0, min=-2.0, step=1e-06),
                io.Float.Input('attn_ADD', default=0.0, max=2.0, min=-2.0, step=1e-06),
                io.Float.Input('other_ADD', default=0.0, max=2.0, min=-2.0, step=1e-06),
                io.Boolean.Input('print_adjustment', default=False),
                io.Custom("WEIGHT_ADJUST").Input('prev_weight_adjust', optional=True),
            ],
            outputs=[
                io.Custom("WEIGHT_ADJUST").Output('WEIGHT_ADJUST'),
            ],
        )


    @classmethod
    def execute(cls, pe_ADD: float, attn_ADD: float, other_ADD: float, print_adjustment: bool, prev_weight_adjust: AdjustGroup=None) -> io.NodeOutput:
        if prev_weight_adjust is None:
            prev_weight_adjust = AdjustGroup()
        prev_weight_adjust = prev_weight_adjust.clone()
        adjust = AdjustWeight(
            pe_ADD=pe_ADD,
            attn_ADD=attn_ADD,
            other_ADD=other_ADD,
            print_adjustment=print_adjustment
        )
        prev_weight_adjust.add(adjust)
        return io.NodeOutput(prev_weight_adjust)


class WeightAdjustIndivMultNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_AdjustWeightIndivMult',
            display_name='Adjust Weight [Indiv◆Mult] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/ad settings/weight adjust',
            inputs=[
                io.Float.Input('pe_MULT', default=1.0, max=2.0, min=0.0, step=1e-06),
                io.Float.Input('attn_MULT', default=1.0, max=2.0, min=0.0, step=1e-06),
                io.Float.Input('other_MULT', default=1.0, max=2.0, min=0.0, step=1e-06),
                io.Boolean.Input('print_adjustment', default=False),
                io.Custom("WEIGHT_ADJUST").Input('prev_weight_adjust', optional=True),
            ],
            outputs=[
                io.Custom("WEIGHT_ADJUST").Output('WEIGHT_ADJUST'),
            ],
        )


    @classmethod
    def execute(cls, pe_MULT: float, attn_MULT: float, other_MULT: float, print_adjustment: bool, prev_weight_adjust: AdjustGroup=None) -> io.NodeOutput:
        if prev_weight_adjust is None:
            prev_weight_adjust = AdjustGroup()
        prev_weight_adjust = prev_weight_adjust.clone()
        adjust = AdjustWeight(
            pe_MULT=pe_MULT,
            attn_MULT=attn_MULT,
            other_MULT=other_MULT,
            print_adjustment=print_adjustment
        )
        prev_weight_adjust.add(adjust)
        return io.NodeOutput(prev_weight_adjust)


class WeightAdjustIndivAttnAddNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_AdjustWeightIndivAttnAdd',
            display_name='Adjust Weight [Indiv-Attn◆Add] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/ad settings/weight adjust',
            inputs=[
                io.Float.Input('pe_ADD', default=0.0, max=2.0, min=-2.0, step=1e-06),
                io.Float.Input('attn_ADD', default=0.0, max=2.0, min=-2.0, step=1e-06),
                io.Float.Input('attn_q_ADD', default=0.0, max=2.0, min=-2.0, step=1e-06),
                io.Float.Input('attn_k_ADD', default=0.0, max=2.0, min=-2.0, step=1e-06),
                io.Float.Input('attn_v_ADD', default=0.0, max=2.0, min=-2.0, step=1e-06),
                io.Float.Input('attn_out_weight_ADD', default=0.0, max=2.0, min=-2.0, step=1e-06),
                io.Float.Input('attn_out_bias_ADD', default=0.0, max=2.0, min=-2.0, step=1e-06),
                io.Float.Input('other_ADD', default=0.0, max=2.0, min=-2.0, step=1e-06),
                io.Boolean.Input('print_adjustment', default=False),
                io.Custom("WEIGHT_ADJUST").Input('prev_weight_adjust', optional=True),
            ],
            outputs=[
                io.Custom("WEIGHT_ADJUST").Output('WEIGHT_ADJUST'),
            ],
        )


    @classmethod
    def execute(cls, pe_ADD: float, attn_ADD: float,
                          attn_q_ADD: float, attn_k_ADD: float, attn_v_ADD: float,
                          attn_out_weight_ADD: float, attn_out_bias_ADD: float,
                          other_ADD: float, print_adjustment: bool, prev_weight_adjust: AdjustGroup=None) -> io.NodeOutput:
        if prev_weight_adjust is None:
            prev_weight_adjust = AdjustGroup()
        prev_weight_adjust = prev_weight_adjust.clone()
        adjust = AdjustWeight(
            pe_ADD=pe_ADD,
            attn_ADD=attn_ADD,
            attn_q_ADD=attn_q_ADD,
            attn_k_ADD=attn_k_ADD,
            attn_v_ADD=attn_v_ADD,
            attn_out_weight_ADD=attn_out_weight_ADD,
            attn_out_bias_ADD=attn_out_bias_ADD,
            other_ADD=other_ADD,
            print_adjustment=print_adjustment
        )
        prev_weight_adjust.add(adjust)
        return io.NodeOutput(prev_weight_adjust)


class WeightAdjustIndivAttnMultNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_AdjustWeightIndivAttnMult',
            display_name='Adjust Weight [Indiv-Attn◆Mult] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/ad settings/weight adjust',
            inputs=[
                io.Float.Input('pe_MULT', default=1.0, max=2.0, min=0.0, step=1e-06),
                io.Float.Input('attn_MULT', default=1.0, max=2.0, min=0.0, step=1e-06),
                io.Float.Input('attn_q_MULT', default=1.0, max=2.0, min=0.0, step=1e-06),
                io.Float.Input('attn_k_MULT', default=1.0, max=2.0, min=0.0, step=1e-06),
                io.Float.Input('attn_v_MULT', default=1.0, max=2.0, min=0.0, step=1e-06),
                io.Float.Input('attn_out_weight_MULT', default=1.0, max=2.0, min=0.0, step=1e-06),
                io.Float.Input('attn_out_bias_MULT', default=1.0, max=2.0, min=0.0, step=1e-06),
                io.Float.Input('other_MULT', default=1.0, max=2.0, min=0.0, step=1e-06),
                io.Boolean.Input('print_adjustment', default=False),
                io.Custom("WEIGHT_ADJUST").Input('prev_weight_adjust', optional=True),
            ],
            outputs=[
                io.Custom("WEIGHT_ADJUST").Output('WEIGHT_ADJUST'),
            ],
        )


    @classmethod
    def execute(cls, pe_MULT: float, attn_MULT: float,
                          attn_q_MULT: float, attn_k_MULT: float, attn_v_MULT: float,
                          attn_out_weight_MULT: float, attn_out_bias_MULT: float,
                          other_MULT: float, print_adjustment: bool, prev_weight_adjust: AdjustGroup=None) -> io.NodeOutput:
        if prev_weight_adjust is None:
            prev_weight_adjust = AdjustGroup()
        prev_weight_adjust = prev_weight_adjust.clone()
        adjust = AdjustWeight(
            pe_MULT=pe_MULT,
            attn_MULT=attn_MULT,
            attn_q_MULT=attn_q_MULT,
            attn_k_MULT=attn_k_MULT,
            attn_v_MULT=attn_v_MULT,
            attn_out_weight_MULT=attn_out_weight_MULT,
            attn_out_bias_MULT=attn_out_bias_MULT,
            other_MULT=other_MULT,
            print_adjustment=print_adjustment
        )
        prev_weight_adjust.add(adjust)
        return io.NodeOutput(prev_weight_adjust)
