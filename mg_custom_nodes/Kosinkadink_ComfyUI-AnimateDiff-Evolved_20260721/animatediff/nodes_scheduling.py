from typing import Union

from comfy_api.latest import io

from .scheduling import (evaluate_prompt_schedule, evaluate_value_schedule, extract_cond_from_schedule, TensorInterp, PromptOptions,
                         verify_key_value)
from .utils_model import BIGMAX
from .logger import logger

class PromptSchedulingLatentsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_PromptSchedulingLatents',
            display_name='Prompt Scheduling [Latents] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/scheduling',
            inputs=[io.String.Input('prompts', default='', multiline=True), io.Clip.Input('clip'), io.Latent.Input('latent'), io.String.Input('prepend_text', default='', force_input=True, multiline=True, optional=True), io.String.Input('append_text', default='', force_input=True, multiline=True, optional=True), io.Custom('VALUES_REPLACE').Input('values_replace', optional=True), io.Boolean.Input('print_schedule', default=False, optional=True), io.Combo.Input('tensor_interp', options=['lerp', 'slerp'] , optional=True)],
            outputs=[io.Conditioning.Output('CONDITIONING'), io.Latent.Output('LATENT')],
            description='Encode a schedule of prompts with automatic interpolation, its length matching passed-in latent count.'
        )
    NodeID = 'ADE_PromptSchedulingLatents'
    NodeName = 'Prompt Scheduling [Latents] 🎭🅐🅓'
    @classmethod
    def execute(cls, prompts: str, clip, latent: dict, print_schedule=False, tensor_interp=TensorInterp.LERP,
                        prepend_text='', append_text='', values_replace=None):
        options = PromptOptions(interp=tensor_interp, prepend_text=prepend_text, append_text=append_text,
                                values_replace=values_replace, print_schedule=print_schedule)
        conditioning = evaluate_prompt_schedule(prompts, latent["samples"].size(0), clip, options)
        return io.NodeOutput(conditioning, latent)


class PromptSchedulingNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_PromptScheduling',
            display_name='Prompt Scheduling 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/scheduling',
            inputs=[io.String.Input('prompts', default='', multiline=True), io.Clip.Input('clip'), io.String.Input('prepend_text', default='', force_input=True, multiline=True, optional=True), io.String.Input('append_text', default='', force_input=True, multiline=True, optional=True), io.Custom('VALUES_REPLACE').Input('values_replace', optional=True), io.Boolean.Input('print_schedule', default=False, optional=True), io.Int.Input('max_length', default=0, max=9007199254740991, min=0, step=1, optional=True), io.Combo.Input('tensor_interp', options=['lerp', 'slerp'] , optional=True)],
            outputs=[io.Conditioning.Output('CONDITIONING')],
            description='Encode a schedule of prompts with automatic interpolation.'
        )
    NodeID = 'ADE_PromptScheduling'
    NodeName = 'Prompt Scheduling 🎭🅐🅓'
    @classmethod
    def execute(cls, prompts: str, clip, print_schedule=False, max_length: int=0, tensor_interp=TensorInterp.LERP,
                        prepend_text='', append_text='', values_replace=None):
        options = PromptOptions(interp=tensor_interp, prepend_text=prepend_text, append_text=append_text,
                                values_replace=values_replace, print_schedule=print_schedule)
        conditioning = evaluate_prompt_schedule(prompts, max_length, clip, options)
        return io.NodeOutput(conditioning)


class ValueSchedulingLatentsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_ValueSchedulingLatents',
            display_name='Value Scheduling [Latents] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/scheduling',
            inputs=[io.String.Input('values', default='', multiline=True), io.Latent.Input('latent'), io.Boolean.Input('print_schedule', default=False, optional=True)],
            outputs=[io.Float.Output('FLOAT'), io.Custom('FLOATS').Output('FLOATS'), io.Int.Output('INT'), io.Custom('INTS').Output('INTS')],
            description='Create a list of values with automatic interpolation, its length matching passed-in latent count.'
        )
    NodeID = 'ADE_ValueSchedulingLatents'
    NodeName = 'Value Scheduling [Latents] 🎭🅐🅓'
    @classmethod
    def execute(cls, values: str, latent: dict, print_schedule=False):
        float_vals = evaluate_value_schedule(values, latent["samples"].size(0))
        int_vals = [round(x) for x in float_vals]
        if print_schedule:
            logger.info(f"ValueScheduling ({len(float_vals)} values):")
            for i, val in enumerate(float_vals):
                logger.info(f"{i} = {val}")
        return io.NodeOutput(float_vals, float_vals, int_vals, int_vals)


class ValueSchedulingNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_ValueScheduling',
            display_name='Value Scheduling 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/scheduling',
            inputs=[io.String.Input('values', default='', multiline=True), io.Boolean.Input('print_schedule', default=False, optional=True), io.Int.Input('max_length', default=0, max=9007199254740991, min=0, step=1, optional=True)],
            outputs=[io.Float.Output('FLOAT'), io.Custom('FLOATS').Output('FLOATS'), io.Int.Output('INT'), io.Custom('INTS').Output('INTS')],
            description='Create a list of values with automatic interpolation.'
        )
    NodeID = 'ADE_ValueScheduling'
    NodeName = 'Value Scheduling 🎭🅐🅓'
    @classmethod
    def execute(cls, values: str, max_length: int, print_schedule=False):
        float_vals = evaluate_value_schedule(values, max_length)
        int_vals = [round(x) for x in float_vals]
        if print_schedule:
            logger.info(f"ValueScheduling ({len(float_vals)} values):")
            for i, val in enumerate(float_vals):
                logger.info(f"{i} = {val}")
        return io.NodeOutput(float_vals, float_vals, int_vals, int_vals)


class AddValuesReplaceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_ValuesReplace',
            display_name='Add Values Replace 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/scheduling',
            inputs=[io.String.Input('value_key', default=''), io.Custom('FLOATS').Input('floats'), io.Custom('VALUES_REPLACE').Input('prev_replace', optional=True)],
            outputs=[io.Custom('VALUES_REPLACE').Output('VALUES_REPLACE')],
            description='Add a values schedule bound to a key to be used in Prompt Scheduling node.'
        )
    NodeID = 'ADE_ValuesReplace'
    NodeName = 'Add Values Replace 🎭🅐🅓'
    @classmethod
    def execute(cls, value_key: str, floats: Union[list[float]], prev_replace: dict=None):
        # key can only have a-z, A-Z, 0-9, and _ characters
        verify_key_value(key=value_key)
        # add/replace value floats
        if prev_replace is None:
            prev_replace = {}
        prev_replace = prev_replace.copy()
        if value_key in prev_replace:
            logger.warn(f"Value key '{value_key}' is already present - corresponding floats value will be overriden.")
        prev_replace[value_key] = floats
        return io.NodeOutput(prev_replace)


class FloatToFloatsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_FloatToFloats',
            display_name='Float to Floats 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/scheduling',
            inputs=[io.Float.Input('FLOAT', default=39, force_input=True)],
            outputs=[io.Custom('FLOATS').Output('FLOATS')]
        )
    NodeID = 'ADE_FloatToFloats'
    NodeName = 'Float to Floats 🎭🅐🅓'
    @classmethod
    def execute(cls, FLOAT: Union[float, list[float]]):
        floats = None
        if isinstance(FLOAT, float):
            floats = [float(FLOAT)]
        else:
            floats = list(FLOAT)
        return io.NodeOutput(floats)

class ConditionExtractionNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_ConditionExtraction',
            display_name='Condition Step Extraction 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/scheduling',
            inputs=[io.Conditioning.Input('conditioning'), io.Int.Input('index', default=0, min=0, step=1)],
            outputs=[io.Conditioning.Output('CONDITIONING')],
            description='Extract a single conditioning step from a schedule of prompts.'
        )
    NodeID = 'ADE_ConditionExtraction'
    NodeName = 'Condition Step Extraction 🎭🅐🅓'
    @classmethod
    def execute(cls, conditioning, index: int=0):
        conditioning_step = extract_cond_from_schedule(conditioning, index)
        return io.NodeOutput(conditioning_step)
