from collections.abc import Iterable
from typing import Union

import torch
from torch import Tensor

from comfy_api.latest import io

from .utils_motion import create_multival_combo, linear_conversion, normalize_min_max, extend_to_batch_size, extend_list_to_batch_size


class ScaleType:
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    LIST = [ABSOLUTE, RELATIVE]


class MultivalDynamicNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_MultivalDynamic',
            display_name='Multival 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/multival',
            inputs=[io.Float.Input('float_val', default=1.0, min=0.0, step=0.001), io.Mask.Input('mask_optional', optional=True)],
            outputs=[io.Custom('MULTIVAL').Output('MULTIVAL')]
        )
    @classmethod
    def execute(cls, float_val: Union[float, list[float]]=1.0, mask_optional: Tensor=None):
        return io.NodeOutput(create_multival_combo(float_val=float_val, mask_optional=mask_optional))


class MultivalScaledMaskNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_MultivalScaledMask',
            display_name='Multival Scaled Mask 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/multival',
            inputs=[io.Float.Input('min_float_val', default=0.0, min=0.0, step=0.001), io.Float.Input('max_float_val', default=1.0, min=0.0, step=0.001), io.Mask.Input('mask'), io.Combo.Input('scaling', options=['absolute', 'relative'] , optional=True)],
            outputs=[io.Custom('MULTIVAL').Output('MULTIVAL')]
        )
    @classmethod
    def execute(cls, min_float_val: float, max_float_val: float, mask: Tensor, scaling: str=ScaleType.ABSOLUTE):
        lengths = [mask.shape[0]]
        iterable_inputs = [False, False]
        val_inputs = [min_float_val, max_float_val]
        if isinstance(min_float_val, Iterable):
            iterable_inputs[0] = True
            val_inputs[0] = list(min_float_val)
            lengths.append(len(min_float_val))
        if isinstance(max_float_val, Iterable):
            iterable_inputs[1] = True
            val_inputs[1] = list(max_float_val)
            lengths.append(len(max_float_val))
        # make sure mask and any iterable float_vals match max length
        max_length = max(lengths)
        mask = extend_to_batch_size(mask, max_length)
        for i in range(len(iterable_inputs)):
            if iterable_inputs[i] == True:
                # make sure tensors will match dimensions of mask
                val_inputs[i] = torch.tensor(extend_list_to_batch_size(val_inputs[i], max_length)).unsqueeze(-1).unsqueeze(-1)
        min_float_val, max_float_val = val_inputs
        if scaling == ScaleType.ABSOLUTE:
            mask = linear_conversion(mask.clone(), new_min=min_float_val, new_max=max_float_val)
        elif scaling == ScaleType.RELATIVE:
            mask = normalize_min_max(mask.clone(), new_min=min_float_val, new_max=max_float_val)
        else:
            raise ValueError(f"scaling '{scaling}' not recognized.")
        return io.NodeOutput(*MultivalDynamicNode.execute(mask_optional=mask).args)


class MultivalDynamicFloatInputNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_MultivalDynamicFloatInput',
            display_name='Multival [Float List] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/multival',
            inputs=[io.Float.Input('float_val', default=1.0, force_input=True, max=10.0, min=0.0, step=0.001), io.Mask.Input('mask_optional', optional=True)],
            outputs=[io.Custom('MULTIVAL').Output('MULTIVAL')]
        )
    @classmethod
    def execute(cls, float_val: Union[float, list[float]]=None, mask_optional: Tensor=None):
        return io.NodeOutput(*MultivalDynamicNode.execute(float_val=float_val, mask_optional=mask_optional).args)


class MultivalDynamicFloatsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_MultivalDynamicFloats',
            display_name='Multival [Floats] 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/multival',
            inputs=[io.Custom('FLOATS').Input('floats', extra_dict={'default': 1.0, 'min': 0.0, 'max': 10.0, 'step': 0.001}), io.Mask.Input('mask_optional', optional=True)],
            outputs=[io.Custom('MULTIVAL').Output('MULTIVAL')]
        )
    @classmethod
    def execute(cls, floats: Union[float, list[float]]=None, mask_optional: Tensor=None):
        return io.NodeOutput(*MultivalDynamicNode.execute(float_val=floats, mask_optional=mask_optional).args)


class MultivalConvertToMaskNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_MultivalConvertToMask',
            display_name='Multival to Mask 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/multival',
            inputs=[io.Custom('MULTIVAL').Input('multival')],
            outputs=[io.Mask.Output('MASK')]
        )
    @classmethod
    def execute(cls, multival: Union[float, Tensor]):
        # if already tensor, assume is a valid mask
        if type(multival) == Tensor:
            return io.NodeOutput(multival)
        # otherwise, make a single 1x1 mask with the proper value
        shape = (1,1,1)
        converted_multival = torch.ones(shape) * multival
        return io.NodeOutput(converted_multival)
