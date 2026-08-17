from typing import Union
from torch import Tensor

from comfy_api.latest import io

from .motion_module_ad import BlockType
from .utils_model import ModelTypeSD
from .utils_motion import AllPerBlocks, PerBlock, PerBlockId, extend_list_to_batch_size


class ADBlockHolder:
    def __init__(self, effect: Union[float, Tensor, None]=None,
                 scales: Union[list[float, Tensor], None]=list()):
        self.effect = effect
        self.scales = scales

    def has_effect(self):
        return self.effect is not None

    def has_scale(self):
        for scale in self.scales:
            if scale is not None:
                return True
        return False

    def is_empty(self):
        has_anything = self.has_effect() or self.has_scale()
        return not has_anything


class ADBlockComboNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_ADBlockCombo',
            display_name='AD Block 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/per block',
            inputs=[io.Custom('MULTIVAL').Input('effect', optional=True), io.Custom('MULTIVAL').Input('scale', optional=True)],
            outputs=[io.Custom('AD_BLOCK').Output('AD_BLOCK')]
        )
    NodeID = 'ADE_ADBlockCombo'
    NodeName = 'AD Block 🎭🅐🅓'
    @classmethod
    def execute(cls, effect: Union[float, Tensor, None]=None, scale: Union[float, Tensor, None]=None):
        scales = [scale, scale]
        block = ADBlockHolder(effect=effect, scales=scales)
        if block.is_empty():
            block = None
        return io.NodeOutput(block)


class ADBlockIndivNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_ADBlockIndiv',
            display_name='AD Block+ 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/per block',
            inputs=[io.Custom('MULTIVAL').Input('effect', optional=True), io.Custom('MULTIVAL').Input('scale_0', optional=True), io.Custom('MULTIVAL').Input('scale_1', optional=True)],
            outputs=[io.Custom('AD_BLOCK').Output('AD_BLOCK')]
        )
    NodeID = 'ADE_ADBlockIndiv'
    NodeName = 'AD Block+ 🎭🅐🅓'
    @classmethod
    def execute(cls, effect: Union[float, Tensor, None]=None,
                      scale_0: Union[float, Tensor, None]=None, scale_1: Union[float, Tensor, None]=None):
        scales = [scale_0, scale_1]
        block = ADBlockHolder(effect=effect, scales=scales)
        if block.is_empty():
            block = None
        return io.NodeOutput(block)


class PerBlockHighLevelNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_PerBlockHighLevel',
            display_name='AD Per Block 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/per block',
            inputs=[io.Custom('AD_BLOCK').Input('down', optional=True), io.Custom('AD_BLOCK').Input('mid', optional=True), io.Custom('AD_BLOCK').Input('up', optional=True)],
            outputs=[io.Custom('PER_BLOCK').Output('PER_BLOCK')]
        )
    NodeID = 'ADE_PerBlockHighLevel'
    NodeName = 'AD Per Block 🎭🅐🅓'
    @classmethod
    def execute(cls,
                         down: Union[ADBlockHolder, None]=None,
                         mid: Union[ADBlockHolder, None]=None,
                         up: Union[ADBlockHolder, None]=None):
        blocks = []
        d = {
            PerBlockId(block_type=BlockType.DOWN): down,
            PerBlockId(block_type=BlockType.MID): mid,
            PerBlockId(block_type=BlockType.UP): up,
        }
        for id, block in d.items():
            if block is not None:
                blocks.append(PerBlock(id=id, effect=block.effect, scales=block.scales))
        if len(blocks) == 0:
            blocks = None
        return io.NodeOutput(AllPerBlocks(blocks))


class PerBlock_SD15_MidLevelNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_PerBlock_SD15_MidLevel',
            display_name='AD Per Block+ (SD1.5) 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/per block',
            inputs=[io.Custom('AD_BLOCK').Input('down_0', optional=True), io.Custom('AD_BLOCK').Input('down_1', optional=True), io.Custom('AD_BLOCK').Input('down_2', optional=True), io.Custom('AD_BLOCK').Input('down_3', optional=True), io.Custom('AD_BLOCK').Input('mid', optional=True), io.Custom('AD_BLOCK').Input('up_0', optional=True), io.Custom('AD_BLOCK').Input('up_1', optional=True), io.Custom('AD_BLOCK').Input('up_2', optional=True), io.Custom('AD_BLOCK').Input('up_3', optional=True)],
            outputs=[io.Custom('PER_BLOCK').Output('PER_BLOCK')]
        )
    NodeID = 'ADE_PerBlock_SD15_MidLevel'
    NodeName = 'AD Per Block+ (SD1.5) 🎭🅐🅓'
    @classmethod
    def execute(cls,
                         down_0: Union[ADBlockHolder, None]=None,
                         down_1: Union[ADBlockHolder, None]=None,
                         down_2: Union[ADBlockHolder, None]=None,
                         down_3: Union[ADBlockHolder, None]=None,
                         mid: Union[ADBlockHolder, None]=None,
                         up_0: Union[ADBlockHolder, None]=None,
                         up_1: Union[ADBlockHolder, None]=None,
                         up_2: Union[ADBlockHolder, None]=None,
                         up_3: Union[ADBlockHolder, None]=None):
        blocks = []
        d = {
            PerBlockId(block_type=BlockType.DOWN, block_idx=0): down_0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=1): down_1,
            PerBlockId(block_type=BlockType.DOWN, block_idx=2): down_2,
            PerBlockId(block_type=BlockType.DOWN, block_idx=3): down_3,
            PerBlockId(block_type=BlockType.MID): mid,
            PerBlockId(block_type=BlockType.UP, block_idx=0): up_0,
            PerBlockId(block_type=BlockType.UP, block_idx=1): up_1,
            PerBlockId(block_type=BlockType.UP, block_idx=2): up_2,
            PerBlockId(block_type=BlockType.UP, block_idx=3): up_3,
        }
        for id, block in d.items():
            if block is not None:
                blocks.append(PerBlock(id=id, effect=block.effect, scales=block.scales))
        if len(blocks) == 0:
            blocks = None
        return io.NodeOutput(AllPerBlocks(blocks, ModelTypeSD.SD1_5))


class PerBlock_SD15_LowLevelNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_PerBlock_SD15_LowLevel',
            display_name='AD Per Block++ (SD1.5) 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/per block',
            inputs=[io.Custom('AD_BLOCK').Input('down_0__0', optional=True), io.Custom('AD_BLOCK').Input('down_0__1', optional=True), io.Custom('AD_BLOCK').Input('down_1__0', optional=True), io.Custom('AD_BLOCK').Input('down_1__1', optional=True), io.Custom('AD_BLOCK').Input('down_2__0', optional=True), io.Custom('AD_BLOCK').Input('down_2__1', optional=True), io.Custom('AD_BLOCK').Input('down_3__0', optional=True), io.Custom('AD_BLOCK').Input('down_3__1', optional=True), io.Custom('AD_BLOCK').Input('mid', optional=True), io.Custom('AD_BLOCK').Input('up_0__0', optional=True), io.Custom('AD_BLOCK').Input('up_0__1', optional=True), io.Custom('AD_BLOCK').Input('up_0__2', optional=True), io.Custom('AD_BLOCK').Input('up_1__0', optional=True), io.Custom('AD_BLOCK').Input('up_1__1', optional=True), io.Custom('AD_BLOCK').Input('up_1__2', optional=True), io.Custom('AD_BLOCK').Input('up_2__0', optional=True), io.Custom('AD_BLOCK').Input('up_2__1', optional=True), io.Custom('AD_BLOCK').Input('up_2__2', optional=True), io.Custom('AD_BLOCK').Input('up_3__0', optional=True), io.Custom('AD_BLOCK').Input('up_3__1', optional=True), io.Custom('AD_BLOCK').Input('up_3__2', optional=True)],
            outputs=[io.Custom('PER_BLOCK').Output('PER_BLOCK')]
        )
    NodeID = 'ADE_PerBlock_SD15_LowLevel'
    NodeName = 'AD Per Block++ (SD1.5) 🎭🅐🅓'
    @classmethod
    def execute(cls,
                         down_0__0: Union[ADBlockHolder, None]=None,
                         down_0__1: Union[ADBlockHolder, None]=None,
                         down_1__0: Union[ADBlockHolder, None]=None,
                         down_1__1: Union[ADBlockHolder, None]=None,
                         down_2__0: Union[ADBlockHolder, None]=None,
                         down_2__1: Union[ADBlockHolder, None]=None,
                         down_3__0: Union[ADBlockHolder, None]=None,
                         down_3__1: Union[ADBlockHolder, None]=None,
                         mid: Union[ADBlockHolder, None]=None,
                         up_0__0: Union[ADBlockHolder, None]=None,
                         up_0__1: Union[ADBlockHolder, None]=None,
                         up_0__2: Union[ADBlockHolder, None]=None,
                         up_1__0: Union[ADBlockHolder, None]=None,
                         up_1__1: Union[ADBlockHolder, None]=None,
                         up_1__2: Union[ADBlockHolder, None]=None,
                         up_2__0: Union[ADBlockHolder, None]=None,
                         up_2__1: Union[ADBlockHolder, None]=None,
                         up_2__2: Union[ADBlockHolder, None]=None,
                         up_3__0: Union[ADBlockHolder, None]=None,
                         up_3__1: Union[ADBlockHolder, None]=None,
                         up_3__2: Union[ADBlockHolder, None]=None):
        blocks = []
        d = {
            PerBlockId(block_type=BlockType.DOWN, block_idx=0, module_idx=0): down_0__0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=0, module_idx=1): down_0__1,
            PerBlockId(block_type=BlockType.DOWN, block_idx=1, module_idx=0): down_1__0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=1, module_idx=1): down_1__1,
            PerBlockId(block_type=BlockType.DOWN, block_idx=2, module_idx=0): down_2__0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=2, module_idx=1): down_2__1,
            PerBlockId(block_type=BlockType.DOWN, block_idx=3, module_idx=0): down_3__0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=3, module_idx=1): down_3__1,
            PerBlockId(block_type=BlockType.MID): mid,
            PerBlockId(block_type=BlockType.UP, block_idx=0, module_idx=0): up_0__0,
            PerBlockId(block_type=BlockType.UP, block_idx=0, module_idx=1): up_0__1,
            PerBlockId(block_type=BlockType.UP, block_idx=0, module_idx=2): up_0__2,
            PerBlockId(block_type=BlockType.UP, block_idx=1, module_idx=0): up_1__0,
            PerBlockId(block_type=BlockType.UP, block_idx=1, module_idx=1): up_1__1,
            PerBlockId(block_type=BlockType.UP, block_idx=1, module_idx=2): up_1__2,
            PerBlockId(block_type=BlockType.UP, block_idx=2, module_idx=0): up_2__0,
            PerBlockId(block_type=BlockType.UP, block_idx=2, module_idx=1): up_2__1,
            PerBlockId(block_type=BlockType.UP, block_idx=2, module_idx=2): up_2__2,
            PerBlockId(block_type=BlockType.UP, block_idx=3, module_idx=0): up_3__0,
            PerBlockId(block_type=BlockType.UP, block_idx=3, module_idx=1): up_3__1,
            PerBlockId(block_type=BlockType.UP, block_idx=3, module_idx=2): up_3__2,
        }
        for id, block in d.items():
            if block is not None:
                blocks.append(PerBlock(id=id, effect=block.effect, scales=block.scales))
        if len(blocks) == 0:
            blocks = None
        return io.NodeOutput(AllPerBlocks(blocks, ModelTypeSD.SD1_5))


class PerBlock_SD15_FromFloatsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_PerBlock_SD15_FromFloats',
            display_name='AD Per Block Floats (SD1.5) 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/per block',
            inputs=[io.Custom('FLOATS').Input('effect_21_floats', optional=True), io.Custom('FLOATS').Input('scale_21_floats', optional=True)],
            outputs=[io.Custom('PER_BLOCK').Output('PER_BLOCK')],
            description='Use Floats from Value Schedules to select SD1.5 effect/scale values for blocks.'
        )
    NodeID = 'ADE_PerBlock_SD15_FromFloats'
    NodeName = 'AD Per Block Floats (SD1.5) 🎭🅐🅓'
    @classmethod
    def execute(cls,
                         effect_21_floats: Union[list[float], None]=None,
                         scale_21_floats: Union[list[float], None]=None):
        if effect_21_floats is None and scale_21_floats is None:
            return io.NodeOutput(AllPerBlocks(None, ModelTypeSD.SD1_5))
        # SD1.5 has 21 blocks
        block_total = 21
        holders = [ADBlockHolder() for _ in range(block_total)]
        if effect_21_floats is not None:
            effect_21_floats = extend_list_to_batch_size(effect_21_floats, block_total)
            for effect, holder in zip(effect_21_floats, holders):
                holder.effect = effect
        if scale_21_floats is not None:
            scale_21_floats = extend_list_to_batch_size(scale_21_floats, block_total)
            for scale, holder in zip(scale_21_floats, holders):
                holder.scales = [scale, scale]
        return PerBlock_SD15_LowLevelNode.execute(*holders)


class PerBlock_SDXL_MidLevelNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_PerBlock_SDXL_MidLevel',
            display_name='AD Per Block+ (SDXL) 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/per block',
            inputs=[io.Custom('AD_BLOCK').Input('down_0', optional=True), io.Custom('AD_BLOCK').Input('down_1', optional=True), io.Custom('AD_BLOCK').Input('down_2', optional=True), io.Custom('AD_BLOCK').Input('mid', optional=True), io.Custom('AD_BLOCK').Input('up_0', optional=True), io.Custom('AD_BLOCK').Input('up_1', optional=True), io.Custom('AD_BLOCK').Input('up_2', optional=True)],
            outputs=[io.Custom('PER_BLOCK').Output('PER_BLOCK')]
        )
    NodeID = 'ADE_PerBlock_SDXL_MidLevel'
    NodeName = 'AD Per Block+ (SDXL) 🎭🅐🅓'
    @classmethod
    def execute(cls,
                         down_0: Union[ADBlockHolder, None]=None,
                         down_1: Union[ADBlockHolder, None]=None,
                         down_2: Union[ADBlockHolder, None]=None,
                         mid: Union[ADBlockHolder, None]=None,
                         up_0: Union[ADBlockHolder, None]=None,
                         up_1: Union[ADBlockHolder, None]=None,
                         up_2: Union[ADBlockHolder, None]=None):
        blocks = []
        d = {
            PerBlockId(block_type=BlockType.DOWN, block_idx=0): down_0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=1): down_1,
            PerBlockId(block_type=BlockType.DOWN, block_idx=2): down_2,
            PerBlockId(block_type=BlockType.MID): mid,
            PerBlockId(block_type=BlockType.UP, block_idx=0): up_0,
            PerBlockId(block_type=BlockType.UP, block_idx=1): up_1,
            PerBlockId(block_type=BlockType.UP, block_idx=2): up_2,
        }
        for id, block in d.items():
            if block is not None:
                blocks.append(PerBlock(id=id, effect=block.effect, scales=block.scales))
        if len(blocks) == 0:
            blocks = None
        return io.NodeOutput(AllPerBlocks(blocks, ModelTypeSD.SDXL))


class PerBlock_SDXL_LowLevelNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_PerBlock_SDXL_LowLevel',
            display_name='AD Per Block++ (SDXL) 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/per block',
            inputs=[io.Custom('AD_BLOCK').Input('down_0__0', optional=True), io.Custom('AD_BLOCK').Input('down_0__1', optional=True), io.Custom('AD_BLOCK').Input('down_1__0', optional=True), io.Custom('AD_BLOCK').Input('down_1__1', optional=True), io.Custom('AD_BLOCK').Input('down_2__0', optional=True), io.Custom('AD_BLOCK').Input('down_2__1', optional=True), io.Custom('AD_BLOCK').Input('mid', optional=True), io.Custom('AD_BLOCK').Input('up_0__0', optional=True), io.Custom('AD_BLOCK').Input('up_0__1', optional=True), io.Custom('AD_BLOCK').Input('up_0__2', optional=True), io.Custom('AD_BLOCK').Input('up_1__0', optional=True), io.Custom('AD_BLOCK').Input('up_1__1', optional=True), io.Custom('AD_BLOCK').Input('up_1__2', optional=True), io.Custom('AD_BLOCK').Input('up_2__0', optional=True), io.Custom('AD_BLOCK').Input('up_2__1', optional=True), io.Custom('AD_BLOCK').Input('up_2__2', optional=True)],
            outputs=[io.Custom('PER_BLOCK').Output('PER_BLOCK')]
        )
    NodeID = 'ADE_PerBlock_SDXL_LowLevel'
    NodeName = 'AD Per Block++ (SDXL) 🎭🅐🅓'
    @classmethod
    def execute(cls,
                         down_0__0: Union[ADBlockHolder, None]=None,
                         down_0__1: Union[ADBlockHolder, None]=None,
                         down_1__0: Union[ADBlockHolder, None]=None,
                         down_1__1: Union[ADBlockHolder, None]=None,
                         down_2__0: Union[ADBlockHolder, None]=None,
                         down_2__1: Union[ADBlockHolder, None]=None,
                         mid: Union[ADBlockHolder, None]=None,
                         up_0__0: Union[ADBlockHolder, None]=None,
                         up_0__1: Union[ADBlockHolder, None]=None,
                         up_0__2: Union[ADBlockHolder, None]=None,
                         up_1__0: Union[ADBlockHolder, None]=None,
                         up_1__1: Union[ADBlockHolder, None]=None,
                         up_1__2: Union[ADBlockHolder, None]=None,
                         up_2__0: Union[ADBlockHolder, None]=None,
                         up_2__1: Union[ADBlockHolder, None]=None,
                         up_2__2: Union[ADBlockHolder, None]=None,):
        blocks = []
        d = {
            PerBlockId(block_type=BlockType.DOWN, block_idx=0, module_idx=0): down_0__0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=0, module_idx=1): down_0__1,
            PerBlockId(block_type=BlockType.DOWN, block_idx=1, module_idx=0): down_1__0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=1, module_idx=1): down_1__1,
            PerBlockId(block_type=BlockType.DOWN, block_idx=2, module_idx=0): down_2__0,
            PerBlockId(block_type=BlockType.DOWN, block_idx=2, module_idx=1): down_2__1,
            PerBlockId(block_type=BlockType.MID): mid,
            PerBlockId(block_type=BlockType.UP, block_idx=0, module_idx=0): up_0__0,
            PerBlockId(block_type=BlockType.UP, block_idx=0, module_idx=1): up_0__1,
            PerBlockId(block_type=BlockType.UP, block_idx=0, module_idx=2): up_0__2,
            PerBlockId(block_type=BlockType.UP, block_idx=1, module_idx=0): up_1__0,
            PerBlockId(block_type=BlockType.UP, block_idx=1, module_idx=1): up_1__1,
            PerBlockId(block_type=BlockType.UP, block_idx=1, module_idx=2): up_1__2,
            PerBlockId(block_type=BlockType.UP, block_idx=2, module_idx=0): up_2__0,
            PerBlockId(block_type=BlockType.UP, block_idx=2, module_idx=1): up_2__1,
            PerBlockId(block_type=BlockType.UP, block_idx=2, module_idx=2): up_2__2,
        }
        for id, block in d.items():
            if block is not None:
                blocks.append(PerBlock(id=id, effect=block.effect, scales=block.scales))
        if len(blocks) == 0:
            blocks = None
        return io.NodeOutput(AllPerBlocks(blocks, ModelTypeSD.SDXL))


class PerBlock_SDXL_FromFloatsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id='ADE_PerBlock_SDXL_FromFloats',
            display_name='AD Per Block Floats (SDXL) 🎭🅐🅓',
            category='Animate Diff 🎭🅐🅓/per block',
            inputs=[io.Custom('FLOATS').Input('effect_16_floats', optional=True), io.Custom('FLOATS').Input('scale_16_floats', optional=True)],
            outputs=[io.Custom('PER_BLOCK').Output('PER_BLOCK')],
            description='Use Floats from Value Schedules to select SDXL effect/scale values for blocks.'
        )
    NodeID = 'ADE_PerBlock_SDXL_FromFloats'
    NodeName = 'AD Per Block Floats (SDXL) 🎭🅐🅓'
    @classmethod
    def execute(cls,
                         effect_16_floats: Union[list[float], None]=None,
                         scale_16_floats: Union[list[float], None]=None):
        if effect_16_floats is None and scale_16_floats is None:
            return io.NodeOutput(AllPerBlocks(None, ModelTypeSD.SDXL))
        # SDXL has 16 blocks
        block_total = 16
        holders = [ADBlockHolder() for _ in range(block_total)]
        if effect_16_floats is not None:
            effect_16_floats = extend_list_to_batch_size(effect_16_floats, block_total)
            for effect, holder in zip(effect_16_floats, holders):
                holder.effect = effect
        if scale_16_floats is not None:
            scale_16_floats = extend_list_to_batch_size(scale_16_floats, block_total)
            for scale, holder in zip(scale_16_floats, holders):
                holder.scales = [scale, scale]
        return PerBlock_SDXL_LowLevelNode.execute(*holders)
