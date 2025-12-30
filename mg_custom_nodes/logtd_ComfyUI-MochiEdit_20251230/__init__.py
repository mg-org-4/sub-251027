# from .nodes.unsampler_node import MochiWrapperUnsamplerNode
# from .nodes.resampler_node import MochiWrapperResamplerNode
from .nodes.sampler_custom_node import MochiWrapperSamplerCustomNode
from .nodes.sampler_unsample_node import MochiUnsamplerNode
from .nodes.sampler_resample_node import MochiResamplerNode
from .nodes.prepare_mochi_sigmas_node import MochiPrepareSigmasNode


NODE_CLASS_MAPPINGS = {
    # "MochiWrapperUnsampler": MochiWrapperUnsamplerNode,
    # "MochiWrapperResampler": MochiWrapperResamplerNode,
    "MochiWrapperSamplerCustom": MochiWrapperSamplerCustomNode,
    "MochiUnsampler": MochiUnsamplerNode,
    "MochiResampler": MochiResamplerNode,
    "MochiPrepareSigmas": MochiPrepareSigmasNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # "MochiWrapperUnsampler": "Mochi Wrapper Unsampler",
    # "MochiWrapperResampler": "Mochi Wrapper Resampler",
    "MochiWrapperSamplerCustom": "SamplerCustom (Mochi Wrapper)",
    "MochiUnsampler": "Mochi Unsampler",
    "MochiResampler": "Mochi Resampler",
    "MochiPrepareSigmas": "Mochi Prepare Sigmas",
}