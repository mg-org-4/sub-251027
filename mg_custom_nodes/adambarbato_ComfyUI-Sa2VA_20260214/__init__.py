# ComfyUI-Sa2VA - Sa2VA Segmentation Nodes for ComfyUI

from .config import be_quiet  # Import the global config

# Import Sa2VA node
from .nodes.sa2va_node_tpl import Sa2VANodeTpl

# Define node class mappings
NODE_CLASS_MAPPINGS = {
    "Sa2VANodeTpl": Sa2VANodeTpl,
}

# Define node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "Sa2VANodeTpl": "Sa2VA Segmentation",
}

# Expose the mappings to ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
