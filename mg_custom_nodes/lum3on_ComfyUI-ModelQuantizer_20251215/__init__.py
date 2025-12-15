# __init__.py
# This file is necessary to make Python treat the directory as a package.
# It's also where ComfyUI looks for node mappings.

import sys
import traceback

# Import node classes from nodes.py
# Renamed QuantizeScaled to QuantizeModel
try:
    from .nodes import ModelToStateDict, QuantizeFP8Format, QuantizeModel, SaveAsSafeTensor
except ImportError:
    # Fallback for direct execution or when relative imports fail
    from nodes import ModelToStateDict, QuantizeFP8Format, QuantizeModel, SaveAsSafeTensor

# Import ControlNet FP8 quantization nodes
ControlNetFP8QuantizeNode = None
ControlNetMetadataViewerNode = None
CONTROLNET_NODES_AVAILABLE = False

try:
    try:
        from .controlnet_fp8_node import ControlNetFP8QuantizeNode, ControlNetMetadataViewerNode
    except ImportError:
        from controlnet_fp8_node import ControlNetFP8QuantizeNode, ControlNetMetadataViewerNode
    CONTROLNET_NODES_AVAILABLE = True
    print("✅ ControlNet FP8 nodes imported successfully")
except Exception as e:
    print(f"❌ Failed to import ControlNet FP8 nodes: {e}")
    print(f"❌ ControlNet error details: {traceback.format_exc()}")

# Import GGUF quantization nodes
GGUFQuantizerNode = None # For the new GGUF node
GGUF_NODES_AVAILABLE = False

try:
    try:
        from .gguf_quantizer_node import GGUFQuantizerNode # Updated filename
    except ImportError:
        from gguf_quantizer_node import GGUFQuantizerNode
    GGUF_NODES_AVAILABLE = True
    print("✅ GGUF quantization nodes imported successfully")
except Exception as e:
    print(f"❌ Failed to import GGUF quantization nodes: {e}")
    print(f"❌ GGUF error details: {traceback.format_exc()}")
    GGUF_NODES_AVAILABLE = False


# A dictionary that ComfyUI uses to map node class names to node objects
NODE_CLASS_MAPPINGS = {
    # Node Utils
    "ModelToStateDict": ModelToStateDict,
    # Direct FP8 Conversion
    "QuantizeFP8Format": QuantizeFP8Format,
    # Scaled Quantization + Casting (FP16/BF16)
    "QuantizeModel": QuantizeModel,         # Renamed class QuantizeScaled -> QuantizeModel
    # Saving Node
    "SaveAsSafeTensor": SaveAsSafeTensor,
}

# Add ControlNet nodes if available
if CONTROLNET_NODES_AVAILABLE and ControlNetFP8QuantizeNode is not None:
    NODE_CLASS_MAPPINGS.update({
        # ControlNet FP8 Quantization Nodes
        "ControlNetFP8QuantizeNode": ControlNetFP8QuantizeNode,
        "ControlNetMetadataViewerNode": ControlNetMetadataViewerNode,
    })
    print("✅ ControlNet FP8 nodes registered in NODE_CLASS_MAPPINGS")

# Add GGUF nodes if available
if GGUF_NODES_AVAILABLE and GGUFQuantizerNode is not None:
    NODE_CLASS_MAPPINGS["GGUFQuantizerNode"] = GGUFQuantizerNode # New GGUF node
    print("✅ GGUF quantization nodes registered in NODE_CLASS_MAPPINGS")
else:
    print(f"❌ GGUF nodes NOT registered. Available: {GGUF_NODES_AVAILABLE}, Node: {GGUFQuantizerNode}")


# A dictionary that ComfyUI uses to map node class names to display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelToStateDict": "Model To State Dict",
    "QuantizeFP8Format": "Quantize Model to FP8 Format",
    "QuantizeModel": "Quantize Model Scaled", # Display name for the renamed node
    "SaveAsSafeTensor": "Save Model as SafeTensor",
}

# Add ControlNet display names if available
if CONTROLNET_NODES_AVAILABLE:
    NODE_DISPLAY_NAME_MAPPINGS.update({
        # ControlNet FP8 Quantization Node Display Names
        "ControlNetFP8QuantizeNode": "ControlNet FP8 Quantizer",
        "ControlNetMetadataViewerNode": "ControlNet Metadata Viewer",
    })
    print("✅ ControlNet FP8 display names registered")

# Add GGUF display names if available
if GGUF_NODES_AVAILABLE and GGUFQuantizerNode is not None: # Check new node for display name
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "GGUFQuantizerNode": "GGUF Quantizer 👾", # Display name for GGUF
    })
    print("✅ GGUF quantization display names registered")


# Optional: Print a message to the console when the extension is loaded
print("----------------------------------------------------")
print("--- ComfyUI Quantization Node Pack Loaded ---")
print("--- Renamed QuantizeScaled to QuantizeModel ---")
# ... (other existing print messages you want to keep) ...
print("--- NEW: ControlNet FP8 Quantization Nodes  ---")
print("--- NEW: GGUF Model Quantization ---")
print("--- Developed by [Lum3on]            ---") # Remember to change this!
print("--- Version 0.8.2                             ---") # Incremented version
print("----------------------------------------------------")

# Tell ComfyUI where to find web files (for appearance.js)
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']