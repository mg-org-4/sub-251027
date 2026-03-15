import os
import sys
import importlib

python = sys.executable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Add HYPIR to path for imports
HYPIR_PATH = os.path.join(CURRENT_DIR, "HYPIR")
if HYPIR_PATH not in sys.path:
    sys.path.append(HYPIR_PATH)

def load_nodes():
    # Add current directory to Python path so modules can find each other
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    
    # Define the order of module loading to avoid circular imports
    node_modules = [
        "hypir_config",      # Load config first
        "model_downloader",  # Load downloader second
        "hypir_node",        # Load basic nodes
        "hypir_advanced_node", # Load advanced nodes
        "hypir_model_manager"  # Load manager last
    ]
    
    # Load node modules in order
    for module_name in node_modules:
        try:
            # Use regular import now that the path is set up
            module = importlib.import_module(module_name)
            
            if hasattr(module, "NODE_CLASS_MAPPINGS"):
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        except Exception as e:
            print(f"Error loading module {module_name}: {e}")
    
    # Also try to load any other Python files that might be node files
    for filename in os.listdir(CURRENT_DIR):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            if module_name not in node_modules:  # Skip already loaded modules
                try:
                    # Use regular import now that the path is set up
                    module = importlib.import_module(module_name)
                    
                    if hasattr(module, "NODE_CLASS_MAPPINGS"):
                        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                except Exception as e:
                    print(f"Error loading module {module_name}: {e}")

load_nodes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
