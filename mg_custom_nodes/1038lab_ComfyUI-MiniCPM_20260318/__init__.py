import os
import importlib
import sys
from pathlib import Path

current_dir = Path(__file__).parent

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

EXCLUDE_FILES = ['__init__.py', '__pycache__']

# Enable Windows color support
if sys.platform == 'win32':
    os.system('color')

# Check if llama-cpp-python is available for GGUF functionality
GGUF_AVAILABLE = False
try:
    import llama_cpp
    GGUF_AVAILABLE = True
except ImportError:
    print("\033[93m⚠️  GGUF functionality unavailable - install llama-cpp-python for GGUF support\033[0m")
    print("\033[96m📖 Installation guide: https://github.com/1038lab/ComfyUI-MiniCPM/tree/main/llama_cpp_install\033[0m")
except Exception as e:
    print(f"\033[91m❌ GGUF loading error: {str(e)}\033[0m")
    print("\033[96m📖 Installation guide: https://github.com/1038lab/ComfyUI-MiniCPM/tree/main/llama_cpp_install\033[0m")

# Process all Python files in the directory (auto-registration functionality)
loaded_modules = []
skipped_modules = []

for file in current_dir.glob('*.py'):
    if file.name not in EXCLUDE_FILES:
        try:
            module_name = file.stem
            spec = importlib.util.spec_from_file_location(module_name, str(file))
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            
            # Skip GGUF module if llama-cpp-python is not available
            if not GGUF_AVAILABLE and 'GGUF' in module_name:
                skipped_modules.append(module_name)
                continue
                
            spec.loader.exec_module(module)
            
            if hasattr(module, 'NODE_CLASS_MAPPINGS'):
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                loaded_modules.append(module_name)
            
            if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                
        except Exception as e:
            print(f"\033[91m❌ Failed to load {module_name}: {str(e)}\033[0m")
            skipped_modules.append(module_name)

# Summary output
if loaded_modules:
    print(f"\033[92m✅ MiniCPM loaded: {len(NODE_CLASS_MAPPINGS)} nodes from {len(loaded_modules)} modules\033[0m")
if skipped_modules:
    print(f"\033[93m⏭️  Skipped: {', '.join(skipped_modules)}\033[0m")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
