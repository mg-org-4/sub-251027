#__init__.py
import os
import sys
from pathlib import Path

# Get the absolute path to the current directory and memo directory
CURRENT_DIR = Path(__file__).parent.absolute()
MEMO_DIR = CURRENT_DIR / "memo"

# Add both directories to Python path if they're not already there
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(MEMO_DIR) not in sys.path:
    sys.path.insert(0, str(MEMO_DIR))

# Create an empty __init__.py in memo directory if it doesn't exist
memo_init = MEMO_DIR / "__init__.py"
if not memo_init.exists():
    memo_init.touch()

# Now import the components using absolute imports
from .memo_model_manager import MemoModelManager
from .IF_MemoAvatar import IF_MemoAvatar
from .IF_MemoCheckpointLoader import IF_MemoCheckpointLoader

NODE_CLASS_MAPPINGS = {
    "IF_MemoAvatar": IF_MemoAvatar,
    "IF_MemoCheckpointLoader": IF_MemoCheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_MemoAvatar": "IF MemoAvatar 🗣️",
    "IF_MemoCheckpointLoader": "IF Memo Checkpoint Loader"
}

# Define web directory relative to this file
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]