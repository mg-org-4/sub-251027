from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import server_routes  # registers /darkroom/list_dir and /darkroom/mkdir

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("\033[34m[ComfyUI-Darkroom] \033[92mLoaded — 38 nodes (Film Emulation + Camera Raw + Color Grading + Lens & Optics + LUT Pipeline + RAW Pipeline)\033[0m")
