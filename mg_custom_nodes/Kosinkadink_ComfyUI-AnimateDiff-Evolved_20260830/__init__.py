import folder_paths
from .animatediff.logger import logger
from .animatediff.utils_model import get_available_motion_models, Folders
from .animatediff.model_injection import prepare_dinklink_register_definitions
from .animatediff.motion_module_ad import prepare_dinklink_motion_module_ad
from .animatediff.nodes import AnimateDiffExtension
from .animatediff.dinklink import init_dinklink

if len(get_available_motion_models()) == 0:
    logger.error(f"No motion models found. Please download one and place in: {folder_paths.get_folder_paths(Folders.ANIMATEDIFF_MODELS)}")

WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY"]

init_dinklink()
prepare_dinklink_register_definitions()
prepare_dinklink_motion_module_ad()


async def comfy_entrypoint() -> AnimateDiffExtension:
    return AnimateDiffExtension()
