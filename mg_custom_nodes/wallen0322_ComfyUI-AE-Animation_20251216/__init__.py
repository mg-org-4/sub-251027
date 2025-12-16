from .ae_animation_core import AEAnimationExtension, comfy_entrypoint

# 前端静态文件目录：使用 ./js（与大多数扩展保持一致）
WEB_DIRECTORY = "./js"

__all__ = [
    "comfy_entrypoint",
    "AEAnimationExtension",
    "WEB_DIRECTORY",
]
