"""ComfyUI entry point. A1111 and Forge load install.py and scripts/tipo.py instead."""

from comfy_api.latest import ComfyExtension, io

from .tipo_installer import ensure_tipo_kgen, prepare_dll_path


class TipoExtension(ComfyExtension):
    async def on_load(self) -> None:
        """Settle tipo-kgen before any schema is built.

        The model dropdown is read from kgen at registration time and never
        rebuilt, so an outdated or missing kgen bakes a wrong list in until the
        next restart. llama-cpp stays lazy: it is a large hardware-specific
        wheel and nothing at startup depends on it.
        """
        ensure_tipo_kgen()
        prepare_dll_path()

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        # Imported here rather than at module scope because importing the nodes
        # imports kgen, which on_load above is responsible for installing first.
        from .nodes.tipo import NODES

        return NODES


async def comfy_entrypoint() -> ComfyExtension:
    return TipoExtension()
