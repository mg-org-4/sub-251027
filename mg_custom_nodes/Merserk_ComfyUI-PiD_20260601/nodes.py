"""ComfyUI registration shim.

Each node implementation lives in its own module. ComfyUI imports this file as
the package entry point, so keep it small and only expose mappings here.
"""

try:
    from .pid_decode import NODE_CLASS_MAPPINGS as DECODE_CLASS_MAPPINGS
    from .pid_decode import NODE_DISPLAY_NAME_MAPPINGS as DECODE_DISPLAY_NAME_MAPPINGS
    from .pid_text_prompt import NODE_CLASS_MAPPINGS as PROMPT_CLASS_MAPPINGS
    from .pid_text_prompt import NODE_DISPLAY_NAME_MAPPINGS as PROMPT_DISPLAY_NAME_MAPPINGS
    from .pid_staged_decode import NODE_CLASS_MAPPINGS as STAGED_CLASS_MAPPINGS
    from .pid_staged_decode import NODE_DISPLAY_NAME_MAPPINGS as STAGED_DISPLAY_NAME_MAPPINGS
    from .pid_capture_sampler import NODE_CLASS_MAPPINGS as CAPTURE_CLASS_MAPPINGS
    from .pid_capture_sampler import NODE_DISPLAY_NAME_MAPPINGS as CAPTURE_DISPLAY_NAME_MAPPINGS
except ImportError:  # Allows `python -c "import nodes"` from this folder.
    from pid_decode import NODE_CLASS_MAPPINGS as DECODE_CLASS_MAPPINGS
    from pid_decode import NODE_DISPLAY_NAME_MAPPINGS as DECODE_DISPLAY_NAME_MAPPINGS
    from pid_text_prompt import NODE_CLASS_MAPPINGS as PROMPT_CLASS_MAPPINGS
    from pid_text_prompt import NODE_DISPLAY_NAME_MAPPINGS as PROMPT_DISPLAY_NAME_MAPPINGS
    from pid_staged_decode import NODE_CLASS_MAPPINGS as STAGED_CLASS_MAPPINGS
    from pid_staged_decode import NODE_DISPLAY_NAME_MAPPINGS as STAGED_DISPLAY_NAME_MAPPINGS
    from pid_capture_sampler import NODE_CLASS_MAPPINGS as CAPTURE_CLASS_MAPPINGS
    from pid_capture_sampler import NODE_DISPLAY_NAME_MAPPINGS as CAPTURE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {
    **DECODE_CLASS_MAPPINGS,
    **PROMPT_CLASS_MAPPINGS,
    **STAGED_CLASS_MAPPINGS,
    **CAPTURE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **DECODE_DISPLAY_NAME_MAPPINGS,
    **PROMPT_DISPLAY_NAME_MAPPINGS,
    **STAGED_DISPLAY_NAME_MAPPINGS,
    **CAPTURE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
