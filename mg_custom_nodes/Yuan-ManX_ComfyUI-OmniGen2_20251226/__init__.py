from .nodes import LoadOmniGen2Image, LoadOmniGen2Model, OmniGen2

NODE_CLASS_MAPPINGS = {
    "LoadOmniGen2Image": LoadOmniGen2Image,
    "LoadOmniGen2Model": LoadOmniGen2Model,
    "OmniGen2": OmniGen2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadOmniGen2Image": "Load OmniGen2 Image",
    "LoadOmniGen2Model": "Load OmniGen2 Model",
    "OmniGen2": "OmniGen2",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
