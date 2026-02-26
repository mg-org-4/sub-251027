# only import if running as a custom node
try:
	import comfy.utils
except ImportError:
	pass
else:
	NODE_CLASS_MAPPINGS = {}
	NODE_DISPLAY_NAME_MAPPINGS = {}

	# FluxDeCLIP
	from .flux_mod.nodes import NODE_CLASS_MAPPINGS as FluxMod_NodeIds
	from .flux_mod.nodes import NODE_DISPLAY_NAME_MAPPINGS as FluxMod_NodeNames
	NODE_CLASS_MAPPINGS.update(FluxMod_NodeIds)
	NODE_DISPLAY_NAME_MAPPINGS.update(FluxMod_NodeNames)

	__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
