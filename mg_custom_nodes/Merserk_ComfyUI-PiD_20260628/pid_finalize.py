from __future__ import annotations

try:
    from .pid_decode import PiDNodeError, _native_pixel_to_comfy_image
    from .pid_sample import PID_SAMPLES_TYPE, PiDSampledBatch
except ImportError:  # pragma: no cover
    from pid_decode import PiDNodeError, _native_pixel_to_comfy_image
    from pid_sample import PID_SAMPLES_TYPE, PiDSampledBatch


class PiDFinalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sampled": (PID_SAMPLES_TYPE,)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "finalize"
    CATEGORY = "PiD/Staged"

    def finalize(self, sampled: PiDSampledBatch):
        if not isinstance(sampled, PiDSampledBatch):
            raise PiDNodeError("PiD Finalize expected a PID_SAMPLES object from PiD Sample.")
        image = _native_pixel_to_comfy_image(sampled.tensor_cpu)
        return (image,)


NODE_CLASS_MAPPINGS = {
    "PiDFinalize": PiDFinalize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDFinalize": "PiD Finalize",
}
