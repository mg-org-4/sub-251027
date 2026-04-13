"""
RAW Metadata Split node for ComfyUI-Darkroom.
Unpacks a RAW_METADATA bundle into typed primitive outputs that feed
the rest of Darkroom (ISO into Film Grain, lens into Lens Profile, etc.).
"""


class DarkroomRAWMetadataSplit:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metadata": ("RAW_METADATA", {
                    "tooltip": "Connect from Darkroom RAW Load's metadata output."
                }),
            },
        }

    RETURN_TYPES = (
        "STRING", "STRING",
        "STRING", "STRING",
        "INT", "FLOAT", "FLOAT", "STRING",
        "FLOAT", "FLOAT",
        "STRING", "STRING",
        "INT", "INT", "STRING",
    )
    RETURN_NAMES = (
        "camera_make", "camera_model",
        "lens_make", "lens_model",
        "iso", "aperture", "shutter_seconds", "shutter_string",
        "focal_length", "focal_length_35mm",
        "datetime", "film_simulation",
        "width", "height", "sensor_type",
    )
    FUNCTION = "execute"
    CATEGORY = "AKURATE/Darkroom/RAW"

    def execute(self, metadata):
        m = metadata or {}
        return (
            str(m.get("camera_make", "")),
            str(m.get("camera_model", "")),
            str(m.get("lens_make", "")),
            str(m.get("lens_model", "")),
            int(m.get("iso", 0)),
            float(m.get("aperture", 0.0)),
            float(m.get("shutter_seconds", 0.0)),
            str(m.get("shutter_string", "")),
            float(m.get("focal_length", 0.0)),
            float(m.get("focal_length_35mm", 0.0)),
            str(m.get("datetime", "")),
            str(m.get("film_simulation", "")),
            int(m.get("image_width", 0)),
            int(m.get("image_height", 0)),
            str(m.get("sensor_type", "")),
        )


NODE_CLASS_MAPPINGS = {"DarkroomRAWMetadataSplit": DarkroomRAWMetadataSplit}
NODE_DISPLAY_NAME_MAPPINGS = {"DarkroomRAWMetadataSplit": "RAW Metadata Split"}
