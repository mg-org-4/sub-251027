from .wavespeed_api.utils import imageurl2tensor
from .wavespeed_api.client import WaveSpeedClient
from .wavespeed_api.requests.seedream_v3 import SeedreamV3

class SeedreamV3Node:
    """
    ByteDance Seedream-V3 Image Generator Node

    This node uses ByteDance's Seedream-V3 model to generate high-quality images.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("WAVESPEED_AI_API_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text description of the image to generate"}),
                "width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 8,
                    "display": "number",
                    "tooltip": "Image width (512 to 2048)"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 8,
                    "display": "number",
                    "tooltip": "Image height (512 to 2048)"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for reproducible results. -1 for random seed"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 2.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Guidance scale for generation (0.0 to 10.0)"
                }),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)

    CATEGORY = "WaveSpeedAI"
    FUNCTION = "execute"

    def execute(self,
                client,
                prompt,
                width=1024,
                height=1024,
                seed=-1,
                guidance_scale=2.5,
                ):

        if prompt is None or prompt == "":
            raise ValueError("Prompt is required")

        request = SeedreamV3(
            prompt=prompt,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
        )

        waveSpeedClient = WaveSpeedClient(client["api_key"])
        response = waveSpeedClient.send_request(request, True, 1)

        # Download and process images
        image_urls = response.get("outputs", [])
        if not image_urls:
            raise ValueError("No image URLs in the generated result")

        images = imageurl2tensor(image_urls)
        return (images,)

NODE_CLASS_MAPPINGS = {
    "WaveSpeedAI SeedreamV3Node": SeedreamV3Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveSpeedAI SeedreamV3Node": "WaveSpeedAI Seedream V3"
} 