from .wavespeed_api.utils import imageurl2tensor
from .wavespeed_api.client import WaveSpeedClient
from .wavespeed_api.requests.seededit_v3 import SeedEditV3

class SeedEditV3Node:
    """
    ByteDance SeedEdit-V3 Image Editor Node

    This node uses ByteDance's SeedEdit-V3 model to edit images based on text prompts.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("WAVESPEED_AI_API_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text description of the desired image modifications"}),
                "image_url": ("STRING", {"default": "", "tooltip": "URL of the source image to edit"}),
                "guidance_scale": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Guidance scale for editing (0.0 to 1.0)"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for reproducible results. -1 for random seed"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)

    CATEGORY = "WaveSpeedAI"
    FUNCTION = "execute"

    def execute(self,
                client,
                prompt,
                image_url,
                guidance_scale=0.5,
                seed=-1,
                ):

        if prompt is None or prompt == "":
            raise ValueError("Prompt is required")
        
        if image_url is None or image_url == "":
            raise ValueError("Image URL is required")

        request = SeedEditV3(
            prompt=prompt,
            image=image_url,
            guidance_scale=guidance_scale,
            seed=seed,
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
    "WaveSpeedAI SeedEditV3Node": SeedEditV3Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveSpeedAI SeedEditV3Node": "WaveSpeedAI SeedEdit V3"
} 