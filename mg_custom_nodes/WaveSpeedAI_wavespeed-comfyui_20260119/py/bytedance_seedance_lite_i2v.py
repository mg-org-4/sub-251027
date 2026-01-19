from PIL import Image
from .wavespeed_api.client import WaveSpeedClient
from .wavespeed_api.requests.bytedance_seedance_v1_lite_i2v_480p import BytedanceSeedanceV1LiteI2V480P
from .wavespeed_api.requests.bytedance_seedance_v1_lite_i2v_720p import ByteDanceSeedanceV1LiteI2V720P
from .wavespeed_api.requests.bytedance_seedance_v1_lite_i2v_1080p import BytedanceSeedanceV1LiteI2V1080P


class BytedanceSeedanceLiteI2VNode:
    """
    ComfyUI Node for bytedance seedance lite i2v API.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("WAVESPEED_AI_API_CLIENT",),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text prompt for generation"}),
                "image_url": ("STRING", {"tooltip": "Input image URL", "forceInput": False}),
                "resolution": (["480p", "720p", "1080p"], {
                    "default": "480p",
                    "tooltip": "Output resolution (480p, 720p or 1080p)"
                }),
                "duration": ("INT", {
                    "default": 5,
                    "min": 5,
                    "max": 10,
                    "step": 5,
                    "display": "number",
                    "tooltip": "Generate video duration length seconds."
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed (-1 for random)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("video_url", )

    FUNCTION = "execute"
    CATEGORY = "WaveSpeedAI"

    def execute(self,
                client,
                prompt,
                image_url=None,
                resolution="480p",
                duration=5,
                seed=-1,):

        if image_url is None or image_url == "":
            raise ValueError("Image URL is required")

        if resolution == '480p':
            request_class = BytedanceSeedanceV1LiteI2V480P
        elif resolution == '720p':
            request_class = ByteDanceSeedanceV1LiteI2V720P
        elif resolution == '1080p':
            request_class = BytedanceSeedanceV1LiteI2V1080P
        else:
            request_class = BytedanceSeedanceV1LiteI2V480P

        request = request_class(
            prompt=prompt,
            image=image_url,
            duration=duration,
            seed=seed
        )

        waveSpeedClient = WaveSpeedClient(client["api_key"])
        response = waveSpeedClient.send_request(request, True, 3)

        video_url = response.get("outputs", [])
        if not video_url or len(video_url) == 0:
            raise ValueError("No video URL in the generated result")

        return (video_url[0],)


NODE_CLASS_MAPPINGS = {
    "WaveSpeedAI BytedanceSeedanceLiteI2VNode": BytedanceSeedanceLiteI2VNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveSpeedAI BytedanceSeedanceLiteI2VNode": "WaveSpeedAI ByteDance Seedance Lite I2V"
}
