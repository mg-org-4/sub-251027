from .wavespeed_api.client import WaveSpeedClient
from .wavespeed_api.requests.wan_2_1_14b_vace import Wan2114BVace


class Wan2114BVaceNode:
    """
    ComfyUI Node for the wan-2.1-14b-vace API.
    VACE is an all-in-one model designed for video creation and editing.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("WAVESPEED_AI_API_CLIENT",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Input prompt for video generation."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The negative prompt for generating the output."
                }),
                "task": (["depth", "pose", "none"], {"default": "depth"}),
                "duration": ("INT", {
                    "default": 5,
                    "min": 5,
                    "max": 10,
                    "step": 5,
                    "tooltip": "Video duration in seconds (5-10)."
                }),
                "size": (["832*480", "480*832", "1280*720", "720*1280"], {
                    "default": "832*480",
                    "tooltip": "The size of the output."
                }),
                "num_inference_steps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 40,
                    "step": 1,
                    "tooltip": "The number of inference steps (1-40)."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.01,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "The guidance scale for generation (1.01-10.0)."
                }),
                "flow_shift": ("FLOAT", {
                    "default": 16.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.1,
                    "tooltip": "The shift value for the timestep schedule for flow matching (0.0-30.0)."
                }),
                "context_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Context scale (0.0-2.0)."
                }),
                "seed": ("INT", {
                    "default": -1,
                    "tooltip": "The seed for random number generation. Use -1 for random."
                }),
                "enable_safety_checker": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to enable the safety checker."
                })
            },
            "optional": {
                "image_urls": ("STRING", {
                    "forceInput": False,
                    "tooltip": "Input images to transform (up to 5)"
                }),
                "video": ("STRING", {
                    "forceInput": False,
                    "tooltip": "The video for generating the output. Publicly accessible URL."
                }),
                "mask_image": ("STRING", {
                    "forceInput": False,
                    "tooltip": "URL of the mask image. Publicly accessible URL."
                }),
                "first_image": ("STRING", {
                    "forceInput": False,
                    "tooltip": "URL of the starting image. Publicly accessible URL."
                }),
                "last_image": ("STRING", {
                    "forceInput": False,
                    "tooltip": "URL of the ending image. Publicly accessible URL."
                })
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("video_url", )

    FUNCTION = "execute"
    CATEGORY = "WaveSpeedAI"

    def execute(self,
                client,
                prompt,
                negative_prompt="",
                task="depth",
                duration=5,
                size="832*480",
                num_inference_steps=30,
                guidance_scale=5.0,
                flow_shift=16.0,
                context_scale=1.0,
                seed=-1,
                enable_safety_checker=True,
                image_urls=None,
                video="",
                mask_image=None,
                first_image=None,
                last_image=None):
        if image_urls is None:
            image_urls = []

        # Check if image_urls is a list
        if not isinstance(image_urls, list):
            image_urls = [image_urls]

        # Ensure we have at most 5 image URLs
        images_param = image_urls[:5]

        request = Wan2114BVace(
            prompt=prompt,
            negative_prompt=negative_prompt,
            task=task,
            duration=duration,
            size=size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            flow_shift=flow_shift,
            context_scale=context_scale,
            seed=seed if seed != -1 else None,
            enable_safety_checker=enable_safety_checker,
            images=images_param,
            video=video,
            mask_image=mask_image,
            first_image=first_image,
            last_image=last_image
        )

        waveSpeedClient = WaveSpeedClient(client["api_key"])
        response = waveSpeedClient.send_request(request, True, 3)

        video_url = response.get("outputs", [])
        if not video_url or len(video_url) == 0:
            raise ValueError("No video URL in the generated result")

        return (video_url[0],)


NODE_CLASS_MAPPINGS = {
    "WaveSpeedAI Wan2114BVaceNode": Wan2114BVaceNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaveSpeedAI Wan2114BVaceNode": "WaveSpeedAI Wan 2.1 14B VACE"
}
