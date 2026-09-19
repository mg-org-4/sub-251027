from pathlib import Path
import tempfile
import time
import traceback

import numpy as np
import soundfile as sf
from PIL import Image
from google import genai

import comfy.model_management as mm
from comfy_api.latest import Types

from ._language_models import GEMINI_LANGUAGE_MODELS, model_choices, validate_gemini_model


class FL_GeminiTextAPI:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (model_choices(GEMINI_LANGUAGE_MODELS), {"default": "gemini-3.7-flash"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 64, "max": 65536, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffff}),
            },
            "optional": {
                "system_instructions": ("STRING", {"multiline": True, "default": ""}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 1, "max": 100, "step": 1}),
                "thinking_level": (["default", "low", "medium", "high"], {"default": "default"}),
                "custom_model": ("STRING", {"default": "", "placeholder": "Optional Gemini model ID override"}),
                "audio": ("AUDIO", {"tooltip": "Send the full audio to Gemini. Each batch item is a separate recording, in order."}),
                "image": ("IMAGE", {"tooltip": "Send every image in the batch, in order. Use VIDEO for frames with timing and sound."}),
                "video": ("VIDEO", {"tooltip": "Send a native ComfyUI video, including its audio and selected trim. Connect Load Video or Create Video."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_text"
    CATEGORY = "🏵️Fill Nodes/AI"

    def __init__(self):
        self.log_messages = []

    def _log(self, message):
        print(f"[FL_GeminiTextAPI] {time.strftime('%Y-%m-%d %H:%M:%S')}: {message}")
        self.log_messages.append(message)

    @staticmethod
    def _media_files(directory, audio, image, video):
        files = []
        if audio is not None:
            waveform, rate = audio["waveform"], int(audio["sample_rate"])
            if waveform.ndim != 3 or waveform.shape[0] == 0 or waveform.shape[1] not in (1, 2) or waveform.shape[2] == 0 or rate <= 0:
                raise ValueError("Audio must contain nonempty mono/stereo recordings with a positive sample rate.")
            for index, clip in enumerate(waveform):
                mm.throw_exception_if_processing_interrupted()
                samples = clip.detach().cpu().float().T.numpy()
                if not np.isfinite(samples).all():
                    raise ValueError("Audio contains non-finite samples.")
                path = directory / f"audio-{index + 1}.flac"
                sf.write(path, samples, rate, subtype="PCM_24")
                files.append(("audio", path, "audio/flac"))
        if image is not None:
            if image.ndim != 4 or image.shape[0] == 0 or image.shape[-1] not in (1, 3, 4):
                raise ValueError("Image must be a nonempty batch of grayscale, RGB, or RGBA images.")
            for index, frame in enumerate(image):
                mm.throw_exception_if_processing_interrupted()
                pixels = frame.detach().cpu().clamp(0, 1).mul(255).byte().numpy()
                if pixels.shape[-1] == 1:
                    pixels = pixels[..., 0]
                path = directory / f"image-{index + 1}.png"
                Image.fromarray(pixels).save(path)
                files.append(("image", path, "image/png"))
        if video is not None:
            mm.throw_exception_if_processing_interrupted()
            path = directory / "video.mp4"
            video.save_to(str(path), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264)
            files.append(("video", path, "video/mp4"))
        return files

    @staticmethod
    def _upload(client, path, mime_type, uploaded):
        mm.throw_exception_if_processing_interrupted()
        file = client.files.upload(file=str(path), config={"mime_type": mime_type})
        uploaded.append(file.name)
        deadline = time.monotonic() + 300
        while file.state.name == "PROCESSING":
            mm.throw_exception_if_processing_interrupted()
            if time.monotonic() >= deadline:
                raise TimeoutError("Gemini media processing timed out; try a shorter clip.")
            time.sleep(1)
            file = client.files.get(name=file.name)
        if file.state.name != "ACTIVE":
            raise ValueError("Gemini could not process the uploaded media.")
        return file.uri

    def generate_text(self, prompt, api_key, model, temperature, max_output_tokens, seed,
                      system_instructions="", top_p=0.95, top_k=64, thinking_level="default", custom_model="",
                      audio=None, image=None, video=None):
        self.log_messages = []
        if not api_key:
            return ("Error: No Google API key provided.",)

        try:
            model, capability = validate_gemini_model(model, custom_model)
            for kind, media in (("audio", audio), ("image", image), ("video", video)):
                if media is not None:
                    validate_gemini_model(model, required_input=kind)
            config = {
                "temperature": temperature,
                "max_output_tokens": min(max_output_tokens, capability.max_output_tokens),
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed or None,
            }
            if system_instructions.strip():
                config["system_instruction"] = system_instructions
            if thinking_level != "default" and thinking_level in capability.thinking_levels:
                config["thinking_config"] = {"thinking_level": thinking_level.upper()}

            with tempfile.TemporaryDirectory(prefix="fl-gemini-") as temporary, genai.Client(api_key=api_key) as client:
                uploaded = []
                try:
                    files = self._media_files(Path(temporary), audio, image, video)
                    inputs = [{"type": "text", "text": prompt}]
                    for kind, path, mime_type in files:
                        uri = self._upload(client, path, mime_type, uploaded)
                        inputs.append({"type": kind, "uri": uri, "mime_type": mime_type})
                    mm.throw_exception_if_processing_interrupted()
                    self._log(f"Sending request to {model} with {len(files)} media file(s).")
                    interaction = client.interactions.create(
                        model=model, store=False, timeout=300,
                        input=inputs if files else prompt,
                        generation_config={key: value for key, value in config.items() if value is not None},
                    )
                finally:
                    for name in uploaded:
                        try:
                            client.files.delete(name=name)
                        except Exception:
                            self._log("Temporary Google file cleanup failed; remove the file in Google AI Studio.")
            text = getattr(interaction, "output_text", "")
            if not text:
                raise ValueError("Gemini returned no text output.")
            return (text.strip(),)
        except mm.InterruptProcessingException:
            raise
        except Exception as error:
            self._log(f"Error: {error}")
            traceback.print_exc()
            return (f"Error: {error}",)
