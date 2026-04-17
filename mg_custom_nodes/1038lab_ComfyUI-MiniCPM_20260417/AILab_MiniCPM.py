import os
import torch
import folder_paths
from transformers import AutoTokenizer, AutoModel
from torchvision.transforms.v2 import ToPILImage
from comfy.comfy_types import IO
from comfy_api.input import VideoInput
import json
import gc
import sys
import io
import warnings
from pathlib import Path

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Suppress transformers FutureWarnings for better user experience
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, 'cuda'):
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'allow_tf32'):
            torch.backends.cuda.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Load configuration file
with open(Path(__file__).parent / "minicpm_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    MODEL_SETTINGS = config["model_settings"]
    PROMPT_TYPES = config.get("prompt_types", {})
    TRANSFORMERS_MODELS = config.get("transformers_models", {})

_MODEL_CACHE = {}


class MiniCPM_Transformers_Models:
    def __init__(self, model: str, processing_mode: str):
        try:
            models_dir = Path(folder_paths.models_dir).resolve()
            prompt_generator_dir = (models_dir / "LLM").resolve()
            prompt_generator_dir.mkdir(parents=True, exist_ok=True)

            model_config = TRANSFORMERS_MODELS[model]
            model_id = model_config["name"]
            self.model_checkpoint = prompt_generator_dir / Path(model_id).name

            if not self.model_checkpoint.exists():
                print(f"Downloading model: {model_id} (this may take several minutes...)")
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(self.model_checkpoint)
                )

            self.device = torch.device("cuda" if processing_mode == "GPU" and torch.cuda.is_available() else "cpu")
            self.bf16_support = (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability(self.device)[0] >= 8
            )

            old_stdout = sys.stdout
            old_stderr = sys.stderr

            try:
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()

                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_checkpoint),
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )

                self.model = AutoModel.from_pretrained(
                    str(self.model_checkpoint),
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa",
                    torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                )

                if processing_mode == "GPU" and torch.cuda.is_available():
                    self.model = self.model.to(self.device)
                self.model.eval()

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e

    def generate(self, images, system: str, prompt: str, max_new_tokens: int,
                temperature: float, top_p: float, top_k: int,
                repetition_penalty: float, video_max_slice_nums: int = 2, seed: int = -1) -> str:
        try:
            if seed != -1:
                torch.manual_seed(seed)

            with torch.no_grad():
                if isinstance(images, list):
                    # Video frame processing
                    msgs = [{"role": "user", "content": images + [prompt]}]
                else:
                    # Single image processing
                    msgs = [{"role": "user", "content": [images, prompt]}]

                params = {"use_image_id": False, "max_slice_nums": video_max_slice_nums}

                result = self.model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    sampling=True,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    **params,
                )

            return result.strip()

        except Exception as e:
            return f"Generation error: {str(e)}"
        finally:
            gc.collect()


class MiniCPM_Transformers_Base:
    def __init__(self):
        self.predictor = None
        self.current_processing_mode = None
        self.current_model = None

    def _load_model(self, model: str, processing_mode: str, memory_management: str):
        cache_key = f"{model}_{processing_mode}"

        try:
            if memory_management == "Global Cache":
                if cache_key in _MODEL_CACHE:
                    self.predictor = _MODEL_CACHE[cache_key]
                else:
                    self.predictor = MiniCPM_Transformers_Models(model, processing_mode)
                    _MODEL_CACHE[cache_key] = self.predictor
            elif (self.predictor is None or self.current_processing_mode != processing_mode or self.current_model != model):
                if self.predictor is not None:
                    del self.predictor
                    self.predictor = None
                    torch.cuda.empty_cache()
                    gc.collect()

                self.predictor = MiniCPM_Transformers_Models(model, processing_mode)
                self.current_processing_mode = processing_mode
                self.current_model = model

        except Exception as model_error:
            raise RuntimeError(f"Model loading error: {str(model_error)}")

    def _cleanup_memory(self, memory_management: str):
        if memory_management == "Clear After Run":
            try:
                del self.predictor
                self.predictor = None
                torch.cuda.empty_cache()
                gc.collect()
            except:
                pass

    def _process_image(self, image):
        try:
            if isinstance(image, (list, tuple)) and len(image) > 0:
                img_tensor = image[0]
            else:
                img_tensor = image
            if not isinstance(img_tensor, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor, got {type(img_tensor)}")
            if img_tensor.dim() == 4:
                img_tensor = img_tensor[0]
            elif img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)
            if img_tensor.shape[0] == 3:
                pass
            elif img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            else:
                raise ValueError(f"Unexpected image tensor shape: {img_tensor.shape}")
            pil_image = ToPILImage()(img_tensor).convert("RGB")
            return pil_image
        except Exception as input_error:
            raise RuntimeError(f"Input processing error: {str(input_error)}")

    def _generate_response(self, images, system_prompt, prompt, **gen_params):
        try:
            with torch.inference_mode():
                response = self.predictor.generate(
                    images=images,
                    system=system_prompt,
                    prompt=prompt,
                    **gen_params
                )
            return response
        except Exception as gen_error:
            raise RuntimeError(f"Generation error: {str(gen_error)}")

    def encode_video(self, source_video, MAX_NUM_FRAMES):
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        components = source_video.get_components()
        vr = components.images
        avg_fps = float(components.frame_rate)
        sample_fps = round(avg_fps / 1)
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = [vr[idx] for idx in frame_idx]
        frames = [ToPILImage()(v.permute([2, 0, 1])).convert("RGB") for v in frames]
        return frames


class AILab_MiniCPM_V(MiniCPM_Transformers_Base):
    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(TRANSFORMERS_MODELS.keys())
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "model": (model_list, {"default": model_list[0]}),
                "preset_prompt": (list(PROMPT_TYPES.keys()), {"default": "Describe"}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "device": (["Auto", "GPU", "CPU"], {"default": "Auto"}),
                "memory_management": (["Keep in Memory", "Clear After Run", "Global Cache"], {"default": "Keep in Memory"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/📝MiniCPM"

    def generate(self, image=None, video=None, model=None, preset_prompt="Describe", custom_prompt="", device="Auto", memory_management="Keep in Memory", seed=-1):
        try:
            if image is None and video is None:
                return ("Error: Please provide either an image or video input.",)
            
            pm = ("GPU" if torch.cuda.is_available() else "CPU") if device == "Auto" else device
            model = model or list(TRANSFORMERS_MODELS.keys())[0]
            self._load_model(model, pm, memory_management)
            
            if video is not None:
                frames = self.encode_video(video, 64)
                if frames:
                    images = frames
                else:
                    return ("Error: No frames extracted from video.",)
            elif image is not None:
                pil_image = self._process_image(image)
                images = pil_image
            else:
                return ("Error: No valid input provided.",)
            
            preset = PROMPT_TYPES.get(preset_prompt, "Describe this image.")
            prompt = custom_prompt.strip() if (isinstance(custom_prompt, str) and custom_prompt.strip()) else preset
            
            if isinstance(seed, int) and seed != -1:
                torch.manual_seed(seed)
            
            response = self._generate_response(
                images=images,
                system_prompt=MODEL_SETTINGS.get("default_system_prompt", ""),
                prompt=prompt,
                max_new_tokens=MODEL_SETTINGS["default_max_tokens"],
                temperature=MODEL_SETTINGS["default_temperature"],
                top_p=MODEL_SETTINGS["default_top_p"],
                top_k=MODEL_SETTINGS["default_top_k"],
                repetition_penalty=MODEL_SETTINGS["default_repetition_penalty"],
                video_max_slice_nums=2,
                seed=seed,
            )
            
            self._cleanup_memory(memory_management)
            return (response,)
        except Exception as e:
            self._cleanup_memory(memory_management)
            return (f"Error: {str(e)}",)


class AILab_MiniCPM_V_Advanced(MiniCPM_Transformers_Base):
    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(TRANSFORMERS_MODELS.keys())
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
                "model": (model_list, {"default": model_list[0]}),
                "preset_prompt": (list(PROMPT_TYPES.keys()), {"default": "Describe"}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "system_prompt": ("STRING", {"default": MODEL_SETTINGS.get("default_system_prompt", ""), "multiline": True}),
                "max_new_tokens": ("INT", {"default": MODEL_SETTINGS["default_max_tokens"], "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": MODEL_SETTINGS["default_temperature"], "min": 0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": MODEL_SETTINGS["default_top_p"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": MODEL_SETTINGS["default_top_k"], "min": 0, "max": 200}),
                "repetition_penalty": ("FLOAT", {"default": MODEL_SETTINGS["default_repetition_penalty"], "min": 0.8, "max": 1.5, "step": 0.01}),
                "video_max_num_frames": ("INT", {"default": 64, "min": 1, "max": 128}),
                "video_max_slice_nums": ("INT", {"default": 2, "min": 1, "max": 4}),
                "device": (["Auto", "GPU", "CPU"], {"default": "Auto"}),
                "memory_management": (["Keep in Memory", "Clear After Run", "Global Cache"], {"default": "Keep in Memory"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("PROMPT", "STRING")
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/📝MiniCPM"

    def generate(self, image=None, video=None, model=None, preset_prompt="Describe", custom_prompt="", system_prompt="", max_new_tokens=None, temperature=None, top_p=None, top_k=None, repetition_penalty=None, video_max_num_frames=64, video_max_slice_nums=2, device="Auto", memory_management="Keep in Memory", seed=-1):
        try:
            if image is None and video is None:
                return ("", "Error: Please provide either an image or video input.")
            
            pm = ("GPU" if torch.cuda.is_available() else "CPU") if device == "Auto" else device
            model = model or list(TRANSFORMERS_MODELS.keys())[0]
            self._load_model(model, pm, memory_management)
            
            if video is not None:
                frames = self.encode_video(video, video_max_num_frames)
                if frames:
                    images = frames
                else:
                    return ("", "Error: No frames extracted from video.")
            elif image is not None:
                pil_image = self._process_image(image)
                images = pil_image
            else:
                return ("", "Error: No valid input provided.")
            
            preset = PROMPT_TYPES.get(preset_prompt, "Describe this image.")
            prompt = custom_prompt.strip() if (isinstance(custom_prompt, str) and custom_prompt.strip()) else preset
            
            if isinstance(seed, int) and seed != -1:
                torch.manual_seed(seed)
            
            max_new_tokens = max_new_tokens if max_new_tokens is not None else MODEL_SETTINGS["default_max_tokens"]
            temperature = temperature if temperature is not None else MODEL_SETTINGS["default_temperature"]
            top_p = top_p if top_p is not None else MODEL_SETTINGS["default_top_p"]
            top_k = top_k if top_k is not None else MODEL_SETTINGS["default_top_k"]
            repetition_penalty = repetition_penalty if repetition_penalty is not None else MODEL_SETTINGS["default_repetition_penalty"]

            response = self._generate_response(
                images=images,
                system_prompt=system_prompt or MODEL_SETTINGS.get("default_system_prompt", ""),
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                video_max_slice_nums=video_max_slice_nums,
                seed=seed,
            )

            self._cleanup_memory(memory_management)
            return (prompt, response)
        except Exception as e:
            self._cleanup_memory(memory_management)
            return ("", f"Error: {str(e)}")


NODE_CLASS_MAPPINGS = {
    "AILab_MiniCPM_V": AILab_MiniCPM_V,
    "AILab_MiniCPM_V_Advanced": AILab_MiniCPM_V_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_MiniCPM_V": "MiniCPM-V",
    "AILab_MiniCPM_V_Advanced": "MiniCPM-V (Advanced)",
}