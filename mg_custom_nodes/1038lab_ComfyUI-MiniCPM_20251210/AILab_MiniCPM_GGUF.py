import torch
import folder_paths
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToPILImage
import json
import base64
import io
import sys
import gc
import os
import re
from huggingface_hub import hf_hub_download

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    LLAMA_CPP_AVAILABLE = True
    LLAMA_CPP_ERROR = None
except Exception as e:
    LLAMA_CPP_AVAILABLE = False
    LLAMA_CPP_ERROR = str(e)
    class Llama:
        pass
    class Llava15ChatHandler:
        pass

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, 'cuda'):
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'allow_tf32'):
            torch.backends.cuda.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

with open(Path(__file__).parent / "minicpm_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    MODEL_SETTINGS = config["model_settings"]
    PROMPT_TYPES = config.get("prompt_types", {})
    GGUF_MODELS = config["gguf_models"]

_MODEL_CACHE = {}

class MiniCPM_GGUF_Models:
    def __init__(self, model: str, processing_mode: str):
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError(f"llama-cpp-python is not available: {LLAMA_CPP_ERROR}")

        try:
            models_dir = Path(folder_paths.models_dir).resolve()
            llm_models_dir = (models_dir / "LLM" / "GGUF").resolve()
            llm_models_dir.mkdir(parents=True, exist_ok=True)

            if "/" not in model:
                raise ValueError("Invalid model path")
            repo_path, filename = model.rsplit("/", 1)

            model_config = None
            model_key = None
            for key, config in GGUF_MODELS.items():
                if config["name"] == model:
                    model_config = config
                    model_key = key
                    break
            
            if not model_config:
                raise ValueError(f"Model configuration not found for: {model}")

            if "download_path" in model_config:
                download_subdir = llm_models_dir / model_config["download_path"]
            else:
                download_subdir = llm_models_dir
            download_subdir.mkdir(parents=True, exist_ok=True)

            model_path = download_subdir / filename
            if not model_path.exists():
                print(f"Downloading GGUF model: {filename} (large file, please wait...)")
                try:
                    model_path = Path(hf_hub_download(
                        repo_id=repo_path,
                        filename=filename,
                        local_dir=str(download_subdir)
                    )).resolve()
                except Exception as e:
                    print(f"GGUF model download failed: {e}")
                    raise

            mmproj_filename = model_config.get("mmproj")
            if not mmproj_filename:
                if "MiniCPM-V-4.5" in model_key or "4_5" in model:
                    mmproj_filename = "openbmb/MiniCPM-V-4_5-gguf/mmproj-model-f16.gguf"
                else:
                    mmproj_filename = "openbmb/MiniCPM-V-4-gguf/mmproj-model-f16.gguf"
            
            mmproj_local = download_subdir / Path(mmproj_filename).name
            if not mmproj_local.exists():
                print(f"Downloading vision model: {Path(mmproj_filename).name}...")
                repo_path, filename = mmproj_filename.rsplit("/", 1)
                try:
                    mmproj_local = Path(hf_hub_download(
                        repo_id=repo_path,
                        filename=filename,
                        local_dir=str(download_subdir)
                    )).resolve()
                except Exception as e:
                    print(f"Vision model download failed: {e}")
                    raise

            n_ctx = MODEL_SETTINGS["context_window"]
            n_batch = 2048
            n_threads = max(4, MODEL_SETTINGS["cpu_threads"])
            n_gpu_layers = -1 if processing_mode == "GPU" else 0

            old_stdout = sys.stdout
            old_stderr = sys.stderr

            try:
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()

                try:
                    self.model = Llama(
                        model_path=str(model_path),
                        n_ctx=n_ctx,
                        n_batch=n_batch,
                        n_threads=n_threads,
                        n_gpu_layers=n_gpu_layers,
                        verbose=False,
                        chat_handler=Llava15ChatHandler(clip_model_path=str(mmproj_local)),
                        offload_kqv=True,
                        numa=True
                    )
                except Exception as model_error:
                    error_msg = str(model_error).lower()
                    if "unknown minicpmv version" in error_msg or "unsupported minicpmv version" in error_msg:
                        # Check if this is a V4.5 model
                        is_v45_model = any([
                            "4.5" in model.lower(),
                            "4_5" in model.lower(),
                            "v4.5" in model.lower()
                        ])
                        
                        if is_v45_model:
                            raise RuntimeError(
                                f"MiniCPM-V-4.5 compatibility issue detected.\n"
                                f"MiniCPM-V-4.5 support was just added to llama.cpp on Aug 26, 2025 (PR #15575).\n"
                                f"Your llama-cpp-python 0.3.16 was compiled before this update.\n\n"
                                f"Solutions:\n"
                                f"1. 🔄 Wait for new llama-cpp-python release (recommended - should be available soon)\n"
                                f"2. 🔨 Compile from source: pip uninstall llama-cpp-python && pip install llama-cpp-python --force-reinstall --no-cache-dir\n"
                                f"3. 🎯 Use MiniCPM-V-4.5 Transformers node instead (works perfectly)\n"
                                f"4. 🔙 Use MiniCPM-V-4.0 GGUF models (fully supported)\n\n"
                                f"Background: MiniCPM-V-4.5 GGUF support requires the latest llama.cpp code.\n"
                                f"Original error: {model_error}"
                            )
                        else:
                            raise RuntimeError(
                                f"MiniCPM version compatibility issue detected.\n"
                                f"Your llama-cpp-python version doesn't support this model.\n\n"
                                f"Try:\n"
                                f"1. Update llama-cpp-python: pip install --upgrade llama-cpp-python\n"
                                f"2. Try different version: pip install llama-cpp-python==0.2.90\n"
                                f"3. Use the MiniCPM transformers node instead\n\n"
                                f"Original error: {model_error}"
                            )
                    else:
                        raise model_error

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")  from e

    def generate(self, image: Image.Image, system: str, prompt: str, max_new_tokens: int,
                temperature: float, top_p: float, top_k: int,
                repetition_penalty: float, seed: int = -1) -> str:
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image = image.resize((336, 336), Image.Resampling.BILINEAR)

            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            data_uri = f"data:image/png;base64,{img_base64}"

            messages = [
                {"role": "system", "content": (system or "").strip()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (prompt or "").strip()},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }
            ]

            completion_params = {
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
                "repeat_penalty": repetition_penalty,
                "mirostat_mode": 0,
                "stop": ["", "User:", "Assistant:", "###"]
            }

            if top_k > 0:
                completion_params["top_k"] = top_k
            if isinstance(seed, int) and seed >= 0:
                completion_params["seed"] = seed

            try:
                import inspect
                allowed = set(inspect.signature(self.model.create_chat_completion).parameters.keys())
                for k in list(completion_params.keys()):
                    if k not in allowed:
                        completion_params.pop(k, None)
            except Exception:
                pass

            old_stdout = sys.stdout
            old_stderr = sys.stderr

            try:
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
                response = self.model.create_chat_completion(**completion_params)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            del messages
            content = ""
            try:
                content = response["choices"][0]["message"]["content"]
            except Exception:
                pass
            if not content:
                content = response["choices"][0].get("text", "")

            if not content:
                retry_params = dict(completion_params)
                retry_params["stop"] = []
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                try:
                    sys.stdout = io.StringIO()
                    sys.stderr = io.StringIO()
                    retry_resp = self.model.create_chat_completion(**retry_params)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                try:
                    content = retry_resp["choices"][0]["message"].get("content", "") or retry_resp["choices"][0].get("text", "")
                except Exception:
                    content = ""

            content = self._clean_output(content)
            return (content or '').strip()

        except Exception as e:
            return f"Generation error: {str(e)}"
        finally:
            gc.collect()
    
    def _clean_output(self, text: str) -> str:
        if not text:
            return text

        patterns = [
            r'^[\s\-•*]+',                      
            r'^(?!1\.)\d+[\.\)\s\-]+',          
            r'^(Assistant|User|MiniCPM|AI):\s*',
            r'^[A-Z][a-z]+:\s*'                 
        ]

        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        text = text.strip()

        return text or "Unable to generate description."

class MiniCPM_GGUF_Base:
    def __init__(self):
        self.predictor = None
        self.current_processing_mode = None
        self.current_model = None

    def _load_model(self, model: str, processing_mode: str, memory_management: str):
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError(f"llama-cpp-python is not available: {LLAMA_CPP_ERROR}")

        cache_key = f"{model}_{processing_mode}"

        try:
            if memory_management == "Global Cache":
                if cache_key in _MODEL_CACHE:
                    self.predictor = _MODEL_CACHE[cache_key]
                else:
                    model_name = GGUF_MODELS[model]["name"]
                    self.predictor = MiniCPM_GGUF_Models(model_name, processing_mode)
                    _MODEL_CACHE[cache_key] = self.predictor
            elif (self.predictor is None or self.current_processing_mode != processing_mode or self.current_model != model):
                if self.predictor is not None:
                    del self.predictor
                    self.predictor = None
                    torch.cuda.empty_cache()
                    gc.collect()

                model_name = GGUF_MODELS[model]["name"]
                self.predictor = MiniCPM_GGUF_Models(model_name, processing_mode)
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
            pil_image = ToPILImage()(img_tensor)
            return pil_image
        except Exception as input_error:
            raise RuntimeError(f"Input processing error: {str(input_error)}")

    def _generate_response(self, pil_image, system_prompt, prompt, **gen_params):
        try:
            with torch.inference_mode():
                response = self.predictor.generate(
                    image=pil_image,
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

class AILab_MiniCPM_V_GGUF(MiniCPM_GGUF_Base):
    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(GGUF_MODELS.keys())
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
            model = model or list(GGUF_MODELS.keys())[0]
            self._load_model(model, pm, memory_management)
            if video is not None:
                frames = self.encode_video(video, 64)
                if frames:
                    pil_image = frames[0]
                else:
                    return ("Error: No frames extracted from video.",)
            elif image is not None:
                pil_image = self._process_image(image)
            else:
                return ("Error: No valid input provided.",)
            preset = PROMPT_TYPES.get(preset_prompt, "Describe this image.")
            prompt = custom_prompt.strip() if (isinstance(custom_prompt, str) and custom_prompt.strip()) else preset
            if isinstance(seed, int) and seed != -1:
                torch.manual_seed(seed)
            response = self._generate_response(
                pil_image=pil_image,
                system_prompt=MODEL_SETTINGS.get("default_system_prompt", ""),
                prompt=prompt,
                max_new_tokens=MODEL_SETTINGS["default_max_tokens"],
                temperature=MODEL_SETTINGS["default_temperature"],
                top_p=MODEL_SETTINGS["default_top_p"],
                top_k=MODEL_SETTINGS["default_top_k"],
                repetition_penalty=MODEL_SETTINGS["default_repetition_penalty"],
                seed=seed,
            )
            self._cleanup_memory(memory_management)
            return (response,)
        except Exception as e:
            self._cleanup_memory(memory_management)
            return (f"Error: {str(e)}",)


class AILab_MiniCPM_V_GGUF_Advanced(MiniCPM_GGUF_Base):
    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(GGUF_MODELS.keys())
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
            model = model or list(GGUF_MODELS.keys())[0]
            self._load_model(model, pm, memory_management)
            if video is not None:
                frames = self.encode_video(video, video_max_num_frames)
                if frames:
                    pil_image = frames[0]
                else:
                    return ("", "Error: No frames extracted from video.")
            elif image is not None:
                pil_image = self._process_image(image)
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
                pil_image=pil_image,
                system_prompt=system_prompt or MODEL_SETTINGS.get("default_system_prompt", ""),
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )

            self._cleanup_memory(memory_management)
            return (prompt, response)
        except Exception as e:
            self._cleanup_memory(memory_management)
            return ("", f"Error: {str(e)}")


NODE_CLASS_MAPPINGS = {
    "AILab_MiniCPM_V_GGUF": AILab_MiniCPM_V_GGUF,
    "AILab_MiniCPM_V_GGUF_Advanced": AILab_MiniCPM_V_GGUF_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_MiniCPM_V_GGUF": "MiniCPM-V GGUF",
    "AILab_MiniCPM_V_GGUF_Advanced": "MiniCPM-V GGUF (Advanced)",
}
