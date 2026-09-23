from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

try:
    import folder_paths
except Exception:  # pragma: no cover - ComfyUI-only import
    folder_paths = None

try:
    from .pid_decode import NATIVE_PID_SUBFOLDER, PiDNodeError, _free_cuda_memory, _preferred_model_folder
except ImportError:  # pragma: no cover
    from pid_decode import NATIVE_PID_SUBFOLDER, PiDNodeError, _free_cuda_memory, _preferred_model_folder


QWEN_CAPTION_REPO_ID = "Qwen/Qwen3.5-0.8B"
QWEN_CAPTION_LOCAL_DIR = "qwen35_caption"
QWEN_CAPTION_PROMPT = (
    "Generate one ultra-precise detailed sentence describing only the visible image. "
    "Use close to 100 words without guessing. Include all clearly visible details. No extra text."
)

_CAPTION_MODEL = None
_CAPTION_PROCESSOR = None
_CAPTION_MODEL_PATH: Optional[Path] = None


def _caption_model_dir() -> Path:
    if folder_paths is not None:
        base = _preferred_model_folder("text_encoders", "text_encoders")
    else:
        base = Path.cwd() / "models" / "text_encoders"
    return base / NATIVE_PID_SUBFOLDER / QWEN_CAPTION_LOCAL_DIR


def _caption_model_is_present(path: Path) -> bool:
    return (path / "config.json").is_file()


def _ensure_caption_model(allow_download: bool = True) -> Path:
    target = _caption_model_dir()
    if _caption_model_is_present(target):
        return target
    if not allow_download:
        raise PiDNodeError(
            f"Missing PiD Caption Creator model: {target}\n"
            f"Download {QWEN_CAPTION_REPO_ID} into this folder or enable auto_download."
        )

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise PiDNodeError(
            "PiD Caption Creator auto_download requires huggingface-hub. "
            "Install this node's requirements.txt and try again."
        ) from exc

    target.mkdir(parents=True, exist_ok=True)
    print(f"[ComfyUI-PiD] downloading {QWEN_CAPTION_REPO_ID} for PiD Caption Creator", flush=True)
    snapshot_download(
        repo_id=QWEN_CAPTION_REPO_ID,
        local_dir=str(target),
    )
    if not _caption_model_is_present(target):
        raise PiDNodeError(f"PiD Caption Creator download finished but config.json is missing: {target}")
    return target


def _load_caption_backend(allow_download: bool = True):
    global _CAPTION_MODEL, _CAPTION_PROCESSOR, _CAPTION_MODEL_PATH

    model_path = _ensure_caption_model(allow_download=allow_download)
    if _CAPTION_MODEL is not None and _CAPTION_PROCESSOR is not None and _CAPTION_MODEL_PATH == model_path:
        return _CAPTION_PROCESSOR, _CAPTION_MODEL

    try:
        import transformers
        from transformers import AutoProcessor
    except Exception as exc:
        raise PiDNodeError(
            "PiD Caption Creator requires transformers. Install this node's requirements.txt and restart ComfyUI."
        ) from exc

    model_cls = None
    for class_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
        model_cls = getattr(transformers, class_name, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise PiDNodeError(
            "PiD Caption Creator requires a transformers build with "
            "AutoModelForImageTextToText or AutoModelForMultimodalLM."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    try:
        model = model_cls.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
    except TypeError:
        model = model_cls.from_pretrained(str(model_path), trust_remote_code=True)

    if device != "cuda":
        model = model.to(device)
    model.eval()

    _CAPTION_PROCESSOR = processor
    _CAPTION_MODEL = model
    _CAPTION_MODEL_PATH = model_path
    return processor, model


def _clean_caption(caption: str) -> str:
    text = str(caption or "").strip()
    for prefix in ("assistant:", "Assistant:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if not text:
        raise PiDNodeError("PiD Caption Creator generated an empty caption.")
    return " ".join(text.split())


def _image_tensor_to_pil(image: torch.Tensor):
    try:
        from PIL import Image
    except Exception as exc:
        raise PiDNodeError("PiD Caption Creator requires Pillow. Install this node's requirements.txt.") from exc

    if not isinstance(image, torch.Tensor):
        raise PiDNodeError("PiD Caption Creator expected a ComfyUI IMAGE tensor.")
    if image.ndim != 3:
        raise PiDNodeError(f"PiD Caption Creator expected one image as [H,W,C], got shape {list(image.shape)}")
    if image.shape[-1] < 3:
        raise PiDNodeError(f"PiD Caption Creator expected RGB image channels, got shape {list(image.shape)}")

    array = image[..., :3].detach().float().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
    return Image.fromarray(array, mode="RGB")


def _ensure_image_batch(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise PiDNodeError("PiD Caption Creator expected a ComfyUI IMAGE tensor.")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise PiDNodeError(f"PiD Caption Creator expected IMAGE as [B,H,W,C], got shape {list(image.shape)}")
    if image.shape[-1] < 3:
        raise PiDNodeError(f"PiD Caption Creator expected at least 3 image channels, got shape {list(image.shape)}")
    return image[..., :3].detach().float().cpu().clamp(0.0, 1.0).contiguous()


def _generate_caption_for_pil(processor, model, image, prompt: str = QWEN_CAPTION_PROMPT) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
        )

    input_len = int(inputs["input_ids"].shape[-1])
    decoded = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return _clean_caption(decoded)


def _generate_captions(
    images: torch.Tensor,
    allow_download: bool = True,
    generator: Optional[Callable[[object, object, object], str]] = None,
) -> List[str]:
    image_batch = _ensure_image_batch(images)
    processor, model = _load_caption_backend(allow_download=allow_download)
    generate_one = generator or _generate_caption_for_pil
    captions = []
    for image in image_batch:
        captions.append(_clean_caption(generate_one(processor, model, _image_tensor_to_pil(image))))
    if not captions:
        raise PiDNodeError("PiD Caption Creator received an empty image batch.")
    return captions


class PiDCaptionCreator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "auto_download": ("BOOLEAN", {"default": True}),
                "preview": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "caption")
    FUNCTION = "create"
    CATEGORY = "PiD"

    def create(self, image, auto_download: bool = True, preview: str = ""):
        del preview
        captions = _generate_captions(image, allow_download=bool(auto_download))
        text = "\n".join(captions)
        return {"ui": {"text": [text]}, "result": (text, text)}


NODE_CLASS_MAPPINGS = {
    "PiDCaptionCreator": PiDCaptionCreator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDCaptionCreator": "PiD Caption Creator",
}
