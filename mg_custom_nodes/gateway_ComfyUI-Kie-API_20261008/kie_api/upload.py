import hashlib
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .http import TransientKieError, requests


UPLOAD_URL = "https://kieai.redpandaai.co/api/file-stream-upload"
IMAGE_UPLOAD_PATH = "images/user-uploads"
VIDEO_UPLOAD_PATH = "videos/user-uploads"
AUDIO_UPLOAD_PATH = "audio/user-uploads"


def _truncate_url(url: str, max_length: int = 80) -> str:
    return url if len(url) <= max_length else url[:max_length] + "..."


def _build_unique_upload_filename(
    payload_bytes: bytes,
    *,
    default_name: str,
    requested_name: str | None = None,
) -> str:
    name = (requested_name or "").strip() or default_name
    path = Path(name)
    stem = path.stem or Path(default_name).stem or "upload"
    suffix = path.suffix or Path(default_name).suffix
    fingerprint = hashlib.sha1(payload_bytes).hexdigest()[:12]
    timestamp_ms = int(time.time() * 1000)
    return f"{stem}_{timestamp_ms}_{fingerprint}{suffix}"


def _image_tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    if image.dim() != 3 or image.shape[2] != 3:
        raise RuntimeError("Image tensor must have shape [H, W, 3].")
    if image.numel() == 0:
        raise RuntimeError("Image tensor is empty.")

    if image.dtype != torch.uint8:
        working = image.detach().cpu().clamp(0, 1) * 255.0
        working = working.round().to(torch.uint8)
    else:
        working = image.detach().cpu()

    working = working.contiguous()
    h, w, _ = working.shape
    data_bytes = bytes(working.view(-1).tolist())

    try:
        pil_image = Image.frombytes("RGB", (w, h), data_bytes)
    except Exception as exc:
        raise RuntimeError("Failed to convert tensor to image.") from exc

    with BytesIO() as output:
        pil_image.save(output, format="PNG")
        return output.getvalue()


def _upload_image(api_key: str, png_bytes: bytes) -> str:
    filename = _build_unique_upload_filename(png_bytes, default_name="image.png")
    try:
        response = requests.post(
            UPLOAD_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (filename, png_bytes, "image/png")},
            data={"uploadPath": IMAGE_UPLOAD_PATH},
            timeout=120,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to upload image: {exc}") from exc

    if response.status_code == 429 or response.status_code >= 500:
        raise TransientKieError(
            f"upload returned HTTP {response.status_code}: {response.text}",
            status_code=response.status_code,
        )

    payload = response.json()
    if not payload.get("success") or payload.get("code") != 200:
        raise RuntimeError(f"Upload failed: {payload.get('msg')}")

    url = (payload.get("data") or {}).get("downloadUrl")
    if not url:
        raise RuntimeError("Upload response missing downloadUrl.")

    return url


def _upload_video(api_key: str, video_bytes: bytes, filename: str = "video.mp4") -> str:
    if not isinstance(video_bytes, (bytes, bytearray)):
        raise RuntimeError("video_bytes must be raw bytes.")
    if len(video_bytes) == 0:
        raise RuntimeError("video_bytes is empty.")

    if not filename.lower().endswith(".mp4"):
        filename = f"{filename}.mp4"

    unique_filename = _build_unique_upload_filename(
        video_bytes,
        default_name="video.mp4",
        requested_name=filename,
    )

    try:
        response = requests.post(
            UPLOAD_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (unique_filename, video_bytes, "video/mp4")},
            data={"uploadPath": VIDEO_UPLOAD_PATH, "fileName": unique_filename},
            timeout=300,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to upload video: {exc}") from exc

    if response.status_code == 429 or response.status_code >= 500:
        raise TransientKieError(
            f"upload returned HTTP {response.status_code}: {response.text}",
            status_code=response.status_code,
        )

    payload = response.json()
    if not payload.get("success") or payload.get("code") != 200:
        raise RuntimeError(f"Upload failed: {payload.get('msg')}")

    url = (payload.get("data") or {}).get("downloadUrl")
    if not url:
        raise RuntimeError("Upload response missing downloadUrl.")

    return url


def _upload_audio(api_key: str, audio_bytes: bytes, filename: str = "audio.wav") -> str:
    if not isinstance(audio_bytes, (bytes, bytearray)):
        raise RuntimeError("audio_bytes must be raw bytes.")
    if len(audio_bytes) == 0:
        raise RuntimeError("audio_bytes is empty.")

    name = filename or "audio.wav"
    lower = name.lower()
    if lower.endswith(".mp3"):
        content_type = "audio/mpeg"
    elif lower.endswith(".wav"):
        content_type = "audio/wav"
    else:
        content_type = "application/octet-stream"

    unique_name = _build_unique_upload_filename(
        audio_bytes,
        default_name="audio.wav",
        requested_name=name,
    )

    try:
        response = requests.post(
            UPLOAD_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (unique_name, audio_bytes, content_type)},
            data={"uploadPath": AUDIO_UPLOAD_PATH, "fileName": unique_name},
            timeout=300,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to upload audio: {exc}") from exc

    if response.status_code == 429 or response.status_code >= 500:
        raise TransientKieError(
            f"upload returned HTTP {response.status_code}: {response.text}",
            status_code=response.status_code,
        )

    payload = response.json()
    if not payload.get("success") or payload.get("code") != 200:
        raise RuntimeError(f"Upload failed: {payload.get('msg')}")

    url = (payload.get("data") or {}).get("downloadUrl")
    if not url:
        raise RuntimeError("Upload response missing downloadUrl.")

    return url
