"""Nano Banana 2 image generation helper."""

import time

import torch

from .auth import _load_api_key
from .credits import _log_remaining_credits
from .http import TransientKieError
from .images import _download_image, _image_bytes_to_tensor
from .jobs import _create_task, _poll_task_until_complete
from .log import _log
from .results import _extract_result_urls
from .upload import _image_tensor_to_png_bytes, _truncate_url, _upload_image
from .validation import _validate_image_tensor_batch, _validate_prompt


MODEL_NAME = "nano-banana-2"
ASPECT_RATIO_OPTIONS = [
    "1:1",
    "1:4",
    "1:8",
    "2:3",
    "3:2",
    "3:4",
    "4:1",
    "4:3",
    "4:5",
    "5:4",
    "8:1",
    "9:16",
    "16:9",
    "21:9",
    "auto",
]
RESOLUTION_OPTIONS = ["1K", "2K", "4K"]
OUTPUT_FORMAT_OPTIONS = ["jpg", "png"]
PROMPT_MAX_LENGTH = 20000
MAX_IMAGE_COUNT = 14


def _validate_options(aspect_ratio: str, resolution: str, output_format: str) -> None:
    if aspect_ratio not in ASPECT_RATIO_OPTIONS:
        raise RuntimeError("Invalid aspect_ratio. Use the pinned enum options.")
    if resolution not in RESOLUTION_OPTIONS:
        raise RuntimeError("Invalid resolution. Use the pinned enum options.")
    if output_format not in OUTPUT_FORMAT_OPTIONS:
        raise RuntimeError("Invalid output_format. Use the pinned enum options.")


def run_nanobanana2_image_job(
    prompt: str,
    aspect_ratio: str,
    resolution: str,
    output_format: str,
    google_search: bool,
    log: bool,
    poll_interval_s: float,
    timeout_s: int,
    retry_on_fail: bool = True,
    max_retries: int = 2,
    retry_backoff_s: float = 3.0,
    images: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a Nano Banana 2 image generation job."""
    _validate_prompt(prompt, max_length=PROMPT_MAX_LENGTH)
    _validate_options(aspect_ratio, resolution, output_format)
    if not isinstance(google_search, bool):
        raise RuntimeError("google_search must be a boolean.")
    if images is not None:
        images = _validate_image_tensor_batch(images)
    payload = {
        "model": MODEL_NAME,
        "input": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "google_search": google_search,
            "resolution": resolution,
            "output_format": output_format,
        },
    }

    attempts = max_retries + 1 if retry_on_fail else 1
    attempts = max(attempts, 1)
    backoff = retry_backoff_s if retry_backoff_s >= 0 else 0.0

    for attempt in range(1, attempts + 1):
        start_time = time.time()
        try:
            api_key = _load_api_key()

            image_urls: list[str] = []
            if images is not None:
                total_images = images.shape[0]
                if total_images > MAX_IMAGE_COUNT:
                    _log(
                        log,
                        f"More than {MAX_IMAGE_COUNT} images provided ({total_images}); only first {MAX_IMAGE_COUNT} used.",
                    )
                upload_count = min(total_images, MAX_IMAGE_COUNT)
                if upload_count > 0:
                    _log(log, f"Uploading {upload_count} image(s)...")
                for idx in range(upload_count):
                    png_bytes = _image_tensor_to_png_bytes(images[idx])
                    image_url = _upload_image(api_key, png_bytes)
                    image_urls.append(image_url)
                    _log(log, f"Image {idx + 1} upload success: {_truncate_url(image_url)}")

            input_payload = dict(payload["input"])
            input_payload["image_input"] = image_urls
            payload_to_send = {"model": MODEL_NAME, "input": input_payload}

            _log(log, "Creating Nano Banana 2 task...")
            task_id, create_response_text = _create_task(api_key, payload_to_send)
            _log(log, f"createTask response (elapsed={time.time() - start_time:.1f}s): {create_response_text}")
            _log(log, f"Task created with ID {task_id}. Polling for completion...")

            record_data = _poll_task_until_complete(
                api_key,
                task_id,
                poll_interval_s,
                timeout_s,
                log,
                start_time,
            )

            result_urls = _extract_result_urls(record_data)
            _log(log, f"Result URLs: {result_urls}")
            _log(log, f"Downloading result image from {result_urls[0]}...")
            image_bytes = _download_image(result_urls[0])
            image_tensor = _image_bytes_to_tensor(image_bytes)
            _log(log, "Image downloaded and decoded.")
            _log_remaining_credits(log, record_data, api_key, _log)
            return image_tensor
        except TransientKieError:
            if not retry_on_fail or attempt >= attempts:
                raise
            _log(log, f"Retrying (attempt {attempt + 1}/{attempts}) after {backoff}s")
            time.sleep(backoff)

    raise RuntimeError("Nano Banana 2 job failed after retry attempts.")
