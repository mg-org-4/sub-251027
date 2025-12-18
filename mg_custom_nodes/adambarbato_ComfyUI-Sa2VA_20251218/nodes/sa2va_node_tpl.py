# Sa2VA Node for ComfyUI - Segment Anything 2 Video Assistant
# Supports both text generation and segmentation mask output
# Based on ByteDance/Sa2VA models that combine SAM2 with LLaVA

import torch
import numpy as np
import os
import gc
import threading
import time
from contextlib import nullcontext
from PIL import Image
from .. import be_quiet  # Import global config from root __init__.py


class Sa2VANodeTpl:
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_name = None  # Track the currently loaded model
        self._download_cancelled = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    [
                        "ByteDance/Sa2VA-InternVL3-2B",
                        "ByteDance/Sa2VA-InternVL3-8B",
                        "ByteDance/Sa2VA-InternVL3-14B",
                        "ByteDance/Sa2VA-Qwen2_5-VL-3B",
                        "ByteDance/Sa2VA-Qwen2_5-VL-7B",
                        "ByteDance/Sa2VA-Qwen3-VL-4B",
                    ],
                    {"default": "ByteDance/Sa2VA-Qwen3-VL-4B"},
                ),
                "image": ("IMAGE",),  # Regular ComfyUI image input
                "mask_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0},
                ),  # Binary threshold
                "use_8bit_quantization": (
                    "BOOLEAN",
                    {"default": False},
                ),  # Enable 8-bit quantization with bitsandbytes
                "use_flash_attn": (
                    "BOOLEAN",
                    {"default": True},
                ),  # Use flash attention for efficiency
                "segmentation_prompt": (
                    "STRING",
                    {
                        "default": "Could you please give me a brief description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.",
                        "multiline": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("LIST", "MASK", "IMAGE")
    RETURN_NAMES = ("text_outputs", "masks", "mask_images")
    FUNCTION = "process_with_sa2va"
    CATEGORY = "Sa2VA"

    def check_transformers_version(self):
        """Check if transformers version supports Sa2VA models."""
        try:
            from transformers import __version__ as transformers_version

            version_parts = transformers_version.split(".")
            major, minor = int(version_parts[0]), int(version_parts[1])

            # Sa2VA models require transformers >= 4.57.0
            if major < 4 or (major == 4 and minor < 57):
                return (
                    False,
                    f"Sa2VA requires transformers >= 4.57.0, found {transformers_version}",
                )
            return True, transformers_version
        except Exception as e:
            return False, f"Error checking transformers version: {e}"

    def install_transformers_upgrade(self):
        """Attempt to upgrade transformers automatically."""
        try:
            import subprocess
            import sys

            print("🔄 Attempting to upgrade transformers...")

            # Try to upgrade transformers
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "transformers>=4.57.0",
                    "--upgrade",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("✅ Transformers upgraded successfully")
                print("🔄 Please restart ComfyUI to use the upgraded version")
                return True
            else:
                print(f"❌ Failed to upgrade transformers: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error upgrading transformers: {e}")
            return False

    def load_model(
        self,
        model_name,
        use_flash_attn=True,
        dtype="auto",
        cache_dir="",
        use_8bit_quantization=False,
    ):
        """Loads the specified Sa2VA model only once and caches it."""
        if (
            self.model is None
            or self.processor is None
            or self.current_model_name != model_name
        ):
            # Clean up any existing model state first
            if self.model is not None:
                try:
                    del self.model
                    self.model = None
                except:
                    pass
            if self.processor is not None:
                try:
                    del self.processor
                    self.processor = None
                except:
                    pass
            self.current_model_name = None

            # Clear CUDA cache before loading new model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            if not be_quiet:
                print(f"🔄 Loading Sa2VA Model: {model_name}")

            # Check transformers version
            version_ok, version_info = self.check_transformers_version()
            if not version_ok:
                print(f"❌ {version_info}")
                print("💡 Attempting automatic upgrade...")

                if self.install_transformers_upgrade():
                    print("⚠️  Restart ComfyUI required for the upgrade to take effect")
                    return False
                else:
                    print(
                        "💡 Manual upgrade required: pip install transformers>=4.57.0 --upgrade"
                    )
                    return False

            # Determine cache directory
            effective_cache_dir = None
            if cache_dir and cache_dir.strip():
                effective_cache_dir = cache_dir.strip()
                if not be_quiet:
                    print(f"   Using custom cache directory: {effective_cache_dir}")
            else:
                # Use local cache in the ComfyUI-Sa2VA folder
                import os

                # Get the directory of this file (Sa2VA nodes folder)
                current_dir = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
                effective_cache_dir = os.path.join(
                    current_dir, ".cache", "huggingface", "hub"
                )

                # Create the cache directory if it doesn't exist
                os.makedirs(effective_cache_dir, exist_ok=True)

                if not be_quiet:
                    print(f"   Using local ComfyUI-Sa2VA cache: {effective_cache_dir}")

            # Handle dtype conversion with proper warnings
            # Resolve target dtype robustly to reduce memory while maintaining compatibility
            auto_selected = False
            if dtype == "auto":
                auto_selected = True
                if torch.cuda.is_available():
                    # Prefer bf16 if supported, else fp16; on CPU stick to fp32
                    if (
                        hasattr(torch.cuda, "is_bf16_supported")
                        and torch.cuda.is_bf16_supported()
                    ):
                        resolved_dtype = torch.bfloat16
                    else:
                        resolved_dtype = torch.float16
                else:
                    resolved_dtype = torch.float32
                if not be_quiet:
                    print(
                        f"   Auto-selected dtype: {resolved_dtype} (based on device capabilities)"
                    )
            else:
                # Map explicit dtype request
                dtype_map = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }
                resolved_dtype = dtype_map.get(str(dtype), torch.float32)
                if not be_quiet:
                    print(f"   Target dtype for model: {resolved_dtype}")

            try:
                # Import here to catch missing dependencies
                from transformers import AutoProcessor, AutoModel

                # Build model loading arguments
                model_kwargs = {
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": True,
                }

                # Add cache directory if specified
                if effective_cache_dir:
                    model_kwargs["cache_dir"] = effective_cache_dir

                # Add 8-bit quantization if requested
                if use_8bit_quantization:
                    try:
                        import bitsandbytes as bnb
                        from transformers import BitsAndBytesConfig

                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_enable_fp32_cpu_offload=True,
                        )
                        model_kwargs["quantization_config"] = quantization_config
                        if not be_quiet:
                            print("   Using 8-bit quantization with bitsandbytes")
                    except ImportError:
                        if not be_quiet:
                            print(
                                "   Warning: bitsandbytes not available, skipping 8-bit quantization"
                            )
                            print("   Install with: pip install bitsandbytes")

                # Add flash attention if available and requested
                if use_flash_attn:
                    try:
                        import flash_attn

                        model_kwargs["use_flash_attn"] = True
                        if not be_quiet:
                            print("   Using Flash Attention")
                    except ImportError:
                        if not be_quiet:
                            print(
                                "   Flash Attention not available, continuing without it"
                            )
                            print("   Install with: pip install flash-attn")
                        # Don't add flash_attn to model_kwargs if not available
                else:
                    if not be_quiet:
                        print("   Flash Attention disabled by user")

                # Use resolved dtype for load to reduce memory (skip if using quantization)
                if resolved_dtype is not None and not use_8bit_quantization:
                    model_kwargs["torch_dtype"] = resolved_dtype

                # Load model with enhanced progress and cancellation support
                print("🔄 Starting model download/load...")
                print("   Note: Large models may take several minutes to download")

                # Check if ComfyUI cancellation is available
                def is_cancelled():
                    try:
                        # Try to access ComfyUI's execution state
                        import execution

                        return (
                            execution.current_task is not None
                            and execution.current_task.cancelled
                        )
                    except:
                        try:
                            # Alternative ComfyUI cancellation check
                            import model_management

                            return model_management.processing_interrupted()
                        except:
                            return False

                # Enhanced download with cancellable snapshot_download and repo size summary
                try:
                    from huggingface_hub import HfApi, snapshot_download
                    from huggingface_hub.utils import tqdm as hub_tqdm

                    # Print repo size summary to set expectations
                    try:
                        api = HfApi()
                        info = api.repo_info(
                            model_name, repo_type="model", files_metadata=True
                        )
                        sizes = []
                        file_entries = []
                        for s in getattr(info, "siblings", []):
                            sz = getattr(s, "size", None)
                            if sz is None:
                                lfs = getattr(s, "lfs", None)
                                sz = (
                                    getattr(lfs, "size", None)
                                    if lfs is not None
                                    else None
                                )
                            if isinstance(sz, int) and sz > 0:
                                sizes.append(sz)
                                file_entries.append(
                                    (
                                        getattr(
                                            s, "rfilename", getattr(s, "path", "file")
                                        ),
                                        sz,
                                    )
                                )
                        total_bytes = sum(sizes)
                        if total_bytes > 0:
                            gb = total_bytes / (1024**3)
                            print(
                                f"   Estimated total download size: {gb:.2f} GB across {len(sizes)} files"
                            )
                            largest = sorted(
                                file_entries, key=lambda x: x[1], reverse=True
                            )[:5]
                            if largest:
                                print("   Largest files:")
                                for name, sz in largest:
                                    print(f"     • {name}: {sz / (1024**2):.1f} MB")
                    except Exception as e:
                        if not be_quiet:
                            print(f"   Could not determine repo size: {e}")

                    class CancellableTqdm(hub_tqdm):
                        def update(self, n=1):
                            if is_cancelled():
                                raise KeyboardInterrupt("Download cancelled by user")
                            return super().update(n)

                    local_dir = snapshot_download(
                        repo_id=model_name,
                        cache_dir=effective_cache_dir if effective_cache_dir else None,
                        resume_download=True,
                        local_files_only=False,
                        tqdm_class=CancellableTqdm,
                    )

                    # Load the model from the local directory to avoid extra network calls
                    model_kwargs_local = dict(model_kwargs)
                    model_kwargs_local["local_files_only"] = True
                    model_kwargs_local.pop("cache_dir", None)
                    self.model = AutoModel.from_pretrained(
                        local_dir, **model_kwargs_local
                    ).eval()
                    print("✅ Model files downloaded and loaded from cache")

                except KeyboardInterrupt:
                    print("\n⚠️ Model download was cancelled")
                    return False
                except Exception as e:
                    if not be_quiet:
                        print(f"   Enhanced download failed: {e}")
                        print("   Using standard download...")
                    self.model = AutoModel.from_pretrained(
                        model_name, **model_kwargs
                    ).eval()

                # Place model on the appropriate device and dtype for lower memory
                target_device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )

                # Skip device/dtype conversion if using 8-bit quantization (already handled)
                if not use_8bit_quantization:
                    try:
                        # Move to device first, then handle dtype if needed
                        self.model = self.model.to(device=target_device)
                        # Only convert dtype if it's different from current and supported
                        if (
                            hasattr(self.model, "dtype")
                            and self.model.dtype != resolved_dtype
                        ):
                            try:
                                self.model = self.model.to(dtype=resolved_dtype)
                            except Exception as e:
                                if not be_quiet:
                                    print(
                                        f"   Note: Could not convert to {resolved_dtype}, keeping original dtype: {e}"
                                    )
                    except Exception as e:
                        if not be_quiet:
                            print(f"   Warning: Model placement failed: {e}")

                if not be_quiet:
                    if use_8bit_quantization:
                        print(
                            f"   Model loaded with 8-bit quantization on {target_device}"
                        )
                    else:
                        actual_dtype = (
                            getattr(self.model, "dtype", "unknown")
                            if hasattr(self.model, "dtype")
                            else "unknown"
                        )
                        print(
                            f"   Model moved to {target_device} with dtype {actual_dtype}"
                        )

                # Load processor (from local_dir if available to avoid refetch)
                processor_kwargs = {"trust_remote_code": True, "use_fast": False}
                if effective_cache_dir:
                    processor_kwargs["cache_dir"] = effective_cache_dir

                processor_source = local_dir if "local_dir" in locals() else model_name
                self.processor = AutoProcessor.from_pretrained(
                    processor_source, **processor_kwargs
                )

                self.current_model_name = model_name

                # Complete device/dtype setup and verify model is ready
                if not use_8bit_quantization:
                    try:
                        # Move to device first, then handle dtype if needed
                        self.model = self.model.to(device=target_device)
                        # Only convert dtype if it's different from current and supported
                        if (
                            hasattr(self.model, "dtype")
                            and self.model.dtype != resolved_dtype
                        ):
                            try:
                                self.model = self.model.to(dtype=resolved_dtype)
                            except Exception as e:
                                if not be_quiet:
                                    print(
                                        f"   Note: Could not convert to {resolved_dtype}, keeping original dtype: {e}"
                                    )
                    except Exception as e:
                        if not be_quiet:
                            print(f"   Warning: Model placement failed: {e}")

                if not be_quiet:
                    if use_8bit_quantization:
                        print(
                            f"   Model loaded with 8-bit quantization on {target_device}"
                        )
                    else:
                        actual_dtype = (
                            getattr(self.model, "dtype", "unknown")
                            if hasattr(self.model, "dtype")
                            else "unknown"
                        )
                        print(
                            f"   Model moved to {target_device} with dtype {actual_dtype}"
                        )

                # Load processor
                processor_kwargs = {"trust_remote_code": True, "use_fast": False}
                if effective_cache_dir:
                    processor_kwargs["cache_dir"] = effective_cache_dir

                self.processor = AutoProcessor.from_pretrained(
                    model_name, **processor_kwargs
                )

                self.current_model_name = model_name

                if not be_quiet:
                    print(f"✅ Sa2VA Model Successfully Loaded: {model_name}")

            except ImportError as e:
                error_str = str(e)
                if "flash_attn" in error_str:
                    print(f"❌ Flash Attention dependency missing: {e}")
                    print("💡 Retrying model load without Flash Attention...")
                    # Remove flash_attn requirement and retry
                    model_kwargs.pop("use_flash_attn", None)
                    try:
                        self.model = AutoModel.from_pretrained(
                            model_name, **model_kwargs
                        ).eval()
                        print("✅ Model loaded successfully without Flash Attention")
                    except Exception as retry_e:
                        print(
                            f"❌ Model loading failed even without Flash Attention: {retry_e}"
                        )
                        return False
                else:
                    print(f"❌ Missing dependencies for Sa2VA model: {e}")
                    print("💡 Try installing: pip install transformers>=4.57.0")
                    return False
            except Exception as e:
                print(f"❌ Error loading Sa2VA model {model_name}: {e}")
                if "qwen_vl_utils" in str(e).lower():
                    print("💡 Missing qwen_vl_utils dependency")
                    print("   Install it with: pip install qwen_vl_utils")
                elif "qwen3_vl" in str(e).lower():
                    print(
                        "💡 This error suggests your transformers version doesn't support Qwen3-VL"
                    )
                    print("   Try upgrading: pip install transformers>=4.57.0")
                elif "trust_remote_code" in str(e).lower():
                    print(
                        "💡 This model requires trust_remote_code=True (enabled by default)"
                    )
                return False

        return True

    def process_single_image(
        self, image, text_prompt, segmentation_mode=False, segmentation_prompt=""
    ):
        """Process a single image with Sa2VA model."""
        try:
            # Use segmentation prompt if segmentation mode is enabled
            prompt = (
                segmentation_prompt
                if segmentation_mode and segmentation_prompt
                else text_prompt
            )

            # Ensure image is PIL Image
            if isinstance(image, str) and os.path.exists(image):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                # Try to convert tensor/array to PIL Image
                if hasattr(image, "numpy"):
                    image_np = image.numpy()
                elif isinstance(image, np.ndarray):
                    image_np = image
                else:
                    print(f"⚠️ Unsupported image type: {type(image)}")
                    return "Error: Unsupported image format", []

                # Convert numpy array to PIL Image
                if image_np.dtype != np.uint8:
                    image_np = (image_np * 255).astype(np.uint8)
                if len(image_np.shape) == 3 and image_np.shape[0] in [1, 3]:
                    image_np = np.transpose(image_np, (1, 2, 0))
                image = Image.fromarray(image_np)

            # Prepare input dictionary for Sa2VA
            input_dict = {
                "image": image,
                "text": f"<image>{prompt}",
                "past_text": "",
                "mask_prompts": None,
                "processor": self.processor,
            }

            # Forward pass through Sa2VA model
            with torch.no_grad():
                return_dict = self.model.predict_forward(**input_dict)

            # Extract text output
            text_output = return_dict.get("prediction", "")

            # Extract segmentation masks if available
            masks = return_dict.get("prediction_masks", [])

            return text_output, masks

        except Exception as e:
            error_msg = f"Error processing image: {e}"
            print(f"❌ {error_msg}")
            return error_msg, []

    def process_video_frames(
        self,
        frame_paths,
        text_prompt,
        segmentation_mode=False,
        segmentation_prompt="",
        max_frames=5,
    ):
        """Process video frames with Sa2VA model."""
        try:
            # Limit and sample frames if necessary
            if len(frame_paths) > max_frames:
                step = max(1, (len(frame_paths) - 1) // (max_frames - 1))
                sampled_paths = (
                    [frame_paths[0]] + frame_paths[1:-1][::step] + [frame_paths[-1]]
                )
                frame_paths = sampled_paths[:max_frames]

            # Use segmentation prompt if segmentation mode is enabled
            prompt = (
                segmentation_prompt
                if segmentation_mode and segmentation_prompt
                else text_prompt
            )

            # Prepare input dictionary for video processing
            input_dict = {
                "video": frame_paths,
                "text": f"<image>{prompt}",
                "past_text": "",
                "mask_prompts": None,
                "processor": self.processor,
            }

            # Forward pass through Sa2VA model
            with torch.no_grad():
                return_dict = self.model.predict_forward(**input_dict)

            # Extract text output
            text_output = return_dict.get("prediction", "")

            # Extract segmentation masks if available
            masks = return_dict.get("prediction_masks", [])

            return text_output, masks

        except Exception as e:
            error_msg = f"Error processing video frames: {e}"
            print(f"❌ {error_msg}")
            return error_msg, []

    def convert_masks_to_comfyui(
        self,
        masks,
        input_height,
        input_width,
        output_format="both",
        normalize=True,
        threshold=0.5,
        apply_threshold=False,
        batchify_mask=True,
    ):
        """
        Convert Sa2VA numpy masks to ComfyUI format.
        """
        try:
            # Handle None input gracefully
            if masks is None or len(masks) == 0:
                if not be_quiet:
                    print("⚠️ No masks to convert, creating blank mask")
                # Return blank mask sized to input
                empty_mask = (
                    torch.zeros((1, input_height, input_width), dtype=torch.float32)
                    if batchify_mask
                    else torch.zeros((input_height, input_width), dtype=torch.float32)
                )
                empty_image = torch.zeros(
                    (1, input_height, input_width, 3), dtype=torch.float32
                )
                return empty_mask, empty_image

            comfyui_masks = []
            image_tensors = []

            for i, mask in enumerate(masks):
                if mask is None:
                    continue

                try:
                    # Convert mask to numpy array if it's not already
                    if isinstance(mask, torch.Tensor):
                        mask_np = mask.detach().cpu().numpy()
                    elif isinstance(mask, np.ndarray):
                        mask_np = mask.copy()
                    elif isinstance(mask, (list, tuple)):
                        mask_np = np.array(mask)
                    else:
                        continue

                    # Handle different mask dimensions
                    if len(mask_np.shape) == 4:  # (batch, channel, height, width)
                        mask_np = mask_np[0, 0]
                    elif len(mask_np.shape) == 3:
                        if mask_np.shape[0] == 1:  # (1, height, width)
                            mask_np = mask_np[0]
                        elif mask_np.shape[2] == 1:  # (height, width, 1)
                            mask_np = mask_np[:, :, 0]
                        elif (
                            mask_np.shape[0] < mask_np.shape[1]
                            and mask_np.shape[0] < mask_np.shape[2]
                        ):
                            mask_np = mask_np[0]
                        else:
                            mask_np = mask_np[:, :, 0]

                    # Ensure we have a 2D mask
                    if len(mask_np.shape) != 2:
                        continue

                    # Handle empty or invalid masks
                    if mask_np.size == 0:
                        continue

                    # Convert to float for processing
                    if mask_np.dtype == bool:
                        mask_np = mask_np.astype(np.float32)
                    elif not np.issubdtype(mask_np.dtype, np.floating):
                        mask_np = mask_np.astype(np.float32)

                    # Handle NaN and infinite values
                    if np.any(np.isnan(mask_np)) or np.any(np.isinf(mask_np)):
                        mask_np = np.nan_to_num(
                            mask_np, nan=0.0, posinf=1.0, neginf=0.0
                        )

                    # Normalize to 0-1 range if requested
                    if normalize:
                        mask_min, mask_max = mask_np.min(), mask_np.max()
                        if mask_max > mask_min:
                            mask_np = (mask_np - mask_min) / (mask_max - mask_min)
                        else:
                            mask_np = (
                                np.ones_like(mask_np)
                                if mask_min > 0
                                else np.zeros_like(mask_np)
                            )

                    # Apply threshold if requested
                    if apply_threshold:
                        mask_np = (mask_np > threshold).astype(np.float32)

                    # Convert to ComfyUI mask format (torch tensor)
                    if output_format in ["comfyui_mask", "both"]:
                        comfyui_mask = torch.from_numpy(mask_np).float()
                        while comfyui_mask.ndim > 2:
                            comfyui_mask = comfyui_mask.squeeze(0)
                        if comfyui_mask.ndim == 2:
                            comfyui_masks.append(comfyui_mask)

                    # Convert to ComfyUI IMAGE tensor [H, W, 3] per mask (normalized 0-1)
                    rgb_np = np.stack([mask_np, mask_np, mask_np], axis=-1)
                    rgb_np = np.clip(rgb_np, 0.0, 1.0).astype(np.float32)
                    image_tensors.append(torch.from_numpy(rgb_np))

                except Exception as e:
                    if not be_quiet:
                        print(f"❌ Error processing mask {i}: {e}")
                    continue

            # Handle case where no masks were successfully processed
            if not comfyui_masks:
                empty_mask = (
                    torch.zeros((1, input_height, input_width), dtype=torch.float32)
                    if batchify_mask
                    else torch.zeros((input_height, input_width), dtype=torch.float32)
                )
                empty_image = torch.zeros(
                    (1, input_height, input_width, 3), dtype=torch.float32
                )
                return empty_mask, empty_image

            # Build final MASK tensor: [B, H, W] if batchify_mask else [H, W]
            final_comfyui_masks = None
            if comfyui_masks:
                try:
                    masks_2d = []
                    for m in comfyui_masks:
                        while m.ndim > 2:
                            m = m.squeeze(0)
                        if m.ndim == 2:
                            masks_2d.append(m.float())

                    if not masks_2d:
                        final_comfyui_masks = (
                            torch.zeros(
                                (1, input_height, input_width), dtype=torch.float32
                            )
                            if batchify_mask
                            else torch.zeros(
                                (input_height, input_width), dtype=torch.float32
                            )
                        )
                    else:
                        if batchify_mask:
                            first_hw = masks_2d[0].shape
                            aligned = [t for t in masks_2d if t.shape == first_hw]
                            final_comfyui_masks = (
                                torch.stack(aligned, dim=0).float()
                                if aligned
                                else torch.zeros(
                                    (1, input_height, input_width), dtype=torch.float32
                                )
                            )
                        else:
                            final_comfyui_masks = masks_2d[0]
                except Exception as e:
                    if not be_quiet:
                        print(f"⚠️ Error processing ComfyUI masks: {e}")
                    final_comfyui_masks = (
                        torch.zeros((1, input_height, input_width), dtype=torch.float32)
                        if batchify_mask
                        else torch.zeros(
                            (input_height, input_width), dtype=torch.float32
                        )
                    )
            else:
                final_comfyui_masks = (
                    torch.zeros((1, input_height, input_width), dtype=torch.float32)
                    if batchify_mask
                    else torch.zeros((input_height, input_width), dtype=torch.float32)
                )

            # Build IMAGE batch tensor [B, H, W, 3]
            if image_tensors:
                try:
                    first_hw = image_tensors[0].shape[:2]
                    aligned = [t for t in image_tensors if t.shape[:2] == first_hw]
                    if not aligned:
                        final_image_tensor = torch.zeros(
                            (1, input_height, input_width, 3), dtype=torch.float32
                        )
                    else:
                        final_image_tensor = torch.stack(aligned, dim=0).float()
                except Exception as e:
                    if not be_quiet:
                        print(f"⚠️ Error stacking IMAGE tensors: {e}")
                    final_image_tensor = torch.zeros(
                        (1, input_height, input_width, 3), dtype=torch.float32
                    )
            else:
                final_image_tensor = torch.zeros(
                    (1, input_height, input_width, 3), dtype=torch.float32
                )

            return final_comfyui_masks, final_image_tensor

        except Exception as e:
            if not be_quiet:
                print(f"❌ Error converting masks: {e}")
            empty_mask = (
                torch.zeros((1, input_height, input_width), dtype=torch.float32)
                if batchify_mask
                else torch.zeros((input_height, input_width), dtype=torch.float32)
            )
            empty_image = torch.zeros(
                (1, input_height, input_width, 3), dtype=torch.float32
            )
            return empty_mask, empty_image

    def process_with_sa2va(
        self,
        model_name,
        image,
        mask_threshold,
        segmentation_prompt,
        use_8bit_quantization,
        use_flash_attn,
    ):
        """Main processing function for Sa2VA model."""
        # Set default values for hidden parameters
        text_prompt = "Please describe the image."
        segmentation_mode = True
        video_mode = False
        max_frames = 5
        dtype = "auto"
        use_inference_mode = True
        use_autocast = True
        autocast_dtype = "bfloat16"
        free_gpu_after = True
        unload_model_after = False
        offload_to_cpu = False
        offload_input_to_cpu = True
        cache_dir = ""
        output_mask_format = "both"
        normalize_masks = True
        apply_mask_threshold = False
        batchify_mask = True

        try:
            # Load model if not already loaded
            model_loaded = self.load_model(
                model_name, use_flash_attn, dtype, cache_dir, use_8bit_quantization
            )
            if not model_loaded:
                error_msg = f"Failed to load Sa2VA model: {model_name}. Check console for details."
                print(f"❌ {error_msg}")
                # Return valid structure to prevent downstream errors
                return ([error_msg], [])

            # Validate inputs
            if image is None:
                error_msg = "No image provided"
                print(f"⚠️ {error_msg}")
                return ([error_msg], [])

            if not be_quiet:
                print(f"🔄 Processing image | Segmentation: {segmentation_mode}")

            # Convert ComfyUI image tensor to PIL Image
            if hasattr(image, "shape") and len(image.shape) == 4:
                # ComfyUI image format: (batch, height, width, channels)
                img_t = image[0]
            elif hasattr(image, "shape") and len(image.shape) == 3:
                # Single image: (height, width, channels)
                img_t = image
            else:
                error_msg = f"Unsupported image format: {type(image)}"
                print(f"❌ {error_msg}")
                return ([error_msg], [])

            # Offload image tensor to CPU and release GPU memory promptly
            if isinstance(img_t, torch.Tensor):
                try:
                    if offload_input_to_cpu and img_t.is_cuda:
                        img_cpu = img_t.detach().to("cpu")
                        del img_t
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            if hasattr(torch.cuda, "ipc_collect"):
                                torch.cuda.ipc_collect()
                        img_t = img_cpu
                    else:
                        img_t = img_t.detach().cpu()
                except Exception:
                    # Fallback to plain .cpu()
                    img_t = img_t.cpu()
                image_np = img_t.numpy()
                # Help GC promptly
                del img_t
            else:
                error_msg = f"Unsupported image tensor type: {type(image)}"
                print(f"❌ {error_msg}")
                return ([error_msg], [])

            # Convert to PIL Image
            if image_np.dtype != "uint8":
                image_np = (image_np * 255).astype("uint8")

            pil_image = Image.fromarray(image_np)

            # Process the single image with memory-friendly contexts
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if use_autocast and device == "cuda":
                if autocast_dtype == "float16":
                    _amp_dtype = torch.float16
                elif autocast_dtype == "bfloat16" or autocast_dtype == "auto":
                    _amp_dtype = torch.bfloat16
                else:
                    _amp_dtype = torch.bfloat16
                autocast_ctx = torch.cuda.amp.autocast(dtype=_amp_dtype)
            else:
                autocast_ctx = nullcontext()

            inference_ctx = (
                torch.inference_mode() if use_inference_mode else nullcontext()
            )

            with inference_ctx:
                with autocast_ctx:
                    text_output, masks = self.process_single_image(
                        pil_image, text_prompt, segmentation_mode, segmentation_prompt
                    )

            text_outputs = [text_output]
            all_masks = masks if masks else []

            # Get input dimensions for mask sizing
            h, w = int(image_np.shape[0]), int(image_np.shape[1])

            # Always ensure we have masks in segmentation mode
            if segmentation_mode and len(all_masks) == 0:
                blank_mask = np.zeros((h, w), dtype=np.float32)
                all_masks = [blank_mask]

            # Convert masks to ComfyUI format
            comfyui_masks, mask_images = self.convert_masks_to_comfyui(
                all_masks,
                h,
                w,
                output_mask_format,
                normalize_masks,
                mask_threshold,
                apply_mask_threshold,
                batchify_mask,
            )

            if not be_quiet:
                print(
                    f"✅ Sa2VA Processing Complete: {len(text_outputs)} text outputs, {len(all_masks)} masks"
                )
                if dtype != "auto":
                    print(f"   Note: Model converted from native precision to {dtype}")
                if comfyui_masks is not None:
                    print(f"   ComfyUI mask shape: {comfyui_masks.shape}")
                if mask_images is not None:
                    print(f"   IMAGE tensor shape: {mask_images.shape}")

            # Ensure we always return valid lists, never empty lists that could cause indexing errors
            if not text_outputs:
                text_outputs = ["Processing completed but no text was generated"]

            # Ensure text_outputs is never empty to prevent index errors downstream
            if len(text_outputs) == 0:
                text_outputs = ["Error: No output generated"]

            # Post-run memory management
            try:
                if torch.cuda.is_available():
                    if free_gpu_after:
                        torch.cuda.empty_cache()
                        if hasattr(torch.cuda, "ipc_collect"):
                            torch.cuda.ipc_collect()
                    if unload_model_after:
                        if offload_to_cpu and self.model is not None:
                            try:
                                # Ensure model is properly moved to CPU with all parameters
                                self.model = self.model.cpu()
                                # Force all parameters to CPU
                                for param in self.model.parameters():
                                    param.data = param.data.cpu()
                                for buffer in self.model.buffers():
                                    buffer.data = buffer.data.cpu()
                                if not be_quiet:
                                    print("   Model offloaded to CPU")
                            except Exception as _e:
                                if not be_quiet:
                                    print(f"   Offload to CPU failed: {_e}")
                                # Fallback to full unload if CPU offload fails
                                try:
                                    del self.model
                                except:
                                    pass
                                self.model = None
                                self.processor = None
                                self.current_model_name = None
                        else:
                            # Fully unload model
                            try:
                                del self.model
                            except:
                                pass
                            try:
                                del self.processor
                            except:
                                pass
                            self.model = None
                            self.processor = None
                            self.current_model_name = None
                            if not be_quiet:
                                print("   Model unloaded")
                        torch.cuda.empty_cache()
                        if hasattr(torch.cuda, "ipc_collect"):
                            torch.cuda.ipc_collect()
                # Always collect Python GC
                gc.collect()
            except Exception as _e:
                if not be_quiet:
                    print(f"⚠️ Memory management step encountered an issue: {_e}")

            return (text_outputs, comfyui_masks, mask_images)

        except Exception as e:
            error_msg = f"Sa2VA processing failed: {e}"
            print(f"❌ {error_msg}")
            import traceback

            traceback.print_exc()

            # Always return valid structure to prevent downstream crashes
            # Get fallback dimensions
            try:
                if hasattr(image, "shape") and len(image.shape) >= 2:
                    if len(image.shape) == 4:
                        fb_h, fb_w = image.shape[1], image.shape[2]
                    elif len(image.shape) == 3:
                        fb_h, fb_w = image.shape[0], image.shape[1]
                    else:
                        fb_h, fb_w = 64, 64
                else:
                    fb_h, fb_w = 64, 64
            except:
                fb_h, fb_w = 64, 64

            empty_mask = (
                torch.zeros((1, fb_h, fb_w), dtype=torch.float32)
                if batchify_mask
                else torch.zeros((fb_h, fb_w), dtype=torch.float32)
            )
            empty_image = torch.zeros((1, fb_h, fb_w, 3), dtype=torch.float32)
            return ([f"Error: {error_msg}"], empty_mask, empty_image)
