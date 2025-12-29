#IF_MemoCheckpointLoader.py
import os
import torch
import folder_paths
import logging
from diffusers import AutoencoderKL
from diffusers.utils import is_xformers_available
from packaging import version
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from memo.models.unet_2d_condition import UNet2DConditionModel
from memo.models.unet_3d import UNet3DConditionModel
from memo.models.image_proj import ImageProjModel
from memo.models.audio_proj import AudioProjModel
from memo.models.emotion_classifier import AudioEmotionClassifierModel
from memo_model_manager import MemoModelManager

logger = logging.getLogger("memo")

class IF_MemoCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enable_xformers": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL", "VAE", "IMAGE_PROJ", "AUDIO_PROJ", "EMOTION_CLASSIFIER")
    RETURN_NAMES = ("reference_net", "diffusion_net", "vae", "image_proj", "audio_proj", "emotion_classifier")
    FUNCTION = "load_checkpoint"
    CATEGORY = "ImpactFrames💥🎞️/MemoAvatar"

    def __init__(self):
        # Initialize model manager to set up all paths and auxiliary models
        self.model_manager = MemoModelManager()
        self.paths = self.model_manager.get_model_paths()

    def load_checkpoint(self, enable_xformers=True):
        try:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            dtype = torch.float16 if str(device) == "cuda" else torch.float32

            logger.info("Loading models")

            # Fallback download function
            def fallback_download(repo_id, filename, local_dir):
                try:
                    return hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=local_dir,
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    logger.warning(f"Failed to download {filename} from {repo_id}: {e}")
                    return None

            # Load VAE with multiple fallback strategies
            try:
                vae = AutoencoderKL.from_pretrained(
                    self.paths["vae"],
                    use_safetensors=True,
                    torch_dtype=dtype
                ).to(device=device)
            except Exception as e:
                logger.warning(f"Local VAE load failed: {e}. Attempting download.")
                fallback_download(
                    "stabilityai/sd-vae-ft-mse", 
                    "diffusion_pytorch_model.safetensors", 
                    self.paths["vae"]
                )
                vae = AutoencoderKL.from_pretrained(
                    "stabilityai/sd-vae-ft-mse",
                    use_safetensors=True,
                    torch_dtype=dtype
                ).to(device=device)

            # Load reference net
            reference_net = UNet2DConditionModel.from_pretrained(
                self.paths["memo_base"],
                subfolder="reference_net",
                use_safetensors=True
            )
            reference_net.requires_grad_(False)
            reference_net.eval()

            # Load diffusion net
            diffusion_net = UNet3DConditionModel.from_pretrained(
                self.paths["memo_base"],
                subfolder="diffusion_net",
                use_safetensors=True
            )
            diffusion_net.requires_grad_(False)
            diffusion_net.eval()

            # Load projectors
            image_proj = ImageProjModel.from_pretrained(
                self.paths["memo_base"],
                subfolder="image_proj",
                use_safetensors=True
            )
            image_proj.requires_grad_(False)
            image_proj.eval()

            audio_proj = AudioProjModel.from_pretrained(
                self.paths["memo_base"],
                subfolder="audio_proj",
                use_safetensors=True
            )
            audio_proj.requires_grad_(False)
            audio_proj.eval()

            # Enable xformers (Optional)
            if enable_xformers and is_xformers_available():
                try:
                    import xformers
                    xformers_version = version.parse(xformers.__version__)
                    if xformers_version == version.parse("0.0.16"):
                        logger.warning("xFormers 0.0.16 cannot be used for training in some GPUs.")
                    reference_net.enable_xformers_memory_efficient_attention()
                    diffusion_net.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}. Proceeding without xformers.")
            else:
                logger.info("Xformers is not enabled or not available. Proceeding without xformers.")

            # Move models to device
            for model in [reference_net, diffusion_net, image_proj, audio_proj]:
                model.to(device=device, dtype=dtype)

            # Load emotion classifier
            emotion_classifier = AudioEmotionClassifierModel()
            emotion_classifier_path = os.path.join(
                self.paths["memo_base"],
                "misc/audio_emotion_classifier/diffusion_pytorch_model.safetensors"
            )
            emotion_classifier.load_state_dict(load_file(emotion_classifier_path))
            emotion_classifier.to(device=device, dtype=dtype)
            emotion_classifier.eval()

            logger.info(f"Models loaded successfully to {device} with dtype {dtype}")
            return (reference_net, diffusion_net, vae, image_proj, audio_proj, emotion_classifier)

        except Exception as e:
            logger.error(f"Comprehensive model loading error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load models: {str(e)}")

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

NODE_CLASS_MAPPINGS = {
    "IF_MemoCheckpointLoader": IF_MemoCheckpointLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_MemoCheckpointLoader": "IF Memo Checkpoint Loader 🎬"
}