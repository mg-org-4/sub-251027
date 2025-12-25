import os
import sys
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL
from transformers import CLIPVisionModelWithProjection
import mediapipe as mp
import folder_paths
from huggingface_hub import snapshot_download

# Add current directory to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.motion_encoder.encoder import MotEncoder
from src.liveportrait.motion_extractor import MotionExtractor
from src.models.pose_guider import PoseGuider
from src.scheduler.scheduler_ddim import DDIMScheduler
from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline
from src.utils.util import crop_face
from diffusers.utils.import_utils import is_xformers_available
from torchvision import transforms

from PIL import Image, ImageFilter

def get_folder_list():
    base_dir = folder_paths.models_dir
    if not os.path.exists(base_dir):
        return ["persona_live"]
        
    candidates = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path):
            if os.path.exists(os.path.join(full_path, "pretrained_weights")):
                candidates.append(name)
    
    if "persona_live" not in candidates:
        candidates.append("persona_live") 
        
    return sorted(candidates)

def download_models_if_missing(root_dir):
    """Auto-download models from HuggingFace if they don't exist."""
    base_model_path = os.path.join(root_dir, "sd-image-variations-diffusers")
    vae_path = os.path.join(root_dir, "sd-vae-ft-mse")
    personalive_path = os.path.join(root_dir, "persona_live")
    
    models_to_download = [
        {
            "repo_id": "lambdalabs/sd-image-variations-diffusers",
            "local_dir": base_model_path,
            "name": "Base Model (sd-image-variations-diffusers)"
        },
        {
            "repo_id": "stabilityai/sd-vae-ft-mse",
            "local_dir": vae_path,
            "name": "VAE (sd-vae-ft-mse)"
        },
        {
            "repo_id": "huaichang/PersonaLive",
            "local_dir": personalive_path,
            "name": "PersonaLive Weights"
        }
    ]
    
    for model_info in models_to_download:
        if not os.path.exists(model_info["local_dir"]) or not os.listdir(model_info["local_dir"]):
            print(f"\n{'='*60}")
            print(f"Downloading {model_info['name']}...")
            print(f"From: {model_info['repo_id']}")
            print(f"To: {model_info['local_dir']}")
            print(f"This may take a while (several GB)...")
            print(f"{'='*60}\n")
            
            try:
                snapshot_download(
                    repo_id=model_info["repo_id"],
                    local_dir=model_info["local_dir"],
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                print(f"\n✓ Successfully downloaded {model_info['name']}\n")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {model_info['name']} from {model_info['repo_id']}: {e}\n"
                    f"Please check your internet connection or download manually."
                )
        else:
            print(f"✓ {model_info['name']} already exists at {model_info['local_dir']}")
    # Move persoanlize 
class PersonaLiveCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_dir": (get_folder_list(), ),
            }
        }

    RETURN_TYPES = ("PERSONALIVE_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "PersonaLive"

    def load_checkpoint(self, model_dir):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weight_dtype = torch.float16 if device == "cuda" else torch.float32
        
        root_dir = os.path.join(folder_paths.models_dir, model_dir)
        
        # Auto-download models if missing
        print(f"\nChecking PersonaLive models in {root_dir}...")
        download_models_if_missing(root_dir)
        
        base_model_path = os.path.join(root_dir, "sd-image-variations-diffusers")
        vae_path = os.path.join(root_dir, "sd-vae-ft-mse")
        personalive_path = os.path.join(root_dir, "persona_live")
        
        image_encoder_path = os.path.join(base_model_path, "image_encoder")

        print(f"\nLoading PersonaLive models from {root_dir}:")
        print(f"  Base Model: {base_model_path}")
        print(f"  VAE: {vae_path}")
        print(f"  Weights: {personalive_path}")

        try:
            vae_model = AutoencoderKL.from_pretrained(vae_path).to(device, dtype=weight_dtype)
        except Exception as e:
             print(f"Failed to load VAE from {vae_path}: {e}")
             print(f"Trying to load VAE from base model: {base_model_path}")
             vae_model = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae").to(device, dtype=weight_dtype)

        reference_unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
        ).to(device=device, dtype=weight_dtype)

        unet_additional_kwargs = {
            "use_inflated_groupnorm": True,
            "unet_use_cross_frame_attention": False,
            "unet_use_temporal_attention": False,
            "use_motion_module": True,
            "motion_module_resolutions": [1, 2, 4, 8],
            "motion_module_mid_block": True,
            "motion_module_decoder_only": False,
            "motion_module_type": "Vanilla",
            "motion_module_kwargs": {
                "num_attention_heads": 8,
                "num_transformer_block": 1,
                "cross_attention_dim": 16,
                "attention_block_types": ["Spatial_Cross", "Spatial_Cross"],
                "temporal_position_encoding": False,
                "temporal_position_encoding_max_len": 32,
                "temporal_attention_dim_div": 1,
            },
            "use_temporal_module": True,
            "temporal_module_type": "Vanilla",
            "temporal_module_kwargs": {
                "num_attention_heads": 8,
                "num_transformer_block": 1,
                "attention_block_types": ["Temporal_Self", "Temporal_Self"],
                "temporal_position_encoding": True,
                "temporal_position_encoding_max_len": 32,
                "temporal_attention_dim_div": 1,
            }
        }

        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs=unet_additional_kwargs,
        ).to(dtype=weight_dtype, device=device)

        motion_encoder = MotEncoder().to(dtype=weight_dtype, device=device).eval()
        pose_guider = PoseGuider().to(device=device, dtype=weight_dtype)
        pose_encoder = MotionExtractor(num_kp=21).to(device=device, dtype=weight_dtype).eval()

        image_enc = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path
        ).to(dtype=weight_dtype, device=device)

        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.02,
            beta_schedule="scaled_linear",
            clip_sample=False,
            steps_offset=1,
            prediction_type="epsilon",
            timestep_spacing="trailing"
        )

        print(f"Loading weights from {personalive_path}")
        
        def load_w(model, filename, strict=True):
             p = os.path.join(personalive_path, "pretrained_weights", "personalive", filename)
             if os.path.exists(p):
                 print(f"Loading {filename} from {p}")
                 model.load_state_dict(torch.load(p, map_location="cpu"), strict=strict)
             else:
                 print(f"WARNING: Could not find {filename} in {personalive_path}")

        load_w(denoising_unet, "denoising_unet.pth", strict=False)
        load_w(reference_unet, "reference_unet.pth", strict=True)
        load_w(motion_encoder, "motion_encoder.pth", strict=True)
        load_w(pose_guider, "pose_guider.pth", strict=True)
        load_w(denoising_unet, "temporal_module.pth", strict=False) 
        load_w(pose_encoder, "motion_extractor.pth", strict=False)

        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()

        pipe = Pose2VideoPipeline(
            vae=vae_model,
            image_encoder=image_enc,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            motion_encoder=motion_encoder,
            pose_encoder=pose_encoder,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        pipe = pipe.to(device)

        return (pipe,)

class PersonaLivePhotoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PERSONALIVE_PIPE",),
                "ref_image": ("IMAGE",),
                "driving_image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate"
    CATEGORY = "PersonaLive"

    def generate(self, pipe, ref_image, driving_image, width, height, guidance_scale, seed):
        device = pipe.device
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        ref_pil = Image.fromarray(np.clip(255. * ref_image[0].cpu().numpy(), 0, 255).astype(np.uint8))
        driving_pil = Image.fromarray(np.clip(255. * driving_image[0].cpu().numpy(), 0, 255).astype(np.uint8))
        
        print(f"Resizing inputs to ({width}, {height})")
        ref_input = ref_pil.resize((width, height))
        driving_input = driving_pil.resize((width, height))

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

        from src.utils.util import crop_face
        
        try:
            ref_patch = crop_face(ref_pil, face_mesh, scale=1.1)
            ref_face = Image.fromarray(ref_patch).convert("RGB")
        except Exception as e:
            print(f"Ref face detection failed: {e}. Using full image.")
            ref_face = ref_input

        try:
            driving_patch = crop_face(driving_pil, face_mesh, scale=1.1)
            driving_face = Image.fromarray(driving_patch).convert("RGB")
        except Exception as e:
            print(f"Driving face detection failed: {e}. Using full image.")
            driving_face = driving_input

        video_length = 4
        
        ori_pose_images = [driving_input] * video_length
        dri_faces = [driving_face] * video_length
        input_ref = ref_input
        input_ref_face = ref_face
        
        print("Running PersonaLive generation...")
        result = pipe(
            ori_pose_images,
            input_ref,
            dri_faces,
            input_ref_face,
            width,
            height,
            video_length,
            num_inference_steps=4,
            guidance_scale=guidance_scale,
            generator=generator,
            temporal_window_size=4,
            temporal_adaptive_step=4,
        )
        
        gen_video = result.videos
        gen_image_tensor = gen_video[:, :, 0:1, :, :] 
        gen_only_pil = Image.fromarray((gen_image_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        
        orig_w, orig_h = ref_pil.size
        if gen_only_pil.size != (orig_w, orig_h):
             final_image = gen_only_pil.resize((orig_w, orig_h))
        else:
             final_image = gen_only_pil
        
        final_tensor = torch.from_numpy(np.array(final_image).astype(np.float32) / 255.0).unsqueeze(0) 
        
        return (final_tensor,)

NODE_CLASS_MAPPINGS = {
    "PersonaLiveCheckpointLoader": PersonaLiveCheckpointLoader,
    "PersonaLivePhotoSampler": PersonaLivePhotoSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PersonaLiveCheckpointLoader": "PersonaLive Checkpoint Loader",
    "PersonaLivePhotoSampler": "PersonaLive Photo Sampler"
}
