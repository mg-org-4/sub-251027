import os
import sys
import torch
import folder_paths
from PIL import Image
import numpy as np

from .cai_utils import download_cai

# Add the parent directory to the Python path so we can import from easycontrol
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from easycontrol.pipeline import FluxPipeline
from easycontrol.transformer_flux import FluxTransformer2DModel
from easycontrol.lora_helper import set_single_lora, set_multi_lora, unset_lora
from huggingface_hub import login

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

from transformers import T5EncoderModel


class EasyControlLoadFlux:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hf_token": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {"load_8bit": ("BOOLEAN", {"default": True}), "cpu_offload": ("BOOLEAN", {"default": True})}
        }
    
    RETURN_TYPES = ("EASYCONTROL_PIPE", "EASYCONTROL_TRANSFORMER")
    FUNCTION = "load_model"
    CATEGORY = "EasyControl"

    def load_model(self, load_8bit, cpu_offload, hf_token=None):
        login(token=hf_token)
        base_path = "black-forest-labs/FLUX.1-dev"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = folder_paths.get_folder_paths("diffusers")[0]
        print(cache_dir)
        if load_8bit:
            quant_config_t5 = TransformersBitsAndBytesConfig(load_in_8bit=True,)
            quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True,)
        else:
            quant_config_t5 = None
            quant_config = None
            
        text_encoder_2 = T5EncoderModel.from_pretrained(
            base_path,
            subfolder="text_encoder_2",
            quantization_config=quant_config_t5,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            base_path, 
            subfolder="transformer",
            torch_dtype=torch.bfloat16, 
            device=device,
            cache_dir=cache_dir,
            quantization_config=quant_config,
        )
        
        pipe = FluxPipeline.from_pretrained(base_path, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=torch.bfloat16, device=device, cache_dir=cache_dir)
        
        if cpu_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)
        
        return (pipe, transformer)

class EasyControlLoadLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer": ("EASYCONTROL_TRANSFORMER", ),
                "lora_name": (folder_paths.get_filename_list("loras"), ),
                "lora_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "cond_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
            },
        }
    
    RETURN_TYPES = ("EASYCONTROL_TRANSFORMER",)
    FUNCTION = "load_lora"
    CATEGORY = "EasyControl"

    def load_lora(self, transformer, lora_name, lora_weight, cond_size):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        set_single_lora(transformer, lora_path, lora_weights=[lora_weight], cond_size=cond_size)
        return (transformer,)
    
# New Node: FLUX Style LoRA Loader
class EasyControlLoadStyleLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("EASYCONTROL_PIPE",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "lora_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("EASYCONTROL_PIPE",)
    FUNCTION = "load_lora"
    CATEGORY = "EasyControl"

    def load_lora(self, pipe, lora_name, lora_weight):
        # Get the full path of the LoRA file
        lora_path = folder_paths.get_full_path("loras", lora_name)
        
        # Load LoRA weights
        print(f"Loading FLUX Style LoRA: {lora_name}, Weight: {lora_weight}")
        
        weight_name = lora_name

        # handle offload
        device = next(pipe.transformer.parameters()).device

        # Load LoRA weights
        pipe.load_lora_weights(lora_path, weight_name=weight_name, device=device)
        
        # Fuse LoRA
        # pipe.fuse_lora(lora_weights=[lora_weight])        
        return (pipe,)


class EasyControlLoadStyleLoraFromCivitai:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("EASYCONTROL_PIPE",),
                "lora_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "civitai_model_id": ("STRING", {"default": "", "tooltip": "The ID of the model to download from CivitAI."}),
            }
        }
    
    RETURN_TYPES = ("EASYCONTROL_PIPE",)
    FUNCTION = "load_lora"
    CATEGORY = "EasyControl"

    def load_lora(self, pipe, lora_weight, civitai_model_id):
        civitai_token_id = os.getenv("CIVITAI_TOKEN", "").strip()
        if not civitai_token_id:
            raise RuntimeError("CIVITAI_TOKEN environment variable is not set or empty.")
        
        loras_dir = folder_paths.get_folder_paths("loras")[0]
        
        # Generate temporary filename for downloaded LoRA
        lora_filename = f"tmp_{civitai_model_id or 'downloaded_lora'}.safetensors"
        lora_path = os.path.join(loras_dir, lora_filename)
        
        try:
            file_exists = os.path.exists(lora_path)
            if not file_exists:
                self.download_from_civitai(civitai_model_id, civitai_token_id, lora_path)
            else:
                print(f"LoRA file already exists at {lora_path}, skipping download")

            # Load LoRA weights
            print(f"Loading FLUX Style LoRA: {lora_filename}, Weight: {lora_weight}")
            
            device = next(pipe.transformer.parameters()).device
            pipe.load_lora_weights(lora_path, weight_name=lora_filename, device=device)
            
            return (pipe,)
        except Exception as e:
            raise RuntimeError(f"Failed to load or download LoRA: {str(e)}")
    
    def download_from_civitai(self, model_id, token_id, lora_path):
        print("Downloading LoRA from CivitAI")
        print(f"\tModel ID: {model_id}")
        print(f"\tToken ID: {token_id}")
        print(f"\tSave path: {lora_path}")
        download_cai(model_id, token_id, lora_path)


class EasyControlLoadMultiLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer": ("EASYCONTROL_TRANSFORMER", ),
                "lora_name1": (folder_paths.get_filename_list("loras"), ),
                "lora_weight1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "lora_name2": (folder_paths.get_filename_list("loras"), ),
                "lora_weight2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "cond_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
            },
        }
    
    RETURN_TYPES = ("EASYCONTROL_TRANSFORMER",)
    FUNCTION = "load_multi_lora"
    CATEGORY = "EasyControl"

    def load_multi_lora(self, transformer, lora_name1, lora_weight1, lora_name2, lora_weight2, cond_size):
        lora_path1 = folder_paths.get_full_path("loras", lora_name1)
        lora_path2 = folder_paths.get_full_path("loras", lora_name2)
        
        set_multi_lora(
            transformer, 
            [lora_path1, lora_path2], 
            lora_weights=[[lora_weight1], [lora_weight2]], 
            cond_size=cond_size
        )
        return (transformer,)

class EasyControlGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("EASYCONTROL_PIPE", ),
                "transformer": ("EASYCONTROL_TRANSFORMER", ),
                "prompt": ("STRING", {"multiline": True}),
                "prompt_2": ("STRING", {"multiline": True, "default": ""}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "cond_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "use_zero_init": ("BOOLEAN", {"default": True}),
                "zero_steps": ("INT", {"default": 1, "min": 0, "max": 100}),
            },
            "optional": {
                "spatial_image": ("IMAGE", ),
                "subject_image": ("IMAGE", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "EasyControl"

    def generate(self, pipe, transformer, prompt, prompt_2, height, width, guidance_scale, 
                num_inference_steps, seed, cond_size, use_zero_init, zero_steps, spatial_image=None, subject_image=None):
        # Clear cache before generation
        for name, attn_processor in transformer.attn_processors.items():
            attn_processor.bank_kv.clear()
        
        
        # Prepare spatial images
        spatial_images = []
        print("spatial_image")
        print(spatial_image)
        
        if spatial_image is not None:
            # Convert from tensor or numpy to PIL
            if isinstance(spatial_image, torch.Tensor):
                # Handle single image or batch
                if spatial_image.dim() == 4:  # [batch, height, width, channels]
                    for i in range(spatial_image.shape[0]):
                        img = spatial_image[i].cpu().numpy()
                        spatial_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                        spatial_images.append(spatial_image_pil)
                else:  # [height, width, channels]
                    img = spatial_image.cpu().numpy()
                    spatial_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                    spatial_images.append(spatial_image_pil)
            elif isinstance(spatial_image, np.ndarray):
                spatial_image_pil = Image.fromarray((spatial_image * 255).astype(np.uint8))
                spatial_images.append(spatial_image_pil)
        
        # Prepare subject images
        subject_images = []
        if subject_image is not None:
            # Convert from tensor or numpy to PIL
            if isinstance(subject_image, torch.Tensor):
                # Handle single image or batch
                if subject_image.dim() == 4:  # [batch, height, width, channels]
                    for i in range(subject_image.shape[0]):
                        img = subject_image[i].cpu().numpy()
                        subject_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                        subject_images.append(subject_image_pil)
                else:  # [height, width, channels]
                    img = subject_image.cpu().numpy()
                    subject_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                    subject_images.append(subject_image_pil)
            elif isinstance(subject_image, np.ndarray):
                subject_image_pil = Image.fromarray((subject_image * 255).astype(np.uint8))
                subject_images.append(subject_image_pil)
        
        # Set prompt_2 to None if empty
        if not prompt_2:
            prompt_2 = None

        print("spatial_images")
        print(spatial_images)

        print("subject_images")
        print(subject_images)
        
        # Generate image
        output = pipe(
            prompt=prompt,
            prompt_2=prompt_2,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),
            spatial_images=spatial_images,
            subject_images=subject_images,
            cond_size=cond_size,
            use_zero_init=use_zero_init,
            zero_steps=int(zero_steps)
        )
        
        # Convert PIL image to numpy array, then to torch.Tensor
        if isinstance(output, FluxPipelineOutput):
            image = np.array(output.images[0]) / 255.0
        else:
            image = np.array(output[0]) / 255.0
        
        # Convert numpy array to torch.Tensor
        image = torch.from_numpy(image).float()
        
        # Add batch dimension to make it [batch, height, width, channels]
        if image.dim() == 3:  # [height, width, channels]
            image = image.unsqueeze(0)  # Add batch dimension to make it [1, height, width, channels]
        
        # Clear cache after generation
        for name, attn_processor in transformer.attn_processors.items():
            attn_processor.bank_kv.clear()
        
        return (image,)

