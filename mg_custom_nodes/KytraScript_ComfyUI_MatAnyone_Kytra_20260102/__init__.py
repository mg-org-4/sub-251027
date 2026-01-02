import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from tqdm import tqdm

from .inference_core import InferenceCore
from .get_default_model import get_matanyone_model
from .utils import gen_dilate, gen_erosion

#Image Conversion Functions
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

#MatAnyone Node
class MatAnyoneNode:
    """
    MatAnyone Video Matting Node for ComfyUI
    
    Takes a video (as a batch of images) and a mask for the first frame,
    then removes the background from the entire video.
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_frames": ("IMAGE",),  # Batch of video frames as tensor
                "mask": ("MASK",),  # First frame mask as tensor
                "warmup_frames": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "display": "number"
                }),
                "erode_kernel": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
                "dilate_kernel": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
                "bg_red": ("INT", {
                    "default": 120,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
                "bg_green": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
                "bg_blue": ("INT", {
                    "default": 155,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("foreground_frames", "alpha_frames")
    FUNCTION = "process_video"
    CATEGORY = "Kytra-MatAnyone"

    def download_model(self, url, save_path):
        """Download the model file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"Downloading MatAnyone model to {save_path}")
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(save_path, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Failed to download model file")
    
    def reset_processor(self):
        """Reset the processor to clear any cached states"""
        if self.model is not None:
            self.processor = InferenceCore(self.model, cfg=self.model.cfg)
    
    def load_model(self):
        """Load the MatAnyone model if not already loaded"""
        if self.model is None:
            # Use local model path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(current_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            
            ckpt_path = os.path.join(model_dir, "matanyone.pth")
            
            # Check if model exists, if not download it
            if not os.path.exists(ckpt_path):
                print("MatAnyone model not found, downloading...")
                model_url = "https://huggingface.co/Mothersuperior/ComfyUI_MatAnyone_Kytra/resolve/main/matanyone.pth?download=true"
                self.download_model(model_url, ckpt_path)
                print("Model downloaded successfully!")
            
            # Load model
            self.model = get_matanyone_model(ckpt_path)
            self.processor = InferenceCore(self.model, cfg=self.model.cfg)
    
    def tensor_to_numpy(self, tensor):
        """Convert ComfyUI tensor to numpy array"""
        # ComfyUI tensors are [B, C, H, W] in range [0, 1]
        return tensor.cpu().numpy()
    
    def numpy_to_tensor(self, array):
        """Convert numpy array to ComfyUI tensor"""
        # Convert to tensor in range [0, 1]
        return torch.from_numpy(array).float()
    
    def process_video(self, video_frames, mask, warmup_frames=10, erode_kernel=10, dilate_kernel=10, 
                     bg_red=120, bg_green=255, bg_blue=155):
        """
        Process video frames with MatAnyone
        
        Args:
            video_frames: Tensor of shape [B, C, H, W] in range [0, 1]
            mask: Tensor of shape [H, W] or [1, H, W] in range [0, 1]
            warmup_frames: Number of warmup iterations
            erode_kernel: Erosion kernel size
            dilate_kernel: Dilation kernel size
            bg_red: Red component of background color (0-255)
            bg_green: Green component of background color (0-255)
            bg_blue: Blue component of background color (0-255)
            
        Returns:
            foreground_frames: Tensor of shape [B, C, H, W] in range [0, 1]
            alpha_frames: Tensor of shape [B, 1, H, W] in range [0, 1]
        """
        # Load model if not already loaded
        self.load_model()
        
        # Reset processor to clear any cached states
        self.reset_processor()
        
        # Convert inputs to the format expected by MatAnyone
        vframes = video_frames.clone()
        
        # Input is [B, H, W, C], convert to [B, C, H, W]
        if vframes.shape[-1] == 3:  # If channels are last
            vframes = vframes.permute(0, 3, 1, 2)
        
        # Scale to [0, 255]
        vframes = vframes * 255.0
        
        # Prepare mask - convert from [0, 1] to [0, 255]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension [1, H, W]
        if mask.ndim == 3:
            if mask.shape[0] == 1:
                mask = mask[0]  # Remove batch dimension if present
            elif mask.shape[2] == 1:  # If [H, W, 1]
                mask = mask.permute(2, 0, 1)[0]  # Convert to [H, W]
            
        # Convert mask to numpy for processing
        mask_np = (mask.cpu().numpy() * 255.0).astype(np.float32)
        
        # Apply erosion and dilation if needed
        if dilate_kernel > 0:
            mask_np = gen_dilate(mask_np, dilate_kernel, dilate_kernel)
        if erode_kernel > 0:
            mask_np = gen_erosion(mask_np, erode_kernel, erode_kernel)
        
        # Convert back to tensor and move to GPU
        mask_tensor = torch.from_numpy(mask_np).cuda()
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor  # Keep as [H, W]
        elif mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor[0]  # Convert [1, H, W] to [H, W]
        
        # Resize mask to match the first frame dimensions
        first_frame = vframes[0]  # [C, H, W]
        img_h, img_w = first_frame.shape[1], first_frame.shape[2]
        mask_h, mask_w = mask_tensor.shape
        
        if mask_h != img_h or mask_w != img_w:
            # Add batch and channel dimensions for interpolation
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            mask_tensor = F.interpolate(mask_tensor, size=(img_h, img_w), mode='nearest')
            mask_tensor = mask_tensor[0, 0]  # Remove batch and channel dimensions
        
        # Prepare for processing
        n_warmup = int(warmup_frames)
        
        # Create repeated frames for warmup
        first_frame = vframes[0:1]  # Already in [B, C, H, W] format
        first_frame = first_frame.repeat(n_warmup, 1, 1, 1)
        vframes_with_warmup = torch.cat([first_frame, vframes], dim=0)
        
        # Process frames
        phas = []
        fgrs = []
        
        # Create background color from RGB values - in correct RGB order
        bgr = (np.array([bg_red, bg_green, bg_blue], dtype=np.float32)/255).reshape((1, 1, 3))
        objects = [1]
        
        # Process each frame
        for ti in range(len(vframes_with_warmup)):
            # Get current frame - already in [B, C, H, W] format
            image = vframes_with_warmup[ti]  # Should be [C, H, W]
            
            # Convert for visualization (after processing)
            image_np = image.permute(1, 2, 0).cpu().numpy() / 255.0
            
            # Prepare for network input - ensure [C, H, W] format (not batched)
            # The model expects a single frame without batch dimension
            image_input = image.clone().cuda().float() / 255.0
            
            # Process frame
            if ti == 0:
                output_prob = self.processor.step(image_input, mask_tensor, objects=objects)  # encode given mask
                output_prob = self.processor.step(image_input, first_frame_pred=True)  # first frame for prediction
            else:
                if ti <= n_warmup:
                    output_prob = self.processor.step(image_input, first_frame_pred=True)  # reinit as the first frame
                else:
                    output_prob = self.processor.step(image_input)
            
            # Convert output to alpha matte
            mask_output = self.processor.output_prob_to_mask(output_prob)
            
            # Create alpha and foreground
            pha = mask_output.unsqueeze(2).cpu().numpy()  # [H, W] -> [H, W, 1]
            
            # Create composite with background color
            com_np = image_np * pha + bgr * (1 - pha)
            
            # Skip warmup frames
            if ti >= n_warmup:
                fgrs.append(com_np)
                phas.append(pha)
        
        # Convert results to tensors
        fgrs_np = np.array(fgrs).transpose(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        phas_np = np.array(phas).transpose(0, 3, 1, 2)  # [B, H, W, 1] -> [B, 1, H, W]
        
        # Convert to ComfyUI tensors [0, 1]
        fgrs_tensor = torch.from_numpy(fgrs_np).float()
        phas_tensor = torch.from_numpy(phas_np).float()
        
        # Ensure correct dimensions for ComfyUI
        # ComfyUI expects [B, C, H, W] for images and [B, 1, H, W] for masks
        if fgrs_tensor.ndim == 3:  # If [C, H, W]
            fgrs_tensor = fgrs_tensor.unsqueeze(0)  # Add batch dimension
        
        if phas_tensor.ndim == 3:  # If [1, H, W]
            phas_tensor = phas_tensor.unsqueeze(0)  # Add batch dimension
        elif phas_tensor.ndim == 4 and phas_tensor.shape[1] != 1:
            # If [B, C, H, W] but C is not 1
            phas_tensor = phas_tensor[:, 0:1, :, :]  # Keep only first channel
        
        # Final permutation for ComfyUI video nodes
        fgrs_tensor = fgrs_tensor.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        
        return (fgrs_tensor, phas_tensor)

#Kytra Images To RGB Node
class Kytra_Images_To_RGB:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_to_rgb"

    CATEGORY = "Kytra-MatAnyone"

    def image_to_rgb(self, images):

        if len(images) > 1:
            tensors = []
            for image in images:
                tensors.append(pil2tensor(tensor2pil(image).convert('RGB')))
            tensors = torch.cat(tensors, dim=0)
            return (tensors, )
        else:
            return (pil2tensor(tensor2pil(images).convert("RGB")), )

NODE_CLASS_MAPPINGS = {
    "MatAnyoneVideoMatting": MatAnyoneNode,
    "Kytra_Images_To_RGB": Kytra_Images_To_RGB
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MatAnyoneVideoMatting": "MatAnyone Video Kytra",
    "Kytra_Images_To_RGB": "Images To RGB Kytra"
} 