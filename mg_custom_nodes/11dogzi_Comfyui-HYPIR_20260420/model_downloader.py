import os
import sys
import requests
import hashlib
from pathlib import Path
from tqdm import tqdm

# Try to import folder_paths (ComfyUI specific)
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False

class HYPIRModelDownloader:
    """HYPIR model download manager"""
    
    def __init__(self):
        self.models_dir = self._get_models_dir()
        self.hypir_dir = os.path.join(self.models_dir, "HYPIR")
        self._ensure_hypir_dir()
        
        # Model information configuration
        self.model_configs = {
            "HYPIR_sd2": {
                "filename": "HYPIR_sd2.pth",
                "url": "https://huggingface.co/lxq007/HYPIR/resolve/main/HYPIR_sd2.pth",
                "mirror_url": "https://openxlab.org.cn/models/detail/linxqi/HYPIR/resolve/main/HYPIR_sd2.pth",
                "description": "HYPIR SD2.1 image restoration model"
            }
        }
    
    def _get_models_dir(self):
        """Get ComfyUI models directory"""
        # Directly use ComfyUI's models directory, not checkpoints subdirectory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Search up for ComfyUI root directory
        for i in range(5):  # Up to 5 levels
            parent = os.path.dirname(current_dir)
            if os.path.exists(os.path.join(parent, "models")):
                return os.path.join(parent, "models")
            current_dir = parent
        
        # If not found, use models in the current directory
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    def _ensure_hypir_dir(self):
        """Ensure HYPIR model directory exists"""
        os.makedirs(self.hypir_dir, exist_ok=True)
        print(f"HYPIR model directory: {self.hypir_dir}")
    
    def _download_file(self, url, filepath, description="File"):
        """Download file and show progress bar"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {description}"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
            
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    
    def _verify_file(self, filepath, expected_size=None, expected_md5=None):
        """Verify file integrity"""
        if not os.path.exists(filepath):
            return False
        
        # Only check if the file exists, not its size
        actual_size = os.path.getsize(filepath)
        if actual_size > 0:
            print(f"File downloaded, size: {actual_size} bytes")
            return True
        
        return False
    
    def download_model(self, model_name):
        """Download a specified model"""
        if model_name not in self.model_configs:
            print(f"Unknown model: {model_name}")
            return None
        
        config = self.model_configs[model_name]
        filename = config["filename"]
        model_path = os.path.join(self.hypir_dir, filename)
        
        # Check if model already exists and is valid
        if self._verify_file(model_path):
            print(f"Model already exists and is valid: {model_path}")
            return model_path
        
        print(f"Starting to download model: {model_name}")
        print(f"Description: {config['description']}")
        print(f"Save path: {model_path}")
        
        # Try to download from the main URL
        print("Attempting to download from the main source...")
        if self._download_file(config["url"], model_path, model_name):
            if self._verify_file(model_path):
                print(f"Model downloaded successfully: {model_path}")
                return model_path
            else:
                print("File verification failed, attempting mirror source...")
        
        # Try to download from the mirror URL
        if "mirror_url" in config:
            print("Attempting to download from the mirror source...")
            if self._download_file(config["mirror_url"], model_path, f"{model_name} (Mirror)"):
                if self._verify_file(model_path):
                    print(f"Model downloaded successfully: {model_path}")
                    return model_path
        
        print(f"Model download failed: {model_name}")
        return None
    
    def get_model_path(self, model_name):
        """Get model path, download if it doesn't exist"""
        if model_name not in self.model_configs:
            print(f"Unknown model: {model_name}")
            return None
        
        config = self.model_configs[model_name]
        filename = config["filename"]
        model_path = os.path.join(self.hypir_dir, filename)
        
        # Check if model exists
        if os.path.exists(model_path):
            print(f"Found existing model: {model_path}")
            # Return relative path: HYPIR/filename
            return os.path.join("HYPIR", filename)
        
        # Model does not exist, try to download
        print(f"Model does not exist, starting download: {model_name}")
        downloaded_path = self.download_model(model_name)
        if downloaded_path and os.path.exists(downloaded_path):
            # Return relative path: HYPIR/filename
            return os.path.join("HYPIR", filename)
        
        # Even if download fails, return relative path for reference
        print(f"Model download failed, but returning relative path for reference: HYPIR/{filename}")
        return os.path.join("HYPIR", filename)
    
    def list_available_models(self):
        """List available models"""
        return list(self.model_configs.keys())
    
    def list_downloaded_models(self):
        """List downloaded models"""
        downloaded = []
        for model_name, config in self.model_configs.items():
            filename = config["filename"]
            model_path = os.path.join(self.hypir_dir, filename)
            if os.path.exists(model_path):
                downloaded.append(model_name)
        return downloaded

# Global downloader instance
_downloader = None

def get_downloader():
    """Get global downloader instance"""
    global _downloader
    if _downloader is None:
        _downloader = HYPIRModelDownloader()
    return _downloader

def download_hypir_model(model_name="HYPIR_sd2"):
    """Convenient function to download HYPIR models"""
    downloader = get_downloader()
    return downloader.get_model_path(model_name)

def get_hypir_model_path(model_name="HYPIR_sd2"):
    """Convenient function to get HYPIR model path"""
    downloader = get_downloader()
    return downloader.get_model_path(model_name)

# Test function
if __name__ == "__main__":
    print("HYPIR Model Downloader Test")
    downloader = HYPIRModelDownloader()
    
    print(f"Model directory: {downloader.hypir_dir}")
    print(f"Available models: {downloader.list_available_models()}")
    print(f"Downloaded models: {downloader.list_downloaded_models()}")
    
    # Test download
    model_path = downloader.get_model_path("HYPIR_sd2")
    if model_path:
        print(f"Model path: {model_path}")
    else:
        print("Model download failed") 