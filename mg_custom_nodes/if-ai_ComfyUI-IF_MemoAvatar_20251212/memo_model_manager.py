#memo_model_manager.py
import os
import logging
import json
import shutil
from huggingface_hub import hf_hub_download
from modelscope import snapshot_download
import folder_paths

logger = logging.getLogger("memo")

class MemoModelManager:
    def __init__(self):
        self.models_base = folder_paths.models_dir
        self._setup_paths()
        self._ensure_model_structure()

    def _setup_paths(self):
        """Initialize base paths structure"""
        self.paths = {
            "memo_base": os.path.join(self.models_base, "checkpoints", "memo"),
            "wav2vec": os.path.join(self.models_base, "wav2vec", "facebook", "wav2vec2-base-960h"),
            "emotion2vec": os.path.join(self.models_base, "emotion2vec", "iic", "emotion2vec_plus_large"),
            "vae": os.path.join(self.models_base, "vae", "stabilityai", "sd-vae-ft-mse")
        }

        # Create directories
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

        # Create memo subfolders
        for subdir in ["reference_net", "diffusion_net", "image_proj", "audio_proj", 
                      "misc/audio_emotion_classifier", "misc/face_analysis", "misc/vocal_separator"]:
            os.makedirs(os.path.join(self.paths["memo_base"], subdir), exist_ok=True)

    def _direct_download(self, repo_id, filename, target_path, force=False):
        """Download directly to target path without extra nesting"""
        try:
            if not force and os.path.exists(target_path):
                return target_path

            download_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=os.path.dirname(target_path),
                local_dir_use_symlinks=False
            )
            
            # Move if downloaded to wrong location
            if download_path != target_path:
                shutil.move(download_path, target_path)
                
            return target_path
        except Exception as e:
            logger.warning(f"Failed to download {filename} from {repo_id} to {target_path}: {e}")
            return None

    def _setup_face_analysis(self):
        """Setup face analysis models with correct structure"""
        face_dir = os.path.join(self.paths["memo_base"], "misc", "face_analysis")
        models_dir = os.path.join(face_dir, "models")  # Create a models subdirectory
        os.makedirs(models_dir, exist_ok=True)
        
        # Create models.json
        models_json = {
            "detection": ["scrfd_10g_bnkps"],
            "recognition": ["glintr100"],
            "analysis": ["genderage", "2d106det", "1k3d68"]
        }
        
        # Write models.json
        models_json_path = os.path.join(face_dir, "models.json")
        with open(models_json_path, "w") as f:
            json.dump(models_json, f, indent=2)

        # Create version.txt
        version_path = os.path.join(face_dir, "version.txt")
        with open(version_path, "w") as f:
            f.write("0.7.3")

        # Download model files if they don't exist
        required_models = {
            "scrfd_10g_bnkps.onnx": "scrfd_10g_bnkps",
            "glintr100.onnx": "glintr100", 
            "genderage.onnx": "genderage",
            "2d106det.onnx": "2d106det",
            "1k3d68.onnx": "1k3d68",
            "face_landmarker_v2_with_blendshapes.task": "face_landmarker"
        }

        for model_file, model_name in required_models.items():
            target_path = os.path.join(models_dir, model_file)  # Save in models subdirectory
            if not os.path.exists(target_path):
                self._direct_download(
                    "memoavatar/memo",
                    f"misc/face_analysis/models/{model_file}",
                    target_path
                )
                # Create symlink in parent directory for compatibility
                parent_target = os.path.join(face_dir, model_file)
                if not os.path.exists(parent_target):
                    if os.name == 'nt':  # Windows
                        import shutil
                        shutil.copy2(target_path, parent_target)
                    else:  # Unix-like
                        os.symlink(target_path, parent_target)

        # Set environment variable for face models
        os.environ["MEMO_FACE_MODELS"] = face_dir
        return face_dir

    def _ensure_model_structure(self):
        """Download all required models to correct locations"""
        # Set up face analysis and environment variables first
        face_dir = self._setup_face_analysis()
        os.environ["MEMO_FACE_MODELS"] = face_dir
        os.environ["MEMO_VOCAL_MODEL"] = os.path.join(
            self.paths["memo_base"], "misc/vocal_separator/Kim_Vocal_2.onnx"
        )

        # Download memo components
        components = {
            "reference_net": ["config.json", "diffusion_pytorch_model.safetensors"],
            "diffusion_net": ["config.json", "diffusion_pytorch_model.safetensors"],
            "image_proj": ["config.json", "diffusion_pytorch_model.safetensors"],
            "audio_proj": ["config.json", "diffusion_pytorch_model.safetensors"]
        }

        for component, files in components.items():
            component_dir = os.path.join(self.paths["memo_base"], component)
            for file in files:
                self._direct_download(
                    "memoavatar/memo",
                    f"{component}/{file}",
                    os.path.join(component_dir, file)
                )
        
        # Download vocal separator
        self._direct_download(
            "memoavatar/memo",
            "misc/vocal_separator/Kim_Vocal_2.onnx",
            os.path.join(self.paths["memo_base"], "misc/vocal_separator/Kim_Vocal_2.onnx")
        )

        # Download emotion classifier
        self._direct_download(
            "memoavatar/memo",
            "misc/audio_emotion_classifier/diffusion_pytorch_model.safetensors",
            os.path.join(self.paths["memo_base"], "misc/audio_emotion_classifier/diffusion_pytorch_model.safetensors")
        )

        # Download wav2vec files
        for file in ["config.json", "preprocessor_config.json", "pytorch_model.bin", 
                    "special_tokens_map.json", "tokenizer_config.json", "vocab.json"]:
            self._direct_download(
                "facebook/wav2vec2-base-960h",
                file, 
                os.path.join(self.paths["wav2vec"], file)
            )

        # Download emotion2vec
        try:
            snapshot_download(
                "iic/emotion2vec_plus_large",
                local_dir=self.paths["emotion2vec"]
            )
        except Exception as e:
            logger.warning(f"Failed to download emotion2vec model: {e}")

        # Download VAE
        for file in ["config.json", "diffusion_pytorch_model.safetensors"]:
            self._direct_download(
                "stabilityai/sd-vae-ft-mse",
                file,
                os.path.join(self.paths["vae"], file)
            )

    def get_model_paths(self):
        """Return paths dictionary"""
        return {
            "memo_base": self.paths["memo_base"],
            "face_models": os.path.join(self.paths["memo_base"], "misc/face_analysis"),
            "vocal_separator": os.path.join(self.paths["memo_base"], "misc/vocal_separator/Kim_Vocal_2.onnx"),
            "wav2vec": self.paths["wav2vec"],
            "emotion2vec": self.paths["emotion2vec"],
            "vae": self.paths["vae"]
        }