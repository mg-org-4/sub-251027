import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from .src.utils import ensure_model_directories
from .src.face_processor_integrated import ForbiddenVisionFaceProcessorIntegrated
from .src.face_fixer_mask_only import ForbiddenVisionFaceFixerMaskOnly
from .src.face_edit_nodes import ForbiddenVisionFaceEditPrep, ForbiddenVisionFaceEditMerge
from .src.latent_refiner import LatentRefiner
from .src.latent_builder import LatentBuilder
from .src.latent_rebuilder import ForbiddenVisionRebuilder
from .src.sampler_scheduler_settings import SamplerSchedulerSettings
from .src.latent_inpaint_lite import ForbiddenVisionInpaintLite

ensure_model_directories()

def initialize_forbidden_vision():
    try:
        from .src.model_manager import ForbiddenVisionModelManager
        from .src.utils import check_forbidden_vision_models
        
        print("=" * 60)
        print("ForbiddenVision: Initializing custom nodes.")
        
        model_status = check_forbidden_vision_models()
        existing_models = [name for name, exists in model_status.items() if exists]
        
        if existing_models:
            print(f"ForbiddenVision: Found {len(existing_models)} existing models")
            for model in existing_models:
                print(f"  ✓ {model}")
        
        model_manager = ForbiddenVisionModelManager.get_instance()
        validation_status = model_manager.validate_model_availability()
        
        if not all(validation_status.values()):
            print("ForbiddenVision: Missing required models, downloading defaults...")
            model_manager.initialize_default_models()
            validation_status = model_manager.validate_model_availability()
        
        print("\nForbiddenVision: Model Status:")
        print(f"  Face Detection:   {'✓' if validation_status['face_detection'] else '✗'}")
        print(f"  Neural Corrector: {'✓' if validation_status['neural_corrector'] else '✗'}")
        
        if validation_status['face_detection'] and validation_status['neural_corrector']:
            print("ForbiddenVision: Ready to use!")
        else:
            print("ForbiddenVision: WARNING - Some models are missing.")
            print("  Please check your internet connection.")
        
        print("=" * 60)
        
        return validation_status
        
    except ImportError as e:
        print(f"ForbiddenVision: Import error during initialization: {e}")
        return None
    except Exception as e:
        print(f"ForbiddenVision: Initialization error: {e}")
        return None

FORBIDDEN_VISION_STATUS = initialize_forbidden_vision()

NODE_CLASS_MAPPINGS = {
    "ForbiddenVisionFaceProcessorIntegrated": ForbiddenVisionFaceProcessorIntegrated,
    "ForbiddenVisionFaceFixerMaskOnly": ForbiddenVisionFaceFixerMaskOnly,
    "ForbiddenVisionFaceEditPrep": ForbiddenVisionFaceEditPrep,
    "ForbiddenVisionFaceEditMerge": ForbiddenVisionFaceEditMerge,
    "LatentRefiner": LatentRefiner,
    "LatentBuilder": LatentBuilder,
    "ForbiddenVisionRebuilder": ForbiddenVisionRebuilder,
    "ForbiddenVisionInpaintLite": ForbiddenVisionInpaintLite,
    "SamplerSchedulerSettings": SamplerSchedulerSettings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ForbiddenVisionFaceProcessorIntegrated": "Forbidden Vision 🎯 Fixer",
    "ForbiddenVisionFaceFixerMaskOnly": "Forbidden Vision 🎯 Fixer Mask Only",
    "ForbiddenVisionFaceEditPrep": "Forbidden Vision 🧩 Face Edit Prep",
    "ForbiddenVisionFaceEditMerge": "Forbidden Vision 🧩 Face Edit Merge",
    "LatentRefiner": "Forbidden Vision 💎 Refiner",
    "LatentBuilder": "Forbidden Vision 🛠️ Builder",
    "ForbiddenVisionRebuilder": "Forbidden Vision 🔧 Rebuilder",
    "ForbiddenVisionInpaintLite": "Forbidden Vision 🎨 Inpainter",
    "SamplerSchedulerSettings": "Forbidden Vision 🎛️ Settings",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

try:
    from nodes import CLIPTextEncode
    original_clip_text_encode = CLIPTextEncode.encode
    
    def fv_encode_wrapper(self, clip, text):
        encoded_output = original_clip_text_encode(self, clip, text)
        if isinstance(encoded_output, tuple) and len(encoded_output) > 0:
            conditioning = encoded_output[0]
            if isinstance(conditioning, list) and len(conditioning) > 0:
                if isinstance(conditioning[0], list) and len(conditioning[0]) == 2:
                    if isinstance(conditioning[0][1], dict):
                        conditioning[0][1]["forbidden_vision_metadata"] = {"original_text": text}
        return encoded_output
    
    CLIPTextEncode.encode = fv_encode_wrapper
    print("ForbiddenVision: Patched CLIPTextEncode for prompt metadata")
    
except Exception as e:
    print(f"ForbiddenVision: Could not patch CLIPTextEncode: {e}")