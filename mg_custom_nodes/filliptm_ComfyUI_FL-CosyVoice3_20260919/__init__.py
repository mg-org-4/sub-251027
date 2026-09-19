"""
FL CosyVoice3 - Advanced Text-to-Speech for ComfyUI
Zero-shot voice cloning, cross-lingual synthesis, and instruction-based control
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import nodes
from .nodes.model_loader import FL_CosyVoice3_ModelLoader
from .nodes.zero_shot import FL_CosyVoice3_ZeroShot
from .nodes.cross_lingual import FL_CosyVoice3_CrossLingual
from .nodes.voice_conversion import FL_CosyVoice3_VoiceConversion
from .nodes.audio_crop import FL_CosyVoice3_AudioCrop
from .nodes.dialog import FL_CosyVoice3_Dialog
from .nodes.instruct2 import FL_CosyVoice3_Instruct2
from .nodes.save_speaker import FL_CosyVoice3_SaveSpeaker
from .nodes.speaker_clone import FL_CosyVoice3_SpeakerClone
from .nodes.speaker_instruct2 import FL_CosyVoice3_SpeakerInstruct2

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "FL_CosyVoice3_ModelLoader": FL_CosyVoice3_ModelLoader,
    "FL_CosyVoice3_ZeroShot": FL_CosyVoice3_ZeroShot,
    "FL_CosyVoice3_CrossLingual": FL_CosyVoice3_CrossLingual,
    "FL_CosyVoice3_VoiceConversion": FL_CosyVoice3_VoiceConversion,
    "FL_CosyVoice3_AudioCrop": FL_CosyVoice3_AudioCrop,
    "FL_CosyVoice3_Dialog": FL_CosyVoice3_Dialog,
    "FL_CosyVoice3_Instruct2": FL_CosyVoice3_Instruct2,
    "FL_CosyVoice3_SaveSpeaker": FL_CosyVoice3_SaveSpeaker,
    "FL_CosyVoice3_SpeakerClone": FL_CosyVoice3_SpeakerClone,
    "FL_CosyVoice3_SpeakerInstruct2": FL_CosyVoice3_SpeakerInstruct2,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_CosyVoice3_ModelLoader": "FL CosyVoice3 Model Loader",
    "FL_CosyVoice3_ZeroShot": "FL CosyVoice3 Zero-Shot Clone",
    "FL_CosyVoice3_CrossLingual": "FL CosyVoice3 Cross-Lingual",
    "FL_CosyVoice3_VoiceConversion": "FL CosyVoice3 Voice Conversion",
    "FL_CosyVoice3_AudioCrop": "FL CosyVoice3 Audio Crop",
    "FL_CosyVoice3_Dialog": "FL CosyVoice3 Dialog",
    "FL_CosyVoice3_Instruct2": "FL CosyVoice3 Instruct2",
    "FL_CosyVoice3_SaveSpeaker": "FL CosyVoice3 Save Speaker",
    "FL_CosyVoice3_SpeakerClone": "FL CosyVoice3 Speaker Clone",
    "FL_CosyVoice3_SpeakerInstruct2": "FL CosyVoice3 Speaker Instruct2",
}

# ASCII art banner
ascii_art = """
⣏⡉ ⡇    ⡎⠑ ⢀⡀ ⢀⣀ ⡀⢀ ⡇⢸ ⢀⡀ ⠄ ⢀⣀ ⢀⡀ ⢉⡹
⠇  ⠧⠤   ⠣⠔ ⠣⠜ ⠭⠕ ⣑⡺ ⠸⠃ ⠣⠜ ⠇ ⠣⠤ ⠣⠭ ⠤⠜
"""
print(f"\033[35m{ascii_art}\033[0m")
print("FL CosyVoice3 Custom Nodes Loaded - Version 1.2.1")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
