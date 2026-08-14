
from .F5TTS import F5TTSAudio, F5TTSAudioInputs, F5TTSAudioAdvanced, F5TTSLoadModel, F5TTSAudioFromModel

NODE_CLASS_MAPPINGS = {
    "F5TTSAudio": F5TTSAudio,
    "F5TTSAudioInputs": F5TTSAudioInputs,
    "F5TTSAudioFromModel": F5TTSAudioFromModel,
    "F5TTSLoadModel": F5TTSLoadModel,
    "F5TTSAudioAdvanced": F5TTSAudioAdvanced
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "F5TTSAudio": "F5-TTS Audio",
    "F5TTSAudioInputs": "F5-TTS Audio from input",
    "F5TTSAudioFromModel": "F5-TTS Audio from model and sample",
    "F5TTSLoadModel": "F5-TTS Load model",
    "F5TTSAudioAdvanced": "F5-TTS Audio advanced",
}
