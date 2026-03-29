
from .F5TTS import F5TTSAudio, F5TTSAudioInputs, F5TTSAudioAdvanced

NODE_CLASS_MAPPINGS = {
    "F5TTSAudio": F5TTSAudio,
    "F5TTSAudioInputs": F5TTSAudioInputs,
    "F5TTSAudioAdvanced": F5TTSAudioAdvanced
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "F5TTSAudio": "F5-TTS Audio",
    "F5TTSAudioInputs": "F5-TTS Audio from input",
    "F5TTSAudioAdvanced": "F5-TTS Audio advanced",
}
