# Compatibility package kept for older local imports.
# Active audio nodes now live in IAMCCS-nodes/audio.
from ..audio.audio_board_arranger import IAMCCS_AudioBoardArranger
from ..audio.audio_bus_out import IAMCCS_BusOut
from ..audio.audio_board_mixer import IAMCCS_AudioBoardMixer
from ..audio.audio_control_efx import IAMCCS_ControlAudEfx
from ..audio.dialogue_tag_editor import IAMCCS_DialogueTagEditor, IAMCCS_DialogueAudioBoardBridge

NODE_CLASS_MAPPINGS = {
    "IAMCCS_AudioBoardArranger": IAMCCS_AudioBoardArranger,
    "IAMCCS_BusOut": IAMCCS_BusOut,
    "IAMCCS_AudioBoardMixer": IAMCCS_AudioBoardMixer,
    "IAMCCS_ControlAudEfx": IAMCCS_ControlAudEfx,
    "IAMCCS_DialogueTagEditor": IAMCCS_DialogueTagEditor,
    "IAMCCS_DialogueAudioBoardBridge": IAMCCS_DialogueAudioBoardBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_AudioBoardArranger": "IAMCCS AudioBoard Arranger",
    "IAMCCS_BusOut": "IAMCCS BusOut",
    "IAMCCS_AudioBoardMixer": "IAMCCS AudioBoard Mixer",
    "IAMCCS_ControlAudEfx": "IAMCCS ControlAudEfx",
    "IAMCCS_DialogueTagEditor": "IAMCCS Dialogue Tag Editor",
    "IAMCCS_DialogueAudioBoardBridge": "IAMCCS Dialogue AudioBoard Bridge",
}
