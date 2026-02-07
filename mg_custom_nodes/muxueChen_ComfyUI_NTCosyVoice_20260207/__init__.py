import sys
import os
nor_dir = os.path.dirname(__file__)
Matcha_path = os.path.join(nor_dir, 'third_party/Matcha-TTS')
sys.path.append(nor_dir)
sys.path.append(Matcha_path)

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch


def nt_load_wav(speech, sample_rate, target_sr):
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


class NTCosyVoiceZeroShotSampler:
    def __init__(self):
        self.__cosyvoice = None

    @property
    def cosyvoice(self):
        if self.__cosyvoice is None:
            model_path = os.path.join(nor_dir, 'pretrained_models/CosyVoice2-0.5B')
            self.__cosyvoice = CosyVoice2(model_path, load_jit=True, load_onnx=False, load_trt=False)
        return self.__cosyvoice

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "text": ("STRING", {"multiline": True}),
                "prompt_text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("tts_speech",)
    FUNCTION = "main_func"
    CATEGORY = "Nineton Nodes"

    def main_func(self, audio, speed, text, prompt_text):
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]
        print(f"waveform:{waveform}, sample_rate:{sample_rate}")
        prompt_speech_16k = nt_load_wav(waveform, sample_rate, 16000)
        speechs = []
        for i, j in enumerate(self.cosyvoice.inference_zero_shot(tts_text=text, prompt_text=prompt_text, prompt_speech_16k=prompt_speech_16k, stream=False, speed=speed)):
            speechs.append(j['tts_speech'])

        tts_speech = torch.cat(speechs, dim=1)
        tts_speech = tts_speech.unsqueeze(0)
        outaudio = {"waveform": tts_speech, "sample_rate": self.cosyvoice.sample_rate}

        return (outaudio,)


class NTCosyVoiceCrossLingualSampler:
    def __init__(self):
        self.__cosyvoice = None

    @property
    def cosyvoice(self):
        if self.__cosyvoice is None:
            model_path = os.path.join(nor_dir, 'pretrained_models/CosyVoice2-0.5B')
            self.__cosyvoice = CosyVoice2(model_path, load_jit=True, load_onnx=False, load_trt=False)
        return self.__cosyvoice

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("tts_speech",)
    FUNCTION = "main_func"
    CATEGORY = "Nineton Nodes"

    def main_func(self, audio, speed, text):
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]

        prompt_speech_16k = nt_load_wav(waveform, sample_rate, 16000)
        speechs = []
        for i, j in enumerate(self.cosyvoice.inference_cross_lingual(tts_text=text,
                prompt_speech_16k=prompt_speech_16k, stream=False, speed=speed)):
            speechs.append(j['tts_speech'])

        tts_speech = torch.cat(speechs, dim=1)
        tts_speech = tts_speech.unsqueeze(0)
        outaudio = {"waveform": tts_speech, "sample_rate": self.cosyvoice.sample_rate}

        return (outaudio,)


class NTCosyVoiceInstruct2Sampler:
    def __init__(self):
        self.__cosyvoice = None

    @property
    def cosyvoice(self):
        if self.__cosyvoice is None:
            model_path = os.path.join(nor_dir, 'pretrained_models/CosyVoice2-0.5B')
            self.__cosyvoice = CosyVoice2(model_path, load_jit=True, load_onnx=False, load_trt=False)
        return self.__cosyvoice

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.1}),
                "text": ("STRING", {"multiline": True}),
                "instruct": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("tts_speech",)
    FUNCTION = "main_func"
    CATEGORY = "Nineton Nodes"

    def main_func(self, audio, speed, text, instruct):
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]

        prompt_speech_16k = nt_load_wav(waveform, sample_rate, 16000)

        speechs = []
        for i, j in enumerate(self.cosyvoice.inference_instruct2(tts_text=text, instruct_text=instruct, prompt_speech_16k=prompt_speech_16k, stream=False, speed=speed)):
            speechs.append(j['tts_speech'])

        tts_speech = torch.cat(speechs, dim=1)
        tts_speech = tts_speech.unsqueeze(0)
        outaudio = {"waveform": tts_speech, "sample_rate": self.cosyvoice.sample_rate}

        return (outaudio,)


NODE_CLASS_MAPPINGS = {
    "NTCosyVoiceZeroShotSampler": NTCosyVoiceZeroShotSampler,
    "NTCosyVoiceInstruct2Sampler": NTCosyVoiceInstruct2Sampler,
    "NTCosyVoiceCrossLingualSampler": NTCosyVoiceCrossLingualSampler
}

__all__ = ['NODE_CLASS_MAPPINGS']