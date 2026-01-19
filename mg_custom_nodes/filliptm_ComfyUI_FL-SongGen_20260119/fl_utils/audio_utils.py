"""
Audio utilities for FL Song Gen.
Handles conversion between ComfyUI AUDIO format and SongGeneration formats.
"""

import torch
import torchaudio


def comfyui_audio_to_tensor(audio: dict) -> tuple:
    """
    Extract tensor and sample rate from ComfyUI AUDIO format.

    Args:
        audio: ComfyUI audio dict with 'waveform' and 'sample_rate'

    Returns:
        (waveform tensor [B, C, S], sample_rate int)
    """
    waveform = audio['waveform']
    sample_rate = audio['sample_rate']

    # Ensure 3D tensor [batch, channels, samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    return waveform, sample_rate


def tensor_to_comfyui_audio(waveform: torch.Tensor, sample_rate: int) -> dict:
    """
    Convert tensor to ComfyUI AUDIO format.

    Args:
        waveform: Audio tensor, can be [S], [C, S], or [B, C, S]
        sample_rate: Sample rate in Hz

    Returns:
        ComfyUI audio dict with 'waveform' [B, C, S] and 'sample_rate'
    """
    # Ensure 3D tensor [batch, channels, samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    return {
        'waveform': waveform,
        'sample_rate': sample_rate
    }


def resample_audio(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """
    Resample audio to target sample rate.

    Args:
        waveform: Audio tensor [B, C, S] or [C, S]
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled tensor
    """
    if orig_sr == target_sr:
        return waveform

    return torchaudio.functional.resample(waveform, orig_sr, target_sr)


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert stereo to mono by averaging channels.

    Args:
        waveform: Audio tensor [B, C, S]

    Returns:
        Mono tensor [B, 1, S]
    """
    if waveform.shape[1] == 1:
        return waveform

    return waveform.mean(dim=1, keepdim=True)


def ensure_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert mono to stereo by duplicating channel.

    Args:
        waveform: Audio tensor [B, C, S]

    Returns:
        Stereo tensor [B, 2, S]
    """
    if waveform.shape[1] == 2:
        return waveform

    return waveform.repeat(1, 2, 1)


def prepare_prompt_audio_for_songgen(
    audio: dict,
    max_duration_sec: float = 10.0,
    target_sr: int = 48000
) -> torch.Tensor:
    """
    Prepare prompt audio for SongGeneration model.
    Resamples to 48kHz and limits to max duration.

    Args:
        audio: ComfyUI AUDIO dict
        max_duration_sec: Maximum duration in seconds (default 10s)
        target_sr: Target sample rate (48kHz for SongGen)

    Returns:
        Prepared audio tensor [B, C, S] at target sample rate
    """
    waveform, orig_sr = comfyui_audio_to_tensor(audio)

    # Resample to target sample rate
    if orig_sr != target_sr:
        waveform = resample_audio(waveform, orig_sr, target_sr)

    # Limit duration
    max_samples = int(max_duration_sec * target_sr)
    if waveform.shape[-1] > max_samples:
        waveform = waveform[..., :max_samples]

    return waveform


def empty_audio(sample_rate: int = 24000, duration_sec: float = 1.0) -> dict:
    """
    Create an empty (silent) audio dict.

    Args:
        sample_rate: Sample rate in Hz
        duration_sec: Duration in seconds

    Returns:
        ComfyUI AUDIO dict with silence
    """
    samples = int(sample_rate * duration_sec)
    waveform = torch.zeros(1, 2, samples)
    return {
        'waveform': waveform,
        'sample_rate': sample_rate
    }
