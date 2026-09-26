# FL CosyVoice3

Advanced text-to-speech nodes for ComfyUI powered by the CosyVoice3 model family. Features zero-shot voice cloning, cross-lingual synthesis, and voice conversion.

[![CosyVoice](https://img.shields.io/badge/CosyVoice-Original%20Repo-blue?style=for-the-badge&logo=github&logoColor=white)](https://github.com/FunAudioLLM/CosyVoice)
[![Patreon](https://img.shields.io/badge/Patreon-Support%20Me-F96854?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/Machinedelusions)

![Workflow Preview](assets/workflow_preview.png)

## Features

- **Zero-Shot Voice Cloning** - Clone any voice from 3-30 seconds of reference audio
- **Cross-Lingual Synthesis** - Speak different languages while preserving voice characteristics
- **Voice Conversion** - Transform one voice to sound like another
- **9 Languages** - Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian
- **Auto Transcription** - Built-in Whisper integration for reference audio
- **Speed Control** - Adjustable speech rate (0.5x - 2.0x)

## Nodes

| Node | Description |
|------|-------------|
| **Model Loader** | Downloads and caches CosyVoice models |
| **Zero-Shot Clone** | Clone voices from reference audio |
| **Cross-Lingual** | Generate speech in different languages |
| **Voice Conversion** | Convert source audio to target voice |
| **Dialog** | Multi-speaker dialog synthesis with up to 4 voices |
| **Audio Crop** | Trim audio to specific time ranges |

## New Nodes
| Node | Description |
|------|-------------|
| **Instruct2** | Clone a voice with instruct text |
| **Save Speaker** | Save voice preset |
| **Speaker Clone** | Voice clone with voice preset |
| **Speaker Instruct2** | Voice clone with voice preset and instruct text |



## Installation

### ComfyUI Manager
Search for "FL CosyVoice3" and install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/filliptm/ComfyUI_FL-CosyVoice3.git
cd ComfyUI_FL-CosyVoice3
pip install -r requirements.txt
```

## Quick Start

1. Add **FL CosyVoice3 Model Loader** and select `Fun-CosyVoice3-0.5B`
2. Connect to **Zero-Shot Clone** or **Cross-Lingual** node
3. Provide reference audio (3-30 seconds recommended)
4. Enter your text and generate

## Models

| Model | Size | Notes |
|-------|------|-------|
| Fun-CosyVoice3-0.5B | ~2GB | Recommended |
| CosyVoice2-0.5B | ~2GB | Alternative |
| CosyVoice-300M | ~1.2GB | Lightweight |

Models download automatically on first use to `ComfyUI/models/cosyvoice/`.

**Notice:** CosyVoice-300M won't work well, do not use. 

## Nodes
### Instruct2
Named Instruct2 because in CosyVoice's source code, there is an instruct1 function only for CosyVoice1 model. Instruct2 is for CosyVoice2 and CosyVoice3 model.  

### Save Speaker
Choose a refernce voice with 3~10 seconds is the best, no more than 30 seconds. 

If reference text is empty, it will try to script reference audio into text as reference text. 

Speaker preset is saved to `Comfyui's model folder/cosyvoice/speaker`. 

**Be notieced:** a voice preset saved with CosyVoice3/2 model, can not be used with CosyVoice2/3 model.   

### Speaker Clone
**Be notieced:** a voice preset saved with CosyVoice3/2 model, can not be used with CosyVoice2/3 model.   

**Be notieced2:** CosyVoice's official speaker preset `spk2info.pt` from `CosyVoice-300M-SFT` model is not supported.  

If you really want to use those speaker presets from `spk2info.pt`, you can find those 8 voices at:
https://fun-audio-llm.github.io/#CosyVoice-basic  

Then you can just download those audios and save them into speaker presets with `Save Speaker node`.  

Using a speaker preset is excatly the same as using that speaker's reference audio for voice clone, same process, same result.   

### Speaker Instruct2
Load all speaker presets saved with `Save Speaker node` into a list, then you can pick one for tts with instruct text. 

**Notice:** instruct text can not be empty.  

## Requirements

- Python 3.10+
- 8GB RAM minimum (16GB+ recommended)
- NVIDIA GPU with 8GB+ VRAM recommended (CPU and Mac MPS supported)

## License

Apache 2.0

## Changelog
