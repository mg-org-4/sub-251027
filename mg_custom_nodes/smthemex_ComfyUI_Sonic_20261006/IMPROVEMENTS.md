# Improvements in this fork

This branch (`fix/torchaudio-soundfile`) carries one focused fix on top of
[`smthemex/ComfyUI_Sonic`](https://github.com/smthemex/ComfyUI_Sonic).

---

## 1. Replace `torchaudio.save` with `soundfile.write`

**File**: `sonic_node.py`

Upstream calls `torchaudio.save(...)` to write the temporary WAV that the
SONIC pipeline feeds back into the inference model. With **torchaudio 2.11**
that call is routed through `torchcodec`, which on certain torch+CUDA
combinations (notably `torch 2.11.0+cu130` with `torchcodec 0.9.1+cu130`)
hits an ABI mismatch on `c10_cuda_check_implementation` and crashes the
worker.

The replacement uses `soundfile.write(...)` (libsndfile-backed) — a
dependency ComfyUI already pulls in for other audio nodes — and produces a
byte-identical WAV file without going through torchcodec.

```python
import soundfile as _sf

# Before:
# torchaudio.save(audio_path, audio["waveform"].squeeze(0),
#                 audio["sample_rate"], format="WAV")

# After:
_sf.write(audio_path, audio["waveform"].squeeze(0).numpy().T,
          audio["sample_rate"], subtype="PCM_16")
```

No behavioural change; just sidesteps the broken torchcodec path.

---

## How to install this fork

```bash
cd /opt/ComfyUI/custom_nodes
git clone https://github.com/svilendotorg/ComfyUI-Sonic
```

Default branch on this fork is `fix/torchaudio-soundfile`, so the clone
lands on the patched code automatically. Restart ComfyUI.

To track upstream:

```bash
cd ComfyUI-Sonic
git remote add upstream https://github.com/smthemex/ComfyUI_Sonic.git
git fetch upstream
git rebase upstream/main
git push --force-with-lease
```
