# ComfyUI-PiD

**ComfyUI custom nodes** for using NVIDIA **PiD** as a pixel diffusion decoder.

<img width="1058" height="604" alt="image" src="https://github.com/user-attachments/assets/cc5a9da3-94c6-4546-9574-c8387d5dffdb" />

<img width="4096" height="2048" alt="1111111111111111" src="https://github.com/user-attachments/assets/7ccd55ee-e571-4996-9c9c-4b5cecbb4418" />

PiD is not a normal ComfyUI `VAE`. It needs a latent, a prompt/caption, a sigma value, and optionally a native decoder baseline image:

```text
LATENT + caption + sigma + optional baseline IMAGE -> PiD -> IMAGE
```

For the official latent-conditioned PiD checkpoints, this node can infer the baseline size from the latent and skip the extra VAE/baseline image path to reduce VRAM use.

## Features

- Direct **PiD Decode** node that returns a ComfyUI `IMAGE`.
- Staged low-VRAM workflow: **PiD Prepare → PiD Sample → PiD Finalize**.
- **PiD Sample** runs in a subprocess so CUDA memory is released after sampling.
- **PiD KSampler Capture** for grabbing an intermediate latent and matching sigma.
- Lazy setup: PiD source, checkpoints, and required assets are prepared on first run when `auto_download=true`.
- Optional sequential block offload for lower VRAM at the cost of speed.

## Install

Clone into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Merserk/ComfyUI-PiD.git
cd ComfyUI-PiD
python -m pip install -r requirements.txt
```

Restart ComfyUI.

Requirements:
- Python `>=3.10`
- NVIDIA CUDA GPU
- Working ComfyUI install
- Enough VRAM for PiD, especially for `2kto4k` or large output scales

`requirements.txt` does not install PyTorch because ComfyUI usually provides it.

## Nodes

| Node | Purpose |
| --- | --- |
| **PiD Decode** | One-node PiD decode from latent to image. |
| **PiD Text Prompt** | One prompt box with `text` for CLIP and `caption` for PiD. |
| **PiD KSampler Capture** | KSampler-compatible sampler that returns final latent, captured PiD latent, and sigma. |
| **PiD Prepare** | Prepares latent, caption, checkpoint, assets, and metadata on CPU. |
| **PiD Sample** | Runs the heavy PiD sampling step in a subprocess. |
| **PiD Finalize** | Converts sampled PiD output back to ComfyUI `IMAGE`. |
| **PiD Decode (Staged)** | Convenience wrapper around the staged path. |

## Supported backbones

| Value | Backbone | Latent channels | Checkpoints |
| --- | --- | ---: | --- |
| `zimage` | Z-Image / Flux-compatible | 16 | `2k`, `2kto4k` |
| `flux` | Flux | 16 | `2k`, `2kto4k` |
| `flux2` | Flux2 | 128 | `2k`, `2kto4k` |
| `sd3` | Stable Diffusion 3 | 16 | `2k`, `2kto4k` |
| `dinov2` | DINOv2 RAE | 768 | `2k` |
| `siglip` | SigLIP Scale-RAE | 1152 | `2k` |

`scale=0` uses NVIDIA's default scale for the selected checkpoint: usually `4x`, or `8x` for SigLIP Scale-RAE.

## Basic workflow

For Z-Image / Flux-style workflows:

```text
PiD Text Prompt text    -> CLIP Text Encode
PiD Text Prompt caption -> PiD Decode caption
KSampler latent         -> PiD Decode latent
PiD Decode image        -> Save Image
```

Recommended first test settings:

```text
backbone = zimage
pid_ckpt_type = 2k
pid_steps = 4
scale = 1 or 2
cfg_scale = 1.0
sigma = 0.0
auto_download = true
unload_comfy_before_pid = true
aggressive_cleanup = true
sequential_offload = disabled
```

For official latent-conditioned checkpoints, leave `vae` and `baseline_image` disconnected unless you specifically need an external baseline size.

## Lowest-VRAM staged workflow

Use the staged nodes when VRAM is tight:

```text
PiD KSampler Capture pid_latent -> PiD Prepare latent
PiD Text Prompt caption         -> PiD Prepare caption
PiD Prepare                     -> PiD Sample
PiD Sample                      -> PiD Finalize
PiD Finalize image              -> Save Image
```

Recommended Z-Image capture settings:

```text
steps = 50
sampler_name = euler
scheduler = beta
capture_step = 46
```

`PiD Sample` runs in a separate Python process, so its CUDA context is destroyed after the sample is finished.

## Output size guide

```text
512x512 base  + 2k     + scale 4 -> 2048x2048
1024x1024 base + 2kto4k + scale 4 -> 4096x4096
```

Large outputs can require a lot of VRAM. If a run fails, try:
1. Lower `scale`.
2. Use a smaller base latent.
3. Keep cleanup options enabled.
4. Try `sequential_blocks`, then `sequential_blocks_aggressive`.
5. Restart ComfyUI after CUDA allocator crashes.

## PiD source and weights

By default, the node uses:

```text
ComfyUI/custom_nodes/ComfyUI-PiD/vendor/PiD
```

You can override the PiD source location with:
- the `pid_source_dir` node input
- `PID_REPO_DIR`
- `COMFYUI_PID_REPO_DIR`

When `auto_download=true`, the node downloads missing PiD source/checkpoints/assets as needed.

## Example workflow

A template workflow is included in:

```text
example_workflows/image_z_image_pid.json
```

After restart, open it from ComfyUI workflow templates or load the JSON manually.

## Notes

- This is a community wrapper around NVIDIA's public PiD code, not an official NVIDIA or ComfyUI project.
- PiD outputs `IMAGE`, not a ComfyUI `VAE`.
- NVIDIA's PiD weights may have separate license/usage terms. Check the model card before commercial use.
- Final latents with `sigma=0.0` can work, but captured intermediate latents usually better match the official PiD recipe.

## License

This project is released under the MIT License.
