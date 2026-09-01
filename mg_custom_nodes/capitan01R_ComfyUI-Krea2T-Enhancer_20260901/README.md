# ComfyUI-Krea2T-Enhancer

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow.svg)](https://buymeacoffee.com/capitan01r)

Prompt-adherence enhancement for Krea2 diffusion models in ComfyUI.

This custom node patches the Krea2 text-fusion path during sampling and applies a controlled internal conditioning adjustment intended to improve how strongly the model follows prompt details.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/capitan01R/ComfyUI-Krea2T-Enhancer.git
```

Restart ComfyUI after installing or updating.

No extra Python packages are required beyond a working ComfyUI Krea2 setup.

## Included Nodes

| Node | Output | Purpose |
|---|---|---|
| **ComfyUI-Krea2T-Enhancer** | `MODEL` | Patches the Krea2 model path to improve prompt adherence during sampling. |
| **Krea2T Enhancer Advanced** | `MODEL` | Same enhancer path, plus a direct post-`txtmlp` `text_scale` control for fused text-token strength. |
| **Krea2 Turbo Reference Sigmas (From Latent)** | `SIGMAS`, `LATENT` | Builds a Turbo sigma schedule based on the official Krea 2 Turbo scheduler settings and validates the connected latent dimensions. |

## Usage

Place **ComfyUI-Krea2T-Enhancer** between your Krea2 diffusion model loader and sampler:

```text
Load Diffusion Model -> ComfyUI-Krea2T-Enhancer -> KSampler
```

Or use **Krea2T Enhancer Advanced** when you want the additional text-scale control:

```text
Load Diffusion Model -> Krea2T Enhancer Advanced -> KSampler
```

Use your normal Krea2 text encoder, VAE, latent, and sampler setup.

For the sigma scheduler, connect both the loaded Krea2 Turbo model and the same
Empty Latent Image that will be sent to the sampler:

```text
Load Diffusion Model --\
                        > Krea2 Turbo Reference Sigmas (From Latent) -> SIGMAS to sampler
Empty Latent Image ----/                                             -> LATENT to sampler
```

## Controls

### ComfyUI-Krea2T-Enhancer

| Parameter | Default | Meaning |
|---|---:|---|
| `enabled` | `true` | Turns the patch on or off. |
| `strength` | `1.0` | Blends the enhancement from neutral `0.0` to full `2.0`. |
| `debug` | `false` | Prints concise runtime diagnostics to the ComfyUI console. |

### Krea2T Enhancer Advanced

| Parameter | Default | Meaning |
|---|---:|---|
| `enabled` | `true` | Turns the patch on or off. |
| `strength` | `1.0` | Same enhancer strength as the original node, from neutral `0.0` to full `2.0`. |
| `text_scale` | `1.0` | Multiplies fused text tokens immediately after `txtmlp`, before they enter the shared Krea2 stream. |
| `debug` | `false` | Prints concise runtime diagnostics to the ComfyUI console. |

Suggested starting range for `text_scale` is `1.50` to `2.00`. The neutral value is `1.0`.

### Krea2 Turbo Reference Sigmas (From Latent)

| Parameter | Default | Meaning |
|---|---:|---|
| `model` | — | The loaded Krea2 Turbo diffusion model. |
| `latent` | — | The same latent used for sampling; it is validated and passed through unchanged. |
| `steps` | `8` | Number of Euler denoising steps. The reference Turbo setup uses eight. |
| `denoise` | `1.0` | Uses the complete schedule at `1.0`; lower values retain the final requested steps from a longer schedule. |

## Notes

- Designed for Krea2 models using the `12 x 2560` Krea2 text-conditioning layout.
- The node returns only a patched `MODEL`; it does not modify prompt text or require extra conditioning nodes.
- If the loaded model does not match the expected Krea2 text-fusion layout, the patch is skipped.
- The advanced node restores every temporary runtime patch after each model call and does not store debug counters or step-local state in the model config. With the same seed and the same node parameters, ComfyUI can reuse cached graph results normally.
- The reference sigma node uses the Turbo fixed timestep shift `mu=1.15`, based on the official Krea 2 Turbo scheduler settings. It validates that the connected image dimensions are divisible by 16. Turbo does not use the RAW checkpoint's resolution-dependent shift rule.
