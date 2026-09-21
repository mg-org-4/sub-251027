# Krea reference validation

Tested locally on 2026-09-11 with Krea 2 Turbo FP8 scaled, Qwen3-VL-4B FP8 scaled, Qwen Image VAE, and an NVIDIA RTX PRO 6000 Blackwell Max-Q. Eight Euler/simple steps, CFG 1.

## Automated checks

17 focused unittest cases pass. They cover the weighted prediction formula, zero influence, the single-reference endpoint, reference order and weight scaling, disabled/zero-weight exclusion, repeat sigma evaluations, fade boundaries, schedule independence, input validation, image/CLIP re-encoding, independent image encoding, twelve-layer preservation, Context suffix copying/mask slicing, and schema/model validation.

The tests use synthetic predictions for numerical invariants. Real server runs additionally exercised the V3 Autogrow inputs, actual model loading, vision encoding, denoising, and VAE decoding.

The weight-slider regression tests now check a single reference at 0, 0.05, 0.25, 0.5, 1, and 1.5; independent removal; and combined weights above one. Weights directly scale the baseline difference. Earlier versions normalized weights, making a lone reference effectively on/off.

After the correction, a live-server sweep rendered a single reference at weights 0, 0.05, 0.25, 0.5, and 1 in both Context and Full modes, with a fixed seed and overall influence 0.7. All five decoded RGB outputs were distinct in each mode. Mean absolute RGB differences from weight zero were 0, 10.92, 28.77, 39.35, and 59.59 for Context, and 0, 12.34, 27.55, 50.15, and 85.53 for Full, on the 0–255 channel scale. The inspected low-weight images stayed close to the baseline; higher Full weights increasingly copied source faces/panels. This demonstrates working numerical strength control in the test case, not universally linear visual response. Images are in `output/Krea_Weight_Test/`.

## Initial fixed-seed visual comparison (before the weight correction)

The same dog-in-a-park prompt was generated at 512x768 with seeds 193328989555475, 193328989555476, and 193328989555477. References were a monochrome portrait with window panels and a blue/violet illustration.

Compared native text-only generation, zero reference influence, each reference alone, independent Full-mode blending, the old joint Encode Rebalance node, and the new Context-mode paths. Zero influence produced pixel-identical decoded RGB images to the native baseline for all three seeds.

The joint encoder copied portraits and panels in all three comparisons. Independent Full-mode blending also retained unwanted source faces and windows. Context-mode style and mixed outputs preserved the dog scene without those unwanted portraits/panels in the inspected set. The palette-only branch visibly changed the color treatment. With style weight 1 and palette weight 0.5, the blend remained closer to the style branch; relative weights are not linear measures of perceptual influence.

These are observations from a small test set, not evidence of universal style/identity separation or grid prevention. Subject and composition roles require further use-case validation.

## Timing and memory

Warm server execution times at 512x768, excluding queue wait:

| Path | Time |
| --- | --- |
| Text-only / zero influence | About 1.7 seconds |
| One Context reference | 3.3–3.6 seconds |
| Two Context references | About 5.0 seconds |
| Four Context references | About 8.4 seconds |

The first Context run incurred model-loading overhead and is excluded from warm timing. A four-reference run and a faded reference window with denoise 0.75 also completed successfully.

The updated live Krea workflow also completed at 1024x1536 in 28.35 seconds of server execution, using the user's current woman-on-a-park-bench prompt and two current illustration references. The result followed the scene without constructing a reference grid. Saved locally as `output/Krea_Reference_Test/live_workflow_00001_.png`.

Polling ComfyUI system stats during the warm Context runs observed about 33.7 GiB total device memory in use. This includes resident models and other GPU allocations; it is not incremental per-reference VRAM or an exact peak. The dynamic allocator is not fully represented by PyTorch's reserved-memory counter.

CUDA was exercised. CPU, ROCm, MPS, DirectML, XPU, NPU, explicit low-VRAM mode, cancellation during sampling, and arbitrary third-party patches were not separately validated. The implementation uses ComfyUI's existing loading, offloading, attention, and guider cleanup paths.
## Standard KSampler outputs

The guider now also outputs a patched MODEL and CONDITIONING for standard samplers. All 20 unit tests pass, including parity for independent weights, reference schedules, CFG, plain conditioning, and preservation of the original model. A real 512x768, eight-step Euler/simple KSampler render with two references and fading matched SamplerCustomAdvanced pixel-for-pixel at CFG 1 and the same seed. This verifies sampler parity, not improved reference composition quality.
