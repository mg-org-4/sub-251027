# V3.8X Release and Migration Policy

> Historical V3.8X / package 3.8.1 record. Current `main` uses package 3.9.0,
> retains the V3.8X2 Sampler/workflow, and adds a separate V3.9 Sampler plus
> Reference Images V3.9 helper. V3.8X2 registers ten IDs, including two
> deprecated loader compatibility IDs; V3.9 registers twelve in total.
> Use the dedicated V3.9 workflow for its new
> Reference wiring. Old workflows and saved Runs/Takes are not automatically
> converted. See the main README for the current installation path and gates.

## Supported release

The historical V3.8X package version `3.8.1` exported seven Node IDs:

```text
H3ContinuumSamplerV38
H3ContinuumReferenceAudios
H3ContinuumAssembleSeamV35
H3EasyLoadImage
H3EasyLoadAudio
H3ContinuumLoadVideo
H3ContinuumSecondPassV35
```

The assembler, loader, and Second Pass IDs are retained while their V3.8 display names are simplified. AUDIO-R1 adds one modular Reference Audios helper and appends one optional Sampler bundle socket. The Resolution / Size Source UX keeps the old Aspect input hidden for compatibility and appends Size Source, Width, and Height; saved fixed-Aspect workflows migrate to an equivalent Manual canvas. Sampling, Run Storage, State/Session, and Assembly contracts are unchanged.

## Saved workflows from older releases

Only the seven IDs listed above were exported by that historical V3.8X package, including the retained Finalize, loader, and Second Pass IDs. The current main source's separate V3.8X2/V3.9 surfaces are described above. ComfyUI can show an unknown node when a saved workflow refers to an ID outside the installed package's mappings. Do not replace it with a superficially similar node without checking its saved contract.

- Use tag `v3.7.0` for V3.7 workflows.
- Use tag `v3.6.0` for V3.6 workflows.
- Use tag `v3.8.0` for the prior V3.8.0 package and its supplied workflow.
- V3.4/V3.5 archives will be finalized during Release Preparation. Until then, use the matching historical commit or release asset already associated with that workflow.

Do not mix a historical workflow's private schema with the V3.8X package. Install the matching historical package, open and render the workflow there, or rebuild the graph explicitly with the V3.8X public nodes.

## V3.8X workflow and external integration

The declared Registry payload and GitHub distribution use `examples/workflows/MiniMax_H3_Continuum_V38x.json` and `examples/workflows/MiniMax_H3_Continuum_V38x.zip`. The ZIP contains exactly the same JSON: one V3.8X graph, not two variants. This is the user-supplied Spectrum-capable workflow, not the former dependency-free Core template. It requires external Spectrum, rgthree, and KJNodes nodes, which Continuum does not bundle or install. ComfyUI-Easy-Use is no longer required. Spectrum and all Turbo LoRA entries are saved disabled; users may enable one task-matched path, but the external custom-node types remain in the graph.

The supplied JSON and ZIP are preserved byte-for-byte, including prompt, media filenames, custom titles, and settings. Select local files and disable unused inputs before rendering. A saved title such as `Save 3x5s Video` does not control duration; the Sampler settings do. The output path uses Core Video/Audio Decode, Finalize, Core Create Video, and Core Save Video directly. External latent processors and upscalers connect through H3 Continuum Second Pass; the old integrated Hi-Res Fix is not part of the V3.8X standard workflow. The supported processor, decoder, and accelerator boundaries are defined in [V3.8 Open Integration Contract](V38_OPEN_INTEGRATION_CONTRACT.md). Registry validate/pack and publication remain separate gates; a GitHub source commit does not claim they have run.

Tests, runner/probe/diagnostic tools, internal development documents, Labs nodes, and historical workflows remain available in the GitHub source for development and audit purposes but are excluded from the Registry archive. The archive retains `tools/__init__.py` and `tools/p20_file_backed_tensor_poc.py`, which are runtime dependencies of the Production file-backed buffer path, plus `tools/verify_runtime.py` for verification. `MANIFEST.sha256` validates its declared source files; the staged archive instead carries `REGISTRY_MANIFEST.sha256`. These development assets are not part of the supported V3.8 searchable node surface.
