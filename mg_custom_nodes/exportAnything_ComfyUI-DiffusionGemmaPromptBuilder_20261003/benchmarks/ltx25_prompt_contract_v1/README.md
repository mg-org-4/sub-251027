# LTX-2.5 prompt contract benchmark

This fixture defines a primary controlled A/B and one diagnostic arm. All
downstream generation controls and paired seeds are identical:

- A / `raw`: send the compact brief through the native LTX-2.5 prompt enhancer.
- B / `revised`: send the mode-aware DiffusionGemma LTX-2.5 prompt with native enhancement disabled.
- `current`: optional legacy DiffusionGemma diagnostic, also with native enhancement disabled.

The checked-in cases cover text-to-video (`t2v`), image-to-video (`i2v`), and
first/last-frame interpolation (`flf`). The default score is entirely offline:
it loads no model, contacts no service, and writes nothing unless an output
option is supplied.

```powershell
C:\ComfyUI\.venv\Scripts\python.exe scripts\benchmark_ltx25_prompts.py
```

To preserve the A/B controls for optional renders, emit the full prompt × seed
matrix:

```powershell
C:\ComfyUI\.venv\Scripts\python.exe scripts\benchmark_ltx25_prompts.py `
  --emit-generation-plan .tmp\ltx25-generation-plan.json `
  --output .tmp\ltx25-structural-report.json
```

Emitting a plan does not execute ComfyUI. Every candidate for a case receives
the same model, precision, resolution, duration, sampler, scheduler, step
counts, CFG, conditioning assets/strength, negative prompt, and seeds. The
primary A/B changes only the prompt-producing condition: native enhancement of
the raw brief versus the DiffusionGemma output. Each plan record includes a
shared `downstream_control_sha256` so a renderer can reject confounded pairs.

Optional runtime results can be merged with `--runtime-results FILE`. The file
uses schema `dg-ltx25-runtime-results/1`, references the manifest SHA-256 shown
in the structural report, and contains one or more runs of this form:

```json
{
  "case_id": "t2v_single_exchange_4s",
  "candidate": "revised",
  "seed": 17,
  "output": "C:/ComfyUI/app/output/ltx25-revised-17.mp4",
  "ratings": {
    "prompt_adherence": 4.5,
    "motion_coherence": 4.0,
    "visual_continuity": 4.0,
    "audio_alignment": 3.5
  },
  "notes": "Blind human rating on a zero-to-five scale."
}
```

Structural scores test contract compliance and feasibility; they do not claim
to predict visual quality. Runtime ratings are therefore reported separately.
