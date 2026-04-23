# Command Log - Chrono/Wan Temporal Edit Scope Split and Defer Contract

Date: 2026-04-23
Environment: Windows PowerShell, repo-local workspace, branch `dev`

## Commands

1. `Get-Content 'reference/workflow_templates/imageEdit/Chrono Edit 14B.json' | Select-Object -First 260`
2. `rg -n 'WanImageToVideo|ScaleROPE|CLIPVisionEncode|CLIPVisionLoader|ImageFromBatch' reference/ComfyUI/comfy_extras -g '*.py'`
3. `Get-Content .planning/plans/260418-R157_OFFICIAL_EDIT_TEMPLATE_I2I_INTAKE_PLAN.md | Select-Object -First 220`

## Manual review checkpoints

- Confirmed `Chrono Edit 14B` is not a normal static image-edit graph.
- Confirmed the temporal/video-like classification is supported by dedicated host nodes rather than guesswork from the filename alone.
- Confirmed the defer must be categorical for Wan-style temporal/video-like edit graphs, not a one-file exception.
- Recorded the current-session user-instruction override that permits `.planning/` artifacts to be committed despite the repo-standard ignore rule.
