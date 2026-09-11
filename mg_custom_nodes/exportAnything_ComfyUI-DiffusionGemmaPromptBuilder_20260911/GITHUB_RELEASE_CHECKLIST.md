# GitHub and Comfy Registry Release Checklist

Use this checklist from the standalone `ComfyUI-DiffusionGemmaPromptBuilder` repository before pushing or publishing a release.

## Product Boundary

- Keep the exported surface at fifty-five node ids: the established thirty-three-node surface remains unchanged, and the separate Advertisement surface adds exactly twenty-two ids:
  - eleven stable Director nodes: Model Loader, Context Hub, H3 Reference Context, Grounding Guard Settings, CoT Generator, JSON Splitter, Generation Gate, Branch Generation Gate, and the LTX-2.5, MiniMax-H3, and Ideogram 4 target profiles
  - one deprecated universal Target Profile retained for saved-workflow compatibility
  - two optional reference-preparation helpers: Reference Prep and H3 Reference Pair Prep
  - two experimental SplatStage planners
  - eleven audio/music-video controls: ten decoded-audio, source-routing, ACE, performance, and production-concept nodes plus Timed Lyrics Analyzer
  - six production-planning controls: Project Master, Audio-Aware H3 Multi-Shot Planner, H3 Relay Reference Gate, Multi-Format Delivery Planner, H3 Shot Seed Fanout, and H3 Shot Assembler
  - six Advertisement contract/settings nodes: Workflow Controls, Campaign Contract, Reference Contract, Reference Asset Prep, Soundtrack Contract, and Music3 Prompt Adapter
  - three Advertisement audio nodes: Soundtrack Source, Audio Audition & Lock, and Motion Guide / Final Mix
  - six Advertisement planning/relay nodes: Director Packet Repair, Master Contract, Planning Defaults, Multi-Shot Planner, Relay Artifact, and Relay Gate
  - six Advertisement finishing/QA nodes: Master Assembler, End Card Renderer, Master Finisher, Adaptation Status, Cutdown Renderer, and Media QA Gate
  - one Advertisement runtime node: Memory Barrier
- Keep the two SplatStage planners visibly experimental. Their mapping ids stay compatible, their display names end in `(Experimental)`, and their category is `prompt/diffusiongemma/experimental-motion-planning`.
- Confirm `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` contain the same fifty-five ids, with no collision between the established thirty-three and the separate twenty-two Advertisement ids.
- Treat `ltx_reframe.py` and the reframe workflows as unexported prototypes. Do not describe them as installed nodes unless the package entry point and frontend are deliberately restored together.
- Do not bundle model weights or generated tensor dumps. The NVFP4 bridge remains source code in `nodes.py`.

## Repository Preflight

1. Run `git rev-parse --show-toplevel` and confirm it resolves to this package directory, not the parent ComfyUI checkout.
2. Confirm `origin` is `https://github.com/exportAnything/ComfyUI-DiffusionGemmaPromptBuilder` and that the branch is based on the public repository history.
3. Confirm the worktree is clean and the intended release commit is pushed.
4. Confirm the repository URLs in `pyproject.toml` are current.
5. Keep `[tool.comfy].PublisherId` exactly `exportanything`; this is the immutable Registry id, while `exportAnything` is its display name.
6. Confirm `project.version` is newer than the latest Registry version. Version `0.1.0` is already published, so this release begins at `0.2.0`.
7. Confirm `requires-comfyui = ">=0.30.1"` and record ComfyUI `0.30.1` as the tested baseline. The Registry field is a compatibility range, not a test-report field.
8. Confirm `LICENSE` and the Python copyright headers match the intended release policy.

## Automated Gates

Run the same three gates enforced by `.github/workflows/ci.yml`:

```text
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/smoke_prompt_builder.py
node --check web/js/model_loader.js
node --test tests/*.mjs
```

- The unit suite must validate the exact fifty-five-node surface, preserve the established thirty-three-node boundary, validate the separate twenty-two-node Advertisement contract/audio/planning/finishing/runtime split, the dedicated target-node boundaries, both bundled SplatStage schemas, and retained reframe helpers.
- The readiness/schema smoke must finish with `Prompt Builder readiness/schema smoke passed`.
- The JavaScript suite must exercise the shipped `model_loader.js`. No test may import the retired `ltx_reframe_layout.js`.
- The default release suite validates checked-in artifacts only. Set `DG_VERIFY_DEPLOYED_V5_ARTIFACTS=1` solely when intentionally auditing the superseded, mutable V5 Desktop and ComfyUI copies; divergence there is deployment evidence to reconcile, not a hermetic release gate.
- Require all three CI jobs to pass on the release commit.

## Manual Runtime Gates

1. Run the NVFP4 proof gate on a CUDA workstation with the real model:

   ```text
   python proof_gates.py --model-path C:\ComfyUI\models\LLM\diffusiongemma-26B-A4B-it-NVFP4
   ```

2. Fresh-install the repository in a clean ComfyUI `0.30.1` custom-nodes folder and confirm the established thirty-three-node surface still imports without regression. On ComfyUI `>=0.33.1`, confirm all fifty-five nodes appear: the established thirty-three plus the separate twenty-two Advertisement nodes under `prompt/diffusiongemma/advertising`.
3. Copy `examples/assets/character_motion_transfer` to `ComfyUI/input/character_motion_transfer` and load `examples/07_ltx23_character_motion_transfer.json`.
4. Load the MiniMax H3 T2VA and Ref2VA examples and confirm the Generation Gate blocks invalid or empty prompt packets.
5. Load `examples/15_minimax_h3_ref2va_music_video_v6.json`, replace both unbundled identity-image choices, and verify the default ACE source remains lazy with a blank upload. Exercise Upload song, Natural, Dance, Lyrics, and optional relay as separate branch tests; do not infer a successful media render from graph validation alone.
6. On ComfyUI `>=0.33.1`, load `examples/16_minimax_h3_ref2va_advertisement_music3_v1.json`, select Picture 1 performer hero, Picture 2 same-performer sheet, and Picture 3 independent product sheet, and confirm the portable Director checkpoint path resolves locally. Verify MiniMax Music 3 remains the default and Upload song / Legacy ACE-Step stay lazy UI fallbacks.
7. Run the separate API fixture through the full two-lane H3 path. Require actual 30-second, 15-second, and 6-second review-draft videos, an exact 30-second soundtrack, all Director/QC/QA diagnostic roots, exact frame/audio clocks, and a persisted Picture 4 retained-tail relay. Queue completion alone is not proof.
8. Keep delivery blocked while performer identity, product identity, copy legibility, or audio sync remains `not_measured`. Review the media and record evidence before changing any QA state to `pass`; never treat deterministic end-card copy as proof that generated in-scene package text is exact.
9. Treat longer experimental SplatStage and reframe workflows as separate opt-in validation, not as evidence for the stable Director compatibility promise.

## Files Tracked on GitHub

- Package entry point and source: `__init__.py`, `nodes.py`, `audio_production_nodes.py`, `production_planning_nodes.py`, `timed_lyrics_nodes.py`, `advertising_contract_nodes.py`, `advertising_audio_nodes.py`, `advertising_planning_nodes.py`, `advertising_finishing_nodes.py`, `advertising_runtime_nodes.py`, `ltx25_contract.py`, `ltx_reframe.py`, `proof_gates.py`
- Metadata and policy: `README.md`, `LICENSE`, `requirements.txt`, `pyproject.toml`, `.gitignore`, `.comfyignore`, this checklist
- Runtime contracts: `schemas/`, `web/`, `scripts/`
- Verification: `tests/`, `.github/workflows/ci.yml`
- Publishing: `.github/workflows/publish_action.yml`
- Documentation, current examples, workflow migrations, offline benchmarks, and GitHub demo assets: `docs/`, `examples/`, `tools/`, `benchmarks/`

## Registry Archive

- Keep `.github/`, `docs/`, `tests/`, `tools/`, `benchmarks/`, runtime artifacts, caches, temporary files, and large demo video/GIF files out of the Registry archive through `.comfyignore`.
- Confirm both SplatStage schemas remain in the Registry archive because the experimental planners load them at runtime.
- Inspect the archive before publishing and confirm it contains no model weights, migration-only local paths, logs, caches, or generated outputs.
- Publish only after the GitHub release commit and version metadata agree.
