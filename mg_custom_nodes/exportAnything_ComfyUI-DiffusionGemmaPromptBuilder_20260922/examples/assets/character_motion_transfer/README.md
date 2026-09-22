# Character Motion Transfer Assets

Assets for `examples/07_ltx23_character_motion_transfer.json`.

- `character_reference.png` - reference image used for character identity and appearance.
- `motion_control_video.mp4` - source video used for pose, action, timing, camera choreography, depth, Canny/edge layout, composition, and scene geometry.
- `diffusiongemma_ltx_character_demo.mp4` - short demonstration output for the Canny, Depth, and DWPose control settings.
- `diffusiongemma_ltx_character_demo.gif` - README preview GIF generated from the demo output.

To run the workflow without changing loader widgets, copy this folder to:

```text
ComfyUI/input/character_motion_transfer
```

The workflow expects:

```text
character_motion_transfer/character_reference.png
character_motion_transfer/motion_control_video.mp4
```
