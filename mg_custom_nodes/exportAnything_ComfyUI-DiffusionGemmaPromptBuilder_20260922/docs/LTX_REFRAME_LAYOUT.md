# LTX Reframe Layout

`LTX Reframe Layout` is a preparation node for LTX video outpainting. It does not load a model or sample a new video.

## What the node does

1. Select or upload a source video.
2. Choose the final canvas aspect ratio and exact LTX-compatible dimensions.
3. Drag the red source rectangle to reposition the video.
4. Drag any red corner to resize it without changing the source aspect ratio. Scale below `1.0` zooms out; scale above `1.0` zooms in and crops the source at the target boundary.
5. Execute the node to decode the video and produce padded frames, an outpaint mask, a source mask, the original frames, audio, FPS, and placement metadata.

Target dimensions are snapped to multiples of 64. The source is never stretched. `source_scale` ranges from `0.1` to `4.0`: the fitted source is `1.0`, smaller values create outpaint space, and larger values crop/zoom into the source. In the preview, video beyond the target boundary is dimmed while the red resize handles remain visible. `position_x`, `position_y`, and `source_scale` are ordinary serialized ComfyUI widgets, so the visual placement is reproducible in saved workflows and API prompts.

## LTX 2.3 connection

- Connect `padded_frames` to `LTXVInpaintPreprocess.images`.
- Connect `outpaint_mask` to `LTXVInpaintPreprocess.mask`.
- Feed the preprocessed image into the official LTX 2.3 IC-LoRA outpaint guide path.
- Use `source_mask` with `padded_frames` for a final source-preserving composite if the generated result changes the original region.
- Connect `audio` and `fps` to the final video assembly nodes.

The official local reference workflow is:

`ComfyUI-LTXVideo/example_workflows/2.3/LTX-2.3_ICLoRA_Outpaint_Two_Stage_Distilled.json`

## Current boundary

The first version supports one static placement for the entire clip. Animated reframing/keyframes, rotation, perspective warping, per-shot layouts, and automatic subject-aware positioning are intentionally left for later nodes or workflow stages.
