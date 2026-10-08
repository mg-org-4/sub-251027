# KIE Kling 3.0 Motion-Control (I2V)

## Overview
Generate a short video clip from a single reference image and a motion-driver video using the Kling 3.0 motion-control model.

## Inputs
- prompt (STRING, multiline, optional): Optional motion prompt. Maximum length: 2500 characters.
- images (IMAGE, required): Source image batch; the first image is used.
- video (VIDEO, required): Motion-driver reference video clip.
- character_orientation (COMBO, default: video): Match character orientation to the image or the video. Options: image, video.
- mode (COMBO, default: 720p): Output resolution. Options: 720p, 1080p.
- log (BOOLEAN, default: true): Enable KIE console logs.

## Outputs
- video (VIDEO): ComfyUI VIDEO output referencing a temporary .mp4 file.

## Validation Notes
- The first image must be larger than 300px in both dimensions.
- The first image aspect ratio must stay between 2:5 and 5:2.
- The uploaded motion video must not exceed 100MB.
- Only the first connected image is uploaded.
- `character_orientation=image` is documented by KIE as supporting up to 10s source video.
- `character_orientation=video` is documented by KIE as supporting up to 30s source video.

## Implementation Notes
- This node follows the same upload/poll/download flow as the existing Kling 2.6 Motion-Control node.
- `callBackUrl` is intentionally not exposed in the ComfyUI UI for the first pass.
- KIE's published docs conflict on the transport value for `mode`. This node exposes and sends `720p` / `1080p`, matching the explicit request example and options table in the docs.
- The KIE callback example shows a `.jpg` result URL, which conflicts with this endpoint type. This node treats the endpoint as video and downloads the first result as VIDEO output.

## Troubleshooting
- This node uses internal defaults for polling/retries/timeouts (not exposed in the UI).
- If a job takes unusually long or fails, check https://kie.ai/logs.
- Insufficient credits: Check remaining credits and top up your KIE account.
