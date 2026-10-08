# KIE Kling 3.0 Motion-Control (I2V) API Spec

## Status
Reference spec for the implemented Kling 3.0 Motion-Control image-to-video node. See [`KIE_Kling3_Motion_I2V.md`](KIE_Kling3_Motion_I2V.md) for the ComfyUI node surface.

This endpoint is structurally very close to the existing Kling 2.6 motion-control flow:
- one required reference image
- one required motion-driver video
- optional prompt
- optional character orientation control
- optional resolution/mode control

See [`KIE_Kling26Motion_I2V.md`](KIE_Kling26Motion_I2V.md) for the earlier node this implementation mirrors closely.

## Endpoint
- Method: `POST`
- Path: `/api/v1/jobs/createTask`
- Base URL: `https://api.kie.ai`
- Model: `kling-3.0/motion-control`

## Request Body
```json
{
  "model": "kling-3.0/motion-control",
  "callBackUrl": "https://your-domain.com/api/callback",
  "input": {
    "prompt": "No distortion, the character's movements are consistent with the video.",
    "input_urls": [
      "https://static.aiquickdraw.com/tools/example/1773115240203_t8pIR73J.png"
    ],
    "video_urls": [
      "https://static.aiquickdraw.com/tools/example/1773115131888_zBZHuynR.mp4"
    ],
    "character_orientation": "video",
    "mode": "720p"
  }
}
```

## Root Parameters
- `model` (STRING, required): Must be `kling-3.0/motion-control`.
- `callBackUrl` (STRING, optional): Callback target for job completion. Transport-level field; not needed in the first ComfyUI node pass.
- `input` (OBJECT, required): Generation parameters for the model.

## Input Parameters
- `prompt` (STRING, optional): Output description. Max length: `2500`.
- `input_urls` (ARRAY[URL], required): Reference image URLs. Current docs describe a single reference image use case. Accepted types: `image/jpeg`, `image/png`, `image/jpg`. Max size: `10MB`. Image must be larger than `300px`. Aspect ratio must be between `2:5` and `5:2`.
- `video_urls` (ARRAY[URL], required): Motion-driver video URLs. Accepted types: `video/mp4`, `video/quicktime`. Max size: `100MB`.
- `character_orientation` (STRING, optional): `image` or `video`.
  - `image`: keep orientation aligned to the image subject. Max reference-video duration: `10s`.
  - `video`: keep orientation aligned to the driver video. Max reference-video duration: `30s`.
- `mode` (STRING, optional): Output resolution mode.

## Important Doc Mismatch
The pasted API docs conflict on `mode`:
- prose says use `std` for 720p and `pro` for 1080p
- options table lists `720p` and `1080p`
- request example sends `720p`

For this ComfyUI implementation, the safest pinned contract is:
- expose `720p` and `1080p` in the UI
- send `720p` or `1080p` in payload

That matches the explicit example payload and the option table. If live API testing later proves that `std` and `pro` are required instead, the node should normalize the UI values before submission and keep the UI surface stable.

## Media Rules
- Provide at least one image URL in `input_urls`.
- Provide at least one video URL in `video_urls`.
- For the first ComfyUI implementation pass, upload and send exactly one image and one video, matching the documented usage pattern.
- If multiple ComfyUI images are connected, upload only the first image and log that behavior.
- The driver video must respect the endpoint's orientation-based duration rule:
  - `character_orientation=image`: max `10s`
  - `character_orientation=video`: max `30s`

## Success Response
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "taskId": "task_12345678"
  }
}
```

## Callback Behavior
If `callBackUrl` is present, KIE posts completion payloads to that URL for both success and failure states.

### Success Callback Example Shape
```json
{
  "code": 200,
  "data": {
    "completeTime": 1755599644000,
    "costTime": 8,
    "createTime": 1755599634000,
    "model": "kling-3.0/motion-control",
    "param": "{\"callBackUrl\":\"https://your-domain.com/api/callback\",\"model\":\"kling-3.0/motion-control\",\"input\":{\"prompt\":\"No distortion, the character's movements are consistent with the video.\",\"input_urls\":[\"https://static.aiquickdraw.com/tools/example/1773115240203_t8pIR73J.png\"],\"video_urls\":[\"https://static.aiquickdraw.com/tools/example/1773115131888_zBZHuynR.mp4\"],\"character_orientation\":\"video\",\"mode\":\"720p\"}}",
    "resultJson": "{\"resultUrls\":[\"https://example.com/generated-image.jpg\"]}",
    "state": "success",
    "taskId": "e989621f54392584b05867f87b160672",
    "failCode": null,
    "failMsg": null
  },
  "msg": "Playground task completed successfully."
}
```

### Failure Callback Example Shape
```json
{
  "code": 501,
  "data": {
    "completeTime": 1755597081000,
    "costTime": 0,
    "createTime": 1755596341000,
    "failCode": "500",
    "failMsg": "Internal server error",
    "model": "kling-3.0/motion-control",
    "param": "{\"callBackUrl\":\"https://your-domain.com/api/callback\",\"model\":\"kling-3.0/motion-control\",\"input\":{\"prompt\":\"No distortion, the character's movements are consistent with the video.\",\"input_urls\":[\"https://static.aiquickdraw.com/tools/example/1773115240203_t8pIR73J.png\"],\"video_urls\":[\"https://static.aiquickdraw.com/tools/example/1773115131888_zBZHuynR.mp4\"],\"character_orientation\":\"video\",\"mode\":\"720p\"}}",
    "state": "fail",
    "taskId": "bd3a37c523149e4adf45a3ddb5faf1a8",
    "resultJson": null
  },
  "msg": "Playground task failed."
}
```

## Callback Notes
- Callback payload shape matches the task query payload shape.
- `param` contains the full original `createTask` request, not only `input`.
- If `callBackUrl` is omitted, no callback is sent.

## Result-Type Note
The provided success callback example shows a `.jpg` URL in `resultUrls`, which conflicts with the endpoint being motion-controlled image-to-video.

For implementation, treat this as a documentation inconsistency and keep the node aligned to the endpoint mode:
- poll for task completion
- extract the first result URL
- download and return it as a `VIDEO` output

## ComfyUI Mapping Notes
Recommended first-pass node shape for this repo:
- `prompt`: `STRING`, multiline, optional, default empty string
- `images`: `IMAGE`, required
- `video`: `VIDEO`, required
- `character_orientation`: `COMBO`, optional, default `video`
- `mode`: `COMBO`, optional, default `720p`
- `log`: `BOOLEAN`, optional
- output: `VIDEO`

Recommended behavior:
- reuse the current Kling 2.6 motion-control upload/poll/download pattern
- keep `callBackUrl` out of the UI
- upload the first image only
- upload one driver video
- validate prompt length against `2500`
- validate `character_orientation` against `image|video`
- validate `mode` against `720p|1080p` unless live testing proves the transport values must be `std|pro`
- fail early if no video is connected
- return a ComfyUI `VIDEO` object backed by the downloaded result

## Implementation Notes
The current implementation follows that low-risk path:
1. reuses the Kling 2.6 motion-control helper shape
2. switches `MODEL_NAME` to `kling-3.0/motion-control`
3. keeps the same image-upload and driver-video upload flow
4. pins validation to the new model's prompt/media constraints
5. documents the mode-value ambiguity and normalizes UI-facing values defensively

The transport and node contract are materially the same as Kling 2.6 Motion-Control, so this ships as a straightforward sibling node rather than a new workflow shape.
