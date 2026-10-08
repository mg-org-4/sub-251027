# KIE Grok Imagine (I2V) API Spec

## Status
Reference spec for the implemented Grok Imagine image-to-video node. See [`KIE_GrokImagine_I2V.md`](KIE_GrokImagine_I2V.md) for the ComfyUI node surface.

## Endpoint
- Method: `POST`
- Path: `/api/v1/jobs/createTask`
- Base URL: `https://api.kie.ai`
- Model: `grok-imagine/image-to-video`

## Request Body
```json
{
  "model": "grok-imagine/image-to-video",
  "callBackUrl": "https://your-domain.com/api/callback",
  "input": {
    "prompt": "POV hand comes into frame handing the girl a cup of take away coffee, the girl steps out of the screen looking tired, then takes it and she says happily: thanks! Back to work.",
    "mode": "normal",
    "duration": "6",
    "resolution": "480p"
  }
}
```

### Example: External Image Source
```json
{
  "model": "grok-imagine/image-to-video",
  "input": {
    "image_urls": [
      "https://file.aiquickdraw.com/custom-page/akr/section-images/1762247692373tw5di116.png"
    ],
    "prompt": "POV hand comes into frame handing the girl a cup of take away coffee, the girl steps out of the screen looking tired, then takes it and she says happily: thanks! Back to work.",
    "mode": "normal",
    "duration": "6",
    "resolution": "480p"
  }
}
```

### Example: Grok Task Source
```json
{
  "model": "grok-imagine/image-to-video",
  "input": {
    "task_id": "previous_grok_task_id",
    "index": 0,
    "prompt": "POV hand comes into frame handing the girl a cup of take away coffee, the girl steps out of the screen looking tired, then takes it and she says happily: thanks! Back to work.",
    "mode": "spicy",
    "duration": "6",
    "resolution": "480p"
  }
}
```

## Root Parameters
- `model` (STRING, required): Must be `grok-imagine/image-to-video`.
- `callBackUrl` (STRING, optional): Receives task completion notifications when provided.
- `input` (OBJECT, required): Generation parameters for the model.

## Input Parameters
- `image_urls` (ARRAY[URL], optional): One external image URL only. Accepted types: `image/jpeg`, `image/png`, `image/webp`. Max size: `10MB`.
- `task_id` (STRING, optional): Task id of a previous Grok image generation to reuse as the source image. Max length: `100`.
- `index` (NUMBER, optional): Which generated image to use from the `task_id` source. Min: `0`, Max: `5`. Grok generates 6 images per task.
- `prompt` (STRING, optional): Motion prompt for the generated video. Max length: `5000`.
- `mode` (STRING, optional): One of `fun`, `normal`, `spicy`.
- `duration` (STRING, optional): One of `6`, `10`, `15` seconds.
- `resolution` (STRING, optional): One of `480p`, `720p`.

## Source Image Rules
- Provide exactly one source method:
  - `image_urls` for an uploaded external image
  - `task_id` plus `index` for a Grok-generated source image
- Do not provide both `image_urls` and `task_id` in the same request.
- External image mode supports one image only.
- When using external images, `spicy` mode is not supported and will automatically switch to `normal` according to the docs.
- `index` only applies when `task_id` is used, and is ignored when `image_urls` is used.

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
If `callBackUrl` is present, KIE posts task completion payloads to that URL for both success and failure states.

### Success Callback Fields
- `code`
- `data.completeTime`
- `data.costTime`
- `data.createTime`
- `data.model`
- `data.param`
- `data.resultJson`
- `data.state`
- `data.taskId`
- `data.failCode`
- `data.failMsg`
- `msg`

### Failure Callback Fields
- `code`
- `data.completeTime`
- `data.costTime`
- `data.createTime`
- `data.failCode`
- `data.failMsg`
- `data.model`
- `data.param`
- `data.state`
- `data.taskId`
- `data.resultJson`
- `msg`

## ComfyUI Mapping Notes
Recommended node shape for this repo:
- `images`: `IMAGE`, optional. If connected, upload the first image only.
- `task_id`: `STRING`, optional.
- `index`: `INT`, optional, default `0`, range `0..5`.
- `prompt`: `STRING` multiline, default empty string.
- `mode`: `COMBO`, optional.
- `duration`: `COMBO`, optional.
- `resolution`: `COMBO`, optional.
- `log`: `BOOLEAN`, optional.
- Output: `VIDEO` for the first implementation pass.

Recommended validation behavior:
- Require exactly one source method:
  - `images`
  - `task_id` plus `index`
- Reject `images` together with `task_id`.
- If multiple ComfyUI images are connected, upload the first image only.
- If `task_id` is not provided, ignore `index` rather than erroring, because the widget will still exist in the ComfyUI node UI.
- If external images are used with `mode=spicy`, log a warning and send `normal`.

## Implementation Notes
- `callBackUrl` should stay transport-level and does not need to be exposed in the first ComfyUI node pass.
- The cleanest future chain is likely from a Grok image-generation node that outputs a `task_id`, which can then feed this node with an `index`.
- The docs do not describe a video-extension or last-frame chaining behavior for reusing this node with the output task id of a prior I2V run.
- For that reason, the first implementation should not expose `task_id` as a chained output from the I2V node itself.
- The provided request example includes `image_urls` and `index` together. That conflicts with the written rule that `index` only applies to `task_id`, so the written rule should be treated as the safer contract.
- The provided callback success example uses a `.jpg` URL in `resultUrls`, which conflicts with the endpoint type. Treat that as a documentation inconsistency unless live API behavior proves otherwise.
