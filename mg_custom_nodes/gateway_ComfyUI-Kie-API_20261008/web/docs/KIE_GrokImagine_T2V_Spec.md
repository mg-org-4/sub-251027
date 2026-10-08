# KIE Grok Imagine (T2V) API Spec

## Status
Reference spec for the implemented Grok Imagine text-to-video node. See [`KIE_GrokImagine_T2V.md`](KIE_GrokImagine_T2V.md) for the ComfyUI node surface.

## Endpoint
- Method: `POST`
- Path: `/api/v1/jobs/createTask`
- Base URL: `https://api.kie.ai`
- Model: `grok-imagine/text-to-video`

## Request Body
```json
{
  "model": "grok-imagine/text-to-video",
  "callBackUrl": "https://your-domain.com/api/callback",
  "input": {
    "prompt": "A couple of doors open to the right one by one randomly and stay open, to show the inside, each is either an living room, or a kitchen, or a bedroom or an office, with little people living inside.",
    "aspect_ratio": "2:3",
    "mode": "normal",
    "duration": "6",
    "resolution": "480p"
  }
}
```

## Root Parameters
- `model` (STRING, required): Must be `grok-imagine/text-to-video`.
- `callBackUrl` (STRING, optional): Receives task completion notifications when provided.
- `input` (OBJECT, required): Generation parameters for the model.

## Input Parameters
- `prompt` (STRING, required): Motion prompt for the generated video. Max length: `5000`.
- `aspect_ratio` (STRING, optional): One of `2:3`, `3:2`, `1:1`, `9:16`, `16:9`.
- `mode` (STRING, optional): One of `fun`, `normal`, `spicy`.
- `duration` (STRING, optional): One of `6`, `10`, `15` seconds.
- `resolution` (STRING, optional): One of `480p`, `720p`.

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
- `prompt`: `STRING` multiline, required.
- `aspect_ratio`: `COMBO`, optional.
- `mode`: `COMBO`, optional.
- `duration`: `COMBO`, optional.
- `resolution`: `COMBO`, optional.
- `log`: `BOOLEAN`, optional.
- Output: `VIDEO`.

This fits the same helper pattern already used by Kling and Seedance text-to-video nodes:
- validate prompt and enum inputs before submission
- create task through `kie_api/jobs.py`
- poll through `kie_api/jobs.py`
- extract `resultUrls` through `kie_api/results.py`
- download video and convert through `kie_api/video.py`
- log credits through `kie_api/credits.py`

## Implementation Notes
- `callBackUrl` should stay transport-level and does not need to be exposed in the first ComfyUI node pass.
- Internal polling defaults can follow the current async video node pattern in this repo.
- The provided callback success example uses a `.jpg` URL in `resultUrls`, which conflicts with the endpoint type. Treat that as a documentation inconsistency unless live API behavior proves otherwise.
