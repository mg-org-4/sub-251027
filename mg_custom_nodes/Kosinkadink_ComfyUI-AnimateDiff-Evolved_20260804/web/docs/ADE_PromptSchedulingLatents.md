# Prompt Scheduling [Latents]

Encode a schedule of prompts with automatic interpolation, its length matching the passed-in latent count.

## Schedule format

Schedules support JSON and Python-like formats. Frame 0 is the first frame and `max_frames - 1` is the last.

```text
"0": "blue rock on mountain",
"16": "green rock in lake"
```

```text
0 = "blue rock on mountain",
16 = "green rock in lake"
```

Prompts must be enclosed in double quotes. Prompt portions may use keys supplied through `values_replace`.

### Allowed indices

- **Single:** A positive integer such as `0` or `2` selects that frame. A negative integer such as `-1` or `-5` selects from the end (`-1` is the last frame). A decimal such as `0.5` or `1.0` selects a relative position (`0.5` is halfway and `1.0` is the last frame).
- **Range:** `start:end` uses an uninterpolated prompt from the included start index to the excluded end index. Examples: `0:12`, `0:-5`, `2:0.5`.
- **Hold:** A colon after one index stops interpolation until the next supplied index. Examples: `0:`, `0.5:`, `16:`.

## Inputs

| Input | Description |
| --- | --- |
| `prompts` | The prompt schedule. |
| `clip` | CLIP used to encode prompts. |
| `latent` | Supplies the frame count used as the schedule length. |
| `prepend_text` | Optional text added before every prompt. |
| `append_text` | Optional text added after every prompt. |
| `values_replace` | Optional value schedules substituted for keys written as `` `some_key` `` in prompts. |
| `tensor_interp` | Prompt-conditioning interpolation method; defaults to linear interpolation. |
| `print_schedule` | Print the resulting schedule when enabled. |

## Outputs

- `CONDITIONING`: Encoded prompts.
- `LATENT`: The unmodified input latents, usable as a pipe or safely ignored.
