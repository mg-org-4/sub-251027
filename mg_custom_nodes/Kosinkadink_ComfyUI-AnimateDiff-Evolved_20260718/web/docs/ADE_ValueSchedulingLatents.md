# Value Scheduling [Latents]

Create a list of values with automatic interpolation, its length matching the passed-in latent count.

## Schedule format

Schedules support JSON and Python-like formats. Values need no special formatting.

```text
"0": 1.0,
"16": 1.3
```

```text
0 = 1.0,
16 = 1.3
```

Frame 0 is the first frame and `max_frames - 1` is the last.

- **Single:** Positive integers select a frame, negative integers select from the end (`-1` is last), and decimals select a relative position (`0.5` is halfway and `1.0` is last).
- **Range:** `start:end` holds the start value without interpolation through the excluded end. Examples: `0:12`, `0:-5`, `2:0.5`.
- **Hold:** A trailing colon stops interpolation until the next index. Examples: `0:`, `0.5:`, `16:`.

## Inputs

- `values`: The value schedule.
- `latent`: Supplies the frame count used as the schedule length.
- `print_schedule`: Print each output value when enabled.

## Outputs

The schedule is returned as `FLOAT`, `FLOATS`, rounded `INT`, and rounded `INTS` outputs.
