# AD Per Block Floats (SD1.5)

Use Floats from Value Schedules to select SD1.5 effect and scale values for blocks.

## Inputs

- `effect_21_floats`: Optional effect values. The list is extended to the required 21 values when needed.
- `scale_21_floats`: Optional scale values. The list is extended to the required 21 values when needed.

## Block index map

SD1.5 motion modules contain 21 blocks.

| Index | Block |
| ---: | --- |
| 0 | Start of down blocks (`down_0__0`) |
| 7 | End of down blocks (`down_3__1`) |
| 8 | Mid block (`mid`) |
| 9 | Start of up blocks (`up_0__0`) |
| 20 | End of up blocks (`up_3__2`) |

## Outputs

- `PER_BLOCK`: Per-block effect and scale configuration for an SD1.5 motion module.
