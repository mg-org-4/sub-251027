# AD Per Block Floats (SDXL)

Use Floats from Value Schedules to select SDXL effect and scale values for blocks.

## Inputs

- `effect_16_floats`: Optional effect values. The list is extended to the required 16 values when needed.
- `scale_16_floats`: Optional scale values. The list is extended to the required 16 values when needed.

## Block index map

SDXL motion modules contain 16 blocks.

| Index | Block |
| ---: | --- |
| 0 | Start of down blocks (`down_0__0`) |
| 5 | End of down blocks (`down_2__1`) |
| 6 | Mid block (`mid`) |
| 7 | Start of up blocks (`up_0__0`) |
| 15 | End of up blocks (`up_2__2`) |

## Outputs

- `PER_BLOCK`: Per-block effect and scale configuration for an SDXL motion module.
