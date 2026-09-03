# Condition Step Extraction

Extract a single conditioning step from a schedule of prompts.

## Inputs

- `conditioning`: Encoded prompts from a Prompt Scheduling node.
- `index`: The step to extract. It must be within the scheduled prompt range.

## Outputs

- `CONDITIONING`: The single conditioning step from the schedule.
