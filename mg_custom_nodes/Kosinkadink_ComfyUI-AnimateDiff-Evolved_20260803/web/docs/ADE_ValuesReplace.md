# Add Values Replace

Add a value schedule bound to a key for use in a Prompt Scheduling node.

## Inputs

- `value_key`: The key for the value schedule. It may contain only `a-z`, `A-Z`, `0-9`, and `_`. Refer to it in a prompt by surrounding it with backticks, for example `` `some_key` ``.
- `floats`: A list of floats, typically produced by a Value Scheduling node.
- `prev_replace`: Optional existing replacements, allowing multiple Values Replace nodes to be chained.

## Outputs

- `VALUES_REPLACE`: The replacement mapping for a Prompt Scheduling node.
