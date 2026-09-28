# R6 Node Facade Delta

## Purpose

R6 reduces only the public V3.8 Sampler method into this fixed boundary:

```text
public inputs
  -> RuntimeRequest
  -> existing hardening/runtime adapter
  -> RuntimeOutput
  -> public tuple
```

`v3/node_facade.py` owns the immutable request container and the V3.8
status-only resolution/diagnostic decoration.  It does not import or create a
Coordinator, Planner, Run Storage controller, Sampling Engine, or ReviewUnit.

## Preserved ownership

- `v2/sequence.run_sequence` remains the R4 signature-compatible hardening
  adapter and keeps its explicit 41-keyword signature.
- `InternalRuntimeCoordinator` remains the sole owner of continuation-source,
  Session/State priority, prefix validation, and Runtime execution ordering.
- `execution_planner` remains the sole owner of physical-group and Projection
  decisions.
- `run_storage` remains the sole owner of R5 immutable raw transactions and
  manifest publication.
- The V3.8 node still resolves its existing Size Source contract before the
  request boundary and calls the unchanged V3.7 runtime inheritance path once.

## Non-goals and compatibility

No public node ID, socket, widget order, output tuple, Workflow, Sampling,
Review, Run Storage schema, canonical selection, or hardening behavior is
changed.  Tensor/model identities are preserved; only keyword containers are
copied into immutable request mappings.  Duplicate runtime keywords fail
instead of being silently overridden, matching the prior explicit-keyword
call's fail-closed behavior.

This phase introduces no temporary adapter, new dependency, or future hook.
