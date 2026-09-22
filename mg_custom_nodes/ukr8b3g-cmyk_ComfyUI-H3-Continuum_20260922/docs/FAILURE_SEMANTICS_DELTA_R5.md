# Failure Semantics Delta — R5

Date: 2026-09-14  
Status: R5 Design Delta PASS / Implementation PASS / Validation PASS  
Baseline: authoritative ComfyUI_W dirty tree, with R0–R4 PASS before R5

## Approved differences

R5 may change only these two externally observable failure details.

### D1: terminal physical-group manifest visibility

Previously, the first logical entry of a terminal merged pair could become
visible in an interrupted manifest before the second entry was committed.
R5 publishes all records of that physical group in one manifest replacement.
An incomplete terminal group is never manifest-visible.

After all raw files have been written and verified, catchable Python exceptions
may recover the entire terminal group as `interrupted`. They may not recover
only one logical half.

### D2: immutable transaction raw names and orphan layout

Legacy raw name:

```text
chunks/chunk_000N.safetensors
```

R5 raw name:

```text
chunks/rev-<revision>-txn-<transaction-id>-chunk-000N.safetensors
```

The reader continues to accept the legacy filename stored in existing v3
manifests. A retry uses a new transaction filename and never overwrites a raw
file referenced by an older manifest or Take. A process stop may leave an
unreferenced raw or temporary file. R5 does not adopt, promote, or delete that
orphan automatically.

## Manifest switch definition

```text
write a complete candidate manifest to a unique temporary file
→ flush and fsync the temporary file
→ atomically replace the active manifest
→ run the available directory durability helper
```

The completed atomic replacement is the **manifest visibility switch**. The
subsequent directory helper completion is the **manifest durability
completion**. Process-stop tests observe the state before replacement,
immediately after replacement, and after the directory helper separately.

On Windows, atomic replacement and media-level durability are not treated as
equivalent. The current Windows directory helper is a no-op; validation records
that limitation and does not claim power-loss or OS-crash durability.

## Behavior that remains unchanged

- non-terminal accepted-prefix meaning and resume position
- manifest status and Review/Take behavior
- canonical eligibility, fallback selection, and pointer meaning
- Sampling, Projection, and finalize failure semantics
- accepted chunk tensors, seeds, prompts, Session, State, and Assembly data
- Run Storage v3 schema readability
- Storage OFF filesystem-I/O behavior

`commit_group()` writes physical-group raw files and publishes their records.
It never updates the canonical project pointer. Canonical promotion remains in
the existing successful-finalize path.

## Process-stop acceptance

- Before manifest replacement, a fresh reader sees the previous complete
  manifest. Any new raw files are unreferenced orphans.
- After manifest replacement, every raw referenced by the new manifest exists,
  has the recorded size, and matches the recorded SHA-256.
- At the replacement boundary, the active file is a complete old manifest or a
  complete new manifest. A partial JSON file and a half terminal group are
  invalid.
- A forced process stop does not run `__exit__`; it is not required to change
  status to `interrupted`.
- Existing canonical fallback selection is preserved even in windows where the
  project pointer has not yet been replaced.

The test fixture covers raw-write, raw-verification, manifest-temporary,
manifest-replace, directory-helper, finalize-manifest, and project-pointer
boundaries with a separately terminated worker and a fresh reader.
