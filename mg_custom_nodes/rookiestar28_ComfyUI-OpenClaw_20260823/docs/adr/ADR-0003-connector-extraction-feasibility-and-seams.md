# ADR-0003: Connector Extraction Feasibility And Split-Package Seams

- Status: Accepted
- Date: 2026-04-24
- Owners: OpenClaw maintainers
- Related roadmap items: `R162` with prior boundary decision in `ADR-0002`

## Context

ADR-0002 established that the connector is an **optional attached subsystem**, not the primary published artifact of this repository. The remaining question is whether that attached subsystem should now be extracted into a separately packaged connector or separate repo.

Current code structure still mixes:

- connector platform adapters and runtime
- shared installation/token lifecycle services
- shared callback signing and replay-protection contracts
- backend delivery/result APIs that the connector calls locally
- tenant/config/auth boundaries that remain owned by the core package

That means extraction is no longer a purely packaging question. It is a shared-contract question.

## Decision

OpenClaw adopts a **no-go-for-split-now** decision for connector extraction.

Current recommendation:

1. Keep the connector **in-repo** as an **optional attached subsystem**.
2. Treat a future **optional extra package** as the only plausible next extraction target.
3. Treat both **sidecar-only distribution** and **separate repo / primary connector package** as **no-go now** options.

## Minimum Stable Seams Required Before Any Split

Any future extraction must first stabilize these seam families:

1. **installation registry and token refs**
   - workspace/account binding records
   - tenant-scoped token-reference ownership
   - installation diagnostics and fail-closed resolution

2. **interactive callback security contract**
   - signed callback envelopes
   - timestamp / replay / idempotency checks
   - action-policy mapping and approval downgrade semantics

3. **delivery and result bridge**
   - submission/result polling contract
   - callback delivery expectations
   - backend result payload compatibility

4. **config/auth and tenant boundary**
   - connector runtime config contract
   - admin token / auth expectations
   - tenant header behavior
   - server-side secret/state ownership

## Why Separate Packaging Is A No-Go Now

Current blockers are concrete, not theoretical:

- shared services import connector types and connector adapters import shared services, so extraction would currently create unstable bidirectional package seams
- installation/token/state ownership still lives in shared repo services rather than a versioned connector-boundary package
- connector API/client flows still assume in-repo backend evolution instead of a versioned external backend contract
- `services/sidecar` still imports connector config/client modules directly, so even a packaging-only split would not isolate ownership yet

## Consequences

Positive:

- maintainers now have one explicit go/no-go answer instead of repeatedly re-litigating extraction
- future connector extraction work can target named seam families instead of rediscovering coupling ad hoc
- admin diagnostics can expose the same contract to future packaging or release automation

Trade-offs:

- the repo intentionally keeps connector and core package code together for now
- packaging hygiene remains a future concern rather than a solved distribution problem
- extraction pressure is deferred until shared contracts are versionable on their own

## Rejected Alternatives

1. Extract connector into a separate repo now
   - Rejected because current coupling would move instability across package boundaries instead of reducing it.

2. Publish connector as a sidecar-only primary distribution now
   - Rejected because current operator workflows still assume the embedded OpenClaw package/runtime remains primary.

3. Leave extraction as an undocumented future possibility
   - Rejected because future packaging work needs an explicit seam map and a clear no-go baseline.
