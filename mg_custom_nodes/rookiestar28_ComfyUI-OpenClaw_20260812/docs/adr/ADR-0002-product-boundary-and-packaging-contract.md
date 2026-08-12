# ADR-0002: Product Boundary And Packaging Contract

- Status: Accepted
- Date: 2026-04-23
- Owners: OpenClaw maintainers
- Related roadmap items: `R160` with follow-up execution in `R161` and `R162`

## Context

OpenClaw started as a ComfyUI-focused node pack, but the repository now also contains:

- embedded HTTP APIs and operator UI surfaces,
- a standalone Remote Admin Console route,
- connector runtime code for multiple chat platforms,
- split control-plane governance for higher-risk deployments.

That evolution made one question increasingly ambiguous: what is the supported identity of this repo/package today, and which parts are first-class versus optional attached subsystems?

Without a boundary decision, future work such as config decomposition, connector extraction, or packaging hygiene is forced to rely on repo intuition instead of an explicit contract.

## Decision

OpenClaw is defined as a **ComfyUI custom node pack** first, with two explicit first-class identities layered on top of that package:

1. **ComfyUI custom node pack**
   - This is the primary distribution artifact and runtime anchor.
   - `__init__.py` remains the package entrypoint loaded from `custom_nodes/`.

2. **embedded operator platform**
   - In-process OpenClaw APIs, runtime/security governance, sidebar UX, and remote admin surfaces are treated as part of the shipped package, not as separate products.

3. **connector-capable control surface**
   - Remote chat control is supported through the in-repo connector sidecar, but the connector remains an **optional attached subsystem**, not the primary package artifact.

## Core vs Attached Subsystems

Core to the package:

- custom node pack entrypoint and exported nodes
- embedded API/runtime governance (`/openclaw/*`, route bootstrap, control-plane policy)
- embedded operator UI surfaces (sidebar plus `/openclaw/admin`)

Optional attached subsystem:

- connector sidecar (`python -m connector`) and platform-specific adapters

This means:

- the connector is supported and intentionally in-repo,
- but the repo is **not** currently defined as a connector-first distribution,
- and the repo is **not** currently defined as a standalone generic backend independent of ComfyUI.

## Supported Topologies

Supported:

1. **embedded local/lan**
   - OpenClaw runs inside the ComfyUI process as the primary package artifact.

2. **embedded package with split high-risk control plane**
   - The same package stays primary, while higher-risk control surfaces are externalized according to the split-mode contract.

3. **embedded package plus optional connector sidecar**
   - The connector runs as a companion process that calls the local OpenClaw APIs.

Unsupported as first-class package identities today:

1. **connector-only distribution**
2. **standalone non-ComfyUI backend package**

Those possibilities are future design questions, not current promises. Connector extraction feasibility remains explicitly deferred to `R162`.

## Consequences

Positive:

- future extraction/pruning decisions now have one explicit contract to evaluate against
- docs and contributor discussions can use the same terms instead of mixing "node pack", "server", and "sidecar" loosely
- `R161` and `R162` can narrow their scope around a known boundary instead of re-litigating product identity

Trade-offs:

- the repo still carries multiple execution surfaces inside one codebase
- connector remains intentionally attached even though it is operationally separable
- some public docs must stay careful not to imply standalone server packaging that does not exist yet

## Rejected Alternatives

1. Treat the connector as an equal primary package artifact today
   - Rejected because there is no separate connector package/distribution contract yet.

2. Define OpenClaw as a generic standalone backend package
   - Rejected because current runtime ownership still assumes a ComfyUI host process.

3. Keep the boundary implicit and rely on contributor convention
   - Rejected because packaging and extraction follow-ups now depend on an explicit contract.
