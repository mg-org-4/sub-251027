# OpenClaw Gateway (Sidecar) - Deprecated

> [!WARNING]
> This page is kept as a compatibility stub to preserve old links.
> The previous architecture draft here is obsolete and intentionally removed.

## Current Source of Truth

- `README.md` (Bridge and Connector sections)
- `docs/connector.md`
- `docs/security_deployment_guide.md`
- `docs/security_key_lifecycle_sop.md`
- `.planning/roadmap.md` (latest implementation status and remaining work)

## Status Note

- Bridge APIs and connector runtime are available.
- Connector/sidecar runtime remains an optional attached subsystem; the primary package artifact is the ComfyUI custom node pack.
- Connector extraction remains a no-go-for-split-now decision until the shared installation/callback/delivery/config seams are independently versioned; see `docs/adr/ADR-0003-connector-extraction-feasibility-and-seams.md`.
- Standalone sidecar/gateway evolution is tracked in `.planning/roadmap.md`.
