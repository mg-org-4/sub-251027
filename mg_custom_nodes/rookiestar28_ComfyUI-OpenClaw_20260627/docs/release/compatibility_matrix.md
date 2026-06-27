# Compatibility Matrix

```openclaw-compat-matrix-meta
{
  "anchors": {
    "comfyui": "f6c162dd (v0.26.0 / pyproject 0.26.0)",
    "comfyui_frontend": "1.47.3 (f076106ca / v1.47.3-5-gf076106ca)",
    "desktop": "0.9.4 (core 0.22.3 / frontend 1.43.18)"
  },
  "evidence": {
    "evidence_id": "compat-matrix-refresh-20260624",
    "updated_at": "2026-06-24T05:10:00+00:00",
    "updated_by": "host-reference-refresh"
  },
  "last_validated_date": "2026-06-24",
  "matrix_version": "v0.2.6",
  "policy": {
    "max_age_days": 45,
    "warn_age_days": 30
  },
  "schema_version": 1
}
```

This document tracks the current reference anchors and validated environments for the active ComfyUI-OpenClaw branch.

## Core Dependencies

| Component | Validated Range | Best Effort / Experimental | Notes |
| :--- | :--- | :--- | :--- |
| **ComfyUI** | `f6c162dd` reference anchor (`v0.26.0`; `pyproject.toml` version `0.26.0`) | Older tagged snapshots | Current local upstream reference repo snapshot used for compatibility review |
| **ComfyUI Frontend** | `1.47.3` reference anchor (`f076106ca`; `v1.47.3-5-gf076106ca`) | Minor drift around the anchor | Sidebar extension contract remains compatible; prefer the current sidebar store API with deprecated facade fallback |
| **ComfyUI Desktop** | `0.9.4 (core 0.22.3 / frontend 1.43.18)` reference anchor | Desktop bundle may lag standalone frontend | Treat desktop parity as a distinct host surface, not an alias of standalone frontend HEAD |
| **Python** | 3.10, 3.11, 3.12 | 3.9 | 3.13 not yet validated |
| **Torch** | 2.1.2+ | 1.13+ | CUDA 11.8/12.1 verified |

## Host-Surface Notes

- **ComfyUI host runtime**: current bootstrap assumptions remain aligned with upstream `PromptServer` startup and route registration flow, including `/api`-prefixed canonical API routing.
- **Frontend host surface**: current sidebar integration contract remains compatible with the standalone frontend reference anchor, while inactive subgraph diagnostics and promoted-widget behavior remain regression-sensitive seams.
- **Desktop host surface**: desktop currently embeds frontend `1.43.18`, which still lags the standalone frontend `1.47.3` reference. Validate desktop-specific behavior against the desktop anchor instead of assuming standalone-frontend parity.

## Residual Host-Contract Decisions

- **SaveImage output refs**: OpenClaw consumes runtime `/history` output refs and does not infer graph-rewrite behavior from output-node socket shape. `SaveImage` output sockets are allowed to exist without changing the normalized output-ref contract.
- **3D output refs**: `Load3DAdvanced` and related 3D preview refs remain media-aware output refs. File-like and hash-backed 3D refs stay on the bounded `/view` preview contract; clients without a 3D renderer should show an explicit fallback/link surface.
- **Asset dimensions and grouped assets**: typed width/height metadata and grouped multi-download behavior are host-frontend display/download concerns. They do not change OpenClaw fetch routing, and asset-service-only identifiers remain explicit `asset_api_required` states rather than implicit `/api/assets` fetches.
- **Sidebar registration**: prefer the current `sidebarTab.registerSidebarTab` host API and retain the deprecated `extensionManager.registerSidebarTab` fallback for older or desktop-embedded frontend hosts.
- **Node runtime policy**: the standalone ComfyUI frontend development workspace currently declares `node >=25 <26`, but OpenClaw keeps its package engine at `>=18.0.0` because this custom-node package runs its own Playwright/Vitest harness and does not build the host frontend workspace. OpenClaw acceptance remains governed by `tests/TEST_SOP.md` and `tests/E2E_TESTING_SOP.md`, which require Node.js 18+ and CI-parity validation on the project test harness.

## Operating Systems

| OS | Status | CI Validation | Notes |
| :--- | :--- | :--- | :--- |
| **Windows 10/11** | ✅ Supported | Manual | Primary dev environment |
| **Linux (Ubuntu 22.04)** | ✅ Supported | Automated | CI environment |
| **macOS (Apple Silicon)** | ⚠️ Best Effort | None | Should work, not guaranteed |
| **WSL2** | ✅ Supported | None | Treated as Linux |

## Browser Support

| Browser | Minimum Version | Notes |
| :--- | :--- | :--- |
| **Chrome / Edge** | Latest - 2 | Primary target |
| **Firefox** | Latest - 2 | |
| **Safari** | Latest - 2 | |

## Hardware Recommendations

- **VRAM**: Minimum 8GB (for SDXL), 16GB recommended (for Flux).
- **RAM**: Minimum 16GB.
- **Disk**: SSD recommended for fast model loading.
