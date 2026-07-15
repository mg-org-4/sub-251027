"""
Machine-readable product/package boundary contract for OpenClaw.

Keep this module dependency-light so contract tests and future packaging work can
read one stable source of truth without importing heavy runtime surfaces.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

PRODUCT_BOUNDARY_CONTRACT_VERSION = 1

_PRODUCT_BOUNDARY_CONTRACT: Dict[str, Any] = {
    "version": PRODUCT_BOUNDARY_CONTRACT_VERSION,
    "package_name": "comfyui-openclaw",
    "primary_distribution": {
        "id": "comfyui_custom_node_pack",
        "label": "ComfyUI custom node pack",
        "summary": (
            "The published package artifact is a ComfyUI custom node pack loaded "
            "from custom_nodes/ and anchored by __init__.py."
        ),
    },
    "supported_identities": [
        {
            "id": "comfyui_node_pack",
            "label": "ComfyUI custom node pack",
            "summary": "Primary distribution and entrypoint ownership surface.",
        },
        {
            "id": "embedded_operator_platform",
            "label": "embedded operator platform",
            "summary": (
                "In-process API, security/runtime governance, embedded sidebar, "
                "and remote admin surfaces that run alongside ComfyUI."
            ),
        },
        {
            "id": "connector_capable_control_surface",
            "label": "connector-capable control surface",
            "summary": (
                "The repo supports remote chat control through an optional "
                "connector sidecar, but the connector is not the primary package "
                "artifact."
            ),
        },
    ],
    "core_subsystems": [
        {
            "id": "node_pack_entrypoint",
            "label": "custom node pack entrypoint",
            "entrypoints": ["__init__.py", "nodes", "web"],
        },
        {
            "id": "embedded_api_and_runtime",
            "label": "embedded API and runtime governance",
            "entrypoints": [
                "api/routes.py",
                "services/route_bootstrap.py",
                "services/control_plane.py",
            ],
        },
        {
            "id": "embedded_operator_ui",
            "label": "embedded operator UI surfaces",
            "entrypoints": [
                "web/openclaw.js",
                "web/openclaw_ui.js",
                "api/remote_admin.py",
            ],
        },
    ],
    "attached_subsystems": [
        {
            "id": "connector_sidecar",
            "label": "connector sidecar",
            "classification": "optional_attached_subsystem",
            "entrypoints": ["connector/__main__.py", "docs/connector.md"],
            "summary": (
                "Standalone chat-platform bridge that remains in-repo but is not "
                "the primary published package boundary."
            ),
        }
    ],
    "supported_topologies": [
        {
            "id": "embedded_local",
            "label": "embedded local/lan",
            "summary": (
                "OpenClaw runs in the ComfyUI process as the primary node-pack "
                "artifact with in-process operator surfaces."
            ),
        },
        {
            "id": "embedded_split_control_plane",
            "label": "embedded package with split high-risk control plane",
            "summary": (
                "The same node-pack artifact remains primary, while high-risk "
                "control surfaces are externalized according to the control-plane "
                "contract."
            ),
        },
        {
            "id": "embedded_with_connector_sidecar",
            "label": "embedded package plus optional connector sidecar",
            "summary": (
                "The connector runs as a companion process that calls the local "
                "OpenClaw APIs; it augments the package but does not replace it."
            ),
        },
    ],
    "unsupported_topologies": [
        {
            "id": "connector_only_distribution",
            "label": "connector-only distribution",
            "summary": (
                "This repo does not currently define the connector as a standalone "
                "published package artifact."
            ),
        },
        {
            "id": "standalone_non_comfyui_backend",
            "label": "standalone non-ComfyUI backend",
            "summary": (
                "This repo does not currently publish OpenClaw as a generic "
                "server package that runs without the ComfyUI host."
            ),
        },
    ],
}


def get_product_boundary_contract() -> Dict[str, Any]:
    return copy.deepcopy(_PRODUCT_BOUNDARY_CONTRACT)
