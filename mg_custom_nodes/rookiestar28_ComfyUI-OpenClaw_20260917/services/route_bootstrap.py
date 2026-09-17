"""Compatibility alias for the bootstrap registration implementation module."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from .bootstrap import registration as _implementation

if TYPE_CHECKING:
    # Static-only exports keep legacy imports typed without duplicating module globals.
    from .bootstrap.registration import register_routes_once as register_routes_once
    from .bootstrap.registration import (
        reset_route_bootstrap_for_tests as reset_route_bootstrap_for_tests,
    )

# IMPORTANT: alias the module object; copied re-exports break accepted patch seams.
sys.modules[__name__] = _implementation
