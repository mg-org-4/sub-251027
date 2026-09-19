"""Compatibility alias for the effective security posture implementation module."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from .posture import effective as _implementation

if TYPE_CHECKING:
    # Static-only exports keep legacy imports typed without duplicating installed state.
    from .posture.effective import EffectiveSecurityPosture as EffectiveSecurityPosture
    from .posture.effective import PostureFinding as PostureFinding
    from .posture.effective import (
        effective_security_posture_diagnostics as effective_security_posture_diagnostics,
    )
    from .posture.effective import (
        get_effective_security_posture as get_effective_security_posture,
    )
    from .posture.effective import (
        get_or_create_effective_security_posture as get_or_create_effective_security_posture,
    )
    from .posture.effective import (
        install_effective_security_posture as install_effective_security_posture,
    )
    from .posture.effective import (
        reset_effective_security_posture_for_tests as reset_effective_security_posture_for_tests,
    )
    from .posture.effective import (
        resolve_effective_security_posture as resolve_effective_security_posture,
    )

# IMPORTANT: alias the module object; copied re-exports split installed posture state.
sys.modules[__name__] = _implementation
