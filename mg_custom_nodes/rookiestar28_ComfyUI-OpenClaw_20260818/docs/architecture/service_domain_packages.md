# Service Domain Packages

Bootstrap lifecycle, route registration, and effective security posture have explicit
implementation owners:

- `services/bootstrap/lifecycle.py` owns startup phase, outcome, and optional-warmup state.
- `services/bootstrap/registration.py` owns host route registration and retry coordination.
- `services/posture/effective.py` owns the immutable process security-posture snapshot.

The historical modules remain compatibility aliases:

- `services/startup_lifecycle.py`
- `services/route_bootstrap.py`
- `services/effective_security_posture.py`

Each alias maps its module name to the implementation module object. This preserves one
process singleton and keeps existing imports and patch points compatible. Do not replace
these aliases with copied re-exports: copied module globals can diverge from the state used
by implementation functions. Type-checker-only exports may describe the legacy interface,
but they must stay behind `TYPE_CHECKING` and must not become a second runtime owner.

New implementation code should import the domain-owned modules. Existing consumers may
continue to use the compatibility paths. An implementation module must never import its
compatibility alias; the repository dependency policy enforces that direction.

Package initializers are navigation-only. They must not register routes, resolve posture,
start threads, or re-export mutable process state during import.
