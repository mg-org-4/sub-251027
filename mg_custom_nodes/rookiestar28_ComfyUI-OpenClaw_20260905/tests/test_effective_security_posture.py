"""R232 immutable effective security-posture contract tests."""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from services.effective_security_posture import (
    EffectiveSecurityPosture,
    effective_security_posture_diagnostics,
    get_effective_security_posture,
    get_or_create_effective_security_posture,
    install_effective_security_posture,
    reset_effective_security_posture_for_tests,
    resolve_effective_security_posture,
)


def _valid_public_env() -> dict[str, str]:
    return {
        "OPENCLAW_DEPLOYMENT_PROFILE": "public",
        "OPENCLAW_RUNTIME_PROFILE": "hardened",
        "OPENCLAW_ADMIN_TOKEN": "PRIVATE_ADMIN_CANARY",
        "OPENCLAW_OBSERVABILITY_TOKEN": "PRIVATE_OBS_CANARY",
        "OPENCLAW_ALLOW_REMOTE_ADMIN": "0",
        "OPENCLAW_PUBLIC_SHARED_SURFACE_BOUNDARY_ACK": "1",
        "OPENCLAW_TRUST_X_FORWARDED_FOR": "1",
        "OPENCLAW_TRUSTED_PROXIES": "10.0.0.0/8",
        "OPENCLAW_WEBHOOK_AUTH_MODE": "hmac",
        "OPENCLAW_WEBHOOK_HMAC_SECRET": "PRIVATE_HMAC_CANARY",
        "OPENCLAW_WEBHOOK_REQUIRE_REPLAY_PROTECTION": "1",
        "OPENCLAW_ENABLE_EXTERNAL_TOOLS": "0",
        "OPENCLAW_ENABLE_REGISTRY_SYNC": "0",
        "OPENCLAW_ENABLE_TRANSFORMS": "0",
        "OPENCLAW_ALLOW_ANY_PUBLIC_LLM_HOST": "0",
        "OPENCLAW_ALLOW_INSECURE_BASE_URL": "0",
        "OPENCLAW_SECURITY_DANGEROUS_BIND_OVERRIDE": "0",
        "OPENCLAW_CONTROL_PLANE_MODE": "split",
        "OPENCLAW_CONTROL_PLANE_URL": "https://private-control.invalid",
        "OPENCLAW_CONTROL_PLANE_TOKEN": "PRIVATE_CP_CANARY",
        "OPENCLAW_CONNECTOR_TELEGRAM_TOKEN": "PRIVATE_CONNECTOR_CANARY",
        "OPENCLAW_CONNECTOR_TELEGRAM_ALLOWED_USERS": "private-user",
    }


class EffectiveSecurityPostureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        reset_effective_security_posture_for_tests()

    def tearDown(self) -> None:
        reset_effective_security_posture_for_tests()

    def test_schema_is_frozen_recursively_immutable_and_secret_free(self):
        env = _valid_public_env()
        snapshot = resolve_effective_security_posture(env, network_exposed=True)

        self.assertIsInstance(snapshot, EffectiveSecurityPosture)
        self.assertEqual(snapshot.schema_version, 1)
        self.assertEqual(snapshot.deployment_profile, "public")
        self.assertEqual(snapshot.runtime_profile, "hardened")
        self.assertEqual(snapshot.mae_profile, "public")
        self.assertTrue(snapshot.admin_token_configured)
        self.assertTrue(snapshot.control_plane_prerequisites_satisfied)
        self.assertEqual(snapshot.connector_active_platforms, ("telegram",))
        self.assertEqual(snapshot.connector_unguarded_platforms, ())

        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot.deployment_profile = "local"  # type: ignore[misc]

        def assert_immutable(value):
            self.assertNotIsInstance(value, (dict, list, set))
            if dataclasses.is_dataclass(value):
                for field in dataclasses.fields(value):
                    assert_immutable(getattr(value, field.name))
            elif isinstance(value, tuple):
                for item in value:
                    assert_immutable(item)

        assert_immutable(snapshot)
        rendered = json.dumps(
            effective_security_posture_diagnostics(snapshot), sort_keys=True
        )
        for private_value in env.values():
            if private_value.startswith("PRIVATE_") or private_value.startswith(
                "https://"
            ):
                self.assertNotIn(private_value, rendered)
        self.assertNotIn("10.0.0.0/8", rendered)
        self.assertNotIn("private-user", rendered)

    def test_diagnostic_projection_is_allowlisted_bounded_and_code_only(self):
        env = {
            "OPENCLAW_DEPLOYMENT_PROFILE": "public",
            "OPENCLAW_WEBHOOK_AUTH_MODE": "PRIVATE_INVALID_MODE",
            "OPENCLAW_CONNECTOR_SLACK_BOT_TOKEN": "PRIVATE_SLACK_CANARY",
        }
        projection = effective_security_posture_diagnostics(
            resolve_effective_security_posture(env, network_exposed=False)
        )

        self.assertEqual(
            set(projection),
            {
                "schema_version",
                "runtime_profile",
                "deployment_profile",
                "mae_profile",
                "network_exposed",
                "authentication",
                "startup_gate",
                "control_plane",
                "connectors",
                "decision_codes",
                "reason_codes",
            },
        )
        rendered = json.dumps(projection, sort_keys=True)
        self.assertNotIn("PRIVATE_INVALID_MODE", rendered)
        self.assertNotIn("PRIVATE_SLACK_CANARY", rendered)
        self.assertLessEqual(len(rendered), 4096)

    def test_profile_matrix_matches_existing_deployment_evaluator(self):
        from services.deployment_profile import evaluate_deployment_profile
        from tests.test_r105_profile_matrix import PROFILE_MATRIX_FIXTURES

        for fixture in PROFILE_MATRIX_FIXTURES:
            with self.subTest(fixture=fixture["id"]):
                env = dict(fixture["env"])
                env["OPENCLAW_DEPLOYMENT_PROFILE"] = fixture["profile"]
                report = evaluate_deployment_profile(fixture["profile"], env)
                snapshot = resolve_effective_security_posture(
                    env, network_exposed=False
                )
                expected_fail_codes = tuple(
                    check.code for check in report.checks if check.severity == "fail"
                )
                self.assertEqual(snapshot.deployment_fail_codes, expected_fail_codes)
                self.assertEqual(
                    snapshot.startup_profile_passed,
                    fixture["profile"] == "local" or not expected_fail_codes,
                )

    def test_startup_gate_snapshot_matches_explicit_environment_path(self):
        from services.startup_profile_gate import evaluate_startup_gate

        rows = [
            {"OPENCLAW_DEPLOYMENT_PROFILE": "local"},
            {"OPENCLAW_DEPLOYMENT_PROFILE": "lan"},
            _valid_public_env(),
            {
                "OPENCLAW_DEPLOYMENT_PROFILE": "public",
                "OPENCLAW_SECURITY_DANGEROUS_PROFILE_OVERRIDE": "1",
            },
        ]
        for env in rows:
            with self.subTest(env=tuple(sorted(env))):
                expected = evaluate_startup_gate(env)
                snapshot = resolve_effective_security_posture(
                    env, network_exposed=False
                )
                actual = evaluate_startup_gate(posture=snapshot)
                self.assertEqual(actual.profile, expected.profile)
                self.assertEqual(actual.passed, expected.passed)
                self.assertEqual(actual.overridden, expected.overridden)
                self.assertEqual(
                    [item["code"] for item in actual.violations],
                    [item["code"] for item in expected.violations],
                )

    def test_control_plane_snapshot_matches_existing_matrix(self):
        from services.control_plane import (
            HIGH_RISK_SURFACES,
            ControlPlaneMode,
            enforce_control_plane_startup,
            get_blocked_surfaces,
            resolve_control_plane_mode,
            validate_split_prerequisites,
        )

        rows = [
            {"OPENCLAW_DEPLOYMENT_PROFILE": "local"},
            {"OPENCLAW_DEPLOYMENT_PROFILE": "public"},
            _valid_public_env(),
            {
                "OPENCLAW_DEPLOYMENT_PROFILE": "public",
                "OPENCLAW_CONTROL_PLANE_MODE": "embedded",
            },
            {
                "OPENCLAW_DEPLOYMENT_PROFILE": "public",
                "OPENCLAW_CONTROL_PLANE_MODE": "embedded",
                "OPENCLAW_SPLIT_COMPAT_OVERRIDE": "1",
            },
        ]
        for env in rows:
            with self.subTest(env=tuple(sorted(env))):
                telemetry_stub = SimpleNamespace(
                    get_security_telemetry=lambda: SimpleNamespace(
                        record_dangerous_override=lambda *_args, **_kwargs: None
                    )
                )
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.dict(
                        sys.modules,
                        {"services.security_telemetry": telemetry_stub},
                    ),
                ):
                    expected_mode = resolve_control_plane_mode(
                        env["OPENCLAW_DEPLOYMENT_PROFILE"]
                    )
                    expected_prereq = validate_split_prerequisites().to_dict()
                    expected_startup = enforce_control_plane_startup()
                    expected_blocked = get_blocked_surfaces(
                        env["OPENCLAW_DEPLOYMENT_PROFILE"], expected_mode
                    )
                    snapshot = resolve_effective_security_posture(
                        env, network_exposed=False
                    )
                    if (
                        snapshot.deployment_profile == "public"
                        and snapshot.control_plane_mode == "split"
                    ):
                        self.assertEqual(
                            set(snapshot.blocked_surface_ids),
                            {surface_id for surface_id, _ in HIGH_RISK_SURFACES},
                        )
                    self.assertEqual(
                        resolve_control_plane_mode(posture=snapshot),
                        expected_mode,
                    )
                    self.assertEqual(
                        validate_split_prerequisites(posture=snapshot).to_dict(),
                        expected_prereq,
                    )
                    self.assertEqual(
                        enforce_control_plane_startup(posture=snapshot),
                        expected_startup,
                    )
                    self.assertEqual(
                        get_blocked_surfaces(
                            snapshot.deployment_profile,
                            ControlPlaneMode(snapshot.control_plane_mode),
                            posture=snapshot,
                        ),
                        expected_blocked,
                    )

    def test_install_is_identity_stable_thread_safe_and_rejects_replacement(self):
        first = resolve_effective_security_posture(
            {"OPENCLAW_DEPLOYMENT_PROFILE": "local"}, network_exposed=False
        )
        second = resolve_effective_security_posture(
            _valid_public_env(), network_exposed=True
        )

        self.assertIs(install_effective_security_posture(first), first)
        self.assertIs(get_effective_security_posture(), first)
        self.assertIs(install_effective_security_posture(first), first)
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            install_effective_security_posture(second)

        reset_effective_security_posture_for_tests()
        barrier = threading.Barrier(16)

        def resolve_once(_index):
            barrier.wait()
            return get_or_create_effective_security_posture(
                {"OPENCLAW_DEPLOYMENT_PROFILE": "local"},
                network_exposed=False,
            )

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(resolve_once, range(16)))

        self.assertEqual(len({id(item) for item in results}), 1)

    def test_installed_snapshot_prevents_ambient_drift_in_migrated_consumers(self):
        from api import routes
        from services.control_plane import (
            ControlPlaneMode,
            is_surface_blocked,
            resolve_control_plane_mode,
        )
        from services.startup_profile_gate import evaluate_startup_gate

        snapshot = get_or_create_effective_security_posture(
            _valid_public_env(), network_exposed=True
        )
        with patch.dict(
            os.environ,
            {
                "OPENCLAW_DEPLOYMENT_PROFILE": "local",
                "OPENCLAW_RUNTIME_PROFILE": "minimal",
            },
            clear=True,
        ):
            self.assertIs(get_or_create_effective_security_posture(), snapshot)
            self.assertEqual(
                resolve_control_plane_mode("local"), ControlPlaneMode.SPLIT
            )
            self.assertTrue(is_surface_blocked("webhook_execute"))
            self.assertEqual(routes._resolve_mae_profile(), "public")
            gate = evaluate_startup_gate(posture=snapshot)
            self.assertTrue(gate.passed)
            self.assertEqual(gate.profile, "public")

    def test_security_gate_uses_explicit_snapshot_for_process_static_branches(self):
        from services.security_gate import SecurityGate

        snapshot = resolve_effective_security_posture(
            {"OPENCLAW_DEPLOYMENT_PROFILE": "local"},
            network_exposed=True,
        )
        drifted_env = {
            "OPENCLAW_DEPLOYMENT_PROFILE": "public",
            "OPENCLAW_RUNTIME_PROFILE": "hardened",
            "OPENCLAW_ADMIN_TOKEN": "DRIFTED_PRIVATE_TOKEN",
            "OPENCLAW_SECURITY_DANGEROUS_BIND_OVERRIDE": "1",
        }
        with (
            patch.dict(os.environ, drifted_env, clear=True),
            patch.object(
                SecurityGate,
                "_check_network_exposure",
                side_effect=AssertionError("ambient network exposure read"),
            ),
            patch(
                "services.security_gate.get_runtime_profile",
                side_effect=AssertionError("ambient runtime profile read"),
            ),
            patch(
                "services.security_gate.is_hardened_mode",
                side_effect=AssertionError("ambient hardened profile read"),
            ),
            patch(
                "services.security_gate.evaluate_connector_allowlist_posture",
                side_effect=AssertionError("ambient connector posture read"),
            ),
            patch("services.modules.is_module_enabled", return_value=False),
            patch("services.tool_runner.is_tools_enabled", return_value=False),
            patch(
                "services.permission_posture.evaluate_startup_permissions",
                return_value=(True, []),
            ),
        ):
            passed, _warnings, fatal_errors = SecurityGate.verify_mandatory_controls(
                posture=snapshot
            )

        self.assertFalse(passed)
        self.assertTrue(
            any(
                "exposed (--listen) without Authentication" in item
                for item in fatal_errors
            )
        )
        self.assertFalse(any("DRIFTED_PRIVATE_TOKEN" in item for item in fatal_errors))

    def test_malformed_or_unknown_inputs_fail_with_content_free_errors(self):
        class ExplodingMapping(dict):
            def __contains__(self, _key):
                raise RuntimeError("PRIVATE_MAPPING_CANARY")

        with self.assertRaisesRegex(
            ValueError, "^security posture input unavailable$"
        ) as mapping_error:
            resolve_effective_security_posture(
                ExplodingMapping(), network_exposed=False
            )
        self.assertNotIn("PRIVATE_MAPPING_CANARY", str(mapping_error.exception))

        with self.assertRaisesRegex(
            ValueError, "^unsupported deployment profile$"
        ) as profile_error:
            resolve_effective_security_posture(
                {"OPENCLAW_DEPLOYMENT_PROFILE": ("PRIVATE_PROFILE_CANARY")},
                network_exposed=False,
            )
        self.assertNotIn("PRIVATE_PROFILE_CANARY", str(profile_error.exception))

        class LateExplodingMapping(dict):
            def get(self, key, default=None):
                if key == "OPENCLAW_CONNECTOR_SLACK_BOT_TOKEN":
                    raise RuntimeError("PRIVATE_DELEGATED_MAPPING_CANARY")
                return super().get(key, default)

        with self.assertRaisesRegex(
            ValueError, "^security posture evaluation failed$"
        ) as delegated_error:
            resolve_effective_security_posture(
                LateExplodingMapping(), network_exposed=False
            )
        self.assertNotIn(
            "PRIVATE_DELEGATED_MAPPING_CANARY", str(delegated_error.exception)
        )

    def test_request_dynamic_security_state_is_not_in_snapshot_schema(self):
        forbidden_names = {
            "presented",
            "header",
            "cookie",
            "registry",
            "tenant",
            "scope",
            "client",
            "origin",
            "replay_state",
            "session",
            "credential_value",
            "token_value",
            "presented_token",
            "request_headers",
            "tenant_id",
            "client_address",
            "request_origin",
            "live_connector_session",
        }
        field_names = {
            field.name for field in dataclasses.fields(EffectiveSecurityPosture)
        }
        for field_name in forbidden_names:
            self.assertNotIn(
                field_name,
                field_names,
                f"request-dynamic field leaked into snapshot: {field_name}",
            )

    def test_consumer_contracts_accept_explicit_posture(self):
        from services.control_plane import (
            enforce_control_plane_startup,
            get_blocked_surfaces,
            resolve_control_plane_mode,
            validate_split_prerequisites,
        )
        from services.security_gate import SecurityGate, enforce_startup_gate
        from services.startup_profile_gate import evaluate_startup_gate

        for function in (
            evaluate_startup_gate,
            SecurityGate.verify_mandatory_controls,
            enforce_startup_gate,
            resolve_control_plane_mode,
            get_blocked_surfaces,
            validate_split_prerequisites,
            enforce_control_plane_startup,
        ):
            with self.subTest(function=function.__qualname__):
                self.assertIn("posture", inspect.signature(function).parameters)

    def test_route_bootstrap_reset_clears_process_snapshot(self):
        from services import route_bootstrap

        get_or_create_effective_security_posture(
            {"OPENCLAW_DEPLOYMENT_PROFILE": "local"}, network_exposed=False
        )
        self.assertIsNotNone(get_effective_security_posture(required=False))

        route_bootstrap.reset_route_bootstrap_for_tests()

        self.assertIsNone(get_effective_security_posture(required=False))


if __name__ == "__main__":
    unittest.main()
