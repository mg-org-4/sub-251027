import asyncio
import json
import runpy
import sys
import threading
import time
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from services import route_bootstrap
from services.startup_lifecycle import (
    MAX_DIAGNOSTIC_MS,
    MAX_WARMUPS,
    STARTUP_DIAGNOSTIC_KEYS,
    StartupLifecycle,
    StartupPhase,
    StartupReason,
    StartupState,
    StartupTransitionError,
    WarmupState,
    get_startup_diagnostics,
    get_startup_outcome,
    mark_bootstrap_import_failed,
    reset_startup_lifecycle_for_tests,
)


class _Clock:
    def __init__(self, value: float = 10.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class TestStartupOutcomeContract(unittest.TestCase):
    def test_initial_snapshot_is_frozen_versioned_and_deterministic(self):
        clock = _Clock()
        lifecycle = StartupLifecycle(monotonic_fn=clock)

        outcome = lifecycle.snapshot()
        first = outcome.to_diagnostics()
        second = outcome.to_diagnostics()

        self.assertEqual(tuple(first), STARTUP_DIAGNOSTIC_KEYS)
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(first["phase"], "package_import")
        self.assertEqual(first["state"], "starting")
        self.assertEqual(first["reason_code"], "bootstrap_started")
        self.assertFalse(first["ready"])
        self.assertFalse(first["degraded"])
        self.assertFalse(first["fatal"])
        self.assertEqual(first["warmups"], [])
        self.assertIsInstance(outcome.warmups, tuple)
        with self.assertRaises(FrozenInstanceError):
            outcome.ready = True

    def test_legal_transition_path_has_bounded_monotonic_timing(self):
        clock = _Clock()
        lifecycle = StartupLifecycle(monotonic_fn=clock)

        lifecycle.mark_required_initialization_started()
        clock.advance(0.125)
        lifecycle.mark_host_waiting(attempt=0, max_attempts=3)
        clock.advance(0.125)
        lifecycle.mark_host_waiting(attempt=1, max_attempts=3)
        lifecycle.mark_route_registration_started(attempt=2, max_attempts=3)
        clock.advance(200_000)
        lifecycle.mark_ready()

        diagnostics = lifecycle.snapshot().to_diagnostics()
        self.assertEqual(diagnostics["phase"], "complete")
        self.assertEqual(diagnostics["state"], "ready")
        self.assertEqual(diagnostics["reason_code"], "route_registration_succeeded")
        self.assertTrue(diagnostics["ready"])
        self.assertFalse(diagnostics["fatal"])
        self.assertEqual(diagnostics["attempt"], 2)
        self.assertEqual(diagnostics["max_attempts"], 3)
        for field in ("elapsed_ms", "phase_elapsed_ms", "ready_elapsed_ms"):
            self.assertIsInstance(diagnostics[field], int)
            self.assertGreaterEqual(diagnostics[field], 0)
            self.assertLessEqual(diagnostics[field], MAX_DIAGNOSTIC_MS)

    def test_invalid_and_post_fatal_transitions_do_not_mutate_state(self):
        lifecycle = StartupLifecycle()
        before = lifecycle.snapshot()

        with self.assertRaises(StartupTransitionError) as invalid:
            lifecycle.mark_ready()
        self.assertEqual(invalid.exception.code, "INVALID_TRANSITION")
        self.assertEqual(lifecycle.snapshot(), before)

        lifecycle.mark_fatal(
            phase=StartupPhase.PACKAGE_IMPORT,
            reason_code=StartupReason.BOOTSTRAP_IMPORT_FAILED,
        )
        fatal = lifecycle.snapshot()
        with self.assertRaises(StartupTransitionError) as terminal:
            lifecycle.mark_required_initialization_started()
        self.assertEqual(terminal.exception.code, "TERMINAL_STATE")
        self.assertEqual(lifecycle.snapshot(), fatal)

    def test_retry_attempt_must_increase_and_stay_within_bound(self):
        lifecycle = StartupLifecycle()
        lifecycle.mark_required_initialization_started()
        lifecycle.mark_host_waiting(attempt=0, max_attempts=2)

        for attempt, code in (
            (0, "ATTEMPT_NOT_INCREASING"),
            (3, "ATTEMPT_OUT_OF_RANGE"),
        ):
            with self.subTest(attempt=attempt):
                before = lifecycle.snapshot()
                with self.assertRaises(StartupTransitionError) as ctx:
                    lifecycle.mark_host_waiting(attempt=attempt, max_attempts=2)
                self.assertEqual(ctx.exception.code, code)
                self.assertEqual(lifecycle.snapshot(), before)

        with self.assertRaises(StartupTransitionError) as route_regression:
            lifecycle.mark_route_registration_started(
                attempt=0,
                max_attempts=2,
            )
        self.assertEqual(
            route_regression.exception.code,
            "ATTEMPT_NOT_INCREASING",
        )

    def test_fatal_reason_must_match_its_phase(self):
        lifecycle = StartupLifecycle()
        before = lifecycle.snapshot()

        with self.assertRaises(StartupTransitionError) as mismatch:
            lifecycle.mark_fatal(
                phase=StartupPhase.PACKAGE_IMPORT,
                reason_code=StartupReason.ROUTE_REGISTRATION_FAILED,
            )

        self.assertEqual(mismatch.exception.code, "FATAL_PHASE_MISMATCH")
        self.assertEqual(lifecycle.snapshot(), before)

        with self.assertRaises(StartupTransitionError) as source:
            lifecycle.mark_fatal(
                phase=StartupPhase.ROUTE_REGISTRATION,
                reason_code=StartupReason.ROUTE_REGISTRATION_FAILED,
            )
        self.assertEqual(source.exception.code, "INVALID_FATAL_TRANSITION")
        self.assertEqual(lifecycle.snapshot(), before)

    def test_exception_payload_is_never_retained_or_serialized(self):
        marker = "PRIVATE_R231_SECRET C:/private/token.txt"

        reset_startup_lifecycle_for_tests()
        mark_bootstrap_import_failed(RuntimeError(marker))
        outcome = get_startup_outcome()
        rendered = json.dumps(get_startup_diagnostics(), sort_keys=True)

        self.assertEqual(outcome.state, StartupState.FATAL)
        self.assertEqual(outcome.reason_code, StartupReason.BOOTSTRAP_IMPORT_FAILED)
        self.assertNotIn(marker, repr(outcome))
        self.assertNotIn(marker, rendered)
        self.assertNotIn("RuntimeError", rendered)
        self.assertNotIn("started_at", rendered)
        self.assertNotIn("traceback", rendered.lower())


class TestStartupWarmupProjection(unittest.TestCase):
    def setUp(self):
        reset_startup_lifecycle_for_tests()
        from services.startup_lifecycle import mark_startup_ready

        mark_startup_ready("routes")

    def tearDown(self):
        reset_startup_lifecycle_for_tests()

    def _wait_for_warmup(self, name: str, state: str) -> dict:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            diagnostics = get_startup_diagnostics()
            warmup = next(
                (item for item in diagnostics["warmups"] if item["name"] == name),
                None,
            )
            if warmup and warmup["state"] == state:
                return diagnostics
            time.sleep(0.005)
        self.fail(f"warmup {name} did not reach {state}")

    def test_failure_and_timeout_degrade_without_leaking_exception_content(self):
        from services.startup_lifecycle import start_optional_warmups

        release = threading.Event()
        marker = "PRIVATE_WARMUP_FAILURE C:/private/model"

        def fail():
            raise RuntimeError(marker)

        def block():
            release.wait(timeout=1)

        with self.assertLogs("ComfyUI-OpenClaw", level="WARNING") as captured:
            start_optional_warmups(
                [
                    ("z_failure", fail, 0.5),
                    ("a_timeout", block, 0.01),
                ]
            )
            self._wait_for_warmup("z_failure", "failed")
            diagnostics = self._wait_for_warmup("a_timeout", "timed_out")
            release.set()
            time.sleep(0.02)

        self.assertTrue(diagnostics["ready"])
        self.assertTrue(diagnostics["degraded"])
        self.assertFalse(diagnostics["fatal"])
        self.assertEqual(diagnostics["state"], "degraded")
        self.assertEqual(
            [item["name"] for item in diagnostics["warmups"]],
            ["a_timeout", "z_failure"],
        )
        rendered = json.dumps(diagnostics, sort_keys=True)
        self.assertNotIn(marker, rendered)
        self.assertNotIn("RuntimeError", rendered)
        self.assertNotIn("error", rendered)
        self.assertNotIn(marker, "\n".join(captured.output))
        final = get_startup_diagnostics()
        timed_out = next(
            item for item in final["warmups"] if item["name"] == "a_timeout"
        )
        self.assertEqual(timed_out["state"], "timed_out")

    def test_terminal_warmup_cannot_restart_or_be_overwritten(self):
        lifecycle = StartupLifecycle()
        lifecycle.mark_required_initialization_started()
        lifecycle.mark_route_registration_started()
        lifecycle.mark_ready()

        should_start, generation, name = lifecycle.begin_warmup("provider", 0.01)
        self.assertTrue(should_start)
        lifecycle.mark_warmup_running(name, generation)
        lifecycle.finish_warmup(
            name,
            generation,
            state=WarmupState.TIMED_OUT,
        )
        should_restart, _, _ = lifecycle.begin_warmup("provider", 0.01)
        lifecycle.finish_warmup(
            name,
            generation,
            state=WarmupState.SUCCEEDED,
        )

        self.assertFalse(should_restart)
        outcome = lifecycle.snapshot()
        self.assertEqual(outcome.state, StartupState.DEGRADED)
        self.assertEqual(outcome.warmups[0].state, WarmupState.TIMED_OUT)

    def test_warmup_projection_has_a_non_mutating_cardinality_bound(self):
        lifecycle = StartupLifecycle()
        lifecycle.mark_required_initialization_started()
        lifecycle.mark_route_registration_started()
        lifecycle.mark_ready()

        for index in range(MAX_WARMUPS):
            should_start, _, _ = lifecycle.begin_warmup(
                f"provider_{index}",
                0.01,
            )
            self.assertTrue(should_start)
        before = lifecycle.snapshot()

        with self.assertRaises(StartupTransitionError) as limit:
            lifecycle.begin_warmup("one_too_many", 0.01)

        self.assertEqual(limit.exception.code, "WARMUP_LIMIT_EXCEEDED")
        self.assertEqual(lifecycle.snapshot(), before)
        self.assertEqual(len(before.warmups), MAX_WARMUPS)

    def test_monitor_thread_start_failure_degrades_and_reraises_content_free(self):
        from services.startup_lifecycle import start_optional_warmups

        marker = "PRIVATE_MONITOR_START C:/private/monitor"
        failure = RuntimeError(marker)
        with patch("services.startup_lifecycle.threading.Thread") as thread_factory:
            thread_factory.return_value.start.side_effect = failure
            with self.assertRaises(RuntimeError) as ctx:
                start_optional_warmups([("monitor_provider", lambda: None, 0.01)])

        self.assertIs(ctx.exception, failure)
        diagnostics = get_startup_diagnostics()
        self.assertTrue(diagnostics["degraded"])
        warmup = next(
            item
            for item in diagnostics["warmups"]
            if item["name"] == "monitor_provider"
        )
        self.assertEqual(warmup["state"], "failed")
        self.assertNotIn(marker, json.dumps(diagnostics))

    def test_worker_thread_start_failure_degrades_without_escaping_monitor(self):
        from services.startup_lifecycle import _LIFECYCLE, _warmup_monitor

        marker = "PRIVATE_WORKER_START C:/private/worker"
        failure = RuntimeError(marker)
        _, generation, name = _LIFECYCLE.begin_warmup(
            "worker_provider",
            0.01,
        )

        with (
            patch("services.startup_lifecycle.threading.Thread") as thread_factory,
            self.assertLogs("ComfyUI-OpenClaw", level="WARNING") as captured,
        ):
            thread_factory.return_value.start.side_effect = failure
            _warmup_monitor(name, generation, lambda: None, 0.01)

        diagnostics = get_startup_diagnostics()
        self.assertTrue(diagnostics["degraded"])
        warmup = next(
            item for item in diagnostics["warmups"] if item["name"] == "worker_provider"
        )
        self.assertEqual(warmup["state"], "failed")
        self.assertNotIn(marker, json.dumps(diagnostics))
        self.assertNotIn(marker, "\n".join(captured.output))


class TestRouteBootstrapOutcomeIntegration(unittest.TestCase):
    def setUp(self):
        route_bootstrap.reset_route_bootstrap_for_tests()
        reset_startup_lifecycle_for_tests()

    def tearDown(self):
        route_bootstrap.reset_route_bootstrap_for_tests()
        reset_startup_lifecycle_for_tests()

    @staticmethod
    def _host_module(instance):
        return SimpleNamespace(PromptServer=SimpleNamespace(instance=instance))

    def test_duplicate_concurrent_calls_share_one_sync_owner_and_retry_owner(self):
        entered = threading.Event()
        release = threading.Event()
        errors = []

        def required_init():
            entered.set()
            release.wait(timeout=1)

        with (
            patch.object(
                route_bootstrap, "_register_plugins_and_shutdown_hooks"
            ) as optional,
            patch.object(
                route_bootstrap,
                "_initialize_registries_and_security_gate",
                side_effect=required_init,
            ) as required,
            patch.object(route_bootstrap, "_start_registration_retry_loop") as retry,
            patch.dict(sys.modules, {"server": self._host_module(None)}),
        ):
            first = threading.Thread(
                target=self._call_registration,
                args=(errors,),
            )
            second = threading.Thread(
                target=self._call_registration,
                args=(errors,),
            )
            first.start()
            self.assertTrue(entered.wait(timeout=1))
            second.start()
            release.set()
            first.join(timeout=1)
            second.join(timeout=1)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(optional.call_count, 1)
        self.assertEqual(required.call_count, 1)
        self.assertEqual(retry.call_count, 1)
        diagnostics = get_startup_diagnostics()
        self.assertEqual(diagnostics["state"], "waiting_for_host")
        self.assertFalse(diagnostics["ready"])

    @staticmethod
    def _call_registration(errors):
        try:
            route_bootstrap.register_routes_once()
        except BaseException as exc:  # test capture; production must not swallow
            errors.append(exc)

    def test_required_failure_is_shared_fail_closed_and_content_free(self):
        marker = "PRIVATE_REQUIRED_FAILURE C:/private/config"
        failure = RuntimeError(marker)

        with (
            patch.object(route_bootstrap, "_register_plugins_and_shutdown_hooks"),
            patch.object(
                route_bootstrap,
                "_initialize_registries_and_security_gate",
                side_effect=failure,
            ) as required,
            patch.dict(sys.modules, {"server": self._host_module(None)}),
        ):
            with self.assertLogs("ComfyUI-OpenClaw", level="ERROR") as captured:
                observed = []
                for _ in range(2):
                    with self.assertRaises(RuntimeError) as ctx:
                        route_bootstrap.register_routes_once()
                    observed.append(ctx.exception)

        self.assertEqual(required.call_count, 1)
        self.assertIs(observed[0], failure)
        self.assertIs(observed[1], failure)
        diagnostics = get_startup_diagnostics()
        self.assertEqual(diagnostics["state"], "fatal")
        self.assertEqual(diagnostics["reason_code"], "required_initialization_failed")
        self.assertNotIn(marker, json.dumps(diagnostics))
        self.assertNotIn(marker, "\n".join(captured.output))

    def test_initial_registration_failure_is_fatal_and_reraised(self):
        marker = "PRIVATE_ROUTE_FAILURE C:/private/route"
        failure = RuntimeError(marker)
        server = SimpleNamespace(app=object())

        with (
            patch.object(route_bootstrap, "_register_plugins_and_shutdown_hooks"),
            patch.object(route_bootstrap, "_initialize_registries_and_security_gate"),
            patch.object(
                route_bootstrap,
                "_do_full_registration",
                side_effect=failure,
            ),
            patch.dict(sys.modules, {"server": self._host_module(server)}),
        ):
            with self.assertLogs("ComfyUI-OpenClaw", level="ERROR") as captured:
                with self.assertRaises(RuntimeError) as ctx:
                    route_bootstrap.register_routes_once()

        self.assertIs(ctx.exception, failure)
        diagnostics = get_startup_diagnostics()
        self.assertEqual(diagnostics["state"], "fatal")
        self.assertEqual(diagnostics["reason_code"], "route_registration_failed")
        self.assertNotIn(marker, json.dumps(diagnostics))
        self.assertNotIn(marker, "\n".join(captured.output))

    def test_initial_host_resolution_failure_is_terminal_and_replayed(self):
        marker = "PRIVATE_HOST_RESOLUTION C:/private/host"
        failure = RuntimeError(marker)

        with (
            patch.object(route_bootstrap, "_register_plugins_and_shutdown_hooks"),
            patch.object(route_bootstrap, "_initialize_registries_and_security_gate"),
            patch.object(
                route_bootstrap,
                "_resolve_prompt_server",
                side_effect=failure,
            ) as resolve,
        ):
            observed = []
            for _ in range(2):
                with self.assertRaises(RuntimeError) as ctx:
                    route_bootstrap.register_routes_once()
                observed.append(ctx.exception)

        self.assertEqual(resolve.call_count, 1)
        self.assertIs(observed[0], failure)
        self.assertIs(observed[1], failure)
        diagnostics = get_startup_diagnostics()
        self.assertTrue(diagnostics["fatal"])
        self.assertEqual(diagnostics["reason_code"], "route_registration_failed")
        self.assertNotIn(marker, json.dumps(diagnostics))

    def test_retry_owner_start_failure_is_terminal_and_replayed(self):
        marker = "PRIVATE_THREAD_START C:/private/thread"
        failure = RuntimeError(marker)

        with (
            patch.object(route_bootstrap, "_register_plugins_and_shutdown_hooks"),
            patch.object(route_bootstrap, "_initialize_registries_and_security_gate"),
            patch.object(route_bootstrap, "_resolve_prompt_server", return_value=None),
            patch.object(
                route_bootstrap,
                "_start_registration_retry_loop",
                side_effect=failure,
            ) as start_retry,
        ):
            observed = []
            for _ in range(2):
                with self.assertRaises(RuntimeError) as ctx:
                    route_bootstrap.register_routes_once()
                observed.append(ctx.exception)

        self.assertEqual(start_retry.call_count, 1)
        self.assertIs(observed[0], failure)
        self.assertIs(observed[1], failure)
        diagnostics = get_startup_diagnostics()
        self.assertTrue(diagnostics["fatal"])
        self.assertEqual(diagnostics["reason_code"], "retry_exhausted")
        self.assertNotIn(marker, json.dumps(diagnostics))

    def test_retry_success_and_exhaustion_have_distinct_outcomes(self):
        from services.startup_lifecycle import (
            mark_host_waiting,
            mark_required_initialization_started,
        )

        server = SimpleNamespace(app=object())
        sequence = iter([None, server])
        mark_required_initialization_started()
        mark_host_waiting(attempt=0, max_attempts=3)
        with (
            patch.object(
                route_bootstrap,
                "_resolve_prompt_server",
                side_effect=lambda: next(sequence),
            ),
            patch.object(route_bootstrap, "_do_full_registration") as register,
            patch.object(
                route_bootstrap,
                "_build_optional_startup_warmups",
                return_value=[],
            ),
        ):
            route_bootstrap._run_registration_retry_loop(
                max_attempts=3,
                initial_delay=0,
                sleep_fn=lambda _delay: None,
            )

        register.assert_called_once_with(server)
        diagnostics = get_startup_diagnostics()
        self.assertTrue(diagnostics["ready"])
        self.assertEqual(diagnostics["reason_code"], "route_registration_succeeded")

        reset_startup_lifecycle_for_tests()
        route_bootstrap.reset_route_bootstrap_for_tests()
        mark_required_initialization_started()
        mark_host_waiting(attempt=0, max_attempts=2)
        with patch.object(
            route_bootstrap,
            "_resolve_prompt_server",
            return_value=None,
        ):
            route_bootstrap._run_registration_retry_loop(
                max_attempts=2,
                initial_delay=0,
                sleep_fn=lambda _delay: None,
            )

        diagnostics = get_startup_diagnostics()
        self.assertFalse(diagnostics["ready"])
        self.assertTrue(diagnostics["fatal"])
        self.assertEqual(diagnostics["reason_code"], "retry_exhausted")
        self.assertEqual(diagnostics["attempt"], 2)

    def test_base_exception_is_not_swallowed_or_converted_to_fatal(self):
        signal = KeyboardInterrupt()

        with (
            patch.object(route_bootstrap, "_register_plugins_and_shutdown_hooks"),
            patch.object(
                route_bootstrap,
                "_initialize_registries_and_security_gate",
                side_effect=signal,
            ),
        ):
            with self.assertRaises(KeyboardInterrupt) as ctx:
                route_bootstrap.register_routes_once()

        self.assertIs(ctx.exception, signal)
        diagnostics = get_startup_diagnostics()
        self.assertFalse(diagnostics["fatal"])
        self.assertEqual(diagnostics["state"], "initializing")


class TestPublicHealthLifecycleProjection(unittest.TestCase):
    def setUp(self):
        reset_startup_lifecycle_for_tests()

    def tearDown(self):
        reset_startup_lifecycle_for_tests()

    @staticmethod
    def _health_payload(*, diagnostics_side_effect=None):
        from api.route_handlers import health_response

        web = SimpleNamespace(json_response=lambda data, **_kwargs: data)
        deps = SimpleNamespace(
            web=web,
            pack_start_time=time.time(),
            pack_name="openclaw",
            pack_version="test",
            metrics=SimpleNamespace(
                get_snapshot=lambda: {
                    "errors_captured": 0,
                    "logs_processed": 0,
                }
            ),
            get_executor_diagnostics=lambda: {},
            check_dependency=lambda _name: True,
        )
        client = MagicMock()
        client.get_provider_summary.return_value = {
            "provider": "openai",
            "model": "test",
            "key_configured": False,
        }
        diagnostics_patch = patch(
            "services.startup_lifecycle.get_startup_diagnostics",
            side_effect=diagnostics_side_effect,
        )
        if diagnostics_side_effect is None:
            diagnostics_patch = patch(
                "services.startup_lifecycle.get_startup_diagnostics",
                wraps=get_startup_diagnostics,
            )

        with (
            patch("services.llm_client.LLMClient", return_value=client),
            patch("services.providers.keys.requires_api_key", return_value=True),
            patch(
                "services.job_events.get_job_event_store",
                return_value=SimpleNamespace(stats=lambda: {}),
            ),
            patch(
                "services.capabilities._get_control_plane_info",
                return_value={},
            ),
            patch(
                "services.runtime_profile.get_runtime_profile",
                return_value="minimal",
            ),
            diagnostics_patch,
        ):
            return asyncio.run(health_response(SimpleNamespace(), deps))

    def test_health_uses_exact_schema_and_content_free_fallback(self):
        normal = self._health_payload()
        self.assertEqual(
            tuple(normal["startup"]),
            STARTUP_DIAGNOSTIC_KEYS,
        )

        marker = "PRIVATE_HEALTH_FAILURE C:/private/health"
        fallback = self._health_payload(diagnostics_side_effect=RuntimeError(marker))
        self.assertEqual(
            tuple(fallback["startup"]),
            STARTUP_DIAGNOSTIC_KEYS,
        )
        self.assertEqual(fallback["startup"]["schema_version"], 1)
        self.assertEqual(fallback["startup"]["state"], "fatal")
        self.assertEqual(
            fallback["startup"]["reason_code"],
            "bootstrap_import_failed",
        )
        self.assertNotIn(marker, json.dumps(fallback["startup"]))

    def test_health_metadata_remains_public_and_all_aliases_share_one_handler(self):
        from api.route_registrars import build_core_route_specs
        from api.routes import health_handler
        from services.endpoint_manifest import AuthTier, get_metadata

        metadata = get_metadata(health_handler)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.auth_tier, AuthTier.PUBLIC)

        handlers = {"health_handler": health_handler}
        sentinel = MagicMock()
        handlers.update(
            {
                key: sentinel
                for key in (
                    "remote_admin_page_handler",
                    "logs_tail_handler",
                    "jobs_handler",
                    "trace_handler",
                    "webhook_handler",
                    "webhook_submit_handler",
                    "webhook_validate_handler",
                    "capabilities_handler",
                    "config_get_handler",
                    "config_put_handler",
                    "llm_test_handler",
                    "llm_chat_handler",
                    "llm_models_handler",
                    "templates_list_handler",
                    "preflight_handler",
                    "inventory_handler",
                    "pnginfo_handler",
                    "list_checkpoints_handler",
                    "create_checkpoint_handler",
                    "get_checkpoint_handler",
                    "delete_checkpoint_handler",
                    "rewrite_recipes_list_handler",
                    "rewrite_recipe_create_handler",
                    "rewrite_recipe_get_handler",
                    "rewrite_recipe_update_handler",
                    "rewrite_recipe_delete_handler",
                    "rewrite_recipe_dry_run_handler",
                    "rewrite_recipe_apply_handler",
                    "model_search_handler",
                    "model_download_create_handler",
                    "model_download_list_handler",
                    "model_download_get_handler",
                    "model_download_cancel_handler",
                    "model_import_handler",
                    "model_installations_list_handler",
                    "secrets_status_handler",
                    "secrets_put_handler",
                    "events_stream_handler",
                    "events_poll_handler",
                    "secrets_delete_handler",
                    "security_doctor_handler",
                    "tools_list_handler",
                    "tools_run_handler",
                    "create_sweep_handler",
                    "create_compare_handler",
                    "list_experiments_handler",
                    "get_experiment_handler",
                    "update_experiment_handler",
                    "select_apply_winner_handler",
                )
            }
        )

        for prefix in ("/openclaw", "/api/openclaw", "/moltbot", "/api/moltbot"):
            specs = build_core_route_specs(prefix, handlers)
            health = next(spec for spec in specs if spec.path == f"{prefix}/health")
            self.assertIs(health.handler, health_handler)


class TestPackageBootstrapLifecycleProjection(unittest.TestCase):
    def setUp(self):
        reset_startup_lifecycle_for_tests()

    def tearDown(self):
        reset_startup_lifecycle_for_tests()

    @staticmethod
    def _run_entrypoint_with_import_signal(signal):
        original_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name == "services.route_bootstrap":
                raise signal
            return original_import(name, *args, **kwargs)

        root = Path(__file__).resolve().parents[1]
        with patch("builtins.__import__", side_effect=guarded_import):
            return runpy.run_path(
                str(root / "__init__.py"),
                run_name="openclaw_bootstrap_lifecycle_probe",
            )

    def test_package_import_failure_is_stably_classified_without_payload(self):
        marker = "PRIVATE_IMPORT_FAILURE C:/private/import"

        self._run_entrypoint_with_import_signal(ImportError(marker))

        diagnostics = get_startup_diagnostics()
        self.assertEqual(diagnostics["state"], "fatal")
        self.assertEqual(
            diagnostics["reason_code"],
            "bootstrap_import_failed",
        )
        self.assertNotIn(marker, json.dumps(diagnostics))

    def test_package_import_base_exception_is_reraised_unchanged(self):
        signal = KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt) as ctx:
            self._run_entrypoint_with_import_signal(signal)

        self.assertIs(ctx.exception, signal)
        self.assertFalse(get_startup_diagnostics()["fatal"])


if __name__ == "__main__":
    unittest.main()
