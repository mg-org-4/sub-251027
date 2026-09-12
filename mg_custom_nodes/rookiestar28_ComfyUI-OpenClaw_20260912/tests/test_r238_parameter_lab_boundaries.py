import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from services import parameter_lab as lab

try:
    from aiohttp import web
    from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
except Exception:  # pragma: no cover
    web = None  # type: ignore
    AioHTTPTestCase = unittest.TestCase  # type: ignore

    def unittest_run_loop(fn):  # type: ignore
        return fn


MIB = 1024 * 1024
KIB = 1024


class TestR238PlannerBoundaries(unittest.TestCase):
    def setUp(self):
        self.planner = lab.SweepPlanner()

    def assert_reason(self, ctx, expected):
        self.assertEqual(expected, getattr(ctx.exception, "code", None))

    def test_policy_constants_are_frozen(self):
        expected = {
            "PARAMETER_LAB_POLICY_VERSION": "1.0",
            "MAX_PARAMETER_LAB_REQUEST_BYTES": 5 * MIB,
            "MAX_PARAMETER_LAB_WORKFLOW_UTF8_BYTES": 4 * MIB,
            "MAX_SWEEP_DIMENSIONS": 8,
            "MAX_VALUES_PER_DIMENSION": 50,
            "MAX_NODE_ID_UTF8_BYTES": 128,
            "MAX_WIDGET_NAME_UTF8_BYTES": 256,
            "MAX_SCALAR_STRING_UTF8_BYTES": 16 * KIB,
            "MAX_PARAMETER_LAB_PLAN_UTF8_BYTES": 8 * MIB,
            "MAX_SWEEP_COMBINATIONS": 50,
        }
        for name, value in expected.items():
            self.assertEqual(value, getattr(lab, name, None), name)

    def test_sweep_rejects_structured_and_non_finite_values(self):
        invalid_values = [
            None,
            [],
            {"rich": "widget"},
            float("nan"),
            float("inf"),
            float("-inf"),
        ]
        for value in invalid_values:
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(ValueError) as ctx:
                    self.planner.generate(
                        "{}",
                        [{"node_id": 1, "widget_name": "seed", "values": [value]}],
                    )
                self.assert_reason(ctx, "invalid_scalar_value")

    def test_sweep_rejects_dimension_and_value_count_limits(self):
        with self.assertRaises(ValueError) as ctx:
            self.planner.generate(
                "{}",
                [
                    {"node_id": index, "widget_name": "seed", "values": [index]}
                    for index in range(9)
                ],
            )
        self.assert_reason(ctx, "too_many_dimensions")

        with self.assertRaises(ValueError) as ctx:
            self.planner.generate(
                "{}",
                [
                    {
                        "node_id": 1,
                        "widget_name": "seed",
                        "values": list(range(51)),
                    }
                ],
            )
        self.assert_reason(ctx, "too_many_values")

    def test_sweep_rejects_malformed_duplicate_and_ambiguous_dimensions(self):
        cases = [
            ([{"node_id": 1, "widget_name": "seed"}], "values_required"),
            (
                [{"node_id": "bad.id", "widget_name": "seed", "values": [1]}],
                "invalid_node_id",
            ),
            (
                [{"node_id": True, "widget_name": "seed", "values": [1]}],
                "invalid_node_id",
            ),
            (
                [{"node_id": 1, "widget_name": "bad\u0000name", "values": [1]}],
                "invalid_widget_name",
            ),
            (
                [
                    {"node_id": 1, "widget_name": "seed", "values": [1]},
                    {"node_id": "1", "widget_name": "seed", "values": [2]},
                ],
                "duplicate_dimension",
            ),
            (
                [{"node_id": 1, "widget_name": "seed", "values": [1, "1"]}],
                "duplicate_ambiguous_value",
            ),
            (
                [
                    {
                        "node_id": 1,
                        "widget_name": "seed",
                        "values": [1],
                        "strategy": "random",
                    }
                ],
                "invalid_strategy",
            ),
        ]
        for params, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaises(ValueError) as ctx:
                    self.planner.generate("{}", params)
                self.assert_reason(ctx, reason)

    def test_sweep_uses_utf8_identifier_and_scalar_string_limits(self):
        cases = [
            (
                [{"node_id": "界" * 43, "widget_name": "seed", "values": [1]}],
                "node_id_too_large",
            ),
            (
                [{"node_id": 1, "widget_name": "界" * 86, "values": [1]}],
                "widget_name_too_large",
            ),
            (
                [{"node_id": 1, "widget_name": "seed", "values": ["界" * 5462]}],
                "scalar_string_too_large",
            ),
        ]
        for params, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaises(ValueError) as ctx:
                    self.planner.generate("{}", params)
                self.assert_reason(ctx, reason)

    def test_exact_workflow_identifier_and_scalar_limits_are_accepted(self):
        plan = self.planner.generate(
            "x" * (4 * MIB),
            [
                {
                    "node_id": "n" * 128,
                    "widget_name": "w" * 256,
                    "values": ["v" * (16 * KIB)],
                }
            ],
        )

        self.assertEqual(4 * MIB, len(plan.workflow_json.encode("utf-8")))
        self.assertEqual(128, len(plan.dimensions[0].node_id.encode("utf-8")))
        self.assertEqual(256, len(plan.dimensions[0].widget_name.encode("utf-8")))
        self.assertEqual(
            16 * KIB,
            len(plan.dimensions[0].values[0].encode("utf-8")),
        )

    def test_workflow_and_plan_limits_run_before_experiment_id_allocation(self):
        with patch.object(lab.uuid, "uuid4", wraps=lab.uuid.uuid4) as mock_uuid:
            with self.assertRaises(ValueError) as ctx:
                self.planner.generate("x" * (4 * MIB + 1), [])
            self.assert_reason(ctx, "workflow_too_large")
            mock_uuid.assert_not_called()

        scalar = "x" * (16 * KIB)
        first_values = [f"{index:02d}" + scalar[2:] for index in range(50)]
        params = [
            {"node_id": 1, "widget_name": "w1", "values": first_values},
            *[
                {"node_id": index, "widget_name": f"w{index}", "values": [scalar]}
                for index in range(2, 9)
            ],
        ]
        with patch.object(lab.uuid, "uuid4", wraps=lab.uuid.uuid4) as mock_uuid:
            with self.assertRaises(ValueError) as ctx:
                self.planner.generate("x" * (4 * MIB - KIB), params)
            self.assert_reason(ctx, "plan_too_large")
            mock_uuid.assert_not_called()

    def test_compare_uses_the_same_scalar_and_identifier_policy(self):
        planner = lab.ComparePlanner()
        cases = [
            ([None], 1, "ckpt_name", "invalid_scalar_value"),
            ([1, "1"], 1, "ckpt_name", "duplicate_ambiguous_value"),
            (["a"], "bad.id", "ckpt_name", "invalid_node_id"),
            (["a"], 1, "界" * 86, "widget_name_too_large"),
        ]
        for items, node_id, widget_name, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaises(ValueError) as ctx:
                    planner.generate("{}", items, node_id, widget_name)
                self.assert_reason(ctx, reason)


class TestR238StoreBoundaries(unittest.TestCase):
    def test_store_revalidates_plan_size_before_file_or_retention_mutation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = lab.ExperimentStore(Path(tmp_dir))
            plan = lab.SweepPlan(
                experiment_id="exp_too_large",
                workflow_json="x" * (8 * MIB + 1),
                dimensions=[],
                runs=[],
            )
            with (
                patch.object(store, "_enforce_retention") as retention,
                self.assertRaises(ValueError) as ctx,
            ):
                store.save_plan(plan)
            self.assertEqual("plan_too_large", getattr(ctx.exception, "code", None))
            retention.assert_not_called()
            self.assertEqual([], list(store.store_dir.glob("*.json")))

    def test_atomic_write_failure_leaves_no_plan_and_skips_retention(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = lab.ExperimentStore(Path(tmp_dir))
            plan = lab.SweepPlanner().generate(
                "{}",
                [{"node_id": "loader-alpha", "widget_name": "seed", "values": [1]}],
            )
            with (
                patch.object(
                    lab,
                    "safe_write_text",
                    create=True,
                    side_effect=OSError("private failure detail"),
                ),
                patch.object(store, "_enforce_retention") as retention,
                self.assertRaises(OSError),
            ):
                store.save_plan(plan)
            retention.assert_not_called()
            self.assertEqual([], list(store.store_dir.glob("*.json")))

    def test_legacy_read_does_not_rewrite_source_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = lab.ExperimentStore(Path(tmp_dir))
            path = store.store_dir / "exp_legacy.json"
            original = b'{"experiment_id":"exp_legacy","runs":[]}'
            path.write_bytes(original)

            loaded = store.get_plan("exp_legacy")

            self.assertEqual("0.9", loaded["schema_version"])
            self.assertEqual(original, path.read_bytes())


@unittest.skipIf(web is None, "aiohttp not available")
class TestR238CreationHandlerBoundaries(AioHTTPTestCase):
    async def get_application(self):
        app = web.Application(client_max_size=6 * MIB)
        app.router.add_post("/openclaw/lab/sweep", lab.create_sweep_handler)
        app.router.add_post("/openclaw/lab/compare", lab.create_compare_handler)
        return app

    async def _post_with_store(self, path, payload, *, store=None):
        store = store or MagicMock()
        with (
            patch("services.parameter_lab.check_rate_limit", return_value=True),
            patch(
                "services.parameter_lab.require_admin_token", return_value=(True, None)
            ),
            patch("services.parameter_lab.get_store", return_value=store),
        ):
            response = await self.client.post(path, json=payload)
            data = await response.json()
        return response, data, store

    @unittest_run_loop
    async def test_structured_value_is_rejected_before_store_access(self):
        response, data, store = await self._post_with_store(
            "/openclaw/lab/sweep",
            {
                "workflow_json": "{}",
                "params": [
                    {
                        "node_id": 1,
                        "widget_name": "video_edit",
                        "values": [{"trim": [0, 1]}],
                    }
                ],
            },
        )
        self.assertEqual(400, response.status)
        self.assertEqual("invalid_scalar_value", data["error"])
        store.save_plan.assert_not_called()

    @unittest_run_loop
    async def test_oversized_workflow_is_rejected_before_store_access(self):
        response, data, store = await self._post_with_store(
            "/openclaw/lab/sweep",
            {
                "workflow_json": "x" * (4 * MIB + 1),
                "params": [{"node_id": 1, "widget_name": "seed", "values": [1]}],
            },
        )
        self.assertEqual(413, response.status)
        self.assertEqual("workflow_too_large", data["error"])
        store.save_plan.assert_not_called()

    @unittest_run_loop
    async def test_oversized_request_is_rejected_before_json_or_store_mutation(self):
        response, data, store = await self._post_with_store(
            "/openclaw/lab/sweep",
            {
                "workflow_json": "{}",
                "params": [
                    {
                        "node_id": 1,
                        "widget_name": "seed",
                        "values": ["x" * (5 * MIB)],
                    }
                ],
            },
        )
        self.assertEqual(413, response.status)
        self.assertEqual("payload_too_large", data["error"])
        store.save_plan.assert_not_called()

    @unittest_run_loop
    async def test_request_at_exact_byte_limit_is_accepted(self):
        prefix = (
            b'{"workflow_json":"{}",'
            b'"params":[{"node_id":1,"widget_name":"seed","values":[1]}],'
            b'"padding":"'
        )
        suffix = b'"}'
        body = prefix + (b"x" * (5 * MIB - len(prefix) - len(suffix))) + suffix
        self.assertEqual(5 * MIB, len(body))

        store = MagicMock()
        with (
            patch("services.parameter_lab.check_rate_limit", return_value=True),
            patch(
                "services.parameter_lab.require_admin_token", return_value=(True, None)
            ),
            patch("services.parameter_lab.get_store", return_value=store),
        ):
            response = await self.client.post(
                "/openclaw/lab/sweep",
                data=io.BytesIO(body),
                headers={"Content-Type": "application/json"},
            )
            data = await response.json()

        self.assertEqual(200, response.status)
        self.assertTrue(data["ok"])
        store.save_plan.assert_called_once()

    @unittest_run_loop
    async def test_store_failure_returns_and_logs_only_content_free_classification(
        self,
    ):
        store = MagicMock()
        store.save_plan.side_effect = OSError("secret=private-state-path")
        with self.assertLogs(lab.logger.name, level="ERROR") as captured:
            response, data, _ = await self._post_with_store(
                "/openclaw/lab/sweep",
                {
                    "workflow_json": "{}",
                    "params": [{"node_id": 1, "widget_name": "seed", "values": [1]}],
                },
                store=store,
            )

        self.assertEqual(500, response.status)
        self.assertEqual("internal_error", data["error"])
        rendered = "\n".join(captured.output)
        self.assertIn("OSError", rendered)
        self.assertNotIn("private-state-path", rendered)
        self.assertNotIn("secret=", rendered)


if __name__ == "__main__":
    unittest.main()
