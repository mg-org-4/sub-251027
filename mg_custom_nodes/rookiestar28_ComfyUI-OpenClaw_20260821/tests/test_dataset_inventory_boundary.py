from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import services.preflight
from services.model_manager import (
    MODEL_TYPE_EXCLUSION_REASONS,
    ModelManager,
    ModelManagerError,
    _model_type_exclusion_reason,
    _norm_model_type,
)

try:
    from aiohttp import web
    from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
except Exception:  # pragma: no cover
    web = None  # type: ignore
    AioHTTPTestCase = unittest.TestCase  # type: ignore

    def unittest_run_loop(fn):  # type: ignore
        return fn


DATASET_SENTINEL = "private-subject-caption-0001.txt"
DATASET_REASON = (
    "datasets contain user-managed training data, not managed model weights"
)


class DatasetInventoryServiceBoundaryTests(unittest.TestCase):
    def setUp(self):
        services.preflight._reset_inventory_state_for_tests()

    def tearDown(self):
        services.preflight._reset_inventory_state_for_tests()

    @staticmethod
    def _folder_paths() -> MagicMock:
        folder_paths = MagicMock()
        folder_paths.folder_names_and_paths = {
            "checkpoints": [],
            "clip": [],
            "unet": [],
            "datasets": [],
            "custom_nodes": [],
        }
        folder_paths.get_filename_list.side_effect = lambda model_type: (
            [DATASET_SENTINEL] if model_type == "datasets" else []
        )
        return folder_paths

    def test_dynamic_resolution_and_scan_exclude_dataset_user_data(self):
        folder_paths = self._folder_paths()

        with patch.object(services.preflight, "folder_paths", folder_paths):
            model_types = services.preflight._resolve_inventory_model_types()
            snapshot = services.preflight._scan_model_inventory()

        self.assertNotIn("datasets", model_types)
        self.assertNotIn("custom_nodes", model_types)
        self.assertIn("text_encoders", model_types)
        self.assertIn("diffusion_models", model_types)
        self.assertNotIn("datasets", snapshot)
        self.assertNotIn(DATASET_SENTINEL, json.dumps(snapshot))
        scanned_types = {
            call.args[0] for call in folder_paths.get_filename_list.call_args_list
        }
        self.assertNotIn("datasets", scanned_types)
        self.assertNotIn("custom_nodes", scanned_types)

    def test_preexisting_cached_dataset_snapshot_is_filtered_on_copy(self):
        folder_paths = self._folder_paths()
        services.preflight._CACHE[services.preflight._INVENTORY_SNAPSHOT_KEY] = {
            "checkpoints": ["safe.safetensors"],
            "datasets": [DATASET_SENTINEL],
        }
        services.preflight._CACHE[services.preflight._INVENTORY_SNAPSHOT_TS_KEY] = 123.0

        with patch.object(services.preflight, "folder_paths", folder_paths):
            snapshot = services.preflight.get_model_inventory_snapshot(
                trigger_refresh=False
            )

        self.assertEqual(snapshot["models"], {"checkpoints": ["safe.safetensors"]})
        self.assertNotIn(DATASET_SENTINEL, json.dumps(snapshot))
        self.assertEqual(snapshot["snapshot_ts"], 123.0)
        self.assertIn(snapshot["scan_state"], {"idle", "error", "refreshing"})

    def test_background_publication_filters_defensive_dataset_result(self):
        folder_paths = self._folder_paths()

        with (
            patch.object(services.preflight, "folder_paths", folder_paths),
            patch.object(
                services.preflight,
                "_scan_model_inventory",
                return_value={
                    "checkpoints": ["safe.safetensors"],
                    "datasets": [DATASET_SENTINEL],
                },
            ),
        ):
            services.preflight._inventory_refresh_worker()
            snapshot = services.preflight.get_model_inventory_snapshot(
                trigger_refresh=False
            )

        self.assertEqual(snapshot["models"], {"checkpoints": ["safe.safetensors"]})
        self.assertEqual(snapshot["scan_state"], "idle")
        self.assertFalse(snapshot["stale"])
        self.assertIsNone(snapshot["last_error"])
        self.assertNotIn(DATASET_SENTINEL, json.dumps(snapshot))


class DatasetModelManagerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.manager = ModelManager(
            state_root=root / "state",
            install_root=root / "install",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_type_has_stable_user_data_exclusion(self):
        self.assertEqual(MODEL_TYPE_EXCLUSION_REASONS["datasets"], DATASET_REASON)
        for value in ("datasets", " DATASETS ", "Datasets"):
            with self.subTest(value=value):
                self.assertEqual(_norm_model_type(value), "other")
                self.assertEqual(_model_type_exclusion_reason(value), DATASET_REASON)
        self.assertNotIn("datasets", self.manager._model_type_to_subdir)

    def test_dataset_download_is_rejected_before_url_path_or_task_work(self):
        payload = {
            "model_id": "dataset-attempt",
            "name": "Dataset Attempt",
            "model_type": "datasets",
            "source": "catalog",
            "source_label": "Catalog",
            "download_url": f"https://example.invalid/{DATASET_SENTINEL}",
            "expected_sha256": "a" * 64,
            "provenance": {
                "publisher": "private-publisher-sentinel",
                "license": "private-license-sentinel",
                "source_url": "https://example.invalid/private-source",
            },
            "destination_subdir": "../private-dataset",
            "filename": DATASET_SENTINEL,
        }

        with (
            patch.object(self.manager, "_assert_budget") as assert_budget,
            patch.object(
                self.manager,
                "_validate_url_policy",
                side_effect=AssertionError(
                    "dataset exclusion must run before URL validation"
                ),
            ) as validate_url,
            patch.object(self.manager, "_validate_provenance") as validate_provenance,
            patch.object(self.manager, "_sanitize_subdir") as sanitize_subdir,
            patch.object(self.manager, "_sanitize_filename") as sanitize_filename,
            self.assertRaises(ModelManagerError) as caught,
        ):
            self.manager.create_download_task(**payload)

        self.assertEqual(caught.exception.code, "unsupported_model_type")
        self.assertEqual(
            caught.exception.detail,
            "model_type 'datasets' is not supported for managed install/import: "
            + DATASET_REASON,
        )
        assert_budget.assert_called_once_with()
        validate_url.assert_not_called()
        validate_provenance.assert_not_called()
        sanitize_subdir.assert_not_called()
        sanitize_filename.assert_not_called()
        self.assertEqual(self.manager._tasks, {})
        self.assertEqual(list(self.manager.install_root.rglob("*")), [])


@unittest.skipIf(web is None, "aiohttp not installed")
class DatasetInventoryAliasApiTests(AioHTTPTestCase):
    def setUp(self):
        super().setUp()
        services.preflight._reset_inventory_state_for_tests()

    def tearDown(self):
        services.preflight._reset_inventory_state_for_tests()
        super().tearDown()

    async def get_application(self):
        from api.preflight_handler import inventory_handler

        app = web.Application()
        for prefix in (
            "/openclaw",
            "/moltbot",
            "/api/openclaw",
            "/api/moltbot",
        ):
            app.router.add_get(f"{prefix}/preflight/inventory", inventory_handler)
        return app

    @patch("api.preflight_handler.check_rate_limit", return_value=True)
    @patch("api.preflight_handler.require_admin_token", return_value=(True, None))
    @patch("api.preflight_handler._get_node_class_mappings", return_value={})
    @unittest_run_loop
    async def test_every_inventory_alias_omits_cached_dataset_names(
        self,
        _nodes_mock,
        _admin_mock,
        _rate_limit_mock,
    ):
        services.preflight._CACHE[services.preflight._INVENTORY_SNAPSHOT_KEY] = {
            "checkpoints": ["safe.safetensors"],
            "datasets": [DATASET_SENTINEL],
        }
        services.preflight._CACHE[services.preflight._INVENTORY_SNAPSHOT_TS_KEY] = (
            time.time()
        )
        folder_paths = DatasetInventoryServiceBoundaryTests._folder_paths()

        with patch.object(services.preflight, "folder_paths", folder_paths):
            for prefix in (
                "/openclaw",
                "/moltbot",
                "/api/openclaw",
                "/api/moltbot",
            ):
                with self.subTest(prefix=prefix):
                    response = await self.client.get(f"{prefix}/preflight/inventory")
                    payload = await response.json()
                    rendered = json.dumps(payload)
                    self.assertEqual(response.status, 200)
                    self.assertEqual(
                        payload["models"], {"checkpoints": ["safe.safetensors"]}
                    )
                    self.assertNotIn("datasets", rendered)
                    self.assertNotIn(DATASET_SENTINEL, rendered)
                    self.assertIsInstance(payload["snapshot_ts"], float)
                    self.assertEqual(payload["scan_state"], "idle")
                    self.assertFalse(payload["stale"])


if __name__ == "__main__":
    unittest.main()
