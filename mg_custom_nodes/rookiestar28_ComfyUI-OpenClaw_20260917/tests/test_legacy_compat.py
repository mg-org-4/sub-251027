import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from services.legacy_compat import (
    ADMIN_TOKEN_HEADERS,
    LEGACY_API_PREFIX,
    OPENCLAW_API_PREFIX,
    build_legacy_route_deprecation_headers,
    get_api_path_candidates,
    get_header_alias_value,
    iter_legacy_compatibility_entries,
)


class TestLegacyCompat(unittest.TestCase):
    def test_get_header_alias_value_prefers_primary(self):
        logger = MagicMock()
        value, used_legacy = get_header_alias_value(
            {
                ADMIN_TOKEN_HEADERS.primary: "new-token",
                ADMIN_TOKEN_HEADERS.legacy: "old-token",
            },
            ADMIN_TOKEN_HEADERS,
            logger=logger,
        )

        self.assertEqual(value, "new-token")
        self.assertFalse(used_legacy)
        logger.warning.assert_not_called()

    def test_get_header_alias_value_uses_legacy_and_logs(self):
        logger = MagicMock()
        with patch("services.legacy_compat._increment_legacy_api_hits") as inc:
            value, used_legacy = get_header_alias_value(
                {ADMIN_TOKEN_HEADERS.legacy: "old-token"},
                ADMIN_TOKEN_HEADERS,
                logger=logger,
            )

        self.assertEqual(value, "old-token")
        self.assertTrue(used_legacy)
        inc.assert_called_once()
        logger.warning.assert_called_once()

    def test_get_api_path_candidates_handles_canonical_and_legacy_prefixes(self):
        self.assertEqual(
            get_api_path_candidates(f"{OPENCLAW_API_PREFIX}/health"),
            (
                f"{OPENCLAW_API_PREFIX}/health",
                f"{LEGACY_API_PREFIX}/health",
            ),
        )
        self.assertEqual(
            get_api_path_candidates(f"{LEGACY_API_PREFIX}/health"),
            (
                f"{LEGACY_API_PREFIX}/health",
                f"{OPENCLAW_API_PREFIX}/health",
            ),
        )
        self.assertEqual(get_api_path_candidates("/history/abc"), ("/history/abc",))

    def test_governance_registry_covers_remaining_legacy_alias_surfaces(self):
        entries = tuple(iter_legacy_compatibility_entries())
        surfaces = {entry.surface for entry in entries}

        self.assertIn("api_path", surfaces)
        self.assertIn("header", surfaces)
        self.assertIn("environment", surfaces)
        self.assertIn("ui_class", surfaces)
        self.assertIn("workflow_node", surfaces)

        for entry in entries:
            with self.subTest(entry=entry.key):
                self.assertTrue(entry.key)
                self.assertTrue(entry.status)
                self.assertGreater(entry.review_cadence_days, 0)
                self.assertTrue(entry.telemetry_signal)
                self.assertTrue(entry.removal_criteria)
                self.assertTrue(entry.review_trigger)

    def test_legacy_route_deprecation_headers_include_canonical_path(self):
        headers = build_legacy_route_deprecation_headers("/api/moltbot/health")

        self.assertEqual(headers["Deprecation"], "true")
        self.assertEqual(headers["X-OpenClaw-Canonical-Path"], "/api/openclaw/health")
        self.assertEqual(
            headers["X-OpenClaw-Compatibility-Status"],
            "deprecated-observed",
        )
        self.assertEqual(
            headers["X-OpenClaw-Compatibility-Telemetry"],
            "legacy_api_hits",
        )

    def test_governance_doc_mentions_every_registry_key(self):
        repo_root = Path(__file__).resolve().parents[1]
        doc = (repo_root / "docs" / "legacy-compatibility-governance.md").read_text(
            encoding="utf-8"
        )

        for entry in iter_legacy_compatibility_entries():
            with self.subTest(entry=entry.key):
                self.assertIn(entry.key, doc)


if __name__ == "__main__":
    unittest.main()
