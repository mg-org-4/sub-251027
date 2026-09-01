import os
import unittest
from unittest.mock import patch

from connector import config as connector_config


class TestR177ConnectorConfigEnv(unittest.TestCase):
    def test_invalid_delivery_envs_fall_back_to_defaults_with_warnings(self):
        with patch.dict(
            os.environ,
            {
                "OPENCLAW_CONNECTOR_DELIVERY_MAX_IMAGES": "NaN",
                "OPENCLAW_CONNECTOR_DELIVERY_MAX_BYTES": "10mb",
                "OPENCLAW_CONNECTOR_DELIVERY_TIMEOUT_SEC": "forever",
            },
            clear=False,
        ):
            with self.assertLogs(
                "ComfyUI-OpenClaw.connector.config", level="WARNING"
            ) as logs:
                cfg = connector_config.load_config()

        self.assertEqual(
            cfg.delivery_max_images,
            connector_config.DEFAULT_DELIVERY_MAX_IMAGES,
        )
        self.assertEqual(
            cfg.delivery_max_bytes,
            connector_config.DEFAULT_DELIVERY_MAX_BYTES,
        )
        self.assertEqual(
            cfg.delivery_timeout_sec,
            connector_config.DEFAULT_DELIVERY_TIMEOUT_SEC,
        )
        rendered = "\n".join(logs.output)
        self.assertIn("OPENCLAW_CONNECTOR_DELIVERY_MAX_IMAGES", rendered)
        self.assertIn("OPENCLAW_CONNECTOR_DELIVERY_MAX_BYTES", rendered)
        self.assertIn("OPENCLAW_CONNECTOR_DELIVERY_TIMEOUT_SEC", rendered)
        self.assertIn("using default", rendered)

    def test_delivery_bounds_are_clamped(self):
        with patch.dict(
            os.environ,
            {
                "OPENCLAW_CONNECTOR_DELIVERY_MAX_IMAGES": "0",
                "OPENCLAW_CONNECTOR_DELIVERY_MAX_BYTES": str(
                    connector_config.MAX_DELIVERY_MAX_BYTES + 1
                ),
                "OPENCLAW_CONNECTOR_DELIVERY_TIMEOUT_SEC": "-5",
            },
            clear=False,
        ):
            with self.assertLogs(
                "ComfyUI-OpenClaw.connector.config", level="WARNING"
            ) as logs:
                cfg = connector_config.load_config()

        self.assertEqual(
            cfg.delivery_max_images,
            connector_config.MIN_DELIVERY_MAX_IMAGES,
        )
        self.assertEqual(
            cfg.delivery_max_bytes,
            connector_config.MAX_DELIVERY_MAX_BYTES,
        )
        self.assertEqual(
            cfg.delivery_timeout_sec,
            connector_config.MIN_DELIVERY_TIMEOUT_SEC,
        )
        rendered = "\n".join(logs.output)
        self.assertIn("clamped", rendered)
        self.assertIn("OPENCLAW_CONNECTOR_DELIVERY_MAX_IMAGES", rendered)
        self.assertIn("OPENCLAW_CONNECTOR_DELIVERY_MAX_BYTES", rendered)
        self.assertIn("OPENCLAW_CONNECTOR_DELIVERY_TIMEOUT_SEC", rendered)

    def test_invalid_ports_fall_back_to_defaults(self):
        with patch.dict(
            os.environ,
            {
                "OPENCLAW_CONNECTOR_LINE_PORT": "0",
                "OPENCLAW_CONNECTOR_SLACK_PORT": "99999",
                "OPENCLAW_CONNECTOR_FEISHU_PORT": "not-a-port",
            },
            clear=False,
        ):
            with self.assertLogs(
                "ComfyUI-OpenClaw.connector.config", level="WARNING"
            ) as logs:
                cfg = connector_config.load_config()

        self.assertEqual(cfg.line_bind_port, connector_config.DEFAULT_LINE_BIND_PORT)
        self.assertEqual(cfg.slack_bind_port, connector_config.DEFAULT_SLACK_BIND_PORT)
        self.assertEqual(
            cfg.feishu_bind_port,
            connector_config.DEFAULT_FEISHU_BIND_PORT,
        )
        rendered = "\n".join(logs.output)
        self.assertIn("OPENCLAW_CONNECTOR_LINE_PORT", rendered)
        self.assertIn("OPENCLAW_CONNECTOR_SLACK_PORT", rendered)
        self.assertIn("OPENCLAW_CONNECTOR_FEISHU_PORT", rendered)
        self.assertIn("using default", rendered)

    def test_timeout_and_media_limits_are_bounded(self):
        with patch.dict(
            os.environ,
            {
                "OPENCLAW_CONNECTOR_SLACK_OAUTH_STATE_TTL_SEC": "10",
                "OPENCLAW_CONNECTOR_MEDIA_TTL_SEC": "999999",
                "OPENCLAW_CONNECTOR_MEDIA_MAX_MB": "0",
                "OPENCLAW_CONNECTOR_RATE_LIMIT_USER_RPM": "-1",
                "OPENCLAW_CONNECTOR_RATE_LIMIT_CHANNEL_RPM": "10000",
                "OPENCLAW_CONNECTOR_MAX_COMMAND_LENGTH": "64",
            },
            clear=False,
        ):
            with self.assertLogs(
                "ComfyUI-OpenClaw.connector.config", level="WARNING"
            ) as logs:
                cfg = connector_config.load_config()

        self.assertEqual(
            cfg.slack_oauth_state_ttl_sec,
            connector_config.MIN_SLACK_OAUTH_STATE_TTL_SEC,
        )
        self.assertEqual(cfg.media_ttl_sec, connector_config.MAX_MEDIA_TTL_SEC)
        self.assertEqual(cfg.media_max_mb, connector_config.MIN_MEDIA_MAX_MB)
        self.assertEqual(cfg.rate_limit_user_rpm, connector_config.MIN_RATE_LIMIT_RPM)
        self.assertEqual(
            cfg.rate_limit_channel_rpm,
            connector_config.MAX_RATE_LIMIT_RPM,
        )
        self.assertEqual(
            cfg.max_command_length,
            connector_config.MIN_MAX_COMMAND_LENGTH,
        )
        rendered = "\n".join(logs.output)
        self.assertIn("OPENCLAW_CONNECTOR_SLACK_OAUTH_STATE_TTL_SEC", rendered)
        self.assertIn("OPENCLAW_CONNECTOR_MEDIA_TTL_SEC", rendered)
        self.assertIn("OPENCLAW_CONNECTOR_MEDIA_MAX_MB", rendered)
        self.assertIn("OPENCLAW_CONNECTOR_RATE_LIMIT_USER_RPM", rendered)
        self.assertIn("OPENCLAW_CONNECTOR_RATE_LIMIT_CHANNEL_RPM", rendered)
        self.assertIn("OPENCLAW_CONNECTOR_MAX_COMMAND_LENGTH", rendered)


if __name__ == "__main__":
    unittest.main()
