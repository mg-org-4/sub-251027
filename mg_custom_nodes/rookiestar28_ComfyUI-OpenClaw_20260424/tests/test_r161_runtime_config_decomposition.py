import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import runtime_config


class TestR161RuntimeConfigDecomposition(unittest.TestCase):
    def test_persistence_wrappers_keep_config_file_patch_seam(self):
        with tempfile.TemporaryDirectory(prefix="r161_cfg_") as tmpdir:
            cfg_path = Path(tmpdir) / "config.json"
            payload = {
                "llm": {
                    "provider": "openai",
                    "runtime_guardrails": {"should": "strip"},
                }
            }

            with patch("services.runtime_config.CONFIG_FILE", str(cfg_path)):
                self.assertTrue(runtime_config._save_file_config(payload))
                loaded = runtime_config._load_file_config()

            self.assertEqual(loaded["llm"]["provider"], "openai")
            self.assertNotIn("runtime_guardrails", loaded["llm"])
            persisted = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertNotIn("runtime_guardrails", persisted["llm"])

    def test_validate_config_update_uses_runtime_config_validate_url_seam(self):
        with patch.dict(
            "os.environ",
            {"OPENCLAW_ALLOW_CUSTOM_BASE_URL": "1"},
            clear=False,
        ):
            with patch(
                "services.runtime_config.validate_outbound_url"
            ) as mock_validate:
                sanitized, errors = runtime_config.validate_config_update(
                    {"base_url": "https://api.example.com"}
                )

        self.assertEqual(errors, [])
        self.assertEqual(sanitized["base_url"], "https://api.example.com")
        mock_validate.assert_called_once()

    def test_get_config_projection_uses_facade_dependencies(self):
        with patch(
            "services.runtime_config.get_effective_config",
            return_value=({"provider": "anthropic"}, {"provider": "runtime_override"}),
        ):
            with patch(
                "services.runtime_config.get_admin_token", return_value="secret"
            ):
                cfg = runtime_config.get_config()

        self.assertEqual(cfg.llm["provider"], "anthropic")
        self.assertTrue(cfg.admin_token_configured)


if __name__ == "__main__":
    unittest.main()
