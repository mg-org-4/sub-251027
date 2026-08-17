import unittest
from unittest.mock import patch

from services import effective_config


class TestR182EffectiveConfigHotspot(unittest.TestCase):
    @patch("services.effective_config.get_effective_config")
    def test_effective_provider_defaults_and_lowercases(self, mock_get_effective):
        mock_get_effective.return_value = ({"provider": "OpenAI"}, {})

        self.assertEqual(effective_config.get_effective_llm_provider(), "openai")

        mock_get_effective.return_value = ({}, {})
        self.assertEqual(
            effective_config.get_effective_llm_provider(),
            effective_config.DEFAULT_PROVIDER,
        )

    @patch("services.effective_config.get_api_key_for_provider")
    @patch(
        "services.effective_config.get_effective_llm_provider", return_value="openai"
    )
    def test_effective_api_key_uses_resolved_provider_and_tenant(
        self,
        mock_provider,
        mock_get_api_key,
    ):
        mock_get_api_key.return_value = "sk-test"

        self.assertEqual(
            effective_config.get_effective_llm_api_key(tenant_id="tenant-a"),
            "sk-test",
        )
        mock_get_api_key.assert_called_once_with("openai", tenant_id="tenant-a")


if __name__ == "__main__":
    unittest.main()
