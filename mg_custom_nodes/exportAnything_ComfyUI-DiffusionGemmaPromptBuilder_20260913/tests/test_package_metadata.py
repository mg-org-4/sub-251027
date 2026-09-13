from __future__ import annotations

from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PackageMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )

    def test_registry_identity_and_release_version_are_current(self) -> None:
        self.assertEqual(self.metadata["project"]["version"], "0.2.0")
        self.assertEqual(self.metadata["tool"]["comfy"]["PublisherId"], "exportanything")

    def test_registry_compatibility_starts_at_the_tested_baseline(self) -> None:
        self.assertEqual(
            self.metadata["tool"]["comfy"]["requires-comfyui"],
            ">=0.30.1",
        )

    def test_registry_archive_excludes_ci_and_large_demo_media(self) -> None:
        ignored = {
            line.strip()
            for line in (ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertTrue({".github/", "tests/", "*.mp4", "*.gif"} <= ignored)


if __name__ == "__main__":
    unittest.main()
