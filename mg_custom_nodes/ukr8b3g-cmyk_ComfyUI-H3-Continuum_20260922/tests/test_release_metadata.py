from __future__ import annotations

import configparser
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "3.8.3"


def test_v35_release_metadata_is_consistent():
    version_source = (ROOT / "version.py").read_text(encoding="utf-8")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    metadata = configparser.ConfigParser()
    metadata.read(ROOT / "metadata.ini", encoding="utf-8")

    assert f'PACKAGE_VERSION = "{EXPECTED_VERSION}"' in version_source
    assert pyproject["project"]["version"] == EXPECTED_VERSION
    assert metadata["project"]["version"] == EXPECTED_VERSION


def test_public_documents_identify_the_current_v38x2_release():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_ja = (ROOT / "README_JA.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    expected_heading = f"# ComfyUI-H3-Continuum {EXPECTED_VERSION} — V3.8X2\n"
    assert readme.startswith(expected_heading)
    assert readme_ja.startswith(expected_heading)
    assert f"## {EXPECTED_VERSION}" in changelog
