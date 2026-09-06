import json
import tempfile
import unittest
from pathlib import Path

from scripts import check_supply_chain_hardening as hardening


def write_lock(path: Path, packages: dict[str, dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"lockfileVersion": 3, "packages": packages}),
        encoding="utf-8",
    )


class TestSupplyChainHardening(unittest.TestCase):
    def test_clean_lockfile_has_no_findings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_lock(
                root / "package-lock.json",
                {
                    "": {"name": "example"},
                    "node_modules/@playwright/test": {"version": "1.0.0"},
                    "node_modules/esbuild": {
                        "version": "0.25.0",
                        "hasInstallScript": True,
                    },
                },
            )

            self.assertEqual(hardening.run_checks(root), [])

    def test_affected_npm_package_family_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_lock(
                root / "package-lock.json",
                {
                    "node_modules/@tanstack/router-core": {"version": "9.9.9"},
                },
            )

            findings = hardening.run_checks(root)

            self.assertEqual(findings[0].code, "mini-shai-hulud-npm-package")

    def test_unexpected_npm_install_script_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_lock(
                root / "package-lock.json",
                {
                    "node_modules/unreviewed-package": {
                        "version": "1.2.3",
                        "hasInstallScript": True,
                    },
                },
            )

            findings = hardening.run_checks(root)

            self.assertEqual(findings[0].code, "unexpected-npm-install-script")

    def test_affected_python_requirement_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "requirements.txt").write_text(
                "mistralai==2.4.6\n", encoding="utf-8"
            )

            findings = hardening.run_checks(root)

            self.assertEqual(findings[0].code, "mini-shai-hulud-pypi-package")

    def test_ioc_filename_is_reported_without_executing_dependency_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / "node_modules" / "package"
            package_dir.mkdir(parents=True)
            (package_dir / "router_init.js").write_text("void 0;\n", encoding="utf-8")

            findings = hardening.run_checks(root)

            self.assertEqual(findings[0].code, "mini-shai-hulud-ioc-file")


if __name__ == "__main__":
    unittest.main()
