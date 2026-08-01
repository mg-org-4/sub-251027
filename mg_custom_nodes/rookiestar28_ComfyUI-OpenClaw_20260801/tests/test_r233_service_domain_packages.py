"""Contract tests for bootstrap/posture implementation package ownership."""

from __future__ import annotations

import ast
import importlib
import importlib.machinery
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]

FACADE_PAIRS = (
    ("services.startup_lifecycle", "services.bootstrap.lifecycle"),
    ("services.route_bootstrap", "services.bootstrap.registration"),
    ("services.effective_security_posture", "services.posture.effective"),
)


class ServiceDomainPackageContractTests(unittest.TestCase):
    def test_old_and_new_paths_resolve_to_the_same_module_objects(self):
        for facade_name, implementation_name in FACADE_PAIRS:
            with self.subTest(facade=facade_name):
                facade = importlib.import_module(facade_name)
                implementation = importlib.import_module(implementation_name)

                self.assertIs(facade, implementation)
                self.assertIs(sys.modules[facade_name], implementation)

    def test_file_loader_package_namespace_preserves_module_identity(self):
        package_name = "r233_comfyui_loader_probe"
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT)]
        package.__package__ = package_name
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name,
            loader=None,
            is_package=True,
        )
        sys.modules[package_name] = package
        try:
            for facade_name, implementation_name in FACADE_PAIRS:
                qualified_facade = f"{package_name}.{facade_name}"
                qualified_implementation = f"{package_name}.{implementation_name}"
                with self.subTest(facade=qualified_facade):
                    facade = importlib.import_module(qualified_facade)
                    implementation = importlib.import_module(qualified_implementation)
                    self.assertIs(facade, implementation)
                    self.assertIs(sys.modules[qualified_facade], implementation)

            posture = importlib.import_module(
                f"{package_name}.services.posture.effective"
            )
            snapshot = posture.resolve_effective_security_posture({})
            self.assertEqual(snapshot.deployment_profile, "local")
        finally:
            for module_name in tuple(sys.modules):
                if module_name == package_name or module_name.startswith(
                    f"{package_name}."
                ):
                    sys.modules.pop(module_name, None)

    def test_lifecycle_and_posture_singletons_are_not_duplicated(self):
        legacy_lifecycle = importlib.import_module("services.startup_lifecycle")
        owned_lifecycle = importlib.import_module("services.bootstrap.lifecycle")
        legacy_posture = importlib.import_module("services.effective_security_posture")
        owned_posture = importlib.import_module("services.posture.effective")

        self.assertIs(legacy_lifecycle._LIFECYCLE, owned_lifecycle._LIFECYCLE)
        self.assertIs(legacy_posture._posture_lock, owned_posture._posture_lock)

        owned_posture.reset_effective_security_posture_for_tests()
        try:
            installed = legacy_posture.get_or_create_effective_security_posture({})
            self.assertIs(owned_posture.get_effective_security_posture(), installed)
        finally:
            legacy_posture.reset_effective_security_posture_for_tests()

    def test_old_path_patches_mutate_the_owned_modules(self):
        owned_lifecycle = importlib.import_module("services.bootstrap.lifecycle")
        owned_registration = importlib.import_module("services.bootstrap.registration")
        owned_posture = importlib.import_module("services.posture.effective")

        lifecycle_thread = MagicMock()
        with patch(
            "services.startup_lifecycle.threading.Thread",
            lifecycle_thread,
        ):
            self.assertIs(owned_lifecycle.threading.Thread, lifecycle_thread)

        registration_step = MagicMock()
        with patch(
            "services.route_bootstrap._do_full_registration",
            registration_step,
        ):
            self.assertIs(
                owned_registration._do_full_registration,
                registration_step,
            )

        deployment_report = MagicMock()
        with patch(
            "services.effective_security_posture._deployment_report",
            deployment_report,
        ):
            self.assertIs(owned_posture._deployment_report, deployment_report)

    def test_compatibility_facades_are_bounded_module_aliases(self):
        expected_imports = {
            "startup_lifecycle.py": "from .bootstrap import lifecycle",
            "route_bootstrap.py": "from .bootstrap import registration",
            "effective_security_posture.py": "from .posture import effective",
        }
        allowed_statement_types = {
            ast.Expr,
            ast.Import,
            ast.ImportFrom,
            ast.If,
            ast.Assign,
        }

        for filename, implementation_suffix in expected_imports.items():
            with self.subTest(filename=filename):
                path = ROOT / "services" / filename
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                self.assertTrue(
                    all(
                        type(statement) in allowed_statement_types
                        for statement in tree.body
                    )
                )
                imports = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                ]
                rendered = "\n".join(ast.unparse(node) for node in imports)
                self.assertIn(implementation_suffix, rendered)
                self.assertIn("sys", rendered)
                self.assertLessEqual(len(tree.body), 7)
                type_only_blocks = [
                    statement
                    for statement in tree.body
                    if isinstance(statement, ast.If)
                ]
                self.assertEqual(len(type_only_blocks), 1)
                self.assertIsInstance(type_only_blocks[0].test, ast.Name)
                self.assertEqual(type_only_blocks[0].test.id, "TYPE_CHECKING")

    def test_package_initializers_are_navigation_only(self):
        for relative_path in (
            "services/bootstrap/__init__.py",
            "services/posture/__init__.py",
        ):
            with self.subTest(path=relative_path):
                path = ROOT / relative_path
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
                self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
