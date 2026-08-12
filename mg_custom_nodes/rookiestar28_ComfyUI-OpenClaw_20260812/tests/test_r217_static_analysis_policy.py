import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from scripts import verify_static_analysis_policy as policy_module


class TestStaticAnalysisPolicy(unittest.TestCase):
    def _policy(self) -> dict:
        return {
            "schema_version": 1,
            "review": {
                "owner": "maintainers",
                "reviewed_at": "2026-07-11",
                "next_review_by": "2026-10-11",
            },
            "tools": {
                "ruff": {"version": "0.15.20"},
                "mypy": {"version": "2.2.0"},
            },
            "production_roots": ["config.py", "pkg"],
            "excluded_paths": [
                {
                    "path": "pkg/generated.py",
                    "reason": "generated fixture excluded from source ownership",
                }
            ],
            "strict_paths": ["pkg/clean.py"],
            "baseline": [
                {
                    "tool": "ruff",
                    "path": "pkg/owned.py",
                    "code": "F401",
                    "message": "unused import",
                    "count": 1,
                }
            ],
        }

    def _create_repo(self, root: Path) -> None:
        (root / "config.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "pkg" / "nested").mkdir(parents=True)
        (root / "pkg" / "owned.py").write_text("import os\n", encoding="utf-8")
        (root / "pkg" / "clean.py").write_text("VALUE: int = 1\n", encoding="utf-8")
        (root / "pkg" / "generated.py").write_text(
            "generated = True\n", encoding="utf-8"
        )
        (root / "pkg" / "nested" / "child.py").write_text(
            "CHILD = True\n", encoding="utf-8"
        )

    def test_policy_validation_rejects_missing_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_repo(root)
            policy = self._policy()
            policy["production_roots"].extend(["missing", "../outside"])
            policy["strict_paths"].append("pkg/missing.py")
            policy["excluded_paths"].append(
                {"path": "reference/untrusted.py", "reason": "not inside an owned root"}
            )

            failures = policy_module.validate_policy(root, policy)

        self.assertTrue(any("missing" in failure for failure in failures))
        self.assertTrue(any("unsafe" in failure for failure in failures))
        self.assertTrue(any("strict path" in failure for failure in failures))
        self.assertTrue(any("excluded path" in failure for failure in failures))

    def test_owned_file_discovery_is_recursive_and_excludes_only_declared_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_repo(root)

            files = policy_module.discover_owned_python_files(root, self._policy())

        self.assertEqual(
            files,
            (
                "config.py",
                "pkg/clean.py",
                "pkg/nested/child.py",
                "pkg/owned.py",
            ),
        )

    def test_git_worktree_discovery_excludes_ignored_and_untracked_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_repo(root)
            (root / ".gitignore").write_text("pkg/local_ignored.py\n", encoding="utf-8")
            (root / "pkg" / "local_ignored.py").write_text(
                "IGNORED = True\n", encoding="utf-8"
            )
            (root / "pkg" / "local_untracked.py").write_text(
                "UNTRACKED = True\n", encoding="utf-8"
            )
            subprocess.run(
                ["git", "init", "--quiet"], cwd=root, check=True, capture_output=True
            )
            subprocess.run(
                [
                    "git",
                    "add",
                    ".gitignore",
                    "config.py",
                    "pkg/owned.py",
                    "pkg/clean.py",
                    "pkg/generated.py",
                    "pkg/nested/child.py",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )

            files = policy_module.discover_owned_python_files(root, self._policy())

        self.assertEqual(
            files,
            (
                "config.py",
                "pkg/clean.py",
                "pkg/nested/child.py",
                "pkg/owned.py",
            ),
        )

    def test_exact_baseline_passes(self):
        policy = self._policy()
        current = Counter(
            {
                policy_module.Diagnostic(
                    tool="ruff",
                    path="pkg/owned.py",
                    code="F401",
                    message="unused import",
                ): 1
            }
        )

        self.assertEqual(policy_module.compare_diagnostics(policy, current), [])

    def test_new_and_resolved_debt_both_fail_the_ratchet(self):
        policy = self._policy()
        key = policy_module.Diagnostic(
            tool="ruff",
            path="pkg/owned.py",
            code="F401",
            message="unused import",
        )

        increased = policy_module.compare_diagnostics(policy, Counter({key: 2}))
        resolved = policy_module.compare_diagnostics(policy, Counter())

        self.assertTrue(any("new debt" in failure for failure in increased))
        self.assertTrue(any("stale baseline" in failure for failure in resolved))

    def test_strict_path_rejects_even_baselined_diagnostic(self):
        policy = self._policy()
        policy["baseline"].append(
            {
                "tool": "mypy",
                "path": "pkg/clean.py",
                "code": "assignment",
                "message": "incompatible assignment",
                "count": 1,
            }
        )
        current = Counter(
            {
                policy_module.Diagnostic(
                    tool="ruff",
                    path="pkg/owned.py",
                    code="F401",
                    message="unused import",
                ): 1,
                policy_module.Diagnostic(
                    tool="mypy",
                    path="pkg/clean.py",
                    code="assignment",
                    message="incompatible assignment",
                ): 1,
            }
        )

        failures = policy_module.compare_diagnostics(policy, current)

        self.assertTrue(any("strict path" in failure for failure in failures))

    def test_tool_version_drift_is_rejected(self):
        failures = policy_module.validate_tool_versions(
            self._policy(), {"ruff": "0.15.19", "mypy": "2.2.0"}
        )

        self.assertEqual(
            failures,
            ["ruff version drift: expected 0.15.20, found 0.15.19"],
        )

    def test_policy_validation_rejects_noncanonical_baseline_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_repo(root)
            policy = self._policy()
            policy["baseline"] = [
                {
                    "tool": "ruff",
                    "path": "pkg/owned.py",
                    "code": "F401",
                    "message": "unused import",
                    "count": 1,
                },
                {
                    "tool": "mypy",
                    "path": "pkg/owned.py",
                    "code": "assignment",
                    "message": "bad assignment",
                    "count": 1,
                },
            ]

            failures = policy_module.validate_policy(root, policy)

        self.assertIn("baseline must use canonical sorted serialization", failures)

    def test_ruff_json_is_normalized_to_repo_relative_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "pkg" / "owned.py"
            raw = json.dumps(
                [
                    {
                        "filename": str(source),
                        "code": "F401",
                        "message": "  unused   import  ",
                        "location": {"row": 1, "column": 1},
                    }
                ]
            )

            result = policy_module.parse_ruff_output(raw, root)

        self.assertEqual(
            result,
            Counter(
                {
                    policy_module.Diagnostic(
                        tool="ruff",
                        path="pkg/owned.py",
                        code="F401",
                        message="unused import",
                    ): 1
                }
            ),
        )

    def test_mypy_json_lines_are_normalized_without_line_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "pkg" / "owned.py"
            raw = json.dumps(
                {
                    "file": str(source),
                    "line": 4,
                    "column": 2,
                    "message": " Name  is not defined ",
                    "hint": None,
                    "code": "name-defined",
                    "severity": "error",
                }
            )

            result = policy_module.parse_mypy_output(raw, root)

        self.assertEqual(
            result,
            Counter(
                {
                    policy_module.Diagnostic(
                        tool="mypy",
                        path="pkg/owned.py",
                        code="name-defined",
                        message="Name is not defined",
                    ): 1
                }
            ),
        )

    def test_serialized_baseline_is_deterministic_and_sorted(self):
        diagnostics = Counter(
            {
                policy_module.Diagnostic("ruff", "z.py", "F401", "unused"): 2,
                policy_module.Diagnostic("mypy", "a.py", "assignment", "bad"): 1,
            }
        )

        self.assertEqual(
            policy_module.serialize_baseline(diagnostics),
            [
                {
                    "tool": "mypy",
                    "path": "a.py",
                    "code": "assignment",
                    "message": "bad",
                    "count": 1,
                },
                {
                    "tool": "ruff",
                    "path": "z.py",
                    "code": "F401",
                    "message": "unused",
                    "count": 2,
                },
            ],
        )

    def test_quality_requirement_pins_must_match_policy(self):
        policy = self._policy()

        self.assertEqual(
            policy_module.validate_requirement_pins(
                policy, ["ruff==0.15.20", "mypy==2.2.0"]
            ),
            [],
        )
        self.assertEqual(
            policy_module.validate_requirement_pins(
                policy, ["ruff>=0.15.20", "mypy==2.1.0"]
            ),
            [
                "mypy requirement drift: expected mypy==2.2.0, found mypy==2.1.0",
                "ruff requirement drift: expected ruff==0.15.20, found ruff>=0.15.20",
            ],
        )

    def test_tool_runner_uses_current_interpreter_and_combines_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_repo(root)
            commands: list[tuple[str, ...]] = []

            def runner(command, **kwargs):
                self.assertEqual(kwargs["cwd"], root)
                self.assertFalse(kwargs.get("shell", False))
                commands.append(tuple(command))
                if command[-1] == "--version":
                    name = command[2]
                    output = (
                        "ruff 0.15.20\n"
                        if name == "ruff"
                        else "mypy 2.2.0 (compiled: yes)\n"
                    )
                    return SimpleNamespace(returncode=0, stdout=output, stderr="")
                if command[2] == "ruff":
                    return SimpleNamespace(
                        returncode=1,
                        stdout=json.dumps(
                            [
                                {
                                    "filename": str(root / "pkg" / "owned.py"),
                                    "code": "F401",
                                    "message": "unused import",
                                }
                            ]
                        ),
                        stderr="",
                    )
                return SimpleNamespace(
                    returncode=1,
                    stdout=json.dumps(
                        {
                            "file": str(root / "pkg" / "nested" / "child.py"),
                            "line": 1,
                            "column": 1,
                            "message": "missing annotation",
                            "code": "no-untyped-def",
                            "severity": "error",
                        }
                    ),
                    stderr="",
                )

            versions, diagnostics = policy_module.run_static_analysis(
                root, self._policy(), runner=runner
            )

        self.assertEqual(versions, {"ruff": "0.15.20", "mypy": "2.2.0"})
        self.assertEqual(sum(diagnostics.values()), 2)
        self.assertEqual(len(commands), 4)
        for command in commands:
            self.assertEqual(command[:2], (sys.executable, "-m"))
            self.assertIn(command[2], {"ruff", "mypy"})
        ruff_command = next(command for command in commands if "check" in command)
        mypy_command = next(command for command in commands if "--output" in command)
        self.assertIn("--no-cache", ruff_command)
        self.assertIn("json", ruff_command)
        self.assertIn("--no-incremental", mypy_command)
        self.assertIn("--explicit-package-bases", mypy_command)
        self.assertIn("--no-warn-unused-configs", mypy_command)
        self.assertIn("--no-error-summary", mypy_command)
        self.assertIn("--no-site-packages", mypy_command)
        self.assertIn("--ignore-missing-imports", mypy_command)
        self.assertIn("json", mypy_command)

    def test_tool_runner_rejects_abnormal_tool_failure_without_echoing_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_repo(root)

            def runner(command, **_kwargs):
                if command[-1] == "--version":
                    name = command[2]
                    return SimpleNamespace(
                        returncode=0,
                        stdout=(
                            f"{name} 0.15.20\n" if name == "ruff" else "mypy 2.2.0\n"
                        ),
                        stderr="",
                    )
                return SimpleNamespace(
                    returncode=2,
                    stdout="",
                    stderr="private C:\\Users\\name\\secret tool failure",
                )

            with self.assertRaises(policy_module.ToolExecutionError) as ctx:
                policy_module.run_static_analysis(root, self._policy(), runner=runner)

        self.assertEqual(str(ctx.exception), "ruff execution failed with exit code 2")
        self.assertNotIn("private", str(ctx.exception))

    def test_updated_policy_baseline_does_not_mutate_input(self):
        policy = self._policy()
        diagnostics = Counter(
            {policy_module.Diagnostic("mypy", "pkg/owned.py", "assignment", "bad"): 1}
        )

        updated = policy_module.with_updated_baseline(policy, diagnostics)

        self.assertNotEqual(updated["baseline"], policy["baseline"])
        self.assertEqual(policy["baseline"][0]["tool"], "ruff")
        self.assertEqual(updated["baseline"][0]["tool"], "mypy")

    def test_evaluate_policy_reports_requirement_and_installed_version_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_repo(root)
            policy = self._policy()
            policy["baseline"] = []

            def runner(command, **_kwargs):
                if command[-1] == "--version":
                    return SimpleNamespace(
                        returncode=0,
                        stdout=(
                            "ruff 0.15.19\n" if command[2] == "ruff" else "mypy 2.2.0\n"
                        ),
                        stderr="",
                    )
                return SimpleNamespace(
                    returncode=0,
                    stdout="[]" if command[2] == "ruff" else "",
                    stderr="",
                )

            failures, diagnostics = policy_module.evaluate_policy(
                root,
                policy,
                requirement_lines=["ruff>=0.15.20", "mypy==2.2.0"],
                runner=runner,
            )

        self.assertEqual(diagnostics, Counter())
        self.assertIn(
            "ruff requirement drift: expected ruff==0.15.20, found ruff>=0.15.20",
            failures,
        )
        self.assertIn("ruff version drift: expected 0.15.20, found 0.15.19", failures)


class TestRepositoryStaticAnalysisPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.policy_path = cls.repo_root / "tests" / "static_analysis_policy.json"
        cls.requirements_path = cls.repo_root / "requirements-quality.txt"

    def test_repository_policy_and_quality_pins_are_valid(self):
        policy = json.loads(self.policy_path.read_text(encoding="utf-8"))
        requirement_lines = self.requirements_path.read_text(
            encoding="utf-8"
        ).splitlines()

        self.assertEqual(policy_module.validate_policy(self.repo_root, policy), [])
        self.assertEqual(
            policy_module.validate_requirement_pins(policy, requirement_lines), []
        )
        self.assertEqual(
            policy["production_roots"],
            [
                "__init__.py",
                "config.py",
                "api",
                "connector",
                "models",
                "nodes",
                "services",
                "scripts",
            ],
        )
        self.assertGreater(
            len(policy_module.discover_owned_python_files(self.repo_root, policy)), 200
        )
        self.assertGreaterEqual(len(policy["strict_paths"]), 1)

    def test_repository_execution_surfaces_use_shared_verifier(self):
        expected = "scripts/verify_static_analysis_policy.py"
        surfaces = {
            ".pre-commit-config.yaml": self.repo_root / ".pre-commit-config.yaml",
            "windows full gate": self.repo_root
            / "scripts"
            / "run_full_tests_windows.ps1",
            "linux full gate": self.repo_root / "scripts" / "run_full_tests_linux.sh",
            "pre-commit CI": self.repo_root
            / ".github"
            / "workflows"
            / "pre-commit.yml",
            "unit CI": self.repo_root / ".github" / "workflows" / "ci.yml",
        }
        for label, path in surfaces.items():
            with self.subTest(surface=label):
                content = path.read_text(encoding="utf-8")
                self.assertIn(expected, content)
                self.assertIn("requirements-quality.txt", content)

    def test_precommit_hook_has_an_isolated_pinned_tool_environment(self):
        content = (self.repo_root / ".pre-commit-config.yaml").read_text(
            encoding="utf-8"
        )
        hook = content.split("- id: static-analysis-policy", 1)[1].split(
            "# Secret detection", 1
        )[0]

        self.assertIn("language: python", hook)
        self.assertIn("ruff==0.15.20", hook)
        self.assertIn("mypy==2.2.0", hook)

    def test_quality_tools_are_not_runtime_requirements(self):
        runtime_requirements = (self.repo_root / "requirements.txt").read_text(
            encoding="utf-8"
        )
        project_config = (self.repo_root / "pyproject.toml").read_text(encoding="utf-8")

        self.assertNotIn("ruff", runtime_requirements.lower())
        self.assertNotIn("mypy", runtime_requirements.lower())
        dependencies_block = project_config.split("[project.urls]", 1)[0]
        self.assertNotIn("ruff", dependencies_block.lower())
        self.assertNotIn("mypy", dependencies_block.lower())
        self.assertIn("explicit_package_bases = true", project_config)
        self.assertIn("no_site_packages = true", project_config)
        self.assertIn("ignore_missing_imports = true", project_config)


if __name__ == "__main__":
    unittest.main()
