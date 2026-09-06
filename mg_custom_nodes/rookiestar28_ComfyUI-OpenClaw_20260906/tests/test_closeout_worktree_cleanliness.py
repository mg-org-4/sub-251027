import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_FULL_GATE = ROOT / "scripts" / "run_full_tests_windows.ps1"
LINUX_FULL_GATE = ROOT / "scripts" / "run_full_tests_linux.sh"
PRE_PUSH_GATE = ROOT / "scripts" / "pre_push_checks.sh"


def _assert_ordered(test: unittest.TestCase, content: str, *needles: str) -> None:
    positions = [content.index(needle) for needle in needles]
    test.assertEqual(positions, sorted(positions))


class CloseoutWorktreeCleanlinessContractTests(unittest.TestCase):
    def test_windows_full_gate_checks_all_public_state_before_and_after_validation(
        self,
    ):
        content = WINDOWS_FULL_GATE.read_text(encoding="utf-8")

        self.assertIn("function Assert-CleanPublicWorktree", content)
        self.assertIn("& git status --porcelain --untracked-files=all", content)
        self.assertEqual(content.splitlines().count("Assert-CleanPublicWorktree"), 2)
        _assert_ordered(
            self,
            content,
            "\nAssert-CleanPublicWorktree\n",
            'Invoke-Checked "npm ci" { npm ci }',
            '\nAssert-CleanPublicWorktree\nWrite-Host "[tests] PASS"',
        )

    def test_bash_full_gates_check_all_public_state_before_and_after_validation(self):
        for path, label in (
            (LINUX_FULL_GATE, "tests"),
            (PRE_PUSH_GATE, "pre-push"),
        ):
            with self.subTest(path=path.name):
                content = path.read_text(encoding="utf-8")

                self.assertIn("assert_clean_public_worktree()", content)
                self.assertIn("git status --porcelain --untracked-files=all", content)
                self.assertEqual(
                    content.splitlines().count("assert_clean_public_worktree"), 2
                )
                _assert_ordered(
                    self,
                    content,
                    "\nassert_clean_public_worktree\n",
                    "npm ci",
                    f'\nassert_clean_public_worktree\necho "[{label}] PASS"',
                )


if __name__ == "__main__":
    unittest.main()
