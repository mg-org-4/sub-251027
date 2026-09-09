from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestGitignoreTrackingBoundary(unittest.TestCase):
    def test_no_tracked_path_is_ignored(self):
        result = subprocess.run(
            ["git", "ls-files", "-ci", "--exclude-standard"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        ignored_tracked_paths = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
        self.assertEqual(ignored_tracked_paths, [])


if __name__ == "__main__":
    unittest.main()
