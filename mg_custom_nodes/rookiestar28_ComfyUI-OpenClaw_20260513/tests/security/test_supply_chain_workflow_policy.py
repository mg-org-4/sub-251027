import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = REPO_ROOT / ".github" / "workflows"

FULL_SHA_RE = re.compile(r"^[a-f0-9]{40}$")
USES_RE = re.compile(r"^\s*uses:\s*([^@\s]+)@([^\s#]+)", re.MULTILINE)

VERSION_TAG_ACTIONS = {
    "actions/checkout",
    "actions/dependency-review-action",
    "actions/setup-node",
    "actions/setup-python",
    "actions/upload-artifact",
    "github/codeql-action/init",
    "github/codeql-action/analyze",
}


class TestSupplyChainWorkflowPolicy(unittest.TestCase):
    def test_workflows_do_not_use_privileged_pr_or_oidc_boundaries(self):
        for workflow_path in sorted(WORKFLOW_ROOT.glob("*.yml")):
            text = workflow_path.read_text(encoding="utf-8")
            with self.subTest(workflow=workflow_path.name):
                self.assertNotIn("pull_request_target:", text)
                self.assertNotIn("id-token: write", text)
                self.assertNotIn("uses: actions/cache@", text)

    def test_third_party_actions_use_immutable_sha_refs(self):
        for workflow_path in sorted(WORKFLOW_ROOT.glob("*.yml")):
            text = workflow_path.read_text(encoding="utf-8")
            for action, ref in USES_RE.findall(text):
                if action.startswith("./"):
                    continue
                with self.subTest(workflow=workflow_path.name, action=action):
                    if action in VERSION_TAG_ACTIONS:
                        self.assertRegex(ref, r"^v\d+(?:\.\d+){0,2}$")
                    else:
                        self.assertRegex(ref, FULL_SHA_RE)

    def test_publish_workflow_keeps_narrow_release_boundary(self):
        text = (WORKFLOW_ROOT / "publish.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("push:", text)
        self.assertNotIn("pull_request:", text)
        self.assertNotIn("pull_request_target:", text)
        self.assertNotIn("id-token: write", text)
        self.assertNotIn("uses: actions/cache@", text)
        self.assertRegex(
            text,
            re.compile(r"uses: Comfy-Org/publish-node-action@[a-f0-9]{40}\b"),
        )
        self.assertIn(
            "personal_access_token: ${{ secrets.REGISTRY_ACCESS_TOKEN }}", text
        )

    def test_dependency_review_gate_is_present_for_dependency_diffs(self):
        text = (WORKFLOW_ROOT / "dependency-review.yml").read_text(encoding="utf-8")
        self.assertIn("pull_request:", text)
        self.assertIn("package-lock.json", text)
        self.assertIn("requirements.txt", text)
        self.assertIn("pyproject.toml", text)
        self.assertIn("permissions:\n  contents: read\n  pull-requests: read\n", text)
        self.assertIn("uses: actions/dependency-review-action@v4", text)
        self.assertIn("fail-on-severity: high", text)
        self.assertNotIn("secrets.", text)


if __name__ == "__main__":
    unittest.main()
