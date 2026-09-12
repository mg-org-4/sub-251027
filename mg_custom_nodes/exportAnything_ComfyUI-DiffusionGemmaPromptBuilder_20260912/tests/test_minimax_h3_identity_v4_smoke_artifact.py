from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import unittest
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "create_minimax_h3_identity_v4_smoke.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_v4_smoke", TOOL_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot load smoke tool: {TOOL_PATH}")
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)

EXPECTED_SMOKE_SHA256 = (
    "19bd21f9e4b168aa3008fa6bf5ec1b8ce7b27e4e8c712f721333b2319f8fa746"
)
EXPECTED_SMOKE_UUID = "f71bf435-daf5-5b5f-9541-d4c7e8ce2228"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _node(workflow: dict, node_id: int) -> dict:
    matches = [item for item in workflow["nodes"] if int(item["id"]) == node_id]
    if len(matches) != 1:
        raise AssertionError(f"Expected one node {node_id}; found {len(matches)}")
    return matches[0]


@unittest.skipUnless(
    SMOKE.SOURCE_V4.is_file()
    and SMOKE.DEFAULT_DESKTOP_OUTPUT.is_file()
    and SMOKE.DEFAULT_USER_OUTPUT.is_file(),
    "verified V4 smoke artifacts are absent",
)
class MiniMaxH3IdentityV4SmokeArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_bytes = SMOKE.SOURCE_V4.read_bytes()
        cls.desktop_bytes = SMOKE.DEFAULT_DESKTOP_OUTPUT.read_bytes()
        cls.user_bytes = SMOKE.DEFAULT_USER_OUTPUT.read_bytes()
        cls.source = json.loads(cls.source_bytes.decode("utf-8"))
        cls.smoke = json.loads(cls.desktop_bytes.decode("utf-8"))

    def test_verified_v4_source_and_both_smoke_copies_have_exact_hashes(self) -> None:
        self.assertEqual(_sha256(SMOKE.SOURCE_V4), SMOKE.SOURCE_V4_SHA256)
        self.assertEqual(_sha256(SMOKE.DEFAULT_DESKTOP_OUTPUT), EXPECTED_SMOKE_SHA256)
        self.assertEqual(_sha256(SMOKE.DEFAULT_USER_OUTPUT), EXPECTED_SMOKE_SHA256)
        self.assertEqual(self.desktop_bytes, self.user_bytes)

    def test_smoke_has_an_independent_workflow_uuid(self) -> None:
        self.assertEqual(self.smoke["id"], EXPECTED_SMOKE_UUID)
        self.assertNotEqual(self.smoke["id"], self.source["id"])
        self.assertEqual(str(uuid.UUID(self.smoke["id"])), self.smoke["id"])
        self.assertEqual(self.smoke["revision"], self.source["revision"] + 1)

    def test_exact_four_smoke_widgets_and_relay_off(self) -> None:
        self.assertEqual(_node(self.smoke, 178)["widgets_values"][0], 5.0)
        target = _node(self.smoke, 673)["widgets_values"]
        self.assertEqual(target[4], "1")
        self.assertEqual(target[5], 15)
        self.assertEqual(target[11], 0)
        self.assertEqual(_node(self.smoke, 187)["widgets_values"][1], 0.2)
        self.assertEqual(
            _node(self.smoke, 651)["widgets_values"][0],
            "minimax_h3/smoke_dual_identity_v4_5s",
        )
        self.assertEqual(_node(self.smoke, 677)["widgets_values"][-1], "Off")

    def test_topology_and_all_non_smoke_node_fields_match_v4(self) -> None:
        SMOKE.validate_smoke_workflow(self.smoke, source=self.source)
        self.assertEqual(self.smoke["links"], self.source["links"])
        self.assertEqual(self.smoke["definitions"], self.source["definitions"])
        self.assertEqual(len(self.smoke["nodes"]), len(self.source["nodes"]))

    def test_root_picture_2_and_provenance_remain_guarded(self) -> None:
        self.assertEqual(
            _node(self.smoke, 721)["widgets_values"][0],
            SMOKE.BASE.SECONDARY_IDENTITY_RELATIVE_PATH,
        )
        self.assertEqual(
            Path(SMOKE.BASE.SECONDARY_IDENTITY_RELATIVE_PATH).parent, Path(".")
        )
        asset = (
            SMOKE.BASE.DEFAULT_COMFY_INPUT_ROOT
            / SMOKE.BASE.SECONDARY_IDENTITY_RELATIVE_PATH
        )
        self.assertEqual(_sha256(asset), SMOKE.BASE.SECONDARY_IDENTITY_SHA256)
        marker = self.smoke["extra"][SMOKE.SMOKE_SCHEMA]
        self.assertEqual(marker["source_workflow_file_sha256"], SMOKE.SOURCE_V4_SHA256)
        self.assertEqual(marker["smoke_workflow_id"], EXPECTED_SMOKE_UUID)
        self.assertTrue(marker["source_is_never_overwritten"])
        self.assertTrue(marker["generation_was_not_queued_by_creator"])

    def test_validator_fails_closed_on_a_fifth_execution_edit(self) -> None:
        broken = copy.deepcopy(self.smoke)
        _node(broken, 713)["widgets_values"][1] = 0.5
        with self.assertRaises(SMOKE.BASE.WorkflowError):
            SMOKE.validate_smoke_workflow(broken, source=self.source)


if __name__ == "__main__":
    unittest.main()
