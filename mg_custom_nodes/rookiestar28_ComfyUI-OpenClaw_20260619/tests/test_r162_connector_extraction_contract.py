import unittest
from pathlib import Path

from services.connector_extraction_contract import get_connector_extraction_contract

ROOT = Path(__file__).resolve().parents[1]
ADR_PATH = (
    ROOT / "docs" / "adr" / "ADR-0003-connector-extraction-feasibility-and-seams.md"
)
CONNECTOR_DOC_PATH = ROOT / "docs" / "connector.md"


class TestR162ConnectorExtractionContract(unittest.TestCase):
    def test_recommendation_and_options_are_stable(self):
        contract = get_connector_extraction_contract()

        self.assertEqual(contract["decision"]["id"], "stay_in_repo_attached_subsystem")
        self.assertEqual(contract["decision"]["go_no_go"], "no_go_for_split_now")
        self.assertEqual(
            contract["decision"]["future_candidate"],
            "optional_extra_package_after_shared_contract_extraction",
        )
        self.assertEqual(
            [option["id"] for option in contract["candidate_packaging_options"]],
            [
                "stay_in_repo_attached_subsystem",
                "optional_extra_package_after_shared_contract_extraction",
                "sidecar_only_distribution",
                "separate_repo_or_primary_connector_package",
            ],
        )

    def test_seam_entrypoints_exist(self):
        contract = get_connector_extraction_contract()

        for seam in contract["minimum_stable_seam_families"]:
            for rel_path in seam["entrypoints"]:
                self.assertTrue((ROOT / rel_path).exists(), rel_path)

    def test_docs_align_with_decision_terms(self):
        adr_text = ADR_PATH.read_text(encoding="utf-8")
        connector_text = CONNECTOR_DOC_PATH.read_text(encoding="utf-8")

        for phrase in (
            "optional attached subsystem",
            "no-go-for-split-now",
            "optional extra package",
        ):
            self.assertIn(phrase, adr_text)

        self.assertIn("in-repo", connector_text)
        self.assertIn("ADR-0003", connector_text)


if __name__ == "__main__":
    unittest.main()
