from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


class ContractDigestPortabilityTests(unittest.TestCase):
    def test_text_digest_is_newline_invariant_but_content_sensitive(self) -> None:
        spec = importlib.util.find_spec("scripts.contract_digest")
        self.assertIsNotNone(spec, "shared contract digest helper must exist")
        from scripts.contract_digest import stable_text_digest

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            variants = {
                "lf.txt": b"alpha\nbeta\n",
                "crlf.txt": b"alpha\r\nbeta\r\n",
                "cr.txt": b"alpha\rbeta\r",
            }
            for name, payload in variants.items():
                (root / name).write_bytes(payload)
            (root / "changed.txt").write_bytes(b"alpha\ngamma\n")

            normalized = {stable_text_digest(root / name) for name in variants}
            self.assertEqual(len(normalized), 1)
            self.assertNotEqual(
                stable_text_digest(root / "lf.txt"),
                stable_text_digest(root / "changed.txt"),
            )

    def test_text_writer_emits_utf8_lf_bytes(self) -> None:
        spec = importlib.util.find_spec("scripts.contract_digest")
        self.assertIsNotNone(spec, "shared contract digest helper must exist")
        from scripts.contract_digest import write_text_lf

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "contract.json"
            write_text_lf(target, '{\n  "label": "測試"\n}\n')
            payload = target.read_bytes()
            self.assertNotIn(b"\r", payload)
            self.assertEqual(payload.decode("utf-8"), '{\n  "label": "測試"\n}\n')


if __name__ == "__main__":
    unittest.main()
