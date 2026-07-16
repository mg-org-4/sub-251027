"""Security and compatibility contract for file-backed text outputs."""

import unittest


class TestR216FileBackedTextOutput(unittest.TestCase):
    @staticmethod
    def _history(files):
        return {
            "outputs": {
                "9": {
                    "files": files,
                    "text": "some generated text",
                }
            }
        }

    def test_official_result_txt_fixture_is_normalized_as_file_backed_text(self):
        from services.comfyui_history import extract_output_refs

        outputs = extract_output_refs(
            self._history(
                [
                    {
                        "filename": "result.txt",
                        "subfolder": "",
                        "type": "output",
                    }
                ]
            )
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["filename"], "result.txt")
        self.assertEqual(outputs[0]["media_type"], "text")
        self.assertEqual(outputs[0]["resolution"], "view")
        self.assertEqual(outputs[0]["content"], "")
        self.assertIn("/view?", outputs[0]["view_url"])
        self.assertIn("filename=result.txt", outputs[0]["view_url"])

    def test_frozen_text_suffix_allowlist_is_case_insensitive(self):
        from services.comfyui_history import extract_output_refs

        suffixes = ("txt", "md", "markdown", "json", "csv", "yaml", "yml", "xml", "log")
        files = [
            {"filename": f"result.{suffix.upper()}", "type": "output"}
            for suffix in suffixes
        ]

        outputs = extract_output_refs(self._history(files))

        self.assertEqual(len(outputs), len(suffixes))
        self.assertTrue(all(output["media_type"] == "text" for output in outputs))

    def test_unknown_binary_and_non_mapping_file_refs_are_omitted(self):
        from services.comfyui_history import extract_output_refs

        outputs = extract_output_refs(
            self._history(
                [
                    {"filename": "image.png", "type": "output"},
                    {"filename": "archive.bin", "type": "output"},
                    {"filename": "README", "type": "output"},
                    "result.txt",
                    None,
                    {"filename": "safe.txt", "type": "output"},
                ]
            )
        )

        self.assertEqual([output["filename"] for output in outputs], ["safe.txt"])
        self.assertNotEqual(outputs[0]["media_type"], "images")

    def test_files_container_and_field_bounds_fail_closed(self):
        from services.comfyui_history import extract_output_refs

        self.assertEqual(extract_output_refs(self._history("result.txt")), [])
        self.assertEqual(
            extract_output_refs(
                self._history(
                    [
                        {"filename": f"result-{index}.txt", "type": "output"}
                        for index in range(65)
                    ]
                )
            ),
            [],
        )

        invalid_refs = [
            {"filename": "x" * 1021 + ".txt", "type": "output"},
            {"filename": "result.txt", "subfolder": "x" * 1025, "type": "output"},
            {"filename": "../result.txt", "type": "output"},
            {"filename": "folder/result.txt", "type": "output"},
            {"filename": "result.txt", "subfolder": "../private", "type": "output"},
            {"filename": "result.txt", "subfolder": "/absolute", "type": "output"},
            {"filename": "result.txt", "type": "unknown"},
            {"filename": 123, "type": "output"},
        ]
        for ref in invalid_refs:
            with self.subTest(ref=ref):
                self.assertEqual(extract_output_refs(self._history([ref])), [])

    def test_view_url_uses_only_normalized_fields_and_ignores_raw_url(self):
        from services.comfyui_history import extract_output_refs

        outputs = extract_output_refs(
            self._history(
                [
                    {
                        "filename": "report 1.txt",
                        "subfolder": "reports/2026",
                        "type": "temp",
                        "url": "https://evil.example/secret.txt",
                    }
                ]
            )
        )

        self.assertEqual(len(outputs), 1)
        self.assertIn("filename=report+1.txt", outputs[0]["view_url"])
        self.assertIn("subfolder=reports%2F2026", outputs[0]["view_url"])
        self.assertIn("type=temp", outputs[0]["view_url"])
        self.assertNotIn("evil.example", outputs[0]["view_url"])

    def test_field_bounds_count_unicode_code_points_and_reject_trim_bypass(self):
        from services.comfyui_history import extract_output_refs

        accepted = extract_output_refs(
            self._history([{"filename": "😀" * 1020 + ".txt", "type": "output"}])
        )
        self.assertEqual(len(accepted), 1)

        rejected = [
            {"filename": "😀" * 1021 + ".txt", "type": "output"},
            {"filename": " " * 1025 + "safe.txt", "type": "output"},
            {
                "filename": "safe.txt",
                "subfolder": " " * 1025 + "reports",
                "type": "output",
            },
        ]
        for ref in rejected:
            with self.subTest(ref_name=ref.get("filename", "")):
                self.assertEqual(extract_output_refs(self._history([ref])), [])


if __name__ == "__main__":
    unittest.main()
