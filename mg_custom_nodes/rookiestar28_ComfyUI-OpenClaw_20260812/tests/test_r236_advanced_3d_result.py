import json
import unittest
from urllib.parse import parse_qs, urlparse

from services.comfyui_history import extract_output_refs


class TestR236Advanced3DResult(unittest.TestCase):
    VALID_SUFFIXES = (
        "glb",
        "gltf",
        "obj",
        "fbx",
        "stl",
        "ply",
        "spz",
        "splat",
        "ksplat",
        "usdz",
    )

    @staticmethod
    def _extract(result):
        return extract_output_refs({"outputs": {"9": {"result": result}}})

    def test_official_result_tuple_projects_only_first_3d_path(self):
        class GuardedResult(list):
            def __getitem__(self, index):
                if index != 0:
                    raise AssertionError("later result entries were inspected")
                return super().__getitem__(index)

        class ExplosiveMetadata:
            def __str__(self):
                raise AssertionError("later result metadata was inspected")

            def __repr__(self):
                raise AssertionError("later result metadata was inspected")

        metadata_canary = "metadata-value-must-not-project"
        outputs = self._extract(
            GuardedResult(
                [
                    "models/scene one.splat",
                    ExplosiveMetadata(),
                    [{"model": metadata_canary}],
                ]
            )
        )

        self.assertEqual(len(outputs), 1)
        output = outputs[0]
        self.assertEqual(
            {
                "filename": output["filename"],
                "subfolder": output["subfolder"],
                "type": output["type"],
                "media_type": output["media_type"],
                "asset_hash": output["asset_hash"],
                "asset_api_id": output["asset_api_id"],
                "asset_api_required": output["asset_api_required"],
                "resolution": output["resolution"],
            },
            {
                "filename": "scene one.splat",
                "subfolder": "models",
                "type": "output",
                "media_type": "3d",
                "asset_hash": "",
                "asset_api_id": "",
                "asset_api_required": False,
                "resolution": "view",
            },
        )
        params = parse_qs(urlparse(output["view_url"]).query)
        self.assertEqual(
            params,
            {
                "filename": ["scene one.splat"],
                "subfolder": ["models"],
                "type": ["output"],
            },
        )
        self.assertNotIn(metadata_canary, json.dumps(output, sort_keys=True))

    def test_accepts_reviewed_suffixes_and_normalizes_backslashes(self):
        history = {
            "outputs": {
                str(index): {
                    "result": [
                        (
                            f"nested\\folder\\scene.{suffix.upper()}"
                            if index == 0
                            else f"nested/scene.{suffix.upper()}"
                        )
                    ]
                }
                for index, suffix in enumerate(self.VALID_SUFFIXES)
            }
        }

        outputs = extract_output_refs(history)

        self.assertEqual(len(outputs), len(self.VALID_SUFFIXES))
        self.assertTrue(all(output["media_type"] == "3d" for output in outputs))
        self.assertEqual(outputs[0]["subfolder"], "nested/folder")
        self.assertEqual(outputs[0]["filename"], "scene.GLB")
        self.assertEqual(
            [output["filename"].rsplit(".", 1)[-1].lower() for output in outputs],
            list(self.VALID_SUFFIXES),
        )
        unicode_output = self._extract(["模型/場景😀.glb"])
        self.assertEqual(
            (unicode_output[0]["subfolder"], unicode_output[0]["filename"]),
            ("模型", "場景😀.glb"),
        )

    def test_enforces_container_and_unicode_path_bounds(self):
        max_path = ("a" * (1024 - len(".glb"))) + ".glb"
        self.assertEqual(len(self._extract([max_path] + [{}] * 7)), 1)

        rejected = (
            [],
            ["scene.glb"] + [{}] * 8,
            [("a" * (1025 - len(".glb"))) + ".glb"],
        )
        for result in rejected:
            with self.subTest(result_length=len(result)):
                self.assertEqual(self._extract(result), [])

    def test_rejects_malformed_or_unsafe_first_entries(self):
        rejected = (
            None,
            "scene.glb",
            {},
            [None],
            [123],
            [""],
            ["   "],
            [" scene.glb"],
            ["scene.glb "],
            ["\u00a0scene.glb"],
            ["/absolute/scene.glb"],
            ["//evil.example/scene.glb"],
            ["https://evil.example/scene.glb"],
            ["file:scene.glb"],
            ["C:\\private\\scene.glb"],
            ["../scene.glb"],
            ["safe/../scene.glb"],
            ["safe/./scene.glb"],
            ["safe//scene.glb"],
            ["safe/\x00scene.glb"],
            ["safe/\u0085scene.glb"],
            ["safe/\u202escene.glb"],
            ["safe/\ud800scene.glb"],
            ["scene.glb?token=secret"],
            ["scene.png"],
            ["scene.glb.exe"],
        )
        for result in rejected:
            with self.subTest(result=result):
                self.assertEqual(self._extract(result), [])

    def test_existing_output_families_remain_unchanged(self):
        outputs = extract_output_refs(
            {
                "outputs": {
                    "1": {
                        "images": [{"filename": "image.png", "type": "output"}],
                        "video": [{"filename": "clip.webm", "type": "output"}],
                        "audio": [{"filename": "sound.wav", "type": "output"}],
                        "3d": ["classic.glb"],
                        "text": ["hello"],
                        "files": [{"filename": "report.txt", "type": "output"}],
                    }
                }
            }
        )

        self.assertEqual(
            [output["media_type"] for output in outputs],
            ["images", "video", "audio", "3d", "text", "text"],
        )


if __name__ == "__main__":
    unittest.main()
