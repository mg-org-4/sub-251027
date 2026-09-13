from __future__ import annotations

import importlib.util
import json
import sys
import unittest
import uuid
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]


def load_nodes_module():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    package_name = f"diffusiongemma_h3_reference_prep_fixture_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        package_name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load DiffusionGemma Prompt Builder")
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    return sys.modules[f"{package_name}.nodes"]


class H3ReferencePairPrepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()
        cls.node = cls.nodes.DiffusionGemmaH3ReferencePairPrep()

    def test_reference_prep_nodes_are_exported_as_optional_nodes(self) -> None:
        self.assertIs(
            self.nodes.NODE_CLASS_MAPPINGS["DiffusionGemmaReferencePrep"],
            self.nodes.DiffusionGemmaReferencePrep,
        )
        self.assertIs(
            self.nodes.NODE_CLASS_MAPPINGS["DiffusionGemmaH3ReferencePairPrep"],
            self.nodes.DiffusionGemmaH3ReferencePairPrep,
        )
        self.assertEqual(
            self.nodes.DiffusionGemmaH3ReferencePairPrep.CATEGORY,
            self.nodes.OPTIONAL_CATEGORY,
        )
        required = self.nodes.DiffusionGemmaH3ReferencePairPrep.INPUT_TYPES()["required"]
        self.assertEqual(required["combined_reference_area_ratio"][1]["default"], 1.0)
        self.assertEqual(required["reference_1_share"][1]["default"], 0.60)

    def test_single_reference_prep_does_not_upscale_by_default(self) -> None:
        image = torch.rand((1, 768, 768, 3), dtype=torch.float32)
        prepared = self.nodes.DiffusionGemmaReferencePrep().prepare(
            image,
            "source",
            "preserve_source",
            1280,
            8,
        )
        self.assertIs(prepared[0], image)
        self.assertEqual((prepared[2], prepared[3]), (768, 768))
        self.assertFalse(json.loads(prepared[1])["upscale_smaller_images"])

    def test_current_square_and_portrait_pair_share_one_generation_frame_budget(self) -> None:
        picture_1 = torch.zeros((1, 768, 768, 3), dtype=torch.float32)
        picture_1[..., 0] = 1.0
        picture_2 = torch.zeros((1, 1808, 1024, 3), dtype=torch.float32)
        picture_2[..., 1] = 1.0

        result = self.node.prepare(picture_1, picture_2, 1376, 768, 1.0, 0.60)
        output_1, output_2 = result[0], result[1]
        metadata = json.loads(result[2])

        self.assertEqual(tuple(output_1.shape), (1, 768, 768, 3))
        self.assertEqual(tuple(output_2.shape), (1, 896, 512, 3))
        self.assertEqual((result[3], result[4]), (768, 768))
        self.assertEqual((result[5], result[6]), (512, 896))
        self.assertLessEqual(
            int(output_1.shape[1] * output_1.shape[2] + output_2.shape[1] * output_2.shape[2]),
            1376 * 768,
        )
        self.assertEqual(result[8], 1024)
        self.assertEqual(metadata["native_match_estimated_packed_reference_rows"], 1608)
        self.assertEqual(metadata["references"][0]["picture_tag"], "<Picture 1>")
        self.assertEqual(metadata["references"][1]["picture_tag"], "<Picture 2>")
        self.assertAlmostEqual(float(output_1[..., 0].mean()), 1.0, places=5)
        self.assertAlmostEqual(float(output_2[..., 1].mean()), 1.0, places=5)

    def test_outputs_are_downscale_only_aligned_and_preserve_each_aspect(self) -> None:
        picture_1 = torch.rand((1, 777, 1234, 3), dtype=torch.float32)
        picture_2 = torch.rand((1, 1401, 733, 3), dtype=torch.float32)
        result = self.node.prepare(picture_1, picture_2, 1024, 576, 0.50, 0.50)
        output_1, output_2 = result[0], result[1]

        for output, source in ((output_1, picture_1), (output_2, picture_2)):
            self.assertLessEqual(output.shape[1], source.shape[1])
            self.assertLessEqual(output.shape[2], source.shape[2])
            self.assertEqual(output.shape[1] % 32, 0)
            self.assertEqual(output.shape[2] % 32, 0)
            source_ratio = source.shape[2] / source.shape[1]
            output_ratio = output.shape[2] / output.shape[1]
            self.assertLess(abs(source_ratio - output_ratio) / source_ratio, 0.08)
            self.assertEqual(output.dtype, source.dtype)
            self.assertEqual(output.device, source.device)

        self.assertLessEqual(
            int(output_1.shape[1] * output_1.shape[2] + output_2.shape[1] * output_2.shape[2]),
            int(1024 * 576 * 0.50),
        )

    def test_unused_budget_is_reassigned_and_an_aligned_small_image_is_not_upscaled(self) -> None:
        picture_1 = torch.rand((1, 320, 320, 3), dtype=torch.float32)
        picture_2 = torch.rand((1, 1024, 1024, 3), dtype=torch.float32)
        result = self.node.prepare(picture_1, picture_2, 1024, 1024, 1.0, 0.80)
        output_1, output_2 = result[0], result[1]
        metadata = json.loads(result[2])

        self.assertIs(output_1, picture_1)
        self.assertEqual(tuple(output_1.shape[1:3]), (320, 320))
        self.assertGreater(output_2.shape[1] * output_2.shape[2], int(1024 * 1024 * 0.20))
        self.assertGreater(metadata["references"][1]["allocated_pixel_budget"], int(1024 * 1024 * 0.20))

    def test_reference_one_share_changes_relative_output_budget(self) -> None:
        picture_1 = torch.rand((1, 1024, 1024, 3), dtype=torch.float32)
        picture_2 = torch.rand((1, 1024, 1024, 3), dtype=torch.float32)
        result = self.node.prepare(picture_1, picture_2, 1024, 1024, 1.0, 0.70)

        pixels_1 = result[3] * result[4]
        pixels_2 = result[5] * result[6]
        self.assertGreater(pixels_1, pixels_2)
        self.assertLessEqual(pixels_1 + pixels_2, 1024 * 1024)

    def test_prepared_pair_is_idempotent_for_the_same_budget(self) -> None:
        picture_1 = torch.rand((1, 768, 768, 3), dtype=torch.float32)
        picture_2 = torch.rand((1, 1808, 1024, 3), dtype=torch.float32)
        first = self.node.prepare(picture_1, picture_2, 1376, 768, 1.0, 0.60)
        second = self.node.prepare(first[0], first[1], 1376, 768, 1.0, 0.60)

        self.assertEqual(tuple(first[0].shape), tuple(second[0].shape))
        self.assertEqual(tuple(first[1].shape), tuple(second[1].shape))
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

    def test_multi_image_batch_is_rejected_instead_of_silently_dropping_a_reference(self) -> None:
        batch = torch.rand((2, 512, 512, 3), dtype=torch.float32)
        singleton = torch.rand((1, 512, 512, 3), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            self.node.prepare(batch, singleton, 1024, 1024, 1.0, 0.50)

    def test_malformed_or_non_rgb_images_are_rejected_clearly(self) -> None:
        singleton = torch.rand((1, 512, 512, 3), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "shaped"):
            self.node.prepare(torch.rand((512, 512, 3)), singleton, 1024, 1024, 1.0, 0.50)
        with self.assertRaisesRegex(ValueError, "RGB"):
            self.node.prepare(torch.rand((1, 512, 512, 4)), singleton, 1024, 1024, 1.0, 0.50)


if __name__ == "__main__":
    unittest.main()
