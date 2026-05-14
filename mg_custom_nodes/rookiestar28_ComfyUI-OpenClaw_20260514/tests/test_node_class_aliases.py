import ast
import inspect
import os
import sys
import textwrap
import unittest

sys.path.append(os.getcwd())

from nodes.batch_variants import MoltbotBatchVariants, OpenClawBatchVariants
from nodes.image_to_prompt import MoltbotImageToPrompt, OpenClawImageToPrompt
from nodes.prompt_planner import MoltbotPromptPlanner, OpenClawPromptPlanner
from nodes.prompt_refiner import MoltbotPromptRefiner, OpenClawPromptRefiner


class TestNodeClassAliases(unittest.TestCase):
    def test_legacy_aliases_resolve_to_openclaw_classes(self):
        self.assertIs(MoltbotPromptPlanner, OpenClawPromptPlanner)
        self.assertIs(MoltbotBatchVariants, OpenClawBatchVariants)
        self.assertIs(MoltbotImageToPrompt, OpenClawImageToPrompt)
        self.assertIs(MoltbotPromptRefiner, OpenClawPromptRefiner)

    def test_current_node_categories_use_openclaw_baseline(self):
        node_classes = (
            OpenClawPromptPlanner,
            OpenClawBatchVariants,
            OpenClawImageToPrompt,
            OpenClawPromptRefiner,
        )

        for node_class in node_classes:
            with self.subTest(node_class=node_class.__name__):
                self.assertEqual(node_class.CATEGORY, "openclaw")

    def test_batch_variants_execution_path_has_no_local_imports(self):
        source = inspect.getsource(OpenClawBatchVariants.generate_variants)
        tree = ast.parse(textwrap.dedent(source))

        local_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        self.assertEqual(local_imports, [])


if __name__ == "__main__":
    unittest.main()
