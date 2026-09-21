from __future__ import annotations

import importlib.util
import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]


def load_nodes_module():
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))
    package_name = f"diffusiongemma_h3_reference_policy_{uuid.uuid4().hex}"
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


class MiniMaxH3ReferencePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = load_nodes_module()

    def target(self, mode: str = "ref2va"):
        return self.nodes.TargetProfileConfig(
            target_profile="minimax_h3",
            minimax_h3_mode=mode,
        )

    def test_policy_node_is_registered_and_director_socket_is_appended(self) -> None:
        self.assertIn("DiffusionGemmaH3ReferencePolicy", self.nodes.NODE_CLASS_MAPPINGS)
        node = self.nodes.DiffusionGemmaH3ReferencePolicy
        self.assertEqual(
            node.RETURN_TYPES,
            (self.nodes.H3_REFERENCE_POLICY_TYPE, "STRING", "STRING"),
        )
        required = node.INPUT_TYPES()["required"]
        self.assertEqual(
            list(required),
            ["layout", "expected_subject_count", "custom_manifest"],
        )
        self.assertEqual(required["layout"][1]["default"], "Auto (recommended)")

        director_optional = self.nodes.DiffusionGemmaCoTGenerator.INPUT_TYPES()["optional"]
        self.assertEqual(list(director_optional)[-1], "h3_reference_policy")
        self.assertEqual(
            director_optional["h3_reference_policy"][0],
            self.nodes.H3_REFERENCE_POLICY_TYPE,
        )

    def test_generic_context_one_image_autopopulates_manifest(self) -> None:
        image = torch.zeros((1, 32, 48, 3), dtype=torch.float32)
        context, _context_json, _preview = self.nodes.DiffusionGemmaContextHub().build(
            "The referenced subject crosses a moonlit room.",
            image=image,
        )
        self.assertNotIn("minimax_h3_reference_manifest", context.media_metadata)

        prepared = self.nodes._context_with_minimax_h3_reference_policy(
            context,
            self.target(),
        )

        manifest = prepared.media_metadata["minimax_h3_reference_manifest"]
        self.assertEqual(self.nodes._minimax_h3_reference_tags(manifest), ["<Picture 1>"])
        self.assertEqual(
            prepared.media_metadata["minimax_h3_reference_manifest_source"],
            "director_auto",
        )
        self.assertEqual(
            prepared.media_metadata["minimax_h3_reference_manifest_reasons"],
            [],
        )
        self.assertNotIn("minimax_h3_reference_manifest", context.media_metadata)

    def test_h3_context_treats_blank_inline_manifest_as_pending_auto(self) -> None:
        image = torch.zeros((1, 32, 48, 3), dtype=torch.float32)
        context, context_json, preview = self.nodes.DiffusionGemmaH3ReferenceContext().build(
            "Animate the attached reference.",
            "",
            reference_images=image,
            reference_manifest_preset="custom",
        )

        metadata = json.loads(context_json)["media"]
        self.assertEqual(
            context.media_metadata["minimax_h3_reference_manifest_source"],
            "pending_auto",
        )
        self.assertEqual(metadata["minimax_h3_reference_manifest_reasons"], [])
        self.assertNotIn("needs correction", "\n".join(context.warnings))
        self.assertNotIn("manifest declares 0", "\n".join(context.warnings))
        self.assertIn("H3 reference roles: Auto", preview)

    def test_context_hub_records_the_real_image_batch_count(self) -> None:
        images = torch.zeros((3, 32, 48, 3), dtype=torch.float32)
        context, context_json, _preview = self.nodes.DiffusionGemmaContextHub().build(
            "Use these three ordered references.",
            image=images,
        )
        self.assertEqual(context.media_metadata["reference_image_count"], 3)
        self.assertEqual(json.loads(context_json)["media"]["reference_image_count"], 3)

        prepared = self.nodes._context_with_minimax_h3_reference_policy(
            context,
            self.target(),
        )
        self.assertEqual(
            self.nodes._minimax_h3_reference_tags(
                prepared.media_metadata["minimax_h3_reference_manifest"]
            ),
            ["<Picture 1>", "<Picture 2>", "<Picture 3>"],
        )

    def test_connected_contact_sheet_policy_overrides_stale_inline_manifest(self) -> None:
        images = torch.zeros((2, 32, 48, 3), dtype=torch.float32)
        stale_manifest = self.nodes._MINIMAX_H3_ONE_PICTURE_MANIFEST
        context = self.nodes.GemmaContext(
            user_prompt="The hero uses the two requested tools from the contact sheet.",
            images=images,
            source="image",
            media_metadata={
                "source": "image",
                "reference_image_count": 2,
                "reference_image_backend_attached": True,
                "minimax_h3_reference_manifest": stale_manifest,
            },
        )
        policy, _json, _preview = self.nodes.DiffusionGemmaH3ReferencePolicy().build(
            "Primary subject + supporting subjects/objects contact sheet",
            3,
            "",
        )

        prepared = self.nodes._context_with_minimax_h3_reference_policy(
            context,
            self.target(),
            policy,
        )
        metadata = prepared.media_metadata
        self.assertEqual(metadata["minimax_h3_reference_manifest_source"], "policy_node")
        self.assertEqual(metadata["minimax_h3_expected_subject_count"], 3)
        self.assertIn("one multi-entity contact sheet", metadata["minimax_h3_reference_manifest"])
        self.assertEqual(
            self.nodes._minimax_h3_reference_tags(metadata["minimax_h3_reference_manifest"]),
            ["<Picture 1>", "<Picture 2>"],
        )
        self.assertNotEqual(metadata["minimax_h3_reference_manifest"], stale_manifest)

    def test_contact_sheet_policy_requires_two_picture_assets(self) -> None:
        policy = self.nodes._make_h3_reference_policy_config(
            "Primary subject + supporting subjects/objects contact sheet"
        )
        with self.assertRaisesRegex(ValueError, "requires exactly 2 attached Picture"):
            self.nodes._minimax_h3_reference_policy_manifest(policy, 1)
        with self.assertRaisesRegex(ValueError, "contact sheet is one Picture asset"):
            self.nodes._minimax_h3_reference_policy_manifest(policy, 3)

    def test_custom_manifest_remains_explicit_and_invalid_instead_of_falling_back(self) -> None:
        image = torch.zeros((1, 32, 48, 3), dtype=torch.float32)
        context = self.nodes.GemmaContext(
            user_prompt="Test custom roles.",
            images=image,
            source="image",
            media_metadata={"source": "image"},
        )
        policy = self.nodes._make_h3_reference_policy_config(
            "Custom manifest",
            "this is intentionally not a tagged manifest",
        )
        prepared = self.nodes._context_with_minimax_h3_reference_policy(
            context,
            self.target(),
            policy,
        )
        self.assertEqual(
            prepared.media_metadata["minimax_h3_reference_manifest_source"],
            "policy_node",
        )
        self.assertIn(
            "minimax_h3_ref_manifest_missing",
            prepared.media_metadata["minimax_h3_reference_manifest_reasons"],
        )

    def test_missing_visual_reference_fails_with_connection_guidance(self) -> None:
        context = self.nodes.GemmaContext(user_prompt="Animate my reference.")
        with self.assertRaisesRegex(ValueError, "no visual reference in DG_CONTEXT"):
            self.nodes._context_with_minimax_h3_reference_policy(
                context,
                self.target(),
            )

    def test_non_ref2va_context_is_unchanged(self) -> None:
        context = self.nodes.GemmaContext(user_prompt="Text-only H3 scene.")
        self.assertIs(
            self.nodes._context_with_minimax_h3_reference_policy(
                context,
                self.target("t2va"),
            ),
            context,
        )

    def test_director_resolves_connected_policy_before_generation(self) -> None:
        images = torch.zeros((2, 32, 48, 3), dtype=torch.float32)
        context = self.nodes.GemmaContext(
            user_prompt="The hero selects one requested object from the contact sheet.",
            images=images,
            source="image",
            media_metadata={"source": "image", "reference_image_count": 2},
        )
        policy = self.nodes._make_h3_reference_policy_config(
            "Primary subject + supporting subjects/objects contact sheet",
            expected_subject_count=2,
        )
        runtime = self.nodes.RuntimeConfig(
            model_path="unused",
            backend="template",
            dtype="auto",
            quantization="none",
            local_files_only=True,
            unload_policy="unload_after_run",
            max_memory_gb=0.0,
        )
        captured = {}

        def fake_run(_runtime, prepared_context, *_args, **_kwargs):
            captured["context"] = prepared_context
            metadata = {
                "ready_for_generation": True,
                "grounding_guard": {
                    "analysis_status": "not_run",
                    "decision": "disabled",
                },
            }
            packet = {
                "ltx_prompt": "",
                "ideogram_prompt": "",
                "minimax_h3_prompt": "valid placeholder",
                "negative_prompt": "",
                "scene_segments": [],
                "metadata": metadata,
            }
            return packet, "raw", "", "", True, "", metadata

        with patch.object(self.nodes, "_run_generation_packet", side_effect=fake_run):
            self.nodes.DiffusionGemmaCoTGenerator().generate(
                runtime,
                context,
                self.target(),
                temperature=0.45,
                creativity_mode="faithful",
                creative_strength=0.0,
                thinking_mode="off",
                director_cache_mode="off",
                grounding_guard_mode="off",
                h3_reference_policy=policy,
            )

        prepared = captured["context"]
        self.assertEqual(
            prepared.media_metadata["minimax_h3_reference_manifest_source"],
            "policy_node",
        )
        self.assertIn(
            "one multi-entity contact sheet",
            prepared.media_metadata["minimax_h3_reference_manifest"],
        )

    def test_splitter_restores_director_resolved_policy_over_blank_context(self) -> None:
        image = torch.zeros((1, 32, 48, 3), dtype=torch.float32)
        context = self.nodes.GemmaContext(
            user_prompt="Hold on the referenced traveler in one quiet shot.",
            images=image,
            source="image",
            media_metadata={
                "source": "image",
                "reference_image_count": 1,
                "reference_image_backend_attached": True,
                "minimax_h3_reference_manifest": "",
                "warnings": [
                    "H3 reference manifest needs correction: minimax_h3_ref_manifest_missing",
                    "DiffusionGemma received 1 reference picture(s) for analysis, while the manifest declares 0; the native H3 connections remain authoritative.",
                    "Media pixels will be passed to the active backend.",
                ],
            },
        )
        policy = self.nodes._make_h3_reference_policy_config("Auto (recommended)")
        prepared = self.nodes._context_with_minimax_h3_reference_policy(
            context,
            self.target(),
            policy,
        )
        packet = {
            "ltx_prompt": "",
            "ideogram_prompt": "",
            "minimax_h3_prompt": "valid diagnostic placeholder",
            "negative_prompt": "",
            "scene_segments": [],
            "metadata": {"media": prepared.media_metadata},
        }

        split = self.nodes.DiffusionGemmaJSONSplitter().split(
            json.dumps(packet),
            context,
            self.target(),
        )
        metadata = json.loads(split[4])

        self.assertEqual(
            metadata["media"]["minimax_h3_reference_manifest_source"],
            "policy_node",
        )
        self.assertEqual(
            metadata["media"]["minimax_h3_reference_tags"],
            ["<Picture 1>"],
        )
        self.assertNotIn(
            "minimax_h3_ref_manifest_missing",
            metadata["blocked_reasons"],
        )
        self.assertNotIn(
            "minimax_h3_ref_undefined_tag",
            metadata["blocked_reasons"],
        )
        self.assertEqual(
            metadata["media"]["warnings"],
            ["Media pixels will be passed to the active backend."],
        )

    def test_policy_layout_changes_director_cache_key(self) -> None:
        images = torch.zeros((2, 32, 48, 3), dtype=torch.float32)
        context = self.nodes.GemmaContext(
            user_prompt="Animate the ordered references.",
            images=images,
            source="image",
            media_metadata={"source": "image", "reference_image_count": 2},
        )
        runtime = self.nodes.RuntimeConfig(
            model_path="unused",
            backend="template",
            dtype="auto",
            quantization="none",
            local_files_only=True,
            unload_policy="unload_after_run",
            max_memory_gb=0.0,
        )
        environment_context = self.nodes._context_with_minimax_h3_reference_policy(
            context,
            self.target(),
            self.nodes._make_h3_reference_policy_config(
                "Primary subject + environment/style"
            ),
        )
        contact_context = self.nodes._context_with_minimax_h3_reference_policy(
            context,
            self.target(),
            self.nodes._make_h3_reference_policy_config(
                "Primary subject + supporting subjects/objects contact sheet"
            ),
        )
        common = {
            "temperature": 0.45,
            "creativity_mode": "faithful",
            "creative_strength": 0.0,
            "thinking_mode": "off",
            "max_new_tokens": 1024,
        }
        environment_key = self.nodes._director_cache_key(
            runtime,
            environment_context,
            self.target(),
            {"mode": "off"},
            **common,
        )
        contact_key = self.nodes._director_cache_key(
            runtime,
            contact_context,
            self.target(),
            {"mode": "off"},
            **common,
        )
        self.assertNotEqual(environment_key, contact_key)


if __name__ == "__main__":
    unittest.main()
