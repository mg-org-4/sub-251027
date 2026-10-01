from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

STRICT_WORKFLOWS = {
    "07_ltx23_character_motion_transfer.json": 0,
    "09_minimax_h3_t2va_director.json": 13,
    "10_minimax_h3_ref2va_director.json": 13,
    "11_ideogram4_strict_grounded_prompt.json": 1,
    "12_minimax_h3_ref2va_grounding_guard_drop_in.json": 13,
}


def _expanded_subgraph_string_input_fragment(
    workflow: dict,
    *,
    subgraph_node_id: int,
    outer_input_name: str,
) -> dict:
    """Materialize the API prompt fragment for one promoted string input."""

    outer_nodes = {node["id"]: node for node in workflow["nodes"]}
    outer_links = {link[0]: link for link in workflow["links"]}
    outer_node = outer_nodes[subgraph_node_id]
    outer_input_index, outer_input = next(
        (index, item)
        for index, item in enumerate(outer_node["inputs"])
        if item["name"] == outer_input_name
    )
    source_link = outer_links[outer_input["link"]]
    if source_link[3:5] != [subgraph_node_id, outer_input_index]:
        raise AssertionError("Outer link target does not match the promoted input slot")
    source_ref = [str(source_link[1]), source_link[2]]

    subgraph = next(
        item
        for item in workflow["definitions"]["subgraphs"]
        if item["id"] == outer_node["type"]
    )
    subgraph_nodes = {node["id"]: node for node in subgraph["nodes"]}
    subgraph_links = {link["id"]: link for link in subgraph["links"]}
    subgraph_input_index = next(
        index
        for index, item in enumerate(subgraph["inputs"])
        if item["name"] == outer_input_name
    )
    if outer_input_index != subgraph_input_index:
        raise AssertionError(
            f"Promoted input index mismatch: outer={outer_input_index}, "
            f"subgraph={subgraph_input_index}"
        )
    inbound_links = [
        link
        for link in subgraph_links.values()
        if link["origin_id"] == -10 and link["origin_slot"] == subgraph_input_index
    ]
    if len(inbound_links) != 1:
        raise AssertionError(
            f"Expected one internal route for {outer_input_name!r}, got {inbound_links!r}"
        )

    inbound = inbound_links[0]
    primitive = subgraph_nodes[inbound["target_id"]]
    primitive_input = primitive["inputs"][inbound["target_slot"]]
    if primitive["type"] != "PrimitiveStringMultiline" or primitive_input["name"] != "value":
        raise AssertionError(
            f"{outer_input_name!r} must enter a PrimitiveStringMultiline before encoding"
        )
    if primitive_input.get("link") != inbound["id"]:
        raise AssertionError("Primitive input does not reference the promoted subgraph link")

    encoder_route = None
    for output_slot, output in enumerate(primitive["outputs"]):
        for link_id in output.get("links") or []:
            link = subgraph_links[link_id]
            target = subgraph_nodes[link["target_id"]]
            target_input = target["inputs"][link["target_slot"]]
            if target["type"] == "CLIPTextEncode" and target_input["name"] == "text":
                encoder_route = (output_slot, link, target, target_input)
                break
        if encoder_route is not None:
            break
    if encoder_route is None:
        raise AssertionError(
            f"{outer_input_name!r} is not routed from the primitive into CLIPTextEncode.text"
        )

    output_slot, encoder_link, encoder, encoder_input = encoder_route
    if encoder_input.get("link") != encoder_link["id"]:
        raise AssertionError("Encoder text input does not reference the primitive output link")

    primitive_api_id = f"{subgraph_node_id}:{primitive['id']}"
    encoder_api_id = f"{subgraph_node_id}:{encoder['id']}"
    return {
        primitive_api_id: {
            "inputs": {"value": source_ref},
            "class_type": "PrimitiveStringMultiline",
        },
        encoder_api_id: {
            "inputs": {"text": [primitive_api_id, output_slot]},
            "class_type": "CLIPTextEncode",
        },
    }


class StrictGroundingWorkflowTests(unittest.TestCase):
    def _load(self, filename: str) -> dict:
        return json.loads((EXAMPLES / filename).read_text(encoding="utf-8"))

    def test_strict_workflow_links_are_serialization_consistent(self):
        for filename in STRICT_WORKFLOWS:
            with self.subTest(filename=filename):
                workflow = self._load(filename)
                nodes = {node["id"]: node for node in workflow["nodes"]}
                links = {link[0]: link for link in workflow["links"]}

                self.assertEqual(len(links), len(workflow["links"]))
                self.assertEqual(workflow["last_node_id"], max(nodes))
                self.assertEqual(workflow["last_link_id"], max(links))

                for link_id, source_id, source_slot, target_id, target_slot, _type in workflow["links"]:
                    self.assertIn(source_id, nodes)
                    self.assertIn(target_id, nodes)
                    source = nodes[source_id]["outputs"][source_slot]
                    target = nodes[target_id]["inputs"][target_slot]
                    self.assertIn(link_id, source.get("links") or [])
                    self.assertEqual(target.get("link"), link_id)

                for node in workflow["nodes"]:
                    for output in node.get("outputs", []):
                        for link_id in output.get("links") or []:
                            self.assertIn(link_id, links)
                    for input_socket in node.get("inputs", []):
                        link_id = input_socket.get("link")
                        if link_id is not None:
                            self.assertIn(link_id, links)

    def test_each_acceptance_workflow_uses_strict_settings_and_gate(self):
        for filename, routed_prompt_slot in STRICT_WORKFLOWS.items():
            with self.subTest(filename=filename):
                workflow = self._load(filename)
                nodes = {node["id"]: node for node in workflow["nodes"]}
                links = {link[0]: link for link in workflow["links"]}

                settings = [n for n in nodes.values() if n["type"] == "DiffusionGemmaGroundingGuardSettings"]
                generators = [n for n in nodes.values() if n["type"] == "DiffusionGemmaCoTGenerator"]
                splitters = [n for n in nodes.values() if n["type"] == "DiffusionGemmaJSONSplitter"]
                gates = [n for n in nodes.values() if n["type"] == "DiffusionGemmaGenerationGate"]
                self.assertEqual((len(settings), len(generators), len(splitters), len(gates)), (1, 1, 1, 1))

                setting, generator, splitter, gate = settings[0], generators[0], splitters[0], gates[0]
                self.assertEqual(setting["widgets_values"][0], "strict")
                self.assertEqual(setting["widgets_values"][-1], "auto")
                guard_input = next(item for item in generator["inputs"] if item["name"] == "grounding_guard_config")
                guard_link = links[guard_input["link"]]
                self.assertEqual(guard_link[1:5], [setting["id"], 0, generator["id"], generator["inputs"].index(guard_input)])

                self.assertEqual(
                    [output["name"] for output in generator["outputs"][:4]],
                    ["final_json", "reasoning_text", "raw_response", "metadata_json"],
                )
                self.assertEqual(
                    [output["name"] for output in generator["outputs"][4:]],
                    ["grounding_status", "grounding_report_json"],
                )

                expected_gate_inputs = {"prompt": routed_prompt_slot, "ready_for_generation": 12, "metadata_json": 4}
                for input_index, input_socket in enumerate(gate["inputs"]):
                    link = links[input_socket["link"]]
                    self.assertEqual(link[1], splitter["id"])
                    self.assertEqual(link[2], expected_gate_inputs[input_socket["name"]])
                    self.assertEqual(link[3:5], [gate["id"], input_index])

                gate_output_links = gate["outputs"][0].get("links") or []
                self.assertTrue(gate_output_links)
                self.assertTrue(all(links[link_id][3] != gate["id"] for link_id in gate_output_links))

    def test_minimax_drop_in_stops_at_a_harmless_gate_preview(self):
        workflow = self._load("12_minimax_h3_ref2va_grounding_guard_drop_in.json")
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}

        self.assertFalse(any(node["type"].startswith("MiniMaxH3") for node in nodes.values()))
        gate = next(node for node in nodes.values() if node["type"] == "DiffusionGemmaGenerationGate")
        self.assertEqual(len(gate["outputs"][0]["links"]), 1)
        sink_link = links[gate["outputs"][0]["links"][0]]
        self.assertEqual(nodes[sink_link[3]]["type"], "PreviewAny")

        context = next(node for node in nodes.values() if node["type"] == "DiffusionGemmaH3ReferenceContext")
        self.assertEqual(next(item for item in context["inputs"] if item["name"] == "reference_video")["link"], None)
        self.assertIn("<Picture 1>", context["widgets_values"][0])
        self.assertIn("<Picture 2>", context["widgets_values"][0])

    def test_minimax_examples_use_the_dedicated_target_widget_order(self):
        expected_modes = {
            "09_minimax_h3_t2va_director.json": "t2va",
            "10_minimax_h3_ref2va_director.json": "ref2va",
            "12_minimax_h3_ref2va_grounding_guard_drop_in.json": "ref2va",
        }
        for filename, expected_mode in expected_modes.items():
            with self.subTest(filename=filename):
                workflow = self._load(filename)
                target = next(
                    node
                    for node in workflow["nodes"]
                    if node["type"] == "DiffusionGemmaMiniMaxH3TargetProfile"
                )
                self.assertEqual(len(target["widgets_values"]), 11)
                self.assertEqual(target["widgets_values"][0], expected_mode)
                self.assertEqual(target["widgets_values"][4], "auto")
                self.assertEqual(target["widgets_values"][6], "auto")

    def test_all_shipped_director_examples_use_model_specific_targets(self):
        expected = {
            "07_ltx23_character_motion_transfer.json": (
                "DiffusionGemmaLTX25TargetProfile",
                7,
            ),
            "09_minimax_h3_t2va_director.json": (
                "DiffusionGemmaMiniMaxH3TargetProfile",
                11,
            ),
            "10_minimax_h3_ref2va_director.json": (
                "DiffusionGemmaMiniMaxH3TargetProfile",
                11,
            ),
            "11_ideogram4_strict_grounded_prompt.json": (
                "DiffusionGemmaIdeogram4TargetProfile",
                6,
            ),
            "12_minimax_h3_ref2va_grounding_guard_drop_in.json": (
                "DiffusionGemmaMiniMaxH3TargetProfile",
                11,
            ),
        }
        for filename, (expected_type, expected_widget_count) in expected.items():
            with self.subTest(filename=filename):
                workflow = self._load(filename)
                targets = [
                    node for node in workflow["nodes"] if "TargetProfile" in node["type"]
                ]
                self.assertEqual(len(targets), 1)
                self.assertEqual(targets[0]["type"], expected_type)
                self.assertEqual(len(targets[0]["widgets_values"]), expected_widget_count)
                self.assertNotEqual(targets[0]["type"], "DiffusionGemmaTargetProfile")

    def test_ltx25_i2v_example_routes_the_visible_resolution_selection(self):
        workflow = self._load("13_ltx25_i2v_director.json")
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}
        splitter = nodes[187]
        ltx = nodes[456]

        self.assertEqual(
            splitter["widgets_values"],
            ["", 0.5, 32, "16:9 (Widescreen)"],
        )
        self.assertEqual(splitter["outputs"][10]["name"], "resolution_width")
        self.assertEqual(splitter["outputs"][11]["name"], "resolution_height")
        self.assertEqual(splitter["outputs"][10]["links"], [26])
        self.assertEqual(splitter["outputs"][11]["links"], [27])
        self.assertEqual(links[26][1:5], [187, 10, 456, 4])
        self.assertEqual(links[27][1:5], [187, 11, 456, 5])
        self.assertEqual(ltx["inputs"][4]["label"], "width")
        self.assertEqual(ltx["inputs"][4]["link"], 26)
        self.assertEqual(ltx["inputs"][5]["label"], "height")
        self.assertEqual(ltx["inputs"][5]["link"], 27)
        self.assertEqual(workflow["last_link_id"], 27)

    def test_ltx_examples_keep_candidate_previews_off_native_generation_paths(self):
        for filename in (
            "13_ltx25_i2v_director.json",
            "14_ltx25_i2v_vs_minimax_h3_ref2va_comparison.json",
        ):
            with self.subTest(filename=filename):
                workflow = self._load(filename)
                nodes = {node["id"]: node for node in workflow["nodes"]}
                links = {link[0]: link for link in workflow["links"]}
                ltx_splitter = next(
                    node
                    for node in nodes.values()
                    if node["type"] == "DiffusionGemmaJSONSplitter"
                    and len(node.get("outputs", [])) >= 16
                    and node["outputs"][14]["name"] == "candidate_ltx_prompt"
                )

                self.assertEqual(
                    [output["name"] for output in ltx_splitter["outputs"][14:16]],
                    ["candidate_ltx_prompt", "candidate_negative_prompt"],
                )

                candidate_preview_ids: set[int] = set()
                for output_slot, title_fragment in (
                    (14, "Candidate LTX positive prompt"),
                    (15, "Candidate LTX negative prompt"),
                ):
                    output_links = ltx_splitter["outputs"][output_slot].get("links") or []
                    self.assertEqual(len(output_links), 1)
                    link = links[output_links[0]]
                    self.assertEqual(link[1:3], [ltx_splitter["id"], output_slot])
                    preview = nodes[link[3]]
                    candidate_preview_ids.add(preview["id"])
                    self.assertEqual(preview["type"], "PreviewAny")
                    self.assertIn(title_fragment, preview.get("title", ""))
                    self.assertEqual(preview["inputs"][link[4]]["link"], link[0])

                positive_links = ltx_splitter["outputs"][0].get("links") or []
                self.assertEqual(len(positive_links), 1)
                positive_link = links[positive_links[0]]
                self.assertEqual(positive_link[1:3], [ltx_splitter["id"], 0])
                gate = nodes[positive_link[3]]
                self.assertIn(
                    gate["type"],
                    {
                        "DiffusionGemmaGenerationGate",
                        "DiffusionGemmaBranchGenerationGate",
                    },
                )
                self.assertEqual(gate["inputs"][positive_link[4]]["name"], "prompt")

                gate_prompt_links = gate["outputs"][0].get("links") or []
                self.assertEqual(len(gate_prompt_links), 1)
                gated_native_link = links[gate_prompt_links[0]]
                self.assertEqual(gated_native_link[1:3], [gate["id"], 0])
                native_generator_id = gated_native_link[3]
                self.assertNotEqual(nodes[native_generator_id]["type"], "PreviewAny")

                negative_links = ltx_splitter["outputs"][2].get("links") or []
                self.assertEqual(len(negative_links), 1)
                negative_native_link = links[negative_links[0]]
                self.assertEqual(negative_native_link[1:3], [ltx_splitter["id"], 2])
                self.assertEqual(negative_native_link[3], native_generator_id)

                candidate_link_ids = {
                    link_id
                    for output_slot in (14, 15)
                    for link_id in (ltx_splitter["outputs"][output_slot].get("links") or [])
                }
                self.assertTrue(candidate_link_ids)
                self.assertTrue(
                    all(links[link_id][3] in candidate_preview_ids for link_id in candidate_link_ids)
                )
                self.assertTrue(
                    all(links[link_id][3] != native_generator_id for link_id in candidate_link_ids)
                )

    def test_ltx_negative_prompt_survives_subgraph_api_expansion(self):
        expected_fragment = {
            "456:509": {
                "inputs": {"value": ["187", 2]},
                "class_type": "PrimitiveStringMultiline",
            },
            "456:427": {
                "inputs": {"text": ["456:509", 0]},
                "class_type": "CLIPTextEncode",
            },
        }
        for filename in (
            "13_ltx25_i2v_director.json",
            "14_ltx25_i2v_vs_minimax_h3_ref2va_comparison.json",
        ):
            with self.subTest(filename=filename):
                workflow = self._load(filename)
                target_node = next(
                    node
                    for node in workflow["nodes"]
                    if node["type"] == "DiffusionGemmaLTX25TargetProfile"
                )
                self.assertEqual(target_node["widgets_values"][-1], "Off")
                self.assertEqual(len(target_node["widgets_values"]), 8)
                outer_node = next(node for node in workflow["nodes"] if node["id"] == 456)
                self.assertEqual(
                    outer_node["widgets_values"],
                    [
                        "",
                        5,
                        992,
                        544,
                        347267244880135,
                        24,
                        "",
                        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
                        "ltx-2.5-video-vae-bf16.safetensors",
                        "ltx-2.5-audio-vae-bf16.safetensors",
                        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                        "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                    ],
                    "The promoted negative-prompt widget needs an empty saved value so "
                    "the native model dropdown values do not shift left in ComfyUI.",
                )
                subgraph = next(
                    item
                    for item in workflow["definitions"]["subgraphs"]
                    if item["id"] == outer_node["type"]
                )
                outer_index = next(
                    index
                    for index, item in enumerate(outer_node["inputs"])
                    if item["name"] == "negative_prompt"
                )
                subgraph_index = next(
                    index
                    for index, item in enumerate(subgraph["inputs"])
                    if item["name"] == "negative_prompt"
                )
                self.assertEqual((outer_index, subgraph_index), (8, 8))
                internal_links = {link["id"]: link for link in subgraph["links"]}
                for input_index, subgraph_input in enumerate(subgraph["inputs"]):
                    for link_id in subgraph_input.get("linkIds") or []:
                        internal_link = internal_links[link_id]
                        self.assertEqual(
                            (internal_link["origin_id"], internal_link["origin_slot"]),
                            (-10, input_index),
                        )
                self.assertEqual(
                    _expanded_subgraph_string_input_fragment(
                        workflow,
                        subgraph_node_id=456,
                        outer_input_name="negative_prompt",
                    ),
                    expected_fragment,
                )

    def test_dual_ltx_h3_comparison_keeps_model_specific_contexts_and_gates(self):
        workflow = self._load("14_ltx25_i2v_vs_minimax_h3_ref2va_comparison.json")
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}

        self.assertEqual(len(links), len(workflow["links"]))
        self.assertEqual(workflow["last_node_id"], max(nodes))
        self.assertEqual(workflow["last_link_id"], max(links))
        for link_id, source_id, source_slot, target_id, target_slot, _type in workflow["links"]:
            self.assertIn(link_id, nodes[source_id]["outputs"][source_slot].get("links") or [])
            self.assertEqual(nodes[target_id]["inputs"][target_slot].get("link"), link_id)

        ltx_context = nodes[183]
        h3_context = nodes[513]
        self.assertEqual(ltx_context["type"], "DiffusionGemmaContextHub")
        self.assertEqual(h3_context["type"], "DiffusionGemmaH3ReferenceContext")
        self.assertEqual(h3_context["widgets_values"][-2], "1 image - all visual attributes")
        self.assertEqual(h3_context["widgets_values"][-1], 1)
        self.assertIn("<Picture 1>", h3_context["widgets_values"][0])

        h3_reference_input = next(
            item for item in h3_context["inputs"] if item["name"] == "reference_images"
        )
        self.assertEqual(links[h3_reference_input["link"]][1:5], [179, 0, 513, 1])
        native_h3 = nodes[527]
        native_reference_input = next(
            item for item in native_h3["inputs"] if item["name"] == "ref_images.ref_image_0"
        )
        native_reference_link = links[native_reference_input["link"]]
        self.assertEqual(native_reference_link[1:5], [541, 0, 527, 3])
        reroute_input = nodes[541]["inputs"][0]
        self.assertEqual(links[reroute_input["link"]][1:5], [179, 0, 541, 0])

        for generator_id, splitter_id, context_id in ((186, 187, 183), (516, 517, 513)):
            generator_context = next(
                item for item in nodes[generator_id]["inputs"] if item["name"] == "gemma_context"
            )
            splitter_context = next(
                item for item in nodes[splitter_id]["inputs"] if item["name"] == "gemma_context"
            )
            self.assertEqual(links[generator_context["link"]][1], context_id)
            self.assertEqual(links[splitter_context["link"]][1], context_id)

        self.assertEqual(nodes[185]["widgets_values"][0], "audit")
        self.assertEqual(nodes[515]["widgets_values"][0], "audit")
        self.assertEqual(nodes[516]["widgets_values"][4], 2048)

        for gate_id, splitter_id, prompt_slot, preview_id in (
            (188, 187, 0, 189),
            (542, 517, 13, 543),
        ):
            gate = nodes[gate_id]
            self.assertEqual(gate["type"], "DiffusionGemmaBranchGenerationGate")
            self.assertEqual(
                [output["name"] for output in gate["outputs"]],
                ["prompt", "status", "ready"],
            )
            expected_inputs = {
                "prompt": prompt_slot,
                "ready_for_generation": 12,
                "metadata_json": 4,
            }
            for input_index, input_socket in enumerate(gate["inputs"]):
                link = links[input_socket["link"]]
                self.assertEqual(
                    link[1:5],
                    [splitter_id, expected_inputs[input_socket["name"]], gate_id, input_index],
                )
            status_link = links[gate["outputs"][1]["links"][0]]
            self.assertEqual(status_link[1:5], [gate_id, 1, preview_id, 0])
            self.assertEqual(nodes[preview_id]["type"], "PreviewAny")

        for native_h3_id in (519, 520, *range(522, 536), 537):
            self.assertEqual(nodes[native_h3_id]["mode"], 0)

        h3_splitter = nodes[517]
        h3_gate = nodes[542]
        expected_gate_inputs = {
            "prompt": 13,
            "ready_for_generation": 12,
            "metadata_json": 4,
        }
        for input_index, input_socket in enumerate(h3_gate["inputs"]):
            link = links[input_socket["link"]]
            self.assertEqual(link[1:5], [h3_splitter["id"], expected_gate_inputs[input_socket["name"]], 542, input_index])

        native_prompt_input = next(item for item in native_h3["inputs"] if item["name"] == "prompt")
        self.assertEqual(links[native_prompt_input["link"]][1:5], [542, 0, 527, 7])
        self.assertNotIn(899, links)

        def ancestors(output_id: int) -> set[int]:
            visited: set[int] = set()
            stack = [output_id]
            while stack:
                node_id = stack.pop()
                if node_id in visited:
                    continue
                visited.add(node_id)
                for input_socket in nodes[node_id].get("inputs", []):
                    link_id = input_socket.get("link")
                    if link_id is not None:
                        stack.append(links[link_id][1])
            return visited

        ltx_branch = ancestors(481)
        h3_branch = ancestors(533)
        self.assertEqual(ltx_branch & h3_branch, {177, 178, 179, 182})
        self.assertTrue({183, 185, 186, 187, 188, 456, 481} <= ltx_branch)
        self.assertTrue({513, 515, 516, 517, 518, 527, 529, 533, 542} <= h3_branch)
        self.assertTrue({513, 515, 516, 517, 518, 527, 529, 533, 542}.isdisjoint(ltx_branch))
        self.assertTrue({183, 185, 186, 187, 188, 456, 481}.isdisjoint(h3_branch))


if __name__ == "__main__":
    unittest.main()
