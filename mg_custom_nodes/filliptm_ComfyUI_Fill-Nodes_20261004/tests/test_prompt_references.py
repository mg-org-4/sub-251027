import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import wave
from unittest.mock import patch

from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("fl_reference_tests")
package.__path__ = [str(ROOT / "nodes" / "audio")]
sys.modules[package.__name__] = package
references = importlib.import_module("fl_reference_tests.prompt_references")
storyboards = importlib.import_module("fl_reference_tests.prompt_storyboards")
actions = importlib.import_module("fl_reference_tests.prompt_storyboard_actions")
spec = importlib.util.spec_from_file_location("h3_reference_test", ROOT.parent / "ComfyUI-FL-MiniMaxH3" / "nodes" / "_shot_references.py")
h3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h3)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("input", "output"):
            (self.root / name).mkdir()
            mock = patch.object(references.folder_paths, f"get_{name}_directory", return_value=str(self.root / name))
            mock.start()
            self.addCleanup(mock.stop)
        self.store = storyboards.StoryboardStore(self.root / "jobs.db")
        self.spec = {"scheduler_id": "project", "section_id": "section-a", "revision": "revision-a", "prompt": "Four chronological running poses", "grid": 2}

    def test_legacy_defaults(self):
        self.assertEqual(references.reference_document("", 2)["assets"], {})
        defaults = (object(), object(), object(), object())
        self.assertEqual(h3.resolve_shot_references([{}], None, *defaults)[:4], defaults)

    def test_no_references_does_not_consume_defaults(self):
        resolved = h3.resolve_shot_references([{"references": {"mode": "none"}}], None, 1, 2, 3, 4)
        self.assertEqual(resolved, (None, None, None, None, []))

    def test_selection_order_and_video_audio_pairing(self):
        assets = {"a": {"kind": "image", "value": "image-a"}, "b": {"kind": "image", "value": "image-b"},
                  "v": {"kind": "video", "value": "video", "audio": "paired"}, "s": {"kind": "audio", "value": "sound"}}
        section = {"references": {"mode": "custom", "asset_ids": ["b", "v", "a", "s"]}}
        image, video, paired, audio, ids = h3.resolve_shot_references([section], {"version": 1, "assets": assets}, None, None, None, None)
        self.assertEqual(list(image.values()), ["image-b", "image-a"])
        self.assertEqual(video, {"ref_video_0": "video"})
        self.assertEqual(paired, {"ref_video_audio_0": "paired"})
        self.assertEqual(audio, {"ref_audio_0": "sound"})
        self.assertEqual(ids, ["b", "v", "a", "s"])

    def test_group_conflict_and_missing_library(self):
        with self.assertRaisesRegex(ValueError, "Grouped render"):
            h3.resolve_shot_references([{}, {"references": {"mode": "none"}}], None, None, None, None, None)
        with self.assertRaisesRegex(ValueError, "Connect"):
            h3.resolve_shot_references([{"references": {"mode": "custom", "asset_ids": ["a"]}}], None, None, None, None, None)

    def test_path_containment(self):
        for asset in ({"filename": "../secret"}, {"filename": "secret", "subfolder": "../../"},
                      {"filename": "secret", "type": "temp"}, {"filename": "C:\\secret"}):
            with self.assertRaises(ValueError):
                references.reference_path(asset, must_exist=False)

    def test_reference_metadata_roundtrip(self):
        value = {"version": 1, "assets": {"a": {"filename": "a.png", "kind": "image"}},
                 "sections": [{"id": "s1", "mode": "custom", "asset_ids": ["a"]}, {"id": "s2", "mode": "none", "asset_ids": []}]}
        document = references.reference_document(json.dumps(value), 2)
        sections = references.apply_reference_sections([{}, {}], document)
        self.assertEqual(sections[0]["section_id"], "s1")
        self.assertEqual(sections[1]["references"]["mode"], "none")
        with self.assertRaisesRegex(ValueError, "match"):
            references.reference_document(value, 1)
        value["sections"][1]["id"] = "s1"
        with self.assertRaisesRegex(ValueError, "unique"):
            references.reference_document(value, 2)

    def test_claim_is_one_shot_and_persists(self):
        job = self.store.create(self.spec)
        self.assertEqual(job["state"], "proposed")
        graph = self.store.claim(job["id"])
        self.assertEqual({node["class_type"] for node in graph.values()}, {"GeminiNanoBanana2V2", "SaveImage"})
        with self.assertRaisesRegex(ValueError, "already submitted"):
            storyboards.StoryboardStore(self.store.path).claim(job["id"])
        self.assertEqual(len(self.store.list("project")), 1)
        self.assertEqual(self.store.list("other-project"), [])

    def test_cancel_prevents_submission(self):
        job = self.store.create(self.spec)
        self.store.update(job["id"], "cancelled", {})
        with self.assertRaises(ValueError):
            self.store.claim(job["id"])

    def test_batch_branches_share_continuity_and_claim_atomically(self):
        jobs = [self.store.create({**self.spec, "section_id": str(i), "continuity": "One pirate, crimson coat, hand-drawn anime."}) for i in range(3)]
        graph = storyboards.storyboard_batch_graph(jobs)
        self.assertEqual(sum(n["class_type"] == "GeminiNanoBanana2V2" for n in graph.values()), 3)
        for job in jobs:
            prefix = job["id"] + ":"
            self.assertIn("One pirate", graph[prefix + "storyboard"]["inputs"]["prompt"])
            self.assertEqual(graph[prefix + "save_storyboard"]["inputs"]["images"], [prefix + "storyboard", 0])
        self.store.update(jobs[-1]["id"], "cancelled", {})
        with self.assertRaises(ValueError):
            self.store.claim_batch([job["id"] for job in jobs])
        self.assertEqual(self.store.get(jobs[0]["id"])["state"], "proposed")
        self.store.claim_batch([job["id"] for job in jobs[:2]])
        self.assertEqual(self.store.get(jobs[1]["id"])["state"], "submitted")

    def test_continuity_size_is_validated(self):
        with self.assertRaisesRegex(ValueError, "continuity"):
            self.store.create({**self.spec, "continuity": "x" * 32001})

    def test_storyboard_actions_have_no_fixed_count_cap(self):
        requests = [{"index": i, "grid": 2, "prompt": "Same character, next action"} for i in range(12)]
        self.assertEqual(actions.normalize_storyboard_actions(requests, set(range(12))), requests)

    def test_replayed_generation_request_does_not_create_another_paid_job(self):
        spec = {**self.spec, "request_key": "assistant-message:section-a"}
        first = self.store.create(spec)
        self.store.claim(first["id"])
        replay = storyboards.StoryboardStore(self.store.path).create(spec)
        self.assertEqual(replay["id"], first["id"])
        self.assertEqual(replay["state"], "submitted")
        self.assertEqual(len(self.store.list("project")), 1)
        reroll = self.store.create({**spec, "request_key": "explicit-reroll"})
        self.assertNotEqual(reroll["id"], first["id"])
        other = self.store.create({**spec, "scheduler_id": "other"})
        self.assertNotEqual(other["id"], first["id"])
        with self.assertRaisesRegex(ValueError, "request key"):
            self.store.create({**spec, "request_key": ""})

    def test_moodboards_are_explicit_and_allowlisted(self):
        Image.new("RGB", (32, 32)).save(self.root / "input" / "mood.png")
        job = self.store.create({**self.spec, "moodboards": [{"filename": "mood.png", "type": "input", "role": "Color palette"}]})
        graph = self.store.claim(job["id"])
        self.assertEqual(graph["storyboard"]["inputs"]["model.images.image_1"], ["moodboard_1", 0])
        self.assertNotIn("auth_token_comfy_org", json.dumps(graph))
        with self.assertRaises(ValueError):
            self.store.create({**self.spec, "grid": 4})

    def test_extraction_exact_pixels_and_immutable_versions(self):
        job = self.store.create(self.spec)
        folder = self.root / "output" / "fl-storyboards" / job["id"]
        folder.mkdir(parents=True)
        sheet = Image.new("RGB", (100, 80), "red")
        sheet.paste("blue", (50, 40, 100, 80))
        sheet.save(folder / "sheet.png")
        source = {"filename": "sheet.png", "subfolder": f"fl-storyboards/{job['id']}", "type": "output"}
        result = storyboards.extract_panels(job, source)
        self.assertEqual(len(result["assets"]), 4)
        last = list(result["assets"].values())[-1]
        with Image.open(references.reference_path(last)) as panel:
            self.assertEqual(panel.size, (50, 40))
            self.assertEqual(panel.getpixel((0, 0)), (0, 0, 255))
        newer = storyboards.extract_panels({**job, "result": result}, source, [10, 10, 90, 70])
        self.assertEqual(len(newer["versions"]), 2)
        self.assertTrue(references.reference_path(last).is_file())
        with self.assertRaises(ValueError):
            storyboards.extract_panels(job, source, [-1, 0, 90, 70])

    def test_agent_proposal_scope(self):
        value = {"index": 2, "grid": 3, "prompt": "Nine clear chronological beats"}
        self.assertEqual(actions.normalize_storyboard_actions([value], {2}), [value])
        for values in ([value, value], [{**value, "index": 99}], [{**value, "grid": 4}]):
            with self.assertRaises(ValueError):
                actions.normalize_storyboard_actions(values, {2})
        assignment = {"index": 2, "mode": "custom", "asset_ids": ["chosen"]}
        self.assertEqual(actions.normalize_reference_assignments([assignment], {2}, ["chosen"]), [assignment])
        with self.assertRaisesRegex(ValueError, "unavailable assets"):
            actions.normalize_reference_assignments([assignment], {2}, [])
        with self.assertRaisesRegex(ValueError, "unavailable or duplicate"):
            actions.normalize_reference_assignments([assignment], {1}, ["chosen"])

    def test_real_reference_library_loads_selected_pixels_only(self):
        library_module = importlib.import_module("fl_reference_tests.FL_Prompt_Reference_Library")
        Image.new("RGB", (64, 48), (255, 0, 0)).save(self.root / "input" / "selected.png")
        schedule = {"reference_assets": {
            "chosen": {"kind": "image", "filename": "selected.png"},
            "unused": {"kind": "image", "filename": "missing.png"},
        }, "sections": [{"references": {"mode": "custom", "asset_ids": ["chosen"]}}]}
        library = library_module.FL_Prompt_Reference_Library()
        value = library.load(schedule)[0]
        self.assertEqual(list(value["assets"]), ["chosen"])
        pixels = value["assets"]["chosen"]["value"]
        self.assertEqual(tuple(pixels.shape), (1, 48, 64, 3))
        self.assertEqual(pixels[0, 0, 0].tolist(), [1, 0, 0])
        self.assertFalse(hasattr(library, "IS_CHANGED"), "connected schedules are unavailable during cache fingerprinting")
        document = {"version": 1, "assets": schedule["reference_assets"],
                    "sections": [{"id": "shot", "mode": "custom", "asset_ids": ["chosen"]}]}
        before = references.reference_file_fingerprint(json.dumps(document))
        self.assertEqual(before, references.reference_file_fingerprint(json.dumps(document)))
        Image.new("RGB", (65, 48), (0, 255, 0)).save(self.root / "input" / "selected.png")
        self.assertNotEqual(before, references.reference_file_fingerprint(json.dumps(document)))
        self.assertEqual(references.reference_file_fingerprint(""), ())

    def test_real_audio_library_preserves_native_audio_shape(self):
        library_module = importlib.import_module("fl_reference_tests.FL_Prompt_Reference_Library")
        with wave.open(str(self.root / "input" / "sound.wav"), "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(bytes(2400 * 2 * 2))
        schedule = {"reference_assets": {"sound": {"kind": "audio", "filename": "sound.wav"}},
                    "sections": [{"references": {"mode": "custom", "asset_ids": ["sound"]}}]}
        audio = library_module.FL_Prompt_Reference_Library().load(schedule)[0]["assets"]["sound"]["value"]
        self.assertEqual(audio["sample_rate"], 24000)
        self.assertEqual(tuple(audio["waveform"].shape), (1, 2, 2400))

    def test_video_frame_rate_conversion_is_execution_cached(self):
        video = torch.arange(30).reshape(30, 1, 1, 1)
        library = {"version": 1, "assets": {"v": {"kind": "video", "value": video, "fps": 30.0}}}
        sections = [{"references": {"mode": "custom", "asset_ids": ["v"]}}]
        cache = {}
        first = h3.resolve_shot_references(sections, library, None, None, None, None, cache)[1]["ref_video_0"]
        second = h3.resolve_shot_references(sections, library, None, None, None, None, cache)[1]["ref_video_0"]
        self.assertEqual(first.shape[0], 24)
        self.assertIs(first, second)
        self.assertEqual(len(cache), 1)


if __name__ == "__main__":
    unittest.main()
