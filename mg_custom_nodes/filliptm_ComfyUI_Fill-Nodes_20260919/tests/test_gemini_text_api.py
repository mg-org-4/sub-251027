import importlib.util
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import av
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from comfy_api.latest import InputImpl, Types


ROOT = Path(__file__).parents[1]
package = ModuleType("gemini_media_tests")
package.__path__ = [str(ROOT / "nodes" / "ai")]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location("gemini_media_tests.FL_GeminiTextAPI", ROOT / "nodes" / "ai" / "FL_GeminiTextAPI.py")
gemini = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gemini)


class GeminiMediaTests(unittest.TestCase):
    def setUp(self):
        self.node = gemini.FL_GeminiTextAPI()
        self.client = mock.MagicMock()
        self.client.__enter__.return_value = self.client
        self.client.interactions.create.return_value = SimpleNamespace(output_text=" A caption ")
        self.client.files.upload.side_effect = lambda **kw: SimpleNamespace(
            name=Path(kw["file"]).name, uri="https://media/" + Path(kw["file"]).name,
            state=SimpleNamespace(name="ACTIVE"))
        self.factory = mock.patch.object(gemini.genai, "Client", return_value=self.client)
        self.factory.start()
        self.addCleanup(self.factory.stop)

    def generate(self, **kwargs):
        return self.node.generate_text("Describe", "test-key", "gemini-3.7-flash", 0.2, 8192, 42, **kwargs)

    def test_text_only_contract(self):
        self.assertEqual(self.generate(), ("A caption",))
        self.assertEqual(self.client.interactions.create.call_args.kwargs["input"], "Describe")
        self.client.files.upload.assert_not_called()
        self.client.__exit__.assert_called_once()

    def test_mixed_media_and_batches_preserve_content_and_clean_up(self):
        audio = {"waveform": torch.tensor([[[0.1, -0.2], [0.3, -0.4]], [[0.2, 0.4], [0.1, 0.3]]]), "sample_rate": 48000}
        images = torch.zeros(2, 8, 12, 3)
        images[1, :, :, 0] = 1
        video = mock.Mock()
        paths = []
        upload = self.client.files.upload.side_effect

        def inspect_upload(**kwargs):
            path = Path(kwargs["file"])
            paths.append(path)
            if path.suffix == ".flac":
                samples, rate = sf.read(path, always_2d=True)
                index = int(path.stem.split("-")[1]) - 1
                np.testing.assert_allclose(samples, audio["waveform"][index].T.numpy(), atol=1e-6)
                self.assertEqual(rate, 48000)
            if path.name == "image-2.png":
                self.assertEqual(Image.open(path).getpixel((0, 0)), (255, 0, 0))
            return upload(**kwargs)

        self.client.files.upload.side_effect = inspect_upload
        self.assertEqual(self.generate(audio=audio, image=images, video=video), ("A caption",))
        inputs = self.client.interactions.create.call_args.kwargs["input"]
        self.assertEqual([part["type"] for part in inputs], ["text", "audio", "audio", "image", "image", "video"])
        self.assertEqual(self.client.files.delete.call_count, 5)
        self.assertFalse(any(path.exists() for path in paths))
        video.save_to.assert_called_once()

    def test_failed_generation_still_deletes_uploads(self):
        self.client.interactions.create.side_effect = RuntimeError("Request failed")
        self.assertEqual(self.generate(image=torch.zeros(1, 8, 8, 3)), ("Error: Request failed",))
        self.client.files.delete.assert_called_once_with(name="image-1.png")
        self.client.__exit__.assert_called_once()

    def test_processing_failure_deletes_file_without_generation(self):
        self.client.files.upload.side_effect = None
        self.client.files.upload.return_value = SimpleNamespace(name="file", state=SimpleNamespace(name="PROCESSING"))
        self.client.files.get.return_value = SimpleNamespace(name="file", state=SimpleNamespace(name="FAILED"))
        with mock.patch.object(gemini.time, "sleep"):
            self.assertIn("could not process", self.generate(image=torch.zeros(1, 8, 8, 3))[0])
        self.client.interactions.create.assert_not_called()
        self.client.files.delete.assert_called_once_with(name="file")

    def test_cancellation_is_not_returned_as_caption(self):
        self.client.interactions.create.side_effect = gemini.mm.InterruptProcessingException()
        with self.assertRaises(gemini.mm.InterruptProcessingException):
            self.generate(image=torch.zeros(1, 8, 8, 3))
        self.client.files.delete.assert_called_once()

    def test_processing_timeout_cleans_up(self):
        self.client.files.upload.side_effect = None
        self.client.files.upload.return_value = SimpleNamespace(name="file", state=SimpleNamespace(name="PROCESSING"))
        with mock.patch.object(gemini.time, "monotonic", side_effect=[0, 301]):
            self.assertIn("timed out", self.generate(image=torch.zeros(1, 8, 8, 3))[0])
        self.client.interactions.create.assert_not_called()
        self.client.files.delete.assert_called_once_with(name="file")

    def test_invalid_audio_never_uploads(self):
        self.assertIn("non-finite", self.generate(audio={"waveform": torch.full((1, 1, 8), float("nan")), "sample_rate": 48000})[0])
        self.client.files.upload.assert_not_called()

    def test_native_video_encoding_preserves_sound(self):
        samples = torch.sin(torch.arange(48000) * (2 * np.pi * 440 / 48000))[None, None] * 0.1
        video = InputImpl.VideoFromComponents(Types.VideoComponents(
            images=torch.zeros(24, 32, 32, 3), frame_rate=Fraction(24),
            audio={"waveform": samples, "sample_rate": 48000}))
        with tempfile.TemporaryDirectory() as temporary:
            files = self.node._media_files(Path(temporary), None, None, video)
            with av.open(str(files[0][1])) as container:
                self.assertEqual(len(container.streams.video), 1)
                self.assertEqual(len(container.streams.audio), 1)
                self.assertEqual(len(list(container.decode(video=0))), 24)
            trimmed = InputImpl.VideoFromFile(str(files[0][1])).as_trimmed(start_time=0.25, duration=0.5, strict_duration=True)
            trim_folder = Path(temporary) / "trim"
            trim_folder.mkdir()
            trimmed_files = self.node._media_files(trim_folder, None, None, trimmed)
            with av.open(str(trimmed_files[0][1])) as container:
                self.assertEqual(len(container.streams.audio), 1)
                self.assertEqual(len(list(container.decode(video=0))), 12)


if __name__ == "__main__":
    unittest.main()
