import importlib.util
import sys
import types
from pathlib import Path


folder_paths = types.ModuleType("folder_paths")
folder_paths.get_output_directory = lambda: "/tmp/comfy-output"
folder_paths.get_temp_directory = lambda: "/tmp/comfy-temp"
sys.modules.setdefault("folder_paths", folder_paths)

HELPER_PATH = Path(__file__).parents[1] / "nodes" / "helper_logging.py"
helper_spec = importlib.util.spec_from_file_location("helper_logging", HELPER_PATH)
assert helper_spec is not None and helper_spec.loader is not None
helper_logging = importlib.util.module_from_spec(helper_spec)
sys.modules["helper_logging"] = helper_logging
helper_spec.loader.exec_module(helper_logging)
MODULE_PATH = Path(__file__).parents[1] / "nodes" / "nodes_metadata.py"
spec = importlib.util.spec_from_file_location("nodes_metadata", MODULE_PATH)
assert spec is not None and spec.loader is not None
metadata_nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata_nodes)
sys.modules.pop("folder_paths", None)


def test_metadata_image_saver_skips_cleanly_without_images(capsys):
    saver = metadata_nodes.DaSiWa_MetadataImageSaver()
    input_types = saver.INPUT_TYPES()

    assert "images" in input_types["optional"]
    assert "images" not in input_types["required"]

    result = saver.save_images(
        filename_prefix="DaSiWa_test",
        file_format="webp",
        compression=0,
        save_output=True,
    )

    assert result == {"ui": {"images": []}, "result": ("", "")}
    assert "\033[38;5;136m[DaSiWa Metadata Image Saver]\033[0m no images received; skipping save." in capsys.readouterr().out
