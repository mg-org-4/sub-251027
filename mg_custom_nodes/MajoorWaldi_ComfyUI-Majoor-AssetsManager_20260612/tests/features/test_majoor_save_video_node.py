import importlib.util
import sys

from tests.repo_root import REPO_ROOT

COMFY_ROOT = REPO_ROOT.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.append(str(COMFY_ROOT))

spec = importlib.util.spec_from_file_location("majoor_assetsmanager_nodes", REPO_ROOT / "nodes.py")
assert spec is not None and spec.loader is not None
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


def test_build_video_ui_uses_comfyui_preview_video_contract():
    result = nodes._build_video_ui("clip_00001_.mp4", "runs", "output", "clip_00001.png")

    ui = result["ui"]
    assert ui["images"] == [{"filename": "clip_00001_.mp4", "subfolder": "runs", "type": "output"}]
    assert ui["animated"] == (True,)
    assert ui["videos"] == ui["images"]
    assert ui["preview_images"] == [{"filename": "clip_00001.png", "subfolder": "runs", "type": "output"}]
