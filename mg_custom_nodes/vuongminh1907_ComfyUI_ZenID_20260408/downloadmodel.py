from huggingface_hub import hf_hub_download, snapshot_download
import shutil
import os


os.makedirs("../models/instantid", exist_ok=True)

hf_hub_download(
    repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="../models/instantid"
)

hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/config.json",
    local_dir="../models/controlnet",
)
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
    local_dir="../models/controlnet",
)

os.makedirs("../models/insightface/", exist_ok=True)

os.makedirs("../models/insightface/", exist_ok=True)

snapshot_download(
    repo_id="vuongminhkhoi4/antelopev2",
    cache_dir ="models",
    repo_type ="model",
    local_dir="../models/insightface",
)
if os.path.exists("./models") and os.path.isdir("./models"):
    shutil.rmtree("./models")
