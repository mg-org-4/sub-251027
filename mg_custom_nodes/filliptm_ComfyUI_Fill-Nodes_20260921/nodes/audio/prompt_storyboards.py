import json
import sqlite3
import uuid

from PIL import Image

from .prompt_references import reference_path
from .prompt_writer_config import DATA_DIR


MODEL = "Nano Banana 2 (Gemini 3.1 Flash Image)"


def validate_storyboard(value):
    if not isinstance(value, dict):
        raise ValueError("Storyboard must be an object.")
    result = {}
    for key in ("scheduler_id", "section_id", "revision", "prompt"):
        text = value.get(key)
        if not isinstance(text, str) or not text.strip() or len(text) > (64000 if key == "prompt" else 128):
            raise ValueError(f"Invalid storyboard {key}.")
        result[key] = text
    request_key = value.get("request_key")
    continuity = value.get("continuity", "")
    if not isinstance(continuity, str) or len(continuity) > 32000:
        raise ValueError("Storyboard continuity brief must be at most 32000 characters.")
    result["continuity"] = continuity
    if request_key is not None:
        if not isinstance(request_key, str) or not 1 <= len(request_key) <= 256:
            raise ValueError("Invalid storyboard request key.")
        result["request_key"] = request_key
    grid = value.get("grid", 2)
    if type(grid) is not int or grid not in (2, 3):
        raise ValueError("Choose a 2×2 or 3×3 storyboard.")
    resolution = value.get("resolution", "2K")
    aspect = value.get("aspect_ratio", "16:9")
    if resolution not in ("1K", "2K", "4K") or aspect not in ("16:9", "9:16", "1:1", "4:3", "3:4"):
        raise ValueError("Unsupported storyboard resolution or aspect ratio.")
    moodboards = value.get("moodboards", [])
    if not isinstance(moodboards, list) or len(moodboards) > 4:
        raise ValueError("Select at most four moodboards.")
    selected = []
    for image in moodboards:
        if not isinstance(image, dict) or not isinstance(image.get("role", ""), str) or len(image.get("role", "")) > 500:
            raise ValueError("Invalid moodboard role.")
        reference_path(image)
        if image.get("type", "input") != "input":
            raise ValueError("Moodboards must be uploaded to ComfyUI input.")
        selected.append({"filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": "input", "role": image.get("role", "")})
    return {**result, "grid": grid, "resolution": resolution, "aspect_ratio": aspect, "moodboards": selected}


def storyboard_graph(job_id, spec):
    inputs = {
        "prompt": f"Create one {spec['grid']}x{spec['grid']} storyboard contact sheet. Equal rectangular cells, no gutters, labels, captions, text or timestamps. Read left to right, top to bottom: each cell progresses chronologically through this section. Keep character identity and style consistent.\n{spec['prompt']}",
        "model": MODEL,
        "model.aspect_ratio": spec["aspect_ratio"],
        "model.resolution": spec["resolution"],
        "model.thinking_level": "MINIMAL",
        "seed": int(uuid.UUID(job_id)) % (2**32),
        "response_modalities": "IMAGE",
    }
    graph = {}
    if spec.get("continuity"):
        inputs["prompt"] += "\nShared production continuity (context only; depict only the requested section):\n" + spec["continuity"]
    inputs["prompt"] += "\nPreserve the same cast identity, face, hair, costume, palette, drawing technique and world design across sections unless the section explicitly requests a change. Treat supplied character/style reference images as visual anchors, not optional inspiration."
    for index, image in enumerate(spec["moodboards"], 1):
        node_id = f"moodboard_{index}"
        name = "/".join(part for part in (image["subfolder"], image["filename"]) if part)
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": name + " [input]"}}
        inputs[f"model.images.image_{index}"] = [node_id, 0]
        if image.get("role"):
            inputs["prompt"] += f"\nReference image {index} role: {image['role']}"
    graph["storyboard"] = {"class_type": "GeminiNanoBanana2V2", "inputs": inputs}
    graph["save_storyboard"] = {"class_type": "SaveImage", "inputs": {
        "images": ["storyboard", 0], "filename_prefix": f"fl-storyboards/{job_id}/sheet"}}
    return graph


def storyboard_batch_graph(jobs):
    graph = {}
    for job in jobs:
        branch = storyboard_graph(job["id"], job["spec"])
        prefix = job["id"] + ":"
        for node_id, node in branch.items():
            graph[prefix + node_id] = {**node, "inputs": {key: [prefix + value[0], value[1]]
                if isinstance(value, list) and len(value) == 2 and value[0] in branch else value
                for key,value in node["inputs"].items()}}
    return graph


class StoryboardStore:
    def __init__(self, path=None):
        self.path = path or DATA_DIR / "storyboards.db"

    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, scheduler TEXT, state TEXT, spec TEXT, result TEXT)")
        return connection

    def create(self, value):
        spec = validate_storyboard(value)
        job_id = str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps([spec["scheduler_id"], spec["request_key"]]))) if "request_key" in spec else str(uuid.uuid4())
        connection = self.connect()
        try:
            with connection:
                connection.execute("INSERT OR IGNORE INTO jobs VALUES (?, ?, 'proposed', ?, '{}')", (job_id, spec["scheduler_id"], json.dumps(spec)))
        finally:
            connection.close()
        return self.get(job_id)

    def get(self, job_id):
        connection = self.connect()
        try:
            row = connection.execute("SELECT id, state, spec, result FROM jobs WHERE id=?", (job_id,)).fetchone()
        finally:
            connection.close()
        if row is None:
            raise ValueError("Storyboard job not found.")
        return {"id": row[0], "state": row[1], "spec": json.loads(row[2]), "result": json.loads(row[3])}

    def list(self, scheduler):
        connection = self.connect()
        try:
            ids = connection.execute("SELECT id FROM jobs WHERE scheduler=? ORDER BY rowid DESC LIMIT 100", (scheduler,)).fetchall()
        finally:
            connection.close()
        return [self.get(row[0]) for row in ids]

    def claim(self, job_id):
        job = self.get(job_id)
        validate_storyboard(job["spec"])
        connection = self.connect()
        try:
            with connection:
                changed = connection.execute("UPDATE jobs SET state='submitted' WHERE id=? AND state='proposed'", (job_id,)).rowcount
        finally:
            connection.close()
        if not changed:
            raise ValueError("This job was already submitted or cancelled. Refresh its status; do not resubmit.")
        return storyboard_graph(job_id, job["spec"])

    def claim_batch(self, job_ids):
        connection = self.connect()
        try:
            with connection:
                for job_id in job_ids:
                    changed = connection.execute("UPDATE jobs SET state='submitted' WHERE id=? AND state='proposed'", (job_id,)).rowcount
                    if not changed:
                        raise ValueError("A storyboard was already submitted or cancelled. Refresh before trying again.")
        finally:
            connection.close()

    def update(self, job_id, state, result):
        connection = self.connect()
        try:
            with connection:
                connection.execute("UPDATE jobs SET state=?, result=? WHERE id=?", (state, json.dumps(result), job_id))
        finally:
            connection.close()
        return self.get(job_id)


def extract_panels(job, source, bounds=None):
    source_path = reference_path(source)
    expected = f"fl-storyboards/{job['id']}"
    if source.get("type") != "output" or source.get("subfolder", "").replace("\\", "/") != expected:
        raise ValueError("The image is not an output of this storyboard job.")
    grid = job["spec"]["grid"]
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        if bounds is None:
            bounds = [0, 0, width, height]
        if not isinstance(bounds, list) or len(bounds) != 4 or any(type(value) is not int for value in bounds):
            raise ValueError("Crop bounds must be four integer pixel coordinates.")
        left, top, right, bottom = bounds
        if not (0 <= left < right <= width and 0 <= top < bottom <= height) or min(right-left, bottom-top) < grid:
            raise ValueError("Crop bounds must be inside the contact sheet.")
        version = uuid.uuid4().hex
        assets = {}
        for index in range(grid * grid):
            row, column = divmod(index, grid)
            box = (left + (right-left)*column//grid, top + (bottom-top)*row//grid,
                   left + (right-left)*(column+1)//grid, top + (bottom-top)*(row+1)//grid)
            panel = image.crop(box)
            filename = f"panel-{version}-{index+1}.png"
            panel.save(source_path.parent / filename)
            assets[f"{version}-{index+1}"] = {
                "kind": "image", "filename": filename, "subfolder": expected, "type": "output",
                "label": f"Storyboard panel {index+1}", "width": panel.width, "height": panel.height,
                "storyboard_id": job["id"], "version": version, "panel": index+1,
                "source": source,
                "generation": {"model": MODEL, "grid": grid, "resolution": job["spec"]["resolution"],
                               "aspect_ratio": job["spec"]["aspect_ratio"], "section_id": job["spec"]["section_id"], "revision": job["spec"]["revision"]},
                "section_moment": index / max(1, grid*grid-1),
            }
    return {"source": source, "assets": assets, "bounds": bounds,
            "versions": [*job["result"].get("versions", []), {"version": version, "assets": assets, "bounds": bounds}],
            "warning": "Panels below 512 pixels may lose detail; consider a larger contact sheet." if min(right-left, bottom-top)//grid < 512 else ""}


storyboard_store = StoryboardStore()
