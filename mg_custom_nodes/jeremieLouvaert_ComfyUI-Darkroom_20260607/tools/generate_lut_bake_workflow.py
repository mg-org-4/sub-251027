"""
Generate the LUT Bake example workflow JSON in ComfyUI's modern UI format.
Run this from the ComfyUI-Darkroom root. Output goes to workflows/lut_bake_and_apply.json.
"""

import json
import os
import uuid

DARKROOM_CNR = "comfyui-darkroom"
DARKROOM_VER = "47a8756932f6b622b7ff3da997e6fcb7daa1bf92"
DARKROOM_AUX = "jeremieLouvaert/ComfyUI-Darkroom"
CORE_CNR = "comfy-core"
CORE_VER = "0.18.1"


def core_props(name):
    return {"cnr_id": CORE_CNR, "ver": CORE_VER, "Node name for S&R": name}


def dr_props(name):
    return {
        "cnr_id": DARKROOM_CNR,
        "ver": DARKROOM_VER,
        "Node name for S&R": name,
        "aux_id": DARKROOM_AUX,
    }


def inp(name, typ, widget=False, localized=None):
    d = {"localized_name": localized or name, "name": name, "type": typ}
    if widget:
        d["widget"] = {"name": name}
    d["link"] = None
    return d


def out(name, typ, localized=None, slot_index=None):
    d = {"localized_name": localized or name, "name": name, "type": typ, "links": []}
    if slot_index is not None:
        d["slot_index"] = slot_index
    return d


def build():
    nodes = []
    links = []
    next_link = [0]

    def add_link(from_node, from_slot, to_node, to_slot, typ):
        next_link[0] += 1
        lid = next_link[0]
        links.append([lid, from_node, from_slot, to_node, to_slot, typ])
        return lid

    # 1. LoadImage
    nodes.append({
        "id": 1, "type": "LoadImage",
        "pos": [40, 200], "size": [320, 320], "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            inp("image", "COMBO", widget=True),
            inp("upload", "IMAGEUPLOAD", widget=True),
        ],
        "outputs": [
            out("IMAGE", "IMAGE", slot_index=0),
            out("MASK", "MASK", slot_index=1),
        ],
        "properties": core_props("LoadImage"),
        "widgets_values": ["example.png", "image"],
    })

    # 2. DarkroomLUTIdentity
    nodes.append({
        "id": 2, "type": "DarkroomLUTIdentity",
        "pos": [40, 580], "size": [320, 80], "flags": {}, "order": 1, "mode": 0,
        "inputs": [
            inp("lut_size", "COMBO", widget=True),
        ],
        "outputs": [
            out("identity_lattice", "IMAGE", slot_index=0),
            out("lut_size", "INT", slot_index=1),
        ],
        "properties": dr_props("DarkroomLUTIdentity"),
        "widgets_values": ["33"],
    })

    # 3. DarkroomLUTBakeInject
    nodes.append({
        "id": 3, "type": "DarkroomLUTBakeInject",
        "pos": [420, 260], "size": [320, 120], "flags": {}, "order": 2, "mode": 0,
        "inputs": [
            inp("photo", "IMAGE"),
            inp("identity_lattice", "IMAGE"),
            inp("lut_size", "INT", widget=True),
        ],
        "outputs": [
            out("batched_image", "IMAGE", slot_index=0),
            out("bake_meta", "LUT_BAKE_META", slot_index=1),
        ],
        "properties": dr_props("DarkroomLUTBakeInject"),
        "widgets_values": [33],
    })

    # 4. DarkroomToneCurve
    nodes.append({
        "id": 4, "type": "DarkroomToneCurve",
        "pos": [800, 200], "size": [320, 420], "flags": {}, "order": 3, "mode": 0,
        "inputs": [
            inp("image", "IMAGE"),
            inp("preset", "COMBO", widget=True),
            inp("shadows", "FLOAT", widget=True),
            inp("darks", "FLOAT", widget=True),
            inp("midtones", "FLOAT", widget=True),
            inp("lights", "FLOAT", widget=True),
            inp("highlights", "FLOAT", widget=True),
            inp("red_shadows", "FLOAT", widget=True),
            inp("red_highlights", "FLOAT", widget=True),
            inp("green_shadows", "FLOAT", widget=True),
            inp("green_highlights", "FLOAT", widget=True),
            inp("blue_shadows", "FLOAT", widget=True),
            inp("blue_highlights", "FLOAT", widget=True),
            inp("strength", "FLOAT", widget=True),
        ],
        "outputs": [
            out("image", "IMAGE", slot_index=0),
        ],
        "properties": dr_props("DarkroomToneCurve"),
        "widgets_values": [
            "S-Curve \u2014 Medium",
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            1.0,
        ],
    })

    # 5. DarkroomLUTBakeExtract
    nodes.append({
        "id": 5, "type": "DarkroomLUTBakeExtract",
        "pos": [1160, 260], "size": [320, 100], "flags": {}, "order": 4, "mode": 0,
        "inputs": [
            inp("batched_image", "IMAGE"),
            inp("bake_meta", "LUT_BAKE_META"),
        ],
        "outputs": [
            out("graded_photo", "IMAGE", slot_index=0),
            out("graded_lattice", "IMAGE", slot_index=1),
            out("lut_size", "INT", slot_index=2),
        ],
        "properties": dr_props("DarkroomLUTBakeExtract"),
        "widgets_values": [],
    })

    # 6. DarkroomLUTExport
    nodes.append({
        "id": 6, "type": "DarkroomLUTExport",
        "pos": [1520, 420], "size": [360, 180], "flags": {}, "order": 5, "mode": 0,
        "inputs": [
            inp("processed_lattice", "IMAGE"),
            inp("lut_size", "INT", widget=True),
            inp("filename", "STRING", widget=True),
            inp("title", "STRING", widget=True),
            inp("output_directory", "STRING", widget=True),
        ],
        "outputs": [
            out("file_path", "STRING", slot_index=0),
        ],
        "properties": dr_props("DarkroomLUTExport"),
        "widgets_values": [33, "darkroom_grade", "Darkroom Grade", ""],
    })

    # 7. PreviewImage (graded photo)
    nodes.append({
        "id": 7, "type": "PreviewImage",
        "pos": [1520, 120], "size": [360, 260], "flags": {}, "order": 6, "mode": 0,
        "inputs": [
            inp("images", "IMAGE"),
        ],
        "outputs": [],
        "properties": core_props("PreviewImage"),
        "widgets_values": [],
    })

    # 8. Note
    note_text = (
        "DARKROOM LUT BAKE - single-chain workflow\n\n"
        "One grading chain processes BOTH your photo and the identity lattice at the same time "
        "(as a 2-image batch). After the chain, LUT Bake Extract splits them back out: the graded "
        "photo goes to Preview, the graded lattice goes to LUT Export which bakes a .cube file to "
        "output/luts/.\n\n"
        "Replace the Tone Curve in the middle with any color-only Darkroom nodes (Film Stock, LGG, "
        "HSL Selective, Hue vs X, Color Warper, ACES Tonemap...).\n\n"
        "DO NOT put spatial effects in this chain (Film Grain, Halation, Vignette, Sharpening, Noise "
        "Reduction, Chromatic Aberration, Lens Distortion) - they'll corrupt the lattice. Apply those "
        "to the graded_photo output AFTER Extract."
    )
    nodes.append({
        "id": 8, "type": "Note",
        "pos": [40, 20], "size": [1840, 140], "flags": {}, "order": 7, "mode": 0,
        "inputs": [],
        "outputs": [],
        "properties": {},
        "widgets_values": [note_text],
    })

    # --- Wire links ---
    # LoadImage.IMAGE -> Inject.photo
    l = add_link(1, 0, 3, 0, "IMAGE")
    nodes[0]["outputs"][0]["links"].append(l)
    nodes[2]["inputs"][0]["link"] = l

    # Identity.identity_lattice -> Inject.identity_lattice
    l = add_link(2, 0, 3, 1, "IMAGE")
    nodes[1]["outputs"][0]["links"].append(l)
    nodes[2]["inputs"][1]["link"] = l

    # Identity.lut_size -> Inject.lut_size (widget input)
    l = add_link(2, 1, 3, 2, "INT")
    nodes[1]["outputs"][1]["links"].append(l)
    nodes[2]["inputs"][2]["link"] = l

    # Inject.batched_image -> ToneCurve.image
    l = add_link(3, 0, 4, 0, "IMAGE")
    nodes[2]["outputs"][0]["links"].append(l)
    nodes[3]["inputs"][0]["link"] = l

    # Inject.bake_meta -> Extract.bake_meta
    l = add_link(3, 1, 5, 1, "LUT_BAKE_META")
    nodes[2]["outputs"][1]["links"].append(l)
    nodes[4]["inputs"][1]["link"] = l

    # ToneCurve.image -> Extract.batched_image
    l = add_link(4, 0, 5, 0, "IMAGE")
    nodes[3]["outputs"][0]["links"].append(l)
    nodes[4]["inputs"][0]["link"] = l

    # Extract.graded_photo -> PreviewImage.images
    l = add_link(5, 0, 7, 0, "IMAGE")
    nodes[4]["outputs"][0]["links"].append(l)
    nodes[6]["inputs"][0]["link"] = l

    # Extract.graded_lattice -> Export.processed_lattice
    l = add_link(5, 1, 6, 0, "IMAGE")
    nodes[4]["outputs"][1]["links"].append(l)
    nodes[5]["inputs"][0]["link"] = l

    # Extract.lut_size -> Export.lut_size (widget input)
    l = add_link(5, 2, 6, 1, "INT")
    nodes[4]["outputs"][2]["links"].append(l)
    nodes[5]["inputs"][1]["link"] = l

    return {
        "id": str(uuid.uuid4()),
        "revision": 0,
        "last_node_id": max(n["id"] for n in nodes),
        "last_link_id": next_link[0],
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "..", "workflows", "lut_bake_and_apply.json")
    out_path = os.path.normpath(out_path)
    workflow = build()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    print(f"Nodes: {len(workflow['nodes'])}, Links: {len(workflow['links'])}")
